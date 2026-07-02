---
title: Qwen3-4B
description: "Running Qwen3-4B on openinfer: launch, serving performance, and architecture notes."
---

Qwen3-4B is the default openinfer model line: pure Rust + CUDA, no Python at
build time or runtime, full-attention GQA, paged KV cache, prefix caching,
CUDA Graph decode, optional pegaflow KV offload, and DSpark speculative
decoding.

## Launch

From the openinfer workspace root:

```bash
huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B

export CUDA_HOME=/usr/local/cuda
cargo run --release
```

The default model path is `models/Qwen3-4B`, and `openinfer-server` is the
workspace default member. To pass an explicit model path or port:

```bash
cargo run --release -p openinfer-server -- \
  --model-path models/Qwen3-4B \
  --port 8000
```

The server exposes an OpenAI-compatible `/v1/completions` endpoint:

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 32}'
```

Streaming:

```bash
curl -N http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a haiku about Rust:", "max_tokens": 64, "stream": true}'
```

Useful Qwen3 flags:

```bash
# Disable CUDA Graph for debugging
cargo run --release -- --cuda-graph=false

# Pure host-tier KV offload benchmark mode
cargo run --release -- \
  --kv-offload \
  --kv-offload-host-gib 16 \
  --no-prefix-cache

# DSpark speculative decoding (greedy, single-GPU)
cargo run --release -- \
  --model-path models/Qwen3-4B \
  --dflash-draft-model-path models/dspark_qwen3_4b_block7
```

### Qwen3-8B

Qwen3-8B uses the same architecture (4096 hidden, 12288 intermediate, 36
layers) and runs on the same single GPU — just point `--model-path` at the
8B weights. No feature flags or build changes needed.

```bash
cargo run --release -- --model-path models/Qwen3-8B
```

## Performance

Measured on **1x RTX 5090 32GB**, driver 590.48.01, CUDA 13.1 build,
Qwen3-4B BF16 weights, TP1. openinfer main `70888b2`, same `vllm bench serve`
client, same host, same GPU, prefix cache on, seed 42, input 1024 / output 128
for the QPS sweep. Reproducible via `tools/bench/run_serving_bench.sh` in the
repo.

![Qwen3-4B RTX 5090 benchmark summary](/models/qwen3-4b/perf.png)

Serving load, warm prefix-cache TTFT, and KV offload numbers are in the
chart above. See the [OpenInfer 0.1.0 release post](/blog/openinfer-010/)
for methodology and discussion.

### Footprint

| Metric | openinfer | vLLM 0.22.1 |
| --- | ---: | ---: |
| RSS before stress, loaded and idle | **771 MB** | 3814 MB |
| RSS after stress | **1064 MB** | 3863 MB |
| Startup to HTTP ready, cold | **2.99 s** | 70.0 s |
| Startup, warm compile cache | **~3.0 s** | 32.7 s |
| GPU memory, default utilization | 28832 MiB | 30290 MiB |

openinfer is a single process; vLLM RSS is summed over its process tree.
The openinfer RSS peak during load is transient while reading safetensors
through `mmap`; steady-state settles at 771 MB after load.

### Serving Load

Poisson arrivals, 1024-token prompts, 128-token outputs, greedy
(`--temperature 0`):

| QPS | output tok/s | TTFT p50 | TPOT p50 |
| ---: | ---: | ---: | ---: |
| 1 | 126.3 | 45.2 ms | 6.53 ms |
| 2 | 252.3 | 30.3 ms | 6.93 ms |
| 4 | 504.1 | 48.8 ms | 8.30 ms |
| 8 | 1007.8 | 51.1 ms | 11.39 ms |
| 10 | 1258.3 | 53.4 ms | 13.55 ms |
| 12 | 1507.7 | 60.0 ms | 16.75 ms |
| 16 | 1979.9 | 203.8 ms | 46.92 ms |

Low load (QPS 1–4) holds sub-10 ms TPOT. At QPS 8–12 the engine sustains
throughput linearly with gradually rising TPOT. QPS 16 is beyond saturation
(peak concurrency 147) but still completes all requests without errors.

### Qwen3-8B Serving Load

Same harness, Qwen3-8B BF16, single RTX 5090 (32 GB). The 8B model is 2×
the weights of 4B; throughput scales accordingly until the GPU saturates
around QPS 8:

| QPS | output tok/s | TTFT p50 | TPOT p50 |
| ---: | ---: | ---: | ---: |
| 1 | 125.1 | 82.2 ms | 11.55 ms |
| 2 | 249.9 | 54.1 ms | 11.46 ms |
| 4 | 498.6 | 88.1 ms | 16.08 ms |
| 8 | 991.9 | 148.0 ms | 30.97 ms |

### Warm Prefix-Cache TTFT

For multi-turn chat and agent workloads, most of the prompt often lands as
a warm prefix-cache hit. In this sweep, the same prompt group is sent cold
once to populate GPU KV cache, then sent warm:

| Input length | openinfer cold | openinfer warm p50 | openinfer warm p99 | vLLM warm p50 | vLLM warm p99 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 16.2 ms | 8.5 ms | 8.8 ms | 14.5 ms | 19.1 ms |
| 512 | 24.6 ms | 8.6 ms | 8.8 ms | 16.0 ms | 16.4 ms |
| 1024 | 44.0 ms | 9.2 ms | 9.5 ms | 18.4 ms | 19.0 ms |
| 2048 | 92.0 ms | 10.4 ms | 10.8 ms | 23.7 ms | 24.4 ms |
| 4096 | 211.5 ms | 12.7 ms | 13.4 ms | 34.1 ms | 36.2 ms |
| 8192 | 460.0 ms | 21.6 ms | 22.8 ms | 58.6 ms | 59.9 ms |
| 16384 | 1143.9 ms | **26.3 ms** | 27.9 ms | 95.6 ms | 98.2 ms |

openinfer wins warm TTFT at every measured length; the 16k warm-cache path
is 3.6x faster than vLLM p50.

### KV Offload

With `--kv-offload`, sealed Qwen3 KV blocks can be restored from the
pegaflow host tier instead of recomputing full prefill. The pure-L2 mode
below disables cross-request HBM prefix reuse, so every prefix hit is
restored from host DRAM:

```bash
cargo run --release -- \
  --kv-offload \
  --kv-offload-host-gib 16 \
  --no-prefix-cache
```

| Input length | Cold full prefill | L2 warm p50, host restore | Speedup |
| ---: | ---: | ---: | ---: |
| 256 | 25.4 ms | 9.8 ms | 2.6x |
| 512 | 25.6 ms | 11.6 ms | 2.2x |
| 1024 | 45.3 ms | 15.4 ms | 2.9x |
| 2048 | 92.5 ms | 22.9 ms | 4.0x |
| 4096 | 211.1 ms | 37.5 ms | 5.6x |
| 8192 | 461.3 ms | 71.4 ms | 6.5x |
| 16384 | 1140.5 ms | **125.5 ms** | 9.1x |

At 16k, the tiering picture is: HBM hit about 26 ms, host-tier restore
about 126 ms, cold prefill about 1.14 s.

### DSpark Speculative Decoding

[DSpark](https://github.com/deepseek-ai/DeepSpec) (DeepSeek-AI, Jun 2026)
adds a semi-autoregressive Markov head to a DFlash parallel drafter, raising
accepted draft length by conditioning each block position on the previously
sampled token. openinfer supports it behind `--dflash-draft-model-path` —
the drafter checkpoint goes in, the target model serves as-is, and greedy
verify keeps output lossless.

```bash
# Download the released DSpark block7 drafter
huggingface-cli download deepseek-ai/dspark_qwen3_4b_block7 \
  --local-dir models/dspark_qwen3_4b_block7

# Launch with speculative decoding (greedy, single-GPU)
cargo run --release -- \
  --model-path models/Qwen3-4B \
  --dflash-draft-model-path models/dspark_qwen3_4b_block7
```

Single-stream TPOT drops from 6.5 ms to 3.7 ms at c1 (2× decode speedup from
amortizing target forwards over accepted drafts). Concurrency sweep, greedy,
random dataset, 1024-in / 128-out:

| Concurrency | output tok/s | TTFT p50 | TPOT p50 |
| ---: | ---: | ---: | ---: |
| 1 | 266.1 | 44.0 ms | 3.67 ms |
| 4 | 731.1 | 55.2 ms | 5.05 ms |
| 8 | 1026.1 | 57.1 ms | 7.01 ms |

DFlash (the non-Markov predecessor) is also supported via the same flag with
a DFlash-format drafter checkpoint. DSpark is the recommended drafter for
Qwen3-4B.

## Architecture Notes

- Full attention with grouped-query attention: 32 query heads, 8 KV heads,
  head dim 128, 36 layers.
- Qwen3-4B and Qwen3-8B are the default pure Rust + CUDA build, with no
  Python build dependency.
- Paged KV cache uses full-lifetime admission, so requests that cannot fit
  are rejected instead of hanging under memory pressure.
- Prefix cache is on by default; `--no-prefix-cache` disables GPU prefix
  matching, or becomes pure-L2 host restore mode when combined with
  `--kv-offload`.
- CUDA Graph decode uses pre-allocated buffers and can be disabled with
  `--cuda-graph=false` for debugging.
- DSpark/DFlash speculative decoding is single-GPU, greedy-only, and forces
  prefix caching off (the drafter needs clean target hidden states).
