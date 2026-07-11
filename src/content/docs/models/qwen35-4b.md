---
title: Qwen3.5-4B / 9B / 27B
description: "Running the Qwen3.5 family on openinfer: build with the qwen35-4b feature, launch, serving performance, and hybrid-attention architecture notes."
---

Qwen3.5 is a hybrid-attention model line: 3 of every 4 layers use linear
attention (gated delta rule), and only every 4th layer is full attention.
openinfer serves the family (4B / 9B / 27B, text-only) behind the
`qwen35-4b` cargo feature with CUDA Graph decode and paged KV cache for
the full-attention layers.

## Build

Qwen3.5 is the only openinfer model line that needs Python at **build
time**: its linear-attention prefill kernels are Triton AOT-generated.
There is no Python at runtime — the compiled kernels link into the same
single Rust binary.

```bash
# One-time: a Python environment with Triton for the AOT step
uv venv && uv pip install triton

# openinfer picks up .venv/bin/python automatically, or point at one:
export OPENINFER_TRITON_PYTHON=.venv/bin/python
```

## Launch

From the openinfer workspace root:

```bash
huggingface-cli download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B

export CUDA_HOME=/usr/local/cuda
cargo run --release --features qwen35-4b -- --model-path models/Qwen3.5-4B
```

The server exposes an OpenAI-compatible `/v1/completions` endpoint:

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "models/Qwen3.5-4B", "prompt": "The capital of France is", "max_tokens": 32}'
```

Streaming:

```bash
curl -N http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "models/Qwen3.5-4B", "prompt": "Write a haiku about Rust:", "max_tokens": 64, "stream": true}'
```

The `model` field must match the served model id — by default the
`--model-path` value, or whatever `--served-model-name` sets
(`curl http://localhost:8000/v1/models` shows it).

### Qwen3.5-9B and 27B

The same feature flag serves the larger Qwen3.5 sizes — point
`--model-path` at the weights and the size is detected from the model
config:

```bash
cargo run --release --features qwen35-4b -- --model-path models/Qwen3.5-9B
```

All three sizes are gated by the same HF bf16 logits golden tests
(short prompts plus 4097/8192-token long prompts per size).

## Performance

Measured on **1x RTX 5070 Ti 16GB**, driver 610.43.02, CUDA 13.3 build,
Qwen3.5-4B BF16 weights, TP1, CUDA Graph decode on, openinfer main
`baaffd0`. `vllm-bench` client on localhost, random dataset, 1024-token
prompts, 128-token outputs, greedy, seed 42 — reproducible via
`tools/bench/run_serving_bench.sh` in the repo.

### Serving Load

Poisson arrivals (`QPS n`) and fixed in-flight concurrency (`c=N`):

| load | req/s | out tok/s | TTFT p50 / p99 | TPOT p50 / p99 |
| ---: | ---: | ---: | ---: | ---: |
| c=1 | 0.60 | 77 | 106 / 113 ms | 12.2 / 12.2 ms |
| QPS 1 | 0.97 | 125 | 124 / 236 ms | 15.6 / 20.2 ms |
| QPS 2 | 1.94 | 249 | 129 / 377 ms | 20.1 / 29.6 ms |
| c=4 | 1.80 | 230 | 235 / 346 ms | 15.7 / 16.5 ms |

Single-stream decode (`c=1`) runs at 12.2 ms/token — about 82 tok/s per
request. All rows completed every request at the full 128-token output.

The table stops at 4 in-flight requests: beyond that, concurrent
1024-token prefills exceed the VRAM this 16 GB card has left after
weights, and the affected requests fail with an allocation error — the
Qwen3.5 scheduler does not yet bound concurrent prefill workspace the
way it bounds KV. Higher-concurrency serving at this prompt shape needs
a larger-VRAM GPU.

## Notes

- Only the 8 full-attention layers keep a paged KV cache; the 24
  linear-attention layers carry a fixed-size per-request state, so KV
  memory grows with context length at 1/4 the rate of a full-attention
  stack.
- CUDA Graph decode is on by default; disable with `--cuda-graph=false`
  for debugging. Greedy and sampled decoding are supported; prefix
  caching is not yet wired up for the hybrid KV/recurrent state.
