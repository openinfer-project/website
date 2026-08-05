---
title: Getting Started
description: Build pegainfer from source and serve your first model.
---

pegainfer builds from source with Cargo. Python is needed once at build time
for Triton AOT kernel compilation — the running server has no Python
dependency.

## Prerequisites

- Rust (2024 edition)
- CUDA Toolkit (nvcc, cuBLAS) and a CUDA-capable GPU
- NVIDIA driver R535 (CUDA 12.2) or newer
- Python 3 + Triton (build-time only)

## Build & run

```bash
git clone https://github.com/pegainfer-project/pegainfer
cd pegainfer

# One-time Python setup for Triton AOT kernel compilation
uv venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# Download a model
huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B

# Build & start the server on port 8000
export CUDA_HOME=/usr/local/cuda
export PEGAINFER_TRITON_PYTHON=.venv/bin/python
cargo run --release -- --model-path models/Qwen3-4B
```

Always build with `--release` — debug builds of the CUDA paths are far too
slow to be usable.

## Send a request

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
  -d '{"prompt": "The capital of France is", "max_tokens": 32, "stream": true}'
```

Any OpenAI SDK works the same way — set the base URL to
`http://localhost:8000/v1`.

## Next steps

Pick a model from the sidebar for model-specific launch flags, performance
numbers, and architecture notes. [Qwen3-4B](/models/qwen3-4b/) is the most
mature line and the best place to start.
