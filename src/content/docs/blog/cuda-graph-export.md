---
title: "See Qwen3 Decode as a CUDA Graph"
description: "Export OpenInfer's live Qwen3 decode CUDA Graph as a detailed DOT for LLMs and a folded high-resolution PNG for people."
---

OpenInfer can now [export the live rank-0, batch-1 SplitKv decode CUDA Graph](https://github.com/openinfer-project/openinfer/pull/640) at startup.
A single flag writes an unfolded `.dot` sidecar with full kernel symbols and launch metadata for LLMs or scripts, plus a folded 192-DPI `.png` for people.

```bash
cargo run --release -- \
  --model-path models/Qwen3-4B \
  --dump-graph-png qwen3-decode.png
```

The Qwen3-4B example below contains 507 kernels and 506 dependency edges, with its 14-kernel layer body collapsed as a ×36 repeated block.

![Qwen3-4B batch-1 decode CUDA Graph with 36 repeated layers folded into one block](/blog/cuda-graph-export/qwen3_4b_decode.png)

*Qwen3-4B decode · batch 1 · SplitKv · 507 kernels / 506 edges*
