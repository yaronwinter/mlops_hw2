#!/usr/bin/env bash
#
# Run 3 - FP8 + smaller prefill chunks (protect the decode/TTFT tail).
#
# DIFF vs Run 1 (01_fp8.sh): --max-num-batched-tokens 8192 -> 4096. One lever.
#
# HYPOTHESIS: if Run 1 showed high TTFT but low decode time (prefill-bound), big
# 3K-token prefills are blocking in-flight decodes within a step. Smaller chunks
# interleave prefill with decode more finely, smoothing the P95 tail under
# concurrent load. WATCH: too small starves prefill throughput - if TTFT rises
# instead of falling, this workload wants larger chunks, not smaller.

set -euo pipefail

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 48 \
    --quantization fp8
