#!/usr/bin/env bash
#
# Run 1 - FP8 (on-the-fly quantization). THE key lever for this 80GB card.
#
# DIFF vs baseline (scripts/start_vllm.sh): + --quantization fp8. Nothing else.
# vLLM quantizes the same BF16 weights at load time -> no second download,
# served model id unchanged (so .env stays put).
#
# HYPOTHESIS: FP8 weights (~30GB vs ~60GB) free ~30GB for KV cache (~4x more
# tokens). Expect KV-cache pressure / preemptions to drop and sustainable RPS
# at P95<5s to rise. QUALITY GUARD: re-run the eval - quantization can regress
# accuracy; confirm pass rate holds.
#
# This is the expected working base for Runs 2-5.

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
    --max-num-batched-tokens 8192 \
    --max-num-seqs 48 \
    --quantization fp8
