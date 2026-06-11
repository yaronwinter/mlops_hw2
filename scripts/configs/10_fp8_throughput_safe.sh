#!/usr/bin/env bash
#
# Run 10 - STACKED best-of: throughput, quality-safe.
#
# Combines the two "more concurrency headroom" levers that DON'T touch quality:
#   Run 2 (--max-num-seqs 96)  +  Run 5 (--max-model-len 4096).
# Weights are FP8; KV cache stays BF16, so accuracy should track Run 1.
#
# WHEN TO RUN: only after Runs 1/2/5, and only if BOTH the seqs bump and the
# tighter context window individually helped (queue was draining slowly AND KV
# was the limiter). If only one helped, prefer that single-lever config - don't
# stack a lever that did nothing.
#
# COMPARE AGAINST: Run 1 (FP8 base), or whichever single-lever run was your best.
# HYPOTHESIS: the two headroom gains compound -> higher sustainable RPS at
# P95<5s than either alone. WATCH: TTFT contention from the bigger batch, and
# http_errors from any prompt that overflows the 4096 window (see Run 5's note).

set -euo pipefail

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 96 \
    --quantization fp8
