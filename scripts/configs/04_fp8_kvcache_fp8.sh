#!/usr/bin/env bash
#
# Run 4 - FP8 weights + FP8 KV cache.
#
# DIFF vs Run 1 (01_fp8.sh): + --kv-cache-dtype fp8. One lever.
#
# HYPOTHESIS: if you are STILL KV-bound after Run 1 (KV cache near 100% at the
# concurrency you need), halving KV bytes/token roughly doubles cache capacity,
# admitting more concurrent sequences without raising max-model-len.
# QUALITY GUARD: FP8 KV is the most accuracy-sensitive lever here - RE-RUN THE
# EVAL and only keep it if the pass rate holds. If you're not KV-bound, skip this
# run; it adds risk for no latency gain.

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
    --quantization fp8 \
    --kv-cache-dtype fp8
