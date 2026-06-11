#!/usr/bin/env bash
#
# Run 5 - FP8 + tighter context window.
#
# DIFF vs Run 1 (01_fp8.sh): --max-model-len 8192 -> 4096. One lever.
#
# HYPOTHESIS: the real need is ~3K prompt + <=512 output, so 8192 over-reserves.
# A tighter cap means each sequence's KV footprint ceiling is smaller, packing
# more requests into cache and reducing fragmentation -> more concurrency / lower
# queueing. WATCH FOR TRUNCATION: if any prompt+output exceeds 4096 the request
# errors; check the driver's http_error count and the longest prompt in
# perf_pool.jsonl before trusting this run. If errors appear, 4096 is too tight.

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
    --max-num-seqs 48 \
    --quantization fp8
