#!/usr/bin/env bash
#
# Run 2 - FP8 + higher concurrency.
#
# DIFF vs Run 1 (01_fp8.sh): --max-num-seqs 48 -> 96. One lever.
#
# HYPOTHESIS: if Run 1's Grafana showed requests WAITING in the queue while the
# GPU was underused, the batch was too small. FP8 freed enough KV to admit more
# concurrent sequences, so raising the cap should lift throughput / achieved RPS.
# WATCH: TTFT must not spike - if per-request latency climbs, you've over-batched
# and contention is hurting the tail; back off toward 64.

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
    --max-num-seqs 96 \
    --quantization fp8
