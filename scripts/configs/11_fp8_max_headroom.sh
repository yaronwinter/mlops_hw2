#!/usr/bin/env bash
#
# Run 11 - STACKED best-of: maximum concurrency headroom (most aggressive).
#
# Everything that frees KV cache, stacked:
#   Run 2 (--max-num-seqs 96) + Run 5 (--max-model-len 4096)
#   + Run 4 (--kv-cache-dtype fp8).
# This is Run 10 plus FP8 KV cache - the highest concurrency you can wring out
# of the 80GB card.
#
# WHEN TO RUN: only if you are STILL KV-bound after Run 10 (KV cache near 100%
# at the RPS you need). If Run 10 already met the SLO comfortably, skip this -
# it adds quality risk for headroom you don't need.
#
# QUALITY GUARD: FP8 KV cache is the most accuracy-sensitive lever here.
# RE-RUN THE EVAL and only keep this config if the pass rate holds.
# COMPARE AGAINST: Run 10 (to isolate the kv-cache-dtype delta) and Run 1.
# WATCH: 4096-window truncation (http_errors), and eval regression vs Run 10.

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
    --quantization fp8 \
    --kv-cache-dtype fp8
