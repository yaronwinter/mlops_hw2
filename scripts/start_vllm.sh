#!/usr/bin/env bash
#
# Run 0 - BASELINE (BF16, no quantization).
#
# This is the starting point of the Phase 6 iteration. Every tuned flag below
# is held constant across the experiment so that each config in scripts/configs/
# changes exactly ONE lever versus this baseline. Quantization is deliberately
# OFF here so Run 1 (scripts/configs/01_fp8.sh) can A/B it on the same weights.
#
# Launch via:  scripts/restart_vllm.sh        (or RESTART_VLLM=1 scripts/start_stack.sh)
# Reference:   https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

set -euo pipefail

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    `# prompt(<=3K)+output(<=512) with headroom; default 262K wastes memory profiling` \
    --max-model-len 8192 \
    `# more KV-cache room; leave headroom for CUDA graphs` \
    --gpu-memory-utilization 0.92 \
    `# schema + system prompt are shared across calls -> prefill the prefix once` \
    --enable-prefix-caching \
    `# interleave big 3K-token prefills with in-flight decodes -> protects TTFT tail` \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    `# concurrency dial: raise if the queue grows, lower if TTFT spikes under load` \
    --max-num-seqs 48
# Run 1 adds: --quantization fp8   (see scripts/configs/01_fp8.sh)
# Deliberately omitted: --tensor-parallel-size (1 GPU), --enforce-eager (keep CUDA graphs).
