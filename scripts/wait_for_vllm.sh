#!/usr/bin/env bash
#
# Poll vLLM's /health until it answers 200, or time out. Use this between
# launching vLLM and starting a load test so you don't measure during warmup.
#
# Usage:
#   scripts/wait_for_vllm.sh [HEALTH_URL] [TIMEOUT_SECONDS]
# Defaults: http://localhost:8000/health, 600s.

set -euo pipefail

URL="${1:-http://localhost:8000/health}"
TIMEOUT="${2:-600}"
INTERVAL=3

echo "Waiting for ${URL} (timeout ${TIMEOUT}s)..."
deadline=$(( $(date +%s) + TIMEOUT ))
while true; do
    if curl -sf -o /dev/null --max-time 2 "$URL"; then
        echo "${URL} is ready."
        exit 0
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "ERROR: ${URL} did not become ready within ${TIMEOUT}s." >&2
        exit 1
    fi
    sleep "$INTERVAL"
done
