#!/usr/bin/env bash
#
# Push-button restart for the Phase 6 iterate-measure loop:
#   1. stop any running vLLM (graceful, then forced),
#   2. wait for the GPU to actually release its memory,
#   3. relaunch scripts/start_vllm.sh in the background (logging to vllm.log),
#   4. block until /health is green.
#
# Workflow:  pick a config  ->  scripts/restart_vllm.sh [config]  ->  load test
#   scripts/restart_vllm.sh                              # baseline (start_vllm.sh)
#   scripts/restart_vllm.sh scripts/configs/01_fp8.sh    # a specific config
#   VLLM_SCRIPT=scripts/configs/02_fp8_seqs96.sh scripts/restart_vllm.sh
#
# Env overrides:
#   VLLM_SCRIPT    (default scripts/start_vllm.sh; first positional arg wins)
#   HEALTH_URL     (default http://localhost:8000/health)
#   READY_TIMEOUT  (default 600  - seconds to wait for /health)
#   GPU_FREE_MIB   (default 2000 - consider the GPU "free" below this many MiB)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Which launch config to run: first positional arg, else $VLLM_SCRIPT, else baseline.
START="${1:-${VLLM_SCRIPT:-${SCRIPT_DIR}/start_vllm.sh}}"
if [ ! -f "$START" ]; then
    echo "ERROR: launch config not found: ${START}" >&2
    exit 1
fi
echo "Using launch config: ${START}"
WAIT="${SCRIPT_DIR}/wait_for_vllm.sh"
LOG="${SCRIPT_DIR}/../vllm.log"
PATTERN="vllm.entrypoints.openai.api_server"
HEALTH_URL="${HEALTH_URL:-http://localhost:8000/health}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"
GPU_FREE_MIB="${GPU_FREE_MIB:-2000}"

# --- 1. Stop any running vLLM -----------------------------------------
pids="$(pgrep -f "$PATTERN" || true)"
if [ -n "$pids" ]; then
    echo "Stopping vLLM (PIDs: ${pids})..."
    # shellcheck disable=SC2086
    kill ${pids} 2>/dev/null || true
    # Give it up to 30s to exit cleanly, then force.
    for _ in $(seq 1 30); do
        pgrep -f "$PATTERN" >/dev/null || break
        sleep 1
    done
    if pgrep -f "$PATTERN" >/dev/null; then
        echo "Still alive after 30s; sending SIGKILL."
        pkill -9 -f "$PATTERN" || true
        sleep 2
    fi
    echo "vLLM stopped."
else
    echo "No running vLLM found."
fi

# --- 2. Wait for GPU memory to free (best-effort) ---------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Waiting for GPU memory to drop below ${GPU_FREE_MIB} MiB..."
    for _ in $(seq 1 30); do
        used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
        echo "  GPU memory used: ${used:-?} MiB"
        if [ "${used:-999999}" -lt "$GPU_FREE_MIB" ]; then
            break
        fi
        sleep 2
    done
else
    echo "nvidia-smi not found; skipping GPU memory check."
fi

# --- 3. Relaunch in the background ------------------------------------
echo "Launching vLLM (logging to ${LOG})..."
nohup bash "$START" >"$LOG" 2>&1 &
echo "vLLM launching in background (wrapper PID $!); model load + compile takes a minute or two."

# --- 4. Block until healthy -------------------------------------------
if ! bash "$WAIT" "$HEALTH_URL" "$READY_TIMEOUT"; then
    echo "ERROR: vLLM failed to become healthy. Last 30 log lines:" >&2
    tail -n 30 "$LOG" >&2 || true
    exit 1
fi

echo
echo "vLLM is up. Now warm up + load test, e.g.:"
echo "  uv run python load_test/driver.py --rps 10 --duration 300"
echo "Follow the server log with:  tail -f ${LOG}"
