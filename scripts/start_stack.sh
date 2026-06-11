#!/usr/bin/env bash
#
# Bring up the full request path for load testing:
#   load_test/driver.py  ->  agent server (:8001)  ->  vLLM (:8000)
#
# It ensures vLLM is healthy (optionally restarting it), (re)starts the agent
# server, waits for both /health endpoints, then fires a few warmup requests so
# the prefix cache and CUDA graphs are hot before you measure.
#
# Per-iteration Phase 6 loop (pick a prepared config, restart, measure):
#   RESTART_VLLM=1 VLLM_SCRIPT=scripts/configs/01_fp8.sh scripts/start_stack.sh
#   uv run python load_test/driver.py --rps 10 --duration 300 --out results/run_01.json
#
# Omit VLLM_SCRIPT to use the baseline (scripts/start_vllm.sh). If you only
# changed agent code (graph.py), omit RESTART_VLLM=1 too - the agent server is
# always restarted, vLLM is just health-checked.
#
# Env overrides:
#   RESTART_VLLM   (default 0; set 1 to force a vLLM restart via restart_vllm.sh)
#   VLLM_SCRIPT    (default scripts/start_vllm.sh; passed through to restart_vllm.sh)
#   AGENT_PORT     (default 8001)
#   WARMUP         (default 3; number of warmup requests, 0 to skip)
#   VLLM_HEALTH / AGENT_HEALTH / READY_TIMEOUT / AGENT_TIMEOUT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WAIT="${SCRIPT_DIR}/wait_for_vllm.sh"
RESTART="${SCRIPT_DIR}/restart_vllm.sh"

VLLM_HEALTH="${VLLM_HEALTH:-http://localhost:8000/health}"
AGENT_PORT="${AGENT_PORT:-8001}"
AGENT_HEALTH="${AGENT_HEALTH:-http://localhost:${AGENT_PORT}/health}"
AGENT_LOG="${ROOT}/agent.log"
AGENT_PATTERN="uvicorn agent.server:app"
WARMUP="${WARMUP:-3}"

# --- 1. vLLM: restart if asked, otherwise just confirm it is healthy ---
if [ "${RESTART_VLLM:-0}" = "1" ]; then
    echo "RESTART_VLLM=1 -> restarting vLLM."
    bash "$RESTART"
else
    echo "Ensuring vLLM is up (set RESTART_VLLM=1 to force a restart)..."
    if ! bash "$WAIT" "$VLLM_HEALTH" "${READY_TIMEOUT:-600}"; then
        echo "ERROR: vLLM is not healthy. Start it first (scripts/restart_vllm.sh)." >&2
        exit 1
    fi
fi

# --- 2. (Re)start the agent server -------------------------------------
pids="$(pgrep -f "$AGENT_PATTERN" || true)"
if [ -n "$pids" ]; then
    echo "Stopping existing agent server (PIDs: ${pids})..."
    # shellcheck disable=SC2086
    kill ${pids} 2>/dev/null || true
    for _ in $(seq 1 15); do
        pgrep -f "$AGENT_PATTERN" >/dev/null || break
        sleep 1
    done
    pkill -9 -f "$AGENT_PATTERN" 2>/dev/null || true
fi

echo "Launching agent server on :${AGENT_PORT} (logging to ${AGENT_LOG})..."
cd "$ROOT"
# One worker on purpose: the handler is async, so a single event loop carries
# many concurrent runs. More workers just fragment the connection pool.
nohup uv run uvicorn agent.server:app --host 0.0.0.0 --port "$AGENT_PORT" --workers 1 \
    >"$AGENT_LOG" 2>&1 &
echo "Agent server launching in background (wrapper PID $!)."

# --- 3. Wait for the agent to be healthy -------------------------------
if ! bash "$WAIT" "$AGENT_HEALTH" "${AGENT_TIMEOUT:-120}"; then
    echo "ERROR: agent server failed to become healthy. Last 30 log lines:" >&2
    tail -n 30 "$AGENT_LOG" >&2 || true
    exit 1
fi

# --- 4. Warmup (best-effort) -------------------------------------------
# Hits the agent with a few real questions so the vLLM prefix cache (shared
# schema/system prompt) and CUDA graphs are warm before measurement. Failures
# are tolerated: pre-Phase-3 the verify node still raises, so /answer may 500 -
# that does not block the stack from being usable for a vLLM-only test.
if [ "${WARMUP}" -gt 0 ]; then
    echo "Warming up with up to ${WARMUP} request(s) (best-effort)..."
    uv run python - "$WARMUP" "$AGENT_PORT" <<'PY' || echo "Warmup skipped/failed; continuing."
import json, sys, urllib.request
from pathlib import Path

n = int(sys.argv[1])
port = sys.argv[2]
pool = Path("load_test/perf_pool.jsonl")
items = [json.loads(l) for l in pool.read_text().splitlines() if l.strip()][:n]
url = f"http://localhost:{port}/answer"
for i, q in enumerate(items, 1):
    payload = json.dumps({"question": q["question"], "db": q["db_id"]}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            print(f"  warmup {i}/{len(items)}: HTTP {r.status}")
    except Exception as e:  # noqa: BLE001
        print(f"  warmup {i}/{len(items)}: {type(e).__name__}: {e}")
PY
fi

echo
echo "Stack is up. Run the load test, e.g.:"
echo "  uv run python load_test/driver.py --rps 10 --duration 300"
echo "Logs:  vLLM -> ${ROOT}/vllm.log   agent -> ${AGENT_LOG}"
