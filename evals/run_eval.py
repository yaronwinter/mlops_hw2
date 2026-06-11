"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


def _sql_candidates(history: list[dict]) -> list[str]:
    """Ordered SQL the agent produced, one per iteration."""
    return [h["sql"] for h in history if h.get("node") in ("generate_sql", "revise") and "sql" in h]


def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]
    q_text = question["question"]

    # Gold result set, computed once. If gold itself can't run, every comparison
    # is a miss (matches() returns False on None) - we record it so a broken gold
    # row is visible rather than silently dragging the pass rate down.
    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)

    final_sql = ""
    history: list[dict] = []
    agent_iterations = 0
    agent_error: str | None = None
    t0 = time.monotonic()
    try:
        resp = httpx.post(
            agent_url,
            json={"question": q_text, "db": db_id, "tags": {"eval": "baseline"}},
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        final_sql = data.get("sql", "")
        agent_iterations = data.get("iterations", 0)
        history = data.get("history", [])
    except Exception as e:  # noqa: BLE001
        agent_error = f"{type(e).__name__}: {e}"
    latency = time.monotonic() - t0

    candidates = _sql_candidates(history)
    # Fallback: an older server (or one with tracing off) may omit history but
    # still return the final SQL. Treat that as a single iteration-0 candidate.
    if not candidates and final_sql:
        candidates = [final_sql]

    per_iteration: list[dict] = []
    for sql in candidates:
        pred_ok, pred_rows, pred_err = run_sql(db_id, sql)
        per_iteration.append({
            "sql": sql,
            "exec_ok": pred_ok,
            "exec_error": pred_err,
            "correct": matches(gold_rows, pred_rows),
        })

    final_correct = per_iteration[-1]["correct"] if per_iteration else False

    return {
        "question": q_text,
        "db_id": db_id,
        "gold_sql": gold_sql,
        "gold_exec_ok": gold_ok,
        "gold_error": gold_err,
        "final_sql": final_sql,
        "agent_iterations": agent_iterations,
        "num_candidates": len(candidates),
        "per_iteration": per_iteration,
        "final_correct": final_correct,
        "agent_error": agent_error,
        "latency_seconds": round(latency, 3),
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results."""
    n = len(results)
    max_iters = max((r["num_candidates"] for r in results), default=0)

    # pass[k] = how many questions are correct if we stop after iteration k,
    # carrying a terminated question's last candidate forward to every later k.
    pass_at_iter = [0] * max_iters
    for r in results:
        pi = r["per_iteration"]
        if not pi:
            continue  # agent never produced runnable SQL -> wrong at every k
        for k in range(max_iters):
            idx = k if k < len(pi) else len(pi) - 1  # carry-forward
            if pi[idx]["correct"]:
                pass_at_iter[idx] += 1

    # How many questions terminated after exactly i iterations (1-based count).
    iter_distribution: dict[str, int] = {}
    for r in results:
        key = str(r["num_candidates"])
        iter_distribution[key] = iter_distribution.get(key, 0) + 1

    n_final_correct = sum(1 for r in results if r["final_correct"])

    return {
        "n_questions": n,
        "execution_accuracy": round(n_final_correct / n, 4) if n else 0.0,
        "n_final_correct": n_final_correct,
        # Indexed by iteration: [0] = stop after first generate, [1] = after 1st
        # revise, ... Compare [0] vs the last entry: if equal, the verify/revise
        # loop earned nothing; if the tail is higher, the loop is paying off.
        "pass_rate_by_iteration": [round(c / n, 4) if n else 0.0 for c in pass_at_iter],
        "pass_count_by_iteration": pass_at_iter,
        "iteration_distribution": iter_distribution,
        "n_agent_errors": sum(1 for r in results if r["agent_error"]),
        "n_gold_unrunnable": sum(1 for r in results if not r["gold_exec_ok"]),
        "mean_latency_seconds": round(sum(r["latency_seconds"] for r in results) / n, 3) if n else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
