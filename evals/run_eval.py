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
from agent.graph import MAX_ITERATIONS

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"

NODE = "node"
SQL = "sql"
ITERATIONS = "iterations"
HISTORY = "history"
QUESTION = "question"
DB_ID = "db_id"
GOLD_SQL = "gold_sql"
TAGS = "tags"
GOLD_EXEC_OK = "gold_exec_ok"
GOLD_ERROR = "gold_error"
FINAL_SQL = "final_sql"
CORRECT = "correct"
AGENT_ERROR = "agent_error"
LATENCY_SECONDS = "latency_seconds"

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


def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    db_id = question[DB_ID]
    gold_sql = question[GOLD_SQL]
    q_text = question[QUESTION]

    # Gold result set, computed once. If gold itself can't run, every comparison
    # is a miss (matches() returns False on None) - we record it so a broken gold
    # row is visible rather than silently dragging the pass rate down.
    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)

    final_sql = ""
    history: list[dict] = []
    agent_error: str | None = None
    t0 = time.monotonic()
    num_iterations = 0
    try:
        resp = httpx.post(
            agent_url,
            json={QUESTION: q_text, "db": db_id, TAGS: {"eval": "baseline"}},
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        final_sql = data.get(SQL, "")
        num_iterations = data.get(ITERATIONS, 0)
        history = [node for node in data.get(HISTORY, []) if node.get(NODE) in ("generate_sql", "revise")]
        assert num_iterations == len(history), f"agent said {num_iterations} iterations but returned {len(history)} history entries: {json.dumps(data, indent=2)}"
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"{type(e).__name__}: {e}")
    latency = time.monotonic() - t0

    pred_ok, pred_rows, pred_err = run_sql(db_id, final_sql)
    is_correct = matches(gold_rows, pred_rows)
    return {
        QUESTION: q_text,
        DB_ID: db_id,
        GOLD_SQL: gold_sql,
        GOLD_EXEC_OK: gold_ok,
        GOLD_ERROR: gold_err,
        FINAL_SQL: final_sql,
        ITERATIONS: num_iterations,
        CORRECT: is_correct,
        AGENT_ERROR: agent_error,
        LATENCY_SECONDS: round(latency, 3),
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results."""
    # pass[k] = how many questions are correct if we stop after iteration k,
    # carrying a terminated question's last candidate forward to every later k.
    pass_at_iter = [0] * (MAX_ITERATIONS + 1)
    iter_distribution = [0] * (MAX_ITERATIONS + 1)
    for r in results:
        iter_distribution[r[ITERATIONS]] += 1            
        if not r[CORRECT]:
            continue  # wrong at every k

        for k in range(r[ITERATIONS], (MAX_ITERATIONS + 1)):
            pass_at_iter[k] += 1

    return pass_at_iter, iter_distribution

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

    assert len(results) == len(questions), f"got {len(results)} results but expected {len(questions)}"
    pass_at_iter, iter_distribution = summarize(results)
    out = {
        "num_questions": len(questions),
        "accuracy": pass_at_iter[-1] / len(questions),
        "pass_at_iter": {i: pass_at_iter[i] for i in range(1, len(pass_at_iter))},
        "accuracy_at_iter": {i: pass_at_iter[i] / len(questions) for i in range(1, len(pass_at_iter))},
        "iter_distribution": {i: iter_distribution[i] for i in range(1, len(iter_distribution))},
        "mean_latency_seconds": elapsed / len(questions),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=4))
    print(f"Wrote {args.out}")
    print(json.dumps(out, indent=4))


if __name__ == "__main__":
    main()
