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
FOUND = "found"
END2END = "end2end"
LATENCY_SECONDS = "latency_seconds"
ISSUE = "issue"


def run_sql(
    db_id: str, sql: str, timeout: float = 5.0
) -> tuple[bool, list[tuple] | None, str | None]:
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

    proposals: list[dict] = []
    verifications: list[dict] = []
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
        num_iterations = data.get(ITERATIONS, 0)
        proposals = [
            node
            for node in data.get(HISTORY, [])
            if node.get(NODE) in ("generate_sql", "revise")
        ]
        verifications = [
            node for node in data.get(HISTORY, []) if node.get(NODE) == "verify"
        ]
        assert (
            num_iterations == len(proposals)
        ), f"agent said {num_iterations} iterations but returned {len(proposals)} proposals entries: {json.dumps(data, indent=2)}"
        assert (
            num_iterations == len(verifications)
        ), f"agent said {num_iterations} iterations but returned {len(verifications)} verifications entries: {json.dumps(data, indent=2)}"
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"{type(e).__name__}: {e}")
    latency = time.monotonic() - t0

    found_at_iter = [False] * (MAX_ITERATIONS + 1)
    e2e_at_iter = [False] * (MAX_ITERATIONS + 1)
    is_correct = False
    i = 0
    while (i < num_iterations) and not is_correct:
        sql = proposals[i].get(SQL)
        issue = verifications[i].get(ISSUE)

        _, pred_rows, _ = run_sql(db_id, sql)
        is_correct = matches(gold_rows, pred_rows)
        is_verified = len(issue) == 0

        for k in range(i + 1, MAX_ITERATIONS + 1):
            found_at_iter[
                k
            ] = is_correct  # SQL is correct at iteration i, so it's also correct at every later iteration
            e2e_at_iter[k] = (
                is_verified and is_correct
            )  # end-to-end success at iteration i, so also at every later iteration
        i += 1

    return {
        QUESTION: q_text,
        DB_ID: db_id,
        ITERATIONS: num_iterations,
        FOUND: found_at_iter,
        END2END: e2e_at_iter,
        LATENCY_SECONDS: round(latency, 3),
    }


def summarize(results: list[dict]) -> tuple:
    """Aggregate per-question results."""
    # pass[k] = how many questions are correct if we stop after iteration k,
    # carrying a terminated question's last candidate forward to every later k.
    found_at_iter = [0] * (MAX_ITERATIONS + 1)
    e2e_at_iter = [0] * (MAX_ITERATIONS + 1)
    iter_distribution = [0] * (MAX_ITERATIONS + 1)
    for r in results:
        iter_distribution[r[ITERATIONS]] += 1
        for k in range(1, (MAX_ITERATIONS + 1)):
            try:
                found_at_iter[k] += int(r[FOUND][k])
                e2e_at_iter[k] += int(r[END2END][k])
            except Exception as e:
                raise ValueError(f"found: {r[FOUND]}, end2end: {r[END2END]}, e: {e}")

    return found_at_iter, e2e_at_iter, iter_distribution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [
        json.loads(line)
        for line in args.eval_set.read_text().splitlines()
        if line.strip()
    ]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        eval_res = eval_one(q, args.agent_url)
        print(
            f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}..., found: {eval_res[FOUND][-1]}, end-to-end: {eval_res[END2END][-1]}, latency: {eval_res[LATENCY_SECONDS]}",
            flush=True,
        )
        results.append(eval_res)
    elapsed = time.monotonic() - t0

    assert len(results) == len(
        questions
    ), f"got {len(results)} results but expected {len(questions)}"
    found_at_iter, e2e_at_iter, iter_distribution = summarize(results)

    out = {
        "num_questions": len(questions),
        "found accuracy": found_at_iter[-1] / len(questions),
        "end-to-end accuracy": e2e_at_iter[-1] / len(questions),
        "found_at_iter": {i: found_at_iter[i] for i in range(1, (MAX_ITERATIONS + 1))},
        "found_at_iter_percent": {
            i: f"{found_at_iter[i] / len(questions) * 100:.3f}"
            for i in range(1, (MAX_ITERATIONS + 1))
        },
        "end2end_at_iter": {i: e2e_at_iter[i] for i in range(1, (MAX_ITERATIONS + 1))},
        "end2end_at_iter_percent": {
            i: f"{e2e_at_iter[i] / len(questions) * 100:.3f}"
            for i in range(1, (MAX_ITERATIONS + 1))
        },
        "iter_distribution": {
            i: iter_distribution[i] for i in range(1, (MAX_ITERATIONS + 1))
        },
        "mean_latency_seconds": elapsed / len(questions),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=4))
    print(f"Wrote {args.out}")
    print(json.dumps(out, indent=4))


if __name__ == "__main__":
    main()
