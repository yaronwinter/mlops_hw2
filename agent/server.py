"""FastAPI wrapper exposing the agent over HTTP.

Run:
    uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001
"""
from __future__ import annotations

import os
import sys
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# override=True makes .env authoritative over any pre-existing shell variables.
load_dotenv(override=True)

from agent.graph import AgentState, graph  # noqa: E402

_langfuse: Any = None
_lf_handler: Any = None
_propagate_attributes: Any = None
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    from langfuse import get_client, propagate_attributes
    from langfuse.langchain import CallbackHandler

    _langfuse = get_client()
    _lf_handler = CallbackHandler()
    _propagate_attributes = propagate_attributes
    _host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    try:
        if _langfuse.auth_check():
            print(f"[langfuse] tracing enabled -> {_host}", file=sys.stderr)
        else:
            print(
                f"[langfuse] WARNING: auth_check failed for {_host}. Traces will "
                "NOT be recorded - check LANGFUSE_PUBLIC_KEY / SECRET_KEY / HOST "
                "against your project's API keys.",
                file=sys.stderr,
            )
    except Exception as e:  # noqa: BLE001
        print(f"[langfuse] WARNING: cannot reach {_host}: {e}", file=sys.stderr)
else:
    print("[langfuse] keys not set - tracing disabled", file=sys.stderr)


app = FastAPI()


class AnswerRequest(BaseModel):
    question: str
    db: str
    tags: dict[str, str] = {}


class AnswerResponse(BaseModel):
    sql: str
    rows: list[list[Any]] | None
    iterations: int
    ok: bool
    error: str | None = None
    history: list[dict[str, Any]] = []


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("shutdown")
def _flush_langfuse() -> None:
    # "Common mistake": without flush(), buffered traces can be lost on exit.
    if _langfuse is not None:
        _langfuse.flush()


# Async handler: the agent run is I/O-bound (it spends almost all its time
# waiting on vLLM over HTTP). An async endpoint + ainvoke lets one event-loop
# thread carry many concurrent runs, instead of a sync handler that pins a
# threadpool worker per request and starves under the ~25 req/s of load.
@app.post("/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest) -> AnswerResponse:
    state = AgentState(question=req.question, db_id=req.db)
    try:
        final = await _run_traced(req, state)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    sql = final.get("sql", "")
    iteration = final.get("iteration", 0)
    history = final.get("history", [])
    execution = final.get("execution")

    if execution is None:
        return AnswerResponse(
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error="agent produced no execution result",
            history=history,
        )
    if not execution.ok:
        return AnswerResponse(
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error=execution.error,
            history=history,
        )

    return AnswerResponse(
        sql=sql,
        rows=[list(r) for r in (execution.rows or [])],
        iterations=iteration,
        ok=True,
        history=history,
    )


async def _run_traced(req: AnswerRequest, state: AgentState) -> dict:
    """Invoke the graph, wrapped in a Langfuse trace when tracing is enabled."""
    if _lf_handler is None:
        return await graph.ainvoke(state, config={"callbacks": []})

    # Langfuse tags are a flat list; carry the feature plus any request tags so
    # they're filterable in the UI (needed for the Phase 6 load-test breakdown).
    tags = ["sql-agent"] + [f"{k}={v}" for k, v in req.tags.items()]
    with _langfuse.start_as_current_observation(
        as_type="span",
        name="sql-agent",
        input={"question": req.question, "db": req.db},
    ) as root_span:
        with _propagate_attributes(
            trace_name="sql-agent",
            tags=tags,
            metadata=dict(req.tags) or None,
        ):
            final = await graph.ainvoke(state, config={"callbacks": [_lf_handler]})

        execution = final.get("execution")
        root_span.update(output={
            "sql": final.get("sql", ""),
            "ok": bool(execution and execution.ok),
            "iterations": final.get("iteration", 0),
            "rows": (execution.row_count if execution and execution.ok else None),
            "error": (execution.error if execution and not execution.ok else None),
        })
        return final
