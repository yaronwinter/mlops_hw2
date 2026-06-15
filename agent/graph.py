"""LangGraph agent: text-to-SQL with verify+revise loop.

Graph shape:

    START -> attach_schema -> generate_sql -> execute -> verify
                                                          |
                                              ok=true ----+----> END
                                                          |
                                              ok=false ---+----> revise -> execute -> verify (loop)

Loop is capped at MAX_ITERATIONS total generate/revise calls.

The execute node and the graph wiring are provided. `generate_sql_node` is
filled in as a worked example; you implement `verify`, `revise`, and the
conditional router following the same shape.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent import prompts
from agent.execution import ExecutionResult, execute_sql
from agent.schema import render_schema

# Total generate + revise calls before the loop is forced to stop.
# 3-5 is a reasonable range; tune it as part of Phase 3.
MAX_ITERATIONS = 3

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")


@dataclass
class AgentState:
    """State threaded through the graph. Extend with fields you need."""

    question: str
    db_id: str
    schema: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


def llm(max_tokens: int = 512, guided_json: dict | None = None) -> ChatOpenAI:
    """Helper to configure the LLM calls in generate/verify/revise nodes.
    * max_tokens bounds decode latency deterministically. Use a small cap for
      short replies (verify's tiny JSON) and a larger one for SQL generation.
    * guided_json, when set, asks vLLM's guided-decoding backend to constrain
      the output to a JSON schema - guarantees a parseable reply (e.g. verify's
      {"ok", "issue"}) and avoids retries from malformed output. It rides along
      via extra_body, which langchain forwards verbatim to the OpenAI-compatible
      request body; vLLM understands it, hosted OpenAI ignores it.
    """
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.0,
        max_tokens=max_tokens,
        extra_body={"guided_json": guided_json} if guided_json is not None else None,
    )


# JSON shape for the verify node's structured output. Pass to llm(guided_json=...)
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "issue": {"type": "string"},
    },
    "required": ["ok", "issue"],
}


def _attach_schema(state: AgentState) -> dict:
    """Provided. Render the DB schema once at the start of the run."""
    return {"schema": render_schema(state.db_id)}


def _extract_sql(text: str) -> str:
    """Pull a SQL statement out of an LLM reply, stripping markdown fences/prose.

    Intentionally simple: take the first ```sql ... ``` block if there is one,
    otherwise the whole reply. You may need to harden this for your prompts.
    """
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else text).strip()


def _parse_verdict(text: str) -> dict:
    """Parse the verifier's JSON reply defensively."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    raw = (fenced.group(1) if fenced else text).strip()
    if not raw.startswith("{"):
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        raw = brace.group(0) if brace else raw
    return json.loads(raw)


async def generate_sql_node(state: AgentState) -> dict:
    """Worked example - the other LLM nodes follow this same shape.

    Build messages from the prompts, call the shared llm(), extract the SQL,
    and return only the state fields you changed. `iteration` is bumped here
    (and in revise) so route_after_verify can enforce MAX_ITERATIONS.

    This node is wired and ready; fill in GENERATE_SQL_SYSTEM / GENERATE_SQL_USER
    in prompts.py to make it produce real queries.

    Async on purpose: `await llm(...).ainvoke(...)` yields the event loop while
    vLLM works, so concurrent agent runs overlap their LLM waits instead of each
    pinning a thread. Make verify/revise `async def` and `await` their calls too.
    """
    # max_tokens caps decode time: a SQL statement is short, so 256 is ample
    # headroom while bounding the worst-case latency of this call.
    response = await llm(max_tokens=256).ainvoke(
        [
            ("system", prompts.GENERATE_SQL_SYSTEM),
            (
                "user",
                prompts.GENERATE_SQL_USER.format(
                    schema=state.schema,
                    question=state.question,
                ),
            ),
        ]
    )
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "generate_sql", "sql": sql}],
    }


def execute_node(state: AgentState) -> dict:
    """Provided. Runs the SQL and stores the result."""
    return {"execution": execute_sql(state.db_id, state.sql)}


async def verify_node(state: AgentState) -> dict:
    """Decide whether state.execution plausibly answers state.question.

    Follow the generate_sql_node pattern: define this `async def` and build
    messages from the VERIFY_* prompts, `await llm(...).ainvoke(...)`, parse the
    reply. Ask the model for a small JSON object
    like {"ok": bool, "issue": str} and parse it defensively - the model may
    wrap it in prose or fences. state.execution.render() gives you a compact
    view of the rows or error to feed into the prompt.

    Return: {"verify_ok": <bool>, "verify_issue": <str>}.
    What counts as "not plausible" is yours to define - see the Phase 3 targets
    in the README.

    Latency tip: this reply is tiny, so call llm() with a small max_tokens and
    the VERIFY_SCHEMA guided-JSON, e.g.
        resp = await llm(max_tokens=128, guided_json=VERIFY_SCHEMA).ainvoke([...])
        data = json.loads(resp.content)  # guaranteed parseable
    """
    execution = state.execution

    # A SQL error (or no result) is unambiguously not plausible: short-circuit
    # to a revise without spending an LLM call. This also saves latency.
    if execution is None or not execution.ok:
        issue = (execution.error if execution else None) or "no execution result"
        return {
            "verify_ok": False,
            "verify_issue": issue,
            "history": state.history
            + [{"node": "verify", "ok": False, "issue": issue}],
        }

    # The query ran - ask the model whether the rows plausibly answer the question.
    response = await llm(max_tokens=128, guided_json=VERIFY_SCHEMA).ainvoke(
        [
            ("system", prompts.VERIFY_SYSTEM),
            (
                "user",
                prompts.VERIFY_USER.format(
                    schema=state.schema,
                    question=state.question,
                    sql=state.sql,
                    result=execution.render(),
                ),
            ),
        ]
    )
    try:
        data = _parse_verdict(response.content)
        ok = bool(data.get("ok", True))
        issue = "" if ok else str(data.get("issue", "")).strip()
    except (json.JSONDecodeError, ValueError, AttributeError):
        # Unparseable verdict: accept rather than burn an iteration on verifier
        # noise. The iteration cap is the backstop if this misjudges.
        ok, issue = True, ""

    return {
        "verify_ok": ok,
        "verify_issue": issue,
        "history": state.history + [{"node": "verify", "ok": ok, "issue": issue}],
    }


async def revise_node(state: AgentState) -> dict:
    """Produce a revised SQL query given state.verify_issue and the prior attempt.

    Same shape as generate_sql_node (also `async def`, `await llm(...).ainvoke`),
    but the prompt should include the failing SQL, its execution result, and the
    verifier's complaint so the model can fix it. Bump the iteration counter the
    same way generate_sql_node does so the loop terminates.

    Return: {"sql": <str>, "iteration": state.iteration + 1, ...}.
    """
    execution = state.execution
    result_view = execution.render() if execution is not None else "no execution result"

    response = await llm(max_tokens=256).ainvoke(
        [
            ("system", prompts.REVISE_SYSTEM),
            (
                "user",
                prompts.REVISE_USER.format(
                    schema=state.schema,
                    question=state.question,
                    sql=state.sql,
                    result=result_view,
                    issue=state.verify_issue,
                ),
            ),
        ]
    )
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history
        + [{"node": "revise", "sql": sql, "issue": state.verify_issue}],
    }


def route_after_verify(state: AgentState) -> str:
    """Conditional router: return "revise" to loop, "end" to terminate.

    Two reasons to end: the verifier was happy (state.verify_ok), or you've hit
    the iteration cap (state.iteration >= MAX_ITERATIONS). Otherwise, revise.
    """
    if state.verify_ok or state.iteration >= MAX_ITERATIONS:
        return "end"
    return "revise"


# ---- Graph wiring -----------------------------------------------------


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("attach_schema", _attach_schema)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise", "end": END},
    )
    g.add_edge("revise", "execute")
    return g.compile()


graph = build_graph()
