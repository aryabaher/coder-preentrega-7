"""Human approval node: interrupt + endpoint POST /tasks/{id}/approve."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.types import interrupt
from pydantic import BaseModel
from redis.exceptions import RedisError

from app import worker

TOOL_NAME = "register_payroll_adjustment"
SIDE_EFFECT = "side_effect"
COST = "cost"

router = APIRouter()
side_effects: list[dict[str, str]] = []


class ApprovalRequest(BaseModel):
    """Body of POST /tasks/{job_id}/approve. approved=false resumes and skips the paused step."""

    approved: bool


class ApprovalResponse(BaseModel):
    job_id: str
    status: str


@router.post("/tasks/{job_id}/approve", response_model=ApprovalResponse)
async def approve_task(job_id: str, body: ApprovalRequest) -> ApprovalResponse:
    """Resume a job paused on interrupt. Returns without waiting for the graph."""
    try:
        await worker.enqueue_resume(job_id, body.approved)
    except worker.JobNotFound as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    except worker.NotAwaitingApproval as exc:
        raise HTTPException(status_code=409, detail="Job is not awaiting approval") from exc
    except (RedisError, ConnectionError, TimeoutError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Redis is unavailable") from exc
    return ApprovalResponse(job_id=job_id, status="RUNNING")


@tool
def register_payroll_adjustment(detail: str) -> str:
    """Register a payroll adjustment. Side effect: it is recorded and cannot be undone."""
    text = detail.strip()
    if not text:
        return "Adjustment not registered: the detail is empty."
    side_effects.append({"tool": TOOL_NAME, "detail": text})
    return f"Adjustment registered: {text}"


def latest_question(messages: list[BaseMessage]) -> str:
    """Text of the last human message: the question of the current turn."""
    for message in reversed(messages):
        if message.type == "human":
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def approval_node(state: dict[str, Any]) -> dict[str, Any]:
    """Pause with interrupt. Nothing with side effects or extra cost runs before the answer."""
    reason = state.get("approval_reason") or SIDE_EFFECT
    question = latest_question(state["messages"])
    if reason == COST:
        payload: dict[str, Any] = {
            "reason": "estimated cost above COSTO_MAXIMO_USD",
            "estimated_cost_usd": state.get("estimated_cost_usd"),
            "limit_usd": state.get("cost_limit_usd"),
            "query": question,
        }
    else:
        payload = {"reason": f"side-effect tool {TOOL_NAME}", "tool": TOOL_NAME, "query": question}
    decision = interrupt(payload)
    approved = _approved(decision)
    if reason == COST:
        return {"cost_approved": approved, "next_agent": "supervisor" if approved else "FINISH"}
    return {"approved": approved, "next_agent": "ejecutor" if approved else "FINISH"}


def executor_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run register_payroll_adjustment only after a person approved it."""
    if state.get("approved") is not True:
        raise PermissionError(f"{TOOL_NAME} without approval. The graph has to pass through interrupt first.")
    result = str(register_payroll_adjustment.invoke({"detail": latest_question(state["messages"])}))
    return {"side_effect_done": True, "side_effect_result": result}


def route_after_approval(state: dict[str, Any]) -> Literal["ejecutor", "supervisor", "sintesis"]:
    """The critical edge: the side effect only runs when the answer was yes."""
    if state.get("approval_reason") == COST:
        return "supervisor" if state.get("cost_approved") is True else "sintesis"
    return "ejecutor" if state.get("approved") is True else "sintesis"


def _approved(decision: object) -> bool:
    if isinstance(decision, dict):
        return bool(decision.get("approved"))
    return bool(decision)
