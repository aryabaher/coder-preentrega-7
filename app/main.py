"""FastAPI: POST /tasks enqueues a job_id. GET /tasks/{id} reads Redis and does not block.

POST /tasks/{id}/approve lives in app/hitl.py. GET /threads/{thread_id} reads the RedisSaver checkpoints.
"""

from requisitos import exigir_python_312

exigir_python_312()

import asyncio
import os
from collections.abc import AsyncIterator, MutableMapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from redis.exceptions import RedisError

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from app.observability import init_observability  # noqa: E402

init_observability()

from app import hitl, worker  # noqa: E402

REDIS_ERRORS = (RedisError, ConnectionError, TimeoutError, OSError)


class TaskRequest(BaseModel):
    """Body of POST /tasks. Send the thread_id of a previous task to continue that conversation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, min_length=1, max_length=200)


class TaskResponse(BaseModel):
    job_id: str
    status: str
    thread_id: str


class TaskStatus(BaseModel):
    job_id: str
    thread_id: str
    status: str
    query: str
    result: str | None = None
    error: str | None = None
    interrupt: dict[str, object] | None = None


class ThreadMessage(BaseModel):
    role: str
    name: str | None = None
    content: str


class ThreadHistory(BaseModel):
    thread_id: str
    messages: list[ThreadMessage]


def worker_in_api() -> bool:
    return os.getenv("WORKER_IN_API", "1") != "0"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await worker.connect()
    loop_task: asyncio.Task[None] | None = None
    if worker_in_api():
        await worker.recover_orphans()
        loop_task = asyncio.create_task(worker.task_worker())
    try:
        yield
    finally:
        if loop_task is not None:
            loop_task.cancel()
            try:
                await loop_task
            except asyncio.CancelledError:
                pass


def skip_api_spans(scope: MutableMapping[str, Any]) -> bool:
    # Phoenix computes p95 and cost per trace. A 2 ms span per polling GET would be a trace of its own.
    return str(scope.get("path", "")).startswith(("/tasks", "/threads"))


app = FastAPI(lifespan=lifespan, telemetry={"exclude": skip_api_spans})
app.include_router(hitl.router)


@app.post("/tasks", response_model=TaskResponse)
async def create_task(request: TaskRequest) -> TaskResponse:
    """Enqueue the query and return the ticket. The graph runs in the worker."""
    try:
        record = await worker.enqueue_task(request.query, request.thread_id)
    except worker.ThreadBusy as exc:
        raise HTTPException(status_code=409, detail="Thread is busy") from exc
    except REDIS_ERRORS as exc:
        raise HTTPException(status_code=503, detail="Redis is unavailable") from exc
    return TaskResponse(job_id=record["job_id"], status="PENDING", thread_id=record["thread_id"])


@app.get("/tasks/{job_id}", response_model=TaskStatus)
async def get_task(job_id: str) -> TaskStatus:
    """Current job record. Polling this does not run the agent."""
    try:
        record = await worker.read_job(job_id)
    except REDIS_ERRORS as exc:
        raise HTTPException(status_code=503, detail="Redis is unavailable") from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return TaskStatus.model_validate(record)


@app.get("/threads/{thread_id}", response_model=ThreadHistory)
async def get_thread(thread_id: str) -> ThreadHistory:
    """Conversation stored by RedisSaver for that thread_id."""
    try:
        messages = await worker.read_thread(thread_id)
    except REDIS_ERRORS as exc:
        raise HTTPException(status_code=503, detail="Redis is unavailable") from exc
    if messages is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return ThreadHistory(thread_id=thread_id, messages=[ThreadMessage.model_validate(m) for m in messages])
