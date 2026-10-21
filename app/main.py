"""FastAPI: POST /tasks enqueues a job_id. GET /tasks/{id} reads Redis and does not block.

POST /tasks/{id}/approve resumes a job paused on interrupt.
"""

from requisitos import exigir_python_312

exigir_python_312()

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from redis import asyncio as aioredis

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from app.observability import iniciar_observabilidad  # noqa: E402

iniciar_observabilidad()

from app import worker  # noqa: E402
from app.graph import compilar, esta_listo, usar  # noqa: E402
from app.worker import JobNotFound, NotAwaitingApproval, encolar, encolar_reanudacion, leer  # noqa: E402


class TaskRequest(BaseModel):
    """Body of POST /tasks. A blank query never reaches the graph."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=4000)


class TaskResponse(BaseModel):
    job_id: str
    status: str


class TaskStatus(BaseModel):
    job_id: str
    status: str
    query: str
    result: str | None = None
    error: str | None = None
    interrupt: dict[str, object] | None = None


class ApprovalRequest(BaseModel):
    """Body of POST /tasks/{job_id}/approve. approved=false resumes and skips the side effect."""

    approved: bool


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not esta_listo():
        usar(compilar())
    if worker.redis_client is None:
        worker.redis_client = aioredis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True,
        )
    tarea = asyncio.create_task(worker.task_worker())
    try:
        yield
    finally:
        tarea.cancel()
        try:
            await tarea
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=lifespan)


def _redis_caido(exc: BaseException) -> HTTPException:
    return HTTPException(status_code=503, detail="Redis is unavailable")


@app.post("/tasks", response_model=TaskResponse)
async def create_task(request: TaskRequest) -> TaskResponse:
    """Enqueue the query and return the ticket. The graph runs in the worker."""
    try:
        job_id = await encolar(request.query)
    except (ConnectionError, TimeoutError, OSError) as exc:
        raise _redis_caido(exc) from exc
    return TaskResponse(job_id=job_id, status="PENDING")


@app.get("/tasks/{job_id}", response_model=TaskStatus)
async def get_task(job_id: str) -> TaskStatus:
    """Current job record. Polling this does not run the agent."""
    try:
        estado = await leer(job_id)
    except (ConnectionError, TimeoutError, OSError) as exc:
        raise _redis_caido(exc) from exc
    if estado is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return TaskStatus.model_validate(estado)


@app.post("/tasks/{job_id}/approve", response_model=TaskResponse)
async def approve_task(job_id: str, body: ApprovalRequest) -> TaskResponse:
    """Resume a job paused on interrupt. Returns without waiting for the graph."""
    try:
        await encolar_reanudacion(job_id, body.approved)
    except JobNotFound as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    except NotAwaitingApproval as exc:
        raise HTTPException(status_code=409, detail="Job is not awaiting approval") from exc
    except (ConnectionError, TimeoutError, OSError) as exc:
        raise _redis_caido(exc) from exc
    return TaskResponse(job_id=job_id, status="RUNNING")
