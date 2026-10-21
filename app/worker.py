"""Background worker. The endpoint only enqueues; this loop updates Redis."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import app.graph as graph_mod

QUEUE_NAME = "tasks_queue"
STATUS_PREFIX = "status:"
RECURSION_LIMIT = 25

redis_client: Any = None


class JobNotFound(Exception):
    """No Redis record for that job_id."""


class NotAwaitingApproval(Exception):
    """approve was called and the job is not paused on the human node."""


def _cliente() -> Any:
    if redis_client is None:
        raise ConnectionError("Redis is unavailable")
    return redis_client


def _clave(job_id: str) -> str:
    return f"{STATUS_PREFIX}{job_id}"


async def _set(job_id: str, estado: dict[str, Any]) -> None:
    await _cliente().set(_clave(job_id), json.dumps(estado))


async def _push(payload: dict[str, Any]) -> None:
    await _cliente().rpush(QUEUE_NAME, json.dumps(payload))


async def leer(job_id: str) -> dict[str, Any] | None:
    raw = await _cliente().get(_clave(job_id))
    if raw is None:
        return None
    return json.loads(raw)


async def encolar(consulta: str) -> str:
    """Write PENDING, then push. The worker must not win that race."""
    job_id = str(uuid.uuid4())
    estado = {
        "job_id": job_id,
        "status": "PENDING",
        "query": consulta,
        "result": None,
        "error": None,
        "interrupt": None,
    }
    await _set(job_id, estado)
    await _push({"action": "start", "job_id": job_id})
    return job_id


async def encolar_reanudacion(job_id: str, approved: bool) -> None:
    estado = await leer(job_id)
    if estado is None:
        raise JobNotFound(job_id)
    if estado["status"] != "AWAITING_APPROVAL":
        raise NotAwaitingApproval(estado["status"])
    estado["status"] = "RUNNING"
    estado["interrupt"] = None
    estado["error"] = None
    await _set(job_id, estado)
    await _push({"action": "resume", "job_id": job_id, "approved": approved})


def _config(job_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": job_id}, "tags": ["preentrega-7"]}


def _llamar(ctx: object, fn: Any, args: tuple[Any, ...]) -> Any:
    """The worker thread keeps the trace that the event loop opened."""
    from opentelemetry import context as otel_context

    token = otel_context.attach(ctx)
    try:
        return fn(*args)
    finally:
        otel_context.detach(token)


async def _en_hilo(nombre: str, job_id: str, fn: Any, *args: Any) -> Any:
    from opentelemetry import context as otel_context
    from opentelemetry import trace

    with trace.get_tracer("preentrega-7").start_as_current_span(nombre) as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("job_id", job_id)
        return await asyncio.to_thread(_llamar, otel_context.get_current(), fn, args)


async def _cerrar(job_id: str, resultado: dict[str, Any]) -> None:
    estado = await leer(job_id)
    if estado is None:
        return
    if resultado["paused"]:
        estado["status"] = "AWAITING_APPROVAL"
        estado["interrupt"] = resultado["interrupt"]
        estado["result"] = None
        estado["error"] = None
    else:
        estado["status"] = "DONE"
        estado["result"] = resultado["result"]
        estado["interrupt"] = None
        estado["error"] = None
    await _set(job_id, estado)


async def _empezar(job_id: str) -> None:
    estado = await leer(job_id)
    if estado is None:
        return
    estado["status"] = "RUNNING"
    estado["error"] = None
    await _set(job_id, estado)
    # The graph calls the LLM synchronously. The thread keeps the event loop free.
    resultado = await _en_hilo(
        "orquestador",
        job_id,
        graph_mod.correr,
        graph_mod.estado_inicial(estado["query"]),
        _config(job_id),
        RECURSION_LIMIT,
    )
    await _cerrar(job_id, resultado)


async def _reanudar(job_id: str, approved: bool) -> None:
    estado = await leer(job_id)
    if estado is None:
        return
    estado["status"] = "RUNNING"
    estado["interrupt"] = None
    await _set(job_id, estado)
    resultado = await _en_hilo(
        "orquestador",
        job_id,
        graph_mod.reanudar,
        {"approved": approved},
        _config(job_id),
        RECURSION_LIMIT,
    )
    await _cerrar(job_id, resultado)


async def _marcar_fallo(job_id: str, exc: BaseException) -> None:
    # A background failure must land on FAILED. Otherwise the client polls forever.
    estado = await leer(job_id)
    if estado is None:
        estado = {
            "job_id": job_id,
            "status": "FAILED",
            "query": "",
            "result": None,
            "error": str(exc),
            "interrupt": None,
        }
    else:
        estado["status"] = "FAILED"
        estado["error"] = str(exc)
        estado["interrupt"] = None
    await _set(job_id, estado)


async def procesar(raw: str) -> None:
    payload = json.loads(raw)
    job_id = str(payload["job_id"])
    try:
        if payload.get("action") == "resume":
            await _reanudar(job_id, bool(payload.get("approved")))
        else:
            await _empezar(job_id)
    except Exception as exc:
        await _marcar_fallo(job_id, exc)


async def task_worker() -> None:
    """blpop forever. Each job is its own task, so one failure does not stop the loop."""
    print("Worker started")
    while True:
        item = await _cliente().blpop(QUEUE_NAME, timeout=0)
        if item is None:
            continue
        _, raw = item
        asyncio.create_task(_seguro(raw))


async def _seguro(raw: str) -> None:
    try:
        await procesar(raw)
    except Exception as exc:
        try:
            payload = json.loads(raw)
            await _marcar_fallo(str(payload["job_id"]), exc)
        except Exception:
            print(f"FAILED: {exc}")
