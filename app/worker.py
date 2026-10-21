"""Background worker: PENDING/RUNNING/FAILED/DONE in Redis. Runs inside the API or as `python -m app.worker`."""

from requisitos import exigir_python_312

exigir_python_312()

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from redis import asyncio as aioredis

QUEUE_NAME = "tasks_queue"
STATUS_PREFIX = "status:"
THREAD_PREFIX = "thread:"
RECURSION_LIMIT = 25
DEFAULT_REDIS_URL = "redis://localhost:6379"
ACTIVE = ("PENDING", "RUNNING", "AWAITING_APPROVAL")

redis_client: Any = None
running_tasks: set[asyncio.Task[None]] = set()


class JobNotFound(Exception):
    """No Redis record for that job_id."""


class NotAwaitingApproval(Exception):
    """approve was called and the job is not paused on the human node."""


class ThreadBusy(Exception):
    """The thread already has a job in PENDING, RUNNING or AWAITING_APPROVAL."""


class RedisUnavailable(RuntimeError):
    """Redis did not answer at startup."""


def _graph() -> Any:
    from app import graph

    return graph


def redis_url() -> str:
    return (os.getenv("REDIS_URL") or DEFAULT_REDIS_URL).strip() or DEFAULT_REDIS_URL


def _client() -> Any:
    if redis_client is None:
        raise ConnectionError("Redis is unavailable")
    return redis_client


def _job_key(job_id: str) -> str:
    return f"{STATUS_PREFIX}{job_id}"


def _thread_key(thread_id: str) -> str:
    return f"{THREAD_PREFIX}{thread_id}"


async def _save(record: dict[str, Any]) -> None:
    await _client().set(_job_key(record["job_id"]), json.dumps(record))


async def _push(payload: dict[str, Any]) -> None:
    await _client().rpush(QUEUE_NAME, json.dumps(payload))


async def read_job(job_id: str) -> dict[str, Any] | None:
    raw = await _client().get(_job_key(job_id))
    if raw is None:
        return None
    return json.loads(raw)


async def enqueue_task(query: str, thread_id: str | None = None) -> dict[str, Any]:
    """Claim the thread, write PENDING, then push. The worker must not win that race."""
    job_id = str(uuid.uuid4())
    thread = thread_id or job_id
    claimed = await _client().set(_thread_key(thread), job_id, nx=True)
    if not claimed:
        raise ThreadBusy(thread)
    record = {
        "job_id": job_id,
        "thread_id": thread,
        "status": "PENDING",
        "query": query,
        "result": None,
        "error": None,
        "interrupt": None,
    }
    try:
        await _save(record)
        await _push({"action": "start", "job_id": job_id})
    except Exception:
        await _client().delete(_thread_key(thread))
        raise
    return record


async def enqueue_resume(job_id: str, approved: bool) -> None:
    record = await read_job(job_id)
    if record is None:
        raise JobNotFound(job_id)
    if record["status"] != "AWAITING_APPROVAL":
        raise NotAwaitingApproval(record["status"])
    record.update(status="RUNNING", interrupt=None, error=None)
    await _save(record)
    await _push({"action": "resume", "job_id": job_id, "approved": approved})


async def read_thread(thread_id: str) -> list[dict[str, str | None]] | None:
    """Thread history from the RedisSaver checkpoints, read off the event loop."""
    return await asyncio.to_thread(_graph().thread_history, thread_id)


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}, "tags": ["preentrega-7"]}


def _call(ctx: otel_context.Context, fn: Any, args: tuple[Any, ...]) -> Any:
    """The worker thread keeps the trace and the session that the event loop opened."""
    token = otel_context.attach(ctx)
    try:
        return fn(*args)
    finally:
        otel_context.detach(token)


async def _run_traced(record: dict[str, Any], query: str, fn: Any, *args: Any) -> dict[str, Any]:
    """One Phoenix trace per execution: span `orquestador`, session = thread_id, a status event per change."""
    from openinference.instrumentation import using_session

    thread_id = record["thread_id"]
    with using_session(thread_id):
        with trace.get_tracer("preentrega-7").start_as_current_span("orquestador") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("session.id", thread_id)
            span.set_attribute("job_id", record["job_id"])
            span.set_attribute("input.value", query)
            span.add_event("status", {"status": "RUNNING"})
            try:
                result = await asyncio.to_thread(_call, otel_context.get_current(), fn, args)
            except Exception as exc:
                span.add_event("status", {"status": "FAILED", "error": str(exc)})
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
            status = "AWAITING_APPROVAL" if result["paused"] else "DONE"
            span.add_event("status", {"status": status})
            span.set_attribute("output.value", result["result"] or json.dumps(result["interrupt"]))
            return result


async def _release_thread(record: dict[str, Any]) -> None:
    thread_id = record.get("thread_id")
    if thread_id and await _client().get(_thread_key(thread_id)) == record["job_id"]:
        await _client().delete(_thread_key(thread_id))


async def _finish(record: dict[str, Any], result: dict[str, Any]) -> None:
    if result["paused"]:
        record.update(status="AWAITING_APPROVAL", interrupt=result["interrupt"], result=None, error=None)
    else:
        record.update(status="DONE", result=result["result"], interrupt=None, error=None)
        await _release_thread(record)
    await _save(record)


async def _start(job_id: str) -> None:
    record = await read_job(job_id)
    if record is None:
        return
    record.update(status="RUNNING", error=None)
    await _save(record)
    # The graph calls the LLM synchronously. The thread keeps the event loop free.
    result = await _run_traced(
        record,
        record["query"],
        _graph().run_graph,
        record["query"],
        _config(record["thread_id"]),
        RECURSION_LIMIT,
    )
    await _finish(record, result)


async def _resume(job_id: str, approved: bool) -> None:
    record = await read_job(job_id)
    if record is None:
        return
    record.update(status="RUNNING", interrupt=None)
    await _save(record)
    result = await _run_traced(
        record,
        f"approved={approved}",
        _graph().resume_graph,
        {"approved": approved},
        _config(record["thread_id"]),
        RECURSION_LIMIT,
    )
    await _finish(record, result)


async def _mark_failed(job_id: str, exc: BaseException) -> None:
    # A background failure must land on FAILED. Otherwise the client polls forever.
    record = await read_job(job_id) or {
        "job_id": job_id,
        "thread_id": job_id,
        "query": "",
        "result": None,
    }
    record.update(status="FAILED", error=str(exc), interrupt=None)
    await _save(record)
    await _release_thread(record)


async def process(raw: str) -> None:
    payload = json.loads(raw)
    job_id = str(payload["job_id"])
    try:
        if payload.get("action") == "resume":
            await _resume(job_id, bool(payload.get("approved")))
        else:
            await _start(job_id)
    except Exception as exc:
        await _mark_failed(job_id, exc)


async def _safe(raw: str) -> None:
    try:
        await process(raw)
    except Exception as exc:
        try:
            await _mark_failed(str(json.loads(raw)["job_id"]), exc)
        except Exception:
            print(f"FAILED: {exc}")


async def task_worker() -> None:
    """blpop forever. Each job is its own task, so one failure does not stop the loop."""
    print("Worker started")
    while True:
        item = await _client().blpop(QUEUE_NAME, timeout=0)
        if item is None:
            continue
        _, raw = item
        task = asyncio.create_task(_safe(raw))
        running_tasks.add(task)
        task.add_done_callback(running_tasks.discard)


async def recover_orphans() -> list[str]:
    """RUNNING jobs left by a dead worker go to FAILED. AWAITING_APPROVAL survives: its checkpoint is in Redis."""
    failed: list[str] = []
    async for key in _client().scan_iter(match=f"{STATUS_PREFIX}*"):
        raw = await _client().get(key)
        if raw is None:
            continue
        record = json.loads(raw)
        if record.get("status") == "RUNNING":
            record.update(status="FAILED", error="worker restarted before the job finished", interrupt=None)
            await _save(record)
            await _release_thread(record)
            failed.append(record["job_id"])
    return failed


async def connect() -> None:
    """Redis client and the graph compiled with RedisSaver. Fails with a readable message if Redis is down."""
    global redis_client
    if redis_client is None:
        client = aioredis.from_url(redis_url(), decode_responses=True)
        try:
            await client.ping()
        except Exception as exc:
            raise RedisUnavailable(
                f"Redis is unavailable at {redis_url()}. "
                "Start it with: docker run -d --name redis-preentrega-7 -p 6379:6379 redis:8"
            ) from exc
        redis_client = client
    graph = _graph()
    if not graph.graph_ready():
        graph.set_graph(await asyncio.to_thread(graph.compile_graph))


async def run_forever() -> None:
    await connect()
    recovered = await recover_orphans()
    if recovered:
        print(f"Recovered orphan jobs as FAILED: {', '.join(recovered)}")
    await task_worker()


def main() -> None:
    from dotenv import load_dotenv

    from app.observability import init_observability

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    init_observability()
    asyncio.run(run_forever())


if __name__ == "__main__":
    main()
