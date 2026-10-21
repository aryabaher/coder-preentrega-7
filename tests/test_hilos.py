import asyncio
import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.graph import get_graph, run_graph, thread_history
from app.main import TaskRequest, app, create_task, get_thread
from app.worker import STATUS_PREFIX, task_worker
from tests.conftest import EspecialistaFalso, LLMFalso
from tests.redis_falso import RedisFalso

PRIMERA = "¿Cuántos días de vacaciones corresponden con 5 años de antigüedad?"
SEGUNDA = "¿Y cuántos acumula en total en esos 5 años?"


def _config(hilo: str) -> dict:
    return {"configurable": {"thread_id": hilo}}


def test_dos_turnos_en_el_mismo_hilo(dobles: tuple[LLMFalso, EspecialistaFalso, EspecialistaFalso]) -> None:
    llm, investigador, _ = dobles
    run_graph(PRIMERA, _config("conversacion"))
    run_graph(SEGUNDA, _config("conversacion"))
    valores = get_graph().get_state(_config("conversacion")).values
    humanos = [m.content for m in valores["messages"] if m.type == "human"]
    finales = [m for m in valores["messages"] if m.type == "ai" and m.name == "supervisor"]
    assert humanos == [PRIMERA, SEGUNDA]
    assert len(finales) == 2
    assert "Conversación previa del hilo:" in llm.supervisor.sistemas[-1]
    assert PRIMERA in str(llm.prompts[-1])
    assert "Pregunta actual: " + SEGUNDA in str(llm.prompts[-1])
    assert PRIMERA in investigador.entradas[-1]["messages"][0].content
    assert SEGUNDA in investigador.entradas[-1]["messages"][0].content


def test_el_segundo_turno_reinicia_lo_del_turno() -> None:
    run_graph(PRIMERA, _config("reinicio"))
    run_graph(SEGUNDA, _config("reinicio"))
    valores = get_graph().get_state(_config("reinicio")).values
    assert [c["agente"] for c in valores["contribuciones"]] == ["investigador", "analista"]
    assert valores["pasos"] == 2
    assert valores["approved"] is None


def test_los_hilos_estan_aislados() -> None:
    run_graph(PRIMERA, _config("hilo-a"))
    run_graph(SEGUNDA, _config("hilo-b"))
    a = thread_history("hilo-a")
    b = thread_history("hilo-b")
    assert a is not None and b is not None
    assert [m["content"] for m in a if m["role"] == "human"] == [PRIMERA]
    assert [m["content"] for m in b if m["role"] == "human"] == [SEGUNDA]
    assert thread_history("no-existe") is None


async def _hasta(redis: RedisFalso, job_id: str, estado: str) -> dict:
    for _ in range(300):
        raw = await redis.get(f"{STATUS_PREFIX}{job_id}")
        if raw is not None and json.loads(raw)["status"] == estado:
            return json.loads(raw)
        await asyncio.sleep(0.01)
    raise AssertionError(raw)


async def test_la_api_continua_el_hilo_y_lo_devuelve(redis: RedisFalso) -> None:
    tarea = asyncio.create_task(task_worker())
    await asyncio.sleep(0)
    try:
        primera = await create_task(TaskRequest(query=PRIMERA))
        await _hasta(redis, primera.job_id, "DONE")
        segunda = await create_task(TaskRequest(query=SEGUNDA, thread_id=primera.thread_id))
        assert segunda.thread_id == primera.thread_id
        assert segunda.job_id != primera.job_id
        await _hasta(redis, segunda.job_id, "DONE")
        historial = await get_thread(primera.thread_id)
    finally:
        tarea.cancel()
        with pytest.raises(asyncio.CancelledError):
            await tarea
    humanos = [m.content for m in historial.messages if m.role == "human"]
    assert humanos == [PRIMERA, SEGUNDA]
    assert [m.name for m in historial.messages if m.role == "ai"].count("supervisor") == 2


async def test_un_hilo_ocupado_responde_409(redis: RedisFalso) -> None:
    primera = await create_task(TaskRequest(query=PRIMERA, thread_id="ocupado"))
    with pytest.raises(HTTPException) as exc:
        await create_task(TaskRequest(query=SEGUNDA, thread_id="ocupado"))
    assert exc.value.status_code == 409
    assert exc.value.detail == "Thread is busy"
    assert primera.status == "PENDING"


async def test_hilo_inexistente_responde_404(redis: RedisFalso) -> None:
    with pytest.raises(HTTPException) as exc:
        await get_thread("no-existe")
    assert exc.value.status_code == 404
    assert exc.value.detail == "Thread not found"


def test_http_threads(redis: RedisFalso) -> None:
    run_graph(PRIMERA, _config("http"))
    with TestClient(app) as cliente:
        bien = cliente.get("/threads/http")
        mal = cliente.get("/threads/no-existe")
    assert bien.status_code == 200
    assert bien.json()["thread_id"] == "http"
    assert bien.json()["messages"][0] == {"role": "human", "name": None, "content": PRIMERA}
    assert mal.status_code == 404
    assert mal.json() == {"detail": "Thread not found"}
