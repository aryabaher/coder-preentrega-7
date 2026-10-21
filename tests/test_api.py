import asyncio
import json
import threading
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import worker
from app.main import (
    ApprovalRequest,
    TaskRequest,
    TaskResponse,
    TaskStatus,
    approve_task,
    app,
    create_task,
    get_task,
)
from app.worker import STATUS_PREFIX, encolar, task_worker
from tests.redis_falso import RedisFalso


async def _ceder() -> None:
    await asyncio.sleep(0)


async def _hasta(redis: RedisFalso, job_id: str, estado: str) -> dict:
    clave = f"{STATUS_PREFIX}{job_id}"
    ultimo = None
    for _ in range(80):
        ultimo = await redis.get(clave)
        if ultimo is not None and json.loads(ultimo)["status"] == estado:
            return json.loads(ultimo)
        await asyncio.sleep(0.01)
    raise AssertionError(ultimo)


@pytest.fixture
def redis() -> RedisFalso:
    falso = RedisFalso()
    anterior = worker.redis_client
    worker.redis_client = falso
    yield falso
    worker.redis_client = anterior


async def test_create_task_responde_pending_y_encola(redis: RedisFalso) -> None:
    respuesta = await create_task(TaskRequest(query="hola"))
    assert isinstance(respuesta, TaskResponse)
    assert respuesta.status == "PENDING"
    guardado = json.loads(await redis.get(f"{STATUS_PREFIX}{respuesta.job_id}"))
    assert guardado["status"] == "PENDING"
    assert guardado["query"] == "hola"
    assert guardado["result"] is None
    nombres = [nombre for nombre, _ in redis.llamadas]
    assert nombres.index("set") < nombres.index("rpush")


async def test_la_respuesta_no_espera_al_grafo(redis: RedisFalso, monkeypatch: pytest.MonkeyPatch) -> None:
    seguir = threading.Event()

    def lento(entrada: object, config: dict, recursion_limit: int = 25) -> dict:
        assert recursion_limit == 25
        assert seguir.wait(timeout=5)
        return {"paused": False, "interrupt": None, "result": "ok"}

    monkeypatch.setattr("app.graph.correr", lento)
    tarea = asyncio.create_task(task_worker())
    await _ceder()
    try:
        inicio = time.perf_counter()
        respuesta = await create_task(TaskRequest(query="hola"))
        assert time.perf_counter() - inicio < 0.5
        assert respuesta == TaskResponse(job_id=respuesta.job_id, status="PENDING")
        assert not seguir.is_set()
        seguir.set()
        final = await _hasta(redis, respuesta.job_id, "DONE")
        assert final["result"] == "ok"
        assert isinstance(await get_task(respuesta.job_id), TaskStatus)
    finally:
        tarea.cancel()
        with pytest.raises(asyncio.CancelledError):
            await tarea


async def test_cinco_peticiones_concurrentes(redis: RedisFalso, monkeypatch: pytest.MonkeyPatch) -> None:
    seguir = threading.Event()
    adentro = threading.Event()
    en_vuelo = 0
    tope = 0
    candado = threading.Lock()

    def lento(entrada: object, config: dict, recursion_limit: int = 25) -> dict:
        nonlocal en_vuelo, tope
        with candado:
            en_vuelo += 1
            tope = max(tope, en_vuelo)
            if en_vuelo == 5:
                adentro.set()
        assert seguir.wait(timeout=5)
        with candado:
            en_vuelo -= 1
        return {"paused": False, "interrupt": None, "result": "ok"}

    monkeypatch.setattr("app.graph.correr", lento)
    tarea = asyncio.create_task(task_worker())
    await _ceder()
    try:
        inicio = time.perf_counter()
        respuestas = await asyncio.gather(*[create_task(TaskRequest(query=f"consulta {n}")) for n in range(5)])
        assert time.perf_counter() - inicio < 0.5
        assert len({r.job_id for r in respuestas}) == 5
        assert {r.status for r in respuestas} == {"PENDING"}
        for _ in range(200):
            if adentro.is_set():
                break
            await asyncio.sleep(0.01)
        assert adentro.is_set()
        assert tope == 5
        seguir.set()
        for respuesta in respuestas:
            final = await _hasta(redis, respuesta.job_id, "DONE")
            assert final["result"] == "ok"
    finally:
        seguir.set()
        tarea.cancel()
        with pytest.raises(asyncio.CancelledError):
            await tarea


async def test_el_worker_guarda_failed_y_sigue(redis: RedisFalso, monkeypatch: pytest.MonkeyPatch) -> None:
    llamadas = {"n": 0}

    def flaky(entrada: object, config: dict, recursion_limit: int = 25) -> dict:
        llamadas["n"] += 1
        if llamadas["n"] == 1:
            raise RuntimeError("corte de prueba")
        return {"paused": False, "interrupt": None, "result": "ok"}

    monkeypatch.setattr("app.graph.correr", flaky)
    tarea = asyncio.create_task(task_worker())
    await _ceder()
    try:
        primera = await create_task(TaskRequest(query="una"))
        fallo = await _hasta(redis, primera.job_id, "FAILED")
        assert fallo["error"] == "corte de prueba"
        segunda = await create_task(TaskRequest(query="otra"))
        bien = await _hasta(redis, segunda.job_id, "DONE")
        assert bien["result"] == "ok"
    finally:
        tarea.cancel()
        with pytest.raises(asyncio.CancelledError):
            await tarea


async def test_401_en_segundo_plano_queda_failed(redis: RedisFalso, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(entrada: object, config: dict, recursion_limit: int = 25) -> dict:
        raise RuntimeError("401/key: falta OPENAI_API_KEY en las variables de entorno. No se reintenta.")

    monkeypatch.setattr("app.graph.correr", boom)
    tarea = asyncio.create_task(task_worker())
    await _ceder()
    try:
        respuesta = await create_task(TaskRequest(query="hola"))
        fallo = await _hasta(redis, respuesta.job_id, "FAILED")
        assert fallo["error"].startswith("401/key:")
    finally:
        tarea.cancel()
        with pytest.raises(asyncio.CancelledError):
            await tarea


async def test_approve_encola_la_reanudacion(redis: RedisFalso, monkeypatch: pytest.MonkeyPatch) -> None:
    seguir = threading.Event()

    def pausa(entrada: object, config: dict, recursion_limit: int = 25) -> dict:
        return {
            "paused": True,
            "interrupt": {"tool": "registrar_ajuste_nomina", "reason": "side-effect tool registrar_ajuste_nomina"},
            "result": None,
        }

    def sigue(decision: dict, config: dict, recursion_limit: int = 25) -> dict:
        assert decision == {"approved": True}
        assert recursion_limit == 25
        assert seguir.wait(timeout=5)
        return {"paused": False, "interrupt": None, "result": "asentado"}

    monkeypatch.setattr("app.graph.correr", pausa)
    monkeypatch.setattr("app.graph.reanudar", sigue)
    tarea = asyncio.create_task(task_worker())
    await _ceder()
    try:
        creado = await create_task(TaskRequest(query="Registrá el ajuste de nómina"))
        espera = await _hasta(redis, creado.job_id, "AWAITING_APPROVAL")
        assert espera["interrupt"]["tool"] == "registrar_ajuste_nomina"
        inicio = time.perf_counter()
        respuesta = await approve_task(creado.job_id, ApprovalRequest(approved=True))
        assert time.perf_counter() - inicio < 0.5
        assert respuesta == TaskResponse(job_id=creado.job_id, status="RUNNING")
        assert not seguir.is_set()
        seguir.set()
        final = await _hasta(redis, creado.job_id, "DONE")
        assert final["result"] == "asentado"
    finally:
        seguir.set()
        tarea.cancel()
        with pytest.raises(asyncio.CancelledError):
            await tarea


async def test_approve_sin_pausa_responde_409(redis: RedisFalso) -> None:
    job_id = await encolar("hola")
    with pytest.raises(HTTPException) as exc:
        await approve_task(job_id, ApprovalRequest(approved=True))
    assert exc.value.status_code == 409
    assert exc.value.detail == "Job is not awaiting approval"


async def test_job_inexistente_responde_404(redis: RedisFalso) -> None:
    with pytest.raises(HTTPException) as exc:
        await get_task("inventado")
    assert exc.value.status_code == 404
    assert exc.value.detail == "Job not found"


def test_http_job_inexistente_y_body_invalido(redis: RedisFalso) -> None:
    with TestClient(app) as cliente:
        faltante = cliente.get("/tasks/inventado")
        assert faltante.status_code == 404
        assert faltante.json() == {"detail": "Job not found"}
        sin_query = cliente.post("/tasks", json={})
        assert sin_query.status_code == 422
        assert sin_query.json()["detail"][0]["loc"] == ["body", "query"]
        espacios = cliente.post("/tasks", json={"query": "   "})
        assert espacios.status_code == 422
        assert espacios.json()["detail"][0]["msg"] == "String should have at least 1 character"
        nulo = cliente.post("/tasks", json={"query": None})
        assert nulo.status_code == 422


def test_redis_apagado_no_encola() -> None:
    class Roto:
        async def set(self, *args: object, **kwargs: object) -> None:
            raise ConnectionError(
                "Error 22 connecting to localhost:6379. El equipo remoto rechazó la conexión de red."
            )

        async def rpush(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("no debe encolar")

        async def blpop(self, *args: object, **kwargs: object) -> None:
            await asyncio.Event().wait()

    anterior = worker.redis_client
    worker.redis_client = Roto()
    try:
        with TestClient(app) as cliente:
            respuesta = cliente.post("/tasks", json={"query": "hola"})
        assert respuesta.status_code == 503
        assert respuesta.json()["detail"] == "Redis is unavailable"
    finally:
        worker.redis_client = anterior


def test_recursion_limit_llega_al_invoke(monkeypatch: pytest.MonkeyPatch) -> None:
    visto: dict[str, object] = {}

    class DobleGrafo:
        def invoke(self, entrada: object, config: dict) -> dict:
            visto["config"] = config
            return {"messages": []}

        def get_state(self, config: dict) -> object:
            class Snap:
                interrupts: tuple = ()
                tasks: tuple = ()
                values: dict = {"messages": []}

            return Snap()

    from app import graph

    graph.usar(DobleGrafo())
    graph.correr(graph.estado_inicial("hola"), {"configurable": {"thread_id": "limite"}}, 17)
    assert visto["config"]["recursion_limit"] == 17  # type: ignore[index]
