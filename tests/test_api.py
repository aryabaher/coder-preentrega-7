import asyncio
import json
import threading
import time

import pytest
import redis.exceptions
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import worker
from app.hitl import ApprovalRequest, ApprovalResponse, approve_task
from app.main import TaskRequest, TaskResponse, TaskStatus, app, create_task, get_task
from app.worker import STATUS_PREFIX, enqueue_task, task_worker
from tests.redis_falso import RedisFalso


async def _ceder() -> None:
    await asyncio.sleep(0)


async def _hasta(redis: RedisFalso, job_id: str, estado: str) -> dict:
    clave = f"{STATUS_PREFIX}{job_id}"
    ultimo = None
    for _ in range(200):
        ultimo = await redis.get(clave)
        if ultimo is not None and json.loads(ultimo)["status"] == estado:
            return json.loads(ultimo)
        await asyncio.sleep(0.01)
    raise AssertionError(ultimo)


async def _con_worker(coro) -> None:
    tarea = asyncio.create_task(task_worker())
    await _ceder()
    try:
        await coro
    finally:
        tarea.cancel()
        with pytest.raises(asyncio.CancelledError):
            await tarea


async def test_create_task_responde_pending_y_encola(redis: RedisFalso) -> None:
    respuesta = await create_task(TaskRequest(query="hola"))
    assert isinstance(respuesta, TaskResponse)
    assert respuesta.status == "PENDING"
    assert respuesta.thread_id == respuesta.job_id
    guardado = json.loads(await redis.get(f"{STATUS_PREFIX}{respuesta.job_id}"))
    assert guardado["status"] == "PENDING"
    assert guardado["query"] == "hola"
    assert guardado["result"] is None
    nombres = [nombre for nombre, _ in redis.llamadas]
    assert nombres.index("set") < nombres.index("rpush")


async def test_la_respuesta_no_espera_al_grafo(redis: RedisFalso, monkeypatch: pytest.MonkeyPatch) -> None:
    seguir = threading.Event()

    def lento(pregunta: str, config: dict, recursion_limit: int = 25) -> dict:
        assert recursion_limit == 25
        assert config["configurable"]["thread_id"]
        assert seguir.wait(timeout=5)
        return {"paused": False, "interrupt": None, "result": "ok"}

    monkeypatch.setattr("app.graph.run_graph", lento)

    async def caso() -> None:
        inicio = time.perf_counter()
        respuesta = await create_task(TaskRequest(query="hola"))
        assert time.perf_counter() - inicio < 0.5
        assert respuesta.status == "PENDING"
        assert not seguir.is_set()
        seguir.set()
        final = await _hasta(redis, respuesta.job_id, "DONE")
        assert final["result"] == "ok"
        assert isinstance(await get_task(respuesta.job_id), TaskStatus)

    try:
        await _con_worker(caso())
    finally:
        seguir.set()


async def test_cinco_peticiones_concurrentes(redis: RedisFalso, monkeypatch: pytest.MonkeyPatch) -> None:
    seguir = threading.Event()
    adentro = threading.Event()
    en_vuelo = 0
    tope = 0
    candado = threading.Lock()

    def lento(pregunta: str, config: dict, recursion_limit: int = 25) -> dict:
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

    monkeypatch.setattr("app.graph.run_graph", lento)

    async def caso() -> None:
        inicio = time.perf_counter()
        respuestas = await asyncio.gather(*[create_task(TaskRequest(query=f"consulta {n}")) for n in range(5)])
        assert time.perf_counter() - inicio < 0.5
        assert len({r.job_id for r in respuestas}) == 5
        assert {r.status for r in respuestas} == {"PENDING"}
        for _ in range(200):
            if adentro.is_set():
                break
            await asyncio.sleep(0.01)
        assert adentro.is_set() and tope == 5
        seguir.set()
        for respuesta in respuestas:
            assert (await _hasta(redis, respuesta.job_id, "DONE"))["result"] == "ok"

    try:
        await _con_worker(caso())
    finally:
        seguir.set()


async def test_el_worker_guarda_failed_y_sigue(redis: RedisFalso, monkeypatch: pytest.MonkeyPatch) -> None:
    llamadas = {"n": 0}

    def flaky(pregunta: str, config: dict, recursion_limit: int = 25) -> dict:
        llamadas["n"] += 1
        if llamadas["n"] == 1:
            raise RuntimeError("corte de prueba")
        return {"paused": False, "interrupt": None, "result": "ok"}

    monkeypatch.setattr("app.graph.run_graph", flaky)

    async def caso() -> None:
        primera = await create_task(TaskRequest(query="una"))
        fallo = await _hasta(redis, primera.job_id, "FAILED")
        assert fallo["error"] == "corte de prueba"
        assert await redis.get(f"thread:{primera.thread_id}") is None
        segunda = await create_task(TaskRequest(query="otra"))
        assert (await _hasta(redis, segunda.job_id, "DONE"))["result"] == "ok"

    await _con_worker(caso())


async def test_401_en_segundo_plano_queda_failed(redis: RedisFalso, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(pregunta: str, config: dict, recursion_limit: int = 25) -> dict:
        raise RuntimeError("401/key: falta OPENAI_API_KEY en las variables de entorno. No se reintenta.")

    monkeypatch.setattr("app.graph.run_graph", boom)

    async def caso() -> None:
        respuesta = await create_task(TaskRequest(query="hola"))
        fallo = await _hasta(redis, respuesta.job_id, "FAILED")
        assert fallo["error"].startswith("401/key:")

    await _con_worker(caso())


async def test_approve_encola_la_reanudacion(redis: RedisFalso, monkeypatch: pytest.MonkeyPatch) -> None:
    seguir = threading.Event()

    def pausa(pregunta: str, config: dict, recursion_limit: int = 25) -> dict:
        return {
            "paused": True,
            "interrupt": {"tool": "register_payroll_adjustment", "reason": "side-effect tool register_payroll_adjustment"},
            "result": None,
        }

    def sigue(decision: dict, config: dict, recursion_limit: int = 25) -> dict:
        assert decision == {"approved": True}
        assert seguir.wait(timeout=5)
        return {"paused": False, "interrupt": None, "result": "asentado"}

    monkeypatch.setattr("app.graph.run_graph", pausa)
    monkeypatch.setattr("app.graph.resume_graph", sigue)

    async def caso() -> None:
        creado = await create_task(TaskRequest(query="Registrá el ajuste de nómina"))
        espera = await _hasta(redis, creado.job_id, "AWAITING_APPROVAL")
        assert espera["interrupt"]["tool"] == "register_payroll_adjustment"
        assert await redis.get(f"thread:{creado.thread_id}") == creado.job_id
        inicio = time.perf_counter()
        respuesta = await approve_task(creado.job_id, ApprovalRequest(approved=True))
        assert time.perf_counter() - inicio < 0.5
        assert respuesta == ApprovalResponse(job_id=creado.job_id, status="RUNNING")
        seguir.set()
        final = await _hasta(redis, creado.job_id, "DONE")
        assert final["result"] == "asentado"
        assert await redis.get(f"thread:{creado.thread_id}") is None

    try:
        await _con_worker(caso())
    finally:
        seguir.set()


async def test_approve_sin_pausa_responde_409(redis: RedisFalso) -> None:
    record = await enqueue_task("hola")
    with pytest.raises(HTTPException) as exc:
        await approve_task(record["job_id"], ApprovalRequest(approved=True))
    assert exc.value.status_code == 409
    assert exc.value.detail == "Job is not awaiting approval"


async def test_approve_de_un_job_inexistente_responde_404(redis: RedisFalso) -> None:
    with pytest.raises(HTTPException) as exc:
        await approve_task("inventado", ApprovalRequest(approved=True))
    assert exc.value.status_code == 404
    assert exc.value.detail == "Job not found"


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
        assert cliente.post("/tasks", json={"query": None}).status_code == 422
        aprobar = cliente.post("/tasks/inventado/approve", json={"approved": True})
        assert aprobar.status_code == 404
        assert aprobar.json() == {"detail": "Job not found"}


def test_las_respuestas_json_declaran_utf8(redis: RedisFalso) -> None:
    pregunta = "¿Cuántos días de vacaciones corresponden con 5 años de antigüedad?"
    with TestClient(app) as cliente:
        creado = cliente.post("/tasks", json={"query": pregunta})
        leido = cliente.get(f"/tasks/{creado.json()['job_id']}")
        faltante = cliente.get("/tasks/inventado")
        invalido = cliente.post("/tasks", json={})
    for respuesta in (creado, leido, faltante, invalido):
        assert respuesta.headers["content-type"] == "application/json; charset=utf-8"
    assert leido.content.decode("utf-8").count("¿Cuántos días") == 1
    assert leido.json()["query"] == pregunta


class _RedisQueSeCae(RedisFalso):
    """Responde al arrancar y después se cae, como un Redis que se apaga con la API levantada."""

    caido = False

    async def set(self, *args: object, **kwargs: object):  # type: ignore[override]
        if self.caido:
            raise redis.exceptions.ConnectionError("Error 10061 connecting to localhost:6379. Connection refused.")
        return await super().set(*args, **kwargs)  # type: ignore[arg-type]

    async def get(self, clave: str):  # type: ignore[override]
        if self.caido:
            raise redis.exceptions.ConnectionError("Error 10061 connecting to localhost:6379. Connection refused.")
        return await super().get(clave)


def test_redis_caido_despues_de_arrancar_es_503() -> None:
    caido = _RedisQueSeCae()
    anterior = worker.redis_client
    worker.redis_client = caido
    try:
        with TestClient(app) as cliente:
            caido.caido = True
            post = cliente.post("/tasks", json={"query": "hola"})
            get = cliente.get("/tasks/algo")
            aprobar = cliente.post("/tasks/algo/approve", json={"approved": True})
        assert post.status_code == get.status_code == aprobar.status_code == 503
        assert post.json() == {"detail": "Redis is unavailable"}
        assert caido.encolados == []
    finally:
        worker.redis_client = anterior


def test_recursion_limit_llega_al_invoke() -> None:
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

    graph.set_graph(DobleGrafo())
    graph.run_graph("hola", {"configurable": {"thread_id": "limite"}}, 17)
    assert visto["config"]["recursion_limit"] == 17  # type: ignore[index]
