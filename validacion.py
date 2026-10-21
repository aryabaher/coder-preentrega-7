"""Demo sin clave y sin Redis: el doble del modelo está acá, no en la API."""

from requisitos import exigir_python_312

exigir_python_312()

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["TRAZAS_OFFLINE"] = "1"
os.environ.pop("PINECONE_API_KEY", None)

import redis.exceptions
from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

import modelo
from agents import analyst_agent, research_agent
from app import graph, worker
from app.graph import DecisionSupervisor, compile_graph, resume_graph, run_graph, set_graph, thread_history
from app.hitl import ApprovalRequest, approve_task, side_effects
from app.main import TaskRequest, app, create_task, get_task, get_thread
from app.observability import init_observability
from errors import ClaveAPIError, CuotaError, RedError
from tests.redis_falso import RedisFalso

init_observability()

RAIZ = Path(__file__).resolve().parent
PREGUNTA = (
    "Un empleado va a cumplir 12 años en TechCorp. Tuvo 14 días de vacaciones los primeros "
    "5 años, 21 los siguientes 5, y ahora tiene derecho a 28. ¿Cuántos días de vacaciones "
    "acumuló en total en esos 12 años?"
)
PREGUNTA_CRITICA = PREGUNTA + " Registrá el ajuste de nómina con register_payroll_adjustment."
SEGUIMIENTO = "¿Y si en vez de 12 fueran 15 años?"


class SupervisorGuion:
    def invoke(self, mensajes: list[dict[str, str]], *args: object, **kwargs: object) -> DecisionSupervisor:
        sistema = mensajes[0]["content"]
        if "- investigador:" not in sistema:
            return DecisionSupervisor(next="investigador", razon="falta un dato")
        if "- analista:" not in sistema:
            return DecisionSupervisor(next="analista", razon="falta calcular")
        return DecisionSupervisor(next="FINISH", razon="listo")


class LLMFalso:
    def invoke(self, prompt: object, *args: object, **kwargs: object) -> AIMessage:
        return AIMessage(content="En 12 años acumuló 231 días de vacaciones.")

    def with_structured_output(self, schema: object, **kwargs: object) -> SupervisorGuion:
        return SupervisorGuion()


class EspecialistaFalso:
    def __init__(self, texto: str) -> None:
        self.texto = texto

    def invoke(self, entrada: dict, *args: object, **kwargs: object) -> dict:
        return {"messages": [AIMessage(content=self.texto)]}


def _exigir(condicion: bool, mensaje: str) -> None:
    if not condicion:
        raise SystemExit(mensaje)


def _git_limpio() -> None:
    listado = subprocess.run(
        ["git", "ls-files", "--", ".env", "*.sqlite"], cwd=RAIZ, capture_output=True, text=True, check=False
    )
    print("git ls-files .env *.sqlite:")
    print(listado.stdout.strip() or "(ninguno)")
    _exigir(listado.stdout.strip() == "", ".env o sqlite están versionados")
    ignorado = subprocess.run(["git", "check-ignore", "-v", ".env"], cwd=RAIZ, capture_output=True, text=True, check=False)
    print("git check-ignore .env:")
    print(ignorado.stdout.strip() or ignorado.stderr.strip())
    _exigir(ignorado.returncode == 0, ".env no está en .gitignore")


async def _esperar(job_id: str, estado: str) -> dict:
    ultimo: dict = {}
    for _ in range(300):
        ultimo = (await get_task(job_id)).model_dump()
        if ultimo["status"] == estado:
            return ultimo
        await asyncio.sleep(0.01)
    raise SystemExit(f"{job_id} no llegó a {estado}: {ultimo}")


async def _ciclo_api() -> None:
    worker.redis_client = RedisFalso()
    tarea = asyncio.create_task(worker.task_worker())
    await asyncio.sleep(0)
    try:
        primera = await create_task(TaskRequest(query=PREGUNTA))
        print("POST /tasks:", primera.model_dump())
        await _esperar(primera.job_id, "DONE")
        segunda = await create_task(TaskRequest(query=SEGUIMIENTO, thread_id=primera.thread_id))
        print("POST /tasks mismo hilo:", segunda.model_dump())
        await _esperar(segunda.job_id, "DONE")
        historial = await get_thread(primera.thread_id)
        humanos = [m.content for m in historial.messages if m.role == "human"]
        print("GET /threads humanos:", humanos)
        _exigir(humanos == [PREGUNTA, SEGUIMIENTO], "el hilo no guardó los dos turnos")
        try:
            await get_thread("no-existe")
        except HTTPException as exc:
            print("404 hilo:", exc.status_code, exc.detail)

        critica = await create_task(TaskRequest(query=PREGUNTA_CRITICA))
        pausa = await _esperar(critica.job_id, "AWAITING_APPROVAL")
        print("GET en pausa:", json.dumps(pausa["interrupt"], ensure_ascii=False)[:160])
        try:
            await create_task(TaskRequest(query="otra", thread_id=critica.thread_id))
        except HTTPException as exc:
            print("409 hilo ocupado:", exc.status_code, exc.detail)
            _exigir(exc.detail == "Thread is busy", "hilo ocupado")
        respuesta = await approve_task(critica.job_id, ApprovalRequest(approved=True))
        print("POST approve:", respuesta.model_dump())
        final = await _esperar(critica.job_id, "DONE")
        print("GET final:", final["status"], "efectos:", len(side_effects))
        _exigir(len(side_effects) == 1, "aprobar no corrió la herramienta")
        try:
            await approve_task(critica.job_id, ApprovalRequest(approved=True))
        except HTTPException as exc:
            print("409 approve:", exc.status_code, exc.detail)

        original = graph.run_graph

        def boom(*args: object) -> dict:
            raise RuntimeError("corte de prueba")

        graph.run_graph = boom  # type: ignore[assignment]
        try:
            fallido = await create_task(TaskRequest(query="hola"))
            mal = await _esperar(fallido.job_id, "FAILED")
            print("FAILED:", mal["status"], mal["error"])
        finally:
            graph.run_graph = original  # type: ignore[assignment]

        huerfano = {"job_id": "huerfano", "thread_id": "huerfano", "status": "RUNNING", "query": "q", "result": None}
        await worker.redis_client.set("status:huerfano", json.dumps(huerfano))
        print("Huérfanos recuperados:", await worker.recover_orphans())
        print("Huérfano:", (await get_task("huerfano")).error)
    finally:
        tarea.cancel()
        try:
            await tarea
        except asyncio.CancelledError:
            pass


def _http() -> None:
    worker.redis_client = RedisFalso()
    with TestClient(app) as cliente:
        print("422 query en blanco:", cliente.post("/tasks", json={"query": "   "}).text)
        print("404 job:", cliente.get("/tasks/inventado").text)
        print("404 approve:", cliente.post("/tasks/inventado/approve", json={"approved": True}).text)

    class Caido(RedisFalso):
        async def set(self, *args: object, **kwargs: object):  # type: ignore[override]
            raise redis.exceptions.ConnectionError("Error 10061 connecting to localhost:6379. Connection refused.")

    worker.redis_client = Caido()
    with TestClient(app) as cliente:
        caido = cliente.post("/tasks", json={"query": "hola"})
    print("503 Redis caído después de arrancar:", caido.status_code, caido.text)
    _exigir(caido.status_code == 503, caido.text)

    class SinRedis:
        async def ping(self) -> bool:
            raise redis.exceptions.ConnectionError("Error 10061 connecting to localhost:6379.")

    worker.redis_client = None
    original = worker.aioredis.from_url
    worker.aioredis.from_url = lambda *a, **k: SinRedis()  # type: ignore[assignment]
    try:
        asyncio.run(worker.connect())
    except worker.RedisUnavailable as exc:
        print("Redis caído al arrancar:", exc)
    finally:
        worker.aioredis.from_url = original  # type: ignore[assignment]


def _costo() -> None:
    real = graph.costo_estimado
    graph.costo_estimado = lambda state: 1.0 if state.get("contribuciones") else 0.0  # type: ignore[assignment]
    os.environ["COSTO_MAXIMO_USD"] = "0.5"
    try:
        pausa = run_graph(PREGUNTA, {"configurable": {"thread_id": "caro"}})
        print("Pausa por costo:", pausa["interrupt"])
        _exigir(pausa["interrupt"]["reason"] == "estimated cost above COSTO_MAXIMO_USD", "sin pausa por costo")
        cierre = resume_graph({"approved": False}, {"configurable": {"thread_id": "caro"}})
        print("Costo rechazado, cierra:", cierre["paused"])
    finally:
        graph.costo_estimado = real  # type: ignore[assignment]
        os.environ.pop("COSTO_MAXIMO_USD", None)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(f"Python {sys.version.split()[0]}")
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ["LLM_PROVIDER"] = "openai"
    try:
        modelo.crear_llm()
    except ClaveAPIError as exc:
        print(exc)
    try:
        raise modelo.clasificar_error(TimeoutError("timed out"))
    except RedError as exc:
        print(exc)
    try:
        cuota = Exception("rate limit")
        cuota.status_code = 429  # type: ignore[attr-defined]
        raise modelo.clasificar_error(cuota)
    except CuotaError as exc:
        print(exc)
    try:
        exigir_python_312((3, 11, 9))
    except SystemExit as exc:
        print(f"Guardia 3.11: {exc}")

    modelo.obtener_llm = lambda: LLMFalso()  # type: ignore[method-assign]
    modelo.ESPERA_INICIAL_S = 0
    research_agent.agente_investigador = EspecialistaFalso("14 días, 21 días y 28 días.")
    analyst_agent.agente_analista = EspecialistaFalso("Resultado: 231")
    side_effects.clear()
    set_graph(compile_graph(MemorySaver()))

    normal = run_graph(PREGUNTA, {"configurable": {"thread_id": "demo"}})
    print("Tarea normal:", normal)
    seguimiento = run_graph(SEGUIMIENTO, {"configurable": {"thread_id": "demo"}})
    print("Seguimiento en el mismo hilo:", seguimiento["result"])
    print("Historial del hilo demo:", [(m["role"], m["name"]) for m in thread_history("demo") or []])

    _costo()
    asyncio.run(_ciclo_api())
    _http()
    _git_limpio()
    print("validacion ok")


if __name__ == "__main__":
    main()
