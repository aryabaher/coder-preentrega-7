"""Demo sin clave y sin Redis: el doble del modelo está acá, no en la API."""

from requisitos import exigir_python_312

exigir_python_312()

import asyncio
import os
import subprocess
import sys
from pathlib import Path

os.environ["TRAZAS_OFFLINE"] = "1"
os.environ.pop("PINECONE_API_KEY", None)

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

import modelo
from agents import analyst_agent, research_agent
from app.graph import DecisionSupervisor, compilar, correr, estado_inicial, reanudar, usar
from errors import ClaveAPIError, CuotaError, RedError
from app.hitl import efectos
from app.observability import iniciar_observabilidad

iniciar_observabilidad()

RAIZ = Path(__file__).resolve().parent
PREGUNTA = (
    "Un empleado va a cumplir 12 años en TechCorp. Tuvo 14 días de vacaciones los primeros "
    "5 años, 21 los siguientes 5, y ahora tiene derecho a 28. ¿Cuántos días de vacaciones "
    "acumuló en total en esos 12 años?"
)
PREGUNTA_CRITICA = PREGUNTA + " Registrá el ajuste de nómina con registrar_ajuste_nomina."


class SupervisorGuion:
    def __init__(self) -> None:
        self.llamadas = 0

    def invoke(self, mensajes: list[dict[str, str]], *args: object, **kwargs: object) -> DecisionSupervisor:
        self.llamadas += 1
        sistema = mensajes[0]["content"]
        if "- investigador:" not in sistema:
            return DecisionSupervisor(next="investigador", razon="falta un dato")
        if "- analista:" not in sistema:
            return DecisionSupervisor(next="analista", razon="falta calcular")
        return DecisionSupervisor(next="FINISH", razon="listo")


class LLMFalso:
    def __init__(self) -> None:
        self.supervisor = SupervisorGuion()

    def invoke(self, prompt: object, *args: object, **kwargs: object) -> AIMessage:
        return AIMessage(content="En 12 años acumuló 231 días de vacaciones.")

    def with_structured_output(self, schema: object, **kwargs: object) -> SupervisorGuion:
        return self.supervisor


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
        ["git", "ls-files", "--", ".env", "*.sqlite"],
        cwd=RAIZ,
        capture_output=True,
        text=True,
        check=False,
    )
    print("git ls-files .env *.sqlite:")
    print(listado.stdout.strip() or "(ninguno)")
    _exigir(listado.stdout.strip() == "", ".env o sqlite están versionados")
    ignorado = subprocess.run(
        ["git", "check-ignore", "-v", ".env"],
        cwd=RAIZ,
        capture_output=True,
        text=True,
        check=False,
    )
    print("git check-ignore .env:")
    print(ignorado.stdout.strip() or ignorado.stderr.strip())
    _exigir(ignorado.returncode == 0, ".env no está en .gitignore")


async def _ciclo_http() -> None:
    from fastapi.testclient import TestClient

    from app import worker
    from app.main import app
    from tests.redis_falso import RedisFalso

    redis = RedisFalso()
    worker.redis_client = redis
    with TestClient(app) as cliente:
        vacio = cliente.post("/tasks", json={"query": "   "})
        print("422 query en blanco:")
        print(vacio.text)
        _exigir(vacio.status_code == 422, vacio.text)
        faltante = cliente.get("/tasks/inventado")
        print("404:")
        print(faltante.text)
        _exigir(faltante.json()["detail"] == "Job not found", faltante.text)

    class Roto:
        async def set(self, *args: object, **kwargs: object) -> None:
            raise ConnectionError(
                "Error 22 connecting to localhost:6379. El equipo remoto rechazó la conexión de red."
            )

        async def rpush(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("no debe encolar")

        async def blpop(self, *args: object, **kwargs: object) -> None:
            await asyncio.Event().wait()

    worker.redis_client = Roto()
    with TestClient(app) as cliente:
        caido = cliente.post("/tasks", json={"query": "hola"})
        print("503 Redis apagado:")
        print(caido.text)
        _exigir(caido.status_code == 503 and caido.json()["detail"] == "Redis is unavailable", caido.text)

    from fastapi import HTTPException

    import app.graph as graph_mod
    from app.main import ApprovalRequest, TaskRequest, approve_task, create_task, get_task
    from app.worker import task_worker

    redis = RedisFalso()
    worker.redis_client = redis

    def pausa(entrada: object, config: dict, recursion_limit: int = 25) -> dict:
        return {
            "paused": True,
            "interrupt": {"tool": "registrar_ajuste_nomina", "reason": "side-effect tool registrar_ajuste_nomina"},
            "result": None,
        }

    def sigue(decision: dict, config: dict, recursion_limit: int = 25) -> dict:
        return {"paused": False, "interrupt": None, "result": "Ajuste registrado."}

    original_correr = graph_mod.correr
    original_reanudar = graph_mod.reanudar
    graph_mod.correr = pausa  # type: ignore[method-assign]
    graph_mod.reanudar = sigue  # type: ignore[method-assign]
    tarea = asyncio.create_task(task_worker())
    await asyncio.sleep(0)
    try:
        creado = await create_task(TaskRequest(query=PREGUNTA_CRITICA))
        print("POST /tasks:", creado.model_dump())
        _exigir(creado.status == "PENDING", "el POST no devolvió PENDING")
        estado = creado
        for _ in range(50):
            estado = await get_task(creado.job_id)
            if estado.status == "AWAITING_APPROVAL":
                break
            await asyncio.sleep(0.01)
        print("GET en pausa:", estado.model_dump())
        _exigir(estado.status == "AWAITING_APPROVAL", "no pausó")
        _exigir(estado.interrupt is not None and estado.interrupt.get("tool") == "registrar_ajuste_nomina", "sin interrupt")
        try:
            await approve_task("inventado", ApprovalRequest(approved=True))
        except HTTPException as exc:
            print("404 approve:", exc.status_code, exc.detail)
            _exigir(exc.status_code == 404 and exc.detail == "Job not found", "approve 404")
        else:
            raise SystemExit("approve de un id inventado no respondió 404")
        respuesta = await approve_task(creado.job_id, ApprovalRequest(approved=True))
        print("POST approve:", respuesta.model_dump())
        _exigir(respuesta.status == "RUNNING", "approve no devolvió RUNNING")
        final = respuesta
        for _ in range(50):
            final = await get_task(creado.job_id)
            if final.status == "DONE":
                break
            await asyncio.sleep(0.01)
        print("GET final:", final.model_dump())
        _exigir(final.status == "DONE" and final.result == "Ajuste registrado.", "no llegó a DONE")
        try:
            await approve_task(creado.job_id, ApprovalRequest(approved=True))
        except HTTPException as exc:
            print("409:", exc.status_code, exc.detail)
            _exigir(exc.status_code == 409 and exc.detail == "Job is not awaiting approval", "approve sin pausa")
        else:
            raise SystemExit("approve de una tarea DONE no respondió 409")

        def boom(entrada: object, config: dict, recursion_limit: int = 25) -> dict:
            raise RuntimeError("corte de prueba")

        graph_mod.correr = boom  # type: ignore[method-assign]
        fallido = await create_task(TaskRequest(query="hola"))
        mal = fallido
        for _ in range(50):
            mal = await get_task(fallido.job_id)
            if mal.status == "FAILED":
                break
            await asyncio.sleep(0.01)
        print("FAILED:", mal.model_dump())
        _exigir(mal.status == "FAILED" and mal.error == "corte de prueba", "no guardó FAILED")
    finally:
        graph_mod.correr = original_correr  # type: ignore[method-assign]
        graph_mod.reanudar = original_reanudar  # type: ignore[method-assign]
        tarea.cancel()
        try:
            await tarea
        except asyncio.CancelledError:
            pass


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
    else:
        raise SystemExit("sin OPENAI_API_KEY no salió 401/key")
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
    efectos.clear()
    usar(compilar(MemorySaver()))

    normal = correr(estado_inicial(PREGUNTA), {"configurable": {"thread_id": "demo"}}, 25)
    print("Tarea normal:", normal)
    _exigir(normal["paused"] is False, "la tarea normal no debía pausar")
    _exigir("231" in normal["result"], normal["result"])

    pausa = correr(estado_inicial(PREGUNTA_CRITICA), {"configurable": {"thread_id": "demo-hitl"}}, 25)
    print("Tarea crítica:", pausa)
    _exigir(pausa["paused"] is True, "la tarea crítica no pausó")
    _exigir(efectos == [], "la herramienta corrió antes de la aprobación")
    cerrado = reanudar({"approved": True}, {"configurable": {"thread_id": "demo-hitl"}}, 25)
    print("Después de aprobar:", cerrado["paused"], efectos)
    _exigir(cerrado["paused"] is False and len(efectos) == 1, "aprobar no asentó el ajuste")

    asyncio.run(_ciclo_http())
    _git_limpio()
    print("validacion ok")


if __name__ == "__main__":
    main()
