"""Demo contra la API real: aprobación humana, un hilo de dos turnos y la pausa que sobrevive a un reinicio.

python demo_api.py hitl              tarea crítica, AWAITING_APPROVAL, approve, DONE
python demo_api.py hilo              dos turnos en el mismo thread_id y GET /threads
python demo_api.py pausar            deja una tarea crítica en AWAITING_APPROVAL y guarda su job_id
python demo_api.py aprobar           aprueba la tarea guardada (correr después de reiniciar la API)
python demo_api.py historial <id>    GET /threads/<id>
"""

from requisitos import exigir_python_312

exigir_python_312()

import json
import os
import sys
import time
from pathlib import Path

import httpx

PREGUNTA = "¿Cuántos días de vacaciones corresponden con 5 años de antigüedad en TechCorp?"
SEGUIMIENTO = "¿Y cuántos días acumula en total en esos 5 años?"
CRITICA = (
    "¿Cuántos días de vacaciones corresponden con 5 años de antigüedad? "
    "Registrá el ajuste de nómina con register_payroll_adjustment para la legajo 1042."
)
PENDIENTE = Path(__file__).with_name(".pausa_pendiente.json")


def cliente() -> httpx.Client:
    return httpx.Client(base_url=os.getenv("API_URL", "http://localhost:8000"), timeout=30)


def mostrar(titulo: str, cuerpo: object) -> None:
    print(f"{titulo}: {json.dumps(cuerpo, ensure_ascii=False)}")


def esperar(http: httpx.Client, job_id: str, estados: tuple[str, ...]) -> dict:
    vistos: list[str] = []
    plazo = time.perf_counter() + 300
    while time.perf_counter() < plazo:
        estado = http.get(f"/tasks/{job_id}").json()
        if not vistos or vistos[-1] != estado["status"]:
            vistos.append(estado["status"])
        if estado["status"] in estados or estado["status"] == "FAILED":
            print(f"Estados vistos por polling: {' -> '.join(vistos)}")
            return estado
        time.sleep(1)
    raise SystemExit(f"timeout esperando {job_id}")


def crear(http: httpx.Client, query: str, thread_id: str | None = None) -> dict:
    cuerpo: dict[str, str] = {"query": query}
    if thread_id:
        cuerpo["thread_id"] = thread_id
    respuesta = http.post("/tasks", json=cuerpo)
    respuesta.raise_for_status()
    mostrar("POST /tasks", respuesta.json())
    return respuesta.json()


def aprobar(http: httpx.Client, job_id: str, approved: bool = True) -> dict:
    respuesta = http.post(f"/tasks/{job_id}/approve", json={"approved": approved})
    mostrar(f"POST /tasks/{job_id}/approve", respuesta.json())
    respuesta.raise_for_status()
    final = esperar(http, job_id, ("DONE",))
    mostrar("GET final", {k: final[k] for k in ("status", "result", "error")})
    return final


def hitl() -> None:
    with cliente() as http:
        creado = crear(http, CRITICA)
        pausa = esperar(http, creado["job_id"], ("AWAITING_APPROVAL",))
        mostrar("GET en pausa", {"status": pausa["status"], "interrupt": pausa["interrupt"]})
        aprobar(http, creado["job_id"])
        rechazo = crear(http, CRITICA)
        esperar(http, rechazo["job_id"], ("AWAITING_APPROVAL",))
        aprobar(http, rechazo["job_id"], approved=False)


def hilo() -> None:
    with cliente() as http:
        primero = crear(http, PREGUNTA)
        final = esperar(http, primero["job_id"], ("DONE",))
        mostrar("Turno 1", {"status": final["status"], "result": final["result"]})
        segundo = crear(http, SEGUIMIENTO, primero["thread_id"])
        final = esperar(http, segundo["job_id"], ("DONE",))
        mostrar("Turno 2", {"status": final["status"], "result": final["result"]})
        historial(primero["thread_id"], http)


def historial(thread_id: str, http: httpx.Client | None = None) -> None:
    propio = http is None
    http = http or cliente()
    try:
        respuesta = http.get(f"/threads/{thread_id}")
        print(f"GET /threads/{thread_id}: {respuesta.status_code}")
        for mensaje in respuesta.json().get("messages", []):
            print(f"  [{mensaje['role']}:{mensaje['name'] or '-'}] {mensaje['content'][:140]}")
    finally:
        if propio:
            http.close()


def pausar() -> None:
    with cliente() as http:
        creado = crear(http, CRITICA)
        pausa = esperar(http, creado["job_id"], ("AWAITING_APPROVAL",))
        mostrar("GET en pausa", {"status": pausa["status"], "interrupt": pausa["interrupt"]})
        PENDIENTE.write_text(json.dumps(creado), encoding="utf-8")
        print(f"Guardado en {PENDIENTE.name}. Reiniciá la API y corré: python demo_api.py aprobar")


def aprobar_pendiente() -> None:
    creado = json.loads(PENDIENTE.read_text(encoding="utf-8"))
    with cliente() as http:
        estado = http.get(f"/tasks/{creado['job_id']}").json()
        mostrar("GET tras el reinicio", {"status": estado["status"], "thread_id": estado["thread_id"]})
        aprobar(http, creado["job_id"])
        historial(creado["thread_id"], http)
    PENDIENTE.unlink()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    modo = sys.argv[1] if len(sys.argv) > 1 else "hitl"
    acciones = {"hitl": hitl, "hilo": hilo, "pausar": pausar, "aprobar": aprobar_pendiente}
    if modo == "historial":
        historial(sys.argv[2])
    elif modo in acciones:
        acciones[modo]()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
