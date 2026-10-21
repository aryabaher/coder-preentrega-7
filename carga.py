"""Cinco POST /tasks concurrentes. El costo y el p95 se leen en el dashboard de Phoenix."""

from requisitos import exigir_python_312

exigir_python_312()

import asyncio
import os
import sys
import time
import urllib.error
import urllib.request

import httpx

CONSULTAS = [
    "Un empleado va a cumplir 12 años en TechCorp. Tuvo 14 días de vacaciones los primeros "
    "5 años, 21 los siguientes 5, y ahora tiene derecho a 28. ¿Cuántos días de vacaciones "
    "acumuló en total en esos 12 años?",
    "¿Cuántos días de teletrabajo por semana permite la política de TechCorp?",
    "¿Qué tiene que completar una persona nueva durante el onboarding?",
    "¿Cuál es la regla de contraseñas de la política de seguridad informática?",
    "¿Cuántos días de vacaciones corresponden al tramo de 5 años de antigüedad?",
]


def _limpiar_proyecto() -> None:
    """Borra el proyecto para que el p95 de esta corrida no mezcle trazas viejas."""
    url = os.getenv("PHOENIX_UI", "http://localhost:6006").rstrip("/") + "/v1/projects/preentrega-7"
    pedido = urllib.request.Request(url, method="DELETE")
    try:
        with urllib.request.urlopen(pedido, timeout=5) as respuesta:
            print(f"Phoenix DELETE {respuesta.status}")
    except urllib.error.HTTPError as exc:
        print(f"Phoenix DELETE {exc.code} (si era 404, el proyecto todavía no existía)")
    except urllib.error.URLError as exc:
        print(f"Phoenix no respondió: {exc.reason}")


async def _esperar(cliente: httpx.AsyncClient, job_id: str) -> dict:
    plazo = time.perf_counter() + 300
    ultimo: dict = {}
    while time.perf_counter() < plazo:
        respuesta = await cliente.get(f"/tasks/{job_id}")
        respuesta.raise_for_status()
        ultimo = respuesta.json()
        if ultimo["status"] in ("DONE", "FAILED"):
            return ultimo
        await asyncio.sleep(1)
    raise SystemExit(f"timeout esperando {job_id}: {ultimo}")


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    _limpiar_proyecto()
    base = os.getenv("API_URL", "http://localhost:8000")
    async with httpx.AsyncClient(base_url=base, timeout=30) as cliente:
        inicio = time.perf_counter()
        respuestas = await asyncio.gather(*[cliente.post("/tasks", json={"query": texto}) for texto in CONSULTAS])
        demora = time.perf_counter() - inicio
        print(f"5 POST /tasks en {demora:.3f}s")
        if demora >= 1:
            raise SystemExit("los 5 POST tardaron demasiado: el endpoint no debería esperar al grafo")
        jobs = []
        for respuesta in respuestas:
            respuesta.raise_for_status()
            cuerpo = respuesta.json()
            print(cuerpo)
            if cuerpo.get("status") != "PENDING" or "job_id" not in cuerpo:
                raise SystemExit(f"respuesta inesperada: {cuerpo}")
            jobs.append(cuerpo["job_id"])
        if len(set(jobs)) != 5:
            raise SystemExit("los job_id no son distintos")
        finales = await asyncio.gather(*[_esperar(cliente, job_id) for job_id in jobs])
    fallos = [item for item in finales if item["status"] != "DONE"]
    for item in finales:
        print(f"{item['job_id']} {item['status']} error={item.get('error')}")
    if fallos:
        raise SystemExit(f"{len(fallos)} tareas en FAILED")
    print("5 tareas DONE. En http://localhost:6006, proyecto preentrega-7:")
    print("- costo por ejecución: lo calcula Phoenix con los tokens de entrada y salida")
    print("- latencia p95: el percentil de esta corrida, con el proyecto recién borrado")


if __name__ == "__main__":
    asyncio.run(main())
