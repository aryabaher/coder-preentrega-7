"""Lee de Phoenix lo que Phoenix calculó: p95, costo por ejecución, y tokens y latencia por nodo."""

from requisitos import exigir_python_312

exigir_python_312()

import json
import os
import statistics
import sys
import urllib.request
from pathlib import Path

PROYECTO = "preentrega-7"
NODOS = ("supervisor", "investigador", "analista", "sintesis", "aprobacion", "ejecutor")
SALIDA = Path(__file__).with_name("evidencias") / "05-metricas.txt"

CONSULTA_PROYECTO = """
query ($id: ID!) {
  node(id: $id) {
    ... on Project {
      traceCount
      p50: latencyMsQuantile(probability: 0.5)
      p95: latencyMsQuantile(probability: 0.95)
      p99: latencyMsQuantile(probability: 0.99)
      costSummary { total { cost } prompt { cost } completion { cost } }
      tokenCountPrompt
      tokenCountCompletion
    }
  }
}
"""

CONSULTA_SPANS = """
query ($id: ID!, $filtro: String!) {
  node(id: $id) {
    ... on Project {
      spans(first: 200, filterCondition: $filtro, sort: {col: startTime, dir: asc}) {
        edges {
          node {
            name
            latencyMs
            cumulativeTokenCountPrompt
            cumulativeTokenCountCompletion
            cumulativeTokenCountTotal
            attributes
            trace { traceId latencyMs costSummary { total { cost } } }
          }
        }
      }
    }
  }
}
"""


def base() -> str:
    return os.getenv("PHOENIX_UI", "http://localhost:6006").rstrip("/")


def graphql(consulta: str, variables: dict) -> dict:
    cuerpo = json.dumps({"query": consulta, "variables": variables}).encode()
    pedido = urllib.request.Request(f"{base()}/graphql", data=cuerpo, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(pedido, timeout=30) as respuesta:
        datos = json.loads(respuesta.read().decode())
    if datos.get("errors"):
        raise SystemExit(f"Phoenix GraphQL: {datos['errors']}")
    return datos["data"]


def id_proyecto(nombre: str = PROYECTO) -> str:
    with urllib.request.urlopen(f"{base()}/v1/projects", timeout=10) as respuesta:
        proyectos = json.loads(respuesta.read().decode())["data"]
    for proyecto in proyectos:
        if proyecto["name"] == nombre:
            return str(proyecto["id"])
    raise SystemExit(f"No existe el proyecto {nombre} en Phoenix. Corré python carga.py primero.")


def spans(proyecto: str, nombre: str) -> list[dict]:
    datos = graphql(CONSULTA_SPANS, {"id": proyecto, "filtro": f"name == '{nombre}'"})
    return [arista["node"] for arista in datos["node"]["spans"]["edges"]]


def costo(traza: dict) -> float:
    return float(((traza.get("costSummary") or {}).get("total") or {}).get("cost") or 0.0)


def informe(proyecto: str) -> str:
    resumen = graphql(CONSULTA_PROYECTO, {"id": proyecto})["node"]
    ejecuciones = spans(proyecto, "orquestador")
    lineas = [
        f"Phoenix {base()}, proyecto {PROYECTO}",
        f"Trazas en el proyecto: {resumen['traceCount']}",
        f"Latencia de traza calculada por Phoenix: p50 {resumen['p50'] / 1000:.2f} s, "
        f"p95 {resumen['p95'] / 1000:.2f} s, p99 {resumen['p99'] / 1000:.2f} s",
        f"Costo total calculado por Phoenix: {resumen['costSummary']['total']['cost']:.6f} USD "
        f"(entrada {resumen['costSummary']['prompt']['cost']:.6f}, salida {resumen['costSummary']['completion']['cost']:.6f})",
        f"Tokens: entrada {int(resumen['tokenCountPrompt'])}, salida {int(resumen['tokenCountCompletion'])}",
        "",
        "Costo por ejecución (una traza `orquestador` por job):",
    ]
    costos = []
    for span in ejecuciones:
        traza = span["trace"]
        valor = costo(traza)
        costos.append(valor)
        atributos = json.loads(span["attributes"]) if isinstance(span["attributes"], str) else span["attributes"]
        job_id = (atributos or {}).get("job_id", "?")
        lineas.append(
            f"- job {job_id}: {traza['latencyMs'] / 1000:.2f} s, {int(span['cumulativeTokenCountTotal'])} tokens, {valor:.6f} USD"
        )
    if costos:
        lineas.append(f"Costo medio por ejecución: {statistics.mean(costos):.6f} USD (n = {len(costos)})")
    lineas += ["", "Por nodo del grafo (suma de todas las ejecuciones):"]
    filas = []
    for nodo in NODOS:
        datos = spans(proyecto, nodo)
        if not datos:
            continue
        tokens = sum(int(s["cumulativeTokenCountTotal"] or 0) for s in datos)
        latencia = sum(float(s["latencyMs"] or 0) for s in datos)
        filas.append((nodo, len(datos), tokens, latencia))
    total_tokens = sum(f[2] for f in filas) or 1
    total_latencia = sum(f[3] for f in filas) or 1
    for nodo, veces, tokens, latencia in filas:
        lineas.append(
            f"- {nodo}: {veces} spans, {tokens} tokens ({100 * tokens / total_tokens:.0f}%), "
            f"{latencia / 1000:.2f} s ({100 * latencia / total_latencia:.0f}% de la latencia de nodos)"
        )
    if filas:
        mas_tokens = max(filas, key=lambda f: f[2])
        mas_lento = max(filas, key=lambda f: f[3])
        lineas += [
            "",
            f"Nodo con más tokens: {mas_tokens[0]}. Nodo donde se concentra la latencia: {mas_lento[0]}.",
        ]
    return "\n".join(lineas) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    texto = informe(id_proyecto())
    print(texto)
    destino = Path(sys.argv[1]) if len(sys.argv) > 1 else SALIDA
    if destino.name != "-":
        destino.write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    main()
