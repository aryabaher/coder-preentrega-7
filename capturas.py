"""Capturas del dashboard de Phoenix con Playwright (usa el Chrome instalado).

python capturas.py carga   01-trazas, 02-costo-por-ejecucion, 02b con el costo exacto y 03-latencia-p95 (después de carga.py)
python capturas.py hitl [thread_id]   04-hitl y 05-sesion (después de demo_api.py hitl y demo_api.py hilo)
"""

from requisitos import exigir_python_312

exigir_python_312()

import sys
import urllib.parse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

from metricas import base, graphql, id_proyecto

CARPETA = Path(__file__).with_name("screenshots")
COLUMNAS_OCULTAS = ("output", "error", "user", "annotations", "trace annotations")
SERIES_OCULTAS = ("Max", "P50", "P75", "P90", "P99", "P99.9")

CONSULTA_TRAZA = """
query ($id: ID!, $filtro: String!) {
  node(id: $id) {
    ... on Project {
      spans(first: 50, filterCondition: $filtro, sort: {col: startTime, dir: desc}) {
        edges { node { trace { traceId } } }
      }
    }
  }
}
"""

CONSULTA_SESIONES = """
query ($id: ID!) {
  node(id: $id) {
    ... on Project {
      sessions(first: 50) { edges { node { id sessionId numTraces } } }
    }
  }
}
"""


def traza_con(proyecto: str, nodo: str) -> str:
    datos = graphql(CONSULTA_TRAZA, {"id": proyecto, "filtro": f"name == '{nodo}'"})
    aristas = datos["node"]["spans"]["edges"]
    if not aristas:
        raise SystemExit(f"No hay trazas con el nodo {nodo}.")
    return str(aristas[0]["node"]["trace"]["traceId"])


def sesion_con_turnos(proyecto: str, thread_id: str | None = None) -> str:
    sesiones = [a["node"] for a in graphql(CONSULTA_SESIONES, {"id": proyecto})["node"]["sessions"]["edges"]]
    candidatas = [s for s in sesiones if s["numTraces"] >= 2 and thread_id in (None, s["sessionId"])]
    if not candidatas:
        raise SystemExit("No hay una sesión con dos turnos. Corré python demo_api.py hilo.")
    return str(candidatas[0]["id"])


def guardar(pagina: Page, nombre: str) -> None:
    destino = CARPETA / nombre
    pagina.screenshot(path=str(destino))
    print(f"{destino.relative_to(Path(__file__).parent)}")


def captura_traza(pagina: Page, proyecto: str, traza: str, nombre: str) -> None:
    pagina.goto(f"{base()}/projects/{proyecto}/traces/{traza}")
    pagina.wait_for_timeout(5000)
    guardar(pagina, nombre)


def captura_costo(pagina: Page, proyecto: str) -> None:
    pagina.goto(f"{base()}/projects/{proyecto}/traces?timeRangeKey=1h")
    pagina.wait_for_timeout(4000)
    pagina.get_by_role("button", name="Columns").click()
    pagina.wait_for_timeout(500)
    for columna in COLUMNAS_OCULTAS:
        casilla = pagina.get_by_role("checkbox", name=columna, exact=True)
        pagina.locator("li", has=casilla).first.click()
    pagina.keyboard.press("Escape")
    pagina.wait_for_timeout(1000)
    guardar(pagina, "02-costo-por-ejecucion.png")


def captura_costo_exacto(navegador: object, proyecto: str) -> None:
    """La UI redondea a <$0.01. El endpoint GraphQL de Phoenix da el costo que calculó, sin redondear."""
    consulta = f"""{{
  node(id: "{proyecto}") {{
    ... on Project {{
      traceCount
      latencyP95Ms: latencyMsQuantile(probability: 0.95)
      costSummary {{ total {{ cost }} }}
      executions: spans(first: 10, filterCondition: "name == 'orquestador'") {{
        edges {{ node {{ latencyMs cumulativeTokenCountTotal trace {{ costSummary {{ total {{ cost }} }} }} }} }}
      }}
    }}
  }}
}}"""
    pagina = navegador.new_page(viewport={"width": 900, "height": 200}, device_scale_factor=1.5)  # type: ignore[attr-defined]
    pagina.goto(f"{base()}/graphql?query=" + urllib.parse.quote(consulta))
    pagina.wait_for_timeout(2000)
    guardar(pagina, "02b-costo-por-ejecucion-phoenix.png")
    pagina.close()


def captura_p95(pagina: Page, proyecto: str) -> str:
    pagina.goto(f"{base()}/projects/{proyecto}/metrics?timeRangeKey=1h")
    pagina.wait_for_timeout(5000)
    for serie in SERIES_OCULTAS:
        pagina.locator(".recharts-legend-item").filter(has=pagina.get_by_text(serie, exact=True)).first.click()
        pagina.wait_for_timeout(200)
    grafico = pagina.locator(".recharts-wrapper").nth(1)
    punto = grafico.locator(".recharts-dot").first
    caja = punto.bounding_box()
    if caja is None:
        raise SystemExit("No encontré el punto P95 en el gráfico de latencia.")
    pagina.mouse.move(caja["x"] + caja["width"] / 2, caja["y"] + caja["height"] / 2)
    pagina.wait_for_timeout(1000)
    tooltip = pagina.locator(".recharts-tooltip-wrapper").nth(1).inner_text()
    grafico.scroll_into_view_if_needed()
    guardar(pagina, "03-latencia-p95.png")
    return tooltip


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    modo = sys.argv[1] if len(sys.argv) > 1 else "carga"
    proyecto = id_proyecto()
    CARPETA.mkdir(exist_ok=True)
    with sync_playwright() as p:
        navegador = p.chromium.launch(channel="chrome")
        pagina = navegador.new_page(viewport={"width": 1700, "height": 1000})
        if modo == "carga":
            captura_traza(pagina, proyecto, traza_con(proyecto, "analista"), "01-trazas.png")
            captura_costo(pagina, proyecto)
            captura_costo_exacto(navegador, proyecto)
            print("Tooltip P95:", captura_p95(pagina, proyecto).replace("\n", " "))
        elif modo == "hitl":
            captura_traza(pagina, proyecto, traza_con(proyecto, "ejecutor"), "04-hitl.png")
            thread_id = sys.argv[2] if len(sys.argv) > 2 else None
            pagina.goto(f"{base()}/projects/{proyecto}/sessions/{sesion_con_turnos(proyecto, thread_id)}")
            pagina.wait_for_timeout(5000)
            guardar(pagina, "05-sesion.png")
        else:
            raise SystemExit(__doc__)
        navegador.close()


if __name__ == "__main__":
    main()
