import asyncio

from opentelemetry.sdk.trace import TracerProvider

import modelo
from app import worker
from app.main import skip_api_spans
from app.observability import _instrument, memory


def test_invocar_abre_el_span_llm_call() -> None:
    memory.clear()
    assert modelo.invocar(lambda: "dato") == "dato"
    assert any(span.name == "llm.call" for span in memory.get_finished_spans())
    assert hasattr(modelo.invocar, "__wrapped__")


def test_el_span_marca_el_error() -> None:
    memory.clear()

    def falla() -> None:
        raise TimeoutError("timed out")

    try:
        modelo.invocar(falla, intentos=1)
    except Exception:
        pass
    spans = [span for span in memory.get_finished_spans() if span.name == "llm.call"]
    assert spans
    assert spans[-1].status.status_code.name == "ERROR"


def test_langchain_instrumentado_y_openai_no() -> None:
    _instrument(TracerProvider())
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor

    assert LangChainInstrumentor().is_instrumented_by_opentelemetry
    # Los dos juntos cuentan cada llamada dos veces y Phoenix duplica el costo.
    assert not OpenAIInstrumentor().is_instrumented_by_opentelemetry


def test_el_polling_no_genera_trazas() -> None:
    assert skip_api_spans({"path": "/tasks/abc", "method": "GET"}) is True
    assert skip_api_spans({"path": "/tasks", "method": "POST"}) is True
    assert skip_api_spans({"path": "/threads/abc", "method": "GET"}) is True
    assert skip_api_spans({"path": "/docs", "method": "GET"}) is False


def test_la_ejecucion_es_una_traza_con_sesion_y_estados() -> None:
    memory.clear()
    record = {"job_id": "job-1", "thread_id": "hilo-1"}

    def grafo(*args: object) -> dict:
        modelo.invocar(lambda: "dato")
        return {"paused": False, "interrupt": None, "result": "listo"}

    resultado = asyncio.run(worker._run_traced(record, "pregunta", grafo))
    assert resultado["result"] == "listo"
    spans = memory.get_finished_spans()
    raiz = next(span for span in spans if span.name == "orquestador")
    hijo = next(span for span in spans if span.name == "llm.call")
    assert raiz.parent is None
    assert hijo.context.trace_id == raiz.context.trace_id
    assert raiz.attributes["session.id"] == "hilo-1"
    assert raiz.attributes["input.value"] == "pregunta"
    assert raiz.attributes["output.value"] == "listo"
    assert [e.attributes["status"] for e in raiz.events] == ["RUNNING", "DONE"]


def test_la_ejecucion_fallida_marca_failed_en_la_traza() -> None:
    memory.clear()

    def grafo(*args: object) -> dict:
        raise RuntimeError("corte de prueba")

    try:
        asyncio.run(worker._run_traced({"job_id": "j", "thread_id": "h"}, "q", grafo))
    except RuntimeError:
        pass
    raiz = next(span for span in memory.get_finished_spans() if span.name == "orquestador")
    assert [e.attributes["status"] for e in raiz.events if e.name == "status"] == ["RUNNING", "FAILED"]
    assert raiz.status.status_code.name == "ERROR"
