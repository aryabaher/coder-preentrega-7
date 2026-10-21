import modelo
from app.observability import _instrumentar, memoria
from opentelemetry.sdk.trace import TracerProvider


def test_invocar_abre_el_span_llm_call() -> None:
    memoria.clear()
    assert modelo.invocar(lambda: "dato") == "dato"
    assert any(span.name == "llm.call" for span in memoria.get_finished_spans())
    assert hasattr(modelo.invocar, "__wrapped__")


def test_el_span_marca_el_error() -> None:
    memoria.clear()

    def falla() -> None:
        raise TimeoutError("timed out")

    try:
        modelo.invocar(falla, intentos=1)
    except Exception:
        pass
    spans = [span for span in memoria.get_finished_spans() if span.name == "llm.call"]
    assert spans
    assert spans[-1].status.status_code.name == "ERROR"


def test_los_instrumentadores_se_registran() -> None:
    _instrumentar(TracerProvider())
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor

    assert LangChainInstrumentor
    assert OpenAIInstrumentor
