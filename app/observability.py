"""Arize Phoenix por OpenInference, y el decorador de trazas de cada llamada al LLM."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

ENDPOINT = "http://localhost:6006/v1/traces"
PHOENIX_UI = "http://localhost:6006"
PROYECTO = "preentrega-7"

memoria = InMemorySpanExporter()
_activo = False
_instrumentado = False
_provider: TracerProvider | None = None

T = TypeVar("T")


def trazar_llm(fn: Callable[..., T]) -> Callable[..., T]:
    """Decorador de trazas: cada llamada al LLM abre el span `llm.call`."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        with trace.get_tracer("preentrega-7").start_as_current_span("llm.call") as span:
            span.set_attribute("openinference.span.kind", "LLM")
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    return wrapper


def iniciar_observabilidad() -> TracerProvider:
    """Registra el exportador y los instrumentadores. Una sola vez por proceso.

    Con TRAZAS_OFFLINE=1 no abre la red: los spans quedan en `memoria`.
    """
    global _activo, _provider
    if _activo and _provider is not None:
        return _provider
    if os.getenv("TRAZAS_OFFLINE") == "1":
        _provider = _provider_local()
    else:
        _provider = _register(_endpoint(), PROYECTO)
        _instrumentar(_provider)
    _activo = True
    return _provider


def _endpoint() -> str:
    return (os.getenv("PHOENIX_ENDPOINT") or ENDPOINT).strip() or ENDPOINT


def _provider_local() -> TracerProvider:
    actual = trace.get_tracer_provider()
    if isinstance(actual, TracerProvider):
        actual.add_span_processor(SimpleSpanProcessor(memoria))
        return actual
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memoria))
    trace.set_tracer_provider(provider)
    return provider


def _register(endpoint: str, project_name: str) -> TracerProvider:
    """OTLP HTTP hacia Phoenix. El path `/v1/traces` es el del colector."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    recurso = Resource.create(
        {
            "service.name": project_name,
            "openinference.project.name": project_name,
        }
    )
    provider = TracerProvider(resource=recurso)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    return provider


def _instrumentar(provider: TracerProvider) -> None:
    """OpenInference: LangChain, el SDK de OpenAI y Pinecone."""
    global _instrumentado
    if _instrumentado:
        return
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from opentelemetry.instrumentation.pinecone import PineconeInstrumentor

    LangChainInstrumentor().instrument(tracer_provider=provider)
    OpenAIInstrumentor().instrument(tracer_provider=provider)
    PineconeInstrumentor().instrument(tracer_provider=provider)  # type: ignore[no-untyped-call]
    _instrumentado = True
