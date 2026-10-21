"""Arize Phoenix through OpenInference, and the trace decorator for every LLM call."""

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
PROJECT = "preentrega-7"

memory = InMemorySpanExporter()
_active = False
_instrumented = False
_provider: TracerProvider | None = None

T = TypeVar("T")


def trace_llm(fn: Callable[..., T]) -> Callable[..., T]:
    """Trace decorator: every LLM call opens the `llm.call` span."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        with trace.get_tracer(PROJECT).start_as_current_span("llm.call") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    return wrapper


def init_observability() -> TracerProvider:
    """Register the exporter and the instrumentors. Once per process.

    With TRAZAS_OFFLINE=1 nothing leaves the process: spans stay in `memory`.
    """
    global _active, _provider
    if _active and _provider is not None:
        return _provider
    if os.getenv("TRAZAS_OFFLINE") == "1":
        _provider = _local_provider()
    else:
        _provider = _register(endpoint(), PROJECT)
        _instrument(_provider)
    _active = True
    return _provider


def endpoint() -> str:
    return (os.getenv("PHOENIX_ENDPOINT") or ENDPOINT).strip() or ENDPOINT


def _local_provider() -> TracerProvider:
    current = trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        current.add_span_processor(SimpleSpanProcessor(memory))
        return current
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    trace.set_tracer_provider(provider)
    return provider


def _register(endpoint_url: str, project_name: str) -> TracerProvider:
    """OTLP HTTP to Phoenix. `/v1/traces` is the collector path."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    resource = Resource.create(
        {
            "service.name": project_name,
            "openinference.project.name": project_name,
        }
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint_url)))
    trace.set_tracer_provider(provider)
    return provider


def _instrument(provider: TracerProvider) -> None:
    """OpenInference: LangChain (every ChatOpenAI call and the graph nodes) and Pinecone."""
    global _instrumented
    if _instrumented:
        return
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from opentelemetry.instrumentation.pinecone import PineconeInstrumentor

    # No OpenAIInstrumentor: ChatOpenAI and ChatCompletion would both carry the tokens, doubling Phoenix cost.
    LangChainInstrumentor().instrument(tracer_provider=provider)
    PineconeInstrumentor().instrument(tracer_provider=provider)  # type: ignore[no-untyped-call]
    _instrumented = True
