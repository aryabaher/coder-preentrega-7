import ast
import re
from pathlib import Path

import pytest

from app import graph

RAIZ = Path(__file__).resolve().parents[1]

# The startup guard keeps its established name in every entrypoint.
GUARDIA = {"requisitos", "exigir_python_312"}
ESPANOL = {
    "estado", "encolar", "leer", "tarea", "compilar", "usar", "clave", "datos", "consulta", "resultado",
    "cliente", "caido", "redis_caido", "listo", "obtener", "esta_listo", "correr", "reanudar", "hilo",
    "trabajo", "pregunta", "respuesta", "aprobado", "efectos", "nodo", "seguir", "iniciar", "trazar",
    "observabilidad", "mensajes", "historial", "ejecutor", "aprobacion", "critica", "costo", "envoltura",
}


def _texto(relativo: str) -> str:
    return (RAIZ / relativo).read_text(encoding="utf-8")


def _identificadores(relativo: str) -> set[str]:
    arbol = ast.parse(_texto(relativo))
    nombres: set[str] = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Name):
            nombres.add(nodo.id)
        elif isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nombres.add(nodo.name)
        elif isinstance(nodo, ast.arg):
            nombres.add(nodo.arg)
        elif isinstance(nodo, ast.Attribute):
            nombres.add(nodo.attr)
        elif isinstance(nodo, ast.alias):
            nombres.add((nodo.asname or nodo.name).split(".")[-1])
    return nombres - GUARDIA


@pytest.mark.parametrize("relativo", ["app/main.py", "app/worker.py", "app/hitl.py", "app/observability.py"])
def test_un_solo_idioma_en_la_api(relativo: str) -> None:
    nombres = _identificadores(relativo)
    partes = {p for nombre in nombres for p in re.split(r"_+", nombre.lower()) if p}
    assert all(nombre.isascii() for nombre in nombres)
    assert not (partes & ESPANOL), sorted(partes & ESPANOL)


@pytest.mark.parametrize("relativo", ["app/main.py", "app/worker.py", "app/hitl.py", "app/observability.py"])
def test_comentarios_y_docstrings_en_ingles(relativo: str) -> None:
    fuente = _texto(relativo)
    comentarios = re.findall(r"#\s*(.+)", fuente)
    docstrings = [ast.get_docstring(n) or "" for n in ast.walk(ast.parse(fuente)) if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    texto = " ".join(comentarios + docstrings)
    assert not re.search(r"[áéíóúñ¿¡]", texto), texto


def test_la_api_tipa_la_salida_y_los_errores() -> None:
    fuente = _texto("app/main.py")
    assert "response_model=TaskResponse" in fuente
    assert "response_model=TaskStatus" in fuente
    assert "response_model=ThreadHistory" in fuente
    assert 'detail="Job not found"' in fuente
    assert 'detail="Thread not found"' in fuente
    assert 'detail="Thread is busy"' in fuente
    assert 'detail="Redis is unavailable"' in fuente
    assert "sk-" not in fuente
    assert ".invoke(" not in fuente
    assert "GET /tasks/{id}" in fuente


def test_el_approve_vive_en_hitl() -> None:
    fuente = _texto("app/hitl.py")
    assert "interrupt(" in fuente
    assert '@router.post("/tasks/{job_id}/approve", response_model=ApprovalResponse)' in fuente
    assert 'detail="Job not found"' in fuente
    assert "app.include_router(hitl.router)" in _texto("app/main.py")
    assert "/approve" not in _texto("app/main.py").split('"""', 2)[2]


def test_el_worker_cubre_el_ciclo_y_no_bloquea_el_loop() -> None:
    fuente = _texto("app/worker.py")
    for estado in ("PENDING", "RUNNING", "FAILED", "DONE", "AWAITING_APPROVAL"):
        assert f'"{estado}"' in fuente
    assert "asyncio.to_thread" in fuente
    assert 'QUEUE_NAME = "tasks_queue"' in fuente
    assert 'STATUS_PREFIX = "status:"' in fuente
    assert "blpop(QUEUE_NAME, timeout=0)" in fuente
    assert '"thread_id": thread_id' in fuente


def test_el_grafo_usa_redis_saver(monkeypatch: pytest.MonkeyPatch) -> None:
    from langgraph.checkpoint.memory import MemorySaver

    fuente = _texto("app/graph.py")
    assert "RedisSaver" in fuente
    assert "checkpointer=checkpointer" in fuente
    llamadas: list[object] = []

    def falso(url: str | None = None) -> MemorySaver:
        llamadas.append(url)
        return MemorySaver()

    monkeypatch.setattr(graph, "abrir_redis_saver", falso)
    graph.compile_graph()
    assert llamadas == [None]


def test_abrir_redis_saver_hace_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    vistos: list[object] = []

    class Falso:
        @classmethod
        def from_conn_string(cls, url: str) -> "Falso":
            vistos.append(url)
            return cls()

        def __enter__(self) -> "Falso":
            vistos.append("enter")
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def setup(self) -> None:
            vistos.append("setup")

    monkeypatch.setattr(graph, "RedisSaver", Falso)
    graph.abrir_redis_saver("redis://localhost:6379")
    assert vistos == ["redis://localhost:6379", "enter", "setup"]


def test_observabilidad_nombra_openinference() -> None:
    fuente = _texto("app/observability.py")
    assert "OpenInference" in fuente
    assert "LangChainInstrumentor().instrument(" in fuente
    assert "OpenAIInstrumentor().instrument(" not in fuente
    assert 'ENDPOINT = "http://localhost:6006/v1/traces"' in fuente
    assert "def trace_llm" in fuente
    assert "@trace_llm" in _texto("modelo.py")
    assert "using_session(thread_id)" in _texto("app/worker.py")


def test_sin_clave_falsa_ni_modelo_local() -> None:
    for relativo in ("main.py", "app/main.py", "app/graph.py", "app/worker.py", "app/hitl.py", "modelo.py"):
        assert "sk-" not in _texto(relativo)
        assert "usar_modelo_local" not in _texto(relativo)


def test_el_alias_type_esta_en_el_grafo() -> None:
    arbol = ast.parse(_texto("app/graph.py"))
    assert any(isinstance(nodo, ast.TypeAlias) for nodo in arbol.body)
