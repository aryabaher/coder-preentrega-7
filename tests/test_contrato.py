import ast
from pathlib import Path

from app import graph

RAIZ = Path(__file__).resolve().parents[1]


def _texto(relativo: str) -> str:
    return (RAIZ / relativo).read_text(encoding="utf-8")


def test_la_api_tipa_la_salida_y_el_404() -> None:
    fuente = _texto("app/main.py")
    assert "response_model=TaskResponse" in fuente
    assert "response_model=TaskStatus" in fuente
    assert 'detail="Job not found"' in fuente
    assert "sk-" not in fuente
    assert ".invoke(" not in fuente
    assert "GET /tasks/{id}" in fuente
    assert "POST /tasks/{id}/approve" in fuente


def test_el_worker_cubre_el_ciclo_y_no_bloquea_el_loop() -> None:
    fuente = _texto("app/worker.py")
    for estado in ("PENDING", "RUNNING", "FAILED", "DONE", "AWAITING_APPROVAL"):
        assert f'"{estado}"' in fuente
    assert "asyncio.to_thread" in fuente
    assert 'QUEUE_NAME = "tasks_queue"' in fuente
    assert 'STATUS_PREFIX = "status:"' in fuente
    assert "blpop(QUEUE_NAME, timeout=0)" in fuente


def test_el_grafo_usa_redis_saver(monkeypatch) -> None:
    from langgraph.checkpoint.memory import MemorySaver

    fuente = _texto("app/graph.py")
    assert "RedisSaver" in fuente
    assert "checkpointer=checkpointer" in fuente
    assert "thread_id" in _texto("app/worker.py")
    llamadas: list[object] = []

    def falso(url: str | None = None) -> MemorySaver:
        llamadas.append(url)
        return MemorySaver()

    monkeypatch.setattr(graph, "abrir_redis_saver", falso)
    graph.compilar()
    assert llamadas == [None]


def test_abrir_redis_saver_hace_setup(monkeypatch) -> None:
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


def test_hitl_nombra_interrupt_y_la_herramienta() -> None:
    fuente = _texto("app/hitl.py")
    assert "interrupt(" in fuente
    assert "registrar_ajuste_nomina" in fuente
    assert "POST /tasks/{id}/approve" in fuente


def test_observabilidad_nombra_openinference() -> None:
    fuente = _texto("app/observability.py")
    assert "OpenInference" in fuente
    assert "LangChainInstrumentor" in fuente
    assert "OpenAIInstrumentor" in fuente
    assert 'ENDPOINT = "http://localhost:6006/v1/traces"' in fuente
    assert "def trazar_llm" in fuente
    assert "@trazar_llm" in _texto("modelo.py")


def test_main_no_tiene_clave_falsa() -> None:
    for relativo in ("main.py", "app/main.py", "app/graph.py", "app/worker.py", "modelo.py"):
        assert "sk-" not in _texto(relativo)
        assert "usar_modelo_local" not in _texto(relativo)


def test_el_alias_type_esta_en_el_grafo() -> None:
    arbol = ast.parse(_texto("app/graph.py"))
    assert any(isinstance(nodo, ast.TypeAlias) for nodo in arbol.body)
