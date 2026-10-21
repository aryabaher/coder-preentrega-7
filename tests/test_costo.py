import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.graph import costo_estimado, get_graph, limite_costo, medido, resume_graph, run_graph

PREGUNTA = "¿Cuántos días de vacaciones acumula en total un empleado en sus primeros 5 años?"


def _config(hilo: str) -> dict:
    return {"configurable": {"thread_id": hilo}}


def test_medido_suma_los_tokens_del_nodo() -> None:
    respuesta = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500},
        response_metadata={"model_name": "gpt-4o-mini"},
    )
    modelo_falso = GenericFakeChatModel(messages=iter([respuesta]))

    def nodo(state: dict) -> dict:
        modelo_falso.invoke("hola")
        return {"pasos": 1}

    salida = medido(nodo)({})  # type: ignore[arg-type]
    assert salida == {"pasos": 1, "input_tokens": 1200, "output_tokens": 300}


def test_costo_estimado_usa_el_precio_de_gpt_4o_mini() -> None:
    costo = costo_estimado({"input_tokens": 1_000_000, "output_tokens": 1_000_000})  # type: ignore[typeddict-item]
    assert costo == pytest.approx(0.75)


def test_limite_costo_lee_el_entorno(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COSTO_MAXIMO_USD", "0.002")
    assert limite_costo() == 0.002
    monkeypatch.setenv("COSTO_MAXIMO_USD", "mucho")
    assert limite_costo() == 0.01


def _gastar(monkeypatch: pytest.MonkeyPatch) -> None:
    """El costo estimado pasa el límite en cuanto hay un aporte."""
    from app import graph

    real = graph.costo_estimado

    def caro(state: dict) -> float:
        return 1.0 if state.get("contribuciones") else real(state)  # type: ignore[arg-type]

    monkeypatch.setattr(graph, "costo_estimado", caro)


def test_costo_alto_pausa_para_aprobar(monkeypatch: pytest.MonkeyPatch) -> None:
    _gastar(monkeypatch)
    monkeypatch.setenv("COSTO_MAXIMO_USD", "0.5")
    salida = run_graph(PREGUNTA, _config("caro"))
    assert salida["paused"] is True
    assert salida["interrupt"]["reason"] == "estimated cost above COSTO_MAXIMO_USD"
    assert salida["interrupt"]["estimated_cost_usd"] == 1.0
    assert salida["interrupt"]["limit_usd"] == 0.5
    valores = get_graph().get_state(_config("caro")).values
    assert [c["agente"] for c in valores["contribuciones"]] == ["investigador"]


def test_aprobar_el_costo_sigue_el_trabajo(monkeypatch: pytest.MonkeyPatch) -> None:
    _gastar(monkeypatch)
    monkeypatch.setenv("COSTO_MAXIMO_USD", "0.5")
    run_graph(PREGUNTA, _config("caro-si"))
    salida = resume_graph({"approved": True}, _config("caro-si"))
    assert salida["paused"] is False
    valores = get_graph().get_state(_config("caro-si")).values
    assert valores["cost_approved"] is True
    assert [c["agente"] for c in valores["contribuciones"]] == ["investigador", "analista"]


def test_rechazar_el_costo_cierra_con_lo_obtenido(monkeypatch: pytest.MonkeyPatch, dobles) -> None:
    llm, _, analista = dobles
    _gastar(monkeypatch)
    monkeypatch.setenv("COSTO_MAXIMO_USD", "0.5")
    run_graph(PREGUNTA, _config("caro-no"))
    salida = resume_graph({"approved": False}, _config("caro-no"))
    assert salida["paused"] is False
    assert analista.entradas == []
    assert "cortó la ejecución por costo" in str(llm.prompts[-1])
