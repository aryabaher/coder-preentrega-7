from app.graph import es_critica, get_graph, leer_interrupcion, resume_graph, run_graph
from app.hitl import TOOL_NAME, side_effects

PREGUNTA = (
    "Un empleado va a cumplir 12 años en TechCorp. Tuvo 14 días de vacaciones los primeros "
    "5 años, 21 los siguientes 5, y ahora tiene derecho a 28. ¿Cuántos días de vacaciones "
    "acumuló en total en esos 12 años?"
)
PREGUNTA_CRITICA = PREGUNTA + " Registrá el ajuste de nómina con register_payroll_adjustment."


def _config(hilo: str) -> dict:
    return {"configurable": {"thread_id": hilo}}


def test_la_pregunta_de_politicas_no_es_critica() -> None:
    assert es_critica(PREGUNTA) is False
    assert es_critica("Esto tiene un costo muy alto, no lo ejecutes todavía.") is True
    assert es_critica("Registrá el ajuste de nómina") is True


def test_la_critica_pausa_antes_del_efecto() -> None:
    salida = run_graph(PREGUNTA_CRITICA, _config("pausa"), 25)
    assert salida["paused"] is True
    assert salida["interrupt"]["tool"] == TOOL_NAME
    assert salida["interrupt"]["reason"] == f"side-effect tool {TOOL_NAME}"
    assert leer_interrupcion(_config("pausa"))["tool"] == TOOL_NAME
    assert side_effects == []


def test_aprobar_ejecuta_la_herramienta(dobles) -> None:
    llm, _, _ = dobles
    run_graph(PREGUNTA_CRITICA, _config("aprueba"), 25)
    salida = resume_graph({"approved": True}, _config("aprueba"), 25)
    assert salida["paused"] is False
    assert side_effects and side_effects[0]["tool"] == TOOL_NAME
    valores = get_graph().get_state(_config("aprueba")).values
    assert valores["side_effect_done"] is True
    assert valores["side_effect_result"].startswith("Adjustment registered:")
    assert "La persona aprobó y se registró el ajuste de nómina" in str(llm.prompts[-1])


def test_rechazar_no_ejecuta_la_herramienta(dobles) -> None:
    llm, _, _ = dobles
    run_graph(PREGUNTA_CRITICA, _config("rechaza"), 25)
    salida = resume_graph({"approved": False}, _config("rechaza"), 25)
    assert salida["paused"] is False
    assert side_effects == []
    valores = get_graph().get_state(_config("rechaza")).values
    assert valores["approved"] is False
    assert valores["side_effect_done"] is False
    assert "rechazó el ajuste" in str(llm.prompts[-1])
