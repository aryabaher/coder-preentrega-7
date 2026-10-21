from app.graph import correr, estado_inicial, leer_interrupcion, obtener, reanudar
from app.hitl import efectos, es_critica

PREGUNTA = (
    "Un empleado va a cumplir 12 años en TechCorp. Tuvo 14 días de vacaciones los primeros "
    "5 años, 21 los siguientes 5, y ahora tiene derecho a 28. ¿Cuántos días de vacaciones "
    "acumuló en total en esos 12 años?"
)
PREGUNTA_CRITICA = PREGUNTA + " Registrá el ajuste de nómina con registrar_ajuste_nomina."


def test_la_pregunta_de_politicas_no_es_critica() -> None:
    assert es_critica(PREGUNTA) is False
    assert es_critica("Esto tiene un costo muy alto, no lo ejecutes todavía.") is True


def test_la_critica_pausa_antes_del_efecto() -> None:
    config = {"configurable": {"thread_id": "pausa"}}
    salida = correr(estado_inicial(PREGUNTA_CRITICA), config, 25)
    assert salida["paused"] is True
    assert salida["interrupt"]["tool"] == "registrar_ajuste_nomina"
    assert leer_interrupcion(config)["tool"] == "registrar_ajuste_nomina"
    assert efectos == []
    assert estado_inicial(PREGUNTA_CRITICA)["critica"] is True


def test_aprobar_ejecuta_registrar_ajuste_nomina() -> None:
    config = {"configurable": {"thread_id": "aprueba"}}
    correr(estado_inicial(PREGUNTA_CRITICA), config, 25)
    salida = reanudar({"approved": True}, config, 25)
    assert salida["paused"] is False
    assert efectos and efectos[0]["herramienta"] == "registrar_ajuste_nomina"
    aportes = obtener().get_state(config).values["contribuciones"]
    assert any(c["agente"] == "ejecutor" and c["aporte"].startswith("Ajuste registrado:") for c in aportes)


def test_rechazar_no_ejecuta_la_herramienta() -> None:
    config = {"configurable": {"thread_id": "rechaza"}}
    correr(estado_inicial(PREGUNTA_CRITICA), config, 25)
    salida = reanudar({"approved": False}, config, 25)
    assert salida["paused"] is False
    assert efectos == []
    aportes = obtener().get_state(config).values["contribuciones"]
    assert all(c["agente"] != "ejecutor" for c in aportes)
    assert any(c["agente"] == "aprobacion" for c in aportes)
