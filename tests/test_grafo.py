import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.research_agent import nodo_investigador
from app.graph import (
    MAX_PASOS,
    DecisionSupervisor,
    _pide_calculo,
    enrutar,
    estado_inicial,
    get_graph,
    grafo,
)
from tests.conftest import EspecialistaFalso, LLMFalso

PREGUNTA = (
    "Un empleado va a cumplir 12 años en TechCorp. Tuvo 14 días de vacaciones los primeros "
    "5 años, 21 los siguientes 5, y ahora tiene derecho a 28. ¿Cuántos días de vacaciones "
    "acumuló en total en esos 12 años?"
)


def _correr(pregunta: str = PREGUNTA, hilo: str = "grafo") -> dict:
    return get_graph().invoke(
        estado_inicial(pregunta),
        {"configurable": {"thread_id": hilo}, "recursion_limit": 25},
    )


def test_topologia() -> None:
    dibujo = get_graph().get_graph()
    assert set(dibujo.nodes) == {
        "__start__",
        "supervisor",
        "investigador",
        "analista",
        "aprobacion",
        "ejecutor",
        "sintesis",
        "__end__",
    }
    aristas = {(a.source, a.target) for a in dibujo.edges}
    for arista in [
        ("__start__", "supervisor"),
        ("supervisor", "investigador"),
        ("supervisor", "analista"),
        ("supervisor", "aprobacion"),
        ("supervisor", "ejecutor"),
        ("supervisor", "sintesis"),
        ("investigador", "supervisor"),
        ("analista", "supervisor"),
        ("aprobacion", "ejecutor"),
        ("aprobacion", "supervisor"),
        ("aprobacion", "sintesis"),
        ("ejecutor", "supervisor"),
        ("sintesis", "__end__"),
    ]:
        assert arista in aristas
    assert ("investigador", "analista") not in aristas
    assert grafo is not None


def test_enrutar_manda_finish_a_sintesis() -> None:
    assert enrutar({"next_agent": "FINISH"}) == "sintesis"  # type: ignore[typeddict-item]
    assert enrutar({"next_agent": "investigador"}) == "investigador"  # type: ignore[typeddict-item]
    assert enrutar({"next_agent": "aprobacion"}) == "aprobacion"  # type: ignore[typeddict-item]
    assert enrutar({"next_agent": "ejecutor"}) == "ejecutor"  # type: ignore[typeddict-item]


def test_delega_busca_calcula_y_sintetiza(dobles: tuple[LLMFalso, EspecialistaFalso, EspecialistaFalso]) -> None:
    llm, investigador, analista = dobles
    salida = _correr()
    assert [c["agente"] for c in salida["contribuciones"]] == ["investigador", "analista"]
    assert salida["contribuciones"][0]["aporte"].startswith("14 días")
    assert salida["contribuciones"][1]["aporte"] == "Resultado: 231"
    assert salida["task_completed"] is True
    assert salida["pasos"] == 2
    assert salida["critical"] is False
    assert salida["messages"][-1].content == "En 12 años acumuló 231 días de vacaciones."
    assert salida["messages"][-1].name == "supervisor"
    assert investigador.entradas[0]["messages"][0].content == PREGUNTA
    assert "14 días" in analista.entradas[0]["messages"][0].content
    assert "Pregunta actual:" in str(llm.prompts[0])


@pytest.mark.parametrize(
    ("pregunta", "esperado"),
    [
        ("¿Cuántos días de vacaciones corresponden con 5 años de antigüedad?", False),
        ("¿Cuántos días de teletrabajo por semana permite la política de TechCorp?", False),
        (PREGUNTA, True),
        ("¿Y cuántos días acumula en total en esos 5 años?", True),
        ("¿Cuántos días corresponden con 5 años? Registrá el ajuste de nómina para el legajo 1042.", False),
    ],
)
def test_pide_calculo_solo_si_hay_una_cuenta(pregunta: str, esperado: bool) -> None:
    assert _pide_calculo(pregunta) is esperado


def test_una_pregunta_de_dato_no_llama_al_analista(
    dobles: tuple[LLMFalso, EspecialistaFalso, EspecialistaFalso],
) -> None:
    llm, investigador, analista = dobles

    def finish_con_dato(mensajes: list[dict[str, str]], *args: object, **kwargs: object) -> DecisionSupervisor:
        if "- investigador:" in mensajes[0]["content"]:
            return DecisionSupervisor(next="FINISH", razon="ya está el dato")
        return DecisionSupervisor(next="investigador", razon="falta el dato")

    llm.supervisor.invoke = finish_con_dato  # type: ignore[method-assign]
    salida = _correr("¿Cuántos días de vacaciones corresponden con 5 años de antigüedad?", hilo="dato")
    assert [c["agente"] for c in salida["contribuciones"]] == ["investigador"]
    assert analista.entradas == []
    assert len(investigador.entradas) == 1


def test_el_analista_recibe_la_pregunta_y_los_datos(
    dobles: tuple[LLMFalso, EspecialistaFalso, EspecialistaFalso],
) -> None:
    _, _, analista = dobles
    _correr(hilo="tarea-analista")
    tarea = analista.entradas[0]["messages"][0].content
    assert f"Pregunta actual: {PREGUNTA}" in tarea
    assert "Datos del equipo:\n- investigador: 14 días, 21 días y 28 días." in tarea


def test_el_supervisor_no_manda_al_analista_sin_cuenta(
    dobles: tuple[LLMFalso, EspecialistaFalso, EspecialistaFalso],
) -> None:
    llm, _, analista = dobles

    def siempre_analista(mensajes: list[dict[str, str]], *args: object, **kwargs: object) -> DecisionSupervisor:
        return DecisionSupervisor(next="analista", razon="calcular por las dudas")

    llm.supervisor.invoke = siempre_analista  # type: ignore[method-assign]
    salida = _correr("¿Cuántos días de vacaciones corresponden al tramo de 5 años de antigüedad?", hilo="sin-cuenta")
    assert [c["agente"] for c in salida["contribuciones"]] == ["investigador"]
    assert analista.entradas == []


def test_la_sintesis_normal_no_menciona_aprobaciones(
    dobles: tuple[LLMFalso, EspecialistaFalso, EspecialistaFalso],
) -> None:
    llm, _, _ = dobles
    _correr(hilo="sin-aprobacion")
    prompt = str(llm.prompts[-1])
    assert "Respondé solo lo que se pregunta." in prompt
    assert "acción pausada" not in prompt
    assert "aprob" not in prompt.lower()


def test_el_especialista_recibe_solo_el_ultimo_mensaje(
    dobles: tuple[LLMFalso, EspecialistaFalso, EspecialistaFalso],
) -> None:
    _, investigador, _ = dobles
    estado = estado_inicial(PREGUNTA)
    estado["messages"] = [HumanMessage(content=PREGUNTA), AIMessage(content="instruccion puntual", name="supervisor")]
    nodo_investigador(estado)  # type: ignore[arg-type]
    assert investigador.entradas[0]["messages"][0].content == "instruccion puntual"


def test_los_hilos_no_comparten_contribuciones() -> None:
    una = _correr(hilo="hilo-a")
    otra = _correr(hilo="hilo-b")
    assert len(una["contribuciones"]) == len(otra["contribuciones"]) == 2


def test_el_checkpointer_recupera_el_hilo() -> None:
    _correr(hilo="hilo-persistido")
    snap = get_graph().get_state({"configurable": {"thread_id": "hilo-persistido"}})
    assert snap.values["messages"][-1].content == "En 12 años acumuló 231 días de vacaciones."
    assert get_graph().get_state({"configurable": {"thread_id": "otro"}}).values.get("messages") in (None, [])


def test_respuesta_truncada_no_se_toma_como_dato(dobles: tuple[LLMFalso, EspecialistaFalso, EspecialistaFalso]) -> None:
    _, investigador, _ = dobles

    def cortada(entrada: dict[str, object], *args: object, **kwargs: object) -> dict[str, list[AIMessage]]:
        return {"messages": [AIMessage(content="14 días y", response_metadata={"finish_reason": "length"})]}

    investigador.invoke = cortada  # type: ignore[method-assign]
    salida = nodo_investigador(estado_inicial(PREGUNTA))  # type: ignore[arg-type]
    assert salida["contribuciones"][0]["aporte"].startswith(
        "ERROR: output truncado: investigador cortó la respuesta por max_tokens."
    )


def test_max_pasos_corta(dobles: tuple[LLMFalso, EspecialistaFalso, EspecialistaFalso], monkeypatch) -> None:
    llm, investigador, analista = dobles
    monkeypatch.setattr("app.graph.MAX_PASOS", 0)
    assert MAX_PASOS == 6

    def siempre_investigador(mensajes: list[dict[str, str]], *args: object, **kwargs: object) -> DecisionSupervisor:
        return DecisionSupervisor(next="investigador", razon="otra vez")

    llm.supervisor.invoke = siempre_investigador  # type: ignore[method-assign]
    salida = _correr(hilo="tope")
    assert salida["task_completed"] is True
    assert investigador.entradas == []
    assert analista.entradas == []
