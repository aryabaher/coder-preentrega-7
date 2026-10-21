"""Orquestador del módulo 6 con checkpoints en RedisSaver y pausa antes del efecto."""

import os
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field, ValidationError

import modelo
from agents.analyst_agent import nodo_analista
from agents.research_agent import nodo_investigador
from agents.texto import texto_de
from app.hitl import es_critica, nodo_aprobacion, nodo_ejecutor, seguir_tras_aprobacion
from errors import DecisionInvalidaError, SolicitudInvalidaError
from state import AgentState, Solicitud

MAX_PASOS = 6
RECURSION_LIMIT = 25

SUPERVISOR_PROMPT = """Sos el Supervisor de un equipo con dos especialistas:
- investigador: busca datos concretos en las políticas internas de TechCorp.
- analista: hace cálculos matemáticos a partir de datos ya obtenidos.

Rúbrica: no des por terminada la tarea hasta cumplir las cuatro reglas.
Reglas:
1. Si falta un dato de la empresa, mandá a 'investigador'.
2. Si ya se buscó el dato pero falta calcular, mandá a 'analista'.
3. Si ya se buscó Y se calculó lo necesario, respondé 'FINISH'.
4. Nunca repitas el mismo agente dos veces seguidas si ya cumplió su parte.
5. Si la tarea pide registrar un ajuste de nómina u otra acción con efectos secundarios, no la des por terminada: falta la aprobación humana.

Contribuciones hasta ahora:
{contribuciones}
"""

_contextos: list[Any] = []
_compilado: CompiledStateGraph | None = None


class DecisionSupervisor(BaseModel):
    next: Literal["investigador", "analista", "FINISH"] = Field(
        description="Próximo agente a invocar, o FINISH si la tarea ya está completa"
    )
    razon: str = Field(description="Breve justificación de la decisión")


class Ruta(BaseModel):
    next: Literal["investigador", "analista", "aprobacion", "ejecutor", "FINISH"]
    razon: str


def nodo_supervisor(state: AgentState) -> dict[str, Any]:
    """Router: quién interviene ahora, o FINISH si la tarea ya está completa."""
    if state.get("pasos", 0) >= MAX_PASOS:
        return {"next_agent": "FINISH", "task_completed": True}

    contribuciones_texto = "\n".join(
        f"- {c['agente']}: {c['aporte'][:200]}" for c in state.get("contribuciones", [])
    ) or "(ninguna todavía)"
    pregunta_original = texto_de(state["messages"][0].content)
    prompt_sistema = SUPERVISOR_PROMPT.format(contribuciones=contribuciones_texto)
    llm_supervisor = modelo.obtener_llm().with_structured_output(DecisionSupervisor)
    decision = modelo.invocar(
        lambda: llm_supervisor.invoke(
            [
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": f"Tarea original: {pregunta_original}"},
            ]
        )
    )
    if not isinstance(decision, DecisionSupervisor):
        raise DecisionInvalidaError(
            "Decisión inválida del supervisor: "
            f"{decision!r}. No se adivina una ruta: revisá el modelo configurado en LLM_PROVIDER."
        )
    ruta = aplicar_guardas(state, decision, pregunta_original)
    return {"next_agent": ruta.next, "task_completed": ruta.next == "FINISH"}


def _cumplio(state: AgentState, agente: str) -> bool:
    """True si ese agente ya dejó un aporte usable. Un error o un 'no encontrado' no cuenta."""
    return any(
        c["agente"] == agente and not c["aporte"].startswith(("ERROR", "No se encontró"))
        for c in state.get("contribuciones", [])
    )


def _guardas_equipo(state: AgentState, decision: DecisionSupervisor, pregunta: str) -> DecisionSupervisor:
    """FINISH solo después de buscar y, si la pregunta pide una cuenta, de calcular."""
    hechos = {c["agente"] for c in state.get("contribuciones", [])}
    if decision.next == "investigador" and _cumplio(state, "investigador"):
        if "analista" not in hechos and _pide_calculo(pregunta):
            return DecisionSupervisor(
                next="analista", razon="[guarda] el investigador ya cumplió su parte: sigue el analista."
            )
        return DecisionSupervisor(next="FINISH", razon="[guarda] el investigador ya cumplió su parte.")
    if decision.next == "analista" and _cumplio(state, "analista"):
        return DecisionSupervisor(next="FINISH", razon="[guarda] el analista ya cumplió su parte.")
    if decision.next != "FINISH":
        return decision
    if "investigador" not in hechos:
        return DecisionSupervisor(
            next="investigador", razon="[guarda] FINISH sin datos de las políticas: primero el investigador."
        )
    if "analista" not in hechos and _pide_calculo(pregunta):
        return DecisionSupervisor(next="analista", razon="[guarda] FINISH sin el cálculo: primero el analista.")
    return decision


def _lista_para_efecto(state: AgentState, pregunta: str) -> bool:
    if not _cumplio(state, "investigador"):
        return False
    if _pide_calculo(pregunta) and not _cumplio(state, "analista"):
        return False
    return True


def _guarda_critica(state: AgentState, decision: DecisionSupervisor, pregunta: str) -> Ruta:
    """Antes de la herramienta con efectos secundarios, la ruta es el nodo de aprobación."""
    if not state.get("critica"):
        return Ruta(next=decision.next, razon=decision.razon)
    if state.get("aprobado") is False or _cumplio(state, "ejecutor"):
        return Ruta(next="FINISH", razon="[guarda] la tarea crítica ya se resolvió.")
    if state.get("aprobado") is True:
        return Ruta(next="ejecutor", razon="[guarda] la persona aprobó registrar_ajuste_nomina.")
    if _lista_para_efecto(state, pregunta):
        return Ruta(
            next="aprobacion",
            razon="[guarda] herramienta con efectos secundarios: hace falta aprobación.",
        )
    return Ruta(next=decision.next, razon=decision.razon)


def aplicar_guardas(state: AgentState, decision: DecisionSupervisor, pregunta: str) -> Ruta:
    """Las guardas del equipo y, si la tarea es crítica, la pausa antes del efecto."""
    return _guarda_critica(state, _guardas_equipo(state, decision, pregunta), pregunta)


def _pide_calculo(pregunta: str) -> bool:
    texto = pregunta.lower()
    return any(palabra in texto for palabra in ("cuánto", "cuanto", "cuántos", "cuantos", "total", "acumul", "calcul"))


def enrutar(state: AgentState) -> Literal["investigador", "analista", "aprobacion", "ejecutor", "sintesis"]:
    if state["next_agent"] == "FINISH":
        return "sintesis"
    destino = state["next_agent"]
    if destino not in ("investigador", "analista", "aprobacion", "ejecutor"):
        raise DecisionInvalidaError(
            f"Decisión inválida del supervisor: {destino!r}. "
            "Tiene que ser investigador, analista, aprobacion, ejecutor o FINISH."
        )
    return destino  # type: ignore[return-value]


def nodo_sintesis(state: AgentState) -> dict[str, Any]:
    """Redacta la respuesta final con la pregunta y las contribuciones. No busca ni calcula."""
    contribuciones_texto = "\n".join(f"- [{c['agente']}] {c['aporte']}" for c in state.get("contribuciones", []))
    pregunta_original = texto_de(state["messages"][0].content)
    prompt = (
        f"Pregunta original: {pregunta_original}\n\nAportes del equipo:\n{contribuciones_texto}\n\n"
        "Redactá una respuesta final clara y breve combinando estos aportes."
    )
    llm = modelo.obtener_llm()
    respuesta_final = modelo.invocar(lambda: llm.invoke(prompt))
    return {"messages": [AIMessage(content=texto_de(respuesta_final.content), name="supervisor")]}


grafo = StateGraph(AgentState)
grafo.add_node("supervisor", nodo_supervisor)
grafo.add_node("investigador", nodo_investigador)
grafo.add_node("analista", nodo_analista)
grafo.add_node("aprobacion", nodo_aprobacion)
grafo.add_node("ejecutor", nodo_ejecutor)
grafo.add_node("sintesis", nodo_sintesis)

grafo.add_edge(START, "supervisor")
grafo.add_conditional_edges(
    "supervisor",
    enrutar,
    {
        "investigador": "investigador",
        "analista": "analista",
        "aprobacion": "aprobacion",
        "ejecutor": "ejecutor",
        "sintesis": "sintesis",
    },
)
grafo.add_edge("investigador", "supervisor")
grafo.add_edge("analista", "supervisor")
# La arista que sale de aprobacion solo llega al efecto si la persona aprobó.
grafo.add_conditional_edges(
    "aprobacion",
    seguir_tras_aprobacion,
    {"ejecutor": "ejecutor", "sintesis": "sintesis"},
)
grafo.add_edge("ejecutor", "supervisor")
grafo.add_edge("sintesis", END)

type Orquestador = CompiledStateGraph[AgentState, None, AgentState, AgentState]


def abrir_redis_saver(url: str | None = None) -> RedisSaver:
    """Checkpointer de los hilos. setup() crea los índices de búsqueda en Redis."""
    destino = (url or os.getenv("REDIS_URL") or "redis://localhost:6379").strip()
    contexto = RedisSaver.from_conn_string(destino)
    saver = contexto.__enter__()
    saver.setup()
    _contextos.append(contexto)
    return saver


def compilar(checkpointer: BaseCheckpointSaver | None = None) -> Orquestador:
    """Una sola compilación. Sin argumento, el checkpointer es RedisSaver."""
    if checkpointer is None:
        checkpointer = abrir_redis_saver()
    return grafo.compile(checkpointer=checkpointer)


def usar(compilado: Any) -> None:
    global _compilado
    _compilado = compilado


def esta_listo() -> bool:
    return _compilado is not None


def obtener() -> Any:
    if _compilado is None:
        raise RuntimeError("El grafo no está compilado. Llamá a compilar() con RedisSaver o a usar().")
    return _compilado


def validar_pregunta(texto: str) -> str:
    """Pydantic antes del grafo: una pregunta vacía no gasta llamadas."""
    try:
        return Solicitud(texto=texto).texto
    except ValidationError as exc:
        campos = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise SolicitudInvalidaError(f"Solicitud inválida: {campos}") from exc


def estado_inicial(pregunta: str) -> AgentState:
    texto = validar_pregunta(pregunta)
    return {
        "messages": [HumanMessage(content=texto)],
        "next_agent": None,
        "contribuciones": [],
        "pasos": 0,
        "task_completed": False,
        "critica": es_critica(texto),
        "aprobado": None,
    }


def leer_interrupcion(config: dict[str, Any]) -> dict[str, Any] | None:
    """Payload de interrupt, si el grafo quedó pausado en el nodo de aprobación."""
    snap = obtener().get_state(config)
    crudos: list[Any] = list(getattr(snap, "interrupts", None) or ())
    if not crudos:
        for tarea in getattr(snap, "tasks", ()) or ():
            crudos.extend(getattr(tarea, "interrupts", ()) or ())
    if not crudos:
        return None
    valor = getattr(crudos[0], "value", crudos[0])
    if isinstance(valor, dict):
        return valor
    return {"reason": str(valor)}


def _empaquetar(resultado: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    pausa = leer_interrupcion(config)
    if pausa is not None:
        return {"paused": True, "interrupt": pausa, "result": None}
    mensajes = resultado.get("messages") or []
    texto = texto_de(mensajes[-1].content) if mensajes else ""
    return {"paused": False, "interrupt": None, "result": texto}


def correr(entrada: AgentState, config: dict[str, Any], recursion_limit: int = RECURSION_LIMIT) -> dict[str, Any]:
    """Corre el grafo. recursion_limit llega al invoke: un turno de N nodos pide al menos N+1."""
    conf = {**config, "recursion_limit": recursion_limit}
    resultado = obtener().invoke(entrada, conf)
    return _empaquetar(resultado, conf)


def reanudar(decision: dict[str, Any], config: dict[str, Any], recursion_limit: int = RECURSION_LIMIT) -> dict[str, Any]:
    """Sigue después de interrupt. El dict es lo que interrupt() devuelve en el nodo."""
    conf = {**config, "recursion_limit": recursion_limit}
    resultado = obtener().invoke(Command(resume=decision), conf)
    return _empaquetar(resultado, conf)
