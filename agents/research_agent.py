"""Investigador: busca en las políticas internas y no calcula."""

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

import modelo
from agents.texto import texto_de
from rag import rag_system
from state import AgentState

PROMPT_INVESTIGADOR = (
    "Sos un agente de investigación. Tu única función es buscar datos concretos en "
    "las políticas internas de TechCorp usando tu herramienta. NO hagas cálculos. "
    "Respondé de forma breve con los datos encontrados y su fuente."
)


@tool
def buscar_en_politicas_internas(pregunta: str) -> str:
    """Busca información en las políticas internas de TechCorp (vacaciones, teletrabajo,
    seguridad informática, onboarding). Usar para cualquier pregunta sobre reglas o
    números concretos de la empresa."""
    if not pregunta.strip():
        return (
            "No se encontró información relevante en las políticas. "
            "Pasá la pregunta con texto, por ejemplo 'días de vacaciones por antigüedad'."
        )
    resultados = rag_system.obtener_top_k(pregunta)
    if not resultados:
        return "No se encontró información relevante en las políticas."
    return "\n\n".join(f"[Fuente: {r['fuente']}] {r['contenido']}" for r in resultados)


def asegurar_agente() -> Any:
    """create_agent con la herramienta de búsqueda. Se arma una sola vez."""
    global agente_investigador
    if agente_investigador is None:
        agente_investigador = create_agent(
            model=modelo.obtener_llm(),
            tools=[buscar_en_politicas_internas],
            system_prompt=PROMPT_INVESTIGADOR,
        )
    return agente_investigador


agente_investigador: Any = None


def nodo_investigador(state: AgentState) -> dict[str, Any]:
    """Busca el dato. Recibe solo el último mensaje, no toda la conversación."""
    tarea = state["messages"][-1].content  # contexto acotado, no todo el historial
    agente_investigador = asegurar_agente()
    resultado = modelo.invocar(lambda: agente_investigador.invoke({"messages": [HumanMessage(content=texto_de(tarea))]}))
    mensaje = resultado["messages"][-1]
    respuesta = (
        "ERROR: output truncado: investigador cortó la respuesta por max_tokens. Pedile una respuesta más corta."
        if modelo.es_truncada(mensaje)
        else texto_de(mensaje.content)
    )
    return {
        "messages": [AIMessage(content=respuesta, name="investigador")],
        "contribuciones": [{"agente": "investigador", "aporte": respuesta}],
        "pasos": state.get("pasos", 0) + 1,
    }
