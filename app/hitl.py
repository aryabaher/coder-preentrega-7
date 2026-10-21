"""Nodo de aprobación humana: interrupt pausa el grafo hasta POST /tasks/{id}/approve."""

from __future__ import annotations

import unicodedata
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.types import interrupt

from agents.texto import texto_de
from state import AgentState

# El body de ApprovalRequest manda esta clave. interrupt() la devuelve al reanudar.
CLAVE_APROBACION = "approved"

MARCAS_CRITICAS = (
    "registrar_ajuste_nomina",
    "ajuste de nomina",
    "registrar ajuste",
    "efecto secundario",
    "costo muy alto",
)

efectos: list[dict[str, str]] = []


def es_critica(texto: str) -> bool:
    """True si la tarea pide una herramienta con efectos secundarios o un costo muy alto."""
    normalizado = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
    return any(marca in normalizado for marca in MARCAS_CRITICAS)


@tool
def registrar_ajuste_nomina(detalle: str) -> str:
    """Registra un ajuste de nómina. Efecto secundario: queda asentado y no se deshace."""
    texto = detalle.strip()
    if not texto:
        return "No se registró el ajuste: el detalle está vacío."
    efectos.append({"herramienta": "registrar_ajuste_nomina", "detalle": texto})
    return f"Ajuste registrado: {texto}"


def nodo_aprobacion(state: AgentState) -> dict[str, Any]:
    """Pausa con interrupt. No corre la herramienta: eso queda del otro lado de la aprobación."""
    pregunta = texto_de(state["messages"][0].content)
    decision = interrupt(
        {
            "reason": "side-effect tool registrar_ajuste_nomina",
            "tool": "registrar_ajuste_nomina",
            "query": pregunta,
        }
    )
    if not _aprobado(decision):
        aporte = "Rechazado: no se ejecuta registrar_ajuste_nomina."
        return {
            "aprobado": False,
            "next_agent": "FINISH",
            "messages": [AIMessage(content=aporte, name="aprobacion")],
            "contribuciones": [{"agente": "aprobacion", "aporte": aporte}],
        }
    return {"aprobado": True, "next_agent": "ejecutor"}


def nodo_ejecutor(state: AgentState) -> dict[str, Any]:
    """Corre registrar_ajuste_nomina solo si la persona ya aprobó."""
    if state.get("aprobado") is not True:
        from errors import DecisionInvalidaError

        raise DecisionInvalidaError(
            "registrar_ajuste_nomina sin aprobación. El grafo tiene que pasar por interrupt antes."
        )
    detalle = texto_de(state["messages"][0].content)
    resultado = str(registrar_ajuste_nomina.invoke({"detalle": detalle}))
    return {
        "messages": [AIMessage(content=resultado, name="ejecutor")],
        "contribuciones": [{"agente": "ejecutor", "aporte": resultado}],
        "pasos": state.get("pasos", 0) + 1,
    }


def seguir_tras_aprobacion(state: AgentState) -> Literal["ejecutor", "sintesis"]:
    """La arista crítica: con aprobación sigue la herramienta; si no, cierra sin efecto."""
    if state.get("aprobado") is True:
        return "ejecutor"
    return "sintesis"


def _aprobado(decision: object) -> bool:
    if isinstance(decision, dict):
        return bool(decision.get(CLAVE_APROBACION))
    return bool(decision)
