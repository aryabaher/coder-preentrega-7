"""Analista: calcula con la calculadora segura y no busca información nueva."""

import ast
import operator as op
from collections.abc import Callable
from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

import modelo
from agents.texto import texto_de
from state import AgentState

PROMPT_ANALISTA = (
    "Sos un agente de análisis matemático. Tu única función es calcular usando tu "
    "herramienta a partir de los números que te den. NO busques información nueva."
)

_OPERADORES = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}
MAX_EXPONENTE = 100


def _evaluar_nodo(nodo: ast.expr) -> int | float:
    if isinstance(nodo, ast.Constant) and isinstance(nodo.value, (int, float)) and not isinstance(nodo.value, bool):
        return nodo.value
    if isinstance(nodo, ast.BinOp) and type(nodo.op) in _OPERADORES:
        derecha = nodo.right
        if isinstance(nodo.op, ast.Pow) and isinstance(derecha, ast.Constant) and isinstance(derecha.value, (int, float)) and abs(derecha.value) > MAX_EXPONENTE:
            raise ValueError(f"exponente fuera de rango (máximo {MAX_EXPONENTE})")
        cuenta = cast(Callable[[int | float, int | float], int | float], _OPERADORES[type(nodo.op)])
        return cuenta(_evaluar_nodo(nodo.left), _evaluar_nodo(nodo.right))
    if isinstance(nodo, ast.UnaryOp) and type(nodo.op) in _OPERADORES:
        signo = cast(Callable[[int | float], int | float], _OPERADORES[type(nodo.op)])
        return signo(_evaluar_nodo(nodo.operand))
    raise ValueError("Expresión no permitida")


@tool
def calculadora(expresion: str) -> str:
    """Evalúa una expresión matemática (+, -, *, /, **, paréntesis). Ejemplo: '5*14 + 5*21 + 2*28'."""
    texto = expresion.replace("$", "").strip()
    if not texto:
        return "Error al calcular '': la expresión está vacía. Pasá solo números y + - * / **, por ejemplo '5*14 + 5*21 + 2*28'."
    if "," in texto or _tiene_miles(texto):
        return (
            f"Error al calcular '{expresion}': el número tiene separador de miles. "
            "Escribilo sin puntos ni comas, por ejemplo 25000, y volvé a llamar a calculadora."
        )
    try:
        resultado = _evaluar_nodo(ast.parse(texto, mode="eval").body)
    except ZeroDivisionError:
        return f"Error al calcular '{expresion}': división por cero. Corregí la expresión y volvé a llamar a calculadora."
    except (ValueError, SyntaxError, KeyError) as exc:
        detalle = exc.msg if isinstance(exc, SyntaxError) else exc
        return (
            f"Error al calcular '{expresion}': {detalle}. "
            "Solo números y + - * / ** con paréntesis. Corregí la expresión y volvé a llamar a calculadora."
        )
    if isinstance(resultado, float) and resultado.is_integer():
        resultado = int(resultado)
    return f"Resultado: {resultado}"


def _tiene_miles(texto: str) -> bool:
    import re

    return re.search(r"(?<![\d.])[1-9]\d{0,2}(?:\.\d{3})+(?![\d.])", texto) is not None


def asegurar_agente() -> Any:
    """create_agent con la calculadora. Se arma una sola vez."""
    global agente_analista
    if agente_analista is None:
        agente_analista = create_agent(
            model=modelo.obtener_llm(),
            tools=[calculadora],
            system_prompt=PROMPT_ANALISTA,
        )
    return agente_analista


agente_analista: Any = None


def nodo_analista(state: AgentState) -> dict[str, Any]:
    """Calcula. Recibe solo el último mensaje, no toda la conversación."""
    tarea = state["messages"][-1].content  # contexto acotado, no todo el historial
    agente_analista = asegurar_agente()
    resultado = modelo.invocar(lambda: agente_analista.invoke({"messages": [HumanMessage(content=texto_de(tarea))]}))
    mensaje = resultado["messages"][-1]
    respuesta = (
        "ERROR: output truncado: analista cortó la respuesta por max_tokens. Pedile una respuesta más corta."
        if modelo.es_truncada(mensaje)
        else texto_de(mensaje.content)
    )
    return {
        "messages": [AIMessage(content=respuesta, name="analista")],
        "contribuciones": [{"agente": "analista", "aporte": respuesta}],
        "pasos": state.get("pasos", 0) + 1,
    }
