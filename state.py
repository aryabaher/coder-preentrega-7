"""Estado compartido del orquestador. Hereda messages y registra quién aportó qué."""

import operator
from typing import Annotated, Dict, List, Optional

from langgraph.graph import MessagesState
from pydantic import BaseModel, ConfigDict, Field


class AgentState(MessagesState):
    """Hereda 'messages' (con su propio reducer add_messages) y suma campos propios.

    messages acumula la conversación del hilo. El resto vale para un solo turno:
    cada pregunta nueva lo reinicia (contribuciones y tokens con Overwrite).
    Los campos de la aprobación humana van en inglés porque los lee app/hitl.py.
    """

    next_agent: Optional[str]
    contribuciones: Annotated[List[Dict[str, str]], operator.add]
    pasos: int
    task_completed: bool
    critical: bool
    approval_reason: Optional[str]
    approved: Optional[bool]
    cost_approved: Optional[bool]
    side_effect_done: bool
    side_effect_result: Optional[str]
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]
    estimated_cost_usd: Optional[float]
    cost_limit_usd: Optional[float]


class Solicitud(BaseModel):
    """La pregunta del usuario, validada antes de entrar al grafo."""

    model_config = ConfigDict(str_strip_whitespace=True)

    texto: str = Field(min_length=1, max_length=4000)
