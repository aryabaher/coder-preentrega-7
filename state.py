"""Estado compartido del orquestador. Hereda messages y registra quién aportó qué."""

import operator
from typing import Annotated, Dict, List, Optional

from langgraph.graph import MessagesState
from pydantic import BaseModel, ConfigDict, Field


class AgentState(MessagesState):
    """Hereda 'messages' (con su propio reducer add_messages) y suma campos propios."""

    next_agent: Optional[str]
    contribuciones: Annotated[List[Dict[str, str]], operator.add]
    pasos: int
    task_completed: bool
    critica: bool
    aprobado: Optional[bool]


class Solicitud(BaseModel):
    """La pregunta del usuario, validada antes de entrar al grafo."""

    model_config = ConfigDict(str_strip_whitespace=True)

    texto: str = Field(min_length=1, max_length=4000)
