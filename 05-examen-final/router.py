"""Router: una llamada al LLM que CLASIFICA la pregunta en una ruta.

El router no responde ni usa tools: solo decide por qué camino debe ir la pregunta.
Después, el orquestador (código normal) ejecuta el camino elegido. Es la salida
estructurada del Lab 01 (JSON + validación con pydantic) usada para tomar una decisión.

Uso (una llamada al LLM, sin documentos ni MCP):
    uv run python labs/04-agent-orchestration/router.py "¿Cuándo es el primer parcial?"
"""

import sys
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ValidationError

from llm_client import LLMClient, LLMResponse
from prompts import build_router_messages

Route = Literal["directa", "documentos", "acciones"]

# Si el LLM devuelve algo que no cumple el esquema, el software elige una ruta segura:
# "documentos" no ejecuta acciones y responde con evidencia o dice que no encontró la información.
FALLBACK_ROUTE: Route = "documentos"


class RouteDecision(BaseModel):
    ruta: Route
    motivo: str


@dataclass
class RouterResult:
    decision: RouteDecision
    response: LLMResponse
    fallback: bool  # True si se usó FALLBACK_ROUTE porque la salida del LLM no era válida


def route(llm: LLMClient, question: str) -> RouterResult:
    response = llm.chat(build_router_messages(question), temperature=0, max_tokens=150, json_mode=True)
    try:
        # Verifica a la vez que sea JSON y que cumpla el esquema (ruta ∈ directa/documentos/acciones).
        decision = RouteDecision.model_validate_json(response.text)
    except ValidationError:
        # Un LLM puede devolver texto libre, JSON incompleto o una ruta inventada: el software no se
        # detiene ni confía en eso, sino que elige la ruta segura.
        decision = RouteDecision(
            ruta=FALLBACK_ROUTE,
            motivo="La clasificación del LLM no fue válida; se usó la ruta de respaldo.",
        )
        return RouterResult(decision, response, fallback=True)
    return RouterResult(decision, response, fallback=False)


if __name__ == "__main__":
    from config import load_settings

    question = " ".join(sys.argv[1:]) or input("Pregunta: ")
    result = route(LLMClient(load_settings()), question)
    print(f"Texto crudo del LLM: {result.response.text}")
    print(f"Ruta: {result.decision.ruta}{'  (respaldo)' if result.fallback else ''}")
    print(f"Motivo: {result.decision.motivo}")
