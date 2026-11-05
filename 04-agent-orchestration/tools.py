"""Tools: funciones del programa que el LLM puede pedir ejecutar.

Una tool tiene dos partes:
  - la DEFINICIÓN (nombre, descripción y JSON Schema de los argumentos): lo único que ve el LLM;
  - la IMPLEMENTACIÓN (código normal): lo que se ejecuta, siempre en el programa, nunca en el LLM.

Las tools locales viven en este proceso. Las tools de un servidor MCP (mcp_client.py) y la
búsqueda en documentos (capabilities.py) se convierten en este mismo tipo `Tool`, así que el
agente las trata igual.

Novedad del Lab 04: dias_hasta. Agregar una capacidad = agregar una tool.

Uso (sin LLM ni API key):
    uv run python labs/04-agent-orchestration/tools.py
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from llm_client import ToolDefinition


class ToolExecutionError(Exception):
    """La tool se ejecutó y reportó un error previsto (p. ej., un código de estudiante inexistente)."""


@dataclass
class Tool:
    name: str
    description: str  # el LLM decide CUÁNDO usar la tool leyendo esto
    parameters: dict[str, Any]  # JSON Schema: el LLM decide CON QUÉ argumentos leyendo esto
    run: Callable[[dict[str, Any]], Awaitable[str]]  # ejecuta y devuelve el resultado como texto
    read_only: bool  # False = acción con efectos (crea, modifica, envía): requiere confirmación
    origin: str  # "local" o el nombre del servidor MCP

    def definition(self) -> ToolDefinition:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def local_tool(
    function: Callable[..., Any], description: str, parameters: dict[str, Any], *, read_only: bool = True
) -> Tool:
    """Envuelve una función de Python como Tool. El resultado se devuelve al LLM como JSON."""

    async def run(arguments: dict[str, Any]) -> str:
        return json.dumps(function(**arguments), ensure_ascii=False)

    return Tool(
        name=function.__name__,
        description=description,
        parameters=parameters,
        run=run,
        read_only=read_only,
        origin="local",
    )


# --- Implementaciones: código tradicional, determinista ----------------------------------

DIAS_SEMANA = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

# Pesos publicados en la sección "Distribución de la nota" del curso.
PESOS = {"parcial_1": 0.20, "parcial_2": 0.20, "laboratorios": 0.35, "proyecto": 0.25}
NOTA_MINIMA = 3.0


def fecha_actual() -> dict[str, str]:
    today = date.today()
    return {"fecha": today.isoformat(), "dia_semana": DIAS_SEMANA[today.weekday()]}


def dias_hasta(fecha: str) -> dict[str, Any]:
    try:
        objetivo = date.fromisoformat(fecha)
    except ValueError as exc:
        raise ValueError(f"fecha debe tener el formato AAAA-MM-DD (recibido: {fecha!r})") from exc
    hoy = date.today()
    return {"hoy": hoy.isoformat(), "fecha": objetivo.isoformat(), "dias": (objetivo - hoy).days}


def calcular_nota_final(parcial_1: float, parcial_2: float, laboratorios: float, proyecto: float) -> dict[str, Any]:
    notas = {"parcial_1": parcial_1, "parcial_2": parcial_2, "laboratorios": laboratorios, "proyecto": proyecto}
    for nombre, nota in notas.items():
        if not 0.0 <= nota <= 5.0:
            raise ValueError(f"{nombre} debe estar entre 0.0 y 5.0 (recibido: {nota})")
    final = sum(nota * PESOS[nombre] for nombre, nota in notas.items())
    return {"nota_final": round(final, 2), "aprueba": final >= NOTA_MINIMA, "pesos": PESOS}


# --- Definiciones: lo que ve el LLM ---------------------------------------------------------

LOCAL_TOOLS = [
    local_tool(
        fecha_actual,
        description="Devuelve la fecha de hoy (AAAA-MM-DD) y el día de la semana. "
        "Úsala siempre que la respuesta dependa de la fecha actual.",
        parameters={"type": "object", "properties": {}, "required": []},
    ),

    local_tool(
        calcular_nota_final,
        description="Calcula la nota final del curso (0.0 a 5.0) con los pesos oficiales: "
        "parciales 20 % cada uno, laboratorios 35 % y proyecto 25 %. Indica si aprueba (mínimo 3.0).",
        parameters={
            "type": "object",
            "properties": {
                "parcial_1": {"type": "number", "minimum": 0, "maximum": 5, "description": "Nota del primer parcial"},
                "parcial_2": {"type": "number", "minimum": 0, "maximum": 5, "description": "Nota del segundo parcial"},
                "laboratorios": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 5,
                    "description": "Promedio de los laboratorios",
                },
                "proyecto": {"type": "number", "minimum": 0, "maximum": 5, "description": "Nota del proyecto final"},
            },
            "required": ["parcial_1", "parcial_2", "laboratorios", "proyecto"],
        },
    ),
]


async def _demo() -> None:
    print("Definiciones que recibe el LLM:\n")
    print(json.dumps([tool.definition() for tool in LOCAL_TOOLS], ensure_ascii=False, indent=2))

    print("\nEjecución directa, sin LLM:")
    tools = {tool.name: tool for tool in LOCAL_TOOLS}
    print("fecha_actual        →", await tools["fecha_actual"].run({}))
    print("dias_hasta          →", await tools["dias_hasta"].run({"fecha": "2026-12-03"}))
    notas = {"parcial_1": 3.5, "parcial_2": 4.0, "laboratorios": 4.2, "proyecto": 3.8}
    print("calcular_nota_final →", await tools["calcular_nota_final"].run(notas))


if __name__ == "__main__":
    asyncio.run(_demo())
