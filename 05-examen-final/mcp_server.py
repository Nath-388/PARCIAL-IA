"""Servidor MCP "Servicios académicos" (ficticio).

Expone, con el protocolo estándar MCP, datos y acciones del sistema académico:
  - tools (el MODELO decide pedirlas): consultar_notas, consultar_solicitudes, solicitar_supletorio
  - resources (la APLICACIÓN decide leerlos): academico://calendario

Es un programa independiente del asistente: no conoce al LLM, ni al proveedor, ni el
prompt. Cualquier cliente MCP (este asistente, un IDE, n8n, otro agente) puede usarlo.

El SDK genera el JSON Schema de cada tool a partir de los tipos y las descripciones
de la función (compáralo con las definiciones escritas a mano en tools.py).

Normalmente no se ejecuta a mano: mcp_client.py lo lanza como subproceso y le habla
por stdin/stdout (transporte stdio). Para exponerlo por HTTP (Streamable HTTP):
    uv run python labs/04-agent-orchestration/mcp_server.py --http
"""

import json
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

DATA_FILE = Path(__file__).resolve().parent / "data" / "academico.json"
DATA = json.loads(DATA_FILE.read_text(encoding="utf-8"))
ESTUDIANTES: dict[str, Any] = DATA["estudiantes"]

# Solicitudes creadas en esta ejecución (en memoria: se pierden al cerrar el servidor).
SOLICITUDES: list[dict[str, Any]] = []

CodigoEstudiante = Annotated[str, Field(description="Código del estudiante, p. ej. 2026001")]

# IMPORTANTE: con el transporte stdio, stdout es el canal del protocolo. Un print() normal
# corrompería los mensajes; para depurar usa print(..., file=sys.stderr).
mcp = MCPServer(
    "servicios-academicos",
    instructions="Servicios académicos (ficticios) del curso de Inteligencia Artificial.",
    log_level="WARNING",  # los errores previstos (ToolError) no se imprimen en la consola del chat
)


def buscar_estudiante(codigo: str) -> dict[str, Any]:
    estudiante = ESTUDIANTES.get(codigo.strip())
    if estudiante is None:
        # ToolError = error previsto: el mensaje le llega al cliente (y al LLM) para que pueda corregir.
        raise ToolError(f"No existe un estudiante con código {codigo!r}.")
    return estudiante


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def consultar_notas(codigo: CodigoEstudiante) -> dict[str, Any]:
    """Consulta las notas registradas de un estudiante del curso.

    Una nota null significa que aún no está registrada. Incluye el promedio de los laboratorios.
    """
    estudiante = buscar_estudiante(codigo)
    notas = estudiante["notas"]
    laboratorios = notas["laboratorios"]
    return {
        "codigo": codigo,
        "nombre": estudiante["nombre"],
        "notas": notas,
        "promedio_laboratorios": round(sum(laboratorios) / len(laboratorios), 2) if laboratorios else None,
    }


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def consultar_solicitudes(codigo: CodigoEstudiante) -> list[dict[str, Any]]:
    """Lista las solicitudes de supletorio registradas por un estudiante."""
    buscar_estudiante(codigo)
    return [solicitud for solicitud in SOLICITUDES if solicitud["codigo"] == codigo]


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
def solicitar_supletorio(
    codigo: CodigoEstudiante,
    parcial: Annotated[Literal[1, 2], Field(description="Número del parcial: 1 o 2")],
    causa: Annotated[
        Literal["incapacidad médica", "calamidad doméstica", "representación institucional"],
        Field(description="Causa justificada de la inasistencia"),
    ],
) -> dict[str, Any]:
    """Registra una solicitud de supletorio para un parcial al que el estudiante no asistió.

    Es una ACCIÓN: crea un registro en el sistema académico.
    """
    estudiante = buscar_estudiante(codigo)
    if any(s["codigo"] == codigo and s["parcial"] == parcial for s in SOLICITUDES):
        raise ToolError(f"Ya existe una solicitud de supletorio del parcial {parcial} para {codigo}.")
    solicitud = {
        "radicado": f"SUP-2026-{len(SOLICITUDES) + 1:03d}",
        "codigo": codigo,
        "nombre": estudiante["nombre"],
        "parcial": parcial,
        "causa": causa,
        "estado": "pendiente: adjuntar el soporte por Teams dentro de los tres días hábiles siguientes al parcial",
    }
    SOLICITUDES.append(solicitud)
    return solicitud


@mcp.resource("academico://calendario", description="Fechas clave del curso (parciales y proyecto final)")
def calendario() -> str:
    return "\n".join(f"{item['fecha']}: {item['evento']}" for item in DATA["calendario"])


if __name__ == "__main__":
    if "--http" in sys.argv:
        mcp.run("streamable-http")  # http://127.0.0.1:8000/mcp
    else:
        mcp.run()  # stdio: lo lanza el cliente
