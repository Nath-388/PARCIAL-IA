"""Tool calling sin ejecución: solo la primera mitad.

El LLM recibe la pregunta y las DEFINICIONES de las tools locales, y responde con
texto o con una petición de tool (nombre + argumentos en JSON). Este programa no
ejecuta nada: muestra que el modelo GENERA la petición y que ejecutarla es
responsabilidad del software (assistant.py).

Uso:
    uv run python labs/03-mcp/tool_calling.py "¿Qué día es hoy?"
"""

import sys

from config import load_settings
from llm_client import LLMClient, LLMError
from prompts import build_messages
from tools import LOCAL_TOOLS


def main() -> None:
    question = " ".join(sys.argv[1:]) or input("Pregunta: ")
    try:
        client = LLMClient(load_settings())
    except ValueError as exc:
        print(f"[configuración] {exc}")
        return

    definitions = [tool.definition() for tool in LOCAL_TOOLS]
    print(f"Tools ofrecidas al LLM: {', '.join(tool['name'] for tool in definitions)}\n")
    try:
        response = client.chat(build_messages([], question), tools=definitions)
    except LLMError as exc:
        print(f"[error del proveedor] {exc}")
        return

    print(f"finish_reason: {response.finish_reason}")
    print(f"texto: {response.text!r}")
    if not response.tool_calls:
        print("\nEl modelo respondió sin pedir tools.")
        return

    print("tool_calls:")
    for call in response.tool_calls:
        print(f"  - id={call.id}  name={call.name}  arguments={call.arguments_json}")
    print("\nNada se ha ejecutado: el modelo solo generó la petición.")


if __name__ == "__main__":
    main()
