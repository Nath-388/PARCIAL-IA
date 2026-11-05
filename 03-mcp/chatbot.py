"""Interfaz de línea de comandos del asistente (Versión 3: Usuario → Asistente → LLM + Tools).

Al iniciar lanza el servidor MCP (mcp_server.py) como subproceso y une sus tools
con las tools locales.

Uso:
    uv run python labs/03-mcp/chatbot.py [--debug] [--local]

    --debug  muestra las tools pedidas, sus resultados y los tokens de cada llamada
    --local  solo tools locales, sin servidor MCP
"""

import asyncio
import json
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import Client

from assistant import AssistantAnswer, CourseAssistant
from config import load_settings
from llm_client import LLMClient, LLMError, Message, ToolCall
from mcp_client import load_mcp_tools, read_resource_text, server_parameters
from tools import LOCAL_TOOLS, Tool


async def ask(prompt: str) -> str:
    # input() bloquea; en un hilo aparte no detiene la conexión con el servidor MCP.
    return (await asyncio.to_thread(input, prompt)).strip()


async def confirm_in_console(tool: Tool, arguments: dict[str, Any]) -> bool:
    print(f"\n[confirmación] El asistente quiere ejecutar una ACCIÓN de {tool.origin}:")
    print(f"  {tool.name}({json.dumps(arguments, ensure_ascii=False)})")
    return (await ask("¿La autorizas? (s/n): ")).lower() in {"s", "si", "sí"}


def describe_call(call: ToolCall) -> str:
    try:
        arguments = json.dumps(call.arguments(), ensure_ascii=False)
    except ValueError:
        arguments = call.arguments_json  # tal como lo generó el modelo
    return f"{call.name}({arguments})"


def print_debug(answer: AssistantAnswer) -> None:
    print("\n--- Tools pedidas por el LLM ---")
    if not answer.tool_runs:
        print("(ninguna: respondió directamente)")
    for run in answer.tool_runs:
        result = run.result.replace("\n", " ")[:100]
        print(f"{describe_call(run.call)}\n   → {run.status}: {result}")
    print("--- Mensajes de la última llamada ---")
    for message in answer.messages:
        content = (message.get("content") or "").replace("\n", " ")[:70]
        extra = f" tool_calls={len(message['tool_calls'])}" if message.get("tool_calls") else ""
        print(f"[{message['role']}]{extra} {content}")
    print("-------------------------------------")


def print_tools(tools: list[Tool]) -> None:
    for tool in tools:
        kind = "solo lectura" if tool.read_only else "ACCIÓN"
        print(f"  - {tool.name} ({tool.origin}, {kind})")


async def chat(assistant: CourseAssistant, mcp_client: Client | None, debug: bool) -> None:
    history: list[Message] = []
    while True:
        try:
            user_input = await ask("Tú: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input == "/salir":
            break
        if user_input == "/reiniciar":
            history.clear()
            print("(historial borrado)\n")
            continue
        if user_input == "/tools":
            print_tools(list(assistant.tools.values()))
            print()
            continue
        if user_input.startswith(("/recursos", "/leer")):
            # Los resources los lee la APLICACIÓN cuando el usuario (o el programa) lo decide, no el LLM.
            if mcp_client is None:
                print("(sin servidor MCP: ejecuta sin --local)\n")
            elif user_input == "/recursos":
                for resource in (await mcp_client.list_resources()).resources:
                    print(f"  - {resource.uri}: {resource.description}")
                print()
            else:
                uri = user_input.removeprefix("/leer").strip()
                try:
                    print(f"{await read_resource_text(mcp_client, uri)}\n")
                except Exception as exc:
                    print(f"[mcp] No se pudo leer {uri!r}: {exc}\n")
            continue

        try:
            answer = await assistant.answer(user_input, history)
        except LLMError as exc:
            print(f"\n[error] {exc}\n")
            continue

        if debug:
            print_debug(answer)
        print(f"\nAsistente: {answer.text}\n")
        if debug:
            for i, response in enumerate(answer.responses, start=1):
                print(
                    f"[llamada {i}: finish_reason={response.finish_reason} · "
                    f"tokens entrada={response.prompt_tokens} salida={response.completion_tokens}]"
                )
            print()

        # Como en el Lab 02, el historial guarda solo la pregunta y la respuesta final,
        # no las peticiones de tools ni sus resultados.
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer.text})


async def main() -> None:
    debug = "--debug" in sys.argv
    local_only = "--local" in sys.argv
    try:
        settings = load_settings()
    except ValueError as exc:
        print(f"[configuración] {exc}")
        return

    async with AsyncExitStack() as stack:
        tools = list(LOCAL_TOOLS)
        mcp_client = None
        if not local_only:
            try:
                mcp_client = await stack.enter_async_context(Client(server_parameters()))
                tools += await load_mcp_tools(mcp_client)
            except Exception as exc:
                print(f"[mcp] No se pudo iniciar el servidor MCP: {exc}")
                return

        assistant = CourseAssistant(LLMClient(settings), tools, confirm=confirm_in_console)
        print(f"Asistente del Curso de IA + Tools  ({settings.provider} · {settings.model})")
        print("Tools disponibles:")
        print_tools(tools)
        print("Comandos: /tools  /recursos  /leer <uri>  /reiniciar  /salir\n")
        await chat(assistant, mcp_client, debug)


if __name__ == "__main__":
    asyncio.run(main())
