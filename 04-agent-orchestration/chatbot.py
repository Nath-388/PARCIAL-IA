"""Interfaz de línea de comandos del asistente (Versión 4: Usuario → Orquestador → LLM + RAG + Tools + MCP).

Requiere el índice creado por ingest.py. Al iniciar lanza el servidor MCP (mcp_server.py).

Uso:
    uv run python labs/04-agent-orchestration/chatbot.py [--modo agente|router] [--debug]

    --modo    agente (por defecto): el modelo decide en un ciclo qué capacidades usar
              router: una clasificación elige una sola ruta (directa, documentos o acciones)
    --debug   muestra la ruta, cada paso del agente y los tokens de cada llamada
"""

import asyncio
import json
import sys
from contextlib import AsyncExitStack
from typing import Any

from llm_client import LLMError, Message, ToolCall
from orchestrator import Mode, Orchestrator, OrchestratorAnswer, create_orchestrator
from tools import Tool

MODES = ("agente", "router")


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


def print_debug(answer: OrchestratorAnswer) -> None:
    print("\n--- Traza ---")
    if answer.routing:
        fallback = "  (respaldo)" if answer.routing.fallback else ""
        print(f"router → {answer.routing.decision.ruta}{fallback}: {answer.routing.decision.motivo}")
    for i, step in enumerate(answer.steps, start=1):
        if not step.tool_runs:
            print(f"paso {i}: responde")
        for run in step.tool_runs:
            result = run.result.replace("\n", " ")[:90]
            print(f"paso {i}: {describe_call(run.call)}\n        → {run.status}: {result}")
    if answer.stopped_by_limit:
        print("(se agotaron los pasos: respuesta forzada sin tools)")
    tokens_in = sum(response.prompt_tokens for response in answer.responses)
    tokens_out = sum(response.completion_tokens for response in answer.responses)
    print(f"llamadas al LLM: {len(answer.responses)} · tokens entrada={tokens_in} salida={tokens_out}")
    print("-------------")


def print_tools(tools: list[Tool]) -> None:
    for tool in tools:
        kind = "solo lectura" if tool.read_only else "ACCIÓN"
        print(f"  - {tool.name} ({tool.origin}, {kind})")


def parse_mode(argv: list[str]) -> Mode:
    if "--modo" in argv:
        index = argv.index("--modo") + 1
        if index < len(argv) and argv[index] in MODES:
            return argv[index]
        raise ValueError(f"--modo debe ser uno de: {', '.join(MODES)}")
    return "agente"


async def chat(orchestrator: Orchestrator, mode: Mode, debug: bool) -> None:
    history: list[Message] = []
    while True:
        try:
            user_input = await ask(f"Tú [{mode}]: ")
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
            print_tools(list(orchestrator.agent.tools.values()))
            print()
            continue
        if user_input.startswith("/modo"):
            new_mode = user_input.removeprefix("/modo").strip()
            if new_mode in MODES:
                mode = new_mode
                print(f"(modo {mode})\n")
            else:
                print(f"Uso: /modo {' | '.join(MODES)}\n")
            continue

        try:
            answer = await orchestrator.answer(user_input, history, mode)
        except LLMError as exc:
            print(f"\n[error] {exc}\n")
            continue

        if debug:
            print_debug(answer)
        print(f"\nAsistente: {answer.text}\n")
        if answer.sources:
            # Las fuentes las lista el programa a partir de lo recuperado, no el LLM.
            print(f"Fuentes: {', '.join(answer.sources)}\n")

        # Como en los labs anteriores, el historial guarda solo la pregunta y la respuesta final.
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer.text})


async def main() -> None:
    debug = "--debug" in sys.argv
    try:
        mode = parse_mode(sys.argv)
    except ValueError as exc:
        print(f"[uso] {exc}")
        return

    async with AsyncExitStack() as stack:
        try:
            orchestrator = await create_orchestrator(stack, confirm=confirm_in_console)
        except ValueError as exc:
            print(f"[configuración] {exc}")
            return
        except FileNotFoundError as exc:
            print(f"[índice] {exc}")
            return
        except Exception as exc:
            print(f"[inicio] No se pudo iniciar el asistente (servidor MCP o TODO pendiente): {exc!r}")
            return

        settings = orchestrator.llm.settings
        print(f"Asistente del Curso de IA · orquestador  ({settings.provider} · {settings.model})")
        print(f"Índice: {len(orchestrator.rag.store)} fragmentos · máximo {orchestrator.agent.max_steps} pasos")
        print("Capacidades:")
        print_tools(list(orchestrator.agent.tools.values()))
        print("Comandos: /modo agente|router  /tools  /reiniciar  /salir\n")
        await chat(orchestrator, mode, debug)

if __name__ == "__main__":
    asyncio.run(main())
