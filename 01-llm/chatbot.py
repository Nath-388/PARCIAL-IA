"""Interfaz de línea de comandos del asistente (Versión 1: Usuario → LLM).

Uso:
    uv run python labs/01-llm/chatbot.py [--debug]
"""

import sys

from config import load_settings
from llm_client import LLMClient, LLMError, Message
from prompts import build_messages


def print_messages(messages: list[Message]) -> None:
    print("\n--- Mensajes enviados al LLM ---")
    for message in messages:
        preview = message["content"].replace("\n", " ")[:80]
        print(f"[{message['role']}] {preview}")
    print("--------------------------------")


def main() -> None:
    debug = "--debug" in sys.argv
    try:
        settings = load_settings()
    except ValueError as exc:
        print(f"[configuración] {exc}")
        return
    client = LLMClient(settings)
    history: list[Message] = []

    print(f"Asistente del Curso de IA  ({settings.provider} · {settings.model})")
    print("Comandos: /reiniciar  /salir\n")

    while True:
        try:
            user_input = input("Tú: ").strip()
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

        messages = build_messages(history, user_input)
        if debug:
            print_messages(messages)

        try:
            response = client.chat(messages)
        except LLMError as exc:
            print(f"\n[error] {exc}\n")
            continue

        print(f"\nAsistente: {response.text}\n")
        if debug:
            print(
                f"[finish_reason={response.finish_reason} · "
                f"tokens entrada={response.prompt_tokens} salida={response.completion_tokens}]\n"
            )

        # TODO 5: agrega al historial la pregunta del usuario y la respuesta del asistente
        #   (con los roles "user" y "assistant"). Primero ejecuta el chatbot SIN este paso
        #   y pregúntale "¿Qué te pregunté antes?". Luego complétalo y repite la prueba.
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response.text})

if __name__ == "__main__":
    main()