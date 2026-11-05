"""Interfaz de línea de comandos del asistente (Versión 2: Usuario → RAG → LLM).

Requiere el índice creado por ingest.py.

Uso:
    uv run python labs/02-rag/chatbot.py [--debug]
"""

import sys

from config import load_rag_settings, load_settings
from embeddings import Embedder
from llm_client import LLMClient, LLMError, Message
from rag import CourseRAG, RAGAnswer
from vector_store import VectorStore


def print_debug(answer: RAGAnswer) -> None:
    print("\n--- Fragmentos recuperados ---")
    if not answer.sources:
        print("(ninguno supera RAG_MIN_SCORE)")
    for i, result in enumerate(answer.sources, start=1):
        preview = result.chunk.text[:70]
        print(f"[{i}] score={result.score:.3f}  {result.chunk.id}: {preview}...")
    print("--- Mensajes enviados al LLM ---")
    for message in answer.messages:
        preview = message["content"].replace("\n", " ")[:80]
        print(f"[{message['role']}] ({len(message['content'])} caracteres) {preview}")
    print("--------------------------------")


def main() -> None:
    debug = "--debug" in sys.argv
    try:
        settings = load_settings()
        rag_settings = load_rag_settings()
    except ValueError as exc:
        print(f"[configuración] {exc}")
        return
    try:
        store = VectorStore.load(rag_settings.index_dir)
    except FileNotFoundError as exc:
        print(f"[índice] {exc}")
        return

    rag = CourseRAG(
        llm=LLMClient(settings),
        # La pregunta se convierte con el mismo modelo que indexó los documentos.
        embedder=Embedder(store.embedding_model),
        store=store,
        top_k=rag_settings.top_k,
        min_score=rag_settings.min_score,
    )
    history: list[Message] = []

    print(f"Asistente del Curso de IA + RAG  ({settings.provider} · {settings.model})")
    print(f"Índice: {len(store)} fragmentos · top_k={rag.top_k} · min_score={rag.min_score}")
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

        try:
            answer = rag.answer(user_input, history)
        except LLMError as exc:
            print(f"\n[error] {exc}\n")
            continue

        if debug:
            print_debug(answer)
        print(f"\nAsistente: {answer.text}\n")
        if answer.sources:
            # Las fuentes las lista el programa a partir de lo recuperado, no el LLM.
            print(f"Fuentes: {', '.join(sorted({r.chunk.source for r in answer.sources}))}\n")
        if debug:
            print(
                f"[finish_reason={answer.response.finish_reason} · "
                f"tokens entrada={answer.response.prompt_tokens} "
                f"salida={answer.response.completion_tokens}]\n"
            )

        # El historial guarda la pregunta original, no el mensaje con los fragmentos:
        # el contexto se recupera de nuevo en cada turno.
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer.text})


if __name__ == "__main__":
    main()
