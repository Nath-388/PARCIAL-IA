"""Recuperación sin LLM: muestra qué fragmentos encontraría el sistema para una pregunta.

Compara dos estrategias:
- semántica (por defecto): embeddings + similitud coseno;
- --keyword: fracción de palabras de la pregunta que aparecen en el fragmento (línea base).

No aplica RAG_MIN_SCORE: muestra todos los puntajes para que puedas elegir el umbral.

Uso:
    uv run python labs/02-rag/search.py "¿Qué pasa si entrego tarde un lab?"
    uv run python labs/02-rag/search.py --keyword "¿Qué pasa si entrego tarde un lab?"
"""

import re
import sys

from config import load_rag_settings
from embeddings import Embedder
from vector_store import SearchResult, VectorStore


def words(text: str) -> set[str]:
    """Palabras en minúscula de 4 o más letras (descarta "el", "de", "que"...)."""
    return {word for word in re.findall(r"\w+", text.lower()) if len(word) >= 4}


def keyword_search(store: VectorStore, question: str, k: int) -> list[SearchResult]:
    query = words(question)
    results = [
        SearchResult(chunk=chunk, score=len(query & words(chunk.text)) / max(len(query), 1))
        for chunk in store.chunks
    ]
    results.sort(key=lambda result: result.score, reverse=True)
    return results[:k]


def main() -> None:
    keyword = "--keyword" in sys.argv
    question = " ".join(arg for arg in sys.argv[1:] if arg != "--keyword") or input("Pregunta: ")
    settings = load_rag_settings()
    try:
        store = VectorStore.load(settings.index_dir)
    except FileNotFoundError as exc:
        print(f"[índice] {exc}")
        return

    if keyword:
        label = "por palabras clave (fracción de palabras de la pregunta presentes)"
        results = keyword_search(store, question, settings.top_k)
    else:
        label = "semántica (similitud coseno)"
        # La pregunta se convierte con el mismo modelo que indexó los documentos.
        embedder = Embedder(store.embedding_model)
        results = store.search(embedder.embed_query(question), settings.top_k)

    print(f"\nBúsqueda {label} · top {settings.top_k} de {len(store)} fragmentos\n")
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. score={result.score:.3f}  [{result.chunk.id}]")
        print(f"   {result.chunk.text[:160]}...\n")


if __name__ == "__main__":
    main()
