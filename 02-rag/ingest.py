"""Pipeline de indexación (offline): documentos → chunks → embeddings → vector store en disco.

Se ejecuta una vez, y de nuevo cada vez que cambian los documentos o la configuración
de chunking o de embeddings. No usa el LLM ni necesita API key.

Uso:
    uv run python labs/02-rag/ingest.py [--show]
"""

import sys
import time

from config import load_rag_settings
from documents import chunk_documents, load_documents
from embeddings import Embedder
from vector_store import VectorStore


def main() -> None:
    show = "--show" in sys.argv
    try:
        settings = load_rag_settings()
    except ValueError as exc:
        print(f"[configuración] {exc}")
        return

    documents = load_documents(settings.data_dir)
    if not documents:
        print(f"No hay documentos .md en {settings.data_dir}")
        return

    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    print(
        f"{len(documents)} documentos → {len(chunks)} chunks "
        f"(chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap} palabras)"
    )
    if show:
        for chunk in chunks:
            print(f"\n[{chunk.id}] ({len(chunk.text.split())} palabras)\n{chunk.text}")

    print(f"\nCalculando embeddings con {settings.embedding_model} ...")
    start = time.perf_counter()
    embedder = Embedder(settings.embedding_model)
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    elapsed = time.perf_counter() - start
    print(f"{vectors.shape[0]} vectores de dimensión {vectors.shape[1]} en {elapsed:.1f} s")

    store = VectorStore(settings.embedding_model)
    store.add(chunks, vectors)
    store.save(settings.index_dir)
    print(f"Índice guardado en {settings.index_dir}")


if __name__ == "__main__":
    main()
