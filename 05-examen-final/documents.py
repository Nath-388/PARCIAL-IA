"""Carga de documentos y chunking.

Un documento completo es demasiado largo y mezcla muchos temas. Lo dividimos en
fragmentos (chunks) que se indexan y se recuperan por separado: al LLM solo le
llegan los fragmentos relevantes para la pregunta.

Uso (prueba de chunking, sin modelos ni API):
    uv run python labs/04-agent-orchestration/documents.py
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    source: str  # nombre del archivo, p. ej. "evaluacion.md"
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str  # documento de origen + posición, p. ej. "evaluacion.md#2"
    source: str
    text: str


def load_documents(data_dir: Path) -> list[Document]:
    paths = sorted(data_dir.glob("*.md"))
    return [Document(source=path.name, text=path.read_text(encoding="utf-8")) for path in paths]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Divide el texto en ventanas de `chunk_size` palabras.

    Ventanas consecutivas comparten `overlap` palabras, para no cortar una idea
    justo en el borde entre dos fragmentos.
    """
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_documents(documents: list[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    return [
        Chunk(id=f"{doc.source}#{i}", source=doc.source, text=text)
        for doc in documents
        for i, text in enumerate(chunk_text(doc.text, chunk_size, overlap))
    ]


if __name__ == "__main__":
    sample = "uno dos tres cuatro cinco seis siete ocho nueve diez"
    for chunk in chunk_text(sample, chunk_size=4, overlap=1):
        print(chunk)
    # Esperado:
    # uno dos tres cuatro
    # cuatro cinco seis siete
    # siete ocho nueve diez
