"""Carga de documentos y chunking.

Un documento completo es demasiado largo y mezcla muchos temas. Lo dividimos en
fragmentos (chunks) que se indexan y se recuperan por separado: al LLM solo le
llegan los fragmentos relevantes para la pregunta.

Los documentos viven en subcarpetas por nivel de acceso:
    publico/                 cualquier usuario
    restringido/estudiante/  estudiantes autenticados
    restringido/docente/     solo docente
El control de acceso lo aplica core.py (SessionRAG) usando el nombre del archivo.

Uso (prueba de chunking, sin modelos ni API):
    uv run python documents.py
"""

from dataclasses import dataclass
from pathlib import Path

# Subcarpetas de data_dir que se indexan.
DOCUMENT_DIRS = ("publico", "restringido")

# Archivos que NO son conocimiento del curso: enunciados de otra asignatura y un solucionario.
# Nunca deben llegar al índice (el solucionario es de uso exclusivo del docente).
EXCLUDED_DOCS = {
    "parcial-90min.md",
    "parcial-90min-respuestas.md",
    "parcial-practico-corte2.md",
}


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
    paths: list[Path] = []
    for sub in DOCUMENT_DIRS:
        folder = data_dir / sub
        if folder.is_dir():
            paths.extend(folder.rglob("*.md"))
    if not paths:  # compatibilidad: carpeta plana con .md
        paths = list(data_dir.glob("*.md"))
    paths = sorted(p for p in paths if p.name not in EXCLUDED_DOCS)
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