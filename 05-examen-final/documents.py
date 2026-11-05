"""Carga de documentos y chunking.

Un documento completo es demasiado largo y mezcla muchos temas. Lo dividimos en
fragmentos (chunks) que se indexan y se recuperan por separado: al LLM solo le
llegan los fragmentos relevantes para la pregunta.

Los documentos viven en subcarpetas por nivel de acceso:
    publico/                 nivel "publico"    (cualquier usuario)
    restringido/estudiante/  nivel "estudiante" (estudiantes autenticados)
    restringido/docente/     nivel "docente"    (solo docente)
El nivel se guarda en cada `Document` y en cada `Chunk`, y lo aplica core.py
(SessionRAG) antes de construir el contexto que se le pasa al LLM.

Uso (prueba de chunking, sin modelos ni API):
    uv run python documents.py
"""

from dataclasses import dataclass
from pathlib import Path

# Cada nivel vive en su propia carpeta bajo data/.
#   publico/                 → nivel "publico"
#   restringido/estudiante/  → nivel "estudiante"
#   restringido/docente/     → nivel "docente"
CARPETAS_POR_NIVEL = (
    ("publico", ("publico",)),
    ("estudiante", ("restringido", "estudiante")),
    ("docente", ("restringido", "docente")),
)

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
    nivel: str  # "publico" | "estudiante" | "docente"


@dataclass(frozen=True)
class Chunk:
    id: str  # documento de origen + posición, p. ej. "evaluacion.md#2"
    source: str
    text: str
    nivel: str = "publico"  # default para compatibilidad con índices antiguos


def load_documents(data_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for nivel, partes in CARPETAS_POR_NIVEL:
        folder = data_dir.joinpath(*partes)
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*.md")):
            if path.name in EXCLUDED_DOCS:
                continue
            docs.append(
                Document(
                    source=path.name,
                    text=path.read_text(encoding="utf-8"),
                    nivel=nivel,
                )
            )
    # Compatibilidad: carpeta plana con .md sueltos → se tratan como "publico".
    if not docs:
        for path in sorted(data_dir.glob("*.md")):
            if path.name in EXCLUDED_DOCS:
                continue
            docs.append(
                Document(
                    source=path.name,
                    text=path.read_text(encoding="utf-8"),
                    nivel="publico",
                )
            )
    return docs


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
        Chunk(
            id=f"{doc.source}#{i}",
            source=doc.source,
            text=text,
            nivel=doc.nivel,
        )
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