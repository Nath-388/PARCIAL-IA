"""Vector store en memoria: guarda fragmentos con sus vectores y busca los más parecidos.

Es deliberadamente simple (una matriz de NumPy y una lista de fragmentos) para ver
qué hacen por dentro Chroma, FAISS o pgvector: guardar vectores y encontrar los más
cercanos a una consulta. Busca por fuerza bruta (compara la consulta con todos los
vectores), suficiente para miles de fragmentos.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from documents import Chunk

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.json"


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # similitud coseno entre la consulta y el fragmento (de -1 a 1)


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Divide cada vector por su norma. Después, producto punto = similitud coseno."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class VectorStore:
    def __init__(self, embedding_model: str):
        # Los vectores solo son comparables con otros producidos por el mismo modelo.
        self.embedding_model = embedding_model
        self.chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None  # una fila normalizada por fragmento

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Debe haber exactamente un vector por fragmento.")
        vectors = normalize(np.asarray(vectors, dtype=np.float32))
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchResult]:
        """Devuelve los k fragmentos más parecidos a la consulta, del más al menos parecido."""
        if self._vectors is None:
            return []
        scores = self._vectors @ normalize(query_vector)
        top = np.argsort(scores)[::-1][:k]
        return [SearchResult(chunk=self.chunks[i], score=float(scores[i])) for i in top]

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / VECTORS_FILE, self._vectors)
        data = {"embedding_model": self.embedding_model, "chunks": [asdict(c) for c in self.chunks]}
        (index_dir / CHUNKS_FILE).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        chunks_path = index_dir / CHUNKS_FILE
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"No existe el índice en {index_dir}. Ejecuta primero ingest.py."
            )
        data = json.loads(chunks_path.read_text(encoding="utf-8"))
        store = cls(data["embedding_model"])
        store.chunks = [Chunk(**c) for c in data["chunks"]]
        store._vectors = np.load(index_dir / VECTORS_FILE)
        return store
