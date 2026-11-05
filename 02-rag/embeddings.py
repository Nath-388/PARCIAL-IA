"""Modelo de embeddings: la única parte de la aplicación que conoce a fastembed.

Convierte texto en vectores. Textos con significado parecido producen vectores
cercanos, aunque no compartan palabras.

El modelo corre en local (ONNX, CPU): no necesita API key y los documentos no salen
del equipo. La primera ejecución descarga el modelo (~220 MB) y lo guarda en caché.
Es un modelo distinto del LLM: el LLM genera texto; este solo lo convierte en vectores.

Uso (prueba de humo):
    uv run python labs/02-rag/embeddings.py
"""

import numpy as np
from fastembed import TextEmbedding


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Una fila por texto: matriz de forma (len(texts), dimensión)."""
        return np.array(list(self._model.passage_embed(texts)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Un vector de forma (dimensión,)."""
        return np.array(next(iter(self._model.query_embed(text))), dtype=np.float32)


if __name__ == "__main__":
    from config import load_rag_settings

    embedder = Embedder(load_rag_settings().embedding_model)
    query = "¿Qué pasa si entrego tarde un laboratorio?"
    texts = [
        "Las entregas extemporáneas tienen una penalización por cada día de retraso.",
        "Si mandas el lab después de la fecha límite pierdes puntos.",
        "El primer parcial evalúa las unidades 1, 2 y 3.",
        "Receta de arepas con queso.",
    ]
    query_vector = embedder.embed_query(query)
    vectors = embedder.embed_documents(texts)

    print(f"Modelo: {embedder.model_name}")
    print(f"Dimensión: {query_vector.shape[0]}")
    print(f"Primeros valores del vector de la pregunta: {np.round(query_vector[:5], 3)}\n")
    print(f"Pregunta: {query}")
    for text, vector in zip(texts, vectors):
        cosine = query_vector @ vector / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
        print(f"  coseno={cosine:.3f}  {text}")
