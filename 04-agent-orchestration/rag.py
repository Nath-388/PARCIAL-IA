"""RAG: recuperar fragmentos relevantes y usarlos como contexto para el LLM.

Pipeline de consulta (online), en cada pregunta:
    pregunta → embedding → búsqueda en el vector store → contexto → prompt → LLM → respuesta

CourseRAG no sabe nada de la consola: recibe una pregunta (y el historial) y devuelve
la respuesta junto con sus fuentes. Por eso otro programa puede reutilizarlo como una
capacidad más.
"""

from dataclasses import dataclass

from embeddings import Embedder
from llm_client import LLMClient, LLMResponse, Message
from prompts import build_rag_messages
from vector_store import SearchResult, VectorStore

NO_CONTEXT = "(No se encontraron fragmentos relevantes en los documentos del curso.)"


@dataclass
class RAGAnswer:
    text: str
    sources: list[SearchResult]  # fragmentos recuperados y enviados al LLM
    messages: list[Message]  # lo que realmente recibió el LLM
    response: LLMResponse


def format_context(results: list[SearchResult]) -> str:
    """Numera los fragmentos para que el LLM pueda citarlos: "[1] (evaluacion.md) texto"."""
    if not results:
        return NO_CONTEXT
    return "\n\n".join(
        f"[{i}] ({result.chunk.source}) {result.chunk.text}" for i, result in enumerate(results, 1)
    )


class CourseRAG:
    def __init__(
        self,
        llm: LLMClient,
        embedder: Embedder,
        store: VectorStore,
        top_k: int,
        min_score: float,
    ):
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, question: str) -> list[SearchResult]:
        query_vector = self.embedder.embed_query(question)
        results = self.store.search(query_vector, self.top_k)
        return [result for result in results if result.score >= self.min_score]

    def answer(self, question: str, history: list[Message] | None = None) -> RAGAnswer:
        sources = self.retrieve(question)
        messages = build_rag_messages(history or [], question, format_context(sources))
        response = self.llm.chat(messages)
        return RAGAnswer(text=response.text, sources=sources, messages=messages, response=response)
