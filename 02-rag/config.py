"""Configuración de la aplicación a partir de variables de entorno (.env).

Responsabilidad: reunir en un solo lugar todo lo que depende del entorno
(proveedor, credenciales, modelo y parámetros). Ningún otro módulo lee
variables de entorno directamente.

Novedad del Lab 02: RAGSettings. Indexar y buscar no usan el LLM, por eso
tienen su propia configuración y no exigen LLM_API_KEY.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Todos estos proveedores exponen la misma API (Chat Completions, compatible con OpenAI).
# Cambiar de proveedor = cambiar URL base + API key + nombre del modelo.
# Groq y OpenRouter tienen planes gratuitos; OpenAI y DeepSeek son de pago.
PROVIDER_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
}

LAB_DIR = Path(__file__).resolve().parent
DATA_DIR = LAB_DIR / "data"  # documentos del curso: la fuente de conocimiento
INDEX_DIR = LAB_DIR / "index"  # generado por ingest.py; no se versiona


@dataclass(frozen=True)
class Settings:
    provider: str
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class RAGSettings:
    data_dir: Path
    index_dir: Path
    embedding_model: str
    chunk_size: int  # palabras por fragmento
    chunk_overlap: int  # palabras compartidas por fragmentos consecutivos
    top_k: int  # cuántos fragmentos recuperar por pregunta
    min_score: float  # similitud mínima para considerar relevante un fragmento


def load_settings() -> Settings:
    # Busca el archivo .env desde el directorio actual hacia arriba.
    load_dotenv(find_dotenv(usecwd=True))

    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if provider not in PROVIDER_BASE_URLS:
        options = ", ".join(PROVIDER_BASE_URLS)
        raise ValueError(f"LLM_PROVIDER desconocido: {provider!r}. Opciones: {options}")

    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Falta LLM_API_KEY. Copia .env.example como .env y agrega tu clave.")

    model = os.getenv("LLM_MODEL", "").strip()
    if not model:
        raise ValueError("Falta LLM_MODEL en el archivo .env.")

    return Settings(
        provider=provider,
        base_url=PROVIDER_BASE_URLS[provider],
        api_key=api_key,
        model=model,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "512")),
    )


def load_rag_settings() -> RAGSettings:
    load_dotenv(find_dotenv(usecwd=True))

    chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "80"))
    chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "20"))
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("RAG_CHUNK_OVERLAP debe ser mayor o igual a 0 y menor que RAG_CHUNK_SIZE.")

    return RAGSettings(
        data_dir=DATA_DIR,
        index_dir=INDEX_DIR,
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ).strip(),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=int(os.getenv("RAG_TOP_K", "4")),
        min_score=float(os.getenv("RAG_MIN_SCORE", "0.35")),
    )
