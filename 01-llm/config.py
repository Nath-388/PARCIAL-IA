"""Configuración de la aplicación a partir de variables de entorno (.env).

Responsabilidad: reunir en un solo lugar todo lo que depende del entorno
(proveedor, credenciales, modelo y parámetros). Ningún otro módulo lee
variables de entorno directamente.
"""

import os
from dataclasses import dataclass

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


@dataclass(frozen=True)
class Settings:
    provider: str
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int


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
