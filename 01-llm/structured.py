"""Salida estructurada: pedir JSON al LLM y validarlo con un esquema.

El LLM produce texto. Convertir ese texto en un objeto confiable es
responsabilidad del software tradicional (parseo + validación).

Uso:
    uv run python labs/01-llm/structured.py "¿Qué es el mecanismo de atención?"
"""

import json
import sys
from typing import Literal

from pydantic import BaseModel, ValidationError

from config import load_settings
from llm_client import LLMClient, LLMError
from prompts import ANALYSIS_PROMPT


class QuestionAnalysis(BaseModel):
    tema: str
    dificultad: Literal["basica", "intermedia", "avanzada"]
    requiere_documentos_del_curso: bool
    respuesta_corta: str


def analyze_question(client: LLMClient, question: str) -> QuestionAnalysis:
    messages = [
        {"role": "system", "content": ANALYSIS_PROMPT},
        {"role": "user", "content": question},
    ]
    response = client.chat(messages, temperature=0, json_mode=True)
    print(f"Texto crudo del LLM:\n{response.text}\n")

    data = json.loads(response.text)
    return QuestionAnalysis.model_validate(data)

def main() -> None:
    question = " ".join(sys.argv[1:]) or input("Pregunta: ")
    try:
        client = LLMClient(load_settings())
    except ValueError as exc:
        print(f"[configuración] {exc}")
        return

    try:
        analysis = analyze_question(client, question)
    except LLMError as exc:
        print(f"[error del proveedor] {exc}")
        return
    except json.JSONDecodeError as exc:
        print(f"[error] El LLM no devolvió JSON válido: {exc}")
        return
    except ValidationError as exc:
        print(f"[error] El JSON no cumple el esquema:\n{exc}")
        return

    print("Objeto validado:")
    print(analysis.model_dump_json(indent=2))

    # Ya es un objeto de Python: el programa puede tomar decisiones con él.
    if analysis.requiere_documentos_del_curso:
        print("\n→ Esta pregunta necesitaría documentos del curso para responderse bien.")


if __name__ == "__main__":
    main()