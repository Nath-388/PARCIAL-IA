"""Cliente del LLM: la única parte de la aplicación que conoce al proveedor.

Recibe una lista de mensajes y parámetros; devuelve un LLMResponse.
El resto de la aplicación no importa `openai` ni sabe qué proveedor se usa.

Novedad del Lab 03: tool calling. `chat` puede recibir definiciones de tools y el
LLMResponse trae las tools que el modelo pide ejecutar (`tool_calls`). El modelo
solo GENERA la petición (un nombre y argumentos en JSON); nunca ejecuta nada.
"""

import json
from dataclasses import dataclass, field
from typing import Any

import openai

from config import Settings

# Un mensaje es un diccionario {"role": ..., "content": ...}. Desde el Lab 03 hay dos formas nuevas:
#   - {"role": "assistant", "content": ..., "tool_calls": [...]}  el modelo pide tools
#   - {"role": "tool", "tool_call_id": ..., "content": ...}        el resultado de una tool
Message = dict[str, Any]

# Definición de una tool, independiente del proveedor:
#   {"name": ..., "description": ..., "parameters": <JSON Schema de los argumentos>}
ToolDefinition = dict[str, Any]


@dataclass
class ToolCall:
    """Una petición del modelo: "ejecuta la tool `name` con estos argumentos"."""

    id: str  # identificador que el resultado debe citar (tool_call_id)
    name: str
    arguments_json: str  # texto JSON generado por el modelo: puede venir mal formado

    def arguments(self) -> dict[str, Any]:
        """Convierte el texto generado en un diccionario. Lanza ValueError si no es válido."""
        try:
            arguments = json.loads(self.arguments_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"argumentos con JSON inválido: {exc}") from exc
        if not isinstance(arguments, dict):
            raise ValueError("los argumentos deben ser un objeto JSON")
        return arguments


@dataclass
class LLMResponse:
    text: str
    model: str
    finish_reason: str  # "stop" = terminó, "length" = se agotó max_tokens, "tool_calls" = pide tools
    prompt_tokens: int
    completion_tokens: int
    tool_calls: list[ToolCall] = field(default_factory=list)

    def as_message(self) -> Message:
        """El turno del asistente tal como debe volver al LLM, incluidas las tools que pidió."""
        message: Message = {"role": "assistant", "content": self.text}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments_json},
                }
                for call in self.tool_calls
            ]
        return message


class LLMError(Exception):
    """Error al comunicarse con el proveedor, expresado sin detalles del SDK."""


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = openai.OpenAI(base_url=settings.base_url, api_key=settings.api_key)

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
        allow_tools: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        extra: dict[str, Any] = {"response_format": {"type": "json_object"}} if json_mode else {}
        if tools:
            # Formato de Chat Completions: cada tool se envuelve en {"type": "function", ...}.
            extra["tools"] = [{"type": "function", "function": tool} for tool in tools]
            # "auto": el modelo decide si pide tools; "none": debe responder con texto.
            extra["tool_choice"] = "auto" if allow_tools else "none"
        try:
            completion = self._client.chat.completions.create(
                model=self.settings.model,
                messages=messages,
                temperature=self.settings.temperature if temperature is None else temperature,
                max_tokens=self.settings.max_tokens if max_tokens is None else max_tokens,
                **extra,
            )
        except openai.APIError as exc:
            raise LLMError(f"Error del proveedor {self.settings.provider}: {exc}") from exc

        choice = completion.choices[0]
        usage = completion.usage
        return LLMResponse(
            text=choice.message.content or "",
            model=completion.model,
            finish_reason=choice.finish_reason or "desconocido",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            tool_calls=[
                ToolCall(id=call.id, name=call.function.name, arguments_json=call.function.arguments)
                for call in choice.message.tool_calls or []
                if call.type == "function"
            ],
        )


if __name__ == "__main__":
    # Prueba de humo: una sola llamada, sin interfaz ni historial.
    from config import load_settings

    client = LLMClient(load_settings())
    response = client.chat(
        [
            {"role": "system", "content": "Responde en una sola frase, en español."},
            {"role": "user", "content": "¿Qué es un Transformer en IA?"},
        ]
    )
    print(response)
