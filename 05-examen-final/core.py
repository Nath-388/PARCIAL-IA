"""Núcleo del asistente del curso: lógica independiente del canal.

- Se puede ejecutar y probar sin Telegram:
      from core import Sesion, autenticar, responder
      asyncio.run(responder("¿Cuándo es el primer parcial?", [], Sesion()))

- Aplica control de acceso por sesión:
    * tools de notas/solicitudes fuerzan el código del usuario
    * RAG oculta los documentos "solo docente" a los no-docentes
    * usuario público: rechaza consultas privadas

- Los errores del LLM o del MCP se convierten en un mensaje claro para el
  usuario y quedan registrados en el log (no tumban el canal).
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent import MAX_STEPS, Agent
from config import load_rag_settings, load_settings
from embeddings import Embedder
from llm_client import LLMClient, LLMError, Message
from mcp_client import load_mcp_tools, open_client
from rag import CourseRAG, format_context
from tools import LOCAL_TOOLS, Tool, ToolExecutionError
from vector_store import VectorStore

logger = logging.getLogger(__name__)

LAB_DIR = Path(__file__).resolve().parent

# Documentos que solo puede ver el rol docente (encabezado "Nivel de acceso: docente").
DOCENTE_ONLY_DOCS = {"casos-especiales.md", "rubrica-interna-proyecto.md"}

MAX_HISTORIAL = 20


# --------------------------------------------------------------------------
# Sesión
# --------------------------------------------------------------------------

@dataclass
class Sesion:
    telegram_user_id: int | None = None
    usuario: dict | None = None
    rol: str = "publico"
    codigo_estudiante: str | None = None
    nombre: str = "Visitante"

    @property
    def autenticado(self) -> bool:
        return self.usuario is not None


def _usuarios_path() -> Path:
    for p in (LAB_DIR / "data" / "usuarios.json", LAB_DIR / "usuarios.json"):
        if p.exists():
            return p
    return LAB_DIR / "data" / "usuarios.json"


def _cargar_usuarios() -> list[dict]:
    path = _usuarios_path()
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("usuarios", [])


def autenticar(identificador: str, telegram_user_id: int | None = None) -> Sesion | None:
    """Devuelve la Sesion autenticada o None si el identificador no existe.

    Acepta el `codigo_estudiante` (estudiante) o el `id` interno (docente).
    """
    identificador = identificador.strip()
    for u in _cargar_usuarios():
        if u.get("codigo_estudiante") == identificador or u.get("id") == identificador:
            return Sesion(
                telegram_user_id=telegram_user_id,
                usuario=u,
                rol=u["rol"],
                codigo_estudiante=u.get("codigo_estudiante"),
                nombre=u["nombre"],
            )
    return None


def sesion_publica(telegram_user_id: int | None = None) -> Sesion:
    return Sesion(telegram_user_id=telegram_user_id)


# --------------------------------------------------------------------------
# Contexto para el LLM
# --------------------------------------------------------------------------

def _contexto_sesion(sesion: Sesion) -> str:
    if not sesion.autenticado:
        return (
            "Sesión actual: usuario PÚBLICO (sin login). "
            "No reveles notas, casos especiales, ni información reservada. "
            "Si piden datos privados, pídeles que usen /login."
        )
    codigo = sesion.codigo_estudiante or "(docente, sin código)"
    return (
        f"Sesión actual: {sesion.nombre} | rol={sesion.rol} | código={codigo}. "
        "Las tools de notas/solicitudes solo aceptan el código de esta sesión. "
        "Si el estudiante intenta otro código, la tool fallará y debes explicárselo."
    )


# --------------------------------------------------------------------------
# Tools con control de sesión
# --------------------------------------------------------------------------

def _tool_con_sesion(tool: Tool, sesion: Sesion) -> Tool:
    """Envuelve una tool MCP para forzar el código de la sesión cuando aplica."""

    async def run(arguments: dict[str, Any]) -> str:
        if sesion.rol == "publico":
            raise ToolExecutionError(
                "Debes iniciar sesión con /login antes de consultar esta información."
            )
        if sesion.rol != "docente":
            codigo = sesion.codigo_estudiante
            if not codigo:
                raise ToolExecutionError(
                    "Tu sesión no tiene un código de estudiante asociado."
                )
            # Ignora cualquier código que el LLM haya puesto: solo el de la sesión.
            arguments = {**arguments, "codigo": codigo}
        return await tool.run(arguments)

    return Tool(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
        run=run,
        read_only=tool.read_only,
        origin=tool.origin,
    )


def _rag_tool_con_sesion(rag: CourseRAG, sesion: Sesion) -> Tool:
    """RAG que oculta los documentos restringidos al rol docente."""

    async def run(arguments: dict[str, Any]) -> str:
        consulta = arguments.get("consulta")
        if not isinstance(consulta, str) or not consulta.strip():
            raise ValueError("falta el argumento 'consulta' (texto no vacío)")
        results = rag.retrieve(consulta)
        if sesion.rol != "docente":
            results = [r for r in results if r.chunk.source not in DOCENTE_ONLY_DOCS]
        return format_context(results)

    return Tool(
        name="buscar_documentos",
        description=(
            "Busca en los documentos oficiales del curso: programa, evaluación, políticas, "
            "anuncios y FAQ. Devuelve fragmentos numerados con su archivo de origen."
        ),
        parameters={
            "type": "object",
            "properties": {
                "consulta": {"type": "string", "description": "Pregunta autocontenida."}
            },
            "required": ["consulta"],
        },
        run=run,
        read_only=True,
        origin="rag",
    )


# --------------------------------------------------------------------------
# Core (singleton perezoso)
# --------------------------------------------------------------------------

async def _rechazar(tool: Tool, arguments: dict) -> bool:
    # En Telegram no hay diálogo de confirmación interactivo: se rechaza por defecto.
    return False


class Core:
    def __init__(self):
        self._stack: AsyncExitStack | None = None
        self._llm: LLMClient | None = None
        self._rag: CourseRAG | None = None
        self._base_tools: list[Tool] = []
        self._lock = asyncio.Lock()
        self._init_failed = False

    async def iniciar(self):
        if self._stack is not None:
            return
        if self._init_failed:
            raise RuntimeError(
                "El core falló al iniciar antes. Revisa la configuración y reinicia el bot."
            )
        async with self._lock:
            if self._stack is not None:
                return
            stack = AsyncExitStack()
            try:
                await stack.__aenter__()
                settings = load_settings()
                rag_settings = load_rag_settings()
                store = VectorStore.load(rag_settings.index_dir)
                llm = LLMClient(settings)
                rag = CourseRAG(
                    llm=llm,
                    embedder=Embedder(store.embedding_model),
                    store=store,
                    top_k=rag_settings.top_k,
                    min_score=rag_settings.min_score,
                )
                mcp_client = await open_client(stack)
                mcp_tools = await load_mcp_tools(mcp_client)
            except Exception:
                self._init_failed = True
                await stack.aclose()
                raise
            self._stack = stack
            self._llm = llm
            self._rag = rag
            self._base_tools = [*LOCAL_TOOLS, *mcp_tools]
            logger.info("Core iniciado: %d tools locales+MCP", len(self._base_tools))

    async def cerrar(self):
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._base_tools = []
            self._rag = None
            self._llm = None

    async def responder(
        self, pregunta: str, historial: list[Message], sesion: Sesion
    ) -> str:
        if self._stack is None:
            await self.iniciar()
        assert self._llm is not None and self._rag is not None

        # Construye un agente con las tools filtradas por sesión.
        tools = [
            _rag_tool_con_sesion(self._rag, sesion),
            *(_tool_con_sesion(t, sesion) for t in self._base_tools),
        ]
        agent = Agent(self._llm, tools, _rechazar, MAX_STEPS)

        # Inyecta el contexto de sesión como segundo mensaje del sistema.
        mensajes = [{"role": "system", "content": _contexto_sesion(sesion)}, *historial]

        try:
            result = await agent.run(pregunta, mensajes)
            return result.text
        except LLMError as exc:
            logger.error("Error del LLM: %s", exc)
            return (
                "El servicio de IA no está disponible en este momento. "
                "Intenta de nuevo en unos minutos."
            )
        except Exception:
            logger.exception("Error inesperado en responder()")
            return "Ocurrió un error inesperado. Intenta de nuevo."


_core: Core | None = None


async def _obtener_core() -> Core:
    global _core
    if _core is None:
        _core = Core()
        await _core.iniciar()
    return _core


# --------------------------------------------------------------------------
# API pública (sin dependencia de Telegram)
# --------------------------------------------------------------------------

async def responder(
    pregunta: str,
    historial: list[Message] | None = None,
    sesion: Sesion | None = None,
) -> str:
    """Punto de entrada del núcleo. No depende de Telegram."""
    core = await _obtener_core()
    return await core.responder(pregunta, historial or [], sesion or Sesion())


async def cerrar_core() -> None:
    global _core
    if _core is not None:
        await _core.cerrar()
        _core = None