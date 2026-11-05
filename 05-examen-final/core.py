"""Núcleo del asistente del curso: lógica independiente del canal (Telegram, consola, tests).

Punto de entrada:
    responder(pregunta, historial, sesion) -> str

Usa el orquestador del Lab 04 (Orchestrator.answer): respuesta directa, RAG y tools/MCP.
Se puede ejecutar y probar sin Telegram:
    asyncio.run(responder("¿Cuándo es el primer parcial?", [], Sesion()))

Control de acceso por sesión (el rol y el código salen de usuarios.json, nunca del chat):
  * las tools de notas/solicitudes fuerzan el código de la sesión (estudiante);
  * el RAG filtra los documentos según el rol (público / estudiante / docente);
  * el usuario público no puede usar tools de datos académicos.

Los errores del LLM o del servidor MCP no se propagan: el usuario recibe un mensaje claro
y el detalle queda en el log.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import load_rag_settings, load_settings
from embeddings import Embedder
from llm_client import LLMClient, LLMError, Message
from mcp_client import load_mcp_tools, open_client
from orchestrator import Mode, Orchestrator
from rag import CourseRAG
from tools import LOCAL_TOOLS, Tool, ToolExecutionError
from vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)

LAB_DIR = Path(__file__).resolve().parent

# Documentos por nivel de acceso (encabezado "Nivel de acceso: ..." de cada archivo).
DOCENTE_ONLY_DOCS = {"casos-especiales.md", "rubrica-interna-proyecto.md"}
ESTUDIANTE_ONLY_DOCS = {"guia-segundo-parcial.md", "monitorias.md"}

MAX_HISTORIAL = 20

# "agente" (el modelo decide en un ciclo) o "router" (clasificación + ruta fija).
MODO: Mode = "router" if os.getenv("ASSISTANT_MODE", "agente").strip().lower() == "router" else "agente"

MSG_LLM_CAIDO = "El servicio de IA no está disponible en este momento. Intenta de nuevo en unos minutos."
MSG_ERROR = "Ocurrió un error inesperado. Intenta de nuevo."


# --------------------------------------------------------------------------
# Sesión
# --------------------------------------------------------------------------

@dataclass
class Sesion:
    telegram_user_id: int | None = None
    usuario: dict | None = None
    rol: str = "publico"  # publico | estudiante | docente
    codigo_estudiante: str | None = None
    nombre: str = "Visitante"

    @property
    def autenticado(self) -> bool:
        return self.usuario is not None


def _usuarios_path() -> Path:
    for p in (LAB_DIR / "data" / "usuarios.json", LAB_DIR / "usuarios.json"):
        if p.exists():
            return p
    return LAB_DIR / "usuarios.json"


def _cargar_usuarios() -> list[dict]:
    path = _usuarios_path()
    if not path.exists():
        logger.warning("No existe %s: nadie podrá iniciar sesión.", path)
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("usuarios", [])


def autenticar(identificador: str, telegram_user_id: int | None = None) -> Sesion | None:
    """Devuelve la Sesion autenticada o None si el identificador no existe.

    Acepta el `codigo_estudiante` (estudiante) o el `id` interno (docente). Si el usuario
    tiene un `telegram_user_id` registrado, solo ese usuario de Telegram puede entrar.
    """
    identificador = identificador.strip()
    for u in _cargar_usuarios():
        if identificador not in (u.get("codigo_estudiante"), u.get("id")):
            continue
        registrado = u.get("telegram_user_id")
        if registrado is not None and telegram_user_id is not None and registrado != telegram_user_id:
            logger.warning("Login rechazado: %s pertenece a otro usuario de Telegram.", identificador)
            return None
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


def _contexto_sesion(sesion: Sesion) -> str:
    if not sesion.autenticado:
        return (
            "Sesión actual: usuario PÚBLICO (sin login). No reveles notas, casos especiales ni "
            "información reservada. Si piden datos privados, indícales que usen /login."
        )
    codigo = sesion.codigo_estudiante or "(docente, sin código)"
    return (
        f"Sesión actual: {sesion.nombre} | rol={sesion.rol} | código={codigo}. "
        "Las tools de notas/solicitudes solo aceptan el código de esta sesión. "
        "Si intenta consultar otro código, la tool fallará y debes explicárselo."
    )


# --------------------------------------------------------------------------
# Control de acceso: RAG y tools por sesión
# --------------------------------------------------------------------------

class SessionRAG(CourseRAG):
    """CourseRAG que solo recupera los documentos permitidos para el rol de la sesión.

    Al sobreescribir `retrieve`, el filtro aplica a TODAS las rutas del orquestador:
    la tool buscar_documentos (modo agente) y CourseRAG.answer (ruta "documentos").
    """

    def __init__(self, base: CourseRAG, rol: str):
        super().__init__(base.llm, base.embedder, base.store, base.top_k, base.min_score)
        self.rol = rol

    def _permitido(self, source: str) -> bool:
        if self.rol == "docente":
            return True
        if source in DOCENTE_ONLY_DOCS:
            return False
        if self.rol == "estudiante":
            return True
        return source not in ESTUDIANTE_ONLY_DOCS  # público

    def retrieve(self, question: str) -> list[SearchResult]:
        vector = self.embedder.embed_query(question)
        # Se busca en todo el índice y se filtra ANTES de aplicar top_k: así los documentos
        # restringidos no desplazan a los permitidos.
        results = self.store.search(vector, len(self.store))
        allowed = [r for r in results if self._permitido(r.chunk.source) and r.score >= self.min_score]
        return allowed[: self.top_k]


def _tool_con_sesion(tool: Tool, sesion: Sesion) -> Tool:
    """Envuelve una tool de datos (MCP) para aplicar la sesión. Las tools locales no se envuelven."""
    if tool.origin == "local":
        return tool
    pide_codigo = "codigo" in tool.parameters.get("properties", {})

    async def run(arguments: dict[str, Any]) -> str:
        if sesion.rol == "publico":
            raise ToolExecutionError("Debes iniciar sesión con /login antes de consultar esta información.")
        if sesion.rol == "estudiante" and pide_codigo:
            if not sesion.codigo_estudiante:
                raise ToolExecutionError("Tu sesión no tiene un código de estudiante asociado.")
            # Se ignora el código que haya puesto el LLM: solo vale el de la sesión.
            arguments = {**arguments, "codigo": sesion.codigo_estudiante}
        return await tool.run(arguments)

    return Tool(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
        run=run,
        read_only=tool.read_only,
        origin=tool.origin,
    )


async def _rechazar(tool: Tool, arguments: dict[str, Any]) -> bool:
    # Telegram no tiene diálogo de confirmación interactivo: las ACCIONES se rechazan por defecto.
    return False


# --------------------------------------------------------------------------
# Core (singleton perezoso)
# --------------------------------------------------------------------------

class Core:
    def __init__(self) -> None:
        self._stack: AsyncExitStack | None = None
        self._llm: LLMClient | None = None
        self._rag: CourseRAG | None = None
        self._base_tools: list[Tool] = []
        self._lock = asyncio.Lock()

    async def iniciar(self) -> None:
        if self._stack is not None:
            return
        async with self._lock:
            if self._stack is not None:
                return
            stack = AsyncExitStack()
            try:
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
                await stack.aclose()
                raise  # no se marca como "fallido para siempre": se reintenta en el siguiente mensaje
            self._stack, self._llm, self._rag = stack, llm, rag
            self._base_tools = [*LOCAL_TOOLS, *mcp_tools]
            logger.info("Core iniciado: %d tools (locales + MCP)", len(self._base_tools))

    async def cerrar(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._base_tools = []
        self._rag = None
        self._llm = None

    def _orquestador(self, sesion: Sesion) -> Orchestrator:
        assert self._llm is not None and self._rag is not None
        rag = SessionRAG(self._rag, sesion.rol)
        tools = [_tool_con_sesion(t, sesion) for t in self._base_tools]
        return Orchestrator(self._llm, rag, tools, _rechazar)

    async def responder(self, pregunta: str, historial: list[Message], sesion: Sesion) -> str:
        try:
            if self._stack is None:
                await self.iniciar()
            orquestador = self._orquestador(sesion)
            # El contexto de sesión viaja como mensaje de sistema antes del historial del chat.
            contexto: list[Message] = [{"role": "system", "content": _contexto_sesion(sesion)}, *historial]
            respuesta = await orquestador.answer(pregunta, contexto, MODO)
        except LLMError as exc:
            logger.error("Error del LLM: %s", exc)
            return MSG_LLM_CAIDO
        except (ValueError, FileNotFoundError) as exc:
            logger.error("Error de configuración o índice: %s", exc)
            return "El asistente no está configurado correctamente. Avisa al administrador."
        except Exception:
            logger.exception("Error inesperado en responder() (¿servidor MCP caído?)")
            return MSG_ERROR

        texto = respuesta.text.strip() or "No pude generar una respuesta. Intenta reformular la pregunta."
        if respuesta.sources:
            texto += f"\n\nFuentes: {', '.join(respuesta.sources)}"
        return texto


_core: Core | None = None
_core_lock = asyncio.Lock()


async def _obtener_core() -> Core:
    global _core
    async with _core_lock:
        if _core is None:
            _core = Core()
        core = _core
    await core.iniciar()  # idempotente; si falla, el siguiente mensaje reintenta
    return core


# --------------------------------------------------------------------------
# API pública (sin dependencia de Telegram)
# --------------------------------------------------------------------------

async def responder(
    pregunta: str,
    historial: list[Message] | None = None,
    sesion: Sesion | None = None,
) -> str:
    """Punto de entrada del núcleo. No depende de Telegram y nunca lanza excepciones por LLM/MCP."""
    try:
        core = await _obtener_core()
    except Exception:
        logger.exception("No se pudo iniciar el core")
        return "El asistente no pudo iniciar (LLM, índice o servidor MCP). Intenta de nuevo en unos minutos."
    return await core.responder(pregunta, historial or [], sesion or Sesion())


async def cerrar_core() -> None:
    global _core
    if _core is not None:
        await _core.cerrar()
        _core = None