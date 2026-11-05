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


# Reemplaza DOCENTE_ONLY_DOCS y ESTUDIANTE_ONLY_DOCS por esta tabla:
NIVELES_PERMITIDOS_POR_ROL = {
    "publico": {"publico"},
    "estudiante": {"publico", "estudiante"},
    "docente": {"publico", "estudiante", "docente"},
}

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

def buscar_usuario(identificador: str) -> dict | None:
    """Busca un usuario en usuarios.json por código de estudiante o id interno.

    Devuelve el dict crudo del JSON o None. autenticar() usa esta función
    internamente para no duplicar la búsqueda.
    """
    if not identificador:
        return None
    identificador = identificador.strip()
    for u in _cargar_usuarios():
        if u.get("codigo_estudiante") == identificador or u.get("id") == identificador:
            return u
    return None


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


def sesion_desde_usuario(usuario: dict, telegram_user_id: int | None = None) -> Sesion:
    """Crea la Sesion de un usuario ya verificado (código + rostro)."""
    return Sesion(
        telegram_user_id=telegram_user_id,
        usuario=usuario,
        rol=usuario["rol"],
        codigo_estudiante=usuario.get("codigo_estudiante"),
        nombre=usuario["nombre"],
    )


def autenticar(identificador: str, telegram_user_id: int | None = None) -> Sesion | None:
    """Autenticación SIN biometría: solo para pruebas locales (test_core.py).

    El bot de Telegram NO la usa: allí el login exige además la verificación
    facial 1:1 (ver bot.py y face.py).
    """
    usuario = buscar_usuario(identificador)
    if usuario is None:
        return None
    return sesion_desde_usuario(usuario, telegram_user_id)


def sesion_publica(telegram_user_id: int | None = None) -> Sesion:
    return Sesion(telegram_user_id=telegram_user_id)


def _contexto_sesion(sesion: Sesion) -> str:
    """Instrucciones de seguridad y alcance de datos específicas para cada rol."""
    if not sesion.autenticado or sesion.rol == "publico":
        return (
            "Sesión actual: usuario PÚBLICO (sin login). No reveles notas, solicitudes, "
            "observaciones docentes ni información reservada. Si piden datos privados, "
            "indícales que inicien sesión con /login."
        )

    if sesion.rol == "estudiante":
        return (
            f"Sesión actual: {sesion.nombre} | rol=estudiante | "
            f"código={sesion.codigo_estudiante}. "
            "SEGURIDAD: solo puedes consultar las notas y solicitudes del estudiante "
            "autenticado. Usa consultar_mis_notas() y consultar_mis_solicitudes() sin "
            "parámetros de código. Nunca intentes consultar datos de otro estudiante. "
            "No reveles observaciones_docente."
        )

    if sesion.rol == "docente":
        return (
            f"Sesión actual: {sesion.nombre} | rol=docente. "
            "PERMISOS DOCENTES: puedes consultar las notas y solicitudes de CUALQUIER "
            "estudiante del curso. Cuando el usuario indique un código, debes utilizar "
            "consultar_notas(codigo) o consultar_solicitudes(codigo), pasando exactamente "
            "el código solicitado. No limites la consulta al código asociado a la cuenta "
            "del docente. Puedes ver observaciones_docente. Para modificar una nota usa "
            "registrar_nota(codigo, evaluacion, valor), que requiere confirmación explícita."
        )

    return (
        f"Sesión actual: {sesion.nombre} | rol={sesion.rol}. "
        "No ejecutes operaciones fuera de las herramientas autorizadas para este rol."
    )


# --------------------------------------------------------------------------
# Control de acceso: RAG y tools por sesión
# --------------------------------------------------------------------------

class SessionRAG(CourseRAG):
    """CourseRAG que solo recupera los niveles permitidos para el rol de la sesión.

    El filtro se aplica a TODAS las rutas del orquestador (tool `buscar_documentos`
    y `CourseRAG.answer`) porque ambas pasan por `retrieve`.
    """

    def __init__(self, base: CourseRAG, rol: str):
        super().__init__(base.llm, base.embedder, base.store, base.top_k, base.min_score)
        self.rol = rol
        # True si, en la última consulta, había evidencia relevante pero restringida.
        self.bloqueado_por_rol = False

    def _permitido(self, nivel: str) -> bool:
        return nivel in NIVELES_PERMITIDOS_POR_ROL.get(self.rol, {"publico"})

    def retrieve(self, question: str) -> list[SearchResult]:
        self.bloqueado_por_rol = False
        vector = self.embedder.embed_query(question)
        results = self.store.search(vector, len(self.store))
        permitidos = [
            r for r in results
            if self._permitido(r.chunk.nivel) and r.score >= self.min_score
        ]
        if not permitidos:
            # ¿Existía evidencia que el rol no puede ver? Entonces el aviso cambia:
            # no es "no encontré", es "no tienes permiso".
            self.bloqueado_por_rol = any(
                (not self._permitido(r.chunk.nivel)) and r.score >= self.min_score
                for r in results
            )
        return permitidos[: self.top_k]



def _schema_sin_codigo(schema: dict[str, Any]) -> dict[str, Any]:
    """Copia el JSON Schema quitando el código, que nunca debe pedir el modelo al estudiante."""
    copia = json.loads(json.dumps(schema))
    copia.get("properties", {}).pop("codigo", None)
    copia["required"] = [x for x in copia.get("required", []) if x != "codigo"]
    return copia


def _tools_para_sesion(base_tools: list[Tool], sesion: Sesion) -> list[Tool]:
    """Lista blanca de herramientas por rol. La autorización se aplica antes del LLM."""
    locales = [t for t in base_tools if t.origin == "local"]
    mcp = {t.name: t for t in base_tools if t.origin != "local"}

    if sesion.rol == "publico":
        return locales

    if sesion.rol == "estudiante":
        if not sesion.codigo_estudiante:
            return locales
        permitidas = []
        # Los alias de estudiante no exponen ni aceptan el código.
        for alias, original in (
            ("consultar_mis_notas", "consultar_notas"),
            ("consultar_mis_solicitudes", "consultar_solicitudes"),
        ):
            tool = mcp.get(original)
            if tool is None:
                continue

            async def run(arguments: dict[str, Any], _tool=tool) -> str:
                raw = await _tool.run({**arguments, "codigo": sesion.codigo_estudiante})
                if _tool.name == "consultar_notas":
                    try:
                        data = json.loads(raw)
                        data.pop("observaciones_docente", None)
                        return json.dumps(data, ensure_ascii=False)
                    except (TypeError, ValueError):
                        raise ToolExecutionError("No fue posible procesar la respuesta de notas.")
                return raw

            permitidas.append(Tool(
                name=alias,
                description=(
                    "Consulta únicamente la información del estudiante autenticado en esta sesión. "
                    "No solicites ni aceptes códigos de estudiante."
                ),
                parameters=_schema_sin_codigo(tool.parameters),
                run=run,
                read_only=True,
                origin=tool.origin,
            ))

        accion = mcp.get("solicitar_supletorio")
        if accion is not None:
            async def solicitar(arguments: dict[str, Any], _tool=accion) -> str:
                return await _tool.run({**arguments, "codigo": sesion.codigo_estudiante})

            schema = _schema_sin_codigo(accion.parameters)
            permitidas.append(Tool(
                name="solicitar_supletorio",
                description=(
                    "Registra una solicitud de supletorio para el estudiante autenticado. "
                    "Requiere confirmación explícita del usuario en Telegram antes de ejecutarse."
                ),
                parameters=schema,
                run=solicitar,
                read_only=False,
                origin=accion.origin,
            ))
        return [*locales, *permitidas]

    if sesion.rol == "docente":
        nombres = ("consultar_notas", "consultar_solicitudes", "registrar_nota")
        return [*locales, *(mcp[n] for n in nombres if n in mcp)]

    return locales


async def _rechazar(tool: Tool, arguments: dict[str, Any]) -> bool:
    # Sin una interfaz de confirmación explícita, las acciones se deniegan.
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

    def _orquestador(self, sesion: Sesion) -> tuple[Orchestrator, SessionRAG]:
        assert self._llm is not None and self._rag is not None
        rag = SessionRAG(self._rag, sesion.rol)
        tools = _tools_para_sesion(self._base_tools, sesion)
        return Orchestrator(self._llm, rag, tools, _rechazar), rag

    async def responder(
        self, pregunta: str, historial: list[Message], sesion: Sesion,
        confirm: Any = None,
    ) -> str:
        try:
            if self._stack is None:
                await self.iniciar()
            orquestador, rag = self._orquestador(sesion)
            if confirm is not None:
                # El callback solo autoriza acciones de esta interacción de Telegram.
                orquestador.agent.confirm = confirm
                orquestador.actions_agent.confirm = confirm
            contexto: list[Message] = [
                {"role": "system", "content": _contexto_sesion(sesion)}, *historial
            ]
            respuesta = await orquestador.answer(pregunta, contexto, MODO)
        except LLMError:
            logger.exception("El proveedor LLM falló al procesar la solicitud")
            return MSG_LLM_CAIDO
        except Exception:
            logger.exception("Error inesperado en el orquestador")
            return MSG_ERROR

        texto = respuesta.text.strip() or "No pude generar una respuesta. Intenta reformular la pregunta."

        # Si la evidencia existía pero el rol no puede verla, se lo decimos sin revelar contenido.
        if rag.bloqueado_por_rol and not respuesta.sources:
            if sesion.rol == "publico":
                texto += (
                    "\n\n(La información que buscas está en documentos que requieren "
                    "iniciar sesión. Usa /login <código> para consultarla.)"
                )
            else:
                texto += (
                    "\n\n(La información que buscas está en documentos reservados "
                    "para otro rol, por lo que no puedo mostrarla en esta sesión.)"
                )

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
    confirm: Any = None,
) -> str:
    """Punto de entrada del núcleo. No depende de Telegram y nunca lanza excepciones por LLM/MCP."""
    try:
        core = await _obtener_core()
    except Exception:
        logger.exception("No se pudo iniciar el core")
        return "El asistente no pudo iniciar (LLM, índice o servidor MCP). Intenta de nuevo en unos minutos."
    return await core.responder(pregunta, historial or [], sesion or Sesion(), confirm)


async def cerrar_core() -> None:
    global _core
    if _core is not None:
        await _core.cerrar()
        _core = None






