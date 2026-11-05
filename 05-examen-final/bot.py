"""Adaptador Telegram: traduce entre Telegram y el núcleo (core.py).

No contiene lógica de negocio; solo:
  - guarda el estado de cada chat (historial + sesión)
  - traduce comandos a llamadas de core.py
  - envía las respuestas por Telegram

Variables de entorno (.env o entorno):
    TELEGRAM_BOT_TOKEN       token del bot de BotFather
    LLM_API_KEY, LLM_MODEL, LLM_PROVIDER   (ya usadas por config.py)
    LIVENESS_TIMEOUT_SECONDS segundos para responder el reto (por defecto 120)
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import face
from core import (
    MAX_HISTORIAL,
    Sesion,
    buscar_usuario,
    cerrar_core,
    responder,
    sesion_desde_usuario,
    sesion_publica,
)

logger = logging.getLogger(__name__)

# Prueba de vida: segundos desde /login hasta recibir la video-nota.
LIVENESS_TIMEOUT_S = int(os.getenv("LIVENESS_TIMEOUT_SECONDS", "120"))

# Instrucciones que ve el usuario para cada reto (las claves viven en face.RETOS_LIVENESS).
RETO_MENSAJE = {
    "girar_izquierda": "gira la cabeza hacia tu izquierda",
    "girar_derecha": "gira la cabeza hacia tu derecha",
    "acercar": "acerca el rostro a la cámara",
}

# Estado por chat: chat_id -> {"historial": [...], "sesion": Sesion,
#                              "login_pendiente": {...} | None}
_ESTADO: dict[int, dict] = {}


def _estado(chat_id: int) -> dict:
    if chat_id not in _ESTADO:
        _ESTADO[chat_id] = {
            "historial": [],
            "sesion": sesion_publica(),
            "login_pendiente": None,
        }
    return _ESTADO[chat_id]


# --------------------------------------------------------------------------
# Comandos
# --------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    sesion = _estado(update.effective_chat.id)["sesion"]
    nombre = (
        sesion.nombre
        if sesion.autenticado
        else (update.effective_user.first_name or "estudiante")
    )
    await update.message.reply_text(
        f"¡Hola, {nombre}! 👋\n\n"
        "Soy el Asistente del Curso de Inteligencia Artificial. Puedo responder:\n"
        "• Fechas y eventos del curso\n"
        "• Políticas de evaluación y entrega\n"
        "• Tus notas (con /login)\n"
        "• Cálculos de nota final\n\n"
        "Usa /ayuda para ver todos los comandos."
    )


async def cmd_quiensoy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    sesion: Sesion = _estado(chat_id)["sesion"]
    lineas = [
        f"Telegram user ID: {update.effective_user.id}",
        f"Chat ID: {chat_id}",
    ]
    if sesion.autenticado:
        lineas.append(f"Estado: autenticado como {sesion.rol}")
        lineas.append(f"Nombre: {sesion.nombre}")
        if sesion.codigo_estudiante:
            lineas.append(f"Código: {sesion.codigo_estudiante}")
    else:
        lineas.append("Estado: público (sin sesión)")
    await update.message.reply_text("\n".join(lineas))


async def cmd_login(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Paso 1 del login: identifica al usuario y genera un reto de prueba de vida.

    Dos factores: posesión (la cuenta de Telegram debe ser la vinculada al
    usuario) + biométrico (rostro verificado 1:1 contra el enrolamiento) +
    prueba de vida (el video debe mostrar el reto pedido).
    """
    chat_id = update.effective_chat.id
    estado = _estado(chat_id)
    if not context.args:
        await update.message.reply_text(
            "Uso: /login <código>\n"
            "Ejemplos:\n"
            "  /login 000548492   (estudiante)\n"
            "  /login integrante_2 (docente)\n\n"
            "Después te pediré una VIDEO-NOTA con una prueba de vida."
        )
        return

    usuario = buscar_usuario(context.args[0])
    if usuario is None:
        await update.message.reply_text(
            "No encontré un usuario con ese identificador."
        )
        return

    # Factor de posesión: solo el dueño de la cuenta vinculada puede continuar.
    if usuario.get("telegram_user_id") != update.effective_user.id:
        face.registrar_auditoria(
            usuario["id"], update.effective_user.id, "cuenta_no_vinculada", None
        )
        await update.message.reply_text(
            "Esta cuenta de Telegram no está vinculada a ese usuario."
        )
        return
    if not face.esta_enrolado(usuario["id"]):
        await update.message.reply_text(
            "Tu rostro aún no está enrolado. El enrolamiento se hace de forma "
            "local con enroll.py (no se puede desde el chat)."
        )
        return

    reto = random.choice(face.RETOS_LIVENESS)
    estado["login_pendiente"] = {
        "usuario": usuario,
        "telegram_user_id": update.effective_user.id,
        "reto": reto,
        "inicio": time.monotonic(),
        "usado": False,
    }
    await update.message.reply_text(
        "Verificación de vida 🔒\n\n"
        f"Envíame una VIDEO-NOTA (video circular) donde {RETO_MENSAJE[reto]}.\n\n"
        f"• Tienes {LIVENESS_TIMEOUT_S} segundos desde ahora.\n"
        "• Debes aparecer tú solo y mirando a la cámara.\n"
        "• Una foto NO sirve: debe ser una video-nota grabada ahora mismo.\n"
        "• El video no debe ser reenviado.\n\n"
        "Si fallas, deberás empezar de nuevo con /login."
    )


async def on_foto(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """R3: las fotos ya no sirven para el login; se rechazan y se pide video-nota."""
    estado = _estado(update.effective_chat.id)
    if estado.get("login_pendiente") is not None:
        await update.message.reply_text(
            "Para iniciar sesión debes enviar una VIDEO-NOTA (video circular), "
            "no una foto. Graba un video con la instrucción indicada."
        )
    else:
        await update.message.reply_text(
            "No proceso fotos. Para iniciar sesión usa /login <código> y "
            "envía una video-nota."
        )


async def _safe_reply(update: Update, texto: str) -> None:
    """Responde sin propagar errores de red (si no se puede, al menos queda en el log)."""
    try:
        await update.message.reply_text(texto)
    except Exception:
        logger.exception("No pude responder al usuario (probablemente red)")


async def on_video_nota(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Paso 2 del login: procesa el video-nota y verifica la prueba de vida."""
    chat_id = update.effective_chat.id
    estado = _estado(chat_id)
    pendiente = estado.get("login_pendiente")

    if pendiente is None:
        await _safe_reply(update, "Recibí un video, pero no hay un login en curso. Usa /login <código>.")
        return

    if update.effective_user.id != pendiente["telegram_user_id"]:
        return

    if pendiente["usado"]:
        await _safe_reply(update, "Este reto ya fue usado. Inicia de nuevo con /login <código>.")
        return

    usuario = pendiente["usuario"]
    uid = update.effective_user.id

    es_reenviado = (
        getattr(update.message, "forward_origin", None) is not None
        or getattr(update.message, "is_forwarded", False)
    )
    if es_reenviado:
        face.registrar_auditoria(usuario["id"], uid, "video_reenviado", None)
        await _safe_reply(update, "No acepto videos reenviados. Graba una video-nota nueva.")
        return

    if time.monotonic() - pendiente["inicio"] > LIVENESS_TIMEOUT_S:
        estado["login_pendiente"] = None
        face.registrar_auditoria(usuario["id"], uid, "reto_vencido", None)
        await _safe_reply(update, "El reto expiró. Vuelve a empezar con /login <código>.")
        return

    # Marca el reto como usado ANTES de verificar (un reto sirve una sola vez).
    pendiente["usado"] = True
    reto = pendiente["reto"]

    # Cosmético: si falla por red, no rompe la verificación.
    try:
        await update.message.chat.send_action("typing")
    except Exception:
        logger.debug("send_action falló (red); continuamos igual", exc_info=True)

    try:
        archivo = await update.message.video_note.get_file()
        datos = await archivo.download_as_bytearray()
        resultado = await asyncio.to_thread(
            face.verificar_liveness, bytes(datos), reto, usuario["id"]
        )
    except Exception:
        logger.exception("Error en la verificación de prueba de vida")
        estado["login_pendiente"] = None
        face.registrar_auditoria(usuario["id"], uid, "error_interno", None)
        await _safe_reply(update, "No se pudo verificar la prueba de vida. Intenta de nuevo con /login.")
        return

    if resultado.ok:
        face.registrar_auditoria(usuario["id"], uid, "aceptado", resultado.score)
        estado["login_pendiente"] = None
        estado["sesion"] = sesion_desde_usuario(usuario, uid)
        estado["historial"] = []
        await _safe_reply(update, f"✅ Sesión iniciada como {usuario['nombre']} ({usuario['rol']}).")
        return

    face.registrar_auditoria(usuario["id"], uid, f"liveness_{resultado.motivo}", resultado.score)
    estado["login_pendiente"] = None
    await _safe_reply(update, "No se pudo verificar la prueba de vida. Intenta de nuevo con /login.")


async def cmd_logout(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    estado = _estado(chat_id)
    estado["sesion"] = sesion_publica(update.effective_user.id)
    estado["historial"] = []
    estado["login_pendiente"] = None
    await update.message.reply_text(
        "Sesión cerrada. Vuelves a ser un usuario público."
    )


async def cmd_ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Comandos disponibles:\n"
        "/start — presentación\n"
        "/login <código> — iniciar sesión (código + video-nota con prueba de vida)\n"
        "/quiensoy — ver tu sesión actual\n"
        "/logout — cerrar sesión\n"
        "/ayuda — este mensaje\n\n"
        "También puedes escribir preguntas en lenguaje natural."
    )


# --------------------------------------------------------------------------
# Mensajes de texto
# --------------------------------------------------------------------------

async def on_texto(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    texto = (update.message.text or "").strip()
    if not texto:
        return

    estado = _estado(chat_id)
    await update.message.chat.send_action("typing")

    try:
        respuesta = await responder(texto, estado["historial"], estado["sesion"])
    except Exception:
        logger.exception("Error al procesar mensaje en el core")
        respuesta = "Ocurrió un error inesperado. Intenta de nuevo en un momento."

    estado["historial"].append({"role": "user", "content": texto})
    estado["historial"].append({"role": "assistant", "content": respuesta})
    if len(estado["historial"]) > MAX_HISTORIAL:
        estado["historial"] = estado["historial"][-MAX_HISTORIAL:]

    # Telegram limita a 4096 caracteres por mensaje.
    for i in range(0, len(respuesta), 4000):
        await update.message.reply_text(respuesta[i : i + 4000])


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    err = context.error
    logger.error("Error en el bot: %r", err, exc_info=err)

    # Los errores de red con Telegram son transitorios: un reintento suave evita
    # que el usuario se quede sin respuesta.
    if isinstance(err, telegram.error.NetworkError):
        await asyncio.sleep(2)

    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "Ocurrió un error de conexión con Telegram. Intenta de nuevo."
            )
        except Exception:
            logger.exception("Tampoco pude responder tras error de red")


# --------------------------------------------------------------------------
# Arranque
# --------------------------------------------------------------------------

async def _post_init(app: Application) -> None:
    """Pre-calienta el core para que el primer mensaje no sea lento."""
    try:
        from core import _obtener_core

        await _obtener_core()
        logger.info("Core pre-iniciado correctamente")
    except Exception:
        logger.exception(
            "Core no se pudo pre-iniciar; se reintentará al primer mensaje"
        )





async def _post_shutdown(app: Application) -> None:
    try:
        await cerrar_core()
        logger.info("Core cerrado")
    except Exception:
        logger.exception("Error al cerrar el core")


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )

    # Cargar .env (mismo mecanismo que config.py)
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print(
            "Falta TELEGRAM_BOT_TOKEN. Agrégalo a .env o como variable de entorno."
        )
        return

    app = (
        Application.builder()
        .token(token)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("quiensoy", cmd_quiensoy))
    app.add_handler(CommandHandler("login", cmd_login))
    app.add_handler(CommandHandler("logout", cmd_logout))
    app.add_handler(CommandHandler("ayuda", cmd_ayuda))
    # R3: video-notas para prueba de vida; fotos se rechazan.
    app.add_handler(MessageHandler(filters.VIDEO_NOTE, on_video_nota))
    app.add_handler(MessageHandler(filters.PHOTO, on_foto))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_texto))
    app.add_error_handler(on_error)

    print("Bot iniciado. Ctrl+C para detener.")
    app.run_polling()


if __name__ == "__main__":
    main()

