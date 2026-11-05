"""Adaptador Telegram: traduce entre Telegram y el núcleo (core.py).

No contiene lógica de negocio; solo:
  - guarda el estado de cada chat (historial + sesión)
  - traduce comandos a llamadas de core.py
  - envía las respuestas por Telegram

Variables de entorno (.env o entorno):
    TELEGRAM_BOT_TOKEN   token del bot de BotFather
    LLM_API_KEY, LLM_MODEL, LLM_PROVIDER   (ya usadas por config.py)
"""

from __future__ import annotations

import logging
import os

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from core import (
    MAX_HISTORIAL,
    Sesion,
    autenticar,
    cerrar_core,
    responder,
    sesion_publica,
)

logger = logging.getLogger(__name__)

# Estado por chat: chat_id -> {"historial": [...], "sesion": Sesion}
_ESTADO: dict[int, dict] = {}


def _estado(chat_id: int) -> dict:
    if chat_id not in _ESTADO:
        _ESTADO[chat_id] = {"historial": [], "sesion": sesion_publica()}
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
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text(
            "Uso: /login <código>\n"
            "Ejemplos:\n"
            "  /login 000548492   (estudiante)\n"
            "  /login integrante_2 (docente)"
        )
        return
    sesion = autenticar(context.args[0], update.effective_user.id)
    if sesion is None:
        await update.message.reply_text(
            "No encontré un usuario con ese identificador."
        )
        return
    estado = _estado(chat_id)
    estado["sesion"] = sesion
    # El historial de la sesión anterior no aplica a la nueva identidad.
    estado["historial"] = []
    await update.message.reply_text(
        f"✅ Sesión iniciada como {sesion.nombre} ({sesion.rol})."
    )


async def cmd_logout(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    estado = _estado(chat_id)
    estado["sesion"] = sesion_publica(update.effective_user.id)
    estado["historial"] = []
    await update.message.reply_text(
        "Sesión cerrada. Vuelves a ser un usuario público."
    )


async def cmd_ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Comandos disponibles:\n"
        "/start — presentación\n"
        "/login <código> — iniciar sesión\n"
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
    logger.exception("Error en el bot", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "Ocurrió un error inesperado. Intenta de nuevo."
            )
        except Exception:
            logger.exception("No se pudo responder al usuario tras error")


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
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_texto))
    app.add_error_handler(on_error)

    print("Bot iniciado. Ctrl+C para detener.")
    app.run_polling()


if __name__ == "__main__":
    main()