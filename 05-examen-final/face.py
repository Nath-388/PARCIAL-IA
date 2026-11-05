from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

LAB_DIR = Path(__file__).resolve().parent
MODELOS_DIR = LAB_DIR / "modelos"
ROSTROS_DIR = LAB_DIR / "rostros"
AUDITORIA = LAB_DIR / "auditoria_login.jsonl"

YUNET = MODELOS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE = MODELOS_DIR / "face_recognition_sface_2021dec.onnx"

# Punto de partida de la guía. Reemplazar por el valor elegido con calibrar.py
# (variable de entorno FACE_UMBRAL en .env).
UMBRAL_REFERENCIA = 0.363
MIN_FOTOS = 3
MAX_FOTOS = 5

# Confianza mínima del detector para considerar que una detección es un rostro.
_SCORE_DETECCION = 0.8

# --- Prueba de vida (R3) ---------------------------------------------------
# Retos disponibles: se elige uno al azar en /login. La instrucción de cada reto
# la arma bot.py (aquí solo viven las claves).
RETOS_LIVENESS = ("girar_izquierda", "girar_derecha", "acercar")

# Umbrales de la prueba de vida (ajustables igual que FACE_UMBRAL).
UMBRAL_MISMO_ROSTRO = 0.30   # similitud mínima entre frames del mismo video
DELTA_GIRA_MIN = 0.10        # desplazamiento horizontal mínimo de la nariz
                             # (fracción del ancho del rostro)
RATIO_ACERCA_MIN = 1.20      # crecimiento mínimo del área del rostro
MIN_FRAMES_VIDEO = 5
MAX_FRAMES_ANALIZAR = 30


class RostroInvalido(Exception):
    """La imagen no sirve: sin rostro, con varios rostros o ilegible."""


@dataclass
class ResultadoLiveness:
    """Resultado de la prueba de vida. `motivo` es para auditoría, nunca para el usuario."""

    ok: bool
    motivo: str
    score: float | None = None  # similitud del frame más frontal contra el usuario enrolado


def umbral() -> float:
    try:
        from config import load_security_settings
        return load_security_settings().face_match_threshold
    except (ValueError, TypeError):
        return UMBRAL_REFERENCIA


# --------------------------------------------------------------------------
# Modelos (carga perezosa)
# --------------------------------------------------------------------------

_detector = None
_reconocedor = None


def _cargar_modelos():
    global _detector, _reconocedor
    if _detector is None:
        for ruta in (YUNET, SFACE):
            if not ruta.exists():
                raise FileNotFoundError(
                    f"Falta el modelo {ruta.name} en {MODELOS_DIR}. "
                    "Descárgalo del OpenCV Zoo (carpeta modelos/ no se versiona)."
                )
        _detector = cv2.FaceDetectorYN.create(
            str(YUNET), "", (320, 320), _SCORE_DETECCION, 0.3, 5000
        )
        _reconocedor = cv2.FaceRecognizerSF.create(str(SFACE), "")
    return _detector, _reconocedor


def _detectar(img_bgr: np.ndarray) -> np.ndarray:
    """Devuelve la matriz de detecciones (N x 15); vacía si no hay rostros."""
    detector, _ = _cargar_modelos()
    h, w = img_bgr.shape[:2]
    detector.setInputSize((w, h))
    _, caras = detector.detect(img_bgr)
    return caras if caras is not None else np.empty((0, 15), dtype=np.float32)


# --------------------------------------------------------------------------
# Imagen -> embedding
# --------------------------------------------------------------------------

def decodificar(datos: bytes | bytearray) -> np.ndarray:
    """Convierte bytes de una imagen (JPEG/PNG) en un arreglo BGR de OpenCV."""
    arr = np.frombuffer(bytes(datos), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RostroInvalido("No pude leer la imagen.")
    return img


def embedding(img_bgr: np.ndarray) -> np.ndarray:
    """Embedding (128,) del único rostro de la imagen.

    Lanza RostroInvalido si no hay rostro o si hay más de uno.
    """
    caras = _detectar(img_bgr)
    if len(caras) == 0:
        raise RostroInvalido("No detecté ningún rostro en la imagen.")
    if len(caras) > 1:
        raise RostroInvalido(
            f"Detecté {len(caras)} rostros; la foto debe tener solo uno."
        )
    _, reconocedor = _cargar_modelos()
    alineada = reconocedor.alignCrop(img_bgr, caras[0])
    return reconocedor.feature(alineada).reshape(-1).astype(np.float32)


def similitud(a: np.ndarray, b: np.ndarray) -> float:
    """Similitud coseno entre dos embeddings (mayor = más parecidos)."""
    _, reconocedor = _cargar_modelos()
    return float(
        reconocedor.match(
            a.reshape(1, -1), b.reshape(1, -1), cv2.FaceRecognizerSF_FR_COSINE
        )
    )


# --------------------------------------------------------------------------
# Almacenamiento: solo embeddings
# --------------------------------------------------------------------------

def _ruta_rostros(usuario_id: str) -> Path:
    seguro = "".join(c for c in usuario_id if c.isalnum() or c in "-_")
    if not seguro or seguro != usuario_id:
        raise ValueError(f"id de usuario no válido: {usuario_id!r}")
    return ROSTROS_DIR / f"{seguro}.npy"


def guardar_embeddings(usuario_id: str, embs: list[np.ndarray]) -> Path:
    ROSTROS_DIR.mkdir(exist_ok=True)
    ruta = _ruta_rostros(usuario_id)
    np.save(ruta, np.stack(embs).astype(np.float32))
    return ruta


def cargar_embeddings(usuario_id: str) -> np.ndarray | None:
    ruta = _ruta_rostros(usuario_id)
    if not ruta.exists():
        return None
    return np.load(ruta)


def esta_enrolado(usuario_id: str) -> bool:
    return _ruta_rostros(usuario_id).exists()


# --------------------------------------------------------------------------
# Verificación 1:1
# --------------------------------------------------------------------------

def puntaje(img_bgr: np.ndarray, usuario_id: str) -> float:
    """Mejor similitud entre el rostro de la foto y las plantillas del usuario.

    Solo se comparan los embeddings de `usuario_id` (1:1), nunca los de otros.
    """
    plantillas = cargar_embeddings(usuario_id)
    if plantillas is None:
        raise LookupError("El usuario no tiene rostro enrolado.")
    emb = embedding(img_bgr)
    return max(similitud(emb, p) for p in plantillas)


def verificar(img_bgr: np.ndarray, usuario_id: str) -> tuple[bool, float]:
    """(aceptado, puntaje). Lanza RostroInvalido o LookupError."""
    p = puntaje(img_bgr, usuario_id)
    return p >= umbral(), p


def registrar_auditoria(
    usuario_id: str, telegram_user_id: int | None, resultado: str, score: float | None
) -> None:
    """Deja constancia del intento (sin fotos ni embeddings)."""
    linea = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "usuario": usuario_id,
        "telegram_user_id": telegram_user_id,
        "resultado": resultado,
        "score": None if score is None else round(score, 4),
        "umbral": umbral(),
    }
    with AUDITORIA.open("a", encoding="utf-8") as f:
        f.write(json.dumps(linea, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Prueba de vida (R3): análisis de una video-nota
# --------------------------------------------------------------------------

def _extraer_frames(
    video_bytes: bytes, max_frames: int = MAX_FRAMES_ANALIZAR
) -> list[np.ndarray]:
    """Extrae hasta `max_frames` frames distribuidos a lo largo del video."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        path = tmp.name
    cap = cv2.VideoCapture(path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // max_frames) if total > 0 else 3
        frames: list[np.ndarray] = []
        idx = 0
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                frames.append(frame)
            idx += 1
            # Tope de seguridad por si CAP_PROP_FRAME_COUNT venía mal.
            if idx > max_frames * 10:
                break
        return frames
    finally:
        cap.release()
        try:
            os.unlink(path)
        except OSError:
            pass


def _desplazamiento_horizontal(caras: list[np.ndarray]) -> float:
    """Rango (máx − mín) de (nariz_x − centro_ojos_x) / ancho_rostro.

    Cuando la persona gira la cabeza, la nariz se desplaza respecto al punto
    medio de los ojos. Sirve para ambos retos de giro (izquierda/derecha): el
    video debe mostrar un giro significativo en cualquier dirección.
    """
    valores = []
    for c in caras:
        ancho = float(c[2]) or 1e-6
        ojo_x = (float(c[4]) + float(c[6])) / 2.0
        nariz_x = float(c[8])
        valores.append((nariz_x - ojo_x) / ancho)
    return float(max(valores) - min(valores))


def _crecimiento_rostro(caras: list[np.ndarray]) -> float:
    """Ratio entre el área máxima y la mínima del rostro a lo largo del video."""
    areas = [float(c[2]) * float(c[3]) for c in caras]
    return max(areas) / max(min(areas), 1e-6)


def _frame_mas_frontal(caras: list[np.ndarray]) -> int:
    """Índice del frame donde la nariz está más centrada respecto a los ojos."""
    mejor = 0
    mejor_valor = float("inf")
    for i, c in enumerate(caras):
        ojo_x = (float(c[4]) + float(c[6])) / 2.0
        ancho = float(c[2]) or 1e-6
        v = abs((float(c[8]) - ojo_x) / ancho)
        if v < mejor_valor:
            mejor_valor = v
            mejor = i
    return mejor


def verificar_liveness(
    video_bytes: bytes, reto: str, usuario_id: str
) -> ResultadoLiveness:
    """Verifica la prueba de vida a partir de los frames de la video-nota.

    Comprueba:
      1. exactamente un rostro en cada frame analizado;
      2. que sea la misma persona a lo largo del video;
      3. que el movimiento pedido ocurra (giro horizontal o acercamiento);
      4. que el frame más frontal supere el umbral contra el usuario enrolado.

    Devuelve `ResultadoLiveness` con un motivo corto para auditoría. Nunca lanza
    por datos del usuario: cualquier fallo se reporta con ok=False.
    """
    frames = _extraer_frames(video_bytes)
    if len(frames) < MIN_FRAMES_VIDEO:
        return ResultadoLiveness(False, "frames_insuficientes")

    caras_por_frame = [_detectar(f) for f in frames]

    # (1) Exactamente un rostro en todos los frames analizados.
    if any(len(c) != 1 for c in caras_por_frame):
        return ResultadoLiveness(False, "no_exactamente_un_rostro")

    # Embeddings por frame (una vez; se reutilizan para (2) y (4)).
    _, reconocedor = _cargar_modelos()
    embs = [
        reconocedor.feature(
            reconocedor.alignCrop(frames[i], caras_por_frame[i][0])
        ).reshape(-1).astype(np.float32)
        for i in range(len(frames))
    ]

    # (2) Misma persona en todo el video.
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            if similitud(embs[i], embs[j]) < UMBRAL_MISMO_ROSTRO:
                return ResultadoLiveness(False, "rostro_inconsistente")

    # (3) Movimiento pedido.
    caras = [c[0] for c in caras_por_frame]
    if reto in ("girar_izquierda", "girar_derecha"):
        if _desplazamiento_horizontal(caras) < DELTA_GIRA_MIN:
            return ResultadoLiveness(False, "sin_giro")
    elif reto == "acercar":
        if _crecimiento_rostro(caras) < RATIO_ACERCA_MIN:
            return ResultadoLiveness(False, "sin_acercamiento")
    else:
        return ResultadoLiveness(False, "reto_desconocido")

    # (4) 1:1 contra el usuario enrolado, con el frame más frontal.
    plantillas = cargar_embeddings(usuario_id)
    if plantillas is None:
        return ResultadoLiveness(False, "sin_enrolamiento")
    idx = _frame_mas_frontal(caras)
    score = max(similitud(embs[idx], p) for p in plantillas)
    if score < umbral():
        return ResultadoLiveness(False, "no_coincide", score)

    return ResultadoLiveness(True, "ok", score)