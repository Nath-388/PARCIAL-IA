from __future__ import annotations

import json
import os
import time
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


class RostroInvalido(Exception):
    """La imagen no sirve: sin rostro, con varios rostros o ilegible."""


def umbral() -> float:
    try:
        return float(os.getenv("FACE_UMBRAL", UMBRAL_REFERENCIA))
    except ValueError:
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
