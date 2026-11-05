"""Enrolamiento facial LOCAL (R2). No existe equivalente en el chat de Telegram.

Uso:
    uv run python enroll.py --id integrante_1 --telegram-user-id 123456789 \
        fotos/a.jpg fotos/b.jpg fotos/c.jpg

Recibe el id del usuario (el de usuarios.json), su telegram_user_id y de 3 a 5
fotos. Guarda SOLO los embeddings en rostros/<id>.npy (las fotos nunca se
copian) y vincula el telegram_user_id al usuario en usuarios.json.

Por qué no se enrola desde Telegram: el rostro enrolado es la referencia contra
la que se verifica después. Si cualquiera pudiera enrolar desde el chat, quien
controle una cuenta (SIM swap, sesión robada, teléfono prestado) registraría su
propia cara y pasaría el login. El enrolamiento por comando local exige acceso
al servidor, un canal de confianza distinto al de la cuenta de Telegram.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

import face

USUARIOS = Path(__file__).resolve().parent / "usuarios.json"


def _cargar() -> dict:
    return json.loads(USUARIOS.read_text(encoding="utf-8"))


def _guardar(data: dict) -> None:
    texto = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    with USUARIOS.open("w", encoding="utf-8", newline="") as f:
        f.write(texto.replace("\n", "\r\n"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Enrola el rostro de un usuario (local).")
    ap.add_argument("--id", required=True, help="id del usuario en usuarios.json")
    ap.add_argument(
        "--telegram-user-id",
        required=True,
        type=int,
        help="ID numérico de Telegram del dueño (se ve con /quiensoy)",
    )
    ap.add_argument("fotos", nargs="+", help="3 a 5 fotos con un solo rostro cada una")
    args = ap.parse_args()

    if not face.MIN_FOTOS <= len(args.fotos) <= face.MAX_FOTOS:
        print(
            f"Se requieren entre {face.MIN_FOTOS} y {face.MAX_FOTOS} fotos "
            f"(recibí {len(args.fotos)}).",
            file=sys.stderr,
        )
        return 2

    data = _cargar()
    usuarios = data.get("usuarios", [])
    usuario = next((u for u in usuarios if u.get("id") == args.id), None)
    if usuario is None:
        print(f"No existe el usuario '{args.id}' en usuarios.json.", file=sys.stderr)
        return 2
    otro = next(
        (
            u
            for u in usuarios
            if u.get("telegram_user_id") == args.telegram_user_id
            and u.get("id") != args.id
        ),
        None,
    )
    if otro is not None:
        print(
            f"Ese telegram_user_id ya pertenece a '{otro['id']}'.", file=sys.stderr
        )
        return 2

    # Todas las fotos deben ser válidas: si una falla no se guarda nada.
    embs = []
    for ruta in args.fotos:
        img = cv2.imread(ruta)
        if img is None:
            print(f"{ruta}: no pude leer la imagen.", file=sys.stderr)
            return 1
        try:
            embs.append(face.embedding(img))
        except face.RostroInvalido as exc:
            print(f"{ruta}: {exc} No se guardó nada.", file=sys.stderr)
            return 1

    destino = face.guardar_embeddings(args.id, embs)
    usuario["telegram_user_id"] = args.telegram_user_id
    _guardar(data)
    print(
        f"OK: {len(embs)} embeddings de '{args.id}' guardados en {destino.name} "
        f"y vinculados a la cuenta de Telegram {args.telegram_user_id}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
