from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np

import face

EXT = {".jpg", ".jpeg", ".png", ".webp"}
MIN_GENUINOS = 10
MIN_IMPOSTORES = 20
UMBRALES_POR_DEFECTO = [0.30, face.UMBRAL_REFERENCIA, 0.45]


def cargar_embeddings(directorio: Path) -> dict[str, list[np.ndarray]]:
    personas: dict[str, list[np.ndarray]] = {}
    for carpeta in sorted(p for p in directorio.iterdir() if p.is_dir()):
        embs = []
        for foto in sorted(carpeta.iterdir()):
            if foto.suffix.lower() not in EXT:
                continue
            img = cv2.imread(str(foto))
            if img is None:
                print(f"  ! {foto}: no se pudo leer, se omite")
                continue
            try:
                embs.append(face.embedding(img))
            except face.RostroInvalido as exc:
                print(f"  ! {foto}: {exc} Se omite")
        if embs:
            personas[carpeta.name] = embs
    return personas


def construir_pares(personas: dict[str, list[np.ndarray]]):
    genuinos, impostores = [], []
    for embs in personas.values():
        for a, b in itertools.combinations(embs, 2):
            genuinos.append(face.similitud(a, b))
    for (_, ea), (_, eb) in itertools.combinations(personas.items(), 2):
        for a, b in itertools.product(ea, eb):
            impostores.append(face.similitud(a, b))
    return np.array(genuinos), np.array(impostores)


def tasas(gen: np.ndarray, imp: np.ndarray, t: float) -> tuple[float, float]:
    far = float(np.mean(imp >= t))
    frr = float(np.mean(gen < t))
    return far, frr


def main() -> int:
    ap = argparse.ArgumentParser(description="Tabla FAR/FRR para elegir el umbral.")
    ap.add_argument("--dir", default="calibracion", type=Path)
    ap.add_argument("--umbrales", nargs="+", type=float, default=UMBRALES_POR_DEFECTO)
    ap.add_argument("--salida", default="calibracion_resultados.md", type=Path)
    args = ap.parse_args()

    if not args.dir.is_dir():
        print(f"No existe la carpeta {args.dir}/ (ver el docstring).", file=sys.stderr)
        return 2
    if len(args.umbrales) < 3:
        print("Evalúa al menos tres umbrales.", file=sys.stderr)
        return 2

    print("Calculando embeddings...")
    personas = cargar_embeddings(args.dir)
    if len(personas) < 2:
        print("Se necesitan al menos 2 personas con fotos válidas.", file=sys.stderr)
        return 2

    gen, imp = construir_pares(personas)
    if len(gen) < MIN_GENUINOS or len(imp) < MIN_IMPOSTORES:
        print(
            f"Pares insuficientes: {len(gen)} genuinos (mín. {MIN_GENUINOS}), "
            f"{len(imp)} impostores (mín. {MIN_IMPOSTORES}). Agrega más fotos.",
            file=sys.stderr,
        )
        return 2

    lineas = [
        "# Calibración del umbral (coseno, SFace)",
        "",
        f"- Personas: {len(personas)} "
        + "(" + ", ".join(f"{n}: {len(e)} fotos" for n, e in personas.items()) + ")",
        f"- Pares genuinos: {len(gen)} | pares impostores: {len(imp)}",
        f"- Puntaje genuino: min {gen.min():.3f}, media {gen.mean():.3f}, "
        f"max {gen.max():.3f}",
        f"- Puntaje impostor: min {imp.min():.3f}, media {imp.mean():.3f}, "
        f"max {imp.max():.3f}",
        "",
        "| Umbral | FAR | FRR | Impostores aceptados | Genuinos rechazados |",
        "| ------ | --- | --- | -------------------- | ------------------- |",
    ]
    for t in sorted(args.umbrales):
        far, frr = tasas(gen, imp, t)
        lineas.append(
            f"| {t:.3f} | {far:.1%} | {frr:.1%} | "
            f"{int(round(far * len(imp)))}/{len(imp)} | "
            f"{int(round(frr * len(gen)))}/{len(gen)} |"
        )

    # Ayuda para decidir (la elección y su justificación son del grupo).
    barrido = np.round(np.arange(0.0, 1.0001, 0.005), 3)
    sin_falsas = [t for t in barrido if tasas(gen, imp, float(t))[0] == 0.0]
    if sin_falsas:
        t0 = float(sin_falsas[0])
        lineas += [
            "",
            f"Menor umbral del barrido con FAR = 0% en estos datos: {t0:.3f} "
            f"(FRR = {tasas(gen, imp, t0)[1]:.1%}).",
        ]
    eer_t = min(barrido, key=lambda t: abs(np.subtract(*tasas(gen, imp, float(t)))))
    far_e, frr_e = tasas(gen, imp, float(eer_t))
    lineas.append(
        f"Punto de cruce FAR≈FRR en el barrido: umbral {float(eer_t):.3f} "
        f"(FAR {far_e:.1%}, FRR {frr_e:.1%})."
    )
    lineas += [
        "",
        "Elección del umbral: _(completar: valor elegido y por qué; en un login "
        "suele priorizarse un FAR bajo aceptando algo más de FRR, porque el "
        "usuario legítimo puede reintentar)_.",
    ]

    texto = "\n".join(lineas) + "\n"
    print("\n" + texto)
    args.salida.write_text(texto, encoding="utf-8")
    print(f"Guardado en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
