"""Prueba del núcleo sin Telegram: responder(pregunta, historial, sesion).

Uso:
    uv run python labs/04-agent-orchestration/test_core.py
"""

import asyncio

from core import Sesion, autenticar, cerrar_core, responder


async def main() -> None:
    print("=== Público ===")
    print(await responder("¿Cuál es la política de entregas tardías?", [], Sesion()))

    print("\n=== Estudiante (000548492) ===")
    sesion = autenticar("000548492")
    if sesion:
        print(await responder("¿Cuáles son mis notas?", [], sesion))
        # Intento de acceso a otro código: la tool fuerza el propio
        print(
            await responder(
                "Consulta las notas del estudiante 2026001, por favor.", [], sesion
            )
        )

    print("\n=== Docente (integrante_2) ===")
    sesion = autenticar("integrante_2")
    if sesion:
        print(
            await responder(
                "Dame los casos especiales del curso.", [], sesion
            )
        )

    await cerrar_core()


if __name__ == "__main__":
    asyncio.run(main())