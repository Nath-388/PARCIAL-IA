"""Evaluación: el mismo conjunto de preguntas en los dos modos, medido con datos.

Para cada pregunta registra la ruta o las tools usadas, si la respuesta contiene el
dato esperado, cuántas llamadas al LLM hizo y cuántos tokens consumió. Las acciones
con efectos se rechazan automáticamente: una evaluación no debe modificar nada.

Uso (consume tokens: con planes gratuitos usa --pausa para no superar el límite por minuto):
    uv run python labs/04-agent-orchestration/evaluate.py [--modo agente|router] [--pausa SEGUNDOS]
"""

import asyncio
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import date
from typing import Any

from llm_client import LLMError
from orchestrator import Mode, Orchestrator, create_orchestrator
from tools import Tool

PRIMER_PARCIAL = date(2026, 10, 20)  # fecha vigente según anuncios.md


@dataclass
class Case:
    question: str
    route: str | None  # ruta esperada en modo router; None = necesita más de una capacidad
    expected: str | None  # texto que debe aparecer en la respuesta; None = no se verifica


CASES = [
    Case("hola", "directa", None),
    Case("¿Qué es el mecanismo de atención?", "directa", None),
    Case("¿Cuál es la nota mínima para aprobar el curso?", "documentos", "3.0"),
    Case("¿Qué pasa si entrego tarde un laboratorio?", "documentos", "48"),
    Case("¿Cuándo es el primer parcial?", "documentos", "20 de octubre"),
    Case("¿Cuáles son mis notas? Mi código es 2026001", "acciones", "4.25"),
    Case(
        "Mi código es 2026001. Si saco 4.0 en los dos parciales y 4.0 en el proyecto, ¿con cuánto quedo?",
        "acciones",
        "4.09",  # 0.2·4 + 0.2·4 + 0.35·4.25 + 0.25·4 = 4.0875
    ),
    Case("¿Cuántos días faltan para el primer parcial?", None, str((PRIMER_PARCIAL - date.today()).days)),
]


async def deny_actions(tool: Tool, arguments: dict[str, Any]) -> bool:
    return False


def check(answer_text: str, expected: str | None) -> str:
    if expected is None:
        return "—"
    return "sí" if expected.lower() in answer_text.lower() else "NO"


async def evaluate(orchestrator: Orchestrator, mode: Mode, pause: float) -> None:
    print(f"\n=== Modo {mode} ===\n")
    print(f"{'#':>2}  {'ruta / tools':<48} {'ok':<3} {'llam.':>5} {'tok. ent.':>9} {'tok. sal.':>9}")
    totals = {"ok": 0, "checked": 0, "calls": 0, "tokens_in": 0, "tokens_out": 0, "route_ok": 0}
    start = time.perf_counter()

    for i, case in enumerate(CASES, start=1):
        try:
            answer = await orchestrator.answer(case.question, [], mode)
        except LLMError as exc:
            print(f"{i:>2}  [error] {str(exc)[:90]}")
            continue

        if answer.routing:
            label = answer.routing.decision.ruta
            if case.route is not None and label == case.route:
                totals["route_ok"] += 1
            if answer.tools_used():
                label += ": " + ", ".join(answer.tools_used())
        else:
            label = ", ".join(answer.tools_used()) or "(sin tools)"
        if answer.stopped_by_limit:
            label += " [límite]"

        ok = check(answer.text, case.expected)
        tokens_in = sum(response.prompt_tokens for response in answer.responses)
        tokens_out = sum(response.completion_tokens for response in answer.responses)
        totals["checked"] += ok != "—"
        totals["ok"] += ok == "sí"
        totals["calls"] += len(answer.responses)
        totals["tokens_in"] += tokens_in
        totals["tokens_out"] += tokens_out
        print(f"{i:>2}  {label[:48]:<48} {ok:<3} {len(answer.responses):>5} {tokens_in:>9} {tokens_out:>9}")
        print(f"    P: {case.question}")
        print(f"    R: {answer.text.replace(chr(10), ' ')[:110]}")
        if pause:
            await asyncio.sleep(pause)

    elapsed = time.perf_counter() - start
    print(f"\nRespuestas con el dato esperado: {totals['ok']}/{totals['checked']}")
    if mode == "router":
        routable = sum(case.route is not None for case in CASES)
        print(f"Rutas correctas: {totals['route_ok']}/{routable}")
    print(
        f"Llamadas al LLM: {totals['calls']} · tokens entrada={totals['tokens_in']} "
        f"salida={totals['tokens_out']} · tiempo={elapsed:.0f} s"
    )


async def main() -> None:
    modes: list[Mode] = ["router", "agente"]
    pause = 0.0
    if "--modo" in sys.argv:
        mode = sys.argv[sys.argv.index("--modo") + 1]
        if mode not in ("router", "agente"):
            print("--modo debe ser router o agente")
            return
        modes = [mode]
    if "--pausa" in sys.argv:
        pause = float(sys.argv[sys.argv.index("--pausa") + 1])

    async with AsyncExitStack() as stack:
        try:
            orchestrator = await create_orchestrator(stack, confirm=deny_actions)
        except (ValueError, FileNotFoundError) as exc:
            print(f"[configuración] {exc}")
            return
        for mode in modes:
            await evaluate(orchestrator, mode, pause)


if __name__ == "__main__":
    asyncio.run(main())
