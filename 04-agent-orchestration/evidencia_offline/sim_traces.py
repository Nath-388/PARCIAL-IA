"""Genera trazas con el formato REAL de chatbot.print_debug, pero con un LLM de guion (SIMULADAS)."""

import asyncio
import hashlib
import json
import re
import sys
import unicodedata
from contextlib import AsyncExitStack, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

LAB = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(LAB))

from agent import Agent  # noqa: E402
from capabilities import rag_tool  # noqa: E402
from chatbot import print_debug  # noqa: E402
from config import load_rag_settings  # noqa: E402
from documents import chunk_documents, load_documents  # noqa: E402
from llm_client import LLMResponse, ToolCall  # noqa: E402
from mcp_client import load_mcp_tools, open_client  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402
from rag import CourseRAG  # noqa: E402
from tools import LOCAL_TOOLS  # noqa: E402
from vector_store import VectorStore  # noqa: E402


class LexicalEmbedder:
    DIM = 512

    def _vec(self, text):
        text = unicodedata.normalize("NFD", text.lower())
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        v = np.zeros(self.DIM, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", text):
            if len(w) > 2:
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1
        return v

    def embed_documents(self, t):
        return np.array([self._vec(x) for x in t], dtype=np.float32)

    def embed_query(self, t):
        return self._vec(t)


def resp(text="", calls=None):
    return LLMResponse(text, "guion", "tool_calls" if calls else "stop", 0, 0, calls or [])


def call(i, name, args):
    return ToolCall(f"call_{i}", name, json.dumps(args, ensure_ascii=False))


def last(m):
    return next(json.loads(x["content"]) for x in reversed(m) if x["role"] == "tool")


class Scripted:
    def __init__(self, s):
        self.s = list(s)
        self.settings = type("S", (), {"provider": "guion", "model": "guion"})()

    def chat(self, messages, *, tools=None, allow_tools=True, **_):
        n = self.s.pop(0) if len(self.s) > 1 else self.s[0]
        return n(messages) if callable(n) else n


async def yes(tool, args):
    return True


def show(title, answer):
    buf = StringIO()
    with redirect_stdout(buf):
        print_debug(answer)
        print(f"\nAsistente: {answer.text}\n")
    return f"### {title}\n{buf.getvalue()}"


async def main():
    s = load_rag_settings()
    chunks = chunk_documents(load_documents(s.data_dir), s.chunk_size, s.chunk_overlap)
    emb = LexicalEmbedder()
    store = VectorStore("lexical")
    store.add(chunks, emb.embed_documents([c.text for c in chunks]))
    out = []
    async with AsyncExitStack() as stack:
        client = await open_client(stack)
        action = [*LOCAL_TOOLS, *await load_mcp_tools(client)]

        def orch(llm, max_steps=5):
            return Orchestrator(llm, CourseRAG(llm, emb, store, 4, 0.2), action, yes, max_steps)

        # Prueba 2
        llm = Scripted([
            resp(calls=[call(1, "consultar_notas", {"codigo": "2026001"})]),
            lambda m: resp(calls=[call(2, "calcular_nota_final", {"parcial_1": 4.0, "parcial_2": 4.0, "laboratorios": last(m)["promedio_laboratorios"], "proyecto": 4.0})]),
            lambda m: resp(f"Con 4.0 en los dos parciales y en el proyecto, y un promedio de laboratorios de 4.25, quedas con {last(m)['nota_final']}; aprobarías."),
        ])
        a = await orch(llm).answer("Mi código es 2026001. Si saco 4.0 en los dos parciales y 4.0 en el proyecto, ¿con cuánto quedo?", [], "agente")
        out.append(show("Prueba 2 (SIMULADA) — encadenar tools", a))

        # Prueba 3
        llm = Scripted([
            resp(calls=[call(1, "buscar_documentos", {"consulta": "fecha primer parcial"})]),
            resp(calls=[call(2, "dias_hasta", {"fecha": "2026-10-20"})]),
            lambda m: resp(f"Faltan {last(m)['dias']} días: el primer parcial se trasladó al martes 20 de octubre [1]."),
        ])
        a = await orch(llm).answer("¿Cuántos días faltan para el primer parcial?", [], "agente")
        out.append(show("Prueba 3 (SIMULADA) — combinar capacidades", a))

        # Prueba 5 (MAX_STEPS = 1)
        llm = Scripted([
            resp(calls=[call(1, "consultar_notas", {"codigo": "2026001"})]),
            resp("Ya consulté tus notas (promedio de laboratorios 4.25), pero no alcancé a calcular la nota final: quedó pendiente."),
        ])
        a = await orch(llm, max_steps=1).answer("Mi código es 2026001. Si saco 4.0 en los dos parciales y 4.0 en el proyecto, ¿con cuánto quedo?", [], "agente")
        out.append(show("Prueba 5 (SIMULADA) — MAX_STEPS = 1", a))
    Path(sys.argv[2]).write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


asyncio.run(main())
