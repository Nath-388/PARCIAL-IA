"""Orquestador: compone LLM, RAG, tools locales y MCP en un solo asistente.

Dos estrategias para decidir qué capacidad usar:

  modo "router" (flujo fijo, decidido por el software después de clasificar):
      pregunta → router → "directa"   → LLM sin tools               (como el Lab 01)
                        → "documentos" → CourseRAG.answer           (como el Lab 02)
                        → "acciones"   → agente con tools y MCP     (como el Lab 03, con ciclo)

  modo "agente" (flujo dinámico, decidido por el modelo en cada paso):
      pregunta → agente con TODAS las capacidades como tools (incluida buscar_documentos)

El orquestador no sabe nada de la consola: recibe una pregunta y devuelve la respuesta
con su traza (ruta, pasos, fuentes, llamadas al LLM).
"""

from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Literal

from agent import MAX_STEPS, Agent, AgentAnswer, AgentStep, ConfirmFn
from capabilities import rag_tool
from config import load_rag_settings, load_settings
from embeddings import Embedder
from llm_client import LLMClient, LLMResponse, Message
from mcp_client import load_mcp_tools, open_client
from prompts import build_direct_messages
from rag import CourseRAG
from router import RouterResult, route
from tools import LOCAL_TOOLS, Tool
from vector_store import VectorStore

Mode = Literal["router", "agente"]


@dataclass
class OrchestratorAnswer:
    text: str
    mode: Mode
    routing: RouterResult | None = None  # solo en modo router
    steps: list[AgentStep] = field(default_factory=list)  # traza del agente, si se usó
    sources: list[str] = field(default_factory=list)  # fuentes calculadas por el programa (ruta documentos)
    responses: list[LLMResponse] = field(default_factory=list)  # TODAS las llamadas al LLM, en orden
    stopped_by_limit: bool = False

    def tools_used(self) -> list[str]:
        return [run.call.name for step in self.steps for run in step.tool_runs]


class Orchestrator:
    def __init__(
        self,
        llm: LLMClient,
        rag: CourseRAG,
        action_tools: list[Tool],
        confirm: ConfirmFn,
        max_steps: int = MAX_STEPS,
    ):
        self.llm = llm
        self.rag = rag
        # Modo agente: un solo agente con todas las capacidades.
        self.agent = Agent(llm, [rag_tool(rag), *action_tools], confirm, max_steps)
        # Modo router, ruta "acciones": un agente solo con las tools de datos y acciones.
        self.actions_agent = Agent(llm, action_tools, confirm, max_steps)

    async def answer(self, question: str, history: list[Message], mode: Mode) -> OrchestratorAnswer:
        if mode == "agente":
            return self._from_agent(await self.agent.run(question, history), mode)

        routing = route(self.llm, question)
        ruta = routing.decision.ruta
        # El software (no el modelo) ejecuta el camino fijo que salió de la clasificación.
        if ruta == "directa":
            response = self.llm.chat(build_direct_messages(history, question))
            answer = OrchestratorAnswer(response.text, mode, responses=[response])
        elif ruta == "documentos":
            rag_answer = self.rag.answer(question, history)  # el pipeline completo del Lab 02
            sources = sorted({result.chunk.source for result in rag_answer.sources})
            answer = OrchestratorAnswer(
                rag_answer.text, mode, sources=sources, responses=[rag_answer.response]
            )
        else:  # "acciones": agente sin buscar_documentos
            answer = self._from_agent(await self.actions_agent.run(question, history), mode)

        answer.routing = routing
        # La clasificación también fue una llamada al LLM: se cuenta al inicio de la traza.
        answer.responses.insert(0, routing.response)
        return answer

    @staticmethod
    def _from_agent(result: AgentAnswer, mode: Mode) -> OrchestratorAnswer:
        return OrchestratorAnswer(
            result.text,
            mode,
            steps=result.steps,
            responses=[step.response for step in result.steps],
            stopped_by_limit=result.stopped_by_limit,
        )


async def create_orchestrator(stack: AsyncExitStack, confirm: ConfirmFn) -> Orchestrator:
    """Arma el sistema completo: configuración, LLM, índice, RAG, tools locales y servidor MCP.

    Lanza ValueError (configuración), FileNotFoundError (falta el índice) o un error de
    conexión con el servidor MCP. El servidor queda abierto hasta que se cierre `stack`.
    """
    settings = load_settings()
    rag_settings = load_rag_settings()
    store = VectorStore.load(rag_settings.index_dir)

    llm = LLMClient(settings)
    rag = CourseRAG(
        llm=llm,
        # La pregunta se convierte con el mismo modelo que indexó los documentos.
        embedder=Embedder(store.embedding_model),
        store=store,
        top_k=rag_settings.top_k,
        min_score=rag_settings.min_score,
    )
    mcp_client = await open_client(stack)
    action_tools = [*LOCAL_TOOLS, *await load_mcp_tools(mcp_client)]
    return Orchestrator(llm, rag, action_tools, confirm)
