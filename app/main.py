"""
FastAPI backend.

Request path:

    POST /query
        -> guardrails gate      (deterministic; blocks before any model call)
        -> LangGraph agent      (planner -> retriever -> grader -> responder)
        -> response + sources + reasoning trace

Everything the agent decided is returned to the caller, not just the answer:
the plan, which passages were used, which LLM target answered, whether a
fallback or cache was hit. A RAG system you cannot inspect is a RAG system you
cannot debug — and it is what the eval suite reads to score tool selection.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

# Observability first: modules imported below emit spans at import time.
from app.observability import configure_observability, report_config_problems

configure_observability("bovine-rag-api")

import logfire  # noqa: E402
from fastapi import FastAPI, HTTPException, Response  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.agents.graph import rag_agent  # noqa: E402
from app.config import settings  # noqa: E402
from app.guardrails import guard, initialize_rails  # noqa: E402
from app.guardrails import status as guardrails_status  # noqa: E402
from app.llm import router  # noqa: E402
from app.services.retrieval import embedding  # noqa: E402
from app.services.retrieval.qdrant_service import collection_stats, list_sources  # noqa: E402

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Guardrails are built at startup so the first user request does not pay the
    # initialisation cost — and so a misconfiguration shows up in the boot log
    # rather than in someone's first query.
    report_config_problems("api")
    mode = initialize_rails()
    logfire.info("API ready", guardrails=mode, **settings.summary())
    yield


app = FastAPI(
    title=settings.APP_NAME,
    description="Agentic RAG over peer-reviewed cattle and buffalo disease literature.",
    version="1.0.0",
    lifespan=lifespan,
)

# Streamlit runs on a different origin; the UI is the only intended browser client.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

try:
    logfire.instrument_fastapi(app, capture_headers=False)
except Exception as exc:  # instrumentation is a nice-to-have, not a dependency
    logfire.warning("FastAPI instrumentation unavailable ({err})", err=str(exc))


class QueryRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=4000, description="The user's question")
    thread_id: str | None = Field(default=None, description="Conversation id — omit to start a new thread")
    source_filter: str | None = Field(default=None, description="Restrict retrieval to one document")


class QueryResponse(BaseModel):
    question: str
    answer: str
    thread_id: str
    status: str
    blocked: bool = False
    block_reason: str | None = None
    thought_process: list[str] = []
    sources: list[dict] = []
    llm: dict = {}
    elapsed_ms: int = 0


@app.get("/")
def root() -> dict:
    return {
        "service": settings.APP_NAME,
        "status": "live",
        "docs": "/docs",
        "endpoints": ["/health", "/query", "/sources", "/stats", "/graph"],
    }


@app.get("/health")
def health() -> dict:
    """Deep health check — reports each dependency separately so failures are locatable."""
    qdrant = collection_stats()
    healthy = "error" not in qdrant and (qdrant.get("points") or 0) > 0

    return {
        "status": "healthy" if healthy else "degraded",
        "qdrant": qdrant,
        "embeddings": embedding.describe(),
        "guardrails": guardrails_status(),
        "llm": router.stats(),
        "config": settings.summary(),
        "hint": (
            None if healthy
            else "Collection is empty or unreachable. Run: python -m app.ingestion.processor --wipe"
        ),
    }


@app.get("/sources")
def sources() -> dict:
    docs = list_sources()
    return {"count": len(docs), "documents": docs}


@app.get("/stats")
def stats() -> dict:
    return {
        "qdrant": collection_stats(),
        "embeddings": embedding.describe(),
        "llm_gateway": router.stats(),
    }


@app.get("/graph")
def graph_image():
    """PNG of the compiled agent graph. Handy for the report; needs network access."""
    try:
        return Response(content=rag_agent.get_graph().draw_mermaid_png(), media_type="image/png")
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not render the graph ({exc}). Try GET /graph/mermaid instead.",
        )


@app.get("/graph/mermaid")
def graph_mermaid() -> dict:
    """Mermaid source for the agent graph — no network round-trip required."""
    try:
        return {"mermaid": rag_agent.get_graph().draw_mermaid()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    started = time.time()
    question = request.q.strip()
    thread_id = request.thread_id or f"thread-{uuid.uuid4().hex[:12]}"

    with logfire.span("Query", question=question[:150], thread_id=thread_id):

        # ── gate 1: guardrails (no model call on the blocked path) ────────────
        rail = guard(question)
        if rail.fired:
            elapsed = int((time.time() - started) * 1000)
            logfire.info("Request blocked", category=rail.category, rule=rail.rule)
            return QueryResponse(
                question=question,
                answer=rail.response or "",
                thread_id=thread_id,
                status="Blocked by guardrails",
                blocked=True,
                block_reason=rail.category,
                thought_process=[
                    f"Guardrails fired: {rail.category} ({rail.tier} tier)",
                    "Retrieval: skipped",
                ],
                elapsed_ms=elapsed,
            )

        # ── gate 2: the agent ─────────────────────────────────────────────────
        initial_state = {
            "messages": [{"role": "user", "content": question}],
            "original_query": question,
            "search_query": question,
            "documents": [],
            "plan": [],
            "refinements": 0,
            "status": "starting",
        }

        try:
            # Invoked synchronously: LangGraph's async path runs nodes on a
            # different context, which detaches the Logfire span tree and makes
            # the trace unreadable.
            final = rag_agent.invoke(
                initial_state,
                config={"configurable": {"thread_id": thread_id}},
            )
        except Exception as exc:
            logfire.exception("Agent execution failed: {err}", err=str(exc)[:400])
            return QueryResponse(
                question=question,
                answer=(
                    "Something went wrong while processing that. The details are in the server "
                    "logs — check /health to see which dependency is unhealthy."
                ),
                thread_id=thread_id,
                status="error",
                thought_process=[f"Error: {str(exc)[:200]}"],
                elapsed_ms=int((time.time() - started) * 1000),
            )

        elapsed = int((time.time() - started) * 1000)
        logfire.info(
            "Query complete",
            elapsed_ms=elapsed,
            intent=final.get("intent"),
            sources=len(final.get("documents", [])),
            refinements=final.get("refinements", 0),
        )

        return QueryResponse(
            question=question,
            answer=final.get("final_answer", ""),
            thread_id=thread_id,
            status=final.get("status", "done"),
            thought_process=final.get("plan", []),
            sources=final.get("documents", []),
            llm=final.get("llm_meta", {}),
            elapsed_ms=elapsed,
        )
