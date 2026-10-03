"""
FastAPI backend.

Request path:

    POST /query
        -> guardrails gate      (deterministic; blocks before any model call)
        -> LangGraph agent      (planner -> retriever -> grader -> responder)
        -> response + sources + reasoning trace

Conversation memory: `/query` loads a bounded context (rolling summary + recent
window) from `app/memory/store.py`, seeds the graph with it, and persists the
turn afterwards; summarising and retention run as background tasks after the
response is sent. The `/conversations` endpoints serve the same rows to the
frontend's history sidebar, paginated so it stays fast however much there is.

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
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Response  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.agents.graph import rag_agent  # noqa: E402
from app.config import settings  # noqa: E402
from app.guardrails import guard, initialize_rails  # noqa: E402
from app.guardrails import status as guardrails_status  # noqa: E402
from app.llm import router  # noqa: E402
from app.memory import ConversationForbidden, store  # noqa: E402
from app.memory.store import Turn, make_title, slim_sources  # noqa: E402
from app.services.retrieval import embedding  # noqa: E402
from app.services.retrieval.qdrant_service import collection_stats, list_sources  # noqa: E402

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Guardrails are built at startup so the first user request does not pay the
    # initialisation cost — and so a misconfiguration shows up in the boot log
    # rather than in someone's first query.
    report_config_problems("api")
    mode = initialize_rails()
    memory = store.initialize()
    logfire.info("API ready", guardrails=mode, memory=memory, **settings.summary())
    yield


app = FastAPI(
    title=settings.APP_NAME,
    description="Agentic RAG over peer-reviewed cattle and buffalo disease literature.",
    version="1.0.0",
    lifespan=lifespan,
)

# Browser clients (the Streamlit UI, the React frontend) run on different
# origins, so they need CORS. Defaults to "*"; set CORS_ORIGINS to your deployed
# frontend domain to stop other sites spending your free-tier quota.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

try:
    logfire.instrument_fastapi(app, capture_headers=False)
except Exception as exc:  # instrumentation is a nice-to-have, not a dependency
    logfire.warning("FastAPI instrumentation unavailable ({err})", err=str(exc))


class QueryRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=4000, description="The user's question")
    thread_id: str | None = Field(
        default=None, max_length=128, description="Conversation id — omit to start a new thread"
    )
    user_id: str | None = Field(
        default=None, max_length=128,
        description="Owner of the conversation. Required for it to appear in GET /conversations.",
    )
    source_filter: str | None = Field(default=None, description="Restrict retrieval to one document")
    include_sources: bool = Field(
        default=False,
        description="Eval only: return retrieved passages for farmer answers too (the UI never sets it)",
    )


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

    # ── consumer-facing fields ────────────────────────────────────────────────
    # Language the user wrote in; the answer is returned in the same one.
    language: str = "en"
    # "home_care" | "vet_soon" | "vet_now" | "info" — lets a UI badge urgency
    # without parsing the prose.
    care_level: str | None = None
    # Populated when the assistant needs more detail before it can advise. The
    # questions are also embedded in `answer`, so a plain chat UI needs no
    # special handling — this field is for clients that want to render chips.
    follow_up_questions: list[str] = []
    awaiting_answer: bool = False
    # Conversation title (set from the first message) and how memory was served:
    # {"backend", "persisted", "summary_used", "window_messages"}.
    title: str | None = None
    memory: dict = {}


class RenameRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)
    title: str = Field(..., min_length=1, max_length=200)


@app.get("/")
def root() -> dict:
    return {
        "service": settings.APP_NAME,
        "status": "live",
        "docs": "/docs",
        "endpoints": ["/health", "/query", "/conversations", "/sources", "/stats", "/graph"],
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
        "memory": store.status(),
        "config": settings.summary(),
        "hint": (
            None if healthy
            else "Collection is empty or unreachable. Run: python -m app.ingestion.processor --wipe"
        ),
    }

@app.head("/health")
def health_head():
    return Response(status_code=200)


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


def _remember(
    background: BackgroundTasks, thread_id: str, user_id: str | None, turn: Turn, backend: str
) -> bool:
    """Persist the turn now (the next message depends on it); summarise and sweep later."""
    persisted = store.save(thread_id, user_id, turn)
    background.add_task(store.maybe_summarize, thread_id)
    background.add_task(store.maybe_cleanup)
    return persisted and backend == "postgres"


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, background: BackgroundTasks) -> QueryResponse:
    started = time.time()
    question = request.q.strip()
    thread_id = request.thread_id or f"thread-{uuid.uuid4().hex[:12]}"
    user_id = (request.user_id or "").strip() or None

    with logfire.span("Query", question=question[:150], thread_id=thread_id):

        # ── memory: bounded context for this turn ─────────────────────────────
        try:
            memory, backend = store.load(thread_id, user_id)
        except ConversationForbidden:
            # Someone else's id — never read or append to it.
            raise HTTPException(status_code=403, detail="This conversation belongs to another user.")

        # ── gate 1: guardrails (no model call on the blocked path) ────────────
        rail = guard(question)
        if rail.fired:
            elapsed = int((time.time() - started) * 1000)
            logfire.info("Request blocked", category=rail.category, rule=rail.rule)
            answer = rail.response or ""
            # Shown in the history, never fed back to the model: an injection
            # attempt must not become part of the next turn's context.
            persisted = _remember(
                background, thread_id, user_id,
                Turn(
                    user_text=question, user_text_en=question, answer=answer, answer_en=answer,
                    in_memory=False, meta={"blocked": True, "block_reason": rail.category},
                    awaiting_clarification=memory.awaiting_clarification,
                    clarification_rounds=memory.clarification_rounds,
                ),
                backend,
            )
            return QueryResponse(
                question=question,
                answer=answer,
                thread_id=thread_id,
                status="Blocked by guardrails",
                blocked=True,
                block_reason=rail.category,
                thought_process=[
                    f"Guardrails fired: {rail.category} ({rail.tier} tier)",
                    "Retrieval: skipped",
                ],
                elapsed_ms=elapsed,
                title=make_title(question) if memory.is_new else None,
                memory={"backend": backend, "persisted": persisted},
            )

        # ── gate 2: the agent ─────────────────────────────────────────────────
        initial_state = {
            # Bounded window of earlier turns (English) + the new message.
            "messages": memory.window + [{"role": "user", "content": question}],
            "memory_summary": memory.summary,
            "original_query": question,
            "query_en": "",          # filled by translate_in
            "search_query": question,
            "documents": [],
            "plan": [],
            "refinements": 0,
            "follow_up_questions": [],
            "care_level": "",
            "status": "starting",
            # Carried over from the previous turn: how the clarifier knows this
            # message answers its questions.
            "awaiting_clarification": memory.awaiting_clarification,
            "clarification_rounds": memory.clarification_rounds,
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
            memory_window=len(memory.window),
            memory_summary=bool(memory.summary),
        )

        care_level = final.get("care_level") or None
        awaiting = bool(final.get("awaiting_clarification"))
        answer = final.get("final_answer", "")
        llm_meta = final.get("llm_meta", {})

        # Sources are returned for research answers, but suppressed for farmer-facing
        # advice: that answer carries no citation markers, so a sources list would be
        # unattached noise. The farmer eval sets include_sources to grade grounding.
        hide = final.get("intent") == "symptom" and not request.include_sources
        sources = [] if hide else final.get("documents", [])

        # Memory keeps the English side of the turn: the graph reasons in English,
        # and the answer is translated only on the way out.
        last = (final.get("messages") or [{}])[-1]
        answer_en = str(last.get("content", "")) if last.get("role") == "assistant" else answer
        persisted = _remember(
            background, thread_id, user_id,
            Turn(
                user_text=question,
                user_text_en=final.get("query_en") or question,
                answer=answer,
                answer_en=answer_en or answer,
                care_level=care_level,
                sources=slim_sources(sources),
                meta={
                    "intent": final.get("intent"),
                    "language": final.get("language", "en"),
                    "follow_up_questions": final.get("follow_up_questions", []),
                    "awaiting_answer": awaiting,
                    "model": llm_meta.get("model"),
                    "fallback_used": bool(llm_meta.get("fallback_used")),
                },
                awaiting_clarification=awaiting,
                clarification_rounds=final.get("clarification_rounds", memory.clarification_rounds),
                language=final.get("language", "en"),
            ),
            backend,
        )

        return QueryResponse(
            question=question,
            answer=answer,
            thread_id=thread_id,
            status=final.get("status", "done"),
            thought_process=final.get("plan", []),
            sources=sources,
            llm=llm_meta,
            elapsed_ms=elapsed,
            language=final.get("language", "en"),
            care_level=care_level,
            follow_up_questions=final.get("follow_up_questions", []),
            awaiting_answer=awaiting,
            title=make_title(question) if memory.is_new else None,
            memory={
                "backend": backend,
                "persisted": persisted,
                "summary_used": bool(memory.summary),
                "window_messages": len(memory.window),
            },
        )


# ── chat history (frontend sidebar) ────────────────────────────────────────────
# Two light endpoints instead of one heavy one: the sidebar list carries no
# messages, and a conversation's messages load only when it is opened, newest
# page first. Both are keyset-paginated on indexed columns, so a page costs the
# same whether the user has 5 conversations or 500.


def _history_unavailable(exc: Exception) -> HTTPException:
    logfire.warning("Chat history unavailable ({err})", err=str(exc)[:300])
    return HTTPException(status_code=503, detail="Chat history is unavailable right now.")


@app.get("/conversations")
def list_conversations(
    user_id: str = Query(..., min_length=1, max_length=128),
    cursor: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=30, ge=1, le=100),
) -> dict:
    try:
        items, next_cursor = store.list_conversations(user_id, cursor, limit)
    except Exception as exc:
        raise _history_unavailable(exc)
    return {"conversations": items, "next_cursor": next_cursor}


@app.get("/conversations/{conversation_id}/messages")
def conversation_messages(
    conversation_id: str,
    user_id: str = Query(..., min_length=1, max_length=128),
    before: int | None = Query(default=None, ge=1),
    limit: int = Query(default=30, ge=1, le=100),
) -> dict:
    try:
        result = store.get_messages(conversation_id, user_id, before, limit)
    except Exception as exc:
        raise _history_unavailable(exc)
    if result is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    items, next_before = result
    return {"messages": items, "next_before": next_before}


@app.patch("/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, request: RenameRequest) -> dict:
    try:
        ok = store.rename(conversation_id, request.user_id, request.title)
    except Exception as exc:
        raise _history_unavailable(exc)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"id": conversation_id, "title": make_title(request.title)}


@app.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str, user_id: str = Query(..., min_length=1, max_length=128)
) -> dict:
    try:
        ok = store.delete(conversation_id, user_id)
    except Exception as exc:
        raise _history_unavailable(exc)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"deleted": conversation_id}
