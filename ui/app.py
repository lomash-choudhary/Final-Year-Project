"""
Streamlit chat interface.

Deliberately shows its work: the agent's reasoning steps, the exact passages
retrieved with source and page, and which LLM target produced the answer. For a
final-year demo, "here is the paragraph on page 4 of this paper that my answer
came from" is the part that actually demonstrates the system works.

    streamlit run ui/app.py
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

import requests
import streamlit as st

# Make `app` importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402

BACKEND = os.getenv("BACKEND_URL", settings.BACKEND_URL).rstrip("/")
REQUEST_TIMEOUT = 180  # the self-correction loop plus a cold reranker can be slow

st.set_page_config(page_title="Bovine Disease Research Assistant", page_icon="🐄", layout="wide")

AI_AVATAR, USER_AVATAR = "🐄", "👤"


# ── session ────────────────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:12]}"
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_data(ttl=30)
def fetch_health() -> dict:
    try:
        return requests.get(f"{BACKEND}/health", timeout=15).json()
    except Exception as exc:
        return {"status": "unreachable", "error": str(exc)}


# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🐄 Agent Console")

    health = fetch_health()
    state = health.get("status", "unknown")

    if state == "healthy":
        st.success("Backend healthy")
    elif state == "degraded":
        st.warning("Backend degraded")
        if health.get("hint"):
            st.caption(health["hint"])
    else:
        st.error("Backend unreachable")
        st.caption(f"Tried {BACKEND} — is uvicorn running?")

    if state in ("healthy", "degraded"):
        qdrant = health.get("qdrant", {})
        emb = health.get("embeddings", {})

        st.markdown("**Knowledge base**")
        col_a, col_b = st.columns(2)
        col_a.metric("Chunks", qdrant.get("points", 0))
        col_b.metric("Vector dim", emb.get("dim", "?"))

        st.markdown("**Pipeline**")
        st.caption(f"Embeddings · `{emb.get('provider', '?')}/{emb.get('model', '?')}`")
        st.caption(f"Guardrails · `{health.get('guardrails', {}).get('mode', '?')}`")
        st.caption(f"Self-correction · `{health.get('config', {}).get('self_correction', '?')}`")

        cache = emb.get("cache", {})
        if cache.get("enabled"):
            st.caption(f"Embedding cache · {cache.get('rows', 0)} vectors")

    st.divider()
    st.caption(f"Thread `{st.session_state.thread_id[-8:]}`")

    if st.button("Clear conversation", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:12]}"
        st.rerun()

    with st.expander("Indexed papers"):
        try:
            docs = requests.get(f"{BACKEND}/sources", timeout=15).json().get("documents", [])
            if docs:
                for doc in docs:
                    st.caption(f"• {doc}")
            else:
                st.caption("Nothing indexed yet.")
        except Exception:
            st.caption("Unavailable.")


# ── main ───────────────────────────────────────────────────────────────────────
st.title("Bovine Disease Research Assistant")
st.caption(
    "Agentic RAG over peer-reviewed cattle and buffalo disease literature — "
    "every answer is grounded in an indexed paper and cited by page."
)

if not st.session_state.messages:
    st.markdown("**Try one of these:**")
    cols = st.columns(3)
    examples = [
        "What is the reported prevalence of theileriosis in cattle and buffaloes in India?",
        "Which season shows the highest incidence of haemoprotozoal disease in crossbred cattle?",
        "What are the most common foot disorders reported in cattle in Bihar?",
    ]
    for col, example in zip(cols, examples):
        if col.button(example, use_container_width=True):
            st.session_state.pending = example
            st.rerun()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources — {len(sources)} passage(s) used"):
        for i, source in enumerate(sources, start=1):
            st.markdown(
                f"**[{i}] {source.get('source', 'unknown')}** · page {source.get('page_label', 'n/a')} "
                f"· relevance `{source.get('score', 0):.3f}`"
            )
            st.text(source.get("content", "")[:1500])
            if i < len(sources):
                st.divider()


# Replay history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AI_AVATAR if message["role"] == "assistant" else USER_AVATAR):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

prompt = st.chat_input("Ask about cattle or buffalo disease research...")
if not prompt and "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=AI_AVATAR):
        with st.status("Agent working...", expanded=True) as status:
            try:
                started = time.time()
                response = requests.post(
                    f"{BACKEND}/query",
                    json={"q": prompt, "thread_id": st.session_state.thread_id},
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.Timeout:
                status.update(label="Timed out", state="error")
                st.error(f"The backend did not respond within {REQUEST_TIMEOUT}s.")
                st.stop()
            except Exception as exc:
                status.update(label="Connection failed", state="error")
                st.error(f"Could not reach the backend at {BACKEND} — {exc}")
                st.stop()

            for step in data.get("thought_process", []):
                st.write(f"• {step}")

            label = "Blocked by guardrails" if data.get("blocked") else f"Done in {time.time() - started:.1f}s"
            status.update(label=label, state="complete", expanded=False)

        answer = data.get("answer", "(no answer returned)")
        st.markdown(answer)

        sources = data.get("sources", [])
        render_sources(sources)

        llm = data.get("llm", {})
        if llm.get("model"):
            badge = "cache hit" if llm.get("cached") else f"{llm.get('target')} · {llm.get('model')}"
            warn = " ⚠︎ fallback" if llm.get("fallback_used") else ""
            st.caption(f"{badge}{warn} · {data.get('elapsed_ms', 0)} ms")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
