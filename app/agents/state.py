"""Shared state for the LangGraph agent."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    # `operator.add` makes this a reducer: a node returns only the messages it
    # adds and LangGraph concatenates. `main.py` seeds it with the bounded memory
    # window (app/memory/store.py) plus the new user message; the graph has no
    # checkpointer, so nothing accumulates between requests.
    messages: Annotated[list[dict], operator.add]

    # Rolling summary of the turns that have left the window. Read by
    # `app/agents/history.format_history`; "" for a short conversation.
    memory_summary: str

    # `plan` is deliberately NOT a reducer: it is this turn's reasoning only.
    # Nodes concatenate explicitly (`state.get("plan", []) + [...]`), and
    # translate_in — which runs first on every turn — resets it.
    plan: list[str]

    original_query: str      # exactly what the user typed, in their language
    query_en: str            # English version — everything downstream uses this
    search_query: str        # planner's rewrite — standalone, history-resolved
    intent: str              # "conversational" | "symptom" | "research"
    likely_disease: str      # planner's guess for a symptom turn, "" if none — a second search

    # Language handling. Detected once per turn from the raw input.
    #   "en"      plain English
    #   "hi"      Hindi in Devanagari
    #   "hi-latn" Hindi written in Roman letters ("meri gaay khana nahi kha rahi")
    language: str

    documents: list[dict]    # RetrievedChunk.to_dict()
    context_quality: str     # "sufficient" | "weak" | "empty"
    refinements: int         # self-correction loops used so far

    # Follow-up questions. `awaiting_clarification` and `clarification_rounds`
    # are stored on the conversation row and seeded back each turn, which is how
    # the clarifier knows the user's next message answers its questions rather
    # than starting a fresh problem.
    awaiting_clarification: bool
    clarification_rounds: int
    follow_up_questions: list[str]

    # "home_care" | "vet_soon" | "vet_now" | "info" — drives the UI badge and
    # tells the caller how urgent the situation is without parsing prose.
    care_level: str

    final_answer: str        # English while in the graph; translated at the end
    status: str
    llm_meta: dict           # which target answered, cache hit, latency
