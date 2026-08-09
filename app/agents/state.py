"""Shared state for the LangGraph agent."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    # `operator.add` makes this a reducer: a node returns only the messages it
    # adds and LangGraph concatenates. Without it each node would overwrite the
    # whole list and conversation memory would be one turn deep.
    messages: Annotated[list[dict], operator.add]

    # `plan` is deliberately NOT a reducer. MemorySaver persists state per
    # thread_id, so an accumulating plan would carry every previous turn's
    # reasoning into the next one. Nodes concatenate explicitly instead, and the
    # planner — which runs first on every turn — resets it.
    plan: list[str]

    original_query: str      # exactly what the user typed, in their language
    query_en: str            # English version — everything downstream uses this
    search_query: str        # planner's rewrite — standalone, history-resolved
    intent: str              # "conversational" | "symptom" | "research"

    # Language handling. Detected once per turn from the raw input.
    #   "en"      plain English
    #   "hi"      Hindi in Devanagari
    #   "hi-latn" Hindi written in Roman letters ("meri gaay khana nahi kha rahi")
    language: str

    documents: list[dict]    # RetrievedChunk.to_dict()
    context_quality: str     # "sufficient" | "weak" | "empty"
    refinements: int         # self-correction loops used so far

    # Follow-up questions. `awaiting_clarification` persists across turns via the
    # checkpointer, which is how the clarifier knows the user's next message is
    # an answer to its questions rather than a fresh problem.
    awaiting_clarification: bool
    clarification_rounds: int
    follow_up_questions: list[str]

    # "home_care" | "vet_soon" | "vet_now" | "info" — drives the UI badge and
    # tells the caller how urgent the situation is without parsing prose.
    care_level: str

    final_answer: str        # English while in the graph; translated at the end
    status: str
    llm_meta: dict           # which target answered, cache hit, latency
