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

    original_query: str      # exactly what the user typed
    search_query: str        # planner's rewrite — standalone, history-resolved
    intent: str              # "conversational" | "research"

    documents: list[dict]    # RetrievedChunk.to_dict()
    context_quality: str     # "sufficient" | "weak" | "empty"
    refinements: int         # self-correction loops used so far

    final_answer: str
    status: str
    llm_meta: dict           # which target answered, cache hit, latency
