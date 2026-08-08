"""
Agent graph.

    planner ──conversational──────────────────────────► responder ──► END
       │
       └──research──► retriever ──► grader ──sufficient──► responder
                          ▲            │
                          └──weak──────┘        (bounded by MAX_REFINEMENTS)

The grader→retriever edge is the cycle, and the cycle is why this is a LangGraph
state machine and not a chain. Set ENABLE_SELF_CORRECTION=false to compile the
linear version instead — useful for measuring exactly what the loop buys you in
the eval suite.

MemorySaver keys conversation state by `thread_id`, so multi-turn follow-ups
("and in buffalo?") resolve against real history rather than a fresh context.
"""

from __future__ import annotations

import logfire
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agents.nodes.grader import grade_node
from app.agents.nodes.planner import planner_node
from app.agents.nodes.responder import generate_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.state import AgentState
from app.config import settings


def _route_after_planner(state: AgentState) -> str:
    return "responder" if state.get("intent") == "conversational" else "retriever"


def _route_after_grader(state: AgentState) -> str:
    """Loop back only when the context is weak AND budget remains."""
    if state.get("context_quality") != "weak":
        return "responder"
    if state.get("refinements", 0) > settings.MAX_REFINEMENTS:
        logfire.warning("Refinement budget exceeded — answering with current context")
        return "responder"
    return "retriever"


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("responder", generate_node)

    workflow.set_entry_point("planner")
    workflow.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"retriever": "retriever", "responder": "responder"},
    )

    if settings.ENABLE_SELF_CORRECTION:
        workflow.add_node("grader", grade_node)
        workflow.add_edge("retriever", "grader")
        workflow.add_conditional_edges(
            "grader",
            _route_after_grader,
            {"retriever": "retriever", "responder": "responder"},
        )
    else:
        workflow.add_edge("retriever", "responder")

    workflow.add_edge("responder", END)

    compiled = workflow.compile(checkpointer=MemorySaver())
    logfire.info(
        "Agent graph compiled",
        self_correction=settings.ENABLE_SELF_CORRECTION,
        max_refinements=settings.MAX_REFINEMENTS,
    )
    return compiled


rag_agent = build_graph()
