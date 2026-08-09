"""
Agent graph.

    translate_in
         │
      planner ──conversational──────────────────────────────► responder ─┐
         │                                                                │
         ├──symptom──► clarifier ──needs info──────────────────────────────┤
         │                 │                                              │
         │              ready                                             │
         │                 ▼                                              │
         └──research──► retriever ──► grader ──sufficient──► advisor ──────┤
                            ▲            │                  (symptom)      │
                            └──weak──────┘                                 │
                                         └──sufficient──► responder ───────┤
                                                          (research)       │
                                                                           ▼
                                                                    translate_out ──► END

Two audiences, one graph:

  **Farmer** ("my cow stopped eating") → clarifier may ask follow-ups first,
  then the advisor gives a home-care-or-vet decision in plain language.

  **Researcher** ("prevalence of theileriosis in India") → straight to
  retrieval and a cited, grounded answer.

Two cycles make this a state machine rather than a pipeline: grader→retriever
(re-search when evidence is weak) and the cross-turn clarifier loop (ask, wait
for the human, then continue).

`translate_in` and `translate_out` bracket everything so the whole middle of the
graph only ever deals with English, whatever the user typed.
"""

from __future__ import annotations

import logfire
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agents.nodes.advisor import advise_node
from app.agents.nodes.clarifier import clarify_node
from app.agents.nodes.grader import grade_node
from app.agents.nodes.planner import planner_node
from app.agents.nodes.responder import generate_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.translator import translate_in_node, translate_out_node
from app.agents.state import AgentState
from app.config import settings


def _route_after_planner(state: AgentState) -> str:
    intent = state.get("intent")
    if intent == "conversational":
        return "responder"
    if intent == "symptom" and settings.ENABLE_CLARIFICATION:
        return "clarifier"
    return "retriever"


def _route_after_clarifier(state: AgentState) -> str:
    """Questions were asked → answer the user now and wait for their reply."""
    return "translate_out" if state.get("awaiting_clarification") else "retriever"


def _route_after_grader(state: AgentState) -> str:
    """Loop back only when the context is weak AND budget remains."""
    if state.get("context_quality") == "weak" and state.get("refinements", 0) <= settings.MAX_REFINEMENTS:
        return "retriever"
    return _answer_node_for(state)


def _answer_node_for(state: AgentState) -> str:
    """Farmers get the advisor, researchers get the cited responder."""
    return "advisor" if state.get("intent") == "symptom" else "responder"


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("translate_in", translate_in_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("responder", generate_node)
    workflow.add_node("advisor", advise_node)
    workflow.add_node("translate_out", translate_out_node)

    workflow.set_entry_point("translate_in")
    workflow.add_edge("translate_in", "planner")

    planner_targets = {"retriever": "retriever", "responder": "responder"}
    if settings.ENABLE_CLARIFICATION:
        workflow.add_node("clarifier", clarify_node)
        planner_targets["clarifier"] = "clarifier"
        workflow.add_conditional_edges(
            "clarifier",
            _route_after_clarifier,
            {"retriever": "retriever", "translate_out": "translate_out"},
        )

    workflow.add_conditional_edges("planner", _route_after_planner, planner_targets)

    if settings.ENABLE_SELF_CORRECTION:
        workflow.add_node("grader", grade_node)
        workflow.add_edge("retriever", "grader")
        workflow.add_conditional_edges(
            "grader",
            _route_after_grader,
            {"retriever": "retriever", "responder": "responder", "advisor": "advisor"},
        )
    else:
        workflow.add_conditional_edges(
            "retriever",
            _answer_node_for,
            {"responder": "responder", "advisor": "advisor"},
        )

    workflow.add_edge("responder", "translate_out")
    workflow.add_edge("advisor", "translate_out")
    workflow.add_edge("translate_out", END)

    compiled = workflow.compile(checkpointer=MemorySaver())
    logfire.info(
        "Agent graph compiled",
        self_correction=settings.ENABLE_SELF_CORRECTION,
        clarification=settings.ENABLE_CLARIFICATION,
        translation=settings.ENABLE_TRANSLATION,
        max_refinements=settings.MAX_REFINEMENTS,
    )
    return compiled


rag_agent = build_graph()
