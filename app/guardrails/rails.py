"""
Guardrail orchestration — two tiers, cheapest first.

  GUARDRAILS_MODE=off    no gate at all (useful when measuring raw RAG quality)
  GUARDRAILS_MODE=fast   deterministic regex rails only — zero API cost (default)
  GUARDRAILS_MODE=full   fast rails, then NeMo Guardrails for anything ambiguous

`full` is opt-in because it costs one LLM call per request and `nemoguardrails`
is a heavy dependency. If it is selected but cannot be initialised, the system
degrades to `fast` and says so — it never silently runs ungated.
"""

from __future__ import annotations

import threading

import logfire

from app.config import settings
from app.guardrails import fast_rails
from app.guardrails.colang_rules import COLANG_CONTENT, RAIL_INDICATORS, YAML_CONTENT
from app.guardrails.fast_rails import GuardResult

_rails = None
_nemo_state = "uninitialised"   # uninitialised | ready | unavailable | disabled
_lock = threading.Lock()


def initialize_rails() -> str:
    """
    Build the NeMo tier if requested. Called once at API startup so the first
    user request does not pay the initialisation cost. Returns a status string.
    """
    global _rails, _nemo_state

    mode = settings.GUARDRAILS_MODE

    if mode == "off":
        _nemo_state = "disabled"
        logfire.warning("Guardrails are OFF — every query goes straight to the agent")
        return "off"

    if mode == "fast":
        _nemo_state = "disabled"
        logfire.info("Guardrails: fast tier only (deterministic, zero API cost)")
        return "fast"

    with _lock:
        if _rails is not None:
            return "full"
        try:
            from nemoguardrails import LLMRails, RailsConfig

            from app.llm.router import router

            # The fast tier is the cheap model's job too — a 70B model classifying
            # "hello" is quota spent for nothing.
            guard_llm = router.get_chat_model(tier="fast")

            config = RailsConfig.from_content(
                colang_content=COLANG_CONTENT,
                yaml_content=YAML_CONTENT,
            )
            _rails = LLMRails(config, llm=guard_llm)
            _nemo_state = "ready"
            logfire.info("Guardrails: full tier ready (fast rails + NeMo)")
            return "full"

        except Exception as exc:
            _nemo_state = "unavailable"
            logfire.warning(
                "NeMo Guardrails could not start ({err}) — degrading to fast tier only. "
                "Install with `pip install nemoguardrails` or set GUARDRAILS_MODE=fast.",
                err=str(exc)[:300],
            )
            return "fast (nemo unavailable)"


def _nemo_check(message: str) -> GuardResult:
    """Tier 2. Returns a fired result only when a Colang flow clearly matched."""
    if _rails is None:
        return fast_rails.PASS

    try:
        result = _rails.generate(messages=[{"role": "user", "content": message}])
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        for indicator in RAIL_INDICATORS:
            if indicator in content:
                return GuardResult(
                    fired=True,
                    category="nemo_rail",
                    response=content,
                    rule=indicator[:40],
                    tier="nemo",
                )
        return fast_rails.PASS

    except Exception as exc:
        # A guardrail outage must not take the app down. Tier 1 already ran.
        logfire.warning("NeMo tier errored ({err}) — relying on fast rails", err=str(exc)[:200])
        return fast_rails.PASS


def guard(message: str) -> GuardResult:
    """
    Run a user message through the guardrail stack.

    Returns a GuardResult. `fired=True` means respond with `result.response`
    immediately and skip the whole RAG pipeline.
    """
    if settings.GUARDRAILS_MODE == "off":
        return fast_rails.PASS

    with logfire.span("Guardrails", query=message[:120]):
        result = fast_rails.check(message)
        if result.fired:
            logfire.info(
                "Rail fired (fast tier)",
                category=result.category, rule=result.rule, query=message[:80],
            )
            return result

        if settings.GUARDRAILS_MODE == "full" and _nemo_state == "ready":
            result = _nemo_check(message)
            if result.fired:
                logfire.info("Rail fired (NeMo tier)", rule=result.rule, query=message[:80])
                return result

        logfire.info("Guardrails passed")
        return fast_rails.PASS


def status() -> dict:
    return {
        "mode": settings.GUARDRAILS_MODE,
        "fast_tier": settings.GUARDRAILS_MODE != "off",
        "nemo_tier": _nemo_state,
    }
