"""
Single entry point for tracing setup.

Must be called *before* importing anything that emits spans, otherwise early
module-level work is invisible in Logfire. Every executable in this repo
(app/main.py, ingestion CLI, Streamlit apps, eval runner) calls this first.

Design notes
------------
- `send_to_logfire="if-token-present"` means a missing LOGFIRE_TOKEN degrades to
  console-only tracing instead of raising. Nothing in this project *requires* a
  Logfire account.
- configure() is idempotent here. Streamlit re-executes the whole script on every
  interaction, and calling logfire.configure() repeatedly leaks OTel providers.
"""

from __future__ import annotations

import os

import logfire

from app.config import settings

_configured = False


def configure_observability(service_name: str = "bovine-rag") -> str:
    """
    Configure Logfire + LangSmith. Returns a short human-readable status string
    for display in the UI ("Connected", "Console only", ...).
    """
    global _configured
    if _configured:
        return "Already configured"

    # ── LangSmith: wired through env vars, which is what LangChain reads. ──────
    # Only enable when a key exists; otherwise LangChain retries a dead endpoint
    # on every node and slows the graph down for no benefit.
    if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
        os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    # ── Logfire ───────────────────────────────────────────────────────────────
    # Tried richest-first. Keyword arguments to logfire.configure() have shifted
    # between releases, so a version that rejects `environment` or
    # `ConsoleOptions` degrades to a plainer call rather than losing all tracing.
    # Each attempt is a callable, not a pre-built dict: `logfire.ConsoleOptions`
    # does not exist in every release, and building the argument eagerly would
    # raise AttributeError before the try block could fall back.
    attempts = [
        lambda: dict(
            token=settings.LOGFIRE_TOKEN or None,
            service_name=service_name,
            environment=settings.ENVIRONMENT,
            send_to_logfire="if-token-present",
            console=logfire.ConsoleOptions(min_log_level="info"),
        ),
        lambda: dict(
            token=settings.LOGFIRE_TOKEN or None,
            service_name=service_name,
            send_to_logfire="if-token-present",
        ),
        lambda: dict(token=settings.LOGFIRE_TOKEN or None),
    ]

    status = "Disabled"
    for index, build_kwargs in enumerate(attempts):
        try:
            logfire.configure(**build_kwargs())
            status = "Connected & tracing" if settings.LOGFIRE_TOKEN else "Console only (no LOGFIRE_TOKEN)"
            break
        except Exception as exc:
            if index == len(attempts) - 1:
                # Telemetry must never take the application down.
                print(f"[observability] Logfire configuration failed, continuing without it: {exc}")
                status = f"Disabled ({str(exc)[:120]})"

    _configured = True
    logfire.info(
        "Observability initialised",
        service=service_name,
        logfire=status,
        langsmith=bool(settings.LANGSMITH_API_KEY and settings.LANGSMITH_TRACING),
    )
    return status


def report_config_problems(scope: str) -> list[str]:
    """Validate settings for a scope and log each problem. Returns the problems."""
    problems = settings.validate(scope)
    for p in problems:
        logfire.warning("Config check: {problem}", problem=p)
    if not problems:
        logfire.info("Config check passed", scope=scope, **settings.summary())
    return problems
