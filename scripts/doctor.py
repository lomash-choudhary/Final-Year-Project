"""
Preflight check — run this before ingestion.

    python -m scripts.doctor            # config + Qdrant + corpus + parsers
    python -m scripts.doctor --live     # also probe the embedding and LLM APIs

The default mode makes zero API calls, so it costs no free-tier quota. `--live`
spends exactly two requests (one embedding, one 5-token completion) to confirm
your keys actually work — far cheaper than discovering a bad key twelve files
into an ingestion run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.observability import configure_observability

configure_observability("bovine-rag-doctor")

from app.config import settings  # noqa: E402
from app.ingestion.loaders.base import SUPPORTED_EXTENSIONS  # noqa: E402

OK, WARN, FAIL = "  OK  ", " WARN ", " FAIL "


def line(state: str, label: str, detail: str = "") -> None:
    print(f"  [{state}] {label:<34} {detail}")


def check_config() -> int:
    print("\nConfiguration")
    problems = settings.validate("ingestion") + settings.validate("api")
    seen: set[str] = set()
    failures = 0

    for problem in problems:
        if problem in seen:
            continue
        seen.add(problem)
        soft = "will fall back" in problem
        line(WARN if soft else FAIL, "config", problem)
        failures += 0 if soft else 1

    if not seen:
        line(OK, "config", "no problems found")

    line(OK if settings.GROQ_API_KEY else WARN, "GROQ_API_KEY", "set" if settings.GROQ_API_KEY else "missing")
    line(
        OK if settings.GROQ_FALLBACK_API_KEY else WARN,
        "GROQ_FALLBACK_API_KEY",
        "set (second free quota)" if settings.GROQ_FALLBACK_API_KEY else "missing — no key-level failover",
    )
    line(OK if settings.GEMINI_API_KEY else FAIL, "GEMINI_API_KEY", "set" if settings.GEMINI_API_KEY else "missing")
    line(
        OK if settings.LOGFIRE_TOKEN else WARN,
        "LOGFIRE_TOKEN",
        "set" if settings.LOGFIRE_TOKEN else "missing — tracing to console only",
    )
    return failures


def check_models() -> int:
    """
    Model names are sourced from .env only — never defaulted in code. A blank one
    silently disables a tier (a dropped rung of the fallback ladder, a skipped
    embedding backend), so it is worth seeing the resolved list before a run.
    """
    print("\nModels  (all names come from .env — none are hardcoded)")
    failures = 0

    def model_line(label: str, value: str, *, required: bool, note: str = "") -> None:
        nonlocal failures
        if value:
            line(OK, label, value)
        else:
            line(FAIL if required else WARN, label, "not set in .env" + (f" — {note}" if note else ""))
            failures += 1 if required else 0

    groq_configured = bool(settings.GROQ_API_KEY or settings.GROQ_FALLBACK_API_KEY)
    model_line("GROQ_PRIMARY_MODEL", settings.GROQ_PRIMARY_MODEL, required=groq_configured,
               note="quality tier has no model")
    model_line("GROQ_FAST_MODEL", settings.GROQ_FAST_MODEL, required=groq_configured,
               note="fast tier has no model")
    line(
        OK if settings.translate_model else WARN,
        "GROQ_TRANSLATE_MODEL",
        settings.GROQ_TRANSLATE_MODEL or f"blank — falls back to GROQ_FAST_MODEL ({settings.GROQ_FAST_MODEL or 'also unset'})",
    )
    model_line("GEMINI_CHAT_MODEL", settings.GEMINI_CHAT_MODEL, required=False,
               note="no last-resort Gemini chat fallback")

    needs_gemini_embed = settings.EMBEDDING_PROVIDER in ("auto", "gemini")
    model_line(
        "GEMINI_EMBEDDING_MODEL", settings.gemini_embedding_model,
        required=needs_gemini_embed,
        note="Gemini embedding tier has no model",
    )

    model_line(
        "LOCAL_EMBEDDING_MODEL", settings.LOCAL_EMBEDDING_MODEL,
        required=settings.EMBEDDING_PROVIDER in ("auto", "local"),
        note="offline embedding fallback unavailable",
    )
    line(OK, "RERANKER_MODEL", settings.RERANKER_MODEL or "blank — FlashRank's own default")
    return failures


def check_qdrant() -> int:
    print("\nQdrant")
    try:
        from app.services.retrieval.qdrant_service import get_client

        collections = get_client().get_collections().collections
        line(OK, "connection", settings.QDRANT_URL)

        names = [c.name for c in collections]
        if settings.QDRANT_COLLECTION in names:
            from app.services.retrieval.qdrant_service import collection_stats
            stats = collection_stats()
            points = stats.get("points", 0)
            line(
                OK if points else WARN,
                "collection",
                f"{settings.QDRANT_COLLECTION}: {points} points, {stats.get('vectors_dim')}-dim",
            )
        else:
            line(WARN, "collection", f"'{settings.QDRANT_COLLECTION}' not created yet (ingestion will create it)")
        return 0

    except Exception as exc:
        line(FAIL, "connection", f"{settings.QDRANT_URL} — {str(exc)[:90]}")
        print("\n         Local Qdrant:  docker compose up -d")
        print("         Cloud Qdrant:  check QDRANT_CLUSTER_ENDPOINT and QDRANT_API_KEY in .env")
        return 1


def check_corpus() -> int:
    print("\nCorpus")
    data_dir = Path(settings.DATA_DIR)

    if not data_dir.exists():
        line(FAIL, "DATA/", f"{data_dir} does not exist")
        return 1

    files = [
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower().lstrip(".") in SUPPORTED_EXTENSIONS and not p.name.startswith(".")
    ]
    unsupported = [
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower().lstrip(".") not in SUPPORTED_EXTENSIONS and not p.name.startswith(".")
    ]

    total_mb = sum(p.stat().st_size for p in files) / (1024 * 1024)
    line(OK if files else FAIL, "supported files", f"{len(files)} files, {total_mb:.1f} MB")
    if unsupported:
        line(WARN, "unsupported files", f"{len(unsupported)} will be skipped")

    return 0 if files else 1


def check_parsers() -> int:
    print("\nParsers and libraries")
    failures = 0

    tiers = [
        ("pypdf", "PDF tier 1", True),
        ("pdfplumber", "PDF tier 2", False),
        ("fitz", "PDF tier 3 (PyMuPDF)", False),
        ("bs4", "HTML", False),
        ("docx", "DOCX", False),
        ("pptx", "PPTX", False),
        ("langchain_text_splitters", "chunker tier 1", False),
        ("flashrank", "reranker", False),
        ("sentence_transformers", "local embedding fallback", False),
        ("langgraph", "agent graph", True),
        ("langchain_groq", "Groq client", True),
        ("langchain_google_genai", "Gemini client", True),
    ]

    for module, label, required in tiers:
        try:
            __import__(module)
            line(OK, label, module)
        except ImportError:
            line(FAIL if required else WARN, label, f"{module} not installed")
            failures += 1 if required else 0

    return failures


def check_live() -> int:
    print("\nLive API probes (spends a little quota)")
    failures = 0

    try:
        from app.services.retrieval import embedding

        vector = embedding.embed_query("bovine theileriosis prevalence")
        backend = embedding.get_backend()
        line(OK, "embeddings", f"{backend.name}/{backend.model} — {len(vector)}-dim")
    except Exception as exc:
        line(FAIL, "embeddings", str(exc)[:110])
        failures += 1

    try:
        from app.llm import router

        response = router.invoke("Reply with the single word: ready", tier="fast", max_tokens=8)
        line(OK, "llm gateway", f"{response.target_label} / {response.model} — '{response.content[:24]}'")
    except Exception as exc:
        line(FAIL, "llm gateway", str(exc)[:110])
        failures += 1

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight check for the RAG pipeline.")
    parser.add_argument("--live", action="store_true", help="Also probe the embedding and LLM APIs")
    args = parser.parse_args()

    print("=" * 74)
    print("  Bovine Disease RAG — preflight check")
    print("=" * 74)

    failures = check_config() + check_models() + check_qdrant() + check_corpus() + check_parsers()
    if args.live:
        failures += check_live()

    print("\n" + "=" * 74)
    if failures:
        print(f"  {failures} blocking problem(s). Fix these before ingesting.")
    else:
        print("  All checks passed.")
        print("  Next: python -m app.ingestion.processor --dry-run")
    print("=" * 74 + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
