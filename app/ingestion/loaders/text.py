"""Plain text / Markdown loader."""

from __future__ import annotations

from pathlib import Path

import logfire

from app.ingestion.loaders.base import ExtractionFailed, LoadedDocument, Page


def load_text(path: Path) -> LoadedDocument:
    with logfire.span("Text extraction", filename=path.name):
        try:
            # errors="replace" rather than "ignore": a mangled character is
            # visible in the output, a silently dropped one is not.
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            raise ExtractionFailed(f"Could not read {path.name}: {exc}") from exc

        if not content.strip():
            raise ExtractionFailed(f"{path.name} is empty.")

        doc = LoadedDocument(
            filename=path.name,
            path=str(path),
            loader="text",
            pages=[Page(number=1, text=content, extractor="builtin")],
        )
        logfire.info("Text extracted", filename=path.name, chars=doc.char_count)
        return doc
