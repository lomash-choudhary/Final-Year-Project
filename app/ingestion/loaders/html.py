"""HTML loader — strips chrome, keeps readable prose."""

from __future__ import annotations

from pathlib import Path

import logfire

from app.ingestion.loaders.base import ExtractionFailed, LoadedDocument, Page

# Elements that never carry document meaning.
_JUNK_TAGS = ["script", "style", "meta", "noscript", "svg", "nav", "footer", "header", "form"]


def load_html(path: Path) -> LoadedDocument:
    from bs4 import BeautifulSoup

    with logfire.span("HTML extraction", filename=path.name):
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            raise ExtractionFailed(f"Could not read {path.name}: {exc}") from exc

        # lxml is dramatically faster on large pages; html.parser is the stdlib
        # safety net for when lxml is not installed.
        try:
            soup = BeautifulSoup(raw, "lxml")
            parser = "lxml"
        except Exception:
            soup = BeautifulSoup(raw, "html.parser")
            parser = "html.parser"

        for tag in soup(_JUNK_TAGS):
            tag.decompose()

        # separator="\n" keeps block boundaries, which the chunker relies on.
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        cleaned = "\n".join(line for line in lines if line)

        if not cleaned.strip():
            raise ExtractionFailed(f"{path.name} produced no readable text.")

        doc = LoadedDocument(
            filename=path.name,
            path=str(path),
            loader="html",
            pages=[Page(number=1, text=cleaned, extractor=parser)],
        )
        logfire.info("HTML extracted", filename=path.name, chars=doc.char_count, parser=parser)
        return doc
