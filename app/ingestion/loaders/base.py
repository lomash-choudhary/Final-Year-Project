"""
Loader contract + dispatch.

Every loader returns a `LoadedDocument` made of `Page` objects rather than one
flat string. Page granularity is what lets the chunker attribute each chunk to a
page range, which is what makes citations in the UI ("Theileriosis..., p. 4-5")
actually verifiable — the whole point of a RAG system over research papers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Page:
    number: int          # 1-indexed
    text: str
    extractor: str       # which tier produced this page ("pypdf", "pdfplumber", ...)


@dataclass
class LoadedDocument:
    filename: str
    path: str
    pages: list[Page] = field(default_factory=list)
    loader: str = "unknown"
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def extractors_used(self) -> list[str]:
        return sorted({p.extractor for p in self.pages if p.text.strip()})

    def content_hash(self) -> str:
        """Stable hash of extracted content — drives incremental re-ingestion."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


class UnsupportedFileType(Exception):
    pass


class ExtractionFailed(Exception):
    pass


# Extension -> loader module function, resolved lazily so a missing optional
# dependency (e.g. python-pptx) only breaks the format that needs it.
SUPPORTED_EXTENSIONS = {"pdf", "txt", "md", "html", "htm", "docx", "pptx"}


def load_document(file_path: str | Path) -> LoadedDocument:
    """Dispatch a file to the right loader based on its extension."""
    path = Path(file_path)
    ext = path.suffix.lower().lstrip(".")

    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(f"No loader registered for '.{ext}' ({path.name})")

    if ext == "pdf":
        from app.ingestion.loaders.pdf import load_pdf
        return load_pdf(path)
    if ext in ("txt", "md"):
        from app.ingestion.loaders.text import load_text
        return load_text(path)
    if ext in ("html", "htm"):
        from app.ingestion.loaders.html import load_html
        return load_html(path)
    if ext in ("docx", "pptx"):
        from app.ingestion.loaders.office import load_office
        return load_office(path)

    raise UnsupportedFileType(f"Unhandled extension '.{ext}'")
