"""
Text normalisation for academic PDFs.

Research papers extract badly in specific, predictable ways, and every one of
these defects poisons retrieval:

  ligatures      "beneﬁt" / "ﬁeld"  -> a query for "benefit" never matches
  hyphenation    "haemo-\nprotozoa" -> the key term is split in half
  running heads  journal name + page number repeated on all 124 pages, which
                 dominates the chunk text and drags every embedding toward the
                 same meaningless centroid
  hard wrapping  a newline mid-sentence every ~80 chars

Cleaning happens once, before chunking, so both the vectors and the text shown
as a citation are the cleaned version.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

import logfire

from app.ingestion.loaders.base import Page

# "haemo-\nprotozoa" -> "haemoprotozoa". Requires lowercase before the hyphen so
# we do not destroy genuine compounds like "Theileria-\nBabesia".
_HYPHEN_LINEBREAK = re.compile(r"([a-z])-\s*\n\s*([a-z])")

# A newline that is not a paragraph break: previous line does not end a sentence
# and the next starts lowercase.
_SOFT_WRAP = re.compile(r"(?<![.!?:;\)\]])\n(?=[a-z(])")

_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t ]{2,}")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Lines that are just a page number, or "Page 4 of 11".
_PAGE_NUMBER_LINE = re.compile(r"^\s*(?:page\s+)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?\s*$", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Repair the standard PDF-extraction defects. Order matters."""
    if not text:
        return ""

    # NFKC folds ligatures (ﬁ -> fi, ﬂ -> fl) and full-width forms into ASCII.
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS.sub("", text)

    # De-hyphenate before un-wrapping, otherwise the soft-wrap pass joins the two
    # halves with the hyphen still in the middle.
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _SOFT_WRAP.sub(" ", text)

    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)

    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _candidate_key(line: str) -> str | None:
    """Normalised form of a line, or None if it is too long to be a running head."""
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return None
    # Digits vary per page (page numbers, years) — mask them so "p. 4" and "p. 5"
    # collapse to the same key.
    return re.sub(r"\d+", "#", stripped.lower())


def remove_recurring_lines(pages: list[Page], min_pages: int = 5, threshold: float = 0.6) -> list[Page]:
    """
    Drop headers/footers that repeat across most pages of a document.

    Only runs on documents with at least `min_pages` pages — on a 3-page paper a
    line appearing twice is far more likely to be real content than boilerplate.
    """
    if len(pages) < min_pages:
        return pages

    counts: Counter[str] = Counter()
    for page in pages:
        # A line is counted once per page even if it repeats within the page.
        keys = {k for k in (_candidate_key(ln) for ln in page.text.splitlines()) if k}
        counts.update(keys)

    cutoff = max(2, int(len(pages) * threshold))
    boilerplate = {key for key, n in counts.items() if n >= cutoff}

    if not boilerplate:
        return pages

    cleaned: list[Page] = []
    removed = 0
    for page in pages:
        kept_lines = []
        for line in page.text.splitlines():
            key = _candidate_key(line)
            if (key and key in boilerplate) or _PAGE_NUMBER_LINE.match(line):
                removed += 1
                continue
            kept_lines.append(line)
        cleaned.append(Page(number=page.number, text="\n".join(kept_lines), extractor=page.extractor))

    logfire.info(
        "Stripped running headers/footers",
        distinct_patterns=len(boilerplate),
        lines_removed=removed,
        pages=len(pages),
    )
    return cleaned


def clean_pages(pages: list[Page]) -> list[Page]:
    """Full cleaning pass: de-boilerplate, then normalise each page."""
    deduped = remove_recurring_lines(pages)
    cleaned = [
        Page(number=p.number, text=normalize_text(p.text), extractor=p.extractor)
        for p in deduped
    ]
    return [p for p in cleaned if p.text.strip()]
