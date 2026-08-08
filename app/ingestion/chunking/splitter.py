"""
Chunking with a strategy cascade and page attribution.

Three strategies, tried in order, each validated before being accepted:

  1. recursive  — LangChain RecursiveCharacterTextSplitter. Splits on the
                  largest natural boundary that fits (section break -> paragraph
                  -> sentence -> word), so chunks stay semantically whole.
  2. paragraph  — greedy paragraph accumulation. No third-party dependency.
                  Used when langchain-text-splitters is unavailable or when the
                  recursive pass produces something unusable.
  3. window     — fixed character window with overlap. Always succeeds. Reserved
                  for pathological input (a 200 KB document with zero newlines,
                  which is what a badly-extracted two-column PDF looks like).

Every strategy is checked by `_validate()` before acceptance. A splitter that
silently returns one 200 KB "chunk" is worse than no splitter at all: it embeds
fine, retrieves for everything, and blows the LLM context budget.

Page attribution: chunks are located back in the source text by offset, then
mapped to the page span they came from. That is what fills "p. 4-5" in a citation.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import asdict, dataclass

import logfire

from app.config import settings
from app.ingestion.loaders.base import LoadedDocument, Page

_PAGE_JOINER = "\n\n"


@dataclass
class Chunk:
    text: str
    index: int
    page_start: int | None
    page_end: int | None
    strategy: str
    char_count: int

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def page_label(self) -> str:
        if self.page_start is None:
            return "n/a"
        if self.page_end is None or self.page_end == self.page_start:
            return str(self.page_start)
        return f"{self.page_start}-{self.page_end}"


# ── strategies ─────────────────────────────────────────────────────────────────

def _split_recursive(text: str, size: int, overlap: int) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        # Ordered widest-to-narrowest. "\n\n" is a paragraph, ". " a sentence.
        separators=["\n\n\n", "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        keep_separator=True,
    )
    return splitter.split_text(text)


def _split_paragraph(text: str, size: int, overlap: int) -> list[str]:
    """Greedy paragraph packing with a tail-overlap carried into the next chunk."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # A single oversized paragraph cannot be packed — window it directly.
        if len(para) > size:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_window(para, size, overlap))
            continue

        if len(current) + len(para) + 2 <= size:
            current += para + "\n\n"
        else:
            if current.strip():
                chunks.append(current.strip())
            tail = current[-overlap:] if overlap and current else ""
            current = (tail + para + "\n\n") if tail else (para + "\n\n")

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _split_window(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-width sliding window. Cannot fail; produces the least coherent chunks."""
    if not text.strip():
        return []

    step = max(1, size - overlap)
    chunks: list[str] = []
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(text):
            break
    return chunks


# ── validation ─────────────────────────────────────────────────────────────────

def _validate(chunks: list[str], source_len: int, size: int) -> tuple[bool, str]:
    """Reject output that would quietly degrade retrieval."""
    if not chunks:
        return False, "produced zero chunks"

    if all(not c.strip() for c in chunks):
        return False, "all chunks are blank"

    longest = max(len(c) for c in chunks)
    # 2x slack: RecursiveCharacterTextSplitter can legitimately overshoot when a
    # single atomic unit exceeds chunk_size. 2x over is a genuine failure.
    if longest > size * 2:
        return False, f"longest chunk is {longest} chars (limit {size * 2})"

    # Overlap makes total > source, so this only catches real content loss.
    total = sum(len(c) for c in chunks)
    if source_len > 0 and total < source_len * 0.5:
        return False, f"retained only {total}/{source_len} chars ({total / source_len:.0%})"

    return True, "ok"


# ── page attribution ───────────────────────────────────────────────────────────

def _build_page_index(pages: list[Page]) -> tuple[str, list[int], list[int]]:
    """
    Concatenate pages and return (full_text, start_offsets, page_numbers) where
    start_offsets[i] is where page_numbers[i] begins in full_text.
    """
    parts: list[str] = []
    offsets: list[int] = []
    numbers: list[int] = []
    cursor = 0

    for page in pages:
        offsets.append(cursor)
        numbers.append(page.number)
        parts.append(page.text)
        cursor += len(page.text) + len(_PAGE_JOINER)

    return _PAGE_JOINER.join(parts), offsets, numbers


def _page_at(offset: int, offsets: list[int], numbers: list[int]) -> int | None:
    if not offsets:
        return None
    idx = bisect_right(offsets, offset) - 1
    return numbers[max(0, idx)]


def _locate(full_text: str, chunk: str, search_from: int) -> int:
    """
    Find a chunk's offset in the source. Chunks are substrings of the source for
    every strategy here, so a plain find works. The fallback handles the one case
    where it does not: a splitter that normalised whitespace inside the chunk.
    """
    pos = full_text.find(chunk, max(0, search_from))
    if pos != -1:
        return pos
    # Retry from the beginning (overlap can push a chunk before the cursor).
    pos = full_text.find(chunk, 0)
    if pos != -1:
        return pos
    # Last resort: anchor on the chunk's first 60 characters.
    probe = chunk[:60]
    return full_text.find(probe, max(0, search_from)) if probe else -1


# ── public API ─────────────────────────────────────────────────────────────────

def chunk_document(
    doc: LoadedDocument,
    size: int | None = None,
    overlap: int | None = None,
    min_chars: int | None = None,
) -> list[Chunk]:
    """Chunk a loaded document, returning page-attributed chunks."""
    size = size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP
    min_chars = settings.MIN_CHUNK_CHARS if min_chars is None else min_chars

    if overlap >= size:
        logfire.warning(
            "CHUNK_OVERLAP >= CHUNK_SIZE, clamping overlap to size//5",
            size=size, overlap=overlap,
        )
        overlap = size // 5

    full_text, offsets, numbers = _build_page_index(doc.pages)

    with logfire.span(
        "Chunking", filename=doc.filename, chars=len(full_text), pages=len(doc.pages)
    ):
        if not full_text.strip():
            logfire.warning("Nothing to chunk", filename=doc.filename)
            return []

        strategies = [
            ("recursive", _split_recursive),
            ("paragraph", _split_paragraph),
            ("window", _split_window),
        ]

        raw_chunks: list[str] = []
        chosen = "none"

        for name, fn in strategies:
            try:
                candidate = fn(full_text, size, overlap)
            except ImportError as exc:
                logfire.info(
                    "Chunk strategy '{name}' unavailable ({err}) — trying next tier",
                    name=name, err=str(exc),
                )
                continue
            except Exception as exc:
                logfire.warning(
                    "Chunk strategy '{name}' raised ({err}) — falling back",
                    name=name, err=str(exc),
                )
                continue

            ok, reason = _validate(candidate, len(full_text), size)
            if ok:
                raw_chunks, chosen = candidate, name
                break

            logfire.warning(
                "Chunk strategy '{name}' rejected: {reason} — falling back",
                name=name, reason=reason, filename=doc.filename,
            )

        if not raw_chunks:
            logfire.error("Every chunk strategy failed", filename=doc.filename)
            return []

        # Drop fragments too small to carry meaning (page-number leftovers,
        # stray footnote markers). They pollute retrieval and cost API quota.
        kept = [c.strip() for c in raw_chunks if len(c.strip()) >= min_chars]
        dropped = len(raw_chunks) - len(kept)

        chunks: list[Chunk] = []
        cursor = 0
        for i, text in enumerate(kept):
            pos = _locate(full_text, text, cursor)
            if pos == -1:
                page_start = page_end = None
            else:
                page_start = _page_at(pos, offsets, numbers)
                page_end = _page_at(pos + len(text) - 1, offsets, numbers)
                cursor = pos + max(1, len(text) - overlap)

            chunks.append(
                Chunk(
                    text=text,
                    index=i,
                    page_start=page_start,
                    page_end=page_end,
                    strategy=chosen,
                    char_count=len(text),
                )
            )

        avg = sum(c.char_count for c in chunks) // len(chunks) if chunks else 0
        logfire.info(
            "Chunked document",
            filename=doc.filename,
            strategy=chosen,
            chunks=len(chunks),
            dropped_small=dropped,
            avg_chars=avg,
        )
        return chunks
