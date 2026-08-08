"""
PDF extraction with a three-tier fallback cascade.

Tiers are applied *per page*, not per document. A 124-page journal issue where
pypdf handles 120 pages and chokes on 4 image-heavy ones should cost three
pdfplumber page-opens, not a full re-parse — and the 4 recovered pages must land
back in their original position, or every downstream page citation is wrong.

  tier 1  pypdf      — fastest, handles the large majority of digital PDFs
  tier 2  pdfplumber — slower, markedly better on multi-column academic layouts
  tier 3  pymupdf    — different engine entirely; recovers files the other two
                       cannot open at all. Optional import: if PyMuPDF is not
                       installed the tier is skipped with a warning.

Pages that stay empty after all three tiers are almost always scanned images.
They are reported (not silently dropped) so you know what your corpus is missing.
"""

from __future__ import annotations

from pathlib import Path

import logfire

from app.ingestion.loaders.base import ExtractionFailed, LoadedDocument, Page


def _tier1_pypdf(path: Path) -> tuple[dict[int, str], int]:
    """Returns {page_number: text} for pages that yielded text, and the page count."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    total = len(reader.pages)
    out: dict[int, str] = {}

    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # a single corrupt page must not kill the document
            logfire.warning("pypdf failed on page {page}: {err}", page=i, err=str(exc))
            text = ""
        if text.strip():
            out[i] = text

    return out, total


def _tier2_pdfplumber(path: Path, page_numbers: list[int]) -> dict[int, str]:
    import pdfplumber

    out: dict[int, str] = {}
    with pdfplumber.open(str(path)) as pdf:
        for n in page_numbers:
            if n - 1 >= len(pdf.pages):
                continue
            try:
                text = pdf.pages[n - 1].extract_text() or ""
            except Exception as exc:
                logfire.warning("pdfplumber failed on page {page}: {err}", page=n, err=str(exc))
                continue
            if text.strip():
                out[n] = text
    return out


def _tier3_pymupdf(path: Path, page_numbers: list[int]) -> dict[int, str]:
    import fitz  # PyMuPDF

    out: dict[int, str] = {}
    with fitz.open(str(path)) as doc:
        for n in page_numbers:
            if n - 1 >= doc.page_count:
                continue
            try:
                text = doc.load_page(n - 1).get_text("text") or ""
            except Exception as exc:
                logfire.warning("pymupdf failed on page {page}: {err}", page=n, err=str(exc))
                continue
            if text.strip():
                out[n] = text
    return out


def load_pdf(path: Path) -> LoadedDocument:
    doc = LoadedDocument(filename=path.name, path=str(path), loader="pdf-cascade")

    with logfire.span("PDF extraction", filename=path.name):
        # ── tier 1 ────────────────────────────────────────────────────────────
        extracted: dict[int, str] = {}
        extractor_of: dict[int, str] = {}
        total_pages = 0

        try:
            extracted, total_pages = _tier1_pypdf(path)
            extractor_of = {n: "pypdf" for n in extracted}
            logfire.info(
                "Tier 1 (pypdf): {ok}/{total} pages extracted",
                ok=len(extracted), total=total_pages, filename=path.name,
            )
        except Exception as exc:
            doc.warnings.append(f"pypdf could not open the file: {exc}")
            logfire.warning("Tier 1 (pypdf) could not open {filename}: {err}", filename=path.name, err=str(exc))

        # If pypdf could not even open the file we do not know the page count.
        # Ask PyMuPDF, which is the most tolerant of malformed PDFs.
        if total_pages == 0:
            try:
                import fitz
                with fitz.open(str(path)) as f:
                    total_pages = f.page_count
            except Exception:
                try:
                    import pdfplumber
                    with pdfplumber.open(str(path)) as f:
                        total_pages = len(f.pages)
                except Exception as exc:
                    raise ExtractionFailed(
                        f"No PDF engine could open {path.name}. The file is likely corrupt or encrypted."
                    ) from exc

        # ── tier 2 ────────────────────────────────────────────────────────────
        missing = [n for n in range(1, total_pages + 1) if n not in extracted]
        if missing:
            logfire.info(
                "Tier 2 (pdfplumber): retrying {count} empty pages",
                count=len(missing), filename=path.name,
            )
            try:
                recovered = _tier2_pdfplumber(path, missing)
                extracted.update(recovered)
                extractor_of.update({n: "pdfplumber" for n in recovered})
                if recovered:
                    logfire.info("Tier 2 recovered {n} pages", n=len(recovered), filename=path.name)
            except Exception as exc:
                doc.warnings.append(f"pdfplumber tier failed: {exc}")
                logfire.warning("Tier 2 (pdfplumber) failed: {err}", err=str(exc))

        # ── tier 3 ────────────────────────────────────────────────────────────
        missing = [n for n in range(1, total_pages + 1) if n not in extracted]
        if missing:
            try:
                recovered = _tier3_pymupdf(path, missing)
                extracted.update(recovered)
                extractor_of.update({n: "pymupdf" for n in recovered})
                if recovered:
                    logfire.info("Tier 3 (pymupdf) recovered {n} pages", n=len(recovered), filename=path.name)
            except ImportError:
                doc.warnings.append("PyMuPDF not installed — tier 3 skipped.")
                logfire.info("Tier 3 skipped: PyMuPDF not installed")
            except Exception as exc:
                doc.warnings.append(f"pymupdf tier failed: {exc}")
                logfire.warning("Tier 3 (pymupdf) failed: {err}", err=str(exc))

        # ── assemble in page order ────────────────────────────────────────────
        doc.pages = [
            Page(number=n, text=extracted[n], extractor=extractor_of.get(n, "unknown"))
            for n in sorted(extracted)
        ]

        still_empty = [n for n in range(1, total_pages + 1) if n not in extracted]
        if still_empty:
            preview = still_empty[:12]
            suffix = "..." if len(still_empty) > 12 else ""
            msg = (
                f"{len(still_empty)}/{total_pages} pages yielded no text "
                f"(pages {preview}{suffix}) — likely scanned images. OCR would be needed."
            )
            doc.warnings.append(msg)
            logfire.warning("Unextractable pages in {filename}: {msg}", filename=path.name, msg=msg)

        if not doc.pages:
            raise ExtractionFailed(
                f"{path.name}: all {total_pages} pages are image-only. "
                "Install an OCR pipeline or exclude this file."
            )

        logfire.info(
            "PDF extracted",
            filename=path.name,
            pages_with_text=len(doc.pages),
            total_pages=total_pages,
            chars=doc.char_count,
            engines=doc.extractors_used,
        )

    return doc
