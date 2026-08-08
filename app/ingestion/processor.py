"""
Ingestion pipeline — CLI entry point.

    load → clean → chunk → persist JSON → embed → index in Qdrant

Run it:
    python -m app.ingestion.processor                    # ingest DATA/, skip unchanged files
    python -m app.ingestion.processor --wipe             # drop the collection and rebuild
    python -m app.ingestion.processor --force            # re-ingest everything, keep the collection
    python -m app.ingestion.processor --dry-run          # parse + chunk only, zero API calls
    python -m app.ingestion.processor --file "paper.pdf" # one document
    python -m app.ingestion.processor DATA/subset        # a different directory

--dry-run is the one to reach for first: it exercises the whole loader and
chunker cascade and reports exactly what would be embedded, without spending a
single unit of free-tier quota.

Failures are isolated per file. One unreadable PDF in a corpus of sixteen must
not take the other fifteen down with it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Tracing must be configured before the modules that emit spans are imported.
from app.observability import configure_observability, report_config_problems

configure_observability("bovine-rag-ingestion")

import logfire  # noqa: E402
from qdrant_client.http import models  # noqa: E402

from app.config import ROOT_DIR, settings  # noqa: E402
from app.ingestion.chunking.splitter import Chunk, chunk_document  # noqa: E402
from app.ingestion.cleaning import clean_pages  # noqa: E402
from app.ingestion.loaders.base import (  # noqa: E402
    SUPPORTED_EXTENSIONS,
    ExtractionFailed,
    UnsupportedFileType,
    load_document,
)
from app.ingestion.manifest import FileRecord, Manifest, file_hash  # noqa: E402
from app.services.retrieval import embedding  # noqa: E402
from app.services.retrieval.embedding import EmbeddingError, embed_texts  # noqa: E402
from app.services.retrieval.qdrant_service import (  # noqa: E402
    DimensionMismatch,
    QdrantUnavailable,
    collection_stats,
    delete_by_source,
    ensure_collection,
    point_id,
    upsert_points,
)


def _save_processed(filename: str, chunks: list[Chunk], meta: dict) -> Path:
    """
    Persist the parsed + chunked output as JSON before embedding.

    This is the debugging artefact that matters: when retrieval returns nonsense,
    the first question is always "what text actually got indexed?", and the
    answer is here rather than buried in a vector database.
    """
    folder = Path(settings.PROCESSED_DATA_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{filename}.json"

    payload = {
        "filename": filename,
        **meta,
        "chunk_count": len(chunks),
        "chunks": [c.to_dict() for c in chunks],
    }
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return dest


def process_file(path: Path, manifest: Manifest, dim: int, force: bool, dry_run: bool) -> FileRecord:
    filename = path.name
    record = FileRecord(filename=filename, file_hash="")

    with logfire.span("Ingest file", filename=filename):
        try:
            record.file_hash = file_hash(path)
        except Exception as exc:
            record.status = "failed"
            record.error = f"unreadable: {exc}"
            logfire.error("Cannot read {filename}: {err}", filename=filename, err=str(exc))
            return record

        if not force:
            skip, reason = manifest.should_skip(filename, record.file_hash, dim)
            if skip:
                previous = manifest.records[filename]
                logfire.info("Skipping {filename} — {reason}", filename=filename, reason=reason)
                previous.status = "ok"
                return previous

        # ── load ──────────────────────────────────────────────────────────────
        try:
            document = load_document(path)
        except UnsupportedFileType as exc:
            record.status = "skipped"
            record.error = str(exc)
            logfire.info("Skipping unsupported file: {filename}", filename=filename)
            return record
        except ExtractionFailed as exc:
            record.status = "failed"
            record.error = f"extraction failed: {exc}"
            logfire.error("Extraction failed for {filename}: {err}", filename=filename, err=str(exc)[:300])
            return record
        except Exception as exc:
            record.status = "failed"
            record.error = f"loader crashed: {exc}"
            logfire.error("Extraction failed for {filename}: {err}", filename=filename, err=str(exc)[:300])
            return record

        record.loader = document.loader
        record.pages = document.page_count
        record.warnings = list(document.warnings)

        # ── clean ─────────────────────────────────────────────────────────────
        document.pages = clean_pages(document.pages)
        if not document.pages:
            record.status = "failed"
            record.error = "no text left after cleaning"
            logfire.warning("{filename} produced no text after cleaning", filename=filename)
            return record

        record.content_hash = document.content_hash()

        # ── chunk ─────────────────────────────────────────────────────────────
        chunks = chunk_document(document)
        if not chunks:
            record.status = "failed"
            record.error = "chunking produced no chunks"
            return record

        record.chunks = len(chunks)
        record.strategy = chunks[0].strategy

        dest = _save_processed(
            filename,
            chunks,
            {
                "path": str(path),
                "loader": document.loader,
                "pages": document.page_count,
                "chars": document.char_count,
                "extractors": document.extractors_used,
                "warnings": document.warnings,
                "chunk_strategy": chunks[0].strategy,
            },
        )
        logfire.info("Saved parsed output", path=str(dest.relative_to(ROOT_DIR)), chunks=len(chunks))

        if dry_run:
            record.status = "ok"
            record.points = 0
            return record

        # ── embed ─────────────────────────────────────────────────────────────
        try:
            vectors = embed_texts([c.text for c in chunks])
        except EmbeddingError as exc:
            record.status = "failed"
            record.error = f"embedding failed: {exc}"
            logfire.error("Embedding failed for {filename}: {err}", filename=filename, err=str(exc)[:300])
            return record

        backend = embedding.get_backend()
        record.embedding_provider = backend.name
        record.embedding_model = backend.model
        record.embedding_dim = backend.dim

        # ── index ─────────────────────────────────────────────────────────────
        # Delete first: if the document shrank, its old tail chunks would
        # otherwise survive as orphans that still match searches.
        delete_by_source(filename)

        points = [
            models.PointStruct(
                id=point_id(filename, chunk.index),
                vector=vector,
                payload={
                    "text": chunk.text,
                    "source": filename,
                    "chunk_index": chunk.index,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "page_label": chunk.page_label,
                    "char_count": chunk.char_count,
                    "chunk_strategy": chunk.strategy,
                    "content_hash": record.content_hash,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        try:
            record.points = upsert_points(points)
        except Exception as exc:
            record.status = "failed"
            record.error = f"indexing failed: {exc}"
            return record

        record.status = "ok"
        logfire.info(
            "Indexed {filename}", filename=filename,
            chunks=record.chunks, points=record.points, pages=record.pages,
        )
        return record


def discover_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    files = [
        p for p in sorted(target.rglob("*"))
        if p.is_file()
        and p.suffix.lower().lstrip(".") in SUPPORTED_EXTENSIONS
        and not p.name.startswith(".")
    ]
    return files


def run(target: Path, wipe: bool, force: bool, dry_run: bool, limit: int | None) -> int:
    started = time.time()

    problems = report_config_problems("ingestion")
    blocking = [p for p in problems if "will fall back" not in p]
    if blocking:
        print("\n Configuration problems:")
        for problem in blocking:
            print(f"   - {problem}")
        print()

    files = discover_files(target)
    if limit:
        files = files[:limit]

    if not files:
        print(f"No supported files found in {target}")
        print(f"   Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return 1

    print(f"\n Ingesting {len(files)} file(s) from {target}")
    print(f"   mode: {'DRY RUN (no API calls, no indexing)' if dry_run else 'full'}"
          f"{' | --wipe' if wipe else ''}{' | --force' if force else ''}\n")

    manifest = Manifest(settings.MANIFEST_PATH)

    if dry_run:
        dim = 0
    else:
        try:
            dim = ensure_collection(wipe=wipe)
        except (QdrantUnavailable, DimensionMismatch) as exc:
            print(f"\n {exc}\n")
            return 2
        except EmbeddingError as exc:
            print(f"\n Embedding backend unavailable:\n   {exc}\n")
            return 2

        if wipe:
            manifest.forget_all()

        backend = embedding.get_backend()
        print(f"   embeddings: {backend.name} / {backend.model} ({backend.dim}-dim)")
        print(f"   qdrant:     {settings.QDRANT_URL} / {settings.QDRANT_COLLECTION}\n")

    with logfire.span("Ingestion run", files=len(files), target=str(target), dry_run=dry_run):
        for i, path in enumerate(files, start=1):
            print(f"[{i}/{len(files)}] {path.name}")
            record = process_file(path, manifest, dim, force, dry_run)
            manifest.record(record)
            manifest.save()  # checkpoint after every file so a crash loses at most one

            icon = {"ok": "  ok  ", "failed": " FAIL ", "skipped": " skip "}.get(record.status, "  ?   ")
            detail = (
                f"{record.chunks} chunks, {record.points} points, {record.pages} pages"
                if record.status == "ok"
                else record.error
            )
            print(f"        [{icon}] {detail}")
            for warning in record.warnings[:2]:
                print(f"          ! {warning}")

    # ── report ────────────────────────────────────────────────────────────────
    ok = [r for r in manifest.records.values() if r.status == "ok"]
    failed = [r for r in manifest.records.values() if r.status == "failed"]
    skipped = [r for r in manifest.records.values() if r.status == "skipped"]

    elapsed = time.time() - started
    print("\n" + "=" * 66)
    print(f" Ingestion complete in {elapsed:.1f}s")
    print(f"   indexed: {len(ok)}   failed: {len(failed)}   unsupported: {len(skipped)}")
    print(f"   chunks:  {sum(r.chunks for r in ok)}   points: {sum(r.points for r in ok)}")

    if not dry_run:
        cache = embedding.describe().get("cache", {})
        if cache.get("enabled"):
            print(f"   embedding cache: {cache.get('rows', 0)} vectors stored, "
                  f"{cache.get('hits', 0)} hits this run")
        print(f"   qdrant: {collection_stats()}")

    if failed:
        print("\n Failed files:")
        for record in failed:
            print(f"   - {record.filename}: {record.error}")
    print("=" * 66 + "\n")

    logfire.info("Ingestion run finished", elapsed_s=round(elapsed, 1), **manifest.summary())
    return 0 if not failed else 3


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingestion.processor",
        description="Parse, chunk, embed and index documents into Qdrant.",
    )
    parser.add_argument("target", nargs="?", default=None, help="File or directory to ingest (default: DATA/)")
    parser.add_argument("--wipe", action="store_true", help="Delete and recreate the Qdrant collection first")
    parser.add_argument("--force", action="store_true", help="Re-ingest files even if unchanged")
    parser.add_argument("--dry-run", action="store_true", help="Parse and chunk only — no API calls, no indexing")
    parser.add_argument("--file", dest="single", default=None, help="Ingest a single file inside DATA/")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N files (useful for a first test)")
    args = parser.parse_args()

    if args.single:
        target = Path(args.single)
        if not target.is_absolute():
            target = Path(settings.DATA_DIR) / args.single
    else:
        target = Path(args.target) if args.target else Path(settings.DATA_DIR)

    if not target.exists():
        print(f"Path does not exist: {target}")
        return 1

    return run(target, wipe=args.wipe, force=args.force, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    sys.exit(main())
