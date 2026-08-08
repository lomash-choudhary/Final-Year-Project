"""
Ingestion manifest — incremental re-ingestion and crash recovery.

Records what was ingested, from which file bytes, and with which embedding
backend. Two things fall out of that:

  resume        A run that dies at file 12 of 16 (network drop, quota wall) is
                restarted by re-running the same command. Files 1-11 are skipped.
  no re-spend   Re-running on an unchanged corpus does nothing at all, so the
                free Gemini quota is not spent proving the corpus is unchanged.

The file hash is over the raw bytes, so editing or replacing a PDF re-ingests it
while leaving the other fifteen alone.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import logfire

MANIFEST_VERSION = 1


@dataclass
class FileRecord:
    filename: str
    file_hash: str
    content_hash: str = ""
    status: str = "pending"          # pending | ok | failed | skipped
    chunks: int = 0
    points: int = 0
    pages: int = 0
    loader: str = ""
    strategy: str = ""
    embedding_provider: str = ""
    embedding_model: str = ""
    embedding_dim: int = 0
    ingested_at: str = ""
    error: str = ""
    warnings: list[str] = field(default_factory=list)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Manifest:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records: dict[str, FileRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for name, data in raw.get("files", {}).items():
                known = {f for f in FileRecord.__dataclass_fields__}
                self.records[name] = FileRecord(**{k: v for k, v in data.items() if k in known})
            logfire.info("Manifest loaded", path=str(self.path), files=len(self.records))
        except Exception as exc:
            # A corrupt manifest must not block ingestion — worst case we redo work.
            logfire.warning("Could not read manifest ({err}) — starting fresh", err=str(exc))
            self.records = {}

    def save(self) -> None:
        payload = {
            "version": MANIFEST_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": {name: asdict(record) for name, record in self.records.items()},
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename: a crash mid-write leaves the old manifest intact
            # rather than a truncated one.
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)
        except Exception as exc:
            logfire.warning("Could not write manifest ({err})", err=str(exc))

    def should_skip(self, filename: str, current_hash: str, embedding_dim: int) -> tuple[bool, str]:
        """
        Skip only when the previous run succeeded, the bytes are unchanged, and
        the vectors were produced at the current dimension. A dimension change
        means the stored vectors are unusable regardless of file state.
        """
        record = self.records.get(filename)
        if record is None:
            return False, "not ingested before"
        if record.status != "ok":
            return False, f"previous status was '{record.status}'"
        if record.file_hash != current_hash:
            return False, "file changed on disk"
        if record.embedding_dim and record.embedding_dim != embedding_dim:
            return False, f"embedding dim changed {record.embedding_dim} -> {embedding_dim}"
        return True, "unchanged since last successful ingest"

    def record(self, entry: FileRecord) -> None:
        entry.ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.records[entry.filename] = entry

    def forget_all(self) -> None:
        """Called on --wipe: the collection is gone, so nothing is ingested."""
        self.records = {}

    def summary(self) -> dict:
        by_status: dict[str, int] = {}
        for record in self.records.values():
            by_status[record.status] = by_status.get(record.status, 0) + 1
        return {
            "files_tracked": len(self.records),
            "by_status": by_status,
            "total_chunks": sum(r.chunks for r in self.records.values()),
            "total_points": sum(r.points for r in self.records.values()),
        }
