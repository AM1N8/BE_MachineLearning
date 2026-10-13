"""Read ragged session CSVs without allocating padded action columns."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Session:
    """One immutable session; row_id preserves zero-based source order."""

    row_id: int
    browser: str
    tokens: tuple[str, ...]
    label: str | None = None


def read_sessions(path: str | Path, *, labelled: bool) -> list[Session]:
    """Read normal CSV, gzip, or whole-line quote-wrapped CSV; reject blank rows."""
    path = Path(path)
    with path.open("rb") as stream:
        compressed = stream.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    sessions = []
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        for row_id, line in enumerate(stream):
            line = line.rstrip("\r\n")
            if line.startswith('"') and line.endswith('"') and line.count('"') == 2:
                line = line[1:-1]
            fields = next(csv.reader([line], strict=True))
            fields = [unicodedata.normalize("NFC", value) for value in fields]
            offset = 2 if labelled else 1
            if len(fields) < offset or any(not value for value in fields):
                raise ValueError(f"{path}:{row_id + 1}: missing or empty field")
            label = fields[0] if labelled else None
            sessions.append(Session(row_id, fields[offset - 1], tuple(fields[offset:]), label))
    if not sessions:
        raise ValueError(f"{path}: empty dataset")
    return sessions


def file_hash(path: str | Path) -> str:
    """Compute a streaming SHA256 input fingerprint."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    """Write strict, readable UTF-8 JSON; reject nonfinite measurements."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
