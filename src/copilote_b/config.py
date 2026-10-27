"""Explicit model configuration and frozen submission weights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    config: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if set(config["members"]) != set(config["weights"]):
        raise ValueError("Every member must have exactly one blend weight")
    weights = list(config["weights"].values())
    if not weights or any(w < 0 for w in weights) or abs(sum(weights) - 1) > 1e-12:
        raise ValueError("Blend weights must be nonnegative and sum to one")
    return config


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(config: dict[str, Any], paths: list[Path]) -> str:
    value = {
        "config": config,
        "data": {p.name: file_hash(p) for p in paths},
        "source": {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")},
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
