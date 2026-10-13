"""Configuration and content-addressed provenance helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import yaml

from copilote.io import file_hash


def load_config(path: str | Path, seed: int | None = None) -> dict[str, Any]:
    """Load YAML, optionally overriding the global seed."""
    config = yaml.safe_load(Path(path).read_text())
    if seed is not None:
        config["seed"] = seed
    return config


def provenance(config: dict[str, Any]) -> dict[str, Any]:
    """Record input, code, configuration and environment fingerprints."""
    source = hashlib.sha256()
    for path in sorted(Path("src/copilote").rglob("*.py")):
        source.update(str(path).encode())
        source.update(path.read_bytes())
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = "unavailable", True
    return {
        "config": config,
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "source_hash": source.hexdigest(),
        "input_hashes": {key: file_hash(config["paths"][key]) for key in ("train", "test")},
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
    }
