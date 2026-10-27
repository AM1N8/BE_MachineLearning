"""Read headerless traces and preserve the sample submission's row order."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

BROWSERS = ("Firefox", "Google Chrome", "Opera", "Microsoft Edge")
TIME = re.compile(r"^t(\d+)$")


@dataclass(frozen=True)
class Session:
    row_id: int
    label: str | None
    browser: str
    tokens: tuple[str, ...]

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(token for token in self.tokens if not TIME.fullmatch(token))

    @property
    def markers(self) -> tuple[int, ...]:
        return tuple(int(token[1:]) for token in self.tokens if TIME.fullmatch(token))


def split_record(line: str) -> list[str]:
    fields: list[str] = []
    buffer: list[str] = []
    depth = 0
    conf = chain = quoted = False
    i = 0
    line = line.rstrip("\r\n")
    while i < len(line):
        char = line[i]
        if quoted:
            if char == '"':
                if i + 1 < len(line) and line[i + 1] == '"':
                    buffer.append('"')
                    i += 1
                else:
                    quoted = False
            else:
                buffer.append(char)
        elif char == '"' and not buffer:
            quoted = True
        elif char == "," and depth == 0 and not conf and not chain:
            fields.append("".join(buffer))
            buffer = []
        else:
            buffer.append(char)
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    raise ValueError("Unbalanced closing parenthesis")
            elif char == "<":
                conf = True
            elif char == ">":
                conf = False
            elif char == "$":
                chain = not chain
        i += 1
    if quoted or depth or conf or chain:
        raise ValueError("Unterminated quote, parenthesis, configuration or chain")
    fields.append("".join(buffer))
    return fields


def parse_line(line: str, row_id: int) -> Session:
    fields = split_record(line)
    if fields[0] in BROWSERS:
        label, browser, tokens = None, fields[0], fields[1:]
    else:
        if len(fields) < 2 or fields[1] not in BROWSERS:
            raise ValueError(f"Row {row_id}: browser not found in field 0 or 1")
        label, browser, tokens = fields[0], fields[1], fields[2:]
        if not re.fullmatch(r"[a-z]{3}", label):
            raise ValueError(f"Invalid trigram {label!r}")
    if not tokens or any(not token for token in tokens):
        raise ValueError(f"Row {row_id}: empty trace/token")
    return Session(row_id, label, browser, tuple(tokens))


def load_sessions(path: Path, labeled: bool) -> list[Session]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        sessions = [parse_line(line, i) for i, line in enumerate(stream, 1)]
    if not sessions or any((s.label is not None) != labeled for s in sessions):
        raise ValueError(f"Unexpected label presence or empty file: {path}")
    return sessions


def data_path(folder: Path, name: str) -> Path:
    matches = [path for path in folder.iterdir() if path.name.lower() == name.lower()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name} in {folder}")
    return matches[0]


def load_sample(folder: Path, n_test: int) -> pd.DataFrame:
    sample = pd.read_csv(data_path(folder, "sample_submission.csv"))
    if list(sample.columns) != ["RowId", "prediction"] or len(sample) != n_test:
        raise ValueError("Sample schema or row count differs from the observed format")
    if sample.RowId.tolist() != list(range(1, n_test + 1)):
        raise ValueError("Sample IDs must match the 1-based test row order")
    return sample
