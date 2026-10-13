"""Lossless token decomposition and explicitly hypothetical context propagation."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

TIME_RE = re.compile(r"^t\d+$")
GROUP_START_RE = re.compile(r"[ (<$]".replace(" ", ""))


def context_spans(raw: str) -> list[tuple[int, int]]:
    """Find all balanced context spans, including nested Java stack locations."""
    spans = []
    cursor = 0
    while match := GROUP_START_RE.search(raw, cursor):
        start = match.start()
        opening = raw[start]
        closing = {"(": ")", "<": ">", "$": "$"}[opening]
        depth = 1
        cursor = start + 1
        while cursor < len(raw) and depth:
            char = raw[cursor]
            if char == closing:
                depth -= 1
            elif char == opening and opening != "$":
                depth += 1
            cursor += 1
        if depth:
            raise ValueError(f"Unbalanced context: {raw!r}")
        spans.append((start, cursor))
    return spans


@dataclass(frozen=True)
class Token:
    """Parsed token retaining every literal span for a genuine round trip."""

    parts: tuple[str, ...]
    action: str
    screens: tuple[str, ...]
    configs: tuple[str, ...]
    chains: tuple[str, ...]
    suffix_one: bool
    seconds: int | None

    def reconstruct(self) -> str:
        """Reconstruct the NFC-normalized source, including multiple groups."""
        return "".join(self.parts)


@lru_cache(maxsize=100_000)
def parse_token(raw: str) -> Token:
    """Preserve apostrophes and embedded digits; strip only a final action marker 1."""
    raw = unicodedata.normalize("NFC", raw)
    if not raw:
        raise ValueError("An event token cannot be empty")
    if TIME_RE.fullmatch(raw):
        return Token((raw,), "", (), (), (), False, int(raw[1:]))
    parts, literals = [], []
    groups: dict[str, list[str]] = {"(": [], "<": [], "$": []}
    end = 0
    for start, stop in context_spans(raw):
        literal = raw[end:start]
        group = raw[start:stop]
        parts.extend((literal, group))
        literals.append(literal)
        groups[group[0]].append(group[1:-1])
        end = stop
    parts.append(raw[end:])
    literals.append(raw[end:])
    action = "".join(literals)
    if any(char in action for char in "()<>$"):
        raise ValueError(f"Unbalanced or unsupported context grammar: {raw!r}")
    suffix = action.endswith("1")
    if suffix:
        action = action[:-1]
    if not action:
        raise ValueError(f"Missing action: {raw!r}")
    return Token(
        tuple(parts),
        action,
        tuple(groups["("]),
        tuple(groups["<"]),
        tuple(groups["$"]),
        suffix,
        None,
    )


def filled_context(
    tokens: Iterable[str],
) -> list[tuple[Token, tuple[str, ...], tuple[str, ...], tuple[str, ...]]]:
    """Carry explicit fields forward within a session; reset at logout/server start.

    This is a feature hypothesis, not a claim about the logging implementation.
    Multiple screen groups are retained; missing fields never consult another row.
    """
    screen: tuple[str, ...] = ()
    config: tuple[str, ...] = ()
    chain: tuple[str, ...] = ()
    result = []
    for raw in tokens:
        token = parse_token(raw)
        if token.seconds is not None:
            continue
        if token.action.startswith("Démarrage serveur"):
            screen, config, chain = (), (), ()
        screen = token.screens or screen
        config = token.configs or config
        chain = token.chains or chain
        result.append((token, screen, config, chain))
        if token.action == "Fermeture de session":
            screen, config, chain = (), (), ()
    return result
