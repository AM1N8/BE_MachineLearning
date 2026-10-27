"""Label-free, action-aligned full and carried-context streams."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from functools import lru_cache

from .data import Session


@dataclass(frozen=True)
class Trace:
    browser: str
    full: tuple[str, ...]
    context: tuple[str, ...]

    def stream(self, view: str) -> tuple[str, ...]:
        if view not in ("full", "context"):
            raise ValueError(f"Unsupported stream: {view}")
        return self.full if view == "full" else self.context


@lru_cache(maxsize=100000)
def atom(value: str) -> str:
    return re.sub(r"\s+", "_", value)


@lru_cache(maxsize=100000)
def components(action: str) -> tuple[str, str, str, str, str]:
    name = re.split(r"[\(<$]", action)[0].removesuffix("1")
    screen = re.search(r"\((.*)\)", action)
    conf = re.search(r"<([^>]*)>", action)
    chain = re.search(r"\$([^$]*)\$", action)
    return (
        atom(name),
        atom(screen[1]) if screen else "NONE",
        atom(conf[1]) if conf else "NONE",
        atom(chain[1]) if chain else "NONE",
        str(int(action.endswith("1"))),
    )


def prepare(session: Session) -> Trace:
    screen = conf = chain = "NONE"
    full: list[str] = []
    context: list[str] = []
    for action in session.actions:
        name, sc, co, ch, mod = components(action)
        if sc != "NONE":
            screen, conf, chain = sc, co, ch
        else:
            if co != "NONE":
                conf = co
            if ch != "NONE":
                chain = ch
        full.append(sys.intern(atom(action)))
        context.append(
            sys.intern(
                f"s={screen}|a={name}~c={conf}|a={name}~h={chain}|a={name}|m={mod}"
            )
        )
    return Trace(session.browser, tuple(full), tuple(context))


def document(trace: Trace, view: str, bounds: tuple[int, int]) -> str:
    return " ".join(trace.stream(view)[bounds[0] : bounds[1]]) or "EMPTY"
