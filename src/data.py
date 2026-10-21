"""Load, validate, and normalize the Copilote usage-trace datasets.

Raw rows have a variable number of action tokens. The normalized representation
keeps one row per session and stores its ordered tokens as a list, rather than
padding every row to the length of the longest session.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

TIME_MARKER_RE = re.compile(r"^t\d+$")
SCREEN_RE = re.compile(r"\((.*?)\)")
CONFIG_RE = re.compile(r"<(.*?)>")
CHAINE_RE = re.compile(r"\$(.*?)\$")
DELIMITERS = ("(", "<", "$", "1")


def _iter_raw_sessions(path: Path, labeled: bool) -> Iterator[dict[str, Any]]:
    """Yield validated sessions, preserving event text and order."""
    required_fields = 2 if labeled else 1
    with path.open(encoding="utf-8", newline="") as source:
        for session_id, row in enumerate(csv.reader(source), start=1):
            if not row:
                raise ValueError(f"{path}: empty CSV row at line {session_id}")
            if len(row) < required_fields:
                raise ValueError(
                    f"{path}: expected at least {required_fields} fields at "
                    f"line {session_id}, got {len(row)}"
                )

            offset = 0
            session: dict[str, Any] = {"session_id": session_id}
            if labeled:
                user = row[0]
                if not user:
                    raise ValueError(f"{path}: missing user at line {session_id}")
                session["util"] = user
                offset = 1

            browser = row[offset]
            if not browser:
                raise ValueError(f"{path}: missing browser at line {session_id}")
            session["navigateur"] = browser

            events = row[offset + 1 :]
            if any(event == "" for event in events):
                raise ValueError(
                    f"{path}: empty event token at line {session_id}"
                )
            session["events"] = events
            yield session


def read_ds(ds_name: str, data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Read ``train`` or ``test`` as one row per session.

    Columns are ``session_id``, optional ``util``, ``navigateur`` and
    ``events``. ``events`` contains the original ordered tokens as a Python
    list. Unlike a padded wide table, this representation does not create
    thousands of mostly-empty action columns.
    """
    if ds_name not in {"train", "test"}:
        raise ValueError("ds_name must be either 'train' or 'test'")

    path = Path(data_dir) / f"{ds_name}.csv"
    return pd.DataFrame(_iter_raw_sessions(path, labeled=ds_name == "train"))


def write_clean_data(
    output_dir: str | Path = DATA_DIR / "processed",
) -> dict[str, int]:
    """Write normalized, compressed session tables without changing raw files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for split in ("train", "test"):
        source = DATA_DIR / f"{split}.csv"
        destination = output_path / f"{split}_sessions.jsonl.gz"
        temporary = destination.with_suffix(destination.suffix + ".tmp")

        try:
            with gzip.open(
                temporary, "wt", encoding="utf-8", newline=""
            ) as compressed:
                count = 0
                for session in _iter_raw_sessions(source, labeled=split == "train"):
                    compressed.write(
                        json.dumps(
                            session,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                    compressed.write("\n")
                    count += 1
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

        counts[split] = count

    return counts


def filter_action(value: str) -> str:
    """Strip screen/configuration/chain suffixes to keep the action verb."""
    for delimiter in DELIMITERS:
        delimiter_index = value.find(delimiter)
        if delimiter_index > 0:
            value = value[:delimiter_index]
    return value


def is_time_marker(value: str) -> bool:
    return bool(TIME_MARKER_RE.fullmatch(value))


def extract(pattern: re.Pattern[str], value: str) -> str | None:
    """Return the first captured value, or ``None`` when it is absent."""
    match = pattern.search(value)
    return match.group(1) if match else None


def _session_feature_counts(events: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    """Return numeric session summaries and counts for actions and contexts."""
    time_values = [int(token[1:]) for token in events if is_time_marker(token)]
    verbs = [filter_action(token) for token in events if not is_time_marker(token)]
    screens = [
        extract(SCREEN_RE, token)
        for token in events
        if "(" in token
    ]
    configs = [
        extract(CONFIG_RE, token)
        for token in events
        if "<" in token
    ]
    chaines = [
        extract(CHAINE_RE, token)
        for token in events
        if "$" in token
    ]
    screens = [value for value in screens if value is not None]
    configs = [value for value in configs if value is not None]
    chaines = [value for value in chaines if value is not None]

    summaries = {
        "n_tokens": len(events),
        "n_actions": len(verbs),
        "n_time_markers": len(time_values),
        "max_time": max(time_values, default=0),
        "duration_seconds": max(time_values, default=0),
        "n_distinct_verbs": len(set(verbs)),
        "n_screens": len(screens),
        "n_distinct_screens": len(set(screens)),
        "n_configs": len(configs),
        "n_distinct_configs": len(set(configs)),
        "n_chaines": len(chaines),
        "n_distinct_chaines": len(set(chaines)),
        "n_modified_actions": sum(token.endswith("1") for token in events),
    }

    counts: Counter[str] = Counter()
    counts.update(f"action__{value}" for value in verbs)
    counts.update(f"screen__{value}" for value in screens)
    counts.update(f"config__{value}" for value in configs)
    counts.update(f"chaine__{value}" for value in chaines)
    return summaries, dict(counts)


def prepare_feature_tables(
    output_dir: str | Path = DATA_DIR / "processed",
) -> dict[str, int]:
    """Export numeric train/test tables using a vocabulary learned from train.

    The target and session ID are retained as metadata columns, not model
    features. Browser values are one-hot encoded. Missing action/context counts
    are written as zero, and test-only action/context values are ignored.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    vocabulary: set[str] = set()
    browsers: set[str] = set()
    for session in _iter_raw_sessions(DATA_DIR / "train.csv", labeled=True):
        browsers.add(session["navigateur"])
        _, counts = _session_feature_counts(session["events"])
        vocabulary.update(counts)

    model_columns = [
        "n_tokens",
        "n_actions",
        "n_time_markers",
        "max_time",
        "duration_seconds",
        "n_distinct_verbs",
        "n_screens",
        "n_distinct_screens",
        "n_configs",
        "n_distinct_configs",
        "n_chaines",
        "n_distinct_chaines",
        "n_modified_actions",
        *(f"browser__{browser}" for browser in sorted(browsers)),
        *sorted(vocabulary),
    ]
    counts_by_split: dict[str, int] = {}

    for split in ("train", "test"):
        labeled = split == "train"
        source = DATA_DIR / f"{split}.csv"
        destination = output_path / f"{split}_features.csv"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        fieldnames = ["session_id"]
        if labeled:
            fieldnames.append("util")
        fieldnames.extend(model_columns)

        try:
            with temporary.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=fieldnames,
                    restval=0,
                    extrasaction="ignore",
                )
                writer.writeheader()
                count = 0
                for session in _iter_raw_sessions(source, labeled=labeled):
                    summaries, feature_counts = _session_feature_counts(
                        session["events"]
                    )
                    record: dict[str, Any] = {
                        "session_id": session["session_id"],
                        **summaries,
                        **feature_counts,
                    }
                    if labeled:
                        record["util"] = session["util"]
                    browser_column = f"browser__{session['navigateur']}"
                    if session["navigateur"] in browsers:
                        record[browser_column] = 1
                    writer.writerow(record)
                    count += 1
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

        counts_by_split[split] = count

    counts_by_split["features"] = len(model_columns)
    return counts_by_split


def session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build simple per-session numeric summaries from normalized sessions.

    ``tNN`` values are kept in the units stated by the dataset description:
    seconds. No extra factor is applied.
    """
    records = []
    for events in df["events"]:
        summaries, _ = _session_feature_counts(events)
        records.append(summaries)

    features = pd.DataFrame(records)
    features.insert(0, "navigateur", df["navigateur"].to_numpy())
    if "util" in df.columns:
        features.insert(0, "util", df["util"].to_numpy())
    return features


def to_categories(df: pd.DataFrame, col: str = "util") -> pd.DataFrame:
    """Convert a string target column into integer category codes."""
    df[col] = pd.Categorical(df[col]).codes
    return df


def main() -> None:
    """Create normalized train/test files under ``data/processed``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "processed",
        help="directory for generated compressed session tables",
    )
    args = parser.parse_args()
    counts = write_clean_data(args.output_dir)
    feature_counts = prepare_feature_tables(args.output_dir)
    for split, count in counts.items():
        print(f"{split}: {count} normalized sessions -> {args.output_dir}")
    print(
        "Feature tables: "
        f"{feature_counts['train']} train rows, "
        f"{feature_counts['test']} test rows, "
        f"{feature_counts['features']} model features -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
