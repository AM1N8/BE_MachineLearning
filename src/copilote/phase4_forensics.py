"""Phase 4 dataset forensics: unused timing channels, suffix-1 semantics, per-user signal."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import kruskal, ks_2samp
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from copilote.improvement import metrics
from copilote.io import Session, file_hash, read_sessions, write_json
from copilote.tokens import filled_context, parse_token

TEMPO_FEATURES = (
    "events",
    "markers",
    "duration",
    "resets",
    "idle_fraction",
    "active_pair_fraction",
    "events_per_second",
    "gap_median",
    "gap_p90",
    "gap_max",
    "events_per_active_tick",
)
DWELL_FEATURES = (
    "dwell_median",
    "dwell_p90",
    "dwell_entropy",
    "dwell_top_fraction",
)
SUFFIX_FEATURES = (
    "suffix_fraction",
    "nogroup_suffix_rate",
    "suffix_after_explicit_rate",
)
SHIFT_FEATURES = (
    "events",
    "duration",
    "events_per_second",
    "idle_fraction",
    "active_pair_fraction",
    "gap_median",
    "dwell_median",
    "suffix_fraction",
)


def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else 0.0


def decompose(row: Session) -> dict[str, Any]:
    """Single-pass timing, dwell and suffix decomposition of one session."""
    parsed = [parse_token(raw) for raw in row.tokens]
    markers = [token.seconds for token in parsed if token.seconds is not None]
    diffs = np.diff(markers) if len(markers) > 1 else np.array([], dtype=float)
    resets = int((diffs < 0).sum())

    last_marker: int | None = None
    pending = 0
    pair_counts: list[int] = []
    pair_durations: list[int] = []
    for token in parsed:
        if token.seconds is None:
            pending += 1
        else:
            if last_marker is not None:
                pair_counts.append(pending)
                pair_durations.append(token.seconds - last_marker)
            last_marker = token.seconds
            pending = 0
    trailing_events = pending
    counts = np.asarray(pair_counts, dtype=float)
    durations = np.asarray(pair_durations, dtype=float)
    valid = durations >= 0
    active = counts > 0
    idle_seconds = float(durations[valid & ~active].sum())
    duration = float(durations[valid].sum())
    active_pair_fraction = float((valid & active).sum() / valid.sum()) if valid.any() else 0.0
    ticks_per_active = counts[active]

    clocks: list[int | None] = []
    clock: int | None = None
    for token in parsed:
        if token.seconds is not None:
            clock = token.seconds
        else:
            clocks.append(clock)
    gaps: list[int | None] = [
        None
        if previous is None or current is None or current - previous < 0
        else current - previous
        for previous, current in zip(clocks, clocks[1:])
    ]

    filled = filled_context(row.tokens)
    screen_seconds: Counter[str] = Counter()
    dwell_values: list[float] = []
    for index in range(len(filled) - 1):
        gap = gaps[index]
        if gap is None:
            continue
        key = "|".join(filled[index][1]) or "__none__"
        screen_seconds[key] += gap
        dwell_values.append(float(gap))
    total_dwell = float(sum(screen_seconds.values()))
    shares = np.asarray(list(screen_seconds.values()), dtype=float)
    entropy = 0.0
    if total_dwell > 0:
        probabilities = shares / total_dwell
        entropy = float(-np.sum(probabilities * np.log2(probabilities)))

    suffix_events = suffix_with_groups = suffix_after_explicit = 0
    group_total = group_suffix = nogroup_total = nogroup_suffix = 0
    previous_explicit = False
    for token, _screens, _configs, _chains in filled:
        has_groups = bool(token.screens or token.configs or token.chains)
        if token.suffix_one:
            suffix_events += 1
            suffix_with_groups += has_groups
            suffix_after_explicit += previous_explicit
        if has_groups:
            group_total += 1
            group_suffix += token.suffix_one
        else:
            nogroup_total += 1
            nogroup_suffix += token.suffix_one
        previous_explicit = has_groups

    gap_values = [float(gap) for gap in gaps if gap is not None]
    events = len(filled)
    return {
        "row_id": row.row_id,
        "label": row.label,
        "browser": row.browser,
        "events": events,
        "markers": len(markers),
        "duration": duration,
        "resets": resets,
        "trailing_events": trailing_events,
        "idle_seconds": idle_seconds,
        "idle_fraction": idle_seconds / duration if duration else 0.0,
        "active_pair_fraction": active_pair_fraction,
        "events_per_second": events / duration if duration else 0.0,
        "gap_median": _quantile(gap_values, 0.5),
        "gap_p90": _quantile(gap_values, 0.9),
        "gap_max": _quantile(gap_values, 1.0),
        "gap_valid_fraction": len(gap_values) / max(events - 1, 1),
        "events_per_active_tick": float(np.median(ticks_per_active)) if ticks_per_active.size else 0.0,
        "dwell_median": _quantile(dwell_values, 0.5),
        "dwell_p90": _quantile(dwell_values, 0.9),
        "dwell_entropy": entropy,
        "dwell_top_fraction": float(shares.max() / total_dwell) if total_dwell else 0.0,
        "suffix_events": suffix_events,
        "suffix_fraction": suffix_events / max(events, 1),
        "suffix_with_groups": suffix_with_groups,
        "suffix_after_explicit_rate": suffix_after_explicit / max(suffix_events, 1),
        "group_suffix_rate": group_suffix / max(group_total, 1),
        "nogroup_suffix_rate": nogroup_suffix / max(nogroup_total, 1),
        "screen_seconds": dict(screen_seconds),
    }


def suffix_action_table(rows: list[Session]) -> pd.DataFrame:
    """Count suffix-1 occurrences per bare action across sessions."""
    total: Counter[str] = Counter()
    suffix: Counter[str] = Counter()
    for row in rows:
        for token, _screens, _configs, _chains in filled_context(row.tokens):
            total[token.action] += 1
            suffix[token.action] += token.suffix_one
    frame = pd.DataFrame(
        [
            {"action": action, "events": count, "suffix_events": suffix[action]}
            for action, count in total.items()
        ]
    )
    frame["suffix_rate"] = frame.suffix_events / frame.events
    return frame.sort_values(["suffix_events", "events"], ascending=False)


def token_association(rows: list[Session]) -> dict[str, Any]:
    """Descriptive in-sample token/user exclusivity and chi-square association."""
    session_counts = [Counter(raw for raw in row.tokens if parse_token(raw).seconds is None) for row in rows]
    token_users: dict[str, set[str]] = {}
    for row, counts in zip(rows, session_counts):
        for token in counts:
            token_users.setdefault(token, set()).add(str(row.label))
    labels = sorted({str(row.label) for row in rows})
    label_index = {label: index for index, label in enumerate(labels)}
    exclusive = {token for token, users in token_users.items() if len(users) == 1}
    vocabulary = sorted(token_users)
    presence = sparse.lil_matrix((len(rows), len(vocabulary)), dtype=np.int64)
    token_index = {token: index for index, token in enumerate(vocabulary)}
    for row_index, counts in enumerate(session_counts):
        for token, count in counts.items():
            presence[row_index, token_index[token]] = count
    presence = presence.tocsr()
    binary = sparse.csr_matrix((presence > 0).astype(np.int64))
    users = np.asarray([label_index[str(row.label)] for row in rows])
    statistics, pvalues = chi2(binary, users)
    top_rows = []
    order = np.argsort(-statistics)[:30]
    for position in order:
        token = vocabulary[position]
        column = np.asarray(presence[:, position].todense()).ravel()
        counts_per_user: Counter[str] = Counter()
        for label, count in zip(users, column):
            if count:
                counts_per_user[labels[label]] += int(count)
        top_rows.append(
            {
                "token": token,
                "chi2": float(statistics[position]),
                "p_value": float(pvalues[position]),
                "users": len(token_users[token]),
                "dominant_user": counts_per_user.most_common(1)[0][0],
                "dominant_user_count": counts_per_user.most_common(1)[0][1],
            }
        )
    coverage = []
    for row, counts in zip(rows, session_counts):
        distinct = set(counts)
        exclusive_hits = distinct & exclusive
        coverage.append(
            {
                "row_id": row.row_id,
                "label": str(row.label),
                "distinct_tokens": len(distinct),
                "exclusive_tokens": len(exclusive_hits),
                "exclusive_fraction": len(exclusive_hits) / max(len(distinct), 1),
            }
        )
    coverage_frame = pd.DataFrame(coverage)
    per_user = (
        coverage_frame.groupby("label")
        .agg(
            sessions=("row_id", "count"),
            mean_exclusive_fraction=("exclusive_fraction", "mean"),
            sessions_with_exclusive=("exclusive_tokens", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )
    return {
        "distinct_event_tokens": len(vocabulary),
        "multi_user_tokens": len(vocabulary) - len(exclusive),
        "exclusive_tokens": len(exclusive),
        "top_chi2_tokens": top_rows,
        "per_user_exclusive": per_user.to_dict(orient="records"),
        "sessions_with_any_exclusive": int((coverage_frame.exclusive_tokens > 0).sum()),
        "caveat": "in-sample descriptive association; fold-local views are used for any modeling",
    }


def kw_table(frame: pd.DataFrame, features: tuple[str, ...]) -> list[dict[str, Any]]:
    """Kruskal-Wallis test of each session-level feature across user labels."""
    groups = {label: block for label, block in frame.groupby("label")}
    rows = []
    for feature in features:
        columns = [group[feature].to_numpy() for group in groups.values()]
        if any(column.size == 0 for column in columns) or frame[feature].nunique() < 2:
            continue
        result = kruskal(*columns)
        rows.append(
            {
                "feature": feature,
                "kruskal_h": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_bonferroni"] = (table.p_value * len(table)).clip(upper=1.0)
        table = table.sort_values("kruskal_h", ascending=False)
    return table.replace({np.nan: None}).to_dict(orient="records")


def probe(
    frame: pd.DataFrame, features: tuple[str, ...], y: np.ndarray, seed: int
) -> dict[str, Any]:
    """Exploratory signal probe: scaled logistic regression on session scalars."""
    matrix = frame[list(features)].to_numpy(dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    prediction = cross_val_predict(pipeline, matrix, y, cv=folds)
    scores = metrics(y, prediction)
    return {
        "features": list(features),
        "n_features": len(features),
        **scores,
        "role": "exploratory signal probe, not model selection",
    }


def run(seed: int = 42) -> dict[str, Any]:
    """Execute the full forensics pass and persist machine-readable artifacts."""
    manifest = {
        "sources": {
            name: file_hash(f"src/copilote/{name}.py")
            for name in ("phase4_forensics", "tokens", "io", "improvement")
        },
        "train": file_hash("data/train.csv"),
        "test": file_hash("data/test.csv"),
        "seed": seed,
        "stage": "forensics",
    }
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    directory = Path("reports/results/phase4") / signature[:12]
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "manifest.json", manifest)

    train = read_sessions("data/train.csv", labelled=True)
    test = read_sessions("data/test.csv", labelled=False)
    train_rows = [decompose(row) for row in train]
    test_rows = [decompose(row) for row in test]
    train_frame = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "screen_seconds"} for row in train_rows]
    )
    test_frame = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "screen_seconds"} for row in test_rows]
    )
    y = np.asarray([str(row.label) for row in train])

    marker_diffs: Counter[str] = Counter()
    for rows in (train, test):
        for row in rows:
            seconds = [parse_token(raw).seconds for raw in row.tokens]
            seconds = [value for value in seconds if value is not None]
            marker_diffs.update(
                f"{int(b) - int(a)}" for a, b in zip(seconds, seconds[1:]) if b - a >= 0
            )
    diff_total = sum(marker_diffs.values())
    timing = {
        "marker_diff_top": [
            {"diff_seconds": int(diff), "count": count, "fraction": count / diff_total}
            for diff, count in marker_diffs.most_common(10)
        ],
        "trailing_events_sessions": int((train_frame.trailing_events > 0).sum())
        + int((test_frame.trailing_events > 0).sum()),
        "resets_sessions_train": int((train_frame.resets > 0).sum()),
        "resets_sessions_test": int((test_frame.resets > 0).sum()),
        "duration_seconds": {
            name: {
                "median": float(frame.duration.median()),
                "p90": float(frame.duration.quantile(0.9)),
                "max": float(frame.duration.max()),
            }
            for name, frame in (("train", train_frame), ("test", test_frame))
        },
    }

    shift = []
    for feature in SHIFT_FEATURES:
        result = ks_2samp(train_frame[feature], test_frame[feature])
        shift.append(
            {
                "feature": feature,
                "ks_statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "train_median": float(train_frame[feature].median()),
                "test_median": float(test_frame[feature].median()),
            }
        )

    per_user = (
        train_frame.groupby("label")
        .agg(
            sessions=("row_id", "count"),
            browser=("browser", "first"),
            events=("events", "median"),
            duration=("duration", "median"),
            events_per_second=("events_per_second", "median"),
            idle_fraction=("idle_fraction", "median"),
            gap_median=("gap_median", "median"),
            dwell_median=("dwell_median", "median"),
            suffix_fraction=("suffix_fraction", "median"),
        )
        .reset_index()
    )
    per_user.to_csv(directory / "tempo_per_user.csv", index=False)

    dwell_records = []
    for row in train_rows:
        for screen, seconds in row["screen_seconds"].items():
            dwell_records.append(
                {
                    "label": row["label"],
                    "screen": screen,
                    "seconds": seconds,
                }
            )
    dwell_frame = pd.DataFrame(dwell_records)
    dwell_per_user = (
        dwell_frame.groupby(["label", "screen"])
        .agg(total_seconds=("seconds", "sum"), visits=("seconds", "count"))
        .reset_index()
        .sort_values(["label", "total_seconds"], ascending=[True, False])
    )
    dwell_per_user.to_csv(directory / "dwell_per_user_screen.csv", index=False)

    suffix_actions = suffix_action_table(train)
    suffix_actions.to_csv(directory / "suffix_by_action.csv", index=False)

    probes = {
        "tempo_scalars": probe(train_frame, TEMPO_FEATURES, y, seed),
        "dwell_scalars": probe(train_frame, DWELL_FEATURES, y, seed),
        "suffix_scalars": probe(train_frame, SUFFIX_FEATURES, y, seed),
        "timing_all_scalars": probe(
            train_frame, TEMPO_FEATURES + DWELL_FEATURES + SUFFIX_FEATURES, y, seed
        ),
    }

    results = {
        "signature": signature,
        "directory": str(directory),
        "timing_structure": timing,
        "train_test_shift_ks": shift,
        "tempo_tests_kruskal_wallis": kw_table(
            train_frame, TEMPO_FEATURES + DWELL_FEATURES + SUFFIX_FEATURES
        ),
        "signal_probes": probes,
        "suffix_semantics": {
            "sessions_train": int(len(train_frame)),
            "sessions_with_suffix": int((train_frame.suffix_events > 0).sum()),
            "suffix_events_total": int(train_frame.suffix_events.sum()),
            "suffix_events_with_explicit_groups": int(train_frame.suffix_with_groups.sum()),
            "mean_group_suffix_rate": float(train_frame.group_suffix_rate.mean()),
            "mean_nogroup_suffix_rate": float(train_frame.nogroup_suffix_rate.mean()),
            "mean_suffix_after_explicit_rate": float(train_frame.suffix_after_explicit_rate.mean()),
        },
        "token_association": token_association(train),
        "seed": seed,
    }
    write_json(directory / "forensics.json", results)
    train_frame.to_csv(directory / "session_timing_train.csv", index=False)
    test_frame.to_csv(directory / "session_timing_test.csv", index=False)
    print(
        json.dumps(
            {
                "directory": str(directory),
                "probes": {name: round(value["macro_f1"], 4) for name, value in probes.items()},
                "suffix_with_groups": results["suffix_semantics"][
                    "suffix_events_with_explicit_groups"
                ],
                "suffix_total": results["suffix_semantics"]["suffix_events_total"],
            },
            indent=2,
        )
    )
    return results


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Phase 4 dataset forensics")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    run(arguments.seed)


if __name__ == "__main__":
    main()
