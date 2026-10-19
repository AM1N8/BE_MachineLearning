"""Training-only provenance, sequence similarity and persistent-error investigation."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import CountVectorizer

from copilote.features import identity
from copilote.io import file_hash, read_sessions, write_json
from copilote.tokens import parse_token


def audit(directory):
    rows = read_sessions("data/train.csv", labelled=True)
    y = np.array([r.label for r in rows])
    events = [tuple(t for t in r.tokens if parse_token(t).seconds is None) for r in rows]
    # Empty trigram sets must not become an artificial shared near-duplicate token.
    documents = [[json.dumps(v) for v in zip(e, e[1:], e[2:])] for e in events]
    matrix = CountVectorizer(analyzer=identity, binary=True, dtype=np.float64).fit_transform(
        documents
    )
    intersections = (matrix @ matrix.T).toarray()
    sizes = np.asarray(matrix.sum(axis=1)).ravel()
    similarity = intersections / np.maximum(1, sizes[:, None] + sizes[None, :] - intersections)
    np.fill_diagonal(similarity, -1)
    lengths = np.array([len(e) for e in events])
    ratio = np.minimum(lengths[:, None], lengths) / np.maximum(
        1, np.maximum(lengths[:, None], lengths)
    )
    links = (similarity >= 0.9) & (ratio >= 0.9)
    # Do not use labels to define dependency groups.
    _, groups = connected_components(csr_matrix(links), directed=False)
    a, b = np.where(np.triu(links, 1))
    pd.DataFrame(
        {
            "a": a,
            "b": b,
            "jaccard": similarity[a, b],
            "length_ratio": ratio[a, b],
            "same_label": y[a] == y[b],
        }
    ).to_csv(directory / "near_duplicates.csv", index=False)
    write_json(directory / "groups.json", groups.tolist())
    prior = Path(
        json.loads(Path("reports/results/improvement/active.json").read_text())["directory"]
    )
    frame = pd.read_csv(prior / "baseline_oof.csv")
    persistent = frame.groupby("row_id").wrong.all()
    indices = persistent.index[persistent].to_numpy()
    nearest = similarity.argmax(axis=1)
    records = []
    first_clock, backwards, screen_count, missing_count = [], 0, 0, 0
    for i, row in enumerate(rows):
        parsed = [parse_token(t) for t in row.tokens]
        clocks = [p.seconds for p in parsed if p.seconds is not None]
        first_clock.append(clocks[0] if clocks else 0)
        backwards += int(any(a > b for a, b in zip(clocks, clocks[1:])))
        actions = [p for p in parsed if p.seconds is None]
        screen_count += sum(bool(p.screens) for p in actions)
        missing_count += sum(not p.screens for p in actions)
        if i in indices:
            same = np.flatnonzero((y == y[i]) & (np.arange(len(y)) != i))
            other = np.flatnonzero(y != y[i])
            records.append(
                {
                    "row_id": i,
                    "truth": y[i],
                    "length": len(actions),
                    "browser": row.browser,
                    "nearest_row": int(nearest[i]),
                    "nearest_label": y[nearest[i]],
                    "nearest_jaccard": float(similarity[i, nearest[i]]),
                    "same_user_max": float(similarity[i, same].max()),
                    "other_user_max": float(similarity[i, other].max()),
                    "explicit_screen_fraction": sum(bool(p.screens) for p in actions)
                    / max(1, len(actions)),
                }
            )
    pd.DataFrame(records).to_csv(directory / "persistent_sessions.csv", index=False)
    browsers = {c: len({r.browser for r in rows if r.label == c}) for c in set(y)}
    counts = Counter(y)
    expected_adjacent = sum(n * (n - 1) for n in counts.values()) / (len(y) * (len(y) - 1))
    summary = {
        "rows": len(rows),
        "classes": len(counts),
        "persistent_sessions": len(indices),
        "near_duplicate_pairs": len(a),
        "cross_label_near_pairs": int((y[a] != y[b]).sum()),
        "groups": len(set(groups)),
        "max_group": max(Counter(groups).values()),
        "exact_event_sequence_duplicates": len(events) - len({tuple(e) for e in events}),
        "multi_browser_users": sum(n > 1 for n in browsers.values()),
        "adjacent_same_label_fraction": float(np.mean(y[:-1] == y[1:])),
        "random_expected_adjacent_fraction": expected_adjacent,
        "row_first_clock_spearman": float(spearmanr(np.arange(len(y)), first_clock).statistic),
        "sessions_clock_decreases": backwards,
        "explicit_screen_events": screen_count,
        "missing_screen_events": missing_count,
        "official_metric": "F1 score moyen; averaging unspecified",
        "metric_verified": False,
        "id_semantics": "row position only; no explicit timestamp, date or source-session group",
        "group_definition": "connected components of event-trigram set Jaccard >= .9 and length ratio >= .9; label-free",
        "source_hashes": {
            str(p): file_hash(p)
            for p in [Path("evaluation.md"), Path("rules.md"), Path("data/README.md")]
        },
    }
    write_json(directory / "data_audit.json", summary)
    print(summary, flush=True)
