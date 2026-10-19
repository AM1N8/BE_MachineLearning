"""Reproducible descriptive analysis, with no learned predictive transformations."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from copilote.io import Session, read_sessions, write_json
from copilote.plots import save_figure
from copilote.settings import provenance
from copilote.tokens import parse_token


def session_statistics(row: Session) -> dict[str, float]:
    """Compute per-session descriptors; clock differences remain approximate proxies."""
    parsed = [parse_token(raw) for raw in row.tokens]
    events = [token for token in parsed if token.seconds is None]
    clocks = [token.seconds for token in parsed if token.seconds is not None]
    differences = np.diff(clocks)
    positive = differences[differences >= 0]
    counts = Counter(token.action for token in events)
    probabilities = np.array(list(counts.values()), dtype=float) / max(len(events), 1)
    return {
        "tokens": float(len(parsed)),
        "actions": float(len(events)),
        "unique_actions": float(len(counts)),
        "action_entropy": float(-np.sum(probabilities * np.log2(probabilities))),
        "clock_markers": float(len(clocks)),
        "clock_max_seconds": float(max(clocks, default=0)),
        "clock_positive_sum": float(positive.sum()),
        "clock_resets": float(np.sum(differences < 0)),
        "clock_gap_mean": float(positive.mean()) if len(positive) else 0.0,
        "clock_gap_max": float(positive.max()) if len(positive) else 0.0,
        "actions_per_proxy_second": len(events) / max(sum(positive), 1),
        "screen_fraction": sum(bool(t.screens) for t in events) / max(len(events), 1),
        "config_fraction": sum(bool(t.configs) for t in events) / max(len(events), 1),
        "chain_fraction": sum(bool(t.chains) for t in events) / max(len(events), 1),
        "suffix_fraction": sum(t.suffix_one for t in events) / max(len(events), 1),
        "unique_screens": float(len({x for t in events for x in t.screens})),
        "unique_configs": float(len({x for t in events for x in t.configs})),
        "unique_chains": float(len({x for t in events for x in t.chains})),
        "repetition_fraction": sum(a.action == b.action for a, b in zip(events, events[1:]))
        / max(len(events) - 1, 1),
    }


def run_eda(config: dict[str, Any]) -> dict[str, Any]:
    """Describe class imbalance, clock/context grammar, duplicates and train/test shift."""
    train = read_sessions(config["paths"]["train"], labelled=True)
    test = read_sessions(config["paths"]["test"], labelled=False)
    frames = {
        name: pd.DataFrame([session_statistics(r) for r in rows])
        for name, rows in (("train", train), ("test", test))
    }
    support = Counter(r.label for r in train)
    browsers: dict[str, set[str]] = defaultdict(set)
    sequences: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for row in train:
        browsers[str(row.label)].add(row.browser)
        sequences[(row.browser, *row.tokens)].append(str(row.label))
    vocabulary = {
        name: {t for r in rows for t in r.tokens}
        for name, rows in (("train", train), ("test", test))
    }
    event_counts = Counter(
        parse_token(t).action for r in train for t in r.tokens if parse_token(t).seconds is None
    )
    drift = {
        column: {
            "ks_statistic": float(
                ks_2samp(frames["train"][column], frames["test"][column]).statistic
            ),
            "pvalue_descriptive": float(
                ks_2samp(frames["train"][column], frames["test"][column]).pvalue
            ),
        }
        for column in frames["train"]
    }
    result = {
        "provenance": provenance(config),
        "train_sessions": len(train),
        "test_sessions": len(test),
        "classes": len(support),
        "class_support": dict(sorted(support.items())),
        "support_summary": pd.Series(list(support.values()))
        .describe()
        .replace({np.nan: None})
        .to_dict(),
        "browsers": {
            name: dict(Counter(r.browser for r in rows))
            for name, rows in (("train", train), ("test", test))
        },
        "users_with_multiple_browsers": sum(len(value) > 1 for value in browsers.values()),
        "duplicate_extra_rows": sum(len(labels) - 1 for labels in sequences.values()),
        "conflicting_duplicate_groups": sum(len(set(labels)) > 1 for labels in sequences.values()),
        "train_test_exact_overlap": sum((r.browser, *r.tokens) in sequences for r in test),
        "vocabulary_sizes": {name: len(v) for name, v in vocabulary.items()},
        "test_only_token_types": len(vocabulary["test"] - vocabulary["train"]),
        "test_oov_token_fraction": sum(t not in vocabulary["train"] for r in test for t in r.tokens)
        / max(sum(len(r.tokens) for r in test), 1),
        "zero_action_sessions": {
            name: int((frame.actions == 0).sum()) for name, frame in frames.items()
        },
        "statistics": {
            name: frame.describe(percentiles=[0.05, 0.5, 0.95]).replace({np.nan: None}).to_dict()
            for name, frame in frames.items()
        },
        "shift": drift,
        "top_actions": event_counts.most_common(25),
        "interpretation": {
            "clock": "tNN est traité comme un repère en secondes; différences positives et remises à zéro sont des proxys, pas des durées exactes par action.",
            "browser": "Un navigateur constant par utilisateur serait un filtre de candidats, pas un identifiant unique.",
            "shift": "Tests KS descriptifs sans correction multiple; le test sert à décrire le décalage, jamais à ajuster un vocabulaire.",
            "duplicates": "Les doublons exacts et conflits sont comptés avant interprétation des scores; les sessions restent l'unité de validation imposée.",
        },
    }
    write_json("reports/results/eda.json", result)
    for name, frame in frames.items():
        frame.to_csv(f"reports/results/session_statistics_{name}.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    axes[0].hist(list(support.values()), bins=25)
    axes[0].set(
        xlabel="Sessions par utilisateur", ylabel="Utilisateurs", title="Support des classes"
    )
    for name, frame in frames.items():
        label = "apprentissage" if name == "train" else "test"
        axes[1].hist(np.log1p(frame.actions), bins=30, density=True, alpha=0.5, label=label)
        axes[2].hist(
            np.log1p(frame.clock_max_seconds), bins=30, density=True, alpha=0.5, label=label
        )
    axes[1].set(xlabel="log(1 + actions)", title="Longueur des sessions")
    axes[2].set(xlabel="log(1 + secondes)", title="Repère temporel maximal")
    axes[1].legend()
    save_figure("eda_overview")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    top = event_counts.most_common(12)
    axes[0].barh([x[0] for x in top][::-1], [x[1] for x in top][::-1])
    axes[0].set(title="Actions les plus fréquentes", xlabel="Occurrences (apprentissage)")
    browser_table = pd.crosstab([r.label for r in train], [r.browser for r in train])
    axes[1].imshow(browser_table.to_numpy(), aspect="auto", interpolation="nearest")
    axes[1].set(
        xticks=range(len(browser_table.columns)),
        xticklabels=browser_table.columns,
        xlabel="Navigateur",
        ylabel="Utilisateur (code catégoriel)",
        title="Utilisateurs × navigateur",
    )
    save_figure("eda_actions_browser")
    return result
