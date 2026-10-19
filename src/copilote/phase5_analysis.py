"""Summarize the frozen phase 5 audit and investigate reverse source discrimination."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from copilote.competitive import EventStream
from copilote.io import file_hash, read_sessions, write_json
from copilote.phase2_models import BehavioralView
from copilote.phase4_candidates import TempoView
from copilote.phase4_challenger import PROTOCOLS
from copilote.phase5_audit import MODELS, PAIRS, vectorizer


def vocabulary_transfer(directory: Path) -> None:
    """Compare full-train/test OOV with honest small-holdout OOV by feature view."""
    rows = read_sessions("data/train.csv", labelled=True)
    test = read_sessions("data/test.csv", labelled=False)
    splits = json.loads((directory / "splits.json").read_text())
    records = []
    for name, view in {
        "raw": EventStream("raw"),
        "signatures": BehavioralView("signatures"),
        "burst": TempoView(("burst",)),
    }.items():
        training_docs, test_docs = view.transform(rows), view.transform(test)
        for fold, split in enumerate([*splits, None]):
            if split is None:
                a = range(len(rows))
                validation = test_docs
                kind = "full_train_test"
            else:
                a = split["train"]
                validation = [training_docs[i] for i in split["validation"]]
                kind = split["kind"]
            vocabulary = {token for i in a for token in training_docs[i]}
            fractions = [
                sum(token not in vocabulary for token in doc) / len(doc) for doc in validation
            ]
            records.append(
                {
                    "view": name,
                    "fold": fold,
                    "kind": kind,
                    "session_mean_oov": float(np.mean(fractions)),
                    "occurrence_oov": float(
                        sum(sum(token not in vocabulary for token in doc) for doc in validation)
                        / sum(map(len, validation))
                    ),
                }
            )
    pd.DataFrame(records).to_csv(directory / "vocabulary_transfer.csv", index=False)


def source_permutations(directory: Path, permutations: int = 39) -> None:
    """Refit the entire source classifier under random source-label assignments.

    The statistic is two-sided distance from .5, precluding post-hoc inversion
    of an unexpected below-chance AUC. Diagnostic resamples are not extra data.
    """
    rows = read_sessions("data/train.csv", labelled=True)
    test = read_sessions("data/test.csv", labelled=False)
    both = rows + test
    views = {
        "raw": EventStream("raw"),
        "signatures": BehavioralView("signatures"),
        "burst": TempoView(("burst",)),
    }
    docs = {name: view.transform(both) for name, view in views.items()}
    combined = [
        [f"{name}:{token}" for name in views for token in docs[name][i]]
        + [f"browser:{row.browser}"]
        for i, row in enumerate(both)
    ]
    actual = np.r_[np.zeros(len(rows), dtype=int), np.ones(len(test), dtype=int)]
    rng = np.random.default_rng(42)
    path = directory / "source_permutations.csv"
    records = pd.read_csv(path).to_dict("records") if path.exists() else []
    for draw in range(permutations):
        labels = rng.permutation(actual)
        if draw < len(records):
            continue
        score = np.zeros(len(both))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(labels, labels):
            model = make_pipeline(
                vectorizer(min_df=2, max_features=60000, sublinear_tf=True),
                LogisticRegression(C=1, class_weight="balanced", max_iter=1000, random_state=42),
            )
            with threadpool_limits(limits=2):
                model.fit([combined[i] for i in a], labels[a])
                score[b] = model.predict_proba([combined[i] for i in b])[:, 1]
        records.append({"draw": draw, "auc": float(roc_auc_score(labels, score))})
        pd.DataFrame(records).to_csv(path, index=False)
    observed = json.loads((directory / "adversarial.json").read_text())["pooled_auc"]
    null = np.array([record["auc"] for record in records])
    write_json(
        directory / "source_permutation_summary.json",
        {
            "observed_auc": observed,
            "permutations": permutations,
            "null_mean": float(null.mean()),
            "null_quantiles": np.quantile(null, [0.025, 0.975]).tolist(),
            "two_sided_p": float(
                (1 + (abs(null - 0.5) >= abs(observed - 0.5)).sum()) / (len(null) + 1)
            ),
            "not_a_hidden_user_label_test": True,
            "interpretation": "source assignment may be non-iid; below-chance is not no-shift evidence",
        },
    )


def tables(directory: Path) -> None:
    """Persist every numerical summary used by the French audit report."""
    scores = pd.read_csv(directory / "holdout_metrics.csv")
    scores.groupby(["kind", "model"]).agg(
        macro_mean=("macro_f1", "mean"),
        macro_sd=("macro_f1", "std"),
        macro_min=("macro_f1", "min"),
        macro_max=("macro_f1", "max"),
        accuracy=("accuracy", "mean"),
        classes_min=("classes", "min"),
        classes_max=("classes", "max"),
    ).reset_index().to_csv(directory / "holdout_summary.csv", index=False)
    simulations = pd.read_csv(directory / "small_sample_simulations.csv")
    bootstrap = pd.read_csv(directory / "cluster_bootstrap.csv")
    output = []
    for source, frame, group in (
        ("finite_subset", simulations, "scenario"),
        ("cluster_bootstrap", bootstrap, "unit"),
    ):
        for scenario, subset in frame.groupby(group):
            for candidate, reference in PAIRS:
                delta = subset[candidate] - subset[reference]
                output.append(
                    {
                        "source": source,
                        "scenario": scenario,
                        "candidate": candidate,
                        "reference": reference,
                        "mean": float(delta.mean()),
                        "q025": float(delta.quantile(0.025)),
                        "q975": float(delta.quantile(0.975)),
                        "fraction_negative": float((delta < 0).mean()),
                        "fraction_zero": float((delta == 0).mean()),
                        "truth_classes_mean": float(subset.truth_classes.mean())
                        if "truth_classes" in subset
                        else None,
                    }
                )
    pd.DataFrame(output).to_csv(directory / "resampling_summary.csv", index=False)
    paired = pd.read_csv(directory / "holdout_paired.csv")
    flips = []
    for kind, frame in paired.groupby("kind"):
        for candidate, reference in PAIRS:
            changed = frame[candidate] != frame[reference]
            fixed = (frame[candidate] == frame.truth) & (frame[reference] != frame.truth)
            broken = (frame[candidate] != frame.truth) & (frame[reference] == frame.truth)
            flips.append(
                {
                    "kind": kind,
                    "candidate": candidate,
                    "reference": reference,
                    "changed": int(changed.sum()),
                    "fixed": int(fixed.sum()),
                    "broken": int(broken.sum()),
                    "still_wrong": int((changed & ~fixed & ~broken).sum()),
                    "changed_unique_sessions": int(frame.loc[changed].row_id.nunique()),
                }
            )
    pd.DataFrame(flips).to_csv(directory / "holdout_flips.csv", index=False)

    # Descriptive difficulty segments; no gate is tuned from these tables.
    stats = pd.read_csv(directory / "train_features.csv")
    counts = stats.label.value_counts()
    joined = paired.merge(stats[["row_id", "events", "label"]], on="row_id", validate="many_to_one")
    joined["class_count"] = joined.label.map(counts)
    joined["size_bin"] = pd.cut(joined.events, [0, 100, 500, np.inf], include_lowest=True).astype(
        str
    )
    joined["support_bin"] = pd.cut(joined.class_count, [0, 7, 15, np.inf]).astype(str)
    segments = []
    for axis in ("size_bin", "support_bin"):
        for (kind, segment), frame in joined.groupby(["kind", axis]):
            for model in MODELS:
                segments.append(
                    {
                        "kind": kind,
                        "axis": axis,
                        "segment": segment,
                        "model": model,
                        "n": len(frame),
                        "errors": int((frame[model] != frame.truth).sum()),
                    }
                )
    pd.DataFrame(segments).to_csv(directory / "difficulty_segments.csv", index=False)
    groups = pd.read_csv(directory / "similarity_train.csv").group.to_numpy()
    integrity = []
    for stage, digest in PROTOCOLS.items():
        splits = json.loads(
            (Path("reports/results/phase4") / digest / f"{stage}_splits.json").read_text()
        )
        for fold, split in enumerate(splits):
            training_groups = set(groups[split["train"]])
            integrity.append(
                {
                    "stage": stage,
                    "fold": fold,
                    "validation_with_train_neighbor": sum(
                        int(group in training_groups) for group in groups[split["validation"]]
                    ),
                }
            )
    pd.DataFrame(integrity).to_csv(directory / "historical_group_crossings.csv", index=False)
    counts = stats.label.value_counts()
    class_browsers = stats.groupby("label").browser.nunique()
    write_json(
        directory / "data_support.json",
        {
            "classes": len(counts),
            "min_sessions_per_class": int(counts.min()),
            "median_sessions_per_class": float(counts.median()),
            "max_sessions_per_class": int(counts.max()),
            "classes_with_at_most_7_sessions": int((counts <= 7).sum()),
            "multi_browser_users": int((class_browsers > 1).sum()),
            "historical_neighbor_crossing_rows": int(
                pd.DataFrame(integrity).validation_with_train_neighbor.sum()
            ),
        },
    )
    write_json(
        directory / "analysis_manifest.json",
        {
            "source_hash": file_hash(__file__),
            "audit_manifest": file_hash(directory / "manifest.json"),
            "seed": 42,
            "source_permutations": 39,
            "conditional_resampling_not_external_confirmation": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    source_permutations(arguments.directory)
    vocabulary_transfer(arguments.directory)
    tables(arguments.directory)
