"""Audit generalization with frozen models, label-free shift and paired resampling.

Run ``static``, ``refit``, then ``summary``. New random splits reuse the same
observations: corrected intervals and clustered sensitivity analyses explicitly
retain that dependence. No stage changes a model or submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import chi2_contingency, ks_2samp, t
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from copilote.competitive import EventStream
from copilote.features import identity
from copilote.improvement import make_model, metrics
from copilote.io import file_hash, read_sessions, write_json
from copilote.phase2_models import BehavioralView, behavioral_model
from copilote.phase4_candidates import TempoView, hybrid_model
from copilote.phase4_challenger import PROTOCOLS
from copilote.phase4_forensics import decompose

MODELS = ("baseline", "signatures", "sig_burst_w05")
PAIRS = (("signatures", "baseline"), ("sig_burst_w05", "baseline"), ("sig_burst_w05", "signatures"))
N_RANDOM = 12
N_GROUP = 4
N_SIM = 2000
N_BOOT = 1000
SUBMISSIONS = ("best_verified", "context_signatures_experimental", "phase4_challenger")


def holm(pvalues: list[float]) -> np.ndarray:
    """Holm adjusted p-values in original order, including monotonicity."""
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1))
    result = np.empty(len(values))
    result[order] = np.minimum(adjusted, 1)
    return result


def macro_f1(truth: np.ndarray, prediction: np.ndarray, weights=None) -> float:
    """Match sklearn macro F1 with union-of-present-labels and zero_division=0."""
    size = int(max(truth.max(), prediction.max())) + 1
    actual = np.bincount(truth, weights=weights, minlength=size)
    predicted = np.bincount(prediction, weights=weights, minlength=size)
    correct = truth == prediction
    hits = np.bincount(
        truth[correct], weights=None if weights is None else weights[correct], minlength=size
    )
    denominator = actual + predicted
    present = denominator > 0
    return float((2 * hits[present] / denominator[present]).mean())


def corrected_contrast(deltas: np.ndarray, ratio: float) -> dict:
    """Corrected resampled t interval; not an independent-data confidence claim."""
    mean = float(deltas.mean())
    se = float(deltas.std(ddof=1) * np.sqrt(1 / len(deltas) + ratio))
    critical = float(t.ppf(0.975, len(deltas) - 1))
    pvalue = float(2 * t.sf(abs(mean / se), len(deltas) - 1)) if se else float(mean == 0)
    return {
        "delta": mean,
        "corrected_se": se,
        "lower95": mean - critical * se,
        "upper95": mean + critical * se,
        "pvalue": pvalue,
        "positive": int((deltas > 0).sum()),
        "negative": int((deltas < 0).sum()),
        "zero": int((deltas == 0).sum()),
    }


def setup() -> tuple[Path, dict]:
    """Content-address the entire audit, its frozen evidence and preservation targets."""
    sources = {str(path): file_hash(path) for path in sorted(Path("src/copilote").glob("*.py"))}
    inputs = {
        str(path): file_hash(path)
        for path in [
            Path("data/train.csv"),
            Path("data/test.csv"),
            Path("data/sample_submission.csv"),
            Path("evaluation.md"),
            Path("uv.lock"),
        ]
    }
    for stage, digest in PROTOCOLS.items():
        root = Path("reports/results/phase4") / digest
        for path in sorted(root.glob(f"{stage}_*.csv")):
            inputs[str(path)] = file_hash(path)
        path = root / f"{stage}_splits.json"
        inputs[str(path)] = file_hash(path)
    margin_path = Path("reports/results/phase4/c8a3d88dbf5c/test_margins.csv")
    inputs[str(margin_path)] = file_hash(margin_path)
    protected = {
        f"submissions/{name}.csv": file_hash(f"submissions/{name}.csv") for name in SUBMISSIONS
    }
    for path in sorted(Path("artifacts").glob("*.joblib")):
        protected[str(path)] = file_hash(path)
    manifest = {
        "sources": sources,
        "inputs": inputs,
        "protected": protected,
        "seed": 42,
        "models": list(MODELS),
        "random_holdouts": N_RANDOM,
        "group_holdouts": N_GROUP,
        "holdout_size": 324,
        "simulations": N_SIM,
        "cluster_bootstraps": N_BOOT,
        "similarity_threshold": 0.98,
        "length_ratio": [0.8, 1.25],
        "new_model_search": False,
        "test_labels_used": False,
        "automatic_submission": False,
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
    directory = Path("reports/results/phase5") / digest
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "manifest.json", manifest)
    return directory, manifest


def preserve(manifest: dict, directory: Path) -> None:
    """Assert submissions and saved models were not changed by the audit."""
    after = {path: file_hash(path) for path in manifest["protected"]}
    assert after == manifest["protected"]
    write_json(directory / "preservation.json", {"before": manifest["protected"], "after": after})


def submission_audit(directory: Path) -> None:
    """Join complete submissions against the saved full-fit decision evidence."""
    sample = pd.read_csv("data/sample_submission.csv")
    merged = sample[["RowId"]].copy()
    for name in SUBMISSIONS:
        frame = pd.read_csv(f"submissions/{name}.csv")
        assert frame.columns.tolist() == ["RowId", "prediction"]
        assert frame.RowId.tolist() == sample.RowId.tolist()
        assert frame.RowId.is_unique and not frame.isna().any().any()
        merged[name] = frame.prediction
    evidence = pd.read_csv("reports/results/phase4/c8a3d88dbf5c/test_margins.csv")
    for name, column in zip(SUBMISSIONS, ("best_verified", "context_signatures", "challenger")):
        np.testing.assert_array_equal(merged[name], evidence[column])
    changed = merged[list(SUBMISSIONS)].nunique(axis=1) > 1
    evidence.loc[changed].to_csv(directory / "submission_differences.csv", index=False)
    write_json(
        directory / "evaluation.json",
        {
            "source": "evaluation.md:14-15",
            "quote": "La métrique d'évaluation pour ce concours est le F1 score moyen.",
            "working_metric": "macro_f1_union_present_labels",
            "official_averaging_explicit": False,
            "public_subset_size_known": False,
            "official_absent_class_policy_known": False,
            "public_scores_user_reported": {
                "baseline": 0.980,
                "signatures": 0.983,
                "sig_burst_w05": 0.970,
            },
            "hidden_label_inference": False,
            "total_test_rows": len(merged),
            "changed_vs_baseline": int((merged.best_verified != merged.phase4_challenger).sum()),
            "changed_vs_signatures": int(
                (merged.context_signatures_experimental != merged.phase4_challenger).sum()
            ),
        },
    )


def old_predictions(y: np.ndarray, directory: Path) -> list[pd.DataFrame]:
    """Verify historical alignment and reconstruct each complete OOF repeat."""
    repeats, audit, validation_sets = [], [], []
    for stage, digest in PROTOCOLS.items():
        root = Path("reports/results/phase4") / digest
        splits = json.loads((root / f"{stage}_splits.json").read_text())
        for repeat in range(2):
            combined = []
            for fold in range(repeat * 4, repeat * 4 + 4):
                a, b = np.array(splits[fold]["train"]), np.array(splits[fold]["validation"])
                assert not set(a) & set(b)
                assert len(set(a) | set(b)) == len(y)
                assert set(y[b]) <= set(y[a])
                train_counts = pd.Series(y[a]).value_counts()
                audit.append(
                    {
                        "stage": stage,
                        "fold": fold,
                        "train_size": len(a),
                        "validation_size": len(b),
                        "validation_classes": len(set(y[b])),
                        "min_train_per_class": int(train_counts.min()),
                    }
                )
                validation_sets.append(set(b))
                frame = pd.DataFrame({"row_id": b, "truth": y[b]})
                for name in MODELS:
                    prior = pd.read_csv(root / f"{stage}_{name}_{fold}.csv")
                    assert prior.row_id.tolist() == b.tolist()
                    np.testing.assert_array_equal(prior.truth, y[b])
                    frame[name] = prior.prediction
                    frame[f"{name}_margin"] = prior.margin
                combined.append(frame)
            whole = pd.concat(combined).sort_values("row_id").reset_index(drop=True)
            assert whole.row_id.tolist() == list(range(len(y)))
            whole["repeat"] = len(repeats)
            repeats.append(whole)
    pd.DataFrame(audit).to_csv(directory / "historical_splits.csv", index=False)
    overlap = [len(a & b) / len(a | b) for a, b in combinations(validation_sets, 2)]
    write_json(
        directory / "historical_dependence.json",
        {
            "sessions": len(y),
            "oof_rows": len(y) * len(repeats),
            "predictions_per_session": 6,
            "folds": len(audit),
            "validation_jaccard_mean": float(np.mean(overlap)),
            "validation_jaccard_max": float(max(overlap)),
            "research_configurations": 19,
            "confirmation_used_for_selection": True,
            "seeds_create_new_observations": False,
        },
    )
    pd.concat(repeats).to_csv(directory / "historical_paired_oof.csv", index=False)
    return repeats


def vectorizer(**kwargs) -> TfidfVectorizer:
    """Token-list vectorizer; each fitting site explicitly defines its training rows."""
    return TfidfVectorizer(tokenizer=identity, token_pattern=None, lowercase=False, **kwargs)


def distribution_audit(rows, test, directory: Path) -> np.ndarray:
    """Compare source distributions and construct label-free similarity stress groups."""
    train_stats = pd.DataFrame(
        [{k: v for k, v in decompose(row).items() if k != "screen_seconds"} for row in rows]
    )
    test_stats = pd.DataFrame(
        [{k: v for k, v in decompose(row).items() if k != "screen_seconds"} for row in test]
    )
    train_stats.to_csv(directory / "train_features.csv", index=False)
    test_stats.to_csv(directory / "test_features.csv", index=False)
    records = []
    features = [
        "events",
        "duration",
        "events_per_second",
        "suffix_fraction",
        "dwell_entropy",
        "dwell_top_fraction",
        "events_per_active_tick",
        "gap_max",
    ]
    for feature in features:
        statistic, pvalue = ks_2samp(train_stats[feature], test_stats[feature])
        records.append(
            {
                "feature": feature,
                "ks": float(statistic),
                "pvalue": float(pvalue),
                "train_median": float(train_stats[feature].median()),
                "test_median": float(test_stats[feature].median()),
            }
        )
    adjusted = holm([record["pvalue"] for record in records])
    for record, pvalue in zip(records, adjusted, strict=True):
        record["holm_pvalue"] = float(pvalue)
    pd.DataFrame(records).to_csv(directory / "shift_scalars.csv", index=False)
    browsers = sorted({row.browser for row in rows + test})
    table = np.array(
        [
            [sum(row.browser == browser for row in data) for browser in browsers]
            for data in (rows, test)
        ]
    )
    statistic, pvalue, _, _ = chi2_contingency(table)
    write_json(
        directory / "shift_browser.json",
        {
            "browsers": browsers,
            "counts": table.tolist(),
            "chi2": float(statistic),
            "pvalue": float(pvalue),
        },
    )

    documents, oov = {}, []
    views = {
        "raw": EventStream("raw"),
        "signatures": BehavioralView("signatures"),
        "burst": TempoView(("burst",)),
    }
    for name, view in views.items():
        training, testing = view.transform(rows), view.transform(test)
        vocabulary = {token for doc in training for token in doc}
        counts = [
            sum(token not in vocabulary for token in doc) / max(len(doc), 1) for doc in testing
        ]
        oov.append(
            {
                "view": name,
                "train_types": len(vocabulary),
                "test_oov_session_mean": float(np.mean(counts)),
                "test_oov_occurrence_fraction": float(
                    sum(sum(token not in vocabulary for token in doc) for doc in testing)
                    / sum(map(len, testing))
                ),
            }
        )
        documents[name] = training + testing
    pd.DataFrame(oov).to_csv(directory / "shift_oov.csv", index=False)
    source_y = np.r_[np.zeros(len(rows), dtype=int), np.ones(len(test), dtype=int)]
    aucs = []
    # Fold-local vocabulary and IDF, source labels only, never hidden user labels.
    combined = [
        [f"{name}:{token}" for name in views for token in documents[name][i]]
        + [f"browser:{row.browser}"]
        for i, row in enumerate(rows + test)
    ]
    source_scores = np.zeros(len(combined))
    for fold, (a, b) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=42).split(source_y, source_y)
    ):
        model = make_pipeline(
            vectorizer(min_df=2, max_features=60000, sublinear_tf=True),
            LogisticRegression(C=1, class_weight="balanced", max_iter=1000, random_state=42),
        )
        with threadpool_limits(limits=2):
            model.fit([combined[i] for i in a], source_y[a])
            score = model.predict_proba([combined[i] for i in b])[:, 1]
        source_scores[b] = score
        aucs.append({"fold": fold, "auc": float(roc_auc_score(source_y[b], score))})
    pd.DataFrame({"source": source_y, "score": source_scores}).to_csv(
        directory / "source_oof.csv", index=False
    )
    write_json(
        directory / "adversarial.json",
        {
            "folds": aucs,
            "pooled_auc": float(roc_auc_score(source_y, source_scores)),
            "label_free_diagnostic_only": True,
        },
    )

    # Binary cosine, learned on train only; same browser and comparable event length.
    encoder = vectorizer(binary=True, use_idf=False)
    matrix = encoder.fit_transform(documents["raw"][: len(rows)])
    other = encoder.transform(documents["raw"][len(rows) :])
    similarity = (matrix @ matrix.T).toarray()
    np.fill_diagonal(similarity, 0)
    lengths = train_stats.events.to_numpy()
    ratio = lengths[:, None] / np.maximum(lengths[None, :], 1)
    browser = np.array([row.browser for row in rows])
    eligible = (ratio >= 0.8) & (ratio <= 1.25) & (browser[:, None] == browser[None, :])
    similarity *= eligible
    _, groups = connected_components(sparse.csr_matrix(similarity >= 0.98), directed=False)
    nearest = similarity.argmax(axis=1)
    cross = (other @ matrix.T).toarray()
    ratio_test = test_stats.events.to_numpy()[:, None] / np.maximum(lengths[None, :], 1)
    cross *= (
        (ratio_test >= 0.8)
        & (ratio_test <= 1.25)
        & (np.array([row.browser for row in test])[:, None] == browser[None, :])
    )
    pd.DataFrame(
        {
            "row_id": range(len(rows)),
            "group": groups,
            "nearest": nearest,
            "cosine": similarity.max(axis=1),
            "nearest_same_user": [rows[i].label == rows[j].label for i, j in enumerate(nearest)],
        }
    ).to_csv(directory / "similarity_train.csv", index=False)
    pd.DataFrame(
        {
            "RowId": range(1, len(test) + 1),
            "nearest": cross.argmax(axis=1),
            "cosine": cross.max(axis=1),
        }
    ).to_csv(directory / "similarity_test.csv", index=False)
    counts = np.bincount(groups)
    write_json(
        directory / "similarity.json",
        {
            "components": len(counts),
            "nontrivial_components": int((counts > 1).sum()),
            "max_component": int(counts.max()),
            "sessions_in_nontrivial_components": int(counts[counts > 1].sum()),
            "train_nearest_median": float(np.median(similarity.max(axis=1))),
            "test_nearest_median": float(np.median(cross.max(axis=1))),
            "similarity_is_not_acquisition_metadata": True,
        },
    )
    return groups


def make_splits(y: np.ndarray, groups: np.ndarray) -> list[dict]:
    """Freeze random and group stress splits without examining model outcomes."""
    output = []
    for a, b in StratifiedShuffleSplit(N_RANDOM, test_size=324, random_state=42).split(y, y):
        output.append({"kind": "random324", "train": a.tolist(), "validation": b.tolist()})
    # Group partition sizes vary; reject only unseen-user tasks, not bad scores.
    for a, b in GroupShuffleSplit(1000, test_size=324 / len(y), random_state=42).split(
        y, y, groups
    ):
        if set(y[b]) <= set(y[a]):
            assert not set(groups[a]) & set(groups[b])
            output.append(
                {"kind": "similarity_group", "train": a.tolist(), "validation": b.tolist()}
            )
        if len(output) == N_RANDOM + N_GROUP:
            break
    assert len(output) == N_RANDOM + N_GROUP
    return output


def simulations(repeats: list[pd.DataFrame], directory: Path) -> None:
    """Conditional paired finite-sample sensitivity, not Kaggle significance tests."""
    classes, truth = np.unique(repeats[0].truth, return_inverse=True)
    predictions = [
        {name: np.searchsorted(classes, frame[name]) for name in MODELS} for frame in repeats
    ]
    rng = np.random.default_rng(42)
    per_class = [np.flatnonzero(truth == label) for label in range(len(classes))]
    records = []
    for scenario, size in (("uniform324", 324), ("coverage324", 324), ("uniform162", 162)):
        for draw in range(N_SIM):
            repeat = int(rng.integers(len(repeats)))
            if scenario == "coverage324":
                one = np.array([rng.choice(indices) for indices in per_class])
                remaining = np.setdiff1d(np.arange(len(truth)), one)
                indices = np.r_[one, rng.choice(remaining, size - len(one), replace=False)]
            else:
                indices = rng.choice(len(truth), size, replace=False)
            result = {
                "scenario": scenario,
                "draw": draw,
                "repeat": repeat,
                "truth_classes": len(np.unique(truth[indices])),
            }
            result.update(
                {
                    name: macro_f1(truth[indices], pred[indices])
                    for name, pred in predictions[repeat].items()
                }
            )
            records.append(result)
    pd.DataFrame(records).to_csv(directory / "small_sample_simulations.csv", index=False)


def refit(rows, splits: list[dict], directory: Path) -> None:
    """Evaluate only the three frozen models; persist each fold for resumability."""
    y = np.array([row.label for row in rows])
    for fold, split in enumerate(splits):
        a, b = np.array(split["train"]), np.array(split["validation"])
        assert set(y[b]) <= set(y[a])
        training, validation = [rows[i] for i in a], [rows[i] for i in b]
        for name in MODELS:
            path = directory / f"holdout_{fold}_{name}.csv"
            if path.exists():
                prior = pd.read_csv(path)
                assert prior.row_id.tolist() == b.tolist()
                np.testing.assert_array_equal(prior.truth, y[b])
                continue
            model = {
                "baseline": lambda: make_model("baseline", y[a]),
                "signatures": lambda: behavioral_model("signatures"),
                "sig_burst_w05": lambda: hybrid_model(signatures_weight=0.5),
            }[name]()
            with threadpool_limits(limits=2):
                model.fit(training, y[a])
                margins = model.decision_function(validation)
            np.testing.assert_array_equal(model.classes_, np.unique(y))
            order = np.argsort(-margins, axis=1, kind="stable")
            prediction = model.classes_[order[:, 0]]
            pd.DataFrame(
                {
                    "row_id": b,
                    "truth": y[b],
                    "prediction": prediction,
                    "margin": margins[np.arange(len(b)), order[:, 0]]
                    - margins[np.arange(len(b)), order[:, 1]],
                    "second": model.classes_[order[:, 1]],
                }
            ).to_csv(path, index=False)
            print(
                {"fold": fold, "kind": split["kind"], "model": name, **metrics(y[b], prediction)},
                flush=True,
            )


def summarize(directory: Path, splits: list[dict], rows) -> None:
    """Correct inference for overlapping splits and bootstrap by session/user."""
    fold_records, contrasts, paired = [], [], []
    for fold, split in enumerate(splits):
        frame = pd.read_csv(directory / f"holdout_{fold}_baseline.csv")[["row_id", "truth"]]
        frame["fold"], frame["kind"] = fold, split["kind"]
        for name in MODELS:
            predictions = pd.read_csv(directory / f"holdout_{fold}_{name}.csv")
            assert predictions.row_id.tolist() == split["validation"]
            np.testing.assert_array_equal(predictions.truth, frame.truth)
            frame[name] = predictions.prediction
            frame[f"{name}_margin"] = predictions.margin
            fold_records.append(
                {
                    "fold": fold,
                    "kind": split["kind"],
                    "model": name,
                    "n": len(frame),
                    "classes": frame.truth.nunique(),
                    **metrics(frame.truth, frame[name]),
                }
            )
        paired.append(frame)
    scores = pd.DataFrame(fold_records)
    scores.to_csv(directory / "holdout_metrics.csv", index=False)
    joined = pd.concat(paired, ignore_index=True)
    joined.to_csv(directory / "holdout_paired.csv", index=False)
    for kind in ("random324", "similarity_group"):
        pivot = scores.loc[scores.kind == kind].pivot(
            index="fold", columns="model", values="macro_f1"
        )
        ratio = float(
            np.mean([len(s["validation"]) / len(s["train"]) for s in splits if s["kind"] == kind])
        )
        current = []
        for candidate, reference in PAIRS:
            delta = (pivot[candidate] - pivot[reference]).to_numpy()
            current.append(
                {
                    "kind": kind,
                    "candidate": candidate,
                    "reference": reference,
                    "folds": len(delta),
                    **corrected_contrast(delta, ratio),
                }
            )
        adjusted = holm([record["pvalue"] for record in current])
        for record, pvalue in zip(current, adjusted, strict=True):
            record["holm_pvalue"] = float(pvalue)
            record["passes"] = record["lower95"] > 0 and pvalue < 0.05
        contrasts.extend(current)
    pd.DataFrame(contrasts).to_csv(directory / "corrected_contrasts.csv", index=False)

    # Reweight ALL predictions of the same session together; user bootstrap is a
    # stronger dependence sensitivity, not an estimate of performance on unseen users.
    ordinary = joined.loc[joined.kind == "random324"].copy()
    classes = np.unique([row.label for row in rows])
    truth = np.searchsorted(classes, ordinary.truth)
    predictions = {name: np.searchsorted(classes, ordinary[name]) for name in MODELS}
    ids = ordinary.row_id.to_numpy()
    row_users = np.searchsorted(classes, [row.label for row in rows])
    boot = []
    rng = np.random.default_rng(42)
    for unit in ("session", "user"):
        n = len(rows) if unit == "session" else len(classes)
        units = ids if unit == "session" else row_users[ids]
        for draw in range(N_BOOT):
            weights = rng.multinomial(n, np.full(n, 1 / n))[units]
            values = {
                name: macro_f1(truth, prediction, weights)
                for name, prediction in predictions.items()
            }
            boot.append({"unit": unit, "draw": draw, **values})
    pd.DataFrame(boot).to_csv(directory / "cluster_bootstrap.csv", index=False)
    sets = [set(split["validation"]) for split in splits[:N_RANDOM]]
    counts = ordinary.row_id.value_counts()
    write_json(
        directory / "new_split_dependence.json",
        {
            "random_holdouts": N_RANDOM,
            "total_validation_rows": len(ordinary),
            "unique_validated_sessions": len(counts),
            "min_repetitions_observed": int(counts.min()),
            "max_repetitions": int(counts.max()),
            "unvalidated_sessions": len(rows) - len(counts),
            "mean_validation_intersection": float(
                np.mean([len(a & b) for a, b in combinations(sets, 2)])
            ),
            "new_independent_observations": 0,
        },
    )


def reuse_refits(source: Path, directory: Path, manifest: dict, splits: list[dict]) -> None:
    """Reuse predictions only for identical frozen dependencies, inputs and splits."""
    prior = json.loads((source / "manifest.json").read_text())
    assert prior["inputs"] == manifest["inputs"]
    assert prior["protected"] == manifest["protected"]
    for path, digest in prior["sources"].items():
        if Path(path).name not in ("phase5_audit.py", "phase5_analysis.py"):
            assert manifest["sources"][path] == digest
    assert prior["models"] == manifest["models"]
    assert json.loads((source / "splits.json").read_text()) == splits
    for fold in range(len(splits)):
        for name in MODELS:
            frame = pd.read_csv(source / f"holdout_{fold}_{name}.csv")
            assert frame.row_id.tolist() == splits[fold]["validation"]
            frame.to_csv(directory / f"holdout_{fold}_{name}.csv", index=False)
    write_json(
        directory / "refit_reuse.json",
        {
            "source": str(source),
            "identical_model_dependencies": True,
            "identical_inputs_and_splits": True,
            "new_fits_in_this_directory": 0,
        },
    )


def run(stage: str, reuse: Path | None = None) -> None:
    """Execute a signed stage; refuse summary before all frozen refits exist."""
    directory, manifest = setup()
    print({"directory": str(directory), "stage": stage}, flush=True)
    rows = read_sessions("data/train.csv", labelled=True)
    y = np.array([row.label for row in rows])
    if stage == "static":
        submission_audit(directory)
        repeats = old_predictions(y, directory)
        simulations(repeats, directory)
        groups = distribution_audit(rows, read_sessions("data/test.csv", labelled=False), directory)
        splits = make_splits(y, groups)
        write_json(directory / "splits.json", splits)
        write_json(
            directory / "preregistration.json",
            {
                "frozen_before_refitting": True,
                "comparisons": [list(pair) for pair in PAIRS],
                "gate": "corrected lower95 > 0 and Holm p < .05; no leaderboard tuning",
                "scope": "new splits of already reused data, not untouched external confirmation",
                "group_stress": "label-free binary cosine >= .98; closed-set only",
                "search": False,
            },
        )
    else:
        splits = json.loads((directory / "splits.json").read_text())
        if stage == "refit":
            if reuse is not None:
                reuse_refits(reuse, directory, manifest, splits)
            refit(rows, splits, directory)
        else:
            summarize(directory, splits, rows)
    preserve(manifest, directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("static", "refit", "summary"))
    parser.add_argument("--reuse-refits", type=Path)
    arguments = parser.parse_args()
    run(arguments.stage, arguments.reuse_refits)
