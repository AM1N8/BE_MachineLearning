"""Training-only competitive search, paired validation, and inductive submission.

Run ``python -m copilote.competitive --help``. Existing report experiments are
preserved. The historical holdout is reused only as a post-selection benchmark,
not described as a new untouched test set.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC, LinearSVC
from threadpoolctl import threadpool_limits

from copilote.evaluate import scores
from copilote.features import build_features, identity
from copilote.io import Session, file_hash, read_sessions, write_json
from copilote.predict import validate_submission
from copilote.settings import load_config, provenance
from copilote.tokens import parse_token

LOGGER = logging.getLogger(__name__)
RESULTS = Path("reports/results/competitive")


class EventStream(TransformerMixin, BaseEstimator):
    """Extract a single row's events; no labels, vocabulary, or global state."""

    def __init__(self, view: str = "raw"):
        self.view = view

    def fit(self, X: Sequence[Session], y: Any = None) -> EventStream:
        """Leave all vocabulary learning to the subsequent fold-local vectorizer."""
        return self

    def transform(self, X: Sequence[Session]) -> list[tuple[str, ...]]:
        """Keep token boundaries and never concatenate different sessions."""
        result = []
        for row in X:
            if self.view == "browser":
                events = (row.browser,)
            else:
                events = tuple(t for t in row.tokens if parse_token(t).seconds is None)
                if self.view == "clean":
                    events = tuple(t[:-1] if parse_token(t).suffix_one else t for t in events)
                elif self.view == "dedup":
                    events = tuple(t for i, t in enumerate(events) if i == 0 or t != events[i - 1])
                elif self.view == "start":
                    events = events[:20]
                elif self.view == "context":
                    events = tuple(
                        f"{kind}={value}"
                        for raw in events
                        for kind, group in (
                            ("screen", parse_token(raw).screens),
                            ("config", parse_token(raw).configs),
                            ("chain", parse_token(raw).chains),
                        )
                        for value in group
                    )
                elif self.view == "style":
                    events = tuple(
                        parse_token(t).action + ("1" if parse_token(t).suffix_one else "")
                        for t in events
                    )
                elif self.view != "raw":
                    raise ValueError(f"Unknown view: {self.view}")
            result.append(events or ("__EMPTY__",))
        return result


class BrowserRestricted(ClassifierMixin, BaseEstimator):
    """Optional learned class/browser support mask; unseen browsers are unrestricted.

    This explicitly assumes deployment browser stability. The restriction is
    tested against its unrestricted counterpart and is never inferred from test.
    """

    def __init__(self, estimator: Any, restrict: bool = False, augment: float = 0, seed: int = 42):
        self.estimator = estimator
        self.restrict = restrict
        self.augment = augment
        self.seed = seed

    def fit(self, X: Sequence[Session], y: Any) -> BrowserRestricted:
        """Fit the complete pipeline and learn support from these training rows only."""
        training, labels = list(X), list(y)
        weights = [1.0] * len(training)
        if self.augment:
            rng = np.random.default_rng(self.seed)
            for row, label in zip(X, y, strict=True):
                events = tuple(t for t in row.tokens if parse_token(t).seconds is None)
                if len(events) < 30:
                    continue
                for fraction in (0.3, 0.6):
                    size = max(10, int(fraction * len(events)))
                    start = int(rng.integers(len(events) - size + 1))
                    training.append(
                        Session(row.row_id, row.browser, events[start : start + size], row.label)
                    )
                    labels.append(label)
                    weights.append(self.augment)
            self.estimator.fit(training, labels, model__sample_weight=np.asarray(weights))
        else:
            self.estimator.fit(training, labels)
        self.classes_ = self.estimator.classes_
        labels = np.asarray(y)
        self.browser_support_ = {
            browser: np.isin(self.classes_, labels[[row.browser == browser for row in X]])
            for browser in {row.browser for row in X}
        }
        return self

    def decision_function(self, X: Sequence[Session]) -> np.ndarray:
        """Return comparable margins, optionally masking incompatible classes."""
        margins = self.estimator.decision_function(X)
        if margins.ndim == 1:
            margins = np.column_stack((-margins, margins))
        if self.restrict:
            margins = margins.copy()
            for i, row in enumerate(X):
                support = self.browser_support_.get(row.browser)
                if support is not None:
                    margins[i, ~support] = -1e6
        return margins

    def predict(self, X: Sequence[Session]) -> np.ndarray:
        """Predict labels in the stable fitted class order."""
        return self.classes_[self.decision_function(X).argmax(axis=1)]


class NBSVM(ClassifierMixin, BaseEstimator):
    """One-vs-rest SVM with fold-local class-specific log-count-ratio weights."""

    def __init__(self, C: float = 1, alpha: float = 1, seed: int = 42):
        self.C = C
        self.alpha = alpha
        self.seed = seed

    def fit(self, X: Any, y: Any) -> NBSVM:
        """Learn supervised feature weights solely from this fold's training matrix."""
        labels = np.asarray(y)
        self.classes_ = np.unique(labels)
        total = np.asarray(X.sum(axis=0)).ravel()
        self.coef_ = np.empty((len(self.classes_), X.shape[1]))
        self.intercept_ = np.empty(len(self.classes_))
        for i, label in enumerate(self.classes_):
            positive = labels == label
            counts = np.asarray(X[positive].sum(axis=0)).ravel()
            p, q = counts + self.alpha, total - counts + self.alpha
            ratio = np.log((p / p.sum()) / (q / q.sum()))
            classifier = LinearSVC(
                C=self.C,
                class_weight="balanced",
                random_state=self.seed,
                max_iter=10000,
                dual="auto",
            )
            classifier.fit(X.multiply(ratio).tocsr(), positive)
            self.coef_[i] = classifier.coef_[0] * ratio
            self.intercept_[i] = classifier.intercept_[0]
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        """Apply the equivalent linear coefficients after supervised weighting."""
        return X @ self.coef_.T + self.intercept_

    def predict(self, X: Any) -> np.ndarray:
        """Return the highest-scoring class."""
        return self.classes_[self.decision_function(X).argmax(axis=1)]


class BrowserExperts(ClassifierMixin, BaseEstimator):
    """Learn separate within-browser vocabularies and class boundaries.

    A global fold-local model handles browsers absent from the training fold.
    Both components fit only training rows, without reading test browsers.
    """

    def __init__(self, spec: dict[str, Any], seed: int = 42):
        self.spec = spec
        self.seed = seed

    def fit(self, X: Sequence[Session], y: Any) -> BrowserExperts:
        """Fit global fallback and one inductive expert per observed browser."""
        spec = {key: value for key, value in self.spec.items() if key != "experts"}
        labels = np.asarray(y)
        self.global_ = build_competitive(spec, self.seed).fit(X, labels)
        self.classes_ = self.global_.classes_
        self.experts_ = {}
        for browser in sorted({row.browser for row in X}):
            indices = [i for i, row in enumerate(X) if row.browser == browser]
            if len(np.unique(labels[indices])) > 1:
                expert_spec = {**spec, "views": spec.get("views", ["raw"])}
                expert = build_competitive(expert_spec, self.seed).fit(
                    [X[i] for i in indices], labels[indices]
                )
            else:
                expert = labels[indices[0]]
            self.experts_[browser] = expert
        return self

    def decision_function(self, X: Sequence[Session]) -> np.ndarray:
        """Align expert margins to the global class order with explicit masking."""
        margins = np.full((len(X), len(self.classes_)), -1e6)
        for browser in {row.browser for row in X}:
            indices = [i for i, row in enumerate(X) if row.browser == browser]
            expert = self.experts_.get(browser, self.global_)
            if isinstance(expert, str):
                margins[indices, np.flatnonzero(self.classes_ == expert)[0]] = 0
            else:
                columns = np.searchsorted(self.classes_, expert.classes_)
                margins[np.ix_(indices, columns)] = expert.decision_function(
                    [X[i] for i in indices]
                )
        return margins

    def predict(self, X: Sequence[Session]) -> np.ndarray:
        """Select among classes supported by each browser expert."""
        return self.classes_[self.decision_function(X).argmax(axis=1)]


def build_competitive(spec: dict[str, Any], seed: int = 42) -> Any:
    """Build a fresh fold-local sparse pipeline from a JSON-serializable spec."""
    if spec.get("experts"):
        return BrowserExperts(spec, seed)
    blocks = []
    for view in spec.get("views", ["raw", "browser"]):
        if view == "numeric":
            blocks.append((view, build_features(("numeric",))))
            continue
        vectorizer = TfidfVectorizer(
            tokenizer=identity,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 1 if view == "browser" else spec.get("ngram_max", 2)),
            min_df=1 if view == "browser" else spec.get("min_df", 2),
            max_features=spec.get("max_features", 60000),
            sublinear_tf=spec.get("tf", "sublinear") == "sublinear",
            binary=spec.get("tf") == "binary",
            use_idf=spec.get("idf", True),
            dtype=np.float64,
        )
        blocks.append((view, Pipeline([("stream", EventStream(view)), ("tfidf", vectorizer)])))
    features = FeatureUnion(blocks, transformer_weights=spec.get("weights"))
    parameters = dict(C=spec.get("C", 1), class_weight=spec.get("class_weight"), random_state=seed)
    kernel = spec.get("kernel", "linear_svc")
    if kernel == "linear_svc":
        classifier = LinearSVC(
            **parameters, max_iter=10000, dual="auto", loss=spec.get("loss", "squared_hinge")
        )
    elif kernel == "logistic":
        classifier = OneVsRestClassifier(
            LogisticRegression(**parameters, solver="liblinear", max_iter=500)
        )
    elif kernel == "multinomial":
        classifier = LogisticRegression(**parameters, solver="lbfgs", max_iter=300, tol=1e-3)
    elif kernel == "ridge":
        classifier = RidgeClassifier(
            alpha=spec.get("alpha", 0.1), class_weight=spec.get("class_weight"), solver="lsqr"
        )
    elif kernel == "nbsvm":
        classifier = NBSVM(C=spec.get("C", 1), alpha=spec.get("alpha", 1), seed=seed)
    else:
        classifier = SVC(
            **parameters,
            kernel=kernel,
            gamma=spec.get("gamma", "scale"),
            degree=spec.get("degree", 2),
            coef0=spec.get("coef0", 0),
            cache_size=256,
            decision_function_shape="ovr",
            break_ties=True,
        )
    return BrowserRestricted(
        Pipeline([("features", features), ("model", classifier)]),
        restrict=spec.get("restrict_browser", False),
        augment=spec.get("augment", 0),
        seed=seed,
    )


def signature(config: dict[str, Any]) -> str:
    """Fingerprint predictive code, input data, dependencies, and protocol."""
    payload = {
        "train": file_hash(config["paths"]["train"]),
        "seed": config["seed"],
        "cv": config["cv"],
        "lock": file_hash("uv.lock"),
        "source": {
            name: file_hash(f"src/copilote/{name}.py")
            for name in ("tokens", "io", "eda", "features", "evaluate", "predict")
        },
        "competitive_source": "\n".join(
            inspect.getsource(obj)
            for obj in (
                EventStream,
                BrowserRestricted,
                NBSVM,
                BrowserExperts,
                build_competitive,
                evaluate,
            )
        ),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def evaluate(
    name: str,
    spec: dict[str, Any],
    rows: list[Session],
    splits: list[tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
    stage: str,
    fingerprint: str,
) -> dict[str, Any]:
    """Record paired fold metrics, original row IDs, errors, and class-aligned margins."""
    path = RESULTS / f"{stage}_{name}.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if cached["signature"] == fingerprint and cached["spec"] == spec:
            return cached
    y = np.array([str(row.label) for row in rows])
    classes = np.unique(y)
    folds, predictions, all_margins = [], [], []
    for fold, (training, validation) in enumerate(splits):
        if set(y[training]) != set(classes):
            raise ValueError("Every user must be represented in each training fold")
        model = build_competitive(spec, config["seed"])
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
            warnings.simplefilter("always")
            model.fit([rows[i] for i in training], y[training])
            margins = model.decision_function([rows[i] for i in validation])
        np.testing.assert_array_equal(classes, model.classes_)
        predicted = classes[margins.argmax(axis=1)]
        metrics = scores(y[validation], predicted, softmax(margins, axis=1), classes)
        metrics.update(fold=fold, seconds=time.perf_counter() - started, warning_count=len(caught))
        folds.append(metrics)
        rank = np.argsort(-margins, axis=1, kind="stable")
        predictions.extend(
            {
                "row_id": rows[i].row_id,
                "fold": fold,
                "truth": y[i],
                "prediction": pred,
                "second": classes[order[1]],
                "margin": float(value[order[0]] - value[order[1]]),
                "tokens": len(rows[i].tokens),
                "browser": rows[i].browser,
            }
            for i, pred, order, value in zip(validation, predicted, rank, margins, strict=True)
        )
        all_margins.append(margins.astype(np.float32))
        LOGGER.info(
            "%s %s fold %d accuracy %.4f macro-F1 %.4f (%.1fs)",
            stage,
            name,
            fold + 1,
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["seconds"],
        )
    result = {
        "signature": fingerprint,
        "spec": spec,
        "stage": stage,
        "classes": classes.tolist(),
        "folds": folds,
        "mean": {
            metric: float(np.mean([f[metric] for f in folds]))
            for metric in folds[0]
            if metric != "fold"
        },
        "std": {
            metric: float(np.std([f[metric] for f in folds], ddof=1))
            for metric in folds[0]
            if metric not in ("fold", "seconds")
        }
        if len(folds) > 1
        else {},
        "probability_kind": "uncalibrated_score_softmax",
    }
    pd.DataFrame(predictions).to_csv(path.with_suffix(".csv"), index=False)
    cache = Path("artifacts/competitive")
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache / f"{stage}_{name}.npz", margins=np.concatenate(all_margins))
    write_json(path, result)
    return result


def run(
    config: dict[str, Any], search: dict[str, Any], stage: str, names: list[str] | None = None
) -> dict[str, Any]:
    """Select only on development CV, then benchmark the frozen choice and refit."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = read_sessions(config["paths"]["train"], labelled=True)
    y = np.array([str(row.label) for row in rows])
    development, holdout = train_test_split(
        np.arange(len(rows)),
        test_size=config["cv"]["holdout_fraction"],
        stratify=y,
        random_state=config["seed"],
    )
    dev_rows = [rows[i] for i in development]
    splits = list(
        StratifiedKFold(3, shuffle=True, random_state=config["seed"]).split(
            development, y[development]
        )
    )
    fingerprint = signature(config)
    candidates = {
        name: {**search.get("defaults", {}), **spec} for name, spec in search["candidates"].items()
    }
    if stage == "search":
        requested = names or list(candidates)
        for name in requested:
            evaluate(name, candidates[name], dev_rows, splits, config, "development", fingerprint)
        results = {}
        for name, spec in candidates.items():
            path = RESULTS / f"development_{name}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if record["signature"] == fingerprint and record["spec"] == spec:
                    results[name] = record
        table = pd.DataFrame(
            [
                {
                    "name": name,
                    **{f"{key}_mean": value for key, value in result["mean"].items()},
                    **{f"{key}_std": value for key, value in result["std"].items()},
                }
                for name, result in results.items()
            ]
        ).sort_values(f"{config['cv']['selection_metric']}_mean", ascending=False)
        table.to_csv(RESULTS / "comparison.csv", index=False)
        LOGGER.info(
            "Development ranking:\n%s",
            table[["name", "accuracy_mean", "macro_f1_mean"]].to_string(index=False),
        )
        return results
    if stage == "select":
        results = {}
        for name, spec in candidates.items():
            path = RESULTS / f"development_{name}.json"
            if not path.exists():
                raise ValueError(f"Candidate {name} has not been evaluated; run search first")
            record = json.loads(path.read_text())
            if record["signature"] != fingerprint or record["spec"] != spec:
                raise ValueError(f"Stale candidate {name}; run search first")
            results[name] = record
        metric = config["cv"]["selection_metric"]
        best = max(results, key=lambda n: results[n]["mean"][metric])
        cutoff = results[best]["mean"][metric] - results[best]["std"][metric]
        eligible = [n for n in results if results[n]["mean"][metric] >= cutoff]
        chosen = min(
            eligible,
            key=lambda n: (candidates[n].get("complexity", 1), -results[n]["mean"][metric]),
        )
        selection = {
            "signature": fingerprint,
            "selected": chosen,
            "spec": candidates[chosen],
            "metric": metric,
            "official_metric_confirmed": False,
            "development": results[chosen],
            "best_development": best,
            "rule": "Simplest within one fold standard deviation; tied complexity uses development metric.",
            "selection_role": "Adaptive development-CV search; selection metrics are optimistic, not nested validation.",
            "holdout_role": "Historical holdout reused for a post-selection benchmark; not a fresh untouched test.",
            "transductive": False,
            "search_config": search,
            "split": {"development": development.tolist(), "holdout": holdout.tolist()},
        }
        write_json(RESULTS / "selection.json", selection)
        return selection
    selection = json.loads((RESULTS / "selection.json").read_text())
    if selection["signature"] != fingerprint or selection["search_config"] != search:
        raise ValueError("Selection is stale; rerun search and select")
    chosen, spec = selection["selected"], selection["spec"]
    if stage == "validate":
        repeated = list(
            RepeatedStratifiedKFold(
                n_splits=config["cv"]["folds"],
                n_repeats=config["cv"]["repeats"],
                random_state=config["seed"],
            ).split(rows, y)
        )
        baseline = candidates["baseline"]
        evaluated = {
            name: evaluate(name, candidate, rows, repeated, config, "repeated", fingerprint)
            for name, candidate in {"baseline": baseline, chosen: spec}.items()
        }
        holdout_result = evaluate(
            chosen, spec, rows, [(development, holdout)], config, "holdout", fingerprint
        )
        delta = [
            a["accuracy"] - b["accuracy"]
            for a, b in zip(evaluated[chosen]["folds"], evaluated["baseline"]["folds"], strict=True)
        ]
        result = {
            "signature": fingerprint,
            "selected": chosen,
            "repeated": evaluated,
            "holdout": holdout_result,
            "role": "Post-selection descriptive paired stability, not nested-CV generalization.",
            "paired_accuracy_gain_mean": float(np.mean(delta)),
            "paired_accuracy_gain_std": float(np.std(delta, ddof=1)),
            "folds_improved": int(np.sum(np.array(delta) > 0)),
        }
        write_json(RESULTS / "validation.json", result)
        return result
    validation = json.loads((RESULTS / "validation.json").read_text())
    if validation["signature"] != fingerprint or validation["selected"] != chosen:
        raise ValueError("Validation does not match selected model")
    test = read_sessions(config["paths"]["test"], labelled=False)
    model = build_competitive(spec, config["seed"])
    with threadpool_limits(limits=2):
        model.fit(rows, y)
        margins = model.decision_function(test)
    predicted = model.classes_[margins.argmax(axis=1)]
    settings = config["submission"]
    frame = pd.DataFrame(
        {
            settings["id_column"]: range(settings["id_base"], settings["id_base"] + len(test)),
            settings["target_column"]: predicted,
        }
    )
    validate_submission(frame, config, set(y), len(test))
    sample = pd.read_csv(config["paths"]["sample_submission"])
    if frame.columns.tolist() != sample.columns.tolist() or not frame.iloc[:, 0].equals(
        sample.iloc[:, 0]
    ):
        raise ValueError("Format disagrees with sample submission")
    destination = Path("submissions/competitive.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    validate_submission(pd.read_csv(destination), config, set(y), len(test))
    joblib.dump(model, "artifacts/competitive_final.joblib")
    rank = np.argsort(-margins, axis=1, kind="stable")
    pd.DataFrame(
        {
            "RowId": frame.iloc[:, 0],
            "prediction": predicted,
            "second": model.classes_[rank[:, 1]],
            "margin": margins[np.arange(len(test)), rank[:, 0]]
            - margins[np.arange(len(test)), rank[:, 1]],
            "tokens": [len(row.tokens) for row in test],
        }
    ).to_csv(RESULTS / "test_diagnostics.csv", index=False)
    metadata = {
        "provenance": provenance(config),
        "signature": fingerprint,
        "model": chosen,
        "spec": spec,
        "cv": validation["repeated"][chosen],
        "holdout": validation["holdout"],
        "rows": len(frame),
        "submission": str(destination),
        "submission_sha256": file_hash(destination),
        "validated": True,
        "transductive": False,
        "kaggle_score": None,
    }
    write_json(destination.with_suffix(".json"), metadata)
    write_json(RESULTS / "submission.json", metadata)
    return metadata


def main() -> None:
    """Expose resumable search, lock, validation, and prediction stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["search", "select", "validate", "predict", "all"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--search-config", default="configs/competitive.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--names", nargs="+")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = load_config(args.config, args.seed)
    search = yaml.safe_load(Path(args.search_config).read_text())
    for stage in (
        ["search", "select", "validate", "predict"] if args.stage == "all" else [args.stage]
    ):
        run(config, search, stage, args.names)


if __name__ == "__main__":
    # Persist estimators under their importable module, not the CLI's __main__.
    from copilote.competitive import main as entrypoint

    entrypoint()
