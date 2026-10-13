"""Development-only selection, locked holdout, and post-selection repeated CV."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from copilote.io import file_hash, read_sessions, write_json
from copilote.models import build_model, catalogue
from copilote.settings import provenance

LOGGER = logging.getLogger(__name__)
METRICS = ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "top3", "top5", "log_loss")


def probabilities(estimator: Any, X: Any) -> tuple[np.ndarray, str]:
    """Expose native probabilities or explicitly uncalibrated score-softmax proxies."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X), "native"
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        if scores.ndim == 1:
            scores = np.column_stack((-scores, scores))
        return softmax(scores, axis=1), "uncalibrated_score_softmax"
    # Hard votes have no probabilities: report empirical vote shares, not calibration.
    votes = np.array([model.predict(X) for model in estimator.estimators_])
    shares = np.stack([(votes == i).mean(axis=0) for i in range(len(estimator.classes_))], axis=1)
    return shares, "hard_vote_shares"


def scores(
    y: np.ndarray, predicted: np.ndarray, proba: np.ndarray, classes: np.ndarray
) -> dict[str, float]:
    """Compute multiclass metrics with an explicit, stable probability-column order."""
    rank = np.argsort(-proba, axis=1, kind="stable")
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, predicted, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "top3": float(np.mean(np.any(classes[rank[:, :3]] == y[:, None], axis=1))),
        "top5": float(np.mean(np.any(classes[rank[:, :5]] == y[:, None], axis=1))),
        "log_loss": float(log_loss(y, proba, labels=classes)),
    }


def cache_signature(config: dict[str, Any]) -> str:
    """Hash effective inputs and predictive source, independent of unrelated report edits."""
    contents = {
        "config": config,
        "data": file_hash(config["paths"]["train"]),
        "lock": file_hash("uv.lock"),
        "source": {
            p: file_hash(f"src/copilote/{p}.py")
            for p in ("io", "tokens", "eda", "features", "models", "evaluate")
        },
    }
    return hashlib.sha256(json.dumps(contents, sort_keys=True).encode()).hexdigest()


def evaluate_candidate(
    name: str,
    spec: dict[str, Any],
    rows: list[Any],
    y: np.ndarray,
    splits: list[Any],
    config: dict[str, Any],
    stage: str,
    signature: str,
    force: bool = False,
) -> dict[str, Any]:
    """Fit a new entire pipeline for each fold and cache only matching experiments."""
    path = Path(f"reports/results/experiments/{stage}_{name}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        existing = json.loads(path.read_text())
        if existing.get("signature") == signature and existing.get("spec") == spec:
            return existing
    folds = []
    oof_truth, oof_predicted, oof_probabilities, oof_rows = [], [], [], []
    for number, (training, validation) in enumerate(splits):
        model = build_model(spec, config)
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
            warnings.simplefilter("always")
            model.fit([rows[i] for i in training], y[training])
            fit_seconds = time.perf_counter() - start
            valid = [rows[i] for i in validation]
            predicted = model.predict(valid)
            proba, kind = probabilities(model, valid)
            result = scores(y[validation], predicted, proba, model.classes_)
            classifier = model.named_steps["model"] if hasattr(model, "named_steps") else model
            if hasattr(classifier, "coef_"):
                result["nonzero_coefficients"] = int(np.count_nonzero(classifier.coef_))
                result["total_coefficients"] = int(classifier.coef_.size)
        oof_truth.extend(y[validation])
        oof_predicted.extend(predicted)
        oof_probabilities.append(proba)
        oof_rows.extend(
            {"row_id": rows[i].row_id, "fold": number, "truth": y[i], "prediction": p}
            for i, p in zip(validation, predicted, strict=True)
        )
        result.update(
            fold=number,
            fit_seconds=fit_seconds,
            total_seconds=time.perf_counter() - start,
            probability_kind=kind,
            warnings=sorted({str(w.message) for w in caught}),
        )
        folds.append(result)
        LOGGER.info(
            "%s %s fold %d/%d macro-F1 %.4f (%.1fs)",
            stage,
            name,
            number + 1,
            len(splits),
            result["macro_f1"],
            result["total_seconds"],
        )
    pd.DataFrame(oof_rows).to_csv(path.with_suffix(".csv"), index=False)
    result = {
        "signature": signature,
        "spec": spec,
        "stage": stage,
        "folds": folds,
        "pooled_oof": scores(
            np.array(oof_truth),
            np.array(oof_predicted),
            np.concatenate(oof_probabilities),
            model.classes_,
        ),
        "pooled_class_report": classification_report(
            oof_truth, oof_predicted, output_dict=True, zero_division=0
        ),
        "mean": {
            metric: float(np.mean([f[metric] for f in folds]))
            for metric in (*METRICS, "fit_seconds")
        },
        "std": {metric: float(np.std([f[metric] for f in folds], ddof=1)) for metric in METRICS},
    }
    write_json(path, result)
    return result


def run_training(
    config: dict[str, Any], *, force: bool = False, fast: bool = False
) -> dict[str, Any]:
    """Select on development CV, lock the choice, evaluate holdout then stability CV."""
    rows = read_sessions(config["paths"]["train"], labelled=True)
    y = np.array([str(r.label) for r in rows])
    development, holdout = train_test_split(
        np.arange(len(rows)),
        test_size=config["cv"]["holdout_fraction"],
        random_state=config["seed"],
        stratify=y,
    )
    dev_rows, dev_y = [rows[i] for i in development], y[development]
    dev_splits = list(
        StratifiedKFold(n_splits=3, shuffle=True, random_state=config["seed"]).split(
            development, dev_y
        )
    )
    signature = cache_signature(config)
    candidates = catalogue()
    if fast:
        candidates = {
            k: candidates[k] for k in ("majority", "browser", "linear", "linear_balanced")
        }
    stage = "fast" if fast else "development"
    write_json(
        "reports/results/splits.json",
        {
            "seed": config["seed"],
            "development": development.tolist(),
            "holdout": holdout.tolist(),
            "development_folds": [
                {"train": development[a].tolist(), "validation": development[b].tolist()}
                for a, b in dev_splits
            ],
        },
    )
    evaluations = {
        name: evaluate_candidate(
            name, spec, dev_rows, dev_y, dev_splits, config, stage, signature, force
        )
        for name, spec in candidates.items()
    }
    metric = config["cv"]["selection_metric"]
    singles = {
        name: result
        for name, result in evaluations.items()
        if "vote" not in name and name != "stacking"
    }
    best = max(singles, key=lambda name: singles[name]["mean"][metric])
    cutoff = singles[best]["mean"][metric] - singles[best]["std"][metric]
    eligible = [name for name, value in singles.items() if value["mean"][metric] >= cutoff]
    chosen = min(
        eligible,
        key=lambda name: (candidates[name].get("complexity", 4), -singles[name]["mean"][metric]),
    )
    for name in evaluations:
        if ("vote" in name or name == "stacking") and evaluations[name]["mean"][
            metric
        ] > evaluations[chosen]["mean"][metric] + evaluations[chosen]["std"][metric]:
            chosen = name
    selection = {
        "provenance": provenance(config),
        "signature": signature,
        "selected": chosen,
        "spec": candidates[chosen],
        "best_development_mean": best,
        "one_std_cutoff": cutoff,
        "selection_rule": "Simplest predeclared complexity within one std of best single; vote requires gain > selected std. Development 3-fold only.",
        "development": evaluations,
        "fast_only": fast,
    }
    write_json(f"reports/results/{'fast' if fast else 'selection'}.json", selection)
    if fast:
        return selection
    # Freeze the choice before any holdout or full-data CV score is observed.
    locked_model = build_model(candidates[chosen], config)
    with threadpool_limits(limits=2):
        locked_model.fit(dev_rows, dev_y)
        valid = [rows[i] for i in holdout]
        prediction = locked_model.predict(valid)
        proba, kind = probabilities(locked_model, valid)
    joblib.dump(
        {
            "model": locked_model,
            "development": development,
            "holdout": holdout,
            "signature": signature,
        },
        "artifacts/holdout_model.joblib",
    )
    holdout_result = {
        "selected": chosen,
        "signature": signature,
        "metrics": scores(y[holdout], prediction, proba, locked_model.classes_),
        "probability_kind": kind,
        "classification_report": classification_report(
            y[holdout], prediction, output_dict=True, zero_division=0
        ),
    }
    # Entire pipeline is refit within each calibration fold; ensemble=False yields
    # one final development fit and cross-fitted sigmoid calibration.
    calibrated = CalibratedClassifierCV(
        build_model(candidates[chosen], config),
        method="sigmoid",
        cv=StratifiedKFold(3, shuffle=True, random_state=config["seed"]),
        ensemble=False,
    )
    with threadpool_limits(limits=2):
        calibrated.fit(dev_rows, dev_y)
        calibrated_proba = calibrated.predict_proba(valid)
    holdout_result["calibrated_metrics"] = scores(
        y[holdout],
        calibrated.classes_[calibrated_proba.argmax(axis=1)],
        calibrated_proba,
        calibrated.classes_,
    )
    joblib.dump(calibrated, "artifacts/holdout_calibrated.joblib")
    write_json("reports/results/holdout.json", holdout_result)
    pd.DataFrame(
        {
            "row_id": holdout,
            "truth": y[holdout],
            "prediction": prediction,
            "confidence": calibrated_proba.max(axis=1),
            "correct": prediction == y[holdout],
            "browser": [r.browser for r in valid],
            "tokens": [len(r.tokens) for r in valid],
        }
    ).to_csv("reports/results/holdout_predictions.csv", index=False)
    repeated = list(
        RepeatedStratifiedKFold(
            n_splits=config["cv"]["folds"],
            n_repeats=config["cv"]["repeats"],
            random_state=config["seed"],
        ).split(rows, y)
    )
    write_json(
        "reports/results/repeated_splits.json",
        [{"train": a.tolist(), "validation": b.tolist()} for a, b in repeated],
    )
    representative = list(catalogue())[:23] + ["linear_unigrams", "linear_balanced"]
    representative += [chosen] if chosen not in representative else []
    repeated_results = {
        name: evaluate_candidate(
            name, candidates[name], rows, y, repeated, config, "repeated", signature, force
        )
        for name in representative
    }
    summary = [
        {
            "name": name,
            "parameters": json.dumps(candidates[name], sort_keys=True),
            **{f"{m}_mean": result["mean"][m] for m in METRICS},
            **{f"{m}_std": result["std"][m] for m in METRICS},
            "fit_seconds_mean": result["mean"]["fit_seconds"],
        }
        for name, result in repeated_results.items()
    ]
    pd.DataFrame(summary).to_csv("reports/results/model_comparison.csv", index=False)
    write_json(
        "reports/results/evaluation.json",
        {
            "provenance": provenance(config),
            "signature": signature,
            "selected": chosen,
            "repeated_cv_role": "Descriptive stability after development selection; not unbiased nested model-selection performance.",
            "repeated": repeated_results,
            "holdout": holdout_result,
        },
    )
    learning = []
    for fraction in (0.35, 0.65, 1.0):
        rng = np.random.default_rng(config["seed"])
        indices = np.concatenate(
            [
                rng.permutation(np.flatnonzero(dev_y == label))[
                    : max(2, int(np.sum(dev_y == label) * fraction))
                ]
                for label in np.unique(dev_y)
            ]
        )
        model = build_model(candidates[chosen], config)
        with threadpool_limits(limits=2):
            model.fit([dev_rows[i] for i in indices], dev_y[indices])
            train_pred = model.predict([dev_rows[i] for i in indices])
            valid_pred = model.predict(valid)
        learning.append(
            {
                "fraction": fraction,
                "train_sessions": len(indices),
                "train_macro_f1": float(f1_score(dev_y[indices], train_pred, average="macro")),
                "holdout_macro_f1": float(f1_score(y[holdout], valid_pred, average="macro")),
            }
        )
    write_json(
        "reports/results/learning_curve.json",
        {"role": "Post-lock diagnostic; never used to change selected model", "points": learning},
    )
    return selection
