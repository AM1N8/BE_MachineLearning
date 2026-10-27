"""Aligned soft voting and OOF-only weight fitting."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from .model import check_probabilities


def metrics(y: Any, probabilities: Any) -> dict[str, float]:
    prediction = probabilities.argmax(axis=1)
    return {
        average: float(f1_score(y, prediction, average=average, zero_division=0))
        for average in ("macro", "micro", "weighted")
    }


def blend(probabilities: dict[str, Any], weights: dict[str, float]) -> Any:
    if set(probabilities) != set(weights):
        raise ValueError("Blend members and weight names must match exactly")
    if any(w < 0 for w in weights.values()) or not np.isclose(sum(weights.values()), 1):
        raise ValueError("Blend weights must be nonnegative and sum to one")
    shape = next(iter(probabilities.values())).shape
    for values in probabilities.values():
        check_probabilities(values, shape)
    result = sum(weights[name] * probabilities[name] for name in weights)
    check_probabilities(result, shape)
    return result


def optimize_weights(
    y: Any, probabilities: dict[str, Any], seed: int = 137
) -> dict[str, float]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    names = list(probabilities)
    if len(names) == 1:
        return {names[0]: 1.0}
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    for i in range(len(names)):
        study.enqueue_trial({f"w{j}": float(i == j) for j in range(len(names))})
    study.enqueue_trial({f"w{j}": 1.0 for j in range(len(names))})

    def objective(trial: Any) -> float:
        weights = np.array(
            [trial.suggest_float(f"w{j}", 0, 1) for j in range(len(names))]
        )
        weights /= max(weights.sum(), 1e-12)
        values = sum(w * probabilities[n] for w, n in zip(weights, names))
        return metrics(y, values)["macro"]

    study.optimize(objective, n_trials=80, timeout=40)
    weights = np.array([study.best_params[f"w{j}"] for j in range(len(names))])
    weights /= weights.sum()
    return dict(zip(names, weights.tolist()))
