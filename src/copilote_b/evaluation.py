"""Fold-local repeated and nested validation for the frozen three-member shortlist."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from .config import write_json
from .ensemble import blend, metrics, optimize_weights
from .features import Trace
from .model import CropSVM, check_probabilities


def splits(y: Any, seed: int) -> Any:
    return StratifiedKFold(5, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y
    )


def fit_predict(
    config: dict[str, Any],
    traces: list[Trace],
    y: Any,
    train: Any,
    valid: Any,
    seed: int,
) -> Any:
    model = CropSVM(config, seed).fit([traces[i] for i in train], y[train])
    if not np.array_equal(model.classes_, np.arange(len(np.unique(y)))):
        raise ValueError("Every fitting partition must contain every class")
    values = model.predict_proba([traces[i] for i in valid])
    del model
    gc.collect()
    return values


def error_analysis(
    folder: Path, name: str, y: Any, probabilities: Any, classes: Any
) -> None:
    prediction = probabilities.argmax(axis=1)
    report = classification_report(
        y,
        prediction,
        labels=np.arange(len(classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame({c: report[c] for c in classes}).T.to_csv(
        folder / f"{name}_per_class.csv", index_label="user"
    )
    matrix = confusion_matrix(y, prediction, labels=np.arange(len(classes)))
    pairs = [
        dict(true=classes[a], predicted=classes[b], count=int(matrix[a, b]))
        for a, b in zip(*np.nonzero(matrix))
        if a != b
    ]
    pd.DataFrame(pairs, columns=["true", "predicted", "count"]).sort_values(
        "count", ascending=False
    ).to_csv(folder / f"{name}_confusions.csv", index=False)
    wrong = np.flatnonzero(y != prediction)
    pd.DataFrame(
        {
            "train_RowId": wrong + 1,
            "true": classes[y[wrong]],
            "predicted": classes[prediction[wrong]],
        }
    ).to_csv(folder / f"{name}_errors.csv", index=False)


def cross_validate(
    traces: list[Trace],
    y: Any,
    classes: Any,
    config: dict[str, Any],
    output: Path,
    repeats: int = 2,
    fold_limit: int = 5,
) -> pd.DataFrame:
    folder = output / "cv"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    with threadpool_limits(limits=1):
        for repeat, seed in enumerate(config["cv_seeds"][:repeats]):
            for fold, (train, valid) in enumerate(splits(y, seed)):
                if fold >= fold_limit:
                    break
                for name, parameters in config["members"].items():
                    path = folder / f"{name}_r{repeat}_f{fold}.npz"
                    if path.exists():
                        with np.load(path) as cache:
                            np.testing.assert_array_equal(cache["train"], train)
                            np.testing.assert_array_equal(cache["valid"], valid)
                            np.testing.assert_array_equal(cache["y"], y[valid])
                            check_probabilities(
                                cache["probabilities"], (len(valid), len(classes))
                            )
                        record = json.loads(path.with_suffix(".json").read_text())
                    else:
                        start = time.monotonic()
                        values = fit_predict(
                            parameters, traces, y, train, valid, seed + fold
                        )
                        record = dict(
                            model=name,
                            repeat=repeat,
                            fold=fold,
                            seconds=time.monotonic() - start,
                            **metrics(y[valid], values),
                        )
                        np.savez_compressed(
                            path,
                            train=train,
                            valid=valid,
                            y=y[valid],
                            probabilities=values.astype(np.float32),
                        )
                        write_json(path.with_suffix(".json"), record)
                    records.append(record)
                    print(json.dumps(record), flush=True)
    frame = pd.DataFrame(records)
    table = frame.groupby("model").agg(
        folds=("macro", "size"),
        macro_mean=("macro", "mean"),
        macro_std=("macro", "std"),
        micro_mean=("micro", "mean"),
        micro_std=("micro", "std"),
        weighted_mean=("weighted", "mean"),
        weighted_std=("weighted", "std"),
    )
    table.to_csv(folder / "results.csv")
    if repeats == 2 and fold_limit == 5:
        for name in config["members"]:
            values = np.zeros((2, len(y), len(classes)), np.float32)
            seen = np.zeros((2, len(y)), int)
            for r, seed in enumerate(config["cv_seeds"]):
                for f, (_, valid) in enumerate(splits(y, seed)):
                    with np.load(folder / f"{name}_r{r}_f{f}.npz") as cache:
                        values[r, valid] = cache["probabilities"]
                    seen[r, valid] += 1
            np.testing.assert_array_equal(seen, np.ones_like(seen))
            np.savez_compressed(
                folder / f"oof_{name}.npz", probabilities=values, y=y, classes=classes
            )
            error_analysis(folder, name, y, values.mean(axis=0), classes)
    return table


def evaluate_blend(
    y: Any, classes: Any, config: dict[str, Any], output: Path
) -> dict[str, Any]:
    arrays = {}
    for name in config["members"]:
        with np.load(output / "cv" / f"oof_{name}.npz") as cache:
            np.testing.assert_array_equal(cache["y"], y)
            np.testing.assert_array_equal(cache["classes"], classes)
            arrays[name] = cache["probabilities"]
    pooled = {name: values.mean(axis=0) for name, values in arrays.items()}
    weights = optimize_weights(y, pooled, config["seed"])
    records = []
    for r, seed in enumerate(config["cv_seeds"]):
        for f, (_, valid) in enumerate(splits(y, seed)):
            values = blend({n: p[r, valid] for n, p in arrays.items()}, weights)
            records.append(dict(repeat=r, fold=f, **metrics(y[valid], values)))
    summary = dict(
        optimized_weights=weights,
        apparent_pooled=metrics(y, blend(pooled, weights)),
        apparent_folds=records,
        frozen_weights_pooled=metrics(y, blend(pooled, config["weights"])),
        note="All scores in this file reuse labels involved in weight optimization; use nested.json for nested validation.",
    )
    write_json(output / "blend.json", summary)
    return summary


def nested_validate(
    traces: list[Trace],
    y: Any,
    classes: Any,
    config: dict[str, Any],
    output: Path,
    fold_limit: int = 5,
) -> dict[str, Any]:
    folder = output / "nested"
    folder.mkdir(parents=True, exist_ok=True)
    seed = config["seed"]
    names = list(config["members"])
    records = []
    oof = np.zeros((len(y), len(classes)))
    seen = np.zeros(len(y), int)
    with threadpool_limits(limits=1):
        for fold, (train, valid) in enumerate(splits(y, seed)):
            if fold >= fold_limit:
                break
            path = folder / f"fold{fold}.npz"
            if path.exists():
                with np.load(path) as cache:
                    np.testing.assert_array_equal(cache["train"], train)
                    np.testing.assert_array_equal(cache["valid"], valid)
                    values = cache["blend"]
                record = json.loads(path.with_suffix(".json").read_text())
            else:
                inner_oof = {n: np.zeros((len(train), len(classes))) for n in names}
                inner_seen = np.zeros(len(train), int)
                inner_splits = StratifiedKFold(
                    3, shuffle=True, random_state=seed + fold
                ).split(np.zeros(len(train)), y[train])
                for inner, (a, b) in enumerate(inner_splits):
                    for name in names:
                        inner_oof[name][b] = fit_predict(
                            config["members"][name],
                            traces,
                            y,
                            train[a],
                            train[b],
                            seed + inner,
                        )
                    inner_seen[b] += 1
                np.testing.assert_array_equal(inner_seen, np.ones_like(inner_seen))
                weights = optimize_weights(y[train], inner_oof, seed + fold)
                predictions = {
                    name: fit_predict(
                        config["members"][name], traces, y, train, valid, seed + fold
                    )
                    for name in names
                }
                values = blend(predictions, weights)
                record = dict(
                    fold=fold,
                    weights=weights,
                    singles={n: metrics(y[valid], p) for n, p in predictions.items()},
                    blend=metrics(y[valid], values),
                )
                np.savez_compressed(
                    path,
                    train=train,
                    valid=valid,
                    blend=values,
                    members=np.stack([predictions[n] for n in names]),
                )
                write_json(path.with_suffix(".json"), record)
            check_probabilities(values, (len(valid), len(classes)))
            oof[valid] = values
            seen[valid] += 1
            records.append(record)
            print(json.dumps(record), flush=True)
    summary: dict[str, Any] = dict(
        complete=fold_limit == 5,
        folds=records,
        note="Weights and preprocessing are nested; model identities were shortlisted in earlier exploratory CV.",
    )
    if fold_limit == 5:
        np.testing.assert_array_equal(seen, np.ones_like(seen))
        summary.update(
            {
                f"{key}_{stat}": float(fn([r["blend"][key] for r in records]))
                for key in ("macro", "micro", "weighted")
                for stat, fn in (
                    ("mean", np.mean),
                    ("std", lambda v: np.std(v, ddof=1)),
                )
            }
        )
        summary["pooled"] = metrics(y, oof)
        np.savez_compressed(folder / "oof.npz", probabilities=oof, y=y, classes=classes)
        error_analysis(folder, "blend", y, oof, classes)
    write_json(
        output / ("nested.json" if fold_limit == 5 else "nested_partial.json"), summary
    )
    return summary
