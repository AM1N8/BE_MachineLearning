"""Execute frozen behavioral ablations and nested top-two correction, with paired gates."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from copilote.improvement import SPLITS, make_model, metrics, summarize
from copilote.io import file_hash, read_sessions, write_json
from copilote.phase2_audit import audit
from copilote.phase2_models import FAMILIES, TopTwo, behavioral_model


def run(stage):
    sources = {
        name: file_hash(f"src/copilote/{name}.py")
        for name in (
            "phase2",
            "phase2_models",
            "phase2_audit",
            "competitive",
            "improvement",
            "features",
            "tokens",
            "io",
        )
    }
    manifest = {
        "sources": sources,
        "train": file_hash("data/train.csv"),
        "lock": file_hash("uv.lock"),
        "historical_splits": file_hash(SPLITS),
        "baseline_submission": file_hash("submissions/best_verified.csv"),
        "families": list(FAMILIES),
        "feature_weight": 0.3,
        "seed": 42,
        "confirmation_seed": 43,
        "additional_seed": 44,
    }
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    directory = Path("reports/results/phase2") / signature[:12]
    directory.mkdir(parents=True, exist_ok=True)
    write_json(
        directory.parent / "active.json", {"directory": str(directory), "signature": signature}
    )
    write_json(directory / "manifest.json", manifest)
    if stage == "audit":
        audit(directory)
        return
    rows = read_sessions("data/train.csv", labelled=True)
    y = np.array([r.label for r in rows])
    historical = [
        (np.array(f["train"]), np.array(f["validation"]))
        for f in json.loads(SPLITS.read_text())["folds"]
    ]
    names = ["baseline", *FAMILIES, "reranker"]
    if stage == "research":
        splits, prefix = historical, "research"
    else:
        prior = pd.read_csv(directory / "comparison.csv")
        # Confirm all positive-F1/accuracy hypotheses and the primary reranker, even null.
        base_acc = prior.loc[prior.experiment == "baseline", "accuracy"].iloc[0]
        names = ["baseline"] + prior.loc[
            (prior.experiment != "baseline")
            & (
                (prior.delta_f1 > 0)
                | (prior.accuracy > base_acc)
                | (prior.experiment == "reranker")
            ),
            "experiment",
        ].tolist()
        seed = 43 if stage in ("confirm", "grouped") else 44
        splits = list(
            RepeatedStratifiedKFold(n_splits=4, n_repeats=2, random_state=seed).split(y, y)
        )
        prefix = "confirm" if stage == "confirm" else stage
        if stage == "grouped":
            groups = np.array(json.loads((directory / "groups.json").read_text()))
            splits = [
                s
                for repeat in range(2)
                for s in StratifiedGroupKFold(4, shuffle=True, random_state=43 + repeat).split(
                    y, y, groups
                )
            ]
    write_json(
        directory / f"{prefix}_splits.json",
        [{"train": a.tolist(), "validation": b.tolist()} for a, b in splits],
    )
    for fold, (a, b) in enumerate(splits):
        training, validation = [rows[i] for i in a], [rows[i] for i in b]
        for name in names:
            path = directory / f"{prefix}_{name}_{fold}.csv"
            if path.exists() and path.with_suffix(".json").exists():
                continue
            start = time.perf_counter()
            with threadpool_limits(limits=2):
                model = (
                    make_model("baseline", y[a])
                    if name == "baseline"
                    else TopTwo()
                    if name == "reranker"
                    else behavioral_model(name)
                )
                model.fit(training, y[a])
                predicted = model.predict(validation)
            frame = pd.DataFrame(
                {
                    "row_id": b,
                    "truth": y[b],
                    "prediction": predicted,
                    "fold": fold,
                    "length": [
                        sum(not t.startswith("t") or not t[1:].isdigit() for t in r.tokens)
                        for r in validation
                    ],
                }
            )
            if name == "reranker":
                write_json(directory / f"{prefix}_reranker_{fold}_inner.json", model.audit_)
                for key, value in model.last_.items():
                    frame[key] = value
            frame.to_csv(path, index=False)
            result = {
                "experiment": name,
                "fold": fold,
                "seconds": time.perf_counter() - start,
                **metrics(y[b], predicted),
            }
            write_json(path.with_suffix(".json"), result)
            print(prefix, result, flush=True)
    # The existing gate is defined for eight folds (two four-fold repetitions).
    summarize(directory, names, prefix)
    summary_path = directory / ("confirmation.csv" if prefix == "confirm" else "comparison.csv")
    if prefix not in ("research", "confirm"):
        # Preserve historical selection summary after generic summarize writes comparison.csv.
        summary_path.rename(directory / f"{prefix}_comparison.csv")
        summarize(directory, ["baseline", *FAMILIES, "reranker"], "research")
    for name in names:
        frames = pd.concat([pd.read_csv(directory / f"{prefix}_{name}_{f}.csv") for f in range(8)])
        frames["length_bin"] = pd.cut(frames.length, [0, 20, 100, 500, np.inf]).astype(str)
        frames["wrong"] = frames.truth != frames.prediction
        frames.groupby("length_bin").wrong.agg(["count", "sum", "mean"]).to_csv(
            directory / f"{prefix}_{name}_length.csv"
        )
    assert file_hash("submissions/best_verified.csv") == manifest["baseline_submission"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["audit", "research", "confirm", "grouped", "seed44"])
    run(parser.parse_args().stage)
