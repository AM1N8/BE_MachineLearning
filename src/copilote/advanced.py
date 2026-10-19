"""Reproducible second-round search over retrieval and nonlinear session kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from copilote.competitive import build_competitive, signature
from copilote.io import file_hash, read_sessions, write_json
from copilote.predict import validate_submission
from copilote.settings import load_config
from copilote.similarity import SimilarityClassifier

ROOT = Path("reports/results/advanced")
LOGGER = logging.getLogger(__name__)


def build(spec, seed):
    """Build fresh estimators without sharing fitted objects across folds."""
    if spec["method"] in ("previous", "experts"):
        expert = spec["method"] == "experts"
        return build_competitive(
            {
                "C": 10 if expert else 3,
                "class_weight": "balanced",
                "experts": expert,
                "restrict_browser": not expert,
            },
            seed,
        )
    return SimilarityClassifier(spec, seed)


def evaluate(name, spec, rows, splits, config, fingerprint, stage):
    """Export cached paired metrics and predictions with original session IDs."""
    path = ROOT / f"{stage}_{name}.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if cached["signature"] == fingerprint and cached["spec"] == spec:
            return cached
    y = np.array([row.label for row in rows])
    records, predictions = [], []
    for fold, (training, validation) in enumerate(splits):
        if set(y[training]) != set(y):
            raise ValueError("Training fold is missing a user")
        started = time.perf_counter()
        with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = build(spec, config["seed"]).fit([rows[i] for i in training], y[training])
            predicted = model.predict([rows[i] for i in validation])
        record = {
            "fold": fold,
            "accuracy": accuracy_score(y[validation], predicted),
            "macro_f1": f1_score(y[validation], predicted, average="macro", zero_division=0),
            "seconds": time.perf_counter() - started,
            "warnings": sorted({str(w.message) for w in caught}),
        }
        records.append(record)
        predictions.extend(
            {"row_id": rows[i].row_id, "fold": fold, "truth": y[i], "prediction": pred}
            for i, pred in zip(validation, predicted, strict=True)
        )
        LOGGER.info(
            "%s %s fold %d acc %.5f F1 %.5f",
            stage,
            name,
            fold,
            record["accuracy"],
            record["macro_f1"],
        )
    result = {
        "signature": fingerprint,
        "spec": spec,
        "folds": records,
        "mean": {
            metric: float(np.mean([r[metric] for r in records]))
            for metric in ("accuracy", "macro_f1")
        },
        "std": {
            metric: float(np.std([r[metric] for r in records], ddof=1)) if len(records) > 1 else 0.0
            for metric in ("accuracy", "macro_f1")
        },
    }
    pd.DataFrame(predictions).to_csv(path.with_suffix(".csv"), index=False)
    write_json(path, result)
    return result


def run(config, search):
    """Search development only, freeze selection, benchmark and export predictions."""
    ROOT.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "base": signature(config),
                "search": search,
                "source": {
                    name: file_hash(f"src/copilote/{name}.py")
                    for name in ("advanced", "similarity")
                },
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    rows = read_sessions(config["paths"]["train"], labelled=True)
    y = np.array([r.label for r in rows])
    dev, holdout = train_test_split(
        np.arange(len(rows)),
        test_size=config["cv"]["holdout_fraction"],
        stratify=y,
        random_state=config["seed"],
    )
    dev_rows = [rows[i] for i in dev]
    splits = list(StratifiedKFold(3, shuffle=True, random_state=config["seed"]).split(dev, y[dev]))
    results = {
        name: evaluate(name, spec, dev_rows, splits, config, fingerprint, "development")
        for name, spec in search["candidates"].items()
    }
    metric = config["cv"]["selection_metric"]
    best = max(results, key=lambda name: results[name]["mean"][metric])
    # Retain the documented simplicity rule, including the stronger ensemble gate.
    threshold = results[best]["mean"][metric] - results[best]["std"][metric]
    eligible = [name for name in results if results[name]["mean"][metric] >= threshold]
    chosen = min(
        eligible,
        key=lambda name: (
            search["candidates"][name].get("complexity", 1),
            -results[name]["mean"][metric],
        ),
    )
    selection = {
        "signature": fingerprint,
        "selected": chosen,
        "best_mean": best,
        "metric": metric,
        "official_metric_confirmed": False,
        "spec": search["candidates"][chosen],
        "development": dev.tolist(),
        "holdout": holdout.tolist(),
        "protocol": "Adaptive development search; one-fold-std simplicity rule. Historical holdout and repeated CV are descriptive after selection.",
    }
    write_json(ROOT / "selection.json", selection)
    table = pd.DataFrame(
        [{"name": name, **record["mean"]} for name, record in results.items()]
    ).sort_values(metric, ascending=False)
    table.to_csv(ROOT / "comparison.csv", index=False)
    LOGGER.info("Selected %s; ranking:\n%s", chosen, table.to_string(index=False))
    repeated_splits = list(
        RepeatedStratifiedKFold(
            n_splits=config["cv"]["folds"],
            n_repeats=config["cv"]["repeats"],
            random_state=config["seed"],
        ).split(rows, y)
    )
    repeated = {
        name: evaluate(
            name, search["candidates"][name], rows, repeated_splits, config, fingerprint, "repeated"
        )
        for name in dict.fromkeys(["previous", chosen])
    }
    benchmark = evaluate(
        chosen, selection["spec"], rows, [(dev, holdout)], config, fingerprint, "holdout"
    )
    gain = [
        a["accuracy"] - b["accuracy"]
        for a, b in zip(repeated[chosen]["folds"], repeated["previous"]["folds"], strict=True)
    ]
    validation = {
        "signature": fingerprint,
        "selected": chosen,
        "repeated": repeated,
        "holdout": benchmark,
        "paired_accuracy_gain": float(np.mean(gain)),
        "improved_folds": int(np.sum(np.array(gain) > 0)),
    }
    write_json(ROOT / "validation.json", validation)
    test = read_sessions(config["paths"]["test"], labelled=False)
    with threadpool_limits(limits=2):
        model = build(selection["spec"], config["seed"]).fit(rows, y)
        predicted = model.predict(test)
    settings = config["submission"]
    frame = pd.DataFrame(
        {
            settings["id_column"]: np.arange(settings["id_base"], settings["id_base"] + len(test)),
            settings["target_column"]: predicted,
        }
    )
    validate_submission(frame, config, set(y), len(test))
    sample = pd.read_csv(config["paths"]["sample_submission"])
    if list(frame.columns) != list(sample.columns) or not frame.iloc[:, 0].equals(
        sample.iloc[:, 0]
    ):
        raise ValueError("Sample schema or row order mismatch")
    destination = Path("submissions/advanced.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.joinpath("advanced.csv").write_text(frame.to_csv(index=False))
    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(model, "artifacts/advanced_final.joblib")
    metadata = {
        "signature": fingerprint,
        "selected": chosen,
        "spec": selection["spec"],
        "rows": len(frame),
        "train_sha256": file_hash(config["paths"]["train"]),
        "test_sha256": file_hash(config["paths"]["test"]),
        "submission_sha256": file_hash(destination),
        "validated": True,
        "transductive": False,
        "kaggle_score": None,
    }
    write_json(ROOT / "submission.json", metadata)
    write_json(destination.with_suffix(".json"), metadata)
    lines = [
        "# Deuxième recherche : similarité et profils utilisateur",
        "",
        f"Modèle retenu : `{chosen}` ; meilleure moyenne brute : `{best}`.",
        "Sélection macro-F1 sur trois folds de développement ; métrique officielle non confirmée.",
        "Toutes les transformations sont ajustées dans les folds. Aucune transduction.",
        "CV répétée et réserve historique : descriptives après sélection, pas un test indépendant.",
        "",
        "## Développement",
        "",
        "| Modèle | Accuracy | Macro-F1 |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(f"| {r.name} | {r.accuracy:.6f} | {r.macro_f1:.6f} |" for r in table.itertuples())
    lines.extend(
        [
            "",
            "## Validation répétée (4 × 3)",
            "",
            "| Modèle | Accuracy | Macro-F1 |",
            "| --- | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {name} | {r['mean']['accuracy']:.6f} ± {r['std']['accuracy']:.6f} | {r['mean']['macro_f1']:.6f} ± {r['std']['macro_f1']:.6f} |"
        for name, r in repeated.items()
    )
    lines.extend(
        [
            "",
            f"Réserve historique : accuracy {benchmark['mean']['accuracy']:.6f}, macro-F1 {benchmark['mean']['macro_f1']:.6f}.",
            f"Gain apparié moyen d'accuracy : {validation['paired_accuracy_gain']:.6f}, {validation['improved_folds']} folds améliorés.",
            "",
            "Soumission : `submissions/advanced.csv`. Score Kaggle inconnu ; dépasser 0,993 n'est pas établi.",
            "Sources : `reports/results/advanced/*.json|csv`. Reproduction : `make advanced`.",
        ]
    )
    Path("reports/advanced.md").write_text("\n".join(lines) + "\n")
    return validation


def main():
    """Run the resumable second-round search."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--search-config", default="configs/advanced.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(load_config(args.config), yaml.safe_load(Path(args.search_config).read_text()))


if __name__ == "__main__":
    main()
