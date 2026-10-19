"""Freeze existing benchmarks and audit their recorded out-of-fold errors."""

from __future__ import annotations

import ast
import itertools
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from copilote.io import file_hash, read_sessions, write_json
from copilote.tokens import parse_token

ROOT = Path("reports/results/extreme")


def freeze():
    """Copy the original submissions/models once and reject subsequent drift."""
    destination = Path("artifacts/extreme/frozen")
    destination.mkdir(parents=True, exist_ok=True)
    sources = [
        Path(p)
        for p in (
            "submissions/competitive.csv",
            "submissions/competitive.json",
            "submissions/advanced.csv",
            "submissions/advanced.json",
            "artifacts/competitive_final.joblib",
            "artifacts/advanced_final.joblib",
            "reports/results/competitive/selection.json",
            "reports/results/competitive/validation.json",
            "reports/results/advanced/selection.json",
            "reports/results/advanced/validation.json",
        )
    ]
    manifest_path = ROOT / "frozen_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {}
    for source in sources:
        digest = file_hash(source)
        key = str(source)
        if key in previous and previous[key]["sha256"] != digest:
            raise ValueError(f"Frozen benchmark changed: {source}")
        target = destination / str(source).replace("/", "__")
        if not target.exists():
            shutil.copy2(source, target)
        if file_hash(target) != digest:
            raise ValueError(f"Snapshot disagrees: {target}")
        manifest[key] = {"sha256": digest, "snapshot": str(target)}
    write_json(manifest_path, manifest)
    return manifest


def inventory():
    """Inventory source, tests, docs, notebooks, configuration and experiment results."""
    records = []
    excluded = {".git", ".venv", "data", "artifacts", "__pycache__", ".pytest_cache", ".ruff_cache"}
    for directory, dirs, files in Path(".").walk():
        dirs[:] = sorted(d for d in dirs if d not in excluded)
        for name in sorted(files):
            path = directory / name
            if path.suffix not in {".py", ".md", ".toml", ".yaml", ".ipynb", ".json"}:
                continue
            record = {"path": str(path), "sha256": file_hash(path)}
            if path.suffix == ".py":
                tree = ast.parse(path.read_text())
                record["definitions"] = [
                    node.name
                    for node in tree.body
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                ]
            elif path.suffix == ".ipynb":
                notebook = json.loads(path.read_text())
                record["cells"] = len(notebook.get("cells", []))
                record["markdown_headings"] = [
                    line.strip()
                    for cell in notebook.get("cells", [])
                    if cell["cell_type"] == "markdown"
                    for line in cell["source"]
                    if line.startswith("#")
                ]
                record["execution_errors"] = sum(
                    output.get("output_type") == "error"
                    for cell in notebook.get("cells", [])
                    for output in cell.get("outputs", [])
                )
            records.append(record)
    write_json(ROOT / "repository_inventory.json", records)
    ledger = []
    for directory in (
        Path("reports/results/competitive"),
        Path("reports/results/advanced"),
        Path("reports/results/experiments"),
    ):
        for path in sorted(directory.glob("*.json")):
            record = json.loads(path.read_text())
            if "mean" not in record or "accuracy" not in record["mean"]:
                continue
            ledger.append(
                {
                    "source": str(path),
                    "experiment": path.stem,
                    "protocol": path.stem.split("_")[0],
                    "accuracy": record["mean"]["accuracy"],
                    "macro_f1": record["mean"].get("macro_f1"),
                    "accuracy_std": record.get("std", {}).get("accuracy"),
                    "macro_f1_std": record.get("std", {}).get("macro_f1"),
                    "official_metric": "unverified",
                    "signature": record.get("signature"),
                    "spec": json.dumps(record.get("spec", {}), sort_keys=True),
                }
            )
    pd.DataFrame(ledger).to_csv(ROOT / "historical_ledger.csv", index=False)


def analyze():
    """Analyze every repeated OOF prediction; never select a new model here."""
    ROOT.mkdir(parents=True, exist_ok=True)
    freeze()
    inventory()
    train = read_sessions("data/train.csv", labelled=True)
    test = read_sessions("data/test.csv", labelled=False)
    signatures, stats = [], []
    for row in train:
        events = tuple(t for t in row.tokens if parse_token(t).seconds is None)
        counts = Counter(events)
        transitions = Counter(zip(events, events[1:]))
        signatures.append((row.browser, events))
        stats.append(
            {
                "row_id": row.row_id,
                "label": row.label,
                "browser": row.browser,
                "tokens": len(row.tokens),
                "events": len(events),
                "unique_events": len(counts),
                "unique_transitions": len(transitions),
                "repetition_ratio": 1 - len(counts) / max(len(events), 1),
                "event_entropy": float(
                    -sum(
                        (n / max(len(events), 1)) * np.log2(n / max(len(events), 1))
                        for n in counts.values()
                    )
                ),
            }
        )
    descriptors = pd.DataFrame(stats)
    descriptors.to_csv(ROOT / "session_descriptors.csv", index=False)
    sources = {
        "champion": Path("reports/results/competitive/repeated_browser_mask_c3.csv"),
        "sqrt_tfidf": Path("reports/results/advanced/repeated_hellinger_svm_c10.csv"),
        "original": Path("reports/results/competitive/repeated_baseline.csv"),
    }
    predictions = {}
    summaries = {}
    for name, path in sources.items():
        frame = pd.read_csv(path).sort_values(["fold", "row_id"]).reset_index(drop=True)
        if frame.duplicated(["fold", "row_id"]).any():
            raise ValueError("Duplicate OOF keys")
        frame["wrong"] = frame.truth != frame.prediction
        predictions[name] = frame
        classes = sorted(frame.truth.unique())
        per_class = classification_report(
            frame.truth, frame.prediction, labels=classes, output_dict=True, zero_division=0
        )
        pd.DataFrame({label: per_class[label] for label in classes}).T.rename_axis("user").to_csv(
            ROOT / f"per_class_{name}.csv"
        )
        pd.DataFrame(
            confusion_matrix(frame.truth, frame.prediction, labels=classes),
            index=classes,
            columns=classes,
        ).to_csv(ROOT / f"confusion_{name}.csv")
        errors = frame.loc[frame.wrong].merge(
            descriptors, on="row_id", suffixes=("", "_descriptor")
        )
        errors.to_csv(ROOT / f"all_errors_{name}.csv", index=False)
        pairs = (
            frame.loc[frame.wrong]
            .groupby(["truth", "prediction"])
            .size()
            .reset_index(name="errors")
            .sort_values("errors", ascending=False)
        )
        pairs.to_csv(ROOT / f"confusion_pairs_{name}.csv", index=False)
        persistent = (
            frame.groupby("row_id")
            .wrong.agg(["sum", "count"])
            .reset_index()
            .merge(descriptors, on="row_id")
        )
        persistent.to_csv(ROOT / f"persistence_{name}.csv", index=False)
        summaries[name] = {
            "oof_predictions": len(frame),
            "errors": int(frame.wrong.sum()),
            "sessions_wrong_all_repeats": int((persistent["sum"] == persistent["count"]).sum()),
            "sessions_wrong_once_or_more": int((persistent["sum"] > 0).sum()),
            "top_confusions": pairs.head(15).to_dict("records"),
        }
    diversity = []
    for (a, left), (b, right) in itertools.combinations(predictions.items(), 2):
        pd.testing.assert_frame_equal(
            left[["fold", "row_id", "truth"]], right[["fold", "row_id", "truth"]]
        )
        diversity.append(
            {
                "a": a,
                "b": b,
                "n": len(left),
                "disagreements": int((left.prediction != right.prediction).sum()),
                "a_wrong_b_correct": int((left.wrong & ~right.wrong).sum()),
                "a_correct_b_wrong": int((~left.wrong & right.wrong).sum()),
                "both_wrong": int((left.wrong & right.wrong).sum()),
                "oracle_accuracy_not_achievable_score": float((~left.wrong | ~right.wrong).mean()),
            }
        )
    pd.DataFrame(diversity).to_csv(ROOT / "diversity.csv", index=False)
    groups = {}
    for row, sig in zip(train, signatures, strict=True):
        groups.setdefault(sig, []).append(row)
    collisions = [
        {
            "row_ids": [r.row_id for r in rows],
            "labels": sorted({r.label for r in rows}),
            "events": len(sig[1]),
        }
        for sig, rows in groups.items()
        if len(rows) > 1
    ]
    write_json(ROOT / "representation_collisions.json", collisions)
    # Accuracy granularity is conditional on the unknown public subset size.
    structure = {
        "train_sessions": len(train),
        "test_sessions": len(test),
        "users": len({r.label for r in train}),
        "class_counts": dict(Counter(r.label for r in train)),
        "train_browsers": dict(Counter(r.browser for r in train)),
        "test_browsers": dict(Counter(r.browser for r in test)),
        "users_with_multiple_browsers": sum(
            len({r.browser for r in train if r.label == label}) > 1
            for label in {r.label for r in train}
        ),
        "public_size": None,
        "official_metric": None,
        "competition_rules": None,
        "label_submission_schema": pd.read_csv("data/sample_submission.csv").columns.tolist(),
        "if_accuracy_on_all_test": {
            "one_error_delta": 1 / len(test),
            "maximum_errors_strictly_above_0993": int(np.ceil(0.007 * len(test)) - 1),
        },
        "representation_duplicate_groups": len(collisions),
        "conflicting_label_groups": sum(len(c["labels"]) > 1 for c in collisions),
    }
    write_json(ROOT / "structure.json", structure)
    write_json(ROOT / "error_summary.json", summaries)
    lines = [
        "# Analyse forensique des erreurs",
        "",
        "Source : prédictions hors-fold existantes, mêmes 4 folds × 3 répétitions.",
        "Ces erreurs ont déjà été observées : elles servent aux hypothèses, pas de confirmation indépendante.",
        "",
        "| Modèle | Erreurs / prédictions | Sessions toujours erronées | Sessions erronées au moins une fois |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {name} | {s['errors']} / {s['oof_predictions']} | {s['sessions_wrong_all_repeats']} | {s['sessions_wrong_once_or_more']} |"
        for name, s in summaries.items()
    )
    lines.extend(
        [
            "",
            "## Confusions principales du champion",
            "",
            "| Vrai utilisateur | Prédit | Erreurs (répétitions incluses) |",
            "| --- | --- | ---: |",
        ]
    )
    lines.extend(
        f"| {p['truth']} | {p['prediction']} | {p['errors']} |"
        for p in summaries["champion"]["top_confusions"]
    )
    lines.extend(
        [
            "",
            "## Traces exhaustives",
            "",
            "`reports/results/extreme/all_errors_*.csv` : chaque erreur, longueur, entropie, répétitions et navigateur.",
            "`per_class_*.csv`, `confusion_*.csv`, `confusion_pairs_*.csv`, `persistence_*.csv` : tous les utilisateurs et sessions.",
            f"{len(collisions)} groupes identiques après suppression des marqueurs temporels ; {structure['conflicting_label_groups']} groupes comportent plusieurs étiquettes.",
            "Aucun extrait d'événements bruts n'est inclus dans ces rapports.",
        ]
    )
    Path("reports/error_analysis.md").write_text("\n".join(lines) + "\n")
    lines = [
        "# Complémentarité des modèles",
        "",
        "Comparaison des mêmes lignes hors-fold, avec alignement vérifié sur fold, row_id et vérité.",
        "",
        "| A | B | Désaccords | A faux, B juste | A juste, B faux | Tous deux faux |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {d['a']} | {d['b']} | {d['disagreements']} | {d['a_wrong_b_correct']} | {d['a_correct_b_wrong']} | {d['both_wrong']} |"
        for d in diversity
    )
    lines.extend(
        [
            "",
            "Le gain oracle est une borne descriptive utilisant les étiquettes, jamais un score réalisable annoncé.",
            "Aucun poids ni biais de classe n'est appris sur ces prédictions puis évalué sur les mêmes étiquettes.",
            "Validation indépendante des ensembles : non exécutée à ce stade ; métrique officielle en attente.",
        ]
    )
    Path("reports/ensemble_analysis.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "structure": {k: v for k, v in structure.items() if k != "class_counts"},
                "errors": summaries,
                "diversity": diversity,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    analyze()
