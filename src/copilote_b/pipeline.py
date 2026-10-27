"""Persist final models, write the submission and verify reference predictions."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

from .config import file_hash, fingerprint, write_json
from .data import Session, data_path, load_sample, load_sessions
from .ensemble import blend
from .features import Trace, prepare
from .model import CropSVM, check_probabilities


def load_training(data: Path) -> tuple[list[Session], list[Trace], Any, Any]:
    sessions = load_sessions(data_path(data, "train.csv"), labeled=True)
    encoder = LabelEncoder().fit([s.label for s in sessions])
    y = encoder.transform([s.label for s in sessions])
    return sessions, [prepare(s) for s in sessions], y, encoder.classes_


def initialize(data: Path, output: Path, config: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        data_path(data, name)
        for name in ("train.csv", "test.csv", "sample_submission.csv")
    ]
    identifier = fingerprint(config, paths)
    path = output / "run.json"
    if path.exists() and json.loads(path.read_text())["fingerprint"] != identifier:
        raise ValueError(
            "Data, code or configuration changed: use a fresh output directory"
        )
    write_json(
        path,
        dict(
            fingerprint=identifier,
            config=config,
            data_sha256={p.name: file_hash(p) for p in paths},
            python=platform.python_version(),
            packages={
                n: importlib.metadata.version(n)
                for n in (
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "pandas",
                    "joblib",
                    "optuna",
                    "threadpoolctl",
                )
            },
        ),
    )


def train(
    traces: list[Trace], y: Any, classes: Any, config: dict[str, Any], output: Path
) -> None:
    folder = output / "models"
    folder.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        for name, parameters in config["members"].items():
            path = folder / f"{name}.joblib"
            if path.exists():
                print(f"Reusing {path}", flush=True)
                continue
            model = CropSVM(parameters, config["seed"]).fit(traces, y)
            np.testing.assert_array_equal(model.classes_, np.arange(len(classes)))
            joblib.dump(dict(model=model, classes=classes), path, compress=3)
            print(f"Fitted {name}", flush=True)
            del model
            gc.collect()


def validate_submission(
    frame: pd.DataFrame, sample: pd.DataFrame, classes: Any
) -> None:
    if list(frame.columns) != list(sample.columns) or len(frame) != len(sample):
        raise ValueError("Submission schema or row count mismatch")
    if frame.isna().any().any() or frame.RowId.tolist() != sample.RowId.tolist():
        raise ValueError("Submission contains missing values or reordered IDs")
    if not set(frame.prediction).issubset(set(classes)):
        raise ValueError("Submission contains unknown labels")


def predict(data: Path, config: dict[str, Any], output: Path) -> pd.DataFrame:
    test = load_sessions(data_path(data, "test.csv"), labeled=False)
    traces = [prepare(s) for s in test]
    sample = load_sample(data, len(test))
    predictions = {}
    classes = None
    with threadpool_limits(limits=1):
        for name in config["members"]:
            bundle = joblib.load(output / "models" / f"{name}.joblib")
            if classes is None:
                classes = bundle["classes"]
            np.testing.assert_array_equal(bundle["classes"], classes)
            np.testing.assert_array_equal(
                bundle["model"].classes_, np.arange(len(classes))
            )
            predictions[name] = bundle["model"].predict_proba(traces)
            np.save(output / f"test_{name}.npy", predictions[name])
            del bundle
            gc.collect()
    if classes is None:
        raise ValueError("At least one configured model is required")
    probabilities = blend(predictions, config["weights"])
    frame = sample.copy()
    frame["prediction"] = classes[probabilities.argmax(axis=1)]
    validate_submission(frame, sample, classes)
    frame.to_csv(output / "submission_B.csv", index=False)
    validate_submission(pd.read_csv(output / "submission_B.csv"), sample, classes)
    np.savez_compressed(
        output / "probabilities.npz", probabilities=probabilities, classes=classes
    )
    write_json(
        output / "submission_validation.json",
        dict(
            rows=len(frame),
            columns=list(frame.columns),
            ids_match=True,
            labels_valid=True,
            no_nan=True,
            weights=config["weights"],
            scores_are_calibrated=False,
        ),
    )
    return frame


def verify(
    output: Path, reference: Path | None = None, reference_artifacts: Path | None = None
) -> dict[str, Any]:
    frame = pd.read_csv(output / "submission_B.csv")
    with np.load(output / "probabilities.npz") as values:
        probabilities, classes = values["probabilities"], values["classes"]
    check_probabilities(probabilities, (len(frame), len(classes)))
    np.testing.assert_array_equal(
        frame.prediction, classes[probabilities.argmax(axis=1)]
    )
    result: dict[str, Any] = dict(rows=len(frame), probabilities_valid=True)
    if reference is not None:
        expected = pd.read_csv(reference)
        pd.testing.assert_frame_equal(frame, expected)
        result["reference_labels_identical"] = True
        result["reference_bytes_identical"] = file_hash(
            output / "submission_B.csv"
        ) == file_hash(reference)
    if reference_artifacts is not None:
        with np.load(reference_artifacts / "final_probabilities.npz") as expected:
            np.testing.assert_array_equal(classes, expected["classes"])
            np.testing.assert_allclose(probabilities, expected["B"], rtol=0, atol=1e-12)
        config = json.loads((output / "run.json").read_text())["config"]
        for name in config["members"]:
            np.testing.assert_allclose(
                np.load(output / f"test_{name}.npy"),
                np.load(reference_artifacts / f"test_{name}.npy"),
                rtol=0,
                atol=1e-12,
            )
        result["reference_member_probabilities_match_at_1e-12"] = True
    write_json(output / "verification.json", result)
    return result
