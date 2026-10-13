"""Final inductive refit and strict competition-format validation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from copilote.io import file_hash, read_sessions, write_json
from copilote.models import build_model
from copilote.settings import provenance


def validate_submission(
    frame: pd.DataFrame, config: dict[str, Any], labels: set[str], expected_rows: int
) -> None:
    """Reject invalid headers, row order, missing values, and unknown user labels."""
    settings = config["submission"]
    columns = [settings["id_column"], settings["target_column"]]
    if frame.columns.tolist() != columns:
        raise ValueError(f"Expected columns {columns}, got {frame.columns.tolist()}")
    if len(frame) != expected_rows or frame.isna().any().any():
        raise ValueError("Wrong row count or missing values")
    if frame[columns[0]].tolist() != list(
        range(settings["id_base"], expected_rows + settings["id_base"])
    ):
        raise ValueError("IDs must match the original test row order")
    if not set(frame[columns[1]].astype(str)).issubset(labels):
        raise ValueError("Predictions contain labels absent from training")


def run_prediction(config: dict[str, Any]) -> dict[str, Any]:
    """Fit the locked specification on all training sessions and transform test only."""
    selection = json.loads(Path("reports/results/selection.json").read_text())
    evaluation = json.loads(Path("reports/results/evaluation.json").read_text())
    if selection["signature"] != evaluation["signature"]:
        raise ValueError("Selection and evaluation fingerprints disagree")
    train = read_sessions(config["paths"]["train"], labelled=True)
    test = read_sessions(config["paths"]["test"], labelled=False)
    model = build_model(selection["spec"], config)
    with threadpool_limits(limits=2):
        model.fit(train, [str(r.label) for r in train])
        predictions = model.predict(test)
    settings = config["submission"]
    frame = pd.DataFrame(
        {
            settings["id_column"]: range(settings["id_base"], settings["id_base"] + len(test)),
            settings["target_column"]: predictions,
        }
    )
    validate_submission(frame, config, {str(r.label) for r in train}, len(test))
    sample_path = Path(config["paths"]["sample_submission"])
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        if frame.columns.tolist() != sample.columns.tolist() or not frame.iloc[:, 0].equals(
            sample.iloc[:, 0]
        ):
            raise ValueError("Configured format disagrees with the supplied sample submission")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    previous_path = Path("reports/results/submission.json")
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        if previous["selection_signature"] == selection["signature"]:
            timestamp = previous["timestamp_utc"]
    destination = Path(f"submissions/submission_{timestamp}_{selection['selected']}.csv")
    destination.parent.mkdir(exist_ok=True)
    frame.to_csv(destination, index=False)
    validate_submission(pd.read_csv(destination), config, set(model.classes_), len(test))
    joblib.dump(model, "artifacts/final_model.joblib")
    metadata = {
        "provenance": provenance(config),
        "model": selection["selected"],
        "spec": selection["spec"],
        "selection_signature": selection["signature"],
        "cv": evaluation["repeated"][selection["selected"]],
        "rows": len(frame),
        "header": frame.columns.tolist(),
        "validated": True,
        "transductive": False,
        "submission": str(destination),
        "submission_sha256": file_hash(destination),
        "timestamp_utc": timestamp,
    }
    write_json(destination.with_suffix(".json"), metadata)
    write_json("reports/results/submission.json", metadata)
    # A stable convenience alias is identical, not a further candidate.
    frame.to_csv("submissions/final.csv", index=False)
    return metadata
