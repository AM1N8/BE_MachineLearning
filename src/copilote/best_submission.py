"""Rebuild the submission selected by the completed, fingerprinted nested campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from copilote.extreme import Campaign, MarginBlend
from copilote.extreme_models import build
from copilote.io import file_hash, read_sessions, write_json
from copilote.predict import validate_submission
from copilote.settings import load_config


def main() -> None:
    """Refit the locked winner on all training rows and verify the frozen predictions."""
    active = json.loads(Path("reports/results/extreme/active_run.json").read_text())
    campaign = Campaign(yaml.safe_load(Path("configs/extreme.yaml").read_text()))
    if campaign.fingerprint != active["signature"]:
        raise ValueError("Campaign inputs changed; rerun nested validation before submission")
    directory = campaign.directory
    validation = json.loads((directory / "nested_validation.json").read_text())
    recorded = json.loads((directory / "submission.json").read_text())
    for record in (validation, recorded):
        if record["signature"] != campaign.fingerprint:
            raise ValueError("Submission and validation signatures differ")
    for name in ("train", "test"):
        if file_hash(f"data/{name}.csv") != recorded[f"{name}_sha256"]:
            raise ValueError(f"The {name} data changed after selection")
    if validation["promote"] != recorded["promoted"]:
        raise ValueError("Recorded submission disagrees with the promotion decision")
    policy = recorded["policy"]
    promoted = validation["promote"]
    model = (
        MarginBlend(policy["spec"], policy["weight"]) if promoted else build({"method": "champion"})
    )
    test = read_sessions("data/test.csv", labelled=False)
    with threadpool_limits(limits=2):
        model.fit(campaign.rows, campaign.y)
        predicted = model.predict(test)
    frame = pd.DataFrame({"RowId": np.arange(1, len(test) + 1), "prediction": predicted})
    validate_submission(frame, load_config("configs/default.yaml"), set(campaign.y), len(test))
    sample = pd.read_csv("data/sample_submission.csv")
    if frame.columns.tolist() != sample.columns.tolist():
        raise ValueError("Submission columns disagree with the sample")
    np.testing.assert_array_equal(frame.RowId, sample.RowId)
    content = frame.to_csv(index=False)
    if hashlib.sha256(content.encode()).hexdigest() != recorded["best_verified_sha256"]:
        raise ValueError("Refit does not reproduce the locked submission")
    destination = Path("submissions/best_verified.csv")
    destination.write_text(content)
    model_path = Path("artifacts/best_verified.joblib")
    joblib.dump(model, model_path)
    np.testing.assert_array_equal(joblib.load(model_path).predict(test), predicted)
    selected = "ensemble" if promoted else "champion"
    metadata = {
        "submission": str(destination),
        "submission_sha256": file_hash(destination),
        "model_path": str(model_path),
        "model_sha256": file_hash(model_path),
        "selected_model": selected if promoted else "browser_mask_c3",
        "campaign_signature": campaign.fingerprint,
        "validation_source": str(directory / "nested_validation.json"),
        "nested_cv": validation[selected],
        "paired": validation["paired"],
        "new_policy_promoted": promoted,
        "train_sha256": recorded["train_sha256"],
        "test_sha256": recorded["test_sha256"],
        "sample_sha256": file_hash("data/sample_submission.csv"),
        "rows": len(frame),
        "training_rows": len(campaign.rows),
        "classes": len(campaign.classes),
        "seed": 42,
        "transductive": False,
        "model_reload_exact": True,
        "locked_predictions_reproduced": True,
        "official_score": None,
    }
    write_json(destination.with_suffix(".json"), metadata)
    write_json("reports/results/best_submission.json", metadata)
    print(f"Verified {destination}: {len(frame)} rows; SHA-256 {file_hash(destination)}")


if __name__ == "__main__":
    main()
