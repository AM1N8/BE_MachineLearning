"""Check exact explanations, calibration bins and competition validation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier

from copilote.interpret import forest_path_contributions, reliability
from copilote.predict import run_prediction, validate_submission
from copilote.settings import load_config


def test_forest_path_increments_reconstruct_probability():
    matrix = csr_matrix([[0, 1], [1, 0], [0, 2], [2, 0], [2, 1], [1, 2]])
    forest = RandomForestClassifier(n_estimators=5, random_state=42).fit(matrix, [0, 1, 0, 1, 1, 0])
    for row in range(matrix.shape[0]):
        for label in (0, 1):
            prior, contributions = forest_path_contributions(forest, matrix[row], label)
            assert np.isclose(
                prior + contributions.sum(), forest.predict_proba(matrix[row])[0, label]
            )


def test_reliability_includes_confidence_one():
    result = reliability(np.array(["a", "b"]), np.array([[1.0, 0], [0, 1.0]]), np.array(["a", "b"]))
    assert result["ece"] == 0
    assert sum(b["count"] for b in result["bins"]) == 2
    assert result["coverage"][-1]["coverage"] == 1


def test_submission_validation_rejects_each_schema_violation():
    config = load_config("configs/default.yaml")
    valid = pd.DataFrame({"RowId": [1, 2], "prediction": ["a", "b"]})
    validate_submission(valid, config, {"a", "b"}, 2)
    for invalid in (
        valid.rename(columns={"RowId": "Id"}),
        valid.iloc[:1],
        valid.assign(RowId=[0, 1]),
        valid.assign(prediction=["a", None]),
        valid.assign(prediction=["a", "new"]),
    ):
        with pytest.raises(ValueError):
            validate_submission(invalid, config, {"a", "b"}, 2)


def test_final_refit_sidecar_and_idempotent_filename(tmp_path, monkeypatch):
    config = load_config("configs/default.yaml")
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    sample = tmp_path / "sample.csv"
    train.write_text("a,F,common,alpha\na,F,common,alpha\nb,C,common,beta\nb,C,common,beta\n")
    test.write_text("F,common,alpha\nC,common,beta\n")
    pd.DataFrame({"RowId": [1, 2], "prediction": ["x", "x"]}).to_csv(sample, index=False)
    config["paths"].update(train=str(train), test=str(test), sample_submission=str(sample))
    monkeypatch.chdir(tmp_path)
    Path("reports/results").mkdir(parents=True)
    Path("artifacts").mkdir()
    spec = {"model": "linear", "families": ["raw", "browser"]}
    Path("reports/results/selection.json").write_text(
        json.dumps({"signature": "toy", "selected": "linear", "spec": spec})
    )
    Path("reports/results/evaluation.json").write_text(
        json.dumps({"signature": "toy", "repeated": {"linear": {"mean": {}}}})
    )
    first = run_prediction(config)
    second = run_prediction(config)
    assert first["submission"] == second["submission"]
    assert first["validated"] and first["rows"] == 2
    assert pd.read_csv("submissions/final.csv").prediction.tolist() == ["a", "b"]
