"""Test class alignment and fitted-only browser behavior."""

import json
from pathlib import Path

import numpy as np
import pytest

from copilote.evaluate import scores
from copilote.io import Session
from copilote.models import BrowserClassifier
from copilote.settings import load_config


@pytest.fixture
def toy_project(tmp_path, monkeypatch):
    config = load_config("configs/default.yaml")
    config["features"]["min_df"] = 1
    config["models"]["trees"] = 3
    lines = [
        f"user{k},browser{k % 2},start,action{k}(screen{k}),t{5 * (i + 1)},end"
        for k in range(4)
        for i in range(12)
    ]
    train = tmp_path / "train.csv"
    train.write_text("\n".join(lines) + "\n")
    test = tmp_path / "test.csv"
    test.write_text("browser0,start,new_action,t10,end\n")
    config["paths"]["train"] = str(train)
    config["paths"]["test"] = str(test)
    (tmp_path / "artifacts").mkdir()
    monkeypatch.chdir(tmp_path)
    return config


def test_metrics_use_given_class_order():
    classes = np.array(["a", "b", "c", "d", "e", "f"])
    probabilities = np.array([[0.05, 0.1, 0.6, 0.1, 0.1, 0.05], [0.7, 0.1, 0.05, 0.05, 0.05, 0.05]])
    result = scores(np.array(["c", "a"]), np.array(["c", "a"]), probabilities, classes)
    assert result["accuracy"] == result["macro_f1"] == result["top3"] == result["top5"] == 1
    assert np.isclose(result["log_loss"], -np.log([0.6, 0.7]).mean())


def test_browser_baseline_uses_training_counts_and_prior():
    rows = [Session(i, browser, ()) for i, browser in enumerate(["F", "F", "C"])]
    model = BrowserClassifier().fit(rows, np.array(["a", "a", "b"]))
    assert model.predict([Session(4, "F", ())]).tolist() == ["a"]
    np.testing.assert_allclose(model.predict_proba([Session(5, "new", ())]), [[2 / 3, 1 / 3]])


def test_end_to_end_selection_calibration_oof_and_cache(toy_project, monkeypatch):
    from copilote import evaluate
    from copilote.models import catalogue

    candidates = catalogue()
    candidates = {
        k: candidates[k]
        for k in ("majority", "browser", "linear", "linear_unigrams", "linear_balanced")
    }
    monkeypatch.setattr(evaluate, "catalogue", lambda: candidates)
    monkeypatch.setattr(evaluate, "cache_signature", lambda config: "test_signature")
    result = evaluate.run_training(toy_project)
    assert result["selected"].startswith("linear")
    split = json.loads(Path("reports/results/splits.json").read_text())
    assert not set(split["development"]) & set(split["holdout"])
    for fold in split["development_folds"]:
        assert not set(fold["train"]) & set(fold["validation"])
        assert not set(fold["train"]) & set(split["holdout"])
    evaluation = json.loads(Path("reports/results/evaluation.json").read_text())
    assert len(evaluation["repeated"]["linear"]["folds"]) == 12
    assert evaluation["repeated"]["linear"]["pooled_oof"]["accuracy"] == 1
    monkeypatch.setattr(
        evaluate, "build_model", lambda *args: pytest.fail("matching cache should not fit")
    )
    rows = evaluate.read_sessions(toy_project["paths"]["train"], labelled=True)
    cached = evaluate.evaluate_candidate(
        "linear",
        candidates["linear"],
        rows,
        np.array([r.label for r in rows]),
        [],
        toy_project,
        "repeated",
        "test_signature",
    )
    assert cached["mean"]["accuracy"] == 1
