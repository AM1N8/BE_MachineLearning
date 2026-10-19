"""Leakage and inference-contract checks for the bounded research hypotheses."""

import joblib
import numpy as np

from copilote.improvement import diagnostics, make_model, pair_predict
from copilote.io import Session


def training_rows():
    return [
        Session(i, "Firefox", (label, "shared", label), label)
        for i, label in enumerate(["a", "a", "a", "b", "b", "c", "c"])
    ]


def test_rare_vocabulary_is_training_only_and_reload_exact(tmp_path):
    rows = training_rows()
    y = np.array([row.label for row in rows])
    model = make_model("rare_binary", y).fit(rows, y)
    heldout = [Session(99, "Firefox", ("heldout_only", "a"), None)]
    predicted = model.predict(heldout)
    rare = dict(model.estimator.named_steps["features"].transformer_list)["rare"]
    assert "heldout_only" not in rare.named_steps["tfidf"].vocabulary_
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(joblib.load(path).predict(heldout), predicted)


def test_pair_inference_does_not_use_validation_labels():
    rows = training_rows()
    y = np.array([row.label for row in rows])
    model = make_model("baseline", y).fit(rows, y)
    heldout = [Session(99, "Firefox", ("a", "shared"), None)]
    relabelled = [Session(99, "Firefox", ("a", "shared"), "fabricated")]
    ambiguous = np.array([[0.2, 0.1, -1.0]])
    np.testing.assert_array_equal(
        pair_predict(model, rows, y, heldout, ambiguous),
        pair_predict(model, rows, y, relabelled, ambiguous),
    )


def test_diagnostic_coverage_uses_only_training_events():
    result = diagnostics(
        training_rows(),
        [Session(99, "Firefox", ("novel", "a"), None)],
        np.array([[1.0, 0.0, -1.0]]),
        np.array(["a", "b", "c"]),
    )
    assert result[0]["oov"] == 0.5
    assert result[0]["unseen_transition"] == 1
