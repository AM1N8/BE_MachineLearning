"""Behavioral boundary and nested reranking leakage regression tests."""

from dataclasses import replace

import joblib
import numpy as np
import pytest

from copilote.io import Session
from copilote.phase2_models import BehavioralView, Profiles, TopTwo, behavioral_model


def test_context_resets_between_rows_and_after_logout():
    first = Session(0, "Chrome", ("Open(secret)<config>", "Click"), "a")
    second = Session(1, "Chrome", ("Click",), "b")
    view = BehavioralView("context")
    output = view.transform([first, second])
    assert any("secret" in token for token in output[0])
    assert not any("secret" in token for token in output[1])
    assert output[1] == view.transform([second])[0]
    ended = replace(first, tokens=("Open(secret)", "Fermeture de session", "Click"))
    assert "secret" not in view.transform([ended])[0][-1]


@pytest.mark.parametrize("family", ["context", "transitions", "position", "edges", "signatures"])
def test_behavioral_vocabulary_and_reload_are_inductive(family, tmp_path):
    rows = [
        Session(i, "Chrome", (f"Open(screen{i % 3})", "Click", "Click"), str(i % 3))
        for i in range(12)
    ]
    model = behavioral_model(family).fit(rows, [r.label for r in rows])
    unseen = [Session(99, "Chrome", ("Open(heldout_secret)", "Click"), "poison")]
    vocabulary = (
        model.estimator.named_steps["features"]
        .transformer_list[1][1]
        .named_steps["tfidf"]
        .vocabulary_.copy()
    )
    predicted = model.predict(unseen)
    assert not any("heldout_secret" in term for term in vocabulary)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(joblib.load(path).predict(unseen), predicted)


def test_profiles_do_not_learn_from_unknown_events():
    rows = [Session(i, "Chrome", (f"Action{i}", "Common"), str(i)) for i in range(3)]
    profile = Profiles().fit(rows, [r.label for r in rows])
    unknown = [Session(99, "Chrome", ("Unknown",), "poison")]
    x, _ = profile.transform(unknown, np.array([[0.2, 0.1, -0.1]]), np.array(["0", "1", "2"]))
    assert np.isfinite(x).all()
    assert x[0, 8] == 1
    assert "Unknown" not in profile.df


def test_nested_meta_scores_never_use_their_own_training_rows(monkeypatch):
    class GuardedBase:
        classes_ = np.array(["a", "b", "c"])

        def fit(self, rows, y):
            self.ids = {r.row_id for r in rows}
            return self

        def decision_function(self, rows):
            assert self.ids.isdisjoint(r.row_id for r in rows)
            return np.array([np.roll([0.2, 0.1, -0.3], r.row_id % 3) for r in rows])

    monkeypatch.setattr("copilote.phase2_models.make_model", lambda *args: GuardedBase())
    rows = [
        Session(i, "Chrome", ("Common", f"Action{i % 4}"), "abc"[(i + i // 3) % 3])
        for i in range(54)
    ]
    model = TopTwo().fit(rows, np.array([r.label for r in rows]))
    for split in model.audit_["inner_splits"]:
        assert set(split["train"]).isdisjoint(split["heldout"])
    heldout = [Session(100, "Chrome", ("Common", "New"), "a")]
    np.testing.assert_array_equal(
        model.predict(heldout), model.predict([replace(heldout[0], label="c")])
    )
