"""Competitive-model leakage, event-boundary, and support-mask regressions."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from copilote.competitive import BrowserRestricted, EventStream, build_competitive
from copilote.io import Session


@pytest.mark.parametrize("kernel", ["linear_svc", "nbsvm"])
def test_competitive_vocabulary_is_fold_local(kernel):
    train = [
        Session(i, f"browser{i % 3}", (f"Open(Screen{i % 3})", "Use1", "t5"), str(i % 3))
        for i in range(9)
    ]
    model = build_competitive({"kernel": kernel, "min_df": 1, "C": 10, "restrict_browser": True})
    model.fit(train, [r.label for r in train])
    vectorizer = model.estimator.named_steps["features"].transformer_list[0][1].named_steps["tfidf"]
    vocabulary = vectorizer.vocabulary_.copy()
    heldout = [Session(99, "new-browser", ("OnlyHeldOut(NewScreen)", "t100"))]
    assert model.decision_function(heldout).shape == (1, 3)
    assert vectorizer.vocabulary_ == vocabulary
    assert not any("OnlyHeldOut" in token or "t5" in token for token in vocabulary)
    assert "Open(Screen0) Use1" in vocabulary
    assert not hasattr(clone(model), "classes_")


def test_event_views_do_not_carry_context_between_sessions():
    rows = [
        Session(0, "b", ("Open(Main)<C>$H$1", "Use1", "Use1", "t10")),
        Session(1, "b", ("Use1",)),
    ]
    assert EventStream("clean").transform(rows)[0] == ("Open(Main)<C>$H$", "Use", "Use")
    assert EventStream("dedup").transform(rows)[0] == ("Open(Main)<C>$H$1", "Use1")
    assert EventStream("context").transform(rows) == [
        ("screen=Main", "config=C", "chain=H"),
        ("__EMPTY__",),
    ]


class FixedMargins(ClassifierMixin, BaseEstimator):
    """Deliberately prefer an incompatible class to exercise mask fallbacks."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def decision_function(self, X):
        return np.tile([1.0, 3.0, 2.0], (len(X), 1))


def test_browser_mask_uses_training_support_and_falls_back_on_unseen_browser():
    train = [Session(0, "A", (), "a"), Session(1, "B", (), "b"), Session(2, "A", (), "c")]
    model = BrowserRestricted(FixedMargins(), restrict=True).fit(train, [r.label for r in train])
    valid = [Session(3, "A", ()), Session(4, "B", ()), Session(5, "unseen", ())]
    assert model.predict(valid).tolist() == ["c", "b", "b"]
    assert not model.browser_support_["A"][1]


def test_browser_experts_align_binary_single_class_and_unseen_browser():
    train = [
        Session(i, "A" if i % 3 != 2 else "B", (f"Open(Screen{i % 3})", "Use1"), str(i % 3))
        for i in range(12)
    ]
    model = build_competitive({"experts": True, "C": 10, "min_df": 1})
    model.fit(train, [r.label for r in train])
    assert model.predict(train).tolist() == [r.label for r in train]
    assert model.decision_function([Session(99, "unseen", ("Novel",))]).shape == (1, 3)
    assert model.decision_function([Session(99, "B", ("Novel",))]).argmax() == 2
    assert "B" not in model.experts_["A"].browser_support_


def test_competitive_stages_refuse_unrun_selection_and_export_valid_submission(
    tmp_path, monkeypatch
):
    import pandas as pd

    import copilote.competitive as competitive

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(competitive, "signature", lambda config: "test-signature")
    monkeypatch.setattr(competitive, "provenance", lambda config: {"tested": True})
    train = tmp_path / "train.csv"
    train.write_text(
        "\n".join(
            f"{label},browser,{label}Event,Use,t5"
            for label in ("aaa", "bbb", "ccc")
            for _ in range(10)
        )
        + "\n"
    )
    test = tmp_path / "test.csv"
    test.write_text("browser,aaaEvent,Use,t5\nbrowser,bbbEvent,Use,t5\n")
    sample = tmp_path / "sample.csv"
    sample.write_text("RowId,prediction\n1,aaa\n2,aaa\n")
    config = {
        "seed": 42,
        "paths": {"train": str(train), "test": str(test), "sample_submission": str(sample)},
        "cv": {"folds": 3, "repeats": 1, "holdout_fraction": 0.2, "selection_metric": "macro_f1"},
        "submission": {"id_column": "RowId", "target_column": "prediction", "id_base": 1},
    }
    search = {"candidates": {"baseline": {"C": 1, "min_df": 1}}}
    with pytest.raises(ValueError, match="not been evaluated"):
        competitive.run(config, search, "select")
    competitive.run(config, search, "search")
    selected = competitive.run(config, search, "select")
    assert selected["selected"] == "baseline"
    assert not set(selected["split"]["development"]) & set(selected["split"]["holdout"])
    competitive.run(config, search, "validate")
    result = competitive.run(config, search, "predict")
    assert result["validated"] and result["kaggle_score"] is None
    assert pd.read_csv("submissions/competitive.csv").prediction.tolist() == ["aaa", "bbb"]
    with pytest.raises(ValueError, match="stale"):
        competitive.run(config, {"candidates": {"baseline": {"C": 100}}}, "predict")
