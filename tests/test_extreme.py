"""Check boundary preservation, fold isolation, conditional geometry and persistence."""

import joblib
import numpy as np
import pytest
from sklearn.base import clone

from copilote.extreme_models import MarkovIdentity, SequenceView, build, encode
from copilote.io import Session


def rows():
    return [
        Session(
            i, "Firefox", ("shared action", f"user {i // 3}", "t5", f"user {i // 3}"), str(i // 3)
        )
        for i in range(9)
    ]


def test_sequence_boundaries_and_clock_isolation():
    a = Session(0, "F", ("a b", "c", "t5"))
    b = Session(1, "F", ("a", "b c"))
    stream = SequenceView("order", 2).transform([a, b])
    assert stream == [[encode(("a b", "c"))], [encode(("a", "b c"))]]
    assert (
        SequenceView("edge", k=1).transform([a])[0]
        == SequenceView("edge", k=1).transform([b, a])[1]
    )
    assert all("t5" not in t for t in SequenceView("skip").transform([a])[0])
    assert all("c a" not in t for t in SequenceView("char").transform([a])[0])


@pytest.mark.parametrize(
    "spec",
    [
        {"raw": 0, "unigram": 1, "bigram": 1, "edge": 0.3, "min_df": 1},
        {"skip": 0.2, "behavior": 0.03, "percentile": 50, "min_df": 1},
        {"method": "markov", "order": 3, "mu": 20},
    ],
)
def test_fold_local_models_and_serialization(spec, tmp_path):
    train = rows()
    model = build(spec).fit(train, [r.label for r in train])
    heldout = [Session(99, "Unseen browser", ("HELDOUT_ONLY", "shared action", "user 1"))]
    before = joblib.hash(model)
    scores = model.decision_function(heldout)
    assert np.isfinite(scores).all()
    assert scores.shape == (1, 3)
    assert joblib.hash(model) == before
    assert not hasattr(clone(model), "classes_")
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(joblib.load(path).predict(heldout), model.predict(heldout))
    if isinstance(model, MarkovIdentity):
        assert all(
            encode(("HELDOUT_ONLY",)) not in vectorizer.vocabulary_
            for _, vectorizer, _ in model.channels_
        )


def test_markov_conditional_probabilities_sum_to_one():
    train = rows()
    model = MarkovIdentity({"order": 2}).fit(train, [r.label for r in train])
    for _, vectorizer, logprob in model.channels_:
        import json

        contexts = [encode(json.loads(s)[:-1]) for s in vectorizer.get_feature_names_out()]
        for context in set(contexts):
            np.testing.assert_allclose(
                np.exp(logprob[:, np.array(contexts) == context]).sum(axis=1), 1
            )


def test_mask_aware_normalization():
    from copilote.extreme import normalized

    values = normalized(np.array([[1.0, 3.0, -1e6], [2.0, -1e6, -1e6]]))
    np.testing.assert_allclose(values[0, :2], [-1, 1])
    assert values[0, 2] == -1e6
    assert values[1, 0] == 0


def test_nested_policy_never_sees_outer_labels(tmp_path, monkeypatch):
    import json

    import copilote.extreme as module

    train = [
        Session(i, "Firefox", ("shared", f"user {i // 10}", "action"), str(i // 10))
        for i in range(40)
    ]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "read_sessions", lambda *args, **kwargs: train)
    monkeypatch.setattr(module, "file_hash", lambda path: "synthetic")
    campaign = module.Campaign(
        {
            "nested_trials": 0,
            "nested_candidates": ["champion", "separated"],
            "objective_status": "synthetic macro-F1",
            "candidates": {
                "champion": {"method": "champion"},
                "separated": {"raw": 0, "unigram": 1, "bigram": 0.5, "min_df": 1},
            },
        }
    )
    campaign.nested()
    splits = json.loads((campaign.directory / "outer_splits.json").read_text())["folds"]
    result = json.loads((campaign.directory / "nested_validation.json").read_text())
    for split, fold in zip(splits, result["folds"], strict=True):
        assert set(fold["policy"]["train_ids"]) == set(split["train"])
        assert not set(fold["policy"]["train_ids"]) & set(split["validation"])
    assert not result["promote"]  # Ties cannot replace the frozen champion.
