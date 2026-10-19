"""Exercise sparse kernel geometry and fold-local retrieval behavior."""

import joblib
import numpy as np
import pytest
from sklearn.base import clone

from copilote.io import Session
from copilote.similarity import SimilarityClassifier


@pytest.mark.parametrize(
    "method", ["nearest", "centroid", "kernel_ridge", "kernel_svc", "weighted_svm"]
)
def test_similarity_is_inductive_and_serializable(method, tmp_path):
    rows = [
        Session(i, "Chrome", (token, token, "common"), label)
        for i, (token, label) in enumerate(
            [
                ("alpha", "a"),
                ("alpha", "a"),
                ("beta", "b"),
                ("beta", "b"),
                ("gamma", "c"),
                ("gamma", "c"),
            ]
        )
    ]
    model = SimilarityClassifier({"method": method}).fit(rows, [r.label for r in rows])
    before = model.vectorizer_.idf_.copy()
    query = [Session(99, "unseen-browser", ("alpha", "heldoutonly"), None)]
    assert model.predict(query).tolist() == ["a"]
    assert "heldoutonly" not in model.vectorizer_.vocabulary_
    np.testing.assert_array_equal(before, model.vectorizer_.idf_)
    assert not hasattr(clone(model), "vectorizer_")
    assert np.isfinite(model.decision_function(query)).all()
    joblib.dump(model, tmp_path / "model.joblib")
    np.testing.assert_array_equal(
        joblib.load(tmp_path / "model.joblib").predict(query), model.predict(query)
    )


def test_browser_support_comes_from_training():
    rows = [Session(0, "Firefox", ("alpha",), "a"), Session(1, "Chrome", ("beta",), "b")]
    model = SimilarityClassifier({"method": "nearest"}).fit(rows, ["a", "b"])
    assert model.predict([Session(2, "Firefox", ("beta",))]).tolist() == ["a"]
    assert model.predict([Session(3, "unknown", ("beta",))]).tolist() == ["b"]


@pytest.mark.parametrize("weighting", ["bm25", "user_idf", "entropy"])
def test_supervised_weights_transform_consistently(weighting):
    rows = [
        Session(i, "Chrome", (token, token, "common"), label)
        for i, (token, label) in enumerate(
            [("alpha", "a"), ("alpha", "a"), ("beta", "b"), ("beta", "b")]
        )
    ]
    model = SimilarityClassifier(
        {"method": "weighted_svm", "weighting": weighting, "power": 0.5}
    ).fit(rows, [r.label for r in rows])
    np.testing.assert_allclose(
        model._features(rows).toarray(), model.training_.toarray(), atol=1e-12
    )
    before = model.training_.copy()
    model.predict([Session(99, "Chrome", ("unseen", "alpha"))])
    np.testing.assert_array_equal(before.toarray(), model.training_.toarray())
