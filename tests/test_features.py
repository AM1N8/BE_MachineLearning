"""Leakage and event-boundary regression tests."""

import numpy as np
from sklearn.base import clone

from copilote.features import build_features, feature_names, views
from copilote.io import Session


def test_unseen_tokens_and_browser_do_not_change_vocabulary():
    train = [
        Session(0, "Firefox", ("Open(Main)", "t10", "Close"), "a"),
        Session(1, "Firefox", ("Open(Main)",), "b"),
    ]
    test = [Session(2, "NeverSeenBrowser", ("OnlyInValidation(NewScreen)", "t999"))]
    features = build_features(("raw", "browser", "numeric"), min_df=1)
    matrix = features.fit_transform(train)
    before = feature_names(features).copy()
    scaler_mean = features.transformer_list[-1][1].named_steps["scale"].mean_.copy()
    transformed = features.transform(test)
    np.testing.assert_array_equal(before, feature_names(features))
    np.testing.assert_array_equal(
        scaler_mean, features.transformer_list[-1][1].named_steps["scale"].mean_
    )
    assert transformed.shape[1] == matrix.shape[1]
    assert not any("OnlyInValidation" in name or "NeverSeen" in name for name in before)
    assert any("Open(Main) Close" in name for name in before)
    assert not hasattr(clone(features).transformer_list[0][1].named_steps["tfidf"], "vocabulary_")


def test_context_and_empty_session_features_are_finite():
    first = Session(0, "b", ("Open(Screen)<Config>$Chain$", "Use", "t5"), "a")
    second = Session(1, "b", ("Use", "t0"), "b")
    assert "Screen" in views(first)["filled"][-1]
    assert "Screen" not in views(second)["filled"][0]
    features = build_features(
        (
            "raw",
            "action",
            "screen",
            "config",
            "chain",
            "pairs",
            "filled",
            "time",
            "position",
            "browser",
            "numeric",
        ),
        min_df=1,
    )
    matrix = features.fit_transform([first, second, Session(2, "b", (), "a")])
    assert np.isfinite(matrix.data).all()
