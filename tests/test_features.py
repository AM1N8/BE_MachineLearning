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


def test_descriptive_analysis_exports_finite_artifacts(tmp_path, monkeypatch):
    """Exercise the plotting path with enough categorical users for course boxplots."""
    import json
    from pathlib import Path

    from copilote.eda import run_eda
    from copilote.features import run_feature_analysis
    from copilote.settings import load_config

    config = load_config("configs/default.yaml")
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    train.write_text(
        "\n".join(
            f"u{k:03},browser{k % 2},start,action{k}(screen{k}),t{5 * (i + 1)},end"
            for k in range(200)
            for i in range(3)
        )
        + "\n"
    )
    test.write_text("browser0,start,unseen(screen),t15,end\n")
    config["paths"].update(train=str(train), test=str(test))
    monkeypatch.chdir(tmp_path)
    Path("reports/results").mkdir(parents=True)
    eda = run_eda(config)
    features = run_feature_analysis(config)
    assert eda["users_with_multiple_browsers"] == 0
    assert features["sparse"]
    assert features["matrix_shape"][0] == 600
    assert Path("reports/figures/feature_projection.png").is_file()
    assert (
        json.loads(Path("reports/results/features.json").read_text())["categorical_code_mapping"][
            "199"
        ]
        == "u199"
    )
