"""Phase 4 candidate tokenization, leakage and fusion regression tests."""

from dataclasses import replace

import joblib
import numpy as np

from copilote.io import Session
from copilote.phase4_candidates import (
    MarginFusion,
    MergedView,
    TempoView,
    hybrid_model,
    merged_model,
    signatures_model,
    tempo_model,
)

ROWS = [
    Session(i, "Chrome", (f"Open(screen{i % 3})", "Click", "Click"), str(i % 3)) for i in range(12)
]
LABELS = [row.label for row in ROWS]


def test_suffix_and_burst_tokens_follow_ticks():
    row = Session(0, "Chrome", ("t0", "Click", "Click1", "t5", "Open(screen)"), "a")
    tokens = TempoView(("suffix", "burst")).transform([row])[0]
    assert "sfx0|Click" in tokens
    assert "sfx1|Click" in tokens
    assert tokens.count("b2|Click") == 2
    assert "b1|Open" in tokens


def test_dwell_and_scalar_buckets_are_session_local():
    row = Session(0, "Chrome", ("Open(a)", "t0", "Open(a)", "t25", "Open(b)", "t35"), "a")
    tokens = TempoView(("dwell", "scalars")).transform([row])[0]
    assert "dwell|a|s1" in tokens
    assert "dwelltop|s1" in tokens
    assert "eps0" in tokens
    assert "sfxfrac0" in tokens
    assert "evtick0" in tokens


def test_marker_only_sessions_keep_a_present_token():
    row = Session(0, "Chrome", ("t0", "t5"), "a")
    assert TempoView(("suffix",)).transform([row]) == [["__present__"]]


def test_tempo_vocabulary_stays_fold_local_and_reloads(tmp_path):
    model = tempo_model(("suffix", "burst")).fit(ROWS, LABELS)
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


def test_predictions_ignore_heldout_labels():
    model = tempo_model(("suffix", "burst")).fit(ROWS, LABELS)
    heldout = Session(100, "Chrome", ("Open(screen1)", "Click1", "t0", "Click"), "poison")
    other = replace(heldout, label="other")
    np.testing.assert_array_equal(model.predict([heldout]), model.predict([other]))


def test_margin_fusion_combines_fold_local_components():
    fusion = MarginFusion(0.3).fit(ROWS, LABELS)
    expected = 0.7 * fusion.base_.decision_function(ROWS) + 0.3 * (
        fusion.signatures_.decision_function(ROWS)
    )
    np.testing.assert_allclose(fusion.decision_function(ROWS), expected)
    np.testing.assert_array_equal(
        fusion.predict(ROWS),
        fusion.base_.classes_[expected.argmax(axis=1)],
    )


def test_standardized_fusion_stays_finite():
    fusion = MarginFusion(0.5, standardize=True).fit(ROWS, LABELS)
    assert np.isfinite(fusion.decision_function(ROWS)).all()
    assert set(fusion.predict(ROWS)) <= set(LABELS)


def test_merged_view_emits_both_token_streams_once():
    row = Session(0, "Chrome", ("t0", "Click", "Click1", "t5", "Open(screen)"), "a")
    tokens = MergedView(("burst",)).transform([row])[0]
    assert tokens.count("__present__") == 1
    assert any(term.startswith('["Open"') for term in tokens)
    assert "b2|Click" in tokens


def test_hybrid_and_merged_models_fit_and_predict():
    for model in (hybrid_model(), merged_model(), signatures_model(0.2)):
        fitted = model.fit(ROWS, LABELS)
        assert set(fitted.predict(ROWS)) <= set(LABELS)
        assert fitted.predict([Session(99, "Chrome", ("Open(x)", "Click"), "poison")])[0] in set(
            LABELS
        )
