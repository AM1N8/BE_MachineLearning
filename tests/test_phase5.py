"""Statistical audit invariants: metric semantics and dependence correction."""

import numpy as np
from sklearn.metrics import f1_score

from copilote.phase5_audit import corrected_contrast, holm, macro_f1, make_splits


def test_fast_macro_matches_sklearn_with_missing_and_predicted_only_classes():
    truth = np.array([0, 0, 1, 3, 3])
    prediction = np.array([0, 2, 1, 2, 3])
    weights = np.array([2, 0, 3, 1, 0])
    assert np.isclose(
        macro_f1(truth, prediction), f1_score(truth, prediction, average="macro", zero_division=0)
    )
    # sklearn includes zero-weight labels if explicitly present in arrays; the
    # weighted bootstrap instead drops zero-weight rows, matching a literal sample.
    chosen = weights > 0
    assert np.isclose(
        macro_f1(truth, prediction, weights),
        f1_score(
            truth[chosen],
            prediction[chosen],
            sample_weight=weights[chosen],
            average="macro",
            zero_division=0,
        ),
    )


def test_holm_monotone_and_corrected_se_retains_overlap_term():
    np.testing.assert_allclose(holm([0.04, 0.001, 0.02]), [0.04, 0.003, 0.04])
    deltas = np.array([0.01, -0.005, 0.02, 0.003])
    result = corrected_contrast(deltas, 0.1)
    assert np.isclose(result["corrected_se"], deltas.std(ddof=1) * np.sqrt(0.25 + 0.1))
    assert result["lower95"] < result["delta"] < result["upper95"]


def test_similarity_splits_are_closed_set_and_group_disjoint():
    y = np.repeat(np.arange(5), 100)
    groups = np.arange(len(y)) // 2
    splits = make_splits(y, groups)
    for split in splits:
        a, b = np.array(split["train"]), np.array(split["validation"])
        assert not set(a) & set(b)
        assert set(y[b]) <= set(y[a])
        if split["kind"] == "similarity_group":
            assert not set(groups[a]) & set(groups[b])
