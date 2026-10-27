from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from copilote_b.config import load_config
from copilote_b.data import parse_line, split_record
from copilote_b.ensemble import blend, metrics
from copilote_b.features import prepare
from copilote_b.model import CropSVM, crop_plan
from copilote_b.pipeline import validate_submission


def test_parser_detects_labeled_and_browser_first_records() -> None:
    train = parse_line("abc,Firefox,Action(Screen)<Conf>$Chain$1,t5,t5", 1)
    test = parse_line("Firefox,Action(Screen)<Conf>$Chain$1,t5,t5", 2)
    assert train.label == "abc" and test.label is None
    assert train.actions == test.actions
    assert test.markers == (5, 5)
    assert prepare(train) == prepare(test)


def test_parser_protects_commas_and_csv_escaped_quotes() -> None:
    assert split_record('Firefox,"Action ""A""",Open(Screen,A)<Conf,B>$X,Y$,t5') == [
        "Firefox",
        'Action "A"',
        "Open(Screen,A)<Conf,B>$X,Y$",
        "t5",
    ]
    with pytest.raises(ValueError):
        parse_line("Firefox,Action(Broken,t5", 1)


def test_context_carries_state_and_resets_on_screen_changes() -> None:
    trace = prepare(parse_line("Firefox,Open(A)<C>$H$,Edit1,Open(B),Click,t5", 1))
    assert len(trace.full) == len(trace.context) == 4
    assert trace.context[1] == "s=A|a=Edit~c=C|a=Edit~h=H|a=Edit|m=1"
    assert trace.context[3] == "s=B|a=Click~c=NONE|a=Click~h=NONE|a=Click|m=0"
    assert prepare(parse_line("Firefox,t5", 1)).full == ()


def test_crop_plan_threshold_bounds_and_positions() -> None:
    assert crop_plan(192, 128) == [(0, 192)]
    assert crop_plan(340, 128) == [(0, 340), (0, 128), (106, 234), (212, 340)]
    assert crop_plan(0, 128) == [(0, 0)]


def test_training_vocabulary_and_browser_support_are_local() -> None:
    training = [
        prepare(parse_line(line, i))
        for i, line in enumerate(
            [
                "Firefox,Alpha(A),t5",
                "Firefox,Alpha(A),t10",
                "Opera,Beta(B),t5",
                "Opera,Beta(B),t10",
            ]
        )
    ]
    model = CropSVM({"length": 128}, 137).fit(training, np.array([2, 2, 9, 9]))
    validation = [
        prepare(parse_line("Firefox,VALIDATION_ONLY,t5", 1)),
        prepare(parse_line("Microsoft Edge,Alpha(A),t5", 2)),
    ]
    probability = model.predict_proba(validation)
    assert model.classes_.tolist() == [2, 9]
    assert probability[0].tolist() == [1.0, 0.0]
    assert (probability[1] > 0).all()
    assert "VALIDATION_ONLY" not in model.vectorizers["full"].vocabulary_


def test_weight_mass_is_conserved_per_session() -> None:
    traces = [
        prepare(parse_line("Firefox," + ",".join([token] * 340) + ",t5", i))
        for i, token in enumerate(["Alpha", "Beta", "Beta"])
    ]
    model = CropSVM({"length": 128})
    captured = {}

    class Recorder:
        def fit(self, x: object, labels: object, sample_weight: object) -> None:
            captured["weights"] = np.array(sample_weight)

    from unittest.mock import patch

    with patch("copilote_b.model.LinearSVC", return_value=Recorder()):
        model.fit(traces, np.array([0, 1, 1]))
    np.testing.assert_allclose(
        captured["weights"].reshape(3, 4).sum(axis=1), [1.5, 0.75, 0.75]
    )


def test_soft_vote_can_overrule_the_hard_majority() -> None:
    values = {
        "a": np.array([[0.51, 0.49]]),
        "b": np.array([[0.1, 0.9]]),
        "c": np.array([[0.51, 0.49]]),
    }
    result = blend(values, {"a": 0.4, "b": 0.5, "c": 0.1})
    assert result.argmax(axis=1).tolist() == [1]
    with pytest.raises(ValueError):
        blend(values, {"a": 1.0})


def test_metrics_use_encoded_targets_and_observed_classes() -> None:
    assert metrics(np.array([0, 1]), np.eye(3)[:2]) == {
        "macro": 1.0,
        "micro": 1.0,
        "weighted": 1.0,
    }


def test_submission_rejects_reordered_ids_and_unknown_labels() -> None:
    sample = pd.DataFrame({"RowId": [1, 2], "prediction": ["abc", "def"]})
    validate_submission(sample, sample, ["abc", "def"])
    with pytest.raises(ValueError):
        validate_submission(sample.iloc[::-1], sample, ["abc", "def"])
    with pytest.raises(ValueError):
        validate_submission(
            sample.assign(prediction=["abc", "bad"]), sample, ["abc", "def"]
        )


def test_frozen_config_is_exactly_the_submitted_blend() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs/solution.json")
    assert list(config["members"]) == [
        "crop128_noidf",
        "fusion_context",
        "reference128",
    ]
    np.testing.assert_allclose(
        list(config["weights"].values()),
        [0.422154785394683, 0.46445268572353543, 0.11339252888178171],
        rtol=0,
        atol=1e-15,
    )


def test_nested_partial_run_is_honest_and_resumable(tmp_path: Path) -> None:
    from unittest.mock import patch

    from copilote_b.evaluation import nested_validate

    traces = [
        prepare(parse_line(f"Firefox,{action},t5", i))
        for i, action in enumerate(["Alpha(A)"] * 10 + ["Beta(B)"] * 10)
    ]
    y = np.array([0] * 10 + [1] * 10)
    config = load_config(Path(__file__).resolve().parents[1] / "configs/solution.json")
    summary = nested_validate(
        traces, y, np.array(["aaa", "bbb"]), config, tmp_path, fold_limit=1
    )
    assert not summary["complete"]
    assert len(summary["folds"]) == 1
    assert summary["folds"][0]["blend"]["macro"] == 1.0
    assert (tmp_path / "nested_partial.json").exists()
    assert not (tmp_path / "nested.json").exists()
    with patch(
        "copilote_b.evaluation.fit_predict",
        side_effect=AssertionError("Cached fold was refitted"),
    ):
        assert (
            nested_validate(
                traces, y, np.array(["aaa", "bbb"]), config, tmp_path, fold_limit=1
            )
            == summary
        )
