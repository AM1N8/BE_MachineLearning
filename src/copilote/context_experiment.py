"""Reproduce and export the explicitly authorized, unpromoted signatures model."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from copilote.improvement import metrics
from copilote.io import file_hash, read_sessions, write_json
from copilote.phase2_models import behavioral_model

SOURCE = Path("reports/results/phase2/a1d191a9b558")
OUTPUT = Path("reports/results/context_signatures_experimental")
CSV = Path("submissions/context_signatures_experimental.csv")
MODEL = Path("artifacts/context_signatures_experimental.joblib")


def verify_sources():
    """Require the evaluated sources, data and baseline to remain unchanged."""
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    for name, digest in manifest["sources"].items():
        assert file_hash(f"src/copilote/{name}.py") == digest, name
    assert file_hash("data/train.csv") == manifest["train"]
    assert file_hash("uv.lock") == manifest["lock"]
    assert file_hash("submissions/best_verified.csv") == manifest["baseline_submission"]
    return manifest


def reproduce():
    """Refit every original signatures fold and require identical predictions."""
    manifest = verify_sources()
    rows = read_sessions("data/train.csv", labelled=True)
    y = np.array([row.label for row in rows])
    results = []
    for stage in ("research", "confirm", "seed44"):
        splits = json.loads((SOURCE / f"{stage}_splits.json").read_text())
        for fold, split in enumerate(splits):
            a, b = split["train"], split["validation"]
            assert not set(a) & set(b)
            with threadpool_limits(limits=2):
                model = behavioral_model("signatures").fit([rows[i] for i in a], y[a])
                prediction = model.predict([rows[i] for i in b])
            prior = pd.read_csv(SOURCE / f"{stage}_signatures_{fold}.csv")
            assert prior.row_id.tolist() == b
            assert np.array_equal(prior.truth, y[b])
            assert np.array_equal(prior.prediction, prediction)
            record = dict(stage=stage, fold=fold, exact=True, **metrics(y[b], prediction))
            results.append(record)
            print(record, flush=True)
    frame = pd.DataFrame(results)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "cv_reproduced.csv", index=False)
    frame.groupby("stage")[["macro_f1", "accuracy", "weighted_f1"]].mean().to_csv(
        OUTPUT / "cv_summary.csv"
    )
    write_json(
        OUTPUT / "cv_verification.json",
        dict(
            folds=24,
            all_predictions_exact=True,
            manifest=manifest,
            exporter_sha256=file_hash(__file__),
        ),
    )


def export():
    """Fit only on labeled sessions and export separate experimental artifacts."""
    manifest = verify_sources()
    verified = json.loads((OUTPUT / "cv_verification.json").read_text())
    assert verified["manifest"] == manifest and verified["all_predictions_exact"]
    protected = {
        p: file_hash(p)
        for p in (
            "submissions/best_verified.csv",
            "artifacts/best_verified.joblib",
            "submissions/competitive.csv",
            "artifacts/competitive_final.joblib",
        )
    }
    rows = read_sessions("data/train.csv", labelled=True)
    test = read_sessions("data/test.csv", labelled=False)
    y = np.array([row.label for row in rows])
    with threadpool_limits(limits=2):
        model = behavioral_model("signatures").fit(rows, y)
        scores = model.decision_function(test)
        prediction = model.classes_[scores.argmax(axis=1)]
        baseline = joblib.load("artifacts/best_verified.joblib")
        base_scores = baseline.decision_function(test)
        assert np.array_equal(model.classes_, baseline.classes_)
        sample = pd.read_csv("data/sample_submission.csv")
        original = pd.read_csv("submissions/best_verified.csv")
        assert np.array_equal(baseline.predict(test), original.prediction)
        assert sample.columns.tolist() == ["RowId", "prediction"]
        frame = pd.DataFrame({"RowId": sample.RowId, "prediction": prediction})
        assert frame.shape == (324, 2) and len(rows) == 3279 and len(test) == 324
        assert frame.RowId.equals(original.RowId) and frame.RowId.is_unique
        assert not frame.isna().any().any() and set(prediction) <= set(y)
        changed = prediction != original.prediction.to_numpy()
        assert changed.any(), "No different predictions: do not export a duplicate."
        payload = frame.to_csv(index=False)
        if CSV.exists():
            assert CSV.read_text() == payload, "Refuse overwriting a different experiment."
        joblib.dump(model, MODEL)
        restored = joblib.load(MODEL)
        assert np.array_equal(restored.predict(test), prediction)
        assert np.array_equal(restored.decision_function(test), scores)
        CSV.write_text(payload)
    ranks, base_ranks = np.argsort(scores, axis=1), np.argsort(base_scores, axis=1)
    index = np.arange(len(test))
    baseline_winner, challenger_winner = base_ranks[:, -1], ranks[:, -1]
    evidence = pd.DataFrame(
        {
            "RowId": sample.RowId,
            "baseline": original.prediction,
            "challenger": prediction,
            "changed": changed,
            "baseline_top2_gap": base_scores[index, base_ranks[:, -1]]
            - base_scores[index, base_ranks[:, -2]],
            "challenger_top2_gap": scores[index, ranks[:, -1]] - scores[index, ranks[:, -2]],
            "baseline_advantage_over_challenger": base_scores[index, baseline_winner]
            - base_scores[index, challenger_winner],
            "challenger_advantage_over_baseline": scores[index, challenger_winner]
            - scores[index, baseline_winner],
            "challenger_was_baseline_second": challenger_winner == base_ranks[:, -2],
            "supported_classes": (scores > -1e5).sum(axis=1),
        }
    )
    evidence.to_csv(OUTPUT / "test_margins.csv", index=False)
    evidence.loc[changed].to_csv(OUTPUT / "changed_predictions.csv", index=False)
    for path, digest in protected.items():
        assert file_hash(path) == digest
    write_json(
        OUTPUT / "export.json",
        dict(
            status="EXPERIMENTAL_NOT_PROMOTED",
            configuration="behavioral_model('signatures')",
            source_directory=str(SOURCE),
            train_rows=len(rows),
            test_rows=len(test),
            changed=int(changed.sum()),
            protected=protected,
            csv_sha256=file_hash(CSV),
            model_sha256=file_hash(MODEL),
            test_sha256=file_hash("data/test.csv"),
            sample_sha256=file_hash("data/sample_submission.csv"),
            model_reload_exact=True,
            transductive=False,
            automatic_submission=False,
            promotion_criteria_changed=False,
        ),
    )
    print(evidence.loc[changed].to_string(index=False))
    print((OUTPUT / "cv_summary.csv").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["reproduce", "export"])
    args = parser.parse_args()
    reproduce() if args.stage == "reproduce" else export()
