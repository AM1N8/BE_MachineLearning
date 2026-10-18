"""Export the unpromoted phase 4 challenger (signatures 0.5 + burst 0.3).

The challenger keeps the verified champion untouched: it is fitted on the full
labelled training set only, never reads test labels (there are none), and is
exported to a separate submission file with explicit experimental status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from copilote.improvement import make_model
from copilote.io import file_hash, read_sessions, write_json
from copilote.phase2_models import behavioral_model
from copilote.phase4_candidates import hybrid_model

RESEARCH = Path("reports/results/phase4/85b76c4630ba")
PROTOCOLS = {
    "research": "85b76c4630ba",
    "confirm": "8492d196609f",
    "seed44": "951bd7da5015",
}
CSV = Path("submissions/phase4_challenger.csv")
MODEL = Path("artifacts/phase4_challenger.joblib")
SPEC = {
    "model": "sig_burst_w05",
    "signatures_weight": 0.5,
    "tempo_weight": 0.3,
    "tempo_parts": ["burst"],
}


def verify_sources() -> dict:
    """Require the evaluated sources, data and submissions to be unchanged."""
    manifest = json.loads((RESEARCH / "manifest.json").read_text())
    for name, digest in manifest["sources"].items():
        assert file_hash(f"src/copilote/{name}.py") == digest, name
    assert file_hash("data/train.csv") == manifest["train"]
    assert file_hash("uv.lock") == manifest["lock"]
    assert file_hash("submissions/best_verified.csv") == manifest["submissions"]["best_verified"]
    assert (
        file_hash("submissions/context_signatures_experimental.csv")
        == manifest["submissions"]["context_signatures_experimental"]
    )
    return manifest


def reproduce() -> None:
    """Refit baseline and challenger folds and require identical predictions."""
    verify_sources()
    rows = read_sessions("data/train.csv", labelled=True)
    y = np.array([row.label for row in rows])
    for stage, signature in PROTOCOLS.items():
        directory = Path("reports/results/phase4") / signature
        splits = json.loads((directory / f"{stage}_splits.json").read_text())
        for fold, split in enumerate(splits):
            a, b = split["train"], split["validation"]
            training = [rows[i] for i in a]
            with threadpool_limits(limits=2):
                challenger_model = hybrid_model(signatures_weight=0.5).fit(training, y[a])
                prediction = challenger_model.predict([rows[i] for i in b])
            prior = pd.read_csv(directory / f"{stage}_sig_burst_w05_{fold}.csv")
            assert prior.row_id.tolist() == b
            assert np.array_equal(prior.truth, y[b])
            assert np.array_equal(prior.prediction, prediction)
            print({"stage": stage, "fold": fold, "challenger_exact": True}, flush=True)
    print("all challenger fold predictions reproduced exactly")


def export() -> None:
    """Fit on the labelled training set only and export experimental artifacts."""
    verify_sources()
    protected = {
        p: file_hash(p)
        for p in (
            "submissions/best_verified.csv",
            "submissions/context_signatures_experimental.csv",
        )
    }
    rows = read_sessions("data/train.csv", labelled=True)
    test = read_sessions("data/test.csv", labelled=False)
    y = np.array([row.label for row in rows])
    manifest = {
        "kind": "challenger",
        "spec": SPEC,
        "validated_against": {
            stage: {
                "directory": f"reports/results/phase4/{signature}",
                "comparison_sha256": file_hash(
                    Path("reports/results/phase4")
                    / signature
                    / (
                        "comparison.csv"
                        if stage == "research"
                        else "confirmation.csv"
                        if stage == "confirm"
                        else "seed44_comparison.csv"
                    )
                ),
            }
            for stage, signature in PROTOCOLS.items()
        },
        "pooled_evidence_sha256": file_hash(RESEARCH / "pooled_evidence.json"),
        "protected_before": protected,
        "seed": 42,
        "transductive": False,
    }
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    output = Path("reports/results/phase4") / signature[:12]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", manifest)

    with threadpool_limits(limits=2):
        challenger = hybrid_model(signatures_weight=0.5).fit(rows, y)
        challenger_scores = challenger.decision_function(test)
        prediction = challenger.classes_[challenger_scores.argmax(axis=1)]
        baseline = make_model("baseline", y).fit(rows, y)
        baseline_scores = baseline.decision_function(test)
        signatures = behavioral_model("signatures").fit(rows, y)
        signatures_scores = signatures.decision_function(test)

    sample = pd.read_csv("data/sample_submission.csv")
    best = pd.read_csv("submissions/best_verified.csv")
    context = pd.read_csv("submissions/context_signatures_experimental.csv")
    assert sample.columns.tolist() == ["RowId", "prediction"]
    frame = pd.DataFrame({"RowId": sample.RowId, "prediction": prediction})
    assert frame.shape == (324, 2) and len(rows) == 3279 and len(test) == 324
    assert frame.RowId.equals(best.RowId) and frame.RowId.is_unique
    assert not frame.isna().any().any() and set(prediction) <= set(y)
    changed_baseline = prediction != best.prediction.to_numpy()
    changed_signatures = prediction != context.prediction.to_numpy()
    assert changed_baseline.any() and changed_signatures.any(), "duplicate of a prior submission"
    payload = frame.to_csv(index=False)
    if CSV.exists():
        assert CSV.read_text() == payload, "refuse overwriting a different challenger export"
    joblib.dump(challenger, MODEL)
    restored = joblib.load(MODEL)
    assert np.array_equal(restored.predict(test), prediction)
    assert np.array_equal(restored.decision_function(test), challenger_scores)
    CSV.write_text(payload)

    ranks = {
        "challenger": np.argsort(challenger_scores, axis=1),
        "baseline": np.argsort(baseline_scores, axis=1),
        "signatures": np.argsort(signatures_scores, axis=1),
    }
    index = np.arange(len(test))
    evidence = pd.DataFrame(
        {
            "RowId": sample.RowId,
            "best_verified": best.prediction,
            "context_signatures": context.prediction,
            "challenger": prediction,
            "changed_vs_best_verified": changed_baseline,
            "changed_vs_context_signatures": changed_signatures,
            "challenger_top2_gap": challenger_scores[index, ranks["challenger"][:, -1]]
            - challenger_scores[index, ranks["challenger"][:, -2]],
            "baseline_top2_gap": baseline_scores[index, ranks["baseline"][:, -1]]
            - baseline_scores[index, ranks["baseline"][:, -2]],
            "signatures_top2_gap": signatures_scores[index, ranks["signatures"][:, -1]]
            - signatures_scores[index, ranks["signatures"][:, -2]],
            "challenger_was_baseline_second": ranks["challenger"][:, -1]
            == ranks["baseline"][:, -2],
            "challenger_was_signatures_second": ranks["challenger"][:, -1]
            == ranks["signatures"][:, -2],
            "supported_classes": (challenger_scores > -1e5).sum(axis=1),
        }
    )
    evidence.to_csv(output / "test_margins.csv", index=False)
    evidence.loc[changed_baseline].to_csv(output / "changed_vs_best_verified.csv", index=False)
    evidence.loc[changed_signatures].to_csv(
        output / "changed_vs_context_signatures.csv", index=False
    )
    for path, digest in protected.items():
        assert file_hash(path) == digest
    write_json(
        output / "challenger.json",
        {
            "status": "EXPERIMENTAL_NOT_PROMOTED",
            "spec": SPEC,
            "output_directory": str(output),
            "train_rows": len(rows),
            "test_rows": len(test),
            "changed_vs_best_verified": int(changed_baseline.sum()),
            "changed_vs_context_signatures": int(changed_signatures.sum()),
            "protected_after": protected,
            "csv_sha256": file_hash(CSV),
            "model_sha256": file_hash(MODEL),
            "model_reload_exact": True,
            "transductive": False,
            "automatic_submission": False,
            "promotion_criteria_changed": False,
        },
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "changed_vs_best_verified": int(changed_baseline.sum()),
                "changed_vs_context_signatures": int(changed_signatures.sum()),
            }
        )
    )


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["reproduce", "export"])
    arguments = parser.parse_args()
    reproduce() if arguments.stage == "reproduce" else export()


if __name__ == "__main__":
    main()
