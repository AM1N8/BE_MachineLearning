"""Second bounded hypothesis: remove browser offsets or relax intercept regularization."""

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from threadpoolctl import threadpool_limits

from copilote.improvement import ART, ROOT, SPLITS, make_model, metrics, summarize
from copilote.io import file_hash, read_sessions, write_json


def run():
    """Use fixed candidates and confirm both on a distinct split seed, without tuning."""
    active = json.loads((ROOT / "active.json").read_text())
    directory = ROOT / active["signature"][:12]
    rows = read_sessions("data/train.csv", labelled=True)
    y = np.array([r.label for r in rows])
    names = ["baseline", "event_only_mask", "intercept10", "rare_binary"]
    write_json(
        directory / "boundary_manifest.json",
        {
            "source_sha256": file_hash(__file__),
            "parent_signature": active["signature"],
            "hypotheses": names,
            "seeds": [191664964, 43],
            "rationale": "Pair-local losses suggest preserving global boundaries; test browser offsets and intercept shrinkage. Rare-binary accuracy gain merits independent checking because official F1 averaging is unknown.",
        },
    )
    for prefix in ("research", "confirm"):
        splits = (
            [
                (np.array(f["train"]), np.array(f["validation"]))
                for f in json.loads(SPLITS.read_text())["folds"]
            ]
            if prefix == "research"
            else list(RepeatedStratifiedKFold(n_splits=4, n_repeats=2, random_state=43).split(y, y))
        )
        write_json(
            directory / f"{prefix}_splits.json",
            [{"train": a.tolist(), "validation": b.tolist()} for a, b in splits],
        )
        for fold, (a, b) in enumerate(splits):
            training, validation = [rows[i] for i in a], [rows[i] for i in b]
            for name in names:
                path = directory / f"{prefix}_{name}_{fold}.csv"
                if path.exists():
                    continue
                with threadpool_limits(limits=2):
                    start = time.perf_counter()
                    if name == "event_only_mask":
                        model = make_model("baseline", y[a])
                        model.estimator.named_steps["features"].transformer_list = [
                            item
                            for item in model.estimator.named_steps["features"].transformer_list
                            if item[0] != "browser"
                        ]
                    else:
                        model = make_model(name, y[a])
                    if name == "intercept10":
                        model.estimator.set_params(model__intercept_scaling=10)
                    model.fit(training, y[a])
                    predicted = model.predict(validation)
                    seconds = time.perf_counter() - start
                pd.DataFrame(
                    {"row_id": b, "truth": y[b], "prediction": predicted, "fold": fold}
                ).to_csv(path, index=False)
                result = {
                    "experiment": name,
                    "fold": fold,
                    "seconds": seconds,
                    **metrics(y[b], predicted),
                }
                write_json(path.with_suffix(".json"), result)
                if prefix == "confirm" and name == "baseline":
                    joblib.dump(model, ART / f"confirmation_baseline_{fold}.joblib")
                print(prefix, result, flush=True)
        summarize(
            directory,
            names + (["sqrt_balance", "pair_local"] if prefix == "research" else []),
            prefix,
        )
        if prefix == "research":
            pd.read_csv(directory / "comparison.csv").to_csv(
                directory / "boundary_comparison.csv", index=False
            )


if __name__ == "__main__":
    run()
