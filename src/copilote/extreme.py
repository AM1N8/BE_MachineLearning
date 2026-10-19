"""Budgeted Optuna exploration and nested evaluation of a complete selection policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from scipy.stats import t
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from copilote.extreme_audit import freeze
from copilote.extreme_models import build
from copilote.io import file_hash, read_sessions, write_json
from copilote.predict import validate_submission
from copilote.settings import load_config

ROOT = Path("reports/results/extreme")
ART = Path("artifacts/extreme")


def metrics(y, predicted):
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
    }


def normalized(scores):
    """Scale within permitted classes; browser masks must never enter moments."""
    valid = scores > -1e5
    count = valid.sum(axis=1, keepdims=True).clip(1)
    mean = np.where(valid, scores, 0).sum(axis=1, keepdims=True) / count
    deviation = np.sqrt(
        np.where(valid, (scores - mean) ** 2, 0).sum(axis=1, keepdims=True) / count
    ).clip(0.01)
    return np.where(valid, (scores - mean) / deviation, -1e6)


class MarginBlend(ClassifierMixin, BaseEstimator):
    """A frozen weight learned exclusively from inner out-of-fold predictions."""

    def __init__(self, spec, weight):
        self.spec = spec
        self.weight = weight

    def fit(self, X, y):
        self.champion_ = build({"method": "champion"}).fit(X, y)
        self.candidate_ = build(self.spec).fit(X, y)
        self.classes_ = self.champion_.classes_
        np.testing.assert_array_equal(self.classes_, self.candidate_.classes_)
        return self

    def decision_function(self, X):
        return (1 - self.weight) * normalized(
            self.champion_.decision_function(X)
        ) + self.weight * normalized(self.candidate_.decision_function(X))

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]


class Campaign:
    """Immutable split/spec cache keys; all validation fits start from unfitted models."""

    def __init__(self, config):
        self.config = config
        self.rows = read_sessions("data/train.csv", labelled=True)
        self.y = np.array([r.label for r in self.rows])
        self.classes = np.unique(self.y)
        self.fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "train": file_hash("data/train.csv"),
                    "models": file_hash("src/copilote/extreme_models.py"),
                    "runner": file_hash(__file__),
                    "lock": file_hash("uv.lock"),
                    "config": config,
                    "dependencies": {
                        name: file_hash(f"src/copilote/{name}.py")
                        for name in ("competitive", "features", "tokens", "io")
                    },
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        ROOT.mkdir(parents=True, exist_ok=True)
        ART.mkdir(parents=True, exist_ok=True)
        self.directory = ROOT / self.fingerprint[:12]
        self.directory.mkdir(exist_ok=True)
        write_json(
            ROOT / "active_run.json",
            {"signature": self.fingerprint, "directory": str(self.directory), "config": config},
        )

    def evaluate(self, name, spec, indices, splits, stage, trial=None):
        key = hashlib.sha256(
            json.dumps(
                {
                    "signature": self.fingerprint,
                    "spec": spec,
                    "indices": indices.tolist(),
                    "splits": [(a.tolist(), b.tolist()) for a, b in splits],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:20]
        result_path = self.directory / f"{stage}_{name}_{key}.json"
        if result_path.exists():
            return json.loads(result_path.read_text())
        frames, fold_records, all_scores = [], [], []
        started = time.perf_counter()
        for fold, (a, b) in enumerate(splits):
            train_ids, val_ids = indices[a], indices[b]
            cache = ART / f"fold_{key}_{fold}.joblib"
            if cache.exists():
                record = joblib.load(cache)
            else:
                tick = time.perf_counter()
                with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model = build(spec).fit([self.rows[i] for i in train_ids], self.y[train_ids])
                    np.testing.assert_array_equal(model.classes_, self.classes)
                    scores = model.decision_function([self.rows[i] for i in val_ids])
                predicted = self.classes[scores.argmax(axis=1)]
                record = {
                    "scores": scores.astype(np.float32),
                    "prediction": predicted,
                    "metrics": metrics(self.y[val_ids], predicted),
                    "seconds": time.perf_counter() - tick,
                    "warnings": [str(w.message) for w in caught],
                }
                joblib.dump(record, cache)
            fold_records.append(
                {
                    "fold": fold,
                    **record["metrics"],
                    "seconds": record["seconds"],
                    "warnings": record["warnings"],
                }
            )
            frames.append(
                pd.DataFrame(
                    {
                        "fold": fold,
                        "row_id": val_ids,
                        "truth": self.y[val_ids],
                        "prediction": record["prediction"],
                    }
                )
            )
            all_scores.append(record["scores"])
            if trial is not None:
                trial.report(float(np.mean([f["macro_f1"] for f in fold_records])), fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        frame = pd.concat(frames, ignore_index=True)
        frame.to_csv(result_path.with_suffix(".csv"), index=False)
        score_path = ART / f"scores_{key}.npz"
        np.savez_compressed(
            score_path,
            scores=np.concatenate(all_scores),
            row_ids=frame.row_id.to_numpy(),
            classes=self.classes,
        )
        result = {
            "name": name,
            "spec": spec,
            "stage": stage,
            "signature": self.fingerprint,
            "mean": {
                m: float(np.mean([f[m] for f in fold_records])) for m in ("accuracy", "macro_f1")
            },
            "std": {
                m: float(np.std([f[m] for f in fold_records], ddof=1))
                if len(fold_records) > 1
                else 0.0
                for m in ("accuracy", "macro_f1")
            },
            "folds": fold_records,
            "wall_seconds": time.perf_counter() - started,
            "fit_seconds": sum(f["seconds"] for f in fold_records),
            "scores": str(score_path),
            "predictions": str(result_path.with_suffix(".csv")),
            "official_metric": self.config["objective_status"],
        }
        write_json(result_path, result)
        print(
            f"{stage}/{name}: F1={result['mean']['macro_f1']:.6f}, accuracy={result['mean']['accuracy']:.6f}",
            flush=True,
        )
        return result

    def search(self, indices, stage, budget, names):
        splits = list(
            StratifiedKFold(3, shuffle=True, random_state=42).split(indices, self.y[indices])
        )
        results = []
        failures = []
        for name in names:
            try:
                results.append(
                    self.evaluate(name, self.config["candidates"][name], indices, splits, stage)
                )
            except (ValueError, MemoryError) as error:
                failures.append({"name": name, "error": repr(error)})
                write_json(self.directory / f"{stage}_failures.json", failures)
                if name == "champion":
                    raise
        sampler_path = ART / f"sampler_{self.fingerprint[:12]}_{stage}.joblib"
        sampler = (
            joblib.load(sampler_path)
            if sampler_path.exists()
            else optuna.samplers.TPESampler(seed=42, n_startup_trials=6)
        )
        study = optuna.create_study(
            direction="maximize",
            study_name=stage,
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=1),
            storage=f"sqlite:///{ART / ('optuna_' + self.fingerprint[:12] + '.db')}",
            load_if_exists=True,
        )

        def objective(trial):
            view = trial.suggest_categorical(
                "view", ["separated", "skip", "edge", "rhythm", "repeat"]
            )
            spec = {
                "C": trial.suggest_float("C", 0.3, 20, log=True),
                "min_df": trial.suggest_categorical("min_df", [1, 2, 3]),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
            }
            if view == "separated":
                spec.update(
                    raw=0,
                    unigram=1,
                    bigram=trial.suggest_float("bigram", 0.2, 2, log=True),
                    trigram=trial.suggest_float("trigram", 0.05, 0.8, log=True),
                )
            else:
                spec[view] = trial.suggest_float("weight", 0.05, 1, log=True)
            trial.set_user_attr("spec", spec)
            result = self.evaluate(f"trial{trial.number:03d}", spec, indices, splits, stage, trial)
            trial.set_user_attr("result", result)
            return result["mean"]["macro_f1"]

        def persist(study, trial):
            joblib.dump(study.sampler, sampler_path)
            study.trials_dataframe().to_csv(
                self.directory / f"{stage}_optuna_trials.csv", index=False
            )

        remaining = max(0, budget - len(study.trials))
        study.optimize(
            objective, n_trials=remaining, callbacks=[persist], catch=(ValueError, MemoryError)
        )
        results.extend(
            trial.user_attrs["result"]
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        )
        results.sort(key=lambda r: -r["mean"]["macro_f1"])
        write_json(self.directory / f"{stage}_ranking.json", results)
        return results, splits

    def broad(self):
        indices, _ = train_test_split(
            np.arange(len(self.rows)), test_size=0.2, stratify=self.y, random_state=42
        )
        self.search(indices, "exploration", self.config["trials"], list(self.config["candidates"]))

    def inner_policy(self, indices, stage):
        ranking, _ = self.search(
            indices, stage, self.config["nested_trials"], self.config["nested_candidates"]
        )
        chosen = ranking[0]
        champion = next(r for r in ranking if r["name"] == "champion")
        a, b = np.load(champion["scores"]), np.load(chosen["scores"])
        np.testing.assert_array_equal(a["row_ids"], b["row_ids"])
        labels = self.y[a["row_ids"]]
        weights = []
        for weight in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
            scores = (1 - weight) * normalized(a["scores"]) + weight * normalized(b["scores"])
            weights.append(
                {"weight": weight, **metrics(labels, self.classes[scores.argmax(axis=1)])}
            )
        # These are tuning scores, never presented as independently validated ensemble results.
        weight = max(weights, key=lambda r: (r["macro_f1"], -r["weight"]))["weight"]
        policy = {
            "chosen": chosen["name"],
            "spec": chosen["spec"],
            "weight": weight,
            "inner_weight_tuning": weights,
            "train_ids": indices.tolist(),
            "signature": self.fingerprint,
        }
        write_json(self.directory / f"{stage}_policy.json", policy)
        return policy

    def nested(self):
        # Derived from master seed, distinct from all historical split seeds.
        seed = int(np.random.default_rng(42).integers(1, 2**31 - 1))
        splits = list(
            RepeatedStratifiedKFold(n_splits=4, n_repeats=2, random_state=seed).split(
                np.arange(len(self.rows)), self.y
            )
        )
        write_json(
            self.directory / "outer_splits.json",
            {
                "seed": seed,
                "folds": [{"train": a.tolist(), "validation": b.tolist()} for a, b in splits],
            },
        )
        records, frames = [], []
        for fold, (a, b) in enumerate(splits):
            path = self.directory / f"outer{fold}_evaluation.json"
            csv = path.with_suffix(".csv")
            if path.exists():
                records.append(json.loads(path.read_text()))
                frames.append(pd.read_csv(csv))
                continue
            policy = self.inner_policy(a, f"outer{fold}_inner")
            with threadpool_limits(limits=2):
                model = MarginBlend(policy["spec"], policy["weight"]).fit(
                    [self.rows[i] for i in a], self.y[a]
                )
                rows = [self.rows[i] for i in b]
                champion = model.champion_.predict(rows)
                candidate = model.candidate_.predict(rows)
                ensemble = model.predict(rows)
            record = {
                "fold": fold,
                "repeat": fold // 4,
                "policy": policy,
                "champion": metrics(self.y[b], champion),
                "candidate": metrics(self.y[b], candidate),
                "ensemble": metrics(self.y[b], ensemble),
            }
            frame = pd.DataFrame(
                {
                    "fold": fold,
                    "row_id": b,
                    "truth": self.y[b],
                    "champion": champion,
                    "candidate": candidate,
                    "ensemble": ensemble,
                }
            )
            frame.to_csv(csv, index=False)
            write_json(path, record)
            records.append(record)
            frames.append(frame)
            print(f"OUTER {fold}: {record['champion']} => {record['ensemble']}", flush=True)
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(self.directory / "nested_predictions.csv", index=False)
        summary = {
            "folds": records,
            "signature": self.fingerprint,
            "outer_seed": seed,
            "protocol": "4x2 outer; 3 inner; all trial selection and blend tuning confined to outer training",
        }
        for model in ("champion", "candidate", "ensemble"):
            summary[model] = {
                m: {
                    "mean": float(np.mean([r[model][m] for r in records])),
                    "std": float(np.std([r[model][m] for r in records], ddof=1)),
                }
                for m in ("accuracy", "macro_f1")
            }
            report = classification_report(
                combined.truth, combined[model], output_dict=True, zero_division=0
            )
            write_json(self.directory / f"nested_per_class_{model}.json", report)
        differences = np.array(
            [r["ensemble"]["macro_f1"] - r["champion"]["macro_f1"] for r in records]
        )
        corrected_se = np.sqrt(1 / len(differences) + 1 / 3) * differences.std(ddof=1)
        lower = float(differences.mean() - t.ppf(0.975, len(differences) - 1) * corrected_se)
        summary["paired"] = {
            "macro_f1_gain": float(differences.mean()),
            "std": float(differences.std(ddof=1)),
            "corrected_95_lower": lower,
            "improved_folds": int((differences > 0).sum()),
            "repeat_gains": differences.reshape(2, 4).mean(axis=1).tolist(),
            "uncertainty": "Nadeau-Bengio-style correction; approximate, not a fresh-data guarantee",
        }
        summary["promote"] = bool(lower > 0 and (differences.reshape(2, 4).mean(axis=1) > 0).all())
        write_json(self.directory / "nested_validation.json", summary)

    def predict(self):
        validation = json.loads((self.directory / "nested_validation.json").read_text())
        policy = self.inner_policy(np.arange(len(self.rows)), "final_inner")
        test = read_sessions("data/test.csv", labelled=False)
        with threadpool_limits(limits=2):
            model = MarginBlend(policy["spec"], policy["weight"]).fit(self.rows, self.y)
            predictions = model.predict(test)
        model_path = ART / f"candidate_{self.fingerprint[:12]}.joblib"
        joblib.dump(model, model_path)
        np.testing.assert_array_equal(joblib.load(model_path).predict(test), predictions)
        frame = pd.DataFrame({"RowId": np.arange(1, len(test) + 1), "prediction": predictions})
        validate_submission(frame, load_config("configs/default.yaml"), set(self.y), len(test))
        sample = pd.read_csv("data/sample_submission.csv")
        np.testing.assert_array_equal(frame.RowId, sample.RowId)
        for path, chosen_frame in (
            (Path("submissions/extreme_candidate.csv"), frame),
            (
                Path("submissions/best_verified.csv"),
                frame
                if validation["promote"]
                else pd.read_csv("artifacts/extreme/frozen/submissions__competitive.csv"),
            ),
        ):
            if path.exists() and not pd.read_csv(path).equals(chosen_frame):
                raise ValueError(f"Refusing to replace different existing submission: {path}")
            chosen_frame.to_csv(path, index=False)
        freeze()
        write_json(
            self.directory / "submission.json",
            {
                "policy": policy,
                "promoted": validation["promote"],
                "model": str(model_path),
                "candidate_sha256": file_hash("submissions/extreme_candidate.csv"),
                "best_verified_sha256": file_hash("submissions/best_verified.csv"),
                "train_sha256": file_hash("data/train.csv"),
                "test_sha256": file_hash("data/test.csv"),
                "signature": self.fingerprint,
                "official_score": None,
                "transductive": False,
                "model_reload_exact": True,
                "rows": len(frame),
            },
        )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["explore", "nested", "predict", "all"])
    parser.add_argument("--config", default="configs/extreme.yaml")
    args = parser.parse_args()
    freeze()
    campaign = Campaign(yaml.safe_load(Path(args.config).read_text()))
    for stage in ["explore", "nested", "predict"] if args.stage == "all" else [args.stage]:
        {"explore": campaign.broad, "nested": campaign.nested, "predict": campaign.predict}[stage]()


if __name__ == "__main__":
    # Keep joblib classes importable from a clean inference process.
    from copilote.extreme import main as entrypoint

    entrypoint()
