"""Bounded, paired research with fold-local diagnostics and fixed hypotheses."""

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from threadpoolctl import threadpool_limits

from copilote.competitive import EventStream, build_competitive
from copilote.features import identity
from copilote.io import file_hash, read_sessions, write_json

ROOT = Path("reports/results/improvement")
ART = Path("artifacts/improvement")
SPLITS = Path("reports/results/extreme/1ba1ee389f73/outer_splits.json")
NAMES = ("baseline", "sqrt_balance", "pair_local", "rare_binary")


def make_model(name, y):
    """Construct a fixed hypothesis; class counts come exclusively from its training fold."""
    weights = "balanced"
    if name == "sqrt_balance":
        counts = Counter(y)
        weights = {label: np.sqrt(len(y) / (len(counts) * n)) for label, n in counts.items()}
    model = build_competitive({"C": 3, "class_weight": weights, "restrict_browser": True})
    if name == "rare_binary":
        rare = Pipeline(
            [
                ("stream", EventStream()),
                (
                    "tfidf",
                    TfidfVectorizer(
                        tokenizer=identity,
                        token_pattern=None,
                        lowercase=False,
                        ngram_range=(1, 2),
                        min_df=1,
                        max_features=180000,
                        binary=True,
                    ),
                ),
            ]
        )
        model.estimator.set_params(
            features=FeatureUnion(
                [("original", model.estimator.named_steps["features"]), ("rare", rare)],
                transformer_weights={"original": 1, "rare": 0.2},
            )
        )
    return model


def pair_predict(model, training, y, validation, scores):
    """Resolve ambiguous top-two classes with local training-only TF-IDF and binary SVM."""
    ranks = np.argsort(scores, axis=1)
    predicted = model.classes_[ranks[:, -1]].copy()
    groups = {}
    for i, rank in enumerate(ranks):
        if scores[i, rank[-1]] - scores[i, rank[-2]] < 0.3:
            pair = tuple(sorted(model.classes_[rank[-2:]]))
            groups.setdefault(pair, []).append(i)
    for pair, indices in groups.items():
        mask = np.isin(y, pair)
        local = build_competitive({"C": 1, "class_weight": "balanced", "min_df": 1})
        local.fit([row for row, keep in zip(training, mask, strict=True) if keep], y[mask])
        predicted[indices] = local.predict([validation[i] for i in indices])
    return predicted


def metrics(truth, predicted):
    return {
        "accuracy": float(np.mean(truth == predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predicted, average="weighted", zero_division=0)),
    }


def diagnostics(training, validation, scores, classes):
    """Use training-only event document frequencies and transition support."""
    stream = EventStream()
    train_events = stream.transform(training)
    df = Counter(e for events in train_events for e in set(events))
    transitions = {pair for events in train_events for pair in zip(events, events[1:])}
    records = []
    for i, (row, events) in enumerate(zip(validation, stream.transform(validation), strict=True)):
        counts = np.array(list(Counter(events).values()), dtype=float)
        probabilities = counts / counts.sum()
        ranked = np.argsort(scores[i])
        records.append(
            {
                "row_id": row.row_id,
                "browser": row.browser,
                "length": len(events),
                "unique": len(counts),
                "entropy": float(-sum(probabilities * np.log2(probabilities))),
                "repetition": 1 - len(counts) / len(events),
                "oov": sum(e not in df for e in events) / len(events),
                "rare": sum(df[e] <= 2 for e in events) / len(events),
                "unseen_transition": sum(p not in transitions for p in zip(events, events[1:]))
                / max(1, len(events) - 1),
                "margin": float(scores[i, ranked[-1]] - scores[i, ranked[-2]]),
                "second": classes[ranked[-2]],
            }
        )
    return records


def run(stage):
    ROOT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    rows = read_sessions("data/train.csv", labelled=True)
    y = np.array([r.label for r in rows])
    signature = hashlib.sha256(
        json.dumps(
            {
                "source": file_hash(__file__),
                "train": file_hash("data/train.csv"),
                "lock": file_hash("uv.lock"),
                "splits": file_hash(SPLITS),
                "dependencies": {
                    name: file_hash(f"src/copilote/{name}.py")
                    for name in ("competitive", "features", "io", "tokens")
                },
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    directory = ROOT / signature[:12]
    directory.mkdir(exist_ok=True)
    write_json(
        ROOT / "active.json",
        {
            "directory": str(directory),
            "signature": signature,
            "python": platform.python_version(),
            "seed": 42,
            "confirmation_seed": 43,
            "metric_verified": False,
            "public_score_user_reported": 0.980,
        },
    )
    splits = [
        (np.array(f["train"]), np.array(f["validation"]))
        for f in json.loads(SPLITS.read_text())["folds"]
    ]
    names = ["baseline"] if stage == "analyze" else list(NAMES)
    if stage == "confirm":
        prior = pd.read_csv(directory / "comparison.csv")
        names = ["baseline"] + prior.loc[
            (prior.delta_f1 > 0) & (prior.experiment != "baseline"), "experiment"
        ].tolist()
        splits = list(RepeatedStratifiedKFold(n_splits=4, n_repeats=2, random_state=43).split(y, y))
    prefix = "confirm" if stage == "confirm" else "research"
    write_json(
        directory / f"{prefix}_splits.json",
        [{"train": a.tolist(), "validation": b.tolist()} for a, b in splits],
    )
    for fold, (a, b) in enumerate(splits):
        training, validation = [rows[i] for i in a], [rows[i] for i in b]
        cache = ART / f"{signature[:12]}_{prefix}_{fold}.joblib"
        with threadpool_limits(limits=2):
            if cache.exists():
                base, elapsed = joblib.load(cache)
            else:
                start = time.perf_counter()
                base = make_model("baseline", y[a]).fit(training, y[a])
                elapsed = time.perf_counter() - start
                joblib.dump((base, elapsed), cache)
            scores = base.decision_function(validation)
            for name in names:
                path = directory / f"{prefix}_{name}_{fold}.csv"
                if path.exists():
                    continue
                start = time.perf_counter()
                if name == "baseline":
                    predicted = base.classes_[scores.argmax(axis=1)]
                elif name == "pair_local":
                    predicted = pair_predict(base, training, y[a], validation, scores)
                else:
                    predicted = make_model(name, y[a]).fit(training, y[a]).predict(validation)
                seconds = (
                    time.perf_counter()
                    - start
                    + (elapsed if name in ("baseline", "pair_local") else 0)
                )
                frame = pd.DataFrame(
                    {"row_id": b, "truth": y[b], "prediction": predicted, "fold": fold}
                )
                if name == "baseline":
                    frame = frame.merge(
                        pd.DataFrame(diagnostics(training, validation, scores, base.classes_)),
                        on="row_id",
                    )
                frame.to_csv(path, index=False)
                result = {
                    "experiment": name,
                    "fold": fold,
                    "seconds": seconds,
                    **metrics(y[b], predicted),
                }
                write_json(path.with_suffix(".json"), result)
                print(prefix, result, flush=True)
    if stage == "analyze":
        analyze(directory)
    else:
        summarize(directory, names, prefix)


def analyze(directory):
    frame = pd.concat(
        [pd.read_csv(directory / f"research_baseline_{fold}.csv") for fold in range(8)]
    )
    frame["wrong"] = frame.truth != frame.prediction
    frame["length_bin"] = pd.cut(frame.length, [0, 20, 100, 500, np.inf]).astype(str)
    frame["margin_bin"] = pd.cut(frame.margin, [-np.inf, 0.1, 0.3, 1, np.inf]).astype(str)
    frame.to_csv(directory / "baseline_oof.csv", index=False)
    classes = sorted(frame.truth.unique())
    pd.DataFrame(
        confusion_matrix(frame.truth, frame.prediction, labels=classes),
        index=classes,
        columns=classes,
    ).to_csv(directory / "confusion.csv")
    write_json(
        directory / "per_class.json",
        classification_report(frame.truth, frame.prediction, output_dict=True, zero_division=0),
    )
    pairs = (
        frame.loc[frame.wrong]
        .groupby(["truth", "prediction"])
        .size()
        .sort_values(ascending=False)
        .rename("errors")
    )
    pairs.to_csv(directory / "confused_pairs.csv")
    persistence = frame.groupby("row_id").wrong.agg(["sum", "count"])
    persistence.to_csv(directory / "persistence.csv")
    summary = {
        "predictions": len(frame),
        "errors": int(frame.wrong.sum()),
        "always_wrong": int((persistence["sum"] == persistence["count"]).sum()),
        "high_margin_errors": int((frame.wrong & (frame.margin >= 1)).sum()),
        "true_in_top2_errors": int((frame.wrong & (frame.truth == frame.second)).sum()),
    }
    sections = [
        "\n## Nouvelle reproduction : huit folds externes\n",
        f"Source : `{directory}`.\n",
        str(summary),
        "\n### Groupes et facteurs associés\n",
    ]
    for column in ("browser", "length_bin", "margin_bin"):
        grouped = frame.groupby(column).wrong.agg(["sum", "count", "mean"])
        grouped.to_csv(directory / f"errors_by_{column}.csv")
        sections += [f"\n{column}\n```\n{grouped.to_string()}\n```\n"]
    continuous = frame.groupby("wrong")[
        ["length", "oov", "rare", "unseen_transition", "entropy", "repetition", "margin"]
    ].mean()
    continuous.to_csv(directory / "error_factors.csv")
    sections += [
        f"\nMoyennes conditionnelles (association, pas causalité) :\n```\n{continuous.to_string()}\n```\n",
        f"\nPrincipales confusions :\n```\n{pairs.iloc[:15].to_string()}\n```\n",
        "\nTaxonomie : faible marge et vrai utilisateur en seconde position → frontière locale ; "
        "événements/transitions inédits → couverture de représentation ; erreurs répétées → "
        "ambiguïté persistante à examiner. Ces catégories ne prouvent ni une mauvaise étiquette "
        "ni un décalage de distribution. Aucun label n'est corrigé. Les marges ne sont pas calibrées.\n",
    ]
    write_json(directory / "error_summary.json", summary)
    report = Path("reports/error_analysis.md")
    marker = "\n## Nouvelle reproduction : huit folds externes\n"
    original = report.read_text()
    if marker not in original:
        report.write_text(original + "\n".join(sections))


def summarize(directory, names, prefix):
    results = []
    base = [json.loads((directory / f"{prefix}_baseline_{f}.json").read_text()) for f in range(8)]
    for name in names:
        records = [
            json.loads((directory / f"{prefix}_{name}_{f}.json").read_text()) for f in range(8)
        ]
        delta = np.array(
            [r["macro_f1"] - b["macro_f1"] for r, b in zip(records, base, strict=True)]
        )
        lower = float(delta.mean() - t.ppf(0.975, 7) * np.sqrt(1 / 8 + 1 / 3) * delta.std(ddof=1))
        corrected = introduced = 0
        for f in range(8):
            b = pd.read_csv(directory / f"{prefix}_baseline_{f}.csv")
            c = pd.read_csv(directory / f"{prefix}_{name}_{f}.csv")
            np.testing.assert_array_equal(b.row_id, c.row_id)
            corrected += int(((b.truth != b.prediction) & (c.truth == c.prediction)).sum())
            introduced += int(((b.truth == b.prediction) & (c.truth != c.prediction)).sum())
        results.append(
            {
                "experiment": name,
                **{
                    m: float(np.mean([r[m] for r in records]))
                    for m in ("macro_f1", "accuracy", "weighted_f1", "seconds")
                },
                "delta_f1": float(delta.mean()),
                "fold_wins": int((delta > 0).sum()),
                "lower95": lower,
                "repeat0": float(delta[:4].mean()),
                "repeat1": float(delta[4:].mean()),
                "corrected": corrected,
                "introduced": introduced,
                "passes": bool(lower > 0 and (delta.reshape(2, 4).mean(axis=1) > 0).all()),
            }
        )
    pd.DataFrame(results).to_csv(
        directory / ("confirmation.csv" if prefix == "confirm" else "comparison.csv"), index=False
    )
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["analyze", "research", "confirm"])
    run(parser.parse_args().stage)
