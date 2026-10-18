"""Phase 4 research candidates: tempo/suffix fusion blocks and margin fusion.

Candidates are evaluated on the frozen historical protocol (4x2 repeated
stratified folds, seed 42) with strictly fold-local vocabulary, IDF, selection
and browser masks. Confirmation protocols (seeds 43 and 44) reuse the same code
path with fresh splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from threadpoolctl import threadpool_limits

from copilote.features import identity
from copilote.improvement import SPLITS, make_model, metrics, summarize
from copilote.io import Session, file_hash, read_sessions, write_json
from copilote.phase2_models import BehavioralView, behavioral_model
from copilote.phase4_forensics import decompose
from copilote.tokens import parse_token

PARTS = ("suffix", "burst", "dwell", "scalars")
TEMPO_CANDIDATES = {
    "tempo_suffix": ("suffix",),
    "tempo_burst": ("burst",),
    "tempo_dwell": ("dwell",),
    "tempo_scalars": ("scalars",),
    "tempo_all": PARTS,
}
FUSIONS = {"fusion_raw": (0.3, False), "fusion_std": (0.5, True)}
HYBRIDS = ("sig_burst", "sig_burst_merge", "sig_burst_w05", "sig_burst_w06")
SIGNATURE_WEIGHTS = {
    "signatures_w02": 0.2,
    "signatures_w04": 0.4,
    "signatures_w05": 0.5,
    "signatures_w06": 0.6,
    "signatures_w07": 0.7,
    "signatures_w10": 1.0,
}
CANDIDATES = ("baseline", "signatures", *TEMPO_CANDIDATES, *FUSIONS, *HYBRIDS, *SIGNATURE_WEIGHTS)
BLOCK_WEIGHT = 0.3
SELECT_PERCENTILE = 20
EPS_EDGES = (0.5, 0.75, 1.0, 1.25, 1.5)
SUFFIX_EDGES = (0.05, 0.15, 0.3, 0.5)
TICK_EDGES = (1, 2, 3)
MASK_LEVEL = -1e5


def burst_bucket(count: int) -> str:
    """Cap tick sizes at five events; larger bursts stay in one bucket."""
    return f"b{min(count, 5)}"


def share_bucket(share: float) -> str:
    """Fixed a priori screen share edges; no fitting, no leakage."""
    if share > 0.75:
        return "s1"
    if share > 0.5:
        return "s2"
    if share > 0.25:
        return "s3"
    if share > 0.1:
        return "s4"
    return "s5"


def range_bucket(value: float, edges: tuple[float, ...], prefix: str) -> str:
    """Map a scalar onto fixed a priori buckets, never onto fitted quantiles."""
    for position, edge in enumerate(edges):
        if value <= edge:
            return f"{prefix}{position}"
    return f"{prefix}{len(edges)}"


class TempoView(TransformerMixin, BaseEstimator):
    """Emit session-local tempo and style tokens; no fitted state, no labels."""

    def __init__(self, parts: tuple[str, ...] = PARTS):
        self.parts = tuple(parts)

    def fit(self, X: list[Session], y: Any = None) -> TempoView:
        """Leave every statistic per session; vocabulary learning stays downstream."""
        return self

    def transform(self, X: list[Session]) -> list[list[str]]:
        """Build one token list per session from its own events and clocks."""
        documents = []
        for row in X:
            tokens = ["__present__"]
            parsed = [parse_token(raw) for raw in row.tokens]
            if "suffix" in self.parts or "burst" in self.parts:
                events = [token for token in parsed if token.seconds is None]
                tick_of = []
                index = -1
                for token in parsed:
                    if token.seconds is not None:
                        index += 1
                    else:
                        tick_of.append(index)
                sizes = Counter(tick_of)
                for token, tick in zip(events, tick_of, strict=True):
                    if "suffix" in self.parts:
                        tokens.append(f"sfx{int(token.suffix_one)}|{token.action}")
                    if "burst" in self.parts:
                        tokens.append(f"{burst_bucket(sizes[tick])}|{token.action}")
            if "dwell" in self.parts or "scalars" in self.parts:
                record = decompose(row)
                if "dwell" in self.parts:
                    total = float(sum(record["screen_seconds"].values()))
                    if total > 0:
                        for screen, seconds in record["screen_seconds"].items():
                            tokens.append(f"dwell|{screen}|{share_bucket(seconds / total)}")
                    tokens.append(f"dwelltop|{share_bucket(record['dwell_top_fraction'])}")
                if "scalars" in self.parts:
                    tokens.append(range_bucket(record["events_per_second"], EPS_EDGES, "eps"))
                    tokens.append(range_bucket(record["suffix_fraction"], SUFFIX_EDGES, "sfxfrac"))
                    tokens.append(
                        range_bucket(record["events_per_active_tick"], TICK_EDGES, "evtick")
                    )
            documents.append(tokens)
        return documents


class MergedView(TransformerMixin, BaseEstimator):
    """Emit signature tokens plus tempo tokens in one document per session."""

    def __init__(self, parts: tuple[str, ...] = ("burst",)):
        self.parts = tuple(parts)
        self.signature_view = BehavioralView("signatures")
        self.tempo_view = TempoView(parts)

    def fit(self, X: list[Session], y: Any = None) -> MergedView:
        """Stateless composition; both halves stay session-local."""
        return self

    def transform(self, X: list[Session]) -> list[list[str]]:
        """Concatenate both token streams, keeping a single present marker."""
        signature_documents = self.signature_view.transform(X)
        tempo_documents = self.tempo_view.transform(X)
        return [
            signature + tempo[1:]
            for signature, tempo in zip(signature_documents, tempo_documents, strict=True)
        ]


def _block_tempo(parts: tuple[str, ...]) -> Pipeline:
    """Binary fold-local tempo block, mirroring the signatures architecture."""
    return Pipeline(
        [
            ("view", TempoView(parts)),
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=identity,
                    token_pattern=None,
                    lowercase=False,
                    min_df=1,
                    max_features=60000,
                    sublinear_tf=True,
                    binary=True,
                ),
            ),
            ("select", SelectPercentile(chi2, percentile=SELECT_PERCENTILE)),
        ]
    )


def _block_signatures() -> Pipeline:
    """Reuse the exact unfitted signatures block from phase 2."""
    model = behavioral_model("signatures")
    return model.estimator.named_steps["features"].transformer_list[1][1]


def _augmented_model(blocks: list[tuple[str, Any]], weights: dict[str, float]) -> Any:
    """Baseline original features plus named augmentation blocks."""
    model = make_model("baseline", None)
    original = model.estimator.named_steps["features"]
    model.estimator.set_params(
        features=FeatureUnion(
            [("original", original), *blocks],
            transformer_weights={"original": 1, **weights},
        )
    )
    return model


def tempo_model(parts: tuple[str, ...], weight: float = BLOCK_WEIGHT) -> Any:
    """Baseline plus a low-weight binary chi-square-selected tempo block."""
    return _augmented_model([("behavior", _block_tempo(parts))], {"behavior": weight})


def signatures_model(weight: float = BLOCK_WEIGHT) -> Any:
    """Baseline plus the signatures block at an explicit weight."""
    return _augmented_model([("behavior", _block_signatures())], {"behavior": weight})


def hybrid_model(
    burst_weight: float = BLOCK_WEIGHT, signatures_weight: float = BLOCK_WEIGHT
) -> Any:
    """Baseline plus separate signatures and burst blocks."""
    return _augmented_model(
        [("signatures", _block_signatures()), ("tempo", _block_tempo(("burst",)))],
        {"signatures": signatures_weight, "tempo": burst_weight},
    )


def merged_model(weight: float = BLOCK_WEIGHT) -> Any:
    """Baseline plus one merged signatures+burst block behind a single selection."""
    behavior = Pipeline(
        [
            ("view", MergedView(("burst",))),
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=identity,
                    token_pattern=None,
                    lowercase=False,
                    min_df=1,
                    max_features=60000,
                    sublinear_tf=True,
                    binary=True,
                ),
            ),
            ("select", SelectPercentile(chi2, percentile=SELECT_PERCENTILE)),
        ]
    )
    return _augmented_model([("behavior", behavior)], {"behavior": weight})


def _margin_stats(margins: np.ndarray) -> tuple[float, float]:
    """Center and scale over unmasked training entries only."""
    valid = margins[margins > MASK_LEVEL]
    if valid.size:
        return float(valid.mean()), float(max(valid.std(), 1e-3))
    return 0.0, 1.0


class MarginFusion:
    """Unsupervised weighted sum of baseline and signature margins."""

    def __init__(self, alpha: float = 0.3, standardize: bool = False):
        self.alpha = alpha
        self.standardize = standardize

    def fit(self, X: list[Session], y: np.ndarray) -> MarginFusion:
        """Fit both components on this fold's rows only."""
        return self.attach(
            make_model("baseline", y).fit(X, y), behavioral_model("signatures").fit(X, y), X
        )

    def attach(self, base: Any, signatures: Any, training: list[Session]) -> MarginFusion:
        """Reuse already fitted fold-local components; stats stay training-only."""
        self.base_ = base
        self.signatures_ = signatures
        np.testing.assert_array_equal(base.classes_, signatures.classes_)
        if self.standardize:
            base_margins = base.decision_function(training)
            signature_margins = signatures.decision_function(training)
            self.base_stats_ = _margin_stats(base_margins)
            self.signature_stats_ = _margin_stats(signature_margins)
        return self

    @property
    def classes_(self) -> np.ndarray:
        """Expose the shared fold-local class order."""
        return self.base_.classes_

    def decision_function(self, X: list[Session]) -> np.ndarray:
        """Combine margins without refitting or consulting labels."""
        base_margins = self.base_.decision_function(X)
        signature_margins = self.signatures_.decision_function(X)
        if self.standardize:
            center, scale = self.base_stats_
            base_margins = (base_margins - center) / scale
            center, scale = self.signature_stats_
            signature_margins = (signature_margins - center) / scale
        return (1 - self.alpha) * base_margins + self.alpha * signature_margins

    def predict(self, X: list[Session]) -> np.ndarray:
        """Argmax over fused margins in the shared class order."""
        return self.base_.classes_[self.decision_function(X).argmax(axis=1)]


def _event_count(row: Session) -> int:
    return sum(not t.startswith("t") or not t[1:].isdigit() for t in row.tokens)


def _build(name: str, y: np.ndarray, base: Any, signatures: Any, training: list[Session]):
    if name == "baseline":
        return base
    if name == "signatures":
        return signatures
    if name in TEMPO_CANDIDATES:
        return tempo_model(TEMPO_CANDIDATES[name]).fit(training, y)
    if name == "sig_burst":
        return hybrid_model().fit(training, y)
    if name == "sig_burst_w05":
        return hybrid_model(signatures_weight=0.5).fit(training, y)
    if name == "sig_burst_w06":
        return hybrid_model(signatures_weight=0.6).fit(training, y)
    if name == "sig_burst_merge":
        return merged_model().fit(training, y)
    if name in SIGNATURE_WEIGHTS:
        return signatures_model(SIGNATURE_WEIGHTS[name]).fit(training, y)
    if name in FUSIONS:
        alpha, standardize = FUSIONS[name]
        return MarginFusion(alpha, standardize).attach(base, signatures, training)
    raise ValueError(f"Unknown candidate: {name}")


def run(stage: str, names: list[str] | None = None) -> dict[str, Any]:
    """Evaluate the requested candidates on the stage's frozen protocol."""
    manifest = {
        "sources": {
            name: file_hash(f"src/copilote/{name}.py")
            for name in (
                "phase4_candidates",
                "phase4_forensics",
                "phase2_models",
                "competitive",
                "improvement",
                "features",
                "tokens",
                "io",
            )
        },
        "train": file_hash("data/train.csv"),
        "lock": file_hash("uv.lock"),
        "historical_splits": file_hash(SPLITS),
        "submissions": {
            "best_verified": file_hash("submissions/best_verified.csv"),
            "context_signatures_experimental": file_hash(
                "submissions/context_signatures_experimental.csv"
            ),
        },
        "seed": 42,
        "confirmation_seed": 43,
        "additional_seed": 44,
        "block_weight": BLOCK_WEIGHT,
        "select_percentile": SELECT_PERCENTILE,
        "tempo_parts": {key: list(value) for key, value in TEMPO_CANDIDATES.items()},
        "fusions": {
            key: {"alpha": value[0], "standardize": value[1]} for key, value in FUSIONS.items()
        },
        "signature_weights": dict(SIGNATURE_WEIGHTS),
        "hybrids": {
            "sig_burst": {"signatures_weight": BLOCK_WEIGHT, "tempo_weight": BLOCK_WEIGHT},
            "sig_burst_w05": {"signatures_weight": 0.5, "tempo_weight": BLOCK_WEIGHT},
            "sig_burst_w06": {"signatures_weight": 0.6, "tempo_weight": BLOCK_WEIGHT},
            "sig_burst_merge": {"merged_weight": BLOCK_WEIGHT, "tempo_parts": ["burst"]},
        },
        "bucket_edges": {
            "events_per_second": list(EPS_EDGES),
            "suffix_fraction": list(SUFFIX_EDGES),
            "events_per_active_tick": list(TICK_EDGES),
        },
        "stage": stage,
    }
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    directory = Path("reports/results/phase4") / signature[:12]
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "manifest.json", manifest)

    rows = read_sessions("data/train.csv", labelled=True)
    y = np.array([r.label for r in rows])
    research_directory = directory
    if stage != "research":
        research_signature = hashlib.sha256(
            json.dumps(dict(manifest, stage="research"), sort_keys=True).encode()
        ).hexdigest()
        research_directory = Path("reports/results/phase4") / research_signature[:12]
    if stage == "research":
        splits = [
            (np.array(f["train"]), np.array(f["validation"]))
            for f in json.loads(SPLITS.read_text())["folds"]
        ]
        requested = list(names or CANDIDATES)
    else:
        seed = 43 if stage == "confirm" else 44
        splits = list(
            RepeatedStratifiedKFold(n_splits=4, n_repeats=2, random_state=seed).split(y, y)
        )
        prior = pd.read_csv(research_directory / "comparison.csv")
        positive = prior.loc[
            (prior.experiment != "baseline") & (prior.delta_f1 > 0), "experiment"
        ].tolist()
        requested = list(dict.fromkeys(["baseline", "signatures", *positive, *(names or [])]))
    unknown = [name for name in requested if name not in CANDIDATES]
    if unknown:
        raise ValueError(f"Unknown candidates: {unknown}")
    write_json(
        directory / f"{stage}_splits.json",
        [{"train": a.tolist(), "validation": b.tolist()} for a, b in splits],
    )

    for fold, (a, b) in enumerate(splits):
        training, validation = [rows[i] for i in a], [rows[i] for i in b]
        pending = [
            name
            for name in requested
            if not (
                (directory / f"{stage}_{name}_{fold}.csv").exists()
                and (directory / f"{stage}_{name}_{fold}.json").exists()
            )
        ]
        base = signatures = None
        if any(name in ("baseline", "fusion_raw", "fusion_std") for name in pending):
            base = make_model("baseline", y[a]).fit(training, y[a])
        if any(name in ("signatures", "fusion_raw", "fusion_std") for name in pending):
            signatures = behavioral_model("signatures").fit(training, y[a])
        for name in pending:
            path = directory / f"{stage}_{name}_{fold}.csv"
            start = time.perf_counter()
            with threadpool_limits(limits=2):
                model = _build(name, y[a], base, signatures, training)
                margins = model.decision_function(validation)
            classes = model.classes_
            np.testing.assert_array_equal(np.unique(y), classes)
            predicted = classes[margins.argmax(axis=1)]
            rank = np.argsort(-margins, axis=1, kind="stable")
            frame = pd.DataFrame(
                {
                    "row_id": b,
                    "truth": y[b],
                    "prediction": predicted,
                    "fold": fold,
                    "length": [_event_count(row) for row in validation],
                    "margin": [
                        float(margins[i, order[0]] - margins[i, order[1]])
                        for i, order in enumerate(rank)
                    ],
                    "second": classes[rank[:, 1]],
                }
            )
            frame.to_csv(path, index=False)
            result = {
                "experiment": name,
                "fold": fold,
                "seconds": time.perf_counter() - start,
                **metrics(y[b], predicted),
            }
            write_json(path.with_suffix(".json"), result)
            print(stage, result, flush=True)

    summarize(directory, requested, stage)
    if stage not in ("research", "confirm"):
        summary = directory / "comparison.csv"
        summary.rename(directory / f"{stage}_comparison.csv")
        if not (research_directory / "comparison.csv").exists():
            summarize(research_directory, list(CANDIDATES), "research")
    assert file_hash("submissions/best_verified.csv") == manifest["submissions"]["best_verified"]
    assert (
        file_hash("submissions/context_signatures_experimental.csv")
        == manifest["submissions"]["context_signatures_experimental"]
    )
    summary_name = {
        "research": "comparison.csv",
        "confirm": "confirmation.csv",
        "seed44": "seed44_comparison.csv",
    }[stage]
    table = pd.read_csv(directory / summary_name)
    return {"directory": str(directory), "signature": signature, "stage": stage, "table": table}


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["research", "confirm", "seed44"])
    parser.add_argument("--names", nargs="+")
    arguments = parser.parse_args()
    run(arguments.stage, arguments.names)


if __name__ == "__main__":
    main()
