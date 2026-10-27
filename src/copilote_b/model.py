"""Crop-weighted sparse SVM and training-only browser compatibility."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from .features import Trace, document


def crop_plan(size: int, length: int) -> list[tuple[int, int]]:
    plan = [(0, size)]
    if size > length * 1.5:
        starts = np.linspace(0, size - length, 3).astype(int)
        plan.extend((int(start), int(start + length)) for start in starts)
    return list(dict.fromkeys(plan))


def check_probabilities(
    probabilities: Any, shape: tuple[int, int] | None = None
) -> None:
    if shape is not None and probabilities.shape != shape:
        raise ValueError(f"Unexpected probability shape: {probabilities.shape}")
    if (
        probabilities.ndim != 2
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
    ):
        raise ValueError(
            "Probabilities must be finite, nonnegative and two-dimensional"
        )
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-5)


class CropSVM:
    def __init__(self, config: dict[str, Any], seed: int = 137) -> None:
        self.config = dict(config)
        self.seed = seed
        self.vectorizers: dict[str, Any] = {}
        self.classes_: Any = None
        self.classifier: Any = None
        self.browser_support: dict[str, set[int]] = {}

    def matrix(
        self,
        traces: list[Trace],
        examples: list[tuple[int, tuple[int, int]]],
        fit: bool = False,
    ) -> Any:
        blocks = []
        for view in self.config.get("view", "full").split("+"):
            if fit:
                vectorizer = TfidfVectorizer(
                    lowercase=False,
                    dtype=np.float32,
                    min_df=1,
                    max_features=self.config.get("max_features", 200000),
                    tokenizer=str.split,
                    token_pattern=None,
                    binary=True,
                    sublinear_tf=False,
                    use_idf=self.config.get("use_idf", True),
                    ngram_range=(1, 3),
                )
                vectorizer.fit(document(t, view, (0, len(t.full))) for t in traces)
                self.vectorizers[view] = vectorizer
            block = self.vectorizers[view].transform(
                document(traces[i], view, bounds) for i, bounds in examples
            )
            blocks.append(block * (1.0 if view == "full" else 0.7))
        return (
            normalize(sparse.hstack(blocks, format="csr"))
            if len(blocks) > 1
            else blocks[0]
        )

    def fit(self, traces: list[Trace], y: Any) -> CropSVM:
        if len(traces) != len(y) or not traces:
            raise ValueError(
                "Training traces and labels must have equal, nonzero length"
            )
        self.classes_, encoded = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("At least two training classes are required")
        supports = np.bincount(encoded)
        balancing = len(y) / (len(supports) * supports)
        balancing /= np.average(balancing, weights=supports)
        examples: list[tuple[int, tuple[int, int]]] = []
        labels = []
        weights = []
        self.browser_support = {}
        for i, (trace, label) in enumerate(zip(traces, encoded)):
            self.browser_support.setdefault(trace.browser, set()).add(int(label))
            plan = crop_plan(len(trace.full), self.config.get("length", 128))
            mass = np.array([1.0] + [0.3] * (len(plan) - 1))
            mass /= mass.sum()
            mass *= balancing[label]
            examples.extend((i, bounds) for bounds in plan)
            labels.extend([int(label)] * len(plan))
            weights.extend(mass.tolist())
        matrix = self.matrix(traces, examples, fit=True)
        self.classifier = LinearSVC(
            C=1.0,
            dual=True,
            max_iter=5000,
            random_state=self.seed,
            loss="squared_hinge",
        )
        self.classifier.fit(matrix, labels, sample_weight=weights)
        return self

    def predict_proba(self, traces: list[Trace]) -> Any:
        if self.classifier is None:
            raise ValueError("Fit the model before prediction")
        examples = [(i, (0, len(t.full))) for i, t in enumerate(traces)]
        scores = self.classifier.decision_function(self.matrix(traces, examples))
        if scores.ndim == 1:
            scores = np.column_stack((-scores, scores))
        probabilities = softmax(scores * 5, axis=1)
        result = probabilities.copy()
        for i, trace in enumerate(traces):
            allowed = self.browser_support.get(trace.browser)
            if allowed:
                mask = np.ones(result.shape[1], dtype=bool)
                mask[list(allowed)] = False
                result[i, mask] = 0
                if result[i].sum() == 0:
                    result[i] = probabilities[i]
        result /= result.sum(axis=1, keepdims=True)
        check_probabilities(result, (len(traces), len(self.classes_)))
        return result
