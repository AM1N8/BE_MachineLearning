"""Inductive session retrieval, user profiles, and nonlinear similarity models."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.linalg import solve
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.svm import SVC, LinearSVC

from copilote.competitive import EventStream, build_competitive
from copilote.features import identity


class SimilarityClassifier(ClassifierMixin, BaseEstimator):
    """Fit a sparse event vocabulary and compare sessions or user profiles.

    Only the small session-by-session kernel is dense; token matrices stay sparse.
    Browser support is learned exclusively from the training observations.
    """

    def __init__(self, spec: dict, seed: int = 42):
        self.spec = spec
        self.seed = seed

    def _features(self, X, fit=False):
        streams = EventStream(self.spec.get("view", "raw")).transform(X)
        matrix = (
            self.vectorizer_.fit_transform(streams) if fit else self.vectorizer_.transform(streams)
        )
        if self.spec.get("weighting") == "bm25":
            lengths = np.asarray(matrix.sum(axis=1)).ravel()
            if fit:
                self.average_length_ = max(float(lengths.mean()), 1)
            k = self.spec.get("k1", 1.2)
            b = self.spec.get("b", 0.75)
            denominator = k * (1 - b + b * lengths / self.average_length_)
            matrix.data = (
                matrix.data
                * (k + 1)
                / (matrix.data + np.repeat(denominator, np.diff(matrix.indptr)))
            )
            if fit:
                frequency = np.asarray((matrix > 0).sum(axis=0)).ravel()
                self.bm25_idf_ = np.log1p((len(X) - frequency + 0.5) / (frequency + 0.5))
            matrix = matrix.multiply(self.bm25_idf_).tocsr()
        if not fit and hasattr(self, "feature_weights_"):
            matrix = matrix.multiply(self.feature_weights_).tocsr()
        power = self.spec.get("power", 1)
        matrix.data **= power
        return normalize(matrix).tocsr()

    def fit(self, X, y):
        """Learn vocabulary, profiles, kernels and optional SVM on this fold only."""
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.vectorizer_ = TfidfVectorizer(
            tokenizer=identity,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, self.spec.get("ngram_max", 2)),
            min_df=1,
            max_features=100000,
            sublinear_tf=self.spec.get("tf", "sublinear") == "sublinear"
            and self.spec.get("weighting") != "bm25",
            binary=self.spec.get("tf") == "binary",
            use_idf=self.spec.get("weighting") != "bm25",
            norm=None,
        )
        self.training_ = self._features(X, fit=True)
        self.encoded_ = encoded
        indicator = sparse.csr_matrix(
            (np.ones(len(X)), (np.arange(len(X)), encoded)),
            shape=(len(X), len(self.classes_)),
        )
        weighting = self.spec.get("weighting")
        if weighting in ("user_idf", "entropy"):
            counts = (indicator.T @ self.training_).tocsc()
            if weighting == "user_idf":
                frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
                weights = np.log((1 + len(self.classes_)) / (1 + frequency)) + 1
            else:
                probabilities = counts.multiply(
                    1 / np.maximum(np.asarray(counts.sum(axis=0)).ravel(), 1e-12)
                ).tocsc()
                probabilities.data *= np.log(np.maximum(probabilities.data, 1e-12))
                entropy = -np.asarray(probabilities.sum(axis=0)).ravel()
                weights = np.maximum(1 - entropy / np.log(len(self.classes_)), 0.01)
            # _features applies the power after weights; training must match exactly.
            self.feature_weights_ = weights ** self.spec.get("weight_power", 1)
            self.training_ = normalize(
                self.training_.multiply(self.feature_weights_ ** self.spec.get("power", 1))
            ).tocsr()
        self.profiles_ = normalize(indicator.T @ self.training_).tocsr()
        self.support_ = {
            browser: np.unique(encoded[[row.browser == browser for row in X]])
            for browser in sorted({row.browser for row in X})
        }
        method = self.spec.get("method", "nearest")
        if method == "weighted_svm":
            self.linear_model_ = LinearSVC(
                C=self.spec.get("C", 3),
                class_weight="balanced",
                max_iter=10000,
                random_state=self.seed,
            ).fit(self.training_, y)
        if method in ("kernel_ridge", "kernel_svc"):
            gram = (self.training_ @ self.training_.T).toarray()
            gram = np.maximum(gram, 0) ** self.spec.get("degree", 2)
            if method == "kernel_ridge":
                gram.flat[:: len(X) + 1] += self.spec.get("alpha", 0.1)
                targets = indicator.toarray()
                if self.spec.get("balance", False):
                    targets /= np.sqrt(np.bincount(encoded))[None, :]
                self.dual_ = solve(gram, targets, assume_a="pos")
            else:
                self.kernel_model_ = SVC(
                    kernel="precomputed",
                    C=self.spec.get("C", 10),
                    class_weight="balanced",
                    decision_function_shape="ovr",
                    break_ties=True,
                    random_state=self.seed,
                ).fit(gram, y)
        self.svm_ = None
        if self.spec.get("blend", 0):
            self.svm_ = build_competitive(
                {"C": 3, "class_weight": "balanced", "restrict_browser": False}, self.seed
            ).fit(X, y)
        return self

    def decision_function(self, X):
        """Produce class-aligned scores without adapting to query sessions."""
        query = self._features(X)
        method = self.spec.get("method", "nearest")
        if method == "weighted_svm":
            scores = self.linear_model_.decision_function(query)
            if scores.ndim == 1:
                scores = np.column_stack((-scores, scores))
        elif method == "centroid":
            scores = (query @ self.profiles_.T).toarray()
        else:
            similarities = (query @ self.training_.T).toarray()
            if method == "kernel_ridge":
                scores = np.maximum(similarities, 0) ** self.spec.get("degree", 2) @ self.dual_
            elif method == "kernel_svc":
                scores = self.kernel_model_.decision_function(
                    np.maximum(similarities, 0) ** self.spec.get("degree", 2)
                )
                if scores.ndim == 1:
                    scores = np.column_stack((-scores, scores))
            else:
                scores = np.zeros((len(X), len(self.classes_)))
                for c in range(len(self.classes_)):
                    values = similarities[:, self.encoded_ == c]
                    k = min(self.spec.get("neighbors", 1), values.shape[1])
                    scores[:, c] = np.sort(values, axis=1)[:, -k:].mean(axis=1)
        if self.svm_ is not None:
            # Fixed, predeclared scale: centered SVM margins and cosine profile scores.
            weight = self.spec["blend"]
            scores = (1 - weight) * (self.svm_.decision_function(X) + 1) / 2 + weight * scores
        if self.spec.get("restrict_browser", True):
            for i, row in enumerate(X):
                supported = self.support_.get(row.browser)
                if supported is not None:
                    scores[i, ~np.isin(np.arange(len(self.classes_)), supported)] = -1e6
        return scores

    def predict(self, X):
        """Return the class with maximum compatible score."""
        return self.classes_[self.decision_function(X).argmax(axis=1)]
