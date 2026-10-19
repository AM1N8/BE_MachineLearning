"""Fold-local, boundary-preserving multiview and conditional sequence models."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC

from copilote.competitive import BrowserRestricted, build_competitive
from copilote.features import identity
from copilote.tokens import parse_token


def events(row):
    """Remove only standalone clock markers, retaining spaces within each event."""
    return tuple(t for t in row.tokens if parse_token(t).seconds is None)


def encode(values):
    """Make event tuple boundaries unambiguous even for arbitrary input strings."""
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


class SequenceView(TransformerMixin, BaseEstimator):
    """Extract explicit n-grams, distant transitions, edges or session-local timing."""

    def __init__(self, view="raw", order=2, k=20):
        self.view = view
        self.order = order
        self.k = k

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = []
        for row in X:
            seq = events(row)
            result = []
            if self.view in {"raw", "order"}:
                orders = range(1, self.order + 1) if self.view == "raw" else (self.order,)
                result = [encode(seq[i : i + n]) for n in orders for i in range(len(seq) - n + 1)]
            elif self.view == "skip":
                result = [
                    encode((gap, seq[i], seq[i + gap]))
                    for gap in (2, 3)
                    for i in range(len(seq) - gap)
                ]
            elif self.view == "edge":
                result = [
                    encode((side, token))
                    for side, part in (("first", seq[: self.k]), ("last", seq[-self.k :]))
                    for token in part
                ]
                result += [encode(("BOS", *seq[:3])), encode(("EOS", *seq[-3:]))]
            elif self.view == "position":
                result = [
                    encode((min(3, 4 * i // max(len(seq), 1)), token))
                    for i, token in enumerate(seq)
                ]
            elif self.view == "repeat":
                runs = []
                for token in seq:
                    if runs and runs[-1][0] == token:
                        runs[-1][1] += 1
                    else:
                        runs.append([token, 1])
                result = [encode((token, int(np.log2(count)))) for token, count in runs]
            elif self.view == "hierarchy":
                for token in seq:
                    p = parse_token(token)
                    result.extend(
                        encode((p.action, "module", ".".join(screen.split(".")[:depth])))
                        for screen in p.screens
                        for depth in (2, 3, 4)
                    )
            elif self.view == "char":
                # Characters never cross event boundaries, unlike joined-session char TF-IDF.
                result = [
                    encode(("char", token[i : i + n]))
                    for token in seq
                    for n in (3, 4)
                    for i in range(len(token) - n + 1)
                ]
            elif self.view == "rhythm":
                previous = None
                gap = 0
                for token in row.tokens:
                    marker = parse_token(token).seconds
                    if marker is not None:
                        gap = 0 if previous is None else max(0, marker - previous)
                        previous = marker
                    else:
                        result.append(encode((parse_token(token).action, int(np.log2(1 + gap)))))
            elif self.view == "browser":
                result = [row.browser]
            else:
                raise ValueError(self.view)
            output.append(result or [encode(("EMPTY", self.view))])
        return output


class Fingerprints(TransformerMixin, BaseEstimator):
    """Length-normalized behavior, transition entropy and repeat characteristics."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = []
        for row in X:
            seq = events(row)
            c = Counter(seq)
            transitions = Counter(zip(seq, seq[1:]))
            n = max(1, len(seq))

            def entropy(counts):
                total = max(sum(counts.values()), 1)
                return -sum((v / total) * np.log(v / total) for v in counts.values())

            values = {
                "length": np.log1p(len(seq)),
                "unique": len(c) / n,
                "entropy": entropy(c),
                "transition_entropy": entropy(transitions),
                "transition_diversity": len(transitions) / n,
                "max_share": max(c.values(), default=0) / n,
                "repeat": sum(a == b for a, b in zip(seq, seq[1:])) / n,
            }
            for token, count in c.items():
                action = parse_token(token).action
                values[f"action:{action}"] = values.get(f"action:{action}", 0) + count / n
            result.append(values)
        return result


def block(view, spec, order=2):
    steps = [
        ("view", SequenceView(view, order, spec.get("k", 20))),
        (
            "tfidf",
            TfidfVectorizer(
                analyzer=identity,
                min_df=1 if view == "browser" else spec.get("min_df", 2),
                max_df=1.0 if view == "browser" else spec.get("max_df", 1.0),
                max_features=60000,
                binary=spec.get("binary", False),
                sublinear_tf=spec.get("sublinear", True),
                use_idf=spec.get("idf", True),
            ),
        ),
    ]
    if spec.get("percentile", 100) < 100 and view != "browser":
        steps.extend(
            [
                ("select", SelectPercentile(chi2, percentile=spec["percentile"])),
                ("norm", Normalizer()),
            ]
        )
    return Pipeline(steps)


def multiview(spec):
    """Independently normalize and weight channels before the margin classifier."""
    blocks, weights = [], {}
    for name, view, order in (
        ("raw", "raw", spec.get("order", 2)),
        ("unigram", "order", 1),
        ("bigram", "order", 2),
        ("trigram", "order", 3),
        ("skip", "skip", 2),
        ("edge", "edge", 2),
        ("position", "position", 2),
        ("repeat", "repeat", 2),
        ("hierarchy", "hierarchy", 2),
        ("rhythm", "rhythm", 2),
        ("char", "char", 2),
        ("browser", "browser", 1),
    ):
        weight = spec.get(name, 1.0 if name in {"raw", "browser"} else 0)
        if weight > 0:
            blocks.append((name, block(view, spec, order)))
            weights[name] = weight
    if spec.get("behavior", 0) > 0:
        blocks.append(
            (
                "behavior",
                Pipeline(
                    [
                        ("view", Fingerprints()),
                        ("dict", DictVectorizer()),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
            )
        )
        weights["behavior"] = spec["behavior"]
    if spec.get("svd", 0) > 0:
        blocks.append(
            (
                "svd",
                Pipeline(
                    [
                        ("raw", block("raw", spec)),
                        ("svd", TruncatedSVD(n_components=spec["svd"], random_state=42)),
                        ("norm", Normalizer()),
                    ]
                ),
            )
        )
        weights["svd"] = spec.get("svd_weight", 0.3)
    model = Pipeline(
        [
            ("features", FeatureUnion(blocks, transformer_weights=weights)),
            (
                "classifier",
                LinearSVC(
                    C=spec.get("C", 3),
                    class_weight=spec.get("class_weight", "balanced"),
                    max_iter=10000,
                    random_state=42,
                ),
            ),
        ]
    )
    return BrowserRestricted(model, restrict=True)


class MarkovIdentity(ClassifierMixin, BaseEstimator):
    """Dirichlet-smoothed per-user conditional event likelihood with global backoff."""

    def __init__(self, spec=None):
        self.spec = spec

    def fit(self, X, y):
        spec = self.spec or {}
        self.classes_, encoded = np.unique(y, return_inverse=True)
        indicator = sparse.csr_matrix(
            (np.ones(len(y)), (encoded, np.arange(len(y)))), shape=(len(self.classes_), len(y))
        )
        self.channels_ = []
        for order in range(1, spec.get("order", 2) + 1):
            vectorizer = CountVectorizer(analyzer=identity, min_df=1, max_features=60000)
            stream = SequenceView("order", order)
            matrix = vectorizer.fit_transform(stream.transform(X))
            counts = (indicator @ matrix).toarray()
            grams = [json.loads(s) for s in vectorizer.get_feature_names_out()]
            contexts = [encode(g[:-1]) for g in grams]
            _, codes = np.unique(contexts, return_inverse=True)
            mapping = sparse.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)))
            totals = np.asarray(counts @ mapping)
            global_counts = counts.sum(axis=0)
            global_totals = totals.sum(axis=0)[codes]
            base = (global_counts + 0.01) / (global_totals + 0.01 * np.bincount(codes)[codes])
            mu = spec.get("mu", 20)
            logprob = np.log(counts + mu * base) - np.log(totals[:, codes] + mu)
            self.channels_.append((stream, vectorizer, logprob))
        self.allowed_ = {
            browser: np.isin(
                self.classes_, np.asarray(y)[np.array([r.browser == browser for r in X])]
            )
            for browser in {r.browser for r in X}
        }
        return self

    def decision_function(self, X):
        scores = np.zeros((len(X), len(self.classes_)))
        for stream, vectorizer, logprob in self.channels_:
            matrix = vectorizer.transform(stream.transform(X))
            scores += (matrix @ logprob.T) / np.maximum(np.asarray(matrix.sum(axis=1)), 1)
        for i, row in enumerate(X):
            if row.browser in self.allowed_:
                scores[i, ~self.allowed_[row.browser]] = -1e6
        return scores

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]


def build(spec):
    if spec.get("method") == "champion":
        return build_competitive({"C": 3, "class_weight": "balanced", "restrict_browser": True}, 42)
    if spec.get("method") == "markov":
        return MarkovIdentity(spec)
    return multiview(spec)
