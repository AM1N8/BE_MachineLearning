"""Session-local behavioral views and strictly out-of-fold global-aware correction."""

import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from copilote.competitive import EventStream
from copilote.features import identity
from copilote.improvement import make_model, metrics
from copilote.tokens import filled_context

FAMILIES = ("context", "transitions", "position", "edges", "signatures")


class BehavioralView(TransformerMixin, BaseEstimator):
    """Emit independent hypotheses; context resets for every session and logout."""

    def __init__(self, family="context"):
        self.family = family

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        documents = []
        for row in X:
            parsed = filled_context(row.tokens)
            events = [(t.action, s, c, h) for t, s, c, h in parsed]
            tokens = ["__present__"]
            for i, event in enumerate(events):
                if self.family == "context":
                    token = parsed[i][0]
                    if not token.screens or not token.configs:
                        tokens.append(json.dumps(event, ensure_ascii=False))
                elif self.family == "transitions" and i:
                    # Context-aware transitions between distinct states, not raw n-grams.
                    if event != events[i - 1]:
                        tokens.append(json.dumps((events[i - 1], event), ensure_ascii=False))
                elif self.family == "position":
                    tokens.append(json.dumps((min(3, 4 * i // len(events)), event)))
                elif self.family == "edges":
                    if i < 20:
                        tokens.extend([json.dumps(("early", event))] * 2)
                    if i >= len(events) - 20:
                        tokens.append(json.dumps(("late", event)))
                elif self.family == "signatures":
                    tokens.append(json.dumps(event))
            documents.append(tokens)
        return documents


def behavioral_model(family):
    """Fixed low-weight augmentation; selection, vocabulary and IDF are fold-local."""
    model = make_model("baseline", None)
    steps = [
        ("view", BehavioralView(family)),
        (
            "tfidf",
            TfidfVectorizer(
                tokenizer=identity,
                token_pattern=None,
                lowercase=False,
                min_df=1 if family == "signatures" else 2,
                max_features=60000,
                sublinear_tf=True,
                binary=family == "signatures",
            ),
        ),
    ]
    if family == "signatures":
        steps.append(("select", SelectPercentile(chi2, percentile=20)))
    model.estimator.set_params(
        features=FeatureUnion(
            [("original", model.estimator.named_steps["features"]), ("behavior", Pipeline(steps))],
            transformer_weights={"original": 1, "behavior": 0.3},
        )
    )
    return model


class Profiles:
    """Training-only smoothed event and conditional transition evidence."""

    def fit(self, rows, y):
        self.events = defaultdict(Counter)
        self.pairs = defaultdict(Counter)
        self.previous = defaultdict(Counter)
        self.df = Counter()
        self.class_df = defaultdict(Counter)
        self.counts = Counter(y)
        self.browser = defaultdict(set)
        for row, label, events in zip(rows, y, EventStream().transform(rows), strict=True):
            self.events[label].update(events)
            self.df.update(set(events))
            self.class_df[label].update(set(events))
            pairs = list(zip(events, events[1:]))
            self.pairs[label].update(pairs)
            self.previous[label].update(a for a, _ in pairs)
            self.browser[label].add(row.browser)
        self.global_events = sum(self.events.values(), Counter())
        self.total = sum(self.global_events.values())
        self.totals = {c: sum(v.values()) for c, v in self.events.items()}
        return self

    def transform(self, rows, scores, classes):
        ranks = np.argsort(scores, axis=1)[:, ::-1]
        pairs = classes[ranks[:, :2]]
        records = []
        for i, (row, events) in enumerate(zip(rows, EventStream().transform(rows), strict=True)):
            n = max(1, len(events))
            top = scores[i, ranks[i, :3]]
            valid = scores[i][scores[i] > -1e5]
            gap = top[0] - top[1]
            features = [np.clip(v, -10, 10) for v in top]
            features += [
                min(gap, 10),
                min(top[1] - top[2], 10),
                (top[0] - valid.mean()) / max(valid.std(), 0.01),
                np.log1p(n),
                len(set(events)) / n,
                sum(e not in self.df for e in events) / n,
                sum(self.df[e] <= 2 for e in events) / n,
            ]
            evidence = []
            for c in pairs[i]:
                unigram = transition = coverage = 0.0
                for j, event in enumerate(events):
                    backoff = (self.global_events[event] + 1) / (self.total + len(self.df) + 1)
                    probability = (self.events[c][event] + 20 * backoff) / (self.totals[c] + 20)
                    unigram += np.log(probability)
                    coverage += self.class_df[c][event] / self.counts[c]
                    if j:
                        conditional = (self.pairs[c][(events[j - 1], event)] + 20 * probability) / (
                            self.previous[c][events[j - 1]] + 20
                        )
                        transition += np.log(conditional)
                evidence.append(
                    [
                        unigram / n,
                        transition / max(1, n - 1),
                        coverage / n,
                        float(row.browser in self.browser[c]),
                    ]
                )
            features.extend(evidence[0])
            features.extend(evidence[1])
            features.extend(np.subtract(evidence[1], evidence[0]))
            records.append(features)
        return np.asarray(records), pairs


def meta_model():
    """Unbalanced logistic loss preserves the observed cost of damaging correct rows."""
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, random_state=42)
    )


class TopTwo:
    """Fit on OOF base scores only; choose a conservative gate inside outer training."""

    def fit(self, rows, y):
        y = np.asarray(y)
        features, pairs, labels, audit = [], [], [], []
        for a, b in StratifiedKFold(3, shuffle=True, random_state=42).split(y, y):
            training, heldout = [rows[i] for i in a], [rows[i] for i in b]
            base = make_model("baseline", y[a]).fit(training, y[a])
            profile = Profiles().fit(training, y[a])
            x, candidates = profile.transform(
                heldout, base.decision_function(heldout), base.classes_
            )
            features.extend(x)
            pairs.extend(candidates)
            labels.extend(y[b])
            audit.append(
                {"train": [rows[i].row_id for i in a], "heldout": [rows[i].row_id for i in b]}
            )
        x, pairs, labels = np.array(features), np.array(pairs), np.array(labels)
        target = (pairs[:, 1] == labels).astype(int)
        probabilities = np.zeros(len(x))
        # Cross-fit the meta learner before choosing its threshold. All selection is inner.
        for a, b in StratifiedKFold(3, shuffle=True, random_state=42).split(x, target):
            probabilities[b] = meta_model().fit(x[a], target[a]).predict_proba(x[b])[:, 1]
        baseline = metrics(labels, pairs[:, 0])
        best = baseline["macro_f1"]
        self.threshold_, self.gap_ = 1.1, 0.0
        trials = []
        for gap in (0.1, 0.3):
            for threshold in (0.5, 0.65, 0.8, 0.9):
                switch = (x[:, 3] < gap) & (probabilities >= threshold)
                predicted = np.where(switch, pairs[:, 1], pairs[:, 0])
                result = metrics(labels, predicted)
                trials.append(
                    {"gap": gap, "threshold": threshold, **result, "switches": int(switch.sum())}
                )
                if result["macro_f1"] > best and result["accuracy"] >= baseline["accuracy"]:
                    best = result["macro_f1"]
                    self.threshold_, self.gap_ = threshold, gap
        self.meta_ = meta_model().fit(x, target)
        self.base_ = make_model("baseline", y).fit(rows, y)
        self.profile_ = Profiles().fit(rows, y)
        self.audit_ = {
            "inner_splits": audit,
            "threshold": self.threshold_,
            "gap": self.gap_,
            "trials": trials,
            "baseline": baseline,
            "positive_meta_examples": int(target.sum()),
        }
        return self

    def predict(self, rows):
        x, pairs = self.profile_.transform(
            rows, self.base_.decision_function(rows), self.base_.classes_
        )
        probability = self.meta_.predict_proba(x)[:, 1]
        switch = (x[:, 3] < self.gap_) & (probability >= self.threshold_)
        self.last_ = {"probability": probability, "switch": switch, "gap": x[:, 3]}
        return np.where(switch, pairs[:, 1], pairs[:, 0])
