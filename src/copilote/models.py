"""Small reproducible candidate catalogue; all learned transforms remain fold-local."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from copilote.features import FAMILIES, build_features


class BrowserClassifier(ClassifierMixin, BaseEstimator):
    """Estimate class frequencies conditional on a browser; unseen browsers use prior."""

    def fit(self, X: Any, y: Any) -> BrowserClassifier:
        """Learn browser groups from training sessions only."""
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.prior_ = np.bincount(encoded, minlength=len(self.classes_)).astype(float)
        self.prior_ /= self.prior_.sum()
        self.groups_ = {}
        for browser in sorted({r.browser for r in X}):
            counts = np.bincount(
                encoded[np.array([r.browser == browser for r in X])], minlength=len(self.classes_)
            ).astype(float)
            self.groups_[browser] = counts / counts.sum()
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return conditional probabilities in the fitted class order."""
        return np.array([self.groups_.get(r.browser, self.prior_) for r in X])

    def predict(self, X: Any) -> np.ndarray:
        """Predict the most frequent training class for each browser."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def catalogue() -> dict[str, dict[str, Any]]:
    """Predeclare model settings and feature ablations before holdout access."""
    raw = ["raw", "browser"]
    candidates = {
        name: {"model": name, "families": raw}
        for name in (
            "majority",
            "random",
            "browser",
            "linear",
            "logistic",
            "sgd",
            "ridge",
            "complement_nb",
            "bernoulli_nb",
            "rbf",
            "tree",
            "forest",
            "extra_trees",
            "hist_gradient",
            "mlp",
            "knn",
            "soft_vote",
            "weighted_vote",
            "hard_vote",
            "stacking",
            "sgd_l1",
            "sgd_elasticnet",
            "poly",
        )
    }
    candidates["linear"]["complexity"] = 1
    for c in (0.5, 2.0):
        candidates[f"linear_C{c:g}"] = {"model": "linear", "families": raw, "C": c, "complexity": 1}
    candidates["linear_balanced"] = {
        "model": "linear",
        "families": raw,
        "class_weight": "balanced",
        "complexity": 1,
    }
    candidates["linear_trigrams"] = {
        "model": "linear",
        "families": raw,
        "ngram_max": 3,
        "complexity": 2,
    }
    candidates["linear_unigrams"] = {
        "model": "linear",
        "families": raw,
        "ngram_max": 1,
        "complexity": 0,
    }
    candidates["linear_binary"] = {
        "model": "linear",
        "families": raw,
        "binary": True,
        "complexity": 1,
    }
    for name, families in {
        "without_browser": ["raw"],
        "action_only": ["action", "browser"],
        "contexts": ["screen", "config", "chain", "browser"],
        "numeric": ["numeric", "browser"],
        "all": list(FAMILIES),
        "no_time": [f for f in FAMILIES if f not in ("time", "numeric")],
        "no_context": ["action", "position", "time", "numeric", "browser"],
        "filled": ["raw", "filled", "browser"],
        "characters": ["char", "browser"],
    }.items():
        candidates[f"linear_{name}"] = {"model": "linear", "families": families, "complexity": 3}
    return candidates


def build_model(spec: dict[str, Any], config: dict[str, Any]) -> Any:
    """Create an unfitted end-to-end estimator from a serializable specification."""
    name = spec["model"]
    seed = config["seed"]
    if name == "browser":
        return BrowserClassifier()
    if name == "stacking":
        return StackingClassifier(
            estimators=[
                (member, build_model({**spec, "model": member}, config))
                for member in ("sgd_elasticnet", "complement_nb")
            ],
            final_estimator=LogisticRegression(C=1, max_iter=150, random_state=seed),
            cv=StratifiedKFold(2, shuffle=True, random_state=seed),
            stack_method="predict_proba",
        )
    if name in ("soft_vote", "weighted_vote", "hard_vote"):
        return VotingClassifier(
            estimators=[
                (member, build_model({**spec, "model": member}, config))
                for member in (
                    "linear" if name == "hard_vote" else "logistic",
                    "complement_nb",
                    "forest",
                )
            ],
            voting="hard" if name == "hard_vote" else "soft",
            weights=[2, 1, 1] if name == "weighted_vote" else None,
        )
    features = build_features(
        spec["families"],
        ngram_max=spec.get("ngram_max", 2),
        min_df=config["features"]["min_df"],
        max_features=config["features"]["max_features"],
        binary=spec.get("binary", False),
    )
    weight = spec.get("class_weight")
    c = spec.get("C", 1.0)
    estimators = {
        "majority": lambda: DummyClassifier(strategy="most_frequent"),
        "random": lambda: DummyClassifier(strategy="stratified", random_state=seed),
        "linear": lambda: LinearSVC(
            C=c, class_weight=weight, random_state=seed, max_iter=4000, dual="auto"
        ),
        "logistic": lambda: OneVsRestClassifier(
            LogisticRegression(C=c, solver="liblinear", max_iter=300, random_state=seed)
        ),
        "sgd": lambda: SGDClassifier(
            loss="hinge", alpha=1e-4, random_state=seed, max_iter=1500, tol=1e-3
        ),
        "sgd_l1": lambda: SGDClassifier(
            loss="log_loss", penalty="l1", alpha=1e-4, random_state=seed, max_iter=1500, tol=1e-3
        ),
        "sgd_elasticnet": lambda: SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            l1_ratio=0.15,
            alpha=1e-4,
            random_state=seed,
            max_iter=1500,
            tol=1e-3,
        ),
        "poly": lambda: SVC(C=10, kernel="poly", degree=2, gamma="scale", random_state=seed),
        "ridge": lambda: RidgeClassifier(alpha=1.0, solver="lsqr"),
        "complement_nb": lambda: ComplementNB(alpha=0.5),
        "bernoulli_nb": lambda: BernoulliNB(alpha=0.5),
        "rbf": lambda: SVC(
            C=10, kernel="rbf", gamma="scale", decision_function_shape="ovr", random_state=seed
        ),
        "tree": lambda: DecisionTreeClassifier(max_depth=30, min_samples_leaf=2, random_state=seed),
        "forest": lambda: RandomForestClassifier(
            n_estimators=config["models"]["trees"],
            max_features="sqrt",
            n_jobs=config["models"]["n_jobs"],
            random_state=seed,
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=config["models"]["trees"],
            max_features="sqrt",
            n_jobs=config["models"]["n_jobs"],
            random_state=seed,
        ),
        "hist_gradient": lambda: HistGradientBoostingClassifier(
            max_iter=20,
            max_leaf_nodes=7,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        ),
        "mlp": lambda: MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=120,
            alpha=0.01,
            early_stopping=False,
            random_state=seed,
        ),
        "knn": lambda: KNeighborsClassifier(
            n_neighbors=3, weights="distance", metric="cosine", algorithm="brute"
        ),
    }
    steps = [("features", features)]
    if name in ("hist_gradient", "mlp"):
        steps += [
            (
                "svd",
                TruncatedSVD(n_components=config["features"]["svd_components"], random_state=seed),
            ),
            ("scale", StandardScaler()),
        ]
    steps.append(("model", estimators[name]()))
    return Pipeline(steps)
