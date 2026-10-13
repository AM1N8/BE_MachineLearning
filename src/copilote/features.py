"""Fold-local sparse session representations and reproducible feature diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import f_oneway, kruskal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from copilote.eda import save_figure, session_statistics
from copilote.io import Session, read_sessions, write_json
from copilote.settings import provenance
from copilote.tokens import filled_context, parse_token


@lru_cache(maxsize=10_000)
def views(row: Session) -> dict[str, tuple[str, ...]]:
    """Extract deterministic per-row streams; cache never learns across sessions."""
    events = [parse_token(t) for t in row.tokens if parse_token(t).seconds is None]
    result = {
        "raw": tuple(t for t in row.tokens if parse_token(t).seconds is None),
        "action": tuple(t.action for t in events),
        "screen": tuple(x for t in events for x in t.screens),
        "config": tuple(x for t in events for x in t.configs),
        "chain": tuple(x for t in events for x in t.chains),
        "pairs": tuple(
            f"{t.action}|{kind}={x}"
            for t in events
            for kind, group in (("s", t.screens), ("c", t.configs), ("h", t.chains))
            for x in group
        ),
        "filled": tuple(
            f"{t.action}|s={screens}|c={configs}|h={chains}"
            for t, screens, configs, chains in filled_context(row.tokens)
        ),
        "browser": (row.browser,),
        "time": tuple(
            f"log2seconds={int(np.log2(1 + parse_token(t).seconds))}"
            for t in row.tokens
            if parse_token(t).seconds is not None
        ),
        "position": tuple(
            f"{position}={token.action}"
            for position, sequence in (("first", events[:3]), ("last", events[-3:]))
            for token in sequence
        ),
    }
    return result


def identity(value: Any) -> Any:
    """Keep event boundaries intact, including whitespace within action names."""
    return value


class SessionText(BaseEstimator, TransformerMixin):
    """Select one session view without fitting global data."""

    def __init__(self, view: str = "raw", characters: bool = False):
        self.view = view
        self.characters = characters

    def fit(self, X: Sequence[Session], y: Any = None) -> SessionText:
        """Stateless fit for sklearn pipeline compatibility."""
        return self

    def transform(self, X: Sequence[Session]) -> list[Any]:
        """Emit sequences or delimiter-separated strings for character n-grams."""
        values = [views(row)[self.view] for row in X]
        return (
            [" ␞ ".join(v) for v in values]
            if self.characters
            else [v or ("__EMPTY__",) for v in values]
        )


class SessionNumbers(BaseEstimator, TransformerMixin):
    """Return log-compressed nonnegative numeric summaries."""

    def fit(self, X: Sequence[Session], y: Any = None) -> SessionNumbers:
        """No population-level statistics are learned here."""
        return self

    def transform(self, X: Sequence[Session]) -> list[dict[str, float]]:
        """Apply fixed log1p; scaling is fitted later inside the pipeline."""
        return [
            {key: float(np.log1p(value)) for key, value in session_statistics(row).items()}
            for row in X
        ]


FAMILIES = (
    "raw",
    "action",
    "screen",
    "config",
    "chain",
    "pairs",
    "position",
    "time",
    "browser",
    "numeric",
)


def build_features(
    families: Sequence[str] = FAMILIES,
    *,
    ngram_max: int = 2,
    min_df: int = 2,
    max_features: int = 60_000,
    binary: bool = False,
) -> FeatureUnion:
    """Build fresh vectorizers, IDF and scaling, all fitted only on training folds."""
    blocks = []
    for family in families:
        if family == "numeric":
            block = Pipeline(
                [
                    ("view", SessionNumbers()),
                    ("dict", DictVectorizer()),
                    ("scale", StandardScaler(with_mean=False)),
                ]
            )
        else:
            chars = family == "char"
            vectorizer = TfidfVectorizer(
                analyzer="char" if chars else "word",
                tokenizer=None if chars else identity,
                preprocessor=None,
                token_pattern=None,
                lowercase=False,
                ngram_range=(3, 5)
                if chars
                else (1, 1 if family in ("browser", "position", "time") else ngram_max),
                min_df=1 if family == "browser" else min_df,
                max_features=max_features,
                sublinear_tf=not binary,
                binary=binary,
                use_idf=not binary,
                norm="l2",
            )
            block = Pipeline(
                [("view", SessionText("raw" if chars else family, chars)), ("tfidf", vectorizer)]
            )
        blocks.append((family, block))
    return FeatureUnion(blocks)


def feature_names(features: FeatureUnion) -> np.ndarray:
    """Recover stable block-qualified names for fitted sparse feature columns."""
    names = []
    for name, pipeline in features.transformer_list:
        fitted = pipeline.named_steps["dict" if name == "numeric" else "tfidf"]
        names.extend(f"{name}::{value}" for value in fitted.get_feature_names_out())
    return np.asarray(names)


def run_feature_analysis(config: dict[str, Any]) -> dict[str, Any]:
    """Describe training features; all supervised tests are exploratory, not selection."""
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list, linkage

    train = read_sessions(config["paths"]["train"], labelled=True)
    y = np.array([str(r.label) for r in train])
    labels = sorted(set(y))
    codes = {label: index for index, label in enumerate(labels)}
    numbers = pd.DataFrame([session_statistics(r) for r in train])
    correlations = numbers.corr(method="spearman").fillna(0)
    order = leaves_list(linkage(correlations.to_numpy(), method="average"))
    plt.figure(figsize=(7, 6))
    plt.imshow(correlations.to_numpy()[order][:, order], vmin=-1, vmax=1, cmap="coolwarm")
    plt.xticks(range(len(order)), numbers.columns[order], rotation=90, fontsize=6)
    plt.yticks(range(len(order)), numbers.columns[order], fontsize=6)
    plt.colorbar(label="Corrélation de Spearman")
    save_figure("feature_correlations")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    selected = [0, 15, 152, 199]
    for ax, column in zip(axes, ["actions", "clock_max_seconds", "unique_screens"]):
        ax.boxplot(
            [numbers.loc[y == labels[code], column] for code in selected],
            tick_labels=selected,
            showfliers=False,
        )
        ax.set(xlabel="Code utilisateur (catégoriel)", ylabel=column)
    save_figure("feature_boxplots")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, column in zip(axes, ["action_entropy", "clock_resets", "repetition_fraction"]):
        ax.hist(numbers[column], bins=30)
        ax.set(xlabel=column, ylabel="Sessions")
    save_figure("feature_distributions")
    features = build_features(("raw", "browser"))
    matrix = features.fit_transform(train)
    svd = TruncatedSVD(n_components=2, random_state=config["seed"])
    projection = svd.fit_transform(matrix)
    plt.figure(figsize=(6, 4))
    plt.scatter(*projection.T, c=[codes[v] for v in y], cmap="turbo", s=5, alpha=0.5)
    plt.xlabel("SVD 1")
    plt.ylabel("SVD 2")
    plt.title("Projection descriptive du train (couleurs catégorielles)")
    save_figure("feature_projection")
    tests = []
    for column in numbers:
        groups = [numbers.loc[y == label, column].to_numpy() for label in labels]
        anova = f_oneway(*groups)
        kw = kruskal(*groups) if numbers[column].nunique() > 1 else None
        grand = numbers[column].mean()
        total = float(((numbers[column] - grand) ** 2).sum())
        between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        tests.append(
            {
                "feature": column,
                "anova_f": float(anova.statistic) if np.isfinite(anova.statistic) else None,
                "anova_p": float(anova.pvalue) if np.isfinite(anova.pvalue) else None,
                "kruskal_h": float(kw.statistic) if kw else None,
                "kruskal_p": float(kw.pvalue) if kw else None,
                "eta_squared": float(between / total) if total else 0.0,
            }
        )
    table = pd.DataFrame(tests)
    table["kruskal_p_bonferroni"] = (table.kruskal_p * len(table)).clip(upper=1)
    table.to_csv("reports/results/feature_tests.csv", index=False)
    correlations.to_csv("reports/results/feature_correlations.csv")
    result = {
        "provenance": provenance(config),
        "descriptive_only": True,
        "categorical_code_mapping": dict(enumerate(labels)),
        "matrix_shape": list(matrix.shape),
        "sparse": sparse.issparse(matrix),
        "density": matrix.nnz / np.prod(matrix.shape).item(),
        "svd_explained_variance": svd.explained_variance_ratio_.tolist(),
        "tests": table.replace({np.nan: None}).to_dict(orient="records"),
        "interpretation": "Codes de classes nominaux, pas de corrélation numérique avec la cible. ANOVA fragile sous asymétrie/hétéroscédasticité; Kruskal et correction de Bonferroni complètent une analyse exploratoire. Les dépendances intra-utilisateur limitent les p-valeurs. La sélection prédictive reste dans les plis.",
    }
    write_json("reports/results/features.json", result)
    return result
