"""Post-selection confidence, local decompositions, permutation and error diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from copilote.evaluate import probabilities, scores
from copilote.features import feature_names
from copilote.io import read_sessions, write_json
from copilote.models import build_model, catalogue
from copilote.plots import save_figure
from copilote.settings import provenance


def reliability(y: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> dict[str, Any]:
    """Return fixed-bin top-label calibration and coverage diagnostics."""
    confidence = proba.max(axis=1)
    correct = classes[proba.argmax(axis=1)] == y
    bins = []
    ece = 0.0
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:], strict=True):
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if mask.any():
            accuracy, mean_confidence = float(correct[mask].mean()), float(confidence[mask].mean())
            ece += float(mask.mean()) * abs(accuracy - mean_confidence)
            bins.append(
                {
                    "low": float(low),
                    "count": int(mask.sum()),
                    "confidence": mean_confidence,
                    "accuracy": accuracy,
                }
            )
    coverage = []
    for fraction in (0.25, 0.5, 0.75, 1.0):
        top = np.argsort(-confidence, kind="stable")[: max(1, int(len(y) * fraction))]
        coverage.append(
            {
                "coverage": len(top) / len(y),
                "accuracy": float(correct[top].mean()),
                "threshold": float(confidence[top].min()),
            }
        )
    return {"ece": ece, "bins": bins, "coverage": coverage}


def forest_path_contributions(forest: Any, row: Any, class_index: int) -> tuple[float, np.ndarray]:
    """Decompose a forest probability into root prior plus mean path increments.

    This is an exact path decomposition, not a Shapley or causal explanation.
    """
    contribution = np.zeros(forest.n_features_in_)
    prior = 0.0
    for estimator in forest.estimators_:
        tree = estimator.tree_
        values = tree.value[:, 0, :]
        values = values / values.sum(axis=1, keepdims=True)
        node = 0
        prior += values[0, class_index]
        while tree.children_left[node] != tree.children_right[node]:
            feature = tree.feature[node]
            child = (
                tree.children_left[node]
                if row[0, feature] <= tree.threshold[node]
                else tree.children_right[node]
            )
            contribution[feature] += values[child, class_index] - values[node, class_index]
            node = child
    return prior / len(forest.estimators_), contribution / len(forest.estimators_)


def strongest(values: np.ndarray, names: np.ndarray, count: int = 10) -> list[dict[str, Any]]:
    """Serialize strongest signed feature effects without discarding negative effects."""
    order = np.argsort(-np.abs(values), kind="stable")[:count]
    return [{"feature": str(names[i]), "value": float(values[i])} for i in order if values[i] != 0]


def run_interpretation(config: dict[str, Any]) -> dict[str, Any]:
    """Interpret the locked model and an RF on untouched holdout sessions."""
    selection = json.loads(Path("reports/results/selection.json").read_text())
    evaluation = json.loads(Path("reports/results/evaluation.json").read_text())
    bundle = joblib.load("artifacts/holdout_model.joblib")
    calibrated = joblib.load("artifacts/holdout_calibrated.joblib")
    rows = read_sessions(config["paths"]["train"], labelled=True)
    development, holdout = bundle["development"], bundle["holdout"]
    train, valid = [rows[i] for i in development], [rows[i] for i in holdout]
    y_train = np.array([str(r.label) for r in train])
    y = np.array([str(r.label) for r in valid])
    winner = bundle["model"]
    forest = build_model(catalogue()["forest"], config)
    with threadpool_limits(limits=2):
        forest.fit(train, y_train)
    joblib.dump(forest, "artifacts/holdout_forest.joblib")
    calibrated_proba = calibrated.predict_proba(valid)
    result = {
        "provenance": provenance(config),
        "selected": selection["selected"],
        "evaluation_signature": selection["signature"],
        "models": {},
        "calibrated": reliability(y, calibrated_proba, calibrated.classes_),
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), constrained_layout=True)
    plot_labels = {"winner": "SVM retenu", "forest": "Forêt aléatoire"}
    for name, model in (("winner", winner), ("forest", forest)):
        proba, probability_kind = probabilities(model, valid)
        predicted = model.predict(valid)
        confidence_matrix = calibrated_proba if name == "winner" else proba
        predicted_indices = np.searchsorted(model.classes_, predicted)
        confidence = confidence_matrix[np.arange(len(y)), predicted_indices]
        correct = predicted == y
        cases = {}
        for case, mask, direction in (
            ("confident_correct", correct, -1),
            ("uncertain_correct", correct, 1),
            ("confident_wrong", ~correct, -1),
        ):
            eligible = np.flatnonzero(mask)
            if len(eligible):
                cases[case] = int(eligible[np.argmin(direction * confidence[eligible])])
        calibration = reliability(y, proba, model.classes_)
        axes[0].plot(
            [b["confidence"] for b in calibration["bins"]],
            [b["accuracy"] for b in calibration["bins"]],
            "o-",
            label=plot_labels[name],
        )
        axes[1].plot(
            [p["coverage"] for p in calibration["coverage"]],
            [p["accuracy"] for p in calibration["coverage"]],
            "o-",
            label=plot_labels[name],
        )
        details = {
            "metrics": scores(y, predicted, proba, model.classes_),
            "probability_kind": probability_kind,
            "calibration": calibration,
            "examples": {},
            "grouped_permutation": [],
        }
        if hasattr(model, "named_steps") and "features" in model.named_steps:
            transform, classifier = model.named_steps["features"], model.named_steps["model"]
            matrix = transform.transform(valid).tocsr()
            names = feature_names(transform)
            if hasattr(classifier, "coef_"):
                details["global_coefficients"] = strongest(
                    np.abs(classifier.coef_).mean(axis=0), names, 20
                )
                details["class_coefficients"] = {
                    str(label): strongest(
                        classifier.coef_[np.searchsorted(model.classes_, label)], names
                    )
                    for label in pd.Series(y_train).value_counts().head(4).index
                }
            if hasattr(classifier, "feature_importances_"):
                details["impurity_importances_biased"] = strongest(
                    classifier.feature_importances_, names, 20
                )
            for case, index in cases.items():
                class_index = int(predicted_indices[index])
                explanation = {
                    "row_id": int(holdout[index]),
                    "truth": str(y[index]),
                    "prediction": str(predicted[index]),
                    "confidence": float(confidence[index]),
                    "tokens": len(valid[index].tokens),
                }
                if hasattr(classifier, "coef_"):
                    contributions = np.asarray(
                        matrix[index].multiply(classifier.coef_[class_index]).toarray()
                    ).ravel()
                    explanation.update(
                        method="coefficient_times_feature",
                        intercept=float(classifier.intercept_[class_index]),
                        contribution_sum=float(contributions.sum()),
                        top_contributions=strongest(contributions, names, 14),
                    )
                elif name == "forest":
                    prior, contributions = forest_path_contributions(
                        classifier, matrix[index], class_index
                    )
                    tree_probabilities = np.array(
                        [
                            tree.predict_proba(matrix[index])[0, class_index]
                            for tree in classifier.estimators_
                        ]
                    )
                    explanation.update(
                        method="exact_tree_path_probability_increments_not_SHAP",
                        root_prior=prior,
                        contribution_sum=float(contributions.sum()),
                        tree_probability_variance=float(tree_probabilities.var()),
                        top_contributions=strongest(contributions, names, 14),
                    )
                details["examples"][case] = explanation
            # Permute every column of a family together, retaining within-family dependence.
            baseline = float(f1_score(y, predicted, average="macro", zero_division=0))
            for family in dict.fromkeys(n.split("::", 1)[0] for n in names):
                columns = np.flatnonzero([n.startswith(family + "::") for n in names])
                mask = np.zeros(matrix.shape[1])
                mask[columns] = 1
                losses = []
                for repeat in range(3):
                    order = np.random.default_rng(config["seed"] + repeat).permutation(len(valid))
                    permuted = matrix.multiply(1 - mask) + matrix[order].multiply(mask)
                    permutation_prediction = classifier.predict(permuted)
                    losses.append(
                        baseline
                        - float(
                            f1_score(y, permutation_prediction, average="macro", zero_division=0)
                        )
                    )
                details["grouped_permutation"].append(
                    {
                        "family": family,
                        "columns": len(columns),
                        "macro_f1_loss_mean": float(np.mean(losses)),
                        "std": float(np.std(losses, ddof=1)),
                    }
                )
        else:
            details["local_explanation_status"] = (
                "Nonlinear/ensemble winner: forest path explanation provided separately; no false linear decomposition."
            )
        frame = pd.DataFrame(
            {
                "truth": y,
                "prediction": predicted,
                "correct": correct,
                "browser": [r.browser for r in valid],
                "length": [len(r.tokens) for r in valid],
            }
        )
        errors = (
            frame[~frame.correct]
            .groupby(["truth", "prediction"])
            .size()
            .sort_values(ascending=False)
            .head(12)
        )
        details["confusion_pairs"] = [
            {"truth": a, "prediction": b, "count": int(n)} for (a, b), n in errors.items()
        ]
        frame["length_bucket"] = pd.cut(
            frame.length, [0, 100, 500, 2000, np.inf], right=False
        ).astype(str)
        details["accuracy_by_length"] = (
            frame.groupby("length_bucket")
            .correct.agg(["mean", "count"])
            .reset_index()
            .to_dict(orient="records")
        )
        details["accuracy_by_browser"] = (
            frame.groupby("browser")
            .correct.agg(["mean", "count"])
            .reset_index()
            .to_dict(orient="records")
        )
        result["models"][name] = details
    cal = result["calibrated"]
    axes[0].plot(
        [b["confidence"] for b in cal["bins"]],
        [b["accuracy"] for b in cal["bins"]],
        "o-",
        label="SVM calibré",
    )
    axes[1].plot(
        [p["coverage"] for p in cal["coverage"]],
        [p["accuracy"] for p in cal["coverage"]],
        "o-",
        label="SVM calibré",
    )
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.7)
    axes[0].set(xlabel="Confiance moyenne", ylabel="Exactitude observée", xlim=(0, 1), ylim=(0, 1))
    coverage_values = [
        point["accuracy"]
        for model_name in ("winner", "forest")
        for point in result["models"][model_name]["calibration"]["coverage"]
    ] + [point["accuracy"] for point in result["calibrated"]["coverage"]]
    axes[1].set(
        xlabel="Couverture retenue",
        ylabel="Exactitude",
        xlim=(0.2, 1.02),
        ylim=(max(0.0, min(coverage_values) - 0.04), 1.01),
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7, loc="lower right")
    axes[1].legend(fontsize=7, loc="lower right")
    save_figure("confidence")
    # A small dense-statistics tree illustrates rules, not the selected predictor.
    from sklearn.feature_extraction import DictVectorizer

    from copilote.features import SessionNumbers

    vectorizer = DictVectorizer(sparse=False)
    dense = vectorizer.fit_transform(SessionNumbers().transform(train))
    tree = DecisionTreeClassifier(
        max_depth=2, min_samples_leaf=20, random_state=config["seed"]
    ).fit(dense, y_train)
    fig, ax = plt.subplots(figsize=(8.5, 3.2), constrained_layout=True)
    ax.set_axis_off()
    structure = tree.tree_
    leaves_below: dict[int, int] = {}

    def count_leaves(node: int) -> int:
        left, right = structure.children_left[node], structure.children_right[node]
        leaves_below[node] = 1 if left == right else count_leaves(left) + count_leaves(right)
        return leaves_below[node]

    count_leaves(0)
    positions: dict[int, tuple[float, float]] = {}
    next_leaf = 0

    def place(node: int, depth: int) -> float:
        nonlocal next_leaf
        left, right = structure.children_left[node], structure.children_right[node]
        if left == right:
            x = (next_leaf + 0.5) / leaves_below[0]
            next_leaf += 1
        else:
            x = (place(left, depth + 1) + place(right, depth + 1)) / 2
        positions[node] = (x, 0.92 - depth * 0.37)
        return x

    place(0, 0)
    feature_labels = vectorizer.get_feature_names_out()
    display_names = {
        "action_entropy": "entropie d'actions",
        "clock_gap_mean": "écart temporel moyen",
        "suffix_fraction": "fraction de suffixes",
        "token_count": "nombre de jetons",
        "unique_actions": "actions distinctes",
    }
    for node, (x, y_node) in positions.items():
        left, right = structure.children_left[node], structure.children_right[node]
        samples = structure.weighted_n_node_samples[node] / structure.weighted_n_node_samples[0]
        if left == right:
            values = structure.value[node][0]
            label = f"Classe {tree.classes_[int(np.argmax(values))]}\n{samples:.0%} des sessions"
            color = "#e7f3ec"
        else:
            feature = str(feature_labels[structure.feature[node]])
            feature = display_names.get(feature, feature.replace("_", " "))
            label = f"{feature} ≤ {structure.threshold[node]:.2f}\n{samples:.0%} des sessions"
            color = "#e8f0f8"
        ax.text(
            x,
            y_node,
            label,
            ha="center",
            va="center",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": color, "edgecolor": "#718096"},
        )
        if left != right:
            for child, branch in ((left, "oui"), (right, "non")):
                cx, cy = positions[child]
                ax.annotate(
                    "",
                    xy=(cx, cy + 0.055),
                    xytext=(x, y_node - 0.055),
                    arrowprops={"arrowstyle": "->", "color": "#718096", "lw": 0.8},
                )
                ax.text((x + cx) / 2, (y_node + cy) / 2, branch, fontsize=6, color="#52606d")
    ax.set(xlim=(0, 1), ylim=(-0.02, 1.05))
    save_figure("pruned_tree")
    result["pruned_tree"] = {
        "role": "Illustration only, depth 2 on log1p dense statistics",
        "holdout_accuracy": float(
            accuracy_score(y, tree.predict(vectorizer.transform(SessionNumbers().transform(valid))))
        ),
    }
    baseline = selection["development"]["linear"]
    redundant = []
    base_features = build_model(catalogue()["linear"], config).named_steps["features"].fit(train)
    base_count = len(feature_names(base_features))
    for name in ("linear_unigrams", "linear_without_browser"):
        current = selection["development"][name]
        reduced = build_model(catalogue()[name], config).named_steps["features"].fit(train)
        loss = baseline["mean"]["macro_f1"] - current["mean"]["macro_f1"]
        redundant.append(
            {
                "variant": name,
                "columns_removed": base_count - len(feature_names(reduced)),
                "development_macro_f1_loss": loss,
                "within_half_percentage_point": loss <= 0.005,
            }
        )
    result["redundancy"] = {
        "reference_columns": base_count,
        "comparisons": redundant,
        "caveat": "Paired development folds; this descriptive tolerance is not an equivalence confidence interval. Dense correlation alone does not establish predictive redundancy.",
    }
    result["interpretation"] = {
        "confidence": "La confiance native, le softmax des marges et la calibration sigmoïde ne sont pas interchangeables. La courbe et l'ECE sont mesurées uniquement sur le holdout. L'abstention est une analyse a posteriori, sans modifier la soumission.",
        "local": "Le SVM se décompose exactement en coefficient × TF-IDF plus intercept. Pour la forêt, les variations de probabilité le long des chemins expliquent la prédiction, sans constituer des valeurs de Shapley. Les identifiants de module suggèrent une habitude, pas une preuve métier.",
        "global": "Les coefficients et importances d'impureté dépendent du codage; ces dernières favorisent les variables riches. La permutation groupée détruit un bloc complet et quantifie sa dépendance prédictive conditionnelle au modèle ajusté.",
        "redundancy": "Corrélation, bigrammes redondants et navigateur constant ne signifient pas causalité. La suppression d'un bloc est vérifiée en CV de développement; la tolérance de 0,5 point ne remplace pas une analyse d'incertitude.",
    }
    write_json("reports/results/interpretability.json", result)
    comparison = pd.read_csv("reports/results/model_comparison.csv")
    representative = [
        "majority",
        "random",
        "browser",
        "linear_unigrams",
        "linear",
        "linear_balanced",
        "complement_nb",
        "rbf",
        "forest",
        "mlp",
        "stacking",
    ]
    comparison = comparison[comparison.name.isin(representative)].sort_values("macro_f1_mean")
    labels = {
        "majority": "Majoritaire",
        "random": "Aléatoire",
        "browser": "Navigateur",
        "linear_unigrams": "SVM, unigrammes",
        "linear": "SVM linéaire",
        "linear_balanced": "SVM pondéré",
        "complement_nb": "Naïf Bayes",
        "rbf": "SVM RBF",
        "forest": "Forêt",
        "mlp": "Perceptron",
        "stacking": "Empilement",
    }
    plt.figure(figsize=(7.5, 3.8))
    plt.barh(comparison.name.map(labels), comparison.macro_f1_mean, xerr=comparison.macro_f1_std)
    plt.xlabel("F1-macro, moyenne ± écart-type (4 plis × 3 répétitions)")
    plt.xlim(left=0)
    plt.grid(axis="x", alpha=0.2)
    save_figure("model_comparison")
    learning = json.loads(Path("reports/results/learning_curve.json").read_text())["points"]
    plt.figure(figsize=(6, 3))
    for key, label in (("train_macro_f1", "apprentissage"), ("holdout_macro_f1", "validation")):
        plt.plot(
            [v["train_sessions"] for v in learning], [v[key] for v in learning], "o-", label=label
        )
    plt.xlabel("Sessions d'apprentissage (toutes classes conservées)")
    plt.ylabel("Macro-F1")
    plt.legend()
    save_figure("learning_curve")
    # Keep the full score table linked from the interpretation artifact.
    result["repeated_selected"] = evaluation["repeated"][selection["selected"]]["mean"]
    write_json("reports/results/interpretability.json", result)
    return result
