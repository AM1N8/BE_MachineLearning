"""Build a French experiment summary exclusively from recorded results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from copilote.competitive import RESULTS, signature
from copilote.competitive_diagnostics import diagnose
from copilote.io import write_json
from copilote.settings import load_config


def render(config: dict) -> dict:
    """Summarize the frozen choice, paired benchmarks, and negative experiments."""
    selection = json.loads((RESULTS / "selection.json").read_text())
    validation = json.loads((RESULTS / "validation.json").read_text())
    submission = json.loads((RESULTS / "submission.json").read_text())
    fingerprint = signature(config)
    for record in (selection, validation, submission):
        if record["signature"] != fingerprint:
            raise ValueError("Stale competitive results; rerun make competitive")
    chosen = selection["selected"]
    diagnostics = {name: diagnose(config, name) for name in dict.fromkeys(["baseline", chosen])}
    comparison = pd.read_csv(RESULTS / "comparison.csv")
    lines = [
        "# Recherche compétitive — bilan reproductible",
        "",
        f"**Soumission : `{submission['submission']}` — {submission['rows']} lignes validées.**",
        "Le score Kaggle n'est pas connu. Aucun résultat local ne garantit de battre 0,993.",
        "",
        "## Protocole",
        "",
        f"- Graine {config['seed']}. {len(comparison)} variantes évaluées sur les mêmes trois folds",
        f"  de développement ({len(selection['split']['development'])} sessions).",
        f"- Sélection par `{selection['metric']}` : métrique officielle non confirmée.",
        "- Vocabulaire, IDF, statistiques NB, normalisation, augmentation et associations",
        "  utilisateur/navigateur appris exclusivement dans la partie entraînement de chaque fold.",
        "- Les utilisateurs sont les classes à reconnaître : chaque classe reste présente",
        "  en entraînement. Un split excluant des utilisateurs ne mesure pas cette tâche.",
        "- Recherche adaptative : les scores de développement sont optimistes après sélection.",
        "- CV complète répétée et réserve historique : benchmarks descriptifs après gel du choix,",
        "  pas une validation imbriquée ni un nouveau test indépendant.",
        "- Prédiction inductive : aucun ajustement sur le test et aucune transduction.",
        "",
        "## Modèle retenu",
        "",
        f"`{chosen}` : TF-IDF d'événements (unigrammes/bigrammes) et navigateur,",
        "SVM linéaire équilibré, avec masque des classes incompatibles avec le navigateur.",
        "Le masque suppose la stabilité utilisateur/navigateur observée sur le train ;",
        "un navigateur inconnu utilise les scores sans masque. Un changement de navigateur",
        "vers un navigateur déjà connu peut pénaliser cette approche.",
        "",
        f"Meilleure moyenne brute de développement : `{selection['best_development']}`.",
        "Règle retenue : plus simple à moins d'un écart-type du meilleur ;",
        "à complexité égale, meilleure moyenne de développement.",
        "",
        "| Évaluation | Modèle | Accuracy | Macro-F1 |",
        "| --- | --- | ---: | ---: |",
    ]
    for name in dict.fromkeys(["baseline", chosen]):
        dev = json.loads((RESULTS / f"development_{name}.json").read_text())
        repeated = validation["repeated"][name]
        for label, record in (("Développement 3 folds", dev), ("CV répétée 4 × 3", repeated)):
            values = [
                f"{record['mean'][metric]:.6f} ± {record['std'][metric]:.6f}"
                for metric in ("accuracy", "macro_f1")
            ]
            lines.append(f"| {label} | {name} | {' | '.join(values)} |")
    holdout = validation["holdout"]["mean"]
    lines.extend(
        [
            f"| Réserve historique ({len(selection['split']['holdout'])} sessions) | {chosen} | "
            f"{holdout['accuracy']:.6f} | {holdout['macro_f1']:.6f} |",
            "",
            f"Gain moyen apparié d'accuracy : {validation['paired_accuracy_gain_mean']:.6f}",
            f"(écart-type entre folds : {validation['paired_accuracy_gain_std']:.6f}) ;",
            f"amélioration sur {validation['folds_improved']} des 12 folds.",
            "Les folds se recouvrent : cet écart-type n'est pas un intervalle de confiance.",
            "",
            "## Classement complet et résultats négatifs",
            "",
            "Les noyaux non linéaires, NB-SVM, augmentations, n-grammes plus longs et",
            "caractéristiques numériques n'apportent pas de gain robuste dans cette recherche.",
            "Les experts par navigateur donnent la meilleure moyenne brute de développement,",
            "mais leur faible avantage ne suffit pas selon la règle de complexité pré-déclarée.",
            "",
            "| Variante | Accuracy développement | Macro-F1 développement |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in comparison.itertuples():
        lines.append(f"| {row.name} | {row.accuracy_mean:.6f} | {row.macro_f1_mean:.6f} |")
    lines.extend(["", "## Erreurs de développement par longueur", ""])
    for name, diagnostic in diagnostics.items():
        lines.extend(
            [f"### `{name}`", "", "| Tokens | Sessions | Accuracy |", "| --- | ---: | ---: |"]
        )
        for bucket in diagnostic["accuracy_by_length"]:
            lines.append(
                f"| {bucket['length_bucket']} | {bucket['count']} | {bucket['mean']:.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "Les sessions courtes concentrent les ambiguïtés ; les marges ne sont pas des",
            "probabilités calibrées. Les extraits bruts restent dans `artifacts/`, exclu du dépôt.",
            "",
            "## Reproduction et sources",
            "",
            "```bash",
            "uv sync --python 3.13 --locked",
            "make competitive",
            "make lint",
            "make test",
            "```",
            "",
            "Sources : `reports/results/competitive/{comparison.csv,selection.json,validation.json,",
            "submission.json,diagnostics_*.json}`. Les expériences sont reprenables grâce aux",
            "empreintes du train, du code, des dépendances et du protocole.",
            "La soumission contient `RowId,prediction`, IDs de 1 à 324 contrôlés contre l'exemple.",
        ]
    )
    destination = Path("reports/competitive.md")
    destination.write_text("\n".join(lines) + "\n")
    result = {
        "signature": fingerprint,
        "report": str(destination),
        "candidates": len(comparison),
        "selected": chosen,
        "repeated_accuracy": validation["repeated"][chosen]["mean"]["accuracy"],
        "repeated_macro_f1": validation["repeated"][chosen]["mean"]["macro_f1"],
        "holdout": holdout,
        "kaggle_score": None,
    }
    write_json(RESULTS / "report.json", result)
    return result


def main() -> None:
    """Render the recorded competitive search without fitting another model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    print(json.dumps(render(load_config(args.config, args.seed)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
