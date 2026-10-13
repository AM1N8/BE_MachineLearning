# BE_ML — Prédiction du regroupement de sessions Copilote

Projet reproductible : à partir de sessions d'utilisation de l'IDE Copilote, prédire
la variable de regroupement par utilisateur (classification multiclasse très
déséquilibrée, 247 classes). Chaîne complète EDA → caractéristiques → validation
croisée → réserve → interprétabilité → soumission.

Reproducible pipeline predicting the per-user grouping of Copilote IDE sessions
(multiclass, 247 imbalanced classes).

## Installation / Setup

```bash
uv sync --python 3.13      # dépendances épinglées (requirements.txt fourni)
make all                   # lint + tests + pipeline complet + notebook
```

## Données / Data

Placez `train.csv`, `test.csv` et `sample_submission.csv` dans `data/` (lecture
seule, exclus du dépôt et de l'archive). Voir `data/README.md` pour le format.
Raw data belong in `data/`, are read-only and excluded from git/archives.

## Cibles `make` / Targets

| Cible | Effet |
| --- | --- |
| `make setup` | installe l'environnement (Python 3.13) |
| `make lint` | ruff (check + format) |
| `make test` | tests pytest avec couverture |
| `make eda` | analyse exploratoire → `reports/results/eda.json` |
| `make features` | diagnostics de caractéristiques |
| `make train` | sélection, réserve, VC répétée |
| `make interpret` | interprétabilité et figures |
| `make predict` | soumission validée dans `submissions/` |
| `make report` | PDF français (`reports/latex/main.pdf`, 5–8 pages) |
| `make notebook` | exécute `notebooks/copilote.ipynb` |
| `make fast` | pipeline rapide (modèles de référence) |
| `make all` | chaîne complète vérifiée |
| `make archive` | `copilote-submission.zip` sans données |

Les nombres reportés proviennent uniquement de `reports/results/*.json|csv`.
Graine `42`, dépendances épinglées ; aucune étiquette de test n'est consultée.
