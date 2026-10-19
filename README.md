# BE_ML — Prédiction du regroupement de sessions Copilote

Projet reproductible : à partir de sessions d'utilisation de l'IDE Copilote, prédire
la variable de regroupement par utilisateur (classification multiclasse très
déséquilibrée, 247 classes). Chaîne complète EDA → caractéristiques → validation
croisée → réserve → interprétabilité → soumission.

Reproducible pipeline predicting the per-user grouping of Copilote IDE sessions
(multiclass, 247 imbalanced classes).

## Soumission recommandée / Recommended submission

**Téléverser [`submissions/best_verified.csv`](submissions/best_verified.csv)** :
324 prédictions, colonnes `RowId,prediction`, identifiants de 1 à 324.
Le modèle retenu est `browser_mask_c3`, réentraîné sur toutes les sessions du train.
La recherche imbriquée n'a pas établi de gain robuste pour les nouveaux ensembles.
Voir [`reports/best_submission.md`](reports/best_submission.md) et la provenance
[`submissions/best_verified.json`](submissions/best_verified.json).

Upload `submissions/best_verified.csv`. Rebuild the locked selection with
`make submission`; the output must reproduce the recorded SHA-256 exactly.
The Kaggle score is unknown; selection uses training-data CV only.

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
| `make competitive` | recherche comparative, validation appariée et `submissions/competitive.csv` |
| `make advanced` | similarité, profils utilisateur, noyaux et pondérations → `submissions/advanced.csv` |
| `make submission` | réentraîne le choix verrouillé et reconstruit `submissions/best_verified.csv` |
| `make fast` | pipeline rapide (modèles de référence) |
| `make all` | chaîne complète vérifiée |
| `make archive` | `copilote-submission.zip` sans données |

Les nombres reportés proviennent uniquement de `reports/results/*.json|csv`.
Graine `42`, dépendances épinglées ; aucune étiquette de test n'est consultée.

## Recherche compétitive / Competitive search

`make competitive` évalue les variantes de `configs/competitive.yaml`, verrouille
le choix sur la CV de développement, mesure sa stabilité et produit une soumission
inductive validée. Les expériences terminées sont mises en cache avec leurs empreintes.
Résultats et limites : [`reports/competitive.md`](reports/competitive.md).
Le score Kaggle reste inconnu ; les scores locaux ne garantissent pas 0,993.

Run `make competitive` for resumable training-only model search and a validated
`submissions/competitive.csv`. The supplied evaluation states mean F1;
selection interprets this as macro-F1 and reports accuracy alongside it.
The exact Kaggle averaging implementation remains unconfirmed.

Deuxième recherche : [`reports/advanced.md`](reports/advanced.md), 33 candidats
dont deux références existantes. Le gagnant de développement ne confirme pas son
avantage en CV répétée : ne pas présenter cette nouvelle soumission comme un gain établi.
Second-round search compares retrieval, user profiles, kernel models and supervised
event weighting; results include unsuccessful approaches and paired validation.
