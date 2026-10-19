# Soumission recommandée — recherche terminée

## Fichier à téléverser

`submissions/best_verified.csv` : 324 lignes, colonnes `RowId,prediction`,
identifiants 1–324 dans l'ordre de `data/sample_submission.csv`.
Le fichier conserve les prédictions de `submissions/competitive.csv`.
Le score Kaggle est inconnu ; aucun réglage n'utilise le classement.

## Modèle retenu

`browser_mask_c3` : TF-IDF d'événements unigrammes/bigrammes et navigateur,
LinearSVC équilibré, C = 3, restriction classe/navigateur apprise sur le train.
Les marqueurs temporels sont retirés du flux d'événements. Le vocabulaire,
l'IDF et le support navigateur sont ajustés dans chaque fold d'entraînement.
L'ajustement final utilise les 3 279 sessions étiquetées, pour 247 classes ;
le test est uniquement transformé, sans transduction.

## Comparaison imbriquée

Source : `reports/results/extreme/1ba1ee389f73/nested_validation.json`.
Protocole : 4 folds externes × 2 répétitions ; dans chaque train externe,
3 folds internes pour comparer sept candidats fixés et huit essais Optuna,
puis sélectionner le poids d'ensemble sur les prédictions internes hors-fold.

| Méthode | Macro-F1 moyen ± écart-type | Exactitude moyenne |
| --- | ---: | ---: |
| Référence retenue | 0,961251 ± 0,010938 | 0,969350 |
| Candidat sélectionné en interne | 0,960826 ± 0,010614 | 0,968588 |
| Ensemble sélectionné en interne | 0,961320 ± 0,010335 | 0,969350 |

Gain apparié de l'ensemble : +0,000069 macro-F1, seulement 3/8 folds améliorés.
Gains moyens par répétition : −0,000337 et +0,000475. Borne inférieure corrigée
à 95 % : −0,002336 (correction approximative de type Nadeau–Bengio).
La règle préexistante exige une borne positive et un gain dans les deux
répétitions : **pas de promotion**. Le très léger gain moyen ne suffit pas.

Les anciens benchmarks et la réserve historique ont déjà été consultés ;
ces résultats ne constituent pas un test indépendant sur de nouvelles données.
La description fournie indique « F1 score moyen » : macro-F1 est l'interprétation
de sélection, sans confirmation du détail d'implémentation Kaggle.

## Reproduction et contrôles

```bash
uv sync --python 3.13 --locked
make submission
```

Cette commande reconstruit le choix verrouillé depuis les résultats enregistrés,
réentraîne sur tout le train, contrôle les empreintes des données et de la campagne,
le format, les classes autorisées, l'ordre du sample et l'identité exacte des
prédictions avec le CSV gelé. Le modèle est sauvegardé dans
`artifacts/best_verified.joblib`, puis rechargé pour vérifier ses prédictions.
La provenance est publiée dans `reports/results/best_submission.json` et
`submissions/best_verified.json`.

Rejouer toute la campagne (beaucoup plus coûteux) :
`uv run python -m copilote.extreme all --config configs/extreme.yaml`.
Les caches sont liés aux données, aux sources prédictives, à la configuration et
au verrou des dépendances. Les instantanés de référence sont dans `artifacts/` ;
la campagne refuse leur modification. Une interruption d'essai Optuna se reprend
avec `uv run python -m copilote.extreme_resume` avant de relancer la campagne.
