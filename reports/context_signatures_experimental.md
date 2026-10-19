# Signatures contextuelles — soumission expérimentale autorisée

## Statut

CSV : `submissions/context_signatures_experimental.csv`. **Expérimental, non promu**.
L'utilisateur a explicitement demandé cet essai limité malgré l'échec du critère
statistique. Aucun changement du modèle de référence, de son CSV ou des critères.
Aucune soumission automatique ni réglage utilisant les données test ou le leaderboard.

## Configuration exacte et absence de fuite

Appel inchangé : `behavioral_model("signatures")` dans `phase2_models.py`.
Base `browser_mask_c3` : LinearSVC C=3, classes équilibrées, graine 42,
TF-IDF original et navigateur, masque navigateur appris sur le train.
Ajout de signatures JSON (action, écran, configuration, chaîne) avec propagation
du contexte à l'intérieur de chaque session seulement et réinitialisations prévues.
TF-IDF binaire, `min_df=1`, `max_features=60000`, `sublinear_tf=True`,
unigrammes, normalisation L2 et IDF par défaut ; sélection chi² des 20 % de
caractéristiques, poids du bloc 0,3 contre 1 pour les caractéristiques originales.
Vocabulaire, IDF, chi², support navigateur et SVM sont appris uniquement sur le
train de chaque pli. Le test n'est que transformé. Aucun nouveau réglage.

Les empreintes des sources évaluées, du train et des dépendances ont été vérifiées.
Les **24 plis ont été réellement réentraînés** : chaque prédiction hors-échantillon
est identique à la campagne Phase 2, et les indices train/validation sont disjoints.
Les tests de propagation locale, vocabulaire hors pli et rechargement restent valides.

## Scores reproduits

Moyennes des huit plis de chaque protocole, sans arrondir les valeurs enregistrées :

| Protocole | Macro-F1 | Accuracy | F1 pondéré |
|---|---:|---:|---:|
| Historique | 0.962920191728837 | 0.970874839929718 | 0.9682481143309134 |
| Graine 43 | 0.9605408130663287 | 0.9704162199589028 | 0.9673879447599774 |
| Graine 44 | 0.9630297660848797 | 0.9714857127966884 | 0.9690435384367403 |

Les bornes inférieures corrigées à 95 % du gain macro-F1 restent négatives :
respectivement −0,001375, −0,002276 et −0,001729 (arrondies).
Ces trois protocoles réutilisent les mêmes sessions. Les gains moyens ne démontrent
donc pas une supériorité suffisamment robuste selon le critère de promotion figé.
La définition officielle exacte de la moyenne F1 reste non confirmée.

## Export et différence réelle

Le modèle exact a été entraîné sur les 3 279 sessions étiquetées, puis appliqué aux
324 sessions test. Le CSV contient exactement `RowId,prediction`, dans l'ordre du
sample, sans valeurs manquantes, avec des labels connus et des identifiants uniques.
Le rechargement du modèle reproduit exactement les scores et prédictions ; une
vérification supplémentaire dans un processus neuf reproduit les 324 prédictions.

**Une seule prédiction change : RowId 237, `czi` → `pgm`.**

| Indice de décision | Valeur arrondie |
|---|---:|
| Marge top-1/top-2 baseline, czi devant pgm | 0,055024 |
| Marge top-1/top-2 challenger, pgm devant czi | 0,003146 |

`pgm` était déjà le deuxième candidat de la baseline. Les 114 classes compatibles
avec le navigateur participent à la comparaison. Ces marges SVM ne sont pas des
probabilités calibrées et leur échelle diffère entre modèles. Le changement est
ambigu ; sa correction réelle ne peut pas être établie sans vérité test.
Les 323 autres prédictions restent identiques, sans modification manuelle.

SHA-256 CSV : `c2d14a684f77e3cbe03c2873e3812f203a3226da5879d31ff845e2097f55760e`.
Le CSV et les modèles de référence ont conservé leurs empreintes avant/après export.

## Reproduction et preuves

```bash
uv run python -m copilote.context_experiment reproduce
uv run python -m copilote.context_experiment export
```

Modèle : `artifacts/context_signatures_experimental.joblib`.
Résultats : `reports/results/context_signatures_experimental/` :
`cv_reproduced.csv`, `cv_summary.csv`, `cv_verification.json`, `export.json`,
`test_margins.csv`, `changed_predictions.csv`.
Les bornes de promotion proviennent des CSV de la campagne
`reports/results/phase2/a1d191a9b558/`.

Vérifications finales : 55 tests réussis, lint réussi (40 fichiers), dix
ResourceWarnings SQLite préexistants. L'export est vérifié de bout en bout,
sans être couvert directement par pytest. Baseline vérifiée toujours conservée.
