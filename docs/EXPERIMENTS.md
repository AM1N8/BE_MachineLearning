# Expériences

Tous les chiffres proviennent des artefacts de `reports/results/` (graine 42). Aucune valeur
n'est saisie à la main; le rapport LaTeX et ce document les relisent.

## Protocole anti-fuite

1. Un holdout stratifié de 20 % est retiré du train **avant** toute observation.
2. Le choix du modèle se fait uniquement sur une CV 3 plis du développement (80 %), avec
   vocabulaire, mise à l'échelle et sélection appris **dans chaque pli**.
3. Le holdout est verrouillé une seule fois, puis la CV répétée complète (4 plis × 3 répétitions)
   sert de mesure de stabilité descriptive, explicitement non imbriquée.
4. Les familles de caractéristiques sont pré-déclarées; les ablations sont des diagnostics, jamais
   des colonnes sélectionnées.

## Sélection sur le développement (CV 3 plis, macro-F1)

| Modèle | macro-F1 (dev) | complexité |
|---|---|---|
| linear_balanced | 0.9434 ± 0.0160 | 1 |
| linear_C2 | 0.9350 ± 0.0064 | 1 |
| linear_no_time | 0.9273 ± 0.0117 | 3 |
| linear_trigrams | 0.9255 ± 0.0149 | 2 |
| linear | 0.9232 ± 0.0106 | 1 |
| linear_characters | 0.8665 ± 0.0174 | 3 |
| mlp | 0.8966 ± 0.0136 | — |
| forest | 0.8441 ± 0.0088 | — |
| linear_no_context | 0.7840 ± 0.0235 | 3 |
| linear_numeric | 0.2909 ± 0.0067 | 3 |
| tree | 0.1838 ± 0.0122 | — |
| stacking | 0.1794 ± 0.0084 | — |
| browser | 0.0053 ± 0.0000 | — |
| majority | 0.0002 ± 0.0000 | — |

Coupure « plus simple à moins d'un écart-type » : 0.9274. Modèle retenu : **linear_balanced**
(`linear`, familles `raw` + `browser`, `class_weight=balanced`). Aucun vote/empilement ne dépasse
le seuil de gain d'un écart-type, ils ne sont donc pas retenus.

## Holdout indépendant (20 %, 656 sessions)

| Métrique | lineaire_balanced | calibré (sigmoïde) |
|---|---|---|
| exactitude | 0.9512 | 0.9512 |
| F1-macro | 0.9405 | 0.9309 |
| F1 pondérée | 0.9470 | — |
| exactitude équilibrée | 0.9466 | 0.9374 |
| top-3 | 0.9893 | — |
| top-5 | 0.9909 | — |
| log-loss | 4.1123 | 0.3235 |

La log-loss native repose sur un softmax de marges non calibré; la calibration sigmoïde interne
corrige la confiance (log-loss 4.11 → 0.32) sans modifier la classe prédite.

## CV répétée (4 × 3, train complet, descriptive)

linear_balanced : macro-F1 0.9585 ± 0.0085, exactitude 0.9661. Suivants : linear 0.9481,
sgd 0.9337, hard_vote 0.9115, mlp 0.9115, poly 0.9025. Vote doux 0.8752 < linear : le vote
n'apporte pas de gain.

## Interprétabilité (holdout)

- Permutation groupée par famille (perte de macro-F1) : `raw` 0.9333 ± 0.0032, `browser` 0.1133.
  Le signal vient des séquences brutes; le navigateur est secondaire.
- ECE de confiance : natif 0.934, calibré 0.137, forêt 0.505.
- Exactitude par longueur : [0,100) 0.902 (41), [100,500) 0.923 (323), [500,2000) 0.987 (226),
  [2000,∞) 1.000 (66). Les sessions courtes sont les plus difficiles.
- Redondance : retirer les unigrammes coûte 0.049 macro-F1; retirer le navigateur 0.036.
- Les confusions sont rares (≤ 2 occurrences) entre utilisateurs aux modules proches.

## Reproductibilité

`make all` reconstruit EDA, caractéristiques, entraînement, interprétation, prédiction, rapport et
notebook. La signature de cache (`reports/results/evaluation.json`) couvre la configuration, le
hash du train, `uv.lock` et le source de `io/tokens/eda/features/models/evaluate.py`; toute
modification invalide le cache et re-déclenche l'entraînement.
