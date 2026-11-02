# Expérience 006 — Résultats du 18 octobre 2026

## Résultat

Le choix par validation croisée est **Ensemble poids SVM=0.75** : 75 % du score SVM séquentiel (`C=20`) et 25 % des probabilités des spécialistes Extra Trees de référence.

| Modèle | F1 macro (%) | F1 pondéré (%) | Accuracy (%) |
| --- | --- | --- | --- |
| Extra Trees reference | 89.05 | 90.42 | 91.46 |
| Ensemble poids SVM=0.75 | 91.87 | 93.41 | 94.21 |

Gain de F1 macro sur les 20 % réservés : **2.82 points**. La référence fait 56 erreurs, l’ensemble 38 sur 656 sessions : **18 erreurs corrigées et aucune régression** sur ce découpage. L’objectif de 97 % de F1 macro **n’est pas atteint** (écart : 5.13 points).

## Sélection sur l’apprentissage

| Modèle | F1 macro CV (%) | Écart-type des plis (points) |
| --- | --- | --- |
| Ensemble poids SVM=0.75 | 92.41 | 0.88 |
| SVM seq C=20 | 91.16 | 1.50 |
| SVM seq C=5 | 90.94 | 1.47 |
| Ensemble poids SVM=0.5 | 89.97 | 1.19 |
| SVM seq C=1 | 89.60 | 1.52 |
| Ensemble poids SVM=0.25 | 89.15 | 1.26 |
| Extra Trees reference | 88.04 | 1.01 |
| SVM full C=5 | 87.24 | 1.32 |
| SVM full C=20 | 87.12 | 1.26 |
| SVM full C=1 | 85.08 | 1.06 |
| Extra Trees enrichi | 82.03 | 1.23 |

Le score de sélection est la moyenne des trois F1 macro des plis. Le CSV conserve aussi le F1 sur les prédictions hors pli regroupées ; ces deux quantités sont différentes. Le poids de l’ensemble est choisi parmi 0.25, 0.50 et 0.75, après choix du meilleur SVM et du meilleur arbre sur ces mêmes plis.

## Contrôle de sensibilité

Sur trois plis avec la graine 137, sans changer la configuration choisie : référence **86.75 %**, ensemble **91.65 %**. Le gain persiste sur ce second découpage de la partie apprentissage.

## Analyse des erreurs restantes

- 38 erreurs concernent 32 utilisateurs.
- De 21 à 100 actions : 12 erreurs sur 89 sessions.
- De 101 à 500 actions : 24 erreurs sur 316 sessions.
- Au-delà de 500 actions : 2 erreurs sur 251 sessions.
- Aucune session de 20 actions ou moins dans cette validation.
- Confusions répétées : `cjr → slq` et `zok → sps`, deux fois chacune.

Les taux d’erreur sont plus élevés pour les sessions courtes dans ce découpage. Cela suggère de tester des représentations moins dépendantes de longues séquences : présence des motifs, normalisation séparée des familles de caractéristiques et similarité entre sessions. Ce sont des pistes, pas des gains démontrés.

## Interprétation

Les actions complètes seules n’améliorent pas la référence. L’ajout des séquences au SVM améliore le F1 de CV, et la combinaison avec les arbres apporte un gain supplémentaire. Extra Trees enrichi régresse à 82.03 % en CV et n’est pas retenu. Les arbres de l’ensemble utilisent donc les caractéristiques initiales de `002`.

## Limites

Le deuxième découpage réutilise les sessions de sélection ; il mesure la sensibilité et n’est pas un test indépendant. Les 20 % réservés ont déjà été examinés dans `003`–`005`. Une évaluation imbriquée reste à effectuer pour mieux estimer le biais de sélection. Aucun modèle n’a été choisi après consultation de la validation réservée dans cette expérience. Le type de moyenne F1 de Kaggle reste à confirmer. Les scores SVM transformés par softmax ne sont pas des probabilités calibrées. Aucune donnée de test Kaggle n’a été lue par cette expérience.

## Reproduction

Depuis la racine du projet : `python datamanip/sequence_experiment.py`, ou exécuter `datamanip/006_sequences.ipynb` dans l’ordre. Versions utilisées dans `datamanip/requirements_sequences.txt`. Les fichiers de résultats de cette expérience sont réécrits lors d’une nouvelle exécution.

Les sorties terminales ont confirmé tous les contrôles. Les anciens notebooks n’ont pas été modifiés par l’expérience 006. Le notebook 006 contient une synthèse des résultats mesurés ; ses cellules n’ont pas été exécutées dans Jupyter pendant cette exécution terminale.
