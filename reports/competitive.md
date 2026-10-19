# Recherche compétitive — bilan reproductible

**Soumission : `submissions/competitive.csv` — 324 lignes validées.**
Le score Kaggle n'est pas connu. Aucun résultat local ne garantit de battre 0,993.

## Protocole

- Graine 42. 52 variantes évaluées sur les mêmes trois folds
  de développement (2623 sessions).
- Sélection par `macro_f1` : métrique officielle non confirmée.
- Vocabulaire, IDF, statistiques NB, normalisation, augmentation et associations
  utilisateur/navigateur appris exclusivement dans la partie entraînement de chaque fold.
- Les utilisateurs sont les classes à reconnaître : chaque classe reste présente
  en entraînement. Un split excluant des utilisateurs ne mesure pas cette tâche.
- Recherche adaptative : les scores de développement sont optimistes après sélection.
- CV complète répétée et réserve historique : benchmarks descriptifs après gel du choix,
  pas une validation imbriquée ni un nouveau test indépendant.
- Prédiction inductive : aucun ajustement sur le test et aucune transduction.

## Modèle retenu

`browser_mask_c3` : TF-IDF d'événements (unigrammes/bigrammes) et navigateur,
SVM linéaire équilibré, avec masque des classes incompatibles avec le navigateur.
Le masque suppose la stabilité utilisateur/navigateur observée sur le train ;
un navigateur inconnu utilise les scores sans masque. Un changement de navigateur
vers un navigateur déjà connu peut pénaliser cette approche.

Meilleure moyenne brute de développement : `experts_c10`.
Règle retenue : plus simple à moins d'un écart-type du meilleur ;
à complexité égale, meilleure moyenne de développement.

| Évaluation | Modèle | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| Développement 3 folds | baseline | 0.956154 ± 0.011587 | 0.943379 ± 0.015973 |
| CV répétée 4 × 3 | baseline | 0.966148 ± 0.005358 | 0.958467 ± 0.008506 |
| Développement 3 folds | browser_mask_c3 | 0.958824 ± 0.009920 | 0.945621 ± 0.015655 |
| CV répétée 4 × 3 | browser_mask_c3 | 0.968487 ± 0.005593 | 0.960052 ± 0.009443 |
| Réserve historique (656 sessions) | browser_mask_c3 | 0.960366 | 0.952335 |

Gain moyen apparié d'accuracy : 0.002339
(écart-type entre folds : 0.002677) ;
amélioration sur 9 des 12 folds.
Les folds se recouvrent : cet écart-type n'est pas un intervalle de confiance.

## Classement complet et résultats négatifs

Les noyaux non linéaires, NB-SVM, augmentations, n-grammes plus longs et
caractéristiques numériques n'apportent pas de gain robuste dans cette recherche.
Les experts par navigateur donnent la meilleure moyenne brute de développement,
mais leur faible avantage ne suffit pas selon la règle de complexité pré-déclarée.

| Variante | Accuracy développement | Macro-F1 développement |
| --- | ---: | ---: |
| experts_c10 | 0.959966 | 0.948295 |
| browser_mask_c3 | 0.958824 | 0.945621 |
| browser_mask_c10 | 0.958442 | 0.945315 |
| browser_mask_c1 | 0.957298 | 0.945172 |
| browser_mask_c100 | 0.957299 | 0.944970 |
| experts_c1 | 0.954629 | 0.944276 |
| style_w05 | 0.955010 | 0.943630 |
| linear_c10 | 0.957298 | 0.943593 |
| hinge_c10 | 0.956536 | 0.943395 |
| baseline | 0.956154 | 0.943379 |
| linear_c100 | 0.956156 | 0.943158 |
| experts_rare | 0.954629 | 0.942685 |
| experts_trigrams | 0.951579 | 0.942370 |
| linear_c10_unweighted | 0.956537 | 0.942147 |
| rare_bigrams | 0.957681 | 0.942122 |
| numeric_w003 | 0.955392 | 0.941685 |
| start_c10 | 0.957298 | 0.941102 |
| context_c10 | 0.959205 | 0.940255 |
| hinge_c1 | 0.947385 | 0.939247 |
| dedup_c10 | 0.955014 | 0.938973 |
| browser_mask_c03 | 0.947004 | 0.938634 |
| style_w1 | 0.950817 | 0.938532 |
| clean_c10 | 0.952724 | 0.937877 |
| rare_trigrams | 0.953866 | 0.937495 |
| augment_c1 | 0.951578 | 0.937480 |
| trigrams_c10 | 0.953866 | 0.937198 |
| nbsvm_c10_a1 | 0.949675 | 0.936716 |
| fivegrams_c10 | 0.948147 | 0.936417 |
| augment_full_c1 | 0.948530 | 0.934847 |
| logistic_c1000 | 0.953106 | 0.934549 |
| numeric_w01 | 0.951198 | 0.934491 |
| nbsvm_c1_a1 | 0.946244 | 0.934107 |
| augment_c10 | 0.949293 | 0.933138 |
| noidf_c10 | 0.947005 | 0.931783 |
| logistic_c100 | 0.949674 | 0.930980 |
| nbsvm_trigrams | 0.943954 | 0.930604 |
| experts_unweighted | 0.949672 | 0.929633 |
| nbsvm_c1_a01 | 0.948150 | 0.927925 |
| experts_c01 | 0.928702 | 0.925835 |
| binary_c10 | 0.951201 | 0.923899 |
| nbsvm_binary | 0.944335 | 0.920668 |
| numeric_w03 | 0.935184 | 0.918335 |
| nbsvm_c01_a01 | 0.933281 | 0.909568 |
| ridge_a01 | 0.931374 | 0.905804 |
| ovo_linear_c10 | 0.926416 | 0.903684 |
| ovo_linear_c100 | 0.924510 | 0.903130 |
| ridge_a001 | 0.927563 | 0.896877 |
| counts_c10 | 0.911169 | 0.894190 |
| multinomial_c100 | 0.918790 | 0.893588 |
| poly_c100 | 0.913835 | 0.885351 |
| rbf_c100 | 0.897438 | 0.852929 |
| rbf_c100_g2 | 0.580256 | 0.477580 |

## Erreurs de développement par longueur

### `baseline`

| Tokens | Sessions | Accuracy |
| --- | ---: | ---: |
| [0.0, 100.0) | 194 | 0.886598 |
| [100.0, 500.0) | 1230 | 0.937398 |
| [2000.0, inf) | 284 | 0.985915 |
| [500.0, 2000.0) | 915 | 0.986885 |

### `browser_mask_c3`

| Tokens | Sessions | Accuracy |
| --- | ---: | ---: |
| [0.0, 100.0) | 194 | 0.891753 |
| [100.0, 500.0) | 1230 | 0.941463 |
| [2000.0, inf) | 284 | 0.985915 |
| [500.0, 2000.0) | 915 | 0.987978 |

Les sessions courtes concentrent les ambiguïtés ; les marges ne sont pas des
probabilités calibrées. Les extraits bruts restent dans `artifacts/`, exclu du dépôt.

## Reproduction et sources

```bash
uv sync --python 3.13 --locked
make competitive
make lint
make test
```

Sources : `reports/results/competitive/{comparison.csv,selection.json,validation.json,
submission.json,diagnostics_*.json}`. Les expériences sont reprenables grâce aux
empreintes du train, du code, des dépendances et du protocole.
La soumission contient `RowId,prediction`, IDs de 1 à 324 contrôlés contre l'exemple.
