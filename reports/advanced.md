# Deuxième recherche : similarité et profils utilisateur

Modèle retenu : `hellinger_svm_c10` ; meilleure moyenne brute : `hellinger_svm_c10`.
Sélection macro-F1 sur trois folds de développement ; métrique officielle non confirmée.
Toutes les transformations sont ajustées dans les folds. Aucune transduction.
CV répétée et réserve historique : descriptives après sélection, pas un test indépendant.

## Développement

| Modèle | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| hellinger_svm_c10 | 0.959203 | 0.948678 |
| experts | 0.959966 | 0.948295 |
| hellinger_svm | 0.957296 | 0.948137 |
| hybrid_kridge_02 | 0.959586 | 0.946239 |
| previous | 0.958824 | 0.945621 |
| hybrid_centroid_02 | 0.956918 | 0.944444 |
| hybrid_nearest_02 | 0.958442 | 0.944337 |
| hybrid_kridge_05 | 0.956537 | 0.941377 |
| kridge_balanced | 0.954247 | 0.940263 |
| hellinger_kridge | 0.952723 | 0.936685 |
| hybrid_nearest_05 | 0.952341 | 0.935795 |
| bm25_svm_b0 | 0.955013 | 0.930523 |
| bm25_svm | 0.954251 | 0.930366 |
| entropy_svm_soft | 0.951198 | 0.928761 |
| bm25_svm_c10 | 0.953869 | 0.927474 |
| kridge_d1_a01 | 0.949293 | 0.924875 |
| kridge_d2_a001 | 0.947387 | 0.923417 |
| useridf_svm | 0.948531 | 0.923201 |
| useridf_svm_c10 | 0.948531 | 0.922241 |
| kridge_d2_a01 | 0.946243 | 0.921552 |
| entropy_svm | 0.943958 | 0.915654 |
| centroid | 0.922218 | 0.912748 |
| entropy_svm_c10 | 0.943194 | 0.912327 |
| kridge_d3_a01 | 0.935567 | 0.909852 |
| kridge_d3_a1 | 0.930611 | 0.906333 |
| kridge_d4_a01 | 0.927940 | 0.904023 |
| nearest3 | 0.913452 | 0.897342 |
| nearest | 0.911926 | 0.896419 |
| hellinger_centroid | 0.886764 | 0.894272 |
| hellinger_nearest | 0.902016 | 0.889342 |
| binary_nearest | 0.910021 | 0.886823 |
| cosine_svc_d2 | 0.773537 | 0.716681 |
| cosine_svc_d3 | 0.487997 | 0.403755 |

## Validation répétée (4 × 3)

| Modèle | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| previous | 0.968487 ± 0.005593 | 0.960052 ± 0.009443 |
| hellinger_svm_c10 | 0.967673 ± 0.005289 | 0.959197 ± 0.007077 |

Réserve historique : accuracy 0.961890, macro-F1 0.945851.
Gain apparié moyen d'accuracy : -0.000814, 3 folds améliorés.

Soumission : `submissions/advanced.csv`. Score Kaggle inconnu ; dépasser 0,993 n'est pas établi.
Sources : `reports/results/advanced/*.json|csv`. Reproduction : `make advanced`.
