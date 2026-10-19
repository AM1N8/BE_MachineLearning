# Phase 2 — dépendances et erreurs

## Structure et métrique

3279 sessions, 247 utilisateurs. Les identifiants sont des positions de ligne, sans date absolue ni identifiant parent. « F1 score moyen » reste ambigu ; macro-F1, F1 pondéré et accuracy sont conservés. Aucun utilisateur multi-navigateur observé : hypothèse empirique du masque, non garantie de production.

Aucun doublon exact ni paire proche selon Jaccard des trigrammes ≥0,9 et ratio de longueur ≥0,9. Tous les groupes inférés sont des singletons. Cela ne prouve pas l'indépendance : d'autres proximités et une provenance cachée restent possibles. Regrouper par utilisateur supprimerait les classes à prédire du train, ce qui évaluerait une autre tâche.

Même utilisateur entre lignes voisines : 0.004881, contre 0.005542 attendu aléatoirement. Corrélation de rang index/première horloge : 0.019282. Pas d'ordre chronologique inter-session établi. Les écrans sont explicites sur 288307 événements et absents sur 1981940 : la propagation reste une hypothèse locale à la session.

## Les 78 sessions toujours mal classées

Longueur médiane : 123 événements ; moyenne : 231.217949. Les trois plus courtes ont 0, 1 et 2 événements. Les horloges ne sont pas comptées comme événements. Les voisins et recouvrements par utilisateur sont exportés dans `persistent_sessions.csv`, sans correction manuelle des labels ni utilisation de ces statistiques globales comme features.

| stage | experiment | observations | errors | corrected | introduced |
| --- | --- | --- | --- | --- | --- |
| research | baseline | 156 | 156 | 0 | 0 |
| research | context | 156 | 146 | 10 | 0 |
| research | transitions | 156 | 149 | 7 | 0 |
| research | position | 156 | 150 | 6 | 0 |
| research | edges | 156 | 147 | 9 | 0 |
| research | signatures | 156 | 148 | 8 | 0 |
| research | reranker | 156 | 154 | 2 | 0 |
| confirm | baseline | 156 | 135 | 0 | 0 |
| confirm | context | 156 | 129 | 8 | 2 |
| confirm | edges | 156 | 129 | 6 | 0 |
| confirm | signatures | 156 | 131 | 5 | 1 |
| confirm | reranker | 156 | 132 | 3 | 0 |
| seed44 | baseline | 156 | 136 | 0 | 0 |
| seed44 | context | 156 | 135 | 3 | 2 |
| seed44 | edges | 156 | 130 | 6 | 0 |
| seed44 | signatures | 156 | 133 | 3 | 0 |
| seed44 | reranker | 156 | 135 | 1 | 0 |

Ces sessions ont été repérées sur les plis historiques : leur analyse sur d'autres graines reste descriptive, non indépendante.

## Corrections par longueur (zéro événement inclus)

| stage | experiment | length_bin | count | errors | corrected | introduced |
| --- | --- | --- | --- | --- | --- | --- |
| research | baseline | 0–20 | 8 | 6 | 0 | 0 |
| research | baseline | 21–100 | 876 | 65 | 0 | 0 |
| research | baseline | 101–500 | 3126 | 113 | 0 | 0 |
| research | baseline | >500 | 2548 | 17 | 0 | 0 |
| research | signatures | 0–20 | 8 | 6 | 0 | 0 |
| research | signatures | 21–100 | 876 | 57 | 8 | 0 |
| research | signatures | 101–500 | 3126 | 112 | 7 | 6 |
| research | signatures | >500 | 2548 | 16 | 1 | 0 |
| research | reranker | 0–20 | 8 | 6 | 0 | 0 |
| research | reranker | 21–100 | 876 | 64 | 3 | 2 |
| research | reranker | 101–500 | 3126 | 115 | 0 | 2 |
| research | reranker | >500 | 2548 | 17 | 0 | 0 |
| confirm | baseline | 0–20 | 8 | 6 | 0 | 0 |
| confirm | baseline | 21–100 | 876 | 64 | 0 | 0 |
| confirm | baseline | 101–500 | 3126 | 114 | 0 | 0 |
| confirm | baseline | >500 | 2548 | 14 | 0 | 0 |
| confirm | signatures | 0–20 | 8 | 6 | 0 | 0 |
| confirm | signatures | 21–100 | 876 | 63 | 3 | 2 |
| confirm | signatures | 101–500 | 3126 | 110 | 8 | 4 |
| confirm | signatures | >500 | 2548 | 15 | 0 | 1 |
| confirm | reranker | 0–20 | 8 | 6 | 0 | 0 |
| confirm | reranker | 21–100 | 876 | 64 | 5 | 5 |
| confirm | reranker | 101–500 | 3126 | 115 | 1 | 2 |
| confirm | reranker | >500 | 2548 | 15 | 0 | 1 |
| seed44 | baseline | 0–20 | 8 | 6 | 0 | 0 |
| seed44 | baseline | 21–100 | 876 | 65 | 0 | 0 |
| seed44 | baseline | 101–500 | 3126 | 111 | 0 | 0 |
| seed44 | baseline | >500 | 2548 | 15 | 0 | 0 |
| seed44 | signatures | 0–20 | 8 | 6 | 0 | 0 |
| seed44 | signatures | 21–100 | 876 | 61 | 4 | 0 |
| seed44 | signatures | 101–500 | 3126 | 106 | 8 | 3 |
| seed44 | signatures | >500 | 2548 | 14 | 1 | 0 |
| seed44 | reranker | 0–20 | 8 | 6 | 0 | 0 |
| seed44 | reranker | 21–100 | 876 | 65 | 3 | 3 |
| seed44 | reranker | 101–500 | 3126 | 110 | 2 | 1 |
| seed44 | reranker | >500 | 2548 | 15 | 0 | 0 |

Les dénombrements sont des observations répétées, pas des sessions uniques. Toutes les ablations sont disponibles dans `length_paired.csv`. Le reranker conserve le contexte global, les alternatives hors top-2 et les couvertures OOV, mais ne satisfait pas les critères. Aucun changement de soumission.

Source : `reports/results/phase2/a1d191a9b558`.