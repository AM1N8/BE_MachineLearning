# Complémentarité des modèles

Comparaison des mêmes lignes hors-fold, avec alignement vérifié sur fold, row_id et vérité.

| A | B | Désaccords | A faux, B juste | A juste, B faux | Tous deux faux |
| --- | --- | ---: | ---: | ---: | ---: |
| champion | sqrt_tfidf | 193 | 65 | 73 | 245 |
| champion | original | 72 | 14 | 37 | 296 |
| sqrt_tfidf | original | 230 | 78 | 93 | 240 |

Le gain oracle est une borne descriptive utilisant les étiquettes, jamais un score réalisable annoncé.
Aucun poids ni biais de classe n'est appris sur ces prédictions puis évalué sur les mêmes étiquettes.
Validation indépendante des ensembles : non exécutée à ce stade ; métrique officielle en attente.
