# Phase 2 — protocole figé avant résultats

## Hypothèses ciblées

La métrique fournie est « F1 score moyen », sans précision sur la moyenne.
Macro-F1 reste le critère principal ; accuracy et F1 pondéré sont conservés.
Les scores publics rapportés ne participent ni au réglage ni à la sélection.

Cinq ajouts indépendants au modèle `browser_mask_c3` : contexte propagé dans
la session ; transitions entre états action/contexte distincts ; position en
quartiles de ces états ; états des 20 premiers/derniers événements avec poids
de début renforcé ; signatures binaires action/contexte sélectionnées par chi².
Poids additionnel fixé à 0,3, sans recherche externe. Vocabulaire, IDF et chi²
sont ajustés seulement sur le train du pli. La propagation est une hypothèse,
avec remise à zéro entre sessions, après fermeture et au démarrage serveur.
Les signatures de position/extrémités brutes ont déjà été testées : ici seul le
couplage au contexte propagé constitue l'information nouvelle.

## Reranker global top-2

Chaque train externe produit trois plis internes. Chaque score et chaque profil
utilisés pour entraîner le reranker proviennent d'un modèle n'ayant jamais vu
la session correspondante. Signaux : trois scores globaux, deux écarts, score
standardisé sur les classes compatibles, longueur, diversité, OOV, rareté,
vraisemblances unigrammes et transitions conditionnelles des deux candidats,
couverture des événements par classe et compatibilité navigateur.
Régression logistique régularisée C=0,1, sans rééquilibrage artificiel.
Les observations dont la vérité n'est dans aucun des deux candidats restent
des exemples négatifs : elles ne sont pas supprimées.

La logistique est elle-même cross-fittée pour choisir le seuil dans le train
externe : marges admissibles 0,1/0,3, probabilités 0,5/0,65/0,8/0,9.
Le choix nul (aucune correction) gagne les égalités. Un seuil doit améliorer
le macro-F1 interne sans diminuer l'accuracy interne. Les prédictions internes
servent au réglage, jamais à une estimation indépendante : les modèles de base
des différents plis internes peuvent partager des observations. L'évaluation
externe n'intervient jamais dans cet ajustement.

## Confirmation et décision

Huit plis historiques appariés (4×2), puis graine 43 pour toute hypothèse avec
gain de macro-F1 ou accuracy et pour le reranker même sans gain. Graine 44 et
stress test groupé, si des groupes non triviaux sont établis, pour ces mêmes candidats. Un groupe est défini sans labels
par Jaccard des trigrammes ≥0,9 et ratio de longueur ≥0,9.
Le groupement est un stress test de proximité, pas une provenance démontrée.

Promotion inchangée : borne inférieure corrigée à 95 % du gain macro-F1 >0,
avec correction `sqrt(1/8+1/3)`, et gains positifs dans les deux répétitions.
Exiger ce résultat sur les validations historique et indépendante ; vérifier
également l'absence de régression sur les contrôles supplémentaires. Aucun
assemblage des nouvelles vues ne sera sélectionné sur les plis externes.
Le CSV `best_verified.csv` reste intact. Pas de challenger sans les critères.

## Reproduction

```bash
uv run python -m copilote.phase2 audit
uv run python -m copilote.phase2 research
uv run python -m copilote.phase2 confirm
uv run python -m copilote.phase2 seed44
uv run python -m copilote.phase2 grouped
```

Sources, données, dépendances et splits sont hachés ; résultats et listes de
sessions internes/externes sous `reports/results/phase2/<empreinte>/`.


## Résultats exécutés

**Décision : KEEP BASELINE.** Aucun challenger produit. 136 évaluations externes ; trois protocoles 4×2 sur les mêmes données, pas trois jeux de données indépendants.

### research

| experiment | macro_f1 | accuracy | weighted_f1 | delta_f1 | lower95 | corrected | introduced | passes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.961251 | 0.969350 | 0.966464 | 0.000000 | 0.000000 | 0 | 0 | False |
| context | 0.961936 | 0.969350 | 0.966715 | 0.000684 | -0.003465 | 15 | 15 | False |
| transitions | 0.960788 | 0.968893 | 0.966087 | -0.000464 | -0.003305 | 11 | 14 | False |
| position | 0.960783 | 0.968435 | 0.965635 | -0.000468 | -0.005330 | 9 | 15 | False |
| edges | 0.962394 | 0.970417 | 0.967628 | 0.001143 | -0.009486 | 21 | 14 | False |
| signatures | 0.962920 | 0.970875 | 0.968248 | 0.001669 | -0.001375 | 16 | 6 | False |
| reranker | 0.960846 | 0.969198 | 0.966318 | -0.000406 | -0.002121 | 3 | 4 | False |

### confirm

| experiment | macro_f1 | accuracy | weighted_f1 | delta_f1 | lower95 | corrected | introduced | passes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.959599 | 0.969806 | 0.966681 | 0.000000 | 0.000000 | 0 | 0 | False |
| context | 0.961364 | 0.970873 | 0.968061 | 0.001765 | -0.001707 | 17 | 10 | False |
| edges | 0.961734 | 0.970569 | 0.967794 | 0.002134 | -0.003821 | 20 | 15 | False |
| signatures | 0.960541 | 0.970416 | 0.967388 | 0.000941 | -0.002276 | 11 | 7 | False |
| reranker | 0.959265 | 0.969501 | 0.966348 | -0.000335 | -0.004815 | 6 | 8 | False |

### seed44

| experiment | macro_f1 | accuracy | weighted_f1 | delta_f1 | lower95 | corrected | introduced | passes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.961707 | 0.969961 | 0.967449 | 0.000000 | 0.000000 | 0 | 0 | False |
| context | 0.961323 | 0.970266 | 0.967792 | -0.000384 | -0.003863 | 10 | 8 | False |
| edges | 0.961186 | 0.970571 | 0.967919 | -0.000521 | -0.005818 | 19 | 15 | False |
| signatures | 0.963030 | 0.971486 | 0.969044 | 0.001322 | -0.001729 | 13 | 3 | False |
| reranker | 0.961671 | 0.970113 | 0.967519 | -0.000037 | -0.001675 | 5 | 4 | False |

Les signatures contextuelles gagnent sur les trois protocoles et les six répétitions : 40 corrections contre 16 dégradations sur 19 674 prédictions répétées. Cependant aucune borne corrigée n'est positive : hypothèse prometteuse, non promue. Les transitions et positions échouent dès l'ablation historique. Le contexte et les extrémités ne confirment pas leur macro-F1 avec la graine 44. Aucun assemblage n'a été ajouté après examen des scores externes.

Le reranker corrige 14 erreurs mais en crée 16 : moins destructeur que l'ancien classifieur local, mais sans gain reproductible. Ses seuils sont choisis exclusivement dans le train externe. Les probabilités internes ne constituent pas une garantie de calibration externe.

Les validations réutilisent des données déjà explorées : le biais adaptatif résiduel demeure. Les bornes corrigées sont un garde-fou conservateur, pas une preuve statistique indépendante. Le score Kaggle futur est inconnu.

Sources numériques : `reports/results/phase2/a1d191a9b558/*.json|csv`. `decision.json` vérifie les empreintes, les 24 audits internes et la reproduction exacte des 324 prédictions conservées. Reproduction complète : `make phase2`. Les tests groupés n'ont pas été exécutés, aucun groupe non trivial n'étant établi.