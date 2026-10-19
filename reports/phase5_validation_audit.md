# Phase 5 — Audit de généralisation avant toute nouvelle optimisation

## 1. Conclusion et décision

**L'avantage du modèle burst sur les signatures seules n'est pas établi. Aucune
nouvelle recherche de paramètres ni nouvelle soumission n'est justifiée par cet
audit.** Sur douze nouveaux découpages de 324 sessions, le gain moyen est de
**+0,000172 en macro-F1**, avec un intervalle corrigé à 95 % de
**[−0,003236 ; +0,003580]**. Les signatures seules restent le meilleur fichier
public connu, sans devenir pour autant un modèle statistiquement promu.

Scores publics communiqués par l'utilisateur : baseline **0,980**, signatures
**0,983**, hybride burst de phase 4 **0,970**. Ces observations sont conservées
dans `evaluation.json` ; elles ne servent ni à déduire les étiquettes cachées,
ni à modifier les prédictions de test.

Le diagnostic étayé est une **fragilité de sélection et de mesure** : réutilisation
adaptative des mêmes observations, peu de désaccords entre modèles, gains faibles
devant l'incertitude, et composition en classes très différente selon le petit
échantillon évalué. Une cause précise de la baisse Kaggle n'est **pas** démontrée.
Un diagnostic train/test multivarié atypique reste à expliquer ; les variables
simples et le transfert de vocabulaire ne montrent pas de rupture manifeste.

La recommandation de phase 4 était trop optimiste en présentant le gain moyen
post-hoc comme une raison suffisante de tenter le challenger. Son critère de
promotion préétabli avait pourtant échoué sur les trois protocoles. Le résultat
public négatif est maintenant enregistré sans réécrire les rapports antérieurs.

**Prochaine direction concrète : verrouiller la métrique et le protocole de petit
holdout avant tout nouveau modèle**, puis obtenir, si possible, un lot réellement
nouveau ou les métadonnées de collecte. De nouvelles graines sur les données
actuelles ne constituent pas une validation externe.

## 2. Périmètre, sources et reproductibilité

Résultats canoniques : **`reports/results/phase5/1ed0b3ef63ea/`**. Tous les nombres
de ce rapport proviennent de ses JSON/CSV ; les scores Kaggle y sont explicitement
identifiés comme communiqués par l'utilisateur. Les tables détaillées de phase 4
sont des entrées hachées de l'audit, pas de nouveaux résultats indépendants.

Code : `src/copilote/phase5_audit.py`, `src/copilote/phase5_analysis.py` ; tests :
`tests/test_phase5.py`. Graine 42. Trois modèles **figés avant les réajustements** :

- `baseline` : `browser_mask_c3` ;
- `signatures` : signatures contextuelles, poids 0,3 ;
- `sig_burst_w05` : signatures au poids 0,5 et burst au poids 0,3, challenger exact
  de phase 4.

Aucun balayage de paramètres. Vocabulaire, IDF, sélection chi², masquage navigateur
et classifieur sont ajustés uniquement sur l'apprentissage de chaque découpage.
Le diagnostic de source train/test utilise seulement l'appartenance au fichier,
jamais les identités cachées du test ; il ne modifie aucun modèle de prédiction.

Reconstruction depuis la racine du dépôt, avec les entrées historiques présentes :

```bash
uv run python -m copilote.phase5_audit static
uv run python -m copilote.phase5_audit refit
uv run python -m copilote.phase5_audit summary
uv run python -m copilote.phase5_analysis reports/results/phase5/1ed0b3ef63ea
make lint
make test
```

Les 48 ajustements de modèles d'identité ont été exécutés dans
`reports/results/phase5/4922593685dd/`. Après ajout du module d'analyse, leurs
prédictions ont été réutilisées dans le répertoire canonique **après vérification
des mêmes dépendances de modèles, entrées, modèles sauvegardés et découpages**
(`refit_reuse.json`). La commande `refit` sans option refait les ajustements si les
sorties manquent ; elle reprend les sorties présentes. Les permutations de source
sont également reprenables. Un changement de sources ou d'entrées change la
signature du répertoire ; il faut alors adapter le dernier chemin.

`manifest.json`, `analysis_manifest.json` et `preservation.json` enregistrent la
provenance. Les trois soumissions et les modèles `.joblib` existants sont vérifiés
inchangés par hachage. Aucun export de soumission n'est produit par la phase 5.

## 3. Différences exactes entre soumissions

Source : `submission_differences.csv`. L'alignement des 324 `RowId` est vérifié
contre l'échantillon de soumission, ainsi que les prédictions contre les marges
sauvegardées lors de l'ajustement complet de phase 4.

| RowId | Baseline | Signatures | Hybride | Marge baseline | Marge signatures | Marge hybride |
|---:|---|---|---|---:|---:|---:|
| 125 | zro | zro | bgn | 0,059256 | 0,029548 | 0,019436 |
| 237 | czi | pgm | pgm | 0,055024 | 0,003146 | 0,096920 |
| 290 | usy | usy | vsb | 0,036470 | 0,019766 | 0,003955 |

La marge est l'écart **premier score − deuxième score** du SVC, sans calibration
probabiliste ; sa valeur n'est pas directement comparable entre modèles.

- Signatures contre baseline : **une** ligne différente, 237.
- Hybride contre baseline : **trois** lignes différentes, 125, 237 et 290.
- Hybride contre signatures : **deux** lignes différentes, 125 et 290.
- `bgn` était deuxième des signatures, mais pas de la baseline ; `pgm` était
  deuxième de la baseline ; `vsb` était deuxième des deux modèles de référence.
- Le masque navigateur autorise 114 classes pour chacune de ces trois lignes.

Ces observations ne disent pas quelles modifications sont correctes. Elles
n'autorisent pas à reconstruire les vérités à partir des scores publics arrondis.
En particulier, la différence de score n'est pas convertible en un nombre
d'erreurs sans connaître exactement la métrique et le sous-ensemble public.

## 4. Ce que la métrique officielle permet de conclure

Source disponible : `evaluation.md`, lignes 14–15, reproduite dans
`evaluation.json` : **« La métrique d'évaluation pour ce concours est le F1 score
moyen. »** Le texte ne précise pas explicitement :

- la moyenne macro, pondérée ou micro et son implémentation ;
- le traitement des classes absentes du sous-ensemble évalué ;
- la taille et la composition du sous-ensemble public.

L'interprétation de travail est le **macro-F1 sklearn**, moyenne sur l'union des
classes présentes dans les vérités et les prédictions, avec `zero_division=0`.
C'est l'implémentation historique du projet, **pas une confirmation du scorer
Kaggle exact**. Les simulations la reproduisent et la fonction rapide est testée
contre sklearn, notamment pour les classes seulement prédites.

Sur un petit échantillon, une classe rare peut peser fortement ; une classe
absente des vérités mais prédite intervient aussi dans cette convention. Réduire
la taille du holdout ne fait donc pas mécaniquement monter ou descendre le
macro-F1. Les scénarios de couverture ci-dessous sont nécessaires tant que le
scorer et l'échantillonnage publics ne sont pas documentés.

## 5. Audit des découpages historiques et du processus de sélection

Sources : `historical_splits.csv`, `historical_dependence.json`,
`data_support.json`, `historical_group_crossings.csv`.

### Intégrité et dépendance

- Tous les découpages audités sont disjoints apprentissage/validation et couvrent
  les données ; les vérités OOF et les identifiants correspondent aux données.
- Les trois protocoles de phase 4 totalisent **24 plis**, mais **3 279 sessions**
  seulement : les **19 674 lignes OOF** sont six prédictions de chaque session.
- Le Jaccard moyen entre ensembles de validation est **0,1243**, maximum
  **0,1675**. La dépendance ne disparaît pas en changeant la graine.
- La recherche de phase 4 comptait **19 configurations**, baseline comprise.
  Les graines 43 et 44 ont ensuite servi à choisir le poids final : elles ne sont
  plus des validations intactes après cette sélection.
- Les intervalles corrigés de phase 4 échouaient déjà au critère de promotion.
  L'intervalle positif obtenu en regroupant les 24 plis était explicitement
  post-hoc ; il ne corrige pas toute la sélection adaptative.

### Déséquilibre et corrélations

Les **247 classes** ont de **4 à 75 sessions**, médiane **12** ; **63 classes**
ont au plus 7 sessions. Le navigateur est constant par utilisateur observé.
Le masque est donc un a priori utile appris dans les plis, mais sa stabilité
hors échantillon reste une hypothèse de déploiement.

Le graphe de proximité structurelle utilise les ensembles de tokens bruts
binaires, une similarité cosinus ≥ 0,98, le même navigateur et un rapport de
longueurs entre 0,8 et 1,25. Il trouve **deux paires**, soit **quatre sessions**,
et **3 277 composantes**, de taille maximale 2. Sur les 24 anciens plis,
**14 apparitions de validation** ont un voisin de ce graphe dans l'apprentissage.
Cela ne suffit pas à attribuer l'optimisme général à des quasi-doublons.

Ce graphe est un stress-test, pas une reconstruction de journées de collecte.
Les horloges sont relatives aux sessions ; aucune chronologie ou métadonnée
d'acquisition exploitable n'établit ici les véritables groupes dépendants.
Grouper par identité pour retirer des utilisateurs entiers testerait un autre
problème, incompatible avec cette classification fermée des utilisateurs connus.

## 6. Distributions train/test sans étiquettes cachées

Sources : `shift_scalars.csv`, `shift_browser.json`, `shift_oov.csv`,
`vocabulary_transfer.csv`, `similarity.json`, `adversarial.json`,
`source_permutation_summary.json`.

### Variables observables et vocabulaire

Aucune des huit comparaisons scalaires KS n'est significative après correction
de Holm. La fraction de suffixes est la plus proche d'un écart : p brute
**0,0635**, p ajustée **0,5084**. Les médianes train/test sont notamment :

| Variable | Train | Test |
|---|---:|---:|
| Événements hors horloges | 340 | 365 |
| Durée en secondes | 390 | 402,5 |
| Événements par seconde | 0,8951 | 0,8829 |
| Fraction de suffixes | 0,0909 | 0,0809 |
| Entropie des durées par écran | 2,1763 | 2,1148 |
| Part de durée de l'écran dominant | 0,4379 | 0,4499 |

Ces comptes utilisent le parseur actuel hors horloges ; ils ne doivent pas être
confondus avec des longueurs de documents incluant d'autres tokens.

La répartition navigateur est également proche : p du χ² **0,9781**. La
similarité médiane au plus proche voisin structurel est **0,5722** entre sessions
train et **0,5824** de test vers train, sous les contraintes ci-dessus.

| Vue | Types train, sentinelle comprise | OOV test, par occurrence | OOV des 12 petits holdouts stratifiés |
|---|---:|---:|---:|
| Tokens bruts | 7 088 | 0,1249 % | 0,1366–0,2042 % |
| Signatures | 98 718 | 7,6959 % | 6,659–8,637 % |
| Burst | 175 | 0 % | Table détaillée disponible |

Les signatures ont un vocabulaire combinatoire beaucoup plus vaste ; leur taux
OOV test se situe dans la plage des holdouts. Il n'y a pas d'effondrement de
vocabulaire propre au test dans ces mesures. Zéro OOV burst ne prouve toutefois
pas la stabilité de la relation entre ces tokens et les utilisateurs.

### Diagnostic multivarié atypique : ne pas l'effacer

Un classifieur de source train/test, avec TF-IDF ajusté dans chacun des cinq plis
et régression logistique fixée, utilise les vues brut/signatures/burst et le
navigateur. Son AUC OOF groupée est **0,3801**, avec des AUC par pli entre
**0,3249 et 0,4066** : systématiquement sous le hasard.

Il serait incorrect d'en conclure « aucune différence train/test ». Un diagnostic
secondaire réajuste toute cette procédure sur **39 permutations** des étiquettes
de source. L'AUC moyenne nulle est **0,5063**, les quantiles 2,5–97,5 % sont
**[0,4665 ; 0,5543]**. Le test bilatéral sur `|AUC − 0,5|` donne **p = 0,025**,
résolution minimale de cette simulation.

Ce résultat **exploratoire**, non corrigé pour l'ensemble des diagnostics de
distribution, suggère que l'affectation aux fichiers mérite un examen de sa
provenance. Il ne prouve ni un drift précis, ni son rôle dans la baisse burst.
Une sélection non indépendante des sessions peut aussi intervenir. Inverser
a posteriori l'AUC et annoncer un nouveau séparateur validé serait injustifié.

## 7. Nouveaux holdouts : trois modèles figés, aucun réglage

Sources : `preregistration.json`, `splits.json`, `holdout_metrics.csv`,
`holdout_summary.csv`, `corrected_contrasts.csv`, `new_split_dependence.json`.

### Protocole et limites

Protocole principal : **12** `StratifiedShuffleSplit`, **324** sessions de
validation et **2 955** d'apprentissage, graine 42. La stratification sklearn
retient ici **232 classes vraies**, pas 247, dans chaque petit holdout. Ce n'est
donc pas une reproduction démontrée de la couverture du test Kaggle.

Stress-test secondaire : **4** `GroupShuffleSplit` sur le graphe précédent,
acceptés seulement si toutes les identités de validation sont vues en
apprentissage. Ils contiennent **324–325 sessions**, **166–175 classes vraies**.
Presque tous les groupes sont des singletons : leur différence avec le protocole
principal reflète surtout une autre composition des classes. Une baisse de score
dans ce stress-test ne démontre pas une fuite par quasi-doublons.

Les douze holdouts principaux ont **3 888 apparitions**, mais seulement
**2 260 sessions distinctes** évaluées ; **1 019** sessions n'y apparaissent pas.
Une session évaluée apparaît de une à sept fois. L'intersection moyenne de deux
validations est de **35,44 sessions**. Toutes ces observations ont déjà servi aux
phases antérieures : **zéro nouvelle observation indépendante**.

### Résultats principaux

| Modèle | Macro-F1 moyen | Écart-type entre découpages | Minimum–maximum | Accuracy moyenne |
|---|---:|---:|---:|---:|
| Baseline | 0,965752 | 0,016584 | 0,943021–0,994253 | 0,975566 |
| Signatures | 0,967428 | 0,013829 | 0,946223–0,994253 | 0,976852 |
| Hybride burst | 0,967600 | 0,014211 | 0,946306–0,993059 | 0,976337 |

Pour un contraste apparié, l'erreur-type est corrigée selon
`sd(delta) × sqrt(1/R + n_validation/n_apprentissage)`, avec quantile t à `R−1`
degrés de liberté. Les trois p-values sont ajustées par Holm **dans chaque
famille de découpages**. Les intervalles ci-dessous sont les intervalles corrigés
ponctuels à 95 %, pas des intervalles simultanés. Le critère combine borne basse
positive et p ajustée < 0,05. Cette correction est une approximation de sensibilité
à la dépendance, pas un remplacement de données externes.

| Contraste principal | Gain moyen | IC corrigé à 95 % | Victoires / défaites / égalités | p Holm |
|---|---:|---:|---:|---:|
| Signatures − baseline | +0,001676 | [−0,005051 ; +0,008403] | 5 / 2 / 5 | 1,000 |
| Hybride − baseline | +0,001848 | [−0,004765 ; +0,008461] | 7 / 4 / 1 | 1,000 |
| Hybride − signatures | +0,000172 | [−0,003236 ; +0,003580] | 4 / 4 / 4 | 1,000 |

**Aucun contraste principal ne franchit le critère.** Le gain de l'hybride contre
signatures est pratiquement nul dans ce protocole ; son accuracy moyenne est
même légèrement inférieure. Les signatures montrent une moyenne un peu meilleure
et moins dispersée que la baseline, sans preuve de supériorité robuste.

Dans le stress-test secondaire, les macro-F1 moyens sont **0,948763 / 0,960242 /
0,963146** pour baseline/signatures/hybride. **Signatures − baseline** franchit
le critère numérique local : gain **+0,011479**, IC **[+0,004550 ; +0,018408]**,
p Holm **0,0399**, quatre victoires. L'hybride contre signatures ne le franchit
pas : **+0,002903**, IC **[−0,012160 ; +0,017967]**. Ce résultat secondaire sur
quatre découpages, avec une autre composition des classes, ne remplace pas le
critère principal. La correction de Holm n'a pas été globale aux six contrastes.

### Corrections, nouvelles erreurs et support des classes

Source : `holdout_flips.csv`, `difficulty_segments.csv`. Sur les 3 888 apparitions
des holdouts principaux :

| Contraste | Prédictions différentes | Erreurs corrigées | Erreurs introduites | Changements restant faux |
|---|---:|---:|---:|---:|
| Signatures contre baseline | 18 | 10 | 5 | 3 |
| Hybride contre baseline | 34 | 16 | 13 | 5 |
| Hybride contre signatures | 16 | 6 | 8 | 2 |

Ces effectifs répètent parfois la même session. Ils expliquent pourquoi quelques
changements suffisent à inverser le classement ; ils ne sont pas des essais
indépendants pour un test binomial.

Pour les classes ayant au plus 7 sessions dans le train complet, les **576
apparitions** comptent respectivement **22 / 25 / 29 erreurs**. Pour les classes
de 8 à 15 sessions : **43 / 39 / 35** erreurs sur **1 440** apparitions. Ce segment
rare est préoccupant pour l'hybride mais reste descriptif et post-hoc ; il ne
justifie pas d'apprendre maintenant un seuil de correction à partir de ces mêmes
erreurs.

## 8. Robustesse à la taille du holdout : simulations conditionnelles

Sources : `small_sample_simulations.csv`, `resampling_summary.csv`.
Chaque tirage choisit l'une des **six répétitions OOF complètes** de phase 4,
puis prélève des sessions **sans remise** ; les trois modèles sont comparés sur
exactement les mêmes sessions. **2 000 tirages par scénario** :

- `uniform324` : 324 sessions uniformes, environ **168,01 classes vraies** ;
- `coverage324` : une session de chaque classe, puis complément uniforme, soit
  **247 classes vraies** ;
- `uniform162` : sensibilité à un ensemble plus petit, environ **111,85 classes**.
  **162 n'est pas une estimation de la taille du public Kaggle.**

| Contraste | Scénario | Gain moyen | Quantiles 2,5–97,5 % | Tirages négatifs | Égalités |
|---|---|---:|---:|---:|---:|
| Signatures − baseline | Uniforme 324 | +0,001724 | [−0,008133 ; +0,013068] | 22,20 % | 28,95 % |
| Signatures − baseline | Couverture 247, n=324 | +0,001469 | [−0,005398 ; +0,010122] | 21,45 % | 30,10 % |
| Signatures − baseline | Uniforme 162 | +0,002036 | [−0,011545 ; +0,018337] | 14,20 % | 56,75 % |
| Hybride − signatures | Uniforme 324 | +0,002281 | [−0,010318 ; +0,017317] | 28,65 % | 15,50 % |
| Hybride − signatures | Couverture 247, n=324 | +0,001926 | [−0,008706 ; +0,012727] | 30,45 % | 11,95 % |
| Hybride − signatures | Uniforme 162 | +0,002363 | [−0,016568 ; +0,023833] | 21,45 % | 40,35 % |

Les signes s'inversent souvent malgré un gain moyen positif. Les égalités sont
nombreuses, particulièrement à 162 sessions : leur fréquence doit être considérée
avec celle des pertes. On ne peut pas résumer la fragilité par une probabilité de
perte seule.

Ces quantiles décrivent une **simulation conditionnelle aux prédictions OOF déjà
sélectionnées**, pas un intervalle de généralisation ni une p-value du score
Kaggle observé. Ces anciens modèles ont été entraînés sur environ 75 % des
données, contre environ 90 % pour les nouveaux holdouts et 100 % pour les
soumissions. Aucune probabilité de reproduire la baisse publique de 0,013 n'est
déduite de ces tirages.

### Sensibilité supplémentaire par bootstrap en grappes

Sources : `cluster_bootstrap.csv`, `resampling_summary.csv`. **1 000 tirages**
réattribuent ensemble les poids de toutes les prédictions d'une même session ;
un second bootstrap réattribue ensemble celles d'un même utilisateur. Les
observations répétées ne sont donc pas rééchantillonnées comme indépendantes.

Pour hybride − signatures, les quantiles sont **[−0,003615 ; +0,002838]** par
session et **[−0,007873 ; +0,015724]** par utilisateur. Pour signatures − baseline :
**[−0,001138 ; +0,007269]** et **[−0,006424 ; +0,021826]**. Tous incluent zéro.

Le score est ici calculé sur les prédictions regroupées et repondérées, pas comme
la moyenne des scores par holdout. Le bootstrap utilisateur modifie en outre la
présence des classes et l'effet des classes seulement prédites. Ces estimands ne
sont pas identiques ; ces tirages sont des analyses de sensibilité, pas des
confirmations indépendantes, ni une évaluation sur de nouveaux utilisateurs.

## 9. Trois hypothèses maximum et direction suivante

### H1 — L'optimisation répond à un protocole trop favorable ou mal aligné

**Appui : fort pour la fragilité, incomplet pour le mécanisme Kaggle.** Sélection
sur les mêmes sessions, gain faible et incertain, couverture variant de 166–175
à 232 ou 247 classes selon le protocole. Le modèle burst ne surpasse pas les
signatures sur le critère principal.

**Action prioritaire, préalable à tout modèle :** faire confirmer le scorer exact
et l'échantillonnage public ; figer une seule métrique primaire et son traitement
des classes absentes. Conserver les trois modèles actuels comme références sans
réglage. Pour une prochaine idée unique, enregistrer avant exécution un comparatif
apparié contre signatures, le protocole principal, les sensibilités de couverture
et la règle d'arrêt. Sur les seules données actuelles, toute conclusion restera
exploratoire. Une confirmation de promotion requiert un lot réellement nouveau,
des sessions chronologiques identifiables, ou un jeu de confirmation non encore
utilisé ; changer la graine ne satisfait pas cette exigence.

### H2 — Les blocs additionnels sont moins stables pour les identités peu soutenues

**Appui : descriptif seulement.** Les erreurs sur les classes ≤7 sessions passent
de 22 à 25 puis 29 dans les petits holdouts. Les signatures ont davantage de
combinaisons rares ; le burst peut être corrélé au comportement sans être stable
conditionnellement à l'identité.

**Piste de modélisation, non exécutée :** un bloc additionnel dont l'influence est
réduite lorsque son support d'apprentissage est faible. Ce serait une nouvelle
hypothèse à comparer, pas une règle pour annuler les lignes 125 ou 290. La forme
et le support devraient être fixés avant les prédictions externes ; tout choix
de seuil demanderait une sélection imbriquée. L'audit ne justifie pas de lancer
cette optimisation maintenant.

### H3 — L'affectation train/test dépend de la collecte ou de groupes non observés

**Appui : signal exploratoire de source, mécanisme inconnu.** L'AUC inversée et
son contrôle par permutations appellent un examen de provenance, tandis que les
quasi-doublons détectés sont trop peu nombreux pour expliquer seuls l'écart.

**Condition pour avancer :** obtenir des identifiants de journée, de lot ou une
chronologie réelle, puis tester un découpage groupé fermé en utilisateurs connus.
Sans ces métadonnées, ni des classes cachées supposées ni les horloges relatives
ne peuvent les remplacer. Aucun modèle d'adaptation au test n'est entraîné.

## 10. Décision opérationnelle finale

1. **Conserver `context_signatures_experimental.csv` comme meilleur résultat
   public observé** (0,983 communiqué), avec son statut expérimental ; la baseline
   `best_verified.csv` conserve le statut de champion vérifié du projet.
2. **Conserver `phase4_challenger.csv` et son échec public 0,970 comme résultat
   négatif**, sans annuler sélectivement ses changements à l'aide du leaderboard.
3. **Ne produire aucun nouveau modèle ni CSV en phase 5** : l'implémentation
   conditionnelle demandée n'est pas déclenchée, puisque le critère principal
   n'appuie pas une nouvelle optimisation.
4. Avancer d'abord sur **H1 : métrique, couverture, protocole figé et données de
   confirmation réellement nouvelles**. H2 et H3 restent des pistes conditionnelles,
   pas une nouvelle grille de recherche.

Vérification finale : `make lint` et `make test` ; **67 tests réussis**. Les
soumissions, modèles sauvegardés et rapports des phases précédentes sont préservés.
