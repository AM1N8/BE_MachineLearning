# Amélioration du classement : décision expérimentale

## Décision : **KEEP BASELINE**

La référence reste **browser_mask_c3**, fichier `submissions/best_verified.csv`.
Aucun nouveau candidat ne satisfait la règle de promotion conservée. Le score public
0,980 est communiqué par l'utilisateur ; aucun nouveau score Kaggle n'a été obtenu.
Atteindre 0,993 n'est donc ni démontré ni promis.

## Résultats réellement exécutés

Sources : `reports/results/improvement/4e7e8c2ad9a6/comparison.csv`,
`confirmation.csv` et fichiers par fold associés. Toutes les transformations sont
ajustées dans le train du fold. Huit folds identiques à la campagne précédente,
puis huit folds avec graine 43 pour la référence et trois candidats. Les temps
sont des moyennes par fold pour ajustement/prédiction, hors diagnostics.

| Expérience | Macro-F1 | Accuracy | Δ F1 | Folds gagnés / 8 | Secondes/fold |
|---|---:|---:|---:|---:|---:|
| Référence | 0,961251 | 0,969350 | — | — | 3,75 |
| Événements seuls + masque navigateur | **0,962986** | 0,969198 | +0,001735 | 3 | 3,85 |
| Intercept moins pénalisé ×10 | 0,959874 | 0,968283 | −0,001377 | 1 | 37,00 |
| Vue binaire rare, poids 0,2 | 0,961127 | **0,969655** | −0,000125 | 1 | 9,16 |
| Racine des poids de classes | 0,960433 | 0,968588 | −0,000819 | 4 | 4,06 |
| SVM local des deux premiers candidats | 0,945510 | 0,958371 | −0,015741 | 0 | 4,53 |

La suppression de la vue navigateur améliore la moyenne F1, mais sa borne corrigée
à 95 % est −0,008017, avec une répétition négative. Elle corrige 19 erreurs et en
introduit 20. Son apparente amélioration vient principalement d'un fold, pas d'un
gain général robuste.

### Confirmation, graine 43 (4×2)

| Expérience | Macro-F1 | Accuracy | Δ F1 | Borne inférieure corrigée 95 % |
|---|---:|---:|---:|---:|
| Référence | 0,959599 | 0,969806 | — | — |
| Événements seuls + masque | 0,957969 | 0,968282 | −0,001631 | −0,006781 |
| Intercept ×10 | 0,959321 | 0,969349 | −0,000279 | −0,002045 |
| Vue binaire rare | 0,959799 | 0,969959 | +0,000200 | −0,000504 |

La vue rare améliore très légèrement l'accuracy sur les deux graines, et le F1
sur la seconde, mais pas suffisamment pour distinguer le gain du bruit. Aucun
paramètre n'a été optimisé sur ces folds ; les deux pistes de frontière ont été
formulées après la première vague et leur nature adaptative est explicite.
Ces graines partagent les mêmes observations : ce ne sont pas deux jeux de données
indépendants. La correction utilise `sqrt(1/8 + 1/3) * std(delta)` et Student à
sept degrés de liberté ; les deux répétitions doivent aussi gagner.

## Enseignements des erreurs

Reproduction exacte : 201 erreurs sur 6 558 prédictions hors-fold ; 78 sessions
erronées dans les deux répétitions. 109 erreurs ont le vrai utilisateur en deuxième
position ; 163 ont une marge inférieure à 0,3, seulement deux une marge supérieure
à 1. L'hypothèse « corriger surtout les erreurs très confiantes » n'est donc pas
le levier principal dans ce corpus.

- Sessions de 20–100 événements : taux d'erreur 7,42 %, contre 0,67 % au-delà de 500.
  Les sessions ≤20 événements ont six erreurs sur huit prédictions : très petit effectif.
- Couverture : proportion moyenne d'événements inconnus 1,94 % pour les erreurs,
  contre 0,26 % pour les bonnes prédictions. Les transitions inédites et la faible
  diversité sont des facteurs associés, sans démonstration causale.
- Chrome : 4,14 % d'erreurs, Firefox : 2,46 %, Edge : 2,00 %. Cela peut refléter
  les classes et les longueurs propres à chaque navigateur ; ce n'est pas une
  preuve d'un effet navigateur intrinsèque.
- Confusions persistantes : notamment flj→skm, ald→fuy, slq→skm. Pas de label
  déclaré faux sur ce seul critère. Les matrices et scores par classe sont enregistrés.
- Les moyennes de marges globales du diagnostic incluent le masque −1e6 : lorsqu'un
  navigateur n'autorise qu'une classe, la marge devient artificiellement énorme.
  Ne pas interpréter cette moyenne comme confiance calibrée. Les deux erreurs de
  forte marge sont un comptage, pas un diagnostic certain de mauvais étiquetage.

## Pourquoi les essais ont échoué

Le reranking local corrige 34 erreurs mais en introduit 106 : isoler deux petites
classes supprime les contre-exemples globaux et dégrade la frontière. La pondération
atténuée corrige trois erreurs contre huit introduites. L'intercept moins régularisé
est environ dix fois plus coûteux et ne corrige aucune erreur sur la première graine.
La conservation de la vue navigateur dans le classifieur reste préférable après
confirmation. La vue rare conserve une piste faible, pas un modèle promu.

## Soumissions et métrique

Le fichier recommandé et son empreinte sont préservés. Aucun
`challenger_verified.csv`/`.json` n'est produit : la condition préalable de validation
n'est pas satisfaite. Le nombre de changements pour un challenger vérifié et la liste
des sessions test modifiées sont **non applicables**, et non zéro. Les modèles rejetés
n'ont pas été sélectionnés en examinant leurs prédictions test.

La page d'évaluation Kaggle est inaccessible (404) et la recherche nécessite une
authentification absente. Le texte local dit « F1 score moyen » sans préciser la
moyenne. Macro-F1 reste le critère local historique, accuracy et F1 pondéré sont
conservés séparément. Les nouvelles expériences ne résolvent pas cette ambiguïté.

## Reproduction

```sh
uv run python -m copilote.improvement analyze
uv run python -m copilote.improvement research
uv run python -m copilote.improvement_boundaries
make test
make lint
```

Le module initial utilise une empreinte des données, du code, des splits et du lock.
La seconde vague enregistre son empreinte dans `boundary_manifest.json`. Les fichiers
par fold permettent de reprendre une interruption. Aucun changement des dépendances,
des données brutes, du modèle historique ou de sa règle de promotion.

## Vérification finale

Vérification finale : 47 tests réussis, lint réussi (34 fichiers), `git diff --check`
réussi. Les 16 avertissements SQLite/Optuna historiques subsistent. Rechargement du
modèle dans un nouveau processus : 324 prédictions identiques, schéma et ordre du
sample respectés, empreinte du CSV inchangée. Trace : `final_verification.json`.

## Trois prochaines recherches à forte valeur

1. **Sessions courtes et contextes implicites** : propagation déterministe du dernier
   contexte écran/configuration connu dans chaque session ; tester un complément
   de features plutôt qu'un reranking qui détruit les frontières globales.
2. **Mutualisation des utilisateurs proches** : régularisation hiérarchique ou modèle
   de mélange avec poids appris en CV interne, en conservant les négatifs globaux.
   Les paires difficiles doivent être découvertes uniquement dans le train interne.
3. **Traces quasi dupliquées et provenance** : mesurer les similitudes train/train et
   réaliser une validation par groupes de traces si des dépendances sont établies ;
   obtenir parallèlement la définition officielle de la métrique et des marqueurs
   temporels. Aucun timestamp absolu ne peut être reconstruit d'un indice d'ordre.
