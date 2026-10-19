# Audit préalable — campagne d'amélioration

## État et provenance

Projet relu : PROMPT, PROGRESS, décisions, règles, description d'évaluation,
parseur, représentations, BrowserRestricted, recherches competitive/advanced/extreme,
diagnostics hors-fold et artefacts de soumission. Les notebooks historiques sont
des analyses descriptives ; la reproduction repose sur les modules du package.
Les résultats existants restent conservés. La référence est `browser_mask_c3`.
Le score public 0,980 et le classement sont communiqués par l'utilisateur, non
vérifiés par une requête authentifiée, et ne servent pas à ajuster les modèles.

La page Kaggle d'évaluation renvoie HTTP 404 depuis cet environnement ; le moteur
de recherche n'est pas authentifié. `evaluation.md` dit « F1 score moyen » sans
préciser macro/micro/pondéré. **La configuration exacte reste non vérifiée.**
Rapporter séparément macro-F1, accuracy et F1 pondéré ; ne pas déduire la métrique
des scores arrondis du leaderboard. Une capture/export de la configuration par
l'organisateur serait nécessaire pour lever cette incertitude.

## Données et modèle

Sessions CSV irrégulières : navigateur, séquence d'événements, utilisateur cible.
3 279 sessions, 247 classes, 324 tests ; classe minimale de quatre observations.
Les tokens contiennent actions et contextes écran/configuration/chaîne. `tNN`
est un marqueur de compteur dont la sémantique temporelle n'est pas garantie :
pas d'heure civile, de jour de semaine ni de durée fiable inventés.
La référence retire ces marqueurs, ajuste TF-IDF sublinéaire unigrammes/bigrammes
(min_df=2, plafond 60 000), une vue navigateur et LinearSVC équilibré C=3.
Le masque apprend uniquement le support navigateur/utilisateur du train du fold ;
un navigateur inconnu ne restreint pas les classes.

## Recherches déjà effectuées

- 52 variantes competitive : C, pertes, représentations binaires/comptages,
  n-grammes jusqu'à cinq, contextes, préfixes, statistiques, experts navigateur,
  SVM OVO linéaire/RBF/polynomial, régressions logistiques, ridge, NB-SVM, augmentation.
- 33 variantes advanced : voisins, centroïdes, noyaux de similarité, Hellinger,
  BM25, IDF utilisateur, entropie supervisée, hybrides.
- Extreme : vues séparées, skip-grams, positions, débuts/fins, répétitions,
  caractères intra-événement, hiérarchie, rythme proxy, chi2, SVD, modèles Markov ;
  recherche Optuna bornée et sélection/ensemble imbriqués.
- Les résultats de référence sont dans `reports/results/best_submission.json`.
  La dernière politique d'ensemble n'a pas franchi la règle de promotion.

## Risques de validation

Le même corpus a servi à plusieurs recherches adaptatives. Les folds externes
déjà examinés ne sont pas un nouveau test indépendant. L'utilisateur est la
classe recherchée : GroupKFold par utilisateur rendrait les classes inconnues et
ne répondrait pas au problème. Aucun identifiant documenté de groupe indépendant
ou date absolue ne justifie actuellement une séparation temporelle. L'absence de
doublons exacts ne prouve pas l'indépendance de traces proches. Le masque navigateur
suppose une stabilité en déploiement. Les marges SVM ne sont pas des probabilités.

## Plan priorisé, fixé avant les nouveaux résultats

1. Reproduire les huit folds externes exacts et enrichir les diagnostics : couverture
   du vocabulaire du train, rareté, transitions inédites, marges, stabilité et groupes.
2. Tester trois mécanismes non redondants : pondération de classe intermédiaire
   (racine des poids équilibrés), classification locale des deux classes concurrentes
   sur les faibles marges (seuil fixe 0,3, C=1), ajout léger d'une vue binaire rare
   (min_df=1, poids fixe 0,2) au modèle conservé. Aucun paramètre n'est choisi sur le test.
3. Comparaisons appariées sur les huit folds ; pour les pistes positives, confirmation
   avec une graine distincte (43), 4×2 folds. Les paramètres restent fixés : pas de
   tuning ou d'ensemble appris nécessitant une nouvelle boucle interne.
4. Promotion seulement si borne corrigée à 95 % positive et gains dans les deux
   répétitions, puis confirmation indépendante positive selon la même règle.
   Le choix parmi plusieurs pistes reste adaptatif ; publier cette limite.
5. Aucun CSV challenger déclaré vérifié si ces conditions échouent. Préserver le
   CSV recommandé ; pas d'envoi automatique à Kaggle ni correction manuelle.

Les modèles d'arbres denses et réseaux profonds ne sont pas prioritaires : les
statistiques seules ont déjà déçu et les classes ont très peu d'exemples.
