# Décisions

## D01 — Préserver le travail initial
Contexte : notebooks et dépendances modifiés avant cette intervention. Options : remplacer / ajouter.
Choix : nouveau package `copilote`, ancien `be_ml` conservé pour les notebooks historiques.
Pourquoi : ne pas écraser les analyses existantes; la nouvelle chaîne évite les tables larges et corrige les unités temporelles.

## D02 — Format de soumission
Contexte : contrairement au prompt, `data/sample_submission.csv` existe.
Choix : `RowId,prediction`, IDs à partir de 1, contrôlés contre le fichier exemple.
La métrique officielle reste non confirmée : sélection macro-F1, accuracy toujours publiée.

## D03 — Temps et contexte
Contexte : marqueurs `tNN`, sens exact du suffixe `1` non documenté.
Choix : valeurs NN utilisées comme secondes présumées (pas de multiplication par cinq);
somme des incréments positifs avec redémarrage lorsque le compteur diminue, indicateurs de resets.
Forward-fill par champ, remis à zéro en fin de session/démarrage serveur, uniquement comme variante comparative.
Les intervalles entre marqueurs ne prouvent pas l'inactivité; aucune durée précise par action n'est revendiquée.

## D04 — Protocole
Le vocabulaire et toute transformation apprise appartiennent au Pipeline de chaque fold.
Les descriptions déterministes d'une seule session peuvent être précalculées.
Un hold-out stratifié est réservé avant le choix; recherche sur le développement seulement.
La CV répétée complète du modèle sélectionné est descriptive après sélection, pas une estimation imbriquée non biaisée.

## D05 — Chargement et métrique
Contexte : le format réel n'est pas encadré de guillemets et les marqueurs temporels sont déjà en secondes.
Choix : lecture tolérante aux deux formats (guillemets ou non), validation stricte du nombre de colonnes, identifiants de test à base 1 conformes à l'exemple fourni.
La métrique Kaggle reste inconnue : tout est piloté par `cv.selection_metric` dans la configuration, par défaut `macro_f1`, avec l'exactitude publiée systématiquement.

## D06 — Sélection, ensembles et calibration
Contexte : 247 classes, certaines à 4 sessions, 324 lignes de test.
Choix : holdout stratifié verrouillé avant toute observation; sélection sur CV de développement seulement; règle « plus simple à moins d'un écart-type »; un vote ou un empilement n'est retenu que s'il gagne de plus d'un écart-type sur le développement et que sa signature est pré-déclarée.
La calibration sigmoïde est ajustée par validation croisée interne et n'est appliquée que pour l'analyse de confiance, jamais pour ré-optimiser la classe prédite.

## D07 — Explicabilité honnête
Contexte : la forêt et le SVM n'ont pas les mêmes justifications possibles.
Choix : le SVM se décompose exactement en `coefficient × TF-IDF` + intercept; la forêt fournit une décomposition exacte des probabilités le long des chemins, jamais présentée comme des valeurs de Shapley. La permutation est groupée par famille pour préserver les dépendances internes.
