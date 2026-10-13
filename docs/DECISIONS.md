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
