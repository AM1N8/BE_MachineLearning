# Features et précautions

Toutes les transformations apprises sont dans des pipelines sklearn clonés par
pli : vocabulaire, seuil de fréquence, IDF, encodage navigateur, échelle, SVD,
sélection éventuelle et estimateur. Aucune transduction n'est activée.

| Famille | Construction | Motivation / limite |
|---|---|---|
| raw | Événements exacts, hors tNN; TF-IDF unigrammes/bigrammes/trigrammes | Signature riche; contexte rare et grande dimension |
| action | Verbe sans annotations et sans suffixe final 1 | Mutualiser les contextes; possible perte d'identité |
| screen/config/chain | Tous les groupes explicites (), <> et $$ | Habitudes fonctionnelles; les noms ne prouvent pas leur sémantique |
| pairs | Action associée à chaque contexte explicite | Interaction interprétable |
| filled | Contexte propagé dans une seule session | Hypothèse optionnelle; reset serveur/déconnexion |
| position | Trois premières/dernières actions | Routines d'entrée/sortie |
| time | Bins fixes log2(1+tNN) | Échelle temporelle robuste; pas d'horodatage absolu |
| browser | One-hot appris via vocabulaire de train | Filtre de candidats, pas identifiant unique |
| numeric | Comptages, diversité, entropie, répétitions, fractions de contexte, proxys temporels | Signature comportementale compacte; log1p puis échelle fold-local |
| char | N-grammes caractères 3–5 sur événements | Robustesse lexicale; coût et interprétation plus difficiles |

Les n-grammes respectent les frontières d'événements : une action contenant des
espaces reste un token. Aucun remplissage global n'est appliqué aux lignes ragged.
La succession des contextes explicitement observés peut sauter des événements non
annotés; elle n'est donc pas décrite comme une transition certaine de l'interface.

`make features` produit histogrammes, boxplots pour les codes **catégoriels**
0, 15, 152 et 199, corrélations de Spearman ordonnées par clustering, projection
SVD descriptive et ANOVA/Kruskal avec tailles d'effet et correction Bonferroni.
Cette analyse du train entier n'est pas utilisée comme sélection de variables.
Le mapping des codes est conservé dans `reports/results/features.json`.

Tests anti-fuite : tokens/navigateur uniquement en validation absents du
vocabulaire, échelle inchangée après transform, clone non ajusté, reset des
contextes entre sessions, matrices finies même pour une session vide.
