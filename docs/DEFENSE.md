# Défense — choix techniques et réponses aux questions

Document de soutenance. Toutes les valeurs citées proviennent de
`reports/results/*.json|csv` ; aucune n'est estimée.

## 1. Le problème

- Entrée : une session = quelques centaines d'événements texte (`tNN`, action,
  contexte entre parenthèses/`<>`/`$$`).
- Cible : la variable de regroupement par utilisateur (classification multiclasse
  **très déséquilibrée**, 247 classes, certaines à 4 sessions seulement).
- Conséquence : la métrique de pilotage est la **macro-F1**, complétée par
  l'exactitude, la F1 pondérée, l'exactitude équilibrée, top-3/top-5 et la log-loss.
  L'accuracy seule masquerait les classes rares.

## 2. Chaîne et anti-fuite

1. **Lecture tolérante** (guillemets ou non), validation stricte du nombre de champs.
2. **Tokenisation sans perte** : un test vérifie que chaque token se reconstruit
   à l'identique sur l'intégralité du jeu d'entraînement.
3. **Caractéristiques construites dans les plis** : `TfidfVectorizer`, `min_df`,
   vocabulaires, `StandardScaler` sont ajustés **uniquement sur l'entraînement** du
   pli. Les tests vérifient qu'un token vu seulement dans la réserve est absent du
   vocabulaire, et que le scaler ne change pas après transformation de la réserve.
4. **Réserve verrouillée** : 20 % stratifiés sont retirés **avant** toute sélection ;
   aucune décision de modèle n'observe la réserve.
5. **Aucune transduction** : le test n'est que transformé (`transductive: false`
   dans les métadonnées de soumission).

## 3. Sélection « plus simple à moins d'un écart-type »

- Comparaison de 23 modèles pré-déclarés + ablations par familles de variables.
- `cutoff = meilleur_moyen − écart_type` ; on retient le modèle **le plus simple**
  (champ `complexity`) parmi les candidats à moins d'un écart-type du meilleur.
- Un **vote** ou un **empilement** n'est adopté que s'il gagne **plus d'un
  écart-type** ; sinon on garde la solution simple.
- Résultat mesuré : le modèle linéaire régularisé équilibré (`linear_balanced`,
  `raw`+`browser`) l'emporte ; les ensembles n'apportent pas de gain suffisant.

## 4. Résultats honnêtes

- **Développement** (CV 3 plis sur 80 %) : macro-F1 ≈ 0,943 pour `linear_balanced`.
- **Réserve indépendante** : exactitude ≈ 0,951, macro-F1 ≈ 0,941,
  équilibrée ≈ 0,947, top-3 ≈ 0,989.
- La log-loss native (softmax des marges) est mauvaise (≈ 4,1) ; la **calibration
  sigmoïde** la ramène à ≈ 0,32 **sans changer la classe prédite**.
- La CV répétée complète (4×3) sur tout l'entraînement est **descriptive** : elle
  mesure la stabilité **après** sélection, ce n'est pas une performance de sélection
  imbriquée non biaisée.

## 5. Interprétabilité (sans surinterpréter)

- **Modèle linéaire** : décomposition **exacte** `coefficient × TF-IDF` + intercept.
- **Forêt** : variations de probabilité le long des chemins — **pas** des valeurs de
  Shapley, ne pas les présenter comme telles.
- **Permutation groupée** par famille : on détruit un bloc entier pour préserver les
  dépendances internes (par ex. tous les tokens de contexte ensemble).
- Le retrait du contexte fait chuter la macro-F1 (~0,94 → ~0,78) : le contexte est
  **le** porteur de signal.

## 6. Limites reconnues (à assumer)

- Horodatage `tNN` supposé en secondes ; les durées par action ne sont pas prouvées.
- Forward-fill du contexte = hypothèse comparative, pas une vérité.
- Codes de classes nominaux : ANOVA/Kruskal sont exploratoires, les p-valeurs
  ignorent la dépendance intra-utilisateur.
- Les données brutes ont été suivies par git avant l'intervention ; elles sont
  exclues des snapshots et archives suivants, mais l'historique antérieur subsiste.

## 7. Reproductibilité

- Graine `42`, dépendances épinglées (`uv.lock`, `requirements.txt`), Python ≥ 3.13.
- `make all` reconstruit tous les livrables ; chaque artefact porte une provenance
  (hash de configuration, des sources, des jeux, commit git).
