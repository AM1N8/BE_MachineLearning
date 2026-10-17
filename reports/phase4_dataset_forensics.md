# Phase 4 — Forensics du jeu de données (stage A)

Artéfacts : `reports/results/phase4/b0e04d98bc12/` (manifeste signé par hachage des sources et
des données, graine 42). Script : `uv run python -m copilote.phase4_forensics`.
Tous les chiffres cités ci-dessous proviennent de `forensics.json`, `session_timing_train.csv`,
`session_timing_test.csv`, `suffix_by_action.csv`, `tempo_per_user.csv`, `dwell_per_user_screen.csv`.

## 1. Résumé exécutif

- **Le canal temporel est grossier mais réellement associé à l'utilisateur.** Les écarts entre
  événements ne prennent que les valeurs {0, 5, 10, 15} s ; l'horloge ne se réinitialise jamais ;
  les sessions sont actives en continu. Les scalaires de tempo (cadence, bouffée d'événements par
  tick, pauses maximales, concentration du temps sur un écran, taux de suffixe « 1 ») sont
  massivement associés aux utilisateurs (Kruskal–Wallis H jusqu'à 1 313, p ~ 1e-144), mais un
  classifieur linéaire seul n'atteint que 0,060 de macro-F1 (247 classes) : ce signal n'est
  exploitable qu'en **fusion complémentaire** avec le modèle sparse, pas isolément.
- **Le suffixe « 1 » n'est pas « même contexte que précédemment »** : 8,7 % des événements
  porteurs du suffixe ont des groupes explicites. C'est plutôt un marqueur style par action et
  par utilisateur (taux par action de 0,11 à 0,57 ; médiane par utilisateur de 0 à 0,71).
- **47,3 % des sessions train contiennent au moins un token événementiel exclusif à leur
  utilisateur** (calcul in-sample, descriptif) : un potentiel de « snap » fold-local existe mais
  ne couvre pas tout (seuls 12/247 utilisateurs ont un token exclusif dans toutes leurs sessions ;
  7 utilisateurs n'en ont aucun).
- **Aucun shift train/test détectable** sur les caractéristiques temporelles (KS tous p > 0,05).

## 2. Structure de l'horloge

- Différences entre marqueurs consécutifs : **5 s = 95,12 %**, **0 s = 4,87 %** (27 523 paires
  de marqueurs dupliqués), 10 s = 55 occurrences, 15 s = 1. Le tick est donc de 5 s, avec des
  marqueurs parfois dupliqués (plusieurs horodatages identiques).
- **Aucune remise à zéro** (différence négative) dans les 3 279 sessions train et 324 test :
  l'horloge est croissante au sens large sur toute la session. La caractéristique
  `clock_resets` de l'EDA vaut toujours 0.
- **3 276/3 279 sessions train et 324/324 test finissent par des événements après le dernier
  marqueur** (« trailing events ») : le dernier tick n'est jamais refermé par un marqueur.
- Durée cumulée : train médiane 390 s (p90 1 816 s, max 11 785 s) ; test médiane 402,5 s
  (p90 1 642 s, max 7 105 s).
- Écarts entre événements consécutifs (gap = horloge de l'événement suivant − horloge du
  précédent, marqueurs partagés compris) : médiane de session **0 s**, p90 **5 s**, maximum
  observé **15 s** ; répartition des maxima par session : 5 s pour 3 259 sessions, 10 s pour
  16, 0 s pour 3, 15 s pour 1. Les événements arrivent donc en bouffées à l'intérieur d'un même
  tick, puis passent au tick suivant.
- 99,91 % des sessions train (100 % test) ont **chaque tick actif** (au moins un événément) ;
  `idle_fraction` > 0 pour 0,06 % des sessions. Il n'y a **pas de pauses longues exploitables**
  au-delà du pattern {0, 5, 10, 15}.
- Cadence : `events_per_second` médiane 0,895 (p10 0,608, p90 1,325) ; `events_per_active_tick`
  (médiane par session) moyenne 2,83. Total : 2 270 247 événements train, 214 596 test.
- `gap_valid_fraction` médiane 0,9948 : les gaps sont calculables presque partout (pas de resets).

## 3. Association tempo ↔ utilisateur

Kruskal–Wallis par caractéristique (niveaux session, 247 groupes, Bonferroni appliqué) —
`forensics.json:tempo_tests_kruskal_wallis` :

| caractéristique | H | p (Bonferroni) |
|---|---|---|
| dwell_top_fraction | 1 313,2 | 7,5e-144 |
| gap_max | 1 239,9 | 5,7e-131 |
| suffix_fraction | 1 228,8 | 4,9e-129 |
| events_per_second | 1 227,6 | 7,9e-129 |
| nogroup_suffix_rate | 1 200,5 | 4,1e-124 |
| suffix_after_explicit_rate | 940,1 | 1,7e-80 |
| events / markers / events_per_active_tick / duration | 889–916 | < 1e-70 |
| dwell_p90 / gap_p90 | 306,7 | 0,087 (non significatif après correction) |

Sondes de signal (régression logistique standardisée, 5 folds stratifiés, graine 42 —
**exploratoires, pas une sélection**) : tempo seul 0,0142, dwell seul 0,0032, suffixe seul 0,0187,
tout combiné **0,0599** de macro-F1. Conclusion : signal marginal réel mais non linéairement
séparable seul → à tester en blocs sparse complémentaires (famille « tempo » : tokens
action×bouffée, action×suffixe, écran×concentration temporelle), selon la même méthodologie que
la famille signatures (bloc binaire pondéré, sélection chi² dans les plis).

Par utilisateur (`tempo_per_user.csv`, médianes) : durée médiane par utilisateur de 80 s à
4 300 s ; `events_per_second` de 0,481 à 1,479 ; `suffix_fraction` de 0,000 à 0,706.

## 4. Suffixe « 1 » : sémantique

- 355 112 événements porteurs du suffixe en train, soit **15,64 %** des 2 270 247 événements ;
  2 557/3 279 sessions (78,0 %) en contiennent au moins un.
- **L'hypothèse « 1 = même contexte que précédemment énoncé » est falsifiée comme règle
  générale** : 30 866 événements suffixés (8,7 %) portent des groupes explicites
  (écran/config/chaîne). Le suffixe est néanmoins nettement plus fréquent sur les événements
  sans groupes : taux moyen par session 0,179 (sans groupes) contre 0,062 (avec groupes).
- Répartition par action (`suffix_by_action.csv`, 36 actions, 31 concernées) — exemples :
  Affichage d'une arborescence 0,570 ; Sélection d'un élément 0,418 ; Désélection d'un élément
  0,410 ; Action de table 0,400 ; Saisie dans un champ 0,320 ; Affichage d'une dialogue 0,215 ;
  Exécution d'un bouton 0,186 ; Fermeture d'une dialogue 0,176 ; Double-clic 0,143 ; Chainage
  0,140 ; Sélection d'un onglet 0,136 ; Sélection d'un écran 0,114.
- Interprétation retenue (hypothèse de travail, non prouvée) : marqueur **de style par
  action×utilisateur** — fortement associé à l'utilisateur (H = 1 228,8), donc un candidat de
  token `action|suffix=1` / `action|suffix=0` pour la fusion sparse.

## 5. Dwell par écran (contexte rempli)

- 416 clés d'écran distinctes (tuples remplis), 7 413 paires (utilisateur, écran) dans
  `dwell_per_user_screen.csv`.
- `dwell_top_fraction` (part du temps totale passée sur l'écran dominant) : médiane 0,438,
  p90 0,808 — la **concentration temporelle varie fortement** et c'est la caractéristique la
  plus associée à l'utilisateur (H = 1 313,2).
- Exemples de dwell cumulés maximaux : vfo/MAINT 23 805 s (35 visites), wvy/MAINT 23 400 s (12),
  nwt/CRM_COMPTE.CompteForm 22 655 s (20), txk/idem 19 790 s (16).
- `dwell_median` et `dwell_p90` sont quasi constants (0 / 5 s) : seul le **profil de
  répartition** du temps entre écrans discrimine, pas les durées élémentaires.

## 6. Association token ↔ utilisateur (in-sample, descriptif)

- 7 087 tokens événementiels distincts en train ; **2 961 exclusifs à un seul utilisateur**
  (41,8 %), 4 126 partagés.
- 1 552/3 279 sessions (47,3 %) contiennent au moins un token exclusif à leur utilisateur ;
  12/247 utilisateurs en ont dans toutes leurs sessions ; 7 n'en ont aucune.
- Top chi² (matrice binaire session×token, descriptive) : tokens de type
  `Raccourci(fr.infologic.superviseur.client.modules.postpreparation...)` (bez, 17 occurrences),
  `Saisie dans un champ(infologic.global.modules.GL_QUESTIONNAIRE.campagn...)` (zll, 5),
  `Sélection d'un écran(...BlankController)<0...>` (cyi, 110), `...<A...>` (zht, 132), etc.
- **Garde-fou** : ce comptage est in-sample ; toute utilisation prédictive (prior/boost sur
  recouvrement de tokens exclusifs) devra être recalculée fold-local et validée par le critère
  strict (borne inférieure corrigée > 0 sur les trois protocoles).

## 7. Shift train/test

Kolmogorov–Smirnov (`forensics.json:train_test_shift_ks`) : events 0,046 (p 0,54), duration
0,042 (p 0,66), events_per_second 0,058 (p 0,27), suffix_fraction 0,076 (p 0,064 — limite),
idle/active/gap/dwell médians ~0 (p 1). **Aucun shift significatif** : les caractéristiques
temporelles mesurées en train se transposent au test.

## 8. Implications pour la Phase C (hypothèses testables, non encore évaluées)

1. **Famille « tempo » (fusion sparse)** : tokens `action|burst=k` (bouffée par tick bucketisée),
   `action|suffix=1`, `screen|dwell_share_bucket` — bloc binaire + chi² dans les plis, poids de
   bloc à caler comme pour signatures (0,3). Premier candidat : c'est le seul canal jamais testé.
2. **Scalaires de tempo en tokens bucketisés** (gap_max 5/10/15, events_per_second décile) plutôt
   qu'en bloc dense numérique (les denses ont perdu en phase 2).
3. **Snap fold-local sur tokens exclusifs** : prior ou correction sur marge uniquement quand le
   recouvrement avec les tokens exclusifs du fold d'entraînement est univoque — direction 4
   (correction résiduelle sensible au score, OOF imbriquée).
4. **Séquence intra-tick** : l'ordre des événements dans un tick de 5 s est possiblement
   arbitraire (générateur) — à vérifier avant d'investir dans des n-grammes d'ordre strict.

## 9. Limites

- Le dwell et les gaps reposent sur des différences d'horloge cumulée (proxys) ; les marqueurs
  dupliqués (0 s) rendent la granularité réelle ≥ 5 s.
- Les associations KW/chi² sont descriptives (in-sample, sans correction de la dépendance
  intra-utilisateur entre sessions d'un même utilisateur) ; la valeur prédictive ne sera établie
  que par les expériences contrôlées de la phase suivante, sur les mêmes splits que la phase 2
  (historique 4×2, graines 43 et 44).
