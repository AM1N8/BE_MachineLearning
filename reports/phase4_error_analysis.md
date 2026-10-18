# Phase 4 — Analyse d'erreurs (historique 8 folds, baseline `browser_mask_c3`)

**Objectif.** Caractériser les 201 erreurs du champion vérifié sur le protocole historique
(4×2 RepeatedStratifiedKFold, seed 42, 6 558 prédictions OOF), y rejoindre les canaux de
tempo/suffixe issus de la forensic (Phase A), et **falsifier ou confirmer avant modélisation**
les mécanismes de sauvetage envisagés (tokens exclusifs fold-locaux, reranking top-2,
fusion de marges).

**Sources (artefacts signés).** Module `src/copilote/phase4_errors.py` ; résultats dans
`reports/results/phase4/b2d89ec23226/` (manifeste : hash des sources + entrées + seed 42).
Entrées : `reports/results/improvement/4e7e8c2ad9a6/baseline_oof.csv` (OOF 8 folds avec
diagnostics), `research_splits.json` (mêmes folds, vérifiés identiques aux splits phase 2 et
extremes), `reports/results/phase4/d524c6ba3719/session_timing_train.csv` (scalaires
tempo/suffixe/dwell par session), OOF phase 2 `research_baseline_{0..7}.csv` et
`research_signatures_{0..7}.csv`. Toute exclusivité de token est **recomputée fold-par-fold
sur la moitié d'entraînement uniquement** (aucune consultation de la session évaluée) ;
la simulation « snap » est donc fold-locale par construction. Les similitudes par paire
d'utilisateurs restent descriptives ( plein entraînement, hors validation).

## 1. Chiffres de base (protocole historique)

| Quantité | Valeur |
|---|---|
| Prédictions OOF | 6 558 (8 folds, 820/819 par fold) |
| Erreurs (instances) | 201 |
| Sessions uniques en erreur | 123 |
| Sessions fausses aux 2 répétitions (« toujours-faux ») | 78 |
| Erreurs avec vérité en top-2 (`second == truth`) | 109 |
| Erreurs avec marge < 0,3 | 163 |
| Paires de confusion distinctes | 113 (dont 55 avec ≥ 2 erreurs) |

## 2. Contraste erreurs vs prédictions correctes (médianes)

| Diagnostic | Erreurs | Correctes | Lecture |
|---|---|---|---|
| Longueur (événements) | 130 | 353 | erreurs ≈ 2,7× plus courtes |
| Tokens distincts | 26 | 49 | appauvrissement lexical |
| Entropie (bits) | 3,76 | 4,41 | moins diversifiées |
| OOV (moyenne) | 0,0194 | 0,0026 | 7,4× plus d'inconnus |
| Rares (moyenne) | 0,0231 | 0,0065 | 3,6× plus de rares |
| Marge | 0,108 | 1,152 | décisions serrées |
| Durée (s) | 155 | 400 | sessions courtes |
| **`suffix_fraction`** | **0,0204** | **0,0918** | **4,5× moins de suffixe « 1 »** |
| `suffix_events` (nombre) | 4 | 35 | idem, en absolu |
| **`dwell_top_fraction`** | **0,571** | **0,436** | **plus concentrées sur l'écran dominant** |
| `events_per_second` | 0,854 | 0,896 | proche |
| `gap_max` (s) | 5 | 5 | sans signal |

Nouveauté par rapport à la phase 1 : `suffix_fraction` et `dwell_top_fraction` — issus de la
forensic Phase A — sont les discriminateurs les plus nets jamais mesurés côté scalaires
(absents des features du champion). Les erreurs sont des sessions **suffixe-pauvres** et
**mono-écran**.

## 3. Couverture par tokens exclusifs fold-locaux — le signal est anti-corrélé

Pour chaque fold, un token (vue « raw » = événement brut ; vue « sig » = signature
(action, écran, config, chaîne) à contexte rempli) est dit exclusif d'un utilisateur s'il
n'apparaît dans **aucune** session d'entraînement d'un autre utilisateur.

| Mesure | Erreurs (201) | Correctes (6 357) |
|---|---|---|
| ≥ 1 token exclusif du vrai user (raw) | 36 (17,9 %) | 26,9 % |
| ≥ 1 token exclusif du vrai user (sig) | 54 (26,9 %) | 69,6 % |
| Nombre moyen de tokens exclusifs (raw) | 0,28 | 0,58 |

**Lecture décisive : les sessions correctes ont PLUS de support exclusif que les erreurs.**
Le TF-IDF du champion exploite déjà implicitement ces tokens ; les erreurs sont précisément
les sessions *sans* ce soutien. L'hypothèse forensic « 47,3 % des sessions ont un token
exclusif » (in-sample) ne se transfère pas en mécanisme de sauvetage : c'est un descripteur
de facilité, pas un résidu exploitable.

## 4. Simulation « snap » (règle dure) — falsifiée, toutes variantes nettes négatives

Règle : si exactement un candidat `u ≠ prédiction` possède des tokens exclusifs présents dans
la session (min `k` tokens, optionnellement marge < 0,3), remplacer la prédiction par `u`.
Simulation fold-locale sur les 6 558 OOF (diagnostic exploratoire ; toute adoption réelle
passerait par le moteur phase C/D).

| Variante | Corrigées | Cassées | Macro-F1 après (avant 0,96125) |
|---|---|---|---|
| raw, min 1 | 27 | **950** | 0,81820 |
| raw, min 2 | 8 | 61 | 0,95749 |
| raw, min 1, marge < 0,3 | 18 | 55 | 0,95864 |
| sig, min 1 | 15 | 432 | 0,89714 |
| sig, min 2 | 8 | 117 | 0,94831 |
| sig, min 1, marge < 0,3 | 9 | 64 | 0,95456 |

Mécanisme du désastre : un token jamais vu pour son vrai utilisateur dans la moitié
d'entraînement (rareté, 247 classes) apparaît « exclusif » d'un autre utilisateur ; la
variante la plus lâche déclenche 1 000 fois et casse 950 prédictions justes. La
**direction 4 (correction résiduelle par snap exclusif) est falsifiée comme override dur** ;
au mieux ce signal pourrait entrer comme feature faible d'un modèle appris — la couverture
(§3) montre qu'il n'y a presque rien à y gagner.

## 5. Taxonomie (catégories mutuellement exclusives, priorité décroissante)

| Catégorie | Effectif | Définition |
|---|---|---|
| `short_length` | 6 | longueur ≤ 20 événements |
| `exclusive_missed` | 36 | ≥ 1 token exclusif du vrai user (raw), pas courte |
| `top2_recoverable` | 86 | vérité en top-2, sans les deux conditions précédentes |
| `close_margin` | 60 | marge < 0,3, vérité hors top-2 |
| `confident_wrong` | 13 | marge ≥ 0,3, aucun soutien : noyau irréductible apparent |

Les 36 `exclusive_missed` (dont 9/11 pour la paire flj→skm) confirment §3 : le support
exclusif ne suffit pas à corriger une décision quand il est noyé par le reste du profil.

## 6. Paires de confusion récurrentes (top 10)

Toutes les paires du top 10 sont **same-browser** (le masque navigateur du champion
concentre donc bien les erreurs résiduelles dans les sous-populations, pas entre elles).
Faits saillants :

- **flj→skm** (11 erreurs, 7 avec vérité en top-2, 9 avec support exclusif) : le vrai
  utilisateur flj (42 sessions, médiane 21 tokens distincts) ressemble à skm (75 sessions,
  médiane 20) ; Jaccard raw 0,221. C'est la cible n° 1 d'un reranker.
- **azn→wht** (4) : azn n'a que **4 sessions** au total, vérité en top-2 sur les 4 —
  classe quasi-inductive, reranking seul espoir.
- **ngs→slq** (4) : vérité **jamais** en top-2 — paire intrinsèquement difficile.
- ald→fuy (6), lfq→czi (4, vérité en top-2 3/4, 4 avec support exclusif), zok→lra (4),
  slq→skm (5), azx→guh (4), vyu→oec (4, 4/4 top-2), xby→xgv (4, 4/4 top-2).

## 7. Sessions toujours-fausses (78)

| Catégorie | Effectif | Définition (mutuellement exclusive, priorité décroissante) |
|---|---|---|
| `snap_fixable` | 14 | candidat exclusif unique == vérité dans ≥ 1 fold (dont 9 ont aussi la vérité en top-2) |
| `rerankable` | 32 | vérité en top-2 dans ≥ 1 répétition, hors `snap_fixable` |
| `hard` | 32 | jamais en top-2, jamais de candidat exclusif unique == vérité |

Un reranker parfait limité au top-2 atteindrait au mieux 41 des 78 sessions toujours-fausses
(32 catégorisées `rerankable` + 9 des 14 `snap_fixable` avec aussi la vérité en top-2, vérifié
dans `always_wrong.csv`) ; 37/78 restent hors de portée du top-2 actuel.

## 8. Delta signatures (protocole historique, recompté fold-par-fold)

Baseline → signatures contextuelles : **16 corrigées / 6 introduites** (net − 10 erreurs).
9 des 16 corrigées portaient un token exclusif du vrai user (raw), 0 des 6 introduites —
le bloc signatures capte du signal là où il existe, sans casser les sessions soutenues.

## 9. Implications pour la phase C

1. **Abandonner la direction « snap exclusif »** (falsifiée §4) ; ne pas réimplémenter en
   override. Un usage résiduel comme feature faible d'un correcteur appris est peu prometteur
   (§3) et reste optionnel en dernier recours.
2. **Prioriser la famille tempo/suffixe sparse** : `suffix_fraction` (0,020 vs 0,092) et
   `dwell_top_fraction` (0,571 vs 0,436) sont les meilleurs discriminateurs inédits ;
   hypotheses A-1/A-2 de la forensic (tokens `action|suffix=1/0`, `action|burst=k`,
   `screen|dwell_share_bucket`, déciles `events_per_second`) à tester en bloc binaire
   pondéré type signatures (poids ~0,3, chi² fold-local).
3. **Reranking/fusion de marges** : 86 erreurs `top2_recoverable` + paires à vérité
   systématiquement en top-2 (azn, oec, xgv) suggèrent une fusion de scores
   baseline+signatures (le reranker supervisé phase 2 avait échoué : 14 corrigées / 16
   introduites ; une fusion de marges non supervisée reste inédite).
4. **Plafond** : 13 `confident_wrong` + 32 `hard` toujours-faux bornent le gain reachable
   sans innovation de représentation ; ne pas viser 0,999 sur ces seules sessions.
5. Séquence/Markov : déjà exploré en phase 1 (unigramme/bigramme/trigramme/backoff, tous
   perdants) — pas de relance sans hypothèse neuve (ordre intra-tick à vérifier d'abord).

**Statut.** Analyse diagnostique sur OOF existants + recomputes fold-locaux ; aucune métrique
n'est issue d'un entraînement nouveau. Les 6 variantes snap sont exploratoires (multiples
regards sur le même OOF) et ne seraient adoptées qu'après validation phase C/D — elles sont
ici **rejetées d'emblée** (toutes négatives).
