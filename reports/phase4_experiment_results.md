# Phase 4 — Résultats d'expériences (directions C, validation D)

**Objectif.** Tester les familles de candidats issues de la forensic (Phase A) et de
l'analyse d'erreurs (Phase B) sur le protocole historique gelé (4×2
RepeatedStratifiedKFold, seed 42, 6 558 prédictions OOF), puis confirmer les candidats
positifs sur deux protocoles à splits neufs (seed 43 « confirm », seed 44). Aucune
sélection, IDF, masque navigateur ou vocabulaire n'est appris hors du fold
d'entraînement ; les bords de seuil (salves, part de dwell, scalaires) sont fixés
a priori, jamais ajustés sur des quantiles observés.

**Sources (artefacts signés).** Module `src/copilote/phase4_candidates.py` (mêmes sources
et protocoles que les rapports précédents ; tests `tests/test_phase4.py`, 9 tests).
Résultats : recherche `reports/results/phase4/85b76c4630ba/comparison.csv`,
confirmation `reports/results/phase4/8492d196609f/confirmation.csv`,
seed 44 `reports/results/phase4/951bd7da5015/seed44_comparison.csv`. Chaque dossier
porte un manifeste signé (hash des sources, du train, du verrou, des soumissions
protégées, des splits). L'export challenger (Phase E/F) est détaillé dans
`reports/phase4_final_recommendation.md`.

## 1. Familles testées (motivées par A/B, aucune inventée a posteriori)

| Famille | Description | Motivation |
|---|---|---|
| `tempo_suffix` | bloc binaire `sfx{0/1}|action` + chi² 20 %, poids 0,3 | suffixe « 1 » : discriminateur 4,5× sur les erreurs (B) |
| `tempo_burst` | bloc `b{1..5}|action` (salve = événements par tick 5 s) | 99,9 % des ticks actifs, 2,83 év./tick médian (A) |
| `tempo_dwell` | bloc `dwell|écran|s1..s5` + `dwelltop` | dwell_top_fraction H=1 313 (A), 0,571 vs 0,436 (B) |
| `tempo_scalars` | tokens `eps/sfxfrac/evtick` à bords fixes | scalaires KW-significatifs, jamais testés en creux |
| `tempo_all` | les quatre à la fois | test de complémentarité globale |
| `fusion_raw` / `fusion_std` | somme pondérée non supervisée des marges baseline+signatures (α=0,3 brut ; 0,5 standardisé) | reranker supervisé falsifié en phase 2, fusion jamais testée |
| `signatures_wXX` | bloc signatures (phase 2) au poids 0,2/0,3/0,4/0,5/0,6/0,7/1,0 | poids 0,3 jamais balayé en phase 2 ; la fusion ≈ signatures à poids réduit |
| `sig_burst*` | bloc signatures + bloc salves, séparés ou fusionnés, poids variés | corrections disjointes (union 20, antagonisme 0 — analyse d'overlap) |

Architecture commune : baseline `browser_mask_c3` (TF-IDF 1-2 grams, LinearSVC C=3
équilibré, masque navigateur appris sur le fold) + FeatureUnion(bloc original ×1,
bloc comportemental × poids), bloc = TfidfVectorizer binaire + SelectPercentile(chi²).
La fusion brute est mathématiquement proche d'un modèle signatures à poids de bloc
réduit (≈ 0,09) : ce lien est vérifié par les résultats (fusion_raw ≈ baseline).

## 2. Recherche (historique, seed 42) — `85b76c4630ba/comparison.csv`

Macro-F1 moyen sur 8 folds (Δ vs baseline 0,961251 ; bornes corrigées
t(0,975,7)·√(1/8+1/3) ; « corr./introd. » = erreurs corrigées/introduites sur 6 558 OOF) :

| Candidat | Macro-F1 | Δ | folds gagnés | lower95 | rep. 0 / rep. 1 | corr./introd. |
|---|---|---|---|---|---|---|
| signatures (réf.) | 0,962920 | +0,001669 | 7/8 | −0,001375 | +0,001036 / +0,002302 | 16/6 |
| tempo_suffix | 0,961633 | +0,000382 | 2/8 | −0,002391 | +0,000413 / +0,000351 | 5/5 |
| **tempo_burst** | 0,962484 | +0,001233 | 4/8 | −0,002200 | +0,000737 / +0,001729 | 8/4 |
| tempo_dwell | 0,959358 | **−0,001893** | 0/8 | −0,004191 | −0,001616 / −0,002170 | 2/7 |
| tempo_scalars | 0,960619 | −0,000632 | 2/8 | −0,002883 | −0,000253 / −0,001010 | 1/4 |
| tempo_all | 0,961817 | +0,000566 | 4/8 | −0,003702 | +0,000435 / +0,000697 | 6/4 |
| fusion_raw | 0,961257 | +0,000005 | 2/8 | −0,001537 | −0,000333 / +0,000344 | 4/1 |
| fusion_std | 0,961570 | +0,000319 | 4/8 | −0,002147 | −0,000490 / +0,001129 | 8/2 |
| sig_burst (0,3/0,3) | 0,963679 | +0,002428 | 6/8 | −0,001991 | +0,003133 / +0,001723 | 18/6 |
| sig_burst_merge | 0,962947 | +0,001696 | 7/8 | −0,001312 | +0,001036 / +0,002355 | 16/6 |
| **sig_burst_w05 (0,5/0,3)** | 0,965777 | **+0,004525** | 7/8 | −0,002088 | +0,002118 / +0,006933 | 34/14 |
| **sig_burst_w06 (0,6/0,3)** | 0,965852 | **+0,004601** | 7/8 | −0,003368 | +0,004341 / +0,004861 | 45/19 |
| signatures_w02 | 0,961570 | +0,000319 | 4/8 | −0,002147 | −0,000490 / +0,001129 | 8/2 |
| signatures_w04 | 0,963352 | +0,002100 | 5/8 | −0,003168 | +0,000688 / +0,003513 | 25/10 |
| signatures_w05 | 0,964033 | +0,002781 | 7/8 | −0,002134 | +0,000906 / +0,004657 | 30/14 |
| **signatures_w06** | 0,965247 | +0,003996 | 7/8 | −0,004283 | +0,002883 / +0,005109 | 47/21 |
| signatures_w07 | 0,963614 | +0,002363 | 5/8 | −0,007448 | +0,001833 / +0,002893 | 51/30 |
| signatures_w10 | 0,956810 | **−0,004441** | 2/8 | −0,017155 | −0,005532 / −0,003350 | 54/57 |

Lecture :
- **Personne ne passe le critère pré-enregistré** (lower95 > 0 ET les deux répétitions
  positives) : le meilleur lower95 reste négatif.
- Le balayage de poids est unimodal (pic 0,6, effondrement à 1,0 où le bloc noie les
  features originales). Attention honnête : 7 poids ont été testés sur ce protocole de
  recherche (multiplicité) ; la confirmation seed 43/44 tranche.
- `tempo_dwell` et `tempo_scalars` sont **nocifs** : l'association KW forte (forensic)
  ne se transfère pas en pouvoir discriminant creux — le dwell médian est quasi constant,
  seule la distribution (déjà capturée par les signatures) différencie.
- `fusion_raw` ≈ baseline, conforme à la prédiction « fusion ≈ signatures à poids réduit ».
- Précédent « style » (phase 1, vue action+'1' en remplacement, perdante) : le suffixe
  seul (`tempo_suffix`) est ici faiblement positif mais non confirmé — cohérent avec ce
  précédent ; **la salve (`burst`) est la composante réellement nouvelle**.
- Overlap recherche : signatures corrige 16 / burst corrige 8, intersection 4, union 20,
  **antagonisme nul** (aucune correction de l'un brisée par l'autre) → hybrides motivés.

## 3. Confirmation (seed 43, splits neufs) — `8492d196609f/confirmation.csv`

Baseline seed 43 : 0,959599. Candidats positifs en recherche uniquement (automatique) :

| Candidat | Δ | folds gagnés | lower95 | rep. 0 / rep. 1 | corr./introd. |
|---|---|---|---|---|---|
| signatures (réf.) | +0,000941 | 4/8 | −0,002276 | +0,000355 / +0,001528 | 11/7 |
| tempo_burst | +0,000921 | 6/8 | −0,000714 | +0,000620 / +0,001222 | 8/3 |
| sig_burst | +0,001512 | **8/8** | −0,000590 | +0,000805 / +0,002220 | 16/8 |
| **sig_burst_w05** | **+0,002336** | 6/8 | −0,002470 | +0,000796 / +0,003876 | 27/17 |
| sig_burst_w06 | +0,002131 | 7/8 | −0,003182 | +0,000073 / +0,004189 | 39/30 |
| signatures_w05 | +0,000684 | 6/8 | −0,003532 | −0,000797 / +0,002166 | 27/22 |
| signatures_w06 | +0,000653 | 4/8 | −0,004359 | −0,001199 / +0,002505 | 36/28 |
| signatures_w07 | +0,001825 | 5/8 | −0,002760 | +0,000069 / +0,003580 | 46/35 |

Lecture honnête : **le gain « poids » (w06 recherche +0,003996) ne se réplique pas**
(+0,000653 en confirmation) — le poids 0,6 était ajusté aux splits de recherche. En
revanche le **bloc salves se réplique** : sig_burst +0,001512 avec 8 folds gagnés sur 8
et lower95 −0,000590 (le plus proche du passage jamais obtenu). Personne ne passe.

## 4. Seed 44 (splits neufs) — `951bd7da5015/seed44_comparison.csv`

Baseline seed 44 : 0,961707 :

| Candidat | Δ | folds gagnés | lower95 | rep. 0 / rep. 1 | corr./introd. |
|---|---|---|---|---|---|
| signatures (réf.) | +0,001322 | 6/8 | −0,001729 | +0,001704 / +0,000941 | 13/3 |
| tempo_burst | +0,000024 | 5/8 | −0,004931 | +0,001250 / −0,001203 | 7/5 |
| sig_burst | +0,001550 | 6/8 | −0,004484 | +0,003470 / −0,000370 | 16/3 |
| **sig_burst_w05** | **+0,003395** | 6/8 | −0,003202 | **+0,003693 / +0,003096** | 33/13 |
| sig_burst_w06 | +0,002566 | 6/8 | −0,004656 | +0,001827 / +0,003305 | 38/20 |
| signatures_w05 | +0,002129 | 5/8 | −0,006278 | +0,002697 / +0,001561 | 32/14 |

`sig_burst_w05` est le **seul candidat avec les deux répétitions positives sur les trois
protocoles** (recherche, confirm, seed 44) ; il domine la référence signatures sur les
trois (0,965777 > 0,962920 ; 0,961935 > 0,960541 ; 0,965102 > 0,963030).

## 5. Preuve secondaire poolée (24 folds, descriptive) — `85b76c4630ba/pooled_evidence.json`

Analyse post-hoc, **hors critère pré-enregistré** (les folds ne sont pas indépendants
entre protocoles — mêmes sessions) ; Δ moyen et borne basse t(0,975,23) :

| Candidat | Δ moyen | lower95 poolé | folds gagnés |
|---|---|---|---|
| signatures (réf.) | +0,001311 | +0,000518 | 17/24 |
| tempo_burst | +0,000726 | −0,000209 | 15/24 |
| sig_burst | +0,001830 | +0,000685 | 20/24 |
| **sig_burst_w05** | **+0,003419** | **+0,001842** | 19/24 |
| sig_burst_w06 | +0,003100 | +0,001293 | 20/24 |

## 6. Triage final des directions (C)

| Direction (mission) | Verdict | Preuve |
|---|---|---|
| 1. Séquences / Markov | **fermée** | phase 1 : tous perdants ; rien de nouveau en A/B pour rouvrir |
| 2. Empreintes informationnelles | **testée** | tempo (salves/dwell/scalaires) : burst seul réplique ; dwell/scalars nocifs (recherche) |
| 3. Reconstruction d'état | **déjà saturée** | signatures = remplissage de contexte ; ajouts dwell contre-productifs |
| 4. Correction résiduelle | **falsifiée** | Phase B : snap net-négatif dans les 6 variantes |
| 5. Ensembles complémentaires | **testée** | fusion ≈ poids réduit ; hybride signatures+salves = le seul gain qui réplique |

## 7. Conclusion honnête

1. **Aucun candidat ne satisfait le critère de promotion conservateur** (lower95 > 0 par
   protocole) — le champion vérifié reste `best_verified.csv` (baseline, Kaggle 0,980).
2. Le candidat le plus solide est `sig_burst_w05` (signatures ×0,5 + salves ×0,3) :
   Δ moyens +0,004525/+0,002336/+0,003395 sur les trois protocoles, deux répétitions
   positives partout, lower95 poolé +0,001842, domination de signatures partout.
3. Le mécanisme qui réplique est la **salve d'événements par tick 5 s** (rythme de saisie,
   jamais exploité avant) ; le réglage fin du poids est du bruit de protocole (w06 non
   répliqué) et n'est pas revendiqué.
4. Limites : gains CV ≈ +0,003 attendus sur 247 classes rares — l'effet sur 324 lignes de
   test Kaggle est à forte variance (voir rapport final) ; multiplicité du balayage de
   poids déclarée et neutralisée par les deux protocoles neufs.
