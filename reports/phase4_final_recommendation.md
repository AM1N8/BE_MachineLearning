# Phase 4 — Recommandation finale (E/F : candidat, soumission, décision)

**Objectif.** Choisir le modèle final selon les règles de mission (champion vérifié vs
challenger expérimental, critère conservateur inchangé), exporter le CSV de soumission
du challenger **sans jamais soumettre automatiquement ni écraser les fichiers protégés**,
et énoncer la recommandation honnête.

**Sources (artefacts signés).** Module `src/copilote/phase4_challenger.py` ; export dans
`reports/results/phase4/c8a3d88dbf5c/` (manifeste signé : sources, train, test, verrou,
hash des soumissions protégées avant/après, spécification du modèle, dossiers validés) ;
CSV `submissions/phase4_challenger.csv` (SHA-256
`c29fd2fe2f04003914467cf29b6ad287f6afc535f70bfe80ce1a69c7fd514003`). Les 24 folds du
challenger (recherche + confirm + seed 44) ont été **reproduits à l'identique** avant
l'export (`reproduce`) ; le rechargement joblib du modèle redonne exactement les mêmes
prédictions et marges. Aucune transduction : ajustement sur les 3 279 sessions
étiquetées uniquement, le test n'est jamais consulté pour l'apprentissage.

## 1. Décision de promotion (critère pré-enregistré, inchangé)

| Protocole | lower95 de `sig_burst_w05` | Les 2 répétitions positives ? | Verdict |
|---|---|---|---|
| Recherche (seed 42) | −0,002088 | oui (+0,002118 / +0,006933) | échoue |
| Confirm (seed 43) | −0,002470 | oui (+0,000796 / +0,003876) | échoue |
| Seed 44 | −0,003202 | oui (+0,003693 / +0,003096) | échoue |

**Le critère n'est pas satisfait : aucune promotion.** Le champion vérifié reste
`submissions/best_verified.csv` (baseline `browser_mask_c3`, Kaggle public 0,980), et le
challenger est exporté en **statut EXPÉRIMENTAL_NON_PROMU**, exactement comme
`context_signatures_experimental.csv` en phase 3 (même discipline, mêmes critères).

## 2. Choix du challenger : pourquoi `sig_burst_w05` et pas `w06` ou signatures

| Indice (artefacts) | sig_burst_w05 | sig_burst_w06 | signatures |
|---|---|---|---|
| Δ moyen poolé (24 folds) | **+0,003419** | +0,003100 | +0,001311 |
| lower95 poolé (t(23)) | **+0,001842** | +0,001293 | +0,000518 |
| Répétitions positives (3 protocoles × 2) | **6/6** | 5/6 (confirm rep. 0 +0,000073) | 4/6 |
| Domine signatures sur les 3 protocoles | oui | oui (macro-F1) | — |

`sig_burst_w05` = baseline (TF-IDF 1-2 g, LinearSVC C=3 équilibré, masque navigateur
fold-local) + bloc signatures (contextes remplis JSON, binaire, chi² 20 %, **poids 0,5**)
+ bloc salves (`b{1..5}|action` par tick 5 s, binaire, chi² 20 %, poids 0,3). Le poids
0,5 (vs 0,3 en phase 2) est le meilleur compromis répliqué ; le poids 0,6, meilleur en
recherche uniquement, est explicitement écarté (non-réplication, see DECISIONS D14).

## 3. Différences sur le test (324 lignes, `c8a3d88dbf5c/`)

| Comparaison | Lignes changées | Détail |
|---|---|---|
| vs `best_verified.csv` (0,980) | **3 / 324** | 125 `zro→bgn`, 237 `czi→pgm`, 290 `usy→vsb` |
| vs `context_signatures_experimental.csv` (0,983) | **2 / 324** | 125 `zro→bgn`, 290 `usy→vsb` |

Ligne 237 : le challenger rejoint le changement déjà fait par signatures (czi→pgm).
Ligne 290 : le challenger promeut le 2ᵉ du baseline ET de signatures (usy→vsb, marge
top-2 challenger 0,004). Ligne 125 : changement propre au challenger (zro→bgn, le
challenger n'était pas 2ᵉ du baseline — flip « tiers », marge 0,019). Fichier complet :
`c8a3d88dbf5c/test_margins.csv` (top-2 gaps des trois modèles, drapeaux « était-2ᵉ »,
classes supportées par le masque navigateur : médiane 93, max 114, et 114 sur chacune
des trois lignes changées).

## 4. Recommandation explicite

1. **Soumission Kaggle recommandée : `submissions/phase4_challenger.csv`** — à soumettre
   manuellement (jamais automatiquement), si le quota le permet, car l'évidence CV
   poolée (+0,0034 ± 0,0016) est positive et cohérente sur trois protocoles et que le
   challenger contient strictement le champion baseline + deux blocs additifs ; à 3 lignes
   près du meilleur état vérifié, le risque réel est borné : au pire 0/3 flips bons →
   0,980 − 3/324 ≈ 0,9707 si les trois étaient mauvaises, au mieux ≈ 0,989 ; le point
   médian attendu (≈ 2/3 bons flips sur les changements à marge faible) est proche de
   l'état actuel. **Si le quota est précieux, conserver `best_verified.csv`/le statu quo
   0,983 est défendable** : le challenger ne différencie que 2 lignes de la soumission
   expérimentale actuelle (0,983).
2. **Ne pas poursuivre** le réglage de poids (non-répliqué), le dwell/scalaires creux
   (nocifs), la fusion de marges (≈ poids réduit) ni la correction résiduelle (falsifiée).
3. **Ce qui reste ouvert après la phase 4** : 13 erreurs « confiantes » et 37 sessions
   toujours-fausses hors top-2 plafonnent tout gain à ~+0,005 CV ; les atteindre exigerait
   des représentations nouvelles (hors budget de la mission). Le rythme de saisie (salves)
   est le seul canal nouveau confirmé ; un balayage plus fin (poids des salves ≠ 0,3,
   bords de salves) n'a **pas** été testé pour contenir la multiplicité.

## 5. Garde-fous vérifiés à l'export

- `best_verified.csv` et `context_signatures_experimental.csv` inchangés (hashs avant =
  après, consignés dans `challenger.json`).
- Ordre des lignes = ordre de l'exemple (`RowId` 1…324, contrôlé contre
  `data/sample_submission.csv`, unicité, aucune valeur manquante, prédictions ⊆ classes
  du train).
- Refus d'écraser un export différent préexistant ; rechargement joblib exact ;
  `transductive = false` ; `automatic_submission = false` ;
  `promotion_criteria_changed = false`.
