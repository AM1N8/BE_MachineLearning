# Décisions

## D15 — Export exploratoire explicitement demandé
Contexte : l'utilisateur souhaite tester le meilleur candidat Phase 2 malgré son
échec au seuil statistique. Choix : exporter exactement `behavioral_model("signatures")`
dans `context_signatures_experimental.csv`, après reproduction des 24 plis et
réentraînement complet. Aucun changement des critères ni promotion de modèle.
La baseline et ses artefacts restent intacts ; les différences et marges sont
consignées dans `reports/results/context_signatures_experimental/`.
Justification : distinguer un essai leaderboard autorisé d'une amélioration validée,
sans inventer une nouvelle configuration ni modifier manuellement les prédictions.

## D14 — Recherche comportementale et correction globale top-2
Contexte : erreurs surtout ambiguës, échec antérieur des classifieurs locaux par paire.
Options : recherche large supplémentaire / hypothèses ciblées avec scores globaux OOF.
Choix : cinq ablations contextuelles séparées et reranker logistique entraîné uniquement
sur scores/profils hors-échantillon internes. Seuils ajustés dans chaque train externe,
avec option de ne rien changer. Confirmation appariée sur graines 43 et 44.
Les groupes de proximité sans labels sont tous triviaux : pas de validation groupée
présentée artificiellement comme une preuve d'indépendance. Chronologie inconnue.
Résultat : signatures contextuelles prometteuses mais aucune borne corrigée positive ;
reranker sans amélioration fiable. Conserver la référence, sans produire de challenger.
Les chiffres sont dans `reports/results/phase2/a1d191a9b558/*.csv` et `decision.json`.
`make phase2` reproduit la campagne et les rapports ; cette cible est incluse dans
`make all`. Les sources prédictives hachées restent figées après exécution.

## D12 — Clôture de la recherche imbriquée et choix du CSV
Contexte : campagne `extreme` existante, sept folds externes terminés, recherche
séquentielle reproductible et règle de promotion déjà codée avant sa clôture.
Options : remplacer la référence au plus petit gain moyen / appliquer la règle gelée.
Choix : terminer le huitième fold et la sélection interne finale sans modifier les
sources ni la configuration de la campagne. Conserver `browser_mask_c3` : gain
d'ensemble +0,000069 macro-F1, borne corrigée −0,002336, une répétition négative.
Pourquoi : le gain est instable et ne satisfait pas la règle préexistante ; aucune
optimisation sur le leaderboard ni modification manuelle des prédictions.
`make submission` reconstruit le choix gelé sur tout le train et exige le même
SHA-256 que le CSV enregistré. `make all` inclut cette reconstruction. Les chiffres
proviennent de `reports/results/extreme/1ba1ee389f73/nested_validation.json`.
La description locale indique « F1 score moyen » ; macro-F1 reste l'interprétation,
le calcul exact du serveur Kaggle n'étant pas confirmé. Le score Kaggle reste inconnu.

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
Contexte : le format réel n'est pas encadré de guillemets ; l'unité des marqueurs temporels n'est pas confirmée (voir D03).
Choix : lecture tolérante aux deux formats (guillemets ou non), validation stricte du nombre de colonnes, identifiants de test à base 1 conformes à l'exemple fourni.
La métrique Kaggle reste inconnue : tout est piloté par `cv.selection_metric` dans la configuration, par défaut `macro_f1`, avec l'exactitude publiée systématiquement.

## D06 — Sélection, ensembles et calibration
Contexte : 247 classes, certaines à 4 sessions, 324 lignes de test.
Choix : holdout stratifié verrouillé avant toute observation; sélection sur CV de développement seulement; règle « plus simple à moins d'un écart-type »; un vote ou un empilement n'est retenu que s'il gagne de plus d'un écart-type sur le développement et que sa signature est pré-déclarée.
La calibration sigmoïde est ajustée par validation croisée interne et n'est appliquée que pour l'analyse de confiance, jamais pour ré-optimiser la classe prédite.

## D07 — Explicabilité honnête
Contexte : la forêt et le SVM n'ont pas les mêmes justifications possibles.
Choix : le SVM se décompose exactement en `coefficient × TF-IDF` + intercept; la forêt fournit une décomposition exacte des probabilités le long des chemins, jamais présentée comme des valeurs de Shapley. La permutation est groupée par famille pour préserver les dépendances internes.

## D08 — Recherche compétitive et isolation des résultats
Contexte : demande d'améliorer le score Kaggle ; métrique officielle inaccessible et réserve historique déjà évaluée.
Options : modifier la chaîne du rapport / ajouter une recherche indépendante et traçable.
Choix : `configs/competitive.yaml` décrit 52 candidats, comparés sur les mêmes trois folds du développement historique (80 % du train, graine 42). Résultats dans `reports/results/competitive/`, nouvelle soumission `submissions/competitive.csv`.
Pourquoi : conserver les expériences et livrables existants, publier les échecs et éviter de choisir sur la réserve ou le classement Kaggle.
Le choix est gelé avant les benchmarks descriptifs de CV répétée et de réserve historique ; ceux-ci ne constituent pas une nouvelle validation indépendante. Aucune transduction ni calibration des marges n'est revendiquée.

## D09 — Navigateur et complexité
Contexte : aucun utilisateur ne change de navigateur dans les observations d'entraînement ; les classes sont les utilisateurs, pas des groupes à exclure des folds.
Options : navigateur comme caractéristique, masque de support classe/navigateur, experts par navigateur.
Choix : comparer les trois approches. Le masque apprend son support uniquement sur les lignes d'entraînement du fold ; un navigateur inconnu ne restreint aucune classe.
Pourquoi : hypothèse vérifiable par CV, gain limité mesuré sans consulter les étiquettes du test. Le risque en cas de changement de navigateur vers un navigateur connu est documenté.
Complexité 1 pour un classifieur à représentation simple, 2 pour les experts et représentations multi-vues supplémentaires ; règle D06 conservée, départage à complexité égale par la métrique de développement. Les copies augmentées sont créées uniquement à l'intérieur du fold d'entraînement.

## D10 — Reproductibilité des soumissions
Choix : empreinte du train, des sources prédictives, du protocole et du verrou de dépendances ; refus des sélections obsolètes. Recalcul des métriques dans chaque fold, sauvegarde des prédictions hors-fold, de la configuration, des avertissements et du hachage du CSV final.
Les extraits d'événements servant au diagnostic restent dans `artifacts/` (ignoré par git et exclu des archives). Le rapport français est généré depuis les JSON/CSV de résultats ; score Kaggle explicitement inconnu.

## D11 — Seconde recherche : représentations et similarité
Contexte : demande de tester des méthodes réellement différentes après la première recherche.
Options : continuer à régler C / changer la représentation et le mode de décision.
Choix : 33 candidats (31 nouvelles variantes, deux références) dans `configs/advanced.yaml` : voisins par classe, centroïdes utilisateur, racine carrée du TF-IDF (inspirée de Hellinger), noyaux polynomiaux sur similarité cosinus, ridge à noyau, hybrides à poids fixés, BM25, IDF par utilisateur et entropie de classe.
Pourquoi : les profils et noyaux peuvent reconnaître plusieurs comportements d'un même utilisateur ; les pondérations peuvent réduire les événements communs. Toutes les statistiques supervisées et non supervisées sont apprises dans les folds, avec test de cohérence fit/transform et d'absence d'adaptation aux requêtes.
Les matrices de tokens restent creuses ; seule la matrice sessions × sessions est dense. Les hybrides sont de complexité 2, les modèles uniques de complexité 1 ; la règle D06 est conservée. Recherche adaptative sur le développement historique ; les benchmarks ultérieurs ne sont jamais utilisés pour régler les paramètres.
Résultat : la meilleure moyenne de développement vient de `hellinger_svm_c10`, mais son avantage ne se confirme pas en CV répétée. `submissions/advanced.csv` matérialise ce choix de développement sans prétendre améliorer le score Kaggle ; `submissions/competitive.csv` et son historique sont conservés. Les deux fichiers ne doivent pas servir à optimiser par essais successifs sur le leaderboard.
# D13 — Bounded post-submission research

Context: user-reported public score 0.980, remaining low-margin and short-session errors.
Choice: test fixed local-pair, partial-balancing, rare-binary, browser-offset and intercept
hypotheses on identical folds; confirm boundary/rare-event hypotheses with seed 43.
Keep the existing corrected uncertainty/repeat-consistency gate and preserve best_verified.csv.
Rationale: broad families already tested; marginal adaptive gains are not evidence of
generalization. No candidate passed; no verified challenger produced. Official averaging
remains unverified. Full evidence: reports/leaderboard_improvement.md and result CSVs.

# D14 — Phase 4 candidate families, weight sweep and challenger export

Context: Phase A forensics exposed an untapped timing channel (5 s ticks, event
bursts, suffix flag, dwell shares) and Phase B showed suffix_fraction and
dwell_top_fraction are the strongest new error discriminators; the signatures
block weight (0.3) had never been swept.

Options: (a) dense numeric fusion of timing scalars, (b) sparse binary tempo
token blocks mirroring the signatures architecture (fold-local TF-IDF + chi2
top 20%), (c) margin fusion of baseline+signatures, (d) signatures block weight
ladder 0.2-1.0, (e) hybrid signatures+burst blocks.

Choice: test all motivated families once on the frozen historical protocol
(seed 42), then confirm every research-positive candidate on two fresh
protocols (seeds 43 and 44) with the pre-registered conservative criterion
(lower95 > 0 per protocol, both repeats positive). Export the best replicating
candidate (sig_burst_w05 = signatures block at weight 0.5 + burst block at
0.3) as an explicitly experimental challenger; no promotion because no
candidate passes the criterion on any protocol.

Rationale: fold-local vocabulary/IDF/selection keeps every block leakage-free;
fixed a priori bucket edges avoid fitted quantiles; the weight sweep is
declared as multiple comparisons and adjudicated only on the fresh protocols
(w06's research gain +0.0040 did not replicate: +0.0007 on seed 43; w05's did
on all three). Pooled 24-fold evidence is reported as post-hoc secondary
evidence only. The burst channel (events per 5 s tick) is the only new
mechanism that replicated; dwell/scalar blocks were harmful and margin fusion
is provably close to a reduced-weight signatures model.

Full evidence: reports/phase4_experiment_results.md,
reports/phase4_final_recommendation.md and signed artifacts under
reports/results/phase4/ (research 85b76c4630ba, confirm 8492d196609f,
seed44 951bd7da5015, challenger export c8a3d88dbf5c).

## D15 — Audit generalization before further model optimization

Context: the user reported public scores of 0.980 for the baseline, 0.983 for
contextual signatures and 0.970 for the phase 4 burst hybrid. The hybrid had
failed the original promotion criterion; its pooled evidence was post-hoc and
the confirmation seeds were subsequently used to select its final weight.

Options: (a) continue parameter search, (b) undo particular test predictions
using leaderboard feedback, (c) audit the frozen models, scorer assumptions,
split dependence, feature transfer and small-holdout uncertainty first.

Choice: execute (c), with three frozen models, twelve 324-session stratified
holdouts and four similarity-group stress holdouts; no new model, export or
automatic submission. Keep the existing signatures submission as the best
user-reported public result, retaining its experimental status. The original
verified baseline remains the project's verified champion.

Rationale: the primary corrected hybrid-minus-signatures interval includes zero
and its mean gain is effectively null. Repeated seeds add no independent data.
The available evaluation text says "F1 score moyen" without specifying exact
averaging, absent-class policy or public subset size. Scalar and vocabulary
transfer checks do not show a clear test-only breakdown; the atypical reverse
source AUC remains exploratory evidence requiring collection provenance, not
an explanation established from hidden labels. No test flip is adjudicated.

Next gate: confirm the scoring and sampling contract, preregister one paired
comparison and coverage sensitivities, then obtain genuinely new confirmation
observations or usable collection metadata before claiming a promotion.
Support-stable feature influence is a possible future hypothesis, not a model
authorized by the present audit.

Evidence: reports/phase5_validation_audit.md and signed JSON/CSV under
reports/results/phase5/1ed0b3ef63ea/. Refits were first run under 4922593685dd
and reused only after exact dependency, input and split checks. Earlier models,
submissions and reports are preserved.
