# BE roadmap — identify the user from an application session

## Recommendation grounded in your training data

Use **whole-event TF-IDF unigrams and bigrams with `LinearSVC(C=1, class_weight='balanced')`** as your reference pipeline. Preserve controller names, screen annotations, dollar-delimited tags and suffixes in this main stream. Exclude the user ID from predictors and remove raw time-marker tokens; engineer timing separately.

This configuration achieved **0.9317 macro-F1, 0.9419 weighted-F1 and 94.51% accuracy** on 820 training-file sessions reserved before comparing models. It was chosen by development cross-validation, not by the reserved-set score. This is the best configuration tested in this analysis, not proof of the global optimum.

This document replaces the earlier browser-oriented roadmap. The earlier browser-classification scores are irrelevant to your user-identification task. The correct validation must keep evaluated users represented in training, rather than holding out entire user classes.

## 1. Data format and findings

`train.csv` has no header. Its rows are variable-length sequences:

```text
user_id, browser, event, event, t5, event, ...
```

The supplied `test.csv` has the same sequence structure but omits the user ID:

```text
browser, event, event, t5, event, ...
```

Consequently, training events begin at index 2 and test events at index 1. Misaligning these indexes is a major implementation risk. Use `csv.reader`, not a rectangular parser that assumes equal column counts or consumes the first row as a header.

| Training-data property | Observed value |
|---|---:|
| Sessions | 3,279 |
| User classes | 247 |
| Sessions/user, min / median / max | 4 / 12 / 75 |
| Sessions/user Q1 / Q3 | 7 / 16 |
| Users with fewer than 8 sessions | 63 |
| Fields/row, min / median / max | 4 / 425 / 14,470 |
| Actual events/session, min / median / max | 0 / 340 / 11,977 |
| Event-count Q1 / Q3 | 153 / 818.5 |
| Distinct exact event tokens | 7,087 |
| Empty CSV fields | 0 |
| Sessions without time markers | 0 |
| Sessions with no actual action events | 1 |
| Duplicate sequences including browser, excluding user ID | 0 |
| Duplicate such sequences after removing time markers | 0 |
| Sessions with user ID as a standalone event token | 0 |

The standalone-token check does not establish that no indirect identifiers exist in annotations. Inspect influential features and the BE rules.

| Browser | Sessions | Distinct users |
|---|---:|---:|
| Firefox | 1,466 | 114 |
| Google Chrome | 1,339 | 93 |
| Microsoft Edge | 451 | 39 |
| Opera | 23 | 1 |

**Each of the 247 users uses exactly one browser across their observed training sessions.** Browser is therefore a potentially valuable predictor or candidate prior. It is not enough to distinguish the many Firefox or Chrome users, but Opera identifies a single observed training user. Do not assume users can never switch browsers outside this sample.

The test file contains 324 unlabeled sessions. It was not used for the user-classifier fitting or scoring below. Its true F1 cannot be measured without labels. In earlier analysis it was mistakenly used for an exploratory browser task; those scores are excluded here.

## 2. Actual user-identification experiments

### Protocol

1. Stratified 75% development / 25% reserved split of `train.csv`, `random_state=42`.
2. Development: 2,459 sessions; reserved evaluation: 820 sessions. All 247 users are represented in both.
3. Compare seven configurations using identical three-fold stratified CV on development, shuffled with seed 123.
4. Fit vocabulary and IDF inside each training fold through a pipeline.
5. Select the configuration with highest mean development macro-F1.
6. Refit on all development sessions and evaluate the selected configuration once on the reserved sessions.

Minimum full-data support is four sessions/user. After reserving approximately one session for the smallest classes, three-fold CV is feasible on development; five-fold CV there would not be appropriate for all classes.

All action models omitted browser as well as the user ID. All removed raw `t<number>` markers. TF-IDF used a whole-event tokenizer, case preserved, `min_df=2`, `sublinear_tf=True`, and default L2 normalization. Exact events retain annotations. Simplified actions use the part before `(`, `$` or `<`, keeping trailing `1`. Context-only extracts values from parentheses, angle brackets and dollar-delimited tags. LinearSVC used the stated C/weights; ComplementNB used `alpha=0.1`.

### Development results

| Representation | Classifier | Mean macro-F1 | Fold SD | Weighted-F1 | Accuracy |
|---|---|---:|---:|---:|---:|
| Simplified action unigrams/bigrams | LinearSVC, C=1, balanced | 0.8353 | 0.0057 | 0.8538 | 0.8609 |
| Exact event unigrams | LinearSVC, C=1, balanced | 0.8364 | 0.0048 | 0.8657 | 0.8723 |
| **Exact event unigrams/bigrams** | **LinearSVC, C=1, balanced** | **0.9081** | **0.0051** | **0.9258** | **0.9305** |
| Exact event unigrams/bigrams | LinearSVC, C=0.1, balanced | 0.8895 | 0.0065 | 0.8985 | 0.9004 |
| Exact event unigrams/bigrams | LinearSVC, C=1, default weights | 0.8839 | 0.0070 | 0.9130 | 0.9223 |
| Annotation/context unigrams | LinearSVC, C=1, balanced | 0.5691 | 0.0183 | 0.6108 | 0.6291 |
| Exact event unigrams/bigrams | ComplementNB, alpha=0.1 | 0.6951 | 0.0162 | 0.7450 | 0.7788 |

A browser-only lookup baseline, choosing the most frequent training-fold user for each browser, obtained development macro-F1 approximately **0.0053**. Thus browser alone is insufficient overall, despite its candidate-filtering potential.

### Reserved-set result for the selected model

| Metric | Value |
|---|---:|
| Macro-F1 | **0.9317** |
| Weighted-F1 | **0.9419** |
| Accuracy | **0.9451** |
| Users with zero F1 on this split | 5 |

The reserved score is higher than the development score; the final fit uses more training sessions per user than each inner-CV fit, and split variability also matters. This does not establish a score on the official test. Some users have only one reserved session, making their per-user estimates particularly uncertain. No temporal identifiers were available, so this measures randomly held-out sessions from known users, not proven future-period robustness.

### What these experiments demonstrate

- **Order matters:** exact bigrams raise development macro-F1 from 0.8364 to 0.9081 relative to exact unigrams.
- **Detail matters for identification:** exact-event bigrams outperform simplified-action bigrams by about 0.073 macro-F1.
- **Class weighting helps this tested configuration:** balanced weights outperform default weights on macro-F1.
- **Context alone is weaker:** annotation-only representations lose important action and ordering information.
- A strong classical model is already available; start optimization from it rather than discarding it for an untested deep model.

## 3. Feature engineering, ranked by likely value

### Priority 1 — retain the winning exact-event representation

One event is one token. For example:

```text
Exécution d'un bouton(infologic.crm...Controller)$EDI$
```

Keep this whole token in the main channel. Add both unigrams and adjacent-event bigrams. Do not use default word tokenization, which splits an event into several French words and fragments controller names. Do not remove all annotations or merge suffix `1` blindly: they may encode habitual workflows or interface choices.

Use TF-IDF with sublinear term frequency so repetitive events in very long sessions have less influence. Compare normalized counts or binary occurrence only as ablations.

### Priority 2 — browser information

Test these alternatives on development folds:

1. One-hot browser appended to sequence features, with a tunable block weight.
2. Browser-conditioned user models if every browser subset has sufficient per-user training support.
3. Candidate scoring that penalizes users inconsistent with the session's browser, based solely on the fold's training mapping.

A soft browser penalty is a more flexible initial experiment than an unconditional hard exclusion. Hard exclusion can be tested for this BE if the dataset construction guarantees fixed browsers. Include fallback behavior for unseen browsers. Validate the procedure without constructing mappings from validation or test labels. These options are **not yet benchmarked**.

### Priority 3 — multiple feature channels

Combine, rather than replace, exact events with:

- Base-action unigrams/bigrams: transferable behavior when exact controllers are rare.
- Controller/module features: coarse `crm`, `orga`, `gmao`, `ventes`, `core` families plus frequent exact controllers.
- Screen tags and code tags as separate channels.
- Optional action trigrams, if they survive validation.

Use `FeatureUnion` or equivalent sparse concatenation, with weights such as 0.25, 0.5, 1 and 2 for auxiliary blocks. More channels are not automatically better; the exact-only baseline is already strong.

### Priority 4 — compact behavioral summaries

For each session derive:

| Family | Useful features |
|---|---|
| Activity size | log1p(event count), unique event/action counts |
| Action composition | counts and proportions of input, click, shortcut, search, dialog, error actions |
| Diversity | action entropy, unique/total ratio |
| Repetition | repeated-action fraction, longest repeated-action run |
| Navigation | screen changes, controller changes, module changes |
| Workflow | dialog-open/close patterns, form-entry patterns, common transitions |
| Position | action proportions in the first/middle/last third |

Use a denominator of at least one for empty sessions. Retain a zero-events indicator. Scale summaries within the training fold before combining them with the sparse SVM. A separate ExtraTrees or CatBoost model can test nonlinear combinations of these features.

### Priority 5 — timing and activity rhythm

Treat `t5`, `t10`, etc. as numeric markers, not thousands of categorical event names. Confirm units and marker semantics with the teacher. The supplied test data includes repeated markers; do not pretend each event has an exact timestamp or derive precise click latency without that information.

Potential summaries: last observed marker, marker count, events per observed bin, mean/median/std/max bin activity, activity coefficient of variation, relative activity over session phases, and gaps between distinct markers. Label event-rate estimates as proxies until duration semantics are confirmed. Missing bins are not necessarily verified idle time.

Do not discard timing permanently merely because it was excluded from the first benchmark; it is a separate feature family to test.

### Priority 6 — nearest-session similarity as a challenger

Fit TF-IDF on fold training sessions and compare cosine nearest-neighbor identification with k=1/3/5 and distance-weighted votes. Different sessions from one user may have several distinct workflow modes, which a nearest-session method can capture. Never include the validation session itself or derived windows from that session in the neighbor database.

This challenger is untested here. Consider combining its scores with the SVM only if out-of-fold error patterns complement each other.

## 4. Classifiers and tuning plan

The first objective is to improve the strong linear baseline with a small targeted search.

| Component | Suggested development search |
|---|---|
| LinearSVC C | 0.25, 0.5, 1, 2, 4 |
| Weights | balanced and default; balanced is the current winner |
| N-grams | (1,2) initially; then compare (1,1) and (1,3) |
| Minimum document frequency | 1, 2, 3 |
| Term frequency | sublinear on/off |
| Auxiliary feature blocks | exact only; exact+base; exact+browser; exact+numeric |

Do not run the full Cartesian product initially. Start with roughly 20–30 carefully chosen candidates, preserve identical folds, and prefer improvements stable across folds.

Classifier challengers, in order:

1. **LinearSVC** on exact/multichannel sparse features.
2. **Logistic regression** on the same features, for an alternative decision boundary and probability outputs.
3. **Cosine nearest neighbors** for recurring workflows.
4. **ExtraTrees/CatBoost** on compact numeric summaries as an auxiliary model.
5. Optional SVD plus RBF SVM if a nonlinear representation experiment is justified.

The logistic-regression results in the previous browser experiment do not establish its user-identification performance; it has not been benchmarked for this target here. Neither have CatBoost, RBF SVM or neural models. A neural sequence model is a lower priority given the high number of users, low per-user support, and strong classical baseline.

## 5. Correct validation for this task

This is closed-set identification: validation users must be known to the classifier. **Do not use leave-user-out GroupKFold as the principal evaluation.** That would test unseen classes the model cannot identify.

Use stratified complete-session splits that keep every evaluated user in training. If you obtain recording dates, compare training on earlier sessions with evaluating later sessions for users that have earlier training examples. If the dataset contains chunks from a shared original recording, group those chunks by original recording while keeping each user represented across independent recordings.

Never split events randomly across training and validation. If you create sliding windows or prefixes, assign all versions of an original session to the same fold. Fit vocabulary, IDF, scaling, SVD, feature selection, browser mappings and calibration on fold training data only.

Optimize the teacher's exact metric. Until specified, use macro-F1 as the main development metric and report weighted-F1 and accuracy too. Macro-F1 gives every user equal importance; weighted-F1 emphasizes users with more sessions. Use an explicit complete label list and report per-user support.

The 820-session holdout has now been evaluated once. For further feature development, keep working on development CV and do not repeatedly optimize against that score. A final claim for a substantially revised model requires fresh independent evaluation, nested CV, or official unseen test feedback. Document any limitation if no fresh data is available.

## 6. Error analysis for the next gains

Examine out-of-fold development errors first:

- Are errors concentrated among users with 4–7 sessions?
- Are very short sessions harder than long sessions?
- Which user pairs share a browser and similar modules?
- Do exact tags identify a user through configuration rather than behavioral style?
- Does the model rely on rare events absent from new sessions?
- Are empty/short sessions effectively ambiguous?

Inspect SVM coefficients by class and the strongest event bigrams. Present examples of behavioral signatures without assuming that a technical tag is inherently leakage. The relevant question is whether the feature is legitimate and available for the BE's intended prediction setting.

If user switches, interface updates or changing work assignments are plausible, add robustness ablations: remove configuration tags, coarse-grain controllers, and simulate predictions from shorter prefixes. Report the performance cost rather than hiding it.

## 7. Executable baseline

This reproduces the selected model and the same outer split. To reproduce the complete model-selection table, also run the seven candidate configurations above on the development three-fold splits. It does not include the untested improvement blocks.

```python
import csv
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

SEP = '\x1f'

def tokenize_events(document):
    return document.split(SEP)

def document(fields):
    events = [s for s in fields
              if not re.fullmatch(r't\d+(?:\.\d+)?', s)]
    return SEP.join(events)

with open('train.csv', encoding='utf-8-sig', newline='') as f:
    rows = list(csv.reader(f))

y = np.array([row[0] for row in rows])
X = np.array([document(row[2:]) for row in rows], dtype=object)
indices = np.arange(len(rows))
dev, reserved = train_test_split(
    indices, test_size=0.25, stratify=y, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        tokenizer=tokenize_events, token_pattern=None,
        lowercase=False, ngram_range=(1, 2),
        min_df=2, sublinear_tf=True)),
    ('model', LinearSVC(C=1, class_weight='balanced')),
])
pipeline.fit(X[dev], y[dev])
predicted = pipeline.predict(X[reserved])
print('macro-F1:', f1_score(y[reserved], predicted, average='macro'))
print(classification_report(y[reserved], predicted, zero_division=0))

# After development and evaluation are complete, refit on permitted full train:
# pipeline.fit(X, y)
# with open('test.csv', encoding='utf-8-sig', newline='') as f:
#     test_rows = list(csv.reader(f))
# X_test = [document(row[1:]) for row in test_rows]  # Different start index!
# user_predictions = pipeline.predict(X_test)
# Export in the exact submission schema requested by the teacher.
```

Browser is intentionally excluded to match the measured baseline. Empty documents become all-zero feature rows and receive an intercept-based prediction; browser or explicit fallback features may improve those cases. A zero-event session cannot provide behavioral evidence that does not exist.

## 8. Implementation schedule and report

| Stage | Deliverable |
|---|---|
| 1. Confirm requirements | Exact F1 variant, submission schema, allowed metadata/features, timing semantics |
| 2. Reproduce baseline | Same split, CV table, selected configuration and reserved-set result |
| 3. Test highest-value additions | Browser block, multichannel events, small C/min_df search |
| 4. Add behavioral/timing blocks | Incremental ablations on unchanged development folds |
| 5. Test challengers | Logistic regression, nearest-session similarity, numeric tree model |
| 6. Analyze errors | Rare-user/short-session breakdown and influential features |
| 7. Select and validate | Independent assessment appropriate to the available data |
| 8. Refit/export | Full-training model and test predictions in required row order/schema |

Report structure: problem and closed-set assumptions; data/labels; EDA and browser association; validation; feature engineering; baselines; model comparison and ablations; per-user errors; limitations; reproducibility and final pipeline.

Avoid claims of guaranteed maximum F1. The strongest demonstrated result is **0.932 macro-F1 on one reserved training-data split**, and the next features are experiments, not confirmed improvements.

## Reference documentation

- [TF-IDF custom tokenization and n-grams](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [LinearSVC regularization and class weights](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
- [F1 averaging definitions](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
- [Leakage and preprocessing pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
