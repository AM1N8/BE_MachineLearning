# Method: the submitted three-model soft vote

## 1. Problem and input handling

Each record is a software-usage session; the target is its user's trigram. `data.py` detects a browser in field zero to identify an unlabeled test record. Labels and row IDs are never included in features.

Time markers are preserved by the parser, including repeated markers. These three final models use **action sequences without explicit time tokens**. Browser identity is used for training-only compatibility filtering.

## 2. Two action-aligned streams

The **full** stream replaces whitespace inside an action with underscores, keeping its controller, configuration, chain and modification suffix. The action remains a single token.

The **context** stream decomposes an action into name, controller, configuration, chain and modification flag. It carries the last observed screen/configuration/chain forward. An explicit new screen resets configuration and chain to the values supplied with that screen, including `NONE` when absent.

One context token is formed per action:

```text
s=SCREEN|a=ACTION~c=CONF|a=ACTION~h=CHAIN|a=ACTION|m=FLAG
```

This preserves action alignment so the same crop boundaries apply to both streams.

## 3. The three members

| Member | Streams | Crop length | IDF | Maximum vocabulary per stream |
|---|---|---:|---|---:|
| `crop128_noidf` | Full | 128 | No | 200,000 |
| `fusion_context` | Full + context | 96 | Yes | 200,000 |
| `reference128` | Full | 128 | Yes | 180,000 |

All vectorizers use case-sensitive, whitespace-delimited action tokens, word n-grams `(1, 3)`, `min_df=1`, binary term occurrence, float32 sparse matrices and L2 normalization. Vocabulary and IDF are learned from **original fitting sessions only**, before crops are transformed.

For the context member, the final feature vector is:

$$x = \operatorname{normalize}\left([x_{\mathrm{full}},\;0.7x_{\mathrm{context}}]\right).$$

For empty action sequences, transformation uses the placeholder `EMPTY`.

## 4. Training crops and sample weights

A full session is always included. When its action count satisfies $n > 1.5L$, three length-$L$ windows are added at integer-truncated, evenly spaced start positions from zero to $n-L$. Duplicate windows are removed.

For 340 actions and $L=128$:

```text
full:   [0, 340)
start:  [0, 128)
middle: [106, 234)
end:    [212, 340)
```

All examples derived from a session remain in that session's training fold. If a class has $n_c$ of the $N$ original training sessions and there are $K$ classes, its session weight is:

$$b_c = \frac{N}{K n_c}.$$

The full example has relative mass 1; each crop has mass 0.3. These masses are normalized to sum to $b_c$. Thus augmentation does not multiply a user's total training weight just because their sessions are long. A session with three crops allocates $10/19$ of its mass to the full trace and $3/19$ to each crop.

## 5. Classifier and browser compatibility

Each member is a one-vs-rest `LinearSVC` with `C=1`, L2 penalty, squared hinge loss, `dual=True`, `max_iter=5000`, and explicit sample weights. Final training uses seed 137.

Class scores become probability-shaped values:

$$p_c = \frac{\exp(5s_c)}{\sum_j\exp(5s_j)}.$$

These are **not calibrated confidence estimates**. The factor 5 controls the sharpness used by the blend.

The fitting partition supplies a browser-to-class support map. At prediction time, incompatible classes are zeroed and remaining mass is normalized. An unseen browser or zero remaining mass falls back to the unfiltered scores. Validation labels are never used for this map.

## 6. Frozen submitted weights

The final soft vote uses:

| Member | Weight |
|---|---:|
| `crop128_noidf` | 0.422154785394683 |
| `fusion_context` | 0.46445268572353543 |
| `reference128` | 0.11339252888178171 |

$$p_{\mathrm{B}} = \sum_m w_m p_m.$$

Prediction is the trigram at the largest blended score. A soft vote can disagree with a majority of member argmax labels because it uses the full score distributions.

The original weights were fitted on pooled repeated OOF predictions with seeded Optuna search: 80 trials, a 40-second soft timeout, nonnegative weights normalized to sum to one, with single-model and equal-weight candidates included. Timing can change how many trials finish; the checked-in weights preserve the actual submitted solution independently of that timing.

The `blend` command can repeat this search but writes a separate result. It does not alter `configs/solution.json` or the submitted prediction recipe.

## 7. Submission integrity

Class order is shared across all models. Prediction reproduces the sample's columns and IDs, checks row count, label membership and missing values, then rereads the written CSV. The optional reference check compares both labels and probability arrays against the historical experiment.

No test session was hand-labeled; no leaderboard result is an input to model fitting.
