# Recreation checks actually executed

The dedicated package was installed into its **own virtual environment** with `uv sync --all-extras`. That environment could not import the original `be_ml_cracked` package.

## Final fitting and prediction

Executed from this project folder:

```bash
uv run python run.py train --data ../data
uv run python run.py predict --data ../data
uv run python run.py verify \
  --reference ../artifacts/round3/submission_B.csv \
  --reference-artifacts ../artifacts/round3
```

All three models were fitted again from the competition training file. The check returned:

```json
{
  "rows": 324,
  "probabilities_valid": true,
  "reference_labels_identical": true,
  "reference_bytes_identical": true,
  "reference_member_probabilities_match_at_1e-12": true
}
```

The blended score matrix also matched the historical B matrix at absolute tolerance `1e-12`. No historical fitted model bundle was used by the standalone training path.

## Real first-fold CV comparison

Executed:

```bash
uv run python run.py cv --data ../data --repeats 1 --folds 1
```

| Member | Macro-F1 | Micro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| `crop128_noidf` | 0.9650795724 | 0.9710365854 | 0.9693149532 |
| `fusion_context` | 0.9638151666 | 0.9725609756 | 0.9700720116 |
| `reference128` | 0.9693804152 | 0.9786585366 | 0.9764624048 |

The saved train/validation indices and all three float32 probability arrays were **exactly equal** to the original experiment's first-fold artifacts. This is a first-fold implementation check, not a new complete repeated-CV result. The complete historical results remain labeled as historical in [VALIDATION.md](VALIDATION.md).

## Notebooks and code quality

All three notebooks were validated with `nbformat` and executed with real Jupyter kernels using:

```bash
uv run python scripts/execute_notebooks.py
```

Notebook 02 executed its synthetic training example with real-data heavy CV disabled by default. Notebook 03 executed real test prediction and the historical-reference check. Executed copies are generated under `outputs/notebooks/`.

Formatting, Ruff, strict mypy, unit tests, dependency compatibility and the lockfile check were run successfully. The tests include an actual small nested-CV run and its cache-resume check. Full real-data nested CV is implemented but was not rerun end-to-end as part of this refactoring; its original results are documented separately.

The final standalone suite passed **13 tests**, including local documentation-link checks and clean-notebook checks. The combined repository suite passed **35 tests**. Strict mypy passed for all 11 Python source and entry-point files checked, and the original round-three artifact audit still passed for six models and 60 fold files.
