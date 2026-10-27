# Round-three B — standalone solution

A focused recreation of `submission_B.csv` for **qui-utilise-mon-appli-challenge**. It contains only the three models used in B, their validation workflow, and the final prediction pipeline.

The user reported **0.999 on the public leaderboard** for this submission. This is an external observation, distinct from the recorded nested-CV macro-F1 of **0.97143 ± 0.00845**. The leaderboard's averaging convention has not been independently verified.

## Layout

```text
solution_b_round3/
├── configs/solution.json       # Three model configurations and submitted weights
├── data/README.md              # Required input files
├── docs/
│   ├── METHOD.md               # Features, crops, models and soft voting
│   ├── VALIDATION.md           # Validation protocol and historical results
│   └── REPRODUCTION_CHECK.md   # Checks actually executed
├── notebooks/
│   ├── 01_data_and_features.ipynb
│   ├── 02_training_and_validation.ipynb
│   └── 03_submission_and_verification.ipynb
├── src/copilote_b/
│   ├── config.py               # Configuration, hashes and manifests
│   ├── data.py                 # Robust session and sample parsing
│   ├── features.py             # Action and carried-context streams
│   ├── model.py                # Weighted crop training and browser conditioning
│   ├── ensemble.py             # Probability averaging and OOF weight fitting
│   ├── evaluation.py           # Repeated/nested CV and error analysis
│   ├── pipeline.py             # Model persistence, prediction and verification
│   └── cli.py                  # Command-line interface
├── scripts/execute_notebooks.py
├── tests/
│   ├── test_solution.py
│   └── test_documentation.py
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── run.py
```

The code has no imports from the original `be_ml_cracked` package. This folder can be copied elsewhere and used with the three competition input files.

## Setup

Use Python 3.11 and run these commands **from this folder**:

```bash
uv sync --frozen --all-extras
```

`--all-extras` installs notebook and development tools. For training and prediction alone, use `uv sync --frozen`.

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[notebooks,dev]'
```

The direct runtime requirements are also pinned in [requirements.txt](requirements.txt); `uv.lock` locks the full dependency graph. Dependencies are open-source packages with permissive licenses. No PyTorch, GPU or boosting library is needed.

## Reproduce the submitted B

Within the current repository, inputs already exist in `../data`:

```bash
uv run python run.py train --data ../data
uv run python run.py predict --data ../data
```

For a standalone copy, put the files listed in [data/README.md](data/README.md) in `data/`, and omit `--data ../data`.

The result is `outputs/reproduction/submission_B.csv`. Training persists one model bundle per member; prediction loads them sequentially to limit memory use. Numerical work uses one thread. Allow several minutes and a few GB of free memory for the real dataset.

The submitted model definitions and weights are frozen in [configs/solution.json](configs/solution.json). The prediction path does not retune weights or use test labels.

### Verify against the original experiment

When running inside this repository:

```bash
uv run python run.py verify \
  --reference ../artifacts/round3/submission_B.csv \
  --reference-artifacts ../artifacts/round3
```

This checks CSV labels and order, all three member probability matrices, class ordering, and the final blend probabilities at absolute tolerance `1e-12`. Reference files are optional verification inputs, not training dependencies.

## Reproduce validation

Quick first-fold run:

```bash
uv run python run.py cv --data ../data --repeats 1 --folds 1
```

Full repeated CV, apparent OOF weight fitting, and independent nested weight evaluation:

```bash
uv run python run.py cv --data ../data
uv run python run.py blend --data ../data
uv run python run.py nested --data ../data
```

The `blend` stage reports newly optimized weights in `outputs/reproduction/blend.json`. It **does not overwrite** the frozen submission weights. Its scores reuse optimization labels and are explicitly apparent. The `nested` stage refits models, vocabularies and weights inside each outer training partition.

To run repeated CV, blending, nested validation, final training and prediction in sequence:

```bash
uv run python run.py all --data ../data --output outputs/full_run
```

`all` requires both complete five-fold repeats. Use a fresh output directory after modifying code, inputs or configuration; a fingerprint prevents silent cache reuse after these changes. Cached fits and folds can otherwise be resumed.

## Notebooks

```bash
uv run jupyter lab notebooks
```

1. [Data and features](notebooks/01_data_and_features.ipynb): input format, imbalance, lengths, browser support, carried context and crop boundaries.
2. [Training and validation](notebooks/02_training_and_validation.ipynb): an executable small training example, optional real CV, OOF metrics and blend evaluation.
3. [Submission and verification](notebooks/03_submission_and_verification.ipynb): final prediction, exact schema checks, member contributions and optional comparison with original B.

The notebooks locate this project from their working directory. Set `COPILOTE_DATA` or `COPILOTE_OUTPUT` to override the default paths. Heavy CV is opt-in in notebook 02; notebook 03 expects trained model bundles from the `train` command.

To execute all three notebooks headlessly after training:

```bash
uv run python scripts/execute_notebooks.py
```

Generated, executed copies are saved under `outputs/notebooks/`; the source notebooks remain clean.

## Outputs

| Path under the chosen output directory | Contents |
|---|---|
| `run.json` | Configuration, data/code fingerprint and dependency versions |
| `models/*.joblib` | Fitted vocabularies, SVMs, browser support and class order |
| `cv/results.csv` | Fold counts and macro/micro/weighted F1 means and sample standard deviations |
| `cv/oof_*.npz` | Complete repeated OOF predictions, labels and class order |
| `cv/*_per_class.csv`, `*_confusions.csv`, `*_errors.csv` | Pooled OOF error analysis |
| `blend.json` | Apparent OOF optimization results; not independent validation |
| `nested.json` | Complete nested evaluation; partial runs use `nested_partial.json` |
| `test_*.npy`, `probabilities.npz` | Member and blended test scores |
| `submission_B.csv` | Exact `RowId,prediction` submission |
| `submission_validation.json`, `verification.json` | Submission and optional reference checks |

## Checks

```bash
uv run pytest -q
uv run ruff check .
uv run black --check src tests run.py scripts
uv run mypy --strict src run.py scripts
```

See [the method](docs/METHOD.md), [validation notes](docs/VALIDATION.md), and [checks actually executed](docs/REPRODUCTION_CHECK.md) for the reasoning, limitations and recreation evidence. Saved model bundles should be loaded only from a trusted local training run.
