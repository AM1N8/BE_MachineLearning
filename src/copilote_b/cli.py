"""Command-line entry points for the standalone solution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

from .config import load_config  # noqa: E402
from .evaluation import cross_validate, evaluate_blend, nested_validate  # noqa: E402
from .pipeline import initialize, load_training, predict, train, verify  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("train", "predict", "cv", "blend", "nested", "verify", "all")
    )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reproduction"))
    parser.add_argument("--config", type=Path, default=Path("configs/solution.json"))
    parser.add_argument("--repeats", type=int, choices=(1, 2), default=2)
    parser.add_argument("--folds", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-artifacts", type=Path)
    args = parser.parse_args()
    if args.stage == "all" and (args.repeats != 2 or args.folds != 5):
        parser.error("all requires two complete five-fold repeats")
    config = load_config(args.config)
    if args.stage == "verify":
        print(
            json.dumps(
                verify(args.output, args.reference, args.reference_artifacts), indent=2
            )
        )
        return
    initialize(args.data, args.output, config)
    if args.stage in ("train", "cv", "blend", "nested", "all"):
        _, traces, y, classes = load_training(args.data)
    if args.stage in ("cv", "all"):
        print(
            cross_validate(
                traces, y, classes, config, args.output, args.repeats, args.folds
            )
        )
    if args.stage in ("blend", "all"):
        print(json.dumps(evaluate_blend(y, classes, config, args.output), indent=2))
    if args.stage in ("nested", "all"):
        print(
            json.dumps(
                nested_validate(traces, y, classes, config, args.output, args.folds),
                indent=2,
            )
        )
    if args.stage in ("train", "all"):
        train(traces, y, classes, config, args.output)
    if args.stage in ("predict", "all"):
        print(predict(args.data, config, args.output).shape)
        print(
            json.dumps(
                verify(args.output, args.reference, args.reference_artifacts), indent=2
            )
        )
