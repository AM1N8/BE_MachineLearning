"""Command-line phase orchestration; run from the repository root."""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from copilote.io import read_sessions, write_json
from copilote.settings import load_config, provenance
from copilote.tokens import parse_token

LOGGER = logging.getLogger(__name__)


def inspect(config: dict[str, Any]) -> dict[str, Any]:
    """Verify source counts and lossless decomposition of all distinct tokens."""
    train = read_sessions(config["paths"]["train"], labelled=True)
    test = read_sessions(config["paths"]["test"], labelled=False)
    vocabulary = {token for row in train + test for token in row.tokens}
    for raw in vocabulary:
        if parse_token(raw).reconstruct() != raw:
            raise ValueError(f"Non-lossless token: {raw!r}")
    classes = Counter(row.label for row in train)
    if min(classes.values()) < config["cv"]["folds"]:
        raise ValueError("Too few sessions in a class for the requested folds")
    result = {
        "train_sessions": len(train),
        "test_sessions": len(test),
        "classes": len(classes),
        "min_class_size": min(classes.values()),
        "distinct_tokens_union": len(vocabulary),
        "roundtrip_failures": 0,
        "nested_context_tokens": sum(
            "(" in value for raw in vocabulary for value in parse_token(raw).screens
        ),
        "provenance": provenance(config),
    }
    write_json("reports/results/loading.json", result)
    LOGGER.info(
        "Loading gate: %s", {key: val for key, val in result.items() if key != "provenance"}
    )
    return result


def main() -> None:
    """Run a requested verified phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=[
            "inspect",
            "eda",
            "features",
            "train",
            "interpret",
            "predict",
            "report",
            "all",
            "fast",
        ],
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--force", action="store_true", help="Ignore cached model evaluations")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config, args.seed)
    for directory in ("reports/results", "reports/figures", "artifacts", "submissions"):
        Path(directory).mkdir(parents=True, exist_ok=True)
    if args.phase == "inspect":
        inspect(config)
    else:
        from copilote.eda import run_eda

        if args.phase == "eda":
            run_eda(config)
        elif args.phase == "features":
            from copilote.features import run_feature_analysis

            run_feature_analysis(config)
        elif args.phase in ("train", "fast"):
            from copilote.evaluate import run_training

            run_training(config, force=args.force, fast=args.phase == "fast")
        elif args.phase == "interpret":
            from copilote.interpret import run_interpretation

            run_interpretation(config)
        elif args.phase == "predict":
            from copilote.predict import run_prediction

            run_prediction(config)
        elif args.phase == "report":
            from copilote.report import run_report

            run_report(config)
        elif args.phase == "all":
            from copilote.evaluate import run_training
            from copilote.features import run_feature_analysis
            from copilote.interpret import run_interpretation
            from copilote.predict import run_prediction
            from copilote.report import run_report

            run_eda(config)
            run_feature_analysis(config)
            run_training(config, force=args.force)
            run_interpretation(config)
            run_prediction(config)
            run_report(config)


if __name__ == "__main__":
    main()
