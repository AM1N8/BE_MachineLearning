"""Summarize development errors without consulting test outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from copilote.competitive import RESULTS
from copilote.io import read_sessions, write_json
from copilote.settings import load_config
from copilote.tokens import parse_token


def diagnose(config: dict, name: str) -> dict:
    """Export error patterns and small session excerpts for a chosen CV candidate."""
    frame = pd.read_csv(RESULTS / f"development_{name}.csv")
    frame["correct"] = frame.truth == frame.prediction
    frame["length_bucket"] = pd.cut(
        frame.tokens, [0, 100, 500, 2000, float("inf")], right=False
    ).astype(str)
    rows = {row.row_id: row for row in read_sessions(config["paths"]["train"], labelled=True)}
    errors = frame[~frame.correct]
    excerpts = []
    for record in errors.sort_values("tokens").head(15).to_dict("records"):
        row = rows[record["row_id"]]
        record["start_events"] = [t for t in row.tokens if parse_token(t).seconds is None][:20]
        excerpts.append(record)
    result = {
        "candidate": name,
        "development_only": True,
        "accuracy_by_length": frame.groupby("length_bucket")
        .correct.agg(["count", "mean"])
        .reset_index()
        .to_dict("records"),
        "confusions": errors.groupby(["truth", "prediction"])
        .size()
        .sort_values(ascending=False)
        .head(20)
        .reset_index(name="count")
        .to_dict("records"),
        "short_errors": [
            {key: value for key, value in record.items() if key != "start_events"}
            for record in excerpts
        ],
    }
    # Session excerpts are raw data and must stay in the git-ignored artifact area.
    write_json(Path("artifacts/competitive") / f"error_excerpts_{name}.json", excerpts)
    write_json(RESULTS / f"diagnostics_{name}.json", result)
    return result


def main() -> None:
    """Write a traceable development-error artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--name", default="baseline")
    args = parser.parse_args()
    result = diagnose(load_config(args.config, args.seed), args.name)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
