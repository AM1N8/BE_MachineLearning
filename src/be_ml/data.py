"""Loading and parsing utilities for the Copilote usage-trace datasets.

Each raw CSV row is a single session (recording) made by one user. The first
two columns of the training set are the target user (``util``) and the browser
(``navigateur``); every following column is an action token of the session.
Because sessions have different lengths, the file is a ragged "CSV" that pandas
pads with NaN, which is normal and expected.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

TIME_MARKER_RE = re.compile(r"^t\d+$")
SCREEN_RE = re.compile(r"\((.*?)\)")
CONFIG_RE = re.compile(r"<(.*?)>")
CHAINE_RE = re.compile(r"\$(.*?)\$")

DELIMITERS = ("(", "<", "$", "1")


def read_ds(ds_name: str, data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Read a raw ``{ds_name}.csv`` file into a padded DataFrame of strings.

    ``ds_name`` is ``"train"`` or ``"test"``. The number of action columns is
    inferred from the longest line so no session is truncated.
    """
    path = Path(data_dir) / f"{ds_name}.csv"
    with open(path) as f:
        max_actions = max(len(line.rstrip("\n").split(",")) for line in f)
        f.seek(0)
        names = ["util", "navigateur"] if "train" in ds_name else ["navigateur"]
        names.extend(range(max_actions - len(names)))
        return pd.read_csv(f, names=names, dtype=str)


def filter_action(value: str) -> str:
    """Strip the screen/config/chain suffix to keep only the action verb."""
    for delim in DELIMITERS:
        low_ind = value.find(delim)
        if low_ind > 0:
            value = value[:low_ind]
    return value


def is_time_marker(value: str) -> bool:
    return bool(TIME_MARKER_RE.match(value))


def extract(pattern: re.Pattern[str], value: str) -> str | None:
    """Return the first group matched by ``pattern`` in ``value`` (or None)."""
    match = pattern.search(value)
    return match.group(1) if match else None


def session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build interpretable per-session features from the raw action columns.

    Time markers ``tNN`` are gaps of ``NN * 5`` seconds; ``duration_seconds`` is
    therefore the largest marker times five.
    """
    action_cols = df.columns[2:] if "util" in df.columns else df.columns[1:]
    actions = df[action_cols]
    records = []
    for _, row in actions.iterrows():
        tokens = row.dropna().tolist()
        time_values = [int(t[1:]) for t in tokens if is_time_marker(t)]
        screens = [extract(SCREEN_RE, t) for t in tokens if "(" in t]
        configs = [extract(CONFIG_RE, t) for t in tokens if "<" in t]
        chaines = [extract(CHAINE_RE, t) for t in tokens if "$" in t]
        verbs = [filter_action(t) for t in tokens if not is_time_marker(t)]
        records.append(
            {
                "n_tokens": len(tokens),
                "n_actions": len(verbs),
                "n_time_markers": len(time_values),
                "max_time": max(time_values) if time_values else 0,
                "duration_seconds": (max(time_values) if time_values else 0) * 5,
                "n_distinct_verbs": len(set(verbs)),
                "n_screens": len(screens),
                "n_distinct_screens": len(set(screens)),
                "n_configs": len(configs),
                "n_distinct_configs": len(set(configs)),
                "n_chaines": len(chaines),
                "n_distinct_chaines": len(set(chaines)),
            }
        )
    features = pd.DataFrame(records)
    features.insert(0, "navigateur", df["navigateur"].values)
    if "util" in df.columns:
        features.insert(0, "util", df["util"].values)
    return features


def to_categories(df: pd.DataFrame, col: str = "util") -> pd.DataFrame:
    """Convert a string label column into its integer categorical codes."""
    df[col] = pd.Categorical(df[col])
    df[col] = df[col].cat.codes
    return df
