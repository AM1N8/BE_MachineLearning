"""Phase 4 error analysis: fold-local exclusivity, error taxonomy, snap-rule diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from copilote.improvement import metrics
from copilote.io import Session, file_hash, read_sessions, write_json
from copilote.tokens import filled_context, parse_token

OOF_PATH = "reports/results/improvement/4e7e8c2ad9a6/baseline_oof.csv"
SPLITS_PATH = "reports/results/improvement/4e7e8c2ad9a6/research_splits.json"
FORENSICS_PATH = "reports/results/phase4/d524c6ba3719/session_timing_train.csv"
PHASE2_DIR = Path("reports/results/phase2/a1d191a9b558")
TIMING_COLUMNS = (
    "duration",
    "events_per_second",
    "suffix_fraction",
    "dwell_top_fraction",
    "gap_max",
)
DIAGNOSTICS = (
    "length",
    "unique",
    "entropy",
    "repetition",
    "oov",
    "rare",
    "unseen_transition",
    "margin",
)
VIEWS = ("raw", "sig")


def session_token_sets(sessions: list[Session]) -> dict[int, tuple[set[str], set[str]]]:
    """Distinct raw event tokens and filled-context signature tokens per session."""
    sets: dict[int, tuple[set[str], set[str]]] = {}
    for row in sessions:
        raw = {token for token in row.tokens if parse_token(token).seconds is None}
        signatures = {
            json.dumps((token.action, screen, config, chain), ensure_ascii=False)
            for token, screen, config, chain in filled_context(row.tokens)
        }
        sets[row.row_id] = (raw, signatures)
    return sets


def exclusive_maps(
    by_id: dict[int, Session], train_ids: list[int], token_sets: dict[int, tuple[set, set]]
) -> list[dict[str, str]]:
    """Fold-local token -> single-owner maps for both views (training sessions only)."""
    owners: list[defaultdict[str, set[str]]] = [defaultdict(set), defaultdict(set)]
    for row_id in train_ids:
        label = str(by_id[row_id].label)
        for view in (0, 1):
            for token in token_sets[row_id][view]:
                owners[view][token].add(label)
    return [
        {token: next(iter(users)) for token, users in owners[view].items() if len(users) == 1}
        for view in (0, 1)
    ]


def candidate_counts(present: set[str], exclusive: dict[str, str]) -> Counter:
    """Distinct exclusive tokens present per candidate user in one validation session."""
    counts: Counter = Counter()
    for token in present:
        user = exclusive.get(token)
        if user is not None:
            counts[user] += 1
    return counts


def exclusive_attributes(
    by_id: dict[int, Session],
    token_sets: dict[int, tuple[set, set]],
    splits: list[dict[str, list[int]]],
) -> list[dict[str, Any]]:
    """Per (fold, row) exclusive-token support recomputed on the fold training half."""
    records = []
    for fold, split in enumerate(splits):
        maps = exclusive_maps(by_id, split["train"], token_sets)
        for row_id in sorted(split["validation"]):
            truth = str(by_id[row_id].label)
            record: dict[str, Any] = {"row_id": row_id, "fold": fold}
            for view, name in enumerate(VIEWS):
                counts = candidate_counts(token_sets[row_id][view], maps[view])
                unique = counts.most_common(1)[0][0] if len(counts) == 1 else None
                record[f"n_truth_excl_{name}"] = counts.get(truth, 0)
                record[f"n_candidates_{name}"] = len(counts)
                record[f"max_excl_{name}"] = max(counts.values(), default=0)
                record[f"unique_cand_{name}"] = unique
            records.append(record)
    return records


def snap_predictions(
    oof: pd.DataFrame, view: str, min_count: int, margin_threshold: float | None
) -> tuple[np.ndarray, pd.Series]:
    """Apply the unambiguous fold-local exclusive-token snap rule to OOF predictions."""
    prediction = oof.prediction.to_numpy(dtype=object)
    fired = (
        (oof[f"n_candidates_{view}"] == 1)
        & (oof[f"max_excl_{view}"] >= min_count)
        & (oof[f"unique_cand_{view}"] != prediction)
    )
    if margin_threshold is not None:
        fired = fired & (oof.margin < margin_threshold)
    fired = fired.reset_index(drop=True)
    new_prediction = prediction.copy()
    new_prediction[fired.to_numpy()] = oof.loc[fired, f"unique_cand_{view}"].to_numpy(dtype=object)
    return new_prediction, fired


def simulate_snap(
    oof: pd.DataFrame, view: str, min_count: int, margin_threshold: float | None
) -> dict[str, Any]:
    """Diagnostic fold-local simulation: fixes, breakages, macro-F1 before and after."""
    truth = oof.truth.to_numpy(dtype=object)
    wrong = oof.wrong.to_numpy(dtype=bool)
    new_prediction, fired = snap_predictions(oof, view, min_count, margin_threshold)
    fixed = wrong & (new_prediction == truth)
    broken = ~wrong & (new_prediction != truth)
    before = metrics(list(truth), list(oof.prediction))
    after = metrics(list(truth), list(new_prediction))
    always = set(oof.loc[oof.always_wrong, "row_id"])
    fixed_sessions = set(oof.loc[fixed, "row_id"])
    return {
        "rule": f"{view} unique candidate, min {min_count}"
        + (f", margin < {margin_threshold}" if margin_threshold is not None else ""),
        "fired": int(fired.sum()),
        "errors_fixed": int(fixed.sum()),
        "correct_broken": int(broken.sum()),
        "net_errors": int(fixed.sum() - broken.sum()),
        "always_wrong_sessions_fixed": len(always & fixed_sessions),
        "macro_f1_before": before["macro_f1"],
        "macro_f1_after": after["macro_f1"],
        "accuracy_after": after["accuracy"],
    }


def user_profiles(
    sessions: list[Session], token_sets: dict[int, tuple[set, set]]
) -> dict[str, dict[str, Any]]:
    """Descriptive full-train per-user profile used for confusion-pair similarity."""
    lengths: defaultdict[str, list[int]] = defaultdict(list)
    browsers: dict[str, set[str]] = defaultdict(set)
    raw_union: defaultdict[str, set[str]] = defaultdict(set)
    sig_union: defaultdict[str, set[str]] = defaultdict(set)
    for row in sessions:
        label = str(row.label)
        lengths[label].append(len(token_sets[row.row_id][0]))
        browsers[label].add(row.browser)
        raw_union[label] |= token_sets[row.row_id][0]
        sig_union[label] |= token_sets[row.row_id][1]
    return {
        str(row.label): {
            "sessions": len(lengths[str(row.label)]),
            "median_tokens": float(np.median(lengths[str(row.label)])),
            "browsers": sorted(browsers[str(row.label)]),
            "raw_union": raw_union[str(row.label)],
            "sig_union": sig_union[str(row.label)],
        }
        for row in sessions
    }


def confusion_pairs(errors: pd.DataFrame, profiles: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Aggregate error instances per (truth, prediction) pair with user-pair similarity."""
    rows = []
    for (truth, prediction), block in errors.groupby(["truth", "prediction"]):
        left, right = profiles[truth], profiles[prediction]
        raw_overlap = len(left["raw_union"] & right["raw_union"])
        raw_union = len(left["raw_union"] | right["raw_union"])
        sig_overlap = len(left["sig_union"] & right["sig_union"])
        sig_union = len(left["sig_union"] | right["sig_union"])
        rows.append(
            {
                "truth": truth,
                "prediction": prediction,
                "n_errors": int(len(block)),
                "n_second_eq_truth": int((block.second == truth).sum()),
                "n_margin_lt_03": int((block.margin < 0.3).sum()),
                "n_truth_excl_raw_ge1": int((block.n_truth_excl_raw >= 1).sum()),
                "n_truth_excl_sig_ge1": int((block.n_truth_excl_sig >= 1).sum()),
                "jaccard_raw": float(raw_overlap / raw_union) if raw_union else 0.0,
                "jaccard_sig": float(sig_overlap / sig_union) if sig_union else 0.0,
                "same_browser": bool(left["browsers"] == right["browsers"]),
                "truth_sessions": int(left["sessions"]),
                "pred_sessions": int(right["sessions"]),
                "truth_median_tokens": float(left["median_tokens"]),
                "pred_median_tokens": float(right["median_tokens"]),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("n_errors", ascending=False).reset_index(drop=True)


def always_wrong_table(oof: pd.DataFrame) -> pd.DataFrame:
    """Both-repeat-wrong sessions with rescue potential per mechanism."""
    rows = []
    for row_id, block in oof[oof.always_wrong].groupby("row_id"):
        snap_raw = block.apply(
            lambda r: r.n_candidates_raw == 1 and r.unique_cand_raw == r.truth, axis=1
        )
        snap_sig = block.apply(
            lambda r: r.n_candidates_sig == 1 and r.unique_cand_sig == r.truth, axis=1
        )
        snap_fixable = bool(snap_raw.any() or snap_sig.any())
        rerankable = bool((block.second == block.truth).any())
        category = "snap_fixable" if snap_fixable else "rerankable" if rerankable else "hard"
        rows.append(
            {
                "row_id": row_id,
                "label": block.truth.iloc[0],
                "browser": block.browser.iloc[0],
                "length": int(block.length.iloc[0]),
                "min_margin": float(block.margin.min()),
                "max_truth_excl_raw": int(block.n_truth_excl_raw.max()),
                "max_truth_excl_sig": int(block.n_truth_excl_sig.max()),
                "second_eq_truth_any": rerankable,
                "snap_fixable": snap_fixable,
                "category": category,
            }
        )
    return pd.DataFrame(rows).sort_values("category")


def signatures_delta(oof: pd.DataFrame) -> dict[str, Any]:
    """Historical-protocol baseline -> signatures corrected/introduced row lists."""
    corrected: list[int] = []
    introduced: list[int] = []
    for fold in range(8):
        base = pd.read_csv(PHASE2_DIR / f"research_baseline_{fold}.csv")
        sigs = pd.read_csv(PHASE2_DIR / f"research_signatures_{fold}.csv")
        merged = base.merge(sigs, on=["row_id", "fold"], suffixes=("_base", "_sig"))
        for record in merged.itertuples():
            if record.truth_base != record.truth_sig:
                raise ValueError(f"fold {fold} row {record.row_id}: truth columns disagree")
            base_wrong = record.truth_base != record.prediction_base
            sig_wrong = record.truth_base != record.prediction_sig
            if base_wrong and not sig_wrong:
                corrected.append(record.row_id)
            if sig_wrong and not base_wrong:
                introduced.append(record.row_id)
    delta = oof[oof.row_id.isin(corrected)]
    introduced_rows = oof[oof.row_id.isin(introduced)]
    return {
        "protocol": "historical 8-fold, baseline -> context signatures",
        "corrected": len(corrected),
        "introduced": len(introduced),
        "corrected_row_ids": corrected,
        "introduced_row_ids": introduced,
        "corrected_with_truth_excl_raw": int((delta.n_truth_excl_raw >= 1).sum()),
        "introduced_with_truth_excl_raw": int((introduced_rows.n_truth_excl_raw >= 1).sum()),
    }


def run(seed: int = 42) -> dict[str, Any]:
    """Execute the full error-analysis pass and persist machine-readable artifacts."""
    manifest = {
        "sources": {
            name: file_hash(f"src/copilote/{name}.py")
            for name in ("phase4_errors", "tokens", "io", "improvement", "phase2_models")
        },
        "train": file_hash("data/train.csv"),
        "baseline_oof": file_hash(OOF_PATH),
        "splits": file_hash(SPLITS_PATH),
        "forensics_timing": file_hash(FORENSICS_PATH),
        "phase2_oof": {
            f"{family}_{fold}": file_hash(PHASE2_DIR / f"research_{family}_{fold}.csv")
            for family in ("baseline", "signatures")
            for fold in range(8)
        },
        "seed": seed,
        "stage": "error_analysis",
    }
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    directory = Path("reports/results/phase4") / signature[:12]
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "manifest.json", manifest)

    sessions = read_sessions("data/train.csv", labelled=True)
    by_id = {row.row_id: row for row in sessions}
    token_sets = session_token_sets(sessions)
    splits = json.loads(Path(SPLITS_PATH).read_text())
    oof = pd.read_csv(OOF_PATH)
    for fold, split in enumerate(splits):
        validation = set(split["validation"])
        observed = set(oof.loc[oof.fold == fold, "row_id"])
        if validation != observed:
            raise ValueError(f"fold {fold}: splits and OOF rows disagree")

    attributes = pd.DataFrame(exclusive_attributes(by_id, token_sets, splits))
    oof = oof.merge(attributes, on=["row_id", "fold"], validate="one_to_one")
    timing = pd.read_csv(FORENSICS_PATH, usecols=["row_id", *TIMING_COLUMNS, "suffix_events"])
    oof = oof.merge(timing, on="row_id", validate="many_to_one")
    wrong_twice = Counter(oof.loc[oof.wrong, "row_id"])
    oof["always_wrong"] = oof.row_id.map(lambda r: wrong_twice.get(r, 0) == 2)

    errors = oof[oof.wrong].reset_index(drop=True)
    correct = oof[~oof.wrong]
    contrast = []
    for feature in DIAGNOSTICS + TIMING_COLUMNS + ("suffix_events",):
        if feature not in oof.columns:
            continue
        contrast.append(
            {
                "feature": feature,
                "errors_median": float(errors[feature].median()),
                "correct_median": float(correct[feature].median()),
                "errors_mean": float(errors[feature].mean()),
                "correct_mean": float(correct[feature].mean()),
            }
        )

    coverage = {
        f"n_truth_excl_{name}": {
            "errors_ge1": int((errors[f"n_truth_excl_{name}"] >= 1).sum()),
            "errors_ge1_fraction": float((errors[f"n_truth_excl_{name}"] >= 1).mean()),
            "correct_ge1_fraction": float((correct[f"n_truth_excl_{name}"] >= 1).mean()),
            "errors_mean": float(errors[f"n_truth_excl_{name}"].mean()),
            "correct_mean": float(correct[f"n_truth_excl_{name}"].mean()),
        }
        for name in VIEWS
    }

    conditions = [
        ("short_length", errors.length <= 20),
        ("exclusive_missed", errors.n_truth_excl_raw >= 1),
        ("top2_recoverable", errors.second == errors.truth),
        ("close_margin", errors.margin < 0.3),
    ]
    assigned = pd.Series(False, index=errors.index)
    taxonomy = {}
    for name, condition in conditions:
        selected = condition & ~assigned
        taxonomy[name] = {
            "count": int(selected.sum()),
            "row_ids": sorted(errors.loc[selected, "row_id"].tolist()),
        }
        assigned |= selected
    taxonomy["confident_wrong"] = {
        "count": int((~assigned).sum()),
        "row_ids": sorted(errors.loc[~assigned, "row_id"].tolist()),
    }

    simulation = {
        "variants": [
            simulate_snap(oof, view, min_count, margin)
            for view in VIEWS
            for min_count, margin in ((1, None), (2, None), (1, 0.3))
        ],
        "caveat": "diagnostic fold-local simulation on historical OOF; adoption requires the phase C/D engine",
    }

    profiles = user_profiles(sessions, token_sets)
    pairs = confusion_pairs(errors, profiles)
    always = always_wrong_table(oof)

    results = {
        "signature": signature,
        "directory": str(directory),
        "stage": "error_analysis",
        "seed": seed,
        "protocol": {
            "oof_rows": len(oof),
            "folds": oof.fold.nunique(),
            "errors": int(oof.wrong.sum()),
            "unique_wrong_sessions": oof.loc[oof.wrong, "row_id"].nunique(),
            "always_wrong_sessions": int(oof.always_wrong.sum() / 2),
            "errors_second_eq_truth": int((errors.second == errors.truth).sum()),
            "errors_margin_lt_03": int((errors.margin < 0.3).sum()),
        },
        "diagnostic_contrast": contrast,
        "exclusive_coverage": coverage,
        "taxonomy": taxonomy,
        "snap_simulation": simulation,
        "confusion_pairs_top": pairs.head(10).to_dict(orient="records"),
        "always_wrong": {
            "total": len(always),
            "snap_fixable": int((always.category == "snap_fixable").sum()),
            "rerankable": int((always.category == "rerankable").sum()),
            "hard": int((always.category == "hard").sum()),
        },
        "signatures_delta": signatures_delta(oof),
    }
    write_json(directory / "errors.json", results)
    oof.to_csv(directory / "oof_exclusive.csv", index=False)
    errors.to_csv(directory / "errors_enriched.csv", index=False)
    pairs.to_csv(directory / "confusion_pairs.csv", index=False)
    always.to_csv(directory / "always_wrong.csv", index=False)
    summary = {
        "directory": str(directory),
        "protocol": results["protocol"],
        "coverage_raw_errors_ge1": coverage["n_truth_excl_raw"]["errors_ge1"],
        "taxonomy_counts": {name: block["count"] for name, block in taxonomy.items()},
        "snap_variants": [
            {
                "rule": variant["rule"],
                "fixed": variant["errors_fixed"],
                "broken": variant["correct_broken"],
                "macro_f1_after": round(variant["macro_f1_after"], 5),
            }
            for variant in simulation["variants"]
        ],
        "always_wrong": results["always_wrong"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return results


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Phase 4 error analysis")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    run(arguments.seed)


if __name__ == "__main__":
    main()
