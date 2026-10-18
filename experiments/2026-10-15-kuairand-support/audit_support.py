"""Exploratory dataset-support probe, not a production loader or paper result."""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
COLUMNS = ["user_id", "video_id", "date", "hourmin", "time_ms", "tab", "long_view", "duration_ms", "play_time_ms", "is_rand"]
KEYS = ["user_id", "video_id", "time_ms", "tab"]


def write_json(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def quantiles(values):
    data = np.asarray(values)
    return {str(q): float(np.quantile(data, q)) for q in [0, .1, .25, .5, .75, .9, .95, .99, 1]} if len(data) else {}


def split_name(frame, policy):
    times = frame["time_ms"].to_numpy()
    result = np.full(len(frame), "outside", dtype="U16")
    t0, t1, t2, t3 = [pd.Timestamp(value, tz="UTC").value // 1_000_000 for value in ["2022-04-08", "2022-04-19", "2022-04-22", "2022-05-09"]]
    if policy == "standard":
        result[(times >= t0) & (times < t1)] = "train"
        result[(times >= t1) & (times < t2)] = "validation"
    result[(times >= t2) & (times < t3)] = "test_" + policy
    return result


def main():
    started = time.perf_counter()
    protocol_bytes = (ROOT / "protocol.json").read_bytes()
    frames, source_info = [], []
    for path in sorted(RAW.glob("log_*_pure.csv")):
        schema = list(pd.read_csv(path, nrows=0).columns)
        missing = sorted(set(COLUMNS) - set(schema))
        if missing:
            raise RuntimeError(f"Missing required fields in {path.name}: {missing}")
        frame = pd.read_csv(path, usecols=COLUMNS)
        if frame[COLUMNS].isna().any().any():
            raise RuntimeError(f"Null required field in {path.name}")
        policy = "random" if path.name.startswith("log_random_") else "standard"
        expected = (frame.play_time_ms >= np.minimum(frame.duration_ms, 18000)).astype("int8")
        mismatches = int((expected != frame.long_view).sum())
        if not set(frame.long_view.unique()).issubset({0, 1}):
            raise RuntimeError("Non-binary target")
        if not frame.tab.between(0, 14).all():
            raise RuntimeError("Unknown context ID outside documented [0,14]")
        if mismatches and not (ROOT / "protocol_amendment_01.json").exists():
            raise RuntimeError(f"Documented target rule disagrees on {mismatches} rows")
        frame["split"] = split_name(frame, policy)
        frame["policy"] = policy
        frame["hour_bin"] = ((frame.time_ms // 3_600_000 % 24) // 4).astype("int8")
        frame["tab_hour_bin"] = (frame.tab * 6 + frame.hour_bin).astype("int16")
        source_info.append({
            "file": path.name, "rows": len(frame), "all_columns": schema,
            "date_min": int(frame.date.min()), "date_max": int(frame.date.max()),
            "target_rate": float(frame.long_view.mean()), "target_rule_mismatches": mismatches,
            "zero_duration_rows": int((frame.duration_ms == 0).sum()),
            "mismatches_zero_duration": int(((expected != frame.long_view) & (frame.duration_ms == 0)).sum()),
            "mismatches_positive_duration": int(((expected != frame.long_view) & (frame.duration_ms > 0)).sum()),
            "label_rule_contingency": {f"native_{native}_expected_{rule}": int(((frame.long_view == native) & (expected == rule)).sum()) for native in [0, 1] for rule in [0, 1]},
            "duplicate_event_keys": int(frame.duplicated(KEYS).sum()),
            "is_rand_counts": {str(k): int(v) for k, v in frame.is_rand.value_counts().items()},
        })
        frames.append(frame)
        print(json.dumps({"loaded": path.name, "rows": len(frame)}), flush=True)
    if len(frames) != 3:
        raise RuntimeError(f"Expected exactly 3 logs; found {len(frames)}")
    data = pd.concat(frames, ignore_index=True)
    cross_file_duplicates = int(data.duplicated(KEYS).sum()) - sum(item["duplicate_event_keys"] for item in source_info)
    duplicate_mask = data.duplicated(KEYS, keep=False)
    quarantine = {
        "rows": int(duplicate_mask.sum()),
        "event_key_groups": int(data.loc[duplicate_mask, KEYS].drop_duplicates().shape[0]),
        "by_policy": {str(k): int(v) for k, v in data.loc[duplicate_mask].groupby("policy").size().items()},
        "by_canonical_split": {str(k): int(v) for k, v in data.loc[duplicate_mask].groupby("split").size().items()},
        "label_independent_rule": "all rows in repeated event-key groups quarantined",
    }
    raw_total_rows = len(data)
    data = data.loc[~duplicate_mask].copy()
    train = data.loc[data.split == "train"]
    if not len(train):
        raise RuntimeError("Empty source training split")
    train_users, train_items = set(train.user_id), set(train.video_id)
    user_activity = train.groupby("user_id").size()
    support, coverage = {}, {}
    for context in ["tab", "hour_bin", "tab_hour_bin"]:
        pair_counts = train.groupby(["user_id", context]).size()
        supported = {}
        for threshold in [3, 5, 10, 20]:
            counts_per_user = pair_counts.loc[pair_counts >= threshold].groupby(level=0).size()
            users = set(counts_per_user.loc[counts_per_user >= 2].index)
            supported[str(threshold)] = {
                "users_with_two_supported_contexts": len(users),
                "fraction_train_users": len(users) / len(train_users),
                "train_event_coverage": float(train.user_id.isin(users).mean()),
                "validation_event_coverage": float(data.loc[data.split == "validation", "user_id"].isin(users).mean()),
                "test_standard_event_coverage": float(data.loc[data.split == "test_standard", "user_id"].isin(users).mean()),
                "test_random_event_coverage": float(data.loc[data.split == "test_random", "user_id"].isin(users).mean()),
            }
        support[context] = {
            "global_context_counts": {str(k): int(v) for k, v in train[context].value_counts().sort_index().items()},
            "observed_contexts_per_user": quantiles(train.groupby("user_id")[context].nunique()),
            "observations_per_observed_user_context": quantiles(pair_counts),
            "thresholds": supported,
        }
    split_info = {}
    split_times = {}
    cache = {}
    for name in ["train", "validation", "test_standard", "test_random", "outside"]:
        part = data.loc[data.split == name]
        if not len(part):
            split_info[name] = {"rows": 0}
            continue
        warm_u, warm_i = part.user_id.isin(train_users), part.video_id.isin(train_items)
        split_info[name] = {
            "rows": len(part), "users": int(part.user_id.nunique()), "items": int(part.video_id.nunique()),
            "date_min": int(part.date.min()), "date_max": int(part.date.max()),
            "event_time_min_utc": str(pd.Timestamp(int(part.time_ms.min()), unit="ms", tz="UTC")),
            "event_time_max_utc": str(pd.Timestamp(int(part.time_ms.max()), unit="ms", tz="UTC")),
            "target_rate": float(part.long_view.mean()),
            "warm_user_fraction": float(warm_u.mean()), "warm_item_fraction": float(warm_i.mean()),
            "warm_user_item_fraction": float((warm_u & warm_i).mean()),
            "context_counts": {str(k): int(v) for k, v in part.tab.value_counts().sort_index().items()},
            "duplicate_event_keys": int(part.duplicated(KEYS).sum()),
        }
        split_times[name] = (int(part.time_ms.min()), int(part.time_ms.max()))
        for column, dtype in [("user_id", "int32"), ("video_id", "int32"), ("tab", "int16"), ("hour_bin", "int8"), ("duration_ms", "int32"), ("long_view", "int8")]:
            cache[name + "__" + column] = part[column].to_numpy(dtype=dtype)
    temporal_checks = {
        "train_before_validation": split_times["train"][1] < split_times["validation"][0],
        "validation_before_standard_test": split_times["validation"][1] < split_times["test_standard"][0],
        "validation_before_random_test": split_times["validation"][1] < split_times["test_random"][0],
    }
    if not all(temporal_checks.values()):
        raise RuntimeError(f"Raw timestamp separation failed: {temporal_checks}")
    if data.duplicated(KEYS).any():
        raise RuntimeError("Duplicate keys remain after conservative quarantine")
    grouped = data.groupby(["split", "policy", "date", "tab"], observed=True).agg(rows=("long_view", "size"), positives=("long_view", "sum"), target_rate=("long_view", "mean"), users=("user_id", "nunique"))
    grouped.to_csv(ROOT / "date_context_counts.csv")
    train.groupby(["user_id", "tab"]).agg(rows=("long_view", "size"), positives=("long_view", "sum")).to_csv(ROOT / "training_user_context_counts.csv")
    np.savez_compressed(ROOT / "audit_arrays.npz", **cache)
    result = {
        "scope": "exploratory support and count-baseline probe",
        "source_protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "protocol_amendments": [{"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(ROOT.glob("protocol_amendment_*.json"))],
        "documentation_consistency_passed": all(item["target_rule_mismatches"] == 0 for item in source_info),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "sources": source_info, "raw_total_rows": raw_total_rows, "total_rows": len(data), "duplicate_key_quarantine": quarantine, "split_counts": split_info,
        "temporal_checks": temporal_checks, "cross_file_duplicate_event_keys": cross_file_duplicates,
        "training_user_activity_quantiles": quantiles(user_activity),
        "training_user_activity_counts": {str(n): int((user_activity >= n).sum()) for n in [1, 5, 20, 50, 100]},
        "context_support": support,
        "features_in_baseline_cache": ["user_id", "video_id", "tab", "hour_bin", "duration_ms"],
        "target_in_cache": "long_view",
        "excluded": "current outcomes other than target; all user/video side-information and month-aggregated statistics",
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json("support_audit.json", result)
    print(json.dumps({"total_rows": len(data), "split_counts": split_info, "tab_support": support["tab"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
