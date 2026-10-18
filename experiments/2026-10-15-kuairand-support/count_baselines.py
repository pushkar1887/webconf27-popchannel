"""Throwaway count-model probe; not a GraphHash/CCE reproduction."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
C = 15
ALPHAS = [10.0, 50.0, 200.0]
WEIGHTS = [0.0, 0.25, 0.5, 1.0]
SEEDS = [11, 29, 47]
BUCKETS = 256


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    return (1 / (1 + np.exp(-np.clip(z, -25, 25)))).astype("float32")


def point_loss(y, p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def hash_users(users, seed):
    with np.errstate(over="ignore"):
        z = np.asarray(users, dtype="uint64") + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> 31)
    return (z % BUCKETS).astype("int64")


def table_counts(ids, contexts, y, size):
    keys = np.asarray(ids, dtype="int64") * C + contexts
    counts = np.bincount(keys, minlength=size * C).reshape(size, C).astype("float64")
    sums = np.bincount(keys, weights=y, minlength=size * C).reshape(size, C)
    return counts, sums


def lookup(table, ids, contexts, fallback):
    result = np.asarray(fallback, dtype="float32").copy()
    valid = (ids >= 0) & (ids < len(table))
    result[valid] = table[ids[valid], contexts[valid]]
    return result


class CountTables:
    def __init__(self, train):
        u, i, c, y = [train[x] for x in ["user_id", "video_id", "tab", "long_view"]]
        self.global_p = float((y.sum() + .5) / (len(y) + 1))
        self.cn = np.bincount(c, minlength=C).astype("float64")
        self.cs = np.bincount(c, weights=y, minlength=C)
        self.un, self.us = table_counts(u, c, y, int(u.max()) + 1)
        self.in_, self.is_ = table_counts(i, c, y, int(i.max()) + 1)
        self.mn = self.un.sum(axis=1)
        self.ms = self.us.sum(axis=1)
        self.gn, self.gs = {}, {}
        for seed in SEEDS:
            self.gn[seed], self.gs[seed] = table_counts(hash_users(u, seed), c, y, BUCKETS)
        self.training_count_bytes = sum(x.nbytes for x in [self.cn, self.cs, self.un, self.us, self.in_, self.is_, self.mn, self.ms]) + sum(x.nbytes for x in list(self.gn.values()) + list(self.gs.values()))

    def fit(self, alpha):
        cp = ((self.cs + alpha * self.global_p) / (self.cn + alpha)).astype("float32")
        return {
            "context": cp,
            "item": ((self.is_ + alpha * cp) / (self.in_ + alpha)).astype("float32"),
            "user": ((self.us + alpha * cp) / (self.un + alpha)).astype("float32"),
            "marginal": ((self.ms + alpha * self.global_p) / (self.mn + alpha)).astype("float32"),
            "groups": {seed: ((self.gs[seed] + alpha * cp) / (self.gn[seed] + alpha)).astype("float32") for seed in SEEDS},
        }

    def predictions(self, data, fitted):
        u, i, c = [data[x] for x in ["user_id", "video_id", "tab"]]
        cp = fitted["context"][c]
        ip = lookup(fitted["item"], i, c, cp)
        up = lookup(fitted["user"], u, c, cp)
        mp = np.full(len(u), self.global_p, dtype="float32")
        valid = (u >= 0) & (u < len(fitted["marginal"]))
        mp[valid] = fitted["marginal"][u[valid]]
        result = {"global_rate": np.full(len(u), self.global_p, dtype="float32"), "context_rate": cp, "item_context_rate": ip, "user_context_rate": up}
        base = logit(ip)
        effects = {"item_plus_user_context": logit(up) - logit(cp), "item_plus_user_marginal": logit(mp) - float(logit(self.global_p))}
        for seed in SEEDS:
            gp = fitted["groups"][seed][hash_users(u, seed), c]
            effects[f"item_plus_random_shared_s{seed}"] = logit(gp) - logit(cp)
        return result, base, effects


def metrics(data, p, mask=None):
    y = data["long_view"]
    if mask is not None:
        y, p = y[mask], p[mask]
    if not len(y):
        return {"rows": 0}
    bins = np.minimum((p * 10).astype("int64"), 9)
    counts = np.bincount(bins, minlength=10)
    positives = np.bincount(bins, weights=y, minlength=10)
    p_sums = np.bincount(bins, weights=p, minlength=10)
    nonempty = counts > 0
    ece = float(np.abs(positives[nonempty] - p_sums[nonempty]).sum() / len(y))
    return {"rows": len(y), "target_rate": float(y.mean()), "mean_prediction": float(p.mean()), "log_loss": float(point_loss(y, p).mean()), "brier": float(np.mean((p.astype("float64") - y) ** 2)), "ece_10_bins": ece, "calibration_bins": [{"bin": k, "rows": int(counts[k]), "target_rate": float(positives[k] / counts[k]), "mean_prediction": float(p_sums[k] / counts[k])} for k in range(10) if counts[k]]}


def paired_bootstrap(data, difference, mask, seed=20261015):
    u, d = data["user_id"][mask], difference[mask]
    if not len(d):
        return {"rows": 0}
    _, inverse = np.unique(u, return_inverse=True)
    sums = np.bincount(inverse, weights=d)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(500):
        indices = rng.integers(0, len(sums), size=len(sums))
        estimates.append(float(sums[indices].sum() / counts[indices].sum()))
    low, high = np.quantile(estimates, [.025, .975])
    return {"rows": len(d), "users": len(sums), "mean_loss_difference": float(d.mean()), "user_cluster_bootstrap_95_interval": [float(low), float(high)], "replicates": 500, "scope": "conditional on observed period and selected models; no temporal-generalization or multiple-comparison guarantee"}


def main():
    started = time.perf_counter()
    source = np.load(ROOT / "audit_arrays.npz")
    splits = {name: {column: source[name + "__" + column] for column in ["user_id", "video_id", "tab", "hour_bin", "duration_ms", "long_view"]} for name in ["train", "validation", "test_standard", "test_random"]}
    train, validation = splits["train"], splits["validation"]
    counts = CountTables(train)
    fitted = {alpha: counts.fit(alpha) for alpha in ALPHAS}
    best, candidates = {}, []
    for alpha in ALPHAS:
        basic, base, effects = counts.predictions(validation, fitted[alpha])
        variants = [(name, None, p) for name, p in basic.items()]
        variants.extend((name, weight, sigmoid(base + weight * effect)) for name, effect in effects.items() for weight in WEIGHTS)
        for name, weight, p in variants:
            score = float(point_loss(validation["long_view"], p).mean())
            config = {"model": name, "alpha": alpha, "weight": weight, "validation_log_loss": score}
            candidates.append(config)
            if name not in best or score < best[name]["validation_log_loss"]:
                best[name] = config
    save("validation_selection.json", {"selection_rule": "minimum validation log loss; stable candidate-order tie break", "best": best, "candidates": candidates, "saved_before_evaluation": True})
    selection_hash = hashlib.sha256((ROOT / "validation_selection.json").read_bytes()).hexdigest()
    rows, detailed, contrasts = [], {}, {}
    models_dir = ROOT / "count_models"
    models_dir.mkdir(exist_ok=True)
    for name, config in best.items():
        f = fitted[config["alpha"]]
        arrays = {"context": f["context"]}
        if name == "global_rate":
            arrays = {"global": np.asarray([counts.global_p], dtype="float32")}
        if name == "item_context_rate" or name.startswith("item_plus_"):
            arrays["item"] = f["item"]
        if name == "user_context_rate" or name == "item_plus_user_context":
            arrays["user"] = f["user"]
        if name == "item_plus_user_marginal":
            arrays["user_marginal"] = f["marginal"]
        if "random_shared" in name:
            arrays["group"] = f["groups"][int(name.split("_s")[-1])]
        np.savez_compressed(models_dir / (name + ".npz"), **arrays)
        best[name]["fitted_probability_array_bytes"] = int(sum(x.nbytes for x in arrays.values()))
    for split_name in ["test_standard", "test_random"]:
        data = splits[split_name]
        u, i = data["user_id"], data["video_id"]
        warm_u = np.zeros(len(u), dtype=bool)
        warm_i = np.zeros(len(i), dtype=bool)
        valid_u, valid_i = u < len(counts.mn), i < len(counts.in_)
        warm_u[valid_u] = counts.mn[u[valid_u]] > 0
        warm_i[valid_i] = counts.in_.sum(axis=1)[i[valid_i]] > 0
        warm = warm_u & warm_i
        losses, predictions = {}, {}
        for name, config in best.items():
            basic, base, effects = counts.predictions(data, fitted[config["alpha"]])
            p = basic[name] if name in basic else sigmoid(base + config["weight"] * effects[name])
            if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
                raise RuntimeError("Invalid prediction probabilities")
            predictions[name] = p
            losses[name] = point_loss(data["long_view"], p)
            m = {"all": metrics(data, p), "warm_user_item": metrics(data, p, warm), "positive_duration": metrics(data, p, data["duration_ms"] > 0), "by_context": {str(c): metrics(data, p, data["tab"] == c) for c in np.unique(data["tab"])}}
            detailed[split_name + "/" + name] = m
            rows.append({"split": split_name, "model": name, "alpha": config["alpha"], "weight": config["weight"], "validation_log_loss": config["validation_log_loss"], "log_loss": m["all"]["log_loss"], "brier": m["all"]["brier"], "ece": m["all"]["ece_10_bins"], "warm_log_loss": m["warm_user_item"].get("log_loss"), "probability_array_bytes": config["fitted_probability_array_bytes"]})
        ref = losses["item_plus_user_context"]
        marginal = losses["item_plus_user_marginal"]
        random_mean_loss = np.mean([losses[f"item_plus_random_shared_s{s}"] for s in SEEDS], axis=0)
        contrasts[split_name] = {
            "item_only_minus_individual_context": paired_bootstrap(data, losses["item_context_rate"] - ref, warm),
            "marginal_user_minus_contextual_user": paired_bootstrap(data, marginal - ref, warm),
            "mean_random_assignment_loss_minus_individual_context": paired_bootstrap(data, random_mean_loss - ref, warm),
            "difference_sign": "positive means lower log loss for item_plus_user_context",
        }
        np.savez_compressed(ROOT / ("predictions_" + split_name + ".npz"), **predictions)
        print(json.dumps({"evaluated": split_name, "rows": len(data["long_view"])}), flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "baseline_metrics.csv", index=False)
    result = {
        "status": "exploratory count-based component evidence only",
        "validation_selection_sha256": selection_hash,
        "protocol_sha256": hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest(),
        "baseline_details_sha256": hashlib.sha256((ROOT / "baseline_details.json").read_bytes()).hexdigest(),
        "selected_models": best, "metrics": detailed, "paired_contrasts": contrasts,
        "training_count_buffer_bytes": counts.training_count_bytes,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": ["Native target disagrees with the documented duration rule on some records", "Duplicate-key groups were quarantined and splits use canonical UTC event time", "Different probability-table capacities; not an equal-byte neural compression comparison", "No GraphHash, CCE, BACO or proposed embedding method was fitted", "Random exposure is reported separately and is not identified as pure context-frequency shift", "Contexts and feedback under logging policies are not intrinsic preferences or welfare", "Intervals condition on this evaluation period and model-selection choices"],
    }
    save("baseline_results.json", result)
    print(pd.DataFrame(rows).to_json(orient="records"), flush=True)
    print(json.dumps({"paired_contrasts": contrasts, "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
