"""E7/E9, local-only Stage 1--2 follow-up. No API, training, or mapping construction.

Legacy weights are deliberately NEVER scored on this CPU process. Their existing
GPU controls/predictions are reusable evidence; absent validation/retuning stays
missing. Modern checkpoints must carry aliases identical to saved mapping files.
All metrics are fractions internally. Deltas marked ``_pp`` are percentage points.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "experiments/2026-10-16-loss-sharing-repro/_work/datasets"
BETAS = (0., .01, .02, .05, .1, .2, .5, 1., 2., 5.)
THRESHOLD_PP = .02


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for b in iter(lambda: stream.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def array_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x, dtype="<i8").tobytes()).hexdigest()


def load_saved_mapping(path, expected_array_hash=None):
    arr = np.load(path, allow_pickle=False)
    if expected_array_hash and array_hash(arr) != expected_array_hash:
        raise ValueError(f"Saved mapping array hash mismatch: {Path(path).name}")
    if arr.dtype.kind not in "iu" or arr.ndim != 1:
        raise ValueError("This analysis supports stored 1-D GraphHash mappings only")
    return np.asarray(arr, dtype=np.int64)


def train_counts(train_edges, n_items):
    return np.bincount(np.asarray(train_edges)[1], minlength=n_items).astype(np.int64)


def quantize_u8(p):
    """256-level zero-offset prior; one uint8/item plus one float32 scale."""
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Quantizer requires a finite, nonnegative prior vector")
    pmax = float(p.max(initial=0.))
    q = np.zeros(len(p), dtype=np.uint8) if not pmax else np.rint(255 * p / pmax).astype(np.uint8)
    scale = np.float32(pmax / 255.)
    decoded = q.astype(np.float32) * scale
    return q, decoded, {"p_max": pmax, "scale_float32": float(scale),
                        "max_abs_error": float(np.abs(decoded - p).max(initial=0.)),
                        "payload_bytes": int(q.nbytes), "scale_bytes": 4,
                        "total_bytes_including_scale": int(q.nbytes + 4),
                        "float32_prior_bytes": int(4 * len(p)),
                        "n_levels_used": int(len(np.unique(q)))}


class SplitIndex:
    """Deduplicated truth membership and the E9 (uncapped) weighted denominator."""
    def __init__(self, edges, n_users, n_items, weights):
        self.edges = np.asarray(edges, dtype=np.int64)
        self.nu, self.ni = n_users, n_items
        self.weights = np.asarray(weights, dtype=np.float64)
        codes = self.edges[0] * n_items + self.edges[1]
        self.codes = np.unique(codes)
        if len(codes) != len(self.codes):
            raise ValueError("Duplicate truth pairs require an explicit evaluation contract")
        self.npos = np.bincount(self.edges[0], minlength=n_users)
        self.denom = np.bincount(self.edges[0], weights=self.weights[self.edges[1]], minlength=n_users)

    def metrics(self, users, top):
        users, top = np.asarray(users, dtype=np.int64), np.asarray(top, dtype=np.int64)
        codes = users[:, None] * self.ni + top
        pos = np.searchsorted(self.codes, codes)
        hits = (pos < len(self.codes)) & (self.codes[np.minimum(pos, len(self.codes) - 1)] == codes)
        counts = hits.sum(1)
        weighted_hits = (hits * self.weights[top]).sum(1)
        return {"n_users": len(users),
                "weighted_recall": float(np.mean(weighted_hits / self.denom[users])),
                "standard_recall": float(np.mean(counts / self.npos[users])),
                "capped_recall20": float(np.mean(counts / np.minimum(20, self.npos[users])))}


def select_beta(val_curve):
    return max(sorted(val_curve), key=lambda b: val_curve[b]["weighted_recall"])


def lex_topk(scores, secondary_rank, k=20):
    """Exact lexicographic reference for small tests; production modern runs use torch.topk."""
    return np.stack([np.lexsort((secondary_rank, -row))[:k] for row in scores])


def load_dataset(name, data_root=DATA):
    directory = Path(data_root) / f"{name}-processed"
    parts = {s: torch.load(directory / f"{s}.pt", weights_only=True, map_location="cpu").numpy()
             for s in ("train", "val", "test")}
    nu = max(int(x[0].max()) for x in parts.values()) + 1
    ni = max(int(x[1].max()) for x in parts.values()) + 1
    counts = train_counts(parts["train"], ni)
    p = np.log1p(counts.astype(np.float64))
    weights = (1. + counts) ** -.5
    indices = {s: SplitIndex(parts[s], nu, ni, weights) for s in ("val", "test")}
    masks = {}
    for split, edges in (("val", parts["train"]),
                         ("test", np.concatenate((parts["train"], parts["val"]), axis=1))):
        order = np.argsort(edges[0], kind="stable")
        counts_u = np.bincount(edges[0], minlength=nu)
        ptr = np.concatenate(([0], np.cumsum(counts_u)))
        masks[split] = (ptr, edges[1, order].astype(np.int64))
    return {"name": name, "nu": nu, "ni": ni, "p": p, "counts": counts,
            "indices": indices, "masks": masks,
            "split_sha256": {s: file_hash(directory / f"{s}.pt") for s in parts}}


def cached_metrics(path, data):
    with np.load(path, allow_pickle=False) as z:
        users, top = z["users"].copy(), z["top20"].copy()
    m = data["indices"]["test"].metrics(users, top)
    m.update({"source": str(Path(path).relative_to(ROOT)), "sha256": file_hash(path),
              "execution": "fresh local metric on saved GPU top20 (no new ranking)"})
    return m


def seed_eval(run_dir):
    d = json.loads((Path(run_dir) / "metrics.json").read_text())
    return next(iter(d["seeds"].values()))["eval"]


def load_modern_vectors(run_dir, data):
    """No legacy fallback: the checkpoint MUST include the actual stored aliases."""
    run_dir = Path(run_dir)
    obj = torch.load(modern_checkpoint_path(run_dir), map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or not obj.get("aliases") or not isinstance(obj.get("model"), dict):
        raise ValueError("Legacy checkpoints may only be ranked on the GPU image")
    config = json.loads((run_dir / "config.json").read_text())
    aliases = []
    for j, side in enumerate(("users", "items")):
        path = run_dir / f"mapping_{side}.npy"
        if file_hash(path) != config["mapping_sha256"][side]:
            raise ValueError(f"Stored {side} mapping file hash mismatch")
        arr = load_saved_mapping(path)
        embedded = obj["aliases"][j]
        embedded = embedded.numpy() if isinstance(embedded, torch.Tensor) else np.asarray(embedded)
        if not np.array_equal(arr, embedded):
            raise ValueError(f"Saved versus checkpoint {side} aliases differ")
        aliases.append(arr)
    if len(aliases[0]) != data["nu"] or len(aliases[1]) != data["ni"]:
        raise ValueError("Mapping versus dataset dimensions differ")
    def find(suffix):
        keys = [k for k in obj["model"] if k.endswith(suffix)]
        if len(keys) != 1:
            raise ValueError(f"Unexpected model keys for {suffix}")
        return obj["model"][keys[0]].float()
    u = find("user_emb_table.weight")[torch.from_numpy(aliases[0])]
    i = find("item_emb_table.weight")[torch.from_numpy(aliases[1])]
    return u, i


@torch.no_grad()
def score_modern(uvec, ivec, data, split, betas, mode, *, quant_beta=None, chunk=192, deadline=None):
    """One CPU catalog pass. Validation can sweep; test receives only its frozen beta.

    Arithmetic/tie policy follows exp_run.sweep: float32 expanded-item dot products,
    torch.topk, validation excludes train, test excludes train+val. No clustering.
    """
    if mode == "cos":
        uvec, ivec = F.normalize(uvec, dim=1), F.normalize(ivec, dim=1)
    truth = data["indices"][split]
    users = np.flatnonzero(truth.npos)
    p = torch.tensor(data["p"], dtype=torch.float32)
    cells = [(f"beta:{b:g}", b * p) for b in betas]
    if quant_beta is not None:
        _, pu8, _ = quantize_u8(data["p"])
        cells += [("quantized_original", quant_beta * torch.from_numpy(pu8))]
    ptr, cols = data["masks"][split]
    sums = {key: {"weighted_recall": 0., "standard_recall": 0., "capped_recall20": 0.}
            for key, _ in cells}
    t0 = time.perf_counter()
    for start in range(0, len(users), chunk):
        if deadline and time.perf_counter() > deadline:
            raise TimeoutError("Local per-case scoring cap reached; results stay explicitly partial")
        us = users[start:start + chunk]
        s = uvec[torch.from_numpy(us)] @ ivec.T
        lengths = ptr[us + 1] - ptr[us]
        mr = np.repeat(np.arange(len(us)), lengths)
        mc = np.concatenate([cols[ptr[x]:ptr[x + 1]] for x in us])
        s[torch.from_numpy(mr), torch.from_numpy(mc)] = -1e9
        for key, additive in cells:
            top = torch.topk(s + additive, min(20, data["ni"]), dim=1).indices.numpy()
            m = truth.metrics(us, top)
            for metric in sums[key]:
                sums[key][metric] += m[metric] * len(us)
        if start == 0 or (start // chunk) % 50 == 0:
            print(json.dumps({"event": "score_progress", "dataset": data["name"], "split": split,
                              "users": start + len(us), "of": len(users), "cells": len(cells),
                              "seconds": round(time.perf_counter() - t0, 1)}), flush=True)
    return {key: {**{metric: value / len(users) for metric, value in sums[key].items()},
                  "n_users": len(users)} for key, _ in cells}


def reused_e7_controls():
    result = []
    for path in sorted((ROOT / "runs/_sessions").glob("*/controls/*/controls.json")):
        c = json.loads(path.read_text())
        meta, base = c["meta"], c["meta"]["base"]
        prior, u8 = c["full_prior"], c[f"{base}_prior_u8"]
        f, q = prior["test"]["recall"], u8["test"]["recall"]
        pairs = [100 * (a - b) for a, b in zip(q["per_seed"], f["per_seed"])]
        primary = meta["method"] == "GraphHash" and base == "cos" and "directau" in meta["run_id"]
        result.append({"run_id": meta["run_id"], "dataset": meta["dataset"], "base": base,
                       "registered_graphhash_directau": primary,
                       "execution": "reused completed GPU measurement; independently threshold-checked locally",
                       "test": {"float_recall20": f["mean"], "u8_recall20": q["mean"],
                                "delta_pp": 100 * (q["mean"] - f["mean"]),
                                "mean_pass": abs(100 * (q["mean"] - f["mean"])) <= THRESHOLD_PP,
                                "paired_tie_draw_delta_pp": pairs,
                                "all_tie_draws_pass": max(abs(x) for x in pairs) <= THRESHOLD_PP},
                       "validation": None, "validation_status": "not saved for quantized GPU cell; GPU re-score needed",
                       "beta": prior["best_beta"], "source": str(path.relative_to(ROOT)),
                       "source_sha256": file_hash(path), "storage": {"payload_bytes": meta["n_items"],
                       "scale_bytes": 4, "total_bytes_including_scale": meta["n_items"] + 4},
                       "gpu_quantizer_note": "GPU used pmax metadata; fresh local format below stores one float32 scale"})
    return result


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def refresh_test_comparators(args):
    """Reuse already frozen validation sweeps; only add missing same-CPU test controls."""
    e7path, e9path = ROOT / "results/E7_stage12.json", ROOT / "results/E9_stage12.json"
    e7, e9 = json.loads(e7path.read_text()), json.loads(e9path.read_text())
    datasets, start_all = {}, time.perf_counter()
    for case in e9["cases"]:
        if case.get("retuning_status") != "complete local modern checkpoint":
            continue
        started = time.perf_counter()
        name = case["dataset"]
        if name not in datasets:
            datasets[name] = load_dataset(name, args.data_root)
        d = datasets[name]
        directory = ROOT / "runs" / case["run_id"]
        uv, iv = load_modern_vectors(directory, d)
        case["checkpoint_sha256"] = file_hash(directory / "best_seed0.pt")
        case["mapping_file_sha256"] = {side: file_hash(directory / f"mapping_{side}.npy")
                                      for side in ("users", "items")}
        beta, chosen, base = case["original_beta"], case["retuned_beta"], case["base"]
        test = score_modern(uv, iv, d, "test", sorted({0., beta, chosen}), base,
                           quant_beta=beta if base == "cos" else None, chunk=args.chunk,
                           deadline=started + args.case_cap_seconds)
        case["retuned_beta_test"] = test[f"beta:{chosen:g}"]
        case["same_policy_original_beta_test"] = test[f"beta:{beta:g}"]
        case["same_policy_beta0_test"] = test["beta:0"]
        sensitivity = 100 * (test[f"beta:{beta:g}"]["capped_recall20"] -
                             case["original_beta_test"]["capped_recall20"])
        case["cached_gpu_vs_cpu_original_delta_recall20_pp"] = sensitivity
        case["retuning_gain_vs_same_policy_original_pp"] = 100 * (
            case["retuned_beta_test"]["weighted_recall"] - case["same_policy_original_beta_test"]["weighted_recall"])
        case["test_selection_note"] = "Original beta, frozen validation-chosen beta, beta=0 controls; no test sweep/selection"
        extra = time.perf_counter() - started
        case["same_policy_comparator_wall_seconds"] = extra
        case["wall_seconds"] += extra
        for qcase in e7["fresh_modern_checks"]:
            if qcase["run_id"] != case["run_id"]:
                continue
            f, q = test[f"beta:{beta:g}"]["capped_recall20"], test["quantized_original"]["capped_recall20"]
            qcase["test"] = {"float_recall20": f, "u8_recall20": q,
                             "delta_pp": 100 * (q - f), "pass": abs(100 * (q - f)) <= THRESHOLD_PP}
            qcase["float_test_source"] = "fresh CPU float and u8 under identical arithmetic and torch.topk policy"
            qcase["cached_gpu_vs_cpu_original_delta_recall20_pp"] = sensitivity
            qcase["same_policy_comparator_wall_seconds"] = extra
            qcase["wall_seconds"] += extra
        print(json.dumps({"event": "same_policy_test_complete", "run_id": case["run_id"],
                          "seconds": round(extra, 2), "gpu_cpu_sensitivity_pp": sensitivity}), flush=True)
    elapsed = time.perf_counter() - start_all
    for doc in (e7, e9):
        doc["wall_seconds"] += elapsed
        doc["same_policy_comparator_refresh_seconds"] = elapsed
        doc["provenance"]["script_sha256"] = file_hash(__file__)
    atomic_json(e7path, e7)
    atomic_json(e9path, e9)
    print(json.dumps({"event": "same_policy_refresh_finished", "seconds": round(elapsed, 2)}), flush=True)


def modern_checkpoint_path(run_dir):
    directory = Path(run_dir)
    seeds = json.loads((directory / "metrics.json").read_text())["seeds"]
    if len(seeds) != 1:
        raise ValueError("A final run directory must identify exactly one training seed")
    path = directory / f"best_seed{next(iter(seeds))}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def final_modern_specs():
    rows = []
    for name, dataset in [("Gowalla", "gowalla"), ("Yelp2018", "yelp2018"), ("AmazonBook", "amazonbook")]:
        for seed in ((0, 1, 2) if dataset == "gowalla" else (0,)):
            rows.append((name, f"{dataset}__mf__graphhash__directau__s{seed}", "cos", "cos_prior", "cosine"))
            if dataset != "amazonbook":
                suffix = "-p500" if dataset == "gowalla" and seed == 0 else ""
                rows.append((name, f"{dataset}__mf__graphhash__bpr__s{seed}{suffix}", "raw", "raw_prior", "raw"))
        if dataset != "yelp2018":
            rows.append((name, f"{dataset}__mf__graphhash__directau__s0-initmatch", "cos", "cos_prior", "cosine"))
    return rows


def final_precision_verdict(records):
    for row in records:
        if any(row.get(key) is not None and abs(row[key]) > THRESHOLD_PP
               for key in ("val_delta_pp", "test_delta_pp")):
            return "FAIL"
    if not records or any(r.get("analysis_status") != "complete" for r in records):
        return "INCONCLUSIVE"
    return "PASS"


def run_final(args):
    """Final registered roster. Legacy GPU-only cells remain explicitly missing."""
    t_all = time.perf_counter()
    provenance = {"script_sha256": file_hash(__file__), "no_network": True, "no_training": True,
                  "no_mapping_recomputation": True, "legacy_weights_scored_locally": False}
    e7 = {"schema": "final-e7/2", "threshold_pp": THRESHOLD_PP, "records": [],
          "expected_checkpoints": 12, "provenance": provenance}
    e9 = {"schema": "final-e9/2", "records": [], "comparisons": [], "provenance": provenance,
          "beta_grid": list(BETAS), "denominator_note": "Sum of ALL held-out truth weights per user; equal weights recover STANDARD recall, not main capped recall. Not IPS.",
          "selection": "Weighted VALIDATION alone selects beta; test evaluates frozen original/selected/zero and quantization controls only."}
    cache = {}
    def data(name):
        if name not in cache:
            cache[name] = load_dataset(name, args.data_root)
        return cache[name]
    def save():
        e7["verdict"] = final_precision_verdict(e7["records"])
        e7["one_byte_claim_allowed"] = (len(e7["records"]) == 12 and e7["verdict"] == "PASS")
        e7["wall_seconds"] = e9["wall_seconds"] = time.perf_counter() - t_all
        atomic_json(ROOT / "results/E7_final.json", e7)
        atomic_json(ROOT / "results/E9_final.json", e9)
    for name, rid, base, mode, base_mode in final_modern_specs():
        started = time.perf_counter()
        directory, d = ROOT / "runs" / rid, data(name)
        ev = seed_eval(directory)
        beta = float(ev[mode]["best_beta"])
        sensitivity_only = rid.endswith("-initmatch")
        meta = next(iter(json.loads((directory / "metrics.json").read_text())["seeds"].values()))
        _, _, storage = quantize_u8(d["p"])
        table_bytes = int(meta["params"]) * 4
        common = {"run_id": rid, "dataset": name, "scoring_mode": mode, "original_beta": beta,
                  "source_policy": "CPU float32 expanded-item matrix and torch.topk; identical float/u8 policy",
                  "checkpoint_sha256": file_hash(modern_checkpoint_path(directory)),
                  "split_sha256": d["split_sha256"]}
        qr = {**common, "analysis_status": "incomplete", "float32_prior_bytes": storage["float32_prior_bytes"],
              "table_bytes": table_bytes, "float32_overhead_pct": 100 * storage["float32_prior_bytes"] / table_bytes,
              "uint8_payload_bytes": storage["payload_bytes"], "uint8_scale_bytes": 4,
              "uint8_total_bytes": storage["total_bytes_including_scale"]}
        wr = {**common, "analysis_status": "incomplete"}
        try:
            uv, iv = load_modern_vectors(directory, d)
            deadline = min(started + args.case_cap_seconds, t_all + 600.)
            val_betas = (beta,) if sensitivity_only else BETAS
            val = score_modern(uv, iv, d, "val", val_betas, base, quant_beta=beta,
                               chunk=args.chunk, deadline=deadline)
            curve = {b: val[f"beta:{b:g}"] for b in val_betas}
            chosen = None if sensitivity_only else select_beta(curve)
            test_betas = [beta] if sensitivity_only else sorted({0., beta, chosen})
            test = score_modern(uv, iv, d, "test", test_betas, base, quant_beta=beta,
                                chunk=args.chunk, deadline=deadline)
            for split, scores in [("val", val), ("test", test)]:
                f = scores[f"beta:{beta:g}"]["capped_recall20"]
                q = scores["quantized_original"]["capped_recall20"]
                qr[split] = {"float_recall20": f, "u8_recall20": q, "delta_pp": 100*(q-f)}
                qr[f"{split}_delta_pp"] = 100*(q-f)
            qr["analysis_status"] = "complete"
            qr["pass"] = final_precision_verdict([qr]) == "PASS"
            qr["cached_gpu_test_recall20"] = cached_metrics(directory / f"per_user_test__{mode}.npz", d)["capped_recall20"]
            qr["cached_gpu_vs_cpu_delta_pp"] = 100*(qr["test"]["float_recall20"]-qr["cached_gpu_test_recall20"])
            if not sensitivity_only:
                wr.update({"analysis_status": "complete", "original_beta_test": test[f"beta:{beta:g}"],
                    "retuned_beta": chosen, "retuned_beta_test": test[f"beta:{chosen:g}"],
                    "beta0_test": test["beta:0"], "weighted_validation_by_beta": curve,
                    "cached_original_beta_test": cached_metrics(directory / f"per_user_test__{mode}.npz", d),
                    "beta_shrank_or_equal": chosen <= beta})
        except (FileNotFoundError, ValueError, RuntimeError, TimeoutError) as error:
            qr["reason"] = wr["reason"] = str(error)
        qr["wall_seconds"] = time.perf_counter()-started
        e7["records"].append(qr)
        if not sensitivity_only:
            e9["records"].append(wr)
        save()
        print(json.dumps({"event": "final_case_done", "run_id": rid, "precision_status": qr["analysis_status"],
                          "val_delta_pp": qr.get("val_delta_pp"), "test_delta_pp": qr.get("test_delta_pp"),
                          "weighted_beta": wr.get("retuned_beta"), "seconds": round(qr["wall_seconds"], 2)}), flush=True)
    # Reuse stored GPU predictions only; this legacy checkpoint is never loaded/ranked locally.
    rid = "amazonbook__mf__graphhash__bpr__s0-legacy"
    folder = ROOT / "runs/_sessions/pc-x2-1017/controls" / rid
    controls = json.loads((folder / "controls.json").read_text())
    d = data("AmazonBook")
    beta = float(controls["full_prior"]["best_beta"])
    reused = next(x for x in reused_e7_controls() if x["run_id"] == rid)
    tables = controls["meta"]["table_rows"]
    table_bytes = 4 * 64 * sum(tables)
    e7["records"].append({"run_id": rid, "dataset": "AmazonBook", "scoring_mode": "raw_prior",
        "analysis_status": "missing_gpu", "source_policy": "Saved GPU bucket-exact random-key controls; quantized validation missing",
        "original_beta": beta, "val_delta_pp": None, "test_delta_pp": reused["test"]["delta_pp"],
        "float32_prior_bytes": 4*d["ni"], "table_bytes": table_bytes,
        "float32_overhead_pct": 100*4*d["ni"]/table_bytes,
        "uint8_payload_bytes": d["ni"], "uint8_scale_bytes": 4,
        "reason": "Legacy quantized validation requires GPU rescoring; no legacy local ranking"})
    e9["records"].append({"run_id": rid, "dataset": "AmazonBook", "scoring_mode": "raw_prior",
        "analysis_status": "missing_gpu", "source_policy": "Saved GPU bucket-exact random-key seed0",
        "original_beta": beta, "original_beta_test": cached_metrics(folder / "per_user__full_prior_s0.npz", d),
        "beta0_test": cached_metrics(folder / "per_user__base_randomkey_s0.npz", d),
        "retuned_beta": None, "retuned_beta_test": None,
        "reason": "Weighted-validation sweep and frozen selected test require GPU rescoring"})
    by = {r["run_id"]: r for r in e9["records"]}
    pairs = [("Gowalla", seed, f"gowalla__mf__graphhash__directau__s{seed}",
              f"gowalla__mf__graphhash__bpr__s{seed}"+("-p500" if seed == 0 else "")) for seed in (0,1,2)]
    pairs += [("Yelp2018", 0, "yelp2018__mf__graphhash__directau__s0", "yelp2018__mf__graphhash__bpr__s0"),
              ("AmazonBook", 0, "amazonbook__mf__graphhash__directau__s0", rid)]
    for name, seed, a, b in pairs:
        aa, bb = by[a], by[b]
        complete = aa["analysis_status"] == bb["analysis_status"] == "complete"
        if complete:
            orig_a, orig_b = aa["original_beta_test"], bb["original_beta_test"]
        else:
            orig_a = aa.get("cached_original_beta_test", aa.get("original_beta_test"))
            orig_b = bb.get("cached_original_beta_test", bb.get("original_beta_test"))
        original = None if orig_a is None or orig_b is None else 100*(orig_a["weighted_recall"]-orig_b["weighted_recall"])
        delta = 100*(aa["retuned_beta_test"]["weighted_recall"]-bb["retuned_beta_test"]["weighted_recall"]) if complete else None
        e9["comparisons"].append({"dataset": name, "seed": seed, "directau_run": a, "bpr_run": b,
            "original_delta_pp": original, "retuned_delta_pp": delta,
            "verdict": "PASS" if delta is not None and delta > 0 else "FAIL" if delta is not None else "INCONCLUSIVE",
            "policy_note": "same CPU scoring policy" if complete else "original saved GPU policies; fully retuned comparison missing"})
    e9["verdict_by_dataset"] = {name: ("INCONCLUSIVE" if any(r["verdict"]=="INCONCLUSIVE" for r in e9["comparisons"] if r["dataset"]==name)
        else "PASS" if all(r["verdict"]=="PASS" for r in e9["comparisons"] if r["dataset"]==name) else "FAIL")
        for name in ("Gowalla","Yelp2018","AmazonBook")}
    save()
    print(json.dumps({"event":"final_complete","E7":e7["verdict"],"E9":e9["verdict_by_dataset"],
                      "seconds":round(e7["wall_seconds"],2)}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-cap-seconds", type=float, default=360.)
    parser.add_argument("--chunk", type=int, default=192)
    parser.add_argument("--cached-only", action="store_true")
    parser.add_argument("--final", action="store_true", help="Run the registered final GraphHash roster")
    parser.add_argument("--refresh-test-comparators", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DATA)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.final:
        run_final(args)
        return
    if args.refresh_test_comparators:
        refresh_test_comparators(args)
        return
    started = time.perf_counter()
    dataset_cache = {}
    def data(name):
        if name not in dataset_cache:
            dataset_cache[name] = load_dataset(name, args.data_root)
        return dataset_cache[name]
    e7 = {"schema": "local-e7/1", "status": "partial (Stage 1-2 runs)", "threshold_pp": THRESHOLD_PP,
          "reused_gpu_controls": reused_e7_controls(), "fresh_modern_checks": [],
          "claim_status": "No unconditional one-byte pass: registered quantized validation cells missing; inspect failures",
          "provenance": {"script_sha256": file_hash(__file__), "no_network": True,
                         "no_training": True, "no_mapping_recomputation": True}}
    e9 = {"schema": "local-e9/1", "status": "partial (Stage 1-2 runs)",
          "metric": "mean_u sum_{i in Top20 intersect truth_u}(1+c_i)^-0.5 / sum_{i in truth_u}(1+c_i)^-0.5",
          "denominator_note": "Equal weights recover STANDARD recall, not capped Recall@20 used by main tables. Not IPS.",
          "selection": "original beta from original unweighted validation; retuned beta from weighted VALIDATION only; ties choose smaller beta",
          "beta_grid": list(BETAS), "cases": [], "provenance": dict(e7["provenance"])}
    modern = [("Gowalla", "gowalla__mf__graphhash__directau__s0", "cos", "cos_prior", "cosine"),
              ("Yelp2018", "yelp2018__mf__graphhash__directau__s0", "cos", "cos_prior", "cosine"),
              ("Yelp2018", "yelp2018__mf__graphhash__bpr__s0", "raw", "raw_prior", "raw")]
    for name, run_id, base, mode, base_mode in modern:
        t0 = time.perf_counter()
        directory, d = ROOT / "runs" / run_id, data(name)
        ev = seed_eval(directory)
        beta = float(ev[mode]["best_beta"])
        case = {"run_id": run_id, "dataset": name, "base": base, "original_beta": beta,
                "original_beta_test": cached_metrics(directory / f"per_user_test__{mode}.npz", d),
                "beta0_test": cached_metrics(directory / f"per_user_test__{base_mode}.npz", d),
                "ranking_policy": "modern runner float32 expanded-item matrix with torch.topk",
                "split_sha256": d["split_sha256"]}
        try:
            if args.cached_only:
                raise RuntimeError("cached-only invocation; validation retuning not requested")
            uv, iv = load_modern_vectors(directory, d)
            case["checkpoint_sha256"] = file_hash(directory / "best_seed0.pt")
            case["mapping_file_sha256"] = {side: file_hash(directory / f"mapping_{side}.npy")
                                          for side in ("users", "items")}
            deadline = t0 + args.case_cap_seconds
            val = score_modern(uv, iv, d, "val", BETAS, base,
                               quant_beta=beta if base == "cos" else None,
                               chunk=args.chunk, deadline=deadline)
            curve = {b: val[f"beta:{b:g}"] for b in BETAS}
            chosen = select_beta(curve)
            case.update({"retuned_beta": chosen, "weighted_validation_by_beta": curve,
                         "beta_shrank_or_equal": chosen <= beta})
            # Frozen original/selected/zero cells are scored once each under the
            # same CPU arithmetic. Cached GPU rankings remain a separate diagnostic:
            # CPU/GPU topk tie differences must not be called quantization effects
            # or attributed solely to a change of beta.
            test_betas = sorted({0., beta, chosen})
            test = score_modern(uv, iv, d, "test", test_betas, base,
                                quant_beta=beta if base == "cos" else None,
                                chunk=args.chunk, deadline=deadline) if test_betas or base == "cos" else {}
            case["retuned_beta_test"] = test[f"beta:{chosen:g}"]
            case["same_policy_original_beta_test"] = test[f"beta:{beta:g}"]
            case["same_policy_beta0_test"] = test["beta:0"]
            case["cached_gpu_vs_cpu_original_delta_recall20_pp"] = 100 * (
                test[f"beta:{beta:g}"]["capped_recall20"] - case["original_beta_test"]["capped_recall20"])
            case["retuning_gain_vs_same_policy_original_pp"] = 100 * (
                case["retuned_beta_test"]["weighted_recall"] - case["same_policy_original_beta_test"]["weighted_recall"])
            case["test_selection_note"] = "Only original beta, validation-chosen beta and beta=0 controls; no test sweep/selection"
            case["retuning_status"] = "complete local modern checkpoint"
            if base == "cos":
                _, _, storage = quantize_u8(d["p"])
                full_val = curve[beta]["capped_recall20"]
                qval = val["quantized_original"]["capped_recall20"]
                full_test = test[f"beta:{beta:g}"]["capped_recall20"]
                qtest = test["quantized_original"]["capped_recall20"]
                e7["fresh_modern_checks"].append({"run_id": run_id, "beta": beta,
                    "registered_original_stage1": False, "storage": storage,
                    "checkpoint_sha256": file_hash(directory / "best_seed0.pt"),
                    "val": {"float_recall20": full_val, "u8_recall20": qval,
                            "delta_pp": 100 * (qval - full_val), "pass": abs(100 * (qval - full_val)) <= THRESHOLD_PP},
                    "test": {"float_recall20": full_test, "u8_recall20": qtest,
                             "delta_pp": 100 * (qtest - full_test), "pass": abs(100 * (qtest - full_test)) <= THRESHOLD_PP},
                    "float_test_source": "fresh CPU float and u8 rankings under identical arithmetic and torch.topk policy",
                    "cached_gpu_vs_cpu_original_delta_recall20_pp": case["cached_gpu_vs_cpu_original_delta_recall20_pp"],
                    "selection": "original unweighted-validation beta held fixed", "wall_seconds": time.perf_counter() - t0})
        except (FileNotFoundError, ValueError, RuntimeError, TimeoutError) as error:
            case.update({"retuning_status": "partial", "reason": str(error)})
        case["wall_seconds"] = time.perf_counter() - t0
        e9["cases"].append(case)
        atomic_json(ROOT / "results/E7_stage12.json", e7)
        atomic_json(ROOT / "results/E9_stage12.json", e9)
    legacy_ids = [("Gowalla", "gowalla__mf__graphhash__bpr__s0-legacy"),
                  ("AmazonBook", "amazonbook__mf__graphhash__directau__s0-legacy41"),
                  ("AmazonBook", "amazonbook__mf__graphhash__bpr__s0-legacy")]
    for name, run_id in legacy_ids:
        t0 = time.perf_counter()
        found = list((ROOT / "runs/_sessions").glob(f"*/controls/{run_id}/controls.json"))
        if len(found) != 1:
            e9["cases"].append({"run_id": run_id, "status": "partial", "reason": "Missing/unmatched GPU controls"})
            continue
        path = found[0]
        controls = json.loads(path.read_text())
        d = data(name)
        beta = controls["full_prior"]["best_beta"]
        e9["cases"].append({"run_id": run_id, "dataset": name, "base": controls["meta"]["base"],
            "original_beta": beta,
            "original_beta_test": cached_metrics(path.parent / "per_user__full_prior_s0.npz", d),
            "beta0_test": cached_metrics(path.parent / "per_user__base_randomkey_s0.npz", d),
            "retuning_status": "partial: legacy weights may only be ranked on GPU; weighted validation sweep absent locally",
            "retuned_beta": None, "retuned_beta_test": None,
            "ranking_policy": "existing Stage1 bucket-exact GPU scorer with random-key tie seed 0",
            "split_sha256": d["split_sha256"], "wall_seconds": time.perf_counter() - t0})
    by_id = {x["run_id"]: x for x in e9["cases"]}
    gow_d = by_id["gowalla__mf__graphhash__directau__s0"]
    gow_b = by_id.get("gowalla__mf__graphhash__bpr__s0-legacy", {})
    if "original_beta_test" in gow_b:
        delta = gow_d["original_beta_test"]["weighted_recall"] - gow_b["original_beta_test"]["weighted_recall"]
        e9["gowalla_k2_original_beta"] = {"delta_weighted_recall_pp": 100 * delta,
            "ordering_preserved": delta > 0, "caveat": "modern DAU versus legacy BPR; scoring tie policies differ",
            "retuned_ordering": "not evaluable until original BPR weighted-validation sweep is run on GPU"}
    primary = [x for x in e7["reused_gpu_controls"] if x["registered_graphhash_directau"]]
    e7["registered_graphhash_test_means_all_pass"] = bool(primary) and all(x["test"]["mean_pass"] for x in primary)
    e7["registered_graphhash_every_tie_draw_pass"] = bool(primary) and all(x["test"]["all_tie_draws_pass"] for x in primary)
    e7["all_stage1_control_test_means_pass"] = all(x["test"]["mean_pass"] for x in e7["reused_gpu_controls"])
    e7["wall_seconds"] = e9["wall_seconds"] = time.perf_counter() - started
    atomic_json(ROOT / "results/E7_stage12.json", e7)
    atomic_json(ROOT / "results/E9_stage12.json", e9)
    print(json.dumps({"event": "finished", "seconds": round(e9["wall_seconds"], 2),
                      "modern_cases": [{"run_id": x["run_id"], "status": x.get("retuning_status"),
                      "beta": x.get("retuned_beta")} for x in e9["cases"]],
                      "e7_graphhash_test_mean_pass": e7["registered_graphhash_test_means_all_pass"]}), flush=True)


if __name__ == "__main__":
    main()
