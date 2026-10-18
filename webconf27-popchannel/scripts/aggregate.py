#!/usr/bin/env python
"""Aggregate every run into results/master_metrics.csv (TASKS T5, rule 11).

    python aggregate.py [--out CSV] [--runs DIR] [--controls DIR] [--results DIR]
                        [--legacy-dir DIR] [--repro-dir DIR] [--registry CSV] [--lenient]

Default output: results/master_metrics.csv. A registered legacy file that is missing is an error unless --lenient.

One row per (run, scoring mode). Recall/NDCG are in PERCENT (source JSONs store fractions; the
popularity-only JSONs are already in percent). Inputs:

  1. New runs   ``runs/*/metrics.json`` (and ``runs/_invalid/*/metrics.json``, forced to status invalid).
                 status = ``validate_run`` result: invalid if not ok, capped if stop_reason is time_cap or
                 max_epochs, else valid.
  2. Legacy     files listed in ``legacy_registry.csv`` (format of each file differs; the ``parser`` column
                 selects the reader). Several files may describe one run; they are merged field by field
                 (first value wins, primary files first). A conflict between two files is reported on stderr.
  3. Controls   recursively discovered ``controls.json`` beneath runs/ and controls/, including
                 ``runs/_sessions/<session>/controls/<run_id>/controls.json`` (score_controls.py, schema
                 "score_controls/1"): one row per cell. ``scoring_mode`` = ``controls/`` + the cell's name in the
                 file (e.g. ``controls/tie_keys/random``, ``controls/decomposition_fixed_beta/mu``,
                 ``controls/cos_prior``); the prefix keeps them apart from the run's own ``cos_prior``/``raw_prior``
                 rows. status / comparison_type are inherited from the scored run when it is in the table.
  4. Pop-only   ``results/A6_poponly_*.json`` (scoring_mode ``pop_only``).

Definitions
  * val_gap / test_gap = raw - cosine Recall@20 of the same run (percent points), repeated on each row of the run.
  * rho_item = Spearman(item length, training count). New runs: norm_log.csv (long format: epoch, kind, value)
    at best_epoch (else the last epoch; rho_item falls back to metrics ``spearman_norm_vs_count``). rho_bucket /
    rho_dir come from the same log (empty for legacy runs and for Full models).
  * extra_bytes = bytes of per-item state a mode needs beyond the embedding table: 4 * n_items (float32) for
    cos_prior, raw_prior, cos_b, cos_b_prior and pop_only; 0 otherwise. The uint8 variant is analysis E7.
  * test_std_recall20 = standard deviation (percent) of test Recall@20 over the cell's random-key seeds, as
    stored by score_controls.py (10 seeds). Empty for training-run rows (one seed; use the user bootstrap, E8).
  * stop_reason of legacy 30-epoch runs is ``fixed_epochs`` (the legacy runner wrote null).
  * comparison_type: legacy = reproduction. New runs: config.json ``comparison_type`` if present, else
    controlled iff DirectAU at the PLAN 7.2 controlled lr/wd and (full, or graphhash with init-match), else
    reproduction. Pop-only rows are labelled reproduction (a baseline, not a controlled model pair).
  * resolution: run-id suffix ``-r{N}``, else config, else 200 for graphhash (PLAN 7.2), else empty.
  * session_slug of a new run: config.json ``session_slug``/``slug``, else a one-line ``session_slug.txt`` next to
    metrics.json (the runner does not know the Kaggle slug; the fetch step should write it).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import validate_run as vr  # noqa: E402

PROJECT = HERE.parent
LEGACY_DIR = Path(r"D:\web_2\experiments\2026-10-16-fast-check\results")
REPRO_DIR = Path(r"D:\web_2\experiments\2026-10-16-loss-sharing-repro")
DEFAULT_REGISTRY = HERE / "legacy_registry.csv"
DEFAULT_OUT = PROJECT / "results" / "master_metrics.csv"

COLUMNS = ["run_id", "dataset", "compression", "resolution", "loss", "seed", "init", "scoring_mode", "beta",
           "val_recall20", "val_ndcg20", "test_recall20", "test_ndcg20", "test_std_recall20",
           "rho_item", "rho_bucket", "rho_dir", "val_gap", "test_gap", "best_epoch", "epochs_run",
           "stop_reason", "train_min", "table_params", "extra_bytes", "mapping_sha256", "session_slug",
           "comparison_type", "status"]
MODE_ORDER = ["raw", "cosine", "cos_prior", "raw_prior", "cos_b", "cos_b_prior", "pop_only"]
PRIOR_MODES = {"cos_prior", "raw_prior", "cos_b", "cos_b_prior", "pop_only"}
CELL_KEYS = ("beta", "val_recall20", "val_ndcg20", "test_recall20", "test_ndcg20", "test_std_recall20")
N_ITEMS = {"gowalla": 40981, "yelp2018": 38048, "amazonbook": 91599}  # GraphHash's processed files
CONTROLLED_LRWD = {"gowalla": (0.01, 1e-8), "amazonbook": (0.001, 1e-8)}  # PLAN 7.2 controlled pairs
NAME_MAP = {  # runner / legacy spellings -> run-id spellings
    "dataset": {"gowalla": "gowalla", "yelp2018": "yelp2018", "amazonbook": "amazonbook"},
    "compression": {"full": "full", "graphhash": "graphhash", "random": "random", "dfreq": "dfreq",
                    "doublefreq": "dfreq", "double_freq": "dfreq"},
    "loss": {"dau": "directau", "directau": "directau", "bpr_author": "bpr", "bpr": "bpr",
             "dau_bias": "directau_bias", "directau_bias": "directau_bias"},
}
EMPTY = (None, "")


# ----------------------------------------------------------------------------- numbers
def pct(x):
    """Fraction -> percent (None stays None)."""
    return None if x is None else 100.0 * float(x)


def fmt(x):
    if x is None:
        return ""
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return "" if not math.isfinite(x) else repr(round(x, 6))
    return x


def num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def cell_from_block(block, scale=100.0):
    """{best_beta, val{recall,ndcg}, test{recall,ndcg}} -> cell (percent when scale=100)."""
    def g(split, key):
        v = (block.get(split) or {}).get(key)
        return None if v is None else scale * float(v)
    beta = block.get("best_beta")
    return {"beta": None if beta is None else float(beta), "val_recall20": g("val", "recall"),
            "val_ndcg20": g("val", "ndcg"), "test_recall20": g("test", "recall"), "test_ndcg20": g("test", "ndcg")}


def by_beta(table, beta):
    """Look up a {"0.2": ...} dict whose keys are strings by numeric beta."""
    for key, value in table.items():
        if abs(float(key) - float(beta)) < 1e-12:
            return value
    raise KeyError(beta)


# ----------------------------------------------------------------------------- legacy parsers
# Each parser returns {"meta": {...}, "modes": {mode: cell}}; values in percent, fractions already converted.
def parse_exp(data, run, sub=None):
    row = data["seeds"][str(run["seed"])]
    return {"meta": {"best_epoch": row.get("best_epoch"), "epochs_run": row.get("epochs_run"),
                     "stop_reason": row.get("stopped"), "train_seconds": row.get("train_seconds"),
                     "table_params": row.get("params"), "rho_item": row.get("spearman_norm_vs_count")},
            "modes": {m: cell_from_block(b) for m, b in row["eval"].items()}}


def parse_fast(data, run, sub=None):
    """fast_run.py output. Files without best_epoch (first fast-check runs) were scored at the LAST epoch."""
    val = {int(k): float(v) for k, v in data["val"].items()}
    last = max(val)
    best_epoch = data.get("best_epoch", last)
    best_val = data.get("best_val", val[best_epoch])
    meta = {"best_epoch": best_epoch, "epochs_run": data.get("epochs_run", last),
            "train_seconds": data.get("train_seconds"),
            "table_params": 64 * (data["user_buckets"] + data["item_buckets"])}
    raw = {"beta": 0.0, "val_recall20": pct(best_val), "test_recall20": pct(data["test_recall"]),
           "test_ndcg20": pct(data["test_ndcg"])}
    return {"meta": meta, "modes": {"raw": raw}}


def parse_popprior(data, run, sub):
    """popprior.py / popprior_full.py: cosine at beta 0 and cos_prior at the best beta (GPU image)."""
    block = data[sub]
    grid = block["val_by_beta"]
    best = float(block["best_beta"])
    def c(beta, val, test):
        return {"beta": beta, "val_recall20": pct(val["recall"]), "val_ndcg20": pct(val["ndcg"]),
                "test_recall20": pct(test["recall"]), "test_ndcg20": pct(test["ndcg"])}
    return {"meta": {}, "modes": {
        "cosine": c(0.0, by_beta(grid, 0.0), block["test_beta0"]),
        "cos_prior": c(best, by_beta(grid, best), block["test_best"]),
        "raw": {"beta": 0.0, "test_recall20": pct(block["test_raw_dot"]["recall"]),
                "test_ndcg20": pct(block["test_raw_dot"]["ndcg"])}}}


def parse_mech(data, run, sub):
    """mech.py: rho of a checkpoint, raw test, and (BPR) the raw_prior sweep."""
    block = data[sub]
    modes = {"raw": {"beta": 0.0, "test_recall20": pct(block["test_raw_dot"]["recall"]),
                     "test_ndcg20": pct(block["test_raw_dot"]["ndcg"])}}
    if "best_beta" in block:
        best = float(block["best_beta"])
        val, test = by_beta(block["val_by_beta_rawdot"], best), block["test_best_rawdot"]
        modes["raw_prior"] = {"beta": best, "val_recall20": pct(val["recall"]), "val_ndcg20": pct(val["ndcg"]),
                              "test_recall20": pct(test["recall"]), "test_ndcg20": pct(test["ndcg"])}
    return {"meta": {"rho_item": block.get("spearman_norm_vs_count")}, "modes": modes}


def parse_rescore(data, run, sub=None):
    entry = next(iter(data.values()))
    return {"meta": {}, "modes": {
        "raw": {"beta": 0.0, "test_recall20": pct(entry["raw_recall"]), "test_ndcg20": pct(entry["raw_ndcg"])},
        "cosine": {"beta": 0.0, "test_recall20": pct(entry["cosine_recall"]),
                   "test_ndcg20": pct(entry["cosine_ndcg"])}}}


def parse_p1ext(data, run, sub=None):
    return {"meta": {"rho_item": data.get("spearman_norm_vs_count")},
            "modes": {"raw_prior": cell_from_block(data["raw_prior"])}}


def parse_poponly(data, run, sub=None):
    """Already in percent (poponly.py rounds to 2 decimals)."""
    return {"meta": {"table_params": 0},
            "modes": {"pop_only": {"beta": None, "val_recall20": data["val"]["recall"],
                                   "val_ndcg20": data["val"]["ndcg"], "test_recall20": data["test"]["recall"],
                                   "test_ndcg20": data["test"]["ndcg"]}}}


def parse_repro_csv(path, run, sub=None):
    """2026-10-16-loss-sharing-repro/metrics.csv: the timed-out Full x DirectAU Gowalla pilot."""
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            same = (NAME_MAP["dataset"].get(row["dataset"].lower()) == run["dataset"]
                    and NAME_MAP["compression"].get(row["method"].lower()) == run["compression"]
                    and NAME_MAP["loss"].get(row["loss"].lower()) == run["loss"] and int(row["seed"]) == run["seed"])
            if not same:
                continue
            last = row.get("last_validation_epoch") or row.get("epochs")
            return {"meta": {"best_epoch": int(row["best_epoch"]), "epochs_run": int(last) if last else None,
                             "train_seconds": float(row["wall_seconds"]),
                             "table_params": int(row["parameters"]),
                             "stop_reason": "time_cap" if row.get("status") == "timeout" else None},
                    "modes": {"raw": {"beta": 0.0, "val_recall20": pct(float(row["val_raw_recall"]))}}}
    raise KeyError(f"no matching row for {run['run_id']} in {path}")


PARSERS = {"exp": parse_exp, "fast": parse_fast, "popprior": parse_popprior, "mech": parse_mech,
           "rescore": parse_rescore, "p1ext": parse_p1ext, "poponly": parse_poponly, "repro_csv": parse_repro_csv}


# ----------------------------------------------------------------------------- merging fragments of one run
class RunAccumulator:
    """Merge fragments from several files: first value wins; a different later value is a conflict."""

    def __init__(self, run_id, status, session_slug):
        self.run_id, self.status, self.session_slug = run_id, status, session_slug
        self.meta, self.modes, self.conflicts = {}, {}, []

    def _put(self, store, key, value, where, source):
        if value in EMPTY:
            return
        if key not in store or store[key] in EMPTY:
            store[key] = value
        elif isinstance(value, (int, float)) and isinstance(store[key], (int, float)):
            if abs(store[key] - value) > 1e-6 and key not in ("train_seconds", "epochs_run"):
                self.conflicts.append(f"{self.run_id} {where}.{key}: {store[key]} (kept) vs {value} ({source})")

    def add(self, fragment, source):
        for key, value in fragment["meta"].items():
            self._put(self.meta, key, value, "meta", source)
        for mode, cell in fragment["modes"].items():
            target = self.modes.setdefault(mode, {})
            for key, value in cell.items():
                self._put(target, key, value, mode, source)


# ----------------------------------------------------------------------------- rows
def blank_row():
    return {c: None for c in COLUMNS}


def base_fields(run_id):
    parsed = vr.parse_run_id(run_id)
    if not parsed.get("valid"):
        return {"run_id": run_id}
    return {"run_id": run_id, "dataset": parsed["dataset"], "compression": parsed["compression"],
            "resolution": parsed["resolution"], "loss": parsed["loss"], "seed": parsed["seed"],
            "init": None if parsed["backbone"] == "poponly" else parsed["init"]}


def mode_sort_key(mode):
    return (MODE_ORDER.index(mode) if mode in MODE_ORDER else len(MODE_ORDER), mode)


def gaps(modes):
    """(val_gap, test_gap) = raw - cosine, percent points."""
    raw, cos = modes.get("raw", {}), modes.get("cosine", {})
    def d(key):
        a, b = num(raw.get(key)), num(cos.get(key))
        return None if a is None or b is None else a - b
    return d("val_recall20"), d("test_recall20")


def n_items_for(dataset, override=None):
    return override if override else N_ITEMS.get(dataset)


def extra_bytes(mode, dataset, n_items=None):
    if mode not in PRIOR_MODES:
        return 0
    n = n_items_for(dataset, n_items)
    return None if n is None else 4 * n


def rows_from_accumulator(acc, comparison_type, n_items=None, fields=None):
    """Rows for one run. ``fields`` (non-None values) override what the run id says."""
    base = {**base_fields(acc.run_id), **{k: v for k, v in (fields or {}).items() if v is not None}}
    if base.get("resolution") is None and base.get("compression") == "graphhash":
        base["resolution"] = 200  # PLAN 7.2: Louvain resolution 200 unless the run id / config says otherwise
    val_gap, test_gap = gaps(acc.modes)
    meta = acc.meta
    rows = []
    modes = acc.modes or {None: {}}
    for mode in sorted(modes, key=lambda m: mode_sort_key(m or "")):
        cell = modes[mode]
        row = blank_row()
        row.update(base)
        row.update({k: cell.get(k) for k in CELL_KEYS})
        row.update({"scoring_mode": mode, "rho_item": meta.get("rho_item"), "rho_bucket": meta.get("rho_bucket"),
                    "rho_dir": meta.get("rho_dir"), "val_gap": val_gap, "test_gap": test_gap,
                    "best_epoch": meta.get("best_epoch"), "epochs_run": meta.get("epochs_run"),
                    "stop_reason": meta.get("stop_reason"),
                    "train_min": None if meta.get("train_seconds") is None else meta["train_seconds"] / 60.0,
                    "table_params": meta.get("table_params"),
                    "extra_bytes": None if mode is None else extra_bytes(mode, base.get("dataset"), n_items),
                    "mapping_sha256": meta.get("mapping_sha256"), "session_slug": acc.session_slug,
                    "comparison_type": comparison_type, "status": acc.status})
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- legacy
def read_registry(path=DEFAULT_REGISTRY):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def legacy_rows(registry_path=DEFAULT_REGISTRY, legacy_dir=LEGACY_DIR, repro_dir=REPRO_DIR, strict=False):
    """-> (rows, warnings). A registered file that is missing or unparsable is skipped with a warning, or raises
    when ``strict`` (the command line default: legacy rows must never disappear silently)."""
    dirs = {"results": Path(legacy_dir), "repro": Path(repro_dir)}
    registry = read_registry(registry_path)
    warnings, runs, order = [], {}, []
    cache = {}
    ranked = sorted(range(len(registry)), key=lambda i: (registry[i]["role"] != "primary", i))
    for i in ranked:
        entry = registry[i]
        run_id = entry["run_id"]
        if entry["role"] == "none":
            continue
        if run_id not in runs:
            runs[run_id] = RunAccumulator(run_id, entry["status"], entry["session_slug"])
        acc = runs[run_id]
        if entry["role"] == "primary":
            acc.status, acc.session_slug = entry["status"], entry["session_slug"]
            if entry["stop_reason"]:
                acc.meta.setdefault("stop_reason", entry["stop_reason"])
        path = dirs[entry["dir"]] / entry["file"]
        if not path.is_file():
            if strict:
                raise FileNotFoundError(f"registered legacy file is missing: {path}")
            warnings.append(f"legacy file missing, skipped: {path}")
            continue
        parser = PARSERS[entry["parser"]]
        if entry["parser"] == "repro_csv":
            payload = path
        else:
            key = str(path)
            if key not in cache:
                cache[key] = json.loads(path.read_text(encoding="utf-8"))
            payload = cache[key]
        run = vr.parse_run_id(run_id)
        try:
            acc.add(parser(payload, run, entry["sub"] or None), entry["file"])
        except (KeyError, ValueError, TypeError) as exc:
            if strict:
                raise ValueError(f"cannot parse {entry['file']} for {run_id}: {exc!r}") from exc
            warnings.append(f"cannot parse {entry['file']} for {run_id}: {exc!r}")
    # keep run order = first appearance in the registry
    for entry in registry:
        if entry["role"] != "none" and entry["run_id"] in runs and entry["run_id"] not in order:
            order.append(entry["run_id"])
    rows = []
    for run_id in order:
        acc = runs[run_id]
        warnings.extend(acc.conflicts)
        if not acc.modes:
            continue
        rows.extend(rows_from_accumulator(acc, "reproduction"))
    return rows, warnings


# ----------------------------------------------------------------------------- new runs
def _cfg_args(cfg):
    args = cfg.get("args") if isinstance(cfg, dict) and isinstance(cfg.get("args"), dict) else {}
    return {**(cfg if isinstance(cfg, dict) else {}), **args}


def infer_comparison_type(cfg, fields):
    explicit = vr.find_key(cfg, ("comparison_type",)) if isinstance(cfg, dict) else None
    if explicit in ("controlled", "reproduction"):
        return explicit
    args = _cfg_args(cfg)
    lrwd = CONTROLLED_LRWD.get(fields.get("dataset"))
    try:
        same = lrwd is not None and math.isclose(float(args["lr"]), lrwd[0]) and math.isclose(float(args["wd"]), lrwd[1],
                                                                                              rel_tol=1e-6)
    except (KeyError, TypeError, ValueError):
        same = False
    if same and fields.get("loss") == "directau" and (
            fields.get("compression") == "full"
            or (fields.get("compression") == "graphhash" and fields.get("init") == "initmatch")):
        return "controlled"
    return "reproduction"


def read_norm_log(path, best_epoch):
    """rho_item / rho_bucket / rho_dir at best_epoch (else the last logged epoch) from norm_log.csv.

    The runner writes a long format: ``epoch, kind, q, n_items, mean_norm, median_norm, value`` with
    kind in {quintile, rho_item, rho_bucket, rho_dir}. A wide format (one column per rho) is also read.
    """
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = [{(k or "").strip().lower(): v for k, v in r.items()} for r in csv.DictReader(f)]
    except OSError:
        return {}

    def epoch_of(r):
        try:
            return int(float(r.get("epoch", "")))
        except ValueError:
            return None
    epochs = [e for e in (epoch_of(r) for r in rows) if e is not None]
    if not epochs:
        return {}
    chosen = best_epoch if best_epoch in epochs else max(epochs)
    out = {}
    for r in rows:
        if epoch_of(r) != chosen:
            continue
        kind = (r.get("kind") or "").strip().lower()
        try:
            if kind in ("rho_item", "rho_bucket", "rho_dir"):
                out[kind] = float(r["value"])
            else:
                for key in ("rho_item", "rho_bucket", "rho_dir"):
                    if r.get(key) not in (None, ""):
                        out[key] = float(r[key])
        except (KeyError, TypeError, ValueError):
            pass  # an empty value (e.g. rho_bucket of a Full model) stays empty
    return out


def mapping_sha_string(cfg):
    decl = vr.declared_mapping_hashes(cfg)
    if decl["users"] or decl["items"]:
        return ";".join(f"{k}={decl[k]}" for k in ("users", "items") if decl[k])
    return decl["single"]


def new_run_rows(run_dir, force_invalid=False):
    """-> (rows, warnings) for one new-format run directory."""
    run_dir = Path(run_dir)
    verdict = vr.validate_run(run_dir)
    warnings = [f"{run_dir.name}: {p}" for p in verdict["problems"]]
    cfg, _ = vr.load_json(run_dir / "config.json")
    metrics, _ = vr.load_json(run_dir / "metrics.json")
    cfg = cfg if isinstance(cfg, dict) else {}
    run_id = run_dir.name
    fields = base_fields(run_id)
    args = _cfg_args(cfg)
    if "dataset" not in fields:  # run id does not follow the naming rule: fall back to the config
        for key in ("dataset", "compression", "loss"):
            raw = str(args.get(key if key != "compression" else "method", args.get(key, ""))).lower()
            fields[key] = NAME_MAP[key].get(raw, raw or None)
        fields["seed"] = verdict["seed"]
        fields["init"] = "initmatch" if args.get("init_match") else "xavier"
    if fields.get("resolution") is None and fields.get("compression") == "graphhash":
        fields["resolution"] = args.get("resolution")  # run-id suffix wins; else the config; else 200 (see below)
    if args.get("init_match") or args.get("init-match"):
        fields["init"] = "initmatch"
    seed = verdict["seed"]
    seeds = metrics.get("seeds") if isinstance(metrics, dict) else None
    row = seeds.get(str(seed)) if isinstance(seeds, dict) and seed is not None else None
    row = row if isinstance(row, dict) else {}
    best_epoch = row.get("best_epoch")
    rho = read_norm_log(run_dir / "norm_log.csv", best_epoch)
    rho.setdefault("rho_item", row.get("spearman_norm_vs_count"))
    n_items = None
    try:
        import numpy as np
        n_items = int(np.load(run_dir / "mapping_items.npy", mmap_mode="r", allow_pickle=False).shape[0])
    except Exception:
        pass
    if force_invalid or not verdict["ok"]:
        status = "invalid"
    elif verdict["stop_reason"] in vr.CAPPED_REASONS:
        status = "capped"
    else:
        status = "valid"
    slug = vr.find_key(cfg, ("session_slug", "slug", "kernel_slug")) or ""
    sidecar = run_dir / "session_slug.txt"
    if not slug and sidecar.is_file():
        lines = sidecar.read_text(encoding="utf-8").strip().splitlines()
        slug = lines[0].strip() if lines else ""
    acc = RunAccumulator(run_id, status, str(slug))
    modes = {}
    eval_block = row.get("eval") if isinstance(row.get("eval"), dict) else {}
    for mode, block in eval_block.items():
        if isinstance(block, dict):
            modes[mode] = cell_from_block(block)
    train_s = row.get("train_seconds")
    acc.add({"meta": {**rho, "best_epoch": best_epoch, "epochs_run": row.get("epochs_run"),
                      "stop_reason": verdict["stop_reason"], "train_seconds": train_s,
                      "table_params": row.get("params"), "mapping_sha256": mapping_sha_string(cfg)},
             "modes": modes}, "metrics.json")
    return rows_from_accumulator(acc, infer_comparison_type(cfg, fields), n_items, fields), warnings


def new_runs_rows(runs_dir):
    rows, warnings = [], []
    runs_dir = Path(runs_dir)
    for path in sorted(glob.glob(str(runs_dir / "*" / "metrics.json"))):
        r, w = new_run_rows(Path(path).parent)
        rows.extend(r)
        warnings.extend(w)
    for path in sorted(glob.glob(str(runs_dir / "_invalid" / "*" / "metrics.json"))):
        r, w = new_run_rows(Path(path).parent, force_invalid=True)
        rows.extend(r)
        warnings.extend(w)
    return rows, warnings


# ----------------------------------------------------------------------------- controls (score_controls.py)
CONTROL_PREFIX = "controls/"


def _rs(block, key):
    """(value, sd) of a metric in a controls block: {"recall": r} (single) or {"recall": {"mean", "sd"}} (agg)."""
    v = block.get(key) if isinstance(block, dict) else None
    if isinstance(v, dict):
        return num(v.get("mean")), num(v.get("sd"))
    return num(v), None


def ctl_cell(test=None, val=None, beta=None):
    """Cell (percent) from controls blocks. ``test`` / ``val`` are single or aggregated {recall, ndcg} dicts."""
    t_r, t_sd = _rs(test, "recall")
    t_n, _ = _rs(test, "ndcg")
    v_r, _ = _rs(val, "recall")
    v_n, _ = _rs(val, "ndcg")
    return {"beta": num(beta), "val_recall20": pct(v_r), "val_ndcg20": pct(v_n), "test_recall20": pct(t_r),
            "test_ndcg20": pct(t_n), "test_std_recall20": pct(t_sd)}


def val_at_best(block):
    """Validation cell at best_beta of a {best_beta, val_by_beta{beta: recall}, val_seed0?} block."""
    best = block.get("best_beta")
    out = {}
    if best is not None and isinstance(block.get("val_by_beta"), dict):
        try:
            out["recall"] = by_beta(block["val_by_beta"], best)
        except KeyError:
            pass
    seed0 = block.get("val_seed0")
    if isinstance(seed0, dict):
        out.setdefault("recall", seed0.get("recall"))
        out["ndcg"] = seed0.get("ndcg")
    return out


def controls_cells(doc):
    """[(cell name, cell, state)] for a controls.json of schema score_controls/1 (names as in the file).

    ``state`` is the per-item state the cell needs: None = none, "item" = float32 per item,
    "bucket" = float32 per bucket (mu), "u8" = one byte per item.
    """
    cells = []
    for b, block in (doc.get("legacy_topk") or {}).items():  # old item-level topk scoring, default tie order
        if isinstance(block, dict):
            beta0, prior = block.get("beta0") or {}, block.get("prior") or {}
            cells.append((f"legacy_topk/{b}/beta0", ctl_cell(beta0.get("test"), beta0.get("val"), 0.0), None))
            cells.append((f"legacy_topk/{b}/prior",
                          ctl_cell(prior.get("test"), prior.get("val"), prior.get("best_beta")), "item"))
    ties = doc.get("tie_keys") or {}
    for key in ("random", "index", "pop"):
        if isinstance(ties.get(key), dict):
            cells.append((f"tie_keys/{key}", ctl_cell(ties[key].get("test"), ties[key].get("val_seed0"), 0.0), None))
    fixed = doc.get("decomposition_fixed_beta") or {}
    means = {m: ((fixed.get(m) or {}).get("cells") or {}) for m in ("recall", "ndcg")}
    for name in ("base", "mu", "eps", "both"):
        if name in means["recall"]:
            test = {"recall": means["recall"][name], "ndcg": means["ndcg"].get(name)}
            cells.append((f"decomposition_fixed_beta/{name}",
                          ctl_cell(test, None, 0.0 if name == "base" else fixed.get("beta_star")),
                          {"base": None, "mu": "bucket"}.get(name, "item")))
    for name, block in ((doc.get("decomposition_retuned") or {}).get("cells") or {}).items():
        if isinstance(block, dict):
            cells.append((f"decomposition_retuned/{name}",
                          ctl_cell(block.get("test"), val_at_best(block) or None, block.get("best_beta")),
                          {"base": None, "mu": "bucket"}.get(name, "item")))
    shuffle = doc.get("within_bucket_shuffle")
    if isinstance(shuffle, dict):
        cells.append(("within_bucket_shuffle", ctl_cell(shuffle.get("test"), None, shuffle.get("beta_star")), "item"))
    for name in ("cos_prior", "raw_prior"):
        block = doc.get(name)
        if isinstance(block, dict):
            cells.append((name, ctl_cell(block.get("test"), val_at_best(block) or None, block.get("best_beta")),
                          "item"))
    for name in ("cos_prior_u8", "raw_prior_u8"):
        block = doc.get(name)
        if isinstance(block, dict):
            cells.append((name, ctl_cell(block.get("test"), None, block.get("beta_star")), "u8"))
    e2e = doc.get("e2e")
    if isinstance(e2e, dict):
        for name in ("unmodified", "len_shuffle_decile", "len_shuffle_global", "len_replace_fit"):
            if isinstance(e2e.get(name), dict):
                cells.append((f"e2e/{name}", ctl_cell(e2e[name].get("test"), e2e[name].get("val"), 0.0), None))
    return cells


def controls_rows(controls_json, run_info=None):
    """Rows of one controls.json -> (rows, warnings).

    ``run_info[run_id] = (status, comparison_type)`` of the scored run: the status is inherited when the controls
    validate (controls that fail validation are always status invalid).
    """
    controls_json = Path(controls_json)
    verdict = vr.validate_controls(controls_json.parent)
    data, _ = vr.load_json(controls_json)
    if not isinstance(data, dict):
        return [], [f"{controls_json}: unreadable"]
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    run_id = str(meta.get("run_id") or controls_json.parent.name)
    fields = base_fields(run_id)
    if "dataset" not in fields:  # run id outside the naming rule: use what score_controls recorded
        method = str(meta.get("method", "")).lower()
        fields.update({"dataset": NAME_MAP["dataset"].get(str(meta.get("dataset", "")).lower()),
                       "compression": NAME_MAP["compression"].get(method),
                       "loss": {"cos": "directau", "raw": "bpr"}.get(meta.get("base"))})
    n_items = meta.get("n_items")
    table_rows = meta.get("table_rows")
    status, comparison = ("valid", "") if verdict["ok"] else ("invalid", "")
    if verdict["ok"] and run_info:
        # Control labels preserve the scored checkpoint's legacy suffix. Such a
        # source must inherit the legacy training status, never a later run's
        # status simply because it has the same canonical condition and seed.
        source_id = run_id
        if source_id not in run_info and re.search(r"-legacy\d*$", run_id):
            canonical = re.sub(r"-legacy\d*$", "", run_id)
            legacy_sources = [k for k in run_info if k.startswith(canonical + "-legacy")]
            if len(legacy_sources) == 1:
                source_id = legacy_sources[0]
            elif not legacy_sources:
                source_id = canonical
        if source_id in run_info:
            status, comparison = run_info[source_id]
    sha = ";".join(f"{k}={meta[f'mapping_{k}_sha256']}" for k in ("users", "items") if meta.get(f"mapping_{k}_sha256"))
    rows = []
    for name, cell, state in controls_cells(data):
        row = blank_row()
        row.update(fields)
        row.update(cell)
        if isinstance(table_rows, list) and len(table_rows) == 2:
            row["table_params"] = 64 * (int(table_rows[0]) + int(table_rows[1]))  # MF dim 64 (PLAN 7.2)
        if state is None:
            row["extra_bytes"] = 0
        elif state == "bucket" and isinstance(table_rows, list) and fields.get("compression") != "dfreq":
            row["extra_bytes"] = 4 * int(table_rows[1])
        elif state == "item" and n_items:
            row["extra_bytes"] = 4 * int(n_items)
        elif state == "u8" and n_items:
            row["extra_bytes"] = int(n_items)
        session_slug = str(data.get("session_slug", ""))
        if not session_slug and "_sessions" in controls_json.parts:
            session_slug = controls_json.parts[controls_json.parts.index("_sessions") + 1]
        row.update({"scoring_mode": CONTROL_PREFIX + name, "mapping_sha256": sha or None,
                    "session_slug": session_slug, "comparison_type": comparison,
                    "status": status})
        rows.append(row)
    warnings = [f"{controls_json.parent.name}: {p}" for p in verdict["problems"]]
    if not rows:
        warnings.append(f"{controls_json}: no scored cells found")
    return rows, warnings


def all_controls_rows(*dirs, run_info=None):
    rows, warnings, seen = [], [], set()
    for d in dirs:
        for path in sorted(Path(d).rglob("controls.json")):
            if str(Path(path).resolve()) in seen:
                continue
            seen.add(str(Path(path).resolve()))
            r, w = controls_rows(path, run_info)
            rows.extend(r)
            warnings.extend(w)
    return rows, warnings


# ----------------------------------------------------------------------------- pop-only (A6)
def poponly_rows(results_dir, existing=frozenset()):
    """Rows for results/A6_poponly_*.json, skipping (run_id, mode) pairs that already exist (legacy Gowalla)."""
    rows = []
    for path in sorted(glob.glob(str(Path(results_dir) / "A6_poponly_*.json"))):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        dataset = NAME_MAP["dataset"].get(str(data.get("dataset", "")).lower())
        if not dataset:
            continue
        acc = RunAccumulator(f"{dataset}__poponly__none__none__s0", "valid", "")
        acc.add(parse_poponly(data, None), Path(path).name)
        rows.extend(r for r in rows_from_accumulator(acc, "reproduction")
                    if (r["run_id"], r["scoring_mode"]) not in existing)
    return rows


# ----------------------------------------------------------------------------- driver
def build_rows(runs_dir=PROJECT / "runs", controls_dir=PROJECT / "controls", results_dir=PROJECT / "results",
               registry=DEFAULT_REGISTRY, legacy_dir=LEGACY_DIR, repro_dir=REPRO_DIR, include_legacy=True,
               strict=False):
    rows, warnings = [], []
    if include_legacy:
        r, w = legacy_rows(registry, legacy_dir, repro_dir, strict)
        rows.extend(r), warnings.extend(w)
    r, w = new_runs_rows(runs_dir)
    modern_ids = {row["run_id"] for row in r}
    rewrites = {}
    # A repeated condition/seed is not the same trained checkpoint. Keep the
    # registry reader's longstanding IDs intact, but disambiguate collisions in
    # the combined ledger. Relabel every mode of the affected legacy run.
    for row in rows:
        rid = row["run_id"]
        if rid in modern_ids and rid not in rewrites:
            suffix = "-legacy"
            if row.get("stop_reason") == "fixed_epochs" and row.get("epochs_run") is not None:
                suffix += str(int(row["epochs_run"]))
            rewrites[rid] = rid + suffix
            warnings.append(f"preserved distinct legacy lineage: {rid} -> {rewrites[rid]}")
    rows = [{**row, "run_id": rewrites.get(row["run_id"], row["run_id"])} for row in rows]
    rows.extend(r), warnings.extend(w)
    run_info = {r["run_id"]: (r["status"], r["comparison_type"]) for r in rows}
    r, w = all_controls_rows(Path(runs_dir), Path(controls_dir), run_info=run_info)
    rows.extend(r), warnings.extend(w)
    rows.extend(poponly_rows(results_dir, {(r["run_id"], r["scoring_mode"]) for r in rows}))
    seen = set()
    for r in rows:
        key = (r["run_id"], r["scoring_mode"])
        if key in seen:
            warnings.append(f"duplicate row for {key}")
        seen.add(key)
    return rows, warnings


def write_csv(rows, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: fmt(row.get(c)) for c in COLUMNS})


# The original writer/schema remain a stable public interface. Analyses use
# their own named columns, never the main capped-Recall@20 cells.
V2_EXTRA_COLUMNS = [
    "schema_version", "row_kind", "lineage", "capped", "converged", "metric_family", "recall_denominator",
    "analysis_status", "analysis_sources", "source_policy", "table_bytes",
    "float32_prior_bytes", "float32_prior_overhead_pct",
    "e5_tail_primary_pct", "e5_tail_standard_pct", "e5_tail_contribution_pct", "e5_catalog_coverage_pct",
    "e5_tail_eligible_users", "e5_n_users", "e5_denominator", "e5_analysis_status", "e5_source_policy",
    "e7_val_delta_pp", "e7_test_delta_pp", "e7_pass", "e7_uint8_payload_bytes", "e7_uint8_scale_bytes",
    "e7_uint8_total_bytes", "e7_uint8_overhead_pct", "e7_analysis_status", "e7_source_policy",
    "e9_beta_original", "e9_beta_retuned", "e9_weighted_recall_original_pct", "e9_weighted_recall_retuned_pct",
    "e9_standard_recall_original_pct", "e9_standard_recall_retuned_pct", "e9_denominator",
    "e9_analysis_status", "e9_source_policy"]
V2_COLUMNS = COLUMNS + V2_EXTRA_COLUMNS
V2_SCHEMA = "master_metrics/2"


def _analysis_records(doc, family):
    if isinstance(doc.get("records"), list):
        return doc["records"]
    if family == "e9":
        return doc.get("cases", [])
    if family == "e7":
        return doc.get("fresh_modern_checks", [])
    return []


def _analysis_mode(record):
    mode = record.get("scoring_mode")
    if mode:
        return {"cos": "cosine", "cos+prior": "cos_prior", "raw+prior": "raw_prior"}.get(mode, mode)
    return "raw_prior" if record.get("base") == "raw" or record.get("loss") == "bpr" else "cos_prior"


def _source_row(run_id, rows, modern_ids):
    own = [r for r in rows if r["run_id"] == run_id and not str(r.get("scoring_mode", "")).startswith("controls/")]
    if own:
        return own[0]
    # A legacy control must never inherit a later modern condition's metadata.
    canonical = re.sub(r"-legacy\d*$", "", run_id)
    candidates = [r for r in rows if (r["run_id"] == canonical or r["run_id"].startswith(canonical + "-legacy"))
                  and r["run_id"] not in modern_ids and not str(r.get("scoring_mode", "")).startswith("controls/")]
    return candidates[0] if candidates else None


def build_v2_rows(rows, results_dir=PROJECT / "results", runs_dir=PROJECT / "runs"):
    """Read optional final analyses; return a separate enriched ledger and warnings.

    Final JSON contract: records[{run_id, scoring_mode, analysis_status,
    source_policy, ...}]. Recall/coverage inputs are fractions; E7 deltas are pp.
    Analysis-only rows preserve legacy checkpoint IDs without inventing main scores.
    """
    results_dir, runs_dir = Path(results_dir), Path(runs_dir)
    modern_ids = {p.parent.name for p in runs_dir.glob("*/metrics.json")}
    output, warnings = [], []
    for original in rows:
        row = {c: original.get(c) for c in V2_COLUMNS}
        mode = str(row.get("scoring_mode") or "")
        kind = "control" if mode.startswith("controls/") else "baseline" if mode == "pop_only" else "training_score"
        lineage = "baseline" if kind == "baseline" else "modern" if row["run_id"] in modern_ids else "legacy"
        if kind == "control" or "-legacy" in row["run_id"]:
            lineage = "legacy"
        source = _source_row(row["run_id"], rows, modern_ids) if kind == "control" else None
        if source:
            for key in ("stop_reason", "best_epoch", "epochs_run"):
                row[key] = source.get(key)
        row.update(schema_version=V2_SCHEMA, row_kind=kind, lineage=lineage,
                   capped=row.get("status") == "capped" or row.get("stop_reason") in vr.CAPPED_REASONS,
                   converged=row.get("status") == "valid" and row.get("stop_reason") == "patience",
                   metric_family="capped_recall20", recall_denominator="macro_mean_hits_over_min(20,n_truth)",
                   source_policy=("saved GPU expanded-item float32 torch.topk" if lineage == "modern" else
                                  "saved Stage 1 GPU scoring controls; exact policy in controls.json" if kind == "control" else
                                  "legacy scalar artifact; source lineage in legacy_registry.csv"),
                   analysis_sources="", analysis_status="not_available")
        for family in ("e5", "e7", "e9"):
            row[family + "_analysis_status"] = "not_available"
        row["e5_denominator"] = "tail_primary: eligible_users,hits_tail/min(20,n_tail_truth); standard: hits_tail/n_tail_truth; contribution: all_users,hits_tail/min(20,n_all_truth)"
        row["e9_denominator"] = "macro_mean_weighted_hits_over_sum_truth_weights; equal_weights=standard_recall"
        ni = N_ITEMS.get(row.get("dataset"))
        if ni is not None:
            row["float32_prior_bytes"] = 4 * ni
        if row.get("table_params") is not None:
            row["table_bytes"] = 4 * int(float(row["table_params"]))
            if row["table_bytes"] and ni is not None:
                row["float32_prior_overhead_pct"] = 100 * 4 * ni / row["table_bytes"]
        output.append(row)
    index = {(r["run_id"], r["scoring_mode"]): r for r in output}
    for family in ("e5", "e7", "e9"):
        path = results_dir / f"{family.upper()}_final.json"
        if not path.is_file():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        used = set()
        for record in _analysis_records(doc, family):
            rid = record.get("run_id")
            if not rid:
                warnings.append(f"{path.name}: record without run_id")
                continue
            key = (rid, _analysis_mode(record))
            if key in used:
                raise ValueError(f"duplicate {family} analysis key: {key}")
            used.add(key)
            if key not in index:
                source = _source_row(rid, rows, modern_ids) or next((r for r in rows if r["run_id"] == rid), None)
                extra = blank_row()
                if source:
                    extra.update(source)
                extra.update(base_fields(rid))
                extra.update(run_id=rid, scoring_mode=key[1])
                for c in CELL_KEYS:
                    extra[c] = None
                initial, _ = build_v2_rows([extra], results_dir / "_no_analysis_inputs", runs_dir)
                target = initial[0]
                target.update(row_kind="analysis_only", metric_family="analysis_only", source_policy="saved analysis; main score absent")
                output.append(target)
                index[key] = target
            target = index[key]
            status = record.get("analysis_status", record.get("retuning_status", "complete"))
            policy = record.get("source_policy", record.get("ranking_policy", record.get("float_test_source", "see analysis artifact")))
            target[family + "_analysis_status"] = status
            target[family + "_source_policy"] = policy
            sources = set(filter(None, str(target["analysis_sources"]).split(";")))
            sources.add(path.name)
            target["analysis_sources"] = ";".join(sorted(sources))
            if family == "e5":
                for src, dst in (("tail_primary", "e5_tail_primary_pct"), ("tail_standard", "e5_tail_standard_pct"),
                                 ("tail_contribution", "e5_tail_contribution_pct"), ("catalog_coverage", "e5_catalog_coverage_pct")):
                    target[dst] = pct(record.get(src))
                target["e5_tail_eligible_users"] = record.get("tail_eligible_users")
                target["e5_n_users"] = record.get("n_users")
            elif family == "e7":
                target["e7_val_delta_pp"] = record.get("val_delta_pp", (record.get("val") or {}).get("delta_pp"))
                target["e7_test_delta_pp"] = record.get("test_delta_pp", (record.get("test") or {}).get("delta_pp"))
                target["e7_pass"] = record.get("pass")
                if "pass" not in record and record.get("val") and record.get("test"):
                    target["e7_pass"] = bool(record["val"].get("pass") and record["test"].get("pass"))
                storage = record.get("storage") or {}
                payload = record.get("uint8_payload_bytes", storage.get("payload_bytes"))
                scale = record.get("uint8_scale_bytes", storage.get("scale_bytes"))
                target["e7_uint8_payload_bytes"], target["e7_uint8_scale_bytes"] = payload, scale
                if payload is not None and scale is not None:
                    target["e7_uint8_total_bytes"] = payload + scale
                for src, dst in (("float32_prior_bytes", "float32_prior_bytes"), ("table_bytes", "table_bytes"),
                                 ("float32_overhead_pct", "float32_prior_overhead_pct")):
                    if record.get(src) is not None:
                        target[dst] = record[src]
                if target.get("table_bytes") and target.get("e7_uint8_total_bytes") is not None:
                    target["e7_uint8_overhead_pct"] = 100 * target["e7_uint8_total_bytes"] / target["table_bytes"]
            else:
                target["e9_beta_original"] = record.get("original_beta")
                target["e9_beta_retuned"] = record.get("retuned_beta")
                for term, block in (("original", record.get("original_beta_test") or {}),
                                    ("retuned", record.get("retuned_beta_test") or {})):
                    for metric in ("weighted_recall", "standard_recall"):
                        target[f"e9_{metric}_{term}_pct"] = pct(block.get(metric))
    for row in output:
        row["analysis_status"] = ";".join(f"{f}:{row[f + '_analysis_status']}" for f in ("e5", "e7", "e9"))
    return output, warnings


def write_csv_v2(rows, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=V2_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: fmt(row.get(c)) for c in V2_COLUMNS})


def write_v2_schema(out):
    doc = {"schema_version": V2_SCHEMA, "primary_key": ["run_id", "scoring_mode"], "columns": V2_COLUMNS,
           "v1_compatibility": "Original COLUMNS/write_csv unchanged; main Recall@20 remains capped and in percent.",
           "units": {"main_recall_ndcg_and_e5_e9_metrics": "percent", "e7_delta": "percentage points",
                     "beta": "unscaled", "storage": "bytes", "overhead": "percent of float32 embedding table"},
           "metric_families": {"capped_recall20": "macro mean hits/min(20,n_truth)",
               "standard_recall": "macro mean hits/n_truth",
               "tail_primary": "tail-eligible users only; macro mean tail_hits/min(20,n_tail_truth)",
               "tail_standard": "tail-eligible users only; macro mean tail_hits/n_tail_truth",
               "tail_contribution": "all evaluated users; macro mean tail_hits/min(20,n_all_truth)",
               "catalog_coverage": "unique recommended original items/full original catalog",
               "weighted_recall": "macro mean sum_hit (1+train_count)^(-1/2) / sum_truth (1+train_count)^(-1/2); equal weights give STANDARD recall",
               "quantization_delta": "uint8 minus float32 capped Recall@20; same checkpoint/beta/users/masks/arithmetic/ties"},
           "missing_values": "Blank/null is unavailable, never zero. analysis_only rows do not invent main metrics.",
           "analysis_inputs": ["E5_final.json", "E7_final.json", "E9_final.json"],
           "lineage": "modern=local new-format training checkpoint; legacy=registered historical training/control artifact; baseline=pop-only"}
    Path(out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--out-v2", type=Path, default=None)
    ap.add_argument("--schema-v2", type=Path, default=None)
    ap.add_argument("--runs", type=Path, default=PROJECT / "runs")
    ap.add_argument("--controls", type=Path, default=PROJECT / "controls")
    ap.add_argument("--results", type=Path, default=PROJECT / "results")
    ap.add_argument("--legacy-dir", type=Path, default=LEGACY_DIR)
    ap.add_argument("--repro-dir", type=Path, default=REPRO_DIR)
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--lenient", action="store_true", help="skip registered legacy files that are missing (warning)")
    args = ap.parse_args(argv)
    args.out_v2 = args.out_v2 or args.out.with_name(args.out.stem + "_v2.csv")
    args.schema_v2 = args.schema_v2 or args.out_v2.with_suffix(".schema.json")
    rows, warnings = build_rows(args.runs, args.controls, args.results, args.registry, args.legacy_dir, args.repro_dir,
                                strict=not args.lenient)
    write_csv(rows, args.out)
    v2_rows, v2_warnings = build_v2_rows(rows, args.results, args.runs)
    write_csv_v2(v2_rows, args.out_v2)
    write_v2_schema(args.schema_v2)
    warnings.extend(v2_warnings)
    for w in warnings:
        print("WARNING:", w, file=sys.stderr)
    runs = {r["run_id"] for r in rows}
    by_status = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print(f"wrote {len(rows)} rows for {len(runs)} runs to {args.out}  status counts: {by_status}")
    print(f"wrote {len(v2_rows)} version-2 rows to {args.out_v2}; schema {args.schema_v2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
