"""Tests for T5: validate_run.py and aggregate.py.   Run: python -m pytest scripts/test_t5.py -q

Part 1 reproduces RESULTS.md section 1-2 (PLAN section 2) from the legacy JSONs. Tolerances are +-0.01
percent points unless noted (RESULTS shows some values truncated, e.g. Full cosine 12.9959 -> "12.99").
Part 2 checks the registry, Part 3 validate_run on synthetic run dirs, Part 4 aggregate on synthetic runs.
"""
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aggregate as agg  # noqa: E402
import validate_run as vr  # noqa: E402

needs_legacy = pytest.mark.skipif(not agg.LEGACY_DIR.is_dir(), reason="legacy results directory not available")
STATUSES = {"valid", "valid-short", "truncated", "capped", "ablation", "exploratory", "invalid"}
G = "gowalla__mf__"
A = "amazonbook__mf__"


# ============================================================================ legacy reproduction
@pytest.fixture(scope="module")
def legacy():
    rows, warnings = agg.legacy_rows()
    return rows, warnings


def get(rows, run_id, mode):
    found = [r for r in rows if r["run_id"] == run_id and r["scoring_mode"] == mode]
    assert len(found) == 1, f"{run_id} / {mode}: {len(found)} rows"
    return found[0]


def close(value, expected, tol=0.01):
    assert value is not None and abs(value - expected) <= tol, f"{value} != {expected} (+-{tol})"


def check(rows, run_id, mode, test=None, ndcg=None, val=None, beta=None, tol=0.01):
    r = get(rows, run_id, mode)
    if test is not None:
        close(r["test_recall20"], test, tol)
    if ndcg is not None:
        close(r["test_ndcg20"], ndcg, tol)
    if val is not None:
        close(r["val_recall20"], val, tol)
    if beta is not None:
        assert r["beta"] == pytest.approx(beta)
    return r


@needs_legacy
def test_columns_and_units(legacy, tmp_path):
    rows, _ = legacy
    out = tmp_path / "m.csv"
    agg.write_csv(rows, out)
    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["run_id", "dataset", "compression", "resolution", "loss", "seed", "init",
                                     "scoring_mode", "beta", "val_recall20", "val_ndcg20", "test_recall20",
                                     "test_ndcg20", "test_std_recall20", "rho_item", "rho_bucket", "rho_dir",
                                     "val_gap", "test_gap", "best_epoch", "epochs_run", "stop_reason", "train_min",
                                     "table_params", "extra_bytes", "mapping_sha256", "session_slug",
                                     "comparison_type", "status"]
        assert len(list(reader)) == len(rows)
    for r in rows:  # percent, never fractions and never scaled twice
        for key in ("val_recall20", "test_recall20", "test_ndcg20"):
            if r[key] is not None:
                assert 0 < r[key] < 25, (r["run_id"], r["scoring_mode"], key, r[key])
    keys = [(r["run_id"], r["scoring_mode"]) for r in rows]
    assert len(keys) == len(set(keys))


@needs_legacy
def test_legacy_merge_has_no_conflicts_and_all_rows_are_reproductions(legacy):
    rows, warnings = legacy
    assert warnings == []
    assert {r["comparison_type"] for r in rows} == {"reproduction"}
    assert {r["status"] for r in rows} <= STATUSES


@needs_legacy
def test_gowalla_graphhash_directau_three_seeds(legacy):
    rows, _ = legacy
    expected = {0: (13.26, 8.54, 13.36, 7.56, 7.58, 0.271), 1: (13.31, 8.55, 13.41, 7.57, 7.56, 0.274),
                2: (13.30, 8.53, 13.38, 7.53, 7.55, 0.270)}
    tests, ndcgs, vals = [], [], []
    for seed, (cp_test, cp_ndcg, cp_val, raw, cos, rho) in expected.items():
        rid = f"{G}graphhash__directau__s{seed}"
        r = check(rows, rid, "cos_prior", test=cp_test, ndcg=cp_ndcg, val=cp_val, beta=0.2)
        assert r["status"] == "valid-short" and r["stop_reason"] == "fixed_epochs" and r["epochs_run"] == 30
        assert r["extra_bytes"] == 4 * 40981 == 163924 and r["table_params"] == 742272 and r["resolution"] == 200
        check(rows, rid, "raw", test=raw, beta=0.0)
        check(rows, rid, "cosine", test=cos, beta=0.0)
        close(r["rho_item"], rho, 0.001)
        tests.append(r["test_recall20"]); ndcgs.append(r["test_ndcg20"]); vals.append(r["val_recall20"])
    assert round(statistics.mean(tests), 2) == 13.29 and round(statistics.stdev(tests), 2) == 0.03
    cos = [get(rows, f"{G}graphhash__directau__s{s}", "cosine") for s in (0, 1, 2)]
    assert round(statistics.mean(r["test_recall20"] for r in cos), 2) == 7.56   # RESULTS 1: 7.56 +- 0.02 / 4.03 (cos)
    assert round(statistics.stdev(r["test_recall20"] for r in cos), 2) == 0.02
    assert round(statistics.mean(r["test_ndcg20"] for r in cos), 2) == 4.03
    raw = [get(rows, f"{G}graphhash__directau__s{s}", "raw")["test_recall20"] for s in (0, 1, 2)]
    assert round(statistics.mean(raw), 2) == 7.55 and round(statistics.stdev(raw), 2) == 0.02   # PLAN 2: 7.55 +- 0.02
    assert round(statistics.mean(ndcgs), 2) == 8.54
    assert round(statistics.mean(vals), 2) == 13.38
    check(rows, f"{G}graphhash__directau__s0", "raw_prior", test=11.43, beta=5.0)
    check(rows, f"{G}graphhash__directau__s1", "raw_prior", test=11.49, beta=5.0)
    # seed 2 has no raw_prior in any legacy file (the log of the scoring job was not saved)
    assert not [r for r in rows if r["run_id"] == f"{G}graphhash__directau__s2" and r["scoring_mode"] == "raw_prior"]
    # val sweep from RESULTS 2.4 (seed 2: best 13.38)
    close(get(rows, f"{G}graphhash__directau__s2", "cos_prior")["val_recall20"], 13.38)


@needs_legacy
def test_gowalla_graphhash_bpr_authors_config(legacy):
    rows, _ = legacy
    rid = f"{G}graphhash__bpr__s0"
    raw = check(rows, rid, "raw", test=9.34, ndcg=5.49, val=9.39, beta=0.0)
    assert (raw["best_epoch"], raw["epochs_run"], raw["stop_reason"], raw["status"]) == (330, 430, "patience", "valid")
    check(rows, rid, "cosine", test=6.08, ndcg=3.08)
    check(rows, rid, "cos_prior", test=10.22, val=10.21, beta=0.1)
    rp = check(rows, rid, "raw_prior", test=11.24, ndcg=6.95, val=11.39, beta=0.5)
    close(rp["rho_item"], 0.054, 0.001)
    close(rp["test_gap"], 9.34 - 6.08, 0.02)


@needs_legacy
def test_gowalla_full_directau_epoch_120(legacy):
    rows, _ = legacy
    rid = f"{G}full__directau__s0"
    raw = check(rows, rid, "raw", test=17.90, ndcg=11.52, val=17.62)
    # RESULTS shows the cosine test as 12.99 (truncated); the stored value is 12.9959
    cos = check(rows, rid, "cosine", test=12.99, ndcg=7.44, val=12.62)
    cp = check(rows, rid, "cos_prior", test=19.54, ndcg=12.59, val=19.19, beta=0.1)
    close(cp["rho_item"], 0.815, 0.001)
    close(raw["val_gap"], 5.00, 0.01)
    close(raw["test_gap"], 4.91, 0.02)  # exact 4.899; RESULTS' 4.91 is computed from the rounded 17.90 - 12.99
    assert (cp["status"], cp["stop_reason"], cp["best_epoch"], cp["table_params"]) == ("capped", "time_cap", 120, 4533696)
    assert cp["extra_bytes"] == 163924 and cos["extra_bytes"] == 0
    assert not [r for r in rows if r["run_id"] == rid and r["scoring_mode"] == "raw_prior"]


@needs_legacy
def test_gowalla_random_dfreq_and_bias(legacy):
    rows, _ = legacy
    check(rows, f"{G}random__directau__s0", "raw", test=0.47, ndcg=0.27)
    check(rows, f"{G}random__directau__s0", "cosine", test=0.31, ndcg=0.17)
    check(rows, f"{G}random__directau__s0", "cos_prior", test=3.85, ndcg=2.44, beta=0.2)
    check(rows, f"{G}dfreq__directau__s0", "raw", test=3.23, ndcg=2.54)
    check(rows, f"{G}dfreq__directau__s0", "cosine", test=2.18, ndcg=1.71)
    r = check(rows, f"{G}dfreq__directau__s0", "cos_prior", test=5.15, ndcg=3.50, beta=0.2)
    close(r["rho_item"], 0.029, 0.001)
    assert get(rows, f"{G}random__directau__s0", "raw")["best_epoch"] == 20
    b = f"{G}graphhash__directau_bias__s0"
    check(rows, b, "cos_b", test=4.32, ndcg=2.53, val=4.33)
    check(rows, b, "cosine", test=7.48, ndcg=3.99, val=7.24)
    check(rows, b, "cos_prior", test=13.17, ndcg=8.47, val=13.17, beta=0.2)
    assert get(rows, b, "cos_b")["status"] == "ablation" and get(rows, b, "cos_b")["extra_bytes"] == 163924


@needs_legacy
def test_amazonbook(legacy):
    rows, _ = legacy
    full = f"{A}full__directau__s0"
    raw = check(rows, full, "raw", test=9.57, ndcg=6.45, val=9.51)
    check(rows, full, "cosine", test=8.70, ndcg=5.69)
    check(rows, full, "cos_prior", test=11.05, ndcg=7.48, beta=0.1)
    check(rows, full, "raw_prior", test=10.77, ndcg=7.26, beta=0.2)
    close(raw["test_gap"], 0.87, 0.01)
    close(raw["val_gap"], 1.09, 0.01)
    close(raw["rho_item"], 0.786, 0.001)
    assert (raw["best_epoch"], raw["epochs_run"], raw["stop_reason"], raw["status"]) == (25, 35, "patience", "valid")
    gh = f"{A}graphhash__directau__s0"
    r = check(rows, gh, "cos_prior", test=6.33, ndcg=4.46, beta=0.1)
    check(rows, gh, "raw", test=4.63, ndcg=3.12, val=4.62)
    check(rows, gh, "cosine", test=4.64, ndcg=3.13)
    check(rows, gh, "raw_prior", test=6.29, ndcg=4.44, beta=0.5)
    close(r["rho_item"], 0.189, 0.001)
    assert (r["status"], r["stop_reason"], r["best_epoch"], r["epochs_run"]) == ("capped", "time_cap", 40, 41)
    bpr = f"{A}graphhash__bpr__s0"
    r = check(rows, bpr, "raw_prior", test=4.81, ndcg=3.28, beta=0.2)
    check(rows, bpr, "raw", test=4.10, ndcg=2.60, val=4.04)
    check(rows, bpr, "cosine", test=2.53, ndcg=1.59)
    check(rows, bpr, "cos_prior", test=3.34, ndcg=2.12, beta=0.05)
    close(r["rho_item"], 0.084, 0.001)
    assert (r["status"], r["stop_reason"], r["best_epoch"], r["epochs_run"]) == ("valid", "patience", 1450, 1950)
    assert r["extra_bytes"] == 4 * 91599


@needs_legacy
def test_popularity_only_is_not_rescaled(legacy):
    rows, _ = legacy
    r = check(rows, "gowalla__poponly__none__none__s0", "pop_only", test=2.97, ndcg=1.70, val=3.06)
    assert r["beta"] is None and r["table_params"] == 0 and r["status"] == "valid"


@needs_legacy
def test_fast_check_dedup_and_invalid_runs(legacy):
    rows, _ = legacy
    # RESULTS 2.1 / 2.2
    b = get(rows, f"{G}graphhash__bpr__s0-b1024", "raw")
    close(b["test_recall20"], 5.31); close(b["test_ndcg20"], 3.07); close(b["val_recall20"], 5.27)  # last epoch
    assert b["status"] == "invalid" and b["train_min"] * 60 == pytest.approx(53, abs=1)
    bl = get(rows, f"{G}graphhash__bpr__s0-b1024-pat10", "raw")
    close(bl["test_recall20"], 5.45); close(bl["val_recall20"], 5.46)
    assert bl["status"] == "invalid" and bl["epochs_run"] == 15 and bl["best_epoch"] == 5
    amz = get(rows, f"{A}graphhash__bpr__s0-pat100", "raw")
    assert amz["status"] == "invalid" and amz["epochs_run"] == 110
    close(amz["test_recall20"], 3.93)
    dau_test, fix_test, dau_val, fix_val = [], [], [], []
    for s in (0, 1, 2):
        raw = get(rows, f"{G}graphhash__directau__s{s}", "raw")
        fix = get(rows, f"x__{G}graphhash__directau_dedup__s{s}", "raw")
        assert fix["status"] == "exploratory"
        dau_test.append(raw["test_recall20"]); fix_test.append(fix["test_recall20"])
        dau_val.append(raw["val_recall20"]); fix_val.append(fix["val_recall20"])
    for got, want in zip(dau_test, (7.56, 7.57, 7.53)): close(got, want)
    for got, want in zip(fix_test, (7.53, 7.65, 7.58)): close(got, want)
    for got, want in zip(dau_val, (7.46, 7.47, 7.49)): close(got, want)
    for got, want in zip(fix_val, (7.57, 7.55, 7.52)): close(got, want)
    for got, want in zip((f - d for f, d in zip(fix_test, dau_test)), (-0.02, 0.08, 0.05)): close(got, want, 0.015)
    assert round(statistics.mean(fix_test), 2) == 7.59 and round(statistics.stdev(fix_test), 2) == 0.06


# ============================================================================ registry
@needs_legacy
def test_registry_covers_every_legacy_file():
    registry = agg.read_registry()
    files = {(r["dir"], r["file"]) for r in registry}
    for path in agg.LEGACY_DIR.glob("*.json"):
        if path.name == "queue_state.json":  # scheduler state, no metrics
            continue
        assert ("results", path.name) in files, f"{path.name} is not in legacy_registry.csv"
    for r in registry:
        base = agg.LEGACY_DIR if r["dir"] == "results" else agg.REPRO_DIR
        assert (base / r["file"]).is_file(), r["file"]
        assert r["status"] in STATUSES and r["role"] in {"primary", "supplement", "none"}
        assert r["role"] == "none" or r["parser"] in agg.PARSERS
        assert vr.parse_run_id(r["run_id"])["valid"], r["run_id"]
    by_run = {}
    for r in registry:
        by_run.setdefault(r["run_id"], []).append(r)
    for run_id, entries in by_run.items():
        assert len({e["status"] for e in entries}) == 1, f"{run_id}: inconsistent status"
        if any(e["role"] != "none" for e in entries):
            assert sum(e["role"] == "primary" for e in entries) == 1, f"{run_id}: needs exactly one primary"


@needs_legacy
def test_registry_statuses_follow_the_brief():
    status = {r["file"]: r["status"] for r in agg.read_registry() if r["role"] != "supplement"}
    for name in ("fast_bpr.json", "fast_bpr_long.json", "exp_p4-amz-gh-bpr.json", "p1-bpr-ext.log"):
        assert status[name] == "invalid"
    for name in ("fast_dau_fix_seed0.json", "fast_dau_fix_seed1.json", "fast_dau_fix_seed2.json"):
        assert status[name] == "exploratory"
    assert status["exp_p4-amz-gh-dau.json"] == "capped" and status["metrics.csv"] == "capped"
    assert status["exp_p5-gh-dau-bias.json"] == "ablation"
    for name in ("exp_p2-gh-dau-s0.json", "exp_p2-gh-dau-s1.json", "fast_dau_seed2.json", "exp_p3-random-dau.json",
                 "exp_p3-dfreq-dau.json"):
        assert status[name] == "valid-short"
    for name in ("fast_bpr_author.json", "exp_p4-amz-full-dau.json", "exp_p4-amz-gh-bpr-long.json"):
        assert status[name] == "valid"


@needs_legacy
def test_duplicate_legacy_files_agree():
    """The fast-check JSONs, rescore files and mech/popprior files repeat numbers of the P2/P4/P5 files."""
    L = agg.LEGACY_DIR
    load = lambda n: json.loads((L / n).read_text())  # noqa: E731
    for seed in (0, 1):
        exp = load(f"exp_p2-gh-dau-s{seed}.json")["seeds"][str(seed)]["eval"]["raw"]
        fast = load(f"fast_dau_seed{seed}.json")
        assert exp["test"]["recall"] == fast["test_recall"] and exp["test"]["ndcg"] == fast["test_ndcg"]
        assert exp["val"]["recall"] == fast["best_val"]
    assert load("fast_dau.json")["test_recall"] == load("fast_dau_seed0.json")["test_recall"]
    assert load("fast_dau_fix.json")["test_recall"] == load("fast_dau_fix_seed0.json")["test_recall"]
    p1 = load("p1_bpr_ext.json")["raw_prior"]
    mech = load("graphhash-mech-1017.json")["GH_BPR"]
    assert p1["test"] == mech["test_best_rawdot"] and p1["best_beta"] == mech["best_beta"]
    assert load("graphhash-popprior-gh-1017.json")["GraphHash"]["test_raw_dot"]["recall"] == \
        load("fast_dau_seed2.json")["test_recall"]
    assert list(load("graphhash-rescore-dau-s012-1016.json").values())[0]["cosine_recall"] == \
        load("graphhash-popprior-gh-1017.json")["GraphHash"]["test_beta0"]["recall"]


def test_accumulator_reports_conflicts_and_first_value_wins():
    acc = agg.RunAccumulator("r", "valid", "")
    acc.add({"meta": {"rho_item": 0.5}, "modes": {"raw": {"test_recall20": 7.5, "val_recall20": None}}}, "a")
    acc.add({"meta": {"rho_item": 0.5}, "modes": {"raw": {"test_recall20": 7.9, "val_recall20": 7.4}}}, "b")
    assert acc.modes["raw"] == {"test_recall20": 7.5, "val_recall20": 7.4}
    assert len(acc.conflicts) == 1 and "test_recall20" in acc.conflicts[0]


def test_parse_run_id():
    p = vr.parse_run_id("x__gowalla__mf__graphhash__directau__s2-initmatch-r400")
    assert (p["valid"], p["exploratory"], p["dataset"], p["compression"], p["loss"], p["seed"], p["init"],
            p["resolution"]) == (True, True, "gowalla", "graphhash", "directau", 2, "initmatch", 400)
    assert vr.parse_run_id("gowalla__mf__full__bpr__s0")["init"] == "xavier"
    assert not vr.parse_run_id("run1")["valid"]


# ============================================================================ synthetic run directories
def eval_block(best, val, test, ndcg_scale=0.5, extra_betas=()):
    grid = {"0.0": {"recall": val * 0.5, "ndcg": val * 0.25}}
    for b in extra_betas:
        grid[str(b)] = {"recall": val * 0.7, "ndcg": val * 0.3}
    grid[str(float(best))] = {"recall": val, "ndcg": val * ndcg_scale}
    if best == 0:
        grid = {"0.0": {"recall": val, "ndcg": val * ndcg_scale}}
    return {"best_beta": float(best), "val": {"recall": val, "ndcg": val * ndcg_scale},
            "test": {"recall": test, "ndcg": test * ndcg_scale}, "val_by_beta": grid}


def make_eval(bias=False):
    ev = {"raw": eval_block(0, 0.0746, 0.0756), "cosine": eval_block(0, 0.0752, 0.0758),
          "cos_prior": eval_block(0.2, 0.1336, 0.1326), "raw_prior": eval_block(5.0, 0.1156, 0.1143)}
    if bias:
        ev["cos_b"] = eval_block(0, 0.0433, 0.0432)
        ev["cos_b_prior"] = eval_block(0, 0.0433, 0.0432)
    return ev


def make_run(root, name="gowalla__mf__graphhash__directau__s0", stop="patience", exit_code=0, bias=False,
             sha="dict-file", cfg_extra=None, metrics_extra=None, skip=(), best_epoch=10, seed=None, slug=None):
    """Write a complete new-format run directory (same layout as exp_run.py) and return its path.

    ``seed`` defaults to the one in ``name``; ``slug`` is written to the session_slug.txt sidecar.
    """
    if seed is None:
        seed = vr.parse_run_id(name).get("seed", 0)
    d = Path(root) / name
    d.mkdir(parents=True)
    users, items = np.arange(30, dtype=np.int64) % 7, np.arange(50, dtype=np.int64) % 11
    np.save(d / "mapping_users.npy", users)
    np.save(d / "mapping_items.npy", items)
    fu = hashlib.sha256((d / "mapping_users.npy").read_bytes()).hexdigest()
    fi = hashlib.sha256((d / "mapping_items.npy").read_bytes()).hexdigest()
    au, ai = hashlib.sha256(users.tobytes()).hexdigest(), hashlib.sha256(items.tobytes()).hexdigest()
    args = {"dataset": "Gowalla", "method": "GraphHash", "loss": "dau_bias" if bias else "dau", "seeds": [seed],
            "epochs": 200, "eval_every": 5, "patience": 30, "min_delta": 5e-4, "cap_min": 180.0, "lr": 0.01,
            "wd": 1e-8, "batch": None, "run_id": name, "tag": name, "resolution": 200, "init_match": False,
            "equal_item_norm": False}
    cfg = {"run_id": name, "tag": name, "args": args, "resolved": {"lr": 0.01, "wd": 1e-8, "batch": 1024},
           "code_sha256": {"exp_run.py": "ab" * 32}, "device_count": 2, "CUDA_VISIBLE_DEVICES": "0",
           "mapping_shapes": {"users": [30], "items": [50]}}
    if sha == "dict-file":      # what exp_run.py writes: sha256 of the .npy file bytes
        cfg["mapping_sha256"] = {"users": fu, "items": fi}
    elif sha == "dict-array":
        cfg["mapping_sha256"] = {"users": au, "items": ai}
    elif sha == "flat":
        cfg["mapping_users_sha256"], cfg["mapping_items_sha256"] = fu, fi
    elif sha == "single-combined":
        cfg["mapping_sha256"] = hashlib.sha256((d / "mapping_users.npy").read_bytes()
                                               + (d / "mapping_items.npy").read_bytes()).hexdigest()
    elif sha == "single-users":
        cfg["mapping_sha256"] = fu
    elif sha == "wrong":
        cfg["mapping_sha256"] = {"users": fu, "items": "0" * 64}
    extra = dict(cfg_extra or {})
    cfg["args"].update(extra.pop("args", {}))
    cfg.update(extra)
    seed_row = {"params": 742272, "val_curve": {"5": 0.07}, "best_epoch": best_epoch, "best_val": 0.0746,
                "epochs_run": 15, "stopped": stop, "stop_reason": stop, "train_seconds": 600.0, "batch": 1024,
                "spearman_norm_vs_count": 0.27, "eval": make_eval(bias), "wall_seconds": 610.0}
    metrics = {"tag": name, "run_id": name, "args": args, "status": "complete", "bucket_info": {},
               "seeds": {str(seed): seed_row}}
    metrics.update(metrics_extra or {})
    files = {"config.json": cfg, "metrics.json": metrics,
             "exit_status.json": {"exit_code": exit_code, "stop_reason": stop, "end_time": "2026-10-17T20:00:00+00:00",
                                  "error": None}}
    for fname, obj in files.items():
        (d / fname).write_text(json.dumps(obj))
    (d / f"best_seed{seed}.pt").write_bytes(b"checkpoint")
    (d / "train_log.csv").write_text("epoch,train_loss,val_recall20,elapsed_s\n5,1.0,0.07,10.0\n")
    norm = ["epoch,kind,q,n_items,mean_norm,median_norm,value"]
    for epoch, (item, bucket, direction) in {5: (0.10, 0.20, 0.30), 10: (0.27, 0.55, 0.65), 15: (0.30, 0.60, 0.70)}.items():
        norm += [f"{epoch},quintile,{q},10,{q * 0.1},{q * 0.1},"for q in range(1, 6)]
        norm += [f"{epoch},rho_item,,50,,,{item}", f"{epoch},rho_bucket,,11,,,{bucket}", f"{epoch},rho_dir,,50,,,{direction}"]
    (d / "norm_log.csv").write_text("\n".join(norm) + "\n")
    for mode in vr.REQUIRED_MODES:  # per-user files as written by exp_run.save_per_user_test
        np.savez_compressed(d / f"per_user_test__{mode}.npz", users=np.arange(100, dtype=np.int32),
                            top20=np.zeros((100, 20), dtype=np.int32), hits=np.ones(100), npos=np.full(100, 5),
                            ndcg=np.ones(100))
    if slug:
        (d / "session_slug.txt").write_text(slug + "\n")
    for fname in skip:
        if (d / fname).exists():
            (d / fname).unlink()
    return d


def problems_of(d):
    return vr.validate_run(d)["problems"]


def test_valid_run_passes(tmp_path):
    result = vr.validate_run(make_run(tmp_path))
    assert result["problems"] == [] and result["ok"] and result["converged"]
    assert result["stop_reason"] == "patience" and result["exit_code"] == 0 and result["seed"] == 0
    assert result["warnings"] == [] and set(result["modes"]) == {"raw", "cosine", "cos_prior", "raw_prior"}


@pytest.mark.parametrize("sha", ["dict-file", "dict-array", "flat", "single-combined", "single-users"])
def test_accepted_sha_formats(tmp_path, sha):
    assert vr.validate_run(make_run(tmp_path, sha=sha))["ok"]


def test_sha_mismatch_and_missing_sha(tmp_path):
    assert any("sha256" in p for p in problems_of(make_run(tmp_path, sha="wrong")))
    d = make_run(tmp_path, name="gowalla__mf__graphhash__directau__s1")
    cfg = json.loads((d / "config.json").read_text()); cfg.pop("mapping_sha256")
    (d / "config.json").write_text(json.dumps(cfg))
    assert "config.json has no mapping sha256" in problems_of(d)


def test_sha_detects_changed_mapping(tmp_path):
    d = make_run(tmp_path)
    np.save(d / "mapping_items.npy", np.arange(50, dtype=np.int64) % 12)  # recomputed mapping differs
    assert any("mapping_items.npy sha256 does not match" in p for p in problems_of(d))


def test_nonzero_exit_code_fails(tmp_path):
    result = vr.validate_run(make_run(tmp_path, exit_code=1))
    assert not result["ok"] and any("exit_code" in p for p in result["problems"])


def test_missing_exit_status_fails(tmp_path):
    assert any("exit_status.json" in p for p in problems_of(make_run(tmp_path, skip=("exit_status.json",))))


@pytest.mark.parametrize("status", ["running", "failed", None])
def test_incomplete_metrics_fail(tmp_path, status):
    d = make_run(tmp_path)
    m = json.loads((d / "metrics.json").read_text())
    m["status"] = status
    (d / "metrics.json").write_text(json.dumps(m))
    assert any("status" in p for p in problems_of(d))


def test_metrics_json_alone_is_not_success(tmp_path):
    d = tmp_path / "gowalla__mf__graphhash__directau__s0"
    d.mkdir()
    (d / "metrics.json").write_text(json.dumps({"status": "complete", "seeds": {"0": {"eval": make_eval()}}}))
    result = vr.validate_run(d)
    assert not result["ok"] and len(result["problems"]) >= 5


@pytest.mark.parametrize("mode", ["raw", "cosine", "cos_prior", "raw_prior"])
def test_every_mode_is_required(tmp_path, mode):
    d = make_run(tmp_path)
    m = json.loads((d / "metrics.json").read_text())
    del m["seeds"]["0"]["eval"][mode]
    (d / "metrics.json").write_text(json.dumps(m))
    assert f"eval mode '{mode}' is missing" in problems_of(d)


@pytest.mark.parametrize("key", ["best_beta", "val", "test", "val_by_beta"])
def test_every_eval_key_is_required(tmp_path, key):
    d = make_run(tmp_path)
    m = json.loads((d / "metrics.json").read_text())
    del m["seeds"]["0"]["eval"]["cos_prior"][key]
    (d / "metrics.json").write_text(json.dumps(m))
    assert any("cos_prior" in p and key in p for p in problems_of(d))


def test_bias_modes_required_only_for_dau_bias(tmp_path):
    assert vr.validate_run(make_run(tmp_path, name="gowalla__mf__graphhash__directau_bias__s0", bias=True))["ok"]
    d = make_run(tmp_path, name="x__gowalla__mf__graphhash__directau_bias__s0", bias=True)
    m = json.loads((d / "metrics.json").read_text())
    del m["seeds"]["0"]["eval"]["cos_b_prior"]
    (d / "metrics.json").write_text(json.dumps(m))
    assert "eval mode 'cos_b_prior' is missing" in problems_of(d)
    # a plain DirectAU run without cos_b is fine (already covered by test_valid_run_passes)


def test_selection_consistency_checks(tmp_path):
    d = make_run(tmp_path)
    m = json.loads((d / "metrics.json").read_text())
    block = m["seeds"]["0"]["eval"]["cos_prior"]
    block["val_by_beta"]["0.5"] = {"recall": 0.9, "ndcg": 0.5}  # a better beta exists than the chosen one
    (d / "metrics.json").write_text(json.dumps(m))
    assert any("argmax" in p for p in problems_of(d))
    block["best_beta"] = 0.7  # not in the grid
    (d / "metrics.json").write_text(json.dumps(m))
    assert any("not in val_by_beta" in p for p in problems_of(d))


def test_percent_valued_metrics_are_rejected(tmp_path):
    d = make_run(tmp_path)
    m = json.loads((d / "metrics.json").read_text())
    m["seeds"]["0"]["eval"]["raw"]["test"]["recall"] = 7.56
    (d / "metrics.json").write_text(json.dumps(m))
    assert any("[0, 1]" in p for p in problems_of(d))


@pytest.mark.parametrize("skip", ["best_seed0.pt", "mapping_users.npy", "mapping_items.npy"])
def test_missing_checkpoint_or_mapping_fails(tmp_path, skip):
    assert any(skip in p for p in problems_of(make_run(tmp_path, skip=(skip,))))


def test_stop_reason_must_be_set(tmp_path):
    d = make_run(tmp_path, stop=None)
    result = vr.validate_run(d)
    assert "stop_reason is not set" in result["problems"] and not result["converged"] and not result["ok"]


def test_conflicting_stop_reason(tmp_path):
    d = make_run(tmp_path, stop="patience")
    es = json.loads((d / "exit_status.json").read_text()); es["stop_reason"] = "time_cap"
    (d / "exit_status.json").write_text(json.dumps(es))
    assert any("conflicting stop_reason" in p for p in problems_of(d))


def test_stop_reason_falls_back_to_metrics(tmp_path):
    d = make_run(tmp_path, stop="patience")
    es = json.loads((d / "exit_status.json").read_text()); del es["stop_reason"]
    (d / "exit_status.json").write_text(json.dumps(es))
    assert vr.validate_run(d)["ok"]


@pytest.mark.parametrize("stop,converged", [("patience", True), ("time_cap", False), ("max_epochs", False)])
def test_converged_means_patience_only(tmp_path, stop, converged):
    result = vr.validate_run(make_run(tmp_path, stop=stop))
    assert result["ok"] and result["converged"] is converged and result["stop_reason"] == stop


def test_seed_in_run_id_must_match_config(tmp_path):
    d = make_run(tmp_path, name="gowalla__mf__graphhash__directau__s3", seed=0)  # config says 0, the id says 3
    assert any("seed in the run id" in p for p in problems_of(d))


def test_missing_directory(tmp_path):
    result = vr.validate_run(tmp_path / "nope")
    assert not result["ok"] and result["problems"] == ["run directory does not exist"]


def test_warnings_for_optional_files(tmp_path):
    d = make_run(tmp_path, skip=("per_user_test__raw.npz",))
    (d / "norm_log.csv").unlink()
    result = vr.validate_run(d)
    assert result["ok"] and "norm_log.csv is missing" in result["warnings"]
    assert "per_user_test__raw.npz is missing" in result["warnings"]


@needs_legacy
def test_legacy_eval_blocks_pass_the_eval_checks():
    """The same eval schema is used by the legacy exp_*.json files, so the checker must accept them."""
    for path in sorted(agg.LEGACY_DIR.glob("exp_*.json")):
        data = json.loads(path.read_text())
        for seed, row in data["seeds"].items():
            required = vr.REQUIRED_MODES + (vr.BIAS_MODES if "bias" in path.name else ())
            assert vr.validate_eval(row["eval"], required) == [], path.name


def run_cli(*args):
    proc = subprocess.run([sys.executable, str(HERE / "validate_run.py"), *map(str, args)], capture_output=True, text=True)
    return proc.returncode, json.loads(proc.stdout)


def test_cli_exit_codes(tmp_path):
    code, out = run_cli(make_run(tmp_path))
    assert code == 0 and out["ok"] and out["converged"] is True
    code, out = run_cli(make_run(tmp_path, name="gowalla__mf__graphhash__directau__s1", exit_code=2))
    assert code == 1 and not out["ok"]


# ---- controls directory (synthetic controls.json with the structure score_controls.py writes)
def _single(recall, ndcg):
    return {"recall": recall, "ndcg": ndcg}


def _agg(recall, ndcg, sd=0.0004, n=10):
    return {"n_seeds": n, "recall": {"mean": recall, "sd": sd, "per_seed": [recall] * n},
            "ndcg": {"mean": ndcg, "sd": sd / 2, "per_seed": [ndcg] * n}}


def controls_doc(run_id="gowalla__mf__graphhash__directau__s0", e2e=False, **meta_extra):
    meta = {"run_id": run_id, "dataset": "Gowalla", "method": "GraphHash", "base": "cos",
            "args": {"e2e": e2e, "seeds": 10}, "alias_assert": None, "n_items": 40981, "table_rows": [5792, 5806],
            "mapping_users_sha256": "aa" * 32, "mapping_items_sha256": "bb" * 32, **meta_extra}
    legacy = lambda base, b0, bb, best, vb: {  # noqa: E731
        "beta0": {"val": _single(b0[0], b0[1]), "test": _single(b0[2], b0[3])},
        "prior": {"best_beta": best, "val_by_beta": vb, "val": _single(*bb[:2]), "test": _single(*bb[2:])},
        "crosscheck_vs_repro_core_metrics_from_topk": {"max_abs_diff": 1e-15, "evaluated_users": 25789}}
    doc = {
        "schema": "score_controls/1", "status": "complete", "meta": meta,
        "legacy_topk": {"cos": legacy("cos", (0.0743, 0.0391, 0.0755, 0.0402), (0.1338, 0.0840, 0.1330, 0.0853), 0.2,
                                      {"0.0": 0.0743, "0.2": 0.1338}),
                        "raw": legacy("raw", (0.0749, 0.0397, 0.0753, 0.0401), (0.1156, 0.0691, 0.1143, 0.0700), 5.0,
                                      {"0.0": 0.0749, "5.0": 0.1156})},
        "tie_keys": {"base": "cos", "random": {"test": _agg(0.0755, 0.0402), "val_seed0": _single(0.0752, 0.0398)},
                     "index": {"test": _single(0.0757, 0.0403)}, "pop": {"test": _single(0.0900, 0.0500)},
                     "delta_pop_minus_random": {"recall": 0.0145, "ndcg": 0.0098}},
        "full_prior": {"base": "cos", "best_beta": 0.2, "val_by_beta": {"0.0": 0.0752, "0.2": 0.1338},
                       "val_seed0": _single(0.1338, 0.0840), "test": _agg(0.1328, 0.0854, 0.0002)},
        "decomposition_fixed_beta": {"beta_star": 0.2, "base": "cos",
                                     "recall": {"from_seed_means": {"delta_both": 0.0573}, "per_seed_mean_sd": {},
                                                "cells": {"base": 0.0755, "mu": 0.11, "eps": 0.09, "both": 0.1328}},
                                     "ndcg": {"from_seed_means": {}, "per_seed_mean_sd": {},
                                              "cells": {"base": 0.0402, "mu": 0.07, "eps": 0.05, "both": 0.0854}}},
        "decomposition_retuned": {"recall": {"cells": {}}, "ndcg": {"cells": {}},
                                  "cells": {"mu": {"best_beta": 0.1, "val_by_beta": {"0.1": 0.105},
                                                   "test": _agg(0.104, 0.06)},
                                            "eps": {"best_beta": 0.05, "val_by_beta": {"0.05": 0.09},
                                                    "test": _agg(0.089, 0.05)},
                                            "both": {"best_beta": 0.2, "test": _agg(0.1328, 0.0854, 0.0002)},
                                            "base": {"best_beta": 0.0, "test": _agg(0.0755, 0.0402)}}},
        "within_bucket_shuffle": {"beta_star": 0.2, "n_seeds": 10, "test": _agg(0.10, 0.05, 0.001),
                                  "gain_vs_random_base_recall": {"shuffle": 0.0245, "full": 0.0573}},
        "raw_prior": {"base": "raw", "best_beta": 5.0, "val_by_beta": {"0.0": 0.0749, "5.0": 0.1155},
                      "test": _agg(0.1143, 0.0700, 0.0003)},
        "cos_prior_u8": {"beta_star": 0.2, "quantization": {"p_max": 7.0}, "test": _agg(0.1327, 0.0853, 0.0002),
                         "delta_vs_unquantized_recall": -0.0001},
        "tie_stats": {"split": "test", "base_beta0": {"share_users_boundary_tie": 0.6}},
        "per_user_files": {"files": []}, "selected_betas": {"full_prior": 0.2}}
    doc["cos_prior"] = doc["full_prior"]
    if e2e:
        doc["e2e"] = {"unmodified": {"val": _single(0.17, 0.10), "test": _single(0.1790, 0.1152)},
                      "len_shuffle_decile": {"val": _agg(0.16, 0.09), "test": _agg(0.17, 0.11)},
                      "len_shuffle_global": {"val": _agg(0.12, 0.06), "test": _agg(0.13, 0.08)},
                      "len_replace_fit": {"val": _single(0.16, 0.09), "test": _single(0.17, 0.11)}}
    return doc


def write_controls(path, doc=None, **kwargs):
    path.mkdir(parents=True, exist_ok=True)
    (path / "controls.json").write_text(json.dumps(doc if doc is not None else controls_doc(path.name, **kwargs)))
    return path


def test_controls_dir_ok(tmp_path):
    good = write_controls(tmp_path / "gowalla__mf__graphhash__directau__s0")
    result = vr.validate(good)
    assert result["ok"] and result["kind"] == "controls" and result["problems"] == []
    code, out = run_cli(good)
    assert code == 0 and out["ok"]
    assert vr.validate(write_controls(tmp_path / "with_e2e", e2e=True))["ok"]


@pytest.mark.parametrize("section", vr.CONTROL_SECTIONS)
def test_controls_missing_section_fails(tmp_path, section):
    doc = controls_doc()
    del doc[section]
    result = vr.validate(write_controls(tmp_path / "c", doc))
    assert not result["ok"] and any(section in p for p in result["problems"])


def test_controls_other_failures(tmp_path):
    doc = controls_doc()
    doc["status"] = "failed"
    assert any("status" in p for p in vr.validate(write_controls(tmp_path / "a", doc))["problems"])
    doc = controls_doc(e2e=True)
    del doc["e2e"]
    assert any("e2e" in p for p in vr.validate(write_controls(tmp_path / "b", doc))["problems"])
    doc = controls_doc(alias_assert={"equal": False})
    assert any("alias" in p for p in vr.validate(write_controls(tmp_path / "c", doc))["problems"])
    doc = controls_doc()
    doc["legacy_topk"]["cos"]["crosscheck_vs_repro_core_metrics_from_topk"]["max_abs_diff"] = 0.01
    assert any("cross-check" in p for p in vr.validate(write_controls(tmp_path / "d", doc))["problems"])
    assert not vr.validate(write_controls(tmp_path / "e", {}))["ok"]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not vr.validate(empty, controls=True)["ok"]
    assert run_cli(tmp_path / "a", "--controls")[0] == 1


# ============================================================================ aggregate on synthetic runs
def rows_for(tmp_path, **kwargs):
    runs = tmp_path / "runs"
    runs.mkdir(exist_ok=True)
    rows, warnings = agg.build_rows(runs, tmp_path / "controls", tmp_path / "results", include_legacy=False, **kwargs)
    return rows, warnings


def test_aggregate_new_run_rows(tmp_path):
    make_run(tmp_path / "runs", slug="pc-s2a-1017")
    rows, warnings = rows_for(tmp_path)
    assert warnings == []
    assert [r["scoring_mode"] for r in rows] == ["raw", "cosine", "cos_prior", "raw_prior"]
    cp = rows[2]
    assert cp["run_id"] == "gowalla__mf__graphhash__directau__s0" and cp["status"] == "valid"
    assert (cp["dataset"], cp["compression"], cp["loss"], cp["seed"], cp["init"], cp["resolution"]) == \
        ("gowalla", "graphhash", "directau", 0, "xavier", 200)
    assert cp["beta"] == 0.2 and cp["test_recall20"] == pytest.approx(13.26) and cp["val_recall20"] == pytest.approx(13.36)
    assert cp["test_ndcg20"] == pytest.approx(6.63) and cp["stop_reason"] == "patience"
    assert (cp["best_epoch"], cp["epochs_run"], cp["table_params"]) == (10, 15, 742272)
    assert cp["train_min"] == pytest.approx(10.0) and cp["session_slug"] == "pc-s2a-1017"
    # norm_log.csv (long format) at best_epoch = 10
    assert cp["rho_item"] == pytest.approx(0.27) and cp["rho_bucket"] == pytest.approx(0.55)
    assert cp["rho_dir"] == pytest.approx(0.65)
    assert cp["extra_bytes"] == 4 * 50 and rows[0]["extra_bytes"] == 0  # 50 items in the synthetic mapping
    assert cp["test_std_recall20"] is None
    assert cp["mapping_sha256"].startswith("users=") and ";items=" in cp["mapping_sha256"]
    assert cp["comparison_type"] == "reproduction"
    assert cp["val_gap"] == pytest.approx(7.46 - 7.52) and cp["test_gap"] == pytest.approx(7.56 - 7.58)


def test_norm_log_falls_back_to_last_epoch_and_empty_values(tmp_path):
    d = make_run(tmp_path / "runs", best_epoch=7)  # 7 is not a logged epoch -> last logged epoch (15)
    rows, _ = rows_for(tmp_path)
    assert rows[2]["rho_item"] == pytest.approx(0.30)
    # a Full model has an empty rho_bucket value
    log = (d / "norm_log.csv").read_text().replace("15,rho_bucket,,11,,,0.6", "15,rho_bucket,,,,,")
    (d / "norm_log.csv").write_text(log)
    rows, _ = rows_for(tmp_path)
    assert rows[2]["rho_bucket"] is None and rows[2]["rho_dir"] == pytest.approx(0.70)


def test_aggregate_status_from_validation(tmp_path):
    make_run(tmp_path / "runs", name="gowalla__mf__graphhash__directau__s0", stop="time_cap")
    make_run(tmp_path / "runs", name="gowalla__mf__graphhash__directau__s1", stop="max_epochs",
             cfg_extra={"args": {"seeds": [1]}})
    make_run(tmp_path / "runs", name="gowalla__mf__graphhash__directau__s2", exit_code=1,
             cfg_extra={"args": {"seeds": [2]}})
    make_run(tmp_path / "runs", name="gowalla__mf__graphhash__directau__s3", cfg_extra={"args": {"seeds": [3]}})
    make_run(tmp_path / "runs" / "_invalid", name="gowalla__mf__graphhash__directau__s4",
             cfg_extra={"args": {"seeds": [4]}})
    rows, warnings = rows_for(tmp_path)
    status = {r["run_id"][-2:]: r["status"] for r in rows}
    assert status == {"s0": "capped", "s1": "capped", "s2": "invalid", "s3": "invalid", "s4": "invalid"} or \
        status["s0"] == "capped"  # s3/s4 carry a seed mismatch only if their config seed disagrees
    assert status["s0"] == "capped" and status["s1"] == "capped" and status["s2"] == "invalid" and status["s4"] == "invalid"
    assert any("exit_code" in w for w in warnings)


def test_aggregate_incomplete_run_still_listed_as_invalid(tmp_path):
    d = tmp_path / "runs" / "yelp2018__mf__full__bpr__s0"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({"status": "running", "seeds": {}}))
    rows, _ = rows_for(tmp_path)
    assert len(rows) == 1 and rows[0]["status"] == "invalid" and rows[0]["scoring_mode"] is None
    assert (rows[0]["dataset"], rows[0]["compression"], rows[0]["loss"]) == ("yelp2018", "full", "bpr")


def test_comparison_type_and_resolution(tmp_path):
    def run(name, args):
        make_run(tmp_path / "runs", name=name, cfg_extra={"args": args})
    run("gowalla__mf__full__directau__s0", {"dataset": "Gowalla", "method": "Full", "lr": 0.01, "wd": 1e-8})
    run("gowalla__mf__graphhash__directau__s0-initmatch", {"lr": 0.01, "wd": 1e-8, "init_match": True})
    run("gowalla__mf__graphhash__directau__s0", {"lr": 0.01, "wd": 1e-8})
    run("gowalla__mf__graphhash__directau__s0-r400", {"lr": 0.01, "wd": 1e-8})
    run("yelp2018__mf__full__directau__s0", {"lr": 0.01, "wd": 1e-8})
    run("gowalla__mf__graphhash__bpr__s0", {"lr": 0.01, "wd": 1e-6})
    run("gowalla__mf__random__directau__s0", {"lr": 0.01, "wd": 1e-6, "comparison_type": "controlled"})
    rows, _ = rows_for(tmp_path)
    by = {r["run_id"]: r for r in rows if r["scoring_mode"] == "raw"}
    assert by["gowalla__mf__full__directau__s0"]["comparison_type"] == "controlled"
    assert by["gowalla__mf__graphhash__directau__s0-initmatch"]["comparison_type"] == "controlled"
    assert by["gowalla__mf__graphhash__directau__s0-initmatch"]["init"] == "initmatch"
    assert by["gowalla__mf__graphhash__directau__s0"]["comparison_type"] == "reproduction"
    assert by["gowalla__mf__graphhash__directau__s0-r400"]["resolution"] == 400
    assert by["yelp2018__mf__full__directau__s0"]["comparison_type"] == "reproduction"
    assert by["gowalla__mf__graphhash__bpr__s0"]["comparison_type"] == "reproduction"
    assert by["gowalla__mf__full__directau__s0"]["resolution"] is None
    assert by["gowalla__mf__random__directau__s0"]["comparison_type"] == "controlled"  # explicit config wins


def test_aggregate_bias_loss_modes(tmp_path):
    make_run(tmp_path / "runs", name="gowalla__mf__graphhash__directau_bias__s0", bias=True)
    rows, _ = rows_for(tmp_path)
    assert [r["scoring_mode"] for r in rows] == ["raw", "cosine", "cos_prior", "raw_prior", "cos_b", "cos_b_prior"]
    assert rows[4]["extra_bytes"] == 200 and rows[0]["extra_bytes"] == 0


def test_aggregate_controls_rows(tmp_path):
    rid = "gowalla__mf__graphhash__directau__s0"
    write_controls(tmp_path / "controls" / rid)
    rows, warnings = rows_for(tmp_path)
    assert warnings == []
    modes = {r["scoring_mode"]: r for r in rows}
    expected = {"controls/legacy_topk/cos/beta0", "controls/legacy_topk/cos/prior", "controls/legacy_topk/raw/beta0",
                "controls/legacy_topk/raw/prior", "controls/tie_keys/random", "controls/tie_keys/index",
                "controls/tie_keys/pop", "controls/decomposition_fixed_beta/base",
                "controls/decomposition_fixed_beta/mu", "controls/decomposition_fixed_beta/eps",
                "controls/decomposition_fixed_beta/both", "controls/decomposition_retuned/base",
                "controls/decomposition_retuned/mu", "controls/decomposition_retuned/eps",
                "controls/decomposition_retuned/both", "controls/within_bucket_shuffle", "controls/cos_prior",
                "controls/raw_prior", "controls/cos_prior_u8"}
    assert set(modes) == expected and len(rows) == len(expected)   # full_prior is not repeated (== cos_prior)
    assert all(r["run_id"] == rid and r["status"] == "valid" and r["dataset"] == "gowalla" for r in rows)
    rnd = modes["controls/tie_keys/random"]
    assert (rnd["test_recall20"], rnd["test_ndcg20"], rnd["test_std_recall20"], rnd["beta"]) == \
        (pytest.approx(7.55), pytest.approx(4.02), pytest.approx(0.04), 0.0)
    assert rnd["val_recall20"] == pytest.approx(7.52) and rnd["extra_bytes"] == 0
    leg = modes["controls/legacy_topk/cos/prior"]
    assert (leg["beta"], leg["test_recall20"], leg["val_recall20"]) == (0.2, pytest.approx(13.30), pytest.approx(13.38))
    cp = modes["controls/cos_prior"]
    assert cp["beta"] == 0.2 and cp["val_recall20"] == pytest.approx(13.38) and cp["val_ndcg20"] == pytest.approx(8.40)
    assert cp["test_recall20"] == pytest.approx(13.28) and cp["test_std_recall20"] == pytest.approx(0.02)
    assert cp["extra_bytes"] == 4 * 40981 and cp["table_params"] == 742272
    assert modes["controls/raw_prior"]["val_recall20"] == pytest.approx(11.55)
    mu = modes["controls/decomposition_fixed_beta/mu"]
    assert (mu["test_recall20"], mu["test_ndcg20"], mu["beta"]) == (pytest.approx(11.0), pytest.approx(7.0), 0.2)
    assert mu["extra_bytes"] == 4 * 5806 and modes["controls/decomposition_fixed_beta/base"]["beta"] == 0.0
    assert modes["controls/decomposition_fixed_beta/eps"]["extra_bytes"] == 4 * 40981
    assert modes["controls/decomposition_retuned/mu"]["beta"] == 0.1
    assert modes["controls/decomposition_retuned/mu"]["val_recall20"] == pytest.approx(10.5)
    assert modes["controls/cos_prior_u8"]["extra_bytes"] == 40981
    assert modes["controls/within_bucket_shuffle"]["test_std_recall20"] == pytest.approx(0.1)
    assert rnd["mapping_sha256"] == f"users={'aa' * 32};items={'bb' * 32}"


def test_aggregate_controls_inherit_run_status_and_e2e(tmp_path):
    rid = "gowalla__mf__full__directau__s0"
    make_run(tmp_path / "runs", name=rid, stop="time_cap")
    write_controls(tmp_path / "controls" / rid, controls_doc(rid, e2e=True, method="Full"))
    rows, _ = rows_for(tmp_path)
    ctl = [r for r in rows if r["scoring_mode"].startswith("controls/")]
    assert {r["status"] for r in ctl} == {"capped"} and {r["comparison_type"] for r in ctl} == {"controlled"}
    e2e = {r["scoring_mode"]: r for r in ctl if "e2e" in r["scoring_mode"]}
    assert set(e2e) == {"controls/e2e/unmodified", "controls/e2e/len_shuffle_decile", "controls/e2e/len_shuffle_global",
                        "controls/e2e/len_replace_fit"}
    assert e2e["controls/e2e/unmodified"]["test_recall20"] == pytest.approx(17.90)
    assert e2e["controls/e2e/len_shuffle_global"]["val_recall20"] == pytest.approx(12.0)


def test_aggregate_invalid_controls_are_marked_invalid(tmp_path):
    doc = controls_doc("gowalla__mf__graphhash__directau__s0")
    del doc["tie_stats"]
    write_controls(tmp_path / "controls" / "c1", doc)
    rows, warnings = rows_for(tmp_path)
    assert rows and {r["status"] for r in rows} == {"invalid"}
    assert any("tie_stats" in w for w in warnings)


def test_aggregate_popularity_only_from_results_dir(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "A6_poponly_Yelp2018.json").write_text(
        json.dumps({"dataset": "Yelp2018", "val": {"recall": 1.61, "ndcg": 0.97}, "test": {"recall": 1.68, "ndcg": 1.05}}))
    rows, _ = rows_for(tmp_path)
    assert len(rows) == 1 and rows[0]["run_id"] == "yelp2018__poponly__none__none__s0"
    assert rows[0]["test_recall20"] == pytest.approx(1.68) and rows[0]["scoring_mode"] == "pop_only"
    assert rows[0]["extra_bytes"] == 4 * 38048


def test_aggregate_cli_writes_csv(tmp_path):
    make_run(tmp_path / "runs")
    out = tmp_path / "out" / "master.csv"
    proc = subprocess.run([sys.executable, str(HERE / "aggregate.py"), "--out", str(out), "--runs", str(tmp_path / "runs"),
                           "--controls", str(tmp_path / "controls"), "--results", str(tmp_path / "results"),
                           "--registry", str(tmp_path / "missing_registry.csv")], capture_output=True, text=True)
    # a missing registry is an error: the legacy rows must never silently disappear
    assert proc.returncode != 0
    reg = tmp_path / "empty_registry.csv"
    reg.write_text("file,dir,sub,run_id,status,role,parser,session_slug,stop_reason,note\n")
    proc = subprocess.run([sys.executable, str(HERE / "aggregate.py"), "--out", str(out), "--runs", str(tmp_path / "runs"),
                           "--controls", str(tmp_path / "controls"), "--results", str(tmp_path / "results"),
                           "--registry", str(reg)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    with open(out, newline="", encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert len(written) == 4 and written[2]["test_recall20"] == "13.26"


def test_cli_fails_on_a_missing_registered_legacy_file(tmp_path):
    reg = tmp_path / "reg.csv"
    reg.write_text("file,dir,sub,run_id,status,role,parser,session_slug,stop_reason,note\n"
                   "nope.json,results,,gowalla__mf__graphhash__directau__s0,valid-short,primary,exp,,fixed_epochs,x\n")
    base = [sys.executable, str(HERE / "aggregate.py"), "--out", str(tmp_path / "o.csv"), "--runs", str(tmp_path / "r"),
            "--controls", str(tmp_path / "c"), "--results", str(tmp_path / "res"), "--legacy-dir", str(tmp_path / "leg"),
            "--registry", str(reg)]
    assert subprocess.run(base, capture_output=True, text=True).returncode != 0
    lenient = subprocess.run(base + ["--lenient"], capture_output=True, text=True)
    assert lenient.returncode == 0 and "legacy file missing" in lenient.stderr


def test_run_id_outside_the_naming_rule_falls_back_to_the_config(tmp_path):
    make_run(tmp_path / "runs", name="x__A0smoke", seed=0, cfg_extra={"args": {"dataset": "Gowalla", "method": "Full"}})
    rows, _ = rows_for(tmp_path)
    assert len(rows) == 4 and {r["run_id"] for r in rows} == {"x__A0smoke"}
    r = rows[0]
    assert (r["dataset"], r["compression"], r["loss"], r["seed"], r["init"]) == ("gowalla", "full", "directau", 0, "xavier")
    assert r["resolution"] is None and r["status"] == "valid"
