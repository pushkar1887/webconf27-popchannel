"""Local CPU unit tests for the T1 patch of exp_run.py (synthetic data only: no clustering, no Kaggle).

python test_t1.py      (test_exp.py covers the legacy scoring paths and must also keep passing)
"""
import ast
import csv
import importlib
import inspect
import io
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "2026-10-16-loss-sharing-repro" / "_work"
tmp = Path(tempfile.mkdtemp())          # repro_run creates ROOT/_work at import: use a throw-away copy
for name in ["repro_core.py", "repro_run.py"]:
    shutil.copy(SRC/name, tmp/name)
shutil.copy(HERE/"exp_run.py", tmp/"exp_run.py")
sys.path.insert(0, str(tmp))
sys.path.insert(1, str(SRC/"GraphHash"))
E = importlib.import_module("exp_run")
R = E.R
from models.mf import MF
from models.mfdau import MFDAU
from loss import BPRLoss, DAULoss

torch.manual_seed(0)
np.random.seed(0)

BASE_ARGV = ["--dataset", "Gowalla", "--method", "GraphHash", "--loss", "dau", "--seeds", "3", "--epochs", "6",
             "--eval-every", "2", "--lr", "0.01", "--wd", "1e-8"]


# ----------------------------------------------------------------------------- synthetic context
def make_ctx(two_d=False, nu=30, ni=40, n_edges=400, buckets=(8, 10), seed=0):
    g = torch.Generator().manual_seed(seed)
    edges = torch.stack([torch.randint(nu, (n_edges,), generator=g), torch.randint(ni, (n_edges,), generator=g)])
    cut = int(n_edges*.75)
    tr, va, te = {}, {}, {}
    for d, (a, b) in zip((tr, va, te), ((0, cut), (cut, cut+n_edges//8), (cut+n_edges//8, n_edges))):
        for u, i in edges[:, a:b].t().tolist():
            d.setdefault(u, []).append(i)
    pop = torch.bincount(edges[1, :cut], minlength=ni).float()
    ua = torch.randint(buckets[0], (nu, 2) if two_d else (nu,), generator=g)
    ia = torch.randint(buckets[1], (ni, 2) if two_d else (ni,), generator=g)
    info = {"users": nu, "items": ni, "interactions": n_edges, "split_counts": [cut, n_edges//8, n_edges-cut-n_edges//8],
            "user_buckets": buckets[0], "item_buckets": buckets[1], "compressed_parameters": sum(buckets)*64,
            "full_parameters": (nu+ni)*64}
    return dict(MF=MF, MFDAU=MFDAU, BPRLoss=BPRLoss, DAULoss=DAULoss, parts=[edges[:, :cut]], info=info,
                train_truth=tr, val_truth=va, test_truth=te,
                merged={u: tr.get(u, [])+va.get(u, []) for u in set(tr)|set(va)},
                device=torch.device("cpu"), aliases=(ua, ia), nu=nu, ni=ni, counts=pop.numpy(), logpop=torch.log1p(pop))


def make_args(run_id, **kw):
    d = dict(loss="dau", tag=run_id, run_id=run_id, epochs=6, eval_every=2, patience=None, min_delta=0.0, cap_min=75.,
             lr=.01, wd=1e-8, batch=64, method="GraphHash", init_match=False, equal_item_norm=False)
    d.update(kw)
    return SimpleNamespace(**d)


def run_dir_of(run_id):
    return E.R.ROOT/"runs"/run_id


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def scripted_run(run_id, vals, ctx=None, **kw):
    """run_seed with val_recall replaced by a scripted sequence. Returns (row, number of checkpoint saves)."""
    ctx = ctx or make_ctx()
    it = iter(vals)
    real_val, real_save = E.val_recall, E.torch.save
    saves = []
    E.val_recall = lambda model, aliases, b, c: next(it)
    E.torch.save = lambda *a, **k: (saves.append(1), real_save(*a, **k))[1]
    try:
        args = make_args(run_id, **kw)
        result = {"seeds": {}}
        E.run_seed(args, 0, ctx, result, tmp/f"{run_id}.json")
    finally:
        E.val_recall, E.torch.save = real_val, real_save
    return result["seeds"]["0"], len(saves)


# ----------------------------------------------------------------------------- (b) min_delta
def test_patience_class_min_delta():
    # best_for_patience semantics: the gain is measured against the value at the last COUNTED improvement.
    p = E.Patience(patience=3, min_delta=0.01)
    seq = [(1, .10), (2, .108), (3, .1105), (4, .1106), (5, .1107), (6, .1108)]
    stops = [p.update(v, e) for e, v in seq]
    assert stops == [False, False, False, False, False, True], stops      # epoch 3 counted (+.0105 over .10)
    assert p.epoch == 3 and abs(p.best-.1105) < 1e-12
    p = E.Patience(patience=3, min_delta=0.01)                            # never counted: stops 3 evals after the first
    assert [p.update(v, e) for e, v in [(1, .10), (2, .105), (3, .108), (4, .1095)]] == [False, False, False, True]
    # a gain of exactly min_delta counts (float noise tolerated)
    p = E.Patience(patience=5, min_delta=0.005)
    p.update(.10, 1)
    p.update(.105, 2)
    assert p.epoch == 2
    # patience=None never stops
    p = E.Patience(None, 0.5)
    assert not any(p.update(.1, e) for e in range(1, 50))


def test_patience_default_equals_legacy_rule():
    """min_delta=0 must reproduce the legacy rule: patience from the last STRICT improvement of the best value."""
    rng = np.random.RandomState(0)
    for _ in range(200):
        vals = np.round(rng.rand(30)*3).astype(int)/10.            # many ties
        pat = int(rng.randint(1, 6))
        best, best_ep, legacy_stop = -1., 0, None
        for e, v in enumerate(vals, 1):
            if v > best:
                best, best_ep = v, e
            if pat and e-best_ep >= pat:
                legacy_stop = e
                break
        p, new_stop = E.Patience(pat, 0.0), None
        for e, v in enumerate(vals, 1):
            if p.update(float(v), e):
                new_stop = e
                break
        assert new_stop == legacy_stop, (vals, pat, new_stop, legacy_stop)


def test_min_delta_in_training_loop():
    # checkpoint on ANY strict gain, patience only on gains >= min_delta
    row, saves = scripted_run("md1", [.10, .105, .108, .1095, .1096, .1097, .1098, .1099], epochs=8, eval_every=1,
                              patience=3, min_delta=0.01)
    assert row["stop_reason"] == row["stopped"] == "patience"
    assert row["epochs_run"] == 4 and row["best_epoch"] == 4 and abs(row["best_val"]-.1095) < 1e-12
    assert saves == 4                      # every evaluation was a strict gain -> 4 checkpoints
    # same series, min_delta=0 (legacy): every gain counts, so no early stop
    row, saves = scripted_run("md2", [.10, .105, .108, .1095, .1096, .1097, .1098, .1099], epochs=8, eval_every=1,
                              patience=3, min_delta=0.0)
    assert row["stop_reason"] == "max_epochs" and row["epochs_run"] == 8 and row["best_epoch"] == 8 and saves == 8
    # gain measured against best_for_patience (not the last value / not best_val): counted at epoch 3 -> stop at 6
    row, _ = scripted_run("md3", [.10, .108, .1105, .1106, .1107, .1108, .1109, .111], epochs=8, eval_every=1,
                          patience=3, min_delta=0.01)
    assert row["stop_reason"] == "patience" and row["epochs_run"] == 6 and row["best_epoch"] == 6
    # a flat series stops by patience from the first evaluation, checkpoint stays at epoch 1
    row, saves = scripted_run("md4", [.2]*8, epochs=8, eval_every=1, patience=3, min_delta=0.0)
    assert row["stop_reason"] == "patience" and row["epochs_run"] == 4 and row["best_epoch"] == 1 and saves == 1


def test_stop_reasons():
    row, _ = scripted_run("sr1", [.1, .2, .3], epochs=6, eval_every=2)
    assert row["stop_reason"] == "max_epochs" and row["epochs_run"] == 6
    row, _ = scripted_run("sr2", [.1, .2, .3], epochs=6, eval_every=2, cap_min=-1.)   # cap exceeded after epoch 1
    assert row["stop_reason"] == "time_cap" and row["epochs_run"] == 1
    row, _ = scripted_run("sr3", [.1, .0, .0], epochs=6, eval_every=2, patience=2)
    assert row["stop_reason"] == "patience" and row["epochs_run"] == 4


# ----------------------------------------------------------------------------- (c) norm_log
def test_popularity_quintiles():
    counts = np.array([5, 1, 1, 1, 3, 3, 9, 0, 2, 2])
    q = E.popularity_quintiles(counts)
    # rank order (count, item index): 7 | 1 2 3 8 9 4 5 0 6 -> groups of two
    assert q.tolist() == [5, 1, 2, 2, 4, 4, 5, 1, 3, 3]
    for ni in (5, 10, 12, 13, 1001):
        c = np.random.RandomState(ni).randint(0, 7, ni)
        qq = E.popularity_quintiles(c)
        sizes = np.bincount(qq, minlength=6)[1:]
        assert sizes.sum() == ni and sizes.max()-sizes.min() <= 1
        assert all(c[qq == g].max() <= c[qq == g+1].min() for g in range(1, 5))     # monotone in popularity


def _rows_by_kind(rows):
    return {(r["kind"], r.get("q")): r for r in rows}


def test_norm_stats_values_and_rho():
    counts = np.array([50, 10, 3, 8, 40, 25, 90, 1, 12, 7], dtype=float)           # all distinct
    order = np.argsort(counts)
    rank = np.empty(10, int); rank[order] = np.arange(10)
    norm = (rank+1).astype(float)                                                   # norm monotone in count
    ivec = torch.zeros(10, 4); ivec[:, 0] = torch.tensor(norm, dtype=torch.float32)
    uvec = torch.randn(6, 4)
    alias = torch.tensor([0, 0, 1, 1, 1, 2, 2, 0, 1, 2])                            # bucket 3 stays EMPTY
    n_b = np.bincount(alias.numpy(), weights=counts, minlength=4)
    bw = torch.zeros(4, 4); bw[:, 0] = torch.tensor(np.argsort(np.argsort(n_b)) + 1., dtype=torch.float32)
    bw[3, 0] = 1000.0                                                               # empty bucket must be ignored
    bw[:3, 0] = torch.tensor(np.argsort(np.argsort(n_b[:3])) + 1., dtype=torch.float32)
    rows = E.norm_stats(uvec, ivec, bw, alias, counts, "GraphHash")
    by = _rows_by_kind(rows)
    for g in range(1, 6):
        r = by[("quintile", g)]
        assert r["n_items"] == 2 and abs(r["mean_norm"]-(2*g-.5)) < 1e-6 and abs(r["median_norm"]-(2*g-.5)) < 1e-6, r
    assert abs(by[("rho_item", None)]["value"]-1.0) < 1e-9 and by[("rho_item", None)]["n_items"] == 10
    assert abs(by[("rho_bucket", None)]["value"]-1.0) < 1e-9 and by[("rho_bucket", None)]["n_items"] == 3
    assert set(r["kind"] for r in rows) == {"quintile", "rho_item", "rho_bucket", "rho_dir"} and len(rows) == 8
    # reversed norms -> -1
    ivec2 = ivec.clone(); ivec2[:, 0] = ivec2[:, 0].max()+1-ivec2[:, 0]
    assert abs(_rows_by_kind(E.norm_stats(uvec, ivec2, bw, alias, counts, "GraphHash"))[("rho_item", None)]["value"]+1) < 1e-9
    # rho_bucket empty for Full and for 2-D item aliases
    assert _rows_by_kind(E.norm_stats(uvec, ivec, bw, alias, counts, "Full"))[("rho_bucket", None)]["value"] is None
    alias2 = torch.stack([alias, alias], dim=1)
    assert _rows_by_kind(E.norm_stats(uvec, ivec, bw, alias2, counts, "DoubleFreq"))[("rho_bucket", None)]["value"] is None


def test_rho_dir_uses_direction_not_length():
    counts = np.array([50, 10, 3, 8, 40, 25, 90, 1, 12, 7], dtype=float)
    order = np.argsort(counts)
    rank = np.empty(10, int); rank[order] = np.arange(10)
    uvec = torch.tensor([[1., 0, 0, 0], [3., 0, 0, 0], [.5, 0, 0, 0]])              # mean direction = e1
    ivec = torch.zeros(10, 4)
    ivec[:, 0] = 1.0
    ivec[:, 1] = torch.tensor(10.-rank, dtype=torch.float32)                         # more popular -> closer to e1
    alias, bw = torch.arange(10), torch.randn(10, 4)
    get = lambda iv: _rows_by_kind(E.norm_stats(uvec, iv, bw, alias, counts, "GraphHash"))[("rho_dir", None)]["value"]
    assert abs(get(ivec)-1.0) < 1e-9
    scale = torch.tensor(np.random.RandomState(0).rand(10, 1)*100+1, dtype=torch.float32)
    assert abs(get(ivec*scale)-1.0) < 1e-9                                           # lengths do not matter
    ivec[:, 1] = torch.tensor(rank-10., dtype=torch.float32)
    assert abs(get(ivec)-1.0) < 1e-9                                                 # sign of y does not matter either
    ivec[:, 1] = torch.tensor(rank*1., dtype=torch.float32)                          # less popular -> closer to e1
    assert abs(get(ivec)+1.0) < 1e-9


def test_norm_log_written_each_eval():
    ctx = make_ctx()
    for two_d in (False, True):
        for method in ("GraphHash", "Full"):
            c = make_ctx(two_d=two_d)
            if method == "Full":
                c["aliases"] = (torch.arange(c["nu"]), torch.arange(c["ni"]))
            args = make_args(f"nl_{two_d}_{method}", method=method, epochs=6, eval_every=2)
            E.run_seed(args, 0, c, {"seeds": {}}, tmp/"x.json")
            rows = read_csv(run_dir_of(args.run_id)/"norm_log.csv")
            assert list(rows[0]) == ["epoch", "kind", "q", "n_items", "mean_norm", "median_norm", "value"]
            assert sorted({int(r["epoch"]) for r in rows}) == [2, 4, 6]
            assert len(rows) == 3*8
            e6 = [r for r in rows if r["epoch"] == "6"]
            assert [r["kind"] for r in e6] == ["quintile"]*5 + ["rho_item", "rho_bucket", "rho_dir"]
            assert [r["q"] for r in e6[:5]] == ["1", "2", "3", "4", "5"]
            assert sum(int(r["n_items"]) for r in e6[:5]) == c["ni"]
            assert all(r["value"] == "" for r in e6[:5]) and all(r["mean_norm"] == "" for r in e6[5:])
            rb = [r for r in e6 if r["kind"] == "rho_bucket"][0]
            assert (rb["value"] == "") == (two_d or method == "Full"), (two_d, method, rb)
            assert all(r["value"] != "" for r in e6 if r["kind"] in ("rho_item", "rho_dir"))


# ----------------------------------------------------------------------------- (j) init-match
def test_init_match_factors_gowalla():
    f_items, f_users = E.init_match_factor(5806, 40981), E.init_match_factor(5792, 29858)
    assert round(f_items, 3) == 0.378 and round(f_users, 3) == 0.442, (f_items, f_users)
    assert E.init_match_factor(10, 10) == 1.0


def test_init_match_equalises_rms_row_norm():
    torch.manual_seed(1)
    bu, bi, nu, ni = 300, 400, 3000, 5000
    m = MFDAU(bu, bi, 64)
    rms = lambda w: float(w.detach().norm(dim=1).pow(2).mean().sqrt())
    r_i0, r_u0 = rms(m.item_emb_table.weight), rms(m.user_emb_table.weight)
    assert abs(r_i0/np.sqrt(128/(bi+64))-1) < .03                                    # Xavier: E|row|^2 = 128/(B+64)
    out = E.apply_init_match(m, nu, ni)
    assert abs(out["items"]["factor"]-np.sqrt((bi+64)/(ni+64))) < 1e-12 and out["items"]["table_rows"] == bi
    assert abs(out["users"]["factor"]-np.sqrt((bu+64)/(nu+64))) < 1e-12 and out["users"]["n_ids"] == nu
    assert abs(rms(m.item_emb_table.weight)/r_i0-out["items"]["factor"]) < 1e-6
    assert abs(rms(m.user_emb_table.weight)/r_u0-out["users"]["factor"]) < 1e-6
    # now equal to what a Full table (N rows) would have
    assert abs(rms(m.item_emb_table.weight)/np.sqrt(128/(ni+64))-1) < .03
    assert abs(rms(m.user_emb_table.weight)/np.sqrt(128/(nu+64))-1) < .03


def test_init_match_in_run_seed():
    ctx = make_ctx(buckets=(8, 10))
    plain = make_args("im_plain", epochs=1, eval_every=1, lr=0.0, wd=0.0)
    matched = make_args("im_match", epochs=1, eval_every=1, lr=0.0, wd=0.0, init_match=True)
    full = make_args("im_full", epochs=1, eval_every=1, lr=0.0, wd=0.0, init_match=True, method="Full")
    res = {k: {"seeds": {}} for k in ("p", "m")}
    E.run_seed(plain, 5, ctx, res["p"], tmp/"p.json")
    E.run_seed(matched, 5, ctx, res["m"], tmp/"m.json")
    ckp = torch.load(run_dir_of("im_plain")/"best_seed5.pt", weights_only=True)["model"]
    ckm = torch.load(run_dir_of("im_match")/"best_seed5.pt", weights_only=True)["model"]
    fi, fu = E.init_match_factor(10, ctx["ni"]), E.init_match_factor(8, ctx["nu"])
    assert torch.allclose(ckm["item_emb_table.weight"], ckp["item_emb_table.weight"]*fi, rtol=1e-6)
    assert torch.allclose(ckm["user_emb_table.weight"], ckp["user_emb_table.weight"]*fu, rtol=1e-6)
    cfg = json.loads((run_dir_of("im_match")/"config.json").read_text())
    assert abs(cfg["init_match_factors"]["items"]["factor"]-fi) < 1e-12 and abs(cfg["init_match_factors"]["users"]["factor"]-fu) < 1e-12
    assert json.loads((run_dir_of("im_plain")/"config.json").read_text())["init_match_factors"] is None
    assert "init_match" in res["m"]["seeds"]["5"] and "init_match" not in res["p"]["seeds"]["5"]
    # Full table: --init-match is a no-op
    fc = make_ctx(); fc["aliases"] = (torch.arange(fc["nu"]), torch.arange(fc["ni"]))
    r2 = {"seeds": {}}
    E.run_seed(full, 5, fc, r2, tmp/"f.json")
    assert "init_match" not in r2["seeds"]["5"]
    assert json.loads((run_dir_of("im_full")/"config.json").read_text())["init_match_factors"] is None


# ----------------------------------------------------------------------------- (k) equal item norm
def test_equalize_item_norms():
    t = torch.nn.Embedding(50, 8)
    w0 = t.weight.detach().clone()
    E.equalize_item_norms(t)
    n = t.weight.detach().norm(dim=1)
    assert float((n-n[0]).abs().max()) < 1e-5
    assert abs(float(n[0])-float(w0.norm(dim=1).mean())) < 1e-5                      # the mean row norm is preserved
    F = torch.nn.functional
    assert torch.allclose(F.normalize(t.weight.detach(), dim=1), F.normalize(w0, dim=1), atol=1e-6)   # directions kept
    t.weight.data[3] = 0.                                                            # a dead row stays finite
    E.equalize_item_norms(t)
    assert torch.isfinite(t.weight).all()


def test_equal_item_norm_in_run_seed():
    for two_d in (False, True):
        ctx = make_ctx(two_d=two_d)
        on = make_args(f"ein_on_{two_d}", equal_item_norm=True, epochs=4, eval_every=2)
        off = make_args(f"ein_off_{two_d}", epochs=4, eval_every=2)
        E.run_seed(on, 0, ctx, {"seeds": {}}, tmp/"a.json")
        E.run_seed(off, 0, ctx, {"seeds": {}}, tmp/"b.json")
        w_on = torch.load(run_dir_of(on.run_id)/"best_seed0.pt", weights_only=True)["model"]
        w_off = torch.load(run_dir_of(off.run_id)/"best_seed0.pt", weights_only=True)["model"]
        n_on = w_on["item_emb_table.weight"].norm(dim=1)
        n_off = w_off["item_emb_table.weight"].norm(dim=1)
        assert float((n_on-n_on[0]).abs().max()) < 1e-4 * float(n_on[0]), n_on          # all item rows equal
        assert float(n_off.std()) > 1e-3                                                # not so without the flag
        u_on = w_on["user_emb_table.weight"].norm(dim=1)
        assert float(u_on.std()) > 1e-3                                                 # users untouched


# ----------------------------------------------------------------------------- (e) per-user arrays
def test_sweep_per_user_matches_sweep():
    gen = torch.Generator().manual_seed(2)
    nu, ni = 40, 60
    uvec, ivec = torch.randn(nu, 6, generator=gen), torch.randn(ni, 6, generator=gen)
    mk = lambda off, n: {u: [(u*3+off+j) % ni for j in range(n + u % 3)] for u in range(nu)}
    tr, va, te = mk(0, 3), mk(10, 2), mk(20, 4)
    te[7] = []                                                                          # a user without positives
    merged = {u: tr[u]+va[u] for u in tr}
    logpop = torch.rand(ni, generator=gen)
    for cosine in (False, True):
        for beta in (0.0, .5, 2.0):
            ref = E.sweep(uvec, ivec, te, merged, [beta], logpop, None, cosine)[beta]
            pu = E.sweep_per_user(uvec, ivec, te, merged, beta, logpop, cosine, chunk=7)
            assert pu["users"].dtype == np.int32 and pu["top20"].dtype == np.int32 and pu["top20"].shape[1] == 20
            assert pu["hits"].dtype == np.int16 and pu["npos"].dtype == np.int16 and pu["ndcg"].dtype == np.float32
            assert 7 not in pu["users"].tolist() and len(pu["users"]) == nu-1
            recall = np.mean(pu["hits"]/np.minimum(20, pu["npos"]))
            assert abs(recall-ref["recall"]) < 1e-12 and abs(float(pu["ndcg"].astype(np.float64).mean())-ref["ndcg"]) < 1e-6
            assert (pu["hits"] <= 20).all() and (pu["npos"] > 0).all()
            # train+val items are masked out of every list
            assert not any(set(row.tolist()) & set(merged[int(u)]) for u, row in zip(pu["users"], pu["top20"]))


# ----------------------------------------------------------------------------- (d)(e)(f)(g)(h) run directory end to end
def _patched_main(ctx, argv, build=None):
    real = E.build_context
    E.build_context = build or (lambda args: ctx)
    try:
        return E.main(argv)
    finally:
        E.build_context = real


def test_main_run_directory_contract():
    ctx = make_ctx(two_d=False)
    E.os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    try:
        _patched_main(ctx, BASE_ARGV + ["--run-id", "e2e1", "--tag", "e2e1tag", "--patience", "50"])
    finally:
        del E.os.environ["CUDA_VISIBLE_DEVICES"]
    d = run_dir_of("e2e1")
    names = {p.name for p in d.iterdir()}
    want = {"config.json", "mapping_users.npy", "mapping_items.npy", "train_log.csv", "norm_log.csv", "best_seed3.pt",
            "metrics.json", "exit_status.json"} | {f"per_user_test__{m}.npz" for m in ("raw", "cosine", "cos_prior", "raw_prior")}
    assert want <= names, want-names
    metrics = json.loads((d/"metrics.json").read_text())
    assert metrics["status"] == "complete" and metrics["tag"] == "e2e1tag"
    assert json.loads((E.R.ROOT/"exp_e2e1tag.json").read_text()) == metrics              # legacy file, same content
    row = metrics["seeds"]["3"]
    assert row["stop_reason"] == "max_epochs" and {"raw", "cosine", "cos_prior", "raw_prior"} <= set(row["eval"])
    for k in ("val_curve", "best_epoch", "best_val", "epochs_run", "stopped", "train_seconds", "batch",
              "spearman_norm_vs_count", "eval", "wall_seconds", "params"):
        assert k in row                                                                   # legacy schema intact
    ex = json.loads((d/"exit_status.json").read_text())
    assert ex["exit_code"] == 0 and ex["stop_reason"] == "max_epochs" and ex["error"] is None and ex["end_time"]
    # mappings: int64, equal to the aliases, sha256 matches config
    mu, mi = np.load(d/"mapping_users.npy"), np.load(d/"mapping_items.npy")
    assert mu.dtype == np.int64 and mi.dtype == np.int64
    assert (mu == ctx["aliases"][0].numpy()).all() and (mi == ctx["aliases"][1].numpy()).all()
    cfg = json.loads((d/"config.json").read_text())
    assert cfg["mapping_sha256"] == {"users": E.sha256_file(d/"mapping_users.npy"), "items": E.sha256_file(d/"mapping_items.npy")}
    assert cfg["exp_run_sha256"] == E.sha256_file(E.__file__) and cfg["torch_version"] == torch.__version__
    assert cfg["CUDA_VISIBLE_DEVICES"] == "1" and cfg["device_count"] == torch.cuda.device_count()
    assert "gpu_name" in cfg and cfg["args"]["run_id"] == "e2e1" and cfg["args"]["seeds"] == [3]
    assert cfg["resolved"]["lr"] == .01 and cfg["resolved"]["wd"] == 1e-8 and cfg["resolved"]["batch"] == 1024
    ds = cfg["dataset_stats"]
    assert ds["val_eval_users"] == len(ctx["val_truth"]) and ds["test_eval_users"] == len(ctx["test_truth"])
    assert ds["users"] == 30 and ds["items"] == 40 and ds["user_buckets"] == 8 and ds["item_buckets"] == 10
    assert ds["split_counts"] == ctx["info"]["split_counts"]
    # checkpoint keys
    ck = torch.load(d/"best_seed3.pt", weights_only=True)
    assert set(ck) == {"model", "b", "aliases"}
    # train_log: one row per epoch, val_recall only on eval epochs
    tl = read_csv(d/"train_log.csv")
    assert list(tl[0]) == ["epoch", "train_loss", "val_recall20", "elapsed_s"] and len(tl) == row["epochs_run"] == 6
    assert [r["val_recall20"] != "" for r in tl] == [False, True, False, True, False, True]
    assert all(np.isfinite(float(r["train_loss"])) for r in tl) and float(tl[-1]["elapsed_s"]) >= float(tl[0]["elapsed_s"])
    assert abs(float(tl[3]["val_recall20"])-row["val_curve"]["4"]) < 1e-12
    # per-user arrays at each mode's chosen beta reproduce the reported test recall/ndcg
    for mode in ("raw", "cosine", "cos_prior", "raw_prior"):
        z = np.load(d/f"per_user_test__{mode}.npz")
        assert set(z.files) == {"users", "top20", "hits", "npos", "ndcg"}
        assert z["users"].dtype == np.int32 and z["top20"].dtype == np.int32 and z["hits"].dtype == np.int16
        assert z["npos"].dtype == np.int16 and z["ndcg"].dtype == np.float32 and z["top20"].shape == (len(z["users"]), 20)
        t = row["eval"][mode]["test"]
        assert abs(np.mean(z["hits"]/np.minimum(20, z["npos"]))-t["recall"]) < 1e-9
        assert abs(float(z["ndcg"].astype(np.float64).mean())-t["ndcg"]) < 1e-6
        assert len(z["users"]) == len(ctx["test_truth"])
        assert row["per_user"][mode]["beta"] == row["eval"][mode]["best_beta"]
        assert row["per_user"][mode]["recall_diff_vs_eval_suite"] < 1e-9


def test_main_2d_aliases_contract():
    ctx = make_ctx(two_d=True)
    _patched_main(ctx, BASE_ARGV[:6] + ["--loss", "bpr_author", "--method", "DoubleFreq"] + BASE_ARGV[8:] +
                  ["--run-id", "e2e2", "--init-match", "--equal-item-norm"])
    d = run_dir_of("e2e2")
    assert np.load(d/"mapping_items.npy").shape == (40, 2)
    assert json.loads((d/"exit_status.json").read_text())["exit_code"] == 0
    assert json.loads((d/"metrics.json").read_text())["status"] == "complete"
    cfg = json.loads((d/"config.json").read_text())
    assert cfg["init_match_factors"]["items"]["table_rows"] == int(ctx["aliases"][1].max())+1


def test_failure_writes_exit_status():
    def boom(args):
        raise RuntimeError("boom-ctx")
    try:
        _patched_main(None, BASE_ARGV + ["--run-id", "fail1"], build=boom)
        raise AssertionError("main must re-raise")
    except RuntimeError:
        pass
    d = run_dir_of("fail1")
    ex = json.loads((d/"exit_status.json").read_text())
    assert ex["exit_code"] == 1 and "boom-ctx" in ex["error"] and ex["stop_reason"] is None and ex["end_time"]
    assert "boom-ctx" in (d/"failure.log").read_text() and not (d/"metrics.json").exists()
    # failure during training: metrics.json says "failed", never "complete"
    real = E.val_recall
    E.val_recall = lambda *a, **k: (_ for _ in ()).throw(ValueError("boom-train"))
    try:
        _patched_main(make_ctx(), BASE_ARGV + ["--run-id", "fail2"])
        raise AssertionError("main must re-raise")
    except ValueError:
        pass
    finally:
        E.val_recall = real
    d = run_dir_of("fail2")
    assert json.loads((d/"metrics.json").read_text())["status"] == "failed"
    ex = json.loads((d/"exit_status.json").read_text())
    assert ex["exit_code"] == 1 and "boom-train" in ex["error"]


def test_parse_rules():
    a = E.parse(BASE_ARGV + ["--run-id", "r1"])
    assert a.min_delta == 0.0 and a.resolution == 200 and a.init_match is False and a.equal_item_norm is False
    assert a.seeds == [3] and a.tag == "r1"
    a = E.parse(BASE_ARGV + ["--run-id", "r1", "--min-delta", "5e-4", "--resolution", "800", "--init-match",
                              "--equal-item-norm", "--tag", "legacy"])
    assert a.min_delta == 5e-4 and a.resolution == 800 and a.init_match and a.equal_item_norm and a.tag == "legacy"
    assert E.parse(["--dataset", "Yelp2018"] + BASE_ARGV[2:] + ["--run-id", "y"]).dataset == "Yelp2018"
    bad = [BASE_ARGV[:6] + ["--seeds", "0,1"] + BASE_ARGV[8:] + ["--run-id", "r"],     # two seeds
           BASE_ARGV, BASE_ARGV + ["--run-id", "../evil"], BASE_ARGV + ["--run-id", "a/b"],
           BASE_ARGV + ["--run-id", "r", "--min-delta", "-1"]]
    for argv in bad:
        try:
            E.parse(argv)
        except SystemExit as e:
            assert e.code == 2
        else:
            raise AssertionError(argv)


# ----------------------------------------------------------------------------- (a)(i) resolution plumbing
def test_resolution_reaches_graph_clustering():
    sig = inspect.signature(R.load_data)
    assert sig.parameters["resolution"].default == 200 and "Yelp2018" in inspect.getsource(R.prepare)
    nu, ni = 12, 15
    g = torch.Generator().manual_seed(3)
    d = R.WORK/"datasets"/"Fake-processed"
    d.mkdir(parents=True, exist_ok=True)
    n = max(nu, ni)                                    # train alone covers every user and item id
    parts = [torch.stack([torch.arange(n) % nu, torch.arange(n) % ni]),
             torch.stack([torch.randint(nu, (20,), generator=g), torch.randint(ni, (20,), generator=g)]),
             torch.stack([torch.randint(nu, (20,), generator=g), torch.randint(ni, (20,), generator=g)])]
    for s, p in zip(("train", "val", "test"), parts):
        torch.save(p, d/f"{s}.pt")
    seen = []

    def fake_clustering(graph, resolution):
        seen.append(resolution)
        return np.arange(nu) % 4, np.arange(ni) % 5, 4, 5
    ns = {"graphClustering": fake_clustering, "map_to_consecutive_integers": lambda x: np.asarray(x)}
    R.load_data("Fake", ns)
    R.load_data("Fake", ns, resolution=800)
    R.load_data("Fake", ns, 1600)
    assert seen == [200, 800, 1600], seen
    # and exp_run's build_context hands --resolution through
    got = {}
    real_prep, real_load = R.prepare, R.load_data

    def fake_load(dataset, namespace, resolution=200):
        got["res"] = resolution
        c = make_ctx()
        maps = {"Full": (np.arange(30), np.arange(40)), "GraphHash": tuple(a.numpy() for a in c["aliases"])}
        return c["parts"], maps, (c["train_truth"], c["val_truth"], c["test_truth"]), c["info"]
    R.prepare, R.load_data = (lambda: (MF, MFDAU, BPRLoss, DAULoss, {})), fake_load
    try:
        E.build_context(E.parse(BASE_ARGV + ["--run-id", "res1", "--resolution", "800"]))
        assert got["res"] == 800
        E.build_context(E.parse(BASE_ARGV + ["--run-id", "res2"]))
        assert got["res"] == 200
    finally:
        R.prepare, R.load_data = real_prep, real_load


# ============================================================================= T2: session bundles (job_queue.py, kapi.py)
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))
JQ = importlib.import_module("job_queue")

PROBE = b'''
import json, os, sys, time
from pathlib import Path
a = sys.argv
run_id = a[a.index("--run-id") + 1]
code = int(a[a.index("--exit") + 1]) if "--exit" in a else 0
nap = float(a[a.index("--sleep") + 1]) if "--sleep" in a else 0.0
start = time.time()
d = Path("runs") / run_id
d.mkdir(parents=True, exist_ok=True)
inp = Path("inputs/data.txt")
print("hello from", run_id, flush=True)
print("to stderr", file=sys.stderr, flush=True)
time.sleep(nap)
(d / "probe.json").write_text(json.dumps({"cvd": os.environ.get("CUDA_VISIBLE_DEVICES"), "start": start,
    "end": time.time(), "argv": a[1:], "input": inp.read_text() if inp.exists() else None, "cwd": os.getcwd()}))
sys.exit(code)
'''
FAKE_FILES = {"repro_core.py": b"", "repro_run.py": b"def prepare():\n    print('prepare ok')\n",
              "exp_run.py": b"", "probe.py": PROBE}


def pj(rid, **kw):
    a = " ".join(f"--{k.replace('_', '-')} {v}" for k, v in kw.items())
    return {"run_id": rid, "script": "probe.py", "args": f"--run-id {rid} {a}".strip()}


def run_entry(session, workdir, device_count, inputs_ok=True, files=None):
    """Build the kernel for `session`, load its run.py as a module (not as __main__) and run the session."""
    os.environ["SESSION_WORKING"], os.environ["SESSION_NO_AUTORUN"] = str(workdir), "1"
    try:
        slug, folder = JQ.build_session_kernel(session, "OWNER", files=files or FAKE_FILES, root=Path(workdir).parent/"k")
        ns = runpy.run_path(str(folder/"run.py"), run_name="entry_test")
        g = ns["run_session"].__globals__                               # (run_path returns a COPY of the globals)
        orig = g["download"]
        g["download"] = lambda u, d: orig(u, d, tries=1)              # no 20 s back-off in tests
        ns["run_session"](device_info={"device_count": device_count, "device_names": ["Fake T4"]*device_count})
    finally:
        del os.environ["SESSION_WORKING"], os.environ["SESSION_NO_AUTORUN"]
    return json.loads((Path(workdir)/"session_status.json").read_text())


def probe(workdir, rid):
    return json.loads((Path(workdir)/"runs"/rid/"probe.json").read_text())


def test_session_validation():
    ok = {"slug": "pc-t1-1017", "gpu_lists": [[pj("r1")], [pj("r2")]]}
    emb = set(FAKE_FILES)
    JQ.validate_sessions([ok], emb)
    bad_cases = {
        "slug": dict(ok, slug="PC bad"), "account": dict(ok, account="account-2"),
        "three lists": dict(ok, gpu_lists=[[pj("r1")], [pj("r2")], [pj("r3")]]),
        "script": dict(ok, gpu_lists=[[dict(pj("r1"), script="nope.py")]]),
        "parallel size": dict(ok, gpu_lists=[[{"parallel": [pj("r1")]}]]),
        "nested parallel": dict(ok, gpu_lists=[[{"parallel": [pj("r1"), {"parallel": [pj("r9"), pj("r8")]}]}]]),
        "dup run_id": dict(ok, gpu_lists=[[pj("r1")], [pj("r1")]]),
        "bad run_id": dict(ok, gpu_lists=[[pj("../x")]]),
        "input path": dict(ok, gpu_lists=[[dict(pj("r1"), inputs={"../x": "u"})]]),
        "exp_run run-id mismatch": dict(ok, gpu_lists=[[dict(pj("r1"), script="exp_run.py", args="--run-id other")]]),
    }
    for name, s in bad_cases.items():
        try:
            JQ.validate_sessions([s], emb)
        except ValueError:
            continue
        raise AssertionError(f"accepted: {name}")
    try:                                                                       # duplicate slug / run_id across sessions
        JQ.validate_sessions([ok, dict(ok, gpu_lists=[[pj("z")]])], emb)
        raise AssertionError("duplicate slug accepted")
    except ValueError:
        pass
    JQ.validate_sessions([ok, dict(ok, slug="pc-t2-1017", gpu_lists=[[pj("z1"), {"parallel": [pj("z2"), pj("z3")]}]])], emb)


def test_dry_run_builds_valid_kernel_folder():
    s = {"slug": "pc-dry-1017", "account": "account-4", "kernel_sources": ["src-kernel-1016", "other/abs-kernel"],
         "gpu_lists": [[pj("d1"), {"parallel": [pj("d2"), pj("d3")]}],
                       [dict(pj("d4"), inputs={"w.pt": "https://example.invalid/x?sig=SECRET"})]]}
    root = tmp/"dry"
    slug, folder = JQ.build_session_kernel(s, "OWNER", files=FAKE_FILES, root=root)
    assert slug == "pc-dry-1017" and folder == root/"pc-dry-1017"
    text = (folder/"run.py").read_text()
    compile(text, "run.py", "exec")
    chk = tmp/"pycheck"; chk.mkdir(exist_ok=True)
    shutil.copy(folder/"run.py", chk/"run_check.py")
    assert subprocess.run([sys.executable, "-m", "py_compile", str(chk/"run_check.py")]).returncode == 0
    meta = json.loads((folder/"kernel-metadata.json").read_text())
    assert meta["id"] == "OWNER/pc-dry-1017" and meta["code_file"] == "run.py" and meta["kernel_type"] == "script"
    assert meta["enable_gpu"] is True and meta["is_private"] is True
    assert meta["kernel_sources"] == ["OWNER/src-kernel-1016", "other/abs-kernel"]
    assert "SECRET" not in json.dumps(meta)
    _, f2 = JQ.build_session_kernel(s, "realuser9", files=FAKE_FILES, root=tmp/"dry2")
    assert "realuser9" in (f2/"kernel-metadata.json").read_text()
    JQ.scrub_metadata(f2, "realuser9")                                          # no username kept after a push
    scrubbed = (f2/"kernel-metadata.json").read_text()
    assert "realuser9" not in scrubbed and "OWNER/pc-dry-1017" in scrubbed
    summary = JQ.describe_session(s, folder)
    assert "pc-dry-1017" in summary and "GPU0" in summary and "GPU1" in summary and "(parallel)" in summary
    assert "SECRET" not in summary and "d1" in summary and "d4" in summary
    assert JQ.source_account(JQ.normalize_session(s)) == "account-4"
    assert JQ.source_account({"account": None, "kernel_sources": ["x"]}) == JQ.SOURCE_ACCOUNT
    assert JQ.source_account({"account": None, "kernel_sources": []}) is None
    # real embedded files: the three required ones exist, and the dry run of the real sessions file builds
    files = JQ.collect_files()
    assert {"repro_core.py", "repro_run.py", "exp_run.py"} <= set(files)


def test_entry_two_lists_in_parallel_one_per_gpu():
    wd = tmp/"e1"
    inp = tmp/"input_file.txt"
    inp.write_text("hello-input")
    s = {"slug": "pc-e1-1017", "prepare": True,
         "gpu_lists": [[pj("a1", sleep=1.2), pj("a2", sleep=0.3, exit=3), pj("a3", sleep=0.1)],
                       [dict(pj("b1", sleep=2.0), inputs={"data.txt": inp.as_uri()}),
                        {"parallel": [pj("p1", sleep=1.0), pj("p2", sleep=1.0)]}]]}
    st = run_entry(s, wd, device_count=2)
    assert st["state"] == "complete" and st["mode"] == "parallel_2gpu" and st["device_count"] == 2
    assert st["n_jobs"] == 6 and st["n_failed"] == 1 and st["failed_run_ids"] == ["a2"]
    assert st["jobs"]["a2"]["exit_code"] == 3 and st["jobs"]["a3"]["exit_code"] == 0   # a failure does not stop the list
    assert st["jobs"]["a1"]["duration_s"] >= 1.0 and st["jobs"]["a1"]["log"] == "logs/a1.log"
    assert st["prepare_exit_code"] == 0 and "prepare ok" in (wd/"logs"/"_prepare.log").read_text()
    info = json.loads((wd/"session_info.json").read_text())
    assert info["device_count"] == 2 and info["device_names"] == ["Fake T4", "Fake T4"] and info["start_time"]
    a1, a2, a3, b1, p1, p2 = (probe(wd, r) for r in ("a1", "a2", "a3", "b1", "p1", "p2"))
    assert {a1["cvd"], a2["cvd"], a3["cvd"]} == {"0"} and {b1["cvd"], p1["cvd"], p2["cvd"]} == {"1"}
    assert a2["start"] >= a1["end"]-0.05 and a3["start"] >= a2["end"]-0.05          # sequential within a list
    assert p1["start"] >= b1["end"]-0.05                                            # parallel group after b1
    assert b1["start"] < a1["end"] and a1["start"] < b1["end"]                      # the two lists overlap in time
    assert p1["start"] < p2["end"] and p2["start"] < p1["end"]                      # co-located pair overlaps
    assert st["jobs"]["p1"]["colocated_with"] == "p2" and st["jobs"]["p2"]["colocated_with"] == "p1"
    assert b1["input"] == "hello-input" and Path(b1["cwd"]).samefile(wd)                    # inputs under inputs/<name>
    assert (wd/"logs"/"a1.log").read_text().count("hello from a1") == 1 and "to stderr" in (wd/"logs"/"a1.log").read_text()
    assert not (wd/"inputs").exists()                                               # cleaned up at the end
    assert (wd/"_work"/"probe.py").read_bytes() == PROBE                            # embedded files written to _work


def test_entry_falls_back_to_gpu0_without_two_gpus():
    for n in (0, 1):
        wd = tmp/f"e2_{n}"
        s = {"slug": "pc-e2-1017", "prepare": False,
             "gpu_lists": [[pj("c1", sleep=0.6)], [pj("c2", sleep=0.1)]]}
        st = run_entry(s, wd, device_count=n)
        assert st["mode"] == "sequential_gpu0" and st["n_failed"] == 0 and "prepare_exit_code" not in st
        c1, c2 = probe(wd, "c1"), probe(wd, "c2")
        assert c1["cvd"] == c2["cvd"] == "0" and c2["start"] >= c1["end"]-0.05      # one after the other


def test_entry_failed_input_and_missing_script_are_recorded_not_raised():
    wd = tmp/"e3"
    s = {"slug": "pc-e3-1017", "prepare": False,
         "gpu_lists": [[dict(pj("f1"), inputs={"data.txt": (tmp/"does_not_exist.bin").as_uri()}), pj("f2")],
                       [{"run_id": "f3", "script": "exp_run.py", "args": "--run-id f3"}]]}
    st = run_entry(s, wd, device_count=2)          # exp_run.py is an empty file -> exit 0; f1 skipped
    assert st["state"] == "complete"
    assert st["inputs"]["data.txt"]["ok"] is False and "error" in st["inputs"]["data.txt"]
    assert st["jobs"]["f1"]["state"] == "skipped" and st["jobs"]["f1"]["exit_code"] is None
    assert st["jobs"]["f2"]["exit_code"] == 0 and st["jobs"]["f3"]["exit_code"] == 0
    assert "f1" in st["failed_run_ids"]


def test_entry_script_always_exits_zero():
    # (1) as a script, normal session; (2) corrupt embedded file -> entry error recorded; both exit code 0
    for name, files, want_state in (("ok", FAKE_FILES, "complete"), ("corrupt", None, "entry_error")):
        wd = tmp/f"e4_{name}"
        s = {"slug": "pc-e4-1017", "prepare": False, "gpu_lists": [[pj("g1")], [pj("g2")]]}
        slug, folder = JQ.build_session_kernel(s, "OWNER", files=FAKE_FILES, root=tmp/f"k4_{name}")
        if name == "corrupt":
            txt = (folder/"run.py").read_text().replace("'probe.py': '", "'probe.py': 12345, 'zzz.py': '")
            assert "'probe.py': 12345" in txt                                       # b64decode(12345) raises TypeError
            (folder/"run.py").write_text(txt)
        env = dict(os.environ, SESSION_WORKING=str(wd))
        p = subprocess.run([sys.executable, str(folder/"run.py")], env=env, capture_output=True, text=True, cwd=str(tmp))
        assert p.returncode == 0, p.stderr[-500:]
        st = json.loads((wd/"session_status.json").read_text())
        assert st["state"] == want_state, st
        if name == "ok":
            assert st["n_jobs"] == 2 and st["mode"] == "sequential_gpu0"             # local torch sees no 2 GPUs
            assert json.loads((wd/"session_info.json").read_text())["device_count"] == torch.cuda.device_count()
        else:
            assert st["errors"] and "TypeError" in st["errors"][0] and st["n_jobs"] == 0


def test_kapi_helpers():
    src = (HERE/"kapi.py").read_text(encoding="utf-8")
    compile(src, "kapi.py", "exec")
    tree = ast.parse(src)
    ns = {"SKIP_PREFIXES": ("datasets/", "_work/")}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("norm", "wanted"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "kapi_fn", "exec"), ns)
    pre = JQ.FETCH_PREFIXES
    assert ns["norm"]("/kaggle/working/runs/x/metrics.json") == "runs/x/metrics.json"
    assert ns["norm"]("runs\\x\\a.json") == "runs/x/a.json"
    ok = ["runs/x/metrics.json", "logs/x.log", "maps/m.npy", "controls/c.json", "session_info.json", "session_status.json"]
    no = ["_work/datasets/Gowalla-processed/train.pt", "datasets/x", "inputs/ckpt.pt", "runs_extra/a", "_work/exp_run.py",
          "random.txt"]
    assert all(ns["wanted"](r, pre) for r in ok) and not any(ns["wanted"](r, pre) for r in no)
    # kapi.py keeps whoami / push / status / fetch and sets KAGGLE_CONFIG_DIR before importing kaggle
    assert src.index('os.environ["KAGGLE_CONFIG_DIR"]') < src.index("from kaggle.api")
    assert all(f'cmd == "{c}"' in src for c in ("whoami", "push", "status", "fetch"))


class FakeKaggle:
    """Stands in for job_queue.kapi: 2 accounts, scripted statuses, fake output files."""
    def __init__(self, statuses, full=(), fetch_fail=()):
        self.owner = {"account-3": "secretuser3", "account-4": "secretuser4"}
        self.statuses, self.full, self.fetch_fail = statuses, set(full), dict(fetch_fail)
        self.pushed, self.calls = [], []

    def __call__(self, account, *args, timeout=0):
        self.calls.append((account, args[0]))
        cmd = args[0]
        if cmd == "whoami":
            return {"owner": self.owner[account]}, ""
        if cmd == "push":
            folder = Path(args[1])
            meta = json.loads((folder/"kernel-metadata.json").read_text())
            assert meta["id"].startswith(self.owner[account] + "/")
            if account in self.full:
                return {"owner": self.owner[account], "error": "Maximum batch GPU session count of 2 reached"}, ""
            self.pushed.append((account, folder.name, meta))
            return {"owner": self.owner[account], "version": 1, "error": None}, ""
        if cmd == "status":
            seq = self.statuses.get(args[1], ["COMPLETE"])
            return {"owner": self.owner[account], "status": seq.pop(0) if len(seq) > 1 else seq[0],
                    "failure_message": f"oops {self.owner[account]}" if seq[0] == "ERROR" else ""}, ""
        if cmd == "fetch":
            slug, outdir, tag, *prefixes = args[1:]
            if self.fetch_fail.get(slug, 0) > 0:
                self.fetch_fail[slug] -= 1
                return None, f"boom for {self.owner[account]}"
            assert prefixes == JQ.FETCH_PREFIXES
            out = Path(outdir)
            rid = "run__" + slug
            files = {f"runs/{rid}/metrics.json": "{}", f"runs/{rid}/exit_status.json": "{}", f"logs/{rid}.log": "LOG",
                     "session_info.json": "{}", "kernel.log": "k",
                     "session_status.json": json.dumps({"state": "complete", "n_jobs": 1, "device_count": 2,
                                                        "failed_run_ids": ["bad1"] if slug.endswith("f") else []})}
            for rel, txt in files.items():
                (out/rel).parent.mkdir(parents=True, exist_ok=True)
                (out/rel).write_text(txt)
            return {"owner": self.owner[account], "saved": list(files)}, ""
        raise AssertionError(cmd)


def _sess(slug, account=None, sources=None):
    return JQ.normalize_session({"slug": slug, "account": account, "kernel_sources": sources or [],
                                 "gpu_lists": [[pj("run__" + slug)]]})


def _isolate_jq():
    root = Path(tempfile.mkdtemp(dir=tmp))
    saved = (JQ.RUNS_DIR, JQ.STATE_FILE, JQ.EVENTS, JQ.KERNELS_V2, JQ.kapi, dict(JQ.OWNERS), JQ.collect_files)
    JQ.RUNS_DIR, JQ.STATE_FILE, JQ.EVENTS, JQ.KERNELS_V2 = root/"runs", root/"state.json", root/"events.log", root/"kernels"
    JQ.OWNERS.clear()
    JQ.collect_files = lambda names=JQ.EMBED_NAMES: dict(FAKE_FILES)
    return root, saved


def _restore_jq(saved):
    JQ.RUNS_DIR, JQ.STATE_FILE, JQ.EVENTS, JQ.KERNELS_V2, JQ.kapi, owners, JQ.collect_files = saved
    JQ.OWNERS.clear()
    JQ.OWNERS.update(owners)


def test_scheduler_slots_accounts_events_and_fetch():
    root, saved = _isolate_jq()
    buf = io.StringIO()
    try:
        fake = FakeKaggle({"s-run-f": ["RUNNING", "COMPLETE"], "s-err": ["RUNNING", "ERROR"]})
        JQ.kapi = fake
        sessions = [_sess("s-aaaa"), _sess("s-bbbb"), _sess("s-cccc"), _sess("s-dddd"), _sess("s-eeee"),
                    _sess("s-pin4", account="account-4"), _sess("s-src3", sources=["some-kernel"]),
                    _sess("s-run-f"), _sess("s-err")]
        state = {"sessions": {s["slug"]: {"state": "pending", "account": None} for s in sessions}}
        with redirect_stdout(buf):
            while JQ.v2_try_schedule(state, sessions):
                pass
            run = {a: JQ.v2_running(state, a) for a in JQ.ACCOUNTS}
            assert run == {"account-3": 2, "account-4": 2}, run                      # max 2 GPU sessions per account
            pend = [k for k, j in state["sessions"].items() if j["state"] == "pending"]
            assert len(pend) == len(sessions)-4
            order = [p[1] for p in fake.pushed]
            assert order[:2] == ["s-aaaa", "s-bbbb"]                                  # pushed in list order
            # pinned / kernel_sources sessions land on their account once a slot frees
            for k in ("s-aaaa", "s-bbbb", "s-cccc", "s-dddd"):
                state["sessions"][k].update(state="done")
            while JQ.v2_try_schedule(state, sessions):
                pass
            assert state["sessions"]["s-pin4"]["account"] == "account-4"
            assert state["sessions"]["s-src3"]["account"] == "account-3"
            pushed_meta = {p[1]: p for p in fake.pushed}
            assert pushed_meta["s-src3"][2]["kernel_sources"] == ["secretuser3/some-kernel"]   # bare slug -> own account
            # nothing running can have a stale/duplicate state: polling completes sessions and fetches
            for k in list(state["sessions"]):
                if state["sessions"][k]["state"] == "pending":
                    state["sessions"][k].update(state="done")
            for k in ("s-pin4", "s-src3"):
                state["sessions"][k].update(state="running")
            state["sessions"]["s-run-f"] = {"state": "running", "account": "account-3"}
            state["sessions"]["s-err"] = {"state": "running", "account": "account-4"}
            JQ.v2_poll_running(state)                                                # RUNNING -> still running
            assert state["sessions"]["s-run-f"]["state"] == "running" and state["sessions"]["s-err"]["state"] == "running"
            JQ.v2_poll_running(state)                                                # COMPLETE / ERROR -> fetched
        assert state["sessions"]["s-run-f"]["state"] == "done" and state["sessions"]["s-err"]["state"] == "failed"
        assert state["sessions"]["s-pin4"]["state"] == "done"                        # default scripted status COMPLETE
        # outputs: run dir hoisted to runs/<run_id>/, stdout.log copied, session files kept per session
        rd = JQ.RUNS_DIR/"run__s-run-f"
        assert (rd/"metrics.json").exists() and (rd/"stdout.log").read_text() == "LOG"
        sd = JQ.RUNS_DIR/"_sessions"/"s-run-f"
        assert (sd/"session_status.json").exists() and (sd/"session_info.json").exists() and (sd/"kernel.log").exists()
        assert not (sd/"runs"/"run__s-run-f").exists()
        assert state["sessions"]["s-run-f"]["run_dirs"] == ["run__s-run-f"]
        assert state["sessions"]["s-run-f"]["session_summary"]["failed"] == ["bad1"]
        # events
        events = JQ.EVENTS.read_text()
        for word in ("PUSHED s-aaaa account-3", "PUSHED s-pin4 account-4", "DONE s-run-f COMPLETE", "FAILED s-err ERROR"):
            assert word in events, (word, events)
        # no Kaggle username anywhere: events, stdout, state, metadata of the pushed folders
        JQ.save_state(state)
        blob = events + buf.getvalue() + JQ.STATE_FILE.read_text()
        for folder in (root/"kernels").iterdir():
            blob += (folder/"kernel-metadata.json").read_text()
        for name in ("secretuser3", "secretuser4"):
            assert name not in blob, name
        assert "<owner>" in state["sessions"]["s-err"]["failure_message"]
    finally:
        _restore_jq(saved)


def test_scheduler_stale_slot_fetch_retry_supersede():
    root, saved = _isolate_jq()
    try:
        fake = FakeKaggle({}, full={"account-3"}, fetch_fail={"s-retry": 2})
        JQ.kapi = fake
        sessions = [_sess("s-stale"), _sess("s-retry")]
        state = {"sessions": {s["slug"]: {"state": "pending", "account": None} for s in sessions}}
        with redirect_stdout(io.StringIO()):
            assert JQ.v2_try_schedule(state, sessions)                                # account-3 reports "full" -> account-4
            assert state["sessions"]["s-stale"]["account"] == "account-4"
            assert state["sessions"]["s-retry"]["account"] == "account-4"
            JQ.v2_poll_running(state)
            assert state["sessions"]["s-retry"]["state"] == "fetching" and state["sessions"]["s-retry"]["fetch_errors"] == 1
            JQ.v2_poll_running(state)
            JQ.v2_poll_running(state)
        assert state["sessions"]["s-retry"]["state"] == "done" and state["sessions"]["s-stale"]["state"] == "done"
        assert "FETCH_RETRY s-retry" in JQ.EVENTS.read_text() and "secretuser" not in JQ.EVENTS.read_text()
        # the same run_id fetched by ANOTHER session keeps the old results as _superseded
        sd = JQ.RUNS_DIR/"_sessions"/"other"
        (sd/"runs"/"run__s-stale").mkdir(parents=True)
        (sd/"runs"/"run__s-stale"/"metrics.json").write_text("NEW")
        JQ.hoist_session(sd, JQ.RUNS_DIR, "other")
        assert (JQ.RUNS_DIR/"run__s-stale"/"metrics.json").read_text() == "NEW"
        old = list((JQ.RUNS_DIR/"_superseded").iterdir())
        assert len(old) == 1 and (old[0]/"metrics.json").read_text() == "{}"
        # the SAME session fetched twice just overwrites
        (sd/"runs"/"run__s-stale").mkdir(parents=True)
        (sd/"runs"/"run__s-stale"/"metrics.json").write_text("NEWER")
        JQ.hoist_session(sd, JQ.RUNS_DIR, "other")
        assert (JQ.RUNS_DIR/"run__s-stale"/"metrics.json").read_text() == "NEWER" and len(list((JQ.RUNS_DIR/"_superseded").iterdir())) == 1
        # fetch that never works -> FAILED after MAX_FETCH_ERRORS, outputs stay on Kaggle
        fake2 = FakeKaggle({}, fetch_fail={"s-never": 99})
        JQ.kapi = fake2
        st2 = {"sessions": {"s-never": {"state": "running", "account": "account-3"}}}
        with redirect_stdout(io.StringIO()):
            for _ in range(JQ.MAX_FETCH_ERRORS+1):
                JQ.v2_poll_running(st2)
        assert st2["sessions"]["s-never"]["state"] == "failed" and st2["sessions"]["s-never"]["final_status"] == "FETCH_ERROR"
    finally:
        _restore_jq(saved)


def test_cli_dry_run_end_to_end():
    root = Path(tempfile.mkdtemp(dir=tmp))
    sess = root/"sessions_demo.py"
    sess.write_text("SESSIONS = [{'slug': 'pc-demo-1017', 'account': None, 'kernel_sources': [], 'gpu_lists': [\n"
                    "  [{'run_id': 'x__demo__a', 'script': 'exp_run.py', 'args': '--run-id x__demo__a --epochs 1'}],\n"
                    "  [{'parallel': [{'run_id': 'x__demo__b', 'script': 'exp_run.py', 'args': '--run-id x__demo__b'},\n"
                    "                 {'run_id': 'x__demo__c', 'script': 'exp_run.py', 'args': '--run-id x__demo__c'}]}]]}]\n")
    saved_k = JQ.KERNELS_V2
    JQ.KERNELS_V2 = root/"kernels_v2"
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            JQ.main(["--sessions", str(sess), "--dry-run"])
    finally:
        JQ.KERNELS_V2 = saved_k
    out = buf.getvalue()
    assert "SESSION pc-demo-1017" in out and "syntax=OK" in out and "nothing pushed" in out
    run_py = root/"kernels_v2"/"pc-demo-1017"/"run.py"
    compile(run_py.read_text(), str(run_py), "exec")
    assert JQ.parse_args(["--sessions", "a.py", "--state", "s.json", "--dry-run"]).state == "s.json"
    assert JQ.V2_STATE.name == "queue_state_v2.json" and JQ.V2_STATE.parent == JQ.PROJECT/"results"


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                t0 = time.time()
                fn()
                print("PASS", name, f"({time.time()-t0:.1f}s)")
            except Exception:
                import traceback
                traceback.print_exc()
                print("FAIL", name)
                fails += 1
    shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASSED" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
