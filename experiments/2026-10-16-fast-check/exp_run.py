"""One training + evaluation job (GraphHash follow-up experiments). ONE seed per process.

python exp_run.py --dataset {Gowalla,AmazonBook,Yelp2018} --method {Full,GraphHash,Random,DoubleFreq,DoubleGraph}
    --loss {dau,bpr_author,dau_bias,dau_resbias} [--model {MF,LightGCN}] [--lightgcn-layers K] --seeds S --epochs N --eval-every K
    [--patience P] [--min-delta D] [--cap-min M] --lr X --wd Y --run-id RUN_ID [--tag NAME]
    [--resolution R] [--init-match] [--equal-item-norm] [--batch B]

Reuses the pinned author data/mapping/loss code through repro_run. Hash maps are fixed across seeds.
2-D aliases (DoubleFreq, shape (n,2)) are handled everywhere: vector = table(alias).sum(-2).
All scoring here is written for 1-D and 2-D aliases (repro_run.evaluate is not used).

Run directory R.ROOT/runs/<run-id>/ (PLAN section 8): config.json, mapping_{users,items}.npy, train_log.csv,
norm_log.csv, best_seed{N}.pt, metrics.json (same schema as exp_{tag}.json + "status"), per_user_test__{mode}.npz,
exit_status.json. exp_{tag}.json is still written in R.ROOT for backward compatibility.
"""
import argparse
import ast
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
import repro_run as R
from repro_core import metrics_from_topk

RESBIAS_WD = 0.01
BETAS = [0.0, .01, .02, .05, .1, .2, .5, 1., 2., 5.]
MASKED = -1e9


# ----------------------------------------------------------------------------- vectors / scoring
def table_vec(table, alias):
    """Embedding of an ID: one bucket row, or the SUM of its bucket rows if alias is 2-D (n,2)."""
    v = table(alias)
    return v.sum(-2) if alias.dim() == 2 else v


@torch.no_grad()
def all_vectors(model, aliases, prop=None):
    """(user_vectors, item_vectors) at ORIGINAL ID granularity (LightGCN-propagated if `prop` is given)."""
    if prop is not None:
        return prop(model, aliases)
    return table_vec(model.user_emb_table, aliases[0]), table_vec(model.item_emb_table, aliases[1])


# ----------------------------------------------------------------------------- LightGCN propagation
def build_norm_adj(edges, nu, ni, device):
    """Symmetric-normalised (nu+ni)x(nu+ni) user-item adjacency D^-1/2 A D^-1/2 from TRAIN edges (2,E) only.
    Duplicate edges are collapsed (GraphHash uses (A>0)); degree has the author's +1e-7."""
    e = edges.detach().cpu().numpy().astype(np.int64)
    key = np.unique(e[0]*ni+e[1])
    u, i = key//ni, key % ni
    deg = np.bincount(np.concatenate([u, nu+i]), minlength=nu+ni).astype(np.float64)+1e-7
    dinv = deg**-0.5
    w = (dinv[u]*dinv[nu+i]).astype(np.float32)
    rows = np.concatenate([u, nu+i])
    cols = np.concatenate([nu+i, u])
    idx = torch.as_tensor(np.stack([rows, cols]), dtype=torch.long)
    val = torch.as_tensor(np.concatenate([w, w]))
    return torch.sparse_coo_tensor(idx, val, (nu+ni, nu+ni)).coalesce().to(device)


class LightProp:
    """K-layer LightGCN propagation (layer-mean of layers 0..K) over the shared (hashed) tables, returning vectors
    at ORIGINAL id granularity: ego = [table(u_alias); table(i_alias)] (2-D aliases summed), no weights/nonlinearity."""
    def __init__(self, adj, nu, ni, layers=2):
        self.adj, self.nu, self.ni, self.layers = adj, nu, ni, int(layers)

    def __call__(self, model, aliases):
        ego = torch.cat([table_vec(model.user_emb_table, aliases[0]), table_vec(model.item_emb_table, aliases[1])])
        acc, x = ego, ego
        for _ in range(self.layers):
            x = torch.sparse.mm(self.adj, x)
            acc = acc + x
        acc = acc/(self.layers+1)
        return acc[:self.nu], acc[self.nu:]


def _mask_coords(users, masks):
    lens = [len(masks.get(x, ())) for x in users]
    if not sum(lens):
        return None, None
    rows = np.repeat(np.arange(len(users)), lens)
    cols = np.concatenate([np.asarray(masks[x], dtype=np.int64) for x in users if len(masks.get(x, ()))])
    return rows, cols


@torch.no_grad()
def sweep(uvec, ivec, truth, masks, betas, logpop=None, fixed_add=None, cosine=False, chunk=512, k=20):
    """Recall/NDCG@k for score = (u.i or cos) + fixed_add[i] + beta*logpop[i], for every beta in one pass.

    masks: uid -> items removed from the ranking (val: train items; test: train+val items).
    Returns {beta: {"recall", "ndcg"}} with metrics_from_topk semantics (recall denominator min(k, #pos)).
    """
    device = ivec.device
    if cosine:
        uvec, ivec = F.normalize(uvec, dim=-1), F.normalize(ivec, dim=-1)
    adds = []
    for beta in betas:
        a = torch.zeros(ivec.shape[0], device=device)
        if fixed_add is not None:
            a = a + fixed_add.to(device)
        if beta:
            a = a + beta*logpop.to(device)
        adds.append(a)
    keys = list(truth)
    preds = [[] for _ in betas]
    for s in range(0, len(keys), chunk):
        users = keys[s:s+chunk]
        base = uvec[torch.as_tensor(users, device=device)] @ ivec.t()
        rows, cols = _mask_coords(users, masks)
        if rows is not None:
            rows, cols = torch.as_tensor(rows, device=device), torch.as_tensor(cols, device=device)
        for j, a in enumerate(adds):
            sc = base + a[None, :]
            if rows is not None:
                sc[rows, cols] = MASKED
            preds[j].append(torch.topk(sc, k, dim=1).indices.cpu().numpy())
    truths = [truth[x] for x in keys]
    out = {}
    for beta, p in zip(betas, preds):
        m = metrics_from_topk(np.concatenate(p), truths, k)
        out[beta] = {"recall": m["recall"], "ndcg": m["ndcg"]}
    return out


def tune_and_test(eval_fn, betas):
    """Pick beta by VALIDATION recall (ties -> smallest beta), then evaluate TEST exactly once at it.

    eval_fn(split, betas) -> {beta: {"recall","ndcg"}}, split in {"val","test"}.
    """
    val = eval_fn("val", list(betas))
    best = max(betas, key=lambda b: val[b]["recall"])
    test = eval_fn("test", [best])[best]
    return {"best_beta": best, "val": val[best], "test": test, "val_by_beta": val}


def eval_suite(uvec, ivec, val_truth, test_truth, train_truth, merged, logpop, b=None, betas=BETAS,
               learned_add=None):
    def make(cosine, fixed_add):
        def f(split, bs):
            truth, masks = (val_truth, train_truth) if split == "val" else (test_truth, merged)
            return sweep(uvec, ivec, truth, masks, bs, logpop, fixed_add, cosine)
        return f
    out = {"raw": tune_and_test(make(False, None), [0.0]),
           "cosine": tune_and_test(make(True, None), [0.0]),
           "cos_prior": tune_and_test(make(True, None), betas),
           "raw_prior": tune_and_test(make(False, None), betas)}
    if b is not None:
        out["cos_b"] = tune_and_test(make(True, b), [0.0])
        out["cos_b_prior"] = tune_and_test(make(True, b), betas)
    if learned_add is not None:      # dau_resbias: cos + beta*log1p(c) + r, no free parameter -> nothing tuned
        out["learned"] = tune_and_test(make(True, learned_add), [0.0])
    return out


def spearman_norm_vs_count(ivec, counts):
    norms = ivec.detach().norm(dim=1).cpu().numpy()
    rho = spearmanr(norms, np.asarray(counts, dtype=np.float64)).correlation
    return None if rho is None or not np.isfinite(rho) else float(rho)


# ----------------------------------------------------------------------------- per-user scoring (new; sweep() is untouched)
@torch.no_grad()
def sweep_per_user(uvec, ivec, truth, masks, beta, logpop=None, cosine=False, chunk=512, k=20, fixed_add=None):
    """Per-user version of sweep() for ONE beta (same arithmetic, same top-k), for the beta chosen on validation.

    Returns arrays over the evaluated users (those with >= 1 positive, in `truth` order):
    users int32, top20 int32 [n,k], hits int16 (# hits in the top k), npos int16, ndcg float32.
    recall_u = hits / min(k, npos) is metrics_from_topk's per-user recall; mean(recall_u), mean(ndcg) = sweep()'s numbers.
    """
    device = ivec.device
    if cosine:
        uvec, ivec = F.normalize(uvec, dim=-1), F.normalize(ivec, dim=-1)
    a = torch.zeros(ivec.shape[0], device=device)
    if fixed_add is not None:
        a = a + fixed_add.to(device)
    if beta:
        a = a + beta*logpop.to(device)
    keys = list(truth)
    preds = []
    for s in range(0, len(keys), chunk):
        users = keys[s:s+chunk]
        sc = uvec[torch.as_tensor(users, device=device)] @ ivec.t() + a[None, :]
        rows, cols = _mask_coords(users, masks)
        if rows is not None:
            sc[torch.as_tensor(rows, device=device), torch.as_tensor(cols, device=device)] = MASKED
        preds.append(torch.topk(sc, k, dim=1).indices.cpu().numpy())
    top = np.concatenate(preds)
    keep = [j for j, x in enumerate(keys) if len(truth[x])]
    hits = np.zeros(len(keep), dtype=np.int16)
    npos = np.zeros(len(keep), dtype=np.int16)
    ndcg = np.zeros(len(keep), dtype=np.float32)
    for r, j in enumerate(keep):
        t = truth[keys[j]]
        hits[r] = int(np.isin(top[j, :k], t).sum())
        npos[r] = len(t)
        ndcg[r] = metrics_from_topk(top[j:j+1], [t], k)["ndcg"]
    return {"users": np.asarray([keys[j] for j in keep], dtype=np.int32), "top20": top[keep].astype(np.int32),
            "hits": hits, "npos": npos, "ndcg": ndcg}


PER_USER_MODES = ("raw", "cosine", "cos_prior", "raw_prior")


def save_per_user_test(run_dir, ev, uvec, ivec, c, learned_add=None):
    """per_user_test__{mode}.npz at each mode's validation-chosen beta. Returns a small check record per mode."""
    out = {}
    for mode in PER_USER_MODES + (("learned",) if learned_add is not None else ()):
        beta = ev[mode]["best_beta"]
        pu = sweep_per_user(uvec, ivec, c["test_truth"], c["merged"], beta, c["logpop"],
                            cosine=mode.startswith("cos") or mode == "learned",
                            fixed_add=learned_add if mode == "learned" else None)
        np.savez_compressed(Path(run_dir)/f"per_user_test__{mode}.npz", **pu)
        recall = float(np.mean(pu["hits"]/np.minimum(20, pu["npos"])))
        out[mode] = {"beta": beta, "n_users": int(len(pu["users"])),
                     "recall_diff_vs_eval_suite": abs(recall-ev[mode]["test"]["recall"])}
        if out[mode]["recall_diff_vs_eval_suite"] > 1e-6:
            R.log("per_user_mismatch", mode=mode, **out[mode])
    return out


# ----------------------------------------------------------------------------- early stopping
class Patience:
    """Patience on validation recall with a minimum counted gain.

    `best` is best_for_patience: the validation value at the last counted improvement. An evaluation counts as an
    improvement iff it exceeds `best` by at least min_delta (and strictly, so min_delta=0 is the legacy rule:
    any strict improvement). Patience = epochs since the last counted improvement. The CHECKPOINT is a separate
    rule (any strict gain of the best validation value) and lives in run_seed.
    """
    def __init__(self, patience=None, min_delta=0.0):
        self.patience, self.min_delta = patience, float(min_delta)
        self.best, self.epoch = -1., 0

    def update(self, v, epoch):
        gain = v-self.best
        if gain > 0 and gain >= self.min_delta-1e-12:
            self.best, self.epoch = v, epoch
        return bool(self.patience) and epoch-self.epoch >= self.patience


# ----------------------------------------------------------------------------- norm diagnostics (norm_log.csv)
NORM_COLUMNS = ["epoch", "kind", "q", "n_items", "mean_norm", "median_norm", "value"]


def popularity_quintiles(counts, groups=5):
    """Group id 1 (least popular) .. `groups` (most popular) per item: equal-count groups in rank order of the
    training count c_i, ties broken by item index."""
    counts = np.asarray(counts)
    order = np.lexsort((np.arange(len(counts)), counts))      # last key is primary: count, then item index
    q = np.empty(len(counts), dtype=np.int64)
    for g, idx in enumerate(np.array_split(order, groups)):
        q[idx] = g+1
    return q


def _spearman(a, b):
    rho = spearmanr(a, b).correlation
    return None if rho is None or not np.isfinite(rho) else float(rho)


@torch.no_grad()
def norm_stats(uvec, ivec, bucket_weight, i_alias, counts, method):
    """Rows (dicts over NORM_COLUMNS minus epoch) for one evaluation.

    quintile rows: mean/median of ORIGINAL-item vector norms (ivec = all_vectors) per popularity quintile.
    rho_item:   Spearman(item norm, c_i) over all items.
    rho_bucket: Spearman(bucket row norm, N_b = sum of c_i over the bucket's items) over non-empty buckets; only for a
                1-D item alias and method != Full (else value empty).
    rho_dir:    Spearman(cos(v_i, normalised mean user vector), c_i).
    """
    counts = np.asarray(counts, dtype=np.float64)
    norms = ivec.norm(dim=1).cpu().numpy().astype(np.float64)
    q = popularity_quintiles(counts)
    rows = []
    for g in range(1, 6):
        sel = norms[q == g]
        rows.append({"kind": "quintile", "q": g, "n_items": int(len(sel)),
                     "mean_norm": float(sel.mean()) if len(sel) else float("nan"),
                     "median_norm": float(np.median(sel)) if len(sel) else float("nan")})
    rows.append({"kind": "rho_item", "n_items": int(len(norms)), "value": spearman_norm_vs_count(ivec, counts)})
    rho_b, n_b = None, None
    if method != "Full" and i_alias.dim() == 1:
        alias = i_alias.cpu().numpy()
        n_buckets = bucket_weight.shape[0]
        keep = np.bincount(alias, minlength=n_buckets) > 0
        n_b = int(keep.sum())
        rho_b = _spearman(bucket_weight.norm(dim=1).cpu().numpy()[keep],
                          np.bincount(alias, weights=counts, minlength=n_buckets)[keep])
    rows.append({"kind": "rho_bucket", "n_items": n_b, "value": rho_b})
    direction = F.normalize(uvec.mean(0), dim=0)
    cos = (F.normalize(ivec, dim=1) @ direction).cpu().numpy().astype(np.float64)
    rows.append({"kind": "rho_dir", "n_items": int(len(cos)), "value": _spearman(cos, counts)})
    return rows


def init_csv(path, columns):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(columns)


def append_csv(path, columns, rows):
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(["" if r.get(k) is None else r[k] for k in columns])


# ----------------------------------------------------------------------------- init matching / norm equalisation
def init_match_factor(table_rows, n_ids, dim=64):
    """Xavier-uniform on a (rows, dim) table has E|row|^2 = 2*dim/(rows+dim), so scaling a compressed table by
    sqrt((B+dim)/(N+dim)) gives it the Full (N rows) table's RMS row norm. Gowalla: items 0.378, users 0.442."""
    return math.sqrt((table_rows+dim)/(n_ids+dim))


@torch.no_grad()
def apply_init_match(model, n_users, n_items):
    """Scale the item / user tables in place (after Xavier init). Returns the factors for config.json."""
    out = {}
    for side, table, n in (("items", model.item_emb_table, n_items), ("users", model.user_emb_table, n_users)):
        rows, dim = table.weight.shape
        out[side] = {"factor": init_match_factor(rows, n, dim), "table_rows": int(rows), "n_ids": int(n)}
        table.weight.mul_(out[side]["factor"])
    return out


@torch.no_grad()
def equalize_item_norms(table):
    """Rescale every row of the table to the table's current mean row norm (the C4 intervention)."""
    w = table.weight
    n = w.norm(dim=1, keepdim=True)
    w.mul_(n.mean()/n.clamp_min(1e-12))


# ----------------------------------------------------------------------------- run-directory plumbing
def _json_default(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"not JSON serialisable: {type(x)}")


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def update_json(path, fields):
    path = Path(path)
    cur = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    cur.update(fields)
    write_json(path, cur)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_result(result, out_path, run_dir):
    """metrics.json (run dir) and the legacy exp_{tag}.json hold the same dict."""
    write_json(out_path, result)
    write_json(Path(run_dir)/"metrics.json", result)


def save_mappings(run_dir, aliases):
    """mapping_users.npy / mapping_items.npy (int64; (n,) or (n,2)). Returns their sha256 (of the file bytes)."""
    out = {}
    for side, a in zip(("users", "items"), aliases):
        p = Path(run_dir)/f"mapping_{side}.npy"
        np.save(p, a.detach().cpu().numpy().astype(np.int64))
        out[side] = sha256_file(p)
    return out


def resolve_batch(args, n_edges):
    return args.batch or (n_edges if args.loss == "bpr_author" else 1024)


def build_config(args, c, mapping_sha):
    """config.json content (written before training)."""
    here = Path(__file__).resolve().parent
    code = {n: sha256_file(here/n) if (here/n).exists() else None
            for n in ("exp_run.py", "repro_run.py", "repro_core.py")}
    return {"run_id": args.run_id, "tag": args.tag, "args": vars(args),
            "resolved": {"lr": args.lr, "wd": args.wd, "batch": resolve_batch(args, int(c["parts"][0].shape[1])),
                         "n_train_edges": int(c["parts"][0].shape[1]), "embedding_dim": 64,
                         "eval_every": args.eval_every, "patience": args.patience, "min_delta": args.min_delta,
                         "cap_min": args.cap_min, "epochs": args.epochs, "resolution": args.resolution},
            "exp_run_sha256": code["exp_run.py"], "code_sha256": code,
            "torch_version": torch.__version__, "python": sys.version.split()[0],
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "mapping_sha256": mapping_sha,
            "mapping_shapes": {"users": list(c["aliases"][0].shape), "items": list(c["aliases"][1].shape)},
            "table_rows": {"users": int(c["aliases"][0].max())+1, "items": int(c["aliases"][1].max())+1},
            "init_match_factors": None,
            "model": getattr(args, "model", "MF"),
            "lightgcn_layers": (getattr(args, "lightgcn_layers", 2) if getattr(args, "model", "MF") == "LightGCN"
                                else None),
            "method_label": ("DoubleGraph: structure-aware SUBSTITUTE run (GraphHash double_graph_hash: universal-hash "
                             "code + Louvain-cluster code, vector = sum of two rows)"
                             if args.method == "DoubleGraph" else args.method),
            "dataset_stats": {**c["info"], "train_users": len(c["train_truth"]), "val_eval_users": len(c["val_truth"]),
                              "test_eval_users": len(c["test_truth"])},
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}


@contextlib.contextmanager
def exclusive_lock(path):
    """Cross-process lock. R.prepare() re-extracts the dataset files and R.load_data() reads them; two jobs of one
    Kaggle session (one per GPU) share /kaggle/working/_work, so that part must not run concurrently."""
    try:
        import fcntl
    except ImportError:          # Windows dev box: no concurrent jobs
        yield
        return
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ----------------------------------------------------------------------------- losses
def cosine_pair(a, b):
    return (F.normalize(a, dim=-1)*F.normalize(b, dim=-1)).sum(-1)


def dau_bias_loss(model, b, dau, bpr, ua, ia, iid, neg, i_map):
    """P5: DAULoss(u, i) + BPRLoss(s_pos, s_neg), s(u,i) = cos(u_emb, i_emb) + b[original item id]."""
    u, i = model(ua, ia)
    neg_emb = table_vec(model.item_emb_table, i_map[neg])
    s_pos = cosine_pair(u, i) + b[iid]
    s_neg = cosine_pair(u, neg_emb) + b[neg]
    return dau(u, i) + bpr(s_pos, s_neg)


def bias_loss_from_vecs(dau, bpr, u, i, neg_emb, add_pos, add_neg):
    """DAULoss(u, i) + BPRLoss(cos(u,i)+add_pos, cos(u,neg)+add_neg) on precomputed (possibly propagated) vectors."""
    return dau(u, i) + bpr(cosine_pair(u, i)+add_pos, cosine_pair(u, neg_emb)+add_neg)


def resbias_add(beta, r, logpop, idx):
    """Item-additive part of the C3 score for ORIGINAL item ids: beta*log(1+count) + r."""
    return beta*logpop[idx] + r[idx]


# ----------------------------------------------------------------------------- double-frequency hashing
def double_freq_maps(utils_path, parts, nu, ni, ku, ki):
    """Author's double_frequency_hash (cutoff .5, bucket counts = GraphHash's), seeded 0. Codes are (n, 2)."""
    from sympy import nextprime
    tree = ast.parse(utils_path.read_text())
    names = {"universal_hash", "double_frequency_hash_code", "double_frequency_hash"}
    selected = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names]
    assert {x.name for x in selected} == names, "author hashing functions not found"
    ns = {"np": np, "math": math, "random": random, "nextprime": nextprime}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "author_hash_functions", "exec"), ns)
    random.seed(0)
    np.random.seed(0)
    graph = csr_matrix((np.ones(parts[0].shape[1]), parts[0].numpy()), shape=(nu, ni))
    uc, ic = ns["double_frequency_hash"](graph, ku, ki, nu, ni)
    uc, ic = np.asarray(uc, dtype=np.int64), np.asarray(ic, dtype=np.int64)
    assert uc.shape == (nu, 2) and ic.shape == (ni, 2)
    assert 0 <= uc.min() and uc.max() < ku and 0 <= ic.min() and ic.max() < ki
    return uc, ic


def double_graph_maps(utils_path, parts, nu, ni, ku, ki, resolution, louvain):
    """Author's double_graph_hash (Louvain at `resolution`, + universal hash), seeded 0. Codes are (n, 2), both
    columns index ONE table of ku (users) / ki (items) rows, vector = sum of the two rows."""
    from sympy import nextprime
    tree = ast.parse(utils_path.read_text())
    names = {"graphClustering", "map_to_consecutive_integers", "universal_hash", "double_graph_hash"}
    selected = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names]
    assert {x.name for x in selected} == names, "author hashing functions not found"
    ns = {"np": np, "math": math, "random": random, "nextprime": nextprime, "Louvain": louvain}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "author_graph_hash_functions", "exec"), ns)
    random.seed(0)
    np.random.seed(0)
    graph = csr_matrix((np.ones(parts[0].shape[1]), parts[0].numpy()), shape=(nu, ni))
    uc, ic = ns["double_graph_hash"](graph, resolution, ku, ki, nu, ni)
    uc, ic = np.asarray(uc, dtype=np.int64), np.asarray(ic, dtype=np.int64)
    assert uc.shape == (nu, 2) and ic.shape == (ni, 2)
    assert 0 <= uc.min() and uc.max() < ku and 0 <= ic.min() and ic.max() < ki, "cluster count differs from GraphHash's"
    return uc, ic


# ----------------------------------------------------------------------------- training
def build_context(args):
    with exclusive_lock(R.WORK/".prepare.lock"):
        MF, MFDAU, BPRLoss, DAULoss, namespace = R.prepare()
        parts, maps, (train_truth, val_truth, test_truth), info = R.load_data(args.dataset, namespace,
                                                                              args.resolution)
    nu, ni = len(maps["Full"][0]), len(maps["Full"][1])
    if args.method == "DoubleFreq":
        maps["DoubleFreq"] = double_freq_maps(R.WORK/"GraphHash"/"utils.py", parts, nu, ni,
                                              info["user_buckets"], info["item_buckets"])
    if args.method == "DoubleGraph":
        maps["DoubleGraph"] = double_graph_maps(R.WORK/"GraphHash"/"utils.py", parts, nu, ni, info["user_buckets"],
                                                info["item_buckets"], args.resolution, namespace["Louvain"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aliases = tuple(torch.tensor(np.asarray(x), dtype=torch.long, device=device) for x in maps[args.method])
    pop = torch.bincount(parts[0][1], minlength=ni).float()
    return dict(MF=MF, MFDAU=MFDAU, BPRLoss=BPRLoss, DAULoss=DAULoss, parts=parts, info=info,
                train_truth=train_truth, val_truth=val_truth, test_truth=test_truth,
                merged={u: train_truth.get(u, [])+val_truth.get(u, []) for u in set(train_truth)|set(val_truth)},
                device=device, aliases=aliases, nu=nu, ni=ni, counts=pop.numpy(), logpop=torch.log1p(pop).to(device))


def val_recall(model, aliases, b, c, prop=None):
    """b: None (raw dot) or an item-additive vector (cosine + b; for dau_resbias b = beta*logpop + r)."""
    uvec, ivec = all_vectors(model, aliases, prop)
    cosine, add = (False, None) if b is None else (True, b.detach())
    return sweep(uvec, ivec, c["val_truth"], c["train_truth"], [0.0], None, add, cosine)[0.0]["recall"]


def run_seed(args, seed, c, result, out_path, run_dir=None):
    device, aliases = c["device"], c["aliases"]
    run_dir = Path(run_dir) if run_dir is not None else R.ROOT/"runs"/args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
    cls = c["MF"] if args.loss == "bpr_author" else c["MFDAU"]
    model = cls(int(aliases[0].max())+1, int(aliases[1].max())+1, 64).to(device)
    init_match = None
    if args.init_match and args.method != "Full":
        init_match = apply_init_match(model, c["nu"], c["ni"])           # after Xavier, before the optimizer
        R.log("init_match", tag=args.tag, **{k: v["factor"] for k, v in init_match.items()})
    update_json(run_dir/"config.json", {"init_match_factors": init_match})
    model_kind = getattr(args, "model", "MF")
    prop = None
    if model_kind == "LightGCN":
        layers = getattr(args, "lightgcn_layers", 2)
        prop = LightProp(build_norm_adj(c["parts"][0], c["nu"], c["ni"], device), c["nu"], c["ni"], layers)
        row_model = {"model": "LightGCN", "layers": layers, "adj_nnz": int(prop.adj._nnz())}
    params = list(model.parameters())
    b = beta = r = None
    emb_params = sum(p.numel() for p in params)
    row = {"params": emb_params, "val_curve": {}}
    if prop is not None:
        row.update(row_model)
    if init_match:
        row["init_match"] = init_match
    if args.loss == "dau_bias":
        b = nn.Parameter(torch.zeros(c["ni"], device=device))
        params.append(b)
        row["memory_overhead"] = {"b_params": b.numel(), "b_bytes_fp32": 4*b.numel(),
                                  "pct_of_embedding_table_params": 100.*b.numel()/emb_params}
        R.log("bias_overhead", tag=args.tag, **row["memory_overhead"])
    groups = [{"params": params, "weight_decay": args.wd}]
    if args.loss == "dau_resbias":
        beta = nn.Parameter(torch.zeros((), device=device))
        r = nn.Parameter(torch.zeros(c["ni"], device=device))
        groups = [{"params": params, "weight_decay": args.wd},
                  {"params": [r], "weight_decay": RESBIAS_WD},
                  {"params": [beta], "weight_decay": 0.0}]
        row["memory_overhead"] = {"r_params": r.numel(), "beta_params": 1, "b_bytes_fp32": 4*(r.numel()+1),
                                  "pct_of_embedding_table_params": 100.*(r.numel()+1)/emb_params}
        R.log("resbias_overhead", tag=args.tag, **row["memory_overhead"])
    opt = torch.optim.Adam(groups, lr=args.lr, weight_decay=args.wd)
    logpop = c["logpop"]

    def learned_vec():
        return None if r is None else (beta.detach()*logpop + r.detach())
    bpr, dau = c["BPRLoss"](), c["DAULoss"](1.)
    edges = c["parts"][0].to(device)
    n_edges = edges.shape[1]
    batch = resolve_batch(args, n_edges)
    ckpt = run_dir/f"best_seed{seed}.pt"
    train_log, norm_log = run_dir/"train_log.csv", run_dir/"norm_log.csv"
    train_cols = ["epoch", "train_loss", "val_recall20", "elapsed_s"]
    init_csv(train_log, train_cols)
    init_csv(norm_log, NORM_COLUMNS)
    result["seeds"][str(seed)] = row
    u_map, i_map = aliases
    started = time.perf_counter()
    best_val, best_epoch, epoch, stopped = -1., 0, 0, None
    patience = Patience(args.patience, args.min_delta)
    for epoch in range(1, args.epochs+1):
        model.train()
        order = torch.randperm(n_edges, device=device) if batch < n_edges else None
        loss_sum, n_batches = torch.zeros((), device=device), 0
        for first in range(0, n_edges, batch):
            uid, iid = edges[:, order[first:first+batch]] if order is not None else edges
            ua, ia = u_map[uid], i_map[iid]
            opt.zero_grad(set_to_none=True)
            if prop is not None:                   # LightGCN: one propagation per optimizer step, full tables
                U, I = prop(model, aliases)
                u, i = U[uid], I[iid]
                if args.loss == "dau":
                    value = dau(u, i)
                else:
                    neg = torch.randint(c["ni"], iid.shape, device=device)
                    if args.loss == "bpr_author":
                        value = bpr((u*i).sum(-1), (u*I[neg]).sum(-1))
                    elif args.loss == "dau_bias":
                        value = bias_loss_from_vecs(dau, bpr, u, i, I[neg], b[iid], b[neg])
                    else:
                        value = bias_loss_from_vecs(dau, bpr, u, i, I[neg], resbias_add(beta, r, logpop, iid),
                                                    resbias_add(beta, r, logpop, neg))
            elif args.loss == "bpr_author":
                neg = torch.randint(c["ni"], iid.shape, device=device)
                value = bpr(*model(ua, ia, i_map[neg]))
            elif args.loss == "dau":
                value = dau(*model(ua, ia))
            elif args.loss == "dau_bias":
                neg = torch.randint(c["ni"], iid.shape, device=device)
                value = dau_bias_loss(model, b, dau, bpr, ua, ia, iid, neg, i_map)
            else:                                   # dau_resbias
                neg = torch.randint(c["ni"], iid.shape, device=device)
                u, i = model(ua, ia)
                neg_emb = table_vec(model.item_emb_table, i_map[neg])
                value = bias_loss_from_vecs(dau, bpr, u, i, neg_emb, resbias_add(beta, r, logpop, iid),
                                            resbias_add(beta, r, logpop, neg))
            if not torch.isfinite(value):
                raise FloatingPointError("Nonfinite objective")
            value.backward()
            opt.step()
            if args.equal_item_norm:
                equalize_item_norms(model.item_emb_table)
            loss_sum += value.detach()
            n_batches += 1
        capped = time.perf_counter()-started > args.cap_min*60
        v = None
        if epoch % args.eval_every == 0 or epoch == args.epochs or capped:
            model.eval()
            vb = b if r is None else learned_vec()
            v = val_recall(model, aliases, vb, c) if prop is None else val_recall(model, aliases, vb, c, prop)
            row["val_curve"][str(epoch)] = v
            R.log("val", tag=args.tag, seed=seed, epoch=epoch, val_recall=v, best=max(best_val, v),
                  elapsed=round(time.perf_counter()-started, 1))
            if v > best_val:                                     # checkpoint: ANY strict gain of the best validation
                best_val, best_epoch = v, epoch
                payload = {"model": model.state_dict(), "b": None if b is None else b.detach().cpu(),
                           "aliases": [a.cpu() for a in aliases]}
                if r is not None:                                # dau_resbias only (keeps legacy checkpoint keys)
                    payload.update(beta=beta.detach().cpu(), r=r.detach().cpu())
                torch.save(payload, ckpt)
            uv, iv = all_vectors(model, aliases, prop)
            stats = norm_stats(uv, iv, model.item_emb_table.weight.detach(), i_map, c["counts"], args.method)
            append_csv(norm_log, NORM_COLUMNS, [{"epoch": epoch, **r} for r in stats])
            save_result(result, out_path, run_dir)
        append_csv(train_log, train_cols,
                   [{"epoch": epoch, "train_loss": float(loss_sum)/n_batches, "val_recall20": v,
                     "elapsed_s": round(time.perf_counter()-started, 3)}])
        if v is not None and patience.update(v, epoch):           # patience: gains of at least min_delta only
            stopped = "patience"
            break
        if capped:
            stopped = "time_cap"
            break
    stopped = stopped or "max_epochs"
    row["stop_reason"] = stopped
    train_seconds = time.perf_counter()-started
    ck = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ck["model"])
    model.eval()
    b_best = None if ck["b"] is None else ck["b"].to(device)
    uvec, ivec = all_vectors(model, aliases, prop)
    learned_add = None
    if r is not None:
        beta_best, r_best = ck["beta"].to(device), ck["r"].to(device)
        learned_add = beta_best*logpop + r_best
        rn = r_best.cpu().numpy().astype(np.float64)
        row["learned"] = {"beta": float(beta_best), "r_norm": float(np.linalg.norm(rn)), "r_mean": float(rn.mean()),
                          "r_std": float(rn.std()), "r_abs_mean": float(np.abs(rn).mean()),
                          "r_abs_max": float(np.abs(rn).max()), "r_spearman_vs_count": _spearman(rn, c["counts"])}
    row.update(best_epoch=best_epoch, best_val=best_val, epochs_run=epoch, stopped=stopped,
               train_seconds=train_seconds, batch=batch,
               spearman_norm_vs_count=spearman_norm_vs_count(ivec, c["counts"]),
               eval=eval_suite(uvec, ivec, c["val_truth"], c["test_truth"], c["train_truth"], c["merged"],
                               c["logpop"], b_best, learned_add=learned_add))
    row["per_user"] = save_per_user_test(run_dir, row["eval"], uvec, ivec, c, learned_add)
    row["wall_seconds"] = time.perf_counter()-started
    if torch.cuda.is_available():
        row["peak_cuda_mib"] = torch.cuda.max_memory_allocated()/1024**2
    save_result(result, out_path, run_dir)
    R.log("seed_done", tag=args.tag, seed=seed, best_epoch=best_epoch, stop_reason=stopped,
          **{f"test_{k}": v["test"]["recall"] for k, v in row["eval"].items()})


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["Gowalla", "AmazonBook", "Yelp2018"])
    p.add_argument("--method", required=True, choices=["Full", "GraphHash", "Random", "DoubleFreq", "DoubleGraph"])
    p.add_argument("--loss", required=True, choices=["dau", "bpr_author", "dau_bias", "dau_resbias"])
    p.add_argument("--model", default="MF", choices=["MF", "LightGCN"],
                   help="LightGCN: K-layer propagation over the shared tables (train-edge adjacency) before every loss/eval")
    p.add_argument("--lightgcn-layers", type=int, default=2, help="K for --model LightGCN")
    p.add_argument("--seeds", default="0", help="exactly ONE seed per process")
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--eval-every", type=int, required=True)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--min-delta", type=float, default=0.0,
                   help="patience only counts validation-recall gains of at least this much over best_for_patience; "
                        "the checkpoint is still saved on any strict gain")
    p.add_argument("--cap-min", type=float, default=75.)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--wd", type=float, required=True)
    p.add_argument("--batch", type=int, default=None, help="override (default 1024; full batch for bpr_author)")
    p.add_argument("--run-id", required=True, help="outputs go to <ROOT>/runs/<run-id>/")
    p.add_argument("--tag", default=None, help="name of the legacy exp_<tag>.json (default: the run id)")
    p.add_argument("--resolution", type=int, default=200, help="GraphHash Louvain resolution")
    p.add_argument("--init-match", action="store_true",
                   help="after Xavier, scale compressed tables so their RMS row norm equals the Full table's")
    p.add_argument("--equal-item-norm", action="store_true",
                   help="after every optimizer step rescale each item-table row to the table's mean row norm")
    a = p.parse_args(argv)
    a.seeds = [int(x) for x in a.seeds.split(",")]
    if len(a.seeds) != 1:
        p.error(f"exactly one seed per process (got --seeds {a.seeds})")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-]*", a.run_id):
        p.error(f"bad --run-id {a.run_id!r}")
    if a.min_delta < 0 or a.resolution <= 0:
        p.error("--min-delta must be >= 0 and --resolution > 0")
    a.tag = a.tag or a.run_id
    return a


def main(argv=None):
    args = parse(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    run_dir = R.ROOT/"runs"/args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = R.ROOT/f"exp_{args.tag}.json"
    status, result = {"exit_code": 1, "stop_reason": None, "end_time": None, "error": None}, None
    try:
        c = build_context(args)
        mapping_sha = save_mappings(run_dir, c["aliases"])
        write_json(run_dir/"config.json", build_config(args, c, mapping_sha))
        result = {"tag": args.tag, "run_id": args.run_id, "args": vars(args), "status": "running",
                  "bucket_info": {k: c["info"][k] for k in
                                  ["user_buckets", "item_buckets", "compressed_parameters", "full_parameters"]},
                  "seeds": {}}
        run_seed(args, args.seeds[0], c, result, out_path, run_dir)
        result["status"] = "complete"
        save_result(result, out_path, run_dir)
        status["exit_code"] = 0
    except BaseException as exc:
        tb = traceback.format_exc()
        status["error"] = f"{type(exc).__name__}: {exc}"
        (R.ROOT/f"exp_{args.tag}_failure.log").write_text(tb)
        (run_dir/"failure.log").write_text(tb)
        if result is not None:
            result["status"] = "failed"
            save_result(result, out_path, run_dir)
        raise
    finally:
        rows = (result or {}).get("seeds", {})
        status["stop_reason"] = next((r.get("stop_reason") for r in rows.values()), None)
        status["end_time"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_json(run_dir/"exit_status.json", status)


if __name__ == "__main__":
    main()
