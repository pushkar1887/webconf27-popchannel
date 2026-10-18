"""T3: scoring-only controls on a stored checkpoint (no training).  PLAN 4.3, 4.6, 5 (Stage 1).

python score_controls.py --ckpt PATH_OR_URL --dataset {Gowalla,Yelp2018,AmazonBook}
    --method {Full,GraphHash,Random,DoubleFreq} --base {cos,raw} --run-id ID
    [--assert-aliases PATH_OR_URL] [--e2e] [--seeds 10] [--out-root /kaggle/working/controls]
    [--data-root DIR] [--max-users N]          (the last two are local-check conveniences)

Output: <out-root>/<run-id>/controls.json (fractions, not percent) + per_user__*.npz for the key modes.

Design (rules in TASKS Part 1; identities in PLAN 4.2-4.6):
  * Scores are computed per BUCKET (user vector x unique bucket / alias-tuple vectors) and copied to items through
    the mapping, so items that share a bucket have bit-identical scores. Item-level additive terms (beta*p, ...) are
    added AFTER the expansion.
  * Ranking is lexicographic: items are put once in the order of a secondary key (random seeded permutation, lower
    item id, or higher train count), then a STABLE descending sort on the score is taken. A tiny beta*p is never
    used to break ties. Masked items (val: train items; test: train+val items) get -1e9.
  * beta is tuned on VALIDATION recall over BETAS (ties -> smaller beta); TEST is scored once per cell (a cell that is
    needed twice is computed once and reused; scoring a test cell twice raises).
  * `legacy_topk` reproduces the OLD evaluation: item-level scores, torch.topk, no secondary key.
Grouping variable: the bucket for 1-D maps; the unique UNORDERED alias pair for multi-alias maps (the vector is
the sum of the rows, so (a,b) and (b,a) are the same vector and tie exactly). Counts of ordered tuples are reported.
"""
import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
LOCAL_REPRO = HERE.parent / "2026-10-16-loss-sharing-repro" / "_work"
BETAS = [0.0, .01, .02, .05, .1, .2, .5, 1., 2., 5.]
MASKED = -1e9
K = 20
CHUNK = 512
DATASETS = ["Gowalla", "Yelp2018", "AmazonBook"]
METHODS = ["Full", "GraphHash", "Random", "DoubleFreq"]
_PREPARED = {}


# ----------------------------------------------------------------------------- small utilities
def log(event, **kv):
    print(json.dumps({"event": event, **kv}, default=_jd), flush=True)


def _jd(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def _repro_path():
    if LOCAL_REPRO.exists() and str(LOCAL_REPRO) not in sys.path:
        sys.path.append(str(LOCAL_REPRO))


def import_repro_core():
    try:
        import repro_core
    except ImportError:
        _repro_path()
        import repro_core
    return repro_core


def import_repro_run():
    try:
        import repro_run
    except ImportError:
        _repro_path()
        import repro_run
    return repro_run


def sha256_array(a):
    """sha256 over the C-contiguous little-endian int64 bytes (shape is not hashed)."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a), dtype="<i8").tobytes()).hexdigest()


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def is_url(s):
    return re.match(r"^https?://", str(s)) is not None


def redact(s):
    """Never print or store signed URLs."""
    return "<url>" if s is not None and is_url(s) else s


def source_label(s):
    """What is stored about an input: '<url>' for a URL, else only the file name (paths may contain account names)."""
    if s is None:
        return None
    return "<url>" if is_url(s) else Path(str(s)).name


def scrub(text):
    """Remove URLs (signed links) from text that is logged or written to disk."""
    return re.sub(r"https?://[^\s'\"<>]+", "<url>", str(text))


def fetch_to(path_or_url, dest_dir, name, tries=6):
    """Local path -> itself. URL -> downloaded file (with retries). The URL itself is never logged."""
    if not is_url(path_or_url):
        p = Path(path_or_url)
        if not p.exists():
            raise FileNotFoundError(f"{name}: {p}")
        return p
    import requests
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / name
    for attempt in range(tries):
        try:
            with requests.get(path_or_url, timeout=120, stream=True) as r:
                r.raise_for_status()
                with open(target, "wb") as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)
            if target.stat().st_size == 0:
                raise IOError("empty download")
            return target
        except Exception as e:
            if attempt == tries - 1:
                raise RuntimeError(f"download failed for {name}: {type(e).__name__}: {scrub(e)}") from None
            time.sleep(15 * (attempt + 1))


def sd(x):
    x = np.asarray(x, dtype=np.float64)
    return float(x.std(ddof=1)) if len(x) > 1 else 0.0


# ----------------------------------------------------------------------------- checkpoint, vectors, groups
def to_alias(a):
    a = a.cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    return np.ascontiguousarray(a, dtype=np.int64)


def load_checkpoint(path):
    """Plain state_dict, or {"model": state_dict, "b": ..., "aliases": [user_alias, item_alias]}."""
    obj = torch.load(str(path), map_location="cpu", weights_only=True)
    aliases = b = None
    if isinstance(obj, dict) and isinstance(obj.get("model"), dict):
        state, b, aliases = obj["model"], obj.get("b"), obj.get("aliases")
    else:
        state = obj

    def find(suffix):
        keys = [k for k in state if k.endswith(suffix)]
        if len(keys) != 1:
            raise KeyError(f"expected exactly one key ending with {suffix}, got {keys}")
        return state[keys[0]].float()
    if aliases is not None:
        aliases = [to_alias(a) for a in aliases]
        if len(aliases) != 2:
            raise ValueError("aliases must be [user_alias, item_alias]")
    return {"user_table": find("user_emb_table.weight"), "item_table": find("item_emb_table.weight"),
            "aliases": aliases, "b": b}


def table_vec(table, alias):
    """Vector of an ID: its bucket row, or the SUM of its bucket rows if the alias is 2-D (n, 2)."""
    idx = torch.as_tensor(np.asarray(alias), dtype=torch.long, device=table.device)
    v = table[idx]
    return v.sum(-2) if idx.dim() == 2 else v


def group_labels(alias, canonical=True):
    """Dense group id per item, number of groups, index of one representative item per group.

    1-D alias: the bucket. 2-D alias: the unique alias tuple; canonical=True sorts each pair first (the vector is a
    sum, so (a,b) and (b,a) are identical vectors and tie exactly); canonical=False keeps the ordered tuple.
    """
    a = np.asarray(alias)
    if a.ndim == 1:
        _, rep, inv = np.unique(a, return_index=True, return_inverse=True)
    else:
        key = np.sort(a, axis=1) if canonical else a
        _, rep, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return inv.reshape(-1).astype(np.int64), int(len(rep)), rep.astype(np.int64)


def mu_eps(p, labels, n_groups):
    """Identity 3: p = mu_{a(i)} + eps_i, mu_b = mean of p in bucket b, sum of eps over a bucket = 0."""
    p = np.asarray(p, dtype=np.float64)
    cnt = np.bincount(labels, minlength=n_groups).astype(np.float64)
    mu_g = np.bincount(labels, weights=p, minlength=n_groups) / cnt
    mu = mu_g[labels]
    return mu_g, mu, p - mu


def shuffle_within_buckets(p, labels, seed):
    """p permuted within each bucket: the multiset per bucket (hence mu) is unchanged."""
    p = np.asarray(p)
    rng = np.random.RandomState(seed)
    r = rng.rand(len(p))
    by_group = np.argsort(labels, kind="stable")          # items grouped, group blocks in the same positions as...
    by_group_random = np.lexsort((r, labels))              # ...the same blocks in random order
    out = np.empty_like(p)
    out[by_group] = p[by_group_random]
    return out


def quantize_u8(p):
    """q = round(255 p / max p) in {0..255}, dequantized p_hat = q max p / 255."""
    p = np.asarray(p, dtype=np.float64)
    pmax = float(p.max())
    q = np.rint(255.0 * p / pmax)
    ph = q * pmax / 255.0
    return ph, {"p_max": pmax, "max_abs_error": float(np.abs(ph - p).max()), "n_levels_used": int(len(np.unique(q)))}


def shapley_decomposition(base, mu, eps, both):
    """Exact 2x2 decomposition of the full repair; all effects relative to the base cell."""
    d_mu, d_eps, d_both = mu - base, eps - base, both - base
    inter = d_both - d_mu - d_eps
    phi_mu = 0.5 * (d_mu + (d_both - d_eps))
    phi_eps = 0.5 * (d_eps + (d_both - d_mu))
    frac = (lambda x: x / d_both) if d_both != 0 else (lambda x: None)
    return {"base": base, "delta_mu": d_mu, "delta_eps": d_eps, "delta_both": d_both, "interaction": inter,
            "shapley_mu": phi_mu, "shapley_eps": phi_eps,
            "frac_delta_mu": frac(d_mu), "frac_delta_eps": frac(d_eps), "frac_interaction": frac(inter),
            "frac_shapley_mu": frac(phi_mu), "frac_shapley_eps": frac(phi_eps)}


def pick_beta(val_by_beta):
    """Highest validation recall; ties go to the smaller beta."""
    best = None
    for b in sorted(val_by_beta):
        if best is None or val_by_beta[b] > val_by_beta[best]:
            best = b
    return best


def per_user_metrics(hits, npos, k=K):
    """Recall (denominator min(k, |T|)) and NDCG per user from a (U, k) hit matrix (same formulas as repro_core)."""
    hits = np.asarray(hits)
    cap = np.minimum(k, np.asarray(npos))
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    recall = hits.sum(1) / cap
    ndcg = (hits.astype(np.float64) @ disc) / np.cumsum(disc)[cap - 1]
    return recall.astype(np.float64), ndcg.astype(np.float64)


def length_variants(ivec, counts, seeds, n_deciles=10):
    """Item lengths with directions kept. Returns {name: length vector} and the OLS fit of log length on p."""
    ivec = np.asarray(ivec, dtype=np.float64)
    r = np.linalg.norm(ivec, axis=1)
    order = np.argsort(counts, kind="stable")
    blocks = np.array_split(order, n_deciles)
    out = {}
    for s in seeds:
        rng = np.random.RandomState(2000 + s)
        dec = r.copy()
        for blk in blocks:
            dec[blk] = r[blk][rng.permutation(len(blk))]
        out[f"len_shuffle_decile_s{s}"] = dec
        out[f"len_shuffle_global_s{s}"] = r[rng.permutation(len(r))]
    p = np.log1p(np.asarray(counts, dtype=np.float64))
    m = np.asarray(counts) >= 1
    x, y = p[m], np.log(r[m])
    g = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
    a = float(y.mean() - g * x.mean())
    resid = y - (a + g * x)
    r2 = float(1.0 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum())
    out["len_replace_fit"] = np.exp(a + g * p)
    return out, {"a": a, "g": g, "r2": r2, "n_items_fit": int(m.sum())}


# ----------------------------------------------------------------------------- authors' double-frequency hashing
def double_freq_maps(utils_path, parts, nu, ni, ku, ki):
    """Authors' double_frequency_hash (default cutoff), seeded 0, bucket counts ku/ki. Codes are (n, 2).

    Copied from exp_run.double_freq_maps (that file is not imported): the three author functions are
    extracted with ast from GraphHash/utils.py and executed in a namespace with np, math, random, nextprime.
    """
    from scipy.sparse import csr_matrix
    from sympy import nextprime
    tree = ast.parse(Path(utils_path).read_text())
    names = {"universal_hash", "double_frequency_hash_code", "double_frequency_hash"}
    selected = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names]
    assert {x.name for x in selected} == names, "author hashing functions not found"
    ns = {"np": np, "math": math, "random": random, "nextprime": nextprime}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "author_hash_functions", "exec"), ns)
    random.seed(0)
    np.random.seed(0)
    edges = parts[0]
    edges = edges.numpy() if isinstance(edges, torch.Tensor) else np.asarray(edges)
    graph = csr_matrix((np.ones(edges.shape[1]), edges), shape=(nu, ni))
    uc, ic = ns["double_frequency_hash"](graph, ku, ki, nu, ni)
    uc, ic = np.asarray(uc, dtype=np.int64), np.asarray(ic, dtype=np.int64)
    assert uc.shape == (nu, 2) and ic.shape == (ni, 2)
    assert 0 <= uc.min() and uc.max() < ku and 0 <= ic.min() and ic.max() < ki
    return uc, ic


# ----------------------------------------------------------------------------- data
def _dataset_dir_candidates(dataset, data_root):
    roots = []
    if data_root:
        roots += [Path(data_root)]
    roots += [HERE / "_work", HERE, Path("/kaggle/working/_work"), Path("/kaggle/working"), LOCAL_REPRO,
              LOCAL_REPRO / "_work"]
    if "repro_run" in sys.modules:
        roots += [sys.modules["repro_run"].WORK, sys.modules["repro_run"].ROOT]
    for r in roots:
        yield r / "datasets" / f"{dataset}-processed"
        yield r / f"{dataset}-processed"


def find_dataset_dir(dataset, data_root=None):
    for c in _dataset_dir_candidates(dataset, data_root):
        if (c / "train.pt").exists() and (c / "val.pt").exists() and (c / "test.pt").exists():
            return c
    return None


def prepare_repro():
    """R.prepare() once (Kaggle image: clustering package, GraphHash repo, dataset extraction)."""
    if "out" not in _PREPARED:
        R = import_repro_run()
        _PREPARED["R"] = R
        _PREPARED["out"] = R.prepare()
    return _PREPARED["R"], _PREPARED["out"]


def load_splits(dataset, data_root=None):
    """Splits, truth dictionaries and counts, exactly as repro_run.load_data builds them, without clustering."""
    rc = import_repro_core()
    path = find_dataset_dir(dataset, data_root)
    if path is None:
        prepare_repro()                      # Kaggle image: extracts the datasets (fails on a CPU-only machine)
        path = find_dataset_dir(dataset, data_root)
    if path is None:
        raise FileNotFoundError(f"{dataset}-processed splits not found (use --data-root)")
    parts = [torch.load(path / f"{s}.pt", weights_only=True).long() for s in ["train", "val", "test"]]
    info = rc.split_statistics(parts)
    nu, ni = info["user_id_max"] + 1, info["item_id_max"] + 1
    assert nu == info["users"] and ni == info["items"], "Nonconsecutive IDs require an explicit mapping."
    dicts = []
    for part in parts:
        d = {}
        for uid, iid in part.t().tolist():
            d.setdefault(uid, []).append(iid)
        dicts.append(d)
    train_truth, val_truth, test_truth = dicts
    counts = np.bincount(parts[0][1].numpy(), minlength=ni).astype(np.int64)
    merged = {u: train_truth.get(u, []) + val_truth.get(u, []) for u in set(train_truth) | set(val_truth)}
    return {"parts": parts, "train_truth": train_truth, "val_truth": val_truth, "test_truth": test_truth,
            "merged": merged, "nu": nu, "ni": ni, "counts": counts, "info": info, "dir": str(path)}


# ----------------------------------------------------------------------------- mapping
def resolve_mapping(args, ck, data):
    """(user_alias, item_alias, source). Hash mappings are never recomputed on the CPU image (TASKS rule 4)."""
    nu, ni = data["nu"], data["ni"]
    ut, it = ck["user_table"].shape[0], ck["item_table"].shape[0]
    if ck["aliases"] is not None:
        ua, ia = ck["aliases"]
        source = "stored"
    elif args.method == "Full":
        ua, ia, source = np.arange(nu, dtype=np.int64), np.arange(ni, dtype=np.int64), "identity"
    elif args.method == "Random":
        # repro_run.load_data: Random = (arange(n) % k_users, arange(n) % k_items) with the GraphHash bucket counts,
        # which are the checkpoint's table sizes; no clustering is involved.
        ua, ia, source = np.arange(nu, dtype=np.int64) % ut, np.arange(ni, dtype=np.int64) % it, "modulo(table rows)"
    else:
        if not torch.cuda.is_available() and os.environ.get("ALLOW_CPU_RECOMPUTE") != "1":
            raise SystemExit(f"TASKS rule 4: refusing to recompute the {args.method} mapping on the CPU image "
                             "(checkpoint has no stored aliases). Run on the GPU image.")
        R, (_, _, _, _, ns) = prepare_repro()
        parts, maps, _, info = R.load_data(args.dataset, ns)
        assert torch.equal(parts[0], data["parts"][0]), "recomputed train split differs from the loaded one"
        if args.method == "GraphHash":
            ua, ia = maps["GraphHash"]
        else:
            ua, ia = double_freq_maps(R.WORK / "GraphHash" / "utils.py", parts, nu, ni,
                                      info["user_buckets"], info["item_buckets"])
        ua, ia, source = to_alias(ua), to_alias(ia), "recomputed_on_gpu_image"
    ua, ia = to_alias(ua), to_alias(ia)
    if ua.shape[0] != nu or ia.shape[0] != ni:
        raise SystemExit(f"alias lengths {ua.shape[0]}/{ia.shape[0]} do not match the dataset {nu}/{ni}")
    if ua.max() >= ut or ia.max() >= it or ua.min() < 0 or ia.min() < 0:
        raise SystemExit(f"aliases index outside the checkpoint tables ({ut}, {it} rows)")
    if args.method == "DoubleFreq" and (ia.ndim != 2 or ua.ndim != 2):
        raise SystemExit("DoubleFreq needs (n, 2) aliases")
    if args.method in ("Full", "GraphHash", "Random") and (ia.ndim != 1 or ua.ndim != 1):
        raise SystemExit(f"{args.method} needs 1-D aliases")
    return ua, ia, source


def check_assert_aliases(args, ua, ia, out_dir, fetch_dir):
    """--assert-aliases: the mapping used must equal the aliases stored in another checkpoint, else abort."""
    if not args.assert_aliases:
        return None
    other = load_checkpoint(fetch_to(args.assert_aliases, fetch_dir, "assert_aliases.pt"))
    if other["aliases"] is None:
        res = {"equal": False, "error": "assert-aliases checkpoint has no stored aliases"}
    else:
        ou, oi = other["aliases"]
        res = {"users_equal": bool(np.array_equal(ua, ou)), "items_equal": bool(np.array_equal(ia, oi)),
               "other_users_sha256": sha256_array(ou), "other_items_sha256": sha256_array(oi)}
        res["equal"] = res["users_equal"] and res["items_equal"]
    if not res["equal"]:
        (out_dir / "alias_assert_failed.json").write_text(json.dumps(res, indent=1, default=_jd))
        log("alias_assert_failed", **res)
        raise SystemExit(3)
    log("alias_assert_ok", **res)
    return res


# ----------------------------------------------------------------------------- cells, plans, scorer
class Cell:
    """One ranking to evaluate: vector set, base score, optional additive item term (beta*term), secondary key."""
    __slots__ = ("name", "vec", "base", "term", "beta", "key", "legacy", "keep_top", "tie_stats")

    def __init__(self, vec="orig", base="cos", term=None, beta=0.0, key=("random", 0), legacy=False,
                 keep_top=False, tie_stats=False):
        if term is None or not beta:
            term, beta = None, 0.0
        self.vec, self.base, self.term, self.beta = vec, base, term, float(beta)
        self.key = None if legacy else tuple(key)
        self.legacy, self.keep_top, self.tie_stats = legacy, keep_top, tie_stats
        k = "-" if self.key is None else ":".join(map(str, self.key))
        self.name = f"{'L' if legacy else 'S'}|{vec}|{base}|{term or '-'}|{self.beta:g}|{k}"


def cname(**kw):
    return Cell(**kw).name


class Plan:
    def __init__(self):
        self.cells = {}

    def add(self, **kw):
        c = Cell(**kw)
        old = self.cells.get(c.name)
        if old is None:
            self.cells[c.name] = c
        else:
            old.keep_top |= c.keep_top
            old.tie_stats |= c.tie_stats
        return c.name


class Scorer:
    """Bucket-level scoring + lexicographic ranking for user chunks."""

    def __init__(self, uvec, n_items, counts, device):
        self.device, self.N = torch.device(device), int(n_items)
        self.uvec = uvec.to(self.device).float()
        self.counts = np.asarray(counts)
        self.vecs, self.terms, self._keys, self._labpos, self.tested = {}, {}, {}, {}, set()

    # -- registration
    def add_vecset(self, name, V, labels=None):
        V = V.to(self.device).float().contiguous()
        lab = None if labels is None else torch.as_tensor(np.asarray(labels), dtype=torch.long, device=self.device)
        self.vecs[name] = (V, F.normalize(V, dim=-1), lab)

    def add_term(self, name, arr):
        self.terms[name] = torch.as_tensor(np.asarray(arr, dtype=np.float32), device=self.device)

    # -- secondary keys: (perm, inv) with perm[j] = item at priority j
    def key(self, spec):
        if spec not in self._keys:
            if spec[0] == "random":
                perm = np.random.RandomState(spec[1]).permutation(self.N)
            elif spec[0] == "index":
                perm = np.arange(self.N)
            elif spec[0] == "pop":
                perm = np.argsort(-self.counts, kind="stable")
            else:
                raise ValueError(spec)
            perm_t = torch.as_tensor(perm, dtype=torch.long, device=self.device)
            inv = torch.empty_like(perm_t)
            inv[perm_t] = torch.arange(self.N, device=self.device)
            self._keys[spec] = (perm_t, inv)
        return self._keys[spec]

    def _labels_in_key_order(self, vec, spec):
        if (vec, spec) not in self._labpos:
            perm, _ = self.key(spec)
            lab = self.vecs[vec][2]
            self._labpos[(vec, spec)] = perm if lab is None else lab[perm]
        return self._labpos[(vec, spec)]

    def bucket_scores(self, u, vec, base):
        """(b, n_groups) scores: user x unique bucket vectors (cosine: both sides normalized)."""
        V, Vn, _ = self.vecs[vec]
        return u @ V.t() if base == "raw" else F.normalize(u, dim=-1) @ Vn.t()

    def item_scores(self, u, vec, base):
        """(b, N) scores copied from the bucket scores: bit-identical within a bucket."""
        sb = self.bucket_scores(u, vec, base)
        lab = self.vecs[vec][2]
        return sb if lab is None else sb[:, lab]

    def _coords(self, users, d):
        lens = [len(d.get(x, ())) for x in users]
        if not sum(lens):
            return None, None
        rows = np.repeat(np.arange(len(users)), lens)
        cols = np.concatenate([np.asarray(d[x], dtype=np.int64) for x in users if len(d.get(x, ()))])
        return (torch.as_tensor(rows, device=self.device), torch.as_tensor(cols, device=self.device))

    @torch.no_grad()
    def run(self, split, plan, truth, masks, max_users=0, chunk=CHUNK, k=K, tag=""):
        """Evaluate every cell of the plan on the users of `truth`. Returns {cell name: result dict}."""
        if split == "test":
            dup = sorted(n for n in plan.cells if n in self.tested)
            if dup:
                raise RuntimeError(f"test cell(s) scored twice: {dup[:3]}")
            self.tested.update(plan.cells)
        dev = self.device
        users = np.fromiter(truth.keys(), dtype=np.int64, count=len(truth))
        if max_users:
            users = users[:max_users]
        npos = np.array([len(truth[u]) for u in users], dtype=np.int64)
        # group cells by (vec, base) so only one (b, n_groups) score matrix is alive at a time
        groups = {}
        for c in plan.cells.values():
            groups.setdefault((c.vec, c.base), []).append(c)
        acc = {n: {"hits": [], "top": [] if c.keep_top else None, "tie": [] if c.tie_stats else None}
               for n, c in plan.cells.items()}
        t0 = time.perf_counter()
        for s in range(0, len(users), chunk):
            uid = users[s:s + chunk]
            u = self.uvec[torch.as_tensor(uid, device=dev)]
            mrow, mcol = self._coords(uid, masks)
            trow, tcol = self._coords(uid, truth)
            tmask = torch.zeros(len(uid), self.N, dtype=torch.bool, device=dev)
            tmask[trow, tcol] = True
            for (vec, base), cells in groups.items():
                sb = self.bucket_scores(u, vec, base)
                lab = self.vecs[vec][2]
                for c in cells:
                    term = None if c.term is None else self.terms[c.term]
                    if c.legacy:
                        assert lab is None, "legacy scoring is item-level"
                        S = sb.clone() if term is None else sb + c.beta * term[None, :]
                        if mrow is not None:
                            S[mrow, mcol] = MASKED
                        top = torch.topk(S, k, dim=1).indices
                    else:
                        perm, inv = self.key(c.key)
                        S = sb[:, self._labels_in_key_order(vec, c.key)]
                        if term is not None:
                            S += c.beta * term[perm][None, :]
                        if mrow is not None:
                            S[mrow, inv[mcol]] = MASKED
                        pos = torch.sort(S, dim=1, descending=True, stable=True).indices[:, :k]
                        top = perm[pos]
                        if c.tie_stats:
                            vals = S.gather(1, pos)
                            s20 = vals[:, k - 1:k]
                            tied_all = ((S == s20) & (S > MASKED / 2)).sum(1)
                            tied_in = (vals == s20).sum(1)
                            acc[c.name]["tie"].append(torch.stack([tied_all, tied_all - tied_in], 1).cpu())
                    acc[c.name]["hits"].append(tmask.gather(1, top).cpu())
                    if c.keep_top:
                        acc[c.name]["top"].append(top.cpu())
            if (s // chunk) % 20 == 0:
                log("progress", tag=tag, split=split, users=int(s + len(uid)), of=len(users), cells=len(plan.cells),
                    seconds=round(time.perf_counter() - t0, 1))
        out = {}
        for n, a in acc.items():
            hits = torch.cat(a["hits"]).numpy().astype(bool)
            rec_u, ndcg_u = per_user_metrics(hits, npos, k)
            r = {"hits": hits, "recall_u": rec_u, "ndcg_u": ndcg_u, "recall": float(rec_u.mean()),
                 "ndcg": float(ndcg_u.mean()), "users": users, "npos": npos, "top": None, "tie": None}
            if a["top"] is not None:
                r["top"] = torch.cat(a["top"]).numpy().astype(np.int32)
            if a["tie"] is not None:
                t = torch.cat(a["tie"]).numpy()
                total, excluded = t[:, 0], t[:, 1]
                tied = excluded >= 1
                r["tie"] = {"n_users": int(len(t)), "share_boundary_tie": float(tied.mean()),
                            "mean_tied_group_size_all_users": float(total.mean()),
                            "mean_tied_group_size_tied_users": float(total[tied].mean()) if tied.any() else 0.0,
                            "mean_tied_items_excluded_all_users": float(excluded.mean()),
                            "mean_tied_items_excluded_tied_users": float(excluded[tied].mean()) if tied.any() else 0.0,
                            "max_tied_group_size": int(total.max())}
            out[n] = r
        log("run_done", tag=tag, split=split, cells=len(plan.cells), users=len(users),
            seconds=round(time.perf_counter() - t0, 1))
        return out


# ----------------------------------------------------------------------------- aggregation helpers
def single(res, name):
    return {"recall": res[name]["recall"], "ndcg": res[name]["ndcg"]}


def agg(res, names):
    r = np.array([res[n]["recall"] for n in names])
    g = np.array([res[n]["ndcg"] for n in names])
    return {"n_seeds": len(names), "recall": {"mean": float(r.mean()), "sd": sd(r), "per_seed": r.tolist()},
            "ndcg": {"mean": float(g.mean()), "sd": sd(g), "per_seed": g.tolist()}}


def grid(res, **kw):
    """{beta: recall} for a cell family on one split."""
    return {b: res[cname(beta=b, **kw)]["recall"] for b in BETAS}


def save_per_user(path, r):
    np.savez_compressed(path, users=r["users"], top20=r["top"], hits=r["hits"].astype(np.uint8), npos=r["npos"],
                        ndcg=r["ndcg_u"], recall=r["recall_u"])


def decomposition_report(base_names, mu_names, eps_names, both_names, res):
    """Decomposition from the seed means (primary) and per seed (paired by key seed) with mean +- sd."""
    out = {}
    for metric in ("recall", "ndcg"):
        cell = lambda names: np.array([res[n][metric] for n in names])
        b, m, e, f = cell(base_names), cell(mu_names), cell(eps_names), cell(both_names)
        from_means = shapley_decomposition(float(b.mean()), float(m.mean()), float(e.mean()), float(f.mean()))
        per_seed = [shapley_decomposition(*map(float, x)) for x in zip(b, m, e, f)]
        stats = {}
        for key in from_means:
            vals = [d[key] for d in per_seed if d[key] is not None]
            stats[key] = {"mean": float(np.mean(vals)), "sd": sd(vals)} if vals else None
        out[metric] = {"from_seed_means": from_means, "per_seed_mean_sd": stats,
                       "cells": {"base": float(b.mean()), "mu": float(m.mean()), "eps": float(e.mean()),
                                 "both": float(f.mean())}}
    return out


# ----------------------------------------------------------------------------- the pipeline
def run_pipeline(args, ck, data, ua, ia, source, out_dir, meta):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(range(args.seeds))
    base = args.base
    other = "raw" if base == "cos" else "cos"
    N = data["ni"]
    counts = data["counts"]
    ut, it = ck["user_table"].to(dev), ck["item_table"].to(dev)
    uvec, ivec = table_vec(ut, ua), table_vec(it, ia)
    labels, G, rep = group_labels(ia, canonical=True)
    _, G_ordered, _ = group_labels(ia, canonical=False)
    sc = Scorer(uvec, N, counts, dev)
    sc.add_vecset("items", ivec, None)                                       # item level: legacy scoring only
    sc.add_vecset("orig", ivec[torch.as_tensor(rep, device=dev)], labels)    # bucket level
    p = np.log1p(counts.astype(np.float64))
    mu_g, mu, eps = mu_eps(p, labels, G)
    eps_sum_max = float(np.abs(np.bincount(labels, weights=eps, minlength=G)).max())
    var_p, var_mu, var_eps = float(p.var()), float(mu.var()), float(eps.var())
    pu8, u8info = quantize_u8(p)
    sc.add_term("p", p)
    sc.add_term("mu", mu)
    sc.add_term("eps", eps)
    sc.add_term("pu8", pu8)
    shuf_check = 0.0
    for s in seeds:
        ps = shuffle_within_buckets(p, labels, 10_000 + s)
        shuf_check = max(shuf_check, float(np.abs(mu_eps(ps, labels, G)[0] - mu_g).max()))
        sc.add_term(f"pshuf{s}", ps)
    meta["identity"] = {"n_groups": G, "n_groups_ordered_tuples": G_ordered, "n_items": N,
                        "var_p": var_p, "var_mu": var_mu, "var_eps": var_eps,
                        "var_mu_over_var_p": var_mu / var_p if var_p > 0 else None,
                        "max_abs_sum_eps_per_bucket": eps_sum_max, "max_abs_mu_shift_within_bucket_shuffle": shuf_check,
                        "p_max": float(p.max()), "u8": u8info, "n_items_count0": int((counts == 0).sum())}
    e2e_names, e2e_fit = [], None
    if args.e2e:
        if not np.array_equal(ia, np.arange(N)):
            raise SystemExit("--e2e needs the Full mapping (identity item aliases)")
        variants, e2e_fit = length_variants(ivec.cpu().numpy(), counts, seeds)
        dirs = ivec / ivec.norm(dim=1, keepdim=True).clamp_min(1e-12)
        sc.add_vecset("e2e:orig", ivec, None)
        e2e_names.append("orig")
        for name, ln in variants.items():
            sc.add_vecset(f"e2e:{name}", dirs * torch.as_tensor(ln, dtype=torch.float32, device=dev)[:, None], None)
            e2e_names.append(name)
        del dirs
    meta["e2e_fit"] = e2e_fit

    def val_plan():
        pl = Plan()
        for b in ("cos", "raw"):
            for beta in BETAS:
                pl.add(vec="items", base=b, term="p", beta=beta, legacy=True)
        for beta in BETAS:
            for term in ("p", "mu", "eps"):
                pl.add(vec="orig", base=base, term=term, beta=beta, key=("random", 0))
            pl.add(vec="orig", base=other, term="p", beta=beta, key=("random", 0))
        for n in e2e_names:
            pl.add(vec=f"e2e:{n}", base="raw", key=("random", 0))
        return pl

    t0 = time.perf_counter()
    vplan = val_plan()
    vres = sc.run("val", vplan, data["val_truth"], data["train_truth"], args.max_users, tag="val")
    meta["n_users_val_evaluated"] = int(len(next(iter(vres.values()))["users"]))

    # ---- selections (validation only)
    kw = lambda b, term, vec="orig": dict(vec=vec, base=b, term=term, key=("random", 0))
    val_curves = {"legacy_cos": grid(vres, vec="items", base="cos", term="p", legacy=True),
                  "legacy_raw": grid(vres, vec="items", base="raw", term="p", legacy=True),
                  "full_prior": grid(vres, **kw(base, "p")), "other_prior": grid(vres, **kw(other, "p")),
                  "mu": grid(vres, **kw(base, "mu")), "eps": grid(vres, **kw(base, "eps"))}
    best = {k: pick_beta(v) for k, v in val_curves.items()}
    bstar = best["full_prior"]
    log("selected", beta_star=bstar, **{f"beta_{k}": v for k, v in best.items()})

    # ---- test plan: every cell once
    tp = Plan()
    L = lambda b, beta: dict(vec="items", base=b, term="p", beta=beta, legacy=True)
    for b in ("cos", "raw"):
        tp.add(**L(b, 0.0), keep_top=True)
        tp.add(**L(b, best[f"legacy_{b}"]))
    R_ = lambda s: ("random", s)
    names = {}
    names["base_rand"] = [tp.add(vec="orig", base=base, key=R_(s), keep_top=(s == 0), tie_stats=(s == 0)) for s in seeds]
    names["base_index"] = tp.add(vec="orig", base=base, key=("index",))
    names["base_pop"] = tp.add(vec="orig", base=base, key=("pop",), keep_top=True)
    names["full"] = [tp.add(vec="orig", base=base, term="p", beta=bstar, key=R_(s), keep_top=(s == 0),
                            tie_stats=(s == 0)) for s in seeds]
    names["mu"] = [tp.add(vec="orig", base=base, term="mu", beta=bstar, key=R_(s), keep_top=(s == 0)) for s in seeds]
    names["eps"] = [tp.add(vec="orig", base=base, term="eps", beta=bstar, key=R_(s), keep_top=(s == 0)) for s in seeds]
    names["mu_rt"] = [tp.add(vec="orig", base=base, term="mu", beta=best["mu"], key=R_(s)) for s in seeds]
    names["eps_rt"] = [tp.add(vec="orig", base=base, term="eps", beta=best["eps"], key=R_(s)) for s in seeds]
    names["shuffle"] = [tp.add(vec="orig", base=base, term=f"pshuf{s}", beta=bstar, key=R_(s)) for s in seeds]
    names["other"] = [tp.add(vec="orig", base=other, term="p", beta=best["other_prior"], key=R_(s)) for s in seeds]
    names["u8"] = [tp.add(vec="orig", base=base, term="pu8", beta=bstar, key=R_(s)) for s in seeds]
    for n in e2e_names:
        tp.add(vec=f"e2e:{n}", base="raw", key=("random", 0))
    tres = sc.run("test", tp, data["test_truth"], data["merged"], args.max_users, tag="test")
    meta["n_users_test_evaluated"] = int(len(next(iter(tres.values()))["users"]))
    meta["n_test_cells_scored"] = len(sc.tested)

    # ---- assemble
    out = {}
    rc = import_repro_core()
    legacy = {}
    for b in ("cos", "raw"):
        lb = best[f"legacy_{b}"]
        t0n = cname(**L(b, 0.0))
        topk = tres[t0n]["top"]
        ref = rc.metrics_from_topk(topk, [data["test_truth"][u] for u in tres[t0n]["users"]], K)
        legacy[b] = {"beta0": {"val": single(vres, t0n), "test": single(tres, t0n)},
                     "prior": {"best_beta": lb, "val_by_beta": val_curves[f"legacy_{b}"],
                               "val": single(vres, cname(**L(b, lb))), "test": single(tres, cname(**L(b, lb)))},
                     "crosscheck_vs_repro_core_metrics_from_topk": {
                         "max_abs_diff": max(abs(ref["recall"] - tres[t0n]["recall"]),
                                             abs(ref["ndcg"] - tres[t0n]["ndcg"])),
                         "evaluated_users": ref["evaluated_users"]}}
    out["legacy_topk"] = legacy
    out["tie_keys"] = {"base": base, "random": {"test": agg(tres, names["base_rand"]),
                                                "val_seed0": single(vres, names["base_rand"][0])},
                       "index": {"test": single(tres, names["base_index"])},
                       "pop": {"test": single(tres, names["base_pop"])},
                       "delta_pop_minus_random": {"recall": tres[names["base_pop"]]["recall"] - np.mean([tres[n]["recall"] for n in names["base_rand"]]),
                                                  "ndcg": tres[names["base_pop"]]["ndcg"] - np.mean([tres[n]["ndcg"] for n in names["base_rand"]])},
                       "delta_index_minus_random": {"recall": tres[names["base_index"]]["recall"] - np.mean([tres[n]["recall"] for n in names["base_rand"]]),
                                                    "ndcg": tres[names["base_index"]]["ndcg"] - np.mean([tres[n]["ndcg"] for n in names["base_rand"]])}}
    full_blk = {"base": base, "best_beta": bstar, "val_by_beta": val_curves["full_prior"],
                "val_seed0": single(vres, names["full"][0]), "test": agg(tres, names["full"])}
    out["full_prior"] = full_blk
    out["decomposition_fixed_beta"] = {"beta_star": bstar, "base": base,
                                       **decomposition_report(names["base_rand"], names["mu"], names["eps"],
                                                              names["full"], tres)}
    rt = {"mu": {"best_beta": best["mu"], "val_by_beta": val_curves["mu"], "test": agg(tres, names["mu_rt"])},
          "eps": {"best_beta": best["eps"], "val_by_beta": val_curves["eps"], "test": agg(tres, names["eps_rt"])},
          "both": {"best_beta": bstar, "test": full_blk["test"]},
          "base": {"best_beta": 0.0, "test": out["tie_keys"]["random"]["test"]}}
    out["decomposition_retuned"] = {**decomposition_report(names["base_rand"], names["mu_rt"], names["eps_rt"],
                                                           names["full"], tres), "cells": rt}
    out["within_bucket_shuffle"] = {"beta_star": bstar, "n_seeds": len(seeds), "test": agg(tres, names["shuffle"]),
                                    "gain_vs_random_base_recall": {
                                        "shuffle": float(np.mean([tres[n]["recall"] for n in names["shuffle"]]) - np.mean([tres[n]["recall"] for n in names["base_rand"]])),
                                        "full": float(np.mean([tres[n]["recall"] for n in names["full"]]) - np.mean([tres[n]["recall"] for n in names["base_rand"]]))}}
    g = out["within_bucket_shuffle"]["gain_vs_random_base_recall"]
    g["share_of_full_gain_lost"] = None if g["full"] == 0 else 1.0 - g["shuffle"] / g["full"]
    other_blk = {"base": other, "best_beta": best["other_prior"], "val_by_beta": val_curves["other_prior"],
                 "test": agg(tres, names["other"])}
    out["cos_prior"], out["raw_prior"] = (full_blk, other_blk) if base == "cos" else (other_blk, full_blk)
    out[f"{base}_prior_u8"] = {"beta_star": bstar, "quantization": u8info, "test": agg(tres, names["u8"]),
                               "delta_vs_unquantized_recall": float(np.mean([tres[n]["recall"] for n in names["u8"]]) - np.mean([tres[n]["recall"] for n in names["full"]]))}
    out["tie_stats"] = {"split": "test", "key": "random seed 0",
                        "base_beta0": tres[names["base_rand"][0]]["tie"],
                        "full_prior_beta_star": tres[names["full"][0]]["tie"]}
    if args.e2e:
        e = {"fit_log_length_on_p": e2e_fit, "scoring": "raw dot product, directions kept, lengths modified",
             "unmodified": {"val": single(vres, cname(vec="e2e:orig", base="raw", key=R_(0))),
                            "test": single(tres, cname(vec="e2e:orig", base="raw", key=R_(0)))}}
        for mode in ("len_shuffle_decile", "len_shuffle_global"):
            vn = [cname(vec=f"e2e:{mode}_s{s}", base="raw", key=R_(0)) for s in seeds]
            e[mode] = {"val": agg(vres, vn), "test": agg(tres, vn)}
        fn = cname(vec="e2e:len_replace_fit", base="raw", key=R_(0))
        e["len_replace_fit"] = {"val": single(vres, fn), "test": single(tres, fn)}
        for mode in ("len_shuffle_decile", "len_shuffle_global"):
            for split in ("val", "test"):
                e[mode][f"delta_{split}_recall"] = e[mode][split]["recall"]["mean"] - e["unmodified"][split]["recall"]
        for split in ("val", "test"):
            e["len_replace_fit"][f"delta_{split}_recall"] = e["len_replace_fit"][split]["recall"] - e["unmodified"][split]["recall"]
        out["e2e"] = e

    files = []
    for tag, name in [("base_randomkey_s0", names["base_rand"][0]), ("base_popkey", names["base_pop"]),
                      ("full_prior_s0", names["full"][0]), ("cell_mu_s0", names["mu"][0]),
                      ("cell_eps_s0", names["eps"][0])]:
        f = f"per_user__{tag}.npz"
        save_per_user(out_dir / f, tres[name])
        files.append(f)
    out["per_user_files"] = {"files": files, "split": "test", "fields": "users, top20, hits, npos, ndcg, recall",
                             "seed": "0 where the cell has seeds"}
    out["selected_betas"] = best
    meta["timing_seconds"] = round(time.perf_counter() - t0, 1)
    return out


# ----------------------------------------------------------------------------- post-Stage-3 rescoring (--post-s3)
def quantize_u8_f32(p):
    """Spec quantizer: q = round(255 p / max p) as uint8, float32 scale = max p / 255, decoded = q * scale (float32)."""
    p = np.asarray(p, dtype=np.float64)
    pmax = float(p.max())
    q = np.rint(255.0 * p / pmax).astype(np.uint8)
    scale = np.float32(pmax / 255.0)
    dec = q.astype(np.float32) * scale
    return q, scale, dec, {"p_max": pmax, "scale_float32": float(scale), "payload_bytes": int(q.nbytes) + 4,
                           "max_abs_error": float(np.abs(dec.astype(np.float64) - p).max()),
                           "n_levels_used": int(len(np.unique(q)))}


def train_count_weights(counts):
    """E9 weights (1 + train_count)^(-1/2)."""
    return (1.0 + np.asarray(counts, dtype=np.float64)) ** -0.5


def weighted_recall_u(hits, top, users, truth, w):
    """Per-user weighted recall: recovered truth weight / weight of ALL held-out truth. Equal weights -> hits/|T| (uncapped)."""
    w = np.asarray(w, dtype=np.float64)
    top = np.asarray(top).astype(np.int64)
    num = (w[top] * np.asarray(hits, dtype=np.float64)).sum(1)
    den = np.array([w[np.asarray(truth[int(u)], dtype=np.int64)].sum() for u in users], dtype=np.float64)
    return num / den, num, den


def seen_overlap(top, users, masks):
    """Number of recommended items that are masked (must be 0)."""
    return int(sum(len(set(t.tolist()) & set(masks.get(int(u), ()))) for t, u in zip(np.asarray(top), users)))


def select_beta_weighted(wval_by_beta):
    """E9 selection: VALIDATION weighted recall only (this function cannot see test); ties -> smaller beta."""
    return pick_beta(wval_by_beta)


def _add_weighted(res, truth, w):
    for r in res.values():
        r["wrecall_u"], r["w_num"], r["w_den"] = weighted_recall_u(r["hits"], r["top"], r["users"], truth, w)
        r["wrecall"] = float(r["wrecall_u"].mean())


def post_s3_val_stage(sc, data, w, beta, max_users=0):
    """All validation cells and the weighted beta selection. Reads only val_truth and train_truth."""
    pl = Plan()
    grid_names = {b: pl.add(vec="orig", base="raw", term="p", beta=b, key=("random", 0), keep_top=True) for b in BETAS}
    if beta not in grid_names:
        grid_names[beta] = pl.add(vec="orig", base="raw", term="p", beta=beta, key=("random", 0), keep_top=True)
    u8_name = pl.add(vec="orig", base="raw", term="pu8", beta=beta, key=("random", 0), keep_top=True)
    res = sc.run("val", pl, data["val_truth"], data["train_truth"], max_users, tag="post_s3_val")
    _add_weighted(res, data["val_truth"], w)
    wcurve = {b: res[grid_names[b]]["wrecall"] for b in BETAS}
    return {"res": res, "plan": pl, "grid_names": grid_names, "f32": grid_names[beta], "u8": u8_name,
            "weighted_val_by_beta": wcurve, "recall_val_by_beta": {b: res[grid_names[b]]["recall"] for b in BETAS},
            "selected_beta": select_beta_weighted(wcurve)}


def post_s3_test_stage(sc, data, w, beta, bsel, seeds, max_users=0):
    """Every test cell once; beta and bsel are frozen. Returns (results by cell name, role -> cell name, plan)."""
    pl = Plan()
    roles = {}
    roles["e5_cos_s0"] = pl.add(vec="orig", base="cos", key=("random", 0), keep_top=True)
    roles["e5_cos_legacy_topk"] = pl.add(vec="items", base="cos", legacy=True, keep_top=True)
    roles["e7_f32"] = pl.add(vec="orig", base="raw", term="p", beta=beta, key=("random", 0), keep_top=True)
    roles["e7_u8"] = pl.add(vec="orig", base="raw", term="pu8", beta=beta, key=("random", 0), keep_top=True)
    roles["e9_fixed"] = roles["e7_f32"]
    roles["e9_selected"] = pl.add(vec="orig", base="raw", term="p", beta=bsel, key=("random", 0), keep_top=True)
    roles["e8_index"] = pl.add(vec="orig", base="raw", term="p", beta=beta, key=("index",), keep_top=True)
    roles["e8_pop"] = pl.add(vec="orig", base="raw", term="p", beta=beta, key=("pop",), keep_top=True)
    for s in seeds:
        roles[f"e8_random_s{s}"] = pl.add(vec="orig", base="raw", term="p", beta=beta, key=("random", s), keep_top=True)
    roles["e8_legacy_topk"] = pl.add(vec="items", base="raw", term="p", beta=beta, legacy=True, keep_top=True)
    res = sc.run("test", pl, data["test_truth"], data["merged"], max_users, tag="post_s3_test")
    _add_weighted(res, data["test_truth"], w)
    return res, roles, pl


def save_cell_npz(path, r, **extra):
    np.savez_compressed(path, users=r["users"], top20=r["top"], hits=r["hits"].astype(np.uint8), npos=r["npos"],
                        ndcg=r["ndcg_u"], recall=r["recall_u"], wrecall=r["wrecall_u"], w_recovered=r["w_num"],
                        w_truth_total=r["w_den"], **extra)


def run_post_s3(args, ck, data, ua, ia, source, out_dir, meta):
    """One scoring pass: E5 cosine, E7 float32 vs uint8 prior, E9 weighted beta retune, E8 full-prior tie policies."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.perf_counter()
    beta, N, counts = float(args.beta), data["ni"], data["counts"]
    seeds = list(range(args.seeds))
    ut, it = ck["user_table"].to(dev), ck["item_table"].to(dev)
    uvec, ivec = table_vec(ut, ua), table_vec(it, ia)
    labels, G, rep = group_labels(ia, canonical=True)
    sc = Scorer(uvec, N, counts, dev)
    sc.add_vecset("items", ivec, None)
    sc.add_vecset("orig", ivec[torch.as_tensor(rep, device=dev)], labels)
    p32 = np.log1p(counts.astype(np.float64)).astype(np.float32)
    q, scale, dec, u8info = quantize_u8_f32(p32)
    sc.add_term("p", p32)
    sc.add_term("pu8", dec)
    w = train_count_weights(counts)
    np.savez_compressed(out_dir / "prior_u8_payload.npz", q=q, scale=np.array(scale, dtype=np.float32))
    meta["u8"] = u8info
    for s in seeds:                                      # every random key is a valid permutation of all items
        assert np.array_equal(np.sort(sc.key(("random", s))[0].cpu().numpy()), np.arange(N))
    vs = post_s3_val_stage(sc, data, w, beta, args.max_users)
    bsel = vs["selected_beta"]
    log("post_s3_selected", beta_fixed=beta, beta_selected=bsel)
    tres, roles, tplan = post_s3_test_stage(sc, data, w, beta, bsel, seeds, args.max_users)
    vres = vs["res"]
    meta["n_users_val_evaluated"] = int(len(next(iter(vres.values()))["users"]))
    meta["n_users_test_evaluated"] = int(len(next(iter(tres.values()))["users"]))
    meta["n_test_cells_scored"] = len(sc.tested)

    files, cells = [], {}

    def put(tag, split, name, role):
        res = (vres if split == "val" else tres)[name]
        masks = data["train_truth"] if split == "val" else data["merged"]
        c = (vs["plan"] if split == "val" else tplan).cells[name]
        f = f"per_user__{tag}.npz"
        save_cell_npz(out_dir / f, res, split=np.array(split), policy=np.array(name), beta=np.float64(c.beta),
                      term=np.array(c.term or "-"), prior_complete=np.array(c.term in ("p", "pu8")))
        files.append(f)
        cells[tag] = {"file": f, "split": split, "role": role, "cell": name, "base": c.base, "term": c.term or "-",
                      "beta": c.beta, "key": None if c.key is None else list(c.key), "legacy": c.legacy,
                      "recall": res["recall"], "ndcg": res["ndcg"], "weighted_recall": res["wrecall"],
                      "n_users": int(len(res["users"])),
                      "seen_item_overlap": seen_overlap(res["top"], res["users"], masks)}
    put("e5_cos_randomkey_s0_test", "test", roles["e5_cos_s0"], "e5")
    put("e5_cos_torchtopk_test", "test", roles["e5_cos_legacy_topk"], "e5_sensitivity")
    put("e7_f32_val", "val", vs["f32"], "e7")
    put("e7_u8_val", "val", vs["u8"], "e7")
    put("e7_f32_test", "test", roles["e7_f32"], "e7")
    put("e7_u8_test", "test", roles["e7_u8"], "e7")
    put("e9_test_beta_fixed", "test", roles["e9_fixed"], "e9")
    put("e9_test_beta_selected", "test", roles["e9_selected"], "e9")
    put("e8_test_index", "test", roles["e8_index"], "e8")
    put("e8_test_pop", "test", roles["e8_pop"], "e8")
    for s in seeds:
        put(f"e8_test_random_s{s}", "test", roles[f"e8_random_s{s}"], "e8")
    put("e8_test_torchtopk", "test", roles["e8_legacy_topk"], "e8_sensitivity")

    def pp(a, b):
        return {"signed_pp": 100.0 * (b - a), "abs_pp": abs(100.0 * (b - a))}
    e7 = {"beta": beta, "quantizer": "q=round(255*p/max p), scale=float32(max p/255), decoded=q*scale", "u8": u8info,
          "tolerance_pp_per_split": 0.02, "tie_policy": "random key seed 0"}
    for split, a, b, R in (("val", vs["f32"], vs["u8"], vres), ("test", roles["e7_f32"], roles["e7_u8"], tres)):
        e7[split] = {"f32": single(R, a), "u8": single(R, b), "recall_change": pp(R[a]["recall"], R[b]["recall"]),
                     "ndcg_change": pp(R[a]["ndcg"], R[b]["ndcg"])}
        e7[split]["within_tolerance"] = bool(e7[split]["recall_change"]["abs_pp"] <= 0.02)
    e9 = {"weights": "(1+train_count)^(-1/2)", "denominator": "all held-out truth weight (uncapped)",
          "beta_grid": BETAS, "weighted_val_by_beta": vs["weighted_val_by_beta"],
          "ordinary_recall_val_by_beta": vs["recall_val_by_beta"], "selected_beta": bsel, "fixed_beta": beta,
          "selection_split": "validation only; ties -> smaller beta",
          "test_fixed": {"weighted_recall": tres[roles["e9_fixed"]]["wrecall"], "recall": tres[roles["e9_fixed"]]["recall"]},
          "test_selected": {"weighted_recall": tres[roles["e9_selected"]]["wrecall"],
                            "recall": tres[roles["e9_selected"]]["recall"], "same_cell_as_fixed": bool(bsel == beta)}}
    e8 = {"beta": beta, "base": "raw", "prior": "complete beta*log1p(train_count)",
          "policies": {k: {"recall": tres[n]["recall"], "ndcg": tres[n]["ndcg"], "cell": n}
                       for k, n in roles.items() if k.startswith("e8_")}}
    policy_list = []
    for k, n in roles.items():
        c = tplan.cells[n]
        policy_list.append({"role": k, "cell": n, "base": c.base, "vec": c.vec, "term": c.term or "-", "beta": c.beta,
                            "key": None if c.key is None else list(c.key), "legacy": c.legacy})
    meta["timing_seconds"] = round(time.perf_counter() - t_start, 1)
    return {"post_s3": {"e5_cosine_test": {"randomkey_s0": single(tres, roles["e5_cos_s0"]),
                                           "torchtopk_sensitivity": single(tres, roles["e5_cos_legacy_topk"])},
                        "e7": e7, "e9": e9, "e8": e8},
            "policy_list": policy_list,
            "ranking": "bucket scores copied to items, lexicographic order (stable descending sort over secondary-key "
                       "order); no score perturbation; val masks train, test masks train+val; recall = hits/min(20,n_truth)",
            "cells": cells, "per_user_files": {"files": files + ["prior_u8_payload.npz"]}}


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path or URL")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--base", required=True, choices=["cos", "raw"])
    p.add_argument("--run-id", required=True)
    p.add_argument("--assert-aliases", default=None, help="path or URL of a checkpoint with stored aliases")
    p.add_argument("--e2e", action="store_true", help="length tests (Full models only)")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--out-root", default="/kaggle/working/controls")
    p.add_argument("--data-root", default=None, help="directory containing datasets/<name>-processed (local checks)")
    p.add_argument("--post-s3", action="store_true", help="post-Stage-3 rescoring: E5 cosine, E7, E9, E8 (raw base)")
    p.add_argument("--beta", type=float, default=0.2, help="fixed prior beta for --post-s3")
    p.add_argument("--max-users", type=int, default=0, help="evaluate only the first N users per split (local checks)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    out_dir = Path(args.out_root) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False          # float32 scoring as in the reproduction runs
    torch.backends.cudnn.allow_tf32 = False
    try:
        ck_path = fetch_to(args.ckpt, out_dir / "_inputs", "ckpt.pt")
        ck = load_checkpoint(ck_path)
        data = load_splits(args.dataset, args.data_root)
        ua, ia, source = resolve_mapping(args, ck, data)
        alias_check = check_assert_aliases(args, ua, ia, out_dir, out_dir / "_inputs")
        meta = {"run_id": args.run_id, "dataset": args.dataset, "method": args.method, "base": args.base,
                "args": {k: (source_label(v) if k in ("ckpt", "assert_aliases") else v) for k, v in vars(args).items()},
                "ckpt_name": source_label(args.ckpt),
                "units": "fractions (x100 for percent)", "betas": BETAS, "seeds": list(range(args.seeds)),
                "ckpt_sha256": sha256_file(ck_path), "ckpt_has_b": ck["b"] is not None,
                "ckpt_note": "b (item bias) is not used by these controls" if ck["b"] is not None else None,
                "mapping_source": source, "mapping_users_sha256": sha256_array(ua),
                "mapping_items_sha256": sha256_array(ia), "alias_assert": alias_check,
                "table_rows": [int(ck["user_table"].shape[0]), int(ck["item_table"].shape[0])],
                "n_users_total": data["nu"], "n_items": data["ni"], "split_counts": data["info"]["split_counts"],
                "n_eval_users_val_total": len(data["val_truth"]), "n_eval_users_test_total": len(data["test_truth"]),
                "max_users": args.max_users, "data_dir": data["dir"],
                "device": "cuda" if torch.cuda.is_available() else "cpu", "torch": torch.__version__,
                "k": K, "masked_value": MASKED, "chunk": CHUNK}
        log("start", **{k: meta[k] for k in ("run_id", "dataset", "method", "base", "mapping_source", "device",
                                             "mapping_items_sha256")})
        if args.post_s3:
            if args.base != "raw":
                raise SystemExit("--post-s3 uses the raw-dot base (--base raw)")
            result = run_post_s3(args, ck, data, ua, ia, source, out_dir, meta)
        else:
            result = run_pipeline(args, ck, data, ua, ia, source, out_dir, meta)
        doc = {"schema": "score_controls/post_s3/1" if args.post_s3 else "score_controls/1", "status": "complete", "meta": meta, **result}
        meta["wall_seconds"] = round(time.perf_counter() - started, 1)
        tmp = out_dir / "controls.json.tmp"
        tmp.write_text(json.dumps(doc, indent=1, default=_jd))
        os.replace(tmp, out_dir / "controls.json")
        log("done", out=str(out_dir / "controls.json"))
        return doc
    except SystemExit:
        raise
    except Exception:
        (out_dir / "controls_failure.log").write_text(scrub(traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
