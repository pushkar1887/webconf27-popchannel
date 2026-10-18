"""Unit tests for score_controls.py (T3) and make_maps.py (T4). CPU only, synthetic data, no clustering, no Kaggle.

python test_t3t4.py        (or: pytest test_t3t4.py)
"""
import itertools
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_controls as SC          # noqa: E402
import make_maps as MM               # noqa: E402

RC = SC.import_repro_core()
TMP = Path(tempfile.mkdtemp(prefix="t3t4_"))


# ----------------------------------------------------------------------------- toy builders
def toy_truth(nu, ni, rng, n_train=12, n_val=3, n_test=3):
    """Disjoint train/val/test items per user (dicts), with a popularity skew."""
    w = 1.0 / (1 + np.arange(ni))
    w /= w.sum()
    tr, va, te = {}, {}, {}
    for u in range(nu):
        items = rng.choice(ni, size=n_train + n_val + n_test, replace=False, p=w)
        tr[u], va[u], te[u] = items[:n_train].tolist(), items[n_train:n_train + n_val].tolist(), items[n_train + n_val:].tolist()
    return tr, va, te


def toy_scorer(nu=24, ni=50, d=5, n_buckets=9, seed=0, two_d=False):
    rng = np.random.RandomState(seed)
    ut = torch.randn(nu, d, generator=torch.Generator().manual_seed(seed))
    it = torch.randn(n_buckets, d, generator=torch.Generator().manual_seed(seed + 1))
    ia = rng.randint(n_buckets, size=(ni, 2) if two_d else (ni,))
    if n_buckets == ni and not two_d:
        ia = np.arange(ni)                    # identity (Full-like): every item has its own vector
    if two_d:
        ia[1] = ia[0][::-1]                 # the same unordered pair, swapped
        ia[2] = ia[0]
    tr, va, te = toy_truth(nu, ni, rng)
    counts = np.bincount([i for u in tr for i in tr[u]], minlength=ni)
    ivec = SC.table_vec(it, ia)
    labels, G, rep = SC.group_labels(ia)
    sc = SC.Scorer(SC.table_vec(ut, np.arange(nu)), ni, counts, "cpu")
    sc.add_vecset("items", ivec, None)
    sc.add_vecset("orig", ivec[torch.as_tensor(rep)], labels)
    p = np.log1p(counts.astype(np.float64))
    mu_g, mu, eps = SC.mu_eps(p, labels, G)
    for n, a in [("p", p), ("mu", mu), ("eps", eps)]:
        sc.add_term(n, a)
    merged = {u: tr[u] + va[u] for u in tr}
    return sc, dict(ia=ia, labels=labels, G=G, counts=counts, p=p, tr=tr, va=va, te=te, merged=merged, ivec=ivec,
                    ut=ut, it=it, nu=nu, ni=ni)


def brute_force_topk(sc, T, split, base, term, beta, key, k=SC.K):
    """Independent numpy implementation: full lexicographic order by (-score, priority of the secondary key)."""
    truth, masks = (T["va"], T["tr"]) if split == "val" else (T["te"], T["merged"])
    users = np.array(list(truth))
    S = sc.item_scores(sc.uvec[torch.as_tensor(users)], "orig", base).numpy().copy()
    if term:
        S = S + np.float32(beta) * sc.terms[term].numpy()[None, :]
    for r, u in enumerate(users):
        S[r, masks[u]] = SC.MASKED
    perm = sc.key(key)[0].numpy()
    prio = np.empty(len(perm), dtype=np.int64)
    prio[perm] = np.arange(len(perm))
    return np.stack([np.lexsort((prio, -row))[:k] for row in S]), S


# ----------------------------------------------------------------------------- T3 tests
def test_lexicographic_tiebreak_never_reorders_unequal_scores():
    for two_d in (False, True):
        sc, T = toy_scorer(two_d=two_d)
        for base, term, beta in [("cos", None, 0.0), ("raw", None, 0.0), ("cos", "p", 0.1), ("raw", "mu", 0.5)]:
            keys = [("random", 0), ("random", 7), ("index",), ("pop",)]
            tops, score_seqs = {}, {}
            for key in keys:
                pl = SC.Plan()
                name = pl.add(vec="orig", base=base, term=term, beta=beta, key=key, keep_top=True)
                res = sc.run("val", pl, T["va"], T["tr"])
                top = res[name]["top"]
                ref, S = brute_force_topk(sc, T, "val", base, term, beta, key)
                assert np.array_equal(top, ref), (two_d, base, term, key)      # exactly the lexicographic order
                sv = np.take_along_axis(S, top.astype(np.int64), 1)
                assert (np.diff(sv, axis=1) <= 0).all()                         # non-increasing scores
                outside = S.copy()
                np.put_along_axis(outside, top.astype(np.int64), -np.inf, 1)
                assert (sv[:, -1] >= outside.max(1)).all()                      # nothing better is left out
                tops[key], score_seqs[key] = top, sv
            # the secondary key may only permute items with EQUAL scores: the score sequence never changes
            for key in keys[1:]:
                assert np.array_equal(score_seqs[key], score_seqs[keys[0]])
    # explicit check with known ties: scores [3,2,2,2,2,1], k=2
    V = torch.tensor([[3., 5], [2, 4], [2, 3], [2, 2], [2, 1], [1, 0]])
    sc2 = SC.Scorer(torch.tensor([[1., 0], [0, 1]]), 6, np.array([1, 1, 1, 1, 9, 1]), "cpu")
    sc2.add_vecset("orig", V, None)
    pl = SC.Plan()
    pl.add(vec="orig", base="raw", key=("index",), keep_top=True)
    pl.add(vec="orig", base="raw", key=("pop",), keep_top=True)
    res = sc2.run("val", pl, {0: [0], 1: [0]}, {}, k=2)
    idx, pop = (res[n]["top"] for n in pl.cells)
    assert idx[0].tolist() == [0, 1] and pop[0].tolist() == [0, 4]            # tie among items 1..4 follows the key
    assert idx[1].tolist() == [0, 1] and pop[1].tolist() == [0, 1]            # no tie -> keys irrelevant


def test_tie_statistics_known_case():
    V = torch.tensor([[3., 5], [2, 4], [2, 3], [2, 2], [2, 1], [1, 0]])
    sc = SC.Scorer(torch.tensor([[1., 0], [0, 1]]), 6, np.ones(6, dtype=int), "cpu")
    sc.add_vecset("orig", V, None)
    pl = SC.Plan()
    name = pl.add(vec="orig", base="raw", key=("random", 0), tie_stats=True, keep_top=True)
    res = sc.run("val", pl, {0: [0], 1: [0]}, {}, k=2)[name]
    t = res["tie"]
    assert t["share_boundary_tie"] == 0.5 and t["mean_tied_group_size_all_users"] == 2.5
    assert t["mean_tied_group_size_tied_users"] == 4.0 and t["mean_tied_items_excluded_all_users"] == 1.5
    assert t["mean_tied_items_excluded_tied_users"] == 3.0 and t["max_tied_group_size"] == 4
    # the random key picks different members of the tied group for different seeds
    chosen = set()
    for s in range(12):
        pl = SC.Plan()
        n = pl.add(vec="orig", base="raw", key=("random", s), keep_top=True)
        chosen.add(int(sc.run("val", pl, {0: [0]}, {}, k=2)[n]["top"][0, 1]))
    assert len(chosen) >= 2 and chosen <= {1, 2, 3, 4}
    # masked items are not counted as tied candidates
    pl = SC.Plan()
    n = pl.add(vec="orig", base="raw", key=("random", 0), tie_stats=True)
    t = sc.run("val", pl, {0: [0]}, {0: [1, 2]}, k=2)[n]["tie"]
    assert t["mean_tied_group_size_all_users"] == 2.0 and t["mean_tied_items_excluded_all_users"] == 1.0


def test_bucket_level_scores_are_bit_identical_within_buckets():
    for two_d in (False, True):
        sc, T = toy_scorer(two_d=two_d, ni=70)
        u = sc.uvec[:16]
        for base in ("cos", "raw"):
            S = sc.item_scores(u, "orig", base)
            labels = T["labels"]
            for g in range(T["G"]):
                cols = np.where(labels == g)[0]
                for c in cols[1:]:
                    assert torch.equal(S[:, cols[0]], S[:, c]), (two_d, base, g)
            # and they agree with the item-level (legacy) computation up to rounding
            V, Vn, _ = sc.vecs["items"]
            item_level = u @ V.t() if base == "raw" else torch.nn.functional.normalize(u, dim=-1) @ Vn.t()
            assert torch.allclose(S, item_level, atol=1e-5)
        if two_d:                                                      # swapped alias pair = same group, same vector
            assert labels[0] == labels[1] == labels[2]
            assert torch.equal(T["ivec"][0], T["ivec"][1])
            assert SC.group_labels(T["ia"], canonical=False)[1] >= SC.group_labels(T["ia"], canonical=True)[1]
    # item-level additive terms are added after the expansion: items of one bucket separate under beta*p
    sc, T = toy_scorer(ni=80, n_buckets=5)
    S = sc.item_scores(sc.uvec[:8], "orig", "cos")
    S2 = S + 1.0 * sc.terms["p"][None, :]
    split = 0
    for g in range(T["G"]):
        cols = np.where(T["labels"] == g)[0]
        if len(set(T["p"][cols].tolist())) > 1:
            split += 1
            assert len(torch.unique(S2[0, cols])) > 1 and len(torch.unique(S[0, cols])) == 1
    assert split > 0


def test_mu_eps_identity():
    rng = np.random.RandomState(3)
    for n, g in [(200, 17), (1000, 3), (50, 50)]:
        labels = rng.randint(g, size=n)
        labels[:g] = np.arange(g)
        c = rng.zipf(1.7, size=n)
        p = np.log1p(c)
        mu_g, mu, eps = SC.mu_eps(p, labels, g)
        assert np.allclose(mu + eps, p, atol=1e-12, rtol=0)
        assert np.abs(np.bincount(labels, weights=eps, minlength=g)).max() < 1e-9
        assert np.allclose(mu_g, [p[labels == b].mean() for b in range(g)])
        assert abs(p.var() - (mu.var() + eps.var())) < 1e-12 * max(1.0, p.var())    # law of total variance
        f = rng.randn(g)
        assert abs(np.mean(eps * f[labels])) < 1e-9                                  # eps is orthogonal to any bucket function
    # via the Scorer registrations: mu is constant within a bucket, mu + eps = p
    sc, T = toy_scorer()
    mu, eps = sc.terms["mu"].numpy(), sc.terms["eps"].numpy()
    for b in range(T["G"]):
        assert np.ptp(mu[T["labels"] == b]) == 0
    assert np.allclose(mu + eps, T["p"], atol=1e-6)


def test_shapley_plus_interaction_reconstruct_delta_both_exactly():
    rng = np.random.RandomState(0)
    for _ in range(500):
        base, mu, eps, both = rng.uniform(0, 0.3, 4)
        d = SC.shapley_decomposition(base, mu, eps, both)
        assert abs(d["shapley_mu"] + d["shapley_eps"] - d["delta_both"]) < 1e-15
        assert abs(d["delta_mu"] + d["delta_eps"] + d["interaction"] - d["delta_both"]) < 1e-15
        assert abs(d["shapley_mu"] - (d["delta_mu"] + d["interaction"] / 2)) < 1e-15
        assert abs(d["delta_both"] - (both - base)) < 1e-15
        if d["delta_both"] != 0:
            assert abs(d["frac_shapley_mu"] + d["frac_shapley_eps"] - 1) < 1e-12
            assert abs(d["frac_delta_mu"] + d["frac_delta_eps"] + d["frac_interaction"] - 1) < 1e-12
    z = SC.shapley_decomposition(0.1, 0.1, 0.1, 0.1)
    assert z["delta_both"] == 0 and z["frac_shapley_mu"] is None
    # additive repair -> no interaction, Shapley = main effects
    a = SC.shapley_decomposition(0.10, 0.15, 0.12, 0.17)
    assert abs(a["interaction"]) < 1e-15 and abs(a["shapley_mu"] - 0.05) < 1e-15 and abs(a["shapley_eps"] - 0.02) < 1e-15


def test_metrics_match_repro_core():
    rng = np.random.RandomState(1)
    U, N, k = 80, 60, 20
    topk = np.stack([rng.permutation(N)[:k] for _ in range(U)])
    truths = [rng.choice(N, size=rng.randint(1, 40), replace=False).tolist() for _ in range(U)]
    ref = RC.metrics_from_topk(topk, truths, k)
    hits = np.stack([np.isin(t, tr) for t, tr in zip(topk, truths)])
    rec, ndcg = SC.per_user_metrics(hits, [len(t) for t in truths], k)
    assert abs(rec.mean() - ref["recall"]) < 1e-12 and abs(ndcg.mean() - ref["ndcg"]) < 1e-12


def test_legacy_matches_repro_core_rank_topk_and_lexicographic_when_no_ties():
    sc, T = toy_scorer(ni=50, n_buckets=50)        # one bucket per item (Full-like): scores are generic floats
    for base in ("cos", "raw"):
        pl = SC.Plan()
        leg = pl.add(vec="items", base=base, legacy=True, keep_top=True)
        lex = pl.add(vec="orig", base=base, key=("random", 0), keep_top=True)
        res = sc.run("val", pl, T["va"], T["tr"])
        users = res[leg]["users"]
        old = RC.rank_topk(sc.uvec[torch.as_tensor(users)], T["ivec"], [T["tr"][u] for u in users], 20,
                           cosine=(base == "cos")).numpy()
        assert np.array_equal(res[leg]["top"], old)
        assert np.array_equal(res[lex]["top"], old)
        ref = RC.metrics_from_topk(old, [T["va"][u] for u in users], 20)
        assert abs(res[leg]["recall"] - ref["recall"]) < 1e-12 and abs(res[leg]["ndcg"] - ref["ndcg"]) < 1e-12


def test_masking_val_vs_test_and_test_scored_once():
    sc, T = toy_scorer()
    pl = SC.Plan()
    n = pl.add(vec="orig", base="cos", key=("random", 0), keep_top=True)
    val = sc.run("val", pl, T["va"], T["tr"])[n]["top"]
    test = sc.run("test", pl, T["te"], T["merged"])[n]["top"]
    for r, u in enumerate(T["va"]):
        assert not set(val[r].tolist()) & set(T["tr"][u])
        assert not set(test[r].tolist()) & set(T["merged"][u])
    try:
        sc.run("test", pl, T["te"], T["merged"])
    except RuntimeError as e:
        assert "scored twice" in str(e)
    else:
        raise AssertionError("a test cell must not be scored twice")
    sc.run("val", pl, T["va"], T["tr"])             # validation may be repeated


def test_plan_dedupes_equal_cells_and_beta_zero_is_the_base():
    pl = SC.Plan()
    a = pl.add(vec="orig", base="cos", term="p", beta=0.0, key=("random", 1))
    b = pl.add(vec="orig", base="cos", term="mu", beta=0.0, key=("random", 1))
    c = pl.add(vec="orig", base="cos", key=("random", 1))
    assert a == b == c and len(pl.cells) == 1
    assert SC.cname(vec="orig", base="cos", term="p", beta=0.1) != c
    assert SC.cname(vec="items", base="cos", legacy=True, key=("random", 3)) == SC.cname(vec="items", base="cos", legacy=True)


def test_beta_selection_ties_go_to_smaller_beta():
    assert SC.pick_beta({0.0: .1, .01: .3, .1: .3, 1.0: .25}) == .01
    assert SC.pick_beta({0.0: .2, .1: .2}) == 0.0
    assert SC.pick_beta({b: .5 for b in SC.BETAS}) == 0.0
    assert SC.pick_beta({0.0: .1, .5: .4, 2.0: .3}) == .5


def test_within_bucket_shuffle_keeps_bucket_multisets():
    rng = np.random.RandomState(5)
    labels = rng.randint(7, size=300)
    p = np.log1p(rng.zipf(1.6, size=300))
    a = SC.shuffle_within_buckets(p, labels, 11)
    b = SC.shuffle_within_buckets(p, labels, 12)
    assert np.array_equal(a, SC.shuffle_within_buckets(p, labels, 11))
    assert not np.array_equal(a, p) and not np.array_equal(a, b)
    for g in range(7):
        m = labels == g
        assert np.array_equal(np.sort(a[m]), np.sort(p[m]))
    assert np.allclose(SC.mu_eps(a, labels, 7)[0], SC.mu_eps(p, labels, 7)[0], atol=1e-12)


def test_u8_quantization():
    p = np.log1p(np.random.RandomState(2).zipf(1.5, size=500).astype(float))
    p[0] = 0.0
    ph, info = SC.quantize_u8(p)
    pmax = p.max()
    q = ph * 255 / pmax
    assert np.allclose(q, np.rint(q), atol=1e-9) and q.min() >= 0 and q.max() <= 255 + 1e-9
    assert info["max_abs_error"] <= pmax / 510 + 1e-12 and ph[0] == 0 and abs(ph.max() - pmax) < 1e-12


def test_length_variants():
    rng = np.random.RandomState(4)
    n = 400
    counts = rng.zipf(1.5, size=n).clip(max=500)
    counts[:20] = 0
    a_true, g_true = 0.7, 0.3
    p = np.log1p(counts)
    r = np.exp(a_true + g_true * p)
    r[:20] = rng.uniform(1, 2, 20)                                  # items with c = 0 do not enter the fit
    ivec = rng.randn(n, 6)
    ivec = ivec / np.linalg.norm(ivec, axis=1, keepdims=True) * r[:, None]
    var, fit = SC.length_variants(ivec, counts, [0, 1])
    assert abs(fit["a"] - a_true) < 1e-9 and abs(fit["g"] - g_true) < 1e-9 and fit["r2"] > 1 - 1e-9
    assert fit["n_items_fit"] == n - 20
    assert np.allclose(var["len_replace_fit"], np.exp(a_true + g_true * p))
    for s in (0, 1):
        assert np.allclose(np.sort(var[f"len_shuffle_global_s{s}"]), np.sort(r))
        dec = var[f"len_shuffle_decile_s{s}"]
        order = np.argsort(counts, kind="stable")
        for blk in np.array_split(order, 10):
            assert np.allclose(np.sort(dec[blk]), np.sort(r[blk]))      # a permutation inside each decile only
        assert not np.allclose(dec, r) and not np.allclose(var[f"len_shuffle_global_s{s}"], r)
    assert not np.allclose(var["len_shuffle_decile_s0"], var["len_shuffle_decile_s1"])


def test_group_labels_and_sha():
    ia = np.array([[0, 1], [1, 0], [2, 2], [0, 1], [3, 1]])
    lab, g, rep = SC.group_labels(ia)
    # unordered pairs: (0,1) x3, (2,2), (1,3) -> 3 groups; ordered tuples: (0,1) x2, (1,0), (2,2), (3,1) -> 4
    assert g == 3 and lab[0] == lab[1] == lab[3] and len({lab[2], lab[4], lab[0]}) == 3
    assert SC.group_labels(ia, canonical=False)[1] == 4 and rep[lab[4]] == 4
    lab1, g1, _ = SC.group_labels(np.array([5, 5, 2, 9, 2]))
    assert g1 == 3 and lab1[0] == lab1[1] and lab1[2] == lab1[4]
    a = np.arange(10)
    assert SC.sha256_array(a) == SC.sha256_array(a.astype(np.int32)) and SC.sha256_array(a) != SC.sha256_array(a[::-1])


def test_double_freq_author_functions():
    utils = SC.LOCAL_REPRO / "GraphHash" / "utils.py"
    if not utils.exists():
        return
    rng = np.random.RandomState(0)
    nu, ni, ku, ki = 200, 120, 40, 24
    e = torch.as_tensor(np.stack([rng.randint(nu, size=3000), rng.randint(ni, size=3000)]))
    uc, ic = SC.double_freq_maps(utils, [e], nu, ni, ku, ki)
    uc2, ic2 = SC.double_freq_maps(utils, [e], nu, ni, ku, ki)
    assert uc.shape == (nu, 2) and ic.shape == (ni, 2) and ic.max() < ki and uc.min() >= 0
    assert np.array_equal(uc, uc2) and np.array_equal(ic, ic2)
    assert (ic[:, 0] == ic[:, 1]).sum() >= ki // 2


# ----------------------------------------------------------------------------- end-to-end on a tiny dataset
def write_toy_dataset(root, nu=60, ni=90, seed=0):
    rng = np.random.RandomState(seed)
    tr, va, te = toy_truth(nu, ni, rng, 14, 3, 3)
    tens = lambda d: torch.tensor([[u, i] for u, items in d.items() for i in items], dtype=torch.long).t().contiguous()
    d = Path(root) / "datasets" / "Gowalla-processed"
    d.mkdir(parents=True, exist_ok=True)
    for name, x in [("train", tr), ("val", va), ("test", te)]:
        torch.save(tens(x), d / f"{name}.pt")
    return nu, ni, tr, va, te


def write_ckpt(path, nu_rows, ni_rows, aliases=None, wrapped=True, seed=0, d=8):
    g = torch.Generator().manual_seed(seed)
    sd = {"user_emb_table.weight": torch.randn(nu_rows, d, generator=g),
          "item_emb_table.weight": torch.randn(ni_rows, d, generator=g)}
    if aliases is None and not wrapped:
        torch.save(sd, path)
    else:
        torch.save({"model": sd, "b": None, "aliases": None if aliases is None else [torch.as_tensor(a) for a in aliases]}, path)


def check_controls(doc, seeds, base):
    assert doc["status"] == "complete"
    for key in ["legacy_topk", "tie_keys", "full_prior", "decomposition_fixed_beta", "decomposition_retuned",
                "within_bucket_shuffle", "cos_prior", "raw_prior", f"{base}_prior_u8", "tie_stats", "per_user_files"]:
        assert key in doc, key
    assert doc["tie_keys"]["random"]["test"]["n_seeds"] == seeds
    for metric in ("recall", "ndcg"):
        d = doc["decomposition_fixed_beta"][metric]["from_seed_means"]
        assert abs(d["delta_mu"] + d["delta_eps"] + d["interaction"] - d["delta_both"]) < 1e-12
        assert abs(d["shapley_mu"] + d["shapley_eps"] - d["delta_both"]) < 1e-12
        cells = doc["decomposition_fixed_beta"][metric]["cells"]
        assert abs(cells["both"] - doc["full_prior"]["test"][metric]["mean"]) < 1e-12
        assert abs(cells["base"] - doc["tie_keys"]["random"]["test"][metric]["mean"]) < 1e-12
    assert doc["full_prior"]["best_beta"] in SC.BETAS
    assert len(doc["full_prior"]["val_by_beta"]) == len(SC.BETAS)
    for b in ("cos", "raw"):
        assert doc["legacy_topk"][b]["crosscheck_vs_repro_core_metrics_from_topk"]["max_abs_diff"] < 1e-12
    assert doc["meta"]["identity"]["max_abs_sum_eps_per_bucket"] < 1e-9
    assert doc["meta"]["identity"]["max_abs_mu_shift_within_bucket_shuffle"] < 1e-9


def test_end_to_end_cos_graphhash_with_stored_aliases_and_alias_assert():
    root = TMP / "e2e1"
    nu, ni, tr, va, te = write_toy_dataset(root)
    rng = np.random.RandomState(1)
    ua, ia = rng.randint(8, size=nu), rng.randint(11, size=ni)
    ck = root / "gh.pt"
    write_ckpt(ck, 8, 11, [ua, ia])
    base_args = ["--ckpt", str(ck), "--dataset", "Gowalla", "--method", "GraphHash", "--base", "cos",
                 "--seeds", "3", "--out-root", str(root / "out"), "--data-root", str(root)]
    doc = SC.main(base_args + ["--run-id", "gh_cos", "--assert-aliases", str(ck)])
    check_controls(doc, 3, "cos")
    out = root / "out" / "gh_cos"
    assert (out / "controls.json").exists() and json.loads((out / "controls.json").read_text())["status"] == "complete"
    assert doc["meta"]["mapping_source"] == "stored" and doc["meta"]["mapping_items_sha256"] == SC.sha256_array(ia)
    assert doc["meta"]["alias_assert"]["equal"] is True
    assert doc["meta"]["n_users_test_evaluated"] == nu and doc["meta"]["n_test_cells_scored"] > 10
    for f in doc["per_user_files"]["files"]:
        z = np.load(out / f)
        assert set(z.files) >= {"users", "top20", "hits", "npos", "ndcg"} and z["top20"].shape == (nu, 20)
        assert (z["npos"] == 3).all() and z["hits"].shape == (nu, 20)
    z = np.load(out / "per_user__full_prior_s0.npz")
    assert abs(z["recall"].mean() - doc["full_prior"]["test"]["recall"]["per_seed"][0]) < 1e-12
    z = np.load(out / "per_user__base_randomkey_s0.npz")
    assert abs(z["ndcg"].mean() - doc["tie_keys"]["random"]["test"]["ndcg"]["per_seed"][0]) < 1e-12
    # hits in the npz are real hits
    zt = np.load(out / "per_user__base_popkey.npz")
    for r, u in enumerate(zt["users"]):
        assert zt["hits"][r].sum() == len(set(zt["top20"][r].tolist()) & set(te[int(u)]))
    # tie statistics exist and are consistent
    ts = doc["tie_stats"]["base_beta0"]
    assert 0 <= ts["share_boundary_tie"] <= 1 and ts["mean_tied_group_size_all_users"] >= 1
    # the equality assert aborts on a different mapping
    bad = root / "bad.pt"
    ia2 = ia.copy()
    ia2[0] = (ia2[0] + 1) % 11
    write_ckpt(bad, 8, 11, [ua, ia2])
    try:
        SC.main(base_args + ["--run-id", "gh_bad", "--assert-aliases", str(bad)])
    except SystemExit as e:
        assert e.code == 3
    else:
        raise AssertionError("alias mismatch must abort")
    assert (root / "out" / "gh_bad" / "alias_assert_failed.json").exists()
    assert not (root / "out" / "gh_bad" / "controls.json").exists()


def test_end_to_end_raw_doublefreq_two_d_aliases():
    root = TMP / "e2e2"
    nu, ni, *_ = write_toy_dataset(root, seed=3)
    rng = np.random.RandomState(2)
    ua, ia = rng.randint(9, size=(nu, 2)), rng.randint(12, size=(ni, 2))
    ia[5] = ia[4][::-1]
    ck = root / "df.pt"
    write_ckpt(ck, 9, 12, [ua, ia])
    doc = SC.main(["--ckpt", str(ck), "--dataset", "Gowalla", "--method", "DoubleFreq", "--base", "raw",
                   "--seeds", "2", "--run-id", "df_raw", "--out-root", str(root / "out"), "--data-root", str(root)])
    check_controls(doc, 2, "raw")
    assert doc["meta"]["identity"]["n_groups"] <= doc["meta"]["identity"]["n_groups_ordered_tuples"]
    assert doc["raw_prior"]["best_beta"] == doc["full_prior"]["best_beta"] and "e2e" not in doc


def test_end_to_end_full_with_e2e_and_plain_state_dict():
    root = TMP / "e2e3"
    nu, ni, *_ = write_toy_dataset(root, seed=4)
    ck = root / "full.pt"
    write_ckpt(ck, nu, ni, None, wrapped=False)            # plain state_dict, no aliases -> identity mapping
    doc = SC.main(["--ckpt", str(ck), "--dataset", "Gowalla", "--method", "Full", "--base", "cos", "--e2e",
                   "--seeds", "2", "--run-id", "full_e2e", "--out-root", str(root / "out"), "--data-root", str(root)])
    check_controls(doc, 2, "cos")
    e = doc["e2e"]
    assert doc["meta"]["mapping_source"] == "identity"
    for mode in ("len_shuffle_decile", "len_shuffle_global"):
        assert e[mode]["test"]["n_seeds"] == 2 and e[mode]["val"]["n_seeds"] == 2
    assert {"val", "test"} <= set(e["len_replace_fit"]) and {"val", "test"} <= set(e["unmodified"])
    # unmodified e2e = legacy raw-dot at beta 0 (same ranking, no ties)
    assert abs(e["unmodified"]["test"]["recall"] - doc["legacy_topk"]["raw"]["beta0"]["test"]["recall"]) < 1e-12
    # Full mapping: every item is its own bucket, so mu = p and eps = 0
    assert doc["meta"]["identity"]["var_eps"] == 0.0
    # e2e needs the Full mapping
    ck2 = root / "gh.pt"
    write_ckpt(ck2, 8, 11, [np.arange(nu) % 8, np.arange(ni) % 11])
    try:
        SC.main(["--ckpt", str(ck2), "--dataset", "Gowalla", "--method", "GraphHash", "--base", "cos", "--e2e",
                 "--seeds", "2", "--run-id", "bad_e2e", "--out-root", str(root / "out"), "--data-root", str(root)])
    except SystemExit:
        pass
    else:
        raise AssertionError("--e2e must reject a non-identity mapping")


def test_mapping_resolution_rules():
    root = TMP / "map"
    nu, ni, *_ = write_toy_dataset(root, seed=5)
    ck = root / "r.pt"
    write_ckpt(ck, 7, 10, None, wrapped=False)
    args = ["--ckpt", str(ck), "--dataset", "Gowalla", "--base", "cos", "--seeds", "1", "--out-root", str(root / "out"),
            "--data-root", str(root)]
    doc = SC.main(args + ["--method", "Random", "--run-id", "rnd"])         # modulo with the table sizes, no clustering
    assert doc["meta"]["mapping_source"] == "modulo(table rows)"
    assert doc["meta"]["mapping_items_sha256"] == SC.sha256_array(np.arange(ni) % 10)
    assert doc["meta"]["mapping_users_sha256"] == SC.sha256_array(np.arange(nu) % 7)
    if not torch.cuda.is_available():                                        # rule 4: no hash recompute on the CPU image
        for m in ("GraphHash", "DoubleFreq"):
            try:
                SC.main(args + ["--method", m, "--run-id", "no_" + m])
            except SystemExit as e:
                assert "rule 4" in str(e)
            else:
                raise AssertionError("must refuse to recompute on CPU")
    assert SC.redact("https://x.y/z?sig=secret") == "<url>" and SC.redact("/a/b") == "/a/b"
    assert SC.source_label("https://x.y/z?sig=secret") == "<url>" and SC.source_label("/kaggle/input/acct/k/best.pt") == "best.pt"
    assert "secret" not in SC.scrub("HTTPError 403 for url: https://x.y/z?sig=secret&a=b, retry")


# ----------------------------------------------------------------------------- T4 tests
def test_eta_squared_identity_and_hand_example():
    n = 25
    x = np.random.RandomState(0).randn(n)
    assert abs(MM.eta_squared(x, np.arange(n)) - 1.0) < 1e-12                  # identity map keeps everything
    assert abs(MM.eta_squared(x, np.zeros(n, dtype=int))) < 1e-12              # one bucket keeps nothing
    # hand computation: x = [1,2,3,4,10,12], buckets {0,1,2} and {3,4,5}
    x = np.array([1., 2, 3, 4, 10, 12])
    lab = np.array([0, 0, 0, 1, 1, 1])
    # bucket means 2 and 26/3, grand mean 16/3:
    #   Var(E[X|B]) = (3*(10/3)^2 + 3*(10/3)^2) / 6 = 100/9
    #   Var(X)      = (1+4+9+16+100+144)/6 - (16/3)^2 = 274/6 - 256/9 = 310/18
    #   eta^2       = (100/9) / (310/18) = 20/31
    mean = x.mean()
    between = (3 * (2 - mean) ** 2 + 3 * (26 / 3 - mean) ** 2) / 6
    total = ((x - mean) ** 2).sum() / 6
    assert abs(between - 100 / 9) < 1e-12 and abs(total - 310 / 18) < 1e-12
    assert abs(MM.eta_squared(x, lab) - 20 / 31) < 1e-12
    assert MM.eta_squared(np.ones(6), lab) is None
    # Var(mu)/Var(p) = eta^2 (PLAN 4.6)
    _, mu, eps = SC.mu_eps(x, lab, 2)
    assert abs(mu.var() / x.var() - 20 / 31) < 1e-12


def test_traffic_proxy_and_rho_proxy():
    c = np.array([5, 1, 3, 2])
    assert MM.traffic_proxy(c, np.array([0, 0, 1, 1])).tolist() == [6, 6, 5, 5]
    assert MM.traffic_proxy(c, np.arange(4)).tolist() == c.tolist()                  # identity: proxy = c
    assert abs(MM.spearman(c, MM.traffic_proxy(c, np.arange(4))) - 1.0) < 1e-12
    # dfreq: N_a over the distinct items having alias a in either slot; proxy = N_a1 + N_a2
    ia = np.array([[0, 1], [0, 1], [2, 2]])
    assert MM.traffic_proxy(np.array([1, 2, 4]), ia).tolist() == [6, 6, 8]
    s = MM.summarize_map(np.array([0, 0, 1, 1]), np.arange(4), c)
    assert abs(s["eta2_p"] - 1) < 1e-12 and abs(s["eta2_rank_c"] - 1) < 1e-12 and abs(s["rho_proxy"] - 1) < 1e-12
    assert s["items_per_group"]["share_in_groups_ge2"] == 0.0 and s["users_per_group"]["share_in_groups_ge2"] == 1.0


def test_summarize_map_toy():
    c = np.array([9, 7, 8, 1, 2, 1, 4])
    ia = np.array([0, 0, 0, 1, 1, 2, 3])
    s = MM.summarize_map(np.array([0, 1, 1, 0]), ia, c)
    ipg = s["items_per_group"]
    assert ipg["n_groups"] == 4 and ipg["max"] == 3 and ipg["mean"] == 7 / 4 and ipg["median"] == 1.5
    assert abs(ipg["share_in_groups_ge2"] - 5 / 7) < 1e-12 and ipg["n_singleton_groups"] == 2
    labels = ia
    p = np.log1p(c.astype(float))
    mu = np.array([p[labels == b].mean() for b in range(4)])[labels]
    manual = ((mu - p.mean()) ** 2).mean() / ((p - p.mean()) ** 2).mean()
    assert abs(s["eta2_p"] - manual) < 1e-12 and abs(s["var_mu_over_var_p"] - manual) < 1e-12
    assert abs(s["var_p"] - (s["var_mu"] + s["var_eps"])) < 1e-12
    assert s["theorem2_check_rho_proxy_sq_le_eta2_rank_c"] is True
    assert s["items_sha256"] == SC.sha256_array(ia) and s["item_table_rows"] == 4
    # random pairings never violate the capacity bound
    rng = np.random.RandomState(0)
    for _ in range(30):
        n = 300
        cc = rng.zipf(1.5, size=n)
        al = rng.randint(rng.randint(2, 60), size=n)
        assert MM.summarize_map(np.arange(10) % 3, al, cc)["theorem2_check_rho_proxy_sq_le_eta2_rank_c"]
        al2 = rng.randint(30, size=(n, 2))
        assert MM.summarize_map(np.arange(10) % 3, al2, cc)["theorem2_check_rho_proxy_sq_le_eta2_rank_c"]


def test_make_maps_end_to_end_with_fake_repro():
    utils = SC.LOCAL_REPRO / "GraphHash" / "utils.py"
    if not utils.exists():
        return
    rng = np.random.RandomState(0)
    nu, ni, E = 120, 90, 2400
    edges = torch.as_tensor(np.stack([rng.randint(nu, size=E), np.minimum(rng.zipf(1.4, size=E) - 1, ni - 1)]))
    edges[0, :nu] = torch.arange(nu)
    edges[1, :ni] = torch.arange(ni)
    parts = [edges, edges[:, :300], edges[:, 300:600]]

    def dicts(part):
        d = {}
        for u, i in part.t().tolist():
            d.setdefault(u, []).append(i)
        return d
    ku, ki = 12, 14

    class FakeR:
        WORK = SC.LOCAL_REPRO
        SEED = 0

        @staticmethod
        def load_data(dataset, ns, resolution=200):
            maps = {"Full": (np.arange(nu), np.arange(ni)),
                    "GraphHash": (np.arange(nu) * 7 % ku, np.arange(ni) * 5 % ki),
                    "Random": (np.arange(nu) % ku, np.arange(ni) % ki)}
            info = {"user_id_max": nu - 1, "item_id_max": ni - 1, "users": nu, "items": ni,
                    "interactions": E, "split_counts": [E, 300, 300], "user_buckets": ku, "item_buckets": ki}
            return parts, maps, tuple(dicts(p) for p in parts), info

    ns = {"graphClustering": lambda g, resolution: (np.arange(nu) % (resolution // 100 + 2), np.arange(ni) % (resolution // 100 + 3),
                                                    resolution // 100 + 2, resolution // 100 + 3),
          "map_to_consecutive_integers": lambda v: np.asarray(v)}
    old = SC.prepare_repro
    SC.prepare_repro = lambda: (FakeR, (None, None, None, None, ns))
    try:
        root = TMP / "maps"
        good, bad = root / "good.pt", root / "bad.pt"
        root.mkdir(parents=True, exist_ok=True)
        write_ckpt(good, ku, ki, [np.arange(nu) * 7 % ku, np.arange(ni) * 5 % ki])
        write_ckpt(bad, ku, ki, [np.arange(nu) * 3 % ku, np.arange(ni) * 3 % ki])
        write_ckpt(root / "plain.pt", ku, ki, None, wrapped=False)
        rc = MM.main(["--datasets", "Gowalla", "--out-root", str(root / "out"), "--gowalla-resolutions", "400,800",
                      "--assert-aliases", f"Gowalla__GraphHash={good}", f"gowalla__random={bad}",
                      f"gowalla__dfreq={root / 'plain.pt'}", f"gowalla__nope={good}"])
    finally:
        SC.prepare_repro = old
    assert rc == 0
    out = root / "out"
    for stem in ["gowalla__graphhash", "gowalla__random", "gowalla__dfreq", "gowalla__graphhash-r400", "gowalla__graphhash-r800"]:
        assert (out / f"{stem}__users.npy").exists() and (out / f"{stem}__items.npy").exists(), stem
    assert np.load(out / "gowalla__dfreq__items.npy").shape == (ni, 2)
    assert np.array_equal(np.load(out / "gowalla__graphhash-r400__items.npy"), np.arange(ni) % 7)
    s = json.loads((out / "maps_summary.json").read_text())
    assert set(s["maps"]) == {"gowalla__full", "gowalla__graphhash", "gowalla__random", "gowalla__dfreq",
                              "gowalla__graphhash-r400", "gowalla__graphhash-r800"}
    assert abs(s["maps"]["gowalla__full"]["eta2_p"] - 1) < 1e-12 and s["maps"]["gowalla__full"]["saved"] is False
    d = s["datasets"]["gowalla"]
    assert d["split_counts"] == [E, 300, 300] and d["eval_users_val"] == len(dicts(parts[1])) and d["eval_users_test"] == len(dicts(parts[2]))
    assert d["graphhash_r200_buckets"] == [ku, ki] and d["graphhash_r200_buckets_match_expected"] is False
    for m in s["maps"].values():
        assert 0 <= m["eta2_p"] <= 1 and m["theorem2_check_rho_proxy_sq_le_eta2_rank_c"] is True
    assert s["maps"]["gowalla__graphhash"]["items_sha256"] == SC.sha256_array(np.arange(ni) * 5 % ki)
    assert s["maps"]["gowalla__dfreq"]["multi_alias"] is True
    a = s["assert_aliases"]
    assert a["gowalla__graphhash"]["equal"] is True
    assert a["gowalla__random"]["equal"] is False and a["gowalla__random"]["users_equal"] is False and a["gowalla__random"]["items_equal"] is False
    assert a["gowalla__dfreq"]["equal"] is None and "no stored aliases" in a["gowalla__dfreq"]["error"]
    assert a["gowalla__nope"]["equal"] is None and "not built" in a["gowalla__nope"]["error"]


def test_parse_assert_aliases():
    assert MM.parse_assert(["A__b=/x/y", "C__d=https://h/p?q=1=2"]) == {"a__b": "/x/y", "c__d": "https://h/p?q=1=2"}
    try:
        MM.parse_assert(["nopath"])
    except SystemExit:
        pass
    else:
        raise AssertionError


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception:
                import traceback
                fails += 1
                print("FAIL", name)
                traceback.print_exc()
    print("FAILED" if fails else "ALL PASSED", fails)
    sys.exit(1 if fails else 0)
