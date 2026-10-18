"""CPU unit tests for score_controls.py --post-s3 (synthetic data, no Kaggle)."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_controls as SC                                           # noqa: E402
from test_t3t4 import brute_force_topk, toy_scorer, write_ckpt, write_toy_dataset   # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="rescore_"))


def toy_data(T):
    return {"val_truth": T["va"], "train_truth": T["tr"], "test_truth": T["te"], "merged": T["merged"],
            "counts": T["counts"], "ni": T["ni"]}


def add_priors(sc, T):
    p32 = np.log1p(T["counts"].astype(np.float64)).astype(np.float32)
    q, scale, dec, info = SC.quantize_u8_f32(p32)
    sc.add_term("p", p32)
    sc.add_term("pu8", dec)
    return p32, q, scale, dec, info


def test_uint8_decode_error_bound_and_payload():
    rng = np.random.RandomState(0)
    p = np.log1p(rng.zipf(1.5, size=2000)).astype(np.float32)
    p[0] = 0.0
    q, scale, dec, info = SC.quantize_u8_f32(p)
    pmax = float(p.max())
    assert q.dtype == np.uint8 and dec.dtype == np.float32 and scale.dtype == np.float32
    assert abs(float(scale) - pmax / 255) < 1e-7 and q.max() == 255 and q[0] == 0
    assert np.array_equal(dec, q.astype(np.float32) * scale)
    assert np.abs(dec.astype(np.float64) - p).max() <= pmax / 510 + 1e-6      # half a quantization step
    assert info["payload_bytes"] == len(q) + 4 and info["max_abs_error"] <= pmax / 510 + 1e-6


def test_weighted_recall_equals_standard_recall_for_equal_weights():
    sc, T = toy_scorer()
    pl = SC.Plan()
    n = pl.add(vec="orig", base="raw", key=("random", 0), keep_top=True)
    r = sc.run("val", pl, T["va"], T["tr"])[n]
    wr, num, den = SC.weighted_recall_u(r["hits"], r["top"], r["users"], T["va"], np.ones(T["ni"]))
    standard = r["hits"].sum(1) / r["npos"]                                   # uncapped (|T|=3 < 20 here too)
    assert np.allclose(wr, standard, atol=1e-15)
    assert np.allclose(wr, r["recall_u"])                                     # npos<=20 so capped == uncapped
    # with |T| > 20 the weighted recall is the UNCAPPED hits/|T|, not the capped recall
    hits = np.zeros((1, 20), dtype=bool)
    hits[0, :5] = True
    top = np.arange(20)[None, :]
    w2, _, _ = SC.weighted_recall_u(hits, top, [0], {0: list(range(40))}, np.ones(40))
    assert abs(w2[0] - 5 / 40) < 1e-15
    # nonuniform weights: recovered weight / all truth weight
    w = np.array([1.0, 2.0, 3.0, 4.0])
    h = np.array([[True, False, True, False]])
    wr, num, den = SC.weighted_recall_u(h, np.array([[0, 1, 2, 3]]), [0], {0: [0, 2, 3]}, w)
    assert abs(wr[0] - (1 + 3) / (1 + 3 + 4)) < 1e-15 and den[0] == 8
    cw = SC.train_count_weights(np.array([0, 3, 8]))
    assert np.allclose(cw, [1.0, 0.5, 1 / 3])


def test_val_stage_never_reads_test_and_selects_smaller_beta_on_ties():
    sc, T = toy_scorer()
    add_priors(sc, T)
    d = {"val_truth": T["va"], "train_truth": T["tr"], "counts": T["counts"]}      # no test_truth / merged at all
    w = SC.train_count_weights(T["counts"])
    vs = SC.post_s3_val_stage(sc, d, w, 0.2)
    assert vs["selected_beta"] in SC.BETAS and set(vs["weighted_val_by_beta"]) == set(SC.BETAS)
    assert not sc.tested                                                          # nothing scored on test
    assert SC.select_beta_weighted({0.0: .1, .01: .3, .1: .3, 1.0: .25}) == .01
    assert SC.select_beta_weighted({b: .5 for b in SC.BETAS}) == 0.0
    # selection is a function of validation numbers only, recomputable from the stored curve
    assert vs["selected_beta"] == SC.pick_beta(vs["weighted_val_by_beta"])
    # test stage with the selected beta: fixed==selected -> one cell
    bsel = vs["selected_beta"]
    res, roles, pl = SC.post_s3_test_stage(sc, toy_data(T), w, 0.2, 0.2, [0, 1])
    assert roles["e9_selected"] == roles["e9_fixed"] == roles["e7_f32"]
    assert sc.tested == set(pl.cells) and bsel in SC.BETAS


def test_each_tie_policy_is_valid_and_uses_the_complete_prior():
    sc, T = toy_scorer(ni=60, n_buckets=7)
    p32, *_ = add_priors(sc, T)
    w = SC.train_count_weights(T["counts"])
    beta, seeds = 0.2, [0, 1, 2]
    res, roles, pl = SC.post_s3_test_stage(sc, toy_data(T), w, beta, 0.5, seeds)
    keys = {"e8_index": ("index",), "e8_pop": ("pop",), **{f"e8_random_s{s}": ("random", s) for s in seeds}}
    for role, key in keys.items():
        c = pl.cells[roles[role]]
        assert c.term == "p" and c.beta == beta and c.base == "raw" and not c.legacy       # complete prior, fixed beta
        perm = sc.key(key)[0].numpy()
        assert np.array_equal(np.sort(perm), np.arange(T["ni"]))                         # valid permutation
        top = res[roles[role]]["top"]
        assert top.shape == (len(T["te"]), 20)
        for r, u in enumerate(res[roles[role]]["users"]):
            row = top[r].tolist()
            assert len(set(row)) == 20 and 0 <= min(row) and max(row) < T["ni"]
            assert not set(row) & set(T["merged"][int(u)])                                # zero seen-item overlap
        ref, _ = brute_force_topk(sc, T, "test", "raw", "p", beta, key)                  # independent lexicographic sort
        assert np.array_equal(top, ref), role
    # the prior really enters: beta=0.2 differs from the beta=0 base for at least one user
    base = sc.run("val", _plan(vec="orig", base="raw", key=("index",)), T["va"], T["tr"])
    full = sc.run("val", _plan(vec="orig", base="raw", term="p", beta=beta, key=("index",)), T["va"], T["tr"])
    assert not np.array_equal(next(iter(base.values()))["top"], next(iter(full.values()))["top"])
    # legacy topk sensitivity is present, separately labelled, with the prior
    leg = pl.cells[roles["e8_legacy_topk"]]
    assert leg.legacy and leg.term == "p" and leg.beta == beta and leg.key is None
    assert pl.cells[roles["e5_cos_s0"]].base == "cos" and pl.cells[roles["e5_cos_s0"]].term is None


def _plan(**kw):
    pl = SC.Plan()
    pl.add(keep_top=True, **kw)
    return pl


def test_u8_and_float_cells_share_everything_but_the_prior():
    sc, T = toy_scorer()
    add_priors(sc, T)
    w = SC.train_count_weights(T["counts"])
    res, roles, pl = SC.post_s3_test_stage(sc, toy_data(T), w, 0.2, 0.2, [0])
    a, b = pl.cells[roles["e7_f32"]], pl.cells[roles["e7_u8"]]
    assert (a.vec, a.base, a.beta, a.key) == (b.vec, b.base, b.beta, b.key) and (a.term, b.term) == ("p", "pu8")
    assert np.array_equal(res[roles["e7_f32"]]["users"], res[roles["e7_u8"]]["users"])


def test_end_to_end_post_s3_files_and_receipt():
    root = TMP / "e2e"
    nu, ni, tr, va, te = write_toy_dataset(root)
    rng = np.random.RandomState(1)
    ua, ia = rng.randint(8, size=nu), rng.randint(11, size=ni)
    ck = root / "gh.pt"
    write_ckpt(ck, 8, 11, [ua, ia])
    args = ["--ckpt", str(ck), "--dataset", "Gowalla", "--method", "GraphHash", "--base", "raw", "--post-s3",
            "--beta", "0.2", "--seeds", "3", "--run-id", "ps3", "--out-root", str(root / "out"),
            "--data-root", str(root), "--assert-aliases", str(ck)]
    doc = SC.main(args)
    out = root / "out" / "ps3"
    d = json.loads((out / "controls.json").read_text())
    assert d["status"] == "complete" and d["schema"] == "score_controls/post_s3/1"
    m = d["meta"]
    assert len(m["ckpt_sha256"]) == 64 and len(m["mapping_items_sha256"]) == 64 and m["alias_assert"]["equal"] is True
    assert m["n_users_test_evaluated"] == nu and m["timing_seconds"] >= 0 and "wall_seconds" in m
    assert d["policy_list"] and d["post_s3"]["e9"]["selected_beta"] in SC.BETAS
    expected = ["e5_cos_randomkey_s0_test", "e5_cos_torchtopk_test", "e7_f32_val", "e7_u8_val", "e7_f32_test",
                "e7_u8_test", "e9_test_beta_fixed", "e9_test_beta_selected", "e8_test_index", "e8_test_pop",
                "e8_test_random_s0", "e8_test_random_s1", "e8_test_random_s2", "e8_test_torchtopk"]
    assert set(d["cells"]) == set(expected)
    for tag in expected:
        c = d["cells"][tag]
        z = np.load(out / c["file"])
        assert set(z.files) >= {"users", "top20", "hits", "npos", "ndcg", "recall", "wrecall"}
        assert z["top20"].shape == (nu, 20) and c["seen_item_overlap"] == 0 and c["n_users"] == nu
        truth = te if c["split"] == "test" else va
        for r, u in enumerate(z["users"]):
            assert z["hits"][r].sum() == len(set(z["top20"][r].tolist()) & set(truth[int(u)]))
        if tag.startswith("e8_") or tag.startswith("e7_"):
            assert bool(z["prior_complete"]) and abs(float(z["beta"]) - 0.2) < 1e-12
    assert (out / "prior_u8_payload.npz").exists()
    pay = np.load(out / "prior_u8_payload.npz")
    assert pay["q"].dtype == np.uint8 and pay["scale"].dtype == np.float32
    e7 = d["post_s3"]["e7"]
    for split in ("val", "test"):
        assert abs(e7[split]["recall_change"]["signed_pp"]) == e7[split]["recall_change"]["abs_pp"] or \
            e7[split]["recall_change"]["signed_pp"] < 0
    assert d["post_s3"]["e9"]["test_selected"]["same_cell_as_fixed"] == (d["post_s3"]["e9"]["selected_beta"] == 0.2)
    # a non-raw base is refused
    try:
        SC.main([a if a != "raw" else "cos" for a in args])
    except SystemExit:
        pass
    else:
        raise AssertionError("--post-s3 must require --base raw")
