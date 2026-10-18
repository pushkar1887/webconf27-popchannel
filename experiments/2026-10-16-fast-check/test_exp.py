"""Local CPU tests for exp_run.py with tiny synthetic data (no clustering, no Kaggle)."""
import importlib
import shutil
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "2026-10-16-loss-sharing-repro" / "_work"
# repro_run creates ROOT/_work at import, so import it from a throw-away copy.
tmp = Path(tempfile.mkdtemp())
for name in ["repro_core.py", "repro_run.py"]:
    shutil.copy(SRC/name, tmp/name)
shutil.copy(HERE/"exp_run.py", tmp/"exp_run.py")
sys.path.insert(0, str(tmp))
sys.path.insert(1, str(SRC/"GraphHash"))
E = importlib.import_module("exp_run")
from models.mf import MF
from models.mfdau import MFDAU
from loss import BPRLoss, DAULoss

torch.manual_seed(0)
np.random.seed(0)


def test_two_d_alias_sum():
    t = nn.Embedding(5, 3)
    alias = torch.tensor([[0, 1], [2, 2], [4, 3]])
    v = E.table_vec(t, alias)
    exp = torch.stack([t.weight[0]+t.weight[1], 2*t.weight[2], t.weight[4]+t.weight[3]])
    assert torch.allclose(v, exp)
    assert E.table_vec(t, torch.tensor([1, 4])).shape == (2, 3)
    # real models: MFDAU forward and all_vectors agree on 2-D aliases
    m = MFDAU(6, 5, 4)
    ua, ia = torch.tensor([[0, 1], [2, 3]]), torch.tensor([[0, 4], [1, 1]])
    u, i = m(ua, ia)
    uv, iv = E.all_vectors(m, (ua, ia))
    assert torch.allclose(u, uv) and torch.allclose(i, iv)
    mf = MF(6, 5, 4)
    ps, ns = mf(ua, ia, ia.flip(0))
    mu, mi = E.all_vectors(mf, (ua, ia))
    assert torch.allclose(ps, (mu*mi).sum(-1))


def test_scoring_and_masking():
    ivec = torch.tensor([[5., 0], [4, 0], [3, 0], [2, 0], [1, 0], [0, 1]])
    uvec = torch.tensor([[1., 0], [1, 0]])
    truth = {0: [2], 1: [4, 5]}
    masks = {0: [0, 1], 1: []}      # user 0: items 0,1 masked
    r = E.sweep(uvec, ivec, truth, masks, [0.0], k=2)[0.0]
    # user0 ranking (masked 0,1): 2,3 -> hit at rank 0: recall 1, ndcg 1 ; user1 ranking 0,1 -> 0
    assert abs(r["recall"] - 0.5) < 1e-9 and abs(r["ndcg"] - 0.5) < 1e-9, r
    # without the mask user 0 would pick items 0,1 first and miss
    r2 = E.sweep(uvec, ivec, {0: [2]}, {}, [0.0], k=2)[0.0]
    assert r2["recall"] == 0.0
    # logpop prior flips the ranking for user 1: large beta on items 4,5
    logpop = torch.tensor([0., 0, 0, 0, 1, 1])
    out = E.sweep(uvec, ivec, {1: [4, 5]}, {}, [0.0, 10.0], logpop, k=2)
    assert out[0.0]["recall"] == 0.0 and out[10.0]["recall"] == 1.0, out
    # cosine ignores norms: item 5 (orthogonal to user) is ranked last among 6
    out = E.sweep(uvec, ivec, {0: [5]}, {}, [0.0], cosine=True, k=5)[0.0]
    assert out["recall"] == 0.0
    out = E.sweep(uvec, ivec, {0: [4]}, {}, [0.0], cosine=True, k=5)[0.0]   # items 0..4 tie at cos 1 -> in top5
    assert out["recall"] == 1.0
    # fixed_add (bias) is an item-additive term
    out = E.sweep(uvec, ivec, {1: [5]}, {}, [0.0], None, fixed_add=torch.tensor([0., 0, 0, 0, 0, 100.]), k=1)
    assert out[0.0]["recall"] == 1.0
    # chunking does not change results
    big_truth = {u: [u % 6] for u in range(7)}
    big_u = torch.randn(7, 2)
    a = E.sweep(big_u, ivec, big_truth, {}, [0.0], chunk=3, k=3)[0.0]
    b = E.sweep(big_u, ivec, big_truth, {}, [0.0], chunk=512, k=3)[0.0]
    assert a == b


def test_tuning_val_then_test_once():
    calls = []
    val_recall = {0.0: .1, .01: .3, .1: .2, 1.0: .25}

    def fake(split, bs):
        calls.append((split, tuple(bs)))
        if split == "val":
            return {b: {"recall": val_recall[b], "ndcg": 0.} for b in bs}
        return {b: {"recall": 0.99, "ndcg": 0.} for b in bs}
    r = E.tune_and_test(fake, list(val_recall))
    assert r["best_beta"] == .01 and r["test"]["recall"] == 0.99 and r["val"]["recall"] == .3
    assert calls == [("val", tuple(val_recall)), ("test", (.01,))], calls

    def fake2(split, bs):   # test prefers the opposite beta; selection must still follow validation
        return {b: {"recall": (val_recall[b] if split == "val" else 1.0 - val_recall[b]), "ndcg": 0.} for b in bs}
    r = E.tune_and_test(fake2, list(val_recall))
    assert r["best_beta"] == .01 and abs(r["test"]["recall"] - .7) < 1e-9
    r = E.tune_and_test(lambda s, bs: {b: {"recall": .5, "ndcg": 0.} for b in bs}, [0.0, .1])
    assert r["best_beta"] == 0.0   # tie -> smallest beta


def test_eval_suite_keys():
    gen = torch.Generator().manual_seed(1)
    nu, ni = 20, 30
    uvec, ivec = torch.randn(nu, 4, generator=gen), torch.randn(ni, 4, generator=gen)
    mk = lambda off: {u: [(u*3+off+j) % ni for j in range(3)] for u in range(nu)}
    tr, va, te = mk(0), mk(10), mk(20)
    merged = {u: tr[u]+va[u] for u in tr}
    out = E.eval_suite(uvec, ivec, va, te, tr, merged, torch.rand(ni), b=torch.randn(ni))
    assert set(out) == {"raw", "cosine", "cos_prior", "raw_prior", "cos_b", "cos_b_prior"}
    assert set(out["raw_prior"]["val_by_beta"]) == set(E.BETAS)
    assert out["raw_prior"]["best_beta"] in E.BETAS
    assert "recall" in out["cos_prior"]["test"] and "ndcg" in out["cos_prior"]["val"]
    assert set(E.eval_suite(uvec, ivec, va, te, tr, merged, torch.rand(ni))) == {"raw", "cosine", "cos_prior", "raw_prior"}


def test_spearman():
    counts = np.array([1, 5, 2, 9, 3])
    ivec = torch.tensor(counts, dtype=torch.float).unsqueeze(1).repeat(1, 3)   # norm monotone in count
    assert abs(E.spearman_norm_vs_count(ivec, counts) - 1.0) < 1e-9
    assert abs(E.spearman_norm_vs_count(ivec, -counts) + 1.0) < 1e-9
    assert E.spearman_norm_vs_count(ivec, np.ones(5)) is None   # constant -> nan -> None


def test_dau_bias_backward():
    for two_d in (False, True):
        nu, ni, nbu, nbi = 12, 15, 5, 6
        m = MFDAU(nbu, nbi, 8)
        u_map = torch.randint(nbu, (nu, 2) if two_d else (nu,))
        i_map = torch.randint(nbi, (ni, 2) if two_d else (ni,))
        b = nn.Parameter(torch.zeros(ni))
        opt = torch.optim.Adam(list(m.parameters())+[b], lr=.01, weight_decay=1e-8)
        uid, iid = torch.randint(nu, (16,)), torch.randint(ni, (16,))
        neg = torch.randint(ni, (16,))
        loss = E.dau_bias_loss(m, b, DAULoss(1.), BPRLoss(), u_map[uid], i_map[iid], iid, neg, i_map)
        assert torch.isfinite(loss)
        loss.backward()
        assert b.grad is not None and b.grad.abs().sum() > 0
        touched = set(iid.tolist()) | set(neg.tolist())
        assert all(b.grad[j] == 0 for j in range(ni) if j not in touched)   # per ORIGINAL item
        assert m.user_emb_table.weight.grad.abs().sum() > 0 and m.item_emb_table.weight.grad.abs().sum() > 0
        before = b.detach().clone()
        opt.step()
        assert not torch.equal(before, b.detach())
        emb = sum(p.numel() for p in m.parameters())
        assert b.numel() == ni and 4*b.numel() == 60 and emb == (nbu+nbi)*8


def test_author_double_freq_extraction():
    nu, ni, ku, ki = 200, 120, 40, 24
    e = torch.stack([torch.randint(nu, (3000,)), torch.randint(ni, (3000,))])
    uc, ic = E.double_freq_maps(SRC/"GraphHash"/"utils.py", [e], nu, ni, ku, ki)
    assert uc.shape == (nu, 2) and ic.shape == (ni, 2)
    assert uc.max() < ku and ic.max() < ki and uc.min() >= 0
    uc2, ic2 = E.double_freq_maps(SRC/"GraphHash"/"utils.py", [e], nu, ni, ku, ki)
    assert (uc == uc2).all() and (ic == ic2).all()          # seeded -> deterministic
    assert (ic[:, 0] == ic[:, 1]).sum() >= ki//2            # frequency-fixed items: both codes equal
    aliases = (torch.tensor(uc), torch.tensor(ic))
    m = MFDAU(ku, ki, 8)
    u, i = m(aliases[0][e[0, :32]], aliases[1][e[1, :32]])
    DAULoss(1.)(u, i).backward()
    assert m.item_emb_table.weight.grad.abs().sum() > 0


def test_run_seed_smoke():
    """End-to-end training loop + eval suite on a tiny synthetic dataset, every loss, 1-D and 2-D aliases."""
    import json
    from types import SimpleNamespace
    nu, ni, E_n = 30, 40, 400
    edges = torch.stack([torch.randint(nu, (E_n,)), torch.randint(ni, (E_n,))])
    tr, va, te = {}, {}, {}
    for d, (a, b) in zip((tr, va, te), ((0, 300), (300, 350), (350, 400))):
        for u, i in edges[:, a:b].t().tolist():
            d.setdefault(u, []).append(i)
    pop = torch.bincount(edges[1, :300], minlength=ni).float()
    for two_d in (False, True):
        for loss in ("dau", "bpr_author", "dau_bias"):
            ua = torch.randint(8, (nu, 2) if two_d else (nu,))
            ia = torch.randint(10, (ni, 2) if two_d else (ni,))
            c = dict(MF=MF, MFDAU=MFDAU, BPRLoss=BPRLoss, DAULoss=DAULoss, parts=[edges[:, :300]],
                     train_truth=tr, val_truth=va, test_truth=te,
                     merged={u: tr.get(u, [])+va.get(u, []) for u in set(tr)|set(va)},
                     device=torch.device("cpu"), aliases=(ua, ia), nu=nu, ni=ni, counts=pop.numpy(),
                     logpop=torch.log1p(pop))
            tag = f"smoke_{loss}_{two_d}"
            args = SimpleNamespace(loss=loss, tag=tag, epochs=4, eval_every=2, patience=None,
                                   cap_min=75., lr=.01, wd=1e-8, batch=64 if loss != "bpr_author" else None,
                                   run_id=tag, method="GraphHash", min_delta=0.0, init_match=False,
                                   equal_item_norm=False)
            result = {"seeds": {}}
            out = Path(tmp)/f"{args.tag}.json"
            E.run_seed(args, 0, c, result, out)
            row = json.loads(out.read_text())["seeds"]["0"]
            assert row["epochs_run"] == 4 and set(row["val_curve"]) == {"2", "4"}
            assert ("memory_overhead" in row) == (loss == "dau_bias")
            assert ("cos_b" in row["eval"]) == (loss == "dau_bias")
            if loss == "dau_bias":
                assert row["memory_overhead"]["b_params"] == ni
            ck = torch.load(E.R.ROOT/"runs"/args.run_id/"best_seed0.pt", weights_only=True)   # run dir (T1 d)
            assert (ck["b"] is not None) == (loss == "dau_bias")


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception:
                import traceback
                traceback.print_exc()
                print("FAIL", name)
                fails += 1
    shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASSED" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
