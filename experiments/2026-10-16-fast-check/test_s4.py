"""Local CPU tests for the S4 features of exp_run.py: --model LightGCN, --loss dau_resbias, --method DoubleGraph."""
import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "2026-10-16-loss-sharing-repro" / "_work"
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

NU, NI = 30, 40


def make_ctx(two_d=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    edges = torch.stack([torch.randint(NU, (400,), generator=g), torch.randint(NI, (400,), generator=g)])
    tr, va, te = {}, {}, {}
    for d, (a, b) in zip((tr, va, te), ((0, 300), (300, 350), (350, 400))):
        for u, i in edges[:, a:b].t().tolist():
            d.setdefault(u, []).append(i)
    pop = torch.bincount(edges[1, :300], minlength=NI).float()
    ua = torch.randint(8, (NU, 2) if two_d else (NU,), generator=g)
    ia = torch.randint(10, (NI, 2) if two_d else (NI,), generator=g)
    return dict(MF=MF, MFDAU=MFDAU, BPRLoss=BPRLoss, DAULoss=DAULoss, parts=[edges[:, :300]],
                train_truth=tr, val_truth=va, test_truth=te,
                merged={u: tr.get(u, [])+va.get(u, []) for u in set(tr)|set(va)},
                device=torch.device("cpu"), aliases=(ua, ia), nu=NU, ni=NI, counts=pop.numpy(), logpop=torch.log1p(pop))


def make_args(tag, **kw):
    d = dict(loss="dau", tag=tag, epochs=4, eval_every=2, patience=None, cap_min=75., lr=.01, wd=1e-8, batch=64,
             run_id=tag, method="GraphHash", min_delta=0.0, init_match=False, equal_item_norm=False,
             model="MF", lightgcn_layers=2)
    d.update(kw)
    return SimpleNamespace(**d)


def dense_adj(c):
    e = c["parts"][0]
    A = torch.zeros(NU+NI, NU+NI)
    A[e[0], NU+e[1]] = 1.
    A[NU+e[1], e[0]] = 1.
    d = A.sum(1)+1e-7
    dinv = d.pow(-.5)
    return dinv[:, None]*A*dinv[None, :]


# ----------------------------------------------------------------------------- LightGCN
def test_norm_adj_symmetric_and_train_only():
    c = make_ctx()
    adj = E.build_norm_adj(c["parts"][0], NU, NI, torch.device("cpu"))
    assert adj.shape == (NU+NI, NU+NI)
    assert torch.allclose(adj.to_dense(), dense_adj(c), atol=1e-6)
    assert torch.allclose(adj.to_dense(), adj.to_dense().t())
    n_unique = len({(u, i) for u, i in c["parts"][0].t().tolist()})
    assert adj._nnz() == 2*n_unique                       # train edges only (val/test are not in parts[0])


def test_lightgcn_forward_shapes_and_math():
    for two_d in (False, True):
        c = make_ctx(two_d)
        m = MFDAU(8, 10, 16)
        adj = E.build_norm_adj(c["parts"][0], NU, NI, torch.device("cpu"))
        prop = E.LightProp(adj, NU, NI, 2)
        U, I = prop(m, c["aliases"])
        assert U.shape == (NU, 16) and I.shape == (NI, 16)
        ego = torch.cat(E.all_vectors(m, c["aliases"]))
        A = dense_adj(c)
        want = (ego + A@ego + A@(A@ego))/3
        assert torch.allclose(torch.cat([U, I]), want, atol=1e-5)
        uv, iv = E.all_vectors(m, c["aliases"], prop)
        assert torch.allclose(uv, U) and torch.allclose(iv, I)
        # propagation makes items of one bucket differ (per ORIGINAL item vectors)
        same = (c["aliases"][1][:, 0] == c["aliases"][1][0, 0]).nonzero().flatten() if two_d else \
            (c["aliases"][1] == c["aliases"][1][0]).nonzero().flatten()
        if len(same) > 1:
            assert not torch.allclose(I[same[0]], I[same[1]])


def test_lightgcn_k0_equals_mf():
    c = make_ctx()
    m = MFDAU(8, 10, 16)
    prop = E.LightProp(E.build_norm_adj(c["parts"][0], NU, NI, torch.device("cpu")), NU, NI, 0)
    U, I = prop(m, c["aliases"])
    uv, iv = E.all_vectors(m, c["aliases"])
    assert torch.allclose(U, uv) and torch.allclose(I, iv)
    # and the MF score equals the K=0 LightGCN score
    mf = MF(8, 10, 16)
    ua, ia = c["aliases"][0][:5], c["aliases"][1][:5]
    ps, _ = mf(ua, ia, ia)
    U, I = prop(mf, c["aliases"])
    assert torch.allclose(ps, (U[:5]*I[:5]).sum(-1), atol=1e-6)


def test_lightgcn_smoke_all_losses():
    for two_d in (False, True):
        for loss in ("dau", "bpr_author", "dau_bias", "dau_resbias"):
            tag = f"s4_lgcn_{loss}_{two_d}"
            args = make_args(tag, loss=loss, model="LightGCN", batch=64 if loss != "bpr_author" else None)
            result = {"seeds": {}}
            out = Path(tmp)/f"{tag}.json"
            E.run_seed(args, 0, make_ctx(two_d), result, out)
            row = json.loads(out.read_text())["seeds"]["0"]
            assert row["model"] == "LightGCN" and row["layers"] == 2 and row["epochs_run"] == 4
            assert {"raw", "cosine", "cos_prior", "raw_prior"} <= set(row["eval"])
            assert ("learned" in row["eval"]) == (loss == "dau_resbias")
            assert (E.R.ROOT/"runs"/tag/"per_user_test__raw.npz").exists()


def test_lightgcn_changes_training_vs_mf():
    c = make_ctx()
    outs = {}
    for model in ("MF", "LightGCN"):
        tag = f"s4_cmp_{model}"
        result = {"seeds": {}}
        E.run_seed(make_args(tag, model=model), 0, c, result, Path(tmp)/f"{tag}.json")
        outs[model] = result["seeds"]["0"]
    assert "layers" not in outs["MF"] and outs["LightGCN"]["layers"] == 2


def test_config_records_model_and_label():
    c = make_ctx()
    c["info"] = {"users": NU, "items": NI}
    for model, method in (("LightGCN", "DoubleGraph"), ("MF", "GraphHash")):
        cfg = E.build_config(make_args("cfg", model=model, method=method, resolution=200, seeds=[0]), c,
                             {"users": "x", "items": "y"})
        assert cfg["model"] == model and cfg["args"]["model"] == model
        assert cfg["lightgcn_layers"] == (2 if model == "LightGCN" else None)
        assert ("SUBSTITUTE" in cfg["method_label"]) == (method == "DoubleGraph")


# ----------------------------------------------------------------------------- dau_resbias
def test_resbias_param_groups_and_learned_mode():
    captured = {}
    real = torch.optim.Adam

    def spy(groups, **kw):
        captured["groups"] = groups
        return real(groups, **kw)
    torch.optim.Adam = spy
    try:
        tag = "s4_resbias"
        result = {"seeds": {}}
        out = Path(tmp)/f"{tag}.json"
        E.run_seed(make_args(tag, loss="dau_resbias"), 0, make_ctx(), result, out)
    finally:
        torch.optim.Adam = real
    g = captured["groups"]
    assert len(g) == 3
    wds = sorted(x["weight_decay"] for x in g)
    assert wds == [0.0, 1e-8, 0.01]
    r_group = next(x for x in g if x["weight_decay"] == 0.01)
    beta_group = next(x for x in g if x["weight_decay"] == 0.0)
    assert len(r_group["params"]) == 1 and r_group["params"][0].shape == (NI,)
    assert len(beta_group["params"]) == 1 and beta_group["params"][0].dim() == 0
    row = json.loads(out.read_text())["seeds"]["0"]
    ev = row["eval"]
    assert {"raw", "cosine", "cos_prior", "raw_prior", "learned"} <= set(ev) and "cos_b" not in ev
    assert ev["learned"]["best_beta"] == 0.0
    ln = row["learned"]
    assert {"beta", "r_norm", "r_abs_max"} <= set(ln) and ln["r_norm"] >= 0
    ck = torch.load(E.R.ROOT/"runs"/tag/"best_seed0.pt", weights_only=True)
    assert {"beta", "r"} <= set(ck) and ck["r"].shape == (NI,) and ck["beta"].dim() == 0
    assert abs(float(ck["beta"])-ln["beta"]) < 1e-7
    assert np.isclose(float(ck["r"].norm()), ln["r_norm"], rtol=1e-5)
    pu = np.load(E.R.ROOT/"runs"/tag/"per_user_test__learned.npz")
    assert pu["top20"].shape[1] == 20


def test_resbias_gradients_and_learned_score():
    c = make_ctx()
    m = MFDAU(8, 10, 8)
    beta = torch.nn.Parameter(torch.zeros(()))
    r = torch.nn.Parameter(torch.zeros(NI))
    uid, iid, neg = torch.randint(NU, (16,)), torch.randint(NI, (16,)), torch.randint(NI, (16,))
    u, i = m(c["aliases"][0][uid], c["aliases"][1][iid])
    negv = E.table_vec(m.item_emb_table, c["aliases"][1][neg])
    loss = E.bias_loss_from_vecs(DAULoss(1.), BPRLoss(), u, i, negv, E.resbias_add(beta, r, c["logpop"], iid),
                                 E.resbias_add(beta, r, c["logpop"], neg))
    loss.backward()
    assert beta.grad is not None and abs(float(beta.grad)) > 0
    touched = set(iid.tolist()) | set(neg.tolist())
    assert all(r.grad[j] == 0 for j in range(NI) if j not in touched) and r.grad.abs().sum() > 0
    # eval_suite emits `learned` only when learned_add is given
    uv, iv = E.all_vectors(m, c["aliases"])
    base = (uv, iv, c["val_truth"], c["test_truth"], c["train_truth"], c["merged"], c["logpop"])
    assert "learned" not in E.eval_suite(*base)
    ev = E.eval_suite(*base, learned_add=5.*c["logpop"])
    assert "learned" in ev
    cp = E.eval_suite(*base)["cos_prior"]["test"]["recall"]
    assert ev["learned"]["test"]["recall"] >= 0 and 0 <= cp <= 1


# ----------------------------------------------------------------------------- DoubleGraph
class FakeLouvain:
    """Stand-in for sknetwork Louvain (not installed locally): deterministic bipartite clustering by index."""
    def __init__(self, resolution=1):
        self.resolution = resolution

    def fit(self, biadjacency, force_bipartite=True):
        nu, ni = biadjacency.shape
        self.labels_row_ = np.arange(nu) % 7 + 100      # non-consecutive labels, as Louvain gives
        self.labels_col_ = np.arange(ni) % 5 + 100


def test_double_graph_maps():
    nu, ni = 200, 120
    e = torch.stack([torch.randint(nu, (3000,)), torch.randint(ni, (3000,))])
    uc, ic = E.double_graph_maps(SRC/"GraphHash"/"utils.py", [e], nu, ni, 7, 5, 200, FakeLouvain)
    assert uc.shape == (nu, 2) and ic.shape == (ni, 2)
    assert uc.dtype == np.int64 and ic.dtype == np.int64
    assert uc.min() >= 0 and uc.max() < 7 and ic.max() < 5
    assert (uc[:, 1] == np.arange(nu) % 7).all() and (ic[:, 1] == np.arange(ni) % 5).all()   # cluster code column
    uc2, ic2 = E.double_graph_maps(SRC/"GraphHash"/"utils.py", [e], nu, ni, 7, 5, 200, FakeLouvain)
    assert (uc == uc2).all() and (ic == ic2).all()                                           # seeded
    al = (torch.tensor(uc), torch.tensor(ic))
    m = MFDAU(7, 5, 8)
    u, i = m(al[0][e[0, :32]], al[1][e[1, :32]])
    DAULoss(1.)(u, i).backward()
    assert m.item_emb_table.weight.grad.abs().sum() > 0
    assert E.all_vectors(m, al)[1].shape == (ni, 8)


def test_double_graph_training_smoke():
    c = make_ctx(two_d=True)
    tag = "s4_dg"
    result = {"seeds": {}}
    E.run_seed(make_args(tag, method="DoubleGraph", model="LightGCN"), 0, c, result, Path(tmp)/f"{tag}.json")
    assert result["seeds"]["0"]["epochs_run"] == 4


def test_cli_parse():
    base = ["--dataset", "Gowalla", "--method", "DoubleGraph", "--loss", "dau_resbias", "--model", "LightGCN",
            "--epochs", "2", "--eval-every", "1", "--lr", "1e-3", "--wd", "1e-8", "--run-id", "x1",
            "--equal-item-norm", "--resolution", "50"]
    a = E.parse(base)
    assert (a.method, a.loss, a.model, a.lightgcn_layers) == ("DoubleGraph", "dau_resbias", "LightGCN", 2)
    assert a.equal_item_norm and a.resolution == 50
    d = E.parse([x for x in base if x not in ("--model", "LightGCN")][:0] or
                ["--dataset", "Gowalla", "--method", "Full", "--loss", "dau", "--epochs", "2", "--eval-every", "1",
                 "--lr", "1e-3", "--wd", "1e-8", "--run-id", "x2"])
    assert d.model == "MF"


def test_equal_item_norm_still_works():
    c = make_ctx()
    tag = "s4_ein"
    result = {"seeds": {}}
    E.run_seed(make_args(tag, equal_item_norm=True, model="LightGCN"), 0, c, result, Path(tmp)/f"{tag}.json")
    ck = torch.load(E.R.ROOT/"runs"/tag/"best_seed0.pt", weights_only=True)
    n = ck["model"]["item_emb_table.weight"].norm(dim=1)
    assert float(n.max()-n.min()) < 1e-4


def test_normalize_job_keeps_gpus():
    import os
    os.environ["SESSION_NO_AUTORUN"] = "1"
    import job_queue as JQ
    j = JQ.normalize_job({"run_id": "x", "script": "exp_run.py", "args": "--a", "gpus": "0,1"})
    assert j["gpus"] == "0,1"
    assert "gpus" not in JQ.normalize_job({"run_id": "y", "script": "exp_run.py"})
