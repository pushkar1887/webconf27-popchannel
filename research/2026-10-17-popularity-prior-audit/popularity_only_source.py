"""Popularity-only baseline: rank items by train count, mask seen items. No model."""
import json, sys
from pathlib import Path
import numpy as np, torch
W = Path(__file__).parent.parent/"2026-10-16-loss-sharing-repro"/"_work"
sys.path.insert(0, str(W))
from repro_core import metrics_from_topk
P = W/"datasets"/"Gowalla-processed"
tr, va, te = [torch.load(P/f"{s}.pt", weights_only=True).long() for s in ["train", "val", "test"]]
ni = int(max(x[1].max() for x in [tr, va, te]))+1
pop = torch.bincount(tr[1], minlength=ni).float()
def d(e):
    out = {}
    for u, i in e.t().tolist(): out.setdefault(u, []).append(i)
    return out
T, V, S = d(tr), d(va), d(te)
M = {u: T.get(u, [])+V.get(u, []) for u in set(T)|set(V)}
def run(truth, mask):
    keys = list(truth); preds = []
    for k in keys:
        s = pop.clone(); s[mask.get(k, [])] = -1
        preds.append(torch.topk(s, 20).indices.numpy())
    m = metrics_from_topk(np.stack(preds), [truth[k] for k in keys], 20)
    return {k: round(100*m[k], 2) for k in ["recall", "ndcg"]}
r = {"val": run(V, T), "test": run(S, M)}
print(json.dumps(r)); Path("results/poponly.json").write_text(json.dumps(r))
