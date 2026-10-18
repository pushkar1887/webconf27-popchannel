"""Popularity-only baseline: rank items by train count, mask seen items. No model.

Usage: python poponly.py [--dataset {Gowalla,Yelp2018,AmazonBook}] [--out result.json]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
W = Path(__file__).parent.parent/"2026-10-16-loss-sharing-repro"/"_work"
sys.path.insert(0, str(W))
from repro_core import metrics_from_topk

K, CHUNK = 20, 1024
DATASETS = ["Gowalla", "Yelp2018", "AmazonBook"]


def group(edges, n_users):
    """edges: 2 x N (user, item). Returns CSR (ptr, users_sorted, items_sorted) grouped by user."""
    u, i = edges
    order = torch.argsort(u, stable=True)
    ptr = torch.zeros(n_users + 1, dtype=torch.long)
    ptr[1:] = torch.bincount(u, minlength=n_users).cumsum(0)
    return ptr, u[order], i[order]


def evaluate(truth, mask, pop, n_users):
    """Top-K most popular items per truth user, with the `mask` items of that user removed."""
    tp, _, ti = group(truth, n_users)
    mp, mu, mi = group(mask, n_users)
    tpl = tp.tolist()
    ti_np = ti.numpy()
    has = ((tp[1:] - tp[:-1]) > 0)
    preds, truths = [], []
    for lo in range(0, n_users, CHUNK):
        hi = min(lo + CHUNK, n_users)
        loc = torch.nonzero(has[lo:hi]).flatten()  # chunk-local rows that have truth
        if loc.numel() == 0:
            continue
        s = pop.unsqueeze(0).repeat(hi - lo, 1)  # masked copy of the popularity vector per user
        a, b = int(mp[lo]), int(mp[hi])
        s[mu[a:b] - lo, mi[a:b]] = -1
        preds.append(torch.topk(s[loc], K, dim=1).indices.numpy())
        for u in (loc + lo).tolist():
            truths.append(ti_np[tpl[u]:tpl[u + 1]].tolist())
    m = metrics_from_topk(np.concatenate(preds), truths, K)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="Gowalla")
    ap.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    args = ap.parse_args()

    P = W/"datasets"/f"{args.dataset}-processed"
    tr, va, te = [torch.load(P/f"{s}.pt", weights_only=True).long() for s in ["train", "val", "test"]]
    n_users = int(max(x[0].max() for x in [tr, va, te])) + 1
    ni = int(max(x[1].max() for x in [tr, va, te])) + 1
    pop = torch.bincount(tr[1], minlength=ni).float()

    val_m = evaluate(va, tr, pop, n_users)                                # val: mask train
    test_m = evaluate(te, torch.cat([tr, va], dim=1), pop, n_users)       # test: mask train+val
    res = {
        "dataset": args.dataset,
        "val": {k: round(100 * val_m[k], 2) for k in ["recall", "ndcg"]},
        "test": {k: round(100 * test_m[k], 2) for k in ["recall", "ndcg"]},
        "n_val_users": val_m["evaluated_users"],
        "n_test_users": test_m["evaluated_users"],
        "split_sizes": {"train": int(tr.shape[1]), "val": int(va.shape[1]), "test": int(te.shape[1])},
    }
    print(json.dumps(res))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
