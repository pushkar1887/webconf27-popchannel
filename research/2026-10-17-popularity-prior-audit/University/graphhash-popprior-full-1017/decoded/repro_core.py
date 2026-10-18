"""Small correctness-critical helpers for the bounded reproduction."""
import math
import numpy as np
import torch
import torch.nn.functional as F


def expired(seconds, limit=5400):
    return seconds >= limit


def split_statistics(parts):
    all_edges = torch.cat(parts, dim=1)
    return {"users": int(torch.unique(all_edges[0]).numel()),
            "items": int(torch.unique(all_edges[1]).numel()),
            "user_id_max": int(all_edges[0].max()),
            "item_id_max": int(all_edges[1].max()),
            "interactions": int(all_edges.shape[1]),
            "split_counts": [int(x.shape[1]) for x in parts]}


def metrics_from_topk(topk, truths, k=20):
    discounts = 1 / np.log2(np.arange(2, k + 2))
    recall, ndcg, standard = [], [], []
    for pred, truth in zip(topk, truths):
        if not len(truth):
            continue
        hit = np.isin(pred[:k], truth).astype(np.float64)
        capped = min(k, len(truth))
        recall.append(float(hit.sum()) / capped)
        standard.append(float(hit.sum()) / len(truth))
        ndcg.append(float(hit @ discounts) / float(discounts[:capped].sum()))
    return {"recall": float(np.mean(recall)), "ndcg": float(np.mean(ndcg)),
            "standard_recall": float(np.mean(standard)), "evaluated_users": len(recall)}


def rank_topk(users, items, masks, k=20, cosine=False):
    if cosine:
        users, items = F.normalize(users, dim=-1), F.normalize(items, dim=-1)
    scores = users @ items.t()
    rows, cols = [], []
    for row, mask in enumerate(masks):
        rows.extend([row] * len(mask))
        cols.extend(mask)
    if rows:
        scores[torch.as_tensor(rows, device=scores.device),
               torch.as_tensor(cols, device=scores.device)] = -1e4
    return torch.topk(scores, k, dim=1).indices


def pair_diagnostics(original_ids, aliases, table):
    original_ids, aliases = original_ids.long(), aliases.long()
    m = len(aliases)
    if m < 2:
        return {"rho": None, "rho_repeat": None, "rho_alias": None,
                "Z": None, "alpha": None, "collision_share": None}
    _, id_counts = torch.unique(original_ids, return_counts=True)
    _, counts = torch.unique(aliases, return_counts=True)
    total = m * (m - 1)
    repeat = int((id_counts * (id_counts - 1)).sum()) / total
    rho = int((counts * (counts - 1)).sum()) / total
    if rho >= 1:
        return {"rho": rho, "rho_repeat": repeat, "rho_alias": rho-repeat,
                "Z": None, "alpha": 0., "collision_share": 1.}
    vectors = F.normalize(table[aliases], dim=-1)
    pair = torch.triu_indices(m, m, offset=1, device=aliases.device)
    collided = aliases[pair[0]] == aliases[pair[1]]
    kernel = (-2 * torch.pdist(vectors).square()).exp()
    z = float(kernel[~collided].mean())
    alpha = (1-rho) * z / (rho + (1-rho) * z)
    return {"rho": rho, "rho_repeat": repeat, "rho_alias": rho-repeat,
            "Z": z, "alpha": alpha, "collision_share": 1-alpha}
