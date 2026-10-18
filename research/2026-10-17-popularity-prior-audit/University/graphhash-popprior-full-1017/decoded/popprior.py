"""Score = cosine(u,i) + beta*log(1+train_pop(i)). Tune beta on val, test once. No training.

Checkpoints found under /kaggle/input: best*.pt -> GraphHash mapping; *Full_DirectAU*.pt -> Full.
Also reports raw-dot at beta=0 as a sanity check against earlier numbers.
"""
import glob, json
import numpy as np
import torch
import torch.nn.functional as F
import repro_run as R
from repro_core import metrics_from_topk

BETAS = [0, .01, .02, .05, .1, .2, .5]


def score(model, aliases, truth, masks, logpop, beta, device, cosine=True):
    u_map, i_map = aliases
    keys, preds = list(truth), []
    with torch.no_grad():
        items = model.item_emb_table(i_map)
        if cosine:
            items = F.normalize(items, dim=-1)
        for s in range(0, len(keys), 512):
            users = keys[s:s+512]
            u = model.user_emb_table(u_map[torch.tensor(users, device=device)])
            if cosine:
                u = F.normalize(u, dim=-1)
            sc = u @ items.t() + beta*logpop[None, :]
            rows = [r for r, x in enumerate(users) for _ in masks.get(x, [])]
            cols = [c for x in users for c in masks.get(x, [])]
            if rows:
                sc[torch.tensor(rows, device=device), torch.tensor(cols, device=device)] = -1e4
            preds.append(torch.topk(sc, 20, dim=1).indices.cpu().numpy())
    m = metrics_from_topk(np.concatenate(preds), [truth[x] for x in keys], 20)
    return {"recall": m["recall"], "ndcg": m["ndcg"]}


def main():
    MF, MFDAU, BPRLoss, DAULoss, namespace = R.prepare()
    parts, maps, (train_truth, val_truth, test_truth), info = R.load_data("Gowalla", namespace)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    merged = {u: train_truth.get(u, [])+val_truth.get(u, []) for u in set(train_truth)|set(val_truth)}
    ni = len(maps["Full"][1])
    pop = torch.bincount(parts[0][1], minlength=ni).float().to(device)
    logpop = torch.log1p(pop)
    out = {}
    for path in glob.glob("/kaggle/input/**/*.pt", recursive=True):
        method = "Full" if "Full_DirectAU" in path else "GraphHash" if "best" in path else None
        if method is None:
            continue
        aliases = tuple(torch.tensor(x, dtype=torch.long, device=device) for x in maps[method])
        model = MF(int(aliases[0].max())+1, int(aliases[1].max())+1, 64).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        val = {b: score(model, aliases, val_truth, train_truth, logpop, b, device) for b in BETAS}
        best = max(BETAS, key=lambda b: val[b]["recall"])
        row = {"method": method, "path": path, "val_by_beta": val, "best_beta": best,
               "test_beta0": score(model, aliases, test_truth, merged, logpop, 0, device),
               "test_best": score(model, aliases, test_truth, merged, logpop, best, device),
               "test_raw_dot": score(model, aliases, test_truth, merged, logpop, 0, device, cosine=False)}
        out[method] = row
        R.log("popprior", **{k: v for k, v in row.items() if k != "val_by_beta"})
    (R.ROOT/"popprior.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
