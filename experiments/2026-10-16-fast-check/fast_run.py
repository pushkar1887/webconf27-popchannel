"""FAST CHECK: one of three fixed-budget GraphHash runs on Gowalla (MF, dim 64).

Runs: bpr | dau | dau_fix. All share lr/wd/batch/gamma, 30 epochs, no early stopping.
Validation Recall@20 every 5 epochs; test at epoch 30 (last epoch, no selection).
dau_fix: uniformity computed over UNIQUE bucket ids in each batch (alignment unchanged).
Reuses the pinned author data/mapping/metric code through repro_run.
"""
import json
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import repro_run as R

RUN = sys.argv[1]
SEEDS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0]
EPOCHS, EVAL_EVERY = 30, 5
LR, WD, BATCH, GAMMA = .01, 1e-8, 1024, 1.
PATIENCE = None
if RUN == "bpr_author":  # authors' GraphHash x BPR command: full batch, lr .01, wd 1e-6
    EPOCHS, EVAL_EVERY, PATIENCE = 3000, 10, 100
    WD, BATCH = 1e-6, 1027370
if RUN == "bpr_long":  # fast-check settings, longer: max 300 epochs, eval every 5, patience 10 epochs
    EPOCHS, PATIENCE = 300, 10
CAP_SECONDS = 4800


def unique_uniformity(emb, buckets):
    _, first = np.unique(buckets.cpu().numpy(), return_index=True)
    return R_DAU.uniformity(emb[torch.as_tensor(first, device=emb.device)])


def main():
    for seed in SEEDS:
        train(seed)


def train(seed):
    global R_DAU
    OUT = R.ROOT / (f"fast_{RUN}.json" if SEEDS == [0] else f"fast_{RUN}_seed{seed}.json")
    torch.backends.cuda.matmul.allow_tf32 = False
    MF, MFDAU, BPRLoss, DAULoss, namespace = R.prepare()
    R_DAU = DAULoss
    parts, maps, (train_truth, val_truth, test_truth), info = R.load_data("Gowalla", namespace)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    device = torch.device("cuda")
    aliases = tuple(torch.tensor(x, dtype=torch.long, device=device) for x in maps["GraphHash"])
    cls = MF if RUN.startswith("bpr") else MFDAU
    model = cls(int(aliases[0].max())+1, int(aliases[1].max())+1, 64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    bpr, dau = BPRLoss(), DAULoss(GAMMA)
    edges = parts[0].to(device)
    merged = {u: train_truth.get(u, [])+val_truth.get(u, []) for u in set(train_truth)|set(val_truth)}
    result = {"run": RUN, "seed": seed, "lr": LR, "wd": WD, "batch": BATCH, "gamma": GAMMA,
              "user_buckets": info["user_buckets"], "item_buckets": info["item_buckets"],
              "val": {}, "dup_frac_user": [], "dup_frac_item": []}
    started = time.perf_counter()
    for epoch in range(1, EPOCHS+1):
        model.train()
        order = torch.randperm(edges.shape[1], device=device)
        du = di = nb = 0.
        for first in range(0, edges.shape[1], BATCH):
            uid, iid = edges[:, order[first:first+BATCH]]
            ua, ia = aliases[0][uid], aliases[1][iid]
            n = len(ua)
            du += 1 - torch.unique(ua).numel()/n; di += 1 - torch.unique(ia).numel()/n; nb += 1
            opt.zero_grad(set_to_none=True)
            if RUN.startswith("bpr"):
                neg = torch.randint(len(aliases[1]), iid.shape, device=device)
                loss = bpr(*model(ua, ia, aliases[1][neg]))
            else:
                u, i = model(ua, ia)
                if RUN.startswith("dau") and not RUN.startswith("dau_fix"):
                    loss = dau(u, i)
                else:
                    loss = DAULoss.alignment(u, i) + GAMMA*(unique_uniformity(u, ua)+unique_uniformity(i, ia))/2
            loss.backward()
            opt.step()
        result["dup_frac_user"].append(du/nb); result["dup_frac_item"].append(di/nb)
        if epoch % EVAL_EVERY == 0:
            model.eval()
            v, _ = R.evaluate(model, aliases, val_truth, train_truth, device)
            result["val"][epoch] = v["recall"]
            R.log("fast_val", run=RUN, epoch=epoch, val_recall=v["recall"],
                  elapsed=round(time.perf_counter()-started, 1))
            if v["recall"] > result.get("best_val", -1):
                result.update(best_val=v["recall"], best_epoch=epoch)
                torch.save(model.state_dict(), R.WORK/f"best_seed{seed}.pt")
            OUT.write_text(json.dumps(result, indent=1))
            if PATIENCE and epoch-result["best_epoch"] >= PATIENCE:
                break
        if time.perf_counter()-started > CAP_SECONDS:
            result["stopped"] = "time_cap"; break
    result["epochs_run"] = epoch
    if PATIENCE:
        model.load_state_dict(torch.load(R.WORK/f"best_seed{seed}.pt", weights_only=True)); model.eval()
    t, _ = R.evaluate(model, aliases, test_truth, merged, device)
    result.update({"test_recall": t["recall"], "test_ndcg": t["ndcg"],
                   "test_standard_recall": t["standard_recall"],
                   "train_seconds": time.perf_counter()-started})
    OUT.write_text(json.dumps(result, indent=1))
    R.log("fast_done", **{k: result[k] for k in ["run", "test_recall", "test_ndcg", "train_seconds"]})


if __name__ == "__main__":
    main()
