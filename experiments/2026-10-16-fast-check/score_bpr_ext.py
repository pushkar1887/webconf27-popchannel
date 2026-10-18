"""P1 (no training): raw_prior sweep on the GraphHash x BPR authors'-config Gowalla checkpoint.

Checkpoint: /kaggle/input/**/best.pt (kernel_sources graphhash-fast-bpr-author-1016, account-3); a plain MF
state_dict. score = u.i + beta*log1p(train_count); beta tuned on VALIDATION, test evaluated once at best beta.
Optional: --ckpt PATH to override the glob. Output: p1_bpr_ext.json
"""
import argparse
import glob
import json
import numpy as np
import torch
import repro_run as R
import exp_run as E


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    args = p.parse_args(argv)
    paths = [args.ckpt] if args.ckpt else sorted(glob.glob("/kaggle/input/**/best.pt", recursive=True))
    assert paths, "no best.pt found"
    path = next((x for x in paths if "bpr" in x), paths[0])
    R.log("p1_ckpt", path=path, candidates=paths)
    MF, MFDAU, BPRLoss, DAULoss, namespace = R.prepare()
    parts, maps, (train_truth, val_truth, test_truth), info = R.load_data("Gowalla", namespace)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aliases = tuple(torch.tensor(np.asarray(x), dtype=torch.long, device=device) for x in maps["GraphHash"])
    model = MF(int(aliases[0].max())+1, int(aliases[1].max())+1, 64).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    merged = {u: train_truth.get(u, [])+val_truth.get(u, []) for u in set(train_truth)|set(val_truth)}
    pop = torch.bincount(parts[0][1], minlength=len(maps["Full"][1])).float()
    logpop = torch.log1p(pop).to(device)
    uvec, ivec = E.all_vectors(model, aliases)

    def eval_fn(split, bs):
        truth, masks = (val_truth, train_truth) if split == "val" else (test_truth, merged)
        return E.sweep(uvec, ivec, truth, masks, bs, logpop, None, cosine=False)

    out = {"checkpoint": path, "betas": E.BETAS, "raw_prior": E.tune_and_test(eval_fn, E.BETAS),
           "spearman_norm_vs_count": E.spearman_norm_vs_count(ivec, pop.numpy())}
    (R.ROOT/"p1_bpr_ext.json").write_text(json.dumps(out, indent=1))
    R.log("p1_done", best_beta=out["raw_prior"]["best_beta"], val=out["raw_prior"]["val"],
          test=out["raw_prior"]["test"])


if __name__ == "__main__":
    main()
