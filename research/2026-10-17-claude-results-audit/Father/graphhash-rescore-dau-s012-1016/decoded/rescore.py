"""Re-score saved GraphHash checkpoints on test: raw dot vs cosine. No training."""
import glob, json, sys
import torch
import repro_run as R


def main():
    MF, MFDAU, BPRLoss, DAULoss, namespace = R.prepare()
    parts, maps, (train_truth, val_truth, test_truth), info = R.load_data("Gowalla", namespace)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aliases = tuple(torch.tensor(x, dtype=torch.long, device=device) for x in maps["GraphHash"])
    merged = {u: train_truth.get(u, [])+val_truth.get(u, []) for u in set(train_truth)|set(val_truth)}
    out = {}
    for path in glob.glob("/kaggle/input/**/best.pt", recursive=True):
        model = MF(int(aliases[0].max())+1, int(aliases[1].max())+1, 64).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        row = {}
        for scoring in ["raw", "cosine"]:
            m, _ = R.evaluate(model, aliases, test_truth, merged, device, cosine=scoring == "cosine")
            row.update({f"{scoring}_recall": m["recall"], f"{scoring}_ndcg": m["ndcg"]})
        out[path] = row
        R.log("rescore", path=path, **row)
    (R.ROOT/"rescore.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
