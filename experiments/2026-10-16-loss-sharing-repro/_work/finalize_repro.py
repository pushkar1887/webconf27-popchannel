"""Verify retained predictions against the frozen split; annotate the two deliverables."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from repro_core import metrics_from_topk

torch.set_num_threads(4)
WORK = Path(__file__).parent
ROOT = WORK.parent
version = json.loads((WORK/"submission.json").read_text())["version"]
retrieved = WORK/f"gpu-output-v{version}"
assert (retrieved/"REPRO_REPORT.md").exists(), "Completed remote report has not been retrieved."
rows = list(csv.DictReader((ROOT/"metrics.csv").open(newline="")))
cache, comparisons, receipts = {}, [], []
seen = set()
for row in rows:
    key = row["dataset"], row["method"], row["loss"], int(row["seed"])
    assert key not in seen
    seen.add(key)
    assert key[3] == 0
    if row["status"] != "complete":
        continue
    assert float(row["wall_seconds"]) <= 5400, "Completed condition exceeded the declared run cap."
    ds = row["dataset"]
    if ds not in cache:
        source = WORK/"datasets"/f"{ds}-processed"
        parts = [torch.load(source/f"{s}.pt", weights_only=True).long() for s in ["train","val","test"]]
        dictionaries = []
        for part in parts:
            d = {}
            for u,i in part.t().tolist():
                d.setdefault(u, []).append(i)
            dictionaries.append(d)
        cache[ds] = parts, dictionaries
    parts, (train, val, test) = cache[ds]
    run_id = f"{ds}_{row['method']}_{row['loss']}_seed0"
    checkpoint = retrieved/"_work"/(run_id+".pt")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert sum(t.numel() for t in state.values()) == int(row["parameters"])
    assert all(t.shape[1] == 64 for t in state.values())
    receipts.append({"file":str(checkpoint),"sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    for split, truth in [("val",val),("test",test)]:
        mask = train if split == "val" else {u:train.get(u,[])+val.get(u,[]) for u in set(train)|set(val)}
        for scoring in ["raw","cosine"]:
            path = retrieved/"_work"/f"{run_id}_{split}_{scoring}.npz"
            with np.load(path) as data:
                ids, predictions = data["users"], data["top20"]
            assert predictions.shape == (len(ids),20)
            assert predictions.min() >= 0 and predictions.max() <= max(int(p[1].max()) for p in parts)
            for uid,pred in zip(ids,predictions):
                assert len(np.unique(pred)) == 20
                assert not set(pred).intersection(mask.get(int(uid),[]))
            metrics = metrics_from_topk(predictions, [truth[int(u)] for u in ids], 20)
            error = max(abs(metrics[m]-float(row[f"{split}_{scoring}_{m}"]))
                        for m in ["recall","ndcg","standard_recall"])
            assert error < 1e-10, (run_id,split,scoring,error)
            comparisons.append({"run":run_id,"split":split,"scoring":scoring,"max_error":error})
    for side in ["user","item"]:
        rho, repeat, alias = (float(row[f"{side}_{x}"]) for x in ["rho","rho_repeat","rho_alias"])
        assert abs(rho-repeat-alias) < 1e-10
        assert 0 <= rho <= 1 and -1e-12 <= alias <= 1 and -1e-12 <= repeat <= 1
        z = float(row[f"{side}_Z"]) if row.get(f"{side}_Z") else None
        alpha = float(row[f"{side}_alpha"])
        expected = (1-rho)*z/(rho+(1-rho)*z) if z is not None else 0.
        assert abs(expected-alpha) < 1e-10

primary = {(r["method"],r["loss"]):r for r in rows if r["dataset"]=="Gowalla" and r["status"]=="complete"}
expected = {(m,l) for m in ["Full","Random","GraphHash"] for l in ["BPR","DirectAU"]}
complete = set(primary) == expected
checks = {}
for method in ["Full","GraphHash","Random"]:
    b,d = primary.get((method,"BPR")), primary.get((method,"DirectAU"))
    if b and d:
        bv,dv = float(b["test_raw_recall"]),float(d["test_raw_recall"])
        checks[method] = dv > bv if method=="Full" else dv < bv
passed = complete and all(checks.values())
summary = {"primary_complete":complete,"primary_passed":passed,"direction_checks":checks,
           "completed_fits":sum(r["status"]=="complete" for r in rows),
           "prediction_metric_comparisons":comparisons,"checkpoint_receipts":receipts}
(WORK/"result_verification.json").write_text(json.dumps(summary,indent=2))
report = (ROOT/"REPRO_REPORT.md").read_text()
marker = "\n## Local verification and execution details\n"
report = report.split(marker)[0]
report += marker + "\n"
if comparisons:
    report += (f"[Certain] Recomputed {len(comparisons)} split/scoring metric sets from retained top-20 "
               "predictions and the original author split. Recall, NDCG and standard Recall agree with "
               "metrics.csv within 1e-10. Checked candidate masks, unique recommendations, checkpoint "
               "parameter counts and collision/alpha identities. These are artifact-consistency checks.\n\n")
else:
    report += "[Certain] No completed-fit predictions were available for metric verification. No reproduction success is established.\n\n"
report += ("Fixed protocol: MF, dimension 64, Adam (default betas 0.9/0.999, epsilon 1e-8), "
           "float32, seed 0; no grid or fixes. Per-condition learning rate, weight decay, batch size, "
           "gamma, epochs, selected epoch and peak CUDA allocation are in metrics.csv.\n\n")
report += ("Execution differences: GPU permutation/negative-sampling RNG rather than the author's "
           "CPU DataLoader RNG; common memory-bounded evaluation chunks; a simple 50-epoch clock "
           "rather than the source's stale-validation patience counter. This is a protocol reproduction, "
           "not a bitwise replication of the authors' seeds. Expensive diagnostic gradients use the fixed "
           "1024-edge probe. Mean embedding norms average over expanded original IDs. "
           "The 90-minute cap is checked between GPU operations and after final diagnostics; "
           "it is not an exact-second interrupt of an in-flight kernel.\n\n")
report += f"Account alias: University. Private notebook slug: graphhash-loss-sharing-repro-1016. Version: {version}.\n\n"
report += "### Compact checkpoint diagnostics\n\n"
report += "| Method | Loss | User repeat rho | User extra-alias rho | User alpha | Item repeat rho | Item extra-alias rho | Item alpha |\n"
report += "|---|---|---:|---:|---:|---:|---:|---:|\n"
for r in rows:
    if r["dataset"]=="Gowalla" and r["status"]=="complete":
        values=[f"{float(r[f'{side}_{name}']):.6g}" for side in ["user","item"]
                for name in ["rho_repeat","rho_alias","alpha"]]
        report += "| "+r["method"]+" | "+r["loss"]+" | "+" | ".join(values)+" |\n"
(ROOT/"REPRO_REPORT.md").write_text(report,encoding="utf-8")
print(json.dumps({k:v for k,v in summary.items() if k not in ["prediction_metric_comparisons","checkpoint_receipts"]},indent=2))
print(json.dumps({"run_times_minutes":[{"dataset":r["dataset"],"method":r["method"],"loss":r["loss"],
                                      "minutes":round(float(r["wall_seconds"])/60,3),"status":r["status"]}
                                     for r in rows]},indent=2))
