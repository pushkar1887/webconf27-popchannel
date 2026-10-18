"""Independently verify saved source and numeric receipts; no model refitting."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_hash(user, seed):
    mask = (1 << 64) - 1
    z = (int(user) + seed + 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return (z ^ (z >> 31)) % 256


def main():
    release = json.loads((ROOT / "release_receipt.json").read_text())
    audit = json.loads((ROOT / "support_audit.json").read_text())
    results = json.loads((ROOT / "baseline_results.json").read_text())
    selected = json.loads((ROOT / "validation_selection.json").read_text())["best"]
    metrics = pd.read_csv(ROOT / "baseline_metrics.csv")
    data = np.load(ROOT / "audit_arrays.npz")
    checks = []
    for source in release["files"]:
        path = Path(source["path"])
        assert path.stat().st_size == source["bytes"]
        assert sha(path) == source["sha256"]
    checks.append("All three extracted CSV sizes and SHA256 receipts verified")
    assert sha(ROOT / "protocol.json") == audit["source_protocol_sha256"] == results["protocol_sha256"]
    assert sha(ROOT / "validation_selection.json") == results["validation_selection_sha256"]
    assert sha(ROOT / "baseline_details.json") == results["baseline_details_sha256"]
    checks.append("Protocol and pre-evaluation validation-selection lineage verified")
    assert audit["raw_total_rows"] - audit["duplicate_key_quarantine"]["rows"] == audit["total_rows"]
    assert sum(x["rows"] for x in audit["split_counts"].values()) == audit["total_rows"]
    assert all(audit["temporal_checks"].values())
    assert all(x.get("duplicate_event_keys", 0) == 0 for x in audit["split_counts"].values())
    checks.append("Row conservation, nonoverlapping event-time bounds, and clean-key receipts verified")
    numeric_max_error, replay_max_error = 0.0, 0.0
    for split in ["test_standard", "test_random"]:
        predictions = np.load(ROOT / ("predictions_" + split + ".npz"))
        y = data[split + "__long_view"].astype(bool)
        users, items, contexts = [data[split + "__" + field] for field in ["user_id", "video_id", "tab"]]
        for row in metrics.loc[metrics.split == split].to_dict("records"):
            name = row["model"]
            p = predictions[name].astype("float64")
            assert len(p) == len(y) == audit["split_counts"][split]["rows"]
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            picked_probability = np.clip(np.where(y, p, 1 - p), 1e-6, 1 - 1e-6)
            ll = float(-np.log(picked_probability).mean())
            brier = float(np.square(p - y).mean())
            error = max(abs(ll - row["log_loss"]), abs(brier - row["brier"]))
            assert error < 1e-10, (split, name, error)
            numeric_max_error = max(numeric_max_error, error)
            model = np.load(ROOT / "count_models" / (name + ".npz"))
            assert sum(model[key].nbytes for key in model.files) == results["selected_models"][name]["fitted_probability_array_bytes"]
            cfg = selected[name]
            for j in np.linspace(0, len(y) - 1, 101).astype(int):
                u, i, c = int(users[j]), int(items[j]), int(contexts[j])
                if name == "global_rate":
                    replay = float(model["global"][0])
                else:
                    pc = float(model["context"][c])
                    pi = float(model["item"][i, c]) if "item" in model.files and i < len(model["item"]) else pc
                    pu = float(model["user"][u, c]) if "user" in model.files and u < len(model["user"]) else pc
                    if name == "context_rate":
                        replay = pc
                    elif name == "item_context_rate":
                        replay = pi
                    elif name == "user_context_rate":
                        replay = pu
                    else:
                        lg = lambda x: math.log(max(1e-6, min(1 - 1e-6, x))) - math.log1p(-max(1e-6, min(1 - 1e-6, x)))
                        if name == "item_plus_user_marginal":
                            train_y = data["train__long_view"]
                            global_p = float((train_y.sum() + .5) / (len(train_y) + 1))
                            pm = float(model["user_marginal"][u]) if u < len(model["user_marginal"]) else global_p
                            effect = lg(pm) - lg(global_p)
                        elif "random_shared" in name:
                            seed = int(name.rsplit("_s", 1)[1])
                            effect = lg(float(model["group"][scalar_hash(u, seed), c])) - lg(pc)
                        else:
                            effect = lg(pu) - lg(pc)
                        replay = 1 / (1 + math.exp(-max(-25, min(25, lg(pi) + cfg["weight"] * effect))))
                replay_error = abs(replay - p[j])
                assert replay_error < 1e-7, (split, name, j, replay_error)
                replay_max_error = max(replay_max_error, replay_error)
    checks.append("All 18 metric rows recomputed independently from saved probabilities")
    checks.append("Saved model replay checked on 101 records per model/policy, including scalar hash implementation")
    validation = json.loads((ROOT / "validation_selection.json").read_text())
    for name, cfg in selected.items():
        minimum = min(x["validation_log_loss"] for x in validation["candidates"] if x["model"] == name)
        assert abs(cfg["validation_log_loss"] - minimum) < 1e-12
    checks.append("Hyperparameter selections are minima of the recorded validation-only grid")
    verification = {
        "status": "saved_artifact_checks_passed",
        "checks": checks,
        "max_metric_recomputation_error": numeric_max_error,
        "max_saved_model_replay_error": replay_max_error,
        "documented_target_rule_check": "FAILED on native source data; preserved and reported rather than repaired",
        "exact_documentation_consistency": audit["documentation_consistency_passed"],
        "scientific_scope": "component feasibility evidence only; no claim that the proposed compression mechanism or method is validated",
    }
    (ROOT / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    main()
