"""Prepare the requested Stage 3 sessions OFFLINE. This module has no Kaggle API calls.

Run here: python make_sessions_s3.py
Then:     python job_queue.py --sessions sessions_s3.py --dry-run
The orchestrator must approve/register/snapshot the predictions before any real submission.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import pprint
import shlex

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1] / "webconf27-popchannel"
RUNS = PROJECT / "runs"
DAU = "--loss dau --eval-every 5 --patience 30 --min-delta 0.0005 --cap-min 180"
BPR = "--loss bpr_author --eval-every 10 --patience 500 --min-delta 0 --cap-min 90"
OVERHEAD_MINUTES = 10.0  # planning reserve per job for setup and the final scoring suite, not a measurement
LANE_ESTIMATE_LIMIT_MINUTES = 540.0
LANE_BUDGET_LIMIT_MINUTES = 660.0
FINAL_BPR_SEEDS = frozenset({
    "gowalla__mf__graphhash__bpr__s0-p500",
    "gowalla__mf__graphhash__bpr__s1",
    "gowalla__mf__graphhash__bpr__s2",
})

# PLAN section 7.2: (BPR lr/wd, DirectAU lr/wd), independently checked against the document by test_s3.py.
HP = {
    "Gowalla": {
        "Full": ((.005, 1e-8), (.01, 1e-8)),
        "GraphHash": ((.01, 1e-6), (.01, 1e-8)),
        "Random": ((.001, 1e-6), (.01, 1e-6)),
        "DoubleFreq": ((.001, 1e-6), (.005, 1e-6)),
    },
    "Yelp2018": {
        "Full": ((.01, 1e-6), (.01, 1e-8)),
        "GraphHash": ((.005, 1e-6), (.005, 1e-6)),
        "Random": ((.001, 1e-6), (.005, 1e-6)),
        "DoubleFreq": ((.001, 1e-6), (.001, 1e-4)),
    },
    "AmazonBook": {
        "Full": ((.01, 1e-8), (.001, 1e-8)),
        "GraphHash": ((.01, 1e-6), (.001, 1e-8)),
        "Random": ((.005, 1e-6), (.001, 1e-6)),
        "DoubleFreq": ((.001, 1e-8), (.001, 1e-6)),
    },
}
METHOD_ID = {"Full": "full", "GraphHash": "graphhash", "Random": "random", "DoubleFreq": "dfreq"}


def validator_module():
    spec = importlib.util.spec_from_file_location("s3_strict_validator", PROJECT / "scripts" / "validate_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train(group, priority, dataset, method, loss, seed=0, suffix="", epochs=None, wd_override=None,
          cap_override=None, estimate_override=None):
    bpr = loss == "bpr"
    lr, wd = HP[dataset][method][0 if bpr else 1]
    if wd_override is not None:
        wd = wd_override
    # The legacy AmazonBook curve was still improving near epoch 40 when capped at epoch 41.
    # The authorized AmazonBook overrides allow 300 minutes for 150 epochs. This reduces
    # wall-cap risk; max_epochs can still bind and does not certify convergence.
    epochs = epochs if epochs is not None else (3000 if bpr else 150 if dataset != "Gowalla" else 200)
    rid = f"{dataset.lower()}__mf__{METHOD_ID[method]}__{loss}__s{seed}{suffix}"
    stopping = shlex.split(BPR if bpr else DAU)
    if cap_override is not None:
        stopping[stopping.index("--cap-min") + 1] = str(cap_override)
    cap = float(stopping[stopping.index("--cap-min") + 1])
    args = (f"--dataset {dataset} --method {method} {' '.join(stopping)} --epochs {epochs} "
            f"--lr {lr:g} --wd {wd:g} --seeds {seed} --run-id {rid}")
    if suffix == "-initmatch":
        args += " --init-match"
    if bpr:
        estimate = 60.0 if epochs == 20000 else 11.0
        basis = "A8r extrapolation from 3000 epochs/~9 min" if epochs == 20000 else "measured BPR range 4-11 min; reserve 11"
    elif dataset == "AmazonBook":
        estimate, basis = cap, "assume the full per-job 300-minute cap; legacy ~1.8 min/epoch; no convergence guarantee"
    elif method == "Full":
        estimate = 180.0 if wd_override is not None else 140.0
        basis = "use cap for unmeasured D7" if wd_override is not None else "Gowalla Full: ~39 s/epoch, up to 200 epochs, conservative reserve"
    elif dataset == "Gowalla":
        estimate = 38.0 if method == "GraphHash" else 45.0
        basis = "measured GraphHash DirectAU ~38 min" if method == "GraphHash" else "unmeasured sharing method; estimate from Gowalla timing"
    else:
        estimate, basis = 90.0, "unmeasured Yelp sharing method; reserve above measured GraphHash ~58 min"
    if estimate_override is not None:
        estimate = float(estimate_override)
        basis = "requested ~10-minute final-protocol seed-zero BPR rerun"
    return {"id": group, "priority": priority, "run_id": rid, "script": "exp_run.py", "args": args,
            "estimated_minutes": estimate + OVERHEAD_MINUTES, "budget_minutes": cap + OVERHEAD_MINUTES,
            "training_estimate_minutes": estimate, "overhead_minutes": OVERHEAD_MINUTES,
            "estimate_basis": basis,
            "hparam_exception": "D7 explicit weight-decay intervention from the hand-off" if wd_override is not None else None}


def planned_jobs():
    jobs = [train("A2", 1, "Gowalla", "Full", "directau"),
            train("A8r", 1.5, "Gowalla", "Full", "bpr", suffix="-long", epochs=20000),
            train("B9", 2, "AmazonBook", "GraphHash", "directau", cap_override=300),
            train("BPR-s0-p500", 3, "Gowalla", "GraphHash", "bpr", suffix="-p500", estimate_override=10)]
    for group, loss, seed in [("B5", "directau", 1), ("B6", "directau", 2),
                              ("B7", "bpr", 1), ("B8", "bpr", 2)]:
        jobs.append(train(group, 3, "Gowalla", "GraphHash", loss, seed))
    group_ids = {"Gowalla": ["B1", "B2", "B3", "B4"],
                 "Yelp2018": ["B15", "B16", "B17", "B18"],
                 "AmazonBook": ["B10", "B11", "B12", "B13"]}
    for dataset in ["Gowalla", "Yelp2018", "AmazonBook"]:
        cells = [(m, l) for m in ["Random", "DoubleFreq"] for l in ["bpr", "directau"]]
        jobs.extend(train(group, 4, dataset, method, loss,
                          cap_override=300 if dataset == "AmazonBook" and loss == "directau" else None)
                    for group, (method, loss) in zip(group_ids[dataset], cells))
    jobs += [train("B14", 4, "AmazonBook", "Full", "bpr"),
             train("I2", 5, "AmazonBook", "GraphHash", "directau", suffix="-initmatch", cap_override=300),
             train("I2-partner", 5, "AmazonBook", "Full", "directau", cap_override=300),
             train("D7-wd0", 7, "Gowalla", "Full", "directau", suffix="-wd0", wd_override=0.0),
             train("D7-wd1e-4", 7, "Gowalla", "Full", "directau", suffix="-wd1e-4", wd_override=1e-4)]
    return jobs


def c3_support(runner=HERE / "exp_run.py"):
    text = Path(runner).read_text(encoding="utf8")
    # The existing dau_bias mode is the earlier unconstrained cos+b experiment. A matching mode
    # needs a learned beta*logpop term and a separately regularized residual parameter group.
    if "def dau_bias_loss" in text and "s_pos = cosine_pair(u, i) + b[iid]" in text:
        return False, "unsupported: existing dau_bias is the old cos+b loss, without learned beta*logpop or the C3 residual optimizer group"
    return False, "unsupported: C3's exact residual-prior training contract has not been verified; no new training mode was requested"


def filter_existing(jobs, runs_dir):
    accepted, skipped = [], []
    validator = None
    for job in jobs:
        path = Path(runs_dir) / job["run_id"]
        if not path.exists():
            accepted.append(job)
            continue
        validator = validator or validator_module()
        verdict = validator.validate_run(path)
        if verdict["ok"]:
            skipped.append({"id": job.get("id"), "run_id": job["run_id"],
                            "reason": "existing run passes strict validation", "stop_reason": verdict["stop_reason"]})
        else:
            raise FileExistsError(f"Refusing to overwrite existing invalid/incomplete run {job['run_id']}; "
                                  "the orchestrator must resolve this directory before regeneration")
    return accepted, skipped


def pack(jobs):
    lanes, estimated, budgets = [[] for _ in range(8)], [0.0] * 8, [0.0] * 8
    # Keep the final Gowalla BPR seeds together as a feasible 300-minute cap+reserve
    # group. Other jobs are single placement units; execute in priority order later.
    seed_group = [j for j in jobs if j["run_id"] in FINAL_BPR_SEEDS]
    units = [[j] for j in jobs if j["run_id"] not in FINAL_BPR_SEEDS]
    if seed_group:
        units.append(seed_group)
    def totals(unit):
        return sum(j["estimated_minutes"] for j in unit), sum(j["budget_minutes"] for j in unit)
    def is_amazon_dau(job):
        return job["run_id"].startswith("amazonbook__") and "__directau__" in job["run_id"]
    for unit in sorted(units, key=lambda u: (-totals(u)[1], -totals(u)[0], min(j["priority"] for j in u))):
        unit_estimate, unit_budget = totals(unit)
        choices = [i for i in range(8)
                   if estimated[i] + unit_estimate <= LANE_ESTIMATE_LIMIT_MINUTES
                   and budgets[i] + unit_budget <= LANE_BUDGET_LIMIT_MINUTES
                   and sum(is_amazon_dau(j) for j in lanes[i] + unit) <= 1]
        if not choices:
            raise ValueError(f"Cannot place {[j['run_id'] for j in unit]} within the per-GPU estimate/budget limits")
        # Favor cap-budget balance so short expected BPR jobs cannot hide a long worst-case list.
        i = min(choices, key=lambda i: (budgets[i], estimated[i], i))
        lanes[i].extend(unit)
        estimated[i] += unit_estimate
        budgets[i] += unit_budget
    sessions = []
    for lane in lanes:
        lane.sort(key=lambda j: j["priority"])
    for i, letter in enumerate("abcd"):
        sessions.append({"slug": f"pc-s3{letter}-1017", "account": "account-3" if i < 2 else "account-4",
                         "kernel_sources": [], "gpu_lists": lanes[2*i:2*i+2]})
    return sessions


def build_stage3(runs_dir=RUNS):
    candidates = planned_jobs()
    jobs, skipped = filter_existing(candidates, runs_dir)
    _, reason = c3_support()
    sessions = pack(jobs)
    lanes = [{"session": s["slug"], "account": s["account"], "gpu": g,
              "jobs": [j["run_id"] for j in lane],
              "estimated_minutes": sum(j["estimated_minutes"] for j in lane),
              "budget_minutes": sum(j["budget_minutes"] for j in lane)}
             for s in sessions for g, lane in enumerate(s["gpu_lists"])]
    report = {"scope": "OFFLINE preparation only; no Kaggle calls or submissions",
              "new_jobs": len(jobs), "requested_potential_jobs": len(candidates) + 1,
              "skipped_valid": skipped, "omitted": [{"id": "C3", "reason": reason}], "gpu_lists": lanes,
              "overhead_reserve_per_job_minutes": OVERHEAD_MINUTES,
              "estimate_limit_minutes": LANE_ESTIMATE_LIMIT_MINUTES,
              "budget_limit_minutes": LANE_BUDGET_LIMIT_MINUTES,
              "rulings": ["Use the existing non-Git shared workspace; no repository migration or new dependency.",
                          "Follow the exact human-approved hand-off roster; do not silently replace it with the earlier advisory core-only list.",
                          "D7 weight decays 0 and 1e-4 are the hand-off's explicit interventions, not accidental deviations from PLAN 7.2.",
                          "The five authorized AmazonBook DirectAU jobs use per-job cap300 and max150; assume the full cap plus reserve, without asserting convergence.",
                          "Final Gowalla BPR seeds use s0-p500/s1/s2 with identical patience500; legacy patience100 seed0 remains a pilot.",
                          "A2 uses plain Full Xavier initialization; --init-match is only for the compressed I-runs.",
                          "C3 is omitted rather than substituting the scientifically different old dau_bias implementation."],
              "prereg_edited_by_builder": False, "submissions": 0}
    return sessions, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS)
    parser.add_argument("--out", type=Path, default=HERE / "sessions_s3.py")
    parser.add_argument("--report", type=Path, default=PROJECT / "results" / "stage3_build_manifest.json")
    args = parser.parse_args(argv)
    sessions, report = build_stage3(args.runs_dir)
    for name, path in [("plan", PROJECT / "PLAN.md"), ("prereg", PROJECT / "PREREG.md"),
                       ("runner", HERE / "exp_run.py")]:
        report[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.out.write_text("# Generated offline by make_sessions_s3.py. Review/register before any submission.\n"
                        + "SESSIONS = " + pprint.pformat(sessions, sort_dicts=False, width=120) + "\n", encoding="utf8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(f"Prepared {len(sessions)} sessions / {report['new_jobs']} new jobs; submitted 0.")
    for lane in report["gpu_lists"]:
        print(f"{lane['session']} {lane['account']} GPU{lane['gpu']}: {len(lane['jobs'])} jobs, "
              f"estimate {lane['estimated_minutes']:.0f} min, cap+reserve {lane['budget_minutes']:.0f} min")
    print("Skipped valid runs:", [x["run_id"] for x in report["skipped_valid"]])
    print("Omitted C3:", report["omitted"][0]["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
