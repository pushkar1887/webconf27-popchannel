"""Stage 4 sessions (post-G3 experiments, items 1-8 of the 18 Oct list). Writes sessions_s4.py.
Run right before pushing (the rescoring job uses a signed URL that expires). PREREG Stage 4 must be snapshotted first."""
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent
EST_LIMIT, BUDGET_LIMIT, RESERVE = 540, 700, 10


def url(account, slug, suffix):
    out = subprocess.run([sys.executable, str(HERE / "geturl.py"), account, slug, suffix],
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()
    return out[-1]


def dau(cap):
    return f"--loss dau --eval-every 5 --patience 30 --min-delta 0.0005 --cap-min {cap}"


def bpr(cap, epochs=3000):
    return f"--loss bpr_author --epochs {epochs} --eval-every 10 --patience 500 --min-delta 0 --cap-min {cap}"


def resb(cap):
    return f"--loss dau_resbias --eval-every 5 --patience 30 --min-delta 0.0005 --cap-min {cap}"


LGN = "--model LightGCN --lightgcn-layers 2 --batch 1024"
LGN_BPR = "--loss bpr_author --epochs 1000 --eval-every 5 --patience 50 --min-delta 0 --cap-min 240"
LGN_DAU = "--loss dau --epochs 200 --eval-every 5 --patience 30 --min-delta 0.0005 --cap-min 240"

# (priority, run_id, args, estimate_min, cap_min)
J = []


def add(prio, run_id, args, est, cap):
    seed = run_id.split("__s")[-1].split("-")[0]
    J.append((prio, run_id, f"{args} --seeds {seed} --run-id {run_id}", est, cap))


# 2. extra seeds (Yelp, AmazonBook GraphHash)
for s in (1, 2):
    add(2, f"yelp2018__mf__graphhash__directau__s{s}", f"--dataset Yelp2018 --method GraphHash {dau(150)} --epochs 150 --lr 0.005 --wd 1e-6", 60, 150)
    add(2, f"yelp2018__mf__graphhash__bpr__s{s}", f"--dataset Yelp2018 --method GraphHash {bpr(30)} --lr 0.005 --wd 1e-6", 6, 30)
    add(2, f"amazonbook__mf__graphhash__directau__s{s}", f"--dataset AmazonBook --method GraphHash {dau(240)} --epochs 150 --lr 0.001 --wd 1e-8", 125, 240)
    add(2, f"amazonbook__mf__graphhash__bpr__s{s}", f"--dataset AmazonBook --method GraphHash {bpr(60)} --lr 0.01 --wd 1e-6", 20, 60)
# 8. AmazonBook Full BPR uncapped
add(8, "amazonbook__mf__full__bpr__s0-long", f"--dataset AmazonBook --method Full {bpr(240, 20000)} --lr 0.01 --wd 1e-8", 180, 240)
# 3. C3 residual bias
add(3, "gowalla__mf__graphhash__directau_resbias__s0", f"--dataset Gowalla --method GraphHash {resb(120)} --epochs 200 --lr 0.01 --wd 1e-8", 45, 120)
add(3, "yelp2018__mf__graphhash__directau_resbias__s0", f"--dataset Yelp2018 --method GraphHash {resb(150)} --epochs 150 --lr 0.005 --wd 1e-6", 65, 150)
add(3, "amazonbook__mf__graphhash__directau_resbias__s0", f"--dataset AmazonBook --method GraphHash {resb(240)} --epochs 150 --lr 0.001 --wd 1e-8", 130, 240)
# 4. LightGCN (authors' lr 1e-3, wd 1e-8, batch 1024 for Full and GraphHash)
for ds in ("Gowalla", "Yelp2018"):
    for m in ("Full", "GraphHash"):
        add(4, f"{ds.lower()}__lightgcn__{m.lower()}__bpr__s0", f"--dataset {ds} --method {m} {LGN} {LGN_BPR} --lr 0.001 --wd 1e-8", 200, 240)
        add(4, f"{ds.lower()}__lightgcn__{m.lower()}__directau__s0", f"--dataset {ds} --method {m} {LGN} {LGN_DAU} --lr 0.001 --wd 1e-8", 200, 240)
# 5. DoubleGraph (structure-aware substitute for BACO; GraphHash lr/wd)
add(5, "gowalla__mf__doublegraph__directau__s0", f"--dataset Gowalla --method DoubleGraph {dau(120)} --epochs 200 --lr 0.01 --wd 1e-8", 45, 120)
add(5, "gowalla__mf__doublegraph__bpr__s0", f"--dataset Gowalla --method DoubleGraph {bpr(30)} --lr 0.01 --wd 1e-6", 6, 30)
add(5, "yelp2018__mf__doublegraph__directau__s0", f"--dataset Yelp2018 --method DoubleGraph {dau(150)} --epochs 150 --lr 0.005 --wd 1e-6", 65, 150)
add(5, "yelp2018__mf__doublegraph__bpr__s0", f"--dataset Yelp2018 --method DoubleGraph {bpr(30)} --lr 0.005 --wd 1e-6", 6, 30)
# 6. C1 Louvain resolution sweep (Gowalla GraphHash)
for r in (400, 800, 1600, 3200):
    add(6, f"gowalla__mf__graphhash-r{r}__directau__s0", f"--dataset Gowalla --method GraphHash --resolution {r} {dau(120)} --epochs 200 --lr 0.01 --wd 1e-8", 45, 120)
    add(6, f"gowalla__mf__graphhash-r{r}__bpr__s0", f"--dataset Gowalla --method GraphHash --resolution {r} {bpr(30)} --lr 0.01 --wd 1e-6", 6, 30)
# 7. C4 equal item norms
add(7, "gowalla__mf__full__directau_eqnorm__s0", f"--dataset Gowalla --method Full {dau(180)} --epochs 200 --lr 0.01 --wd 1e-8 --equal-item-norm", 115, 180)
add(7, "gowalla__mf__graphhash__directau_eqnorm__s0", f"--dataset Gowalla --method GraphHash {dau(120)} --epochs 200 --lr 0.01 --wd 1e-8 --equal-item-norm", 45, 120)


def rescore_job():
    u = url("account-4", "gh-exp-p4-amz-gh-bpr-long-1017", "p4-amz-gh-bpr-long_best_seed0.pt")
    return {"run_id": "amazonbook__mf__graphhash__bpr__s0-legacy-posts3-controls", "script": "score_controls.py",
            "args": "--ckpt inputs/amz_gh_bpr --dataset AmazonBook --method GraphHash --base raw --post-s3 --beta 0.2 "
                    "--run-id amazonbook__mf__graphhash__bpr__s0-legacy-posts3-controls --seeds 10 "
                    "--out-root /kaggle/working/controls",
            "inputs": {"amz_gh_bpr": u}}


def pack(n_lanes=8):
    lanes = [{"jobs": [], "est": 0, "budget": 0} for _ in range(n_lanes)]
    # lane 0 starts with the 30-min rescoring job
    lanes[0]["est"] += 30; lanes[0]["budget"] += 40
    for prio, rid, args, est, cap in sorted(J, key=lambda j: -(j[4])):  # biggest caps first
        lane = min(lanes, key=lambda l: (l["budget"], l["est"]))
        lane["jobs"].append((prio, rid, args)); lane["est"] += est + RESERVE; lane["budget"] += cap + RESERVE
    for l in lanes:
        l["jobs"].sort(key=lambda j: j[0])
        assert l["est"] <= EST_LIMIT and l["budget"] <= BUDGET_LIMIT, l
    return lanes


def build(with_urls=True):
    lanes = pack()
    gl = [[{"run_id": rid, "script": "exp_run.py", "args": args} for _, rid, args in l["jobs"]] for l in lanes]
    if with_urls:
        gl[0].insert(0, rescore_job())
    accts = ["account-3", "account-3", "account-4", "account-4"]
    sessions = [{"slug": f"pc-s4{c}-1018", "account": accts[k], "kernel_sources": [], "gpu_lists": gl[2 * k:2 * k + 2]}
                for k, c in enumerate("abcd")]
    return sessions, lanes


if __name__ == "__main__":
    S, lanes = build(with_urls="--no-urls" not in sys.argv)
    ids = [j["run_id"] for s in S for g in s["gpu_lists"] for j in g]
    assert len(ids) == len(set(ids))
    (HERE / "sessions_s4.py").write_text("SESSIONS = " + repr(S) + "\n", encoding="utf-8")
    for k, l in enumerate(lanes):
        print(f"lane {k}: est {l['est']} min, budget {l['budget']} min, jobs {len(l['jobs'])}")
    print("wrote sessions_s4.py:", len(ids), "jobs")
