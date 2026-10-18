"""Write sessions_s12.py (Stage 1 scoring X1/X2 + Stage 2 training S2a/S2b) with fresh signed checkpoint URLs.
Run right before pushing: signed URLs expire after a few hours. PLAN §5, TASKS Stage 1/2."""
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent


def url(account, slug, suffix):
    out = subprocess.run([sys.executable, str(HERE / "geturl.py"), account, slug, suffix],
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()
    return out[-1]


U = {
    "gh_dau_s0": url("account-3", "gh-exp-p2-gh-dau-s0-1017", "p2-gh-dau-s0_best_seed0.pt"),
    "gh_dau_s1": url("account-4", "gh-exp-p2-gh-dau-s1-1017", "p2-gh-dau-s1_best_seed1.pt"),
    "gh_dau_s2": url("account-4", "graphhash-fast-dau-s012-1016", "best.pt"),
    "gh_bpr": url("account-3", "graphhash-fast-bpr-author-1016", "best.pt"),
    "rnd_dau": url("account-3", "gh-exp-p3-random-dau-1017", "p3-random-dau_best_seed0.pt"),
    "df_dau": url("account-4", "gh-exp-p3-dfreq-dau-1017", "p3-dfreq-dau_best_seed0.pt"),
    "amz_full_dau": url("account-3", "gh-exp-p4-amz-full-dau-1017", "p4-amz-full-dau_best_seed0.pt"),
    "amz_gh_dau": url("account-4", "gh-exp-p4-amz-gh-dau-1017", "p4-amz-gh-dau_best_seed0.pt"),
    "amz_gh_bpr": url("account-4", "gh-exp-p4-amz-gh-bpr-long-1017", "p4-amz-gh-bpr-long_best_seed0.pt"),
    "p5_gow_gh": url("account-3", "gh-exp-p5-gh-dau-bias-1017", "p5-gh-dau-bias_best_seed0.pt"),
    "gow_full_dau": url("account-3", "graphhash-loss-sharing-repro-1016", "Gowalla_Full_DirectAU_seed0.pt"),
}


def ctl(key, run_id, dataset, method, base, assert_key=None, e2e=False):
    inputs = {key: U[key]}
    args = (f"--ckpt inputs/{key} --dataset {dataset} --method {method} --base {base} --run-id {run_id} "
            f"--seeds 10 --out-root /kaggle/working/controls")
    if assert_key:
        inputs[assert_key] = U[assert_key]
        args += f" --assert-aliases inputs/{assert_key}"
    if e2e:
        args += " --e2e"
    return {"run_id": run_id, "script": "score_controls.py", "args": args, "inputs": inputs}


def train(run_id, args):
    return {"run_id": run_id, "script": "exp_run.py", "args": args + f" --seeds 0 --run-id {run_id}"}


DAU = "--loss dau --eval-every 5 --patience 30 --min-delta 0.0005 --cap-min 180"
BPR = "--loss bpr_author --epochs 3000 --eval-every 10 --patience 500 --min-delta 0 --cap-min 90"

SESSIONS = [
    {"slug": "pc-x1-1017", "account": "account-3", "kernel_sources": [], "gpu_lists": [
        [ctl("gh_dau_s0", "gowalla__mf__graphhash__directau__s0-legacy30", "Gowalla", "GraphHash", "cos", "p5_gow_gh"),
         ctl("gh_bpr", "gowalla__mf__graphhash__bpr__s0-legacy", "Gowalla", "GraphHash", "raw", "p5_gow_gh"),
         ctl("rnd_dau", "gowalla__mf__random__directau__s0-legacy30", "Gowalla", "Random", "cos"),
         ctl("gow_full_dau", "gowalla__mf__full__directau__s0-legacy120", "Gowalla", "Full", "cos", e2e=True),
         ctl("amz_full_dau", "amazonbook__mf__full__directau__s0-legacy", "AmazonBook", "Full", "cos", e2e=True)],
        [{"run_id": "c0_maps", "script": "make_maps.py",
          "args": "--datasets Gowalla,Yelp2018,AmazonBook --out-root /kaggle/working/maps "
                  "--gowalla-resolutions 400,800,1600,3200 "
                  "--assert-aliases gowalla__graphhash=inputs/p5_gow_gh amazonbook__graphhash=inputs/amz_gh_dau",
          "inputs": {"p5_gow_gh": U["p5_gow_gh"], "amz_gh_dau": U["amz_gh_dau"]}},
         train("yelp2018__mf__graphhash__bpr__s0", f"--dataset Yelp2018 --method GraphHash {BPR} --lr 0.005 --wd 1e-6"),
         train("yelp2018__mf__full__bpr__s0", f"--dataset Yelp2018 --method Full {BPR} --lr 0.01 --wd 1e-6"),
         train("gowalla__mf__full__bpr__s0", f"--dataset Gowalla --method Full {BPR} --lr 0.005 --wd 1e-8")],
    ]},
    {"slug": "pc-x2-1017", "account": "account-4", "kernel_sources": [], "gpu_lists": [
        [ctl("gh_dau_s1", "gowalla__mf__graphhash__directau__s1-legacy30", "Gowalla", "GraphHash", "cos", "p5_gow_gh"),
         ctl("gh_dau_s2", "gowalla__mf__graphhash__directau__s2-legacy30", "Gowalla", "GraphHash", "cos", "p5_gow_gh")],
        [ctl("df_dau", "gowalla__mf__dfreq__directau__s0-legacy30", "Gowalla", "DoubleFreq", "cos"),
         ctl("amz_gh_dau", "amazonbook__mf__graphhash__directau__s0-legacy41", "AmazonBook", "GraphHash", "cos"),
         ctl("amz_gh_bpr", "amazonbook__mf__graphhash__bpr__s0-legacy", "AmazonBook", "GraphHash", "raw")],
    ]},
    {"slug": "pc-s2a-1017", "account": "account-3", "kernel_sources": [], "gpu_lists": [
        [train("gowalla__mf__graphhash__directau__s0",
               f"--dataset Gowalla --method GraphHash {DAU} --epochs 200 --lr 0.01 --wd 1e-8")],
        [train("gowalla__mf__graphhash__directau__s0-initmatch",
               f"--dataset Gowalla --method GraphHash {DAU} --epochs 200 --lr 0.01 --wd 1e-8 --init-match")],
    ]},
    {"slug": "pc-s2b-1017", "account": "account-4", "kernel_sources": [], "gpu_lists": [
        [train("yelp2018__mf__full__directau__s0", f"--dataset Yelp2018 --method Full {DAU} --epochs 150 --lr 0.01 --wd 1e-8")],
        [train("yelp2018__mf__graphhash__directau__s0",
               f"--dataset Yelp2018 --method GraphHash {DAU} --epochs 150 --lr 0.005 --wd 1e-6")],
    ]},
]

if __name__ == "__main__":
    body = ("# AUTO-GENERATED by make_sessions_s12.py (contains signed URLs that expire; regenerate before pushing)\n"
            + Path(__file__).read_text().split("SESSIONS = [", 1)[0].split('U = {', 1)[0]
            + "SESSIONS = " + repr(SESSIONS) + "\n")
    (HERE / "sessions_s12.py").write_text("SESSIONS = " + repr(SESSIONS) + "\n", encoding="utf-8")
    print("wrote sessions_s12.py:", [s["slug"] for s in SESSIONS])
