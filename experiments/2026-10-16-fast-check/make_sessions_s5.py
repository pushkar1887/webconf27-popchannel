"""Stage 5 (amended): LightGCN Gowalla Full x {BPR, DirectAU}, each ONE job on BOTH T4s of its own session
(--multi-gpu row-split propagation, CUDA_VISIBLE_DEVICES=0,1), the two sessions running in parallel on account-5.
Cap 690 min (11.5 h). DirectAU max epochs 600, BPR 3000, so patience (not max epochs) can stop them."""
from pathlib import Path
HERE = Path(__file__).parent
LGN = "--model LightGCN --lightgcn-layers 2 --batch 1024 --multi-gpu"
COMMON = "--dataset Gowalla --method Full --lr 0.001 --wd 1e-8 --seeds 0"
JOBS = [
    ("pc-s5d-1018", "gowalla__lightgcn__full__bpr__s0-2gpu-v2",
     f"{COMMON} {LGN} --loss bpr_author --epochs 3000 --eval-every 5 --patience 50 --min-delta 0 --cap-min 690"),
    ("pc-s5e-1018", "gowalla__lightgcn__full__directau__s0-2gpu-v2",
     f"{COMMON} {LGN} --loss dau --epochs 600 --eval-every 5 --patience 30 --min-delta 0.0005 --cap-min 690"),
]
SESSIONS = [{"slug": slug, "account": "account-5", "kernel_sources": [],
             "gpu_lists": [[{"run_id": r, "script": "exp_run.py", "args": f"{a} --run-id {r}", "gpus": "0,1"}]]}
            for slug, r, a in JOBS]
if __name__ == "__main__":
    (HERE / "sessions_s5.py").write_text("SESSIONS = " + repr(SESSIONS) + "\n", encoding="utf-8")
    print("wrote sessions_s5.py:", [r for _, r, _ in JOBS])
