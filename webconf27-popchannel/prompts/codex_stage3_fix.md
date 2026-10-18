# Prompt for Codex: apply the Stage 3 review fixes, register, snapshot
(Paste everything below the line into Codex. This is a hand-off note, not one of the 4 project docs.)

---

You built the Stage 3 queue (`make_sessions_s3.py`, `sessions_s3.py`, `test_s3.py`) and the draft `prompts/stage3_prereg_draft.md`. The orchestrator (Claude Code) reviewed them; see the PLAN §11 LOG entry at 2026-10-17 22:10 IST. **The human (Pushkar) approves these fixes and authorizes you, for this task only, to append the Stage 3 entries to PREREG.md and take the snapshot.** All other TASKS.md Part 1 rules still apply:
- no secrets printed;
- account-3 / account-4 only;
- no CPU-image mappings;
- **do not push, stop or restart Kaggle kernels** (Part D below covers pushing).

Code folder: `D:\web_2\experiments\2026-10-16-fast-check\`. Project root: `D:\web_2\webconf27-popchannel\`.

## A. Two queue changes in `make_sessions_s3.py`

### A1. Longer cap for the 5 AmazonBook DirectAU jobs
**Why:** the legacy AmazonBook GraphHash DirectAU run did 41 epochs in a 75-minute cap, about 1.8 min/epoch. A 180-min cap therefore reaches only about 98 epochs. Validation was still rising at epoch 41, and patience is 30, so these runs would very likely be capped. A capped run is INCONCLUSIVE for every convergence gate.

**Change:**
- Use `--cap-min 300` (keep `--epochs 150`; 150 × 1.8 ≈ 275 min fits) for exactly these run_ids:
  - `amazonbook__mf__graphhash__directau__s0` (B9);
  - `amazonbook__mf__graphhash__directau__s0-initmatch` (I2);
  - `amazonbook__mf__full__directau__s0` (I2 partner);
  - `amazonbook__mf__random__directau__s0`;
  - `amazonbook__mf__dfreq__directau__s0`.
- Every other job keeps its current cap (DirectAU 180, BPR 90).
- Implement this as a per-job override, not a change to the shared `DAU` string.
- Set each job's scheduling estimate to the full 300 min.

### A2. Add one BPR seed-0 rerun for a homogeneous three-seed comparison
**Why:** the legacy Gowalla GraphHash BPR s0 used patience 100, while the new s1/s2 (B7/B8) use 500. Without a matching s0, the Gowalla per-seed comparison mixes stopping rules.

**Add the job:**
- run_id `gowalla__mf__graphhash__bpr__s0-p500`;
- args: `--dataset Gowalla --method GraphHash` + the shared `BPR` string + `--epochs 3000 --lr 0.01 --wd 1e-6 --seeds 0`;
- estimate about 10 min;
- place it on the same GPU list as B7/B8 if it fits.

### A3. Rebalance and update the lane rule
- Rebalance the 8 GPU lists so that:
  - each list's **estimated** minutes are ≤ 540 (9 h);
  - each list's **cap + reserve** sum is ≤ 660 (11 h; the Kaggle session limit is 12 h).
- Keep at most one AmazonBook DirectAU job per GPU list where possible.
- Keep priority order inside each list: A2, A8r, B9, B5–B8 first.
- Update `LANE_LIMIT_MINUTES` and `test_s3.py`:
  - split the single 540 limit into separate estimate (540) and budget (660) limits;
  - allow `--cap-min 300` only for the 5 run_ids above (assert this explicitly);
  - assert that the new run_id exists and does not collide with an existing valid `runs/` dir.
- Regenerate `sessions_s3.py` and `results/stage3_build_manifest.json`.
- Run `python job_queue.py --sessions sessions_s3.py --dry-run`.
- Run all tests: `test_s3.py test_exp.py test_t1.py test_t3t4.py` and `D:\web_2\webconf27-popchannel\scripts\test_t5.py`.

## B. Update the draft, then register in PREREG.md
1. Edit `prompts/stage3_prereg_draft.md`:
   - **S3-B9 and S3-I2:** state the cap of 300 min and why (the A1 reasoning above).
   - **S3-B5–B8:** add `gowalla__mf__graphhash__bpr__s0-p500`. The Gowalla seed check (PR-E8: every seed's Δ > 0 with its own CI excluding 0) uses s0-p500 / s1 / s2 for BPR. The legacy s0 (patience 100) is reported as a pilot only.
   - **New entry S3-E5 [I]**, for the final Stage 3 runs. Report tail-recall (bottom-80% items, `tail_primary`) and catalog coverage for:
     - each compressed model's cosine, raw and repaired scoring;
     - the Full model of the same dataset and loss.

     Prediction: repaired GraphHash DirectAU tail-recall < the Full DirectAU raw tail-recall on all 3 datasets (Yelp pilot: 0.29 vs 1.03; see the RESULTS "PR-E5 addendum"). Decision: the paper calls the repair a head-accuracy repair with a tail cost; no "tail-safe" claim.
   - **New entry S3-E7 [I]:** the "one byte per item" claim is **dropped** unless every final GraphHash checkpoint passes |Δ| ≤ 0.02 on both val and test. The default stated cost is float32 (5.52% of the Gowalla GraphHash table). The uint8 version needs 1 extra float32 scale.
   - Keep every threshold's label honest: thresholds taken from observed pilot numbers are [I].
2. Append the final entries to `PREREG.md`:
   - under a new heading `### Stage 3: registered predictions (registered YYYY-MM-DD HH:MM IST, before push)`;
   - copy the entries from the draft verbatim;
   - do **not** edit any earlier PREREG text.
3. **Snapshot**:
   - copy PREREG.md to `results/snapshots/PREREG_<YYYYMMDD-HHMM>.txt`;
   - compute its sha256 (follow the earlier convention: `results/snapshots/PREREG_20261017-1836.txt`).

## C. Records
- **TASKS.md:**
  - add the board rows for Stage 3 sessions `pc-s3a-1017` … `pc-s3d-1017`, each with its jobs, status `ready (registered, not pushed)`;
  - mark C3 as `blocked: needs new code (learned β·p + regularized residual)`.
- **PLAN.md §11 LOG:** append one line in the format `YYYY-MM-DD HH:MM IST | codex | IDs | what changed | issues | next`. It must include:
  - the snapshot path and its sha256;
  - the per-lane estimate and budget minutes;
  - the test counts;
  - "next: orchestrator pushes Stage 3".

## D. Pushing: do NOT do this unless this line is changed to "AUTHORIZED"
NOT AUTHORIZED. The orchestrator (Claude Code, `scripts/kaggle/ORCHESTRATOR.lock`) pushes Stage 3 after checking your snapshot.

## Reply with
1. The files changed.
2. The final lane table: estimate / budget per GPU list.
3. The dry-run summary (no URLs, no usernames).
4. Test counts.
5. The snapshot path and sha256.
6. Any deviation or open question (also written in the LOG).

Do not claim anything you did not run.
