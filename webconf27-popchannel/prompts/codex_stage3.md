# Prompt for Codex: Stage 3 preparation + post-Stage-2 analyses
(Paste everything below the line into Codex. This file is a hand-off note, not one of the 4 project docs.)

---

You are joining a research project as **engineer / analyst** (see the roles table in TASKS.md Part 1). The project is a WebConf 2027 paper. Its claim: embedding sharing (GraphHash hashing) erases item-level popularity information in recommenders, and `score = cos(u,i) + β·log(1+count_i)` repairs it. This also explains the published flip: DirectAU beats BPR on full tables but loses under GraphHash. The abstract is due **Mon 19 Oct 2026, 17:29 IST**, and the abstract numbers freeze **Sun 18 Oct 12:00 IST**, so Stage 3 must run overnight tonight.

## 0. Read first (in this order, before writing anything)
Project root: `D:\web_2\webconf27-popchannel\`
1. `TASKS.md` Part 1 (rules 1–13 are binding), then the board.
2. `PLAN.md` §1–§5 and §7 (setup: §7.2 lr/wd table, §7.3 stopping rules, §7.4 Kaggle GPU T4 x2), then §8 (run contract) and §9 (gates).
3. `PREREG.md`: the Stage 3 intentions (PR-B, PR-E8, PR-E5, PR-E7, PR-E9, PR-C3, PR-C1, PR-C4, PR-D7).
4. `RESULTS.md`: §0 outcome board and the blocks for PR-A1, PR-I1, PR-Y, PR-H (the latest numbers).
5. The last 6 entries of the PLAN §11 LOG.

Code lives in `D:\web_2\experiments\2026-10-16-fast-check\`:
- `exp_run.py`: training, run-dir contract.
- `job_queue.py` / `kapi.py`: Kaggle session runner.
- `make_sessions_s12.py`: the session-builder **template to copy**.
- `score_controls.py`, `make_maps.py`, `poponly.py`.
- Tests: `test_exp.py`, `test_t1.py`, `test_t3t4.py`.

GraphHash and the datasets live in `D:\web_2\experiments\2026-10-16-loss-sharing-repro\_work\`. Run outputs are in `D:\web_2\webconf27-popchannel\runs\<run_id>\`.

## 1. Hard limits (violating any of these is a failed task)
- **Do NOT push, stop or restart any Kaggle kernel.** Do not run `job_queue.py` without `--dry-run`. Only the orchestrator (named in `scripts/kaggle/ORCHESTRATOR.lock`) pushes.
- **Do NOT edit existing PREREG.md entries.** You draft predictions in a separate file (Task A2). The orchestrator copies them into PREREG after the human approves.
- Never print, log or commit Kaggle usernames, keys or `kaggle.json`. Credential folders: `C:\Users\PUSHKAR\.kaggle-accounts\account-3` and `account-4`. **account-2 is invalid.**
- Never recompute a GraphHash mapping on a CPU image. Scoring of legacy checkpoints happens on the GPU image only.
- **Validation-only selection**: β and checkpoints are chosen on validation, and test is scored once.
- Don't change lr/wd away from PLAN §7.2 without a LOG entry and the human's OK.
- When finished, append **one** LOG line at the end of PLAN.md §11, in this format:
  `YYYY-MM-DD HH:MM IST | codex | IDs | what changed | issues | next`
- Update your rows in TASKS.md.

## 2. Task A: Stage 3 job specs (highest priority; needed tonight)

### A1. Write `make_sessions_s3.py`
Copy the structure of `make_sessions_s12.py` (`train()` helper, `DAU` / `BPR` stopping strings, `SESSIONS` list) into `D:\web_2\experiments\2026-10-16-fast-check\make_sessions_s3.py`.

**Layout:**
- 4 sessions: 2 on `account-3`, 2 on `account-4`.
- Each session has 2 `gpu_lists`, so 8 GPUs in total.
- Each GPU list is run sequentially.
- Balance so that every GPU list finishes in **≤ 9 h** of estimated wall time. Use these measured times (training to patience on a T4):
  - Gowalla GraphHash DirectAU: 38 min.
  - Yelp Full DirectAU: 136 min.
  - Yelp GraphHash DirectAU: 58 min.
  - BPR runs: 4–11 min.
  - For AmazonBook (≈ 2–3× Gowalla), assume the 180-min cap is reached and say so.

**Run IDs** follow `<dataset>__mf__<compression>__<loss>__s<seed>`, with loss ∈ {directau, bpr} and compression ∈ {full, graphhash, random, dfreq}. Use exactly these jobs, in this priority order (put higher priorities first on each GPU list):

| Prio | ID | run_id(s) | exp_run.py args (plus the shared stopping string) |
|---|---|---|---|
| 1 | A2 | `gowalla__mf__full__directau__s0` | `--dataset Gowalla --method Full --epochs 200 --lr 0.01 --wd 1e-8`. This is the controlled partner of I1 (same stopping rule as `pc-s2a-1017`). **Check** whether I1 used `--init-match` semantics that A2 must mirror; if so, write the question in the LOG and keep the plain run |
| 1b | A8r | `gowalla__mf__full__bpr__s0-long` | BPR Gowalla Full, lr 5e-3, wd 1e-8, `--epochs 20000`. The previous run stopped at max epochs 3000 after 10 min, so it is not converged |
| 2 | B9 | `amazonbook__mf__graphhash__directau__s0` | lr 1e-3, wd 1e-8. Pick `--epochs` from the legacy AmazonBook curve (`results/master_metrics.csv` / legacy rows) and justify it in a comment |
| 3 | B5–B8 | `gowalla__mf__graphhash__{directau,bpr}__s{1,2}` | DirectAU lr 1e-2 / wd 1e-8, `--epochs 200`; BPR lr 1e-2 / wd 1e-6 |
| 4 | Main grid | random / dfreq × {bpr, directau} × {Gowalla, Yelp2018, AmazonBook}, s0, plus `amazonbook__mf__full__bpr__s0` | lr/wd from the PLAN §7.2 table, **cell by cell**. Skip any run_id that already exists and passes `scripts/validate_run.py`; list the skipped ones |
| 5 | I2 + partner | `amazonbook__mf__graphhash__directau__s0-initmatch` (`--init-match`) and `amazonbook__mf__full__directau__s0` | lr 1e-3, wd 1e-8, same epochs as B9 |
| 6 | C3 | `gowalla__mf__graphhash__directau_resbias__s0` | Only if `exp_run.py` already supports a regularized residual bias (check `--loss dau_bias` and what it does). If it doesn't, do **not** add the job: write in the LOG what code change is needed |
| + | D7 | `gowalla__mf__full__directau__s0-wd0`, `…-wd1e-4` | Same as A2, with `--wd 0` and `--wd 1e-4` |

Use these stopping strings, exactly as in `make_sessions_s12.py`:
- DirectAU: `--loss dau --eval-every 5 --patience 30 --min-delta 0.0005 --cap-min 180`.
- BPR: `--loss bpr_author --eval-every 10 --patience 500 --min-delta 0 --cap-min 90`, with `--epochs 3000` unless stated otherwise.

Session slugs: `pc-s3a-1017` … `pc-s3d-1017`.

### A2. Draft the Stage 3 predictions (do not edit PREREG.md)
Write `D:\web_2\webconf27-popchannel\prompts\stage3_prereg_draft.md`:
- one entry per job group above, in the same format as the existing PREREG entries;
- each with a numeric prediction, a label [P] / [I] / [H] / [X], and a decision rule;
- based on the registered intentions PR-B (B-i … B-iv), PR-E8, PR-D7 and PR-C3, plus the measured Stage 1–2 numbers in RESULTS.md.

Mark every threshold you derived from an observed number as [I], not [P]. **Note on Y3:** Yelp failed Y3 because its buckets keep more popularity (η² 0.48), so make the predictions for Random/DF depend on η² from PR-M1 where relevant.

### A3. Dry run + checks
- `python make_sessions_s3.py`, then `python job_queue.py --sessions sessions_s3.py --dry-run`. Paste the dry-run summary (no URLs, no usernames) into your final message.
- Assert in a small test (`test_s3.py`):
  - run_ids are unique across all sessions;
  - no run_id collides with an existing valid `runs/` dir;
  - every lr/wd matches PLAN §7.2;
  - every GPU list's estimated time is ≤ 9 h.
- Run all existing tests (`test_exp.py`, `test_t1.py`, `test_t3t4.py`, `scripts/test_t5.py`); they must still pass.

## 3. Task B: analyses on finished runs (do after Task A; all local, no Kaggle)
- **B1 (E1 / T5 refresh):** run `scripts/aggregate.py` so `results/master_metrics.csv` includes all Stage 1–2 runs and controls. Regenerate the RESULTS.md §1–§2 tables from it. Fix the known rounding issue: Full Gowalla cosine must read **13.00** and the gap **4.90**. Mark each row "controlled" or "reproduction".
- **B2 (E7, PR-E7):** uint8-quantized prior check. Quantize log(1+c) to 256 levels and re-score the Stage 1 GraphHash checkpoints. Report |cos_prior_u8 − cos_prior| on validation and test. The "one byte per item" claim is allowed only if ≤ 0.02 on every checkpoint.
- **B3 (E5, PR-E5):** who gains, using `per_user_test__*.npz`. Report what share of the added hits falls in the top-20%-popularity items, the change in bottom-80% recall, and user activity quartiles. Cover Gowalla A1, Yelp A4, Yelp A5 and AmazonBook.
- **B4 (E9, PR-E9):** popularity-reweighted recall with w_i = (1+c_i)^(−1/2), at both the original β and a β re-tuned on validation under the weighted metric. Add the denominator note (PLAN §5).
- Write each outcome as a block appended to RESULTS.md §5, in the existing format: prediction (copied), observed, verdict, runs, notes. Mark PR-E7/E5/E9 as "partial (Stage 1–2 runs)" until Stage 3 lands.

## 4. When done, reply with
1. The files created or changed.
2. The dry-run summary and the per-GPU time estimates.
3. Every open question or deviation (also in the LOG).
4. Test results.
5. The B1–B4 headline numbers.

Do not claim success for anything you did not run.
