# TASKS: agent rules first, then the task board (v3, Sat 17 Oct 17:30 IST)

## Part 1. Rules for every agent (Codex, Claude Code, subagents) and humans. Read before anything else

**The project.** A WebConf 2027 paper on **what embedding sharing destroys** in recommenders, and the few bytes per item that repair it (`cos + β·log(1+count)`). It also explains the published DirectAU-vs-BPR flip under GraphHash. **Prior work we build on (not our claim):** PRISM (KDD 2025: DirectAU lengths track popularity) and DNA (2026: normalize + item bias). Deadlines: abstract **Mon 19 Oct 17:29 IST**, paper **Mon 26 Oct 17:29 IST**.

**The only 4 docs** (project root `D:\web_2\webconf27-popchannel\`):
| Doc | Use | Edit rule |
|---|---|---|
| `PLAN.md` | What, why and how; theory (§4, with the status of each statement); stages (§5); statistics (§6); setup (§7); gates (§9); **LOG (§11)** | Plan text changes only with the human's OK. **The LOG is append-only and is the only log** |
| `TASKS.md` | Rules + board | Claim and update your rows |
| `PREREG.md` | Predictions, labelled [P] prospective / [I] informed, written **before** launch | Append only. Snapshot before each stage push |
| `RESULTS.md` | Outcomes per ID | Append outcomes; tables come from `results/master_metrics.csv` |

**Read order:** Part 1 → PLAN §1–§5 → your task's section → its PREREG entry.

**Roles:**
| Role | Who | May | Must not |
|---|---|---|---|
| Orchestrator | **One** Claude Code session (name + time in `scripts/kaggle/ORCHESTRATOR.lock`) | Push, stop or restart Kaggle sessions; edit job specs and `queue_state.json` | Launch anything without a PREREG entry and snapshot |
| Engineer / analyst | Codex or a Claude subagent | Code, analyses, figures, novelty reading, LaTeX; propose jobs in `configs/experiments/proposed/` | Push Kaggle jobs; edit PREREG entries; change hyperparameters without a LOG entry and the human's OK |
| Subagents | Haiku / Sonnet under the orchestrator | Mechanical work with a precise brief | Decide gates or framing |
| Human | Pushkar (+ co-author) | Gates, quota checks, submissions, writing | — |

**Non-negotiable rules:**
1. **Pre-register** each prediction and decision rule in PREREG.md, labelled [P] or [I], and **snapshot** PREREG before the stage's first push (sha256 in the LOG).
2. **Validation-only selection** of checkpoints and β. **Test is scored once** per final configuration, and once per control.
3. **One condition per GPU process.** Kaggle sessions are **GPU T4 x2**: up to two processes per session (`CUDA_VISIBLE_DEVICES=0/1`), each with its own `runs/<run_id>/`.
4. **Never recompute a hash mapping on the CPU image.**
   - New runs save `mapping_*.npy`.
   - Legacy checkpoints without stored mappings are scored **on the GPU image only**, with an equality assert against the stored aliases of P4 (AmazonBook GraphHash) and P5 (Gowalla GraphHash).
5. **lr/wd** come from PLAN §7.2. **Controlled** comparisons (same lr/wd/batch/budget/init; the mapping is the only difference) are labelled separately from **author-config reproductions**.
6. **Strict validation** (PLAN §8): exit code 0, complete `metrics.json` with all expected modes, checkpoint + mappings present, sha256 matches, stop_reason set. The existence of `metrics.json` alone is not success.
7. **Flag** every capped or short run. **No rescue runs**: anything added after a failed gate is `exploratory` (prefix `x__`).
8. **Report negative results.** Invalid runs move to `runs/_invalid/` with a reason.
9. **Theory statements keep their status** (Theorem / Identity / Hypothesis / Heuristic, PLAN §4). Never present a hypothesis or heuristic as proven.
10. **Secrets:** never print or commit Kaggle usernames, keys or `kaggle.json`. Credentials live in `C:\Users\PUSHKAR\.kaggle-accounts\account-3` and `account-4` (account-2 invalid).
11. Every paper number appears in `results/master_metrics.csv`.
12. **Log in PLAN.md §11** when you finish: `YYYY-MM-DD HH:MM IST | agent | IDs | what changed | issues | next`.
13. If a rule or number looks wrong, write the evidence in the LOG and ask the human. Do not silently deviate.

**Kaggle facts:**
- T4 x2 per session; 2 sessions/account; 2 main accounts (account-3, account-4) → 8 GPUs.
- **account-5** (added 18 Oct; about 30 h of quota then) is *pinned-only*: used only by sessions that name it (`PINNED_ONLY_ACCOUNTS` in `job_queue.py`). account-3 / account-4 had about 6 h / 8 h left on 18 Oct evening.
- **One job on both T4s:** a job may set `"gpus": "0,1"` (one GPU list per session); LightGCN then needs `--multi-gpu` (row-split propagation across cuda:0 and cuda:1).
- **Cancelling** a kernel is done by the human in the Kaggle web UI; this kaggle client has no cancel call. A queued or running kernel holds one of its account's 2 slots.
- Quota ≈ 30 h/week/account [Guessing], per session-hour [Likely]; the human checks it in the web UI.
- kaggle 1.7.4.5 sets only `enable_gpu`, so **A0 verifies `device_count == 2`**.
- Set `KAGGLE_CONFIG_DIR` **before** `import kaggle` (one API call per subprocess, `kapi.py`).
- A kernel can mount only **its own account's** outputs, and never outputs of ERROR'd kernels. Otherwise pass signed URLs fetched at push time (they expire in hours).
- Google Drive may 500 (downloads retry).
- Never name files after stdlib modules.

**Naming:**
- Run ID `{dataset}__{backbone}__{compression}__{loss}__s{seed}`, plus `-initmatch` / `-r{res}` / `-long` / `-p500` / `-wd{x}` / `-2gpu-v2` suffixes; `x__` = exploratory.
- dataset ∈ {gowalla, yelp2018, amazonbook}; backbone ∈ {mf, lightgcn}; compression ∈ {full, random, dfreq, graphhash, graphhash-r{res}, doublegraph}; loss ∈ {bpr, directau, directau_resbias, directau_eqnorm}.
- Session slug `pc-{stage}{n}-{MMDD}`, e.g. `pc-s4a-1018`.

**Status values:** `todo`, `doing (HH:MM, owner)`, `done (HH:MM)`, `blocked (reason)`, `cut (reason)`.

---

## Part 2. Task board

### Code: minimal, next to the validated runner in `D:\web_2\experiments\2026-10-16-fast-check\`
| ID | Task (exact) | Owner | Est. | Acceptance test | Status |
|---|---|---|---|---|---|
| T1 | Patch `exp_run.py` (+ `repro_run.py`) **in place**: (a) extract Yelp2018 splits; (b) `--min-delta` (patience counts gains ≥ min_delta; the checkpoint saves on any gain); (c) each eval appends to `norm_log.csv` (popularity-quintile mean/median item length; ρ_item, ρ_bucket, ρ_dir); (d) save `mapping_{users,items}.npy`; (e) `per_user_test__{mode}.npz`; (f) mean train loss per epoch → `train_log.csv`; (g) `config.json` with `device_count`, `CUDA_VISIBLE_DEVICES`, mapping sha256; (h) `exit_status.json`; (i) `--resolution`; (j) `--init-match` (after Xavier, scale each table so its RMS row norm equals the Full table's: items × √((B_I+64)/(N_I+64)), users likewise; Gowalla 0.378 / 0.442); (k) `--equal-item-norm` (optional C4) | sub (Sonnet) → claude review | 50 min | `test_exp.py` passes; new unit tests for (b), (c), (j); diff shows no change to metric code | doing (18:00, sub-sonnet) |
| T2 | T4 x2 kernel entry in `job_queue.py`: a session = two GPU lists run in parallel, each list sequential; per-process `stdout.log`; per-process exit codes recorded | sub → claude review | 25 min | Dry run builds a 2-list kernel folder | doing (18:00, sub-sonnet) |
| T3 | `score_controls.py`. **Input:** a checkpoint + mapping (stored, or recomputed **on the GPU image** with the P4/P5 equality assert). **Scoring:** compute user × **bucket** scores and copy them to items, so ties are exact. Rank by a **lexicographic sort** (score, then a secondary key); never add a tiny β·p. **Modes:** (1) tie keys: random (10 seeds), default index, popularity. (2) 2×2 at **fixed β\*** (the full prior's val-tuned β): base, +β\*μ, +β\*ε, +β\*p, giving main effects, interaction and Shapley shares. (3) The same cells re-tuned per cell (secondary). (4) `within_bucket_shuffle` at β\* (10 seeds). (5) raw_prior. (6) E2e length tests (`len_shuffle_decile`, `len_shuffle_global`, `len_replace_fit`; 10 seeds each). (7) Tie statistics. Test is scored once per cell. **Outputs:** `controls.json` + per-user npz. Inputs via `kernel_sources` (same account) or signed URLs | sub (Sonnet) → claude review | 70 min | (a) On the Gowalla GraphHash s2 legacy checkpoint, cosine (default key) 7.55 and cos_prior 13.30 (±0.01); P5 cos_prior 13.17. (b) Unit test: the lexicographic tie-break never reorders items with unequal scores. (c) Unit test: Shapley shares + interaction reconstruct Δ_both exactly | done (18:45) |
| T4 | `make_maps.py` (C0/M1): maps for {gowalla, yelp2018, amazonbook} × {graphhash r200, random, dfreq} (+ gowalla graphhash r400/800/1600/3200 for S3). Per map: items/bucket distribution, η² for rank(c) and p by **alias tuple**, Var(μ)/Var(p), ρ_proxy, sha256 → `maps_summary.json` + `.npy` files | sub | 25 min | Gowalla r200 bucket counts 5,792 / 5,806; the maps equal the P5/P4 stored aliases | done (18:45) |
| T5 | `validate_run.py` (strict rules, PLAN §8; also sets `converged = (stop_reason == "patience")`, so capped runs can never pass a convergence gate) + `aggregate.py` → `results/master_metrics.csv` (columns: run_id, dataset, compression, resolution, loss, seed, init, scoring_mode, beta, val/test recall20 + ndcg20, test_std_recall20, rho_item, rho_bucket, rho_dir, val_gap, test_gap, best_epoch, epochs_run, stop_reason, train_min, table_params, extra_bytes, mapping_sha256, session_slug, comparison_type (controlled \| reproduction), status) | codex | 45 min | Reproduces PLAN §2 from the legacy JSONs | done (18:35; refreshed 22:01: 298 rows, nested controls and legacy lineage fixed, 83 T5/regression tests pass) |
| T6 | `poponly.py --dataset D` | sub (Haiku) | 10 min | Gowalla 2.97 / 1.70 | done (18:10) |

### Smoke (G0)
| Session | GPU 0 | GPU 1 | Est. | Status |
|---|---|---|---|---|
| A0 | `x__gowalla__mf__graphhash__directau__s0` (2 epochs, with `--init-match` off), then a co-location test (2 processes on GPU 0, 1 epoch each) | `x__yelp2018__mf__graphhash__bpr__s0` (20 epochs) | 15 min | done (18:35) |

### Stage 1: no training (push with Stage 2 after G0). PLAN §5
| Session | GPU 0 list | GPU 1 list | Status |
|---|---|---|---|
| X1 (account-3) | `score_controls` on: Gowalla GraphHash DirectAU **s0** (gh-exp-p2-gh-dau-s0-1017), Gowalla GraphHash BPR (graphhash-fast-bpr-author-1016), Gowalla Random DirectAU (gh-exp-p3-random-dau-1017), **Full DirectAU Gowalla** (repro kernel, signed URL; E2e), **Full DirectAU AmazonBook** (gh-exp-p4-amz-full-dau-1017; E2e) | **C0** `make_maps` (all datasets) → **A5** `yelp2018__mf__graphhash__bpr__s0` → **A7** `yelp2018__mf__full__bpr__s0` → **A8** `gowalla__mf__full__bpr__s0` | done (19:08) |
| X2 (account-4) | `score_controls` on: Gowalla GraphHash DirectAU **s1** (gh-exp-p2-gh-dau-s1-1017), **s2** (graphhash-fast-dau-s012-1016) | `score_controls` on: Gowalla DF DirectAU (gh-exp-p3-dfreq-dau-1017), AmazonBook GraphHash DirectAU (gh-exp-p4-amz-gh-dau-1017, has aliases), AmazonBook GraphHash BPR (gh-exp-p4-amz-gh-bpr-long-1017, has aliases) | done (19:09) |
| local | **A6** `poponly.py` for yelp2018 and amazonbook | — | done (18:10) |
| reading | **N1**: PRISM full text incl. appendix (setup: optimizer, lr, wd, epochs; DirectAU results without weight decay), DNA, papers citing GraphHash / PRISM / DNA → RESULTS §N1 | codex or human | done (first pass 17 Oct 20:00; deep sweep 18 Oct 22:06, results/N1_novelty.md §d–g; full texts verified 22:56, §h; G1.d awaits human) |
| model-free | **M1**: from `maps_summary.json`, η², Var(μ)/Var(p), items/bucket per mapping; theorem check against legacy ρ's | codex | done (19:40) |

### Stage 2: small training (push with Stage 1 after G0; caps ≤ 3 h). PLAN §5
| Session | GPU 0 | GPU 1 | Est. | Status |
|---|---|---|---|---|
| S2a (account-3) | **A1** `gowalla__mf__graphhash__directau__s0` | **I1** `gowalla__mf__graphhash__directau__s0-initmatch` (`--init-match`; an init-sensitivity control, not a sharing test) | ≤ 3 h | done (20:05, both patience) |
| S2b (account-4) | **A3** `yelp2018__mf__full__directau__s0` | **A4** `yelp2018__mf__graphhash__directau__s0` | ≤ 3 h | done (21:15, both patience) |
(A5, A7, A8 and C0 run in X1's GPU 1 list.)

### Gates
| ID | What | Owner | When | Status |
|---|---|---|---|---|
| Q1 | Kaggle GPU quota on both accounts → LOG | human | Sat ~17:30 | done (18:15) |
| G0 | T1–T4 tests + A0 smoke OK; `device_count`, co-location speed → LOG | claude | Sat ~19:00 | done (18:36) |
| G1 | Mechanism framing + novelty (PLAN §9) | human + claude | Sat ~20:30 | done (19:50, framing confirmed; G1.d pending N1) |
| G2 | Abstract framing + Stage 3 scope (PLAN §9) | human + claude | Sat ~22:30 | done (21:15 all PASS; scope approved; Stage 3 pushed 23:52) |
| G3 | Abstract numbers frozen | human | Sun 12:00 | todo (package prompts/g3_abstract_numbers.md, incl. the orchestrator's revised abstract; **human sign-off pending**) |
| G4 | Paper numbers frozen | human | Fri 23, 18:00 | todo |

### Stage 3: candidates (final list and predictions registered at G2). PLAN §5
Offline preparation, separate from the training statuses below:

| ID | Engineer deliverable | Owner | Status |
|---|---|---|---|
| S3-prep | `make_sessions_s3.py`, `sessions_s3.py`, `test_s3.py`; 25 jobs / 4 sessions / 8 lists; estimate limit 540 min, budget limit 660 min | codex | done (22:38; 12 queue tests and four syntax-valid dry-run bundles; zero submissions) |
| S3-draft | `prompts/stage3_prereg_draft.md`; informed predictions including E5/E7 and final BPR seed-0 protocol | codex | done (22:38; copy block registered verbatim; snapshot `results/snapshots/PREREG_20261017-2238.txt`) |

Registered sessions (all jobs are prepared; only the orchestrator may push):

| Session | Account alias | GPU 0 run IDs, in order | GPU 1 run IDs, in order | Status |
|---|---|---|---|---|
| `pc-s3a-1017` | account-3 | `amazonbook__mf__graphhash__directau__s0` → `gowalla__mf__random__directau__s0` | `amazonbook__mf__random__directau__s0` → `gowalla__mf__dfreq__directau__s0` | done (04:23) |
| `pc-s3b-1017` | account-3 | `gowalla__mf__graphhash__directau__s1` → `amazonbook__mf__dfreq__directau__s0` → `amazonbook__mf__full__bpr__s0` | `gowalla__mf__graphhash__directau__s2` → `amazonbook__mf__graphhash__directau__s0-initmatch` | done (04:23) |
| `pc-s3c-1017` | account-4 | `gowalla__mf__full__bpr__s0-long` → `yelp2018__mf__random__bpr__s0` → `amazonbook__mf__full__directau__s0` | `gowalla__mf__graphhash__bpr__s0-p500` → `gowalla__mf__graphhash__bpr__s1` → `gowalla__mf__graphhash__bpr__s2` → `yelp2018__mf__dfreq__directau__s0` → `amazonbook__mf__dfreq__bpr__s0` | done (04:23) |
| `pc-s3d-1017` | account-4 | `gowalla__mf__full__directau__s0` → `gowalla__mf__dfreq__bpr__s0` → `amazonbook__mf__random__bpr__s0` → `gowalla__mf__full__directau__s0-wd0` | `yelp2018__mf__random__directau__s0` → `gowalla__mf__random__bpr__s0` → `yelp2018__mf__dfreq__bpr__s0` → `gowalla__mf__full__directau__s0-wd1e-4` | done (04:23) |

| Priority | ID | Run | Status |
|---|---|---|---|
| 1 | A2 | `gowalla__mf__full__directau__s0`, matched Full partner of I1 | done (04:23) |
| 1b | A8r | `gowalla__mf__full__bpr__s0-long`, fresh max20000/cap90 run; preserve old max3000 pilot | done (04:23) |
| 2 | B9 | `amazonbook__mf__graphhash__directau__s0`, max150 / cap300; convergence determined by stop_reason | done (04:23) |
| 3 | B5–B8 | Gowalla GraphHash × {DirectAU, BPR} s1/s2, final stopping protocol | done (04:23) |
| 3 | BPR-s0-p500 | `gowalla__mf__graphhash__bpr__s0-p500`; patience500; final BPR seeds s0-p500/s1/s2 | done (04:23) |
| 4 | B1–B4, B10–B18 | Random / DoubleFreq × {BPR, DirectAU} on 3 datasets; AmazonBook Full BPR | done (04:23) |
| 5 | I2 + partner | AmazonBook GraphHash DirectAU `s0-initmatch` and Full DirectAU `s0`, both max150/cap300 | done (04:23) |
| 6 | C3 | `{gowalla,yelp2018,amazonbook}__mf__graphhash__directau_resbias__s0` | done in Stage 4 (code added 18 Oct; **FAIL**: learned score far below post-hoc; code review pending) |
| 7 | C1 | gowalla graphhash-r{400,800,1600,3200} × {bpr, directau} | done in Stage 4 (gain trend PASS; ρ_item trend FAIL; r1600 DirectAU capped) |
| 8 | C4 | Gowalla {full, graphhash} `directau_eqnorm__s0` | done in Stage 4 (as predicted) |
| 9 | D1, D2 | LightGCN (8 runs in Stage 4; Gowalla Full reruns in Stage 5); BACO replaced by the **DoubleGraph substitute** (4 runs in Stage 4; BACO code not available locally) | done in Stage 4; Stage 5 running |
| + | D7 | Gowalla Full DirectAU `s0-wd0` and `s0-wd1e-4`, cap180 | done (04:23) |

### Stage 4: post-G3 experiments for the paper (registered 13:55, snapshot PREREG_20261018-1355; pushed 13:56). PLAN §5
Builder `make_sessions_s4.py` → `sessions_s4.py`; 35 jobs in 4 sessions (pc-s4a/s4b on account-3, pc-s4c/s4d on account-4).

| # | Experiment | Run IDs | Status |
|---|---|---|---|
| 1 | AmazonBook GPU rescoring | `amazonbook__mf__graphhash__bpr__s0-legacy-posts3-controls` | done (21:43; PASS) |
| 2 | Extra seeds | `{yelp2018,amazonbook}__mf__graphhash__{directau,bpr}__s{1,2}` | done (PASS) |
| 3 | C3 residual bias | `*__mf__graphhash__directau_resbias__s0` × 3 | done (FAIL) |
| 4 | LightGCN | `{gowalla,yelp2018}__lightgcn__{full,graphhash}__{bpr,directau}__s0` | done (PASS; 5 rows capped at 4 h) |
| 5 | DoubleGraph | `{gowalla,yelp2018}__mf__doublegraph__{directau,bpr}__s0` | done (PASS) |
| 6 | C1 resolutions | `gowalla__mf__graphhash-r{400,800,1600,3200}__{directau,bpr}__s0` | done (gain PASS, ρ FAIL) |
| 7 | C4 equal norms | `gowalla__mf__{full,graphhash}__directau_eqnorm__s0` | done |
| 8 | AmazonBook Full BPR | `amazonbook__mf__full__bpr__s0-long` | done (patience; PASS) |
| C3-review | Read the `dau_resbias` code: bug or real negative? (no GPU) | claude / codex | todo |

### Stage 5: LightGCN Gowalla Full, one job on both T4s (registered 22:04, amended 22:10; account-5)
| Session | Job | Status |
|---|---|---|
| pc-s5a-1018 | 1-GPU version | cut (cancelled by the human; replaced by the 2-GPU plan) |
| pc-s5b-1018, pc-s5c-1018 | 2-GPU, first try | cut (bug: `"gpus"` dropped from the bundle, so the job saw 1 GPU; pc-s5b failed in 7 s, pc-s5c cancelled) |
| pc-s5d-1018 | `gowalla__lightgcn__full__bpr__s0-2gpu-v2`, `--multi-gpu`, cap 690 min | running (pushed 22:15) |
| pc-s5e-1018 | `gowalla__lightgcn__full__directau__s0-2gpu-v2`, `--multi-gpu`, cap 690 min | running (pushed 22:47) |

### Analyses
| ID | What (PLAN) | Owner | After | Status |
|---|---|---|---|---|
| E-tie | Tie statistics; random / default / popularity secondary keys (§4.3) | codex | X1, X2 | done (19:40) |
| E-dec | 2×2 fixed-β decomposition: main effects, interaction, Shapley shares; re-tuned cells secondary (§4.6) | codex | X1, X2 | done (19:40) |
| E2e | Length tests on the Full models (§5 S1) | codex | X1 | done (19:40) |
| M1 | Mapping statistics + theorem check (§4.5) | codex | C0 | done (19:40) |
| E2b, E2c, E3 | ρ_bucket, ρ_dir, length growth (hypotheses, §4.8) | codex | S2 norm logs | done (21:15) |
| E1 | Final matrix, v1-compatible master and versioned v2 schema; controlled/reproduction and lineage labels | codex | S2/S3 | done (11:49; 398 v1 rows / 401 v2 rows, all 25 S3 runs, 24 matrix cells, final mean/sample SD) |
| E5 | Final tail-primary/standard/contribution and catalog coverage; same-loss Full references | codex | saved NPZ | done (11:49; 30 lineages, 89/90 cells validated; negative direction PASS on all 3 datasets; legacy cosine missing_gpu) |
| E7 | Final uint8 prior checks | codex | final GH checkpoints | done (11:49; 11 modern complete: 3 FAIL / 8 PASS; 1 legacy validation missing; float32 default) |
| E8 | Retained paired bootstrap and paired-t Holm; separate secondary NDCG; applicable Amazon alternatives | codex | final saved pairs | done (11:49; Recall primary PASS and NDCG CIs positive; alternate Amazon paired CIs unavailable) |
| E9 | Final popularity-reweighted original/validation-retuned beta checks | codex | final GH pairs | done (11:49; Gowalla all 3 seeds/Yelp PASS; fully retuned Amazon INCONCLUSIVE, GPU-only legacy cells missing) |

| GPU-postS3 | Missing AmazonBook legacy controls: cosine / quantized validation / weighted retuning / full-prior alternate paired CIs; spec prompts/post_stage3_gpu_rescore.md | orchestrator | reviewed GPU scorer | done in Stage 4 (`score_controls.py --post-s3`; all tie policies CI > 0, E7 PASS, E9 PASS) |

### Writing
| ID | What | Owner | When | Status |
|---|---|---|---|---|
| W0 | `paper/` skeleton (acmart sigconf,anonymous,review) | codex | Sat–Sun | todo |
| W1 | Abstract (G1/G2 framing, real numbers, PRISM/DNA positioned, both authors) | human | Sun 12:00–20:00 | todo |
| W1s | **Submit abstract** | human | Mon 19 by 12:00 | todo |
| W2 | §1–§6 drafts | human + codex | Mon–Thu | todo |
| W3 | 8 pages, anonymity, number check | human | Sat 24 | todo |
| W3s | **Submit paper** | human | Mon 26 by 12:00 | todo |
