# WebConf 2027: what embedding sharing erases (popchannel)

**Paper:** *If Sharing Saves Memory, What Does It Cost? Popularity Without Identity*

| Milestone | Date | Status |
|---|---|---|
| Abstract | Mon 19 Oct 2026 | submitted |
| G4 numbers freeze | Fri 23 Oct 2026, 18:00 IST | due |
| Paper | Mon 26 Oct 2026, 17:29 IST | due; plan in `paper/OUTLINE.md` |

**Claim:** embedding sharing (GraphHash and other hashing schemes) pools embedding length at bucket level and erases item-level popularity.
- This explains why DirectAU beats BPR on full MF tables but loses under sharing.
- A scoring-time term `β·log(1+count)` is used as a minimal test of the claim, not as the contribution.

## Where things are
| Path | What |
|---|---|
| `webconf27-popchannel/PLAN.md` | Story, theory, stages, statistics, setup, gates; **§11 LOG** (append-only, the only log) |
| `webconf27-popchannel/TASKS.md` | Agent rules (Part 1) and task board (Part 2) |
| `webconf27-popchannel/PREREG.md` | Predictions registered before each push; snapshots in `results/snapshots/` |
| `webconf27-popchannel/RESULTS.md` | Outcome board (§0) and one block per prediction |
| `webconf27-popchannel/results/` | `master_metrics_v2.csv`, E5/E7/E8/E9 JSON, `N1_novelty.md`, queue logs |
| `webconf27-popchannel/runs/` | Per-run directories: config, metrics, logs, per-user scores, mappings |
| `webconf27-popchannel/paper/OUTLINE.md` | Paper plan v5: sections, claim ledger, framing, wording, figures/tables, writing order |
| `webconf27-popchannel/results/ALL_RESULTS.md`, `run_ledger_all.csv` | Consolidated result tables and the full run ledger (source for Appendix C) |
| `research/2026-10-22-paper-writing-audit/` | Writing-pattern audit of 23 WWW'25-era papers (reports only; extracted texts and renders not committed) |
| `webconf27-popchannel/prompts/` | Hand-off prompts and the G3 abstract package |
| `experiments/2026-10-16-fast-check/` | Training (`exp_run.py`), scoring (`score_controls.py`), Kaggle queue (`job_queue.py`), session builders, tests |
| `experiments/2026-10-16-loss-sharing-repro/_work/` | GraphHash reproduction code (datasets not included) |

## Compute
Kaggle GPU T4 x2 sessions, run in Stages 1–5:

| Stage | Jobs |
|---|---|
| 1 | 14 |
| 2 | 6 |
| 3 | 25 |
| 4 | 35 |
| 5 | 2 |

Run IDs follow `{dataset}__{backbone}__{compression}__{loss}__s{seed}`.

## Not in this repo (on purpose)
- Kaggle credentials, kernel folders, and session files with signed URLs.
- Any file containing account usernames.
- Checkpoints (`*.pt`), datasets, and files over 50 MB.
