# Results compendium: *If Sharing Saves Memory, What Does It Cost? Popularity Lost in Shared Embeddings*

WebConf 2027 submission (due Mon 26 Oct 2026, 17:29 IST). Compiled 22 Oct 2026 from repo commit `5807856`.

This file collects every table and experiment result in the project in one place, in paper order (RQ1–RQ6, then failures, controls and the full run ledger). It does not replace `RESULTS.md` (the registered outcome record); it reorganises it and adds the tables that `RESULTS.md` does not have.

**Contents**
0. [How to read this document](#0-how-to-read-this-document)
1. [Headline results](#1-headline-results)
2. [Setup: datasets, tables, hyperparameters, compute](#2-setup)
3. [RQ1: Does sharing remove DirectAU's advantage?](#3-rq1-does-sharing-remove-directaus-advantage)
4. [RQ2: Is popularity lost at item level? (diagnosis)](#4-rq2-is-popularity-lost-at-item-level)
5. [RQ3: Is the recovered gain item-specific? (correction test)](#5-rq3-is-the-recovered-gain-item-specific)
6. [RQ4: Does erasing popularity without sharing reproduce the symptom?](#6-rq4-does-erasing-popularity-without-sharing-reproduce-the-symptom)
7. [RQ5: Does it generalise?](#7-rq5-does-it-generalise)
8. [RQ6: What does it cost?](#8-rq6-what-does-it-cost)
9. [Negative and failed results](#9-negative-and-failed-results)
10. [Reproduction and control runs](#10-reproduction-and-control-runs)
11. [Pre-registration outcome board](#11-pre-registration-outcome-board)
12. [Novelty verification](#12-novelty-verification)
13. [Issues found while compiling](#13-issues-found-while-compiling)
- [Appendix A: complete run ledger (93 runs)](#appendix-a-complete-run-ledger)
- [Appendix B: tail and coverage for every E5 record (90 cells)](#appendix-b-tail-and-coverage-for-every-e5-record)

---

## 0. How to read this document

**Units.** Recall@20 and NDCG@20 are test-set values in **percent** (authors' denominator `min(20, |T_u|)`), unless a column says *pp* (percentage points). "Δ" is a difference in pp.

**Scoring modes** (the same checkpoint is scored four ways):

| Mode | Score | Notes |
|---|---|---|
| Raw | ⟨p_u, q_i⟩ | the model's own score |
| Cosine | cos(p_u, q_i) | removes both lengths |
| Cos + prior | cos(p_u, q_i) + β·log(1 + c_i) | the corrected score for **DirectAU** |
| Raw + prior | ⟨p_u, q_i⟩ + β·log(1 + c_i) | the corrected score for **BPR** |

c_i is the item's **training** count. β is chosen on validation from a fixed grid, and test is scored once. "Repaired" or "corrected" means Cos + prior for DirectAU and Raw + prior for BPR.

**Correlations.**
- ρ_item = Spearman(‖q_i‖, c_i) over all items.
- ρ_bucket = Spearman(bucket-row norm, Σ c_i over the bucket), over non-empty buckets.
- ρ_dir = Spearman(cos(q_i, mean user direction), c_i).
- All three are taken at the **best-checkpoint epoch**, which is what `master_metrics_v2.csv` does. (`RESULTS.md` prose quotes the *last evaluated* epoch in some places; Table 11 uses the last epoch and says so.)

**Run IDs** follow `{dataset}__{backbone}__{sharing}__{loss}__s{seed}[-variant]`. Sharing is `full` (no sharing), `graphhash` (Louvain, resolution 200 unless `-rNNN`), `random`, `dfreq` (DoubleFreq) or `doublegraph` (the BACO substitute). `x__*` runs are Stage 0/1 smoke and determinism checks.

**Status labels.**
- *converged*: stopped by patience.
- *capped*: stopped by `time_cap` or `max_epochs`. A capped run is **inconclusive for convergence** and is descriptive only.
- *fixed-epoch / valid-short*: legacy 30-epoch runs.
- *invalid*: excluded (Section 9).
- † marks a single-seed row.

**Lineage.** *Modern* runs use patience-based stopping and are fully logged in `runs/`. *Legacy* runs are the Stage 1 runs (fixed epochs or time-capped); they survive only as rows in `master_metrics_v2.csv` and as saved top-20 lists for the control analyses (decomposition, ties, interventions, E7 pilot).

**Where the numbers come from.**

| Source | Covers |
|---|---|
| `runs/*/metrics.json`, `config.json`, `norm_log.csv` | all 73 training runs, Stages 0–5 (the source of truth for the ledger) |
| `results/master_metrics_v2.csv` | Stages 1–3 only: 36 of the 73 modern runs plus the legacy rows (see Section 13, issue 1) |
| `results/E5_final.json`, `E7_final.json`, `E8_final.json`, `E9_final.json` | tail/coverage, uint8 storage, paired inference, weighted recall |
| `runs/_sessions/pc-x*-1017/controls/*/controls.json` | decomposition, ties, shuffle, equal-norm and E2e controls (legacy checkpoints) |
| `runs/_sessions/pc-x1-1017/maps/maps_summary.json` | bucket structure and η² |
| `RESULTS.md` | outcome blocks; tables copied verbatim are marked *(verbatim from RESULTS.md lines a–b)* |

**Checks run while compiling** (details in Section 13):
- Ledger vs `master_metrics_v2.csv`: 144 of 144 overlapping score cells and 93 of 93 ρ cells match.
- Decomposition means over 3 seeds reproduce the PR-DEC block (+1.74 / +4.22 / +5.79 / −0.17, shuffle −0.36).
- All 9 seed-pair Recall CIs match `RESULTS.md` (the 4 Stage 4 pairs were recomputed from per-user files).
- η² table, resolution-sweep Spearman values and E2e numbers match their `RESULTS.md` blocks.

---

## 1. Headline results

| # | Claim | Numbers | Where |
|---|---|---|---|
| 1 | Sharing removes DirectAU's lead on 2 of 3 MF datasets | Gowalla: Full DAU 18.02 > BPR 16.55, GraphHash DAU 7.55 < BPR 9.37. Yelp: 11.10 > 8.02, then 2.41 < 3.78. AmazonBook: 9.57 > 8.38, then 4.63 > 4.10 (lead shrinks +1.19 → +0.53). LightGCN Gowalla Full: BPR already ahead (18.48 vs 18.02) | Tables 6, 8 |
| 2 | Under sharing, length follows the bucket, not the item | GraphHash DAU ρ_bucket / ρ_item: Gowalla 0.925 / 0.275, Yelp 0.944 / 0.520, AmazonBook 0.746 / 0.189. Full DAU ρ_item: 0.815 / 0.912 / 0.786 | Tables 11, 13 |
| 3 | Removing length hurts Full but not shared tables | Δ(cos − raw), Full DAU: −4.96 / −4.13 / −0.87. GraphHash DAU: +0.02 / −0.44 / +0.00 | Table 12 |
| 4 | Putting item counts back repairs shared DirectAU | GraphHash DAU raw → cos+prior: 7.55 → 13.30, 2.41 → 5.68, 4.63 → 6.33. Gowalla 3-seed 13.325 ± 0.024 | Tables 6, 10 |
| 5 | Repaired DAU beats repaired BPR on 9 of 9 dataset × seed pairs | Δ +0.83 to +2.13 pp, every paired 95% CI above 0 | Table 16 |
| 6 | The recovered gain is item-specific | Shapley item-residual (ε) share 0.71 (Gowalla), 0.76 (AmazonBook); within-bucket shuffle −0.36 / −0.17 pp | Table 20 |
| 7 | It is not MostPop | popularity-only 2.97 / 1.68 / 0.96 vs repaired 13.30 / 5.68 / 6.33 | Table 21 |
| 8 | Erasing length on Full tables reproduces the symptom | Gowalla Full DAU 17.90 → 6.28 (global shuffle), 13.64 (decile shuffle); length replaced by f(popularity) 18.69; equal norms 12.77 raw → 19.32 with prior | Tables 23, 15 |
| 9 | It generalises beyond GraphHash and MF | LightGCN GraphHash corrected DAU 17.06 vs BPR 16.27 (Gowalla), 9.12 vs 6.96 (Yelp; DAU run capped). DoubleGraph corrected DAU 10.91 vs 7.27 (Gowalla), 5.52 vs 2.78 (Yelp) | Tables 24, 25 |
| 10 | Finer buckets shrink the gain | Gowalla gain 5.76 → 4.99 as resolution rises 200 → 3200 (Spearman −0.90) | Table 26 |
| 11 | Costs | Tail recall below Full raw by 0.50 / 0.74 / 1.63 pp. Coverage (cosine → cos+prior) 95.2 → 28.4% (Gowalla). float32 prior +5.52 / 3.90 / 7.30% of the table. uint8 fails on 3 of 11 modern checkpoints | Section 8 |
| 12 | What did not work | learned residual variant 4.56 vs post-hoc 13.21 (Gowalla); wd = 1e-4 collapses Full DAU (raw 1.41); 3 pilot tail/storage predictions failed | Section 9 |

---

## 2. Setup

**Table 1. Datasets.** Standard LightGCN / GraphHash splits (80 / 10 / 10 by interaction).

| | Gowalla | Yelp2018 | AmazonBook |
|---|---|---|---|
| Users | 29,858 | 31,668 | 52,643 |
| Items | 40,981 | 38,048 | 91,599 |
| Interactions (total) | 1,027,370 | 1,561,406 | 2,984,108 |
| Train / val / test | 821,896 / 102,737 / 102,737 | 1,249,124 / 156,140 / 156,142 | 2,387,286 / 298,410 / 298,412 |
| Density (total) | 0.084% | 0.130% | 0.062% |
| Evaluated users (val / test) | 25,738 / 25,789 | 30,261 / 30,304 | 50,355 / 50,407 |
| Max item train count | 1,880 | 1,180 | 1,539 |
| Items with zero train count | 0 | 0 | 0 |

*Source: `runs/*/config.json` (`dataset_stats`) and `maps_summary.json`.*

**Table 2. Table sizes under GraphHash.** Resolution 200 is the main setting; the other rows are the resolution sweep (Table 26). Random, DoubleFreq and DoubleGraph use the same row counts as GraphHash-200 on each dataset.

| Dataset | Sharing | User rows | Item rows | Compressed params | Full params | Table ratio vs Full |
|---|---|---|---|---|---|---|
| AmazonBook | GraphHash (resolution 200) | 9,808 | 9,804 | 1,255,168 | 9,231,488 | 13.60% |
| Gowalla | GraphHash (resolution 200) | 5,792 | 5,806 | 742,272 | 4,533,696 | 16.37% |
| Gowalla | GraphHash (resolution 400) | 8,882 | 8,924 | 1,139,584 | 4,533,696 | 25.14% |
| Gowalla | GraphHash (resolution 800) | 12,502 | 12,880 | 1,624,448 | 4,533,696 | 35.83% |
| Gowalla | GraphHash (resolution 1600) | 15,350 | 17,699 | 2,115,136 | 4,533,696 | 46.65% |
| Gowalla | GraphHash (resolution 3200) | 19,216 | 25,993 | 2,893,376 | 4,533,696 | 63.82% |
| Yelp2018 | GraphHash (resolution 200) | 7,604 | 7,656 | 976,640 | 4,461,824 | 21.89% |

The table ratios (Gowalla 16.37%, Yelp2018 21.89%, AmazonBook 13.60%) are computed from the stored `bucket_info` parameter counts, not from byte estimates.

**Table 3. Hyperparameters of the main-grid runs.** Embedding dimension 64 everywhere; LightGCN uses 2 propagation layers. MF BPR runs are full-batch (batch = number of training edges, as in the authors' code); DirectAU and LightGCN use batch 1,024. The AmazonBook GraphHash BPR s0 row is a legacy run with no stored config, so its seed-1 sibling is shown.

| Dataset | Backbone | Sharing | Loss | lr | weight decay | Batch | Patience | min-Δ | Eval every | Max epochs | Wall cap (min) | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gowalla | MF | Full | BPR | 0.005 | 1e-08 | 821,896 | 500 | 0 | 10 | 20000 | 90.0 | `gowalla__mf__full__bpr__s0-long` |
| Gowalla | MF | Full | DirectAU | 0.01 | 1e-08 | 1,024 | 30 | 0.0005 | 5 | 200 | 180.0 | `gowalla__mf__full__directau__s0` |
| Gowalla | MF | GraphHash | BPR | 0.01 | 1e-06 | 821,896 | 500 | 0 | 10 | 3000 | 90.0 | `gowalla__mf__graphhash__bpr__s0-p500` |
| Gowalla | MF | GraphHash | DirectAU | 0.01 | 1e-08 | 1,024 | 30 | 0.0005 | 5 | 200 | 180.0 | `gowalla__mf__graphhash__directau__s0` |
| Gowalla | MF | Random | BPR | 0.001 | 1e-06 | 821,896 | 500 | 0 | 10 | 3000 | 90.0 | `gowalla__mf__random__bpr__s0` |
| Gowalla | MF | Random | DirectAU | 0.01 | 1e-06 | 1,024 | 30 | 0.0005 | 5 | 200 | 180.0 | `gowalla__mf__random__directau__s0` |
| Gowalla | MF | DoubleFreq | BPR | 0.001 | 1e-06 | 821,896 | 500 | 0 | 10 | 3000 | 90.0 | `gowalla__mf__dfreq__bpr__s0` |
| Gowalla | MF | DoubleFreq | DirectAU | 0.005 | 1e-06 | 1,024 | 30 | 0.0005 | 5 | 200 | 180.0 | `gowalla__mf__dfreq__directau__s0` |
| Gowalla | MF | DoubleGraph | BPR | 0.01 | 1e-06 | 821,896 | 500 | 0 | 10 | 3000 | 30.0 | `gowalla__mf__doublegraph__bpr__s0` |
| Gowalla | MF | DoubleGraph | DirectAU | 0.01 | 1e-08 | 1,024 | 30 | 0.0005 | 5 | 200 | 120.0 | `gowalla__mf__doublegraph__directau__s0` |
| Yelp2018 | MF | Full | BPR | 0.01 | 1e-06 | 1,249,124 | 500 | 0 | 10 | 3000 | 90.0 | `yelp2018__mf__full__bpr__s0` |
| Yelp2018 | MF | Full | DirectAU | 0.01 | 1e-08 | 1,024 | 30 | 0.0005 | 5 | 150 | 180.0 | `yelp2018__mf__full__directau__s0` |
| Yelp2018 | MF | GraphHash | BPR | 0.005 | 1e-06 | 1,249,124 | 500 | 0 | 10 | 3000 | 90.0 | `yelp2018__mf__graphhash__bpr__s0` |
| Yelp2018 | MF | GraphHash | DirectAU | 0.005 | 1e-06 | 1,024 | 30 | 0.0005 | 5 | 150 | 180.0 | `yelp2018__mf__graphhash__directau__s0` |
| Yelp2018 | MF | Random | BPR | 0.001 | 1e-06 | 1,249,124 | 500 | 0 | 10 | 3000 | 90.0 | `yelp2018__mf__random__bpr__s0` |
| Yelp2018 | MF | Random | DirectAU | 0.005 | 1e-06 | 1,024 | 30 | 0.0005 | 5 | 150 | 180.0 | `yelp2018__mf__random__directau__s0` |
| Yelp2018 | MF | DoubleFreq | BPR | 0.001 | 1e-06 | 1,249,124 | 500 | 0 | 10 | 3000 | 90.0 | `yelp2018__mf__dfreq__bpr__s0` |
| Yelp2018 | MF | DoubleFreq | DirectAU | 0.001 | 0.0001 | 1,024 | 30 | 0.0005 | 5 | 150 | 180.0 | `yelp2018__mf__dfreq__directau__s0` |
| Yelp2018 | MF | DoubleGraph | BPR | 0.005 | 1e-06 | 1,249,124 | 500 | 0 | 10 | 3000 | 30.0 | `yelp2018__mf__doublegraph__bpr__s0` |
| Yelp2018 | MF | DoubleGraph | DirectAU | 0.005 | 1e-06 | 1,024 | 30 | 0.0005 | 5 | 150 | 150.0 | `yelp2018__mf__doublegraph__directau__s0` |
| AmazonBook | MF | Full | BPR | 0.01 | 1e-08 | 2,387,286 | 500 | 0 | 10 | 20000 | 240.0 | `amazonbook__mf__full__bpr__s0-long` |
| AmazonBook | MF | Full | DirectAU | 0.001 | 1e-08 | 1,024 | 30 | 0.0005 | 5 | 150 | 300.0 | `amazonbook__mf__full__directau__s0` |
| AmazonBook | MF | GraphHash | BPR | 0.01 | 1e-06 | 2,387,286 | 500 | 0 | 10 | 3000 | 60.0 | `amazonbook__mf__graphhash__bpr__s1` (legacy; modern s1 shown) |
| AmazonBook | MF | GraphHash | DirectAU | 0.001 | 1e-08 | 1,024 | 30 | 0.0005 | 5 | 150 | 300.0 | `amazonbook__mf__graphhash__directau__s0` |
| AmazonBook | MF | Random | BPR | 0.005 | 1e-06 | 2,387,286 | 500 | 0 | 10 | 3000 | 90.0 | `amazonbook__mf__random__bpr__s0` |
| AmazonBook | MF | Random | DirectAU | 0.001 | 1e-06 | 1,024 | 30 | 0.0005 | 5 | 150 | 300.0 | `amazonbook__mf__random__directau__s0` |
| AmazonBook | MF | DoubleFreq | BPR | 0.001 | 1e-08 | 2,387,286 | 500 | 0 | 10 | 3000 | 90.0 | `amazonbook__mf__dfreq__bpr__s0` |
| AmazonBook | MF | DoubleFreq | DirectAU | 0.001 | 1e-06 | 1,024 | 30 | 0.0005 | 5 | 150 | 300.0 | `amazonbook__mf__dfreq__directau__s0` |
| Gowalla | LightGCN | Full | BPR | 0.001 | 1e-08 | 1,024 | 50 | 0 | 5 | 3000 | 690.0 | `gowalla__lightgcn__full__bpr__s0-2gpu-v2` |
| Gowalla | LightGCN | Full | DirectAU | 0.001 | 1e-08 | 1,024 | 30 | 0.0005 | 5 | 600 | 690.0 | `gowalla__lightgcn__full__directau__s0-2gpu-v2` |
| Gowalla | LightGCN | GraphHash | BPR | 0.001 | 1e-08 | 1,024 | 50 | 0 | 5 | 1000 | 240.0 | `gowalla__lightgcn__graphhash__bpr__s0` |
| Gowalla | LightGCN | GraphHash | DirectAU | 0.001 | 1e-08 | 1,024 | 30 | 0.0005 | 5 | 200 | 240.0 | `gowalla__lightgcn__graphhash__directau__s0` |
| Yelp2018 | LightGCN | GraphHash | BPR | 0.001 | 1e-08 | 1,024 | 50 | 0 | 5 | 1000 | 240.0 | `yelp2018__lightgcn__graphhash__bpr__s0` |
| Yelp2018 | LightGCN | GraphHash | DirectAU | 0.001 | 1e-08 | 1,024 | 30 | 0.0005 | 5 | 200 | 240.0 | `yelp2018__lightgcn__graphhash__directau__s0` |

**Table 4. Kaggle jobs per stage.**

| Stage | Kaggle jobs |
|---|---|
| 1 | 14 |
| 2 | 6 |
| 3 | 25 |
| 4 | 35 |
| 5 | 2 |

Hardware: Kaggle Tesla T4. Legacy runs used one GPU per session; from Stage 1/2 on, sessions use two T4s with one condition per GPU.

**Table 5. Measured runtimes** *(verbatim from RESULTS.md lines 208–214)*

| Job | Measured |
|---|---|
| DirectAU MF, Gowalla | 37–39 s/epoch |
| DirectAU MF, AmazonBook | 112–119 s/epoch |
| BPR full batch, Gowalla / AmazonBook | 0.22 / 0.41 s/epoch |
| Kaggle job overhead (setup + final eval) | 2–4 min |
| Peak GPU memory, Full × DirectAU Gowalla | 380 MiB |

Stage 4–5 wall times are in Appendix A (train min). Two-GPU row-split of LightGCN gave no speedup: 0.70 min/epoch (BPR) and 1.33 min/epoch (DirectAU), the same as one GPU.

---

## 3. RQ1: Does sharing remove DirectAU's advantage?

**Table 6. Main grid, MF, seed 0, test Recall@20 %.** One row per dataset × sharing × loss. Final reference runs: Gowalla Full BPR is `s0-long` (converged), Gowalla GraphHash BPR is `s0-p500`, AmazonBook Full BPR is `s0-long` (converged at epoch 3800). The AmazonBook GraphHash BPR row is the retained legacy patience-500 checkpoint. † DoubleGraph is the BACO substitute (single seed).

| Dataset | Sharing | Loss | Raw | Cosine | Cos + prior (β) | Raw + prior (β) | ρ_item | ρ_bucket | Run | Stop (best/ran) |
|---|---|---|---|---|---|---|---|---|---|---|
| gowalla | Full | BPR | 16.55 | 11.96 | 16.71 (β 0.05) | 17.06 (β 0.5) | 0.770 | — | `gowalla__mf__full__bpr__s0-long` | patience; 2790/3290 |
| gowalla | Full | DirectAU | 18.02 | 13.06 | 19.57 (β 0.1) | 18.02 (β 0.01) | 0.815 | — | `gowalla__mf__full__directau__s0` | patience; 160/170 |
| gowalla | GraphHash | BPR | 9.37 | 5.86 | 9.96 (β 0.1) | 11.31 (β 0.5) | 0.066 | 0.862 | `gowalla__mf__graphhash__bpr__s0-p500` | patience; 460/960 |
| gowalla | GraphHash | DirectAU | 7.55 | 7.57 | 13.30 (β 0.2) | 10.33 (β 5) | 0.275 | 0.925 | `gowalla__mf__graphhash__directau__s0` | patience; 55/60 |
| gowalla | Random | BPR | 0.76 | 0.38 | 3.44 (β 0.2) | 3.42 (β 0.5) | 0.093 | 0.389 | `gowalla__mf__random__bpr__s0` | patience; 240/740 |
| gowalla | Random | DirectAU | 0.45 | 0.29 | 3.80 (β 0.2) | 3.70 (β 2) | 0.275 | 0.990 | `gowalla__mf__random__directau__s0` | patience; 40/40 |
| gowalla | DoubleFreq | BPR | 3.93 | 2.95 | 4.06 (β 0.1) | 4.11 (β 0.5) | 0.100 | — | `gowalla__mf__dfreq__bpr__s0` | patience; 450/950 |
| gowalla | DoubleFreq | DirectAU | 3.22 | 2.20 | 5.11 (β 0.2) | 4.87 (β 2) | 0.028 | — | `gowalla__mf__dfreq__directau__s0` | patience; 65/80 |
| gowalla | DoubleGraph † | BPR | 6.24 | 4.49 | 6.97 (β 0.1) | 7.27 (β 0.5) | 0.146 | — | `gowalla__mf__doublegraph__bpr__s0` | patience; 30/530 |
| gowalla | DoubleGraph † | DirectAU | 3.54 | 5.84 | 10.91 (β 0.1) | 7.38 (β 5) | -0.090 | — | `gowalla__mf__doublegraph__directau__s0` | patience; 5/35 |
| yelp2018 | Full | BPR | 8.02 | 5.54 | 8.50 (β 0.1) | 8.02 (β 0) | 0.970 | — | `yelp2018__mf__full__bpr__s0` | patience; 2150/2650 |
| yelp2018 | Full | DirectAU | 11.10 | 6.97 | 11.26 (β 0.1) | 11.10 (β 0) | 0.912 | — | `yelp2018__mf__full__directau__s0` | patience; 110/140 |
| yelp2018 | GraphHash | BPR | 3.78 | 2.03 | 4.82 (β 0.2) | 4.80 (β 0.5) | 0.190 | 0.822 | `yelp2018__mf__graphhash__bpr__s0` | patience; 360/860 |
| yelp2018 | GraphHash | DirectAU | 2.41 | 1.97 | 5.68 (β 0.2) | 4.96 (β 1) | 0.520 | 0.944 | `yelp2018__mf__graphhash__directau__s0` | patience; 50/60 |
| yelp2018 | Random | BPR | 0.55 | 0.38 | 2.15 (β 0.2) | 2.18 (β 1) | 0.075 | 0.159 | `yelp2018__mf__random__bpr__s0` | patience; 420/920 |
| yelp2018 | Random | DirectAU | 0.42 | 0.29 | 2.39 (β 0.2) | 2.41 (β 2) | 0.353 | 0.998 | `yelp2018__mf__random__directau__s0` | patience; 30/40 |
| yelp2018 | DoubleFreq | BPR | 2.55 | 1.90 | 2.52 (β 0.1) | 2.58 (β 0.1) | 0.145 | — | `yelp2018__mf__dfreq__bpr__s0` | patience; 460/960 |
| yelp2018 | DoubleFreq | DirectAU | 2.51 | 1.56 | 3.33 (β 0.2) | 3.17 (β 0.05) | 0.138 | — | `yelp2018__mf__dfreq__directau__s0` | patience; 15/45 |
| yelp2018 | DoubleGraph † | BPR | 2.14 | 1.14 | 2.85 (β 0.2) | 2.78 (β 0.5) | 0.167 | — | `yelp2018__mf__doublegraph__bpr__s0` | patience; 30/530 |
| yelp2018 | DoubleGraph † | DirectAU | 1.12 | 2.17 | 5.52 (β 0.2) | 3.48 (β 5) | -0.057 | — | `yelp2018__mf__doublegraph__directau__s0` | patience; 45/45 |
| amazonbook | Full | BPR | 8.38 | 6.59 | 8.62 (β 0.05) | 8.38 (β 0) | 0.862 | — | `amazonbook__mf__full__bpr__s0-long` | patience; 3800/4300 |
| amazonbook | Full | DirectAU | 9.57 | 8.70 | 11.05 (β 0.1) | 10.77 (β 0.2) | 0.786 | — | `amazonbook__mf__full__directau__s0` | patience; 25/40 |
| amazonbook | GraphHash | BPR | 4.10 | 2.53 | 3.34 (β 0.05) | 4.81 (β 0.2) | 0.084 | — | `amazonbook__mf__graphhash__bpr__s0` | patience; 1450/1950 |
| amazonbook | GraphHash | DirectAU | 4.63 | 4.64 | 6.33 (β 0.1) | 6.29 (β 0.5) | 0.189 | 0.746 | `amazonbook__mf__graphhash__directau__s0` | patience; 40/65 |
| amazonbook | Random | BPR | 0.17 | 0.11 | 1.31 (β 0.2) | 1.28 (β 1) | 0.083 | 0.296 | `amazonbook__mf__random__bpr__s0` | patience; 90/590 |
| amazonbook | Random | DirectAU | 0.15 | 0.12 | 1.55 (β 0.2) | 1.49 (β 0.5) | 0.242 | 0.998 | `amazonbook__mf__random__directau__s0` | patience; 35/35 |
| amazonbook | DoubleFreq | BPR | 1.28 | 1.01 | 1.35 (β 0.1) | 1.35 (β 0.5) | 0.136 | — | `amazonbook__mf__dfreq__bpr__s0` | patience; 420/920 |
| amazonbook | DoubleFreq | DirectAU | 1.51 | 1.12 | 2.13 (β 0.2) | 2.07 (β 0.5) | 0.026 | — | `amazonbook__mf__dfreq__directau__s0` | patience; 15/40 |

*Reading guide.*
- Compare **Raw** across losses for RQ1.
- **Cos + prior** (DAU) versus **Raw + prior** (BPR) is the registered corrected comparison.
- Random hashing is near-degenerate (raw recall 0.15–0.76%); treat it as a stress case.
- ρ_bucket is blank for Full (no buckets) and for DoubleFreq / DoubleGraph (two rows per item, no single bucket).

**Table 7. Same grid, NDCG@20 %.**

| Dataset | Sharing | Loss | Raw | Cosine | Cos + prior | Raw + prior |
|---|---|---|---|---|---|---|
| gowalla | Full | BPR | 10.09 | 6.69 | 10.12 | 10.67 |
| gowalla | Full | DirectAU | 11.61 | 7.46 | 12.59 | 11.62 |
| gowalla | GraphHash | BPR | 5.53 | 2.94 | 6.44 | 6.96 |
| gowalla | GraphHash | DirectAU | 4.06 | 4.03 | 8.57 | 6.41 |
| gowalla | Random | BPR | 0.38 | 0.20 | 2.19 | 2.17 |
| gowalla | Random | DirectAU | 0.26 | 0.15 | 2.43 | 2.38 |
| gowalla | DoubleFreq | BPR | 2.56 | 1.84 | 2.70 | 2.74 |
| gowalla | DoubleFreq | DirectAU | 2.56 | 1.73 | 3.50 | 3.38 |
| gowalla | DoubleGraph † | BPR | 3.71 | 2.46 | 4.45 | 4.60 |
| gowalla | DoubleGraph † | DirectAU | 1.75 | 3.14 | 6.92 | 4.31 |
| yelp2018 | Full | BPR | 5.13 | 3.27 | 5.43 | 5.13 |
| yelp2018 | Full | DirectAU | 7.20 | 4.24 | 7.35 | 7.20 |
| yelp2018 | GraphHash | BPR | 2.36 | 1.16 | 3.03 | 3.00 |
| yelp2018 | GraphHash | DirectAU | 1.36 | 1.10 | 3.55 | 3.08 |
| yelp2018 | Random | BPR | 0.32 | 0.23 | 1.41 | 1.43 |
| yelp2018 | Random | DirectAU | 0.25 | 0.18 | 1.62 | 1.57 |
| yelp2018 | DoubleFreq | BPR | 1.71 | 1.25 | 1.72 | 1.74 |
| yelp2018 | DoubleFreq | DirectAU | 1.89 | 1.20 | 2.38 | 2.27 |
| yelp2018 | DoubleGraph † | BPR | 1.34 | 0.67 | 1.81 | 1.78 |
| yelp2018 | DoubleGraph † | DirectAU | 0.65 | 1.31 | 3.58 | 2.11 |
| amazonbook | Full | BPR | 5.33 | 4.12 | 5.49 | 5.33 |
| amazonbook | Full | DirectAU | 6.45 | 5.69 | 7.48 | 7.26 |
| amazonbook | GraphHash | BPR | 2.60 | 1.59 | 2.12 | 3.28 |
| amazonbook | GraphHash | DirectAU | 3.12 | 3.13 | 4.46 | 4.44 |
| amazonbook | Random | BPR | 0.11 | 0.07 | 0.87 | 0.84 |
| amazonbook | Random | DirectAU | 0.09 | 0.07 | 1.07 | 1.01 |
| amazonbook | DoubleFreq | BPR | 0.91 | 0.70 | 0.98 | 0.97 |
| amazonbook | DoubleFreq | DirectAU | 1.25 | 0.96 | 1.62 | 1.58 |

**Table 8. The flip, MF and LightGCN.** DirectAU beats BPR on Full tables and loses under GraphHash on Gowalla and Yelp. On AmazonBook DirectAU keeps a smaller lead; on LightGCN Gowalla the premise does not hold.

| Setting | Full: DirectAU | Full: BPR | Full order | GraphHash: DirectAU | GraphHash: BPR | GraphHash order | Flip? |
|---|---|---|---|---|---|---|---|
| MF Gowalla | 18.02 | 16.55 | DAU > BPR | 7.55 | 9.37 | DAU < BPR | **yes** |
| MF Yelp2018 | 11.10 | 8.02 | DAU > BPR | 2.41 | 3.78 | DAU < BPR | **yes** |
| MF AmazonBook | 9.57 | 8.38 | DAU > BPR | 4.63 | 4.10 | DAU > BPR | no (lead +1.19 → +0.53) |
| LightGCN Gowalla | 18.02 | 18.48 | DAU < BPR | 11.29 | 16.22 | DAU < BPR | no premise (BPR already ahead on Full) |

*The paper claim is therefore "sharing removes DirectAU's advantage", not a universal reversal.*

Random and DoubleFreq (Table 6): raw DAU < raw BPR in 5 of 6 cells; AmazonBook DoubleFreq is the exception (1.51 vs 1.28).

**Table 9. Per-seed results, MF GraphHash** (training seeds 0, 1, 2; BPR seed 0 on Gowalla is `s0-p500`; AmazonBook BPR s0 is legacy).

| Dataset | Loss | Seed | Run | Raw | Cosine | Cos + prior (β) | Raw + prior (β) | ρ_item | ρ_bucket | Stop (best/ran) |
|---|---|---|---|---|---|---|---|---|---|---|
| gowalla | DirectAU | s0 | `gowalla__mf__graphhash__directau__s0` | 7.55 | 7.57 | 13.30 (β 0.2) | 10.33 (β 5) | 0.275 | 0.925 | patience; 55/60 |
| gowalla | DirectAU | s1 | `gowalla__mf__graphhash__directau__s1` | 7.61 | 7.59 | 13.32 (β 0.2) | 10.45 (β 5) | 0.277 | 0.924 | patience; 50/80 |
| gowalla | DirectAU | s2 | `gowalla__mf__graphhash__directau__s2` | 7.59 | 7.58 | 13.35 (β 0.2) | 10.54 (β 5) | 0.274 | 0.924 | patience; 50/60 |
| gowalla | BPR | s0 | `gowalla__mf__graphhash__bpr__s0-p500` | 9.37 | 5.86 | 9.96 (β 0.1) | 11.31 (β 0.5) | 0.066 | 0.862 | patience; 460/960 |
| gowalla | BPR | s1 | `gowalla__mf__graphhash__bpr__s1` | 9.39 | 5.41 | 9.73 (β 0.1) | 11.19 (β 0.5) | 0.080 | 0.885 | patience; 780/1280 |
| gowalla | BPR | s2 | `gowalla__mf__graphhash__bpr__s2` | 9.41 | 5.30 | 9.68 (β 0.1) | 11.27 (β 0.5) | 0.089 | 0.894 | patience; 1540/2040 |
| yelp2018 | DirectAU | s0 | `yelp2018__mf__graphhash__directau__s0` | 2.41 | 1.97 | 5.68 (β 0.2) | 4.96 (β 1) | 0.520 | 0.944 | patience; 50/60 |
| yelp2018 | DirectAU | s1 | `yelp2018__mf__graphhash__directau__s1` | 2.48 | 2.02 | 5.69 (β 0.2) | 5.01 (β 1) | 0.520 | 0.947 | patience; 20/45 |
| yelp2018 | DirectAU | s2 | `yelp2018__mf__graphhash__directau__s2` | 2.42 | 2.05 | 5.65 (β 0.2) | 5.01 (β 1) | 0.520 | 0.944 | patience; 50/60 |
| yelp2018 | BPR | s0 | `yelp2018__mf__graphhash__bpr__s0` | 3.78 | 2.03 | 4.82 (β 0.2) | 4.80 (β 0.5) | 0.190 | 0.822 | patience; 360/860 |
| yelp2018 | BPR | s1 | `yelp2018__mf__graphhash__bpr__s1` | 3.72 | 2.15 | 4.91 (β 0.2) | 4.83 (β 0.5) | 0.216 | 0.848 | patience; 410/910 |
| yelp2018 | BPR | s2 | `yelp2018__mf__graphhash__bpr__s2` | 3.71 | 2.09 | 4.96 (β 0.1) | 4.82 (β 0.5) | 0.210 | 0.838 | patience; 390/890 |
| amazonbook | DirectAU | s0 | `amazonbook__mf__graphhash__directau__s0` | 4.63 | 4.64 | 6.33 (β 0.1) | 6.29 (β 0.5) | 0.189 | 0.746 | patience; 40/65 |
| amazonbook | DirectAU | s1 | `amazonbook__mf__graphhash__directau__s1` | 4.59 | 4.60 | 6.28 (β 0.1) | 6.31 (β 0.5) | 0.173 | 0.736 | patience; 35/45 |
| amazonbook | DirectAU | s2 | `amazonbook__mf__graphhash__directau__s2` | 4.61 | 4.60 | 6.30 (β 0.1) | 6.16 (β 0.2) | 0.184 | 0.683 | patience; 20/45 |
| amazonbook | BPR | s0 | `amazonbook__mf__graphhash__bpr__s0` | 4.10 | 2.53 | 3.34 (β 0.05) | 4.81 (β 0.2) | 0.084 | — | patience; 1450/1950 |
| amazonbook | BPR | s1 | `amazonbook__mf__graphhash__bpr__s1` | 4.08 | 2.64 | 3.47 (β 0.05) | 4.79 (β 0.2) | 0.084 | 0.885 | patience; 780/1280 |
| amazonbook | BPR | s2 | `amazonbook__mf__graphhash__bpr__s2` | 4.09 | 2.51 | 3.39 (β 0.05) | 4.75 (β 0.2) | 0.085 | 0.889 | patience; 1300/1800 |

**Table 10. Mean ± sample SD over the 3 training seeds.** Note these are training-seed spreads and are separate from the paired-user bootstrap intervals in Table 16.

| Dataset | Loss | Repaired mode | Recall mean ± SD (3 seeds) | NDCG mean ± SD | Raw mean ± SD |
|---|---|---|---|---|---|
| gowalla | DirectAU | cos + prior | 13.325 ± 0.024 | 8.568 ± 0.011 | 7.581 ± 0.031 |
| gowalla | BPR | raw + prior | 11.257 ± 0.063 | 6.911 ± 0.040 | 9.389 ± 0.017 |
| yelp2018 | DirectAU | cos + prior | 5.677 ± 0.021 | 3.550 ± 0.018 | 2.436 ± 0.040 |
| yelp2018 | BPR | raw + prior | 4.819 ± 0.017 | 3.025 ± 0.019 | 3.738 ± 0.040 |
| amazonbook | DirectAU | cos + prior | 6.302 ± 0.025 | 4.448 ± 0.016 | 4.611 ± 0.021 |
| amazonbook | BPR | raw + prior | 4.786 ± 0.030 | 3.279 ± 0.011 | 4.092 ± 0.007 |

---

## 4. RQ2: Is popularity lost at item level?

### 4.1 Norm–popularity correlations

**Table 11. Item-quintile mean vector lengths** (Q1 = least popular fifth of items, Q5 = most popular), taken at the last logged epoch as in the `RESULTS.md` outcome blocks; ρ values in this table are at that same epoch. Table 6's ρ is at the best-checkpoint epoch, so BPR rows differ (for example Yelp GraphHash BPR ρ_item 0.190 at best vs 0.325 here).

| Run | Last logged epoch | Q1 (least popular) | Q2 | Q3 | Q4 | Q5 (most popular) | Q5/Q1 | ρ_item | ρ_bucket | ρ_dir |
|---|---|---|---|---|---|---|---|---|---|---|
| `gowalla__mf__full__directau__s0` | 170 | 7.27 | 7.71 | 8.11 | 8.78 | 10.67 | 1.47 | 0.815 | — | -0.028 |
| `gowalla__mf__graphhash__directau__s0` | 60 | 11.71 | 12.07 | 12.18 | 12.29 | 12.71 | 1.09 | 0.276 | 0.925 | -0.057 |
| `gowalla__mf__full__bpr__s0-long` | 3290 | 3.27 | 3.44 | 3.60 | 3.84 | 4.31 | 1.32 | 0.783 | — | 0.331 |
| `gowalla__mf__graphhash__bpr__s0-p500` | 960 | 3.03 | 3.14 | 3.16 | 3.15 | 3.16 | 1.04 | 0.084 | 0.889 | 0.196 |
| `yelp2018__mf__full__directau__s0` | 140 | 6.95 | 7.55 | 8.26 | 9.25 | 11.40 | 1.64 | 0.911 | — | 0.137 |
| `yelp2018__mf__graphhash__directau__s0` | 60 | 1.89 | 2.07 | 2.15 | 2.25 | 2.52 | 1.34 | 0.520 | 0.944 | 0.164 |
| `yelp2018__mf__full__bpr__s0` | 2650 | 1.19 | 1.34 | 1.53 | 1.79 | 2.39 | 2.02 | 0.970 | — | 0.375 |
| `yelp2018__mf__graphhash__bpr__s0` | 860 | 2.45 | 2.67 | 2.73 | 2.77 | 2.89 | 1.18 | 0.325 | 0.916 | 0.352 |
| `amazonbook__mf__full__directau__s0` | 40 | 1.98 | 2.06 | 2.12 | 2.19 | 2.28 | 1.15 | 0.737 | — | 0.013 |
| `amazonbook__mf__graphhash__directau__s0` | 65 | 2.94 | 2.95 | 2.96 | 2.97 | 2.98 | 1.02 | 0.212 | 0.812 | 0.046 |
| `amazonbook__mf__full__bpr__s0-long` | 4300 | 2.89 | 3.11 | 3.34 | 3.65 | 4.27 | 1.48 | 0.863 | — | 0.294 |
| `amazonbook__mf__graphhash__bpr__s1` | 1280 | 2.92 | 2.97 | 2.98 | 2.94 | 2.91 | 0.99 | 0.085 | 0.888 | 0.283 |

Reading: Full tables stretch length with popularity (Q5/Q1 1.15–2.02); under GraphHash DirectAU the spread flattens to 1.02–1.34, while ρ_bucket stays 0.81–0.94 at the last epoch (0.75–0.94 at the best checkpoint). BPR's ρ_dir is higher than DirectAU's on the same dataset: Yelp GraphHash 0.352 vs 0.164, Yelp Full 0.375 vs 0.137 (H-dir).

### 4.2 The cosine ablation

**Table 12. Effect of removing length, and of adding the prior** (DirectAU, seed 0).

| Dataset | Δ(cos − raw), Full DirectAU | Δ(cos − raw), GraphHash DirectAU | Prior gain over raw (cos+prior − raw), Full | Prior gain over raw, GraphHash | Prior gain over cosine, Full | Prior gain over cosine, GraphHash |
|---|---|---|---|---|---|---|
| gowalla | -4.959 | +0.021 | +1.55 | +5.76 | +6.51 | +5.74 |
| yelp2018 | -4.126 | -0.442 | +0.16 | +3.27 | +4.29 | +3.71 |
| amazonbook | -0.873 | +0.004 | +1.47 | +1.70 | +2.34 | +1.69 |

Under BPR, cosine still loses under sharing (Gowalla GraphHash 9.37 → 5.86). This suggests BPR's length carries something other than popularity; it is an open point, not a result.

### 4.3 Bucket structure and the correlation-ratio bound

**Table 13. Bucket structure, η², and the proxy correlation.** η²_p = Var(μ_b) / Var(p) with p = log(1 + c): the share of item-popularity variance that any bucket-level quantity can reach (Proposition 2). ρ_proxy = Spearman(c_i, bucket traffic of i's bucket). The check ρ² ≤ η² holds in every row.

| Map | Item groups | Items/group mean (median, max) | Share of items in groups ≥ 2 | η²_p = Var(μ)/Var(p) | η² (rank c) | ρ_proxy | ρ_proxy² ≤ η²(rank c)? |
|---|---|---|---|---|---|---|---|
| `gowalla__full` | 40,981 | 1.00 (1, 1) | 0.000 | 1.000 | 1.000 | 1.000 | True |
| `gowalla__graphhash` | 5,806 | 7.06 (6, 124) | 0.996 | 0.359 | 0.271 | 0.189 | True |
| `gowalla__random` | 5,806 | 7.06 (7, 8) | 1.000 | 0.142 | 0.141 | 0.276 | True |
| `gowalla__dfreq` | 40,902 | 1.00 (1, 2) | 0.004 | 0.999 | 0.998 | -0.050 | True |
| `gowalla__graphhash-r400` | 8,924 | 4.59 (4, 79) | 0.975 | 0.493 | 0.412 | 0.377 | True |
| `gowalla__graphhash-r800` | 12,880 | 3.18 (3, 40) | 0.911 | 0.630 | 0.551 | 0.531 | True |
| `gowalla__graphhash-r1600` | 17,699 | 2.32 (1, 14) | 0.772 | 0.785 | 0.677 | 0.550 | True |
| `gowalla__graphhash-r3200` | 25,993 | 1.58 (1, 11) | 0.514 | 0.924 | 0.827 | 0.513 | True |
| `yelp2018__full` | 38,048 | 1.00 (1, 1) | 0.000 | 1.000 | 1.000 | 1.000 | True |
| `yelp2018__graphhash` | 7,656 | 4.97 (4, 183) | 0.993 | 0.484 | 0.437 | 0.468 | True |
| `yelp2018__random` | 7,656 | 4.97 (5, 5) | 1.000 | 0.199 | 0.198 | 0.353 | True |
| `yelp2018__dfreq` | 37,161 | 1.02 (1, 3) | 0.047 | 0.987 | 0.981 | -0.105 | True |
| `amazonbook__full` | 91,599 | 1.00 (1, 1) | 0.000 | 1.000 | 1.000 | 1.000 | True |
| `amazonbook__graphhash` | 9,804 | 9.34 (8, 343) | 0.999 | 0.358 | 0.303 | 0.228 | True |
| `amazonbook__random` | 9,804 | 9.34 (9, 10) | 1.000 | 0.109 | 0.108 | 0.241 | True |
| `amazonbook__dfreq` | 91,454 | 1.00 (1, 2) | 0.003 | 0.999 | 0.998 | -0.060 | True |

**Table 14. Bound check.**

| Dataset | GraphHash η²_p | Random η²_p | ρ_item² (GraphHash DAU) | Bound ρ_item² ≤ η² |
|---|---|---|---|---|
| Gowalla | 0.359 | 0.142 | 0.076 | holds |
| Yelp2018 | 0.484 | 0.199 | 0.270 | holds |
| AmazonBook | 0.358 | 0.109 | 0.036 | holds |

DoubleFreq: η² ≈ 0.99–1.00, so the bound does **not** bind, yet ρ_item is 0.03 (Gowalla) and DAU still loses to BPR (3.22 vs 3.93). The bound is sufficient, not necessary: the norm of a sum of two shared rows follows bucket traffic even when identity survives. Gowalla has 79 collisions (158 items, 0.4%) in its DoubleFreq alias pairs, so the within-bucket decomposition is degenerate there (Table 20).

η² against resolution (Gowalla GraphHash): 0.359 (200), 0.493 (400), 0.630 (800), 0.785 (1600), 0.924 (3200).

### 4.4 Norm dynamics, weight decay and equal norms

**H-Adam (norm growth).** With DirectAU, lengths only grow: the largest consecutive drop of any per-quintile mean norm is −0.15% (Yelp GraphHash), −0.11% (Yelp Full), none on Gowalla GraphHash. For contrast Yelp Full BPR drops −19%, since BPR is not scale-invariant. This supports the Euler-identity argument (Proposition 1).

**Table 15. Weight decay and forced-equal-norm runs, Gowalla.**

| Variant | Run | Raw | Cosine | Cos + prior (β) | Raw + prior (β) | ρ_item | Stop (best/ran) |
|---|---|---|---|---|---|---|---|
| Full DirectAU (reference) | `gowalla__mf__full__directau__s0` | 18.02 | 13.06 | 19.57 (β 0.1) | 18.02 (β 0.01) | 0.81 | patience; 160/170 |
| Full DirectAU, wd = 0 | `gowalla__mf__full__directau__s0-wd0` | 16.12 | 12.61 | 19.08 (β 0.1) | 17.89 (β 2) | 0.74 | patience; 15/45 |
| Full DirectAU, wd = 1e-4 | `gowalla__mf__full__directau__s0-wd1e-4` | 1.41 | 2.75 | 8.69 (β 0.1) | 6.76 (β 0.05) | 0.22 | patience; 30/35 |
| Full DirectAU, forced equal item norms | `gowalla__mf__full__directau_eqnorm__s0` | 12.77 | 12.77 | 19.32 (β 0.1) | 19.01 (β 5) | -0.02 | patience; 25/55 |
| GraphHash DirectAU (reference) | `gowalla__mf__graphhash__directau__s0` | 7.55 | 7.57 | 13.30 (β 0.2) | 10.33 (β 5) | 0.28 | patience; 55/60 |
| GraphHash DirectAU, forced equal item norms | `gowalla__mf__graphhash__directau_eqnorm__s0` | 7.56 | 7.56 | 13.27 (β 0.2) | 11.55 (β 5) | -0.01 | patience; 30/60 |

- wd = 0 keeps ρ_item at 0.74, so the length–popularity link persists without weight decay in this setup (D7 PASS). This is not a controlled contradiction of PRISM.
- wd = 1e-4 collapses the model (raw 1.41).
- With item norms forced equal, raw equals cosine (12.77 on Full, 7.56 on GraphHash) and the prior then restores Full to 19.32 and GraphHash to 13.27.

---

## 5. RQ3: Is the recovered gain item-specific?

**Table 16. Corrected DirectAU vs corrected BPR, 9 dataset × seed pairs.** Paired user bootstrap (2,000 resamples, seed 0, percentile CI). Holm p-values come from paired t-tests over the three seed-0 pairs. For the four Stage 4 pairs the Recall CI is also recomputed here from `per_user_test__*.npz`; NDCG for these four is recomputed here as well (the project reports Recall CIs only). Recomputed CIs differ from the reported ones by ≤ 0.01 pp, which is bootstrap noise.

| Dataset | Seed | DirectAU cos+prior | BPR raw+prior | Recall Δ pp | Recall 95% CI | NDCG Δ pp | NDCG 95% CI | Holm p (Recall) | Source |
|---|---|---|---|---|---|---|---|---|---|
| Gowalla | s0 | 13.305 | 11.315 | +1.990 | [1.771, 2.238] | +1.615 | [1.492, 1.743] | 2.43e-63 | E8_final.json (project script) |
| Gowalla | s1 | 13.319 | 11.189 | +2.130 | [1.909, 2.370] | +1.663 | [1.543, 1.794] | — | E8_final.json (project script) |
| Gowalla | s2 | 13.352 | 11.267 | +2.084 | [1.855, 2.329] | +1.692 | [1.573, 1.821] | — | E8_final.json (project script) |
| Yelp2018 | s0 | 5.682 | 4.801 | +0.882 | [0.735, 1.031] | +0.546 | [0.460, 0.630] | 6.14e-33 | E8_final.json (project script) |
| Yelp2018 | s1 | 5.695 | 4.834 | +0.860 | [0.717, 1.004] (RESULTS S4-SEED: +0.86 [0.71, 1.01]) | +0.530 | [0.446, 0.623] | — | per-user files, recomputed here (2,000 paired resamples, seed 0) |
| Yelp2018 | s2 | 5.654 | 4.822 | +0.832 | [0.686, 0.983] (RESULTS S4-SEED: +0.83 [0.69, 0.98]) | +0.499 | [0.412, 0.582] | — | per-user files, recomputed here (2,000 paired resamples, seed 0) |
| AmazonBook | s0 | 6.328 | 4.810 | +1.518 | [1.405, 1.629] | +1.180 | [1.107, 1.254] | 1.19e-159 | E8_final.json (project script) |
| AmazonBook | s1 | 6.278 | 4.793 | +1.485 | [1.370, 1.592] (RESULTS S4-SEED: +1.48 [1.38, 1.60]) | +1.147 | [1.070, 1.221] | — | per-user files, recomputed here (2,000 paired resamples, seed 0) |
| AmazonBook | s2 | 6.298 | 4.754 | +1.545 | [1.436, 1.658] (RESULTS S4-SEED: +1.54 [1.44, 1.66]) | +1.180 | [1.109, 1.256] | — | per-user files, recomputed here (2,000 paired resamples, seed 0) |

*Gowalla seed 0 pairs `gowalla__mf__graphhash__directau__s0` with `…bpr__s0-p500`. AmazonBook seed 0 pairs DAU with the legacy BPR checkpoint (E8 control file); seeds 1–2 pair modern runs.*

**Table 17. Stage 3 registered inference** *(verbatim from RESULTS.md lines 361–367)*

| Dataset | DirectAU cos+prior | BPR raw+prior | Δ | 95% CI | Holm p |
|---|---|---|---|---|---|
| Gowalla s0 (A1 vs s0-p500) | 13.30 | 11.31 | +1.99 | [1.77, 2.24] | 2.4e-63 |
| Gowalla s1 | 13.32 | 11.19 | +2.13 | [1.91, 2.37] | — |
| Gowalla s2 | 13.35 | 11.27 | +2.08 | [1.85, 2.33] | — |
| Yelp2018 s0 | 5.68 | 4.80 | +0.88 | [0.73, 1.03] | 6.1e-33 |
| AmazonBook s0 (B9 vs legacy BPR-long) | 6.33 | 4.81 | +1.52 | [1.41, 1.63] | 1.2e-159 |

**Table 18. Retained inference, Recall** *(verbatim from RESULTS.md lines 465–471)*

| Pair | DAU Recall % | BPR Recall % | Δ pp | 95% CI | Primary Holm p |
|---|---|---|---|---|---|
| Gowalla s0 | 13.305 | 11.315 | 1.990 | [1.771, 2.238] | 2.43e-63 |
| Gowalla s1 | 13.319 | 11.189 | 2.130 | [1.909, 2.370] | — |
| Gowalla s2 | 13.352 | 11.267 | 2.084 | [1.855, 2.329] | — |
| Yelp2018 | 5.682 | 4.801 | 0.882 | [0.735, 1.031] | 6.14e-33 |
| AmazonBook | 6.328 | 4.810 | 1.518 | [1.405, 1.629] | 1.19e-159 |

**Table 19. Secondary NDCG** *(verbatim from RESULTS.md lines 475–481)*

| Pair | DAU NDCG % | BPR NDCG % | Δ pp | 95% CI | Secondary Holm p |
|---|---|---|---|---|---|
| Gowalla s0 | 8.572 | 6.957 | 1.615 | [1.492, 1.743] | 7.86e-137 |
| Gowalla s1 | 8.555 | 6.891 | 1.663 | [1.543, 1.794] | — |
| Gowalla s2 | 8.576 | 6.884 | 1.692 | [1.573, 1.821] | — |
| Yelp2018 | 3.549 | 3.003 | 0.546 | [0.460, 0.630] | 7.96e-36 |
| AmazonBook | 4.464 | 3.285 | 1.180 | [1.107, 1.254] | 1.62e-208 |

AmazonBook robustness (S4-RESCORE): with the full prior at β 0.2, every tie policy (index, popularity key, `torch.topk`, random seeds 0–9) gives Δ +1.51 to +1.52 pp, every CI within [+1.39, +1.63]. All ten stored random-key aggregates are positive (+1.508 to +1.524).

**Table 20. Fixed-β decomposition and tie analysis per checkpoint** (legacy checkpoints, test Recall@20; base is the random-key tie order). Δ_μ = bucket-level prior only, Δ_ε = item-residual prior only, Δ_both = full prior; Shapley shares split Δ_both between the two. "Within-bucket shuffle" permutes the item part of the prior inside each bucket. Tie columns refer to the 20th slot of the base ranking.

| Checkpoint (legacy) | Base | Δ_μ | Δ_ε | Δ_both | Interaction | Shapley μ | Shapley ε | Within-bucket shuffle gain | Boundary-tie users | Mean tied group | Pop-key tie-break gain | Tie-break-only share of gain | Index − random key |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `gowalla__mf__graphhash__bpr__s0-legacy` | 9.29 | -0.76 | +3.23 | +1.96 | -0.50 | -0.52 | 1.52 | -1.25 | 78% | 11.2 | +0.67 | 34% | +0.06 |
| `gowalla__mf__graphhash__directau__s0-legacy30` | 7.54 | +1.72 | +4.16 | +5.73 | -0.15 | 0.29 | 0.71 | -0.39 | 87% | 14.7 | +0.67 | 12% | +0.04 |
| `gowalla__mf__random__directau__s0-legacy30` | 0.30 | +0.14 | +3.43 | +3.56 | -0.02 | 0.04 | 0.96 | +0.31 | 84% | 6.9 | +0.04 | 1% | +0.01 |
| `amazonbook__mf__graphhash__bpr__s0-legacy` | 4.10 | -0.17 | +0.91 | +0.71 | -0.02 | -0.26 | 1.26 | -0.23 | 87% | 17.2 | +0.43 | 60% | +0.00 |
| `amazonbook__mf__graphhash__directau__s0-legacy41` | 4.62 | +0.38 | +1.25 | +1.70 | +0.07 | 0.24 | 0.76 | -0.17 | 89% | 17.6 | +0.42 | 25% | +0.01 |
| `gowalla__mf__dfreq__directau__s0-legacy30` | 2.18 | +2.96 | -0.00 | +2.97 | +0.00 | 1.00 | 0.00 | +2.97 | 0% | 1.0 | +0.00 | 0% | +0.00 |
| `gowalla__mf__graphhash__directau__s1-legacy30` | 7.51 | +1.76 | +4.25 | +5.81 | -0.20 | 0.29 | 0.71 | -0.35 | 87% | 14.7 | +0.70 | 12% | +0.05 |
| `gowalla__mf__graphhash__directau__s2-legacy30` | 7.49 | +1.74 | +4.24 | +5.82 | -0.16 | 0.29 | 0.71 | -0.34 | 87% | 14.8 | +0.71 | 12% | +0.05 |

- Gowalla GraphHash DAU, mean of seeds 0–2: Δ_μ +1.74, Δ_ε +4.22, Δ_both +5.79, interaction −0.17; Shapley μ 0.29 / ε 0.71; shuffle −0.36 pp (the shuffled prior is *worse than no prior*; it keeps −6% of the gain).
- AmazonBook GraphHash DAU: μ 0.24 / ε 0.76, shuffle −0.17. Random DAU: μ 0.04 / ε 0.96.
- BPR checkpoints (Gowalla, AmazonBook) have a *negative* μ share: the bucket-level term alone hurts raw BPR.
- Ties: boundary-tie users 87% (Gowalla), 89% (AmazonBook); the choice of tie key moves Recall by +0.04 to +0.06 pp; resolving ties alone accounts for 12% (Gowalla DAU), 25% (AmazonBook DAU), 34% (Gowalla BPR), 60% (AmazonBook BPR) of the gain.
- DoubleFreq has ε ≈ 0 (degenerate decomposition) and no ties; Δ_both +2.97 is almost all μ.

**Table 21. Popularity-only baseline (not MostPop).**

| Dataset | Recall@20 % | NDCG@20 % | Evaluated test users | Repaired GraphHash DirectAU (A1/A4/B9) | Gap |
|---|---|---|---|---|---|
| Gowalla | 2.97 | 1.70 | 25,789 | 13.30 | 10.33 |
| Yelp2018 | 1.68 | 1.05 | 30,304 | 5.68 | 4.00 |
| AmazonBook | 0.96 | 0.61 | 50,407 | 6.33 | 5.37 |

Gowalla's 13.29 in `RESULTS.md` PR-A6 is the mean of the three legacy-30 seeds (13.26, 13.31, 13.30); the modern A1 value is 13.30.

**Table 22. Activity quartiles.** The gain is spread across users (change in Recall@20, points; Q1 = least active users) *(verbatim from RESULTS.md lines 296–302)*:

| Case | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| Gowalla A1 DirectAU | +6.382 | +5.462 | +5.762 | +5.346 |
| Yelp A4 DirectAU | +3.600 | +3.973 | +3.782 | +3.503 |
| Yelp A5 BPR | +0.722 | +0.987 | +1.252 | +1.103 |
| AmazonBook legacy DirectAU | +1.798 | +1.707 | +1.725 | +1.574 |
| AmazonBook legacy BPR | +0.779 | +0.669 | +0.740 | +0.602 |

---

## 6. RQ4: Does erasing popularity without sharing reproduce the symptom?

**Table 23. Length interventions on Full DirectAU** (legacy checkpoints, 10 seeds for the shuffles; Δ vs the unmodified checkpoint in brackets). Scoring is the raw dot product with directions kept and lengths modified.

| Checkpoint (legacy) | Unmodified | Decile-preserving shuffle | Global shuffle (10 seeds) | Replace by fitted f(popularity) | Fit: log‖q‖ = a + g·p | R² |
|---|---|---|---|---|---|---|
| `gowalla__mf__full__directau__s0-legacy120` | val 17.62; test 17.90 | val 13.43 (-4.19); test 13.64 (-4.26) | val 6.15 (-11.47); test 6.28 (-11.61) | val 18.45 (+0.83); test 18.69 (+0.79) | a = 1.491, g = 0.231 | 0.71 |
| `amazonbook__mf__full__directau__s0-legacy` | val 9.51; test 9.57 | val 9.18 (-0.33); test 9.30 (-0.28) | val 7.28 (-2.23); test 7.44 (-2.14) | val 10.59 (+1.08); test 10.64 (+1.07) | a = 0.325, g = 0.063 | 0.49 |

- Gowalla: global shuffle −11.47 (val) / −11.61 (test), decile shuffle −4.19 / −4.26, replacement +0.83 / +0.79.
- AmazonBook: global −2.23, decile −0.33, replacement +1.08 (val).
- A smooth function of popularity fitted to log-length does as well as or better than the learned lengths: popularity is the useful content of length.
- The pre-registered bands failed (decile \|Δ\| ≤ 0.5 / 0.3; replacement \|Δ\| ≤ 0.7 / 0.3), though replacement failed in the favourable direction. The registered thresholds were not changed.

These checkpoints are the legacy ones (Gowalla Full DAU time-capped at 130 epochs, raw 17.90), so the baseline differs from the modern 18.02 in Table 6 by 0.12.

Equal-norm and weight-decay runs are in Table 15.

---

## 7. RQ5: Does it generalise?

### 7.1 LightGCN

**Table 24. LightGCN (2 layers), test Recall@20 %.** Five rows are capped at the 4 h wall (`time_cap`); the 2-GPU reruns of Gowalla Full converged. ρ_item is measured on the propagated item vectors but ρ_bucket on the raw table rows, so ρ_bucket here is not comparable with MF.

| Dataset | Sharing | Loss | Raw | Cosine | Cos + prior (β) | Raw + prior (β) | ρ_item | ρ_bucket | Stop (best/ran) | Train min | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gowalla | Full | BPR | 18.44 | 13.65 | 18.89 (β 0.1) | 18.44 (β 0) | 0.86 | — | time_cap; 340/345 | 240 | capped |
| Gowalla | Full | BPR (2-GPU, converged) | 18.48 | 13.65 | 19.04 (β 0.1) | 18.46 (β 0.01) | 0.87 | — | patience; 410/460 | 320 | converged |
| Gowalla | Full | DirectAU | 17.94 | 13.32 | 19.78 (β 0.1) | 17.99 (β 0.1) | 0.66 | — | time_cap; 182/182 | 241 | capped |
| Gowalla | Full | DirectAU (2-GPU, converged) | 18.02 | 13.33 | 19.76 (β 0.1) | 18.02 (β 0.02) | 0.70 | — | patience; 215/215 | 285 | converged |
| Gowalla | GraphHash | BPR | 16.22 | 10.52 | 15.79 (β 0.1) | 16.27 (β 0.1) | 0.38 | -0.12 | patience; 100/150 | 98 | converged |
| Gowalla | GraphHash | DirectAU | 11.29 | 10.95 | 17.06 (β 0.1) | 12.38 (β 0.2) | 0.15 | 0.11 | patience; 120/150 | 193 | converged |
| Yelp2018 | Full | BPR | 9.55 | 6.89 | 9.48 (β 0.1) | 9.55 (β 0.01) | 0.88 | — | time_cap; 165/169 | 240 | capped |
| Yelp2018 | Full | DirectAU | 10.70 | 7.11 | 11.43 (β 0.1) | 10.82 (β 0.1) | 0.78 | — | time_cap; 95/99 | 242 | capped |
| Yelp2018 | GraphHash | BPR | 6.96 | 3.91 | 6.95 (β 0.1) | 6.96 (β 0.05) | 0.50 | -0.48 | patience; 75/125 | 174 | converged |
| Yelp2018 | GraphHash | DirectAU | 4.71 | 5.09 | 9.12 (β 0.2) | 5.51 (β 0.5) | 0.13 | -0.20 | time_cap; 100/102 | 242 | capped |

- The flip appears with LightGCN under GraphHash (raw DAU < raw BPR on both datasets: 11.29 vs 16.22, 4.71 vs 6.96) and the correction reverses it (17.06 vs 16.27, 9.12 vs 6.96).
- On Gowalla Full, BPR is ahead before any correction (18.48 vs 18.02), so the flip premise is MF-specific.
- LightGCN GraphHash BPR is much stronger than MF's (16.22 vs 9.37): propagation partly restores item-level differences.
- S5-LGNFULL: the converged 2-GPU runs met the prediction (raw 18.48 ≥ 18.44 and 18.02 ≥ 17.94; repaired DAU 19.76 > repaired BPR 18.46).

### 7.2 DoubleGraph (BACO substitute)

**Table 25. DoubleGraph, single seed.**

| Dataset | Run | Raw | Cosine | Cos + prior (β) | Raw + prior (β) | ρ_item | Stop |
|---|---|---|---|---|---|---|---|
| gowalla | DoubleGraph DirectAU (`gowalla__mf__doublegraph__directau__s0`) | 3.54 | 5.84 | 10.91 (β 0.1) | 7.38 (β 5) | -0.09 | patience; 5/35 |
| gowalla | DoubleGraph BPR (`gowalla__mf__doublegraph__bpr__s0`) | 6.24 | 4.49 | 6.97 (β 0.1) | 7.27 (β 0.5) | 0.15 | patience; 30/530 |
| yelp2018 | DoubleGraph DirectAU (`yelp2018__mf__doublegraph__directau__s0`) | 1.12 | 2.17 | 5.52 (β 0.2) | 3.48 (β 5) | -0.06 | patience; 45/45 |
| yelp2018 | DoubleGraph BPR (`yelp2018__mf__doublegraph__bpr__s0`) | 2.14 | 1.14 | 2.85 (β 0.2) | 2.78 (β 0.5) | 0.17 | patience; 30/530 |

ρ_item ≈ 0 for DoubleGraph DirectAU (−0.09 / −0.06), yet the corrected DAU beats corrected BPR (10.91 vs 7.27; 5.52 vs 2.78).

### 7.3 Resolution sweep

**Table 26. Gowalla GraphHash, resolution 200 → 3200** (single seed †; the 1600 DirectAU run is `time_cap`-capped, so its row is inconclusive for convergence). The gain at 200 is 5.76 from unrounded values; `RESULTS.md` S4-C1 prints 5.75 from the rounded ones.

| Resolution | Item buckets | Table ratio | DAU raw | DAU cos+prior | Gain (cos+prior − raw) | ρ_item (DAU) | ρ_bucket (DAU) | BPR raw | BPR raw+prior | DAU repaired − BPR repaired | DAU stop | BPR stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200 | 5806 | 16.4% | 7.55 | 13.30 | +5.76 | 0.28 | 0.92 | 9.37 | 11.31 | +1.99 | patience; 55/60 | patience; 460/960 |
| 400 | 8924 | 25.1% | 7.49 | 13.15 | +5.66 | 0.44 | 0.93 | 9.79 | 10.96 | +2.19 | patience; 65/80 | patience; 2100/2600 |
| 800 | 12880 | 35.8% | 7.86 | 13.15 | +5.29 | 0.56 | 0.95 | 10.42 | 11.16 | +1.99 | patience; 170/170 | patience; 1520/2020 |
| 1600 | 17699 | 46.7% | 8.62 | 14.02 | +5.40 | 0.57 | 0.95 | 11.57 | 11.77 | +2.26 | time_cap; 185/185 | patience; 1090/1590 |
| 3200 | 25993 | 63.8% | 10.87 | 15.86 | +4.99 | 0.54 | 0.93 | 12.61 | 12.76 | +3.10 | patience; 165/175 | patience; 1290/1790 |

Spearman(resolution, gain) = −0.90 (recomputed here from the five rows) passes (≤ −0.8). Spearman(resolution, ρ_item) = +0.70 fails (< +0.8): ρ_item plateaus after 800. Repaired DAU beats repaired BPR at every resolution.

### 7.4 Random and DoubleFreq (registered range checks)

Counts from the Stage 3 outcome block (cells are the 24-cell grid in Table 6):

**Table 27. Registered range checks, Random and DoubleFreq.**

| Check | Result |
|---|---|
| B-i: cos+prior ≥ max(raw, cos) + 1.0 on compressed DirectAU cells | 6 of 8 (fails: AmazonBook DoubleFreq +0.62, Yelp DoubleFreq +0.82) |
| B-ii: DAU β ∈ {0.1, 0.2} | 8 of 8 |
| B-iii: ρ ≤ ρ_full − 0.3 | 8 of 8 |
| B-iv: BPR raw+prior β ∈ [0.2, 0.5] | 6 of 9 (fails: Yelp Random 1.0, AmazonBook Random 1.0, Yelp DoubleFreq 0.1) |
| [X] raw DAU < raw BPR | 5 of 6 Random/DoubleFreq cells |
| [X] repaired DAU > repaired BPR | 6 of 6 |

---

## 8. RQ6: What does it cost?

### 8.1 Tail recall and catalog coverage (E5)

The repair is a **head-accuracy repair with a tail cost**. "Tail" is the bottom 80% of items by training count; tail recall is `tail_hits / min(20, n_tail_truth)` among users with tail truth (`tail_primary`). Catalog coverage is the share of the catalog that appears in any user's top-20.

**Table 28. Repaired GraphHash DirectAU vs the Full DirectAU raw reference** (all 7 retained GraphHash DAU variants; 2,000 paired tail-eligible-user resamples). Every variant has a lower tail recall than Full raw: PASS for the registered negative direction, which establishes a tail deficit, not a tail-safe repair.

| Dataset | Repaired GraphHash DirectAU run | Repaired tail % | Full raw tail % | Δ pp | Paired 95% CI (pp) | n tail-eligible users | Direction |
|---|---|---|---|---|---|---|---|
| Gowalla | `gowalla__mf__graphhash__directau__s0` | 2.005 | 2.501 | -0.496 | [-0.677, -0.330] | 19,512 | PASS |
| Yelp2018 | `yelp2018__mf__graphhash__directau__s0` | 0.290 | 1.030 | -0.740 | [-0.830, -0.647] | 23,011 | PASS |
| AmazonBook | `amazonbook__mf__graphhash__directau__s0` | 3.781 | 5.407 | -1.626 | [-1.799, -1.453] | 39,697 | PASS |
| Gowalla | `gowalla__mf__graphhash__directau__s0-initmatch` | 1.986 | 2.501 | -0.515 | [-0.699, -0.344] | 19,512 | PASS |
| Gowalla | `gowalla__mf__graphhash__directau__s1` | 1.957 | 2.501 | -0.544 | [-0.718, -0.373] | 19,512 | PASS |
| Gowalla | `gowalla__mf__graphhash__directau__s2` | 1.983 | 2.501 | -0.518 | [-0.700, -0.349] | 19,512 | PASS |
| AmazonBook | `amazonbook__mf__graphhash__directau__s0-initmatch` | 3.760 | 5.407 | -1.647 | [-1.822, -1.477] | 39,697 | PASS |

**Table 29. Tail and coverage profile of the primary checkpoints.** The AmazonBook Full BPR row is the capped `s0` run; the AmazonBook GraphHash BPR row is the legacy checkpoint (its cosine cell is missing). Every record is in Appendix B.

| Dataset | Run | Scoring mode | Test Recall@20 % | Tail recall % (bottom 80% items) | Catalog coverage % |
|---|---|---|---|---|---|
| Gowalla | `gowalla__mf__full__directau__s0` | raw | 18.02 | 2.501 | 30.8 |
| Gowalla | `gowalla__mf__full__directau__s0` | cosine | 13.06 | 12.295 | 96.3 |
| Gowalla | `gowalla__mf__full__directau__s0` | cos_prior | 19.57 | 5.191 | 45.7 |
| Gowalla | `gowalla__mf__full__bpr__s0-long` | raw | 16.55 | 5.912 | 72.2 |
| Gowalla | `gowalla__mf__full__bpr__s0-long` | cosine | 11.96 | 10.681 | 98.4 |
| Gowalla | `gowalla__mf__full__bpr__s0-long` | raw_prior | 17.06 | 3.870 | 50.0 |
| Gowalla | `gowalla__mf__graphhash__directau__s0` | raw | 7.55 | 7.668 | 80.4 |
| Gowalla | `gowalla__mf__graphhash__directau__s0` | cosine | 7.57 | 8.813 | 95.2 |
| Gowalla | `gowalla__mf__graphhash__directau__s0` | cos_prior | 13.30 | 2.005 | 28.4 |
| Gowalla | `gowalla__mf__graphhash__bpr__s0-p500` | raw | 9.37 | 6.149 | 52.8 |
| Gowalla | `gowalla__mf__graphhash__bpr__s0-p500` | cosine | 5.86 | 5.145 | 78.2 |
| Gowalla | `gowalla__mf__graphhash__bpr__s0-p500` | raw_prior | 11.31 | 3.030 | 25.1 |
| Yelp2018 | `yelp2018__mf__full__directau__s0` | raw | 11.10 | 1.030 | 32.0 |
| Yelp2018 | `yelp2018__mf__full__directau__s0` | cosine | 6.97 | 7.013 | 97.8 |
| Yelp2018 | `yelp2018__mf__full__directau__s0` | cos_prior | 11.26 | 2.290 | 43.9 |
| Yelp2018 | `yelp2018__mf__full__bpr__s0` | raw | 8.02 | 0.123 | 9.2 |
| Yelp2018 | `yelp2018__mf__full__bpr__s0` | cosine | 5.54 | 3.475 | 84.7 |
| Yelp2018 | `yelp2018__mf__full__bpr__s0` | raw_prior | 8.02 | 0.123 | 9.2 |
| Yelp2018 | `yelp2018__mf__graphhash__directau__s0` | raw | 2.41 | 1.469 | 61.0 |
| Yelp2018 | `yelp2018__mf__graphhash__directau__s0` | cosine | 1.97 | 2.301 | 96.2 |
| Yelp2018 | `yelp2018__mf__graphhash__directau__s0` | cos_prior | 5.68 | 0.290 | 21.7 |
| Yelp2018 | `yelp2018__mf__graphhash__bpr__s0` | raw | 3.78 | 0.800 | 22.9 |
| Yelp2018 | `yelp2018__mf__graphhash__bpr__s0` | cosine | 2.03 | 0.919 | 49.3 |
| Yelp2018 | `yelp2018__mf__graphhash__bpr__s0` | raw_prior | 4.80 | 0.275 | 7.3 |
| AmazonBook | `amazonbook__mf__full__directau__s0` | raw | 9.57 | 5.407 | 72.4 |
| AmazonBook | `amazonbook__mf__full__directau__s0` | cosine | 8.70 | 7.344 | 93.4 |
| AmazonBook | `amazonbook__mf__full__directau__s0` | cos_prior | 11.05 | 2.698 | 44.3 |
| AmazonBook | `amazonbook__mf__full__bpr__s0` | raw | 8.37 | 2.108 | 53.3 |
| AmazonBook | `amazonbook__mf__full__bpr__s0` | cosine | 6.57 | 5.821 | 98.3 |
| AmazonBook | `amazonbook__mf__full__bpr__s0` | raw_prior | 8.37 | 2.108 | 53.3 |
| AmazonBook | `amazonbook__mf__graphhash__directau__s0` | raw | 4.63 | 5.044 | 92.3 |
| AmazonBook | `amazonbook__mf__graphhash__directau__s0` | cosine | 4.64 | 5.090 | 93.2 |
| AmazonBook | `amazonbook__mf__graphhash__directau__s0` | cos_prior | 6.33 | 3.781 | 80.7 |
| AmazonBook | `amazonbook__mf__graphhash__bpr__s0-legacy` | raw | 4.10 | 3.132 | 44.4 |
| AmazonBook | `amazonbook__mf__graphhash__bpr__s0-legacy` | cosine | 2.53 | — | — |
| AmazonBook | `amazonbook__mf__graphhash__bpr__s0-legacy` | raw_prior | 4.81 | 2.661 | 34.3 |

**Table 30. Pilot tail analysis (Stage 1–2)** *(verbatim from RESULTS.md lines 286–292)*

| Case | Main capped Recall@20, baseline → prior (%) | Head share of gross added hits (%) | Conditional tail recall Δ (points) | Tail contribution Δ (points) | Catalog coverage, baseline → prior (%) |
|---|---|---|---|---|---|
| Gowalla A1 DirectAU | 7.5666 → 13.3047 | 96.78 | −6.8077 | −3.7239 | 95.19 → 28.40 |
| Yelp A4 DirectAU | 1.9675 → 5.6823 | 98.48 | −2.0107 | −0.9009 | 96.25 → 21.65 |
| Yelp A5 BPR | 3.7845 → 4.8007 | 96.42 | −0.5250 | −0.2543 | 22.88 → 7.34 |
| AmazonBook legacy DirectAU | 4.6264 → 6.3275 | 89.45 | −1.3024 | −0.6248 | 93.11 → 80.68 |
| AmazonBook legacy BPR | 4.1128 → 4.8103 | 74.64 | −0.4715 | −0.2068 | 44.39 → 34.33 |

Head share of added hits is 89–98% for DirectAU. The original pilot prediction (tail change ≤ 0.5 points) **failed on all three datasets** for DirectAU. Cheng (2026) reports the same coverage direction on full tables (0.985 → 0.726).

### 8.2 Storage (E7)

**Table 31. Float32 prior overhead and the uint8 option.**

| Dataset | Float32 prior (B) | GraphHash table (B) | Float32 overhead | uint8 + float32 scale (B) |
|---|---|---|---|---|
| Gowalla | 163,924 | 2,969,088 | 5.521% | 40,985 |
| Yelp2018 | 152,192 | 3,906,560 | 3.896% | 38,052 |
| AmazonBook | 366,396 | 5,020,672 | 7.298% | 91,603 |

**Table 32. Does uint8 change Recall by ≤ 0.02 pp?** Eleven modern checkpoints, validation and test, at the same frozen β *(verbatim from RESULTS.md lines 418–431)*

| Run | Primary repaired mode | Validation Δ | Test Δ | Tolerance |
|---|---|---|---|---|
| `gowalla__mf__graphhash__directau__s0` | cos_prior | -0.026409 | 0.024596 | FAIL |
| `gowalla__mf__graphhash__bpr__s0-p500` | raw_prior | 0.019056 | 0.004739 | PASS |
| `gowalla__mf__graphhash__directau__s1` | cos_prior | -0.023500 | -0.013114 | FAIL |
| `gowalla__mf__graphhash__bpr__s1` | raw_prior | -0.001381 | 0.023872 | FAIL |
| `gowalla__mf__graphhash__directau__s2` | cos_prior | 0.001384 | -0.018167 | PASS |
| `gowalla__mf__graphhash__bpr__s2` | raw_prior | 0.010358 | 0.001553 | PASS |
| `gowalla__mf__graphhash__directau__s0-initmatch` | cos_prior | -0.015783 | 0.006350 | PASS |
| `yelp2018__mf__graphhash__directau__s0` | cos_prior | 0.008968 | 0.015300 | PASS |
| `yelp2018__mf__graphhash__bpr__s0` | raw_prior | 0.001403 | -0.011454 | PASS |
| `amazonbook__mf__graphhash__directau__s0` | cos_prior | -0.015991 | -0.001621 | PASS |
| `amazonbook__mf__graphhash__directau__s0-initmatch` | cos_prior | -0.000530 | -0.007753 | PASS |
| `amazonbook__mf__graphhash__bpr__s0-legacy` | raw_prior | missing | 0.001169 | missing_gpu |

Verdict: **FAIL**. Three modern checkpoints fail on at least one split (Gowalla DAU s0, DAU s1, BPR s1); eight pass; the legacy checkpoint lacks validation. The "one byte per item" claim is dropped; the stated cost is float32. S4-RESCORE adds AmazonBook B9: val −0.0011 pp, test +0.0021 pp (within tolerance, one checkpoint).

### 8.3 Weighted recall (E9)

Per-user weighted recall with weights (1 + c_i)^(−1/2) over **all** held-out positives (equal weights recover standard recall, not the capped metric).

**Table 33. Original vs weighted-validation-retuned β** *(verbatim from RESULTS.md lines 449–455)*

| Pair | Original β D/B | Original weighted D/B | Retuned β D/B | Retuned weighted D/B | Retuned Δ pp | Ordering |
|---|---|---|---|---|---|---|
| Gowalla s0 | 0.2/0.5 | 9.636/8.138 | 0.100/0.200 | 10.821/8.580 | 2.241 | PASS |
| Gowalla s1 | 0.2/0.5 | 9.653/8.002 | 0.100/0.200 | 10.902/8.562 | 2.340 | PASS |
| Gowalla s2 | 0.2/0.5 | 9.688/8.134 | 0.100/0.200 | 10.898/8.763 | 2.135 | PASS |
| Yelp2018 s0 | 0.2/0.5 | 3.689/2.873 | 0.200/0.500 | 3.689/2.873 | 0.816 | PASS |
| AmazonBook s0 | 0.1/0.2 | 5.543/3.979 | 0.050/missing | 5.531/missing | missing | INCONCLUSIVE |

S4-RESCORE closes the AmazonBook cell: after weighted-validation retuning, DirectAU 5.53 (β 0.05) vs BPR 4.04 (β 0.05); DirectAU still wins.

---

## 9. Negative and failed results

**Table 34. Failed, partial and reported-only predictions** (all with numbers).

| ID | Prediction | Observed | Verdict |
|---|---|---|---|
| PR-A8 | Full BPR Gowalla ∈ [15.4, 15.8] | 16.55; our Full BPR is stronger than the published 15.62. Converged `s0-long` gives the same 16.55 | FAIL (outside range) |
| PR-Y / Y3 | Yelp \|raw − cos\| < 0.3 and ρ ≤ 0.4 | gap 0.44, ρ 0.52 (Yelp buckets keep more popularity: η² 0.484) | FAIL (both parts) |
| PR-E5 | tail-recall change ≤ 0.5 points (DirectAU) | −6.81 / −2.01 / −1.30 conditional tail change, failing on all three datasets | FAIL |
| S3-E7 | uint8 within 0.02 pp on every GraphHash checkpoint | 3 of 11 modern checkpoints fail | FAIL; float32 default |
| PR-E2e | decile \|Δ\| ≤ 0.5 / 0.3; replacement \|Δ\| ≤ 0.7 / 0.3 | decile −4.19 / −0.33; replacement +0.83 / +1.08 (favourable) | FAIL / FAIL (favourable) |
| S4-C1 | ρ_item rises with resolution (Spearman ≥ +0.8) | +0.70 (plateaus after 800) | FAIL |
| PR-M1 / M-iii | product rule ρ_item ≈ ρ_proxy·ρ_full ±0.10 | Gowalla 0.154 vs 0.27 (fail); AmazonBook 0.179 vs 0.189 and Random 0.225 vs 0.274 (ok) | mixed (exploratory) |
| S3-B / B-i, B-iv | range checks on Random / DoubleFreq | 6/8 and 6/9 | partial |
| PR-E9 (Amazon) | weighted-validation retuning, AmazonBook | missing at Stage 3 | inconclusive; closed in Stage 4 (S4-RESCORE) |
| S4-C3 | learned score within ±0.5 of post-hoc | see Table 35 | FAIL (code review pending) |

**Table 35. C3: learned residual score cos + β̂·p + r vs the post-hoc prior** (same runs). The learned β̂ is positive (0.90 / 0.93 / 0.93) but the jointly trained residual r hurts ranking badly. An implementation issue is not excluded, so report this as "the learned variant did not work in our implementation".

| Dataset | Learned score (cos + β̂·p + r) | Post-hoc cos+prior, same run | Post-hoc cos+prior, A1/A4/B9 | Learned β̂ | Raw (same run) | ρ_item | Stop | Run |
|---|---|---|---|---|---|---|---|---|
| Gowalla | 4.56 | 13.21 | 13.30 | 0.90 | 7.33 | 0.23 | patience; 30/45 | `gowalla__mf__graphhash__directau_resbias__s0` |
| Yelp2018 | 2.68 | 5.68 | 5.68 | 0.93 | 2.35 | 0.50 | patience; 30/35 | `yelp2018__mf__graphhash__directau_resbias__s0` |
| AmazonBook | 1.63 | 6.25 | 6.33 | 0.93 | 4.56 | 0.15 | patience; 20/35 | `amazonbook__mf__graphhash__directau_resbias__s0` |

**Learned item bias, legacy** (`gowalla__mf__graphhash__directau_bias__s0`): raw 7.38 vs 7.55; cos+prior 13.17. A learned bias does not restore raw ranking.

**Table 36. Invalid or superseded runs (never cite as results)** *(verbatim from RESULTS.md lines 196–201)*

| Run | Why |
|---|---|
| GraphHash × BPR, batch 1024 (5.31 / 5.45) | Wrong setup for BPR |
| P1 BPR β-sweep on the Kaggle CPU image (val 4.24 at β = 0) | CPU image numbers buckets differently; the GPU rerun gave the 2.3 numbers |
| P1 GPU, first attempt | Google Drive HTTP 500 |
| AmazonBook BPR, patience 100 (3.93) | Stopped on an early spike |

Also excluded: `gowalla__lightgcn__full__bpr__s0-2gpu` (status `failed`, first 2-GPU attempt; superseded by `-2gpu-v2`).

---

## 10. Reproduction and control runs

**Table 37. Reproduction, smoke, determinism, initialization and legacy controls.**

| Purpose | Run | Raw | Cosine | Cos + prior (β) | Raw + prior (β) | ρ_item | Stop (best/ran) | Status |
|---|---|---|---|---|---|---|---|---|
| GraphHash BPR reproduction (paper: 9.30) | `gowalla__mf__graphhash__bpr__s0` | 9.34 | 6.08 | 10.22 (β 0.1) | 11.24 (β 0.5) | 0.054 | patience; 330/430 | valid |
| Smoke test, 2 epochs (A0) | `x__gowalla__mf__graphhash__directau__s0` | 6.94 | 7.34 | 12.99 (β 0.2) | 12.96 (β 0.5) | 0.231 | max_epochs; 2/2 | capped |
| Smoke test, Yelp GraphHash BPR, 20 epochs | `x__yelp2018__mf__graphhash__bpr__s0` | 1.60 | 1.61 | 4.66 (β 0.2) | 4.36 (β 0.1) | 0.393 | max_epochs; 20/20 | capped |
| Co-location determinism A | `x__coloc_a` | 6.73 | 7.07 | 12.57 (β 0.2) | 12.32 (β 0.2) | 0.209 | max_epochs; 1/1 | capped |
| Co-location determinism B | `x__coloc_b` | 6.73 | 7.07 | 12.57 (β 0.2) | 12.32 (β 0.2) | 0.209 | max_epochs; 1/1 | capped |
| Dedup variant s0 | `x__gowalla__mf__graphhash__directau_dedup__s0` | 7.53 | — | — | — | — | fixed_epochs; 30/30 | exploratory |
| Dedup variant s1 | `x__gowalla__mf__graphhash__directau_dedup__s1` | 7.65 | — | — | — | — | fixed_epochs; 30/30 | exploratory |
| Dedup variant s2 | `x__gowalla__mf__graphhash__directau_dedup__s2` | 7.58 | — | — | — | — | fixed_epochs; 30/30 | exploratory |
| I1 matched-init GraphHash DirectAU (Gowalla) | `gowalla__mf__graphhash__directau__s0-initmatch` | 7.54 | 7.54 | 13.35 (β 0.2) | 10.54 (β 5) | 0.274 | patience; 50/80 | converged |
| I2 matched-init GraphHash DirectAU (AmazonBook) | `amazonbook__mf__graphhash__directau__s0-initmatch` | 4.59 | 4.63 | 6.30 (β 0.1) | 6.33 (β 0.5) | 0.188 | patience; 35/60 | converged |
| Legacy 30-epoch GraphHash DirectAU s0 | `gowalla__mf__graphhash__directau__s0-legacy30` | 7.56 | 7.58 | 13.26 (β 0.2) | 11.43 (β 5) | 0.271 | fixed_epochs; 30/30 | valid-short |
| Legacy 30-epoch GraphHash DirectAU s1 | `gowalla__mf__graphhash__directau__s1-legacy30` | 7.57 | 7.56 | 13.31 (β 0.2) | 11.49 (β 5) | 0.274 | fixed_epochs; 30/30 | valid-short |
| Legacy 30-epoch GraphHash DirectAU s2 | `gowalla__mf__graphhash__directau__s2-legacy30` | 7.53 | 7.55 | 13.30 (β 0.2) | — | 0.270 | fixed_epochs; 30/30 | valid-short |
| Legacy 30-epoch Random DirectAU | `gowalla__mf__random__directau__s0-legacy30` | 0.47 | 0.31 | 3.85 (β 0.2) | 3.69 (β 2) | 0.274 | fixed_epochs; 20/30 | valid-short |
| Legacy 30-epoch DoubleFreq DirectAU | `gowalla__mf__dfreq__directau__s0-legacy30` | 3.23 | 2.18 | 5.15 (β 0.2) | 4.90 (β 2) | 0.029 | fixed_epochs; 30/30 | valid-short |
| Legacy Full DirectAU, time-capped at 130 epochs | `gowalla__mf__full__directau__s0-legacy` | 17.90 | 13.00 | 19.54 (β 0.1) | — | 0.815 | time_cap; 120/130 | capped |
| Legacy Full DirectAU AmazonBook | `amazonbook__mf__full__directau__s0-legacy` | 9.57 | 8.70 | 11.05 (β 0.1) | 10.77 (β 0.2) | 0.786 | patience; 25/35 | valid |
| Legacy GraphHash DirectAU AmazonBook, time-capped (41 epochs) | `amazonbook__mf__graphhash__directau__s0-legacy` | 4.63 | 4.64 | 6.33 (β 0.1) | 6.29 (β 0.5) | 0.189 | time_cap; 40/41 | capped |
| Legacy GraphHash DirectAU + learned item bias (ablation) | `gowalla__mf__graphhash__directau_bias__s0` | 7.38 | 7.48 | 13.17 (β 0.2) | 11.77 (β 5) | 0.219 | fixed_epochs; 25/30 | ablation |

- GraphHash BPR on Gowalla reproduces the paper: 9.34 here vs 9.30 published.
- I1 / I2 (matched initial norms; 2.64× norm difference on Gowalla): raw −0.01 and cos+prior +0.05 vs A1 on Gowalla; |I2 − B9| 0.04 / 0.03 on AmazonBook. Results are insensitive to initial norms. Controlled Full-vs-GraphHash pairs: Gowalla 18.02 vs 7.54; AmazonBook 9.57 vs 4.63.
- Co-location determinism: `x__coloc_a` and `x__coloc_b` give identical outputs (6.73 / 7.07 / 12.57 / 12.32).
- The legacy 30-epoch GraphHash DAU rows were already converged: the validation curve is flat from about epoch 20 (7.40 → 7.48).

---

## 11. Pre-registration outcome board

*(verbatim from RESULTS.md lines 19–49; this board is the single source for the registry in the paper appendix)*

| ID | Short prediction | Label | Observed | Verdict |
|---|---|---|---|---|
| PR-TIE | ≥ 50% users with a boundary tie; random vs default key \|Δ\| ≤ 0.2; tie-break-only gain (lexicographic) reported | P | 87% users tied (14.7 items); key Δ +0.04–0.06; tie-break-only = 12% of gain | PASS |
| PR-DEC | 2×2 at fixed β\*: main effects, interaction, Shapley shares; within-bucket shuffle keeps ≤ 70% of the gain | P | Shapley μ 0.29 / ε 0.71; shuffle keeps −6% | PASS |
| PR-E2e | Decile shuffle \|Δ\| small; global shuffle drop large (Full models) | P | decile −4.19; global −11.47; replacement +0.83 (favourable) | decile FAIL, global PASS, repl. FAIL (favourable) |
| PR-M1 | Theorem check holds; GraphHash η² ≤ 0.5 | P | η² GraphHash 0.36–0.48; bound holds | M-i PASS, M-ii PASS, M-iii mixed [X] |
| PR-A6 | Popularity-only ≥ 2 points below the repair (Yelp, AmazonBook) | P | pop-only 2.97 / 1.68 / 0.96 vs repair 13.29 / 5.68 / 6.33 | PASS (3/3) |
| PR-N1 | Novelty verdict | — | Deep sweep 18 Oct: kill rule not triggered; repair shape has precedent (MEmCom, Hash Embeddings, CoRRe); DirectAU-hashing problem reported by Shiao et al. + GraphHash without mechanism | recommended G1.d PASS, narrowed contribution (human decides); full texts verified 22:56 (Cheng eq. 53 = our score form on full tables; MEmCom Alg. 2 = learned per-item scale) |
| PR-A1 | GraphHash-DirectAU raw ≤ 9.04 **and** stopped by patience (capped = INCONCLUSIVE) | I | raw 7.55; cos+prior 13.30; stop = patience (ep 55) | PASS |
| PR-I1 | Init-sensitivity control: within ±0.3 of A1, both converged (not a sharing test) | P | raw Δ −0.01; cos+prior Δ +0.05 | PASS |
| PR-A8 | Full-BPR ∈ [15.4, 15.8] | I | 16.55, capped at max epochs | FAIL (not converged) |
| PR-Y | Yelp: Y1 (I), Y3 (P), Y4 repair wins (P), Y2 (X) | mixed | Y1 2.41<3.78; Y3 gap 0.44, ρ 0.52; Y4 +0.88 [0.74,1.02]; Y2 E 2.70 | Y1 PASS, Y3 FAIL, Y4 PASS, Y2 reported |
| PR-H | ρ_bucket ≥ 0.6; lengths only grow; ρ_dir BPR > DirectAU | H | ρ_bucket 0.925/0.944; max r² drop 0.15%; ρ_dir BPR>DAU 2/2 | H-pool, H-Adam, H-dir PASS |
| PR-E5 | Head share ≥60%; tail recall change ≤0.5 points | informed operationalization | Head share 89.45–98.48% for DirectAU; conditional tail losses 1.30–6.81 points | partial (Stage 1–2); DirectAU tail prediction FAIL |
| PR-E7 | uint8 prior changes Recall by ≤0.02 points | Stage 1 intention; Stage 2 sensitivity | Original GraphHash test means pass; legacy validation missing; modern A1 −0.0264 val / +0.0246 test | partial (Stage 1–2); universal one-byte claim unsupported |
| PR-E9 | Reweighted-validation β shrinks or stays; Gowalla K2 ordering kept | informed follow-up | Gowalla A1 β .2→.1; weighted recall 9.6356→10.8214; legacy BPR retuning unavailable | partial (Stage 1–2); fully retuned ordering INCONCLUSIVE |
| PR-E8 final | Repaired DAU > repaired BPR, three datasets and every Gowalla seed | I | All five Recall CIs >0; retained separate NDCG inference | PASS; alternate Amazon paired CI missing |
| S3-E5 final | Repaired GH DAU tail < Full DAU raw tail | I | 7/7 variants; 89/90 mode records available | PASS means tail cost; legacy cosine missing |
| S3-E7 final | Every GH checkpoint within 0.02pp on val and test | I | 3 modern failures; 8 modern passes; 1 legacy missing validation | FAIL; float32 default |
| PR-E9 final | Weighted-validation β shrinks/stays; corrected ordering retained | I | Gowalla 3 seeds and Yelp PASS; Amazon retuning missing | partial; Amazon INCONCLUSIVE |
| S3-B, D7 | Main grid and decay checks | I/H | See Stage 3 outcome block; wd0 persistence under this setup | Report failures; no universal loss reversal |
| S3-A2, A8r, B9, B5–B8, I2 | Stage 3 range predictions | I | all inside their ranges | PASS |
| S4-SEED | Yelp / AmazonBook seeds 1–2 in range, every Δ CI > 0 | I | Δ +0.83 to +1.54 | PASS; PR-E8 now holds on all 9 seed pairs |
| S4-AMZBPR | AmazonBook Full BPR converges, raw ≥ 8.37 | I | 8.38, stopped on patience | PASS |
| S4-RESCORE | alternative-tie CIs > 0; uint8; weighted | I/P | +1.51 to +1.52; within 0.002 pp; 5.53 vs 4.04 | PASS (closes the AmazonBook gaps in PR-E8 / PR-E9) |
| S4-C3 | learned score within ±0.5 of post-hoc | I | 4.56 vs 13.21 (Gowalla), 2.68 vs 5.68, 1.63 vs 6.25 | FAIL (code review pending) |
| S4-LGN | LightGCN L-i, L-ii, L-iii | H | all three hold; 5 rows capped at 4 h | PASS (budget-limited rows) |
| S4-DG | DoubleGraph (BACO substitute) repair; DirectAU > BPR | I | 10.91 vs 7.27; 5.52 vs 2.78 | PASS |
| S4-C1 | gain falls and ρ_item rises with resolution | P | Spearman −0.90 / +0.70 | gain PASS, ρ FAIL |
| S4-C4 | equal norms (exploratory) | X | Full 12.77 → 19.32; GraphHash 13.27 | as predicted |
| S5-LGNFULL | LightGCN Gowalla Full, 11.5 h on 2 GPUs | H | both converged; BPR 18.48 > DAU 18.02 raw; repaired DAU 19.76 > BPR 18.46 | PASS (no 2-GPU speedup) |

Labels: **P** = prospective prediction, **I** = informed (made after seeing pilot results), **H** = hypothesis, **X** = exploratory. The `PR-N1` row is a novelty verdict, not a numeric prediction.

---

## 12. Novelty verification

Question (kill rule): does any paper show that sharing/hashing destroys item-level popularity information **and** repairs it with a per-item popularity term? Verdict: **not triggered**; recommended G1.d PASS with a narrowed contribution (diagnosis, decomposition, capacity bound, explanation of the flip; the post-hoc term is the instrument, not the novelty). Full texts were read after network access was opened. Details: `results/N1_novelty.md`.

*(verbatim from RESULTS.md lines 535–540, the closest prior work)*

- **MEmCom** (Pansare et al., MLSys 2022): hashed table + one learned weight per entity.
- **Hash Embeddings** (Svenstrup et al., NeurIPS 2017): per-token importance parameters.
- **Proxy-based items** (Seol et al., WSDM 2024): bias terms for frequent items.
- **CoRRe** (Kim, Loveland et al., arXiv 2608.19665): training-free popularity magnitudes, no sharing.
- **Shiao et al.** (WWW 2025 Companion; GraphHash ref [45]): a second, mechanism-free report that DirectAU suits hashing poorly.
- **Cheng** (arXiv 2606.21275): log p(i) term; prior adjustment lowers coverage 0.985 → 0.726, which parallels our E5.

*(verbatim from RESULTS.md lines 546–551, the full-text verification)*

- **Cheng 2606.21275:** no hashing or sharing. Eq. (53) r = hᵀe_i + β·log π_i, with β tuned for accuracy vs coverage. Coverage 0.985 → 0.726.
- **PRISM:** no hashing or sharing anywhere; no optimizer named; λ = 0 DirectAU is qualitative only.
- **Shiao et al.:** "BPR outperforms DirectAU on OOV users across all three datasets", with no mechanism.
- **MEmCom Algorithm 2:** U[i mod m] ⊙ V[i] with V ∈ R^{v×1}. Under DirectAU or cosine scoring, ∂L/∂V[i] = 0 for V[i] > 0 (PLAN Prop 1), so it cannot carry popularity there.
- **CoRRe:** no sharing.
- **Citers** (Semantic Scholar): no new overlap. New sharing method GDE (SIGIR 2026), cite.

Not read in full: Google Scholar was not queried and the citer lists were not refreshed after 18 Oct.

---

## 13. Issues found while compiling

These are things the paper author should know before the numbers freeze. Items 1–3 change what you can cite from the CSV.

1. **`master_metrics_v2.csv` stops at Stage 3.** It was last generated 18 Oct 11:44 and holds 36 of the 73 modern runs. The other 37 (every Stage 4–5 run: all LightGCN, DoubleGraph, the resolution sweep, equal-norm, C3, the Yelp/AmazonBook seeds 1–2, AmazonBook Full BPR `s0-long`) exist only in `runs/*/metrics.json` and `RESULTS.md` prose. The outline's "from CSV" plan for Tab 2 and Appendix B would silently omit them. `results/run_ledger_all.csv` (new) covers all 93 runs.
2. **Several analyses use legacy checkpoints.** Decomposition, ties, E2e and the E7 pilot were run on the Stage 1 legacy checkpoints, so for example the E2e baseline is 17.90 (legacy, time-capped at 130 epochs) while Table 6 has 18.02 (modern).
3. **ρ epoch convention.** The CSV and the ledger take ρ at the best-checkpoint epoch; `RESULTS.md` quotes last-evaluated-epoch values for the quintile and H-pool text. They agree for DirectAU GraphHash (0.925 / 0.944) but differ for long-trained BPR (Yelp GraphHash BPR ρ_bucket 0.822 at best vs 0.916 at last). State which one the paper uses.
4. **LightGCN ρ columns are not comparable with MF.** In `exp_run.py` (`norm_stats`), ρ_item uses the propagated item vectors while ρ_bucket uses the raw shared-table rows. The low LightGCN ρ_bucket at the best checkpoint (Gowalla DAU 0.11, BPR −0.12; Yelp DAU −0.20, BPR −0.48) does not test pooling; do not use it as evidence for or against the mechanism on LightGCN.
5. **Random × BPR ρ_bucket has no stable sign.** It oscillates during training (Gowalla: +0.24, −0.02, +0.38, +0.27, +0.10, −0.05, −0.14, −0.22 across the run), so the best-checkpoint value (+0.39 / +0.16 / +0.30) and the last-epoch value (−0.22 / −0.22 / −0.20) disagree in sign. Random × DAU is stable at 0.99–1.00 and GraphHash BPR rises steadily (Gowalla 0.81 → 0.89). If the paper quotes ρ_bucket for BPR, say which epoch.
6. **AmazonBook Full BPR.** The `RESULTS.md` §1 matrix uses the capped `s0` (8.37); the converged `s0-long` (8.38) supersedes it and is what Table 6 uses.
7. **AmazonBook GraphHash BPR s0 is a legacy run** with no config or checkpoint in `runs/` (raw 4.10, raw+prior 4.81, random-key tie order). Its seed-1/2 siblings are modern.
8. **`prompts/g3_abstract_numbers.md` predates Stage 4.** It still says C3 was not run and that AmazonBook alternates are unavailable. Stage 4 superseded both; the abstract is already submitted.
9. **Small wording and rounding items.**
   - "79 items share an alias pair" is 79 collisions (158 items) for Gowalla DoubleFreq; Yelp has 887 (1,773 items) and AmazonBook 145 (290 items).
   - S4-C1's gain at resolution 200 is 5.76 from unrounded values (5.75 printed).
   - 13.29 in PR-A6 is the legacy three-seed mean; the modern A1 is 13.30.
10. **Missing cells (not zeros).**
    - Legacy AmazonBook BPR cosine (E5).
    - Quantized validation for the legacy E7 checkpoints.
    - AmazonBook BPR weighted retuning in `E9_final.json` (closed by S4-RESCORE, not by the JSON).
    - Alternate-tie-policy paired CIs for AmazonBook (only seed 0 has stored per-user full-prior predictions).
11. **Verified, previously open.** The table ratios (16.37 / 21.89 / 13.60%) the outline marked "verify against the mapping files" match the stored `bucket_info`.

**Not tested** (carry into Limitations): GDE, BACO itself, CTR models, DNA and MEmCom baselines, and a code review of the C3 learned variant.

---

## Appendix A: complete run ledger

All 93 runs: 73 modern run directories (one failed) plus 20 legacy or baseline rows from `master_metrics_v2.csv`. Machine-readable copy: `results/run_ledger_all.csv`. Sorted by dataset, then run ID. Stage S0–S5 is the Kaggle session folder; legacy rows show their original session slug. β is shown for the two prior modes; a dash means the cell was not scored or is not applicable (Full has no ρ_bucket).

| # | Run | Lineage | Stage / session | Stop (best/ran) | Train min | Raw | Cosine | Cos + prior (β) | Raw + prior (β) | ρ_item | ρ_bucket | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `gowalla__lightgcn__full__bpr__s0` | modern | S4 · pc-s4c-1018 | time_cap; 340/345 | 240 | 18.44 | 13.65 | 18.89 (β 0.1) | 18.44 (β 0) | 0.864 | — | capped |
| 2 | `gowalla__lightgcn__full__bpr__s0-2gpu` | modern | S5 · pc-s5b-1018 | — | — | — | — | — | — | — | — | failed |
| 3 | `gowalla__lightgcn__full__bpr__s0-2gpu-v2` | modern | S5 · pc-s5d-1018 | patience; 410/460 | 320 | 18.48 | 13.65 | 19.04 (β 0.1) | 18.46 (β 0.01) | 0.869 | — | converged |
| 4 | `gowalla__lightgcn__full__directau__s0` | modern | S4 · pc-s4d-1018 | time_cap; 182/182 | 241 | 17.94 | 13.32 | 19.78 (β 0.1) | 17.99 (β 0.1) | 0.655 | — | capped |
| 5 | `gowalla__lightgcn__full__directau__s0-2gpu-v2` | modern | S5 · pc-s5e-1018 | patience; 215/215 | 285 | 18.02 | 13.33 | 19.76 (β 0.1) | 18.02 (β 0.02) | 0.704 | — | converged |
| 6 | `gowalla__lightgcn__graphhash__bpr__s0` | modern | S4 · pc-s4d-1018 | patience; 100/150 | 98 | 16.22 | 10.52 | 15.79 (β 0.1) | 16.27 (β 0.1) | 0.377 | -0.121 | converged |
| 7 | `gowalla__lightgcn__graphhash__directau__s0` | modern | S4 · pc-s4a-1018 | patience; 120/150 | 193 | 11.29 | 10.95 | 17.06 (β 0.1) | 12.38 (β 0.2) | 0.145 | 0.106 | converged |
| 8 | `gowalla__mf__dfreq__bpr__s0` | modern | S3 · pc-s3d-1017 | patience; 450/950 | 3 | 3.93 | 2.95 | 4.06 (β 0.1) | 4.11 (β 0.5) | 0.100 | — | converged |
| 9 | `gowalla__mf__dfreq__directau__s0` | modern | S3 · pc-s3a-1017 | patience; 65/80 | 51 | 3.22 | 2.20 | 5.11 (β 0.2) | 4.87 (β 2) | 0.028 | — | converged |
| 10 | `gowalla__mf__dfreq__directau__s0-legacy30` | legacy | gh-exp-p3-dfreq-dau-1017 | fixed_epochs; 30/30 | 19 | 3.23 | 2.18 | 5.15 (β 0.2) | 4.90 (β 2) | 0.029 | — | valid-short |
| 11 | `gowalla__mf__doublegraph__bpr__s0` | modern | S4 · pc-s4d-1018 | patience; 30/530 | 2 | 6.24 | 4.49 | 6.97 (β 0.1) | 7.27 (β 0.5) | 0.146 | — | converged |
| 12 | `gowalla__mf__doublegraph__directau__s0` | modern | S4 · pc-s4c-1018 | patience; 5/35 | 22 | 3.54 | 5.84 | 10.91 (β 0.1) | 7.38 (β 5) | -0.090 | — | converged |
| 13 | `gowalla__mf__full__bpr__s0` | modern | pc-x1-1017 | max_epochs; 2790/3000 | 9 | 16.55 | 11.96 | 16.71 (β 0.05) | 17.06 (β 0.5) | 0.770 | — | capped |
| 14 | `gowalla__mf__full__bpr__s0-long` | modern | S3 · pc-s3c-1017 | patience; 2790/3290 | 10 | 16.55 | 11.96 | 16.71 (β 0.05) | 17.06 (β 0.5) | 0.770 | — | converged |
| 15 | `gowalla__mf__full__directau__s0` | modern | S3 · pc-s3d-1017 | patience; 160/170 | 112 | 18.02 | 13.06 | 19.57 (β 0.1) | 18.02 (β 0.01) | 0.815 | — | converged |
| 16 | `gowalla__mf__full__directau__s0-legacy` | legacy | graphhash-loss-sharing-repro-1016 | time_cap; 120/130 | 90 | 17.90 | 13.00 | 19.54 (β 0.1) | — | 0.815 | — | capped |
| 17 | `gowalla__mf__full__directau__s0-wd0` | modern | S3 · pc-s3d-1017 | patience; 15/45 | 30 | 16.12 | 12.61 | 19.08 (β 0.1) | 17.89 (β 2) | 0.744 | — | converged |
| 18 | `gowalla__mf__full__directau__s0-wd1e-4` | modern | S3 · pc-s3d-1017 | patience; 30/35 | 23 | 1.41 | 2.75 | 8.69 (β 0.1) | 6.76 (β 0.05) | 0.223 | — | converged |
| 19 | `gowalla__mf__full__directau_eqnorm__s0` | modern | S4 · pc-s4c-1018 | patience; 25/55 | 36 | 12.77 | 12.77 | 19.32 (β 0.1) | 19.01 (β 5) | -0.024 | — | converged |
| 20 | `gowalla__mf__graphhash-r1600__bpr__s0` | modern | S4 · pc-s4b-1018 | patience; 1090/1590 | 4 | 11.57 | 7.98 | 12.52 (β 0.1) | 11.77 (β 0.2) | 0.491 | 0.971 | converged |
| 21 | `gowalla__mf__graphhash-r1600__directau__s0` | modern | S4 · pc-s4b-1018 | time_cap; 185/185 | 120 | 8.62 | 7.49 | 14.02 (β 0.2) | 10.90 (β 5) | 0.573 | 0.950 | capped |
| 22 | `gowalla__mf__graphhash-r3200__bpr__s0` | modern | S4 · pc-s4c-1018 | patience; 1290/1790 | 5 | 12.61 | 8.90 | 13.75 (β 0.1) | 12.76 (β 0.1) | 0.458 | 0.966 | converged |
| 23 | `gowalla__mf__graphhash-r3200__directau__s0` | modern | S4 · pc-s4c-1018 | patience; 165/175 | 111 | 10.87 | 9.17 | 15.86 (β 0.1) | 13.20 (β 5) | 0.535 | 0.930 | converged |
| 24 | `gowalla__mf__graphhash-r400__bpr__s0` | modern | S4 · pc-s4a-1018 | patience; 2100/2600 | 7 | 9.79 | 6.06 | 10.77 (β 0.1) | 10.96 (β 0.2) | 0.313 | 0.943 | converged |
| 25 | `gowalla__mf__graphhash-r400__directau__s0` | modern | S4 · pc-s4a-1018 | patience; 65/80 | 49 | 7.49 | 7.31 | 13.15 (β 0.1) | 10.02 (β 5) | 0.443 | 0.933 | converged |
| 26 | `gowalla__mf__graphhash-r800__bpr__s0` | modern | S4 · pc-s4a-1018 | patience; 1520/2020 | 6 | 10.42 | 6.80 | 11.63 (β 0.1) | 11.16 (β 0.2) | 0.486 | 0.965 | converged |
| 27 | `gowalla__mf__graphhash-r800__directau__s0` | modern | S4 · pc-s4a-1018 | patience; 170/170 | 106 | 7.86 | 7.21 | 13.15 (β 0.2) | 9.83 (β 5) | 0.562 | 0.945 | converged |
| 28 | `gowalla__mf__graphhash__bpr__s0` | legacy | graphhash-fast-bpr-author-1016 | patience; 330/430 | 2 | 9.34 | 6.08 | 10.22 (β 0.1) | 11.24 (β 0.5) | 0.054 | — | valid |
| 29 | `gowalla__mf__graphhash__bpr__s0-b1024` | legacy | graphhash-fast-bpr-1016 | fixed_epochs; 30/30 | 1 | 5.31 | — | — | — | — | — | invalid |
| 30 | `gowalla__mf__graphhash__bpr__s0-b1024-pat10` | legacy | graphhash-fast-bpr-long-1016 | patience; 5/15 | 0 | 5.45 | — | — | — | — | — | invalid |
| 31 | `gowalla__mf__graphhash__bpr__s0-p500` | modern | S3 · pc-s3c-1017 | patience; 460/960 | 3 | 9.37 | 5.86 | 9.96 (β 0.1) | 11.31 (β 0.5) | 0.066 | 0.862 | converged |
| 32 | `gowalla__mf__graphhash__bpr__s1` | modern | S3 · pc-s3c-1017 | patience; 780/1280 | 4 | 9.39 | 5.41 | 9.73 (β 0.1) | 11.19 (β 0.5) | 0.080 | 0.885 | converged |
| 33 | `gowalla__mf__graphhash__bpr__s2` | modern | S3 · pc-s3c-1017 | patience; 1540/2040 | 6 | 9.41 | 5.30 | 9.68 (β 0.1) | 11.27 (β 0.5) | 0.089 | 0.894 | converged |
| 34 | `gowalla__mf__graphhash__directau__s0` | modern | S2 · pc-s2a-1017 | patience; 55/60 | 37 | 7.55 | 7.57 | 13.30 (β 0.2) | 10.33 (β 5) | 0.275 | 0.925 | converged |
| 35 | `gowalla__mf__graphhash__directau__s0-initmatch` | modern | S2 · pc-s2a-1017 | patience; 50/80 | 49 | 7.54 | 7.54 | 13.35 (β 0.2) | 10.54 (β 5) | 0.274 | 0.927 | converged |
| 36 | `gowalla__mf__graphhash__directau__s0-legacy30` | legacy | gh-exp-p2-gh-dau-s0-1017 | fixed_epochs; 30/30 | 19 | 7.56 | 7.58 | 13.26 (β 0.2) | 11.43 (β 5) | 0.271 | — | valid-short |
| 37 | `gowalla__mf__graphhash__directau__s1` | modern | S3 · pc-s3b-1017 | patience; 50/80 | 50 | 7.61 | 7.59 | 13.32 (β 0.2) | 10.45 (β 5) | 0.277 | 0.924 | converged |
| 38 | `gowalla__mf__graphhash__directau__s1-legacy30` | legacy | gh-exp-p2-gh-dau-s1-1017 | fixed_epochs; 30/30 | 19 | 7.57 | 7.56 | 13.31 (β 0.2) | 11.49 (β 5) | 0.274 | — | valid-short |
| 39 | `gowalla__mf__graphhash__directau__s2` | modern | S3 · pc-s3b-1017 | patience; 50/60 | 39 | 7.59 | 7.58 | 13.35 (β 0.2) | 10.54 (β 5) | 0.274 | 0.924 | converged |
| 40 | `gowalla__mf__graphhash__directau__s2-legacy30` | legacy | graphhash-fast-dau-s012-1016 | fixed_epochs; 30/30 | 19 | 7.53 | 7.55 | 13.30 (β 0.2) | — | 0.270 | — | valid-short |
| 41 | `gowalla__mf__graphhash__directau_bias__s0` | legacy | gh-exp-p5-gh-dau-bias-1017 | fixed_epochs; 25/30 | 19 | 7.38 | 7.48 | 13.17 (β 0.2) | 11.77 (β 5) | 0.219 | — | ablation |
| 42 | `gowalla__mf__graphhash__directau_eqnorm__s0` | modern | S4 · pc-s4b-1018 | patience; 30/60 | 37 | 7.56 | 7.56 | 13.27 (β 0.2) | 11.55 (β 5) | -0.012 | -0.007 | converged |
| 43 | `gowalla__mf__graphhash__directau_resbias__s0` | modern | S4 · pc-s4d-1018 | patience; 30/45 | 28 | 7.33 | 7.45 | 13.21 (β 0.2) | 11.21 (β 5) | 0.233 | 0.883 | converged |
| 44 | `gowalla__mf__random__bpr__s0` | modern | S3 · pc-s3d-1017 | patience; 240/740 | 2 | 0.76 | 0.38 | 3.44 (β 0.2) | 3.42 (β 0.5) | 0.093 | 0.389 | converged |
| 45 | `gowalla__mf__random__directau__s0` | modern | S3 · pc-s3a-1017 | patience; 40/40 | 26 | 0.45 | 0.29 | 3.80 (β 0.2) | 3.70 (β 2) | 0.275 | 0.990 | converged |
| 46 | `gowalla__mf__random__directau__s0-legacy30` | legacy | gh-exp-p3-random-dau-1017 | fixed_epochs; 20/30 | 19 | 0.47 | 0.31 | 3.85 (β 0.2) | 3.69 (β 2) | 0.274 | — | valid-short |
| 47 | `gowalla__poponly__none__none__s0` | baseline | — | — | — | 2.97 | — | — | — | — | — | valid |
| 48 | `yelp2018__lightgcn__full__bpr__s0` | modern | S4 · pc-s4a-1018 | time_cap; 165/169 | 240 | 9.55 | 6.89 | 9.48 (β 0.1) | 9.55 (β 0.01) | 0.877 | — | capped |
| 49 | `yelp2018__lightgcn__full__directau__s0` | modern | S4 · pc-s4b-1018 | time_cap; 95/99 | 242 | 10.70 | 7.11 | 11.43 (β 0.1) | 10.82 (β 0.1) | 0.778 | — | capped |
| 50 | `yelp2018__lightgcn__graphhash__bpr__s0` | modern | S4 · pc-s4c-1018 | patience; 75/125 | 174 | 6.96 | 3.91 | 6.95 (β 0.1) | 6.96 (β 0.05) | 0.498 | -0.478 | converged |
| 51 | `yelp2018__lightgcn__graphhash__directau__s0` | modern | S4 · pc-s4b-1018 | time_cap; 100/102 | 242 | 4.71 | 5.09 | 9.12 (β 0.2) | 5.51 (β 0.5) | 0.132 | -0.197 | capped |
| 52 | `yelp2018__mf__dfreq__bpr__s0` | modern | S3 · pc-s3d-1017 | patience; 460/960 | 4 | 2.55 | 1.90 | 2.52 (β 0.1) | 2.58 (β 0.1) | 0.145 | — | converged |
| 53 | `yelp2018__mf__dfreq__directau__s0` | modern | S3 · pc-s3c-1017 | patience; 15/45 | 43 | 2.51 | 1.56 | 3.33 (β 0.2) | 3.17 (β 0.05) | 0.138 | — | converged |
| 54 | `yelp2018__mf__doublegraph__bpr__s0` | modern | S4 · pc-s4c-1018 | patience; 30/530 | 2 | 2.14 | 1.14 | 2.85 (β 0.2) | 2.78 (β 0.5) | 0.167 | — | converged |
| 55 | `yelp2018__mf__doublegraph__directau__s0` | modern | S4 · pc-s4d-1018 | patience; 45/45 | 43 | 1.12 | 2.17 | 5.52 (β 0.2) | 3.48 (β 5) | -0.057 | — | converged |
| 56 | `yelp2018__mf__full__bpr__s0` | modern | pc-x1-1017 | patience; 2150/2650 | 10 | 8.02 | 5.54 | 8.50 (β 0.1) | 8.02 (β 0) | 0.970 | — | converged |
| 57 | `yelp2018__mf__full__directau__s0` | modern | S2 · pc-s2b-1017 | patience; 110/140 | 135 | 11.10 | 6.97 | 11.26 (β 0.1) | 11.10 (β 0) | 0.912 | — | converged |
| 58 | `yelp2018__mf__graphhash__bpr__s0` | modern | pc-x1-1017 | patience; 360/860 | 3 | 3.78 | 2.03 | 4.82 (β 0.2) | 4.80 (β 0.5) | 0.190 | 0.822 | converged |
| 59 | `yelp2018__mf__graphhash__bpr__s1` | modern | S4 · pc-s4c-1018 | patience; 410/910 | 3 | 3.72 | 2.15 | 4.91 (β 0.2) | 4.83 (β 0.5) | 0.216 | 0.848 | converged |
| 60 | `yelp2018__mf__graphhash__bpr__s2` | modern | S4 · pc-s4a-1018 | patience; 390/890 | 3 | 3.71 | 2.09 | 4.96 (β 0.1) | 4.82 (β 0.5) | 0.210 | 0.838 | converged |
| 61 | `yelp2018__mf__graphhash__directau__s0` | modern | S2 · pc-s2b-1017 | patience; 50/60 | 57 | 2.41 | 1.97 | 5.68 (β 0.2) | 4.96 (β 1) | 0.520 | 0.944 | converged |
| 62 | `yelp2018__mf__graphhash__directau__s1` | modern | S4 · pc-s4d-1018 | patience; 20/45 | 42 | 2.48 | 2.02 | 5.69 (β 0.2) | 5.01 (β 1) | 0.520 | 0.947 | converged |
| 63 | `yelp2018__mf__graphhash__directau__s2` | modern | S4 · pc-s4d-1018 | patience; 50/60 | 56 | 2.42 | 2.05 | 5.65 (β 0.2) | 5.01 (β 1) | 0.520 | 0.944 | converged |
| 64 | `yelp2018__mf__graphhash__directau_resbias__s0` | modern | S4 · pc-s4a-1018 | patience; 30/35 | 34 | 2.35 | 1.99 | 5.68 (β 0.2) | 4.87 (β 1) | 0.503 | 0.938 | converged |
| 65 | `yelp2018__mf__random__bpr__s0` | modern | S3 · pc-s3c-1017 | patience; 420/920 | 4 | 0.55 | 0.38 | 2.15 (β 0.2) | 2.18 (β 1) | 0.075 | 0.159 | converged |
| 66 | `yelp2018__mf__random__directau__s0` | modern | S3 · pc-s3d-1017 | patience; 30/40 | 38 | 0.42 | 0.29 | 2.39 (β 0.2) | 2.41 (β 2) | 0.353 | 0.998 | converged |
| 67 | `yelp2018__poponly__none__none__s0` | baseline | — | — | — | 1.68 | — | — | — | — | — | valid |
| 68 | `amazonbook__mf__dfreq__bpr__s0` | modern | S3 · pc-s3c-1017 | patience; 420/920 | 9 | 1.28 | 1.01 | 1.35 (β 0.1) | 1.35 (β 0.5) | 0.136 | — | converged |
| 69 | `amazonbook__mf__dfreq__directau__s0` | modern | S3 · pc-s3b-1017 | patience; 15/40 | 74 | 1.51 | 1.12 | 2.13 (β 0.2) | 2.07 (β 0.5) | 0.026 | — | converged |
| 70 | `amazonbook__mf__full__bpr__s0` | modern | S3 · pc-s3b-1017 | max_epochs; 2690/3000 | 24 | 8.37 | 6.57 | 8.51 (β 0.05) | 8.37 (β 0) | 0.861 | — | capped |
| 71 | `amazonbook__mf__full__bpr__s0-long` | modern | S4 · pc-s4b-1018 | patience; 3800/4300 | 30 | 8.38 | 6.59 | 8.62 (β 0.05) | 8.38 (β 0) | 0.862 | — | converged |
| 72 | `amazonbook__mf__full__directau__s0` | modern | S3 · pc-s3c-1017 | patience; 25/40 | 79 | 9.57 | 8.70 | 11.05 (β 0.1) | 10.77 (β 0.2) | 0.786 | — | converged |
| 73 | `amazonbook__mf__full__directau__s0-legacy` | legacy | gh-exp-p4-amz-full-dau-1017 | patience; 25/35 | 69 | 9.57 | 8.70 | 11.05 (β 0.1) | 10.77 (β 0.2) | 0.786 | — | valid |
| 74 | `amazonbook__mf__graphhash__bpr__s0` | legacy | gh-exp-p4-amz-gh-bpr-long-1017 | patience; 1450/1950 | 13 | 4.10 | 2.53 | 3.34 (β 0.05) | 4.81 (β 0.2) | 0.084 | — | valid |
| 75 | `amazonbook__mf__graphhash__bpr__s0-pat100` | legacy | gh-exp-p4-amz-gh-bpr-1017 | patience; 10/110 | 1 | 3.93 | 3.89 | 5.06 (β 0.1) | 4.97 (β 0.02) | 0.128 | — | invalid |
| 76 | `amazonbook__mf__graphhash__bpr__s1` | modern | S4 · pc-s4d-1018 | patience; 780/1280 | 10 | 4.08 | 2.64 | 3.47 (β 0.05) | 4.79 (β 0.2) | 0.084 | 0.885 | converged |
| 77 | `amazonbook__mf__graphhash__bpr__s2` | modern | S4 · pc-s4d-1018 | patience; 1300/1800 | 14 | 4.09 | 2.51 | 3.39 (β 0.05) | 4.75 (β 0.2) | 0.085 | 0.889 | converged |
| 78 | `amazonbook__mf__graphhash__directau__s0` | modern | S3 · pc-s3a-1017 | patience; 40/65 | 122 | 4.63 | 4.64 | 6.33 (β 0.1) | 6.29 (β 0.5) | 0.189 | 0.746 | converged |
| 79 | `amazonbook__mf__graphhash__directau__s0-initmatch` | modern | S3 · pc-s3b-1017 | patience; 35/60 | 113 | 4.59 | 4.63 | 6.30 (β 0.1) | 6.33 (β 0.5) | 0.188 | 0.726 | converged |
| 80 | `amazonbook__mf__graphhash__directau__s0-legacy` | legacy | gh-exp-p4-amz-gh-dau-1017 | time_cap; 40/41 | 76 | 4.63 | 4.64 | 6.33 (β 0.1) | 6.29 (β 0.5) | 0.189 | — | capped |
| 81 | `amazonbook__mf__graphhash__directau__s1` | modern | S4 · pc-s4a-1018 | patience; 35/45 | 81 | 4.59 | 4.60 | 6.28 (β 0.1) | 6.31 (β 0.5) | 0.173 | 0.736 | converged |
| 82 | `amazonbook__mf__graphhash__directau__s2` | modern | S4 · pc-s4b-1018 | patience; 20/45 | 84 | 4.61 | 4.60 | 6.30 (β 0.1) | 6.16 (β 0.2) | 0.184 | 0.683 | converged |
| 83 | `amazonbook__mf__graphhash__directau_resbias__s0` | modern | S4 · pc-s4c-1018 | patience; 20/35 | 66 | 4.56 | 4.58 | 6.25 (β 0.1) | 6.07 (β 0.2) | 0.150 | 0.642 | converged |
| 84 | `amazonbook__mf__random__bpr__s0` | modern | S3 · pc-s3d-1017 | patience; 90/590 | 4 | 0.17 | 0.11 | 1.31 (β 0.2) | 1.28 (β 1) | 0.083 | 0.296 | converged |
| 85 | `amazonbook__mf__random__directau__s0` | modern | S3 · pc-s3a-1017 | patience; 35/35 | 65 | 0.15 | 0.12 | 1.55 (β 0.2) | 1.49 (β 0.5) | 0.242 | 0.998 | converged |
| 86 | `amazonbook__poponly__none__none__s0` | baseline | — | — | — | 0.96 | — | — | — | — | — | valid |
| 87 | `x__coloc_a` | modern | S0 · pc-a0-smoke-1017 | max_epochs; 1/1 | 1 | 6.73 | 7.07 | 12.57 (β 0.2) | 12.32 (β 0.2) | 0.209 | 0.596 | capped |
| 88 | `x__coloc_b` | modern | S0 · pc-a0-smoke-1017 | max_epochs; 1/1 | 1 | 6.73 | 7.07 | 12.57 (β 0.2) | 12.32 (β 0.2) | 0.209 | 0.596 | capped |
| 89 | `x__gowalla__mf__graphhash__directau__s0` | modern | S0 · pc-a0-smoke-1017 | max_epochs; 2/2 | 1 | 6.94 | 7.34 | 12.99 (β 0.2) | 12.96 (β 0.5) | 0.231 | 0.660 | capped |
| 90 | `x__gowalla__mf__graphhash__directau_dedup__s0` | legacy | graphhash-fast-dau-fix-s012-1016 | fixed_epochs; 30/30 | 14 | 7.53 | — | — | — | — | — | exploratory |
| 91 | `x__gowalla__mf__graphhash__directau_dedup__s1` | legacy | graphhash-fast-dau-fix-s012-1016 | fixed_epochs; 30/30 | 14 | 7.65 | — | — | — | — | — | exploratory |
| 92 | `x__gowalla__mf__graphhash__directau_dedup__s2` | legacy | graphhash-fast-dau-fix-s012-1016 | fixed_epochs; 30/30 | 14 | 7.58 | — | — | — | — | — | exploratory |
| 93 | `x__yelp2018__mf__graphhash__bpr__s0` | modern | S0 · pc-a0-smoke-1017 | max_epochs; 20/20 | 0 | 1.60 | 1.61 | 4.66 (β 0.2) | 4.36 (β 0.1) | 0.393 | 0.891 | capped |

## Appendix B: tail and coverage for every E5 record

All 90 planned raw / cosine / repaired cells over the 30 lineages; 89 are validated observations and one legacy AmazonBook BPR cosine cell is missing (`missing_gpu`). Source: `results/E5_final.json`.

| Dataset | Loss | Sharing | Run | Lineage / stop | Mode | Tail recall % (tail_primary) | Catalog coverage % |
|---|---|---|---|---|---|---|---|
| AmazonBook | bpr | dfreq | `amazonbook__mf__dfreq__bpr__s0` | modern / patience | cosine | 0.002 | 22.9 |
| AmazonBook | bpr | dfreq | `amazonbook__mf__dfreq__bpr__s0` | modern / patience | raw | 0.002 | 21.9 |
| AmazonBook | bpr | dfreq | `amazonbook__mf__dfreq__bpr__s0` | modern / patience | raw_prior | 0.001 | 11.6 |
| AmazonBook | directau | dfreq | `amazonbook__mf__dfreq__directau__s0` | modern / patience | cos_prior | 0.000 | 5.4 |
| AmazonBook | directau | dfreq | `amazonbook__mf__dfreq__directau__s0` | modern / patience | cosine | 0.033 | 99.5 |
| AmazonBook | directau | dfreq | `amazonbook__mf__dfreq__directau__s0` | modern / patience | raw | 0.013 | 55.1 |
| AmazonBook | bpr | full | `amazonbook__mf__full__bpr__s0` | modern / max_epochs (capped) | cosine | 5.821 | 98.3 |
| AmazonBook | bpr | full | `amazonbook__mf__full__bpr__s0` | modern / max_epochs (capped) | raw | 2.108 | 53.3 |
| AmazonBook | bpr | full | `amazonbook__mf__full__bpr__s0` | modern / max_epochs (capped) | raw_prior | 2.108 | 53.3 |
| AmazonBook | directau | full | `amazonbook__mf__full__directau__s0` | modern / patience | cos_prior | 2.698 | 44.3 |
| AmazonBook | directau | full | `amazonbook__mf__full__directau__s0` | modern / patience | cosine | 7.344 | 93.4 |
| AmazonBook | directau | full | `amazonbook__mf__full__directau__s0` | modern / patience | raw | 5.407 | 72.4 |
| AmazonBook | bpr | graphhash | `amazonbook__mf__graphhash__bpr__s0-legacy` | legacy / fixed_epochs | cosine | — | — |
| AmazonBook | bpr | graphhash | `amazonbook__mf__graphhash__bpr__s0-legacy` | legacy / fixed_epochs | raw | 3.132 | 44.4 |
| AmazonBook | bpr | graphhash | `amazonbook__mf__graphhash__bpr__s0-legacy` | legacy / fixed_epochs | raw_prior | 2.661 | 34.3 |
| AmazonBook | directau | graphhash | `amazonbook__mf__graphhash__directau__s0` | modern / patience | cos_prior | 3.781 | 80.7 |
| AmazonBook | directau | graphhash | `amazonbook__mf__graphhash__directau__s0` | modern / patience | cosine | 5.090 | 93.2 |
| AmazonBook | directau | graphhash | `amazonbook__mf__graphhash__directau__s0` | modern / patience | raw | 5.044 | 92.3 |
| AmazonBook | directau | graphhash | `amazonbook__mf__graphhash__directau__s0-initmatch` | modern / patience | cos_prior | 3.760 | 80.6 |
| AmazonBook | directau | graphhash | `amazonbook__mf__graphhash__directau__s0-initmatch` | modern / patience | cosine | 5.096 | 93.2 |
| AmazonBook | directau | graphhash | `amazonbook__mf__graphhash__directau__s0-initmatch` | modern / patience | raw | 5.014 | 92.3 |
| AmazonBook | bpr | random | `amazonbook__mf__random__bpr__s0` | modern / patience | cosine | 0.021 | 39.6 |
| AmazonBook | bpr | random | `amazonbook__mf__random__bpr__s0` | modern / patience | raw | 0.026 | 32.7 |
| AmazonBook | bpr | random | `amazonbook__mf__random__bpr__s0` | modern / patience | raw_prior | 0.000 | 1.3 |
| AmazonBook | directau | random | `amazonbook__mf__random__directau__s0` | modern / patience | cos_prior | 0.000 | 4.9 |
| AmazonBook | directau | random | `amazonbook__mf__random__directau__s0` | modern / patience | cosine | 0.054 | 78.4 |
| AmazonBook | directau | random | `amazonbook__mf__random__directau__s0` | modern / patience | raw | 0.049 | 64.7 |
| Gowalla | bpr | dfreq | `gowalla__mf__dfreq__bpr__s0` | modern / patience | cosine | 0.008 | 24.0 |
| Gowalla | bpr | dfreq | `gowalla__mf__dfreq__bpr__s0` | modern / patience | raw | 0.007 | 25.7 |
| Gowalla | bpr | dfreq | `gowalla__mf__dfreq__bpr__s0` | modern / patience | raw_prior | 0.002 | 10.4 |
| Gowalla | directau | dfreq | `gowalla__mf__dfreq__directau__s0` | modern / patience | cos_prior | 0.000 | 5.7 |
| Gowalla | directau | dfreq | `gowalla__mf__dfreq__directau__s0` | modern / patience | cosine | 0.106 | 99.6 |
| Gowalla | directau | dfreq | `gowalla__mf__dfreq__directau__s0` | modern / patience | raw | 0.040 | 67.4 |
| Gowalla | bpr | full | `gowalla__mf__full__bpr__s0-long` | modern / patience | cosine | 10.681 | 98.4 |
| Gowalla | bpr | full | `gowalla__mf__full__bpr__s0-long` | modern / patience | raw | 5.912 | 72.2 |
| Gowalla | bpr | full | `gowalla__mf__full__bpr__s0-long` | modern / patience | raw_prior | 3.870 | 50.0 |
| Gowalla | directau | full | `gowalla__mf__full__directau__s0` | modern / patience | cos_prior | 5.191 | 45.7 |
| Gowalla | directau | full | `gowalla__mf__full__directau__s0` | modern / patience | cosine | 12.295 | 96.3 |
| Gowalla | directau | full | `gowalla__mf__full__directau__s0` | modern / patience | raw | 2.501 | 30.8 |
| Gowalla | bpr | graphhash | `gowalla__mf__graphhash__bpr__s0-p500` | modern / patience | cosine | 5.145 | 78.2 |
| Gowalla | bpr | graphhash | `gowalla__mf__graphhash__bpr__s0-p500` | modern / patience | raw | 6.149 | 52.8 |
| Gowalla | bpr | graphhash | `gowalla__mf__graphhash__bpr__s0-p500` | modern / patience | raw_prior | 3.030 | 25.1 |
| Gowalla | bpr | graphhash | `gowalla__mf__graphhash__bpr__s1` | modern / patience | cosine | 4.786 | 77.5 |
| Gowalla | bpr | graphhash | `gowalla__mf__graphhash__bpr__s1` | modern / patience | raw | 6.269 | 53.8 |
| Gowalla | bpr | graphhash | `gowalla__mf__graphhash__bpr__s1` | modern / patience | raw_prior | 3.076 | 25.6 |
| Gowalla | bpr | graphhash | `gowalla__mf__graphhash__bpr__s2` | modern / patience | cosine | 4.734 | 76.9 |
| Gowalla | bpr | graphhash | `gowalla__mf__graphhash__bpr__s2` | modern / patience | raw | 6.269 | 54.9 |
| Gowalla | bpr | graphhash | `gowalla__mf__graphhash__bpr__s2` | modern / patience | raw_prior | 3.147 | 26.5 |
| Gowalla | directau | graphhash | `gowalla__mf__graphhash__directau__s0` | modern / patience | cos_prior | 2.005 | 28.4 |
| Gowalla | directau | graphhash | `gowalla__mf__graphhash__directau__s0` | modern / patience | cosine | 8.813 | 95.2 |
| Gowalla | directau | graphhash | `gowalla__mf__graphhash__directau__s0` | modern / patience | raw | 7.668 | 80.4 |
| Gowalla | directau | graphhash | `gowalla__mf__graphhash__directau__s0-initmatch` | modern / patience | cos_prior | 1.986 | 28.3 |
| Gowalla | directau | graphhash | `gowalla__mf__graphhash__directau__s0-initmatch` | modern / patience | cosine | 8.955 | 95.1 |
| Gowalla | directau | graphhash | `gowalla__mf__graphhash__directau__s0-initmatch` | modern / patience | raw | 7.785 | 81.5 |
| Gowalla | directau | graphhash | `gowalla__mf__graphhash__directau__s1` | modern / patience | cos_prior | 1.957 | 28.4 |
| Gowalla | directau | graphhash | `gowalla__mf__graphhash__directau__s1` | modern / patience | cosine | 8.923 | 95.2 |
| Gowalla | directau | graphhash | `gowalla__mf__graphhash__directau__s1` | modern / patience | raw | 7.793 | 81.3 |
| Gowalla | directau | graphhash | `gowalla__mf__graphhash__directau__s2` | modern / patience | cos_prior | 1.983 | 28.4 |
| Gowalla | directau | graphhash | `gowalla__mf__graphhash__directau__s2` | modern / patience | cosine | 8.922 | 95.1 |
| Gowalla | directau | graphhash | `gowalla__mf__graphhash__directau__s2` | modern / patience | raw | 7.816 | 81.2 |
| Gowalla | bpr | random | `gowalla__mf__random__bpr__s0` | modern / patience | cosine | 0.057 | 24.7 |
| Gowalla | bpr | random | `gowalla__mf__random__bpr__s0` | modern / patience | raw | 0.035 | 16.1 |
| Gowalla | bpr | random | `gowalla__mf__random__bpr__s0` | modern / patience | raw_prior | 0.000 | 1.6 |
| Gowalla | directau | random | `gowalla__mf__random__directau__s0` | modern / patience | cos_prior | 0.000 | 5.1 |
| Gowalla | directau | random | `gowalla__mf__random__directau__s0` | modern / patience | cosine | 0.126 | 84.4 |
| Gowalla | directau | random | `gowalla__mf__random__directau__s0` | modern / patience | raw | 0.083 | 62.4 |
| Yelp2018 | bpr | dfreq | `yelp2018__mf__dfreq__bpr__s0` | modern / patience | cosine | 0.008 | 21.8 |
| Yelp2018 | bpr | dfreq | `yelp2018__mf__dfreq__bpr__s0` | modern / patience | raw | 0.009 | 21.5 |
| Yelp2018 | bpr | dfreq | `yelp2018__mf__dfreq__bpr__s0` | modern / patience | raw_prior | 0.003 | 16.6 |
| Yelp2018 | directau | dfreq | `yelp2018__mf__dfreq__directau__s0` | modern / patience | cos_prior | 0.001 | 9.6 |
| Yelp2018 | directau | dfreq | `yelp2018__mf__dfreq__directau__s0` | modern / patience | cosine | 0.168 | 99.9 |
| Yelp2018 | directau | dfreq | `yelp2018__mf__dfreq__directau__s0` | modern / patience | raw | 0.054 | 69.4 |
| Yelp2018 | bpr | full | `yelp2018__mf__full__bpr__s0` | modern / patience | cosine | 3.475 | 84.7 |
| Yelp2018 | bpr | full | `yelp2018__mf__full__bpr__s0` | modern / patience | raw | 0.123 | 9.2 |
| Yelp2018 | bpr | full | `yelp2018__mf__full__bpr__s0` | modern / patience | raw_prior | 0.123 | 9.2 |
| Yelp2018 | directau | full | `yelp2018__mf__full__directau__s0` | modern / patience | cos_prior | 2.290 | 43.9 |
| Yelp2018 | directau | full | `yelp2018__mf__full__directau__s0` | modern / patience | cosine | 7.013 | 97.8 |
| Yelp2018 | directau | full | `yelp2018__mf__full__directau__s0` | modern / patience | raw | 1.030 | 32.0 |
| Yelp2018 | bpr | graphhash | `yelp2018__mf__graphhash__bpr__s0` | modern / patience | cosine | 0.919 | 49.3 |
| Yelp2018 | bpr | graphhash | `yelp2018__mf__graphhash__bpr__s0` | modern / patience | raw | 0.800 | 22.9 |
| Yelp2018 | bpr | graphhash | `yelp2018__mf__graphhash__bpr__s0` | modern / patience | raw_prior | 0.275 | 7.3 |
| Yelp2018 | directau | graphhash | `yelp2018__mf__graphhash__directau__s0` | modern / patience | cos_prior | 0.290 | 21.7 |
| Yelp2018 | directau | graphhash | `yelp2018__mf__graphhash__directau__s0` | modern / patience | cosine | 2.301 | 96.2 |
| Yelp2018 | directau | graphhash | `yelp2018__mf__graphhash__directau__s0` | modern / patience | raw | 1.469 | 61.0 |
| Yelp2018 | bpr | random | `yelp2018__mf__random__bpr__s0` | modern / patience | cosine | 0.081 | 35.3 |
| Yelp2018 | bpr | random | `yelp2018__mf__random__bpr__s0` | modern / patience | raw | 0.081 | 29.5 |
| Yelp2018 | bpr | random | `yelp2018__mf__random__bpr__s0` | modern / patience | raw_prior | 0.000 | 3.5 |
| Yelp2018 | directau | random | `yelp2018__mf__random__directau__s0` | modern / patience | cos_prior | 0.004 | 10.0 |
| Yelp2018 | directau | random | `yelp2018__mf__random__directau__s0` | modern / patience | cosine | 0.159 | 91.5 |
| Yelp2018 | directau | random | `yelp2018__mf__random__directau__s0` | modern / patience | raw | 0.103 | 70.7 |
