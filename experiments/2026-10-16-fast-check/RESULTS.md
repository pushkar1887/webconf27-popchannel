# GraphHash × loss: all runs (16–17 Oct 2026)

MF, dim 64. Recall@20 / NDCG@20 in %. Recall denominator = min(20, #positives), as in the authors' code.
Val masks train items; test masks train+val items. "Prior" = + β·log(1+train count of item); β is picked on
validation from {0, .01, .02, .05, .1, .2, .5, 1, 2, 5} and test is scored once. ρ = Spearman(item vector norm, train count).
GraphHash buckets: Gowalla 5,792 users / 5,806 items (742k params vs 4.53M full).

## 1. Headline (test Recall@20 / NDCG@20)

| Dataset | Model | No prior | With prior (β) | ρ |
|---|---|---|---|---|
| Gowalla | Full × DirectAU | 17.90 / 11.52 (raw) | 19.54 / 12.59 (cos, 0.1) | 0.815 |
| Gowalla | **GraphHash × DirectAU** (3 seeds) | 7.56 ± 0.02 / 4.03 (cos) | **13.29 ± 0.03 / 8.54 ± 0.01** (cos, 0.2) | 0.27 |
| Gowalla | GraphHash × BPR (authors' config) | 9.34 / 5.49 (raw) | 11.24 / 6.95 (raw, 0.5) | 0.054 |
| Gowalla | Popularity only | — | 2.97 / 1.70 | — |
| AmazonBook | Full × DirectAU | 9.57 / 6.45 (raw) | 11.05 / 7.48 (cos, 0.1) | 0.786 |
| AmazonBook | **GraphHash × DirectAU** | 4.64 / 3.13 (cos) | **6.33 / 4.46** (cos, 0.1) | 0.189 |
| AmazonBook | GraphHash × BPR (authors' config) | 4.10 / 2.60 (raw) | 4.81 / 3.28 (raw, 0.2) | 0.084 |

## 2. Gowalla runs

### 2a. Fast check: 30 epochs, batch 1024, lr 0.01, wd 1e-8, seed 0
| Run | Val R@20 at epoch 5 / 10 / 15 / 20 / 25 / 30 | Test R@20 | Test NDCG@20 | Train s |
|---|---|---|---|---|
| GraphHash × BPR (batch 1024: **invalid config**) | 5.46 / 5.17 / 5.13 / 5.02 / 4.93 / 5.27 | 5.31 | 3.07 | 53 |
| GraphHash × DirectAU | 7.11 / 7.19 / 7.40 / 7.40 / 7.44 / 7.46 | 7.56 | 4.04 | 1122 |
| GraphHash × DirectAU + unique-bucket fix | 7.15 / 7.32 / 7.51 / 7.45 / 7.50 / 7.57 | 7.53 | 4.01 | 867 |

Duplicate-bucket fraction per batch: 0.154 users, 0.146 items.

### 2b. Seed check for the unique-bucket fix (30 epochs, seeds 0/1/2)
| Run | Val R@20 at epoch 30 | Test R@20 |
|---|---|---|
| DirectAU | 7.46 / 7.47 / 7.49 → 7.47 ± 0.01 | 7.56 / 7.57 / 7.53 → 7.55 ± 0.02 |
| DirectAU + fix | 7.57 / 7.55 / 7.52 → 7.55 ± 0.03 | 7.53 / 7.65 / 7.58 → 7.59 ± 0.06 |
| Paired difference (fix − DirectAU) | +0.11 / +0.08 / +0.03 | −0.02 / +0.08 / +0.05 |

The gain is consistent but negligible, so the fix was dropped.

### 2c. GraphHash × BPR
| Run | Best val R@20 (epoch) | Epochs run | Test R@20 | Test NDCG@20 |
|---|---|---|---|---|
| Batch 1024, long (patience 10) | 5.46 (5) | 15 | 5.45 | — |
| **Authors' config** (full batch, lr 0.01, wd 1e-6; 94 s) | 9.39 (330) | 430 | **9.34** | 5.49 |

Authors' config re-scored:
| Scoring | β | Val R@20 | Test R@20 | Test NDCG@20 |
|---|---|---|---|---|
| raw | — | 9.39 | 9.34 | 5.49 |
| cosine | — | — | 6.08 | 3.08 |
| raw + prior | 0.5 | 11.39 | 11.24 | 6.95 |

Val R@20 by β (raw + prior): 0→9.39, 0.01→10.18, 0.02→10.23, 0.05→10.40, 0.1→10.70, 0.2→11.16, **0.5→11.39**, 1→9.85, 2→7.50, 5→5.16.

### 2d. GraphHash × DirectAU, 30 epochs, 3 seeds (P2)
| Seed | Raw test | Cosine test | Cos+prior β | Cos+prior val | Cos+prior test R / N | ρ |
|---|---|---|---|---|---|---|
| 0 | 7.56 | 7.58 | 0.2 | 13.36 | 13.26 / 8.54 | 0.271 |
| 1 | 7.57 | 7.56 | 0.2 | 13.41 | 13.31 / 8.55 | 0.274 |
| 2 | 7.53 | 7.55 | 0.2 | 13.38 | 13.30 / 8.53 | 0.270 |
| **Mean ± std** | 7.55 | **7.56 ± 0.02** | 0.2 | 13.38 ± 0.02 | **13.29 ± 0.03 / 8.54 ± 0.01** | 0.272 |

Val R@20 by β (cos + prior, seed 2): 0→7.43, 0.01→8.95, 0.02→9.64, 0.05→11.47, 0.1→13.24, **0.2→13.38**, 0.5→6.31.

### 2e. Full × DirectAU (checkpoint from the 90-min-capped run, epoch 120)
| Scoring | β | Val R@20 | Test R@20 | Test NDCG@20 |
|---|---|---|---|---|
| raw | — | — | 17.90 | 11.52 |
| cosine | — | 12.62 | 12.99 | 7.44 |
| cos + prior | 0.1 | 19.19 | 19.54 | 12.59 |

ρ = 0.815. Paper target for raw test: 17.98.

### 2f. Other hashing schemes × DirectAU (P3; 30 epochs, seed 0, authors' per-method lr/wd)
| Method (lr, wd) | Best epoch | Raw test R / N | Cosine test R / N | Cos+prior β | Cos+prior test R / N | ρ |
|---|---|---|---|---|---|---|
| GraphHash (0.01, 1e-8), from 2d | 30 | 7.56 / 4.04 | 7.56 / 4.03 | 0.2 | 13.29 / 8.54 | 0.27 |
| Random (0.01, 1e-6) | 20 | **0.47** / 0.27 | 0.31 / 0.17 | 0.2 | 3.85 / 2.44 | 0.274 |
| Double-frequency (0.005, 1e-6) | 30 | 3.23 / 2.54 | 2.18 / 1.71 | 0.2 | 5.15 / 3.50 | 0.029 |

Random's validation stayed flat at 0.41–0.48 from epoch 5, so it looks collapsed. It hasn't been checked against the paper yet.

### 2g. Learned per-item bias b_i (P5; GraphHash × DirectAU + BPR on cos+b, 30 epochs, seed 0)
| Scoring | β | Val R@20 | Test R@20 | Test NDCG@20 |
|---|---|---|---|---|
| **cos + b_i** (the method) | — | 4.33 | **4.32** | 2.53 |
| cos + b_i + prior | 0 | 4.33 | 4.32 | 2.53 |
| cosine (same vectors) | — | 7.24 | 7.48 | 3.99 |
| cos + prior (same vectors) | 0.2 | 13.17 | 13.17 | 8.47 |

Memory: 40,981 floats = 163,924 bytes = **5.52%** of the GraphHash embedding table. ρ = 0.219. Best epoch 25.
**Failed.** b_i swamps the cosine term (scale mismatch, no effective L2), so ranking is close to popularity-only.

## 3. AmazonBook runs (P4; authors' settings, seed 0)

| Run | Stop reason | Best epoch | Epochs | Train s | ρ |
|---|---|---|---|---|---|
| Full × DirectAU (lr 1e-3, wd 1e-8) | patience 10 | 25 | 35 | 4162 | 0.786 |
| GraphHash × DirectAU (lr 1e-3, wd 1e-8) | **75-min cap** (still rising slowly) | 40 | 41 | 4581 | 0.189 |
| GraphHash × BPR, full batch (lr 0.01, wd 1e-6) | patience 500 | 1450 | 1950 | 809 | 0.084 |

| Run | Raw val / test R, test N | Cosine test R / N | Cos+prior β, test R / N | Raw+prior β, test R / N |
|---|---|---|---|---|
| Full × DirectAU | 9.51 / 9.57, 6.45 | 8.70 / 5.69 | 0.1, **11.05 / 7.48** | 0.2, 10.77 / 7.26 |
| GraphHash × DirectAU | 4.62 / 4.63, 3.12 | 4.64 / 3.13 | 0.1, **6.33 / 4.46** | 0.5, 6.29 / 4.44 |
| GraphHash × BPR | 4.04 / 4.10, 2.60 | 2.53 / 1.59 | 0.05, 3.34 / 2.12 | 0.2, **4.81 / 3.28** |

A first AmazonBook BPR run with patience 100 stopped at epoch 110 on an early spike (test 3.93), so it is superseded.

## 4. Invalid or failed runs (do not cite)
- GraphHash × BPR at batch 1024 (5.31 / 5.45): wrong config.
- P1 on the Kaggle **CPU** image (val 4.24 at β = 0): different GraphHash bucket labelling than the GPU image. Rerun on GPU gave the correct numbers (2c).
- P1 GPU, first attempt: Google Drive HTTP 500 during download. Retried successfully.
- AmazonBook BPR, patience 100: stopped early. Superseded.
- Original Full × DirectAU repro: hit the 90-min cap at epoch ~130. Its best checkpoint (epoch 120) is the one used in 2e.

## 5. Caveats
- Most DirectAU runs are 30 epochs. AmazonBook GraphHash × DirectAU was time-capped. Gowalla Full × DirectAU is from a capped run.
- Gowalla GraphHash × DirectAU has 3 seeds. Everything else is 1 seed.
- β is tuned on validation per run. Test is scored once.
