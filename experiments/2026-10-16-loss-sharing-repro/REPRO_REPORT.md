# Reproduction stopped at the runtime gate

[Certain] **The first condition, Full MF + DirectAU, reached the 90-minute limit. The pilot stopped automatically. The other five conditions and optional AmazonBook runs were not started.**

This is a runtime-gate failure, not a test of whether the published loss × sharing pattern is true or false. No completed test-set comparison is available.

## What actually ran

| Item | Measured or fixed value |
|---|---|
| Account / notebook | University / graphhash-loss-sharing-repro-1016, version 2 |
| Hardware | Tesla T4 |
| Runtime | Python 3.13.15; Torch 2.11.0+cu128 |
| Source commit | 7a1d5eb48dce9ab4c4a453d4482dd0d3bdec4467 |
| Condition | Full MF, DirectAU, seed 0 |
| Embedding dimension / parameters | 64 / 4,533,696 |
| Optimizer | Adam; default betas 0.9/0.999 and epsilon 1e-8 |
| Learning rate / weight decay | 0.01 / 1e-8 |
| Batch size / gamma | 1024 / 1.0 |
| Validation / stopping | Raw-dot Recall@20 every 10 epochs; 50-epoch patience; maximum 1000 epochs |
| Training/evaluation attempt wall time | 5400.047 seconds = 90.001 minutes |
| Peak CUDA tensor allocation | 380.24 MiB |
| Remote session including setup | 5,461.7 seconds = 91.03 minutes |
| Median observed epoch time | 38.986 seconds |
| Best observed validation Recall@20 | 17.621% at epoch 120 |
| Last recorded validation Recall@20 | 17.542% at epoch 130 |

Epoch labels are zero-based, as in the log. The saved best checkpoint is from epoch 120. The runtime gate was reached during later training. It was checked between batches; the recorded overshoot is about 0.05 seconds. Kaggle reports ERROR because the guard raised TimeoutError intentionally.

## Requested comparison

| Condition | Status | Test Recall@20 / NDCG@20, raw and cosine |
|---|---|---|
| Full + DirectAU | Timed out | Not evaluated |
| Full + BPR | Not started | Not evaluated |
| Random + BPR | Not started | Not evaluated |
| Random + DirectAU | Not started | Not evaluated |
| GraphHash + BPR | Not started | Not evaluated |
| GraphHash + DirectAU | Not started | Not evaluated |
| Optional AmazonBook boundary | Not started | Not evaluated |

**All three direction checks are incomplete.** The validation score above must not be compared directly with the paper's test-set target of 17.984%. No fix, tuning grid, extra seed or alternative training variant was run.

## Why the run did not stop earlier

The best validation score was still improving at epoch 120, approximately 78.74 minutes into the run. The declared patience required 50 epochs without improvement. The 90-minute wall-time guard arrived before that stopping condition.

The observed epoch time was approximately 39 seconds. DirectAU evaluates pairwise distances within minibatches; a small MF parameter count does not imply a cheap loss. This pilot measures the actual cost on this T4, without making an unsupported completion-time forecast.

## Checkpoint diagnostics

The training loop computed diagnostics at validation snapshots, but its timeout path did not retain those trajectories. To avoid fabricating missing values, checkpoint diagnostics were recomputed **on CPU, without training**, from the retained best-validation checkpoint and the same fixed 1024-edge probe. This took 0.082 seconds. These are post-timeout checkpoint measurements, not GPU trajectory averages.

| Quantity | Users | Items |
|---|---:|---:|
| Repeat-ID collision fraction | 0.000105007 | 0.000147009 |
| Extra alias collision fraction | 0.000000000 | 0.000000000 |
| Total rho | 0.000105007 | 0.000147009 |
| Cross-bucket kernel mean Z | 0.023654085 | 0.023985976 |
| Alpha | 0.995579894 | 0.993907472 |
| Alignment gradient norm | 0.004326791 | 0.004501356 |
| Uniformity-term gradient norm | 0.001322285 | 0.001434344 |
| Mean embedding norm, expanded original IDs | 8.946589 | 8.504127 |

Alignment on the probe: 0.711327076. User/item uniformity: -3.739894629 / -3.724321842. Uniformity gradient norms use the averaged user-plus-item uniformity term. Extra alias collisions are zero because this condition has full individual embeddings. These measurements establish no compression mechanism.

The CSV leaves unmeasured test, cosine, NDCG and completed-epoch fields empty. Its observed validation and post-timeout diagnostic fields explicitly identify their source and scope.

## Data and reproducibility

- Author-released Gowalla split: 821,896 training, 102,737 validation and 102,737 test events; total 1,027,370.
- Loaded vocabulary: 29,858 users and **40,981 items**. The earlier task text listed 49,981 items. The actual author files were used without padding or resplitting; their full parameter count matches Table 1.
- Training-only GraphHash preprocessing produced 5,792 user buckets and 5,806 item buckets, or 742,272 embedding parameters. The compressed models were not trained.
- The per-condition settings came from the pinned author's reproduction commands, not a new parameter search.
- GPU RNG replaced the CPU DataLoader sampling path; this is not a bitwise replication of the authors' seeds. Evaluation used memory-bounded chunks. The simplified patience clock differs slightly from the source's stale-validation bookkeeping.
- The pinned scikit-network 0.32.1 sources were rebuilt unchanged from a SHA256-verified official PyPI wheel for Python 3.13/NumPy 2. The user explicitly authorized this package after the Sonatype check returned HTTP 401.
- Nine correctness checks passed, including metric parity, collision decomposition, scoring masks, configuration preservation and timing guards. Those checks are distinct from empirical reproduction success.

[Certain] **No training is continuing. The result is that the original Full + DirectAU condition did not complete under this 90-minute limit; the six-condition anomaly remains untested.**
