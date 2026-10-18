# Loss × embedding sharing: reproduction only

Status: **stopped**. StopIteration: 

Pinned GraphHash commit: 7a1d5eb48dce9ab4c4a453d4482dd0d3bdec4467. Seed: 0.
Runtime: Python 3.13.15, Torch 2.11.0+cu128; GPU Tesla T4.

## Contract

- One seed (0), no grid and no loss fixes. Author-published per-condition command settings.
- Primary Recall follows released quality_metrics: hits/min(20, positives); standard Recall is separate.
- Model selection: validation raw-dot Recall@20, every 10 epochs; 50-epoch patience, maximum 1000 epochs.
- Diagnostic trajectories use one fixed 1024-positive-edge probe per validation snapshot; gradient norms are diagnostic objective gradients, including for BPR.
- Raw-dot and cosine final metrics use the same validation-selected checkpoint; cosine does not select checkpoints.
- CUDA matrix precision float32, TF32 disabled. Evaluation chunking is reduced for memory and vectorized for speed.
- Sonatype authentication returned HTTP 401; user explicitly authorized the pinned clustering package after PyPI hash verification.

## Loaded author data

{}

Gowalla item count in the loaded split is 40,981; the earlier specification listed 49,981. The loaded 4.534M full parameters agree with Table 1. No resplitting or padding was performed.

## Results

Metrics below are percentage units. CSV Recall/NDCG values are fractions.

| Dataset | Method | Loss | Raw test R@20 | Cosine test R@20 | Raw test NDCG@20 | Minutes | Status |
|---|---|---|---:|---:|---:|---:|---|

## Direction check

- Full: incomplete.
- GraphHash: incomplete.
- Random: incomplete.

This checks directions for one seed and fixed released configurations. It does not establish statistical robustness, a collision mechanism, repair effectiveness or novelty.

## Diagnostics

metrics.csv contains separate repeat-ID and extra-alias fractions, Z, alpha, gradient norms and mean vector norms for users and items, on the selected checkpoint and as a mean over validation snapshots. BPR uniformity/alignment gradients are diagnostic gradients, not BPR training terms.

## Timing and limits

Elapsed session seconds including setup: 0.4. Each training condition has a 5400-second cap. No tuning or loss fixes were run.
