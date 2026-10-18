# Loss × embedding sharing: reproduction only

Status: **stopped**. TimeoutError: 90-minute training cap reached

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
- scikit-network 0.32.1 rebuilt from unchanged .pyx sources bundled in its SHA256-verified official PyPI wheel, using preinstalled Cython/NumPy.

## Loaded author data

{
  "Gowalla": {
    "users": 29858,
    "items": 40981,
    "user_id_max": 29857,
    "item_id_max": 40980,
    "interactions": 1027370,
    "split_counts": [
      821896,
      102737,
      102737
    ],
    "file_sha256": {
      "train": "6c14aa059709aa8fe1c0f8f5b3f624d0320564a58903e465265549c5e7dca9d8",
      "val": "18f263af248eb0b898aab05c411e66db0faa5b72a7e4e2e940ec1d054d2bf0cc",
      "test": "d20eb91da03d2e9133843cf8505b2f7c4402f8a9b57b28a24420eead15dcafaa"
    },
    "user_buckets": 5792,
    "item_buckets": 5806,
    "compressed_parameters": 742272,
    "full_parameters": 4533696
  }
}

Gowalla item count in the loaded split is 40,981; the earlier specification listed 49,981. The loaded 4.534M full parameters agree with Table 1. No resplitting or padding was performed.

## Results

Metrics below are percentage units. CSV Recall/NDCG values are fractions.

| Dataset | Method | Loss | Raw test R@20 | Cosine test R@20 | Raw test NDCG@20 | Minutes | Status |
|---|---|---|---:|---:|---:|---:|---|
| Gowalla | Full | DirectAU | NA | NA | NA | 90.00 | timeout |

## Direction check

- Full: incomplete.
- GraphHash: incomplete.
- Random: incomplete.

This checks directions for one seed and fixed released configurations. It does not establish statistical robustness, a collision mechanism, repair effectiveness or novelty.

## Diagnostics

metrics.csv contains separate repeat-ID and extra-alias fractions, Z, alpha, gradient norms and mean vector norms for users and items, on the selected checkpoint and as a mean over validation snapshots. BPR uniformity/alignment gradients are diagnostic gradients, not BPR training terms.

## Timing and limits

Elapsed session seconds including setup: 5461.7. Each training condition has a 5400-second cap. No tuning or loss fixes were run.
