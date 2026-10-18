# Reproduction ledger — one seed, six conditions; no fixes or grid

Spec: user's 16 October 2026 reproduction-only message.
Outputs to deliver: REPRO_REPORT.md and metrics.csv only.
Internal source, data, tests and checkpoints stay under _work.

Ruling: use the pinned authors' per-condition commands for lr/wd and batches. The shared Gowalla YAML's full batch is a BPR setting; applying it to DirectAU would create an invalid pairwise workload. Cost if wrong: this is a published-config reproduction rather than a shared-global-default comparison; state exact values.
Ruling: preserve the released Recall@20 denominator min(20, number of positives), and also record standard recall as an explicitly separate column. Cost if wrong: the main paper targets cannot be compared with standard recall.
Ruling: compute expensive diagnostic gradients on a fixed 1024-edge probe at validation snapshots; do not form a full-batch BPR pairwise-uniformity matrix. Cost if wrong: sampled diagnostic trajectories, not exhaustive per-step diagnostics.

Preflight: repository cloned and pinned to 7a1d5eb48dce9ab4c4a453d4482dd0d3bdec4467.
Preflight: authors' 74,772,058-byte archive downloaded; real Gowalla and AmazonBook files extracted without resplitting.
Preflight: Sister read-only API returned HTTP 401. University, Default and Father authenticated. University selected and announced.
Preflight: University runtime preflight script submitted privately with GPU and internet, version 1.
Plan: verify data and metric/collision/time-cap helpers; run first full DirectAU condition; enforce 5400 seconds per condition; then remaining five only if the timing gate permits; optional AmazonBook only after the three-direction gate passes and time remains.
Task 1: six helper tests RED (missing implementation) then GREEN.
Task 2: runner uses pinned author MF/DAULoss/BPRLoss and extracts author mapping functions; fixed-config, no grid.
Final review: independent read-only reviewer found optional AmazonBook BPR wd mismatch and final-diagnostics timing gap. Both corrected with new tests observed RED then GREEN; final suite 9 tests.
Final: fixed AmazonBook BPR wd to 1e-6 — author-settings regression RED→GREEN.
Final: fixed final diagnostics exceeding cap before completion — final timing regression RED→GREEN.
Ruling: GPU RNG sampling reproduces the source sampling distributions but not its bitwise CPU DataLoader sequence. Cost if wrong: do not claim bitwise seed reproduction.
Ruling: retain simple 50-epoch patience measured against best validation epoch; source counter bookkeeping may stop a few unselected epochs later. Cost if wrong: slight stopping-clock difference, reported explicitly.
Ruling: per-run cap is checked between GPU operations and after final diagnostics, not an exact-second interrupt of an in-flight CUDA kernel. Cost if wrong: a small overshoot can occur, but the next condition is not launched after a timed-out condition.
Dependency exception: user explicitly authorized only scikit-network 0.32.1 in the private Kaggle runtime after official PyPI hash verification; Sonatype endpoint returned HTTP401.

Execution: University v1 stopped before training on missing sdist / Python3.13 incompatibility. v2 uses unchanged .pyx sources inside SHA256-verified official PyPI wheel and preinstalled compiler/Cython; submitted and observed RUNNING.
Verification: vectorized evaluation compared with pinned author quality_metrics on deterministic mixed-size ground truths; exact agreement within 1e-12.
Live state: University reproduction version 2 RUNNING at about 59 minutes after submission. Monitor exec session ID 83747 polls every 30 seconds and fetches selected artifacts automatically on completion/error. Do not resubmit or cancel. Use write_stdin on that session, or kaggle_ops.py status/fetch. No partial Kaggle outputs available through public output API so far. Main cap checked inside each model run; optional Amazon skipped if primary session exceeds 3600 seconds.

FINAL: University version2 stopped by the intended first-condition 90-minute guard; only Full/DirectAU was started. No remaining or boundary training. Best GPU validation Recall .17621450387636384 at epoch120; last validation epoch130. Runtime5400.0475s, peak380.2437MiB. Post-timeout best-checkpoint CPU diagnostic inspection0.0825s, no training. Report and CSV finalized; missing comparisons/test metrics left explicit. Monitor session83747 completed. No running pilot remains.
