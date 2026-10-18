# Missing AmazonBook legacy BPR controls — orchestrator job specification

Status: **NOT SUBMITTED.** This is a scoring-only specification. Codex has not pushed, stopped, or restarted any Kaggle job. The orchestrator must review the evaluation-control entry/snapshot requirements before submission. No new training, mapping construction, checkpoint selection, or test-based beta selection is requested.

## Fixed input contract

- Dataset: the pinned GraphHash AmazonBook processed train/validation/test files already used in the project; original users/items and masks unchanged.
- Checkpoint: `p4-amz-gh-bpr-long_best_seed0.pt` (the legacy `amz_gh_bpr` checkpoint represented by `runs/_sessions/pc-x2-1017/controls/amazonbook__mf__graphhash__bpr__s0-legacy/controls.json`). Retrieve the existing private artifact using the orchestrator's authorized account; no credentials or signed URLs belong in this document.
- Use its **stored aliases**, checking exact array equality against the recorded Stage 1 alias digests. Never rebuild Louvain mappings on a CPU image or silently renumber buckets.
- Backbone MF, dimension 64. Primary base is **raw dot**, full prior is `beta*log1p(training_event_count)`, frozen original beta **0.2**. Counts use train only.
- Validation masks train interactions; test masks train+validation. Evaluate exactly 50,407 test users with nonempty truth. Capped Recall@20 is `hits/min(20,n_truth)`. NDCG follows the existing scorer.
- Rank bucket scores copied to original items, then lexicographically sort with explicit secondary keys. Never add a tiny score perturbation. Record every ranking/tie policy.

## Required scoring outputs

1. **E5 missing cosine:** save the test cosine top-20 and per-user `users, top20, hits, npos, ndcg`. Preserve existing cached raw/random-key-seed-0 and raw+prior outputs; also save an original torch.topk sensitivity if desired, labelled separately.
2. **E7:** at raw+prior beta 0.2, compute float32 and the 256-level uint8 prior on **both validation and test**, under the identical checkpoint, users, masks, bucket arithmetic and secondary key (random seed 0). The quantizer is `q=round(255*p/max(p))`, float32 scale `max(p)/255`, decoded `q*scale`; store uint8 payload plus the extra four-byte scale. Report signed and absolute Recall changes in **percentage points**. The registered tolerance is 0.02 points per split; no tuning to pass.
3. **E9:** use weights `(1+train_count)^(-1/2)`. Sweep the existing beta grid `{0,.01,.02,.05,.1,.2,.5,1,2,5}` on **weighted validation only**, ties choosing the smaller beta. Freeze that selection, then evaluate test once for original beta 0.2 and once for the selected beta (one cell if equal). Weighted per-user recall divides recovered truth weight by **all held-out truth weight**; equal weights recover standard recall, not capped recall. Save per-user weighted values and ordinary metrics as separate fields.
4. **E8 alternate full-prior ties:** keep checkpoint, raw base and beta 0.2 fixed. Save full-prior test top-20/per-user metrics for lexicographic index and popularity keys and random-key seeds 0–9. Retain the existing original torch.topk full-prior policy as a separately labelled sensitivity with saved per-user outputs. Every alternative must include the **complete prior**, not just the bucket mean or residual. Existing `base_popkey` is beta 0 and cannot serve this purpose.

The existing `score_controls.py` starts from these arguments, but **does not currently export every required per-user cell or weighted/quantized-validation output**:

```text
score_controls.py --ckpt /kaggle/working/inputs/p4-amz-gh-bpr-long_best_seed0.pt --dataset AmazonBook --method GraphHash --base raw --run-id amazonbook__mf__graphhash__bpr__s0-legacy-posts3-controls --seeds 10 --out-root /kaggle/working/post_stage3_controls
```

This command is an input-contract anchor, not a claim that the unmodified scorer produces all outputs. Before launching, extend/review its export points and add the weighted validation and uint8 validation cells. A scorer patch must preserve masks, fixed beta and lexicographic semantics. Alternatively use an independently reviewed GPU scorer with this exact contract. Do not submit the unmodified command and report the missing cells as completed.

## Output and analysis acceptance

Write `controls.json` with dataset/split/checkpoint/mapping receipts, beta source, policy, runtime and each cell's validation/test metrics. Save per-user NPZs for every required cell. After download, verify exact evaluated-user coverage, original truth/hits/NDCG and zero seen-item overlap using the existing E5 validators.

For each applicable full-prior tie policy, pair the BPR per-user values with the **frozen B9 DirectAU** `per_user_test__cos_prior.npz` and run 2,000 paired user bootstrap resamples, seed 0, for Recall and NDCG. These are secondary robustness results; do not replace the primary policy after seeing test. Keep Recall and NDCG Holm families distinct and label paired t-test p values separately from bootstrap intervals.

Current available evidence: the primary random-key-seed-0 paired CI excludes zero; all ten saved random-key aggregate means and the legacy torch.topk aggregate retain a positive difference. The alternate-policy paired CIs, weighted-validation retuning, quantized validation and legacy cosine predictions are **unavailable**, not failures or inferred measurements. No GPU runtime has been measured for this combined scoring job.
