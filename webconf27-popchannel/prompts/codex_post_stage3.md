# Prompt for Codex: post-Stage-3 analyses and the G3 abstract-number package
(Paste everything below the line into Codex. This is a hand-off note, not one of the 4 project docs.)

---

Stage 3 has finished. 25/25 runs are valid. Read the RESULTS.md block "Stage 3 outcomes, 2026-10-18 04:45 IST" and the last 4 LOG lines in PLAN.md §11 first.

**Your role and limits:** you are engineer / analyst. All TASKS.md Part 1 rules apply:
- no Kaggle pushes;
- no PREREG edits;
- no secrets;
- validation-only selection; test is scored once;
- one LOG line when done.

**Deadline:** the G3 abstract numbers freeze **Sun 18 Oct 12:00 IST**. Finish Tasks 1–4 by **10:30 IST**.

Project root: `D:\web_2\webconf27-popchannel\`. Code: `D:\web_2\experiments\2026-10-16-fast-check\`.

## 1. Refresh aggregation and the main tables (E1)
- Run `scripts/aggregate.py` so `results/master_metrics.csv` contains all 25 Stage 3 runs.
- Each row must carry:
  - stop_reason;
  - a capped flag: `amazonbook__mf__full__bpr__s0` hit max epochs and is capped;
  - controlled vs reproduction;
  - a lineage tag: legacy vs modern.
- Bump the CSV schema to a versioned name (e.g. `master_metrics_v2`), and add the E5/E7/E9 columns, as you flagged last time.
- Regenerate the RESULTS.md §1–§2 tables with `scripts/render_stage12_tables.py`; rename or extend it for Stage 3.
- **Main table rows:** dataset × {Full, GraphHash, Random, DoubleFreq} × {BPR, DirectAU}.
- **Columns:**
  - raw;
  - cosine;
  - cos+prior (β);
  - raw+prior (β);
  - ρ;
  - stop status.
- **Controlled pairs to mark:**
  - Gowalla A2 vs I1 (and A1);
  - AmazonBook `full__directau__s0` vs `graphhash__directau__s0-initmatch` (and B9).
- **Gowalla GraphHash seeds:** report mean ± SD over s0/s1/s2. For BPR, s0 is `s0-p500`; the legacy patience-100 s0 is a pilot only.

## 2. Final E5 / E7 / E9 on the final runs (PREREG S3-E5, S3-E7, PR-E9)
Use your existing `scripts/analyze_e5.py` and `scripts/analyze_e7_e9.py`, extended to the final run list.

- **E5 runs:**
  - GraphHash DirectAU: Gowalla s0/s1/s2, Yelp, AmazonBook B9;
  - GraphHash BPR: Gowalla s0-p500/s1/s2, Yelp, AmazonBook;
  - Random and DF DirectAU on all 3 datasets;
  - the Full references: Gowalla A2, Yelp A3, AmazonBook `full__directau__s0`.
- **E5 measures:** tail-recall (`tail_primary`, plus the standard denominator) and catalog coverage for the cosine, raw and repaired modes.
- **E5 rule (S3-E5):** repaired GraphHash DirectAU tail-recall < Full DirectAU raw tail-recall on all 3 datasets. Report PASS / FAIL per dataset.
- **E7:** uint8 check on every final GraphHash checkpoint (DirectAU and BPR, all seeds, all datasets), validation and test, with identical tie handling.
  - The "one byte per item" claim is allowed only if **every** checkpoint has |Δ| ≤ 0.02 on both splits.
  - Otherwise the stated cost stays float32: report bytes and % of each GraphHash table per dataset.
  - Disclose the extra float32 scale.
- **E9:** weighted recall with w = (1+c)^(−1/2), at the original β and at a β re-tuned on validation, for the PR-E8 pairs on all 3 datasets. Report whether repaired DirectAU still beats repaired BPR under the weighted metric. Include the denominator note.
- Append one RESULTS.md §5 block each for S3-E5, S3-E7 and PR-E9 (final): prediction copied, observed, verdict, runs.

## 3. PR-E8 robustness check (AmazonBook tie order)
The orchestrator's AmazonBook E8 used the BPR side from the Stage 1 control file `runs/_sessions/pc-x2-1017/controls/amazonbook__mf__graphhash__bpr__s0-legacy/per_user__full_prior_s0.npz`, which has random-key tie order.

- Recompute the AmazonBook Δ with every other available tie order in that folder (e.g. the `base_popkey` variants combined with the prior if present). If none applies, state that.
- Confirm the Δ stays > 0 with the CI excluding 0.
- **Do not** score legacy checkpoints locally; mappings are GPU-only. If a GPU rescoring is needed, write the exact job spec in `prompts/` for the orchestrator. Do not push.

Also report NDCG@20 versions of the PR-E8 table: same pairs, same bootstrap, 2,000 resamples, seed 0. Write them in RESULTS.md as a secondary result.

## 4. G3 package: `prompts/g3_abstract_numbers.md`
One page for the human to sign off. Every number in it must come from `master_metrics.csv` or a RESULTS block, cited by run_id.

1. **Headline table:** Full vs GraphHash, each × {BPR, DirectAU}, raw and repaired, on the 3 datasets.
2. **PR-E8 Δs** with CIs and Holm p.
3. **Mechanism numbers:**
   - ρ_bucket vs ρ_item;
   - Shapley ε share (0.71 Gowalla, 0.76 AmazonBook);
   - tie share 12%.
4. **Honest limitations:**
   - tail cost (E5);
   - storage claim (E7 outcome);
   - Y3 failure;
   - PR-A8 range failure;
   - the capped AmazonBook Full BPR run;
   - C3 not run.
5. **A draft abstract** (≤ 200 words), framed per G1: "sharing erases item-level popularity information". Ties are secondary.
   - No "tail-safe" or "free" claims.
   - "One byte" only if E7 passes.
   - The PRISM / DNA prior work must be credited; magnitude ≈ popularity is not our claim.

## 5. Optional, only if Tasks 1–4 are done before 10:30
Write a design note (no training) in `prompts/c3_design.md`:
- the code change for C3 (learned β·p + L2-regularized residual r_i with a separate Adam decay of 0.01) in `exp_run.py`;
- the tests it needs;
- a job spec.

Do not add it to any queue.

## Reply with
1. The files changed.
2. The main table (markdown).
3. The E5/E7/E9 verdicts.
4. The AmazonBook E8 robustness result and the NDCG table.
5. The path to the G3 package.
6. Test counts.
7. Open questions (also in the LOG).

Do not claim anything you did not run.
