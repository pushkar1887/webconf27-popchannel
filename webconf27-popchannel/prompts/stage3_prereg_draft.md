# Stage 3 prediction draft — prepared for registration

This is a hand-off artifact, not a fifth project document and not a registration. Nothing in this file authorizes a submission. For the authorized registration task, copy the block between the markers verbatim under a fresh timestamped Stage 3 heading in PREREG.md, then snapshot PREREG.md before the orchestrator pushes. Existing PREREG.md entries remain append-only.

<!-- BEGIN STAGE3 REGISTRATION -->

#### Stage 3 shared contract

The queue follows `prompts/codex_stage3.md` as amended by `prompts/codex_stage3_fix.md`: **25 queued jobs**, including the new patience-500 Gowalla GraphHash BPR seed-0 rerun. **C3 is blocked: needs new code (learned beta*p + regularized residual)** and has no job in this queue. No existing valid candidate run was found to skip. All queued runs use newly named run directories.

- MF, dimension 64; pinned data/splits and GraphHash source; stored mappings accompany every checkpoint.
- PLAN section 7.2 lr/wd cell by cell. D7's weight decays are its explicitly requested intervention.
- DirectAU: eval every 5 epochs, patience 30 epochs, min_delta .0005. The cap is **300 minutes for exactly** `amazonbook__mf__graphhash__directau__s0`, `amazonbook__mf__graphhash__directau__s0-initmatch`, `amazonbook__mf__full__directau__s0`, `amazonbook__mf__random__directau__s0`, and `amazonbook__mf__dfreq__directau__s0`; every other DirectAU job keeps 180 minutes. BPR: eval every 10, patience 500, min_delta 0, cap 90 minutes. Normal BPR max epochs 3000; A8r 20000.
- Gowalla DirectAU max epochs 200; Yelp and AmazonBook 150. AmazonBook's legacy GraphHash DirectAU reached 41 epochs in a 75-minute cap, about 1.8 minutes/epoch, with best epoch 40. At that rate 180 minutes permits only about 98 epochs; 150 epochs require about 275 minutes. The five AmazonBook DirectAU scheduling estimates therefore use the entire **300-minute training cap** plus the separate ten-minute job reserve. This is a runtime estimate, not a convergence guarantee.
- Raw-dot validation Recall@20 selects checkpoints; each prior-augmented scoring mode selects beta on validation from {0,.01,.02,.05,.1,.2,.5,1,2,5}; raw and cosine use beta 0. Test never selects beta or checkpoints. The primary comparison retains PR-E8: DirectAU cosine+prior against BPR raw+prior. Show both raw+prior and cosine+prior as secondary comparisons. Reuse stored final top-20 lists for group analyses; do not repeatedly score test to choose a presentation.
- PASS/FAIL refer to a named prediction. A time/epoch-capped run is **INCONCLUSIVE for convergence**, even if its metric crosses a gate. Keep its measured score with the cap label.
- Every expected range or threshold below derived from inspected pilot results is **[I]**, not a confidence interval or a prospective [P] prediction. A new job does not erase prior exposure to related test outcomes. [H] additionally identifies an inherited hypothesis; fixed analysis definitions and stopping rules are protocol choices.
- Seed-level spread and user-bootstrap uncertainty are separate. User intervals are conditional on the trained checkpoints.

#### S3-A2 [I]: Full DirectAU and the controlled partner of I1

Run: `gowalla__mf__full__directau__s0` (priority 1).

The inspected I1 config used lr .01, wd 1e-8, batch 1024, 200 max epochs, eval every 5, patience 30, min_delta .0005, cap 180. Its item/user initialization factors were .3781716084/.4423899219. **A2 uses plain Full Xavier initialization**: Full's distribution is the target to which I1 was scaled; applying I1's compressed factors to Full would be incorrect.

Predictions: raw Recall@20 in [17.4,18.5]%; cosine+prior in [19.0,20.2]%; norm-count Spearman >= .75; validation raw-minus-cosine gap >= 3.5 percentage points.

Decision: report the ranges and compare A2 with I1 under the matching protocol. A2 stopped by patience and A2 raw above I1 raw supports the full-versus-sharing contrast. If capped, the contrast is budget-limited; do not claim a completed convergence comparison. This also preserves the separate A1-versus-I1 initialization-sensitivity result.

#### S3-A8r [I]: extend the Full BPR baseline

Run: `gowalla__mf__full__bpr__s0-long` (priority 1b), lr .005, wd 1e-8, max 20000 epochs and cap 90 minutes. This is a fresh run, not a checkpoint continuation with missing optimizer state. Preserve the prior max-3000 run.

Prediction: raw Recall@20 in [16.3,18.3]%; raw+prior >= raw. The legacy 16.55% is an anchor, not a lower bound on test performance: selection is on validation.

Decision: stop-by-patience permits the registered operational convergence label. If capped, keep the strongest validation-selected result with an INCONCLUSIVE convergence label. A result above the publication's 15.617 is not automatically invalid and must not be tuned back down to that number. Audit any material difference in the intended computation/split contract before drawing the loss-interaction conclusion.

#### S3-B9 [I]: finish AmazonBook GraphHash DirectAU

Run: `amazonbook__mf__graphhash__directau__s0` (priority 2), max 150 epochs, **cap 300 minutes**, lr .001, wd 1e-8. The legacy run took about 75 minutes for 41 epochs (about 1.8 minutes/epoch), so a 180-minute cap would permit only about 98 epochs while validation had still been rising near the legacy cap. The 300-minute cap accommodates the approximately 275-minute estimate for 150 epochs without certifying patience stopping.

Predictions: raw in [4.3,5.5]%; cosine+prior in [6.0,7.2]%; cosine+prior exceeds the existing BPR raw+prior anchor 4.81% by >= .3 points. The uncorrected DirectAU score may still exceed BPR, as already seen on this dataset; no universal reversal is predicted.

Decision: PR-E8's final AmazonBook comparison uses the validation-selected final checkpoints and paired user inference. Patience required for the convergence label. Report any cap without substituting an optimistic prediction.

#### S3-B5/B6/B7/B8 [I]: Gowalla final-protocol seed replication

Runs: GraphHash DirectAU s1/s2 (B5/B6), GraphHash BPR s1/s2 (B7/B8), and **`gowalla__mf__graphhash__bpr__s0-p500`**, priority 3. The new seed-0 BPR run uses lr .01, wd 1e-6, max 3000 epochs, eval every 10, patience 500, min_delta 0 and cap 90 minutes; its training scheduling estimate is 10 minutes plus the ten-minute reserve. It joins B7/B8 on the same GPU list if the lane limits permit.

Predictions: DirectAU raw [7.2,7.9]%, cosine+prior [12.9,13.8]%; BPR raw [9.0,10.1]%, raw+prior [10.8,12.1]%. Every matched seed's DirectAU cosine+prior minus BPR raw+prior is positive, with its paired-user CI excluding zero, per PR-E8.

Decision: report every seed and the mean/sample SD, not only the best seed. A negative seed or a seed CI including zero fails the all-seeds success criterion. The final Gowalla matched-seed comparisons use DirectAU s0/s1/s2 against BPR **s0-p500/s1/s2**, respectively. The DirectAU s0 is the patience-stopped Stage 2 A1 checkpoint. The legacy Gowalla BPR s0 used patience 100 and is **pilot only**; it cannot replace s0-p500 in the final three-seed check.

#### S3-B-main-grid [I/H]: other sharing methods

Runs: Random and DoubleFreq x {BPR,DirectAU} on all three datasets (B1-B4, B10-B13, B15-B18), plus AmazonBook Full BPR (B14), priority 4. Each cell uses its own PLAN lr/wd.

Informed anchors on Gowalla: Random BPR [.7,.9]%, DirectAU [.35,.65]%; DoubleFreq BPR [3.6,4.2]%, DirectAU [3.0,3.8]%. These are consistency expectations, not exclusion filters.

Inherited PR-B predictions: corrected DirectAU >= max(raw,cosine)+1.0 point (B-i); selected DirectAU beta in {.1,.2} (B-ii); compressed norm-count Spearman rho <= rho_full - .30 against the same-dataset Full DirectAU anchor (B-iii); BPR raw+prior beta in [.2,.5] (B-iv). Report the fraction of cells satisfying each, including failures. Record whether each Full anchor is converged or capped and whether the comparison is controlled or an author-config reproduction. Mapping-dependent hypotheses below are secondary; they do not retroactively remove a failed B-iii prediction. AmazonBook Full BPR is descriptive with no pre-existing full-table score anchor: compare its corrected/uncorrected metrics and uncertainty to Full DirectAU.

Mapping-dependent interpretation:

- For one-alias maps with eta2_p <= .60, predict the per-item residual accounts for >= 50% of the fixed-beta Shapley-attributed repair gain [I/H, informed by Stage 1]. If this is not measured for a new checkpoint, keep it pending rather than infer it from Recall alone.
- For full alias tuples with eta2_p >= .95 (e.g. Gowalla DoubleFreq .999), make **no prediction of a large within-group residual or tie benefit**. A nearly injective alias tuple makes that decomposition nearly degenerate even though the underlying parameters are shared. Generic calibration can still help.
- eta2 is not an accuracy ceiling. Do not infer mandatory norm-erasure strength, or a monotone loss ordering, from it. Yelp's failed Y3 shows why those are separate observations.

Decision: retain all valid cells, label capped ones, and treat observed ordering as dataset/method-dependent. Do not add rescue hyperparameter sweeps to satisfy these ranges.

#### S3-I2 and partner [I]: matched initialization on AmazonBook

Runs: `amazonbook__mf__graphhash__directau__s0-initmatch` and `amazonbook__mf__full__directau__s0`, priority 5. Both lr .001, wd 1e-8, max epochs 150, **cap 300 minutes** and identical DirectAU stopping strings. The longer cap follows the same measured legacy AmazonBook timing as B9: about 1.8 minutes/epoch implies about 275 minutes for 150 epochs, whereas 180 minutes would permit only about 98 epochs. Both partners receive the full 300-minute training estimate plus the ten-minute reserve; a time or epoch cap remains INCONCLUSIVE for convergence.

Prediction: I2 versus B9 differs by <= .3 points in raw and corrected Recall@20; Full norm-count correlation >= .70; GraphHash correlation is at least .30 lower than Full. The initialization-sensitivity tolerance is extrapolated from Gowalla, not established on AmazonBook.

Decision: distinguish I2-versus-B9 (initialization sensitivity) from I2-versus-Full (sharing under the matched protocol). Capped partners make a convergence interpretation INCONCLUSIVE. A larger initialization effect must be reported rather than absorbed into the erasure narrative.

#### S3-C3 — BLOCKED, no job in this queue

Current `--loss dau_bias` implements DirectAU+BPR on cosine+b_i. It does not implement learned beta*logpop plus a separate residual group with decay .01. No training code was added for C3, as required by the hand-off.

Status: **blocked: needs new code (learned beta*p + regularized residual)**. If implemented and reviewed later, the inherited intention [I] is corrected score within +/- .5 points of the post-hoc correction; call it better only at >=+.3. These are conditional intentions outside this launch, not results or approval to run C3. A later C3 job requires its own reviewed implementation, entry and snapshot before push.

#### S3-D7 [I/H]: exact-zero and stronger weight decay

Runs: Full Gowalla DirectAU `s0-wd0` and `s0-wd1e-4`, priority 7. Same A2 protocol; wd is the sole intended intervention.

Prediction for wd=0: norm-count Spearman >= .70, extrapolated from the observed .815 at wd=1e-8. Report raw/cosine/corrected performance and norm trajectories for both settings, without predicting a mandatory ordering for wd=1e-4.

Decision: persistence at wd=0 supports persistence **under our configuration**, not a causal claim that Adam explains the difference from PRISM. PRISM's public trainer also defaults to Adam; compare implementations, batching, losses, stopping and exact decay before claiming a contradiction. No new claim of monotone Adam norm growth follows from this test.

#### S3-E5 [I]: final tail recall and catalog coverage against Full references

Scope: every final compressed checkpoint (GraphHash, Random and DoubleFreq, both losses, and each retained seed/init variant) on Gowalla, Yelp2018 and AmazonBook. For each, report test `tail_primary` and catalog coverage for **cosine, raw and repaired scoring**, together with the **Full model of the same dataset and loss** under those same modes. Repaired means cosine+validation-selected prior for DirectAU and raw+validation-selected prior for BPR. Full references are Gowalla A2 DirectAU / A8r BPR, Yelp A3 DirectAU / A7 BPR, and AmazonBook I2's Full partner DirectAU / B14 BPR. Keep each reference's stopping status and reproduction/controlled label visible; do not swap Full losses or choose a Full scoring mode after seeing test.

Frozen definitions: head is the exact top `ceil(.2*N_catalog)` original item IDs ordered by descending **training-only** event count, ties by ascending original item ID; tail is the complement (bottom 80%, up to that boundary rounding). `tail_primary` is the macro mean of `tail_hits_u/min(20,n_tail_truth_u)` **only among evaluated users with at least one held-out tail item**. Catalog coverage is `100*|union_u Top20_u|/N_catalog`, using all evaluated test users and the full original-item catalog denominator. Tail recall is reported in percent and differences in percentage points. Also report uncapped conditional tail recall (`tail_hits_u/n_tail_truth_u`) and the all-user capped-recall tail contribution (`tail_hits_u/min(20,n_all_truth_u)`) separately; neither substitutes for `tail_primary`.

Informed pilot check: RESULTS' PR-E5 addendum reports Yelp repaired GraphHash DirectAU tail **0.29%** versus Full DirectAU raw **1.03%**. The saved test top-20 lists and original split truth reproduce `tail_primary` **0.289988% versus 1.029795%**, over the same 23,011 tail-eligible users out of 30,304 evaluated users; coverage is **21.654226% versus 31.975399%**. These are conditional tail recalls, not contributions to overall recall (the corresponding all-user contributions are 0.153327% versus 0.515562%). The pilot values and final prediction are [I].

Prediction: for **all three datasets**, repaired GraphHash DirectAU `tail_primary` is **below Full DirectAU raw `tail_primary`**; report every final GraphHash seed/init variant, without selecting the most favourable checkpoint. Decision: report the per-dataset differences and paired-user CIs on the same eligible users, including failures of this directional prediction. Preserve the original PR-E5 negative result: the <=.5-point DirectAU tail-change prediction failed on all three pilot datasets. The paper calls the repair a **head-accuracy repair with a tail cost**; make no "tail-safe" or cost-free repair claim. Stage 3 remains an informed check, not a retroactive validation of the original tail prediction.

#### S3-E7 [I]: float32 default; conditional uint8 precision check

The default stated prior-storage cost is **float32: 163,924 bytes, 5.52% of the Gowalla GraphHash table**. The uint8 representation requires **one byte per item plus one extra float32 scale** (40,985 bytes on Gowalla); scale metadata is not free. Report total bytes and the table-relative overhead.

Informed threshold and decision: the "one byte per item" precision claim is **dropped unless every final GraphHash checkpoint passes absolute Recall@20 deviation <= .02 percentage points on both validation and test**, for its final repaired scoring mode (cosine+prior for DirectAU; raw+prior for BPR). Include every retained dataset, seed and initialization variant; one failure, a missing cell or an invalid checkpoint blocks the universal claim. Compare uint8 against float32 at the **same frozen validation-selected beta**, checkpoint, users, masking, arithmetic and tie policy; test never chooses beta or representation. The .02-point tolerance is [I], inherited after observing pilot outcomes, not a new prospective threshold. Any passing uint8 claim must still state the additional float32 scale.

Negative evidence remains: modern Gowalla A1 failed on both validation (−.026409 points) and test (+.024596 points); legacy quantized-validation outcomes are missing. Do not average away a failing checkpoint or substitute the legacy test-only means. Until the complete final-checkpoint condition passes, retain float32 and make no one-byte-per-item claim.

#### Final PR-E8 analysis [I]

Primary family: GraphHash DirectAU cosine+prior versus GraphHash BPR raw+prior on Gowalla, Yelp2018 and AmazonBook. Report paired-user CIs and Holm adjustment across the three datasets. PASS requires positive delta and Holm-adjusted p < .05 on **all three** datasets. Separately, every Gowalla matched seed must have positive delta with its own paired-user CI excluding zero: DirectAU `gowalla__mf__graphhash__directau__s0` / `s1` / `s2` versus BPR `gowalla__mf__graphhash__bpr__s0-p500` / `s1` / `s2`. The legacy patience-100 BPR seed 0 is pilot only. Use the final validation-selected runs and preserve the Stage 1-2 pilot results. Quantization and reweighted evaluations are secondary and must not select the main score from test outcomes.

#### Stage 3 queue contract

- Four sessions, two GPU lists each, account-3 twice and account-4 twice. Priorities are sorted inside each list.
- Each job has a ten-minute planning reserve. Each GPU list's **estimated duration including reserves is <=540 minutes (9 hours)**; its separate **sum of run caps plus reserves is <=660 minutes (11 hours)**. Every AmazonBook DirectAU job is estimated at its full 300-minute cap before adding the reserve. At most one such job is assigned per list where possible. Priorities inside each list remain A2, A8r, B9, B5-B8/new BPR s0-p500, then the remaining groups. Estimates are not observed runtimes or guarantees.
- No Kaggle access is needed by the builder: all jobs train from the ordinary preparation pipeline and need no signed checkpoint URLs.
- Run IDs are new; a pre-existing valid run is skipped, while an invalid/incomplete collision blocks generation rather than being overwritten.
- Registration and snapshot are authorized for this task before push. Only the orchestrator submits Stage 3; this preparation does not submit, cancel or restart kernels.

<!-- END STAGE3 REGISTRATION -->
