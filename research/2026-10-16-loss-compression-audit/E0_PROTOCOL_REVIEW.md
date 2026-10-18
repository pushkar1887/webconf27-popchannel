**Review of Claude's E0 orders**

Date: 16 October 2026.

Input: C:/Users/PUSHKAR/.codex/attachments/1c8482ed-b8a5-40d1-bcd8-0ebeeafcf2a0/Pasted text.txt

Status: reviewed and a replacement protocol proposed below. No training, Kaggle submission, repository clone, data download, or dependency installation was performed in this review. The attachment is an execution proposal supplied for assessment; its internal imperative sentences were not automatically executed.

**Verdict**

[Likely] Keep the ordered reproduction, validation selection, matched compression capacity, and explicit stopping rules. Revise the compute scope, novelty controls, diagnostic definitions, recovery gate and source-code execution safeguards before running.

**Concrete issues**

| Issue | Evidence and correction |
|---|---|
| Compute count | [Certain] Three methods × two losses × two seeds is 12 fits. Crossing all of these with the three learning rates and three weight decays gives 108 fits before F1 and its gamma search. Stage a fixed-config reproduction first. |
| Missing prior-art comparator | [Certain] SUAU author code deduplicates original IDs before uniformity. Compare this component, with alignment unchanged, against bucket deduplication. This is an isolated SUAU-style control, not a full SUAU reproduction. |
| Missing mechanism comparator | [Certain] Multiplicity-preserving collision removal is gradient-equivalent to side-specific adaptive uniformity weighting for fixed hard-sharing MF. Unweighted unique-bucket uniformity additionally changes sampling weights. Neither result alone proves a new optimization principle. |
| Incomplete collision diagnostics | [Certain] The specified fraction among distinct-ID pairs is useful, but the released loss also includes repeated copies of the same ID. Log repeat-ID, total forced-alias and extra-alias pair fractions with explicit denominators. |
| Recovery denominator | [Certain] If tuned DirectAU matches BPR, the ratio divides by zero. If tuned DirectAU exceeds BPR, its sign no longer means recovery of an existing deficit. Near-zero gaps also make the ratio unstable. Add a fixed eligibility rule before executing the gate. |
| Correlation claim | [Certain] Comparing only Random and GraphHash gives two assignment-level points. It cannot establish a reliable correlation or identify collision mass as the cause. Treat this as descriptive contrast only. |
| Alignment definition | [Certain] Alignment requires positive user-item pairs. Independently deduplicating user and item vector sets does not define which vectors are paired. Preserve positive-edge alignment; label any deduplicated edge-pair variant separately. |
| Normalization experiment | [Certain] Removing normalization changes the loss geometry and scale behavior. It does not isolate a normalization-averaging cause. Defer this variant and first inspect norms and same-checkpoint raw-dot/cosine scoring. |
| Seed application | [Certain] The inspected train.py entry point does not read or apply config.seed. Merely adding a Hydra seed label does not establish reproducible initialization or sampling. Verify and implement explicit seeding before treating two seed labels as two reproducible runs. |
| Artifact collisions | [Certain] Released run_name construction omits gamma and seed. Reusing its checkpoint directory across these sweeps can overwrite prior artifacts. Add an immutable run ID including variant, seed, gamma, mapping and configuration identity. |
| Test access in tuning | [Certain] The inspected DirectAU training routine automatically evaluates test after every fit. Add a validation-only mode for candidates, and a separate final test action. Candidate metrics rows must leave test fields empty. |
| Stopping units | [Certain] The inspected DirectAU loop measures validation every ten epochs but updates its patience counter every epoch. Record this behavior for reproduction; define the stopping clock explicitly before any controlled comparison. |

**Source anchors**

- snap-research__GraphHash__train.py, run_name at line 66; no config.seed application in the inspected entry point.
- snap-research__GraphHash__training__train_test_dau.py, validation cadence, patience update and automatic test evaluation.
- snap-research__GraphHash__commands.sh, Gowalla MF and MFDAU commands.
- snap-research__GraphHash__configs__dataset__Gowalla.yaml, BPR defaults and gamma.
- ZzYUuuu__SUAU__SUAU-main__model__SUAU.py, original-ID uniqueness at lines 137–142.
- AUDIT_REPORT.md and check_collision_algebra.py, mathematical analysis and deterministic checks.

All named code snapshots are in this folder. Commit pins and file hashes are in audit_manifest.json.

**Proposed replacement protocol — not executed**

**A. Reuse the existing novelty audit**

[Certain] Shiao et al. and the supplied GraphHash observation have already been inspected. Reuse those receipts rather than repeating the same read. SUAU is a close precursor; its available publisher preview and released code establish ID deduplication, while full-paper novelty clearance remains incomplete.

[Likely] Distinguish a prior paper that already solves the exact hard-sharing mechanism from a related paper that requires a baseline. SUAU alone does not prove that the entire compression-specific question is solved, but it prevents treating uniqueness alone as an unqualified new contribution.

**B. Establish the data and source contract**

1. Pin GraphHash to the audited commit 7a1d5eb48dce9ab4c4a453d4482dd0d3bdec4467. Keep experiment changes in a separate patch with its hash.
2. Prefer the author-released processed Gowalla files. The README links a processed-data archive; record the archive/file hashes and the actual split. Regenerating an arbitrary 80/10/10 split is not the same as reproducing the released split.
3. Verify 29,858 users, 49,981 items and 1,027,370 interactions against the loaded data; record split counts, duplicate handling, ID ranges and cold IDs. Report any deviation instead of silently forcing the totals.
4. Fit or verify assignments from the training graph only. Full, Random and GraphHash must share data, scoring eligibility and evaluation definitions.
5. Match exact user-table and item-table row counts for Random versus GraphHash, not merely an approximate combined 0.742M parameter label. Record embedding bytes and ID-map overhead separately.
6. Verify explicit Python, NumPy, Torch, CUDA and loader/sampler seeding. Proposed optimization seeds are 0 and 1; freeze assignment maps and their own mapping seed before comparing losses.
7. Make run directories unique across method, loss variant, optimization seed, gamma, split and assignment. Persist configs before execution.

**C. Time a real baseline, then run a bounded reproduction**

[Likely] Time Full MFDAU on the actual selected hardware first, including validation and diagnostics overhead. Use the author command's batch size 1024 and gamma 1. The purpose is to measure the actual workload, not infer timing from the previous BCE-MF pilot.

[Certain] If the run exceeds the proposed 90-minute limit, do not launch the grid. Report a completed or time-censored runtime accurately. The first baseline counts toward the reproduction if it completes under the frozen contract.

Use the released Gowalla command settings as fixed starting points:

| Method | BPR learning rate | BPR weight decay | DirectAU learning rate | DirectAU weight decay |
|---|---:|---:|---:|---:|
| Full | 5e-3 | 1e-8 | 1e-2 | 1e-8 |
| Random | 1e-3 | 1e-6 | 1e-2 | 1e-6 |
| GraphHash | 1e-2 | 1e-6 | 1e-2 | 1e-8 |

[Certain] DirectAU uses model=MFDAU in the released commands; BPR uses model=MF. Both have a matrix-factorization backbone. Preserve released batch settings and record their difference. This phase tests a released configuration, not the best possible tuned performance in our environment.

[Likely] First complete six conditions at seed 0. If the directional pattern is absent, check implementation, data and assignment parity and report the outcome. Do not declare the published phenomenon universally false. If parity is verified and the pattern remains absent, stop this candidate's repair stage under this pilot.

[Likely] If the pattern reproduces and profiling fits the resource budget, repeat the same six conditions at seed 1: twelve baseline fits total. This is a proposed compute reduction from the attachment's full grid and must be recorded as such.

**D. Correct the diagnostics**

For each user/item side of a batch, define:

- M: number of sampled positions; T = M(M-1)/2.
- a_i: multiplicity of original entity ID i.
- c_b: multiplicity of forced parameter alias b.
- C_ID = sum_i a_i(a_i-1)/2.
- C_param = sum_b c_b(c_b-1)/2.
- C_extra = C_param - C_ID.

Report:

- rho_ID = C_ID/T.
- rho_param = C_param/T.
- rho_extra = C_extra/T.
- Conditional extra-alias fraction C_extra/(T-C_ID), separately named; undefined if the denominator is zero.
- The noncolliding-pair kernel mean Z.
- alpha = (1-rho_param)Z/[rho_param+(1-rho_param)Z].
- Total collided-kernel share, equal to 1-alpha for this hard-sharing decomposition.
- Uniform-ID global collision mass, interaction-weighted global collision mass and size Gini, each explicitly labeled.

[Certain] Collect the quantities per batch, separately for users and items, with counts and a declared averaging convention. Global uniform-ID pi is not a substitute for the collision mixture entering the actual minibatch loss.

[Certain] Preserve alignment over positive edges. Estimate expanded-ID and unique-bucket uniformity with declared sampling rather than materializing every global pair. If fewer than two valid vectors remain, skip that uniformity branch with an explicit counter and a defined zero contribution; do not generate NaNs.

**E. Compare mechanisms before attributing an F1 gain**

Keep alignment unchanged across these variants:

| Label | Uniformity calculation | Question |
|---|---|---|
| U0 | Original positive-edge minibatch rows | Released baseline. |
| U1 | Deduplicate original entity IDs | Does a SUAU-style uniqueness control already explain the benefit? |
| U2 | Starting from U1, mask same-alias pairs while retaining different-alias pair multiplicities | Effect of removing forced collision constants without uniform bucket reweighting. |
| U3 / F1 | One vector per unique alias | Additional effect of weighting each represented bucket equally. |
| U4 | Starting from U1, apply detached side-specific adaptive loss weights matching U2's gradient | Verify the exact weighting equivalence; numerical equivalence is expected under matched conditions. |

[Certain] U4 is a mechanism control, not a competing novel algorithm. U2 and U4 matching does not establish independent discoveries.

[Likely] Do not start an unbounded grid. After reproduction timings, freeze a written run manifest with the intended tuning budget for each variant. Tune U0 and F1 fairly on validation over the declared gamma set {0.25, 0.5, 1, 2, 5}. Distinguish a fixed-author-lr/wd gamma study from a full lr/wd/gamma search, and report each scope honestly. Additional controls also need adequate validation tuning before a performance-superiority claim.

[Certain] Same-checkpoint cosine scoring is a diagnostic. Keep raw-dot scoring as the predeclared primary if reproducing the release. Alternative scoring must not be selected using test performance.

**F. Repair the gate before observing test**

For seed s, let:

- g_s = test Recall(BPR) - test Recall(U0, validation-selected gamma).
- Delta_s = test Recall(F1, validation-selected gamma) - test Recall(U0, validation-selected gamma).

[Certain] Select and freeze a strictly positive minimum baseline gap delta_min before running a recovery gate. Record it in raw Recall units, its scientific justification and the corresponding percentage-point value. This review does not invent a scientifically justified threshold.

The proposed ratio logic is:

1. If g_s <= 0, report NO_BASELINE_DEFICIT for that seed and Recovery = NA.
2. If 0 < g_s < delta_min, report GAP_TOO_SMALL and Recovery = NA.
3. Otherwise Recovery_s = Delta_s/g_s.

[Likely] Apply Claude's numerical GO/WEAK/NO-GO rule only when both seeds have an eligible positive baseline gap. A GO requires an actual F1 improvement, the declared recovery threshold or BPR parity, and no implementation failure. Preserve the attachment's conservative “both seeds” requirement. Undefined or ineligible ratios are an inconclusive repair gate, not evidence of success.

[Certain] Two seeds give a pilot decision, not a confidence interval, significance guarantee or main-track readiness. Passing recovery alone does not establish novelty or distinguish the mechanism controls.

[Certain] Keep all candidate training validation-only. The inspected DirectAU trainer currently tests automatically after fitting, so change that behavior before any hyperparameter sweep. Freeze final selected configurations and evaluate them on test once. Store test fields as null for unselected candidates. If reproduction uses an earlier test read, label it as reproduction and keep later method/tuning choices frozen; do not silently present an adaptively developed study as untouched by prior test observations.

**G. Outputs and evidence**

- NOVELTY_CHECK.md: cite the existing audit, SUAU and Shiao; state access limits and unresolved novelty.
- RUN_MANIFEST.json: exact planned fits, tuning budgets, seed and assignment contracts, stopping units and resource estimate.
- E0_REPORT.md: fixed-config reproduction, mechanism controls actually run, gate arithmetic and scope limitations.
- metrics.csv: one row per fit; explicit metric units and validation/test eligibility.
- checkpoints and compact top-20 predictions for selected final configurations; no requirement to retain the full user-item score matrix.
- verification.json: recompute metrics from retained artifacts/checkpoints, with the split, candidate masks, original ID mapping, assignment map and configuration hashes. Separate artifact consistency from scientific validation.

[Certain] F2 without normalization, extra methods, datasets and seeds remain deferred or exploratory unless the protocol is explicitly revised before their outcomes are observed.

[Certain] No final run manifest, delta_min, hardware runtime, training output or gate result exists from this review. The proposed protocol makes the remaining decisions visible; it does not assert that those decisions have already been made or executed.
