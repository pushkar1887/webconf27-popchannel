# Review of the proposed P1-P5 expansion

17 October 2026. Existing code/results and primary literature were reviewed. No experiment was submitted, cancelled, or modified during this review.

## Decision

[Certain] The supplied results do not establish the proposed mechanism. [Likely] The scoring improvement merits further investigation, but launching the entire proposed expansion before resolving the novelty boundary is poor use of the compute budget.

## Closest precedent found

[PRISM: On the Role of Weight Decay in Collaborative Filtering: A Popularity Perspective](https://arxiv.org/html/2505.11318v2), KDD 2025, is a closer precedent than the previously cited PDA. Sections 3.1.2 and 3.2 explicitly examine how normalized recommendation objectives, including DirectAU, acquire magnitude-popularity associations through optimization. Section 4.2 supplies popularity-aware magnitude initialization. This removes the broad full-model norm mechanism from our novelty claim.

Primary anchors:

- [Versioned paper](https://arxiv.org/html/2505.11318v2): sections 3.1.2, 3.2, 4.2.
- [Official code](https://github.com/snap-research/PRISM): models.py initializes embedding magnitudes using supplied popularity-derived values.
- [Official KDD 2025 listing](https://kdd.org/kdd2025/research-track-papers-2/): confirms the paper's venue and DOI 10.1145/3711896.3737068.

[Certain] The inspected version does not establish our specific hashing effect. [Likely] A compression-specific loss of item information might remain a contribution, but it requires direct comparison rather than an inference that a different setting automatically implies novelty.

[DNA](https://cau.scholarworks.kr/item/f5ed8bb7-57e6-401c-bb9d-bf0e84c9edf1) is also close to P5: its abstract describes item normalization plus an item-specific learned bias to restore useful popularity signals in GCN recommendation. That is not proof P5 is identical, but it rules out treating normalized scoring plus a scalar bias as self-evidently new.

## Why the new correlations are not causal identification

The current Full and GraphHash checkpoints differ in seed, training duration, parameterization, and effective update statistics. An item-level correlation over shared buckets also repeats the same norm across several IDs. Reduced correlation is consistent with compression, but cannot identify why the objective comparison changed.

For BPR, correlation 0.054 does not imply popularity is stored in a particular direction. It does not even imply the norm has no useful nonlinear or conditional relationship to popularity. The observed ranking change establishes that norms matter, not what information they encode.

Standard deviation 0.0202 percentage points was measured for plain GraphHash DirectAU test recall. It is not the standard deviation of the new prior-adjusted scores, which are available for only one seed.

## Cheap distinction between competing explanations

Let h(i) be an item's bucket and p_i = log(1+c_i) its training log count. Define:

`mu_b = mean of p_i over items with h(i)=b`,

`r_i = p_i - mu_{h(i)}`.

Then the existing successful score is exactly:

`cos(u,i) + beta*mu_{h(i)} + beta*r_i`.

The bucket component changes bucket preferences. The residual component restores distinctions between original items sharing an embedding. Compare the existing scorer against the bucket-only component and a within-bucket shuffle of the original p_i values. Preserve masks, checkpoint, and the parameter-selection contract. Use several predeclared shuffle seeds; do not pick the shuffle giving a convenient answer.

If item-specific residuals account for much of the gain, the result supports the value of retaining item identity information. It does not isolate accidental optimizer norms as the historical cause. If bucket-only values perform similarly, the result is more consistent with bucket calibration. Either outcome must be reported without rewriting the hypothesis to fit it.

For a closer intervention on the norm explanation, hold embedding directions fixed and compare the additive prior with a positive multiplicative item-scale replacement. This is a diagnostic inspired by the geometry; it is not an implementation or reproduction of PRISM's training initialization.

## Assessment of each proposed priority

| Proposal | Decision | Needed correction |
| --- | --- | --- |
| P1: extend BPR beta | Keep first | Use validation to bracket the optimum. If the new best is again a boundary, 1/2/5 is not sufficient by itself. Freeze the selected rule before reporting test. |
| P2: two more DirectAU seeds | After P1 and diagnostic controls | Use the same checkpoint selection rule for all seeds. Current script saves best_seedN but evaluates the final epoch unless PATIENCE is enabled; loading a best checkpoint for the prior can silently change the comparison. |
| P3: more hashing methods | After the mechanism controls | Replicate useful observations, but two more methods do not identify the cause. Include the same prior baseline and memory accounting. |
| P4: AmazonBook | Valuable boundary check, conditional on a viable contribution | Predeclare norm/cosine sensitivity, correlation change, and prior benefit. 'Either correlation survives or popularity matters less' requires fixed measurable definitions; otherwise either outcome can be rationalized. A cap is not a convergence criterion. |
| P5: jointly learned item bias with BPR | Defer as a proposed main method | Requires an auxiliary-loss coefficient, sampling contract, bias regularization, and matched-capacity controls. It adds a different training objective and overlaps established bias models and DNA. |

## P5 is not merely an additional DirectAU parameter

For `s_ui = cos(u,i) + b_i` with DirectAU unchanged, `d L_DirectAU / d b_i = 0`. Thus DirectAU alone cannot learn the proposed biases. Adding BPR gives a hybrid objective such as:

`L = L_DirectAU + lambda_rank * L_BPR(s_ui, s_uj) + lambda_bias * norm(b)^2`.

The auxiliary BPR term can also update the embedding directions unless those gradients are explicitly blocked or the directions frozen. An improvement would therefore not isolate a scalar's effect without additional controls. Bias-only, frozen-direction bias fitting, and an equal-capacity BPR comparator are relevant if P5 is pursued.

## Timing supported by existing observations

Three completed plain Gowalla MF DirectAU 30-epoch runs averaged **19.13 minutes** under their reported training-loop timer. Two comparable sequential repetitions would therefore be approximately **38.26 minutes**, plus setup and scoring. This is an estimate anchored to previous execution, not a guarantee on another runtime. The prior-adjusted seed spread is not yet measured.

No measured runtime here supports a precise AmazonBook completion estimate. Two 75-minute caps are resource limits, not a prediction of wall time or evidence that the models converge. Avoid turning the proposed 1-2 hours into an assurance.

## Abstract and deadline corrections

Remove the claims that the norm-growth mechanism is established and that cross-method/cross-dataset characterization is complete. Compare against BPR with the same additional information, including the extended validation-selected prior. Report the extra approximately **160 KiB / 5.52%** embedding bytes rather than calling the overhead negligible.

An evidence-compatible description is: 'On a Gowalla pilot, per-item popularity calibration raises GraphHash DirectAU from 7.55% cosine Recall@20 to 13.30%. We investigate whether the gain reflects norm information lost under sharing or item distinctions restored by the additional scalar.' This is a research-status description, not a finished submission abstract.

The [official research call](https://www2027.thewebconf.org/research-track-papers/) lists 18 October 2026 for abstracts and 25 October 2026 for papers, end-of-day AoE. These convert to **Monday 19 October 2026, 17:29 IST** and **Monday 26 October 2026, 17:29 IST**, respectively. The pasted abstract deadline conversion is correct; abstract registration does not mean the full empirical paper must be finished by that same Monday.
