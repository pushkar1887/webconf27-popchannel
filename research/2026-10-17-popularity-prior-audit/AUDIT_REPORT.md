# Audit of the new popularity-prior results

Date: 17 October 2026. Scope: read existing sources, completed Kaggle outputs, and local results. This audit launched no training or scoring jobs and changed no existing experiment code.

## Verdict

[Certain] The scoring improvement is present in saved outputs. [Likely] This merits a narrowly controlled follow-up. [Certain] Neither the proposed optimizer mechanism nor compression-specific novelty has been established.

The new candidate is distinct from the earlier unique-bucket uniformity fix. The weak result for that fix does not invalidate this new scoring observation.

## Evidence inspected

| Alias / origin | Notebook or file | Result |
| --- | --- | --- |
| Father | graphhash-popprior-gh-1017 | COMPLETE; GraphHash DirectAU scoring sweep |
| University | graphhash-popprior-full-1017 | COMPLETE, but output is an empty object; no model evaluated |
| University | graphhash-popprior-full2-1017 | COMPLETE; Full DirectAU and GraphHash BPR scoring sweeps |
| University | graphhash-mech-1017 | COMPLETE; norm correlations and GraphHash BPR raw-dot prior sweep |
| Local | experiments/2026-10-16-fast-check/results/poponly.json | Saved popularity-only result; source inspected, not rerun by this audit |

The Full2 output's `GraphHash` key refers to **BPR**, not DirectAU. Its checkpoint path identifies `graphhash-fast-bpr-author-1016`. Treating all `GraphHash` output keys as the same loss would conflate different models.

Downloaded sources and outputs are retained under the alias/slug folders next to this report. `analysis.json` contains computed differences. The local popularity-only result and source were copied here for provenance.

## Verified scores

All values below are percentages on the **test** split. The score named Recall@20 follows the repository's capped denominator, min(20, number of relevant test items). This is the established reproduction metric, not a new standard-recall calculation.

| Model | Raw dot Recall@20 | Cosine Recall@20 | Cosine + prior Recall@20 | Selected beta |
| --- | ---: | ---: | ---: | ---: |
| Full DirectAU | 17.8951 | 12.9959 | 19.5444 | 0.1 |
| GraphHash DirectAU | 7.5325 | 7.5456 | 13.3024 | 0.2 |
| GraphHash BPR | 9.3423 | 6.0829 | 10.2228 | 0.1 |

Additional controls already available:

- GraphHash BPR, **raw dot + prior**: Recall@20 **11.2388**, NDCG@20 **6.9486**, beta **0.5**.
- GraphHash DirectAU, cosine + prior: NDCG@20 **8.5263**.
- Full DirectAU, cosine + prior: NDCG@20 **12.5899**.
- Popularity only: saved local test Recall@20 **2.97**, NDCG@20 **1.70**; validation **3.06 / 1.75**. These local results were saved rounded to two decimals.

[Certain] GraphHash DirectAU plus the prior exceeds the tested BPR raw-dot prior setting by **2.0636 percentage points**. [Certain] BPR's best beta is the **upper boundary** of its tested grid, and its validation recall is still increasing there. It is therefore premature to describe BPR as sufficiently tuned. Identical numeric beta grids also need not cover equivalent strengths for raw dot and cosine, because their score scales differ.

[Certain] Popularity alone does not explain the 13.3024 score: the saved popularity-only result is 2.97. [Certain] This comparison does not isolate why the combination helps.

## Source checks

The inspected scorers:

1. Count popularity with `torch.bincount(parts[0][1])`: **training interactions only**, counted per original item ID.
2. Compute `cosine(u,i) + beta * log1p(count_i)`, or raw dot for the explicitly labeled BPR control.
3. Select beta by validation Recall@20 over `[0, .01, .02, .05, .1, .2, .5]`.
4. Mask training items for validation and training plus validation items for test.
5. Evaluate beta zero and the validation-selected beta on test. The source does not choose beta from test results.
6. Reconstruct the GraphHash mapping from training data with the pinned clustering setup. Raw test results match the previously saved checkpoints' scores.

[Certain] These checks establish the inspected scoring contract. They are not a new end-to-end rerun, proof against every possible provenance error, or confirmation of a causal explanation.

The current local `fast_run.py` now uses `best_seed{seed}.pt`. This is a prospective fix. It does not reconstruct checkpoints that older multi-seed jobs overwrote.

## The compression interaction depends on the comparison

Replacing each checkpoint's normal raw-dot score with cosine plus prior gives:

- GraphHash DirectAU: **+5.7699 pp**.
- Full DirectAU: **+1.6494 pp**.
- Difference of gains: **+4.1205 pp**.

[Certain] This supports an operational observation: the new scoring rule repairs more of the compressed checkpoint's performance relative to its normal scorer.

Adding the prior to a common cosine baseline instead gives:

- GraphHash DirectAU: **+5.7568 pp**.
- Full DirectAU: **+6.5485 pp**.
- Difference of gains: **-0.7917 pp**.

[Certain] The prior does not help compressed DirectAU more in absolute points when both models start from cosine. These comparisons answer different questions. The first changes both normalization and the prior; it is not a pure effect of adding popularity to the original raw-dot scorer.

The pasted chat used Full's approximately 17.6 **validation** score alongside test scores. The completed outputs now supply the correct Full raw-dot **test** value, 17.8951. All comparisons in this report use test consistently.

Full DirectAU is seed 0 at the best checkpoint from the interrupted longer run, epoch 120. GraphHash DirectAU is seed 2, epoch 30. These are useful diagnostics, but the interaction is not a matched-seed, matched-convergence causal estimate.

## Norm evidence and limits

| Checkpoint | Item norm versus training count, Spearman |
| --- | ---: |
| Full DirectAU | 0.814957 |
| GraphHash DirectAU | 0.270229 |
| GraphHash BPR | 0.054099 |

[Certain] Full DirectAU has a strong positive association in this checkpoint, and GraphHash DirectAU a weaker positive association, not zero. [Likely] This is consistent with some item-level popularity information being unavailable in shared norms. [Certain] Association alone does not establish that this caused the loss-ranking anomaly.

[Certain] BPR losing accuracy under cosine does not show that its useful norm information is popularity. The low measured norm-count correlation makes that identification especially unsupported here.

For fixed user u, raw score is `norm(u) * norm(i) * cosine(u,i)`. The positive user norm is constant across candidate items. Item norms can affect ranking. However, multiplying cosine by an item norm and **adding** log popularity are different interventions.

The optimizer argument in the pasted chat is not valid as stated. For a differentiable, scale-invariant loss and a nonzero embedding w, `w dot grad L = 0`. Under plain SGD without momentum or decay, this implies:

`norm(w - eta*g)^2 = norm(w)^2 + eta^2 * norm(g)^2`.

That relates growth to accumulated gradient energy, not simply exposure counts. Actual runs use **Adam with weight decay**. Momentum, coordinate-wise preconditioning, and decay invalidate the inference that every update is perpendicular or every norm must increase. A simple first-Adam-step counterexample is w=(2,1), g=(1,-2), with w dot g=0. Ignoring epsilon and using eta=.01 gives w'=(1.99,1.01), whose squared norm **decreases** from 5 to 4.9802. Small positive epsilon does not change the conclusion.

The broader issue of norm dynamics under scale invariance is established in [AdamP, ICLR 2021](https://clovaai.github.io/AdamP/). This does not rule out the observed popularity association; it rules out treating the proposed explanation as an already proved theorem for these runs.

## A second explanation that must be separated

For items i and j sharing a GraphHash item bucket, their cosine scores for any fixed user are identical. Adding **per-ID** popularity makes their score difference:

`s(u,i) - s(u,j) = beta * [log(1+c_i) - log(1+c_j)]`.

[Certain] The successful scorer therefore restores distinctions between original item IDs that the shared embedding cannot represent. This breaks structural ties and adds item-specific information. Its benefit need not come solely from restoring a lost norm channel.

[Certain] A bias or length **per shared bucket** cannot reproduce those within-bucket distinctions. The proposed learned per-bucket follow-up is not equivalent to the successful per-ID scorer.

For this loaded dataset there are **40,981 items**, not approximately 50,000. One additional float32 scalar per item costs **163,924 bytes = 160.08 KiB**, about **5.52%** of the 742,272-float compressed embedding tables. These are incremental embedding/scalar bytes, excluding mappings, optimizer state, and other runtime memory. Quantizing to one byte is an untested proposal.

## Novelty boundary

[Certain] Follow-up search found a substantially closer precedent: [PRISM, KDD 2025](https://arxiv.org/html/2505.11318v2). It already examines magnitude-popularity encoding with normalized recommendation losses, including DirectAU. The full-model norm story cannot be presented as our new discovery. See `P1_P5_REVIEW.md` for the revised contribution boundary and proposed experiment priorities.

[Certain] Combining published papers can yield a contribution, but award status and recency cannot establish that a particular combination is new.

- [PDA, SIGIR 2021](https://arxiv.org/abs/2105.06067) already adjusts recommendation scores with popularity during inference. The novelty cannot be merely adding a popularity prior.
- The publisher-indexed abstract and introduction of [GCNs meet long-tail: Embedding norm bias in GCN-based recommendations, 2026](https://doi.org/10.1016/j.asoc.2025.114226) describe normalizing item embeddings and restoring useful popularity information through an item-specific bias. Its setting is GCNs, not this compressed MF diagnosis. The full article was not accessible during this audit. Treat it as a close lead requiring full comparison, not proof that the compression-specific question is solved.
- [Likely] A potentially defensible contribution is identifying which **per-item information must survive embedding sharing**, explaining a specific objective-by-compression failure under controls, and measuring the accuracy-memory tradeoff. This is a candidate research direction, not an established novelty claim.

## Smallest next evidence needed

No new training is justified solely by these observations. The next checks can use existing checkpoints:

1. Finish BPR's validation-only tuning so its optimum is bracketed rather than at the grid boundary. Predeclare any further test evaluation; do not repeatedly tune from test differences.
2. Separate per-item popularity from per-bucket popularity and shuffled-within-bucket item priors. This distinguishes generic calibration from restoring useful distinctions among items that share an embedding.
3. Examine the norm association at bucket level and distinguish it from within-bucket item differences. Correlation computed over expanded item IDs is descriptive, not independent evidence from one observation per item.

If these controls support the effect, a small matched replication and a second dataset become warranted. Until then: retain the promising empirical candidate, remove the claim 'mechanism confirmed', and do not present an award-paper merge as the novelty proof.
