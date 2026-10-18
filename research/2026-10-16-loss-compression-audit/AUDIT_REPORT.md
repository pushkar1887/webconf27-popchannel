**Research audit: three combination ideas and the loss × sharing proposal**

Audit date: 16 October 2026. Scope: the two supplied proposal attachments, five supplied PDFs, targeted primary literature, and pinned released source code. This is a research-design audit. No GraphHash, BACO, DirectAU, CCE, or G-Refer real-data training was performed for this audit.

**Verdict**

[Certain] The Gowalla observation in Claude's proposal is accurately transcribed from GraphHash Table 1. The universal claim that DirectAU hurts compressed models is contradicted by the same table. Several mathematical and prior-art claims also require correction.

[Likely] Loss × embedding sharing is the most focused next diagnostic among the supplied candidates. It is not yet a cleared novel method or a submission-ready paper. In particular, a uniqueness correction is already closely related to SUAU, and one collision correction is exactly equivalent to adaptive loss reweighting under the assumptions below.

**1. What was actually checked**

- Both supplied proposal attachments were read. They describe DriftGuardHash, DualCompress, and ExplainHash.
- The supplied PDF methods and evaluation sections relevant to those proposals were inspected. GraphHash Table 1 was also rendered and visually checked; its rendering is saved as graphhash-table1.png.
- Released code was fetched as individual public source files and pinned to repository commits. No repository installation or training pipeline was executed.
- Deterministic synthetic examples checked the finite-pair floor, gradient decomposition, BPR collisions, mixed-volume non-equivalence, raw-dot score sensitivity, and the LightGCN caveat.
- Shiao et al. was checked in its primary PDF. SUAU was checked through its publisher abstract/introduction/section preview and author code. The full SUAU article was not obtained; the SSRN article page returned HTTP 403. A claim that our mathematical analysis is new therefore remains unverified.
- This is not an exhaustive survey of every alignment/uniformity paper or an independent reproduction of any published performance.

**2. The three original ideas**

| Candidate | Correct assessment | Unresolved requirement |
|---|---|---|
| DriftGuardHash | [Certain] Our previous KuaiRand count/context and exposure pilots did not implement temporal GraphHash reassignment. Calling this candidate “rejected twice” is incorrect. [Likely] Dynamic compression is crowded and the candidate is too broad for the remaining short schedule. | A precise drift signal and reassignment rule; comparison with CCE and periodic/triggered reclustering; temporal split; update overhead and memory accounting. |
| DualCompress | [Certain] The proposal combines sharing groups and per-cluster capacity; DML alone does not prove that the entire joint problem is solved. [Likely] Novelty risk is high because mixed dimensions, learned sizes and pruning are established. | A new allocation principle and a physically smaller exported representation, including mapping, projection, optimizer and lookup costs as appropriate. |
| ExplainHash | [Certain] G-Refer reports LoRA training on eight A100s, but the supplied proposal explicitly begins with a graph-evidence audit and adds G-Refer only later. Eight A100s are not a lower bound for this proposed audit. [Likely] The present problem definition is incomplete. | Show how evidence retrieval depends on compressed representations. Embedding compression alone does not remove the original interaction graph. Define faithfulness before generating explanations. |
| Loss × sharing | [Certain] There is a published performance interaction and accessible code. [Likely] This gives the narrowest diagnostic, but a deduplication-only contribution has a serious novelty obstacle. | Separate extra parameter aliasing from ordinary repeated IDs, sampling changes, adaptive loss weighting, and score normalization. Then establish generalization and meaningful recovery. |

[Certain] The earlier reports explicitly bound their conclusions. E0_REPORT.md states that no GraphHash/CCE/BACO execution occurred. MECHANISM_REPORT.md states that two-user groups do not reproduce these systems. EXPOSURE_MF_REPORT.md leaves the broader exposure hypothesis open and reports weak latent-interaction gains. These experiments do not empirically refute DriftGuardHash.

[Certain] DML §3.1 explicitly allows replacing the training mask with the learned hard-coded dimension at the end of training. A dense masked tensor alone does not save allocated bytes; a compact export can. The blanket dismissal of masking is too broad.

[Certain] G-Refer's reported semantic explanation metrics do not by themselves establish that an explanation causally reflects the recommender's decision. SPRec concerns a different LLM preference-optimization mechanism and supplies no direct fix for the proposed hashing phenomenon.

**3. What GraphHash Table 1 establishes**

The following Recall@20 values are in the paper's percentage units. They are published results, not results reproduced in this audit.

| Dataset | MF variant | BPR | DirectAU | DirectAU minus BPR, percentage points |
|---|---|---:|---:|---:|
| Gowalla | Full | 15.617 | 17.984 | +2.367 |
| Gowalla | GraphHash | 9.300 | 7.660 | -1.640 |
| Gowalla | Random hash | 0.766 | 0.478 | -0.288 |
| Gowalla | Double frequency | 3.888 | 3.440 | -0.448 |
| AmazonBook | GraphHash | 3.970 | 4.602 | +0.632 |
| AmazonBook | Double frequency | 1.311 | 1.466 | +0.155 |
| Yelp2018 | Double frequency | 2.478 | 2.536 | +0.058 |

[Certain] These counterexamples rule out “DirectAU always degrades compressed models.” Even Gowalla double-frequency NDCG@20 increases, from 2.532 to 2.677, while its Recall@20 decreases.

[Certain] A more appropriate response variable is the loss × sharing interaction:

\[
I_m =
\big[M(\mathrm{DirectAU},m)-M(\mathrm{BPR},m)\big]
-
\big[M(\mathrm{DirectAU},\mathrm{Full})-M(\mathrm{BPR},\mathrm{Full})\big].
\]

For Gowalla GraphHash, \(I_m=-4.007\) percentage points. This measures the change in DirectAU's advantage relative to the full model. It does not identify the cause. Use matched data, validation, initialization policy and compression maps before interpreting it as an effect of sharing rather than of training-protocol choices.

[Certain] GraphHash already discusses reduced compatibility in §6.1 and sweeps the uniformity weight in Figure 4 on Gowalla and Yelp2018. A new study must compare with well-tuned DirectAU, rather than recover an avoidable bad default.

**4. The missing novelty check: SUAU**

[Certain] Sang et al., “Towards similar alignment and unique uniformity in collaborative filtering,” Expert Systems with Applications 259:125346 (2025), identifies duplicated samples as a source of uniformity bias and proposes a uniqueness strategy. This is a direct precursor to the general idea of deduplicating uniformity.

[Certain] The author code, pinned below, uses torch.unique on original user and positive-item IDs before computing uniformity, at model/SUAU.py lines 137–142. These are entity IDs, not compression aliases. Distinct entity IDs can still share a compressed representation, so alias-aware deduplication is not proved identical to SUAU's implementation. That distinction must be demonstrated and compared; it cannot be assumed to constitute a full contribution.

[Likely] A paper whose only new algorithmic ingredient is “apply unique to bucket IDs” is vulnerable to a straightforward prior-art and incremental-contribution objection. The broader compression-specific mechanism question remains open after this targeted check.

Primary sources:

- https://doi.org/10.1016/j.eswa.2024.125346
- https://www.sciencedirect.com/science/article/pii/S0957417424022139
- https://github.com/ZzYUuuu/SUAU/blob/d1062bd7fedc137894dcd02bf846dd1e1261aec7/SUAU-main/model/SUAU.py#L137-L142

**5. Shiao et al. does not clear the entire novelty question**

[Certain] GraphHash reference 45 is Shiao et al., “Improving Out-of-Vocabulary Handling in Recommendation Systems,” arXiv:2403.18280. Its §5.3 and Table 4 compare BPR and DirectAU for OOV users and report poor DirectAU OOV transfer. This is an earlier related loss-dependent observation.

[Certain] I did not find the proposed compression-specific collision-floor repair in the inspected paper. This finding is narrower than proving that no prior paper has the mechanism or fix. SUAU is the more immediate obstacle to the proposed uniqueness repair.

Primary PDF: https://arxiv.org/pdf/2403.18280

**6. Correct finite-minibatch mathematics**

[Certain] GraphHash's released DAULoss normalizes vectors and applies torch.pdist, exponentiation, a mean, and a log. The diagonal of batch positions is excluded. Repeated entity IDs and different IDs sharing the same parameter row can both produce zero-distance off-diagonal pairs.

For one side of a fixed hard-sharing MF model, let a minibatch contain \(M\ge2\) sampled positions, with \(c_b\) positions mapped to bucket \(b\). Define the fraction of unordered position pairs forced to be equal:

\[
\rho=\frac{\sum_b c_b(c_b-1)}{M(M-1)}.
\]

If \(\pi_{\rm batch}=\sum_b(c_b/M)^2\), then

\[
\rho=\frac{M\pi_{\rm batch}-1}{M-1}.
\]

For \(0\le\rho<1\), let \(Z\) be the mean of
\(\exp[-2\|\hat e_j-\hat e_k\|^2]\) over pairs in different buckets, retaining their original multiplicities. The released uniformity term decomposes exactly as

\[
L_u=\log\big[\rho+(1-\rho)Z\big].
\]

Consequently,

\[
L_u\ge\log\rho
\quad(\rho>0),\qquad
L_u\ge\log\big[\rho+(1-\rho)e^{-8}\big].
\]

[Certain] The second inequality uses the maximum squared distance of four between unit vectors. These are bounds, not a guarantee that the optimum achieves the bound; geometry can impose a larger minimum. At \(\rho=1\), all positions share one representation, \(L_u=0\), the uniformity gradient is zero, and the cross-bucket conditional objective is undefined. Code must explicitly handle empty pair sets.

[Certain] The proposal's \(\sum_b(n_b/N)^2\) is a valid collision probability for independent draws from a uniform-ID population with replacement. It is not automatically the observed collision fraction for the released finite-position-pair objective. Positive-edge sampling also changes bucket masses to degree-weighted masses. Distinguish the population, finite batch, and sampling measure.

A deterministic counterexample uses four distinct IDs tied into two buckets, two IDs per bucket, with antipodal unit embeddings. Here \(\pi_{\rm batch}=1/2\), \(\rho=1/3\), and

\[
L_u=\log(1/3+(2/3)e^{-8})\approx-1.09794
<\log(1/2)\approx-0.69315.
\]

[Certain] This falsifies the claimed log-pi bound when that pi is used for this finite off-diagonal objective. It does not falsify the appropriately defined population-with-replacement bound.

For a fixed assignment and batch, differentiation yields

\[
\nabla L_u
=\alpha\nabla\log Z,\qquad
\alpha=
\frac{(1-\rho)Z}{\rho+(1-\rho)Z}.
\]

[Certain] Gradient attenuation depends on both \(\rho\) and \(Z\), not collision mass alone. Log these quantities separately for users and items. Ordinary repeated IDs produce the same kind of constant pair even in an uncompressed model.

[Certain] Masking forced same-bucket pairs while retaining all other pair multiplicities gives \(\log Z\). Its gradient is exactly matched by scaling the original uniformity gradient by \(1/\alpha\). An implementation using a weighted scalar loss must stop gradients through that coefficient. Because DirectAU averages separate user and item uniformities, exact matching generally needs separate adaptive coefficients for the two sides. A single fixed global gamma sweep is not an equivalent control.

[Certain] Taking one representative per unique bucket also reweights the cross-bucket distribution: frequent buckets no longer contribute more cross pairs. That can change gradient direction as well as magnitude. It is therefore not a pure intervention on attenuation.

[Certain] The deterministic check verified the factorization to maximum absolute gradient error \(5.55\times10^{-17}\) in float64. In the checked example, unique-bucket and multiplicity-preserving cross-bucket gradients had cosine similarity 0.95052. These are mathematical checks on constructed vectors, not evidence of ranking-quality improvement.

**7. BACO's balance is related, not the same objective**

[Certain] BACO §4.1, Eqs. (4)–(6), balances a mixed user-plus-item cluster volume,

\[
\operatorname{vol}(C_k)=
\sum_{u\in U_k}w_u+\sum_{i\in V_k}w_i,
\]

and minimizes its squared concentration together with a connectivity objective. Its hybrid weighting (§4.3) uses different weights for users and items. DirectAU computes separate user and item uniformities under the training sampler.

[Certain] Squaring the mixed volume introduces user–item cross terms. For example, user masses \((.8,.2)\) and item masses \((.8,.2)\), or swapped item masses \((.2,.8)\), both have separate collision-mass sum 1.36. Their mixed squared-volume sums are 2.72 and 2.00. The two objectives are not equal even in this simple example.

[Certain] BACO also adds secondary user clusters (§4.5). Sharing the primary cluster does not by itself force the final user representations to be identical. Any collision statistic must reflect the complete effective assignment and count all memory used.

[Likely] Balancing may reduce certain collision concentrations under specified weights and representations, but the exact “BACO already minimizes DirectAU's floor” statement is unsupported. Rebalancing clusters also changes semantic composition and graph coherence, so a damage-versus-rho plot alone cannot isolate attenuation.

**8. BPR, normalization and LightGCN controls**

[Certain] The claim that an aliased positive/negative problem appears in contrastive losses but not BPR is wrong. With standard dot-product BPR, if positive and negative items share the complete final vector, their scores are equal:

\[
\ell_{\rm BPR}=-\log\sigma(0)=\log 2,
\qquad \nabla\ell_{\rm BPR}=0.
\]

The standard ranking-loss calculation is constant for that triple; regularization may still update the parameters. GraphHash adds a tiny numerical epsilon inside its log, which does not remove the zero ranking gradient. If studying alias-negative masking, include a BPR masking control.

[Certain] Both GraphHash's MFDAU code and the original DirectAU author code use normalized representations in the training objective but raw-dot full-sort scoring. This is not a newly discovered GraphHash-specific implementation bug. Positive per-vector scaling leaves the normalized objective invariant but can change raw-dot rankings. Evaluate raw-dot and cosine scoring on the same checkpoints and inspect norm distributions before attributing all performance differences to collision attenuation.

[Certain] Removing normalization changes the objective's geometry and scale behavior. It is not a clean isolated test of one sharing mechanism.

[Certain] LightGCN in the released GraphHash code propagates over original-node neighborhoods. IDs sharing an initial table row can acquire different final representations. The hard-sharing MF bound cannot automatically be transferred to final LightGCN embeddings by counting initial bucket collisions.

**9. A defensible, staged diagnostic — proposed, not executed**

[Likely] A useful research question is: “When parameter sharing attenuates or reverses DirectAU's benefit, how much is explained by repeated-ID sampling, additional parameter aliasing, loss weighting and inference-score norms?”

| Stage | Minimum work | Scientific purpose |
|---|---|---|
| A: reproduction | Gowalla, Full/Random/GraphHash × BPR/DirectAU, initially one seed; pinned data, split and assignment receipts. Use the actual released commands/configuration. | Verify the reported interaction and measure real runtime. One seed does not establish seed robustness. |
| B: mechanism controls | Well-tuned gamma; SUAU-style original-ID dedup; multiplicity-preserving alias masking; unweighted unique-alias uniformity; exact side-specific adaptive weighting; same-checkpoint raw-dot/cosine scoring. | Separate pre-existing deduplication, sampling changes, scalar weighting and norm sensitivity. |
| C: confirmation | Freeze the selected correction on validation and run at least three seeds. Add AmazonBook, where GraphHash's DirectAU absolute gain is positive, before treating the effect as general. | Estimate seed variation, establish a scope boundary and avoid choosing only failure cases. |
| D: optional scope | Effective multi-hash aliases, BACO with secondary codes, BPR negative masking, then LightGCN with a correctly reformulated argument. | Determine which conclusions survive beyond exact single-lookup MF. |

Log user and item values separately: repeat-ID pair fraction, forced representation-alias pair fraction, their difference, \(Z\), \(\alpha\), gradient norms/angles, vector norms, alignment/uniformity, validation/test ranking metrics, allocated bytes and wall time.

[Certain] Aliases mean the complete parameter representation. A shared component of a double hash is not necessarily a shared complete vector. For summed hash rows, account for the full effective row multiset; for node-dependent propagation, inspect the final representation rather than assuming equality from the initial map.

[Certain] Claude's E3 alone has \(4\times5\times2\times3=120\) fits before full references, hyperparameter searches, fixes and backbone checks. Pairwise uniformity has quadratic cost in batch positions. Runtime from our earlier pointwise MF pilot does not establish runtime for this pipeline.

[Certain] The released commands also use method-specific learning rates and batch settings; DirectAU is selected with model=MFDAU. A faithful reproduction and a controlled mechanism comparison are distinct tasks. Record both convergence opportunity and wall time, and do not silently substitute the earlier MF implementation.

[Likely] Advance only if the anomaly reproduces and the diagnosis explains something beyond original-ID deduplication and the exact adaptive-weighting equivalence. A useful negative or boundary result is possible, but neither a fix nor main-track publishability follows from the existing table.

[Certain] The supplied abstract skeleton contains claims we have not established: a compression-specific cause, general applicability, recovery percentage, and tested datasets/methods. Those remain hypotheses and placeholders.

**10. Evidence receipts and sources**

Supplied PDFs:

- D:/web_2/papers/2412.17245v2.pdf — GraphHash, Table 1 p.6, discussion §6.1 and Figure 4.
- D:/web_2/papers/2604.18351v1.pdf — BACO, §4.1, §4.3 and §4.5.
- D:/web_2/papers/NeurIPS-2023-clustering-the-sketch-dynamic-compression-for-embedding-tables-Paper-Conference.pdf — CCE, dynamic compression algorithm and experimental setup.
- D:/web_2/papers/2502.12586v1.pdf — G-Refer, graph retrieval, RAFT/LoRA and evaluation setup.
- D:/web_2/papers/2412.09243v3.pdf — SPRec, LLM preference-optimization mechanism.

Pinned released source:

| Repository | Commit | Inspected targets |
|---|---|---|
| snap-research/GraphHash | 7a1d5eb48dce9ab4c4a453d4482dd0d3bdec4467 | loss.py; models/mf.py, mfdau.py and lightgcn.py; training/train_test_dau.py; train.py; commands.sh; relevant configs |
| ZzYUuuu/SUAU | d1062bd7fedc137894dcd02bf846dd1e1261aec7 | model/SUAU.py; utility/losses.py; utility/trainer.py; README |
| THUwangcy/DirectAU | 60c964bb07abc0eda9b5044b6843fd9c4476a167 | recbole/model/general_recommender/directau.py |
| snap-research/improving-inductive-oov-recsys | 982bd7e67e25209551da0edd13b90b698e0d39ae | RecBole/recbole/model/general_recommender/directau.py |

Additional primary references:

- DirectAU: https://arxiv.org/abs/2206.12811
- Shiao et al.: https://arxiv.org/pdf/2403.18280
- SUAU: https://doi.org/10.1016/j.eswa.2024.125346
- DML: https://arxiv.org/html/2510.15308v1, §3.1.
- Mixed Dimension Embeddings: https://arxiv.org/abs/1909.11810
- PEP, Learnable Embedding Sizes for Recommender Systems: https://arxiv.org/abs/2101.07577

Verification artifacts in this folder:

- local_pdf_manifest.json: supplied PDF hashes, titles and page counts.
- source_code_receipts.json and directau_author_receipt.json: initial source-fetch receipts.
- audit_manifest.json: complete cached-source and generated-audit artifact hashes, with explicit execution scope.
- check_collision_algebra.py and algebra_checks.json: reproducible deterministic mathematical checks.
- graphhash-table1.png: rendered table inspected during the audit.

[Certain] Status at the end of this audit: literature/code inspection completed for the bounded scope above; synthetic algebra checked; real-data reproduction, repair efficacy, runtime, novelty clearance and submission readiness unestablished.
