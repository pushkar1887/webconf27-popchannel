# Citable backing for the behavioral-compatibility version of Idea 2

Checked: 15 October 2026. Status: primary-source claim verification; no proposed experiment has been executed. This is a targeted evidence map, not an exhaustive novelty review. Author statements, observed design boundaries, and our deductions are distinct.

## Recommended scope

[Likely] The strongest question is whether sharing decisions learned from historical behavior preserve user-conditioned predictions when context frequencies change, under a fixed total resource budget. Literature establishes related restrictions and coverage dependence, but does not yet establish the magnitude or practical significance of this exact proposed failure.

## 1. Calibration coverage constrains behavioral merge decisions

**Source:** *UniMoMo: Expert Merging-Based MoE Acceleration for Large Recommendation Models*, arXiv:2608.08627v1, §5, “Limitations and ethical considerations.”

**Primary text:** https://arxiv.org/html/2608.08627v1

[Certain] The authors explicitly state that the partition depends on representative calibration traffic and that absent behavior cannot influence affinity or exposure estimates. This concerns MoE expert merging, not user-embedding sharing.

**Citation-ready paraphrase:** “UniMoMo identifies calibration-traffic representativeness as a limitation of behavioral expert merging.”

[Likely] **Motivated experiment:** vary calibration coverage of contexts at matched sample count; evaluate later shifts in context frequency. Distinguish contexts that are entirely unseen from contexts that are rare but sufficiently observed.

**Do not claim:** UniMoMo demonstrates that compressed user identities fail under context-frequency shifts. That is our proposed extension.

## 2. Sparse data limits reliable behavioral-policy estimation

**Source:** Yuan, Schneider, and Rizoiu, *Behavioral Homophily in Social Media via Inverse Reinforcement Learning: A Reddit Case Study*, WWW 2025; inspected arXiv:2502.02943v1, §5, “Conclusion.”

**Primary text:** https://arxiv.org/html/2502.02943v1

[Certain] The authors acknowledge substantial data requirements as state/action spaces grow and lack of guaranteed meaningful convergence on sparse data. They also discuss platform-data-access constraints.

**Citation-ready paraphrase:** “Yuan et al. identify data sparsity and state/action-space size as limitations of IRL-based behavioral inference.”

[Likely] **Motivated experiment:** compare smoothed conditional counts, a small conditional predictor, and optional IRL as observations per user/context decrease. Measure predictive error, uncertainty, sharing decisions, and cost.

**Do not claim:** the original paper establishes that IRL is unnecessary, or that our estimator solves sparse behavioral inference.

## 3. Visitation-weighted similarity emphasizes frequent contexts

**Source:** the same Behavioral Homophily paper, §3.6, “Homophily Inference,” SWKL definition.

[Certain] SWKL weights policy divergence using empirical state visitation; rare states receive lower weight to reduce noise. This is a documented methodological choice, not an author-admitted failure.

**Citation-ready paraphrase:** “The SWKL behavioral-similarity measure emphasizes frequently visited states and downweights infrequent states.”

[Certain] A small historical weighted average does not imply a uniform bound for each state. If visitation weights change, the same per-state differences can yield a different average distortion. This is our mathematical deduction; it is not an empirical finding reported by the paper.

[Likely] **Motivated experiment:** hold conditional behavior fixed, vary context frequencies, and compare historical-weighted versus supported-context-sensitive sharing. Include uncertainty controls so noise is not mistaken for an important rare behavior.

**Do not claim:** the paper demonstrates rare-context failures in compressed recommenders or omits all temporal analysis. It already includes §4.4, “Homophily Stability Over Time.”

## 4. Random-split compression evaluations do not establish temporal robustness

**Sources:**

- *GraphHash: Graph Clustering Enables Parameter Efficiency in Recommender Systems*, arXiv:2412.17245v2, Appendix B, “Data Splitting”: https://arxiv.org/html/2412.17245v2
- *Balanced Co-Clustering of Users and Items for Embedding Table Compression in Recommender Systems*, arXiv:2604.18351v1, §5.1: https://arxiv.org/html/2604.18351v1

[Certain] GraphHash uses random 80/10/10 splits for its stated retrieval datasets and MovieLens CTR datasets; Frappe uses an externally supplied split. BACO reports random 80/10/10 splits for its four retrieval datasets.

**Citation-ready paraphrase:** “The reported retrieval evaluations of GraphHash and BACO use random interaction splits.”

[Likely] **Motivated experiment:** rolling chronological evaluation, with clustering, contexts, and parameter updates restricted to available history; controlled context shifts should be a separate axis.

**Do not claim:** random splits are inherently leakage, that the reported methods fail temporally, or that temporal recommendation has not been studied elsewhere.

## 5. One shared user representation has an expressiveness limitation

**Source:** BACO, arXiv:2604.18351v1, §4.5, “Generating Secondary Clusters for Users.”

[Certain] The authors identify a limitation of representing each user by one codebook embedding when users have multiple/evolving interests. Their SCU mechanism assigns a secondary cluster while adjusting the storage budget.

**Citation-ready paraphrase:** “BACO motivates secondary user clusters through the representational limitations of a single codebook embedding.”

[Likely] **Motivated experiment:** determine whether a context-sensitive allocation offers value beyond BACO's secondary clusters at the same total bytes and maintenance budget.

**Do not claim:** multiple interests or assigning users to multiple compressed representations is a new gap. BACO already addresses it.

## 6. Hard sharing can erase distinguishability

**Source:** Tsang and Ahle, *Clustering the Sketch: Dynamic Compression for Embedding Tables*, NeurIPS 2023, §1, printed p.3, discussion beginning “Why is this hard?”.

**Primary PDF:** https://thomasahle.com/papers/cqr.pdf

**Official record:** https://papers.neurips.cc/paper_files/paper/2023/hash/e468a76212a58c1af94a3d235151944a-Abstract-Conference.html

[Certain] The introduction describes the inability to distinguish IDs after a hard shared-vector decision. CCE addresses this problem through repeated clustering and compositional sketching.

**Citation-ready paraphrase:** “Tsang and Ahle explicitly discuss distinguishability restrictions introduced by hard embedding sharing.”

[Likely] **Motivated experiment:** establish what remains unresolved after allowing CCE-style dynamic updates and simpler rebuilding policies.

**Do not claim:** identical shared vectors, dynamic assignment updates, or the hard-sharing mechanism is our discovery.

## 7. Population-level behavioral fidelity is a different objective from personalization

**Source:** *Unsupervised Behavioral Compression: Learning Low-Dimensional Policy Manifolds through State-Occupancy Matching*, arXiv:2603.27044v1, §4, Eq.(6): https://arxiv.org/html/2603.27044v1

[Certain] OPC explicitly shifts from individual policy matching toward matching the population's mixture occupancy. Its experiments concern continuous-control policy manifolds.

**Citation-ready paraphrase:** “OPC formulates behavioral compression through population-level mixture-occupancy matching.”

[Certain] Preserving a population mixture alone does not determine which behavior belongs to each named individual: swapping two equally weighted components leaves the mixture unchanged. This is a mathematical scope distinction, not a reported failure of OPC at its intended task.

[Likely] **Motivated experiment:** report per-user/context prediction and aggregate behavioral fidelity separately; include a controlled identity-swap diagnostic.

**Do not claim:** OPC is intended to preserve named Web-user identities, or that this mixture ambiguity is a novel theorem.

## 8. Collision-related update instability is reported, but its cause is hypothesized

**Source:** *Multi-Probe Zero Collision Hash (MPZCH): Mitigating Embedding Collisions and Enhancing Model Freshness in Large-Scale Recommenders*, arXiv:2602.17050v4, §4.2.2: https://arxiv.org/html/2602.17050v4

[Certain] The authors report volatile baseline video-embedding trajectories and smoother trajectories with MPZCH. They hypothesize that shared-slot gradient interference contributes to the volatility.

**Citation-ready paraphrase:** “MPZCH reports embedding-update instability consistent with collision interference and more stable trajectories under its collision-management scheme.”

[Likely] **Motivated experiment:** test update interference under controlled sharing assignments and matched optimizer/migration settings.

**Do not claim:** the paper causally identifies our user-policy/context-shift mechanism, or that MPZCH concerns only new-ID insertion.

## Strongest defensible research rationale

[Likely] Combine calibration coverage dependence (1), historical state weighting (3), and the temporal evaluation boundary (4) as the main motivation. Treat sparse estimation (2) as a required practical constraint. Use (5), (6), and existing dynamic methods to establish strong baselines; do not sell their already-addressed problems as new discoveries. The individual-versus-population distinction (7) is useful analytical support, not a reason to compare incompatible tasks unfairly.

[Likely] A defensible proposal sentence is: “We investigate whether historical behavior-based sharing decisions preserve user-conditioned predictions when context frequencies change, and whether a coverage- and uncertainty-aware allocation improves the quality–memory–maintenance trade-off.”

[Certain] This evidence map establishes related limitations and motivates a hypothesis. It does not establish that the exact proposed phenomenon occurs, that our correction works, or that no previous paper has studied the same target. Newer arXiv sources are cited in the inspected versions; this document does not promote them to peer-reviewed status without a verified publication record. AI-assisted primary-source verification was used.
