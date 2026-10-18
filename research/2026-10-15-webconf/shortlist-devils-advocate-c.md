# Adversarial review of the three shortlisted directions

[Likely] **Rank 1 > 2 > 3 for a new three-day research project.** This is a ranking of conditional scientific opportunity, not predicted acceptance. Direction 1 offers the cleanest controlled intervention; direction 2 has a useful deployment constraint but a strong obvious-baseline threat; direction 3 is a rigorous audit topic whose general phenomenon is already established. Direction 3 can become the best fallback if cached epoch predictions exist and the user wants a reproducibility study rather than a new mechanism.

[Certain] Basis: the root's supplied protocols and prior-art screening, partition B's explicit source receipts for test-based selection, and this agent's first-principles analysis. I did not independently reread Thematic-LM, GraphHash, MPZCH, DeliberationBank, LLooM, or the clinical thematic-analysis collision papers for this memo. Their inclusion as required comparators comes from the root's investigation. No claims of novel literature absence are made.

## 1. Thematic analysis: path dependence and bounded residual-evidence replay

**Keep provisionally; the fatal threat is an ungrounded outcome measure.** [Certain] Holding the final post multiset and leaf extractions constant can identify the effect of downstream processing order, conditional on those leaves. It cannot recover themes already omitted at extraction. [Likely] If the same LLM invents the reference themes, detects their survival, and judges replay success, the study can merely demonstrate consistency with its own preferred taxonomy. Human theme existence and human granularity choices must enter independently.

[Likely] The other major rejection risk is treating path dependence itself as surprising. Adaptive clustering/codebook construction is intrinsically order-dependent unless constrained otherwise. The contribution must show an important, repeatable, semantically harmful effect under realistic web corpora, isolate where it occurs, and improve the precision/coverage/cost frontier. A toy stream deliberately arranged to bury one rare label cannot establish that practical claim.

**Necessary controls:**

- [Certain] Define an independently adjudicated theme-to-post incidence relation, plus permissible theme merges. Measure support retention, theme precision, spurious-theme count, and supported minority-theme recall; do not equate preservation of a label string with semantic survival.
- [Certain] Condition the primary survival denominator on themes whose supporting evidence was present in frozen leaves. Report full-corpus recall separately so this conditioning does not hide upstream failures.
- [Certain] Use several within-order random seeds to establish stochastic baseline variation. Compare between-order variation with within-order variation on the same corpus and model; streams, seeds, and intermediate states are not independent datasets.
- [Certain] Match residual replay to random replay, recency replay, and a simple reservoir of unassigned evidence at the same total token/call budget. Equal nominal tokens are insufficient if one method repeats more theme-bearing evidence; quantify theme exposure too.
- [Certain] Keep a fixed full-data reference, a deterministic merge/coverage baseline, and applicable LLooM/DeliberationBank alternatives. If a cheap deterministic evidence index solves the problem, an agentic replay controller is unnecessary.

**Smallest useful pilot:** [Likely] One real corpus with independently labeled coarse themes and minority subthemes; about100-200 posts; frozen leaves; four-six orderings including natural chronology, reverse, and dispersion controls; three seeds; baseline plus residual and random replay. Inspect stagewise loss on a blinded subset of states. This is a mechanism pilot, not a complete publication evaluation. A second corpus and second backbone are necessary before generality claims.

**Kill criterion:** [Likely] Stop this trajectory if order-induced semantic loss is comparable to ordinary within-order variation, depends on one arbitrary taxonomy, or vanishes under equal-exposure random/reservoir replay. A positive result should survive label-granularity sensitivity and show a useful bounded-budget intervention. **Three-day feasibility: medium-high computationally; annotation readiness is the gating resource.**

## 2. Persistent-ID preference drift under graph-based embedding sharing

**Keep as backup; the fatal threat is a dynamic-clustering baseline doing the same thing.** [Certain] A static sharing map can impose representational restrictions: IDs mapped to the same parameter vector cannot receive independent vector updates. If their target distributions diverge, some distinctions become unrepresentable without extra capacity or changed assignments. This supplies a mechanism, but not a theorem that any particular split-merge rule improves held-out recommendation.

[Likely] Restricting all IDs to existing IDs separates the question from insertion/eviction, but it does not automatically separate it from ordinary online embedding retraining, dynamic graph clustering, drift detection, or adaptive hash reassignment. MPZCH's stale/new-ID handling remains a required comparison where operationally relevant. Claiming novelty solely from a different trigger for reclustering will be weak.

**Necessary controls:**

- [Certain] Build clusters only from interactions available before each prediction window. Future degree, full-data communities, and final-window identities cannot influence the online mapping.
- [Certain] Charge the complete memory budget: bucket parameters, per-ID mappings, graph/reservoir state, optimizer state, and migration metadata. Charge clustering and embedding migration time; offline preprocessing is not free maintenance.
- [Certain] Compare static GraphHash, random/CCE/BACO, periodic full or budgeted reclustering, and an elementary drift-triggered reclustering rule. A bounded split requires a compensating merge if bucket count is fixed.
- [Certain] Warm-start/migration behavior needs a matched reset/remap control. Otherwise gains can come from optimization resets or changed update frequency rather than the claimed reduction in harmful sharing.
- [Certain] Use both real chronological windows and controlled rewiring. Rewiring demonstrates a conditional mechanism; it is not evidence that a real user's preference changed. In real data, distinguish affinity change from popularity and exposure shifts.

**Smallest useful pilot:** [Likely] One timestamped dataset, a fixed existing-ID cohort, one pre-drift and two later windows, three seeds, static/random/periodic/dynamic policies at one tight memory budget. Log sharing collisions, prediction degradation, and maintenance cost. Introduce a controlled community change separately, with unchanged degrees when possible. Do not start with a giant backbone; a simple matrix-factorization model makes the collision mechanism easier to isolate.

**Kill criterion:** [Likely] Stop if periodic reclustering achieves the same accuracy at equal maintenance cost, or if the proposed method needs hidden O(number-of-interactions) state that destroys its memory advantage. **Three-day feasibility: medium**, conditional on ready data/code; strongest main-track version needs a clear accuracy-memory-maintenance frontier and more than one drift regime.

## 3. Test-selected-checkpoint inflation in misinformation detection

**Useful audit; weak default choice for a new algorithm paper.** [Certain] Partition B's paper receipts identify best-test-epoch selection in MIMoE-FND (Appendix A.2) and meme-to-video transfer (§4.1.3). This establishes the protocol, not the size of the resulting optimism. Multiple epochs are correlated; a large epoch count does not imply a large inflation.

**Fatal estimation flaw:** [Certain] The difference between original-test-oracle performance and validation-selected performance on that same test set is a test-selection advantage. It is not, by itself, an unbiased estimate of population generalization inflation. Underlying epoch qualities can differ; the selected test winner needs an independent evaluation sample to separate signal from reused test noise.

[Certain] A clean audit caches identical per-example predictions for every epoch, freezes the selection rules, and repeatedly partitions a reporting set into a selection part and untouched audit part, with clustering by event/source where appropriate. Select checkpoints using test-oracle, validation, or fixed-stop criteria; then report on the independent audit part. Keep the model, training, predictions, and data access identical. Domain shift is a separate axis and should not be mixed into the primary selection-bias estimate.

[Likely] The novelty risk is severe: test-set selection optimism is a standard methodological issue. A meaningful empirical contribution needs broader consequence: method ranking reversals, reproducibility of claimed margins, sensitivity to effective checkpoint diversity, and defensible correction without retraining. Two papers and a few repeated runs may support an incisive audit but not a broad claim about web misinformation research.

**Smallest useful pilot:** [Likely] One MIMoE dataset with full epochwise validation/test predictions, three seeds, preregistered fixed-stop and validation rules, and repeated event/source-aware selection/audit partitions. If logs are unavailable, reproduce only the smallest faithful configuration first; launching the expensive video model before that gate is unnecessary.

**Kill/reframe criterion:** [Likely] If optimism is tiny relative to claimed margins and rankings are stable, report that negative result rather than escalating an accusation. If substantial, expand the audit corpus and frame the paper around protocol-sensitive conclusions. **Three-day feasibility: high with cached predictions, low-medium when retraining is required.** Do not imply wrongdoing from a documented reporting choice.

## Recommended decision gate

[Likely] Start with direction1 only if a human theme-incidence reference and accessible baseline implementation are already available. The first pilot should test whether order creates an effect beyond stochasticity and whether residual replay beats equal-exposure random/reservoir replay. Do not spend the whole budget designing a controller before those conditions hold. Direction2 is the more credible fallback when annotation is unavailable but timestamped interaction data and compression code are ready. Direction3 is the most reliable short audit if epoch predictions exist, but should not be advertised as a novel discovery of test leakage.
