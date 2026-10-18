# Web Conference research direction: evidence audit and feasible designs

Research date: 15 October 2026. User constraint: three Kaggle accounts, 23 NVIDIA keys, and at most three days of main experiment runtime across the accounts. Coding, annotation, source access, and writing are separate costs.

**[Likely] Recommendation:** investigate whether adaptive analysis of the same Web contributions changes which specific viewpoints survive because of update history and rare-viewpoint dispersion. Treat this as a falsifiable mechanism study, with a small pilot before committing to an algorithm. A temporal graph-compression study is the main alternative; a checkpoint-selection audit is the conservative measurement alternative.

**[Certain] Execution update:** a subsequent user-approved local support audit and count-baseline probe has now been executed. See the [executed KuaiRand-Pure audit report](D:/web_2/experiments/2026-10-15-kuairand-support/AUDIT_REPORT.md). It establishes limited component evidence, exposes label/timestamp/duplicate and context-coverage issues, and does not validate a learned compression method or a pure context-shift claim.

**[Certain] Subsequent count-level E0 decision:** the supplied equal-budget count-sharing/context-reweighting probe has now been executed with 27 grouped predictors and three references. The fixed primary gate is **NO-GO**: at B=256, the frequency-weighted sharing penalty decreases as the interface mix flattens, and a tilted-objective growth advantage is not established. A narrower supported-user B=64 signal does not demonstrate that the tilted remedy closes the gap. See the [executed E0 report and plots](D:/web_2/experiments/2026-10-15-count-e0/E0_REPORT.md). This does not justify escalation to a neural campaign or establish a general impossibility result.

**[Certain] Final bounded mechanism-check decision (16 October): CLOSE THIS COMPRESSION PROPOSAL AND PIVOT.** A training-half-defined, activity/marginal-matched pairing diagnostic with three seeds fails its fixed validation gate: half-B contextual differences are not established as repeatable, and high/low sharing effects reverse sign across pairings. One positive adjusted association does not repeat. The previous optimizer was extended beyond 12 rounds under three initializations; validation improvements are not uniform. See the [mechanism-check report](D:/web_2/experiments/2026-10-16-mechanism-check/MECHANISM_REPORT.md). Preserve the exploratory records; do not expand this grid or claim general impossibility. The earlier agentic Web-analysis direction remains a proposed pivot requiring its own novelty/baseline screen and pilot.

**[Certain] Separate exposure-evaluation hypothesis, subsequently executed (16 October):** a user-approved MF pilot now compares full, random-shared, frequency-shared and nearly parameter-matched rank2 user interaction factors. This is a different hypothesis from the closed rare-context/tilted proposal. The raw policy-gap difference is small and positive; after intercept calibration it is near zero with intervals including zero, and no sharing-specific distinction from the rank2 control is established. Weak full-reference interaction gains limit the test. See the [executed MF report](D:/web_2/experiments/2026-10-16-exposure-mf/EXPOSURE_MF_REPORT.md) and [mathematical specification](D:/web_2/experiments/2026-10-16-exposure-mf/MATHEMATICAL_FOUNDATION.md). No broad campaign, independent confirmation, or claim that random exposure measures intrinsic preferences is justified by this pilot.

**[Certain] Original literature-stage status:** this dossier contains literature analysis and proposed experiments. At that stage no new Kaggle runs, model training, NVIDIA generations, or empirical validation of the proposed ideas had been performed. Primary-source observations and reviewer inferences are separated. A verified paper title does not imply its entire text was accessible.

## 1. What this review establishes

**[Certain]** The supplied attachment contains 61 entries: three award-related papers, ten oral selections, eight industry papers, ten Web4Good papers, ten workshop papers, ten research posters, and ten short papers. The Test of Time entry is an older paper. These categories describe the supplied list; presentation and award distinctions are not treated as evidence of methodological superiority.

The companion [61-paper audit](D:/web_2/research/2026-10-15-webconf/PAPER_BY_PAPER_AUDIT.md) records a source, reading depth, contribution, evaluation boundary, possible extension, and feasibility for every entry. **[Certain] Coverage is 50 F, 4 P, and 7 A.** `F` means methods/evaluation sections were inspected in accessible full text; `P` means substantive but incomplete sections or author artifacts; `A` means abstract or primary bibliographic record only. `F` is not a claim that every mathematical proof and appendix has been independently reproduced. The P cases are A12, A18, B16 and C07. The A cases are B11, B13, B15, B17, B19, C11 and C20; limitations of those papers cannot be fully audited from the available material.

**[Certain]** This is a targeted corpus review and novelty search, not a PRISMA systematic review or an exhaustive census of 2026 research. Searches used exact supplied titles, publisher/author/arXiv records, and mechanism-specific forward searches. Sources published after 15 October 2026 were not used to establish prior art. Later preprints are identified as preprints, rather than silently promoted to established empirical facts.

**[Likely]** There is no scientifically meaningful limitation shared by all 61 papers: next-basket recommendation, clinical extraction, inverse reinforcement learning, image safety, and UI focusability have different targets. The useful commonalities occur within clusters of papers. Missing evaluation is an unestablished boundary, not evidence that a method fails.

## 2. Cross-paper gaps worth considering

| Gap | Grounded examples | What is still a question | Feasibility and caution |
|---|---|---|---|
| **Selection protocol can contaminate reported test performance** | **[Certain]** MIMoE-FND Appendix A.2 reports the best test accuracy over 50 epochs. Cross-Modal Transfer from Memes to Videos §4.1.3 selects its best epoch on the video test set. | **[Likely]** The magnitude of optimism and its effect on relative model rankings remain to be measured with independent evaluation. | Small if checkpoint predictions exist; expensive if exact VLM retraining is necessary. Correct selection alone is sound practice, not a new algorithm. |
| **A proxy can measure something different from the intended property** | **[Certain]** EARec evaluates aspects of explainability using generated-title overlap and correlation with ranking. Thematic-LM's dependability includes lexical similarity across reruns. Simulator success is measured within the simulated interaction. | **[Likely]** Evidence faithfulness, preservation of specific viewpoints, and agreement with real user decisions require separate validation. | Strong fit for short, controlled inference experiments; human reference work is a separate bottleneck. Do not call all proxies invalid merely because they are incomplete. |
| **Adaptive pipelines can change the information retained before final generation** | **[Certain]** Thematic-LM uses an adaptive codebook; HtmlRAG prunes HTML; Graphusion fuses local and global graph information. | **[Likely]** Order, dispersion, and intermediate compression may alter retention of semantically specific evidence even when the complete input corpus is unchanged. | Promising mechanism question. Provenance tracking and generic minority omission already have prior work. |
| **Static performance does not settle behavior under changing Web populations** | **[Certain]** GraphHash tests sparse graphs and frequency groups, and explicitly discusses future adaptation. A static tail comparison is already present. | **[Likely]** The effect of persistent-ID preference drift on learned sharing assignments needs a distinct temporal design, rather than another head-versus-tail table. | Small recommenders can fit the budget; temporal data availability and equal update cost must be verified. |
| **Multiple models need not contribute independent information** | **[Certain]** JudgeBlender evaluates ensembles; 2026 papers directly study correlated judge errors and effective panel size. | **[Likely]** A new study needs a sharper intervention than adding model families or a weighted vote. | Cheap inference, but high novelty collision risk. Deprioritize as a standalone proposal. |
| **Grounded explanations and graphs can still discard distinctions** | **[Certain]** Graphusion's fusion prompt restricts an entity pair to one relation. Generated explanations and relation extraction are different tasks. | **[Likely]** Simultaneously valid relations, exceptions, negation, and temporal scope may require a richer representation and an explicit fidelity test. | A focused component audit is feasible. Generic provenance/conflict-aware GraphRAG is already occupied. |
| **Industrial and safety claims have data or outcome constraints** | **[Certain]** Several industry papers rely on proprietary data or online evaluation; the ledger records access limits. Sherlock reports substantial A100 training time. | **[Likely]** A public proxy experiment cannot establish production revenue, real learning benefit, clinical efficacy, or broad safety. | Exclude unavailable production claims from the three-day target. |

Primary anchors for the strongest concrete findings: [MIMoE-FND Appendix A.2](https://arxiv.org/html/2501.12431), [meme-to-video transfer §4.1.3](https://arxiv.org/html/2501.15438), [GraphHash](https://arxiv.org/html/2412.17245), and the per-paper sources in the audit.

## 3. Ideas rejected or substantially narrowed by newer work

**[Certain]** The following are occupied directions, not fresh contributions merely because they combine two papers from the attachment:

| Initial direction | Primary-source collision | Consequence |
|---|---|---|
| Duplicate or paraphrased evidence changes RAG | [How retriever redundancy and diversity impact RAG effectiveness](https://arxiv.org/abs/2608.13956); [CUE-R](https://arxiv.org/abs/2604.05467) | Do not propose a duplicate-versus-diverse comparison as the main novelty. |
| More diverse LLM judges, dependence-aware voting | [Nine Judges, Two Effective Votes](https://arxiv.org/abs/2605.29800); [Agreement Overstates Evidence](https://arxiv.org/abs/2609.22512) | Cross-family correlation and effective panel size are already directly studied. |
| Calibrating LLM relevance judgments with a little human supervision | [PRECISE](https://arxiv.org/abs/2601.18777); [ARES](https://arxiv.org/abs/2311.09476) | A generic prediction-powered judge evaluator is insufficiently distinct. |
| Tournament reranking with fewer calls or handling cycles | [BLITZRANK](https://arxiv.org/abs/2602.05448); [Contextual Relevance and Adaptive Sampling](https://arxiv.org/abs/2511.01208) | TourRank plus an uncertainty or tournament controller needs a stronger distinction. |
| Minority-aware thematic summaries | [DeliberationBank](https://arxiv.org/html/2510.05154v3) | Minority underrepresentation is already studied. Its v3 Appendix A finds no consistent positional trend in its tested setting; do not cite it as universal evidence of order failure. |
| Add codebook refinement or provenance to thematic agents | [Automated Thematic Analysis for Clinical Qualitative Data](https://arxiv.org/html/2603.08989v1); [LLooM](https://doi.org/10.1145/3613904.3642830) | These are required baselines. LLooM already revisits uncovered and overly generic examples. |
| Compare hierarchical versus direct thematic coding | [A Multi-Modular Human-AI Workflow](https://doi.org/10.3233/SHTI260838) | Hierarchical consolidation and direct grouping are already compared. Mere restructuring is insufficient. |
| Balanced graph embedding compression | [BACO](https://arxiv.org/html/2604.18351v1) | Balance, GraphHash's resolution limit, and multiple user interests are occupied. |
| Freshness-aware embedding hashing | [MPZCH](https://arxiv.org/abs/2602.17050) | Stale-ID eviction and new-ID freshness are already addressed. |
| Agentic, open-set government-language bias detection | [The 10th Juror](https://arxiv.org/abs/2610.11136) | A direct October 2026 collision; generic legal retrieval plus a bias-detection panel is not a fresh idea. |

**[Likely]** None of this proves that a narrow extension is impossible. It prevents a false novelty claim. The shortlist below consists of conditional research directions, not certified discoveries.

## 4. Recommended direction: same posts, different conclusions

**Working title:** *Same Posts, Different Conclusions: Order-Induced Lock-In in Agentic Web Analysis*.

**[Likely] Research question:** when an agentic system incrementally builds a codebook from a fixed collection of Web contributions, can the arrival order and dispersion of a low-prevalence viewpoint change whether its meaning survives? If so, does the effect arise during extraction, codebook update, evidence selection, or final synthesis?

The Web problem is the reliability of social listening, consultation analysis, and community summarization when batching and ingestion order are operational choices. The paper must establish a consequence for these Web systems, rather than merely running an LLM on Reddit text.

### 4.1 What would be new enough to investigate

**[Certain]** Generic minority omission, broad codebook quality, lineage tracking, and hierarchical-versus-direct comparison already have close prior work. Thematic-LM is an adaptive codebook pipeline; it should not be misdescribed as a hierarchical merge-tree system.

**[Likely]** The narrower candidate contribution is a controlled, stagewise account of path dependence: hold the corpus and specific supported viewpoints fixed; vary dispersion and update history; quantify semantic loss beyond the variation observed when rerunning the same history. Determine whether an inexpensive intervention reduces that dependence without inventing rare themes or consuming a larger effective budget. The search did not establish that this exact contribution is already solved, but it also did not certify priority.

### 4.2 Formal target

Let `E(x)` be a leaf record extracted from post `x`. Let the adaptive codebook update be

`C[t+1] = U(C[t], E(x[pi(t)]))`.

**[Certain]** If the update operator discards information, two sequences containing the same records can produce different final codebooks. This is a possibility under the model definition, not a measured assertion about Thematic-LM. For the simple threshold model in which each batch retains a theme only when it has at least `tau` supporting records, a theme with total support `m` disappears at the batch stage exactly when every batch has support below `tau`. The same `m` can therefore survive when concentrated and disappear when dispersed. Whether an LLM pipeline follows this mechanism must be tested.

Define a prespecified semantic coverage score `Q` against human-verified specific viewpoints. The main estimand is the paired difference in `Q` between concentrated and dispersed placements of the same support records, averaged across corpora. Stream order, partition, final display order, and decoding randomness are separate factors. Do not combine them into one undiagnosed perturbation.

### 4.3 Data and reference construction

**[Certain]** [DeliberationBank](https://arxiv.org/abs/2510.05154) contains human-authored opinions and summary evaluations. [HumAID](https://crisisnlp.qcri.org/humaid_dataset) supplies event-based disaster text and humanitarian categories. [Dreaddit](https://arxiv.org/abs/1911.00133) supplies Reddit stress data. These labels are not automatically gold thematic codebooks.

**[Likely]** Use one human-opinion corpus family and one existing Web-post corpus family, subject to obtaining the actual released text and license. Begin with 8–12 corpus units of 120–200 contributions each, not millions of posts. Preserve each corpus's complete multiset in the main paired intervention. Vary rare-record prevalence only in a separate controlled experiment.

**[Certain]** A small independently annotated set of specific viewpoints and supporting spans is necessary to claim semantic preservation. Existing stress, emotion, or broad topic labels cannot substitute for those annotations. Two coders should work independently on a prespecified held-out audit subset, record disagreement, and adjudicate without seeing which system produced an output. Corpus-level generalization is limited when the number of distinct topics or events is small.

### 4.4 Baselines and intervention

**[Likely]** Baselines should include an accurately documented Thematic-LM implementation or explicitly labeled reimplementation; LLooM or an equivalent global concept-induction pipeline; a recent codebook-refinement pipeline; simple shuffle/reverse-order controls; and ordinary random, recency, and reservoir replay with the same call budget as the proposed intervention. A direct long-context baseline is useful when the full corpus fits. The original Thematic-LM was only partly accessible in this audit, so obtaining the intended manuscript and usable baseline artifacts is a prerequisite to an exact replication claim.

The proposed intervention keeps immutable evidence records and a bounded reserve of unresolved or overly broad codes. A later pass revisits specific lost distinctions before themes are finalized. This mechanism is not novel by itself: LLooM already revisits uncovered and generic examples. A contribution requires showing that a particular history-dependent failure survives those baselines, then identifying a distinct correction. Report all evidence storage, rescoring, and extra tokens.

### 4.5 Main experiments

1. **End-to-end history intervention.** Same complete posts, different ingestion orders and rare-support layouts. Include repeated unchanged-order runs.
2. **Frozen-leaf component intervention.** Cache extraction outputs, vary only codebook update history. This isolates a component; it is not equivalent to end-to-end reproduction. Condition the component survival denominator on themes whose support is actually present in the leaves, and report full-corpus recall separately to expose extraction losses.
3. **Stagewise survival.** Identify where a supported distinction first becomes unrepresented. A retained ID does not prove that its semantic content survives.
4. **Final-display control.** Fix the learned codebook, change only presentation order to distinguish synthesis attention effects from codebook history.
5. **Semantic specificity and false positives.** Measure coverage of minority and common viewpoints, incorrect merges, unsupported themes, polarity reversal, and abstraction that erases a meaningful distinction.
6. **Equal-budget repair.** Compare the proposed replay mechanism against random, recency, and reservoir replay and LLooM-style residual coverage at the same total input/output tokens and measured latency. Log how often theme-bearing evidence is actually exposed; the same total token count can hide unequal exposure to the decisive records.
7. **Transfer.** Freeze settings on development corpora and evaluate on held-out topics/events with a second model family. Repetitions of one corpus do not create independent new datasets.

**[Likely]** Use paired corpus-level effects with uncertainty intervals; nested resampling may account for execution repeats within a corpus. Freeze one primary outcome and practically meaningful effect threshold before the reserved evaluation. Report the full accuracy/coverage–budget curve. A lexical matching improvement alone cannot establish semantic recovery.

### 4.6 Pilot and stop rule

**[Likely]** First run a development-only pilot on 2–3 corpus units with a frozen reference audit. Continue only if the history/dispersion effect exceeds ordinary repeat variation and the effect reflects lost meaning rather than changed labels. Then perform a held-out confirmation before claiming a result.

Stop or reframe if the effect is absent, if simple random/reservoir replay removes it at equal budget and exposure, if LLooM already solves it, if the apparent improvement comes from added tokens or oracle knowledge of minority labels, or if human judges cannot reliably distinguish true semantic loss. A null pilot is not a reason to keep changing prompts until a positive result appears.

**[Guessing] Runtime envelope:** a bounded study may require roughly 10,000–25,000 short API requests after caching, plus modest local embedding/scoring work. This is a planning envelope, not a benchmark measurement. Annotation and implementation are additional work. Avoid per-post/per-theme exhaustive judging, which can expand cost by an order of magnitude.

## 5. Alternative: persistent preferences and compressed recommenders

**Working title:** *When Yesterday's Communities Share Tomorrow's Embeddings: A Temporal Audit of Graph-Based Recommendation Compression*.

**Follow-up novelty audit, 15 October 2026:** **[Certain]** CCE, *Clustering the Sketch: Dynamic Compression for Embedding Tables* (NeurIPS 2023), explicitly discusses the loss of distinguishability after hard sharing and develops dynamic clustering during training. Monolith (2022), especially §3.2.1, already examines collision effects as distributions change over time. MPZCH §4.2.2 also discusses collision-induced instability of embedding updates. Thus hard-sharing restrictions, dynamic reassignment, and collision/temporal-change interaction are established prior art, not available broad novelty claims. [CCE](https://papers.neurips.cc/paper_files/paper/2023/hash/e468a76212a58c1af94a3d235151944a-Abstract-Conference.html), [Monolith](https://arxiv.org/html/2209.07663v2), [MPZCH](https://arxiv.org/html/2602.17050v4).

**[Likely] Revised assessment:** broad problem/method novelty is low. A narrowly controlled study of persistent-ID sharing-map staleness, with total-memory and maintenance-cost comparisons against CCE, BACO and periodic/drift-triggered reclustering, remains a conditional opportunity. This search has not established that precise combination is absent from the literature. Do not commit a full main-track project before an inexpensive discriminating pilot. The identity `s(u,i)=s(u,j)` for identical item features is an elementary mechanism explanation, not an original theoretical contribution.

**[Likely] Research question:** do graph-conditioned sharing assignments become harmful when preferences change among persistent users and items, and can a bounded update repair the damage at fixed memory and update cost?

**[Certain]** GraphHash already tests frequency groups and sparsity; BACO addresses balance and richer interests; MPZCH addresses collisions and freshness. The proposed distinction concerns changed relationships among existing IDs. New-ID insertion and stale-ID eviction must be separate controls, not the main claimed novelty.

**[Certain] Conditional analytical fact:** in an item-bias-free dot-product recommender where items `i` and `j` share exactly the same embedding and have no additional item-specific features, `s(u,i) = s(u,j)` for every user. No optimizer can represent different personalized scores for those two items without changing the representation. This statement does not imply an overall lower bound on the risk of an arbitrary recommender architecture.

**[Likely]** Use real chronological interaction windows plus a controlled graph experiment that changes affinities while keeping IDs, degree distribution, interaction count, and noise as comparable as possible. The controlled experiment establishes behavior under the imposed intervention, not naturally occurring causal preference changes. Freeze the mapping using training history only; exclude future interactions from clustering and all preprocessing.

Compare GraphHash, random sharing, CCE or another learned dynamic sharing baseline, BACO, a full-table accuracy reference, and a fixed-memory update candidate. Periodic and simple drift-triggered reclustering are mandatory obvious controls, even if a first pilot uses fewer methods. A split must have a compensating merge at fixed bucket count. Match warm-start, reset, and migration behavior, or optimizer resets can explain apparent improvement. Measure Recall/NDCG, changed-versus-stable user groups, mapping changes, optimizer-state effects, update time, and total stored bytes. Include assignment arrays, graph state, duplicate tables during rebuilding, and temporary memory. Split/merge/replay mechanisms are established tools; the new contribution would need to be a specific finding and correction beyond these baselines.

**[Guessing] Runtime envelope:** two small timestamped datasets, approximately five compressed conditions, two memory budgets and three seeds give 60 training jobs. At 20–40 minutes per job this would require 20–40 accelerator-hours before preprocessing and overhead. Those job times are assumptions to verify; cap a pilot first and reduce the grid if the measured total does not fit actual quotas. A full-table reference can be an additional cost. NVIDIA keys are not needed for this direction.

**[Likely] Stop rule:** do not continue if the effect is entirely ordinary staleness also seen in the full-table reference, if a standard rebuild or CCE/BACO removes it at the same cost, or if new-ID churn explains the result. The static tail-compression hypothesis is not an empty gap.

## 6. Alternative: quantify what test-selected checkpoints add

**Working title:** *How Much Robustness Did We Select? A Checkpoint-Selection Audit of Multimodal Web Classification*.

**[Certain]** There is a concrete source-level basis: MIMoE-FND reports its best test accuracy over training, and the meme-to-video paper selects its best video-test epoch. This establishes a protocol concern; it does not establish how large the inflation is or that every reported comparison would reverse.

**[Likely]** The research target is a paired audit: hold training trajectories and evaluation examples fixed, then compare fixed stopping, validation-only selection, and test-oracle reporting. Analyze how the number and correlation of candidate checkpoints affect the apparent uplift, and whether small reported model gains survive legitimate selection.

**[Certain]** For unbiased checkpoint estimates, the expected maximum observed score is at least the maximum expected score. This follows from convexity of the maximum. It does not supply an inflation number: adjacent checkpoints can have strongly correlated errors, and the true checkpoint qualities can differ.

**[Likely]** Save all checkpoint predictions during each training run. Construct a selection split and a separate outer evaluation split at the relevant group/event level. A checkpoint selected on a pseudo-test split should be assessed on the untouched outer split; the difference estimates selection optimism under that split design. Merely subtracting validation-selected from test-selected accuracy on the same original test set does not isolate all sources of bias.

Begin with one reproducible public-data architecture and simple unimodal/fusion controls, using several seeds. The video paper motivates the question but need not be fully retrained for an initial study. Keep source overlap, modality availability, and domain shift as separate analyses. An inexpensive frozen-feature diagnostic must be labeled as a component study, rather than a reproduction of full VLM fine-tuning.

**[Guessing] Runtime envelope:** postprocessing checkpoint predictions is cheap; recreating the predictions can dominate. The original MIMoE-FND reports approximately five hours for 50 epochs on an A40. This is not a measured Kaggle T4/P100 runtime. Exact VLM transfer reproduction may exceed the three-day budget. Continue only after data/checkpoint access and one measured training pilot fit the total schedule.

**[Likely] Publication assessment:** this is the most directly grounded audit and may yield a valuable short measurement paper. The statistical principle is established; a main-track contribution requires meaningful and reproducible consequences across more than one carefully chosen setting, not merely identifying an incorrect stopping rule.

## 7. Budget and execution design

**[Certain]** Key count is not measured aggregate throughput or model diversity. NVIDIA documents API credentials as account-linked, and hosted model listing does not establish successful generation for every credential/model combination. This dossier did not test the user's keys. [NVIDIA deployment FAQ](https://docs.api.nvidia.com/nim/docs/deployment), [API catalog quickstart](https://docs.api.nvidia.com/nim/docs/api-quickstart).

**[Certain]** Kaggle documents per-session execution limits and GPU availability/weekly quotas. Three accounts should not be assumed to provide three uninterrupted 72-hour GPU jobs. [Notebook documentation](https://www.kaggle.com/docs/notebooks), [efficient GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage).

**[Likely]** Choose one direction. Run the pilot before spending the remaining budget. Divide independent model/data/seed shards across the three selected account aliases, with account-specific metadata, immutable splits, cached outputs, and resumable jobs that fit the actual session limit. The shared GPU-hour budget is the sum of the live available quotas, not `3 × 72` by default.

For an API study, a rough planning lower bound is

`hours >= requests / (60 × measured aggregate completed requests per minute)`.

**[Certain]** At 20 aggregate completed requests per minute, 25,000 requests alone take approximately 20.8 hours; at five per minute, approximately 83.3 hours. These are arithmetic scenarios, not measured service rates. Latency, tokens, backoff, queueing, retries and analysis add overhead. Use a bounded throughput pilot to decide whether the proposed grid fits.

**[Likely]** Account A can cover the first model family or data shard; Account B the second; Account C controls and a small robustness replication. Do not triple the hypothesis count merely because three accounts are available. A bounded design with paired interventions is more informative than many poorly controlled agent variants.

## 8. Venue and timing

**[Certain]** As checked on 15 October 2026, WWW 2027 lists research abstracts due 18 October and papers due 25 October; short-paper abstracts are due 9 November and papers 16 November, at 23:59 AoE. Web4Good will not run as a separate 2027 track. [Official dates](https://www2027.thewebconf.org/important-dates/), [conference news](https://www2027.thewebconf.org/).

**[Certain]** The research call requires a specific Web scientific challenge; merely using a Web dataset or API is insufficient. [Research scope](https://www2027.thewebconf.org/research-track-papers/).

**[Likely]** If starting from no implementation, the November short-paper route is more credible than promising a main-track paper by 25 October. Three days of experiments do not include the time to verify data, implement baselines, annotate semantic references, analyze failures, and write a defensible submission. Main-track acceptance cannot be inferred from a promising idea.

## 9. Final decision and acknowledged limits

**[Likely]** Prioritize the agentic analysis mechanism pilot if novelty and creative analytical design dominate and an independent human semantic reference can be prepared. Prioritize temporal graph compression if new annotation is unavailable and timestamped interaction data/code are ready. Use checkpoint-selection auditing as the evidence-grounded alternative when the necessary prediction traces are accessible. The separate [adversarial memo](D:/web_2/research/2026-10-15-webconf/shortlist-devils-advocate-c.md) identifies the strongest rejection threats and pilot controls.

**[Likely]** A coherent combination for the first direction is Thematic-LM's adaptive aggregation + the measurement concern illustrated by EARec/GenMentor/simulator work + fixed-budget controls motivated by efficient reranking. These are ingredients for one claim about semantic retention, not three separate contribution claims. Graphusion's one-relation constraint could become a later independent-domain replication of lossy consolidation, provided simultaneously true relations have independent gold support. Do not add that second domain merely to enlarge the paper; it introduces a distinct relation-typing and evaluation burden. A generic multigraph/provenance fix already has strong prior art.

**[Certain]** The audit does not establish that any proposed phenomenon occurs, that any proposed intervention works, or that any idea is the first of its kind. Some source texts remained partly inaccessible; the audit labels those limits. Runtime envelopes are estimates. Human reference quality is an unresolved resource requirement for the semantic study. Logs, public data availability, model invocation, and current account quotas remain to be verified before execution.

AI-assisted research disclosure: this dossier was prepared with AI-assisted primary-source retrieval, parallel source inspection, synthesis, and adversarial checking. It is research advice, not a report of newly executed results.
