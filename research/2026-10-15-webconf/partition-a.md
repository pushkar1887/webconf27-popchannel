# Partition A: verified methods, industry systems, and bounded extensions

[Certain] This is a literature investigation dated 2026-10-15, not an executed experiment. No Kaggle/API operation, credential access, training, or model inference was performed. Small public paper files were fetched for source inspection.

[Certain] Coverage is **19 F, 2 P, 0 A**. **F** means the actual full-text methods and evaluation were inspected selectively; it does not mean a word-for-word reading of every appendix. **P** means only indexed full-text passages or author documentation were inspected, with the inaccessible parts explicitly identified. **A** would mean abstract/metadata only. Local extracted source receipts are in `sources-a/`; PDF/HTML text extraction can duplicate mathematical tokens, so numerical claims below were checked against surrounding prose/tables.

[Likely] Feasibility refers to a narrowly scoped public-data experiment under three days of main runtime across three Kaggle accounts. It is conditional on actual endpoint/model/data availability; 23 API keys do not establish aggregate throughput. No proposed extension below has established novelty or measured benefit.

## A01 — Behavioral Homophily in Social Media via Inverse Reinforcement Learning: A Reddit Case Study

[Certain] **Record/read:** WWW 2025; arXiv 2502.02943v1; F. The Best Paper Award is confirmed by the authors' institution, EPFL. [Paper](https://arxiv.org/html/2502.02943v1), [publication](https://doi.org/10.1145/3696410.3714618), [award](https://actu.epfl.ch/news/philipp-schneider-wins-2025-web-conference-best--3/).

- [Certain] **Contribution/evaluation:** Maximum-entropy deep IRL estimates a 12-state, six-action policy per user; symmetric visitation-weighted KL compares policies. The case study contains 662 users, 15 home subreddits, and 5.9 million interactions; stance comes from a fine-tuned DeBERTaV3 classifier (§§3–4).
- [Certain] **Author boundary:** §5 explicitly identifies restricted conversation access and sparse state/action observations as limitations. Appendix C.2 compares fitted-policy trajectory likelihood with random policies.
- [Likely] **Reviewer boundary:** That comparison does not establish that IRL is necessary relative to smoothed empirical transition frequencies, nor identify a causal preference for similar others. Similar action distributions can arise from similar opportunity sets.
- [Likely] **Extension:** Compare IRL, Dirichlet-smoothed behavior counts, and opportunity-conditioned policies on held-out chronological trajectories, including stance-error sensitivity. **Medium feasibility** if the trajectory artifact is accessible; a synthetic identifiability test alone would not validate Reddit findings.

## A02 — Responsible Diffusion Models via Constraining Text Embeddings within Safe Regions

[Certain] **Record/read:** WWW 2025; arXiv 2505.15427v1; F. [Paper](https://arxiv.org/html/2505.15427v1), [publication](https://doi.org/10.1145/3696410.3714912), [author code](https://github.com/lzws/Responsible-Diffusion). The award category was not independently confirmed from a conference/institutional record in this partition.

- [Certain] **Contribution/evaluation:** Learns low-rank semantic direction vectors in text-embedding space. Evaluation includes I2P and Ring-A-Bell prompts with NudeNet/Q16, occupational gender/race balance using WinoBias templates and CLIP classification, and COCO-30K fidelity/alignment (§4).
- [Certain] **Author boundary:** §4.3 requires retraining the vector for SDXL's different encoder arrangement. Appendix B reports that early guidance damages semantics while late guidance becomes ineffective, with prompt-dependent sensitivity.
- [Likely] **Reviewer boundary:** Finite prompt tests and imperfect classifiers do not certify that a geometrically named “safe region” excludes all harmful outputs. Uniformity under a classifier is also a narrower target than comprehensive social fairness.
- [Likely] **Extension:** Audit joint safety–attribute-preservation under controlled prompt compositions and unseen benign constraints, with independent human checks. **Medium feasibility** on a small frozen checkpoint/prompt sample; broad safety certification is unsupported.

## A03 — Factorizing Personalized Markov Chains for Next-Basket Recommendation

[Certain] **Record/read:** WWW 2010, pp. 811–820; F, original ten-page proceedings PDF. Its **2025 Seoul Test of Time Award** is separately confirmed. [Original paper](https://archives.iw3c2.org/www2010/pub/pdfs/RendleFreudenthaler2010-FPMC.pdf), [publication](https://doi.org/10.1145/1772690.1772773), [official award](https://www.iw3c2.org/papers/PressRelease-ToT-Award-20250430.pdf).

- [Certain] **Contribution/evaluation:** Jointly learns long-term user–item and previous-basket–next-item factors through sequential BPR. Evaluates sparse/dense Rossmann drugstore subsets against MF, factorized/nonfactorized Markov chains, and popularity; metrics include HLU, top-five F-measure, and AUC (§6).
- [Certain] **Author boundary:** §4.1.1 explicitly adopts first-order chains and simplified itemwise transitions. §6.2 removes previously bought items from test baskets, making the evaluation a novel-item discovery task.
- [Likely] **Reviewer boundary:** The reported results therefore do not establish effectiveness for replenishment or the full next basket containing repeat purchases. This is a protocol boundary, not evidence that the model fails there.
- [Likely] **Extension:** Reevaluate repeat/novel partitions with transaction-level chronology and a repeat-aware baseline. **High computational feasibility**, but this is a mature literature and a weak novelty claim by itself; original proprietary-data reproduction remains unavailable.

## A04 — HtmlRAG: HTML is Better Than Plain Text for Modeling Retrieved Knowledge in RAG Systems

[Certain] **Record/read:** WWW 2025; arXiv 2411.02959v2; F. [Paper](https://arxiv.org/html/2411.02959v2), [publication](https://doi.org/10.1145/3696410.3714546), [author code/data instructions](https://github.com/plageon/HtmlRAG).

- [Certain] **Contribution/evaluation:** Cleans HTML, collapses redundant structure, forms adjustable-granularity block trees, then prunes using embeddings followed by a generative pruner (§3). Tests 400 sampled questions per dataset on ASQA, HotpotQA, NQ, TriviaQA, MuSiQue, and ELI5, using answer/overlap metrics (§4).
- [Certain] **Author boundary:** §4.6.2 reports excessive cost and degraded performance when the generative pruner receives overlong HTML without initial embedding pruning. No dedicated limitations section was identified.
- [Likely] **Reviewer boundary:** Format comparisons do not isolate whether gains depend on stable DOM organization rather than retained facts. Semantically equivalent template changes could alter block construction; robustness remains unestablished.
- [Likely] **Extension:** Apply paired DOM transformations preserving visible facts and answer evidence; separately control token budget, segmentation, and retained evidence. **High feasibility** with released test HTML and frozen models. A robustness experiment needs a clear mechanism beyond generic HTML compression.

## A05 — TourRank: Utilizing Large Language Models for Documents Ranking with a Tournament-Inspired Strategy

[Certain] **Record/read:** WWW 2025; arXiv 2406.11678v2; F. [Paper](https://arxiv.org/html/2406.11678v2), [publication](https://doi.org/10.1145/3696410.3714863), [author code](https://github.com/chenyiqun/TourRank).

- [Certain] **Contribution/evaluation:** Groups candidates into parallel tournaments, awards advancement points, and aggregates repeated shuffled tournaments (§3). Uses TREC DL19/DL20 (43/54 queries), eight BEIR datasets, several retrieval backbones/LLMs, order perturbations, and cost/latency analyses (§4).
- [Certain] **Author boundary:** §3.2 explains coarse single-tournament scores and variability from decoding/input order; repeated tournaments seek to reduce that variance. §4.5 explicitly measures the effectiveness–resource trade-off.
- [Likely] **Reviewer boundary:** Aggregate NDCG and perturbation robustness do not provide a per-query guarantee that early elimination preserves every highly relevant document. More rounds can improve stability without yielding calibrated uncertainty.
- [Likely] **Extension:** Audit elimination regret conditioned on first-stage relevance concentration and tournament grouping, with fixed calls/tokens. **High feasibility**, but generic adaptive-budget tournament ranking is already crowded; a new scheduler alone is an unsafe novelty claim.

## A06 — LLM4Rerank: LLM-based Auto-Reranking Framework for Recommendations

[Certain] **Record/read:** WWW 2025, pp. 228–239; arXiv 2406.12433v4; F. The preprint contains placeholder publication metadata, so the institutional record resolves the venue/DOI. [Paper](https://arxiv.org/html/2406.12433v4), [institutional record](https://scholars.cityu.edu.hk/en/publications/llm4rerank-llm-based-auto-reranking-framework-for-recommendations/), [publication](https://doi.org/10.1145/3696410.3714922).

- [Certain] **Contribution/evaluation:** An LLM navigates accuracy, diversity, fairness, backward, and stop nodes using a natural-language goal and reranking history. Evaluates 20-item GMF candidate lists on ML-1M, KuaiRand-Pure, and Douban-Movie with HR/NDCG, α-NDCG, and MAD (§§2–3).
- [Certain] **Author boundary:** Appendix F acknowledges slower reasoning than traditional rerankers and problems parsing lengthy contexts.
- [Likely] **Reviewer boundary:** Prompted “fairness” is operationalized using specific binary item partitions and average rank-derived scores; this does not establish general exposure fairness or reliable satisfaction of user constraints.
- [Likely] **Extension:** Test paraphrase-equivalent goals and explicit numerical constraints against deterministic constrained reranking at matched candidate sets and budgets. **High feasibility** for an inference audit; generic multiobjective agent reranking is already the paper's contribution.

## A07 — G-Refer: Graph Retrieval-Augmented Large Language Model for Explainable Recommendation

[Certain] **Record/read:** WWW 2025; arXiv 2502.12586v1; F. [Paper](https://arxiv.org/html/2502.12586v1), [publication](https://doi.org/10.1145/3696410.3714727), [author code](https://github.com/Yuhan1i/G-Refer).

- [Certain] **Contribution/evaluation:** Combines PaGE-Link-style prediction-sensitive graph paths, semantically retrieved nodes, graph-to-text translation, knowledge pruning, and retrieval-augmented LoRA tuning (§3). Tests Amazon-books, Yelp, and Google-reviews with semantic explanation metrics and five researchers choosing between explanations for 20 test pairs per dataset (§4).
- [Certain] **Author boundary:** §6 leaves fully training-free retrievers and transfer to other tasks for future work. Main RAFT experiments use eight A100 GPUs (§4.1.4).
- [Likely] **Reviewer boundary:** Prediction-sensitive path selection does not by itself guarantee that every assertion in generated prose is supported or faithful to the recommender. Review similarity and preference votes are different estimands.
- [Likely] **Extension:** Separately measure ranker-prediction fidelity, retrieved-evidence support, and prose claim fidelity under graph/text interventions. **Medium feasibility** with released graph/checkpoints; reproducing all fine-tuning exceeds the demonstrated resource setting. Faithful recommendation explanations require a separate novelty audit.

## A08 — PEAR: Position-Embedding-Agnostic Attention Re-weighting Enhances Retrieval-Augmented Generation with Zero Inference Overhead

[Certain] **Record/read:** WWW 2025, pp. 1693–1702; arXiv 2409.19745v2; F. [Paper](https://arxiv.org/html/2409.19745v2), [publication](https://doi.org/10.1145/3696410.3714795), [author code](https://github.com/TTArch/PEAR-RAG).

- [Certain] **Contribution/evaluation:** Discovers copying-suppressing attention heads using repeated random-token sequences and learns fixed head-output coefficients (§4). Evaluates RAG QA on 2WikiMultihopQA, MuSiQue, Qasper, document-position tests, and MMLU; models include Llama2-7B, OPT-6.7B, and Baichuan-13B (§5).
- [Certain] **Author boundary:** §4.1 does not claim the selected heads share one mechanism; detailed head-function analysis is deferred. §5.5 reports that reweighting too many heads can hurt performance.
- [Likely] **Reviewer boundary:** Improved copying under trustworthy evidence does not establish appropriate rejection of false, contradictory, or irrelevant retrieved context. MMLU is limited evidence for preservation of all model capabilities.
- [Likely] **Extension:** Measure the copy-versus-reject trade-off using paired correct/invalid evidence and abstention, then test evidence-conditioned interventions. **Medium feasibility** on one open model; API-only endpoints cannot support internal head intervention.

## A09 — MA4DIV: Multi-Agent Reinforcement Learning for Search Result Diversification

[Certain] **Record/read:** WWW 2025; arXiv 2403.17421v3; F. [Paper](https://arxiv.org/html/2403.17421v3), [publication](https://doi.org/10.1145/3696410.3714862), [author code](https://github.com/chenyiqun/MA4DIV).

- [Certain] **Contribution/evaluation:** Each document is a parameter-sharing agent choosing a ranking score; global observations, self-attention, and QMIX-style value decomposition optimize α-NDCG (§4). An episode has exactly one decision step. Evaluation uses 198 TREC Web Track queries with five-fold query splits and industrial DU-DIV's 4,473 queries (§5/Appendix E).
- [Certain] **Author boundary:** §6 proposes extending the approach beyond diversification to relevance/fairness; these remain future tasks.
- [Likely] **Reviewer boundary:** The setup demonstrates one-step structured ranking, not long-horizon coordination or LLM deliberation. Public-query sample size limits precision of subgroup conclusions; global observations also prevent a simple impossibility claim based solely on QMIX monotonicity.
- [Likely] **Extension:** Compare training objectives with identical observations/embeddings and measure query-level confidence intervals plus out-of-distribution intent structure. **Medium feasibility** on TREC; reproducing industrial results requires DU-DIV access, and objective-matched baselines may yield an audit rather than a novel algorithm.

## A10 — GraphHash: Graph Clustering Enables Parameter Efficiency in Recommender Systems

[Certain] **Record/read:** WWW 2025; arXiv 2412.17245v2; F. [Paper](https://arxiv.org/html/2412.17245v2), [institutional publication record](https://deepblue.lib.umich.edu/handle/2027.42/198682), [author code](https://github.com/snap-research/GraphHash).

- [Certain] **Contribution/evaluation:** Louvain modularity clusters user–item graphs into shared embedding buckets; DoubleGraphHash adds a random hash (§3). Tests retrieval on Gowalla/Yelp2018/AmazonBook and CTR on Frappe/MovieLens with several backbones, hashing baselines, and full tables (§5).
- [Certain] **Author boundary:** §7 leaves OOV adaptation and integration of frequency information open. Importantly, §5.3 already evaluates activity groups, Appendix D.4 tests popular/unpopular items, and D.8 tests sparsity.
- [Likely] **Reviewer boundary:** A generic tail-harm experiment repeats existing work. Static results do not establish that persistent-ID bucket assignments remain stable when preferences change, or that retraining cost stays low after reassignment.
- [Likely] **Extension:** Isolate assignment churn and recovery under chronological graph drift at fixed memory/update budgets. **High feasibility** on public MF data, but compare BACO (2604.18351) and MPZCH (2602.17050); generic balancing/freshness is already occupied.

## A11 — ImageScope: Unifying Language-Guided Image Retrieval via Large Multimodal Model Collective Reasoning

[Certain] **Record/read:** WWW 2025; arXiv 2503.10166v1; F. Primary title uses “ImageScope”. [Paper](https://arxiv.org/html/2503.10166v1), [publication](https://doi.org/10.1145/3696410.3714777), [author code](https://github.com/pengfei-luo/ImageScope).

- [Certain] **Contribution/evaluation:** Training-free semantic synthesis, dual-path retrieval, predicate verification, and holistic pairwise evaluation (§3). Tests CIRR, CIRCO, FashionIQ, Flickr30K, MSCOCO, and VisDial; the default pipeline uses several distinct 3B–8B models, verifies 20 candidates, then evaluates three (§4.1).
- [Certain] **Author boundary:** §3.4 explicitly relaxes strict conjunction because verifier/image limitations make satisfying every proposition too stringent; ranking counts satisfied propositions.
- [Likely] **Reviewer boundary:** Equal predicate counts need not respect essential constraints: satisfying many optional attributes can outweigh violation of one indispensable attribute. This is a potential objective mismatch, not an observed failure rate.
- [Likely] **Extension:** Annotate mandatory/optional constraints and audit logical violations, then compare lexicographic or calibrated constraint satisfaction at the same candidate/model budget. **Medium feasibility** on a small public subset; a complete multimodel reproduction needs careful GPU scheduling and a contemporary composed-retrieval novelty check.

## A12 — Thematic-LM: A LLM-based Multi-agent System for Large-scale Thematic Analysis

[Certain] **Record/read:** WWW 2025, pp. 649–658; **P**. Methods/evaluation passages of an anonymous submission were inspected through the primary PDF's search index; direct OpenReview requests hit browser verification and ACM PDF returned 403. Camera-ready equivalence is unverified. [Indexed submission](https://openreview.net/pdf?id=jiv0Gl6sto), [publication](https://doi.org/10.1145/3696410.3714595), [official proceedings record](https://archives.iw3c2.org/www2025/www2025-proceedings.pdf).

- [Certain] **Contribution/evaluation:** Coders emit codes, quotes, and IDs; an aggregator merges codes; a reviewer updates a codebook using top-ten embedding retrieval. Theme coders receive LLMLingua-compressed codebooks (§3.1). Experiments use GPT-4o, Dreaddit/Reddit Climate Change, three-run ROUGE dependability, and 50/50 transferability; quotes are capped at 20 per code/theme (§4).
- [Certain] **Author limitations:** No author-stated limitations passage was verified in the accessible sections. Code is described as supplementary; public artifact access remains unverified.
- [Likely] **Reviewer boundary/extension:** Freeze leaf annotations and vary stream order, rare-record partition dispersion, and consolidation depth; measure semantic survival separately from ID conservation. The original describes sequential updates, not a recursive merge-tree design. **High compute feasibility** on public subsets, but substantial 2026 novelty collisions below prevent a generic novelty claim.

## A13 — Unleash LLMs Potential for Sequential Recommendation by Coordinating Dual Dynamic Index Mechanism

[Certain] **Record/read:** WWW 2025; inspected arXiv 2409.09253v2, revised 2026-09-16; F. [Paper](https://arxiv.org/html/2409.09253v2), [publication](https://doi.org/10.1145/3696410.3714866), [author code](https://github.com/Esperanto-mega/ED2).

- [Certain] **Contribution/evaluation:** ED² jointly learns user/item hierarchical discrete indices and an LLM recommender, with multigrained alignment and instruction tasks for high-order interactions (§2). Tests Amazon Musical Instruments, Video Games, and Arts/Crafts/Sewing using five-core filtering, chronological per-user leave-one-out, HR/NDCG (§3).
- [Certain] **Author boundary:** No explicit self-limitations passage was verified; Appendix C reports eight 32GB V100 GPUs. The paper compares concurrent ETEGRec, STORE, and TokenRec rather than establishing an uncontested first claim.
- [Likely] **Reviewer boundary:** Offline leave-one-out does not establish stable index behavior under new users/items or continual updates. Historical data used for associated-user collections require cutoff verification; the text alone does not prove leakage.
- [Likely] **Extension:** Audit temporal cutoff contracts and index churn before attempting an incremental method. **Low full-reproduction feasibility**, **medium** for a released-checkpoint audit; dynamic semantic indexing is already a crowded direction.

## A14 — Knowledge Distillation for Enhancing Walmart E-commerce Search Relevance Using Large Language Models

[Certain] **Record/read:** WWW 2025 Companion; arXiv 2505.07105v1; F. [Paper](https://arxiv.org/html/2505.07105v1), [publication](https://doi.org/10.1145/3701716.3715242).

- [Certain] **Contribution/evaluation:** Fine-tuned 7B LLM teachers label expanded Walmart query–item logs; a BERT cross-encoder learns pairwise score margins, excluding descriptions for serving latency (§3). Uses a 32,586-pair tail-query golden set, human ranking judgments, interleaving, and a two-week production A/B test (§§4–5).
- [Certain] **Author boundary:** §6 notes limited teacher-data-version experiments, missing query attributes, relevance-only objectives, and deployment restricted to tail queries.
- [Likely] **Reviewer boundary:** A student lacking a label-relevant input field cannot reproduce arbitrary teacher distinctions that depend solely on that field. Reported average parity does not remove this information limit or show robustness to missing attributes.
- [Likely] **Extension:** Construct matched public product pairs with identical student-visible fields but description-dependent relevance, testing selective distillation/abstention. **High audit feasibility**, **low original-system reproduction feasibility**: proprietary data and eight-A100 teacher training are required. Learning with privileged information is established prior art.

## A15 — LREF: A Novel LLM-based Relevance Framework for E-commerce Search

[Certain] **Record/read:** WWW 2025 Companion; arXiv 2503.09223v1; F. The arXiv record/title omits the final word “Search”; the publication record includes it. [Paper](https://arxiv.org/html/2503.09223v1), [publication](https://doi.org/10.1145/3701716.3715246).

- [Certain] **Contribution/evaluation:** Uses initial/challenge/mislabeled auxiliary models for data selection, expert/rule/reflection CoT supervision, then DPO against optimistic relevance judgments (§2). Evaluates JD query–product classification with five ESMTR classes and relevance classification; reports offline F1/accuracy and production A/B comparisons (§3).
- [Certain] **Author limitations:** No dedicated limitations or concrete self-limitation passage was verified in §4's conclusion.
- [Likely] **Reviewer boundary:** Disagreement among auxiliary models does not certify label cleanliness; correlated model errors can pass a selection rule. Business improvements also do not isolate which stage generalizes to another relevance taxonomy.
- [Likely] **Extension:** Use known-corruption public relevance data to measure selection precision, rare-class retention, and calibration under correlated teachers, before optimizing labels. **Medium feasibility** for a small audit, **low** for a complete industrial reproduction. Generic CoT/DPO relevance tuning offers insufficient novelty.

## A16 — RTBAgent: A LLM-based Agent System for Real-Time Bidding

[Certain] **Record/read:** WWW 2025 Companion; arXiv 2502.00792v1; F. [Paper](https://arxiv.org/html/2502.00792v1), [publication](https://doi.org/10.1145/3701716.3715259), [author code](https://github.com/CaiLeng/RTBAgent).

- [Certain] **Contribution/evaluation:** An LLM agent uses tools, memories, reflection, and two-step decisions to adjust an expert-policy bid multiplier (§3). iPinYou replay uses 24 hourly decisions per day, fixed logged auction prices, shared CTR estimation, and three budget fractions (§4).
- [Certain] **Author boundary:** §6 acknowledges response latency, limited bidding knowledge, and undisclosed real-world revenue despite deployment.
- [Likely] **Reviewer boundary:** Logged-price replay evaluates a controller under a fixed market trajectory; it does not estimate how competitors or auction prices respond to changed bidding policies. Hourly multiplier decisions also differ from per-impression LLM reasoning.
- [Likely] **Extension:** Stress-test delayed/stale recommendations and price-distribution shifts against simple feedback controllers under equal information and call budgets. **High feasibility** on public replay subsets; interactive economic claims still require a justified simulator or deployment. Adding RAG is explicitly anticipated by the authors.

## A17 — GAS: Generative Auto-bidding with Post-training Search

[Certain] **Record/read:** WWW 2025 Companion; arXiv 2412.17018v2; F. [Paper](https://arxiv.org/html/2412.17018v2), [publication](https://doi.org/10.1145/3701716.3715226), [author-hosted PDF](https://www3.ntu.edu.sg/home/boan/papers/WWW25_GAS.pdf).

- [Certain] **Contribution/evaluation:** Perturbs a Decision Transformer's bid proposal, scores actions using history-conditioned transformer critics, and votes over estimates; offers inference search and fine-tuning variants (§3). Evaluates AuctionNet/sparse AuctionNet, preference variants, constraint-exceeding rates, and a Kuaishou live experiment (§§4–5).
- [Certain] **Author boundary:** §7 notes approximate expansion/rollout and weaker performance of the cheaper fine-tuned variant. §4.1 trains 14M-parameter critics for 400,000 steps on two H100 GPUs.
- [Likely] **Reviewer boundary:** Maximization over learned action values can exploit critic errors outside logged action support; voting can reduce disagreement without proving calibrated constraint safety.
- [Likely] **Extension:** Quantify search gain versus critic overoptimism as proposal radius/budget and logging-policy support vary; compare conservative selection. **Medium feasibility** for a reduced public-data study, **low** for the reported full setting. Generic offline-RL pessimism is established and cannot alone be claimed novel.

## A18 — Towards Explainable Search Results in E-commerce

[Certain] **Record/read:** WWW 2025 Companion, pp. 476–484; **P** (official abstract plus released author README). Full paper unavailable here: ACM PDF returned 403; author homepage points to ACM, with no independent paper copy found. [Publication](https://doi.org/10.1145/3701716.3715264), [author repository](https://github.com/tonnyaudio/explainable), [author homepage](https://www.liujingping.com/).

- [Certain] **Contribution/evaluation available:** Combines domain relevance explanations with BehaviorRAG evidence from behavior. The abstract reports private/public experiments and a Fliggy A/B improvement in UV-list-to-order. Repository documentation specifies Qwen2-72B label/reasoning generation and reverse verification, Qwen2-7B training, GraphRAG, BGE, and an Amazon public-data route.
- [Certain] **Author limitations:** Not verified; the full evaluation/splits/limitations were inaccessible. The README is not evidence of successful reproduction.
- [Likely] **Reviewer boundary:** Conversion lift alone cannot distinguish faithful explanations from persuasive ones. That concern is an estimand distinction, not an allegation of deceptive outputs.
- [Likely] **Extension:** Public-data claim-support and behavior-evidence swapping audit, holding relevance scores fixed. **Medium feasibility**, contingent on usable released examples; recover the paper before specifying exact replication claims.

## A19 — Transferable and Forecastable User Targeting Foundation Model

[Certain] **Record/read:** WWW 2025 Companion; arXiv 2412.12468v2; F. [Paper](https://arxiv.org/html/2412.12468v2), [publication](https://doi.org/10.1145/3701716.3715266).

- [Certain] **Contribution/evaluation:** Found aligns multimodal user-history representations with descriptions of future behavior; supports text-demand zero/few-shot targeting (§4). Pretraining uses approximately 500 million Alipay user–text pairs; evaluation includes targeting scenarios and linear probes (§5).
- [Certain] **Author boundary:** The final “Limitations and Future Work” section explicitly says transfer beyond Alipay is not fully tested and training remains expensive. The acknowledgement says the desensitized experimental datasets were destroyed.
- [Likely] **Reviewer boundary:** Future behavior used as a supervised target is not inherently leakage. Validity instead depends on strict separation of feature history, target horizon, pretraining examples, and downstream evaluation timestamps/users; those contracts need artifact auditing.
- [Likely] **Extension:** Test future-target alignment on public event streams with explicit rolling cutoffs and unseen-user/platform transfer. **Low direct-reproduction feasibility**; **medium** for a smaller analogue, which cannot substantiate Alipay-scale claims or by itself establish new foundation-model novelty.

## A20 — Dimension Mask Layer: Optimizing Embedding Efficiency for Scalable ID-based Models

[Certain] **Record/read:** WWW 2025 Companion; arXiv 2510.15308v1, uploaded after the conference; F. [Paper](https://arxiv.org/html/2510.15308v1), [publication](https://doi.org/10.1145/3701716.3715217).

- [Certain] **Contribution/evaluation:** Learns a continuous prefix embedding dimension using regularized masks and stochastic sigmoid gates, then hardens dimensions (§3). Evaluates ShareChat production models with a 14-day A/B test and public Avazu using a small MLP on one T4 (§4).
- [Certain] **Author boundary:** §4.1.1 notes exposure bias from existing ranking; §6 warns that small initial dimensions can shrink prematurely. The production treatment fixes the high-cardinality user-ID dimension at eight (§4.1.3).
- [Likely] **Reviewer boundary:** Sum of embedding dimensions is not memory bytes: table cardinalities and optimizer states matter. Therefore a reported reduction in summed dimensions cannot alone certify the same percentage reduction in actual model memory.
- [Likely] **Extension:** Optimize and report cardinality-weighted bytes, realized allocation/latency, and temporal calibration at matched quality. **High feasibility** on Avazu subsets, but mixed-dimension/resource-aware embedding research substantially predates this work.

## A21 — LLM-powered Multi-agent Framework for Goal-oriented Learning in Intelligent Tutoring System (GenMentor)

[Certain] **Record/read:** WWW 2025 Companion, pp. 510–519; arXiv 2501.15749v1; F. Industry-track attribution comes from the official author repository. [Paper](https://arxiv.org/html/2501.15749v1), [publication](https://doi.org/10.1145/3701716.3715244), [author code](https://github.com/GeminiLight/gen-mentor).

- [Certain] **Contribution/evaluation:** Fine-tunes goal-to-skill mapping, dynamically models learners, schedules paths, and generates resources (§4). Automatic evaluation uses 200 randomly paired resumes/job goals, GPT-4o judgments, and job-posting skill references; human preference tests and a questionnaire/interview study involve 20 employees (§§5–6).
- [Certain] **Author boundary:** §6.4 records requests for deeper subject matter, multimedia/interaction, and faster responses. The current repository identifies a separate frozen Python paper branch; current main is not automatically the experimental version.
- [Likely] **Reviewer boundary:** Random resume–job pairs are simulated learner inputs. Preference ratings and employee self-reports do not establish causal improvements in retained knowledge, transfer, or goal attainment.
- [Likely] **Extension:** Freeze a prerequisite graph and independently scored skill tests; audit mapping/path errors before a prospective controlled learning study. **High offline-audit feasibility**, but credible learning-effectiveness evidence requires human-study design beyond a compute-only reproduction.

## Novelty collisions and strongest provisional opportunities

[Certain] The findings below reject broad claims of novelty. They do not establish that the remaining narrower hypotheses are unstudied.

1. [Likely] **Fixed-leaf thematic aggregation under rare-evidence dispersion** is the most compatible inference-heavy candidate here. Preserve the complete corpus and leaf annotations; randomize only the placement of the same rare records across partitions and the aggregation/update schedule. Measure semantic survival at every stage, source-ID conservation, false merges, and minority-theme recall at a fixed token budget. Exact-ID preservation is a bookkeeping property; it cannot prove semantic preservation. A left-fold adaptive codebook and balanced recursive consolidation are distinct baselines. Public Dreaddit and Climate Change subsets support a bounded experiment, but rare-theme reference labels need independent human verification and preregistered definitions.

   [Certain] **Collision:** Bai and Finkelstein, *A Multi-Modular Human-AI Workflow for LLM-Assisted Thematic Analysis: Application to COPD Telerehabilitation Interviews* (2026), explicitly compares per-transcript initial codes, hierarchical L2/L3 consolidation, direct grouping of initial codes, and direct coding. It evaluates one model/dataset using semantic theme matching and reports direct grouping/L3 as the strongest alignments. Thus “hierarchical versus direct merging changes themes” is already tested. The paper's §4 identifies the single-model/dataset and cosine-similarity limitations. [Primary full text](https://journals.sagepub.com/doi/10.3233/SHTI260838).

   [Certain] **Collision:** *Large Language Model–Assisted Thematic Coding in Medical Education Research: Comparative Methodological Study* reports order randomization during inductive grouping and automatic tracking/repair of missing initial-code IDs. Only indexed passages were inspected; direct full-text access failed. This precludes a “first order audit” or “first ID conservation” claim. [Primary indexed record](https://www.sciencedirect.com/org/science/article/pii/S2561326X26006505).

   [Certain] **Collision:** *Automated Thematic Analysis for Clinical Qualitative Data: Iterative Codebook Refinement with Full Provenance* (2026) occupies generic iterative refinement/provenance, including Thematic-LM as a baseline. Abstract/official metadata inspected in this partition; the parent investigation inspected further text. [Primary record](https://arxiv.org/abs/2603.08989). [Likely] The residual differentiation must be a controlled dispersion/retention mechanism with stronger evaluation, not a renamed pipeline.

2. [Likely] **Persistent-ID graph-hash reassignment under preference drift** is the strongest non-LLM fallback. With a fixed bucket budget, distinguish freezing assignments, full reclustering, incremental local edits, and stable residual allocations. Audit assignment churn, recovery time, tail/per-community regret, memory, and total update cost. A small MF setup on timestamped public interactions could fit three days; this is conditional, not a runtime measurement.

   [Certain] **Collisions:** BACO explicitly improves GraphHash using balanced weighted co-clustering and secondary user clusters, including subgroup tests; its abstract/method/introduction were inspected. MPZCH addresses model freshness through probing, eviction, and slot reset; official abstract/metadata were inspected. Therefore balancing, tail evaluation, new-ID freshness, and collision reduction are not available generic novelty claims. [BACO primary full text](https://arxiv.org/html/2604.18351), [MPZCH primary record, v4](https://arxiv.org/abs/2602.17050v4). [Likely] Persistent existing-ID preference drift differs from obsolete-ID eviction, but requires a focused novelty search before selection.

3. [Likely] **Whether IRL earns its complexity for behavioral homophily** is a defensible audit question with a clear falsifiable baseline, but a weaker feasibility bet because original trajectory access is unverified. Compare held-out behavior prediction and similarity stability against count-based policies with opportunity conditioning and uncertainty. Do not claim latent motivations or causal homophilic choice from descriptive policy agreement. A convincing result must show an advantage beyond maximum-entropy smoothing and classification noise. A purely synthetic demonstration would explain identifiability, not prove an empirical social finding.

[Certain] Additional practical but novelty-uncleared audits are HtmlRAG's paired DOM invariance, DML's actual-byte objective, and G-Refer's prose-support versus ranker-fidelity distinction. [Likely] They are feasible components, not yet justified full-paper proposals. Generic adaptive reranking, provenance, multiagent diversity, RAG copying enhancement, and teacher–student relevance distillation all have substantial direct prior art in this partition alone.

## Blockers and evidence boundaries

- [Certain] A12 remains P because only primary indexed methods/evaluation passages were accessible; A18 remains P because the paper's full text was not obtained. No missing limitations were invented.
- [Certain] A13 was read in a September 2026 revision of a WWW 2025 paper; any reproduction must pin the intended version. A06/GraphHash/GenMentor preprints include placeholder publication metadata, resolved using publication/institutional records.
- [Certain] Main industrial data for Walmart, JD, Fliggy, and Alipay are not established as publicly reproducible here. Found's acknowledgement states the experimental dataset was destroyed. Public analogues support narrower conclusions only.
- [Certain] Several resource receipts exceed the proposed environment: G-Refer/Walmart use eight A100s; ED² uses eight 32GB V100s; GAS uses two H100s. Existing trained artifacts could lower costs, but availability was not validated by downloading model weights.
- [Certain] Primary author code links are availability pointers, not executed reproduction receipts. No dataset benchmark was actually downloaded, cleaned, split, or evaluated during this investigation.
