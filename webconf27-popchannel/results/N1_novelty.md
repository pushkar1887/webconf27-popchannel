# N1 Novelty check (run 2026-10-17)

Evidence base: WebFetch/WebSearch only. "Verified" = title, authors and venue found on a primary or index page. Anything else is marked UNVERIFIED.
Limits: Semantic Scholar returned 6 citers for PRISM and 10 for GraphHash (it may undercount). The DNA page returned 403 (ScienceDirect), so there is no DNA citer list. Google Scholar was not queried. A negative result means "not found", not "does not exist".

## (a) Table

| Paper | Verified | Venue/Year | What it shows | Overlap | How to position |
|---|---|---|---|---|---|
| GraphHash (Wu, Loveland et al.), arXiv 2412.17245 | yes | WWW 2025 | Modularity clustering of the bipartite graph for table sharing. Reports every hashing method drops when going BPR to DirectAU (Sec 5.2.1, 6.1). Gives no mechanism: "may not be compatible with hashing-based methods" (cites [45], attributed to Shiao et al. 2024). Table 10 (Gowalla): double frequency hashing retrieves far more popular items (avg degree MF 722.99 vs 267.92 full model). Adam, lr {1e-2,5e-3,1e-3}, wd {1e-4,1e-6,1e-8}. Norm is not mentioned anywhere. | partial (the phenomenon, not the mechanism) | Our anchor. We supply the unexplained mechanism: sharing ties items in a bucket, which removes item-level popularity, and DirectAU relies on norm-encoded popularity. |
| PRISM, "On the Role of Weight Decay in Collaborative Filtering: A Popularity Perspective" (Loveland, Ju, Zhao, Shah, Koutra), arXiv 2505.11318 | yes | KDD 2025 (V2), pp 1975-1986, DOI 10.1145/3711896.3737068 | Weight decay encodes popularity into embedding magnitudes. DirectAU with weak or no wd loses the popularity-magnitude correlation and can fall below BPR. Proposes init-time popularity-aware magnitudes. | partial (mechanism for full tables); none on hashing | Source of "popularity lives in the norm". We extend it: under sharing the norm is per-bucket, item-level popularity is lost, and a score-time term restores it. |
| DNA, "GCNs meet long-tail: Embedding norm bias in GCN-based recommendations" (Debiasing with Norm Adjustment) | partial. Title and method come from a search snippet. Authors (Choi) and venue (Applied Soft Computing 2026) UNVERIFIED (ScienceDirect 403; PII S156849462501539X). | 2025/26 | Normalizes item embeddings only and adds a learnable item-specific bias, trained separately from the GCN. | partial (norm normalization plus item bias, full tables); none on hashing | Bias is learned and on full tables. Ours is a fixed log-count term applied under sharing. Cite as a conceptual cousin. |
| Improving Out-of-Vocabulary Handling in RecSys (Shiao et al.), arXiv 2403.18280 | yes (arXiv) | arXiv only on the abs page; Semantic Scholar lists "Improving Out-of-Vocabulary Hashing in Recommendation Systems", WWW 2025, as a GraphHash citer | LSH-based embedding tables for OOV. Abstract does not mention DirectAU or popularity. | none (not confirmed to be GraphHash's ref [45]) | Read the full text before claiming anything about it. |
| Balanced Co-Clustering ... Embedding Table Compression (BACO), arXiv 2604.18351 | yes | SIGIR 2026 (per Semantic Scholar) | Sharing method. BPR/LightGCN only, no DirectAU. Degree weights in clustering. No per-item bias. | none | Newer sharing method; our fix could plug into it. |
| Learning to Collide, arXiv 2203.15837 | exists; venue UNVERIFIED | - | Frequency-aware learned hash. | none | Frequency used for allocation, not for scoring. |
| A Rank-One Popularity Component in Dot-Product Recommender Scores (Cheng), arXiv 2606.21275 | yes (abstract only) | arXiv, June 2026 | Score = PMI + log p(i) + context offset. Separating the log-popularity term cut popularity-aligned score energy by 98.6% in a matched intervention. No hashing. | partial (log-popularity score term in full models) | Closest to the repair form. Cite. Not about sharing. |
| Rethinking Popularity Bias in CF via Analytical Vector Decomposition (DDC), arXiv 2512.10688 | yes (abstract) | KDD 2025 per Semantic Scholar (UNVERIFIED independently) | BPR creates a popularity direction in embedding magnitude. No hashing or DirectAU. | none | Related to the norm-popularity story for BPR. |
| Menon et al., Long-tail learning via logit adjustment, arXiv 2007.07314 | yes | ICLR 2021 | Post-hoc or in-loss logit adjustment by log class prior. | partial (method template) | Origin of the log-prior score adjustment. |
| Yi et al., Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations | title and venue yes (RecSys 2019, Google). First author name UNVERIFIED: search listed "Ji Yang" first, likely a parse artifact; check the ACM entry. | RecSys 2019 | logQ correction with streaming frequency estimates. | partial (log-frequency correction in training) | Cite for the logQ lineage. |
| Zhang et al., "Causal Intervention for Leveraging Popularity Bias in Recommendation" (PDA), arXiv 2105.06067 | yes (exact title) | SIGIR 2021, DOI 10.1145/3404835.3462875 | Deconfound popularity in training, add a chosen amount back at inference. | partial | Precedent for a score-time popularity term. |
| Chen et al., Adap-tau: Adaptively Modulating Embedding Magnitude for Recommendation, arXiv 2302.04775 | yes | WWW 2023, pp 1085-1096 | Normalized embeddings with temperature; magnitude tied to popularity bias. | none to partial | Norm-popularity link. |
| Park et al., "Toward a Better Understanding of Loss Functions for Collaborative Filtering" (MAWU), arXiv 2308.06091 | yes. The exact title is NOT "A Comparative Analysis...". | CIKM 2023 | Alignment/uniformity view of CF losses; MAWU = margin-aware alignment + weighted uniformity. | none | Fix the title in the bib. |
| Wu et al., On the Effectiveness of Sampled Softmax Loss for Item Recommendation, arXiv 2201.02327 | yes | ACM TOIS 2024 (DOI 10.1145/3637061; preprint 2022) | SSM reduces popularity bias; cosine limits magnitude learning. | none | Cite as TOIS 2024, not 2023. |
| Wei et al., Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System (MACR), arXiv 2010.15363 | yes | KDD 2021, pp 1791-1800, DOI 10.1145/3447548.3467289 | Item node's direct effect on score removed by counterfactual inference at test time. | partial | Precedent for score-time popularity removal. |
| Heo et al., AdamP, arXiv 2006.08217 | yes | ICLR 2021 | Momentum optimizers shrink effective step on scale-invariant weights. | none | Optimizer/norm dynamics only. |
| Wan et al., Spherical Motion Dynamics, arXiv 2006.08419 | yes. NeurIPS 2021 title differs: "...Learning Dynamics of Normalized Neural Network using SGD and Weight Decay" (arXiv title: "...Neural Network with Normalization, Weight Decay, and SGD") | NeurIPS 2021 | Weight-norm equilibrium under normalization, wd and SGD. | none | Cite with the NeurIPS title. |

Citation lists (Semantic Scholar, may be incomplete):
- PRISM citers (6): CanonFedRec (KDD 2026); Adaptive Rich-kernelized Contrastive Learning for CF (SIGIR 2026); GeoGR (arXiv 2026); Rethinking Popularity Bias via Analytical Vector Decomposition (KDD 2025); AgentDR (WWW 2025); Generative Rec with Semantic IDs: A Practitioner's Handbook (CIKM 2025). None studies hashing x loss.
- GraphHash citers (10): Federated Rec with Discrete GCN (AI 2026); Graph Diffusion Gated Embeddings (SIGIR 2026); Robust Contrastive Graph Clustering (IJCAI 2026); Robust Multimodal Rec via Graph Retrieval (SIGIR 2026); Balanced Co-Clustering for Embedding Table Compression (SIGIR 2026); Semantic IDs at Snapchat (SIGIR 2026); Beyond the Academic Monoculture (arXiv 2026); Paper Skygest (arXiv 2026); Lighter-X (VLDB 2025); Improving Out-of-Vocabulary Hashing (WWW 2025). Only BACO is a compression paper, and it uses BPR; no loss x compression study.
- DNA citers: NOT RETRIEVED.

Searches run (none found a paper combining the claim): "embedding compression popularity", "hashing trick popularity bias recommendation", "embedding sharing collisions popularity", "DirectAU compression / alignment uniformity hashing", "logit adjustment recommendation compression".

## (b) PRISM setup (arXiv HTML v2; characters 0-130000 of 133844 read; last ~3.8k unread)
- Optimizer: NOT NAMED in the text I read. The theory uses plain gradient descent with learning rate eta; weight decay is described as applied to "all parameters attached to the optimizer" in PyTorch. Do not state Adam or SGD for PRISM.
- Learning-rate grid {0.1, 0.01, 0.001}. Weight-decay grid {0.0, 1e-4, 1e-6, 1e-8}.
- Embedding dim 64. Up to 1000 epochs, early stopping on validation NDCG@20, patience 10.
- Init: PyTorch uniform Xavier.
- Data: MovieLens1M, Gowalla, Yelp2018, AmazonBook; random 80/10/10 split.
- Batch size 16384 (BPR, SSM), 4096 (DirectAU, MAWU). SSM uses 10 negatives. DirectAU gamma in {1, 2, 5}. Single V100.
- DirectAU with no weight decay (lambda = 0): the text I read gives NO numeric table for it; findings are qualitative or in figures.
  - Sec 3.1.1: weak wd lets BPR beat DirectAU on MovieLens1M and Gowalla (Fig 1).
  - Fig 2(b) and Sec 3.1.2: at lambda = 0 the norm-popularity correlation "is significantly reduced, and minimal popularity information is encoded". Caption: "The weak correlation highlights that popularity information is not encoded."
  - Fig 3 / Sec 4.1.1: batched weight decay performs about the same as lambda = 0.
  - Appendix: Fig 6 (training curves), D.1 LightGCN+DirectAU, D.2 debiasing vs ReSN/IPL/MACR, D.3 SSM/MAWU, D.4 non-linear MF.
- Hashing / sharing / compression: none of these words appear in the text read. Only related remark: in distributed settings embedding tables are partitioned, which affects how wd is applied.

## (c) Kill-rule verdict
No. Among everything retrieved, no paper shows both that sharing/hashing destroys item-level popularity information and that a per-item popularity term repairs it. GraphHash reports the DirectAU-under-hashing drop without a mechanism; PRISM gives the norm mechanism for full tables only; DDC, Cheng 2026, MACR, PDA and logit adjustment give score-time popularity terms without hashing. Caveat: the sweep was shallow (DNA citers and Google Scholar unchecked; the last ~3.8k chars of PRISM unread), so recheck before submission.

Three closest: (1) GraphHash (observes the drop, no mechanism); (2) PRISM (norm encodes popularity; DirectAU needs wd); (3) Cheng 2026, arXiv 2606.21275 (log p(i) score term), with DNA next.

---

# N1 deep sweep (run 2026-10-18, 22:06 IST)

Evidence base: search-engine snippets (WebSearch), SciSpace abstracts, and the local full texts in `research/2026-10-16-loss-compression-audit/` (GraphHash, BACO, CCE). This cloud session's network policy **blocked** arxiv.org, export.arxiv.org, api.semanticscholar.org, dl.acm.org, ScienceDirect, OpenReview, DBLP, alphaXiv and par.nsf.gov, so no full text outside the local PDFs was read and no citer list was refreshed. "Verified" below means title, authors and venue appear in a primary or index record.

## (d) New or upgraded entries

| Paper | Verified | Venue/Year | What it shows | Overlap with our claim | How to position |
|---|---|---|---|---|---|
| **MEmCom**: Pansare, Katukuri, Arora, Cipollone, Shaik, Tokgozoglu, Venkataraman, *Learning Compressed Embeddings for On-Device Inference*, arXiv 2203.10135 | yes (abstract) | MLSys 2022 (oral) | A hashed shared table plus a second table with **one trainable weight per entity**, "allowing the model to distinguish between entities sharing the same embedding". The survey arXiv 2408.02304 writes it as e_i = w_i · T1[i mod m] (a per-entity scalar times the shared row); not read in the original | **High for the repair's shape** (a few bytes per item on top of a shared table). None on popularity, DirectAU or diagnosis | Must be cited. If the multiplicative form is right, PLAN Prop 1 (scale invariance) gives ∂L_DAU/∂w_i = 0 for w_i > 0: a learned per-item scale gets **no gradient** under DirectAU and is invisible to cosine scoring. That explains why a score-time term is the natural repair for DirectAU. Likely reviewer request: a MEmCom baseline under BPR |
| **Hash Embeddings**: Svenstrup, Hansen, Winther, arXiv 1709.03933 | yes | NeurIPS 2017 | Each token combines k shared component vectors with **per-token importance parameters** | Medium (per-token parameters resolve collisions); NLP; no popularity | Cite as the origin of "shared vectors + per-entity parameters" |
| **Proxy-based item representation**: Seol, Gang, Lee, Park | yes (abstract) | WSDM 2024, doi 10.1145/3616855.3635824 | Items = weighted sums of shared proxy embeddings; "may incorporate bias terms in case of frequent items" | Medium (sharing + per-item bias for frequent items); no diagnosis | Cite in the per-item-term related work |
| **CoRRe**: Kim (KAIST), Loveland (Snap) et al., *Training-Free LLM-Based Recommendation with Post-LLM Item Refinement Using Collaborative Signals*, arXiv 2608.19665 | partial (snippets; arXiv Aug 2026; no venue found) | 2026 | Training-free: refines LLM item-embedding **directions** with a co-purchase graph and **magnitudes with item popularity** | Medium for "training-free popularity injected into magnitudes"; no sharing/hashing | Cite. It also shows the GraphHash/PRISM group (Loveland, Ju, Zhao, Shah, Koutra) is actively publishing on magnitude = popularity: **highest scoop risk and the most likely reviewers** |
| **Shiao et al.**, *Improving Out-of-Vocabulary Handling (Hashing) in Recommendation Systems*, arXiv 2403.18280 | yes | WWW 2025 Companion, doi 10.1145/3701716.3717581 | This is GraphHash's ref [45] for "DirectAU may not be compatible with hashing-based methods" (GraphHash §5.2.1, local text). Snippets: BPR beats DirectAU for OOV users in its LSH/hashing settings. No mechanism visible in snippets; full text not read | Low–medium: a **second prior report** of the DirectAU-under-hashing problem, without an explanation | Cite next to GraphHash as the observation we explain. Do not say the flip "has never been reported" |
| **Cheng**, arXiv 2606.21275, *A Rank-One Popularity Component in Dot-Product Recommender Scores: Population Theory and Prior-Separation Evidence* | title + abstract (snippets); first author not re-verified | arXiv, June 2026 | Population-optimal score = PMI + log p(i) + context offset. Prior adjustment raises Macro NDCG@50 (≈0.187 → 0.214) and **lowers catalog coverage (≈0.985 → 0.726)**. No hashing in the visible text | Medium for the log p(i) score term | Cite. Its coverage drop is independent evidence for our E5 tail/coverage cost: a log-prior term trades coverage for accuracy in general, not only under sharing |
| **DNA**: Yeojun Choi, Woo-Seong Yun, Chanwoo Jeong, Yoon-Sik Cho, *GCNs meet long-tail: Embedding norm bias in GCN-based recommendations* | **yes (DBLP record)**; this resolves the first pass's "UNVERIFIED" | Applied Soft Computing 186:114226 (2026) | As in the first pass | As in the first pass | Citers still not retrievable |
| **GraphHash** (local full text) | yes | WWW 2025 | Double-frequency hashing "still tend[s] to bias towards popular items in retrieval"; GraphHash "maintains a better balance by retrieving both popular and unpopular items" (Table 10) | Context, not overlap | Say openly that our repair re-concentrates GraphHash's top-20 on head items (Gowalla coverage 95% → 28%, E5). It trades away part of GraphHash's own diversity advantage for accuracy |
| **BACO**, arXiv 2604.18351 (local full text) | yes | SIGIR 2026 | Degree-related weights in the clustering objective; BPR/LightGCN only; reports beating GraphHash | None (no DirectAU, no score-time popularity term) | Newer sharing method; future work for our analysis |
| Frequency-aware allocation: frequency/double hashing (arXiv 2007.14523), Learning to Collide (arXiv 2203.15837), CAFE (SIGMOD 2024), MEC (arXiv 2502.15355, popularity-weighted quantization) | yes (abstracts) | 2020–2025 | Popularity decides **which IDs get rows or codes** | Low: popularity used for allocation, not to repair scores | Our double-frequency cells (alias tuples ≈ unique, η² 0.999) still show DirectAU raw < BPR raw on Gowalla (3.22 vs 3.93): allocation alone does not restore what DirectAU loses |
| *Curse of "Low" Dimensionality in Recommender Systems*, arXiv 2305.13597 | yes (abstract) | SIGIR 2023 | Low-dimensional dot-product models amplify popularity bias | Low | Cite in capacity ↔ popularity background |
| Meehan & Pauwels, *On Inherited Popularity Bias in Cold-Start Item Recommendation* | yes (abstract) | RecSys 2025 | Post-hoc mitigation using embedding magnitude as a popularity proxy | Low | Optional cite (magnitude ↔ popularity, post-hoc) |

Searches run this pass (none found a paper combining both kill-rule parts): hashing collisions × item-level popularity; embedding compression × popularity bias / long tail / fairness; DirectAU × compression / hashing; hashed embeddings + per-item bias / scalar; QR trick + item bias; hashing trick + log-popularity prior (2025–2026); GraphHash follow-ups 2026; Loveland/Koutra 2026 papers; SciSpace semantic queries on the same three questions.

## (e) Kill-rule verdict after the deep sweep
**Not triggered.** No paper found shows that sharing/hashing destroys item-level popularity information **and** repairs it with a per-item popularity term.

The repair's **shape** is not new, though:
- per-item scalars on a hashed table (MEmCom, Hash Embeddings);
- per-item bias on shared proxies (Seol et al.);
- training-free popularity magnitudes (CoRRe);
- log-prior score terms (logit adjustment, PDA, Cheng).

The DirectAU-under-hashing problem has also been **reported twice** (Shiao et al.; GraphHash), both times without a mechanism.

**Recommended G1.d: PASS, with a narrowed contribution** (the human decides). The contribution is:
1. the diagnosis: bucket pooling of length (ρ_bucket 0.93–0.94 on Gowalla/Yelp vs GraphHash ρ_item 0.19–0.52 across the 3 datasets);
2. the exact decomposition (item residual 0.71–0.76 of the gain; ties 12%);
3. the capacity bound (Theorem 2);
4. the explanation of the reported DirectAU-vs-BPR flip, with controlled comparisons.

The post-hoc term is the **instrument**, not the novelty.

## (f) Consequences for the abstract and paper
1. Never write "first", "novel repair", or "propose a repair". Present the per-item term as a known, minimal, training-free probe.
2. **Title risk** [Likely]: "the Few Bytes That Repair It" invites the reviewer reply "that is MEmCom with a fixed weight". Put the diagnosis in the title.
3. Related work must cite MEmCom, Hash Embeddings, Shiao et al. and CoRRe in addition to PRISM, DNA and Cheng.
4. Add the Prop 1 corollary: under DirectAU a learned multiplicative per-item scale receives zero gradient. This must be checked against MEmCom's actual equation before it goes in the paper.
5. Expect the GraphHash/PRISM authors as reviewers. Credit their observations precisely.

## (g) Still not done (access-blocked here; about 30 min on a normal connection)
- Full text of Cheng 2606.21275 (scope check for hashing).
- The last ~3.8k characters of PRISM.
- The DirectAU section of Shiao et al. (does it propose a reason?).
- MEmCom's exact embedding equation.
- DNA citers and Google Scholar citers of GraphHash/PRISM after August 2026.

---

# (h) Full-text verification after network access was opened (2026-10-18, 22:56 IST)

The human opened network access. arxiv.org, export.arxiv.org, Crossref, DBLP, OpenReview and NSF PAR now return 200. Semantic Scholar and OpenAlex return 429 (rate-limited, reachable after retries). dl.acm.org returns 403 (bot block). The PDFs were read with `pdftotext`; facts below are from the full texts.

| Check from (g) | Full-text finding | Consequence |
|---|---|---|
| **Cheng 2606.21275** (Yang Cheng, independent researcher; v1 19 Jun 2026; 21 pp.) | **0** matches for hash / compress / collision. "shar" appears only as "context-shared". DirectAU is cited once, as related work. Eq. (53): **r_ci = h_c^T e_i + β·log π_i**, "where β is selected for the desired balance among accuracy, novelty, coverage, and tail exposure". Prior restoration: catalog coverage 0.98482 → 0.72612; Macro NDCG@50 0.18662 → 0.21350; Macro Recall@50 Δ CI includes 0. Setting: sequential recommendation, Alibaba/Tianchi, full tables | **Our scoring formula already exists on full models.** Cite eq. (53) as the score form. Our contribution is what sharing removes and why the β·log-count term recovers it under sharing, not the formula |
| **PRISM 2505.11318** (18 pp., incl. appendix) | **0** mentions of hashing, compression, buckets or collisions (the one "hash" hit is the name "Mahashweta" in the references). **No optimizer is named anywhere** (the one "Adam" hit is "Jay Adams" in the references). Grid lr {0.1, 0.01, 0.001}, wd {0, 1e-4, 1e-6, 1e-8}; dim 64; uniform Xavier; ≤1000 epochs; patience 10 on NDCG@20; V100. λ = 0 for DirectAU appears only qualitatively (Fig. 2b "weak correlation"). Fig. 3: batched weight decay performs like λ = 0. "Embedding tables are partitioned" appears only in a distributed-training remark about how weight decay is applied | No overlap with sharing. Our D7 (ρ 0.74 at wd 0, Adam) remains a **difference under our setup**. PRISM gives no optimizer to compare against |
| **Shiao et al. 2403.18280** (11 pp.) | DirectAU appears 4 times. Only result statement: "Surprisingly, BPR outperforms DirectAU on OOV users across all three of the datasets" (context-free OOV setting, NDCG@20 for **unseen** users). No mechanism; popularity mentions are generic | An observation for OOV users only, not for in-vocabulary compressed tables. Cite as a prior report; our explanation is new relative to it |
| **MEmCom 2203.10135** (16 pp.) | Algorithm 2 (no bias): x = U[i mod m] ⊙ V[i], with **V ∈ R^{v×1}**: **one learned scalar per entity multiplies the shared row**. Indices are sorted by frequency. Algorithm 3 adds W ∈ R^{v×1} broadcast as a per-entity **vector offset** (not a score bias); "MEmCom with no bias performs equally well". Softmax-trained models on Newsgroup, MovieLens, Netflix, MSD, Games. **No popularity analysis; no cosine/normalized loss** (0 matches for "cosine") | Prop 1 corollary **holds for Algorithm 2** [Certain]: DirectAU and cosine scoring see only x/‖x‖ = θ̂_b for V[i] > 0, so ∂L_DAU/∂V[i] = 0. A learned per-item scale cannot carry popularity under DirectAU. It does **not** hold for Algorithm 3, whose offset changes the direction |
| **CoRRe 2608.19665** (full text) | **0** matches for hash / compress / bucket / collision / DirectAU / GraphHash / PRISM. Training-free LLM recommender: co-purchase-graph direction refinement + **popularity-based magnitude refinement** | No sharing. Keep it as a positioning cite and as the scoop-risk signal |
| **Citers** (Semantic Scholar, 18 Oct) | GraphHash 10, the same list as the first pass. PRISM 6, the same. DNA 1 (*A Robust Non-negative Tucker Factorization of Tensor Model for Dynamic QoS Prediction*, unrelated). Shiao 3 (unrelated, plus GraphHash). Cheng 0. OpenAlex lists fewer (GraphHash 4, PRISM 1, DNA 1) | No citing paper overlaps our claim |
| New: **GDE**, Lee, Roh, Seo, Lim (LINE Plus), *Graph Diffusion Gated Embeddings for Recommender Systems*, SIGIR 2026, pp. 913–923, doi 10.1145/3805712.3809569 (a GraphHash citer) | Abstract (search snippet): soft heat-kernel-diffusion gating over seed-cluster embeddings replaces hard buckets; beats hashing baselines on top-K and CTR. No popularity or DirectAU in the abstract | Newer sharing method; cite. Soft mixtures give items distinct vectors, so our exact-tie analysis would not apply there |
| arXiv API structured search (DirectAU × compression; hashing/sharing × popularity; compression × embedding norm) | No new candidates. Weak evidence: `all:DirectAU` returned only 1 entry, so the API indexes that token poorly | Not counted as a negative |

**Updated kill-rule verdict: not triggered.** This now rests on the full texts of every close paper that was readable. The two closest overlaps are now confirmed in full text:
1. **Cheng eq. (53)** has our score form on full tables.
2. **MEmCom Algorithm 2** has our "one number per item on a shared table" shape, learned and multiplicative.

Neither studies what sharing removes, the decomposition, the capacity bound, or the DirectAU-vs-BPR flip.

**Remaining gaps:** DNA full text (ScienceDirect not tried; low priority, since its citer list shows no overlap); Google Scholar; papers not yet indexed.

