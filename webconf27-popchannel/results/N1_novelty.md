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
