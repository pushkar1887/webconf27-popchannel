# 1 Introduction

<!-- Draft v1, 23 Oct. About 930 words of text, plus Figure 1. Citation keys are in [brackets]; resolve them in the .bib. Every number was checked against results/master_metrics_v2.csv and RESULTS.md. -->

Web recommender systems learn one embedding for every user and every item, and these tables account for most of a production model's memory [DLRM; Zhang2020freq]. The standard remedy is embedding sharing: many IDs are mapped to one row. The mapping can use the hashing trick [Weinberger2009], frequency-aware hashing [Zhang2020freq], or, more recently, graph-clustered buckets that place items with similar interaction patterns together [GraphHash; BACO; GDE]. Sharing schemes are compared by how much accuracy they retain per byte. Web platforms serve large and growing catalogs under fixed memory budgets, so what sharing preserves also decides which training objective a deployment should use. This paper asks what else sharing removes.

Two findings about full embedding tables set up the question. First, embedding magnitude encodes item popularity: frequently interacted items acquire longer vectors, and dot-product scores use that length to rank [PRISM; AdapTau; DDC]. Second, training objectives differ in whether they see magnitude at all. BPR [BPR] trains on dot products. DirectAU [DirectAU] optimises alignment and uniformity of ℓ2-normalised embeddings, so its loss is blind to magnitude, yet its models are still served with dot products.

These two findings collide under sharing. GraphHash [GraphHash] reports that every hashing method it evaluates loses accuracy when the objective changes from BPR to DirectAU, and suggests that DirectAU may be incompatible with hashing. Shiao et al. [Shiao2025] likewise find BPR ahead of DirectAU for out-of-vocabulary users represented by hashed embeddings. Neither work gives a mechanism. We reproduce the effect in a controlled setting (Figure 1a). On Gowalla with matrix factorisation, DirectAU leads BPR with full tables (18.02% vs 16.55% Recall@20) but trails it under GraphHash (7.55% vs 9.37%). On Yelp2018 the order also reverses (11.10% vs 8.02% becomes 2.41% vs 3.78%). The observation is not new; the explanation is missing. Without one, a practitioner cannot tell whether DirectAU is unsuited to compression or whether compression removes something DirectAU relies on.

None of the obvious remedies settles this.
- **More buckets** narrow the gap only by giving back memory. Even with 25,993 buckets for 40,981 Gowalla items, raw DirectAU still trails BPR (10.87% vs 12.61%).
- **A learned per-item scale on top of a shared row**, as in MEmCom [MEmCom], cannot help DirectAU: we show that such a scale receives no gradient from the DirectAU loss.
- **Better tie-breaking among items that share a row** recovers only 12% of the gain we report below.

A diagnosis has to come first.

Our central observation is that sharing makes magnitude a property of the bucket rather than of the item. A shared row's norm tracks the total traffic of its bucket: the Spearman correlation is 0.92–0.94 on Gowalla and Yelp2018, and 0.75 on AmazonBook. Its correlation with the popularity of the individual item falls from 0.79–0.91 on full tables to 0.19–0.52 (Figure 1b). We call this *bucket-level magnitude pooling*. Write an item's log-popularity as its bucket's mean plus a residual, p_i = μ_{b(i)} + ε_i. The residual ε_i is exactly the part of popularity that no shared-row representation can store.

We turn this observation into a diagnosis in three steps.

1. **Theory.** We prove three results. Items that share a row in matrix factorisation tie exactly. Under GraphHash, no bucket-level quantity can explain more than 36–48% of log-popularity variance. A learned per-item scale receives zero gradient from the DirectAU loss.
2. **Where the accuracy goes.** We decompose the drop from full to shared tables into a *magnitude channel* and a *direction channel*.
   - On Gowalla, 4.96 of the 10.47 percentage-point drop runs through magnitude.
   - On AmazonBook, the one dataset where the order does not reverse, the magnitude channel is only 0.87 points.

   We then test nine alternative explanations, including lost capacity, ties, initialisation, early stopping, weight decay and generic popularity.
3. **The diagnosis's prediction.** If pooled magnitude is the problem, replacing it with item counts should let DirectAU outperform BPR again. We test this with the log-prior score form of Cheng [Cheng2026, eq. 53], cos(p_u, q_i) + β·log(1 + c_i), applied after training, with β selected on validation. We call it the *count-restored score*. It is a probe of the diagnosis, not a proposed method.

Our contributions are as follows.

- **Diagnosis.** Under embedding sharing, magnitude tracks bucket traffic rather than item popularity, and for DirectAU it no longer helps ranking. Discarding it changes GraphHash DirectAU's Recall@20 by at most 0.44 points, against 0.87–4.96 points on full tables.
- **Analysis.**
  - Exact intra-bucket ties.
  - A correlation-ratio bound on the popularity that any bucket-level quantity can represent.
  - A zero-gradient result for learned per-item scales under DirectAU.
  - A two-channel decomposition that separates the loss of popularity information from the loss of directional capacity.
- **Test and cost.**
  - Restoring item counts raises compressed DirectAU from 7.55% to 13.30% on Gowalla.
  - Corrected DirectAU outperforms equally corrected BPR on all nine dataset–seed pairs (+0.83 to +2.13 points; every paired 95% confidence interval excludes zero).
  - 71–76% of the correction gain comes from item-specific popularity.
  - The correction adds 3.9–7.3% to table storage. It concentrates recommendations on popular items: on Gowalla, catalog coverage falls from 80.4% to 28.4%, close to the full model's 30.8%, and tail recall stays below the full model's.

We do not propose a new score form, and we do not claim that the correction is tail-safe or that the loss ordering reverses everywhere. It does not reverse on AmazonBook, and with LightGCN, BPR already leads on the full table. Predictions were registered before each experimental stage, and we report those that failed.

---

**Figure 1.** Under embedding sharing, DirectAU loses its lead over BPR, consistent with magnitude no longer following the item (Gowalla, matrix factorisation).
- **(a)** Test Recall@20 for full and GraphHash tables. Solid bars use raw dot-product scores; hatched bars add the count-restored correction under GraphHash (BPR 11.31%, DirectAU 13.30%).
- **(b)** Item embedding norm against log(1 + training count). Full table: Spearman 0.81. GraphHash: 0.28, with each horizontal stripe being one bucket whose items share a single norm.

---

## Notes for the author (delete before submission)

- **Citations to resolve:**
  - [DLRM] or another source for the claim that tables dominate memory. GraphHash §1 makes it; cite GraphHash if nothing better is at hand.
  - [AdapTau] = Chen et al., WWW'23.
  - [DDC] = KDD'25.
  - [Shiao2025] = WWW'25 Companion, doi 10.1145/3701716.3717581.
  - [Cheng2026] = arXiv 2606.21275.
- **Deliberate wording:**
  - "Frequently interacted items acquire longer vectors" does not attribute this to weight decay. Our wd = 0 run keeps ρ = 0.74.
  - The Shiao sentence matches what that paper tested: out-of-vocabulary users.
- **Numbers and their sources:**

  | Number | Source |
  |---|---|
  | 10.87 vs 12.61 | Resolution-3200 raw DirectAU vs BPR (S4-C1); 25,993 buckets (ALL_RESULTS Table 13) |
  | 12% | PR-TIE, tie-break-only share of the Gowalla DirectAU gain |
  | 0.92–0.94, 0.75 | ρ_bucket for GraphHash DirectAU |
  | 0.79–0.91 | Full DirectAU ρ_item (0.814 / 0.912 / 0.785) |
  | 80.4 → 28.4, 30.8 | E5 coverage: GraphHash raw → corrected; Full raw |

- **Length:** if the introduction runs over 1.1 pages, cut the Yelp2018 sentence in paragraph 3 first, then the "More buckets" bullet.
