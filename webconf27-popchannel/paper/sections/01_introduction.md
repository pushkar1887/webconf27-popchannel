# 1 Introduction

<!-- Draft v2, 23 Oct. Follows the seven-block pattern seen in WWW'25 introductions (P1–P7), about 1,120 words, inside the 900–1,200 target. Citation keys are in [brackets]. Every number was checked against results/master_metrics_v2.csv and RESULTS.md. -->

<!-- P1 Context and importance (~140 words) -->
Web recommender systems learn one embedding for every user and every item. On platforms with large catalogs, these tables account for most of a model's parameters and memory [GraphHash; DLRM]. Embedding sharing is the standard way to fit them into a fixed budget: many IDs are mapped to one row, so the table grows with the number of buckets rather than with the catalog. Sharing schemes are routinely compared by how much accuracy they keep per byte. That comparison is usually made under a single training objective, yet the objective a deployment should choose also depends on what the compressed table still represents. If sharing silently removes information that one objective relies on, a memory decision becomes a modelling decision. Understanding what sharing removes is therefore a practical question for Web-scale recommendation, not only an efficiency question.

<!-- P2 Existing approaches (~160 words) -->
Two lines of work bear on this question. The first, compression research, has steadily improved how IDs are assigned to rows.
- The hashing trick maps IDs to rows uniformly [Weinberger2009].
- Frequency-aware and double hashing give popular items private rows or row pairs [Zhang2020freq].
- Graph-based methods place items with similar interaction patterns in the same bucket. GraphHash [GraphHash] clusters the user–item graph by modularity and outperforms hashing baselines at equal parameter budgets. BACO [BACO] balances co-clusters of users and items, and GDE [GDE] replaces hard buckets with soft gating.

The second line studies full embedding tables. It shows that embedding magnitude encodes item popularity: frequently interacted items acquire longer vectors, and dot-product scores exploit that length [PRISM; AdapTau; DDC]. Training objectives differ in how they treat magnitude. BPR [BPR] trains on dot products. DirectAU [DirectAU] optimises the alignment and uniformity of ℓ2-normalised embeddings, so its loss never sees magnitude.

<!-- P3 Remaining limitation (~185 words) -->
These two lines have not been connected, and the gap shows up as an unexplained result. GraphHash reports that every hashing method it evaluates loses accuracy when the objective changes from BPR to DirectAU, and suggests that DirectAU may be incompatible with hashing. Shiao et al. [Shiao2025] likewise find BPR ahead of DirectAU for out-of-vocabulary users represented by hashed embeddings. Neither work identifies a mechanism.

We reproduce the effect under a controlled protocol (Figure 1a). On Gowalla with matrix factorisation, DirectAU leads BPR with full tables (18.02% vs 16.55% Recall@20) but trails it under GraphHash (7.55% vs 9.37%). On Yelp2018 the order also reverses.

The obvious remedies do not resolve this:
- **More buckets** give memory back, yet raw DirectAU still trails BPR even with 25,993 buckets for 40,981 items (10.87% vs 12.61%).
- **A learned per-item scale on a shared row**, as in MEmCom [MEmCom], receives no gradient from the DirectAU loss.
- **Better tie-breaking within shared rows** recovers only 12% of the gain reported below.

A practitioner therefore cannot tell whether DirectAU is unsuited to compression, or whether compression removes something DirectAU needs.

<!-- P4 Research question and insight (~95 words) -->
This motivates our central question: *what does embedding sharing erase, and does that loss explain why the preferred objective changes under compression?* Our key observation is that sharing makes magnitude a property of the bucket rather than of the item (Figure 1b). Under GraphHash, a shared row's norm tracks the total traffic of its bucket (Spearman 0.75–0.94). It tracks the popularity of the individual item far less: 0.19–0.52, against 0.79–0.91 on full tables. We call this *bucket-level magnitude pooling*, and we test the hypothesis that it accounts for DirectAU's loss under sharing.

<!-- P5 Approach (~200 words) -->
We investigate this hypothesis through a diagnosis in three parts rather than a new model.

1. **What a shared table can represent.** Write an item's log-popularity as its bucket's mean plus a residual, ℓ_i = μ_{b(i)} + ε_i. We prove three results:
   - items that share a row tie exactly in matrix factorisation;
   - under GraphHash, no bucket-level quantity can explain more than 36–48% of log-popularity variance;
   - a learned per-item scale receives zero gradient from the DirectAU loss.
2. **Where the accuracy goes.** We decompose the drop from full to shared tables into a *magnitude channel* and a *direction channel*. We then test nine alternative explanations, including lost capacity, ties, initialisation, early stopping, weight decay and generic popularity.
3. **The prediction the diagnosis makes.** If pooled magnitude is the problem, replacing it with item counts should let DirectAU outperform BPR again. We use the published log-prior score form of Cheng [Cheng2026, eq. 53], cos(p_u, q_i) + β·log(1 + c_i), applied after training with β selected on validation. We call it the *count-restored score*. It is a probe of the diagnosis, not a proposed method.

<!-- P6 Evidence (~115 words) -->
We evaluate on Gowalla, Yelp2018 and AmazonBook, with matrix factorisation and LightGCN, under GraphHash and three other sharing schemes.
- On Gowalla, 4.96 points of the 10.47-point drop run through the magnitude channel. On AmazonBook, the one dataset where the order does not reverse, the magnitude channel is only 0.87 points.
- Restoring item counts raises compressed DirectAU from 7.55% to 13.30% on Gowalla.
- Corrected DirectAU outperforms equally corrected BPR on all nine dataset–seed pairs (+0.83 to +2.13 points; every paired 95% confidence interval lies above zero).
- The correction is not free. It adds 3.9–7.3% to table storage and re-concentrates recommendations on popular items, which lowers tail recall and catalog coverage.

<!-- P7 Contributions (~180 words) -->
Our contributions are:

- **Diagnosis.** We show that under embedding sharing, magnitude tracks bucket traffic rather than item popularity, and that for DirectAU it no longer helps ranking. Discarding it changes GraphHash DirectAU's Recall@20 by at most 0.44 points, against 0.87–4.96 points on full tables.
- **Analysis.** We prove three results:
  - exact intra-bucket ties;
  - a correlation-ratio bound on the popularity that any bucket-level quantity can represent;
  - a zero-gradient result for learned per-item scales under DirectAU.

  We also separate the loss of popularity information from the loss of directional capacity.
- **Controlled test with costs.** We show that:
  - 71–76% of the gain from restoring item counts is item-specific;
  - the ordering result holds across seeds, sharing schemes, bucket resolutions and LightGCN.

  We also report the correction's storage, tail and coverage costs.

We do not propose a new score form, and we do not claim that the correction is tail-safe. The loss ordering does not reverse on AmazonBook, and with LightGCN, BPR already leads on full tables. Predictions were registered before each experimental stage, and we report those that failed.

---

**Figure 1.** Under embedding sharing, DirectAU loses its lead over BPR, consistent with magnitude no longer following the item (Gowalla, matrix factorisation).
- **(a)** Test Recall@20 for full and GraphHash tables. Solid bars use raw dot-product scores. Hatched bars add the count-restored correction under GraphHash (BPR 11.31%, DirectAU 13.30%).
- **(b)** Item embedding norm against log(1 + training count). Full table: Spearman 0.81. GraphHash: 0.28. Each horizontal stripe is one bucket whose items share a single norm.

---

## Notes for the author (delete before submission)

**Block map (template → this draft):**

| Block | Content | Words |
|---|---|---|
| P1 | Context and importance | 140 |
| P2 | Existing approaches | 160 |
| P3 | Limitation | 185 |
| P4 | Question and insight | 95 |
| P5 | Approach (diagnosis, not a method name) | 200 |
| P6 | Evidence | 115 |
| P7 | Contributions + scope | 180 |

**Page budget.** At about 1,120 words, the introduction needs about 1.3 pages, not 1.1. Take the extra 0.2 pages from Related Work: P2 already covers the compression and magnitude lines, so §2 can shrink to about 0.45 pages and focus on per-item terms and score-time popularity.

**If the page is tight, cut in this order:**
1. The GDE clause in P2.
2. The Yelp2018 sentence in P3.
3. The "More buckets" bullet in P3.

**Citations to resolve:**
- [DLRM] for the memory claim (GraphHash §1 also makes it).
- [AdapTau]: Chen et al., WWW'23.
- [DDC]: KDD'25.
- [Shiao2025]: WWW'25 Companion, doi 10.1145/3701716.3717581.
- [Cheng2026]: arXiv 2606.21275.

**Deliberate wording:**
- No "under weight decay" in P2. Our wd = 0 run keeps ρ at 0.74.
- The Shiao sentence matches the setting that paper tested: out-of-vocabulary users.

**Number sources:**

| Number | Source |
|---|---|
| 10.87 vs 12.61 | Raw DirectAU vs raw BPR at resolution 3200 (S4-C1) |
| 25,993 buckets | ALL_RESULTS Table 13 |
| 12% | PR-TIE |
| 0.75–0.94 | ρ_bucket |
| 0.79–0.91 | Full ρ_item (0.814 / 0.912 / 0.785) |
| 4.96, 10.47, 0.87 | Two-channel decomposition, OUTLINE §6.1 |
