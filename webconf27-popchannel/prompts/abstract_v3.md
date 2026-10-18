# Abstract v3 (draft for both authors; 2026-10-18)

Embedding sharing cuts recommender memory, but we show it also erases item-level popularity information. Prior work showed that full-table recommenders store item popularity in embedding lengths, and that normalization, per-item biases or a log-popularity score term can control it. Recent sharing methods group items by hashing, frequency or user–item graph clustering such as GraphHash, and some learn per-item scalars to separate colliding items. Neither line asks what sharing does to the popularity signal, and a reported anomaly remains unexplained: DirectAU beats BPR with full tables (Gowalla 18.02 vs 16.55 Recall@20; Yelp2018 11.10 vs 8.02) but falls behind it under GraphHash (7.55 vs 9.37; 2.41 vs 3.78). We present a diagnosis and a minimal fix. A shared row's length follows its bucket's total traffic (Spearman 0.93–0.94), not the item's own popularity (0.28–0.52). We prove that no per-bucket quantity can explain more than a fraction η² of popularity variance (0.36 on Gowalla). An exact decomposition attributes 71–76% of the recoverable gain to the item-level residual no shared table can store; exact ties explain 12%. Learned per-item scales cannot recover it: DirectAU's scale-invariant loss gives them zero gradient. A training-free β·log(1+count) term added to cosine scores can: compressed DirectAU on Gowalla rises from 7.55 to 13.30, 42% above the best unrepaired GraphHash model, and beats equally repaired BPR by 0.88–2.13 points on all three datasets, with positive paired-user confidence intervals for every dataset and Gowalla seed. Shuffling popularity within buckets removes the gain. The float32 prior costs 3.9–7.3% of the compressed table and trades tail recall and coverage for head accuracy.

## Number sources (all in RESULTS.md / results/master_metrics_v2.csv)
| Claim | Value | Source |
|---|---|---|
| Full DAU vs BPR | Gowalla 18.02 vs 16.55; Yelp 11.10 vs 8.02 | §1 headline (A2, A8r s0-long; A3, A7) |
| GraphHash DAU vs BPR raw | 7.55 vs 9.37; 2.41 vs 3.78 | §1 headline (A1, s0-p500; A4, A5) |
| Bucket vs item length–traffic Spearman | 0.925 / 0.944 vs 0.276 / 0.520 | PR-H (Gowalla A1, Yelp A4) |
| η² | 0.359 (Gowalla GraphHash r200) | PR-M1 |
| Item-residual Shapley share | 0.71 Gowalla (3 seeds) / 0.76 AmazonBook | PR-DEC (legacy controls) |
| Tie-break-only share | 12% (Gowalla) | PR-TIE |
| Zero gradient for learned per-item scales | Prop 1 (PLAN §4.4) applied to MEmCom Alg. 2 | results/N1_novelty.md §(h) |
| 7.55 → 13.30; +42% vs 9.37 | 13.30/9.37 = 1.419 | §1 headline |
| Repaired DAU − repaired BPR | +0.88 to +2.13; all CIs > 0 (5 pairs) | PR-E8 final |
| Within-bucket shuffle | −0.36 (keeps −6% of gain) | PR-DEC |
| Float32 overhead | 3.90–7.30% | S3-E7 final |
| Tail / coverage cost | tail below Full raw on 3 datasets; coverage contracts | S3-E5 final, PR-E5 |
