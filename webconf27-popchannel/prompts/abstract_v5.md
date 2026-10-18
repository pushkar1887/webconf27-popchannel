# Abstract v5 (draft for both authors; 2026-10-19)

## Title options
1. **If Sharing Saves Memory, What Does It Cost? Popularity Without Item Identity** (recommended: adding "Item" makes "identity" unambiguous)
2. If Sharing Saves Memory, What Does It Cost? Popularity Without Identity (the GPT draft's title)
3. If Sharing Saves Memory, What Does It Cost? Buckets Keep Popularity, Items Lose It

## Abstract (264 words)
Embedding sharing shrinks recommender tables 4.6–7.4×, but it can erase item-level popularity and reverse which training objective performs best. Prior work links embedding magnitudes to item popularity and controls it with item biases or log-popularity score terms. Compression methods share rows through random or frequency-aware hashing, and recent graph-based methods group items by interaction structure. Neither line asks what sharing does to the popularity signal, and a reported reversal remains unexplained: DirectAU beats BPR with full tables (Gowalla Recall@20 18.02 vs 16.55) but trails it under graph-clustered sharing (7.55 vs 9.37). We present an empirical and analytical diagnosis. Shared embedding norms track bucket traffic (Spearman 0.93–0.94), not the item's own popularity (0.28–0.52), and a correlation-ratio bound caps the popularity variance any bucket-level quantity can explain (0.36 on Gowalla). Because DirectAU's loss is scale-invariant, a learned per-item scale receives zero gradient and cannot restore this signal. We therefore probe the loss with an established, training-free correction: a validation-tuned β·log(1+count) term added to frozen cosine scores. Scoring-time controls attribute 71–76% of its gain to item-specific popularity and 12% to tie-breaking, and shuffling counts within buckets removes it. The correction lifts graph-compressed DirectAU by 1.7–5.8 Recall@20 points (+37% to +136%) across three datasets, and corrected DirectAU exceeds equally corrected BPR on all nine dataset–seed pairs (+0.83 to +2.13 points, every paired-user confidence interval above zero). The ordering also holds for LightGCN, three other sharing schemes and every clustering resolution tested. It costs one float per item (3.9–7.3% of the compressed table), and its gains favor popular items: tail recall stays below full-table references and catalog coverage contracts.

## Review of the GPT draft (claim → evidence)
| GPT claim | Verdict | Fix in v5 |
|---|---|---|
| "three datasets and three seeds each", "0.83–2.13" | Correct after S4-SEED (9/9 pairs, all CIs > 0) | Kept |
| "71–76% … on two datasets" | Correct (PR-DEC: Gowalla 3 seeds, AmazonBook) | Kept |
| "improves … by 76% on Gowalla and 136% on Yelp2018" | Correct, but omits AmazonBook (+37%), and relative gains on a 2.41 base read as inflated | Absolute range + relative range over all three datasets |
| "shared embedding norms track bucket traffic more strongly than individual-item popularity" | Correct, but gives no numbers | Spearman 0.93–0.94 vs 0.28–0.52 |
| "controlled interventions distinguish …" | Overstated: these are scoring-time controls on frozen checkpoints, not training interventions | "Scoring-time controls" |
| "adding 3.9–7.3% to embedding-table storage" | Ambiguous (relative to the full table it is far smaller) | "of the compressed table" |
| "reverse the relative performance of training objectives" | Stated without evidence anywhere in the abstract | Flip numbers added (18.02 vs 16.55 → 7.55 vs 9.37) |
| (missing) why learned per-item fixes fail | Prop 1 corollary; supported by S4-C3 | Zero-gradient sentence |
| (missing) controls and generalization | Within-bucket shuffle (PR-DEC); LightGCN, random/DF/DoubleGraph, resolution sweep (S4-LGN, S3-B, S4-DG, S4-C1) | Added |

## New number sources (Stage 4, RESULTS on main)
- 4.6–7.4×: parameter counts 4,533,696/742,272 = 6.11 (Gowalla), 4,461,824/976,640 = 4.57 (Yelp), 9,231,488/1,255,168 = 7.35 (AmazonBook).
- Repair gain on graph-compressed DirectAU: +5.75 / +3.27 / +1.70 points = +76% / +136% / +37%.
- Corrected DirectAU − corrected BPR: +0.83 to +2.13 on all 9 dataset–seed pairs (PR-E8 + S4-SEED).
- LightGCN: flip present (GraphHash DAU raw 11.29 < BPR 16.22; 4.71 < 6.96); repaired DAU 17.06 > 16.27 and 9.12 > 6.96 (S4-LGN; Yelp GraphHash DAU and Full LightGCN rows are time-capped).
- Other sharing schemes: random and double-frequency on 3 datasets (S3-B), DoubleGraph on 2 (S4-DG); every resolution 200–3200 (S4-C1; r1600 capped).
- Learned residual bias (C3) underperforms the post-hoc term (4.56 vs 13.21 on Gowalla); implementation not ruled out, so it is **kept out of the abstract**.
