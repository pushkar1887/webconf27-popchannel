# G3 abstract-number package — 2026-10-18 11:46 IST

**For human sign-off; not automatically frozen.** Scores are test capped Recall@20 (%); arrows are raw → repaired. Repair is raw+prior for BPR and cosine+prior for DirectAU, with β selected on validation. Source: `results/master_metrics_v2.csv` and the final RESULTS §5 blocks. Exact run IDs are below.

| Dataset | Full BPR | Full DirectAU | GraphHash BPR | GraphHash DirectAU |
|---|---|---|---|---|
| Gowalla | 16.55 → 17.06 | 18.02 → 19.57 | 9.37 → 11.31 | 7.55 → 13.30 |
| Yelp2018 | 8.02 → 8.02 | 11.10 → 11.26 | 3.78 → 4.80 | 2.41 → 5.68 |
| AmazonBook | 8.37 → 8.37 | 9.57 → 11.05 | 4.10 → 4.81 | 4.63 → 6.33 |

AmazonBook Full BPR is capped and descriptive. Gowalla final three-seed repaired Recall: **DAU 13.325 ± 0.024; BPR 11.257 ± 0.063** (sample SD).

**PR-E8:** paired-user bootstrap 2,000, seed 0; Holm p is from paired t-tests across the three dataset seed-0 comparisons.

| Pair | Recall Δ pp | 95% CI | Holm p |
|---|---|---|---|
| Gowalla s0 | 1.990 | [1.771, 2.238] | 2.43e-63 |
| Gowalla s1 | 2.130 | [1.909, 2.370] | — |
| Gowalla s2 | 2.084 | [1.855, 2.329] | — |
| Yelp2018 | 0.882 | [0.735, 1.031] | 6.14e-33 |
| AmazonBook | 1.518 | [1.405, 1.629] | 1.19e-159 |

**Mechanism (Stage 1/2 controls, not new causal proof):** A1 bucket/item Spearman **0.925/0.276**; Yelp A4 **0.944/0.520**. Fixed-β Shapley item-residual shares **0.71 Gowalla / 0.76 AmazonBook**; tie-only share **12%** (RESULTS PR-H/PR-DEC/PR-TIE; legacy GraphHash DAU control checkpoints). NDCG also favors DAU in every pair; its separate secondary CIs are in RESULTS.

**Limits to retain:** tail-primary deficits versus Full raw are **0.496 / 0.740 / 1.626 pp** on Gowalla/Yelp/AmazonBook seed 0; E7 fails, so float32 overhead is **5.52% / 3.90% / 7.30%**. Y3 and the original A8 range failed; stronger converged A8r does not erase the old failure. AmazonBook Full BPR is epoch-capped. C3 was not run. Alternate AmazonBook full-prior paired CIs and its fully retuned BPR weighted evaluation remain unavailable; alternate aggregate means retain the positive primary difference. E9 retuned ordering holds for Gowalla and Yelp only; AmazonBook remains incomplete. Novelty remains provisional; do not claim “first,” a universal loss reversal, tail safety, or a controlled contradiction of PRISM.

**Draft abstract (162 words)**

Embedding sharing reduces recommender memory, but can remove item-level popularity information. Building on PRISM’s analysis of embedding magnitudes and DNA’s normalization-and-bias design, we examine this loss across three datasets and graph-based, random, and double-frequency sharing. Under graph-based sharing, embedding lengths track bucket traffic more strongly than individual-item popularity. A fixed-beta decomposition attributes 71–76% of the observed repair gain to within-bucket item-level popularity rather than bucket calibration; resolving exact ties explains a smaller share. A validation-tuned per-item popularity term improves compressed DirectAU and exceeds equally corrected BPR by 0.88–2.13 Recall@20 points under GraphHash, with positive paired-user confidence intervals across all three datasets and three Gowalla seeds. This accuracy gain has a cost: repaired GraphHash DirectAU has lower tail recall than the corresponding Full DirectAU raw reference, and recommendation coverage contracts relative to its uncorrected cosine baseline. Quantized priors fail our precision criterion, so we report float32 storage overhead. The study identifies what sharing discards while separating accuracy recovery, tail costs, and ranking-policy limitations.

<details><summary>Exact source run IDs and supporting artifacts</summary>

- `amazonbook__mf__full__bpr__s0`
- `amazonbook__mf__full__directau__s0`
- `amazonbook__mf__graphhash__bpr__s0`
- `amazonbook__mf__graphhash__bpr__s0-legacy`
- `amazonbook__mf__graphhash__directau__s0`
- `gowalla__mf__full__bpr__s0-long`
- `gowalla__mf__full__directau__s0`
- `gowalla__mf__graphhash__bpr__s0-p500`
- `gowalla__mf__graphhash__bpr__s1`
- `gowalla__mf__graphhash__bpr__s2`
- `gowalla__mf__graphhash__directau__s0`
- `gowalla__mf__graphhash__directau__s1`
- `gowalla__mf__graphhash__directau__s2`
- `yelp2018__mf__full__bpr__s0`
- `yelp2018__mf__full__directau__s0`
- `yelp2018__mf__graphhash__bpr__s0`
- `yelp2018__mf__graphhash__directau__s0`

The E8 AmazonBook BPR source is the legacy control NPZ; the headline legacy scalar row is separately retained. Tail comparisons cite the same-dataset Full DAU run and each GraphHash variant in `results/E5_final.json`. Mechanism item-residual/tie shares come from legacy `gowalla__mf__graphhash__directau__s{0,1,2}-legacy30` and `amazonbook__mf__graphhash__directau__s0-legacy41` controls, documented in RESULTS PR-DEC/PR-TIE. Main table/seed values: `master_metrics_v2.csv`; intervals and policy boundaries: RESULTS final PR-E8 and `E8_final.json`; storage/tail/weighted values: RESULTS final S3-E5/S3-E7/PR-E9 and their JSONs. Missing controls: `prompts/post_stage3_gpu_rescore.md`.

</details>

**Orchestrator-suggested revision (adds the flip hook; same numbers, 185 words)**

Embedding sharing cuts recommender memory, but we show it also erases item-level popularity information, and this changes which training loss looks best. With full tables, DirectAU beats BPR (Gowalla 18.02 vs 16.55 Recall@20; Yelp2018 11.10 vs 8.02); under GraphHash sharing the order flips (7.55 vs 9.37; 2.41 vs 3.78). Building on PRISM's observation that embedding magnitudes track popularity and DNA's normalization-and-bias design, we trace the flip to pooling: a shared row's length follows its bucket's total traffic (Spearman 0.93–0.94), not the item's own popularity. A fixed-β decomposition attributes 71–76% of the repair gain to within-bucket item-level popularity; exact ties explain a smaller share. Adding a validation-tuned per-item term β·log(1+count) to cosine scores restores the order: repaired DirectAU exceeds equally repaired BPR by 0.88–2.13 points, with positive paired-user confidence intervals on Gowalla, Yelp2018 and AmazonBook and across three Gowalla seeds; the pattern extends to random and double-frequency sharing. The gain has a cost: tail recall stays below the full-table model and coverage contracts. The prior costs 3.9–7.3% of the compressed table in float32.

---

## v2 after the N1 full-text check (2026-10-18 23:10 IST) — for both authors' sign-off

**What changed and why**
- Cheng (arXiv 2606.21275, eq. 53) already publishes the score form `hᵀe_i + β·log π_i` on full tables. MEmCom (MLSys 2022) and Hash Embeddings already give each item a learned weight over a shared row. So the **repair is not our contribution**: it is presented as a minimal, training-free *test*, and the title leads with the diagnosis.
- Shiao et al. and GraphHash already *reported* that DirectAU does poorly under hashing, so we say "reported but not explained", never "never reported".
- New point that survives from the theory: DirectAU's loss is scale-invariant (Euler), so a learned per-item scale like MEmCom's gets **exactly zero gradient** from it. That is why a scoring-time term is needed. To add to PLAN §4 as a corollary [Identity] before the paper.
- Stage 4 is added: all 9 seed pairs pass, and the result also holds under random, double-frequency and double-graph sharing. LightGCN is left out of the abstract because 5 of its rows are budget-capped.
- We say plainly that the repair trades away part of GraphHash's popular/unpopular balance: coverage falls.

**Title options (pick one)**
1. *What Embedding Sharing Erases: Item-Level Popularity and the DirectAU–BPR Flip* ← recommended
2. *Shared Embeddings Forget Popularity: Why DirectAU Loses to BPR Under Hashing*
3. *Bucket-Level Popularity: How Embedding Sharing Reverses Loss Rankings in Recommendation*

**Abstract v2 (≈ 195 words)**

Embedding sharing, which hashes many items into one table row, is a standard way to shrink recommender memory. We show that it also erases item-level popularity information, and that this changes which training loss looks best. With full tables, DirectAU beats BPR (Gowalla 18.02 vs 16.55 Recall@20; Yelp2018 11.10 vs 8.02); under GraphHash sharing the order flips (7.55 vs 9.37; 2.41 vs 3.78), a gap reported before but not explained. Building on evidence that embedding magnitudes track popularity (PRISM), we trace the flip to pooling: a shared row's length follows its bucket's total traffic (Spearman 0.93–0.94), not the item's own. A capacity bound limits what any per-bucket quantity can encode, and because DirectAU is scale-invariant, a learned per-item scale receives no gradient from it. A validation-tuned β·log(1+count) term added at scoring time, a known score form, serves as a minimal test: a fixed-β decomposition attributes 71–76% of its gain to within-bucket item-level popularity, and repaired DirectAU exceeds equally repaired BPR by 0.83–2.13 points, with positive paired confidence intervals on three datasets with three seeds each, and under random, double-frequency and double-graph sharing. The recovered accuracy concentrates on popular items: tail recall and catalog coverage fall.

**Numbers used and their sources:** Full / GraphHash rows: RESULTS Stage 3 table and PR-Y. 0.93–0.94: PR-H. 71–76%: PR-DEC. 0.83–2.13 on 9 seed pairs: PR-E8 + S4-SEED. Other sharing schemes: Stage 3 main grid (6/6) + S4-DG. Tail and coverage: S3-E5 + PR-E5 addendum.

**Cite in the paper (from N1):** Cheng eq. (53) for the score form; MEmCom, Hash Embeddings, Seol et al. (WSDM 2024) for per-item parameters on shared tables; PRISM and CoRRe for popularity in vector length; Shiao et al. + GraphHash for the reported DirectAU-under-hashing gap; GDE (SIGIR 2026) for soft gating instead of hard buckets.

**Decisions for the authors:** (1) G1.d PASS with the narrower contribution? (2) title; (3) abstract v2 as is, or edits.
