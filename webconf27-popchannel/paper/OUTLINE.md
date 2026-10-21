# Paper outline: If Sharing Saves Memory, What Does It Cost? Popularity Without Identity

WebConf 2027, anonymous, 8 pages + references + appendix. Due Mon 26 Oct, 17:29 IST.

Every number here was checked against `results/master_metrics_v2.csv` and the RESULTS.md blocks. Look each one up again by run_id before it goes into the paper.

---

## 0. Fix these before writing (abstract vs. data)

- **F1. ρ ranges cover only 2 of 3 datasets.**
  - The abstract's "0.93–0.94 vs 0.28–0.52" is Gowalla and Yelp. AmazonBook is 0.75 / 0.19.
  - In the paper, write: ρ_bucket 0.75–0.94 vs ρ_item 0.19–0.52, across 3 datasets.
- **F2. 13.30 is cos + prior.** Raw + prior gives only 10.33.
  - Frame it as: under sharing, learned length carries nothing usable (raw 7.55 ≈ cosine 7.57). So the test replaces length with counts instead of adding counts to it.
  - Define once: s(u,i) = cos(p_u, q_i) + β·log(1 + c_i).
- **F3. The flip is not universal.**

  | Setting | Full: DAU vs BPR | GraphHash: DAU vs BPR | Flip? |
  |---|---|---|---|
  | MF Gowalla | 18.02 > 16.55 | 7.55 < 9.37 | yes |
  | MF Yelp | 11.10 > 8.02 | 2.41 < 3.78 | yes |
  | MF AmazonBook | 9.57 > 8.38 | 4.63 > 4.10 | no; DAU's lead shrinks +1.19 → +0.53 |
  | LightGCN Gowalla | 18.02 < 18.48 | — | no premise (BPR already ahead on Full) |

  - Claim instead: "sharing removes DirectAU's advantage".
- **F4. The prior also helps the Full table.** Gowalla Full DAU: cos + prior 19.57 vs raw 18.02.
  - Always report the interaction (Full vs. shared), never the shared gain alone.
- **F5. DoubleFreq.** η² = 0.999, so the bound does not bind. Yet ρ_item = 0.027 and DAU < BPR (3.22 vs 3.93).
  - Say the bound is sufficient, not necessary: the norm of a sum of shared rows follows bucket traffic even when identity survives.
- **F6. Random hashing is near-degenerate.** Raw Recall is 0.15–0.76%. Call it a stress case and report the table ratio.

---

## 1. Thesis and contributions

**Thesis.** Sharing makes vector length a property of the bucket, not the item. Losses that store popularity in length (DirectAU) therefore lose item-level popularity. A training-free item-count test recovers most of the loss, and the recovered part is mostly item-specific.

**Contributions** (3, none of them "a new method"):
1. **Diagnosis.**
   - Norms follow bucket traffic: ρ_bucket 0.75–0.94 vs ρ_item 0.19–0.52.
   - Removing length changes Recall by ≤ 0.44 under sharing, against 0.87–4.96 on Full tables.
2. **Analysis.**
   - Exact within-bucket ties.
   - A correlation-ratio bound: η² = 0.36–0.48 for GraphHash, so 52–64% of popularity variance is out of reach of any bucket-level quantity.
   - DirectAU is scale-invariant, so a learned per-item scale (MEmCom-style) gets zero gradient.
3. **A controlled test.**
   - Uses the published log-prior score form (Cheng 2026, eq. 53; Menon et al. 2021), applied post hoc.
   - Fixed-β decomposition: item-specific (ε) share 0.71 on Gowalla, 0.76 on AmazonBook. A within-bucket shuffle keeps −6% of the gain.
   - Corrected DAU beats corrected BPR on 9 of 9 dataset × seed pairs: Δ +0.83 to +2.13, every CI > 0.
   - Costs reported in full.

**Framing.** A diagnosis paper. The repair is a probe, not the contribution. State this early and once.

---

## 2. Page budget

| § | Section | Pages |
|---|---|---|
| 1 | Introduction | 1.25 |
| 2 | Related work | 0.75 |
| 3 | Setup and notation | 0.5 |
| 4 | Theory: what sharing does to popularity | 1.5 |
| 5 | Protocol | 0.5 |
| 6 | Results (RQ1–RQ6) | 2.75 |
| 7 | Limitations and failed predictions | 0.5 |
| 8 | Conclusion | 0.25 |
| A–J | Appendix | unlimited |

The main text must stand alone. The appendix holds everything else: every run, analysis and diagnosis.

---

## 3. Main text, section by section

### Abstract
- Keep the structure: problem → prior → recent → gap → ours → numbers.
- Revise using F1, F2, F3.
- Keep one sentence on cost.

### 1. Introduction
**Purpose:** pose the title's question, show the symptom, name the mechanism, list the contributions.

1. **Context.** Embedding tables dominate memory. Sharing schemes cut it: hashing; GraphHash (WWW'25); BACO, GDE (SIGIR'26). They are evaluated by recall per byte.
2. **Symptom.**
   - GraphHash and Shiao et al. (WWW'25 Companion) both saw DirectAU fail under hashing. Neither explained it.
   - Our numbers: Gowalla Full DAU 18.02 > BPR 16.55; GraphHash DAU 7.55 < BPR 9.37.
3. **Hypothesis.** On full tables popularity lives in length (PRISM, KDD'25). Under sharing, length belongs to the bucket.
4. **What we do.** Theory, then a minimal test: put item-level popularity back, and check that the gain is item-specific.
5. **Findings.** One bullet per RQ, with numbers.
6. **Contributions.** Three bullets, plus what we do not claim:
   - no new score form;
   - not tail-safe;
   - the flip is not universal.

**Fig 1 (page 1).** Grouped bars, Gowalla MF: {Full, GraphHash} × {BPR, DirectAU}.
- Raw values: 16.55 / 18.02 / 9.37 / 7.55.
- Corrected values: GraphHash BPR 11.31, GraphHash DAU 13.30.
- Caption: "Sharing removes DirectAU's lead; item counts bring it back."

### 2. Related work (3 paragraphs, each ending with what the cited work does not do)
1. **Sharing and compression.**
   - Cite: hashing trick; frequency / double hashing (2020); Learning to Collide; CAFE (SIGMOD'24); MEC; GraphHash; BACO; GDE (soft gating, so no exact ties).
   - Gap: none of these studies the popularity signal.
2. **Popularity in length and in losses.**
   - Cite: PRISM; Adap-τ (WWW'23); DDC (KDD'25); DirectAU (KDD'22); Wu et al. (TOIS'24); DNA (ASOC'26); CoRRe (2026).
   - Gap: all of these use full tables.
3. **Per-item terms and score-time popularity.**
   - Cite: MEmCom (MLSys'22); Hash Embeddings (NeurIPS'17); Seol et al. (WSDM'24); logit adjustment (ICLR'21); logQ (RecSys'19); PDA (SIGIR'21); MACR (KDD'21); Cheng (2026, eq. 53).
   - Gap: no link to what sharing erases, and no account of why a learned scale cannot help DirectAU.

### 3. Setup
- Notation:
  - users U, items I, train counts c_i;
  - score s(u,i) = ⟨p_u, q_i⟩;
  - sharing map b: I → B, with q_i = E[b(i)]; double hashing uses a sum of two rows.
- Losses: BPR (pairwise logistic), and DirectAU (alignment + uniformity on normalized vectors).
- Scoring modes: raw; cosine; cos + prior; raw + prior. β is picked on validation from a fixed grid.
- Measures:
  - ρ_item = Spearman(‖q_i‖, c_i);
  - ρ_bucket = Spearman(‖E_b‖, C_b), where C_b = Σ_{i∈b} c_i;
  - η² = Var(μ_b) / Var(p).

### 4. Theory (core section)
- **Lemma 1 (user scale).** Ranking does not depend on ‖p_u‖.
  - So only the item side can carry popularity.
- **Lemma 2 (exact ties).** Items in the same bucket score identically.
  - 87% of users have a tie at the rank-20 boundary; mean tied group 14.7 items.
- **Proposition 1 (scale invariance of DirectAU).**
  - Corollary: a learned per-item scale w_i has ∂L/∂w_i = 0.
  - Lengths are set only by weight decay and optimizer dynamics, and under sharing these aggregate over the bucket.
- **Proposition 2 (correlation-ratio bound).**
  - Decompose p_i = μ_{b(i)} + ε_i.
  - For any bucket-level f: corr(f, p)² ≤ η².
  - ε is orthogonal to every shared-row representation.
  - Measured η² for GraphHash: 0.36–0.48.
  - Remark: the bound is sufficient, not necessary (F5).
- **Mechanism schematic (Fig 2a).**
  - Left, Full: one arrow per item, length ∝ popularity.
  - Right, shared: one arrow per bucket, with items' count bars differing underneath. Annotate p = μ_b + ε.

### 5. Protocol
- **Datasets.** Gowalla, Yelp2018, AmazonBook (standard LightGCN / GraphHash splits).
- **Models.** MF with d = 64; LightGCN with 2 layers.
- **Sharing.**
  - GraphHash (Louvain, resolution 200), Random, DoubleFreq, DoubleGraph.
  - Table ratio vs Full: Gowalla 16.4%, Yelp 21.9%, AmazonBook 13.6%. These were derived from table bytes; verify against the mapping files.
- **Training and stopping.**
  - Converged means stopped by patience. Capped runs are flagged and never used for convergence claims.
- **Selection and evaluation.**
  - β is chosen on validation; test is scored once.
  - Recall@20 with denominator min(20, |T_u|); NDCG@20 is secondary.
- **Statistics.**
  - Paired user bootstrap with 2,000 resamples; paired t-test with Holm correction.
  - 3 seeds for the main pairs. Single-seed rows are marked †.
- **Pre-registration.** Predictions were registered before each stage. Give the PASS/FAIL tally here and the full registry in App. I.

### 6. Results
- **RQ1. Does sharing remove DirectAU's advantage?** (Tab 2)
  - The order flips on Gowalla and Yelp. On AmazonBook DAU's lead shrinks.
  - Random and DoubleFreq: raw DAU < BPR in 5 of 6 cells; AmazonBook DoubleFreq is the exception.
- **RQ2. Is popularity lost at item level?** (Fig 2b; Fig 3)
  - ρ_bucket vs ρ_item.
  - Cosine ablation:

    | Dataset | Δ(cos − raw), Full | Δ(cos − raw), GraphHash |
    |---|---|---|
    | Gowalla | −4.96 | +0.02 |
    | Yelp | −4.13 | −0.44 |
    | AmazonBook | −0.87 | +0.01 |

  - Gain from the prior, Gowalla: Full +1.55, GraphHash +5.75.
- **RQ3. Is the recovered gain item-specific?** (Fig 4; Tab 3)
  - Shapley shares: μ 0.29 / ε 0.71 (Gowalla); ε 0.76 (AmazonBook).
  - Within-bucket shuffle: −6% of the gain.
  - PR-E8 holds on 9 of 9 pairs.
  - Popularity-only baseline: 2.97 / 1.68 / 0.96, against 13.29 / 5.68 / 6.33 corrected. The test is not MostPop.
- **RQ4. Does erasing popularity without sharing reproduce the symptom?** (Tab 4; Full Gowalla DAU unless noted)

  | Intervention | Effect on test Recall@20 |
  |---|---|
  | Global length shuffle (10 seeds) | 17.90 → 6.28 |
  | Decile-preserving shuffle | 17.90 → 13.64 |
  | Length replaced by a fitted function of popularity | 17.90 → 18.69 |
  | Forced equal norms | 12.77 → 19.32 with prior |

  - AmazonBook, same order: global 7.44, decile 9.30, replacement 10.64 (baseline 9.57).
  - This is the causal argument; give it space.
- **RQ5. Does it generalize?** (Tab 5; Fig 5)
  - DoubleGraph: corrected DAU vs corrected BPR is 10.91 vs 7.27 (Gowalla) and 5.52 vs 2.78 (Yelp).
  - Resolution sweep: the gain falls from 5.75 to 4.99 as resolution rises (Spearman −0.90).
  - LightGCN with GraphHash: raw DAU < BPR (11.29 vs 16.22; 4.71 vs 6.96). Corrected DAU wins (17.06 vs 16.27; 9.12 vs 6.96).
  - LightGCN Full Gowalla: BPR is ahead on the full table, so the flip premise is MF-specific.
- **RQ6. What does it cost?** (Tab 6)
  - Storage: float32 prior adds 5.52% (Gowalla), 3.90% (Yelp), 7.30% (AmazonBook) to the table. uint8 does not hold universally.
  - Tail recall is below Full DAU raw on all 3 datasets.
  - Catalog coverage, Gowalla: 95.2 → 28.4%.
  - Weighted recall: the ordering is kept.
  - Cheng reports the same coverage drop on full tables (0.985 → 0.726).

### 7. Limitations and failed predictions
- The flip scope (F3).
- Tail and coverage cost: this is a head-accuracy correction.
- Single-seed rows (†).
- Capped LightGCN rows.
- Failed predictions, listed with numbers (from App. G).
- C3 (learned variant): report it only after code review.
- Not tested: GDE, BACO itself, CTR models, DNA and MEmCom baselines.

### 8. Conclusion
- Answer the title in one line.
- Practical takeaway: compare losses under sharing only after restoring item counts.
- Future work: popularity-aware bucketing.

---

## 4. Figures and tables (main text)

| ID | Shows | Data |
|---|---|---|
| Fig 1 | 2×2 bars, Gowalla MF, raw + corrected | CSV, s0 runs (GraphHash BPR = s0-p500) |
| Fig 2a | Mechanism schematic | — |
| Fig 2b | log c_i vs ‖q_i‖ scatter, Full vs GraphHash, with bucket plateaus | checkpoints + stored mappings (GPU-only mappings; confirm they exist) |
| Fig 3 | Cosine ablation + prior gain, Full vs GraphHash, 3 datasets | CSV |
| Fig 4 | Shapley μ/ε bars + shuffle bar | PR-DEC |
| Fig 5 | Gain vs Louvain resolution | S4-C1 |
| Tab 1 | Dataset stats, buckets, table ratio, prior bytes | data + E7 byte table |
| Tab 2 | Main grid: dataset × compression × loss; raw/cos/corrected; ρ | CSV |
| Tab 3 | 9 seed pairs: Δ, CI, Holm p | PR-E8 + S4-SEED |
| Tab 4 | Full-table interventions | PR-E2e + S4-C4 |
| Tab 5 | LightGCN + DoubleGraph | S4-LGN, S4-DG, S5 |
| Tab 6 | Costs | E5, E7, E9 |

Use one colour per loss throughout. Solid fill = raw, hatched = corrected.

---

## 5. Words

| Use | Avoid |
|---|---|
| diagnosis, probe, test, restore item-level popularity | novel method |
| removes DirectAU's advantage | reverses the losses (as a universal claim) |
| head-accuracy correction with a tail cost | free, tail-safe, fixes |
| float32 prior, +3.9–7.3% | one byte per item |
| Cheng (2026, eq. 53) score form | anything implying the formula is ours |

---

## 6. Reviewer objections → where answered

| Objection | Answer | Where |
|---|---|---|
| "It's just MostPop" | Popularity-only scores 2.97 vs 13.29 | RQ3 |
| "The prior helps Full too" | Interaction plus cosine ablation | RQ2, Fig 3 |
| "Why not learn the bias (DNA, MEmCom)?" | Zero-gradient corollary; this is a training-free probe | §4, §7 |
| "The flip isn't general" | Scoped to 2/3 MF datasets | RQ1, §7 |
| "Single seeds" | 9 seed pairs for the main claim; † elsewhere | Tab 3, App. B |
| "Capped runs" | Flagged; excluded from convergence claims | §5, App. B |
| "Random is degenerate" | Treated as a stress case; ratio reported | Tab 1 |
| "Tail cost" | Reported; Cheng finds the same | RQ6 |
| "The bound fails for DoubleFreq" | Sufficient, not necessary | §4, App. D |

---

# APPENDIX PLAN (detailed: every run, analysis and diagnosis)

Rule: each appendix table names its run_ids and its source file. Nothing appears in the main text that is not traceable here.

## A. Reproduction and setup fidelity
- **A.1 GraphHash reproduction.**
  - Gowalla GraphHash BPR: 9.34 (`gowalla__mf__graphhash__bpr__s0`, legacy, patience) vs 9.30 in the paper.
  - Reproduction / sanity rows:
    - `x__gowalla__mf__graphhash__directau__s0` (2-epoch smoke test);
    - `x__yelp2018__mf__graphhash__bpr__s0`;
    - `x__coloc_a` / `x__coloc_b` (co-location determinism, identical outputs);
    - `x__..._directau_dedup__s0/1/2` (7.53 / 7.65 / 7.58).
- **A.2 Lineage.**
  - Legacy runs (Stage 1, fixed 30 epochs or time-capped) vs modern runs (patience).
  - The legacy 30-epoch GraphHash DAU rows (`*-legacy30`, s0/s1/s2, Random, DF) and how the modern runs supersede them.
  - Legacy val curve flat from epoch ~20 (7.40 → 7.48), so the legacy numbers were already near convergence.
- **A.3 Initialization controls.**
  - I1, `gowalla__mf__graphhash__directau__s0-initmatch`: raw 7.54, cos + prior 13.35 vs A1 7.55 / 13.30. Insensitive to a 2.64× initial-norm difference.
  - I2, `amazonbook__mf__graphhash__directau__s0-initmatch`: raw 4.59, cos + prior 6.30 vs B9 4.63 / 6.33.
  - Controlled pairs (same protocol):
    - Gowalla Full 18.02 vs GraphHash 7.54;
    - AmazonBook Full 9.57 vs GraphHash 4.63.
- **A.4 Hyperparameters.** Per loss and dataset:
  - lr and wd: GraphHash authors' grid, lr {1e-2, 5e-3, 1e-3}, wd {1e-4, 1e-6, 1e-8};
  - batch size: full-batch BPR; DirectAU and LightGCN at 1024;
  - patience, min-delta, eval interval;
  - β grids per mode, noting where β hit the grid edge (e.g. raw + prior β = 5.0 on Gowalla GraphHash DAU).
- **A.5 Compute.**
  - Kaggle T4.
  - Measured runtimes:
    - DirectAU MF: 37–39 s/epoch (Gowalla), 112–119 s/epoch (AmazonBook);
    - full-batch BPR: 0.22 / 0.41 s/epoch;
    - LightGCN Full: 0.70 (BPR) / 1.33 (DAU) min/epoch.
  - Peak memory 380 MiB.
  - Two-GPU row-split: no speedup. One line; this is useful to others.
- **A.6 Dataset statistics.**
  - Users / items: Gowalla 29,858 / 40,981; Yelp 31,668 / 38,048; AmazonBook 52,643 / 91,599.
  - Interactions, density, splits.
  - GraphHash bucket counts. Yelp: 7,604 user + 7,656 item buckets.
  - Table rows and bytes.
- **A.7 Evaluation details.**
  - Masking of train and validation items.
  - Recall denominator min(20, |T_u|).
  - Evaluated users: Gowalla 25,789; Yelp 30,304; AmazonBook 50,407.
  - Tie policy (`torch.topk` default; alternatives in E.3).

## B. Complete run ledger (every training run)
Columns: run_id | dataset | backbone | sharing | loss | seed | stop reason | best / run epochs | train min | raw | cos | cos + prior (β) | raw + prior (β) | ρ_item | ρ_bucket | status.

- **B.1 MF Full.**
  - `gowalla__mf__full__directau__s0`: patience 160/170; raw 18.02; cos 13.06; cos + prior 19.57 (β 0.1); ρ 0.81.
  - `gowalla__mf__full__bpr__s0-long`: patience 2790/3290; raw 16.55; raw + prior 17.06 (β 0.5); ρ 0.77.
  - `gowalla__mf__full__bpr__s0`: capped at max epochs; same value, so descriptive only.
  - `yelp2018__mf__full__{directau,bpr}__s0`: 11.10 / 8.02; ρ 0.91 / 0.97.
  - `amazonbook__mf__full__directau__s0`: 9.57; ρ 0.79.
  - `amazonbook__mf__full__bpr__s0`: capped at 3000 epochs.
  - `amazonbook__mf__full__bpr__s0-long`: 8.38, patience at epoch 3800.
- **B.2 MF GraphHash, main pairs (3 seeds each).**
  - Gowalla DAU s0/s1/s2: raw 7.55 / 7.61 / 7.59; cos + prior 13.30 / 13.32 / 13.35.
  - Gowalla BPR s0-p500/s1/s2: raw 9.37 / 9.39 / 9.41; raw + prior 11.31 / 11.19 / 11.27.
  - Yelp DAU and BPR s0/s1/s2.
  - AmazonBook DAU s0/s1/s2 and BPR (legacy s0, s1, s2).
  - Report mean ± SD. Gowalla: DAU 13.325 ± 0.024, BPR 11.257 ± 0.063.
- **B.3 MF Random and DoubleFreq (single seed, †).** All 12 cells, with β and ρ (Stage 3 "B main grid" table).
- **B.4 MF DoubleGraph (†).**
  - Gowalla DAU raw 3.54, cos 5.84, cos + prior 10.91; BPR raw + prior 7.27.
  - Yelp DAU 1.12 / 2.17 / 5.52; BPR 2.78.
  - ρ ≈ 0.
- **B.5 Resolution sweep (Gowalla GraphHash, †).** r ∈ {200, 400, 800, 1600 (capped), 3200} × {DAU, BPR}: the full S4-C1 table.
- **B.6 LightGCN.**
  - Stage 4: 8 runs; 5 capped at 4 h (S4-LGN table).
  - Stage 5, Gowalla Full, both converged:
    - BPR 18.48 / 19.04 / 18.46;
    - DAU 18.02 / 19.76 / 18.02.
  - Mark which rows are capped.
- **B.7 Intervention and variant runs.**
  - `*_eqnorm` (Full and GraphHash).
  - `gowalla__mf__full__directau__s0-wd0`: raw 16.12; cos + prior 19.08; ρ 0.74.
  - `s0-wd1e-4`: collapse, raw 1.41.
  - `gowalla__mf__graphhash__directau_bias__s0`: learned item bias, legacy, raw 7.38; cos + prior 13.17.
  - `*_resbias` × 3 (C3; see G.2).
- **B.8 Baselines.** Popularity-only on 3 datasets: R@20 2.97 / 1.68 / 0.96; NDCG 1.70 / 1.05 / 0.61.
- **B.9 Excluded runs, with reasons.**
  - GraphHash BPR at batch 1024 (5.31 / 5.45): wrong setup.
  - CPU-image β sweep: CPU bucket numbering differs.
  - First GPU attempt: Drive HTTP 500.
  - AmazonBook BPR with patience 100 (3.93): stopped on an early spike.
  - The failed 2-GPU attempts.
- **B.10 Stop-status summary.**
  - Counts per stage: patience / capped / fixed-epoch.
  - "Capped = inconclusive" rule restated.

## C. Diagnosis I: where popularity lives
- **C.1 Norm–popularity correlations.**
  - ρ_item, ρ_bucket and ρ_dir for every run in B.
  - Highlights:
    - GraphHash DAU: ρ_bucket 0.924 / 0.944 / 0.745, ρ_item 0.275 / 0.519 / 0.189;
    - Full DAU: ρ_item 0.81 / 0.91 / 0.79.
- **C.2 Item-quintile mean lengths (Q1 → Q5).**
  - Gowalla GraphHash DAU: 11.71 → 12.71 (flat).
  - Yelp Full DAU: 6.95 → 11.40.
  - Yelp GraphHash DAU: 1.89 → 2.52.
- **C.3 Norm dynamics (H-Adam).**
  - DirectAU lengths only grow: largest drop between evaluations is −0.15%.
  - Contrast: Full BPR drops 19% (not scale-invariant).
  - This supports Euler / scale invariance.
- **C.4 Direction vs length (H-dir).** ρ_dir is higher for BPR than DAU:
  - Yelp GraphHash 0.352 vs 0.164;
  - Yelp Full 0.375 vs 0.137.
- **C.5 Cosine-ablation table.** Δ(cos − raw) and Δ(prior) for all runs.
  - Under sharing, BPR still loses from cosine (9.37 → 5.86). BPR's length carries something other than popularity; discuss.
- **C.6 Weight decay (D7).**
  - wd = 0 keeps ρ 0.74: the link persists without decay in our setup. This is not a contradiction of PRISM.
  - wd = 1e-4 collapses.

## D. Diagnosis II: theory checks
- **D.1 η² table.**
  - GraphHash: 0.359 (Gowalla), 0.484 (Yelp), 0.358 (AmazonBook).
  - Random: 0.142 / 0.199 / 0.109.
  - DoubleFreq: 0.999.
  - Gowalla resolutions 400 / 800 / 1600 / 3200: 0.49 / 0.63 / 0.79 / 0.92.
- **D.2 Bound check.** ρ_item² ≤ η² everywhere: 0.073 ≤ 0.36 (Gowalla), 0.27 ≤ 0.48 (Yelp), 0.036 ≤ 0.36 (AmazonBook).
- **D.3 ρ_proxy and the product rule (exploratory, mixed).**
  - Gowalla: 0.154 vs 0.27, fail.
  - AmazonBook: 0.179 vs 0.189, ok.
  - Random: 0.225 vs 0.274, ok.
- **D.4 DoubleFreq discussion.**
  - Only 79 items share an alias pair (40,902 groups), so the decomposition is degenerate.
  - Yet ρ_item is 0.03: length loses popularity without identity loss.
- **D.5 Yelp exception (Y3).** Higher η² (0.48) means less is erased. This matches the bound.

## E. Diagnosis III: the correction test in detail
- **E.1 Ties (PR-TIE).**
  - Boundary-tie users: 87% (Gowalla), 89% (AmazonBook); tied group 14.7 / 17.6 items.
  - Key choice changes results by +0.04 to +0.06.
  - Tie-break-only share of the gain:
    - DAU: 12% (Gowalla), 25% (AmazonBook);
    - BPR: 34% (Gowalla), 60% (AmazonBook).
- **E.2 Fixed-β decomposition (PR-DEC), every checkpoint.**

  | Checkpoint | Δ_μ | Δ_ε | Δ_both | Interaction / shares |
  |---|---|---|---|---|
  | Gowalla DAU | +1.74 | +4.22 | +5.79 | interaction −0.17 |
  | AmazonBook DAU | | | | μ 0.24 / ε 0.76 |
  | Gowalla BPR | | | | μ −0.52 / ε 1.52 |
  | AmazonBook BPR | | | | μ −0.26 / ε 1.26 |
  | Random DAU | | | | μ 0.04 / ε 0.96 |

  - Within-bucket shuffle: −0.36 (Gowalla), −0.17 (AmazonBook).
  - Also give the retuned-β version.
- **E.3 Tie-policy robustness, AmazonBook.** Index, popularity key, topk and 10 random seeds: Δ +1.508 to +1.524, every CI ⊂ [1.39, 1.63].
- **E.4 PR-E8, full tables.**
  - Recall and NDCG for all 9 pairs.
  - NDCG Δ: +1.62 / +1.66 / +1.69 (Gowalla), +0.55 (Yelp s0), +1.18 (AmazonBook s0).
  - State the bootstrap method, the t-test and the Holm family.
- **E.5 β selection.** Validation curves per run, the selected β, and grid-edge cases.
- **E.6 Activity quartiles.** Gains are spread across users. Gowalla DAU Q1–Q4: +6.38 / +5.46 / +5.76 / +5.35.

## F. Costs
- **F.1 Tail and coverage (E5), every variant.**
  - All 7 GraphHash DAU variants vs the Full raw reference, with paired CIs:
    - Gowalla: 2.005 vs 2.501, Δ −0.50 [−0.68, −0.33];
    - Yelp: 0.290 vs 1.030;
    - AmazonBook: 3.781 vs 5.407.
  - Head share of added hits: 89–98%.
  - Coverage before → after:
    - Gowalla 95.2 → 28.4;
    - Yelp 96.3 → 21.7;
    - AmazonBook 93.1 → 80.7.
- **F.2 Storage (E7).**
  - Byte table:
    - Gowalla: 163,924 B, 5.52%;
    - Yelp: 152,192 B, 3.90%;
    - AmazonBook: 366,396 B, 7.30%.
  - uint8 + scale option.
  - Per-checkpoint uint8 Δ: 8 pass, 3 fail, 1 missing → no "one byte" claim.
- **F.3 Weighted recall (E9).**
  - Original β and weighted-retuned β for every pair.
  - Retuned Δ: +2.24 / +2.34 / +2.14 (Gowalla), +0.82 (Yelp), and AmazonBook 5.53 vs 4.04.

## G. Negative and failed results (all of them, with numbers)
- **G.1 Failed or partial predictions.**
  - PR-A8: Full BPR 16.55 outside [15.4, 15.8]. Our Full BPR is stronger than published.
  - Y3: Yelp |raw − cos| = 0.44 and ρ = 0.52.
  - PR-E5 pilot: the ≤ 0.5 tail-change prediction failed.
  - E7: uint8 not universal.
  - E2e:
    - decile shuffle −4.19 / −0.33 vs predicted |Δ| ≤ 0.5 / 0.3;
    - replacement +0.83 / +1.08, outside the predicted band but in the favourable direction.
  - C1: ρ_item vs resolution, Spearman +0.70 < +0.8.
  - M-iii: product rule mixed.
  - B-i: 6 of 8 cells. B-iv: 6 of 9 cells.
  - E9: AmazonBook inconclusive at Stage 3, closed in Stage 4.
- **G.2 C3 learned variant (cos + β̂·p + r).**
  - 4.56 vs 13.21; 2.68 vs 5.68; 1.63 vs 6.25.
  - β̂ ≈ 0.9 on all three datasets.
  - Report only after code review. Otherwise write "did not work in our implementation".
- **G.3 Learned item bias, legacy.** `directau_bias`: raw 7.38. A learned bias does not restore raw ranking.

## H. Proofs
- Lemma 1, Lemma 2, Proposition 1 with the corollary (MEmCom form U[i mod m] ⊙ V[i]), and Proposition 2 with the decomposition.
- The Euler-identity argument for monotone norm growth.

## I. Pre-registration record
- **Process:**
  - predictions registered before each stage;
  - snapshot + sha256;
  - test scored once;
  - labels P / I / H / X defined.
- **Registry:**
  - every prediction ID → prediction → observed → verdict, taken from the RESULTS §0 board;
  - the tally of PASS / FAIL / partial / inconclusive.
- **Snapshots:** list the dated hashes (anonymized supplement).

## J. Novelty and related-work verification (optional, short)
- What was checked: GraphHash and PRISM citers, Cheng eq. 53, the MEmCom algorithm, DNA, CoRRe, GDE.
- The kill rule was not triggered.
- This shows reviewers the overlap was checked deliberately.

---

## 7. Writing order

| Day | Work |
|---|---|
| Thu 22 | §3–§4 theory; Tab 2 numbers; Appendix B ledger (copy from CSV) |
| Fri 23 | §5–§6 results. Numbers freeze at 18:00 |
| Sat 24 | §1 intro, §2 related work, §7 limitations; Appendix C–G |
| Sun 25 | Abstract, conclusion; Appendix H–J; full read against §0 and §6; trim to 8 pages |
| Mon 26 | Anonymization pass (no repo links, no account names); submit by 15:00 |

Write the introduction after the results section.
