# Paper plan v4: If Sharing Saves Memory, What Does It Cost? Popularity Without Identity

- **Venue:** WebConf 2027, track "Search, Recommendation, and Retrieval-Augmented AI".
- **Format:** `acmart` with `sigconf, anonymous, review`.
- **Length:** 8 main pages, 12 pages total including references and appendix. The first 8 pages must stand alone.
- **Deadline:** Mon 26 Oct, 17:29 IST.
- Every number below was checked against `results/master_metrics_v2.csv` and RESULTS.md on 22 Oct. Re-check each one by its run_id before it goes into the paper.

---

## Suggested changes to the section list (Pushkar decides)

| # | Change | Why |
|---|---|---|
| S1 | **Put limitations only in §7. The conclusion gets future work, no limitations.** | Your list has limitations twice. Repeating hedges costs about 0.2 of the 8 pages, and reviewers read repeated hedging as padding. One place, stated once, reads as honest. |
| S2 | **Keep Related Work at §2, but show the symptom in the introduction (Fig 1).** | In a diagnosis paper the reader should see the failure on page 1. With Fig 1 in the intro, related work at §2 costs nothing. The alternative, moving related work after the results, is not needed. |
| S3 | **Rename "Methodology" to "Diagnostic Methodology".** | In a recsys paper, "Methodology" signals a new model. A reviewer then treats the count-restored score as *the method* and rejects it as known logit adjustment. The name tells them the method is the diagnosis. |
| S4 | **Put the theory in §4, not in Preliminaries.** | Preliminaries should hold only definitions and the observed failure (the problem). The lemmas and propositions *are* the diagnosis, so they belong in the methodology. |
| S5 | **List the RQs at the end of §4 and title the §6 subsections by RQ.** | Each method claim then has a named result. Reviewers can check the claims against the evidence directly. |

The final structure below already applies S1–S5. If you reject any of them, only the names change; the content map still holds.

---

## Structure and page budget

| § | Section | Pages |
|---|---|---|
| — | Title, Abstract, CCS concepts, keywords | 0.4 |
| 1 | Introduction | 1.1 |
| 2 | Related Work | 0.65 |
| 3 | Preliminaries and Problem Diagnosis | 0.85 |
| 4 | Diagnostic Methodology and Research Questions | 1.5 |
| 5 | Experimental Setup | 0.5 |
| 6 | Results and Analysis | 2.2 |
| 7 | Trade-offs, Robustness, and Limitations | 0.6 |
| 8 | Conclusion | 0.2 |
| | **Main text** | **8.0** |
| — | References (about 45) | about 1.2 |
| A–E | Appendix | about 2.8 |
| | **Total** | **12.0** |

---

## Abstract

> Embedding sharing cuts the memory of Web-scale recommenders by mapping many items to one embedding row, but it also changes what the embedding can encode. Prior work shows that on full tables embedding magnitude encodes item popularity, and that DirectAU performs poorly under hashing; why remains unexplained. We show that sharing pools magnitude at the bucket level: shared-row norms track bucket traffic (Spearman 0.75–0.94) far more than item popularity (0.19–0.52). We prove that items in a bucket tie exactly, that under GraphHash any bucket-level quantity explains at most 36–48% of log-popularity variance, and that a learned per-item scale receives zero gradient under DirectAU. Consistent with this, discarding magnitude costs compressed models at most 0.44 points of Recall@20, against up to 4.96 for full tables. On Gowalla, DirectAU leads BPR with full tables (18.02% vs 16.55%) but trails under GraphHash (7.55% vs 9.37%). Replacing pooled magnitude with a validation-tuned log-count prior, a published score form applied without retraining, raises compressed DirectAU to 13.30%; 71–76% of the gain is item-specific. Equally corrected, DirectAU exceeds BPR on all nine dataset–seed pairs (+0.83 to +2.13 points). The prior adds 3.9–7.3% to table storage and lowers tail recall and catalog coverage.

- **CCS concepts:** Information systems → Recommender systems; Collaborative filtering.
- **Keywords:** embedding table compression, parameter sharing, popularity, embedding magnitude, collaborative filtering, DirectAU.

---

## 1. Introduction (1.1 pages)

**Purpose:** pose the title's question, show the symptom, state the insight, and list the contributions. Web relevance goes in paragraph 1; the CFP desk-rejects papers that don't state it on page 1.

1. **Web problem.**
   - ID embedding tables dominate memory in Web-scale recommenders.
   - Sharing schemes cut that memory: the hashing trick, frequency hashing, GraphHash (WWW'25), BACO and GDE (SIGIR'26).
   - These schemes are judged by recall per byte, under whatever loss the deployment already uses.
2. **Closest answers.**
   - GraphHash: graph-clustered buckets beat random hashing.
   - PRISM (KDD'25): on full tables, magnitude stores popularity.
3. **Open question.**
   - GraphHash and Shiao et al. (WWW'25 Companion) report that DirectAU fails under hashing, with no explanation.
   - Our numbers: 18.02 > 16.55 on the full table, but 7.55 < 9.37 under GraphHash (Fig 1a).
   - Credit sentence: the score form is Cheng (2026, eq. 53); per-item parameters on shared tables already exist (MEmCom, Hash Embeddings).
4. **Why the obvious fixes fail.**
   - More buckets: less compression.
   - A learned per-item scale: zero gradient under DirectAU.
   - Better tie-breaking: only 12% of the gain.
5. **Insight.** Sharing turns magnitude into a property of the bucket (Fig 1b), so the item-specific part of popularity cannot be represented.
6. **What we do.** Theory → nine competing explanations, each tested → the intervention the diagnosis predicts → its costs.
7. **Contributions (3 bullets, each stating new knowledge):**
   - **Diagnosis:** sharing pools magnitude at the bucket level, and magnitude stops being useful for ranking.
   - **Analysis and tests:** exact ties, a bound on representable popularity, zero gradient for learned scales; capacity, ties, initialisation, stopping, weight decay and MostPop are ruled out.
   - **Intervention and costs:** restoring item counts reverses the ordering on 9 of 9 pairs; 71–76% of the gain is item-specific; there is a tail and coverage cost.
8. **Scope sentence:** "We do not claim a new score form, a tail-safe repair, or a universal loss reversal."

**Fig 1 (page 1).**
- (a) Gowalla MF bars: Full / GraphHash × BPR / DirectAU.
  - Raw: 16.55 / 18.02 / 9.37 / 7.55.
  - Corrected: 11.31 / 13.30, hatched.
- (b) Scatter of log(1 + c_i) against ‖q_i‖ for Full DirectAU (ρ 0.81) and GraphHash DirectAU (ρ 0.28), with bucket plateaus visible.
- Caption: "Sharing removes DirectAU's lead because shared magnitude no longer follows the item."

---

## 2. Related Work (0.65 pages)

Three paragraphs. Each ends with what the cited work does not do.

1. **Embedding table compression.**
   - Hashing trick; frequency/double hashing; Learning to Collide; CAFE (SIGMOD'24); MEC; CCE; GraphHash; BACO; GDE.
   - Gap: they optimise the bucket assignment. None asks what sharing does to magnitude or to the popularity signal.
2. **Magnitude, popularity, and losses.**
   - PRISM; Adap-τ (WWW'23); DDC (KDD'25); DirectAU (KDD'22); Wu et al. (TOIS'24); DNA (ASOC'26); CoRRe.
   - Gap: all on full tables.
3. **Per-item terms and score-time popularity.**
   - MEmCom (MLSys'22); Hash Embeddings (NeurIPS'17); Seol et al. (WSDM'24); logit adjustment (ICLR'21); logQ (RecSys'19); PDA (SIGIR'21); MACR (KDD'21); Cheng (2026, eq. 53).
   - Gap: none links these to what sharing erases, or explains why a learned scale cannot help DirectAU.

---

## 3. Preliminaries and Problem Diagnosis (0.85 pages)

**Purpose:** define everything, then state the failure as the problem the paper explains. No theory here.

- **3.1 Notation.**
  - Users U, items I, train counts c_i, p_i = log(1 + c_i).
  - Score s(u,i) = ⟨p_u, q_i⟩.
  - Sharing map b: I → B with q_i = E[b(i)]. Double hashing adds two rows.
  - Bucket traffic C_b = Σ_{i∈b} c_i.
- **3.2 Sharing schemes.** GraphHash (Louvain, resolution 200), Random, DoubleFreq, DoubleGraph. Table size relative to Full: 16.4% / 21.9% / 13.6%.
- **3.3 Losses.**
  - BPR: pairwise logistic.
  - DirectAU: alignment + uniformity on ℓ2-normalised vectors.
- **3.4 Scoring modes.**
  - raw ⟨p, q⟩;
  - cosine;
  - **count-restored score**: cos(p_u, q_i) + β·p_i, the form of Cheng eq. 53;
  - raw + β·p_i.
- **3.5 Diagnostic quantities.**
  - ρ_item = Spearman(‖q_i‖, c_i).
  - ρ_bucket = Spearman(‖E_b‖, C_b).
  - η² = Var(μ_b) / Var(p), with μ_b the bucket mean of p.
- **3.6 The observed failure (problem statement).**
  - Short table, raw Recall@20, MF:

    | Dataset | Full: DAU vs BPR | GraphHash: DAU vs BPR |
    |---|---|---|
    | Gowalla | 18.02 > 16.55 | 7.55 < 9.37 |
    | Yelp | 11.10 > 8.02 | 2.41 < 3.78 |
    | AmazonBook | 9.57 > 8.38 | 4.63 > 4.10 (lead shrinks from +1.19 to +0.53) |

  - Problem: *why does sharing remove DirectAU's advantage, and what does sharing erase?*
  - Boundary, stated here: with LightGCN, BPR already leads on the full Gowalla table (18.48 vs 18.02), so the premise is MF-specific.

---

## 4. Diagnostic Methodology and Research Questions (1.5 pages)

**Purpose:** the theory, the hypotheses, and how each hypothesis is tested. Results go in §6.

- **4.1 Theory** (statements here, proofs in App. A).
  - **Lemma 1.** A user's ranking does not depend on ‖p_u‖, so only the item side can carry popularity.
  - **Lemma 2.** Items in the same bucket get identical scores under every mode that depends only on q.
  - **Proposition 1.** The DirectAU loss depends only on q/‖q‖.
    - Corollary: a learned per-item positive scale (MEmCom's U[i mod m] ⊙ V[i]) has ∂L/∂w_i = 0.
    - Magnitude is therefore set by optimiser dynamics, which under sharing aggregate over the bucket. Euler's identity implies the norms only grow.
  - **Proposition 2.** With p = μ_b + ε, every bucket-level f satisfies corr(f(b), p)² ≤ η².
    - ε is orthogonal to every representation built from shared rows.
    - Remark: this condition is sufficient, not necessary (DoubleFreq).
  - **Fig 3:** conceptual schematic, labelled conceptual. Full table: one arrow per item. Shared table: one arrow per bucket, with item count bars beneath. Annotate p = μ_b + ε.
- **4.2 Main hypothesis (H_pool).** Under sharing, magnitude tracks bucket traffic, not items, so the item-level ε is lost.
  - Predictions:
    - (i) ρ_bucket ≫ ρ_item;
    - (ii) cosine ≈ raw under sharing, but cosine ≪ raw on full tables;
    - (iii) restoring p recovers a gain that is mostly ε-driven.
- **4.3 Competing explanations and the test for each** (the test plan; outcomes are in §6.3):

  | Alternative | Test designed |
  |---|---|
  | A1 Lost capacity | Erase magnitude on the **full** table (equal norms); capacity stays intact |
  | A2 Ties only (H_tie) | Popularity tie-break without a prior |
  | A3 Bucket-level popularity suffices | Fixed-β μ/ε decomposition (Shapley); shuffle counts within buckets |
  | A4 Magnitude carries non-popularity information | On the full table: replace magnitude by a fitted function of p; global and decile shuffles |
  | A5 Undertraining | Patience-stopped runs only; capped runs flagged |
  | A6 Initialisation | Init-matched reruns |
  | A7 Weight decay | wd = 0 run |
  | A8 MostPop | Popularity-only ranking |
  | A9 A learned scale would fix it | Prop 1 (theory) |

- **4.4 The intervention as a test.**
  - If H_pool holds, replacing pooled magnitude with item counts should restore DirectAU above BPR.
  - Both losses get the same treatment: each is scored with the mode and β chosen on validation.
  - This is a probe, not a contribution.
- **4.5 Research questions.**
  - RQ1: Does sharing remove DirectAU's advantage? (§3.6 gives the motivation; §6.1 has the full grid.)
  - RQ2: Does magnitude follow buckets instead of items?
  - RQ3: Do the alternatives A1–A9 explain the drop?
  - RQ4: Does restoring item counts reverse the ordering, and is the gain item-specific?
  - RQ5: Does it hold for other sharing schemes, other resolutions, and LightGCN?
  - Costs and robustness go in §7.

---

## 5. Experimental Setup (0.5 pages)

- **Datasets.** Gowalla, Yelp2018, AmazonBook, with the GraphHash splits.
  - Users / items: 29,858 / 40,981; 31,668 / 38,048; 52,643 / 91,599.
  - Tab 1: interactions, buckets, table bytes, prior bytes.
- **Models.** MF with d = 64; 2-layer LightGCN.
- **Training.**
  - Each loss uses the GraphHash authors' lr/wd grid.
  - Full-batch BPR; DirectAU at batch 1024.
  - Early stopping by patience. Converged means stopped by patience; capped runs are flagged and never support a convergence claim.
- **Selection.**
  - β and the scoring mode are chosen on validation from a fixed grid.
  - Test is scored once.
- **Metrics.** Recall@20 with denominator min(20, |T_u|) (primary); NDCG@20 (secondary).
- **Statistics.**
  - Paired user bootstrap, 2,000 resamples; paired t-test with Holm correction.
  - 3 seeds on the main pairs. Single-seed rows are marked †.
- **Pre-registration.** Predictions were registered before each stage. The tally goes here; the registry goes in the anonymous repository.
- **Compute.** Kaggle T4; full details in App. B.
- **Reproduction.** GraphHash BPR gives 9.34 vs 9.30 published. One sentence.

---

## 6. Results and Analysis (2.2 pages)

Write each paragraph as: question → contrast → observation → interpretation → boundary.

- **6.1 RQ1: sharing removes DirectAU's advantage.**
  - Tab 2, the main grid: dataset × {Full, GraphHash, Random, DoubleFreq, DoubleGraph} × {BPR, DAU}, with columns raw / cos / corrected (β) / ρ_item / ρ_bucket.
  - Under Random and DoubleFreq the order reverses in 5 of 6 cells.
  - Random is a stress case: raw recall is below 1%.
- **6.2 RQ2: magnitude follows buckets.**
  - ρ_bucket is 0.924 / 0.944 / 0.745; ρ_item is 0.275 / 0.519 / 0.189. On the full table ρ_item is 0.81 / 0.91 / 0.79.
  - Quintile lengths under GraphHash: 11.71 → 12.71, nearly flat.
  - η² is 0.36 / 0.48 / 0.36. The bound holds: ρ_item² is 0.073 / 0.27 / 0.036.
  - Fig 2 shows the cosine ablation (cos − raw):
    - Full: −4.96 / −4.13 / −0.87.
    - Shared: +0.02 / −0.44 / +0.01.
    - Prior gain on Gowalla: Full +1.55, shared +5.75. This difference-in-differences answers "the prior helps everything".
  - Yelp has the highest ρ_item (0.52) and the highest η² (0.48): less is erased there. This is consistent with the bound; report it as the failed Y3 prediction.
- **6.3 RQ3: competing explanations.** Tab 3 gives the outcomes from the §4.3 plan.

  | Alternative | Result | Verdict |
  |---|---|---|
  | A1 Lost capacity | Full table with equal norms: 18.02 → 12.77; the prior then gives 19.32 | Not capacity |
  | A2 Ties only | 12% of the gain (DAU), 34% (BPR) | Minor |
  | A3 Bucket-level popularity suffices | ε share 0.71 / 0.76; within-bucket shuffle keeps −6% of the gain | Item level is needed |
  | A4 Magnitude carries other information | Replacement 17.90 → 18.69 (AmazonBook 9.57 → 10.64); global shuffle 6.28; decile shuffle 13.64 (a failed prediction: coarse bins) | Magnitude ≈ popularity |
  | A5 Undertraining | All main runs patience-stopped | Ruled out |
  | A6 Initialisation | Δ ≤ 0.05 | Ruled out |
  | A7 Weight decay | wd = 0 keeps ρ at 0.74 | Not weight decay alone |
  | A8 MostPop | 2.97 / 1.68 / 0.96 vs 13.29 / 5.68 / 6.33 | Ruled out |
  | A9 Learned scale | Zero gradient | Excluded for DirectAU |

- **6.4 RQ4: restoring item counts.**
  - Tab 4 covers the 9 pairs: Δ, 95% CI, Holm p, NDCG Δ.
    - Gowalla: +1.99 / +2.13 / +2.08.
    - Yelp: +0.88 / +0.86 / +0.83.
    - AmazonBook: +1.52 / +1.48 / +1.54.
  - Gowalla mean ± SD: 13.325 ± 0.024 vs 11.257 ± 0.063.
  - The correction recovers 55% of the raw Full-vs-shared gap on Gowalla. It does not reach the full table.
  - The ε share in Fig 4 restates A3 as the size of the gain.
- **6.5 RQ5: generality.**
  - DoubleGraph: corrected DAU vs BPR 10.91 vs 7.27 (Gowalla), 5.52 vs 2.78 (Yelp); ρ ≈ 0.
  - Resolution sweep (Fig 5):
    - The gain falls from 5.75 to 4.99 (Spearman −0.90).
    - Corrected DAU beats BPR at every resolution.
    - ρ_item plateaus after resolution 800 (a failed trend prediction).
  - LightGCN + GraphHash:
    - Raw DAU < BPR: 11.29 vs 16.22; 4.71 vs 6.96.
    - Corrected DAU wins: 17.06 vs 16.27; 9.12 vs 6.96.
    - Capped rows are marked.

---

## 7. Trade-offs, Robustness, and Limitations (0.6 pages)

- **7.1 Trade-offs: what the correction costs.** Tab 5.
  - Storage: the float32 prior adds 5.52 / 3.90 / 7.30% to the compressed table. uint8 is not universal (3 of 11 checkpoints exceed 0.02 pp), so no one-byte claim.
  - Tail recall: below the Full DAU raw reference on all 3 datasets.
    - Gowalla 2.01 vs 2.50, CI [−0.68, −0.33].
    - Yelp 0.29 vs 1.03.
    - AmazonBook 3.78 vs 5.41.
  - 89–98% of the added hits are head items.
  - Catalog coverage falls: 95.2 → 28.4% / 96.3 → 21.7% / 93.1 → 80.7%.
  - Cheng reports the same drop on full tables (0.985 → 0.726), so the cost belongs to the prior, not to sharing.
  - Frame it as a head-accuracy correction with a tail cost.
- **7.2 Robustness of the main claim.**
  - Seeds: 9 of 9 pairs.
  - Tie policy (AmazonBook, 13 policies): +1.508 to +1.524.
  - Metric: NDCG agrees (+0.55 to +1.69).
  - Popularity-weighted recall with w = (1+c)^−½ keeps the ordering after re-tuning β: Gowalla +2.14 to +2.34, Yelp +0.82, AmazonBook 5.53 vs 4.04.
  - Initialisation and stopping: see A5 and A6.
- **7.3 Limitations** (each one changes how a result should be read).
  - The reversal is not universal: AmazonBook MF and the LightGCN full table.
  - Single-seed rows (†) and capped LightGCN rows.
  - Failed predictions: the Full-BPR range, Y3, uint8 storage, the decile shuffle, the ρ_item trend (full list in App. E).
  - The learned end-to-end variant (C3): report it only if the code review clears it.
  - Not tested: soft gating (GDE), BACO itself, CTR models, DNA and MEmCom baselines, online effects.

---

## 8. Conclusion (0.2 pages)

- Answer the title: sharing saves memory, and it costs item identity in the popularity signal.
- Practical rule: compare training losses under sharing only after restoring item counts.
- Future work: popularity-aware bucketing; extending the analysis to soft gating.
- No limitations here (S1).

---

## References (about 1.2 pages)
- Bibliography fixes from N1:
  - Park et al. (CIKM'23): correct title.
  - Wu et al.: TOIS 2024, not 2023.
  - Wan et al.: NeurIPS title.
  - Yi et al.: confirm the first author.
  - Shiao et al.: WWW'25 Companion.
  - DNA: ASOC 186:114226.
- Avoid self-citation (CFP rule).

---

## Appendix (about 2.8 pages)

| App. | Content | Pages |
|---|---|---|
| A | Proofs: Lemmas 1–2; Prop 1 and its corollary (MEmCom form); Prop 2 and the decomposition; the Euler argument for monotone norm growth (DirectAU's largest drop −0.15% vs BPR −19%) | 0.8 |
| B | Protocol details: lr/wd/batch/patience per loss; β grids and grid-edge cases (β = 5.0); evaluation masks; denominators; evaluated users (25,789 / 30,304 / 50,407); bootstrap and Holm details; compute and runtimes (2 GPUs gave no speedup) | 0.4 |
| C | **Complete run ledger**, one table of about 75 runs. Columns: run_id, stop reason, best/run epochs, raw, cos, corrected (β), ρ_item, ρ_bucket, flags (†, capped, legacy, excluded). Covers: reproduction rows; init controls; wd variants; equal-norm; resolution sweep; LightGCN stages 4–5; popularity-only; the 4 excluded runs with reasons | 1.0 |
| D | Extra diagnostics: η² per scheme and resolution (0.49 / 0.63 / 0.79 / 0.92); quintile lengths; norm dynamics; ρ_dir (BPR > DAU); tie statistics per checkpoint; μ/ε decomposition for every checkpoint; AmazonBook length interventions; gains by user-activity quartile | 0.4 |
| E | Failed predictions with numbers; pre-registration tally | 0.2 |

**Anonymous repository**, linked once in §5.
- Contents: full CSV, RESULTS blocks, PREREG with snapshot hashes, E5/E7/E9 JSON, code.
- Scrub it before upload:
  - no `runs/_sessions`, no `sessions_*.py`, no account or slug strings;
  - no git history;
  - grep for usernames.
- The CFP does not mention supplementary material; check the OpenReview form.

---

## Figures and tables

| ID | Section | Job | Status |
|---|---|---|---|
| Fig 1a/b | §1 | Symptom; magnitude follows the bucket | 1a ready; **1b: confirm per-item norm data before G4 (Fri 18:00)** |
| Fig 2 | §6.2 | Cosine ablation + prior gain (difference-in-differences) | ready |
| Fig 3 | §4.1 | Conceptual schematic | to draw |
| Fig 4 | §6.4 | μ/ε shares + shuffle | ready |
| Fig 5 | §6.5 | Gain vs resolution | ready |
| Tab 1 | §5 | Datasets, buckets, bytes | fill bucket counts |
| Tab 2 | §6.1 | Main grid | ready |
| Tab 3 | §6.3 | Competing explanations | ready |
| Tab 4 | §6.4 | 9 paired comparisons | ready |
| Tab 5 | §7.1 | Costs | ready |

If space runs out, merge Fig 4 into Tab 3 and Fig 5 into Tab 5.

---

## Wording

| Use | Avoid |
|---|---|
| embedding table compression, parameter sharing, bucket | "sharing rows" |
| magnitude (ℓ2-norm), direction | "length" in formal text |
| intra-bucket score ties; bucket-level magnitude pooling | invented acronyms |
| count-restored score; post hoc; without retraining | "our method", "novel correction" |
| removes DirectAU's advantage | a universal "reverses the losses" |
| head-accuracy correction with a tail cost | free, tail-safe, fixes |
| float32 prior, +3.9–7.3% | one byte per item |
| percentage points (pp) | bare % for differences |
| the Cheng (2026, eq. 53) score form | anything that implies the formula is ours |

---

## Writing order

| Day | Work |
|---|---|
| Thu 22 | §3, §4 |
| Fri 23 | §5, §6. Confirm the Fig 1b data. Numbers freeze at 18:00 |
| Sat 24 | §1 (written after §6), §2, §7; App. A–C |
| Sun 25 | Abstract, §8, App. D–E; check against the scope sentence; trim to 8 + 4 pages |
| Mon 26 | Scrubbed repository; PDF metadata anonymity check; submit by 15:00 |
