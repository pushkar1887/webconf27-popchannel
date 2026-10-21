# Paper outline: *If Sharing Saves Memory, What Does It Cost? Popularity Without Identity*
WebConf 2027 · ACM sigconf · anonymous · due **Mon 26 Oct 17:29 IST** · numbers freeze at G4 (**Fri 23 Oct 18:00**)

All numbers below come from `results/master_metrics_v2.csv` and the RESULTS.md blocks. **This is a writing plan, not a source of numbers.** Before a number goes in the paper, look it up again by run_id.

---

## 0. Problems in the submitted abstract that the paper has to handle

- **[FLAG] F1. The ρ ranges cover 2 of 3 datasets.**
  - The abstract gives ρ_bucket "0.93–0.94" and ρ_item "0.28–0.52". These are Gowalla (0.924 / 0.275) and Yelp (0.944 / 0.519) only.
  - AmazonBook is **0.745 / 0.189**.
  - Consequence: a reviewer who reads Table 2 sees a range that leaves out a dataset.
  - Fix: in the paper, give all three, **ρ_bucket 0.75–0.94 vs ρ_item 0.19–0.52**. The gap still holds on every dataset, so the claim gets stronger, not weaker.
- **[FLAG] F2. 13.30 is cosine + prior, not raw + prior.**
  - On Gowalla GraphHash DAU: raw + prior is 10.33; cos + prior is 13.30.
  - Consequence: a reviewer finds that "adding the prior" gives only +2.78, and that the headline uses a different score.
  - Fix: make this the point. Under sharing, the learned length carries **nothing usable**. Removing it costs nothing (raw 7.55 → cosine 7.57). The correct test is therefore to **replace** pooled length with item counts, not to add counts on top of it. Define it once, in §4: `s(u,i) = cos(p_u, q_i) + β·log(1+c_i)`.
- **[FLAG] F3. The loss "flip" does not hold everywhere.** On MF, raw Recall@20 is:

  | Dataset | Full BPR | Full DAU | GraphHash BPR | GraphHash DAU | Outcome |
  |---|---|---|---|---|---|
  | Gowalla | 16.55 | 18.02 | 9.37 | 7.55 | flips |
  | Yelp | 8.02 | 11.10 | 3.78 | 2.41 | flips |
  | AmazonBook | 8.38 | 9.57 | 4.10 | 4.63 | **does not flip**; DAU's lead shrinks from +1.19 to +0.53 |
  | LightGCN Full (Gowalla) | 18.48 | 18.02 | | | BPR already ahead on the full table, so no flip premise |

  - Fix: state the claim as "sharing **removes DirectAU's advantage**: it reverses the order on 2 of 3 MF datasets and shrinks it on the third". The flip is a symptom. The claim is the loss of popularity information.
- **[FLAG] F4. The prior also helps the full table.**
  - Full DAU Gowalla: cos + prior 19.57 vs raw 18.02.
  - Consequence: "your correction helps everywhere, so it says nothing about sharing."
  - Fix: always report the **interaction** (difference-in-differences), never the GraphHash gain alone. See Fig 3.
- **F5. Double-frequency hashing does not fit the bound.** Alias tuples are almost unique (η² ≈ 0.999), so the correlation-ratio bound says nothing there. Yet ρ_item is 0.027 and DAU < BPR (3.22 vs 3.93). State this: the bound is **sufficient, not necessary**. Length can lose popularity even when item identity survives, because the norm of a sum of shared rows follows bucket traffic. This turns a reviewer objection into a finding.
- **F6. Random hashing reaches < 1% raw Recall** (0.15–0.76). It is degenerate at this bucket count. Report the bucket/item ratio in Table 1 and call Random a stress case, not a main comparison.

---

## 1. Thesis and contributions (memorise these; every section serves them)

**Thesis, one sentence:** embedding sharing makes vector length a function of the bucket, not the item. Losses that store popularity in length (DirectAU) therefore lose item-level popularity. A training-free item-count test recovers most of the loss, and most of what it recovers is item-specific.

**Contributions** (exactly 3; no "novel method"):
1. **Diagnosis.**
   - Under sharing, norms track bucket traffic: ρ_bucket 0.75–0.94 vs ρ_item 0.19–0.52.
   - Learned length becomes useless for ranking: the cosine ablation changes Recall by ≤ 0.44 under sharing, against 0.87–4.96 on full tables.
2. **Analysis.**
   - Exact ties within a bucket.
   - A correlation-ratio bound on what any bucket-level quantity can encode (η² = 0.36–0.48 for GraphHash).
   - Scale invariance of DirectAU: a learned per-item scale, as in MEmCom, gets **zero gradient**.
3. **A controlled test, not a method.**
   - The published log-prior score form is applied post hoc (Cheng 2026, eq. 53; Menon et al. 2021).
   - Fixed-coefficient interventions and within-bucket shuffles show the gain is item-specific: Shapley ε share 0.71–0.76; shuffling keeps −6% of the gain.
   - Corrected DAU beats corrected BPR on **9 of 9 dataset × seed pairs**: Δ +0.83 to +2.13, every CI > 0, Holm p ≤ 6e-33.
   - Costs are reported in full: storage +3.9–7.3%, tail recall, coverage.

**Framing rule:** this is a **diagnosis paper**. The question in the title is answered in the intro: "what sharing costs is item identity in the popularity signal". The repair is a probe. Say early, once and plainly, that the score form is not ours. Reviewers who know Cheng, DNA and PRISM will respect this; hiding it gets you rejected.

---

## 2. Page budget (8 pages + references; check the CFP for whether appendices are allowed)

| § | Section | Pages |
|---|---|---|
| 1 | Introduction (with Fig 1) | 1.25 |
| 2 | Related work | 0.75 |
| 3 | Setup and notation | 0.5 |
| 4 | What sharing does to popularity: theory | 1.5 |
| 5 | Experimental protocol | 0.5 |
| 6 | Results (RQ1–RQ6) | 2.75 |
| 7 | Costs, limitations, failed predictions | 0.5 |
| 8 | Conclusion | 0.25 |
| App. | Proofs, full grids, single-seed rows, hyperparameters | unlimited, but reviewers may skip it |

The main text must stand on its own. Anything a reviewer needs in order to believe the claim goes in the 8 pages.

---

## 3. Section by section

### Abstract
- Keep the submitted structure: problem → prior work → recent work → gap → ours → numbers.
- Changes for the camera-ready version of the paper (the abstract may be edited until the paper deadline; check the CFP):
  - ρ ranges over 3 datasets (F1);
  - "reverses the loss ordering on two of three datasets" (F3);
  - mention "cosine plus log-count score" (F2).
- Keep one cost sentence. It is your honesty signal.

### 1. Introduction (1.25 pages)
**Purpose:** pose the title's question, show the symptom, name the mechanism, state the three contributions.

Paragraph plan:
1. **Hook.** Embedding tables dominate model memory. Sharing (hashing; graph-based bucketing such as GraphHash, WWW'25; BACO; GDE) cuts it. The field evaluates sharing by recall per byte.
2. **Symptom.** GraphHash and Shiao et al. (WWW'25 Companion) report that DirectAU does poorly under hashing, without an explanation. Our Gowalla numbers: DAU 18.02 > BPR 16.55 on the full table, but 7.55 < 9.37 under GraphHash. Point to **Fig 1**.
3. **Why it should happen.** PRISM (KDD'25) shows that for full tables popularity lives in embedding length. Under sharing, length belongs to the bucket. A bucket cannot hold the popularity of each of its items.
4. **What we do.** Theory (ties, bound, zero-gradient corollary), then a minimal test. Add back item-level popularity alone (the published log-prior form), and check that the recovered gain is item-specific, not bucket-level.
5. **Findings,** with numbers. Each one is a bullet that maps to an RQ.
6. **Contributions:** 3 bullets (§1 above), plus one sentence saying what is **not** claimed: no new score form, no tail-safe repair, and the flip is not universal.

**Fig 1 (page 1, top right):** a 2×2 grouped bar chart for Gowalla MF.
- Groups: {Full, GraphHash}. Bars: {BPR, DirectAU}.
- Raw bars are solid; corrected bars are hatched on top.
- Raw values: 16.55 / 18.02 / 9.37 / 7.55.
- Corrected values (GraphHash): BPR 11.31, DAU 13.30.
- Caption: "Sharing removes DirectAU's lead; restoring item counts brings it back."

### 2. Related work (0.75 pages): three paragraphs, each ending with "what they don't do"
1. **Embedding compression and sharing.**
   - Hashing trick; frequency and double hashing (Zhang et al. 2020); Learning to Collide; CAFE (SIGMOD'24); MEC.
   - GraphHash (WWW'25); BACO (SIGIR'26); GDE (SIGIR'26; soft gating, so our exact-tie result does not apply there).
   - *Gap:* all of them evaluate recall per byte; none asks what sharing does to popularity.
2. **Popularity in embedding length and losses.**
   - PRISM (KDD'25); Adap-τ (WWW'23); DDC (KDD'25); DirectAU (Wang et al., KDD'22); sampled-softmax analysis (Wu et al., TOIS'24); DNA (Applied Soft Computing'26; normalization + learned item bias, full tables); CoRRe (2026).
   - *Gap:* all on full tables.
3. **Per-item terms on shared tables, and score-time popularity.**
   - MEmCom (MLSys'22, a per-item scalar); Hash Embeddings (NeurIPS'17); Seol et al. (WSDM'24).
   - Logit adjustment (Menon et al., ICLR'21); logQ (Yi et al., RecSys'19); PDA (SIGIR'21); MACR (KDD'21); **Cheng (2026), eq. 53, our exact score form**.
   - *Gap:* nobody connects these terms to what sharing erases, or explains why a learned scale cannot fix DirectAU (our corollary).

### 3. Setup and notation (0.5 pages)
- Users U, items I, counts c_i, scores s(u,i) = ⟨p_u, q_i⟩.
- Sharing is a map b: I → B with q_i = E[b(i)]. Double hashing is a sum of two rows.
- Losses:
  - BPR, the pairwise logistic loss;
  - DirectAU, alignment + uniformity on **normalized** embeddings.
- Scoring modes: raw ⟨p, q⟩; cosine; cos + β·log(1+c_i). β is selected on validation from a fixed grid.
- Bucket traffic C_b = Σ_{i∈b} c_i.
- ρ_item = Spearman(‖q_i‖, c_i); ρ_bucket = Spearman(‖E_b‖, C_b).

### 4. What sharing does to popularity (1.5 pages): the intellectual core
State each result, prove it in the appendix, and give one sentence on why it matters.

- **Lemma 1 (user scale).** The ranking for user u does not depend on ‖p_u‖. *Why:* only item-side length can carry popularity into a ranking.
- **Lemma 2 (exact ties).** Items in the same bucket get identical scores under every mode that depends only on q. *Measured:* 87% of users have a tie at the top-20 boundary (14.7 tied items on average). Tie-breaking alone gives 12% of the gain.
- **Proposition 1 (scale invariance of DirectAU).** L_DAU depends on q/‖q‖ only.
  - **Corollary:** a learned per-item positive scale w_i (MEmCom-style) has ∂L_DAU/∂w_i = 0. A trained per-item scalar cannot store popularity under DirectAU. Length is set only by weight decay and optimizer dynamics, which under sharing aggregate over the bucket (H-Adam check: lengths only grow).
  - *This is why a score-time term is the natural probe.*
- **Proposition 2 (correlation-ratio bound).**
  - Decompose log-popularity as p_i = μ_{b(i)} + ε_i, where μ is the bucket mean.
  - For any bucket-level function f: corr(f(b(i)), p_i)² ≤ η² = Var(μ)/Var(p).
  - The residual ε is orthogonal to **every** representation that shares rows.
  - *Measured:* η² for GraphHash is 0.36–0.48, so 52–64% of popularity variance cannot be represented by any bucket-level quantity.
  - Remark: the bound is sufficient, not necessary; see F5 on double-frequency hashing.
- **Diagram (Fig 2a): the mechanism, one schematic.**
  - Left, full table: items as vectors with different lengths, length ∝ popularity.
  - Right, shared: a bucket is one vector of length set by C_b. Items inside it have identical arrows; the count bars c_i differ, but the vector cannot show it.
  - Label the decomposition p = μ_b + ε.

### 5. Experimental protocol (0.5 pages)
- **Data and backbones.**
  - Gowalla, Yelp2018, AmazonBook (GraphHash splits). MF dimension 64; LightGCN with 2 layers.
  - Sharing methods: GraphHash (Louvain, resolution 200); Random; DoubleFreq; DoubleGraph (a structure-aware stand-in for BACO).
  - Also report the bucket/item ratio.
- **Training.** Each loss uses the GraphHash authors' lr/wd grid.
  - "Converged" means stopped by patience. **Capped runs are flagged** and are not used for any convergence claim.
- **Selection and testing.** β is chosen on validation and the test set is scored once. Recall@20 uses the denominator min(20, |T_u|); NDCG@20 is secondary.
- **Statistics.**
  - Paired user bootstrap, 2,000 resamples; paired t-test with Holm correction.
  - 3 seeds for the main GraphHash pairs on all datasets. **Single-seed rows are marked †.**
- **Pre-registration.** Predictions were registered before each stage. Report the tally (PASS / FAIL counts) and put the registry in the anonymized supplement. Few papers do this, and it costs one sentence.

### 6. Results (2.75 pages), organised as research questions
Open each RQ with a one-line answer, then give the evidence.

- **RQ1. Does sharing remove DirectAU's advantage?**
  - Table 2, raw columns.
  - Answer: yes. The order reverses on Gowalla and Yelp; on AmazonBook DAU's lead shrinks from +1.19 to +0.53.
  - It also reverses under Random and DoubleFreq on Gowalla and Yelp. Under DoubleFreq on AmazonBook it does not (1.51 vs 1.28).
- **RQ2. Is popularity lost at item level?**
  - Fig 2b: ρ_bucket vs ρ_item per dataset.
  - Fig 3: the cosine ablation.
  - Answer: norms follow buckets, and removing length changes nothing under sharing.
- **RQ3. Does restoring item-level popularity close the gap, and is the gain item-specific?**
  - Fig 4: fixed-β decomposition with Shapley shares (μ 0.29 / ε 0.71 on Gowalla; ε 0.76 on AmazonBook), plus the within-bucket shuffle (keeps −6%).
  - Table 3: PR-E8 paired results.
  - Popularity-only baseline: 2.97 / 1.68 / 0.96 against the corrected 13.29 / 5.68 / 6.33. *The test is not MostPop.*
- **RQ4. Does erasing popularity without sharing reproduce the symptom?** This is the causal check; Table 4, all Full Gowalla DAU.
  - Global shuffle of item lengths: −11.47 (10 seeds).
  - Decile-preserving shuffle: −4.19.
  - Replacing length by a fitted function of popularity: 17.90 → 18.69.
  - Forcing equal norms: raw 12.77 → 19.32 with the prior.
  - *Remove item-level popularity from a full table and you get the sharing symptom; put it back and you get the fix.* This table is what turns correlation into mechanism. Give it space.
- **RQ5. Does the result generalise?** Table 5 and Fig 5.
  - DoubleGraph: corrected DAU vs BPR 10.91 vs 7.27 (Gowalla), 5.52 vs 2.78 (Yelp).
  - Resolution sweep: the gain falls as buckets get finer (Spearman −0.90 over 5 resolutions).
  - LightGCN with GraphHash: corrected DAU 17.06 vs BPR 16.27 (Gowalla), 9.12 vs 6.96 (Yelp). Some rows hit the 4 h cap; mark them.
  - LightGCN Full: BPR 18.48 > DAU 18.02 raw. Say the flip premise is MF-specific.
- **RQ6. What does it cost?** Table 6 and Fig 6 (optional).
  - Storage: float32 prior +5.52 / 3.89 / 7.29% of the GraphHash table. uint8 is **not** universal (3 checkpoints exceed 0.02 pp).
  - Tail recall: corrected GraphHash DAU is below Full DAU raw on all 3 datasets (Gowalla 2.01 vs 2.50; Yelp 0.29 vs 1.03; AmazonBook 3.78 vs 5.41).
  - Coverage: Gowalla 80.4 → 28.4%.
  - Weighted recall (w = (1+c)^−½): DAU still ahead after re-tuning (AmazonBook 5.53 vs 4.04).
  - Cite Cheng's own coverage drop (0.985 → 0.726): the cost belongs to the prior in general, not only to sharing.

### 7. Limitations and failed predictions (0.5 pages; required, not optional)
- The flip is MF-specific and holds on 2 of 3 datasets (F3).
- Failed predictions, stated as such:
  - Yelp Y3 (gap 0.44; ρ 0.52);
  - the Full-BPR range (PR-A8, 16.55 outside [15.4, 15.8]);
  - uint8 storage (E7);
  - the ρ_item resolution trend (C1, +0.70, the wrong sign);
  - the decile shuffle was larger than predicted (E2e).
- The tail and coverage cost: this is a head-accuracy correction.
- Single-seed rows (†): Random, DoubleFreq, DoubleGraph, LightGCN, the sweep.
- The learned end-to-end variant (C3) is **not reported** unless its code review is finished. If reviewed and still failing, write one sentence: "a jointly learned β·p + residual underperformed the post-hoc probe in our implementation (Appendix X)". Do not claim learned variants cannot work.
- Not tested: soft-gating sharing (GDE), BACO itself, CTR models, and the DNA and MEmCom baselines (if not run).

### 8. Conclusion (0.25 pages)
- Answer the title in one line: *sharing saves memory and costs item identity in the popularity signal*.
- A practical takeaway for people who build sharing schemes: **evaluate sharing schemes with item counts restored, or the loss you pick is decided by an artefact.**
- One future-work sentence: popularity-aware bucketing; soft gating.

### Appendix
- Proofs of Lemmas 1–2 and Propositions 1–2.
- Full grids (all modes × all runs).
- Hyperparameters and stopping status per run.
- NDCG tables; tie-policy robustness (+1.51 to +1.52); the reproduction rows (`x__*`).
- The pre-registration tally.

---

## 4. Figures and tables: what each shows, and where its data is

| ID | Type | Message (the caption's first sentence) | Data source | Status |
|---|---|---|---|---|
| Fig 1 | 2×2 grouped bars, Gowalla MF | Sharing removes DirectAU's lead; item counts restore it | CSV raw / cos_prior / raw_prior rows; `gowalla__mf__{full,graphhash}__{bpr,directau}__s0` (BPR GraphHash = `s0-p500`) | ready |
| Fig 2a | Schematic (TikZ or vector) | Length belongs to the bucket | none | to draw |
| Fig 2b | Scatter: log c_i vs ‖q_i‖, Full vs GraphHash (Gowalla DAU), bucket plateaus visible | Norms flatten within buckets | checkpoints + stored mappings (**do not recompute mappings on CPU**) | **check the artifacts exist**; if not, run one GPU job before G4 |
| Fig 3 | Paired bars per dataset: Δ(cosine − raw) and Δ(prior gain), Full vs GraphHash | Under sharing, length carries nothing usable | CSV: Full Δcos −4.96 / −4.13 / −0.87; GraphHash +0.02 / −0.44 / +0.01 | ready |
| Fig 4 | Stacked bar: Shapley μ / ε, plus the shuffle bar | Most of the gain is item-specific | RESULTS PR-DEC | ready |
| Fig 5 | Line: gain vs Louvain resolution | Finer buckets leave less to recover | RESULTS S4-C1 | ready |
| Fig 6 (opt.) | Accuracy–coverage curve over β | The cost is controllable | CPU rescoring of stored scores | optional, about 1 h CPU |
| Tab 1 | Dataset stats, buckets, bucket/item ratio, prior bytes % | Setup | data + mappings | fill the ratio |
| Tab 2 | Main grid: dataset × {Full, GH, Random, DF, DG} × {BPR, DAU}; raw / cos / corrected; ρ_item; ρ_bucket; † marks | Overall results | CSV | ready |
| Tab 3 | 9 seed pairs: Δ, 95% CI, Holm p | The ordering after correction holds every time | RESULTS PR-E8 + S4-SEED | ready |
| Tab 4 | Full-table interventions | The inverse experiment | RESULTS PR-E2e + S4-C4 | ready |
| Tab 5 | LightGCN + DoubleGraph | Generalisation, scoped | RESULTS S4-LGN, S4-DG, S5 | ready (mark capped rows) |
| Tab 6 | Costs: storage, uint8, tail, coverage, weighted recall | What the correction costs | CSV e5_* / e7_* / e9_* | ready |

**Figure rules.**
- One message per figure, stated in the caption's first sentence.
- Colours stay the same across all figures: BPR always one colour, DirectAU always another.
- Raw bars are solid, corrected bars hatched.
- Fonts in figures ≥ 7 pt.
- Generate everything from CSV with one script, `paper/make_figs.py`. No hand-typed numbers in the LaTeX: emit `numbers.tex` macros (`\gowGHdauRaw` and so on) from the CSV.

---

## 5. Words: use / avoid

| Use | Avoid |
|---|---|
| "diagnosis", "probe", "test", "restore item-level popularity" | "novel method", "our correction (as a contribution)" |
| "removes DirectAU's advantage" | "reverses the losses" (universal claims) |
| "head-accuracy correction with a tail cost" | "free", "tail-safe", "fixes" |
| "float32 prior, +3.9–7.3% of the table" | "one byte per item" |
| "item-specific share 0.71–0.76 (two datasets)" | "most of the gain" without the number |
| "the score form of Cheng (2026, eq. 53)" | anything that implies the formula is ours |

---

## 6. Reviewer objections, and where each is answered

| Objection | Answer | Where |
|---|---|---|
| "The prior is just MostPop" | Popularity-only gets 2.97 vs 13.29 | RQ3 |
| "The prior helps the full table too" | Difference-in-differences; cosine ablation | Fig 3, F4 |
| "Why not learn the bias (DNA, MEmCom)?" | Corollary: zero gradient under DirectAU. DNA changes training; ours is a training-free probe. DNA/MEmCom runs if time allows | §4, §7 |
| "The flip isn't general" | We scope it: 2/3 MF datasets; LightGCN Full has no flip premise | RQ1, §7 |
| "Single seeds" | 9 seed pairs for the main claim; † on everything else | Tab 2, Tab 3 |
| "Capped runs" | Flagged; excluded from convergence claims | §5, Tab 5 |
| "Random hashing is degenerate" | Reported as a stress case, with the bucket ratio given | Tab 1, F6 |
| "The correction hurts the tail" | Yes; reported, and Cheng sees the same | RQ6, §7 |
| "The bound doesn't explain DoubleFreq" | Sufficient, not necessary; ρ_item 0.027 measured | §4 remark, F5 |

---

## 7. Writing order (writing is the bottleneck; experiments are done)

| When | Work |
|---|---|
| **Thu 22** | `make_figs.py` + `numbers.tex`; Tab 2, Fig 1, Fig 3; draft §3–§4 (theory, from PLAN §4) |
| **Fri 23**, by **18:00 G4** | Check Fig 2b artifacts; draft §5–§6. **Freeze numbers at 18:00**; after that, only text changes |
| **Sat 24** | §1 Introduction, §2 Related work, §7 Limitations; appendix proofs |
| **Sun 25** | Revised abstract, §8 Conclusion; full read-through against §0 flags and §6 objections; trim to 8 pages |
| **Mon 26**, by 15:00 | Anonymisation pass (no repo links, no account names, no "our previous work"); anonymous supplement; compile; submit. 2.5 h buffer before 17:29 |

Write the intro **after** §6. You only know which numbers to put in the hook once the results section exists.
