# Paper plan v5: If Sharing Saves Memory, What Does It Cost? Popularity Without Identity

**Submission:** WebConf 2027, "Search, Recommendation, and Retrieval-Augmented AI" track.
- Format: `acmart` with `sigconf, anonymous, review`.
- Length: 8 main pages, 12 pages in total including references and appendix.
- The first 8 pages must stand alone.
- Due Mon 26 Oct, 17:29 IST.

**Sources:** every number below was checked against `results/master_metrics_v2.csv`, `results/ALL_RESULTS.md` and RESULTS.md on 22 Oct. Re-check each one by run_id before it goes into the paper.

---

## How to read this plan
- **Add:** what the section must contain.
- **Frame:** the stance and the exact wording to use. Sentences in quotes are drafts you can paste.
- **Highlight:** what the reader must not miss, and how to make it visible.
- **Avoid:** wording that a reviewer will attack.

## Global highlighting rules
- **One "Finding" per research question.** End each §6 subsection with a bold lead-in, e.g. **Finding 2.** followed by one sentence. Use five findings in total. They are what a skimming reviewer reads.
- **Italicise each defined term once, at first use.** The terms:
  - *count-restored score*
  - *bucket-level magnitude pooling*
  - *magnitude channel*
  - *direction channel*
  - *intra-bucket score ties*
- **Differences** are always in percentage points (pp). Paired differences always carry a 95% CI.
- **Table marks:**
  - † single seed; ‡ capped run.
  - Bold only the cell the table exists to show. Never bold a whole "best" column.
- **Causal verbs:**
  - Use "is consistent with", "accounts for X pp in this test", "does not explain the drop in our test".
  - Never use "proves", "rules out" or "because" for an empirical result. "Proves" is only for the lemmas and propositions.

---

## Review suggestions (11): what I took, and how

| # | Suggestion | Decision | Change made |
|---|---|---|---|
| 1 | Euler does not give monotone norm growth under Adam | **Taken** | Under SGD without decay, norms provably cannot shrink. Under Adam with decay, there is no guarantee, so growth is reported as an empirical fact (largest drop 0.15%, against 19% for BPR) |
| 2 | Keep every lemma inside its assumptions | **Taken** | Ties hold only for single-row sharing in MF under q-only scoring (not DoubleFreq, DoubleGraph or LightGCN final embeddings). User-norm invariance fails for raw+βp. Zero gradient holds for the DirectAU loss and w_i > 0, but a decay term on w_i still acts. Orthogonality is with respect to the uniform item inner product |
| 3 | The Spearman bound needs rank-based η² | **Taken** | ρ_item² ≤ η²(rank c) = 0.271 / 0.437 / 0.303 (holds: 0.076 / 0.269 / 0.036). The variance statement keeps η²_p = 0.36 / 0.48 / 0.36 |
| 4 | Remove causal overclaims | **Taken, and it changed the core analysis** | The equal-norm test does not show "capacity doesn't matter". New **two-channel decomposition**: on Gowalla the 10.47 pp drop = 4.96 magnitude + 5.49 direction. The count term addresses only the magnitude channel |
| 5 | Model-selection protocol | **Taken** | Scoring modes were fixed in advance (pre-registered): DirectAU cos+prior, BPR raw+prior. Only β is chosen on validation. The other modes go to the appendix |
| 6 | Wording of 9/9 and of 0.44 | **Taken** | "Corrected DirectAU outperforms corrected BPR on all nine pairs"; the order does not *reverse* on AmazonBook. The 0.44 pp figure applies only to GraphHash DirectAU (GraphHash BPR loses 3.51 pp under cosine) |
| 7 | Shapley shares explain the correction gain, not the compression loss | **Taken** | "71–76% of the *correction gain*" |
| 8 | Coverage needs a three-way comparison | **Taken, and it improved the cost story** | On the same checkpoint: GraphHash raw has 80.4% coverage and 7.67 tail recall; corrected has 28.4 / 2.01; Full raw has 30.8 / 2.50. The correction moves concentration back to roughly the Full model's level. Sharing had *raised* coverage by erasing popularity |
| 9 | Conclusion recommendation | **Taken** | "Report raw and count-restored results, with tail and coverage", not "always restore" |
| 10 | Organisation | **Mostly taken** | §3 renamed; RQs open §4; Full vs GraphHash is the main table; Random and DoubleFreq grids go to the appendix; 3 figures instead of 5. **Not taken:** moving DoubleGraph and LightGCN out of the main text, because they are the generality evidence |
| 11 | Self-citation | **Taken** | The CFP says to identify your own prior work in the third person. This paper needs none |

---

## Structure and page budget

| § | Section | Pages |
|---|---|---|
| — | Title, Abstract, CCS concepts, keywords | 0.4 |
| 1 | Introduction | 1.1 |
| 2 | Related Work | 0.65 |
| 3 | Problem Formulation and Preliminaries | 0.85 |
| 4 | Diagnostic Methodology | 1.5 |
| 5 | Experimental Setup | 0.5 |
| 6 | Results and Analysis | 2.2 |
| 7 | Trade-offs, Robustness, and Limitations | 0.6 |
| 8 | Conclusion | 0.2 |
| — | **Main text** | **8.0** |
| — | References (about 45) | about 1.2 |
| A–E | Appendix | about 2.8 |
| — | **Total** | **12.0** |

---

## Title
- Keep: *If Sharing Saves Memory, What Does It Cost? Popularity Without Identity*.
- The question names the trade-off and the subtitle gives the answer.
- No acronym. An acronym would signal a method.

## Abstract (about 205 words)

**Add.** One sentence each, in this order:
1. Setting.
2. Known facts.
3. Gap.
4. Finding.
5. Theory.
6. Diagnostic evidence.
7. Symptom.
8. Intervention.
9. Comparison.
10. Cost.

**Frame.** Use a diagnosis voice: "we show", "we prove", "consistent with". Name the score form as published.

**Highlight.** Every claim carries a number. The symptom (18.02 vs 16.55 → 7.55 vs 9.37) is the memorable line.

> Embedding sharing cuts the memory of Web-scale recommenders by mapping many items to one embedding row, but it also changes what the embedding can encode. On full tables, embedding magnitude encodes item popularity, and DirectAU is known to perform poorly under hashing, without an explanation. We show that sharing pools magnitude at the bucket level: under GraphHash, shared-row norms track bucket traffic (Spearman 0.75–0.94) far more than item popularity (0.19–0.52). Items sharing a row tie exactly, any bucket-level quantity explains at most 36–48% of log-popularity variance, and a learned per-item scale receives no gradient from the DirectAU loss. Consistent with this, discarding magnitude changes compressed DirectAU's Recall@20 by at most 0.44 points, against up to 4.96 points on full tables. On Gowalla, DirectAU leads BPR with full tables (18.02% vs 16.55%) but trails under GraphHash (7.55% vs 9.37%). Replacing pooled magnitude with a validation-tuned log-count prior, a published score form applied without retraining, raises compressed DirectAU to 13.30%; 71–76% of this gain comes from item-specific popularity. Corrected DirectAU outperforms corrected BPR on all nine dataset–seed pairs (+0.83 to +2.13 points). The prior adds 3.9–7.3% to table storage and concentrates recommendations on popular items, lowering tail recall and catalog coverage.

**Avoid.**
- "novel", "extensive experiments", "one byte", "fixes", "reverses on all datasets".

**CCS concepts:** Information systems → Recommender systems; Collaborative filtering.

**Keywords:** embedding table compression; parameter sharing; popularity; embedding magnitude; collaborative filtering; DirectAU.

---

## 1. Introduction (1.1 pages)

### ¶1 Web problem. This is mandatory under the CFP: Web relevance on page 1, or desk rejection.
- **Add.**
  - Per-ID embedding tables dominate the memory of Web-scale recommenders.
  - Sharing schemes cut that memory: the hashing trick, frequency hashing, GraphHash (WWW'25), BACO, GDE (SIGIR'26).
  - These schemes are judged by recall per byte.
- **Frame.** "Sharing is evaluated by how much accuracy survives per byte. We ask what else it removes."
- **Highlight.** One explicit Web-relevance sentence: "Because Web platforms serve catalogs of millions of items under fixed memory budgets, what sharing preserves determines which training objective a deployment should use."

### ¶2 What is known
- **Add.**
  - PRISM (KDD'25): on full tables, embedding magnitude encodes popularity.
  - GraphHash: graph-clustered buckets beat random hashing.
- **Frame.** Credit first. These works establish the two facts we connect.

### ¶3 The open problem
- **Add.**
  - GraphHash §5.2.1 and Shiao et al. (WWW'25 Companion) both report DirectAU doing poorly under hashing, with no mechanism.
  - Our controlled numbers: Gowalla Full DirectAU 18.02 vs BPR 16.55; GraphHash 7.55 vs 9.37.
- **Frame.** "This is not a new observation; it is an unexplained one."
- **Highlight.** Point to Fig 1a.

### ¶4 Why the obvious remedies fail
- **Add.**
  - More buckets: less compression.
  - A learned per-item scale (MEmCom): it gets no gradient from DirectAU (Prop 1).
  - Smarter tie-breaking: it recovers 12% of the gain.
- **Frame.** These are three reasons a diagnosis is needed before any fix.

### ¶5 Insight
- **Add.**
  - Under sharing, magnitude belongs to the bucket.
  - The item-specific part of popularity, ε = p − μ_bucket, has no place in a shared-row representation.
- **Highlight.** Point to Fig 1b. Introduce the term *bucket-level magnitude pooling*.

### ¶6 What we do and find
- **Add.**
  - Theory.
  - A two-channel decomposition of the accuracy drop (magnitude vs direction).
  - Nine alternative explanations, each tested.
  - The intervention the diagnosis predicts, and its costs.
- **Frame.** "Restoring item counts is a test of the diagnosis, not a proposed method. The score form is Cheng's (2026, eq. 53)."

### ¶7 Contributions (three bullets, each stating new knowledge)
- **Diagnosis.** "Under embedding sharing, magnitude tracks bucket traffic rather than item popularity, and for DirectAU it no longer helps ranking."
- **Analysis.**
  - "Items sharing a row tie exactly. Any bucket-level quantity reaches at most 36–48% of popularity variance. A learned per-item scale gets no gradient from DirectAU."
  - "On Gowalla, 4.96 of the 10.47 pp drop is the magnitude channel."
- **Test and cost.**
  - "Restoring item counts lets corrected DirectAU outperform corrected BPR on all nine dataset–seed pairs. 71–76% of the gain is item-specific."
  - "It re-concentrates recommendations to roughly the full model's coverage."

### ¶8 Scope
- "We do not propose a new score form, do not claim a tail-safe correction, and do not claim the loss ordering reverses everywhere."

### Fig 1 (top of page 1 or 2)
- **(a)** Bars, Gowalla MF: Full and GraphHash × BPR and DirectAU.
  - Raw (solid): 16.55 / 18.02 / 9.37 / 7.55.
  - Corrected (hatched): 11.31 / 13.30.
- **(b)** Scatter of log(1+c_i) against ‖q_i‖: Full DirectAU (ρ 0.81) next to GraphHash DirectAU (ρ 0.28). Bucket plateaus should show as horizontal stripes.
- **Caption:** "Under sharing, DirectAU loses its lead, consistent with magnitude no longer following the item."
- **Data: confirmed (23 Oct).** Panel (b) is computed on CPU from `runs/gowalla__mf__{full,graphhash}__directau__s0/best_seed0.pt` and the stored `mapping_items.npy`, with train counts from the GraphHash Gowalla split. It reproduces ρ_item 0.815 / 0.275. Never recompute the mappings on CPU.

**Avoid in the introduction.**
- "First to discover" (Shiao and GraphHash saw the effect first).
- "Simple yet effective".
- "Our method".

---

## 2. Related Work (0.65 pages)

**Add.** Three paragraphs. Each ends with one sentence starting "These works do not…".

### 2.1 Embedding table compression
- Covers: hashing trick; frequency/double hashing; Learning to Collide; CAFE (SIGMOD'24); MEC; CCE; GraphHash; BACO; GDE.
- **Frame.** Group them by what they optimise: which IDs collide.
- Closing sentence: "These works choose assignments; none asks what sharing does to magnitude or to the popularity signal."
- Note: GDE uses soft gating, so exact ties do not occur there. Say so.

### 2.2 Magnitude, popularity, and training objectives
- Covers: PRISM; Adap-τ (WWW'23); DDC (KDD'25); DirectAU (KDD'22); Wu et al. (TOIS'24); DNA (ASOC'26); CoRRe.
- **Frame.** "Magnitude encodes popularity" is their finding; we build on it.
- Closing sentence: "All study full tables."

### 2.3 Per-item terms and score-time popularity
- Covers: MEmCom (MLSys'22); Hash Embeddings (NeurIPS'17); Seol et al. (WSDM'24); logit adjustment (ICLR'21); logQ (RecSys'19); PDA (SIGIR'21); MACR (KDD'21); Cheng (2026, eq. 53).
- **Frame.** State plainly: "Our count-restored score is the form of Cheng (2026, eq. 53)."
- Closing sentence: "None connects these terms to what sharing erases, or shows why a learned scale cannot carry popularity under DirectAU."

**Highlight.** The three closing sentences together are the gap statement. Write them in parallel form.

**Avoid.** "To the best of our knowledge, no prior work…" without naming the closest work.

---

## 3. Problem Formulation and Preliminaries (0.85 pages)

**Purpose.** Definitions and the problem only. No theory.

### 3.1 Notation
- Users U and items I.
- Training counts c_i, and p_i = log(1 + c_i).
- Score s(u,i) = ⟨p_u, q_i⟩.

### 3.2 Embedding sharing
- The map b: I → B, with q_i = E[b(i)] (single-row sharing).
- Double schemes sum two rows.
- Bucket traffic C_b = Σ_{i∈b} c_i.
- GraphHash item buckets: 5,806 / 7,656 / 9,804 (mean 7.06 / 4.97 / 9.34 items per bucket; largest 124 / 183 / 343).
- Compressed table as a share of the full table: 16.4% / 21.9% / 13.6%.

### 3.3 Losses
- BPR: pairwise logistic loss on dot products.
- DirectAU: alignment plus uniformity on ℓ2-normalised embeddings.
- **Frame.** One line on why the distinction matters: "BPR sees magnitude; DirectAU's loss does not."

### 3.4 Scoring modes
- raw ⟨p_u, q_i⟩.
- cosine.
- The *count-restored score*, cos(p_u, q_i) + β p_i.
- raw + β p_i.

### 3.5 Diagnostic quantities
- ρ_item = Spearman(‖q_i‖, c_i).
- ρ_bucket = Spearman(‖E_b‖, C_b).
- η²(t) = Var(μ_b(t)) / Var(t), defined for any item-level target t. We use t = p (variance) and t = rank(c) (Spearman).

### 3.6 Problem statement
- **Add.** A small table, raw MF Recall@20:

  | Dataset | Full: DirectAU vs BPR | GraphHash: DirectAU vs BPR |
  |---|---|---|
  | Gowalla | 18.02 > 16.55 | 7.55 < 9.37 |
  | Yelp | 11.10 > 8.02 | 2.41 < 3.78 |
  | AmazonBook | 9.57 > 8.38 | 4.63 > 4.10 |

- **Frame the question.** "Why does sharing remove DirectAU's advantage, and what does sharing erase?"
- **Highlight: the AmazonBook boundary.** DirectAU's lead shrinks from +1.19 to +0.53 but does not reverse.
- **Highlight: the LightGCN boundary.** BPR already leads on the full Gowalla table (18.48 vs 18.02). The premise is MF-specific.
- **Frame.** State both boundaries here, not only in §7. This builds trust early.

---

## 4. Diagnostic Methodology (1.5 pages)

**Frame.** The method is the diagnosis: hypotheses, theory, and a test plan. The count-restored score is one of the tests.

### 4.1 Research questions (open the section with them)
- **RQ1:** Where does the Full→shared accuracy drop come from: magnitude or direction?
- **RQ2:** Does magnitude follow buckets instead of items?
- **RQ3:** Do alternative explanations account for the drop?
- **RQ4:** Does restoring item counts let DirectAU outperform BPR again, and is the gain item-specific?
- **RQ5:** Does this hold for other sharing schemes, bucket resolutions and LightGCN?
- Costs and robustness are handled in §7.

### 4.2 Theory
Statements go here; proofs in App. A. State each result's assumptions in its first clause.

- **Lemma 1 (user scale).**
  - Statement: under raw, cosine and count-restored scoring, positive rescaling of p_u leaves u's ranking unchanged. Under raw + βp it does not.
  - Consequence: under these scorings, popularity can reach a ranking only through item-side quantities, i.e. magnitude and direction.
- **Lemma 2 (exact ties).**
  - Statement: under single-row sharing in MF, items with b(i) = b(j) receive identical scores under every scoring that depends on the item only through q_i.
  - Does not apply to: two-row schemes (DoubleFreq, DoubleGraph) unless both rows match; LightGCN final embeddings, since propagation differs per item.
  - The prior breaks the ties by design.
  - Measured: 87% of users have a tie at the rank-20 boundary; the mean tied group is 14.7 items.
- **Proposition 1 (scale invariance).**
  - Statement: the DirectAU loss depends on q only through q/‖q‖.
  - Corollary: for a learned per-item scale q_i = w_i E[b(i)] with w_i > 0 (MEmCom's U[i mod m] ⊙ V[i]), the loss gradient with respect to w_i is 0. Under cosine scoring, w_i is also invisible.
  - Boundary: a weight-decay term on w_i still shrinks it, and BPR is not scale-invariant.
- **Remark (norm dynamics; replaces the old claim).**
  - Under SGD without decay, Euler's identity ⟨∇L, q⟩ = 0 gives ‖q_{t+1}‖² = ‖q_t‖² + η²‖g_t‖², so norms cannot shrink.
  - Adam's per-coordinate scaling and weight decay break this, so we report growth as an observation (§6.2), not a theorem.
- **Proposition 2 (correlation-ratio bound).**
  - Statement: for any target t and decomposition t = μ_b(t) + ε, every bucket-level f satisfies corr(f(b), t)² ≤ η²(t). Correlation is taken over items with uniform weight, and ε is orthogonal to every bucket-level function in that inner product.
  - With t = p, a bucket-level quantity reaches at most η²_p = 0.36 / 0.48 / 0.36 of log-popularity variance.
  - With t = rank(c), and since a bucket-level norm has bucket-level average ranks, ρ_item² ≤ η²(rank c) = 0.271 / 0.437 / 0.303.
  - Remark: the condition is sufficient, not necessary. Under DoubleFreq, η² is 0.998, yet ρ_item is 0.03.

**Fig 3.** Conceptual schematic, labelled "conceptual".
- Full table: one arrow per item, with length tracking count.
- Shared table: one arrow per bucket, with the items' count bars under it.
- Annotate t = μ_b + ε.

### 4.3 Main hypothesis H_pool and its predictions
- (P1) ρ_bucket ≫ ρ_item under sharing.
- (P2) For DirectAU, cosine ≈ raw under sharing but cosine ≪ raw on full tables.
- (P3) Restoring p recovers a gain driven mostly by ε.
- (P4) A larger magnitude channel on the full table goes with a larger loss under sharing.

### 4.4 Two-channel decomposition (answers RQ1)
- **Path:** Full raw → Full cosine → GraphHash cosine → GraphHash raw.
  - Magnitude channel = Full raw − Full cos.
  - Direction channel = Full cos − GraphHash cos.
  - Residual = GraphHash cos − GraphHash raw.
- **Frame.** This is a path decomposition, not a unique attribution. Say so in one clause.

### 4.5 Alternative explanations and their tests
Results are reported in §6.3.

| Alternative | Test |
|---|---|
| A1 All of the drop is lost capacity | Erase magnitude on the full table (train with equal norms), keeping every parameter |
| A2 Ties alone | Break ties by popularity, with no prior |
| A3 Bucket-level popularity suffices | Fixed-β μ/ε decomposition (Shapley); shuffle counts within buckets |
| A4 Magnitude carries other information | On the full table: replace magnitude by a fitted function of p; global and decile shuffles |
| A5 Undertraining | Use only runs that stopped by patience; flag capped runs |
| A6 Initialisation | Init-matched reruns |
| A7 Weight decay | Train with wd = 0 |
| A8 Generic popularity (MostPop) | Popularity-only ranking |
| A9 A learned scale would suffice | Proposition 1 |

### 4.6 The intervention as a test
- **Add.**
  - The scoring mode for each loss was fixed before testing and pre-registered: DirectAU with the count-restored score, which matches its normalised training geometry; BPR with raw + βp, which matches its dot-product training.
  - Only β is chosen on validation.
  - The other modes are reported in App. C. Both score lower on test: BPR cos+prior 9.96; DirectAU raw+prior 10.33.
- **Frame.** "If H_pool holds, replacing pooled magnitude by item counts should let DirectAU outperform BPR again under the same table."

---

## 5. Experimental Setup (0.5 pages)
- **Datasets.** Gowalla, Yelp2018 and AmazonBook, using the GraphHash splits.
  - Users / items: 29,858 / 40,981; 31,668 / 38,048; 52,643 / 91,599.
  - Tab 1: interactions, buckets, table bytes, prior bytes.
- **Sharing schemes.**
  - GraphHash (Louvain, resolution 200) is the main scheme.
  - Random, DoubleFreq and DoubleGraph are secondary.
  - Random uses the same bucket count as GraphHash. Higher resolutions use more buckets (8,924–25,993 on Gowalla).
- **Models.** MF with d = 64; LightGCN with 2 layers.
- **Training.**
  - Each loss uses the GraphHash authors' lr/wd grid.
  - BPR is trained full-batch; DirectAU uses batch size 1024.
  - Early stopping by patience.
  - **Frame.** "We call a run converged only if it stopped by patience; capped runs (‡) are reported but never support a convergence claim."
- **Selection.** Scoring modes are fixed in advance (§4.6). Only β is selected on validation, from a fixed grid. The test set is scored once.
- **Metrics.** Recall@20 with denominator min(20, |T_u|) is primary. NDCG@20 is secondary.
- **Statistics.**
  - Paired user bootstrap with 2,000 resamples.
  - Paired t-test with Holm correction across datasets.
  - 3 seeds for the main pairs; single-seed rows are marked †.
- **Pre-registration.**
  - "Predictions were registered before each experimental stage; we report failed ones." Give the tally here.
  - The full registry goes in the anonymous repository.
- **Reproduction check.** "Our GraphHash BPR reaches 9.34 Recall@20 on Gowalla (reported: 9.30)."
- **Highlight.** Pre-registration and validation-only selection are the paper's credibility signals. Give each its own sentence; don't bury them in a list.

---

## 6. Results and Analysis (2.2 pages)

**Frame for every paragraph:** question → contrast → observation → interpretation → boundary.

### 6.1 RQ1: where the drop comes from
- **Add.** Tab 2, the main table.
  - Rows: Full and GraphHash × BPR and DirectAU × 3 datasets.
  - Columns: raw / cos / count-restored (β) / raw+βp (β) / ρ_item / ρ_bucket.
- **Add.** Fig 2: stacked bars per dataset showing the two-channel decomposition for DirectAU.

  | Dataset | Full→GraphHash drop | Magnitude channel | Direction channel | Residual |
  |---|---|---|---|---|
  | Gowalla | 10.47 | 4.96 | 5.49 | 0.02 |
  | Yelp | 8.69 | 4.13 | 5.00 | −0.44 |
  | AmazonBook | 4.94 | 0.87 | 4.06 | 0.01 |

- **Highlight: the AmazonBook boundary as consistency.**
  - AmazonBook has the smallest magnitude channel (0.87 pp), and it is the one dataset without a reversal.
  - **Frame (post hoc, say so):** "The dataset where magnitude carried least of DirectAU's full-table score is the one where sharing does not reverse the order."
- **Add.** One sentence on Random and DoubleFreq: the raw order reverses in 5 of 6 cells. Random is a stress case (raw Recall below 1%). Full grid in App. C.
- **Finding 1.** "Sharing costs DirectAU through two channels; the magnitude channel, which carries popularity, is 0.87–4.96 pp of the drop."

### 6.2 RQ2: magnitude follows buckets
- **Add.**
  - ρ_bucket 0.924 / 0.944 / 0.745 against ρ_item 0.275 / 0.519 / 0.189. On full tables ρ_item is 0.81 / 0.91 / 0.79.
  - GraphHash quintile mean lengths: 11.71 → 12.71, nearly flat.
  - The bound holds: ρ_item² = 0.076 / 0.269 / 0.036 ≤ η²(rank c) = 0.271 / 0.437 / 0.303.
  - Norm dynamics (observation): the largest drop in quintile-mean squared norm between evaluations is 0.15% for DirectAU, against 19% for BPR.
- **Add (P2).** For GraphHash DirectAU, cos − raw = +0.02 / −0.44 / +0.01, against −4.96 / −4.13 / −0.87 on full tables.
- **Boundary.** This holds for DirectAU only. GraphHash BPR still loses 3.51 pp under cosine: BPR's shared magnitude carries something else (ρ_item is only 0.07).
- **Boundary (Yelp).** Yelp has the highest ρ_item (0.52) and η² (0.48), so less is erased. This is consistent with the bound, and it is our failed Y3 prediction; say so.
- **Finding 2.** "Under GraphHash, magnitude tracks bucket traffic, not items, and for DirectAU it no longer helps ranking."

### 6.3 RQ3: alternative explanations
- **Add.** Tab 3, outcomes. The verdict column is headed "Accounts for the drop in this test?"

  | Alternative | Result | Answer |
  |---|---|---|
  | A1 Capacity explains all of it | Full table with equal norms: 18.02 → 12.77 (−5.25 pp, about half the GraphHash drop). Adding the prior gives 19.32, against GraphHash 13.30 | Partly. Magnitude erasure alone produces half the drop; the remaining 6.0 pp after correction is the direction channel |
  | A2 Ties alone | Tie-breaking recovers 12% of the DirectAU gain (34% for BPR) | A small part |
  | A3 Bucket popularity suffices | ε share 0.71 / 0.76; within-bucket shuffle keeps −6% of the gain | No: the gain needs item-level counts |
  | A4 Magnitude carries other information | Legacy Full checkpoint (17.90): fitted replacement 18.69; global shuffle 6.28; decile shuffle 13.64 | On full tables, magnitude's ranking value is reproduced by popularity |
  | A5 Undertraining | All main runs stopped by patience | Not supported |
  | A6 Initialisation | Change within 0.05 | Not supported |
  | A7 Weight decay | wd = 0 keeps ρ 0.74 | Not weight decay alone |
  | A8 MostPop | 2.97 / 1.68 / 0.96 vs 13.29 / 5.68 / 6.33 | No |
  | A9 Learned scale | Zero gradient from the DirectAU loss (Prop 1) | Excluded for DirectAU by theory, within its assumptions |

- **Frame A4's failed part.** The decile shuffle was predicted to change little and dropped 4.19 pp. "Coarse bins lose within-decile popularity; the prediction failed."
- **Finding 3.** "In our tests, ties, initialisation, stopping, weight decay and generic popularity do not account for the magnitude channel; item-level popularity does."

### 6.4 RQ4: restoring item counts
- **Add.** Tab 4: the nine pairs.

  | Dataset | Seed Δ (pp), s0 / s1 / s2 | Holm p (s0 pair) |
  |---|---|---|
  | Gowalla | +1.99 / +2.13 / +2.08 | 2.4e-63 |
  | Yelp | +0.88 / +0.86 / +0.83 | 6.1e-33 |
  | AmazonBook | +1.52 / +1.48 / +1.54 | 1.2e-159 |

  - Also report the 95% CIs and the NDCG Δ (+0.55 to +1.69).
- **Add.**
  - Gowalla mean ± SD: 13.325 ± 0.024 vs 11.257 ± 0.063.
  - The correction recovers 55% of the raw Full→GraphHash gap on Gowalla ((13.30 − 7.55) / (18.02 − 7.55)).
  - Shapley at fixed β, Gowalla: Δ_μ +1.74, Δ_ε +4.22, both +5.79; shares μ 0.29 / ε 0.71 (AmazonBook ε 0.76).
- **Frame.** "71–76% of the *correction gain* comes from item-specific popularity." Do not say "of the compression loss".
- **Boundary.** It does not reach the full table. The direction channel remains (13.30 vs 19.57 for Full count-restored).
- **Finding 4.** "Restoring item counts lets corrected DirectAU outperform corrected BPR on all nine dataset–seed pairs; most of the gain is item-specific."

### 6.5 RQ5: generality
- **Add.** Tab 5, top half.
  - **DoubleGraph:** corrected DirectAU vs BPR 10.91 vs 7.27 on Gowalla and 5.52 vs 2.78 on Yelp. ρ ≈ 0. †
  - **Resolution sweep:**
    - The gain falls 5.75 → 4.99 as resolution rises 200 → 3200 (Spearman −0.90).
    - η²_p rises 0.36 → 0.92.
    - Corrected DirectAU leads at every resolution.
    - ρ_item plateaus after resolution 800: a failed trend prediction. †
  - **LightGCN + GraphHash:**
    - Raw DirectAU < BPR: 11.29 vs 16.22; 4.71 vs 6.96.
    - Corrected DirectAU wins: 17.06 vs 16.27; 9.12 vs 6.96.
    - ρ_item 0.15 / 0.13.
    - Lemma 2 does not apply after propagation, yet the pattern holds. Rows ‡ where capped.
- **Finding 5.** "The pattern holds across two other sharing schemes, five bucket resolutions and LightGCN, but the full-table premise is MF-specific."

---

## 7. Trade-offs, Robustness, and Limitations (0.6 pages)

### 7.1 Trade-offs: what the correction costs (Tab 5, bottom half)

**Storage.**
- The float32 prior adds 5.52 / 3.90 / 7.30% to the compressed table (0.90 / 0.85 / 0.99% of the full table).
- uint8 is not universal: 3 of 11 checkpoints exceed 0.02 pp. Make no one-byte claim.

**Three-way comparison (frame).** Compare the same GraphHash checkpoint, uncorrected and corrected, against the full model. Recall@20 / tail recall / coverage:

| Model | Gowalla | Yelp | AmazonBook |
|---|---|---|---|
| GraphHash raw | 7.55 / 7.67 / 80.4 | 2.41 / 1.47 / 61.0 | 4.63 / 5.04 / 92.3 |
| GraphHash corrected | 13.30 / 2.01 / 28.4 | 5.68 / 0.29 / 21.7 | 6.33 / 3.78 / 80.7 |
| Full raw | 18.02 / 2.50 / 30.8 | 11.10 / 1.03 / 32.0 | 9.57 / 5.41 / 72.4 |

**Highlight.**
- "Sharing had *raised* coverage and tail recall by erasing popularity."
- "The correction moves concentration back towards the full model's level (Gowalla coverage 28.4 vs 30.8), while tail recall stays below it."
- 89–98% of the added hits are head items.

**Frame.**
- A head-accuracy correction with a tail cost.
- Cheng sees a similar coverage drop with the same score form on full tables (0.985 → 0.726).
- Do **not** say the cost belongs only to the prior.

### 7.2 Robustness of the main comparison
- Seeds: 9 of 9.
- Tie policy (AmazonBook, 13 policies): +1.508 to +1.524.
- NDCG agrees.
- Weighted recall (w = (1+c)^−½, β re-tuned on weighted validation): the order is kept. Gowalla +2.14 to +2.34; Yelp +0.82; AmazonBook 5.53 vs 4.04.
- Initialisation and stopping: A5 and A6.

### 7.3 Limitations
Each limitation names the boundary and what it changes.
- **Scope.** The order does not reverse on AmazonBook MF, and the premise fails for LightGCN full tables. The claim is "removes the advantage".
- **Single seeds and caps.** † rows (DoubleGraph, Random, DoubleFreq, the sweep, LightGCN) and ‡ capped LightGCN rows.
- **Failed predictions.** The Full-BPR range (16.55 vs [15.4, 15.8]), Y3, uint8 storage, the decile shuffle, the ρ_item trend. Full list in App. E.
- **Learned end-to-end variant (C3).** Report it only if the code review clears it. Otherwise: "not evaluated".
- **Not tested.** Soft gating (GDE), BACO itself, CTR models, DNA and MEmCom baselines, online effects.

**Highlight.** Put the failed predictions in the main text, as one sentence with numbers. That is the honesty signal.

---

## 8. Conclusion (0.2 pages)
- **Answer the title.** "Sharing saves memory and costs item identity in the popularity signal: under GraphHash, magnitude describes the bucket, not the item."
- **Practical recommendation (revised).** "When comparing training objectives under embedding sharing, report raw and count-restored results together with tail and coverage. Raw comparisons alone mix popularity loss with loss quality."
- **Future work.** Popularity-aware bucketing; soft-gated sharing.
- No limitations here; they live in §7.

---

## References (about 1.2 pages)
**Fix these entries (from N1):**
- Park et al. (CIKM'23): use the exact title.
- Wu et al.: TOIS 2024.
- Wan et al.: use the NeurIPS title.
- Yi et al.: confirm the first author.
- Shiao et al.: WWW'25 Companion.
- DNA: ASOC 186:114226.
- Cheng: arXiv 2606.21275.

**Self-citation.** Cite your own work only in the third person (CFP). This paper needs none.

---

## Appendix (about 2.8 pages)
- **A. Proofs (0.8 page).**
  - Lemmas 1–2, each with its assumptions.
  - Prop 1 and its corollary, MEmCom form.
  - The SGD norm remark, and why it fails under Adam.
  - Prop 2 for a general target t: the p and rank(c) cases, and how tied ranks are handled.
- **B. Protocol (0.4).**
  - lr / wd / batch / patience for each loss.
  - β grids, including the grid-edge case β = 5.0.
  - Masks and denominators.
  - Evaluated users: 25,789 / 30,304 / 50,407.
  - Bootstrap and Holm details.
  - Compute: T4 runtimes. The two-GPU split gave no speedup.
- **C. Complete run ledger (1.0).** One table with every training run (about 75).
  - Columns: run_id, stop reason, best/run epochs, raw, cos, count-restored (β), raw+βp (β), ρ_item, ρ_bucket, flags (†, ‡, legacy, excluded).
  - Includes:
    - the Random and DoubleFreq grids;
    - the reproduction rows;
    - init controls;
    - weight-decay variants;
    - equal-norm runs;
    - the resolution sweep;
    - LightGCN Stages 4 and 5;
    - popularity-only;
    - the 4 excluded runs, with reasons;
    - the alternative scoring modes for the pre-registered pairs.
- **D. Extra diagnostics (0.4).**
  - η²_p and η²(rank c) for every map: Random 0.14 / 0.20 / 0.11; DoubleFreq 0.998; sweep 0.49 / 0.63 / 0.79 / 0.92.
  - ρ_proxy.
  - Quintile lengths.
  - Norm dynamics.
  - ρ_dir (BPR > DirectAU, 2 of 2 pairs).
  - Tie statistics for each checkpoint (tie-only share 12–60%).
  - μ/ε decomposition for each checkpoint.
  - AmazonBook length interventions.
  - Gains by activity quartile.
- **E. Failed predictions and the pre-registration tally (0.2).** Every failed prediction with its numbers.

**Anonymous repository.** Linked once, in §5.
- **Contents:**
  - the full CSV;
  - `ALL_RESULTS.md`;
  - the RESULTS blocks;
  - PREREG with snapshot hashes;
  - the E5/E7/E9 JSON files;
  - the code.
- **[FLAG] Scrub before upload:**
  - no `runs/_sessions`, no `sessions_*.py`, no account or slug strings;
  - no git history;
  - grep for usernames;
  - check the PDF metadata.
- The CFP does not mention supplementary material; check the OpenReview form.

---

## Figures and tables (3 figures, 5 tables)

| ID | Section | Job | Status |
|---|---|---|---|
| Fig 1a/b | §1 | Symptom; magnitude follows the bucket | ready (1b data confirmed on CPU) |
| Fig 2 | §6.1 | Two-channel decomposition | ready |
| Fig 3 | §4.2 | Conceptual schematic | to draw |
| Tab 1 | §5 | Datasets, buckets, bytes | ready (bucket counts now known) |
| Tab 2 | §6.1 | Full vs GraphHash main table | ready |
| Tab 3 | §6.3 | Alternative explanations | ready |
| Tab 4 | §6.4 | Nine paired comparisons | ready |
| Tab 5 | §6.5 / §7.1 | Generality (top half); costs, three-way (bottom half) | ready |

---

## Wording

| Use | Avoid |
|---|---|
| embedding table compression, parameter sharing, bucket | "sharing rows" |
| magnitude (ℓ2-norm), direction; magnitude channel, direction channel | "length" in formal text |
| intra-bucket score ties; bucket-level magnitude pooling | invented acronyms |
| count-restored score; post hoc; without retraining | "our method", "novel correction" |
| removes DirectAU's advantage; outperforms corrected BPR | "reverses on all datasets" |
| consistent with; accounts for X pp in this test | proves (for experiments), rules out, because |
| head-accuracy correction with a tail cost | free, tail-safe, fixes |
| float32 prior, +3.9–7.3% of the compressed table | one byte per item |
| percentage points | bare % for differences |
| the Cheng (2026, eq. 53) score form | anything implying the formula is ours |

---

## Writing order

| Day | Work |
|---|---|
| Thu 22 | §3, §4 (theory with assumptions; Tab 3 plan) |
| Fri 23 | §5, §6. Numbers freeze at 18:00 |
| Sat 24 | §1 (written after §6), §2, §7; App. A–C |
| Sun 25 | Abstract, §8, App. D–E; check every causal verb; trim to 8 + 4 pages |
| Mon 26 | Scrubbed repository; anonymity check; submit by 15:00 |
