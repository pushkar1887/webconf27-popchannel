# Paper plan: If Sharing Saves Memory, What Does It Cost? Popularity Without Identity

- **Target:** WebConf 2027, research track "Search, Recommendation, and Retrieval-Augmented AI".
- **Format:** `\documentclass[sigconf, anonymous, review]{acmart}`.
- **Length:** 8 main pages, **12 pages total including references and appendix**. The first 8 must stand alone.
- **Due:** Mon 26 Oct, 17:29 IST.

Every number below was checked against `results/master_metrics_v2.csv` and RESULTS.md on 22 Oct. Re-check by run_id before it enters the paper.

---

## 0. What I kept and rejected from the Codex and Gemini reviews

### Kept
| Source | Idea | Where it goes |
|---|---|---|
| Codex | Diagnosis-family structure: phenomenon → competing explanations → discriminating tests → intervention → costs | Whole paper (§3) |
| Codex | Every claim creates an obligation that a later section must discharge | Claim ledger (§2) |
| Codex | Write the "what we do not establish" line before drafting | §1, line 7 |
| Codex | Test what the intervention must preserve, not only what it gains (Responsible Diffusion) | Tail and coverage costs, §5.4 |
| Codex | Results paragraph: question → contrast → observation → interpretation → boundary | Every result paragraph |
| Codex | Web relevance stated on page 1 (CFP: desk reject otherwise) | Intro ¶1 |
| Codex | 12 pages total | Appendix plan (§6) |
| Gemini | Pit H_tie (ties) against H_pool (magnitude pooling) | §4.3, extended to 9 alternatives |
| Gemini | Two-panel Fig 1: bars plus a norm–popularity scatter | Fig 1 |
| Gemini | RQ-numbered result subsections; standard terms (embedding table compression, ℓ2-norm, intra-bucket score ties) | §3, §8 |

### Rejected: Gemini is wrong on these, and copying them would put false statements in the paper
| Gemini says | Reality | Do this instead |
|---|---|---|
| "1 byte per item", "<0.4% overhead" | E7 **failed**: 3 of 11 checkpoints exceed 0.02 pp. The float32 prior is 3.9–7.3% of the compressed table (0.85–0.99% of the full table) | Claim float32, +3.9–7.3% |
| "17.98 vs 15.62", "7.66 vs 9.30" | These are GraphHash's published numbers. Ours are 18.02 / 16.55 and 7.55 / 9.37 | Ours in the paper; theirs cited as the prior observation |
| Repaired 13.29 vs BPR 11.24 | 11.24 is the legacy BPR row. The registered comparator is s0-p500 at 11.31 | 13.30 vs 11.31 |
| "restores up to 75% of the gap" | (13.30 − 7.55) / (18.02 − 7.55) = **55%**. The 0.71–0.76 figure is the Shapley ε share, a different quantity | Say "55% of the raw Full-vs-shared gap" |
| Acronym titles ("DePool: …") | Implies a method. Contradicts the submitted title and the diagnosis framing | Keep the submitted title |
| GitHub footnote on page 1 | Breaks double-blind | Anonymous repository only (§6) |
| Memory–accuracy Pareto frontier with int8; LSH baseline | Not run, and the int8 result failed | Drop |
| Weight decay + mini-batch causes the pooling | D7: the link persists at wd = 0 (ρ 0.74). Our argument is scale invariance plus bucket-level aggregation of updates | State that |
| "De-pooled restoration", "decoupled scoring" | Invented jargon | One plain name: **count-restored score** |

---

## 1. The spine: seven lines, before any prose

1. **Web problem.** Web-scale recommenders spend most of their memory on per-ID embedding tables. Embedding sharing (hashing, graph-clustered buckets) is the standard way to cut it, and sharing schemes are judged by recall per byte.
2. **What the closest work establishes.**
   - On full tables, embedding magnitude encodes item popularity (PRISM, KDD'25).
   - DirectAU performs poorly under hashing (GraphHash, WWW'25; Shiao et al., WWW'25 Companion), with no explanation.
   - A log-popularity score term exists (Cheng 2026, eq. 53; logit adjustment).
3. **What is unresolved.** What sharing does to the popularity signal, and why the losses swap under sharing.
4. **Central insight.** Sharing makes magnitude a property of the bucket, not the item. The item-level part of popularity, ε = p − μ_bucket, becomes unrepresentable.
5. **Contribution.** A diagnosis with theory and discriminating tests. Restoring item counts serves as the intervention that tests the diagnosis.
6. **Evidence.** See the claim ledger (§2).
7. **What we do not establish:**
   - that the flip is universal (it fails on AmazonBook MF and LightGCN Full);
   - that the correction is tail-safe or free;
   - that a learned correction cannot work;
   - anything about soft-gated sharing (GDE), CTR models, or online behaviour.

---

## 2. Claim ledger (each claim → test → permitted conclusion)

| # | Claim | Discriminating evidence | Location | Permitted conclusion |
|---|---|---|---|---|
| C1 | Sharing removes DirectAU's advantage | Raw Recall, Full vs shared, both losses, 3 datasets | §4.1, Tab 2 | The lead reverses on Gowalla and Yelp (MF) and shrinks on AmazonBook (+1.19 → +0.53) |
| C2 | Under sharing, magnitude tracks buckets, not items | ρ_bucket 0.75–0.94 vs ρ_item 0.19–0.52 (Full DAU ρ_item 0.81 / 0.91 / 0.79) | §4.2, Fig 1b | A measured association on 3 datasets |
| C3 | Under sharing, the learned magnitude is useless for ranking | Cosine ablation: Full −4.96 / −4.13 / −0.87; shared +0.02 / −0.44 / +0.01 | §4.2, Fig 2 | Removing magnitude costs nothing under sharing |
| C4 | Item-level popularity is unrepresentable by any bucket-level quantity | Prop 2: η² 0.36–0.48 (GraphHash); ρ_item² ≤ η² on every dataset | §4.2 | A bound, sufficient but not necessary (DoubleFreq: η² 0.999 yet ρ_item 0.03) |
| C5 | The loss is item-level popularity, not capacity or the other alternatives | 9 alternatives, each tested (§4.3) | §4.3, Tab 3 | The listed alternatives do not explain the drop |
| C6 | Restoring item counts reverses the ordering | 9 of 9 dataset × seed pairs, Δ +0.83 to +2.13, every CI > 0, Holm p ≤ 6e-33 | §5.1, Tab 4 | Within MF, three datasets, GraphHash |
| C7 | It generalises beyond GraphHash and MF | DoubleGraph; resolution sweep; LightGCN with GraphHash | §5.2 | Holds in these settings; some LightGCN rows are capped; single seed (†) |
| C8 | It has costs | Storage, tail recall, coverage, weighted recall | §5.3 | A head-accuracy correction with a tail cost |

---

## 3. Structure and page budget (diagnosis family)

| § | Section | Pages | Reader question it answers |
|---|---|---|---|
| — | Title, abstract, CCS concepts, keywords | 0.4 | — |
| 1 | Introduction (Fig 1) | 1.1 | Why should a Web-recsys reader care, and what is new? |
| 2 | Setting and protocol | 0.75 | What exactly is compared, and is it fair? |
| 3 | The observed failure (RQ1) | 0.5 | Is there a real symptom? |
| 4 | What sharing erases: theory and discriminating tests (RQ2, RQ3) | 2.3 | Why does it happen, and what else could explain it? |
| 5 | Restoring item counts: effect, generality, cost (RQ4–RQ6) | 1.75 | Does the diagnosis predict an intervention, and what does that intervention cost? |
| 6 | Related work | 0.65 | How does this differ from PRISM, Cheng, MEmCom, GraphHash? |
| 7 | Limitations and scope | 0.35 | Where does the claim stop? |
| 8 | Conclusion | 0.2 | What is now known? |
| | **Main text** | **8.0** | |
| | References (~45 refs) | ~1.2 | |
| A–E | Appendix | ~2.8 | |
| | **Total** | **12.0** | |

**Related work goes late.** Diagnosis papers should reach the phenomenon by page 2. Credit attribution still happens early: intro ¶3 names PRISM, Cheng, MEmCom and GraphHash. G-Refer also places related work late.

---

## 4. Section by section

### Abstract (proposed revision; check whether the submitted abstract can still be edited)

> Embedding sharing cuts the memory of Web-scale recommenders by mapping many items to one embedding row, but it also changes what the embedding can encode. Prior work shows that on full tables embedding magnitude encodes item popularity, and that DirectAU performs poorly under hashing; why remains unexplained. We show that sharing pools magnitude at the bucket level: shared-row norms track bucket traffic (Spearman 0.75–0.94) far more than item popularity (0.19–0.52). We prove that items in a bucket tie exactly, that under GraphHash any bucket-level quantity explains at most 36–48% of log-popularity variance, and that a learned per-item scale receives zero gradient under DirectAU. Consistent with this, discarding magnitude costs compressed models at most 0.44 points of Recall@20, against up to 4.96 for full tables. On Gowalla, DirectAU leads BPR with full tables (18.02% vs 16.55%) but trails under GraphHash (7.55% vs 9.37%). Replacing pooled magnitude with a validation-tuned log-count prior, a published score form applied without retraining, raises compressed DirectAU to 13.30%; 71–76% of the gain is item-specific. Equally corrected, DirectAU exceeds BPR on all nine dataset–seed pairs (+0.83 to +2.13 points). The prior adds 3.9–7.3% to table storage and lowers tail recall and catalog coverage.

**CCS concepts:** Information systems → Recommender systems; Collaborative filtering.

**Keywords:** embedding table compression, parameter sharing, popularity, embedding magnitude, collaborative filtering, DirectAU.

### 1. Introduction (Codex's move sequence filled with our content)
1. **Web problem (Web relevance must appear here).**
   - ID embedding tables dominate memory in Web-scale recommenders.
   - Sharing schemes cut it: the hashing trick, frequency hashing, GraphHash (WWW'25), BACO, GDE (SIGIR'26).
   - These schemes are compared by recall per byte, under whatever training loss the deployment already uses.
2. **Closest answer.** GraphHash shows that graph-clustered buckets beat random hashing. PRISM shows that on full tables popularity is stored in embedding magnitude.
3. **Unresolved condition.**
   - GraphHash and Shiao et al. report that DirectAU does poorly under hashing, and give no explanation.
   - Our numbers: Gowalla Full DAU 18.02 > BPR 16.55; GraphHash DAU 7.55 < BPR 9.37 (Fig 1a).
   - Credit sentence: the score form we use later is Cheng (2026, eq. 53); per-item parameters on shared tables exist (MEmCom, Hash Embeddings).
4. **Obstacle.** Why the obvious fixes fail:
   - "Use more buckets" lowers the compression.
   - "Learn a per-item scale" gets zero gradient under DirectAU (Prop 1).
   - "Tie-break better" recovers only 12% of the gain.
5. **Insight.** Sharing turns magnitude into a bucket property (Fig 1b). The item-specific part ε of popularity becomes unrepresentable.
6. **Contribution derived from the insight.**
   - Theory: ties, a correlation-ratio bound, zero gradient.
   - Nine competing explanations, each tested.
   - Restoring item counts as the intervention the diagnosis predicts.
   - Its costs.
7. **Contribution bullets.** Three, each stating new knowledge, not an activity:
   - **(i) Diagnosis:** sharing pools magnitude at the bucket level; the learned magnitude becomes useless for ranking.
   - **(ii) Analysis and discriminating tests:** exact ties; a bound on representable popularity; zero gradient for learned scales; capacity, ties, initialisation, stopping, weight decay and MostPop ruled out.
   - **(iii) Intervention with costs:** restoring item counts reverses the ordering on 9 of 9 pairs; 71–76% of the gain is item-specific; a tail and coverage cost.
8. **Scope line:** "We do not claim a new score form, a tail-safe repair, or a universal loss reversal."

**Fig 1 (page 1, two panels):**
- (a) Grouped bars, Gowalla MF. Full and GraphHash × BPR and DAU, raw solid, corrected hatched: 16.55 / 18.02 / 9.37 / 7.55, corrected 11.31 / 13.30.
- (b) Scatter of log(1 + c_i) against ‖q_i‖ for Full DAU (ρ 0.81) and GraphHash DAU (ρ 0.28). Bucket plateaus should be visible as horizontal stripes.
- Caption: "Sharing removes DirectAU's lead because shared magnitude no longer follows the item."

### 2. Setting and protocol (0.75 pages)
- **Notation:**
  - users U, items I, train counts c_i, p_i = log(1 + c_i);
  - s(u,i) = ⟨p_u, q_i⟩;
  - sharing map b: I → B, q_i = E[b(i)]; double hashing uses a sum of two rows;
  - bucket traffic C_b.
- **Losses:** BPR; DirectAU (alignment + uniformity on ℓ2-normalised vectors).
- **Scoring modes:** raw; cosine; cos + βp (the count-restored score); raw + βp. β is selected on validation from a fixed grid.
- **Data and models:**
  - Gowalla, Yelp2018 and AmazonBook, with the GraphHash splits.
  - MF with d = 64; 2-layer LightGCN.
  - Sharing: GraphHash (Louvain, resolution 200), Random, DoubleFreq, DoubleGraph.
  - Table size relative to Full: 16.4% / 21.9% / 13.6%.
- **Fairness of the comparison:**
  - The same table and the same bucket map for both losses.
  - The authors' lr/wd grid for each loss.
  - Converged = stopped by patience; capped runs are flagged.
  - β is chosen on validation; test is scored once.
- **Statistics:**
  - Recall@20 with the min(20, |T_u|) denominator; NDCG secondary.
  - Paired user bootstrap with 2,000 resamples, paired t-test, Holm correction.
  - 3 seeds on the main pairs, † on single-seed rows.
- **Pre-registration:** predictions registered before each stage. Give the tally, with the full registry in the anonymous repository.
- **Tab 1:** dataset statistics, buckets, table ratio, prior bytes.

### 3. The observed failure (RQ1, 0.5 pages)
- **Tab 2 (main grid):** dataset × {Full, GraphHash, Random, DoubleFreq, DoubleGraph} × {BPR, DAU}, with columns raw / cos / corrected (β) / ρ_item / ρ_bucket.
- **Paragraph:**
  - The order reverses on Gowalla and Yelp and the lead shrinks on AmazonBook.
  - Random and DoubleFreq reverse it in 5 of 6 cells.
  - Random is a stress case (raw < 1%).
- **Boundary:** with LightGCN, BPR already leads on the full Gowalla table (18.48 vs 18.02). The premise is MF-specific, so say so here, not only in §7.

### 4. What sharing erases (2.3 pages; the core of the paper)
**§4.1 Theory** (statements here, proofs in App. A):
- **Lemma 1:** a user's ranking does not depend on ‖p_u‖.
- **Lemma 2:** items in a bucket tie exactly. Measured: 87% of users have a boundary tie; 14.7 tied items on average.
- **Proposition 1:** L_DAU depends only on normalised vectors.
  - Corollary: a learned per-item scale (MEmCom's U[i mod m] ⊙ V[i]) gets ∂L/∂w_i = 0.
  - Magnitude is set by optimiser dynamics and decay, which under sharing aggregate over the bucket. DirectAU norms only grow (largest drop −0.15%; BPR drops −19%).
- **Proposition 2:**
  - Write p = μ_b + ε. Then corr(f(b), p)² ≤ η² for every bucket function f.
  - ε is orthogonal to every shared-row representation.
  - Remark: sufficient, not necessary (DoubleFreq).
- **Fig 3 (small, conceptual, labelled as such):** a full table with one arrow per item, next to a shared table with one arrow per bucket, and p = μ_b + ε annotated.

**§4.2 Evidence that magnitude follows buckets (RQ2):**
- ρ_bucket vs ρ_item: Fig 1b plus the Tab 2 columns.
- Quintile lengths for Gowalla GraphHash: 11.71 → 12.71, almost flat.
- η² for GraphHash 0.36 / 0.48 / 0.36. The bound holds: ρ_item² 0.073 / 0.27 / 0.036.
- **Fig 2:** per dataset, Δ(cos − raw) and the prior gain, Full vs shared.
  - Cosine ablation: Full −4.96 / −4.13 / −0.87, shared +0.02 / −0.44 / +0.01.
  - Gowalla prior gain: Full +1.55, shared +5.75.
  - This is the difference-in-differences that answers "the prior helps everything".

**§4.3 Competing explanations (RQ3).** Tab 3, the paper's most important table:

| Alternative | Test | Result | Verdict |
|---|---|---|---|
| A1. Lost capacity (fewer rows) | Erase magnitude on the **full** table (equal norms), capacity intact | Raw 18.02 → 12.77; the prior restores 19.32 | Not capacity |
| A2. Ties only (H_tie) | Tie-break by popularity, no prior | 12% of the gain (DAU); 34% (BPR) | Minor |
| A3. Bucket-level popularity is enough | Fixed-β μ/ε decomposition; within-bucket shuffle of counts | ε share 0.71 / 0.76; shuffle keeps −6% | Item-level is needed |
| A4. Magnitude carries non-popularity information | Replace Full magnitude by a fitted function of popularity | 17.90 → 18.69 (AmazonBook 9.57 → 10.64); global shuffle 6.28 | On full tables magnitude ≈ popularity |
| A5. Under-training or stopping | Every main run stopped by patience; legacy validation flat | — | Not stopping |
| A6. Initial-norm artefact | Init-matched reruns | Δ ≤ 0.05 | Not initialisation |
| A7. Weight decay artefact | wd = 0 on Full DAU | ρ 0.74 retained | Not weight decay alone |
| A8. It is just MostPop | Popularity-only ranking | 2.97 / 1.68 / 0.96 vs 13.29 / 5.68 / 6.33 | Not MostPop |
| A9. A learned scale would fix it | Prop 1 | Zero gradient | Excluded for DirectAU by theory |

- **Result paragraph per row:** question → contrast → observation → interpretation → boundary.
- **A4 boundary:** decile shuffle −4.19. Coarse bins lose information inside the top decile. Report this as the failed prediction it was.

### 5. Restoring item counts (1.75 pages)
- **§5.1 Effect (RQ4).**
  - Tab 4: 9 seed pairs with Δ, 95% CI, Holm p, and NDCG Δ.
  - Gowalla mean ± SD: 13.325 ± 0.024 vs 11.257 ± 0.063.
  - Tie-policy robustness on AmazonBook: +1.508 to +1.524 across 13 policies.
  - The restored score recovers **55%** of the raw Full-vs-shared gap on Gowalla.
- **§5.2 Generality (RQ5).** Tab 5 (top) and Fig 4:
  - DoubleGraph: corrected DAU vs BPR 10.91 vs 7.27 (Gowalla), 5.52 vs 2.78 (Yelp).
  - Resolution sweep: the gain falls from 5.75 to 4.99 (Spearman −0.90). Corrected DAU beats BPR at every resolution.
  - LightGCN + GraphHash: raw DAU < BPR (11.29 vs 16.22; 4.71 vs 6.96); corrected DAU wins (17.06 vs 16.27; 9.12 vs 6.96). Capped rows marked.
- **§5.3 Cost (RQ6).** Tab 5 (bottom). This is what the intervention must preserve:
  - **Storage:** float32 prior +5.52 / 3.90 / 7.30% of the compressed table. uint8 is not universal.
  - **Tail recall:** below the Full DAU raw reference on all 3 datasets (Gowalla 2.01 vs 2.50, CI [−0.68, −0.33]; Yelp 0.29 vs 1.03; AmazonBook 3.78 vs 5.41).
  - **Head share:** 89–98% of the added hits are head items.
  - **Coverage:** Gowalla 95.2 → 28.4%; Yelp 96.3 → 21.7%; AmazonBook 93.1 → 80.7%.
  - **Weighted recall:** the ordering is kept (Gowalla +2.14 to +2.34; Yelp +0.82; AmazonBook 5.53 vs 4.04).
  - **Context:** Cheng reports the same coverage drop on full tables (0.985 → 0.726). The cost belongs to the prior, not to sharing.

### 6. Related work (0.65 pages; each paragraph ends with what the cited work does not do)
1. **Embedding table compression.**
   - Hashing trick, frequency/double hashing, Learning to Collide, CAFE, MEC, CCE, GraphHash, BACO, GDE.
   - Gap: these optimise the bucket assignment; none asks what sharing does to magnitude.
2. **Magnitude, popularity, and losses.**
   - PRISM, Adap-τ, DDC, DirectAU, Wu et al. (TOIS'24), DNA, CoRRe.
   - Gap: full tables only.
3. **Per-item terms and score-time popularity.**
   - MEmCom, Hash Embeddings, Seol et al. (WSDM'24), logit adjustment, logQ, PDA, MACR, Cheng eq. 53.
   - Gap: no link to sharing, and no account of why a learned scale cannot help DirectAU.

### 7. Limitations and scope (0.35 pages; each limitation changes how a result should be read)
- **Flip scope:** the claim is that the advantage is removed. The reversal is not universal.
- **Tail and coverage cost:** a head-accuracy correction.
- **Seeds and caps:** † single-seed rows; capped LightGCN rows.
- **Failed predictions:** Full-BPR range; Yelp Y3; uint8 storage; decile shuffle; ρ_item vs resolution trend. Point to App. E for the full list.
- **Learned end-to-end variant (C3):** report it only if the code review clears it; otherwise state "not evaluated".
- **Not tested:** soft gating (GDE), BACO itself, CTR models, DNA and MEmCom baselines, online effects.

### 8. Conclusion (0.2 pages)
- Answer the title: sharing saves memory and costs item identity in the popularity signal.
- Practical rule: compare training losses under sharing only after restoring item counts.
- Future work: popularity-aware bucketing.

---

## 5. Figures and tables (main text: 4 figures, 5 tables)

| ID | Job | Data | Status |
|---|---|---|---|
| Fig 1a | Symptom + intervention, Gowalla bars | CSV | data ready |
| Fig 1b | Magnitude follows buckets (scatter) | Checkpoints + **stored** mappings; never recompute on CPU | **Confirm before G4 (Fri 18:00) that the per-item norms exist**; otherwise one GPU job |
| Fig 2 | Cosine ablation + prior gain (difference-in-differences) | CSV | data ready |
| Fig 3 | Conceptual schematic (labelled conceptual) | — | to draw |
| Fig 4 | Gain vs resolution | S4-C1 | data ready |
| Tab 1 | Setting | data + E7 bytes | fill bucket counts |
| Tab 2 | Main grid | CSV | data ready |
| Tab 3 | Competing explanations | RESULTS blocks | data ready |
| Tab 4 | 9 paired comparisons | PR-E8 + S4-SEED | data ready |
| Tab 5 | Generality + costs | S4-DG, S4-C1, S4-LGN, S5, E5, E7, E9 | data ready |

**Caption rule:** state the setting, the comparison, and the one supported observation.

---

## 6. Appendix (about 2.8 pages) and the anonymous repository

Space is the constraint. The appendix shows **every run in one dense table** and the key diagnostics. Everything else goes in an anonymised repository linked once. The CFP does not mention supplementary material, so check the OpenReview form.

| App. | Content | Pages |
|---|---|---|
| A | Proofs: Lemmas 1–2; Prop 1 and corollary, with the MEmCom form; Prop 2 and the decomposition; the Euler argument for monotone norm growth | 0.8 |
| B | Protocol details: per-loss lr/wd/batch/patience; β grids, with the grid-edge cases (β = 5.0); evaluation masks; denominators; evaluated users (25,789 / 30,304 / 50,407); bootstrap and Holm details; compute (T4 runtimes; 2-GPU split gave no speedup) | 0.4 |
| C | **Complete run ledger.** All ~75 training runs in one table: run_id, stop reason, best/run epochs, raw, cos, corrected (β), ρ_item, ρ_bucket, flags (†, capped, legacy, excluded). It includes the reproduction rows (GraphHash BPR 9.34 vs 9.30 published), init controls, wd variants, equal-norm runs, the resolution sweep, LightGCN Stage 4 and 5, popularity-only, and the 4 excluded runs with reasons | 1.0 |
| D | Extra diagnostics: η² for every scheme and resolution (0.49 / 0.63 / 0.79 / 0.92); quintile lengths; norm dynamics; ρ_dir (BPR > DAU, 2 of 2); tie statistics per checkpoint (12–60% tie-only share); μ/ε decomposition for every checkpoint; AmazonBook E2e; activity-quartile gains | 0.4 |
| E | Failed predictions with numbers, and the pre-registration tally | 0.2 |

**Anonymous repository (anonymous.4open.science or similar), linked once in §2:**
- the full `master_metrics_v2.csv`;
- the RESULTS outcome blocks;
- PREREG with snapshot hashes;
- E5/E7/E9 JSON;
- training and scoring code.

**[FLAG] Do not link the current repository.** It contains Kaggle slugs, session names and lineage tags that can identify the author. Build a scrubbed export:
- strip `runs/_sessions`, `sessions_*.py` and every account or slug string;
- rewrite the git history (or export without history);
- run a grep for usernames before uploading.

---

## 7. Wording

| Use | Avoid |
|---|---|
| embedding table compression, parameter sharing, bucket | "sharing rows" |
| magnitude (ℓ2-norm), direction | "length" in formal text |
| intra-bucket score ties; bucket-level magnitude pooling | invented acronyms |
| count-restored score; post-hoc; without retraining | "our method", "novel correction" |
| removes DirectAU's advantage | "reverses the losses" as a universal claim |
| head-accuracy correction with a tail cost | free, tail-safe, fixes |
| float32 prior, +3.9–7.3% of the compressed table | one byte per item |
| percentage points (pp) | bare "%" for differences |
| Cheng (2026, eq. 53) score form | anything implying the formula is ours |

---

## 8. Writing order

| Day | Work |
|---|---|
| Thu 22 | Fill the claim ledger into text: §4 (theory + Tab 3), §2 |
| Fri 23 | §3, §5. **Confirm the Fig 1b data.** Numbers freeze at 18:00 |
| Sat 24 | §1 (written after the results), §6, §7; App. A–C |
| Sun 25 | Abstract, §8, App. D–E; read against §1 line 7 and §2; trim to 8 + 4 |
| Mon 26 | Scrubbed anonymous repository; anonymity grep on PDF metadata; submit by 15:00 |
