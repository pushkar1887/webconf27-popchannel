# PLAN: *What Embedding Sharing Destroys, and the Few Bytes That Repair It*
### Loss × compression in Web recommenders (WebConf 2027, Research Track). Working title; final title decided at G2

| | |
|---|---|
| **Version** | v3, Sat 17 Oct 2026, 17:30 IST. Owner: Pushkar (+ co-author). Orchestrator: Claude Code |
| **What changed from v2** | Reframed after the novelty check found **PRISM (KDD 2025)** and **DNA (2026)**. Math corrected. Mechanisms are now *competing hypotheses* tested by cheap controls. Work reordered into **3 stages** (§5). Details in the LOG §11 |
| **Deadlines** | Abstract **Mon 19 Oct, 17:29 IST** (authors frozen after it). Paper **Mon 26 Oct, 17:29 IST** |
| **Hardware** | Kaggle **GPU T4 x2** sessions: 2 × Tesla T4 per session, **one condition per GPU**. 2 accounts × 2 sessions × 2 GPUs = 8 parallel conditions |
| **The only 4 docs** | `PLAN.md` (this file; **the shared LOG is §11**), `TASKS.md` (agent rules, then the board), `PREREG.md` (predictions before runs), `RESULTS.md` (outcomes) |

> **Confidence tags:** **[Certain]** measured, read from code, or proven. **[Likely]** strong inference. **[Guessing]** estimate.
> **Status of each statement in §4:** **Theorem/Identity** (proven), **Hypothesis** (to be measured), **Heuristic** (exploratory model, not evidence).
> **IDs:**
> - **K** = paper claim; **S1/S2/S3** = stage; **A, B, C, D, I, X** = runs; **E, M** = analyses; **T** = code; **G** = gate; **N** = novelty; **W** = writing.
> - IDs are identical in all 4 docs.

---

## 1. The story in plain words

**The setting.** A Web recommender keeps one learned vector, an *embedding*, per user and per item; a match score is a dot product. With millions of IDs these tables dominate memory, so production systems **share** rows: many items use one vector. *GraphHash* (WWW 2025) picks who shares by clustering the user–item graph.

**The puzzle (published, unexplained).**
- **BPR** makes clicked items score above random ones.
- **DirectAU** trains on **length-normalized** vectors: *alignment* pulls users to their items, and *uniformity* spreads vectors over the sphere.
- On full tables DirectAU wins (Gowalla: 17.98 vs 15.62 Recall@20). Under GraphHash it **loses** on Gowalla (7.66 vs 9.30) and Yelp, yet **wins** on AmazonBook. The GraphHash paper reports this without a reason.

**What is already known (so it is *not* our contribution):**
- **PRISM** (Loveland et al., KDD 2025): with weight decay, DirectAU's vector lengths track item popularity. In-batch updates grow magnitudes, weight decay shrinks them, and the result is a log-linear link. Proposes popularity-aware magnitude initialization.
- **DNA** (Choi et al., *Applied Soft Computing* 2026): normalize item vectors, then add a learned item-specific bias to restore popularity (GCN recommenders).
- **Logit adjustment / logQ correction / PDA:** adding or subtracting log-popularity at scoring time is known machinery.
- None of these studies **embedding sharing**. We checked the visible text of both papers; the full PRISM appendix is still to be read (N1).

**What is *not* known, which is our question:** what exactly does sharing destroy, how can it be measured, and how little per-item information repairs it? And does that explain the DirectAU-vs-BPR flip?

**Two competing explanations of what sharing destroys** (Stage 1 measures which one dominates):
1. **Ties (H-tie).** Items in one bucket have *identical* vectors, so they get *identical* scores for every user (Lemma 2). On Gowalla a bucket holds about 7.1 items. Which tied items enter a top-20 list is decided by arbitrary index order, so any per-item signal (like popularity) resolves ties usefully.
2. **Bucket pooling of the length signal (H-pool).** In full DirectAU, vector length tracks popularity (ρ = 0.82, consistent with PRISM). Under sharing, a row's length reflects its whole bucket, so the item-level signal is lost (ρ = 0.27). Raw-dot scoring then cannot reward popular items.

**The repair we study:** `score = cos(u,i) + β·log(1 + count_i)`. One scalar β (tuned on validation) plus one stored number per item (float32 now; 1 byte if E7 confirms). No retraining.
- Compressed DirectAU on Gowalla goes 7.56 → **13.29**, and beats BPR given the same repair (11.24). AmazonBook: 6.33 vs 4.81.

**Why the Web conference cares.** Compression is how Web-scale recommenders fit in memory. We show which information sharing removes, which part of it matters, and that a few bytes per item restore it. This also explains a published, unexplained result.

**Planned claims.** Each lists what would falsify it.

| # | Claim | False if… | Evidence |
|---|---|---|---|
| K1 | The repair's gain under sharing decomposes exactly into **bucket calibration** + **within-bucket ordering**, and we identify which dominates (ties vs bucket pooling) | (It's a measurement; the framing follows the result) | S1 controls (E-dec, E-tie) |
| K2 | Compressed DirectAU + repair **beats** compressed BPR + the same repair | Δ ≤ 0 on any dataset, or the Holm-corrected CI includes 0, or any Gowalla seed is negative | S2/S3 grid, E8 |
| K3 | The gain requires popularity attached to the **right item**, not just more popular recommendations | Within-bucket shuffle keeps most of the gain; popularity-only is close | E-dec (shuffle), A6, E5, E9 |
| K4 | A per-bucket quantity can represent only an η² share of popularity variance (**theorem**); η² is low for real mappings | (Theorem; the measurement is η² per mapping) | M1 |
| K5 (exploratory) | DirectAU depends more on the length channel than BPR does, which explains why it suffers more | Removing length information (E2e) costs DirectAU no more than BPR | E2e, E2c, Yelp |

---

## 2. What we know so far (measured 16–17 Oct) [Certain]

Test Recall@20 (%). "+prior" means β was tuned on validation and test scored once. ρ = Spearman(item length, training count). Full detail in RESULTS.md.

| Dataset | Model | Raw dot | Cosine | Cos + prior (β) | ρ | Caveat |
|---|---|---|---|---|---|---|
| Gowalla | Full × DirectAU | 17.90 | 12.99 | 19.54 (0.1) | 0.815 | capped at epoch 120 (replacement planned) |
| Gowalla | GraphHash × DirectAU, 3 seeds | 7.55 ± 0.02 | 7.56 ± 0.02 | **13.29 ± 0.03** (0.2) | 0.27 | 30 epochs (convergence pilot planned) |
| Gowalla | GraphHash × BPR | 9.34 | 6.08 | raw+prior 11.24 (0.5) | 0.054 | converged |
| Gowalla | Popularity only | — | — | 2.97 | — | — |
| AmazonBook | Full × DirectAU | 9.57 | 8.70 | 11.05 (0.1) | 0.786 | patience-10 stop |
| AmazonBook | GraphHash × DirectAU | 4.63 | 4.64 | **6.33** (0.1) | 0.189 | 75-min cap, epoch 41 |
| AmazonBook | GraphHash × BPR | 4.10 | 2.53 | raw+prior 4.81 (0.2) | 0.084 | converged |

**Negative interventions:**
- **Unique-bucket dedup** inside uniformity gave no gain (paired seed differences −0.02 / +0.08 / +0.05). This does **not** rule out collision effects, because dedup also changes the pair weighting.
- **A freely learned per-item bias** failed (4.32): its unconstrained scale swamped the cosine term.

**Discrepancy with PRISM (to explain, not to claim):** PRISM reports that DirectAU's magnitude–popularity correlation is **weak without weight decay**. We measure ρ = 0.815 at wd = 1e-8 (effectively zero) with Adam, lr 0.01. Possible causes: optimizer, learning rate, training length, initialization. N1 compares settings; the optional run D7 checks it directly.

---

## 3. Gaps and limitations: what we found and what we do about each

### 3.1 Gaps in published work
| Gap | Status in the literature | How we address it |
|---|---|---|
| The DirectAU-vs-BPR flip under compression has no explanation | Reported by GraphHash, not explained | K1, K5 |
| What sharing removes (ties vs pooled length) has never been decomposed | — | Exact decomposition + tie controls (§4.6, S1) |
| A formal limit on what a shared table can express about item popularity | — | Capacity bound (§4.5, K4) |
| Length ↔ popularity in DirectAU | **Known (PRISM)** | Cited; we build on it |
| Normalize + item bias | **Known (DNA)**, non-compressed GCN setting | Cited; our learned bias variants are ablations |

### 3.2 Weaknesses in our own evidence
| # | Weakness | Why it matters | Fix | Stage |
|---|---|---|---|---|
| L1 | Compressed DirectAU is 30-epoch; Full DirectAU capped | Early stopping makes DirectAU look worse, which *favours* the anomaly | Convergence pilot A1; A2 later | S2 / S3 |
| L2 | **Initialization differs**: Xavier variance depends on table size, so GraphHash rows start 2.64× longer (item RMS 0.148 vs 0.056) | The anomaly could partly be an init effect | **I1** (init-sensitivity control); controlled Full-vs-GraphHash pair = I1 vs A2 under the same stopping rule | S2 / S3 |
| L3 | **Tie confound**: identical vectors produce exact ties, and tie order follows item index | The prior may "work" mainly by breaking ties; default tie order may already be non-neutral | Tie statistics + random-tie baseline + tie-break-only control | S1 |
| L4 | Mechanism evidence is correlational | Confounding | Length permutation/replacement on existing checkpoints (E2e); equal-length training (C4) optional | S1 / S3 |
| L5 | Author hyperparameters differ across compressions for BPR | Author-config comparisons don't isolate sharing | Separate **controlled** comparisons (same lr/wd/batch/budget/init) from **reproduction** tables | all |
| L6 | Only 2 datasets | Weak generalization | Yelp in S2 | S2 |
| L7 | Single seed outside Gowalla | Training noise | Seed replication where claims rest (Gowalla); user bootstrap answers a different question (§6) | S3 |
| L8 | Random splits favour popularity | "Just popularity bias" | Within-bucket shuffle, popularity-only, popularity-reweighted evaluation | S1 / S3 |
| L9 | Old checkpoints lack saved mappings | Kaggle CPU image renumbers buckets | Score on the GPU image; assert equality with the stored mappings of P4/P5 | S1 |

### 3.3 Limitations that will remain
- The repair is post hoc.
- Random 80/10/10 splits (no timestamps in the processed data).
- MF backbone (LightGCN optional).
- Offline logged data.
- The boundary model (erasure index) is calibrated on few datasets and stays exploratory.

---

## 4. Theory: what is proven, what is hypothesis

### 4.1 Setup
- c_i = training interactions of item i; p_i = log(1+c_i).
- Mapping a: item → bucket (for double-frequency: an **alias pair** (a₁(i), a₂(i)), with v_i = θ_{a₁} + θ_{a₂}).
- Item vector v_i = θ_{a(i)}; r_i = ‖v_i‖; hats denote unit vectors.

**Scores:**
$$s_{\text{raw}}=e_u^\top v_i,\quad s_{\cos}=\hat e_u^\top\hat v_i,\quad s_{\text{prior}}=\hat e_u^\top\hat v_i+\beta p_i.$$

**Losses:**
$$\mathcal L_{\text{BPR}}=-\sum\log\sigma(s_{\text{raw}}(u,i)-s_{\text{raw}}(u,j))$$
$$\mathcal L_{\text{DAU}}=\mathbb E\|\hat e_u-\hat v_i\|^2+\tfrac{\gamma}{2}\big[\log\mathbb E e^{-t\|\hat e_u-\hat e_{u'}\|^2}+\log\mathbb E e^{-t\|\hat v_i-\hat v_{i'}\|^2}\big],\quad t=2,\ \gamma=1.$$

**Metric:** R_u = |Top20(u) ∩ T_u| / min(20, |T_u|); Recall@20 = mean of R_u over evaluated users.

### 4.2 Lemma 1 (Identity): a user's own length never changes that user's ranking
$$e_u^\top v_i=\|e_u\|\,r_i\cos\theta_{ui},$$
and ‖e_u‖ is constant across items. Any per-item length effect therefore lives in r_i.

### 4.3 Lemma 2 (Identity): sharing creates exact ties
For a 1-D mapping, a(i) = a(j) implies v_i = v_j, so s(u,i) = s(u,j) for **every** user, under both raw and cosine scoring.
- Top-20 membership among tied candidates is decided by the sort's tie order, which carries no relevance information unless index order happens to correlate with something.
- **Any** per-item term (such as βp_i) reorders tied items.
- To change **only** the tie order, do not add a tiny β·p: even 10⁻⁶·p (up to ~7e-6) can reorder unequal but close scores. Instead:
  1. compute scores **per bucket** (user × bucket) and copy them to items, so ties are exact by construction;
  2. sort **lexicographically**: score first, popularity second (random key for the neutral baseline).
- For double-frequency hashing, ties occur only between items with the same alias pair.
- Measured quantity (E-tie): the share of users whose 20th slot falls inside a tied group, and the size of that group.

### 4.4 Proposition 1 (Theorem) and the growth approximation
DirectAU sees each row only through its direction, so L(cx) = L(x). Euler's theorem for degree-0 homogeneous functions gives

$$x^\top\nabla_x\mathcal L=0,\qquad \|\nabla_x\mathcal L\|=\frac{\|P_\perp\nabla f(\hat x)\|}{\|x\|}.$$

- **Plain SGD, no weight decay (exact):** ‖x − ηg‖² = ‖x‖² + η²‖g‖² (Pythagoras), so lengths never shrink.
- With ‖g‖ = G/r and a = η²G², the **exact** recurrence is

$$r_{k+1}^4=r_k^4+2a_k+\frac{a_k^2}{r_k^4}.$$

  When a_k ≪ r_k⁴ this gives r_n⁴ ≈ r_0⁴ + 2Σ a_k, i.e. r ∝ n^{1/4} for constant G. **This is an approximation under those assumptions.**
- **Adam (our optimizer): Hypothesis H-Adam.** Elementwise rescaling breaks exact orthogonality, so no growth law follows from the loss. We *measure* length growth per popularity quintile (E3) and report a fitted slope γ̂ of log r on log(1+c) as a **descriptive** number.
- Related results: PRISM (in-batch growth plus weight-decay shrinkage), AdamP (momentum inflates norms of scale-invariant weights), spherical motion dynamics (equilibrium under weight decay).

### 4.5 Theorem 2 (capacity bound): what any per-bucket quantity can represent
Let B be the grouping variable: the bucket for 1-D maps, the **full alias tuple** for multi-alias maps. Let X = rank(c_i) or p_i over items, with Var(X) > 0 and Var(g(B)) > 0. Then for **any** g:

$$\operatorname{Corr}(X,g(B))^2\le\eta^2_{X\mid B}=\frac{\operatorname{Var}(\mathbb E[X\mid B])}{\operatorname{Var}(X)}.$$

**Proof.** Cov(X, g(B)) = Cov(E[X|B], g(B)) by the tower property. Cauchy–Schwarz then gives Cov² ≤ Var(E[X|B])·Var(g(B)). Divide by Var(X)·Var(g(B)). ∎

- **What it says:** a bucket's length, a learned bucket bias, a per-bucket prior — none can **correlate** with item popularity beyond η². η² is computable from the mapping and counts alone (M1).
- **What it does not say:**
  - It does not bound Recall@20.
  - It does not imply that a bucket-level prior cannot improve recommendation.
  - It does not make c → bucket traffic → length a Markov chain.
  - It does not imply that more Louvain buckets give monotone gains: partitions at different resolutions are not nested.

### 4.6 Identity 3 (exact decomposition of the repair)
$$p_i=\mu_{a(i)}+\varepsilon_i,\qquad \mu_b=\operatorname{mean}_{j\in b}p_j,\qquad \textstyle\sum_{j\in b}\varepsilon_j=0.$$

So s_prior = cos + βμ_{a(i)} + βε_i.
- The **bucket component** βμ is a per-bucket function, covered by Theorem 2 with η² = Var(μ)/Var(p).
- The **residual** ε is orthogonal to *every* bucket-level function (E[ε|B] = 0), so a shared table can never represent it. The empirical variance splits exactly: Var(p) = Var(μ) + Var(ε) (law of total variance).

Note that **mean of log ≠ log of mean.** The bucket control uses μ_b, the mean of p.

**Controls built on the identity** (S1, no training):

**(a) Ranking controls** (scores computed per bucket, so ties are exact):

| Control | Ranking | Isolates |
|---|---|---|
| Neutral baseline | Sort by cos; exact ties broken by a **random** key (10 seeded draws) | Removes any accidental index-order effect |
| Tie-break only | Sort by cos; exact ties broken by **popularity** p_i (lexicographic) | Value of resolving exact ties, and nothing else |
| Within-bucket shuffle | cos + β*·p_{π(i)}, with π permuting items **within** each bucket | Keeps μ exactly, destroys item attachment (K3) |

**(b) The score decomposes exactly; Recall does not. Use a 2×2 design at fixed β.**
- Fix β* = the full prior's validation-tuned β.
- Evaluate four cells: cos (neither), cos + β*μ (bucket term), cos + β*ε (residual term), cos + β*p (both = full repair).
- From these, compute the main effect of each term, the **interaction** = Δ_both − Δ_bucket − Δ_res, and two-factor Shapley shares (each term's effect averaged over both orders).
- These fixed-β results are **primary**. Versions with β re-tuned per control are reported separately as secondary.
- Test is scored once per cell.

### 4.7 Calibration reading of the repair (motivation, not derivation)
Bayes' identity: log p(i|u) = [log p(u|i) − log p(u)] + log p(i).
- **If** cosine behaves like a scaled, popularity-free affinity, then cos + β·log p(i) is the natural combined score, with β a scale.
- **Assumptions (not established):** cosine is calibrated and popularity-free; p(i) ∝ 1 + c_i. β is **tuned**, not derived.
- **Sign:** debiasing methods (logit adjustment) subtract log-popularity. We add it because random-split test data follow training popularity. Under popularity-reweighted evaluation (E9) the tuned β is expected to shrink.

### 4.8 Hypotheses and heuristics (exploratory; never used as proof)
| Name | Statement | Status | Test |
|---|---|---|---|
| H-tie | Most of the repair's gain under sharing comes from resolving exact ties | Hypothesis | E-tie, tie-break-only |
| H-pool | A bucket's length grows with its traffic N_b = Σ c_i. Gradients can **cancel** (‖Σg_i‖² = Σ‖g_i‖² + 2Σ_{i<j} g_iᵀg_j), so this is not guaranteed | Hypothesis | E2b (ρ_bucket) |
| H-Adam | Under Adam, mean r² grows roughly linearly with update count | Hypothesis | E3 |
| H-dir | BPR keeps popularity partly in *direction*, DirectAU less so | Hypothesis (uniformity does **not** imply it) | E2c (ρ_dir) |
| Product rule | ρ_item(compressed) ≈ ρ_proxy·ρ_full, with ρ_proxy = Spearman(c_i, N_{a(i)}) | Heuristic (Spearman correlations do not multiply in general) | M1 + E2d, exploratory |
| Erasure index | E = G·(1 − ρ_c²/ρ_f²) predicts DirectAU vs BPR under sharing (G = Full-DirectAU val raw−cos gap) | Heuristic, 2 calibration points (Gowalla 4.45, AmazonBook 1.03) | Yelp, exploratory |

---

## 5. Work plan in 3 stages (what, why, how, expected)

Numeric predictions and decision rules: `PREREG.md` under the same IDs. Session layouts: `TASKS.md`.

### Stage 1: no training; existing checkpoints and mappings (Sat ~19:00 → ~20:30)
**Why first:** it decides **which mechanism leads** (ties vs bucket pooling vs item residual) before money is spent on a large campaign. It costs about 1 GPU session.

| ID | What | How | Expected |
|---|---|---|---|
| **E-tie** | Tie statistics + tie-break-only | For GraphHash checkpoints, with bucket-level scoring: share of users whose 20th slot sits in a tied group; mean tied-group size; Recall with random-key vs default vs **popularity-key lexicographic** tie breaking | Large tie share [Guessing ≥ 50% of users]; random vs default order differ by < 0.2 [Guessing] |
| **E-dec** | 2×2 fixed-β decomposition + within-bucket shuffle (§4.6) | On GraphHash × DirectAU Gowalla s0/s1/s2, GraphHash × BPR Gowalla, AmazonBook GraphHash × DirectAU/BPR, Gowalla Random/DF × DirectAU. Fixed β* (primary), re-tuned (secondary); BPR uses raw dot as its base | Descriptive. **Framing follows the largest Shapley share** (§9). Within-bucket shuffle loses ≥ 30% of the gain [Guessing] |
| **E2e** | What the length carries (Full models) | On Full × DirectAU Gowalla (epoch-120 checkpoint) and AmazonBook: shuffle lengths within popularity deciles; shuffle globally; replace lengths by exp(â + γ̂p_i). Validation metric; test once. Same for Full × BPR when available | Within-decile shuffle \|Δ\| ≤ 0.5; global shuffle costs DirectAU much more than its decile shuffle |
| **M1** | Mapping statistics (model-free) | For GraphHash, Random, DF on Gowalla and AmazonBook: items/bucket distribution, η² by alias tuple, Var(μ)/Var(p), ρ_proxy | η² for GraphHash well below 1 [Likely]; the theorem check ρ_item² ≤ η² holds (code check) |
| **A6** | Popularity-only, Yelp and AmazonBook | Local CPU | Far below the repaired models |
| **N1** | Novelty | Read PRISM (full, incl. appendix) and DNA; compare their settings to ours; scan papers citing GraphHash, PRISM, DNA | Decides G1.d |

### Stage 2: small training campaign (Sat ~19:00 → ~22:30, parallel with Stage 1)
**Why:** convergence, an initialization control and a third dataset are needed whatever Stage 1 shows. Caps are kept small (≤ 3 h).

| ID | Run | Why | Expected |
|---|---|---|---|
| A0 | Smoke on T4 x2 | Verify `device_count = 2`, file contract, co-location speed | — |
| **A1** | GraphHash × DirectAU Gowalla s0, to convergence (cap 3 h) | Is the anomaly real at convergence? (L1) | raw ≤ 9.04, i.e. still below BPR [informed by seen results]. If the cap is hit: **INCONCLUSIVE** for convergence (the gap only "persists within budget") |
| **I1** | Same as A1, **rows rescaled at init to Full's RMS norms** (items × 0.378, users × 0.442) | **Initialization-sensitivity control for GraphHash** (L2). It does **not** by itself isolate sharing: the controlled Full-vs-GraphHash comparison is I1 vs A2 (Full × DirectAU) trained under the same stopping rule and budget | Within ±0.3 of A1 [Guessing] |
| **A3, A4** | Yelp Full × DirectAU, GraphHash × DirectAU (cap 3 h) | Third dataset; Full gap; erasure index (exploratory) | DirectAU < BPR under GraphHash (published) |
| **A5, A7** | Yelp GraphHash × BPR, Full × BPR | Fair repair comparison on Yelp | A4 + prior > A5 + prior [prospective] |
| **A8** | Gowalla Full × BPR | Reproduce the paper's 15.62 | ∈ [15.4, 15.8] |
| **C0** | Mapping job for Yelp (+ Gowalla resolutions for later) | M1 on Yelp; inputs for S3 | — |

### Stage 3: scope chosen at G2 from Stage 1–2 results (Sat night → Sun)
Candidates, in priority order. The final list and predictions are registered before the push.
1. **A2:** Full × DirectAU Gowalla, under the same stopping rule and budget as I1. It replaces the capped checkpoint and is **the controlled partner of I1**.
2. **B9:** AmazonBook GraphHash × DirectAU converged.
3. **B5–B8:** Gowalla GraphHash seeds 1–2 for both losses.
4. **Main grid:** Random/DF × {BPR, DirectAU} on 3 datasets (B1–B4, B10–B18).
5. **I2:** init-sensitivity control on AmazonBook. A controlled Full-vs-GraphHash pair there also needs a new Full run under the same stopping rule (lr/wd already match).
6. **C3:** regularized residual bias. Score = cos + βp_i + r_i, with r_i under L2.
7. **C1:** budget sweep at other Louvain resolutions (exploratory; non-nested partitions).
8. **C4:** equal-length training intervention (optional; optimizer dynamics change, so it is not decisive).
9. **D-runs:** LightGCN (D1), BACO (D2), the PRISM discrepancy check (D7: Full × DirectAU with wd = 0 and wd = 1e-4).

**Analyses after Stage 3:**
- E1 main table; E2a–c, E3 lengths; E5 who gains; E7 uint8 (makes the "1 byte" claim conditional on its result); E8 statistics.
- **E9 popularity-reweighted evaluation**, reporting both the original β and a re-tuned β. Note that its weighted recall reduces to *standard* recall when weights are equal, unlike the main capped metric.

### Alternative explanations and how each is handled
| Alternative | Handling |
|---|---|
| Undertraining | A1 convergence pilot |
| Initialization scale | I1/I2 (sensitivity) + the controlled pair A2 vs I1 |
| Tie order of the sort | Bucket-level scoring + random-key lexicographic baseline (E-tie) |
| Generic popularity boost | Within-bucket shuffle, popularity-only, E9 |
| Hash collisions in uniformity | Dedup intervention gave no gain; reported, not called "ruled out" |
| Hyperparameter differences | Controlled comparisons separated from author-config reproductions |
| Metric artifact | Standard recall in the appendix |
| Training noise | Seeds (Gowalla), reported separately from user-sampling CIs |
| Mapping bugs | Stored mappings; equality checks on the GPU image |

---

## 6. Statistics protocol
1. **Two different uncertainties.** Report both; neither replaces the other.
   - **(a) Evaluation users:** paired user bootstrap, 10,000 resamples. Δ̂ = mean over users of (R_u^A − R_u^B), with a 95% percentile CI.
   - **(b) Training randomness:** retraining seeds, reported as mean ± sd and as paired differences by seed.
   - A claim resting on Gowalla needs **every seed's** Δ > 0 **and** each seed's user-bootstrap CI excluding 0.
2. **Evaluated users** [Certain]: Gowalla 25,789 test / 25,738 val; AmazonBook 50,407 test / 50,355 val; Yelp counted in S2.
3. **Expected precision of (a)** [Guessing sd(d_u) ≈ 0.25; measured in S1]: SE ≈ 0.156 points (Gowalla) and 0.111 (AmazonBook). The minimum detectable difference at 80% power and α = 0.05 is ≈ 2.8·SE ≈ 0.44 / 0.31 points.
4. **Multiple testing:** the primary claim K2 is one test per dataset (3 tests), with Holm–Bonferroni at familywise α = 0.05. Everything else is reported as estimates with CIs.
5. **Correlations:** Spearman ρ with a 2,000-resample item-bootstrap CI. Differences between models on the same items use a paired item bootstrap.
6. **Descriptive fits:** the length–popularity slope γ̂ by least squares (Theil–Sen as a check), labelled descriptive.
7. **Preregistration:**
   - Each PREREG entry is labelled **[P] prospective** (no related test results seen) or **[I] informed** (made after seeing related test results; only limited confirmatory value).
   - Before each stage's first push, PREREG.md is snapshotted to `results/snapshots/PREREG_<YYYYMMDD-HHMM>.txt`, and its sha256 is written in the LOG.

---

## 7. Setup: datasets, hyperparameters, compute

### 7.1 Datasets (GraphHash's processed files, random 80/10/10)
| | Gowalla | Yelp2018 | AmazonBook |
|---|---|---|---|
| Users / Items | 29,858 / 40,981 | 31,668 / 38,048 | 52,643 / 91,599 |
| Interactions | 1,027,370 | 1,561,406 | 2,984,108 |
| Train / Val / Test | 821,896 / 102,737 / 102,737 | ≈1.249M / 156k / 156k [Likely; recorded in S2] | 2,387,286 / 298,410 / 298,412 |
| Evaluated test users | 25,789 | recorded in S2 | 50,407 |
| GraphHash buckets U / I (res 200) | 5,792 / 5,806 (≈7.1 items/bucket) | from C0 | 9,808 / 9,804 (≈9.3) |
| Params Full → GraphHash | 4.53M → 0.742M | 4.46M → from C0 | 9.23M → 1.255M |
| Xavier initial row norm, items Full / GraphHash | 0.0558 / 0.1477 | from C0 | 0.0374 / 0.1139 |
| DirectAU s/epoch (one T4) | 37–39 | ≈57 [Guessing] | 112–119 |
| BPR s/epoch (full batch) | 0.22 | ≈0.3 [Guessing] | 0.41 |

Archive sha256 `b6098e3b…894a8`. GraphHash commit `7a1d5eb`.

### 7.2 Hyperparameters
**Global:**
- MF, dim 64, Xavier init (I-runs rescale it).
- Adam (0.9, 0.999, 1e-8) with weight decay wd.
- DirectAU γ = 1, t = 2, batch 1024. BPR full batch with one uniform negative.
- GraphHash Louvain resolution 200, mapping seed 0. Random = id mod #buckets. Double-frequency = authors' code, cutoff 0.5. Budgets equal GraphHash's bucket counts.
- Checkpoints chosen on raw-dot validation Recall@20.
- β grid {0, .01, .02, .05, .1, .2, .5, 1, 2, 5}.

**Per-condition lr / wd** (authors' `commands.sh`; used for **reproduction** tables):

| Dataset | Compression | BPR lr / wd | DirectAU lr / wd |
|---|---|---|---|
| Gowalla | Full / GraphHash / Random / Double-freq | 5e-3/1e-8 · 1e-2/1e-6 · 1e-3/1e-6 · 1e-3/1e-6 | 1e-2/1e-8 · 1e-2/1e-8 · 1e-2/1e-6 · 5e-3/1e-6 |
| Yelp2018 | same order | 1e-2/1e-6 · 5e-3/1e-6 · 1e-3/1e-6 · 1e-3/1e-6 | 1e-2/1e-8 · 5e-3/1e-6 · 5e-3/1e-6 · 1e-3/1e-4 |
| AmazonBook | same order | 1e-2/1e-8 · 1e-2/1e-6 · 5e-3/1e-6 · 1e-3/1e-8 | 1e-3/1e-8 · 1e-3/1e-8 · 1e-3/1e-6 · 1e-3/1e-6 |

**Controlled comparisons** (the mapping is the only difference):
- Full vs GraphHash **DirectAU** on Gowalla (both 1e-2/1e-8) and AmazonBook (both 1e-3/1e-8).
- **Both sides** must be trained in the same campaign, under the same stopping rule and budget, with matched init: Gowalla = **A2 vs I1**; AmazonBook = a new Full run vs I2.
- The legacy Full checkpoints (capped) are **not** valid controlled partners.
- BPR author configs differ across compressions, so BPR Full-vs-GraphHash rows are reproductions only.

### 7.3 Stopping rules (Stage 2 starts small; expanded only from measured curves)
| Loss | Eval every | Patience | Min. gain counted | Max epochs | Cap (S2) |
|---|---|---|---|---|---|
| BPR | 10 | 500 epochs | 0 | 3000 | 90 min |
| DirectAU | 5 | 30 epochs | 5e-4 | 200 (Gowalla), 150 (Yelp) | **180 min** |

**"Converged" means `stop_reason = patience`.** A run that hits its time cap or max epochs is **capped**:
- it can show a gap *persists within its budget*;
- any gate that asks about convergence is **INCONCLUSIVE** for it.

Direction of bias: a truncated DirectAU favours the anomaly and works against K2.

### 7.4 Compute: Kaggle GPU T4 x2
- **One session = 2 × T4,** with one condition per GPU (`CUDA_VISIBLE_DEVICES=0/1`). Each GPU may run a short sequential list. A crash on one GPU does not stop the other. Pair jobs of similar length.
- **Limits:** 2 sessions per account × 2 working accounts = **8 GPUs**. Session max 12 h [Likely]. Quota ≈ 30 h/week/account [Guessing], counted per session-hour [Likely]. The human checks the quota in the web UI.
- **API:** kaggle 1.7.4.5 sets only `enable_gpu`. Sessions report "Tesla T4", and Kaggle's T4 option is T4 x2 [Likely]. **A0 verifies `device_count == 2`**; otherwise fall back to one condition per session.
- **Memory:** ≤ 380 MiB per job. A0 measured 2 jobs per GPU at 82 s/epoch each vs 37 s solo, so we keep 1 job per GPU.
- **Added 18 Oct:**
  - **account-5** is a third account, *pinned-only*: only sessions that name it use it. It had about 30 h of quota then; account-3 / account-4 had about 6 h / 8 h left.
  - **One job on both T4s:** a job with `"gpus": "0,1"` sees both GPUs. LightGCN with `--multi-gpu` splits the propagation rows between them.
    - Its output and gradients are identical to 1-GPU propagation (checked on CPU).
    - The speedup is below 2× because of the copies between the GPUs at every layer.
  - Every job has a cap because of the 12 h session limit: a session that overruns loses **all** its outputs.

### 7.5 Budget [Guessing]
| Stage | Conditions | Sessions | Session-hours | Wall-clock |
|---|---|---|---|---|
| S1 | scoring only | 1 (shared with S2's short list) | ≈ 1 | ≈ 1–1.5 h |
| S2 | 9 (incl. smoke) | 3 + smoke | ≈ 7 | ≈ 3.5 h |
| S3 | decided at G2 (≈ 15–25) | ≈ 8–12 | ≈ 15–25 | overnight |

---

## 8. Code and workflow (minimal now; migrate later only if needed)
- **Code stays where it is validated:** `D:\web_2\experiments\2026-10-16-fast-check\` (`exp_run.py`, `job_queue.py`, `kapi.py`) and `…\2026-10-16-loss-sharing-repro\_work\` (`repro_run.py`, `repro_core.py`, GraphHash, datasets).
- **New files** are added next to them. Repository migration is deferred until it is clearly needed.
- **Run directory contract** (each process writes `runs/<run_id>/`):
  - `config.json`: args, resolved settings, code sha256, GPU, `device_count`, mapping sha256;
  - `mapping_{users,items}.npy`;
  - `train_log.csv`;
  - `norm_log.csv` (quintile lengths, ρ_item, ρ_bucket, ρ_dir per evaluation);
  - `best_seed{N}.pt` (model + aliases);
  - `metrics.json`;
  - `per_user_test__{mode}.npz`;
  - `exit_status.json` (exit code, stop reason, end time).
- **Validation is strict.** A run counts only if:
  - exit code is 0;
  - `metrics.json` has status complete and **every expected scoring mode**;
  - the checkpoint and both mapping files exist;
  - the mapping sha256 in `config.json` matches the files;
  - `stop_reason` is set.
- **Run ID:** `{dataset}__{backbone}__{compression}__{loss}__s{seed}` (+ suffix `-initmatch`, `-r{res}`, `-long`, `-p500`, `-wd{x}`, `-2gpu-v2`). Exploratory runs are prefixed `x__`.
- **Code added after Stage 3 (18 Oct):**
  - `exp_run.py`:
    - `--model LightGCN [--lightgcn-layers K] [--multi-gpu]`;
    - `--loss dau_resbias` (C3: learned β·p plus residual r_i with decay 0.01; eval mode `learned`);
    - `--method DoubleGraph` (the BACO substitute).
  - `score_controls.py --post-s3` (AmazonBook rescoring: E5 cosine, E7 uint8, E9 weighted, E8 tie policies).
  - `job_queue.py`: per-job `"gpus"`, `PINNED_ONLY_ACCOUNTS`.
  - Session builders `make_sessions_s{3,4,5}.py`.
  - Tests: `test_s3.py`, `test_s4.py`, `test_rescore.py`.
- **GitHub:** the private repo `pushkar1887/webconf27-popchannel` mirrors `D:\web_2`. It excludes:
  - credentials, kernel folders, session files with signed URLs, and files containing account usernames;
  - checkpoints, datasets, and files over 50 MB.

  Every commit is scanned for usernames and signed URLs first.
- **Loop:** PREREG (+ snapshot) → spec → launch (orchestrator only) → monitor → fetch + strict validate → aggregate → gate (OUTCOME in RESULTS.md, a line in the LOG).

---

## 9. Gates, decisions, schedule

**G0 (≈ 19:00): smoke OK.** Tests pass; A0 files are complete; `device_count` and co-location speed are logged.

**G1 (≈ 20:30): after Stage 1, which mechanism leads, and is it new?**

Inputs, each the mean of 3 seeds on Gowalla GraphHash × DirectAU:
- the **tie-break-only gain** (lexicographic), as a fraction of the full-repair gain;
- the fixed-β **Shapley shares** of the bucket term (μ) and residual term (ε), and their interaction.

| Stage 1 result | Framing direction |
|---|---|
| Tie-break-only recovers ≥ 50% of the full-repair gain | **"Sharing makes items indistinguishable; resolving ties with per-item popularity is the repair"** (H-tie leads) |
| Bucket-term Shapley share ≥ 0.5 | **"Sharing mis-calibrates buckets"** (bucket calibration leads) |
| Residual-term Shapley share ≥ 0.5 | **"Sharing erases item-level popularity"** (pooling leads) |
| None ≥ 0.5, or \|interaction\| ≥ 25% of the gain | Report the decomposition itself as the main result (K1) |

**G1.d (novelty):**
- If a paper already shows that sharing or hashing destroys item-level popularity information **and** repairs it with a per-item popularity term, the contribution shrinks to the decomposition, the DirectAU-vs-BPR explanation and the controlled comparisons.
- If those are covered too, **stop** and the human decides (backups exist).
- Erasure itself cannot be the contribution in that case.

**G2 (≈ 22:30): abstract framing and Stage 3 scope.**

| Check | Rule | Label |
|---|---|---|
| G2.a anomaly persists at convergence | A1 raw ≤ 9.04 **and** A1 stop_reason = patience. If capped: **INCONCLUSIVE** (report "persists within budget") | [I] |
| G2.b GraphHash is insensitive to init scale | \|I1 raw − A1 raw\| ≤ 0.3 and \|I1 cos+prior − A1 cos+prior\| ≤ 0.3 (both converged, else INCONCLUSIVE). This is **not** a sharing-vs-Full test | [P] |
| G2.c repair wins on Yelp | A4 cos+prior − A5 raw+prior ≥ 0.3, CI excludes 0 | [P] |

| Outcome | Action |
|---|---|
| All pass | Abstract per G1 framing, 3 datasets |
| G2.a fails | The anomaly was undertraining; the paper becomes the decomposition and repair only, or the human stops |
| G2.a INCONCLUSIVE (capped) | Abstract may say the gap "persists within a fixed training budget"; push a longer A1 into Stage 3 |
| G2.b fails | GraphHash results depend on init scale; init moves into the story (and I2 becomes mandatory) |
| G2.c fails | "Repair wins on 2 of 3 datasets"; K2 weakened |

**G3 (Sun 12:00):** abstract numbers frozen. **G4 (Fri 23, 18:00):** all paper numbers frozen.

### Schedule (IST)
| When | Who | What |
|---|---|---|
| **Sat 17:30–18:45** | Sonnet sub + Claude review | Minimal code (TASKS T1–T6) |
| Sat 17:30–21:00 | Codex or human | N1: PRISM full text, DNA, citing papers |
| Sat 17:30–17:50 | Haiku sub | A6 popularity-only (local) |
| Sat 18:45–19:05 | Claude | A0 smoke → **G0** |
| Sat 19:05 | Claude | Snapshot PREREG; push Stage 1 + Stage 2 sessions |
| Sat ~17:30 | Human | Check the Kaggle quota on both accounts (→ LOG) |
| **Sat ~20:30** | Human + Claude | **G1** (mechanism + novelty) |
| **Sat ~22:30** | Human + Claude | **G2**; register Stage 3; push overnight |
| Sun 08:00–12:00 | Claude + Codex | Stage 3 results, analyses, figures |
| **Sun 12:00** | Human | **G3**; abstract writing until 20:00 (both authors approve) |
| **Mon 19 by 12:00** | Human | **Submit the abstract** |
| Mon–Thu | Authors + Codex | Paper; optional runs if quota allows |
| **Fri 23, 18:00** | Human | **G4** freeze |
| **Mon 26 by 12:00** | Human | **Submit the paper** |

---

## 10. Pitfalls already hit
| Pitfall | Fix |
|---|---|
| Kaggle CPU image renumbers GraphHash buckets (BPR 9.39 → 4.24) | Never recompute mappings for scoring; legacy checkpoints are scored on the GPU image, with an equality assert against the P4/P5 stored aliases |
| 2 GPU sessions per account | The scheduler waits or uses the other account |
| Google Drive 500 | Download retries |
| ERROR'd kernels' outputs can't be mounted | Signed URL (expires) or rerun |
| `KAGGLE_CONFIG_DIR` read at import | One API call per subprocess |
| Full-batch BPR early spike | Patience 500 |
| BPR at batch 1024 is broken | Full batch only |
| `queue.py` shadows stdlib | Never use stdlib names |
| Laptop sleep stops the scheduler | Keep-awake; restartable state |
| Old `job_queue.py` has no argparse | Never pass flags to it |
| `torch.topk` tie order is implementation-defined; float scores of identical vectors may differ in the last bit | Score per bucket and copy to items (exact ties); lexicographic sort with an explicit secondary key |
| Adding a tiny β·p to break ties also reorders close, unequal scores | Never do it; use the lexicographic sort |
| A capped run counted as "converged" | Converged only if stop_reason = patience; otherwise the gate is INCONCLUSIVE |

---

## 11. LOG (append-only; **every agent logs here and only here**)
Format: `YYYY-MM-DD HH:MM IST | agent | IDs | what changed | issues | next`

2026-10-16 → 2026-10-17 07:08 | claude | history | Reproduction, fast check, seeds, re-scoring, P1–P5 batch. Raw outputs in `D:\web_2\experiments\2026-10-16-fast-check\results\` | P2/P3 checkpoints lack stored mappings | —

2026-10-17 14:45 | claude | plan v1 | First plan docs | — | —

2026-10-17 15:00 | claude | plan v2 | 4-doc structure, theory, stages, T4 x2 | — | —

2026-10-17 17:30 | claude | plan v3 | Applied the Codex (GPT 6.1) review after verifying its claims. PRISM (arXiv 2505.11318, KDD 2025) and DNA (ASOC 2026, doi 10.1016/j.asoc.2025.114226) confirmed as prior work and put first in N1. The r⁴ recurrence is now exact with its approximation stated. The pooling, Adam and direction statements are now hypotheses. The capacity bound keeps its scope: correlation only, alias tuple, non-nested resolutions. Product rule and erasure index are labelled heuristics. Added the exact decomposition p = μ + ε and five scoring controls, a tie analysis (H-tie) and a random-tie baseline, and the init-matched control I1 (Xavier norms 0.0558 vs 0.1477). C4 is optional (no gate). E9 renamed popularity-reweighted. C3 renamed regularized residual bias. Real user counts (25,789 / 50,407). [P]/[I] labels and snapshots added. Work reordered into Stages 1–3 with caps ≤ 3 h. **No code changed, no jobs launched** | Quota unknown; T4 x2 to verify in A0 | T1–T6 → A0 → Stage 1 + 2

2026-10-17 18:00 | claude | plan v3.1 | Applied 4 more review fixes. (1) Tie-break-only is now a lexicographic sort on bucket-level scores (exact ties), never a tiny β·p. (2) The decomposition is a 2×2 design at fixed β* with interaction + Shapley shares (primary); re-tuned controls are secondary; G1 uses Shapley shares. (3) I1 is renamed an initialization-sensitivity control; the controlled Full-vs-GraphHash pair is A2 vs I1 under the same stopping rule (legacy Full checkpoints don't qualify). (4) "Converged" only if stop_reason = patience; a capped run makes convergence gates INCONCLUSIVE. Files: PLAN, PREREG, TASKS, RESULTS. **No code changed, no jobs launched** | Optionally move A2 into Stage 2 so the controlled pair exists sooner | T1–T6

2026-10-17 18:00 | claude | T1–T6 | Launched 4 subagents: T1+T2 (runner + T4x2 scheduler), T3+T4 (score_controls, make_maps), T5 (validate/aggregate), T6+A6 (popularity-only). Orchestrator lock written. No Kaggle pushes yet | Q1 quota check pending (human) | review → A0 smoke

2026-10-17 18:15 IST | human→claude | Q1 | GPU quota from the web UI: two working accounts show 27h36m of 30h each; a third account shows 30h/30h (not configured for the scheduler; account-2 dir returns 401) | Quota is not a constraint for Stages 1–3 | continue

2026-10-17 18:35 IST | sub-sonnet/claude | T5 | scripts/validate_run.py, aggregate.py, legacy_registry.csv, test_t5.py (77 tests pass). master_metrics.csv generated: 56 rows / 20 legacy runs. Reproduces RESULTS §1–§2; exact Full Gowalla cosine = 12.996 (RESULTS rounds to 12.99), test gap 4.90 | — | waiting on T1–T4

2026-10-17 18:45 IST | sub-sonnet/claude | T3,T4 | score_controls.py, make_maps.py, test_t3t4.py (24/24 pass). Local Full check reproduces 17.895/12.996/19.544. Preliminary E2e recorded in RESULTS | GraphHash acceptance (7.55/13.30) only on the GPU image | wait for T1/T2, then review

2026-10-17 18:36 IST | claude | G0, snapshot | A0 smoke PASS: device_count=2 (2× Tesla T4), 4/4 runs strictly valid. Co-location: 2 jobs/GPU take 82 s/epoch each vs 37 s solo → no gain, keep 1 job/GPU. Yelp2018: split 1,249,124/156,140/156,142; eval users val 30,261 / test 30,304; GraphHash buckets 7,604/7,656 (976,640 params). Early hint (2 epochs): Gowalla GraphHash ρ_bucket 0.66, ρ_item 0.23. PREREG snapshot results/snapshots/PREREG_20261017-1836.txt sha256 be704ebe0632dd7aa4ab1b42ec69502bb2d1f421ebcfc0939aa8d8ba8b140de4 | — | push Stage 1 (X1, X2) + Stage 2 (S2a, S2b)

2026-10-17 19:40 IST | claude | Stage 1, A5/A7/A8, C0 | X1 + X2 complete (14 jobs, 0 failures). Outcomes in RESULTS: PR-TIE PASS, PR-DEC PASS with residual-dominant decomposition (Shapley ε 0.71 Gowalla, 0.76 AmazonBook; tie-break-only 12%; within-bucket shuffle destroys the gain), PR-E2e formal (decile FAIL, global PASS, replacement beats learned lengths), PR-M1, PR-A8 FAIL (16.55, not converged, above paper). G1 framing rule points to 'sharing erases item-level popularity' | Human to confirm G1 | wait for S2a (A1, I1) and S2b (A3, A4)

2026-10-17 20:10 IST | human→claude | G1 | **G1 confirmed by the human:** frame the paper as 'sharing erases item-level popularity information'. Ties are a secondary finding (87% of users have a boundary tie; tie-break-only = 12% of the gain). Evidence: fixed-beta Shapley ε 0.71 (Gowalla, 3 seeds) / 0.76 (AmazonBook); within-bucket shuffle destroys the gain. G1.d novelty still pending (N1 full reading) | — | G2 when S2a/S2b finish

2026-10-17 20:00 IST | sub-sonnet/claude | N1 | First-pass novelty: no paper found showing sharing/hashing erases item-level popularity + per-item repair. Closest: GraphHash, PRISM (no hashing; no λ=0 DirectAU numbers), Cheng arXiv 2606.21275 (unverified). G1.d provisional PASS; deeper sweep before submission. File results/N1_novelty.md | Cheng 2606.21275 to read | wait for Stage 2

2026-10-17 20:05 IST | claude | A1, I1 | S2a complete. A1 converged (patience, ep 55): raw 7.55, cos+prior 13.30 → G2.a PASS. I1 converged: raw 7.54, cos+prior 13.35 → G2.b PASS. H-pool: rho_bucket 0.925 vs rho_item 0.28 → PASS (strong) | — | wait for S2b (Yelp A3/A4) → G2

2026-10-17 21:15 IST | claude | A3, A4, PR-Y, PR-H, PR-A6 | S2b complete (both stop = patience; validate_run ok). Yelp flip reproduced (Full DAU 11.10 > BPR 8.02; GraphHash DAU 2.41 < BPR 3.78). Y4 repair wins +0.88 [0.74, 1.02] → G2.c PASS. Y3 FAIL (gap 0.44, rho 0.52; Yelp buckets keep more popularity, eta² 0.48). H-pool 0.944, H-Adam (max drop 0.15%), H-dir 2/2 PASS. PR-A6 PASS 3/3. Stage 1+2 queue finished, 0 failed jobs | — | G2: human decides abstract framing and Stage 3 scope

2026-10-17 22:01 IST | codex + three analysis subagents + independent reviewer | S3-prep, T5/E1, E5, E7, E9 | Prepared 24 jobs in four offline sessions (account-3 twice, account-4 twice); max estimated lane 311 min, cap+reserve 480 min; builder/dry-run bundles and informed prediction draft saved, PREREG and training runner unchanged, zero Kaggle operations. A2 plain Full Xavier matches I1's target; AmazonBook DAU max150 justified by legacy best40 plus patience; D7 wd0/1e-4 are explicitly authorized interventions. Refreshed CSV (298 unique rows, 10 controls), RESULTS tables and partial outcomes; 168 tests pass; independent review corrected missing draft B-iii, no remaining P1/P2 defect | C3 omitted: existing dau_bias lacks learned beta*p + residual r_i with separate Adam decay .01; no new training code. E5 DirectAU tail condition fails all three datasets. E7 modern Gowalla deviations exceed .02pp; legacy quantized val absent, so no universal one-byte claim. E9 modern Gowalla beta .2→.1, weighted recall 9.6356→10.8214; legacy weighted-val retuning remains GPU-only and missing. Existing Gow BPR s0 used patience100 versus new500; homogeneous three-seed inference unresolved. Partial E5/E7/E9 measures need an explicitly versioned analysis CSV schema before paper freeze | Orchestrator reviews sessions/draft, resolves scope/protocol notes, copies human-approved predictions to PREREG, snapshots before any push; preserve negative results and later complete missing GPU-only controls

2026-10-17 22:10 IST | claude | Stage 3 review, E5 | Reviewed Codex Stage 3 queue (24 jobs, lr/wd all match §7.2, 147 tests pass) and prereg draft. Found: (1) AmazonBook DirectAU ~1.8 min/epoch (legacy 41 ep in 75 min) → 180-min cap ≈ 98 epochs, likely capped → propose cap 300 min for the 5 AmazonBook DirectAU jobs; (2) Gowalla GraphHash BPR s0 used patience 100 → propose adding gowalla__mf__graphhash__bpr__s0-p500 (~5 min). E5 addendum: repair tail cost vs Full reference recorded in RESULTS. E7 A1 fails ±0.02 → keep float32 storage claim | — | human approves edits → register in PREREG, snapshot, push

2026-10-17 22:38 IST | codex | Stage 3 fixes A1/A2/A3, S3-E5/S3-E7, registration | Authorized per-job cap300 for exactly five AmazonBook DirectAU jobs (max150, full-cap estimate); added Gowalla GraphHash BPR s0-p500 and colocated all three final BPR seeds; 25 jobs prepared, four dry-run bundles syntax OK, zero pushes. Appended 13 draft headings verbatim without changing earlier PREREG bytes; snapshot results/snapshots/PREREG_20261017-2238.txt sha256 a3c1b585f8f3b1f3e5c4bd7033c799345955999ffc406313980b939909775392. Lane estimate/budget minutes: pc-s3a-1017 GPU0 365/500; pc-s3a-1017 GPU1 365/500; pc-s3b-1017 GPU0 379/600; pc-s3b-1017 GPU1 358/500; pc-s3c-1017 GPU0 401/510; pc-s3c-1017 GPU1 183/590; pc-s3d-1017 GPU0 382/580; pc-s3d-1017 GPU1 332/580. Tests 171 pass (queue12, runner8, contract29, controls24, T5/regression83, E5 7, E7/E9 8); independent review found no blocking defect | C3 blocked: needs learned beta*p plus residual with separate Adam decay .01. E5 registers head-accuracy/tail-cost interpretation; E7 float32 default unless every final GraphHash val/test precision check passes .02pp. Scheduling estimates and cap300 do not guarantee patience stopping; max150 can still cap convergence. No lr/wd changes, new research training, mapping recomputation, orchestrator-lock edits or Kaggle actions | next: orchestrator pushes Stage 3 after checking snapshot

2026-10-17 23:53 IST | claude | Stage 3 push | Verified PREREG snapshot results/snapshots/PREREG_20261017-2238.txt (sha256 a3c1b585f8f3b1f3e5c4bd7033c799345955999ffc406313980b939909775392 = PREREG.md); human approved push. Pushed pc-s3a/s3b-1017 (account-3), pc-s3c/s3d-1017 (account-4): 25 jobs, 8 T4 GPUs, longest lane est. 6h41m. Scheduler log results/queue_s3.log | — | validate runs as they land; score PR Stage 3 entries; G3 Sun 12:00

2026-10-18 04:45 IST | claude | Stage 3 | All 4 sessions DONE (04:23), 25/25 runs valid, 24 patience, AmazonBook Full BPR capped (max epochs). PR-E8 PASS: Gowalla +1.99/+2.13/+2.08 (all CIs > 0), Yelp +0.88, AmazonBook +1.52; Holm p ≤ 6e-33. A2, A8r, B9, B5–B8, I2 PASS; D7 wd0 ρ 0.74 PASS; main grid B-i 6/8, B-ii 8/8, B-iii 8/8, B-iv 6/9; repaired DAU > repaired BPR 6/6 Random/DF cells. Recorded in RESULTS | AmazonBook BPR side of E8 uses Stage 1 control file (random-key ties) | E5/E7/E9 on final runs; aggregate.py refresh; G3 Sun 12:00

2026-10-18 11:49 IST | codex + three implementer subagents + independent reviewer | E1, S3-E5, S3-E7, PR-E9, PR-E8 secondary, G3 package | Approved bounded design implemented; v1 API retained, versioned master_metrics_v2.csv/schema written:398 v1/401 unique v2 rows,25 S3 runs and24 matrix cells. Final E5:30 lineages,89/90 cached mode cells validated,7/7 GH DAU tail-deficit variants PASS on3datasets. E7:11 modern scored in354.16s,3fail/8pass plus1legacy missing validation; one-byte claim dropped, float32 overhead5.52/3.90/7.30%. E9:retuned ordering PASS Gowalla3seeds/Yelp; Amazon fullyretuned INCONCLUSIVE. Retained Recall/NDCG paired2000-seed0 bootstrap and separate paired-t Holm families; primaryE8 PASS. Corrected seed Recall mean/sampleSD DAU13.325+/-.024,BPR11.257+/-.063. Wrote prompts/g3_abstract_numbers.md with162-word PRISM/DNA-credited abstract and prompts/post_stage3_gpu_rescore.md; RESULTS updated,TASKS updated;124 tests pass,independent implementation/integration review no P1/P2 | Amazon legacy cosine,quantized validation,weighted retuning and alternate full-prior paired CIs remain GPU-only/missing. Alternate aggregate means stay positive but are not paired-CI robustness. GPU spec NOT SUBMITTED and requires reviewed scorer extensions. Amazon Full BPR remains epoch-capped; original Y3/A8/tail failures retained. Optional C3 note omitted because10:30 target had elapsed before work;12:00 G3 human freeze remains. PREREG unchanged;no new training,API/Kaggle operations or mappings | next: human signs off G3 candidate; orchestrator reviews missing-controls GPU specification; no automatic freeze or submission

2026-10-18 14:00 IST | claude (+2 sonnet subagents) | Stage 4 (items 1-8) | Human asked to run all 8 remaining experiments. Code: exp_run.py --model LightGCN, --loss dau_resbias (C3), --method DoubleGraph (BACO substitute; BACO code not available locally); score_controls.py --post-s3 (AmazonBook rescoring per prompts/post_stage3_gpu_rescore.md). 168 tests pass. PREREG Stage 4 registered; snapshot results/snapshots/PREREG_20261018-1355.txt sha256 7a115933314098090775381db2d48f3cbfbebbee6c7d92d24fcc0a581e36a86a. Pushed pc-s4a…s4d-1018: 35 jobs, lanes est 6.4–7.6 h | LightGCN/DoubleGraph/resbias untested on real data (unit tests only) | validate + score Stage 4 when queue_s4.log shows QUEUE_DONE

2026-10-18 22:20 IST | claude | Stage 4 | All 4 sessions done (scheduler crashed once, exit 4; restarted, resumed, nothing lost). 34 runs + rescoring valid, 0 failed. S4-SEED, S4-AMZBPR, S4-RESCORE, S4-DG PASS; S4-LGN L-i/ii/iii PASS (5 LightGCN rows capped); S4-C1 gain PASS, rho_item FAIL; S4-C3 FAIL (learned score far below post-hoc; code review pending); S4-C4 as predicted. Recorded in RESULTS | 6 capped runs (5 LightGCN, C1 r1600) | rerun capped runs with ~11 h caps, one job per GPU, pending human quota check

2026-10-18 22:07 IST | claude | Stage 5 | Human asked to use the third Kaggle account (30 h quota) for option B. Credentials copied from the default kaggle.json into account-5 (key not printed); job_queue ACCOUNTS += account-5. PREREG S5-LGNFULL registered, snapshot results/snapshots/PREREG_20261018-2204.txt sha256 2c498316897c84c0abbd262df4d37990c450cd015a9587decc947870b2a7f7d8. Pushed pc-s5a-1018: LightGCN Gowalla Full BPR + DirectAU, one per GPU, cap 690 min | — | expect results by ~09:45 IST Mon

2026-10-18 22:13 IST | claude | Stage 5 (2-GPU) | Added --multi-gpu row-split LightGCN propagation (exp_run.py SplitProp; forward+grad equal to 1-GPU on CPU) and per-job "gpus" override (job_queue.py); account-5 pinned-only. 91 tests pass. PREREG amendment snapshot results/snapshots/PREREG_20261018-2210.txt sha256 d79476fe3b56d4d5a04e3120deba6333b0a610a85ae47911a07a59ee5476e86f. pc-s5a-1018 cancelled by human. Pushed pc-s5b-1018 (BPR, 2 GPUs); pc-s5c-1018 (DirectAU, 2 GPUs) pending: Kaggle reports account-5 slots full while pc-s5a still shows QUEUED | — | scheduler retries pc-s5c automatically

2026-10-18 22:23 IST | claude | Stage 5 fix | pc-s5b-1018 failed in 7 s: '--multi-gpu needs 2 visible GPUs' because normalize_job dropped the per-job "gpus" key from the kernel bundle (bug; fixed + test). pc-s5c-1018 carries the same bug and will fail the same way. Re-pushed as pc-s5d-1018 (BPR) and pc-s5e-1018 (DirectAU), run_ids *-2gpu-v2; pc-s5d running past the GPU check; pc-s5e waits for pc-s5c to free the slot | — | fetch + score when done

2026-10-18 22:55 IST | claude | docs | Human asked to update the docs. TASKS: Kaggle facts (account-5 pinned-only, 2-GPU jobs, cancelling), naming, G2 done, C1/C3/C4/D1/D2/GPU-postS3 statuses, new Stage 4 and Stage 5 boards, N1 still open. RESULTS: outcome-board rows for Stages 3–5. PLAN: §7.4 compute additions, §8 code added after Stage 3 and the GitHub mirror | — | Stage 5 running; G3 sign-off pending; C3 code review todo
2026-10-18 22:06 IST | claude (cloud session) | N1, G1.d | Deep novelty sweep (WebSearch, SciSpace, local PDFs) appended to results/N1_novelty.md §(d)–(g); RESULTS PR-N1 block and board row updated. Kill rule not triggered. New must-cite prior work for the repair's shape: MEmCom (MLSys 2022, per-entity scalar on hashed table), Hash Embeddings (NeurIPS 2017), Seol et al. (WSDM 2024), CoRRe (arXiv 2608.19665, training-free popularity magnitudes; GraphHash/PRISM group). The DirectAU-hashing issue was already reported by Shiao et al. (GraphHash ref [45]). DNA authors verified (DBLP). Recommended G1.d PASS with narrowed contribution; title "the Few Bytes That Repair It" flagged as a MEmCom-comparison risk. No code, PREREG, runs or Kaggle touched | arxiv.org, Semantic Scholar, ACM DL, ScienceDirect, DBLP blocked by the environment network policy: Cheng full text, PRISM tail, Shiao DirectAU section, MEmCom equation and citer lists remain unread | human confirms G1.d and title; finish the 5 access-blocked checks on a normal connection before the paper deadline

2026-10-18 22:56 IST | claude (cloud session) | N1 | The human opened network access. Read the full texts of Cheng 2606.21275, PRISM 2505.11318, Shiao 2403.18280, MEmCom 2203.10135 and CoRRe 2608.19665; refreshed Semantic Scholar citers (GraphHash 10, PRISM 6, DNA 1, Shiao 3, Cheng 0). Kill rule still not triggered. Cheng eq. (53) is our score form on full tables; MEmCom Alg. 2 is a learned per-item scale on a hashed row, which gets zero gradient under DirectAU (Prop 1 corollary). PRISM names no optimizer and has no hashing. New cite: GDE (SIGIR 2026). Appended results/N1_novelty.md §(h) and a RESULTS PR-N1 block | dl.acm.org 403; DNA full text and Google Scholar not checked | human confirms G1.d and title

2026-10-18 23:10 IST | claude | G3, N1 | Merged the cloud N1 branch (commits ec22088, fa97a54) into main; conflicts in PLAN/RESULTS/TASKS resolved by keeping both append-only blocks and the newer N1 status. Wrote G3 package v2: 3 diagnosis-first titles and a ~195-word abstract that presents the β·log(1+count) term as a known score form (Cheng eq. 53) used as a minimal test, credits PRISM and the reported-but-unexplained gap, adds the zero-gradient corollary for learned per-item scales under DirectAU, and adds Stage 4 seeds + other sharing schemes | Corollary to add to PLAN §4 before the paper | authors decide G1.d, title, abstract

2026-10-19 06:35 IST | claude | Stage 5 | pc-s5d/pc-s5e done (03:41 / 06:26), both valid and converged on patience. S5-LGNFULL PASS. LightGCN Full Gowalla raw: BPR 18.48 > DirectAU 18.02 (flip premise is MF-specific); repaired DirectAU 19.76 > repaired BPR 18.46. 2-GPU row-split gave no speedup (0.70 / 1.33 min/epoch, same as 1 GPU) | — | G3 freeze pending (abstract v4); abstract due 17:29 today

2026-10-21 IST | human→claude | W1s, G3 | Human reports the abstract is submitted. G3 numbers frozen as in prompts/g3_abstract_numbers.md v4. Next: paper (due Mon 26 Oct 17:29 IST), G4 Fri 23 Oct 18:00 | — | paper writing; C3 code review; PLAN §4 zero-gradient corollary

2026-10-23 IST | claude | W0a, F1b | Paper plan `paper/OUTLINE.md` v1→v5 (commits 44eaa8f…3fc39f5): diagnosis structure on the human's section list; CFP checked (8 main + 12 total incl. refs/appendix; Web relevance on page 1; third-person self-citation). Review fixes adopted: Adam norm growth stated as observation; lemma assumptions; Spearman bound uses η²(rank c) 0.271/0.437/0.303; causal wording; modes fixed a priori, only β on validation; new two-channel decomposition (Gowalla drop 10.47 = magnitude 4.96 + direction 5.49; AmazonBook magnitude channel 0.87); three-way coverage comparison. Fig 1b data confirmed on CPU from stored checkpoints + stored mappings (ρ_item 0.275 / 0.815). Tracked results/ALL_RESULTS.md, results/run_ledger_all.csv and the writing-audit reports | Gemini numbers rejected (one-byte, 75%, published 17.98/15.62) | W0 LaTeX skeleton; W2 drafts; G4 Fri 18:00

2026-10-19 04:20 IST | claude (cloud session) | G3, W1 | Human supplied the final abstract text (200 words). Saved verbatim + mechanically corrected version in prompts/abstract_final.md (double period, spacing, subject agreement, one term for the correction, connective before the flip). All numbers verified against RESULTS incl. Stage 4 S4-SEED (9/9 pairs +0.83 to +2.13). No numbers changed | Optional LightGCN and zero-gradient sentences left to the human | human freezes G3 and submits by Mon 12:00

2026-10-22 IST | claude (cloud session) | W0, W2 (§3–§4), MAG-INT | Paper skeleton `paper/main.tex` (acmart sigconf,anonymous,review; section files; verified refs.bib, 10 entries). Drafted §3 (484 words) and §4 (1,085 words) per OUTLINE v5 with every number checked against ALL_RESULTS/RESULTS. New post hoc analysis MAG-INT (`scripts/analyze_magnitude_interaction.py` → `results/MAGNITUDE_INTERACTION.json`): raw drop = D + I exactly; interaction I = 4.98 [4.57, 5.37] / 3.68 [3.44, 3.93] / 0.88 [0.77, 0.99]; replaces the three-term path decomposition (no residual term). Checked LightGCN Full DirectAU raw 18.02 = MF 18.02: coincidence, top-20 lists differ for 92% of users. Fixed OUTLINE η² label: Gowalla DoubleFreq η²_p 0.999 (0.998 is rank c). Notation: user embedding e_u (not p_u, which collides with p_i = log-count) | No TeX in this container: not compiled. Pairings described as "registered before the final runs, after pilot runs" (they were chosen after pilot test results) | human reviews §3–§4 and the I-decomposition; §5–§6 next (Fri 23, numbers freeze 18:00)
