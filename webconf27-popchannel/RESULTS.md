# RESULTS: measured outcomes (v3)

**Rules:**
- One block per PREREG ID is appended in §5: prediction (copied), observed values, verdict, run_ids, stop reasons.
- Tables in §1–§2 are regenerated from `results/master_metrics.csv` once `aggregate.py` (T5) exists.
- A run counts only if it passes **strict validation** (PLAN §8).

**Hardware:** Kaggle Tesla T4. Legacy runs used **one GPU per session**. From Stage 1/2 on, sessions are **GPU T4 x2** with one condition per GPU (A0 records `device_count`).

**Units and definitions:**
- Recall@20 / NDCG@20 in %, authors' denominator min(20, |T_u|).
- "+prior" = `+ β·log(1+c_i)`, β picked on validation, test scored once.
- ρ = Spearman(item length, training count). Gap = raw − cosine on the same checkpoint.
- **Comparison type:** "controlled" (only the mapping differs) or "reproduction" (author per-condition settings).

---

## 0. Outcome board
| ID | Short prediction | Label | Observed | Verdict |
|---|---|---|---|---|
| PR-TIE | ≥ 50% users with a boundary tie; random vs default key \|Δ\| ≤ 0.2; tie-break-only gain (lexicographic) reported | P | 87% users tied (14.7 items); key Δ +0.04–0.06; tie-break-only = 12% of gain | PASS |
| PR-DEC | 2×2 at fixed β\*: main effects, interaction, Shapley shares; within-bucket shuffle keeps ≤ 70% of the gain | P | Shapley μ 0.29 / ε 0.71; shuffle keeps −6% | PASS |
| PR-E2e | Decile shuffle \|Δ\| small; global shuffle drop large (Full models) | P | decile −4.19; global −11.47; replacement +0.83 (favourable) | decile FAIL, global PASS, repl. FAIL (favourable) |
| PR-M1 | Theorem check holds; GraphHash η² ≤ 0.5 | P | η² GraphHash 0.36–0.48; bound holds | M-i PASS, M-ii PASS, M-iii mixed [X] |
| PR-A6 | Popularity-only ≥ 2 points below the repair (Yelp, AmazonBook) | P | pop-only 2.97 / 1.68 / 0.96 vs repair 13.29 / 5.68 / 6.33 | PASS (3/3) |
| PR-N1 | Novelty verdict | — | PRISM, DNA confirmed (see §N1); full reading pending | pending |
| PR-A1 | GraphHash-DirectAU raw ≤ 9.04 **and** stopped by patience (capped = INCONCLUSIVE) | I | raw 7.55; cos+prior 13.30; stop = patience (ep 55) | PASS |
| PR-I1 | Init-sensitivity control: within ±0.3 of A1, both converged (not a sharing test) | P | raw Δ −0.01; cos+prior Δ +0.05 | PASS |
| PR-A8 | Full-BPR ∈ [15.4, 15.8] | I | 16.55, capped at max epochs | FAIL (not converged) |
| PR-Y | Yelp: Y1 (I), Y3 (P), Y4 repair wins (P), Y2 (X) | mixed | Y1 2.41<3.78; Y3 gap 0.44, ρ 0.52; Y4 +0.88 [0.74,1.02]; Y2 E 2.70 | Y1 PASS, Y3 FAIL, Y4 PASS, Y2 reported |
| PR-H | ρ_bucket ≥ 0.6; lengths only grow; ρ_dir BPR > DirectAU | H | ρ_bucket 0.925/0.944; max r² drop 0.15%; ρ_dir BPR>DAU 2/2 | H-pool, H-Adam, H-dir PASS |
| PR-E5 | Head share ≥60%; tail recall change ≤0.5 points | informed operationalization | Head share 89.45–98.48% for DirectAU; conditional tail losses 1.30–6.81 points | partial (Stage 1–2); DirectAU tail prediction FAIL |
| PR-E7 | uint8 prior changes Recall by ≤0.02 points | Stage 1 intention; Stage 2 sensitivity | Original GraphHash test means pass; legacy validation missing; modern A1 −0.0264 val / +0.0246 test | partial (Stage 1–2); universal one-byte claim unsupported |
| PR-E9 | Reweighted-validation β shrinks or stays; Gowalla K2 ordering kept | informed follow-up | Gowalla A1 β .2→.1; weighted recall 9.6356→10.8214; legacy BPR retuning unavailable | partial (Stage 1–2); fully retuned ordering INCONCLUSIVE |
| PR-E8 final | Repaired DAU > repaired BPR, three datasets and every Gowalla seed | I | All five Recall CIs >0; retained separate NDCG inference | PASS; alternate Amazon paired CI missing |
| S3-E5 final | Repaired GH DAU tail < Full DAU raw tail | I | 7/7 variants; 89/90 mode records available | PASS means tail cost; legacy cosine missing |
| S3-E7 final | Every GH checkpoint within 0.02pp on val and test | I | 3 modern failures; 8 modern passes; 1 legacy missing validation | FAIL; float32 default |
| PR-E9 final | Weighted-validation β shrinks/stays; corrected ordering retained | I | Gowalla 3 seeds and Yelp PASS; Amazon retuning missing | partial; Amazon INCONCLUSIVE |
| S3-B, D7 | Main grid and decay checks | I/H | See Stage 3 outcome block; wd0 persistence under this setup | Report failures; no universal loss reversal |
| S3-A2, A8r, B9, B5–B8, I2 | Stage 3 range predictions | I | all inside their ranges | PASS |
| S4-SEED | Yelp / AmazonBook seeds 1–2 in range, every Δ CI > 0 | I | Δ +0.83 to +1.54 | PASS; PR-E8 now holds on all 9 seed pairs |
| S4-AMZBPR | AmazonBook Full BPR converges, raw ≥ 8.37 | I | 8.38, stopped on patience | PASS |
| S4-RESCORE | alternative-tie CIs > 0; uint8; weighted | I/P | +1.51 to +1.52; within 0.002 pp; 5.53 vs 4.04 | PASS (closes the AmazonBook gaps in PR-E8 / PR-E9) |
| S4-C3 | learned score within ±0.5 of post-hoc | I | 4.56 vs 13.21 (Gowalla), 2.68 vs 5.68, 1.63 vs 6.25 | FAIL (code review pending) |
| S4-LGN | LightGCN L-i, L-ii, L-iii | H | all three hold; 5 rows capped at 4 h | PASS (budget-limited rows) |
| S4-DG | DoubleGraph (BACO substitute) repair; DirectAU > BPR | I | 10.91 vs 7.27; 5.52 vs 2.78 | PASS |
| S4-C1 | gain falls and ρ_item rises with resolution | P | Spearman −0.90 / +0.70 | gain PASS, ρ FAIL |
| S4-C4 | equal norms (exploratory) | X | Full 12.77 → 19.32; GraphHash 13.27 | as predicted |
| S5-LGNFULL | LightGCN Gowalla Full, 11.5 h on 2 GPUs | H | — | running |

---

## 1. Headline: final Stage 3 and retained references

Generated from `results/master_metrics_v2.csv` by `scripts/render_stage3_tables.py`. Cells are test capped Recall@20 / NDCG@20 in percent. Both prior modes show their own validation-selected β; DirectAU cosine+prior versus BPR raw+prior is the registered repaired comparison. The 24-cell matrix uses final seed-0 checkpoints, with retained seeds and initialization variants below. The AmazonBook GraphHash BPR row is the retained legacy patience-500 checkpoint; all other matrix rows are modern.

| Dataset | Method × loss | Run ID | Raw | Cosine | Cos + prior (β) | Raw + prior (β) | ρ_item | Lineage / comparison | Status / stopping |
|---|---|---|---|---|---|---|---|---|---|
| gowalla | full × bpr | `gowalla__mf__full__bpr__s0-long` | 16.55 / 10.09 | 11.96 / 6.69 | 16.71 / 10.12 (β 0.05) | 17.06 / 10.67 (β 0.5) | 0.770 | modern / reproduction | valid / patience; best 2790 / ran 3290 |
| gowalla | full × directau | `gowalla__mf__full__directau__s0` | 18.02 / 11.61 | 13.06 / 7.46 | 19.57 / 12.59 (β 0.1) | 18.02 / 11.62 (β 0.01) | 0.815 | modern / controlled | valid / patience; best 160 / ran 170 |
| gowalla | graphhash × bpr | `gowalla__mf__graphhash__bpr__s0-p500` | 9.37 / 5.53 | 5.86 / 2.94 | 9.96 / 6.44 (β 0.1) | 11.31 / 6.96 (β 0.5) | 0.066 | modern / reproduction | valid / patience; best 460 / ran 960 |
| gowalla | graphhash × directau | `gowalla__mf__graphhash__directau__s0` | 7.55 / 4.06 | 7.57 / 4.03 | 13.30 / 8.57 (β 0.2) | 10.33 / 6.41 (β 5.0) | 0.275 | modern / reproduction | valid / patience; best 55 / ran 60 |
| gowalla | random × bpr | `gowalla__mf__random__bpr__s0` | 0.76 / 0.38 | 0.38 / 0.20 | 3.44 / 2.19 (β 0.2) | 3.42 / 2.17 (β 0.5) | 0.093 | modern / reproduction | valid / patience; best 240 / ran 740 |
| gowalla | random × directau | `gowalla__mf__random__directau__s0` | 0.45 / 0.26 | 0.29 / 0.15 | 3.80 / 2.43 (β 0.2) | 3.70 / 2.38 (β 2.0) | 0.275 | modern / reproduction | valid / patience; best 40 / ran 40 |
| gowalla | dfreq × bpr | `gowalla__mf__dfreq__bpr__s0` | 3.93 / 2.56 | 2.95 / 1.84 | 4.06 / 2.70 (β 0.1) | 4.11 / 2.74 (β 0.5) | 0.100 | modern / reproduction | valid / patience; best 450 / ran 950 |
| gowalla | dfreq × directau | `gowalla__mf__dfreq__directau__s0` | 3.22 / 2.56 | 2.20 / 1.73 | 5.11 / 3.50 (β 0.2) | 4.87 / 3.38 (β 2.0) | 0.028 | modern / reproduction | valid / patience; best 65 / ran 80 |
| yelp2018 | full × bpr | `yelp2018__mf__full__bpr__s0` | 8.02 / 5.13 | 5.54 / 3.27 | 8.50 / 5.43 (β 0.1) | 8.02 / 5.13 (β 0.0) | 0.970 | modern / reproduction | valid / patience; best 2150 / ran 2650 |
| yelp2018 | full × directau | `yelp2018__mf__full__directau__s0` | 11.10 / 7.20 | 6.97 / 4.24 | 11.26 / 7.35 (β 0.1) | 11.10 / 7.20 (β 0.0) | 0.912 | modern / reproduction | valid / patience; best 110 / ran 140 |
| yelp2018 | graphhash × bpr | `yelp2018__mf__graphhash__bpr__s0` | 3.78 / 2.36 | 2.03 / 1.16 | 4.82 / 3.03 (β 0.2) | 4.80 / 3.00 (β 0.5) | 0.190 | modern / reproduction | valid / patience; best 360 / ran 860 |
| yelp2018 | graphhash × directau | `yelp2018__mf__graphhash__directau__s0` | 2.41 / 1.36 | 1.97 / 1.10 | 5.68 / 3.55 (β 0.2) | 4.96 / 3.08 (β 1.0) | 0.520 | modern / reproduction | valid / patience; best 50 / ran 60 |
| yelp2018 | random × bpr | `yelp2018__mf__random__bpr__s0` | 0.55 / 0.32 | 0.38 / 0.23 | 2.15 / 1.41 (β 0.2) | 2.18 / 1.43 (β 1.0) | 0.075 | modern / reproduction | valid / patience; best 420 / ran 920 |
| yelp2018 | random × directau | `yelp2018__mf__random__directau__s0` | 0.42 / 0.25 | 0.29 / 0.18 | 2.39 / 1.62 (β 0.2) | 2.41 / 1.57 (β 2.0) | 0.353 | modern / reproduction | valid / patience; best 30 / ran 40 |
| yelp2018 | dfreq × bpr | `yelp2018__mf__dfreq__bpr__s0` | 2.55 / 1.71 | 1.90 / 1.25 | 2.52 / 1.72 (β 0.1) | 2.58 / 1.74 (β 0.1) | 0.145 | modern / reproduction | valid / patience; best 460 / ran 960 |
| yelp2018 | dfreq × directau | `yelp2018__mf__dfreq__directau__s0` | 2.51 / 1.89 | 1.56 / 1.20 | 3.33 / 2.38 (β 0.2) | 3.17 / 2.27 (β 0.05) | 0.138 | modern / reproduction | valid / patience; best 15 / ran 45 |
| amazonbook | full × bpr | `amazonbook__mf__full__bpr__s0` | 8.37 / 5.30 | 6.57 / 4.11 | 8.51 / 5.45 (β 0.05) | 8.37 / 5.30 (β 0.0) | 0.861 | modern / reproduction | capped / max_epochs; best 2690 / ran 3000 |
| amazonbook | full × directau | `amazonbook__mf__full__directau__s0` | 9.57 / 6.45 | 8.70 / 5.69 | 11.05 / 7.48 (β 0.1) | 10.77 / 7.26 (β 0.2) | 0.786 | modern / controlled | valid / patience; best 25 / ran 40 |
| amazonbook | graphhash × bpr | `amazonbook__mf__graphhash__bpr__s0` | 4.10 / 2.60 | 2.53 / 1.59 | 3.34 / 2.12 (β 0.05) | 4.81 / 3.28 (β 0.2) | 0.084 | legacy / reproduction | valid / patience; best 1450 / ran 1950 |
| amazonbook | graphhash × directau | `amazonbook__mf__graphhash__directau__s0` | 4.63 / 3.12 | 4.64 / 3.13 | 6.33 / 4.46 (β 0.1) | 6.29 / 4.44 (β 0.5) | 0.189 | modern / reproduction | valid / patience; best 40 / ran 65 |
| amazonbook | random × bpr | `amazonbook__mf__random__bpr__s0` | 0.17 / 0.11 | 0.11 / 0.07 | 1.31 / 0.87 (β 0.2) | 1.28 / 0.84 (β 1.0) | 0.083 | modern / reproduction | valid / patience; best 90 / ran 590 |
| amazonbook | random × directau | `amazonbook__mf__random__directau__s0` | 0.15 / 0.09 | 0.12 / 0.07 | 1.55 / 1.07 (β 0.2) | 1.49 / 1.01 (β 0.5) | 0.242 | modern / reproduction | valid / patience; best 35 / ran 35 |
| amazonbook | dfreq × bpr | `amazonbook__mf__dfreq__bpr__s0` | 1.28 / 0.91 | 1.01 / 0.70 | 1.35 / 0.98 (β 0.1) | 1.35 / 0.97 (β 0.5) | 0.136 | modern / reproduction | valid / patience; best 420 / ran 920 |
| amazonbook | dfreq × directau | `amazonbook__mf__dfreq__directau__s0` | 1.51 / 1.25 | 1.12 / 0.96 | 2.13 / 1.62 (β 0.2) | 2.07 / 1.58 (β 0.5) | 0.026 | modern / reproduction | valid / patience; best 15 / ran 40 |

**Capped baseline:** `amazonbook__mf__full__bpr__s0` reached 3000 epochs (best 2690); its scores are descriptive and do not establish convergence. The final Gowalla Full BPR reference is `s0-long`, and GraphHash BPR seed 0 is `s0-p500`; their earlier capped/patience-100 runs remain in the ledger.

**Final Gowalla GraphHash seeds 0/1/2:** mean ± sample SD over trained seeds. DirectAU uses A1/s1/s2; BPR uses s0-p500/s1/s2. These are training-seed spreads, separate from paired-user bootstrap intervals conditional on each checkpoint.

| Loss | Scoring mode | Recall@20 mean ± SD | NDCG@20 mean ± SD | β by seed 0/1/2 |
|---|---|---|---|---|
| bpr | raw | 9.389 ± 0.017 | 5.543 ± 0.014 | 0.0, 0.0, 0.0 |
| bpr | cosine | 5.523 ± 0.300 | 2.786 ± 0.147 | 0.0, 0.0, 0.0 |
| bpr | cos_prior | 9.790 ± 0.153 | 6.321 ± 0.112 | 0.1, 0.1, 0.1 |
| bpr | raw_prior | 11.257 ± 0.063 | 6.911 ± 0.040 | 0.5, 0.5, 0.5 |
| directau | raw | 7.581 ± 0.031 | 4.074 ± 0.016 | 0.0, 0.0, 0.0 |
| directau | cosine | 7.580 ± 0.012 | 4.036 ± 0.008 | 0.0, 0.0, 0.0 |
| directau | cos_prior | 13.325 ± 0.024 | 8.568 ± 0.011 | 0.2, 0.2, 0.2 |
| directau | raw_prior | 10.437 ± 0.106 | 6.443 ± 0.027 | 5.0, 5.0, 5.0 |

**Controlled and initialization comparisons:** A2 versus I1 and AmazonBook Full versus I2 are the eligible controlled pairs. A1 versus I1 and B9 versus I2 assess initialization sensitivity. A `controlled` label denotes the declared configuration and does not alone establish a causal mechanism.

| Role | Run | Raw | Cos + prior (β) | Comparison | Status / stopping |
|---|---|---|---|---|---|
| Gowalla controlled Full reference A2 | `gowalla__mf__full__directau__s0` | 18.02 / 11.61 | 19.57 / 12.59 (β 0.1) | controlled | valid / patience; best 160 / ran 170 |
| Gowalla matched-init GraphHash I1 | `gowalla__mf__graphhash__directau__s0-initmatch` | 7.54 / 4.08 | 13.35 / 8.59 (β 0.2) | controlled | valid / patience; best 50 / ran 80 |
| Gowalla A1: initialization sensitivity versus I1 | `gowalla__mf__graphhash__directau__s0` | 7.55 / 4.06 | 13.30 / 8.57 (β 0.2) | reproduction | valid / patience; best 55 / ran 60 |
| AmazonBook controlled Full partner | `amazonbook__mf__full__directau__s0` | 9.57 / 6.45 | 11.05 / 7.48 (β 0.1) | controlled | valid / patience; best 25 / ran 40 |
| AmazonBook matched-init GraphHash I2 | `amazonbook__mf__graphhash__directau__s0-initmatch` | 4.59 / 3.11 | 6.30 / 4.45 (β 0.1) | controlled | valid / patience; best 35 / ran 60 |
| AmazonBook B9: initialization sensitivity versus I2 | `amazonbook__mf__graphhash__directau__s0` | 4.63 / 3.12 | 6.33 / 4.46 (β 0.1) | reproduction | valid / patience; best 40 / ran 65 |

---

## 2. Complete run and control ledger (CSV-derived)

`results/master_metrics.csv` preserves the v1 schema. `results/master_metrics_v2.csv` adds explicit lineage, capped/converged flags and separate E5/E7/E9 fields; its contract is `results/master_metrics_v2.schema.json`. Blank analysis values are unavailable, never zero. Weighted standard-denominator recall never replaces main capped Recall@20. Distinct legacy checkpoints retain their own IDs and stopping labels.

### 2.1 Training, baseline and retained checkpoint lineages

| Run | Raw / pop-only | Cosine | Cos + prior (β) | Raw + prior (β) | ρ_item | Lineage | Comparison | Status / stopping |
|---|---|---|---|---|---|---|---|---|
| `amazonbook__mf__dfreq__bpr__s0` | 1.28 / 0.91 | 1.01 / 0.70 | 1.35 / 0.98 (β 0.1) | 1.35 / 0.97 (β 0.5) | 0.136 | modern | reproduction | valid / patience; best 420 / ran 920 |
| `amazonbook__mf__dfreq__directau__s0` | 1.51 / 1.25 | 1.12 / 0.96 | 2.13 / 1.62 (β 0.2) | 2.07 / 1.58 (β 0.5) | 0.026 | modern | reproduction | valid / patience; best 15 / ran 40 |
| `amazonbook__mf__full__bpr__s0` | 8.37 / 5.30 | 6.57 / 4.11 | 8.51 / 5.45 (β 0.05) | 8.37 / 5.30 (β 0.0) | 0.861 | modern | reproduction | capped / max_epochs; best 2690 / ran 3000 |
| `amazonbook__mf__full__directau__s0` | 9.57 / 6.45 | 8.70 / 5.69 | 11.05 / 7.48 (β 0.1) | 10.77 / 7.26 (β 0.2) | 0.786 | modern | controlled | valid / patience; best 25 / ran 40 |
| `amazonbook__mf__full__directau__s0-legacy` | 9.57 / 6.45 | 8.70 / 5.69 | 11.05 / 7.48 (β 0.1) | 10.77 / 7.26 (β 0.2) | 0.786 | legacy | reproduction | valid / patience; best 25 / ran 35 |
| `amazonbook__mf__graphhash__bpr__s0` | 4.10 / 2.60 | 2.53 / 1.59 | 3.34 / 2.12 (β 0.05) | 4.81 / 3.28 (β 0.2) | 0.084 | legacy | reproduction | valid / patience; best 1450 / ran 1950 |
| `amazonbook__mf__graphhash__bpr__s0-pat100` | 3.93 / 2.68 | 3.89 / 2.67 | 5.06 / 3.58 (β 0.1) | 4.97 / 3.59 (β 0.02) | 0.128 | legacy | reproduction | invalid / patience; best 10 / ran 110 |
| `amazonbook__mf__graphhash__directau__s0` | 4.63 / 3.12 | 4.64 / 3.13 | 6.33 / 4.46 (β 0.1) | 6.29 / 4.44 (β 0.5) | 0.189 | modern | reproduction | valid / patience; best 40 / ran 65 |
| `amazonbook__mf__graphhash__directau__s0-initmatch` | 4.59 / 3.11 | 4.63 / 3.12 | 6.30 / 4.45 (β 0.1) | 6.33 / 4.45 (β 0.5) | 0.188 | modern | controlled | valid / patience; best 35 / ran 60 |
| `amazonbook__mf__graphhash__directau__s0-legacy` | 4.63 / 3.12 | 4.64 / 3.13 | 6.33 / 4.46 (β 0.1) | 6.29 / 4.44 (β 0.5) | 0.189 | legacy | reproduction | capped / time_cap; best 40 / ran 41 |
| `amazonbook__mf__random__bpr__s0` | 0.17 / 0.11 | 0.11 / 0.07 | 1.31 / 0.87 (β 0.2) | 1.28 / 0.84 (β 1.0) | 0.083 | modern | reproduction | valid / patience; best 90 / ran 590 |
| `amazonbook__mf__random__directau__s0` | 0.15 / 0.09 | 0.12 / 0.07 | 1.55 / 1.07 (β 0.2) | 1.49 / 1.01 (β 0.5) | 0.242 | modern | reproduction | valid / patience; best 35 / ran 35 |
| `amazonbook__poponly__none__none__s0` | 0.96 / 0.61 | — | — | — | — | baseline | reproduction | valid / not recorded |
| `gowalla__mf__dfreq__bpr__s0` | 3.93 / 2.56 | 2.95 / 1.84 | 4.06 / 2.70 (β 0.1) | 4.11 / 2.74 (β 0.5) | 0.100 | modern | reproduction | valid / patience; best 450 / ran 950 |
| `gowalla__mf__dfreq__directau__s0` | 3.22 / 2.56 | 2.20 / 1.73 | 5.11 / 3.50 (β 0.2) | 4.87 / 3.38 (β 2.0) | 0.028 | modern | reproduction | valid / patience; best 65 / ran 80 |
| `gowalla__mf__dfreq__directau__s0-legacy30` | 3.23 / 2.54 | 2.18 / 1.71 | 5.15 / 3.50 (β 0.2) | 4.90 / 3.36 (β 2.0) | 0.029 | legacy | reproduction | valid-short / fixed_epochs; best 30 / ran 30 |
| `gowalla__mf__full__bpr__s0` | 16.55 / 10.09 | 11.96 / 6.69 | 16.71 / 10.12 (β 0.05) | 17.06 / 10.67 (β 0.5) | 0.770 | modern | reproduction | capped / max_epochs; best 2790 / ran 3000 |
| `gowalla__mf__full__bpr__s0-long` | 16.55 / 10.09 | 11.96 / 6.69 | 16.71 / 10.12 (β 0.05) | 17.06 / 10.67 (β 0.5) | 0.770 | modern | reproduction | valid / patience; best 2790 / ran 3290 |
| `gowalla__mf__full__directau__s0` | 18.02 / 11.61 | 13.06 / 7.46 | 19.57 / 12.59 (β 0.1) | 18.02 / 11.62 (β 0.01) | 0.815 | modern | controlled | valid / patience; best 160 / ran 170 |
| `gowalla__mf__full__directau__s0-legacy` | 17.90 / 11.52 | 13.00 / 7.44 | 19.54 / 12.59 (β 0.1) | — | 0.815 | legacy | reproduction | capped / time_cap; best 120 / ran 130 |
| `gowalla__mf__full__directau__s0-wd0` | 16.12 / 10.00 | 12.61 / 7.13 | 19.08 / 12.31 (β 0.1) | 17.89 / 11.45 (β 2.0) | 0.744 | modern | reproduction | valid / patience; best 15 / ran 45 |
| `gowalla__mf__full__directau__s0-wd1e-4` | 1.41 / 0.64 | 2.75 / 1.47 | 8.69 / 5.52 (β 0.1) | 6.76 / 4.19 (β 0.05) | 0.223 | modern | reproduction | valid / patience; best 30 / ran 35 |
| `gowalla__mf__graphhash__bpr__s0` | 9.34 / 5.49 | 6.08 / 3.08 | 10.22 / 6.62 (β 0.1) | 11.24 / 6.95 (β 0.5) | 0.054 | legacy | reproduction | valid / patience; best 330 / ran 430 |
| `gowalla__mf__graphhash__bpr__s0-b1024` | 5.31 / 3.07 | — | — | — | — | legacy | reproduction | invalid / fixed_epochs; best 30 / ran 30 |
| `gowalla__mf__graphhash__bpr__s0-b1024-pat10` | 5.45 / 3.13 | — | — | — | — | legacy | reproduction | invalid / patience; best 5 / ran 15 |
| `gowalla__mf__graphhash__bpr__s0-p500` | 9.37 / 5.53 | 5.86 / 2.94 | 9.96 / 6.44 (β 0.1) | 11.31 / 6.96 (β 0.5) | 0.066 | modern | reproduction | valid / patience; best 460 / ran 960 |
| `gowalla__mf__graphhash__bpr__s1` | 9.39 / 5.55 | 5.41 / 2.76 | 9.73 / 6.31 (β 0.1) | 11.19 / 6.89 (β 0.5) | 0.080 | modern | reproduction | valid / patience; best 780 / ran 1280 |
| `gowalla__mf__graphhash__bpr__s2` | 9.41 / 5.55 | 5.30 / 2.65 | 9.68 / 6.22 (β 0.1) | 11.27 / 6.88 (β 0.5) | 0.089 | modern | reproduction | valid / patience; best 1540 / ran 2040 |
| `gowalla__mf__graphhash__directau__s0` | 7.55 / 4.06 | 7.57 / 4.03 | 13.30 / 8.57 (β 0.2) | 10.33 / 6.41 (β 5.0) | 0.275 | modern | reproduction | valid / patience; best 55 / ran 60 |
| `gowalla__mf__graphhash__directau__s0-initmatch` | 7.54 / 4.08 | 7.54 / 4.03 | 13.35 / 8.59 (β 0.2) | 10.54 / 6.52 (β 5.0) | 0.274 | modern | controlled | valid / patience; best 50 / ran 80 |
| `gowalla__mf__graphhash__directau__s0-legacy30` | 7.56 / 4.04 | 7.58 / 4.04 | 13.26 / 8.54 (β 0.2) | 11.43 / 7.00 (β 5.0) | 0.271 | legacy | reproduction | valid-short / fixed_epochs; best 30 / ran 30 |
| `gowalla__mf__graphhash__directau__s1` | 7.61 / 4.09 | 7.59 / 4.05 | 13.32 / 8.55 (β 0.2) | 10.45 / 6.46 (β 5.0) | 0.277 | modern | reproduction | valid / patience; best 50 / ran 80 |
| `gowalla__mf__graphhash__directau__s1-legacy30` | 7.57 / 4.06 | 7.56 / 4.03 | 13.31 / 8.55 (β 0.2) | 11.49 / 7.02 (β 5.0) | 0.274 | legacy | reproduction | valid-short / fixed_epochs; best 30 / ran 30 |
| `gowalla__mf__graphhash__directau__s2` | 7.59 / 4.07 | 7.58 / 4.03 | 13.35 / 8.58 (β 0.2) | 10.54 / 6.46 (β 5.0) | 0.274 | modern | reproduction | valid / patience; best 50 / ran 60 |
| `gowalla__mf__graphhash__directau__s2-legacy30` | 7.53 / 4.01 | 7.55 / 4.02 | 13.30 / 8.53 (β 0.2) | — | 0.270 | legacy | reproduction | valid-short / fixed_epochs; best 30 / ran 30 |
| `gowalla__mf__graphhash__directau_bias__s0` | 7.38 / 3.95 | 7.48 / 3.99 | 13.17 / 8.47 (β 0.2) | 11.77 / 7.17 (β 5.0) | 0.219 | legacy | reproduction | ablation / fixed_epochs; best 25 / ran 30 |
| `gowalla__mf__random__bpr__s0` | 0.76 / 0.38 | 0.38 / 0.20 | 3.44 / 2.19 (β 0.2) | 3.42 / 2.17 (β 0.5) | 0.093 | modern | reproduction | valid / patience; best 240 / ran 740 |
| `gowalla__mf__random__directau__s0` | 0.45 / 0.26 | 0.29 / 0.15 | 3.80 / 2.43 (β 0.2) | 3.70 / 2.38 (β 2.0) | 0.275 | modern | reproduction | valid / patience; best 40 / ran 40 |
| `gowalla__mf__random__directau__s0-legacy30` | 0.47 / 0.27 | 0.31 / 0.17 | 3.85 / 2.44 (β 0.2) | 3.69 / 2.38 (β 2.0) | 0.274 | legacy | reproduction | valid-short / fixed_epochs; best 20 / ran 30 |
| `gowalla__poponly__none__none__s0` | 2.97 / 1.70 | — | — | — | — | baseline | reproduction | valid / not recorded |
| `x__coloc_a` | 6.73 / 3.63 | 7.07 / 3.78 | 12.57 / 8.08 (β 0.2) | 12.32 / 7.81 (β 0.2) | 0.209 | modern | reproduction | capped / max_epochs; best 1 / ran 1 |
| `x__coloc_b` | 6.73 / 3.63 | 7.07 / 3.78 | 12.57 / 8.08 (β 0.2) | 12.32 / 7.81 (β 0.2) | 0.209 | modern | reproduction | capped / max_epochs; best 1 / ran 1 |
| `x__gowalla__mf__graphhash__directau__s0` | 6.94 / 3.77 | 7.34 / 3.91 | 12.99 / 8.37 (β 0.2) | 12.96 / 8.31 (β 0.5) | 0.231 | modern | reproduction | capped / max_epochs; best 2 / ran 2 |
| `x__gowalla__mf__graphhash__directau_dedup__s0` | 7.53 / 4.01 | — | — | — | — | legacy | reproduction | exploratory / fixed_epochs; best 30 / ran 30 |
| `x__gowalla__mf__graphhash__directau_dedup__s1` | 7.65 / 4.08 | — | — | — | — | legacy | reproduction | exploratory / fixed_epochs; best 30 / ran 30 |
| `x__gowalla__mf__graphhash__directau_dedup__s2` | 7.58 / 4.03 | — | — | — | — | legacy | reproduction | exploratory / fixed_epochs; best 30 / ran 30 |
| `x__yelp2018__mf__graphhash__bpr__s0` | 1.60 / 0.93 | 1.61 / 0.91 | 4.66 / 2.87 (β 0.2) | 4.36 / 2.73 (β 0.1) | 0.393 | modern | reproduction | capped / max_epochs; best 20 / ran 20 |
| `yelp2018__mf__dfreq__bpr__s0` | 2.55 / 1.71 | 1.90 / 1.25 | 2.52 / 1.72 (β 0.1) | 2.58 / 1.74 (β 0.1) | 0.145 | modern | reproduction | valid / patience; best 460 / ran 960 |
| `yelp2018__mf__dfreq__directau__s0` | 2.51 / 1.89 | 1.56 / 1.20 | 3.33 / 2.38 (β 0.2) | 3.17 / 2.27 (β 0.05) | 0.138 | modern | reproduction | valid / patience; best 15 / ran 45 |
| `yelp2018__mf__full__bpr__s0` | 8.02 / 5.13 | 5.54 / 3.27 | 8.50 / 5.43 (β 0.1) | 8.02 / 5.13 (β 0.0) | 0.970 | modern | reproduction | valid / patience; best 2150 / ran 2650 |
| `yelp2018__mf__full__directau__s0` | 11.10 / 7.20 | 6.97 / 4.24 | 11.26 / 7.35 (β 0.1) | 11.10 / 7.20 (β 0.0) | 0.912 | modern | reproduction | valid / patience; best 110 / ran 140 |
| `yelp2018__mf__graphhash__bpr__s0` | 3.78 / 2.36 | 2.03 / 1.16 | 4.82 / 3.03 (β 0.2) | 4.80 / 3.00 (β 0.5) | 0.190 | modern | reproduction | valid / patience; best 360 / ran 860 |
| `yelp2018__mf__graphhash__directau__s0` | 2.41 / 1.36 | 1.97 / 1.10 | 5.68 / 3.55 (β 0.2) | 4.96 / 3.08 (β 1.0) | 0.520 | modern | reproduction | valid / patience; best 50 / ran 60 |
| `yelp2018__mf__random__bpr__s0` | 0.55 / 0.32 | 0.38 / 0.23 | 2.15 / 1.41 (β 0.2) | 2.18 / 1.43 (β 1.0) | 0.075 | modern | reproduction | valid / patience; best 420 / ran 920 |
| `yelp2018__mf__random__directau__s0` | 0.42 / 0.25 | 0.29 / 0.18 | 2.39 / 1.62 (β 0.2) | 2.41 / 1.57 (β 2.0) | 0.353 | modern | reproduction | valid / patience; best 30 / ran 40 |
| `yelp2018__poponly__none__none__s0` | 1.68 / 1.05 | — | — | — | — | baseline | reproduction | valid / not recorded |

### 2.2 Retained scoring-control artifacts

| Checkpoint | Scored control cells | Lineage | Comparison | Source status / stopping | Session |
|---|---|---|---|---|---|
| `amazonbook__mf__full__directau__s0-legacy` | 23 | legacy | reproduction | valid / patience; best 25 / ran 35 | pc-x1-1017 |
| `amazonbook__mf__graphhash__bpr__s0-legacy` | 19 | legacy | reproduction | valid / patience; best 1450 / ran 1950 | pc-x2-1017 |
| `amazonbook__mf__graphhash__directau__s0-legacy41` | 19 | legacy | reproduction | capped / time_cap; best 40 / ran 41 | pc-x2-1017 |
| `gowalla__mf__dfreq__directau__s0-legacy30` | 19 | legacy | reproduction | valid-short / fixed_epochs; best 30 / ran 30 | pc-x2-1017 |
| `gowalla__mf__full__directau__s0-legacy120` | 23 | legacy | reproduction | capped / time_cap; best 120 / ran 130 | pc-x1-1017 |
| `gowalla__mf__graphhash__bpr__s0-legacy` | 19 | legacy | reproduction | valid / patience; best 330 / ran 430 | pc-x1-1017 |
| `gowalla__mf__graphhash__directau__s0-legacy30` | 19 | legacy | reproduction | valid-short / fixed_epochs; best 30 / ran 30 | pc-x1-1017 |
| `gowalla__mf__graphhash__directau__s1-legacy30` | 19 | legacy | reproduction | valid-short / fixed_epochs; best 30 / ran 30 | pc-x2-1017 |
| `gowalla__mf__graphhash__directau__s2-legacy30` | 19 | legacy | reproduction | valid-short / fixed_epochs; best 30 / ran 30 | pc-x2-1017 |
| `gowalla__mf__random__directau__s0-legacy30` | 19 | legacy | reproduction | valid-short / fixed_epochs; best 20 / ran 30 | pc-x1-1017 |

Every individual control cell remains in both CSVs, including negative, capped and fixed-budget results. Validation of a control file does not establish training convergence. E5 tail denominators, E7 precision deviations and storage scale, and E9 weighted/original-retuned-beta results have named columns in v2 and source-policy/status fields. Analysis-only legacy rows have no invented main score. The existing unconstrained learned-bias ablation remains distinct from the unexecuted C3 regularized residual model.

---
## 3. Invalid runs (never cite as results)
| Run | Why |
|---|---|
| GraphHash × BPR, batch 1024 (5.31 / 5.45) | Wrong setup for BPR |
| P1 BPR β-sweep on the Kaggle CPU image (val 4.24 at β = 0) | CPU image numbers buckets differently; the GPU rerun gave the 2.3 numbers |
| P1 GPU, first attempt | Google Drive HTTP 500 |
| AmazonBook BPR, patience 100 (3.93) | Stopped on an early spike |

Superseded-but-valid legacy rows (capped or short) are not in this table. They stay in §1–§2 with their caveats until their replacements exist.

---

## 4. Measured runtimes (one T4, legacy)
| Job | Measured |
|---|---|
| DirectAU MF, Gowalla | 37–39 s/epoch |
| DirectAU MF, AmazonBook | 112–119 s/epoch |
| BPR full batch, Gowalla / AmazonBook | 0.22 / 0.41 s/epoch |
| Kaggle job overhead (setup + final eval) | 2–4 min |
| Peak GPU memory, Full × DirectAU Gowalla | 380 MiB |

---

## N1. Novelty check (PREREG PR-N1)
| Paper | Verified? | What it shows | Overlap | Positioning |
|---|---|---|---|---|
| **PRISM**: Loveland, Ju, Zhao, Shah, Koutra, *On the Role of Weight Decay in Collaborative Filtering: A Popularity Perspective*, KDD 2025 (arXiv 2505.11318) | Yes (abstract + main text, 17 Oct; appendix not yet read) | Weight decay's main role is to encode popularity into embedding magnitudes. In DirectAU, in-batch updates grow magnitudes and weight decay shrinks them, giving a log-linear popularity–magnitude link. **Without weight decay the correlation is reported as much weaker.** Proposes a popularity-aware magnitude initialization. No hashing or sharing in the text read | **High** for "length encodes popularity in DirectAU" | Cite as the source of that fact. Explain our ρ = 0.815 at wd = 1e-8 (optimizer/lr difference? → D7). Our contribution is what **sharing** destroys |
| **DNA**: Choi, Yun, Jeong, Cho, *GCNs meet long-tail: Embedding norm bias in GCN-based recommendations*, Applied Soft Computing 186 (2026), doi 10.1016/j.asoc.2025.114226 | Yes (abstract) | Normalizes item embeddings and adds a learnable item-specific bias to restore popularity in GCN recommenders. No hashing in the abstract | **High** for "normalize + item bias" | Cite. Our learned-bias runs (P5, C3) are ablations relative to DNA; the post-hoc prior needs no training |
| Remaining list (PREREG PR-N1) | pending | | | |

**Verdict for G1.d:** pending (full PRISM appendix + citing papers).

---

## 5. Outcome blocks (append below, newest last)
<!-- Format:
### <PREREG ID>, <date time IST>
Prediction: … | Label: P/I/H/X | Observed: … | Verdict: PASS/FAIL/inconclusive | Runs: … | Stop reasons: … | Notes: …
-->

### PR-A6, 2026-10-17 18:10 IST
Prediction: popularity-only ≥ 2.0 points below the same dataset's GraphHash × DirectAU cos+prior | Label: P | Observed (test R@20 / NDCG@20): Gowalla 2.97 / 1.70 (reproduced), Yelp2018 1.68 / 1.05, AmazonBook 0.96 / 0.61. Evaluated test users: Gowalla 25,789, Yelp2018 30,304, AmazonBook 50,407 | Verdict: AmazonBook PASS (0.96 vs 6.33); Gowalla PASS (2.97 vs 13.29); Yelp pending A4 | Runs: local poponly.py | Notes: files results/A6_poponly_{Yelp2018,AmazonBook}.json

### PR-E2e (preliminary, Gowalla Full × DirectAU only), 2026-10-17 18:45 IST
Prediction: within-decile length shuffle |Δ| ≤ 0.5; global shuffle drop ≥ 3.0; fitted-popularity replacement |Δ| ≤ 0.7 | Label: P | Observed (local CPU, 2 seeds, **test**; Δ vs raw 17.895): decile shuffle −4.03; global shuffle −11.52; replacement by exp(a + g·p) **+0.79** | Verdict (preliminary): decile FAIL, global PASS, replacement FAIL by 0.09 **in the favourable direction** (popularity-predicted lengths score *higher* than the true lengths) | Runs: score_controls.py --e2e on gpu-output-v2 Gowalla_Full_DirectAU_seed0.pt | Notes: legacy_topk reproduces 17.895 / 12.996 / 19.544 exactly. Reading: a smooth function of popularity carries the useful length information (replacement ≥ original), but deciles are too coarse a bin (wide count range inside the top decile). Formal result (10 seeds, validation primary) comes from X1; prediction thresholds are not changed.

### PR-TIE, 2026-10-17 19:40 IST
Prediction: T-i boundary-tie users ≥ 50%; T-ii |random − default key| ≤ 0.2 | Label: P | Observed (Gowalla GraphHash × DirectAU s0/s1/s2): boundary-tie users **87%**, mean tied group at the 20th slot **14.7 items**; index-key − random-key **+0.04 / +0.05 / +0.06**; AmazonBook GraphHash × DirectAU 89% / 17.6 items. T-iii tie-break-only (popularity key) gain **+0.67 / +0.70 / +0.71** = **12%** of the full-repair gain (AmazonBook DirectAU 25%; Gowalla BPR 34%; AmazonBook BPR 60%) | Verdict: T-i PASS, T-ii PASS | Runs: pc-x1-1017, pc-x2-1017 controls | Notes: ties are pervasive, but resolving them explains only a small part of the DirectAU repair.

### PR-DEC, 2026-10-17 19:40 IST
Prediction: D-i within-bucket shuffle keeps ≤ 70% of Δ_both | Label: P | Observed, fixed β* (test Recall@20, Δ vs random-key base; Gowalla GraphHash × DirectAU mean of 3 seeds): Δ_μ **+1.74**, Δ_ε **+4.22**, Δ_both **+5.79**, interaction **−0.17**; Shapley shares **μ 0.29, ε 0.71**. Within-bucket shuffle: **−0.36** (keeps −6% of the gain; worse than no prior). AmazonBook GraphHash × DirectAU: μ 0.24, ε 0.76, shuffle −0.17. Gowalla GraphHash × BPR: Δ_μ −0.76, Δ_ε +3.23 (μ −0.52, ε 1.52). AmazonBook BPR: μ −0.26, ε 1.26. Random × DirectAU: μ 0.04, ε 0.96. Double-freq: only 79 items share an alias pair (40,902 groups), so ε ≈ 0 and the decomposition is degenerate | Verdict: D-i PASS (strongly). D-ii framing rule: **residual (item-level) share ≥ 0.5 on every GraphHash/Random checkpoint** | Notes: legacy_topk reproduces 7.58/7.56/7.55 and 13.26/13.31/13.30; recomputed GPU-image maps equal the stored P5/P4 aliases.

### PR-E2e (formal, 10 seeds), 2026-10-17 19:40 IST
Prediction: decile shuffle |Δ| ≤ 0.5 (Gowalla) / 0.3 (AmazonBook); global drop ≥ 3.0 / 0.5; replacement |Δ| ≤ 0.7 / 0.3 | Label: P | Observed (validation; test in brackets): Gowalla Full × DirectAU unmodified 17.62 [17.90]: decile shuffle 13.43 (Δ **−4.19**) [13.64], global 6.15 (**−11.47**) [6.28], replacement 18.45 (**+0.83**) [18.69]. AmazonBook Full × DirectAU 9.51 [9.57]: decile 9.18 (**−0.33**) [9.30], global 7.28 (**−2.23**) [7.44], replacement 10.59 (**+1.08**) [10.64]. Fit: log length = a + g·p with g 0.23 (Gowalla, R² 0.71), 0.063 (AmazonBook, R² 0.49) | Verdict: decile FAIL (both), global PASS (both), replacement FAIL (both) **in the favourable direction**: a smooth popularity function beats the learned lengths | Notes: deciles are too coarse; the replacement result says popularity is the useful content of length.

### PR-M1, 2026-10-17 19:40 IST
Prediction: M-i theorem check holds; M-ii η² (GraphHash Gowalla) ≤ 0.5; M-iii [X] product rule ±0.10 | Observed: η²_p = Var(μ)/Var(p): Gowalla GraphHash 0.359, Random 0.142, DF 0.999 (alias pairs ≈ unique); Yelp GraphHash 0.484, Random 0.199; AmazonBook GraphHash 0.358, Random 0.109. Gowalla GraphHash resolutions 400/800/1600/3200: η² 0.49/0.63/0.79/0.92. ρ_proxy: Gowalla GraphHash 0.189, AmazonBook 0.228, Yelp 0.468. M-i: ρ_item² (0.073 Gowalla, 0.036 AmazonBook) ≤ η² ✓. M-iii: Gowalla 0.189·0.815 = 0.154 vs ρ_item 0.27 (off 0.12, FAIL); AmazonBook 0.179 vs 0.189 ✓; Random 0.225 vs 0.274 ✓ | Verdict: M-i PASS, M-ii PASS (η²_p; rank version to add), M-iii mixed (exploratory) | Runs: c0_maps (pc-x1-1017).

### PR-A8, 2026-10-17 19:40 IST
Prediction: Full × BPR Gowalla ∈ [15.4, 15.8] | Label: I | Observed: raw **16.55** / 10.09 (val 16.66), still improving: best epoch 2790, **hit max epochs 3000** (not converged); cos 11.96; raw+prior 17.06 (β 0.5); ρ = **0.77** | Verdict: FAIL (above range, outside [15.0, 16.2] → audit). Likely cause: our patience-500 rule trains far longer than the authors' patience 50, which stops full-batch BPR during its noisy phase; no pipeline error found (GraphHash BPR reproduces 9.34 vs paper 9.30) | Notes: Full BPR's length also tracks popularity (ρ 0.77), as does Yelp Full BPR (0.97).

### Stage 2 BPR runs (inputs to PR-Y), 2026-10-17 19:40 IST
Yelp GraphHash × BPR (A5): converged (patience, best epoch 360): raw 3.78 / 2.36, cos 2.03, cos+prior 4.82 (β 0.2), **raw+prior 4.80** (β 0.5), ρ 0.19. Yelp Full × BPR (A7): converged (best 2150): raw 8.02 / 5.13, cos 5.54, cos+prior 8.50 (β 0.1), ρ 0.97. PR-Y verdicts wait for A3/A4.

### PR-N1 (first pass), 2026-10-17 20:00 IST
Kill-rule question: does any paper show that sharing/hashing destroys item-level popularity information AND repairs it with a per-item popularity term? | Observed: **no such paper found** (shallow sweep: citations of GraphHash and PRISM via Semantic Scholar + keyword searches; DNA citers not retrieved). Closest: (1) GraphHash (WWW 2025) reports the DirectAU drop, no mechanism; (2) PRISM (KDD 2025) has no hashing/sharing/compression anywhere, names no optimizer, gives no DirectAU numbers at λ = 0 (qualitative/figures only), so our ρ = 0.815 at wd 1e-8 cannot be compared numerically yet; (3) Cheng, arXiv 2606.21275 (June 2026): log p(i) score term on full models, no hashing (**not yet verified by us**). DNA verified earlier (CAU repository record: Choi, Yun, Jeong, Cho, ASOC 186, 2026). Bib fixes: Park et al. CIKM 2023 exact title; Wu et al. TOIS 2024; Wan et al. NeurIPS title differs from arXiv; Yi et al. first author to confirm | Verdict: G1.d provisional PASS; a deeper sweep is required before the paper submission | Details: results/N1_novelty.md

### PR-A1, 2026-10-17 20:05 IST
Prediction: raw ∈ [7.55, 7.85]; |raw − cos| < 0.15; cos+prior ∈ [13.3, 13.7] at β 0.2; ρ ∈ [0.25, 0.32]; PASS if raw ≤ 9.04 and stopped by patience | Label: I | Observed: **converged** (patience; best epoch 55, ran 60). Raw **7.55** / 4.06, cos 7.57, cos+prior **13.30** / 8.57 (β 0.2), raw+prior 10.33 (β 5.0, grid edge), ρ_item 0.275 | Verdict: **PASS** (G2.a): the anomaly survives convergence; all four predictions inside their ranges (raw at the lower edge) | Runs: gowalla__mf__graphhash__directau__s0 (pc-s2a-1017) | Notes: the 30-epoch legacy numbers were already at convergence (val curve flat from epoch ~20: 7.40 → 7.48).

### PR-I1, 2026-10-17 20:05 IST
Prediction: |I1 − A1| ≤ 0.3 for raw and cos+prior, both converged | Label: P | Observed: converged (patience; best epoch 50, ran 80). Raw **7.54** (Δ −0.01), cos+prior **13.35** (Δ +0.05), ρ_item 0.274 | Verdict: **PASS** (G2.b): GraphHash results are insensitive to the 2.64× initial-norm difference. (Not a sharing-vs-Full test; the controlled pair is I1 vs A2.)

### PR-H (Gowalla part), 2026-10-17 20:05 IST
Prediction: H-pool ρ_bucket ≥ 0.6 on A1; H-dir ρ_dir(BPR) > ρ_dir(DirectAU) | Label: H | Observed at the last evaluation: A1 ρ_bucket **0.925**, ρ_item 0.276, ρ_dir −0.057; I1 ρ_bucket 0.927, ρ_item 0.276, ρ_dir −0.055. Item-quintile mean lengths in A1 (Q1→Q5): 11.71, 12.07, 12.18, 12.29, 12.71 (narrow spread) | Verdict: H-pool **PASS (strong)**: a shared row's length tracks its bucket's total traffic almost perfectly, while the item-level link is weak; H-dir pending BPR norm logs (A5 shows ρ_item 0.19 for Yelp GraphHash BPR) | Notes: this is the direct evidence for "sharing pools the length signal at bucket level".

### PR-Y, 2026-10-17 21:15 IST
Prediction: Y1 A4 raw < A5 raw; Y3 |A4 raw − cos| < 0.3 and ρ(A4) ≤ 0.4; Y4 A4 cos+prior − A5 raw+prior ≥ 0.3 with CI excluding 0; Y2 [X] E ≥ 2.5 | Label: I / P / P / X | Observed (test R@20; val in brackets): A3 Full × DirectAU raw **11.10** [10.78], cos 6.97 [6.78], cos+prior 11.26 (β 0.1), ρ 0.911. A4 GraphHash × DirectAU raw **2.41** [2.43], cos 1.97 [1.99], cos+prior **5.68** (β 0.2), raw+prior 4.96 (β 1.0), ρ 0.520. A5 GraphHash × BPR raw 3.78, raw+prior **4.80** (β 0.5). A7 Full × BPR raw 8.02. Y4 paired user bootstrap (30,304 users, 2,000 resamples): **Δ +0.88, 95% CI [0.74, 1.02]**. A4 repair gain over raw +3.27 [3.11, 3.43]. Y2: G(A3) val 4.00 × (1 − 0.520²/0.911²) = **2.70** | Verdict: **Y1 PASS** (flip reproduced on Yelp: Full DirectAU 11.10 > Full BPR 8.02, GraphHash DirectAU 2.41 < GraphHash BPR 3.78). **Y3 FAIL** on both parts (gap 0.44, ρ 0.52). **Y4 PASS** (G2.c). Y2 reported (2.70) | Runs: yelp2018__mf__full__directau__s0 (stop patience, best ep 110), yelp2018__mf__graphhash__directau__s0 (patience, best ep 50), yelp2018__mf__graphhash__bpr__s0, yelp2018__mf__full__bpr__s0; session pc-s2b-1017 | Notes: Y3 failing is consistent with PR-M1: Yelp GraphHash buckets keep more popularity (η² 0.484 vs 0.36 on Gowalla/AmazonBook; ρ_proxy 0.468), so less is erased and ρ stays higher. The capacity bound still holds (ρ_item² 0.27 ≤ η² 0.48). Evaluated users: test 30,304, val 30,261. C0 buckets: 7,604 users / 7,656 items.

### PR-H (Yelp + H-Adam + H-dir), 2026-10-17 21:15 IST
Prediction: H-pool ρ_bucket ≥ 0.6 on A1 and A4; H-Adam per-quintile mean r² never drops > 1% between evaluations (A1, A3, A4); H-dir ρ_dir(BPR) > ρ_dir(DirectAU), same dataset and compression | Label: H | Observed (last evaluation): **H-pool** A4 ρ_bucket **0.944** (ρ_item 0.520); A1 0.925. **H-Adam** largest consecutive drop: A1 none (all increases), A3 −0.11%, A4 −0.15%; for contrast Full × BPR Yelp drops −19% (weight decay with no scale invariance). Quintile mean lengths Q1→Q5: A3 6.95→11.40, A4 1.89→2.52. **H-dir**: Yelp GraphHash BPR 0.352 vs DirectAU 0.164; Yelp Full BPR 0.375 vs DirectAU 0.137 | Verdict: **H-pool PASS (both), H-Adam PASS, H-dir PASS (2/2 available pairs)**; no Gowalla pair (legacy BPR runs lack norm logs) | Notes: Euler's identity predicts monotone length growth only for the scale-invariant DirectAU loss; the BPR counterexample supports this.

### PR-A6 (Yelp), 2026-10-17 21:15 IST
Observed: Yelp popularity-only 1.68 vs A4 cos+prior 5.68 (gap 4.00 ≥ 2.0) | Verdict: **PASS**; PR-A6 now PASS on all 3 datasets.

### PR-E5 — partial (Stage 1–2 runs), 2026-10-17, Codex local analysis

**Prediction (copied):** "for the GraphHash prior gains, ≥ 60% of the added hits fall in the top-20%-popularity items; bottom-80% recall changes by ≤ 0.5 points [Guessing]. Reported regardless."

**Observed:** From saved test top-20 lists and the original train/validation/test splits. DirectAU compares cosine to cosine+prior; BPR compares raw-dot to raw+prior. No new ranking or model training. All stored user coverage, hits, npos, NDCG and exclusion masks were checked against original split truth.

| Case | Main capped Recall@20, baseline → prior (%) | Head share of gross added hits (%) | Conditional tail recall Δ (points) | Tail contribution Δ (points) | Catalog coverage, baseline → prior (%) |
|---|---|---|---|---|---|
| Gowalla A1 DirectAU | 7.5666 → 13.3047 | 96.78 | −6.8077 | −3.7239 | 95.19 → 28.40 |
| Yelp A4 DirectAU | 1.9675 → 5.6823 | 98.48 | −2.0107 | −0.9009 | 96.25 → 21.65 |
| Yelp A5 BPR | 3.7845 → 4.8007 | 96.42 | −0.5250 | −0.2543 | 22.88 → 7.34 |
| AmazonBook legacy DirectAU | 4.6264 → 6.3275 | 89.45 | −1.3024 | −0.6248 | 93.11 → 80.68 |
| AmazonBook legacy BPR | 4.1128 → 4.8103 | 74.64 | −0.4715 | −0.2068 | 44.39 → 34.33 |

Activity-quartile changes in main capped Recall@20 (points; Q1 least active):

| Case | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| Gowalla A1 DirectAU | +6.382 | +5.462 | +5.762 | +5.346 |
| Yelp A4 DirectAU | +3.600 | +3.973 | +3.782 | +3.503 |
| Yelp A5 BPR | +0.722 | +0.987 | +1.252 | +1.103 |
| AmazonBook legacy DirectAU | +1.798 | +1.707 | +1.725 | +1.574 |
| AmazonBook legacy BPR | +0.779 | +0.669 | +0.740 | +0.602 |

**Verdict:** Head-share condition passes in all five cases. DirectAU's ≤0.5-point tail-change prediction **FAILS on all three datasets**, using either the conditional tail denominator or its contribution to overall capped recall. Yelp BPR fails under the conditional definition but passes using the overall contribution; that case is denominator-sensitive. AmazonBook BPR passes both conditions. This does not establish a tail-safe repair.

**Runs / stopping:** A1 `gowalla__mf__graphhash__directau__s0`, A4 `yelp2018__mf__graphhash__directau__s0`, A5 `yelp2018__mf__graphhash__bpr__s0` all stopped by patience. AmazonBook uses X2 saved legacy DirectAU (epoch-41 cap) and BPR (patience) controls, tie-key seed 0. They are not final Stage 3 replications.

**Notes:** Head = exact top ceil(0.2 × catalog size) by training count, with original item ID breaking boundary ties. Added hits are gross newly recovered true-positive (user,item) pairs; losses are separate. Conditional tail recall averages `tail_hits/min(20, tail_truth_count)` among users with tail truth. Tail contribution averages `tail_hits/min(20, total_truth_count)` over all test users. Equal-count activity quartiles use training counts and original user ID for ties. PR-E5 did not freeze these operational details; they are explicit analysis-time choices, not retroactive preregistration. Uncapped tail recall, all item quintiles, hit additions/losses and input lineage are in `results/E5_stage12.json`.

### PR-E7 — partial (Stage 1–2 runs), 2026-10-17, Codex local analysis

**Prediction (copied):** "|cos_prior_u8 − cos_prior| ≤ 0.02 on the Stage 1 checkpoints. The 'one byte per item' claim is made only if this passes. Until then the stated cost is float32: 163,924 B = 5.52% of the Gowalla GraphHash table."

**Observed:** Threshold is 0.02 **percentage points**, not 0.02 of a metric stored as a fraction. Existing GPU controls were reused for legacy checkpoints; missing quantized validation outcomes were not invented. Fresh modern checks hold the original validation-selected β fixed and compare float32 versus uint8 using the **same CPU arithmetic and torch.topk policy**.

| Checkpoint | Validation Δ (points) | Test Δ (points) | Scope / tolerance result |
|---|---|---|---|
| Gowalla legacy GraphHash DirectAU s0 | missing | −0.017336 | GPU mean over 10 tie keys passes test |
| Gowalla legacy GraphHash DirectAU s1 | missing | −0.004685 | GPU mean passes test |
| Gowalla legacy GraphHash DirectAU s2 | missing | −0.001398 | GPU mean passes test |
| AmazonBook legacy GraphHash DirectAU s0 | missing | +0.000463 | GPU mean passes test |
| Gowalla modern A1 GraphHash DirectAU | −0.026409 | +0.024596 | Both exceed absolute tolerance: FAIL |
| Yelp modern A4 GraphHash DirectAU | +0.008968 | +0.015300 | Both within tolerance: PASS |

**Verdict:** **Partial; no unconditional one-byte claim.** The four original requested GraphHash test means pass, but their quantized validation cells are unavailable. Modern Gowalla A1 fails the tolerance, including on validation. A positive change still fails an absolute-deviation rule. As sensitivities, original Gowalla seed 0 has an individual paired tie draw −0.025659 points; the Full Gowalla control changes +0.025691 points and fails if "Stage 1 checkpoints" is interpreted to include all Full controls. Neither sensitivity replaces the primary mean statistic.

**Runs / stopping:** Original Stage 1 controls retain their source short/capped/patience status. Modern A1 and A4 stopped by patience. No legacy model was ranked locally; modern checkpoints use stored aliases checked against saved mapping files. No GraphHash mapping was recomputed.

**Notes:** The tested storage format is one uint8 per item **plus one float32 scale**: 40,985 B Gowalla, 38,052 B Yelp, 91,603 B AmazonBook. Scale metadata is not free. Original GPU and fresh CPU results remain separate in `results/E7_stage12.json`, including same-policy and cached-versus-fresh diagnostics. Retain float32 as the currently supported universal prior-storage claim; any later precision choice needs an explicitly specified tolerance and fresh verification.

### PR-E9 — partial (Stage 1–2 runs), 2026-10-17, Codex local analysis

**Prediction (copied):** "Weights w_i = (1+c_i)^(−1/2); the per-user weighted recall equals standard recall when weights are equal. Report the original β and a β re-tuned on reweighted validation. Prediction [Guessing]: re-tuned β ≤ original β, and the K2 ordering is kept on Gowalla."

**Observed:** Counts use training events only. Per-user weighted recall is `sum_{i in Top20 ∩ T_u} w_i / sum_{i in T_u} w_i`, then macro-averaged over test users. The denominator includes **all** held-out positives, so equal weights recover **standard recall**, rather than the main capped metric. This is popularity-reweighted evaluation, not IPS or randomized-exposure evaluation.

| Case | Original β | Weighted test recall at original β (%) | Re-tuned β | Weighted test recall at re-tuned β (%) | Ranking policy |
|---|---|---|---|---|---|
| Gowalla A1 DirectAU | 0.2 | 9.6356 | 0.1 | 10.8214 | Same CPU arithmetic / topk for both columns |
| Yelp A4 DirectAU | 0.2 | 3.6890 | 0.2 | 3.6890 | Same CPU arithmetic / topk |
| Yelp A5 BPR | 0.5 | 2.8727 | 0.5 | 2.8727 | Same CPU arithmetic / topk |
| Gowalla legacy BPR | 0.5 | 8.0878 | missing | missing | Saved GPU top20 only |
| AmazonBook legacy DirectAU | 0.1 | 5.5421 | missing | missing | Saved GPU top20 only |
| AmazonBook legacy BPR | 0.2 | 3.9790 | missing | missing | Saved GPU top20 only |

**Verdict:** β shrinks on Gowalla A1 and stays equal for both Yelp runs: descriptive prediction passes for the three available modern checks. At the original β, saved GPU predictions give Gowalla DirectAU 9.6326 versus legacy BPR 8.0878, a +1.5448-point ordering consistent with K2. **Fully re-tuned Gowalla K2 is INCONCLUSIVE:** legacy BPR has no local weighted-validation predictions and is not ranked on CPU. No final all-dataset, all-seed or Holm-adjusted claim follows from this partial analysis.

**Runs / stopping:** Modern A1/A4/A5 stopped by patience. Legacy AmazonBook DirectAU is capped; other legacy labels are preserved. Modern test scores use only original β and the β selected from weighted **validation** over the pre-existing grid; test scores never select β. Each frozen configuration is evaluated as a requested control.

**Notes:** Cached GPU versus fresh CPU tie/arithmetic sensitivity is kept separate: original-β cached values are 9.6326 / 3.6852 / 2.8689 for A1/A4/A5; same-policy values are in the table. The three cached-versus-fresh main capped-Recall differences are only +0.00473 / +0.00374 / +0.00387 points, but are not attributed to β. Results, curves, denominator and provenance: `results/E9_stage12.json`. The combined E7/E9 local scoring checks used 49.51 seconds; no research training, network operation or mapping construction was performed.

**Analysis-artifact boundary:** E5/E7/E9 remain partial companion JSONs keyed to their input checkpoints. The current fixed `master_metrics.csv` schema represents ordinary capped-recall results and controls; it has no honest fields for E5 group/count measures or E9 weighted standard recall. Version the analysis-table schema explicitly before freezing these new measures as paper numbers; do not put them into ordinary `test_recall20` columns.

### PR-E5 addendum: Full-model reference for the tail trade-off (Yelp2018), 2026-10-17 22:10 IST
Observed (test; tail = bottom-80% items by train count; tail-recall = Codex's `tail_primary`; coverage = % of catalog appearing in any top-20): A3 Full DAU raw 11.10 / tail 1.03 / cov 32.0%; A3 cosine 6.97 / 7.01 / 97.8%; A3 cos+prior 11.26 / 2.29 / 43.9%. A7 Full BPR raw 8.02 / 0.12 / 9.2%. A4 GraphHash DAU cosine 1.97 / 2.30 / 96.2%; raw 2.41 / 1.47 / 61.0%; cos+prior 5.68 / **0.29** / 21.7%. A5 GraphHash BPR raw 3.78 / 0.80 / 22.9%; raw+prior 4.80 / 0.27 / 7.3% | Reading [X]: the tail loss measured against cosine is mostly the repair restoring a Full-model-like popularity profile (Full DirectAU raw is itself head-heavy). But the repair **overshoots** it: tail 0.29 vs Full 1.03, coverage 21.7% vs 32.0%. On Full, cos+prior is *more* tail-friendly than the learned lengths (2.29 vs 1.03). The paper must report the repair as a head-accuracy repair with a tail cost; no "free" or "tail-safe" claim. Script: ad hoc, reusing scripts/analyze_e5.py loaders.

### Stage 3 outcomes, 2026-10-18 04:45 IST (snapshot PREREG_20261017-2238, sha256 a3c1b585…775392)
All 25 runs pass validate_run.py. 24 stop on patience; **amazonbook__mf__full__bpr__s0 hit max epochs (3000; best ep 2690) → capped**, used descriptively only.

**PR-E8 (K2, primary): PASS.** GraphHash × DirectAU cos+prior vs GraphHash × BPR raw+prior, test R@20, paired user bootstrap (2,000) and paired t-test, Holm over 3 datasets:
| Dataset | DirectAU cos+prior | BPR raw+prior | Δ | 95% CI | Holm p |
|---|---|---|---|---|---|
| Gowalla s0 (A1 vs s0-p500) | 13.30 | 11.31 | +1.99 | [1.77, 2.24] | 2.4e-63 |
| Gowalla s1 | 13.32 | 11.19 | +2.13 | [1.91, 2.37] | — |
| Gowalla s2 | 13.35 | 11.27 | +2.08 | [1.85, 2.33] | — |
| Yelp2018 s0 | 5.68 | 4.80 | +0.88 | [0.73, 1.03] | 6.1e-33 |
| AmazonBook s0 (B9 vs legacy BPR-long) | 6.33 | 4.81 | +1.52 | [1.41, 1.63] | 1.2e-159 |
Every Gowalla seed's Δ > 0 with its own CI excluding 0. Seed spread (mean ± SD): DirectAU cos+prior 13.32 ± 0.03, BPR raw+prior 11.26 ± 0.06. Note: the AmazonBook BPR side is the Stage 1 control file (fixed β*, random-key tie order) rather than torch.topk order; tie effects measured in PR-TIE are ≤ 0.06.

**S3-A2 (Gowalla Full DirectAU, patience ep 160): PASS** — raw 18.02 ∈ [17.4, 18.5]; cos+prior 19.57 ∈ [19.0, 20.2]; ρ 0.81 ≥ 0.75; test gap 4.96. Controlled contrast vs I1 (same protocol): raw 18.02 vs 7.54.
**S3-A8r (Gowalla Full BPR, now patience at ep 2790): PASS** — raw 16.55 ∈ [16.3, 18.3]; raw+prior 17.06 ≥ raw. Converged value equals the earlier capped 16.55, so PR-A8's [15.4, 15.8] stays FAIL (our Full BPR is stronger than the published 15.62).
**S3-B9 (AmazonBook GraphHash DirectAU, patience ep 40): PASS** — raw 4.63 ∈ [4.3, 5.5]; cos+prior 6.33 ∈ [6.0, 7.2]; +1.52 over BPR 4.81.
**S3-B5–B8 (Gowalla seeds): PASS** — DirectAU raw 7.61 / 7.59, cos+prior 13.32 / 13.35; BPR raw 9.39 / 9.41 (s0-p500 9.37), raw+prior 11.19 / 11.27 (s0-p500 11.31). All inside the registered ranges.
**S3-I2 (AmazonBook init control): PASS** — |I2 − B9| raw 0.04, cos+prior 0.03; Full ρ 0.79 ≥ 0.70; GraphHash ρ 0.19 ≤ 0.49. Controlled AmazonBook contrast (same protocol): Full raw 9.57 vs GraphHash 4.63.
**S3-D7 (weight decay):** wd 0 → ρ 0.74 ≥ 0.70 **PASS** (raw 16.12, cos+prior 19.08): the length–popularity link persists without decay under our setup. wd 1e-4 collapses the model (raw 1.41, cos 2.75, cos+prior 8.69, ρ 0.22); reported, no ordering predicted.
**S3-B main grid (Random / Double-freq):**
| Cell | DAU raw | DAU cos+prior (β) | BPR raw | BPR raw+prior (β) | ρ DAU |
|---|---|---|---|---|---|
| Gowalla Random | 0.45 | 3.80 (0.2) | 0.76 | 3.42 (0.5) | 0.27 |
| Gowalla DF | 3.22 | 5.11 (0.2) | 3.93 | 4.11 (0.5) | 0.03 |
| Yelp Random | 0.42 | 2.39 (0.2) | 0.55 | 2.18 (1.0) | 0.35 |
| Yelp DF | 2.51 | 3.33 (0.2) | 2.55 | 2.58 (0.1) | 0.14 |
| AmazonBook Random | 0.15 | 1.55 (0.2) | 0.17 | 1.28 (1.0) | 0.24 |
| AmazonBook DF | 1.51 | 2.13 (0.2) | 1.28 | 1.35 (0.5) | 0.03 |
- B-i (cos+prior ≥ max(raw, cos) + 1.0): **6/8** compressed DirectAU cells (fails: AmazonBook DF +0.62, Yelp DF +0.82).
- B-ii (β ∈ {0.1, 0.2}): **8/8**.
- B-iii (ρ ≤ ρ_full − 0.3; ρ_full 0.81 / 0.91 / 0.79): **8/8**.
- B-iv (BPR raw+prior β ∈ [0.2, 0.5]): **6/9** (fails: Yelp Random 1.0, AmazonBook Random 1.0, Yelp DF 0.1).
- [X] Generalization: raw DirectAU < raw BPR in 5/6 Random/DF cells (AmazonBook DF the exception); repaired DirectAU > repaired BPR in **6/6**.


### S3-E5 final — 2026-10-18 11:46 IST

**Prediction (copied from the registration):** Prediction: for **all three datasets**, repaired GraphHash DirectAU `tail_primary` is **below Full DirectAU raw `tail_primary`**; report every final GraphHash seed/init variant, without selecting the most favourable checkpoint. Decision: report the per-dataset differences and paired-user CIs on the same eligible users, including failures of this directional prediction. Preserve the original PR-E5 negative result: the <=.5-point DirectAU tail-change prediction failed on all three pilot datasets. The paper calls the repair a **head-accuracy repair with a tail cost**; make no "tail-safe" or cost-free repair claim. Stage 3 remains an informed check, not a retroactive validation of the original tail prediction.

**Observed:** 30 lineages and 90 planned raw/cosine/repaired mode cells: **89 validated observations**, one missing legacy AmazonBook BPR cosine cell. All observed rankings were checked against original test truth, duplicate IDs, denominators, NDCG and train+validation masks. Tail-primary and coverage are distinct from ordinary capped recall.

| Dataset | Repaired GraphHash DirectAU run | Tail % | Full raw tail % | Δ points | Paired 95% CI | Direction |
|---|---|---|---|---|---|---|
| Gowalla | `gowalla__mf__graphhash__directau__s0` | 2.005 | 2.501 | -0.496 | [-0.677, -0.330] | PASS |
| Yelp2018 | `yelp2018__mf__graphhash__directau__s0` | 0.290 | 1.030 | -0.740 | [-0.830, -0.647] | PASS |
| AmazonBook | `amazonbook__mf__graphhash__directau__s0` | 3.781 | 5.407 | -1.626 | [-1.799, -1.453] | PASS |
| Gowalla | `gowalla__mf__graphhash__directau__s0-initmatch` | 1.986 | 2.501 | -0.515 | [-0.699, -0.344] | PASS |
| Gowalla | `gowalla__mf__graphhash__directau__s1` | 1.957 | 2.501 | -0.544 | [-0.718, -0.373] | PASS |
| Gowalla | `gowalla__mf__graphhash__directau__s2` | 1.983 | 2.501 | -0.518 | [-0.700, -0.349] | PASS |
| AmazonBook | `amazonbook__mf__graphhash__directau__s0-initmatch` | 3.760 | 5.407 | -1.647 | [-1.822, -1.477] | PASS |

**Verdict:** Gowalla PASS, Yelp2018 PASS, AmazonBook PASS under the registered negative direction. All seven retained GraphHash DirectAU variants have lower tail-primary than their same-dataset Full raw reference. This PASS establishes a **tail deficit**, not a tail-safe repair. CIs use 2,000 paired tail-eligible-user resamples, seed 0, and are descriptive; the directional rule is mean Δ < 0. The earlier ≤0.5-point tail-change pilot prediction remains failed.

**Runs / notes:** Full references are `gowalla__mf__full__directau__s0`, `yelp2018__mf__full__directau__s0`, `amazonbook__mf__full__directau__s0`. All these DAU comparisons stopped by patience. The wider ledger also retains same-loss Full BPR references, including the capped AmazonBook Full BPR and Gowalla `s0-long`. `results/E5_final.json` stores all 90 mode records, exact denominators, catalog coverage, lineage and source receipts; missing legacy cosine remains `missing_gpu`.

### S3-E7 final — 2026-10-18 11:46 IST

**Prediction (copied):** Informed threshold and decision: the "one byte per item" precision claim is **dropped unless every final GraphHash checkpoint passes absolute Recall@20 deviation <= .02 percentage points on both validation and test**, for its final repaired scoring mode (cosine+prior for DirectAU; raw+prior for BPR). Include every retained dataset, seed and initialization variant; one failure, a missing cell or an invalid checkpoint blocks the universal claim. Compare uint8 against float32 at the **same frozen validation-selected beta**, checkpoint, users, masking, arithmetic and tie policy; test never chooses beta or representation. The .02-point tolerance is [I], inherited after observing pilot outcomes, not a new prospective threshold. Any passing uint8 claim must still state the additional float32 scale.

**Observed:** all eleven modern final GraphHash checkpoints scored on validation and test using stored verified aliases; every float/uint8 pair shares its checkpoint, original validation-selected β, masking, CPU arithmetic and topk policy. Legacy AmazonBook BPR is not ranked locally; its quantized validation cell remains missing. Signed changes below are percentage points.

| Run | Primary repaired mode | Validation Δ | Test Δ | Tolerance |
|---|---|---|---|---|
| `gowalla__mf__graphhash__directau__s0` | cos_prior | -0.026409 | 0.024596 | FAIL |
| `gowalla__mf__graphhash__bpr__s0-p500` | raw_prior | 0.019056 | 0.004739 | PASS |
| `gowalla__mf__graphhash__directau__s1` | cos_prior | -0.023500 | -0.013114 | FAIL |
| `gowalla__mf__graphhash__bpr__s1` | raw_prior | -0.001381 | 0.023872 | FAIL |
| `gowalla__mf__graphhash__directau__s2` | cos_prior | 0.001384 | -0.018167 | PASS |
| `gowalla__mf__graphhash__bpr__s2` | raw_prior | 0.010358 | 0.001553 | PASS |
| `gowalla__mf__graphhash__directau__s0-initmatch` | cos_prior | -0.015783 | 0.006350 | PASS |
| `yelp2018__mf__graphhash__directau__s0` | cos_prior | 0.008968 | 0.015300 | PASS |
| `yelp2018__mf__graphhash__bpr__s0` | raw_prior | 0.001403 | -0.011454 | PASS |
| `amazonbook__mf__graphhash__directau__s0` | cos_prior | -0.015991 | -0.001621 | PASS |
| `amazonbook__mf__graphhash__directau__s0-initmatch` | cos_prior | -0.000530 | -0.007753 | PASS |
| `amazonbook__mf__graphhash__bpr__s0-legacy` | raw_prior | missing | 0.001169 | missing_gpu |

**Verdict: FAIL.** Three modern checkpoints fail at least one split: Gowalla DAU s0, DAU s1, and BPR s1. Eight modern checkpoints pass both splits. The twelfth legacy checkpoint lacks validation, so it cannot support universality either. The one-byte-per-item claim is dropped; a positive change still violates the absolute-deviation rule.

| Dataset | Float32 prior B | GraphHash table B | Float32 overhead % | Uint8 + scale B |
|---|---|---|---|---|
| Gowalla | 163924 | 2969088 | 5.521 | 40985 |
| Yelp2018 | 152192 | 3906560 | 3.896 | 38052 |
| AmazonBook | 366396 | 5020672 | 7.298 | 91603 |

**Notes:** float32 is the stated default; uint8 requires an extra four-byte float32 scale. CPU/GPU topk sensitivity is recorded separately from quantization. Source: `results/E7_final.json`; all final run IDs and scoring modes are shown above.

### PR-E9 final available checks — 2026-10-18 11:46 IST

**Prediction (copied):** Prediction [Guessing]: re-tuned β ≤ original β, and the K2 ordering is kept on Gowalla.

**Observed:** counts use train only. Weighted recall is the macro mean of recovered truth weight divided by **all held-out truth weight**, weights `(1+c)^(-1/2)`. Equal weights recover standard recall, not the main capped metric. Original β is frozen from unweighted validation; re-tuned β uses weighted validation only. The columns below show DAU/BPR values, in percent.

| Pair | Original β D/B | Original weighted D/B | Retuned β D/B | Retuned weighted D/B | Retuned Δ pp | Ordering |
|---|---|---|---|---|---|---|
| Gowalla s0 | 0.2/0.5 | 9.636/8.138 | 0.100/0.200 | 10.821/8.580 | 2.241 | PASS |
| Gowalla s1 | 0.2/0.5 | 9.653/8.002 | 0.100/0.200 | 10.902/8.562 | 2.340 | PASS |
| Gowalla s2 | 0.2/0.5 | 9.688/8.134 | 0.100/0.200 | 10.898/8.763 | 2.135 | PASS |
| Yelp2018 s0 | 0.2/0.5 | 3.689/2.873 | 0.200/0.500 | 3.689/2.873 | 0.816 | PASS |
| AmazonBook s0 | 0.1/0.2 | 5.543/3.979 | 0.050/missing | 5.531/missing | missing | INCONCLUSIVE |

**Verdict:** the final weighted-validation-retuned ordering is PASS on all three Gowalla seeds and on Yelp. AmazonBook is **INCONCLUSIVE for the fully retuned comparison**: its legacy BPR weighted-validation and selected-test cells require GPU rescoring. At the original β, saved GPU predictions retain a +1.565159-point DAU advantage on AmazonBook; this is a descriptive mixed-recorded-policy comparison. Original/retuned modern values use the same CPU scoring policy; cached GPU diagnostics are separate. AmazonBook DAU selected β .05 scores 5.530712 weighted test versus 5.542877 at its original β .1; the slightly worse test score is retained rather than changing the validation choice.

**Runs / notes:** final Gowalla BPR seed 0 is `s0-p500`, never the patience-100 pilot. `results/E9_final.json` lists all ten run IDs, validation curves, policies and missing cells. The two init variants belong to E7 sensitivity coverage rather than replacing a primary E9 pair. These are weighted-metric ordering checks, not randomized-exposure evidence or a new Holm-adjusted primary result.

### PR-E8 retained inference, secondary NDCG and AmazonBook robustness — 2026-10-18 11:46 IST

The retained implementation `scripts/analyze_e8_final.py` reproduces the five registered Recall comparisons. CIs use 2,000 paired-user percentile bootstrap resamples, seed 0; Recall and NDCG share paired user draws. Extremely small p values are from **two-sided paired t-tests**, then Holm-adjusted across the three dataset seed-0 pairs, not empirical bootstrap p values. NDCG has a separate secondary Holm family.

| Pair | DAU Recall % | BPR Recall % | Δ pp | 95% CI | Primary Holm p |
|---|---|---|---|---|---|
| Gowalla s0 | 13.305 | 11.315 | 1.990 | [1.771, 2.238] | 2.43e-63 |
| Gowalla s1 | 13.319 | 11.189 | 2.130 | [1.909, 2.370] | — |
| Gowalla s2 | 13.352 | 11.267 | 2.084 | [1.855, 2.329] | — |
| Yelp2018 | 5.682 | 4.801 | 0.882 | [0.735, 1.031] | 6.14e-33 |
| AmazonBook | 6.328 | 4.810 | 1.518 | [1.405, 1.629] | 1.19e-159 |

Secondary NDCG:

| Pair | DAU NDCG % | BPR NDCG % | Δ pp | 95% CI | Secondary Holm p |
|---|---|---|---|---|---|
| Gowalla s0 | 8.572 | 6.957 | 1.615 | [1.492, 1.743] | 7.86e-137 |
| Gowalla s1 | 8.555 | 6.891 | 1.663 | [1.543, 1.794] | — |
| Gowalla s2 | 8.576 | 6.884 | 1.692 | [1.573, 1.821] | — |
| Yelp2018 | 3.549 | 3.003 | 0.546 | [0.460, 0.630] | 7.96e-36 |
| AmazonBook | 4.464 | 3.285 | 1.180 | [1.107, 1.254] | 1.62e-208 |

**Corrected final Gowalla seed summary:** DirectAU repaired Recall **13.325 ± 0.024%** and BPR **11.257 ± 0.063%**, using sample SD. NDCG is 8.568 ± 0.011 versus 6.911 ± 0.040. This supersedes the earlier rounded 13.32 ± 0.03 statement. Source runs are DAU s0/s1/s2 and BPR s0-p500/s1/s2.

**AmazonBook robustness:** primary B9 versus the saved legacy BPR raw+prior random-key-seed-0 CI excludes zero. The same-checkpoint/base/β alternate legacy torch.topk summary retains Recall Δ **+1.516886** and NDCG Δ **+1.179439** points. All ten saved random-key aggregate Recall differences are positive, **+1.508196 to +1.524243** points. Only seed 0 has stored per-user full-prior predictions. Alternative-policy paired CIs therefore remain **unavailable**, not inferred from small aggregate changes. `base_popkey` has β=0 and cannot substitute for an alternate full-prior ranking.

**Sources:** exact pair IDs and validated input receipts are in `results/E8_final.json`. AmazonBook BPR is `amazonbook__mf__graphhash__bpr__s0-legacy`, NPZ `runs/_sessions/pc-x2-1017/controls/amazonbook__mf__graphhash__bpr__s0-legacy/per_user__full_prior_s0.npz`. Proposed missing scoring controls are specified in `prompts/post_stage3_gpu_rescore.md`; nothing is queued or pushed. User bootstrap uncertainty is conditional on fixed trained checkpoints and does not replace training-seed variability. D7 establishes persistence at wd=0 under this setup, not a controlled contradiction of PRISM or proof of an optimizer mechanism.

### Stage 4 outcomes, 2026-10-18 22:20 IST (snapshot PREREG_20261018-1355, sha256 7a115933…e36a86a)
All 34 training runs + 1 rescoring job pass validation; 0 failed jobs. **Capped (time_cap):** LightGCN Gowalla Full BPR (ep 345/1000), LightGCN Gowalla Full DirectAU (ep 182/200), LightGCN Yelp Full BPR (ep 169), LightGCN Yelp Full DirectAU (ep 99), LightGCN Yelp GraphHash DirectAU (ep 102), C1 Gowalla r1600 DirectAU (ep 185). Capped runs are INCONCLUSIVE for convergence.

**S4-SEED: PASS.** Test R@20, paired bootstrap 2,000:
| Pair | DirectAU cos+prior | BPR raw+prior | Δ | 95% CI |
|---|---|---|---|---|
| Yelp s1 | 5.69 | 4.83 | +0.86 | [0.71, 1.01] |
| Yelp s2 | 5.65 | 4.82 | +0.83 | [0.69, 0.98] |
| AmazonBook s1 | 6.28 | 4.79 | +1.48 | [1.38, 1.60] |
| AmazonBook s2 | 6.30 | 4.75 | +1.54 | [1.44, 1.66] |
All values inside the registered ranges. With Stage 3, PR-E8 now holds on **every seed of every dataset** (3 × 3 pairs).

**S4-AMZBPR: PASS.** AmazonBook Full BPR long stops on patience (best ep 3800): raw 8.38 (≥ 8.37); raw+prior 8.38.

**S4-RESCORE: PASS (all three).** AmazonBook B9 DirectAU vs legacy BPR, full prior at β 0.2, every tie policy (index, popularity, torch.topk, random seeds 0–9): Δ +1.51 to +1.52, every CI ⊂ [+1.39, +1.63]. E7 uint8: val −0.0011 pp, test +0.0021 pp (≤ 0.02 ✓; one checkpoint, the universal one-byte claim stays dropped). E9 weighted recall after weighted-val retuning: DirectAU 5.53 (β 0.05) vs BPR 4.04 (β 0.05) → DirectAU still wins.

**S4-C3: FAIL.** The learned score (cos + β̂·p + r) is far below the post-hoc repair: Gowalla 4.56 vs 13.21; Yelp 2.68 vs 5.68; AmazonBook 1.63 vs 6.25. Learned β̂ > 0 on all 3 (0.90 / 0.93 / 0.93 ✓). Post-hoc modes of the same runs are normal (13.21 / 5.68 / 6.25). Reading: the jointly trained residual r_i hurts ranking badly; an implementation issue is not excluded (code review pending), so report as "learned variant did not work in our implementation".

**S4-LGN: L-i, L-ii, L-iii PASS** (Yelp GraphHash DirectAU and both Full rows capped).
| Run | raw | cos | cos+prior | raw+prior | ρ | stop |
|---|---|---|---|---|---|---|
| Gowalla Full BPR | 18.44 | 13.65 | 18.89 | 18.44 | 0.86 | cap |
| Gowalla Full DirectAU | 17.94 | 13.32 | 19.78 | 17.99 | 0.66 | cap |
| Gowalla GraphHash BPR | 16.22 | 10.52 | 15.79 | 16.27 | 0.38 | patience |
| Gowalla GraphHash DirectAU | 11.29 | 10.95 | **17.06** | 12.38 | 0.15 | patience |
| Yelp Full BPR | 9.55 | 6.89 | 9.48 | 9.55 | 0.88 | cap |
| Yelp Full DirectAU | 10.70 | 7.11 | 11.43 | 10.82 | 0.78 | cap |
| Yelp GraphHash BPR | 6.96 | 3.91 | 6.95 | 6.96 | 0.50 | patience |
| Yelp GraphHash DirectAU | 4.71 | 5.09 | **9.12** | 5.51 | 0.13 | cap |
The flip appears with LightGCN too (GraphHash DirectAU raw < BPR raw on both datasets), and the repair reverses it (17.06 > 16.27; 9.12 > 6.96). LightGCN GraphHash BPR is far stronger than MF (16.22 vs 9.37): propagation partly restores item-level differences.

**S4-DG (DoubleGraph, BACO substitute): PASS.** Gowalla DirectAU raw 3.54, cos 5.84, cos+prior 10.91 vs BPR raw+prior 7.27; Yelp 1.12 / 2.17 / 5.52 vs BPR 2.78. ρ ≈ 0 (−0.09 / −0.06).

**S4-C1 (resolution sweep, Gowalla GraphHash DirectAU): gain PASS, ρ_item FAIL.**
| Resolution | 200 (A1) | 400 | 800 | 1600 (cap) | 3200 |
|---|---|---|---|---|---|
| raw | 7.55 | 7.49 | 7.86 | 8.62 | 10.87 |
| cos+prior | 13.30 | 13.15 | 13.15 | 14.02 | 15.86 |
| gain | 5.75 | 5.66 | 5.29 | 5.40 | 4.99 |
| ρ_item | 0.28 | 0.44 | 0.56 | 0.57 | 0.54 |
| BPR raw / raw+prior | 9.37 / 11.31 | 9.79 / 10.96 | 10.42 / 11.16 | 11.57 / 11.77 | 12.61 / 12.76 |
Spearman(resolution, gain) = −0.90 (≤ −0.8 ✓). Spearman(resolution, ρ_item) = +0.70 (< +0.8, FAIL; it plateaus after 800). Repaired DirectAU beats repaired BPR at every resolution.

**S4-C4 [X]: as predicted.** Full equal-norm: raw = cos = 12.77, cos+prior 19.32 (≥ raw + 3 ✓; close to Full DirectAU's own 19.57). GraphHash equal-norm: cos+prior 13.27 (A1 13.30 ✓).
