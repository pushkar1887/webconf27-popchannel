# KuaiRand-Pure: executed support audit and minimal baseline probe

Date: 15 October 2026. Scope: the user-approved next step, a local feasibility probe. All records are real released data. All models below are smoothed count-based component baselines, not learned embedding models or reproductions of GraphHash, CCE, or BACO.

**[Likely] Verdict:** the data support a narrow chronological native-feedback prediction pilot after explicit integrity amendments. Context-specific user effects improve the standard-traffic count baseline, but the ordering reverses against item-only/random-sharing baselines on random exposures. This does not establish the proposed compression mechanism, a context-frequency-only effect, or a novel method. Do not launch the full campaign on the strength of this probe.

## 1. Provenance and execution

[Certain] Downloaded the published 47,432,272-byte archive from Zenodo record 10439422; its MD5 matches the published checksum. Extracted only the three interaction CSVs. DOI: 10.1145/3511808.3557624. Record license: CC BY 4.0. The exact files, URLs, hashes and retrieval time are in release_receipt.json; the source record is in zenodo_record.json.

[Certain] Runtime: Python 3.10.9, NumPy 2.2.6, pandas 2.3.1. No dependency installation, Kaggle operation, NVIDIA-key access, model download, GPU run or paid inference occurred. The Sonatype checking connector was unavailable; existing runtime libraries were used without a new dependency recommendation.

[Certain] Measured durations: download 69.82 s; successful support audit 13.40 s; count fitting/tuning/evaluation 10.96 s. These exclude investigation, scripting, failed integrity attempts and verification. They do not estimate neural baseline runtime.

## 2. Integrity findings and recorded amendments

[Certain] Raw rows: **2,622,668**. Quarantined **46,885 rows** belonging to **23,402** repeated (user_id, video_id, time_ms, tab) groups. Clean cohort: **2,575,783**. Every row in such a group was quarantined, regardless of label. This conservative choice does not establish that every repeated key is an erroneous event.

[Certain] The native long_view label disagrees with the documented duration formula on **67,596 raw rows**: 62,968 with zero duration and 4,628 with positive duration. Labels were retained, not recomputed. No row was filtered according to target-rule agreement. Positive-duration sensitivity metrics are saved in baseline_results.json. Exact operational label semantics remain partly unresolved.

[Certain] Splitting by the supplied date field did not produce strict raw event-time separation. The amended clock is Unix time_ms interpreted in UTC, with fixed calendar boundaries. No model had been fitted when either amendment was recorded. Protocol.json is the original record; protocol_amendment_01.json and protocol_amendment_02.json document the changes.

[Certain] Only user_id, video_id and tab enter the fitted count predictors. Duration is used for an outcome-independent sensitivity subset; UTC hour bins are audited separately. Current play time, all other current feedback, all side-information files, and month-aggregated statistics are excluded from the models. The target is native long_view.

## 3. Canonical chronological splits

| Split | Clean rows | Users | Native positive rate | Warm user/item event coverage |
|---|---:|---:|---:|---:|
| train | 1,051,988 | 25,977 | 34.4960% | 100.0000% |
| validation | 59,072 | 16,958 | 32.4519% | 98.8590% |
| test_standard | 278,684 | 25,748 | 32.4091% | 96.3001% |
| test_random | 1,186,039 | 27,285 | 8.4963% | 95.6314% |

[Certain] Cutoffs are train before 2022-04-19 00:00 UTC, validation from then until 2022-04-22 00:00 UTC, and evaluation from then through the released period ending 8 May. Training and validation use standard traffic. Standard and random evaluation policies are reported separately. Clean raw timestamp ranges do not overlap between train, validation and evaluation.

## 4. Support for conditional user behavior

| Minimum observations in each of at least two training interfaces | Users | Fraction of training users | Standard evaluation event coverage | Random evaluation event coverage |
|---|---:|---:|---:|---:|
| 3 | 9,935 | 38.25% | 46.98% | 37.51% |
| 5 | 7,295 | 28.08% | 37.56% | 27.32% |
| 10 | 4,229 | 16.28% | 25.04% | 16.11% |
| 20 | 1,847 | 7.11% | 13.06% | 7.50% |

[Certain] The random evaluation is **99.3243% tab 1**. The median training user has two observed interfaces, and the median observed user/interface pair has eight records. Most users do not have strongly supported responses in several interfaces. Additional hour-bin support statistics are in support_audit.json; they did not change the primary tab context.

[Likely] This limits a broad rare-context claim. Start with a training-defined supported cohort and report its coverage; do not silently discard sparsely observed users or treat missing behavior as agreement. The random policy provides an exposure-change test, not a clean isolated manipulation of context frequency.

## 5. Minimal baseline results

[Certain] The models use shrinkage probability tables. Combined models add a validation-selected user log-odds effect to the item/interface log odds. Smoothing candidates are 10, 50 and 200; effect weights are 0, .25, .5 and 1. Three random-sharing assignments use 256 buckets. All hyperparameters were selected on validation only; validation_selection.json was saved before test evaluation. There was no subsequent test-driven refit.

**Lower log loss is better.** Probability-array bytes are not total process memory or a neural-compression benchmark.

| Count baseline | Standard log loss | Random log loss | Fitted probability-array bytes |
|---|---:|---:|---:|
| Global training rate | 0.630888 | 0.477543 | 4 |
| Interface/context rate | 0.599856 | 0.527011 | 60 |
| Item + interface counts | 0.570253 | 0.500446 | 455,040 |
| User + interface counts | 0.566008 | 0.558211 | 1,637,160 |
| Item/context + user-marginal effect | 0.555486 | 0.541103 | 564,180 |
| Item/context + user/context effect | 0.550850 | 0.526277 | 2,092,140 |
| Item/context + 256 random user buckets, seed 11 | 0.569752 | 0.500962 | 470,400 |
| Item/context + 256 random user buckets, seed 29 | 0.569503 | 0.501059 | 470,400 |
| Item/context + 256 random user buckets, seed 47 | 0.569739 | 0.500767 | 470,400 |

[Certain] Per-context metrics, Brier scores, calibration summaries, warm-user/item results, and the positive-duration subset are saved in baseline_results.json. These comparisons have unequal table capacities. They show what the specified count models do; they do not demonstrate the superiority of a compression algorithm.

### Paired component contrasts on warm-user/warm-item rows

[Certain] Intervals use 500 paired user-cluster bootstrap replicates. They condition on these periods, models and hyperparameter choices; they do not account for every source of temporal dependence or establish generalization across future periods.

| Loss difference, positive favors individual user/context effects | Standard mean [95% interval] | Random mean [95% interval] |
|---|---|---|
| Item/context baseline minus individual user/context | 0.020137 [0.018527, 0.021744] | -0.026598 [-0.030769, -0.022630] |
| User-marginal control minus user/context | 0.004817 [0.004176, 0.005446] | 0.015305 [0.013951, 0.016643] |
| Mean loss across random assignments minus user/context | 0.019472 [0.017936, 0.020960] | -0.026111 [-0.030178, -0.022258] |

[Likely] The standard-traffic result supplies a reason for a small next-stage probe: the contextual user-count effect improves over both item/context and a marginal-user control. Its larger capacity and smoothing behavior remain alternative explanations, so this is not evidence for the proposed shared-embedding mechanism.

[Certain] The ordering against item-only and random-sharing controls reverses on random exposures. The native positive rate is also much lower there. The fitted models were not recalibrated on random traffic; policy/exposure differences and calibration are confounded with any proposed context mechanism. A claim that user-specific representations are universally more robust is contradicted by this probe.

## 6. Baseline artifact availability

| Author repository | Read-only revision pin | Executed in this probe? |
|---|---|---|
| https://github.com/snap-research/GraphHash | 7a1d5eb48dce9ab4c4a453d4482dd0d3bdec4467 | No |
| https://github.com/thomasahle/cce | 5b8a93c1a3f4d3741872338d659702aee3ddae34 | No |
| https://github.com/HKBU-LAGAS/BACO | b6691fa64daedd9d5fd41f87b3988e7bfcdace68 | No |

[Certain] GraphHash declares an older pinned GPU environment; CCE provides a compressed-embedding interface with clustering calls; BACO documents a Cython compilation step and preprocessed-data workflow. Their README/manifests were saved as source receipts. Availability is not executable compatibility with this feedback task. Task adapters, environment validation and a shared neural backbone remain unfinished.

## 7. What this proves, and what remains open

- [Certain] Executed: official-release checksum verification, real-data support/integrity audit, chronological cleaned-cohort construction, validation-selected count-baseline evaluation, saved-prediction checks, and read-only author-code availability checks.
- [Certain] Not executed: a learned embedding model, GraphHash/CCE/BACO reproduction, behavior-aware sharing method, equal-total-byte comparison, maintenance-cost benchmark, pure context-frequency intervention, or independent-dataset confirmation.
- [Likely] Next experiment: a small common-backbone study on the training-defined supported cohort, comparing an uncompressed user representation, fixed/random sharing, an existing dynamic comparator, and conditional-count grouping before adding a proposed mechanism.
- [Likely] Do not use random-traffic rank reversal as evidence of the proposed mechanism. First separate calibration/exposure effects; keep future confirmation independent of these now-consumed exploratory evaluation sets.
- [Certain] This probe does not establish novelty, causal preference changes, user welfare, or main-track readiness. It resolves part of the data/execution uncertainty while exposing real limitations.

## 8. Reproduction and verification

Run with the existing recorded Python environment from this directory:

```powershell
python -X utf8 .\fetch_release.py
python -X utf8 .\audit_support.py
python -X utf8 .\count_baselines.py
python -X utf8 .\verify_probe.py
python -X utf8 .\make_report.py
```

[Certain] Saved-artifact verification status: **saved_artifact_checks_passed**. Maximum metric recomputation error: 8.33e-17; maximum saved-model replay error: 2.97e-08. The documented target-rule consistency check remains failed and is explicitly preserved in verification.json.

[Certain] Original records and source hashes are preserved. Scripts are a throwaway research-probe harness, not production software. No paper result should be copied without retaining the split, cohort, target, amendments, selection and uncertainty qualifications above.
