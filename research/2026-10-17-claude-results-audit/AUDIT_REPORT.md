**Audit of the supplied Claude / Claude Code discussion**

Date: 17 October 2026. Scope: read-only inspection of eight existing private Kaggle jobs, their downloaded scalar outputs, decoded submitted scripts and checkpoint inventories; arithmetic on the reported seed results; comparison with previously inspected primary code and literature. No new training, rescoring job, cancellation or external write was performed by this audit.

**Verdict**

[Certain] The author-config BPR result and the small 30-epoch fix effects are backed by completed job outputs. Several stronger interpretations in the conversation are not supported: the fully trained loss × compression interaction has not been established by these runs, the collision mechanism has not been refuted, and the requested three-seed cosine check actually rescored only the last retained DirectAU seed.

[Likely] Parking the unique-bucket fix as the central main-track contribution is a defensible resource decision, given its small observed accuracy benefit and close prior art. That decision must not be represented as a proof that every longer-training outcome fails or that the underlying mechanism is impossible.

**1. Verified results**

All Recall values below use the released capped denominator hits/min(20, number of positives), in percentage units. They are not standard uncapped Recall. Standard Recall is also present in the source JSON outputs.

| GraphHash / Gowalla condition | Test Recall@20 | Test NDCG@20 | Training/selection | Recorded loop time |
|---|---:|---:|---|---:|
| BPR, published command settings | 9.3423% | 5.4938% | Best validation epoch 330; stopped at 430 | 93.72 s |
| BPR, minibatch fast-check setting | 5.4536% | 3.1309% | Best validation epoch 5; stopped at 15 | 26.77 s |
| DirectAU, three seeds | 7.5546% mean | 4.0381% mean | Final epoch 30 | 19.13 min mean |
| Unique-bucket DirectAU, three seeds | 7.5891% mean | 4.0419% mean | Final epoch 30 | 14.10 min mean |

[Certain] The full-batch BPR result is close to GraphHash Table 1's 9.300%. The two DirectAU arms use a fixed 30-epoch budget and no early stopping or gamma search.

[Certain] The fast-check BPR comparison changes both batch size and weight decay: batch 1024 / wd 1e-8 versus full batch / wd 1e-6. It also changes the number of optimizer steps per epoch by about 803-fold. It establishes a result for that optimizer regime, not that minibatch BPR is intrinsically broken. The poor result should remain in the exploratory record with its configuration.

[Certain] The reported loop timers include validation and final test evaluation, while setup occurs before the timer. Their mean reduction for the fix is 26.31% at the same 30-epoch budget. This is a recorded runtime benefit, not a controlled general hardware speedup claim or a novelty result.

**2. The test improvement is not positive in every seed**

| Seed | Plain DirectAU test R@20 | Fix test R@20 | Fix minus plain, percentage points |
|---|---:|---:|---:|
| 0 | 7.559249% | 7.534841% | -0.024408 |
| 1 | 7.572057% | 7.652829% | +0.080771 |
| 2 | 7.532547% | 7.579670% | +0.047123 |
| Mean | 7.554618% | 7.589113% | +0.034495 |

[Certain] Validation gains are positive in all three seeds, averaging +0.074243 percentage points. Test gains are positive in two of three seeds. The mean test gain is +0.034495 percentage points, with paired sample SD 0.053715 percentage points.

[Certain] A conventional paired t interval with n=3 and its normality assumptions gives an exploratory 95% interval of [-0.09894, +0.16793] percentage points. This small-sample calculation is not strong evidence for a stable positive population effect. Comparing the two arms' separate SDs is not the correct paired uncertainty calculation.

[Certain] Under the currently observed short-budget test comparison:

\[
G=9.3423276-7.5546180=1.7877096\ \text{percentage points},
\]
\[
\Delta=7.5891133-7.5546180=0.0344953\ \text{percentage points},
\qquad \Delta/G=1.9296\%.
\]

[Certain] Thus descriptive test recovery is about 2%, not a calculation mixing the validation numerator with the test denominator. A validation-only recovery calculation would be different. Neither calculation establishes the converged recovery or the originally proposed best-gamma gate: gamma was fixed at 1 here.

**3. The cosine check has finished, but it covers one DirectAU seed**

| Retained model | Raw test R@20 | Cosine test R@20 | Change, percentage points | Raw NDCG | Cosine NDCG |
|---|---:|---:|---:|---:|---:|
| BPR, author config, seed 0 | 9.342328% | 6.082944% | -3.259383 | 5.493753% | 3.076618% |
| DirectAU, seed 2, epoch 30 | 7.532547% | 7.545586% | +0.013039 | 4.014800% | 4.019878% |

[Certain] Each raw re-score matches its original recorded test metrics. This supports consistency of the loaded checkpoint, regenerated mapping and evaluation for those two retained models. It is not a three-seed replay.

[Certain] The DirectAU score does not jump toward 9%. Cosine changes it by only +0.0130 percentage points for the available seed. BPR becomes much worse with cosine. A reversal of BPR/DirectAU ordering under cosine would therefore not, by itself, mean that DirectAU's accuracy was restored.

[Certain] Both the three-seed training script and the fix script save each improving model to the same path:

**R.WORK / best.pt**

The loop runs seeds 0, 1 and 2 sequentially. The output inventory contains one model checkpoint in each multiseed job, and the raw re-score numerically matches seed 2. Earlier checkpoint states were overwritten. The scalar JSON results for all three seeds remain available.

[Certain] The single-seed precursor jobs may preserve separate seed-0 checkpoints, but they are not what this one-file multiseed rescoring job loaded. No seed-1 checkpoint was found in the reviewed multiseed artifacts. The rescoring result must be labeled one seed, not three.

[Certain] The normalized-training/raw-dot-inference convention is also present in the original DirectAU author code. It is not an established GraphHash-specific implementation defect:

https://github.com/THUwangcy/DirectAU/blob/60c964bb07abc0eda9b5044b6843fd9c4476a167/recbole/model/general_recommender/directau.py

**4. Which claims are too strong**

| Claim in the discussion | Supported conclusion |
|---|---|
| “The anomaly is real” from 9.34 versus 7.56 | [Certain] The optimized BPR reference beats this 30-epoch DirectAU setting. DirectAU's curve is still rising and the full-model loss comparison is not completed in these artifacts. This is not a completed four-cell loss × compression interaction study. |
| “If the fix helps, cause found” | [Certain] Unique-bucket uniformity changes pair weighting as well as constant collision contributions. Improvement alone would not identify the mechanism. |
| “Collision mass is not the main cause because the fix is weak” | [Certain] The intervention is not isolated, gamma is not fairly tuned, original-ID deduplication and adaptive-weight controls are absent, and the logged duplicate quantity is not pair collision mass. The causal exclusion does not follow. |
| “Converging DirectAU cannot change the decision because the fix will still add 0.05” | [Certain] The premise that its gain stays fixed with training is untested. A decision to stop spending can be reasonable; the claimed proof about every possible outcome is invalid. |
| “Nobody explained or fixed it” | [Certain] An exhaustive absence claim has not been established. SUAU is a direct uniqueness/uniformity precursor and needs an explicit comparison. |
| “The fast-check BPR isn't a real result” | [Certain] It is a real result for a different batch/regularization/update regime. It should not replace the published-config baseline or be generalized to all minibatch BPR. |
| “Full DirectAU is already done” | [Certain] If that refers to our previous Full/DirectAU job, it timed out with validation observations and no test result. The reviewed jobs do not supply a completed Full BPR/DirectAU test comparison. |

[Certain] Equal epochs across BPR and DirectAU are not required. Here BPR full-batch has roughly one optimizer step per epoch, while minibatch DirectAU has about 803. The unresolved issue is adequate convergence or an explicitly defined compute-budget comparison, not simply making the epoch counts equal.

**5. The duplicate statistic does not measure the proposed collision floor**

[Certain] The fast scripts log:

\[
1-\frac{\#\{\text{unique bucket IDs in the batch}\}}{M}.
\]

This is the fraction of redundant rows. It combines repeated original IDs and different IDs sharing parameters. It is not

\[
\rho=\frac{\sum_b c_b(c_b-1)}{M(M-1)},
\]

the fraction of batch-position pairs tied by the representation map.

[Certain] For example, two buckets occupied twice give row redundancy 1/2 but pair collision fraction 1/3. A full interaction batch can have over 99% repeated rows while its pair collision fraction is much smaller. The logged arrays do not provide Z or the attenuation factor alpha and do not isolate extra alias collisions.

[Certain] A forced identical-pair kernel term has zero gradient. Its constant contribution can attenuate the noncolliding-pair gradient through the log mean, as previously derived. Describing this as a nonzero force persistently trying to separate the same vector is misleading.

**6. Novelty and the resource decision**

[Certain] SUAU, “Towards similar alignment and unique uniformity in collaborative filtering,” already identifies duplicated samples in uniformity and introduces a uniqueness strategy. Its author code deduplicates original IDs. Compression-alias deduplication is not automatically identical, but the related-work overlap prevents an unqualified new-uniqueness claim.

- Publisher: https://doi.org/10.1016/j.eswa.2024.125346
- Code: https://github.com/ZzYUuuu/SUAU/blob/d1062bd7fedc137894dcd02bf846dd1e1261aec7/SUAU-main/model/SUAU.py

[Likely] Given the user's short experiment budget, the small observed test benefit and the prior-art obstacle justify parking this simple fix as the main-track centerpiece. That does not scientifically refute the broader loss × sharing question. The recorded runtime reduction may remain useful engineering evidence.

[Certain] No current result establishes that the ArgKP backup will show its proposed effect or lead to an accepted paper. Treat that as a separate, untested candidate.

**7. Evidence and limits**

Reviewed jobs:

- University: graphhash-fast-bpr-author-1016; graphhash-fast-bpr-long-1016; graphhash-fast-dau-1016; graphhash-rescore-bpr-author-1016.
- Father: graphhash-fast-dau-s012-1016; graphhash-fast-dau-fix-s012-1016; graphhash-fast-dau-fix-1016; graphhash-rescore-dau-s012-1016.

[Certain] All eight were COMPLETE at their latest reviewed state. Account identifiers are represented by aliases here. The raw scalar outputs, source blobs, decoded scripts and retrieval receipts are cached under the corresponding alias/slug subfolders. paired_analysis.json contains reproducible arithmetic. audit_manifest.json records source/output hashes and latest statuses.

[Certain] Verification level: completed output inspection, source-path review, paired arithmetic and consistency of the two raw re-scores with the original metrics. I did not independently replay every original seed's test prediction; the per-seed checkpoints were not retained separately. No new training or scoring job was launched for this audit.
