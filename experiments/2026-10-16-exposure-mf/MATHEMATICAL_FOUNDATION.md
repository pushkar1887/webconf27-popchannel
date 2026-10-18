# Exposure policy and measured compression damage: mathematical specification

Status: prospective specification for new model fits on an already explored dataset, 16 October 2026. The identities below follow from standard risk analysis; they are not claimed as new theorems.

## 1. Observed target and population

Let \(P\in\{L,R\}\) denote ordinary logged exposure and randomized exposure. The observed target \(Y\in\{0,1\}\) is native long_view, whose agreement with the documentation's duration rule remains unresolved. Fix interface \(S=1\), the existing evaluation period, a common training-seen item support, and a common user roster \(\mathcal U^\star\).

Let \(X\) denote item and event state, including time/history when relevant. The model uses only user/item IDs. For a declared user distribution \(w(u)\), define

\[
 R_P(f)=\sum_{u\in\mathcal U^\star}w(u)
       \mathbb E_P[\ell(Y,f(u,I))\mid U=u,S=1].
\]

The primary \(w(u)=1/|\mathcal U^\star|\) is the same in both policies. This standardizes user frequency; it does not standardize the item distribution or eliminate unobserved event-state differences. The estimand is observed response under these collection policies, not intrinsic preference, welfare, or catalogue-wide relevance.

For logit \(z\), use the proper logistic loss

\[
 \ell(y,z)=\log(1+\exp z)-yz.
\]

All labeled exposures, both positive and negative, enter training. Unexposed pairs are not declared negative.

## 2. Compression penalties and the primary contrast

For full model \(f_0\) and compressed model \(f_m\), both fitted before evaluation:

\[
 \Delta_P(m)=R_P(f_m)-R_P(f_0),\qquad
 G_m=\Delta_R(m)-\Delta_L(m).
\]

A positive penalty means compression worsens loss. Negative \(G_m\) means the observed penalty is smaller under random exposure for this estimand. There is no general requirement that \(G_m<0\).

Use \(G_m\), absolute penalties, and uncertainty, rather than the unstable ratio \(\Delta_R/\Delta_L\). The ratio changes interpretation when the denominator is zero or negative.

The empirical estimator on the common evaluation roster is

\[
 \widehat R_P(f)=\frac1{n_E}\sum_{u=1}^{n_E}
       \frac1{N_{Pu}}\sum_{j=1}^{N_{Pu}}\ell(Y_{Puj},f(u,I_{Puj})).
\]

It averages interactions within each user, then users. Under independent sampling of users from the declared target and appropriate within-user sampling, it estimates that standardized risk. User bootstrapping conditions on the observed period and trained models; it does not establish temporal or training-sample generalization.

## 3. Exact decomposition and what it does not identify

Write \(q_P(x\mid u)\) for exposure/event-state distribution and

\[
 d_{m,P}(u,x)=
 \mathbb E_P[\ell(Y,f_m)-\ell(Y,f_0)\mid U=u,X=x,S=1].
\]

When the relevant conditional risks are defined on common support, addition and subtraction gives

\[
 G_m=\sum_u w(u)\sum_x
  [q_R(x\mid u)-q_L(x\mid u)]d_{m,L}(u,x)
 +\sum_u w(u)\sum_x q_R(x\mid u)
  [d_{m,R}(u,x)-d_{m,L}(u,x)].
\]

The first term is a change in exposure weighting. The second includes changes in conditional outcome law or omitted event state. Restricting interface and user roster alone does not identify the first term separately.

If conditional response is invariant and random exposure has support within logged exposure, define \(\omega(u,x)=q_R(x\mid u)/q_L(x\mid u)\). Then

\[
 G_m=\mathbb E_{w,q_L}[(\omega-1)d_m]
    =\operatorname{Cov}_{w,q_L}(\omega,d_m).
\]

The covariance identity uses \(\mathbb E\omega=1\). A directional claim requires a sign assumption on this covariance; randomization alone does not supply that sign. We do not know the required logging propensities and do not estimate this covariance by substituting inverse popularity. No IPS training is performed in this pilot.

## 4. Why compression can improve risk without proving policy imitation

For a stable conditional probability \(\eta(u,x)\), conditional logistic risk satisfies

\[
 r(z;\eta)=H(\eta)
 +D_{\mathrm{KL}}(\operatorname{Bern}(\eta)\|
                 \operatorname{Bern}(\sigma(z))).
\]

Hence

\[
 d_m=
 D_{\mathrm{KL}}(\eta\|p_m)-D_{\mathrm{KL}}(\eta\|p_0).
\]

Only when \(p_0=\eta\), a Bayes-oracle reference, is the conditional penalty necessarily nonnegative. A fitted full model is not that oracle. A compressed model can outperform it because of regularization, estimation error or optimization. Therefore a negative measured penalty does not prove that the full model learned the old recommendation policy.

## 5. A genuine interaction model and capacity control

The previous count model was \(z_{ui}=a_i+b_u\) within a fixed interface. For any two items, \(z_{ui}-z_{uj}=a_i-a_j\), independent of \(b_u\). It cannot identify a user-dependent ranking change.

This pilot instead uses

\[
 z_m(u,i)=b_0+b_u+c_i+
            \langle p_{g_m(u)},q_i\rangle,\quad d=16.
\]

Full: \(g_0(u)=u\). Random: seeded SplitMix64 bucket mapping. Frequency: the most active train users receive half of the buckets individually and the remainder are hashed into the remaining buckets. User and item bias tables remain uncompressed in all models, so the manipulated component is the user interaction-factor table. Item-factor table sizes are identical, but parameters are fitted independently for each model.

The dense capacity control uses \(p_u=a_uR\), with \(a_u\in\mathbb R^2\) and \(R\in\mathbb R^{2\times16}\), retaining individual user IDs. It has \(2|\mathcal U|+32\) interaction parameters. Choose

\[
 B=\left\lceil\frac{2|\mathcal U|+32}{16}\right\rceil.
\]

The shared interaction table has \(16B\) parameters, differing from the rank-2 control by fewer than 16 scalar parameters. Do not fake exact matching with unused padding. All actual array, bias and mapping bytes are reported, including the small remaining discrepancy.

The common objective is logged per-event BCE plus effective-factor norm regularization:

\[
 \widehat R_{\mathrm{train,event}}+
 \lambda_f\frac1N\sum_j
  \frac{\|p_{u_j}\|^2+\|q_{i_j}\|^2}{2d}
 +\lambda_b\frac1N\sum_j\frac{b_{u_j}^2+c_{i_j}^2}{2}.
\]

For the rank-2 model, regularize the effective \(a_uR\), not arbitrary factor rescalings. Initialization, optimizer, maximum epochs, logged-validation early stopping and tuning allowance are fixed equally before fitting. A single configuration is a diagnostic, not a claim of optimized published baselines.

## 6. Calibration is a separately defined sensitivity estimand

Assign approximately 10% of the common user roster to calibration by a label-independent hash. Calibration and evaluation users are disjoint in the later evaluation period; both can have earlier training data.

For each model and policy, fit only an intercept:

\[
 \widehat b_{mP}=\arg\min_b \widehat R_{P,C}(z_m+b).
\]

The derivative is the macro-user average of \(\sigma(z+b)-y\); the Hessian is the average of \(\sigma(z+b)(1-\sigma(z+b))>0\) in a nondegenerate sample. This is a one-dimensional convex optimization, solved by bisection.

The calibrated risk contrast uses \(z_m+\widehat b_{mP}\). These are policy-adapted predictors, so calibrated \(G_m\) is a distinct estimand from transport of one unmodified model. An intercept does not change within-user rankings. It controls one important calibration difference, not every possible calibration error or selection mechanism.

Under the additional pure label-shift assumption \(P_R(X\mid Y)=P_L(X\mid Y)\), Bayes logits differ by the class-prior log-odds shift. Random item exposure does not imply that assumption. Thus successful intercept calibration is not proof of a unique causal explanation.

## 7. Calibration uncertainty and paired user inference

For calibration user \(v\), let

\[
 g_{mPv}=\operatorname{mean}_{j\in v}
       [\sigma(z_{mj}+\widehat b_{mP})-Y_j],\quad
 H_{mP}=\operatorname{mean}_{v,j\in v}
       [p_{mj}(1-p_{mj})].
\]

The nested mean gives each user equal weight. For a multinomial user-bootstrap weight vector \(W_C\), an implicit-function expansion gives

\[
 \delta b^*_{mP}\simeq
 -\frac{\sum_v(W_{Cv}-1)g_{mPv}}{n_CH_{mP}}.
\]

Let \(D_{mP}\) be the evaluation macro-user average of \(p-y\). Bootstrap the evaluation risk exactly at fixed fitted intercepts and add the first-order correction

\[
 \delta G_C^*=
 D_{mR}\delta b^*_{mR}-D_{0R}\delta b^*_{0R}
 -D_{mL}\delta b^*_{mL}+D_{0L}\delta b^*_{0L}.
\]

Use the same calibration resample for both policies and all models, and the same evaluation-user resample for both policies and all models. Calibration and evaluation rosters are disjoint. This retains the correlations needed for the paired contrast. Check the approximation against exact recalibration on a fixed small subset of bootstrap draws. This is first-order, conditional-on-training inference, not exact finite-sample confidence coverage.

Report 1,000 bootstrap draws and both ordinary pointwise 95% intervals and Bonferroni-adjusted intervals for the five declared contrasts: random/frequency/rank2 \(G^{cal}\), and random-minus-rank2/frequency-minus-rank2 \(G^{cal}\). Average three training-seed effects without treating seeds as independent user samples.

## 8. Ranking sensitivity and scope

For users with both classes in both policies,

\[
 \operatorname{AUC}_{Pu}=
 \frac{\sum_{j:Y=1}\sum_{k:Y=0}
  [\mathbf1(z_j>z_k)+\tfrac12\mathbf1(z_j=z_k)]}
      {N^+_{Pu}N^-_{Pu}}.
\]

Report macro-user AUC, penalties with sign \(AUC(full)-AUC(m)\), and their policy gap. The outcome-defined eligible cohort is different from the all-user risk roster and its attrition must be shown. This measures ranking among exposed labeled examples, not catalogue-wide NDCG or an exposure-invariant utility. Global intercept calibration leaves this AUC unchanged.

## 9. Development decision

A method flags a material calibration-robust policy gap only if its average calibrated \(|G|\ge .001\), its family-adjusted interval excludes zero, every training seed has the same calibrated direction, and the raw mean has the same direction. The .001 threshold is an analyst-chosen development criterion.

Differences from the rank-2 control are reported separately. Absence of that distinction means sharing-specific interpretation is unestablished, even if capacity-dependent policy sensitivity exists. A flag permits a broader benchmark/controlled-data experiment, not a claim of intrinsic preference loss, published ranking reversal, or main-track readiness.

## Sources and provenance

- Selection-bias foundations: Schnabel et al., Recommendations as Treatments (ICML 2016), https://proceedings.mlr.press/v48/schnabel16.html
- Randomized collection setting: Gao et al., KuaiRand (CIKM 2022), https://arxiv.org/abs/2208.08696
- Related compression/bias and ranking precedents remain in EXPOSURE_COMPRESSION_PROPOSAL_REVIEW.md: UnKD and CLAX. The elementary identities here do not establish algorithmic novelty.
- Existing standard/random outcomes already informed development. This is an exploratory pilot with a recorded prospective fitting protocol, not untouched confirmation or a retroactive public preregistration.
