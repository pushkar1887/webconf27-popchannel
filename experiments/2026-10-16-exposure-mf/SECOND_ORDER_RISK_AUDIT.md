# Supplementary finite-sample mathematical audit

Status: post-evaluation descriptive audit, not a new primary hypothesis or confidence interval.

For logistic loss, \(\ell'(y,z)=\sigma(z)-y\) and
\(0\leq\ell''(y,z)=\sigma(z)(1-\sigma(z))\leq1/4\).
For any two finite logits \(z_0,z_m\), let \(\delta=z_m-z_0\). Convexity and the bounded Hessian give the deterministic inequality

\[
 0\leq
 \ell(y,z_m)-\ell(y,z_0)-(\sigma(z_0)-y)\delta
 \leq\delta^2/8.
\]

The first term is the score perturbation's alignment with the full model's error; it can be negative. The nonnegative remainder is curvature. This statement does not require knowing a Bayes preference function, exposure propensities, or independent observations.

Applying exactly the same positive macro-user averaging weights as the main estimand gives

\[
 \Delta_P=A_P+\mathcal R_P,\quad
 A_P=\widehat{\mathbb E}_P[(\sigma(z_0)-Y)\delta],
 \quad0\leq\mathcal R_P\leq B_P
 =\widehat{\mathbb E}_P[\delta^2]/8.
\]

Therefore

\[
 A_R-A_L-B_L\ \leq G\leq A_R-A_L+B_R.
\]

This is a conditional, finite-sample bound on the already observed paired loss difference, not a population confidence interval. It neither establishes a new theorem nor identifies why the two models' errors differ. The same statement applies to separately calibrated logits, but those remain policy-adapted predictors.

The companion risk_certificate_audit.py computes and checks the inequality for every nonfull method, training seed, evaluation policy, and raw/calibrated scoring variant. It changes no fits, primary metrics, confidence intervals or decision flags.
