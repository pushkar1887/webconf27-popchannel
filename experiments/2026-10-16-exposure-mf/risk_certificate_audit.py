"""Descriptive second-order risk certificate; no fits or primary-hypothesis changes."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / '2026-10-15-kuairand-support'


def main():
    d = np.load(ROOT / 'data_contract.npz')
    z = np.load(SOURCE / 'audit_arrays.npz')
    metrics = {(r['name'], r['policy']): r for r in csv.DictReader((ROOT / 'model_metrics.csv').open())}
    nusers = len(d['evaluation_users'])
    rows = []
    for policy in ['standard', 'random']:
        score = np.load(ROOT / f'scores_{policy}_evaluation.npz')
        uid = d[f'evaluation_{policy}_roster_u']
        y = z[f'test_{policy}__long_view'][d[f'evaluation_{policy}_indices']]
        count = np.bincount(uid, minlength=nusers)
        weights = 1 / (nusers * count[uid])
        for method in ['random', 'frequency', 'rank2']:
            for seed in [11, 29, 47]:
                name, ref = f'{method}_s{seed}', f'full_s{seed}'
                for version in ['raw', 'cal']:
                    zm, z0 = score[name].copy(), score[ref].copy()
                    if version == 'cal':
                        zm += float(metrics[name, policy]['calibration_intercept'])
                        z0 += float(metrics[ref, policy]['calibration_intercept'])
                    delta = zm - z0
                    paired = np.logaddexp(0, zm) - y * zm - (np.logaddexp(0, z0) - y * z0)
                    first = (1 / (1 + np.exp(-z0)) - y) * delta
                    remainder = paired - first
                    bound = delta ** 2 / 8
                    assert np.min(remainder) >= -2e-12
                    assert np.max(remainder - bound) <= 2e-12
                    rows.append({'method': method, 'seed': seed, 'policy': policy, 'version': version,
                                 'loss_penalty': float(np.sum(weights * paired)),
                                 'first_order_alignment': float(np.sum(weights * first)),
                                 'curvature_remainder': float(np.sum(weights * remainder)),
                                 'curvature_upper_bound': float(np.sum(weights * bound)),
                                 'mean_absolute_score_change': float(np.sum(weights * np.abs(delta))),
                                 'minimum_pointwise_remainder': float(remainder.min()),
                                 'maximum_pointwise_upper_bound_violation': float((remainder - bound).max())})
    with (ROOT / 'risk_certificates.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    output = {'status': 'all_pointwise_logistic_curvature_bounds_verified', 'comparisons': len(rows),
              'minimum_remainder_across_all_comparisons': min(r['minimum_pointwise_remainder'] for r in rows),
              'maximum_upper_bound_violation': max(r['maximum_pointwise_upper_bound_violation'] for r in rows),
              'scope': 'Post-evaluation descriptive finite-sample identity audit; not a confidence interval, new theorem, or primary test.'}
    (ROOT / 'risk_certificate_verification.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
