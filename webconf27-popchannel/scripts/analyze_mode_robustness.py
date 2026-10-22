"""Robustness of the corrected DirectAU vs corrected BPR comparison to the choice of correction mode.

Rule (symmetric, validation only): each loss uses whichever correction, cos+beta*p or raw+beta*p,
has the higher validation Recall@20 (beta already validation-selected per mode).
Also reports the worst-case cross-mode margin: min DAU correction minus max BPR correction (test).
Paired user bootstrap, 2,000 resamples, seed 0. Post hoc robustness check, not preregistered.
"""
import csv, json, os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'results', 'MODE_ROBUSTNESS.json')
LEGACY = {('amazonbook__mf__graphhash__bpr__s0', 'raw_prior'):
          'runs/_sessions/pc-x2-1017/controls/amazonbook__mf__graphhash__bpr__s0-legacy/per_user__full_prior_s0.npz'}
PAIRS = [(f'{ds}__mf__graphhash__directau__{s}', f'{ds}__mf__graphhash__bpr__{b}')
         for ds, b0 in (('gowalla', 's0-p500'), ('yelp2018', 's0'), ('amazonbook', 's0'))
         for s, b in (('s0', b0), ('s1', 's1'), ('s2', 's2'))]


def scores():
    out = {}
    with open(os.path.join(ROOT, 'results', 'master_metrics_v2.csv')) as f:
        for r in csv.DictReader(f):
            if r['scoring_mode'] in ('cos_prior', 'raw_prior') and r['val_recall20'] and r['test_recall20']:
                out.setdefault(r['run_id'], {})[r['scoring_mode']] = (float(r['val_recall20']), float(r['test_recall20']))
    return out


def scores_for(sc, run):
    """CSV row if present, else the run's own metrics.json (stored as fractions)."""
    if run in sc:
        return sc[run]
    with open(os.path.join(ROOT, 'runs', run, 'metrics.json')) as f:
        d = json.load(f)
    seeds = d['seeds']
    ev = (seeds[next(iter(seeds))] if isinstance(seeds, dict) else seeds[0])['eval']
    return {m: (100 * ev[m]['val']['recall'], 100 * ev[m]['test']['recall']) for m in ('cos_prior', 'raw_prior')}


def per_user(run, mode):
    path = LEGACY.get((run, mode), f'runs/{run}/per_user_test__{mode}.npz')
    z = np.load(os.path.join(ROOT, path))
    hits = z['hits'].astype(float)
    if hits.ndim > 1:
        hits = hits.sum(1)
    return z['users'], 100.0 * hits / np.minimum(20, z['npos'].astype(float))


def main(n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    sc, res = scores(), []
    for d, b in PAIRS:
        sd, sb = scores_for(sc, d), scores_for(sc, b)
        md = max(sd, key=lambda m: sd[m][0])
        mb = max(sb, key=lambda m: sb[m][0])
        ud, xd = per_user(d, md)
        ub, xb = per_user(b, mb)
        assert np.array_equal(ud, ub)
        diff = xd - xb
        bs = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(1)
        worst = min(v[1] for v in sd.values()) - max(v[1] for v in sb.values())
        res.append({'directau': d, 'bpr': b, 'directau_mode': md, 'bpr_mode': mb,
                    'registered_pairing': md == 'cos_prior' and mb == 'raw_prior',
                    'delta_pp': round(float(diff.mean()), 3),
                    'ci95': [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                    'worst_case_cross_mode_pp': round(worst, 3)})
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=2)
    return res


if __name__ == '__main__':
    for r in main():
        print(f"{r['directau'][:34]:34s} DAU:{r['directau_mode']:9s} BPR:{r['bpr_mode']:9s} reg={r['registered_pairing']!s:5s} "
              f"delta {r['delta_pp']:+.2f} {r['ci95']}  worst-case {r['worst_case_cross_mode_pp']:+.2f}")
