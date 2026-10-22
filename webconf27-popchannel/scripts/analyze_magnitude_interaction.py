"""Magnitude x sharing interaction for MF DirectAU (post hoc, not preregistered).

For each dataset, uses the stored per-user test arrays of the Full and GraphHash
DirectAU seed-0 runs under raw and cosine scoring. Recall_u = hits / min(20, npos).

  M_full   = R(Full, raw) - R(Full, cos)       value of magnitude on the full table
  M_shared = R(GH,   raw) - R(GH,   cos)       value of magnitude under sharing
  D        = R(Full, cos) - R(GH,   cos)       drop under direction-only scoring
  I        = M_full - M_shared                 interaction (difference in differences)
  Identity: R(Full, raw) - R(GH, raw) = D + I  (exact)

CIs: paired user bootstrap, 2,000 resamples, seed 0 (same users in all four cells).
"""
import json, os, sys
import numpy as np

RUNS = os.path.join(os.path.dirname(__file__), '..', 'runs')
OUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'MAGNITUDE_INTERACTION.json')


def recall(run, mode):
    z = np.load(os.path.join(RUNS, run, f'per_user_test__{mode}.npz'))
    hits = z['hits'].astype(float)
    if hits.ndim > 1:
        hits = hits.sum(1)
    return z['users'], 100.0 * hits / np.minimum(20, z['npos'].astype(float))


def main(n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for ds in ('gowalla', 'yelp2018', 'amazonbook'):
        cells = {}
        for comp in ('full', 'graphhash'):
            for mode in ('raw', 'cosine'):
                run = f'{ds}__mf__{comp}__directau__s0'
                cells[(comp, mode)] = recall(run, mode)
        users = cells[('full', 'raw')][0]
        assert all(np.array_equal(users, u) for u, _ in cells.values())
        fr, fc = cells[('full', 'raw')][1], cells[('full', 'cosine')][1]
        gr, gc = cells[('graphhash', 'raw')][1], cells[('graphhash', 'cosine')][1]
        terms = {'drop_raw': fr - gr, 'M_full': fr - fc, 'M_shared': gr - gc,
                 'D_direction': fc - gc, 'I_interaction': (fr - fc) - (gr - gc)}
        idx = rng.integers(0, len(users), (n_boot, len(users)))
        rec = {'n_users': int(len(users)),
               'cells': {f'{c}_{m}': round(float(v[1].mean()), 4) for (c, m), v in cells.items()}}
        for k, x in terms.items():
            bs = x[idx].mean(1)
            rec[k] = {'mean': round(float(x.mean()), 4),
                      'ci95': [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
        assert abs(terms['drop_raw'].mean() - terms['D_direction'].mean() - terms['I_interaction'].mean()) < 1e-9
        out[ds] = rec
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == '__main__':
    res = main()
    for ds, r in res.items():
        print(ds, {k: r[k]['mean'] for k in ('drop_raw', 'M_full', 'M_shared', 'D_direction', 'I_interaction')},
              'I CI', r['I_interaction']['ci95'])
