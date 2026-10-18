#!/usr/bin/env python
"""Retain final E8 paired inference from saved rankings, without model scoring.

Run from the project: python -B scripts/analyze_e8_final.py
Only the requested output JSON is written. No mappings are constructed and no
checkpoint is loaded for inference. Recall uses hits/min(20,npos); NDCG uses
the saved per-user values after checking them against original split truth.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import ttest_rel

from analyze_e5 import load_dataset, load_npz, sha256, validate_predictions
from validate_run import validate


METRICS = ('recall', 'ndcg')
PRIMARY_LABELS = ('Gowalla_s0', 'Yelp2018_s0', 'AmazonBook_s0')


def paired_vectors(a, b, k=20):
    """Align original IDs before computing capped recall and reading NDCG."""
    aligned = []
    for z in (a, b):
        users = np.asarray(z['users'])
        if users.ndim != 1 or not len(users):
            raise ValueError('users must be a nonempty vector')
        if len(np.unique(users)) != len(users):
            raise ValueError('duplicate users in paired input')
        order = np.argsort(users, kind='stable')
        npos = np.asarray(z['npos'])[order]
        if npos.shape != users.shape or np.any(npos <= 0):
            raise ValueError('invalid npos')
        hits = np.asarray(z['hits'])
        if hits.ndim == 2:
            hits = hits.sum(axis=1)
        if hits.ndim != 1 or hits.shape != users.shape:
            raise ValueError('invalid hits shape')
        hits = hits[order]
        if np.any(hits < 0) or np.any(hits > np.minimum(k, npos)):
            raise ValueError('hits exceed capped truth denominator')
        recall = hits / np.minimum(k, npos)
        if 'recall' in z and not np.allclose(np.asarray(z['recall'])[order], recall,
                                            atol=1e-10, rtol=0):
            raise ValueError('stored recall disagrees with hits/npos')
        ndcg = np.asarray(z['ndcg'], dtype=np.float64)[order]
        if ndcg.shape != users.shape or not np.all(np.isfinite(ndcg)):
            raise ValueError('invalid ndcg')
        if np.any(ndcg < 0) or np.any(ndcg > 1):
            raise ValueError('ndcg outside [0,1]')
        aligned.append((users[order], npos, np.stack([recall, ndcg], axis=1)))
    if not np.array_equal(aligned[0][0], aligned[1][0]):
        raise ValueError('paired user coverage differs')
    if not np.array_equal(aligned[0][1], aligned[1][1]):
        raise ValueError('paired npos differs')
    return {'users': aligned[0][0], 'npos': aligned[0][1],
            'a': aligned[0][2], 'b': aligned[1][2]}


def bootstrap_difference(differences, n_resamples=2000, seed=0):
    """Same sampled user indices for both metrics; reset rng for each pair."""
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 2 or not len(differences) or n_resamples < 1:
        raise ValueError('need nonempty user-by-metric differences and resamples')
    if not np.all(np.isfinite(differences)):
        raise ValueError('nonfinite paired differences')
    rng = np.random.default_rng(seed)
    n = len(differences)
    means = np.empty((n_resamples, differences.shape[1]), dtype=np.float64)
    for first in range(0, n_resamples, 20):
        count = min(20, n_resamples - first)
        indexes = rng.integers(0, n, size=(count, n))
        means[first:first + count] = differences[indexes].mean(axis=1)
    return {'ci95': np.percentile(means, [2.5, 97.5], axis=0),
            'nonpositive_counts': np.count_nonzero(means <= 0, axis=0)}


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.all(np.isfinite(p)):
        raise ValueError('Holm requires finite p values')
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError('p outside [0,1]')
    order = np.argsort(p, kind='stable')
    result = np.empty_like(p)
    result[order] = np.minimum(1., np.maximum.accumulate(
        p[order] * np.arange(len(p), 0, -1)))
    return result


def paired_summary(a, b, n_resamples=2000, seed=0):
    vectors = paired_vectors(a, b)
    d = vectors['a'] - vectors['b']
    boot = bootstrap_difference(d, n_resamples=n_resamples, seed=seed)
    metrics = {}
    for j, name in enumerate(METRICS):
        if np.all(d[:, j] == 0):
            p, t = 1., 0.
        elif np.std(d[:, j]) == 0:
            p, t = 0., None
        else:
            result = ttest_rel(vectors['a'][:, j], vectors['b'][:, j], alternative='two-sided')
            p, t = float(result.pvalue), float(result.statistic)
        metrics[name] = {
            'a_mean_percent': float(100 * vectors['a'][:, j].mean()),
            'b_mean_percent': float(100 * vectors['b'][:, j].mean()),
            'delta_pp': float(100 * d[:, j].mean()),
            'ci95_pp': (100 * boot['ci95'][:, j]).tolist(),
            'paired_t_p_two_sided': p, 'paired_t_statistic': t,
            'holm_adjusted_p': None,
            'bootstrap_nonpositive_means': int(boot['nonpositive_counts'][j]),
        }
    return {'n_users': int(len(vectors['users'])), 'metrics': metrics}


def seed_spread(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError('sample SD needs at least two seeds')
    return {'n_seeds': int(len(values)), 'mean_percent': float(values.mean()),
            'sample_sd_pp': float(values.std(ddof=1)), 'values_percent': values.tolist()}


def amazon_aggregate_sensitivity(control, directau_means):
    """Aggregate alternatives are descriptive: no missing paired CI invented."""
    main = control['full_prior']
    alternate = control['legacy_topk']['raw']['prior']
    applicable = main['base'] == 'raw' and float(alternate['best_beta']) == float(main['best_beta'])
    legacy = {'applicable_same_base_beta': applicable,
              'same_checkpoint_evidence': 'same controls.json source checkpoint',
              'base': 'raw', 'beta': float(alternate['best_beta']),
              'policy': 'legacy torch.topk scoring path', 'ci95_pp': None,
              'paired_ci_status': 'missing_per_user_predictions',
              'metrics': {name: {'directau_percent': float(100 * directau_means[name]),
                                  'bpr_percent': float(100 * alternate['test'][name]),
                                  'delta_pp': float(100 * (directau_means[name] - alternate['test'][name]))}
                          for name in METRICS}}
    random_seeds = {}
    for name in METRICS:
        values = np.asarray(main['test'][name]['per_seed'], dtype=np.float64)
        deltas = 100 * (directau_means[name] - values)
        random_seeds[name] = {'bpr_values_percent': (100 * values).tolist(),
                              'deltas_pp': deltas.tolist(),
                              'delta_min_pp': float(deltas.min()),
                              'delta_max_pp': float(deltas.max()),
                              'all_deltas_positive': bool(np.all(deltas > 0)),
                              'per_user_predictions_available_seeds': [0],
                              'other_seed_paired_ci_status': 'missing_per_user_predictions'}
    return {'alternate_paired_ci_status': 'missing_per_user_predictions',
            'legacy_topk_raw_prior': legacy, 'random_key_seed_means': random_seeds,
            'excluded_substitutes': {
                'per_user__base_popkey.npz': 'no prior (beta 0), not alternate full-prior ties',
                'per_user__base_randomkey_s0.npz': 'no prior (beta 0)',
                'per_user__cell_mu_s0.npz': 'bucket-only prior; different intervention',
                'per_user__cell_eps_s0.npz': 'residual-only prior; different intervention'},
            'conclusion': 'Primary random-key paired CI excludes zero; aggregate alternatives remain positive. '
                          'Alternate full-prior paired-CI robustness is not established.'}


def input_receipt(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': sha256(path), 'bytes': path.stat().st_size}


def build_analysis(root, data_root):
    root, data_root = Path(root).resolve(), Path(data_root).resolve()
    runs = root / 'runs'
    control_dir = runs / '_sessions/pc-x2-1017/controls/amazonbook__mf__graphhash__bpr__s0-legacy'
    control_path = control_dir / 'controls.json'
    control = json.loads(control_path.read_text(encoding='utf-8'))
    control_check = validate(control_dir, controls=True)
    if not control_check['ok']:
        raise ValueError(f'Invalid Amazon control: {control_check["problems"]}')
    specs = [
        ('Gowalla_s0', 'Gowalla', 'gowalla__mf__graphhash__directau__s0',
         'gowalla__mf__graphhash__bpr__s0-p500'),
        ('Gowalla_s1', 'Gowalla', 'gowalla__mf__graphhash__directau__s1',
         'gowalla__mf__graphhash__bpr__s1'),
        ('Gowalla_s2', 'Gowalla', 'gowalla__mf__graphhash__directau__s2',
         'gowalla__mf__graphhash__bpr__s2'),
        ('Yelp2018_s0', 'Yelp2018', 'yelp2018__mf__graphhash__directau__s0',
         'yelp2018__mf__graphhash__bpr__s0'),
        ('AmazonBook_s0', 'AmazonBook', 'amazonbook__mf__graphhash__directau__s0',
         'amazonbook__mf__graphhash__bpr__s0-legacy'),
    ]
    rows, split_receipts = [], {}
    current_ds, data = None, None
    for label, dataset, a_id, b_id in specs:
        if dataset != current_ds:
            data = load_dataset(data_root, dataset)
            split_receipts[dataset] = {'split_sizes': data['split_sizes'], 'files': data['files']}
            current_ds = dataset
        pa = runs / a_id / 'per_user_test__cos_prior.npz'
        pb = (control_dir / 'per_user__full_prior_s0.npz' if dataset == 'AmazonBook'
              else runs / b_id / 'per_user_test__raw_prior.npz')
        a, b = load_npz(pa), load_npz(pb)
        checks = [validate_predictions(z, data['truths'], data['counts'], seen=data['seen'])
                  for z in (a, b)]
        modern_dirs = [runs / a_id] + ([] if dataset == 'AmazonBook' else [runs / b_id])
        validation = [validate(p) for p in modern_dirs]
        if any(not v['ok'] for v in validation):
            raise ValueError(f'Invalid run in pair {label}')
        row = {'label': label, 'dataset': dataset, 'a_run_id': a_id, 'b_run_id': b_id,
               'a_mode': 'cos_prior', 'b_mode': 'raw_prior',
               'comparison_type': 'reproduction',
               'a_lineage': 'modern', 'b_lineage': 'legacy' if dataset == 'AmazonBook' else 'modern',
               'inputs': [input_receipt(pa), input_receipt(pb)],
               'truth_validation': checks, 'run_validation': validation,
               **paired_summary(a, b)}
        row['selection'] = {}
        for side, run_dir, mode in [('a', runs / a_id, 'cos_prior'), ('b', runs / b_id, 'raw_prior')]:
            if side == 'b' and dataset == 'AmazonBook':
                row['selection'][side] = {'beta': float(control['full_prior']['best_beta']),
                                          'scoring_policy': 'random-key tie seed 0',
                                          'selection_source': str(control_path)}
            else:
                metrics = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
                info = next(iter(metrics['seeds'].values()))
                row['selection'][side] = {'beta': float(info['eval'][mode]['best_beta']),
                                          'scoring_policy': 'saved GPU torch.topk',
                                          'stop_reason': info['stop_reason'],
                                          'best_epoch': info['best_epoch'], 'epochs_run': info['epochs_run'],
                                          'selection_source': str(run_dir / 'metrics.json')}
                row['inputs'].append(input_receipt(run_dir / 'metrics.json'))
        rows.append(row)
        print(f'{label}: {row["n_users"]} users, both saved rankings validated', flush=True)
    primary_rows = [next(row for row in rows if row['label'] == label) for label in PRIMARY_LABELS]
    families = {}
    for name in METRICS:
        adjusted = holm_adjust([row['metrics'][name]['paired_t_p_two_sided'] for row in primary_rows])
        for row, value in zip(primary_rows, adjusted):
            row['metrics'][name]['holm_adjusted_p'] = float(value)
        families[name] = {'role': 'primary' if name == 'recall' else 'secondary',
                          'members': list(PRIMARY_LABELS), 'n_tests': 3,
                          'p_source': 'two-sided paired t-test', 'adjustment': 'Holm step-down',
                          'all_positive_adjusted_p_lt_005': all(
                              row['metrics'][name]['delta_pp'] > 0 and
                              row['metrics'][name]['holm_adjusted_p'] < .05 for row in primary_rows)}
    gowalla = [row for row in rows if row['dataset'] == 'Gowalla']
    spreads = {name: {side: seed_spread([row['metrics'][name][f'{side}_mean_percent'] for row in gowalla])
                      for side in ('a', 'b')} for name in METRICS}
    amazon = next(row for row in rows if row['dataset'] == 'AmazonBook')
    sensitivity = amazon_aggregate_sensitivity(control, {
        name: amazon['metrics'][name]['a_mean_percent'] / 100 for name in METRICS})
    sensitivity['controls_json'] = input_receipt(control_path)
    sensitivity['available_per_user_files'] = sorted(p.name for p in control_dir.glob('*.npz'))
    sensitivity['source_checkpoint_name'] = control['meta'].get('ckpt_name')
    sensitivity['required_gpu_rescoring_spec'] = {
        'dataset': 'AmazonBook', 'method': 'GraphHash',
        'bpr_run_id': 'amazonbook__mf__graphhash__bpr__s0-legacy',
        'source_checkpoint': 'original BPR-long source used by controls.json; no retraining',
        'source_checkpoint_name': control['meta'].get('ckpt_name'),
        'source_metadata': str(control_path), 'mapping_source': 'same stored GPU checkpoint aliases',
        'base': 'raw', 'prior': 'p_i=log(1+training_count_i)', 'beta': .2,
        'beta_selection': 'frozen existing validation-selected beta; do not retune',
        'k': 20, 'split': 'test', 'n_users': 50407,
        'exclude': 'train plus validation seen items, identical original item IDs',
        'arithmetic': 'same float32 GPU score tensor for each tie-key variant',
        'tie_policies': ['random-key seeds 0..9', 'index-key', 'popularity-key', 'legacy torch.topk'],
        'ties': 'secondary order only among exactly equal scores, no score jitter',
        'outputs_per_policy': ['users', 'top20', 'hits', 'npos', 'recall', 'ndcg'],
        'save': 'separate NPZ per full-prior policy plus checkpoint/mapping/beta receipts',
        'directau_partner': amazon['inputs'][0]['path'],
        'inference': 'paired 2000-user-resample percentile CI seed 0 against fixed B9 saved predictions',
        'execution_status': 'not run; job specification only; orchestrator submission required'}
    human = [{'label': row['label'], 'a_run_id': row['a_run_id'], 'b_run_id': row['b_run_id'],
              'n_users': row['n_users'], 'recall': row['metrics']['recall'], 'ndcg': row['metrics']['ndcg']}
             for row in rows]
    return {
        'schema': 'E8-final-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_script': input_receipt(Path(__file__)),
        'runtime': {'numpy': np.__version__, 'scipy': scipy.__version__},
        'method': {'recall': 'macro mean hits/min(20,npos)',
                   'ndcg': 'saved per-user NDCG checked against exact original test truth',
                   'bootstrap': 'paired user percentile 95% CI; conditional on trained checkpoints',
                   'n_resamples': 2000, 'rng': 'numpy.random.default_rng', 'seed': 0,
                   'rng_reset': 'each pair; same sampled user indices for Recall and NDCG',
                   'p_values': 'two-sided paired t-test; Holm across three datasets separately per metric',
                   'not_bootstrap_p': 'Zero nonpositive means in 2000 draws does not establish tiny empirical p values',
                   'training_seed_uncertainty': 'reported separately as three-seed sample SD, ddof=1',
                   'limitations': ['paired-user uncertainty is conditional on checkpoints',
                                   'shared training graph means users are not independent training replications',
                                   'AmazonBook BPR uses legacy random-key predictions; alternate full-prior CIs unavailable']},
        'split_lineage': split_receipts, 'pairs': rows, 'human_table': human,
        'holm_families': families, 'gowalla_three_seed_spread': spreads,
        'gowalla_all_seeds_positive_ci': {name: all(row['metrics'][name]['ci95_pp'][0] > 0 for row in gowalla)
                                         for name in METRICS},
        'primary_verdict': 'PASS' if families['recall']['all_positive_adjusted_p_lt_005'] and
                           all(row['metrics']['recall']['ci95_pp'][0] > 0 for row in gowalla) else 'FAIL',
        'amazon_tie_robustness': sensitivity,
        'mechanism_pointers': [{'path': str(root / 'RESULTS.md'),
                                'blocks': ['PR-H', 'PR-DEC', 'PR-TIE'],
                                'status': 'prior mechanism evidence; no new mechanism computation in E8'}],
        'side_effects': 'Read saved NPZs, metrics and original split tensors; wrote this JSON only. No model inference or mapping construction.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path, default=Path(
        r'D:\web_2\experiments\2026-10-16-loss-sharing-repro\_work\datasets'))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    result = build_analysis(args.project_root, args.data_root)
    output = args.out or args.project_root / 'results/E8_final.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Wrote {output}; primary {result["primary_verdict"]}; alternate Amazon paired CI unavailable')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
