"""Independent replay and arithmetic checks; does not import the fit/eval code."""
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / '2026-10-15-kuairand-support'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.perf_counter()
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    manifest = json.loads((ROOT / 'fit_manifest.json').read_text())
    result = json.loads((ROOT / 'results.json').read_text())
    receipt = json.loads((ROOT / 'evaluation_started.json').read_text())
    assert digest(SOURCE / 'audit_arrays.npz') == protocol['source_cache_sha256']
    assert digest(ROOT / 'protocol.json') == manifest['protocol_sha256'] == result['protocol_sha256']
    assert digest(ROOT / 'fit_manifest.json') == result['fit_manifest_sha256'] == receipt['fit_manifest_sha256']
    for file, value in manifest['sources'].items():
        assert digest(ROOT / file) == value
    assert manifest['frozen_utc'] <= receipt['utc'] <= result['completed_utc']
    cache = np.load(SOURCE / 'audit_arrays.npz')
    uid, vid, ctx, target = [cache['test_standard__' + x] for x in ['user_id', 'video_id', 'tab', 'long_view']]
    tu, tc, ty = [cache['train__' + x] for x in ['user_id', 'tab', 'long_view']]
    profiles = np.load(ROOT / 'training_profiles.npz')
    common = np.load(ROOT / 'common.npz')
    saved_predictions = np.load(ROOT / 'predictions_standard.npz')
    aggregates = np.load(ROOT / 'per_user_context_loss_sums.npz')
    boot = np.load(ROOT / 'bootstrap_context_deltas.npz')
    counts = aggregates['counts']
    event_keys = uid.astype('int64') * 15 + ctx
    independent_counts = np.bincount(event_keys, minlength=counts.size).reshape(counts.shape)
    np.testing.assert_array_equal(counts, independent_counts)
    train_n = np.bincount(tu.astype('int64') * 15 + tc, minlength=profiles['train_counts'].size).reshape(profiles['train_counts'].shape)
    np.testing.assert_array_equal(train_n, profiles['train_counts'])
    np.testing.assert_array_equal(aggregates['supported'][:len(train_n)], (train_n >= 10).sum(axis=1) >= 2)
    errors = {'prediction': 0., 'group_probability': 0., 'aggregate': 0., 'sweep': 0., 'raw': 0., 'bootstrap': 0.}
    byte_sets = {}
    independent = {}
    zlog = lambda x: np.log(np.clip(x.astype('float64'), 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(x.astype('float64'), 1e-6, 1 - 1e-6))
    for row in manifest['models']:
        name = row['name']
        file = ROOT / 'models' / (name + '.npz')
        assert digest(file) == row['model_sha256']
        model = np.load(file)
        prior = common['context'][ctx]
        item = prior.copy()
        ok = vid < len(common['item'])
        item[ok] = common['item'][vid[ok], ctx[ok]]
        residual = np.zeros(len(uid))
        if row['kind'] == 'global':
            pred = np.full(len(uid), common['global_'][0])
        elif row['kind'] == 'item_only':
            pred = item
        else:
            if row['kind'] == 'ungrouped':
                ok = uid < len(model['user'])
                residual[ok] = zlog(model['user'][uid[ok], ctx[ok]]) - zlog(prior[ok])
            else:
                mapping, table = model['mapping'], model['group']
                assert mapping.dtype == np.int16 and table.dtype == np.float32
                assert table.shape == (row['budget'], 15)
                groups = mapping[profiles['active']]
                sizes = np.bincount(groups, minlength=row['budget'])
                assert sizes.min() >= 1 and sizes.max() <= row['group_cap']
                keys = mapping[tu].astype('int64') * 15 + tc
                n = np.bincount(keys, minlength=row['budget'] * 15).reshape(row['budget'], 15)
                s = np.bincount(keys, weights=ty, minlength=row['budget'] * 15).reshape(row['budget'], 15)
                probability = ((s + 10 * common['context']) / (n + 10)).astype('float32')
                errors['group_probability'] = max(errors['group_probability'], float(np.abs(probability - table).max()))
                np.testing.assert_array_equal(probability, table)
                ok = uid < len(mapping)
                group_ids = np.full(len(uid), -1)
                group_ids[ok] = mapping[uid[ok]]
                ok &= group_ids >= 0
                residual[ok] = zlog(table[group_ids[ok], ctx[ok]]) - zlog(prior[ok])
                actual_bytes = mapping.nbytes + table.nbytes
                assert actual_bytes == row['varying_array_bytes']
                byte_sets.setdefault(row['budget'], set()).add(row['total_array_bytes'])
                if row['history']:
                    values = [v['objective'] for v in row['history']]
                    assert np.all(np.diff(values) < 0)
                    # Direct per-user KL recomputation of the final clustering objective.
                    r = profiles['profile']
                    center = np.stack([np.bincount(groups, weights=r[:, c], minlength=row['budget']) / sizes for c in range(15)], axis=1)
                    div = r * np.log(r / center[groups]) + (1 - r) * np.log((1 - r) / (1 - center[groups]))
                    cgs = np.stack([np.bincount(groups, weights=div[:, c], minlength=row['budget']) / sizes for c in range(15)], axis=1)
                    if row['lambda'] is None:
                        objective = (sizes * (cgs @ profiles['p'])).sum() / sizes.sum()
                    else:
                        temperature = row['lambda']
                        objective = (sizes * temperature * np.log(np.exp(cgs / temperature) @ profiles['p'])).sum() / sizes.sum()
                    assert abs(objective - values[-1]) < 1e-10
            pred = (1 / (1 + np.exp(-np.clip(zlog(item) + residual, -25, 25)))).astype('float32')
        errors['prediction'] = max(errors['prediction'], float(np.abs(pred - saved_predictions[name]).max()))
        np.testing.assert_allclose(pred, saved_predictions[name], atol=1e-7, rtol=0)
        pp = np.clip(pred.astype('float64'), 1e-6, 1 - 1e-6)
        ll = -np.where(target == 1, np.log(pp), np.log1p(-pp))
        sm = np.bincount(event_keys, weights=ll, minlength=counts.size).reshape(counts.shape)
        independent[name] = sm
        errors['aggregate'] = max(errors['aggregate'], float(np.abs(sm - aggregates[name]).max()))
        np.testing.assert_allclose(sm, aggregates[name], atol=1e-10, rtol=0)
    assert all(len(s) == 1 for s in byte_sets.values())
    previous = np.load(SOURCE / 'predictions_test_standard.npz')['item_plus_user_context']
    np.testing.assert_array_equal(previous, saved_predictions['ungrouped'])
    masks = {'all': counts.sum(axis=1) > 0, 'supported': aggregates['supported'] & (counts.sum(axis=1) > 0)}
    contexts, bootstrap_replays = {}, 0
    for cohort, mask in masks.items():
        denom = counts[mask].sum(axis=0)
        for name, sm in independent.items():
            contexts[cohort, name] = np.divide(sm[mask].sum(axis=0), denom, out=np.full(15, np.nan), where=denom > 0)
        rng = np.random.default_rng(20261015)
        ids = np.flatnonzero(mask)
        for replicate in range(5):
            sampled = ids[rng.integers(0, len(ids), size=len(ids))]
            d = counts[sampled].sum(axis=0)
            for name in independent:
                num = (independent[name][sampled] - independent['ungrouped'][sampled]).sum(axis=0)
                expected = np.divide(num, d, out=np.full(15, np.nan), where=d > 0)
                actual = boot[cohort + '__' + name][replicate]
                np.testing.assert_allclose(expected, actual, atol=1e-11, rtol=0, equal_nan=True)
                if np.isfinite(expected).any():
                    errors['bootstrap'] = max(errors['bootstrap'], float(np.nanmax(np.abs(expected - actual))))
                bootstrap_replays += 1
    raw = list(csv.DictReader((ROOT / 'raw_metrics.csv').open()))
    for row in raw:
        mask = masks[row['cohort']]
        expected = independent[row['model']][mask].sum() / counts[mask].sum()
        errors['raw'] = max(errors['raw'], abs(expected - float(row['log_loss'])))
        assert abs(expected - float(row['log_loss'])) < 1e-12
    curves = list(csv.DictReader((ROOT / 'sweep_metrics.csv').open()))
    for row in curves:
        cohort, name = row['cohort'], row['model']
        q = np.array(result['sweeps'][row['sweep']][[0., .25, .5, .75, 1.].index(float(row['gamma']))])
        mean, ref = contexts[cohort, name], contexts[cohort, 'ungrouped']
        active = q > 0
        expected = float(mean[active] @ q[active]) if np.isfinite(mean[active]).all() else None
        assert (expected is None) == (row['loss'] == '')
        if expected is not None:
            errors['sweep'] = max(errors['sweep'], abs(expected - float(row['loss'])))
            assert abs(expected - float(row['loss'])) < 1e-12
            assert abs(float((mean - ref)[active] @ q[active]) - float(row['sharing_penalty'])) < 1e-12
        b = boot[cohort + '__' + name][:, active] @ q[active]
        valid = b[np.isfinite(b)]
        assert len(valid) == int(row['valid_bootstrap_replicates'])
        if len(valid):
            interval = np.quantile(valid, [.025, .975])
            np.testing.assert_allclose(interval, [float(row['delta_ci_low']), float(row['delta_ci_high'])], atol=1e-12, rtol=0)
    # Primary gate: independently reconstruct both signed endpoint contrasts.
    q0, q1 = [np.array(x) for x in [result['sweeps']['all_flatten'][0], result['sweeps']['all_flatten'][-1]]]
    f, t = boot['all__frequency_B256'], boot['all__tilted_B256_L0.3']
    fg, tg, mix = f @ (q1 - q0), t @ (q1 - q0), (t - f) @ q0
    lower_f = float(np.quantile(fg, .05 / 3))
    lower_a = float(np.quantile(fg - tg, .05 / 3))
    upper_m = float(np.quantile(mix, 1 - .05 / 3))
    np.testing.assert_allclose([lower_f, lower_a, upper_m], [result['primary_gate']['one_sided_adjusted_lower_frequency_growth'], result['primary_gate']['one_sided_adjusted_lower_growth_advantage'], result['primary_gate']['one_sided_adjusted_upper_in_mix_difference']], atol=1e-12, rtol=0)
    assert result['primary_gate']['conditions_pass'] == [lower_f > 0, lower_a > 0, upper_m <= .001]
    assert result['primary_gate']['gate_pass'] == (lower_f > 0 and lower_a > 0 and upper_m <= .001)
    report = {'status': 'independent_artifact_checks_passed', 'models_replayed': len(independent),
              'raw_metric_rows_verified': len(raw), 'sweep_rows_and_intervals_verified': len(curves),
              'bootstrap_context_vectors_replayed': bootstrap_replays,
              'equal_bytes_at_each_budget': {str(k): list(v)[0] for k, v in byte_sets.items()},
              'ungrouped_prediction_matches_previous_probe_exactly': True,
              'max_absolute_errors': errors, 'primary_gate_recomputed': result['primary_gate']['gate_pass'],
              'scientific_limits': ['Native label rule mismatch remains unresolved.',
                  'Supported-cohort all-context bootstrap retains only 545/1000 resamples because rare contexts disappear; these conditional intervals are not dependable coverage estimates.',
                  'One optimization seed and capped, limited-iteration heuristic; no global-optimum guarantee.',
                  'Existing test period is exploratory; no neural or deployment robustness validation.'],
              'elapsed_seconds': time.perf_counter() - started}
    (ROOT / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
