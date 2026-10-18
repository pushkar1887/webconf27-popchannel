"""Independent replay; does not import fitting or evaluation helpers."""
import csv
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / '2026-10-15-kuairand-support'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def means(values, uid, size):
    n = np.bincount(uid, minlength=size)
    return np.bincount(uid, weights=values, minlength=size) / n


def logistic_loss(y, score):
    return np.logaddexp(0., score) - y * score


def independent_score(a, u, i):
    z = np.empty(len(u))
    for start in range(0, len(u), 40000):
        stop = min(start + 40000, len(u))
        uu, ii = u[start:stop], i[start:stop]
        user = a['user_factors'][a['mapping'][uu]].astype('float64')
        if 'projection' in a:
            user = user @ a['projection'].astype('float64')
        z[start:stop] = float(a['global_bias'][0]) + a['user_bias'][uu].astype('float64') + a['item_bias'][ii].astype('float64') + np.sum(user * a['item_factors'][ii].astype('float64'), axis=1)
    return z


def scalar_hash(uid, seed):
    mask = (1 << 64) - 1
    v = (int(uid) + seed + 0x9E3779B97F4A7C15) & mask
    v = ((v ^ (v >> 30)) * 0xBF58476D1CE4E5B9) & mask
    v = ((v ^ (v >> 27)) * 0x94D049BB133111EB) & mask
    return v ^ (v >> 31)


def main():
    started = time.perf_counter()
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    manifest = json.loads((ROOT / 'selected_fit_manifest.json').read_text())
    original = json.loads((ROOT / 'fit_manifest.json').read_text())
    trials = json.loads((ROOT / 'regularization_trials.json').read_text())['trials']
    receipt = json.loads((ROOT / 'evaluation_started.json').read_text())
    result = json.loads((ROOT / 'results.json').read_text())
    assert sha(SOURCE / 'audit_arrays.npz') == protocol['source_cache_sha256']
    assert sha(ROOT / 'data_contract.npz') == protocol['data_contract_sha256']
    assert sha(ROOT / 'MATHEMATICAL_FOUNDATION.md') == protocol['math_spec_sha256']
    assert sha(ROOT / 'protocol.json') == manifest['protocol_sha256'] == result['protocol_sha256']
    assert sha(ROOT / 'protocol_amendment_01.json') == manifest['amendment_sha256'] == result['amendment_sha256']
    assert sha(ROOT / 'selected_fit_manifest.json') == receipt['selected_fit_manifest_sha256'] == result['selected_fit_manifest_sha256']
    assert sha(ROOT / 'evaluate_pilot.py') == receipt['evaluation_code_sha256'] == result['evaluation_code_sha256']
    for file, value in manifest['code_hashes'].items():
        assert sha(ROOT / file) == value
    assert protocol['recorded_utc'] < manifest['frozen_utc'] <= receipt['utc'] <= result['completed_utc']
    d = np.load(ROOT / 'data_contract.npz')
    source = np.load(SOURCE / 'audit_arrays.npz')
    ne, nc = len(d['evaluation_users']), len(d['calibration_users'])
    assert not np.intersect1d(d['evaluation_users'], d['calibration_users']).size
    np.testing.assert_array_equal([scalar_hash(x, 20261016) % 10 == 0 for x in d['calibration_users']], True)
    np.testing.assert_array_equal([scalar_hash(x, 20261016) % 10 != 0 for x in d['evaluation_users']], True)
    for name in ['train', 'validation']:
        idx = d[name + '_source_indices']
        assert np.all(source[name + '__tab'][idx] == 1)
        np.testing.assert_array_equal(d[name + '_y'], source[name + '__long_view'][idx])
    bootstrap = np.load(ROOT / 'effect_bootstrap.npz')
    stored = np.load(ROOT / 'user_risks_and_calibration.npz')
    model_metrics = {(r['name'], r['policy']): r for r in csv.DictReader((ROOT / 'model_metrics.csv').open())}
    datasets, scores = {}, {}
    for policy in ['standard', 'random']:
        scores[policy, 'evaluation'] = np.load(ROOT / f'scores_{policy}_evaluation.npz')
        scores[policy, 'calibration'] = np.load(ROOT / f'scores_{policy}_calibration.npz')
        for part, roster in [('calibration', d['calibration_users']), ('evaluation', d['evaluation_users'])]:
            idx = d[f'{part}_{policy}_indices']
            assert np.all(source[f'test_{policy}__tab'][idx] == 1)
            raw_u, raw_i, y = [source[f'test_{policy}__{c}'][idx] for c in ['user_id', 'video_id', 'long_view']]
            np.testing.assert_array_equal(roster[d[f'{part}_{policy}_roster_u']], raw_u)
            assert np.isin(raw_i, d['common_items']).all()
            datasets[policy, part] = {'u': d['user_lookup'][raw_u], 'i': d['item_lookup'][raw_i], 'y': y, 'uid': d[f'{part}_{policy}_roster_u']}
        assert not np.intersect1d(d[f'calibration_{policy}_indices'], d[f'evaluation_{policy}_indices']).size
    rng = np.random.default_rng(20261016)
    we = np.array([np.bincount(rng.integers(0, ne, size=ne), minlength=ne) for _ in range(1000)], dtype='float32')
    wc = np.array([np.bincount(rng.integers(0, nc, size=nc), minlength=nc) for _ in range(1000)], dtype='float32')
    risks, calibration = {}, {}
    errors = {'score': 0., 'risk': 0., 'auc_sample': 0., 'bootstrap': 0.}
    auc_users = np.flatnonzero(stored['auc_eligible'])
    sample_users = np.random.default_rng(1610).choice(auc_users, size=min(101, len(auc_users)), replace=False)
    replayed_auc = 0
    for row in manifest['models']:
        name = row['name']; a = np.load(ROOT / row['path'])
        assert sha(ROOT / row['path']) == row['model_sha256']
        candidates = [x for x in original['models'] + trials if x['method'] == row['method'] and x['seed'] == row['seed']]
        assert len(candidates) == 3
        assert abs(row['validation_macro_loss'] - min(c['validation_macro_loss'] for c in candidates)) < 1e-12
        val_score = independent_score(a, d['validation_u'], d['validation_i'])
        vn = np.bincount(d['validation_u'], minlength=len(d['raw_users']))
        vl = np.bincount(d['validation_u'], weights=logistic_loss(d['validation_y'], val_score), minlength=len(vn))
        assert abs((vl[vn > 0] / vn[vn > 0]).mean() - row['validation_macro_loss']) < 1e-12
        parameter_bytes = sum(a[k].nbytes for k in a.files if k not in ['mapping', 'method'])
        assert parameter_bytes == row['parameter_bytes']
        assert parameter_bytes + a['mapping'].nbytes + d['user_lookup'].nbytes + d['item_lookup'].nbytes == row['inference_array_bytes']
        if row['method'] in ['random', 'frequency']:
            assert a['user_factors'].shape == (protocol['group_budget'], 16)
            for u in np.linspace(0, len(d['raw_users']) - 1, 101).astype(int):
                value = scalar_hash(d['raw_users'][u], row['seed'])
                if row['method'] == 'random':
                    assert a['mapping'][u] == value % protocol['group_budget']
            if row['method'] == 'frequency':
                n = np.bincount(d['train_u'], minlength=len(d['raw_users']))
                order = np.lexsort((d['raw_users'], -n)); head = protocol['group_budget'] // 2
                np.testing.assert_array_equal(a['mapping'][order[:head]], np.arange(head))
                assert np.all(a['mapping'][order[head:]] >= head)
        for policy in ['standard', 'random']:
            mm = model_metrics[name, policy]
            for part in ['calibration', 'evaluation']:
                ds = datasets[policy, part]
                z = independent_score(a, ds['u'], ds['i'])
                saved = scores[policy, part][name]
                errors['score'] = max(errors['score'], float(np.abs(z - saved).max()))
                np.testing.assert_allclose(z, saved, atol=1e-11, rtol=0)
            cd, ed = datasets[policy, 'calibration'], datasets[policy, 'evaluation']
            shift = float(mm['calibration_intercept'])
            zc, ze = scores[policy, 'calibration'][name], scores[policy, 'evaluation'][name]
            prc = 1 / (1 + np.exp(-(zc + shift)))
            g = means(prc - cd['y'], cd['uid'], nc)
            h = float(means(prc * (1 - prc), cd['uid'], nc).mean())
            assert abs(g.mean()) < 1e-10
            assert abs(h - float(mm['calibration_hessian'])) < 1e-12
            pre = 1 / (1 + np.exp(-(ze + shift)))
            derivative = float(means(pre - ed['y'], ed['uid'], ne).mean())
            calibration[name, policy] = derivative * (-(wc @ g - g.sum()) / (nc * h))
            rr, rc = means(logistic_loss(ed['y'], ze), ed['uid'], ne), means(logistic_loss(ed['y'], ze + shift), ed['uid'], ne)
            risks[name, policy, 'raw'], risks[name, policy, 'cal'] = rr, rc
            for metric, value in [('raw', rr), ('cal', rc)]:
                np.testing.assert_allclose(value, stored[f'{name}__{policy}__{metric}'], atol=1e-12, rtol=0)
            assert abs(rr.mean() - float(mm['raw_macro_log_loss'])) < 1e-12
            assert abs(rc.mean() - float(mm['calibrated_macro_log_loss'])) < 1e-12
            for u in sample_users:
                us = ed['uid'] == u
                pos, neg = ze[us & (ed['y'] == 1)], ze[us & (ed['y'] == 0)]
                expected = float(((pos[:, None] > neg).astype(float) + .5 * (pos[:, None] == neg)).mean())
                actual = stored[f'{name}__{policy}__auc'][u]
                errors['auc_sample'] = max(errors['auc_sample'], abs(expected - actual))
                assert abs(expected - actual) < 1e-12
                replayed_auc += 1
    for method in ['random', 'frequency', 'rank2']:
        for seed in [11, 29, 47]:
            name, ref = f'{method}_s{seed}', f'full_s{seed}'
            for version in ['raw', 'cal']:
                gu = risks[name, 'random', version] - risks[ref, 'random', version] - risks[name, 'standard', version] + risks[ref, 'standard', version]
                expected = we @ gu / ne
                if version == 'cal':
                    expected += calibration[name, 'random'] - calibration[ref, 'random'] - calibration[name, 'standard'] + calibration[ref, 'standard']
                actual = bootstrap[f'{name}_{version}']
                errors['bootstrap'] = max(errors['bootstrap'], float(np.abs(expected - actual).max()))
                np.testing.assert_allclose(expected, actual, atol=1e-12, rtol=0)
        for version in ['raw', 'cal', 'auc']:
            expected = np.mean([bootstrap[f'{method}_s{s}_{version}'] for s in [11, 29, 47]], axis=0)
            np.testing.assert_array_equal(expected, bootstrap[f'mean_{method}_{version}'])
            mean = result['averaged_effects'][method][version]
            np.testing.assert_allclose(np.quantile(expected, [.025, .975]), mean['ci_95'], atol=1e-12)
            if version == 'cal':
                np.testing.assert_allclose(np.quantile(expected, [.005, .995]), mean['ci_family_99'], atol=1e-12)
    for method in ['random', 'frequency']:
        x = bootstrap[f'mean_{method}_cal'] - bootstrap['mean_rank2_cal']
        np.testing.assert_allclose(np.quantile(x, [.005, .995]), result['specificity_vs_rank2'][method]['ci_family_99'], atol=1e-12)
        a, raw = result['averaged_effects'][method]['cal'], result['averaged_effects'][method]['raw']
        conditions = {'material_gap': abs(a['gap']) >= .001, 'family_interval_excludes_zero': a['ci_family_99'][0] > 0 or a['ci_family_99'][1] < 0, 'all_seed_directions_agree': all(np.sign(v) == np.sign(a['gap']) for v in a['seed_gaps']), 'raw_and_calibrated_direction_agree': np.sign(raw['gap']) == np.sign(a['gap']), 'calibration_approximation_guard': result['maximum_calibration_bootstrap_spotcheck_error'] <= .0005}
        assert conditions == result['flags'][method]['conditions']
        assert all(conditions.values()) == result['flags'][method]['descriptive_material_policy_gap']
    # Check all 240 exact-refit intercepts solve the sampled calibration equations.
    approx = json.loads((ROOT / 'calibration_approximation_check.json').read_text())
    for row in approx['exact_intercepts']:
        cd = datasets[row['policy'], 'calibration']
        n = np.bincount(cd['uid'], minlength=nc)
        weights = wc[row['draw'], cd['uid']] / n[cd['uid']]
        prob = 1 / (1 + np.exp(-(scores[row['policy'], 'calibration'][row['model']] + row['exact_shift'])))
        assert abs(float(np.sum(weights * (prob - cd['y']))) / nc) < 1e-10
    shifts = {(x['draw'], x['model'], x['policy']): x['exact_shift'] for x in approx['exact_intercepts']}
    cached_scores = {(name, policy): scores[policy, 'evaluation'][name] for name in approx['exact_bootstrap_gaps'] for policy in ['standard', 'random']}
    for seed in [11, 29, 47]:
        for policy in ['standard', 'random']:
            name = f'full_s{seed}'
            cached_scores[name, policy] = scores[policy, 'evaluation'][name]
    exact_errors = []
    for b in range(10):
        exact_risks = {}
        for policy in ['standard', 'random']:
            ed = datasets[policy, 'evaluation']
            n = np.bincount(ed['uid'], minlength=ne)
            ew = we[b, ed['uid']] / n[ed['uid']] / ne
            for row in manifest['models']:
                name = row['name']
                exact_risks[name, policy] = float(np.sum(ew * logistic_loss(ed['y'], cached_scores[name, policy] + shifts[b, name, policy])))
        for name, values in approx['exact_bootstrap_gaps'].items():
            ref = 'full_s' + name.split('_s')[-1]
            expected = exact_risks[name, 'random'] - exact_risks[ref, 'random'] - exact_risks[name, 'standard'] + exact_risks[ref, 'standard']
            assert abs(expected - values[b]) < 1e-12
            exact_errors.append(abs(expected - bootstrap[name + '_cal'][b]))
    assert abs(max(exact_errors) - approx['maximum_absolute_exact_vs_first_order_gap_error']) < 1e-12
    check = {'status': 'independent_artifact_checks_passed', 'selected_MF_models_replayed': len(manifest['models']), 'all_policy_score_vectors_replayed': 48, 'raw_and_calibrated_macro_risks_recomputed': 48, 'individual_user_AUCs_bruteforce_checked': replayed_auc, 'paired_risk_bootstrap_vectors_recomputed': 18, 'exact_calibration_root_equations_checked': len(approx['exact_intercepts']), 'exact_calibration_bootstrap_gaps_recomputed': len(exact_errors), 'selection_from_equal_three_candidate_validation_grid_verified': True, 'max_absolute_errors': errors, 'continue_benchmark_recomputed': result['continue_broader_benchmark'], 'limitations': ['Conditional on selected trained models and observed period.', 'Calibration uncertainty uses an asymptotic influence approximation, spot-checked rather than exact bootstrap coverage.', 'Full interaction validation gain is weak; no broad embedding-compression conclusion.', 'Native-label documentation mismatch and incomplete sequences remain unresolved.'], 'elapsed_seconds': time.perf_counter() - started}
    (ROOT / 'verification.json').write_text(json.dumps(check, indent=2), encoding='utf-8')
    print(json.dumps(check), flush=True)


if __name__ == '__main__':
    main()
