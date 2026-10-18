"""Frozen evaluation of exposure-policy risk contrasts, with calibration influence correction."""
import json
import time
from pathlib import Path
import numpy as np
import run_pilot as p
from exposure_math import loss, user_means, user_auc, fit_intercept, calibration_if

ROOT = Path(__file__).resolve().parent


def bounds(a, alpha=.05):
    return np.quantile(a[np.isfinite(a)], [alpha / 2, 1 - alpha / 2]).tolist()


def exact_recalibration(y, z, uid, count, weights, initial):
    ew = weights[uid] / count[uid]
    b = initial
    for _ in range(20):
        pr = 1 / (1 + np.exp(-np.clip(z + b, -80, 80)))
        g = float(np.sum(ew * (pr - y)))
        h = float(np.sum(ew * pr * (1 - pr)))
        step = np.clip(g / h, -2, 2)
        b -= step
        if abs(step) < 1e-11:
            break
    return b


def main():
    started = time.perf_counter()
    if (ROOT / 'results.json').exists():
        raise RuntimeError('Results exist; no overwrite')
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    manifest = json.loads((ROOT / 'selected_fit_manifest.json').read_text())
    assert manifest['protocol_sha256'] == p.sha(ROOT / 'protocol.json')
    assert manifest['amendment_sha256'] == p.sha(ROOT / 'protocol_amendment_01.json')
    assert p.sha(ROOT / 'data_contract.npz') == protocol['data_contract_sha256']
    for file, digest in manifest['code_hashes'].items():
        assert p.sha(ROOT / file) == digest
    p.save('evaluation_started.json', {'utc': p.utc(), 'selected_fit_manifest_sha256': p.sha(ROOT / 'selected_fit_manifest.json'), 'evaluation_code_sha256': p.sha(ROOT / 'evaluate_pilot.py'), 'protocol_sha256': p.sha(ROOT / 'protocol.json'), 'exact_recalibration_draws': 10, 'calibration_approximation_guard': 'max absolute exact-vs-IF bootstrap contrast error <=.0005 on ten fixed draws; otherwise statistical flag withheld', 'reference_quality_guard': 'Broader benchmark escalation additionally requires full interactions to improve both validation loss and validation AUC over the frozen zero-interaction ablation in every seed. Failure is a reference-quality limitation, not a rejection of the exposure hypothesis.'})
    source = np.load(p.SOURCE / 'audit_arrays.npz')
    d = np.load(ROOT / 'data_contract.npz')
    nc, ne = len(d['calibration_users']), len(d['evaluation_users'])
    datasets, score_eval, score_cal = {}, {'standard': {}, 'random': {}}, {'standard': {}, 'random': {}}
    cohort, auc_ok = {}, np.ones(ne, dtype=bool)
    for policy in ['standard', 'random']:
        cohort[policy] = {}
        for part, nn in [('calibration', nc), ('evaluation', ne)]:
            idx = d[f'{part}_{policy}_indices']
            uid = d[f'{part}_{policy}_roster_u']
            raw_u = source[f'test_{policy}__user_id'][idx]
            raw_i = source[f'test_{policy}__video_id'][idx]
            y = source[f'test_{policy}__long_view'][idx]
            du, di = p.map_ids(raw_u, d['user_lookup']), p.map_ids(raw_i, d['item_lookup'])
            n = np.bincount(uid, minlength=nn)
            assert np.all(n > 0) and np.all(du >= 0) and np.all(di >= 0)
            positive = np.bincount(uid, weights=y, minlength=nn)
            datasets[policy, part] = {'u': du, 'i': di, 'y': y, 'uid': uid, 'counts': n}
            cohort[policy][part] = {'rows': len(y), 'users': nn, 'event_positive_rate': float(y.mean()), 'macro_user_positive_rate': float((positive / n).mean())}
            if part == 'evaluation':
                auc_ok &= (positive > 0) & (positive < n)
    rng = np.random.default_rng(20261016)
    we = np.array([np.bincount(rng.integers(0, ne, size=ne), minlength=ne) for _ in range(1000)], dtype='float32')
    wc = np.array([np.bincount(rng.integers(0, nc, size=nc), minlength=nc) for _ in range(1000)], dtype='float32')
    summary, cal_info, metric_rows, user_arrays = {}, {}, [], {'auc_eligible': auc_ok, 'evaluation_users': d['evaluation_users'], 'calibration_users': d['calibration_users']}
    for model in manifest['models']:
        name = model['name']
        assert p.sha(ROOT / model['path']) == model['model_sha256']
        arrays = np.load(ROOT / model['path'])
        for policy in ['standard', 'random']:
            cd, ed = datasets[policy, 'calibration'], datasets[policy, 'evaluation']
            zc = p.score_numpy(arrays, cd['u'], cd['i'])
            ze = p.score_numpy(arrays, ed['u'], ed['i'])
            score_cal[policy][name], score_eval[policy][name] = zc, ze
            shift, g, h = fit_intercept(cd['y'], zc, cd['uid'], nc)
            raw_risk = user_means(loss(ed['y'], ze), ed['uid'], ne)
            cal_risk = user_means(loss(ed['y'], ze + shift), ed['uid'], ne)
            probability = 1 / (1 + np.exp(-np.clip(ze + shift, -80, 80)))
            derivative = float(user_means(probability - ed['y'], ed['uid'], ne).mean())
            auc = user_auc(ed['y'], ze, ed['uid'], ne)
            summary[name, policy] = {'raw': raw_risk, 'cal': cal_risk, 'auc': auc, 'derivative': derivative}
            cal_info[name, policy] = {'shift': shift, 'g': g, 'h': h, 'delta_boot': calibration_if(wc, g, h)}
            for key, value in [('raw', raw_risk), ('cal', cal_risk), ('auc', auc), ('cal_g', g)]:
                user_arrays[f'{name}__{policy}__{key}'] = value
            row = {'name': name, 'method': model['method'], 'seed': model['seed'], 'policy': policy, 'raw_macro_log_loss': float(raw_risk.mean()), 'calibrated_macro_log_loss': float(cal_risk.mean()), 'macro_auc_on_common_eligible_users': float(auc[auc_ok].mean()), 'auc_users': int(auc_ok.sum()), 'calibration_intercept': shift, 'calibration_hessian': h, 'calibration_mean_gradient': float(g.mean()), 'evaluation_calibration_derivative': derivative, 'inference_bytes': model['inference_array_bytes'], 'factor_lambda': model['factor_lambda']}
            metric_rows.append(row)
        print(json.dumps({'evaluated': name}), flush=True)
    for policy in ['standard', 'random']:
        np.savez_compressed(ROOT / f'scores_{policy}_evaluation.npz', **score_eval[policy])
        np.savez_compressed(ROOT / f'scores_{policy}_calibration.npz', **score_cal[policy])
    np.savez_compressed(ROOT / 'user_risks_and_calibration.npz', **user_arrays)
    p.csv_save('model_metrics.csv', metric_rows)
    effects, effect_arrays, seed_rows = {}, {}, []
    for method in ['random', 'frequency', 'rank2']:
        for seed in p.SEEDS:
            name, ref = f'{method}_s{seed}', f'full_s{seed}'
            effect = {}
            for version in ['raw', 'cal']:
                dl = summary[name, 'standard'][version] - summary[ref, 'standard'][version]
                dr = summary[name, 'random'][version] - summary[ref, 'random'][version]
                gap_user = dr - dl
                boot = we @ gap_user / ne
                if version == 'cal':
                    for policy, sign in [('random', 1.), ('standard', -1.)]:
                        boot += sign * (summary[name, policy]['derivative'] * cal_info[name, policy]['delta_boot'] - summary[ref, policy]['derivative'] * cal_info[ref, policy]['delta_boot'])
                effect[version] = {'delta_standard': float(dl.mean()), 'delta_random': float(dr.mean()), 'gap': float(gap_user.mean()), 'bootstrap': boot}
                effect_arrays[f'{name}_{version}'] = boot
            # AUC penalty has higher-is-better sign convention.
            al = summary[ref, 'standard']['auc'][auc_ok] - summary[name, 'standard']['auc'][auc_ok]
            ar = summary[ref, 'random']['auc'][auc_ok] - summary[name, 'random']['auc'][auc_ok]
            wg = we[:, auc_ok]
            ab = wg @ (ar - al) / wg.sum(axis=1)
            effect['auc'] = {'delta_standard': float(al.mean()), 'delta_random': float(ar.mean()), 'gap': float((ar - al).mean()), 'bootstrap': ab}
            effect_arrays[f'{name}_auc'] = ab
            effects[name] = effect
            seed_rows.append({'method': method, 'seed': seed, **{f'{v}_{k}': effect[v][k] for v in ['raw', 'cal', 'auc'] for k in ['delta_standard', 'delta_random', 'gap']}})
    p.csv_save('seed_effects.csv', seed_rows)
    averaged, contrast_rows = {}, []
    for method in ['random', 'frequency', 'rank2']:
        averaged[method] = {}
        for version in ['raw', 'cal', 'auc']:
            per_seed = [effects[f'{method}_s{s}'][version] for s in p.SEEDS]
            boot = np.mean([e['bootstrap'] for e in per_seed], axis=0)
            mean = {k: float(np.mean([e[k] for e in per_seed])) for k in ['delta_standard', 'delta_random', 'gap']}
            mean.update(ci_95=bounds(boot), ci_family_99=bounds(boot, .01) if version == 'cal' else None, seed_gaps=[e['gap'] for e in per_seed])
            averaged[method][version] = mean
            effect_arrays[f'mean_{method}_{version}'] = boot
            contrast_rows.append({'method': method, 'metric': version, **{k: mean[k] for k in ['delta_standard', 'delta_random', 'gap']}, 'ci95_low': mean['ci_95'][0], 'ci95_high': mean['ci_95'][1], 'ci_family99_low': mean['ci_family_99'][0] if mean['ci_family_99'] else None, 'ci_family99_high': mean['ci_family_99'][1] if mean['ci_family_99'] else None})
    specificity = {}
    for method in ['random', 'frequency']:
        boot = effect_arrays[f'mean_{method}_cal'] - effect_arrays['mean_rank2_cal']
        estimate = averaged[method]['cal']['gap'] - averaged['rank2']['cal']['gap']
        specificity[method] = {'estimate': estimate, 'ci_95': bounds(boot), 'ci_family_99': bounds(boot, .01)}
        effect_arrays[f'{method}_minus_rank2_cal'] = boot
    # Exact recalibration of the first10 predetermined bootstrap draws.
    exact = {f'{method}_s{s}': [] for method in ['random', 'frequency', 'rank2'] for s in p.SEEDS}
    exact_shifts = []
    for b in range(10):
        exact_risks = {}
        for policy in ['standard', 'random']:
            cd, ed = datasets[policy, 'calibration'], datasets[policy, 'evaluation']
            ew = we[b, ed['uid']] / ed['counts'][ed['uid']] / ne
            for model in manifest['models']:
                name = model['name']
                shift = exact_recalibration(cd['y'], score_cal[policy][name], cd['uid'], cd['counts'], wc[b], cal_info[name, policy]['shift'])
                value = float(np.sum(ew * loss(ed['y'], score_eval[policy][name] + shift)))
                exact_risks[name, policy] = value
                exact_shifts.append({'draw': b, 'model': name, 'policy': policy, 'exact_shift': shift})
        for name in exact:
            seed = int(name.split('_s')[-1]); ref = f'full_s{seed}'
            exact[name].append(exact_risks[name, 'random'] - exact_risks[ref, 'random'] - exact_risks[name, 'standard'] + exact_risks[ref, 'standard'])
    errors = [abs(np.asarray(vals) - effect_arrays[name + '_cal'][:10]).max() for name, vals in exact.items()]
    max_error = float(max(errors))
    p.save('calibration_approximation_check.json', {'draws': 10, 'exact_bootstrap_gaps': exact, 'exact_intercepts': exact_shifts, 'maximum_absolute_exact_vs_first_order_gap_error': max_error, 'guard_threshold': .0005, 'guard_pass': max_error <= .0005, 'scope': 'Finite spot check of a first-order approximation, not exact CI coverage.'})
    np.savez_compressed(ROOT / 'effect_bootstrap.npz', **effect_arrays)
    p.csv_save('average_effects.csv', contrast_rows)
    flags = {}
    for method in ['random', 'frequency']:
        cal, raw = averaged[method]['cal'], averaged[method]['raw']
        lo, hi = cal['ci_family_99']
        sign = np.sign(cal['gap'])
        conditions = {'material_gap': abs(cal['gap']) >= .001, 'family_interval_excludes_zero': lo > 0 or hi < 0, 'all_seed_directions_agree': all(np.sign(x) == sign for x in cal['seed_gaps']), 'raw_and_calibrated_direction_agree': np.sign(raw['gap']) == sign, 'calibration_approximation_guard': max_error <= .0005}
        flags[method] = {'conditions': conditions, 'descriptive_material_policy_gap': all(conditions.values()), 'sharing_specificity_interval_excludes_zero': specificity[method]['ci_family_99'][0] > 0 or specificity[method]['ci_family_99'][1] < 0}
    reference_quality = manifest['full_reference_quality']
    quality_pass = all(q['full_validation_loss'] < q['zero_interaction_validation_loss'] and q['full_validation_auc'] > q['zero_interaction_validation_auc'] for q in reference_quality)
    result = {'completed_utc': p.utc(), 'status': 'exploratory_MF_exposure_pilot_executed', 'protocol_sha256': p.sha(ROOT / 'protocol.json'), 'amendment_sha256': p.sha(ROOT / 'protocol_amendment_01.json'), 'selected_fit_manifest_sha256': p.sha(ROOT / 'selected_fit_manifest.json'), 'evaluation_code_sha256': p.sha(ROOT / 'evaluate_pilot.py'), 'models_evaluated': len(manifest['models']), 'total_training_trials': manifest['total_fits'], 'cohorts': cohort, 'auc_common_eligible_users': int(auc_ok.sum()), 'averaged_effects': averaged, 'specificity_vs_rank2': specificity, 'flags': flags, 'full_reference_quality': reference_quality, 'full_interaction_validation_quality_pass': quality_pass, 'continue_broader_benchmark': quality_pass and any(f['descriptive_material_policy_gap'] for f in flags.values()), 'maximum_calibration_bootstrap_spotcheck_error': max_error, 'fit_seconds': manifest['fit_seconds'], 'evaluation_seconds': time.perf_counter() - started, 'scope': 'Policy-dependent predictive risk for this fitted model family and declared population. Not intrinsic preferences, identified exposure mechanism, published-method rankings, independent confirmation or novelty.'}
    p.save('results.json', result)
    print(json.dumps(p.clean(result)), flush=True)


if __name__ == '__main__':
    main()
