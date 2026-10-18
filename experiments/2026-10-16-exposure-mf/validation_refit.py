"""One recorded, validation-only regularization check; preserve original fits."""
import math
import time
from pathlib import Path
import numpy as np
import torch
import run_pilot as p
from exposure_math import loss, user_means, user_auc

ROOT = Path(__file__).resolve().parent


def main():
    if (ROOT / 'selected_fit_manifest.json').exists():
        raise RuntimeError('Selected fits already frozen')
    original = __import__('json').loads((ROOT / 'fit_manifest.json').read_text())
    protocol = __import__('json').loads((ROOT / 'protocol.json').read_text())
    amendment = {'recorded_utc': p.utc(), 'stage': 'before_first_new_MF_policy_evaluation',
        'reason': 'Validation-only ablation and learning curves show rapid latent-interaction overfit under initial regularization. Synthetic opposite-ranking component check passed.',
        'original_fit_manifest_sha256': p.sha(ROOT / 'fit_manifest.json'),
        'quality_check_sha256': p.sha(ROOT / 'pre_evaluation_quality_check.json'),
        'change': 'Expand factor-regularization candidates from .0001 to [.0001,.16,1.6], equally for every method/seed. Change only factor regularization; all other initialization, optimizer, bias regularization, epoch/patience limits and cohorts fixed.',
        'selection': 'Minimum logged-validation macro-user loss across candidate/epoch; original candidate preserved. No evaluation scores or calibration outcomes accessed. No further validation-repair grid planned.',
        'interpretation': 'Still exploratory; report validation zero-interaction ablation and do not treat a weak full reference as a validated preference model.'}
    p.save('protocol_amendment_01.json', amendment)
    data = np.load(ROOT / 'data_contract.npz')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    u, i, y = [torch.from_numpy(data['train_' + k].astype('int64' if k != 'y' else 'float32')) for k in ['u', 'i', 'y']]
    rows, trial_records = [], []
    destination = ROOT / 'regularization_models'
    destination.mkdir(exist_ok=True)
    started = time.perf_counter()
    for seed in p.SEEDS:
        for method in p.METHODS:
            for factor_lambda in [.16, 1.6]:
                model = p.MF(method, seed, data, protocol['group_budget'])
                optimizer = torch.optim.Adam(model.parameters(), lr=.03)
                best_loss, progress_loss, stale, best, best_epoch = math.inf, math.inf, 0, None, None
                history = []
                for epoch in range(1, 26):
                    order = np.random.default_rng(seed + 100003 * epoch).permutation(len(u))
                    train_loss = 0.
                    for start in range(0, len(u), 32768):
                        idx = torch.from_numpy(order[start:start + 32768])
                        bu, bi = u[idx], i[idx]
                        optimizer.zero_grad(set_to_none=True)
                        score, base_regularizer = model(bu, bi)
                        effective = model.user_factors[model.mapping[bu]]
                        if model.projection is not None:
                            effective = effective @ model.projection
                        q = model.item_factors[bi]
                        factor_penalty = (effective.square().sum(dim=1) + q.square().sum(dim=1)).mean() / 32
                        bce = torch.nn.functional.binary_cross_entropy_with_logits(score, y[idx])
                        objective = bce + base_regularizer + (factor_lambda - .0001) * factor_penalty
                        if not bool(torch.isfinite(objective)):
                            raise RuntimeError('Nonfinite regularization trial')
                        objective.backward(); optimizer.step()
                        train_loss += float(bce.detach()) * len(idx)
                    arrays = model.arrays()
                    score = p.score_numpy(arrays, data['validation_u'], data['validation_i'])
                    vl = float(np.nanmean(user_means(loss(data['validation_y'], score), data['validation_u'], len(data['raw_users']))))
                    row = {'method': method, 'seed': seed, 'factor_lambda': factor_lambda, 'epoch': epoch, 'train_event_bce': train_loss / len(u), 'validation_macro_loss': vl}
                    rows.append(row); history.append(row)
                    if vl < best_loss:
                        best_loss, best, best_epoch = vl, arrays, epoch
                    if vl < progress_loss - 1e-5:
                        progress_loss, stale = vl, 0
                    else:
                        stale += 1
                    if stale >= 4:
                        break
                name = f'{method}_s{seed}_lambda{factor_lambda:g}'
                path = destination / (name + '.npz')
                np.savez_compressed(path, **best)
                trial_records.append({'name': name, 'method': method, 'seed': seed, 'factor_lambda': factor_lambda, 'validation_macro_loss': best_loss, 'best_epoch': best_epoch, 'epochs_run': len(history), 'path': str(path.relative_to(ROOT)), 'model_sha256': p.sha(path), 'history': history})
                print(__import__('json').dumps({'validation_trial': name, 'best_loss': best_loss, 'best_epoch': best_epoch, 'seconds': time.perf_counter() - started}), flush=True)
    p.csv_save('regularization_training_history.csv', rows)
    p.save('regularization_trials.json', {'trials': trial_records, 'seconds': time.perf_counter() - started})
    selected, quality = [], []
    for old in original['models']:
        candidates = [{**old, 'factor_lambda': .0001, 'path': 'models/' + old['name'] + '.npz'}]
        candidates += [x for x in trial_records if x['method'] == old['method'] and x['seed'] == old['seed']]
        chosen = min(candidates, key=lambda x: x['validation_macro_loss'])
        selected.append({'name': old['name'], 'method': old['method'], 'seed': old['seed'], 'factor_lambda': chosen['factor_lambda'], 'best_epoch': chosen['best_epoch'], 'epochs_run': chosen['epochs_run'], 'validation_macro_loss': chosen['validation_macro_loss'], 'path': chosen['path'], 'model_sha256': chosen['model_sha256'], 'parameter_bytes': old['parameter_bytes'], 'inference_array_bytes': old['inference_array_bytes']})
        if old['method'] == 'full':
            a = np.load(ROOT / chosen['path'])
            score = p.score_numpy(a, data['validation_u'], data['validation_i'])
            zero = float(a['global_bias'][0]) + a['user_bias'][data['validation_u']].astype('float64') + a['item_bias'][data['validation_i']].astype('float64')
            n = len(data['raw_users'])
            quality.append({'seed': old['seed'], 'selected_factor_lambda': chosen['factor_lambda'], 'full_validation_loss': float(np.nanmean(user_means(loss(data['validation_y'], score), data['validation_u'], n))), 'zero_interaction_validation_loss': float(np.nanmean(user_means(loss(data['validation_y'], zero), data['validation_u'], n))), 'full_validation_auc': float(np.nanmean(user_auc(data['validation_y'], score, data['validation_u'], n))), 'zero_interaction_validation_auc': float(np.nanmean(user_auc(data['validation_y'], zero, data['validation_u'], n))), 'interaction_score_sd': float(np.std(score - zero))})
    p.save('selected_fit_manifest.json', {'frozen_utc': p.utc(), 'protocol_sha256': p.sha(ROOT / 'protocol.json'), 'amendment_sha256': p.sha(ROOT / 'protocol_amendment_01.json'), 'original_fit_manifest_sha256': p.sha(ROOT / 'fit_manifest.json'), 'code_hashes': {'run_pilot.py': p.sha(ROOT / 'run_pilot.py'), 'exposure_math.py': p.sha(ROOT / 'exposure_math.py'), 'validation_refit.py': p.sha(ROOT / 'validation_refit.py')}, 'models': selected, 'total_fits': 36, 'fit_seconds': original['fit_seconds'] + time.perf_counter() - started, 'full_reference_quality': quality, 'test_outcomes_loaded_in_fit': False})
    print(__import__('json').dumps({'selected_quality': quality, 'selected_models': [{k: x[k] for k in ['name', 'factor_lambda', 'validation_macro_loss']} for x in selected]}), flush=True)


if __name__ == '__main__':
    main()
