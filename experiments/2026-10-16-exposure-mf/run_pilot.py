"""Throwaway MF/exposure-policy pilot; fixed protocol, local CPU, no IPS or causal claim."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import numpy as np
import torch
from exposure_math import loss, user_means, user_auc, fit_intercept, calibration_if, matched_capacity

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / '2026-10-15-kuairand-support'
SEEDS = [11, 29, 47]
METHODS = ['full', 'random', 'frequency', 'rank2']


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def save(name, value):
    (ROOT / name).write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding='utf-8')


def csv_save(name, rows):
    with (ROOT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(clean(rows))


def hash_ids(ids, seed):
    with np.errstate(over='ignore'):
        z = np.asarray(ids, dtype='uint64') + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> 31)


def map_ids(ids, table):
    result = np.full(len(ids), -1, dtype='int32')
    ok = (ids >= 0) & (ids < len(table))
    result[ok] = table[ids[ok]]
    return result


def prepare():
    if (ROOT / 'protocol.json').exists():
        raise RuntimeError('Protocol exists; no overwrite')
    z = np.load(SOURCE / 'audit_arrays.npz')
    u, i, s = [z['train__' + k] for k in ['user_id', 'video_id', 'tab']]
    train_mask = s == 1
    raw_users, raw_items = np.unique(u[train_mask]), np.unique(i[train_mask])
    ul = np.full(int(raw_users.max()) + 1, -1, dtype='int32')
    il = np.full(int(raw_items.max()) + 1, -1, dtype='int32')
    ul[raw_users], il[raw_items] = np.arange(len(raw_users)), np.arange(len(raw_items))
    budget, dense_parameters, shared_parameters = matched_capacity(len(raw_users), 16, 2)
    data = {'raw_users': raw_users, 'raw_items': raw_items, 'user_lookup': ul, 'item_lookup': il}
    for name in ['train', 'validation']:
        uu, ii, ss = [z[name + '__' + k] for k in ['user_id', 'video_id', 'tab']]
        du, di = map_ids(uu, ul), map_ids(ii, il)
        keep = (ss == 1) & (du >= 0) & (di >= 0)
        data[name + '_u'], data[name + '_i'] = du[keep], di[keep]
        data[name + '_y'] = z[name + '__long_view'][keep].astype('float32')
        data[name + '_source_indices'] = np.flatnonzero(keep)
    policy_indices = {}
    item_sets = []
    for policy in ['standard', 'random']:
        name = 'test_' + policy
        uu, ii, ss = [z[name + '__' + k] for k in ['user_id', 'video_id', 'tab']]
        keep = (ss == 1) & (map_ids(uu, ul) >= 0) & (map_ids(ii, il) >= 0)
        policy_indices[policy] = np.flatnonzero(keep)
        item_sets.append(np.unique(ii[keep]))
    common_items = np.intersect1d(item_sets[0], item_sets[1])
    for policy, idx in policy_indices.items():
        ii = z['test_' + policy + '__video_id']
        policy_indices[policy] = idx[np.isin(ii[idx], common_items)]
    common_users = np.intersect1d(*[np.unique(z['test_' + p + '__user_id'][idx]) for p, idx in policy_indices.items()])
    is_cal = (hash_ids(common_users, 20261016) % 10) == 0
    calibration_users, evaluation_users = common_users[is_cal], common_users[~is_cal]
    data.update(common_items=common_items, calibration_users=calibration_users, evaluation_users=evaluation_users)
    cohort = {}
    for policy, idx in policy_indices.items():
        uu = z['test_' + policy + '__user_id']
        cohort[policy] = {}
        for part, roster in [('calibration', calibration_users), ('evaluation', evaluation_users)]:
            selected = idx[np.isin(uu[idx], roster)]
            data[f'{part}_{policy}_indices'] = selected
            data[f'{part}_{policy}_roster_u'] = np.searchsorted(roster, uu[selected]).astype('int32')
            cohort[policy][part] = {'rows': len(selected), 'users': len(roster)}
    np.savez_compressed(ROOT / 'data_contract.npz', **data)
    protocol = {
        'recorded_utc': utc(), 'status': 'exploratory_new_MF_fits_on_previously_explored_data',
        'authorization': 'User: okay do it; make it more mathematically grounded. Scope: corrected small MF pilot and formal analysis.',
        'source_cache_sha256': sha(SOURCE / 'audit_arrays.npz'), 'source_verification_sha256': sha(SOURCE / 'verification.json'),
        'data_contract_sha256': sha(ROOT / 'data_contract.npz'), 'math_spec_sha256': sha(ROOT / 'MATHEMATICAL_FOUNDATION.md'),
        'data': 'Existing quarantined duplicate groups, canonical UTC splits and native long_view retained. Native label-rule consistency remains false.',
        'context': 'tab=1 for training, validation, calibration and evaluation',
        'training': {'rows': len(data['train_u']), 'users': len(raw_users), 'items': len(raw_items), 'positive_rate': float(data['train_y'].mean())},
        'validation': {'rows': len(data['validation_u']), 'selection': 'minimum ordinary validation macro-user logistic loss; same fixed config and early-stop rule for all methods'},
        'target_population': {'common_users': len(common_users), 'common_warm_items': len(common_items), 'calibration_users': len(calibration_users), 'evaluation_users': len(evaluation_users), 'cohorts': cohort},
        'roster_selection': 'Label-independent warm-ID and tab support; common observed item support and users across both policies; calibration hash seed20261016 modulo10=0. No test outcomes loaded by prepare/fit.',
        'inference_population': 'Uniform common evaluation users, each policy keeps its observed conditional item/event distribution. This is not intrinsic preferences or full-catalogue utility.',
        'methods': METHODS, 'seeds': SEEDS, 'dimension': 16, 'control_rank': 2, 'group_budget': budget,
        'shared_user_interaction_parameters': shared_parameters, 'rank2_user_interaction_parameters': dense_parameters,
        'model': 'global + individual user bias + item bias + dot(user/group interaction factor,item factor); only the user interaction factor table is shared. Bias tables remain individual in every condition.',
        'capacity': 'Rank2 user factor A_u times shared 2x16 projection; shared budget ceil((2U+32)/16). Shared/random/frequency allocation exactly equal; rank2 differs by <16 scalar parameters. Actual inference arrays and maps reported, not zip bytes.',
        'hash': 'SplitMix64 on original user IDs with training seed. Frequency reserves floor(B/2) private rows for top training-event counts, stable original-ID ties; tail hashes into remaining rows.',
        'initialization': 'Identical item factors per training seed, all biases zero except global train log-odds. User effective-factor scale .1; rank2 projection scale1/sqrt(2). CPU deterministic operations.',
        'training_config': {'device': 'CPU', 'threads': 4, 'batch_size': 32768, 'max_epochs': 25, 'patience': 4, 'patience_min_delta': 1e-5, 'optimizer': 'Adam', 'learning_rate': .03, 'factor_regularizer': .0001, 'bias_regularizer': .00001, 'negative_sampling': False, 'hyperparameter_search': False},
        'calibration': 'Per-model/per-policy intercept-only macro-user BCE minimum on disjoint calibration users. One-dimensional convex bisection; rankings unchanged. This is a separate policy-adapted-risk estimand.',
        'primary_metric': 'macro-user logistic loss; DeltaP=R_P(method)-R_P(full), G=Delta_random-Delta_standard',
        'ranking_secondary': 'macro-user event AUC on users with both classes under both policies; finite-sample exposed-candidate estimand; no catalogue-wide NDCG',
        'uncertainty': '1000 paired user-bootstrap resamples. Separate disjoint calibration/evaluation resamples; same user weights across policies/models/seeds. Calibrated intervals include first-order implicit-function correction; exact-recalibration check on first10 draws.',
        'five_family_contrasts': ['random_G_cal', 'frequency_G_cal', 'rank2_G_cal', 'random_minus_rank2_G_cal', 'frequency_minus_rank2_G_cal'],
        'decision': 'For random/frequency: mean abs(Gcal)>=.001, Bonferroni99% bootstrap interval excludes0 across five declared contrasts, all3 seeds same calibrated sign, raw mean same sign. A flag only permits broader benchmark/controlled exposure study. Rank2 specificity reported separately.',
        'unidentified': 'No exposure propensity estimation, IPS, causal imitation mechanism, universal unbiased population, exact GraphHash/CCE/BACO reproduction, independently confirmed hypothesis, or new-theorem claim.',
        'resource_scope': 'Local CPU pilot only; no GPU/Kaggle/NVIDIA inference, package installation or new data download.',
        'environment': {'numpy': np.__version__, 'torch': str(torch.__version__), 'cuda': torch.cuda.is_available()},
    }
    save('protocol.json', protocol)
    print(json.dumps(clean({'stage': 'prepared', 'training': protocol['training'], 'validation': protocol['validation'], 'target_population': protocol['target_population'], 'budget': budget, 'capacity_difference_floats': shared_parameters - dense_parameters})), flush=True)


class MF(torch.nn.Module):
    def __init__(self, method, seed, data, budget):
        super().__init__()
        nu, ni, dim = len(data['raw_users']), len(data['raw_items']), 16
        mapping = np.arange(nu, dtype='int32')
        if method == 'random':
            mapping = (hash_ids(data['raw_users'], seed) % budget).astype('int32')
        if method == 'frequency':
            head = budget // 2
            n = np.bincount(data['train_u'], minlength=nu)
            order = np.lexsort((data['raw_users'], -n))
            mapping = ((hash_ids(data['raw_users'], seed) % (budget - head)) + head).astype('int32')
            mapping[order[:head]] = np.arange(head)
        self.method = method
        self.register_buffer('mapping', torch.from_numpy(mapping.astype('int64')))
        latent_rows = budget if method in ['random', 'frequency'] else nu
        width = 2 if method == 'rank2' else dim
        factor = np.random.default_rng(seed + 101).normal(0, .1, (latent_rows, width)).astype('float32')
        item = np.random.default_rng(seed + 202).normal(0, .1, (ni, dim)).astype('float32')
        self.user_factors = torch.nn.Parameter(torch.from_numpy(factor))
        self.item_factors = torch.nn.Parameter(torch.from_numpy(item))
        self.user_bias = torch.nn.Parameter(torch.zeros(nu))
        self.item_bias = torch.nn.Parameter(torch.zeros(ni))
        rate = float(data['train_y'].mean())
        self.global_bias = torch.nn.Parameter(torch.tensor([math.log(rate / (1 - rate))], dtype=torch.float32))
        if method == 'rank2':
            projection = np.random.default_rng(seed + 303).normal(0, 1 / math.sqrt(2), (2, dim)).astype('float32')
            self.projection = torch.nn.Parameter(torch.from_numpy(projection))
        else:
            self.register_parameter('projection', None)

    def forward(self, u, i):
        p = self.user_factors[self.mapping[u]]
        if self.projection is not None:
            p = p @ self.projection
        q = self.item_factors[i]
        bu, ci = self.user_bias[u], self.item_bias[i]
        score = self.global_bias + bu + ci + (p * q).sum(dim=1)
        regularizer = .0001 * (p.square().sum(dim=1) + q.square().sum(dim=1)).mean() / 32
        regularizer += .00001 * (bu.square() + ci.square()).mean() / 2
        return score, regularizer

    def arrays(self):
        result = {k: v.detach().cpu().numpy().copy() for k, v in self.named_parameters()}
        result.update(mapping=self.mapping.numpy().astype('int32'), method=np.asarray(self.method))
        return result


def score_numpy(arrays, u, i):
    result = np.empty(len(u), dtype='float64')
    for start in range(0, len(u), 65536):
        stop = min(start + 65536, len(u))
        uu, ii = u[start:stop], i[start:stop]
        p = arrays['user_factors'][arrays['mapping'][uu]].astype('float64')
        if 'projection' in arrays:
            p = p @ arrays['projection'].astype('float64')
        q = arrays['item_factors'][ii].astype('float64')
        result[start:stop] = float(arrays['global_bias'][0]) + arrays['user_bias'][uu].astype('float64') + arrays['item_bias'][ii].astype('float64') + np.einsum('ij,ij->i', p, q)
    return result


def fit():
    if (ROOT / 'fit_manifest.json').exists():
        raise RuntimeError('Frozen fits exist; no overwrite')
    started = time.perf_counter()
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    assert sha(ROOT / 'data_contract.npz') == protocol['data_contract_sha256']
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data = np.load(ROOT / 'data_contract.npz')
    uu, ii, yy = [torch.from_numpy(data['train_' + c].astype('int64' if c != 'y' else 'float32')) for c in ['u', 'i', 'y']]
    train_n = len(uu)
    records, epoch_rows = [], []
    models_dir = ROOT / 'models'
    models_dir.mkdir(exist_ok=True)
    for seed in SEEDS:
        for method in METHODS:
            model_started = time.perf_counter()
            model = MF(method, seed, data, protocol['group_budget'])
            optimizer = torch.optim.Adam(model.parameters(), lr=.03)
            best_loss, progress_loss, stale = float('inf'), float('inf'), 0
            best, best_epoch = None, None
            model_history = []
            for epoch in range(1, 26):
                epoch_started = time.perf_counter()
                order = np.random.default_rng(seed + 100003 * epoch).permutation(train_n)
                total_loss = 0.
                for start in range(0, train_n, 32768):
                    idx = torch.from_numpy(order[start:start + 32768])
                    optimizer.zero_grad(set_to_none=True)
                    score, penalty = model(uu[idx], ii[idx])
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(score, yy[idx])
                    objective = bce + penalty
                    if not bool(torch.isfinite(objective)):
                        raise RuntimeError('Nonfinite training objective')
                    objective.backward()
                    optimizer.step()
                    total_loss += float(bce.detach()) * len(idx)
                arrays = model.arrays()
                vp = score_numpy(arrays, data['validation_u'], data['validation_i'])
                vl = float(np.nanmean(user_means(loss(data['validation_y'], vp), data['validation_u'], len(data['raw_users']))))
                row = {'method': method, 'seed': seed, 'epoch': epoch, 'training_event_bce': total_loss / train_n, 'validation_macro_loss': vl, 'epoch_seconds': time.perf_counter() - epoch_started}
                model_history.append(row); epoch_rows.append(row)
                if vl < best_loss:
                    best_loss, best, best_epoch = vl, arrays, epoch
                if vl < progress_loss - 1e-5:
                    progress_loss, stale = vl, 0
                else:
                    stale += 1
                if epoch == 1 and seed == SEEDS[0] and method == 'full':
                    save('throughput_receipt.json', {'first_full_epoch_seconds': row['epoch_seconds'], 'training_rows': train_n, 'batch_size': 32768, 'device': 'CPU', 'does_not_predict_all_method_runtimes': True})
                    print(json.dumps({'first_epoch': row}), flush=True)
                if stale >= 4:
                    break
            name = f'{method}_s{seed}'
            path = models_dir / (name + '.npz')
            np.savez_compressed(path, **best)
            parameter_bytes = sum(v.nbytes for k, v in best.items() if k not in ['mapping', 'method'])
            inference_bytes = parameter_bytes + best['mapping'].nbytes + data['user_lookup'].nbytes + data['item_lookup'].nbytes
            record = {'name': name, 'method': method, 'seed': seed, 'best_epoch': best_epoch, 'epochs_run': len(model_history), 'validation_macro_loss': best_loss, 'parameter_bytes': parameter_bytes, 'mapping_and_lookup_bytes': inference_bytes - parameter_bytes, 'inference_array_bytes': inference_bytes, 'model_sha256': sha(path), 'elapsed_seconds': time.perf_counter() - model_started, 'last_epoch_was_best': best_epoch == len(model_history), 'stopped_by': 'patience' if stale >= 4 else 'epoch_cap', 'history': model_history}
            records.append(record)
            save('fit_progress.json', {'completed': records, 'elapsed_seconds': time.perf_counter() - started})
            print(json.dumps({'fitted': name, 'best_epoch': best_epoch, 'epochs_run': len(model_history), 'validation_macro_loss': best_loss, 'seconds': record['elapsed_seconds'], 'inference_bytes': inference_bytes}), flush=True)
    csv_save('training_history.csv', epoch_rows)
    save('fit_manifest.json', {'frozen_utc': utc(), 'protocol_sha256': sha(ROOT / 'protocol.json'), 'code_hashes': {f: sha(ROOT / f) for f in ['run_pilot.py', 'exposure_math.py']}, 'models': records, 'fit_seconds': time.perf_counter() - started, 'test_outcomes_loaded_in_fit': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'fit'])
    args = parser.parse_args()
    globals()[args.stage]()
