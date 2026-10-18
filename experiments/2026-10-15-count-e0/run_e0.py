"""Throwaway count-level E0: parameter sharing under empirical context reweighting.

Stages are separate so the protocol and fitted model manifest precede evaluation.
No neural model, graph baseline, real deployment intervention, or novelty proof.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

# Bound only this child's numerical worker pools; no persistent environment change.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import numpy as np

from e0_math import (
    bernoulli_kl, capped_assignment, context_mix, distortion,
    fitted_group_table, reweighted_loss, tilted_cost,
)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / '2026-10-15-kuairand-support'
C = 15
ALPHA = 10.
BUDGETS = [64, 256, 1024]
LAMBDAS = [.1, .3, 1., 3.]
SEEDS = [11, 29, 47]
GAMMAS = [0., .25, .5, .75, 1.]


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save(name, value):
    (ROOT / name).write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding='utf-8')


def write_csv(name, rows):
    if not rows:
        return
    with (ROOT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(clean(rows))


def load_split(cache, name):
    return {col: cache[name + '__' + col] for col in ['user_id', 'video_id', 'tab', 'long_view']}


def logit(p):
    p = np.clip(np.asarray(p, dtype='float64'), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    return (1 / (1 + np.exp(-np.clip(z, -25, 25)))).astype('float32')


def loss(y, p):
    p = np.clip(np.asarray(p, dtype='float64'), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def tabulate(ids, tabs, y, size):
    key = ids.astype('int64') * C + tabs
    n = np.bincount(key, minlength=size * C).reshape(size, C).astype('float64')
    s = np.bincount(key, weights=y, minlength=size * C).reshape(size, C)
    return n, s


def training_tables(train):
    nusers = int(train['user_id'].max()) + 1
    nitems = int(train['video_id'].max()) + 1
    un, us = tabulate(train['user_id'], train['tab'], train['long_view'], nusers)
    inn, ins = tabulate(train['video_id'], train['tab'], train['long_view'], nitems)
    cn, cs = un.sum(axis=0), us.sum(axis=0)
    global_p = (train['long_view'].sum() + .5) / (len(train['long_view']) + 1)
    cp = ((cs + ALPHA * global_p) / (cn + ALPHA)).astype('float32')
    up = ((us + ALPHA * cp) / (un + ALPHA)).astype('float32')
    ip = ((ins + ALPHA * cp) / (inn + ALPHA)).astype('float32')
    active = np.flatnonzero(un.sum(axis=1) > 0)
    supported = (un >= 10).sum(axis=1) >= 2
    profiles = up[active].astype('float64')
    profiles[un[active] < 10] = np.broadcast_to(cp, profiles.shape)[un[active] < 10]
    return {'un': un, 'us': us, 'cn': cn, 'cp': cp, 'up': up, 'ip': ip,
            'active': active, 'supported': supported, 'profiles': profiles,
            'p': cn / cn.sum(), 'global_p': float(global_p)}


def prepare():
    if (ROOT / 'protocol.json').exists():
        raise RuntimeError('Protocol already exists; do not silently overwrite a recorded experiment')
    source_verification = json.loads((SOURCE / 'verification.json').read_text(encoding='utf-8'))
    cache = np.load(SOURCE / 'audit_arrays.npz')
    tables = training_tables(load_split(cache, 'train'))
    eligible = (tables['cn'] >= 1000) & ((tables['un'] >= 10).sum(axis=0) >= 30)
    context_rows = []
    for s in range(C):
        row = {'tab': s, 'train_users_with_at_least_10': int((tables['un'][:, s] >= 10).sum()),
               'train_qualified_context': bool(eligible[s])}
        for split in ['train', 'validation', 'test_standard', 'test_random']:
            tabs = cache[split + '__tab']
            n = int((tabs == s).sum())
            row[split + '_rows'] = n
            row[split + '_fraction'] = n / len(tabs)
        context_rows.append(row)
    write_csv('context_frequencies.csv', context_rows)
    protocol = {
        'recorded_utc': utc(), 'status': 'exploratory_count_level_E0_not_confirmatory',
        'authorization': 'User supplied COUNT-LEVEL E0 spec in attached request; local CPU probe only.',
        'source_cache': str(SOURCE / 'audit_arrays.npz'), 'source_cache_sha256': sha(SOURCE / 'audit_arrays.npz'),
        'inherited_protocol_sha256': sha(SOURCE / 'protocol.json'),
        'inherited_audit_sha256': sha(SOURCE / 'support_audit.json'),
        'inherited_verification_status': source_verification.get('status'),
        'inherited_label_rule_consistency': False,
        'data_contract': 'Existing cleaned UTC splits; native long_view; duplicate-key quarantine unchanged. Standard test already used for exploration.',
        'features': ['user_id', 'video_id', 'tab'], 'target': 'native long_view',
        'budgets': BUDGETS, 'random_seeds': SEEDS, 'behavioral_initialization_seed': 11,
        'temperatures': LAMBDAS, 'alpha': ALPHA, 'user_effect_weight': 1.,
        'hyperparameters': 'All fixed before this evaluation; use earlier validation-selected alpha=10 and individual weight=1 for every grouped and ungrouped model. No per-model retuning.',
        'prediction': 'sigmoid(logit(item_context_probability) + logit(group_context_probability) - logit(context_probability)); counts from train only.',
        'profile': 'r_us=(positive_us+alpha*context_prior_s)/(n_us+alpha), replaced by context_prior_s if n_us<10; delta=logit(r_us)-logit(context_prior_s). Clustering uses reconstructed Bernoulli r_us, not a debiased preference.',
        'frequency_objective': 'sum_g (|g|/N) sum_s p_train(s) c_g(s)',
        'tilted_objective': 'sum_g (|g|/N) lambda log(sum_s p_train(s) exp(c_g(s)/lambda))',
        'dispersion': 'c_g(s)=mean_{u in g} KL(Bernoulli(r_us)||Bernoulli(mean_{v in g} r_vs)); users weighted uniformly in grouping.',
        'optimization': 'Common balanced random initial partition for frequency and all tilted variants. Generalized Lloyd first-order assignment, arithmetic Bernoulli centroids, capped repair, exact objective acceptance, partial-step fallback. Max 12 accepted rounds; heuristic, no global-optimum guarantee.',
        'capacity': 'Every group nonempty, at most ceil(2*N_train_users/B) users; random/marginal partitions also satisfy this common cap.',
        'marginal': 'Balanced quantile bins on smoothed user-marginal residual, random stable tie order seed=11; one-dimensional similarity control, not a graph method.',
        'storage': 'Identical dense B*15 float32 group probabilities and fixed-length int16 user-to-group map at each budget; common item/context tables shared equally; report actual uncompressed array bytes, not zip sizes.',
        'unknown_users': 'Residual zero for users absent from train; never hash unseen users into a learned group.',
        'cohorts': 'All evaluation users; separately training-defined users with >=10 train records in each of >=2 contexts.',
        'sweeps': 'q_gamma proportional to p_train^(1-gamma), gamma=0,.25,.5,.75,1; targeted mix toward rarest train context. Both requested all-context sweeps and secondary train-qualified-context sweeps.',
        'train_qualified_context_rule': '>=1000 train events and >=30 users with >=10 train events in that context; selected using train only.',
        'train_qualified_contexts': np.flatnonzero(eligible).tolist(),
        'rarest_all_context': int(np.argmin(tables['p'])),
        'rarest_qualified_context': int(np.flatnonzero(eligible)[np.argmin(tables['p'][eligible])]),
        'evaluation': 'Standard traffic only. Hold observed P(user,item,label|context) fixed and reweight context means; this is an empirical distribution stress test, not an intervention or identified deployment shift.',
        'estimands': 'Primary event-weighted mean within each context, then q-weight across contexts; secondary macro-user-within-context; raw observed-mixture loss also saved. No renormalization over missing contexts.',
        'uncertainty': '1000 paired user-cluster bootstrap resamples, fixed models and q; pointwise 95% intervals. Missing-context replicates excluded and valid replicate counts recorded. No time-generalization guarantee.',
        'primary_gate': {'budget': 256, 'lambda': .3, 'cohort': 'all', 'sweep': 'all_flatten',
            'conditions': ['frequency penalty(1)-penalty(0)>0', 'frequency growth minus tilted growth>0', 'tilted-minus-frequency loss at gamma=0 <= .001'],
            'inference': 'Bonferroni-adjusted one-sided bounds across the three fixed primary conditions, family alpha=.05. Noninferiority margin .001 log loss is an analyst choice, not established practical significance.'},
        'secondary_gates': 'All other budgets, lambdas and sweeps exploratory; no GO selected by scanning them.',
        'graphhash': 'Optional author GraphHash baseline omitted from this count-only probe; source README is pinned in earlier audit, no adapted graph grouping was executed. No GraphHash superiority claim.',
        'recalibration': 'Optional random-set recalibration not included; no calibration explanation is established by this run.',
        'limitations': ['Support shrinkage can eliminate genuine low-support differences; it does not prove absence of false disagreement.',
                        'Uniform-user grouping and event-weighted prediction are different objectives.',
                        'Count grouping is a diagnostic sharing model, not a neural embedding-compression reproduction.',
                        'Single optimization initialization for nonrandom criteria; random assignment seeds are not training replications.',
                        'Repeated exploratory use of test data; no confirmatory paper claim.'],
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'device': 'CPU'},
    }
    save('protocol.json', protocol)
    print(json.dumps({'stage': 'prepared', 'active_train_users': len(tables['active']), 'supported_train_users': int(tables['supported'].sum()), 'qualified_contexts': protocol['train_qualified_contexts'], 'source_cache_sha256': protocol['source_cache_sha256']}), flush=True)


def assignment_costs(profiles, centers, weights, group_bias):
    # KL(q||m)=Hneg(q)-q log(m)-(1-q)log(1-m).
    entropy = profiles * np.log(profiles) + (1 - profiles) * np.log1p(-profiles)
    cross_a = weights * (np.log1p(-centers) - np.log(centers))
    cross_b = -(weights * np.log1p(-centers)).sum(axis=1) + group_bias
    result = np.empty((len(profiles), len(centers)), dtype='float32')
    for start in range(0, len(profiles), 4096):
        stop = min(start + 4096, len(profiles))
        result[start:stop] = entropy[start:stop] @ weights.T + profiles[start:stop] @ cross_a.T + cross_b
    return result


def refine(profiles, initial, budget, p, temperature, cap):
    groups = initial.copy()
    objective, dispersion, centers = distortion(profiles, groups, budget, p, temperature)
    history = [{'iteration': 0, 'objective': objective, 'moved_users': 0, 'step_fraction': 0.}]
    rng = np.random.default_rng(611)
    for iteration in range(1, 13):
        if temperature is None:
            weights = np.broadcast_to(p, centers.shape)
        else:
            z = dispersion / temperature
            z -= z.max(axis=1, keepdims=True)
            weights = p * np.exp(z)
            weights /= weights.sum(axis=1, keepdims=True)
        bias = tilted_cost(dispersion, p, temperature) - (weights * dispersion).sum(axis=1)
        costs = assignment_costs(profiles, centers, weights, bias)
        candidate = capped_assignment(costs, cap)
        moved = candidate != groups
        if not moved.any():
            break
        accepted = False
        order = rng.random(len(groups))
        for fraction in [1., .5, .25, .125]:
            proposed = candidate.copy() if fraction == 1 else np.where(moved & (order < fraction), candidate, groups)
            sizes = np.bincount(proposed, minlength=budget)
            if np.any(sizes == 0) or np.any(sizes > cap):
                continue
            score, d, m = distortion(profiles, proposed, budget, p, temperature)
            if score < objective - 1e-12:
                previous = objective
                actual_moved = int((proposed != groups).sum())
                groups, objective, dispersion, centers = proposed, score, d, m
                history.append({'iteration': iteration, 'objective': objective, 'moved_users': actual_moved, 'step_fraction': fraction})
                accepted = True
                break
        if not accepted or previous - objective < 1e-10:
            break
    return groups, history


def predict(data, model, common):
    u, i, c = data['user_id'], data['video_id'], data['tab']
    cp = common['context'][c]
    ip = cp.copy()
    valid_i = (i >= 0) & (i < len(common['item']))
    ip[valid_i] = common['item'][i[valid_i], c[valid_i]]
    kind = str(model['kind'].item())
    if kind == 'item_only':
        return ip
    if kind == 'global':
        return np.full(len(u), float(common['global'][0]), dtype='float32')
    residual = np.zeros(len(u), dtype='float64')
    if kind == 'ungrouped':
        valid_u = (u >= 0) & (u < len(model['user']))
        residual[valid_u] = logit(model['user'][u[valid_u], c[valid_u]]) - logit(cp[valid_u])
    else:
        mapped = np.full(len(u), -1, dtype='int32')
        valid_u = (u >= 0) & (u < len(model['mapping']))
        mapped[valid_u] = model['mapping'][u[valid_u]]
        valid_u &= mapped >= 0
        residual[valid_u] = logit(model['group'][mapped[valid_u], c[valid_u]]) - logit(cp[valid_u])
    return sigmoid(logit(ip) + residual)


def fit():
    started = time.perf_counter()
    if (ROOT / 'fit_manifest.json').exists():
        raise RuntimeError('Fit manifest already exists; refusing overwrite')
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    assert sha(SOURCE / 'audit_arrays.npz') == protocol['source_cache_sha256']
    cache = np.load(SOURCE / 'audit_arrays.npz')
    train, val = load_split(cache, 'train'), load_split(cache, 'validation')
    t = training_tables(train)
    (ROOT / 'models').mkdir(exist_ok=True)
    np.savez_compressed(ROOT / 'common.npz', context=t['cp'], item=t['ip'], global_=np.asarray([t['global_p']], dtype='float32'))
    common = {'context': t['cp'], 'item': t['ip'], 'global': np.asarray([t['global_p']], dtype='float32')}
    np.savez_compressed(ROOT / 'training_profiles.npz', active=t['active'], profile=t['profiles'], supported=t['supported'], train_counts=t['un'], p=t['p'])
    common_bytes = sum(a.nbytes for a in common.values())
    models = []

    def add_model(name, kind, budget=None, criterion=None, temperature=None, seed=None, groups=None, history=None):
        arrays = {'kind': np.asarray(kind)}
        if kind == 'ungrouped':
            arrays['user'] = t['up']
        elif kind == 'grouped':
            mapping = np.full(len(t['un']), -1, dtype='int16')
            mapping[t['active']] = groups.astype('int16')
            arrays['mapping'] = mapping
            arrays['group'] = fitted_group_table(t['un'][t['active']], t['us'][t['active']], groups, budget, t['cp'], ALPHA).astype('float32')
        path = ROOT / 'models' / (name + '.npz')
        np.savez_compressed(path, **arrays)
        vp = predict(val, arrays, common)
        size = sum(a.nbytes for k, a in arrays.items() if k != 'kind')
        row = {'name': name, 'kind': kind, 'budget': budget, 'criterion': criterion, 'lambda': temperature, 'seed': seed,
               'validation_log_loss': float(loss(val['long_view'], vp).mean()),
               'varying_array_bytes': int(size), 'common_array_bytes': int(common_bytes),
               'total_array_bytes': int(size + common_bytes), 'model_sha256': sha(path)}
        if groups is not None:
            sizes = np.bincount(groups, minlength=budget)
            row.update({'group_min_users': int(sizes.min()), 'group_max_users': int(sizes.max()),
                        'group_cap': int(math.ceil(2 * len(groups) / budget)), 'history': history})
        models.append(row)
        print(json.dumps({'fitted': name, 'validation_log_loss': row['validation_log_loss'], 'seconds': time.perf_counter() - started}), flush=True)

    add_model('ungrouped', 'ungrouped')
    add_model('item_only', 'item_only')
    add_model('global', 'global')
    for budget in BUDGETS:
        cap = int(math.ceil(2 * len(t['active']) / budget))
        initial = None
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            permutation = rng.permutation(len(t['active']))
            groups = np.empty(len(permutation), dtype='int32')
            groups[permutation] = np.arange(len(permutation)) % budget
            if seed == SEEDS[0]:
                initial = groups.copy()
            add_model(f'random_B{budget}_s{seed}', 'grouped', budget, 'random', seed=seed, groups=groups)
        mn, ms = t['un'][t['active']].sum(axis=1), t['us'][t['active']].sum(axis=1)
        marginal = logit((ms + ALPHA * t['global_p']) / (mn + ALPHA)) - logit(t['global_p'])
        tie_order = np.random.default_rng(11).permutation(len(marginal))
        order = tie_order[np.argsort(marginal[tie_order], kind='stable')]
        marginal_groups = np.empty(len(order), dtype='int32')
        marginal_groups[order] = np.minimum(np.arange(len(order)) * budget // len(order), budget - 1)
        add_model(f'marginal_B{budget}', 'grouped', budget, 'marginal_quantile', groups=marginal_groups)
        for temperature in [None] + LAMBDAS:
            groups, history = refine(t['profiles'], initial, budget, t['p'], temperature, cap)
            name = f'frequency_B{budget}' if temperature is None else f'tilted_B{budget}_L{temperature:g}'
            add_model(name, 'grouped', budget, 'frequency' if temperature is None else 'tilted', temperature, 11, groups, history)
    manifest = {'frozen_utc': utc(), 'protocol_sha256': sha(ROOT / 'protocol.json'),
                'sources': {f: sha(ROOT / f) for f in ['run_e0.py', 'e0_math.py']},
                'common_sha256': sha(ROOT / 'common.npz'), 'models': models,
                'elapsed_seconds': time.perf_counter() - started,
                'test_labels_loaded_in_fit_stage': False, 'selection': 'No selection; fixed primary B=256,lambda=.3; full requested grid retained.'}
    save('fit_manifest.json', manifest)


def make_sweeps(p, protocol):
    allowed = np.asarray(protocol['train_qualified_contexts'])
    pp = np.zeros_like(p)
    pp[allowed] = p[allowed]
    pp /= pp.sum()
    return {
        'all_flatten': [context_mix(p, g) for g in GAMMAS],
        'all_target_rarest': [context_mix(p, g, protocol['rarest_all_context']) for g in GAMMAS],
        'qualified_flatten': [context_mix(pp, g) for g in GAMMAS],
        'qualified_target_rarest': [context_mix(pp, g, protocol['rarest_qualified_context']) for g in GAMMAS],
    }


def interval(a, alpha=.05):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    if not len(a):
        return [None, None]
    return [float(x) for x in np.quantile(a, [alpha / 2, 1 - alpha / 2])]


def bootstrap_context(loss_sums, counts, weights):
    denominator = weights @ counts
    numerator = weights @ loss_sums
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def q_bootstrap(by_context, q):
    active = q > 0
    return by_context[:, active] @ q[active]


def evaluate():
    started = time.perf_counter()
    if (ROOT / 'results.json').exists():
        raise RuntimeError('Results already exist; refusing overwrite')
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    manifest = json.loads((ROOT / 'fit_manifest.json').read_text())
    assert manifest['protocol_sha256'] == sha(ROOT / 'protocol.json')
    for f, expected in manifest['sources'].items():
        assert sha(ROOT / f) == expected
    assert sha(SOURCE / 'audit_arrays.npz') == protocol['source_cache_sha256']
    save('evaluation_started.json', {'utc': utc(), 'fit_manifest_sha256': sha(ROOT / 'fit_manifest.json'), 'protocol_sha256': sha(ROOT / 'protocol.json')})
    # First evaluation-label access in this stage occurs below the frozen receipt.
    cache = np.load(SOURCE / 'audit_arrays.npz')
    data = load_split(cache, 'test_standard')
    profiles = np.load(ROOT / 'training_profiles.npz')
    cp = np.load(ROOT / 'common.npz')
    common = {'context': cp['context'], 'item': cp['item'], 'global': cp['global_']}
    nusers = max(len(profiles['supported']), int(data['user_id'].max()) + 1)
    event_key = data['user_id'].astype('int64') * C + data['tab']
    counts = np.bincount(event_key, minlength=nusers * C).reshape(nusers, C).astype('float64')
    supported = np.zeros(nusers, dtype=bool)
    supported[:len(profiles['supported'])] = profiles['supported']
    masks = {'all': counts.sum(axis=1) > 0, 'supported': supported & (counts.sum(axis=1) > 0)}
    sweeps = make_sweeps(profiles['p'], protocol)
    predictions, aggregates, raw_rows, context_rows = {}, {}, [], []
    for row in manifest['models']:
        name = row['name']
        assert sha(ROOT / 'models' / (name + '.npz')) == row['model_sha256']
        model = np.load(ROOT / 'models' / (name + '.npz'))
        p = predict(data, model, common)
        predictions[name] = p
        event_loss = loss(data['long_view'], p)
        sums = np.bincount(event_key, weights=event_loss, minlength=nusers * C).reshape(nusers, C)
        aggregates[name] = sums
        for cohort, user_mask in masks.items():
            event_mask = user_mask[data['user_id']]
            raw_rows.append({'model': name, 'cohort': cohort, 'rows': int(event_mask.sum()),
                             'users': int(user_mask.sum()), 'log_loss': float(event_loss[event_mask].mean()),
                             'macro_user_log_loss': float((sums[user_mask].sum(axis=1) / counts[user_mask].sum(axis=1)).mean()),
                             'total_array_bytes': row['total_array_bytes']})
            for c in range(C):
                cn = counts[user_mask, c]
                ss = sums[user_mask, c]
                nonzero = cn > 0
                context_rows.append({'model': name, 'cohort': cohort, 'tab': c, 'rows': int(cn.sum()),
                                     'users': int(nonzero.sum()),
                                     'log_loss': float(ss.sum() / cn.sum()) if cn.sum() else None,
                                     'macro_user_log_loss': float((ss[nonzero] / cn[nonzero]).mean()) if nonzero.any() else None})
    np.savez_compressed(ROOT / 'predictions_standard.npz', **predictions)
    np.savez_compressed(ROOT / 'per_user_context_loss_sums.npz', counts=counts, supported=supported, **aggregates)
    write_csv('raw_metrics.csv', raw_rows)
    write_csv('per_context_metrics.csv', context_rows)
    curve_rows, gates = [], []
    bootstrap_saved = {}
    for cohort, mask in masks.items():
        n, ref_sums = counts[mask], aggregates['ungrouped'][mask]
        users = int(mask.sum())
        rng = np.random.default_rng(20261015)
        weights = np.empty((1000, users), dtype='float64')
        for b in range(len(weights)):
            weights[b] = np.bincount(rng.integers(0, users, size=users), minlength=users)
        denominator = weights @ n
        ref_boot_num = weights @ ref_sums
        ref_boot = np.divide(ref_boot_num, denominator, out=np.full_like(ref_boot_num, np.nan), where=denominator > 0)
        ref_ctx = np.divide(ref_sums.sum(axis=0), n.sum(axis=0), out=np.full(C, np.nan), where=n.sum(axis=0) > 0)
        ctx_means, boots = {}, {}
        for row in manifest['models']:
            name = row['name']
            sm = aggregates[name][mask]
            mean = np.divide(sm.sum(axis=0), n.sum(axis=0), out=np.full(C, np.nan), where=n.sum(axis=0) > 0)
            # Bootstrap paired differences directly, reducing cancellation error.
            numerator = weights @ (sm - ref_sums)
            delta_boot = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
            ctx_means[name], boots[name] = mean, delta_boot
            bootstrap_saved[cohort + '__' + name] = delta_boot
            for sweep, qs in sweeps.items():
                for gamma, q in zip(GAMMAS, qs):
                    point = reweighted_loss(mean, q)
                    delta = reweighted_loss(mean - ref_ctx, q)
                    boot = q_bootstrap(delta_boot, q)
                    macro_u = np.divide(sm, n, out=np.full_like(sm, np.nan), where=n > 0)
                    macro_ctx = np.array([float(np.nanmean(macro_u[:, c])) if np.any(n[:, c] > 0) else np.nan for c in range(C)])
                    ci = interval(boot)
                    curve_rows.append({'model': name, 'cohort': cohort, 'sweep': sweep, 'gamma': gamma,
                                       'loss': point, 'sharing_penalty': delta, 'delta_ci_low': ci[0], 'delta_ci_high': ci[1],
                                       'valid_bootstrap_replicates': int(np.isfinite(boot).sum()),
                                       'macro_user_within_context_loss': reweighted_loss(macro_ctx, q),
                                       'missing_positive_weight_contexts': ','.join(map(str, np.flatnonzero((q > 0) & (n.sum(axis=0) == 0))))})
        for budget in BUDGETS:
            freq = f'frequency_B{budget}'
            for temperature in LAMBDAS:
                tilted = f'tilted_B{budget}_L{temperature:g}'
                for sweep, qs in sweeps.items():
                    dq = qs[-1] - qs[0]
                    f_growth = reweighted_loss(ctx_means[freq] - ref_ctx, dq)
                    # dq is signed; direct dot products require all nonzero contexts finite.
                    active = dq != 0
                    f_growth = float((ctx_means[freq] - ref_ctx)[active] @ dq[active]) if np.isfinite((ctx_means[freq] - ref_ctx)[active]).all() else None
                    t_growth = float((ctx_means[tilted] - ref_ctx)[active] @ dq[active]) if np.isfinite((ctx_means[tilted] - ref_ctx)[active]).all() else None
                    fg_boot = boots[freq][:, active] @ dq[active]
                    tg_boot = boots[tilted][:, active] @ dq[active]
                    advantage_boot = fg_boot - tg_boot
                    in_mix_boot = q_bootstrap(boots[tilted] - boots[freq], qs[0])
                    mix_diff = reweighted_loss(ctx_means[tilted] - ctx_means[freq], qs[0])
                    alpha_one_sided = .05 / 3
                    def bound(a, q):
                        a = a[np.isfinite(a)]
                        return float(np.quantile(a, q)) if len(a) else None
                    lower_growth = bound(fg_boot, alpha_one_sided)
                    lower_advantage = bound(advantage_boot, alpha_one_sided)
                    upper_in_mix = bound(in_mix_boot, 1 - alpha_one_sided)
                    conditions = [lower_growth is not None and lower_growth > 0,
                                  lower_advantage is not None and lower_advantage > 0,
                                  upper_in_mix is not None and upper_in_mix <= .001]
                    primary = cohort == 'all' and budget == 256 and temperature == .3 and sweep == 'all_flatten'
                    gates.append({'cohort': cohort, 'budget': budget, 'lambda': temperature, 'sweep': sweep,
                                  'primary': primary, 'frequency_penalty_growth': f_growth, 'tilted_penalty_growth': t_growth,
                                  'growth_advantage': f_growth - t_growth if f_growth is not None and t_growth is not None else None,
                                  'frequency_growth_95_ci': interval(fg_boot), 'growth_advantage_95_ci': interval(advantage_boot),
                                  'tilted_minus_frequency_at_gamma0': mix_diff, 'in_mix_difference_95_ci': interval(in_mix_boot),
                                  'one_sided_adjusted_lower_frequency_growth': lower_growth,
                                  'one_sided_adjusted_lower_growth_advantage': lower_advantage,
                                  'one_sided_adjusted_upper_in_mix_difference': upper_in_mix,
                                  'conditions_pass': conditions, 'gate_pass': all(conditions),
                                  'valid_growth_bootstrap_replicates': int(np.isfinite(fg_boot).sum()),
                                  'scope': 'fixed primary gate' if primary else 'exploratory grid diagnostic; not multiplicity-adjusted across grid'})
        print(json.dumps({'evaluated_cohort': cohort, 'users': users, 'rows': int(n.sum())}), flush=True)
    write_csv('sweep_metrics.csv', curve_rows)
    np.savez_compressed(ROOT / 'bootstrap_context_deltas.npz', **bootstrap_saved)
    save('gates.json', gates)
    primary = next(g for g in gates if g['primary'])
    result = {'completed_utc': utc(), 'status': 'exploratory_count_level_E0_executed',
              'fit_manifest_sha256': sha(ROOT / 'fit_manifest.json'), 'protocol_sha256': sha(ROOT / 'protocol.json'),
              'models': len(manifest['models']), 'grouped_models': sum(m['kind'] == 'grouped' for m in manifest['models']),
              'test_rows': len(data['long_view']), 'cohorts': {k: {'users': int(v.sum()), 'rows': int(counts[v].sum())} for k, v in masks.items()},
              'sweeps': {k: [q.tolist() for q in v] for k, v in sweeps.items()},
              'primary_gate': primary, 'fit_seconds': manifest['elapsed_seconds'],
              'evaluation_seconds': time.perf_counter() - started,
              'graphhash_executed': False, 'neural_model_executed': False,
              'scope': 'Frozen count-sharing models under empirical context reweighting. No actual deployment shift, causal compression damage claim, neural superiority, or novelty validation.'}
    save('results.json', result)
    print(json.dumps(clean(result)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'fit', 'evaluate'])
    args = parser.parse_args()
    globals()[args.stage]()
