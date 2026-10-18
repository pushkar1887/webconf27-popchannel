"""Throwaway, validation-only mechanism diagnostic. No neural or deployment claim."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / '2026-10-15-kuairand-support'
E0 = ROOT.parent / '2026-10-15-count-e0'
sys.path.insert(0, str(E0))
import run_e0 as old
from e0_math import capped_assignment, distortion, tilted_cost
from mechanism_math import pairing_indices, select_pairings, fixed_effect_fit

SEEDS = [11, 29, 47]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sanitize(x):
    if isinstance(x, dict):
        return {str(k): sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sanitize(v) for v in x]
    if isinstance(x, np.ndarray):
        return sanitize(x.tolist())
    if isinstance(x, np.generic):
        return sanitize(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def save(name, value):
    (ROOT / name).write_text(json.dumps(sanitize(value), indent=2, allow_nan=False), encoding='utf-8')


def write_csv(name, rows):
    if not rows:
        return
    with (ROOT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sanitize(rows))


def load_train():
    cache = np.load(SOURCE / 'audit_arrays.npz')
    return old.load_split(cache, 'train')


def prepare():
    if (ROOT / 'protocol.json').exists():
        raise RuntimeError('Recorded protocol exists; no overwrite')
    train = load_train()
    t = old.training_tables(train)
    supported_pairs = [(int(((t['un'][:, a] >= 20) & (t['un'][:, b] >= 20)).sum()), a, b)
                       for a in range(15) for b in range(a + 1, 15)]
    support, a, b = sorted(supported_pairs, key=lambda x: (-x[0], x[1], x[2]))[0]
    protocol = {
        'recorded_utc': now(), 'status': 'exploratory_validation_only_research_spike',
        'authorization': 'User: okay do it, after the bounded mechanism-check plan.',
        'resource_scope': 'Local CPU only; two-hour investigation cap, no dependency changes, downloads, external writes, GPU, Kaggle or NVIDIA calls.',
        'source_cache_sha256': sha(SOURCE / 'audit_arrays.npz'),
        'previous_E0_results_sha256': sha(E0 / 'results.json'),
        'reused_sources': {str(E0 / f): sha(E0 / f) for f in ['run_e0.py', 'e0_math.py']},
        'native_label_rule_consistency': False,
        'data': 'Existing quarantine and UTC chronology; native long_view. Only train and validation arrays accessed; existing standard/random tests not evaluated.',
        'context_pair': [a, b], 'pair_selection': 'Maximize count of training users with >=20 records in both interfaces; stable lexical tie break; no outcomes used.',
        'users_with_20_in_both_before_split': support,
        'training_split': 'Seed 20261016 random disjoint halves A/B at event level, probability .5. A defines profiles and pairings. B fits all paired predictors and checks profile repeatability. Not a claim of statistical independence of temporally correlated events.',
        'eligible_users': 'At least 20 records per selected context in full train and at least 8 in each context in each of A and B. Counts only.',
        'profile': 'Alpha=10 smoothing toward A context rate. Contrast k_u=logit(p_A(u,a))-logit(p_A(u,b)); marginal component m_u=.5*(logit(p_A(u,a))+logit(p_A(u,b))). Overall user logit uses all A contexts.',
        'matching': 'Greedy blocks of four users, seeds 11/29/47. Every block: max-min overall A user logit <=.5, max-min marginal component <=.5, full-train counts in each selected context differ by <=2x. Nearest standardized covariates; all selection uses A outcomes and train counts.',
        'pairing': 'Three exhaustive perfect pairings in each block. Select low/high using A mean squared pair contrast difference, stable first tie break. Neutral reference is expected loss across the three uniform random pairings, not an ensemble probability.',
        'fit': 'B counts only; alpha=10 and residual weight=1 for every predictor. Common B video/context probabilities. Pair tables use all 15 contexts; evaluate selected two contexts only. Two users per group; identical mapping/table bytes for high, low and all three random-pairing outcomes.',
        'evaluation': 'Validation only, matched users and selected context pair. Primary per-event log loss on this fixed cohort; secondary equal-context mean. This is not pure causal preference identification.',
        'prediction': 'sigmoid(logit(p_B(video,context)) + logit(p_B(group,context)) - logit(p_B(context)))',
        'references': ['individual B user/context counts', 'three exhaustive uniform-random pairing outcomes'],
        'adjustment': 'All six possible pairs per block; weighted within-block linear regression of validation excess loss on A squared contextual contrast, controlling squared marginal and overall-logit differences, activity mismatch, mean activity, mean marginal component, and A video-set Jaccard overlap. Weights=validation events. Descriptive association, not a causal estimator.',
        'uncertainty': '1000 paired block-cluster bootstrap replicates, seed 20261016; user-cluster sensitivity for predictor contrasts. Models fixed. No time-generalization or confirmatory claim.',
        'primary_seed': 11,
        'primary_gate': {
            'support': '>=50 matched blocks and >=1000 validation events',
            'reliable_context_difference': 'High-minus-low B squared contrast difference has positive adjusted one-sided lower bound.',
            'extra_sharing_damage': 'High-minus-low validation log loss >=.001 and positive adjusted lower bound.',
            'useful_grouping': 'Neutral-minus-low validation log loss >=.001 and positive adjusted lower bound.',
            'adjusted_association': 'Contextual-disagreement coefficient has positive adjusted lower bound after stated controls.',
            'bounds': 'Bonferroni .05/4 one-sided bootstrap bounds for the four effects; .001 log-loss margin is an analyst-set development threshold.',
            'replication': 'At least two of three pairing seeds show positive B repeatability, high-minus-low loss and neutral-minus-low loss; seeds reuse data and are not independent experiments.'},
        'optimizer_check': 'Original full-train profiles, B=64/256, frequency and lambda=.3, three common initializations. Extend same heuristic to max100 rounds; save round12 and final models; validation only. No accepted proposal is heuristic stalling, not certified stationarity.',
        'interpretation': 'GO permits only a narrowed mechanism study and novelty screen. Failure closes this current compression proposal rather than authorizing further tuning. No SRS rescue, paper readiness, neural superiority or novel mathematical claim.',
    }
    save('protocol.json', protocol)
    print(json.dumps({'prepared': True, 'contexts': [a, b], 'eligible_before_halving': support}), flush=True)


def make_blocks(eligible, overall, marginal, full_counts, contexts, seed):
    rng = np.random.default_rng(seed)
    features = np.column_stack([overall[eligible], marginal[eligible], np.log(full_counts[eligible][:, contexts])])
    scaled = (features - features.mean(axis=0)) / np.maximum(features.std(axis=0), 1e-8)
    available = np.ones(len(eligible), dtype=bool)
    blocks = []
    for anchor in rng.permutation(len(eligible)):
        if not available[anchor]:
            continue
        chosen = [anchor]
        for _ in range(3):
            remaining = available.copy()
            remaining[chosen] = False
            low = np.minimum(features[chosen].min(axis=0), features)
            high = np.maximum(features[chosen].max(axis=0), features)
            feasible = remaining & ((high[:, :2] - low[:, :2]) <= .5 + 1e-12).all(axis=1)
            feasible &= ((high[:, 2:] - low[:, 2:]) <= np.log(2) + 1e-12).all(axis=1)
            if not feasible.any():
                break
            distance = ((scaled - scaled[anchor]) ** 2).sum(axis=1)
            chosen.append(int(np.argmin(np.where(feasible, distance, np.inf))))
        if len(chosen) == 4:
            blocks.append(eligible[chosen])
            available[chosen] = False
        else:
            available[anchor] = False
    return np.asarray(blocks, dtype='int32').reshape(-1, 4)


def refine_long(profiles, initial, budget, p, temperature, cap):
    groups = initial.copy()
    objective, dispersion, centers = distortion(profiles, groups, budget, p, temperature)
    history = [{'iteration': 0, 'objective': objective, 'moved': 0}]
    rng = np.random.default_rng(611)
    checkpoint = None
    status = 'iteration_cap'
    for iteration in range(1, 101):
        if temperature is None:
            weights = np.broadcast_to(p, centers.shape)
        else:
            z = dispersion / temperature
            z -= z.max(axis=1, keepdims=True)
            weights = p * np.exp(z)
            weights /= weights.sum(axis=1, keepdims=True)
        bias = tilted_cost(dispersion, p, temperature) - (weights * dispersion).sum(axis=1)
        candidate = capped_assignment(old.assignment_costs(profiles, centers, weights, bias), cap)
        moved = candidate != groups
        if not moved.any():
            status = 'assignments_unchanged'
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
                improvement = objective - score
                actual_moved = int((proposed != groups).sum())
                groups, objective, dispersion, centers = proposed, score, d, m
                history.append({'iteration': iteration, 'objective': objective, 'moved': actual_moved})
                accepted = True
                break
        if iteration == 12:
            checkpoint = groups.copy()
        if not accepted:
            status = 'heuristic_no_accepted_proposal'
            break
        if improvement < 1e-10:
            status = 'objective_tolerance'
            break
    if checkpoint is None:
        checkpoint = groups.copy()
    return groups, checkpoint, history, status


def fit():
    started = time.perf_counter()
    if (ROOT / 'fit_manifest.json').exists():
        raise RuntimeError('Frozen fits exist; no overwrite')
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    assert sha(SOURCE / 'audit_arrays.npz') == protocol['source_cache_sha256']
    train = load_train()
    contexts = np.asarray(protocol['context_pair'])
    split_a = np.random.default_rng(20261016).random(len(train['user_id'])) < .5
    halves = [{k: v[mask] for k, v in train.items()} for mask in [split_a, ~split_a]]
    ta, tb, tf = [old.training_tables(x) for x in halves + [train]]
    # Both halves must use the same full training ID domain.
    for t in [ta, tb]:
        missing = len(tf['un']) - len(t['un'])
        if missing:
            for key in ['un', 'us']:
                t[key] = np.pad(t[key], ((0, missing), (0, 0)))
    full_n, an, bn = tf['un'], ta['un'], tb['un']
    eligible_mask = (full_n[:, contexts] >= 20).all(axis=1)
    eligible_mask &= (an[:, contexts] >= 8).all(axis=1) & (bn[:, contexts] >= 8).all(axis=1)
    eligible = np.flatnonzero(eligible_mask)
    pa = (ta['us'] + 10 * ta['cp']) / (an + 10)
    pb = (tb['us'] + 10 * tb['cp']) / (bn + 10)
    la, lb = old.logit(pa[:, contexts]), old.logit(pb[:, contexts])
    ca, cb = la[:, 0] - la[:, 1], lb[:, 0] - lb[:, 1]
    marginal = la.mean(axis=1)
    overall = old.logit((ta['us'].sum(axis=1) + 10 * ta['global_p']) / (an.sum(axis=1) + 10))
    (ROOT / 'models').mkdir(exist_ok=True)
    np.savez_compressed(ROOT / 'training_halves.npz', split_a=split_a, eligible=eligible, counts_a=an, positives_a=ta['us'], counts_b=bn, positives_b=tb['us'], contrast_a=ca, contrast_b=cb, marginal=marginal, overall=overall)
    np.savez_compressed(ROOT / 'common_B.npz', context=tb['cp'], item=tb['ip'], global_=np.asarray([tb['global_p']], dtype='float32'))
    seeds, model_records = [], []
    individual = {'kind': np.asarray('ungrouped'), 'user': pb.astype('float32')}
    np.savez_compressed(ROOT / 'models' / 'individual_B.npz', **individual)
    model_records.append({'name': 'individual_B', 'kind': 'individual', 'path': 'models/individual_B.npz', 'sha256': sha(ROOT / 'models' / 'individual_B.npz')})
    for seed in SEEDS:
        blocks = make_blocks(eligible, overall, marginal, full_n, contexts, seed)
        if len(blocks) == 0:
            raise RuntimeError('No matched blocks; stop rather than relaxing calipers')
        low, high, score_a = select_pairings(ca[blocks])
        _, _, score_b = select_pairings(cb[blocks])
        np.savez_compressed(ROOT / f'blocks_s{seed}.npz', users=blocks, low=low, high=high, score_a=score_a, score_b=score_b)
        choices = {'matching0': np.zeros(len(blocks), dtype='int32'), 'matching1': np.ones(len(blocks), dtype='int32'), 'matching2': np.full(len(blocks), 2, dtype='int32'), 'low': low, 'high': high}
        budget = 2 * len(blocks)
        for name, choice in choices.items():
            gmap = np.full(len(full_n), -1, dtype='int16')
            for block_idx, users in enumerate(blocks):
                pairs = pairing_indices()[choice[block_idx]]
                gmap[users[pairs[0]]] = 2 * block_idx
                gmap[users[pairs[1]]] = 2 * block_idx + 1
            selected = gmap >= 0
            table = old.fitted_group_table(bn[selected], tb['us'][selected], gmap[selected], budget, tb['cp'], 10).astype('float32')
            file = f'models/s{seed}_{name}.npz'
            np.savez_compressed(ROOT / file, kind=np.asarray('grouped'), mapping=gmap, group=table)
            model_records.append({'name': f's{seed}_{name}', 'kind': name, 'seed': seed, 'path': file, 'sha256': sha(ROOT / file), 'budget': budget, 'varying_array_bytes': int(gmap.nbytes + table.nbytes)})
        row = {'seed': seed, 'blocks': len(blocks), 'users': int(blocks.size), 'budget': budget,
               'mean_A_high_minus_low': float((score_a[np.arange(len(blocks)), high] - score_a[np.arange(len(blocks)), low]).mean()),
               'mean_B_high_minus_low': float((score_b[np.arange(len(blocks)), high] - score_b[np.arange(len(blocks)), low]).mean())}
        seeds.append(row)
        print(json.dumps({'pairing_fit': row}), flush=True)
    np.savez_compressed(ROOT / 'common_full.npz', context=tf['cp'], item=tf['ip'], global_=np.asarray([tf['global_p']], dtype='float32'))
    optimizer_records = []
    for budget in [64, 256]:
        for seed in SEEDS:
            permutation = np.random.default_rng(seed).permutation(len(tf['active']))
            initial = np.empty(len(permutation), dtype='int32')
            initial[permutation] = np.arange(len(permutation)) % budget
            cap = math.ceil(2 * len(initial) / budget)
            for temperature in [None, .3]:
                groups, checkpoint, history, status = refine_long(tf['profiles'], initial, budget, tf['p'], temperature, cap)
                criterion = 'frequency' if temperature is None else 'tilted_L0.3'
                stem = f'optimizer_{criterion}_B{budget}_s{seed}'
                for stage, g in [('round12', checkpoint), ('final', groups)]:
                    mapping = np.full(len(full_n), -1, dtype='int16')
                    mapping[tf['active']] = g.astype('int16')
                    table = old.fitted_group_table(tf['un'][tf['active']], tf['us'][tf['active']], g, budget, tf['cp'], 10).astype('float32')
                    file = f'models/{stem}_{stage}.npz'
                    np.savez_compressed(ROOT / file, kind=np.asarray('grouped'), mapping=mapping, group=table)
                record = {'name': stem, 'budget': budget, 'seed': seed, 'lambda': temperature,
                          'rounds': len(history) - 1, 'termination': status, 'history': history,
                          'round12_model_sha256': sha(ROOT / 'models' / (stem + '_round12.npz')),
                          'final_model_sha256': sha(ROOT / 'models' / (stem + '_final.npz'))}
                optimizer_records.append(record)
                print(json.dumps({'optimizer_fit': stem, 'rounds': record['rounds'], 'termination': status, 'seconds': time.perf_counter() - started}), flush=True)
    manifest = {'frozen_utc': now(), 'protocol_sha256': sha(ROOT / 'protocol.json'),
                'code_hashes': {f: sha(ROOT / f) for f in ['run_mechanism.py', 'mechanism_math.py']},
                'pair_models': model_records, 'pairing_seeds': seeds, 'optimizer_models': optimizer_records,
                'eligible_users': len(eligible), 'validation_labels_accessed_in_fit': False,
                'fit_seconds': time.perf_counter() - started}
    save('fit_manifest.json', manifest)


def ci(array):
    a = np.asarray(array)
    a = a[np.isfinite(a)]
    return np.quantile(a, [.025, .975]).tolist() if len(a) else [None, None]


def evaluate():
    started = time.perf_counter()
    if (ROOT / 'results.json').exists():
        raise RuntimeError('Recorded results exist; no overwrite')
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    manifest = json.loads((ROOT / 'fit_manifest.json').read_text())
    assert manifest['protocol_sha256'] == sha(ROOT / 'protocol.json')
    for f, h in manifest['code_hashes'].items():
        assert sha(ROOT / f) == h
    save('evaluation_started.json', {'utc': now(), 'fit_manifest_sha256': sha(ROOT / 'fit_manifest.json'), 'only_evaluation_split': 'validation'})
    cache = np.load(SOURCE / 'audit_arrays.npz')
    val = old.load_split(cache, 'validation')
    train = old.load_split(cache, 'train')
    contexts = protocol['context_pair']
    h = np.load(ROOT / 'training_halves.npz')
    c = np.load(ROOT / 'common_B.npz')
    common = {'context': c['context'], 'item': c['item'], 'global': c['global_']}
    pred = {}
    for model in manifest['pair_models']:
        assert sha(ROOT / model['path']) == model['sha256']
        pred[model['name']] = old.predict(val, np.load(ROOT / model['path']), common)
    losses = {k: old.loss(val['long_view'], p) for k, p in pred.items()}
    patterns = pairing_indices()
    result_seeds, metrics, pair_rows, boot_arrays, aggregates = [], [], [], {}, {}
    # Outcome-free A item exposure sets, used only as an adjustment covariate.
    exposure = {}
    for u, video, isa in zip(train['user_id'], train['video_id'], h['split_a']):
        if isa:
            exposure.setdefault(int(u), set()).add(int(video))
    for seed in SEEDS:
        b = np.load(ROOT / f'blocks_s{seed}.npz')
        blocks, low, high = b['users'], b['low'], b['high']
        user_block = np.full(len(h['counts_a']), -1, dtype='int32')
        user_block[blocks.ravel()] = np.repeat(np.arange(len(blocks)), 4)
        block_id = np.full(len(val['user_id']), -1, dtype='int32')
        valid_u = val['user_id'] < len(user_block)
        block_id[valid_u] = user_block[val['user_id'][valid_u]]
        keep = (block_id >= 0) & np.isin(val['tab'], contexts)
        ix = np.flatnonzero(keep)
        block_count = np.bincount(block_id[keep], minlength=len(blocks)).astype('float64')
        sums = {'individual': np.bincount(block_id[keep], weights=losses['individual_B'][keep], minlength=len(blocks))}
        for name in ['matching0', 'matching1', 'matching2', 'low', 'high']:
            sums[name] = np.bincount(block_id[keep], weights=losses[f's{seed}_{name}'][keep], minlength=len(blocks))
        sums['neutral'] = np.mean([sums[f'matching{k}'] for k in range(3)], axis=0)
        for name, sm in sums.items():
            value = float(sm.sum() / block_count.sum())
            source_loss = losses['individual_B'] if name == 'individual' else np.mean([losses[f's{seed}_matching{k}'] for k in range(3)], axis=0) if name == 'neutral' else losses[f's{seed}_{name}']
            context_losses = [float(source_loss[keep & (val['tab'] == s)].mean()) if (keep & (val['tab'] == s)).any() else None for s in contexts]
            metrics.append({'seed': seed, 'model': name, 'users': blocks.size, 'blocks': len(blocks), 'validation_rows': len(ix), 'log_loss': value, 'excess_vs_individual': value - float(sums['individual'].sum() / block_count.sum()), 'equal_context_loss': float(np.mean(context_losses)) if all(x is not None for x in context_losses) else None})
        rng = np.random.default_rng(20261016)
        weights = np.array([np.bincount(rng.integers(0, len(blocks), size=len(blocks)), minlength=len(blocks)) for _ in range(1000)])
        denominator = weights @ block_count
        repeat = b['score_b'][np.arange(len(blocks)), high] - b['score_b'][np.arange(len(blocks)), low]
        boot_repeat = (weights @ repeat) / len(blocks)
        contrasts = {'high_minus_low': sums['high'] - sums['low'], 'neutral_minus_low': sums['neutral'] - sums['low'], 'high_minus_individual': sums['high'] - sums['individual']}
        summaries = {}
        for name, d in contrasts.items():
            values = np.divide(weights @ d, denominator, out=np.full(1000, np.nan), where=denominator > 0)
            summaries[name] = {'estimate': float(d.sum() / block_count.sum()), 'block_bootstrap_95': ci(values), 'adjusted_one_sided_lower': float(np.nanquantile(values, .05 / 4))}
            boot_arrays[f's{seed}_{name}'] = values
        # Six unique pairs per block: independent-of-validation predictor variables.
        xs, ys, ws, block_rows = [], [], [], []
        for block, users in enumerate(blocks):
            for matching in range(3):
                for pair in patterns[matching]:
                    u, v = [int(x) for x in users[pair]]
                    pair_mask = keep & ((val['user_id'] == u) | (val['user_id'] == v))
                    n = int(pair_mask.sum())
                    if not n:
                        continue
                    a_counts = h['counts_a'][[u, v]][:, contexts]
                    count_logs = np.log1p(a_counts)
                    eu, ev = exposure.get(u, set()), exposure.get(v, set())
                    jaccard = len(eu & ev) / max(len(eu | ev), 1)
                    x = [(h['contrast_a'][u] - h['contrast_a'][v]) ** 2,
                         (h['marginal'][u] - h['marginal'][v]) ** 2,
                         (h['overall'][u] - h['overall'][v]) ** 2,
                         float(((count_logs[0] - count_logs[1]) ** 2).sum()),
                         float(count_logs.mean()), float((h['marginal'][u] + h['marginal'][v]) / 2), jaccard]
                    y = float((losses[f's{seed}_matching{matching}'][pair_mask] - losses['individual_B'][pair_mask]).mean())
                    xs.append(x); ys.append(y); ws.append(n); block_rows.append(block)
                    pair_rows.append({'seed': seed, 'block': block, 'matching': matching, 'u': u, 'v': v, 'validation_rows': n, 'excess_loss': y, **{f'x{k}': float(t) for k, t in enumerate(x)}})
        x, y, w, ids = np.array(xs), np.array(ys), np.array(ws), np.array(block_rows)
        scale = np.maximum(x.std(axis=0), 1e-12)
        sx = x / scale
        beta = fixed_effect_fit(sx, y, w, ids)
        beta_boot = np.array([fixed_effect_fit(sx, y, w * wb[ids], ids)[0] for wb in weights])
        summaries['adjusted_context_coefficient_per_feature_sd'] = {'estimate': float(beta[0]), 'block_bootstrap_95': ci(beta_boot), 'adjusted_one_sided_lower': float(np.nanquantile(beta_boot, .05 / 4)), 'features': ['context_disagreement_sq', 'marginal_difference_sq', 'overall_difference_sq', 'activity_mismatch', 'mean_log_activity', 'mean_marginal', 'item_overlap_jaccard'], 'coefficients_per_feature_sd': beta.tolist(), 'design_rank': int(np.linalg.matrix_rank(sx))}
        # User-cluster sensitivity: retain paired model contrast for each sampled user.
        uu, inv = np.unique(val['user_id'][keep], return_inverse=True)
        uc = np.bincount(inv)
        user_weights = np.array([np.bincount(rng.integers(0, len(uu), size=len(uu)), minlength=len(uu)) for _ in range(1000)])
        for name, event_difference in [('high_minus_low', losses[f's{seed}_high'] - losses[f's{seed}_low']), ('neutral_minus_low', np.mean([losses[f's{seed}_matching{k}'] for k in range(3)], axis=0) - losses[f's{seed}_low'])]:
            usum = np.bincount(inv, weights=event_difference[keep])
            ub = (user_weights @ usum) / (user_weights @ uc)
            summaries[name]['user_bootstrap_95_sensitivity'] = ci(ub)
        summaries['B_profile_repeatability'] = {'estimate': float(repeat.mean()), 'block_bootstrap_95': ci(boot_repeat), 'adjusted_one_sided_lower': float(np.nanquantile(boot_repeat, .05 / 4))}
        boot_arrays[f's{seed}_B_repeatability'] = boot_repeat
        boot_arrays[f's{seed}_adjusted_coefficient'] = beta_boot
        aggregates[f's{seed}_counts'] = block_count
        aggregates[f's{seed}_repeatability'] = repeat
        aggregates[f's{seed}_validation_indices'] = ix
        for name, sm in sums.items():
            aggregates[f's{seed}_{name}_loss_sums'] = sm
        result_seeds.append({'seed': seed, 'matched_blocks': len(blocks), 'matched_users': int(blocks.size), 'observed_validation_users': len(uu), 'validation_events': len(ix), 'contrasts': summaries})
        print(json.dumps({'evaluated_pairing_seed': seed, 'blocks': len(blocks), 'validation_events': len(ix), 'contrasts': summaries}), flush=True)
    np.savez_compressed(ROOT / 'pair_predictions_validation.npz', **pred)
    np.savez_compressed(ROOT / 'block_aggregates.npz', **aggregates)
    np.savez_compressed(ROOT / 'bootstrap_contrasts.npz', **boot_arrays)
    write_csv('pair_metrics.csv', metrics)
    write_csv('adjustment_pairs.csv', pair_rows)
    full = np.load(ROOT / 'common_full.npz')
    full_common = {'context': full['context'], 'item': full['item'], 'global': full['global_']}
    optimizer_rows = []
    optimizer_predictions = {}
    for model in manifest['optimizer_models']:
        row = {k: model[k] for k in ['name', 'budget', 'seed', 'lambda', 'rounds', 'termination']}
        for stage in ['round12', 'final']:
            file = ROOT / 'models' / (model['name'] + '_' + stage + '.npz')
            assert sha(file) == model[stage + '_model_sha256']
            pp = old.predict(val, np.load(file), full_common)
            optimizer_predictions[model['name'] + '_' + stage] = pp
            row[stage + '_validation_loss'] = float(old.loss(val['long_view'], pp).mean())
        row['validation_loss_change_after_round12'] = row['final_validation_loss'] - row['round12_validation_loss']
        optimizer_rows.append(row)
    write_csv('optimizer_validation_metrics.csv', optimizer_rows)
    np.savez_compressed(ROOT / 'optimizer_predictions_validation.npz', **optimizer_predictions)
    primary = next(s for s in result_seeds if s['seed'] == 11)
    cc = primary['contrasts']
    conditions = {
        'adequate_support': primary['matched_blocks'] >= 50 and primary['validation_events'] >= 1000,
        'B_repeatability': cc['B_profile_repeatability']['adjusted_one_sided_lower'] > 0,
        'extra_sharing_damage': cc['high_minus_low']['estimate'] >= .001 and cc['high_minus_low']['adjusted_one_sided_lower'] > 0,
        'useful_grouping': cc['neutral_minus_low']['estimate'] >= .001 and cc['neutral_minus_low']['adjusted_one_sided_lower'] > 0,
        'adjusted_association': cc['adjusted_context_coefficient_per_feature_sd']['adjusted_one_sided_lower'] > 0,
    }
    replicated = sum(s['contrasts']['B_profile_repeatability']['estimate'] > 0 and s['contrasts']['high_minus_low']['estimate'] > 0 and s['contrasts']['neutral_minus_low']['estimate'] > 0 for s in result_seeds)
    conditions['direction_repeats_at_least_two_seeds'] = replicated >= 2
    results = {'completed_utc': now(), 'status': 'exploratory_mechanism_diagnostic_executed',
               'protocol_sha256': sha(ROOT / 'protocol.json'), 'fit_manifest_sha256': sha(ROOT / 'fit_manifest.json'),
               'primary_seed': 11, 'gate_conditions': conditions, 'gate_pass': all(conditions.values()),
               'pairing_results': result_seeds, 'optimizer_results': optimizer_rows,
               'fit_seconds': manifest['fit_seconds'], 'evaluation_seconds': time.perf_counter() - started,
               'test_standard_or_random_labels_accessed': False,
               'scientific_scope': 'Logged-response association and count-sharing intervention on validation; not intrinsic preferences, causal effect of context conflict, new-method novelty, independent confirmation, or main-track readiness.'}
    save('results.json', results)
    print(json.dumps(sanitize({'gate_conditions': conditions, 'gate_pass': results['gate_pass'], 'fit_seconds': results['fit_seconds'], 'evaluation_seconds': results['evaluation_seconds']})), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'fit', 'evaluate'])
    args = parser.parse_args()
    globals()[args.stage]()
