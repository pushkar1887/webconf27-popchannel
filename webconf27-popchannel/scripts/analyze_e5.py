"""E5: local analysis of stored top-20 lists; never scores/trains a model.

The default path preserves the pilot's analysis-time operational definitions.
The --final path uses the frozen S3-E5 roster and tail/coverage definitions.
All metric values and deltas in JSON are fractions, not percentages/percentage points.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np


def item_strata(counts, head_fraction=0.2):
    """Exact top ceil(f*N) catalog items; descending count, ascending original ID."""
    counts = np.asarray(counts)
    if counts.ndim != 1 or not len(counts) or np.any(counts < 0):
        raise ValueError('counts must be a nonnegative, nonempty item vector')
    ids = np.arange(len(counts))
    order = np.lexsort((ids, -counts.astype(np.int64)))
    head = np.zeros(len(counts), dtype=bool)
    head[order[:math.ceil(head_fraction * len(counts))]] = True
    quintile = np.zeros(len(counts), dtype=np.int8)
    for q, block in enumerate(np.array_split(order, 5), 1):
        quintile[block] = q  # Q1 is the most popular fifth.
    return head, quintile


def activity_quartiles(users, train_activity):
    """Near-equal groups among evaluated users; Q1 is least active."""
    users = np.asarray(users)
    order = np.lexsort((users, np.asarray(train_activity)[users]))
    groups = np.zeros(len(users), dtype=np.int8)
    for q, block in enumerate(np.array_split(order, 4), 1):
        groups[block] = q
    return groups


def _change(base, prior):
    return {'baseline': float(base), 'corrected': float(prior), 'delta': float(prior - base)}


def _mean_or_none(values):
    return float(np.mean(values)) if len(values) else None


def _eligible_metric(base, prior, eligible):
    if not np.any(eligible):
        return {'baseline': None, 'corrected': None, 'delta': None}
    return _change(np.mean(base[eligible]), np.mean(prior[eligible]))


def validate_predictions(arrays, truths, counts, k=20, seen=None, prepared=None, return_hits=False):
    """Validate NPZ original user/item IDs, exact truth, hit fields and masking."""
    users = np.asarray(arrays['users'], dtype=np.int64)
    top = np.asarray(arrays['top20'], dtype=np.int64)
    npos = np.asarray(arrays['npos'], dtype=np.int64)
    if users.ndim != 1 or len(set(users.tolist())) != len(users):
        raise ValueError('duplicate or malformed users')
    if top.shape != (len(users), k):
        raise ValueError('top20 shape does not match users and k')
    if np.any(top < 0) or np.any(top >= len(counts)):
        raise ValueError('item IDs outside original catalog')
    if np.any(np.diff(np.sort(top, axis=1), axis=1) == 0):
        raise ValueError('duplicate recommendations within a row')
    if set(users.tolist()) != set(truths):
        raise ValueError('evaluated users do not exactly cover test-truth users')
    expected_npos = np.array([len(truths[int(u)]) for u in users])
    if npos.shape != expected_npos.shape or not np.array_equal(npos, expected_npos):
        raise ValueError('npos does not match original test truth')
    if np.any(npos <= 0):
        raise ValueError('empty test truth in evaluated cohort')
    codes = users[:, None] * len(counts) + top if prepared is not None else None
    hit = (_indexed_membership(codes, prepared['truth_codes']) if prepared is not None else
           np.array([np.isin(row, truths[int(u)]) for u, row in zip(users, top)]))
    stored = np.asarray(arrays['hits'])
    expected = hit if stored.ndim == 2 else hit.sum(axis=1)
    if stored.shape != expected.shape or not np.array_equal(stored, expected):
        raise ValueError('hits do not match original test truth')
    recall = hit.sum(axis=1) / np.minimum(k, npos)
    if 'recall' in arrays and not np.allclose(arrays['recall'], recall, atol=1e-10, rtol=0):
        raise ValueError('stored recall does not match reconstructed hits')
    disc = 1 / np.log2(np.arange(2, k + 2))
    ndcg = (hit @ disc) / np.cumsum(disc)[np.minimum(k, npos) - 1]
    if 'ndcg' in arrays and not np.allclose(arrays['ndcg'], ndcg, atol=1e-7, rtol=0):
        raise ValueError('stored ndcg does not match reconstructed hits')
    seen_overlap = 0
    if seen is not None:
        seen_overlap = (int(_indexed_membership(codes, prepared['seen_codes']).sum())
                        if prepared is not None else
                        sum(int(np.isin(row, seen.get(int(u), [])).sum()) for u, row in zip(users, top)))
        if seen_overlap:
            raise ValueError(f'{seen_overlap} recommendations overlap train+validation mask')
    result = {'n_users': int(len(users)), 'reconstructed_recall': float(recall.mean()),
            'reconstructed_ndcg': float(ndcg.mean()), 'total_hits': int(hit.sum()),
            'seen_overlap': seen_overlap, 'truth_hit_lineage': 'validated'}
    return (result, hit) if return_hits else result


def analyze_pair(users, base_top, prior_top, truths, counts, train_activity, k=20):
    """Descriptive paired analysis; gross hit additions/losses remain separate."""
    users = np.asarray(users, dtype=np.int64)
    base_top = np.asarray(base_top, dtype=np.int64)
    prior_top = np.asarray(prior_top, dtype=np.int64)
    counts = np.asarray(counts)
    head, quintiles = item_strata(counts)
    activity = activity_quartiles(users, train_activity)
    n = len(users)
    b_hits = np.zeros(n, dtype=np.int32)
    p_hits = np.zeros(n, dtype=np.int32)
    b_tail = np.zeros(n, dtype=np.int32)
    p_tail = np.zeros(n, dtype=np.int32)
    npos = np.zeros(n, dtype=np.int32)
    ntail = np.zeros(n, dtype=np.int32)
    # columns base hits, corrected hits, additions, losses, truth events
    q_counts = np.zeros((5, 5), dtype=np.int64)
    per_q_truth = np.zeros((n, 5), dtype=np.int16)
    per_q_base = np.zeros((n, 5), dtype=np.int16)
    per_q_prior = np.zeros((n, 5), dtype=np.int16)
    for row, u in enumerate(users):
        truth = set(map(int, truths[int(u)]))
        bh = set(map(int, base_top[row])) & truth
        ph = set(map(int, prior_top[row])) & truth
        added, lost = ph - bh, bh - ph
        b_hits[row], p_hits[row], npos[row] = len(bh), len(ph), len(truth)
        b_tail[row] = sum(not head[i] for i in bh)
        p_tail[row] = sum(not head[i] for i in ph)
        ntail[row] = sum(not head[i] for i in truth)
        for q in range(1, 6):
            qset = lambda s: sum(quintiles[i] == q for i in s)
            per_q_base[row, q - 1] = qset(bh)
            per_q_prior[row, q - 1] = qset(ph)
            per_q_truth[row, q - 1] = qset(truth)
            q_counts[q - 1] += [qset(bh), qset(ph), qset(added), qset(lost), qset(truth)]
    total_added, total_lost = int(q_counts[:, 2].sum()), int(q_counts[:, 3].sum())
    denom = np.minimum(k, npos)
    base_recall, prior_recall = b_hits / denom, p_hits / denom
    tail_eligible = ntail > 0
    tail_cap = np.maximum(1, np.minimum(k, ntail))
    tail_standard = np.maximum(1, ntail)
    head_added, head_lost = map(int, q_counts[0, [2, 3]])
    head_share = head_added / total_added if total_added else None
    tail_cond = _eligible_metric(b_tail / tail_cap, p_tail / tail_cap, tail_eligible)
    quintile_results = []
    for q, qc in enumerate(q_counts, 1):
        elig = per_q_truth[:, q - 1] > 0
        d = np.maximum(1, np.minimum(k, per_q_truth[:, q - 1]))
        quintile_results.append({
            'quintile': q, 'definition': 'Q1 most popular; Q5 least popular',
            'n_items': int((quintiles == q).sum()), 'truth_events': int(qc[4]),
            'baseline_hits': int(qc[0]), 'corrected_hits': int(qc[1]),
            'gross_added': int(qc[2]), 'gross_lost': int(qc[3]), 'net': int(qc[2] - qc[3]),
            'eligible_users': int(elig.sum()),
            'conditional_capped_recall': _eligible_metric(per_q_base[:, q - 1] / d,
                                                          per_q_prior[:, q - 1] / d, elig),
            'overall_capped_recall_contribution': _change(np.mean(per_q_base[:, q - 1] / denom),
                                                         np.mean(per_q_prior[:, q - 1] / denom)),
        })
    quartile_results = []
    for q in range(1, 5):
        select = activity == q
        activities = np.asarray(train_activity)[users[select]]
        if not select.any():
            continue
        quartile_results.append({
            'quartile': q, 'n_users': int(select.sum()), 'train_count_min': int(activities.min()),
            'train_count_max': int(activities.max()), 'train_count_mean': float(activities.mean()),
            'capped_recall': _change(base_recall[select].mean(), prior_recall[select].mean()),
            'share_users_improved': float((p_hits[select] > b_hits[select]).mean()),
            'share_users_worse': float((p_hits[select] < b_hits[select]).mean()),
            'tail_eligible_users': int((select & tail_eligible).sum()),
            'tail_conditional_capped_recall': _eligible_metric(b_tail / tail_cap, p_tail / tail_cap,
                                                              select & tail_eligible),
        })
    tail_delta = tail_cond['delta']
    head_pass = None if head_share is None else head_share >= 0.6
    tail_pass = None if tail_delta is None else abs(tail_delta) <= 0.005
    return {
        'n_users': n, 'n_items': int(len(counts)), 'head_items': int(head.sum()),
        'head_train_count_min': int(counts[head].min()),
        'tail_train_count_max': int(counts[~head].max()) if np.any(~head) else None,
        'head_boundary_tied_counts': bool(np.any(~head) and counts[head].min() == counts[~head].max()),
        'overall': {'capped_recall': _change(base_recall.mean(), prior_recall.mean()),
                    'standard_recall': _change(np.mean(b_hits / npos), np.mean(p_hits / npos)),
                    'baseline_hits': int(b_hits.sum()), 'corrected_hits': int(p_hits.sum()),
                    'share_users_improved': float((p_hits > b_hits).mean()),
                    'share_users_worse': float((p_hits < b_hits).mean())},
        'hit_changes': {'gross_added': total_added, 'gross_lost': total_lost,
                        'net': total_added - total_lost, 'head_added': head_added,
                        'head_lost': head_lost, 'head_net': head_added - head_lost,
                        'tail_added': total_added - head_added, 'tail_lost': total_lost - head_lost,
                        'tail_net': (total_added - head_added) - (total_lost - head_lost),
                        'head_share_of_gross_added': head_share},
        'tail': {'eligible_users': int(tail_eligible.sum()), 'empty_tail_users': int((~tail_eligible).sum()),
                 'truth_events': int(ntail.sum()), 'baseline_hits': int(b_tail.sum()),
                 'corrected_hits': int(p_tail.sum()), 'conditional_capped_recall': tail_cond,
                 'conditional_standard_recall': _eligible_metric(b_tail / tail_standard,
                                                                  p_tail / tail_standard, tail_eligible),
                 'overall_capped_recall_contribution': _change(np.mean(b_tail / denom), np.mean(p_tail / denom))},
        'coverage': {'baseline_unique_items': int(np.unique(base_top).size),
                     'corrected_unique_items': int(np.unique(prior_top).size),
                     'baseline_catalog_fraction': float(np.unique(base_top).size / len(counts)),
                     'corrected_catalog_fraction': float(np.unique(prior_top).size / len(counts)),
                     'baseline_recommended_tail_share': float((~head[base_top]).mean()),
                     'corrected_recommended_tail_share': float((~head[prior_top]).mean())},
        'popularity_quintiles': quintile_results, 'user_activity_quartiles': quartile_results,
        'PR_E5_operational_check': {'head_at_least_60pct': head_pass,
                                    'abs_tail_delta_at_most_0_5pp': tail_pass,
                                    'joint': ('PASS' if head_pass and tail_pass else
                                              'INCONCLUSIVE' if head_pass is None or tail_pass is None else 'FAIL')},
    }


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def edge_groups(edges):
    users, items = np.asarray(edges)
    order = np.argsort(users, kind='stable')
    su, si = users[order], items[order]
    unique, start = np.unique(su, return_index=True)
    ends = np.r_[start[1:], len(si)]
    groups = {int(u): si[a:b].astype(np.int64) for u, a, b in zip(unique, start, ends)}
    if any(len(np.unique(i)) != len(i) for i in groups.values()):
        raise ValueError('duplicate user-item edges in split; denominator contract needs review')
    return groups


def load_dataset(data_root, dataset):
    import torch
    torch.set_num_threads(1)
    d = Path(data_root) / f'{dataset}-processed'
    paths = [d / f'{s}.pt' for s in ('train', 'val', 'test')]
    arrays = [torch.load(p, weights_only=True, map_location='cpu').long().numpy() for p in paths]
    if any(a.ndim != 2 or a.shape[0] != 2 or np.any(a < 0) for a in arrays):
        raise ValueError('invalid original edge tensors')
    nu = int(max(a[0].max() for a in arrays)) + 1
    ni = int(max(a[1].max() for a in arrays)) + 1
    truth = edge_groups(arrays[2])
    seen = edge_groups(np.concatenate(arrays[:2], axis=1))
    return {'truths': truth, 'seen': seen,
            'counts': np.bincount(arrays[0][1], minlength=ni),
            'activity': np.bincount(arrays[0][0], minlength=nu),
            'split_sizes': {s: int(a.shape[1]) for s, a in zip(('train', 'val', 'test'), arrays)},
            'files': [{'path': str(p), 'sha256': sha256(p)} for p in paths]}


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def align_rankings(a, b):
    if set(a['users'].tolist()) != set(b['users'].tolist()):
        raise ValueError('paired files have different user coverage')
    oa, ob = np.argsort(a['users']), np.argsort(b['users'])
    if not np.array_equal(a['users'][oa], b['users'][ob]):
        raise ValueError('paired users cannot be aligned')
    return a['users'][oa], a['top20'][oa], b['top20'][ob]


def final_roster(root):
    """Frozen S3-E5 roster, including retained seed/init variants and same-loss Fulls."""
    root = Path(root)
    specs = []
    for dataset in ('Gowalla', 'Yelp2018', 'AmazonBook'):
        for compression in ('full', 'graphhash', 'random', 'dfreq'):
            for loss in ('directau', 'bpr'):
                suffix = 's0'
                if dataset == 'Gowalla' and loss == 'bpr':
                    suffix = {'full': 's0-long', 'graphhash': 's0-p500'}.get(compression, suffix)
                legacy = dataset == 'AmazonBook' and compression == 'graphhash' and loss == 'bpr'
                if legacy:
                    suffix = 's0-legacy'
                run_id = f'{dataset.lower()}__mf__{compression}__{loss}__{suffix}'
                folder = (root / 'runs/_sessions/pc-x2-1017/controls' / run_id if legacy else
                          root / 'runs' / run_id)
                specs.append(dict(run_id=run_id, dataset=dataset, loss=loss,
                                  compression=compression, folder=folder,
                                  kind='legacy' if legacy else 'modern'))
    for dataset, loss, suffixes in (
            ('Gowalla', 'directau', ('s0-initmatch', 's1', 's2')),
            ('Gowalla', 'bpr', ('s1', 's2')),
            ('AmazonBook', 'directau', ('s0-initmatch',))):
        for suffix in suffixes:
            run_id = f'{dataset.lower()}__mf__graphhash__{loss}__{suffix}'
            specs.append(dict(run_id=run_id, dataset=dataset, loss=loss, compression='graphhash',
                              folder=root / 'runs' / run_id, kind='modern'))
    return specs


def final_mode_sources(spec):
    folder = Path(spec['folder'])
    repaired = 'cos_prior' if spec['loss'] == 'directau' else 'raw_prior'
    if spec['kind'] == 'legacy':
        return {'raw': folder / 'per_user__base_randomkey_s0.npz', 'cosine': None,
                repaired: folder / 'per_user__full_prior_s0.npz'}
    return {mode: folder / f'per_user_test__{mode}.npz' for mode in ('raw', 'cosine', repaired)}


def _indexed_membership(values, sorted_codes):
    if not len(sorted_codes):
        return np.zeros(np.shape(values), dtype=bool)
    pos = np.searchsorted(sorted_codes, values)
    return (pos < len(sorted_codes)) & (sorted_codes[np.minimum(pos, len(sorted_codes) - 1)] == values)


def prepare_final_dataset(data):
    """Index original split truth/masks once, without touching any model scores."""
    ni = len(data['counts'])
    head, _ = item_strata(data['counts'])
    ntail = np.zeros(len(data['activity']), dtype=np.int64)
    for u, truth in data['truths'].items():
        ntail[u] = (~head[truth]).sum()
    def codes(groups):
        return np.sort(np.concatenate([int(u) * ni + items for u, items in groups.items()]))
    return dict(head=head, ntail=ntail, truth_codes=codes(data['truths']),
                seen_codes=codes(data['seen']))


def final_metric_summary(arrays, hit, data, prepared, k=20):
    users, top = arrays['users'], arrays['top20']
    ntail = prepared['ntail'][users]
    eligible = ntail > 0
    tail_hits = (hit & ~prepared['head'][top]).sum(axis=1)
    capped = tail_hits / np.maximum(1, np.minimum(k, ntail))
    standard = tail_hits / np.maximum(1, ntail)
    contribution = tail_hits / np.minimum(k, arrays['npos'])
    order = np.argsort(users[eligible])
    result = dict(tail_primary=float(capped[eligible].mean()) if eligible.any() else None,
                  tail_standard=float(standard[eligible].mean()) if eligible.any() else None,
                  tail_contribution=float(contribution.mean()),
                  catalog_coverage=float(np.unique(top).size / len(data['counts'])),
                  tail_eligible_users=int(eligible.sum()), n_users=int(len(users)),
                  tail_truth_events=int(ntail.sum()), tail_hits=int(tail_hits.sum()),
                  catalog_items=int(len(data['counts'])), unique_recommended_items=int(np.unique(top).size))
    return result, dict(users=users[eligible][order], values=capped[eligible][order])


def paired_tail_comparison(variant, full, n_resamples=2000, seed=0, chunk=64):
    """Paired eligible-user percentile CI; direction uses the frozen mean-only rule."""
    vo, fo = np.argsort(variant['users']), np.argsort(full['users'])
    if not np.array_equal(variant['users'][vo], full['users'][fo]):
        raise ValueError('paired tail comparisons require identical eligible user IDs')
    delta = np.asarray(variant['values'])[vo] - np.asarray(full['values'])[fo]
    if not len(delta):
        raise ValueError('paired tail comparison has no eligible users')
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, chunk):
        size = min(chunk, n_resamples - start)
        means[start:start + size] = delta[rng.integers(0, len(delta), (size, len(delta)))].mean(axis=1)
    mean = float(delta.mean())
    return dict(delta=mean, ci95=np.quantile(means, [.025, .975]).tolist(),
                n_paired_users=int(len(delta)), n_resamples=n_resamples, seed=seed,
                method='paired eligible-user percentile bootstrap; seed reset to 0 for each comparison',
                direction='repaired GraphHash DirectAU minus same-dataset Full DirectAU raw',
                verdict='PASS' if mean < 0 else 'FAIL',
                decision_rule='mean delta < 0; CI is descriptive and does not change this rule')


def run_final(args):
    root = args.project_root.resolve()
    specs = final_roster(root)
    result = dict(schema='E5-final-v2', created_utc=datetime.now(timezone.utc).isoformat(),
                  scope='S3-E5 full registered 30-lineage roster; cached test predictions only',
                  units='All metric values, CI limits and deltas are fractions; x100 gives percent or percentage points.',
                  operational_definitions={
                      'head': 'exact top ceil(.2*N_catalog), training event count descending, original item ID ascending',
                      'tail_primary': 'macro tail_hits/min(20,n_tail_truth) among users with tail truth',
                      'tail_standard': 'macro tail_hits/n_tail_truth among users with tail truth',
                      'tail_contribution': 'mean tail_hits/min(20,n_all_truth) among ALL evaluated users',
                      'catalog_coverage': 'union of stored top20 item IDs / full original item catalog',
                      'repaired': 'cos_prior for DirectAU; raw_prior for BPR; frozen validation-selected beta',
                      'selection': 'No model scoring, ranking, training, clustering, or test-dependent selection'},
                  datasets={}, roster=[], records=[], comparisons=[], missing_cells=[])
    cache, prepared_cache, per_user = {}, {}, {}
    for spec in specs:
        dataset = spec['dataset']
        if dataset not in cache:
            print(f'Loading/indexing original {dataset} splits', flush=True)
            cache[dataset] = load_dataset(args.data_root, dataset)
            prepared_cache[dataset] = prepare_final_dataset(cache[dataset])
            result['datasets'][dataset] = {key: cache[dataset][key] for key in ('split_sizes', 'files')}
        data, prepared = cache[dataset], prepared_cache[dataset]
        folder = spec['folder']
        if spec['kind'] == 'modern':
            mp, cp = folder / 'metrics.json', folder / 'config.json'
            meta = json.loads(mp.read_text(encoding='utf-8'))
            config = json.loads(cp.read_text(encoding='utf-8'))
            if meta['run_id'] != spec['run_id'] or config['run_id'] != spec['run_id']:
                raise ValueError(f"{spec['run_id']}: metadata run ID mismatch")
            seed_data = next(iter(meta['seeds'].values()))
            ev = seed_data['eval']
            stop = seed_data.get('stop_reason', seed_data.get('stopped'))
            expected_hashes = config.get('dataset_stats', {}).get('file_sha256', {})
            for split, file in zip(('train', 'val', 'test'), data['files']):
                if expected_hashes.get(split) != file['sha256']:
                    raise ValueError(f"{spec['run_id']}: original {split} split hash differs from config")
            controlled = (dataset in ('Gowalla', 'AmazonBook') and spec['loss'] == 'directau' and
                          (spec['compression'] == 'full' or 'initmatch' in spec['run_id']))
            comparison_type = config.get('comparison_type', 'controlled' if controlled else 'reproduction')
            provenance = [dict(path=str(p), sha256=sha256(p)) for p in (mp, cp)]
        else:
            mp = folder / 'controls.json'
            meta = json.loads(mp.read_text(encoding='utf-8'))
            if meta['meta']['run_id'] != spec['run_id'] or meta['tie_keys']['base'] != 'raw':
                raise ValueError('legacy controls run/base lineage mismatch')
            stop, comparison_type = 'fixed_epochs', 'reproduction'
            provenance = [dict(path=str(mp), sha256=sha256(mp))]
        result['roster'].append({k: str(v) if isinstance(v, Path) else v for k, v in spec.items()})
        for mode, path in final_mode_sources(spec).items():
            record = {k: spec[k] for k in ('run_id', 'dataset', 'loss', 'compression')}
            record.update(scoring_mode=mode, mode=mode, lineage=spec['kind'], stop_reason=stop,
                          capped=stop in ('time_cap', 'max_epochs'), comparison_type=comparison_type,
                          source_policy=('stored_modern_test_top20' if spec['kind'] == 'modern' else
                                         'legacy_cached_random_key_seed0'), provenance=provenance)
            if path is None or not path.is_file():
                record.update(analysis_status='missing_gpu' if path is None else 'missing_input',
                              source_policy='no_cached_cosine; legacy mappings require GPU' if path is None else record['source_policy'])
                record.update({key: None for key in ('tail_primary', 'tail_standard', 'tail_contribution',
                                                     'catalog_coverage', 'tail_eligible_users', 'n_users')})
                result['missing_cells'].append(dict(run_id=spec['run_id'], scoring_mode=mode,
                                                    reason=record['analysis_status']))
            else:
                arrays = load_npz(path)
                validation, hit = validate_predictions(arrays, data['truths'], data['counts'],
                                                       seen=data['seen'], prepared=prepared, return_hits=True)
                metrics, individual = final_metric_summary(arrays, hit, data, prepared)
                beta = (float(ev[mode]['best_beta']) if spec['kind'] == 'modern' else
                        float(meta['full_prior']['best_beta']) if mode == 'raw_prior' else 0.0)
                if spec['kind'] == 'modern':
                    difference = abs(validation['reconstructed_recall'] - float(ev[mode]['test']['recall']))
                    if difference > 1e-7:
                        raise ValueError(f"{spec['run_id']} {mode}: saved scalar recall mismatch")
                    validation['saved_metric_abs_difference'] = difference
                record.update(metrics, metrics=metrics, beta=beta, validation=validation,
                              analysis_status='complete', input=dict(path=str(path), sha256=sha256(path)),
                              tie_contract='stored torch.topk' if spec['kind'] == 'modern' else 'random key seed 0')
                per_user[(spec['run_id'], mode)] = individual
            result['records'].append(record)
        print(f"Validated {spec['run_id']}", flush=True)
    lookup = {(r['run_id'], r['scoring_mode']): r for r in result['records']}
    fulls = {(s['dataset'], s['loss']): s['run_id'] for s in specs if s['compression'] == 'full'}
    for record in result['records']:
        full_id = fulls[record['dataset'], record['loss']]
        ref = lookup[(full_id, record['scoring_mode'])]
        record['same_loss_full_reference'] = dict(run_id=full_id, scoring_mode=record['scoring_mode'],
            stop_reason=ref['stop_reason'], capped=ref['capped'], comparison_type=ref['comparison_type'],
            tail_primary=ref['tail_primary'], catalog_coverage=ref['catalog_coverage'])
    for spec in specs:
        if spec['compression'] != 'graphhash' or spec['loss'] != 'directau':
            continue
        full_id = fulls[spec['dataset'], 'directau']
        comparison = paired_tail_comparison(per_user[(spec['run_id'], 'cos_prior')], per_user[(full_id, 'raw')])
        comparison.update(dataset=spec['dataset'], run_id=spec['run_id'], scoring_mode='cos_prior',
                          full_run_id=full_id, full_scoring_mode='raw',
                          repaired_tail_primary=lookup[spec['run_id'], 'cos_prior']['tail_primary'],
                          full_raw_tail_primary=lookup[full_id, 'raw']['tail_primary'])
        result['comparisons'].append(comparison)
    result['dataset_verdicts'] = {dataset: ('PASS' if all(c['verdict'] == 'PASS' for c in result['comparisons']
                                     if c['dataset'] == dataset) else 'FAIL') for dataset in cache}
    old_path = root / 'results/E5_stage12.json'
    result['original_PR_E5_negative_result'] = dict(
        verdict='FAIL', statement='Original <=0.5 percentage-point DirectAU tail-change prediction failed on all three pilot datasets; final E5 is an informed directional check, not retroactive validation.',
        evidence_file=str(old_path), evidence_sha256=sha256(old_path) if old_path.is_file() else None)
    result['summary'] = dict(registered_lineages=len(specs), expected_mode_cells=3 * len(specs),
                            complete_mode_cells=sum(r['analysis_status'] == 'complete' for r in result['records']),
                            missing_mode_cells=len(result['missing_cells']),
                            directional_prediction='PASS' if all(v == 'PASS' for v in result['dataset_verdicts'].values()) else 'FAIL',
                            evidence_scope='cached predictions validated against original splits; one legacy cosine mode unavailable')
    result['cases'] = result['records']
    out = args.out or root / 'results/E5_final.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result['summary']), flush=True)
    print(f'Saved {out}', flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--data-root', type=Path,
                    default=Path(r'D:\web_2\experiments\2026-10-16-loss-sharing-repro\_work\datasets'))
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--final', action='store_true', help='Analyze the complete frozen S3-E5 roster from cached predictions')
    args = ap.parse_args(argv)
    if args.final:
        return run_final(args)
    root = args.project_root.resolve()
    runs = root / 'runs'
    controls = runs / '_sessions/pc-x2-1017/controls'
    specs = [
        ('A1', 'Gowalla', 'directau', runs / 'gowalla__mf__graphhash__directau__s0', 'modern'),
        ('A4', 'Yelp2018', 'directau', runs / 'yelp2018__mf__graphhash__directau__s0', 'modern'),
        ('A5', 'Yelp2018', 'bpr', runs / 'yelp2018__mf__graphhash__bpr__s0', 'modern'),
        ('AmazonBook_legacy_DAU', 'AmazonBook', 'directau', controls / 'amazonbook__mf__graphhash__directau__s0-legacy41', 'legacy'),
        ('AmazonBook_legacy_BPR', 'AmazonBook', 'bpr', controls / 'amazonbook__mf__graphhash__bpr__s0-legacy', 'legacy'),
    ]
    result = {'schema': 'E5-v1', 'scope': 'partial (Stage 1-2 runs; not final Stage 3)',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'units': 'metric values and deltas are fractions; multiply by 100 for percent/percentage points',
              'operational_definitions': {
                  'head': 'exact top ceil(0.2*N_catalog) original item IDs by descending training event count; ties ascending item ID',
                  'added_hits': 'gross newly recovered (user,item) true positives present in corrected top20 but absent in baseline top20; rank changes alone do not count',
                  'lost_hits': 'baseline true positives absent from corrected top20; reported separately, never subtracted before head-share denominator',
                  'tail_primary': 'mean over users with tail truth of hits_tail/min(20,n_tail_truth)',
                  'tail_standard': 'mean over users with tail truth of hits_tail/n_tail_truth',
                  'tail_contribution': 'mean over ALL evaluated users of hits_tail/min(20,n_all_truth); head+tail contributions reconstruct main capped recall',
                  'activity_quartiles': 'equal-count groups of evaluated users ordered by training event count, ascending original user ID as tiebreak; Q1 least active',
                  'selection': 'all beta/checkpoint selections reused from saved validation choices; no scoring, test tuning, model training or clustering performed',
              },
              'prereg_caveat': 'PR-E5 did not freeze a head boundary tie rule, gross-versus-net hit convention, or tail denominator. These are explicit analysis-time operational choices, not retroactively preregistered. Report both tail denominators and contribution.',
              'datasets': {}, 'comparisons': []}
    cache = {}
    for tag, dataset, loss, folder, kind in specs:
        if dataset not in cache:
            print(f'Loading original {dataset} splits', flush=True)
            cache[dataset] = load_dataset(args.data_root, dataset)
            result['datasets'][dataset] = {'split_sizes': cache[dataset]['split_sizes'], 'files': cache[dataset]['files']}
        data = cache[dataset]
        if kind == 'modern':
            base_mode = 'cosine' if loss == 'directau' else 'raw'
            prior_mode = 'cos_prior' if loss == 'directau' else 'raw_prior'
            bp, pp = folder / f'per_user_test__{base_mode}.npz', folder / f'per_user_test__{prior_mode}.npz'
            meta = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
            ev = meta['seeds']['0']['eval']
            beta = ev[prior_mode]['best_beta']
            scorebase = base_mode
        else:
            bp, pp = folder / 'per_user__base_randomkey_s0.npz', folder / 'per_user__full_prior_s0.npz'
            meta = json.loads((folder / 'controls.json').read_text(encoding='utf-8'))
            scorebase = meta['tie_keys']['base']
            expected = 'cos' if loss == 'directau' else 'raw'
            if scorebase != expected:
                raise ValueError(f'{tag}: legacy primary base {scorebase} != {expected}')
            beta = meta['full_prior']['best_beta']
        a, b = load_npz(bp), load_npz(pp)
        validation = {'baseline': validate_predictions(a, data['truths'], data['counts'], seen=data['seen']),
                      'corrected': validate_predictions(b, data['truths'], data['counts'], seen=data['seen'])}
        if kind == 'modern':
            for label, mode in [('baseline', base_mode), ('corrected', prior_mode)]:
                measured = validation[label]['reconstructed_recall']
                reference = float(ev[mode]['test']['recall'])
                if abs(measured - reference) > 1e-7:
                    raise ValueError(f'{tag} {label} does not reproduce saved scalar recall')
                validation[label]['saved_metric_abs_difference'] = abs(measured - reference)
        users, bt, pt = align_rankings(a, b)
        primary = analyze_pair(users, bt, pt, data['truths'], data['counts'], data['activity'])
        record = {'id': tag, 'dataset': dataset, 'loss': loss, 'run_id': folder.name,
                  'baseline_mode': scorebase, 'corrected_mode': f'{scorebase}+prior', 'beta': beta,
                  'tie_contract': ('stored torch.topk order' if kind == 'modern' else 'exact bucket scores, random key seed 0 for both cells'),
                  'scope_notes': ('modern patience-stopped Stage 2 checkpoint' if kind == 'modern' else
                                  'legacy Stage 1 control checkpoint; provisional budget/convergence, one tie-key seed, not final Stage 3'),
                  'inputs': [{'path': str(p), 'sha256': sha256(p)} for p in (bp, pp)],
                  'validation': validation, 'primary': primary}
        if kind == 'modern' and loss == 'directau':
            rp = folder / 'per_user_test__raw.npz'
            raw = load_npz(rp)
            record['raw_baseline_validation'] = validate_predictions(raw, data['truths'], data['counts'], seen=data['seen'])
            ru, rt, rpt = align_rankings(raw, b)
            record['secondary_original_raw_baseline'] = analyze_pair(ru, rt, rpt, data['truths'], data['counts'], data['activity'])
            record['inputs'].append({'path': str(rp), 'sha256': sha256(rp)})
        result['comparisons'].append(record)
        p = primary
        print(json.dumps({'id': tag, 'base_pct': 100 * p['overall']['capped_recall']['baseline'],
                          'prior_pct': 100 * p['overall']['capped_recall']['corrected'],
                          'head_share_added_pct': 100 * p['hit_changes']['head_share_of_gross_added'],
                          'tail_conditional_delta_pp': 100 * p['tail']['conditional_capped_recall']['delta'],
                          'tail_contribution_delta_pp': 100 * p['tail']['overall_capped_recall_contribution']['delta'],
                          'verdict': p['PR_E5_operational_check']['joint']}), flush=True)
    out = args.out or root / 'results/E5_stage12.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Saved {out}', flush=True)


if __name__ == '__main__':
    main()
