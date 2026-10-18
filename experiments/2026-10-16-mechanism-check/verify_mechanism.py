"""Independent data, pairing, prediction and inference replay for the research spike."""
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / '2026-10-15-kuairand-support'
E0 = ROOT.parent / '2026-10-15-count-e0'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def logit(p):
    p = np.clip(p.astype('float64'), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def replay(model, common, u, i, s):
    prior = common['context'][s]
    item = prior.copy()
    valid = i < len(common['item'])
    item[valid] = common['item'][i[valid], s[valid]]
    effect = np.zeros(len(u))
    if str(model['kind'].item()) == 'ungrouped':
        valid = u < len(model['user'])
        effect[valid] = logit(model['user'][u[valid], s[valid]]) - logit(prior[valid])
    else:
        group = np.full(len(u), -1, dtype='int32')
        valid = u < len(model['mapping'])
        group[valid] = model['mapping'][u[valid]]
        valid &= group >= 0
        effect[valid] = logit(model['group'][group[valid], s[valid]]) - logit(prior[valid])
    z = np.clip(logit(item) + effect, -25, 25)
    return (1 / (1 + np.exp(-z))).astype('float32')


def direct_beta(x, y, w, block):
    # Explicit within-block weighted demeaning, independent of the fitted helper.
    xd, yd = x.copy(), y.copy()
    for b in np.unique(block):
        m = block == b
        ww = w[m]
        if ww.sum() > 0:
            xd[m] -= np.average(x[m], axis=0, weights=ww)
            yd[m] -= np.average(y[m], weights=ww)
    root = np.sqrt(w)
    return np.linalg.lstsq(xd * root[:, None], yd * root, rcond=1e-10)[0]


def main():
    started = time.perf_counter()
    p = json.loads((ROOT / 'protocol.json').read_text())
    m = json.loads((ROOT / 'fit_manifest.json').read_text())
    r = json.loads((ROOT / 'results.json').read_text())
    receipt = json.loads((ROOT / 'evaluation_started.json').read_text())
    assert sha(SOURCE / 'audit_arrays.npz') == p['source_cache_sha256']
    assert sha(ROOT / 'protocol.json') == m['protocol_sha256'] == r['protocol_sha256']
    assert sha(ROOT / 'fit_manifest.json') == r['fit_manifest_sha256'] == receipt['fit_manifest_sha256']
    assert m['frozen_utc'] <= receipt['utc'] <= r['completed_utc']
    for file, digest in m['code_hashes'].items():
        assert sha(ROOT / file) == digest
    for file, digest in p['reused_sources'].items():
        assert sha(file) == digest
    cache = np.load(SOURCE / 'audit_arrays.npz')
    tu, ti, ts, ty = [cache['train__' + k] for k in ['user_id', 'video_id', 'tab', 'long_view']]
    u, i, s, y = [cache['validation__' + k] for k in ['user_id', 'video_id', 'tab', 'long_view']]
    h = np.load(ROOT / 'training_halves.npz')
    split = np.random.default_rng(20261016).random(len(tu)) < .5
    np.testing.assert_array_equal(split, h['split_a'])
    counts, positives = {}, {}
    for name, mask in [('a', split), ('b', ~split), ('full', np.ones(len(split), dtype=bool))]:
        key = tu[mask].astype('int64') * 15 + ts[mask]
        counts[name] = np.bincount(key, minlength=h['counts_a'].size).reshape(h['counts_a'].shape)
        positives[name] = np.bincount(key, weights=ty[mask], minlength=h['counts_a'].size).reshape(h['counts_a'].shape)
        if name != 'full':
            np.testing.assert_array_equal(counts[name], h['counts_' + name])
            np.testing.assert_array_equal(positives[name], h['positives_' + name])
    contexts = p['context_pair']
    eligible = (counts['full'][:, contexts] >= 20).all(axis=1)
    eligible &= (counts['a'][:, contexts] >= 8).all(axis=1) & (counts['b'][:, contexts] >= 8).all(axis=1)
    np.testing.assert_array_equal(np.flatnonzero(eligible), h['eligible'])
    c = np.load(ROOT / 'common_B.npz')
    gp = (ty[~split].sum() + .5) / ((~split).sum() + 1)
    cp = ((positives['b'].sum(axis=0) + 10 * gp) / (counts['b'].sum(axis=0) + 10)).astype('float32')
    np.testing.assert_array_equal(cp, c['context'])
    # Independently reconstruct the item baseline from B, not validation labels.
    item_key = ti[~split].astype('int64') * 15 + ts[~split]
    item_n = np.bincount(item_key, minlength=c['item'].size).reshape(c['item'].shape)
    item_s = np.bincount(item_key, weights=ty[~split], minlength=c['item'].size).reshape(c['item'].shape)
    np.testing.assert_array_equal(((item_s + 10 * cp) / (item_n + 10)).astype('float32'), c['item'])
    for name, mask in [('a', split), ('b', ~split)]:
        g = (ty[mask].sum() + .5) / (mask.sum() + 1)
        prior = ((positives[name].sum(axis=0) + 10 * g) / (counts[name].sum(axis=0) + 10)).astype('float32')
        rates = (positives[name] + 10 * prior) / (counts[name] + 10)
        logs = logit(rates[:, contexts])
        np.testing.assert_allclose(logs[:, 0] - logs[:, 1], h['contrast_' + name], atol=1e-12, rtol=0)
        if name == 'a':
            np.testing.assert_allclose(logs.mean(axis=1), h['marginal'], atol=1e-12, rtol=0)
    up = ((positives['b'] + 10 * cp) / (counts['b'] + 10)).astype('float32')
    np.testing.assert_array_equal(up, np.load(ROOT / 'models' / 'individual_B.npz')['user'])
    preds = np.load(ROOT / 'pair_predictions_validation.npz')
    blockagg = np.load(ROOT / 'block_aggregates.npz')
    boot = np.load(ROOT / 'bootstrap_contrasts.npz')
    errors = {'prediction': 0., 'count_table': 0., 'bootstrap': 0., 'adjusted_coefficient': 0.}
    losses = {}
    for model in m['pair_models']:
        file = ROOT / model['path']
        assert sha(file) == model['sha256']
        f = np.load(file)
        if 'mapping' in f:
            mapped = f['mapping'][tu]
            good = mapped >= 0
            key = mapped[good & ~split].astype('int64') * 15 + ts[good & ~split]
            n = np.bincount(key, minlength=f['group'].size).reshape(f['group'].shape)
            sums = np.bincount(key, weights=ty[good & ~split], minlength=f['group'].size).reshape(f['group'].shape)
            expected = ((sums + 10 * cp) / (n + 10)).astype('float32')
            np.testing.assert_array_equal(expected, f['group'])
            assert np.all(np.bincount(f['mapping'][f['mapping'] >= 0]) == 2)
        prediction = replay(f, c, u, i, s)
        errors['prediction'] = max(errors['prediction'], float(np.max(np.abs(prediction - preds[model['name']]))))
        np.testing.assert_allclose(prediction, preds[model['name']], atol=6e-8, rtol=0)
        # Use exact stored probabilities for arithmetic replay; alternative logit formulas may round at 1 ULP.
        pr = np.clip(preds[model['name']].astype('float64'), 1e-6, 1 - 1e-6)
        losses[model['name']] = -np.where(y == 1, np.log(pr), np.log1p(-pr))
    patterns = np.array([[[0, 1], [2, 3]], [[0, 2], [1, 3]], [[0, 3], [1, 2]]])
    metrics = list(csv.DictReader((ROOT / 'pair_metrics.csv').open()))
    adjustment = list(csv.DictReader((ROOT / 'adjustment_pairs.csv').open()))
    sizes = {}
    for seed in [11, 29, 47]:
        b = np.load(ROOT / f'blocks_s{seed}.npz')
        blocks, low, high = b['users'], b['low'], b['high']
        assert len(np.unique(blocks)) == blocks.size
        assert eligible[blocks].all()
        assert np.max(np.ptp(h['overall'][blocks], axis=1)) <= .5 + 1e-12
        assert np.max(np.ptp(h['marginal'][blocks], axis=1)) <= .5 + 1e-12
        for context in contexts:
            nn = counts['full'][blocks, context]
            assert np.max(nn.max(axis=1) / nn.min(axis=1)) <= 2 + 1e-12
        ca = h['contrast_a'][blocks]
        cb = h['contrast_b'][blocks]
        scores_a = np.array([((ca[:, pat[:, 0]] - ca[:, pat[:, 1]]) ** 2).mean(axis=1) for pat in patterns]).T
        scores_b = np.array([((cb[:, pat[:, 0]] - cb[:, pat[:, 1]]) ** 2).mean(axis=1) for pat in patterns]).T
        np.testing.assert_array_equal(low, scores_a.argmin(axis=1))
        np.testing.assert_array_equal(high, scores_a.argmax(axis=1))
        np.testing.assert_allclose(scores_a, b['score_a'], atol=1e-12)
        np.testing.assert_allclose(scores_b, b['score_b'], atol=1e-12)
        # Check model labels really implement the specified high/low pairings.
        for label in ['low', 'high', 'matching0', 'matching1', 'matching2']:
            mapping = np.load(ROOT / 'models' / f's{seed}_{label}.npz')['mapping']
            choice = low if label == 'low' else high if label == 'high' else np.full(len(blocks), int(label[-1]))
            for block, users in enumerate(blocks):
                partners = users[patterns[choice[block]]]
                assert mapping[partners[0, 0]] == mapping[partners[0, 1]]
                assert mapping[partners[1, 0]] == mapping[partners[1, 1]]
                assert mapping[partners[0, 0]] != mapping[partners[1, 0]]
            np.testing.assert_array_equal(np.flatnonzero(mapping >= 0), np.sort(blocks.ravel()))
        sizes[seed] = sorted(set(row['varying_array_bytes'] for row in m['pair_models'] if row.get('seed') == seed))
        assert len(sizes[seed]) == 1
        ub = np.full(len(counts['full']), -1, dtype='int32')
        ub[blocks.ravel()] = np.repeat(np.arange(len(blocks)), 4)
        event_block = np.full(len(u), -1, dtype='int32')
        ok = u < len(ub)
        event_block[ok] = ub[u[ok]]
        keep = (event_block >= 0) & np.isin(s, contexts)
        np.testing.assert_array_equal(np.flatnonzero(keep), blockagg[f's{seed}_validation_indices'])
        n = np.bincount(event_block[keep], minlength=len(blocks))
        np.testing.assert_array_equal(n, blockagg[f's{seed}_counts'])
        sums = {}
        for name in ['individual', 'low', 'high', 'matching0', 'matching1', 'matching2']:
            ll = losses['individual_B' if name == 'individual' else f's{seed}_{name}']
            sums[name] = np.bincount(event_block[keep], weights=ll[keep], minlength=len(blocks))
            np.testing.assert_allclose(sums[name], blockagg[f's{seed}_{name}_loss_sums'], atol=1e-11, rtol=0)
        sums['neutral'] = (sums['matching0'] + sums['matching1'] + sums['matching2']) / 3
        for row in metrics:
            if int(row['seed']) == seed:
                assert abs(sums[row['model']].sum() / n.sum() - float(row['log_loss'])) < 1e-12
        rng = np.random.default_rng(20261016)
        weights = np.array([np.bincount(rng.integers(0, len(blocks), size=len(blocks)), minlength=len(blocks)) for _ in range(1000)])
        for name, diff in [('high_minus_low', sums['high'] - sums['low']), ('neutral_minus_low', sums['neutral'] - sums['low']), ('high_minus_individual', sums['high'] - sums['individual'])]:
            expected = (weights @ diff) / (weights @ n)
            np.testing.assert_allclose(expected, boot[f's{seed}_{name}'], atol=1e-12, rtol=0)
            errors['bootstrap'] = max(errors['bootstrap'], float(np.max(np.abs(expected - boot[f's{seed}_{name}']))))
        repeat = scores_b[np.arange(len(blocks)), high] - scores_b[np.arange(len(blocks)), low]
        np.testing.assert_allclose((weights @ repeat) / len(blocks), boot[f's{seed}_B_repeatability'], atol=1e-12, rtol=0)
        rr = [row for row in adjustment if int(row['seed']) == seed]
        x = np.array([[float(row[f'x{k}']) for k in range(7)] for row in rr])
        yy = np.array([float(row['excess_loss']) for row in rr])
        ww = np.array([int(row['validation_rows']) for row in rr])
        bb = np.array([int(row['block']) for row in rr])
        sx = x / np.maximum(x.std(axis=0), 1e-12)
        expected_beta = direct_beta(sx, yy, ww, bb)
        actual = next(z for z in r['pairing_results'] if z['seed'] == seed)['contrasts']['adjusted_context_coefficient_per_feature_sd']['coefficients_per_feature_sd']
        np.testing.assert_allclose(expected_beta, actual, atol=1e-11, rtol=0)
        # Five independent full within-block bootstrap regressions per seed.
        for k in range(5):
            expected = direct_beta(sx, yy, ww * weights[k, bb], bb)[0]
            actual = boot[f's{seed}_adjusted_coefficient'][k]
            errors['adjusted_coefficient'] = max(errors['adjusted_coefficient'], abs(expected - actual))
            assert abs(expected - actual) < 1e-11
        seed_result = next(z for z in r['pairing_results'] if z['seed'] == seed)
        for name, array_name in [('high_minus_low', 'high_minus_low'), ('neutral_minus_low', 'neutral_minus_low'), ('high_minus_individual', 'high_minus_individual'), ('B_profile_repeatability', 'B_repeatability'), ('adjusted_context_coefficient_per_feature_sd', 'adjusted_coefficient')]:
            vals = boot[f's{seed}_{array_name}']
            summary = seed_result['contrasts'][name]
            np.testing.assert_allclose(np.quantile(vals, [.025, .975]), summary['block_bootstrap_95'], atol=1e-12, rtol=0)
            assert abs(np.quantile(vals, .05 / 4) - summary['adjusted_one_sided_lower']) < 1e-12
    # All 24 saved optimizer snapshots replay; each objective history is monotone.
    full_common = np.load(ROOT / 'common_full.npz')
    optimizer_preds = np.load(ROOT / 'optimizer_predictions_validation.npz')
    optimizer_csv = {row['name']: row for row in csv.DictReader((ROOT / 'optimizer_validation_metrics.csv').open())}
    for model in m['optimizer_models']:
        values = [v['objective'] for v in model['history']]
        assert np.all(np.diff(values) < 0)
        for stage in ['round12', 'final']:
            path = ROOT / 'models' / (model['name'] + '_' + stage + '.npz')
            assert sha(path) == model[stage + '_model_sha256']
            pp = replay(np.load(path), full_common, u, i, s)
            fm = np.load(path)
            active = counts['full'].sum(axis=1) > 0
            groups = fm['mapping'][active]
            group_sizes = np.bincount(groups, minlength=model['budget'])
            assert group_sizes.min() > 0 and group_sizes.max() <= np.ceil(2 * active.sum() / model['budget'])
            gn = np.stack([np.bincount(groups, weights=counts['full'][active, context], minlength=model['budget']) for context in range(15)], axis=1)
            gs = np.stack([np.bincount(groups, weights=positives['full'][active, context], minlength=model['budget']) for context in range(15)], axis=1)
            np.testing.assert_array_equal(((gs + 10 * full_common['context']) / (gn + 10)).astype('float32'), fm['group'])
            actual = optimizer_preds[model['name'] + '_' + stage]
            np.testing.assert_allclose(pp, actual, atol=6e-8, rtol=0)
            pr = np.clip(actual.astype('float64'), 1e-6, 1 - 1e-6)
            ll = -np.where(y == 1, np.log(pr), np.log1p(-pr))
            assert abs(ll.mean() - float(optimizer_csv[model['name']][stage + '_validation_loss'])) < 1e-12
            if stage == 'final':
                profiles = ((positives['full'] + 10 * full_common['context']) / (counts['full'] + 10)).astype('float32').astype('float64')
                prior = np.broadcast_to(full_common['context'], profiles.shape)
                profiles[counts['full'] < 10] = prior[counts['full'] < 10]
                profiles = profiles[active]
                centers = np.stack([np.bincount(groups, weights=profiles[:, context], minlength=model['budget']) / group_sizes for context in range(15)], axis=1)
                div = profiles * np.log(profiles / centers[groups]) + (1 - profiles) * np.log((1 - profiles) / (1 - centers[groups]))
                dispersion = np.stack([np.bincount(groups, weights=div[:, context], minlength=model['budget']) / group_sizes for context in range(15)], axis=1)
                context_weights = counts['full'].sum(axis=0) / counts['full'].sum()
                costs = dispersion @ context_weights if model['lambda'] is None else model['lambda'] * np.log(np.exp(dispersion / model['lambda']) @ context_weights)
                objective = float(group_sizes @ costs / group_sizes.sum())
                assert abs(objective - values[-1]) < 1e-10
    # Earlier E0 snapshot consistency for the four shared seed-11 cases.
    for budget in [64, 256]:
        for new_prefix, old_prefix in [('frequency', 'frequency'), ('tilted_L0.3', 'tilted')]:
            new_file = ROOT / 'models' / f'optimizer_{new_prefix}_B{budget}_s11_round12.npz'
            old_file = E0 / 'models' / (f'frequency_B{budget}.npz' if old_prefix == 'frequency' else f'tilted_B{budget}_L0.3.npz')
            np.testing.assert_array_equal(np.load(new_file)['mapping'], np.load(old_file)['mapping'])
    cc = r['pairing_results'][0]['contrasts']
    expected_conditions = {
        'adequate_support': r['pairing_results'][0]['matched_blocks'] >= 50 and r['pairing_results'][0]['validation_events'] >= 1000,
        'B_repeatability': cc['B_profile_repeatability']['adjusted_one_sided_lower'] > 0,
        'extra_sharing_damage': cc['high_minus_low']['estimate'] >= .001 and cc['high_minus_low']['adjusted_one_sided_lower'] > 0,
        'useful_grouping': cc['neutral_minus_low']['estimate'] >= .001 and cc['neutral_minus_low']['adjusted_one_sided_lower'] > 0,
        'adjusted_association': cc['adjusted_context_coefficient_per_feature_sd']['adjusted_one_sided_lower'] > 0,
        'direction_repeats_at_least_two_seeds': sum(z['contrasts']['B_profile_repeatability']['estimate'] > 0 and z['contrasts']['high_minus_low']['estimate'] > 0 and z['contrasts']['neutral_minus_low']['estimate'] > 0 for z in r['pairing_results']) >= 2}
    assert expected_conditions == r['gate_conditions']
    assert r['gate_pass'] == all(expected_conditions.values())
    result = {'status': 'independent_artifact_checks_passed', 'pair_predictors_replayed': len(m['pair_models']),
              'optimizer_snapshots_replayed': 2 * len(m['optimizer_models']), 'matched_seeds_checked': 3,
              'full_bootstrap_contrast_vectors_replayed': 12, 'independent_adjusted_regression_bootstraps': 15,
              'same_seed_round12_matches_previous_E0': True, 'equal_grouped_varying_bytes_by_seed': sizes,
              'gate_recomputed': r['gate_pass'], 'max_absolute_errors': errors,
              'limits': ['Native label-rule inconsistency remains unresolved.', 'Validation and train were previously used; no independent confirmatory evidence.', 'Matching and within-block regression do not prove intrinsic preferences or eliminate all item-exposure confounding.', 'Two-user groups are a mechanism diagnostic, not a severe-compression/neural benchmark.', 'Heuristic stalling is not certified optimality.'],
              'elapsed_seconds': time.perf_counter() - started}
    (ROOT / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
