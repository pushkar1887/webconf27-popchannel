"""E8 scientific-contract tests; no model scoring or network access."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np


class E8Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name('analyze_e8_final.py')
        if path.exists():
            spec = importlib.util.spec_from_file_location('analyze_e8_final', path)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)
        else:
            cls.module = None

    def api(self):
        self.assertIsNotNone(self.module, 'E8 retained analysis is not implemented')
        return self.module

    def fixture(self):
        # B reverses users: pairing by row would pair different people.
        a = dict(users=np.array([4, 2]), hits=np.array([2, 1]),
                 npos=np.array([40, 2]), ndcg=np.array([.8, .3]))
        b = dict(users=np.array([2, 4]), hits=np.array([[1, 1], [1, 0]]),
                 npos=np.array([2, 40]), ndcg=np.array([.5, .4]))
        return a, b

    def test_aligns_users_and_uses_capped_denominator_for_both_hit_shapes(self):
        a, b = self.fixture()
        result = self.api().paired_vectors(a, b)
        np.testing.assert_array_equal(result['users'], [2, 4])
        np.testing.assert_allclose(result['a'][:, 0], [.5, .1])
        np.testing.assert_allclose(result['b'][:, 0], [1., .05])
        np.testing.assert_allclose(result['a'][:, 1], [.3, .8])

    def test_rejects_different_user_coverage(self):
        a, b = self.fixture()
        b['users'] = np.array([2, 9])
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.api().paired_vectors(a, b)

    def test_rejects_duplicate_users(self):
        a, b = self.fixture()
        a['users'] = np.array([2, 2])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.api().paired_vectors(a, b)

    def test_rejects_same_users_with_different_truth_counts(self):
        a, b = self.fixture()
        b['npos'][0] = 3
        with self.assertRaisesRegex(ValueError, 'npos'):
            self.api().paired_vectors(a, b)

    def test_rejects_stored_recall_inconsistent_with_hits(self):
        a, b = self.fixture()
        a['recall'] = np.array([.9, .9])
        with self.assertRaisesRegex(ValueError, 'recall'):
            self.api().paired_vectors(a, b)

    def test_both_metrics_share_the_same_bootstrap_user_draws(self):
        e = self.api()
        d = np.array([[-.2, -.4], [.1, .2], [.3, .6], [.6, 1.2]])
        result = e.bootstrap_difference(d, n_resamples=121, seed=0)
        np.testing.assert_allclose(result['ci95'][:, 1], 2 * result['ci95'][:, 0])
        repeat = e.bootstrap_difference(d, n_resamples=121, seed=0)
        np.testing.assert_array_equal(result['ci95'], repeat['ci95'])
        np.testing.assert_array_equal(result['nonpositive_counts'], repeat['nonpositive_counts'])

    def test_holm_stepdown_preserves_order_and_monotonicity(self):
        np.testing.assert_allclose(self.api().holm_adjust([.01, .04, .03]), [.03, .06, .06])
        np.testing.assert_allclose(self.api().holm_adjust([.9, .8]), [1., 1.])

    def test_seed_spread_is_sample_sd_not_standard_error(self):
        result = self.api().seed_spread([1., 2., 3.])
        self.assertEqual(result['n_seeds'], 3)
        self.assertEqual(result['mean_percent'], 2.)
        self.assertEqual(result['sample_sd_pp'], 1.)

    def test_summary_reports_ttest_separately_from_bootstrap(self):
        a, b = self.fixture()
        result = self.api().paired_summary(a, b, n_resamples=100, seed=0)
        self.assertAlmostEqual(result['metrics']['recall']['delta_pp'], -22.5)
        self.assertAlmostEqual(result['metrics']['ndcg']['delta_pp'], 10.)
        self.assertIn('paired_t_p_two_sided', result['metrics']['recall'])
        self.assertNotIn('bootstrap_p', result['metrics']['recall'])

    def test_alternative_aggregate_does_not_manufacture_a_paired_ci(self):
        control = {'legacy_topk': {'raw': {'prior': {
            'best_beta': .2, 'test': {'recall': .04, 'ndcg': .03}}}},
            'full_prior': {'base': 'raw', 'best_beta': .2, 'test': {
                'recall': {'per_seed': [.04, .041]},
                'ndcg': {'per_seed': [.03, .031]}}}}
        result = self.api().amazon_aggregate_sensitivity(control, {'recall': .06, 'ndcg': .05})
        self.assertEqual(result['alternate_paired_ci_status'], 'missing_per_user_predictions')
        self.assertIsNone(result['legacy_topk_raw_prior']['ci95_pp'])
        self.assertAlmostEqual(result['legacy_topk_raw_prior']['metrics']['recall']['delta_pp'], 2.)
        self.assertTrue(result['random_key_seed_means']['recall']['all_deltas_positive'])

    def test_alternative_with_different_beta_is_marked_inapplicable(self):
        control = {'legacy_topk': {'raw': {'prior': {'best_beta': .5, 'test': {
            'recall': .04, 'ndcg': .03}}}}, 'full_prior': {'base': 'raw', 'best_beta': .2,
            'test': {'recall': {'per_seed': [.04]}, 'ndcg': {'per_seed': [.03]}}}}
        result = self.api().amazon_aggregate_sensitivity(control, {'recall': .06, 'ndcg': .05})
        self.assertFalse(result['legacy_topk_raw_prior']['applicable_same_base_beta'])


if __name__ == '__main__':
    unittest.main()
