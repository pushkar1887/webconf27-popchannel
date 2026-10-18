"""Final E5 contract tests; synthetic rankings never stand in for study results."""
import unittest
from pathlib import Path
import numpy as np
import analyze_e5 as e5


class E5FinalTests(unittest.TestCase):
    def test_final_roster_covers_all_30_registered_lineages(self):
        self.assertTrue(hasattr(e5, 'final_roster'), 'final E5 roster is missing')
        roster = e5.final_roster(Path('.'))
        self.assertEqual(len(roster), 30)
        self.assertEqual(len({s['run_id'] for s in roster}), 30)
        self.assertEqual(sum(s['compression'] == 'graphhash' for s in roster), 12)
        self.assertEqual(sum(s['compression'] in ('random', 'dfreq') for s in roster), 12)
        self.assertEqual(sum(s['compression'] == 'full' for s in roster), 6)
        ids = {s['run_id'] for s in roster}
        self.assertIn('gowalla__mf__graphhash__bpr__s0-p500', ids)
        self.assertNotIn('gowalla__mf__graphhash__bpr__s0', ids)
        self.assertIn('gowalla__mf__full__bpr__s0-long', ids)
        self.assertNotIn('gowalla__mf__full__bpr__s0', ids)
        self.assertIn('amazonbook__mf__graphhash__directau__s0-initmatch', ids)

    def test_legacy_cosine_is_explicit_missing_not_substituted(self):
        self.assertTrue(hasattr(e5, 'final_mode_sources'), 'final mode source policy is missing')
        spec = {'folder': Path('legacy'), 'loss': 'bpr', 'kind': 'legacy'}
        modes = e5.final_mode_sources(spec)
        self.assertEqual(set(modes), {'raw', 'cosine', 'raw_prior'})
        self.assertIsNone(modes['cosine'])
        self.assertEqual(modes['raw'].name, 'per_user__base_randomkey_s0.npz')
        self.assertEqual(modes['raw_prior'].name, 'per_user__full_prior_s0.npz')

    @staticmethod
    def fixture():
        return dict(truths={0: np.array([1, 2, 3]), 1: np.array([0])},
                    seen={0: np.array([4]), 1: np.array([4])},
                    counts=np.array([10, 8, 6, 4, 2]), activity=np.array([1, 1]))

    def test_final_denominators_keep_conditional_and_contribution_distinct(self):
        self.assertTrue(hasattr(e5, 'prepare_final_dataset'), 'cached original-truth indexes missing')
        data = self.fixture()
        prepared = e5.prepare_final_dataset(data)
        a = dict(users=np.array([1, 0]), top20=np.array([[0, 3], [1, 2]]),
                 npos=np.array([1, 3]), hits=np.array([[1, 0], [1, 1]]))
        stats, hit = e5.validate_predictions(a, data['truths'], data['counts'], k=2,
                                            seen=data['seen'], prepared=prepared, return_hits=True)
        result, per_user = e5.final_metric_summary(a, hit, data, prepared, k=2)
        self.assertEqual(stats['truth_hit_lineage'], 'validated')
        self.assertAlmostEqual(result['tail_primary'], 1.0)
        self.assertAlmostEqual(result['tail_standard'], 2 / 3)
        self.assertAlmostEqual(result['tail_contribution'], 0.5)
        self.assertAlmostEqual(result['catalog_coverage'], 0.8)
        self.assertEqual(result['tail_eligible_users'], 1)
        self.assertEqual(per_user['users'].tolist(), [0])

    def test_paired_ci_aligns_ids_and_rejects_cohort_mismatch(self):
        self.assertTrue(hasattr(e5, 'paired_tail_comparison'), 'paired tail CI missing')
        variant = dict(users=np.array([2, 0, 1]), values=np.array([0.2, 0.2, 0.2]))
        full = dict(users=np.array([0, 1, 2]), values=np.array([0.3, 0.3, 0.3]))
        result = e5.paired_tail_comparison(variant, full, n_resamples=2000, seed=0)
        self.assertAlmostEqual(result['delta'], -0.1)
        self.assertTrue(all(abs(x + 0.1) < 1e-12 for x in result['ci95']))
        self.assertEqual(result['verdict'], 'PASS')
        self.assertEqual(result['n_paired_users'], 3)
        with self.assertRaisesRegex(ValueError, 'eligible'):
            e5.paired_tail_comparison(variant, dict(users=np.array([0, 1]), values=np.zeros(2)))

    def test_fast_lineage_validation_rejects_seen_and_wrong_hits(self):
        self.assertTrue(hasattr(e5, 'prepare_final_dataset'), 'cached original-truth indexes missing')
        data = self.fixture()
        prepared = e5.prepare_final_dataset(data)
        a = dict(users=np.array([0, 1]), top20=np.array([[1, 4], [0, 3]]),
                 npos=np.array([3, 1]), hits=np.array([[1, 0], [1, 0]]))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            e5.validate_predictions(a, data['truths'], data['counts'], k=2,
                                    seen=data['seen'], prepared=prepared)
        a['top20'][0, 1] = 2
        with self.assertRaisesRegex(ValueError, 'hits'):
            e5.validate_predictions(a, data['truths'], data['counts'], k=2,
                                    seen=data['seen'], prepared=prepared)


if __name__ == '__main__':
    unittest.main()
