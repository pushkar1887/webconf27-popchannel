"""Synthetic E5 tests: count gross hit recovery, avoid net-share mistakes."""
import unittest

import numpy as np

from analyze_e5 import analyze_pair, item_strata, activity_quartiles, validate_predictions


class E5Tests(unittest.TestCase):
    def test_gross_additions_and_losses_are_not_net_gain(self):
        # One newly recovered head hit and one lost tail hit: zero net gain.
        users = np.array([0, 1])
        truths = {0: np.array([0, 1]), 1: np.array([1])}
        base = np.array([[1, 3], [1, 3]])
        prior = np.array([[0, 1], [2, 3]])
        result = analyze_pair(users, base, prior, truths, np.array([10, 8, 6, 4, 2]),
                              np.array([2, 3]), k=2)
        self.assertEqual(result['hit_changes']['gross_added'], 1)
        self.assertEqual(result['hit_changes']['gross_lost'], 1)
        self.assertEqual(result['hit_changes']['net'], 0)
        self.assertEqual(result['hit_changes']['head_share_of_gross_added'], 1.0)
        self.assertEqual(result['hit_changes']['head_lost'], 0)
        self.assertEqual(result['hit_changes']['tail_lost'], 1)

    def test_empty_tail_users_excluded_from_conditional_not_overall(self):
        users = np.array([0, 1])
        truths = {0: np.array([0, 1, 2]), 1: np.array([0])}
        base = np.array([[1, 3], [0, 3]])
        prior = np.array([[1, 2], [0, 3]])
        result = analyze_pair(users, base, prior, truths, np.array([10, 8, 6, 4, 2]),
                              np.array([2, 3]), k=2)
        tail = result['tail']
        self.assertEqual(tail['eligible_users'], 1)
        self.assertAlmostEqual(tail['conditional_capped_recall']['delta'], 0.5)
        self.assertAlmostEqual(tail['overall_capped_recall_contribution']['delta'], 0.25)
        self.assertAlmostEqual(result['overall']['capped_recall']['delta'], 0.25)

    def test_standard_tail_recall_uses_all_tail_truth(self):
        result = analyze_pair(np.array([0]), np.array([[1, 4]]), np.array([[1, 2]]),
                              {0: np.array([1, 2, 3])}, np.array([10, 8, 6, 4, 2]),
                              np.array([3]), k=2)
        self.assertAlmostEqual(result['tail']['conditional_capped_recall']['delta'], 0.5)
        self.assertAlmostEqual(result['tail']['conditional_standard_recall']['delta'], 1 / 3)

    def test_tied_item_counts_have_deterministic_id_order(self):
        head, quintiles = item_strata(np.ones(7), head_fraction=0.2)
        self.assertEqual(np.flatnonzero(head).tolist(), [0, 1])
        self.assertEqual(quintiles.tolist(), [1, 1, 2, 2, 3, 4, 5])

    def test_activity_ties_are_broken_by_original_user_id(self):
        users = np.array([7, 3, 1, 9, 5])
        assignments = activity_quartiles(users, np.ones(10))
        # Sorted IDs [1,3] Q1, [5] Q2, [7] Q3, [9] Q4.
        self.assertEqual(assignments.tolist(), [3, 1, 1, 4, 2])

    def test_prediction_lineage_checks_counts_and_per_rank_hits(self):
        truth = {2: np.array([1, 3])}
        counts = np.array([1, 0, 0, 0])
        arrays = dict(users=np.array([2]), top20=np.array([[3, 0]]),
                      hits=np.array([[1, 0]]), npos=np.array([2]))
        stats = validate_predictions(arrays, truth, counts, k=2)
        self.assertEqual(stats['n_users'], 1)
        arrays['hits'] = np.array([[0, 1]])
        with self.assertRaisesRegex(ValueError, 'hits'):
            validate_predictions(arrays, truth, counts, k=2)
        arrays['hits'] = np.array([1])
        arrays['npos'] = np.array([3])
        with self.assertRaisesRegex(ValueError, 'npos'):
            validate_predictions(arrays, truth, counts, k=2)

    def test_duplicate_recommendations_are_rejected(self):
        arrays = dict(users=np.array([2]), top20=np.array([[1, 1]]),
                      hits=np.array([2]), npos=np.array([1]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_predictions(arrays, {2: np.array([1])}, np.ones(3), k=2)


if __name__ == '__main__':
    unittest.main()
