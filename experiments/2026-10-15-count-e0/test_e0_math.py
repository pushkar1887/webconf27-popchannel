"""Independent numerical contracts for the exploratory E0 calculations."""
import math
import unittest

import numpy as np

from e0_math import (
    bernoulli_kl, capped_assignment, context_mix, distortion,
    fitted_group_table, reweighted_loss, tilted_cost,
)


class NumericalContracts(unittest.TestCase):
    def test_kl_catches_wrong_direction_or_missing_complement(self):
        self.assertAlmostEqual(float(bernoulli_kl(.9, .5)), .3680642071684971)
        np.testing.assert_allclose(bernoulli_kl([.1, .5, .9], [.1, .5, .9]), 0, atol=1e-14)

    def test_tilt_limits_and_stability(self):
        c, p = np.array([[0., 2.]]), np.array([.9, .1])
        self.assertAlmostEqual(float(tilted_cost(c, p, None)[0]), .2)
        self.assertGreater(float(tilted_cost(c, p, .3)[0]), .2)
        self.assertLess(float(tilted_cost(c, p, .3)[0]), 2.)
        self.assertAlmostEqual(float(tilted_cost(c, p, 1e7)[0]), .2, places=6)
        self.assertTrue(np.isfinite(tilted_cost(c, p, 1e-5)).all())

    def test_context_sweeps_use_training_distribution(self):
        p = np.array([.9, .1])
        np.testing.assert_allclose(context_mix(p, 0.), [.9, .1])
        np.testing.assert_allclose(context_mix(p, 1.), [.5, .5])
        np.testing.assert_allclose(context_mix(p, .25, target=1), [.675, .325])

    def test_group_dispersion_matches_hand_derived_bernoulli_kl(self):
        profiles = np.array([[.8, .9], [.8, .1]])
        objective, by_group, centers = distortion(profiles, np.array([0, 0]), 1, np.array([.9, .1]), None)
        np.testing.assert_allclose(centers, [[.8, .5]])
        np.testing.assert_allclose(by_group, [[0., .3680642071684971]], atol=1e-14)
        self.assertAlmostEqual(objective, .03680642071684971)

    def test_context_specific_damage_becomes_visible_when_upweighted(self):
        # Expected excess log loss: no damage in common context, .368064 in rare context.
        d = np.array([0., .3680642071684971])
        self.assertAlmostEqual(reweighted_loss(d, np.array([.9, .1])), .03680642071684971)
        self.assertAlmostEqual(reweighted_loss(d, np.array([.5, .5])), .18403210358424855)
        self.assertIsNone(reweighted_loss(np.array([.2, np.nan]), np.array([.5, .5])))
        self.assertAlmostEqual(reweighted_loss(np.array([.2, np.nan]), np.array([1., 0.])), .2)

    def test_capacity_enforcement_catches_collapsed_groups(self):
        costs = np.array([[0., 10., 20.]] * 6)
        g = capped_assignment(costs, 2)
        np.testing.assert_array_equal(np.bincount(g, minlength=3), [2, 2, 2])

    def test_identity_grouping_preserves_smoothed_count_predictions(self):
        n = np.array([[10., 2.], [3., 7.]])
        s = np.array([[8., 1.], [1., 6.]])
        expected = np.array([[.65, .5], [6 / 13, 11 / 17]])
        np.testing.assert_allclose(fitted_group_table(n, s, np.array([0, 1]), 2, np.array([.5, .5]), 10), expected)


if __name__ == '__main__':
    unittest.main()
