"""Independent numerical contracts for the exploratory exposure/MF pilot."""
import unittest
import numpy as np

from exposure_math import loss, user_means, user_auc, fit_intercept, calibration_if, matched_capacity


class ExposureContracts(unittest.TestCase):
    def test_logistic_excess_risk_matches_bernoulli_kl(self):
        eta = .2
        comp = np.log(.8 / .2)
        full = np.log(.2 / .8)
        excess = eta * (loss(np.array([1]), np.array([comp])) - loss(np.array([1]), np.array([full])))
        excess += (1 - eta) * (loss(np.array([0]), np.array([comp])) - loss(np.array([0]), np.array([full])))
        self.assertAlmostEqual(float(excess[0]), .8317766166719344)

    def test_user_standardization_does_not_reweight_to_heavy_users(self):
        values = np.array([2.] + [0.] * 9)
        uid = np.array([0] + [1] * 9)
        self.assertAlmostEqual(float(user_means(values, uid, 2).mean()), 1.)

    def test_auc_ties_and_user_offsets(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        score = np.array([.1, .5, .5, .9, 10., 9.])
        uid = np.array([0, 0, 0, 0, 1, 1])
        actual = user_auc(y, score, uid, 2)
        np.testing.assert_allclose(actual, [.875, 0.])
        np.testing.assert_allclose(user_auc(y, score + np.where(uid == 0, 100., -100.), uid, 2), actual)

    def test_intercept_optimum_and_first_order_calibration_uncertainty(self):
        y, score, uid = np.array([0, 0, 0, 1]), np.zeros(4), np.arange(4)
        shift, g, h = fit_intercept(y, score, uid, 4)
        self.assertAlmostEqual(shift, -1.0986122886681098, places=9)
        np.testing.assert_allclose(g, [.25, .25, .25, -.75], atol=1e-10)
        eps = 1e-4
        weights = np.array([[1., 1., 1., 1. + eps]])
        actual = float(calibration_if(weights, g, h)[0])
        self.assertLess(abs(actual - np.log1p(eps)), 1e-7)

    def test_capacity_control_differs_by_less_than_one_latent_row(self):
        b, lowrank_parameters, shared_parameters = matched_capacity(24837, 16, 2)
        self.assertEqual(b, 3107)
        self.assertEqual(lowrank_parameters, 49706)
        self.assertEqual(shared_parameters, 49712)
        self.assertLess(shared_parameters - lowrank_parameters, 16)


if __name__ == '__main__':
    unittest.main()
