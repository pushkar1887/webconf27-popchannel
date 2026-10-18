"""Hand-derived contracts for the throwaway mechanism diagnostic."""
import unittest
import numpy as np

from mechanism_math import pairing_indices, select_pairings, expected_random_loss, fixed_effect_fit


class MechanismContracts(unittest.TestCase):
    def test_pairings_cover_each_user_once_and_choose_conflict_not_user_order(self):
        choices = pairing_indices()
        for choice in choices:
            self.assertEqual(sorted(choice.ravel().tolist()), [0, 1, 2, 3])
        low, high, score = select_pairings(np.array([[-2., -1., 1., 2.]]))
        self.assertEqual(int(low[0]), 0)
        self.assertEqual(int(high[0]), 2)
        np.testing.assert_allclose(score, [[1., 9., 10.]])

    def test_random_reference_is_expected_loss_not_ensemble_probability(self):
        losses = -np.log(np.array([[.9, .1, .5]]))
        self.assertAlmostEqual(float(expected_random_loss(losses)[0]), 1.033697596403939)
        self.assertGreater(float(expected_random_loss(losses)[0]), -np.log(.5))

    def test_block_effects_do_not_become_contextual_effect(self):
        x = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]] * 2)
        block = np.repeat([0, 1], 4)
        y = 2 * x[:, 0] + 3 * x[:, 1] + np.repeat([100., -50.], 4)
        beta = fixed_effect_fit(x, y, np.ones(8), block)
        np.testing.assert_allclose(beta, [2., 3.], atol=1e-10)


if __name__ == '__main__':
    unittest.main()
