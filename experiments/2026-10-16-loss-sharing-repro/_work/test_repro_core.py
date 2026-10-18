import importlib
import math
import unittest
import numpy as np
import torch

try:
    core = importlib.import_module("repro_core")
except ModuleNotFoundError:
    core = None


class ReproductionChecks(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(core, "reproduction core has not been implemented")

    def test_paper_recall_uses_capped_denominator(self):
        result = core.metrics_from_topk(np.array([[1, 4]]), [[1, 2, 3]], 2)
        self.assertAlmostEqual(result["recall"], .5)
        self.assertAlmostEqual(result["standard_recall"], 1/3)
        self.assertAlmostEqual(result["ndcg"], 1/(1+1/math.log2(3)))

    def test_repeated_ids_separated_from_extra_aliases(self):
        vectors = torch.tensor([[1., 0.], [-1., 0.]], dtype=torch.float64)
        result = core.pair_diagnostics(
            torch.tensor([0, 0, 1, 2]), torch.tensor([0, 0, 0, 1]), vectors)
        self.assertAlmostEqual(result["rho"], .5)
        self.assertAlmostEqual(result["rho_repeat"], 1/6)
        self.assertAlmostEqual(result["rho_alias"], 1/3)
        self.assertAlmostEqual(result["Z"], math.exp(-8))
        self.assertAlmostEqual(result["alpha"], math.exp(-8)/(1+math.exp(-8)))
        self.assertAlmostEqual(result["collision_share"], 1-result["alpha"])

    def test_all_collided_batch_has_no_usable_cross_pairs(self):
        result = core.pair_diagnostics(
            torch.tensor([0, 1]), torch.tensor([0, 0]), torch.tensor([[1., 0.]]))
        self.assertEqual(result["rho"], 1)
        self.assertEqual(result["alpha"], 0)
        self.assertIsNone(result["Z"])

    def test_masks_apply_to_raw_and_cosine_scoring(self):
        users = torch.tensor([[1., 0.]])
        items = torch.tensor([[10., 0.], [.8, .6], [1.4, math.sqrt(2.04)]])
        raw = core.rank_topk(users, items, [[0]], 1, cosine=False)
        cos = core.rank_topk(users, items, [[0]], 1, cosine=True)
        self.assertEqual(int(raw[0, 0]), 2)
        self.assertEqual(int(cos[0, 0]), 1)

    def test_timed_out_run_stops_at_ninety_minutes(self):
        self.assertFalse(core.expired(5399.9))
        self.assertTrue(core.expired(5400))

    def test_dataset_statistics_use_actual_split(self):
        parts = [torch.tensor([[0, 1], [0, 1]]),
                 torch.tensor([[0], [1]]), torch.tensor([[1], [2]])]
        result = core.split_statistics(parts)
        self.assertEqual(result["users"], 2)
        self.assertEqual(result["items"], 3)
        self.assertEqual(result["interactions"], 4)
        self.assertEqual(result["split_counts"], [2, 1, 1])

    def test_author_amazon_graphhash_settings_are_preserved(self):
        runtime = importlib.import_module("repro_run")
        self.assertTrue(hasattr(runtime, "settings_for"), "author settings resolver missing")
        settings = runtime.settings_for("AmazonBook", "GraphHash", "BPR")
        self.assertEqual(settings, (.01, 1e-6, 2984108, 1.))

    def test_directau_uses_minibatch_instead_of_full_bpr_batch(self):
        runtime = importlib.import_module("repro_run")
        self.assertTrue(hasattr(runtime, "settings_for"), "author settings resolver missing")
        self.assertEqual(runtime.settings_for("Gowalla", "Full", "DirectAU"),
                         (.01, 1e-8, 1024, 1.))

    def test_final_diagnostics_cannot_mark_overtime_run_complete(self):
        runtime = importlib.import_module("repro_run")
        self.assertTrue(hasattr(runtime, "mark_completed_before_deadline"), "final timing gate missing")
        row = {"status": "running"}
        with self.assertRaises(TimeoutError):
            runtime.mark_completed_before_deadline(row, 0.)
        self.assertNotEqual(row["status"], "complete")


if __name__ == "__main__":
    unittest.main()
