"""Synthetic scientific-contract checks for the local E7/E9 analyses."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("e7e9", Path(__file__).with_name("analyze_e7_e9.py"))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class ScientificContracts(unittest.TestCase):
    def test_weighted_denominator_is_standard_not_capped_recall(self):
        # One hit among 25 truths is 1/25, not the main metric's 1/20.
        edges = np.array([[0] * 25, list(range(25))])
        d = M.SplitIndex(edges, 1, 26, np.ones(26))
        m = d.metrics(np.array([0]), np.array([[0] + [25] * 19]))
        self.assertAlmostEqual(m["weighted_recall"], 1 / 25)
        self.assertAlmostEqual(m["standard_recall"], 1 / 25)
        self.assertAlmostEqual(m["capped_recall20"], 1 / 20)

    def test_weighted_numerator_and_denominator(self):
        weights = np.array([1., .5, .25, .1])
        d = M.SplitIndex(np.array([[0, 0], [0, 2]]), 1, 4, weights)
        m = d.metrics(np.array([0]), np.array([[2, 3]]))
        self.assertAlmostEqual(m["weighted_recall"], .25 / 1.25)

    def test_train_counts_ignore_validation_and_test(self):
        train = np.array([[0, 1, 1], [0, 0, 2]])
        np.testing.assert_array_equal(M.train_counts(train, 4), [2, 0, 1, 0])

    def test_quantization_codes_storage_and_error(self):
        p = np.log1p(np.array([0, 1, 3, 15, 10000]))
        q, decoded, info = M.quantize_u8(p)
        self.assertEqual(q.dtype, np.uint8)
        self.assertEqual(q.nbytes, 5)
        self.assertEqual(info["total_bytes_including_scale"], 9)
        self.assertLessEqual(np.max(np.abs(decoded - p)), float(p.max()) / 510 + 1e-6)
        self.assertEqual(int(q[0]), 0)
        self.assertEqual(int(q[-1]), 255)

    def test_all_zero_quantization_is_defined(self):
        q, decoded, info = M.quantize_u8(np.zeros(3))
        np.testing.assert_array_equal(q, np.zeros(3))
        np.testing.assert_array_equal(decoded, np.zeros(3))
        self.assertEqual(info["scale_float32"], 0)

    def test_beta_selected_on_validation_metric_only_and_smallest_tie(self):
        curves = {0.: {"weighted_recall": .2}, .1: {"weighted_recall": .3},
                  .2: {"weighted_recall": .3}}
        self.assertEqual(M.select_beta(curves), .1)

    def test_saved_mapping_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "mapping.npy"
            np.save(f, np.array([0, 0, 1]))
            np.testing.assert_array_equal(M.load_saved_mapping(f, M.array_hash([0, 0, 1])), [0, 0, 1])
            with self.assertRaises(ValueError):
                M.load_saved_mapping(f, "bad_hash")

    def test_lexicographic_ties_do_not_reorder_strict_scores(self):
        scores = np.array([[.5000002, .5, .5, .3]], dtype=np.float32)
        rank = np.array([3, 1, 0, 2])
        got = M.lex_topk(scores, rank, 3)
        np.testing.assert_array_equal(got, [[0, 2, 1]])


if __name__ == "__main__":
    unittest.main()
