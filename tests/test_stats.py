"""Wilcoxon signed-rank test, the two-condition comparison, and descriptives.

The study compares two conditions -- the random-search baseline and the
BOPIS-recommended configuration -- on shared prompts, so the inferential test is
the Wilcoxon signed-rank test. (A three-condition ordering test is not used, so
neither the Friedman test nor a Nemenyi post-hoc exists in :mod:`bopis.stats`.)

Correctness is established two ways rather than by citing a printed table:

* **Hand-computed cases** small enough to verify with arithmetic in the
  docstring, so the expected values are derivable here rather than taken on
  trust.
* **A cross-check against scipy** on larger random inputs, including a
  tie-heavy case, skipped automatically when scipy is absent.
"""

from __future__ import annotations

import math
import unittest

from bopis.stats import (
    bootstrap_ci,
    coefficient_of_variation,
    compare_conditions,
    describe,
    wilcoxon,
)


class TestWilcoxon(unittest.TestCase):
    def test_hand_computed_all_positive_differences(self) -> None:
        """Four blocks, every first condition larger, distinct magnitudes.

        Differences are 1, 2, 3, 4, so their ranks are 1, 2, 3, 4 and
        ``W- = 0`` (no negative ranks), giving ``W = 0``. With n=4::

            mean W = n(n+1)/4 = 5
            var W  = n(n+1)(2n+1)/24 = 7.5
            z = (0 - 5 + 0.5) / sqrt(7.5)     [continuity correction]
            p = erfc(|z| / sqrt(2))
        """
        result = wilcoxon([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
        self.assertEqual(result.n_effective, 4)
        self.assertEqual(result.w_statistic, 0.0)
        self.assertEqual(result.tie_correction, 0.0)
        expected_z = (0.0 - 5.0 + 0.5) / math.sqrt(7.5)
        self.assertAlmostEqual(result.z_statistic, expected_z, places=12)
        self.assertAlmostEqual(
            result.p_value, math.erfc(abs(expected_z) / math.sqrt(2.0)), places=14
        )
        self.assertFalse(result.significant)

    def test_large_consistent_effect_is_significant(self) -> None:
        blocks = [[float(i), 0.0] for i in range(1, 21)]  # every diff positive
        result = wilcoxon(blocks)
        self.assertLess(result.p_value, 0.001)
        self.assertTrue(result.significant)

    def test_zero_differences_are_dropped(self) -> None:
        # One tied pair (diff 0) contributes no sign information.
        result = wilcoxon([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
        self.assertEqual(result.n_effective, 2)

    def test_all_ties_is_not_significant(self) -> None:
        result = wilcoxon([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
        self.assertEqual(result.n_effective, 0)
        self.assertEqual(result.w_statistic, 0.0)
        self.assertEqual(result.p_value, 1.0)
        self.assertFalse(result.significant)

    def test_tie_correction_is_applied(self) -> None:
        # Two tied absolute differences of size 1 → one tie group of size 2.
        result = wilcoxon([[2.0, 1.0], [2.0, 1.0], [5.0, 1.0]])
        self.assertGreater(result.tie_correction, 0.0)

    def test_p_value_is_in_the_unit_interval(self) -> None:
        for blocks in (
            [[1.0, 2.0], [2.0, 3.0], [3.0, 5.0]],
            [[10.0, 1.0], [9.0, 2.0], [8.0, 3.0], [7.0, 4.0]],
        ):
            result = wilcoxon(blocks)
            self.assertGreaterEqual(result.p_value, 0.0)
            self.assertLessEqual(result.p_value, 1.0)

    def test_orientation_is_symmetric(self) -> None:
        """Swapping the two columns flips z but not W or the p-value."""
        forward = wilcoxon([[1.0, 3.0], [2.0, 2.0], [4.0, 1.0], [5.0, 6.0]])
        reverse = wilcoxon([[3.0, 1.0], [2.0, 2.0], [1.0, 4.0], [6.0, 5.0]])
        self.assertEqual(forward.w_statistic, reverse.w_statistic)
        self.assertAlmostEqual(forward.p_value, reverse.p_value, places=12)
        self.assertAlmostEqual(forward.z_statistic, -reverse.z_statistic, places=12)

    def test_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            wilcoxon([])

    def test_rejects_three_conditions(self) -> None:
        with self.assertRaises(ValueError):
            wilcoxon([[1.0, 2.0, 3.0]])

    def test_matches_scipy_when_available(self) -> None:
        try:
            from scipy.stats import wilcoxon as scipy_wilcoxon
        except ImportError:
            self.skipTest("scipy not installed (core does not need it)")
        import random

        rng = random.Random(7)
        x = [round(rng.gauss(100, 15), 1) for _ in range(40)]
        y = [round(v + rng.gauss(0, 8), 1) for v in x]
        mine = wilcoxon(list(zip(x, y)))
        reference = scipy_wilcoxon(
            x, y, zero_method="wilcox", correction=True, method="asymptotic"
        )
        self.assertAlmostEqual(mine.w_statistic, float(reference.statistic), places=9)
        self.assertAlmostEqual(mine.p_value, float(reference.pvalue), places=12)

    def test_matches_scipy_with_ties(self) -> None:
        try:
            from scipy.stats import wilcoxon as scipy_wilcoxon
        except ImportError:
            self.skipTest("scipy not installed")
        blocks = [
            [1.0, 1.0], [2.0, 2.0], [3.0, 2.0],
            [2.0, 3.0], [1.0, 2.0], [3.0, 1.0],
        ]
        mine = wilcoxon(blocks)
        reference = scipy_wilcoxon(
            [b[0] for b in blocks], [b[1] for b in blocks],
            zero_method="wilcox", correction=True, method="asymptotic",
        )
        self.assertAlmostEqual(mine.w_statistic, float(reference.statistic), places=9)
        self.assertAlmostEqual(mine.p_value, float(reference.pvalue), places=12)


class TestCompareConditions(unittest.TestCase):
    def test_compares_random_search_against_bopis_only(self) -> None:
        """The unoptimized reference column is ignored, not tested."""
        # columns: [unoptimized, random_search, bopis]
        blocks = [[100.0, 10.0, 1.0] for _ in range(30)]
        payload = compare_conditions(blocks, ["unoptimized", "random_search", "bopis"])
        self.assertIn("wilcoxon", payload)
        self.assertNotIn("friedman", payload)
        self.assertNotIn("nemenyi", payload)
        self.assertEqual(payload["conditions"], ["random_search", "bopis"])
        self.assertEqual(payload["n_blocks"], 30)
        self.assertTrue(payload["wilcoxon"]["significant"])

    def test_identical_conditions_are_not_significant(self) -> None:
        blocks = [[7.0, 7.0, 7.0] for _ in range(30)]
        payload = compare_conditions(blocks, ["unoptimized", "random_search", "bopis"])
        self.assertFalse(payload["wilcoxon"]["significant"])
        self.assertEqual(payload["wilcoxon"]["p_value"], 1.0)

    def test_indexes_by_name_not_position(self) -> None:
        """Ordering of the condition list must not silently pick the wrong pair."""
        names = ["bopis", "unoptimized", "random_search"]
        # bopis is column 0 (value 1), random_search is column 2 (value 10):
        # the same 9-unit gap, so it stays significant regardless of position.
        blocks = [[1.0, 100.0, 10.0] for _ in range(30)]
        payload = compare_conditions(blocks, names)
        self.assertTrue(payload["wilcoxon"]["significant"])


class TestDescriptives(unittest.TestCase):
    def test_basic_statistics(self) -> None:
        stats = describe([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        self.assertEqual(stats.n, 8)
        self.assertAlmostEqual(stats.mean, 5.0, places=12)
        self.assertAlmostEqual(stats.minimum, 2.0, places=12)
        self.assertAlmostEqual(stats.maximum, 9.0, places=12)
        # Sample SD (n-1 denominator) of that classic sequence.
        self.assertAlmostEqual(stats.sd, 2.13808993529939, places=10)

    def test_single_value_has_zero_sd(self) -> None:
        stats = describe([42.0])
        self.assertEqual(stats.n, 1)
        self.assertEqual(stats.sd, 0.0)

    def test_empty_yields_nan(self) -> None:
        stats = describe([])
        self.assertEqual(stats.n, 0)
        self.assertTrue(math.isnan(stats.mean))

    def test_ignores_none_and_nan(self) -> None:
        stats = describe([1.0, None, 3.0, float("nan")])  # type: ignore[list-item]
        self.assertEqual(stats.n, 2)
        self.assertAlmostEqual(stats.mean, 2.0, places=12)

    def test_coefficient_of_variation(self) -> None:
        self.assertAlmostEqual(
            coefficient_of_variation([10.0, 10.0, 10.0]), 0.0, places=12
        )
        self.assertGreater(coefficient_of_variation([8.0, 10.0, 12.0]), 0.0)


class TestBootstrapCI(unittest.TestCase):
    def test_point_estimate_matches_simple_mean(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = bootstrap_ci(values, n_boot=200, seed=1)
        self.assertAlmostEqual(result["point"], 3.0, places=12)

    def test_degenerate_samples_give_nan(self) -> None:
        result = bootstrap_ci([], n_boot=50)
        self.assertTrue(math.isnan(result["point"]))
        self.assertTrue(math.isnan(result["ci_lo"]))
        self.assertTrue(math.isnan(result["ci_hi"]))
        self.assertTrue(math.isnan(result["se"]))

    def test_confidence_interval_brackets_point(self) -> None:
        values = [2.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        result = bootstrap_ci(values, n_boot=1000, seed=42)
        self.assertLessEqual(result["ci_lo"], result["point"])
        self.assertGreaterEqual(result["ci_hi"], result["point"])

    def test_wider_confidence_gives_wider_interval(self) -> None:
        values = [2.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        narrow = bootstrap_ci(values, confidence=0.80, n_boot=1000, seed=42)
        wide = bootstrap_ci(values, confidence=0.95, n_boot=1000, seed=42)
        self.assertGreaterEqual(
            wide["ci_hi"] - wide["ci_lo"],
            narrow["ci_hi"] - narrow["ci_lo"],
        )

    def test_reproducible_with_same_seed(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        a = bootstrap_ci(values, n_boot=500, seed=7)
        b = bootstrap_ci(values, n_boot=500, seed=7)
        self.assertEqual(a["ci_lo"], b["ci_lo"])
        self.assertEqual(a["ci_hi"], b["ci_hi"])

    def test_ignores_none_and_nan_values(self) -> None:
        values = [1.0, None, 3.0, float("nan"), 5.0]  # type: ignore[list-item]
        result = bootstrap_ci(values, n_boot=500, seed=3)
        self.assertAlmostEqual(result["point"], 3.0, places=12)


if __name__ == "__main__":
    unittest.main()
