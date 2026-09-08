"""Friedman test, Nemenyi post-hoc, chi-square CDF, and descriptives.

Correctness is established two ways rather than by citing a printed table:

* **Hand-computed cases** small enough to verify with arithmetic in the
  docstring, so the expected values are derivable here rather than taken on
  trust.
* **A cross-check against scipy** on larger random inputs, including a
  tie-heavy case, skipped automatically when scipy is absent.

The Nemenyi critical-difference constant ``q_alpha`` for k=3 at alpha=0.05 is
2.343, tabulated by Demsar (2006), "Statistical Comparisons of Classifiers over
Multiple Data Sets", JMLR 7:1-30, Table 5.
"""

from __future__ import annotations

import math
import unittest

from bopis.stats import (
    NEMENYI_Q,
    average_ranks,
    chi2_sf,
    coefficient_of_variation,
    compare_conditions,
    describe,
    friedman,
    nemenyi,
)


class TestChiSquareSurvival(unittest.TestCase):
    def test_df2_is_exact_closed_form(self) -> None:
        """At df=2 the survival function is exp(-x/2) exactly."""
        for x in (0.1, 1.0, 2.0, 5.991, 9.21, 20.0):
            self.assertAlmostEqual(chi2_sf(x, 2), math.exp(-x / 2.0), places=14)

    def test_critical_values(self) -> None:
        # Standard 0.05 critical values.
        self.assertAlmostEqual(chi2_sf(5.991, 2), 0.05, places=4)
        self.assertAlmostEqual(chi2_sf(7.815, 3), 0.05, places=4)
        self.assertAlmostEqual(chi2_sf(9.488, 4), 0.05, places=4)
        self.assertAlmostEqual(chi2_sf(3.841, 1), 0.05, places=4)

    def test_monotone_decreasing(self) -> None:
        values = [chi2_sf(x, 3) for x in (0.5, 1, 2, 5, 10, 20)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_bounds(self) -> None:
        self.assertEqual(chi2_sf(0.0, 2), 1.0)
        self.assertEqual(chi2_sf(-1.0, 2), 1.0)
        self.assertLess(chi2_sf(200.0, 2), 1e-40)

    def test_general_path_agrees_with_closed_form_at_df2(self) -> None:
        """The incomplete-gamma path must reproduce the df=2 special case."""
        from bopis.stats import _gamma_p_series, _gamma_q_continued_fraction

        for x in (0.5, 3.0, 12.0):
            half = x / 2.0
            general = (
                1.0 - _gamma_p_series(1.0, half)
                if half < 2.0
                else _gamma_q_continued_fraction(1.0, half)
            )
            self.assertAlmostEqual(general, math.exp(-half), places=10)


class TestRanking(unittest.TestCase):
    def test_lower_is_rank_one_by_default(self) -> None:
        self.assertEqual(average_ranks([10.0, 20.0, 30.0]), [1.0, 2.0, 3.0])

    def test_higher_is_rank_one_when_requested(self) -> None:
        self.assertEqual(
            average_ranks([10.0, 20.0, 30.0], higher_is_better=True), [3.0, 2.0, 1.0]
        )

    def test_ties_receive_the_average_rank(self) -> None:
        # Two values tied for ranks 1 and 2 both get 1.5.
        self.assertEqual(average_ranks([5.0, 5.0, 9.0]), [1.5, 1.5, 3.0])
        # Three-way tie across ranks 1-3 gives 2.0 each.
        self.assertEqual(average_ranks([7.0, 7.0, 7.0]), [2.0, 2.0, 2.0])

    def test_rank_sum_is_invariant(self) -> None:
        """Ranks of k items always sum to k(k+1)/2, ties or not."""
        for row in ([1.0, 2.0, 3.0], [4.0, 4.0, 9.0], [2.0, 2.0, 2.0]):
            self.assertAlmostEqual(sum(average_ranks(row)), 6.0, places=12)


class TestFriedman(unittest.TestCase):
    def test_hand_computed_perfect_separation(self) -> None:
        """A case small enough to verify with arithmetic on paper.

        Four blocks, three conditions, each block ordered 1 < 2 < 3::

            rank sums   R = [4, 8, 12]           sum R^2 = 224
            chi2 = 12/(n k (k+1)) * sum R^2 - 3 n (k+1)
                 = 12/(4*3*4) * 224 - 3*4*4
                 = 0.25 * 224 - 48
                 = 8.0
            p = exp(-chi2 / 2) = exp(-4)          [df = 2]

        8.0 is also the maximum attainable statistic for n=4, k=3, which is
        what perfect separation should produce.
        """
        blocks = [[1.0, 2.0, 3.0]] * 4
        result = friedman(blocks, ["a", "b", "c"])
        self.assertEqual(result.n_blocks, 4)
        self.assertEqual(result.df, 2)
        self.assertEqual(result.rank_sums, [4.0, 8.0, 12.0])
        self.assertAlmostEqual(result.chi_square, 8.0, places=12)
        self.assertAlmostEqual(result.p_value, math.exp(-4.0), places=14)

    def test_hand_computed_mixed_orderings(self) -> None:
        """A second paper-checkable case, this time not fully separated.

        Blocks ``[1,2,3], [2,1,3], [1,3,2], [3,1,2]`` are already rank rows::

            R = [7, 7, 10]                        sum R^2 = 198
            chi2 = 0.25 * 198 - 48 = 1.5
            p = exp(-0.75)
        """
        blocks = [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
            [1.0, 3.0, 2.0],
            [3.0, 1.0, 2.0],
        ]
        result = friedman(blocks, ["a", "b", "c"])
        self.assertEqual(result.rank_sums, [7.0, 7.0, 10.0])
        self.assertAlmostEqual(result.chi_square, 1.5, places=12)
        self.assertAlmostEqual(result.p_value, math.exp(-0.75), places=14)
        self.assertFalse(result.significant)

    def test_identical_conditions_give_zero_statistic(self) -> None:
        blocks = [[5.0, 5.0, 5.0] for _ in range(20)]
        result = friedman(blocks, ["a", "b", "c"])
        self.assertAlmostEqual(result.chi_square, 0.0, places=12)
        self.assertAlmostEqual(result.p_value, 1.0, places=12)
        self.assertFalse(result.significant)

    def test_perfectly_separated_conditions_are_significant(self) -> None:
        blocks = [[3.0, 2.0, 1.0] for _ in range(30)]
        result = friedman(blocks, ["a", "b", "c"])
        self.assertLess(result.p_value, 0.001)
        self.assertTrue(result.significant)
        # Mean ranks must be 3, 2, 1 for a perfectly consistent ordering.
        self.assertEqual(result.mean_ranks, [3.0, 2.0, 1.0])

    def test_orientation_flips_mean_ranks(self) -> None:
        blocks = [[1.0, 2.0, 3.0] for _ in range(10)]
        low = friedman(blocks, ["a", "b", "c"], higher_is_better=False)
        high = friedman(blocks, ["a", "b", "c"], higher_is_better=True)
        self.assertEqual(low.mean_ranks, [1.0, 2.0, 3.0])
        self.assertEqual(high.mean_ranks, [3.0, 2.0, 1.0])
        # The statistic itself is orientation-invariant.
        self.assertAlmostEqual(low.chi_square, high.chi_square, places=12)

    def test_tie_correction_is_applied(self) -> None:
        blocks = [[1.0, 1.0, 2.0], [2.0, 2.0, 1.0], [1.0, 2.0, 2.0]]
        result = friedman(blocks, ["a", "b", "c"])
        self.assertLess(result.tie_correction, 1.0)
        self.assertGreater(result.tie_correction, 0.0)

    def test_rejects_ragged_blocks(self) -> None:
        with self.assertRaises(ValueError):
            friedman([[1.0, 2.0], [1.0, 2.0, 3.0]], ["a", "b", "c"])

    def test_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            friedman([], ["a"])

    def test_rejects_single_condition(self) -> None:
        with self.assertRaises(ValueError):
            friedman([[1.0], [2.0]], ["only"])

    def test_matches_scipy_when_available(self) -> None:
        try:
            from scipy.stats import friedmanchisquare
        except ImportError:
            self.skipTest("scipy not installed (core does not need it)")
        import random

        rng = random.Random(17)
        blocks = [
            [rng.gauss(100, 12), rng.gauss(92, 12), rng.gauss(80, 12)]
            for _ in range(60)
        ]
        mine = friedman(blocks, ["a", "b", "c"])
        stat, p_value = friedmanchisquare(*list(zip(*blocks)))
        self.assertAlmostEqual(mine.chi_square, float(stat), places=9)
        self.assertAlmostEqual(mine.p_value, float(p_value), places=12)

    def test_matches_scipy_with_ties(self) -> None:
        try:
            from scipy.stats import friedmanchisquare
        except ImportError:
            self.skipTest("scipy not installed")
        blocks = [
            [1.0, 1.0, 2.0],
            [2.0, 2.0, 1.0],
            [1.0, 2.0, 2.0],
            [3.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [1.0, 3.0, 2.0],
        ]
        mine = friedman(blocks, ["a", "b", "c"])
        stat, p_value = friedmanchisquare(*list(zip(*blocks)))
        self.assertAlmostEqual(mine.chi_square, float(stat), places=9)
        self.assertAlmostEqual(mine.p_value, float(p_value), places=12)


class TestNemenyi(unittest.TestCase):
    def test_critical_difference_formula(self) -> None:
        """CD = q_alpha * sqrt(k(k+1) / 6n)."""
        blocks = [[3.0, 2.0, 1.0] for _ in range(100)]
        result = friedman(blocks, ["a", "b", "c"])
        post = nemenyi(result, alpha=0.05)
        expected = NEMENYI_Q[0.05][3] * math.sqrt(3 * 4 / (6.0 * 100))
        self.assertAlmostEqual(post.critical_difference, expected, places=12)

    def test_tabulated_constant_for_three_conditions(self) -> None:
        self.assertAlmostEqual(NEMENYI_Q[0.05][3], 2.343, places=3)

    def test_all_pairs_reported(self) -> None:
        blocks = [[3.0, 2.0, 1.0] for _ in range(40)]
        post = nemenyi(friedman(blocks, ["u", "r", "b"]))
        self.assertEqual(len(post.pairs), 3)
        labels = {pair["pair"] for pair in post.pairs}
        self.assertEqual(labels, {"u vs r", "u vs b", "r vs b"})

    def test_detects_separation(self) -> None:
        blocks = [[3.0, 2.0, 1.0] for _ in range(60)]
        post = nemenyi(friedman(blocks, ["a", "b", "c"]))
        self.assertEqual(len(post.significant_pairs()), 3)

    def test_finds_nothing_when_identical(self) -> None:
        blocks = [[1.0, 1.0, 1.0] for _ in range(60)]
        post = nemenyi(friedman(blocks, ["a", "b", "c"]))
        self.assertEqual(post.significant_pairs(), [])

    def test_critical_difference_shrinks_with_more_blocks(self) -> None:
        small = nemenyi(friedman([[3.0, 2.0, 1.0]] * 10, ["a", "b", "c"]))
        large = nemenyi(friedman([[3.0, 2.0, 1.0]] * 500, ["a", "b", "c"]))
        self.assertLess(large.critical_difference, small.critical_difference)

    def test_rejects_untabulated_alpha(self) -> None:
        result = friedman([[3.0, 2.0, 1.0]] * 10, ["a", "b", "c"])
        with self.assertRaises(ValueError):
            nemenyi(result, alpha=0.01)


class TestCompareConditions(unittest.TestCase):
    def test_runs_posthoc_only_when_significant(self) -> None:
        separated = compare_conditions(
            [[3.0, 2.0, 1.0]] * 40, ["a", "b", "c"]
        )
        self.assertIsNotNone(separated["nemenyi"])

        identical = compare_conditions(
            [[1.0, 1.0, 1.0]] * 40, ["a", "b", "c"]
        )
        self.assertIsNone(identical["nemenyi"])


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


if __name__ == "__main__":
    unittest.main()
