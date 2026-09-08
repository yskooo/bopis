"""Success indicators, surrogate reliability, convergence, and amortization.

Expected values are hand-computed in the test bodies so each formula is checked
against arithmetic rather than against itself.
"""

from __future__ import annotations

import math
import unittest

from bopis import metrics
from bopis.metrics import (
    Amortization,
    amortization,
    best_so_far,
    energy_improvement_ratio,
    improvement_per_iteration,
    iterations_to_within,
    joules_per_token,
    q_min,
    quality_retention_ratio,
    s_min,
    sample_efficiency_ratio,
    ser_summary,
    speed_retention_ratio,
    success_indicators,
    surrogate_reliability,
)


class TestRatios(unittest.TestCase):
    def test_eir(self) -> None:
        # (100 - 60) / 100 = 40%
        self.assertAlmostEqual(energy_improvement_ratio(100.0, 60.0), 40.0, places=12)

    def test_eir_is_negative_when_energy_increases(self) -> None:
        self.assertAlmostEqual(
            energy_improvement_ratio(100.0, 130.0), -30.0, places=12
        )

    def test_eir_is_zero_for_no_change(self) -> None:
        self.assertAlmostEqual(energy_improvement_ratio(100.0, 100.0), 0.0, places=12)

    def test_srr(self) -> None:
        # 19 / 20 = 95%
        self.assertAlmostEqual(speed_retention_ratio(20.0, 19.0), 95.0, places=12)

    def test_srr_above_100_when_faster(self) -> None:
        self.assertAlmostEqual(speed_retention_ratio(20.0, 30.0), 150.0, places=12)

    def test_qrr(self) -> None:
        # 0.784 / 0.8 = 98%
        self.assertAlmostEqual(quality_retention_ratio(0.8, 0.784), 98.0, places=10)

    def test_zero_baseline_is_handled(self) -> None:
        self.assertEqual(energy_improvement_ratio(0.0, 10.0), 0.0)
        self.assertEqual(speed_retention_ratio(0.0, 10.0), 0.0)
        self.assertEqual(quality_retention_ratio(0.0, 0.5), 0.0)


class TestThresholdDerivations(unittest.TestCase):
    def test_s_min_is_95_percent_of_baseline(self) -> None:
        """Amendment A-26: S_min was referenced but never defined."""
        self.assertAlmostEqual(s_min(20.0), 19.0, places=12)

    def test_q_min_is_98_percent_of_baseline(self) -> None:
        """Amendment A-25: Q_min(task) was referenced but never defined."""
        self.assertAlmostEqual(q_min(0.80), 0.784, places=12)

    def test_derived_thresholds_agree_with_the_ratio_criteria(self) -> None:
        """A value exactly at S_min must score exactly SRR = 95%."""
        baseline = 27.5
        self.assertAlmostEqual(
            speed_retention_ratio(baseline, s_min(baseline)),
            metrics.SRR_THRESHOLD,
            places=10,
        )
        baseline_q = 0.83
        self.assertAlmostEqual(
            quality_retention_ratio(baseline_q, q_min(baseline_q)),
            metrics.QRR_THRESHOLD,
            places=10,
        )


class TestSuccessIndicators(unittest.TestCase):
    def make(self, energy=60.0, speed=20.0, quality=0.80):
        return success_indicators(
            energy_default=100.0,
            energy_optimized=energy,
            speed_default=20.0,
            speed_optimized=speed,
            quality_default=0.80,
            quality_optimized=quality,
        )

    def test_all_criteria_met_is_optimal(self) -> None:
        indicators = self.make()
        self.assertTrue(indicators.is_bopis_optimal)
        self.assertEqual(indicators.failed_criteria(), [])

    def test_eir_at_or_above_15_is_substantial(self) -> None:
        self.assertEqual(self.make(energy=85.0).verdict, "substantial")
        self.assertEqual(self.make(energy=95.0).verdict, "optimal")

    def test_energy_increase_is_negative_verdict(self) -> None:
        indicators = self.make(energy=120.0)
        self.assertFalse(indicators.is_bopis_optimal)
        self.assertEqual(indicators.verdict, "negative")
        self.assertFalse(indicators.energy_improved)

    def test_speed_failure_is_partial(self) -> None:
        indicators = self.make(energy=50.0, speed=10.0)
        self.assertEqual(indicators.verdict, "partial")
        self.assertFalse(indicators.speed_retained)
        self.assertTrue(any("SRR" in f for f in indicators.failed_criteria()))

    def test_quality_failure_is_partial(self) -> None:
        indicators = self.make(energy=50.0, quality=0.60)
        self.assertEqual(indicators.verdict, "partial")
        self.assertFalse(indicators.quality_retained)
        self.assertTrue(any("QRR" in f for f in indicators.failed_criteria()))

    def test_thresholds_are_inclusive(self) -> None:
        """Exactly 95% SRR and exactly 98% QRR must pass."""
        indicators = success_indicators(
            energy_default=100.0,
            energy_optimized=90.0,
            speed_default=20.0,
            speed_optimized=19.0,
            quality_default=0.80,
            quality_optimized=0.784,
        )
        self.assertTrue(indicators.speed_retained)
        self.assertTrue(indicators.quality_retained)
        self.assertTrue(indicators.is_bopis_optimal)

    def test_a_large_energy_win_cannot_excuse_a_quality_collapse(self) -> None:
        """The whole point of QRR: 90% energy saved is still not success."""
        indicators = self.make(energy=10.0, quality=0.30)
        self.assertGreater(indicators.eir_percent, 80.0)
        self.assertFalse(indicators.is_bopis_optimal)

    def test_serializes_with_the_verdict(self) -> None:
        payload = self.make().as_dict()
        self.assertIn("verdict", payload)
        self.assertIn("is_bopis_optimal", payload)
        self.assertIn("failed_criteria", payload)


class TestSurrogateReliability(unittest.TestCase):
    def test_perfect_prediction(self) -> None:
        values = [10.0, 20.0, 30.0]
        result = surrogate_reliability(values, values)
        self.assertAlmostEqual(result.mae, 0.0, places=12)
        self.assertAlmostEqual(result.npe_percent, 0.0, places=12)
        self.assertAlmostEqual(result.r_squared, 1.0, places=12)
        self.assertTrue(result.reliable)

    def test_hand_computed_mae_and_npe(self) -> None:
        # errors 1, 2, 3 -> MAE = 2; mean measured = 20 -> NPE = 10%
        predicted = [11.0, 18.0, 33.0]
        measured = [10.0, 20.0, 30.0]
        result = surrogate_reliability(predicted, measured)
        self.assertAlmostEqual(result.mae, 2.0, places=12)
        self.assertAlmostEqual(result.mean_measured, 20.0, places=12)
        self.assertAlmostEqual(result.npe_percent, 10.0, places=12)

    def test_npe_threshold_is_exclusive(self) -> None:
        """NPE of exactly 10% is not below the threshold."""
        result = surrogate_reliability([11.0, 18.0, 33.0], [10.0, 20.0, 30.0])
        self.assertAlmostEqual(result.npe_percent, 10.0, places=12)
        self.assertFalse(result.reliable)

    def test_r_squared_is_zero_for_a_mean_predictor(self) -> None:
        measured = [10.0, 20.0, 30.0]
        mean = sum(measured) / len(measured)
        result = surrogate_reliability([mean] * 3, measured)
        self.assertAlmostEqual(result.r_squared, 0.0, places=12)

    def test_r_squared_is_negative_when_worse_than_the_mean(self) -> None:
        result = surrogate_reliability([30.0, 20.0, 10.0], [10.0, 20.0, 30.0])
        self.assertLess(result.r_squared, 0.0)

    def test_ucr_counts_coverage(self) -> None:
        """Amendment A-28: UCR is the share inside mu +/- 1.96 sigma."""
        predicted = [10.0, 10.0, 10.0, 10.0]
        measured = [10.0, 11.0, 12.0, 100.0]
        sigmas = [1.0, 1.0, 1.0, 1.0]  # 1.96 sigma = 1.96
        result = surrogate_reliability(predicted, measured, sigmas)
        # Covered: 10 (err 0), 11 (err 1). Not: 12 (err 2), 100.
        self.assertAlmostEqual(result.ucr, 0.5, places=12)

    def test_ucr_is_nan_without_sigmas(self) -> None:
        result = surrogate_reliability([1.0, 2.0], [1.0, 2.0])
        self.assertTrue(math.isnan(result.ucr))

    def test_empty_input_is_nan_not_a_crash(self) -> None:
        result = surrogate_reliability([], [])
        self.assertEqual(result.n, 0)
        self.assertTrue(math.isnan(result.mae))

    def test_rmse_penalizes_outliers_more_than_mae(self) -> None:
        result = surrogate_reliability([10.0, 10.0], [10.0, 30.0])
        self.assertGreater(result.rmse, result.mae)


class TestConvergence(unittest.TestCase):
    def test_best_so_far_is_a_running_minimum(self) -> None:
        self.assertEqual(
            best_so_far([100.0, 120.0, 80.0, 90.0, 70.0]),
            [100.0, 100.0, 80.0, 80.0, 70.0],
        )

    def test_best_so_far_is_non_increasing(self) -> None:
        curve = best_so_far([50.0, 60.0, 40.0, 45.0, 30.0, 35.0])
        for earlier, later in zip(curve, curve[1:]):
            self.assertLessEqual(later, earlier)

    def test_delta_energy(self) -> None:
        # best-so-far: 100, 100, 80, 80, 70 -> deltas 0, 0, 20, 0, 10
        self.assertEqual(
            improvement_per_iteration([100.0, 120.0, 80.0, 90.0, 70.0]),
            [0.0, 0.0, 20.0, 0.0, 10.0],
        )

    def test_delta_energy_is_never_negative(self) -> None:
        deltas = improvement_per_iteration([90.0, 100.0, 110.0, 80.0])
        self.assertTrue(all(d >= 0.0 for d in deltas))

    def test_first_delta_is_zero_by_convention(self) -> None:
        self.assertEqual(improvement_per_iteration([42.0])[0], 0.0)

    def test_deltas_sum_to_total_improvement(self) -> None:
        energies = [100.0, 95.0, 120.0, 60.0, 61.0]
        deltas = improvement_per_iteration(energies)
        self.assertAlmostEqual(sum(deltas), energies[0] - min(energies), places=12)

    def test_iterations_to_within_tolerance(self) -> None:
        # final best is 100; within 5% means <= 105.
        energies = [200.0, 150.0, 104.0, 100.0]
        self.assertEqual(iterations_to_within(energies, 0.05), 3)

    def test_iterations_to_within_handles_immediate_hit(self) -> None:
        self.assertEqual(iterations_to_within([100.0, 200.0, 300.0], 0.05), 1)

    def test_iterations_to_within_empty(self) -> None:
        self.assertIsNone(iterations_to_within([]))


class TestSampleEfficiency(unittest.TestCase):
    def test_ratio(self) -> None:
        self.assertAlmostEqual(sample_efficiency_ratio(20, 5), 4.0, places=12)

    def test_above_one_means_bopis_was_faster(self) -> None:
        self.assertGreater(sample_efficiency_ratio(20, 5), 1.0)
        self.assertLess(sample_efficiency_ratio(5, 20), 1.0)

    def test_equal_iterations_gives_one(self) -> None:
        self.assertAlmostEqual(sample_efficiency_ratio(7, 7), 1.0, places=12)

    def test_zero_denominator_is_none_not_a_crash(self) -> None:
        self.assertIsNone(sample_efficiency_ratio(10, 0))

    def test_summary_reports_spread(self) -> None:
        summary = ser_summary([2.0, 4.0, 0.5, 3.0])
        self.assertEqual(summary["n"], 4)
        self.assertAlmostEqual(summary["mean"], 2.375, places=12)
        self.assertGreater(summary["sd"], 0.0)
        self.assertAlmostEqual(summary["fraction_above_1"], 0.75, places=12)

    def test_summary_of_empty(self) -> None:
        summary = ser_summary([])
        self.assertEqual(summary["n"], 0)
        self.assertIsNone(summary["mean"])


class TestAmortization(unittest.TestCase):
    def test_break_even_is_ceiling_of_cost_over_saving(self) -> None:
        # 1000 J spent, 30 J saved per prompt -> ceil(33.33) = 34
        result = amortization(1000.0, 100.0, 70.0)
        self.assertAlmostEqual(result.energy_saved_per_prompt_j, 30.0, places=12)
        self.assertEqual(result.breakeven_prompts, 34)

    def test_never_recovers_when_no_energy_is_saved(self) -> None:
        result = amortization(1000.0, 100.0, 100.0)
        self.assertIsNone(result.breakeven_prompts)

    def test_never_recovers_when_energy_increases(self) -> None:
        result = amortization(1000.0, 70.0, 100.0)
        self.assertIsNone(result.breakeven_prompts)
        self.assertLess(result.energy_saved_per_prompt_j, 0.0)

    def test_tariff_conversion(self) -> None:
        # 100 J/prompt x 1000 prompts = 100 kJ = 100000/3.6e6 kWh
        result = amortization(0.0, 100.0, 50.0, tariff_php_per_kwh=10.0)
        expected = 1000.0 * 100.0 / 3.6e6 * 10.0
        self.assertAlmostEqual(result.cost_per_1000_default_php, expected, places=12)
        self.assertAlmostEqual(
            result.cost_per_1000_optimized_php, expected / 2.0, places=12
        )

    def test_cost_saving_is_the_difference(self) -> None:
        result = amortization(500.0, 100.0, 60.0)
        payload = result.as_dict()
        self.assertAlmostEqual(
            payload["cost_saved_per_1000_php"],
            payload["cost_per_1000_default_php"]
            - payload["cost_per_1000_optimized_php"],
            places=12,
        )

    def test_serializes(self) -> None:
        payload = amortization(100.0, 10.0, 5.0).as_dict()
        for key in (
            "calibration_energy_j",
            "energy_saved_per_prompt_j",
            "breakeven_prompts",
            "tariff_php_per_kwh",
        ):
            self.assertIn(key, payload)
        self.assertIsInstance(amortization(1.0, 2.0, 1.0), Amortization)


class TestJoulesPerToken(unittest.TestCase):
    def test_normalization(self) -> None:
        self.assertAlmostEqual(joules_per_token(100.0, 50), 2.0, places=12)

    def test_zero_tokens_is_none(self) -> None:
        self.assertIsNone(joules_per_token(100.0, 0))


if __name__ == "__main__":
    unittest.main()
