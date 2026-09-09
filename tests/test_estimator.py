"""The resource-allocation energy estimate: its arithmetic and its guards.

These tests exist because the estimate is the one energy path in the tool that
produces a number no instrument checked. Every property that makes it defensible
therefore has to be asserted rather than assumed -- in particular that it does
not double-discount the idle baseline, that it charges only the workload's own
CPU time, and that it can never be reached on a host where a real measurement
was available.
"""

from __future__ import annotations

import os
import time
import unittest

from bopis.monitor import estimator, platform_os
from bopis.monitor.estimator import PowerBudget, estimate_energy

# The estimator's integration with the NVML ladder -- that it is reached only
# when both measured paths fail -- is asserted in test_monitor.py, where the
# fake device lives.


class TestModelArithmetic(unittest.TestCase):
    """Cases with an exact analytic answer."""

    def test_full_cpu_load_for_known_time(self) -> None:
        # 10 W dynamic range, fully allocated, 2 s -> 20 J.
        budget = PowerBudget(cpu_tdp_w=10.0, gpu_tdp_w=0.0, uncertainty_frac=0.0)
        result = estimate_energy(
            duration_s=2.0,
            cpu_fraction=1.0,
            gpu_percent_mean=0.0,
            budget=budget,
        )
        self.assertAlmostEqual(result.energy_j, 20.0, places=9)
        self.assertAlmostEqual(result.cpu_component_j, 20.0, places=9)
        self.assertAlmostEqual(result.gpu_component_j, 0.0, places=9)

    def test_coefficient_is_the_dynamic_range_not_the_nameplate(self) -> None:
        """Half load on a 4-15 W package is 4 + 0.5*11, so 5.5 J of margin.

        The marginal figure is 0.5 * (15 - 4) = 5.5 W, not 0.5 * 15 = 7.5 W and
        not 0.5 * 15 - 4 = 3.5 W. The last of those is the double-discount the
        first draft of this model committed.
        """
        budget = PowerBudget(
            cpu_tdp_w=15.0,
            cpu_idle_w=4.0,
            gpu_tdp_w=0.0,
            uncertainty_frac=0.0,
        )
        result = estimate_energy(
            duration_s=1.0,
            cpu_fraction=0.5,
            gpu_percent_mean=0.0,
            budget=budget,
        )
        self.assertAlmostEqual(result.energy_j, 5.5, places=9)

    def test_light_load_does_not_collapse_to_zero(self) -> None:
        """The regression the double-discount form produced.

        Subtracting a whole-system idle from an already utilization-scaled
        product drove short, light windows to exactly zero, which would tell the
        optimizer that some configurations cost nothing.
        """
        budget = PowerBudget(
            cpu_tdp_w=15.0,
            cpu_idle_w=4.0,
            gpu_tdp_w=25.0,
            gpu_idle_w=2.0,
            uncertainty_frac=0.0,
        )
        result = estimate_energy(
            duration_s=0.4,
            cpu_fraction=0.05,
            gpu_percent_mean=3.0,
            budget=budget,
        )
        self.assertGreater(result.energy_j, 0.0)

    def test_gpu_idle_duty_cycle_is_subtracted(self) -> None:
        # 50% observed, 10% idle -> 40% of a 20 W range over 1 s -> 8 J.
        budget = PowerBudget(
            cpu_tdp_w=0.0, gpu_tdp_w=20.0, uncertainty_frac=0.0
        )
        result = estimate_energy(
            duration_s=1.0,
            cpu_fraction=0.0,
            gpu_percent_mean=50.0,
            gpu_percent_idle=10.0,
            budget=budget,
        )
        self.assertAlmostEqual(result.energy_j, 8.0, places=9)
        self.assertAlmostEqual(result.gpu_fraction, 0.4, places=9)
        self.assertAlmostEqual(result.gpu_fraction_observed, 0.5, places=9)
        self.assertAlmostEqual(result.gpu_fraction_baseline, 0.1, places=9)

    def test_below_baseline_gpu_clamps_instead_of_going_negative(self) -> None:
        """A window quieter than calibration means the baseline drifted.

        Energy cannot be negative, so the term clamps -- the same reasoning the
        measured path applies to negative excess power.
        """
        budget = PowerBudget(
            cpu_tdp_w=0.0, gpu_tdp_w=20.0, uncertainty_frac=0.0
        )
        result = estimate_energy(
            duration_s=1.0,
            cpu_fraction=0.0,
            gpu_percent_mean=2.0,
            gpu_percent_idle=30.0,
            budget=budget,
        )
        self.assertEqual(result.energy_j, 0.0)
        self.assertEqual(result.gpu_fraction, 0.0)

    def test_zero_duration_yields_zero(self) -> None:
        result = estimate_energy(
            duration_s=0.0, cpu_fraction=1.0, gpu_percent_mean=100.0
        )
        self.assertEqual(result.energy_j, 0.0)
        self.assertIsNone(result.mean_power_w)

    def test_components_are_additive_and_separable(self) -> None:
        budget = PowerBudget(
            cpu_tdp_w=10.0, gpu_tdp_w=20.0, uncertainty_frac=0.0
        )
        result = estimate_energy(
            duration_s=2.0,
            cpu_fraction=0.5,
            gpu_percent_mean=25.0,
            budget=budget,
        )
        self.assertAlmostEqual(result.cpu_component_j, 10.0, places=9)
        self.assertAlmostEqual(result.gpu_component_j, 10.0, places=9)
        self.assertAlmostEqual(result.energy_j, 20.0, places=9)
        self.assertAlmostEqual(result.mean_power_w, 10.0, places=9)


class TestMissingInputs(unittest.TestCase):
    """Absent signals must contribute zero, not raise and not guess."""

    def test_no_cpu_signal_gives_a_gpu_only_estimate(self) -> None:
        budget = PowerBudget(
            cpu_tdp_w=10.0, gpu_tdp_w=20.0, uncertainty_frac=0.0
        )
        result = estimate_energy(
            duration_s=1.0,
            cpu_fraction=None,
            gpu_percent_mean=100.0,
            budget=budget,
        )
        self.assertEqual(result.cpu_component_j, 0.0)
        self.assertAlmostEqual(result.gpu_component_j, 20.0, places=9)

    def test_no_gpu_at_all_gives_a_cpu_only_estimate(self) -> None:
        """The right answer for CPU-only inference on a host with no GPU."""
        budget = PowerBudget(
            cpu_tdp_w=10.0, gpu_tdp_w=20.0, uncertainty_frac=0.0
        )
        result = estimate_energy(
            duration_s=1.0,
            cpu_fraction=1.0,
            gpu_percent_mean=None,
            budget=budget,
        )
        self.assertAlmostEqual(result.cpu_component_j, 10.0, places=9)
        self.assertEqual(result.gpu_component_j, 0.0)

    def test_fractions_are_clamped_to_unit_interval(self) -> None:
        budget = PowerBudget(
            cpu_tdp_w=10.0, gpu_tdp_w=10.0, uncertainty_frac=0.0
        )
        result = estimate_energy(
            duration_s=1.0,
            cpu_fraction=3.7,
            gpu_percent_mean=480.0,
            budget=budget,
        )
        self.assertEqual(result.cpu_fraction, 1.0)
        self.assertEqual(result.gpu_fraction_observed, 1.0)
        self.assertAlmostEqual(result.energy_j, 20.0, places=9)


class TestUncertainty(unittest.TestCase):
    def test_band_brackets_the_estimate(self) -> None:
        result = estimate_energy(
            duration_s=1.0,
            cpu_fraction=0.5,
            gpu_percent_mean=50.0,
            budget=PowerBudget(uncertainty_frac=0.3),
        )
        self.assertLess(result.energy_low_j, result.energy_j)
        self.assertGreater(result.energy_high_j, result.energy_j)

    def test_components_combine_in_quadrature_not_linearly(self) -> None:
        """Independent budget errors, so sqrt(a^2 + b^2), not a + b."""
        budget = PowerBudget(
            cpu_tdp_w=10.0, gpu_tdp_w=10.0, uncertainty_frac=0.5
        )
        result = estimate_energy(
            duration_s=1.0,
            cpu_fraction=1.0,
            gpu_percent_mean=100.0,
            budget=budget,
        )
        # Each component is 10 J; each contributes 5 J of uncertainty.
        expected_sigma = (5.0**2 + 5.0**2) ** 0.5
        self.assertAlmostEqual(
            result.energy_high_j - result.energy_j, expected_sigma, places=9
        )
        self.assertLess(expected_sigma, 10.0)  # strictly below the linear sum

    def test_lower_bound_never_goes_negative(self) -> None:
        result = estimate_energy(
            duration_s=1.0,
            cpu_fraction=0.5,
            gpu_percent_mean=50.0,
            budget=PowerBudget(uncertainty_frac=1.0),
        )
        self.assertGreaterEqual(result.energy_low_j, 0.0)

    def test_zero_declared_uncertainty_gives_a_degenerate_band(self) -> None:
        result = estimate_energy(
            duration_s=1.0,
            cpu_fraction=1.0,
            gpu_percent_mean=100.0,
            budget=PowerBudget(uncertainty_frac=0.0),
        )
        self.assertAlmostEqual(result.energy_low_j, result.energy_j, places=9)
        self.assertAlmostEqual(result.energy_high_j, result.energy_j, places=9)


class TestRelativeComparison(unittest.TestCase):
    """The property the estimate is actually claimed to support."""

    def test_ordering_follows_allocated_resources(self) -> None:
        budget = PowerBudget(cpu_tdp_w=15.0, gpu_tdp_w=25.0)
        light = estimate_energy(
            duration_s=1.0, cpu_fraction=0.2, gpu_percent_mean=20.0,
            budget=budget,
        )
        heavy = estimate_energy(
            duration_s=1.0, cpu_fraction=0.8, gpu_percent_mean=80.0,
            budget=budget,
        )
        self.assertLess(light.energy_j, heavy.energy_j)

    def test_ratio_is_independent_of_the_budget_for_one_component(self) -> None:
        """Why relative claims survive a wrong TDP but absolute ones do not.

        For two windows whose load sits on the same component, the estimate's
        ratio is the ratio of allocated resource-time and cancels the budget
        entirely -- so a mis-declared TDP moves both joule figures but not the
        percentage between them.
        """
        ratios = []
        for tdp in (5.0, 15.0, 45.0):
            budget = PowerBudget(
                cpu_tdp_w=tdp, gpu_tdp_w=0.0, uncertainty_frac=0.0
            )
            a = estimate_energy(
                duration_s=1.0, cpu_fraction=0.25, gpu_percent_mean=0.0,
                budget=budget,
            )
            b = estimate_energy(
                duration_s=1.0, cpu_fraction=0.75, gpu_percent_mean=0.0,
                budget=budget,
            )
            ratios.append(a.energy_j / b.energy_j)
        for ratio in ratios:
            self.assertAlmostEqual(ratio, 1.0 / 3.0, places=9)


class TestPowerBudgetGuards(unittest.TestCase):
    def test_idle_above_tdp_is_rejected(self) -> None:
        """It would make the dynamic range negative, so energy negative."""
        with self.assertRaises(ValueError):
            PowerBudget(cpu_tdp_w=10.0, cpu_idle_w=12.0)
        with self.assertRaises(ValueError):
            PowerBudget(gpu_tdp_w=10.0, gpu_idle_w=12.0)

    def test_negative_budgets_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PowerBudget(cpu_tdp_w=-1.0)
        with self.assertRaises(ValueError):
            PowerBudget(gpu_idle_w=-1.0)

    def test_uncertainty_outside_the_unit_interval_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PowerBudget(uncertainty_frac=1.5)
        with self.assertRaises(ValueError):
            PowerBudget(uncertainty_frac=-0.1)

    def test_dynamic_range_is_the_difference(self) -> None:
        budget = PowerBudget(
            cpu_tdp_w=15.0, cpu_idle_w=2.0, gpu_tdp_w=25.0, gpu_idle_w=1.0
        )
        self.assertAlmostEqual(budget.cpu_dynamic_w, 13.0)
        self.assertAlmostEqual(budget.gpu_dynamic_w, 24.0)


class TestProvenanceTravelsWithTheValue(unittest.TestCase):
    """A figure this soft must not be separable from its qualification."""

    def test_as_dict_carries_formula_assumptions_and_caveat(self) -> None:
        payload = estimate_energy(
            duration_s=1.0, cpu_fraction=0.5, gpu_percent_mean=50.0
        ).as_dict()
        self.assertEqual(payload["formula"], estimator.FORMULA)
        self.assertEqual(len(payload["assumptions"]), len(estimator.ASSUMPTIONS))
        self.assertIn("ESTIMATED, NOT MEASURED", str(payload["caveat"]))
        self.assertIn("budget", payload)
        self.assertIn("cpu_attribution", payload)

    def test_caveat_refuses_the_word_measured_as_a_claim(self) -> None:
        text = estimator.caveat()
        self.assertIn("not valid as absolute energy", text.lower())
        self.assertIn("must not be reported as measured", text.lower())

    def test_describe_states_both_what_it_supports_and_what_it_does_not(
        self,
    ) -> None:
        payload = estimator.describe()
        self.assertFalse(payload["measured"])
        self.assertIn("relative comparison", str(payload["valid_for"]))
        self.assertIn("absolute energy", str(payload["not_valid_for"]))
        self.assertEqual(len(payload["assumptions"]), 6)

    def test_every_assumption_is_identified(self) -> None:
        """So a reader can cite the one they disagree with."""
        for index, assumption in enumerate(estimator.ASSUMPTIONS, start=1):
            self.assertTrue(assumption.startswith(f"A{index} "), assumption)


class TestProcessCpuAccounting(unittest.TestCase):
    """The signal that keeps other applications out of the energy figure."""

    def test_own_process_cpu_time_advances_under_load(self) -> None:
        before = platform_os.process_cpu_times(os.getpid())
        if before is None:
            self.skipTest("per-process CPU time not observable on this host")
        sum(i * i for i in range(2_000_000))
        after = platform_os.process_cpu_times(os.getpid())
        self.assertIsNotNone(after)
        self.assertGreater(after.total, before.total)
        self.assertGreaterEqual(after.user, before.user)

    def test_missing_process_is_not_an_error(self) -> None:
        """A server that exited mid-window must degrade, not crash."""
        # PID 0 is never a normal user process on either platform.
        self.assertIsNone(platform_os.process_cpu_times(0))

    def test_fraction_is_bounded_by_the_unit_interval(self) -> None:
        before = platform_os.process_cpu_times(os.getpid())
        if before is None:
            self.skipTest("per-process CPU time not observable on this host")
        start = time.monotonic()
        sum(i for i in range(500_000))
        elapsed = time.monotonic() - start
        after = platform_os.process_cpu_times(os.getpid())
        fraction = platform_os.process_cpu_fraction(before, after, elapsed)
        self.assertIsNotNone(fraction)
        self.assertGreaterEqual(fraction, 0.0)
        self.assertLessEqual(fraction, 1.0)

    def test_single_thread_cannot_exceed_its_share_of_the_machine(self) -> None:
        """One busy thread on an N-core host is at most 1/N of capacity."""
        before = platform_os.process_cpu_times(os.getpid())
        if before is None:
            self.skipTest("per-process CPU time not observable on this host")
        start = time.monotonic()
        while time.monotonic() - start < 0.15:
            pass
        elapsed = time.monotonic() - start
        after = platform_os.process_cpu_times(os.getpid())
        cores = 8
        fraction = platform_os.process_cpu_fraction(
            before, after, elapsed, logical_cores=cores
        )
        self.assertIsNotNone(fraction)
        # Generous slack for timer granularity, but it cannot be near 1.0.
        self.assertLess(fraction, 0.5)

    def test_backwards_counters_are_reported_as_unusable(self) -> None:
        """A reused PID must not read as low utilization."""
        high = platform_os.ProcessCpuTimes(user=5.0, kernel=5.0)
        low = platform_os.ProcessCpuTimes(user=1.0, kernel=1.0)
        self.assertIsNone(platform_os.process_cpu_fraction(high, low, 1.0))

    def test_zero_duration_is_reported_as_unusable(self) -> None:
        mark = platform_os.ProcessCpuTimes(user=1.0, kernel=1.0)
        self.assertIsNone(platform_os.process_cpu_fraction(mark, mark, 0.0))

    def test_thread_count_of_this_process_is_positive(self) -> None:
        count = platform_os.process_threads(os.getpid())
        if count is None:
            self.skipTest("thread count not observable on this host")
        self.assertGreater(count, 0)

    def test_thread_count_of_a_missing_process_is_none_not_zero(self) -> None:
        """Zero would read as a live process doing nothing, which is wrong."""
        self.assertIsNone(platform_os.process_threads(999_999))


if __name__ == "__main__":
    unittest.main()
