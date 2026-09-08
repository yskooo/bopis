"""Telemetry sampling, energy integration, and the platform abstraction.

Runs on any host. NVML-dependent behaviour is tested through a fake device, so
the energy-method ladder is exercised even on machines whose GPU reports no
power telemetry at all — which is exactly the case on the development machine.
"""

from __future__ import annotations

import platform
import time
import unittest

from bopis.monitor import nvml, platform_os
from bopis.monitor.nvml import EnergyMethod
from bopis.monitor.sampler import (
    Sample,
    TelemetrySampler,
    integrate_power,
    measure_idle_power,
)


class FakeDevice:
    """Stands in for :class:`bopis.monitor.nvml.GpuDevice`.

    Yields a scripted sequence of readings so integration can be checked against
    an analytically known answer.
    """

    def __init__(
        self,
        power_series=(),
        energy_series=(),
        power_supported=True,
        energy_supported=False,
    ) -> None:
        self.power_series = list(power_series)
        self.energy_series = list(energy_series)
        self.power_supported = power_supported
        self.energy_counter_supported = energy_supported
        self._power_index = 0
        self._energy_index = 0

    def power_milliwatts(self):
        if not self.power_supported or not self.power_series:
            return None
        value = self.power_series[min(self._power_index, len(self.power_series) - 1)]
        self._power_index += 1
        return value

    def total_energy_millijoules(self):
        if not self.energy_counter_supported or not self.energy_series:
            return None
        value = self.energy_series[
            min(self._energy_index, len(self.energy_series) - 1)
        ]
        self._energy_index += 1
        return value

    def utilization(self):
        return (55, 30)

    def memory_used_bytes(self):
        return 512 * 1024 * 1024


class TestIntegratePower(unittest.TestCase):
    def test_constant_power_gives_power_times_time(self) -> None:
        """The one case with an exact analytic answer: 10 W for 1 s is 10 J."""
        samples = [Sample(t=i * 0.1, power_mw=10_000) for i in range(11)]
        net, gross, clamped = integrate_power(samples, p_idle_w=0.0)
        self.assertAlmostEqual(gross, 10.0, places=9)
        self.assertAlmostEqual(net, 10.0, places=9)
        self.assertEqual(clamped, 0)

    def test_idle_subtraction(self) -> None:
        # 10 W measured, 4 W idle, over 1 s -> 6 J net, 10 J gross.
        samples = [Sample(t=i * 0.1, power_mw=10_000) for i in range(11)]
        net, gross, _ = integrate_power(samples, p_idle_w=4.0)
        self.assertAlmostEqual(net, 6.0, places=9)
        self.assertAlmostEqual(gross, 10.0, places=9)

    def test_trapezoid_rule_on_a_linear_ramp(self) -> None:
        """A linear ramp 0->10 W over 1 s integrates to exactly 5 J."""
        samples = [
            Sample(t=i * 0.1, power_mw=int(i * 1000)) for i in range(11)
        ]
        _net, gross, _ = integrate_power(samples, p_idle_w=0.0)
        self.assertAlmostEqual(gross, 5.0, places=9)

    def test_negative_excess_is_clamped_and_counted(self) -> None:
        """Energy cannot be negative; clamping must be visible, not silent."""
        samples = [Sample(t=i * 0.1, power_mw=2_000) for i in range(11)]
        net, gross, clamped = integrate_power(samples, p_idle_w=10.0)
        self.assertAlmostEqual(net, 0.0, places=9)
        self.assertAlmostEqual(gross, 2.0, places=9)
        self.assertEqual(clamped, 10)

    def test_uses_measured_timestamps_not_the_nominal_interval(self) -> None:
        """Irregular spacing must be honoured, or energy is biased."""
        # 10 W held for 2 s, sampled unevenly.
        samples = [
            Sample(t=0.0, power_mw=10_000),
            Sample(t=0.5, power_mw=10_000),
            Sample(t=2.0, power_mw=10_000),
        ]
        _net, gross, _ = integrate_power(samples)
        self.assertAlmostEqual(gross, 20.0, places=9)

    def test_too_few_samples_yields_zero(self) -> None:
        self.assertEqual(integrate_power([]), (0.0, 0.0, 0))
        self.assertEqual(integrate_power([Sample(t=0.0, power_mw=1000)]), (0.0, 0.0, 0))

    def test_ignores_samples_without_power(self) -> None:
        samples = [
            Sample(t=0.0, power_mw=None),
            Sample(t=0.1, power_mw=10_000),
            Sample(t=1.1, power_mw=10_000),
        ]
        _net, gross, _ = integrate_power(samples)
        self.assertAlmostEqual(gross, 10.0, places=9)

    def test_non_increasing_timestamps_are_skipped(self) -> None:
        samples = [
            Sample(t=1.0, power_mw=10_000),
            Sample(t=1.0, power_mw=10_000),
            Sample(t=2.0, power_mw=10_000),
        ]
        _net, gross, _ = integrate_power(samples)
        self.assertAlmostEqual(gross, 10.0, places=9)


class TestSamplerEnergyMethodLadder(unittest.TestCase):
    def test_prefers_the_energy_counter(self) -> None:
        """With the counter available, energy is a counter difference."""
        device = FakeDevice(
            power_series=[8_000] * 200,
            energy_series=[1_000_000 + i * 1_000 for i in range(200)],
            energy_supported=True,
        )
        sampler = TelemetrySampler(device=device, p_idle_w=1.0, interval_s=0.01)
        sampler.start()
        time.sleep(0.2)
        window = sampler.stop()

        self.assertEqual(window.energy_method, EnergyMethod.NVML_ENERGY_COUNTER)
        self.assertIsNotNone(window.energy_j)
        # The integration becomes an independent cross-check.
        self.assertEqual(
            window.crosscheck_method, EnergyMethod.NVML_POWER_INTEGRATION
        )
        self.assertIsNotNone(window.energy_crosscheck_j)

    def test_falls_back_to_power_integration(self) -> None:
        device = FakeDevice(power_series=[8_000] * 200, energy_supported=False)
        sampler = TelemetrySampler(device=device, interval_s=0.01)
        sampler.start()
        time.sleep(0.15)
        window = sampler.stop()

        self.assertEqual(window.energy_method, EnergyMethod.NVML_POWER_INTEGRATION)
        self.assertIsNotNone(window.energy_j)
        self.assertEqual(window.crosscheck_method, "none")

    def test_reports_unavailable_when_nothing_is_supported(self) -> None:
        """The MX330 case: no power, no counter, but util/mem still work."""
        device = FakeDevice(power_supported=False, energy_supported=False)
        sampler = TelemetrySampler(device=device, interval_s=0.01)
        sampler.start()
        time.sleep(0.1)
        window = sampler.stop()

        self.assertEqual(window.energy_method, EnergyMethod.UNAVAILABLE)
        self.assertIsNone(window.energy_j)
        # Utilization and memory must still be collected.
        self.assertIsNotNone(window.gpu_percent_mean)
        self.assertIsNotNone(window.vram_mib_mean)

    def test_works_with_no_device_at_all(self) -> None:
        sampler = TelemetrySampler(device=None, interval_s=0.01)
        sampler.start()
        time.sleep(0.08)
        window = sampler.stop()
        self.assertEqual(window.energy_method, EnergyMethod.UNAVAILABLE)
        self.assertGreater(window.n_samples, 0)


class TestSamplerMechanics(unittest.TestCase):
    def test_collects_samples_at_roughly_the_requested_rate(self) -> None:
        device = FakeDevice(power_series=[5_000] * 500)
        sampler = TelemetrySampler(device=device, interval_s=0.02)
        sampler.start()
        time.sleep(0.4)
        window = sampler.stop()
        # ~20 expected; allow wide slack for scheduler jitter and the coarse
        # Windows timer, but it must be in the right order of magnitude.
        self.assertGreaterEqual(window.n_samples, 5)
        self.assertLessEqual(window.n_samples, 60)

    def test_reports_a_positive_duration(self) -> None:
        sampler = TelemetrySampler(device=None, interval_s=0.01)
        sampler.start()
        time.sleep(0.05)
        window = sampler.stop()
        self.assertGreater(window.duration_s, 0.0)

    def test_stop_is_idempotent_enough_to_be_safe(self) -> None:
        sampler = TelemetrySampler(device=None, interval_s=0.01)
        sampler.start()
        sampler.stop()
        second = sampler.stop()  # must not hang or raise
        self.assertIsNotNone(second)

    def test_restart_clears_previous_samples(self) -> None:
        device = FakeDevice(power_series=[5_000] * 500)
        sampler = TelemetrySampler(device=device, interval_s=0.01)
        sampler.start()
        time.sleep(0.1)
        first = sampler.stop()
        sampler.start()
        time.sleep(0.02)
        second = sampler.stop()
        self.assertLess(second.n_samples, first.n_samples + 1)

    def test_cpu_percent_is_a_valid_percentage(self) -> None:
        sampler = TelemetrySampler(device=None, interval_s=0.01)
        sampler.start()
        # Do some work so the interval is not degenerate.
        sum(i * i for i in range(200_000))
        window = sampler.stop()
        if window.cpu_percent is not None:
            self.assertGreaterEqual(window.cpu_percent, 0.0)
            self.assertLessEqual(window.cpu_percent, 100.0)

    def test_window_reports_clamp_rate(self) -> None:
        device = FakeDevice(power_series=[1_000] * 200)
        sampler = TelemetrySampler(device=device, p_idle_w=50.0, interval_s=0.01)
        sampler.start()
        time.sleep(0.12)
        window = sampler.stop()
        self.assertGreaterEqual(window.clamp_rate, 0.0)
        self.assertLessEqual(window.clamp_rate, 1.0)
        self.assertIn("clamp_rate", window.as_dict())


class TestIdleCalibration(unittest.TestCase):
    def test_returns_unsupported_without_a_device(self) -> None:
        result = measure_idle_power(None, seconds=0.05)
        self.assertFalse(result["supported"])
        self.assertEqual(result["p_idle_w"], 0.0)
        self.assertIn("reason", result)

    def test_returns_unsupported_when_power_is_unavailable(self) -> None:
        device = FakeDevice(power_supported=False)
        result = measure_idle_power(device, seconds=0.05)
        self.assertFalse(result["supported"])

    def test_reports_mean_and_spread(self) -> None:
        """The spread matters: a wide one means the machine was not idle."""
        device = FakeDevice(power_series=[10_000, 12_000] * 500)
        result = measure_idle_power(device, seconds=0.1, interval_s=0.005)
        self.assertTrue(result["supported"])
        self.assertGreater(result["n_samples"], 1)
        self.assertGreaterEqual(result["p_idle_w"], 10.0)
        self.assertLessEqual(result["p_idle_w"], 12.0)
        self.assertGreater(result["sd_w"], 0.0)
        self.assertEqual(result["min_w"], 10.0)
        self.assertEqual(result["max_w"], 12.0)


class TestPlatformTelemetry(unittest.TestCase):
    """The stdlib /proc and kernel32 paths, on whichever host this is."""

    def test_cpu_times_are_monotonically_increasing(self) -> None:
        first = platform_os.cpu_times()
        sum(i for i in range(100_000))
        second = platform_os.cpu_times()
        self.assertGreaterEqual(second.total, first.total)
        self.assertGreaterEqual(second.busy, first.busy)

    def test_cpu_percent_between_is_bounded(self) -> None:
        first = platform_os.cpu_times()
        time.sleep(0.02)
        second = platform_os.cpu_times()
        percent = platform_os.cpu_percent_between(first, second)
        self.assertGreaterEqual(percent, 0.0)
        self.assertLessEqual(percent, 100.0)

    def test_identical_snapshots_give_zero(self) -> None:
        snapshot = platform_os.cpu_times()
        self.assertEqual(platform_os.cpu_percent_between(snapshot, snapshot), 0.0)

    def test_memory_info_is_plausible(self) -> None:
        memory = platform_os.memory_info()
        self.assertGreater(memory.total_bytes, 0)
        self.assertGreaterEqual(memory.available_bytes, 0)
        self.assertLessEqual(memory.available_bytes, memory.total_bytes)
        self.assertGreaterEqual(memory.percent_used, 0.0)
        self.assertLessEqual(memory.percent_used, 100.0)

    def test_cpu_info_reports_cores(self) -> None:
        info = platform_os.cpu_info()
        self.assertGreater(int(info["physical_cores"]), 0)
        self.assertGreaterEqual(
            int(info["logical_cores"]), int(info["physical_cores"])
        )
        self.assertIsInstance(info["model"], str)
        self.assertTrue(info["model"])

    def test_describe_names_the_telemetry_source(self) -> None:
        payload = platform_os.describe()
        expected = {
            "Linux": "/proc",
            "Windows": "kernel32",
        }.get(platform.system(), "none")
        self.assertEqual(payload["telemetry_source"], expected)
        self.assertIn("is_wsl", payload)

    def test_process_memory_for_this_process(self) -> None:
        import os as _os

        rss = platform_os.process_memory_bytes(_os.getpid())
        if rss is not None:  # not observable on every platform
            self.assertGreater(rss, 0)


class TestNvmlProbe(unittest.TestCase):
    def test_probe_never_raises(self) -> None:
        """`profile` must work on machines with no GPU or no driver."""
        result = nvml.probe()
        self.assertIn("available", result)
        if result["available"]:
            self.assertIn("energy_method", result)
            self.assertIn(
                result["energy_method"],
                (
                    EnergyMethod.NVML_ENERGY_COUNTER,
                    EnergyMethod.NVML_POWER_INTEGRATION,
                    EnergyMethod.UNAVAILABLE,
                ),
            )
        else:
            self.assertIn("error", result)

    def test_return_code_names(self) -> None:
        self.assertEqual(nvml.rc_name(0), "SUCCESS")
        self.assertEqual(nvml.rc_name(3), "NOT_SUPPORTED")
        self.assertTrue(nvml.rc_name(9999).startswith("RC_"))


if __name__ == "__main__":
    unittest.main()
