"""The 100 ms telemetry sampler and energy integration.

Chapter 3 specifies that GPU power, GPU utilization and VRAM are sampled at
fixed 100-millisecond intervals throughout each inference call, and that
inference energy is

    E = sum[ (P_t - P_idle) * dt ]

with ``P_idle`` the mean idle draw measured over 60 seconds before the run.

Four departures from a literal reading, each for a measurable reason
--------------------------------------------------------------------
1. **Prefer the driver's energy counter.** Where
   ``nvmlDeviceGetTotalEnergyConsumption`` is available, energy is the difference
   of that monotonic millijoule counter across the window. It needs no
   integration and no idle subtraction, so it carries none of the error the sum
   above does. The Riemann sum remains as the documented fallback, and
   ``energy_method`` records which was used (amendment A-8).

2. **Integrate over measured timestamps, not the nominal interval.** A sampler
   thread cannot hit 100 ms exactly -- the default Windows timer granularity is
   ~15.6 ms, and any scheduling delay lands directly in ``dt``. Using the
   nominal 0.1 s would bias energy by whatever the thread actually drifted.
   Intervals come from :func:`time.monotonic` and the trapezoid rule is used
   rather than a left Riemann sum, which halves the error on a ramping signal.

3. **Clamp negative excess power per sample, and count the clamps.**
   ``P_t - P_idle`` goes negative whenever the GPU is briefly quieter than it was
   during idle calibration, or when the idle baseline drifted upward with
   temperature. Negative energy is not physical, so samples are clamped at zero
   and the clamp count is reported: a high rate means ``P_idle`` needs
   re-measuring, and silently summing negatives would understate energy.

4. **Report gross and net energy.** ``E_gross`` integrates raw board power;
   ``E_net`` subtracts ``P_idle``. Chapter 3 defines only the net figure, but the
   gross one is what makes a suspicious net value diagnosable.

An opt-in CPU package instrument
-------------------------------
Passing a :class:`~bopis.monitor.hwmon.HwmonPowerSource` through
*cpu_power_source* adds Intel RAPL package power, read from a running
OpenHardwareMonitor/LibreHardwareMonitor. It is a measurement, not an estimate,
and it is used whenever it is supplied: alone when the GPU reports nothing, and
summed with the NVML figure when it does. The monitor publishes the mean power
of the *previous* refresh interval, so the sampler keeps reading for one
interval after inference ends and integrates the lag-shifted window; see
:func:`integrate_held`.

An opt-in fifth rung below the ladder
-------------------------------------
When neither NVML path is available the window's energy is ``None`` and the run
is refused. Passing an :class:`~bopis.monitor.estimator.PowerBudget` through
*estimator_budget* adds one rung below that: a labelled resource-allocation
estimate, described in full in :mod:`bopis.monitor.estimator`. It is consulted
only after both measured paths have failed, so it can never displace a real
measurement, and it is off unless explicitly requested.

Standard library only. Sampling runs on a daemon thread so an interrupted run
cannot hang on it.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Dict, List, Optional, Tuple

from bopis.monitor import estimator, platform_os
from bopis.monitor.hwmon import HwmonPowerSource
from bopis.monitor.nvml import EnergyMethod, GpuDevice

#: Chapter 3's sampling interval.
DEFAULT_INTERVAL_S = 0.1

#: Duration of the idle-power calibration, per Chapter 3.
DEFAULT_IDLE_SECONDS = 60.0


@dataclasses.dataclass
class Sample:
    """One telemetry reading."""

    t: float  # seconds from time.monotonic()
    power_mw: Optional[int] = None
    gpu_percent: Optional[int] = None
    vram_used_bytes: Optional[int] = None
    energy_counter_mj: Optional[int] = None
    cpu_power_w: Optional[float] = None  # RAPL package, via OHM/LHM


@dataclasses.dataclass
class Window:
    """Aggregated telemetry for one inference call."""

    duration_s: float
    n_samples: int
    energy_method: str
    energy_basis: str = ""

    energy_j: Optional[float] = None  # net of P_idle
    energy_gross_j: Optional[float] = None  # raw integration
    energy_crosscheck_j: Optional[float] = None
    crosscheck_method: str = "none"
    clamped_samples: int = 0

    # Populated only in resource-estimate mode; the band covers the declared
    # uncertainty in the nameplate budgets, not the model's own form error.
    energy_low_j: Optional[float] = None
    energy_high_j: Optional[float] = None
    estimate: Optional[Dict[str, object]] = None

    p_idle_w: Optional[float] = None
    power_mean_w: Optional[float] = None
    power_max_w: Optional[float] = None

    # Populated only when a CPU package source is attached.
    cpu_energy_j: Optional[float] = None  # net of p_cpu_idle_w
    cpu_energy_gross_j: Optional[float] = None
    p_cpu_idle_w: Optional[float] = None
    cpu_power_mean_w: Optional[float] = None
    cpu_power_max_w: Optional[float] = None
    hwmon_refresh_s: Optional[float] = None

    gpu_percent_mean: Optional[float] = None
    vram_mib_mean: Optional[float] = None
    cpu_percent: Optional[float] = None  # system-wide, for diagnostics
    process_cpu_percent: Optional[float] = None  # this workload's own share
    cpu_attribution: str = estimator.CPU_ATTRIBUTION_NONE
    memory_mib_mean: Optional[float] = None
    n_threads: Optional[int] = None

    @property
    def clamp_rate(self) -> float:
        if not self.n_samples:
            return 0.0
        return self.clamped_samples / self.n_samples

    def as_dict(self) -> Dict[str, object]:
        payload = dataclasses.asdict(self)
        payload["clamp_rate"] = self.clamp_rate
        return payload


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def integrate_power(
    samples: List[Sample], p_idle_w: float = 0.0
) -> tuple[float, float, int]:
    """Trapezoidal integration of power over *samples*.

    Returns ``(net_joules, gross_joules, clamped_count)``. Excess power is
    clamped at zero per interval; see the module docstring for why.
    """
    usable = [s for s in samples if s.power_mw is not None]
    if len(usable) < 2:
        return 0.0, 0.0, 0

    net = gross = 0.0
    clamped = 0
    for earlier, later in zip(usable, usable[1:]):
        dt = later.t - earlier.t
        if dt <= 0:
            continue
        p0 = earlier.power_mw / 1000.0  # type: ignore[operator]
        p1 = later.power_mw / 1000.0  # type: ignore[operator]
        gross += 0.5 * (p0 + p1) * dt

        excess0 = p0 - p_idle_w
        excess1 = p1 - p_idle_w
        if excess0 < 0.0 or excess1 < 0.0:
            clamped += 1
        contribution = 0.5 * (max(excess0, 0.0) + max(excess1, 0.0)) * dt
        net += contribution
    return net, gross, clamped


def integrate_held(
    readings: List[Tuple[float, float]],
    t0: float,
    t1: float,
    p_idle_w: float = 0.0,
) -> Optional[Dict[str, float]]:
    """Integrate a piecewise-constant power series over ``[t0, t1]``.

    *readings* are ``(t, watts)`` pairs polled from a monitor that publishes a
    new value once per refresh interval. Between publications the value is
    constant, so the right integral is a zero-order hold -- the trapezoid rule
    would invent a ramp between two bins that the monitor never observed.

    The caller passes the lag-shifted window (``[start + Δ, stop + Δ]``); this
    function only integrates. Excess power is clamped at zero per segment and
    the clamps counted, for the same reason as :func:`integrate_power`.

    Returns None when no reading covers the window.
    """
    usable = sorted(r for r in readings if r[1] is not None)
    if not usable or t1 <= t0:
        return None

    # The value in force at t0 is the last reading at or before it; if the
    # series starts later, the first reading stands in (and is still bounded
    # by the resolution band the caller reports).
    held = usable[0][1]
    for t, watts in usable:
        if t <= t0:
            held = watts
        else:
            break

    points = [(t0, held)] + [(t, w) for t, w in usable if t0 < t < t1]
    net = gross = 0.0
    clamped = 0
    used = [held]
    for (start, watts), (end, _next) in zip(points, points[1:] + [(t1, 0.0)]):
        dt = end - start
        if dt <= 0:
            continue
        gross += watts * dt
        excess = watts - p_idle_w
        if excess < 0.0:
            clamped += 1
        net += max(excess, 0.0) * dt
        used.append(watts)
    return {
        "net_j": net,
        "gross_j": gross,
        "clamped": float(clamped),
        "p_min_w": min(used),
        "p_max_w": max(used),
        "p_mean_w": gross / (t1 - t0),
    }


class TelemetrySampler:
    """Samples GPU and CPU telemetry on a background thread.

    Usage::

        sampler = TelemetrySampler(device, p_idle_w=12.4)
        sampler.start()
        ...run inference...
        window = sampler.stop()
    """

    def __init__(
        self,
        device: Optional[GpuDevice] = None,
        p_idle_w: float = 0.0,
        interval_s: float = DEFAULT_INTERVAL_S,
        pid: Optional[int] = None,
        estimator_budget: Optional[estimator.PowerBudget] = None,
        gpu_percent_idle: float = 0.0,
        logical_cores: Optional[int] = None,
        cpu_power_source: Optional[HwmonPowerSource] = None,
        p_cpu_idle_w: float = 0.0,
    ) -> None:
        self.device = device
        #: When set, measures CPU package energy (RAPL via OHM/LHM).
        self.cpu_power_source = cpu_power_source
        self.p_cpu_idle_w = p_cpu_idle_w
        self.p_idle_w = p_idle_w
        self.interval_s = interval_s
        self.pid = pid
        #: When set, enables the resource-allocation estimate as a last resort.
        self.estimator_budget = estimator_budget
        #: Idle GPU duty cycle from the same 60 s protocol as ``p_idle_w``.
        self.gpu_percent_idle = gpu_percent_idle
        self.logical_cores = logical_cores

        self._samples: List[Sample] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started_at = 0.0
        self._stopped_at = 0.0
        self._cpu_before: Optional[platform_os.CpuTimes] = None
        self._cpu_after: Optional[platform_os.CpuTimes] = None
        self._proc_cpu_before: Optional[platform_os.ProcessCpuTimes] = None
        self._proc_cpu_after: Optional[platform_os.ProcessCpuTimes] = None
        self._memory_samples: List[float] = []

    # ------------------------------------------------------------------ #

    def _read_once(self) -> Sample:
        sample = Sample(t=time.monotonic())
        if self.device is not None:
            sample.power_mw = self.device.power_milliwatts()
            gpu_percent, _memory_percent = self.device.utilization()
            sample.gpu_percent = gpu_percent
            sample.vram_used_bytes = self.device.memory_used_bytes()
            sample.energy_counter_mj = self.device.total_energy_millijoules()
        if self.cpu_power_source is not None:
            sample.cpu_power_w = self.cpu_power_source.power_watts()
        return sample

    def _loop(self) -> None:
        # Schedule against an absolute deadline so drift does not accumulate.
        next_at = time.monotonic()
        while not self._stop.is_set():
            self._samples.append(self._read_once())
            try:
                memory = platform_os.memory_info()
                self._memory_samples.append(memory.used_mib)
            except Exception:  # pragma: no cover - telemetry is best-effort
                pass
            next_at += self.interval_s
            delay = next_at - time.monotonic()
            if delay < 0:
                # Fell behind; resynchronize rather than spin.
                next_at = time.monotonic()
                delay = 0.0
            self._stop.wait(delay)

    # ------------------------------------------------------------------ #

    def start(self) -> None:
        self._samples.clear()
        self._memory_samples.clear()
        self._stop.clear()
        try:
            self._cpu_before = platform_os.cpu_times()
        except Exception:  # pragma: no cover
            self._cpu_before = None
        self._proc_cpu_before = self._read_process_cpu()
        self._started_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._loop, name="bopis-sampler", daemon=True
        )
        self._thread.start()

    @property
    def _lag_s(self) -> float:
        """How long to keep polling after inference ends; see integrate_held."""
        if self.cpu_power_source is None:
            return 0.0
        return self.cpu_power_source.refresh_s

    def stop(self) -> Window:
        lag = self._lag_s
        if lag <= 0.0:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=2.0 + self.interval_s)
        self._stopped_at = time.monotonic()
        try:
            self._cpu_after = platform_os.cpu_times()
        except Exception:  # pragma: no cover
            self._cpu_after = None
        # Read the process counter before the backend has any chance to reap
        # the server; once it exits, its CPU time is unrecoverable.
        self._proc_cpu_after = self._read_process_cpu()
        if lag > 0.0:
            # The package monitor reports the previous interval's mean, so the
            # energy of the window's last Δ seconds is only published Δ later.
            # Poll a little beyond one interval so that value is captured.
            time.sleep(lag + 2.0 * self.interval_s)
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=2.0 + self.interval_s)
        return self._aggregate()

    # ------------------------------------------------------------------ #

    def _read_process_cpu(self) -> Optional[platform_os.ProcessCpuTimes]:
        if self.pid is None:
            return None
        try:
            return platform_os.process_cpu_times(self.pid)
        except Exception:  # pragma: no cover - telemetry is best-effort
            return None

    def _cpu_share(self, duration: float) -> tuple[Optional[float], str]:
        """The workload's CPU share in ``[0, 1]``, and how it was attributed.

        Prefers the inference process's own CPU time so that a browser or
        editor running alongside the study is not charged to it. Falls back to
        the system-wide figure -- which does include those -- and labels the
        fallback, because the two quantities mean different things and a run
        that silently used the weaker one would be unauditable.
        """
        if self._proc_cpu_before is not None and self._proc_cpu_after is not None:
            fraction = platform_os.process_cpu_fraction(
                self._proc_cpu_before,
                self._proc_cpu_after,
                duration,
                self.logical_cores,
            )
            if fraction is not None:
                return fraction, estimator.CPU_ATTRIBUTION_PROCESS

        if self._cpu_before is not None and self._cpu_after is not None:
            percent = platform_os.cpu_percent_between(
                self._cpu_before, self._cpu_after
            )
            return percent / 100.0, estimator.CPU_ATTRIBUTION_SYSTEM

        return None, estimator.CPU_ATTRIBUTION_NONE

    def _aggregate(self) -> Window:
        duration = max(0.0, self._stopped_at - self._started_at)
        all_samples = list(self._samples)
        # Samples polled during the post-window lag belong to the CPU package
        # integral only; the NVML paths read instantaneous values and must not
        # see them.
        samples = [s for s in all_samples if s.t <= self._stopped_at]

        method = EnergyMethod.UNAVAILABLE
        energy_net: Optional[float] = None
        energy_gross: Optional[float] = None
        crosscheck: Optional[float] = None
        crosscheck_method = "none"
        clamped = 0

        counters = [
            s.energy_counter_mj for s in samples if s.energy_counter_mj is not None
        ]
        powers = [s.power_mw for s in samples if s.power_mw is not None]

        if len(counters) >= 2:
            # Preferred: exact driver-side counter difference.
            method = EnergyMethod.NVML_ENERGY_COUNTER
            energy_net = (counters[-1] - counters[0]) / 1000.0
            energy_gross = energy_net
            if len(powers) >= 2:
                # The integration is then an independent cross-check of the
                # counter rather than the primary measurement.
                net, gross, clamped = integrate_power(samples, self.p_idle_w)
                crosscheck = net
                energy_gross = gross
                crosscheck_method = EnergyMethod.NVML_POWER_INTEGRATION
        elif len(powers) >= 2:
            method = EnergyMethod.NVML_POWER_INTEGRATION
            energy_net, energy_gross, clamped = integrate_power(
                samples, self.p_idle_w
            )

        cpu_percent = None
        if self._cpu_before is not None and self._cpu_after is not None:
            cpu_percent = platform_os.cpu_percent_between(
                self._cpu_before, self._cpu_after
            )

        cpu_share, cpu_attribution = self._cpu_share(duration)
        gpu_percent_mean = _mean(
            [float(s.gpu_percent) for s in samples if s.gpu_percent is not None]
        )

        energy_low = energy_high = None
        estimate_detail: Optional[Dict[str, object]] = None
        energy_basis = "measured telemetry"

        # CPU package (RAPL), when attached. A measurement: it takes precedence
        # over the estimator, and adds to -- never replaces -- a GPU figure.
        cpu_package: Optional[Dict[str, float]] = None
        refresh = self._lag_s
        if self.cpu_power_source is not None:
            cpu_package = integrate_held(
                [(s.t, s.cpu_power_w) for s in all_samples],  # type: ignore[misc]
                self._started_at + refresh,
                self._stopped_at + refresh,
                self.p_cpu_idle_w,
            )
        if cpu_package is not None:
            cpu_net = cpu_package["net_j"]
            # Each window edge falls somewhere inside one refresh bin, whose
            # energy the monitor spread uniformly; the band bounds that
            # misplacement. It does not cover RAPL's own error, which is small
            # on parts with on-die telemetry but not zero.
            resolution = refresh * (cpu_package["p_max_w"] - cpu_package["p_min_w"])
            clamped += int(cpu_package["clamped"])
            if energy_net is None:
                method = EnergyMethod.RAPL_HWMON_POWER_INTEGRATION
                energy_net = cpu_net
                energy_gross = cpu_package["gross_j"]
                energy_basis = (
                    "E = sum[(P_pkg,t - P_pkg,idle) * dt], zero-order hold over "
                    f"[start + {refresh:.2f}s, stop + {refresh:.2f}s] (monitor lag)"
                )
            else:
                method = f"{method}+{EnergyMethod.RAPL_HWMON_POWER_INTEGRATION}"
                energy_net += cpu_net
                energy_gross = (energy_gross or 0.0) + cpu_package["gross_j"]
                energy_basis = "NVML GPU board energy + RAPL CPU package energy"
            energy_low = max(energy_net - resolution, 0.0)
            energy_high = energy_net + resolution

        # The fifth rung: reached only once every measured path has failed, so
        # an estimate can never displace a measurement. Labelled at every layer
        # it touches, never presented as a measured watt value.
        if energy_net is None and self.estimator_budget is not None:
            estimated = estimator.estimate_energy(
                duration_s=duration,
                cpu_fraction=cpu_share,
                gpu_percent_mean=gpu_percent_mean,
                gpu_percent_idle=self.gpu_percent_idle,
                budget=self.estimator_budget,
                cpu_attribution=cpu_attribution,
            )
            method = EnergyMethod.RESOURCE_ALLOCATION_ESTIMATE
            energy_net = estimated.energy_j
            # There is no "gross" counterpart: the model produces a marginal
            # figure directly and never integrates a raw board-power signal.
            energy_gross = None
            energy_low = estimated.energy_low_j
            energy_high = estimated.energy_high_j
            estimate_detail = estimated.as_dict()
            energy_basis = estimator.FORMULA

        return Window(
            duration_s=duration,
            n_samples=len(samples),
            energy_method=method,
            energy_basis=energy_basis,
            energy_j=energy_net,
            energy_gross_j=energy_gross,
            energy_crosscheck_j=crosscheck,
            crosscheck_method=crosscheck_method,
            clamped_samples=clamped,
            energy_low_j=energy_low,
            energy_high_j=energy_high,
            estimate=estimate_detail,
            p_idle_w=self.p_idle_w,
            cpu_energy_j=None if cpu_package is None else cpu_package["net_j"],
            cpu_energy_gross_j=(
                None if cpu_package is None else cpu_package["gross_j"]
            ),
            p_cpu_idle_w=(
                None if self.cpu_power_source is None else self.p_cpu_idle_w
            ),
            cpu_power_mean_w=(
                None if cpu_package is None else cpu_package["p_mean_w"]
            ),
            cpu_power_max_w=None if cpu_package is None else cpu_package["p_max_w"],
            hwmon_refresh_s=None if self.cpu_power_source is None else refresh,
            power_mean_w=_mean([p / 1000.0 for p in powers]),  # type: ignore[misc]
            power_max_w=max((p / 1000.0 for p in powers), default=None),  # type: ignore[misc]
            gpu_percent_mean=gpu_percent_mean,
            vram_mib_mean=_mean(
                [
                    s.vram_used_bytes / (1024.0 * 1024.0)
                    for s in samples
                    if s.vram_used_bytes is not None
                ]
            ),
            cpu_percent=cpu_percent,
            process_cpu_percent=(
                None if cpu_share is None else 100.0 * cpu_share
            ),
            cpu_attribution=cpu_attribution,
            memory_mib_mean=_mean(self._memory_samples),
            n_threads=(
                platform_os.process_threads(self.pid) if self.pid else None
            ),
        )


def measure_idle_power(
    device: Optional[GpuDevice],
    seconds: float = DEFAULT_IDLE_SECONDS,
    interval_s: float = DEFAULT_INTERVAL_S,
    progress=None,
) -> Dict[str, object]:
    """Measure ``P_idle`` with no inference running.

    Chapter 3 specifies 60 seconds at 100 ms. The standard deviation and range
    are also returned: a wide spread means the machine was not actually idle,
    which would corrupt every subsequent energy figure, and that is worth
    knowing before a multi-hour run rather than after.
    """
    if device is None or not device.power_supported:
        return {
            "supported": False,
            "p_idle_w": 0.0,
            "reason": "device reports no power telemetry",
        }

    readings: List[float] = []
    deadline = time.monotonic() + seconds
    next_at = time.monotonic()
    while time.monotonic() < deadline:
        milliwatts = device.power_milliwatts()
        if milliwatts is not None:
            readings.append(milliwatts / 1000.0)
        if progress and len(readings) % 50 == 0:
            progress(len(readings))
        next_at += interval_s
        time.sleep(max(0.0, next_at - time.monotonic()))

    if not readings:
        return {
            "supported": False,
            "p_idle_w": 0.0,
            "reason": "no readings collected",
        }

    mean = sum(readings) / len(readings)
    variance = (
        sum((v - mean) ** 2 for v in readings) / (len(readings) - 1)
        if len(readings) > 1
        else 0.0
    )
    return {
        "supported": True,
        "p_idle_w": mean,
        "sd_w": variance**0.5,
        "min_w": min(readings),
        "max_w": max(readings),
        "n_samples": len(readings),
        "duration_s": seconds,
        "interval_s": interval_s,
    }


def _spread(values: List[float]) -> Dict[str, object]:
    """Mean, sample SD and range of *values*, or zeros when empty."""
    if not values:
        return {"mean": 0.0, "sd": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    mean = sum(values) / len(values)
    variance = (
        sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        if len(values) > 1
        else 0.0
    )
    return {
        "mean": mean,
        "sd": variance**0.5,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def measure_idle_baseline(
    device: Optional[GpuDevice],
    seconds: float = DEFAULT_IDLE_SECONDS,
    interval_s: float = DEFAULT_INTERVAL_S,
    progress=None,
    cpu_power_source: Optional[HwmonPowerSource] = None,
) -> Dict[str, object]:
    """Characterise the host at rest, with or without a power sensor.

    :func:`measure_idle_power` answers "what does the board draw at idle", and
    returns nothing at all on a GPU that reports no power. But the idle *state*
    is still fully observable on such a host, and it still has to be measured:
    the resource-allocation estimator subtracts an idle GPU duty cycle for the
    same reason the measured path subtracts ``P_idle``, and without it the
    compositor's steady background work is charged to inference.

    So this runs Chapter 3's 60-second, 100-millisecond idle protocol over
    every signal the host does expose -- GPU duty cycle, system-wide CPU, and
    board power where available -- and reports the spread of each. The spread is
    the point: a GPU idling at 40% duty cycle, or a CPU at 30%, means the
    machine was not idle and every subsequent estimate is contaminated. Better
    to see that before a multi-hour run than to explain it afterwards.

    With *cpu_power_source* attached, CPU package watts are sampled over the
    same window and their mean becomes ``p_cpu_idle_w``, the package's
    counterpart of ``P_idle``.
    """
    cpu_power_readings: List[float] = []
    gpu_readings: List[float] = []
    power_readings: List[float] = []
    cpu_readings: List[float] = []

    cpu_mark = None
    try:
        cpu_mark = platform_os.cpu_times()
    except Exception:  # pragma: no cover
        cpu_mark = None

    deadline = time.monotonic() + seconds
    next_at = time.monotonic()
    while time.monotonic() < deadline:
        if device is not None:
            gpu_percent, _memory_percent = device.utilization()
            if gpu_percent is not None:
                gpu_readings.append(float(gpu_percent))
            milliwatts = device.power_milliwatts()
            if milliwatts is not None:
                power_readings.append(milliwatts / 1000.0)
        if cpu_power_source is not None:
            watts = cpu_power_source.power_watts()
            if watts is not None:
                cpu_power_readings.append(watts)
        if cpu_mark is not None:
            try:
                now = platform_os.cpu_times()
                cpu_readings.append(
                    platform_os.cpu_percent_between(cpu_mark, now)
                )
                cpu_mark = now
            except Exception:  # pragma: no cover
                pass
        if progress and len(gpu_readings) % 50 == 0:
            progress(len(gpu_readings))
        next_at += interval_s
        time.sleep(max(0.0, next_at - time.monotonic()))

    gpu = _spread(gpu_readings)
    power = _spread(power_readings)
    cpu = _spread(cpu_readings)
    package = _spread(cpu_power_readings)
    return {
        "duration_s": seconds,
        "interval_s": interval_s,
        "power_supported": bool(power_readings),
        "p_idle_w": power["mean"],
        "p_idle_sd_w": power["sd"],
        "gpu_percent_idle": gpu["mean"],
        "gpu_percent_idle_sd": gpu["sd"],
        "gpu_percent_idle_max": gpu["max"],
        "cpu_percent_idle": cpu["mean"],
        "cpu_percent_idle_sd": cpu["sd"],
        "cpu_percent_idle_max": cpu["max"],
        "n_gpu_samples": gpu["n"],
        "n_power_samples": power["n"],
        "n_cpu_samples": cpu["n"],
        "cpu_power_supported": bool(cpu_power_readings),
        "p_cpu_idle_w": package["mean"],
        "p_cpu_idle_sd_w": package["sd"],
        "p_cpu_idle_max_w": package["max"],
        "n_cpu_power_samples": package["n"],
        # A quiet machine sits near zero on both. These thresholds are advisory
        # and are what the CLI keys its warning on.
        "quiet": gpu["mean"] <= 10.0 and cpu["mean"] <= 15.0,
    }
