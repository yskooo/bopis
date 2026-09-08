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

Standard library only. Sampling runs on a daemon thread so an interrupted run
cannot hang on it.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Dict, List, Optional

from bopis.monitor import platform_os
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


@dataclasses.dataclass
class Window:
    """Aggregated telemetry for one inference call."""

    duration_s: float
    n_samples: int
    energy_method: str

    energy_j: Optional[float] = None  # net of P_idle
    energy_gross_j: Optional[float] = None  # raw integration
    energy_crosscheck_j: Optional[float] = None
    crosscheck_method: str = "none"
    clamped_samples: int = 0

    p_idle_w: Optional[float] = None
    power_mean_w: Optional[float] = None
    power_max_w: Optional[float] = None

    gpu_percent_mean: Optional[float] = None
    vram_mib_mean: Optional[float] = None
    cpu_percent: Optional[float] = None
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
    ) -> None:
        self.device = device
        self.p_idle_w = p_idle_w
        self.interval_s = interval_s
        self.pid = pid

        self._samples: List[Sample] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started_at = 0.0
        self._stopped_at = 0.0
        self._cpu_before: Optional[platform_os.CpuTimes] = None
        self._cpu_after: Optional[platform_os.CpuTimes] = None
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
        self._started_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._loop, name="bopis-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> Window:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0 + self.interval_s)
        self._stopped_at = time.monotonic()
        try:
            self._cpu_after = platform_os.cpu_times()
        except Exception:  # pragma: no cover
            self._cpu_after = None
        return self._aggregate()

    # ------------------------------------------------------------------ #

    def _aggregate(self) -> Window:
        duration = max(0.0, self._stopped_at - self._started_at)
        samples = list(self._samples)

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

        return Window(
            duration_s=duration,
            n_samples=len(samples),
            energy_method=method,
            energy_j=energy_net,
            energy_gross_j=energy_gross,
            energy_crosscheck_j=crosscheck,
            crosscheck_method=crosscheck_method,
            clamped_samples=clamped,
            p_idle_w=self.p_idle_w,
            power_mean_w=_mean([p / 1000.0 for p in powers]),  # type: ignore[misc]
            power_max_w=max((p / 1000.0 for p in powers), default=None),  # type: ignore[misc]
            gpu_percent_mean=_mean(
                [float(s.gpu_percent) for s in samples if s.gpu_percent is not None]
            ),
            vram_mib_mean=_mean(
                [
                    s.vram_used_bytes / (1024.0 * 1024.0)
                    for s in samples
                    if s.vram_used_bytes is not None
                ]
            ),
            cpu_percent=cpu_percent,
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
