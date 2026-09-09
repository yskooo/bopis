"""Resource-allocation energy estimation, for hosts with no power sensor.

Why this module exists (amendment A-41)
---------------------------------------
Chapter 3's energy objective is measured from NVML: either the driver's
millijoule counter or the integral of ``nvmlDeviceGetPowerUsage``. A large
share of consumer laptop GPUs -- the GeForce MX330 on the development machine
among them -- expose *utilization* and *memory* but return
``NVML_ERROR_NOT_SUPPORTED`` for both power queries. On such a host there is no
instrument, and the tool's default behaviour is to refuse the run rather than
publish a number it cannot defend.

Refusal is correct but it is not the only defensible option. The workload's
*resource allocation* is still observable at full fidelity: how much CPU time
the inference process consumed, and what fraction of the sampling window the
GPU was executing kernels. Combining those observations with the vendor's
nameplate power figures yields an **estimate**, not a measurement. This module
produces that estimate and, more importantly, carries the formula, the inputs
and the assumptions alongside every value it returns, so that a reader can see
exactly what was assumed and reject it if they disagree.

Nothing here ever overrides a real measurement. The estimator is consulted only
when the NVML ladder in :mod:`bopis.monitor.sampler` has already returned
``UNAVAILABLE``, and every row it produces is tagged
``energy_method="resource_allocation_estimate"`` and
``energy_scope="estimated_resource_allocation"``.

The model
---------
Let ``T`` be the wall duration of one inference window. Then

    E_est = [ (P_cpu^tdp - P_cpu^idle) * u_cpu^proc
            + (P_gpu^tdp - P_gpu^idle) * du_gpu ] * T

with

``u_cpu^proc``
    CPU-seconds charged to the inference process over the window, divided by
    ``N_logical * T``. Bounded to ``[0, 1]``.
``du_gpu``
    ``max(mean(u_gpu) - u_gpu^idle, 0)``, expressed as a fraction. ``u_gpu^idle``
    is measured over the same 60-second idle protocol Chapter 3 specifies for
    ``P_idle``.

Three properties of that form are deliberate.

1. **The coefficient is a dynamic range, not a nameplate.** Multiplying TDP
   directly by utilization would charge the idle draw again at every level of
   load: a package that burns 2 W at rest and 15 W saturated does not consume
   ``0.5 * 15 = 7.5 W`` at half load but roughly ``2 + 0.5 * 13``. Since the
   objective is the *marginal* energy of inference -- the same quantity
   ``E = sum (P_t - P_idle) dt`` isolates in the measured path -- the idle
   component is excluded from the coefficient and not subtracted afterwards.
   Subtracting a whole-system ``P_idle`` from an already utilization-scaled
   product, as a first cut at this module did, discounts it twice and drives
   short low-load windows to a spurious zero.

2. **CPU power is attributed per process, not per host.** System-wide CPU
   utilization is the wrong signal on a laptop that is also running a browser
   and an editor: their load would be charged to the inference energy, and the
   figure would move with whatever else the operator happened to open. The
   process-scoped CPU time of the ``llama-server`` child is used instead, which
   is exactly the "label the data by what is actually running" requirement.
   When the process handle cannot be opened the estimator falls back to the
   system-wide signal and says so in ``cpu_attribution``, because a silent
   substitution would change what the number means without recording it.

3. **GPU utilization is baseline-corrected but cannot be per-process.** NVML's
   consumer-driver utilization is a device-level duty cycle: the percentage of
   the sample period during which at least one kernel was resident. There is no
   per-process decomposition available without accounting-mode support that
   these drivers also withhold. Subtracting the measured idle duty cycle
   removes the compositor's steady contribution; what remains assumes the
   inference process is the dominant GPU consumer. That assumption is recorded
   rather than hidden, and it is the reason the estimator asks the operator to
   close other GPU clients before calibrating.

What the estimate is good for, and what it is not
-------------------------------------------------
The model is linear in utilization, ignores DVFS, and treats a duty cycle as if
it were proportional to power. Real CPU power is convex in frequency and real
GPU power depends on which units a kernel exercises, so the *absolute* joule
figure carries an error that no uncertainty band computed here can bound
honestly -- the band below propagates only the declared uncertainty in the
nameplate budgets, not the error in the model's form.

What survives those limitations is *relative* comparison. Two configurations
measured on the same host, within the same session, against the same idle
baseline, share the same systematic error; the ratio between them is far better
determined than either value alone. That is the claim the estimate can support:
"configuration A allocates ~30% less compute-energy than configuration B on
this machine", and never "inference consumed 41.2 J".

Consequently a run performed in this mode must not be reported as a measured
energy result. :func:`caveat` returns the sentence the CLI prints, the manifest
stores, and the dashboard displays, so that the qualification travels with the
data instead of living only in a thesis footnote.

Standard library only.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Tuple

#: Nominal CPU package power, in watts, when every logical core is saturated.
#: The default is a 15 W U-series laptop part; override it from the vendor's
#: specification for the host actually under test.
DEFAULT_CPU_TDP_W = 15.0

#: Nominal GPU board power, in watts. The default matches the GeForce MX330.
DEFAULT_GPU_TDP_W = 25.0

#: Fractional uncertainty assigned to each nameplate budget. Vendor TDP is a
#: sustained thermal figure rather than an electrical maximum, and the mapping
#: from either to instantaneous draw is loose; 30% is a deliberately wide
#: default that the operator can tighten only with an external instrument.
DEFAULT_UNCERTAINTY_FRAC = 0.30

#: How the CPU term was attributed.
CPU_ATTRIBUTION_PROCESS = "process_cpu_time"
CPU_ATTRIBUTION_SYSTEM = "system_wide_cpu_time"
CPU_ATTRIBUTION_NONE = "unavailable"

#: The model, as a string, stored next to every value it produces.
FORMULA = (
    "E_est = [(P_cpu_tdp - P_cpu_idle) * u_cpu_proc "
    "+ (P_gpu_tdp - P_gpu_idle) * max(u_gpu - u_gpu_idle, 0)] * T"
)


def caveat() -> str:
    """The qualification that must travel with any estimated energy figure."""
    return (
        "ESTIMATED, NOT MEASURED. This host's GPU reports no power or energy "
        "telemetry, so these joule figures come from a linear "
        "resource-allocation model over observed CPU-time and GPU-utilization, "
        "scaled by nameplate power budgets. They are valid for comparing "
        "configurations measured on this host in this session; they are not "
        "valid as absolute energy values and must not be reported as measured "
        "energy."
    )


@dataclasses.dataclass(frozen=True)
class PowerBudget:
    """The declared power envelope of one host.

    ``*_idle_w`` default to zero, which makes each coefficient the full
    nameplate figure and therefore the conservative (largest) estimate. Filling
    them in from a vendor specification or a wall meter narrows the coefficient
    to the true dynamic range; leaving them at zero overstates the marginal
    energy of light loads, which is why the CLI reports the values in force.
    """

    cpu_tdp_w: float = DEFAULT_CPU_TDP_W
    gpu_tdp_w: float = DEFAULT_GPU_TDP_W
    cpu_idle_w: float = 0.0
    gpu_idle_w: float = 0.0
    uncertainty_frac: float = DEFAULT_UNCERTAINTY_FRAC

    def __post_init__(self) -> None:
        if self.cpu_tdp_w < 0 or self.gpu_tdp_w < 0:
            raise ValueError("power budgets must be non-negative")
        if self.cpu_idle_w < 0 or self.gpu_idle_w < 0:
            raise ValueError("idle power must be non-negative")
        if self.cpu_idle_w > self.cpu_tdp_w:
            raise ValueError(
                f"cpu_idle_w ({self.cpu_idle_w} W) exceeds cpu_tdp_w "
                f"({self.cpu_tdp_w} W); the dynamic range would be negative"
            )
        if self.gpu_idle_w > self.gpu_tdp_w:
            raise ValueError(
                f"gpu_idle_w ({self.gpu_idle_w} W) exceeds gpu_tdp_w "
                f"({self.gpu_tdp_w} W); the dynamic range would be negative"
            )
        if not 0.0 <= self.uncertainty_frac <= 1.0:
            raise ValueError("uncertainty_frac must lie in [0, 1]")

    @property
    def cpu_dynamic_w(self) -> float:
        """Watts attributable to the CPU going from rest to saturation."""
        return self.cpu_tdp_w - self.cpu_idle_w

    @property
    def gpu_dynamic_w(self) -> float:
        """Watts attributable to the GPU going from rest to saturation."""
        return self.gpu_tdp_w - self.gpu_idle_w

    def as_dict(self) -> Dict[str, object]:
        return {
            "cpu_tdp_w": self.cpu_tdp_w,
            "gpu_tdp_w": self.gpu_tdp_w,
            "cpu_idle_w": self.cpu_idle_w,
            "gpu_idle_w": self.gpu_idle_w,
            "cpu_dynamic_w": self.cpu_dynamic_w,
            "gpu_dynamic_w": self.gpu_dynamic_w,
            "uncertainty_frac": self.uncertainty_frac,
        }


#: The modelling assumptions, enumerated so a reader can audit them one by one.
ASSUMPTIONS: Tuple[str, ...] = (
    "A1 Power is linear in utilization between the declared idle and TDP "
    "points. Real CPU power is convex in frequency, so mid-load windows are "
    "the least accurate.",
    "A2 Vendor TDP is treated as the power drawn at full utilization. TDP is a "
    "sustained thermal rating, not an electrical maximum; short bursts can "
    "exceed it and thermally throttled parts never reach it.",
    "A3 The CPU term is charged from the inference process's own CPU time, so "
    "concurrent applications are excluded. Reported in cpu_attribution; a "
    "value of system_wide_cpu_time means this exclusion did not hold.",
    "A4 The GPU term uses NVML device-level utilization, which is a duty cycle "
    "and not an occupancy measure, and which cannot be split per process on "
    "consumer drivers. After subtracting the measured idle duty cycle, the "
    "remainder is charged entirely to inference.",
    "A5 Only CPU package and GPU board are modelled. DRAM, storage, display, "
    "chipset and voltage-regulator losses are excluded, matching the "
    "gpu_only scope's exclusion of everything off the board.",
    "A6 Frequency and voltage scaling are not observed, so an unplugged or "
    "power-limited run is modelled as though it ran at the same operating "
    "point as a plugged-in one.",
)


@dataclasses.dataclass
class EstimatedEnergy:
    """One window's estimated energy, with every input that produced it."""

    energy_j: float
    energy_low_j: float
    energy_high_j: float

    cpu_component_j: float
    gpu_component_j: float

    duration_s: float
    cpu_fraction: float
    gpu_fraction: float
    gpu_fraction_observed: float
    gpu_fraction_baseline: float

    budget: PowerBudget
    cpu_attribution: str

    @property
    def mean_power_w(self) -> Optional[float]:
        """Implied mean attributed power, for sanity-checking the estimate."""
        if self.duration_s <= 0:
            return None
        return self.energy_j / self.duration_s

    def as_dict(self) -> Dict[str, object]:
        return {
            "energy_j": self.energy_j,
            "energy_low_j": self.energy_low_j,
            "energy_high_j": self.energy_high_j,
            "cpu_component_j": self.cpu_component_j,
            "gpu_component_j": self.gpu_component_j,
            "implied_mean_power_w": self.mean_power_w,
            "duration_s": self.duration_s,
            "u_cpu_proc": self.cpu_fraction,
            "u_gpu_net": self.gpu_fraction,
            "u_gpu_observed": self.gpu_fraction_observed,
            "u_gpu_idle_baseline": self.gpu_fraction_baseline,
            "cpu_attribution": self.cpu_attribution,
            "budget": self.budget.as_dict(),
            "formula": FORMULA,
            "assumptions": list(ASSUMPTIONS),
            "caveat": caveat(),
        }


def _clamp_fraction(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def estimate_energy(
    duration_s: float,
    cpu_fraction: Optional[float],
    gpu_percent_mean: Optional[float],
    gpu_percent_idle: float = 0.0,
    budget: Optional[PowerBudget] = None,
    cpu_attribution: str = CPU_ATTRIBUTION_PROCESS,
) -> EstimatedEnergy:
    """Apply the resource-allocation model to one sampling window.

    *cpu_fraction* is the share of total machine CPU capacity charged to the
    workload, in ``[0, 1]`` -- process CPU-seconds over ``N_logical * T``, not a
    percentage. *gpu_percent_mean* and *gpu_percent_idle* are NVML utilization
    percentages in ``[0, 100]``; the idle figure comes from the same 60-second
    calibration Chapter 3 uses for ``P_idle``.

    Missing inputs contribute zero rather than raising: a host with no GPU at
    all yields a CPU-only estimate, which is the right answer for CPU-only
    inference, and a window whose process handle could not be read yields the
    GPU term alone. Both cases stay visible through the component fields.
    """
    budget = budget or PowerBudget()
    duration = max(0.0, float(duration_s))

    u_cpu = _clamp_fraction(cpu_fraction)
    u_gpu_observed = _clamp_fraction(
        None if gpu_percent_mean is None else gpu_percent_mean / 100.0
    )
    u_gpu_baseline = _clamp_fraction(gpu_percent_idle / 100.0)
    # Clamped at zero for the same reason the measured path clamps negative
    # excess power: a window quieter than the idle calibration cannot have
    # consumed negative energy, it means the baseline needs re-measuring.
    u_gpu = max(u_gpu_observed - u_gpu_baseline, 0.0)

    cpu_component = budget.cpu_dynamic_w * u_cpu * duration
    gpu_component = budget.gpu_dynamic_w * u_gpu * duration
    energy = cpu_component + gpu_component

    # The two nameplate figures are independent sources of error, so their
    # contributions combine in quadrature rather than summing linearly; adding
    # them would assume both budgets are wrong in the same direction by the
    # same proportion. This band covers only the declared budget uncertainty --
    # the error in the model's linear form is out of its reach (see A1).
    frac = budget.uncertainty_frac
    sigma = ((frac * cpu_component) ** 2 + (frac * gpu_component) ** 2) ** 0.5

    return EstimatedEnergy(
        energy_j=energy,
        energy_low_j=max(energy - sigma, 0.0),
        energy_high_j=energy + sigma,
        cpu_component_j=cpu_component,
        gpu_component_j=gpu_component,
        duration_s=duration,
        cpu_fraction=u_cpu,
        gpu_fraction=u_gpu,
        gpu_fraction_observed=u_gpu_observed,
        gpu_fraction_baseline=u_gpu_baseline,
        budget=budget,
        cpu_attribution=cpu_attribution,
    )


def describe(budget: Optional[PowerBudget] = None) -> Dict[str, object]:
    """The estimator's full self-description, for the manifest and dashboard."""
    budget = budget or PowerBudget()
    return {
        "method": "resource_allocation_estimate",
        "measured": False,
        "formula": FORMULA,
        "budget": budget.as_dict(),
        "assumptions": list(ASSUMPTIONS),
        "valid_for": (
            "relative comparison of configurations measured on the same host "
            "in the same session against the same idle baseline"
        ),
        "not_valid_for": (
            "absolute energy claims, cross-host comparison, or any figure "
            "reported as measured energy"
        ),
        "caveat": caveat(),
    }
