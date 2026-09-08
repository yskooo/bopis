"""Configuration success indicators, GP reliability, convergence, amortization.

Everything Chapter 3's Data Analysis section defines, plus the definitions it
references but never states.

Defined here because the manuscript leaves them open
-----------------------------------------------------
* ``R^2`` and ``UCR`` are named as deliverables in Table F1/M1 but never defined
  (amendment A-10). ``UCR`` uses the *predictive* standard deviation
  ``sqrt(sigma^2(x) + sigma_n^2)``; omitting the noise term would make a
  well-calibrated GP appear broken.
* ``S_min`` and ``Q_min(task)`` appear as columns in Tables B.4 and B.5 with no
  definition (amendments A-25/A-26). They are set to ``0.95x`` and ``0.98x`` the
  unoptimized means, the only values consistent with the SRR/QRR criteria.
* Amortization (``C_bo``, ``N*``, tariff cost) is not in Chapter 3 at all, but is
  computable from data already logged -- Table B.7 already carries a
  ``C_boTotal`` "Sum J" footer -- and answers the obvious question of whether the
  search paid for its own energy.

Standard library only.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Sequence

#: Chapter 3's decision thresholds.
EIR_IMPROVEMENT_THRESHOLD = 0.0  # EIR > 0 counts as improvement
EIR_SUBSTANTIAL_THRESHOLD = 15.0  # EIR >= 15% flagged as substantial
SRR_THRESHOLD = 95.0
QRR_THRESHOLD = 98.0
NPE_RELIABILITY_THRESHOLD = 10.0  # NPE < 10% treated as reliable
UCR_TARGET = 0.95

#: Residential electricity tariff in PHP per kWh, used for the amortization
#: card. Overridable; only ever presented as an illustrative figure.
DEFAULT_TARIFF_PHP_PER_KWH = 14.35

JOULES_PER_KWH = 3.6e6


# --------------------------------------------------------------------------- #
# Configuration success indicators
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class SuccessIndicators:
    """EIR / SRR / QRR and the resulting verdict."""

    eir_percent: float
    srr_percent: float
    qrr_percent: float
    energy_default: float
    energy_optimized: float
    speed_default: float
    speed_optimized: float
    quality_default: float
    quality_optimized: float
    srr_threshold: float = SRR_THRESHOLD
    qrr_threshold: float = QRR_THRESHOLD

    @property
    def energy_improved(self) -> bool:
        return self.eir_percent > EIR_IMPROVEMENT_THRESHOLD

    @property
    def substantial(self) -> bool:
        return self.eir_percent >= EIR_SUBSTANTIAL_THRESHOLD

    @property
    def speed_retained(self) -> bool:
        return self.srr_percent >= self.srr_threshold

    @property
    def quality_retained(self) -> bool:
        return self.qrr_percent >= self.qrr_threshold

    @property
    def is_bopis_optimal(self) -> bool:
        """Chapter 3's formal criterion: EIR > 0 AND SRR >= 95% AND QRR >= 98%."""
        return self.energy_improved and self.speed_retained and self.quality_retained

    @property
    def verdict(self) -> str:
        """A single word for the dashboard badge and the run summary."""
        if self.is_bopis_optimal:
            return "substantial" if self.substantial else "optimal"
        if not self.energy_improved:
            return "negative"  # consumed more energy than the default
        return "partial"  # saved energy but failed a retention threshold

    def failed_criteria(self) -> List[str]:
        failures: List[str] = []
        if not self.energy_improved:
            failures.append(f"EIR={self.eir_percent:.2f}% is not > 0")
        if not self.speed_retained:
            failures.append(
                f"SRR={self.srr_percent:.2f}% < {self.srr_threshold:g}%"
            )
        if not self.quality_retained:
            failures.append(
                f"QRR={self.qrr_percent:.2f}% < {self.qrr_threshold:g}%"
            )
        return failures

    def as_dict(self) -> Dict[str, object]:
        return {
            "eir_percent": self.eir_percent,
            "srr_percent": self.srr_percent,
            "qrr_percent": self.qrr_percent,
            "energy_improved": self.energy_improved,
            "substantial": self.substantial,
            "speed_retained": self.speed_retained,
            "quality_retained": self.quality_retained,
            "is_bopis_optimal": self.is_bopis_optimal,
            "verdict": self.verdict,
            "failed_criteria": self.failed_criteria(),
            "energy_default_j": self.energy_default,
            "energy_optimized_j": self.energy_optimized,
            "speed_default_tps": self.speed_default,
            "speed_optimized_tps": self.speed_optimized,
            "quality_default_f1": self.quality_default,
            "quality_optimized_f1": self.quality_optimized,
        }


def energy_improvement_ratio(default_j: float, optimized_j: float) -> float:
    """``EIR = [(E_default - E_BOPIS) / E_default] x 100%``."""
    if not default_j:
        return 0.0
    return 100.0 * (default_j - optimized_j) / default_j


def speed_retention_ratio(default_tps: float, optimized_tps: float) -> float:
    """``SRR = (S_BOPIS / S_default) x 100%``."""
    if not default_tps:
        return 0.0
    return 100.0 * optimized_tps / default_tps


def quality_retention_ratio(default_f1: float, optimized_f1: float) -> float:
    """``QRR = (Q_BOPIS / Q_default) x 100%``."""
    if not default_f1:
        return 0.0
    return 100.0 * optimized_f1 / default_f1


def success_indicators(
    energy_default: float,
    energy_optimized: float,
    speed_default: float,
    speed_optimized: float,
    quality_default: float,
    quality_optimized: float,
) -> SuccessIndicators:
    return SuccessIndicators(
        eir_percent=energy_improvement_ratio(energy_default, energy_optimized),
        srr_percent=speed_retention_ratio(speed_default, speed_optimized),
        qrr_percent=quality_retention_ratio(quality_default, quality_optimized),
        energy_default=energy_default,
        energy_optimized=energy_optimized,
        speed_default=speed_default,
        speed_optimized=speed_optimized,
        quality_default=quality_default,
        quality_optimized=quality_optimized,
    )


def s_min(default_tps: float) -> float:
    """``S_min`` for Table B.4: 95% of the unoptimized mean throughput."""
    return default_tps * SRR_THRESHOLD / 100.0


def q_min(default_f1: float) -> float:
    """``Q_min(task)`` for Table B.5: 98% of that task's unoptimized mean F1."""
    return default_f1 * QRR_THRESHOLD / 100.0


# --------------------------------------------------------------------------- #
# Surrogate reliability (Table M1 / B.11)
# --------------------------------------------------------------------------- #


class ReliabilityBasis:
    """Which set of predictions a reliability figure was computed from.

    The distinction is load-bearing, not bookkeeping (amendment A-40).

    ``ONE_STEP_AHEAD``
        ``mu(x)`` recorded immediately before each BO-guided configuration was
        run. **Systematically pessimistic**: Expected Improvement deliberately
        evaluates wherever the surrogate is least certain, so these are by
        construction the model's worst predictions. Measured on the simulator
        across ten seeds, one-step-ahead NPE averaged ~23% and never fell below
        10%, while leave-one-out NPE on the *same fitted models* averaged ~11%
        with R^2 ~ 0.94. Scoring the acquisition function's own probes measures
        exploration, not fit.
    ``LEAVE_ONE_OUT``
        Each observation predicted from the other n-1, with hyperparameters
        held fixed. The appropriate basis for judging whether the surrogate
        captured the energy response surface.
    """

    ONE_STEP_AHEAD = "one_step_ahead"
    LEAVE_ONE_OUT = "leave_one_out"


@dataclasses.dataclass
class SurrogateReliability:
    """How well the GP predicted energy, on a stated basis."""

    n: int
    mae: float
    npe_percent: float
    r_squared: float
    ucr: float
    mean_measured: float
    rmse: float
    basis: str = ReliabilityBasis.ONE_STEP_AHEAD

    @property
    def meets_npe_threshold(self) -> bool:
        """Chapter 3's researcher-defined criterion: NPE below 10%."""
        return self.npe_percent < NPE_RELIABILITY_THRESHOLD

    #: Retained for readability at call sites; identical to
    #: :attr:`meets_npe_threshold`.
    @property
    def reliable(self) -> bool:
        return self.meets_npe_threshold

    def as_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "n": self.n,
            "basis": self.basis,
            "mae": self.mae,
            "npe_percent": self.npe_percent,
            "r_squared": self.r_squared,
            "ucr": self.ucr,
            "ucr_target": UCR_TARGET,
            "rmse": self.rmse,
            "mean_measured_j": self.mean_measured,
            "reliable": self.reliable,
            "npe_threshold": NPE_RELIABILITY_THRESHOLD,
        }
        if self.basis == ReliabilityBasis.ONE_STEP_AHEAD:
            payload["caveat"] = (
                "One-step-ahead predictions are the acquisition function's own "
                "high-uncertainty probes and are therefore a pessimistic "
                "reliability estimate. Judge surrogate fit on the "
                "leave-one-out figures."
            )
        return payload


def surrogate_reliability(
    predicted: Sequence[float],
    measured: Sequence[float],
    predictive_std: Optional[Sequence[float]] = None,
    basis: str = ReliabilityBasis.ONE_STEP_AHEAD,
) -> SurrogateReliability:
    """MAE, NPE, R^2 and UCR for GP energy predictions.

    Args:
        predicted: ``mu(x_i)`` -- predicted *before* the configuration was run.
        measured: ``E_measured(x_i)`` -- recorded after execution.
        predictive_std: ``sqrt(sigma^2(x_i) + sigma_n^2)``. When supplied, UCR is
            the fraction of measurements falling inside ``mu +/- 1.96 * std``;
            for a correctly calibrated surrogate this should be near 0.95.
        basis: Which prediction set this is -- see :class:`ReliabilityBasis`.
    """
    pairs = [
        (float(p), float(m))
        for p, m in zip(predicted, measured)
        if p is not None and m is not None
        and not math.isnan(float(p)) and not math.isnan(float(m))
    ]
    if not pairs:
        nan = float("nan")
        return SurrogateReliability(0, nan, nan, nan, nan, nan, nan, basis=basis)

    n = len(pairs)
    errors = [abs(p - m) for p, m in pairs]
    mae = sum(errors) / n
    rmse = math.sqrt(sum((p - m) ** 2 for p, m in pairs) / n)
    mean_measured = sum(m for _p, m in pairs) / n
    npe = 100.0 * mae / mean_measured if mean_measured else float("nan")

    ss_res = sum((m - p) ** 2 for p, m in pairs)
    ss_tot = sum((m - mean_measured) ** 2 for _p, m in pairs)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if predictive_std is not None:
        covered = 0
        counted = 0
        for (p, m), sd in zip(pairs, predictive_std):
            if sd is None or math.isnan(float(sd)):
                continue
            counted += 1
            if abs(m - p) <= 1.96 * float(sd):
                covered += 1
        ucr = covered / counted if counted else float("nan")
    else:
        ucr = float("nan")

    return SurrogateReliability(
        n=n,
        mae=mae,
        npe_percent=npe,
        r_squared=r_squared,
        ucr=ucr,
        mean_measured=mean_measured,
        rmse=rmse,
        basis=basis,
    )


# --------------------------------------------------------------------------- #
# Convergence and sample efficiency
# --------------------------------------------------------------------------- #


def best_so_far(energies: Sequence[float]) -> List[float]:
    """``E*_k = min{ E_measured(x_i) : i <= k }`` -- a non-increasing curve."""
    curve: List[float] = []
    running = math.inf
    for value in energies:
        running = min(running, float(value))
        curve.append(running)
    return curve


def improvement_per_iteration(energies: Sequence[float]) -> List[float]:
    """``dE_k = E*_{k-1} - E*_k``: energy saved at each iteration.

    The first entry is 0.0 by convention (there is no prior best to improve on).
    Positive values mean a better configuration was found; a run of zeros
    indicates convergence or stagnation.
    """
    curve = best_so_far(energies)
    deltas = [0.0]
    for i in range(1, len(curve)):
        deltas.append(curve[i - 1] - curve[i])
    return deltas


def first_iteration_of(energies: Sequence[float], target_config_index: int) -> int:
    """1-based iteration at which the configuration at *target_config_index* ran."""
    return target_config_index + 1


def sample_efficiency_ratio(
    k_star_random_search: int, k_star_bopis: int
) -> Optional[float]:
    """``SER = k*_RandomSearch / k*_BOPIS``.

    ``SER > 1.0`` means BOPIS reached its selected configuration in fewer
    evaluations than random search did. Note that this is a **single-run point
    estimate** with no variance (amendment A-16): ``k*`` is just an iteration
    index, so one SER value is a single draw from a highly variable statistic.
    Report it as a mean over repeated seeded runs wherever possible -- see
    :func:`ser_summary`.
    """
    if not k_star_bopis:
        return None
    return k_star_random_search / k_star_bopis


def ser_summary(values: Sequence[float]) -> Dict[str, object]:
    """Mean, SD and win-rate of SER across repeated runs."""
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return {"n": 0, "mean": None, "sd": None, "fraction_above_1": None}
    n = len(clean)
    mean = sum(clean) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in clean) / (n - 1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "min": min(clean),
        "max": max(clean),
        "fraction_above_1": sum(1 for v in clean if v > 1.0) / n,
    }


def iterations_to_within(
    energies: Sequence[float], tolerance_fraction: float = 0.05
) -> Optional[int]:
    """First iteration whose best-so-far is within *tolerance_fraction* of final.

    A more stable companion to SER: it does not depend on which configuration was
    ultimately selected, only on how quickly the search got close to its own best
    value.
    """
    curve = best_so_far(energies)
    if not curve:
        return None
    final = curve[-1]
    if final <= 0:
        return None
    for i, value in enumerate(curve, start=1):
        if (value - final) / final <= tolerance_fraction:
            return i
    return None


# --------------------------------------------------------------------------- #
# Amortization
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Amortization:
    """Whether the search recovered the energy it spent."""

    calibration_energy_j: float  # C_bo
    energy_saved_per_prompt_j: float  # dE
    breakeven_prompts: Optional[int]  # N*
    tariff_php_per_kwh: float
    cost_per_1000_default_php: float
    cost_per_1000_optimized_php: float
    calibration_cost_php: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "calibration_energy_j": self.calibration_energy_j,
            "energy_saved_per_prompt_j": self.energy_saved_per_prompt_j,
            "breakeven_prompts": self.breakeven_prompts,
            "tariff_php_per_kwh": self.tariff_php_per_kwh,
            "cost_per_1000_default_php": self.cost_per_1000_default_php,
            "cost_per_1000_optimized_php": self.cost_per_1000_optimized_php,
            "cost_saved_per_1000_php": (
                self.cost_per_1000_default_php - self.cost_per_1000_optimized_php
            ),
            "calibration_cost_php": self.calibration_cost_php,
        }


def amortization(
    calibration_energy_j: float,
    energy_default_per_prompt_j: float,
    energy_optimized_per_prompt_j: float,
    tariff_php_per_kwh: float = DEFAULT_TARIFF_PHP_PER_KWH,
) -> Amortization:
    """Break-even analysis for the one-off cost of the search.

    BOPIS is an offline, pre-deployment procedure: the search runs once and the
    chosen configuration is then used for every subsequent inference. That makes
    the search's own energy a fixed cost, recovered after

        ``N* = ceil(C_bo / dE)``

    prompts, where ``dE`` is the per-prompt saving. ``N*`` is ``None`` when the
    selected configuration does not actually save energy, in which case the cost
    is never recovered.
    """
    delta = energy_default_per_prompt_j - energy_optimized_per_prompt_j
    breakeven = (
        int(math.ceil(calibration_energy_j / delta)) if delta > 0 else None
    )

    def php_per_1000(joules_per_prompt: float) -> float:
        return 1000.0 * joules_per_prompt / JOULES_PER_KWH * tariff_php_per_kwh

    return Amortization(
        calibration_energy_j=calibration_energy_j,
        energy_saved_per_prompt_j=delta,
        breakeven_prompts=breakeven,
        tariff_php_per_kwh=tariff_php_per_kwh,
        cost_per_1000_default_php=php_per_1000(energy_default_per_prompt_j),
        cost_per_1000_optimized_php=php_per_1000(energy_optimized_per_prompt_j),
        calibration_cost_php=calibration_energy_j / JOULES_PER_KWH * tariff_php_per_kwh,
    )


def joules_per_token(energy_j: float, generated_tokens: int) -> Optional[float]:
    """``J/token = E / N_generated``.

    Note that this charges *prefill* energy to generated tokens, so it retains a
    dependence on prompt length -- exactly what the normalization was meant to
    remove (amendment A-35). Where the backend reports the prefill/decode split,
    prefer the separate figures.
    """
    if not generated_tokens:
        return None
    return energy_j / generated_tokens
