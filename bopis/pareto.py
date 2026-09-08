"""Pareto dominance, non-dominated sorting, hypervolume, and ``x*`` selection.

Objective vector, per Chapter 3::

    f(x) = { E(x), -S(x), -Q(x) }

Energy is minimized; speed and quality are maximized, and are therefore negated
so that every objective is minimized internally.

Dominance (Chapter 3)::

    x dominates x'  iff  E(x) <= E(x') AND S(x) >= S(x') AND Q(x) >= Q(x')
                         with at least one strict inequality

Two departures from the manuscript
----------------------------------
* **Hypervolume is normalized** (amendment A-15). The equation as written yields
  a quantity with units J x (tok/s) x F1, so whichever axis happens to have the
  largest numeric range dominates the indicator, and values are incomparable
  across machines or models. Each objective is min-max scaled onto ``[0, 1]``
  against the reference point before the volume is computed, giving
  ``HV in [0, 1]`` where 1 means the front reaches the best observed value on
  every axis.
* **Selection has a defined fallback** (amendment A-34). Chapter 3 specifies
  ``x* = argmax EIR`` subject to ``SRR >= 95%`` and ``QRR >= 98%`` but says
  nothing about the case where no front member satisfies both -- a likely
  outcome on constrained hardware. A fixed relaxation ladder is applied and the
  resulting :class:`SelectionStatus` is always reported, so a relaxed selection
  can never be mistaken for a clean one.

Standard library only.
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from bopis.config_space import Config


@dataclasses.dataclass
class Objectives:
    """Measured objectives for one configuration."""

    energy_j: float  # E(x) -- minimize
    tokens_per_s: float  # S(x) -- maximize
    quality_f1: float  # Q(x) -- maximize

    def to_minimization(self) -> Tuple[float, float, float]:
        """``(E, -S, -Q)``: all three to be minimized."""
        return (self.energy_j, -self.tokens_per_s, -self.quality_f1)


@dataclasses.dataclass
class Evaluation:
    """One evaluated configuration: the unit of Pareto analysis."""

    config: Config
    objectives: Objectives
    iteration: int = -1
    source: str = ""  # "seed" | "bo" | "random_search" | "default"

    @property
    def energy_j(self) -> float:
        return self.objectives.energy_j

    @property
    def tokens_per_s(self) -> float:
        return self.objectives.tokens_per_s

    @property
    def quality_f1(self) -> float:
        return self.objectives.quality_f1


class SelectionStatus:
    """Which rung of the relaxation ladder produced ``x*``."""

    OPTIMAL = "optimal"  # SRR >= 95% and QRR >= 98%
    RELAXED_QRR = "relaxed_qrr_95"
    RELAXED_SRR = "relaxed_srr_90"
    MIN_ENERGY_FALLBACK = "min_energy_fallback"
    EMPTY = "empty"


# --------------------------------------------------------------------------- #
# Dominance and fronts
# --------------------------------------------------------------------------- #


def dominates(a: Objectives, b: Objectives) -> bool:
    """True when *a* Pareto-dominates *b*, per the Chapter 3 definition."""
    at_least_as_good = (
        a.energy_j <= b.energy_j
        and a.tokens_per_s >= b.tokens_per_s
        and a.quality_f1 >= b.quality_f1
    )
    if not at_least_as_good:
        return False
    strictly_better = (
        a.energy_j < b.energy_j
        or a.tokens_per_s > b.tokens_per_s
        or a.quality_f1 > b.quality_f1
    )
    return strictly_better


def pareto_front(evaluations: Sequence[Evaluation]) -> List[Evaluation]:
    """The non-dominated subset of *evaluations*.

    ``PF = { x in X_evaluated : there is no x' that dominates x }``. Brute-force
    pairwise comparison: with at most ~30 evaluations the O(n^2) cost is
    irrelevant and the implementation stays obviously correct.
    """
    front: List[Evaluation] = []
    for candidate in evaluations:
        if not any(
            dominates(other.objectives, candidate.objectives)
            for other in evaluations
            if other is not candidate
        ):
            front.append(candidate)
    return front


def front_mask(evaluations: Sequence[Evaluation]) -> List[bool]:
    """Per-evaluation front membership, aligned with *evaluations*."""
    front_ids = {id(e) for e in pareto_front(evaluations)}
    return [id(e) in front_ids for e in evaluations]


# --------------------------------------------------------------------------- #
# Hypervolume
# --------------------------------------------------------------------------- #


def _normalize_points(
    points: Sequence[Tuple[float, float, float]],
    reference: Tuple[float, float, float],
) -> Tuple[List[Tuple[float, float, float]], Tuple[float, float, float]]:
    """Min-max scale minimization-space points so the reference maps to 1.

    The reference is first **clamped component-wise to the nadir** of the front:

        ``r_eff[d] = max(r[d], max over front of p[d])``

    Without that clamp, hypervolume against the unoptimized default is
    degenerately zero for any realistic run (amendment A-15). The default uses
    F32 -- the highest-precision, highest-quality variant in the space -- so no
    configuration can beat it on quality. Chapter 3's reference point
    ``r = (E_default, -S_default, -Q_default)`` therefore has *zero span* on the
    quality axis, and a front member must be strictly better than the reference
    on every axis to enclose any volume at all. Measured on a real run: nine
    front members, none beating the default's 0.8615 F1, HV = 0 regardless of
    the 41% energy saving.

    Clamping to the nadir is the standard multi-objective convention -- the
    reference is the worst corner of the considered set -- and it makes the
    indicator measure what Chapter 3 describes it as measuring: how much of the
    achievable trade-off space the front covers.

    An axis with zero span even after clamping is genuinely constant across the
    front *and* the reference, meaning no improvement was available there; such
    an axis maps to 1.0 so it contributes no depth, rather than to 0.0 which
    would credit the front with improvement it never achieved.
    """
    if not points:
        return [], (1.0, 1.0, 1.0)

    effective_reference = [
        max(reference[d], max(p[d] for p in points)) for d in range(3)
    ]
    best = [min(min(p[d] for p in points), reference[d]) for d in range(3)]
    span = [effective_reference[d] - best[d] for d in range(3)]

    def scale(value: float, d: int) -> float:
        if span[d] <= 0.0:
            return 1.0
        return (value - best[d]) / span[d]

    scaled = [
        (scale(p[0], 0), scale(p[1], 1), scale(p[2], 2)) for p in points
    ]
    return scaled, (1.0, 1.0, 1.0)


def _hypervolume_2d(
    points: Sequence[Tuple[float, float]], reference: Tuple[float, float]
) -> float:
    """Area of the union of boxes ``[p, reference]`` for minimization."""
    usable = [p for p in points if p[0] < reference[0] and p[1] < reference[1]]
    if not usable:
        return 0.0

    # Reduce to the 2-D staircase: x ascending, y strictly descending.
    usable.sort(key=lambda p: (p[0], p[1]))
    staircase: List[Tuple[float, float]] = []
    best_y = math.inf
    for x, y in usable:
        if y < best_y:
            staircase.append((x, y))
            best_y = y

    area = 0.0
    prev_y = reference[1]
    for x, y in staircase:
        area += (reference[0] - x) * (prev_y - y)
        prev_y = y
    return area


def hypervolume(
    front: Sequence[Evaluation],
    reference: Objectives,
    normalize: bool = True,
) -> float:
    """Hypervolume dominated by *front*, bounded by *reference*.

    The reference point is the unoptimized default -- ``r = (E_default,
    -S_default, -Q_default)`` -- so HV measures how much better than the default
    the front is, across all three objectives at once.

    Computed exactly by sweeping the third objective and accumulating 2-D slice
    areas. With ``normalize=True`` (the default, and the only form that should be
    reported) the result lies in ``[0, 1]``.
    """
    if not front:
        return 0.0

    points = [e.objectives.to_minimization() for e in front]
    ref = reference.to_minimization()

    if normalize:
        points, ref = _normalize_points(points, ref)

    # Keep only points that actually dominate the reference on every axis.
    usable = [p for p in points if all(p[d] < ref[d] for d in range(3))]
    if not usable:
        return 0.0

    usable.sort(key=lambda p: p[2])
    volume = 0.0
    for i, point in enumerate(usable):
        z_lo = point[2]
        z_hi = usable[i + 1][2] if i + 1 < len(usable) else ref[2]
        depth = z_hi - z_lo
        if depth <= 0.0:
            continue
        slice_area = _hypervolume_2d(
            [(p[0], p[1]) for p in usable[: i + 1]], (ref[0], ref[1])
        )
        volume += slice_area * depth
    return volume


def hypervolume_monte_carlo(
    front: Sequence[Evaluation],
    reference: Objectives,
    samples: int = 200_000,
    seed: int = 0,
    normalize: bool = True,
) -> float:
    """Monte-Carlo hypervolume estimate, used only to cross-check the exact form.

    Not for reporting: this exists so ``tests/test_pareto.py`` can validate the
    sweep implementation against an independent estimator.
    """
    if not front:
        return 0.0

    points = [e.objectives.to_minimization() for e in front]
    ref = reference.to_minimization()
    if normalize:
        points, ref = _normalize_points(points, ref)

    usable = [p for p in points if all(p[d] < ref[d] for d in range(3))]
    if not usable:
        return 0.0

    lows = [min(p[d] for p in usable) for d in range(3)]
    box = 1.0
    for d in range(3):
        box *= ref[d] - lows[d]
    if box <= 0.0:
        return 0.0

    rng = random.Random(seed)
    hits = 0
    for _ in range(samples):
        probe = tuple(rng.uniform(lows[d], ref[d]) for d in range(3))
        if any(all(p[d] <= probe[d] for d in range(3)) for p in usable):
            hits += 1
    return box * hits / samples


# --------------------------------------------------------------------------- #
# x* selection
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Selection:
    """The configuration BOPIS recommends, with full provenance."""

    evaluation: Optional[Evaluation]
    status: str
    eir_percent: Optional[float] = None
    srr_percent: Optional[float] = None
    qrr_percent: Optional[float] = None
    n_front: int = 0
    n_candidates_considered: int = 0
    notes: str = ""

    @property
    def config(self) -> Optional[Config]:
        return self.evaluation.config if self.evaluation else None

    @property
    def is_clean(self) -> bool:
        """True only when both retention thresholds were met unrelaxed."""
        return self.status == SelectionStatus.OPTIMAL

    def as_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "status": self.status,
            "is_clean": self.is_clean,
            "eir_percent": self.eir_percent,
            "srr_percent": self.srr_percent,
            "qrr_percent": self.qrr_percent,
            "n_front": self.n_front,
            "n_candidates_considered": self.n_candidates_considered,
            "notes": self.notes,
        }
        if self.evaluation:
            payload["config"] = self.evaluation.config.as_row()
            payload["config_key"] = self.evaluation.config.key()
            payload["energy_j"] = self.evaluation.energy_j
            payload["tokens_per_s"] = self.evaluation.tokens_per_s
            payload["quality_f1"] = self.evaluation.quality_f1
        return payload


def _ratios(
    candidate: Evaluation, baseline: Objectives
) -> Tuple[float, float, float]:
    """``(EIR, SRR, QRR)`` in percent for *candidate* against *baseline*."""
    eir = (
        100.0 * (baseline.energy_j - candidate.energy_j) / baseline.energy_j
        if baseline.energy_j
        else 0.0
    )
    srr = (
        100.0 * candidate.tokens_per_s / baseline.tokens_per_s
        if baseline.tokens_per_s
        else 0.0
    )
    qrr = (
        100.0 * candidate.quality_f1 / baseline.quality_f1
        if baseline.quality_f1
        else 0.0
    )
    return eir, srr, qrr


#: Relaxation ladder: ``(status, min SRR %, min QRR %)`` in order of preference.
_LADDER: Tuple[Tuple[str, float, float], ...] = (
    (SelectionStatus.OPTIMAL, 95.0, 98.0),
    (SelectionStatus.RELAXED_QRR, 95.0, 95.0),
    (SelectionStatus.RELAXED_SRR, 90.0, 95.0),
)


def select_xstar(
    evaluations: Sequence[Evaluation],
    baseline: Objectives,
    srr_threshold: float = 95.0,
    qrr_threshold: float = 98.0,
) -> Selection:
    """Select ``x*`` from the Pareto front of *evaluations*.

    ``x* = argmax EIR(x)`` subject to ``x in PF``, ``SRR(x) >= srr_threshold``
    and ``QRR(x) >= qrr_threshold``. When that set is empty the ladder relaxes
    QRR to 95%, then SRR to 90%, and finally falls back to the lowest-energy
    front member. The rung used is recorded in
    :attr:`Selection.status`; only :data:`SelectionStatus.OPTIMAL` may be
    reported as a successful optimization.
    """
    if not evaluations:
        return Selection(None, SelectionStatus.EMPTY, notes="no evaluations supplied")

    front = pareto_front(evaluations)
    ladder = (
        (_LADDER[0][0], srr_threshold, qrr_threshold),
    ) + _LADDER[1:]

    for status, min_srr, min_qrr in ladder:
        admissible: List[Tuple[Evaluation, float, float, float]] = []
        for candidate in front:
            eir, srr, qrr = _ratios(candidate, baseline)
            if srr >= min_srr and qrr >= min_qrr:
                admissible.append((candidate, eir, srr, qrr))
        if admissible:
            best, eir, srr, qrr = max(admissible, key=lambda row: row[1])
            note = (
                ""
                if status == SelectionStatus.OPTIMAL
                else f"thresholds relaxed to SRR>={min_srr:g}%, QRR>={min_qrr:g}%"
            )
            return Selection(
                evaluation=best,
                status=status,
                eir_percent=eir,
                srr_percent=srr,
                qrr_percent=qrr,
                n_front=len(front),
                n_candidates_considered=len(admissible),
                notes=note,
            )

    fallback = min(front, key=lambda e: e.energy_j)
    eir, srr, qrr = _ratios(fallback, baseline)
    return Selection(
        evaluation=fallback,
        status=SelectionStatus.MIN_ENERGY_FALLBACK,
        eir_percent=eir,
        srr_percent=srr,
        qrr_percent=qrr,
        n_front=len(front),
        n_candidates_considered=0,
        notes=(
            "no Pareto member met any retention threshold; fell back to the "
            "minimum-energy front member. This is NOT a successful optimization."
        ),
    )
