"""Expected Improvement acquisition function.

Correction to the manuscript (amendment A-2)
---------------------------------------------
Chapter 3 gives

    EI(x) = (mu(x) - f(x+)) * Phi(Z) + sigma(x) * phi(Z)
    Z     = (mu(x) - f(x+)) / sigma(x)

with ``f(x+)`` defined as "the best energy value observed so far". Because BOPIS
**minimizes** energy, "best" means *lowest*, so improvement at ``x`` is
``f(x+) - mu(x)``. As printed the sign is inverted: the expression rewards
configurations predicted to consume *more* energy than the incumbent, and a
search driven by it would converge on the worst configuration in the space.
This module implements the correct minimization form

    imp   = f(x+) - mu(x)
    Z     = imp / sigma(x)
    EI(x) = imp * Phi(Z) + sigma(x) * phi(Z)

``tests/test_acquisition.py`` asserts that EI prefers lower ``mu``; that test
fails against the formula as printed, which is what keeps the correction from
silently regressing.

Standard library only.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar

#: Below this posterior standard deviation a point is treated as fully known and
#: its Expected Improvement is exactly zero.
SIGMA_FLOOR = 1e-12

T = TypeVar("T")

#: A GP posterior: feature vector -> ``(mu, sigma)``.
Predictor = Callable[[Sequence[float]], Tuple[float, float]]


def standard_normal_cdf(z: float) -> float:
    """``Phi(z)`` via :func:`math.erf`."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def standard_normal_pdf(z: float) -> float:
    """``phi(z)``."""
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def expected_improvement(
    mu: float,
    sigma: float,
    f_best: float,
    xi: float = 0.0,
) -> float:
    """Expected Improvement for a **minimization** objective.

    Args:
        mu: GP posterior mean at the candidate.
        sigma: GP posterior standard deviation at the candidate.
        f_best: Best (lowest) objective value observed so far, ``f(x+)``.
        xi: Optional exploration margin; the improvement must exceed the
            incumbent by ``xi`` to count. Chapter 3 specifies no such margin, so
            the default of 0.0 reproduces the documented acquisition exactly.

    Returns:
        A non-negative Expected Improvement. Zero when the candidate is already
        known (``sigma`` at or below :data:`SIGMA_FLOOR`).
    """
    if sigma <= SIGMA_FLOOR:
        return 0.0
    improvement = f_best - mu - xi
    z = improvement / sigma
    return improvement * standard_normal_cdf(z) + sigma * standard_normal_pdf(z)


def probability_at_least(mu: float, sigma: float, threshold: float) -> float:
    """``P(f(x) >= threshold)`` under a Gaussian posterior ``N(mu, sigma^2)``.

    The feasibility factor of constrained Expected Improvement (Gardner et
    al., 2014; Gelbart, Snoek and Adams, 2014). With no posterior spread it is
    the indicator of the prediction meeting the threshold.
    """
    if sigma <= SIGMA_FLOOR:
        return 1.0 if mu >= threshold else 0.0
    return standard_normal_cdf((mu - threshold) / sigma)


def argmax_constrained_ei(
    candidates: Sequence[T],
    encode: Callable[[T], Sequence[float]],
    predict: Callable[[Sequence[float]], Tuple[float, float]],
    f_best: Optional[float],
    constraints: Sequence[Tuple[Predictor, float]],
    xi: float = 0.0,
    exclude: Optional[Sequence[T]] = None,
) -> Tuple[Optional[T], float, float]:
    """Exhaustively maximize ``EI(x) * prod_k P(c_k(x) >= threshold_k)``.

    *constraints* pairs a posterior predictor for each constrained objective
    (quality, speed) with its floor. *f_best* is the lowest energy among
    observations that **met every floor**; when there is none yet it is
    ``None`` and the acquisition is the probability of feasibility alone --
    until a feasible point is known, finding one is the only improvement that
    counts (Gelbart et al., 2014).

    Returns ``(best_candidate, best_score, its_probability_of_feasibility)``.
    """
    excluded = set(exclude or ())
    best: Optional[T] = None
    best_score = -math.inf
    best_pf = 0.0
    for candidate in candidates:
        if candidate in excluded:
            continue
        features = encode(candidate)
        pf = 1.0
        for constraint_predict, threshold in constraints:
            c_mu, c_sigma = constraint_predict(features)
            pf *= probability_at_least(c_mu, c_sigma, threshold)
        if f_best is None:
            score = pf
        else:
            mu, sigma = predict(features)
            score = expected_improvement(mu, sigma, f_best, xi=xi) * pf
        if score > best_score:
            best, best_score, best_pf = candidate, score, pf
    if best is None:
        return None, 0.0, 0.0
    return best, best_score, best_pf


def argmax_expected_improvement(
    candidates: Sequence[T],
    encode: Callable[[T], Sequence[float]],
    predict: Callable[[Sequence[float]], Tuple[float, float]],
    f_best: float,
    xi: float = 0.0,
    exclude: Optional[Sequence[T]] = None,
) -> Tuple[Optional[T], float, List[float]]:
    """Exhaustively maximize EI over *candidates*.

    The feasible configuration space is discrete and small -- at most
    ``4 x 4 x 4 x 4 x 3 = 768`` points, and typically far fewer after the
    Table H1 and model-size rules -- so the acquisition maximum is found by
    direct enumeration. No inner continuous optimizer is needed, and unlike a
    continuous relaxation this returns the *exact* argmax over the admissible
    set, with no risk of proposing a configuration that cannot be run.

    Args:
        candidates: The full feasible set.
        encode: Maps a candidate to its GP feature vector.
        predict: Maps a feature vector to ``(mu, sigma)``.
        f_best: Best objective value observed so far.
        xi: Exploration margin, passed through to :func:`expected_improvement`.
        exclude: Already-evaluated candidates, skipped.

    Returns:
        ``(best_candidate, best_ei, all_ei)``. ``best_candidate`` is ``None``
        only when every candidate is excluded. ``all_ei`` is aligned with
        *candidates* and carries ``-inf`` for excluded entries, so callers can
        log the full acquisition surface.
    """
    excluded = set(exclude or ())
    scores: List[float] = []
    best: Optional[T] = None
    best_score = -math.inf

    for candidate in candidates:
        if candidate in excluded:
            scores.append(-math.inf)
            continue
        mu, sigma = predict(encode(candidate))
        score = expected_improvement(mu, sigma, f_best, xi=xi)
        scores.append(score)
        if score > best_score:
            best_score, best = score, candidate

    if best is None:
        return None, 0.0, scores
    return best, best_score, scores
