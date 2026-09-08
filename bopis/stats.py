"""Inferential and descriptive statistics -- standard library only.

Chapter 3 specifies the Friedman test applied separately to each dependent
variable across the three related conditions (unoptimized default, random search
baseline, BOPIS-recommended), with a Nemenyi post-hoc comparison when the
Friedman result is significant, at ``alpha = 0.05``.

Why no scipy is needed
----------------------
The Friedman statistic is referred to a chi-square distribution with
``df = k - 1``. With three conditions ``df = 2``, and the chi-square survival
function at two degrees of freedom has an exact closed form::

    P(X > x) = exp(-x / 2)

so the p-value is computed exactly rather than approximated. A general
regularized incomplete gamma implementation is also provided
(:func:`chi2_sf`) so the module stays correct if the number of conditions ever
changes.

The Nemenyi critical difference uses the studentized range statistic
``q_alpha`` divided by ``sqrt(2)``, tabulated by Demsar (2006), "Statistical
Comparisons of Classifiers over Multiple Data Sets", *JMLR* 7:1-30, Table 5.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #

_MAX_ITER = 300
_EPS = 3e-16
_FPMIN = 1e-300


def _log_gamma(x: float) -> float:
    return math.lgamma(x)


def _gamma_p_series(a: float, x: float) -> float:
    """Lower regularized incomplete gamma ``P(a, x)`` by series expansion."""
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(_MAX_ITER):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - _log_gamma(a))


def _gamma_q_continued_fraction(a: float, x: float) -> float:
    """Upper regularized incomplete gamma ``Q(a, x)`` by continued fraction."""
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _MAX_ITER + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - _log_gamma(a)) * h


def chi2_sf(x: float, df: int) -> float:
    """Survival function ``P(X > x)`` for a chi-square variate with *df* df.

    Exact for ``df == 2``; otherwise the regularized incomplete gamma function.
    """
    if x <= 0.0:
        return 1.0
    if df == 2:
        return math.exp(-0.5 * x)
    a, xx = df / 2.0, x / 2.0
    if xx < a + 1.0:
        return 1.0 - _gamma_p_series(a, xx)
    return _gamma_q_continued_fraction(a, xx)


#: Studentized-range constants ``q_alpha / sqrt(2)`` for the Nemenyi test,
#: indexed by number of conditions ``k``. Demsar (2006), Table 5.
NEMENYI_Q = {
    0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
           7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164},
    0.10: {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589,
           7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920},
}


# --------------------------------------------------------------------------- #
# Descriptive statistics
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Descriptives:
    """Mean, standard deviation, min and max -- the Table B.8 statistics."""

    n: int
    mean: float
    sd: float
    minimum: float
    maximum: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "n": self.n,
            "mean": self.mean,
            "sd": self.sd,
            "min": self.minimum,
            "max": self.maximum,
        }


def describe(values: Sequence[float]) -> Descriptives:
    """Descriptive statistics with the *sample* standard deviation (n-1)."""
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return Descriptives(0, float("nan"), float("nan"), float("nan"), float("nan"))
    n = len(clean)
    mean = sum(clean) / n
    if n > 1:
        sd = math.sqrt(sum((v - mean) ** 2 for v in clean) / (n - 1))
    else:
        sd = 0.0
    return Descriptives(n, mean, sd, min(clean), max(clean))


def coefficient_of_variation(values: Sequence[float]) -> float:
    """``sd / mean``, the repeatability measure used for the noise floor."""
    stats = describe(values)
    if not stats.mean:
        return float("nan")
    return stats.sd / abs(stats.mean)


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #


def average_ranks(row: Sequence[float], higher_is_better: bool = False) -> List[float]:
    """Rank one block, assigning the average rank to tied values.

    Rank 1 is the *best* value: the smallest when ``higher_is_better`` is False
    (energy, memory) and the largest when it is True (speed, quality). Getting
    this orientation right per dependent variable is what makes the Nemenyi mean
    ranks interpretable.
    """
    k = len(row)
    order = sorted(range(k), key=lambda i: row[i], reverse=higher_is_better)
    ranks = [0.0] * k
    position = 0
    while position < k:
        span = position + 1
        while span < k and row[order[span]] == row[order[position]]:
            span += 1
        average = (position + span + 1) / 2.0  # ranks are 1-based
        for idx in range(position, span):
            ranks[order[idx]] = average
        position = span
    return ranks


# --------------------------------------------------------------------------- #
# Friedman test
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class FriedmanResult:
    """Outcome of a Friedman test over *k* related conditions and *n* blocks."""

    n_blocks: int
    k_conditions: int
    chi_square: float
    df: int
    p_value: float
    mean_ranks: List[float]
    rank_sums: List[float]
    tie_correction: float
    condition_names: List[str]
    alpha: float = 0.05

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_blocks": self.n_blocks,
            "k_conditions": self.k_conditions,
            "chi_square": self.chi_square,
            "df": self.df,
            "p_value": self.p_value,
            "significant": self.significant,
            "alpha": self.alpha,
            "mean_ranks": dict(zip(self.condition_names, self.mean_ranks)),
            "rank_sums": dict(zip(self.condition_names, self.rank_sums)),
            "tie_correction": self.tie_correction,
        }


def friedman(
    blocks: Sequence[Sequence[float]],
    condition_names: Optional[Sequence[str]] = None,
    higher_is_better: bool = False,
    alpha: float = 0.05,
) -> FriedmanResult:
    """Friedman test with tie correction.

    Args:
        blocks: One row per block (per prompt), each row holding one value per
            condition, in a consistent condition order.
        condition_names: Labels for the conditions.
        higher_is_better: Orientation of the dependent variable.
        alpha: Significance level.

    The uncorrected statistic is

        ``chi2 = 12 / (n k (k+1)) * sum_j R_j^2 - 3 n (k+1)``

    divided by the tie-correction factor

        ``1 - sum_i sum_g (t_gi^3 - t_gi) / (n k (k^2 - 1))``

    where ``t_gi`` are tie-group sizes within block ``i``. Ties are common here
    because quality scores repeat across configurations, so omitting the
    correction would understate the statistic.
    """
    if not blocks:
        raise ValueError("friedman requires at least one block")
    k = len(blocks[0])
    if k < 2:
        raise ValueError("friedman requires at least two conditions")
    if any(len(row) != k for row in blocks):
        raise ValueError("all blocks must have the same number of conditions")

    n = len(blocks)
    names = list(condition_names) if condition_names else [f"c{i}" for i in range(k)]

    rank_sums = [0.0] * k
    tie_term = 0.0
    for row in blocks:
        ranks = average_ranks(row, higher_is_better=higher_is_better)
        for j in range(k):
            rank_sums[j] += ranks[j]
        # Tie-group sizes in this block.
        counts: Dict[float, int] = {}
        for value in row:
            counts[value] = counts.get(value, 0) + 1
        tie_term += sum(t**3 - t for t in counts.values() if t > 1)

    raw = 12.0 / (n * k * (k + 1)) * sum(r * r for r in rank_sums) - 3.0 * n * (k + 1)

    denominator = n * k * (k * k - 1)
    correction = 1.0 - (tie_term / denominator) if denominator else 1.0
    if correction <= 0.0:
        # Every block fully tied: no evidence of any difference.
        chi_square, correction = 0.0, 1.0
    else:
        chi_square = raw / correction

    chi_square = max(chi_square, 0.0)
    df = k - 1
    return FriedmanResult(
        n_blocks=n,
        k_conditions=k,
        chi_square=chi_square,
        df=df,
        p_value=chi2_sf(chi_square, df),
        mean_ranks=[r / n for r in rank_sums],
        rank_sums=rank_sums,
        tie_correction=correction,
        condition_names=names,
        alpha=alpha,
    )


# --------------------------------------------------------------------------- #
# Nemenyi post-hoc
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class NemenyiResult:
    """Pairwise post-hoc comparison of mean ranks."""

    critical_difference: float
    alpha: float
    q_alpha: float
    pairs: List[Dict[str, object]]

    def significant_pairs(self) -> List[str]:
        return [p["pair"] for p in self.pairs if p["significant"]]  # type: ignore[misc]

    def as_dict(self) -> Dict[str, object]:
        return {
            "critical_difference": self.critical_difference,
            "alpha": self.alpha,
            "q_alpha": self.q_alpha,
            "pairs": self.pairs,
            "significant_pairs": self.significant_pairs(),
        }


def nemenyi(result: FriedmanResult, alpha: float = 0.05) -> NemenyiResult:
    """Nemenyi post-hoc test on the mean ranks of a Friedman result.

    ``CD = q_alpha * sqrt(k (k + 1) / (6 n))``; two conditions differ
    significantly when the absolute difference of their mean ranks exceeds
    ``CD``. Unlike a pairwise test, the Nemenyi procedure controls the
    family-wise error rate across all pairs, which is why Chapter 3 pairs it
    with Friedman rather than running three separate tests.
    """
    k, n = result.k_conditions, result.n_blocks
    table = NEMENYI_Q.get(alpha)
    if table is None:
        raise ValueError(f"no tabulated q_alpha for alpha={alpha}")
    if k not in table:
        raise ValueError(f"no tabulated q_alpha for k={k} conditions")
    q_alpha = table[k]
    cd = q_alpha * math.sqrt(k * (k + 1) / (6.0 * n))

    pairs: List[Dict[str, object]] = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(result.mean_ranks[i] - result.mean_ranks[j])
            pairs.append(
                {
                    "pair": f"{result.condition_names[i]} vs {result.condition_names[j]}",
                    "a": result.condition_names[i],
                    "b": result.condition_names[j],
                    "mean_rank_a": result.mean_ranks[i],
                    "mean_rank_b": result.mean_ranks[j],
                    "abs_diff": diff,
                    "critical_difference": cd,
                    "significant": diff > cd,
                }
            )
    return NemenyiResult(critical_difference=cd, alpha=alpha, q_alpha=q_alpha, pairs=pairs)


def compare_conditions(
    blocks: Sequence[Sequence[float]],
    condition_names: Sequence[str],
    higher_is_better: bool = False,
    alpha: float = 0.05,
) -> Dict[str, object]:
    """Friedman test plus, if significant, the Nemenyi post-hoc comparison."""
    result = friedman(
        blocks,
        condition_names=condition_names,
        higher_is_better=higher_is_better,
        alpha=alpha,
    )
    payload: Dict[str, object] = {"friedman": result.as_dict(), "nemenyi": None}
    if result.significant:
        payload["nemenyi"] = nemenyi(result, alpha=alpha).as_dict()
    return payload
