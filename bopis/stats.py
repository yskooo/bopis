"""Inferential and descriptive statistics -- standard library only.

Chapter 3 compares exactly **two** conditions on each dependent variable: the
random-search baseline and the BOPIS-recommended configuration. The prompts are
shared across both conditions, so the samples are paired, and the comparison
uses the **Wilcoxon signed-rank test** at ``alpha = 0.05``. The test is
non-parametric because energy, latency and BERTScore F1 are not assumed to be
normally distributed.

The unoptimized default is still measured in every run -- it is the reference a
condition is judged against -- but it is not part of the inferential comparison,
so no hypothesis test is run against it. A three-condition ordering test
(Friedman) and its post-hoc (Nemenyi) are deliberately absent: with only two
compared conditions a pairwise signed-rank test is the correct instrument.

Why no scipy is needed
---------------------
The signed-rank statistic is referred to a normal approximation with the tie
correction, using :func:`math.erfc` for the two-sided p-value. :func:`wilcoxon`
is therefore entirely self-contained, keeping the measurement core's
standard-library-only guarantee (amendment A-6).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Sequence

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
# Wilcoxon signed-rank test
# --------------------------------------------------------------------------- #


<<<<<<< Updated upstream
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
=======
def _rank_data(data: Sequence[float]) -> List[float]:
    """Helper to rank data, assigning average ranks to ties."""
    order = sorted(range(len(data)), key=data.__getitem__)
    ranks = [0.0] * len(data)
    position = 0
    while position < len(data):
        span = position + 1
        while span < len(data) and data[order[span]] == data[order[position]]:
            span += 1
        average = (position + span + 1) / 2.0
>>>>>>> Stashed changes
        for idx in range(position, span):
            ranks[order[idx]] = average
        position = span
    return ranks


<<<<<<< Updated upstream
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
=======
@dataclasses.dataclass
class WilcoxonResult:
    """Outcome of a Wilcoxon signed-rank test over 2 related conditions."""

    n_effective: int
    w_statistic: float
    z_statistic: float
    p_value: float
>>>>>>> Stashed changes
    tie_correction: float
    condition_names: List[str]
    alpha: float = 0.05

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def as_dict(self) -> Dict[str, object]:
        return {
<<<<<<< Updated upstream
            "n_blocks": self.n_blocks,
            "k_conditions": self.k_conditions,
            "chi_square": self.chi_square,
            "df": self.df,
            "p_value": self.p_value,
            "significant": self.significant,
            "alpha": self.alpha,
            "mean_ranks": dict(zip(self.condition_names, self.mean_ranks)),
            "rank_sums": dict(zip(self.condition_names, self.rank_sums)),
=======
            "n_effective": self.n_effective,
            "w_statistic": self.w_statistic,
            "z_statistic": self.z_statistic,
            "p_value": self.p_value,
            "significant": self.significant,
            "alpha": self.alpha,
>>>>>>> Stashed changes
            "tie_correction": self.tie_correction,
        }


<<<<<<< Updated upstream
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
=======
def wilcoxon(
    blocks: Sequence[Sequence[float]],
    condition_names: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
) -> WilcoxonResult:
    """Wilcoxon signed-rank test with tie correction and normal approximation.

    Args:
        blocks: One row per block (per prompt), each holding the two paired
            values, in a consistent condition order.
        condition_names: Labels for the two conditions.
        alpha: Significance level.

    Zero differences are discarded (they carry no sign information), and the
    remaining absolute differences are ranked with average ranks for ties. The
    statistic is the smaller of the positive- and negative-signed rank sums,
    with a continuity correction and the tie-corrected variance before the
    normal approximation.
    """
    if not blocks:
        raise ValueError("wilcoxon requires at least one block")
    k = len(blocks[0])
    if k != 2:
        raise ValueError("wilcoxon requires exactly two conditions")

    if condition_names is None:
        condition_names = ["Condition A", "Condition B"]

    diffs = []
    for block in blocks:
        diff = block[0] - block[1]
        if diff != 0:
            diffs.append(diff)

    n = len(diffs)
    if n == 0:
        return WilcoxonResult(0, 0.0, 0.0, 1.0, 0.0, list(condition_names), alpha)

    abs_diffs = [abs(d) for d in diffs]
    ranks = _rank_data(abs_diffs)

    w_pos = sum(r for d, r in zip(diffs, ranks) if d > 0)
    w_neg = sum(r for d, r in zip(diffs, ranks) if d < 0)

    w_stat = min(w_pos, w_neg)

    t_sum = 0
    unique_vals = set(abs_diffs)
    for v in unique_vals:
        t = abs_diffs.count(v)
        if t > 1:
            t_sum += (t ** 3 - t)

    expected_w = n * (n + 1) / 4.0
    var_w = (n * (n + 1) * (2 * n + 1)) / 24.0 - (t_sum / 48.0)

    if var_w == 0:
        z = 0.0
        p_val = 1.0
    else:
        correction = 0.5 if abs(w_stat - expected_w) > 0.5 else 0.0
        if w_stat < expected_w:
            z = (w_stat - expected_w + correction) / math.sqrt(var_w)
        else:
            z = (w_stat - expected_w - correction) / math.sqrt(var_w)
        p_val = math.erfc(abs(z) / math.sqrt(2.0))

    return WilcoxonResult(n, w_stat, z, p_val, t_sum / 48.0, list(condition_names), alpha)
>>>>>>> Stashed changes


def compare_conditions(
    blocks: Sequence[Sequence[float]],
    condition_names: Sequence[str],
    higher_is_better: bool = False,
    alpha: float = 0.05,
) -> Dict[str, object]:
<<<<<<< Updated upstream
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
=======
    """Wilcoxon signed-rank test comparing random_search and bopis.

    The three-way design carries an unoptimized *reference* condition, but the
    hypothesis test is the paired comparison of the random-search baseline and
    the BOPIS-recommended configuration -- so only those two columns are handed
    to :func:`wilcoxon`.

    *higher_is_better* is accepted for call-site compatibility and is unused:
    a two-sided signed-rank test is symmetric, so swapping the two conditions
    flips ``z`` but leaves ``w_statistic`` and the p-value unchanged.
    """
    try:
        rs_idx = condition_names.index("random_search")
        bp_idx = condition_names.index("bopis")
    except ValueError:
        rs_idx, bp_idx = 1, 2

    wilcoxon_blocks = [[row[rs_idx], row[bp_idx]] for row in blocks]

    wilcox_result = wilcoxon(
        wilcoxon_blocks,
        condition_names=["random_search", "bopis"],
        alpha=alpha,
    )

    return {
        "wilcoxon": wilcox_result.as_dict(),
        "n_blocks": len(blocks),
        "conditions": ["random_search", "bopis"],
    }
>>>>>>> Stashed changes


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals
# --------------------------------------------------------------------------- #


def bootstrap_ci(
    values: Sequence[float],
    statistic: str = "mean",
    n_boot: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Non-parametric bootstrap confidence interval for a summary statistic.

    Resamples *values* with replacement *n_boot* times and reports the
    percentile interval at the requested *confidence* level.  The standard
    library :mod:`random` module provides the PRNG; no scipy dependency is
    introduced.

    Args:
        values: Observed sample (e.g. one condition's energy-per-prompt vector).
        statistic: ``"mean"`` or ``"median"``.
        confidence: Central interval width, default 0.95 → 2.5th / 97.5th
            percentiles.
        seed: PRNG seed for reproducibility.

    Returns:
        A dict with keys ``point``, ``ci_lo``, ``ci_hi``, ``se`` (bootstrap
        standard error).
    """
    import random as _random

    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return {"point": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "se": float("nan")}

    n = len(clean)
    rng = _random.Random(seed)

    if statistic == "median":
        def _stat(sample: Sequence[float]) -> float:
            s = sorted(sample)
            mid = len(s) // 2
            return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0
    else:  # mean
        def _stat(sample: Sequence[float]) -> float:
            return sum(sample) / len(sample)

    point = _stat(clean)
    boot_stats: List[float] = []
    for _ in range(n_boot):
        sample = [clean[rng.randrange(n)] for _ in range(n)]
        boot_stats.append(_stat(sample))
    boot_stats.sort()

    alpha = (1.0 - confidence) / 2.0
    lo_idx = int(math.floor(alpha * n_boot))
    hi_idx = int(math.floor((1.0 - alpha) * n_boot)) - 1
    lo_idx = max(0, min(lo_idx, n_boot - 1))
    hi_idx = max(0, min(hi_idx, n_boot - 1))

    se = math.sqrt(
        sum((b - point) ** 2 for b in boot_stats) / (n_boot - 1)
    )
    return {
        "point": point,
        "ci_lo": boot_stats[lo_idx],
        "ci_hi": boot_stats[hi_idx],
        "se": se,
    }
