"""Task taxonomy and the task-informed precision prior (Table T1).

Chapter 3 introduces a researcher-defined prior ``P(precision | task_type)``
that weights the initial random sampling phase of the Bayesian Optimization
engine, operationalizing Ma et al. (2023)'s finding that NLP task types differ
in their sensitivity to quantization-induced degradation.

Two corrections to Table T1
---------------------------
**A-4: the table has three precision columns for a four-variant space.**
Table T1 lists ``P(FP32)``, ``P(FP16)`` and ``P(INT8)``, but the configuration
space contains F32, F16, **Q8_0 and Q4_K_M**. As printed, Q4_K_M receives zero
prior mass and would never be drawn during seeding -- despite being the variant
most likely to win on energy. The INT8 mass is therefore split between Q8_0 and
Q4_K_M by quality sensitivity: the share going to the more aggressive Q4_K_M is
:data:`_Q4_SHARE_BY_SENSITIVITY` (0.25 for High, 0.40 for Medium, 0.50 for Low).
Every row still sums to 1.0.

**A-12: Dolly's ``general_qa`` is not "general instruction following".**
Table T1's last row is labelled "General Instr.", but the corresponding
Databricks Dolly 15k category is ``general_qa`` -- open-domain question
answering without a supplied context. The literal Dolly ``category`` strings are
used as canonical keys here so the stratification is reproducible.

Standard library only.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from bopis import config_space as cs


class Sensitivity:
    """Quality sensitivity to quantization (Table T1, column 2)."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclasses.dataclass(frozen=True)
class TaskType:
    """One task category."""

    key: str  # Dolly's literal `category` value -- the canonical identifier
    label: str  # Human-readable name used in tables and the dashboard
    sensitivity: str
    has_context: bool  # whether Dolly rows in this category carry a `context`


#: The eight task categories, keyed by Dolly's literal ``category`` field.
#:
#: ``has_context`` matters because the three context-bearing categories are
#: exactly the ones Table T1 rates most quality-sensitive, and they carry
#: substantially longer prompts (mean ~300 tokens vs ~14 for the rest).
TASK_TYPES: Tuple[TaskType, ...] = (
    TaskType("open_qa", "Open QA", Sensitivity.LOW, has_context=False),
    TaskType("closed_qa", "Closed QA", Sensitivity.HIGH, has_context=True),
    TaskType("summarization", "Summarization", Sensitivity.LOW, has_context=True),
    TaskType("classification", "Classification", Sensitivity.LOW, has_context=False),
    TaskType("creative_writing", "Creative Writing", Sensitivity.LOW, has_context=False),
    TaskType("brainstorming", "Brainstorming", Sensitivity.LOW, has_context=False),
    TaskType(
        "information_extraction",
        "Information Extraction",
        Sensitivity.HIGH,
        has_context=True,
    ),
    TaskType("general_qa", "General QA", Sensitivity.MEDIUM, has_context=False),
)

TASK_BY_KEY: Dict[str, TaskType] = {t.key: t for t in TASK_TYPES}
TASK_KEYS: Tuple[str, ...] = tuple(t.key for t in TASK_TYPES)


# --------------------------------------------------------------------------- #
# Table T1
# --------------------------------------------------------------------------- #

#: Table T1's three original columns, retained verbatim so the derivation of the
#: four-column version below is auditable against the manuscript.
_T1_ORIGINAL: Dict[str, Tuple[float, float, float]] = {
    #                       P(FP32) P(FP16) P(INT8)
    "open_qa": (0.15, 0.45, 0.40),
    "closed_qa": (0.35, 0.45, 0.20),
    "summarization": (0.15, 0.40, 0.45),
    "classification": (0.10, 0.35, 0.55),
    "creative_writing": (0.15, 0.45, 0.40),
    "brainstorming": (0.10, 0.35, 0.55),
    "information_extraction": (0.35, 0.45, 0.20),
    "general_qa": (0.20, 0.45, 0.35),
}

#: Share of the original INT8 mass assigned to Q4_K_M rather than Q8_0. More
#: quality-sensitive tasks keep more mass on the safer 8-bit variant.
_Q4_SHARE_BY_SENSITIVITY: Dict[str, float] = {
    Sensitivity.HIGH: 0.25,
    Sensitivity.MEDIUM: 0.40,
    Sensitivity.LOW: 0.50,
}


def _build_prior() -> Dict[str, Dict[str, float]]:
    prior: Dict[str, Dict[str, float]] = {}
    for key, (p_f32, p_f16, p_int8) in _T1_ORIGINAL.items():
        share = _Q4_SHARE_BY_SENSITIVITY[TASK_BY_KEY[key].sensitivity]
        prior[key] = {
            "F32": p_f32,
            "F16": p_f16,
            "Q8_0": p_int8 * (1.0 - share),
            "Q4_K_M": p_int8 * share,
        }
    return prior


#: ``P(precision | task_type)`` over all four GGUF variants. Replaces Table T1.
PRECISION_PRIOR: Dict[str, Dict[str, float]] = _build_prior()


def prior_for_task(task_key: str) -> Dict[str, float]:
    """``P(precision | task_key)``. Raises ``KeyError`` for unknown tasks."""
    return dict(PRECISION_PRIOR[task_key])


def dataset_prior(task_proportions: Mapping[str, float]) -> Dict[str, float]:
    """The dataset-level prior of Chapter 3.

    ``P(precision) = sum_t [ P(precision | task_t) * P(task_t) ]``

    Args:
        task_proportions: ``P(task_t)`` -- the share of the evaluation sample in
            each category. Need not be normalized; it is normalized here.
    """
    total = sum(task_proportions.values())
    if total <= 0:
        raise ValueError("task_proportions must sum to a positive value")

    combined = {variant: 0.0 for variant in cs.P_VALUES}
    for task_key, weight in task_proportions.items():
        if task_key not in PRECISION_PRIOR:
            raise KeyError(f"unknown task category: {task_key!r}")
        share = weight / total
        for variant, probability in PRECISION_PRIOR[task_key].items():
            combined[variant] += share * probability
    return combined


def mask_and_renormalize(
    prior: Mapping[str, float], permitted: Iterable[str]
) -> Dict[str, float]:
    """Restrict *prior* to *permitted* variants and renormalize to sum to 1.

    Necessary because the Table H1 rules can exclude precision variants
    entirely: on a 2 GB GPU only Q8_0 and Q4_K_M survive, and an unmasked prior
    would waste most of its seeding budget proposing infeasible F32/F16
    configurations. Falls back to a uniform distribution when the permitted set
    carries no prior mass at all.
    """
    allowed = [v for v in cs.P_VALUES if v in set(permitted)]
    if not allowed:
        raise ValueError("no permitted precision variants")

    masked = {v: max(0.0, float(prior.get(v, 0.0))) for v in allowed}
    total = sum(masked.values())
    if total <= 0:
        uniform = 1.0 / len(allowed)
        return {v: uniform for v in allowed}
    return {v: p / total for v, p in masked.items()}


# --------------------------------------------------------------------------- #
# Prior-weighted seed sampling
# --------------------------------------------------------------------------- #


def sample_seed_configs(
    space: Sequence[cs.Config],
    n_seeds: int,
    precision_prior: Mapping[str, float],
    rng: random.Random,
) -> List[cs.Config]:
    """Draw *n_seeds* distinct configurations, weighting precision by the prior.

    Implements Chapter 3's Step 1: "Draw 10 seed configurations from X_feasible
    using the task-informed variant prior P(p | task)."

    Precision is drawn from the (masked, renormalized) prior; the remaining four
    parameters are drawn uniformly, since Chapter 3 supplies prior knowledge for
    precision only. Sampling is **without replacement** so the seeding budget is
    never wasted on duplicates. If a drawn precision has no unused
    configurations left, the prior is re-masked over the variants that do.
    """
    if n_seeds <= 0:
        return []
    if n_seeds > len(space):
        raise ValueError(
            f"cannot draw {n_seeds} distinct seeds from a space of {len(space)}"
        )

    remaining: Dict[str, List[cs.Config]] = {}
    for cfg in space:
        remaining.setdefault(cfg.p, []).append(cfg)

    chosen: List[cs.Config] = []
    while len(chosen) < n_seeds:
        available = [v for v, group in remaining.items() if group]
        weights = mask_and_renormalize(precision_prior, available)
        variant = rng.choices(
            list(weights.keys()), weights=list(weights.values()), k=1
        )[0]
        group = remaining[variant]
        chosen.append(group.pop(rng.randrange(len(group))))
    return chosen


def t1_table_rows() -> List[Dict[str, object]]:
    """Table T1 as rows, for the artifact writers and the dashboard."""
    rows: List[Dict[str, object]] = []
    for task in TASK_TYPES:
        prior = PRECISION_PRIOR[task.key]
        rows.append(
            {
                "task_key": task.key,
                "task_label": task.label,
                "sensitivity": task.sensitivity,
                "has_context": task.has_context,
                "p_f32": round(prior["F32"], 4),
                "p_f16": round(prior["F16"], 4),
                "p_q8_0": round(prior["Q8_0"], 4),
                "p_q4_k_m": round(prior["Q4_K_M"], 4),
                "sum": round(sum(prior.values()), 6),
            }
        )
    return rows
