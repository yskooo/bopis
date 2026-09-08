"""The BOPIS inference-configuration space.

Implements the configuration vector ``x = (t, b, p, g, c)`` of Chapter 3
(Table 3.2) and its encoding into the real-valued feature vector consumed by the
Gaussian Process surrogate.

Deviation from the manuscript (amendment A-1)
---------------------------------------------
Table 3.2 labels ``t`` "input token length / context window size". It is
implemented here as the **maximum generation length** (llama.cpp ``n_predict``),
with ``--ctx-size`` fixed at :data:`FIXED_CTX_SIZE` and held outside the search
space. Rationale: ``--ctx-size`` allocates KV cache but does not drive compute --
attention cost scales with the *actual* sequence length -- so as a context size
``t`` would either have almost no energy effect or would act only by truncating
long prompts, confounding the precision comparison for exactly the three
context-bearing Dolly categories (closed_qa, information_extraction,
summarization). As a generation cap ``t`` is a genuine energy lever, consistent
with Husom et al. (2024), who report that response length dominates inference
energy.

Only the standard library is used.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, Iterable, List, NamedTuple, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Parameter domains (Table 3.2)
# --------------------------------------------------------------------------- #

#: Maximum generated tokens (``n_predict``). See module docstring, amendment A-1.
T_VALUES: Tuple[int, ...] = (128, 256, 512, 1024)

#: Concurrent requests per inference batch (llama-server ``--parallel``).
B_VALUES: Tuple[int, ...] = (1, 2, 4, 8)

#: GGUF precision / quantization variants of the base model.
P_VALUES: Tuple[str, ...] = ("F32", "F16", "Q8_0", "Q4_K_M")

#: GPU layers offloaded. ``ALL_LAYERS`` is the sentinel for "all".
ALL_LAYERS: int = -1
G_VALUES: Tuple[int, ...] = (0, 14, 28, ALL_LAYERS)

#: CPU threads allocated to the inference process.
C_VALUES: Tuple[int, ...] = (2, 4, 8)

#: Context window, fixed outside the search space (amendment A-1).
FIXED_CTX_SIZE: int = 2048

#: Unoptimized default configuration (Table 3.2 "Default Value" column).
#:
#: The manuscript gives the default ``t`` as 2048, which is the *context* default
#: and is not a member of ``T_VALUES``. Under the ``n_predict`` reading the
#: corresponding default is "generate until the context is full", which we cap at
#: the largest explored value so the default remains directly comparable to the
#: searched configurations. ``c`` defaults to "system default", resolved at
#: profile time to the logical core count.
DEFAULT_T: int = 1024
DEFAULT_B: int = 1
DEFAULT_P: str = "F32"
DEFAULT_G: int = ALL_LAYERS

#: Effective bits per weight, used to encode the categorical precision variant on
#: a meaningful ordinal scale. F32/F16 are exact; Q8_0 and Q4_K_M are the
#: measured llama.cpp figures (block quantization carries scale/min metadata, so
#: Q8_0 is 8.5 rather than 8.0 and Q4_K_M is ~4.83 rather than 4.0).
BITS_PER_WEIGHT: Dict[str, float] = {
    "F32": 32.0,
    "F16": 16.0,
    "Q8_0": 8.5,
    "Q4_K_M": 4.83,
}


class Config(NamedTuple):
    """An inference configuration ``x = (t, b, p, g, c)``."""

    t: int  # max generated tokens (n_predict)
    b: int  # concurrent requests
    p: str  # precision / quantization variant
    g: int  # GPU layers offloaded (ALL_LAYERS == all)
    c: int  # CPU threads

    # -- presentation -------------------------------------------------------- #

    @property
    def g_label(self) -> str:
        return "All" if self.g == ALL_LAYERS else str(self.g)

    def key(self) -> str:
        """Stable, filesystem-safe identifier used in artifact paths."""
        return f"t{self.t}_b{self.b}_{self.p}_g{self.g_label}_c{self.c}"

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return f"(t={self.t}, b={self.b}, p={self.p}, g={self.g_label}, c={self.c})"

    def as_row(self) -> Dict[str, object]:
        """Flat mapping for CSV writers (see :mod:`bopis.schemas`)."""
        return {
            "t_max_gen_tokens": self.t,
            "b_batch_size": self.b,
            "p_precision": self.p,
            "g_gpu_layers": self.g_label,
            "c_cpu_threads": self.c,
        }

    # -- helpers ------------------------------------------------------------- #

    def resolved_gpu_layers(self, total_layers: int) -> int:
        """``g`` with the ``All`` sentinel resolved against a real model."""
        return total_layers if self.g == ALL_LAYERS else min(self.g, total_layers)

    def gpu_fraction(self, total_layers: int) -> float:
        """Fraction of model layers placed on the GPU, in ``[0, 1]``."""
        if total_layers <= 0:
            return 0.0
        return self.resolved_gpu_layers(total_layers) / float(total_layers)


def default_config(cpu_threads: int) -> Config:
    """The unoptimized default configuration for a host with *cpu_threads*.

    ``c`` is "system default" in Table 3.2; we resolve it to the largest explored
    thread count that the host permits, which is what an untuned llama.cpp
    invocation effectively uses.
    """
    permitted = [v for v in C_VALUES if v <= max(cpu_threads, min(C_VALUES))]
    c = max(permitted) if permitted else min(C_VALUES)
    return Config(t=DEFAULT_T, b=DEFAULT_B, p=DEFAULT_P, g=DEFAULT_G, c=c)


# --------------------------------------------------------------------------- #
# GP encoding
# --------------------------------------------------------------------------- #

#: Names of the encoded dimensions, in order. Used for ARD length-scale labels.
ENCODED_DIMS: Tuple[str, ...] = ("t", "b", "p_bpw", "g_frac", "c")


def _norm_log2(value: float, lo: float, hi: float) -> float:
    """Min-max normalize ``log2(value)`` into ``[0, 1]``."""
    lv, llo, lhi = math.log2(value), math.log2(lo), math.log2(hi)
    if lhi == llo:
        return 0.0
    return (lv - llo) / (lhi - llo)


def encode(cfg: Config, total_layers: int = 32) -> List[float]:
    """Encode *cfg* as a 5-vector in ``[0, 1]^5`` for the GP.

    All four ordinal parameters are log2-scaled before normalization because
    their effect on cost is multiplicative rather than additive (doubling
    generated tokens roughly doubles decode energy; halving bits per weight
    roughly halves memory traffic).

    The categorical precision variant is encoded as normalized log2 effective
    bits-per-weight rather than one-hot. One-hot would add three dimensions,
    giving an 8-dimensional input for at most 30 observations -- far too sparse
    to fit a length scale to -- and would discard the natural ordering
    F32 > F16 > Q8_0 > Q4_K_M that the surrogate can otherwise exploit.
    """
    return [
        _norm_log2(cfg.t, min(T_VALUES), max(T_VALUES)),
        _norm_log2(cfg.b, min(B_VALUES), max(B_VALUES)),
        _norm_log2(
            BITS_PER_WEIGHT[cfg.p],
            min(BITS_PER_WEIGHT.values()),
            max(BITS_PER_WEIGHT.values()),
        ),
        cfg.gpu_fraction(total_layers),
        _norm_log2(cfg.c, min(C_VALUES), max(C_VALUES)),
    ]


# --------------------------------------------------------------------------- #
# Space construction
# --------------------------------------------------------------------------- #


def full_space() -> List[Config]:
    """The unconstrained Cartesian product ``T x B x P x G x C`` (768 configs)."""
    return [
        Config(t=t, b=b, p=p, g=g, c=c)
        for t, b, p, g, c in itertools.product(
            T_VALUES, B_VALUES, P_VALUES, G_VALUES, C_VALUES
        )
    ]


def build_space(
    t_values: Iterable[int] = T_VALUES,
    b_values: Iterable[int] = B_VALUES,
    p_values: Iterable[str] = P_VALUES,
    g_values: Iterable[int] = G_VALUES,
    c_values: Iterable[int] = C_VALUES,
) -> List[Config]:
    """Cartesian product of the supplied per-parameter domains.

    ``X_feasible = T x B x P(HW) x G(HW) x C(HW)`` -- see
    :func:`bopis.hardware.feasible_space`, which supplies the hardware-permitted
    domains from the Table H1 rules.
    """
    return [
        Config(t=t, b=b, p=p, g=g, c=c)
        for t, b, p, g, c in itertools.product(
            sorted(t_values),
            sorted(b_values),
            [p for p in P_VALUES if p in set(p_values)],  # keep canonical order
            sorted(g_values, key=lambda v: (v == ALL_LAYERS, v)),
            sorted(c_values),
        )
    ]


def index_of(space: Sequence[Config], cfg: Config) -> int:
    """Index of *cfg* within *space*, or ``-1`` if absent."""
    try:
        return list(space).index(cfg)
    except ValueError:
        return -1
