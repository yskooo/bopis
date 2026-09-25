"""The BOPIS inference-configuration space.

Implements the configuration vector ``x = (m, t, b, p, g, c)`` -- Chapter 3's
``(t, b, p, g, c)`` extended with the model ``m`` -- and its encoding into the
real-valued feature vector consumed by the Gaussian Process surrogate.

Amendment A-40: the model is a searched parameter, and F32 is dropped
----------------------------------------------------------------------
Chapter 3 fixed one 7B base model and searched its precision. On a low-end
laptop that collapses the choice: once memory rules out the high-precision
variants of a 7B model, "selection" reduces to "quantize until it fits", which
is one inequality, not an optimization. The question such a machine actually
poses is different -- *is a larger model at 4 bits better than a smaller one at
16?* -- and only measurement answers it, because a larger file is neither
necessarily slower nor necessarily worse. So ``m`` ranges over one model family
at several sizes (:data:`MODELS`, the Qwen2.5-Instruct ladder). One family, so
that size is the only thing that differs: tokenizer, training data and chat
template are shared, and an energy or quality difference can be attributed to
size rather than to vendor.

F32 is removed from ``P``. The Qwen2.5 checkpoints are released in BF16, so F32
adds no information over F16 while doubling memory traffic; it is rarely
published as GGUF, and on CPU inference -- the study's measured mode -- it is
never competitive. Removing it also frees budget for the model dimension.

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
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Parameter domains (Table 3.2)
# --------------------------------------------------------------------------- #

#: Maximum generated tokens (``n_predict``). See module docstring, amendment A-1.
T_VALUES: Tuple[int, ...] = (128, 256, 512, 1024)

#: Concurrent requests per inference batch (llama-server ``--parallel``).
B_VALUES: Tuple[int, ...] = (1, 2, 4, 8)

#: GGUF precision / quantization variants. F32 is dropped (amendment A-40).
P_VALUES: Tuple[str, ...] = ("F16", "Q8_0", "Q4_K_M")

#: GPU layers offloaded. ``ALL_LAYERS`` is the sentinel for "all".
ALL_LAYERS: int = -1
G_VALUES: Tuple[int, ...] = (0, 14, 28, ALL_LAYERS)

#: CPU threads allocated to the inference process.
C_VALUES: Tuple[int, ...] = (2, 4, 8)

class ModelVariant(NamedTuple):
    """One model in the searched ladder: enough shape to predict its footprint.

    Values are from each checkpoint's published ``config.json``. ``n_params``
    includes the embedding matrix, which llama.cpp stores alongside the layers.
    """

    key: str
    label: str
    n_params: float
    n_layers: int
    n_kv_heads: int
    head_dim: int
    hf_repo: str


#: The Qwen2.5-Instruct ladder (amendment A-40). All four use grouped-query
#: attention, so their KV caches are small next to their weights.
MODELS: Dict[str, ModelVariant] = {
    v.key: v
    for v in (
        ModelVariant("qwen2.5-0.5b", "Qwen2.5-0.5B-Instruct", 0.494e9, 24, 2, 64,
                     "Qwen/Qwen2.5-0.5B-Instruct-GGUF"),
        ModelVariant("qwen2.5-1.5b", "Qwen2.5-1.5B-Instruct", 1.54e9, 28, 2, 128,
                     "Qwen/Qwen2.5-1.5B-Instruct-GGUF"),
        ModelVariant("qwen2.5-3b", "Qwen2.5-3B-Instruct", 3.09e9, 36, 2, 128,
                     "Qwen/Qwen2.5-3B-Instruct-GGUF"),
        ModelVariant("qwen2.5-7b", "Qwen2.5-7B-Instruct", 7.62e9, 28, 4, 128,
                     "Qwen/Qwen2.5-7B-Instruct-GGUF"),
    )
}

#: Searched models, smallest first.
M_VALUES: Tuple[str, ...] = tuple(MODELS)

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
#:
#: The default model is the one the study's demonstrations already run, at its
#: unquantized release precision (F16, since F32 is no longer searched). It is
#: deliberately neither the smallest nor the largest rung: it is what a user
#: who had optimized nothing would plausibly pick.
DEFAULT_M: str = "qwen2.5-1.5b"
DEFAULT_T: int = 1024
DEFAULT_B: int = 1
DEFAULT_P: str = "F16"
DEFAULT_G: int = ALL_LAYERS

#: Effective bits per weight, used to encode the categorical precision variant on
#: a meaningful ordinal scale. F32/F16 are exact; Q8_0 and Q4_K_M are the
#: measured llama.cpp figures (block quantization carries scale/min metadata, so
#: Q8_0 is 8.5 rather than 8.0 and Q4_K_M is ~4.83 rather than 4.0). F32 stays
#: here so the dropped variant's footprint can still be *reported* (e.g. in
#: ``bopis profile``'s requirements table); it is not searched.
BITS_PER_WEIGHT: Dict[str, float] = {
    "F32": 32.0,
    "F16": 16.0,
    "Q8_0": 8.5,
    "Q4_K_M": 4.83,
}


class Config(NamedTuple):
    """An inference configuration ``x = (m, t, b, p, g, c)``.

    ``m`` is last so that Chapter 3's positional ``Config(t, b, p, g, c)``
    still constructs a configuration of the default model.
    """

    t: int  # max generated tokens (n_predict)
    b: int  # concurrent requests
    p: str  # precision / quantization variant
    g: int  # GPU layers offloaded (ALL_LAYERS == all)
    c: int  # CPU threads
    m: str = DEFAULT_M  # model (amendment A-40)

    # -- presentation -------------------------------------------------------- #

    @property
    def g_label(self) -> str:
        return "All" if self.g == ALL_LAYERS else str(self.g)

    def key(self) -> str:
        """Stable, filesystem-safe identifier used in artifact paths."""
        return f"{self.m}_t{self.t}_b{self.b}_{self.p}_g{self.g_label}_c{self.c}"

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"(m={self.m}, t={self.t}, b={self.b}, p={self.p}, "
            f"g={self.g_label}, c={self.c})"
        )

    @property
    def n_layers(self) -> Optional[int]:
        """Depth of this configuration's model, when it is a known rung."""
        variant = MODELS.get(self.m)
        return variant.n_layers if variant else None

    def as_row(self) -> Dict[str, object]:
        """Flat mapping for CSV writers (see :mod:`bopis.schemas`)."""
        return {
            "m_model": self.m,
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
    return Config(
        t=DEFAULT_T, b=DEFAULT_B, p=DEFAULT_P, g=DEFAULT_G, c=c, m=DEFAULT_M
    )


# --------------------------------------------------------------------------- #
# GP encoding
# --------------------------------------------------------------------------- #

#: Names of the encoded dimensions, in order. Used for ARD length-scale labels.
ENCODED_DIMS: Tuple[str, ...] = ("t", "b", "p_bpw", "g_frac", "c", "m_params")


def _norm_log2(value: float, lo: float, hi: float) -> float:
    """Min-max normalize ``log2(value)`` into ``[0, 1]``."""
    lv, llo, lhi = math.log2(value), math.log2(lo), math.log2(hi)
    if lhi == llo:
        return 0.0
    return (lv - llo) / (lhi - llo)


def _model_params(key: str) -> float:
    variant = MODELS.get(key)
    return variant.n_params if variant else MODELS[DEFAULT_M].n_params


def encode(cfg: Config, total_layers: int = 32) -> List[float]:
    """Encode *cfg* as a 6-vector in ``[0, 1]^6`` for the GP.

    All four ordinal parameters are log2-scaled before normalization because
    their effect on cost is multiplicative rather than additive (doubling
    generated tokens roughly doubles decode energy; halving bits per weight
    roughly halves memory traffic).

    The categorical precision variant is encoded as normalized log2 effective
    bits-per-weight rather than one-hot. One-hot would add three dimensions,
    giving an 8-dimensional input for at most 30 observations -- far too sparse
    to fit a length scale to -- and would discard the natural ordering
    F32 > F16 > Q8_0 > Q4_K_M that the surrogate can otherwise exploit.

    The model is encoded the same way, as normalized log2 parameter count:
    memory traffic per token, and hence decode energy, scales with it, and the
    ordinal encoding lets one observation of a 3B model inform predictions for
    its 1.5B and 7B neighbours. ``g`` is the fraction of *that model's* layers,
    since the ladder's depths differ (24 to 36).
    """
    layers = cfg.n_layers or total_layers
    params = [MODELS[k].n_params for k in M_VALUES]
    precisions = [BITS_PER_WEIGHT[v] for v in P_VALUES]
    return [
        _norm_log2(cfg.t, min(T_VALUES), max(T_VALUES)),
        _norm_log2(cfg.b, min(B_VALUES), max(B_VALUES)),
        _norm_log2(BITS_PER_WEIGHT[cfg.p], min(precisions), max(precisions)),
        cfg.gpu_fraction(layers),
        _norm_log2(cfg.c, min(C_VALUES), max(C_VALUES)),
        _norm_log2(_model_params(cfg.m), min(params), max(params)),
    ]


# --------------------------------------------------------------------------- #
# Space construction
# --------------------------------------------------------------------------- #


def full_space() -> List[Config]:
    """The unconstrained product ``M x T x B x P x G x C`` (2304 configs)."""
    return [
        Config(t=t, b=b, p=p, g=g, c=c, m=m)
        for m, t, b, p, g, c in itertools.product(
            M_VALUES, T_VALUES, B_VALUES, P_VALUES, G_VALUES, C_VALUES
        )
    ]


def build_space(
    t_values: Iterable[int] = T_VALUES,
    b_values: Iterable[int] = B_VALUES,
    p_values: Iterable[str] = P_VALUES,
    g_values: Iterable[int] = G_VALUES,
    c_values: Iterable[int] = C_VALUES,
    m_values: Iterable[str] = M_VALUES,
) -> List[Config]:
    """Cartesian product of the supplied per-parameter domains.

    ``X_feasible = T x B x P(HW) x G(HW) x C(HW)`` -- see
    :func:`bopis.hardware.feasible_space`, which supplies the hardware-permitted
    domains from the Table H1 rules.
    """
    wanted = set(m_values)
    return [
        Config(t=t, b=b, p=p, g=g, c=c, m=m)
        for m, t, b, p, g, c in itertools.product(
            [m for m in M_VALUES if m in wanted],  # keep ladder order
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
