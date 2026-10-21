"""Host profiling and construction of the feasible configuration space.

Implements Table H1 (hardware-aware configuration constraint rules) and the
derivation ``X_feasible = T x B x P(HW) x G(HW) x C(HW)``. Every applied rule is
recorded by ID so the manifest can state exactly why a parameter domain was
narrowed -- this module is where the "hardware-agnostic" claim is either kept or
broken.

Two rules are added beyond the manuscript (amendment A-15/A-29)
----------------------------------------------------------------
Table H1 as written is *model-size-agnostic*, which makes it wrong in general:
``HW-P3`` permits F32 on any card with >= 8 GB VRAM, yet a 7B model at F32 is
~29 GB and will not fit a 12 GB card under any offload setting, and ``HW-P1``
permits Q8_0 (~7.7 GB) on a 2 GB GPU. Two guards close the gap:

``HW-P0``
    The GPU-resident share of the weights, plus the KV cache, must fit in
    available VRAM; the whole footprint must fit VRAM + system RAM.
``HW-B0``
    Batch size is limited by KV-cache VRAM, not by system RAM. Each concurrent
    slot allocates its own KV cache.

Both are skipped when no :class:`ModelSpec` is supplied (e.g. simulator runs),
in which case only the literal Table H1 rules apply.

Standard library only.
"""

from __future__ import annotations

import dataclasses
import platform
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from bopis import config_space as cs
from bopis.config_space import ALL_LAYERS, Config
from bopis.monitor import nvml, platform_os

GIB = 1024**3
MIB = 1024**2


# --------------------------------------------------------------------------- #
# Model description (for the HW-P0 / HW-B0 guards)
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """Enough of a GGUF model's shape to predict its memory footprint."""

    name: str
    n_params: float
    n_layers: int
    n_kv_heads: int
    head_dim: int
    kv_bytes_per_element: int = 2  # F16 KV cache, llama.cpp default

    def weight_bytes(self, precision: str) -> float:
        """On-disk / in-memory size of the weights at *precision*."""
        return cs.BITS_PER_WEIGHT[precision] / 8.0 * self.n_params

    def kv_cache_bytes(self, ctx_size: int, batch: int) -> float:
        """KV cache footprint for *batch* concurrent slots of *ctx_size* tokens.

        Two tensors (K and V) per layer per slot.
        """
        return (
            2.0
            * self.n_layers
            * self.n_kv_heads
            * self.head_dim
            * ctx_size
            * batch
            * self.kv_bytes_per_element
        )


#: Table 3.3's original base model. Superseded as the study model by the Qwen2.5
#: ladder (amendment A-40); kept so a single-model footprint check is still
#: expressible and earlier runs remain interpretable.
MISTRAL_7B_INSTRUCT_V03 = ModelSpec(
    name="Mistral-7B-Instruct-v0.3",
    n_params=7.248e9,
    n_layers=32,
    n_kv_heads=8,
    head_dim=128,
)


def spec_for(key: str) -> ModelSpec:
    """The footprint model for one rung of :data:`bopis.config_space.MODELS`."""
    v = cs.MODELS[key]
    return ModelSpec(
        name=v.label,
        n_params=v.n_params,
        n_layers=v.n_layers,
        n_kv_heads=v.n_kv_heads,
        head_dim=v.head_dim,
    )


#: Footprint models for the searched ladder, keyed like ``Config.m``.
MODEL_LADDER: Dict[str, ModelSpec] = {key: spec_for(key) for key in cs.M_VALUES}


# --------------------------------------------------------------------------- #
# Host profile
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class HostProfile:
    """Detected hardware plus the Table H1 domains derived from it."""

    # CPU / RAM
    cpu_model: str
    physical_cores: int
    logical_cores: int
    ram_total_bytes: int
    ram_available_bytes: int
    is_wsl: bool
    telemetry_source: str

    # GPU
    gpu_available: bool
    gpu_name: Optional[str] = None
    vram_total_bytes: int = 0
    vram_free_bytes: int = 0
    compute_capability: Optional[str] = None
    driver_version: Optional[str] = None
    cuda_driver_version: Optional[str] = None
    power_supported: bool = False
    energy_counter_supported: bool = False
    energy_method: str = nvml.EnergyMethod.UNAVAILABLE
    gpu_error: Optional[str] = None

    # Environment
    python_version: str = dataclasses.field(
        default_factory=lambda: sys.version.split()[0]
    )
    platform_name: str = dataclasses.field(default_factory=platform.system)

    # Derived (populated by apply_h1_rules)
    rules_fired: List[str] = dataclasses.field(default_factory=list)
    permitted_precisions: Tuple[str, ...] = ()
    permitted_gpu_layers: Tuple[int, ...] = ()
    permitted_batch_sizes: Tuple[int, ...] = ()
    permitted_cpu_threads: Tuple[int, ...] = ()

    # -- convenience --------------------------------------------------------- #

    @property
    def vram_gib(self) -> float:
        return self.vram_total_bytes / GIB

    @property
    def ram_gib(self) -> float:
        return self.ram_total_bytes / GIB

    @property
    def can_measure_energy(self) -> bool:
        return self.energy_method != nvml.EnergyMethod.UNAVAILABLE

    def as_dict(self) -> Dict[str, object]:
        d = dataclasses.asdict(self)
        d["permitted_gpu_layers"] = [
            "All" if v == ALL_LAYERS else v for v in self.permitted_gpu_layers
        ]
        d["vram_total_gib"] = round(self.vram_gib, 2)
        d["ram_total_gib"] = round(self.ram_gib, 2)
        return d


def profile_host() -> HostProfile:
    """Detect the current machine and derive its Table H1 domains."""
    cpu = platform_os.cpu_info()
    mem = platform_os.memory_info()
    gpu = nvml.probe()

    profile = HostProfile(
        cpu_model=str(cpu["model"]),
        physical_cores=int(cpu["physical_cores"]),
        logical_cores=int(cpu["logical_cores"]),
        ram_total_bytes=mem.total_bytes,
        ram_available_bytes=mem.available_bytes,
        is_wsl=platform_os._detect_wsl(),
        telemetry_source="/proc" if platform_os.IS_LINUX else (
            "kernel32" if platform_os.IS_WINDOWS else "none"
        ),
        gpu_available=bool(gpu.get("available")),
    )

    if profile.gpu_available:
        profile.gpu_name = gpu.get("name")
        profile.vram_total_bytes = int(gpu.get("vram_total_bytes") or 0)
        profile.compute_capability = gpu.get("compute_capability")
        profile.driver_version = gpu.get("driver_version")
        profile.cuda_driver_version = gpu.get("cuda_driver_version")
        profile.power_supported = bool(gpu.get("power_supported"))
        profile.energy_counter_supported = bool(gpu.get("energy_counter_supported"))
        profile.energy_method = str(gpu.get("energy_method"))
        # Free VRAM is queried live; fall back to total when unavailable.
        try:
            with nvml.Nvml.open() as handle:
                free = handle.device(0).memory_free_bytes()
                profile.vram_free_bytes = int(free or profile.vram_total_bytes)
        except Exception:
            profile.vram_free_bytes = profile.vram_total_bytes
    else:
        profile.gpu_error = str(gpu.get("error"))

    apply_h1_rules(profile)
    return profile


# --------------------------------------------------------------------------- #
# Table H1
# --------------------------------------------------------------------------- #


def h1_precision_domain(vram_bytes: int) -> Tuple[Tuple[str, ...], str]:
    """Rules HW-P1 / HW-P2 / HW-P3."""
    vram_gib = vram_bytes / GIB
    if vram_gib < 4:
        return ("Q8_0", "Q4_K_M"), "HW-P1"
    if vram_gib < 8:
        return ("F16", "Q8_0", "Q4_K_M"), "HW-P2"
    return ("F32", "F16", "Q8_0", "Q4_K_M"), "HW-P3"


def h1_gpu_layer_domain(vram_bytes: int) -> Tuple[Tuple[int, ...], str]:
    """Rules HW-G1 / HW-G2 / HW-G3."""
    vram_gib = vram_bytes / GIB
    if vram_gib < 4:
        return (0, 14), "HW-G1"
    if vram_gib < 8:
        return (0, 14, 28), "HW-G2"
    return (0, 14, 28, ALL_LAYERS), "HW-G3"


def h1_batch_domain(ram_bytes: int) -> Tuple[Tuple[int, ...], str]:
    """Rules HW-B1 / HW-B2 / HW-B3.

    *ram_bytes* must be the memory visible to the **inference process**. Under
    WSL2 that is the ``.wslconfig`` ceiling, not the host's physical RAM; on the
    development machine those differ (11.7 GiB vs 15.8 GiB) and straddle the
    16 GiB boundary, flipping HW-B3 to HW-B2 (amendment A-14).
    """
    ram_gib = ram_bytes / GIB
    if ram_gib < 8:
        return (1,), "HW-B1"
    if ram_gib < 16:
        return (1, 2), "HW-B2"
    return (1, 2, 4, 8), "HW-B3"


def h1_thread_domain(physical_cores: int) -> Tuple[Tuple[int, ...], str]:
    """Rules HW-C1 / HW-C2 / HW-C3.

    Keyed on *physical* cores: Table H1 says "CPU Cores", and llama.cpp's own
    guidance is that ``--threads`` should not exceed the physical core count.
    """
    if physical_cores < 4:
        return (2,), "HW-C1"
    if physical_cores < 8:
        return (2, 4), "HW-C2"
    return (2, 4, 8), "HW-C3"


def apply_h1_rules(profile: HostProfile) -> HostProfile:
    """Populate *profile*'s permitted domains and ``rules_fired`` in place."""
    rules: List[str] = []

    if profile.gpu_available:
        precisions, p_rule = h1_precision_domain(profile.vram_total_bytes)
        layers, g_rule = h1_gpu_layer_domain(profile.vram_total_bytes)
    else:
        # No GPU: CPU-only inference. Full precision range remains
        # *representable*, but no layer can be offloaded.
        precisions, p_rule = cs.P_VALUES, "HW-P-NOGPU"
        layers, g_rule = (0,), "HW-G-NOGPU"
    rules += [p_rule, g_rule]

    batches, b_rule = h1_batch_domain(profile.ram_total_bytes)
    threads, c_rule = h1_thread_domain(profile.physical_cores)
    rules += [b_rule, c_rule]

    profile.permitted_precisions = precisions
    profile.permitted_gpu_layers = layers
    profile.permitted_batch_sizes = batches
    profile.permitted_cpu_threads = threads
    profile.rules_fired = rules
    return profile


# --------------------------------------------------------------------------- #
# Model-size guards (HW-P0 / HW-B0)
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Rejection:
    """Why a configuration was excluded from ``X_feasible``."""

    config: Config
    rule: str
    detail: str


def _fits_memory(
    cfg: Config,
    profile: HostProfile,
    model: ModelSpec,
    ctx_size: int,
    weight_bytes: Optional[float] = None,
) -> Optional[Rejection]:
    """Apply HW-P0 and HW-B0 to a single configuration.

    *weight_bytes* is the actual GGUF file size when known, which beats the
    bits-per-weight estimate: K-quants keep some tensors at higher precision,
    so a real Q4_K_M file is somewhat larger than 4.83 bits x parameters.
    """
    total_layers = model.n_layers
    weights = weight_bytes if weight_bytes else model.weight_bytes(cfg.p)
    kv = model.kv_cache_bytes(ctx_size, cfg.b)
    gpu_fraction = cfg.gpu_fraction(total_layers)

    vram_budget = profile.vram_free_bytes or profile.vram_total_bytes
    # llama.cpp keeps the KV cache on the GPU whenever any layer is offloaded.
    gpu_bytes = weights * gpu_fraction + (kv if gpu_fraction > 0 else 0.0)
    host_bytes = weights * (1.0 - gpu_fraction) + (0.0 if gpu_fraction > 0 else kv)

    if gpu_fraction > 0 and gpu_bytes > vram_budget:
        return Rejection(
            cfg,
            "HW-P0",
            f"GPU share {gpu_bytes / GIB:.2f} GiB "
            f"({cfg.p}, g={cfg.g_label}, kv b={cfg.b}) "
            f"exceeds {vram_budget / GIB:.2f} GiB available VRAM",
        )

    if host_bytes > profile.ram_available_bytes:
        return Rejection(
            cfg,
            "HW-P0",
            f"host share {host_bytes / GIB:.2f} GiB exceeds "
            f"{profile.ram_available_bytes / GIB:.2f} GiB available system RAM",
        )

    if gpu_fraction > 0 and kv > vram_budget * 0.5:
        return Rejection(
            cfg,
            "HW-B0",
            f"KV cache {kv / GIB:.2f} GiB for b={cfg.b} exceeds half of "
            f"{vram_budget / GIB:.2f} GiB VRAM",
        )

    return None


def feasible_space(
    profile: HostProfile,
    model: Optional[ModelSpec] = None,
    ctx_size: int = cs.FIXED_CTX_SIZE,
    min_gpu_layers: Optional[int] = None,
    max_gpu_layers: Optional[int] = None,
    models: Optional[Mapping[str, ModelSpec]] = None,
    gguf_bytes: Optional[Mapping[Tuple[str, str], int]] = None,
) -> Tuple[List[Config], List[Rejection]]:
    """Build ``X_feasible`` for *profile*.

    Returns ``(configs, rejections)``. Which models are searched, and how their
    memory is checked:

    * *models* -- the searched rungs, each checked against its own footprint
      (HW-P0 / HW-B0). The normal study mode: ``models=MODEL_LADDER``.
    * *model* -- a single footprint applied to the default model only. The
      pre-A-40 single-model mode.
    * neither -- every rung, literal Table H1 domains, no memory guards (the
      simulator).

    *gguf_bytes* maps ``(model, precision)`` to the size of the GGUF file the
    run will actually load. When given, it replaces the footprint estimate, and
    any pair without a file is rejected ``NO-GGUF`` -- a configuration whose
    weights are not on disk cannot be measured, so offering it to the search
    would waste an evaluation.

    **Precision and offload (amendment A-41).** Table H1's HW-P rules key the
    precision domain on VRAM. That is right for layers placed *on* the GPU and
    wrong for ``g = 0``, where the weights live in system RAM and VRAM plays no
    part: on a 2 GB MX330 it was excluding F16 for CPU-only inference even
    though a 1.5B model at F16 needs under 3 GiB of RAM. So the VRAM precision
    rule now applies only to configurations with ``g > 0``; at ``g = 0`` the
    HW-P0 RAM guard alone decides.

    *min_gpu_layers* excludes configurations with too little GPU offload. This
    exists because GPU-only energy accounting is not a valid objective at
    ``g = 0``: with no layer on the GPU the measured energy approaches idle, and
    Bayesian Optimization would "discover" that CPU-only inference is free
    (amendment A-19). Study runs should set this; simulator runs need not.

    *max_gpu_layers* is the reverse guard, for CPU-package energy accounting
    (RAPL): that instrument sees nothing on the discrete GPU, so offloaded
    layers would run for free as far as the objective could tell. ``0`` pins
    the search to CPU-only inference.
    """
    if models is not None:
        m_values: Sequence[str] = [k for k in cs.M_VALUES if k in models]
    elif model is not None:
        m_values = (cs.DEFAULT_M,)
    else:
        m_values = cs.M_VALUES

    space = cs.build_space(
        t_values=cs.T_VALUES,
        b_values=profile.permitted_batch_sizes,
        # Every precision is built; the VRAM rule is applied per configuration
        # below, because it only binds when layers are offloaded (A-41).
        p_values=cs.P_VALUES,
        g_values=profile.permitted_gpu_layers,
        c_values=profile.permitted_cpu_threads,
        m_values=m_values,
    )
    vram_precisions = set(profile.permitted_precisions)
    p_rule = next(
        (r for r in profile.rules_fired if r.startswith("HW-P")), "HW-P"
    )

    kept: List[Config] = []
    rejections: List[Rejection] = []

    def spec(cfg: Config) -> Optional[ModelSpec]:
        if models is not None:
            return models.get(cfg.m)
        return model

    for cfg in space:
        cfg_spec = spec(cfg)
        layers = cfg_spec.n_layers if cfg_spec else (cfg.n_layers or 32)

        if gguf_bytes is not None and (cfg.m, cfg.p) not in gguf_bytes:
            rejections.append(
                Rejection(cfg, "NO-GGUF", f"no GGUF file supplied for {cfg.m} {cfg.p}")
            )
            continue

        resolved_g = cfg.resolved_gpu_layers(layers)
        if cfg.g != 0 and cfg.p not in vram_precisions:
            rejections.append(
                Rejection(
                    cfg,
                    p_rule,
                    f"{cfg.p} is not permitted for GPU offload on "
                    f"{profile.vram_gib:.2f} GiB VRAM (g={cfg.g_label}); at g=0 "
                    "the weights stay in system RAM and this rule does not apply",
                )
            )
            continue

        # On a shallow model a numeric g can reach every layer, making it the
        # same placement as "All". Evaluating both would spend budget measuring
        # one configuration twice.
        if (
            cfg.g != ALL_LAYERS
            and cfg.g > 0
            and resolved_g >= layers
            and ALL_LAYERS in profile.permitted_gpu_layers
        ):
            rejections.append(
                Rejection(
                    cfg,
                    "DUPLICATE",
                    f"g={cfg.g} covers all {layers} layers of {cfg.m}; "
                    "identical to g=All",
                )
            )
            continue

        if min_gpu_layers is not None:
            resolved = resolved_g
            if resolved < min_gpu_layers:
                rejections.append(
                    Rejection(
                        cfg,
                        "ENERGY-SCOPE",
                        f"g={cfg.g_label} resolves to {resolved} layers, below the "
                        f"--min-gpu-layers floor of {min_gpu_layers}; GPU-only "
                        "energy accounting would not measure the workload",
                    )
                )
                continue

        if max_gpu_layers is not None:
            resolved = resolved_g
            if resolved > max_gpu_layers:
                rejections.append(
                    Rejection(
                        cfg,
                        "ENERGY-SCOPE",
                        f"g={cfg.g_label} resolves to {resolved} layers, above the "
                        f"GPU-layer ceiling of {max_gpu_layers}; CPU-package "
                        "energy accounting would not measure the offloaded work",
                    )
                )
                continue

        if cfg_spec is not None:
            rejection = _fits_memory(
                cfg,
                profile,
                cfg_spec,
                ctx_size,
                weight_bytes=(gguf_bytes or {}).get((cfg.m, cfg.p)),
            )
            if rejection is not None:
                rejections.append(rejection)
                continue

        kept.append(cfg)

    return kept, rejections


def summarize_space(
    space: Sequence[Config], rejections: Sequence[Rejection]
) -> Dict[str, object]:
    """Counts and per-rule rejection tallies for the manifest."""
    by_rule: Dict[str, int] = {}
    for rej in rejections:
        by_rule[rej.rule] = by_rule.get(rej.rule, 0) + 1
    variants = sorted({cfg.p for cfg in space}, key=cs.P_VALUES.index)
    known = [m for m in cs.M_VALUES if any(cfg.m == m for cfg in space)]
    return {
        "n_feasible": len(space),
        "n_rejected": len(rejections),
        "rejected_by_rule": by_rule,
        "n_unconstrained": len(cs.full_space()),
        "precision_variants_present": variants,
        "models_present": known,
        "model_precision_pairs": sorted(
            {f"{cfg.m}:{cfg.p}" for cfg in space},
            key=lambda s: (cs.M_VALUES.index(s.split(":")[0])
                           if s.split(":")[0] in cs.M_VALUES else 99, s),
        ),
    }


#: Share of installed RAM assumed unavailable to inference even on an idle
#: machine, for the requirements table's "could this ever fit" verdict.
OS_RAM_RESERVE_FRACTION = 0.15


def requirements_table(
    profile: HostProfile,
    ctx_size: int = cs.FIXED_CTX_SIZE,
    precisions: Sequence[str] = ("F32",) + cs.P_VALUES,
    models: Optional[Mapping[str, ModelSpec]] = None,
) -> List[Dict[str, object]]:
    """What each (model, precision) needs, and whether this host has it.

    Deterministic arithmetic, the same as HW-P0 -- nothing is trained or
    learned: ``weights = params x bits_per_weight / 8`` plus the KV cache for
    one slot at *ctx_size*. It answers the question a user asks before forcing
    a model onto a machine, and it reports the dropped F32 variant too, so the
    answer to "what would full precision need?" is on the table rather than
    implied.

    Verdicts, for CPU-only inference (``g = 0``, weights in system RAM):

    * ``fits`` -- fits in the RAM free *now*.
    * ``free RAM`` -- fits in installed RAM, but other applications are using
      the difference; closing them is enough.
    * ``needs more RAM`` -- exceeds installed RAM less the OS reserve
      (:data:`OS_RAM_RESERVE_FRACTION`); the machine cannot run it.

    ``full_gpu_fits`` says whether the whole model plus KV cache fits in VRAM,
    i.e. whether ``g = All`` is possible. llama.cpp's compute buffers add a few
    hundred MiB beyond both figures, so a verdict at the margin is optimistic.
    """
    models = models or MODEL_LADDER
    # Installed RAM is never all available to one process: the OS and its
    # services keep a share even on an otherwise idle machine. The reserve is
    # a round, stated figure, not a measurement.
    usable_installed = profile.ram_total_bytes * (1.0 - OS_RAM_RESERVE_FRACTION)
    rows: List[Dict[str, object]] = []
    for key, spec in models.items():
        kv = spec.kv_cache_bytes(ctx_size, 1)
        for precision in precisions:
            weights = spec.weight_bytes(precision)
            need = weights + kv
            if need <= profile.ram_available_bytes:
                verdict = "fits"
            elif need <= usable_installed:
                verdict = "free RAM"
            else:
                verdict = "needs more RAM"
            rows.append(
                {
                    "model": key,
                    "precision": precision,
                    "searched": precision in cs.P_VALUES,
                    "weights_gib": weights / GIB,
                    "kv_cache_gib": kv / GIB,
                    "cpu_only_ram_gib": need / GIB,
                    "full_gpu_vram_gib": need / GIB,
                    "cpu_only_verdict": verdict,
                    "full_gpu_fits": bool(
                        profile.gpu_available and need <= profile.vram_total_bytes
                    ),
                    "shortfall_gib": max(0.0, need - profile.ram_available_bytes)
                    / GIB,
                }
            )
    return rows
