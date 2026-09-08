"""Per-prompt measurement: energy, throughput, resources, and generated text.

Separates *what was generated* from *what it cost*, because the two are produced
by different instruments and, for real runs, at different times: energy and
throughput are recorded live around each request, while BERTScore F1 is computed
in a later offline pass that is allowed to import ``transformers``.

Two measurement strategies share one interface:

:class:`SimulatedMeasurer`
    Reads energy, throughput and quality from the analytic simulator. Every row
    it produces is tagged ``energy_scope="simulated"`` so a computed figure can
    never be mistaken for a measured one.
:class:`HardwareMeasurer`
    Brackets each request with a 100 ms sampler thread and derives energy from
    the NVML energy counter or by integrating power, exactly as Chapter 3
    specifies. Leaves ``quality_f1`` unset for the offline scorer to fill.

Energy scope (amendment A-19)
-----------------------------
``energy_scope`` records *what the energy figure covers*. GPU-only accounting is
not a valid objective when no layer is offloaded: with ``g = 0`` the measured GPU
energy approaches idle regardless of how much work the CPU did, and an optimizer
told to minimize it would "discover" that CPU-only inference is free. Rows
carrying ``gpu_only`` at ``g = 0`` are flagged scope-invalid so they can be
excluded from comparison.

Standard library only.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Protocol

from bopis.backends import Backend, GenResult
from bopis.config_space import Config
from bopis.monitor.nvml import EnergyMethod


class EnergyScope:
    """What an energy measurement actually covers."""

    GPU_ONLY = "gpu_only"  # NVML board power; excludes CPU/DRAM
    GPU_PLUS_RAPL = "gpu_plus_rapl"  # NVML + Intel RAPL package energy
    SIMULATED = "simulated"  # computed by the analytic model
    NONE = "none"  # no energy instrument available


@dataclasses.dataclass
class PromptMeasurement:
    """Everything recorded for one (condition, prompt) pair."""

    prompt_index: int
    prompt_id: str
    task_type: str
    condition: str
    config: Config

    # Generation
    text: str = ""
    n_prompt_tokens: int = 0
    n_generated_tokens: int = 0
    wall_s: float = 0.0
    truncated: bool = False
    decode_tokens_per_s: Optional[float] = None

    # Energy
    energy_j: Optional[float] = None
    prefill_energy_j: Optional[float] = None
    decode_energy_j: Optional[float] = None
    energy_method: str = EnergyMethod.UNAVAILABLE
    energy_scope: str = EnergyScope.NONE
    energy_crosscheck_j: Optional[float] = None
    crosscheck_method: str = "none"
    clamped_samples: int = 0
    n_samples: int = 0

    # Resources
    cpu_percent: Optional[float] = None
    gpu_percent: Optional[float] = None
    memory_mib: Optional[float] = None
    vram_mib: Optional[float] = None
    n_threads: Optional[int] = None

    # Quality -- filled inline by the simulator, offline by the BERTScore pass
    quality_f1: Optional[float] = None
    quality_precision: Optional[float] = None
    quality_recall: Optional[float] = None
    quality_scorer: str = "unscored"
    baseline_rescaled: bool = False

    error: Optional[str] = None

    # -- derived ------------------------------------------------------------- #

    @property
    def tokens_per_s(self) -> float:
        if self.wall_s <= 0:
            return 0.0
        return self.n_generated_tokens / self.wall_s

    @property
    def j_per_token(self) -> Optional[float]:
        if self.energy_j is None or not self.n_generated_tokens:
            return None
        return self.energy_j / self.n_generated_tokens

    @property
    def scope_valid(self) -> bool:
        """False when the energy figure does not cover the work performed."""
        if self.energy_scope == EnergyScope.NONE:
            return False
        if self.energy_scope == EnergyScope.GPU_ONLY:
            return self.config.g != 0
        return True

    # -- table rows ---------------------------------------------------------- #

    def energy_row(self) -> Dict[str, object]:
        """A Table B.3 row."""
        return {
            "prompt_index": self.prompt_index,
            "prompt_id": self.prompt_id,
            "task_type": self.task_type,
            "condition": self.condition,
            "energy_j": self.energy_j,
            "j_per_token": self.j_per_token,
            "n_generated_tokens": self.n_generated_tokens,
            "prefill_energy_j": self.prefill_energy_j,
            "decode_energy_j": self.decode_energy_j,
            "energy_method": self.energy_method,
            "energy_scope": self.energy_scope,
            "energy_crosscheck_j": self.energy_crosscheck_j,
            "crosscheck_method": self.crosscheck_method,
            "clamped_samples": self.clamped_samples,
            "n_samples": self.n_samples,
        }

    def speed_row(self, s_min: Optional[float]) -> Dict[str, object]:
        """A Table B.4 row."""
        return {
            "prompt_index": self.prompt_index,
            "prompt_id": self.prompt_id,
            "task_type": self.task_type,
            "condition": self.condition,
            "tokens_per_s": self.tokens_per_s,
            "decode_tokens_per_s": self.decode_tokens_per_s,
            "wall_s": self.wall_s,
            "n_generated_tokens": self.n_generated_tokens,
            "s_min": s_min,
            "meets_s_min": (
                None if s_min is None else self.tokens_per_s >= s_min
            ),
        }

    def quality_row(self, q_min: Optional[float]) -> Dict[str, object]:
        """A Table B.5 row."""
        return {
            "prompt_index": self.prompt_index,
            "prompt_id": self.prompt_id,
            "task_type": self.task_type,
            "condition": self.condition,
            "quality_f1": self.quality_f1,
            "quality_precision": self.quality_precision,
            "quality_recall": self.quality_recall,
            "q_min_task": q_min,
            "meets_q_min": (
                None
                if (q_min is None or self.quality_f1 is None)
                else self.quality_f1 >= q_min
            ),
            "truncated": self.truncated,
            "scorer": self.quality_scorer,
            "baseline_rescaled": self.baseline_rescaled,
        }

    def resource_row(self) -> Dict[str, object]:
        """A Table B.6 row."""
        return {
            "prompt_index": self.prompt_index,
            "prompt_id": self.prompt_id,
            "task_type": self.task_type,
            "condition": self.condition,
            "cpu_percent": self.cpu_percent,
            "gpu_percent": self.gpu_percent,
            "memory_mib": self.memory_mib,
            "vram_mib": self.vram_mib,
            "n_threads": self.n_threads,
        }


class Measurer(Protocol):
    """Measures one prompt under one configuration."""

    def measure(
        self,
        prompt_index: int,
        prompt_id: str,
        task_type: str,
        condition: str,
        config: Config,
        prompt: str,
    ) -> PromptMeasurement: ...

    def describe(self) -> Dict[str, object]: ...


# --------------------------------------------------------------------------- #
# Simulated
# --------------------------------------------------------------------------- #


class SimulatedMeasurer:
    """Derives every quantity from the analytic simulator.

    Resource utilization is also modelled, so the Table B.6 columns and the
    Friedman tests over CPU/GPU/memory exercise the same code path they will on
    real hardware.
    """

    def __init__(self, simulator, total_layers: int = 32) -> None:
        self.simulator = simulator
        self.total_layers = total_layers

    def measure(
        self,
        prompt_index: int,
        prompt_id: str,
        task_type: str,
        condition: str,
        config: Config,
        prompt: str,
    ) -> PromptMeasurement:
        result: GenResult = self.simulator.generate(prompt, config)
        energy = self.simulator.energy_joules(config, prompt)
        quality = self.simulator.quality_f1(config, prompt)

        gpu_fraction = config.gpu_fraction(self.total_layers)
        # Prefill is a small share of total energy at typical response lengths.
        prefill_share = 0.0
        if result.prompt_ms and result.predicted_ms:
            total_ms = result.prompt_ms + result.predicted_ms
            prefill_share = result.prompt_ms / total_ms if total_ms else 0.0

        return PromptMeasurement(
            prompt_index=prompt_index,
            prompt_id=prompt_id,
            task_type=task_type,
            condition=condition,
            config=config,
            text=result.text,
            n_prompt_tokens=result.n_prompt_tokens,
            n_generated_tokens=result.n_generated_tokens,
            wall_s=result.wall_s,
            truncated=result.truncated,
            decode_tokens_per_s=result.decode_tokens_per_s,
            energy_j=energy,
            prefill_energy_j=energy * prefill_share,
            decode_energy_j=energy * (1.0 - prefill_share),
            energy_method="analytic_model",
            energy_scope=EnergyScope.SIMULATED,
            crosscheck_method="none",
            n_samples=0,
            cpu_percent=round(18.0 + 70.0 * (1.0 - gpu_fraction), 2),
            gpu_percent=round(6.0 + 88.0 * gpu_fraction, 2),
            memory_mib=round(1800.0 + 900.0 * (1.0 - gpu_fraction) * config.b, 1),
            vram_mib=round(420.0 + 6800.0 * gpu_fraction, 1),
            n_threads=config.c,
            quality_f1=quality,
            quality_scorer="analytic_model",
            baseline_rescaled=False,
        )

    def describe(self) -> Dict[str, object]:
        payload = dict(self.simulator.describe())
        payload["measurer"] = "SimulatedMeasurer"
        return payload


# --------------------------------------------------------------------------- #
# Hardware
# --------------------------------------------------------------------------- #


class HardwareMeasurer:
    """Brackets each request with live GPU/CPU sampling.

    Leaves ``quality_f1`` unset: BERTScore requires a transformer forward pass
    and is therefore computed by a separate offline stage, keeping the
    measurement core free of third-party imports.
    """

    def __init__(
        self,
        backend: Backend,
        sampler_factory,
        allow_no_power: bool = False,
    ) -> None:
        self.backend = backend
        self.sampler_factory = sampler_factory
        self.allow_no_power = allow_no_power

    def measure(
        self,
        prompt_index: int,
        prompt_id: str,
        task_type: str,
        condition: str,
        config: Config,
        prompt: str,
    ) -> PromptMeasurement:
        sampler = self.sampler_factory()
        measurement = PromptMeasurement(
            prompt_index=prompt_index,
            prompt_id=prompt_id,
            task_type=task_type,
            condition=condition,
            config=config,
        )

        sampler.start()
        started = time.perf_counter()
        try:
            result = self.backend.generate(prompt, config)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            sampler.stop()
            measurement.error = f"{type(exc).__name__}: {exc}"
            return measurement
        wall = time.perf_counter() - started
        window = sampler.stop()

        measurement.text = result.text
        measurement.n_prompt_tokens = result.n_prompt_tokens
        measurement.n_generated_tokens = result.n_generated_tokens
        measurement.wall_s = result.wall_s or wall
        measurement.truncated = result.truncated
        measurement.decode_tokens_per_s = result.decode_tokens_per_s
        measurement.error = result.error

        measurement.energy_j = window.energy_j
        measurement.energy_method = window.energy_method
        measurement.energy_scope = (
            EnergyScope.GPU_ONLY
            if window.energy_j is not None
            else EnergyScope.NONE
        )
        measurement.energy_crosscheck_j = window.energy_crosscheck_j
        measurement.crosscheck_method = window.crosscheck_method
        measurement.clamped_samples = window.clamped_samples
        measurement.n_samples = window.n_samples
        measurement.cpu_percent = window.cpu_percent
        measurement.gpu_percent = window.gpu_percent_mean
        measurement.memory_mib = window.memory_mib_mean
        measurement.vram_mib = window.vram_mib_mean
        measurement.n_threads = window.n_threads

        # Split energy by the prefill/decode time boundary that llama.cpp
        # reports, since J/token would otherwise charge prefill to generated
        # tokens (amendment A-35).
        if (
            measurement.energy_j is not None
            and result.prompt_ms
            and result.predicted_ms
        ):
            total_ms = result.prompt_ms + result.predicted_ms
            if total_ms > 0:
                share = result.prompt_ms / total_ms
                measurement.prefill_energy_j = measurement.energy_j * share
                measurement.decode_energy_j = measurement.energy_j * (1.0 - share)

        return measurement

    def describe(self) -> Dict[str, object]:
        payload = dict(self.backend.describe())
        payload["measurer"] = "HardwareMeasurer"
        payload["allow_no_power"] = self.allow_no_power
        return payload


def aggregate(measurements: List[PromptMeasurement], column: str) -> List[float]:
    """Extract a numeric column, dropping missing values."""
    values: List[float] = []
    for m in measurements:
        value = getattr(m, column, None)
        if value is None and column == "j_per_token":
            continue
        if value is None:
            continue
        values.append(float(value))
    return values
