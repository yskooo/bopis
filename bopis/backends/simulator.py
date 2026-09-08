"""Deterministic analytic inference simulator.

This is not a toy stub. It exists for three reasons:

1. **The pipeline must be verifiable without a GPU.** GPU power telemetry is
   simply unavailable on some hardware -- on the development machine
   ``nvmlDeviceGetPowerUsage`` returns ``NVML_ERROR_NOT_SUPPORTED`` -- so the
   only way to exercise the full BO -> Pareto -> statistics path there is against
   a model whose outputs are computed rather than measured.
2. **The true optimum is knowable.** Because energy, speed and quality are
   closed-form functions of the configuration vector, a test can brute-force the
   real Pareto front over ``X_feasible`` and assert that BOPIS finds it, and that
   its sample efficiency beats random search. Against real hardware no such
   ground truth exists.
3. **Sensitivity analysis is cheap.** Seeded repeats characterize the variance of
   SER and of the final EIR without burning GPU hours.

Physical shape of the model
---------------------------
The trade-off structure mirrors what the literature reports for quantized local
inference, so the Pareto front is genuinely three-dimensional rather than
degenerate:

* Lower bit-width reduces memory traffic, so it cuts energy *and* raises
  throughput, but costs output quality -- Q4_K_M is the cheapest and fastest and
  the least accurate.
* GPU offload raises instantaneous power but finishes far sooner, so total energy
  per prompt *falls* as layers move to the GPU.
* Larger batches amortize fixed per-step overhead, with diminishing returns.
* Very small generation caps truncate responses, depressing quality for reasons
  unrelated to precision.

Every quantity is a deterministic function of ``(config, prompt)`` plus seeded
multiplicative noise, so runs are exactly reproducible for a given seed.

Standard library only.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from typing import Dict, Optional

from bopis import config_space as cs
from bopis.backends import GenResult
from bopis.config_space import Config

# --------------------------------------------------------------------------- #
# Calibration constants (chosen to sit in a plausible range for a 7B model on a
# mid-range consumer GPU: ~20-40 tok/s, ~0.5-2.5 J per generated token)
# --------------------------------------------------------------------------- #

_BASE_ENERGY_PER_TOKEN_J = 0.85
_BASE_TOKENS_PER_S = 22.0
_FIXED_OVERHEAD_J = 4.0
_QUALITY_CEILING = 0.862

#: Quality lost to each precision variant, relative to F32.
_QUALITY_PENALTY = {"F32": 0.0, "F16": 0.004, "Q8_0": 0.012, "Q4_K_M": 0.038}

#: Coefficients of variation for the seeded multiplicative noise.
_ENERGY_CV = 0.030
_SPEED_CV = 0.020
_QUALITY_CV = 0.006

_LOREM = (
    "The optimized configuration produced this response under the simulator "
    "backend. Energy, throughput and quality are computed analytically from the "
    "configuration vector rather than measured."
)


def _relative_bpw(precision: str) -> float:
    """Bits per weight relative to F16, the reference point of the model."""
    return cs.BITS_PER_WEIGHT[precision] / cs.BITS_PER_WEIGHT["F16"]


class SimulatorBackend:
    """Analytic stand-in for llama.cpp.

    Args:
        seed: Controls the multiplicative noise. Identical seeds reproduce
            identical runs.
        noise: Set to ``False`` for a noiseless oracle, used when computing the
            ground-truth Pareto front in tests.
        total_layers: Model depth, for resolving the ``All`` GPU-layer sentinel.
        latency: Optional real sleep per generation, to exercise the sampler
            thread. Zero by default so test runs are instant.
    """

    name = "sim"

    def __init__(
        self,
        seed: int = 0,
        noise: bool = True,
        total_layers: int = 32,
        latency: float = 0.0,
    ) -> None:
        self.seed = seed
        self.noise = noise
        self.total_layers = total_layers
        self.latency = latency
        self._config: Optional[Config] = None
        self._started = False
        self._oracle_cache: Dict[tuple, Dict[str, float]] = {}

    # -- lifecycle ----------------------------------------------------------- #

    def start(self, config: Config) -> None:
        self._config = config
        self._started = True

    def stop(self) -> None:
        self._started = False
        self._config = None

    def describe(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "kind": "analytic simulator",
            "seed": self.seed,
            "noise_enabled": self.noise,
            "total_layers": self.total_layers,
            "energy_per_token_base_j": _BASE_ENERGY_PER_TOKEN_J,
            "tokens_per_s_base": _BASE_TOKENS_PER_S,
            "quality_ceiling_f1": _QUALITY_CEILING,
            "note": (
                "Outputs are computed, not measured. No energy figure from this "
                "backend may be reported as an empirical result."
            ),
        }

    def count_tokens(self, text: str) -> int:
        """Deterministic token-count heuristic (~4 characters per token)."""
        return max(1, len(text) // 4)

    # -- noise --------------------------------------------------------------- #

    def _noise_factor(self, config: Config, prompt: str, channel: str, cv: float) -> float:
        if not self.noise or cv <= 0.0:
            return 1.0
        digest = hashlib.sha256(
            f"{self.seed}|{config.key()}|{channel}|{prompt}".encode("utf-8")
        ).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        return max(0.5, 1.0 + rng.gauss(0.0, cv))

    # -- the analytic model -------------------------------------------------- #

    def _generated_tokens(self, config: Config, prompt: str) -> tuple[int, bool]:
        """Response length, capped by ``t``.

        Natural length is a deterministic function of the prompt, so quality and
        energy depend on the prompt in a stable way. Hitting the cap marks the
        response truncated.
        """
        digest = hashlib.sha256(f"len|{prompt}".encode("utf-8")).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        natural = int(rng.gauss(180.0, 90.0))
        natural = max(24, min(natural, 900))
        if natural >= config.t:
            return config.t, True
        return natural, False

    def energy_joules(self, config: Config, prompt: str) -> float:
        """Total system energy for one prompt, in Joules."""
        generated, _ = self._generated_tokens(config, prompt)
        rel_bpw = _relative_bpw(config.p)
        gpu_fraction = config.gpu_fraction(self.total_layers)

        # Narrower weights move less data per token.
        precision_factor = rel_bpw**0.60
        # CPU-resident layers are far less energy-efficient per token overall.
        gpu_factor = 1.0 + 1.20 * (1.0 - gpu_fraction)
        # Batching amortizes fixed per-step cost, with diminishing returns.
        batch_factor = 1.0 / (1.0 + 0.35 * math.log2(config.b))
        # Extra threads only help the CPU-resident share.
        thread_deficit = (max(cs.C_VALUES) / config.c - 1.0) / 3.0
        thread_factor = 1.0 + 0.15 * (1.0 - gpu_fraction) * thread_deficit

        energy = (
            generated
            * _BASE_ENERGY_PER_TOKEN_J
            * precision_factor
            * gpu_factor
            * batch_factor
            * thread_factor
        ) + _FIXED_OVERHEAD_J
        return energy * self._noise_factor(config, prompt, "energy", _ENERGY_CV)

    def tokens_per_second(self, config: Config, prompt: str) -> float:
        rel_bpw = _relative_bpw(config.p)
        gpu_fraction = config.gpu_fraction(self.total_layers)

        precision_speed = (1.0 / rel_bpw) ** 0.45
        gpu_speed = 0.25 + 0.75 * gpu_fraction
        batch_speed = 1.0 + 0.50 * math.log2(config.b)
        thread_speed = 1.0 - 0.10 * (1.0 - gpu_fraction) * (
            (max(cs.C_VALUES) / config.c - 1.0) / 3.0
        )

        speed = (
            _BASE_TOKENS_PER_S
            * precision_speed
            * gpu_speed
            * batch_speed
            * thread_speed
        )
        return max(0.5, speed * self._noise_factor(config, prompt, "speed", _SPEED_CV))

    def quality_f1(self, config: Config, prompt: str) -> float:
        _generated, truncated = self._generated_tokens(config, prompt)
        quality = _QUALITY_CEILING - _QUALITY_PENALTY[config.p]
        if truncated:
            # A response cut off at the cap loses content, and the shorter the
            # cap the more it loses.
            shortfall = min(1.0, 256.0 / max(config.t, 1))
            quality -= 0.045 * shortfall
        quality *= self._noise_factor(config, prompt, "quality", _QUALITY_CV)
        return max(0.0, min(1.0, quality))

    # -- Backend protocol ---------------------------------------------------- #

    def generate(self, prompt: str, config: Config) -> GenResult:
        if not self._started:
            self.start(config)
        if self.latency:
            time.sleep(self.latency)

        generated, truncated = self._generated_tokens(config, prompt)
        speed = self.tokens_per_second(config, prompt)
        wall = generated / speed
        # Prefill is bandwidth-bound and much faster per token than decode.
        prompt_tokens = self.count_tokens(prompt)
        prompt_ms = 1000.0 * prompt_tokens / (speed * 8.0)
        predicted_ms = 1000.0 * wall

        return GenResult(
            text=_LOREM,
            n_prompt_tokens=prompt_tokens,
            n_generated_tokens=generated,
            wall_s=wall + prompt_ms / 1000.0,
            prompt_ms=prompt_ms,
            predicted_ms=predicted_ms,
            truncated=truncated,
        )

    # -- ground truth for tests ---------------------------------------------- #

    def oracle(self, config: Config, prompts: list[str]) -> Dict[str, float]:
        """Mean objectives over *prompts*, as the evaluation module would record.

        Used by tests to brute-force the true Pareto front, which is what makes
        "BOPIS found the optimum" an assertable claim rather than an assumption.

        Memoized: the model is deterministic given ``(seed, config, prompts)``,
        and brute-forcing the ground-truth front re-evaluates the same
        configurations many times.
        """
        cache_key = (config.key(), len(prompts), hash(tuple(prompts)))
        cached = self._oracle_cache.get(cache_key)
        if cached is not None:
            return cached

        energies = [self.energy_joules(config, p) for p in prompts]
        speeds = [self.tokens_per_second(config, p) for p in prompts]
        qualities = [self.quality_f1(config, p) for p in prompts]
        n = max(1, len(prompts))
        result = {
            "energy_j": sum(energies) / n,
            "tokens_per_s": sum(speeds) / n,
            "quality_f1": sum(qualities) / n,
        }
        self._oracle_cache[cache_key] = result
        return result
