"""Inference backend adapters.

Chapter 3 and the Scope section describe two *different* invocation mechanisms:
Chapter 3 says each prompt is submitted individually to llama.cpp through
``subprocess``, while the Scope section says all calls are issued through
llama.cpp's OpenAI-compatible **server** endpoint with ``temperature = 0.0``.
These are mutually exclusive as written (amendment A-5). This package resolves
it in favour of the server, behind an adapter interface:

* :mod:`bopis.backends.llama_server` -- primary. Launches ``llama-server`` and
  issues HTTP requests. **Launch-time vs request-time parameters matter**:
  precision variant ``p``, GPU layers ``g``, CPU threads ``c`` and the context
  size are all server *launch flags*, so the server is restarted once per
  configuration; only ``t`` (``n_predict``) and ``b`` (concurrency) vary per
  request. A per-prompt subprocess would pay the model-load cost on every one of
  1500 generations, which is why the server is the right reading.
* :mod:`bopis.backends.simulator` -- deterministic analytic model. Requires no
  weights and no GPU, so the entire pipeline is runnable and testable anywhere.
* :mod:`bopis.backends.ollama` -- convenience adapter for an existing Ollama
  install. Not thesis-faithful; for demonstration only.

Standard library only.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Protocol, Sequence

from bopis.config_space import Config


@dataclasses.dataclass
class GenResult:
    """One generation, with the token accounting the metrics depend on."""

    text: str
    n_prompt_tokens: int
    n_generated_tokens: int
    wall_s: float

    #: Prefill/decode split when the backend reports it. Chapter 3's
    #: ``J/token = E / N_generated`` charges prefill energy to generated tokens,
    #: so the split is recorded where available (amendment A-35).
    prompt_ms: Optional[float] = None
    predicted_ms: Optional[float] = None

    #: True when generation stopped because it hit ``n_predict`` rather than an
    #: end-of-sequence token. A truncated response depresses quality for reasons
    #: unrelated to precision, so it must be visible in the logs.
    truncated: bool = False

    error: Optional[str] = None

    @property
    def tokens_per_s(self) -> float:
        if self.wall_s <= 0:
            return 0.0
        return self.n_generated_tokens / self.wall_s

    @property
    def decode_tokens_per_s(self) -> Optional[float]:
        """Throughput over the decode phase alone, excluding prefill."""
        if not self.predicted_ms:
            return None
        return self.n_generated_tokens / (self.predicted_ms / 1000.0)


class Backend(Protocol):
    """What the evaluation module requires of an inference backend."""

    name: str

    def start(self, config: Config) -> None:
        """Bring the backend up under *config*'s launch-time parameters."""

    def generate(self, prompt: str, config: Config) -> GenResult:
        """Generate one completion. ``start`` must have been called first."""

    def stop(self) -> None:
        """Tear the backend down, releasing weights and VRAM."""

    def describe(self) -> Dict[str, object]:
        """Provenance for the run manifest (versions, model, endpoint)."""

    def count_tokens(self, text: str) -> Optional[int]:
        """Authoritative token count, or ``None`` if the backend cannot tokenize."""


class BackendError(RuntimeError):
    """The backend failed in a way that invalidates the measurement."""


def available_backends() -> Sequence[str]:
    return ("sim", "llama-server", "llama-cli", "ollama")
