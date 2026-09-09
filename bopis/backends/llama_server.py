"""llama.cpp ``llama-server`` backend -- the primary inference adapter.

Resolves the contradiction between Chapter 3's Inference Engine section ("each
prompt is submitted individually to llama.cpp through subprocess") and the Scope
section ("all inference calls are issued through llama.cpp's OpenAI-compatible
server endpoint with temperature set to 0.0") in favour of the server
(amendment A-5). A per-prompt subprocess would reload the model on every one of
~1,500 generations in the validation stage.

Launch-time versus request-time parameters
------------------------------------------
This distinction drives the whole design and is easy to get wrong:

===============================  =========================================
Launch flag (server restart)     Per-request parameter
===============================  =========================================
``p`` precision variant          ``t`` maximum generated tokens
  -- selects the GGUF file         -- ``n_predict``
``g`` GPU layers ``-ngl``        ``b`` concurrency
``c`` CPU threads ``-t``           -- how many requests are in flight
context size ``-c``
``--parallel`` slot count
===============================  =========================================

So the server is started **once per configuration**, not once per prompt, and
``start()`` blocks until the model is loaded and the health endpoint reports
ready. Model-load energy is therefore outside every measurement window, which is
what makes per-prompt energy attributable.

Only the standard library is used: ``subprocess`` to launch the server and
``urllib`` to talk to it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Sequence

from bopis import config_space as cs
from bopis.backends import BackendError, GenResult
from bopis.config_space import Config

#: How long to wait for the server to load a model and become healthy. Large
#: F32/F16 GGUF files on a cold page cache can take minutes.
DEFAULT_STARTUP_TIMEOUT_S = 600.0

#: Per-request HTTP timeout. Generous, because a CPU-only configuration at
#: n_predict=1024 is genuinely slow.
DEFAULT_REQUEST_TIMEOUT_S = 900.0


class LlamaServerBackend:
    """Drives a ``llama-server`` process over its HTTP API.

    Args:
        binary: Path to ``llama-server`` (or ``server``).
        model_paths: Maps each precision variant to its GGUF file, e.g.
            ``{"Q4_K_M": "/models/mistral-7b-instruct-v0.3.Q4_K_M.gguf"}``.
            Variants absent from this mapping cannot be run and are rejected
            loudly rather than silently substituted.
        host, port: Where to bind the server.
        ctx_size: ``--ctx-size``, fixed outside the search space per
            amendment A-1.
        extra_args: Additional flags appended verbatim to every launch.
        temperature: Fixed at 0.0 for greedy decoding.
        seed: Fixed sampling seed, so repeated runs of the *same* configuration
            reproduce. Note that outputs may still differ *across*
            configurations, because quantization, offload split, thread count and
            batch size change floating-point reduction order and kernel
            selection (amendment A-31).
    """

    name = "llama-server"

    def __init__(
        self,
        binary: str,
        model_paths: Dict[str, str],
        host: str = "127.0.0.1",
        port: int = 8080,
        ctx_size: int = cs.FIXED_CTX_SIZE,
        extra_args: Optional[Sequence[str]] = None,
        temperature: float = 0.0,
        seed: int = 0,
        startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        log_path: Optional[str] = None,
    ) -> None:
        self.binary = binary
        self.model_paths = dict(model_paths)
        self.host = host
        self.port = port
        self.ctx_size = ctx_size
        self.extra_args = list(extra_args or ())
        self.temperature = temperature
        self.seed = seed
        self.startup_timeout_s = startup_timeout_s
        self.request_timeout_s = request_timeout_s
        self.log_path = log_path

        self._process: Optional[subprocess.Popen] = None
        self._config: Optional[Config] = None
        self._log_handle = None
        self._server_version: Optional[str] = None

    # ------------------------------------------------------------------ #
    # URLs
    # ------------------------------------------------------------------ #

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def pid(self) -> Optional[int]:
        """PID of the running server, or ``None`` when it is not up.

        The telemetry sampler needs this to charge CPU time to the process that
        actually performs inference rather than to the whole machine. It has to
        be read per measurement, not once: the server is relaunched whenever
        the search moves to a configuration that changes a launch-time
        parameter, and each relaunch is a new PID.
        """
        if self._process is None or self._process.poll() is not None:
            return None
        return self._process.pid

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def launch_args(self, config: Config, total_layers: int = 32) -> List[str]:
        """The full command line for *config*. Exposed so it can be logged."""
        model = self.model_paths.get(config.p)
        if not model:
            raise BackendError(
                f"no GGUF path configured for precision variant {config.p!r}; "
                f"known variants: {sorted(self.model_paths)}"
            )
        if not os.path.exists(model):
            raise BackendError(f"GGUF file not found: {model}")

        return [
            self.binary,
            "--model", model,
            "--host", self.host,
            "--port", str(self.port),
            "--ctx-size", str(self.ctx_size),
            # `g` is a launch-time placement decision, not a request parameter.
            "--n-gpu-layers", str(config.resolved_gpu_layers(total_layers)),
            "--threads", str(config.c),
            # `b` concurrency is realized as server slots.
            "--parallel", str(config.b),
            "--seed", str(self.seed),
            *self.extra_args,
        ]

    def start(self, config: Config, total_layers: int = 32) -> None:
        """Launch the server under *config* and block until it is ready."""
        if self._process is not None:
            self.stop()

        args = self.launch_args(config, total_layers=total_layers)
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            self._log_handle = open(self.log_path, "a", encoding="utf-8")
            self._log_handle.write(f"\n=== {config.key()} ===\n{' '.join(args)}\n")
            self._log_handle.flush()
            stderr = self._log_handle
        else:
            stderr = subprocess.DEVNULL

        try:
            self._process = subprocess.Popen(
                args, stdout=stderr, stderr=subprocess.STDOUT
            )
        except OSError as exc:
            raise BackendError(
                f"could not launch {self.binary!r}: {exc}. Build llama.cpp and "
                "pass --llama-binary, or use --backend sim."
            ) from exc

        self._config = config
        self._await_ready(config)

    def _await_ready(self, config: Config) -> None:
        """Poll ``/health`` until the model is loaded, or fail with context."""
        deadline = time.monotonic() + self.startup_timeout_s
        last_error = "no response"
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise BackendError(
                    f"llama-server exited with code {self._process.returncode} "
                    f"while loading {config.p} at g="
                    f"{config.g_label}. This usually means the model does not "
                    "fit in the requested VRAM; check the HW-P0 guard and the "
                    f"server log{f' at {self.log_path}' if self.log_path else ''}."
                )
            try:
                with urllib.request.urlopen(
                    f"{self.base_url}/health", timeout=5
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if payload.get("status") in ("ok", "no slot available"):
                    self._server_version = self._probe_version()
                    return
                last_error = str(payload)
            except (urllib.error.URLError, OSError, ValueError) as exc:
                last_error = str(exc)
            time.sleep(0.5)

        self.stop()
        raise BackendError(
            f"llama-server did not become ready within "
            f"{self.startup_timeout_s:g}s (last: {last_error})"
        )

    def _probe_version(self) -> Optional[str]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/props", timeout=5) as r:
                payload = json.loads(r.read().decode("utf-8"))
            return str(payload.get("build_info") or payload.get("version") or "")
        except Exception:
            return None

    def stop(self) -> None:
        """Terminate the server and release its VRAM."""
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self._process.kill()
                self._process.wait(timeout=10)
            self._process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        self._config = None

    def __enter__(self) -> "LlamaServerBackend":
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def _post(self, path: str, payload: Dict[str, object]) -> Dict[str, object]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(
            request, timeout=self.request_timeout_s
        ) as response:
            return json.loads(response.read().decode("utf-8"))

    def generate(self, prompt: str, config: Config) -> GenResult:
        """Generate one completion under *config*.

        Uses llama.cpp's native ``/completion`` endpoint rather than
        ``/v1/chat/completions``, because it returns the ``timings`` block --
        ``prompt_ms`` and ``predicted_ms`` -- which is what allows prefill and
        decode energy to be separated (amendment A-35). The OpenAI-compatible
        route omits it.
        """
        if self._process is None:
            raise BackendError("generate() called before start()")

        started = time.perf_counter()
        try:
            payload = self._post(
                "/completion",
                {
                    "prompt": prompt,
                    "n_predict": config.t,  # t is the generation cap (A-1)
                    "temperature": self.temperature,
                    "top_k": 1,  # greedy, with temperature 0
                    "seed": self.seed,
                    "cache_prompt": False,  # no cross-prompt cache reuse
                    "stream": False,
                },
            )
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return GenResult(
                text="",
                n_prompt_tokens=0,
                n_generated_tokens=0,
                wall_s=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
            )
        wall = time.perf_counter() - started

        return self._parse_completion(payload, wall, config)

    @staticmethod
    def _parse_completion(
        payload: Dict[str, object], wall_s: float, config: Config
    ) -> GenResult:
        """Turn a ``/completion`` response into a :class:`GenResult`.

        Separated out and kept static so it can be unit-tested against captured
        response payloads without a running server.
        """
        timings = payload.get("timings") or {}
        if not isinstance(timings, dict):
            timings = {}

        prompt_tokens = int(
            payload.get("tokens_evaluated")
            or timings.get("prompt_n")
            or 0
        )
        generated_tokens = int(
            payload.get("tokens_predicted")
            or timings.get("predicted_n")
            or 0
        )
        stop_reason_truncated = bool(payload.get("stopped_limit", False))

        return GenResult(
            text=str(payload.get("content", "")),
            n_prompt_tokens=prompt_tokens,
            n_generated_tokens=generated_tokens,
            wall_s=wall_s,
            prompt_ms=(
                float(timings["prompt_ms"]) if "prompt_ms" in timings else None
            ),
            predicted_ms=(
                float(timings["predicted_ms"])
                if "predicted_ms" in timings
                else None
            ),
            # Generation hit n_predict rather than an end-of-sequence token, so
            # the response is cut off -- which depresses quality for reasons
            # unrelated to precision and must be visible in the logs.
            truncated=stop_reason_truncated
            or (generated_tokens >= config.t and config.t > 0),
        )

    def count_tokens(self, text: str) -> Optional[int]:
        """Authoritative token count from the server's own tokenizer.

        This is what the over-length dataset filter should be validated against;
        the characters-per-token heuristic used during offline preparation is
        only a pre-filter (amendment A-20).
        """
        if self._process is None:
            return None
        try:
            payload = self._post("/tokenize", {"content": text})
        except Exception:
            return None
        tokens = payload.get("tokens")
        return len(tokens) if isinstance(tokens, list) else None

    # ------------------------------------------------------------------ #

    def describe(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "binary": self.binary,
            "server_version": self._server_version,
            "endpoint": f"{self.base_url}/completion",
            "ctx_size": self.ctx_size,
            "temperature": self.temperature,
            "top_k": 1,
            "seed": self.seed,
            "model_paths": self.model_paths,
            "extra_args": self.extra_args,
            "determinism_note": (
                "Greedy decoding with a fixed seed makes repeated runs of the "
                "same configuration reproducible. Outputs may still differ "
                "across configurations, because quantization, offload split, "
                "thread count and batch size alter floating-point reduction "
                "order and kernel selection."
            ),
        }


# --------------------------------------------------------------------------- #
# Log parsing
# --------------------------------------------------------------------------- #

#: llama.cpp prints both ``prompt eval time`` (prefill) and ``eval time``
#: (decode). A pattern for the latter must exclude the former, or decode timing
#: silently reads the prefill value -- so the decode patterns carry a negative
#: lookbehind on ``"prompt "``.
_TIMING_PATTERNS = {
    "load_time_ms": re.compile(r"load time\s*=\s*([\d.]+)\s*ms"),
    "prompt_ms": re.compile(r"prompt eval time\s*=\s*([\d.]+)\s*ms"),
    "predicted_ms": re.compile(r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms"),
    "total_ms": re.compile(r"total time\s*=\s*([\d.]+)\s*ms"),
}
_TOKEN_PATTERNS = {
    "n_prompt_tokens": re.compile(r"prompt eval time[^\n]*?/\s*(\d+)\s*tokens"),
    "n_generated_tokens": re.compile(
        r"(?<!prompt )eval time[^\n]*?/\s*(\d+)\s*runs"
    ),
}


def parse_server_log(text: str) -> Dict[str, float]:
    """Extract timing and token counts from ``llama-server`` stderr output.

    A fallback for when the HTTP ``timings`` block is unavailable (older builds),
    and a cross-check otherwise. Pure string operations, per Chapter 3.
    """
    found: Dict[str, float] = {}
    for key, pattern in _TIMING_PATTERNS.items():
        match = pattern.search(text)
        if match:
            found[key] = float(match.group(1))
    for key, pattern in _TOKEN_PATTERNS.items():
        match = pattern.search(text)
        if match:
            found[key] = float(match.group(1))
    return found
