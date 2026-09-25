"""``bopis ui`` -- the chat UI with real instruments behind it.

``bopis.html`` on its own can only talk to llama-server. It cannot read a power
sensor, cannot run a transformer, and -- opened from ``file://`` -- is barred by
the browser from reading a hardware monitor's feed. So on its own the chat can
only show an in-browser guess at energy and ``n/a`` for BERTScore.

This server closes that gap. It serves the UI and adds four endpoints, all on
localhost:

``POST /api/chat``
    Forwards an OpenAI-style chat request to llama-server, bracketed by the same
    :class:`~bopis.monitor.sampler.TelemetrySampler` a study uses. With a live
    LibreHardwareMonitor / Open Hardware Monitor feed the reply carries
    **measured** CPU package energy (RAPL, net of a calibrated idle draw).
    Without one it carries a Mode C estimate built from llama-server's *own*
    CPU time -- still an estimate, and labelled as one, but no longer the
    browser's assumption that every thread was saturated for the whole request.

``POST /api/score``
    BERTScore F1 of a reply against a reference, with the study's scorer
    configuration. Loaded lazily on the first request; this is the only path
    by which the UI process imports ``torch``.

``GET /api/dolly``
    A random Dolly 15k prompt, rendered with the study's own template, together
    with its human reference -- so a chat reply can be scored at all. Free chat
    has no reference, and still reports ``n/a``.

``GET /api/status``
    What is available: llama-server, the energy instrument, the scorer.

Chat measurement and scoring share one lock. A BERTScore forward pass is heavy
CPU work, and letting it overlap an open energy window would charge it to
inference.

Standard library only, except that ``/api/score`` goes through
:func:`bopis.quality.get_scorer`, which imports its dependencies lazily.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from bopis import dataset, metrics
from bopis.monitor import estimator, hwmon, platform_os
from bopis.monitor.nvml import EnergyMethod
from bopis.monitor.sampler import TelemetrySampler, measure_idle_baseline

#: Where the UI is served. 8080 is llama-server's.
DEFAULT_PORT = 8090
DEFAULT_LLAMA_URL = "http://127.0.0.1:8080"

#: Files the server will hand out, relative to the repository root. Anything
#: else is 404: this is a UI server, not a file browser for the whole disk.
STATIC_FILES = {
    "/": ("bopis.html", "text/html; charset=utf-8"),
    "/bopis.html": ("bopis.html", "text/html; charset=utf-8"),
    "/bopis_profile.js": ("bopis_profile.js", "text/javascript; charset=utf-8"),
    "/bopis_rules.js": ("bopis_rules.js", "text/javascript; charset=utf-8"),
    "/bopis_model.js": ("bopis_model.js", "text/javascript; charset=utf-8"),
    "/dashboard_data.js": ("dashboard_data.js", "text/javascript; charset=utf-8"),
    "/image.png": ("image.png", "image/png"),
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_pid(image: str = "llama-server") -> Optional[int]:
    """PID of the first running process named *image*, or None.

    The Mode C fallback attributes CPU time per process, so it needs to know
    which process is inference. ``tasklist`` and ``pgrep`` ship with the OS.
    """
    try:
        if platform_os.IS_WINDOWS:
            out = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {image}.exe", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
            for line in out.splitlines():
                fields = [f.strip('"') for f in line.split('","')]
                if len(fields) > 1 and fields[1].isdigit():
                    return int(fields[1])
            return None
        out = subprocess.run(
            ["pgrep", "-x", image], capture_output=True, text=True, timeout=5
        ).stdout
        first = out.split()
        return int(first[0]) if first else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


class Instruments:
    """The energy instrument, the scorer and the Dolly pool, shared by handlers."""

    def __init__(
        self,
        llama_url: str = DEFAULT_LLAMA_URL,
        hwmon_url: str = hwmon.DEFAULT_URL,
        hwmon_sensor: Optional[str] = None,
        idle_seconds: float = 30.0,
        llama_pid: Optional[int] = None,
        data_dir: str = "data",
        bertscore_model: Optional[str] = None,
        bertscore_layer: Optional[int] = None,
        cpu_tdp_w: float = estimator.DEFAULT_CPU_TDP_W,
        cpu_idle_w: float = 0.0,
        log=print,
        llama_binary: Optional[str] = None,
        model_paths: Optional[Dict[str, str]] = None,
        compare_port: int = 8081,
        ctx_size: int = 2048,
        deploy_port: int = 8083,
    ) -> None:
        #: The configuration the chat is answering with, when the user picked
        #: one in the UI; None means the llama-server at *llama_url* as launched.
        self.deploy_port = deploy_port
        self.deployed = None  # LlamaServerBackend
        self.deployed_config = None  # Config
        self.llama_binary = llama_binary
        self.model_paths = dict(model_paths or {})
        self.compare_port = compare_port
        self.ctx_size = ctx_size
        self.llama_url = llama_url.rstrip("/")
        self.log = log
        self.lock = threading.Lock()
        self._fixed_pid = llama_pid
        self._pid: Optional[int] = llama_pid

        self.source: Optional[hwmon.HwmonPowerSource] = None
        self.source_error: Optional[str] = None
        self.idle: Dict[str, object] = {}
        self.budget = estimator.PowerBudget(
            cpu_tdp_w=cpu_tdp_w,
            gpu_tdp_w=0.0,
            cpu_idle_w=cpu_idle_w,
            gpu_idle_w=0.0,
        )
        self._connect_hwmon(hwmon_url, hwmon_sensor, idle_seconds)

        self.bertscore_model = bertscore_model
        self.bertscore_layer = bertscore_layer
        self._scorer = None
        self.scorer_error: Optional[str] = None

        self.dolly: List[dataset.Prompt] = []
        path = os.path.join(data_dir, dataset.DOLLY_FILENAME)
        if os.path.exists(path):
            self.dolly, _report = dataset.filter_rows(dataset.load_jsonl(path))
            log(f"  Dolly pool                   {len(self.dolly)} prompts from {path}")
        else:
            log(f"  Dolly pool                   none ({path} not found)")

    # ------------------------------------------------------------------ #
    # Energy
    # ------------------------------------------------------------------ #

    def _connect_hwmon(
        self, url: str, sensor_hint: Optional[str], idle_seconds: float
    ) -> None:
        source = hwmon.HwmonPowerSource(url, sensor_hint)
        try:
            sensor = source.discover()
        except hwmon.HwmonUnavailable as exc:
            self.source_error = str(exc)
            self.log("  Energy                       Mode C estimate "
                     "(no RAPL feed; see `bopis profile`)")
            return
        self.log(
            f"  RAPL sensor                  {sensor.label} ({sensor.watts:.2f} W)"
        )
        refresh = source.calibrate_refresh()
        self.log(f"  Monitor refresh              {refresh:.2f} s")
        self.source = source
        if idle_seconds > 0:
            self.log(
                f"  Calibrating idle package power over {idle_seconds:g}s -- "
                "leave the machine alone..."
            )
            self.idle = measure_idle_baseline(
                None, seconds=idle_seconds, cpu_power_source=source
            )
            idle_w = float(self.idle.get("p_cpu_idle_w") or 0.0)
            idle_sd = float(self.idle.get("p_cpu_idle_sd_w") or 0.0)
            self.log(f"  P_pkg,idle                   {idle_w:.2f} W (sd {idle_sd:.2f})")
            if not self.idle.get("quiet", True):
                self.log("  WARNING: the machine was not idle; energy is biased low.")

    @property
    def measuring(self) -> bool:
        return self.source is not None

    def llama_pid(self) -> Optional[int]:
        if self._fixed_pid is not None:
            return self._fixed_pid
        # Re-resolved each request: the server may have been restarted.
        self._pid = find_pid()
        return self._pid

    # ------------------------------------------------------------------ #
    # Answering with a chosen configuration
    # ------------------------------------------------------------------ #

    def deploy(self, key: Optional[str]) -> Dict[str, object]:
        """Make the chat answer with configuration *key* (None: as launched).

        This is what makes the UI's configuration picker real rather than
        decorative: the chosen model, precision, threads, slots and GPU layers
        are launched as their own llama-server on :attr:`deploy_port`, and chat
        requests are routed there. Replies are capped at the configuration's
        ``t``, so the chat behaves as the configuration it claims to be.
        """
        from bopis import config_space as cs
        from bopis.backends.llama_server import LlamaServerBackend

        with self.lock:
            if self.deployed is not None:
                self.deployed.stop()
                self.deployed = self.deployed_config = None
            if not key:
                return {"deployed": None}
            if not self.llama_binary:
                raise RuntimeError(
                    "choosing a configuration needs `bopis ui --llama-binary "
                    "<llama-server.exe>`"
                )
            config = cs.parse_key(key)
            if config is None:
                raise ValueError(f"malformed configuration key {key!r}")
            backend = LlamaServerBackend(
                binary=self.llama_binary,
                model_paths=self.model_paths,
                port=self.deploy_port,
                ctx_size=self.ctx_size,
            )
            if not backend.gguf_path(config):
                raise ValueError(f"no GGUF for {config.m} {config.p} in the models folder")
            started = time.monotonic()
            backend.start(config)
            self.deployed, self.deployed_config = backend, config
            return {
                "deployed": config.key(),
                "load_seconds": time.monotonic() - started,
            }

    def shutdown(self) -> None:
        if self.deployed is not None:
            self.deployed.stop()

    def chat(self, body: bytes) -> Dict[str, object]:
        """Forward one completion to llama-server, measuring it."""
        with self.lock:
            if self.deployed is not None:
                pid = self.deployed.pid
                target = self.deployed.base_url
                request_body = json.loads(body or b"{}")
                request_body["max_tokens"] = min(
                    int(request_body.get("max_tokens") or self.deployed_config.t),
                    self.deployed_config.t,
                )
                body = json.dumps(request_body).encode("utf-8")
            else:
                pid = self.llama_pid()
                target = self.llama_url
            sampler = TelemetrySampler(
                device=None,
                pid=pid,
                estimator_budget=None if self.measuring else self.budget,
                logical_cores=os.cpu_count(),
                cpu_power_source=self.source,
                p_cpu_idle_w=float(self.idle.get("p_cpu_idle_w") or 0.0),
            )
            request = urllib.request.Request(
                target + "/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            sampler.start()
            try:
                with urllib.request.urlopen(request, timeout=600) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            finally:
                window = sampler.stop()
        return {
            "completion": payload,
            "energy": self._energy_summary(window, pid),
            "config": self.deployed_config.key() if self.deployed_config else None,
        }

    def _energy_summary(self, window, pid: Optional[int]) -> Dict[str, object]:
        measured = window.energy_method == EnergyMethod.RAPL_HWMON_POWER_INTEGRATION
        joules = window.energy_j
        return {
            "measured": measured,
            "method": window.energy_method,
            "scope": (
                "cpu_package_rapl" if measured else "estimated_resource_allocation"
            ),
            "energy_j": joules,
            "energy_low_j": window.energy_low_j,
            "energy_high_j": window.energy_high_j,
            "energy_gross_j": window.energy_gross_j,
            "duration_s": window.duration_s,
            "p_cpu_idle_w": window.p_cpu_idle_w,
            "cpu_power_mean_w": window.cpu_power_mean_w,
            "process_cpu_percent": window.process_cpu_percent,
            "cpu_attribution": window.cpu_attribution,
            "llama_pid": pid,
            "basis": window.energy_basis,
            "cost_php": (
                None
                if joules is None
                else joules
                / metrics.JOULES_PER_KWH
                * metrics.DEFAULT_TARIFF_PHP_PER_KWH
            ),
            "tariff_php_per_kwh": metrics.DEFAULT_TARIFF_PHP_PER_KWH,
            "caveat": (
                "Measured CPU package energy (Intel RAPL via hardware monitor), "
                "net of idle. Excludes DRAM, display and the discrete GPU; "
                "valid as inference energy only when llama-server runs with "
                "--n-gpu-layers 0."
                if measured
                else estimator.caveat()
            ),
        }

    # ------------------------------------------------------------------ #
    # Quality
    # ------------------------------------------------------------------ #

    def _get_scorer(self):
        if self._scorer is None:
            from bopis.quality import get_scorer
            from bopis.quality.bertscore import DEFAULT_LAYER, DEFAULT_MODEL

            model = self.bertscore_model or DEFAULT_MODEL
            layer = self.bertscore_layer
            if layer is None and model == DEFAULT_MODEL:
                layer = DEFAULT_LAYER
            self.log(f"  Loading BERTScore ({model}) for the first score...")
            self._scorer = get_scorer("bertscore", model=model, layer=layer)
        return self._scorer

    def score(self, candidate: str, reference: str) -> Dict[str, object]:
        with self.lock:
            scorer = self._get_scorer()
            started = time.monotonic()
            result = scorer.score([candidate], [reference])[0]
            elapsed = time.monotonic() - started
        payload = result.as_dict()
        payload["model"] = scorer.describe().get("model")
        payload["seconds"] = elapsed
        return payload

    # ------------------------------------------------------------------ #
    # Three-way comparison on one prompt
    # ------------------------------------------------------------------ #

    def compare(self, request: Dict[str, object]) -> Dict[str, object]:
        """Run one Dolly prompt under several configurations, as the study does.

        *request* carries ``prompt_id`` and ``configs`` -- ``{condition:
        config_key}``, typically the unoptimized default, the random-search
        pick and ``x*`` from the loaded run. For each, a llama-server is
        launched on :attr:`compare_port` with that configuration (model,
        precision, threads, slots, GPU layers), the prompt is sent through the
        study's own ``/completion`` path with the study's own template and
        ``n_predict = t``, and the request is bracketed by the telemetry
        sampler. Model loading happens before the window opens, so it is not
        charged to the answer. All replies are BERTScored against the Dolly
        reference only after every energy window has closed.
        """
        from bopis import config_space as cs
        from bopis.backends.llama_server import LlamaServerBackend

        if not self.llama_binary:
            raise RuntimeError(
                "comparison needs `bopis ui --llama-binary <llama-server.exe>` so "
                "it can launch each configuration"
            )
        prompt = self._dolly_by_id(str(request.get("prompt_id") or ""))
        if prompt is None:
            raise ValueError(f"unknown Dolly prompt id {request.get('prompt_id')!r}")
        configs = request.get("configs") or {}
        if not isinstance(configs, dict) or not configs:
            raise ValueError("configs must map condition -> configuration key")

        backend = LlamaServerBackend(
            binary=self.llama_binary,
            model_paths=self.model_paths,
            port=self.compare_port,
            ctx_size=self.ctx_size,
        )
        rows: List[Dict[str, object]] = []
        with self.lock:
            for condition, key in configs.items():
                config = cs.parse_key(str(key))
                row: Dict[str, object] = {"condition": condition, "config": key}
                rows.append(row)
                if config is None:
                    row["error"] = "malformed configuration key"
                    continue
                if not backend.gguf_path(config):
                    row["error"] = (
                        f"no GGUF for {config.m} {config.p} in the models folder"
                    )
                    continue
                try:
                    self._run_one(backend, config, prompt, row)
                except Exception as exc:  # noqa: BLE001 - reported per condition
                    row["error"] = f"{type(exc).__name__}: {exc}"

            # Quality only after every window is closed.
            scorable = [r for r in rows if r.get("text")]
            if scorable:
                try:
                    scores = self._get_scorer().score(
                        [r["text"] for r in scorable],
                        [prompt.response] * len(scorable),
                    )
                    for row, score in zip(scorable, scores):
                        row["quality"] = score.as_dict()
                except RuntimeError as exc:
                    for row in scorable:
                        row["quality_error"] = str(exc)
        return {
            "prompt": self._prompt_payload(prompt),
            "rows": rows,
            "protocol": (
                "study protocol: /completion, study template, n_predict = t, no "
                "system prompt; one prompt, so an illustration of the study's "
                "comparison, not a replacement for its 500-prompt validation"
            ),
        }

    def _run_one(self, backend, config, prompt, row: Dict[str, object]) -> None:
        """Load *config*, measure one generation of *prompt*, stop the server."""
        try:
            backend.start(config)  # model load: outside the energy window
            sampler = TelemetrySampler(
                device=None,
                pid=backend.pid,
                estimator_budget=None if self.measuring else self.budget,
                logical_cores=os.cpu_count(),
                cpu_power_source=self.source,
                p_cpu_idle_w=float(self.idle.get("p_cpu_idle_w") or 0.0),
            )
            sampler.start()
            try:
                result = backend.generate(prompt.text, config)
            finally:
                window = sampler.stop()
            pid = backend.pid
        finally:
            backend.stop()
        row.update(
            {
                "text": result.text,
                "n_generated_tokens": result.n_generated_tokens,
                "tokens_per_s": result.decode_tokens_per_s,
                "latency_s": result.wall_s,
                "truncated": result.truncated,
                "energy": self._energy_summary(window, pid),
            }
        )

    # ------------------------------------------------------------------ #

    def _dolly_by_id(self, prompt_id: str) -> Optional[dataset.Prompt]:
        return next((p for p in self.dolly if p.prompt_id == prompt_id), None)

    def _prompt_payload(self, prompt: dataset.Prompt) -> Dict[str, object]:
        return {
            "prompt_id": prompt.prompt_id,
            "task_type": prompt.task_type,
            "instruction": prompt.instruction,
            "context": prompt.context,
            "text": prompt.text,
            "reference": prompt.response,
        }

    def dolly_prompt(
        self, category: Optional[str] = None, prompt_id: Optional[str] = None
    ) -> Optional[Dict]:
        """A random Dolly prompt, or a specific one by id (to replay it)."""
        if prompt_id:
            found = self._dolly_by_id(prompt_id)
            return self._prompt_payload(found) if found else None
        pool = [p for p in self.dolly if not category or p.task_type == category]
        if not pool:
            return None
        prompt = random.choice(pool)
        return {
            "prompt_id": prompt.prompt_id,
            "task_type": prompt.task_type,
            "instruction": prompt.instruction,
            "context": prompt.context,
            "text": prompt.text,
            "reference": prompt.response,
        }

    def status(self) -> Dict[str, object]:
        llama_ok = False
        try:
            with urllib.request.urlopen(self.llama_url + "/health", timeout=2) as r:
                llama_ok = r.status == 200
        except (urllib.error.URLError, OSError):
            llama_ok = False
        import importlib.util

        quality_ready = all(
            importlib.util.find_spec(m) is not None for m in ("torch", "transformers")
        )
        return {
            "llama": {"url": self.llama_url, "ready": llama_ok, "pid": self._pid},
            "energy": {
                "measured": self.measuring,
                "instrument": self.source.describe() if self.source else None,
                "p_cpu_idle_w": self.idle.get("p_cpu_idle_w"),
                "idle_quiet": self.idle.get("quiet"),
                "reason": self.source_error,
            },
            "quality": {
                "available": quality_ready,
                "loaded": self._scorer is not None,
                "model": self.bertscore_model or "roberta-large",
            },
            "dolly": {"n": len(self.dolly)},
            "deployed": self.deployed_config.key() if self.deployed_config else None,
            "compare": {
                "available": bool(self.llama_binary),
                "ggufs": sorted(self.model_paths),
            },
        }


def make_handler(instruments: Instruments, port: int):
    allowed_origins = {
        "null",  # bopis.html opened from file://
        f"http://127.0.0.1:{port}",
        f"http://localhost:{port}",
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "bopis-ui"

        def log_message(self, fmt, *args):  # quieter than the default
            if not self.path.startswith("/api/status"):
                instruments.log("  " + (fmt % args))

        def _cors(self) -> None:
            origin = self.headers.get("Origin")
            if origin in allowed_origins:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _json(self, status: int, payload: object) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length else b""

        def do_OPTIONS(self) -> None:  # noqa: N802 - http.server naming
            self.send_response(HTTPStatus.NO_CONTENT)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if url.path == "/api/status":
                return self._json(200, instruments.status())
            if url.path == "/api/dolly":
                query = parse_qs(url.query)
                prompt = instruments.dolly_prompt(
                    (query.get("category") or [None])[0],
                    (query.get("id") or [None])[0],
                )
                if prompt is None:
                    return self._json(404, {"error": "no Dolly data; run "
                                            "`python -m bopis dataset`"})
                return self._json(200, prompt)
            entry = STATIC_FILES.get(url.path)
            if entry is None:
                return self._json(404, {"error": f"not found: {url.path}"})
            path = os.path.join(REPO_ROOT, entry[0])
            if not os.path.exists(path):
                return self._json(404, {"error": f"{entry[0]} has not been generated"})
            with open(path, "rb") as handle:
                data = handle.read()
            self.send_response(200)
            self.send_header("Content-Type", entry[1])
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            try:
                if url.path == "/api/chat":
                    return self._json(200, instruments.chat(self._body()))
                if url.path == "/api/deploy":
                    request = json.loads(self._body() or b"{}")
                    return self._json(200, instruments.deploy(request.get("config")))
                if url.path == "/api/compare":
                    return self._json(
                        200, instruments.compare(json.loads(self._body() or b"{}"))
                    )
                if url.path == "/api/score":
                    request = json.loads(self._body() or b"{}")
                    candidate = str(request.get("candidate") or "")
                    reference = str(request.get("reference") or "")
                    if not candidate.strip() or not reference.strip():
                        return self._json(400, {"error": "candidate and reference "
                                                "are both required"})
                    return self._json(200, instruments.score(candidate, reference))
            except urllib.error.HTTPError as exc:
                return self._json(
                    502, {"error": f"llama-server returned HTTP {exc.code}"}
                )
            except urllib.error.URLError as exc:
                return self._json(502, {"error": f"llama-server unreachable at "
                                        f"{instruments.llama_url}: {exc.reason}"})
            except RuntimeError as exc:  # get_scorer's missing-dependency message
                return self._json(503, {"error": str(exc)})
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - reported to the UI, not hidden
                return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return self._json(404, {"error": f"not found: {url.path}"})

    return Handler


def serve(
    instruments: Instruments, host: str = "127.0.0.1", port: int = DEFAULT_PORT
) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(instruments, port))
    instruments.log(f"\n  UI ready at http://{host}:{port}/   (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        instruments.shutdown()
        server.server_close()


__all__ = ["DEFAULT_PORT", "Instruments", "find_pid", "make_handler", "serve"]
