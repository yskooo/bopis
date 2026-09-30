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
from typing import Dict, List, Optional, Sequence
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

#: Where ``bopis run`` writes studies, relative to the repository.
RUNS_DIR = os.path.join(REPO_ROOT, "runs")

#: Name of the progress file ``bopis run`` rewrites at every stage.
PROGRESS_NAME = "progress.json"


def find_llama_binary(root: str = REPO_ROOT) -> Optional[str]:
    """The bundled llama-server, so choosing a configuration needs no flag.

    The Vulkan build comes first: the configuration space offloads layers to
    the GPU (g), and a CPU-only build silently ignores ``--n-gpu-layers``, which
    would serve a different configuration from the one the user picked.
    """
    import shutil

    for flavour in ("vulkan", "cpu"):
        for name in ("llama-server.exe", "llama-server"):
            path = os.path.join(root, "tools", flavour, name)
            if os.path.isfile(path):
                return path
    return shutil.which("llama-server")


def _live_row(row: Dict[str, str]) -> Dict[str, object]:
    def number(name: str) -> Optional[float]:
        try:
            value = float(row.get(name) or "nan")
        except ValueError:
            return None
        return value if value == value else None  # NaN is not valid JSON

    return {
        "config": row.get("config"),
        "source": row.get("source"),
        "iteration": int(number("iteration") or 0),
        "energy_j": number("energy_j"),
        "tokens_per_s": number("tokens_per_s"),
        "quality_f1": number("quality_f1"),
        "on_front": row.get("on_pareto_front") == "True",
        "mu": number("gp_mu"),
        "sigma": number("gp_sigma"),
        "expected_improvement": number("expected_improvement"),
    }


def _run_names(runs_dir: str) -> List[str]:
    if not os.path.isdir(runs_dir):
        return []
    return sorted(
        entry for entry in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, entry))
        and os.path.isdir(os.path.join(runs_dir, entry, "calibration"))
    )


def list_runs(runs_dir: str = RUNS_DIR) -> List[Dict[str, object]]:
    """Every run under *runs_dir*, newest first, for the dashboard's run picker."""
    runs = []
    for name in reversed(_run_names(runs_dir)):
        root = os.path.join(runs_dir, name)
        backend = None
        try:
            with open(os.path.join(root, "manifest.json"), encoding="utf-8") as handle:
                backend = (json.load(handle).get("backend") or {}).get("backend")
        except (OSError, ValueError):
            backend = None
        try:
            with open(os.path.join(root, "calibration", "bo_log.csv"),
                      encoding="utf-8") as handle:
                n_evals = max(0, sum(1 for _ in handle) - 1)
        except OSError:
            n_evals = 0
        runs.append({
            "run": name,
            "finished": os.path.exists(os.path.join(root, "dashboard_data.js")),
            "backend": backend,
            "n_evals": n_evals,
        })
    return runs


def run_prompts(name: str, runs_dir: str = RUNS_DIR,
                dolly: Sequence = ()) -> Optional[Dict[str, object]]:
    """The proxy prompts a run's search measured every configuration on.

    ``dataset/sample_*.csv`` records ids and task types, not text, so the text
    is looked up in the loaded Dolly pool; synthetic prompts have none.
    """
    import csv
    import glob

    root = os.path.join(runs_dir, os.path.basename(name))
    files = sorted(glob.glob(os.path.join(root, "dataset", "sample_*.csv")))
    if not files:
        return None
    by_id = {p.prompt_id: p for p in dolly}
    with open(files[0], newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    proxy = []
    for row in rows:
        if row.get("in_proxy_subset") != "True":
            continue
        prompt = by_id.get(row.get("prompt_id"))
        proxy.append({
            "prompt_id": row.get("prompt_id"),
            "task_type": row.get("task_type"),
            "prompt_tokens": row.get("estimated_prompt_tokens"),
            "instruction": prompt.instruction if prompt else None,
        })
    return {"run": os.path.basename(name), "n_sample": len(rows), "proxy": proxy}


def live_run(runs_dir: str = RUNS_DIR) -> Dict[str, object]:
    """State of the newest run under *runs_dir*, for the dashboard's Live mode.

    A study streams one row per evaluation into ``calibration/bo_log.csv`` and
    writes ``dashboard_data.js`` only at the end, so an unfinished run reports
    its evaluations so far and a finished one points at its full payload.
    """
    import csv

    names = _run_names(runs_dir)
    if not names:
        return {"run": None}
    name = names[-1]
    root = os.path.join(runs_dir, name)
    progress: Dict[str, object] = {}
    try:
        with open(os.path.join(root, PROGRESS_NAME), encoding="utf-8") as handle:
            progress = json.load(handle)
    except (OSError, ValueError):
        progress = {}
    evaluations: List[Dict[str, object]] = []
    try:
        with open(os.path.join(root, "calibration", "bo_log.csv"),
                  newline="", encoding="utf-8") as handle:
            evaluations = [_live_row(row) for row in csv.DictReader(handle)]
    except OSError:
        evaluations = []
    finished = os.path.exists(os.path.join(root, "dashboard_data.js"))
    return {
        "run": name,
        "finished": finished,
        "stage": progress.get("stage"),
        "backend": progress.get("backend"),
        "n_total": progress.get("n_total"),
        "updated": progress.get("updated", os.path.getmtime(root)),
        "evaluations": evaluations,
        "data_url": f"/api/live/data?run={name}" if finished else None,
    }


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
        self.data_dir = data_dir
        self.study = None  # subprocess.Popen of a `bopis run` started from the UI
        self.study_args: List[str] = []
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
        self.stop_study()

    # -- studies started from the dashboard ----------------------------------- #

    def study_running(self) -> bool:
        return self.study is not None and self.study.poll() is None

    def study_command(self, request: Dict[str, object]) -> List[str]:
        """The ``bopis run`` command for a dashboard preset.

        ``sim`` runs the analytic simulator on the Dolly pool (or synthetic
        prompts without it): about a minute, simulated energy. ``real`` runs
        the actual study on llama-server with the model files in models\,
        on its own port so the chat's server on 8080 keeps answering.
        """
        import sys

        preset = str(request.get("preset") or "sim")
        iterations = max(4, min(60, int(request.get("iterations") or 30)))
        # At least one prompt per Dolly task type: the proxy set is stratified.
        proxy = max(8, min(100, int(request.get("proxy_size") or 20)))
        command = [sys.executable, "-m", "bopis", "run", "--out", RUNS_DIR,
                   "--iterations", str(iterations), "--proxy-size", str(proxy),
                   "--skip-validation", "--data-dir", self.data_dir]
        if not self.dolly:
            command.append("--synthetic")
        if preset == "sim":
            return command + ["--backend", "sim", "--label", "ui-sim"]
        if preset != "real":
            raise ValueError(f"unknown preset {preset!r}; expected sim or real")
        if not self.llama_binary:
            raise ValueError("a real study needs llama-server; none was found in "
                             "tools\vulkan or tools\cpu")
        if not self.model_paths:
            raise ValueError("a real study needs GGUF files in models\\")
        command += ["--backend", "llama-server", "--model-aware",
                    "--llama-binary", self.llama_binary,
                    "--llama-port", "8084", "--label", "ui-real"]
        for key, path in sorted(self.model_paths.items()):
            command += ["--model", f"{key}={path}"]
        # Measured CPU package energy when the sensor feed is up; otherwise the
        # labelled Mode C estimate (this laptop's MX330 reports no power).
        command += ["--energy-mode", "cpu-rapl" if self.measuring else "resource-estimate"]
        return command

    def start_study(self, request: Dict[str, object]) -> Dict[str, object]:
        if self.study_running():
            raise ValueError("a study is already running; stop it first")
        command = self.study_command(request)
        os.makedirs(RUNS_DIR, exist_ok=True)
        log = open(os.path.join(RUNS_DIR, "ui_study.log"), "w", encoding="utf-8")
        self.study = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log,
                                      stderr=subprocess.STDOUT)
        self.study_args = command
        self.log(f"  Study started (pid {self.study.pid}): {' '.join(command[1:])}")
        return self.study_state()

    def stop_study(self) -> Dict[str, object]:
        if self.study_running():
            self.study.terminate()
            try:
                self.study.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.study.kill()
        return self.study_state()

    def study_state(self) -> Dict[str, object]:
        state: Dict[str, object] = {"running": self.study_running()}
        if self.study is not None:
            state["pid"] = self.study.pid
            state["exit_code"] = self.study.poll()
            state["command"] = " ".join(self.study_args[1:])
            if not state["running"] and state["exit_code"]:
                try:
                    with open(os.path.join(RUNS_DIR, "ui_study.log"),
                              encoding="utf-8", errors="replace") as handle:
                        state["log_tail"] = handle.read()[-1500:]
                except OSError:
                    pass
        return state

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
            "study": self.study_state(),
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
            if url.path == "/api/live":
                live = live_run()
                live["study"] = instruments.study_state()
                return self._json(200, live)
            if url.path == "/api/runs":
                return self._json(200, {"runs": list_runs()})
            if url.path == "/api/run/prompts":
                run = (parse_qs(url.query).get("run") or [""])[0]
                prompts = run_prompts(run, dolly=instruments.dolly) if run else None
                if prompts is None:
                    return self._json(404, {"error": f"no prompt sample for run {run!r}"})
                return self._json(200, prompts)
            if url.path == "/api/live/data":
                run = os.path.basename((parse_qs(url.query).get("run") or [""])[0])
                path = os.path.join(RUNS_DIR, run, "dashboard_data.js")
                if not run or not os.path.isfile(path):
                    return self._json(404, {"error": f"no finished run {run!r}"})
                with open(path, "rb") as handle:
                    data = handle.read()
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "text/javascript; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return None
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
                if url.path == "/api/study":
                    request = json.loads(self._body() or b"{}")
                    return self._json(200, instruments.start_study(request))
                if url.path == "/api/study/stop":
                    return self._json(200, instruments.stop_study())
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
