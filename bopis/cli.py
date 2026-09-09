"""Command-line interface: ``python -m bopis <command>``.

Commands
--------
``profile``
    Detect the host, apply the Table H1 rules, and report the resulting feasible
    space and energy-measurement capability. Run this first on any new machine:
    it tells you whether GPU energy can be measured at all before you commit to
    a multi-hour study.
    Use ``--write-js`` to export the same live profile for ``bopis.html``.
``dataset``
    Download Databricks Dolly 15k and build the fixed 500-prompt evaluation set
    and its nested 50-prompt proxy subset.
``run``
    Execute a full study: search, three-way validation, statistics, artifacts,
    dashboard data.
``report``
    Rebuild ``dashboard_data.js`` from an existing run directory.
``serve``
    Launch llama-server using the selected BOPIS configuration from a completed
    run. This is the deployment handoff for the chatbot; it does not run the
    Dolly evaluation again.

Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import re
import shutil
import sys
import textwrap
import time
from typing import List, Optional, Sequence

from bopis import __version__, artifacts, dashboard, dataset, hardware, metrics
from bopis import config_space as cs
from bopis import optimizer, runner, tasks
from bopis.backends.llama_server import LlamaServerBackend
from bopis.config_space import Config
from bopis.measure import SimulatedMeasurer
from bopis.monitor import estimator
from bopis.monitor.nvml import EnergyMethod

GIB = 1024**3


# --------------------------------------------------------------------------- #
# Presentation helpers
# --------------------------------------------------------------------------- #


def _rule(title: str = "", width: int = 74) -> str:
    if not title:
        return "-" * width
    return f"-- {title} " + "-" * max(0, width - len(title) - 4)


def _kv(key: str, value: object, indent: int = 2) -> str:
    return f"{' ' * indent}{key:<28} {value}"


def _wrap(text: str, width: int = 74) -> List[str]:
    return textwrap.wrap(text, width=width) or [""]


# --------------------------------------------------------------------------- #
# profile
# --------------------------------------------------------------------------- #


def cmd_profile(args: argparse.Namespace) -> int:
    profile = hardware.profile_host()
    model = hardware.MISTRAL_7B_INSTRUCT_V03 if args.model_aware else None
    space, rejections = hardware.feasible_space(
        profile,
        model=model,
        ctx_size=args.ctx_size,
        min_gpu_layers=args.min_gpu_layers,
    )
    summary = hardware.summarize_space(space, rejections)

    if args.write_js:
        payload = {
            "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "host_profile": profile.as_dict(),
            "configuration_space": summary,
            "rejections": [
                {"config": r.config.key(), "rule": r.rule, "detail": r.detail}
                for r in rejections
            ],
        }
        target = os.path.abspath(args.write_js)
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("window.BOPIS_PROFILE = ")
            json.dump(payload, handle, indent=2, default=str)
            handle.write(";\n")
        print(f"Wrote live hardware profile to {target}")
        if not args.json:
            return 0

    if args.json:
        print(
            json.dumps(
                {
                    "host_profile": profile.as_dict(),
                    "configuration_space": summary,
                    "rejections": [
                        {
                            "config": r.config.key(),
                            "rule": r.rule,
                            "detail": r.detail,
                        }
                        for r in rejections[: args.max_rejections]
                    ],
                },
                indent=2,
                default=str,
            )
        )
        return 0

    print(_rule("HOST"))
    print(_kv("CPU", profile.cpu_model))
    print(
        _kv(
            "Cores",
            f"{profile.physical_cores} physical / {profile.logical_cores} logical",
        )
    )
    print(
        _kv(
            "RAM",
            f"{profile.ram_gib:.2f} GiB total, "
            f"{profile.ram_available_bytes / GIB:.2f} GiB available",
        )
    )
    print(_kv("Platform", f"{profile.platform_name} ({profile.telemetry_source})"))
    if profile.is_wsl:
        print(
            _kv(
                "WSL",
                "yes -- RAM shown is the WSL ceiling, not host RAM "
                "(governs the HW-B rule)",
            )
        )

    print()
    print(_rule("GPU"))
    if not profile.gpu_available:
        print(_kv("GPU", f"none usable ({profile.gpu_error})"))
    else:
        print(_kv("GPU", profile.gpu_name))
        print(_kv("VRAM", f"{profile.vram_gib:.2f} GiB"))
        print(_kv("Compute capability", profile.compute_capability))
        print(_kv("Driver / CUDA", f"{profile.driver_version} / {profile.cuda_driver_version}"))
        print(_kv("Power query", "supported" if profile.power_supported else "NOT SUPPORTED"))
        print(
            _kv(
                "Energy counter",
                "supported" if profile.energy_counter_supported else "NOT SUPPORTED",
            )
        )
        print(_kv("Energy method", profile.energy_method))

    print()
    print(_rule("ENERGY MEASUREMENT"))
    if profile.energy_method == EnergyMethod.NVML_ENERGY_COUNTER:
        print("  nvmlDeviceGetTotalEnergyConsumption is available.")
        print("  Energy is read from the driver's millijoule counter: exact,")
        print("  with no polling-interval integration error and no P_idle")
        print("  subtraction required.")
    elif profile.energy_method == EnergyMethod.NVML_POWER_INTEGRATION:
        print("  Energy counter unavailable; falling back to Chapter 3's")
        print("  documented method: integrate (P_t - P_idle) over the inference")
        print("  window at a 100 ms sampling interval.")
    else:
        print("  This GPU reports neither power nor energy telemetry.")
        print("  GPU energy CANNOT be measured on this machine. Three options,")
        print("  in descending order of what the result can claim:")
        print()
        print("  1. Run the study on a GPU whose driver exposes")
        print("     nvmlDeviceGetPowerUsage. This is the only path to a")
        print("     measured energy result, and no software change substitutes")
        print("     for the missing instrument.")
        print("  2. Run with `--energy-mode resource-estimate` for a labelled")
        print("     resource-allocation estimate: CPU-time and GPU-utilization")
        print("     scaled by declared power budgets. Valid for comparing")
        print("     configurations on this host, not as absolute energy.")
        print("     See docs/ENERGY_MODES.md.")
        print("  3. Use `--backend sim` to exercise the pipeline with no")
        print("     hardware claim at all.")

    print()
    print(_rule("TABLE H1 -> FEASIBLE SPACE"))
    print(_kv("Rules fired", ", ".join(profile.rules_fired)))
    print(_kv("Precisions permitted", ", ".join(profile.permitted_precisions)))
    print(
        _kv(
            "GPU layers permitted",
            ", ".join(
                "All" if v == cs.ALL_LAYERS else str(v)
                for v in profile.permitted_gpu_layers
            ),
        )
    )
    print(_kv("Batch sizes permitted", ", ".join(map(str, profile.permitted_batch_sizes))))
    print(_kv("CPU threads permitted", ", ".join(map(str, profile.permitted_cpu_threads))))
    print(
        _kv(
            "|X_feasible|",
            f"{summary['n_feasible']} of {summary['n_unconstrained']} unconstrained",
        )
    )
    if rejections:
        print(_kv("Rejected", summary["rejected_by_rule"]))
        print()
        print("  First few rejections:")
        for rejection in rejections[: args.max_rejections]:
            print(f"    [{rejection.rule}] {rejection.config}")
            print(f"        {rejection.detail}")
    if args.model_aware and not space:
        print()
        print("  WARNING: no configuration survives the model-size guards.")
        print("  This model cannot be run on this hardware at any setting.")
    return 0


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #


def cmd_dataset(args: argparse.Namespace) -> int:
    if args.synthetic:
        samples = dataset.synthetic_samples(
            eval_size=args.eval_size, proxy_size=args.proxy_size, seed=args.seed
        )
        print("Using the synthetic offline fallback (no network access).")
    else:
        samples = dataset.load_samples(
            data_dir=args.data_dir,
            eval_size=args.eval_size,
            proxy_size=args.proxy_size,
            seed=args.seed,
        )

    info = samples.as_dict()
    print(_rule("DATASET"))
    print(_kv("Source", info["source"]))
    print(_kv("License", info["license"]))
    if info["source_sha256"]:
        print(_kv("SHA-256", str(info["source_sha256"])[:32] + "..."))
    print(_kv("Seed", info["seed"]))
    print(_kv("Prompt template hash", info["prompt_template_hash"]))
    print()
    report = samples.filter_report
    print(_rule("FILTERING"))
    print(_kv("Rows read", report.n_input))
    print(_kv("Kept", report.n_kept))
    print(_kv("Dropped: no response", report.dropped_missing_response))
    print(_kv("Dropped: no instruction", report.dropped_missing_instruction))
    print(_kv("Dropped: bad category", report.dropped_unknown_category))
    print(
        _kv(
            "Dropped: over-length",
            f"{report.dropped_too_long} "
            f"(> {dataset.MAX_PROMPT_TOKENS} estimated tokens)",
        )
    )
    print()
    print(_rule("STRATIFICATION"))
    evaluation = samples.distribution()
    proxy = samples.proxy_distribution()
    print(f"  {'category':<26}{'eval':>8}{'proxy':>8}{'share':>9}")
    total = sum(evaluation.values()) or 1
    for key in tasks.TASK_KEYS:
        print(
            f"  {key:<26}{evaluation.get(key, 0):>8}{proxy.get(key, 0):>8}"
            f"{100.0 * evaluation.get(key, 0) / total:>8.1f}%"
        )
    print(f"  {'TOTAL':<26}{sum(evaluation.values()):>8}{sum(proxy.values()):>8}")
    print()
    prior = tasks.dataset_prior(samples.task_proportions())
    print(_rule("DATASET-LEVEL PRECISION PRIOR"))
    for variant in cs.P_VALUES:
        print(_kv(variant, f"{prior[variant]:.4f}"))
    return 0


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def _parse_model_paths(values: Optional[Sequence[str]]) -> dict:
    """Parse repeated ``--model VARIANT=/path/to.gguf`` arguments."""
    paths: dict = {}
    for item in values or ():
        if "=" not in item:
            raise SystemExit(
                f"--model expects VARIANT=/path/to.gguf, got {item!r}"
            )
        variant, path = item.split("=", 1)
        variant = variant.strip()
        if variant not in cs.P_VALUES:
            raise SystemExit(
                f"unknown precision variant {variant!r}; "
                f"expected one of {', '.join(cs.P_VALUES)}"
            )
        paths[variant] = path.strip()
    return paths


def _build_measurer(args: argparse.Namespace, profile: hardware.HostProfile):
    """Instantiate the backend and its measurement strategy."""
    if args.backend == "sim":
        from bopis.backends.simulator import SimulatorBackend

        simulator = SimulatorBackend(
            seed=args.seed, noise=not args.no_noise, total_layers=args.total_layers
        )
        return simulator, SimulatedMeasurer(simulator, total_layers=args.total_layers)

    if args.backend == "llama-server":
        from bopis.backends.llama_server import LlamaServerBackend
        from bopis.measure import HardwareMeasurer
        from bopis.monitor import nvml
        from bopis.monitor.sampler import (
            TelemetrySampler,
            measure_idle_baseline,
        )

        model_paths = _parse_model_paths(args.model)
        if not model_paths:
            raise SystemExit(
                "--backend llama-server needs at least one model, e.g.\n"
                "  --model Q4_K_M=/models/mistral-7b-instruct-v0.3.Q4_K_M.gguf\n"
                "Pass one per precision variant you want searched."
            )
        if not args.llama_binary:
            raise SystemExit("--backend llama-server needs --llama-binary")

        backend = LlamaServerBackend(
            binary=args.llama_binary,
            model_paths=model_paths,
            host=args.llama_host,
            port=args.llama_port,
            ctx_size=args.ctx_size,
            seed=args.seed,
            log_path=os.path.join(args.out, "llama-server.log"),
        )

        # Calibrate the idle baseline once, before any inference, per
        # Chapter 3. On a host with a power sensor this is P_idle. On one
        # without, the same 60 s protocol still yields the idle GPU duty cycle
        # the estimator subtracts, so estimate mode is never left guessing it.
        device = None
        estimating = args.energy_mode == "resource-estimate"
        idle: dict = {"power_supported": False, "p_idle_w": 0.0}
        try:
            handle = nvml.Nvml.open()
            device = handle.device(0)
        except nvml.NvmlUnavailable:
            device = None

        if device is not None or estimating:
            print(
                f"Calibrating the idle baseline over {args.idle_seconds:g}s "
                "with no inference running..."
            )
            idle = measure_idle_baseline(device, seconds=args.idle_seconds)
            if idle["power_supported"]:
                print(
                    _kv(
                        "P_idle",
                        f"{idle['p_idle_w']:.2f} W "
                        f"(sd {idle['p_idle_sd_w']:.2f}, "
                        f"n={idle['n_power_samples']})",
                    )
                )
            print(
                _kv(
                    "Idle GPU utilization",
                    f"{idle['gpu_percent_idle']:.1f}% "
                    f"(sd {idle['gpu_percent_idle_sd']:.1f}, "
                    f"peak {idle['gpu_percent_idle_max']:.0f}%)",
                )
            )
            print(
                _kv(
                    "Idle CPU utilization",
                    f"{idle['cpu_percent_idle']:.1f}% "
                    f"(sd {idle['cpu_percent_idle_sd']:.1f}, "
                    f"peak {idle['cpu_percent_idle_max']:.0f}%)",
                )
            )
            if not idle["quiet"]:
                print(
                    "  WARNING: this machine was not idle during calibration. "
                    "The baseline\n"
                    "  is contaminated and every energy figure derived from it "
                    "is biased.\n"
                    "  Close other applications and re-run."
                )

        budget = None
        if estimating:
            budget = estimator.PowerBudget(
                cpu_tdp_w=args.cpu_tdp_w,
                gpu_tdp_w=args.gpu_tdp_w,
                cpu_idle_w=args.cpu_idle_w,
                gpu_idle_w=args.gpu_idle_w,
                uncertainty_frac=args.estimate_uncertainty,
            )

        # ``pid`` is read at each call, not captured once: the server is
        # relaunched whenever a launch-time parameter changes, and a stale PID
        # would attribute the CPU term to a dead process.
        sampler_factory = lambda: TelemetrySampler(  # noqa: E731
            device=device,
            p_idle_w=float(idle.get("p_idle_w") or 0.0),
            pid=backend.pid,
            estimator_budget=budget,
            gpu_percent_idle=float(idle.get("gpu_percent_idle") or 0.0),
            logical_cores=profile.logical_cores,
        )
        return backend, HardwareMeasurer(
            backend, sampler_factory, allow_no_power=args.allow_no_power
        )

    if args.backend == "ollama":
        raise SystemExit(
            "the Ollama adapter is a demonstration convenience and is not "
            "thesis-faithful: Chapter 3 mandates llama.cpp. Use "
            "--backend llama-server for a reportable run, or --backend sim to "
            "exercise the pipeline."
        )

    raise SystemExit(f"unknown backend: {args.backend!r}")


def cmd_run(args: argparse.Namespace) -> int:
    profile = hardware.profile_host()

    if args.backend == "llama-server" and not args.model_aware:
        print(
            "REFUSING TO RUN: real llama-server studies require "
            "--model-aware so the model footprint, VRAM, RAM, and batch "
            "constraints are applied before search.\n"
            "  Re-run with --model-aware, or use --backend sim for the "
            "unconstrained analytic simulator.",
            file=sys.stderr,
        )
        return 2

    estimate_energy = args.energy_mode == "resource-estimate"
    if (
        args.backend != "sim"
        and not profile.can_measure_energy
        and not args.allow_no_power
        and not estimate_energy
    ):
        print(
            "REFUSING TO RUN: this GPU reports neither power nor energy "
            "telemetry, so no energy figure would be measurable.\n"
            "  Pass --energy-mode resource-estimate to proceed with a "
            "labelled resource-allocation\n"
            "  estimate instead of a measurement (see docs/ENERGY_MODES.md), "
            "--allow-no-power to\n"
            "  proceed with energy recorded as null, or --backend sim.",
            file=sys.stderr,
        )
        return 2

    if estimate_energy:
        # Printed before anything else so that nobody can later claim the run
        # was presented as a measurement. The same text goes into the manifest
        # and the dashboard.
        print(_rule("ENERGY MODE: RESOURCE-ALLOCATION ESTIMATE"))
        if args.backend == "sim":
            print(
                "  The simulator computes energy from its analytic model and "
                "never samples\n"
                "  hardware, so --energy-mode has no effect here. These rows "
                "stay labelled\n"
                "  energy_scope: simulated."
            )
        elif profile.can_measure_energy:
            print(
                "  This GPU does report power telemetry, so the measured NVML "
                "path takes\n"
                "  precedence; the estimator will not be reached. Drop "
                "--energy-mode to silence\n"
                "  this notice."
            )
        else:
            for line in _wrap(estimator.caveat(), width=72):
                print(f"  {line}")
            print()
            print("  Formula")
            print(f"    {estimator.FORMULA}")
            print(
                _kv(
                    "CPU budget",
                    f"{args.cpu_idle_w:g}-{args.cpu_tdp_w:g} W "
                    f"(dynamic range {args.cpu_tdp_w - args.cpu_idle_w:g} W)",
                )
            )
            print(
                _kv(
                    "GPU budget",
                    f"{args.gpu_idle_w:g}-{args.gpu_tdp_w:g} W "
                    f"(dynamic range {args.gpu_tdp_w - args.gpu_idle_w:g} W)",
                )
            )
            print(
                _kv(
                    "Declared uncertainty",
                    f"+/-{100.0 * args.estimate_uncertainty:.0f}% on each "
                    "budget",
                )
            )
        print()

    model = hardware.MISTRAL_7B_INSTRUCT_V03 if args.model_aware else None
    space, rejections = hardware.feasible_space(
        profile,
        model=model,
        ctx_size=args.ctx_size,
        min_gpu_layers=args.min_gpu_layers,
    )
    if not space:
        print(
            "REFUSING TO RUN: X_feasible is empty. The Table H1 rules plus the "
            "model-size guards exclude every configuration on this hardware.\n"
            "  Run `python -m bopis profile --model-aware` to see which rule "
            "is responsible.",
            file=sys.stderr,
        )
        return 2

    samples = (
        dataset.synthetic_samples(
            eval_size=args.eval_size, proxy_size=args.proxy_size, seed=args.data_seed
        )
        if args.synthetic
        else dataset.load_samples(
            data_dir=args.data_dir,
            eval_size=args.eval_size,
            proxy_size=args.proxy_size,
            seed=args.data_seed,
        )
    )

    settings = runner.RunSettings(
        backend=args.backend,
        n_total=args.iterations,
        n_seeds=args.seeds,
        seed=args.seed,
        eval_size=args.eval_size,
        proxy_size=args.proxy_size,
        ard=args.ard,
        xi=args.xi,
        min_gpu_layers=args.min_gpu_layers,
        allow_no_power=args.allow_no_power,
        # The simulator never reaches the sampler, so recording estimate mode
        # here would attach the estimator's caveat to analytic-model rows and
        # misdescribe them.
        energy_mode=("auto" if args.backend == "sim" else args.energy_mode),
        estimate_cpu_tdp_w=args.cpu_tdp_w,
        estimate_gpu_tdp_w=args.gpu_tdp_w,
        estimate_cpu_idle_w=args.cpu_idle_w,
        estimate_gpu_idle_w=args.gpu_idle_w,
        estimate_uncertainty=args.estimate_uncertainty,
        tariff_php_per_kwh=args.tariff,
        ctx_size=args.ctx_size,
        total_layers=args.total_layers,
        skip_validation=args.skip_validation,
    )

    backend, measurer = _build_measurer(args, profile)
    run_dir = artifacts.RunDirectory.create(base=args.out, label=args.label)

    print(_rule("BOPIS"))
    print(_kv("Run directory", run_dir.root))
    print(_kv("Backend", args.backend))
    print(_kv("|X_feasible|", f"{len(space)} ({len(rejections)} rejected)"))
    print(
        _kv(
            "Budget",
            f"{settings.n_total} evaluations "
            f"({settings.n_seeds} seeds + {settings.n_total - settings.n_seeds} BO)",
        )
    )
    print(
        _kv(
            "Prompts",
            f"{len(samples.evaluation)} evaluation / {len(samples.proxy)} proxy",
        )
    )
    print()

    study = runner.Study(
        run_dir=run_dir,
        settings=settings,
        profile=profile,
        space=space,
        samples=samples,
        measurer=measurer,
        backend_start=backend.start,
        backend_stop=backend.stop,
        progress=(lambda message: print(message)) if not args.quiet else (lambda _m: None),
    )
    result = study.run()

    # -- provenance --------------------------------------------------------- #
    run_dir.write_manifest(
        artifacts.build_manifest(
            host_profile=profile.as_dict(),
            space_summary=hardware.summarize_space(space, rejections),
            backend=measurer.describe(),
            settings=settings.as_dict(),
            dataset=samples.as_dict(),
        )
    )
    _write_dataset_sample(run_dir, samples)
    dashboard_path = dashboard.write(result)
    published_dashboard = os.path.abspath("dashboard_data.js")
    shutil.copyfile(dashboard_path, published_dashboard)
    print(f"  Published latest dashboard data to {published_dashboard}")

    _print_summary(result)
    return 0


def _write_dataset_sample(run_dir: artifacts.RunDirectory, samples) -> None:
    """Persist the sampled prompt IDs so the sample is reconstructible."""
    import csv

    from bopis import schemas as _schemas

    path = run_dir.path("dataset", "sample_500.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_schemas.DATASET_SAMPLE))
        writer.writeheader()
        writer.writerows(samples.rows())


def _print_summary(result: runner.StudyResult) -> None:
    summary = result.summary
    print()
    print(_rule("RESULT"))

    selection = result.bo_selection
    print(_kv("x* (recommended)", selection.config))
    print(_kv("Selection status", selection.status))
    if not selection.is_clean:
        print(_kv("", selection.notes))

    indicators = summary.get("success_indicators")
    if indicators:
        print()
        print(_kv("EIR (energy improvement)", f"{indicators['eir_percent']:.2f}%"))
        print(_kv("SRR (speed retention)", f"{indicators['srr_percent']:.2f}%"))
        print(_kv("QRR (quality retention)", f"{indicators['qrr_percent']:.2f}%"))
        print(_kv("Verdict", indicators["verdict"].upper()))
        if indicators["failed_criteria"]:
            for failure in indicators["failed_criteria"]:
                print(_kv("  failed", failure))
    else:
        print(_kv("Validation", "skipped -- no EIR/SRR/QRR computed"))

    # Report both reliability bases. Leave-one-out is the one to judge fit on:
    # one-step-ahead predictions are the acquisition function's own
    # high-uncertainty probes and are pessimistic by construction.
    loo = summary.get("surrogate_loo")
    if loo:
        print()
        print(_kv("GP fit (leave-one-out)", ""))
        print(_kv("  MAE", f"{loo['mae']:.3f} J", indent=2))
        print(
            _kv(
                "  NPE",
                f"{loo['npe_percent']:.2f}% "
                f"(threshold {metrics.NPE_RELIABILITY_THRESHOLD:g}%)",
                indent=2,
            )
        )
        print(_kv("  R^2", f"{loo['r_squared']:.4f}", indent=2))
        ucr = loo.get("ucr")
        if ucr is not None and ucr == ucr:  # not NaN
            print(
                _kv("  UCR", f"{ucr:.3f} (target ~{metrics.UCR_TARGET:g})", indent=2)
            )

    reliability = summary.get("surrogate_reliability")
    if reliability:
        print()
        print(_kv("GP forecast (one-step-ahead)", ""))
        print(_kv("  MAE", f"{reliability['mae']:.3f} J", indent=2))
        print(_kv("  NPE", f"{reliability['npe_percent']:.2f}%", indent=2))
        print(_kv("  R^2", f"{reliability['r_squared']:.4f}", indent=2))
        print(
            "    (pessimistic by construction: Expected Improvement targets\n"
            "     the configurations the surrogate is least able to predict)"
        )

    convergence = summary.get("convergence") or {}
    if convergence.get("ser") is not None:
        print()
        print(
            _kv(
                "SER",
                f"{convergence['ser']:.2f} "
                f"(k*_BOPIS={convergence['bopis']['k_star']}, "
                f"k*_RS={convergence['random_search']['k_star']})",
            )
        )
    pareto = summary.get("pareto") or {}
    if pareto:
        print(_kv("Pareto front size", pareto.get("n_front")))
        print(
            _kv(
                "Hypervolume (normalized)",
                f"{pareto.get('hypervolume_normalized', float('nan')):.4f}",
            )
        )

    amortization = summary.get("amortization")
    if amortization:
        print()
        print(_kv("C_bo (calibration)", f"{amortization['calibration_energy_j']:.1f} J"))
        breakeven = amortization["breakeven_prompts"]
        print(
            _kv(
                "N* (break-even)",
                f"{breakeven} prompts" if breakeven else "never -- no energy saved",
            )
        )

    warning = summary.get("scope_warning")
    if warning:
        print()
        print("  WARNING: " + str(warning))

    # Repeated at the end as well as the start: the number a reader copies out
    # of this summary is the one that needs the qualification attached.
    caveat = summary.get("energy_caveat")
    if caveat:
        print()
        print(_rule("ENERGY CAVEAT"))
        for line in _wrap(str(caveat), width=72):
            print(f"  {line}")
        print()
        print("  Every energy figure above carries energy_low_j/energy_high_j "
              "in\n  Table B.3, and the full model is in the run manifest "
              "under\n  settings.energy_estimator.")

    print()
    print(_rule("ARTIFACTS"))
    print(f"  {result.run_dir.root}")
    print("  Open dashboard/index.html and load dashboard_data.js from this run.")


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = (
        artifacts.RunDirectory(args.run)
        if args.run
        else artifacts.RunDirectory.latest(args.out)
    )
    if run_dir is None:
        print(f"no runs found under {args.out!r}", file=sys.stderr)
        return 1
    manifest = run_dir.read_json(artifacts.MANIFEST_NAME)
    metrics_payload = run_dir.read_json(artifacts.METRICS_NAME)
    if manifest is None or metrics_payload is None:
        print(
            f"{run_dir.root} is missing manifest.json or metrics.json",
            file=sys.stderr,
        )
        return 1

    print(_rule("RUN"))
    print(_kv("Directory", run_dir.root))
    print(_kv("Backend", (manifest.get("backend") or {}).get("backend")))
    indicators = metrics_payload.get("success_indicators") or {}
    if indicators:
        print(_kv("Verdict", str(indicators.get("verdict", "?")).upper()))
        print(_kv("EIR", f"{indicators.get('eir_percent', float('nan')):.2f}%"))
    print()
    print(_rule("TABLES"))
    for table_id, filename in sorted(
        __import__("bopis.schemas", fromlist=["TABLE_FILES"]).TABLE_FILES.items()
    ):
        path = run_dir.path(filename)
        exists = os.path.exists(path)
        rows = len(run_dir.read_table(table_id)) if exists else 0
        marker = "ok " if exists else "-- "
        print(f"  {marker}{table_id:<6}{filename:<40}{rows:>6} rows")
    return 0


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #


_CONFIG_KEY = re.compile(
    r"^t(\d+)_b(\d+)_(F32|F16|Q8_0|Q4_K_M)_g(All|\d+)_c(\d+)$"
)


def _selected_config(run: artifacts.RunDirectory) -> Config:
    """Read the BOPIS-selected configuration from ``selection.csv``."""
    path = run.path("selection.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing; run a completed BOPIS study first")

    with open(path, "r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("selected", "").strip().lower() not in {"true", "1", "yes"}:
                continue
            match = _CONFIG_KEY.fullmatch(row.get("config", "").strip())
            if not match:
                raise SystemExit(
                    f"invalid selected configuration key: {row.get('config')!r}"
                )
            t, batch, precision, gpu_layers, threads = match.groups()
            return Config(
                t=int(t),
                b=int(batch),
                p=precision,
                g=cs.ALL_LAYERS if gpu_layers == "All" else int(gpu_layers),
                c=int(threads),
            )
    raise SystemExit(f"{path} has no selected BOPIS configuration")


def cmd_serve(args: argparse.Namespace) -> int:
    """Deploy x* as a llama-server process for a chatbot frontend."""
    run = artifacts.RunDirectory(args.run)
    config = _selected_config(run)
    profile = hardware.profile_host()
    space, rejections = hardware.feasible_space(
        profile,
        model=hardware.MISTRAL_7B_INSTRUCT_V03,
        ctx_size=args.ctx_size,
        min_gpu_layers=args.min_gpu_layers,
    )
    if config not in space:
        rejection = next((r for r in rejections if r.config == config), None)
        detail = (
            rejection.detail if rejection else "configuration is outside X_feasible"
        )
        raise SystemExit(
            f"refusing deployment of {config}: {detail}.\n"
            "Run the study on this machine or use a selected configuration "
            "from hardware with sufficient resources."
        )

    model_paths = _parse_model_paths(args.model)
    backend = LlamaServerBackend(
        binary=args.llama_binary,
        model_paths=model_paths,
        host=args.host,
        port=args.port,
        ctx_size=args.ctx_size,
        seed=args.seed,
        log_path=os.path.join(run.root, "chatbot_server.log"),
    )
    print(_rule("BOPIS CHATBOT"))
    print(_kv("Run", run.root))
    print(_kv("Selected x*", config))
    print(_kv("Endpoint", f"http://{args.host}:{args.port}"))
    print(_kv("Dolly", "evaluation dataset; not used as chatbot memory"))
    print("  Starting llama-server. Press Ctrl+C to stop.")
    try:
        backend.start(config, total_layers=args.total_layers)
        print("  Server is ready. Chatbot clients may call /v1/chat/completions.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping llama-server...")
    finally:
        backend.stop()
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bopis",
        description=(
            "BOPIS -- Bayesian Optimization and Pareto-Based Intelligent "
            "Configuration Selection for Energy-Efficient Local LLM Inference."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"bopis {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # profile
    p_profile = sub.add_parser(
        "profile", help="detect the host and derive the feasible configuration space"
    )
    p_profile.add_argument("--json", action="store_true", help="machine-readable output")
    p_profile.add_argument(
        "--model-aware",
        action="store_true",
        help="also apply the HW-P0/HW-B0 model-size guards for Mistral 7B",
    )
    p_profile.add_argument("--ctx-size", type=int, default=cs.FIXED_CTX_SIZE)
    p_profile.add_argument(
        "--min-gpu-layers",
        type=int,
        default=None,
        help="exclude configs below this GPU-layer floor (energy-scope guard)",
    )
    p_profile.add_argument("--max-rejections", type=int, default=5)
    p_profile.add_argument(
        "--write-js",
        metavar="PATH",
        help="export the live profile for the standalone chatbot HTML",
    )
    p_profile.set_defaults(func=cmd_profile)

    # dataset
    p_data = sub.add_parser("dataset", help="fetch Dolly 15k and build the samples")
    p_data.add_argument("--data-dir", default="data")
    p_data.add_argument("--eval-size", type=int, default=dataset.DEFAULT_EVAL_SIZE)
    p_data.add_argument("--proxy-size", type=int, default=dataset.DEFAULT_PROXY_SIZE)
    p_data.add_argument("--seed", type=int, default=20260101)
    p_data.add_argument(
        "--synthetic",
        action="store_true",
        help="use the offline synthetic fallback instead of downloading",
    )
    p_data.set_defaults(func=cmd_dataset)

    # run
    p_run = sub.add_parser("run", help="execute a full study")
    p_run.add_argument("--backend", default="sim", choices=["sim", "llama-server", "ollama"])
    p_run.add_argument("--out", default="runs", help="base directory for run outputs")
    p_run.add_argument("--label", default=None, help="suffix for the run directory name")
    p_run.add_argument(
        "--iterations",
        type=int,
        default=optimizer.DEFAULT_TOTAL_ITERATIONS,
        help="N, total evaluations per method (default 30 = 10 seeds + 20 BO)",
    )
    p_run.add_argument("--seeds", type=int, default=optimizer.DEFAULT_SEED_COUNT)
    p_run.add_argument("--seed", type=int, default=0, help="RNG seed for the search")
    p_run.add_argument("--data-seed", type=int, default=20260101)
    p_run.add_argument("--eval-size", type=int, default=dataset.DEFAULT_EVAL_SIZE)
    p_run.add_argument("--proxy-size", type=int, default=dataset.DEFAULT_PROXY_SIZE)
    p_run.add_argument("--ctx-size", type=int, default=cs.FIXED_CTX_SIZE)
    p_run.add_argument("--total-layers", type=int, default=32)
    p_run.add_argument("--tariff", type=float, default=metrics.DEFAULT_TARIFF_PHP_PER_KWH)
    p_run.add_argument("--xi", type=float, default=0.0, help="EI exploration margin")
    p_run.add_argument(
        "--ard", action="store_true", help="per-dimension GP length scales"
    )
    p_run.add_argument("--model-aware", action="store_true")
    p_run.add_argument("--min-gpu-layers", type=int, default=None)
    p_run.add_argument("--allow-no-power", action="store_true")
    p_run.add_argument(
        "--energy-mode",
        choices=["auto", "resource-estimate"],
        default="auto",
        help=(
            "auto (default) measures energy from NVML and refuses the run "
            "when it cannot; resource-estimate adds a labelled "
            "resource-allocation estimate below that ladder for hosts with no "
            "power sensor. Estimated runs are not measured runs -- see "
            "docs/ENERGY_MODES.md before reporting one."
        ),
    )
    p_run.add_argument(
        "--cpu-tdp-w",
        type=float,
        default=estimator.DEFAULT_CPU_TDP_W,
        help=(
            "resource-estimate: CPU package power at full utilization, from "
            "the vendor specification (default: %(default)s)"
        ),
    )
    p_run.add_argument(
        "--gpu-tdp-w",
        type=float,
        default=estimator.DEFAULT_GPU_TDP_W,
        help=(
            "resource-estimate: GPU board power at full utilization "
            "(default: %(default)s)"
        ),
    )
    p_run.add_argument(
        "--cpu-idle-w",
        type=float,
        default=0.0,
        help=(
            "resource-estimate: CPU package power at rest. Leaving this at 0 "
            "makes the coefficient the full TDP, which overstates light "
            "loads; set it to narrow the estimate to the true dynamic range"
        ),
    )
    p_run.add_argument(
        "--gpu-idle-w",
        type=float,
        default=0.0,
        help="resource-estimate: GPU board power at rest (default: 0)",
    )
    p_run.add_argument(
        "--estimate-uncertainty",
        type=float,
        default=estimator.DEFAULT_UNCERTAINTY_FRAC,
        help=(
            "resource-estimate: fractional uncertainty on each declared power "
            "budget, propagated in quadrature into energy_low_j/energy_high_j "
            "(default: %(default)s)"
        ),
    )
    p_run.add_argument("--no-noise", action="store_true", help="simulator: noiseless")
    p_run.add_argument(
        "--llama-binary",
        default=None,
        help="path to llama-server (required for --backend llama-server)",
    )
    p_run.add_argument(
        "--model",
        action="append",
        metavar="VARIANT=PATH",
        help="GGUF path per precision variant, e.g. --model "
        "Q4_K_M=/models/mistral.Q4_K_M.gguf (repeatable)",
    )
    p_run.add_argument("--llama-host", default="127.0.0.1")
    p_run.add_argument("--llama-port", type=int, default=8080)
    p_run.add_argument(
        "--idle-seconds",
        type=float,
        default=60.0,
        help="duration of the P_idle calibration (Chapter 3 specifies 60 s)",
    )
    p_run.add_argument("--synthetic", action="store_true", help="synthetic prompts")
    p_run.add_argument(
        "--skip-validation",
        action="store_true",
        help="search only; skip the 500-prompt three-way validation",
    )
    p_run.add_argument("--quiet", action="store_true")
    p_run.set_defaults(func=cmd_run)

    # report
    p_report = sub.add_parser("report", help="summarize an existing run directory")
    p_report.add_argument("--run", default=None, help="path to a run directory")
    p_report.add_argument("--out", default="runs")
    p_report.set_defaults(func=cmd_report)

    # serve
    p_serve = sub.add_parser(
        "serve",
        help="deploy the selected BOPIS configuration as a chatbot server",
    )
    p_serve.add_argument(
        "--run", required=True, help="completed run directory containing selection.csv"
    )
    p_serve.add_argument(
        "--llama-binary", required=True, help="path to llama-server executable"
    )
    p_serve.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="VARIANT=PATH",
        help="GGUF path per precision variant; repeat for available variants",
    )
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--ctx-size", type=int, default=cs.FIXED_CTX_SIZE)
    p_serve.add_argument("--total-layers", type=int, default=32)
    p_serve.add_argument("--seed", type=int, default=0)
    p_serve.add_argument(
        "--min-gpu-layers",
        type=int,
        default=None,
        help="require this GPU-layer floor when deploying x*",
    )
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args) or 0)
