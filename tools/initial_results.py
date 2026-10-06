"""Generate a defense-ready initial-results Markdown report from a BOPIS run.

The report intentionally preserves the run's provenance and labels simulator
outputs as computed rather than measured. It uses only the Python standard
library so it can run in the same environment as the BOPIS core.

Usage:
    python tools\initial_results.py
    python tools\initial_results.py runs\20261005T085002Z_ui-sim
    python tools\initial_results.py RUN_DIR --output docs\INITIAL_RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Optional


DEFAULT_RUN = os.path.join("runs", "20261005T085002Z_ui-sim")
DEFAULT_OUTPUT = os.path.join("docs", "INITIAL_RESULTS.md")


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def _config(config: Dict[str, Any]) -> str:
    return (
        f"`{config.get('m_model', 'n/a')}` / "
        f"`{config.get('p_precision', 'n/a')}` / "
        f"`t={config.get('t_max_gen_tokens', 'n/a')}` / "
        f"`b={config.get('b_batch_size', 'n/a')}` / "
        f"`g={config.get('g_gpu_layers', 'n/a')}` / "
        f"`c={config.get('c_cpu_threads', 'n/a')}`"
    )


def render(manifest: Dict[str, Any], metrics: Dict[str, Any], run_dir: str) -> str:
    backend = manifest.get("backend", {})
    host = manifest.get("host_profile", {})
    settings = manifest.get("settings", {})
    selection = metrics.get("selection", {})
    random = metrics.get("random_search_selection", {})
    reference = metrics.get("search_reference", {})
    reliability = metrics.get("surrogate_reliability", {})
    loo = metrics.get("surrogate_loo", {})
    convergence = metrics.get("convergence", {})
    pareto = metrics.get("pareto", {})
    dataset = manifest.get("dataset", {})

    return f"""# BOPIS Initial Results and Instruments

**Generated from:** `{run_dir}`  
**Run timestamp:** `{manifest.get("generated_utc", "n/a")}`  
**Backend:** `{backend.get("backend", "n/a")}` ({backend.get("kind", "n/a")})

## Executive result

This is an initial, simulator-backed result for defense preparation. It
validates the optimization pipeline and its reporting instruments; it is **not
an empirical hardware-energy result**. The run's own manifest states that the
analytic simulator produces computed outputs.

| Indicator | BOPIS | Criterion | Status |
|---|---:|---:|---|
| Energy Improvement Ratio (EIR) | {_fmt(selection.get("eir_percent"))}% | > 0% | {"PASS" if selection.get("eir_percent", 0) > 0 else "FAIL"} |
| Speed Retention Ratio (SRR) | {_fmt(selection.get("srr_percent"))}% | >= 95% | {"PASS" if selection.get("srr_percent", 0) >= 95 else "FAIL"} |
| Quality Retention Ratio (QRR) | {_fmt(selection.get("qrr_percent"))}% | >= 98% | {"PASS" if selection.get("qrr_percent", 0) >= 98 else "FAIL"} |
| Selection verdict | `{selection.get("status", "n/a")}` | clean selection | {"PASS" if selection.get("is_clean") else "CHECK"} |

**Selected BOPIS configuration:** {_config(selection.get("config", {}))}

| Outcome | Baseline Reference | Random search | BOPIS |
|---|---:|---:|---:|
| Energy (J) | {_fmt(reference.get("energy_j"))} | {_fmt(random.get("energy_j"))} | {_fmt(selection.get("energy_j"))} |
| Throughput (tokens/s) | {_fmt(reference.get("tokens_per_s"))} | {_fmt(random.get("tokens_per_s"))} | {_fmt(selection.get("tokens_per_s"))} |
| Quality (BERTScore F1 proxy) | {_fmt(reference.get("quality_f1"), 4)} | {_fmt(random.get("quality_f1"), 4)} | {_fmt(selection.get("quality_f1"), 4)} |

The **Baseline Reference** is the measurement of the unoptimized/default
configuration used only as the formal denominator for EIR, SRR, and QRR. It is
not a third optimization method. Random Search is the method comparator with
the same search budget, while BOPIS is the Bayesian-optimization method. In
this run, Random Search consumed less computed energy but failed the
quality-retention requirement (`QRR = {_fmt(random.get("qrr_percent"))}%`), so
it is not a valid BOPIS-success configuration.

## Optimization-process metrics

| Metric | One-step-ahead | Leave-one-out | Interpretation |
|---|---:|---:|---|
| Observations | {reliability.get("n", "n/a")} | {loo.get("n", "n/a")} | Search evidence / fit evidence |
| MAE (J) | {_fmt(reliability.get("mae"))} | {_fmt(loo.get("mae"))} | Lower is better |
| NPE | {_fmt(reliability.get("npe_percent"))}% | {_fmt(loo.get("npe_percent"))}% | Target < 10% |
| R² | {_fmt(reliability.get("r_squared"), 3)} | {_fmt(loo.get("r_squared"), 3)} | Higher is better |
| UCR | {_fmt(reliability.get("ucr"), 3)} | {_fmt(loo.get("ucr"), 3)} | Target >= 0.95 |

The one-step-ahead figures are intentionally conservative because acquisition
selects uncertain points. The leave-one-out figures are the appropriate
surrogate-fit diagnostic, but this run does not meet the configured NPE/UCR
reliability targets. Do not present the simulator result as proof of real-world
GP accuracy.

| Search behavior | BOPIS | Random search |
|---|---:|---:|
| Evaluated configurations | {settings.get("n_total", "n/a")} | {settings.get("n_total", "n/a")} |
| Iterations to within 5% | {convergence.get("bopis", {}).get("iterations_to_within_5pct", "n/a")} | {convergence.get("random_search", {}).get("iterations_to_within_5pct", "n/a")} |
| Sample-efficiency ratio | {_fmt(convergence.get("ser"))} | — |
| Pareto-front size | {pareto.get("n_front", "n/a")} | {random.get("n_front", "n/a")} |

The best-configuration iteration indices are `k* = {convergence.get("bopis", {}).get("k_star", "n/a")}` for BOPIS and
`k* = {convergence.get("random_search", {}).get("k_star", "n/a")}` for random
search. The total search budget is `{settings.get("n_total", "n/a")}` evaluations
per method.

## Why the EIR criterion is `> 0%`

The **formula** uses the Baseline Reference:

```text
EIR = ((E_baseline - E_BOPIS) / E_baseline) x 100%
```

For this run:

```text
EIR = ((42.6109 - 36.3042) / 42.6109) x 100% = 14.80%
```

The `0%` shown in the criterion column is only the minimum decision threshold:
any positive EIR means BOPIS used less energy than the Baseline Reference. It
does not mean the formula divides by zero or ignores the magnitude of the
improvement. The other constraints prevent an energy reduction from being
accepted when it damages speed or quality: `SRR >= 95%` and `QRR >= 98%`.

For transparency, comparing BOPIS directly against Random Search is a separate
analysis and is not the formal EIR denominator:

```text
((14.8845 - 36.3042) / 14.8845) x 100% = -143.81%
```

That negative value reflects that the Random Search fallback selected a smaller
model with lower quality; it should not replace the default-reference EIR.

## Instruments and captured fields

| Instrument / component | Captured fields | Current run status |
|---|---|---|
| `SimulatedMeasurer` | Energy, tokens/s, generated tokens, quality proxy | Computed by analytic simulator |
| NVML via `ctypes` | GPU power/energy, utilization, VRAM | Capability detected, power unsupported |
| Kernel/process telemetry | CPU, process CPU share, RAM, thread count | Available for diagnostics |
| OHM/LHM RAPL transport | CPU package power/energy | Optional; not used by this run |
| BERTScore offline scorer | Precision, recall, F1 | Not invoked by this simulator run |
| GP/Pareto/statistics modules | MAE, NPE, R², UCR, EI, Pareto, EIR/SRR/QRR | Executed and serialized |

## Host and dataset provenance

- **CPU:** {host.get("cpu_model", "n/a")} ({host.get("logical_cores", "n/a")} logical cores)
- **GPU:** {host.get("gpu_name", "n/a")} ({host.get("vram_total_gib", "n/a")} GiB VRAM)
- **GPU power support:** `{host.get("power_supported", "n/a")}`
- **Energy counter support:** `{host.get("energy_counter_supported", "n/a")}`
- **Configuration space:** {manifest.get("configuration_space", {}).get("n_feasible", "n/a")} feasible of {manifest.get("configuration_space", {}).get("n_unconstrained", "n/a")} unconstrained
- **Dataset:** `{dataset.get("source", "n/a")}`, {dataset.get("n_evaluation", "n/a")} evaluation prompts and {dataset.get("n_proxy", "n/a")} proxy prompts
- **Seed:** `{settings.get("seed", "n/a")}`; backend seed `{backend.get("seed", "n/a")}`

## Defense-safe interpretation

1. **Claim now:** BOPIS has a functioning and traceable instrument pipeline:
   configuration search, GP surrogate, acquisition, Pareto selection, and
   reproducible artifact generation are implemented and exercised.
2. **Do not claim now:** The Joule values above are not direct measurements.
   The MX330 reports utilization and VRAM but no supported GPU power or energy
   counter.
3. **Quality caveat:** The displayed quality values are simulator proxies in
   this run. Real output-quality evidence requires the offline BERTScore pass
   over generated responses and references.
4. **Next empirical step:** Run a hardware campaign using an instrument with a
   supported NVML power/energy signal, or explicitly use CPU-package RAPL for
   CPU-only inference and label the energy scope.

## Chapter 3 alignment note

The revised system-architecture narrative remains consistent with the
implementation: profiling and feasibility filtering feed the search, inference
feeds the measurement layer, GP/EI guides candidate selection, Pareto analysis
selects `x*`, and artifacts feed the statistical/reporting layer. The main
wording difference to preserve in the manuscript is the distinction between
**measured**, **estimated**, and **simulated** energy.

## Reproduce

```powershell
python tools\\initial_results.py {run_dir}
```

The generator reads only `manifest.json` and `metrics.json`; it does not alter
the run artifacts.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default=DEFAULT_RUN)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    manifest_path = os.path.join(run_dir, "manifest.json")
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.isfile(manifest_path) or not os.path.isfile(metrics_path):
        parser.error(f"run directory must contain manifest.json and metrics.json: {run_dir}")

    report = render(_load(manifest_path), _load(metrics_path), args.run_dir)
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
