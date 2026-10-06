# BOPIS Initial Results and Instruments

**Generated from:** `runs\20261005T085002Z_ui-sim`  
**Run timestamp:** `2026-10-05T08:50:42.086420+00:00`  
**Backend:** `sim` (analytic simulator)

## Executive result

This is an initial, simulator-backed result for defense preparation. It
validates the optimization pipeline and its reporting instruments; it is **not
an empirical hardware-energy result**. The run's own manifest states that the
analytic simulator produces computed outputs.

| Indicator | BOPIS | Criterion | Status |
|---|---:|---:|---|
| Energy Improvement Ratio (EIR) | 14.80% | > 0% | PASS |
| Speed Retention Ratio (SRR) | 122.59% | >= 95% | PASS |
| Quality Retention Ratio (QRR) | 98.26% | >= 98% | PASS |
| Selection verdict | `optimal` | clean selection | PASS |

**Selected BOPIS configuration:** `qwen2.5-1.5b` / `Q8_0` / `t=512` / `b=2` / `g=14` / `c=4`

| Outcome | Baseline Reference | Random search | BOPIS |
|---|---:|---:|---:|
| Energy (J) | 42.61 | 14.88 | 36.30 |
| Throughput (tokens/s) | 67.57 | 184.70 | 82.84 |
| Quality (BERTScore F1 proxy) | 0.8096 | 0.6692 | 0.7955 |

The **Baseline Reference** is the measurement of the unoptimized/default
configuration used only as the formal denominator for EIR, SRR, and QRR. It is
not a third optimization method. Random Search is the method comparator with
the same search budget, while BOPIS is the Bayesian-optimization method. In
this run, Random Search consumed less computed energy but failed the
quality-retention requirement (`QRR = 82.66%`), so
it is not a valid BOPIS-success configuration.

## Optimization-process metrics

| Metric | One-step-ahead | Leave-one-out | Interpretation |
|---|---:|---:|---|
| Observations | 20 | 30 | Search evidence / fit evidence |
| MAE (J) | 16.69 | 13.36 | Lower is better |
| NPE | 35.54% | 19.29% | Target < 10% |
| R² | -1.345 | 0.874 | Higher is better |
| UCR | 0.950 | 0.900 | Target >= 0.95 |

The one-step-ahead figures are intentionally conservative because acquisition
selects uncertain points. The leave-one-out figures are the appropriate
surrogate-fit diagnostic, but this run does not meet the configured NPE/UCR
reliability targets. Do not present the simulator result as proof of real-world
GP accuracy.

| Search behavior | BOPIS | Random search |
|---|---:|---:|
| Evaluated configurations | 30 | 30 |
| Iterations to within 5% | 6 | 9 |
| Sample-efficiency ratio | 0.53 | — |
| Pareto-front size | 11 | 14 |

The best-configuration iteration indices are `k* = 17` for BOPIS and
`k* = 9` for random
search. The total search budget is `30` evaluations
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

- **CPU:** 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (8 logical cores)
- **GPU:** NVIDIA GeForce MX330 (2.0 GiB VRAM)
- **GPU power support:** `False`
- **Energy counter support:** `False`
- **Configuration space:** 320 feasible of 2304 unconstrained
- **Dataset:** `databricks/databricks-dolly-15k`, 500 evaluation prompts and 20 proxy prompts
- **Seed:** `0`; backend seed `0`

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
python tools\initial_results.py runs\20261005T085002Z_ui-sim
```

The generator reads only `manifest.json` and `metrics.json`; it does not alter
the run artifacts.
