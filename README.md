# BOPIS

**Bayesian Optimization and Pareto-Based Intelligent Configuration Selection for
Energy-Efficient Local LLM Inference**

The software artifact for the thesis of the same name. BOPIS searches the space
of llama.cpp runtime configurations for one that minimizes GPU energy per prompt
while retaining inference speed and output quality, then validates that choice
against an unoptimized default and a random-search baseline.

```
python -m bopis profile                          # what can this machine measure?
python -m bopis dataset                          # fetch and stratify Dolly 15k
python -m bopis run --backend sim --synthetic    # full study, no GPU needed
```

---

## What it does

Five stages, following Chapter 3 of the manuscript:

| Stage | What happens |
|---|---|
| **0. Reference** | The unoptimized default is measured on the 50-prompt proxy subset, outside both search budgets, to supply search-time retention ratios. |
| **1. BOPIS search** | 10 configurations drawn using a task-informed precision prior, then 20 chosen by maximizing Expected Improvement over a Gaussian Process surrogate of energy. |
| **2. Random search** | The same 30-evaluation budget, sampled uniformly without replacement, selected by the same rule — so the comparison isolates the surrogate's contribution. |
| **3. Validation** | Three-way comparison on the full 500-prompt evaluation set: default, random-search winner, and `x*`. |
| **4. Analysis** | Descriptives, Friedman + Nemenyi per dependent variable, EIR/SRR/QRR, surrogate reliability, hypervolume, amortization. |

Everything lands in `runs/<UTC-timestamp>/` as the Appendix B tables, plus a
manifest recording which machine, which rules narrowed the search space, and
**which energy instrument was actually used**.

## Dependency policy

The measurement and optimization core uses **only the Python standard library** —
the Gaussian Process (RBF kernel, Cholesky, log marginal likelihood,
Nelder–Mead hyperparameter fitting), Expected Improvement, Pareto dominance and
hypervolume, Friedman and Nemenyi, NVML via `ctypes`, `/proc` and `kernel32`
telemetry, and all CSV I/O.

The single exception is `bopis/quality/bertscore.py`, which needs a transformer
forward pass and therefore imports `transformers` and `torch`. It runs as a
separate offline scoring stage; the measurement path never loads it.

This is enforced mechanically, not by intention — `tests/test_provenance.py`
parses every module's imports *and* re-imports the whole core with
`site-packages` stripped from `sys.path`.

## Install

Nothing to install for the core. Python 3.9+.

```bash
git clone https://github.com/yskooo/bopis
cd bopis
python -m bopis --help
```

> On this development machine, invoke `python` (3.11) rather than `python3`,
> which resolves to a different, bare 3.10 installation.

For the quality-scoring stage only:

```bash
python -m pip install transformers torch bert-score
```

## Start here: `profile`

Run this first on any new machine. It detects the host, applies the Table H1
rules, and — importantly — tells you whether GPU energy is measurable at all
before you commit to a multi-hour study.

```
$ python -m bopis profile --model-aware

-- GPU -------------------------------------------------------------------
  GPU                          NVIDIA GeForce MX330
  VRAM                         2.00 GiB
  Power query                  NOT SUPPORTED
  Energy counter               NOT SUPPORTED
  Energy method                unavailable

-- ENERGY MEASUREMENT ----------------------------------------------------
  This GPU reports neither power nor energy telemetry.
  GPU energy CANNOT be measured on this machine. Three options,
  in descending order of what the result can claim:

  1. Run the study on a GPU whose driver exposes
     nvmlDeviceGetPowerUsage. [...]
  2. Run with `--energy-mode resource-estimate` for a labelled
     resource-allocation estimate [...]
  3. Use `--backend sim` to exercise the pipeline with no
     hardware claim at all.

-- TABLE H1 -> FEASIBLE SPACE --------------------------------------------
  Rules fired                  HW-P1, HW-G1, HW-B2, HW-C2
  |X_feasible|                 32 of 768 unconstrained
  Rejected                     {'HW-P0': 32}
```

Energy measurement follows a fallback ladder, and every record states which rung
it used:

1. `nvmlDeviceGetTotalEnergyConsumption` — a driver-maintained millijoule
   counter. Exact: no integration error, no idle-power subtraction.
2. Integrating `(P_t − P_idle)` over the inference window at 100 ms — Chapter 3's
   documented method, used when the counter is unavailable.
3. `resource_allocation_estimate` — **opt-in, and not a measurement.** Observed
   per-process CPU time and GPU duty cycle scaled by declared power budgets.
   See [Estimating energy without a power sensor](#estimating-energy-without-a-power-sensor).
4. `unavailable` — the run refuses to emit energy claims unless
   `--allow-no-power` is passed.

### Estimating energy without a power sensor

Most entry-level mobile GPUs — the MX330 among them — have no power-monitoring
circuit, so no software can measure their energy. If the requirement is a
*measured* GPU-only figure, the only answer is a GPU whose driver exposes the
power query; nothing in this repository substitutes for the missing instrument.

What can be done honestly is to estimate from what *is* observable, and label
the difference everywhere the number appears:

```bash
python -m bopis run \
  --backend llama-server --model-aware \
  --llama-binary /path/to/llama-server \
  --model Q4_K_M=/models/mistral-7b-instruct-v0.3.Q4_K_M.gguf \
  --energy-mode resource-estimate \
  --cpu-tdp-w 15 --cpu-idle-w 2.5 \
  --gpu-tdp-w 25 --gpu-idle-w 1.5 \
  --estimate-uncertainty 0.30 \
  --idle-seconds 60 \
  --out runs --label estimated
```

```
E_est = [ (P_cpu_tdp - P_cpu_idle) * u_cpu_proc
        + (P_gpu_tdp - P_gpu_idle) * max(u_gpu - u_gpu_idle, 0) ] * T
```

Three properties make it defensible rather than decorative:

- **`u_cpu_proc` is the inference process's own CPU share**, read from
  `GetProcessTimes` / `/proc/[pid]/stat`, so a browser running alongside the
  study is not charged to it. Each row records which signal was used in
  `cpu_attribution`, and Table B.6 carries `process_cpu_percent` next to the
  machine-wide `cpu_percent`.
- **The coefficients are dynamic ranges, not nameplates.** Multiplying TDP by
  utilization would charge the idle draw again at every load level.
- **`u_gpu_idle` is measured, not assumed** — Chapter 3's 60-second idle
  protocol runs over every signal the host exposes, and the run warns loudly
  when the machine was not actually idle during it.

Every estimated row carries `energy_low_j` / `energy_high_j`, the formula in
`energy_basis`, and `energy_scope: estimated_resource_allocation`; the manifest
stores all six modelling assumptions verbatim, and the CLI and dashboard both
render the caveat. The estimate supports **relative** comparison between
configurations on one host — not absolute joule figures, and never a measured
energy claim.

Full derivation, the assumptions with their error directions, the external-meter
calibration procedure, and paste-ready Chapter 3 text:
**[docs/ENERGY_MODES.md](docs/ENERGY_MODES.md)**.

## Backends

| Backend | Use |
|---|---|
| `sim` | Deterministic analytic model. No weights, no GPU. The whole pipeline is runnable and testable anywhere, and because the objectives are closed-form the true Pareto front can be brute-forced — which is what makes "BOPIS found the optimum" an assertion rather than a hope. |
| `llama-server` | llama.cpp's OpenAI-compatible server, `temperature=0.0`. Precision, GPU layers, threads and context are *launch* flags, so the server restarts once per configuration; generation cap and concurrency are per-request. |
| `ollama` | Convenience adapter for an existing Ollama install. Not thesis-faithful; demonstration only. |

Simulated runs are labelled `energy_scope: simulated` everywhere they appear,
including a banner across the dashboard, so a computed figure can never be
mistaken for a measured one.

### A real measured run

```bash
python -m bopis run \
  --backend llama-server \
  --llama-binary /path/to/llama-server \
  --model F16=/models/mistral-7b-instruct-v0.3.F16.gguf \
  --model Q8_0=/models/mistral-7b-instruct-v0.3.Q8_0.gguf \
  --model Q4_K_M=/models/mistral-7b-instruct-v0.3.Q4_K_M.gguf \
  --model-aware --min-gpu-layers 14
```

What happens, in order: `P_idle` is calibrated for 60 s with nothing running;
the Table H1 rules plus the model-size guards build `X_feasible`; then for each
configuration the server is launched once (precision, GPU layers, threads and
context are *launch* flags), and each prompt is bracketed by a 100 ms sampler
thread while `n_predict` and concurrency vary per request.

`--model-aware` enables the memory guards, and `--min-gpu-layers` sets the
energy-scope floor. Pass a `--model` per variant you want searched; a variant
with no GGUF is rejected loudly rather than silently substituted.

Output quality is *not* scored during the run — BERTScore is a separate offline
stage, so the measurement path never loads torch. `quality_f1` is left empty in
Table B.5 until it runs.

## The dashboard

`python -m bopis run` writes `dashboard_data.js` into the run directory. Copy it
next to `dashboard/index.html` and open that file — no server, no build step.

The integrated UI is [bopis.html](bopis.html). After every completed
`python -m bopis run`, BOPIS automatically publishes the newest payload to the
repository root as `dashboard_data.js`; opening `bopis.html` then shows that
run without manually selecting a file. The authoritative copy remains in
`runs/<your-run>/dashboard_data.js`. The chart is not live telemetry: its
points, surface, Pareto front, and metrics come from the completed run artifact.

### Easiest Windows demo

From Command Prompt or PowerShell in the repository folder:

```powershell
cd C:\Users\User\Downloads\bopis
python -m bopis profile --model-aware --write-js bopis_profile.js
python -m bopis run --backend sim --synthetic
start .\bopis.html
```

After the run completes, the chart data is already published automatically. The
simulated energy values are for demonstrating the pipeline only; they are not
measured GPU energy.

### Easiest local chatbot launch on Windows

This starts llama.cpp directly and is useful for testing the chat UI. Replace
`MODEL_PATH` with a local GGUF file. Keep this terminal open:

```cmd
"C:\path\to\llama-server.exe" ^
  --model "C:\path\to\model.Q4_K_M.gguf" ^
  --n-gpu-layers 0 ^
  --threads 4 ^
  --parallel 1 ^
  --ctx-size 2048 ^
  --host 127.0.0.1 ^
  --port 8080
```

Then open `bopis.html` and use **Optimization Chat**. The page calls
`http://127.0.0.1:8080/v1/chat/completions` with structured role messages.
This prevents the model from interpreting the conversation as text to continue.
For a measured BOPIS deployment using the selected `x*`, use `python -m bopis
serve` instead; see [docs/CHATBOT_INTEGRATION.md](docs/CHATBOT_INTEGRATION.md).

Eight panels: run summary with the verdict badge, the computed Pareto front,
convergence and SER, GP reliability, the three-way comparison with Friedman and
Nemenyi, per-variant descriptives, hardware and feasible space, and amortization.

Every value is derived from the run's artifacts. A test asserts that no literal
metric appears in the HTML, because the mockup this replaces hardcoded all of
them — its "Pareto front" was an eleven-element array with membership *asserted*
rather than computed.

## Applying the result to a chatbot

The Dolly 15k dataset is used for standardized evaluation, not as chatbot
memory. After a completed study, `bopis serve` reads the selected `x*` from
`selection.csv`, validates it against the current machine, and launches
`llama-server` with the selected model, GPU layers, CPU threads, and parallel
slots. See [`docs/CHATBOT_INTEGRATION.md`](docs/CHATBOT_INTEGRATION.md) for the
evaluation-to-deployment workflow and the native llama.cpp endpoint example.

## Tests

```bash
python -m unittest discover -s tests
```

No GPU, no model weights, no network, and no third-party packages required.
Where `numpy`/`scipy`/`sklearn` happen to be installed they are used as
cross-checks and skipped otherwise:

- the from-scratch GP reproduces sklearn's log marginal likelihood to 8 decimal
  places and its posterior mean to ~1e-7;
- the Friedman statistic matches scipy exactly, ties included;
- exact hypervolume matches a Monte-Carlo estimate;
- Pareto fronts match brute force on random populations.

Two tests exist specifically to catch quiet regressions:
`test_prefers_lower_predicted_energy` fails against the manuscript's printed EI
formula, and `TestStandardLibraryOnly` fails the moment the core acquires a
third-party import.

## Layout

```
bopis/
  cli.py  __main__.py       python -m bopis {profile,dataset,run,report}
  config_space.py           x = (t, b, p, g, c); GP encoding
  hardware.py               host profiling; Table H1; model-size guards
  tasks.py                  8 task categories; the precision prior
  gp.py                     RBF kernel, Cholesky, log-ML, Nelder-Mead
  acquisition.py            Expected Improvement (minimization)
  optimizer.py              the 6-step BO loop; random-search baseline
  pareto.py                 dominance, front, hypervolume, x* selection
  metrics.py                EIR/SRR/QRR, MAE/NPE/R2/UCR, SER, amortization
  stats.py                  Friedman, Nemenyi, chi-square, descriptives
  dataset.py                Dolly 15k fetch, filter, stratified sampling
  measure.py                per-prompt measurement; energy scope
  runner.py                 study orchestration
  artifacts.py  schemas.py  run directories; frozen table schemas
  dashboard.py              dashboard payload builder
  monitor/                  NVML via ctypes; /proc and kernel32 telemetry
    nvml.py                 the driver bindings and the energy-method ladder
    sampler.py              100 ms sampling; integration; idle calibration
    estimator.py            resource-allocation estimate for unmetered hosts
    platform_os.py          system and per-process CPU/RAM, both platforms
  backends/                 llama-server, ollama, simulator
dashboard/index.html        results dashboard (file:// openable)
docs/AMENDMENTS.md          manuscript edits the artifact requires
docs/ENERGY_MODES.md        measured vs estimated energy; the estimator's model
tests/                      stdlib unittest
```

## Where the code and the manuscript disagree

Building this surfaced several places where Chapter 3 cannot be implemented as
written. They are catalogued with reasons and evidence in
**[`docs/AMENDMENTS.md`](docs/AMENDMENTS.md)**. The ones that change results:

- **The Expected Improvement equation has an inverted sign.** Energy is
  minimized, so improvement is `f(x⁺) − μ(x)`, not `μ(x) − f(x⁺)`. As printed it
  rewards configurations predicted to use *more* energy.
- **Table H1 permits configurations that cannot physically run** — F32 on a
  12 GB card (a 7B model at F32 is ~27 GiB), Q8_0 on a 2 GB card. Two
  model-size guards were added.
- **GPU-only energy accounting is invalid at `g = 0`.** With no layer offloaded,
  measured GPU energy approaches idle regardless of CPU work, so the optimizer
  would "discover" that CPU-only inference is free.
- **`t` had to be redefined** from context window to generation cap, or it either
  does nothing to energy or destroys the three context-bearing task categories by
  truncation.
- **The NPE < 10% reliability threshold is unreachable as defined** — measured
  across ten seeds, one-step-ahead NPE averaged 23% and never cleared 10%, while
  leave-one-out NPE on the same models averaged 11% with R² ≈ 0.94. Expected
  Improvement deliberately probes where the surrogate is least certain, so
  scoring it there measures exploration, not fit.

## Licence and attribution

Databricks Dolly 15k is used under **CC BY-SA 3.0**; the dataset SHA-256 and the
sampled row IDs are recorded in every run for reproducibility. llama.cpp and
Mistral 7B Instruct v0.3 are used under their respective licences. No model
weights are redistributed.

Polytechnic University of the Philippines · BS Computer Science · 2026
