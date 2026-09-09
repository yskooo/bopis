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
  GPU energy CANNOT be measured on this machine. Use
  --backend sim to exercise the pipeline, or run the study on a
  GPU whose driver exposes nvmlDeviceGetPowerUsage.

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
3. `unavailable` — the run refuses to emit energy claims unless
   `--allow-no-power` is passed.

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
  backends/                 llama-server, ollama, simulator
dashboard/index.html        results dashboard (file:// openable)
docs/AMENDMENTS.md          manuscript edits the artifact requires
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
