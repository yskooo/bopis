# BOPIS

**Bayesian Optimization and Pareto-Based Intelligent Configuration Selection for
Energy-Efficient Local LLM Inference**

The software artifact for the thesis of the same name. BOPIS searches the space
of llama.cpp runtime configurations for one that minimizes GPU energy per prompt
while retaining inference speed and output quality, then validates that choice
against an unoptimized default and a random-search baseline.

## Quick Start Guide (For Code Checkers & Evaluators)

If you are evaluating this repository for the thesis defense or code check, follow these steps to see BOPIS in action:

1. **Check Hardware Feasibility**:
   ```bash
   python -m bopis profile --model-aware
   ```
   *This detects the host machine, applies Table H1 constraints, and reports if energy measurement is supported.*

2. **Run the Full Pipeline (Simulation Mode)**:
   ```bash
   python -m bopis run --backend sim --synthetic
   ```
   *This executes all 5 stages of the pipeline (Task Classification, BOPIS Search, Random Search, Validation, Analysis) using an analytic surrogate model, requiring no GPU or real weights.*

3. **Launch the Interactive Demo UI**:
   ```bash
   python -m bopis ui
   ```
   *Opens `bopis.html` in your browser, connecting to the local backend. This UI contains the Hardware Profiler, the interactive Gaussian Process demo, the Pareto frontiers, Random Search baseline comparisons, and the Chat Interface.*

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

## Full Dictionary of Folder Structure

Below is the comprehensive dictionary of the repository's architecture, separated by domain:

### Core Pipeline & Orchestration
* `bopis/cli.py`, `__main__.py` — The command-line interface entry points (`python -m bopis`).
* `bopis/runner.py` — Orchestrates the full 5-stage study pipeline.
* `bopis/optimizer.py` — The 6-step Bayesian Optimization loop and Random Search baseline execution.
* `bopis/config_space.py` — Defines the search space `x = (t, b, p, g, c)` and handles Gaussian Process encoding.

### Math, Surrogate & Statistics
* `bopis/gp.py` — The custom Gaussian Process implementation (RBF kernel, Cholesky decomposition, log-marginal likelihood, Nelder-Mead fitting).
* `bopis/acquisition.py` — Computes Expected Improvement (EI) for minimization.
* `bopis/pareto.py` — Calculates Pareto dominance, the Pareto front, normalized hypervolume, and handles `x*` selection.
* `bopis/metrics.py` — Calculates EIR, SRR, QRR, MAE, NPE, R², UCR, SER, and amortization.
* `bopis/stats.py` — Implements Friedman and Nemenyi tests, chi-square, and descriptives.

### Hardware & Telemetry
* `bopis/hardware.py` — Host profiling, applies Table H1 constraints, and enforces model-size memory guards.
* `bopis/monitor/` — Directory containing all telemetry and energy measurement modules.
  * `nvml.py` — NVIDIA driver bindings (via `ctypes`) and the energy-method fallback ladder.
  * `sampler.py` — Executes the 100ms background sampling thread, energy integration, and idle calibration.
  * `estimator.py` — Resource-allocation energy estimator for hosts lacking hardware power sensors.
  * `platform_os.py` — System and per-process CPU/RAM telemetry for Windows and Linux.

### Classification & Quality
* `bopis/tasks.py` / `classify.py` / `classify_trained.py` — Defines the 8 task categories, implements rule-based and Naive Bayes Stage 1 Task Classification, and handles the precision prior distributions.
* `bopis/quality/bertscore.py` — The offline scoring stage for output quality evaluation using BERTScore.
* `bopis/dataset.py` — Fetches, filters, and performs stratified sampling on the Databricks Dolly 15k dataset.

### UI & Presentation
* `bopis.html` — The primary interactive User Interface for demonstrating the thesis (Chat UI, GP Demo, Pareto Charts, Random Search comparisons).
* `dashboard_data.js` — The auto-published payload containing all metrics from the latest run, consumed by the UI.
* `bopis_rules.js` / `bopis_model.js` — Exported Stage 1 Task Classification rules and trained weights for local UI execution.
* `bopis_profile.js` — Exported hardware profile data for the UI.

### Miscellaneous
* `runs/` — Automatically generated directory where all study artifacts, tables (CSV), and logs are saved per run.
* `models/` — Default directory for downloaded GGUF model weights.
* `tests/` — Standard library `unittest` suite ensuring mathematical correctness without external dependencies.
* `docs/` — Contains supplementary documentation (`AMENDMENTS.md`, `ENERGY_MODES.md`, `UI_GUIDE.md`).

---

## Dependency policy

The measurement and optimization core uses **only the Python standard library** —
the Gaussian Process, Expected Improvement, Pareto dominance, Friedman and Nemenyi, NVML via `ctypes`, `/proc` and `kernel32` telemetry, and all CSV I/O.

The single exception is `bopis/quality/bertscore.py`, which needs a transformer forward pass and therefore imports `transformers` and `torch`. It runs as a separate offline scoring stage; the measurement path never loads it.

This is enforced mechanically: `tests/test_provenance.py` parses every module's imports and re-imports the whole core with `site-packages` stripped from `sys.path`.

## Install

Nothing to install for the core. Python 3.9+.

```bash
git clone https://github.com/yskooo/bopis
cd bopis
python -m bopis --help
```

For the quality-scoring stage only:

```bash
python -m pip install transformers torch bert-score
```

## Tests

```bash
python -m unittest discover -s tests
```

No GPU, no model weights, no network, and no third-party packages required.
Where `numpy`/`scipy`/`sklearn` happen to be installed they are used as cross-checks and skipped otherwise:

- the from-scratch GP reproduces sklearn's log marginal likelihood to 8 decimal places.
- the Friedman statistic matches scipy exactly, ties included.
- exact hypervolume matches a Monte-Carlo estimate.
- Pareto fronts match brute force on random populations.

Two tests exist specifically to catch quiet regressions:
`test_prefers_lower_predicted_energy` fails against the manuscript's printed EI formula, and `TestStandardLibraryOnly` fails the moment the core acquires a third-party import.

## Where the code and the manuscript disagree

Building this surfaced several places where Chapter 3 cannot be implemented as
written. They are catalogued with reasons and evidence in
**[`docs/AMENDMENTS.md`](docs/AMENDMENTS.md)**. 

## Licence and attribution

Databricks Dolly 15k is used under **CC BY-SA 3.0**; the dataset SHA-256 and the
sampled row IDs are recorded in every run for reproducibility. llama.cpp and
Mistral 7B Instruct v0.3 are used under their respective licences. No model
weights are redistributed.

Polytechnic University of the Philippines · BS Computer Science · 2026
