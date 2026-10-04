# BOPIS Data Storytelling and Mock Defense Guide

## 1. The one-minute story

> Local LLM inference has competing objectives: energy, speed, and output
> quality. Choosing a configuration manually is difficult because the best
> setting depends on the model, hardware, and runtime parameters.
>
> BOPIS treats configuration selection as a black-box optimization problem. It
> first removes configurations that the hardware cannot run, evaluates seed
> configurations, fits a Gaussian Process surrogate, and uses Expected
> Improvement to choose promising candidates. It then applies Pareto analysis
> to retain non-dominated trade-offs and selects the lowest-energy eligible
> configuration that satisfies the predefined speed and quality floors.
>
> The dashboard is an evidence trail: it shows the configuration space, search
> decisions, predictions and uncertainty, measured or estimated outcomes,
> Pareto front, comparison with Random Search, selection status, and deployment
> state.

The story is:

1. **Problem:** Configuration choices affect energy, speed, and quality.
2. **Conflict:** These objectives can improve in different directions.
3. **Method:** Bayesian Optimization reduces expensive evaluations.
4. **Decision:** Pareto analysis and explicit thresholds select `x*`.
5. **Evidence:** Logs, metrics, run manifests, charts, and statistical outputs
   make every decision traceable.
6. **Limitation:** The current laptop cannot directly measure MX330 GPU power,
   so simulator and resource-allocation runs must not be presented as direct
   empirical GPU-energy measurements.

## 2. What to say during the demo

### Beat 1 — Define

> This tool does not optimize energy in isolation. It searches for a
> configuration that reduces energy while retaining acceptable speed and
> output quality.

Point to the dashboard journey:

`Define -> Search -> Compare -> Deploy`

### Beat 2 — Search

Click **Replay search**.

> The initial evaluations provide observations for the model. After that,
> Bayesian Optimization uses the observations and their uncertainty to select
> the next candidate. The search log shows the proposed configuration, the
> predicted energy, the uncertainty, the actual result, and the prediction
> error.

### Beat 3 — Explain the surrogate

Open the explanation view and point to:

- the number of observations used by the GP;
- GP signal variance, noise variance, and length scale;
- leave-one-out `R²`;
- normalized prediction error;
- the percentage of observations inside the prediction band.

> A surrogate model is a cheaper stand-in for the expensive experiment. The
> Gaussian Process predicts two values for an unevaluated configuration:
> expected energy `mu` and uncertainty `sigma`.

### Beat 4 — Explain Expected Improvement

> Expected Improvement balances exploitation and exploration. A candidate is
> attractive if its predicted energy is low, but an uncertain candidate can
> also be selected if it has a meaningful chance of improving the current
> best result.

For constrained search:

> The acquisition score is also multiplied by the probability that the
> candidate meets the speed and quality floors.

### Beat 5 — Compare

> Random Search uses the same configuration space, evaluation budget, and
> selection rule. The intended difference is the search strategy: random
> selection versus Bayesian-guided selection.

Do not claim that Bayesian Optimization always wins. Say that the comparison
tests whether Bayesian guidance is more sample-efficient under the controlled
setup.

### Beat 6 — Select and deploy

> `x*` is not simply the lowest-energy point. It is selected from the Pareto
> front after checking the predefined floors:
>
> - `EIR > 0%`
> - `SRR >= 95%`
> - `QRR >= 98%`
>
> The selected configuration can then be loaded into the local llama.cpp
> server for deployment.

The current system chooses one workload-level configuration. It does not
switch precision, GPU layers, batch size, or CPU threads for every individual
prompt.

## 3. How joules are obtained

### 3.1 Direct measurement path

When the hardware exposes a usable NVML power or energy signal, BOPIS uses a
measurement path:

1. Prefer the driver's monotonic energy counter, when available:

   ```text
   E = E_counter_end - E_counter_start
   ```

2. Otherwise integrate sampled board power over measured timestamps:

   ```text
   E_net = sum over intervals [
       0.5 * (max(P_0 - P_idle, 0) + max(P_1 - P_idle, 0)) * dt
   ]
   ```

   where:

   - `P_0` and `P_1` are power readings in watts;
   - `P_idle` is the idle baseline;
   - `dt` is the actual elapsed time between samples;
   - energy is in joules because watts multiplied by seconds equals joules.

3. If an external CPU-package source is configured, BOPIS can also record the
   CPU-package energy scope. This is not the same scope as GPU-only energy.

Afterward:

```text
J/token = E / generated_tokens
```

The implementation uses timestamp-based trapezoidal integration rather than
assuming that every sampler interval is exactly 0.1 seconds.

### 3.2 Fallback when GPU energy cannot be fetched

The development laptop's NVIDIA GeForce MX330 reports utilization and VRAM,
but does not expose the required NVML power or total-energy counter. In
resource-estimate mode, BOPIS uses:

```text
E_est = [
    (P_cpu_tdp - P_cpu_idle) * u_cpu_proc
  + (P_gpu_tdp - P_gpu_idle) * max(u_gpu - u_gpu_idle, 0)
] * T
```

Where:

- `T` = inference-window duration in seconds;
- `P_cpu_tdp` = declared CPU full-load power budget in watts;
- `P_gpu_tdp` = declared GPU power budget in watts;
- `P_cpu_idle` and `P_gpu_idle` = declared idle powers;
- `u_cpu_proc` = inference process CPU-seconds divided by
  `logical_cores * T`, bounded to `[0, 1]`;
- `u_gpu` = observed mean GPU utilization as a fraction;
- `u_gpu_idle` = idle GPU utilization fraction;
- `max(u_gpu - u_gpu_idle, 0)` = baseline-corrected GPU utilization.

The result is in joules:

```text
watts * seconds = joules
```

Then the same token normalization may be reported:

```text
J_est/token = E_est / generated_tokens
```

The default resource-estimate budgets in the implementation are:

- CPU: `15 W`;
- GPU: `25 W`;
- declared fractional budget uncertainty: `0.30` or `30%`.

These are power-budget assumptions, not readings from the GPU power sensor.

## 4. What does the 30% uncertainty mean?

The estimator calculates a band using independent uncertainty on the CPU and
GPU budget contributions:

```text
sigma = sqrt(
    (0.30 * E_cpu)^2
  + (0.30 * E_gpu)^2
)

energy_low  = max(E_est - sigma, 0)
energy_high = E_est + sigma
```

This is a **declared uncertainty band for the power-budget inputs**. It is not
a validated confidence interval for the true physical joules.

If the independent errors were actually Gaussian and the model form were
correct, `+-1 sigma` would conventionally correspond to approximately 68%
coverage. BOPIS must not claim “68% confidence” here because:

- the 30% value is researcher-declared, not calibrated from repeated
  instrument measurements;
- the linear utilization-to-power relationship is an approximation;
- DVFS, workload-dependent power, and GPU-kernel behavior are not fully
  observed;
- model-form error is not included in the band.

The defensible statement is:

> The fallback gives a labelled resource-allocation estimate with a declared
> budget-uncertainty band. It supports relative comparison on the same host,
> using the same estimator and session. It does not support reporting an
> absolute measured energy value or a validated statistical confidence level.

Relative ratios are more defensible than absolute joules because common
systematic assumptions partially cancel:

```text
relative reduction = (E_A_est - E_B_est) / E_A_est
```

Still call the result an estimated or resource-allocation comparison.

## 5. Have we implemented Bayesian Optimization?

Yes. The implementation contains the core parts:

### Gaussian Process

`bopis/gp.py` implements:

- an RBF/squared-exponential kernel;
- GP posterior mean;
- GP posterior variance;
- Cholesky factorization;
- log marginal likelihood;
- hyperparameter fitting;
- predictive uncertainty.

The code uses a coarse log-space search followed by multi-start
Nelder-Mead. The manuscript's earlier wording said L-BFGS-B, but the actual
standard-library implementation uses Nelder-Mead. Defense wording should match
the code:

> The GP hyperparameters are fitted by maximizing the log marginal likelihood
> using a derivative-free grid and multi-start Nelder-Mead procedure.

### Surrogate modelling

Yes. The GP is the surrogate model. It approximates the expensive relationship
between configuration features and observed energy. For each unevaluated
configuration it returns:

```text
mu     = predicted energy
sigma  = predictive uncertainty
```

### Expected Improvement

Yes. `bopis/acquisition.py` implements the minimization form:

```text
improvement = f_best - mu - xi
Z = improvement / sigma
EI = improvement * Phi(Z) + sigma * phi(Z)
```

For constrained acquisition:

```text
score = EI * P(quality meets floor) * P(speed meets floor)
```

This sign matters because BOPIS minimizes energy. A maximization-style
`mu - f_best` expression would reward higher predicted energy and would be
incorrect for this objective.

### Sequential loop

The optimizer follows this sequence:

1. generate feasible configurations;
2. evaluate initial seed configurations;
3. fit/update the GP;
4. score unevaluated candidates with EI;
5. evaluate the selected candidate;
6. add the result to the observations;
7. repeat until the evaluation budget is exhausted;
8. apply Pareto analysis and select `x*`.

## 6. Can we demo the Bayesian method in the UI?

Yes. Use this sequence:

1. Open the **Optimization Dashboard**.
2. Load a run or start a study.
3. Click **Replay search** in the Pareto card.
4. Let the first seed evaluations appear.
5. Point to the search log as the guided evaluations appear.
6. Open the explanation view.
7. Show the GP section and its `mu`, `sigma`, hyperparameters, LOO metrics,
   and prediction-band coverage.
8. Show the Expected Improvement table with:
   - proposed configuration;
   - predicted `mu +- sigma`;
   - EI score;
   - probability of meeting floors, if constrained;
   - measured or simulated energy;
   - prediction error.
9. Show the convergence chart and Pareto front.
10. Show the final selection status and whether it is a clean threshold-valid
    selection or a fallback.

Use this sentence:

> The UI demonstrates the decision trace of Bayesian Optimization: the
> surrogate predicts, Expected Improvement chooses, the evaluator produces an
> outcome, and the new observation updates the next decision.

The UI demonstrates implementation and traceability. It does not by itself
prove that the method generalizes to all hardware or workloads.

## 7. Important defense limitations

### Simulator runs

If the run manifest says `backend: sim`, say:

> This is an instrument-validation and pipeline-demonstration run. Its
> outputs are computed by the analytic simulator and are not empirical
> measurements.

### Resource-estimate runs

Say:

> The energy values are Mode C resource-allocation estimates because the GPU
> has no usable power telemetry. They are useful for same-host relative
> comparison, not as calibrated absolute joules.

### Fallback selection

If the selection status is `min_energy_fallback`, say:

> No Pareto member met all formal retention thresholds, so the system reported
> a fallback minimum-energy member. This run demonstrates correct fallback
> handling, but it is not a formally successful optimization.

### Model scope

The manuscript emphasizes Qwen2.5-1.5B, but some development runs use
Qwen2.5-0.5B. Do not use a 0.5B development run as evidence for a 1.5B
experimental claim without explaining the difference.

## 8. Common panel questions

### “Did you train the language model?”

> No. The LLM is the black-box system being evaluated. The learned components
> are the task classifier and the Gaussian Process surrogate.

### “Is this supervised learning or reinforcement learning?”

> The classifier is supervised learning. The GP is supervised regression used
> inside a sequential Bayesian Optimization loop. It is not reinforcement
> learning because there is no policy learning from a reward signal.

### “Why not exhaustive search?”

> Each candidate requires an inference evaluation. Bayesian Optimization uses
> previous observations and uncertainty to prioritize candidates, reducing the
> number of expensive evaluations.

### “Why Pareto analysis?”

> Energy, speed, and quality conflict. Pareto analysis preserves the
> non-dominated choices instead of hiding the trade-off inside an arbitrary
> weighted score.

### “What proves that you followed the SOP?”

Use this structure:

> SOP step -> implementation action -> stored artifact -> acceptance criterion.

Examples:

| SOP step | Implementation evidence |
|---|---|
| Define feasible space | Hardware profile and rejected-configuration reasons |
| Initialize search | Seed rows in the search log and run manifest |
| Fit surrogate | GP hyperparameters, predictions, and uncertainty |
| Select candidate | EI and constrained-feasibility values |
| Evaluate | Raw generation records and metric rows |
| Compare | BOPIS and Random Search using the same budget |
| Select `x*` | Pareto membership, EIR, SRR, QRR, and selection status |
| Validate | Surrogate diagnostics and statistical-validation output |
| Deploy | Active llama.cpp configuration and chat response |

## 9. Final credibility rule

Always distinguish these four statements:

1. **Implemented:** the code path exists and tests exercise it.
2. **Demonstrated:** the UI can show the decision trace.
3. **Observed:** a particular run produced a particular output.
4. **Proven empirically:** repeated controlled experiments support a
   generalizable conclusion.

The current simulator and Mode C runs support the first three, with explicit
limitations. They do not automatically support the fourth.
