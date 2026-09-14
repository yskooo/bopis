# BOPIS Team Brief — Energy Estimator Defense + How the ML Actually Works

**Status:** draft for team sync
**Covers:** (1) why energy stays the thesis's core claim, (2) the estimator formula,
(3) accuracy & confidence, (4) RRL from 2025–2026 local-LLM energy literature,
(5) a worked walkthrough of the Bayesian Optimization / Gaussian Process machinery,
(6) Chapter 3 edit checklist, (7) open issues needing a team decision.

---

## 0. Energy is still the point — read this first

An earlier version of this advice leaned hard on a mathematical result: that
multiplicative bias in our power budgets **cancels exactly** in the Energy
Improvement Ratio (proof in §3.1). That result is correct and it is our strongest
defensive card, but it must not be mistaken for "we don't need real joules."

Energy consumption **is** the dependent variable this thesis contributes on. RQ1,
RQ2 and RQ3 all lead with it. So we need both of the following, and they are
different problems:

| Claim | Instrument needed | Status |
|---|---|---|
| "BOPIS reduces energy by X% vs. default" (EIR) | Ratio only — survives our estimator intact | **Defensible today** |
| "This configuration consumes N J/token" | Absolute joules — needs calibration or validation | **Needs §3.2** |

The plan below secures the first immediately and gives two independent routes to
the second, one of which costs nothing.

---

## 1. The formula (defense-ready statement)

Our model is a **two-component linear load-line power model**, integrated over the
inference window. Implemented in [`bopis/monitor/estimator.py`](../bopis/monitor/estimator.py).

Instantaneous inference-attributable power:

```
ΔP̂(t) = (P_cpu_max − P_cpu_idle) · u_cpu_proc(t)          ← CPU term
       + (P_gpu_max − P_gpu_idle) · max(u_gpu(t) − ū_gpu_idle, 0)   ← GPU term
```

Energy over a window of wall duration `T`:

```
Ê = ∫₀ᵀ ΔP̂(t) dt  ≈  [ (P_cpu_max − P_cpu_idle) · ū_cpu_proc
                      + (P_gpu_max − P_gpu_idle) · Δū_gpu ] · T

Ê_token = Ê / N_gen
```

### Symbol table

| Symbol | Meaning | Source | Measured? |
|---|---|---|---|
| `T` | Window wall duration (s) | `time.monotonic()` | **Yes** |
| `ū_cpu_proc` | Process CPU-seconds ÷ (`N_logical` × `T`), in [0,1] | `GetProcessTimes` / `/proc/[pid]/stat` | **Yes** |
| `ū_gpu` | Mean GPU duty cycle over window | NVML utilization, 100 ms | **Yes** |
| `ū_gpu_idle` | Idle GPU duty cycle | 60 s idle calibration | **Yes** |
| `P_*_max`, `P_*_idle` | Power budgets at saturation / rest | Vendor nameplate or meter | No — **assumed** |
| `N_gen` | Generated tokens | llama.cpp log | **Yes** |

Only the **conversion from allocation to watts** is assumed. Every utilization and
timing input is directly measured. That framing matters in a defense: we are not
guessing at the workload, we are guessing at one scalar per component.

### Three design choices to be ready to justify

1. **The coefficient is a dynamic range, not a nameplate.** A package burning 2 W
   at rest and 15 W saturated draws ≈ `2 + 0.5×13` at half load, not `0.5×15`.
   Since the objective is the *marginal* energy of inference, idle is excluded
   from the coefficient and **not** subtracted again afterwards.
2. **CPU is attributed per-process.** System-wide CPU would charge the browser and
   editor to inference energy. We use the `llama-server` child's own CPU time.
   Rows where this failed are labelled `cpu_attribution = system_wide_cpu_time`.
3. **GPU is baseline-corrected but device-level.** Consumer drivers expose no
   per-process GPU decomposition, so after subtracting the idle duty cycle the
   remainder is charged to inference (assumption A4).

---

## 2. Where the measurement actually stands on our hardware

From [`bopis_profile.js`](../bopis_profile.js):

```
gpu_name:                  "NVIDIA GeForce MX330"
power_supported:           false
energy_counter_supported:  false
energy_method:             "unavailable"
```

The MX330 has no power-monitoring circuit. Both `nvmlDeviceGetPowerUsage` and
`nvmlDeviceGetTotalEnergyConsumption` return `NVML_ERROR_NOT_SUPPORTED`. No driver
flag, elevation, or library changes this.

Available modes, per [`docs/ENERGY_MODES.md`](ENERGY_MODES.md):

| Mode | Energy source | Claim strength |
|---|---|---|
| A — Measured (NVML) | Driver counter or power integration | Absolute GPU-only J |
| B — External meter | Wall/DC-inline meter | Absolute whole-system J |
| **C — Resource estimate** | Our formula, §1 | Relative + (with §3.2) bounded absolute |

We are in Mode C. The rest of this document is about making Mode C defensible.

---

## 3. Accuracy and confidence

Three separate things get conflated here. Keep them apart in the manuscript.

### 3.1 The cancellation result — protects EIR for free

EIR is a ratio of energies measured on the same host with the same budgets. If the
estimator's bias is multiplicative — `Ê = k · E` for unknown `k` — then:

```
EIR̂ = (k·E_def − k·E_bop) / (k·E_def) = (E_def − E_bop) / E_def = EIR
```

**`k` cancels identically.** The ±30% nameplate uncertainty *is* a multiplicative
scale error on the two dynamic-range coefficients, so it does not propagate into
EIR at all.

It goes further on our actual feasible space. Because only `g = 0` survives the
VRAM constraint (see §7.1), the GPU term is ~0 in *every* condition, so:

```
EIR̂|g=0 = 1 − (P_cpu_dyn · ū_b · T_b) / (P_cpu_dyn · ū_d · T_d)
         = 1 − CPU-seconds_BOPIS / CPU-seconds_default
```

**Our EIR reduces to one minus the ratio of measured process CPU-seconds.** No TDP,
no nameplate, no assumption A1 or A2 — all cancel. That is a *measured* quantity
from `GetProcessTimes`.

What does **not** cancel: the model's *form* error (A1 linearity), because the two
conditions sit at different utilization points; and differential bias between the
CPU and GPU budgets when conditions have different CPU/GPU mixes — which at `g = 0`
also vanishes.

> **Defense line:** "For the energy *ratio* our hypotheses test, systematic error in
> the power budgets cancels by construction; only sampling uncertainty remains, and
> we quantify that by paired bootstrap."

### 3.2 Systematic error — needed for absolute J/token

Two routes, pick at least one:

**Route 1 — literature-anchored acceptance threshold (free).**
Published energy *prediction* models for LLM inference report the following
accuracy, which establishes what is normal for this class of model:

| System | Platform | Reported accuracy |
|---|---|---|
| EnerInfer (2026) | high-end phone / mid phone / **laptop** / edge board | 6.5% / 12% / **9.7%** / 5.4% deviation |
| Multi-Level Hybrid Predictor (2026) | dense + MoE | ~10% MAPE |
| EnergyLens (2026) | multi-GPU | 9.25–13.19% MAPE |
| LLMCO2 | Gemma, Bloom, Qwen2, Mixtral, Llama3.1 | 15.5% mean MAPE |
| CO2-Meter | rk3588, AGX Orin (edge) | MAPE + 10% error bounds |

So a **predefined acceptance threshold of MAPE ≤ 15%** is *literature-justified*,
not arbitrary — it sits at the permissive end of published work in exactly our
regime. This mirrors how we already justify the NPE < 10% surrogate threshold.
⚠️ Verify each figure against the paper before quoting it.

**Route 2 — validate against a real instrument.**
- **RAPL (recommended).** The i5-1135G7 exposes `MSR_PKG_ENERGY_STATUS`, a real
  monotonic joule counter. Regress measured package energy on our CPU component
  over ~50 windows; report MAE, MAPE, R², calibration slope β with 95% CI, and
  Bland–Altman limits of agreement. Because `g = 0` makes CPU dominant, this
  validates essentially the whole estimate. Needs an admin kernel driver
  (LibreHardwareMonitorLib) — **verify it works on our machine before writing it
  into Chapter 3**, and note it breaks our standard-library-only constraint.
- **Wall meter / battery discharge.** Aggregate over blocks of ~50 prompts; too
  coarse for a single 2 s prompt.

**Route 3 — external plausibility band (free, do this regardless).**
Zähl & Hennig (2026) benchmarked nine local LLMs (1B–7B) on a consumer RTX 4060Ti
via `nvidia-smi` and report **0.56 J/token (gemma3:1b)**, **0.65 J/token
(llama3.2:1b)**, with **Mistral-7B up to 4.4× higher**. If our estimated J/token
lands inside that envelope after scaling for hardware class, that is a genuine
external sanity check obtainable with zero additional equipment. If it lands far
outside, our budgets are wrong and we learn that before the defense.

### 3.3 Statistical confidence — do this now, no hardware

This is what a panel usually means by "confidence level," and
[`bopis/stats.py`](../bopis/stats.py) currently has **Friedman and Nemenyi but no
bootstrap**. That is the gap.

Paired bootstrap over the 500 prompts (resample *prompts*, not conditions — all
three conditions see the same prompt set):

```
for b in 1..10000:
    resample prompt indices with replacement
    EIR̂⁽ᵇ⁾ = 1 − Σᵢ Ê_bop,i⁽ᵇ⁾ / Σᵢ Ê_def,i⁽ᵇ⁾
report the 2.5th and 97.5th percentiles
```

Report as: **"estimated EIR = 22.4% (95% CI 18.1–26.9%)."** Same for SRR and QRR.
This is additional to Friedman/Nemenyi, which test *ordering*, not effect size.

---

## 4. RRL — 2025/2026 local-LLM energy literature

Organized by what each citation is being used to support. All arXiv IDs listed;
**pull each paper and verify the exact figures before citing.**

### 4.1 Supports: modelling/estimating inference energy is established practice

This is the core defense of the estimator — we are not improvising, we are using a
method with a current published literature.

| Citation | What it supports |
|---|---|
| **EnerInfer: Energy-Aware On-Device LLM Inference** — Zou, Liu, Sun, Mascherin, Roy, Liu, Peng, Jia & Chen (2026), *ACM SIGOPS ATC*. [arXiv:2606.23001](https://arxiv.org/abs/2606.23001) | **Our single best citation.** On-device (incl. laptop-class) energy-aware LLM inference with per-platform accuracy reported. Read its methods section for the model form. |
| **Multi-Level Modeling of LLM Inference Latency and Energy via Hybrid Analytical–ML Predictors** (2026). [arXiv:2608.06723](https://arxiv.org/html/2608.06723) | Analytical + ML energy prediction, ~10% MAPE. Precedent for a formula-based predictor. |
| **EnergyLens: Predictive Energy-Aware Exploration for Multi-GPU LLM Inference Optimization** (2026). [arXiv:2605.14249](https://arxiv.org/abs/2605.14249) | Predictive energy for *configuration exploration* — same use-case as ours. |
| **LLMCO2: Advancing Accurate Carbon Footprint Prediction for LLM Inferences**. [arXiv:2410.02950](https://arxiv.org/pdf/2410.02950) | Prediction accuracy benchmark (15.5% mean MAPE). |
| **CO2-Meter: A Comprehensive Carbon Footprint Estimator for LLMs on Edge Devices**. [arXiv:2511.08575](https://arxiv.org/pdf/2511.08575) | Estimation on resource-constrained edge hardware. |
| **PIE-P: Fine-Grained Energy Prediction for Parallelized LLM Inference**. [arXiv:2512.12801](https://arxiv.org/pdf/2512.12801) | Fine-grained (per-window) energy prediction. |
| **Where Do the Joules Go? Diagnosing Inference Energy Consumption**. [arXiv:2601.22076](https://arxiv.org/pdf/2601.22076) | Component-level energy attribution — supports our CPU/GPU term split. |

### 4.2 Supports: local/consumer-hardware LLM energy measurement (our setting)

| Citation | What it supports |
|---|---|
| **Energy Efficiency of Locally Deployed LLMs: A Preliminary Quantitative GPU Power Benchmark on Consumer Hardware** — Zähl & Hennig (2026). [arXiv:2608.00008](https://arxiv.org/abs/2608.00008) | Nine models 1B–7B on consumer RTX 4060Ti, Ollama, `nvidia-smi` @ 2 Hz. **Provides the J/token plausibility band for §3.2 Route 3.** Closest published work to our study. |
| **Scaling Laws for Energy Efficiency of Local LLMs**. [arXiv:2512.16531](https://arxiv.org/pdf/2512.16531) | Energy scaling across local model sizes — supports variant comparison. |
| **Beyond Test-Time Compute Strategies: Advocating Energy-per-Token in LLM Inference** — *EuroMLSys '25*. [arXiv:2603.20224](https://arxiv.org/pdf/2603.20224) | Justifies **J/token** as the normalized primary metric. |
| **TokenPowerBench: Benchmarking the Power Consumption of LLM Inference** — Niu et al., *AAAI*. [ojs.aaai.org](https://ojs.aaai.org/index.php/AAAI/article/view/40535/44496) | Already in our RRL — **now has a real venue (AAAI)**; fix the incomplete entry. |
| **Understanding the Energy Scaling of LLM Inference Across Context Lengths and Attention Architectures** (2026). [arXiv:2608.25096](https://arxiv.org/html/2608.25096) | Justifies **input token length** as a configuration parameter. |
| **Energy-Aware Computing in the Year 2026**. [arXiv:2605.24569](https://arxiv.org/pdf/2605.24569) | Current framing / Green-AI motivation. |

### 4.3 Supports: BO + GP surrogate for LLM inference configuration

| Citation | What it supports |
|---|---|
| **MobileLLM-Flash: Latency-Guided On-Device LLM Design** (2026). [arXiv:2603.15954](https://arxiv.org/pdf/2603.15954) | Two-stage BO for a latency–quality Pareto frontier; ~800 on-device measurements; **GP surrogate cross-validated R² = 0.97**. Direct precedent for our amendment A-40 (LOOCV instead of one-step-ahead). |
| **SCOOT: SLO-Oriented Performance Tuning for LLM Inference Engines**. [arXiv:2408.04323](https://arxiv.org/pdf/2408.04323) | **Closest related system.** Multi-objective BO over LLM inference-engine parameters, Pareto frontier of TTFT/TPOT. Belongs in Related Systems. |
| **LLM-Guided Runtime Parameter Optimization for Energy-Efficient Model Inference** (2026). [arXiv:2604.27032](https://arxiv.org/html/2604.27032) | Runtime-parameter tuning toward a lower-energy / higher-throughput Pareto region. |
| **Pimp My LLM: Leveraging Variability Modeling to Tune Inference Hyperparameters** (2026). [arXiv:2602.17697](https://arxiv.org/pdf/2602.17697) | Configuration-space framing for inference hyperparameters. |

### 4.4 Already in our reference list — repurpose here

- **Lannelongue et al. (2021)**, Green Algorithms — TDP × usage × time. Cited for EIR
  thresholds; it is also the **direct precedent for nameplate-scaled estimation**.
  Reuse it in the estimator justification.
- **Wilkins et al. (2024)** — workload-based LLM inference energy models. Cited for the
  GP surrogate; independently supports that inference energy is predictable from
  workload features.
- **Husom et al. (2024)**, MELODI — per-token normalization; energy driven by response
  characteristics.
- **Jain (1991)** — predefined decision criteria, for our acceptance threshold.

### 4.5 Optional — model-form pedigree

If asked "where does this formula come from," the linear load-line form
`P = P_idle + (P_max − P_idle)·u` originates in datacenter power provisioning:
**Fan, Weber & Barroso (2007)**, *ISCA*, and its accuracy evaluation
**Rivoire, Ranganathan & Kozyrakis (2008)**, *HotPower*
([USENIX PDF](https://www.usenix.org/legacy/events/hotpower08/tech/full_papers/rivoire/rivoire.pdf)) —
the latter finds utilization-metric models most accurate across machines, and is
especially useful where dynamic power is *not* CPU-dominated. One sentence citing
these is enough; lead the RRL with §4.1–4.2.

---

## 5. How the ML actually works (BO / GP / surrogate)

### 5.1 First, the misconception to clear up

**There is no training set and no training phase.** We do not "train BOPIS" the way
one trains a classifier. There are no epochs, no gradient descent over weights, no
train/test split of prompts.

Bayesian Optimization is **sequential experimental design**. It answers: *given that
each experiment is expensive, which experiment should I run next?*

| Conventional ML | Bayesian Optimization (ours) |
|---|---|
| Thousands of labelled examples | **30 total** inference evaluations |
| Fit once, then predict | **Refit after every single evaluation** (20×) |
| Goal: generalize to new data | Goal: **find the best configuration** in this space |
| "Training data" | "Observations" — configs we actually ran |

Our budget, from [`bopis/optimizer.py`](../bopis/optimizer.py): **10 prior-weighted
random seeds + 20 BO-guided steps = 30 evaluations**, each on the 50-prompt proxy
subset.

### 5.2 The three pieces

**(a) The surrogate — a Gaussian Process.**
Running one configuration costs minutes of inference. A GP is a cheap stand-in that,
after seeing a few real measurements, predicts for *every* unevaluated configuration:

- `μ(x)` — predicted energy
- `σ(x)` — **how unsure it is** ← this is what makes it a GP and not a regression

The `σ(x)` is the whole point. A plain regressor gives a number; a GP gives a number
*plus an honest admission of ignorance*, and BO spends its budget on that ignorance.

Implemented in [`bopis/gp.py`](../bopis/gp.py): RBF (squared-exponential) kernel,
`y` standardized, Cholesky with escalating jitter, and three hyperparameters
`θ = {σ_f², ℓ, σ_n²}` fitted by maximizing the log marginal likelihood (coarse
log-space grid + multi-start Nelder-Mead — see amendment A-7; the manuscript says
L-BFGS-B, which has no standard-library implementation).

The kernel encodes one assumption: **similar configurations have similar energy.**
`ℓ` (length scale) is *how* similar — learned from data, not set by us.

**(b) The acquisition function — Expected Improvement.**
Given `μ(x)` and `σ(x)` everywhere, which config do we actually run next? EI scores
each candidate by *expected* gain over the best result so far, `f(x⁺)`:

```
imp   = f(x⁺) − μ(x)          ← minimizing energy, so improvement is a decrease
Z     = imp / σ(x)
EI(x) = imp · Φ(Z) + σ(x) · φ(Z)
         └─ exploitation ─┘   └─ exploration ─┘
```

**(c) Pareto + constraints — where speed and quality enter.**
⚠️ **Important and often misunderstood on our own team:** the GP models **energy
only**. Confirmed in [`bopis/optimizer.py`](../bopis/optimizer.py) — the surrogate is
fitted on `energy_j`, and `gp_prediction_pairs` compares μ against measured *energy*.

Speed and quality are **not** learned by the GP. They enter afterwards:

1. BO searches for low energy → 30 evaluated configurations
2. Non-dominated sorting over (E↓, S↑, Q↑) → Pareto front
3. Filter: `SRR ≥ 95%` **and** `QRR ≥ 98%`
4. From survivors: `x* = argmax EIR(x)`

So the honest answer to *"do we train BOPIS to pick lowest energy with best speed and
quality?"* is: **BO hunts for low energy; Pareto analysis plus the SRR/QRR thresholds
guarantee speed and quality were not sacrificed to get it.** Two stages, not one
trained model.

### 5.3 Worked example — the numbers, so you can verify by hand

Suppose after 3 seed evaluations we have measured:

| Config | Measured E |
|---|---|
| c1 | 28.0 J |
| c4 | 22.0 J ← best so far, `f(x⁺) = 22.0` |
| c7 | 41.0 J |

The GP interpolates between them and reports `μ`/`σ` for the unevaluated configs.
Three candidates:

| Candidate | μ(x) | σ(x) | imp = 22−μ | Z | Φ(Z) | φ(Z) | **EI** |
|---|---|---|---|---|---|---|---|
| **A** | 20.0 | 1.00 | +2.0 | 2.00 | 0.9772 | 0.0540 | **2.008** ✅ |
| **B** | 26.0 | 8.00 | −4.0 | −0.50 | 0.3085 | 0.3521 | **1.583** |
| **C** | 21.5 | 0.05 | +0.5 | 10.00 | ≈1.000 | ≈0.000 | **0.500** |

Arithmetic for B: `−4.0 × 0.3085 + 8.0 × 0.3521 = −1.234 + 2.817 = 1.583`

**Read the ranking carefully — it is the entire intuition of BO:**

- **A wins** — predicted better than the incumbent *and* reasonably certain.
- **B comes second even though it is predicted to be WORSE than what we already have
  (26 J vs 22 J).** Its `σ = 8.0` means the GP has almost no idea what's there — it
  could plausibly be 18 J. That possibility is worth buying. **This is exploration.**
- **C comes last even though it is predicted BETTER than the incumbent (21.5 J).** Its
  `σ = 0.05` means we already effectively know the answer, so there is nothing left to
  learn. A greedy optimizer would pick C and stall.

### 5.4 Visualizing the posterior

After 3 seeds (● = measured, ▓ = ±σ uncertainty band):

```
  E (J)
   45 |                              ●  c7
      |                          ▓▓▓▓
   40 |                      ▓▓▓▓▓▓
      |                  ▓▓▓▓▓
   35 |              ▓▓▓▓▓▓▓
      |          ▓▓▓▓▓▓▓▓▓        ← wide band: GP is UNSURE here
   30 |  ●   ▓▓▓▓▓▓▓▓▓                (high EI → explore, candidate B)
      |  c1    ╲
   25 |         ╲___
      |             ●  c4          ← narrow band near observations
   20 |             ↑                 (low EI → nothing to learn, candidate C)
      +----+----+----+----+----+----+----+----+
        c1   c2   c3   c4   c5   c6   c7   c8
             ↑              ↑
          candidate C    candidate B
        (certain, dull) (unknown, tempting)
```

Each iteration: run the argmax-EI config → add the real measurement → the band
collapses *there* → refit → repeat. Uncertainty shrinks where we look, so the search
naturally stops re-testing what it already knows.

### 5.5 What gets reported about the search itself

From [`bopis/optimizer.py`](../bopis/optimizer.py), per iteration:

| Field | Meaning |
|---|---|
| `mu`, `sigma` | **One-step-ahead** predictions — from the GP fitted on *previous* iterations only, recorded *before* the config runs. Genuine out-of-sample. |
| `energy_j` | What we then actually measured |
| `best_energy_so_far` | Convergence curve |
| `delta_energy` | ΔE, the per-iteration improvement |
| `on_pareto_front` | Whether it survived non-dominated sorting |
| `calibration_energy_j` | `C_bo` — the energy **the search itself consumed**. Nice honesty metric: optimization is not free. |

MAE and NPE come from the `(mu, energy_j)` pairs, **excluding seeds** (no surrogate
existed yet). Per amendment A-40, report **LOOCV** as the primary reliability figure and
one-step-ahead as a deliberately pessimistic bound — EI *by design* probes where the GP
is least certain, so scoring it on its own probes measures exploration, not accuracy.
MobileLLM-Flash (§4.3) is the precedent for the LOOCV-R² framing.

---

## 6. Chapter 3 edit checklist

- [ ] **Table 3.3** says *RTX 3060 (12 GB)* with CPU/RAM as `[fill in]`. Real host is
      MX330 2 GB / i5-1135G7 / 16 GB. Fix, or declare two machines and state which
      produced which table.
- [ ] **Research Instrument → Hardware Profiling.** "GPU power draw is measured through
      NVML… integrated over the inference duration" → make conditional on Mode A; add
      the `NOT_SUPPORTED` fact and the three-mode ladder.
- [ ] **Data Analysis → energy computation.** Keep `E = Σ(P_t − P_idle)·Δt` for Mode A;
      add the §1 formula, its six assumptions, and the §3.1 cancellation derivation.
- [ ] **Definition of Terms.** `P_idle` currently means idle GPU *power*. In Mode C the
      analogue is idle GPU *duty cycle*. Note explicitly that idle CPU is recorded but
      **not** subtracted (the CPU term is already process-exclusive).
- [ ] **Statistical Treatment / Table 3.5.** Add bootstrap 95% CI rows for EIR/SRR/QRR.
      Relabel EIR as *estimated* EIR. Add the MAPE ≤ 15% acceptance criterion with its
      §4.1 literature justification.
- [ ] **Appendix B Table B.3** has a *"CodeCarbon Validation"* column. On this host
      CodeCarbon has no NVML power and no RAPL access, so it falls back to TDP
      estimation — the **same class of estimate** as ours. It cannot validate us.
      Relabel to "second estimator (agreement check)" or drop it.
- [ ] **RQ1/RQ2/RQ3 wording.** "Energy consumption, measured in Joules and Joules per
      token" needs the *estimated* qualifier, or must cite the §3.2 validation.
- [ ] **EI formula — actual sign error, see §7.2.**
- [ ] Paste-ready limitation paragraph already drafted at
      [`docs/ENERGY_MODES.md`](ENERGY_MODES.md) line 284.

---

## 7. Open issues needing a team decision

### 7.1 🔴 BLOCKER — the feasible configuration space is empty

`bopis_profile.js` reports:

```
n_feasible: 0
n_rejected: 64   (all by rule HW-P0)
```

**Zero configurations can execute.** Energy methodology is moot if nothing runs.

But note *why*: every rejection cites `exceeds 1.91 GiB available system RAM`, and the
machine has **15.78 GiB total** — only 2.05 GiB was free at profile time. **We profiled
a loaded laptop.** On a fresh boot, Q4_K_M at `g=0` needs 4.33 GiB and Q8_0 needs
7.42 GiB; both fit comfortably.

**Action: re-run `python -m bopis profile` on a freshly booted machine with everything
closed.** This likely recovers the CPU-only configurations at zero cost.

What it will **not** recover: `g=14` needs 2.03–3.64 GiB VRAM against 1.94 GiB
available on a 2 GB card. Q4_K_M misses by ~90 MiB. Those stay rejected permanently,
which means **GPU layer offloading collapses to a single level (`g = 0`)** — one of our
five configuration parameters loses its dimension. Consequences:

- Table 3.2 and the RQ3.5 per-variant analysis both need amending
- Inference becomes CPU-only, which is *why* the CPU term dominates (§3.1)
- Raise with the adviser explicitly — do not bury this

Options if a fuller space is required: swap to a ~1–3B model (Qwen2.5-1.5B,
Llama-3.2-1B/3B, Phi-3-mini) at Q4_K_M/Q8_0 — which also aligns us with the models in
Zähl & Hennig (§4.2) — or run on a machine with a larger GPU.

### 7.2 🔴 The EI formula in the manuscript has an inverted sign

The manuscript prints:

```
EI(x) = (μ(x) − f(x⁺)) · Φ(Z)  +  σ(x) · φ(Z)
Z     = (μ(x) − f(x⁺)) / σ(x)
```

Because BOPIS **minimizes** energy, improvement is `f(x⁺) − μ(x)`. As printed, the
expression **rewards configurations predicted to consume more energy**, and a search
driven by it would converge on the worst configuration in the space.

The code is correct — [`bopis/acquisition.py`](../bopis/acquisition.py) implements the
minimization form under amendment A-2, and `tests/test_acquisition.py` asserts EI
prefers lower μ. **The manuscript text still needs fixing.** This is exactly the kind
of thing a panel catches.

### 7.3 🟡 `bopis/stats.py` has no bootstrap

Friedman and Nemenyi are implemented; there is no CI machinery. §3.3 needs it.
~40 lines, standard library only.

### 7.4 🟡 RAPL decision

Would upgrade the CPU term from estimated to **measured**, which given §7.1 is most of
our energy. Costs: admin kernel driver, breaks standard-library-only. **Needs a
feasibility spike before we commit to it in Chapter 3.**

---

## 8. Recommended order of work

1. **Re-profile on a clean boot** (§7.1) — free, and determines whether anything else
   matters.
2. **Fix the EI sign in the manuscript** (§7.2) — free, and it is a real error.
3. **Add the bootstrap CI** (§7.3) — ~40 lines; it is the literal answer to
   "what is your confidence level?"
4. **Write the §3.1 cancellation derivation into Chapter 3** — free, and it is our
   strongest card.
5. **Pull the §4 papers and verify every figure**; add to RRL.
6. **Run the §3.2 Route 3 plausibility check** against Zähl & Hennig's J/token band — free.
7. **Spike RAPL** (§7.4) — only if the panel insists on absolute joules.

---

## Sources

- [Energy Efficiency of Locally Deployed LLMs: A Preliminary Quantitative GPU Power Benchmark on Consumer Hardware (arXiv:2608.00008)](https://arxiv.org/abs/2608.00008)
- [EnerInfer: Energy-Aware On-Device LLM Inference (arXiv:2606.23001)](https://arxiv.org/abs/2606.23001)
- [Multi-Level Modeling of LLM Inference Latency and Energy (arXiv:2608.06723)](https://arxiv.org/html/2608.06723)
- [EnergyLens: Predictive Energy-Aware Exploration for Multi-GPU LLM Inference (arXiv:2605.14249)](https://arxiv.org/abs/2605.14249)
- [LLMCO2: Accurate Carbon Footprint Prediction for LLM Inferences (arXiv:2410.02950)](https://arxiv.org/pdf/2410.02950)
- [CO2-Meter: Carbon Footprint Estimator for LLMs on Edge Devices (arXiv:2511.08575)](https://arxiv.org/pdf/2511.08575)
- [PIE-P: Fine-Grained Energy Prediction for Parallelized LLM Inference (arXiv:2512.12801)](https://arxiv.org/pdf/2512.12801)
- [Where Do the Joules Go? Diagnosing Inference Energy Consumption (arXiv:2601.22076)](https://arxiv.org/pdf/2601.22076)
- [Scaling Laws for Energy Efficiency of Local LLMs (arXiv:2512.16531)](https://arxiv.org/pdf/2512.16531)
- [Beyond Test-Time Compute Strategies: Advocating Energy-per-Token (arXiv:2603.20224)](https://arxiv.org/pdf/2603.20224)
- [TokenPowerBench: Benchmarking the Power Consumption of LLM Inference (AAAI)](https://ojs.aaai.org/index.php/AAAI/article/view/40535/44496)
- [Understanding the Energy Scaling of LLM Inference Across Context Lengths (arXiv:2608.25096)](https://arxiv.org/html/2608.25096)
- [Energy-Aware Computing in the Year 2026 (arXiv:2605.24569)](https://arxiv.org/pdf/2605.24569)
- [MobileLLM-Flash: Latency-Guided On-Device LLM Design (arXiv:2603.15954)](https://arxiv.org/pdf/2603.15954)
- [SCOOT: SLO-Oriented Performance Tuning for LLM Inference Engines (arXiv:2408.04323)](https://arxiv.org/pdf/2408.04323)
- [LLM-Guided Runtime Parameter Optimization for Energy-Efficient Model Inference (arXiv:2604.27032)](https://arxiv.org/html/2604.27032)
- [Pimp My LLM: Leveraging Variability Modeling to Tune Inference Hyperparameters (arXiv:2602.17697)](https://arxiv.org/pdf/2602.17697)
- [A Comparison of High-Level Full-System Power Models — Rivoire et al., HotPower 2008](https://www.usenix.org/legacy/events/hotpower08/tech/full_papers/rivoire/rivoire.pdf)
