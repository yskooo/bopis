# BOPIS — Status and Action Items

**As of:** 2026-09-18
**Branch:** `aaron-g2-changelog`
**Scope:** what changed in this working session, what is verified, what is still
open, and the framing for the "is this a machine learning thesis?" question.

Every number in this document was produced by running the code in this repo on
this laptop, not estimated. Commands are given so each one can be reproduced.

---

## 1. Verified state of the tool

| Component | State | Evidence |
| :--- | :--- | :--- |
| BO loop, GP surrogate, EI, Pareto, selection ladder | Implemented | `bopis/optimizer.py`, `gp.py`, `acquisition.py`, `pareto.py` |
| Bootstrap CI for EIR/SRR/QRR | Implemented | `bopis/stats.py:385`, `TestBootstrapCI` |
| EI minimisation sign | Correct in code **and** manuscript | `acquisition.py:72`, amendment A-2 |
| HW-P0 / HW-B0 feasibility guards | Implemented, now exercised | `hardware.py`, table in §4 below |
| Energy cost in PHP | Implemented | `metrics.py:447`, ₱14.35/kWh default |
| Task classification (Stage 1), rule-based | **New this session** | `bopis/classify.py`, 49.7% on Dolly |
| Task classification, supervised/trained | **New this session** | `bopis/classify_trained.py`, **69.7%** held-out |
| llama.cpp backend | Coded and unit-tested, **never executed** | `backends/llama_server.py` |
| Real energy measurement | **None.** Mode C estimate only | MX330 exposes no power telemetry |
| Full test suite | **467 tests, OK, 243 s** | `python -m unittest discover -s tests -t .` |

### The one framing point to be clear about

**The pipeline runs end to end, but only against the analytic simulator.** All 11
run directories under `runs/` carry `"backend": "sim"`, whose manifest says:
*"Outputs are computed, not measured. No energy figure from this backend may be
reported as an empirical result."* There is no GGUF file and no `llama-server`
binary in the working tree.

So the defensible claim is **"the instrument is built and verified,"** not "we
have results."

---

## 2. What changed this session

### 2.1 New code

| File | Purpose |
| :--- | :--- |
| `bopis/classify.py` | Rule-based prompt → Dolly-category classifier (Stage 1 at deployment time). 42 rules, stdlib only. Exports its rule table to JS. |
| `bopis/classify_trained.py` | **Supervised** Naive Bayes classifier fitted to Dolly's labels, with stratified split and validation-selected smoothing. 69.7% held-out vs 48.2% rules. |
| `tests/test_classify.py` | 31 tests for the rule classifier. |
| `tests/test_classify_trained.py` | 43 tests for the learned classifier. |
| `tests/__init__.py` | **Bug fix.** See §2.3. |
| `demo.ps1` | One-command demo launcher. |
| `bopis_rules.js` | Generated. The rule table the browser UI reads. |

### 2.2 Modified

- **`bopis.html`** — shows a Stage 1 task badge on each submitted prompt
  (detected category, quality-sensitivity tier, confidence, seed prior, which
  classifier produced it, and the accuracy caveat). Neither classifier is
  duplicated by hand in JavaScript: both are authored and fitted in Python and
  exported (`bopis_rules.js`, `bopis_model.js`); the browser reimplements only
  the scoring arithmetic. Both ports were cross-checked against Python:
  - rules — 16/16 prompts agree;
  - Naive Bayes — **155/155 agree with a maximum confidence delta of 0.0**,
    i.e. bitwise-identical posteriors, checked on 150 real Dolly rows plus
    edge cases (empty prompt, all-out-of-vocabulary input).

  The badge prefers the trained model and falls back to the rules if
  `bopis_model.js` is missing.
- **`docs/CHATBOT_INTEGRATION.md`** — quick start, the classifier section, and
  the limits of "adaptive" (§3.2 below).
- **`docs/RRL.md`** — five citation records corrected. Two of the cited papers do
  not exist. See §5.
- **`docs/CHANGELOG_G2_EDITS.md`** — citation audit recorded; test-suite record
  corrected.
- **`docs/ENERGY_ESTIMATOR_AND_ML_BRIEF.md`** — new §0.0 status block; the
  EnerInfer accuracy figures removed as a misread; §7.2 and §7.3 marked resolved;
  §7.1 confirmed still open with live numbers.

### 2.3 Two real bugs found and fixed

**a) The documented test command did not work.**
`python -m unittest tests.test_stats` failed for all 14 modules with
`ModuleNotFoundError`. Cause: an unrelated package named `tests` is installed in
`site-packages`, and because the repo's `tests/` had no `__init__.py` it was only
a *namespace* package candidate — so `import tests` resolved to site-packages.
Fixed by adding `tests/__init__.py`. The changelog's claimed "436 tests, 3
skipped, 127 s" was not reproducible; the real figures are **467 tests, 0
skipped**, in 243–455 s depending on machine load.

**b) The classifier reported a contradiction on its fallback path.**
A prompt matching no rule returned `confidence = 0.0` *and* `is_confident =
True`, because the synthetic `{fallback: 1.0, rest: 0.0}` distribution gives a
margin of 1.0 over the runner-up. Found by diffing the Python against the
JavaScript port. Fixed with an explicit `fell_back` flag plus a regression test.

---

## 3. Stage 1: task classification

### 3.1 Measured accuracy

Against all 15,011 human-labelled Dolly rows
(`python -m bopis.classify --data-dir data`):

| Basis | Accuracy |
| :--- | ---: |
| Exact 8-way category | **49.7%** |
| With `open_qa`/`general_qa` merged | 64.3% |
| Quality-sensitivity tier (High/Medium/Low) | **67.5%** |

Per-category recall is bimodal: `closed_qa` 88.3% and `open_qa` 82.0%, against
`summarization` 19.5% and `brainstorming` 31.2%.

**Quote the sensitivity-tier figure**, because that is the property that actually
drives the prior's shape. Be ready to explain the ceiling: the largest single
error class is `general_qa` predicted as `open_qa` (1,798 rows, 12% of the
corpus). Dolly separates those two by whether answering needs a specific
retrievable fact or general world knowledge — a property of the **answer**, not
of the instruction's surface form. No lexical rule can recover it, which is why
`general_qa` carries no positive rules and is reachable only as the fallback.

**Why 49.7% is survivable:** the prior weights *only the 10 seed draws* of the
30-evaluation budget. The 20 BO-guided steps follow Expected Improvement over the
GP, and `x*` is selected by Pareto dominance plus the SRR/QRR thresholds. A
misclassification makes the search slightly less sample-efficient; **it cannot
make `x*` wrong.** Prefer `soft=True` in `prior_for_prompt()` so a low-margin
prediction degrades toward the blended prior instead of committing to a
coin-flip label.

### 3.2 What "adaptive" can and cannot mean — read before claiming it

Only `t` is a per-request parameter (`n_predict`). `p`, `g`, `c` and `b` are
`llama-server` **launch flags** (`backends/llama_server.py:151-159`). A single
running server therefore **cannot switch precision or GPU-layer count per
prompt.** Per-prompt adaptation over the full `x = (t, b, p, g, c)` tuple would
need one server process per configuration, or a model reload between prompts.

Today BOPIS selects **one `x*` for the workload.** The classifier informs seeding
and supplies the per-task `Q_min(task)` threshold required by amendment A-26. Do
not describe the current system as switching configuration per prompt — that is
the one claim in this area a panel can falsify by reading the code.

---

## 4. Model selection — the laptop can do far more with a smaller model

Computed by running this laptop's live profile through the project's own
HW-P0/HW-B0 guards at `ctx_size = 2048`, with Table H1 opened up so the
model-size guard is the only filter. Weight sizes in GiB.

### Feasible configurations after a clean boot (~13 GiB free RAM)

| Model | Q4_K_M | Q8_0 | F16 | F32 | Feasible | Variants | GPU offload |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| Qwen2.5-0.5B-Instruct | 0.28 | 0.49 | 0.92 | 1.84 | **744**/768 | all four | g = 14, 28, all |
| TinyLlama-1.1B-Chat | 0.62 | 1.09 | 2.05 | 4.10 | 528/768 | all four | g = 14, 28, all |
| Llama-3.2-1B-Instruct | 0.69 | 1.22 | 2.30 | 4.60 | 480/768 | all four | g = 14, 28, all |
| **Qwen2.5-1.5B-Instruct** | 0.87 | 1.53 | 2.88 | 5.75 | **504**/768 | **all four** | **g = 14, 28, all** |
| SmolLM2-1.7B-Instruct | 0.96 | 1.69 | 3.19 | 6.37 | 288/768 | all four | g = 14, 28, all |
| Gemma-2-2b-it | 1.47 | 2.59 | 4.87 | 9.74 | 300/768 | all four | g = 14, 28, all |
| Llama-3.2-3B-Instruct | 1.81 | 3.18 | 5.98 | 11.97 | 228/768 | all four | g = 14 only |
| Phi-3-mini-4k-instruct | 2.15 | 3.78 | 7.12 | 14.23 | 144/768 | F16/Q8_0/Q4_K_M | g = 14 only |
| Mistral-7B-Instruct | 4.08 | 7.17 | 13.50 | 27.00 | 96/768 | Q8_0/Q4_K_M | **NONE — CPU-only** |

### Feasible configurations right now, without rebooting (1.49 GiB free)

| Model | Feasible | Variants | GPU offload |
| :--- | ---: | :--- | :--- |
| Qwen2.5-0.5B-Instruct | **696**/768 | all four | g = 14, 28, all |
| TinyLlama-1.1B-Chat | 432/768 | F16/Q8_0/Q4_K_M | g = 14, 28, all |
| Llama-3.2-1B-Instruct | 372/768 | Q8_0/Q4_K_M | g = 14, 28, all |
| Qwen2.5-1.5B-Instruct | 360/768 | F16/Q8_0/Q4_K_M | g = 14, 28, all |
| Llama-3.2-3B-Instruct | 36/768 | Q4_K_M | g = 14 |
| Mistral-7B-Instruct | **0**/768 | — | NONE |

### 4a. Measured: GPU offload on this laptop makes generation SLOWER

The feasibility table above says what *fits*. This says what is actually *fast*,
which turned out to be the opposite of the assumption baked into the manuscript.

Measured 2026-09-18 with a real `llama-server` (build b11026, Vulkan backend),
Qwen2.5-1.5B-Instruct Q4_K_M, `ctx 2048`, `threads 4`, one request of 64 tokens.
Reproduce with `tools/bench.ps1`.

| Target | GPU layers | Prefill | Decode | vs. CPU |
| :--- | ---: | ---: | ---: | ---: |
| **CPU only** | 0 | 2.18 s | **11.52 tok/s** | — |
| Intel Iris Xe (Vulkan0) | 28 (all) | 3.04 s | 7.55 tok/s | 0.66x |
| NVIDIA MX330 (Vulkan1) | 14 (half) | 2.15 s | 4.67 tok/s | 0.41x |
| NVIDIA MX330 (Vulkan1) | 28 (all) | 1.89 s | 3.28 tok/s | **0.28x** |

**Moving layers onto the MX330 monotonically degrades throughput.** Full offload
is 3.5x *slower* than not using the GPU at all. The MX330 is a GP108 part with a
64-bit memory bus; it has less bandwidth than the CPU's path to system RAM, and
decode is memory-bandwidth-bound.

Consequences worth stating to the adviser:

- `g` is not a *beneficial* dimension on this host, it is a **harmful** one. BOPIS
  optimizing for energy should discover `g = 0` and be right to. That is a
  legitimate finding — "the optimizer correctly rejects the GPU" — but it is a
  very different story from "we optimize GPU offload," and the manuscript
  currently implies the latter.
- The host also has a **second, faster GPU** that the study never considered: the
  Intel Iris Xe iGPU, which Vulkan reports with **7.4 GiB available** against the
  MX330's 1.9 GiB. It is still slower than CPU here, but it dominates the MX330
  on both capacity and speed. NVML cannot see it, which is why `bopis/hardware.py`
  profiling never noticed it.
- **Caveat:** these figures are from the **Vulkan** backend. The manuscript
  assumes CUDA. A CUDA build may narrow the gap, and this test did not measure
  it — the CUDA 12.4 binaries want a newer driver than this host's 528.96.
  Do not report the Vulkan numbers as CUDA numbers. Re-measure before drawing a
  final conclusion about `g`.
- Free RAM was 2.2–3.0 GiB during these runs, so absolute throughput is
  depressed. The *ordering* is robust (the effect is 3.5x), but re-run after a
  clean boot before quoting absolute tok/s.

### What this means

This is the most consequential finding of the session. Switching from Mistral 7B
to **Qwen2.5-1.5B-Instruct** does not merely make the demo faster — it makes the
manuscript's *stated experimental design actually executable*:

- **All four precision variants become feasible**, including F32. Amendment A-38
  and the RQ 5.5 rewording had to soften the variant enumeration to
  "the feasible variants" precisely because F32 Mistral 7B (27 GiB) can never
  run here. On Qwen2.5-1.5B, F32 is 5.75 GiB and fits.
- **The GPU-layer dimension `g` is restored.** On Mistral 7B, `g` collapses to a
  single level (`g = 0`), so one of the five configuration parameters has no
  range — a real weakness the brief flags in §7.1. On Qwen2.5-1.5B, `g = 14`,
  `g = 28` and full offload are all feasible.
- It aligns the study with the 1B–7B consumer-hardware range benchmarked by
  Zähl & Hennig (2026), which is the closest published work and the source of
  the J/token plausibility band.

**Qwen2.5-0.5B-Instruct is the no-reboot fallback**: 696/768 configurations with
all four variants even on the currently loaded machine. Useful insurance if the
reboot does not free as much RAM as expected.

### Caveats on the table

- Architecture parameters (layer count, KV heads, head dim) were entered from
  published model cards. **Confirm them against `llama-server`'s load log**
  before quoting the table in the manuscript; the weight column depends only on
  parameter count and is reliable, but the KV-cache term depends on the
  attention shape.
- `--total-layers` defaults to 32 (sized for Mistral 7B). Qwen2.5-1.5B has 28
  layers, Llama-3.2-1B has 16. Pass the right value or `g` will be misresolved.
- Switching models is a **disclosed substitution**, not a silent one. The
  manuscript says Mistral 7B.

---

## 5. Citation audit — two cited papers do not exist

Corrected in `docs/RRL.md`, which now carries inline verification comments.

| Record as cited | Finding |
| :--- | :--- |
| Pham, Qian, Wang & Yu (2020), *Problems and opportunities in neural network robustness and reproducibility*, arXiv:2206.04236 | **Does not exist.** No match on arXiv or Scholar; a 2020 paper cannot hold a 2022 arXiv ID. Replaced with Goldberg (1991) on floating-point non-associativity. |
| Xu, Z., et al. (2023), *Evaluating quantization-induced energy reduction in local LLM deployment* | **Not found**, and the entry carried no arXiv ID. Replaced with Shi & Ding (2025), arXiv:2508.16712. |
| Efron & Tibshirani (1993), DOI `10.1017/CBO9780511802843` | Wrong DOI — it resolves to Davison & Hinkley (1997). Cite by ISBN. |
| Lim, Rawson & Ballew (2014), EuroSys '14 | Wrong authors, year and proceedings. The paper is Colmant, Kurpicz, Felber, Huertas, Rouvoy & Sobe, EuroSys **'15**. |
| Frantar & Alistarh, SparseGPT, for a quantization claim | SparseGPT is a **pruning** paper. Replaced with GPTQ (Frantar, Ashkboos, Hoefler & Alistarh, ICLR 2023). |

Also obsolete: the summary table's "INT8 Benefits" role. Amendment A-23 replaced
FP16/INT8 with F32/F16/Q8_0/Q4_K_M; INT8 is not in this study.

**Separately:** the Table T1 draft circulating with two columns
(`P(FP16)`/`P(INT8)` = 0.53/0.47 for Open QA, etc.) does **not** match the code.
`bopis/tasks.py` implements the three-column F32/F16/INT8 form
(`open_qa` = 0.15/0.45/0.40), expanded to four columns by A-4. That draft also
cites the non-existent Xu et al. (2023). Do not circulate it.

Verified correct and untouched: Jones et al. (1998), Gerganov (2023), Zitzler &
Thiele (1998), Emmerich et al. (2005), Arlot & Celisse (2010), Agrawal et al.
(OSDI '24), Zhong et al. (DistServe), Stojkovic et al. (2408.00741), Rotem et al.
(2012), Narayanan et al. (SC21). All 18 arXiv IDs in the brief's §4 resolve to
real papers with matching titles.

---

## 6. "If we don't train on our dataset, how is this a machine learning thesis?"

This is the right question to have prepared, and the answer is stronger than it
feels. The premise to reject is that "machine learning" means "fine-tune a
neural network."

### 6.1 BOPIS does train a model — just not the language model

The Gaussian Process surrogate is a **supervised regression model that is fitted
to data**:

- `GaussianProcess.fit(x, y)` — `bopis/gp.py:334`. It has a `.fit()` method with
  the same contract as scikit-learn's. More usefully,
  `tests/test_gp.py::TestAgainstSklearn::test_posterior_and_likelihood_match`
  checks this implementation's posterior mean, variance and log marginal
  likelihood against `sklearn.gaussian_process`'s reference implementation — and
  it **runs and passes on this machine** (sklearn 1.7.0, numpy 2.0.2), it is not
  a skipped test. That is a strong answer to "did you actually implement a
  model, or just call a library?": the model is written from scratch in the
  standard library *and* verified equivalent to the standard library-grade
  reference.
- Three hyperparameters `θ = {σ_f², ℓ, σ_n²}` are **learned from the
  observations** by maximizing the log marginal likelihood (coarse log-space grid
  then multi-start Nelder-Mead, `gp.py:454-505`). Nothing about them is set by
  hand. A real fitted example from `runs/20260909T172504Z/metrics.json`:
  `signal_variance = 3.99`, `length_scale = 0.937`, `noise_variance = 0.00157`.
- It is **validated the way any ML model is validated**: leave-one-out
  cross-validation, R², MAE, NPE, and uncertainty calibration (UCR). Measured on
  that run: **LOO R² = 0.980, LOO NPE = 3.77%, UCR = 0.967**.

Training set size is 30. That is *small-n regression*, not an absence of
learning — and it is precisely why a Gaussian Process is the correct model class
rather than a neural network: GPs are the standard tool when each label costs
minutes of computation and you need calibrated uncertainty from a handful of
points.

### 6.2 Bayesian Optimization is a recognized ML subfield

BO is **active learning / sequential experimental design**. It is published at
NeurIPS and ICML continuously, and the foundational citation already in the RRL
(Jones, Schonlau & Welch, 1998) is a core optimization-and-surrogate-modelling
paper. The methodological content of this thesis is:

1. a surrogate model fitted to data, with learned hyperparameters and
   quantified predictive uncertainty (the GP);
2. a decision rule that uses that model's uncertainty to choose the next
   experiment (Expected Improvement);
3. multi-objective selection under constraints (Pareto dominance + SRR/QRR);
4. a validation protocol with cross-validation and bootstrap confidence
   intervals.

That is a complete machine learning contribution. What is *not* being learned is
the LLM — and correctly so. **The LLM is the expensive black-box objective
function being optimized, not the model being fitted.** Say it that way.

### 6.3 The honest one-sentence version

> "We do not train a language model. We fit a Gaussian Process surrogate to 30
> measured configurations — hyperparameters learned by marginal-likelihood
> maximization, validated at leave-one-out R² = 0.98 — and use its calibrated
> uncertainty to decide which configuration to measure next. The language model
> is the objective function, not the learner."

### 6.4 What kind of ML? — supervised, unsupervised, or reinforcement?

Expect this question. The short answer is **supervised learning, specifically
active learning.** Component by component:

| Component | Learning type | Notes |
| :--- | :--- | :--- |
| GP surrogate (`bopis/gp.py`) | **Supervised regression** | Features `x = (t, b, p, g, c)`, continuous target `y` = measured energy. |
| The BO loop (`bopis/optimizer.py`) | **Active learning** | Also called sequential experimental design: supervised learning in which the learner *chooses its own next training point* rather than receiving a fixed dataset. |
| Task classifier (`bopis/classify_trained.py`) | **Supervised classification** | Multi-class text classification, 8 classes, trained on Dolly's human labels. |
| Pareto sorting (`bopis/pareto.py`) | **Not learning at all** | A deterministic multi-objective selection algorithm. |

**Nothing in BOPIS is unsupervised.** Be careful not to let Pareto
non-dominated sorting be described as unsupervised learning — it does not learn
anything, it sorts. Misnaming it is an easy way to lose credibility on an
otherwise correct answer.

**Is it reinforcement learning? No — and the precise reason is worth knowing.**
Bayesian Optimization is the **stateless limit** of sequential decision-making:
formally a structured (or continuum) bandit with a Gaussian Process prior, the
family that includes GP-UCB and GP-EI. It shares one thing with RL — the
exploration/exploitation trade-off, which is exactly what Expected Improvement
balances — but the defining RL machinery is absent:

- **No state.** The configuration space does not change as a result of
  evaluating a configuration.
- **No state transitions.** Evaluating `x₁` does not move the system into a
  different situation for `x₂`; each evaluation is an independent measurement of
  a fixed unknown function.
- **No policy is learned.** The acquisition function is a fixed analytic rule
  applied to a learned model, not a parameterized policy improved over time.
- **No temporal credit assignment or discounting.** There is no reward sequence
  to attribute across time steps.

Bandits are sometimes filed under RL as the no-state special case, so the most
defensible phrasing is: *"Supervised at the core — a GP regression surrogate —
inside an active-learning loop. If you want to locate it relative to RL, it is
bandit-adjacent: it shares the explore/exploit dilemma but has no state, no
transitions and no learned policy."*

### 6.5 There is now a model trained on Dolly — here are its numbers

Implemented this session: `bopis/classify_trained.py`, a **multinomial Naive
Bayes** classifier fitted to Dolly's human labels, with a proper protocol.

```powershell
python -m bopis.classify_trained --data-dir data
```

**Protocol.** Stratified 70/15/15 train/validation/test split (seed 20260101),
stratified by category because Dolly is unbalanced (`open_qa` 3,742 rows against
`creative_writing` 709). 10,507 train / 2,251 validation / 2,253 test. The single
hyperparameter `alpha` (Lidstone smoothing) is selected **on validation only**;
the test set is touched once. 8,081 features after pruning at `min_df = 2`.

**Result on the held-out test set:**

| Basis | Rule baseline | Learned | Delta |
| :--- | ---: | ---: | ---: |
| Exact 8-way | 48.2% | **69.7%** | **+21.4 pts** |
| `open_qa`/`general_qa` merged | 63.6% | **78.6%** | +15.0 pts |
| Quality-sensitivity tier | 67.1% | **79.2%** | +12.1 pts |

Train-set accuracy is 81.5%, so there is a ~12-point generalization gap —
moderate and expected for Naive Bayes on 8k features; worth stating rather than
hiding.

**Where the gain comes from:** `classification` 33.3% → 91.9%,
`general_qa` 11.3% → 49.1%, `summarization` 20.2% → 58.4%,
`information_extraction` 36.3% → 58.4%. Notably the learned model is *worse* on
two categories — `closed_qa` 89.1% → 81.2% and `open_qa` 78.5% → 77.0% — where
the rules' structural context gate is genuinely strong. That is an honest and
interesting finding, not a defect: it suggests a hybrid would beat either.

**Fairness caveat to disclose:** the rule weights in `bopis/classify.py` were
hand-tuned while observing whole-corpus accuracy, so the rule baseline has
effectively seen the test set. Its figure is optimistic and the learned model's
margin is therefore a **lower bound**.

**The residual error is still the intrinsic one.** The dominant confusion remains
`general_qa` ↔ `open_qa` (132 + 68 = 200 of 683 test errors). Both classifiers
fail there, which supports the reading that the boundary is not recoverable from
the instruction text rather than that either model is weak.

**Features used** (all discrete, fed to one multinomial model): word unigrams
from the instruction; the leading word as its own feature (Dolly annotators
worked from per-category prompts, so the leading imperative is unusually
diagnostic); bigrams over the first three tokens only (separates "what are some"
from "what are the"); and a context-present flag. The context passage
contributes **no** word features — it averages ~300 tokens against ~14 for the
instruction and would swamp the task-type signal with topical vocabulary.

### 6.7 Where training does *not* belong

For completeness, because the question tends to come back:

- **Not the LLM.** Fine-tuning Mistral 7B (or any of the §4 candidates) on Dolly
  would change the objective function mid-study and invalidate every energy
  comparison, because the unoptimized baseline and the BOPIS winner would no
  longer be running the same model. The LLM must stay fixed for EIR/SRR/QRR to
  mean anything.
- **Not the energy estimator.** The Mode C load-line coefficients are vendor
  power budgets, not fitted parameters. Fitting them would require measured
  ground-truth joules, which is exactly what this host cannot provide — that is
  the RAPL question (A-7), not a training question.
- **Not the Pareto stage.** Non-dominated sorting is exact and deterministic.
  There is nothing to learn.

---

## 7. Action items

Ordered by whether they block a demonstration.

### Blocking

| # | Item | Owner | Notes |
| :--- | :--- | :--- | :--- |
| B-1 | **Reboot, then re-run `python -m bopis profile --model-aware`** | you | Still `n_feasible: 0` — 1.49 GiB free of 15.78. Free, and nothing else matters until it is done. |
| B-2 | **Done.** `llama-server` installed | — | Build `b11026`, both `tools/vulkan/` and `tools/cpu/`. Vulkan chosen because the CUDA 12.4 binaries want a newer driver than this host's 528.96. Verified running. |
| B-3 | **Done.** Q4_K_M GGUF downloaded | — | `models/qwen2.5-1.5b-instruct-q4_k_m.gguf`, 1.12 GB, verified `GGUF` header and loads in ~10 s. Still to fetch if you want the full `p` sweep: `q8_0` (1.89 GB) and `fp16` (3.56 GB). |
| B-5 | Re-measure GPU offload with a **CUDA** build | you | §4a is Vulkan-only. Needs either a driver update past 528.96 or the bundled `cudart` redistributable. This decides whether `g` is a real dimension. |
| B-4 | **Decide and disclose the model substitution** | team + adviser | §4. This is a thesis-scope decision, not a technical one. |

### High value, not blocking

| # | Item | Owner | Notes |
| :--- | :--- | :--- | :--- |
| A-1 | Grep `docs/THESIS_WRITING2_G2.md` for the stale FP16/INT8 prior table and the Xu et al. (2023) citation | you | §5. The corrections landed in `RRL.md`, not in the manuscript. |
| A-2 | Run a real study with `--backend llama-server --model-aware` and produce the first non-sim `selection.csv` | you | Everything downstream (`serve`, the dashboard, all reported metrics) is gated on this. |
| A-10 | **BERTScore is implemented but never invoked** — wire it up | open | Found 2026-09-18. `grep get_scorer` outside `bopis/quality/` returns nothing. On a real run `quality_f1` stays `None`, so **QRR cannot be computed** and the quality column of the BOPIS-vs-random comparison would be empty. The 98.6% QRR now on the dashboard is the *simulator's* analytic value (`scorer = analytic_model`), not BERTScore. Needs two changes: (1) persist `GenResult.text` into the per-prompt validation artifacts — no run artifact stores the generations today, so there is nothing to score; (2) add a `bopis score <run-dir>` command that pairs those generations with the Dolly references in `dataset/sample_500.csv`, calls `get_scorer("bertscore")`, and writes `quality_f1` back with `scorer = bertscore`. Keep it a separate offline command: amendment A-6 and `tests/test_provenance.py` forbid importing torch in the measurement path. **This blocks any real quality claim.** See `UI_GUIDE.md` §4.1a. |
| A-3 | A-36 repeatability run: one configuration, five repeats, CV of measured energy | you | Still has no manuscript text. Last open leg of A-36. |
| A-4 | **Done.** Supervised classifier trained on Dolly | — | §6.5. 69.7% held-out vs 48.2% rules. `bopis/classify_trained.py`. |
| A-8 | **Done.** UI badge uses the trained model | — | `bopis_model.js` (855 KiB) is exported by `--write-js` and the badge prefers it, falling back to the rules when absent. The badge says which one produced the answer. |
| A-9 | Consider a hybrid classifier | optional | The rules beat the learned model on `closed_qa` (89.1% vs 81.2%) and `open_qa` (78.5% vs 77.0%) via the structural context gate. A gate-then-model hybrid would likely beat both. |
| A-5 | Verify at least two of the four MAPE figures backing the ≤ 15% threshold | you | The EnerInfer figure was a misread and has been removed; the rest are unverified against full texts. |
| A-6 | Confirm model architecture params in §4 against `llama-server`'s load log | you | Affects only the KV-cache term, not the weight column. |
| A-7 | Decide on RAPL for a measured CPU energy term | team | Brief §7.4. Needs a feasibility spike; breaks stdlib-only. |

### Resolved this session — do not re-raise

- EI sign error in the manuscript (amendment A-2 applied).
- `bopis/stats.py` has no bootstrap (`bootstrap_ci` implemented).
- `unittest discover` failing (`tests/__init__.py` added).
- Two fabricated and three misattributed citations (corrected in `RRL.md`).

---

## 8. Reproducing the numbers in this document

```powershell
# full test suite -> 467 tests, OK
python -m unittest discover -s tests -t .

# rule-based classifier vs Dolly's labels -> 49.7 / 64.3 / 67.5
python -m bopis.classify --data-dir data

# train the supervised classifier -> 69.7% held-out vs 48.2% rule baseline
python -m bopis.classify_trained --data-dir data

# train and persist it
python -m bopis.classify_trained --data-dir data --save task_classifier.json

# classify one prompt, see the prior it selects
python -m bopis.classify --prompt "Summarize this report."

# live feasible space under the model-size guards -> currently 0 of 768
python -m bopis profile --model-aware --ctx-size 2048

# refresh all three generated files the UI reads
python -m bopis profile --model-aware --write-js bopis_profile.js
python -m bopis.classify --write-js bopis_rules.js
python -m bopis.classify_trained --data-dir data --write-js bopis_model.js

# one-command demo, using what is now installed in this repo.
# GpuLayers 0 is deliberate -- see section 4a, the MX330 is 3.5x slower.
.\demo.ps1 -LlamaBinary .\tools\vulkan\llama-server.exe `
           -Model Q4_K_M=.\models\qwen2.5-1.5b-instruct-q4_k_m.gguf `
           -GpuLayers 0 -Threads 4 -MaxTokens 256

# what devices does this host expose?
.\tools\vulkan\llama-server.exe --list-devices

# reproduce the device benchmark in section 4a
.\tools\bench.ps1
```

**Killing a stuck server:** use PowerShell, not `pkill`. `pkill -f llama-server`
from Git Bash does **not** reap Windows processes — during this session it
silently left four orphaned servers holding 5.6 GB, which starved the machine to
0.92 GiB free and invalidated a first run of the §4a benchmark. Use:

```powershell
Get-Process -Name llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
```

The small-model feasibility table in §4 was produced by driving
`bopis.hardware.feasible_space()` with a `ModelSpec` per candidate against the
live host profile. It is not a saved artifact; re-derive it if the host changes.
