# Manuscript amendments required by the artifact

Building BOPIS surfaced places where Chapter 3, as written, cannot be
implemented, is internally contradictory, or would produce an unsound result.
Each entry below states the change, where it applies, and why — so the
manuscript and the code can be brought into agreement rather than diverging
quietly.

Amendments are grouped by how urgent they are. **Blocking** items are ones where
the document currently asserts something false or unachievable. **Definitional**
items fill gaps the document leaves open. **Clarifying** items remove ambiguity
that would otherwise make the study unreproducible.

Every amendment is referenced by its ID in the code, so any claim here can be
traced to the implementation that depends on it.

---

## Blocking

### A-2 — The Expected Improvement equation has an inverted sign

**Where:** Chapter 3, System Architecture, EI equation.

**Currently:**
```
EI(x) = (mu(x) - f(x+)) * Phi(Z) + sigma(x) * phi(Z)
Z     = (mu(x) - f(x+)) / sigma(x)
```

**Should be:**
```
EI(x) = (f(x+) - mu(x)) * Phi(Z) + sigma(x) * phi(Z)
Z     = (f(x+) - mu(x)) / sigma(x)
```

**Why:** BOPIS *minimizes* energy, and `f(x+)` is defined in the text as "the
best energy value observed so far" — that is, the *lowest*. Improvement at a
candidate is therefore `f(x+) - mu(x)`. As printed, the expression is maximized
by configurations predicted to consume *more* energy than the incumbent, so a
search driven by it would converge on the worst configuration in the space.

**Enforced by:** `tests/test_acquisition.py::test_prefers_lower_predicted_energy`,
which fails against the printed formula.

---

### A-6 — "Python standard library only" cannot cover BERTScore

**Where:** Table 3.4 (`Only Python standard library modules are used`); Research
Instrument, Output Quality Measurement. `chapter3.md` states the claim more
strongly still: BERTScore "implemented by the researchers using Python standard
library math operations."

**Should be:** Scope the claim to the measurement core, and declare BERTScore as
a separate evaluation dependency:

> The measurement and optimization core is implemented using only the Python
> standard library (`subprocess`, `ctypes`, `math`, `csv`, `json`, `time`, `os`,
> `threading`). Output quality is scored in a separate offline stage using
> BERTScore F1, which requires a pretrained transformer forward pass and
> therefore depends on `transformers` and `torch`. That stage reads generated
> text from the run's CSV logs and writes quality scores back; it is not part of
> the measurement path.

**Why:** BERTScore requires a BERT forward pass over contextual embeddings.
There is no standard-library implementation of a transformer, and hand-rolling
one would no longer be canonical BERTScore. The claim as written is false.

**Enforced by:** `tests/test_provenance.py::TestStandardLibraryOnly`, which
parses every module's imports and additionally re-imports the core with
`site-packages` stripped from `sys.path`. Exactly one module,
`bopis/quality/bertscore.py`, is exempt.

---

### A-7 — L-BFGS-B is not available in the standard library

**Where:** Chapter 3, GP hyperparameter optimization: "`theta` … optimized using
L-BFGS-B implemented through Python standard library math operations."

**Should be:**

> GP hyperparameters are fitted by maximizing the log marginal likelihood using a
> coarse grid search over log-hyperparameters followed by multi-start
> Nelder–Mead, a derivative-free simplex method implemented directly in the
> standard library. Nelder–Mead replaces L-BFGS-B because the latter is
> gradient-based and no standard-library implementation exists; at three to seven
> hyperparameters the two converge to indistinguishable optima, and the grid
> stage additionally guards against the local optima that make single-start fits
> of GP hyperparameters unreliable.

**Why:** L-BFGS-B needs gradients and a constrained quasi-Newton implementation.
"L-BFGS-B implemented through Python standard library math operations" describes
something that does not exist.

**Verified by:** `tests/test_gp.py::TestAgainstSklearn` — the from-scratch GP
reproduces sklearn's log marginal likelihood to 8 decimal places and its
posterior mean to ~1e-7 on identical hyperparameters.

---

### A-19 — GPU-only energy accounting is invalid at zero GPU offload

**Where:** Chapter 3, Data Analysis, energy computation.

**Add:** an `energy_scope` field recorded per measurement, plus a limitation
paragraph:

> Energy accounting covers GPU board power only. Configurations with `g = 0`
> place no layer on the GPU, so their measured GPU energy approaches idle
> regardless of the work performed on the CPU. Because BOPIS minimizes measured
> energy, such configurations would appear near-free and the search would be
> driven toward them for reasons unrelated to actual efficiency. Study runs
> therefore apply a minimum GPU-offload floor (`--min-gpu-layers`), and every
> record carries `energy_scope` so out-of-scope comparisons are identifiable.

**Why:** This is a soundness defect, not a precision caveat. Without it the
optimizer optimizes the instrument rather than the system.

**Enforced by:** `bopis/hardware.py` (`ENERGY-SCOPE` rejection rule),
`bopis/measure.py` (`EnergyScope`, `PromptMeasurement.scope_valid`), and
`tests/test_hardware.py::TestEnergyScopeFloor`.

---

### A-29 / A-30 — Table H1 permits configurations that cannot physically run

**Where:** Table H1; Research Design, model selection.

**Add** two rules, and reconsider the variant set:

| Rule | Condition | Effect |
|---|---|---|
| `HW-P0` | GPU-resident weights + KV cache must fit available VRAM; total footprint must fit VRAM + system RAM | reject otherwise |
| `HW-B0` | KV cache for `b` concurrent slots must fit VRAM | reject otherwise |

**Why:** Table H1 is model-size-agnostic, which makes it wrong in general.
`HW-P3` permits F32 on any card with ≥ 8 GB VRAM, but Mistral 7B at F32 is
**~27 GiB** and cannot be fully offloaded to a 12 GB card. `HW-P1` permits Q8_0
on a card under 4 GB, but 14 of 32 layers of a 7B Q8_0 model needs **~3.4 GiB**.
Measured on the development machine (MX330, 2 GiB), Table H1 alone admits 64
configurations and the guards reject 32 of them — every configuration with any
GPU offload at all.

**Consequence for the study (A-30):** the four-variant precision analysis that
research question 4.5 depends on is not achievable with a 7B model on a 12 GB
card. Either drop F32 and report three variants, or select a base model whose
F32 variant fits. State the decision and the resulting variant set explicitly.

**Enforced by:** `tests/test_hardware.py::TestModelSizeGuards`.

---

### A-40 — The NPE < 10% reliability criterion is not attainable as defined

**Where:** Chapter 3, Data Analysis, surrogate accuracy; Table F1 GP-validation
row.

**Currently:** MAE and NPE are computed from the GP's predictions for each
configuration "before inference is executed", with "an NPE below 10% treated as
a researcher-defined reliability threshold."

**Should be:** report reliability on **leave-one-out cross-validation**, and
present the one-step-ahead figures separately as a pessimistic bound:

> Surrogate reliability is assessed by leave-one-out cross-validation over all
> evaluated configurations, with hyperparameters held at their fitted values.
> One-step-ahead prediction errors are also reported, but are a systematically
> pessimistic estimate of surrogate fit: the Expected Improvement acquisition
> function deliberately evaluates the configurations about which the surrogate is
> least certain, so scoring it on its own probes measures exploration rather than
> accuracy.

**Why — measured, not assumed.** Across ten seeds on the analytic simulator with
the standard N = 30 budget:

| basis | mean NPE | s.d. | max | clears NPE < 10% |
|---|---|---|---|---|
| one-step-ahead | 23.0% | 6.8 | 32.5% | **0 / 10** |
| leave-one-out | 11.4% | 4.8 | 22.2% | 4 / 10 |

Leave-one-out R² averaged **0.94** (range 0.79–0.99) on the same fitted models,
which is consistent with Wilkins et al. (2024) — cited in the manuscript for
R² > 0.96 on LLM inference energy models. One-step-ahead R² was frequently
negative. The threshold as written would report a well-fitting surrogate as
unreliable in every single run.

**Recommendation:** state the criterion on LOO R² (e.g. ≥ 0.85) alongside LOO
NPE, since R² is the quantity the cited literature reports.

**Enforced by:** `tests/test_e2e_sim.py::test_leave_one_out_fit_is_strong` and
`::test_one_step_ahead_npe_is_pessimistic_by_construction`.

---

## Definitional

### A-1 — Define `t` as maximum generation length, not context window

**Where:** Table 3.2 ("Input token length … Context window size"); Chapter 1
Statement of the Problem; Chapter 2 Table 2.1.

**Should be:** `t` = maximum generated tokens (`n_predict`), values unchanged at
{128, 256, 512, 1024}. Add `--ctx-size 2048` to Table 3.3 as a fixed setting
outside the search space.

**Why:** `--ctx-size` allocates KV cache; it does not drive compute, because
attention cost scales with the *actual* sequence length. As a context size, `t`
would either have almost no energy effect or would act only by truncating long
prompts. That truncation is not neutral: Dolly's median `context` length is
**zero**, but the three context-bearing categories — `closed_qa`,
`information_extraction`, `summarization` — average roughly 300 tokens and reach
~5,900. Those are precisely the categories Table T1 rates most
quality-sensitive, so `t = 128` would collapse their quality for reasons
unrelated to precision, confounding the central comparison.

As a generation cap, `t` is a genuine energy lever, and this is consistent with
Husom et al. (2024), whom the manuscript already cites for the finding that
inference energy is driven primarily by response length rather than prompt
complexity.

---

### A-3 — State N, the total iteration budget

**Where:** Chapter 3, BO procedure Step 6 ("after N total iterations");
Statement of the Problem.

**Should be:** `N = 30` (10 prior-weighted seeds + 20 BO-guided), with random
search given the identical 30-evaluation budget.

**Why:** N is never given a value anywhere in the document. Table B.1's template
has 30 rows and the UI mockup says 34, which is where the ambiguity came from.
Equal budgets are required for SER to compare like with like.

---

### A-4 — Table T1 needs four precision columns

**Where:** Table T1.

**Currently:** `P(FP32)`, `P(FP16)`, `P(INT8)` — three columns for a
four-variant space.

**Should be:**

| Task | F32 | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|---|
| Open QA (Low) | 0.15 | 0.45 | 0.20 | 0.20 |
| Closed QA (High) | 0.35 | 0.45 | 0.15 | 0.05 |
| Summarization (Low) | 0.15 | 0.40 | 0.225 | 0.225 |
| Classification (Low) | 0.10 | 0.35 | 0.275 | 0.275 |
| Creative Writing (Low) | 0.15 | 0.45 | 0.20 | 0.20 |
| Brainstorming (Low) | 0.10 | 0.35 | 0.275 | 0.275 |
| Info Extraction (High) | 0.35 | 0.45 | 0.15 | 0.05 |
| General QA (Medium) | 0.20 | 0.45 | 0.21 | 0.14 |

Derivation: the original INT8 mass is split between Q8_0 and Q4_K_M, with the
share going to the more aggressive Q4_K_M set by quality sensitivity — 0.25 for
High, 0.40 for Medium, 0.50 for Low. F32 and F16 are unchanged; every row sums
to 1.0.

**Why:** As printed, Q4_K_M receives zero prior mass and would never be drawn
during seeding — despite being the variant most likely to win on energy.

Also add the feasibility rule: when the Table H1 rules exclude a variant, the
prior is masked to the permitted variants and renormalized, so the seeding budget
is never spent proposing configurations that cannot run.

**Enforced by:** `tests/test_tasks.py::TestPrior`.

---

### A-10 — Define R² and UCR

**Where:** Table F1, GP Validation row (`Compute MAE, NPE, R², UCR`).

**Add:**
```
R^2 = 1 - sum (E_i - mu_i)^2 / sum (E_i - E_bar)^2
UCR = fraction of measured energies within  mu_i +/- 1.96 * sigma_pred,i
      where sigma_pred = sqrt(sigma^2(x_i) + sigma_n^2)     target ~ 0.95
```

**Why:** Both are named as deliverables and neither is defined anywhere. The
noise term in `sigma_pred` is not optional: omitting it understates the interval
and makes a correctly calibrated GP appear badly calibrated.

---

### A-25 / A-26 — Define `Q_min(task)` and `S_min`

**Where:** Table B.5 column `Q_min(task)`; Table B.4 column `S >= S_min?`.

**Add:**
- `S_min = 0.95 x mean unoptimized tokens/sec`
- `Q_min(task) = 0.98 x mean unoptimized BERTScore F1 for that task category`

**Why:** Both columns are referenced but never defined. These are the only
definitions consistent with the global SRR ≥ 95% and QRR ≥ 98% criteria.

---

### A-33 — `x*` selection needs an iteration-0 reference

**Where:** Chapter 3, Stage 3 and the `x*` selection rule.

**Add:**

> Before either search begins, the unoptimized default configuration is
> evaluated once on the 50-prompt proxy subset, outside both methods' evaluation
> budgets. This reference supplies the SRR and QRR values used during `x*`
> selection. Final EIR, SRR and QRR are recomputed against the default's
> 500-prompt results.

**Why:** `x*` selection is constrained by SRR ≥ 95% and QRR ≥ 98%, but the
document defines both only against the 500-prompt default — which does not exist
yet at selection time. Without a reference the constraints cannot be evaluated.

---

### A-34 — Define behaviour when no Pareto member meets the thresholds

**Where:** Chapter 3, `x* = argmax EIR` subject to SRR ≥ 95% and QRR ≥ 98%.

**Add** the relaxation ladder and a reported status:

1. `SRR >= 95%` and `QRR >= 98%` → status `optimal`
2. else `SRR >= 95%` and `QRR >= 95%` → status `relaxed_qrr_95`
3. else `SRR >= 90%` and `QRR >= 95%` → status `relaxed_srr_90`
4. else the minimum-energy front member → status `min_energy_fallback`, and
   explicitly **not** a successful optimization

**Why:** The feasible set can be empty — a likely outcome on constrained
hardware — and the document currently specifies no behaviour for that case. A
relaxed selection must never be reportable as a clean one. The Scope section
already commits to reporting null, negative and partial outcomes; this makes that
commitment operational.

**Enforced by:** `tests/test_pareto.py::TestSelection`.

---

## Clarifying

### A-5 — Reconcile the two invocation mechanisms

**Where:** Chapter 3, Inference Engine and Stage 2 ("each prompt is submitted
individually to llama.cpp") versus Scope, p. ~230 ("all inference calls are
issued through llama.cpp's OpenAI-compatible server endpoint").

**Should be:**

> Inference is served by `llama-server`, llama.cpp's OpenAI-compatible HTTP
> server, launched as a subprocess. The precision variant, GPU-layer count, CPU
> thread count and context size are server *launch* flags, so the server is
> restarted once per configuration; the generation cap and request concurrency
> are per-request parameters. All requests use `temperature = 0.0`.

**Why:** The two descriptions are mutually exclusive as written. The server is
the right reading: a per-prompt subprocess would reload the model on each of
~1,500 generations.

---

### A-8 — Add the NVML energy-counter path and the fallback ladder

**Where:** Chapter 3, GPU Power Measurement; Data Analysis energy equation.

**Add:**

> GPU energy is obtained from `nvmlDeviceGetTotalEnergyConsumption`, a monotonic
> millijoule counter maintained by the driver, where the device supports it. This
> is exact and requires neither numerical integration nor idle-power subtraction.
> Where unavailable, energy is computed by integrating `(P_t - P_idle)` over the
> inference window from `nvmlDeviceGetPowerUsage` samples at 100 ms intervals.
> Every record carries `energy_method` identifying which instrument was used.

**Why:** Not all GPUs expose power telemetry. Measured directly on the
development machine, an NVIDIA MX330 returns `NVML_ERROR_NOT_SUPPORTED` (rc = 3)
for **both** `nvmlDeviceGetPowerUsage` and
`nvmlDeviceGetTotalEnergyConsumption`, while utilization and memory queries
succeed. A study that silently recorded zeros there would be worse than one that
records its measurement path. The energy counter also removes integration error
where it is available.

Add too: `(P_t - P_idle)` is clamped at zero per sample, with the clamp count
reported; `dt` comes from measured timestamps rather than the nominal 100 ms.

---

### A-9 — `/proc` does not exist on every host

**Where:** Chapter 3, CPU and Memory Utilization; Hardware Profiling.

**Add:** either restrict the study to Linux/WSL2 explicitly, or document the
alternative:

> CPU utilization, memory and thread counts are read from the Linux `/proc`
> filesystem. On Windows hosts the equivalent values are obtained through
> `ctypes` calls into `kernel32` (`GetSystemTimes`, `GlobalMemoryStatusEx`,
> `GetLogicalProcessorInformation`, `K32GetProcessMemoryInfo`), preserving the
> standard-library-only constraint.

**Why:** The tool is hardware-agnostic by design, and `/proc` is Linux-specific.

---

### A-14 — Table H1's RAM rule must use process-visible memory

**Where:** Table H1, rules `HW-B1`–`HW-B3`.

**Add:** "System RAM refers to the memory visible to the inference process."

**Why:** Under WSL2 the memory ceiling comes from `.wslconfig`, not the host.
On the development machine Windows reports **15.78 GiB** while WSL reports
**11.68 GiB** — the same physical machine, straddling no boundary here but
demonstrating that the two figures differ materially and must be distinguished.

---

### A-12 — Use Dolly's literal category values

**Where:** Table T1 (row "General Instr."); Sources of Data; Sampling Data.

**Should be:** the eight canonical strata are Dolly's own `category` strings:
`open_qa`, `closed_qa`, `information_extraction`, `summarization`,
`classification`, `creative_writing`, `brainstorming`, `general_qa`.

**Why:** Table T1's "General Instr." does not correspond to any Dolly category.
The actual category is `general_qa` — open-domain question answering without a
supplied context, not general instruction following. Using anything other than
the literal strings makes the stratification unreproducible.

Record too the verified counts, which the allocation depends on: open_qa 3742,
general_qa 2191, classification 2136, closed_qa 1773, brainstorming 1766,
information_extraction 1506, summarization 1188, creative_writing 709
(**15,011** rows total).

---

### A-15 — Fix the hypervolume reference point, and normalize

**Where:** Chapter 3, `HV(PF, r)` equation and the definition
`r = (E_default, −S_default, −Q_default)`.

Two changes are needed, and the second is not cosmetic.

**(a) Normalize.** Add: "Objectives are min–max normalized before the volume is
computed, so HV ∈ [0, 1]." Unnormalized, HV carries units of J × (tok/s) × F1
and is dominated by whichever axis has the largest numeric range — energy,
spanning tens to hundreds of joules, swamps quality, spanning ~0.1 of an F1
point. The number is otherwise uninterpretable and incomparable across machines.

**(b) Clamp the reference to the nadir.** Add:

> The reference point is clamped component-wise to the nadir of the evaluated
> front, `r_eff[d] = max(r[d], max over PF of f_d(x))`, so that it is not
> dominated by any front member on any axis.

**Why (b) is required, with evidence.** Hypervolume is positive only for front
members strictly better than the reference on **every** axis. The unoptimized
default runs **F32** — the highest-precision, highest-quality variant in the
space — so by construction no configuration can beat it on quality. The quality
axis of `r = (…, −Q_default)` therefore has zero span, and HV collapses to
exactly **0** no matter how good the front is.

This is not hypothetical. A completed run produced nine front members and a
41.78% energy reduction, and HV against the stated reference was `0.0000`:

| front member | Energy (J) | tok/s | F1 | beats default's 0.8615 F1? |
|---|---|---|---|---|
| best energy | 63.21 | 28.35 | 0.7934 | no |
| best quality | 141.20 | 22.48 | 0.8507 | no |
| … (7 more) | 83–135 | 12–29 | 0.79–0.83 | no |

With the reference clamped to the nadir, the same front scores **HV = 0.5134** —
a figure that varies with front quality and is therefore usable as the indicator
Chapter 3 describes. Clamping to the nadir is also the standard multi-objective
convention: the reference is the worst corner of the considered set.

**Enforced by:** `tests/test_pareto.py::TestHypervolume`, including
`test_positive_when_the_front_cannot_beat_the_reference_on_one_axis`, which uses
the measured values above as its fixture.

---

### A-16 — SER is a single-run point estimate

**Where:** Chapter 3, SER definition; Statement of the Problem 5.3.

**Add:** "SER is reported as a mean and standard deviation over repeated seeded
runs. A supplementary measure, the number of iterations required to reach within
5% of a method's own final best energy, is also reported."

**Why:** `k*` is an iteration index, so a single SER value is one draw from a
high-variance statistic. Measured across six simulator seeds, SER ranged from
0.59 to 13.0 with a mean of 4.5 — the mean supports the claim, but any individual
value could mislead in either direction.

---

### A-21 — State the prompt template and the proxy-subset floor

**Where:** Sampling Data; Research Instrument.

**Add:** the prompt template verbatim with its hash, and:

> The 50-prompt proxy subset allocates at least one prompt to every non-empty
> task category, a minor deviation from strict proportionality.

**Why:** The template materially affects BERTScore and is currently unspecified.
Without the per-category floor, the smallest category (`creative_writing`, 709
rows) rounds to zero in a 50-prompt subset, and the claim that the subset
"preserves proportional representation across task categories" becomes false.

---

### A-20 — Name the tokenizer used for the length filter

**Where:** Sampling Data ("prompts that exceed the maximum token length").

**Add:** "Over-length filtering during dataset preparation uses a documented
characters-per-token heuristic; the authoritative token count is taken from the
inference backend at run time and any prompt exceeding the context budget is
flagged in the logs."

**Why:** The exclusion presupposes a tokenizer that is never identified, so the
filter is not reproducible as stated.

---

### A-22 / A-23 / A-24 — Appendix B schema corrections

**Table B.1** — reduce to three rows (one per condition) with an explicit
`n_prompts` column. Thirty rows for three conditions over 500 prompts is not a
coherent unit of observation, and its resemblance to N is what created the
30-vs-34 ambiguity about the iteration budget.

**Table B.2** — replace `FP32 / FP16 / INT8` with `F32 / F16 / Q8_0 / Q4_K_M`,
and make `CPU Threads` numeric. The template's final row reads "All", which is a
GPU-layer value, not a thread count.

**Table B.3** — remove the `CodeCarbon Validation` columns. CodeCarbon is a
third-party package, contradicts the standard-library-only constraint, and
appears nowhere else in the document. Replace with `energy_crosscheck_j` from
NVML's own energy counter, which is an independent driver-side measurement of the
same quantity and a strictly better validation of the Riemann sum.

---

### A-27 / A-11 — Table F1 references eleven tables that do not exist

**Where:** Table F1.

**Should be:**

| Table F1 cites | Actual artifact |
|---|---|
| A.7 | **B.7** calibration log |
| A.1–A.6 | **B.1** trials, **B.2** per-configuration, **B.3** energy, **B.4** speed, **B.5** quality, **B.6** resources |
| A.6 (statistical analysis) | **B.8** summary comparison |
| A.10 | **B.9** `x*` selection *(new)* |
| A.11 | **B.10** Pareto front and hypervolume *(new)* |
| M1 | **B.11** surrogate reliability *(new)* |

**Why:** Tables A.1–A.11 and M1 are cited as deliverables but appear nowhere.
Appendix A contains only an image. Three genuinely new tables are needed; the
rest map onto the existing B-tables.

---

### A-31 — The determinism claim is too strong

**Where:** Scope, p. ~230: "temperature set to 0.0 to support deterministic
output across experimental conditions."

**Should be:**

> Greedy decoding (`temperature = 0.0`) with a fixed seed makes repeated runs of
> the *same* configuration reproducible. Outputs may still differ *across*
> configurations, because quantization, offload split, thread count and batch
> size alter floating-point reduction order and kernel selection.

**Why:** The claim as written is false, and it matters: part of the
cross-configuration BERTScore variance is numerical rather than semantic.

---

### A-32 — Random search samples without replacement and uses the same rule

**Where:** Chapter 3, Stage 4.

**Add:** "Candidate configurations are sampled uniformly at random **without
replacement**, and the winner is selected by the same Pareto + retention +
argmax-EIR rule used by BOPIS."

**Why:** With replacement, duplicate draws would silently shrink the baseline's
effective budget and the equal-budget premise of SER would fail. SER only
compares like with like if both methods use the same selection rule.

---

### A-35 — `J/token` charges prefill energy to generated tokens

**Where:** Chapter 3, Data Analysis, `J/token = E / N_generated`.

**Add:** "Prefill and decode energy are additionally reported separately, split at
the `prompt_ms` / `predicted_ms` boundary reported by llama.cpp."

**Why:** Prefill and decode have very different energy profiles. Attributing all
energy to generated tokens leaves `J/token` dependent on prompt length — exactly
what the normalization was intended to remove.

---

### A-36 — Strengthen the RQ5 evidence base

**Where:** Statement of the Problem 5; Data Analysis.

**Add:** leave-one-out cross-validation (see A-40), the fitted noise standard
deviation, and a direct repeatability measurement — one configuration
re-evaluated five times, reporting the coefficient of variation of measured
energy.

**Why:** Twenty to thirty sequential predictions is a thin basis for a
reliability claim, and the repeatability CV gives the reader the noise floor
against which a 15% EIR threshold should be judged. It is a short experiment that
materially strengthens the chapter.

---

### A-38 — Enable BERTScore baseline rescaling

**Where:** Statistical Treatment, BERTScore F1.

**Add:** "BERTScore baseline rescaling is enabled, and absolute F1 deltas are
reported alongside QRR."

**Why:** Unrescaled BERTScore F1 occupies a narrow high band, so `QRR ≥ 98%`
would be nearly impossible to fail and the quality-retention criterion would
carry no information.

---

### A-39 — Complete Table 3.3 and state that profiling is dynamic

**Where:** Table 3.3.

**Add:** fill the five `[fill in: …]` placeholders (CPU, RAM, storage, NVIDIA
driver), add rows for `nvmlDeviceGetTotalEnergyConsumption` availability and the
fixed `--ctx-size`, and note that the feasible configuration space is derived
from the profiled machine at run time rather than assumed.

**Why:** The placeholders are unfilled, and the hardware-agnostic design is a
claim worth making explicitly.

---

### A-13 — Remove the contradictory backend card from the UI

**Where:** `bopis.html`, "Model & Execution Backend".

The mockup names Ollama, pynvml, pyRAPL and ROUGE-L. Chapter 3 mandates
llama.cpp, NVML via `ctypes`, `/proc`, and BERTScore F1 only. The replacement
dashboard reads the backend, energy method and versions from the run manifest
instead, so it cannot contradict the run it displays.

**Enforced by:** `tests/test_provenance.py::TestDashboardHasNoHardcodedMetrics`.
