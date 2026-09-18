# Manuscript Revision Plan

**Written for:** the BOPIS thesis team, to plan the revised manuscript together.
**Source:** working session of 2026-09-18. Every claim below was produced by
running the code on the study laptop; nothing here is estimated or recalled.
**Companion docs:** `STATUS_AND_ACTION_ITEMS.md` (what the tool does today),
`UI_GUIDE.md` (how to demonstrate it), `CHANGELOG_G2_EDITS.md` (amendment log).

---

## 0. How to use this document

Each section below is one finding, in this shape:

- **What we found** — the verified fact.
- **Why it matters** — the consequence for the thesis.
- **Manuscript impact** — the specific sections to change.
- **Suggested wording** — a starting draft, not final text.
- **Decision needed** — what the team or the adviser must settle.

Work top to bottom: §1 and §2 change the experimental design, so they must be
settled before anything downstream is rewritten.

**Priority key:** 🔴 changes the design · 🟡 changes claims · 🟢 additive.

---

## 1. 🔴 The model should change: Mistral 7B → Qwen2.5-1.5B

### What we found

Every candidate model was run through the project's own HW-P0/HW-B0 guards
against the live host profile at `ctx_size = 2048`. Feasible configurations after
a clean boot (~13 GiB free RAM):

| Model | Q4_K_M | Q8_0 | F16 | F32 | Feasible | Variants | GPU offload |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| Qwen2.5-0.5B-Instruct | 0.28 | 0.49 | 0.92 | 1.84 | **744**/768 | all four | g = 14, 28, all |
| **Qwen2.5-1.5B-Instruct** | 0.87 | 1.53 | 2.88 | 5.75 | **504**/768 | **all four** | **g = 14, 28, all** |
| Llama-3.2-1B-Instruct | 0.69 | 1.22 | 2.30 | 4.60 | 480/768 | all four | g = 14, 28, all |
| Llama-3.2-3B-Instruct | 1.81 | 3.18 | 5.98 | 11.97 | 228/768 | all four | g = 14 only |
| **Mistral-7B-Instruct** | 4.08 | 7.17 | 13.50 | 27.00 | **96**/768 | Q8_0/Q4_K_M | **NONE — CPU-only** |

Sizes in GiB. On the currently-loaded machine (1.49 GiB free) Mistral 7B gives
**0 of 768** feasible configurations; Qwen2.5-0.5B still gives 696.

### Why it matters

This is the single most consequential finding of the session. Switching to
Qwen2.5-1.5B does not merely make the demo faster — **it makes the manuscript's
stated experimental design executable for the first time:**

- **All four precision variants become feasible, including F32.** Amendment A-38
  and the RQ 5.5 rewording had to soften the variant enumeration to "the feasible
  variants" precisely because F32 Mistral 7B is ~27 GiB and can never run here.
  At 5.75 GiB, F32 Qwen2.5-1.5B fits. The softening could be reverted.
- **The GPU-layer dimension `g` gets its range back.** On Mistral 7B, `g`
  collapses to a single level, so one of five configuration parameters has no
  variation — a real weakness flagged in the energy brief §7.1. On Qwen2.5-1.5B,
  `g = 14`, `g = 28` and full offload are all feasible.
- It aligns the study with the 1B–7B consumer-hardware range benchmarked by
  Zähl & Hennig (2026), the closest published work and the source of our J/token
  plausibility band.

### Manuscript impact

| Section | Change |
| :--- | :--- |
| Table 3.3 (Hardware & model) | Model row: Mistral-7B-Instruct-v0.3 → Qwen2.5-1.5B-Instruct; layer count 32 → 28 |
| Table 3.2 (Configuration space) | `g` domain becomes genuinely populated; state the feasible set per variant |
| Table H1 notes / Stage 1 | HW-P0 example changes; F32 is no longer the canonical infeasible case |
| RQ 5.5 | The "feasible GGUF variants" softening may be reverted to the full enumeration |
| Amendment A-38 | Revisit — its motivation was F32 infeasibility |
| Scope & Limitations | Add the disclosed substitution and its reason |

### Suggested wording

> The base model is Qwen2.5-1.5B-Instruct in GGUF format. The manuscript's
> original instrument specified Mistral-7B-Instruct-v0.3; that model was
> replaced after hardware profiling established that no configuration of it
> satisfies the HW-P0 memory guard on the study host, and that even its
> most compressed feasible variant admits no GPU-layer offload. Qwen2.5-1.5B
> admits all four precision variants and the full GPU-layer domain, making the
> five-parameter configuration space of Table 3.2 executable as specified.

### Decision needed

**This is a thesis-scope decision, not a technical one.** Confirm with the
adviser before rewriting anything. Fallback if the adviser wants to keep a 7B
model: the study becomes CPU-only with a two-variant precision domain, and
Table 3.2 must say so.

---

## 2. 🔴 GPU offload makes inference *slower* on this hardware

### What we found

Measured with a real `llama-server` (build b11026, Vulkan backend),
Qwen2.5-1.5B-Instruct Q4_K_M, `ctx 2048`, 4 threads, 64-token request.
Reproduce with `tools/bench.ps1`.

| Target | GPU layers | Prefill | Decode | vs. CPU |
| :--- | ---: | ---: | ---: | ---: |
| **CPU only** | 0 | 2.18 s | **11.52 tok/s** | — |
| Intel Iris Xe (Vulkan0) | 28 (all) | 3.04 s | 7.55 tok/s | 0.66x |
| NVIDIA MX330 (Vulkan1) | 14 (half) | 2.15 s | 4.67 tok/s | 0.41x |
| NVIDIA MX330 (Vulkan1) | 28 (all) | 1.89 s | 3.28 tok/s | **0.28x** |

Throughput degrades **monotonically** in the number of layers moved to the
MX330. Full offload is 3.5x slower than not using the GPU at all. The MX330 is a
GP108 part with a 64-bit memory bus; decode is memory-bandwidth-bound, and the
CPU's path to system RAM is wider.

### Why it matters

`g` is not a *beneficial* dimension on this host — it is a **harmful** one. BOPIS
optimizing for energy should discover `g = 0` and be right to.

That is a legitimate and even interesting result — *"the optimizer correctly
rejects the GPU"* — but it is a **different story** from "we optimize GPU
offload," which is what the manuscript currently implies. Presenting the second
while the data supports the first is the most likely place to be caught.

### Manuscript impact

| Section | Change |
| :--- | :--- |
| RQ 3 / RQ 3.5 | Reframe from "how much does offload help" to "whether offload helps at all on consumer hardware" |
| Chapter 3 — Configuration space | Note that `g` may be selected *downward*; this is a finding, not a failure |
| Chapter 4 (when written) | Report the device benchmark as a result in its own right |
| Scope & Limitations | The conclusion about `g` is specific to this GPU class |

### Suggested wording

> Offloading transformer layers to the study host's discrete GPU reduced
> generation throughput monotonically in the number of layers offloaded, to
> 0.28x of CPU-only throughput at full offload. The device is a GP108-class part
> with a 64-bit memory bus; because decode is memory-bandwidth-bound, offloading
> moves work to the slower of the two available memory paths. Accordingly, GPU
> layer count is retained as a search dimension but is expected to be optimized
> toward zero on this hardware class.

### ⚠️ Caveat that must be stated

These figures are from the **Vulkan** backend. The manuscript assumes CUDA. The
CUDA 12.4 binaries require a newer driver than this host's 528.96, so CUDA was
not measured. **Do not report Vulkan numbers as CUDA numbers.** Re-measure with
a CUDA build before drawing a final conclusion about `g` (action item B-5).

Also: free RAM was 2.2–3.0 GiB during these runs, so absolute throughput is
depressed. The *ordering* is robust (the effect is 3.5x), but re-run after a
clean boot before quoting absolute tok/s.

---

## 3. 🟡 A second GPU exists that the study never considered

### What we found

`llama-server --list-devices` reports **two** Vulkan devices:

```
Vulkan0: Intel(R) Iris(R) Xe Graphics  (8081 MiB, 7421 MiB free)
Vulkan1: NVIDIA GeForce MX330          (2196 MiB, 1893 MiB free)
```

The integrated Iris Xe has **7.4 GiB available** against the MX330's 1.9 GiB,
and is faster (7.55 vs 3.28 tok/s at full offload).

### Why it matters

`bopis/hardware.py` profiles via NVML, which sees only the NVIDIA card. The
study's entire feasibility analysis therefore silently excluded the larger and
faster accelerator on the same machine. A panellist who knows Tiger Lake laptops
may ask about this.

### Manuscript impact

- **Scope & Limitations** — state that device discovery is NVML-based and
  therefore NVIDIA-only, and that an integrated GPU was present but out of scope.
- **Future Work** — a backend-agnostic device enumeration (Vulkan) would widen
  the feasible space.

### Suggested wording

> Hardware profiling enumerates accelerators through NVML, which reports only
> NVIDIA devices. The study host additionally exposes an Intel Iris Xe
> integrated GPU with greater available memory than the discrete card; it was
> outside the instrument's detection scope and is not included in the feasible
> configuration space. Extending device discovery beyond NVML is identified as
> future work.

### Decision needed

Whether to extend scope now (more work, larger feasible space) or declare it a
limitation (cheap, honest). **Recommendation: declare it a limitation** — this
close to submission, widening the instrument invalidates existing profiling.

---

## 4. 🔴 BERTScore is implemented but never invoked — blocks any quality claim

### What we found

`bopis/quality/bertscore.py` is a complete BERTScore implementation, and
`bopis.quality.get_scorer()` is the documented entry point. **Nothing in the
codebase calls it** — `grep get_scorer` outside its own package returns nothing.

On a real (`--backend llama-server`) run:

1. `LlamaServerMeasurer` leaves `quality_f1` unset — deliberately, and it says so.
2. No later stage fills it in.
3. So `quality_f1` stays `None`, **QRR cannot be computed**, and the quality
   column of the BOPIS-vs-random comparison would be empty.

The QRR 98.6% currently on the dashboard comes from the **simulator**, which
synthesises `quality_f1` analytically (`scorer = analytic_model` in
`validation/per_prompt_quality.csv`). **It is not a BERTScore number.**

### Why it matters

QRR is one of the three dependent variables. Without this wired up there is no
measured quality result, which means no QRR, no quality constraint on `x*`, and
no answer to "how did you evaluate output quality?"

### What the fix requires — two changes, in order

1. **Persist the generations.** `GenResult.text` carries the model output during
   a run, but no run artifact stores it. Without the candidate text there is
   nothing to score afterwards. Add the response text to the per-prompt
   validation output.
2. **Add a scoring command**, e.g. `bopis score <run-dir>`, that loads those
   generations plus the Dolly references already saved in
   `dataset/sample_500.csv`, calls `get_scorer("bertscore")`, and writes
   `quality_f1` back with `scorer = bertscore`.

Keeping it a separate offline command is required, not a shortcut: amendment A-6
and `tests/test_provenance.py` forbid importing torch into the measurement path,
and loading it mid-run would contaminate the energy measurement beside it.

### Manuscript impact

- No change to the *method* — Chapter 3 already specifies BERTScore correctly and
  as an offline stage. The gap is implementation, not design.
- **Chapter 4 cannot report QRR until this lands.**

### Honest interim statement

> Quality is instrumented end to end and validated on the simulator; BERTScore
> scoring of real generations is the remaining implementation step.

---

## 5. 🟡 Citation corrections — two cited papers do not exist

Corrected in `docs/RRL.md`, which now carries inline verification comments.

| Record as previously cited | Finding |
| :--- | :--- |
| Pham, Qian, Wang & Yu (2020), *Problems and opportunities in neural network robustness and reproducibility*, arXiv:2206.04236 | **Does not exist.** No match on arXiv or Scholar; a 2020 paper cannot hold a 2022 arXiv ID. Replaced with **Goldberg (1991)** on floating-point non-associativity, which is a stronger fit for A-31. |
| Xu, Z., et al. (2023), *Evaluating quantization-induced energy reduction in local LLM deployment* | **Not found**, and carried no arXiv ID. Replaced with **Shi & Ding (2025)**, arXiv:2508.16712. |
| Efron & Tibshirani (1993), DOI `10.1017/CBO9780511802843` | Wrong DOI — it resolves to Davison & Hinkley (1997). Cite by ISBN. |
| Lim, Rawson & Ballew (2014), EuroSys '14 | Wrong authors, year and proceedings. The paper is **Colmant, Kurpicz, Felber, Huertas, Rouvoy & Sobe, EuroSys '15**, DOI `10.1145/2741948.2741971`. |
| Frantar & Alistarh, SparseGPT, for a quantization claim | SparseGPT is a **pruning** paper. Replaced with **GPTQ** (Frantar, Ashkboos, Hoefler & Alistarh, ICLR 2023). |

Also obsolete: the summary table's "INT8 Benefits" role — amendment A-23
replaced FP16/INT8 with F32/F16/Q8_0/Q4_K_M, so INT8 is not in this study.

**A draft Table T1 in circulation is wrong.** A two-column version
(`P(FP16)`/`P(INT8)` = 0.53/0.47 for Open QA, etc.) does **not** match
`bopis/tasks.py`, which implements the three-column F32/F16/INT8 form
(`open_qa` = 0.15/0.45/0.40) expanded to four columns by A-4. That draft also
cites the non-existent Xu et al. (2023). **Do not circulate it.**

Also: the EnerInfer figures quoted in the energy brief as prediction accuracy
("6.5% / 12% / 9.7% / 5.4%") were a misread — the paper reports *energy
efficiency improvements* of up to 65% / 12% / 24%. Row removed. This weakens the
MAPE ≤ 15% threshold justification, which now rests on four figures none of which
have been verified against a full text (action item A-5).

**Verified correct and untouched:** Jones et al. (1998), Gerganov (2023),
Zitzler & Thiele (1998), Emmerich et al. (2005), Arlot & Celisse (2010),
Agrawal et al. (OSDI '24), Zhong et al. (DistServe), Stojkovic et al.
(2408.00741), Rotem et al. (2012), Narayanan et al. (SC21). All 18 arXiv IDs in
the energy brief §4 resolve to real papers with matching titles.

### Manuscript impact

The corrections landed in `RRL.md`, **not in the manuscript.** Grep
`docs/THESIS_WRITING2_G2.md` for the stale Table T1 and the Xu et al. citation
before printing anything (action item A-1).

---

## 6. 🟢 New component: Stage 1 task classification

### What we found

Stage 1 of the pipeline table needs `P(precision | task)`, keyed by a Dolly
category. During *evaluation* that category is ground truth — Dolly's own
`category` field. During *deployment* there is no label. That gap had no
implementation. It now has two.

**Rule-based** (`bopis/classify.py`): 42 hand-written rules, standard library
only, matching the "Rule-based probability table" method the pipeline table
already names.

**Supervised** (`bopis/classify_trained.py`): multinomial Naive Bayes fitted to
Dolly's human labels. Stratified 70/15/15 split (seed 20260101), smoothing
selected on validation only, test set touched once. 10,507 train / 2,251
validation / 2,253 test, 8,081 features at `min_df = 2`.

**Held-out test results:**

| Basis | Rule baseline | Learned | Delta |
| :--- | ---: | ---: | ---: |
| Exact 8-way | 48.2% | **69.7%** | **+21.4 pts** |
| `open_qa`/`general_qa` merged | 63.6% | 78.6% | +15.0 pts |
| Quality-sensitivity tier | 67.1% | **79.2%** | +12.1 pts |

Train accuracy 81.5%, so a ~12-point generalization gap — moderate and expected;
state it rather than hide it.

### Three things to report honestly

1. **The residual error is intrinsic, not a model failure.** The dominant
   confusion is `general_qa` ↔ `open_qa` (200 of 683 test errors). Dolly separates
   those two by whether answering needs a specific retrievable fact or general
   world knowledge — a property of the *answer*, not of the instruction's surface
   form. Neither classifier can recover it.
2. **The rules still beat the model on two categories** — `closed_qa` 89.1% vs
   81.2% and `open_qa` 78.5% vs 77.0% — where the structural context gate is
   genuinely strong. That is an interesting finding and points at a hybrid.
3. **The rule baseline is optimistic.** Its weights were hand-tuned while
   observing whole-corpus accuracy, so it has effectively seen the test set. The
   +21.4 is therefore a **lower bound**.

### Why a 69.7% classifier is acceptable

The prior weights **only the 10 seed draws** of the 30-evaluation budget. The 20
BO-guided steps follow Expected Improvement over the GP, and `x*` is chosen by
Pareto dominance plus the SRR/QRR floors. A misclassification makes the search
slightly less sample-efficient; **it cannot make `x*` wrong.**

### Manuscript impact

| Section | Change |
| :--- | :--- |
| Chapter 3 — Stage 1 | New subsection: deployment-time task inference, both classifiers, the measured accuracies, and the seeding-only scope |
| Research Instrument | Add the classifier as an instrument component |
| Chapter 4 | The train/test protocol and the comparison table are reportable results |
| Scope & Limitations | Classifier accuracy bounds; the intrinsic `open_qa`/`general_qa` ambiguity |

This is also the component that makes **"we trained a model on Databricks Dolly
15k, with a held-out test set"** literally true — see §8.

---

## 7. 🟡 What "adaptive" can and cannot mean — a claim to correct

### What we found

Only `t` is a per-request parameter (`n_predict`). `p`, `g`, `c` and `b` are
`llama-server` **launch flags** (`backends/llama_server.py:151-159`). A single
running server therefore **cannot switch precision or GPU-layer count per
prompt.** Per-prompt adaptation over the full `x = (t, b, p, g, c)` tuple would
require one server process per configuration, or a model reload between prompts.

### Why it matters

BOPIS selects **one `x*` for the workload.** If the manuscript describes the
system as adapting configuration per prompt, that is falsifiable by reading the
code — and it is the kind of claim a panel checks.

### Manuscript impact

- Anywhere the system is described as per-prompt adaptive, restate it as
  workload-level selection informed by a task-aware prior.
- The classifier's role: it informs **seeding** and supplies the per-task
  `Q_min(task)` threshold required by amendment A-26.
- **Future Work** — true per-prompt adaptation via a configuration-pool of
  warm servers.

### Suggested wording

> BOPIS selects a single configuration `x*` for the deployed workload. The
> task-informed prior operates on the search's seeding phase and on the
> per-category quality floor `Q_min(task)`, not on per-request configuration
> switching: of the five search parameters, only maximum generation length is a
> per-request quantity, while precision, GPU layer count, thread count and slot
> count are inference-server launch parameters.

---

## 8. 🟢 Defence preparation: "what kind of machine learning is this?"

Expect this question. The premise to reject is that machine learning means
fine-tuning a neural network.

### The answer: supervised learning, specifically active learning

| Component | Learning type |
| :--- | :--- |
| GP surrogate (`bopis/gp.py`) | **Supervised regression** — features `x = (t,b,p,g,c)`, continuous target = measured energy |
| The BO loop (`bopis/optimizer.py`) | **Active learning** / sequential experimental design — the learner chooses its own next training point |
| Task classifier (`bopis/classify_trained.py`) | **Supervised classification** — 8 classes, trained on Dolly |
| Pareto sorting (`bopis/pareto.py`) | **Not learning at all** — a deterministic selection algorithm |

**Nothing in BOPIS is unsupervised.** Do not let Pareto non-dominated sorting be
described as unsupervised learning; it sorts, it does not learn, and misnaming it
loses credibility on an otherwise correct answer.

**Is it reinforcement learning? No.** BO is the *stateless limit* of sequential
decision-making — formally a structured bandit with a GP prior. It shares the
exploration/exploitation trade-off (that is what Expected Improvement balances)
but has **no state, no state transitions, no learned policy, and no temporal
credit assignment.** Evaluating one configuration does not change the environment
for the next. Bandits are sometimes filed under RL as the no-state special case,
so the defensible phrasing is *"bandit-adjacent, not RL."* Do not volunteer the
bandit comparison; use it only if pushed.

### The GP really is trained — the evidence

- `GaussianProcess.fit()` at `bopis/gp.py:334`, with the same contract as
  scikit-learn's.
- Hyperparameters `θ = {σ_f², ℓ, σ_n²}` are **learned** by maximizing the log
  marginal likelihood (`gp.py:454-505`), not set by hand. Fitted example from
  `runs/20260909T172504Z`: `signal_variance = 3.99`, `length_scale = 0.937`,
  `noise_variance = 0.00157`.
- Validated as any ML model is: **LOO R² = 0.980, LOO NPE = 3.77%, UCR = 0.967.**
- `tests/test_gp.py::TestAgainstSklearn` checks the posterior mean, variance and
  log marginal likelihood against `sklearn.gaussian_process` — and **it runs and
  passes** on this machine (sklearn 1.7.0). The model is written from scratch in
  the standard library *and* verified equivalent to the reference implementation.

Training-set size is 30. That is *small-n regression*, not an absence of
learning — and it is exactly why a Gaussian Process is the right model class
rather than a neural network: GPs are the standard tool when each label costs
minutes of computation and calibrated uncertainty is required from few points.

### The one-sentence version

> "We do not train a language model. We fit a Gaussian Process surrogate to 30
> measured configurations — hyperparameters learned by marginal-likelihood
> maximization, validated at leave-one-out R² = 0.98 — and use its calibrated
> uncertainty to decide which configuration to measure next. The language model
> is the objective function, not the learner."

### Where training does *not* belong

- **Not the LLM.** Fine-tuning would change the objective function mid-study and
  invalidate every energy comparison, because the baseline and the BOPIS winner
  would no longer run the same model.
- **Not the energy estimator.** Its coefficients are vendor power budgets, not
  fitted parameters. Fitting them needs measured ground-truth joules — which is
  the RAPL question, not a training question.
- **Not the Pareto stage.** Non-dominated sorting is exact and deterministic.

---

## 9. 🟢 Energy: what is claimable today

The MX330 exposes no power sensor — both `nvmlDeviceGetPowerUsage` and
`nvmlDeviceGetTotalEnergyConsumption` return `NVML_ERROR_NOT_SUPPORTED`. No
driver flag changes this, and llama.cpp reports no energy of its own.

| Claim | Instrument needed | Status |
| :--- | :--- | :--- |
| "BOPIS reduces energy by X%" (EIR) | Ratio only — survives the estimator intact | **Defensible** |
| "This configuration consumes N J/token" | Absolute joules — needs calibration | **Not yet** |

The first survives because systematic bias cancels in a ratio (the §3.1
cancellation proof in the energy brief). The chat UI now shows a labelled Mode C
estimate — prefixed `~`, coloured amber, tagged `estimated (Mode C)`, with the
full caveat on hover.

**A useful external check passed:** the UI reports roughly 1–3 J/token. Zähl &
Hennig (2026) measured 0.56–0.65 J/token for 1B models on an RTX 4060Ti. Being
somewhat higher on CPU inference is plausible, so this satisfies the Route 3
plausibility check in the energy brief §3.2.

Routes to a real number: **RAPL** (`MSR_PKG_ENERGY_STATUS`, measured CPU joules —
and since `g = 0`, nearly all of it; costs an admin kernel driver and breaks the
stdlib-only policy), a GPU with NVML power (different machine), or an external
wall meter (~PHP 1–2k).

### Manuscript impact

Chapter 3 already carries this via amendment A-19. What is new is that the
**deployment UI** also displays an estimate, so the labelling requirement extends
to the chatbot surface — worth one sentence in the Research Instrument section.

---

## 10. Verification record

Everything below is reproducible on the study laptop.

| Check | Result |
| :--- | :--- |
| Full test suite | **517 tests, OK** (`python -m unittest discover -s tests -t .`) |
| Rule classifier vs Dolly | 49.7% exact / 64.3% QA-merged / 67.5% sensitivity tier |
| Trained classifier, held out | **69.7%** exact / 78.6% / 79.2% vs 48.2% rule baseline |
| GP vs scikit-learn | passes (sklearn 1.7.0, numpy 2.0.2) |
| Browser rule classifier vs Python | 16/16 prompts agree |
| Browser Naive Bayes vs Python | **155/155 agree, max confidence delta 0.0** (bitwise identical) |
| Markdown renderer XSS | `<img onerror>` and `<script>` render inert |
| Real inference | Qwen2.5-1.5B Q4_K_M via llama-server b11026, ~11.5 tok/s CPU |
| Feasible space, model-aware | **0 of 768** on the loaded host — all rejected by HW-P0 |

### Two corrections to earlier records in this repo

1. **The changelog's "436 tests, 3 skipped, 127 s" is wrong.** The documented
   per-module invocation did not work at all — an unrelated `tests` package in
   `site-packages` shadowed the repo's `tests/`, which had no `__init__.py`.
   Fixed by adding `tests/__init__.py`. Real figures: **517 tests, 0 skipped**,
   in 287–1535 s depending on machine load. Do not quote a single runtime.
2. **A first device benchmark was invalid.** `pkill -f llama-server` does not
   reap Windows processes, so four orphaned servers held 5.6 GB and starved the
   machine to 0.92 GiB free. Use
   `Get-Process -Name llama-server | Stop-Process -Force`. The §2 table is the
   clean re-run.

---

## 11. Ordered work plan

### Blocking — nothing downstream is valid until these are done

| # | Item | Owner |
| :--- | :--- | :--- |
| 1 | **Reboot, then `python -m bopis profile --model-aware`.** Still 0 feasible. Free, and also fixes the swapping that inflated test runtime. | you |
| 2 | **Settle the model substitution** (§1) with the adviser. | team |
| 3 | **Run the first real study** — `--backend llama-server --model-aware`. Replaces the `sim` dashboard payload. | you |
| 4 | **Wire up BERTScore** (§4). Blocks every quality claim. | code |

### Then

| # | Item |
| :--- | :--- |
| 5 | Re-measure GPU offload with a **CUDA** build (§2 caveat) |
| 6 | Grep the manuscript for the stale Table T1 and Xu et al. (§5) |
| 7 | A-36 repeatability run: one configuration, five repeats, CV of measured energy |
| 8 | Verify two of the four MAPE figures behind the ≤ 15% threshold |
| 9 | Decide RAPL (§9) — needs a feasibility spike |
| 10 | Confirm model architecture params in §1 against `llama-server`'s load log |

### Already resolved — do not re-raise

- EI sign error in the manuscript (A-2 applied; code was always correct).
- `bopis/stats.py` has no bootstrap (`bootstrap_ci` implemented).
- `unittest discover` failing (`tests/__init__.py`).
- Two fabricated and three misattributed citations (corrected in `RRL.md`).
- No task classifier for deployment (two now exist, both measured).
