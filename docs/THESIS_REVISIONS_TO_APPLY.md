# BOPIS — Consolidated Revision Checklist for the Revised Thesis

**Purpose:** one file holding **every pending change** to the revised thesis, so a
second reader can map each item to its exact location in the manuscript and
state what to write there.

**Baseline (verified 2026-09-20):** the Google Doc you will edit matches the old
PDF draft `docs/THESIS_WRITING2_G2.pdf` (102 pp), **not**
`docs/THESIS_WRITING2_G2.md`. The `.md` is a newer draft but carries its own stale
spots; **§0.5** lists the PDF's verified state chapter by chapter so each item in
§2/§3 can be re-anchored to the text actually present in your Google Doc. PDF
verification anchors are line numbers in the extracted text `tw2_g2_pdf.txt`
(kept in the opencode temp dir).

**How to use this file (for the reader):** for each numbered item, reply with
(1) the section / paragraph / table in the Google Doc (= PDF draft), (2) the
current text to replace (quote the anchor), and (3) the replacement text. Where a
suggested wording already exists, it is included verbatim below.

**Priority key:** 🔴 blocks correctness / design · 🟡 changes a claim · 🟢 additive.

**Status key:** `decision` = must be settled with the adviser first ·
`write` = can be written now · `verify` = needs a re-run/measurement first ·
`code` = needs code work before it can be reported.

---

## 0. The manuscript's section map (for locating items)

Section names in `THESIS_WRITING2_G2.md` (line numbers shift; search by name):

| Ch. | Section | Notes |
|---|---|---|
| 1 | Conceptual Framework | states base model + variants (~line 284) |
| 1 | Statement of the Problem | RQ block; RQ 3.5 = per-variant enumeration (~line 318) |
| 1 | Scope and Limitation of the Study | base model, backend, determinism claim (~line 326) |
| 1 | Definition of Terms | "Mistral 7B Instruct v0.3", GGUF variants (~line 416) |
| 2 | (body text) | background; check for fake citations used anywhere |
| 2 | Table T1 (Task-Informed Prior) | draft table: **verify 4 columns, verify citations** |
| 3 | Research Design / Sources of Data / Sampling Data / System Architecture / Research Instrument / Data Analysis / Statistical Treatment | Chapter-3 edits |
| 3 | Table 3.2 (configuration space), Table 3.3 (hardware & model), Table H1 (feasibility), Table T1 | tables to update |
| 4 | (not yet present in the source) | results — gates on many items below |
| — | References | citation corrections |
| — | Appendix B (Tables B.1–B.11) | unit-of-observation, precision columns, energy method |

---

## 0.5 Verified state of the Google Doc (= PDF draft)

Extracted from `THESIS_WRITING2_G2.pdf`; anchors are txt-line numbers in the
extracted text (`tw2_g2_pdf.txt`). This is what the manuscript *currently* says —
each §2/§3 item tells you what to change it to.

| Where (PDF) | Verified current text | Notes |
|---|---|---|
| Ch1 Scope | "The locally based model used in the study is Mistral 7B Instruct v0.3, served through llama.cpp using three GGUF precision/quantization variants: F16, Q8_0, and Q4_K_M." (~6330) | three-variant scope; no F32 |
| Ch1 Scope | "The F32 (full-precision) variant is excluded …" (~6536) | F32 exclusion sentence present; hangs on D-1 |
| Ch1 Definition of Terms | GGUF = "three GGUF precision/quantization variants: F16, Q8_0, Q4_K_M"; F16 = "highest-precision variant evaluated" (~9820–9969) | |
| Ch1 RQ 3(e) | "Per-variant performance, as measured across FP16, Q8_0, and Q4_K_M GGUF precision/quantization variants" (~6138) | **reads FP16 here, F16 elsewhere** — unify; feasibility gate at D-1 |
| Ch2 prompt-adaptive | "empirical basis for BOPIS's prompt-adaptive precision selection: the system classifies the incoming prompt … constrains which quantization levels are permissible for that prompt" (~18041–18127) | overclaim → C-04 |
| Ch2 Table T1 | **two columns** P(FP16) \| P(INT8), + Min. Precision + Literature Basis; every row cites Ma et al. (2023) **except** General Instr. = Xu et al. (2023); caption "Researcher-Defined Task-Informed Prior…" (~29628–29823); values e.g. Open QA 0.53/0.47 | stale vs code → C-12 §1 |
| Ch3 Table H1 | HW-P1 / HW-P2 / HW-P3 only — **no HW-P0 / HW-B0 rows** (~28740–28879) | add P0/B0 (C-01 note) |
| Ch3 Table 3.2 | Input token length 128…1024, **default 2048**; Precision F16/Q8_0/Q4_K_M, **default F16**; GPU layers 0/14/28/All, default All; CPU threads 2/4/8 (~32950–33109) | `t=2048` is a config value, not context size — relabel (below) |
| Ch3 Table 3.3 | GPU **RTX 3060 (12 GB GDDR6 VRAM)**; CPU **\[fill in\]**, RAM **\[fill in\]**, Storage **\[fill in\]**, driver **\[fill in\]**; **llama.cpp (build b3447, CUDA backend enabled)**; Mistral 7B F16/Q8_0/Q4_K_M; "The RTX 3060's 12 GB … is the binding hardware constraint" (~34325–34544) | real host ≠ this → C-01/C-02 |
| Ch3 Data Analysis | `E = Σ[(P_t − P_idle) × Δt]`, mW→W, `J/token = E / N_generated` (~39041–39290) | supplies Mode C only — cannot give absolute J without a power sensor → C-06 |
| Ch3 reliability | "An NPE below 10% is treated as a researcher-defined reliability threshold" (~39690–39816) | **no LOO / σ_n² / R²≥0.85 anchors in the PDF** → C-10/C-11 |
| App. B.2 | Precision **FP16/INT8/FP16**; GPU layers 0/14/28/All; CPU threads 2/4/8/All; CodeCarbon Validation row (~44890–45026) | clean up → C-12 §2 |
| References | Ma, X. et al. (2023) and Xu, Z. et al. (2023) **both printed** (~43841–43863) | avoid into Google Doc → §3 (note: `.md` *also* has both at lines 1342/1418) |

**Two cross-cutting PDF fixes that apply to every location above:** (1) FP16/F16
naming is inconsistent between the RQ and the Scope/Definition-of-Terms — pick one
spelling; (2) every §2 item below that already named `.md` line numbers should be
re-anchored to the PDF text in this table before writing into the Google Doc.

*Section map below still uses `.md` line numbers (the newer draft); for the
Google Doc use the PDF anchors in this table instead.*

---

## 1. Decisions that must be settled with the adviser first

### D-1. 🔴 Model substitution — Mistral 7B → Qwen2.5-1.5B-Instruct
Everything under C-01 depends on this. **Decision needed before rewriting.**

**Facts (measured on the study laptop, 2026-09-18):** at `ctx_size = 2048`, after
a clean boot (~13 GiB free RAM):

| Model | F32 | Feasible configs | Variants | GPU offload |
|---|---|---|---|---|
| **Qwen2.5-1.5B-Instruct** | 5.75 GiB | **504/768** | all four | g = 14, 28, all |
| **Mistral-7B-Instruct** | ~27 GiB | **96/768** | Q8_0/Q4_K_M only | **NONE — CPU-only** |

On the currently-loaded machine (1.49 GiB free RAM) Mistral 7B offers **0 of 768**
feasible configurations; Qwen2.5-1.5B still offers 360/768.

**Consequences:** switching makes the manuscript's design executable (F32 fits —
amendment A-38's motivation disappears; the `g` dimension regains its range).
**Fallback** if the adviser insists on a 7B model: the study becomes CPU-only with
a two-variant precision domain, and Table 3.2 must say so.

- Source: `MANUSCRIPT_REVISION_PLAN.md` §1; `STATUS_AND_ACTION_ITEMS.md` §4.
- Locations touched: C-01.

### D-2. 🟡 Device discovery is NVIDIA-only; a second GPU exists
The host exposes **two** GPUs: Vulkan0 = Intel Iris Xe (8081 MiB, ~7.4 GiB free)
and Vulkan1 = NVIDIA MX330 (2196 MiB, ~1.9 GiB free). Profiling uses NVML so it
never saw the Iris Xe. **Recommendation: declare it a limitation**, and add one
sentence to Scope & Limitations + one to Future Work. If the team instead wants
to extend scope, it invalidates existing profiling — do not do this close to
hand-in.

- Source: `MANUSCRIPT_REVISION_PLAN.md` §3.
- Suggested wording (limitation):
  > Hardware profiling enumerates accelerators through NVML, which reports only
  > NVIDIA devices. The study host additionally exposes an Intel Iris Xe
  > integrated GPU with greater available memory than the discrete card; it was
  > outside the instrument's detection scope and is not included in the feasible
  > configuration space. Extending device discovery beyond NVML is identified as
  > future work.

---

## 2. Changes to write into the manuscript

### C-01. 🔴 Base model, scope, and definition changes (depends on D-1)
**What:** replace "Mistral 7B Instruct v0.3" with Qwen2.5-1.5B-Instruct (28
layers, was 32) everywhere, and state the substitution as a **disclosed one**
with its reason.

**Current locations (quote-and-replace):**
- Conceptual Framework — "The study uses the same base model, Mistral 7B Instruct
  v0.3, across four GGUF variants: F32, F16, Q8_0, and Q4_K_M."
- Scope and Limitation — "The locally based model used in the study is Mistral 7B
  Instruct v0.3, served through llama.cpp using four GGUF precision/quantization
  variants…"
- Definition of Terms — *Mistral 7B Instruct v0.3*, *Quantization*, *GGUF*,
  *F16 GGUF* ("highest-precision variant" line).
- Chapter 3 — Table 3.3 (Hardware & model; model + layer-count rows), Table 3.2
  (`g` domain), Table H1 notes / Stage 1 (HW-P0 example; F32 is no longer the
  canonical infeasible case).
- Statement of the Problem, RQ 3.5 (~line 318) — shown in C-10.

**Suggested wording (Scope):**
> The base model is Qwen2.5-1.5B-Instruct in GGUF format. The manuscript's
> original instrument specified Mistral-7B-Instruct-v0.3; that model was
> replaced after hardware profiling established that no configuration of it
> satisfies the HW-P0 memory guard on the study host, and that even its most
> compressed feasible variant admits no GPU-layer offload. Qwen2.5-1.5B admits
> all four precision variants and the full GPU-layer domain, making the
> five-parameter configuration space of Table 3.2 executable as specified.

- Source: `MANUSCRIPT_REVISION_PLAN.md` §1. Status: `decision` + `write`.

### C-02. 🔴 GPU offload makes inference *slower* on this hardware
**What:** reported result — offloading layers **monotonically reduces** decode
throughput here; full offload is 0.28x of CPU-only. `g` is a dimension BOPIS is
expected to optimize **toward zero**, not a benefit. Reframe any language that
implies "we optimize GPU offload".

**Measured 2026-09-18** (`llama-server` b11026, Qwen2.5-1.5B Q4_K_M, ctx 2048,
4 threads, 64-token request):

| Target | GPU layers | Prefill | Decode | vs. CPU |
|---|---|---|---|---|
| CPU only | 0 | 2.18 s | **11.52 tok/s** | — |
| Intel Iris Xe (Vulkan0) | 28 | 3.04 s | 7.55 tok/s | 0.66x |
| NVIDIA MX330 (Vulkan1) | 14 | 2.15 s | 4.67 tok/s | 0.41x |
| NVIDIA MX330 (Vulkan1) | 28 | 1.89 s | 3.28 tok/s | **0.28x** |

**Locations:** RQ 3 / RQ 3.5 reframing; Chapter 3 configuration-space note that
`g` may be selected downward (a finding, not a failure); Chapter 4 when written —
report the device benchmark as a result; Scope & Limitations — conclusion is
specific to this GPU class (GP108, 64-bit memory bus).

**Suggested wording:**
> Offloading transformer layers to the study host's discrete GPU reduced
> generation throughput monotonically in the number of layers offloaded, to
> 0.28x of CPU-only throughput at full offload. The device is a GP108-class part
> with a 64-bit memory bus; because decode is memory-bandwidth-bound, offloading
> moves work to the slower of the two available memory paths. Accordingly, GPU
> layer count is retained as a search dimension but is expected to be optimized
> toward zero on this hardware class.

**⚠️ Mandatory caveat (do not skip):** these are **Vulkan-backend** numbers. The
manuscript assumes CUDA; CUDA 12.4 binaries need a newer driver than this host's
528.96, so CUDA was not measured. **Do not report Vulkan numbers as CUDA
numbers.** Free RAM was depressed during the run — re-run after a clean boot
before quoting absolute tok/s (the ordering/3.5x is robust).

- Source: `MANUSCRIPT_REVISION_PLAN.md` §2. Status: `verify` for absolute
  numbers (clean-boot + CUDA re-measure); `write` for the ordering claim.

### C-03. 🟡 Task classifier for Stage 1 — new component to report
**What:** add the deployment-time task classification to the manuscript. Two
implementations exist: rule-based (42 rules, stdlib only) and supervised
(multinomial Naive Bayes on Dolly labels, 70/15/15 split, seed 20260101).

**Held-out test results:**

| Basis | Rule baseline | Learned | Delta |
|---|---|---|---|
| Exact 8-way | 48.2% | **69.7%** | +21.4 pts |
| `open_qa`/`general_qa` merged | 63.6% | 78.6% | +15.0 pts |
| Quality-sensitivity tier | 67.1% | **79.2%** | +12.1 pts |

**Honest points to include:** (1) the dominant residual confusion is
`general_qa ↔ open_qa` (200 of 683 test errors, intrinsic to the label
definition — Dolly splits them by a property of the *answer*); (2) the rules
still beat the model on `closed_qa` (89.1% vs 81.2%) and `open_qa` (78.5% vs
77.0%) — points at a hybrid; (3) the rule baseline is optimistic (hand-tuned with
corpus-level feedback, so it has effectively seen the test set) — the margin is a
**lower bound**; (4) a misclassification cannot make `x*` wrong — the prior only
weights the 10 seed draws, the 20 BO steps are acquisition-driven.

**Locations:** Chapter 3 — new subsection for Stage 1 task inference (both
classifiers, measured accuracies, seeding-only scope); Research Instrument — add
the classifier as an instrument component; Chapter 4 — train/test protocol +
comparison table as results; Scope & Limitations — accuracy bounds and the
intrinsic ambiguity.

- Source: `MANUSCRIPT_REVISION_PLAN.md` §6; `STATUS_AND_ACTION_ITEMS.md` §6.5.
  Status: `write`.

### C-04. 🟡 Correct the "adaptive" claim
**What:** the system selects **one `x*` for the workload**, not a config per
prompt. Only `t` (maximum generation length / `n_predict`) is a per-request
parameter; `p`, `g`, `c`, `b` are `llama-server` **launch flags**, so a single
running server cannot switch precision or GPU-layer count per prompt.

**Locations:** anywhere the system is described as per-prompt adaptive → restate
as workload-level selection informed by a task-aware prior. The classifier's
role: inform **seeding** + supply the per-task `Q_min(task)` threshold (A-26).
Future Work → true per-prompt adaptation via a configuration pool of warm servers.

**Suggested wording (paste-ready):**
> BOPIS selects a single configuration `x*` for the deployed workload. The
> task-informed prior operates on the search's seeding phase and on the
> per-category quality floor `Q_min(task)`, not on per-request configuration
> switching: of the five search parameters, only maximum generation length is a
> per-request quantity, while precision, GPU layer count, thread count and slot
> count are inference-server launch parameters.

- Source: `MANUSCRIPT_REVISION_PLAN.md` §7; `STATUS_AND_ACTION_ITEMS.md` §3.2.
  Status: `write`.

### C-05. 🔴 BERTScore: quality results cannot be reported until wired up
**What:** BERTScore is implemented (`bopis/quality/bertscore.py`) but **never
invoked** — nothing stores the generated text, so `quality_f1` stays `None` on a
real run and **QRR cannot be computed**. The dashboard's 98.6% QRR is the
simulator's analytic value, not BERTScore.

**Locations:** Chapter 4 must not report QRR until this lands. The method text in
Chapter 3 is already correct (BERTScore as an offline stage) — **no method
change**. Add the honest interim sentence where results are discussed:
> Quality is instrumented end to end and validated on the simulator; BERTScore
> scoring of real generations is the remaining implementation step.

**Code fix (not manuscript text):** (1) persist `GenResult.text` into per-prompt
validation artifacts; (2) add a `bopis score <run-dir>` command that loads
generations + Dolly references, calls `get_scorer("bertscore")`, writes
`quality_f1` with `scorer = bertscore`. Must be an offline command — A-6 and
`tests/test_provenance.py` forbid torch in the measurement path.

- Source: `MANUSCRIPT_REVISION_PLAN.md` §4; `STATUS_AND_ACTION_ITEMS.md` §7 A-10.
  Status: `code` → `write`.

### C-06. 🟡 Energy claims: what is claimable
**What:** the MX330 exposes no power sensor (both NVML power APIs return
`NOT_SUPPORTED`). **EIR is defensible** (ratio — systematic bias cancels, §3.1
cancellation proof). **Absolute J/token is not claimable** without calibration.
The UI now shows a labelled Mode C estimate (prefixed `~`, amber, tagged
`estimated (Mode C)`).

**Locations:** add one sentence to Research Instrument — the deployment UI also
displays an estimate, so the labelling requirement extends to the chatbot
surface. Keep the existing Chapter 3 A-19 language.

- Source: `MANUSCRIPT_REVISION_PLAN.md` §9. Status: `write`.

### C-07. 🟢 Defence-day framing: "what kind of machine learning is this?"
**What:** supervised learning, specifically **active learning**. The GP surrogate
is supervised regression (`fit()` at `bopis/gp.py:334`, hyperparameters `θ =
{σ_f², ℓ, σ_n²}` **learned** by log-marginal-likelihood maximization — fitted
example: signal_variance 3.99, length_scale 0.937, noise_variance 0.00157;
validated LOO R² = 0.980, LOO NPE = 3.77%, UCR = 0.967). The BO loop is active
learning. The classifier is supervised classification. Pareto sorting is **not
learning**. **Not RL** — bandit-adjacent only (no state, no transitions, no
learned policy).

**One-sentence version (for defence / oral):**
> "We do not train a language model. We fit a Gaussian Process surrogate to 30
> measured configurations — hyperparameters learned by marginal-likelihood
> maximization, validated at leave-one-out R² = 0.98 — and use its calibrated
> uncertainty to decide which configuration to measure next. The language model
> is the objective function, not the learner."

**Where training does *not* belong (do not claim otherwise):** not the LLM
(fine-tuning would invalidate energy comparisons), not the energy estimator
(coefficients are vendor budgets), not the Pareto stage.

- Source: `MANUSCRIPT_REVISION_PLAN.md` §8; `STATUS_AND_ACTION_ITEMS.md` §6.
  Status: defence prep — mostly not a manuscript edit.

### C-08. 🟢 RQ 5.5 — revert the "feasible-only" softening (only if D-1 is yes)
**What:** RQ 3.5 currently reads "as measured across **the feasible** GGUF
precision/quantization variants". The softening existed because F32 Mistral 7B
(~27 GiB) can never run. If the model becomes Qwen2.5-1.5B, F32 fits, and the
full enumeration can be restored.

**Current location:** Statement of the Problem, RQ 3(e) (sub-item of RQ 3; the
line ending "...and output quality?"). Verification anchors (PDF-verified,
§0.5): `Per-variant performance, as measured across FP16, Q8_0, and Q4_K_M GGUF
precision/quantization variants` (note: **FP16 here, F16 in Scope** — unify).

- Source: `CHANGELOG_G2_EDITS.md` "RQ 5.5" entry; `MANUSCRIPT_REVISION_PLAN.md`
  §1. Status: `decision` + `write`.

### C-09. 🟡 Amendment A-38 — BERTScore baseline rescaling sentence missing
**What:** commit `d745a49` lists A-38 ("Enable BERTScore baseline rescaling"),
and `AMENDMENTS.md` defines it, but **no text is present** in the manuscript.

**Add** (Statistical Treatment / BERTScore F1):
> "BERTScore baseline rescaling is enabled, and absolute F1 deltas are reported
> alongside QRR."

**Why:** unrescaled BERTScore F1 occupies a narrow high band, so `QRR ≥ 98%`
would be nearly impossible to fail and the retention criterion would carry no
information. This also motivates the SUCCESS criterion threshold.

- Source: `CHANGELOG_G2_EDITS.md` §D; `AMENDMENTS.md` A-38. Status: `write`
  (confirm the claim is wanted first).

### C-10. 🟡 Amendment A-36 — repeatability leg has no manuscript text
**What:** A-36 has three legs; two landed (leave-one-out CV, mandatory noise term
in σ_pred) but the third — a **repeatability run** (one configuration, five
repeats, CV of measured energy) — has no manuscript sentence yet. **Run the
experiment first, then add one sentence** to Statement of the Problem 5 and/or
Data Analysis reporting the CV as the noise floor against which a 15% EIR
threshold should be judged.

**Verification anchors (already applied):** `The noise term σ_n² is mandatory in
σ_pred`; `leave-one-out cross-validation`; `LOO R² ≥ 0.85`.

- Source: `CHANGELOG_G2_EDITS.md` A-36 + §D; `AMENDMENTS.md` A-36.
  Status: `verify` → `write`.

### C-11. 🟡 MAPE ≤ 15% threshold justification is weakened
**What:** the EnerInfer accuracy figures quoted as prediction accuracy were a
misread (the paper reports energy-efficiency *improvements* of up to 65%/12%/24%;
row removed). The remaining figures behind `MAPE ≤ 15%` are **all unverified
against full texts**. Either verify at least two of them (needs web access) or
present the threshold as **researcher-defined** with its rationale.

**Location:** Statistical Treatment / Data Analysis — the ≤15% NPE/MAPE
reliability threshold justification.

- Source: `CHANGELOG_G2_EDITS.md` §D; `MANUSCRIPT_REVISION_PLAN.md` §5 (brief);
  `AMENDMENTS.md` A-40. Status: `verify` or rewrite as researcher-defined `write`.

### C-12. 🟢 Appendix/Front-matter: Table T1 + units-of-observation
**What (two things):**
1. **The Google Doc's Table T1 is the stale two-column form** — verified in the
   PDF: `P(FP16) | P(INT8)` = 0.53/0.47 for Open QA, etc., and "General Instr."
   cites Xu et al. (2023). It matches **neither** the code nor the newer `.md`
   draft. `bopis/tasks.py` implements the four-column form (A-4), e.g. Open
   QA = `0.15 | 0.45 | 0.20 | 0.20`. Replace the column headers, all values, and
   the Literature Basis column in the Google Doc (fix the caption to match the
   four-variant form; see §0.5 and §3 for the citation rows). Confirm the final
   text is the 4-column form.
2. **Appendix B Table B.1** — three rows (Unoptimized / Random Search / BOPIS)
   with an explicit `n_prompts` column; it is the unit-of-observation fix from
   A-22. Consider whether the reader will confuse "30 rows" with the N = 30
   budget (already fixed per A-3 — do not un-fix). In Appendix B.2, replace the
   stale `FP16/INT8` precision column with the four precision variants and drop
   the CodeCarbon Validation row (CodeCarbon is not part of the final method).

- Source: `MANUSCRIPT_REVISION_PLAN.md` §5; `CHANGELOG_G2_EDITS.md` §A/B.
  Status: `write` (check).

---

## 3. Citation corrections to port into References (src: `RRL.md` §5)

Five records were wrong. **Two cited papers do not exist** — remove/replace:

| Record as previously cited | Finding / replacement |
|---|---|
| Pham, Qian, Wang & Yu (2020), *Problems and opportunities in NN robustness and reproducibility*, arXiv:2206.04236 | **Does not exist.** (A 2020 paper cannot hold a 2022 ID.) Replace with **Goldberg (1991)**, floating-point non-associativity — stronger fit for A-31. |
| Xu, Z., et al. (2023), *Evaluating quantization-induced energy reduction in local LLM deployment* | **Not found**, no arXiv ID. Replace with **Shi & Ding (2025)**, arXiv:2508.16712. ⚠️ Scope caveat from `RRL.md` §3.6: Shi & Ding ran A100/H100 datacenter GPUs at 7B–70B — it supports the energy/quality trade-off claim but **not** a claim about consumer-hardware VRAM limits (that one is our derived HW-P0/HW-B0 arithmetic, not a citation). |
| Efron & Tibshirani (1993), DOI `10.1017/CBO9780511802843` | **Wrong DOI** — resolves to Davison & Hinkley (1997). Cite by ISBN. |
| Lim, Rawson & Ballew (2014), EuroSys '14 | Wrong authors/year/proceedings. The paper is **Colmant, Kurpicz, Felber, Huertas, Rouvoy & Sobe, EuroSys '15**, DOI `10.1145/2741948.2741971`. |
| Frantar & Alistarh, SparseGPT, used for a quantization claim | SparseGPT is a **pruning** paper. Replace with **GPTQ** (Frantar, Ashkboos, Hoefler & Alistarh, ICLR 2023, arXiv:2210.17323). |

**Also:** the summary table's "INT8 Benefits" role is obsolete — A-23 replaced
FP16/INT8 with F32/F16/Q8_0/Q4_K_M; INT8 is not in this study. **Grep the
manuscript** for the stale two-column Table T1 and the Xu et al. citation before
printing (action A-1).

**Verified correct — do not touch:** Jones et al. (1998), Gerganov (2023),
Zitzler & Thiele (1998), Emmerich et al. (2005), Arlot & Celisse (2010), Agrawal
et al. (OSDI '24), Zhong et al. (DistServe), Stojkovic et al. (2408.00741), Rotem
et al. (2012), Narayanan et al. (SC21). All 18 arXiv IDs in
`ENERGY_ESTIMATOR_AND_ML_BRIEF.md` §4 resolve to real papers.

- Source: `MANUSCRIPT_REVISION_PLAN.md` §5; `STATUS_AND_ACTION_ITEMS.md` §5;
  `CHANGELOG_G2_EDITS.md` §D. Status: `write` (references already fixed in RRL).

---

## 4. Verify-then-report items (not pure text edits)

| # | Item | What to do before writing |
|---|---|---|
| V-1 | Re-profile under clean boot | `python -m bopis profile --model-aware`; refresh HW-P0 blocks + Figure 3.5 numbers (currently the loaded machine rejects 0/768 — stale). **Do before final hand-in.** |
| V-2 | CUDA re-measure of GPU offload (C-02) | Driver ≥ 528.96 or pip cudart redistributable; decides whether `g` is a real dimension. Do not report Vulkan as CUDA. |
| V-3 | A-36 repeatability run | One config, five repeats, CV of measured energy (→ C-10). |
| V-4 | BERTScore wiring | Persist generations + `bopis score <run-dir>` (→ C-05). |
| V-5 | MAPE figures | Verify ≥2 of the four ≤15% figures, or reword as researcher-defined (→ C-11). |
| V-6 | Confirm model architecture params | 28 layers for Qwen2.5-1.5B against `llama-server`'s load log before quoting Table 3.3. |

---

## 5. A-37 — unresolved reference (please confirm)

The commit message `d745a49` lists "A-37", but **there is no A-37 entry in
`docs/AMENDMENTS.md`** (register ends A-40; A-36 and A-38 exist, A-37 does not).
Confirm which amendment A-37 refers to before trusting the label.

- Source: `CHANGELOG_G2_EDITS.md` §D.

---

## 6. Verified numbers you may quote today (2026-09-18, reproducible)

- Full test suite: **517 tests, OK** (`python -m unittest discover -s tests -t .`).
  Do **not** quote "436 / 3 skipped / 127 s" — that record was fixed; real run
  time varies 287–1535 s by machine load.
- Rule classifier vs Dolly: 49.7% exact / 64.3% QA-merged / 67.5% tier.
- Trained classifier held-out: **69.7%** / 78.6% / 79.2% (vs 48.2% rule).
- GP vs scikit-learn: passes (sklearn 1.7.0).
- Browser ports vs Python: 16/16 rules; **155/155 Naive Bayes, max delta 0.0**.
- Real inference: Qwen2.5-1.5B Q4_K_M via `llama-server` b11026, ~11.5 tok/s CPU.
- Energy plausibility: UI shows ~1–3 J/token; Zähl & Hennig (2026) measured
  0.56–0.65 J/token on 1B models on an RTX 4060Ti — higher on CPU is plausible.

---

*Generated 2026-09-20 from `MANUSCRIPT_REVISION_PLAN.md`,
`STATUS_AND_ACTION_ITEMS.md`, `CHANGELOG_G2_EDITS.md`, `AMENDMENTS.md` and
`RRL.md`. Send this file plus `docs/THESIS_WRITING2_G2.md` to the reviewer so each
item can be located and a replacement text produced.*