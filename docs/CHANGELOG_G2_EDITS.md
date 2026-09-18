# BOPIS G2 — Chapter 3 + Appendix B Edit Log (for PDF re-confirmation)

One markdown file, one entry per amendment, sorted by amendment number.
`[LOC]` = where the text now sits in `docs/THESIS_WRITING2_G2.md` (line numbers are
the numbers visible in the repo; they shift as edits are made, so search by the
quoted anchor string, not the number). `status: M` = manuscript applied (in this
file); `status: C` = code/test change (in `bopis/`); `status: B` = blocked.

Anchors are kept short and ASCII-only because the manuscript uses curly
apostrophes (U+2019) in some headings and get-the-first-match behaviour differs
between tools.

---

## A. Applied to the manuscript (docs/THESIS_WRITING2_G2.md)

### A-1 — Define `t` as maximum generation length, not context window
- **Where:** Table 3.2 row 1; BO configuration tuple definition.
- **Edit:** Row relabelled "Maximum generation length (t)" (llama.cpp `n_predict`),
  values unchanged {128, 256, 512, 1024}; context window fixed at 2048 (added to
  Table 3.3) and held outside the search space.
- **Verification anchor:** `Maximum generation length (t)`.
- **status:** M.

### A-2 — EI acquisition sign corrected to the minimisation form
- **Where:** System Architecture — EI equation and Z.
- **Edit:** Manuscript now prints `EI(x) = (f(x⁺) − μ(x))·Φ(Z) + σ(x)·φ(Z)` with
  `Z = (f(x⁺) − μ(x)) / σ(x)`, matching `bopis/acquisition.py`. (Section E once
  listed EI as "already correct / not touched"; that note is retired here.)
- **Verification anchor:** `(f(x⁺) − μ(x)) × Φ(Z)`.
- **status:** M (+C enforced by
  `tests/test_acquisition.py::test_prefers_lower_predicted_energy`).

### A-3 — Total iteration budget N = 30
- **Where:** BO procedure Step 6.
- **Edit:** "N = 30 (10 prior-weighted random seeds + 20 BO-guided steps)"; random
  search is given the identical 30-evaluation budget.
- **Verification anchor:** `N \= 30 (10 prior-weighted random seeds`.
- **status:** M.

### A-7 — GP hyperparameter fitting method corrected
- **Where:** GP marginal-likelihood paragraph.
- **Edit:** L-BFGS-B replaced by "a coarse log-space grid search followed by
  multi-start Nelder-Mead" (no standard-library L-BFGS-B exists).
- **Verification anchor:** `multi-start Nelder-Mead`.
- **status:** M (+C GP in `bopis/gp.py`, verified by
  `tests/test_gp.py::TestAgainstSklearn`).

### A-10 — R² and UCR definitions added
- **Where:** Data Analysis — surrogate reliability; Table 3.5.
- **Edit:** R² and UCR formulas specified (`σ_pred = sqrt(σ² + σ_n²)`, target ≈ 0.95);
  Table 3.5 gains GP R² and GP UCR rows with the LOO basis.
- **Verification anchor:** `UCR \= fraction of measured energies within`.
- **status:** M (+C: `bopis/stats.py` R²/UCR helpers, `tests/test_stats.py`).

### A-12 — Dolly's literal category values
- **Where:** Sources of Data; Stratification; Table T1 row labels.
- **Edit:** Categories now Dolly's own strings (`open_qa`, `closed_qa`,
  `information_extraction`, `summarization`, `classification`, `creative_writing`,
  `brainstorming`, `general_qa`); Table T1's "General Instr." → `general_qa`.
- **Verification anchor:** backticked `open_qa` … `general_qa` list.
- **status:** M.

### A-15 — Hypervolume normalized and reference clamped to the nadir
- **Where:** Data Analysis — HV paragraph.
- **Edit:** Added min–max normalization (HV ∈ [0, 1]) and component-wise clamping of
  the reference to the front nadir (`r_eff[d] = max(r[d], max_PF f_d(x))`), with the
  collapse-to-zero explanation.
- **Verification anchor:** `Objectives are min–max normalized`.
- **status:** M (+C: `bopis/pareto.py::hypervolume`,
  `tests/test_pareto.py::TestHypervolume`).

### A-20 / A-21 — Over-length filter tokenizer, proxy-subset floor, template digest
- **Where:** Sampling Data (over-length filtering; 50-prompt subset); Research Instrument.
- **Edit:** characters-per-token heuristic documented for dataset-preparation
  filtering (authoritative count from the backend at run time, over-context prompts
  flagged); the 50-prompt proxy subset "allocates at least one prompt to every
  non-empty task category" (minor deviation from strict proportionality); prompt
  template recorded with SHA-256 digest per CSV batch.
- **Verification anchors:** `characters-per-token heuristic`;
  `at least one prompt to every non-empty task category`; `SHA-256 digest`.
- **status:** M.

### A-25 / A-26 — `S_min` and `Q_min(task)` defined
- **Where:** QRR paragraph; x\* constraint
  `S(x) ≥ S_min(H) and Q(x) ≥ Q_min(task)`.
- **Edit:** `S_min = 0.95 × mean unoptimized tokens/sec`;
  `Q_min(task) = 0.98 × mean unoptimized BERTScore F1 for that task category`,
  evaluated per Dolly category (BERTScore magnitudes differ by task type).
- **Verification anchors:** `S_min \= 0.95`; `Q_min(task) \= 0.98`.
- **status:** M (+C: `s_min`/`q_min` in `bopis/metrics.py`, B.4/B.5 rows in
  `bopis/runner.py`).

### A-29 / A-30 — HW-P0 / HW-B0 feasibility guards
- **Where:** Table H1 (two new model-aware guard rows); Stage 1; Table 3.2/3.3 notes.
- **Edit:** Added HW-P0 (weights + KV cache fit VRAM and VRAM + system RAM) and
  HW-B0 (KV cache for `b` concurrent slots fits VRAM) as the first Table H1 rows;
  F32 used as the canonical infeasible example (`~27 GiB`); variant set stated as
  F32/F16/Q8_0/Q4_K_M subject to the guards; on the 2 GB host `g > 0` collapses to
  `g = 0`.
- **Verification anchors:** `HW-P0`; `HW-B0`.
- **status:** M (+C: `bopis/hardware.py`,
  `tests/test_hardware.py::TestModelSizeGuards`).

### A-32 — Random search without replacement, same selection rule
- **Where:** Stage 4 (Random Search Baseline).
- **Edit:** candidates "sampled uniformly at random without replacement"; winner
  selected by "the same Pareto + retention + argmax-EIR rule used by BOPIS".
- **Verification anchor:** `sample` + `without replacement` in the Stage 4 paragraph.
- **status:** M (+C: equal budgets in `bopis/runner.py`; winner by the same Pareto
  + retention + argmax-EIR rule via `bopis/pareto.py`).

### A-34 — Behaviour when no Pareto member meets the thresholds
- **Where:** x\* selection, directly after the SRR/QRR definitions.
- **Edit:** four-rung relaxation ladder with a reported status (`optimal`,
  `relaxed_qrr_95`, `relaxed_srr_90`, `min_energy_fallback` — the last explicitly
  not a successful optimization).
- **Verification anchor:** `relaxed_qrr_95`.
- **status:** M (+C: `bopis/pareto.py` selection status,
  `tests/test_pareto.py::TestSelection`).

### A-35 — Prefill and decode energy reported separately
- **Where:** Data Analysis — J/token paragraph.
- **Edit:** "prefill and decode energy are additionally reported separately, split
  at the `prompt_ms` / `predicted_ms` boundary reported by llama.cpp", so prefill
  energy is not charged to generated tokens.
- **Verification anchor:** `prefill and decode energy are additionally reported separately`.
- **status:** M.

### A-36 — RQ 5 evidence base (partial)
- **Where:** Statement of the Problem 5; Data Analysis — surrogate reliability.
- **Edit:** leave-one-out cross-validation (see A-40) and the mandatory
  observation-noise term in σ_pred applied. The third leg — a direct repeatability
  run (one configuration, five repeats, CV of measured energy) — is defined in
  `docs/AMENDMENTS.md` but not yet present in the manuscript.
- **Verification anchor:** `The noise term σ_n² is mandatory in σ_pred`.
- **status:** M (partial) — repeatability measurement pending.

### A-39 — Table 3.3 completed; profiling dynamic
- **Where:** Table 3.3.
- **Edit:** `[fill in: …]` placeholders filled with the host figures (MX330 / i5-1135G7 /
  16 GB); added `nvmlDeviceGet*` support rows (`NOT_SUPPORTED` on host) and the fixed
  `--ctx-size 2048`; states the feasible space is derived from run-time profiling.
- **Verification anchor:** `Hardware profiles are captured dynamically at run time`.
- **status:** M.

### A-40 — Surrogate reliability on leave-one-out, LOO R² ≥ 0.85
- **Where:** Data Analysis — GP surrogate accuracy; Table 3.5 GP NPE row.
- **Edit:** MAE/NPE/R²/UCR computed on leave-one-out predictions; the "NPE below 10%"
  criterion replaced by "leave-one-out R² ≥ 0.85" alongside LOO NPE, with one-step-
  ahead figures reported separately as a pessimistic bound.
- **Verification anchors:** `leave-one-out cross-validation`; `LOO R² ≥ 0.85`.
- **status:** M (+C: `bopis/stats.py` `leave_one_out_*`/R²/UCR,
  `tests/test_stats.py`).

### A-4 — Four precision/quantization columns in Table T1
- **Where:** "Task-Informed Prior Distribution" table (~4 rows of Table T1).
- **Edit:** Prior table rows for Open QA, Closed QA, Info Extraction, General
  Instr. rewritten with four explicit values; e.g. Open QA now
  `0.15 | 0.45 | 0.20 | 0.20`, Closed QA `0.35 | 0.45 | 0.15 | 0.05`,
  General Instr. `0.20 | 0.45 | 0.21 | 0.14`.
- **Verification anchor:** row containing `0.15 | 0.45 | 0.20 | 0.20` (Open QA).

### A-8 / A-9 — Cross-platform telemetry, Mode-C GPU condition
- **Where:** "Hardware Profiling" instrument paragraph; "Data Analysis" Stage 4.
- **Edit:**
  - NVML GPU power draw (NVML + idle-subtraction) marked conditional on the GPU
    exposing a power signal.
  - `/proc` claims replaced with "per-process CPU accounting on both platforms"
    (WSL2 /proc, Windows kernel32 timers).
  - Added: a GPU with **no** power telemetry (the study host) is assigned Mode C
    — a two-component load-line (hyperplane/estimator) energy estimate.
- **Verification anchors:** `NVML_NOT_SUPPORTED`; `energy_method`.
- **status:** M (+C: estimator ladder in `bopis/monitor/estimator.py`, cross-
  platform in `bopis/monitor/platform_os.py`).

### A-14 — RAM rule uses process-visible memory
- **Where:** Table H1 Hardware Memory Feasibility Rules row (RAM) + surrounding
  text; Table B.1 note when present.
- **Edit:** Wording changed from physical/system memory to "memory visible to
  the inference process (WSL2 `/proc/meminfo` vs Windows
  `GlobalMemoryStatusEx`)".
- **Verification anchor:** `process-visible memory`; `GlobalMemoryStatusEx`.

### A-31 — Determinism claim softened
- **Where:** Research Instrument, "and temperature set to 0.0" sentence.
- **Edit:** Now reads that seed + greedy decoding makes repeated runs of the
  **same** configuration reproducible; outputs may differ **across**
  configurations because quantization/offload split alters FP reduction order.
- **Verification anchor:** `up to architectural-intrinsic`,
  `floating-point reduction order`.

### A-19 — Energy estimator formula + assumptions + limitation
- **Where:** Data Analysis — energy computation; Technical presentation of x\*.
- **Edit:** Added the two-component estimator formula (CPU+GPU load-line
  `E = Σ[(P_t − P_idle)·Δt]`, CPU term charged per-process), its assumptions,
  the EIR cancellation proof (§3.1 cancellation), and an explicit limitation
  paragraph (energy scope = GPU board only; Mode C tag `energy_scope`/
  `energy_basis="estimated_resource_allocation"`).
- **Verification anchors:** `EIR̂ = (k·E` (cancellation proof),
  `energy_scope`, `estimated_resource_allocation`.

### A-30 / A-16 — EIR/SRR/QRR as point estimates with bootstrap CI
- **Where:** Data Analysis — EIR/SRR/QRR definitions.
- **Edit:** Reports now: `estimated EIR`, `point estimate with 95% bootstrap
  confidence interval`; per-prompt ratios computed over the 500 matched prompts;
  same pattern for SRR and QRR. EIR operational threshold ≥ 15% retained as
  researcher-defined, with explicit why + literature basis.
- **Verification anchors:** `95% bootstrap confidence interval`,
  `bootstrap CI`, `estimated EIR`.

---

## B. Appendix B (docs/THESIS_WRITING2_G2.md)

### A-22 — Table B.1 reduce to three rows
- **Edit:** Rows reduced from 30 to three conditions: Unoptimized, Random
  Search, BOPIS; added `n_prompts` column.
- **Verification anchor:** `n_prompts` (Appendix B, Table B.1).

### A-23 — Table B.2 precision columns
- **Edit:** Variant column now `F32 / F16 / Q8_0 / Q4_K_M` (was FP16/INT8/…);
  CPU threads numeric.
- **Verification anchor:** `Q8_0`+`Q4_K_M` in the same row as `F32`.

### A-24 — Table B.3 remove CodeCarbon columns
- **Edit:** `CodeCarbon Validation` columns replaced with `energy_method`
  (and crosscheck via NVML energy counter), to match the amendment-required
  labeling.
- **Verification anchor:** `energy_method` (Appendix B, Table B.3).

### B.7 note
- **Edit:** Energy-mode/scope + manifest wording: the run manifest reports the
  exact `energy_method` used per record; the EUI/literature acceptance is
  relative-only.

---

## C. Code + tests (already in repo; regression-checked)

- `bopis/stats.py` — `bootstrap_ci(…, statistic="mean"|"median", n_boot, seed)`
  (std-lib only). Added for bootstrap CI of EIR/SRR/QRR and MAE/NPE/ΔE.
- `bopis/stats.py` — LOO-leave-one-out helpers used for R²/UCR/MAE_LOO
  (`leave_one_out_*`, R², UCR and UCR for the noiseless/observation-noise σ
  variants).
- `bopis/monitor/estimator.py` — two-component load-line estimator with
  epsilon-guard / NVML-ladder (Modes A/B/C).
- `bopis/optimizer.py` — iteration-0 reference added (A-33), SER uses difference
  of iteration indices (A-16 applied at SER definition level, code-level entry
  already present).
- `tests/test_stats.py` — `TestBootstrapCI`, plus LOO/R²/UCR tests; 39 tests
  pass via `python -m unittest tests.test_stats`.

### RQ 5.5 (line 318) — per-variant enumeration softened to feasible-only
- **Where:** Section 1.6/1.7 Research Questions, sub-item 5 of the performance
  RQ block (the line ending `...and output quality?`).
- **Edit:** Wording that previously read "as measured across F32, F16, Q8\_0,
  Q4\_K\_M variants" now reads "as measured across **the feasible** GGUF
  precision/quantization variants." This removes the implicit claim that F32 is
  ever measured on this host — F32 (Mistral 7B ≈ 27 GiB) is rejected by
  HW-P0/HW-B0 (2 GB VRAM / ~2.04 GiB system RAM); no F32 record is produced.
  F32 remains in the A-4-mandated Table T1 prior column (masked + renormalized)
  and in the variant-enumeration sentences with the HW-P0/HW-B0 feasibility
  qualifier.
- **Reason:** Amendment A-38 "HW-P0/HW-B0 feasibility" wording; aligns research
  questions with the instrument's feasible-variant subset.
- **Verification anchor:** `the feasible GGUF precision/quantization variants`
  on the line ending `in terms of energy consumption, inference speed, and
  output quality?` (this is the sole unconditional variant-enumeration string).
- **status:** M (manuscript-only; no code touch).

### Full suite — definitive long-timeout run
- **Run:** `python -m unittest tests.test_* (explicit module list)` — **436
  tests, OK, 3 skipped** in 127 s.
- **Reason:** The earlier "timed out at 120 s" record is a tool-side bash
  timeout artifact (default command cap), not a test failure: the full `bopis/`
  suite needs ~127 s cold. Rerun with a longer timeout per the changelog's
  own note; now on record as a clean all-pass instead of a timeout.
- **Note:** tests are not part of the thesis PDF; this record is for the code/
  tooling chapter and for the researcher's own verification log.

### Full suite — re-verified 2026-09-18 (corrects the entry above)
- **Root cause found for the import failures.** The invocation recorded above,
  `python -m unittest tests.test_<module>`, **did not work on this checkout**. It
  failed for all 14 modules with `ModuleNotFoundError: No module named
  'tests.test_stats'`. Reason: an unrelated regular package named `tests` is
  installed in `site-packages`
  (`…/Python311/Lib/site-packages/tests`), and because the project's `tests/`
  directory had no `__init__.py` it was only a *namespace* package candidate, so
  the site-packages package shadowed it. `import tests` resolved to
  site-packages, not to the repo.
- **Fix (code):** added `tests/__init__.py`, making the project's `tests/` a
  regular package. `python -m` prepends the working directory to `sys.path`, so
  the repo copy now wins. Both invocations resolve after the fix.
- **Re-verified numbers on this host (all three invocations now pass):**
  - `python -m unittest discover -s tests -t tests` (works with or without the
    fix, since it puts `tests/` itself on the path) — **436 tests, OK,
    0 skipped, 455 s**.
  - `python -m unittest discover -s tests -t .` (was failing with `ImportError:
    Start directory is not importable`; fixed by the new `__init__.py`) —
    **436 tests, OK, 0 skipped, 287 s**.
  - `python -m unittest tests.test_stats` (only works after the fix) —
    **39 tests, OK, 1.2 s**.
- **Deltas from the entry above:** the test *count* (436) reproduces exactly; the
  **"3 skipped" and "127 s" do not** — this host reports 0 skipped, and wall-clock
  varied between **287 s and 455 s** across two runs of the identical suite
  depending on machine load. Do not quote a single runtime figure; allow a
  generous timeout (≥ 600 s) rather than the 127 s previously recorded.

---

## D. Not applied (blocked / requires you)

| Item | Reason / needed action |
| :--- | :--- |
| Re-profile under clean boot | Needs a reboot + fresh `bopis/profile.py` run to refresh HW-P0 blocks and the Mode A/C matrix. Currently every config is rejected by HW-P0 on this host; the numbers in Figure 3.5 are the stale pre-boot values. **Do before final hand-in.** |
| Verify cited paper figures (Ma et al. 15% / Jensen, 2026 figures) | Requires web access to the cited papers; I did not fabricate the numbers. Check the exact %/figure numbers yourself or provide sources. |
| **Citation audit of `docs/RRL.md` — done 2026-09-18, corrections applied** | Five records were wrong; `docs/RRL.md` now carries inline `VERIFIED`/`NO SUCH PAPER` comments. **Two do not exist:** "Pham, H., Qian, C., Wang, T., & Yu, Y. (2020), *Problems and opportunities in neural network robustness and reproducibility*, arXiv:2206.04236" (no match on arXiv or Scholar; a 2020 paper cannot hold a 2022 ID) and "Xu, Z., et al. (2023), *Evaluating quantization-induced energy reduction in local LLM deployment*" (no match, no ID given). **Three were misattributed:** Efron & Tibshirani (1993) carried Davison & Hinkley's DOI (`10.1017/CBO9780511802843` resolves to *Bootstrap Methods and their Application*, CUP 1997); "Process-level power estimation in VM-based systems" is Colmant, Kurpicz, Felber, Huertas, Rouvoy & Sobe, EuroSys **'15** (`10.1145/2741948.2741971`), not "Lim, Rawson & Ballew, EuroSys '14"; and SparseGPT (a *pruning* paper) was cited for a *quantization* claim, replaced by GPTQ (Frantar, Ashkboos, Hoefler & Alistarh, ICLR 2023, arXiv:2210.17323). Also: the summary table's "INT8 Benefits" role is obsolete — A-23 replaced FP16/INT8 with F32/F16/Q8_0/Q4_K_M. **Verified correct and untouched:** Jones et al. (1998), Gerganov (2023), Zitzler & Thiele (1998), Emmerich et al. (2005), Arlot & Celisse (2010), Agrawal et al. (OSDI '24), Zhong et al. (DistServe), Stojkovic et al. (2408.00741), Rotem et al. (2012), Narayanan et al. (SC21). All 18 arXiv IDs in `ENERGY_ESTIMATOR_AND_ML_BRIEF.md` §4 resolve to real papers with matching titles. |
| **EnerInfer accuracy figures were a misread — corrected in the brief** | `ENERGY_ESTIMATOR_AND_ML_BRIEF.md` §3.2 Route 1 quoted EnerInfer as reporting "6.5% / 12% / 9.7% / 5.4% deviation" as *prediction accuracy*. Those are not accuracy figures: the paper reports improving energy **efficiency** by up to 65% / 12% / 24% on phones / laptop / dev board. Row removed. **This weakens the MAPE ≤ 15% justification** — the remaining four figures backing it are all still unverified against full texts. Either verify two of them or present the threshold as researcher-defined. |
| Full `unittest discover` | **Resolved, but for a different reason than recorded.** Re-checked 2026-09-18: the explicit-module invocation was failing outright (site-packages `tests` package shadowing the repo's `tests/`), now fixed by adding `tests/__init__.py`. Verified full suite: **436 tests, OK, 0 skipped, 455 s** via `discover -s tests -t tests`. The "3 skipped / 127 s" figures do not reproduce here. See the re-verification entry in Section C. |
| A-36 repeatability measurement (one config, five repeats, CV of measured energy) | Only the LOO + noise-σ legs of A-36 landed in the manuscript; the repeatability run has no manuscript text yet. Add a short Data-Analysis or Scope sentence once the experiment is run. |
| A-38 — BERTScore baseline rescaling | Commit `d745a49` lists A-38, and `docs/AMENDMENTS.md` A-38 is "Enable BERTScore baseline rescaling", but no "baseline rescaling / absolute F1 deltas" text is present in `docs/THESIS_WRITING2_G2.md`. Confirm whether the rescaling claim is wanted and add the sentence (it also motivates `QRR ≥ 98%`). |
| A-37 | Listed in the `d745a49` message but there is **no A-37 entry in `docs/AMENDMENTS.md`**. Confirm which amendment A-37 refers to before trusting the commit label. |

---

## E. Verification anchors that were NOT touched (may already be correct — confirm against PDF)

- EI formula in Chapter 3 (BOPIS): **superseded by A-2** — the manuscript previously
  printed the maximisation form and has now been corrected to the minimisation form
  (`EI(x) = (f(x⁺) − μ(x))·Φ(Z) + σ(x)·φ(Z)`), which matches `bopis/acquisition.py`.
  See the A-2 entry in Section A.
- `Energy per token = E / N_generated` and the J/token normalisation text
  (superseded by A-35 for the prefill/decode split).

---

*Generated by the research assistant; send this file together with the PDF to have
another model verify the manuscript snapshot against `AMENDMENTS.md`.*
