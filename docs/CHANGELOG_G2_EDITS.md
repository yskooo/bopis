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

---

## D. Not applied (blocked / requires you)

| Item | Reason / needed action |
| :--- | :--- |
| Re-profile under clean boot | Needs a reboot + fresh `bopis/profile.py` run to refresh HW-P0 blocks and the Mode A/C matrix. Currently every config is rejected by HW-P0 on this host; the numbers in Figure 3.5 are the stale pre-boot values. **Do before final hand-in.** |
| Verify cited paper figures (Ma et al. 15% / Jensen, 2026 figures) | Requires web access to the cited papers; I did not fabricate the numbers. Check the exact %/figure numbers yourself or provide sources. |
| Full `unittest discover` | **Resolved.** Explicit-module full suite (all 14 `tests.test_*` modules): **436 tests, OK, 3 skipped** in 127 s. The earlier 120 s "timeout" was my bash tool's default wall-clock (120 s) cutting off a run that legitimately needs ~127 s — not a test failure. |

---

## E. Verification anchors that were NOT touched (may already be correct — confirm against PDF)

- EI formula in Chapter 3 (BOPIS) was already correct in the manuscript
  (`EI(x) = (μ(x) − f(x⁺))·Φ(Z) + σ(x)·φ(Z)`, minimisation form). The code in
  `bopis/acquisition.py` uses the same form. No edit applied here.
- `Energy per token = E / N_generated` and the J/token normalisation text.

---

*Generated by the research assistant; send this file together with the PDF to have
another model verify the manuscript snapshot against `AMENDMENTS.md`.*
