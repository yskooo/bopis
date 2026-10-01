# BOPIS — Defense Audit Report

**Generated:** 2026-09-30  
**Purpose:** Pre-defense audit — which claims pass, which fail, and what to do tonight.

---

## Quick-read verdict

| # | Check | Status | Action |
|---|-------|--------|--------|
| 1 | EI sign (minimising energy) | ✅ PASS | Nothing |
| 2 | GP fitted by log-marginal-likelihood | ✅ PASS | Nothing |
| 3 | UCR metric present and ≥ 0.95 | ✅ PASS (0.967) | Nothing |
| 4 | Classifier accuracy on held-out test | ✅ PASS (69.7%) | Nothing |
| 5 | Test suite passes (517 tests) | ✅ PASS | Re-run tonight to confirm |
| 6 | All recorded runs are `sim` backend | ⚠️ KNOWN LIMIT | Volunteer it; do not hide it |
| 7 | Feasibility guard uses Qwen MODEL_LADDER | ✅ PASS | `hardware.MODEL_LADDER` built from Qwen `config_space.MODELS` |
| 8 | BERTScore never called on real runs | ⚠️ KNOWN LIMIT | Volunteer it |
| 9 | QRR = 98.7% comes from simulator | ⚠️ KNOWN LIMIT | Label it "simulator value" in demo |
| 10 | Port 8080 hardcoded in bopis.html | ✅ PASS (documented) | Nothing |
| 11 | Energy figure labelled `~estimated (Mode C)` | ✅ PASS | Point at the label in demo |
| 12 | BOPIS vs Random comparison present | ✅ PASS | Use it in Beat 7 |

---

## Detailed findings

### ✅ PASS — Claim you can defend

**1. EI sign is correct.**
`acquisition.py:72` — `improvement = f_best - mu - xi`. Minimising energy means improvement is a *decrease*. The code was always correct; the manuscript was corrected under amendment A-2. If a panel member challenges it, point at line 72 and the worked example in ML_ELEMENTS.md §4A.

**2. GP hyperparameters are learned, not hand-tuned.**
`gp.py:334` — `GaussianProcess.fit()` runs log-space grid + multi-start Nelder–Mead to maximise log marginal likelihood. Achieved LML: **−12.41**. Fitted values: σ_f² = 3.993, ℓ = 0.937, σ_n² = 0.00157. The same contract as `sklearn.GaussianProcessRegressor`; `tests/test_gp.py::TestAgainstSklearn` verifies equivalence.

**3. Surrogate reliability metrics all pass.**
LOO cross-validated: R² = **0.980**, NPE = **3.77%** (threshold < 10%), MAE = **4.70 J**, UCR = **0.967** (threshold ≥ 0.95). UCR is the key metric — it proves the uncertainty σ(x) is calibrated. A model with high R² but low UCR would break Expected Improvement.

**4. Classifier: 69.7% on a clean held-out test set.**
Stratified 70/15/15 split, seed 20260101. Test set was touched exactly once, after α was chosen on validation. Rule baseline: 48.2% (+21.4 points). Tier accuracy (the operationally relevant figure): **79.2%**. Cross-implementation: 155/155 Python ↔ JS agree.

**5. 517 tests, OK.**
`python -m unittest discover -s tests -t .` — re-run this tonight before sleeping.

**6. Energy ratio argument (bias cancels).**
Both BOPIS and default see the same Mode C estimator. If estimator bias is multiplicative `Ê = k·E`, then `EIR = (k·E_def − k·E_bop)/(k·E_def) = EIR`. The ±30% nameplate uncertainty does not propagate into the *ratio*. State this proactively.

---

### ✅ Gap 3 is NOT a failure — code already uses Qwen

`MISTRAL_7B_INSTRUCT_V03` in `hardware.py` is only a **legacy named constant** kept for backwards compatibility. When `--model-aware` is passed, `cli.py` line 81 passes `hardware.MODEL_LADDER` to `feasible_space()` — and `MODEL_LADDER` is built directly from `config_space.MODELS`, which is the full **Qwen2.5-Instruct ladder** (0.5B, 1.5B, 3B, 7B), defined at `config_space.py` lines 89–100.

**If `FEASIBLE SPACE 0 / 768` appears at the defense**, the cause is **low available RAM** (other apps consuming memory before profiling), not the wrong model. Fix: close Chrome and VS Code before the panel walks in, or reboot. After a clean boot you should see ~13 GiB free and a non-zero feasible count.

---

### ⚠️ KNOWN LIMITS — Volunteer these; don't wait to be asked

**Gap 1: Every run is simulated.**
All `runs/` directories carry `backend: "sim"`. No `selection.csv` from real hardware exists yet. The defensible claim is: *"the instrument is complete and verified — we demonstrate that it works, not that we have measured energy results."*

**Gap 2: BERTScore never called on real runs.**
`get_scorer()` is never invoked outside `bopis/quality/`. On a real run `quality_f1` stays `None`. The 98.7% QRR in the dashboard is the simulator's analytic value. Say: *"Quality retention is measured by BERTScore on real runs; the simulator provides an analytic proxy for this demonstration."*

**Gap 4: GPU-layer dimension may be degenerate.**
Every GPU offload setting is slower than CPU-only (3.5× measured). The optimizer likely drives `g` to 0. Answer: *"We measured it and CPU-only wins on this hardware — that is the answer the optimizer found."*

**Gap 5: No repeatability run.**
Without five repeat measurements of one configuration, you cannot quantify noise vs. signal. Answer honestly: *"This is the next step after the defense."*

---

## Tonight's priority order

1. **Reboot** — ensures ~13 GiB free RAM so `FEASIBLE SPACE` shows non-zero
2. **Re-run tests** — `py -3 -m unittest discover -s tests -t .` — confirm 517 pass
3. **Copy dashboard_data.js** — `Copy-Item runs\20260909T172504Z\dashboard_data.js .\dashboard_data.js`
4. **Dry-run the full demo** — `.\demo.ps1` with the server — record it as backup
5. **Rehearse** the 60-second story and all 5 Q&As from DEMO_RUNBOOK §3–4

---

## The one-sentence defense claim

> "BOPIS is a complete, tested instrument for finding energy-efficient LLM server configurations using Bayesian Optimization. We demonstrate the instrument on a simulated workload; the real measurement campaign is the next step."

Do not claim measured energy. Do not claim final results. Everything else flows from that anchor.
