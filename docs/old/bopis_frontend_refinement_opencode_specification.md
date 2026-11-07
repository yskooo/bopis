# BOPIS Tool Refactoring Specification & Panel Revision Blueprint

## 1. Executive Summary: Why the Panelists Got Confused & How to Fix It

When panelists see an interface dominated by a single chat window, they naturally assume the thesis product is a **chatbot**. However, as established in Chapters 1–3 of your manuscript, **BOPIS** (*Bayesian Optimization and Pareto-Based Intelligent Configuration Selection*) is an **offline pre-deployment configuration optimization engine** for `llama.cpp`.

To eliminate this confusion while directly satisfying every comment in your **BSCS Thesis Tool Revision Matrix**, you do not need to throw away your existing Python tool or chatbot code. Instead, you must **reframe and restructure the UI into a 3-Tab Engineering Workbench** where the "chatbot" is demoted from being the main app into **Tab 2: A Side-by-Side Live Telemetry Verification Arena**.

| Feature | Old "Chatbot" UI (Why Panelists Got Confused) | New "BOPIS Workbench & Arena" UI (The Fix) |
| :--- | :--- | :--- |
| **App Title / Header** | Generic LLM Chat / Local Chatbot | **BOPIS: Offline Configuration Optimizer & Live Inference Telemetry Workbench** |
| **Default Landing Screen** | Opens directly to a chat prompt | Opens to **Tab 1: Offline Optimization Engine (BO + Pareto Visualizer)** |
| **Primary Visual Ratio** | 90% chat bubbles, 10% settings | 30% side-by-side LLM text responses, **70% Energy, Speed, Quality, & Hardware Telemetry** |
| **Role of User Prompt** | Looks like you are testing the LLM's conversational ability | Explicitly framed as **"Live Workload Verification Input"** with real-time Task Classification |
| **Visibility of Offline BOPIS** | Hidden in backend scripts/CSV files | Interactive step-by-step playback of `Table B.7` (GP Surrogate, Expected Improvement, Pareto Front) |
| **Visibility of System Priority** | Unclear what BOPIS prioritizes | Explicit visual constraint gates: **Maximize $\text{EIR}$** subject to **$\text{SRR} \ge 95\%$** and **$\text{QRR} \ge 98\%$** |

---

## 2. Architecture & UI Specification for OpenCode

> **How to use this section:** You can copy the instructions below and pass them directly as a prompt or `SPEC.md` file into **OpenCode** so it can inspect your existing Python repository and refactor your current frontend without breaking your backend `llama.cpp`, `NVML`, and `/proc` scripts.

### Global Persistent Header & Sidebar (Visible Across All Tabs)
1. **Application Banner:**
   * **Title:** `BOPIS: Bayesian Optimization & Pareto-Based Intelligent Configuration Selection`
   * **Subtitle:** `Offline Pre-Deployment Optimizer & Live Local LLM Telemetry Sandbox (Base Model: Qwen2.5-1.5B-Instruct via llama.cpp)`
2. **System Priority & Decision Rule Banner (Addresses Panel Comment: *"Show the priority of the system"*):**
   * Display a prominent, persistent status bar at the top of the interface stating the formal mathematical objective from Chapter 3 (p. 70, 74):
     $$\text{Primary Objective: } x^* = \arg\max_{x \in PF} \text{EIR}(x) \quad \text{subject to} \quad \text{SRR}(x) \ge 95\% \text{ (Speed Floor)} \quad \text{and} \quad \text{QRR}(x) \ge 98\% \text{ (Quality Floor)}$$
   * Include visual badges showing the priority hierarchy:
     * **Priority 1 (Optimization Target):** Minimize Energy Consumption ($\text{Joules}$, $\text{J/token}$) $\rightarrow$ Maximize Energy Improvement Ratio ($\text{EIR} > 0\%$, substantial flag $\ge 15\%$).
     * **Priority 2 (Hard Constraint Gate):** Retain Inference Speed ($\text{SRR} \ge 95\%$ of Unoptimized Default $\text{tokens/sec}$).
     * **Priority 3 (Hard Constraint Gate):** Retain Semantic Output Quality ($\text{QRR} \ge 98\%$ of Unoptimized Default $\text{BERTScore F1}$).
     * **Supporting Telemetry:** Hardware Resource Utilization ($\text{CPU\%}$, $\text{GPU\%}$, $\text{RAM MB}$, $\text{VRAM MB}$).
3. **Sidebar Hardware Profile & Constraint Inspector (Stage 0 of Pipeline):**
   * Read and display the detected host hardware specs from your existing profiling module:
     * **CPU:** 11th Gen Intel Core i5-1135G7 @ 2.40 GHz
     * **Discrete GPU:** NVIDIA GeForce MX330 (2 GB GDDR5 VRAM, 64-bit bus — Mode C Power Estimation)
     * **System RAM:** 16 GB (~15.78 GiB available)
     * **Idle Baseline Power ($P_{\text{idle}}$):** Display the recorded 60-second pre-inference baseline idle power in Watts/milliwatts.
   * Show the active **Hardware Constraint Rules (Table H1, p. 62–64)** applied to filter the search space into $X_{\text{feasible}} = T \times B(HW) \times P(HW) \times G(HW) \times C(HW)$:
     * Token Length ($t$): `{128, 256, 512, 1024}`
     * Batch Size ($b$): `{1, 2, 4, 8}`
     * GGUF Precision Variant ($p$): `{F16, Q8_0, Q4_K_M}` (`F32` excluded by VRAM guard)
     * GPU Layers Offloaded ($g$): `{0, 14, 28, All}` (Note on UI: highlight the empirical finding from p. 17 that on the 64-bit bus MX330, `g=0` avoids the memory-bandwidth bottleneck)
     * CPU Threads ($c$): `{2, 4, 8}`

---

### Tab 1: Offline Optimization Engine & Calibration Replay
**Purpose:** Directly answers the panelist requirement: *"Explain how BOPIS optimizes energy use"* and *"State why BOPIS is more expensive than other models (and runs once offline before deployment)"*.

Because running 50 Dolly prompts across $N$ Bayesian Optimization iterations takes hours offline, **Tab 1 must load your existing calibration log CSV (`Table B.7`, p. 103–104)** and allow the presenter to **step through the optimization iterations interactively** via a slider (`Iteration 1` to `Iteration N`).

#### Required Components in Tab 1:
1. **Offline Cost vs. Production Benefit Callout Box:**
   * Display a clear metric card explaining why BOPIS is computationally expensive *only once* during offline pre-deployment calibration (evaluating the 50-prompt proxy subset across 10 random seeds + $N-10$ BO iterations), and how the resulting $x^*$ configuration is locked in statically for production with **zero runtime optimization overhead**.
2. **Interactive Iteration Slider (`k = 1 ... N`):**
   * Stepping the slider from Iteration 1 to $N$ dynamically updates the charts and tables below to show what BOPIS knew at Iteration $k$:
     * **Iterations 1–10:** Labeled as `Stage 1: Task-Informed Random Seed Initialization`.
     * **Iterations 11–N:** Labeled as `Stage 2: Gaussian Process Surrogate + Expected Improvement (EI) Guided Search`.
3. **Chart 1: Gaussian Process (GP) Surrogate Accuracy & Convergence Curve:**
   * Plot actual measured energy $E_{\text{measured}}(x_i)$ vs. GP predicted mean $\mu(x_i)$ with shaded uncertainty bands $\pm \sigma(x_i)$ across iterations.
   * Plot the monotonically non-increasing **Best Energy Found So Far ($E^*_k$)** and per-iteration improvement **$\Delta E_k = E^*_{k-1} - E^*_k$** (comparing BOPIS convergence against Random Search convergence).
4. **Process Validation Metric Cards (Stage 5 Validation, p. 85–88):**
   * **GP Mean Absolute Error ($\text{MAE}_{\text{GP}}$):** Average $|\mu(x_i) - E_{\text{measured}}(x_i)|$ in Joules.
   * **Normalized Prediction Error ($\text{NPE}$):** Show value and badge (`PASS: < 10% Reliability Threshold`).
   * **Leave-One-Out Cross-Validation ($R^2_{\text{LOO}}$):** Show value and badge (`PASS: >= 0.85 Threshold`).
   * **Sample Efficiency Ratio ($\text{SER} = k^*_{\text{RandomSearch}} / k^*_{\text{BOPIS}}$):** Show badge (`PASS: SER > 1.0 — BOPIS found optimal configuration in fewer evaluations than Random Search`).
5. **Chart 2: Pareto Front & Constrained Decision Filter (Stage 2 of BOPIS):**
   * Render an interactive 2D/3D scatter plot of all evaluated configurations from `Table B.7` plotting **Energy (J/token)** vs. **Inference Speed (tokens/sec)** vs. **Output Quality (BERTScore F1)**.
   * Provide 3 visual checkboxes/toggles so panelists can watch BOPIS narrow down the candidates live:
     * `[x] Step 1: Highlight Non-Dominated Pareto Front (PF)`
     * `[x] Step 2: Apply Retention Threshold Gates (SRR >= 95% AND QRR >= 98%)` — visually gray out or cross out low-energy configurations that failed the speed or quality floor!
     * `[x] Step 3: Select Final Optimal Configuration x* = argmax EIR(x)` — highlight the winning configuration $x^*$ in bright green and display its exact parameter vector $(t^*, b^*, p^*, g^*, c^*)$.

---

### Tab 2: Live A/B/C Inference Telemetry Arena (Refactoring Your Existing Chatbot)
**Purpose:** Directly answers the panelist requirements: *"Show an actual use case where a real user inputs the question"*, *"Indicate that inputs are not predefined to avoid showing bias"*, and *"Show the priority of the system"*.

**Crucial Instruction for OpenCode:** Refactor the existing single-window chatbot UI into a **3-Column Comparative Inference Arena**. Do not use conversational chat bubbles (`User` / `Assistant` history). Instead, treat every prompt entered by a panelist as a **Live Workload Benchmark Run**.

#### Required Components in Tab 2:
1. **Top Workload Input Bar:**
   * **Arbitrary User Prompt Input:** A text area labeled `"Enter Any Custom User Prompt (Live Unbiased Workload Test)"` where a panelist can type any question on the spot.
   * **Dataset Sampler Button:** A secondary button labeled `"Sample Random Unseen Prompt from Databricks Dolly 15k"` (loads a random prompt + its ground-truth reference answer from your Dolly 15k JSON/CSV so live BERTScore F1 can also be computed against a reference).
2. **Real-Time Task Classification & Prior Strip (Executes Immediately on Prompt Input):**
   * Hook up your existing **Task Classification Module** (the 42-rule lexical classifier and/or Multinomial Naive Bayes classifier from Chapter 3, p. 63–66).
   * As soon as a prompt is typed or sampled, display anlive diagnostic strip before inference even starts:
     * **Detected Task Category:** e.g., `Closed QA`, `Summarization`, `Open QA`, `Brainstorming`, etc.
     * **Quantization Sensitivity Tier:** `High`, `Medium`, or `Low` (from Table T1, p. 66).
     * **Task-Informed Precision Prior $P(\text{precision} \mid \text{task})$:** Display the probabilities $P(\text{F16})$, $P(\text{Q8\_0})$, $P(\text{Q4\_K\_M})$.
     * **Active Quality Floor ($Q_{\text{min}}$):** `QRR >= 98% required`.
3. **Execution Mode Selector & Run Button:**
   * Primary button: `"⚡ Run Live 3-Way Comparative Inference (Default vs. Random Search vs. BOPIS x*)"`.
   * Because all three configurations run on the **same single host machine**, execute `llama.cpp` sequentially across the 3 locked configurations for the user's prompt while streaming or updating each column one by one:
     * **Column 1: Unoptimized Default Reference** — Locked to `(t=1024, b=1, p=F16, g=All, c=Default)` (Denominator for $\text{EIR}$, $\text{SRR}$, $\text{QRR}$).
     * **Column 2: Random Search Baseline ($x_{\text{RS}}$)** — Locked to the configuration selected by your Random Search baseline run.
     * **Column 3: BOPIS Optimal Configuration ($x^*$)** — Locked to the Pareto-optimal configuration selected by BOPIS.
4. **Layout Inside Each of the 3 Columns (30% Text / 70% Telemetry):**
   * **Header & Config Vector Badge:** Clear title and monospace badge showing `x = (t, b, p, g, c)`.
   * **Generated LLM Response Box (Compact, Fixed-Height Scrollable Box — 30% of vertical space):** Shows the text generated by `Qwen2.5-1.5B-Instruct` under that configuration (`temperature = 0.0` greedy decoding).
   * **Primary Objective & Retention Gate Cards (70% of vertical space):**
     * **Total Energy (J) & Normalized Energy (J/token):** Marked as *Mode C Estimated* (p. 84). In the **BOPIS ($x^*$)** column, display **two explicit comparison badges** to satisfy all panelist interpretations:
       1. `EIR vs. Default: +XX.X% Energy Saved (PASS >= 15% Substantial Flag)`
       2. `Energy Saved vs. Random Search: +XX.X% Lower J/token`
     * **Inference Speed (tokens/sec) & Wall-Clock Latency (s):** Show `SRR = XX.X% of Default` with a green `PASS (>= 95%)` or red `FAIL (< 95%)` gate badge.
     * **Output Quality (BERTScore F1):**
       * If the prompt was sampled from Dolly 15k, compute BERTScore F1 against the Dolly reference output and show `QRR = XX.X%` (`PASS >= 98%`).
       * If the panelist typed a custom free-form prompt (no ground-truth reference in Dolly 15k), compute **Self-Reference BERTScore F1** comparing the `Random Search` and `BOPIS x*` outputs directly against the `Unoptimized Default (F16)` output to prove that quantization/configuration tuning preserved semantic meaning (`>= 98% similarity`).
     * **Hardware Resource Utilization Gauges (`/proc` and `NVML`):** Progress bars or metric readouts for **CPU Utilization (%)**, **GPU Utilization (%)**, and **System RAM / VRAM (MB)**.

---

### Tab 3: 500-Prompt Benchmark Results, Per-Variant Breakdown & Statistical Proof
**Purpose:** Directly answers Statement of the Problem (SOP) 1, 2, and 3 (pp. 14–15) and the panelist comment: *"Identify the specific parameters in which BOPIS performs better than random search, and the strengths that support this."*

#### Required Components in Tab 3:
1. **Two-Way Statistical Comparison Table (`Table B.8`, p. 104–105):**
   * Load your consolidated 500-prompt evaluation CSV (`Tables B.3–B.6`).
   * Display Mean, Standard Deviation (SD), Min, and Max for **Random Search Baseline** vs. **BOPIS-Optimized ($x^*$)** (plus **Unoptimized Default** as a reference column) across all dependent variables:
     * Total Energy ($\text{J}$) & Normalized Energy ($\text{J/token}$)
     * Inference Speed ($\text{tokens/sec}$)
     * Output Quality ($\text{BERTScore F1}$ with baseline rescaling enabled)
     * Resource Utilization ($\text{CPU\%}$, $\text{GPU\%}$, $\text{Memory MB}$)
   * Display the **Wilcoxon Signed-Rank Test $p$-values ($\alpha = 0.05$)** for every row with a clear `Statistically Significant (p < 0.05)` badge.
2. **Per-Variant Performance Breakdown (SOP 3e, p. 15):**
   * Render grouped bar charts comparing **`F16` vs. `Q8_0` vs. `Q4_K_M`** across Energy ($\text{J/token}$), Speed ($\text{tokens/sec}$), BERTScore F1, and Memory (MB) so panelists can visually see how GGUF quantization interacts with hardware parameters.
3. **Per-Task Category Breakdown (8 Dolly 15k Categories):**
   * Show a table or chart of `Random Search` vs. `BOPIS` grouped by the 8 Databricks Dolly 15k task categories (`Open QA`, `Closed QA`, `Summarization`, `Info Extraction`, `Classification`, `Creative Writing`, `Brainstorming`, `General Instr.`) demonstrating that $\text{QRR} \ge 98\%$ is respected across high-sensitivity and low-sensitivity tasks alike.

---

## 3. Exact Copy-Paste Text for Your Tool Revision Matrix

In your `BSCS-Thesis-Tool-Revision-Matrix-AY-2026-2027.docx`, the **ACTION TAKEN by the Proponent/s** column for the **Tool (Software/Hardware)** row (Page 3) is currently blank. Once your team applies the UI restructuring above, copy and paste the text below directly into that table cell:

| Panel Comment (Tool Section) | Action Taken by the Proponents (Copy-Paste into Matrix) |
| :--- | :--- |
| **Explain how BOPIS optimizes energy use.** | Refactored the frontend from a single-screen chat interface into a **3-Tab BOPIS Optimization Workbench**. Added **Tab 1 (Offline Optimization Engine & Calibration Replay)**, which visually steps through the 6-stage Bayesian Optimization loop (`Table B.7`, pp. 59–61, 103), plotting the Gaussian Process surrogate predictions ($\mu(x)$ and uncertainty $\sigma(x)$), Expected Improvement acquisition ($\text{EI}(x)$), validation metrics ($\text{MAE}$, $\text{NPE} < 10\%$, $\Delta E$, and $\text{SER}$), and the 3D Pareto Front non-dominated sorting process that isolates the lowest-energy configuration $x^*$. |
| **Show an actual use case where a real user inputs the question.** | Replaced the standalone chatbot view with **Tab 2 (Live A/B/C Inference Telemetry Arena)**. A real user or panelist can input any arbitrary custom question (or sample an unseen prompt from Databricks Dolly 15k), watch the **Task Classification Module** dynamically classify the prompt's task type and sensitivity tier in real time (proving inputs are not predefined or biased), and execute a side-by-side live `llama.cpp` comparison across the **Unoptimized Default**, **Random Search Baseline**, and **BOPIS-Optimized Configuration ($x^*$)**. |
| **Show the priority of the system (e.g. energy consumption, resource utilization, etc.)** | Added a persistent **System Priority & Constraint Gate Banner** and live telemetry cards across the workbench that explicitly enforce BOPIS's mathematical hierarchy (pp. 60, 70, 87–88): **Priority 1** maximizes the **Energy Improvement Ratio ($\text{EIR} = \arg\max \text{EIR}(x)$)** in Joules and $\text{J/token}$; **Priority 2 & 3** enforce strict pass/fail retention gates for **Inference Speed ($\text{SRR} \ge 95\%$)** and **Output Quality ($\text{QRR} \ge 98\%$ BERTScore F1)**; and **Supporting Telemetry** displays real-time $\text{CPU\%}$, $\text{GPU\%}$, and $\text{RAM (MB)}$ via `/proc` and `NVML`. |

---

## 4. Manuscript & Tool Consistency Checklist (Fix Before Defense!)

While cross-checking your 107-page manuscript against the Tool Revision Matrix, three minor discrepancies appeared in Chapter 3 that you should align in both your manuscript and your tool UI so panelists cannot fault your consistency:

1. **Table 3.3 Hardware & Model Typos (Manuscript Page 75):**
   * **Issue 1:** Chapters 1–3 state that your base model is **`Qwen2.5-1.5B-Instruct`** (and Page 15 explicitly explains replacing `Qwen2.5-1.5B` with `Qwen2.5-1.5B-Instruct`). However, in **Table 3.3 (p. 75)**, the `LLM Model` row says **`Qwen2.5-0.5B-Instruct`**.
   * **Issue 2:** In **Table 3.3 (p. 75)**, the labels for **GPU** and **CPU** are swapped (`GPU` says `11th Gen Intel Core i5-1135G7` and `CPU` says `NVIDIA GeForce MX330`).
   * **Action:** Fix Table 3.3 in the manuscript and ensure the Sidebar Hardware Profile in your tool displays `CPU: Intel Core i5-1135G7`, `GPU: NVIDIA GeForce MX330 (2GB GDDR5)`, and `Model: Qwen2.5-1.5B-Instruct`.
2. **Denominator of $\text{EIR}$, $\text{SRR}$, and $\text{QRR}$ in Table 3.5 (Manuscript Page 92):**
   * **Issue:** On pages 53, 60, 69, 87, and 89, your equations define $\text{EIR}$, $\text{SRR}$, and $\text{QRR}$ using the **Unoptimized Default Configuration (`E_default`, `S_default`, `Q_default`)** as the denominator. However, in **Table 3.5 (p. 92)**, the "Applied To" column says *"BOPIS versus Random Search"*.
   * **Action:** In your tool's Tab 2 and Tab 3 telemetry cards, display **both** metrics side by side:
     * `EIR (vs. Unoptimized Default): +XX.X%`
     * `Relative Energy Improvement (vs. Random Search): +XX.X%`
     This guarantees that whichever baseline a panelist asks about during the demonstration, the exact percentage is immediately visible on screen.
3. **NullHypothesis Wording in Statistical Treatment (Manuscript Page 90 vs. Page 91):**
   * **Issue:** On Page 89–90, you correctly state that the Wilcoxon signed-rank test compares the **two** configurations (Random Search vs. BOPIS), while at the top of Page 91, a leftover paragraph says *"among the three configurations"*.
   * **Action:** Remove the duplicate leftover sentence at the top of Page 91 so it strictly matches your two-way Wilcoxon signed-rank test design.