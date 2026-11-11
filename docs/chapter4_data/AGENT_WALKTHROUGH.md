# AI Agent Guide & Data Walkthrough for Chapter 4 and Chapter 5

> [!CAUTION]
> **CRITICAL DIRECTIVE FOR ALL AI AGENTS (CURSOR, COPILOT, CLAUDE, GEMINI, CHATGPT, ETC.):**
> **DO NOT EDIT, OVERWRITE, REGENERATE, OR DELETE ANY CSV FILES IN THIS DIRECTORY.**
> All files in `docs/chapter4_data/` represent **empirically measured benchmark data** collected from physical hardware runs (AMD Ryzen 5 5600G + NVIDIA GeForce GTX 1660 SUPER 6GB, direct NVML hardware energy counters, `is_measured: true`).
> Re-running scripts or modifying these files will destroy the authoritative empirical thesis results. Treat this entire folder as **STRICTLY READ-ONLY**.

---

## 1. Thesis Context & Overview

* **Project Title:** BOPIS (Bayesian Optimization and Pareto-Based Intelligent Configuration Selection for Energy-Efficient Local LLM Inference)
* **Goal:** Intelligently optimize runtime parameters (token length $t$, batch size $b$, quantization $p$, GPU offload $g$, and CPU threads $c$) to minimize energy consumption (Joules) while maintaining inference speed (tokens/sec) and generation quality (BERTScore F1).
* **Workstation Hardware:**
  * **CPU:** AMD Ryzen 5 5600G (6 Cores / 12 Threads, ~3.9 GHz)
  * **GPU:** NVIDIA GeForce GTX 1660 SUPER (6 GB GDDR6 VRAM)
  * **RAM:** 16 GB DDR4
  * **Host OS / Runtime:** Windows 11 / llama.cpp / Python 3.13.2
  * **Energy Monitoring:** Physical hardware counters via NVIDIA NVML (Mode A, physical measurement).

---

## 2. Directory Map ("Where is Where")

Use this lookup table to find the exact CSV file needed when writing or verifying each section of Chapter 4, Chapter 5, and Appendix B:

| File Name | Description & Manuscript Placement | Key Metrics / Columns Contained |
| :--- | :--- | :--- |
| **`B1_performance_comparison.csv`** | **Manuscript Table 4.1 & Appendix B.1:** High-level performance comparison across Unoptimized, Random Search, and BOPIS. | Total Energy (J), Joules/Token, Speed (tok/s), BERTScore F1, CPU%, GPU%, RAM MB. |
| **`B2_configuration_evaluation.csv`** | **Manuscript Table 4.2 & Appendix B.2:** Pre-experiment evaluation across tested parameter grid. | $t, b, p, g, c$, Energy, Speed, BERTScore F1. |
| **`B3_energy_per_prompt.csv`** | **Appendix B.3:** Prompt-by-prompt energy records across all 500 test prompts. | Prompt ID, Random Search Energy (J), BOPIS Energy (J), Delta > 15% Flag. |
| **`B4_speed_per_prompt.csv`** | **Appendix B.4:** Prompt-by-prompt inference speed across all 500 test prompts. | Prompt ID, Random Search Speed, BOPIS Speed, Speed Gate ($S \ge S_{\min}$). |
| **`B5_quality_per_prompt.csv`** | **Appendix B.5:** Prompt-by-prompt semantic quality (BERTScore F1) across 500 prompts. | Prompt ID, Task Category, Random Search F1, BOPIS F1, Quality Gate ($Q \ge Q_{\min}$). |
| **`B6_resource_utilization_per_prompt.csv`** | **Appendix B.6:** Hardware telemetry across all 500 test prompts. | Prompt ID, CPU %, GPU %, RAM (MB) for both Random Search and BOPIS. |
| **`B7_bopis_calibration_log.csv`** | **Manuscript Table 4.3 & Appendix B.7:** The 30-iteration Bayesian Optimization search trajectory. | Iteration (10 random seeds + 20 BO iterations), Config evaluated, $E$ measured, $S$, $Q$, GP $\mu(x)$, GP $\sigma(x)$, $EI(x)$, Pareto front membership. |
| **`B8_summary_comparison.csv`** | **Manuscript Table 4.4 & Appendix B.8:** Complete descriptive statistics and Wilcoxon Signed-Rank Test results. | Dependent variables, Mean, SD, Min, Max, and Wilcoxon $p$-values comparing RS vs BOPIS. |
| **`B9_xstar_selection.csv`** | **Manuscript Section 4.2:** Pareto optimal configuration candidates and final $x^*$ selection. | Ranking, configuration strings, EIR%, SRR%, QRR%, `meets_srr`, `meets_qrr`, `selected`. |
| **`B10_pareto_front.csv`** | **Manuscript Section 4.2:** Complete non-dominated Pareto frontier points. | Energy, Speed, Quality coordinates of all Pareto-optimal configurations. |
| **`B11_surrogate_reliability.csv`** | **Manuscript Section 4.3:** Gaussian Process surrogate model fit & reliability diagnostics. | LOO $R^2$, NPE (Normalized Prediction Error %), MAE, RMSE, Uncertainty Calibration Ratio (UCR). |
| **`table_b9_gate_viability_confusion_matrix.csv`** | **UREC Form 10 & Thesis Appendix B.9:** Binary configuration viability classification confusion matrix. | True Positives (5), False Positives (1), False Negatives (1), True Negatives (23), Accuracy (93.33%), F1 (83.33%). |
| **`table_b10_task_category_confusion_matrix.csv`** | **UREC Form 10 & Thesis Appendix B.10:** Multiclass 8-way prompt task category confusion matrix on Dolly 15k test set ($N=2,253$). | $8 \times 8$ matrix of Actual vs Predicted task categories, Class Recalls, Class Precisions, Overall Accuracy (69.68%), Macro F1 (68.16%). |
| **`supplementary/indicators.csv`** | **Key Executive Ratios:** Ready-to-cite headline thesis indicators. | EIR (%), SRR (%), QRR (%), SER, Break-Even Prompts ($N^*$). |

---

## 3. Executive Cheatsheet for Writing Chapters 4 & 5

When drafting prose or filling manuscript tables, use these exact authoritative values:

### A. The Winning Configuration ($x^*$)
* **Model:** `qwen2.5-1.5b`
* **Precision / Quantization ($p$):** `Q4_K_M`
* **GPU Layers Offloaded ($g$):** `28` (Full GPU offload)
* **CPU Threads ($c$):** `4`
* **Max Generation Tokens ($t$):** `128`
* **Batch Size ($b$):** `1`

### B. Core Headline Findings (SOP 1 & SOP 2)
* **Energy Improvement Ratio (EIR):** **74.63% energy reduction** vs. unoptimized default (BOPIS used **78.95 J** vs. Unoptimized **311.15 J**, and Random Search **106.90 J**).
* **Speed Retention Ratio (SRR):** **265.79%** (BOPIS generated **93.70 tok/s** vs. Unoptimized **35.25 tok/s**).
* **Quality Retention Ratio (QRR):** **95.05%** (BOPIS achieved **0.1433 BERTScore F1** vs. Unoptimized **0.1508**).
* **Sample Efficiency Ratio (SER):** **1.23** (BOPIS found the optimal configuration in **13 iterations** vs Random Search taking **16 iterations**).
* **Break-Even Point ($N^*$):** **19 prompts** (The energy saved by BOPIS amortizes the entire 30-trial search cost after serving just 19 requests).

### C. Inferential Statistics (SOP 3 - Wilcoxon Signed-Rank Test, $\alpha = 0.05$)
* **Energy:** $p = 2.60 \times 10^{-10}$ ($p < 0.05$, statistically significant energy savings).
* **Inference Speed:** $p = 4.87 \times 10^{-48}$ ($p < 0.05$, statistically significant speedup).
* **Quality (BERTScore F1):** $p = 0.1256$ ($p > 0.05$, **no statistically significant quality loss**, confirming quality retention).

### D. Optimizer & Surrogate Diagnostics
* **Leave-One-Out Cross Validation $R^2$:** **0.9366** (indicates strong Gaussian Process fit).
* **Normalized Prediction Error (NPE):** **7.51%** (satisfies the $< 10\%$ target threshold).
* **Mean Absolute Error (MAE):** **11.02 J**.
* **Gate Viability Classification Accuracy:** **93.33%** ($\text{F}_1 = 83.33\%$).
* **Task Category Classification Accuracy:** **69.68%** ($\text{Macro } \text{F}_1 = 68.16\%$, beating rule baseline by $+21.4\%$).

---

## 4. Instructions for Formatting & Editing

1. **For Microsoft Word / Overleaf:** Import numbers directly from these CSVs into Chapter 4 text and Appendix B tables.
2. **Do Not Re-run Benchmark Scripts:** Any command such as `python -m bopis benchmark` or `python -m bopis run` will start an intensive hardware test that overwrites local databases. Read from `docs/chapter4_data/` instead.
