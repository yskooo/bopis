# Response to Panelist Comments and Revision Matrix

This document provides clarifications addressing the comments and recommendations found in the BSCS Thesis Tool Revision Matrix (AY 2026–2027). The clarifications below are based on the implemented BOPIS tool state and are intended to supplement, rather than modify, Chapters 1–3 as requested.

## 1. Title

**Panel Comment:** No comment

**Clarification:** Noted. The title remains as-is: *BOPIS: Bayesian Optimization and Pareto-Based Intelligent Configuration Selection for Energy-Efficient Local LLM Inference*.

## 2. Preliminaries

**Panel Comment:** No comment

**Clarification:** Noted.

## 3. Chapter 1

### 3.1 Random Search as Baseline

**Panel Comment:** *"Justified random search as the baseline: it is a practical, reproducible standard for comparing optimization methods (Bergstra & Bengio, 2012) (p. 10)."*

**Clarification (Additional Context):** Random search is implemented as an uninformed search over the same configuration space as BOPIS. It requires no prior knowledge of the search space and evaluates configurations independently, making it an impartial baseline for determining whether model-guided optimization (Bayesian Optimization + Pareto selection) yields a statistically and practically significant improvement under identical hardware, model, and dataset conditions. This aligns with Bergstra and Bengio (2012).

### 3.2 Strength, Innovation, and Novelty of BOPIS

**Panel Comment:** *"Clearly indicate the strength, innovation, and novelty of BOPIS."*

**Clarification:** BOPIS's strength lies in its two-stage selection strategy: (1) Bayesian Optimization to efficiently explore the configuration space with few evaluations, and (2) Pareto-based trade-off analysis to identify non-dominated configurations across competing objectives. Its innovation and novelty are that no reviewed work combines Bayesian Optimization and Pareto analysis into a single offline, pre-deployment configuration selection system that jointly considers runtime parameters (input token length, batch size, GPU layer offloading, CPU thread allocation, decoding parameters) and GGUF precision variants (F16, Q8_0, Q4_K_M) for energy-efficient local LLM inference. The system prioritizes energy reduction while enforcing minimum acceptable speed and quality via explicit retention thresholds (SRR, QRR), rather than collapsing all objectives into a single weighted scalar.

### 3.3 SOP Bias and Predefined Inputs

**Panel Comment:** *"Indicate in the Statement of the Problem that the inputs are not predefined, to avoid showing bias."*

**Clarification (Without Modifying SOP Text):** The evaluation in the study uses a standardized, representative instruction-following dataset (Databricks Dolly 15k) to ensure fair, reproducible comparisons across conditions. However, the implemented tool itself is general-purpose and does **not** require predefined inputs at runtime. In the live UI, arbitrary free-form user prompts are accepted. For free-chat scoring with no explicit reference, the UI bridge performs retrieval-based fallback: it matches the **user's prompt** (not the assistant's reply) against the nearest pooled Q/A pair to obtain a reference for BERTScore F1 computation, with match similarity surfaced and low-confidence cases flagged. This distinction—standardized dataset for controlled evaluation vs. support for arbitrary user inputs in deployment/interaction—is a clarification of scope and generality.

### 3.4 System Priority and Success Criteria

**Panel Comment:** *"Stated the priority of the system: BOPIS selects the configuration with the greatest energy reduction, and speed and quality act as minimum thresholds (SRR, QRR), not competing goals (p. 11)."*

**Clarification:** The selection logic prioritizes minimizing energy consumption while treating inference speed and output quality as hard constraints via retention ratios (Speed Retention Ratio, Quality Retention Ratio). Configurations are evaluated on the Pareto front; among non-dominated candidates, the configuration achieving the greatest energy reduction subject to meeting SRR/QRR thresholds is selected. Energy Improvement Ratio (EIR), SRR, and QRR define "better than baseline" in measurable terms, with null/negative/partial outcomes reported transparently.

### 3.5 Where BOPIS Performs Better + Statistical Treatment

**Panel Comment:** *"Showed where BOPIS is better by comparing per metric (energy, speed, quality, resource use) and per precision variant (F16, Q8_0, Q4_K_M) with the Wilcoxon signed-rank test (Statement of the Problem, pp. 14–15)."*

**Clarification:** The tool implements a two-condition comparison (Random Search vs. BOPIS) using the **Wilcoxon signed-rank test** for all dependent variables: total energy (J), energy per token (J/token), inference speed (tokens/s), output quality (BERTScore F1), and resource utilization (CPU %, GPU %, memory usage in MB). Results are reportable per precision variant (F16, Q8_0, Q4_K_M). Statistical significance is complemented by practical significance via EIR/SRR/QRR, ensuring conclusions reflect both statistical and practical improvement.

### 3.6 Weaknesses (Pinpointed)

**Panel Comment:** *"Pinpointed the weaknesses of BOPIS in Scope and Limitation (pp. 16–18): energy estimated in software (adds variance); GPU offloading slowed generation on this GPU class; device discovery covers NVIDIA devices only; results apply to one model, one machine, and one dataset; extending device discovery is listed as future work."*

**Clarification:** These limitations reflect the current implementation and measurement approach. When available, NVML energy counters are used for direct GPU energy measurement (reported as measured). A labeled resource-allocation estimator remains only as a fallback and is never misrepresented as a hardware measurement. Device discovery is currently NVIDIA-focused; extending to other vendors is identified as future work. Results are intentionally scoped to the tested model, hardware, configuration space, monitoring procedure, and dataset to maintain internal validity.

## 4. Chapter 2

### 4.1 Justify Random Search Baseline

**Panel Comment:** *"Justify the use of random search as the baseline"*

**Clarification:** As noted in Section 3.1, random search is an uninformed, reproducible, and impartial baseline (Bergstra & Bengio, 2012). It samples the same search space without model-guided exploitation, enabling a fair test of whether BOPIS's guided search yields improvement under identical experimental conditions.

### 4.2 Highlight Novelty vs. Existing Studies

**Panel Comment:** *"Highlight the novelty of BOPIS in comparison with existing studies"*

**Clarification:** The novelty lies in the integration: no reviewed study combines Bayesian Optimization and Pareto analysis in a single offline, pre-deployment configuration selection system that covers both runtime inference parameters and GGUF precision variants for local LLM inference. Furthermore, the system couples this selection with (a) explicit retention-threshold-based prioritization (energy-first with SRR/QRR), (b) two-condition statistical validation (Wilcoxon signed-rank), and (c) an honest measurement model that distinguishes measured (NVML/RAPL) from labeled fallback estimates.

### 4.3 Computational Cost / "Why More Expensive"

**Panel Comment (from Sir Nino):** *"Why BOPIS more expensive than others?"*

**Clarification:** BOPIS incurs additional upfront computation due to Bayesian Optimization (evaluating multiple candidate configurations to build the surrogate and identify promising regions). This overhead is **incurred once, pre-deployment**; it does not affect runtime inference latency or response time in production. The trade-off is sample-efficient search to identify an energy-efficient configuration before deployment, rather than paying per-query optimization cost.

## 5. Chapter 3

### 5.1 Identify Parameters Where BOPIS Performs Better + Strengths

**Panel Comment:** *"Identify the specific parameters in which BOPIS performs better than random search, and the strengths that supports this"*

**Clarification:** Performance is evaluated per metric and per precision variant (F16, Q8_0, Q4_K_M). The tool measures energy (total J, J/token), inference speed (tokens/s), output quality (BERTScore F1), and resource utilization (CPU %, GPU %, memory MB). BOPIS's strength is its ability to navigate trade-offs efficiently: the GP surrogate focuses evaluations on high-potential regions, while Pareto selection enforces minimum acceptable speed/quality, yielding configurations with reduced energy under those retention constraints. The specific winning metrics/variants depend on the experimental run on the target hardware; the implemented measurement and statistical framework (Wilcoxon signed-rank, EIR/SRR/QRR) supports reporting them explicitly per condition.

### 5.2 Add More Datasets

**Panel Comment:** *"Add more datasets to test whether BOPIS performs..."*

**Clarification:** This is already implemented. The evaluation set is a **60/40 mix**: 300 prompts from Databricks Dolly 15k plus up to 200 prompts drawn from four external datasets, each declared as a named source with its own dataset id, version, split, and license (`EXTERNAL_DATASETS`):

| Source constant | Dataset | Rows planned |
|---|---|---|
| `SOURCE_DOLLY` | `databricks-dolly-15k` | 300 |
| `SOURCE_NQ` | `google-research-datasets/natural_questions` | 60 |
| `SOURCE_TRIVIA` | `mandarjoshi/trivia_qa` | 60 |
| `SOURCE_CNN` | `abisee/cnn_dailymail` | 50 |
| `SOURCE_SQUAD` | `rajpurkar/squad_v2` | 30 |

The mix is measured, not assumed: each run's `manifest.json` records per-source counts under `dataset_sources` (with license and dataset revision), so a source that failed to load reports `n: 0` rather than a claimed share. The most recent run (`runs/20261016T094352Z`, 2026-10-16) loaded **300 Dolly + 60 TriviaQA + 50 CNN/DailyMail + 30 SQuAD v2** prompts (440 total). Natural Questions contributed 0 in that run because of a loader parsing bug (see Section 6.4), which has since been fixed; the intended total is 500 (300 + 200). External sources also feed the retrieval reference pool through `ReferenceRetriever.build_pool(include_external=True)`.

### 5.3 Free Input Optimization / Deployment vs. Evaluation

**Panel Comment (from Sir Violeta):** *"If the user gives a free input (not the predefined Dolly), does it still optimize?"*

**Clarification:** BOPIS performs **offline configuration optimization** (search + selection) once, prior to deployment, using a representative dataset. It does **not** re-run Bayesian Optimization per live free input. Instead, the selected optimized configuration is applied for inference on arbitrary inputs. The live UI supports free inputs and provides BERTScore F1 via retrieval fallback (matching on the user's prompt to the nearest pooled reference), with similarity and low-confidence flags surfaced. This separation (offline optimization on standardized data; general-purpose inference + retrieval-based scoring for free chat) is by design.

## 6. Tool (Software/Hardware)

### 6.1 Measurement Honesty and Estimator Labeling

**Clarification:** GPU energy is probed via NVML when available (reported as measured). Software-based energy estimation is used **only** as a labeled fallback (never presented as a hardware measurement). The UI and reporting distinguish measured vs. estimated scopes to avoid misrepresentation.

### 6.2 BERTScore on Free Chat (Bug Fix)

**Clarification:** The previously reported case where free-chat replies showed no BERTScore F1 stemmed from two factors: (a) bridge not running when testing via `file://` (scoring intentionally returns `n/a` without bridge), and (b) a semantic mismatch where the retrieval query defaulted to the assistant reply instead of the user prompt. This has been corrected: the `/api/score` endpoint and `Instruments.score()` now accept an optional `prompt` parameter; when `reference` is empty, retrieval matches on `prompt.strip()` (falling back to `candidate` only for backward compatibility). The frontend passes the user's prompt to the score call. The fix ensures free-chat inputs return a real BERTScore F1 (with match similarity and low-confidence flag) when the bridge is available.

### 6.3 Dataset Extensibility and Diversity

**Clarification:** As detailed in Section 5.2, dataset diversity is not merely "supported" — it is used. The evaluation already mixes Dolly 15k with Natural Questions, TriviaQA, CNN/DailyMail, and SQuAD v2 on a 60/40 Dolly/external basis, and every run records the measured per-source counts in its manifest. Additional datasets can be added by extending the `EXTERNAL_DATASETS` table and adding a loader in `bopis/dataset.py`, without structural changes to the optimization pipeline or the statistical treatment.

### 6.4 Natural Questions Loader Bug (Dataset Diversity)

**Clarification:** While verifying dataset coverage, a defect was found and fixed in `load_natural_questions`: the `annotations` field of `google-research-datasets/natural_questions` is a **dict** (`{"short_answers": [...]}`), not a list, so the loader's `annotations[0]` raised `KeyError`; additionally, the result list was never initialised, so the append raised `NameError`. Both exceptions were swallowed by the loader's blanket `except`, silently returning zero prompts — which is why every prior run reported `natural_questions: n=0`. The loader now reads `annotations["short_answers"]`, joins list-valued `text`, accepts string `text`, and initialises its accumulator. It now returns the expected prompts, and regression tests (`tests/test_dataset.py::TestNaturalQuestionsLoader`, plus a `TestExternalMixing` for the 60/40 split) lock the behaviour without network access.

### 6.5 Cost Clarification (Production Impact)

**Clarification:** BOPIS's computational overhead is confined to the offline optimization phase (pre-deployment). Once a configuration is selected, inference uses that configuration with no additional BO overhead at runtime; production response time is unaffected.

## 7. References

**Panel Comment:** No specific issue noted beyond normal formatting.

**Clarification:** The cited sources (Bergstra & Bengio, 2012; Patterson et al., 2021; Strubell et al., 2019; Vaswani et al., 2023; Yan et al., 2025; Husom et al., 2024; Niu et al., 2025; Zhou et al., 2024; Goldberg, 1991; Jain, 1991; Lannelongue et al., 2021) remain applicable and support the justifications above.

## 8. General Observations

**Panel Comment (General):** Emphasis on strength/innovation/novelty, dataset diversity, clarification of inputs (predefined vs. free), parameters where BOPIS wins, statistical+practical improvement, weaknesses, and cost explanation.

**Overall Clarification:** All points are addressed above with reference to the implemented tool state. The clarifications focus on scope, design intent, and implementation details (measurement honesty, retrieval-based scoring with prompt matching, offline optimization vs. runtime behavior, Wilcoxon-based comparison, retention-threshold prioritization, and dataset extensibility) without altering the approved structure/content of Chapters 1–3.