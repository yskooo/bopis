# Review of Related Literature (RRL) Citations & Justifications
**Project:** BOPIS (Energy-Aware LLM Optimization Framework)  
**Document:** Manuscript Amendments & Theoretical Grounding Reference

---

## 1. Priority 1: Critical Core Citations (Panel Defense Focus)

### 1.1 Expected Improvement (EI) Acquisition Function
* **Reference:** Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. *Journal of Global Optimization*, 13(4), 455–492. https://www.researchgate.net/publication/235709802_Efficient_Global_Optimization_of_Expensive_Black-Box_Functions
* **Amendment Mapping:** A-1, A-2, A-3, A-7, A-40
* **Justification & Application:** Serves as the foundational theoretical origin for the Expected Improvement (EI) acquisition function. It directly supports the minimization form of EI formulation ($f(x^+) - \mu(x)$) utilized in the Bayesian Optimization loop.

### 1.2 Inference Engine & Quantization Ecosystem
* **Reference:** Gerganov, G. (2023). *llama.cpp: Inference of LLaMA model in pure C/C++* [Computer software]. GitHub. https://github.com/ggerganov/llama.cpp
* **Amendment Mapping:** A-1, A-4, A-23, A-29
* **Justification & Application:** Validates the underlying execution runtime, GGUF binary formats, and memory-quantization schemes (Q8_0, Q4_K_M). Grounding parameter redefinitions such as evaluating $t$ as `n_predict` directly traces to the llama.cpp engine architecture.

### 1.3 Non-Parametric Statistical Inference
* **Reference:** Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman and Hall/CRC. ISBN 978-0-412-04231-7.
  <!-- VERIFIED 2026-09-18: the DOI 10.1017/CBO9780511802843 previously listed here
       belongs to Davison & Hinkley (1997) (see §3.1), not to this book. Efron &
       Tibshirani has no Cambridge DOI; cite by ISBN. -->

* **Amendment Mapping:** A-16, A-30
* **Justification & Application:** Establishes the statistical principle for non-parametric 95% bootstrap confidence intervals used when reporting evaluation metrics (EIR, SRR, QRR).

### 1.4 Multi-Objective Pareto Hypervolume Evaluation
* **Reference:** Zitzler, E., & Thiele, L. (1998). Multiobjective optimization using evolutionary algorithms — a comparative case study. In *Parallel Problem Solving from Nature — PPSN V* (pp. 292–301). Springer, Berlin, Heidelberg. https://doi.org/10.1007/BFb0056872
* **Amendment Mapping:** A-15
* **Justification & Application:** Provides the original mathematical derivation for hypervolume (HV) as a quality metric in multi-objective optimization, justifying the scaling and normalization of HV space to $[0, 1]$ anchored at the nadir point.

### 1.5 Model Validation & Cross-Validation Strategy
* **Reference:** Arlot, S., & Celisse, A. (2010). A survey of cross-validation procedures for model selection. *Statistics Surveys*, 4, 40–79. https://doi.org/10.1214/09-SS054
* **Amendment Mapping:** A-10, A-40
* **Justification & Application:** Validates Leave-One-Out Cross-Validation (LOO-CV) over simple train-test splitting for small-sample surrogate model evaluations, justifying the surrogate reliability threshold of LOO $R^2 \ge 0.85$.

---

## 2. Priority 2: Methodology & Architectural Grounding

### 2.1 Prefill vs. Decode Phase Energy Attribution
* **Reference:** Agrawal, A., Kedia, N., Panwar, A., Mohan, J., Kwatra, A., Gulavani, B. S., Tumanov, A., & Ramjee, R. (2024). Taming throughput-latency tradeoff in LLM inference with Sarathi-Serve. In *20th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)* (pp. 112–126). arXiv. https://arxiv.org/abs/2403.02310
* **Amendment Mapping:** A-35
* **Justification & Application:** Provides theoretical and empirical support for splitting energy measurements across prefill (`prompt_ms`) and decode (`predicted_ms`) phases due to distinct computational characteristics (compute-bound vs. memory-bound).

* **Reference (Secondary):** Zhong, Y., Liu, S., Chen, J., Hu, B., Wang, Z., Liu, X., Lin, X., & Zhang, H. (2024). DistServe: Disaggregating prefill and decoding for goodput-optimized LLM serving. In *20th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)*. arXiv. https://arxiv.org/abs/2401.09670
* **Amendment Mapping:** A-35
* **Justification & Application:** Reinforces phase-level separation of energy profiling by demonstrating how performance and energy profiles decouple during context processing versus token generation.

### 2.2 SLO Compliance & Quality Threshold Definitions
* **Reference:** Stojkovic, J., Zhang, E., Kim, D., Ryoo, J. H., & Torrellas, J. (2024). DynamoLLM: Designing LLM inference clusters for performance and energy. *arXiv preprint arXiv:2408.00741*. https://arxiv.org/abs/2408.00741
* **Amendment Mapping:** A-25, A-26, A-30
* **Justification & Application:** Supports setting Service Level Objective (SLO) thresholds based on relative baseline performance, directly backing the operational thresholds for Speedup Retention Ratio ($S_{\text{min}} = 0.95$) and Quality Retention Ratio ($Q_{\text{min}} = 0.98$).

### 2.3 Hardware Power & Load-Line Modeling
* **Reference:** Rotem, E., Naveh, A., Ananthakrishnan, A., Rajwan, D., & Weissmann, E. (2012). Power management architecture of the Intel microarchitecture code-named Sandy Bridge. *IEEE Micro*, 32(2), 20–27. https://doi.org/10.1109/MM.2012.6
* **Amendment Mapping:** A-8, A-19
* **Justification & Application:** Groundwork for dual-component dynamic power modeling (CPU + GPU). Supports the differential power allocation equation:
  $P_{\text{active}} = (P_t - P_{\text{idle}}) \times \Delta t$

---

## 3. Priority 3: Supporting & Statistical Foundations

### 3.1 Secondary Resampling Validation
* **Reference:** Davison, A. C., & Hinkley, D. V. (1997). *Bootstrap methods and their application*. Cambridge University Press. https://doi.org/10.1017/CBO9780511802843
* **Amendment Mapping:** A-16, A-30
* **Justification & Application:** Supports algorithmic choices in Monte Carlo bootstrap iterations (`n_boot` in execution scripts) for stable confidence interval estimation.

### 3.2 Hypervolume Metric Selection
* **Reference:** Emmerich, M., Beume, N., & Naujoks, B. (2005). An EMO algorithm using the hypervolume measure as selection criterion. In *International Conference on Evolutionary Multi-Criterion Optimization* (pp. 62–76). Springer, Berlin, Heidelberg. https://doi.org/10.1007/11405963_5
* **Amendment Mapping:** A-15
* **Justification & Application:** Validates hypervolume computation behavior and boundary enforcement at the nadir reference point under constrained multi-objective spaces.

### 3.3 Low-Bit Quantization Validity
* **Reference:** Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2023). GPTQ: Accurate post-training quantization for generative pre-trained transformers. In *International Conference on Learning Representations (ICLR 2023)*. https://arxiv.org/abs/2210.17323
* **Amendment Mapping:** A-4, A-23, A-29
* **Justification & Application:** Establishes that 3–4-bit post-training weight quantization retains accuracy with "negligible degradation relative to the uncompressed baseline," justifying Q4_K_M inclusion in the variant set.
  <!-- VERIFIED 2026-09-18: replaces the previous SparseGPT citation (Frantar &
       Alistarh, 2023, arXiv:2301.00774). SparseGPT is a *pruning* paper and does
       not support a claim about low-bit quantization; GPTQ is the same group's
       quantization result and is the correct source for this claim. -->

### 3.4 Process-Level Power Accounting
* **Reference:** Colmant, M., Kurpicz, M., Felber, P., Huertas, L., Rouvoy, R., & Sobe, A. (2015). Process-level power estimation in VM-based systems. In *Proceedings of the Tenth European Conference on Computer Systems (EuroSys '15)*. ACM. https://doi.org/10.1145/2741948.2741971
* **Amendment Mapping:** A-8
* **Justification & Application:** Grounding for non-NVML / CPU fallback mode (Mode C power accounting), enabling per-process resource attribution. The BitWatts middleware described here infers fine-grained per-process power from resource usage without a hardware power meter — the same inference our Mode C estimator performs.
  <!-- VERIFIED 2026-09-18: this paper was previously attributed to "Lim, M. Y.,
       Rawson, F., & Ballew, W. (2014), EuroSys '14". Wrong authors, year and
       proceedings edition. Correct record confirmed via ACM DL. -->

### 3.5 Floating-Point Reproducibility Limitations
* **Reference:** Goldberg, D. (1991). What every computer scientist should know about floating-point arithmetic. *ACM Computing Surveys*, 23(1), 5–48. https://doi.org/10.1145/103162.103163
* **Amendment Mapping:** A-31
* **Justification & Application:** Canonical source for the non-associativity of floating-point addition — the property that makes summation results depend on reduction order. This is the mechanism behind the Amendment A-31 caveat: changing the quantization scheme or the CPU/GPU offload split changes the order in which partial sums are accumulated, so bitwise-identical outputs cannot be guaranteed *across* configurations even under a fixed seed and greedy decoding.
  <!-- VERIFIED 2026-09-18: replaces "Pham, H., Qian, C., Wang, T., & Yu, Y.
       (2020). Problems and opportunities in neural network robustness and
       reproducibility. arXiv:2206.04236" — NO SUCH PAPER EXISTS. The title
       returns no match on arXiv or Google Scholar, and a 2020 paper cannot
       carry a 2022 arXiv identifier. Do not cite it. -->
* **Reference (Secondary, verified real):** Gundersen, O. E., Coakley, K., Kirkpatrick, C., & Gil, Y. (2022). Sources of irreproducibility in machine learning: A review. *arXiv preprint arXiv:2204.07610*. https://arxiv.org/abs/2204.07610
* **Amendment Mapping:** A-31
* **Justification & Application:** Taxonomy of irreproducibility sources in ML, used for the general framing that implementation- and hardware-level variation is a recognized reproducibility factor. ⚠️ Confirm in the full text that it treats hardware/floating-point nondeterminism specifically before leaning on it for that narrower claim; the abstract does not say so.

* **Reference (Secondary):** Narayanan, D., Shoeybi, M., Casper, J., LeGresley, P., Patwary, M., Kulkarni, A., ... & Catanzaro, B. (2021). Efficient large-scale language model training on GPU clusters using Megatron-LM. In *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC21)*. https://arxiv.org/abs/2104.04473
* **Amendment Mapping:** A-31
* **Justification & Application:** Demonstrates how tensor parallelism, quantization layers, and hardware execution pipelines alter floating-point reduction order during matrix operations.

### 3.6 Quantization's Energy/Quality Trade-off
* **Reference:** Shi, T., & Ding, Y. (2025). Systematic characterization of LLM quantization: A performance, energy, and quality perspective. *arXiv preprint arXiv:2508.16712*. https://arxiv.org/abs/2508.16712
* **Amendment Mapping:** A-23, A-29
* **Justification & Application:** Empirical characterization of 11 post-training quantization methods across 4 model sizes (7B–70B) on a joint performance/energy/quality basis — the same three-way trade-off BOPIS optimizes. Supports treating precision as an energy-relevant configuration parameter rather than a quality-only one.
  <!-- VERIFIED 2026-09-18: replaces "Xu, Z., et al. (2023). Evaluating
       quantization-induced energy reduction in local LLM deployment. arXiv
       preprint." — NO SUCH PAPER FOUND, and the entry carried no arXiv ID.
       Do not cite it.
       SCOPE CAVEAT on the replacement: Shi & Ding evaluate A100/H100
       datacenter GPUs at 7B-70B. It supports the energy/quality trade-off
       claim but NOT a claim about consumer-hardware VRAM limits. -->

### 3.6a Hardware Feasibility Guards (HW-P0 / HW-B0)
* **Basis:** No external citation. The HW-P0 / HW-B0 guards are *derived*, not borrowed: weight footprint and KV-cache size are computed from the GGUF model shape in `bopis/hardware.py::ModelSpec`, and the VRAM/RAM budgets come from run-time host profiling (`bopis/profile.py`). The F32 Mistral 7B ≈ 27 GiB figure is an arithmetic consequence of the model's parameter count at 4 bytes/parameter, not an empirical finding from the literature.
* **Amendment Mapping:** A-29, A-30, A-39
* **Justification & Application:** Cite Gerganov (2023) (§1.2) for the GGUF footprint semantics and present the guards as the study's own instrument design, verified by `tests/test_hardware.py::TestModelSizeGuards`. Presenting a derived arithmetic bound as a literature finding is a weaker defense than owning it, and invites a request for the source.

---

## 4. Citation Summary Table

| Citation Key | Primary Amendment | Category | Role in Defense |
| :--- | :--- | :--- | :--- |
| **Jones et al. (1998)** | A-1, A-2, A-7 | Optimization | EI Acquisition Origin |
| **Gerganov (2023)** | A-1, A-23, A-29 | Runtime | llama.cpp / GGUF Foundation |
| **Efron & Tibshirani (1993)** | A-16, A-30 | Statistics | Bootstrap CI Grounding |
| **Zitzler & Thiele (1998)** | A-15 | Metrics | Hypervolume Normalization |
| **Arlot & Celisse (2010)** | A-10, A-40 | Validation | LOO Cross-Validation |
| **Agrawal et al. (2024)** | A-35 | Profiling | Prefill / Decode Separation |
| **Stojkovic et al. (2024)** | A-25, A-26 | Thresholds | SLO Retention Targets |
| **Rotem et al. (2012)** | A-8, A-19 | Power | Load-Line Energy Model |
| **Goldberg (1991)** | A-31 | Determinism | FP Non-Associativity Caveat |
| **Colmant et al. (2015)** | A-8 | Power | Per-Process Power Attribution |
| **Frantar et al. (2023), GPTQ** | A-4, A-23, A-29 | Quantization | 4-bit Accuracy Retention |
| **Shi & Ding (2025)** | A-23, A-29 | Quantization | Energy/Quality Trade-off |
| *(no citation — derived)* | A-29, A-30, A-39 | Hardware Guard | HW-P0/HW-B0 are own instrument design |

<!-- VERIFIED 2026-09-18. Four rows of the previous table pointed at records that
     were wrong or non-existent:
       - "Pham et al. (2020)"  -> paper does not exist; replaced by Goldberg (1991).
       - "Xu et al. (2023)"    -> paper not found; replaced by Shi & Ding (2025),
                                  and the VRAM-bound claim de-cited (it is derived).
       - "INT8 Benefits"       -> INT8 is not in this study at all. Amendment A-23
                                  replaced FP16/INT8 with F32/F16/Q8_0/Q4_K_M.
       - SparseGPT             -> pruning paper cited for a quantization claim;
                                  replaced by GPTQ.
     Also fixed: Efron & Tibshirani carried Davison & Hinkley's DOI, and the
     EuroSys per-process power paper was attributed to the wrong authors/year. -->
