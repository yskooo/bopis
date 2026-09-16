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
* **Reference:** Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman and Hall/CRC. https://doi.org/10.1017/CBO9780511802843
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
* **Reference:** Frantar, E., & Alistarh, D. (2023). SparseGPT: Massive language models can be accurately pruned in one shot. In *International Conference on Machine Learning (ICML)*. PMLR. https://arxiv.org/abs/2301.00774
* **Amendment Mapping:** A-4, A-23, A-29
* **Justification & Application:** Establishes that high-compression 4-bit and mixed-precision schemes retain sufficient model capacity for task execution, justifying Q4_K_M inclusion.

### 3.4 Process-Level Power Accounting
* **Reference:** Lim, M. Y., Rawson, F., & Ballew, W. (2014). Process-level power estimation in VM-based systems. In *Proceedings of the 9th European Conference on Computer Systems (EuroSys '14)*. ACM.
* **Amendment Mapping:** A-8
* **Justification & Application:** Grounding for non-NVML / CPU fallback mode (Mode C power accounting), enabling per-process resource attribution.

### 3.5 Floating-Point Reproducibility Limitations
* **Reference:** Pham, H., Qian, C., Wang, T., & Yu, Y. (2020). Problems and opportunities in neural network robustness and reproducibility. *arXiv preprint arXiv:2206.04236*.
* **Amendment Mapping:** A-31
* **Justification & Application:** Justifies the theoretical caveat in Amendment A-31 regarding non-deterministic variance in LLM outputs due to non-associative floating-point reduction order across CPU/GPU offloads.

* **Reference (Secondary):** Narayanan, D., Shoeybi, M., Casper, J., LeGresley, P., Patwary, M., Kulkarni, A., ... & Catanzaro, B. (2021). Efficient large-scale language model training on GPU clusters using Megatron-LM. In *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC21)*. https://arxiv.org/abs/2104.04473
* **Amendment Mapping:** A-31
* **Justification & Application:** Demonstrates how tensor parallelism, quantization layers, and hardware execution pipelines alter floating-point reduction order during matrix operations.

### 3.6 Hardware Feasibility Guards
* **Reference:** Xu, Z., et al. (2023). Evaluating quantization-induced energy reduction in local LLM deployment. *arXiv preprint*.
* **Amendment Mapping:** A-29, A-30, A-39
* **Justification & Application:** Provides empirical backing for VRAM sizing rules (HW-P0 guard) and establishes why unquantized baseline deployments (e.g., F32 Mistral 7B requiring ~27 GiB VRAM) are infeasible on consumer-grade hardware.

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
| **Pham et al. (2020)** | A-31 | Determinism | FP Variance Caveats |
| **Xu et al. (2023)** | A-29, A-39 | Hardware Guard | VRAM Bounds & INT8 Benefits |