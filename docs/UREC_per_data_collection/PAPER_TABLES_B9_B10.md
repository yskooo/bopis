# Paper-Ready Tables: B.9, B.10, B.10A, B.11

These are the UREC appendix tables for the experimental paper. They contain no
row-level data; the CSV files in this folder are only the backing evidence.

Final appendix naming:

```text
Table B.9.   Pre-Experiment Instrument Template for Configuration Viability Classification (Confusion Matrix)
Table B.10.  Pre-Experiment Instrument Template for Multiclass Task Category Classification Matrix
Table B.10A. Pre-Experiment Instrument Template for Per-Class Classification and Confusion Matrix Metrics
Table B.11.  Pre-Experiment Instrument Template for Per-Prompt Task Category Classification Record
```

## Table B.9. Pre-Experiment Instrument Template for Configuration Viability Classification (Confusion Matrix)

| Predicted status | Empirical pass | Empirical fail | Total |
| --- | ---: | ---: | ---: |
| Predicted viable | 5 (TP) | 1 (FP) | 6 |
| Predicted infeasible | 1 (FN) | 23 (TN) | 24 |
| Total actual | 6 | 24 | 30 |

| Metric | Value |
| --- | ---: |
| Accuracy | 93.33% |
| Precision | 83.33% |
| Recall | 83.33% |
| F1 score | 83.33% |

## Table B.10. Pre-Experiment Instrument Template for Multiclass Task Category Classification Matrix

Learned classifier, held-out Dolly test set, N = 2,253.

| Actual category | Open QA | Closed QA | Summarization | Info. extraction | General QA | Classification | Brainstorming | Creative writing | Actual N | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Open QA | **433** | 0 | 2 | 1 | 68 | 7 | 47 | 4 | 562 | 77.05% |
| Closed QA | 4 | **216** | 34 | 10 | 0 | 1 | 1 | 0 | 266 | 81.20% |
| Summarization | 3 | 50 | **104** | 16 | 2 | 0 | 3 | 0 | 178 | 58.43% |
| Information extraction | 2 | 53 | 34 | **132** | 3 | 0 | 2 | 0 | 226 | 58.41% |
| General QA | 132 | 0 | 1 | 0 | **161** | 3 | 22 | 9 | 328 | 49.09% |
| Classification | 16 | 1 | 0 | 1 | 4 | **295** | 3 | 1 | 321 | 91.90% |
| Brainstorming | 32 | 0 | 0 | 0 | 54 | 4 | **171** | 4 | 265 | 64.53% |
| Creative writing | 14 | 0 | 0 | 0 | 24 | 2 | 9 | **58** | 107 | 54.21% |
| **Total predicted** | 636 | 320 | 175 | 160 | 316 | 312 | 258 | 76 | **2,253** | |

## Table B.10A. Pre-Experiment Instrument Template for Per-Class Classification and Confusion Matrix Metrics

| Category | TP | FP | FN | TN | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Open QA | 433 | 203 | 129 | 1,488 | 68.08% | 77.05% | 72.29% | 562 |
| Closed QA | 216 | 104 | 50 | 1,883 | 67.50% | 81.20% | 73.72% | 266 |
| Summarization | 104 | 71 | 74 | 2,004 | 59.43% | 58.43% | 58.92% | 178 |
| Information extraction | 132 | 28 | 94 | 1,999 | 82.50% | 58.41% | 68.39% | 226 |
| General QA | 161 | 155 | 167 | 1,770 | 50.95% | 49.09% | 50.00% | 328 |
| Classification | 295 | 17 | 26 | 1,915 | 94.55% | 91.90% | 93.21% | 321 |
| Brainstorming | 171 | 87 | 94 | 1,901 | 66.28% | 64.53% | 65.39% | 265 |
| Creative writing | 58 | 18 | 49 | 2,128 | 76.32% | 54.21% | 63.39% | 107 |

## Table B.10B. Model-Level Summary (Optional, not part of the agreed B.10A/B.11 pair)

| Model | Macro precision | Macro recall | Macro F1 | Exact accuracy | QA-merged accuracy | Sensitivity-tier accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Learned Naive Bayes | 70.70% | 66.85% | 68.16% | 69.68% | 78.56% | 79.18% |
| Rule baseline | 65.28% | 43.10% | 45.59% | 48.25% | 63.56% | 67.07% |

Overall exact accuracy: **1,570 / 2,253 = 69.68%**. Macro F1: **68.16%**.

## Table B.11. Pre-Experiment Instrument Template for Per-Prompt Task Category Classification Record

This is the unsummarized per-item record for the learned classifier. For Table
B.10 and B.10A, per-item means per prompt. The `Model` column has been removed
and must not be added back.

| Prompt No. | Prompt ID | Actual Category | Predicted Category | Correct? |
| ---: | --- | --- | --- | --- |
| 1 | `<prompt-id>` | `<category>` | `<category>` | Yes/No |
| 2 | `<prompt-id>` | `<category>` | `<category>` | Yes/No |
| 3 | `<prompt-id>` | `<category>` | `<category>` | Yes/No |
| … | … | … | … | … |
| 2,253 | `<prompt-id>` | `<category>` | `<category>` | Yes/No |

**Table note:** Prompt No. runs from 1 to 2,253. Table B.11 is learned-classifier
only and has 2,253 rows. Do not use the 500-prompt BOPIS validation subset for
this table unless it is explicitly labelled as a subset and its summary metrics
are recomputed.
