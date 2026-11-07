# Chapters 1–3 Revision Plan — Dataset Mixing (60/40) and the SOP "Inputs Are Not Predefined"

**Purpose:** map, section by section, the **paraphrase-only** edits to Chapters 1,
2, and 3 that follow from two panel-driven changes, without altering the design,
method, or claims of the study.

**Two changes driving every edit below**

- **(A) Dataset mixing.** The 500-prompt evaluation set is now a 60/40 mixture:
  300 prompts from Databricks Dolly 15k plus 200 prompts from four independent
  NLP benchmarks (Natural Questions, TriviaQA, CNN/DailyMail, SQuAD 2.0).
- **(B) SOP bias framing.** The panel asked that the Statement of the Problem
  state that **inputs are not predefined**, so the evaluation does not appear
  biased toward the system.

**Source of truth for the change itself:** `docs/DATASET_MIXING_REVISIONS.md`
and `bopis/dataset.py` (`EXTERNAL_DATASETS`, `DEFAULT_DOLLY_SHARE`,
`DEFAULT_EXTERNAL_SHARE`). Line anchors below are from
`docs/LATEST_MANUSCRIPT_THESIS_WRITING2_G2.md` (line numbers drift; search by
the quoted text).

**Status:** `plan` = proposed, not applied. This file is non-destructive; the
manuscript is untouched until the team approves applying the edits.

---

## 0. The one distinction that drives every edit

- **The bias concern belongs to quality evaluation only.** The 500-prompt
  *evaluation set* is mixed; the **50-prompt proxy used inside the optimizer
  stays Dolly-only**. Keep these two separate in every sentence.
- **The optimizer is not prompt-biased.** The Gaussian Process learns the
  configuration-to-energy relationship; the prompt distribution does not bias it.
  So (B) is answered in the SOP and in the evaluation design, not in the search
  design.

---

## 1. Chapter 1

| Line | Section | Current wording | Paraphrase (essence preserved) |
| :--- | :--- | :--- | :--- |
| 193 | Statement of the Problem | ¶1 ends "...evaluated through a two-way comparison." | Append one sentence: *"The evaluation inputs are not predefined or hand-picked by the researchers; they are drawn from standardized, publicly available prompt sources, so that the measured outcomes are not biased toward any configuration."* |
| 217 | Scope and Limitation | "the same standardized prompt dataset from Databricks Dolly 15k" | "the same standardized prompt mixture drawn from Databricks Dolly 15k and four independent NLP benchmarks" |
| 219, 227 | Scope and Limitation | "prompt dataset used in the study" | "prompt mixture used in the study" |
| 235 | Significance → Students and Researchers | "By using a fixed dataset, consistent metrics…" | "By using a standardized, multi-source prompt set, consistent metrics…" |
| 265 | Definition of Terms — *Prompt* | "prompts are sampled from the Databricks Dolly 15k dataset" | "prompts are sampled from a standardized mixture of Databricks Dolly 15k and four independent NLP benchmarks" |
| 267 | Definition of Terms — *500-Prompt Evaluation Dataset* | "selected from Databricks Dolly 15k" | "drawn from five public sources — Databricks Dolly 15k (300 prompts) plus Natural Questions, TriviaQA, CNN/DailyMail, and SQuAD 2.0 (200 prompts)" |
| 269 | Definition of Terms — *50-Prompt Proxy Subset* | "selected from the 500-prompt evaluation dataset" | "drawn from the Databricks Dolly 15k portion of the evaluation pool" (the proxy stays Dolly-only) |
| 353 | Definition of Terms — *Databricks Dolly 15k* | "used as the source of prompts and reference responses for evaluation" | "used as **one of the sources** of prompts and reference responses for evaluation" |
| 249+ | Definition of Terms | — | Add term **"Multi-Source Evaluation Set"** = the 500-prompt mixture of five sources used for final evaluation. |

The conceptual-framework mention of Dolly (~181/187) only needs "primary source"
→ "one of five sources".

---

## 2. Chapter 2

| Line | Current wording | Paraphrase |
| :--- | :--- | :--- |
| 496 | "reference outputs provided by Databricks Dolly 15k … across all 500 sampled prompts" | "the reference outputs accompanying each source in the evaluation mixture … across all 500 sampled prompts" (every source carries references, so BERTScore still applies uniformly) |
| 496/498 | — | Add one sentence: *"Because the external benchmarks were not used to design the task classifier or prior probability table, quality retention is evaluated on prompts independent of the system's design assumptions, reducing distribution bias."* |
| 510 | "utilizing similar datasets, hardware conditions, and statistical validation procedures" | "utilizing a comparable prompt mixture, hardware conditions, and statistical validation procedures" |
| 516–528 | Synthesis | "under the same hardware environment, prompt dataset, and inference backend" | "under the same hardware environment, prompt mixture, and inference backend" |

---

## 3. Chapter 3

| Line | Section | Current wording | Paraphrase |
| :--- | :--- | :--- | :--- |
| 536 | Research Design | "same … prompt dataset, inference backend" | "same … prompt mixture, inference backend" |
| 544 | Research Design | "A fixed stratified sample from Databricks Dolly 15k is used" | "A fixed stratified sample drawn from five public sources (300 Dolly + 200 external) is used" |
| 554–556 | Sources of Data | "The primary prompt dataset … is Databricks Dolly 15k" | Replace with the text in `DATASET_MIXING_REVISIONS.md` (see §3.1 below). Keep the Dolly description and license as the first source, then add a short paragraph for the four external sources and why they were chosen. |
| 564 | Sampling Data | "subset of prompts from the Databricks Dolly 15k dataset … final evaluation sample of 500 prompts" | "the evaluation prompts. A final evaluation sample of 500 prompts is extracted: 300 sampled proportionally from Databricks Dolly 15k and 200 from four independent benchmarks." |
| 566 | Sampling Data | "Stratification … based on the task categories of the dataset" | "Stratification is performed across the task categories of the combined five-source pool" |
| 564–570 | Sampling Data | — | Add: "The external sources were not used in classifier/prior design; per-source proportions are measured at run time and recorded in the run manifest (`dataset_sources`)." |
| 572, 574 | Sampling Data (proxy) | "selected from the 500-prompt evaluation dataset" | "selected from the Databricks Dolly 15k portion" |
| 595 | System Architecture — Input Layer | "the Databricks Dolly 15K dataset used for benchmark prompts and reference outputs" | "the standardized multi-source prompt set used for benchmark prompts and reference outputs" |
| 688 | Performance Measurement Layer | "BERTScore F1 against Databricks Dolly 15K reference outputs" | "BERTScore F1 against the reference outputs accompanying each prompt source" |
| 819 | Research Instrument — Output Quality Measurement | "reference outputs from the Databricks Dolly 15k dataset" | "reference outputs included with each source in the evaluation mixture" |
| 846 | Data Generation — Stage 1 | "The 500-prompt stratified sample from Databricks Dolly 15k is prepared" | "The 500-prompt stratified sample from the five-source evaluation mixture is prepared" |
| 852 | Data Generation — Stage 2 | "Databricks Dolly 15k reference outputs" | "the corresponding reference outputs" |
| 878 | Ethical Considerations | names only Dolly / CC BY-SA 3.0 | Add: "The four external benchmarks are used under their respective licenses (Natural Questions, Apache 2.0; TriviaQA, Apache 2.0; CNN/DailyMail, Apache 2.0; SQuAD 2.0, CC BY-SA 4.0)." |
| 998–1188 | References | only `Databricks (2023)` | Insert the four dataset references (see §4 below) alphabetically. |

### 3.1 Drop-in text for "Sources of Data" (§3, line 554–556)

> "The evaluation set consists of 500 prompts drawn from five sources: 300
> prompts sampled proportionally from the Databricks Dolly 15k dataset
> (Conover et al., 2023), and 200 prompts drawn from four independent NLP
> benchmarks — Natural Questions (Kwiatkowski et al., 2019), TriviaQA
> (Joshi et al., 2017), CNN/DailyMail (Hermann et al., 2015), and SQuAD 2.0
> (Rajpurkar et al., 2018). These external benchmarks were not used in the
> design of the BOPIS task classifier or prior probability table, making the
> evaluation independent of the system's design assumptions."

---

## 4. References to add

Insert alphabetically into the Chapter 3 references (currently only
`Databricks. (2023)` covers the prompt source). Exact entries are already
drafted in `docs/DATASET_MIXING_REVISIONS.md` (lines ~269–275):

- Hermann, K. M., Kociský, T., Grefenstette, E., Espeholt, L., Kay, W., Suleyman, M., & Blunsom, P. (2015). Teaching machines to read and comprehend. *Advances in Neural Information Processing Systems, 28*.
- Joshi, M., Choi, E., Weld, D. S., & Zettlemoyer, L. (2017). TriviaQA: A reading comprehension dataset containing 650k question-answer-evidence triples. *Proceedings of ACL 2017*.
- Kwiatkowski, T., et al. (2019). Natural Questions: A benchmark for question answering research. *Transactions of the Association for Computational Linguistics, 7*, 452–466.
- Rajpurkar, P., Jia, R., & Liang, P. (2018). Know what you don't know: Unanswerable questions for SQuAD. *Proceedings of ACL 2018*.

---

## 5. What explicitly does NOT change

Design, the five-stage method, the Wilcoxon treatment, EIR/SRR/QRR thresholds,
the configuration space, the GP/Pareto pipeline, the base model, and the
50-prompt proxy search budget. Only the **prompt source composition** and the
**independence/bias sentence** change. This is a rename/paraphrase pass, not a
design change.

---

## 6. Verified source facts (from the code, 2026)

From `bopis/dataset.py`:

| Source | Key | HF dataset | Split | Share | License (in code) |
| :--- | :--- | :--- | :--- | ---: | :--- |
| Dolly 15k | `databricks-dolly-15k` | `databricks/databricks-dolly-15k` | train | 300 | CC BY-SA 3.0 |
| Natural Questions | `natural_questions` | `google-research-datasets/natural_questions` | validation | 60 | Apache 2.0 |
| TriviaQA | `trivia_qa` | `mandarjoshi/trivia_qa` | validation | 60 | Apache 2.0 |
| CNN/DailyMail | `cnn_dailymail` | `abisee/cnn_dailymail` | test | 50 | Apache 2.0 |
| SQuAD 2.0 | `squad_v2` | `rajpurkar/squad_v2` | validation | 30 | CC BY-SA 4.0 |

> **Verify before writing §3 Ethics:** the code labels Natural Questions as
> Apache 2.0, but the upstream `google-research-datasets/natural_questions` card
> states CC BY-SA 3.0. Confirm the correct license so the ethical-considerations
> paragraph matches reality, and correct `bopis/dataset.py` if the code is wrong.

---

## 7. Open items / decisions

1. **License of Natural Questions** — resolve code vs upstream (§6).
2. **`fitted_references.json` is stale** — it holds 500 Dolly-only reference
   pairs, so it predates the 60/40 mix. The manuscript should not cite it as the
   mixed evaluation reference; regenerate before a real run.
3. **Workbench "Dataset prompt" button serves Dolly only.** The UI bridge loads
   `self.dolly` from `data/databricks-dolly-15k.jsonl` alone
   (`bopis/ui_server.py:405`); the four external datasets feed only the batch
   evaluation pipeline, not the live arena. Not a study problem, but it is why
   the button shows `dolly-…` ids. Decide whether to relabel the button "Dolly
   replay" or extend the bridge to sample the mixed pool.
