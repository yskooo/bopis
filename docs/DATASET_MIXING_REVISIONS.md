# Dataset Mixing Revisions
**BOPIS — Post-Defense Panel Recommendation**
**Status:** In Progress / Implemented
**Raised by:** Panel during defense
**Concern:** 500-prompt evaluation set drawn entirely from Dolly 15k is biased toward the same distribution BOPIS was designed around.

---

## What the panel said

> "Your evaluation dataset is biased for your model. Add more dataset."

Their concern is valid on the quality evaluation side. The 8 task categories, the prior probabilities in Table T1, and the task classifier in `classify.py` were all designed around Dolly 15k's structure. Evaluating on Dolly 15k exclusively means BOPIS is being tested on the same distribution it was built for, which weakens the generalizability claim.

**Important clarification (in case they ask again):**
The Gaussian Process does not learn from the prompts. It learns the relationship between configuration settings and energy consumption. So the prompt distribution does not bias the optimizer itself. The bias concern applies specifically to the BERTScore quality evaluation — which needs reference answers, and those references all came from one source.

---

## What is NOT changing

- The model (Qwen2.5-1.5B-Instruct) — untouched
- The GP surrogate, acquisition function, Pareto filter — untouched
- The 50-prompt proxy search set — stays Dolly 15k only (search quality is less critical than eval quality)
- Total evaluation set size — stays 500
- The 8 task categories — unchanged
- `bopis/classify.py` and `bopis/classify_trained.py` — unchanged

---

## What IS changing

The 500-prompt evaluation set changes from **100% Dolly 15k** to a **60/40 mix**:

| Source | Count | % |
|---|---|---|
| Databricks Dolly 15k (existing) | 300 | 60% |
| External independent datasets (new) | 200 | 40% |
| **Total** | **500** | **100%** |

---

## The four external datasets to add

All four were chosen because they have **reference answers built in**, which BERTScore requires. None of them were used in designing BOPIS.

| Dataset | HuggingFace ID | Task category | Prompts to use | Why |
|---|---|---|---|---|
| Natural Questions | `google-research-datasets/natural_questions` | `open_qa` | 60 | Independent open QA with verified answers |
| TriviaQA | `trivia_qa` (rc.nocontext) | `closed_qa` | 60 | Factual recall, very different style from Dolly closed QA |
| CNN / DailyMail | `cnn_dailymail` (3.0.0) | `summarization` | 50 | Real news summarization, widely used benchmark |
| SQuAD 2.0 | `rajpurkar/squad_v2` | `information_extraction` | 30 | Span extraction, structurally different from Dolly |
| **Total external** | | | **200** | |

### Why these four specifically

- They are **standard NLP benchmarks** — the panel cannot argue they were cherry-picked for BOPIS
- They each map cleanly to **one of your existing 8 task categories** — no label remapping needed at the logic level, only field name mapping
- They all have **reference answers** in a predictable field — no annotation required
- They cover the **four most quality-sensitive categories** in Table T1 (open_qa, closed_qa, summarization, information_extraction) — exactly where BERTScore matters most

---

## New prompt allocation across 500

After the mix, the 500 prompts break down as:

| Task category | From Dolly (300) | From external (200) | Total |
|---|---|---|---|
| `open_qa` | 75 | 60 (Natural Questions) | **135** |
| `closed_qa` | 35 | 60 (TriviaQA) | **95** |
| `summarization` | 23 | 50 (CNN/DailyMail) | **73** |
| `information_extraction` | 30 | 30 (SQuAD 2.0) | **60** |
| `general_qa` | 44 | 0 | **44** |
| `classification` | 43 | 0 | **43** |
| `brainstorming` | 35 | 0 | **35** |
| `creative_writing` | 15 | 0 | **15** |
| **Total** | **300** | **200** | **500** |

The 300 Dolly prompts use the same proportional allocation as before, scaled to 300 instead of 500. The `largest_remainder()` function in `dataset.py` already handles this — just change the target size passed to it.

---

## Changes to `bopis/dataset.py`

This is the primary file to change.

### 1. Add a new `ExternalPrompt` source enum or constant

At the top of the file, add source tracking:

```python
# Source identifiers — recorded in manifest for reproducibility
SOURCE_DOLLY = "databricks-dolly-15k"
SOURCE_NQ    = "natural_questions"
SOURCE_TRIVIA = "trivia_qa"
SOURCE_CNN   = "cnn_dailymail"
SOURCE_SQUAD = "squad_v2"
```

### 2. Add the `source` field to the `Prompt` dataclass

The existing `Prompt` dataclass at line 82 needs one new field:

```python
@dataclasses.dataclass
class Prompt:
    index: int
    prompt_id: str
    task_type: str
    instruction: str
    context: str
    response: str
    source: str = SOURCE_DOLLY   # ← add this, default keeps existing behaviour
```

Also add `source` to `as_row()` so it appears in the CSV output:

```python
def as_row(self, in_proxy: bool) -> Dict[str, object]:
    return {
        "prompt_index": self.index,
        "prompt_id": self.prompt_id,
        "task_type": self.task_type,
        "source": self.source,          # ← add this line
        "in_proxy_subset": in_proxy,
        ...
    }
```

### 3. Add four loader functions — one per external dataset

Each function downloads (or loads from cache) the external dataset and returns a list of `Prompt` objects with the correct field mapping.

```python
def load_natural_questions(data_dir: str, seed: int, n: int = 60) -> List[Prompt]:
    """
    Maps NQ 'question' → instruction, 'short_answers' → response.
    Only rows with at least one short answer are kept.
    task_type = 'open_qa', source = SOURCE_NQ
    """
    ...

def load_trivia_qa(data_dir: str, seed: int, n: int = 60) -> List[Prompt]:
    """
    Uses rc.nocontext split. Maps 'question' → instruction, 'answer.value' → response.
    task_type = 'closed_qa', source = SOURCE_TRIVIA
    """
    ...

def load_cnn_dailymail(data_dir: str, seed: int, n: int = 50) -> List[Prompt]:
    """
    Uses version 3.0.0. Maps article → context, 'summarize this' → instruction,
    highlights → response.
    task_type = 'summarization', source = SOURCE_CNN
    """
    ...

def load_squad_v2(data_dir: str, seed: int, n: int = 30) -> List[Prompt]:
    """
    Only rows where answers.text is non-empty (answerable questions).
    Maps question → instruction, context → context, answers.text[0] → response.
    task_type = 'information_extraction', source = SOURCE_SQUAD
    """
    ...
```

### 4. Add a `load_external_prompts()` combiner

```python
def load_external_prompts(seed: int) -> List[Prompt]:
    """Load and combine all external dataset prompts."""
    prompts = []
    prompts.extend(load_natural_questions(seed, n=60))
    prompts.extend(load_trivia_qa(seed, n=60))
    prompts.extend(load_cnn_dailymail(seed, n=50))
    prompts.extend(load_squad_v2(seed, n=30))
    return prompts  # 200 total
```

### 5. Modify `SampleSet` / the main sampling function

The function that currently draws 500 from Dolly needs to:
1. Draw 300 from Dolly (using `stratified_sample()` with size=300)
2. Draw 200 from external (using `load_external_prompts()`)
3. Merge them into a single 500-prompt list
4. Re-index so `prompt_index` is 0–499 continuously
5. Record both sources in the manifest

```python
DEFAULT_EVAL_SIZE     = 500
DEFAULT_DOLLY_SHARE   = 300   # ← new constant
DEFAULT_EXTERNAL_SHARE = 200  # ← new constant
```

### 6. Apply the same token-length filter to external prompts

The existing `MAX_PROMPT_TOKENS = 900` filter in `filter_rows()` must also run on external prompts before they enter the sample. Each loader should call `estimated_tokens` and drop rows over the limit — same rule, same threshold.

### 7. Reproducibility — record sources in the manifest

The `manifest.json` written by `artifacts.py` should include a new `dataset_sources` key:

```json
"dataset_sources": {
    "dolly_15k": { "n": 300, "sha256": "...", "license": "CC BY-SA 3.0" },
    "natural_questions": { "n": 60, "version": "v1.0", "split": "validation", "license": "Apache 2.0" },
    "trivia_qa": { "n": 60, "version": "rc.nocontext", "split": "validation", "license": "Apache 2.0" },
    "cnn_dailymail": { "n": 50, "version": "3.0.0", "split": "test", "license": "Apache 2.0" },
    "squad_v2": { "n": 30, "split": "validation", "license": "CC BY-SA 4.0" }
}
```

---

## Changes to `bopis/quality/bertscore.py`

### The secondary fix — fallback for prompts with no reference

This addresses the live demo failure where the panel's arbitrary prompt had no reference answer and BERTScore returned nothing.

Add a fallback mode to `score()`:

```python
def score(
    self,
    candidates: Sequence[str],
    references: Sequence[str],
    fallback_to_self: bool = False,
) -> List[Optional[QualityScore]]:
    """
    Score each candidate against its reference, pairwise.

    If a reference is empty or None and fallback_to_self is True,
    the candidate is scored against itself (score = 1.0 by definition,
    flagged in the output as a self-reference score rather than a
    true quality score).

    If fallback_to_self is False and a reference is empty, that
    position returns None and is excluded from the aggregate mean.
    """
```

For the 500-prompt formal evaluation, `fallback_to_self=False` — keep the current strict behaviour, because all 500 prompts have references.

For the chat's single-prompt live measurement, `fallback_to_self=True` — use the default configuration's output as the reference so the quality gate still has something to evaluate against.

---

## Changes to the thesis manuscript (Chapter 3)

After the code change, update these sections:

### Sources of Data

Change from:
> "The study uses the Databricks Dolly 15k dataset..."

To:
> "The evaluation set consists of 500 prompts drawn from five sources: 300 prompts sampled proportionally from the Databricks Dolly 15k dataset (Conover et al., 2023), and 200 prompts drawn from four independent NLP benchmarks — Natural Questions (Kwiatkowski et al., 2019), TriviaQA (Joshi et al., 2017), CNN/DailyMail (Hermann et al., 2015), and SQuAD 2.0 (Rajpurkar et al., 2018). These external benchmarks were not used in the design of the BOPIS task classifier or prior probability table, making the evaluation independent of the system's design assumptions."

### Research Instrument — Table B.1 header

Add a `source` column to the prompt log table so reviewers can see which prompts came from which dataset.

### References to add

```
Hermann, K. M., Kociský, T., Grefenstette, E., Espeholt, L., Kay, W., Suleyman, M., & Blunsom, P. (2015). Teaching machines to read and comprehend. Advances in Neural Information Processing Systems, 28.

Joshi, M., Choi, E., Weld, D. S., & Zettlemoyer, L. (2017). TriviaQA: A reading comprehension dataset containing 650k question-answer-evidence triples. Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL).

Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Devlin, J., Lee, K., Toutanova, K., Jones, L., Kelcey, M., Chang, M. W., Dai, A. M., Uszkoreit, J., Le, Q., & Petrov, S. (2019). Natural Questions: A benchmark for question answering research. Transactions of the Association of Computational Linguistics, 7, 452–466.

Rajpurkar, P., Jia, R., & Liang, P. (2018). Know what you don't know: Unanswerable questions for SQuAD. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL).
```

---

## What to say to the panel when presenting the revision

> "In response to the panel's feedback, we diversified the 500-prompt evaluation set. We retained 300 prompts from Databricks Dolly 15k and added 200 prompts from four independent NLP benchmarks — Natural Questions, TriviaQA, CNN/DailyMail, and SQuAD 2.0. None of these four datasets were used in designing BOPIS's task classifier or prior probability table, so the evaluation is now independent of the system's design assumptions. The total evaluation size remains 500, the methodology is otherwise unchanged, and BERTScore continues to function because all four external datasets include reference answers."

---

## Summary of all files to change

| File | Change |
|---|---|
| `bopis/dataset.py` | Add 4 loaders, `source` field on `Prompt`, mix 300+200 in `SampleSet` |
| `bopis/quality/bertscore.py` | Add `fallback_to_self` parameter for live chat scoring |
| `bopis/artifacts.py` | Add `dataset_sources` block to `manifest.json` |
| `data/` | New cache files for the 4 external datasets land here |
| `tests/test_dataset.py` | Add tests for each external loader and the 300+200 merge |
| Chapter 3 manuscript | Update Sources of Data, add 4 references |

---

*Generated from panel feedback — October 2026*
