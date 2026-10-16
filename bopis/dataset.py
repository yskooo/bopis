"""Databricks Dolly 15k: fetch, cache, filter, and stratified sampling.

Chapter 3 specifies a fixed stratified sample of 500 prompts drawn
proportionally across the eight task categories, with a nested stratified
50-prompt proxy subset used during configuration search.

Verified dataset facts
----------------------
The dataset has **15,011 rows** with fields ``instruction``, ``context``,
``response`` and ``category``. The eight categories and their counts are:
``open_qa`` 3742, ``general_qa`` 2191, ``classification`` 2136, ``closed_qa``
1773, ``brainstorming`` 1766, ``information_extraction`` 1506,
``summarization`` 1188, ``creative_writing`` 709.

Only ``closed_qa``, ``information_extraction`` and ``summarization`` carry a
non-empty ``context``; the median ``context`` length across the whole dataset is
zero. That asymmetry matters, because those three are also the categories
Table T1 rates most quality-sensitive, and their prompts are far longer
(hundreds of tokens rather than a dozen).

Exact allocation, largest remainder
-----------------------------------
Proportional allocation of 500 over those counts gives fractional targets, so
seats are assigned by the largest-remainder (Hare-Niemeyer) method: floor every
target, then hand the leftover seats to the largest fractional remainders. On
the unfiltered counts this yields 125 / 73 / 71 / 59 / 59 / 50 / 39 / 24 = 500
exactly. Filtering runs *before* allocation, so the shipped numbers are
recomputed against the surviving pool.

Reproducibility
---------------
Sampling uses ``random.Random(seed)`` without replacement, and the selected row
IDs are written to CSV, so a reader can reconstruct the exact sample. The
downloaded file's SHA-256 is recorded in the manifest.

Standard library only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
import urllib.error
import urllib.request
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from bopis.tasks import TASK_KEYS

<<<<<<< Updated upstream
=======
# Source identifiers — recorded in manifest for reproducibility
SOURCE_DOLLY = "databricks-dolly-15k"
SOURCE_NQ    = "natural_questions"
SOURCE_TRIVIA = "trivia_qa"
SOURCE_CNN   = "cnn_dailymail"
SOURCE_SQUAD = "squad_v2"

#: The four independent benchmarks mixed into the 500-prompt evaluation set
#: alongside 300 Dolly prompts (post-defense 60/40 recommendation). ``n`` is
#: the requested share; the manifest reports the *measured* count, which can be
#: lower when a dataset is unavailable or its rows fail the length filter.
EXTERNAL_DATASETS: Dict[str, Dict[str, object]] = {
        SOURCE_NQ: {
            "n": 60,
            "dataset": "google-research-datasets/natural_questions",
            "version": "v1.0",
            "split": "validation",
            "license": "Apache 2.0",
        },
        SOURCE_TRIVIA: {
            "n": 60,
            "dataset": "mandarjoshi/trivia_qa",
            "version": "rc.nocontext",
            "split": "validation",
            "license": "Apache 2.0",
        },
        SOURCE_CNN: {
            "n": 50,
            "dataset": "abisee/cnn_dailymail",
            "version": "3.0.0",
            "split": "test",
            "license": "Apache 2.0",
        },
        SOURCE_SQUAD: {
            "n": 30,
            "dataset": "rajpurkar/squad_v2",
            "version": "v1.0",
            "split": "validation",
            "license": "CC BY-SA 4.0",
        },
    }

>>>>>>> Stashed changes
DOLLY_URL = (
    "https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
    "resolve/main/databricks-dolly-15k.jsonl"
)
DOLLY_FILENAME = "databricks-dolly-15k.jsonl"
DOLLY_LICENSE = "CC BY-SA 3.0"

#: Characters per token, for the pre-flight length filter. The authoritative
#: count comes from the backend's tokenizer at run time; this heuristic only has
#: to be conservative enough to keep prompts inside the context window
#: (amendment A-20).
CHARS_PER_TOKEN = 4.0

#: Maximum estimated prompt tokens. With ``--ctx-size 2048`` and ``n_predict``
#: up to 1024, a prompt must stay under ~1024 tokens; 900 leaves headroom for
#: the chat template and for the heuristic's error.
MAX_PROMPT_TOKENS = 900

DEFAULT_EVAL_SIZE = 500
DEFAULT_PROXY_SIZE = 50

#: The prompt template. Recorded in the manifest with its hash because the
#: template materially affects BERTScore and is otherwise unreproducible
#: (amendment A-21).
PROMPT_TEMPLATE_WITH_CONTEXT = "{instruction}\n\n{context}"
PROMPT_TEMPLATE_PLAIN = "{instruction}"


@dataclasses.dataclass
class Prompt:
    """One evaluation prompt with its reference response."""

    index: int
    prompt_id: str
    task_type: str
    instruction: str
    context: str
    response: str

    @property
    def text(self) -> str:
        """The rendered prompt sent to the model."""
        if self.context.strip():
            return PROMPT_TEMPLATE_WITH_CONTEXT.format(
                instruction=self.instruction.strip(), context=self.context.strip()
            )
        return PROMPT_TEMPLATE_PLAIN.format(instruction=self.instruction.strip())

    @property
    def estimated_tokens(self) -> int:
        return max(1, int(len(self.text) / CHARS_PER_TOKEN))

    def as_row(self, in_proxy: bool) -> Dict[str, object]:
        return {
            "prompt_index": self.index,
            "prompt_id": self.prompt_id,
            "task_type": self.task_type,
            "in_proxy_subset": in_proxy,
            "instruction_chars": len(self.instruction),
            "context_chars": len(self.context),
            "response_chars": len(self.response),
            "estimated_prompt_tokens": self.estimated_tokens,
        }


def template_hash() -> str:
    """SHA-256 of both prompt templates, for the manifest."""
    payload = (PROMPT_TEMPLATE_WITH_CONTEXT + "||" + PROMPT_TEMPLATE_PLAIN).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Fetch and cache
# --------------------------------------------------------------------------- #


def download(data_dir: str = "data", force: bool = False) -> str:
    """Download the Dolly JSONL once into *data_dir*; return its path."""
    os.makedirs(data_dir, exist_ok=True)
    target = os.path.join(data_dir, DOLLY_FILENAME)
    if os.path.exists(target) and os.path.getsize(target) > 0 and not force:
        return target
    try:
        request = urllib.request.Request(
            DOLLY_URL, headers={"User-Agent": "bopis/0.1 (thesis artifact)"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"could not download Dolly 15k from {DOLLY_URL}: {exc}. "
            "Download it manually and place it at " + target
        ) from exc
    with open(target, "wb") as handle:
        handle.write(payload)
    return target


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class FilterReport:
    n_input: int
    n_kept: int
    dropped_missing_response: int
    dropped_missing_instruction: int
    dropped_unknown_category: int
    dropped_too_long: int

    def as_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


def filter_rows(
    rows: Sequence[Dict[str, str]],
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
) -> Tuple[List[Prompt], FilterReport]:
    """Apply Chapter 3's exclusions and render each survivor as a Prompt.

    Excluded: rows with no reference ``response`` (BERTScore needs one), rows
    with no ``instruction``, rows in an unrecognized category, and rows whose
    estimated prompt length exceeds the context budget.
    """
    kept: List[Prompt] = []
    missing_response = missing_instruction = unknown_category = too_long = 0
    known = set(TASK_KEYS)

    for offset, row in enumerate(rows):
        category = (row.get("category") or "").strip()
        instruction = (row.get("instruction") or "").strip()
        context = (row.get("context") or "").strip()
        response = (row.get("response") or "").strip()

        if category not in known:
            unknown_category += 1
            continue
        if not instruction:
            missing_instruction += 1
            continue
        if not response:
            missing_response += 1
            continue

        candidate = Prompt(
            index=-1,
            prompt_id=f"dolly-{offset:05d}",
            task_type=category,
            instruction=instruction,
            context=context,
            response=response,
        )
        if candidate.estimated_tokens > max_prompt_tokens:
            too_long += 1
            continue
        kept.append(candidate)

    return kept, FilterReport(
        n_input=len(rows),
        n_kept=len(kept),
        dropped_missing_response=missing_response,
        dropped_missing_instruction=missing_instruction,
        dropped_unknown_category=unknown_category,
        dropped_too_long=too_long,
    )


# --------------------------------------------------------------------------- #
# Stratified allocation
# --------------------------------------------------------------------------- #


def largest_remainder(
    counts: Dict[str, int], total: int, floor_per_stratum: int = 0
) -> Dict[str, int]:
    """Allocate exactly *total* seats proportionally to *counts*.

    Uses the largest-remainder method: take the floor of each proportional
    target, then distribute the leftover seats to the largest fractional
    remainders (ties broken by stratum size, then name, so the result is
    deterministic).

    *floor_per_stratum* guarantees a minimum allocation to every non-empty
    stratum. This is needed for the 50-prompt proxy subset (amendment A-21):
    without it, ``creative_writing`` -- the smallest category -- can round to
    zero, and the claim that the subset "preserves proportional representation
    across task categories" would be false.
    """
    population = sum(counts.values())
    if population <= 0 or total <= 0:
        return {key: 0 for key in counts}

    nonempty = [key for key, value in counts.items() if value > 0]
    if floor_per_stratum * len(nonempty) > total:
        raise ValueError(
            f"floor of {floor_per_stratum} across {len(nonempty)} strata "
            f"exceeds the total of {total}"
        )

    allocation = {key: 0 for key in counts}
    for key in nonempty:
        allocation[key] = floor_per_stratum
    remaining = total - floor_per_stratum * len(nonempty)

    exact = {
        key: remaining * counts[key] / population for key in nonempty
    }
    floors = {key: int(math.floor(value)) for key, value in exact.items()}
    # Never allocate more than a stratum actually has.
    for key in nonempty:
        floors[key] = min(floors[key], counts[key] - allocation[key])
        allocation[key] += floors[key]

    leftover = total - sum(allocation.values())
    if leftover > 0:
        ranked = sorted(
            nonempty,
            key=lambda key: (-(exact[key] - math.floor(exact[key])), -counts[key], key),
        )
        for key in ranked:
            if leftover <= 0:
                break
            if allocation[key] < counts[key]:
                allocation[key] += 1
                leftover -= 1
    return allocation


def stratified_sample(
    prompts: Sequence[Prompt],
    size: int,
    seed: int,
    floor_per_stratum: int = 0,
) -> List[Prompt]:
    """Proportional stratified sample of *size* prompts, without replacement."""
    by_category: Dict[str, List[Prompt]] = {key: [] for key in TASK_KEYS}
    for prompt in prompts:
        by_category.setdefault(prompt.task_type, []).append(prompt)

    counts = {key: len(group) for key, group in by_category.items()}
    allocation = largest_remainder(counts, size, floor_per_stratum=floor_per_stratum)

    rng = random.Random(seed)
    chosen: List[Prompt] = []
    for key in TASK_KEYS:
        group = by_category.get(key, [])
        take = min(allocation.get(key, 0), len(group))
        if take:
            chosen.extend(rng.sample(group, take))

    rng.shuffle(chosen)
    return chosen


@dataclasses.dataclass
class SampleSet:
    """The fixed 500-prompt evaluation set and its nested 50-prompt subset."""

    evaluation: List[Prompt]
    proxy: List[Prompt]
    filter_report: FilterReport
    seed: int
    source_path: Optional[str] = None
    source_sha256: Optional[str] = None
    #: True when the build attempted the 60/40 Dolly/external mix, even if the
    #: external loaders returned nothing. Keeps the manifest honest about a
    #: mixing attempt that came up short.
    external_requested: bool = False

    def distribution(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for prompt in self.evaluation:
            counts[prompt.task_type] = counts.get(prompt.task_type, 0) + 1
        return counts

    def proxy_distribution(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for prompt in self.proxy:
            counts[prompt.task_type] = counts.get(prompt.task_type, 0) + 1
        return counts

    def task_proportions(self) -> Dict[str, float]:
        """``P(task_t)`` over the evaluation set, for the dataset-level prior."""
        counts = self.distribution()
        total = sum(counts.values()) or 1
        return {key: value / total for key, value in counts.items()}

    def rows(self) -> List[Dict[str, object]]:
        proxy_ids = {p.prompt_id for p in self.proxy}
        return [p.as_row(p.prompt_id in proxy_ids) for p in self.evaluation]

    def source_counts(self) -> Dict[str, int]:
        """Measured prompt counts per source in the evaluation set."""
        counts: Dict[str, int] = {}
        for prompt in self.evaluation:
            counts[prompt.source] = counts.get(prompt.source, 0) + 1
        return counts

    def dataset_sources(self) -> Dict[str, Dict[str, object]]:
        """Provenance of the evaluation set, counted from the prompts themselves.

        The counts are measured, never assumed: if the external loaders were
        unavailable, the entry reports ``n: 0`` instead of the requested share,
        so the manifest cannot claim a mix that did not happen.
        """
        counts = self.source_counts()
        sources: Dict[str, Dict[str, object]] = {
            SOURCE_DOLLY: {
                "n": counts.get(SOURCE_DOLLY, 0),
                "sha256": self.source_sha256,
                "license": DOLLY_LICENSE,
            }
        }
        for name, meta in EXTERNAL_DATASETS.items():
            if self.external_requested or counts.get(name, 0):
                sources[name] = dict(meta, n=counts.get(name, 0))
        return sources

    def as_dict(self) -> Dict[str, object]:
        return {
            "source": "databricks/databricks-dolly-15k",
            "license": DOLLY_LICENSE,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
<<<<<<< Updated upstream
=======
            "dataset_sources": self.dataset_sources(),
>>>>>>> Stashed changes
            "seed": self.seed,
            "n_evaluation": len(self.evaluation),
            "n_proxy": len(self.proxy),
            "evaluation_distribution": self.distribution(),
            "proxy_distribution": self.proxy_distribution(),
            "filter_report": self.filter_report.as_dict(),
            "prompt_template_hash": template_hash(),
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "chars_per_token_heuristic": CHARS_PER_TOKEN,
        }


<<<<<<< Updated upstream
=======
def load_natural_questions(seed: int, n: int = 60) -> List[Prompt]:
    try:
        import importlib

        datasets = importlib.import_module("datasets")
    except Exception:
        warnings.warn("datasets unavailable — cannot load natural_questions", RuntimeWarning)
        return []
    try:
        ds = datasets.load_dataset(
            EXTERNAL_DATASETS[SOURCE_NQ]["dataset"],
            split=EXTERNAL_DATASETS[SOURCE_NQ]["split"],
            streaming=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=min(n * 8, 1024))
        iterator = iter(ds)
        got = 0
        prompts: List[Prompt] = []
        while got < n:
            try:
                row = next(iterator)
            except StopIteration:
                break
            except Exception:
                continue  # skip bad row
            try:
                # ``annotations`` is a dict -- ``{"short_answers": [...]}`` -- not
                # a list, so indexing it as one raises KeyError and every row was
                # silently dropped (natural_questions loaded 0 prompts). Each
                # annotator's ``text`` may itself be a list of token strings.
                annotations = row.get("annotations")
                if not isinstance(annotations, dict):
                    continue
                short_answers = annotations.get("short_answers") or []
                text = ""
                for answer in short_answers:
                    if not isinstance(answer, dict):
                        continue
                    raw = answer.get("text")
                    if isinstance(raw, str) and raw.strip():
                        text = raw.strip()
                        break
                    if isinstance(raw, (list, tuple)):
                        joined = " ".join(
                            str(part).strip() for part in raw if str(part).strip()
                        ).strip()
                        if joined:
                            text = joined
                            break
                if not text:
                    continue
                question = row.get("question")
                instruction = (
                    question.get("text") if isinstance(question, dict) else None
                )
                if not instruction:
                    continue
                candidate = Prompt(
                    index=-1,
                    prompt_id=f"nq-{got}",
                    task_type="open_qa",
                    instruction=instruction,
                    context="",
                    response=text,
                    source=SOURCE_NQ,
                )
                if candidate.estimated_tokens <= MAX_PROMPT_TOKENS:
                    prompts.append(candidate)
                    got += 1
            except Exception:
                continue
        return prompts
    except Exception as exc:
        warnings.warn(f"failed to load natural_questions: {exc}", RuntimeWarning)
        return []

def load_trivia_qa(seed: int, n: int = 60) -> List[Prompt]:
    try:
        import importlib

        datasets = importlib.import_module("datasets")
    except Exception:
        warnings.warn("datasets unavailable — cannot load trivia_qa", RuntimeWarning)
        return []
    try:
        ds = datasets.load_dataset(
            EXTERNAL_DATASETS[SOURCE_TRIVIA]["dataset"],
            EXTERNAL_DATASETS[SOURCE_TRIVIA]["version"],
            split=EXTERNAL_DATASETS[SOURCE_TRIVIA]["split"],
            streaming=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=min(n * 8, 1024))
    except Exception as exc:
        warnings.warn(f"failed to load trivia_qa: {exc}", RuntimeWarning)
        return []
    prompts = []
    for row in ds:
        try:
            answer = row.get("answer", {})
            ans = answer.get("value") or answer.get("normalized_value") or answer.get("text") or ""
            if not ans and isinstance(answer, str):
                ans = answer
            if not ans:
                continue
            candidate = Prompt(
                index=-1,
                prompt_id=f"trivia-{len(prompts)}",
                task_type="closed_qa",
                instruction=row.get("question", ""),
                context="",
                response=ans,
                source=SOURCE_TRIVIA,
            )
            if candidate.estimated_tokens <= MAX_PROMPT_TOKENS:
                prompts.append(candidate)
            if len(prompts) >= n:
                break
        except Exception:
            continue
    return prompts

def load_cnn_dailymail(seed: int, n: int = 50) -> List[Prompt]:
    try:
        import importlib

        datasets = importlib.import_module("datasets")
    except Exception:
        warnings.warn("datasets unavailable — cannot load cnn_dailymail", RuntimeWarning)
        return []
    try:
        ds = datasets.load_dataset(
            EXTERNAL_DATASETS[SOURCE_CNN]["dataset"],
            EXTERNAL_DATASETS[SOURCE_CNN]["version"],
            split=EXTERNAL_DATASETS[SOURCE_CNN]["split"],
            streaming=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=min(n * 8, 1024))
    except Exception as exc:
        warnings.warn(f"failed to load cnn_dailymail: {exc}", RuntimeWarning)
        return []
    prompts = []
    for row in ds:
        try:
            candidate = Prompt(
                index=-1,
                prompt_id=f"cnn-{len(prompts)}",
                task_type="summarization",
                instruction="Summarize the following article.",
                context=row["article"],
                response=row["highlights"],
                source=SOURCE_CNN,
            )
            if candidate.estimated_tokens <= MAX_PROMPT_TOKENS:
                prompts.append(candidate)
            if len(prompts) >= n:
                break
        except Exception:
            continue
    return prompts

def load_squad_v2(seed: int, n: int = 30) -> List[Prompt]:
    try:
        import importlib

        datasets = importlib.import_module("datasets")
    except Exception:
        warnings.warn("datasets unavailable — cannot load squad_v2", RuntimeWarning)
        return []
    try:
        ds = datasets.load_dataset(
            EXTERNAL_DATASETS[SOURCE_SQUAD]["dataset"],
            split=EXTERNAL_DATASETS[SOURCE_SQUAD]["split"],
            streaming=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=min(n * 8, 1024))
    except Exception as exc:
        warnings.warn(f"failed to load squad_v2: {exc}", RuntimeWarning)
        return []
    prompts = []
    for row in ds:
        try:
            ans_text = row.get("answers", {}).get("text", [])
            if not ans_text:
                continue
            candidate = Prompt(
                index=-1,
                prompt_id=f"squad-{len(prompts)}",
                task_type="information_extraction",
                instruction=row.get("question", ""),
                context=row.get("context", ""),
                response=ans_text[0],
                source=SOURCE_SQUAD,
            )
            if candidate.estimated_tokens <= MAX_PROMPT_TOKENS:
                prompts.append(candidate)
            if len(prompts) >= n:
                break
        except Exception:
            continue
    return prompts

def load_external_prompts(seed: int) -> List[Prompt]:
    """Load and combine all external dataset prompts."""
    prompts = []
    prompts.extend(load_natural_questions(seed, n=EXTERNAL_DATASETS[SOURCE_NQ]["n"]))
    prompts.extend(load_trivia_qa(seed, n=EXTERNAL_DATASETS[SOURCE_TRIVIA]["n"]))
    prompts.extend(load_cnn_dailymail(seed, n=EXTERNAL_DATASETS[SOURCE_CNN]["n"]))
    prompts.extend(load_squad_v2(seed, n=EXTERNAL_DATASETS[SOURCE_SQUAD]["n"]))
    return prompts

>>>>>>> Stashed changes
def build_samples(
    prompts: Sequence[Prompt],
    filter_report: FilterReport,
    eval_size: int = DEFAULT_EVAL_SIZE,
    proxy_size: int = DEFAULT_PROXY_SIZE,
    seed: int = 20260101,
    source_path: Optional[str] = None,
    source_sha256: Optional[str] = None,
<<<<<<< Updated upstream
) -> SampleSet:
    """Draw the evaluation set, then the proxy subset nested inside it."""
    evaluation = stratified_sample(prompts, eval_size, seed=seed)
=======
        allow_external: bool = False,
) -> SampleSet:
    """Draw the evaluation set, then the proxy subset nested inside it."""
    # Modified: 60/40 Dolly/External split for eval_size == 500
    is_synthetic = source_path == "synthetic" or any(p.prompt_id.startswith("synthetic-") for p in prompts)
    external_requested = False
    external: List[Prompt] = []
    if eval_size == DEFAULT_EVAL_SIZE and allow_external and not is_synthetic:
        external_requested = True
        evaluation = stratified_sample(prompts, DEFAULT_DOLLY_SHARE, seed=seed)
        external = load_external_prompts(seed=seed + 2)
        evaluation.extend(external)
        if not external:
            warnings.warn(
                "60/40 dataset mixing requested but no external prompts were loaded; "
                "evaluation set will be 100% Dolly (500 prompts) but mixing was "
                "attempted. Install 'datasets' (and ensure HF cache/network if "
                "needed) or pass `allow_external=False`/`--synthetic`.",
                RuntimeWarning,
            )
    else:
        evaluation = stratified_sample(prompts, eval_size, seed=seed)
    
    # Re-index
>>>>>>> Stashed changes
    for position, prompt in enumerate(evaluation, start=1):
        prompt.index = position

    # The proxy subset is drawn from the evaluation set, not the raw pool, so it
    # is genuinely nested; a per-stratum floor of 1 keeps small categories
    # represented.
    proxy = stratified_sample(
        evaluation, proxy_size, seed=seed + 1, floor_per_stratum=1
    )
    return SampleSet(
        evaluation=evaluation,
        proxy=proxy,
        filter_report=filter_report,
        seed=seed,
        source_path=source_path,
        source_sha256=source_sha256,
        external_requested=external_requested,
    )


def load_samples(
    data_dir: str = "data",
    eval_size: int = DEFAULT_EVAL_SIZE,
    proxy_size: int = DEFAULT_PROXY_SIZE,
    seed: int = 20260101,
    allow_download: bool = True,
) -> SampleSet:
    """Fetch (or reuse) Dolly 15k and build both samples."""
    path = os.path.join(data_dir, DOLLY_FILENAME)
    if not os.path.exists(path):
        if not allow_download:
            raise FileNotFoundError(
                f"{path} not found and downloading is disabled; run "
                "`python -m bopis dataset` first"
            )
        path = download(data_dir)

    rows = load_jsonl(path)
    prompts, report = filter_rows(rows)
    return build_samples(
        prompts,
        report,
        eval_size=eval_size,
        proxy_size=proxy_size,
        seed=seed,
        source_path=path,
        source_sha256=file_sha256(path),
    )


# --------------------------------------------------------------------------- #
# Offline synthetic fallback
# --------------------------------------------------------------------------- #

#: Verified category counts, used to shape the synthetic fallback so its
#: stratification matches the real dataset's proportions.
REAL_CATEGORY_COUNTS: Dict[str, int] = {
    "open_qa": 3742,
    "general_qa": 2191,
    "classification": 2136,
    "closed_qa": 1773,
    "brainstorming": 1766,
    "information_extraction": 1506,
    "summarization": 1188,
    "creative_writing": 709,
}


def synthetic_samples(
    eval_size: int = DEFAULT_EVAL_SIZE,
    proxy_size: int = DEFAULT_PROXY_SIZE,
    seed: int = 20260101,
) -> SampleSet:
    """Build samples without touching the network.

    Used by unit tests and by simulator runs on machines with no dataset copy.
    Category proportions follow :data:`REAL_CATEGORY_COUNTS`, and the
    context-bearing categories receive long contexts, so the length asymmetry
    that drives the truncation behaviour is preserved.
    """
    from bopis.tasks import TASK_BY_KEY

    rng = random.Random(seed)
    pool: List[Prompt] = []
    offset = 0
    for category, count in REAL_CATEGORY_COUNTS.items():
        # A pool 4x the target is ample for stratified sampling.
        for _ in range(max(8, count // 12)):
            has_context = TASK_BY_KEY[category].has_context
            instruction = (
                f"[{category}] " + " ".join(
                    rng.choice(
                        ["summarize", "identify", "explain", "list", "classify",
                         "compare", "describe", "extract"]
                    )
                    for _ in range(rng.randint(4, 12))
                )
            )
            context = (
                " ".join(f"context-token-{i}" for i in range(rng.randint(40, 260)))
                if has_context
                else ""
            )
            response = " ".join(
                f"reference-token-{i}" for i in range(rng.randint(10, 90))
            )
            pool.append(
                Prompt(
                    index=-1,
                    prompt_id=f"synthetic-{offset:05d}",
                    task_type=category,
                    instruction=instruction,
                    context=context,
                    response=response,
                )
            )
            offset += 1

    prompts, report = filter_rows(
        [
            {
                "instruction": p.instruction,
                "context": p.context,
                "response": p.response,
                "category": p.task_type,
            }
            for p in pool
        ]
    )
    for prompt, original in zip(prompts, pool):
        prompt.prompt_id = original.prompt_id

    sample = build_samples(
        prompts,
        report,
        eval_size=eval_size,
        proxy_size=proxy_size,
        seed=seed,
        source_path=None,
        source_sha256=None,
    )
    return sample
