"""Rule-based task classification: free-text prompt -> Dolly category.

Stage 1 of the BOPIS pipeline needs ``P(precision | task)`` from
:mod:`bopis.tasks`, and that lookup is keyed by a Dolly category. During
*evaluation* the category is known -- it is Dolly's own ``category`` field, which
is ground truth. During *deployment* it is not: a chatbot user types a sentence,
with no label attached. This module closes that gap.

What this is and is not
-----------------------
This is a **rule-based classifier**, matching the "Rule-based probability table"
method already named for Stage 1 in the pipeline table. It is not a trained
model: there are no weights, no fitting step and no training corpus. Every rule
below was written by hand and is inspectable, which is the point -- the prior it
feeds is itself researcher-defined, so an opaque classifier in front of it would
add unauditable variance to an already researcher-defined quantity.

Because Dolly ships 15k human-labelled examples, the accuracy of these rules is
*measurable* rather than asserted. :func:`evaluate_against_dolly` scores the
classifier against Dolly's own labels; run it via::

    python -m bopis.classify --data-dir data

Measured accuracy, all 15,011 labelled Dolly rows (2026-09-18)
--------------------------------------------------------------
=========================  =======  ===============================================
Basis                      Accuracy Interpretation
=========================  =======  ===============================================
8-way category               49.7%  Weak. Do not claim category-level accuracy.
``open_qa``/``general_qa``   64.3%  Those two labels merged (see below).
Quality-sensitivity tier     67.5%  The figure that actually drives the prior.
=========================  =======  ===============================================

Per-category recall is strongly bimodal: ``closed_qa`` 88.3% and ``open_qa``
82.0%, against ``summarization`` 19.5% and ``brainstorming`` 31.2%.

**Why the 8-way figure is low, and why that is mostly a labelling artifact.**
The single largest error class is ``general_qa`` predicted as ``open_qa`` (1,798
rows, 12% of the corpus). Dolly distinguishes those two by whether answering
needs a specific retrievable fact or general world knowledge -- a property of the
*answer*, not of the instruction's surface form. No lexical rule can recover it,
which is why ``general_qa`` carries no positive rules here and is reachable only
as the fallback. The second largest classes (``summarization`` and
``information_extraction`` losing to ``closed_qa``) are rows whose instruction
contains no lexical cue for its category at all.

**Why 49% is nonetheless tolerable here.** The prior weights *only* the 10 seed
draws of a 30-evaluation budget (:func:`bopis.tasks.sample_seed_configs`). The 20
BO-guided steps are driven by Expected Improvement over the GP, and ``x*`` is
chosen by Pareto dominance plus the SRR/QRR thresholds. A misclassification
therefore makes the search slightly less sample-efficient; it cannot make ``x*``
wrong. Prefer ``soft=True`` in :func:`prior_for_prompt` so a low-margin
prediction degrades toward the blended prior instead of committing to a
coin-flip label.

Report these figures alongside any deployment-time use of the classifier. Do not
present rule-based classification as a learned component.

Signals used
------------
Three kinds of cue, in decreasing priority:

1. **Structural.** Whether the request carries a context passage. Dolly's
   context-bearing categories are exactly ``closed_qa``, ``summarization`` and
   ``information_extraction`` (see :data:`bopis.tasks.TASK_TYPES`), so the
   presence or absence of context is the single most discriminating feature
   available and is used as a gate, not just a weight.
2. **Imperative verb cues.** The leading verb of an instruction ("summarize",
   "classify", "brainstorm") is highly diagnostic in Dolly, which was written by
   annotators following per-category prompts.
3. **Interrogative form.** Question words with no context suggest open-domain QA.

Standard library only.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from bopis import tasks

#: Minimum score margin between the top two candidates for a prediction to be
#: called confident. Below this the prompt is genuinely ambiguous and the caller
#: should fall back to the dataset-level prior rather than a per-task one.
CONFIDENCE_MARGIN = 0.15

#: Category used when no rule fires at all. ``general_qa`` is the honest default:
#: it is Dolly's open-domain catch-all and carries Medium quality sensitivity, so
#: guessing it is neither the most nor the least conservative choice.
FALLBACK_TASK = "general_qa"


@dataclasses.dataclass(frozen=True)
class TaskPrediction:
    """The classifier's output for one prompt."""

    task_key: str
    confidence: float
    scores: Dict[str, float]
    matched_rules: Tuple[str, ...]
    has_context: bool
    fell_back: bool = False

    @property
    def task(self) -> tasks.TaskType:
        return tasks.TASK_BY_KEY[self.task_key]

    @property
    def is_confident(self) -> bool:
        """Whether the margin over the runner-up clears :data:`CONFIDENCE_MARGIN`.

        A fallback prediction is never confident. Without that guard the
        fallback's synthetic ``{fallback: 1.0, ...: 0.0}`` distribution yields a
        margin of 1.0 and reports as maximally confident, directly contradicting
        its own ``confidence`` of 0.0. Caught by cross-checking this method
        against the JavaScript port in ``bopis.html``.
        """
        if self.fell_back:
            return False
        ordered = sorted(self.scores.values(), reverse=True)
        if len(ordered) < 2:
            return True
        return (ordered[0] - ordered[1]) >= CONFIDENCE_MARGIN

    def to_dict(self) -> Dict[str, object]:
        return {
            "task_key": self.task_key,
            "task_label": self.task.label,
            "sensitivity": self.task.sensitivity,
            "confidence": round(self.confidence, 4),
            "is_confident": self.is_confident,
            "fell_back": self.fell_back,
            "has_context": self.has_context,
            "matched_rules": list(self.matched_rules),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
        }


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #

# Weights are relative and deliberately coarse -- 3.0 for an unambiguous leading
# imperative, 2.0 for a strong phrase, 1.0 for a weak hint. Precision of the
# weights is not meaningful beyond that ordering.
_RULES_BY_CATEGORY: Dict[str, Tuple[Tuple[float, str, str], ...]] = {
    "summarization": (
        (3.0, r"^\s*(please\s+)?(summarize|summarise|sum up)\b", "lead:summarize"),
        (2.0, r"\b(summary|summarize|summarise)\b", "kw:summary"),
        (2.0, r"\b(tl;?dr|in a nutshell|condense)\b", "kw:tldr"),
        (
            2.5,
            r"\b(key|main|important)\s+(points?|findings?|takeaways?|ideas?)\b",
            "kw:key-points",
        ),
        (2.0, r"\bin (your|one|a few|fewer) own words\b", "kw:own-words"),
        (2.0, r"\b(shorten|abbreviate|paraphrase|rephrase)\b", "kw:shorten"),
        (
            2.0,
            r"\bwhat (is|was) (this|the) (passage|text|article|paragraph) about\b",
            "kw:what-about",
        ),
        (1.5, r"\bgive me a (brief|short|quick|one)\b", "kw:give-brief"),
    ),
    "classification": (
        (
            3.0,
            r"^\s*(please\s+)?(classify|categorize|categorise|label)\b",
            "lead:classify",
        ),
        (2.0, r"\b(classify|categorize|categorise)\b", "kw:classify"),
        (2.0, r"\bwhich (category|class|type|group)\b", "kw:which-category"),
        (2.0, r"\b(true or false|positive or negative|yes or no)\b", "kw:binary"),
        (1.5, r"\bis (this|it|the following)\s+\w+\s+or\s+\w+", "kw:a-or-b"),
        (1.5, r"\b(identify|determine|tell me) whether\b", "kw:whether"),
        (1.0, r"\b(sort|group|bucket) (the|these|those|following)\b", "kw:sort"),
    ),
    "information_extraction": (
        (3.0, r"^\s*(please\s+)?extract\b", "lead:extract"),
        (2.0, r"\bextract\b", "kw:extract"),
        (
            2.0,
            r"\b(from|using) the (passage|text|paragraph|article|reference)\b",
            "kw:from-passage",
        ),
        (
            1.5,
            r"\b(list|give me) (all|the|every)\b.*"
            r"\b(mentioned|listed|named|referenced)\b",
            "kw:list-mentioned",
        ),
        (1.5, r"\bpull (out|from)\b", "kw:pull-out"),
        (
            1.0,
            r"\bwhat are the (names?|dates?|numbers?|values?)\b",
            "kw:what-are-names",
        ),
    ),
    "brainstorming": (
        (3.0, r"^\s*(please\s+)?brainstorm\b", "lead:brainstorm"),
        (2.5, r"\bbrainstorm\b", "kw:brainstorm"),
        (
            3.0,
            r"\b(give|list|name|suggest|tell)\s+(me\s+)?(some|a few|several|\d+)\b",
            "kw:give-some",
        ),
        # Outweighs open_qa's generic leading-interrogative cue (2.0) plus its
        # "what are" cue (1.5), which would otherwise claim this phrasing.
        (4.5, r"\bwhat are some\b", "kw:what-are-some"),
        (
            2.0,
            r"\b(ideas?|suggestions?|options?)\s+(for|to|about|on)\b",
            "kw:ideas-for",
        ),
        (2.0, r"\bways to\b", "kw:ways-to"),
        (2.0, r"^\s*(please\s+)?(list|name|suggest|recommend)\b", "lead:list"),
        (1.5, r"\b(recommend|recommendations?)\b", "kw:recommend"),
        (1.5, r"\bwhat should i\b", "kw:what-should-i"),
        (1.0, r"\ba list of\b", "kw:a-list-of"),
    ),
    "creative_writing": (
        (
            3.0,
            r"^\s*(please\s+)?(write|compose|draft|craft)\s+(a|an|me)\b",
            "lead:write-a",
        ),
        (
            2.5,
            r"\b(poem|story|haiku|sonnet|limerick|screenplay|lyrics?)\b",
            "kw:literary-form",
        ),
        (
            2.0,
            r"\bwrite (a|an)\s+(short\s+)?(story|essay|poem|letter|song|dialogue)\b",
            "kw:write-form",
        ),
        (1.5, r"\b(imagine|pretend|fictional|make up)\b", "kw:imagine"),
        (1.0, r"\bin the style of\b", "kw:in-style-of"),
    ),
    "open_qa": (
        (2.0, r"^\s*(who|what|when|where|why|how|which)\b", "lead:interrogative"),
        (1.5, r"^\s*(what|who) (is|are|was|were)\b", "kw:what-is"),
        (1.0, r"\b(explain|describe|tell me about)\b", "kw:explain"),
        (1.0, r"\bhow (do|does|did|can)\b", "kw:how-do"),
    ),
    # closed_qa is distinguished from open_qa almost entirely by the presence of
    # context, handled by the structural gate rather than by keywords.
    "closed_qa": (
        (
            1.5,
            r"\b(according to|based on|per) the "
            r"(passage|text|paragraph|article|reference|above)\b",
            "kw:according-to",
        ),
        (
            1.0,
            r"\bgiven (the|this) (passage|text|paragraph|article|context|reference)\b",
            "kw:given-passage",
        ),
    ),
    # general_qa has no positive rules: see the module docstring. Dolly separates
    # it from open_qa on a basis that is not present in the instruction's surface
    # form, so it is reachable only as the no-rule-fired fallback.
    "general_qa": (),
}

_RULE_SOURCES: Tuple[Tuple[str, float, str, str], ...] = tuple(
    (category, weight, pattern, label)
    for category, rules in _RULES_BY_CATEGORY.items()
    for weight, pattern, label in rules
)

_RULES: Tuple[Tuple[str, float, "re.Pattern[str]", str], ...] = tuple(
    (category, weight, re.compile(pattern, re.IGNORECASE), label)
    for category, weight, pattern, label in _RULE_SOURCES
)

#: Categories whose Dolly rows carry a context passage. Derived from
#: :data:`bopis.tasks.TASK_TYPES` so the two never drift apart.
_CONTEXT_CATEGORIES: Tuple[str, ...] = tuple(
    t.key for t in tasks.TASK_TYPES if t.has_context
)
_NO_CONTEXT_CATEGORIES: Tuple[str, ...] = tuple(
    t.key for t in tasks.TASK_TYPES if not t.has_context
)

#: How strongly the structural gate suppresses categories that disagree with the
#: observed presence/absence of context. Not a hard zero: an "extract the dates"
#: instruction pasted without its passage is still an extraction request.
_GATE_PENALTY = 0.35

#: Fires when a context-bearing request also looks like a question. Module-level
#: so :func:`rules_payload` can export it and the browser port stays in sync.
_CONTEXT_QUESTION_SOURCE = r"\?|^\s*(who|what|when|where|why|how|which)\b"
_CONTEXT_QUESTION_PATTERN = re.compile(_CONTEXT_QUESTION_SOURCE, re.IGNORECASE)

#: Bonus applied to ``closed_qa`` when the context-plus-question gate fires.
_CLOSED_QA_GATE_BONUS = 2.0


def classify_prompt(
    instruction: str,
    context: str = "",
) -> TaskPrediction:
    """Predict the Dolly task category for a free-text request.

    Args:
        instruction: The user's instruction or question.
        context: Any supplied passage the instruction refers to. Pass the empty
            string when the user gave none; this is a strong signal, not a
            missing value.

    Returns:
        A :class:`TaskPrediction`. ``scores`` is a normalized distribution over
        all eight categories, suitable for a soft blend of per-task priors.
    """
    text = (instruction or "").strip()
    has_context = bool((context or "").strip())

    raw: Dict[str, float] = {key: 0.0 for key in tasks.TASK_KEYS}
    matched: List[str] = []

    for category, weight, pattern, label in _RULES:
        if pattern.search(text):
            raw[category] += weight
            matched.append(label)

    # Structural gate. Context-bearing categories are implausible without a
    # passage and vice versa, so scale the disagreeing side down.
    if has_context:
        for key in _NO_CONTEXT_CATEGORIES:
            raw[key] *= _GATE_PENALTY
        # A passage with a question and no competing intent is the textbook
        # closed_qa shape. The bonus is withheld when a summarization or
        # extraction cue already fired: those categories also carry context, and
        # an unconditional bonus here drowns them out (measured: summarization
        # recall collapsed to 15%, with 741 rows lost to closed_qa).
        competing = raw["summarization"] + raw["information_extraction"]
        if competing <= 0.0 and _CONTEXT_QUESTION_PATTERN.search(text):
            raw["closed_qa"] += _CLOSED_QA_GATE_BONUS
            matched.append("gate:context+question")
    else:
        for key in _CONTEXT_CATEGORIES:
            raw[key] *= _GATE_PENALTY

    total = sum(raw.values())
    if total <= 0.0:
        scores = {key: 0.0 for key in tasks.TASK_KEYS}
        scores[FALLBACK_TASK] = 1.0
        return TaskPrediction(
            task_key=FALLBACK_TASK,
            confidence=0.0,
            scores=scores,
            matched_rules=("fallback:no-rule-fired",),
            has_context=has_context,
            fell_back=True,
        )

    scores = {key: value / total for key, value in raw.items()}
    best = max(scores, key=lambda k: scores[k])
    return TaskPrediction(
        task_key=best,
        confidence=scores[best],
        scores=scores,
        matched_rules=tuple(matched),
        has_context=has_context,
    )


def prior_for_prompt(
    instruction: str,
    context: str = "",
    permitted: Optional[Sequence[str]] = None,
    soft: bool = False,
) -> Tuple[Dict[str, float], TaskPrediction]:
    """``P(precision | prompt)``, via the predicted task category.

    This is the deployment-time analogue of
    :func:`bopis.tasks.dataset_prior`, which blends over the *known* category
    proportions of the evaluation sample. Here the category is predicted.

    Args:
        instruction: User instruction.
        context: Supplied passage, if any.
        permitted: Feasible precision variants; the prior is masked and
            renormalized onto them when given.
        soft: When ``True``, blend the per-task priors weighted by the
            classifier's own score distribution instead of committing to the
            single top category. Preferable when the prediction is not
            confident, since it degrades toward the dataset prior rather than
            betting everything on a coin-flip label.

    Returns:
        ``(prior, prediction)``.
    """
    prediction = classify_prompt(instruction, context)

    if soft:
        prior = tasks.dataset_prior(
            {k: v for k, v in prediction.scores.items() if v > 0.0}
        )
    else:
        prior = tasks.prior_for_task(prediction.task_key)

    if permitted is not None:
        prior = tasks.mask_and_renormalize(prior, permitted)
    return prior, prediction


# --------------------------------------------------------------------------- #
# Accuracy measurement against Dolly's own labels
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ClassifierReport:
    """Accuracy of the rules against Dolly's human labels."""

    n: int
    n_correct: int
    per_category: Dict[str, Tuple[int, int]]  # key -> (correct, total)
    confusion: Dict[str, Dict[str, int]]  # true -> predicted -> count
    n_fallback: int
    n_correct_sensitivity: int = 0
    n_correct_qa_merged: int = 0

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else 0.0

    @property
    def sensitivity_accuracy(self) -> float:
        """Accuracy on the High/Medium/Low tier rather than the 8-way label.

        This is the figure that matters operationally. The precision prior
        varies across categories, but the *shape* of that variation -- how much
        mass sits on the aggressive Q4_K_M variant -- is driven by the
        sensitivity tier via
        :data:`bopis.tasks._Q4_SHARE_BY_SENSITIVITY`. A prompt misrouted from
        ``brainstorming`` to ``open_qa`` (both Low) costs the optimizer almost
        nothing; one misrouted from ``closed_qa`` (High) to ``open_qa`` (Low)
        genuinely over-quantizes.
        """
        return self.n_correct_sensitivity / self.n if self.n else 0.0

    @property
    def qa_merged_accuracy(self) -> float:
        """Accuracy when ``open_qa`` and ``general_qa`` are treated as one class.

        Dolly separates these two by whether the answer needs a specific
        retrievable fact or general world knowledge -- a distinction that is not
        present in the surface form of the instruction, and which these rules
        therefore cannot recover. Reporting the merged figure separates "the
        rules are weak" from "the label boundary is not lexical."
        """
        return self.n_correct_qa_merged / self.n if self.n else 0.0

    def format_text(self) -> str:
        lines = [
            "BOPIS rule-based task classifier vs. Dolly 15k human labels",
            "=" * 64,
            f"  8-way accuracy     {self.accuracy:6.1%}  ({self.n_correct} / {self.n})",
            f"  open/general_qa    {self.qa_merged_accuracy:6.1%}"
            "  (those two labels merged)",
            f"  sensitivity tier   {self.sensitivity_accuracy:6.1%}"
            "  <-- the figure that drives the prior",
            f"  No rule fired      {self.n_fallback} prompts fell back to"
            f" {FALLBACK_TASK!r}",
            "",
            f"  {'category':<24}{'recall':>9}{'correct':>10}{'total':>8}",
            "  " + "-" * 49,
        ]
        for task in tasks.TASK_TYPES:
            correct, total = self.per_category.get(task.key, (0, 0))
            recall = correct / total if total else 0.0
            lines.append(
                f"  {task.key:<24}{recall:>8.1%}{correct:>10}{total:>8}"
            )
        lines.append("")
        lines.append("  Most frequent confusions (true -> predicted, count):")
        pairs: List[Tuple[int, str, str]] = []
        for true_key, row in self.confusion.items():
            for pred_key, count in row.items():
                if true_key != pred_key:
                    pairs.append((count, true_key, pred_key))
        pairs.sort(reverse=True)
        for count, true_key, pred_key in pairs[:8]:
            lines.append(f"    {true_key:<24} -> {pred_key:<24} {count:>5}")
        return "\n".join(lines)


def evaluate_against_dolly(rows: Sequence[Mapping[str, str]]) -> ClassifierReport:
    """Score :func:`classify_prompt` against Dolly's ``category`` field.

    Args:
        rows: Dolly records, each with ``instruction``, ``context`` and
            ``category`` keys.
    """
    per_category: Dict[str, List[int]] = {k: [0, 0] for k in tasks.TASK_KEYS}
    confusion: Dict[str, Dict[str, int]] = {k: {} for k in tasks.TASK_KEYS}
    n_correct = 0
    n = 0
    n_fallback = 0
    n_correct_sensitivity = 0
    n_correct_qa_merged = 0
    qa_pair = {"open_qa", "general_qa"}

    for row in rows:
        true_key = row.get("category", "")
        if true_key not in tasks.TASK_BY_KEY:
            continue
        prediction = classify_prompt(
            row.get("instruction", ""), row.get("context", "")
        )
        n += 1
        per_category[true_key][1] += 1
        if prediction.fell_back:
            n_fallback += 1
        if prediction.task_key == true_key:
            n_correct += 1
            per_category[true_key][0] += 1
        if (
            tasks.TASK_BY_KEY[prediction.task_key].sensitivity
            == tasks.TASK_BY_KEY[true_key].sensitivity
        ):
            n_correct_sensitivity += 1
        if prediction.task_key == true_key or {
            prediction.task_key,
            true_key,
        } == qa_pair:
            n_correct_qa_merged += 1
        confusion[true_key][prediction.task_key] = (
            confusion[true_key].get(prediction.task_key, 0) + 1
        )

    return ClassifierReport(
        n=n,
        n_correct=n_correct,
        per_category={k: (v[0], v[1]) for k, v in per_category.items()},
        confusion=confusion,
        n_fallback=n_fallback,
        n_correct_sensitivity=n_correct_sensitivity,
        n_correct_qa_merged=n_correct_qa_merged,
    )


# --------------------------------------------------------------------------- #
# Export for the browser UI
# --------------------------------------------------------------------------- #

#: Accuracy measured against all 15,011 labelled Dolly rows. Refresh by running
#: ``python -m bopis.classify --data-dir data`` and updating these three numbers
#: together with the module docstring table.
MEASURED_ACCURACY: Dict[str, float] = {
    "n": 15011.0,
    "eight_way": 0.497,
    "qa_merged": 0.643,
    "sensitivity_tier": 0.675,
}


def rules_payload() -> Dict[str, object]:
    """The rule table and priors as JSON, for ``bopis.html``.

    ``bopis.html`` is opened as a local file and talks only to ``llama-server``;
    there is no Python process behind it, so it cannot call
    :func:`classify_prompt`. Rather than maintain a second hand-written copy of
    the rules in JavaScript -- which would silently drift from the Python ones --
    the table is authored here and *exported*. The browser reimplements only the
    scoring loop, which is a dozen lines.

    The regex sources are Python patterns. They are restricted to constructs
    that mean the same thing in JavaScript's regex dialect (character classes,
    alternation, non-capturing groups, ``\\b``, ``\\s``, ``^``), so they can be
    fed straight to ``new RegExp(source, 'i')``.
    """
    import datetime

    return {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "confidence_margin": CONFIDENCE_MARGIN,
        "fallback_task": FALLBACK_TASK,
        "gate_penalty": _GATE_PENALTY,
        "closed_qa_gate_bonus": _CLOSED_QA_GATE_BONUS,
        "context_question_pattern": _CONTEXT_QUESTION_SOURCE,
        "context_categories": list(_CONTEXT_CATEGORIES),
        "no_context_categories": list(_NO_CONTEXT_CATEGORIES),
        "task_keys": list(tasks.TASK_KEYS),
        "tasks": {
            t.key: {
                "label": t.label,
                "sensitivity": t.sensitivity,
                "has_context": t.has_context,
            }
            for t in tasks.TASK_TYPES
        },
        "priors": {
            key: {k: round(v, 6) for k, v in prior.items()}
            for key, prior in tasks.PRECISION_PRIOR.items()
        },
        "rules": [
            {
                "category": category,
                "weight": weight,
                "pattern": pattern,
                "label": label,
            }
            for category, weight, pattern, label in _RULE_SOURCES
        ],
        "measured_accuracy": dict(MEASURED_ACCURACY),
        "note": (
            "Rule-based classifier. 49.7% exact 8-way accuracy on Dolly 15k; "
            "67.5% on the quality-sensitivity tier that drives the prior. The "
            "prior weights only the 10 BO seed draws, not the selection of x*."
        ),
    }


def write_js(path: str) -> str:
    """Write :func:`rules_payload` as ``window.BOPIS_TASK_RULES = {...};``.

    Matches the convention already used by ``bopis profile --write-js``.
    """
    import json
    import os

    target = os.path.abspath(path)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("window.BOPIS_TASK_RULES = ")
        json.dump(rules_payload(), handle, indent=2)
        handle.write(";\n")
    return target


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bopis.classify",
        description="Score the rule-based task classifier against Dolly's labels.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="directory holding databricks-dolly-15k.jsonl (default: data)",
    )
    parser.add_argument(
        "--prompt",
        help="classify a single prompt instead of scoring the corpus",
    )
    parser.add_argument("--context", default="", help="context passage for --prompt")
    parser.add_argument(
        "--write-js",
        metavar="PATH",
        help="export the rule table and priors for bopis.html and exit",
    )
    args = parser.parse_args(argv)

    if args.write_js:
        target = write_js(args.write_js)
        payload = rules_payload()
        print(f"wrote {target}")
        print(
            f"  {len(payload['rules'])} rules over "
            f"{len(payload['task_keys'])} task categories"
        )
        return 0

    if args.prompt:
        prediction = classify_prompt(args.prompt, args.context)
        print(f"task        {prediction.task_key} ({prediction.task.label})")
        print(f"sensitivity {prediction.task.sensitivity}")
        margin_note = "" if prediction.is_confident else "  (LOW MARGIN)"
        print(f"confidence  {prediction.confidence:.1%}{margin_note}")
        print(f"rules       {', '.join(prediction.matched_rules)}")
        prior, _ = prior_for_prompt(args.prompt, args.context)
        print("prior       " + ", ".join(f"{k}={v:.2f}" for k, v in prior.items()))
        return 0

    # Imported here so the module stays importable without the dataset present.
    import os

    from bopis import dataset

    path = os.path.join(args.data_dir, dataset.DOLLY_FILENAME)
    if not os.path.exists(path):
        parser.error(
            f"{path} not found. Fetch it first with: python -m bopis dataset "
            f"--data-dir {args.data_dir}"
        )
    report = evaluate_against_dolly(dataset.load_jsonl(path))
    print(report.format_text())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
