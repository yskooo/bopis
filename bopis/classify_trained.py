"""Supervised task classifier trained on Databricks Dolly 15k.

Why this module exists
----------------------
:mod:`bopis.classify` infers a prompt's Dolly category from hand-written rules.
Measured against Dolly's own labels it reaches 49.7% exact 8-way accuracy. Dolly
ships 15,011 human-labelled ``instruction -> category`` pairs, which is a
textbook supervised text-classification dataset, so the rules can be replaced by
a model *fitted* to those labels and the improvement can be measured rather than
asserted.

This is also the component that makes the "we trained a model on our dataset"
claim literally true, with a held-out test set and a baseline to beat. Note what
it is *not*: it is not training the language model. See
``docs/STATUS_AND_ACTION_ITEMS.md`` §6.

Model
-----
Multinomial Naive Bayes with add-alpha (Lidstone) smoothing over a bag of
discrete features. Chosen because:

* it is a genuine supervised classifier with a closed-form maximum-likelihood
  fit, so there is no optimizer to tune or seed;
* it needs no third-party package, preserving the project's standard-library-only
  policy for everything but the BERTScore stage;
* it is the conventional strong baseline for small-vocabulary topical text
  classification, and Dolly's categories are largely distinguished by
  imperative verbs -- exactly the signal a bag-of-words model captures.

``alpha`` is the single hyperparameter and is selected on a validation split,
never on test.

Features
--------
Four kinds, all discrete tokens fed to the same multinomial model:

* ``w:<token>`` -- lowercased word unigrams from the instruction.
* ``first:<token>`` -- the instruction's leading word, as its own feature. Dolly
  annotators worked from per-category prompts, so the leading imperative
  ("summarize", "classify", "brainstorm") is unusually diagnostic and deserves
  weight independent of its unigram count.
* ``bg:<a>_<b>`` -- bigrams over the first three tokens only. Captures "what are
  some" (brainstorming) versus "what are the" (extraction) without inflating the
  vocabulary across the whole document.
* ``ctx:yes`` / ``ctx:no`` -- whether a context passage was supplied. The rule
  classifier found this to be the single most discriminating feature, since
  Dolly's context-bearing categories are exactly ``closed_qa``,
  ``summarization`` and ``information_extraction``.

The instruction and the context are tokenized separately and the context
contributes no word features -- only the ``ctx:`` flag. Context passages are long
(mean ~300 tokens versus ~14 for instructions) and would otherwise swamp the
instruction's signal with topical vocabulary that is irrelevant to *task type*.

Standard library only.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import random
import re
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from bopis import classify, tasks

#: Candidate smoothing values searched on the validation split.
ALPHA_GRID: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)

#: Train / validation / test proportions. Stratified by category.
DEFAULT_SPLIT: Tuple[float, float, float] = (0.70, 0.15, 0.15)

#: Seed for the split. Fixed so the test set is stable across runs; a moving
#: test set would let hyperparameter choices leak into the reported figure.
DEFAULT_SEED = 20260101

#: Features occurring in fewer than this many training documents are dropped.
#: Prunes hapax tokens (names, typos) that cannot generalize and would otherwise
#: dominate the vocabulary.
MIN_DOCUMENT_FREQUENCY = 2

_WORD = re.compile(r"[a-z][a-z']*")


def tokenize(instruction: str, context: str = "") -> List[str]:
    """Feature tokens for one request. See the module docstring."""
    text = (instruction or "").lower()
    words = _WORD.findall(text)

    features: List[str] = [f"w:{w}" for w in words]

    if words:
        features.append(f"first:{words[0]}")
        head = words[:3]
        for a, b in zip(head, head[1:]):
            features.append(f"bg:{a}_{b}")

    features.append("ctx:yes" if (context or "").strip() else "ctx:no")
    return features


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


class MultinomialNaiveBayes:
    """Multinomial Naive Bayes with add-alpha smoothing, fitted by counting.

    The fit is closed-form maximum likelihood: class priors are label
    frequencies and per-class feature distributions are smoothed relative
    frequencies. Scoring is done in log space to avoid underflow.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = float(alpha)
        self.classes_: Tuple[str, ...] = ()
        self.vocabulary_: Dict[str, int] = {}
        self.log_prior_: Dict[str, float] = {}
        # class -> feature -> log P(feature | class)
        self.log_likelihood_: Dict[str, Dict[str, float]] = {}
        self.log_unseen_: Dict[str, float] = {}
        self.n_training_docs_ = 0

    def fit(
        self,
        documents: Sequence[Sequence[str]],
        labels: Sequence[str],
        min_df: int = MIN_DOCUMENT_FREQUENCY,
    ) -> "MultinomialNaiveBayes":
        """Fit on tokenized *documents* with *labels*."""
        if len(documents) != len(labels):
            raise ValueError("documents and labels must be the same length")
        if not documents:
            raise ValueError("cannot fit on an empty training set")

        document_frequency: Dict[str, int] = collections.Counter()
        for tokens in documents:
            for feature in set(tokens):
                document_frequency[feature] += 1
        vocabulary = {
            feature
            for feature, count in document_frequency.items()
            if count >= min_df
        }
        if not vocabulary:
            raise ValueError(
                f"no feature survived min_df={min_df}; training set too small"
            )

        self.classes_ = tuple(sorted(set(labels)))
        self.vocabulary_ = {f: i for i, f in enumerate(sorted(vocabulary))}
        self.n_training_docs_ = len(documents)

        class_counts: Dict[str, int] = collections.Counter(labels)
        feature_counts: Dict[str, collections.Counter] = {
            cls: collections.Counter() for cls in self.classes_
        }
        for tokens, label in zip(documents, labels):
            counter = feature_counts[label]
            for feature in tokens:
                if feature in vocabulary:
                    counter[feature] += 1

        vocabulary_size = len(vocabulary)
        total_docs = len(documents)

        self.log_prior_ = {
            cls: math.log(class_counts[cls] / total_docs) for cls in self.classes_
        }
        self.log_likelihood_ = {}
        self.log_unseen_ = {}
        for cls in self.classes_:
            counter = feature_counts[cls]
            total = sum(counter.values()) + self.alpha * vocabulary_size
            self.log_likelihood_[cls] = {
                feature: math.log((count + self.alpha) / total)
                for feature, count in counter.items()
            }
            # Features in the vocabulary but unseen for this class still get
            # smoothed mass; precompute it once rather than per lookup.
            self.log_unseen_[cls] = math.log(self.alpha / total)
        return self

    def predict_log_scores(self, tokens: Sequence[str]) -> Dict[str, float]:
        """Unnormalized log posterior per class."""
        if not self.classes_:
            raise RuntimeError("model is not fitted")
        scores: Dict[str, float] = {}
        for cls in self.classes_:
            likelihood = self.log_likelihood_[cls]
            unseen = self.log_unseen_[cls]
            total = self.log_prior_[cls]
            for feature in tokens:
                if feature in self.vocabulary_:
                    total += likelihood.get(feature, unseen)
            scores[cls] = total
        return scores

    def predict_proba(self, tokens: Sequence[str]) -> Dict[str, float]:
        """Normalized posterior, computed with the log-sum-exp trick."""
        scores = self.predict_log_scores(tokens)
        highest = max(scores.values())
        weights = {cls: math.exp(v - highest) for cls, v in scores.items()}
        total = sum(weights.values())
        return {cls: w / total for cls, w in weights.items()}

    def predict(self, tokens: Sequence[str]) -> str:
        scores = self.predict_log_scores(tokens)
        return max(scores, key=lambda cls: scores[cls])

    # -- persistence -------------------------------------------------------- #

    def to_dict(self) -> Dict[str, object]:
        return {
            "model": "multinomial_naive_bayes",
            "alpha": self.alpha,
            "classes": list(self.classes_),
            "n_training_docs": self.n_training_docs_,
            "vocabulary": sorted(self.vocabulary_),
            "log_prior": self.log_prior_,
            "log_likelihood": self.log_likelihood_,
            "log_unseen": self.log_unseen_,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MultinomialNaiveBayes":
        model = cls(alpha=float(payload["alpha"]))  # type: ignore[arg-type]
        model.classes_ = tuple(payload["classes"])  # type: ignore[arg-type]
        model.vocabulary_ = {
            f: i for i, f in enumerate(payload["vocabulary"])  # type: ignore[arg-type]
        }
        model.log_prior_ = dict(payload["log_prior"])  # type: ignore[arg-type]
        model.log_likelihood_ = {
            k: dict(v)
            for k, v in payload["log_likelihood"].items()  # type: ignore[union-attr]
        }
        model.log_unseen_ = dict(payload["log_unseen"])  # type: ignore[arg-type]
        n_docs = payload.get("n_training_docs", 0)
        model.n_training_docs_ = int(n_docs)  # type: ignore[arg-type]
        return model


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def stratified_split(
    rows: Sequence[Mapping[str, str]],
    fractions: Tuple[float, float, float] = DEFAULT_SPLIT,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[Mapping[str, str]], List[Mapping[str, str]], List[Mapping[str, str]]]:
    """Split *rows* into train/validation/test, stratified by ``category``.

    Stratification matters here because Dolly is markedly unbalanced --
    ``open_qa`` has 3,742 rows against ``creative_writing``'s 709 -- and an
    unstratified split would give the rarer categories unstable test counts.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1.0, got {sum(fractions)}")

    by_category: Dict[str, List[Mapping[str, str]]] = collections.defaultdict(list)
    for row in rows:
        category = row.get("category", "")
        if category in tasks.TASK_BY_KEY:
            by_category[category].append(row)

    rng = random.Random(seed)
    train: List[Mapping[str, str]] = []
    validation: List[Mapping[str, str]] = []
    test: List[Mapping[str, str]] = []

    for category in sorted(by_category):
        group = list(by_category[category])
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(fractions[0] * n))
        n_validation = int(round(fractions[1] * n))
        train.extend(group[:n_train])
        validation.extend(group[n_train:n_train + n_validation])
        test.extend(group[n_train + n_validation:])

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

_QA_PAIR = frozenset({"open_qa", "general_qa"})


@dataclasses.dataclass(frozen=True)
class Scores:
    """Accuracy of one classifier on one split, on the three bases."""

    n: int
    exact: int
    qa_merged: int
    sensitivity: int
    per_category: Dict[str, Tuple[int, int]]
    confusion: Dict[str, Dict[str, int]]

    @property
    def exact_accuracy(self) -> float:
        return self.exact / self.n if self.n else 0.0

    @property
    def qa_merged_accuracy(self) -> float:
        return self.qa_merged / self.n if self.n else 0.0

    @property
    def sensitivity_accuracy(self) -> float:
        return self.sensitivity / self.n if self.n else 0.0


def score_predictions(
    truths: Sequence[str], predictions: Sequence[str]
) -> Scores:
    """Score parallel sequences of true and predicted category keys."""
    if len(truths) != len(predictions):
        raise ValueError("truths and predictions must be the same length")

    per_category: Dict[str, List[int]] = {k: [0, 0] for k in tasks.TASK_KEYS}
    confusion: Dict[str, Dict[str, int]] = {k: {} for k in tasks.TASK_KEYS}
    exact = qa_merged = sensitivity = 0

    for truth, prediction in zip(truths, predictions):
        per_category[truth][1] += 1
        if prediction == truth:
            exact += 1
            qa_merged += 1
            sensitivity += 1
            per_category[truth][0] += 1
        else:
            if {truth, prediction} == _QA_PAIR:
                qa_merged += 1
            if (
                tasks.TASK_BY_KEY[prediction].sensitivity
                == tasks.TASK_BY_KEY[truth].sensitivity
            ):
                sensitivity += 1
        confusion[truth][prediction] = confusion[truth].get(prediction, 0) + 1

    return Scores(
        n=len(truths),
        exact=exact,
        qa_merged=qa_merged,
        sensitivity=sensitivity,
        per_category={k: (v[0], v[1]) for k, v in per_category.items()},
        confusion=confusion,
    )


def _rows_to_xy(
    rows: Sequence[Mapping[str, str]]
) -> Tuple[List[List[str]], List[str]]:
    documents = [
        tokenize(row.get("instruction", ""), row.get("context", "")) for row in rows
    ]
    labels = [row["category"] for row in rows]
    return documents, labels


def rule_baseline_predictions(rows: Sequence[Mapping[str, str]]) -> List[str]:
    """Predictions from the hand-written rules, for the same rows."""
    return [
        classify.classify_prompt(
            row.get("instruction", ""), row.get("context", "")
        ).task_key
        for row in rows
    ]


@dataclasses.dataclass
class TrainingReport:
    """Everything needed to defend the numbers."""

    n_train: int
    n_validation: int
    n_test: int
    alpha: float
    alpha_validation: Dict[float, float]
    vocabulary_size: int
    learned_test: Scores
    rule_test: Scores
    learned_train: Scores

    def format_text(self) -> str:
        lines = [
            "Supervised task classifier -- Multinomial Naive Bayes on Dolly 15k",
            "=" * 72,
            f"  split            train {self.n_train} / validation "
            f"{self.n_validation} / test {self.n_test}"
            f"  (stratified, seed {DEFAULT_SEED})",
            f"  vocabulary       {self.vocabulary_size} features "
            f"(min_df={MIN_DOCUMENT_FREQUENCY})",
            f"  alpha            {self.alpha} (selected on validation)",
            "",
            "  alpha search (validation exact accuracy):",
        ]
        for alpha in sorted(self.alpha_validation):
            marker = "  <-- selected" if alpha == self.alpha else ""
            lines.append(
                f"    alpha={alpha:<6} {self.alpha_validation[alpha]:6.1%}{marker}"
            )

        lines += [
            "",
            "  HELD-OUT TEST SET -- learned vs. rule baseline",
            "  " + "-" * 60,
            f"  {'basis':<26}{'rules':>10}{'learned':>10}{'delta':>10}",
        ]
        rows = [
            ("exact 8-way", self.rule_test.exact_accuracy,
             self.learned_test.exact_accuracy),
            ("open/general_qa merged", self.rule_test.qa_merged_accuracy,
             self.learned_test.qa_merged_accuracy),
            ("sensitivity tier", self.rule_test.sensitivity_accuracy,
             self.learned_test.sensitivity_accuracy),
        ]
        for name, rule, learned in rows:
            lines.append(
                f"  {name:<26}{rule:>9.1%}{learned:>10.1%}"
                f"{learned - rule:>+10.1%}"
            )

        lines += [
            "",
            f"  (train-set exact accuracy {self.learned_train.exact_accuracy:.1%} "
            f"-- gap to test indicates overfitting)",
            "",
            f"  {'category':<24}{'rules':>8}{'learned':>9}{'test n':>8}",
            "  " + "-" * 49,
        ]
        for task in tasks.TASK_TYPES:
            lc, lt = self.learned_test.per_category.get(task.key, (0, 0))
            rc, _ = self.rule_test.per_category.get(task.key, (0, 0))
            if not lt:
                continue
            lines.append(
                f"  {task.key:<24}{rc / lt:>7.1%}{lc / lt:>9.1%}{lt:>8}"
            )

        lines += ["", "  Most frequent learned-model confusions:"]
        pairs: List[Tuple[int, str, str]] = []
        for truth, row in self.learned_test.confusion.items():
            for prediction, count in row.items():
                if truth != prediction:
                    pairs.append((count, truth, prediction))
        pairs.sort(reverse=True)
        for count, truth, prediction in pairs[:6]:
            lines.append(f"    {truth:<24} -> {prediction:<24} {count:>4}")

        lines += [
            "",
            "  NOTE ON FAIRNESS: the rule weights in bopis/classify.py were hand-",
            "  tuned while observing whole-corpus accuracy, so the rule baseline",
            "  has effectively seen this test set. Its figure is therefore",
            "  optimistic, and the learned model's margin is a lower bound.",
        ]
        return "\n".join(lines)


def train_and_evaluate(
    rows: Sequence[Mapping[str, str]],
    alpha_grid: Sequence[float] = ALPHA_GRID,
    fractions: Tuple[float, float, float] = DEFAULT_SPLIT,
    seed: int = DEFAULT_SEED,
) -> Tuple[MultinomialNaiveBayes, TrainingReport]:
    """Fit on train, select ``alpha`` on validation, report on test."""
    train_rows, validation_rows, test_rows = stratified_split(
        rows, fractions=fractions, seed=seed
    )
    if not train_rows or not test_rows:
        raise ValueError("split produced an empty train or test set")

    x_train, y_train = _rows_to_xy(train_rows)
    x_validation, y_validation = _rows_to_xy(validation_rows)
    x_test, y_test = _rows_to_xy(test_rows)

    alpha_validation: Dict[float, float] = {}
    best_alpha = float(alpha_grid[0])
    best_accuracy = -1.0
    for alpha in alpha_grid:
        candidate = MultinomialNaiveBayes(alpha=alpha).fit(x_train, y_train)
        predictions = [candidate.predict(tokens) for tokens in x_validation]
        accuracy = score_predictions(y_validation, predictions).exact_accuracy
        alpha_validation[float(alpha)] = accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = float(alpha)

    # Refit at the selected alpha on train only. Deliberately NOT on
    # train+validation: the reported test figure should describe the model that
    # was actually selected, without a second fit changing it.
    model = MultinomialNaiveBayes(alpha=best_alpha).fit(x_train, y_train)

    report = TrainingReport(
        n_train=len(train_rows),
        n_validation=len(validation_rows),
        n_test=len(test_rows),
        alpha=best_alpha,
        alpha_validation=alpha_validation,
        vocabulary_size=len(model.vocabulary_),
        learned_test=score_predictions(
            y_test, [model.predict(tokens) for tokens in x_test]
        ),
        rule_test=score_predictions(y_test, rule_baseline_predictions(test_rows)),
        learned_train=score_predictions(
            y_train, [model.predict(tokens) for tokens in x_train]
        ),
    )
    return model, report


# --------------------------------------------------------------------------- #
# Prediction helpers mirroring bopis.classify's interface
# --------------------------------------------------------------------------- #


def predict_prompt(
    model: MultinomialNaiveBayes,
    instruction: str,
    context: str = "",
) -> classify.TaskPrediction:
    """Classify with *model*, returning the same type the rules return.

    Interface-compatible with :func:`bopis.classify.classify_prompt` so callers
    (and :func:`bopis.classify.prior_for_prompt`) can switch between the rule
    and learned classifiers without changing shape.
    """
    tokens = tokenize(instruction, context)
    posterior = model.predict_proba(tokens)
    scores = {key: posterior.get(key, 0.0) for key in tasks.TASK_KEYS}
    best = max(scores, key=lambda k: scores[k])
    return classify.TaskPrediction(
        task_key=best,
        confidence=scores[best],
        scores=scores,
        matched_rules=(f"naive_bayes:alpha={model.alpha}",),
        has_context=bool((context or "").strip()),
        fell_back=False,
    )


def save(model: MultinomialNaiveBayes, path: str) -> str:
    import os

    target = os.path.abspath(path)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(model.to_dict(), handle)
    return target


def load(path: str) -> MultinomialNaiveBayes:
    with open(path, encoding="utf-8") as handle:
        return MultinomialNaiveBayes.from_dict(json.load(handle))


def write_js(
    model: MultinomialNaiveBayes,
    path: str,
    report: Optional[TrainingReport] = None,
) -> str:
    """Export the fitted model as ``window.BOPIS_TASK_MODEL = {...};``.

    Lets ``bopis.html`` show the learned classifier's prediction rather than the
    weaker rule-based one. The payload is ~0.9 MB of JSON, which is immaterial
    for a page opened from local disk.

    Only the arithmetic is reimplemented in the browser; the fit happens here.
    The scoring loop is a sum of per-feature log-likelihoods, so the JavaScript
    and Python predictions agree exactly rather than approximately.
    """
    import os

    payload: Dict[str, object] = dict(model.to_dict())
    # The browser needs the vocabulary for membership tests only, and a set is
    # cheaper to build there from a list than from an object.
    payload["tasks"] = {
        t.key: {"label": t.label, "sensitivity": t.sensitivity}
        for t in tasks.TASK_TYPES
    }
    payload["priors"] = {
        key: {k: round(v, 6) for k, v in prior.items()}
        for key, prior in tasks.PRECISION_PRIOR.items()
    }
    if report is not None:
        payload["measured_accuracy"] = {
            "n_test": report.learned_test.n,
            "exact": round(report.learned_test.exact_accuracy, 4),
            "qa_merged": round(report.learned_test.qa_merged_accuracy, 4),
            "sensitivity_tier": round(report.learned_test.sensitivity_accuracy, 4),
            "rule_baseline_exact": round(report.rule_test.exact_accuracy, 4),
        }
    payload["note"] = (
        "Multinomial Naive Bayes fitted to Databricks Dolly 15k human labels. "
        "Stratified 70/15/15 split; smoothing selected on validation only. "
        "The prior it produces weights BO seeding, not the selection of x*."
    )

    target = os.path.abspath(path)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("window.BOPIS_TASK_MODEL = ")
        json.dump(payload, handle)
        handle.write(";\n")
    return target


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bopis.classify_trained",
        description=(
            "Train a Naive Bayes task classifier on Dolly 15k and report "
            "held-out accuracy against the rule-based baseline."
        ),
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--save", metavar="PATH", help="write the fitted model as JSON"
    )
    parser.add_argument(
        "--write-js",
        metavar="PATH",
        help="export the fitted model for the bopis.html task badge",
    )
    parser.add_argument(
        "--prompt", help="classify one prompt with the freshly fitted model"
    )
    parser.add_argument("--context", default="")
    args = parser.parse_args(argv)

    import os

    from bopis import dataset

    path = os.path.join(args.data_dir, dataset.DOLLY_FILENAME)
    if not os.path.exists(path):
        parser.error(
            f"{path} not found. Fetch it with: python -m bopis dataset "
            f"--data-dir {args.data_dir}"
        )

    rows = dataset.load_jsonl(path)
    model, report = train_and_evaluate(rows, seed=args.seed)
    print(report.format_text())

    if args.save:
        print()
        print(f"saved model -> {save(model, args.save)}")

    if args.write_js:
        target = write_js(model, args.write_js, report=report)
        size_kb = os.path.getsize(target) / 1024.0
        print()
        print(f"exported for the UI -> {target}  ({size_kb:.0f} KiB)")

    if args.prompt:
        prediction = predict_prompt(model, args.prompt, args.context)
        print()
        print(f"prompt      {args.prompt!r}")
        print(f"task        {prediction.task_key} ({prediction.task.label})")
        print(f"confidence  {prediction.confidence:.1%}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
