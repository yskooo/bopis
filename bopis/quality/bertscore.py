"""BERTScore F1 -- the one module permitted to import third-party packages.

Chapter 3 uses BERTScore F1 as the sole output-quality metric: token-level cosine
similarity between contextual embeddings of the generated and reference
responses, combined into precision, recall and F1. It is chosen over exact-match
metrics because instruction-following tasks admit many valid phrasings of the
same correct answer.

Why this module is exempt from the standard-library-only policy
---------------------------------------------------------------
Computing BERTScore requires a forward pass through a pretrained transformer.
There is no standard-library implementation of one, and hand-rolling a
BERT forward pass would no longer be canonical BERTScore -- so the honest
resolution is a stdlib measurement core plus one declared *evaluation*
dependency (amendment A-6). This module therefore imports ``transformers`` and
``torch``, and nothing in the measurement path imports this module: scoring is a
separate offline stage that reads generated text from a run's CSV logs and
writes quality scores back.

Baseline rescaling
------------------
Rescaling is **on by default** (amendment A-38). Raw BERTScore F1 occupies a
narrow high band -- unrelated sentence pairs still score around 0.8 with a
RoBERTa backbone -- which would make the ``QRR >= 98%`` retention criterion
nearly impossible to fail, and therefore uninformative. Rescaling maps scores
against a corpus of random pairings so that differences become legible.
Absolute F1 deltas should be reported alongside QRR either way.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from bopis.quality import QualityScore

#: Default embedding backbone. ``roberta-large`` layer 17 is the reference
#: configuration from Zhang et al. (2020) for English text and is what
#: ``bert-score`` selects by default, so results are comparable with published
#: figures.
DEFAULT_MODEL = "roberta-large"
DEFAULT_LAYER = 17


class BertScoreScorer:
    """BERTScore F1 over a locally cached transformer checkpoint.

    Two implementations, preferred in order:

    1. The ``bert_score`` package, which carries the published baseline-rescaling
       tables. Preferred, because it reproduces the reference implementation
       exactly.
    2. A direct ``transformers`` implementation, used when ``bert_score`` is not
       installed. Numerically equivalent for the unrescaled score; rescaling is
       unavailable, and that fact is recorded in ``baseline_rescaled`` so a
       reader can tell which path produced a given number.
    """

    name = "bertscore"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        layer: int = DEFAULT_LAYER,
        rescale_with_baseline: bool = True,
        language: str = "en",
        batch_size: int = 16,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.layer = layer
        self.rescale_with_baseline = rescale_with_baseline
        self.language = language
        self.batch_size = batch_size
        self.device = device

        self._backend: Optional[str] = None
        self._scorer = None
        self._tokenizer = None
        self._encoder = None
        self._torch = None

    # ------------------------------------------------------------------ #
    # Backend selection
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self) -> None:
        if self._backend is not None:
            return

        import torch  # noqa: F401  (third-party: see module docstring)

        self._torch = torch
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            from bert_score import BERTScorer

            self._scorer = BERTScorer(
                model_type=self.model,
                num_layers=self.layer,
                lang=self.language,
                rescale_with_baseline=self.rescale_with_baseline,
                batch_size=self.batch_size,
                device=self.device,
            )
            self._backend = "bert_score"
            return
        except ImportError:
            pass

        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._encoder = AutoModel.from_pretrained(
            self.model, output_hidden_states=True
        ).to(self.device)
        self._encoder.eval()
        self._backend = "transformers"
        if self.rescale_with_baseline:
            # Be explicit rather than silently reporting unrescaled numbers as
            # though they were rescaled.
            self.rescale_with_baseline = False

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def score(
        self, candidates: Sequence[str], references: Sequence[str]
    ) -> List[QualityScore]:
        """Score each candidate against its reference, pairwise."""
        if len(candidates) != len(references):
            raise ValueError(
                f"{len(candidates)} candidates but {len(references)} references"
            )
        if not candidates:
            return []

        self._ensure_loaded()
        if self._backend == "bert_score":
            return self._score_with_package(candidates, references)
        return self._score_with_transformers(candidates, references)

    def _score_with_package(
        self, candidates: Sequence[str], references: Sequence[str]
    ) -> List[QualityScore]:
        precision, recall, f1 = self._scorer.score(
            list(candidates), list(references)
        )
        return [
            QualityScore(
                f1=float(f1[i]),
                precision=float(precision[i]),
                recall=float(recall[i]),
                scorer=f"bertscore/{self.model}@{self.layer}",
                baseline_rescaled=self.rescale_with_baseline,
            )
            for i in range(len(candidates))
        ]

    def _score_with_transformers(
        self, candidates: Sequence[str], references: Sequence[str]
    ) -> List[QualityScore]:
        """Greedy token-matching BERTScore, computed directly.

        For each candidate token, take the maximum cosine similarity against any
        reference token (precision); symmetrically for recall; combine as the
        harmonic mean. Special tokens are excluded, as in the reference
        implementation.
        """
        torch = self._torch
        results: List[QualityScore] = []

        for start in range(0, len(candidates), self.batch_size):
            batch_candidates = list(candidates[start : start + self.batch_size])
            batch_references = list(references[start : start + self.batch_size])

            candidate_embeddings = self._embed(batch_candidates)
            reference_embeddings = self._embed(batch_references)

            for cand, ref in zip(candidate_embeddings, reference_embeddings):
                if cand.shape[0] == 0 or ref.shape[0] == 0:
                    results.append(
                        QualityScore(
                            f1=0.0,
                            precision=0.0,
                            recall=0.0,
                            scorer=f"bertscore-direct/{self.model}@{self.layer}",
                            baseline_rescaled=False,
                        )
                    )
                    continue

                similarity = torch.matmul(cand, ref.transpose(0, 1))
                precision = similarity.max(dim=1).values.mean().item()
                recall = similarity.max(dim=0).values.mean().item()
                denominator = precision + recall
                f1 = (
                    2.0 * precision * recall / denominator
                    if denominator > 0
                    else 0.0
                )
                results.append(
                    QualityScore(
                        f1=f1,
                        precision=precision,
                        recall=recall,
                        scorer=f"bertscore-direct/{self.model}@{self.layer}",
                        baseline_rescaled=False,
                    )
                )
        return results

    def _embed(self, texts: Sequence[str]):
        """L2-normalized hidden states for *texts*, special tokens dropped."""
        torch = self._torch
        encoded = self._tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._encoder(**encoded)
        hidden = outputs.hidden_states[self.layer]
        hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)

        special = self._tokenizer.all_special_ids
        input_ids = encoded["input_ids"]
        attention = encoded["attention_mask"].bool()
        keep = attention.clone()
        for token_id in special:
            keep &= input_ids != token_id

        return [hidden[i][keep[i]] for i in range(hidden.shape[0])]

    # ------------------------------------------------------------------ #

    def describe(self) -> Dict[str, object]:
        return {
            "scorer": self.name,
            "backend": self._backend or "not loaded",
            "model": self.model,
            "layer": self.layer,
            "baseline_rescaled": self.rescale_with_baseline,
            "device": self.device,
            "dependency_note": (
                "Requires transformers and torch. This is the only BOPIS module "
                "that imports third-party packages; it runs as a separate "
                "offline scoring stage outside the measurement path."
            ),
        }


def mean_f1(scores: Sequence[QualityScore]) -> float:
    """Mean F1 over *scores*, ignoring non-finite values."""
    values = [s.f1 for s in scores if s.f1 is not None and math.isfinite(s.f1)]
    if not values:
        return float("nan")
    return sum(values) / len(values)
