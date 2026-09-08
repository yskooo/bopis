"""Output-quality scoring.

Kept in its own package because it is the one part of BOPIS that may import
third-party code. :mod:`bopis.quality.bertscore` needs a transformer forward
pass, so it depends on ``transformers`` and ``torch``.

**This module must not import that one at module scope.** The measurement core
imports ``bopis.quality`` to obtain the scorer *interface*; loading torch during
a measurement run would be both slow and a violation of the dependency policy
enforced by ``tests/test_provenance.py``. Use :func:`get_scorer`, which imports
lazily and only when scoring is actually requested.

Standard library only.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Protocol, Sequence


@dataclasses.dataclass
class QualityScore:
    """One candidate/reference comparison."""

    f1: float
    precision: Optional[float] = None
    recall: Optional[float] = None
    scorer: str = "unknown"
    baseline_rescaled: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "quality_f1": self.f1,
            "quality_precision": self.precision,
            "quality_recall": self.recall,
            "scorer": self.scorer,
            "baseline_rescaled": self.baseline_rescaled,
        }


class Scorer(Protocol):
    """Scores generated text against reference text."""

    name: str

    def score(
        self, candidates: Sequence[str], references: Sequence[str]
    ) -> List[QualityScore]: ...

    def describe(self) -> Dict[str, object]: ...


def get_scorer(name: str = "bertscore", **kwargs) -> Scorer:
    """Construct a scorer by name, importing its dependencies lazily.

    Raises :class:`RuntimeError` with actionable instructions when the requested
    scorer's dependencies are missing, rather than failing with a bare
    ``ImportError`` several frames deep.
    """
    if name == "bertscore":
        try:
            from bopis.quality.bertscore import BertScoreScorer
        except ImportError as exc:
            raise RuntimeError(
                "BERTScore requires the evaluation dependencies, which the "
                "measurement core deliberately does not install. Run:\n"
                "    python -m pip install transformers torch\n"
                f"(underlying import error: {exc})"
            ) from exc
        return BertScoreScorer(**kwargs)
    raise ValueError(f"unknown scorer: {name!r}")


__all__ = ["QualityScore", "Scorer", "get_scorer"]
