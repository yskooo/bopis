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
import importlib.util
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

#: Third-party packages each scorer needs before it can produce a number. The
#: modules that use them import lazily, so the import guard in :func:`get_scorer`
#: cannot see a missing dependency on its own -- ``bopis.quality.bertscore``
#: imports torch inside ``_ensure_loaded``, at first scoring, not at import.
#: Probing with ``find_spec`` closes that gap while still never loading torch:
#: asking whether scoring is possible must not cost a multi-second import.
SCORER_REQUIREMENTS: Dict[str, Tuple[str, ...]] = {
    "bertscore": ("torch", "transformers"),
}


def missing_dependencies(name: str) -> List[str]:
    """The packages *name*'s scorer needs that are not importable, in order.

    Returns an empty list when the scorer can run. Used by :func:`get_scorer` to
    fail with instructions at construction time, and by callers that want to
    report readiness without building a scorer.
    """
    missing: List[str] = []
    for module in SCORER_REQUIREMENTS.get(name, ()):
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(module)
    return missing


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
    ``ImportError`` several frames deep -- or, worse, constructing successfully
    and only failing at the first :meth:`Scorer.score` call, where no caller is
    left to catch it usefully. :func:`missing_dependencies` is checked first for
    exactly that reason.
    """
    if name == "bertscore":
        try:
            missing = missing_dependencies(name)
            if missing:
                raise ImportError(f"no module named {', '.join(missing)}")
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


__all__ = [
    "SCORER_REQUIREMENTS",
    "QualityScore",
    "Scorer",
    "get_scorer",
    "missing_dependencies",
]
