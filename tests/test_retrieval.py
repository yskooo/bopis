"""Retrieval-based reference matching: the pool, the thresholds, the honesty.

These tests exist because this path is what keeps the quality gate firing on a
prompt the panel typed live instead of letting random search win by default
when BERTScore returns nothing. Two things must therefore be asserted rather
than assumed:

* the *accept* threshold is scaled to the backend. TF-IDF cosine over a short
  query against full questions rarely clears 0.5 even for a good match, so
  reusing an embedding-scale threshold would reject every real match and
  silently restore the exact failure this feature exists to fix. Anything
  under 0.5 is still surfaced as a low-confidence match, never hidden.
* a query sharing no vocabulary with the pool is rejected outright rather
  than matched to an unrelated answer.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from bopis.retrieval import ReferenceRetriever

#: Question/answer pairs with deliberately disjoint vocabularies so TF-IDF
#: matching has a clear right answer and a clear wrong one.
_POOL = [
    ("how do I bake a chocolate cake", "Mix flour, sugar, cocoa, eggs, then bake.", "dolly"),
    ("what is the capital of France", "The capital of France is Paris.", "dolly"),
    ("explain how photosynthesis works", "Plants convert sunlight into chemical energy.", "dolly"),
]


def _retriever_with_pool(pairs, **kwargs) -> ReferenceRetriever:
    """Build a TF-IDF retriever over *pairs* in a temp directory."""
    tmp = tempfile.mkdtemp()
    pool_path = os.path.join(tmp, "pool.json")
    with open(pool_path, "w", encoding="utf-8") as handle:
        json.dump(
            [{"question": q, "answer": a, "source": s} for q, a, s in pairs],
            handle,
            ensure_ascii=False,
        )
    retriever = ReferenceRetriever(
        pool_path=pool_path,
        use_sentence_transformers=False,
        **kwargs,
    )
    assert retriever.load_pool(), "synthetic pool failed to load"
    return retriever


class TestBackendThresholds(unittest.TestCase):
    def test_tfidf_accept_threshold_is_below_embedding_scale(self) -> None:
        retriever = _retriever_with_pool(_POOL)
        self.assertEqual(retriever.backend, "tfidf")
        self.assertLess(
            retriever.accept_similarity,
            retriever.min_similarity,
            "TF-IDF must accept below the 0.5 embedding-scale flag, or every "
            "real match would be rejected and BERTScore would stay n/a",
        )

    def test_embedding_backend_keeps_the_half_threshold(self) -> None:
        retriever = _retriever_with_pool(_POOL)
        retriever._backend = "sentence-transformers"
        self.assertEqual(retriever.accept_similarity, retriever.min_similarity)


class TestMatching(unittest.TestCase):
    def test_matching_vocabulary_is_accepted(self) -> None:
        retriever = _retriever_with_pool(_POOL)
        match = retriever.find_match("chocolate cake baking steps")
        self.assertIsNotNone(match)
        self.assertEqual(match.matched_question, _POOL[0][0])
        self.assertEqual(match.reference, _POOL[0][1])
        # The similarity is exposed so the caller applies the 0.5 honesty rule.
        self.assertGreater(match.similarity, 0.0)

    def test_disjoint_vocabulary_is_rejected(self) -> None:
        retriever = _retriever_with_pool(_POOL)
        self.assertIsNone(retriever.find_match("zzz qqq vvv www jjj"))

    def test_explicit_threshold_overrides_the_backend_default(self) -> None:
        retriever = _retriever_with_pool(_POOL)
        match = retriever.find_match("chocolate cake baking steps")
        self.assertIsNotNone(match)
        # A threshold above the observed similarity must reject it, proving the
        # caller's value -- not the backend default -- decides acceptance.
        self.assertIsNone(
            retriever.find_match(
                "chocolate cake baking steps",
                min_similarity=match.similarity + 0.05,
            )
        )

    def test_empty_pool_or_query_is_none(self) -> None:
        retriever = _retriever_with_pool(_POOL)
        self.assertIsNone(retriever.find_match(""))
        empty = ReferenceRetriever(use_sentence_transformers=False)
        self.assertIsNone(empty.find_match("anything"))


class TestPoolPersistence(unittest.TestCase):
    def test_load_pool_rebuilds_embeddings_when_cache_missing(self) -> None:
        tmp = tempfile.mkdtemp()
        pool_path = os.path.join(tmp, "pool.json")
        with open(pool_path, "w", encoding="utf-8") as handle:
            json.dump(
                [{"question": q, "answer": a, "source": s} for q, a, s in _POOL],
                handle,
            )
        # No embedding cache on disk: load_pool must still yield a usable
        # retriever rather than reporting an empty pool.
        retriever = ReferenceRetriever(
            pool_path=pool_path,
            emb_path=os.path.join(tmp, "emb.pkl"),
            use_sentence_transformers=False,
        )
        self.assertTrue(retriever.load_pool())
        self.assertEqual(len(retriever.questions), len(_POOL))
        self.assertIsNotNone(retriever.find_match("capital of France"))


if __name__ == "__main__":
    unittest.main()
