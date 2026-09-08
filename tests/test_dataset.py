"""Dataset filtering, largest-remainder allocation, and stratified sampling.

Network access is never required: the real Dolly file is used only if it is
already cached under ``data/``, and the synthetic fallback covers everything
else.
"""

from __future__ import annotations

import os
import unittest

from bopis import dataset
from bopis.dataset import (
    DOLLY_FILENAME,
    MAX_PROMPT_TOKENS,
    REAL_CATEGORY_COUNTS,
    Prompt,
    build_samples,
    filter_rows,
    largest_remainder,
    stratified_sample,
    synthetic_samples,
    template_hash,
)
from bopis.tasks import TASK_KEYS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestVerifiedDatasetFacts(unittest.TestCase):
    def test_category_counts_sum_to_the_published_row_count(self) -> None:
        """Dolly 15k has 15,011 rows across eight categories."""
        self.assertEqual(sum(REAL_CATEGORY_COUNTS.values()), 15011)
        self.assertEqual(set(REAL_CATEGORY_COUNTS), set(TASK_KEYS))

    def test_open_qa_is_the_largest_category(self) -> None:
        largest = max(REAL_CATEGORY_COUNTS, key=REAL_CATEGORY_COUNTS.get)
        self.assertEqual(largest, "open_qa")

    def test_creative_writing_is_the_smallest(self) -> None:
        smallest = min(REAL_CATEGORY_COUNTS, key=REAL_CATEGORY_COUNTS.get)
        self.assertEqual(smallest, "creative_writing")


class TestLargestRemainder(unittest.TestCase):
    def test_documented_500_allocation(self) -> None:
        """The exact allocation the module docstring claims.

        Proportional targets are fractional, so seats go by largest remainder.
        These numbers must be reproducible or the sample is not replicable.
        """
        allocation = largest_remainder(REAL_CATEGORY_COUNTS, 500)
        self.assertEqual(
            allocation,
            {
                "open_qa": 125,
                "general_qa": 73,
                "classification": 71,
                "closed_qa": 59,
                "brainstorming": 59,
                "information_extraction": 50,
                "summarization": 39,
                "creative_writing": 24,
            },
        )
        self.assertEqual(sum(allocation.values()), 500)

    def test_always_sums_to_the_requested_total(self) -> None:
        for total in (7, 10, 50, 100, 333, 500, 1000):
            allocation = largest_remainder(REAL_CATEGORY_COUNTS, total)
            self.assertEqual(sum(allocation.values()), total, f"total={total}")

    def test_is_proportional_within_one_seat(self) -> None:
        total = 500
        population = sum(REAL_CATEGORY_COUNTS.values())
        allocation = largest_remainder(REAL_CATEGORY_COUNTS, total)
        for key, count in REAL_CATEGORY_COUNTS.items():
            exact = total * count / population
            self.assertLessEqual(abs(allocation[key] - exact), 1.0, key)

    def test_never_over_allocates_a_stratum(self) -> None:
        counts = {"a": 2, "b": 100}
        allocation = largest_remainder(counts, 50)
        self.assertLessEqual(allocation["a"], 2)
        self.assertEqual(sum(allocation.values()), 50)

    def test_floor_guarantees_representation(self) -> None:
        """Amendment A-21: small strata must not round to zero.

        Without a floor, creative_writing can vanish from a 50-prompt subset and
        the "preserves proportional representation" claim becomes false.
        """
        tiny = {"big": 10000, "small": 3}
        without = largest_remainder(tiny, 50)
        self.assertEqual(without["small"], 0)
        with_floor = largest_remainder(tiny, 50, floor_per_stratum=1)
        self.assertGreaterEqual(with_floor["small"], 1)
        self.assertEqual(sum(with_floor.values()), 50)

    def test_floor_exceeding_total_raises(self) -> None:
        with self.assertRaises(ValueError):
            largest_remainder({str(i): 10 for i in range(10)}, 5, floor_per_stratum=1)

    def test_empty_and_zero_cases(self) -> None:
        self.assertEqual(largest_remainder({}, 10), {})
        allocation = largest_remainder({"a": 5}, 0)
        self.assertEqual(sum(allocation.values()), 0)

    def test_is_deterministic(self) -> None:
        first = largest_remainder(REAL_CATEGORY_COUNTS, 137)
        second = largest_remainder(REAL_CATEGORY_COUNTS, 137)
        self.assertEqual(first, second)


class TestPrompt(unittest.TestCase):
    def test_renders_context_when_present(self) -> None:
        prompt = Prompt(1, "id", "closed_qa", "What year?", "Some context.", "1999")
        self.assertIn("What year?", prompt.text)
        self.assertIn("Some context.", prompt.text)

    def test_omits_empty_context(self) -> None:
        prompt = Prompt(1, "id", "open_qa", "What year?", "", "1999")
        self.assertEqual(prompt.text, "What year?")

    def test_token_estimate_is_positive(self) -> None:
        prompt = Prompt(1, "id", "open_qa", "hi", "", "yes")
        self.assertGreaterEqual(prompt.estimated_tokens, 1)

    def test_token_estimate_scales_with_length(self) -> None:
        short = Prompt(1, "i", "open_qa", "a" * 40, "", "r")
        long = Prompt(2, "i", "open_qa", "a" * 400, "", "r")
        self.assertGreater(long.estimated_tokens, short.estimated_tokens)

    def test_template_hash_is_stable(self) -> None:
        self.assertEqual(template_hash(), template_hash())
        self.assertEqual(len(template_hash()), 16)


class TestFiltering(unittest.TestCase):
    def test_drops_rows_without_a_reference_response(self) -> None:
        """BERTScore needs a reference, so an empty response is unusable."""
        rows = [
            {"instruction": "q", "context": "", "response": "", "category": "open_qa"},
            {"instruction": "q", "context": "", "response": "a", "category": "open_qa"},
        ]
        kept, report = filter_rows(rows)
        self.assertEqual(len(kept), 1)
        self.assertEqual(report.dropped_missing_response, 1)

    def test_drops_rows_without_an_instruction(self) -> None:
        rows = [
            {"instruction": "  ", "context": "", "response": "a", "category": "open_qa"}
        ]
        kept, report = filter_rows(rows)
        self.assertEqual(kept, [])
        self.assertEqual(report.dropped_missing_instruction, 1)

    def test_drops_unknown_categories(self) -> None:
        rows = [
            {"instruction": "q", "context": "", "response": "a", "category": "weird"}
        ]
        kept, report = filter_rows(rows)
        self.assertEqual(kept, [])
        self.assertEqual(report.dropped_unknown_category, 1)

    def test_drops_over_length_prompts(self) -> None:
        rows = [
            {
                "instruction": "q",
                "context": "x" * (MAX_PROMPT_TOKENS * 8),
                "response": "a",
                "category": "closed_qa",
            }
        ]
        kept, report = filter_rows(rows)
        self.assertEqual(kept, [])
        self.assertEqual(report.dropped_too_long, 1)

    def test_report_accounts_for_every_input_row(self) -> None:
        rows = [
            {"instruction": "q", "context": "", "response": "a", "category": "open_qa"},
            {"instruction": "", "context": "", "response": "a", "category": "open_qa"},
            {"instruction": "q", "context": "", "response": "", "category": "open_qa"},
            {"instruction": "q", "context": "", "response": "a", "category": "nope"},
        ]
        kept, report = filter_rows(rows)
        accounted = (
            report.n_kept
            + report.dropped_missing_response
            + report.dropped_missing_instruction
            + report.dropped_unknown_category
            + report.dropped_too_long
        )
        self.assertEqual(accounted, report.n_input)
        self.assertEqual(len(kept), report.n_kept)


class TestStratifiedSampling(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = synthetic_samples(eval_size=500, proxy_size=50, seed=1234)

    def test_evaluation_set_has_the_requested_size(self) -> None:
        self.assertEqual(len(self.samples.evaluation), 500)

    def test_proxy_subset_has_the_requested_size(self) -> None:
        self.assertEqual(len(self.samples.proxy), 50)

    def test_proxy_is_nested_inside_the_evaluation_set(self) -> None:
        """Chapter 3 draws the proxy from the 500, not from the raw pool."""
        evaluation_ids = {p.prompt_id for p in self.samples.evaluation}
        for prompt in self.samples.proxy:
            self.assertIn(prompt.prompt_id, evaluation_ids)

    def test_no_duplicates_in_either_set(self) -> None:
        self.assertEqual(
            len({p.prompt_id for p in self.samples.evaluation}), 500
        )
        self.assertEqual(len({p.prompt_id for p in self.samples.proxy}), 50)

    def test_every_category_is_represented_in_the_proxy(self) -> None:
        self.assertEqual(len(self.samples.proxy_distribution()), 8)
        for count in self.samples.proxy_distribution().values():
            self.assertGreaterEqual(count, 1)

    def test_indices_are_one_based_and_contiguous(self) -> None:
        indices = sorted(p.index for p in self.samples.evaluation)
        self.assertEqual(indices, list(range(1, 501)))

    def test_is_reproducible_for_a_fixed_seed(self) -> None:
        again = synthetic_samples(eval_size=500, proxy_size=50, seed=1234)
        self.assertEqual(
            [p.prompt_id for p in self.samples.evaluation],
            [p.prompt_id for p in again.evaluation],
        )
        self.assertEqual(
            [p.prompt_id for p in self.samples.proxy],
            [p.prompt_id for p in again.proxy],
        )

    def test_different_seeds_give_different_samples(self) -> None:
        other = synthetic_samples(eval_size=500, proxy_size=50, seed=4321)
        self.assertNotEqual(
            [p.prompt_id for p in self.samples.evaluation],
            [p.prompt_id for p in other.evaluation],
        )

    def test_task_proportions_sum_to_one(self) -> None:
        proportions = self.samples.task_proportions()
        self.assertAlmostEqual(sum(proportions.values()), 1.0, places=12)

    def test_rows_flag_proxy_membership(self) -> None:
        rows = self.samples.rows()
        self.assertEqual(len(rows), 500)
        self.assertEqual(sum(1 for row in rows if row["in_proxy_subset"]), 50)

    def test_serializes_with_provenance(self) -> None:
        payload = self.samples.as_dict()
        for key in (
            "source",
            "license",
            "seed",
            "n_evaluation",
            "n_proxy",
            "evaluation_distribution",
            "filter_report",
            "prompt_template_hash",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["license"], "CC BY-SA 3.0")

    def test_sampling_more_than_the_pool_is_capped(self) -> None:
        pool = [
            Prompt(-1, f"p{i}", "open_qa", "q", "", "a") for i in range(10)
        ]
        chosen = stratified_sample(pool, 50, seed=0)
        self.assertLessEqual(len(chosen), 10)

    def test_build_samples_assigns_indices_before_nesting(self) -> None:
        pool = [
            Prompt(-1, f"p{i}", TASK_KEYS[i % 8], "q", "", "a") for i in range(200)
        ]
        _kept, report = filter_rows(
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
        built = build_samples(pool, report, eval_size=40, proxy_size=8, seed=5)
        self.assertTrue(all(p.index > 0 for p in built.evaluation))
        self.assertTrue(all(p.index > 0 for p in built.proxy))


class TestRealDollyIfCached(unittest.TestCase):
    """Uses the real file only when it is already downloaded."""

    def setUp(self) -> None:
        self.path = os.path.join(REPO_ROOT, "data", DOLLY_FILENAME)
        if not os.path.exists(self.path):
            self.skipTest(
                "Dolly 15k not cached; run `python -m bopis dataset` to fetch it"
            )

    def test_row_count_and_schema(self) -> None:
        rows = dataset.load_jsonl(self.path)
        self.assertEqual(len(rows), 15011)
        self.assertEqual(
            set(rows[0]), {"instruction", "context", "response", "category"}
        )

    def test_category_counts_match_the_recorded_values(self) -> None:
        rows = dataset.load_jsonl(self.path)
        counts = {}
        for row in rows:
            counts[row["category"]] = counts.get(row["category"], 0) + 1
        self.assertEqual(counts, REAL_CATEGORY_COUNTS)

    def test_real_sample_is_exactly_500(self) -> None:
        samples = dataset.load_samples(
            data_dir=os.path.join(REPO_ROOT, "data"), allow_download=False
        )
        self.assertEqual(len(samples.evaluation), 500)
        self.assertEqual(len(samples.proxy), 50)
        self.assertEqual(len(samples.proxy_distribution()), 8)


if __name__ == "__main__":
    unittest.main()
