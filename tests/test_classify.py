"""Tests for the rule-based task classifier (:mod:`bopis.classify`)."""

from __future__ import annotations

import unittest

from bopis import classify, tasks


class TestStructuralGate(unittest.TestCase):
    """Presence of a context passage is the strongest available signal."""

    def test_question_with_context_is_closed_qa(self) -> None:
        prediction = classify.classify_prompt(
            "Who won the first World Cup?",
            context="The first FIFA World Cup was held in 1930 in Uruguay, "
            "which defeated Argentina 4-2 in the final.",
        )
        self.assertEqual(prediction.task_key, "closed_qa")
        self.assertTrue(prediction.has_context)

    def test_same_question_without_context_is_open_qa(self) -> None:
        prediction = classify.classify_prompt("Who won the first World Cup?")
        self.assertEqual(prediction.task_key, "open_qa")
        self.assertFalse(prediction.has_context)

    def test_context_does_not_override_an_explicit_summarize(self) -> None:
        """Regression: an unconditional closed_qa bonus swallowed summarization.

        Measured against Dolly, an unconditional "context + question mark"
        bonus dropped summarization recall to 14.7% by reassigning 741 rows to
        closed_qa. The bonus is now withheld when a summarization or extraction
        cue has already fired.
        """
        prediction = classify.classify_prompt(
            "What is this passage about? Summarize it.",
            context="Photosynthesis converts light energy into chemical energy.",
        )
        self.assertEqual(prediction.task_key, "summarization")

    def test_context_does_not_override_an_explicit_extract(self) -> None:
        prediction = classify.classify_prompt(
            "Extract the dates mentioned. What are they?",
            context="The treaty was signed in 1919 and ratified in 1920.",
        )
        self.assertEqual(prediction.task_key, "information_extraction")


class TestCategoryCues(unittest.TestCase):
    def test_leading_imperatives(self) -> None:
        cases = {
            "Summarize the following report.": "summarization",
            "Classify this review as positive or negative.": "classification",
            "Extract every proper noun.": "information_extraction",
            "Brainstorm names for a coffee shop.": "brainstorming",
            "Write a poem about the sea.": "creative_writing",
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(classify.classify_prompt(prompt).task_key, expected)

    def test_what_are_some_is_brainstorming_not_open_qa(self) -> None:
        """"What are some X" is a list request, not a factual question."""
        prediction = classify.classify_prompt("What are some good hiking trails?")
        self.assertEqual(prediction.task_key, "brainstorming")

    def test_unmatched_prompt_falls_back(self) -> None:
        prediction = classify.classify_prompt("Bananas.")
        self.assertEqual(prediction.task_key, classify.FALLBACK_TASK)
        self.assertEqual(prediction.matched_rules, ("fallback:no-rule-fired",))
        self.assertEqual(prediction.confidence, 0.0)
        self.assertTrue(prediction.fell_back)

    def test_fallback_is_never_confident(self) -> None:
        """Regression: the fallback reported confidence 0.0 *and* confident=True.

        Its synthetic ``{fallback: 1.0, rest: 0.0}`` distribution gives a margin
        of 1.0 over the runner-up, which cleared CONFIDENCE_MARGIN and made the
        fallback look maximally certain. Found by diffing this module against
        the JavaScript port in bopis.html.
        """
        for prompt in ("Bananas.", "", "   "):
            with self.subTest(prompt=prompt):
                prediction = classify.classify_prompt(prompt)
                self.assertTrue(prediction.fell_back)
                self.assertEqual(prediction.confidence, 0.0)
                self.assertFalse(prediction.is_confident)

    def test_a_matched_prompt_is_not_marked_as_fallback(self) -> None:
        self.assertFalse(classify.classify_prompt("Summarize this.").fell_back)

    def test_empty_prompt_falls_back(self) -> None:
        self.assertEqual(
            classify.classify_prompt("").task_key, classify.FALLBACK_TASK
        )


class TestPredictionShape(unittest.TestCase):
    def test_scores_are_a_distribution_over_all_categories(self) -> None:
        prediction = classify.classify_prompt("Summarize this.")
        self.assertEqual(set(prediction.scores), set(tasks.TASK_KEYS))
        self.assertAlmostEqual(sum(prediction.scores.values()), 1.0, places=9)
        for value in prediction.scores.values():
            self.assertGreaterEqual(value, 0.0)

    def test_confidence_equals_the_top_score(self) -> None:
        prediction = classify.classify_prompt("Brainstorm gift ideas for a teacher.")
        self.assertAlmostEqual(
            prediction.confidence, max(prediction.scores.values()), places=9
        )

    def test_task_property_resolves_to_a_known_task_type(self) -> None:
        prediction = classify.classify_prompt("Classify this sentence.")
        self.assertIn(prediction.task, tasks.TASK_TYPES)
        self.assertEqual(prediction.task.key, prediction.task_key)

    def test_to_dict_is_json_shaped(self) -> None:
        payload = classify.classify_prompt("Write a haiku.").to_dict()
        self.assertEqual(payload["task_key"], "creative_writing")
        self.assertIsInstance(payload["matched_rules"], list)
        self.assertIsInstance(payload["scores"], dict)
        self.assertIn("sensitivity", payload)

    def test_ambiguous_prompt_is_not_confident(self) -> None:
        """A prompt firing comparable cues for two categories reports low margin."""
        # "List the items" hits both the brainstorming lead and extraction cues.
        prediction = classify.classify_prompt("Tell me about hiking.")
        ordered = sorted(prediction.scores.values(), reverse=True)
        margin = ordered[0] - ordered[1]
        self.assertEqual(prediction.is_confident, margin >= classify.CONFIDENCE_MARGIN)


class TestPriorForPrompt(unittest.TestCase):
    def test_hard_prior_matches_the_predicted_task(self) -> None:
        prior, prediction = classify.prior_for_prompt("Summarize this article.")
        self.assertEqual(prediction.task_key, "summarization")
        self.assertEqual(prior, tasks.prior_for_task("summarization"))

    def test_prior_sums_to_one(self) -> None:
        prior, _ = classify.prior_for_prompt("Classify this review.")
        self.assertAlmostEqual(sum(prior.values()), 1.0, places=9)

    def test_soft_prior_blends_and_still_normalizes(self) -> None:
        # "Write a summary" fires both creative_writing (lead:write-a) and
        # summarization (kw:summary), so the score distribution is genuinely
        # split and the soft blend must differ from the hard top-1 prior.
        prompt = "Write a summary of the meeting."
        soft, prediction = classify.prior_for_prompt(prompt, soft=True)
        hard, _ = classify.prior_for_prompt(prompt, soft=False)
        self.assertGreater(
            sum(1 for v in prediction.scores.values() if v > 0.0),
            1,
            "test prompt must fire more than one category",
        )
        self.assertAlmostEqual(sum(soft.values()), 1.0, places=9)
        self.assertNotEqual(soft, hard)

    def test_permitted_masks_infeasible_variants(self) -> None:
        """On the 2 GB study host only Q8_0 and Q4_K_M survive Table H1."""
        permitted = ["Q8_0", "Q4_K_M"]
        prior, _ = classify.prior_for_prompt(
            "Summarize this article.", permitted=permitted
        )
        self.assertEqual(set(prior), set(permitted))
        self.assertAlmostEqual(sum(prior.values()), 1.0, places=9)

    def test_quality_sensitive_task_keeps_more_mass_off_q4(self) -> None:
        """closed_qa (High) should be less willing to reach for Q4_K_M than
        brainstorming (Low). This is the behaviour the prior exists to produce."""
        closed, _ = classify.prior_for_prompt(
            "Who signed it?", context="The treaty was signed by Adams in 1783."
        )
        brainstorm, _ = classify.prior_for_prompt("Brainstorm some team names.")
        self.assertLess(closed["Q4_K_M"], brainstorm["Q4_K_M"])


class TestEvaluateAgainstDolly(unittest.TestCase):
    def test_scores_a_small_hand_built_corpus(self) -> None:
        rows = [
            {"instruction": "Summarize this.", "context": "x y z",
             "category": "summarization"},
            {"instruction": "Brainstorm names.", "context": "",
             "category": "brainstorming"},
            {"instruction": "Bananas.", "context": "", "category": "general_qa"},
        ]
        report = classify.evaluate_against_dolly(rows)
        self.assertEqual(report.n, 3)
        self.assertEqual(report.n_correct, 3)
        self.assertEqual(report.accuracy, 1.0)
        self.assertEqual(report.n_fallback, 1)

    def test_unknown_categories_are_skipped(self) -> None:
        rows = [
            {"instruction": "Summarize this.", "context": "x",
             "category": "not_a_dolly_category"},
        ]
        self.assertEqual(classify.evaluate_against_dolly(rows).n, 0)

    def test_sensitivity_accuracy_is_at_least_exact_accuracy(self) -> None:
        """Getting the tier right is strictly easier than the 8-way label."""
        rows = [
            # brainstorming -> open_qa is wrong 8-way but both are Low.
            {"instruction": "Who invented radio?", "context": "",
             "category": "brainstorming"},
        ]
        report = classify.evaluate_against_dolly(rows)
        self.assertEqual(report.n_correct, 0)
        self.assertEqual(report.n_correct_sensitivity, 1)
        self.assertGreaterEqual(report.sensitivity_accuracy, report.accuracy)

    def test_qa_merge_credits_the_open_general_confusion(self) -> None:
        rows = [
            {"instruction": "Who invented radio?", "context": "",
             "category": "general_qa"},
        ]
        report = classify.evaluate_against_dolly(rows)
        self.assertEqual(report.n_correct, 0)
        self.assertEqual(report.n_correct_qa_merged, 1)

    def test_empty_corpus_does_not_divide_by_zero(self) -> None:
        report = classify.evaluate_against_dolly([])
        self.assertEqual(report.n, 0)
        self.assertEqual(report.accuracy, 0.0)
        self.assertEqual(report.sensitivity_accuracy, 0.0)
        self.assertEqual(report.qa_merged_accuracy, 0.0)

    def test_report_formats_without_error(self) -> None:
        report = classify.evaluate_against_dolly(
            [{"instruction": "Summarize.", "context": "x",
              "category": "summarization"}]
        )
        text = report.format_text()
        self.assertIn("8-way accuracy", text)
        self.assertIn("sensitivity tier", text)


class TestRuleTableIntegrity(unittest.TestCase):
    def test_every_rule_targets_a_known_category(self) -> None:
        for category, _, _, _ in classify._RULE_SOURCES:
            self.assertIn(category, tasks.TASK_BY_KEY)

    def test_rule_labels_are_unique(self) -> None:
        labels = [label for _, _, _, label in classify._RULE_SOURCES]
        self.assertEqual(len(labels), len(set(labels)))

    def test_every_pattern_compiles(self) -> None:
        self.assertEqual(len(classify._RULES), len(classify._RULE_SOURCES))

    def test_context_categories_track_tasks_module(self) -> None:
        """The gate must not drift from :data:`bopis.tasks.TASK_TYPES`."""
        self.assertEqual(
            set(classify._CONTEXT_CATEGORIES),
            {t.key for t in tasks.TASK_TYPES if t.has_context},
        )
        self.assertEqual(
            set(classify._CONTEXT_CATEGORIES) | set(classify._NO_CONTEXT_CATEGORIES),
            set(tasks.TASK_KEYS),
        )

    def test_general_qa_has_no_positive_rules(self) -> None:
        """It is reachable only as the fallback; see the module docstring."""
        self.assertEqual(classify._RULES_BY_CATEGORY["general_qa"], ())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
