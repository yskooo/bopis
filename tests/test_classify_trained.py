"""Tests for the supervised task classifier (:mod:`bopis.classify_trained`)."""

from __future__ import annotations

import os
import tempfile
import unittest

from bopis import classify, classify_trained as ct, tasks


def _row(instruction: str, category: str, context: str = ""):
    return {"instruction": instruction, "context": context, "category": category}


#: A tiny separable corpus: each category has a distinctive keyword, repeated
#: enough times to survive MIN_DOCUMENT_FREQUENCY and to fill a 70/15/15 split.
def _toy_rows(per_category: int = 20):
    templates = {
        "summarization": "summarize this document briefly",
        "classification": "classify this item into a bucket",
        "brainstorming": "brainstorm several playful options",
        "open_qa": "who discovered this element",
        "closed_qa": "according to the passage what happened",
        "creative_writing": "write a whimsical poem",
        "information_extraction": "extract every listed date",
        "general_qa": "explain this concept generally",
    }
    rows = []
    for category, text in templates.items():
        has_context = tasks.TASK_BY_KEY[category].has_context
        for i in range(per_category):
            rows.append(
                _row(f"{text} number {i % 3}", category,
                     context="a passage" if has_context else "")
            )
    return rows


class TestTokenize(unittest.TestCase):
    def test_produces_word_first_bigram_and_context_features(self) -> None:
        tokens = ct.tokenize("Summarize this report", "")
        self.assertIn("w:summarize", tokens)
        self.assertIn("w:report", tokens)
        self.assertIn("first:summarize", tokens)
        self.assertIn("bg:summarize_this", tokens)
        self.assertIn("ctx:no", tokens)

    def test_context_flag_flips(self) -> None:
        self.assertIn("ctx:yes", ct.tokenize("What happened?", "A passage."))
        self.assertIn("ctx:no", ct.tokenize("What happened?", "   "))

    def test_context_contributes_no_word_features(self) -> None:
        """Context is 20x longer than the instruction; only its flag is used."""
        tokens = ct.tokenize("Summarize.", "photosynthesis chloroplast thylakoid")
        self.assertNotIn("w:photosynthesis", tokens)
        self.assertNotIn("w:chloroplast", tokens)

    def test_bigrams_are_limited_to_the_first_three_tokens(self) -> None:
        tokens = ct.tokenize("alpha beta gamma delta epsilon", "")
        bigrams = [t for t in tokens if t.startswith("bg:")]
        self.assertEqual(bigrams, ["bg:alpha_beta", "bg:beta_gamma"])

    def test_empty_instruction_still_yields_a_context_flag(self) -> None:
        self.assertEqual(ct.tokenize("", ""), ["ctx:no"])

    def test_is_case_insensitive(self) -> None:
        self.assertEqual(ct.tokenize("SUMMARIZE IT", ""),
                         ct.tokenize("summarize it", ""))


class TestNaiveBayesFit(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            ["w:summarize", "ctx:no"], ["w:summarize", "ctx:no"],
            ["w:classify", "ctx:no"], ["w:classify", "ctx:no"],
        ]
        self.labels = ["summarization", "summarization",
                       "classification", "classification"]

    def test_learns_a_separable_problem(self) -> None:
        model = ct.MultinomialNaiveBayes(alpha=0.1).fit(
            self.documents, self.labels, min_df=1
        )
        self.assertEqual(model.predict(["w:summarize", "ctx:no"]), "summarization")
        self.assertEqual(model.predict(["w:classify", "ctx:no"]), "classification")

    def test_classes_are_sorted_and_complete(self) -> None:
        model = ct.MultinomialNaiveBayes().fit(self.documents, self.labels, min_df=1)
        self.assertEqual(model.classes_, ("classification", "summarization"))

    def test_predict_proba_is_a_distribution(self) -> None:
        model = ct.MultinomialNaiveBayes().fit(self.documents, self.labels, min_df=1)
        posterior = model.predict_proba(["w:summarize", "ctx:no"])
        self.assertAlmostEqual(sum(posterior.values()), 1.0, places=9)
        for value in posterior.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_priors_reflect_label_frequency(self) -> None:
        documents = self.documents + [["w:classify", "ctx:no"]]
        labels = self.labels + ["classification"]
        model = ct.MultinomialNaiveBayes().fit(documents, labels, min_df=1)
        self.assertGreater(
            model.log_prior_["classification"], model.log_prior_["summarization"]
        )

    def test_unknown_features_are_ignored_not_fatal(self) -> None:
        model = ct.MultinomialNaiveBayes().fit(self.documents, self.labels, min_df=1)
        self.assertIn(
            model.predict(["w:neverseen", "w:summarize", "ctx:no"]),
            model.classes_,
        )

    def test_min_df_prunes_rare_features(self) -> None:
        documents = self.documents + [["w:hapax", "ctx:no"]]
        labels = self.labels + ["summarization"]
        model = ct.MultinomialNaiveBayes().fit(documents, labels, min_df=2)
        self.assertNotIn("w:hapax", model.vocabulary_)

    def test_predict_before_fit_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            ct.MultinomialNaiveBayes().predict(["w:anything"])

    def test_empty_training_set_raises(self) -> None:
        with self.assertRaises(ValueError):
            ct.MultinomialNaiveBayes().fit([], [])

    def test_mismatched_lengths_raise(self) -> None:
        with self.assertRaises(ValueError):
            ct.MultinomialNaiveBayes().fit([["w:a"]], ["x", "y"])

    def test_min_df_that_prunes_everything_raises(self) -> None:
        with self.assertRaises(ValueError):
            ct.MultinomialNaiveBayes().fit(
                self.documents, self.labels, min_df=999
            )


class TestPersistence(unittest.TestCase):
    def test_round_trip_preserves_predictions(self) -> None:
        rows = _toy_rows()
        documents, labels = ct._rows_to_xy(rows)
        model = ct.MultinomialNaiveBayes(alpha=0.25).fit(documents, labels)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.json")
            ct.save(model, path)
            self.assertTrue(os.path.exists(path))
            restored = ct.load(path)

        self.assertEqual(restored.alpha, model.alpha)
        self.assertEqual(restored.classes_, model.classes_)
        for tokens in documents[:40]:
            self.assertEqual(restored.predict(tokens), model.predict(tokens))


class TestJsExport(unittest.TestCase):
    """The browser port in bopis.html reads this payload."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.report = ct.train_and_evaluate(_toy_rows(per_category=30))

    def _payload(self, **kwargs):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bopis_model.js")
            ct.write_js(self.model, path, **kwargs)
            text = open(path, encoding="utf-8").read()
        prefix = "window.BOPIS_TASK_MODEL = "
        self.assertTrue(text.startswith(prefix))
        self.assertTrue(text.rstrip().endswith(";"))
        return json.loads(text[len(prefix):].rstrip().rstrip(";"))

    def test_assigns_the_expected_global(self) -> None:
        self._payload()  # assertions live in the helper

    def test_carries_everything_the_scoring_loop_needs(self) -> None:
        payload = self._payload()
        for key in ("classes", "vocabulary", "log_prior", "log_likelihood",
                    "log_unseen", "alpha", "tasks", "priors"):
            self.assertIn(key, payload)

    def test_every_class_has_a_prior_and_an_unseen_mass(self) -> None:
        payload = self._payload()
        for cls in payload["classes"]:
            self.assertIn(cls, payload["log_prior"])
            self.assertIn(cls, payload["log_unseen"])
            self.assertIn(cls, payload["log_likelihood"])

    def test_tasks_and_priors_cover_every_category(self) -> None:
        payload = self._payload()
        self.assertEqual(set(payload["tasks"]), set(tasks.TASK_KEYS))
        self.assertEqual(set(payload["priors"]), set(tasks.TASK_KEYS))
        for prior in payload["priors"].values():
            self.assertAlmostEqual(sum(prior.values()), 1.0, places=4)

    def test_accuracy_block_is_included_when_a_report_is_passed(self) -> None:
        payload = self._payload(report=self.report)
        accuracy = payload["measured_accuracy"]
        self.assertEqual(accuracy["n_test"], self.report.learned_test.n)
        self.assertIn("rule_baseline_exact", accuracy)

    def test_accuracy_block_is_omitted_without_a_report(self) -> None:
        self.assertNotIn("measured_accuracy", self._payload())

    def test_payload_round_trips_back_into_a_working_model(self) -> None:
        """The exported form must be loadable, so JS and Python agree."""
        payload = self._payload()
        restored = ct.MultinomialNaiveBayes.from_dict(payload)
        documents, _ = ct._rows_to_xy(_toy_rows())
        for tokens in documents[:40]:
            self.assertEqual(restored.predict(tokens), self.model.predict(tokens))


class TestStratifiedSplit(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = _toy_rows(per_category=20)

    def test_split_sizes_are_roughly_the_requested_fractions(self) -> None:
        train, validation, test = ct.stratified_split(self.rows)
        self.assertEqual(len(train) + len(validation) + len(test), len(self.rows))
        self.assertAlmostEqual(len(train) / len(self.rows), 0.70, delta=0.02)
        self.assertAlmostEqual(len(test) / len(self.rows), 0.15, delta=0.02)

    def test_every_category_appears_in_every_split(self) -> None:
        train, validation, test = ct.stratified_split(self.rows)
        for split in (train, validation, test):
            self.assertEqual(
                {row["category"] for row in split}, set(tasks.TASK_KEYS)
            )

    def test_splits_are_disjoint(self) -> None:
        train, validation, test = ct.stratified_split(self.rows)
        ids = [{id(r) for r in split} for split in (train, validation, test)]
        self.assertEqual(len(ids[0] & ids[1]), 0)
        self.assertEqual(len(ids[0] & ids[2]), 0)
        self.assertEqual(len(ids[1] & ids[2]), 0)

    def test_same_seed_gives_the_same_split(self) -> None:
        first = ct.stratified_split(self.rows, seed=7)
        second = ct.stratified_split(self.rows, seed=7)
        for a, b in zip(first, second):
            self.assertEqual([r["instruction"] for r in a],
                             [r["instruction"] for r in b])

    def test_different_seed_gives_a_different_split(self) -> None:
        a = ct.stratified_split(self.rows, seed=1)[2]
        b = ct.stratified_split(self.rows, seed=2)[2]
        self.assertNotEqual([r["instruction"] for r in a],
                            [r["instruction"] for r in b])

    def test_rows_with_unknown_categories_are_dropped(self) -> None:
        rows = self.rows + [_row("something", "not_a_category")]
        train, validation, test = ct.stratified_split(rows)
        self.assertEqual(len(train) + len(validation) + len(test), len(self.rows))

    def test_fractions_must_sum_to_one(self) -> None:
        with self.assertRaises(ValueError):
            ct.stratified_split(self.rows, fractions=(0.5, 0.2, 0.2))


class TestScorePredictions(unittest.TestCase):
    def test_all_correct(self) -> None:
        keys = list(tasks.TASK_KEYS)
        scores = ct.score_predictions(keys, keys)
        self.assertEqual(scores.exact_accuracy, 1.0)
        self.assertEqual(scores.qa_merged_accuracy, 1.0)
        self.assertEqual(scores.sensitivity_accuracy, 1.0)

    def test_qa_confusion_counts_as_merged_but_not_exact(self) -> None:
        scores = ct.score_predictions(["general_qa"], ["open_qa"])
        self.assertEqual(scores.exact, 0)
        self.assertEqual(scores.qa_merged, 1)

    def test_same_tier_confusion_counts_for_sensitivity_only(self) -> None:
        # brainstorming and open_qa are both Low sensitivity.
        scores = ct.score_predictions(["brainstorming"], ["open_qa"])
        self.assertEqual(scores.exact, 0)
        self.assertEqual(scores.sensitivity, 1)

    def test_cross_tier_confusion_counts_for_nothing(self) -> None:
        # closed_qa is High, open_qa is Low.
        scores = ct.score_predictions(["closed_qa"], ["open_qa"])
        self.assertEqual(scores.exact, 0)
        self.assertEqual(scores.sensitivity, 0)
        self.assertEqual(scores.qa_merged, 0)

    def test_confusion_matrix_is_populated(self) -> None:
        scores = ct.score_predictions(
            ["closed_qa", "closed_qa"], ["open_qa", "closed_qa"]
        )
        self.assertEqual(scores.confusion["closed_qa"]["open_qa"], 1)
        self.assertEqual(scores.confusion["closed_qa"]["closed_qa"], 1)

    def test_empty_does_not_divide_by_zero(self) -> None:
        scores = ct.score_predictions([], [])
        self.assertEqual(scores.exact_accuracy, 0.0)

    def test_mismatched_lengths_raise(self) -> None:
        with self.assertRaises(ValueError):
            ct.score_predictions(["open_qa"], ["open_qa", "closed_qa"])


class TestTrainAndEvaluate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.report = ct.train_and_evaluate(_toy_rows(per_category=30))

    def test_selects_an_alpha_from_the_grid(self) -> None:
        self.assertIn(self.report.alpha, [float(a) for a in ct.ALPHA_GRID])
        self.assertEqual(len(self.report.alpha_validation), len(ct.ALPHA_GRID))

    def test_learns_the_separable_toy_corpus(self) -> None:
        self.assertGreater(self.report.learned_test.exact_accuracy, 0.9)

    def test_reports_a_rule_baseline_on_the_same_split(self) -> None:
        self.assertEqual(self.report.rule_test.n, self.report.learned_test.n)

    def test_split_counts_are_recorded(self) -> None:
        total = (self.report.n_train + self.report.n_validation
                 + self.report.n_test)
        self.assertEqual(total, 8 * 30)

    def test_report_formats_without_error(self) -> None:
        text = self.report.format_text()
        self.assertIn("HELD-OUT TEST SET", text)
        self.assertIn("alpha", text)
        self.assertIn("NOTE ON FAIRNESS", text)

    def test_empty_corpus_raises(self) -> None:
        with self.assertRaises(ValueError):
            ct.train_and_evaluate([])


class TestPredictPromptInterface(unittest.TestCase):
    """The learned classifier must be drop-in for the rule classifier."""

    @classmethod
    def setUpClass(cls) -> None:
        documents, labels = ct._rows_to_xy(_toy_rows())
        cls.model = ct.MultinomialNaiveBayes(alpha=0.25).fit(documents, labels)

    def test_returns_the_same_type_the_rules_return(self) -> None:
        prediction = ct.predict_prompt(self.model, "summarize this document")
        self.assertIsInstance(prediction, classify.TaskPrediction)

    def test_scores_cover_every_category_and_normalize(self) -> None:
        prediction = ct.predict_prompt(self.model, "classify this item")
        self.assertEqual(set(prediction.scores), set(tasks.TASK_KEYS))
        self.assertAlmostEqual(sum(prediction.scores.values()), 1.0, places=6)

    def test_confidence_matches_the_top_score(self) -> None:
        prediction = ct.predict_prompt(self.model, "brainstorm several options")
        self.assertAlmostEqual(
            prediction.confidence, max(prediction.scores.values()), places=9
        )

    def test_has_context_is_reported(self) -> None:
        with_context = ct.predict_prompt(self.model, "what happened", "a passage")
        without = ct.predict_prompt(self.model, "what happened", "")
        self.assertTrue(with_context.has_context)
        self.assertFalse(without.has_context)

    def test_task_property_resolves(self) -> None:
        prediction = ct.predict_prompt(self.model, "summarize this document")
        self.assertIn(prediction.task, tasks.TASK_TYPES)

    def test_works_with_prior_for_prompt_contract(self) -> None:
        """The prior lookup must accept the learned prediction's task key."""
        prediction = ct.predict_prompt(self.model, "summarize this document")
        prior = tasks.prior_for_task(prediction.task_key)
        self.assertAlmostEqual(sum(prior.values()), 1.0, places=9)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
