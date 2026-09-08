"""Task taxonomy and the four-column precision prior that replaces Table T1."""

from __future__ import annotations

import random
import unittest

from bopis import config_space as cs
from bopis import tasks
from bopis.tasks import (
    PRECISION_PRIOR,
    TASK_BY_KEY,
    TASK_KEYS,
    TASK_TYPES,
    Sensitivity,
    dataset_prior,
    mask_and_renormalize,
    prior_for_task,
    sample_seed_configs,
    t1_table_rows,
)


class TestTaxonomy(unittest.TestCase):
    def test_has_eight_categories(self) -> None:
        self.assertEqual(len(TASK_TYPES), 8)
        self.assertEqual(len(TASK_KEYS), 8)

    def test_keys_are_dollys_literal_category_values(self) -> None:
        """Amendment A-12: the keys must be Dolly's own strings.

        Anything else makes the stratification unreplicable, and the manuscript's
        "General Instr." label does not correspond to a Dolly category.
        """
        self.assertEqual(
            set(TASK_KEYS),
            {
                "open_qa",
                "closed_qa",
                "summarization",
                "classification",
                "creative_writing",
                "brainstorming",
                "information_extraction",
                "general_qa",
            },
        )

    def test_general_qa_is_not_labelled_instruction_following(self) -> None:
        """Dolly's general_qa is open-domain QA without a supplied context."""
        self.assertEqual(TASK_BY_KEY["general_qa"].label, "General QA")
        self.assertFalse(TASK_BY_KEY["general_qa"].has_context)

    def test_high_sensitivity_categories(self) -> None:
        high = {t.key for t in TASK_TYPES if t.sensitivity == Sensitivity.HIGH}
        self.assertEqual(high, {"closed_qa", "information_extraction"})

    def test_context_bearing_categories(self) -> None:
        """Exactly the three categories with long prompts in Dolly."""
        with_context = {t.key for t in TASK_TYPES if t.has_context}
        self.assertEqual(
            with_context,
            {"closed_qa", "information_extraction", "summarization"},
        )

    def test_every_key_resolves(self) -> None:
        for key in TASK_KEYS:
            self.assertIn(key, TASK_BY_KEY)
            self.assertIn(key, PRECISION_PRIOR)


class TestPrior(unittest.TestCase):
    def test_every_row_sums_to_one(self) -> None:
        for key in TASK_KEYS:
            self.assertAlmostEqual(
                sum(PRECISION_PRIOR[key].values()), 1.0, places=12, msg=key
            )

    def test_covers_all_four_variants(self) -> None:
        """Amendment A-4: Q4_K_M must not have zero prior mass."""
        for key in TASK_KEYS:
            row = PRECISION_PRIOR[key]
            self.assertEqual(set(row), set(cs.P_VALUES))
            self.assertGreater(
                row["Q4_K_M"],
                0.0,
                f"{key} gives Q4_K_M no prior mass, so it would never be seeded",
            )

    def test_all_probabilities_are_valid(self) -> None:
        for key in TASK_KEYS:
            for variant, probability in PRECISION_PRIOR[key].items():
                self.assertGreaterEqual(probability, 0.0, f"{key}/{variant}")
                self.assertLessEqual(probability, 1.0, f"{key}/{variant}")

    def test_int8_mass_is_preserved_by_the_split(self) -> None:
        """Q8_0 + Q4_K_M must equal the original INT8 column exactly."""
        from bopis.tasks import _T1_ORIGINAL

        for key, (_f32, _f16, int8) in _T1_ORIGINAL.items():
            row = PRECISION_PRIOR[key]
            self.assertAlmostEqual(
                row["Q8_0"] + row["Q4_K_M"], int8, places=12, msg=key
            )

    def test_f32_and_f16_are_unchanged(self) -> None:
        from bopis.tasks import _T1_ORIGINAL

        for key, (f32, f16, _int8) in _T1_ORIGINAL.items():
            self.assertAlmostEqual(PRECISION_PRIOR[key]["F32"], f32, places=12)
            self.assertAlmostEqual(PRECISION_PRIOR[key]["F16"], f16, places=12)

    def test_high_sensitivity_favours_the_safer_quantization(self) -> None:
        """Quality-sensitive tasks keep more mass on Q8_0 than Q4_K_M."""
        for key in ("closed_qa", "information_extraction"):
            row = PRECISION_PRIOR[key]
            self.assertGreater(row["Q8_0"], row["Q4_K_M"], key)

    def test_low_sensitivity_splits_evenly(self) -> None:
        for key in ("open_qa", "summarization", "classification"):
            row = PRECISION_PRIOR[key]
            self.assertAlmostEqual(row["Q8_0"], row["Q4_K_M"], places=12, msg=key)

    def test_prior_for_task_returns_a_copy(self) -> None:
        first = prior_for_task("open_qa")
        first["F32"] = 99.0
        self.assertNotEqual(PRECISION_PRIOR["open_qa"]["F32"], 99.0)

    def test_unknown_task_raises(self) -> None:
        with self.assertRaises(KeyError):
            prior_for_task("not_a_category")

    def test_table_rows_are_auditable(self) -> None:
        rows = t1_table_rows()
        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertAlmostEqual(row["sum"], 1.0, places=6)
            self.assertIn(row["sensitivity"], ("High", "Medium", "Low"))


class TestDatasetPrior(unittest.TestCase):
    def test_uniform_weights_average_the_rows(self) -> None:
        combined = dataset_prior({key: 1.0 for key in TASK_KEYS})
        self.assertAlmostEqual(sum(combined.values()), 1.0, places=12)
        for variant in cs.P_VALUES:
            expected = sum(PRECISION_PRIOR[k][variant] for k in TASK_KEYS) / 8
            self.assertAlmostEqual(combined[variant], expected, places=12)

    def test_single_task_reproduces_its_row(self) -> None:
        combined = dataset_prior({"closed_qa": 1.0})
        for variant, probability in PRECISION_PRIOR["closed_qa"].items():
            self.assertAlmostEqual(combined[variant], probability, places=12)

    def test_unnormalized_weights_are_normalized(self) -> None:
        a = dataset_prior({"open_qa": 1.0, "closed_qa": 1.0})
        b = dataset_prior({"open_qa": 50.0, "closed_qa": 50.0})
        for variant in cs.P_VALUES:
            self.assertAlmostEqual(a[variant], b[variant], places=12)

    def test_weighting_shifts_the_result(self) -> None:
        mostly_closed = dataset_prior({"closed_qa": 9.0, "classification": 1.0})
        mostly_class = dataset_prior({"closed_qa": 1.0, "classification": 9.0})
        # closed_qa is quality-sensitive, so it puts more mass on F32.
        self.assertGreater(mostly_closed["F32"], mostly_class["F32"])
        self.assertLess(mostly_closed["Q4_K_M"], mostly_class["Q4_K_M"])

    def test_rejects_nonpositive_weights(self) -> None:
        with self.assertRaises(ValueError):
            dataset_prior({"open_qa": 0.0})

    def test_rejects_unknown_task(self) -> None:
        with self.assertRaises(KeyError):
            dataset_prior({"nonsense": 1.0})


class TestMasking(unittest.TestCase):
    def test_renormalizes_over_the_permitted_subset(self) -> None:
        prior = dataset_prior({key: 1.0 for key in TASK_KEYS})
        masked = mask_and_renormalize(prior, ["Q8_0", "Q4_K_M"])
        self.assertEqual(set(masked), {"Q8_0", "Q4_K_M"})
        self.assertAlmostEqual(sum(masked.values()), 1.0, places=12)

    def test_preserves_relative_weights(self) -> None:
        prior = {"F32": 0.1, "F16": 0.2, "Q8_0": 0.3, "Q4_K_M": 0.4}
        masked = mask_and_renormalize(prior, ["Q8_0", "Q4_K_M"])
        self.assertAlmostEqual(masked["Q8_0"] / masked["Q4_K_M"], 0.75, places=12)

    def test_falls_back_to_uniform_when_no_mass_remains(self) -> None:
        prior = {"F32": 1.0, "F16": 0.0, "Q8_0": 0.0, "Q4_K_M": 0.0}
        masked = mask_and_renormalize(prior, ["Q8_0", "Q4_K_M"])
        self.assertAlmostEqual(masked["Q8_0"], 0.5, places=12)
        self.assertAlmostEqual(masked["Q4_K_M"], 0.5, places=12)

    def test_rejects_empty_permitted_set(self) -> None:
        with self.assertRaises(ValueError):
            mask_and_renormalize({"F32": 1.0}, [])


class TestSeedSampling(unittest.TestCase):
    def setUp(self) -> None:
        self.space = cs.full_space()
        self.prior = dataset_prior({key: 1.0 for key in TASK_KEYS})

    def test_draws_the_requested_count(self) -> None:
        seeds = sample_seed_configs(self.space, 10, self.prior, random.Random(0))
        self.assertEqual(len(seeds), 10)

    def test_samples_without_replacement(self) -> None:
        seeds = sample_seed_configs(self.space, 30, self.prior, random.Random(1))
        self.assertEqual(len(set(seeds)), 30)

    def test_all_seeds_come_from_the_space(self) -> None:
        restricted = cs.build_space(p_values=["Q8_0", "Q4_K_M"], g_values=[0])
        seeds = sample_seed_configs(restricted, 8, self.prior, random.Random(2))
        allowed = set(restricted)
        for seed in seeds:
            self.assertIn(seed, allowed)

    def test_never_proposes_an_infeasible_variant(self) -> None:
        """Masking must keep seeding inside the hardware-permitted variants."""
        restricted = cs.build_space(p_values=["Q4_K_M"])
        seeds = sample_seed_configs(restricted, 10, self.prior, random.Random(3))
        self.assertTrue(all(cfg.p == "Q4_K_M" for cfg in seeds))

    def test_is_reproducible_for_a_fixed_seed(self) -> None:
        first = sample_seed_configs(self.space, 10, self.prior, random.Random(7))
        second = sample_seed_configs(self.space, 10, self.prior, random.Random(7))
        self.assertEqual(first, second)

    def test_different_seeds_give_different_draws(self) -> None:
        first = sample_seed_configs(self.space, 10, self.prior, random.Random(1))
        second = sample_seed_configs(self.space, 10, self.prior, random.Random(2))
        self.assertNotEqual(first, second)

    def test_prior_biases_the_variant_distribution(self) -> None:
        """A prior concentrated on one variant must dominate the draws."""
        skewed = {"F32": 0.97, "F16": 0.01, "Q8_0": 0.01, "Q4_K_M": 0.01}
        counts = {"F32": 0}
        trials = 40
        for trial in range(trials):
            seeds = sample_seed_configs(
                self.space, 5, skewed, random.Random(trial)
            )
            counts["F32"] += sum(1 for cfg in seeds if cfg.p == "F32")
        # Well above the ~25% a uniform draw would give.
        self.assertGreater(counts["F32"] / (trials * 5), 0.8)

    def test_zero_seeds(self) -> None:
        self.assertEqual(
            sample_seed_configs(self.space, 0, self.prior, random.Random(0)), []
        )

    def test_rejects_more_seeds_than_configurations(self) -> None:
        small = cs.build_space(
            t_values=[128], b_values=[1], p_values=["F32"], g_values=[0], c_values=[2]
        )
        with self.assertRaises(ValueError):
            sample_seed_configs(small, 5, self.prior, random.Random(0))

    def test_exhausting_one_variant_falls_through_to_others(self) -> None:
        """When a variant runs out, remaining draws come from what is left."""
        space = cs.build_space(
            t_values=[128, 256],
            b_values=[1],
            p_values=["F32", "Q4_K_M"],
            g_values=[0],
            c_values=[2],
        )  # 2 configs per variant, 4 total
        seeds = sample_seed_configs(space, 4, self.prior, random.Random(5))
        self.assertEqual(len(set(seeds)), 4)
        self.assertEqual({cfg.p for cfg in seeds}, {"F32", "Q4_K_M"})


if __name__ == "__main__":
    unittest.main()
