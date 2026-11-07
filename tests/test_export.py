"""Tests for the Chapter 4/5 appendix export.

The export is entirely a reshape, so the tests pin the two things a reshape can
silently get wrong: which repeat a per-prompt row came from, and whether the
Mean/SD footer is computed from the same rows that were printed.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest

from bopis import artifacts, export, schemas


def _read(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class ExportFixture:
    """A tiny, complete run directory with two repeats and two prompts."""

    def __init__(self, root: str) -> None:
        self.run = artifacts.RunDirectory(root)

    def write(self) -> None:
        run = self.run
        run.write_table(
            "B.1",
            [
                {
                    "trial_no": 1,
                    "condition": "unoptimized",
                    "condition_label": "Unoptimized Default",
                    "n_prompts": 2,
                    "energy_j_mean": 40.0,
                    "energy_j_sd": 2.0,
                    "j_per_token_mean": 0.4,
                    "tokens_per_s_mean": 50.0,
                    "quality_f1_mean": 0.8,
                    "cpu_percent_mean": 10.0,
                    "gpu_percent_mean": 90.0,
                    "memory_mib_mean": 1000.0,
                },
                {
                    "trial_no": 2,
                    "condition": "random_search",
                    "condition_label": "Random Search",
                    "n_prompts": 2,
                    "energy_j_mean": 30.0,
                    "energy_j_sd": 2.0,
                    "j_per_token_mean": 0.3,
                    "tokens_per_s_mean": 60.0,
                    "quality_f1_mean": 0.8,
                    "cpu_percent_mean": 11.0,
                    "gpu_percent_mean": 91.0,
                    "memory_mib_mean": 1100.0,
                },
                {
                    "trial_no": 3,
                    "condition": "bopis",
                    "condition_label": "BOPIS",
                    "n_prompts": 2,
                    "energy_j_mean": 15.0,
                    "energy_j_sd": 1.0,
                    "j_per_token_mean": 0.15,
                    "tokens_per_s_mean": 70.0,
                    "quality_f1_mean": 0.9,
                    "cpu_percent_mean": 12.0,
                    "gpu_percent_mean": 92.0,
                    "memory_mib_mean": 1200.0,
                },
            ],
        )
        run.write_table(
            "B.2",
            [
                {
                    "config_id": "a",
                    "method": "random_search",
                    "iteration": 1,
                    "t_max_gen_tokens": 128,
                    "b_batch_size": 1,
                    "p_precision": "F16",
                    "g_gpu_layers": "0",
                    "c_cpu_threads": 2,
                    "energy_j": 10.0,
                    "tokens_per_s": 10.0,
                    "quality_f1": 0.8,
                    "on_pareto_front": False,
                },
                {
                    "config_id": "b",
                    "method": "bopis",
                    "iteration": 1,
                    "t_max_gen_tokens": 256,
                    "b_batch_size": 2,
                    "p_precision": "Q8_0",
                    "g_gpu_layers": "14",
                    "c_cpu_threads": 4,
                    "energy_j": 5.0,
                    "tokens_per_s": 20.0,
                    "quality_f1": 0.9,
                    "on_pareto_front": True,
                },
            ],
        )

        # condition-major, trial-minor: RS trial1, RS trial2, BOPIS trial1,
        # BOPIS trial2. eval_size is 2 (max prompt_index).
        energy = {
            "random_search": [(10, 20), (30, 40)],
            "bopis": [(5, 10), (15, 20)],
        }
        rows_energy = []
        rows_speed = []
        rows_quality = []
        rows_resources = []
        for condition, repeats in energy.items():
            for repeat in repeats:
                for offset, total in enumerate(repeat, start=1):
                    rows_energy.append(
                        {
                            "prompt_index": offset,
                            "prompt_id": f"p{offset}",
                            "task_type": "open_qa",
                            "condition": condition,
                            "energy_j": float(total),
                            "j_per_token": float(total) / 100.0,
                        }
                    )
                    rows_speed.append(
                        {
                            "prompt_index": offset,
                            "prompt_id": f"p{offset}",
                            "task_type": "open_qa",
                            "condition": condition,
                            "tokens_per_s": 10.0 if condition == "random_search" else 20.0,
                            "s_min": 9.0,
                            "meets_s_min": True,
                        }
                    )
                    rows_quality.append(
                        {
                            "prompt_index": offset,
                            "prompt_id": f"p{offset}",
                            "task_type": "open_qa",
                            "condition": condition,
                            "quality_f1": 0.8 if condition == "random_search" else 0.9,
                            "q_min_task": 0.85,
                            "meets_q_min": condition == "bopis",
                        }
                    )
                    rows_resources.append(
                        {
                            "prompt_index": offset,
                            "prompt_id": f"p{offset}",
                            "task_type": "open_qa",
                            "condition": condition,
                            "cpu_percent": 10.0 if condition == "random_search" else 20.0,
                            "gpu_percent": 90.0 if condition == "random_search" else 80.0,
                            "memory_mib": 1000.0 if condition == "random_search" else 1100.0,
                        }
                    )
        run.write_table("B.3", rows_energy)
        run.write_table("B.4", rows_speed)
        run.write_table("B.5", rows_quality)
        run.write_table("B.6", rows_resources)

        run.write_table(
            "B.7",
            [
                {
                    "iteration": 1,
                    "source": "seed",
                    "config": "a",
                    "energy_j": 1.0,
                    "tokens_per_s": 10.0,
                    "quality_f1": 0.8,
                    "on_pareto_front": False,
                },
                {
                    "iteration": 11,
                    "source": "bo",
                    "config": "b",
                    "energy_j": 2.0,
                    "tokens_per_s": 20.0,
                    "quality_f1": 0.9,
                    "gp_mu": 2.5,
                    "gp_sigma": 0.1,
                    "expected_improvement": 0.3,
                    "on_pareto_front": True,
                },
            ],
        )
        run.write_table(
            "B.8",
            [
                {
                    "dependent_variable": "energy_j",
                    "dv_label": "Energy -- Total (J)",
                    "statistic": "mean",
                    "unoptimized": 40.0,
                    "random_search": 30.0,
                    "bopis": 15.0,
                },
                {
                    "dependent_variable": "energy_j",
                    "dv_label": "Energy -- Total (J)",
                    "statistic": "sd",
                    "unoptimized": 2.0,
                    "random_search": 2.0,
                    "bopis": 1.0,
                },
            ],
        )
        run.write_table(
            "B.9",
            [
                {
                    "rank": 1,
                    "config": "b",
                    "energy_j": 5.0,
                    "tokens_per_s": 20.0,
                    "quality_f1": 0.9,
                    "selected": True,
                    "selection_status": "optimal",
                }
            ],
        )
        run.write_table(
            "B.10",
            [
                {
                    "config": "b",
                    "method": "bopis",
                    "iteration": 1,
                    "energy_j": 5.0,
                    "tokens_per_s": 20.0,
                    "quality_f1": 0.9,
                    "on_pareto_front": True,
                    "is_xstar": True,
                }
            ],
        )
        run.write_table(
            "B.11",
            [
                {
                    "iteration": 1,
                    "config": "b",
                    "gp_mu": 5.0,
                    "gp_sigma": 0.1,
                    "gp_predictive_sigma": 0.2,
                    "energy_measured_j": 5.0,
                    "abs_error": 0.0,
                    "within_95_ci": True,
                }
            ],
        )

        run.write_manifest(
            {
                "bopis_version": "test",
                "backend": {"backend": "sim"},
                "settings": {"repeats": 2},
            }
        )
        run.write_metrics(
            {
                "statistical_tests": {
                    "energy_j": {"wilcoxon": {"p_value": 0.00001}},
                },
                "repeats": {
                    "per_trial": [
                        {
                            "trial": 1,
                            "seed": 0,
                            "ser": 0.5,
                            "bopis": {"k_star": 2, "config": "b"},
                            "random_search": {"k_star": 1, "config": "a"},
                        }
                    ]
                },
                "per_variant": {
                    "Q8_0": {
                        "energy_j": {
                            "n": 2,
                            "mean": 15.0,
                            "sd": 1.0,
                            "min": 10.0,
                            "max": 20.0,
                        }
                    }
                },
                "success_indicators": {"eir_percent": 50.0, "verdict": "substantial"},
                "random_search_indicators": {"eir_percent": 25.0, "verdict": "substantial"},
            }
        )


class TestExport(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_root = os.path.join(self.tmp.name, "run")
        self.out = os.path.join(self.tmp.name, "out")
        ExportFixture(self.run_root).write()

    def test_paper_tables_have_expected_shapes(self) -> None:
        export.export_run(self.run_root, self.out)

        self.assertEqual(len(_read(os.path.join(self.out, "B1_performance_comparison.csv"))), 3)
        self.assertEqual(len(_read(os.path.join(self.out, "B2_configuration_evaluation.csv"))), 2)
        self.assertEqual(len(_read(os.path.join(self.out, "B7_bopis_calibration_log.csv"))), 3)
        self.assertEqual(len(_read(os.path.join(self.out, "B8_summary_comparison.csv"))), 2 + 7)

        b3 = _read(os.path.join(self.out, "B3_energy_per_prompt.csv"))
        self.assertEqual(len(b3), 2 + 2)
        self.assertEqual(b3[0]["Prompt #"], "1")
        self.assertEqual(b3[-2]["Prompt #"], "Mean")
        self.assertEqual(b3[-1]["Prompt #"], "SD")

    def test_pooled_prompt_mean_matches_footer(self) -> None:
        export.export_run(self.run_root, self.out)
        b3 = _read(os.path.join(self.out, "B3_energy_per_prompt.csv"))

        # prompt 1: RS (10 + 30) / 2 = 20; BOPIS (5 + 15) / 2 = 10
        self.assertAlmostEqual(float(b3[0]["Random Search Total J"]), 20.0)
        self.assertAlmostEqual(float(b3[0]["BOPIS Total J"]), 10.0)
        self.assertEqual(b3[0]["Flag (> 15%)"], "Yes")

        mean_row = b3[-2]
        self.assertAlmostEqual(float(mean_row["Random Search Total J"]), 25.0)
        self.assertAlmostEqual(float(mean_row["BOPIS Total J"]), 12.5)

    def test_single_trial_mode_filters_a_repeat(self) -> None:
        export.export_run(self.run_root, self.out, trial=1)
        b3 = _read(os.path.join(self.out, "B3_energy_per_prompt.csv"))
        self.assertEqual(len(b3), 2 + 2)
        self.assertAlmostEqual(float(b3[0]["Random Search Total J"]), 10.0)
        self.assertAlmostEqual(float(b3[0]["BOPIS Total J"]), 5.0)

    def test_tidy_copies_are_verbatim(self) -> None:
        export.export_run(self.run_root, self.out)
        source = os.path.join(self.run_root, "validation", "per_prompt_energy.csv")
        copy = os.path.join(self.out, "tidy", "B3_per_prompt_energy.csv")
        with open(source, "rb") as handle:
            expected = handle.read()
        with open(copy, "rb") as handle:
            self.assertEqual(expected, handle.read())

    def test_manifest_labels_simulated_runs(self) -> None:
        export.export_run(self.run_root, self.out)
        with open(os.path.join(self.out, "EXPORT_MANIFEST.json"), encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertFalse(manifest["is_measured"])
        self.assertEqual(manifest["trial_used"], "pooled (per-prompt mean across repeats)")

    def test_every_schema_table_reaches_the_export(self) -> None:
        export.export_run(self.run_root, self.out)
        for table_id, filename in export.PAPER_FILES.items():
            self.assertTrue(os.path.exists(os.path.join(self.out, filename)), table_id)
        for table_id, filename in export.TIDY_FILES.items():
            self.assertTrue(os.path.exists(os.path.join(self.out, "tidy", filename)), table_id)
        self.assertEqual(set(export.TIDY_FILES), set(schemas.TABLES))


if __name__ == "__main__":
    unittest.main()
