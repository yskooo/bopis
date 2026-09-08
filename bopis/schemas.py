"""Frozen column schemas for every BOPIS artifact.

Single source of truth. The CSV writers in :mod:`bopis.artifacts`, the table
assemblers, and the dashboard panels all key off this module, so a column can be
renamed in exactly one place and nothing silently drifts out of sync.

Mapping to the manuscript's appendix tables
-------------------------------------------
Table F1 references Tables A.1-A.11 and M1, none of which exist anywhere in the
document (amendment A-11/A-27). The six A-tables map cleanly onto the six
existing B-tables; the three that have no counterpart are added as B.9-B.11:

===============  ==========================================================
Table F1 cites   Actual artifact
===============  ==========================================================
A.7              ``B.7`` calibration log -- one row per BO iteration
A.1-A.6          ``B.1`` trials, ``B.2`` per-configuration, ``B.3`` energy,
                 ``B.4`` speed, ``B.5`` quality, ``B.6`` resources
A.6              ``B.8`` summary comparison (carries the Friedman/Nemenyi rows)
A.10             ``B.9``  x* selection *(new)*
A.11             ``B.10`` Pareto front and hypervolume *(new)*
M1               ``B.11`` surrogate reliability *(new)*
===============  ==========================================================

Two schema corrections
----------------------
* ``B.3`` drops the "CodeCarbon Validation" columns (amendment A-24). CodeCarbon
  is a third-party package, contradicts the standard-library-only constraint, and
  appears nowhere else in Chapter 3. The independent cross-check is instead
  ``energy_crosscheck_j`` from NVML's own energy counter, which is a driver-side
  measurement of the same quantity and a strictly better validation of the
  Riemann sum.
* ``B.2``'s precision column uses F32/F16/Q8_0/Q4_K_M rather than the earlier
  draft's FP32/FP16/INT8, and ``c_cpu_threads`` is numeric (the template's last
  row read "All", which is a GPU-layer value) (amendment A-23).

Standard library only.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

#: The five configuration-vector columns, shared by every table that records a
#: configuration. Produced by :meth:`bopis.config_space.Config.as_row`.
CONFIG_COLUMNS: Tuple[str, ...] = (
    "t_max_gen_tokens",
    "b_batch_size",
    "p_precision",
    "g_gpu_layers",
    "c_cpu_threads",
)

#: The three experimental conditions of the three-way comparison, in order.
CONDITIONS: Tuple[str, ...] = ("unoptimized", "random_search", "bopis")

CONDITION_LABELS: Dict[str, str] = {
    "unoptimized": "Unoptimized Default",
    "random_search": "Random Search",
    "bopis": "BOPIS",
}

#: Dependent variables, with the orientation each one is ranked by. The
#: orientation matters: Friedman mean ranks are only interpretable if rank 1
#: means "best", which is the *smallest* value for energy and memory and the
#: *largest* for speed and quality.
DEPENDENT_VARIABLES: Tuple[Tuple[str, str, bool], ...] = (
    # (column, label, higher_is_better)
    ("energy_j", "Energy -- Total (J)", False),
    ("j_per_token", "Energy -- J/token", False),
    ("tokens_per_s", "Inference Speed (tok/s)", True),
    ("quality_f1", "Output Quality (BERTScore F1)", True),
    ("cpu_percent", "CPU Utilization (%)", False),
    ("gpu_percent", "GPU Utilization (%)", False),
    ("memory_mib", "Memory Usage (MB)", False),
)

DV_COLUMNS: Tuple[str, ...] = tuple(name for name, _label, _hib in DEPENDENT_VARIABLES)
DV_LABELS: Dict[str, str] = {n: label for n, label, _ in DEPENDENT_VARIABLES}
DV_HIGHER_IS_BETTER: Dict[str, bool] = {n: hib for n, _l, hib in DEPENDENT_VARIABLES}


# --------------------------------------------------------------------------- #
# Table schemas
# --------------------------------------------------------------------------- #

#: Table B.1 -- Performance comparison, one row per condition.
#:
#: Reduced from the template's 30 rows to three (amendment A-22). Thirty rows for
#: three conditions over 500 prompts is not a coherent unit of observation, and
#: its accidental resemblance to N is what created the 30-vs-34 ambiguity about
#: the iteration budget in the first place. ``n_prompts`` makes the aggregation
#: explicit.
B1_TRIALS: Tuple[str, ...] = (
    "trial_no",
    "condition",
    "condition_label",
    "n_prompts",
    *CONFIG_COLUMNS,
    "energy_j_mean",
    "energy_j_sd",
    "j_per_token_mean",
    "tokens_per_s_mean",
    "quality_f1_mean",
    "cpu_percent_mean",
    "gpu_percent_mean",
    "memory_mib_mean",
    "energy_method",
    "energy_scope",
)

#: Table B.2 -- Configuration evaluation, one row per evaluated configuration.
B2_CONFIG_EVAL: Tuple[str, ...] = (
    "config_id",
    "method",
    "iteration",
    *CONFIG_COLUMNS,
    "energy_j",
    "tokens_per_s",
    "quality_f1",
    "on_pareto_front",
)

#: Table B.3 -- Energy consumption per prompt.
B3_ENERGY: Tuple[str, ...] = (
    "prompt_index",
    "prompt_id",
    "task_type",
    "condition",
    "energy_j",
    "j_per_token",
    "n_generated_tokens",
    "prefill_energy_j",
    "decode_energy_j",
    "energy_method",
    "energy_scope",
    "energy_crosscheck_j",
    "crosscheck_method",
    "clamped_samples",
    "n_samples",
)

#: Table B.4 -- Inference speed per prompt.
B4_SPEED: Tuple[str, ...] = (
    "prompt_index",
    "prompt_id",
    "task_type",
    "condition",
    "tokens_per_s",
    "decode_tokens_per_s",
    "wall_s",
    "n_generated_tokens",
    "s_min",
    "meets_s_min",
)

#: Table B.5 -- Output quality per prompt (BERTScore F1).
B5_QUALITY: Tuple[str, ...] = (
    "prompt_index",
    "prompt_id",
    "task_type",
    "condition",
    "quality_f1",
    "quality_precision",
    "quality_recall",
    "q_min_task",
    "meets_q_min",
    "truncated",
    "scorer",
    "baseline_rescaled",
)

#: Table B.6 -- Resource utilization per prompt.
B6_RESOURCES: Tuple[str, ...] = (
    "prompt_index",
    "prompt_id",
    "task_type",
    "condition",
    "cpu_percent",
    "gpu_percent",
    "memory_mib",
    "vram_mib",
    "n_threads",
)

#: Table B.7 -- BOPIS calibration log, one row per search iteration.
B7_CALIBRATION: Tuple[str, ...] = (
    "iteration",
    "source",
    "config",
    *CONFIG_COLUMNS,
    "energy_j",
    "tokens_per_s",
    "quality_f1",
    "gp_mu",
    "gp_sigma",
    "gp_predictive_sigma",
    "expected_improvement",
    "best_energy_so_far",
    "delta_energy",
    "on_pareto_front",
)

#: Table B.8 -- Summary comparison; long format, one row per (DV, statistic).
B8_SUMMARY: Tuple[str, ...] = (
    "dependent_variable",
    "dv_label",
    "statistic",
    *CONDITIONS,
)

#: Table B.9 -- x* selection (referenced by Table F1 as "A.10").
B9_SELECTION: Tuple[str, ...] = (
    "rank",
    "config",
    *CONFIG_COLUMNS,
    "energy_j",
    "tokens_per_s",
    "quality_f1",
    "eir_percent",
    "srr_percent",
    "qrr_percent",
    "meets_srr",
    "meets_qrr",
    "on_pareto_front",
    "selected",
    "selection_status",
)

#: Table B.10 -- Pareto front and hypervolume (referenced as "A.11").
B10_PARETO: Tuple[str, ...] = (
    "config",
    "method",
    "iteration",
    *CONFIG_COLUMNS,
    "energy_j",
    "tokens_per_s",
    "quality_f1",
    "on_pareto_front",
    "is_xstar",
)

#: Table B.11 -- Surrogate reliability (referenced as "M1").
B11_SURROGATE: Tuple[str, ...] = (
    "iteration",
    "config",
    "gp_mu",
    "gp_sigma",
    "gp_predictive_sigma",
    "energy_measured_j",
    "abs_error",
    "within_95_ci",
)

#: Dataset sample manifest, so the 500- and 50-prompt sets are reproducible.
DATASET_SAMPLE: Tuple[str, ...] = (
    "prompt_index",
    "prompt_id",
    "task_type",
    "in_proxy_subset",
    "instruction_chars",
    "context_chars",
    "response_chars",
    "estimated_prompt_tokens",
)

#: Every table, by its manuscript identifier.
TABLES: Dict[str, Tuple[str, ...]] = {
    "B.1": B1_TRIALS,
    "B.2": B2_CONFIG_EVAL,
    "B.3": B3_ENERGY,
    "B.4": B4_SPEED,
    "B.5": B5_QUALITY,
    "B.6": B6_RESOURCES,
    "B.7": B7_CALIBRATION,
    "B.8": B8_SUMMARY,
    "B.9": B9_SELECTION,
    "B.10": B10_PARETO,
    "B.11": B11_SURROGATE,
}

#: Filename each table is written to, relative to the run directory.
TABLE_FILES: Dict[str, str] = {
    "B.1": "trials.csv",
    "B.2": "config_eval.csv",
    "B.3": "validation/per_prompt_energy.csv",
    "B.4": "validation/per_prompt_speed.csv",
    "B.5": "validation/per_prompt_quality.csv",
    "B.6": "validation/per_prompt_resources.csv",
    "B.7": "calibration/bo_log.csv",
    "B.8": "summary.csv",
    "B.9": "selection.csv",
    "B.10": "pareto.csv",
    "B.11": "gp_validation.csv",
}

TABLE_TITLES: Dict[str, str] = {
    "B.1": "Pre-Experiment Instrument Template for Performance Comparison",
    "B.2": "Configuration Evaluation",
    "B.3": "Energy Consumption Per Prompt",
    "B.4": "Inference Speed per Prompt",
    "B.5": "Output Quality per Prompt (BERTScore F1)",
    "B.6": "Resource Utilization Per Prompt",
    "B.7": "BOPIS Calibration Log",
    "B.8": "Summary Comparison",
    "B.9": "x* Selection from the Pareto Front",
    "B.10": "Pareto Front and Hypervolume",
    "B.11": "Surrogate Reliability (GP Validation)",
}


def validate_row(table: str, row: Dict[str, object]) -> List[str]:
    """Return the names of columns in *row* that the *table* schema lacks.

    Used by the writers to fail loudly on a typo rather than silently dropping a
    measurement.
    """
    known = set(TABLES[table])
    return sorted(key for key in row if key not in known)
