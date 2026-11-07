# Chapter 4/5 data dictionary

All files are UTF-8 CSV unless noted. `NA` means the artifact does not persist
that field for the row; it is never a zero or a guess. Paper-layout tables use
the exact columns printed in Appendix B. `Mean` and `SD` rows are appended to
the per-prompt sheets (sample SD, n-1).

## Paper-layout tables

| File | Appendix table | Source artifact | Notes |
| --- | --- | --- | --- |
| `B1_performance_comparison.csv` | B.1 | `trials.csv` | One row per condition (amendment A-22 reduces the 30-row template to the three conditions actually observed). |
| `B1_performance_per_config.csv` | B.1 | `config_eval.csv` | One row per evaluated configuration; per-config CPU/GPU/RAM are `NA` because only condition-level resources are persisted. |
| `B2_configuration_evaluation.csv` | B.2 | `config_eval.csv` | Search configuration vector with energy, throughput, and BERTScore F1. |
| `B3_energy_per_prompt.csv` | B.3 | `validation/per_prompt_energy.csv` | Primary trial only. Flag `Yes` when BOPIS J/token is more than 15% below random search. |
| `B4_speed_per_prompt.csv` | B.4 | `validation/per_prompt_speed.csv` | Primary trial. `S >= S_min?` is the BOPIS row's own gate result. |
| `B5_quality_per_prompt.csv` | B.5 | `validation/per_prompt_quality.csv` | Primary trial. `Q_min(task)` is the per-task floor used for QRR. |
| `B6_resource_utilization_per_prompt.csv` | B.6 | `validation/per_prompt_resources.csv` | Primary trial. |
| `B7_bopis_calibration_log.csv` | B.7 | `calibration/bo_log.csv` | One row per search iteration; `C_boTotal` row sums search energy and reports front size. |
| `B8_summary_comparison.csv` | B.8 | `summary.csv` + `metrics.json` | Mean/SD/Min/Max per dependent variable plus the paired Wilcoxon p-value (RS vs BOPIS). |
| `B9_xstar_selection.csv` | B.9 | `selection.csv` | Pareto-front ranking and the selected x*. |
| `B10_pareto_front.csv` | B.10 | `pareto.csv` | Evaluated configurations with Pareto membership. |
| `B11_surrogate_reliability.csv` | B.11 | `gp_validation.csv` | GP prediction vs measured energy per validation point. |

## Tidy copies (`tidy/`)

Verbatim copies of the run's B.1-B.11 files, long format, full precision. Use
these for statistics and plots; use the paper-layout files for the appendix.

## Supplementary (`supplementary/`)

| File | Contents |
| --- | --- |
| `per_trial.csv` | One row per repeated seeded trial: seed, k*, selected config, EIR/SRR/QRR, and SER. Empty when `--repeats 1`. |
| `per_task.csv` | Task-type x condition means over the selected trial(s). |
| `per_variant.csv` | Descriptive statistics per precision variant and dependent variable. |
| `indicators.csv` | Success indicators (EIR/SRR/QRR, gates, verdict) for BOPIS and random search. |

## Reading caveats

1. **Per-prompt tables.** By default the 500 rows are the mean per prompt across
   every repeat, which matches the pooled descriptive statistics the run
   reports. Use `--trial N` to export one repeat instead. Repeat 1 can select
   the same configuration for both methods, so a single-trial sheet may show no
   per-prompt difference even when the pooled means differ.
2. **Simulated runs.** Check `EXPORT_MANIFEST.json` (`is_measured`). A simulator
   run is a pipeline check, not an empirical result.
3. **SER < 1** means BOPIS reached its final pick later than random search in
   that trial; do not claim BOPIS is more sample-efficient without checking
   `per_trial.csv`.
