# Chapter 4/5 data export

Run: `20261107T125343Z_real-full`  
Backend: `llama-server`  
Measured (not simulated): **YES**

## Files

The paper-layout CSVs mirror Appendix B of the manuscript; `tidy/` holds
the run's original long-format tables; `supplementary/` holds derived
tables used by Chapter 4/5 text but not printed as appendix instruments.

| File | Rows |
| --- | ---: |
| `B1_performance_comparison.csv` | 3 |
| `B1_performance_per_config.csv` | 60 |
| `B2_configuration_evaluation.csv` | 60 |
| `B3_energy_per_prompt.csv` | 502 |
| `B4_speed_per_prompt.csv` | 502 |
| `B5_quality_per_prompt.csv` | 502 |
| `B6_resource_utilization_per_prompt.csv` | 502 |
| `B7_bopis_calibration_log.csv` | 31 |
| `B8_summary_comparison.csv` | 35 |
| `B9_xstar_selection.csv` | 8 |
| `B10_pareto_front.csv` | 30 |
| `B11_surrogate_reliability.csv` | 20 |
| `tidy/B1_trials.csv` | 3 |
| `tidy/B2_config_eval.csv` | 60 |
| `tidy/B3_per_prompt_energy.csv` | 1500 |
| `tidy/B4_per_prompt_speed.csv` | 1500 |
| `tidy/B5_per_prompt_quality.csv` | 1500 |
| `tidy/B6_per_prompt_resources.csv` | 1500 |
| `tidy/B7_bo_log.csv` | 30 |
| `tidy/B8_summary.csv` | 35 |
| `tidy/B9_selection.csv` | 8 |
| `tidy/B10_pareto.csv` | 30 |
| `tidy/B11_gp_validation.csv` | 20 |
| `supplementary/per_task.csv` | 24 |
| `supplementary/per_trial.csv` | 1 |
| `supplementary/per_variant.csv` | 6 |
| `supplementary/indicators.csv` | 16 |

## Regenerate

```
python -m bopis export --run C:\Aaron\4thyear1stsem\Thesis\bopis\runs\20261107T125343Z_real-full --dest docs/chapter4_data
```

Re-run the same command after the measured run; only the source run changes.
