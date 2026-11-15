# UREC per-data-collection export

This folder contains the row-level data behind the Chapter 4/5 summary tables,
organised in the same "one row per test item per model/condition" style as the
example image supplied for the UREC query.

It is generated from the authoritative run and does **not** modify
`docs/chapter4_data/`, which remains strictly read-only.

- Source run: `runs/20261107T125343Z_real-full`
- Source manifest: `docs/chapter4_data/EXPORT_MANIFEST.json`
- Generator: `docs/UREC_per_data_collection/generate_urec_export.py`
- Validation: `validation_report.txt` (row counts, checksums, B10 reproduction)

## Paper tables and final Appendix B numbering

The formatted manuscript tables are in `PAPER_TABLES_B9_B10.md`:

- `Table B.9` — configuration viability confusion matrix
- `Table B.10` — 8×8 task-category confusion matrix
- `Table B.10A` — per-class TP/FP/FN/TN, precision, recall, F1
- `Table B.11` — per-prompt task-category classification record

`Table B.11` is the unsummarized per-item table for the **learned classifier
only**. Its unit is a **prompt**, not a BOPIS iteration. It has 2,253 rows and the
columns `Prompt No.`, `Prompt ID`, `Actual Category`, `Predicted Category`, and
`Correct?`. The `Model` column has been removed and must not be added back. See
`ANTIGRAVITY_HANDOFF.md` for the full decision record and implementation notes.

## Files and units of analysis

| File | Rows | One row is | Same data as |
| --- | ---: | --- | --- |
| `UREC_Table2A_B10a_task_category_per_test_input.csv` | 2,253 | one Dolly test prompt, with both classifier predictions side by side | Backing evidence for B.10; manuscript B.11 uses the learned-classifier columns only |
| `UREC_Table2A_task_category_per_test_input.csv` | 4,506 | one Dolly test prompt x one classifier (rule baseline or learned Naive Bayes); 2,253 prompts x 2 models | Tables B.10 and 3A/3B/3C |
| `UREC_Table2B_bopis_per_test_input.csv` | 1,500 | one of the 500 final-evaluation prompts x one condition (unoptimized, random search, BOPIS); 500 x 3 | Appendix B.3-B.6 |
| `UREC_Table2C_b9_gate_viability_per_iteration.csv` | 30 | one BOPIS search iteration/configuration | Appendix B.7 and the raw side of B.9 |
| `UREC_Table3A_task_category_metrics_by_model.csv` | 2 | one classifier, with accuracy, QA-merged accuracy, sensitivity-tier accuracy, macro precision/recall/F1 | Table B.10 summary |
| `UREC_Table3B_task_category_per_class_metrics.csv` | 16 | one classifier x one of the 8 Dolly categories, with TP/FP/FN, precision, recall, F1 | Table B.10 expanded |
| `UREC_Table3C_task_category_confusion_matrix_learned.csv` | 8 | one actual category, with the 8 predicted-category counts for the learned classifier | Table B.10 matrix |
| `UREC_Table3D_bopis_condition_summary.csv` | 24 | one condition x one metric, with n, mean, SD, min, max | Appendix B.1/B.8 |

## How this answers "per data collection"

The example image has a per-test-input table (Test Item No., Model, and the
value for that test input) followed by a summary metrics table. These files use
the same structure:

- **Task-category classification:** `UREC_Table2A_B10a_...` is the wide
  per-prompt backing record: 2,253 rows, with `learned_predicted_category` and
  `rule_predicted_category` side by side. The manuscript Table B.11 uses the
  learned-classifier columns only and omits the `Model` column.
  `UREC_Table2A_...` is the same data in long form (4,506 rows); filter
  `model == learned_naive_bayes` for the rows that reproduce the B.10 matrix.
- **BOPIS evaluation:** `UREC_Table2B_...` is the per-test-input table for the
  500 prompts under all three conditions. The `condition` column is the
  analogue of the example's model column.
- **BOPIS search:** `UREC_Table2C_...` is the per-iteration table for all 30
  configurations. It is the raw side of B.9.

## Reproduction check

The learned classifier is re-fitted from `data/databricks-dolly-15k.jsonl`
using the deterministic stratified 70/15/15 split (seed `20260101`). The
reproduction matches the protected B.10 confusion matrix exactly:

- train 10,507 / validation 2,251 / test 2,253
- learned exact accuracy 1,570 / 2,253 = 0.696849
- protected accuracy 0.696849
- macro F1 0.681639 vs protected 0.681643 (0.000004 difference from rounding)
- rule baseline exact accuracy 1,087 / 2,253 = 0.482468
- confusion-cell mismatches: 0

## B.9 caveat

The intended B.9 procedure is configuration-gate viability classification:
compare the optimizer surrogate's predicted viability with the authoritative
empirical SRR/QRR gate outcome. The protected B.9 file reports a
5 TP / 1 FP / 1 FN / 23 TN confusion matrix over the 30 search configurations.

The run does **not** persist the per-configuration *predicted* viability labels
that produced those cells, and it does not contain 500-prompt validation results
for all 30 candidates. The 500-prompt Stage 3 logs cover only the three final
conditions (unoptimized, random search, BOPIS), while the 30 search candidates
were measured on the 50-prompt proxy. Recomputing the two gates from the B.7
proxy measurements yields 16 positives, not 6, so the protected 6/24 split
cannot be reconstructed from the exported run artifacts.

Therefore `UREC_Table2C_...` contains the measured per-iteration data and the
SRR/QRR gate booleans, but it does not invent a `predicted_viable` column. The
protected B.9 summary remains the authoritative record. A fully reproducible
B.9 per-row table would require either the original surrogate-prediction rule
or 500-prompt validation of all 30 candidates.

## Column notes

- `test_item_no` restarts at 1 for each logical collection (prompt set, test
  set, or search trajectory).
- `srr_percent` and `qrr_percent` in B.9 are computed against the run's
  `search_reference` (`S_ref = 38.4603676364 tok/s`,
  `Q_ref = 0.1236795945` BERTScore F1), matching `B9_xstar_selection.csv`.
- `UREC_Table2C_...` now also keeps the surrogate fields behind the intended
  B.9 procedure: `gp_mu_energy_j`, `gp_sigma_energy_j`,
  `expected_improvement`, `p_feasible`, `best_energy_so_far`, and
  `delta_energy_j`. These are diagnostics only; no binary `predicted_viable`
  column is fabricated because the original threshold/rule is not in the run.
- `meets_srr_gate` uses SRR >= 95%; `meets_qrr_gate` uses QRR >= 98%.
- In `UREC_Table2B_...`, `meets_s_min` and `meets_q_min` are the per-prompt
  gate results for each condition; `s_min` and `q_min_task` are the thresholds
  used for that prompt's task.
- `NA` is never used for a measured value here; blanks appear only where the
  source artifact does not persist a field.

## Regenerate

```powershell
python docs/UREC_per_data_collection/generate_urec_export.py
```

The script reads the protected files, writes only to this folder, and exits
non-zero if the reproduced B.10 matrix or metrics disagree with the protected
summary. See `validation_report.txt` for the source/output SHA-256 hashes.
