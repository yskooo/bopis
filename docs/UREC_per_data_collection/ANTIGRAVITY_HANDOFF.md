# Antigravity Handoff — UREC Tables B.9, B.10, B.10A, B.11

Last updated: 2026-11-13  
Working directory: `C:\Aaron\4thyear1stsem\Thesis\bopis`  
Status: Table B.10A and Table B.11 agreed; manuscript copy/paste not yet done.

## 1. What the user needs

The user is preparing the UREC appendix for an experimental paper. The UREC
professor asked for a **not summarized, per-item data collection** version of the
task-classification evidence, similar in spirit to the per-test-input table in
`docs/chapter4_data/Example .png`.

The user does **not** need new computation right now. The immediate need is the
correct table template and numbering for the manuscript appendices.

The user is not asking for every appendix table to be redesigned. Only these
tables are in scope:

- Table B.9 — configuration viability classification
- Table B.10 — multiclass task-category classification matrix
- Table B.10A — per-class TP/FP/FN/TN and metrics
- Table B.11 — per-prompt task-category classification record

The final agreed pair for the B.10/B.11 requirement is **Table B.10A** and
**Table B.11**. All other experimental tables already have answers and must be
left alone.

## 2. Final appendix numbering and titles

Use the Appendix B prefix consistently. The manuscript currently ends at Table
B.8, so the new sequence is:

```text
Table B.9.   Pre-Experiment Instrument Template for Configuration Viability Classification (Confusion Matrix)
Table B.10.  Pre-Experiment Instrument Template for Multiclass Task Category Classification Matrix
Table B.10A. Pre-Experiment Instrument Template for Per-Class Classification and Confusion Matrix Metrics
Table B.11.  Pre-Experiment Instrument Template for Per-Prompt Task Category Classification Record
```

If the existing surrogate-reliability table is also inserted as B.11, then the
per-prompt table must become **Table B.12**. Do not use plain "Table 11" inside
an Appendix B sequence.

### Agreed final form of Table B.10A and Table B.11

**Table B.10A. Pre-Experiment Instrument Template for Per-Class Classification and Confusion Matrix Metrics**

- Learned classifier, held-out Dolly test set, N = 2,253.
- One row per Dolly task category.
- Columns: `Category`, `TP`, `FP`, `FN`, `TN`, `Precision`, `Recall`, `F1`,
  `Support`.
- This is the per-class expansion of Table B.10. Do not replace Table B.10's
  8×8 matrix with this table.

**Table B.11. Pre-Experiment Instrument Template for Per-Prompt Task Category Classification Record**

- Learned classifier only.
- One row per Dolly test prompt: 2,253 rows.
- Columns: `Prompt No.`, `Prompt ID`, `Actual Category`,
  `Predicted Category`, `Correct?`.
- The `Model` column has been removed and must not be added back.
- `Prompt No.` runs from 1 to 2,253.
- If the rule baseline is needed later, use a separate table or a separately
  agreed variant; do not silently re-add a `Model` column to B.11.

## 3. Critical conceptual distinctions

### B.9 is not B.10

- **B.9** is binary configuration-gate viability over the BOPIS search
  configurations. The protected summary is 5 TP / 1 FP / 1 FN / 23 TN over 30
  configurations.
- **B.10** is multiclass task-category classification over 2,253 held-out Dolly
  test prompts, using eight categories.

Do not mix these units.

### Per-iteration belongs to B.9, not B.10

- B.9's unit is a **configuration / search iteration**.
- B.10's per-item unit is a **prompt**.
- The task classifier is not retrained per BOPIS iteration, so an
  "8 categories × TP/TN/FP/FN per iteration" table is not meaningful.

### Per-item for B.10 means per prompt

The UREC professor's "not summarized" requirement is satisfied by **Table B.11**,
a per-prompt record for the learned classifier. It replaces the user's earlier
idea of a per-iteration table.

### TP/FP/FN/TN are per-class counts, not per-row labels

For each of the eight categories:

- TP = actual is that category and predicted as that category
- FP = actual is another category but predicted as that category
- FN = actual is that category but predicted as another category
- TN = actual is another category and predicted as another category

In the 8×8 matrix, the diagonal cells are the TP counts. The full per-class
TP/FP/FN/TN breakdown belongs in **Table B.10A**, not in the per-prompt table.

Do not add TP/TN/FP/FN columns to Table B.11. A single wrong prediction is
simultaneously an FN for the actual class and an FP for the predicted class, so
per-row confusion labels are ambiguous.

## 4. What was created

All new work is in:

```text
docs/UREC_per_data_collection/
```

`docs/chapter4_data/` remains read-only. No protected file was modified.

### Files in the UREC folder

| File | Purpose |
| --- | --- |
| `ANTIGRAVITY_HANDOFF.md` | This handoff |
| `PAPER_TABLES_B9_B10.md` | Paper-ready B.9, B.10, B.10A, optional B.10B, and learned-only B.11 template |
| `README.md` | Explanation of the export and B.9 caveat |
| `generate_urec_export.py` | Reproducible generator for the backing CSVs |
| `validation_report.txt` | Row counts, source hashes, B.10 reproduction check |
| `UREC_Table2A_B10a_task_category_per_test_input.csv` | 2,253-row wide per-prompt backing record; B.11 uses the learned-classifier columns only |
| `UREC_Table2A_task_category_per_test_input.csv` | 4,506-row long per-prompt/per-model record |
| `UREC_Table2B_bopis_per_test_input.csv` | 1,500-row BOPIS per-prompt table |
| `UREC_Table2C_b9_gate_viability_per_iteration.csv` | 30-row B.9 per-iteration raw table |
| `UREC_Table3A_task_category_metrics_by_model.csv` | Model-level metrics |
| `UREC_Table3B_task_category_per_class_metrics.csv` | Per-class TP/FP/FN, precision, recall, F1 |
| `UREC_Table3C_task_category_confusion_matrix_learned.csv` | Learned-classifier 8×8 matrix |
| `UREC_Table3D_bopis_condition_summary.csv` | BOPIS condition summary |

The CSVs are backing evidence only. The paper should use the formatted tables
in `PAPER_TABLES_B9_B10.md`, not the raw CSVs.

### Important edits already made

`generate_urec_export.py` was updated to:

- Build the wide 2,253-row B10a per-prompt CSV.
- Add surrogate diagnostics to the B.9 per-iteration CSV:
  `gp_mu_energy_j`, `gp_sigma_energy_j`, `expected_improvement`, `p_feasible`,
  `best_energy_so_far`, `delta_energy_j`.
- Produce `validation_report.txt`.

`PAPER_TABLES_B9_B10.md` was updated to the final numbering:

- B.9 gate viability
- B.10 8×8 matrix
- B.10A per-class metrics with TP/FP/FN/TN and support
- B.10B optional model-level summary
- B.11 learned-only per-prompt record template; no `Model` column

## 5. Verification status

B.10 was reproduced deterministically from `bopis/classify_trained.py`:

- Split: 10,507 train / 2,251 validation / 2,253 test
- Learned exact accuracy: 1,570 / 2,253 = 0.696849
- Rule baseline: 1,087 / 2,253 = 0.482468
- Confusion-cell mismatches: 0
- Learned macro F1: 0.681639 vs protected 0.681643 (rounding difference)

B.9 caveat:

- The protected B.9 summary is 5 TP / 1 FP / 1 FN / 23 TN over 30 configurations.
- The per-configuration predicted-viability labels were not persisted.
- The 30 search candidates were measured on the 50-prompt proxy, not validated
  on 500 prompts.
- Recomputing both SRR/QRR gates from the B.7 proxy data yields 16 positives, not
  6.
- Therefore the protected B.9 6/24 split cannot be reconstructed from the
  exported run artifacts. Do not invent a per-row predicted viability column.
- The 500-prompt Stage 3 validation covers only the three final conditions, not
  all 30 candidates.

## 6. Table B.11 numbering details

- Final `Prompt No.` is **2,253**.
- Table B.11 has **2,253 rows**, one per Dolly test prompt.
- Table B.11 is **learned-classifier only**.
- The `Model` column is removed. Do not re-add it.
- Do not use the 500-prompt BOPIS validation subset for Table B.11 unless it is
  explicitly labelled as a subset and its metrics are recomputed.
- The 4,506-row long CSV remains backing evidence only; it is not the manuscript
  table.

## 7. What remains for Antigravity

1. Copy the four tables from `PAPER_TABLES_B9_B10.md` into the manuscript
   Appendix B in the agreed order.
2. Ensure B.10's actual/predicted matrix is not altered. It remains the summary.
3. Add B.10A immediately after B.10.
4. Add B.11 at the end of Appendix B, unless B.11 is already reserved for
   surrogate reliability; in that case use B.12.
5. Keep Table B.11 learned-only with no `Model` column.
6. If the manuscript uses Word, preserve the agreed table titles exactly and
   update all in-text cross-references.
7. Do not change any other appendix table.

## 8. Do not do these

- Do not modify anything in `docs/chapter4_data/`; it is strictly read-only.
- Do not delete or rewrite the protected empirical CSVs.
- Do not add a per-iteration 8-category table for B.10.
- Do not add per-prompt TP/TN/FP/FN columns.
- Do not use 500 prompts for B.10/B.11 unless explicitly labelled as a subset.
- Do not renumber B.9/B.10 without checking all manuscript cross-references.
- Do not treat the raw CSVs as the paper tables. The paper tables are the
  formatted tables in `PAPER_TABLES_B9_B10.md`.

## 9. Regeneration command

Only needed if the backing CSVs must be rebuilt:

```powershell
python docs/UREC_per_data_collection/generate_urec_export.py
```

The script writes only to `docs/UREC_per_data_collection/` and exits non-zero if
the reproduced B.10 matrix or metrics disagree with the protected summary.

## 10. Current git state

At the time of this handoff:

- `docs/UREC_per_data_collection/` is untracked.
- `docs/chapter4_data/Example .png` is untracked.
- No protected data file was modified.

## 11. Open questions

- Is Table B.11 part of the manuscript Appendix B, or a separate UREC submission
  attachment?
- Is B.11 already reserved for the surrogate-reliability table? If yes, the
  per-prompt table becomes B.12.

Table B.11 is already agreed as learned-only with no `Model` column; this is not
an open question.
