# Conversation Summary — BOPIS Thesis Verification

**Date:** 2026-09-13
**Project:** BOPIS — Bayesian Optimization and Pareto-Based Intelligent Configuration
Selection for Energy-Efficient Local LLM Inference

---

## Files involved

| File | Role |
|---|---|
| `THESIS_WRITING2_G2.pdf` | Revised thesis PDF (post-revision, authoritative target) |
| `G2_THESIS_WRITING-1_MANUSCRIPT (1).md` | Old working manuscript |
| `BOPIS_thesis_revised.md` | New, rewritten/reorganized manuscript |

---

## Verification performed

- Extracted PDF text directly from content streams using the PDF's own ToUnicode
  CMaps (Word-exported PDFs store glyph codes, not plain text).
- Compared prose across PDF vs. manuscripts using word-frequency and word-sequence
  (n-gram) similarity.
- Diffed the Statement of the Problem (SOP) and the quantization-variant text.

### Key numbers

- Word-frequency cosine similarity between revised MD and revised PDF: **0.9999**
- ~95.6% of the revised MD's 5-grams appear in the PDF (PDF coverage ~94.2%)

---

## Findings / decisions confirmed

1. **SOP reduced from 5 to 3 questions** (matches the revised PDF verbatim):
   - Q1 — random search baseline performance
   - Q2 — BOPIS-optimized configuration performance (incl. EIR / SRR / QRR improvement)
   - Q3 — Friedman test among unoptimized / random search / BOPIS configurations
   - Old Q1 (unoptimized default performance) and old Q5 (BO efficiency/reliability)
     were removed; sub-items use letters (a.–e.).

2. **Quantization variants reduced from 4 to 3:** only **F16, Q8_0, Q4_K_M**.
   - FP32/F32 removed from the configuration space (FP32 appears only in legitimate
     literature-comparison citations, e.g., Dettmers et al. 2022, Xu et al. 2023).
   - `F32 GGUF` definition entry removed.
   - F32-exclusion sentence present in both the revised PDF and revised MD.

3. **`BOPIS_thesis_revised.md` is fully aligned with the revised PDF.**
   - "GGUP" typo fixed (→ GGUF).
   - Table T1 prior probabilities renormalized (P(FP32) column removed).
   - RQ5 reference in Data Analysis replaced with "Stage 5" pipeline reference.
   - Author list = 3 members (De Guzman, Patacsil, Piastro). Azusano, John Paul — who
     appeared in the old manuscript — is **not** in the revised PDF either.

4. **The old `G2_THESIS_WRITING-1_MANUSCRIPT (1).md` is outdated** relative to the
   revision (still contained FP32 / 4 variants / 5-question SOP).

5. **Note:** `BOPIS_thesis_revised.md` is condensed vs. the old manuscript (~21k vs
   ~37k tokens) — inline base64 appendix images and bulky appendix tables from the old
   file are not carried over. The PDF still contains that content as embedded images.

---

## Open questions for the leader

1. **Exact scope of the BOPIS tool regarding "no training to our model"** —
   confirmation that the model stays untouched (no training/fine-tuning), with BOPIS
   acting purely as a configuration-selection optimizer.
2. Whether to **archive/replace the old `G2_THESIS_WRITING-1_MANUSCRIPT (1).md`**
   (superseded by `BOPIS_thesis_revised.md`).
3. Whether to **re-embed appendix figure images** into `BOPIS_thesis_revised.md`.

---

## Next steps (once the leader answers)

- Confirm tool scope re: model training.
- Decide fate of the old G2 manuscript (archive vs. replace vs. update).
- Decide on appendix figures in the revised markdown.