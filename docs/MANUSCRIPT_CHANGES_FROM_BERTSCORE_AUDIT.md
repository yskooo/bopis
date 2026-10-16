# Manuscript changes implied by the BERTScore audit

**Status: not committed, not pushed, not part of the artifact change set.** This is
a working document for rewriting the manuscript text, kept out of git on purpose.

**How it relates to `docs/AMENDMENTS.md`:** that file is organised by *defect and
urgency* (A-1 … A-43) and is the register of everything wrong with Chapter 3 as
written. This file is organised by *manuscript section* and answers a different
question: given that the artifact now behaves as documented, what does the prose
have to say? Where the two overlap this file points at the amendment rather than
restating it.

**Source of every claim here** is `changes-from-aaron.md` (this session) and
`docs/AMENDMENTS.md`. Nothing is inferred.

---

## 0. Blocking: every results number currently in the manuscript is simulated

This outranks every wording change below. Nothing in this file makes the
manuscript correct until a real run exists.

All three runs in `runs/` are `--backend sim`. BERTScore is never imported on that
path; the F1 values come from a closed form of the configuration
(`bopis/backends/simulator.py:260`). The floors are therefore 98% of a synthetic
baseline, and the energy savings are synthetic too.

**Consequence:** every quantitative claim in Chapter 4/5, every table, and every
abstract sentence containing a number is currently a placeholder. This is not a
rounding or wording defect. If the manuscript is submitted in this state the
central claim — that BOPIS reduces inference energy at ≥ 98% quality retention —
has no evidence behind it.

**Action:** complete the runbook in `changes-from-aaron.md` §10, then regenerate
§6 of this file against real output. Do not edit the numbers by hand at any
point; they come from `dashboard_data.js` and `metrics.json`.

---

## 1. Chapter 3 — Statistical Treatment: BERTScore subsection

**Scope limit, and it is deliberate: everything in this section is additive.** The
metric choice and the F1/QRR presentation have already been through panel review
and are settled. Do not restructure them, do not reorder them, and do not reframe
the thesis around this finding. What follows is a limitation sentence, a set of
suggested wordings, and a list of specific sentences to check for overclaiming —
a copyedit, not a rewrite.

### 1a. Absolute F1 is not a quality scale *(new — A-43)*

The metric is length-sensitive in a way that is measurable and, on this dataset,
severe. Measured with the study's own scorer (`roberta-large` layer 17,
baseline-rescaled) against the real Dolly `response` for a `closed_qa` row:

| candidate | rescaled | unrescaled |
|---|---|---|
| correct, terse — `William Shakespeare` | 1.000 | 1.000 |
| correct, reworded — `It was written by William Shakespeare, the English playwright.` | 0.273 | 0.877 |
| wrong author — `Charles Dickens` | 0.511 | 0.918 |

A wrong answer outscores a correct one in both modes. Suggested text:

> BERTScore F1 is length-sensitive. Recall is maximised over the reference tokens
> while precision is averaged over every candidate token, so a verbose answer is
> penalised for tokens the short reference does not contain, and a terse answer
> pays no precision penalty regardless of whether it is correct. On this dataset
> the effect is material: for a `closed_qa` item whose reference is
> `"William Shakespeare"`, a correct but reworded answer scores 0.877 while the
> incorrect `"Charles Dickens"` scores 0.918. **Absolute F1 is therefore not
> interpreted as a quality scale anywhere in this work, and is reported only as a
> within-study relative measure.**

### 1b. One sentence on why the comparative claim is sound *(new)*

No reordering implied. This is a single supporting sentence, and it earns its place
because the length bias in §1a is only a problem for *absolute* readings — the
retention claim is unaffected by construction, and saying so pre-empts the obvious
objection:

> QRR compares two configurations scored on identical prompts against identical
> references, so any systematic length bias in BERTScore applies to both
> conditions and substantially cancels in the ratio. The quality-retention claim
> is therefore a comparative statement and is not affected by the length
> sensitivity described above; the same does not hold for the absolute F1 values,
> which are reported for transparency and are not interpreted as quality
> percentages.

If the manuscript already says something equivalent in the retention-results
paragraph, this is redundant — check before adding.

### 1c. Baseline rescaling *(already A-38 — confirm it is in the text)*

Unrescaled BERTScore occupies a narrow high band, so `QRR ≥ 98%` would be nearly
impossible to fail. Rescaling is enabled. Confirm A-38 is applied in the
Statistical Treatment section and that the rescaled *and* unrescaled absolute
deltas are both reported, as A-38 requires.

### 1d. Report output-length parity — this is what makes 1b a result, not a claim

*(new — `payload["output_length_parity"]`, `changes-from-aaron.md` §6)*

§1b's cancellation argument is only sound if the two conditions generate
comparable lengths. That was an unstated assumption; it is now measured, and it
belongs in the descriptives table. Greedy decoding (`temperature=0.0`) means
quantization, offload and thread count should not change content — but the search
includes **model size** (A-40), and a smaller model answers more briefly. If that
happened, the brevity bias would not cancel and QRR would be partly reporting
brevity as retention.

The artifact now emits, per condition: mean generated tokens, standard deviation,
`length_retention_percent` (optimized as a percentage of unoptimized), and a
**paired Wilcoxon signed-rank test on generated length (unoptimized vs optimized)**.
A non-significant Wilcoxon is direct statistical evidence that length does not
differ between the reference and optimized configurations, which
converts A1b from an argument into a measurement.

**If parity holds** (the expected case), add to the descriptives table:

> Generated-output length did not differ significantly between the reference and
> optimized configurations (Wilcoxon W = [·], p = [·]), with mean generated tokens
> of [·] for the unoptimized configuration and [·] for the optimized configuration
> ([·]% of baseline). Because BERTScore's length bias applies equally to both
> conditions, it cancels in the QRR ratio, and the reported quality retention is
> not an artefact of differential response length.

**If it does not hold,** `balanced: false` and the interpretation string fires —
report it alongside QRR rather than omitting it, and treat QRR as an upper bound
on genuine quality retention. That is a caveat, not a failure, but it cannot be
quietly dropped.

**Fill these from `metrics.json` → `output_length_parity`, never by hand.** Note
that a `--backend sim` run reports `length_retention_percent` exactly `100.0` and
Wilcoxon `p = 1.0` **by construction** — the simulator returns an identical token
count for every configuration. This row is empty until a real run exists.

---

## 2. Chapter 3 — Methods: reproducibility and two corrections

### 2a. Pin the scoring implementation

A thesis that reports BERTScore F1 to three decimals must say exactly what
produced it. Currently unspecified, and the unpinned install actively breaks:

> Quality was scored with `bert-score` 0.3.13 using `roberta-large` (layer 17) with
> baseline rescaling enabled, under `transformers` 4.57.6 and `torch` 2.14.0
> (CPU). The `transformers` version is constrained to the 4.x series: the
> `bert-score` package declares no upper bound, and version 5.x removes a
> tokenizer internal it depends on, causing scoring to fail at first use rather
> than at install time.

That last clause is unusual to include and worth keeping — it is the sort of
detail that reads as care rather than as an apology.

### 2b. Correct the empty-answer justification *(defect, code not yet changed)*

`runner.py` assigns `0.0` to an empty candidate and documents it as "the rescaled
random-pair level." Measured, an empty candidate rescales to `-4.925`. The
*value* is defensible on different grounds; the *reason* is wrong.

Either change the comment in `runner.py` to the retention-neutral argument, or
change the value. **Do not leave the current wording in the manuscript**, because
it asserts something false about the method:

> An empty generation is scored 0.0. This is the retention-neutral point rather
> than a floor: it prevents non-responses from inflating or deflating QRR in
> either direction. The scorer's own rescaled value for an empty candidate is
> -4.925.

### 2c. State that scoring happens outside the measured window

This is a soundness property and belongs in Methods, because it is what stops the
quality metric from contaminating the energy metric. `roberta-large` is a large
CPU forward pass; run inside the measurement window it would inflate both energy
and latency for the configuration under test.

> BERTScore is executed after each batch of generations and after that batch's
> energy measurement window has closed. The scorer's forward pass is therefore
> never included in a reported energy or latency figure. The scorer is constructed
> and warmed up before the run begins, so model initialisation is likewise
> excluded.

Verify the wording matches the implementation before submitting — this is a
description of `bopis/runner.py`, and it should be re-read against the final
code, not trusted from this document.

---

## 3. Chapter 3 — Dataset: define prompt and reference

The prompt/reference distinction caused real confusion during implementation and
is worth one explicit sentence, because BERTScore is meaningless without it:

> Each evaluation item supplies a **prompt** — the concatenation of the Dolly
> `instruction` and `context` fields, which is the question posed to the model —
> and a **reference**, which is the human-written answer from the Dolly `response`
> field. The model generates from the prompt; the reference is used only to score
> that generation. The two are never interchanged: scoring a generation against
> another item's prompt would measure similarity to a question rather than to an
> answer.

The evaluation set is 500 prompts, stratified across the eight task categories,
drawn without replacement with `random.Random(20260101)`; the proxy set is 50
further prompts with one floor per stratum. Confirm these match the text.

---

## 4. Threats to Validity

Three additions, all supported by measurements in `changes-from-aaron.md` §6.

**Length sensitivity of the quality metric.** As in §1a. State that absolute F1
is not a quality scale, and that the comparative design is what rescues the
claim.

**Confound between the metric and the system prompt.** The deployed system
prompt instructs the model to be "concise and helpful," and BERTScore rewards
brevity. Any observed terseness is therefore partly an artefact of the harness
rather than of the selected configuration:

> Because the deployed system prompt requests concise responses, and BERTScore
> F1 rewards shorter answers on short references, a measured change in output
> length cannot be attributed solely to the configuration. This confound affects
> any length-related finding and is not corrected for.

**Unvalidated absolute scale.** The floors are 98% of the unoptimized baseline,
so they are internally consistent but externally unanchored — nothing here
establishes what F1 value constitutes acceptable output quality in absolute
terms, and no such claim should be made.

---

## 5. Livechat / deployment chapter: keep the scope where the spec puts it

`docs/CHATBOT_INTEGRATION.md:252-257` already settles this:

> The UI displays generated tokens, decode speed, and latency when llama.cpp
> returns them. Energy remains unavailable on the MX330, and **BERTScore remains
> an offline Dolly evaluation metric**.

The panel demonstration's specified job is tokens, decode speed and latency.
Quality is not measured in the chat.

**The simple fix is to state it, and the spec already states it.** No new framing
is needed — carry the existing scope sentence into the manuscript text:

> BERTScore F1 was not evaluated during the livechat demonstration. The interface
> reports generated tokens, decode speed and latency only; quality retention is
> assessed offline, over the fixed evaluation set, as described in Section [3].

**One loose end, so the statement matches the artifact.** The UI *does* compute a
single BERTScore F1 when the user loads an unmodified Dolly prompt. It is a
diagnostic for the scoring path, not a reported measurement — but a panelist
opening the demo could see a number where the text says none was measured. Two
one-line options:

- **Caption it** (keeps the working feature): *"A BERTScore value may appear for
  an unmodified Dolly prompt. It is a diagnostic confirming the offline scoring
  path is operational, is not a reported measurement, and is not interpretable on
  its own given the length sensitivity in Section [1a]."*
- **Remove the badge from the chat view** (makes the statement literally true of
  the artifact, at the cost of a working feature). `/api/score` stays available;
  only the chat display changes.

**Do not add a user-supplied reference field.** It was considered and rejected
(`changes-from-aaron.md` §4). It invites precisely the misreading above, and a
hand-typed reference scored against one ad-hoc question is not a measurement.

---

## 6. Numbers that must be regenerated from a real run

Treat every row as a placeholder. Do not carry any of them into a draft.

| quantity | current (simulated) value | source after a real run |
|---|---|---|
| unoptimized mean F1 | 0.8101 | `metrics.json` |
| random-search mean F1 | 0.7960 | `metrics.json` |
| BOPIS mean F1 | 0.7963 | `metrics.json` |
| per-task floors `q_min_by_task` | 0.7935 – 0.7948 | `metrics.json` |
| `s_min` | 63.371471751206734 | `metrics.json` |
| QRR | — (unmeasured) | `metrics.json` |
| SRR | — (unmeasured) | `metrics.json` |
| output-length parity | `100.0%`, Wilcoxon `p = 1.0` — **synthetic, always balanced on sim** | `metrics.json` → `output_length_parity` |
| EIR | 44.84% (previous sim run) | `metrics.json` |
| SER convergence (`k*`) | BOPIS 21 vs random search 2 | `metrics.json` |
| GP reliability (LOO R², NPE) | R² ≈ 0.9 → 0.75, NPE ≈ 36% | metrics + surrogate diagnostics |
| **headline energy saving** | **synthetic** | `metrics.json` |
| SRR / QRR used to select `x*` | from the simulator | `metrics.json` |

The last three rows are the ones a panel will press hardest. A-40 already records
that the energy GP fits the larger 6-D space less well than the smaller one, so
the reliability figures are expected to be reported as measured — do not quote the
pre-A-40 values.

---

## 7. Do not write

- Any statement that BERTScore F1 measures answer *correctness*. It measures
  contextual embedding similarity, and on this dataset a wrong answer can outscore
  a right one.
- Absolute F1 as a quality percentage, or as evidence that output quality is
  "good" or "acceptable."
- A BERTScore figure attributed to the livechat, the panel demonstration, or the
  deployed system.
- Any chat measurement of quality, energy savings, or latency taken from a
  simulated run.
- A quality-retention claim stronger than `QRR ≥ 98%`, which is what the design
  supports.
- A typed-reference F1 in Table B.5 or any results table.

---

## 8. Citations to verify before submission

Flagged, not verified here — confirm each against the published proceedings.

1. **BERTScore itself** — the canonical paper for the metric. Verify volume,
   venue and year against the published proceedings rather than a preprint.
2. **Baseline rescaling** — confirm this is described in the BERTScore paper
   itself. It is treated as a non-standard option in A-38, so the manuscript must
   either cite where it comes from or present it as a modification.
3. **roberta-large layer 17** — confirm this is the documented default for
   `roberta-large`, since the manuscript reports a specific layer.
4. The Gardner et al. and Gelbart et al. citations already flagged in A-40/A-41.

---

## 9. Sign-off

The manuscript is ready to be rewritten against this artifact when all of the
following hold.

- [ ] A real-backend run exists, and `generations.jsonl` carries non-null
      `quality_f1` with `scorer = bert-score/roberta-large@17`
- [ ] `QRR ≥ 98%` and `SRR ≥ 95%` measured, or the shortfall reported honestly
- [ ] §6 of this file regenerated from `metrics.json`, every row
- [ ] `runner.py`'s empty-answer comment corrected in code, not just in prose
- [ ] A-38 and A-43 both reflected in the Statistical Treatment section
- [ ] Output-length parity reported: mean tokens per condition plus the Wilcoxon
      on length, from `output_length_parity` — and `balanced: false` disclosed if
      it ever fires
- [ ] Length sensitivity and the system-prompt confound in Threats to Validity
- [ ] Prompt/reference definitions present in the dataset section
- [ ] Scoring-outside-the-measurement-window stated in Methods
- [ ] Chat demonstration material captioned or stripped of BERTScore badges
- [ ] Citations in §8 verified against published proceedings
- [ ] `bopis_profile.js` matches the machine that produced the reported run
