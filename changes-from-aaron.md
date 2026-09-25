# changes-from-aaron

Working notes for this session, so the changes can be reviewed, reverted, or
carried into the manuscript. Committed on `feature/bertscore`.

The manuscript-facing companion, `docs/MANUSCRIPT_CHANGES_FROM_BERTSCORE_AUDIT.md`,
is deliberately **not** committed — it is a drafting aid, not part of the artifact.

Baseline `6ce7f01`. Tests: **577 passed** (was 563).

> **Here to run the study? Go to §10.** It is the ordered runbook: environment,
> acceptance gates at every step, the one check that separates a real run from a
> simulated one, and a troubleshooting table of the exact errors hit in this
> session. Everything below is the reasoning behind it.

---

## 1. The most important finding: no study run has used BERTScore

Three different things get called "BERTScore F1". Only two are real.

| | real? |
|---|---|
| Chat `POST /api/score` | **yes** — roberta-large, CPU, baseline-rescaled. Self-score **1.0000** |
| Study `runner.py:235-236` | real in code, **never executed on this machine** |
| **All three runs** | **simulated** — see below |

Every run in `runs/` is `--backend sim`:

| run | backend |
|---|---|
| `20260925T181332Z` | sim |
| `20260925T190232Z` | sim |
| `20260925T200229Z` | sim |

Their F1 comes from `bopis/backends/simulator.py:260`, `quality_f1(config,
prompt)` — a closed form of the configuration. BERTScore is never imported on
that path. So the numbers currently on screen (unoptimized 0.8101, random search
0.7960, BOPIS 0.7963, floors 0.7935–0.7948) are synthetic.

**Correction.** §3 originally reported the thresholds "verified end-to-end in a
real DOM". Accurate about the **plumbing**; not about the **values**. `Q_min` and
`S_min` travel `runner.py → dashboard.py → bopis.html` correctly, but what they
carry is synthetic until a real-backend run happens. The hardcoded `0.82` was
fiction; its replacement is not yet a measurement either.

The tool is not broken. The evidence is. The models and the energy sensor are
being handled separately by the groupmate; what is left to settle here is BERTScore.

### `bopis/quality/` is dataset-agnostic — the Dolly coupling is all in the callers

Worth recording, because it is easy to assume the scorer is tied to Dolly. It is
not:

| file | Dolly refs | role |
|---|---:|---|
| `bopis/quality/` (`bertscore.py`, `__init__.py`) | **0** | `score(candidates, references)` — two lists of strings |
| `runner.py` | 1 | passes `p.response` as the reference (`runner.py:236`) |
| `dataset.py` | 15 | builds the samples |
| `ui_server.py` | 22 | `/api/dolly` *supplies* references to the UI |
| `classify.py`, `classify_trained.py` | 38 | task classification, unrelated to scoring |

And the endpoint takes **any** reference string — `ui_server.py:648-655` requires
only two non-empty strings, with no prompt-id lookup or dataset validation:

```python
if url.path == "/api/score":
    candidate = str(request.get("candidate") or "")
    reference = str(request.get("reference") or "")
    if not candidate.strip() or not reference.strip():
        return self._json(400, {"error": "candidate and reference are both required"})
```

So the chat was never limited to Dolly 15k. Dolly is only the reference *source*
wired to a button. Any answer can be scored against any other answer.

### Prompt vs reference — the distinction that caused the confusion

A Dolly row has `instruction`, `context`, `response`, `category`. The **prompt**
is the question (`instruction` + `context`); the **reference** is the human-written
answer (`response`). `dataset.py:82-90` keeps them as separate `Prompt` fields.

```
question ──► model ──► model's answer ─┐
                                       ├─► BERTScore F1
Dolly's `response` (human answer) ─────┘
```

Another Dolly row's *prompt* cannot serve as a reference, because that would score
an answer against a question. This is also why the earlier "borrow the nearest
Dolly neighbour" idea failed: *"Who wrote Hamlet?"* retrieves *"Who wrote
Macbeth?"*, whose `response` is `"William Shakespeare"` — a real answer to a
different question. Valid-looking, meaningless.

---

## 2. BERTScore could not run at all: torch was missing

`POST /api/score` returned **HTTP 500** `ModuleNotFoundError: No module named
'torch'`. Installed:

```
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers bert-score
```

`torch 2.14.0+cpu`, `transformers 5.17.0`, `bert-score 0.3.13`. The two commands
must be separate — a single `--index-url .../whl/cpu` replaces PyPI entirely and
the other two come back `No matching distribution found`.

First score took 111 s (the roberta-large download). Cached now, 1,356 MB, no
further downloads needed.

Three defects fixed on the way:

- **The helpful error was dead code.** `quality/__init__.py` wrapped the scorer's
  dependency import to raise an actionable `RuntimeError`, and `ui_server.py`
  caught it to return a 503. But `bertscore.py` imports torch *lazily* inside
  `_ensure_loaded()`, at first scoring — so `get_scorer` succeeded and the
  `ModuleNotFoundError` escaped later from `score()`, past the wrapper, into the
  generic handler.
- **`scorer_error` was declared and never assigned** (`ui_server.py:159`).
- **`/api/status` reported `loaded: true` while scoring was impossible**, because
  `BertScoreScorer.__init__` needs no torch.

| file | change |
|---|---|
| `bopis/quality/__init__.py` | `SCORER_REQUIREMENTS`, `missing_dependencies()` via `find_spec` (reports readiness without importing torch); `get_scorer` fails at construction; `__all__` extended |
| `bopis/quality/bertscore.py` | late `import torch` wrapped, same `RuntimeError` |
| `bopis/ui_server.py` | `_get_scorer` assigns `_scorer` only on success and caches `scorer_error`; `status()` uses `missing_dependencies`, reports `loaded = _scorer is not None and scorer_error is None`, adds `quality.error` |

Stdlib-only core intact: `test_exemption_is_a_single_documented_file` still
passes, so the exemption is still exactly one module.

---

## 3. Calibration thresholds were invented

`bopis.html`'s Calibration Thresholds card hardcoded six values:

| Row | Was | Actually |
|---|---|---|
| Q_min (Closed QA / extraction) | `0.82 F1` | 0.7937 / 0.7943 |
| Q_min (Summarization / Open QA) | `0.78 F1` | 0.7936 / 0.7938 / 0.7940 |
| Q_min (Classification) | `0.78 F1` | 0.7944 |
| Q_min (Creative / Brainstorming) | `0.72 F1` | 0.7935 / 0.7948 |
| S_min (Minimum speed floor) | `derived from p10` | 63.37 tok/s |
| BO iterations completed | `34` | 30 |

`Q_min(task)` is 98% of that task's unoptimized mean F1 (`metrics.q_min`,
`metrics.py:177`), computed per run at `runner.py:737`. **The `0.82` was higher
than any real floor**, so every Closed QA / extraction prompt would have been
reported as failing quality retention. `0.72` was too lenient.

Broke in two places:

- `runner.py:849-850` computed `s_min` and `q_min_by_task` into the **manifest**,
  but `dashboard.py` never propagated them into the payload. The UI had nothing
  to read.
- **The guard that should have caught this only reads
  `dashboard/index.html`** — `TestDashboardHasNoHardcodedMetrics.setUp` hardcodes
  that path, so nothing inspected the integrated UI. That is the root cause of
  the escape.

Edits: `dashboard.py` adds both keys to `thresholds`; `bopis.html` replaces the
literals with `data-qmin` / `data-smin` / `data-iterations` placeholders defaulting
to `&mdash;` and gains `renderThresholds()` (grouped rows render as a range, e.g.
`0.794–0.795 F1`); `tests/test_e2e_sim.py` gains
`test_dashboard_carries_the_quality_floors`; `tests/test_provenance.py` gains
`TestIntegratedUiHasNoHardcodedThresholds`, which slices the card out of
`bopis.html` and fails on any literal `\d+\.\d+ F1`.

A bug in my own first draft: the note condition was
`Number.isFinite(byTask && Object.keys(byTask).length)`, which is
`Number.isFinite(0)` → `true` on an empty object, so the note claimed the values
came from the run while showing five dashes. Now `Object.keys(byTask).length`.

---

## 4. Live chat: a dropped reference was blamed on free chat

BERTScore is reference-based, so free-typed chat legitimately has no F1. A loaded
Dolly prompt the user then **edited** is a different case — the reference existed
and was discarded because it no longer answers the edited text. Both rendered the
same badge.

| State | Before | After |
|---|---|---|
| No Dolly loaded | "free chat has no reference" | unchanged (correct) |
| Dolly unmodified | live F1 | unchanged |
| Dolly edited | "free chat has no reference" | "prompt edited, so the reference for `dolly-10586` no longer answers it. Reload it to score." |

`doChat()` now keeps `loaded` and derives
`const edited = (loaded && !reference) ? loaded : null;`. The exact-match gate
`loaded.text.trim() === txt` is **unchanged** — it is the only thing preventing a
stale reference from being scored against a question it does not answer, and
`TestIntegratedUiExplainsAMissingReference` asserts it survives. The dropped case
shows the original prompt text but **not** the reference answer. New
`.q-dropped` style, amber.

Verified in jsdom across four states:

```
no Dolly loaded          BERTScore n/a (free chat has no reference — use Dolly prompt)
Dolly unmodified         BERTScore F1 0.987 (P 0.990 / R 0.980, rescaled)
Dolly + trailing space   BERTScore F1 0.987     <- doChat trims, whitespace still scores
Dolly edited             BERTScore n/a — prompt edited, so the reference for
                         dolly-10586 no longer answers it. Reload it to score.
```

Two proposals examined and rejected, recorded so they are not relitigated:

- **Borrow the nearest Dolly reference.** Invalid. *"Who wrote Hamlet?"* retrieves
  *"Who wrote Macbeth?"* (reference `"William Shakespeare"`), so a correct reply
  scores ≈1.0 unchecked, while an off-topic question craters to ≈0.6 and reads as
  "the optimized configuration destroyed quality". F1 would measure retrieval
  distance. Also impossible today: `MultinomialNaiveBayes`
  (`classify_trained.py:119-216`) is bag-of-words log-likelihood — no TF-IDF, no
  cosine, no neighbour ranking.
- **Self-consistency F1** (sample k completions, mean pairwise BERTScore among
  them). Storage-free, and it does produce a number, but it conflicts with the
  measurement core: it needs k extra generations per turn, so the chat would
  report inflated latency and energy for the deployed configuration — the exact
  numbers the UI exists to measure. It also measures confidence, not correctness;
  its baseline-rescaling baseline is wrong for same-model pairs; and QRR's 98%
  floor was calibrated on the reference-based distribution, so it would need its
  own floor, validation and tables. Scope creep, correctly avoided.

### What the manuscript already says about this

`docs/CHATBOT_INTEGRATION.md:252-257` settles the scope question, so nothing here
needs inventing:

> The terminal and `curl.exe` commands are backend diagnostics; they are not the
> normal place for the panel demonstration. The UI displays generated tokens,
> decode speed, and latency when llama.cpp returns them. Energy remains
> unavailable on the MX330, and **BERTScore remains an offline Dolly evaluation
> metric**.

So the chat's specified job in the panel demo is tokens, decode speed and latency.
BERTScore is explicitly *excluded* from it and assigned to the offline evaluation.
A free-typed turn showing `n/a` is therefore the **specified behaviour, not a
defect** — which §4's fix makes explicit rather than leaving it looking broken.

Writing the reference yourself is *annotating*, not training: nothing is trained,
it just supplies the second string BERTScore needs. It splits cleanly:

- **As a UI affordance** — an optional field, passed to the existing
  `/api/score`, no methodology change, no new metric, no extra generations, no
  effect on any measurement. Not scope creep.
- **As thesis evidence** — scope creep, and self-contradicting. The methodology
  measures quality on a fixed stratified 500-prompt validation set against Dolly's
  provided references (seed 20260101). Ad-hoc human references on self-invented
  questions would be a second, unvalidated protocol: no held-out set, no floor,
  no statistics, nothing comparable. It cannot go in Table B.5.

If it is added for the demo, two guards matter: the reference must be written
**before** sending (a reference authored after reading the reply is contaminated),
and its F1 must not be pooled into `SESSION.f1` alongside Dolly F1, since the two
are different measurement protocols.

---

## 5. BERTScore was completely broken: `bert-score` is incompatible with `transformers` 5.x

**Found while trying to answer a conceptual question, and more urgent than
anything else in this file.** `bert-score 0.3.13` declares `transformers` with no
upper bound, so pip installed `transformers 5.17.0`, which removed tokenizer
internals the package depends on:

```
AttributeError: RobertaTokenizer has no attribute build_inputs_with_special_tokens
```

Every `/api/score` request would have returned HTTP 500. Fix:

```
python -m pip install "transformers<5"     # -> 4.57.6
```

**This must be pinned.** `bert-score` is last published against the 4.x line, so
any environment that resolves `transformers` unpinned will break it. Worth
recording in the install instructions next to the CPU-torch command, because the
failure appears as an `AttributeError` from inside a third-party package rather
than as a version conflict.

### The module's own fallback did not catch it

`bertscore.py` documents two backends, preferring `bert_score` and falling back to
a direct `transformers` implementation. The fallback is guarded by
`except ImportError` **around construction** (`bertscore.py:122`). But
`BERTScorer(...)` *succeeds* under transformers 5.x; the incompatibility only
surfaces on the first `score()` call, long after `_backend` was set to
`"bert_score"`. So the fallback never engaged and the error escaped as a 500.

This is the same shape of bug as §2: the failure surfaces later than the guard
that is supposed to catch it. **Not fixed** — the environment pin resolves it, and
a real fix needs a decision about whether a runtime failure should fall back to
the *unrescaled* direct path (which would silently change every number, since
amendment A-38 depends on rescaling) or raise. Given the comparability
requirement, raising with a clear message is the right answer; silently dropping
to unrescaled is not.

---

## 6. BERTScore F1 does not separate correct from incorrect on short references

Measured with the real scorer, reference `"William Shakespeare"` (the actual
Dolly `response` for a `closed_qa` row):

| candidate | rescaled (A-38) | unrescaled (raw) |
|---|---|---|
| correct, terse — `William Shakespeare` | **1.000** | **1.000** |
| correct, reworded — `It was written by William Shakespeare, the English playwright.` | **0.273** | **0.877** |
| wrong author — `Charles Dickens` | **0.511** | **0.918** |
| wrong, hedged — `I believe it may have been Christopher Marlowe…` | -0.117 | — |
| empty | -4.925 | — |

**A wrong answer outscores a correct one, in both modes.** This is not a rescaling
artefact — it persists unrescaled, so it is inherent to BERTScore when the
reference is two tokens long. Recall is maximised over the two reference tokens
and precision is averaged over every candidate token, so the function words in a
verbose correct answer ("It was written by the English playwright") are scored as
mismatches, while a terse wrong answer matches nothing but is equally short, so it
pays no precision penalty. Brevity is rewarded; length is punished.

Consequences worth stating plainly:

- **Absolute F1 is not "quality."** The docstring's own concern (line 22-24) is
  confirmed empirically — unrelated proper nouns reach 0.918 raw. The F1 values in
  the payload (0.7963 optimised, 0.8101 unoptimized) must be read as
  *within-study relative* numbers, not as a quality scale, and the manuscript
  should say so if it does not already.
- **QRR is largely protected, and this is why it is the right primary metric.**
  It is a ratio of means between two conditions measured on the *same* prompts
  against the *same* references, so a systematic length/brevity bias applies to
  both sides and substantially cancels. The comparative claim survives even though
  the absolute numbers are soft.
- **Length bias interacts with the system prompt.** `bopis.html` instructs "You
  are BOPIS, a concise and helpful local assistant", and
  `CHATBOT_INTEGRATION.md` asks for a concise assistant. The metric rewards that
  instruction, so a terseness finding would be partly an artefact of the harness
  rather than of the configuration. Worth a sentence in the threats-to-validity
  section.
- **It makes a hand-typed demo reference risky.** Type `William Shakespeare`,
  the model answers correctly and fluently, and the badge reads **0.27** — which
  looks like a failure. Any typed reference should be roughly the length and
  shape of the answer expected, and the demo should use a closed-QA question
  where that is natural.

### Secondary: the empty-answer justification is wrong

`runner.py:226-231` scores an empty candidate as `0.0` and justifies it as "the
rescaled random-pair level". Measured, an empty candidate rescales to
**-4.925**, so 0.0 is far more lenient than the scorer's own value. The choice may
still be defensible — 0.0 is the retention-neutral point, so empties neither help
nor hurt QRR — but the stated reason is factually incorrect and should be
corrected or the value revisited.

---

### The one confound that survives, and the check that closes it

§6's bias cancels in QRR *only if* both conditions generate comparable lengths.
QRR is a ratio over identical prompts against identical references, so a bias
applying to both sides cancels — that part is safe. The precondition is length
parity, and it is not automatic.

Decoding is greedy (`temperature=0.0`, `README.md:200`), so quantization, GPU
offload and thread count should not change content. **Model size can**, and the
search includes it (A-40 adds model size and drops F32). So if the optimizer
picks a smaller model, it answers more briefly, the brevity bias stops cancelling,
and QRR would be partly reporting brevity as quality retention. That is the one
objection to the quality claim that survives everything above.

**This was an unstated assumption in the methodology.** It is now computed.

`n_generated_tokens` was already recorded on every measurement
(`measure.py:79`, serialized at `:154`), so nothing new needed recording — the
data was being logged and never aggregated. `metrics.output_length_parity()` and
`payload["output_length_parity"]` now report, per condition:

| field | meaning |
|---|---|
| `tokens_by_condition` | mean generated tokens |
| `sd_by_condition`, `n_by_condition` | spread and prompt count |
| `length_retention_percent` | optimized mean as % of unoptimized |
| `deviation_percent` | absolute departure from parity |
| `friedman` | Friedman across conditions on length — a **non-significant** result is direct evidence length does not differ |
| `balanced` | parity holds within `LENGTH_PARITY_TOLERANCE_PERCENT` (5%) |
| `interpretation` | the sentence to paste, either way |

The Friedman is the useful part: it upgrades "trust me, the bias cancels" to a
tested claim. If length does not differ significantly across conditions, the
cancellation is demonstrated rather than argued. Eight unit tests in
`tests/test_metrics.py::TestOutputLengthParity`, including the boundary case and
the short-optimized case that must report `balanced: false`.

**Caveat, and it matters:** the simulator returns an identical token count for
every configuration, so a `--backend sim` run reports `length_retention_percent`
exactly `100.0`, Friedman `chi_square 0.0`, `p_value 1.0`, `balanced: true`
**every time, by construction.** Verified on `runs/20260925T221100Z`. This block
is therefore plumbing-verified only, exactly like §1's other numbers. It becomes
evidence on a real-backend run and not before. Do not quote it from a sim run.

---

## 7. Also fixed: the Dolly pool was empty

`/api/dolly` returned 404, `dolly.n = 0`. Cause: `--data-dir` defaults to the
**relative** path `data` (`cli.py:1447`), so it only resolves when launched from
the repo root. The data was there — `data/databricks-dolly-15k.jsonl`, 15,011
rows → **14,825 usable prompts**.

```
python -m bopis ui --port 8090 --llama-url http://127.0.0.1:8080 \
  --idle-seconds 30 --data-dir "C:\Aaron\4thyear1stsem\Thesis\bopis\data"
```

`bopis_profile.js` was also regenerated — the committed copy described a
different machine (Intel i5-1135G7, MX330, `energy_method: unavailable`).

---

## 8. BERTScore verification checklist

Energy and the model ladder are covered by the groupmate, so this is only the
BERTScore part.

**1. The scorer in isolation, with no model and no dataset.** Confirms torch,
transformers, bert-score and the roberta-large cache in one step. Expect ≈1.0:

```
python -c "from bopis.quality import get_scorer; print(get_scorer('bertscore').score(['a cat sat on the mat'],['a cat sat on the mat'])[0].f1)"
```

**2. Confirm the chat path end to end.** Start `llama-server`, then
`python -m bopis ui --data-dir <absolute path>`, open `bopis.html`, click
**Dolly prompt**, send. The badge must reach a real F1 (expect 1.000 when the
model answers correctly). If the badge says n/a, the reference was dropped — see
§4. Check `/api/status` reports `quality.loaded: true` with no `error`.

**3. A smoke run before committing hours.** This is the one that actually answers
"does the *study* path produce real F1", and it yields the per-prompt wall time
needed to estimate a full run:

```
python -m bopis run --backend llama-server --llama-binary <path> \
  --models-dir <dir> --iterations 3 --seeds 1 --eval-size 20 --proxy-size 10
```

Then open `runs/<newest>/raw/generations.jsonl` and confirm the `quality_f1` values
are plausible and varied. **If they are absent, the run was still simulated** —
that is the single check that separates §1's finding from a real measurement. The
`scorer` field on each row records the backend, so `bert-score/roberta-large@17`
confirms the published path rather than the `bertscore-direct` fallback.

---

## 9. Found, not fixed

- **`bopis/cli.py:144` — inverted condition suppresses the profile report.**
  `if not args.json: return 0` sits directly after the "Wrote live hardware
  profile" print, so the README's own `--write-js` demo (`README.md:251`) prints
  one line and no report, while the same command without `--write-js` prints the
  full output. One-line fix.
- **`cmd_report` never rebuilds the payload.** `README.md:18` and `cli.py:17` say
  it does. It only prints a table inventory. This is why §3 needed a full sim run
  to verify. A real fix means reconstructing a `StudyResult` from artifacts.
- **`HW-P0` keys off *available* RAM**, so `|X_feasible|` moved between 256 and
  344 on this machine minutes apart. It gates which configurations the optimizer
  may consider, so the space is not reproducible from artifacts alone unless
  available RAM at profile time is read from `host_profile.ram_available_bytes`.
- **`SER 0.10`** on the newest sim run (`k*_BOPIS=21` vs `k*_RS=2`): BOPIS took 21
  prompts to match what random search found in 2, on that seed. EIR was 44.84% on
  the previous run, so the search is sound, but a low SER invites the question
  Chapter 3 answers with the GP reliability table.
- **`serve()` has no exclusive-bind guard.** `ThreadingHTTPServer` inherits
  `allow_reuse_address = 1`, which on Windows sets `SO_REUSEADDR` and lets a
  second `bopis ui` bind a port already in use. Two were running simultaneously;
  only one received traffic. Worth `SO_EXCLUSIVEADDRUSE` so a second instance
  fails loudly.
- **A jsdom trap.** `drawConvergence()` runs at load; if the 2D-context stub
  throws, the rest of the page script — including `let DOLLY_PENDING` — never
  initialises, and the symptom is a misleading
  `ReferenceError: Cannot access 'DOLLY_PENDING' before initialization` at *call*
  time. The stub must be chainable and tolerate every property, and
  `Element.prototype.scrollTo` must exist. Cost real debugging time; worth
  recording before trusting any headless "the page is broken" result.

---

## 10. What to run — the handoff runbook

Ordered. Steps 0–2 take minutes. Step 3 is the gate. Step 4 is hours. Do not skip
to step 4.

### Step 0 — the scoring environment

On the machine that will run the study, before anything else:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers bert-score
python -m pip install "transformers<5"
```

Three commands, not one. A single `--index-url .../whl/cpu` replaces PyPI
entirely, so `transformers` and `bert-score` come back `No matching distribution
found` if they share a command with torch.

**Accept:** `torch 2.14.0+cpu`, `transformers 4.57.6`, `bert-score 0.3.13`.

The first scoring call downloads `roberta-large` (~1.36 GB) and takes ~111 s.
After that, no further downloads. A cached model in someone else's user profile
is not enough — the run machine needs its own, or the first prompt of the run
stalls.

### Step 1 — the scorer in isolation

No model, no dataset, no `llama-server`. Proves torch, transformers, bert-score
and the roberta-large cache in one shot.

```bash
python -c "from bopis.quality import get_scorer; print(get_scorer('bertscore').score(['a cat sat on the mat'],['a cat sat on the mat'])[0].f1)"
```

**Accept:** a number very close to `1.0`. Under ~2 min means the model was
already cached; much longer means it downloaded. Anything else is in the
troubleshooting table.

### Step 2 — the chat path, end to end

```bash
python -m bopis ui --port 8090 --llama-url http://127.0.0.1:8080 --idle-seconds 30 --data-dir "<ABSOLUTE>\bopis\data"
```

`--data-dir` must be **absolute** — the default is the relative path `data`
(§7). Then open `bopis.html`, click **Dolly prompt**, send.

**Accept, all three:**
- `/api/status` reports `quality.loaded: true` **and no `quality.error`**
- the Dolly pool is ~14,825, not `0`
- the badge reaches a real F1

A badge reading `n/a` on an *edited* Dolly prompt is correct behaviour (§4), not
a failure. Only unmodified prompts should score.

### Step 3 — smoke run: the gate

```bash
python -m bopis run --backend llama-server --llama-binary <path> \
  --models-dir <dir> --iterations 3 --seeds 1 --eval-size 20 --proxy-size 10
```

~30 prompts instead of 550. Two purposes: it proves the **study** path produces
real F1, and its wall time per prompt is what tells you whether the full run is
2 hours or 2 days.

The check that actually matters — no other check in this document matters more:

```bash
python -c "import json;r=[json.loads(l) for l in open(r'runs/<newest>/raw/generations.jsonl',encoding='utf-8')];print('rows',len(r));print('with f1',sum(1 for x in r if x.get('quality_f1') is not None));print('backends',set(x.get('scorer') for x in r))"
```

**Accept:**
- `with f1` equals `rows`. **If `quality_f1` is absent, the run was still
  simulated** and §1 stands unchanged.
- `backends` is `bert-score/roberta-large@17` — not `bertscore-direct`, which is
  the unrescaled fallback and would mean A-38 rescaling did not happen.

### Step 4 — the full run

```bash
python -m bopis run --backend llama-server --llama-binary <path> --models-dir <dir>
```

The defaults *are* the study protocol: `--eval-size 500 --proxy-size 50`, seed
`20260101` (`dataset.py:71-73`, `dataset.py:383`). Add `--iterations` or
`--seeds` only if the smoke run showed the budget is insufficient — changing
them changes the protocol.

`run` writes the run directory **and** copies the payload to the root
`dashboard_data.js` (`cli.py:966-969`). `report` does not (§9).

### Step 5 — what "it worked" means

Do not stop at "it finished." The claim is specific, so check the specific thing:

| check | expected | where |
|---|---|---|
| QRR, quality retention | ≥ 98% | `metrics` → `qrr` |
| SRR, speed retention | ≥ 95% | `metrics` → `srr` |
| per-task floors | 8 tasks, none null | payload `thresholds.q_min_by_task` |
| floors not invented | they vary by run | dashboard card shows a range, not a fixed `0.82` |
| scoring backend | `bert-score/roberta-large@17` | `generations.jsonl` → `scorer` |
| output-length parity | `balanced: true` | `metrics.json` → `output_length_parity` |
| energy actually measured | not `unavailable` | `bopis_profile.js` → `energy_method` |

**If QRR < 98%, report it — do not move the floor.** The floor is the
pre-registered criterion (A-30/A-38); adjusting it to fit the result is the thing
a panel is most likely to catch.

### Troubleshooting — the exact errors hit this session

| symptom | cause | fix |
|---|---|---|
| `AttributeError: RobertaTokenizer has no attribute build_inputs_with_special_tokens` | `transformers` 5.x | `pip install "transformers<5"` |
| `ModuleNotFoundError: No module named 'torch'` | scoring stage never installed | step 0 |
| `No matching distribution found` for `transformers` | torch's `--index-url` replaced PyPI | separate the commands |
| HTTP 500 from `/api/score` | late import, past the wrapper | read `quality.error` in `/api/status` |
| `loaded: true` but scoring 500s | `__init__` needs no torch | fixed; confirm `quality.error` absent |
| badge `n/a` after editing a Dolly prompt | reference correctly withheld | §4, expected |
| `/api/dolly` 404, or `dolly.n = 0` | relative `--data-dir` | absolute path, §7 |
| `quality_f1` missing in `generations.jsonl` | the run was simulated | §1; check `--backend` |
| a second `bopis ui` binds the port silently | Windows `SO_REUSEADDR` | §9, unfixed |

---

## 11. Also changed this round

- **`README.md:69-77`, dependency policy.** It said the scorer "runs as a separate
  offline scoring stage; the measurement path never loads it." The second clause
  is right, the first was wrong — the same error as the `bertscore.py` docstring.
  Now: constructed by `bopis.cli`, warmed up before the run, injected into
  `StudyRunner` as a plain argument, and **scoring happens inline, after each
  batch's generation and after that batch's energy window has closed**, so the
  forward pass is never billed to inference.
- **`README.md:90-113`, install.** The single command
  `pip install transformers torch bert-score` is what produced the 5.x break.
  Replaced with the three-command sequence, the reason it must be pinned, and the
  verified version set. This is the line the groupmate will read.
- **`docs/AMENDMENTS.md:738` gains A-43** — the length bias: absolute F1 is not a
  quality scale, QRR is primary, the system prompt's brevity is a partial
  confound, and `runner.py`'s empty-answer justification is wrong. Written with
  the §6 measurements rather than asserted.
- **`runner.py` now reports output-length parity**, closing the one confound that
  survives §6. `payload["output_length_parity"]`, eight tests, and a
  `balanced: false` path that must be reported if it ever fires.
- **A-38 and A-43 are different failures.** A-38 covers the narrow high band;
  A-43 covers length sensitivity. Both are now stated.

**Process note, because it will bite again:** PowerShell's
`Add-Content -Encoding UTF8` re-encoded all 734 lines of `docs/AMENDMENTS.md`
(CRLF, no BOM) — 780 insertions / 734 deletions in `git diff --stat`. Repaired
with `git checkout --` plus a byte-level append in Python; final diff is 45
insertions, 0 deletions. Check `git diff --stat` after any scripted append to a
CRLF file in this repo.

---

## Files touched

```
 README.md                      |  27 +-    install: + transformers<5; corrected scoring-stage note
 bopis.html                     |  93 ++-   thresholds interpolated; 3-way reference states
 bopis/dashboard.py             |   9 +    thresholds: + s_min, + q_min_by_task
 bopis/metrics.py               | 138 ++-   output_length_parity + OutputLengthParity
 bopis/quality/__init__.py      |  47 ++-   SCORER_REQUIREMENTS, missing_dependencies
 bopis/quality/bertscore.py     |  38 ++-   actionable RuntimeError; corrected docstring
 bopis/runner.py                |  35 ++    output_length_parity into the payload
 bopis/ui_server.py             |  24 +-    scorer_error, honest `loaded`, status error
 docs/AMENDMENTS.md             |  45 ++    A-43: length bias, QRR primary, empty-answer defect
 tests/test_e2e_sim.py          |  39 ++    test_dashboard_carries_the_quality_floors
 tests/test_metrics.py          |  92 ++    TestOutputLengthParity (8 cases)
 tests/test_provenance.py       |  96 ++    threshold guards + reference-state guards
 bopis_profile.js               | 1372 +-  regenerated for this machine
 dashboard_data.js              |          republished from runs/20260925T221100Z
```

Environment, not tracked by git — **must be replicated on any machine that runs
the study**, or every `/api/score` call returns HTTP 500 and the run finishes
with no quality numbers:

```
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers bert-score
python -m pip install "transformers<5"        # bert-score 0.3.13 breaks on 5.x
```

Resulting: `torch 2.14.0+cpu`, `transformers 4.57.6`, `bert-score 0.3.13`.

Not part of this change set, and untracked deliberately:

- `docs/THESIS_REVISIONS_TO_APPLY.md` — your own notes, left alone
- `docs/THESIS_WRITING2_G2.pdf` — source document
- `docs/MANUSCRIPT_CHANGES_FROM_BERTSCORE_AUDIT.md` — what this session implies
  for the manuscript text; **not committed**, as requested
