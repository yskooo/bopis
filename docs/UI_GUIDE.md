# BOPIS UI Guide — what every panel shows, and what it does not

**Written for:** the BOPIS team, to be read before demonstrating the tool.
**Updated:** 2026-09-18

This is the walkthrough. Every number the UI displays is traced back to the file
that produced it, and every place the UI deliberately shows `n/a` is explained,
because "why is this blank?" is the first question a panellist asks.

---

## 0. The 60-second demo path

1. Start the server and UI (§1).
2. Click **Workspace** in the sidebar — chat left, dashboard right.
3. Type a prompt. Point at the **Stage 1 badge**: the detected task, and the
   precision prior it feeds.
4. Point at the reply's metrics: tokens, decode speed, latency, and the
   **`~estimated (Mode C)`** energy.
5. On the right, click **▶ Replay search**. Narrate it: *"the first ten are
   prior-weighted random seeds; after that Expected Improvement closes in."*
   Then point at the **search log** below the chart, which streams the same run
   as text: what the surrogate predicted, what was measured, and the error.
6. Point at **x\* and why it was chosen** — the status, the two threshold checks,
   and the selection rule.
7. Finish on **BOPIS vs. Random Search** — equal budget, same selection rule.

Then say the two limits out loud before anyone asks: the energy figure is an
estimate, and the run on screen is `sim` until a real study has been run.

## 1. Start it

The server on its own — this is all the chat panel needs:

```powershell
.\tools\cpu\llama-server.exe --model .\models\qwen2.5-1.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8080 --ctx-size 2048 --n-gpu-layers 0 --threads 4 --parallel 1
```

Then, for measured energy and BERTScore in the chat, the instrument bridge
(start LibreHardwareMonitor as administrator with *Remote Web Server → Run*
first, see [ENERGY_MODES.md](ENERGY_MODES.md#mode-d-cpu-package-energy-via-rapl)):

```powershell
python -m bopis ui          # then open http://127.0.0.1:8090/
```

It calibrates idle package power for 30 s. Leave the machine alone while it
does. Without a hardware monitor it still runs, and reports a per-process
Mode C estimate instead. Without the bridge, `bopis.html` opened as a file
talks to llama-server directly and falls back to the in-browser estimate.

Port 8080 is hardcoded in `bopis.html`'s `CHATBOT_CONFIG`. Or the launcher,
which also refreshes the generated files and waits for `/health`:

```powershell
.\demo.ps1 -LlamaBinary .\tools\cpu\llama-server.exe `
           -Model Q4_K_M=.\models\qwen2.5-1.5b-instruct-q4_k_m.gguf `
           -GpuLayers 0 -Threads 4 -MaxTokens 256
```

`GpuLayers 0` is deliberate. Offloading to the MX330 is **3.5x slower** than
CPU-only — see `STATUS_AND_ACTION_ITEMS.md` §4a. Both `tools\cpu\` and
`tools\vulkan\` are installed; the CPU build is faster on this host.

For the demo sequence and the words to say, see
[DEMO_RUNBOOK.md](DEMO_RUNBOOK.md) — this guide is the reference, that one is
the script.

The launcher refreshes three generated files, then starts the server and waits
for `/health` before opening the browser:

| File | Written by | Feeds |
| :--- | :--- | :--- |
| `bopis_profile.js` | `bopis profile --write-js` | Hardware panel, energy-estimate constants |
| `bopis_rules.js` | `bopis.classify --write-js` | Rule classifier (fallback) |
| `bopis_model.js` | `bopis.classify_trained --write-js` | **Trained** classifier (preferred) |
| `dashboard_data.js` | a completed study run | Every dashboard panel |

`dashboard_data.js` is **not** created by the launcher. Copy it from a run:

```powershell
Copy-Item runs\<run-id>\dashboard_data.js .\dashboard_data.js
```

or use the **Load run data** button in the dashboard. Without it the dashboard
panels read "Load a run to…" and the Pareto chart stays empty. That is correct
behaviour, not a bug: the UI never invents a number.

To stop the server, use PowerShell. `pkill` silently fails on Windows:

```powershell
Get-Process -Name llama-server | Stop-Process -Force
```

---

## 2. Optimization Chat

Three things render per prompt.

### 2.1 Your message (right, blue)

### 2.2 The Stage 1 badge

```
● Stage 1 · Classification · Low sensitivity · 51% posterior · trained model
Seed prior P(precision | task): F32 10% · F16 35% · Q8_0 28% · Q4_K_M 28%
```

| Field | Meaning |
| :--- | :--- |
| Task label | Predicted Dolly category |
| Sensitivity | High/Medium/Low quality sensitivity to quantization |
| Posterior | The model's confidence. `fallback` means no rule matched (rule mode only) |
| `trained model` / `rules` | Which classifier answered |
| Seed prior | `P(precision \| task)` this category contributes to BO seeding |

**Expect it to be wrong about 3 times in 10.** Held-out accuracy is 69.7%. It
has already been observed calling *"Summarize why quantization saves energy"* a
**General QA**. The correct response to that on stage is not embarrassment:

> "That is a misclassification. Our measured held-out accuracy is 69.7% exact,
> 79.2% on the sensitivity tier. It cannot corrupt `x*` — the prior only weights
> the 10 seed draws out of 30 evaluations; the other 20 follow Expected
> Improvement over the GP, and `x*` is chosen by Pareto dominance plus the
> SRR/QRR floors."

### 2.3 The reply, with its metrics strip

```
Prompt 98 tok   Generated 8 tok   Decode 8.72 tok/s   Latency 1.77 s
~Energy 26.6 J (3.33 J/tok) · ~PHP 0.11/1k prompts · estimated (Mode C)
BERTScore n/a (needs a reference answer)
```

Prompt/generated/decode/latency come straight from llama.cpp's `timings`.

With `bopis ui` running, the strip reads instead:

```
Energy 41.3 J [38.9–43.7] (1.62 J/tok) · PHP 0.165/1k replies · measured (RAPL)
BERTScore F1 0.412 (P 0.398 / R 0.427, rescaled)
```

(These numbers illustrate the format only.)

- **Green `measured (RAPL)`** is CPU package energy, net of idle, over the
  request window. The bracket is the monitor's refresh-bin resolution. **Amber
  `estimated (Mode C)`** means no hardware monitor was reachable. The two are
  never rendered alike.
- **BERTScore** appears only for a **Dolly prompt** (the button left of the
  input). That prompt is sent as a fresh single turn with the study's own
  template, and after the energy window closes the reply is scored against
  Dolly's human reference. The reference is shown under the reply. Editing the
  loaded prompt drops the reference, because it no longer answers the edited
  text. Free chat still says `n/a`: there is nothing to compare against.
- The header shows which instruments are live, and a **session tally**: total
  joules, J/token, PHP, and mean F1.

In **Workspace**, sending a message now keeps you in Workspace. Earlier, send
called the chat tab's switch, which exited the split view.

### 2.4 Same prompt again, and the three-way comparison

Every Dolly prompt has a fixed ID (e.g. `dolly-07308`), shown on the Dolly
button once loaded. Under each Dolly reply there are two buttons:

- **↺ Ask again** loads that exact prompt back into the input. The dock's
  **↺ recent** list does the same for the last 15 Dolly prompts.
- **⚖ Compare default · random search · BOPIS** runs that same prompt under
  the three configurations from the loaded run: the unoptimized default, the
  random-search pick, and `x*`. Each one starts its own llama-server with that
  configuration (model, precision, threads, batch, GPU layers) on port 8081,
  measures the answer, and then all three are BERTScored against the same Dolly
  reference. The result is a table: energy, speed, F1, and the answer itself.

This is the study's Stage 3 comparison on **one** prompt, using the study
protocol (`/completion`, study template, `n_predict = t`, no system prompt). It
is an illustration. The reported comparison is still the run's 500-prompt
validation. It takes a minute or two, because each configuration loads its own
model. It needs `bopis ui --llama-binary ...` (demo.ps1 passes it) and the model
files in `models\`.

**Random search is automated too.** Both search arms are algorithms in the tool.
Random search tries 30 configurations at random (the baseline). BOPIS chooses
its 30 with the surrogate model. Same budget, same selection rule. Neither is a
person choosing.

## 2b. Configurations: what BOPIS chooses from

The **Configurations** tab makes the "selection" in Intelligent Configuration
Selection visible:

- **The funnel**: 2304 possible configurations. Minus Table H1's limits for this
  machine (batch, GPU layers, threads). Minus each rejection rule, with its
  count (doesn't fit in memory, precision not allowed on this GPU, and so on).
  That leaves the feasible space BOPIS searches. Then the 30 it measured, the
  Pareto front, and `x*`.
- **Model × precision grid**: for each model and precision, how many runtime
  settings are still possible and how much memory it needs, or why it was ruled
  out. With a run loaded, it also shows how many each search arm tried, and
  marks ★ `x*` and ◆ the random-search pick.
- **The full list**: every feasible configuration, filterable by model, and by
  whether BOPIS or random search tried it or it is on the front.

Every number comes from `bopis_profile.js` (`hardware.feasible_space`) and the
loaded run; the page only counts.

## 2c. Starting LibreHardwareMonitor automatically

```powershell
.\tools\start-hwmon.ps1 -Install     # or: .\demo.ps1 ... -StartHwmon
```

This installs it with winget if needed and writes its settings: web server on
port 8085, no authentication, start in the tray, and a **250 ms** refresh
instead of 1 s (so every energy figure's ± range is about 4× narrower). Then it
starts it as administrator and waits until the CPU Package sensor answers. The
one Windows admin prompt cannot be avoided: RAPL is read through a kernel
driver, so no tool (CLI or not) reads it without admin rights.

---

## 3. Dashboard

## 2a. Workspace — chat and dashboard in one window

The **Workspace** item in the sidebar puts the conversation and the evidence side
by side, the way an editor shows source beside a preview:

```
┌──────────────────────┬────────────────────────────────────────┐
│ Optimization Chat    │ Dashboard                              │
│                      │   indicator cards                      │
│  your prompt         │   x* and why · BOPIS vs Random         │
│  Stage 1 badge       │   surrogate reliability                │
│  model reply         │   Pareto front · convergence           │
│                      │                                        │
│  [ input dock ]      │                                        │
└──────────────────────┴────────────────────────────────────────┘
```

This is the layout to demonstrate in: ask a question on the left, and the
configuration that answered it is visible on the right at the same time.

Details worth knowing:

- **Both columns scroll independently.** Reading the Pareto front does not
  scroll the chat away.
- The input dock is constrained to the chat column rather than spanning the
  window. It is fixed-position, so its right edge is recomputed from the chat
  column on entry and on resize.
- Clicking any other sidebar item leaves workspace mode, so the two layouts can
  never both claim the main area.
- Below 1100px it stacks into one column.
- No DOM is moved when toggling — it is a CSS-grid switch plus a visibility
  class — so the chat transcript and the 3D camera angle survive.

### 3.0 Layout — the research split view

The dashboard is a two-column research view:

```
┌─────────────────────────────┬──────────────────────────────────┐
│ LEFT — evidence & analysis  │ RIGHT — charts                   │
│                             │                                  │
│  Selected configuration x*  │  Pareto front (3D, rotatable)    │
│  BOPIS vs. Random Search    │  Search convergence              │
│  Surrogate reliability      │                                  │
└─────────────────────────────┴──────────────────────────────────┘
```

It collapses to a single column below 1200px, so a projector or a small laptop
screen still works. The run-evidence banner and the four indicator cards stay
full width above the split.

Both canvases size themselves from their parent's width, which is zero while the
panel is hidden, so they are redrawn on tab switch and on window resize. If a
chart ever looks squashed, resizing the window re-renders it.

### 3.0a Live refresh — watching a run as it happens

The **Live** pill in the run-evidence header re-reads `dashboard_data.js` every
4 seconds and redraws only when the run has actually advanced, so the 3D view
does not reset under you. The label reports progress, e.g. `Live: 18/30 evals`.

Turn it on before starting a study, and the dashboard fills in as the search
proceeds.

**How it works, and its one limitation.** `bopis.html` is opened over `file://`,
where `fetch()` is blocked by the same-origin policy. So live refresh re-appends
a `<script>` element with a cache-busting query string, which the browser does
re-request. That works, but it means **the run must be writing
`dashboard_data.js` to the repo root as it goes.** Today `dashboard_data.js` is
written when a run *finishes*, so the pill is a progress monitor across
successive runs rather than a true within-run tick. To get a genuine live view,
have the runner write the payload incrementally to the root path the page loads.

If the file is missing the label reads `Live: no data file`, which is accurate
rather than silent.

### 3.1 Run evidence banner

`20260909T172504Z · sim · simulated energy · x* t1024_b2_Q8_0_g14_c4`

**Read the second field every time.** `sim` means the analytic simulator, whose
own manifest says *"No energy figure from this backend may be reported as an
empirical result."* If it says `sim`, you are demonstrating the pipeline, not
reporting a measurement. An estimated run additionally renders an amber block
with the formula and every assumption.

### 3.2 The four indicator cards

| Card | Metric | Meaning |
| :--- | :--- | :--- |
| Session Prompts | budget | Configurations measured (30 = 10 seeds + 20 BO) |
| Energy Total & Savings | **EIR** | Energy reduction vs. unoptimized default |
| Avg Generation Speed | **SRR** | Speed *retention* vs. default |
| Overall Quality | **QRR** | Quality *retention* vs. default |

SRR above 100% is not a bug — it means `x*` is **faster** than the default, and
`154.8%` means 1.55x the default's tokens/second. SRR and QRR are retention
ratios with floors (95% and 98%), not improvement targets.

### 3.3 Pareto front chart

- **Circles** — configurations BOPIS evaluated.
- **Squares** — the random-search arm. Off by default; tick **Show random
  search**. Shape rather than colour, because colour already encodes quality.
- **Colour** — quality (BERTScore F1), dark purple low to yellow high.
- **Bubble size** — generation speed.
- **Blue dashed line** — the Pareto front.
- **`x*`** — BOPIS's pick. **Not** the lowest-energy point, by design.
- **`default`** (red) — the unoptimized reference.
- The shaded surface is *interpolated* for readability. Only the markers are
  measured. Say so if asked.

#### Replay search — the single best way to explain BO

The **▶ Replay search** button animates the optimization in evaluation order.
Points appear one at a time, the interpolated surface firms up as evidence
accumulates, the convergence curve grows beside it, and the readout names the
phase you are watching:

```
evaluation 5 / 30 · prior-weighted seed
evaluation 18 / 30 · Expected Improvement
```

Watch what it shows, because it is the argument for the whole method: the first
10 evaluations scatter across the space (prior-weighted random seeding), and then
Expected Improvement visibly stops wandering and concentrates on the low-energy
region. That is Bayesian optimization working, without needing to say a word
about acquisition functions.

`x*` is deliberately **withheld until the replay finishes**. It is a conclusion,
and showing it from frame one would misrepresent how it was reached. Press the
button again to stop early and restore the full view.

**↻ Auto-rotate** spins the trade-off surface slowly so all three objectives can
be read from any angle. It pauses while you drag, so it never fights you.

#### The search log — the same run as text

Below the chart, in the same card, is the optimizer's log. It exists because a
3D scatter plot is not evidence to everyone: some panellists want the numbers.
On load it shows the whole run (90 lines for a 30-evaluation study); during a
replay it streams, driven by the **same tick** as the chart, so the line and the
point for an evaluation always appear together.

Four line types per Bayesian-optimization trial:

```
+5.42s  acq    EI=0.067 J · proposes t128_b2_Q8_0_g14_c2 · predicts 98.34 ± 15.65 J
+5.42s  bo     16/30 t128_b2_Q8_0_g14_c2 E=91.39J v=21.22tok/s F1=0.8206
+5.42s  gp     residual -6.96J vs µ · 0.44σ inside 1σ
+5.42s  front  non-dominated · front = 7 configurations
```

- **`acq`** — the acquisition step: the expected improvement, the configuration
  it proposes, and the surrogate's prediction **with its uncertainty**. Read
  from `surrogate.points[].expected_improvement / mu / sigma`.
- **`bo`** / **`seed`** — the measurement. `seed` for the prior-weighted seeds,
  `bo` once Expected Improvement is driving.
- **`gp`** — the one-step-ahead residual: measured minus predicted, in units of
  the GP's own sigma. This is the honest calibration check, because `mu` was
  predicted *before* this measurement existed. `inside 1σ` means the surrogate's
  stated confidence was justified.
- **`front`** — the running non-dominated count. Deliberately **recomputed on
  the evaluations revealed so far**, not read from `on_front` (which is the
  *final* front), so a line claiming "non-dominated" is true at that moment. You
  can watch configurations get dominated and dropped. It converges to
  `pareto.n_front`.

Two caveats the footer states in the UI itself: the elapsed column is
**interpolated** from the run's total wall time — per-evaluation timings were
never recorded — and the closing `gp final fit` / `leave-one-out` lines are
whole-run values, which is why they appear at the end rather than beside any
single evaluation.

**Click any line to scrub the chart to that evaluation.** The log doubles as a
timeline control, which is useful when a panellist asks "go back to the one
where it found the big improvement". **Follow** autoscrolls and switches itself
off if you scroll up to re-read. **Copy** puts the whole log on the clipboard as
plain text, which is the fastest way to get it into an appendix.

Click any point to inspect it. The readout names which arm it came from:

```
x* (BOPIS pick) · (t=1024, b=2, p=Q8_0, g=14, c=4) · Q8_0
 · Energy 63.21 J · Speed 23.40 tok/s · Quality (BERTScore F1) 0.8412 · Iter 11
```

### 3.4 Selected configuration x* — and why

This panel answers "how does the user see that it really picked the best one?"
It shows all five search parameters **with the domain each was drawn from**, the
size of the searched space, the unoptimized default for contrast, and then the
justification:

- **Status** — `optimal`, `relaxed_qrr_95`, `relaxed_srr_90`, or
  `min_energy_fallback`. This is the four-rung relaxation ladder from amendment
  A-34. **`min_energy_fallback` means the optimization did not succeed** — the
  panel says so in red. Do not let a fallback run be presented as a win.
- **SRR ≥ 95% / QRR ≥ 98%** — ✓ met or ✗ not met, against the live thresholds.
- **On Pareto front** — e.g. 9 of 30 evaluated.
- **Hypervolume (normalized)** — front quality in [0, 1].
- **The selection rule, stated in full:** keep the non-dominated set over
  (energy↓, speed↑, quality↑), drop anything below the SRR/QRR floors, then take
  argmax EIR.

That last line is the honest answer to "is this the best config?" — it is the
best *under a stated rule*, and the rule is on screen.

### 3.5 BOPIS vs. Random Search

The controlled comparison, side by side with a per-metric delta and winner:

| Metric | BOPIS | Random search | Delta | Better |
| :--- | ---: | ---: | ---: | :--- |
| EIR — energy improvement | 41.8% | 38.6% | +3.2% | BOPIS |
| SRR — speed retention | 154.8% | 148.8% | +6.0% | BOPIS |
| QRR — quality retention | 98.6% | 98.6% | +0.0% | tie (within 0.1) |

Why this is a fair comparison, and worth saying out loud: **both arms get the
identical 30-evaluation budget, and both pick their winner with the same rule.**
Random search is not a straw man — it samples uniformly without replacement
(amendment A-32) and then runs the same Pareto + retention + argmax-EIR
selection. So any difference is attributable to the search strategy alone.

A verdict is only claimed when the difference survives the displayed precision;
otherwise the row reads `tie (within 0.1)`. On the sim run, BOPIS wins on energy
and speed and ties on quality — which is the expected shape, since BO optimizes
energy and the thresholds merely protect speed and quality.

### 3.5a Surrogate reliability — is the GP trustworthy?

Left column, third card. This is the panel to open when someone asks "where is
the machine learning, and does it work?"

| Metric | LOO (primary) | One-step-ahead | Target |
| :--- | ---: | ---: | :--- |
| R² | 0.980 ✓ | 0.799 | ≥ 0.85 |
| NPE | 3.77% ✓ | 8.03% | ≤ 10% |
| UCR (calibration) | 0.967 ✓ | 0.850 | ≈ 0.95 |
| MAE (J) | 4.70 | 8.76 | lower |

Plus the **learned** hyperparameters: length scale, signal variance, noise
variance, and the log marginal likelihood they were fitted by.

Two columns, because they answer different questions. **Leave-one-out is the
primary basis** (amendment A-40) — it measures how well the surrogate
generalizes. **One-step-ahead is deliberately pessimistic**: it scores the GP on
the points Expected Improvement chose *because* the GP was least certain there,
so it measures exploration as much as accuracy. Quote LOO, show both.

### 3.5b Search convergence

Right column, under the Pareto chart. Best-energy-so-far per evaluation, both
arms overlaid, with a dashed line marking the seeds/BO boundary at evaluation 10.
A flat tail is convergence, not failure — it means the search stopped finding
improvements.

### 3.6 Available models and configurations

The **System Settings** panel shows the live feasible space: permitted
precisions, GPU layers, batch sizes, CPU threads, rules fired, and
`n_feasible of 768`. The x* panel repeats each parameter's domain inline, so the
selected value is always shown next to the alternatives it beat.

---

## 4. The two `n/a` fields — read before the defence

### 4.0 BERTScore *is* what drives the quality comparison — read this first

A point that is easy to get backwards: **BERTScore is exactly what the
BOPIS-vs-random-search quality comparison is built on.** The chain is

```
BERTScore F1 per prompt        (bopis/quality/bertscore.py)
  -> mean F1 per condition
  -> QRR = (Q_bopis / Q_default) x 100        (bopis/metrics.py:145)
  -> the "QRR — quality retention" row of the comparison table
  -> and the colour of every point on the Pareto chart
```

So when the table reports QRR 98.6% for BOPIS against 98.6% for random search,
that *is* a BERTScore result. The `n/a` in the chat panel is a different
situation entirely, described next. Two contexts, one metric:

| Context | BERTScore | Why |
| :--- | :--- | :--- |
| Dolly evaluation prompts | **Computed** | Dolly ships a human reference for every row |
| Free chat message | **`n/a`** | No reference answer exists to compare against |

### 4.1 Why BERTScore says `n/a` in chat, and why that cannot be fixed

**BERTScore compares a generated answer against a reference answer.** A live
chat prompt has no reference — there is no ground truth for "explain quantization
to me", so there is nothing to compare against. This is a property of the metric,
not a missing feature.

Where BERTScore legitimately lives:

```
Dolly instruction + context
   -> llama-server using x*
   -> generated response
   -> compared against Dolly's reference response
   -> BERTScore F1
```

Dolly ships a human-written reference for all 15,011 rows, which is exactly why
the study evaluates on Dolly and not on free chat. The implementation is
`bopis/quality/bertscore.py`; it needs `transformers` and `torch`, so it runs as
a **separate offline stage** after a study and nothing in the measurement path
imports it (amendment A-6). Its output is the `quality_f1` on every Pareto point
and the QRR indicator.

So: **quality is measurable on Dolly, never on free chat.** If you want a quality
number in a live demo, run a Dolly prompt through the study path and show the
dashboard, not the chat box.

### 4.1a CLOSED 2026-09-24 — BERTScore now runs inside real studies

`bopis run --backend llama-server` now BERTScores every generation against its
Dolly reference after each batch. That happens after llama-server is stopped,
so it never overlaps an energy window. So `quality_f1`, QRR and the Pareto
colour scale are real on a real run, and selection of `x*` uses them. The
generations are saved to `raw/generations.jsonl` with their references and
scores. `--quality none` turns it off. Without torch/transformers, `auto`
prints a loud *QUALITY: UNSCORED* banner rather than proceeding silently. The
scorer loads (and on first use downloads roberta-large, ~1.4 GB) **before** the
study starts.

The history below is kept for the record.

Found 2026-09-18.

`bopis/quality/bertscore.py` is a complete, working BERTScore implementation, and
`bopis.quality.get_scorer()` is the documented way in. **Nothing in the codebase
calls it.** A grep for `get_scorer` outside its own package returns no hits.

Consequences on a **real** (`--backend llama-server`) run:

- `LlamaServerMeasurer` leaves `quality_f1` unset, by design — it says so:
  *"Leaves `quality_f1` unset: BERTScore requires a transformer forward pass."*
- No later stage fills it in.
- So `quality_f1` stays `None`, QRR cannot be computed, and the QRR row plus the
  Pareto colour scale would be empty.

The QRR 98.6% currently on the dashboard comes from the **simulator**, which
synthesises `quality_f1` analytically (`scorer = analytic_model` in
`validation/per_prompt_quality.csv`). It is not a BERTScore number.

**What the fix requires — two changes, in this order:**

1. **Persist the generations.** `GenResult.text` carries the model's output
   during a run, but no run artifact stores it. Without the candidate text there
   is nothing to score afterwards. Add the response text to the per-prompt
   validation output.
2. **Add a scoring command**, e.g. `bopis score <run-dir>`, that loads those
   generations plus the Dolly references already saved in
   `dataset/sample_500.csv`, calls `get_scorer("bertscore")`, and writes
   `quality_f1` back into `validation/per_prompt_quality.csv` with
   `scorer = bertscore`.

Doing it as a separate offline command is the right shape, not a shortcut: it is
what amendment A-6 requires. Loading torch inside the measurement loop would
both violate the standard-library-only policy enforced by
`tests/test_provenance.py` and contaminate the energy measurement it runs beside.

Until this lands, the honest statement is: *"quality is instrumented end to end
and validated on the simulator; BERTScore scoring of real generations is the
remaining step."*

### 4.2 Why energy is an estimate, and how to make it real

The MX330 exposes no power sensor: both `nvmlDeviceGetPowerUsage` and
`nvmlDeviceGetTotalEnergyConsumption` return `NVML_ERROR_NOT_SUPPORTED`. No
driver flag changes this. Separately, llama.cpp does not report energy at all.

The chat now shows a **Mode C estimate**, computed in `estimateEnergy()` from
constants exported into `bopis_profile.js` (so the browser and `bopis.metrics`
cannot disagree):

```
E_hat = P_cpu_dyn * T          (g = 0, so the GPU term is zero)
PHP   = E_hat / 3.6e6 * tariff
```

with `P_cpu_dyn = 15 W` from `bopis/monitor/estimator.py` and
`tariff = PHP 14.35/kWh` from `bopis/metrics.py`. It assumes the process
saturates its threads for the request duration — **that assumption is the weak
link**, which is why every figure is prefixed `~`, coloured amber, labelled
`estimated (Mode C)`, and carries the full caveat on hover.

A useful external check: the UI reports roughly **1–3 J/token**. Zähl & Hennig
(2026) measured 0.56–0.65 J/token for 1B models on an RTX 4060Ti. Being somewhat
higher on CPU inference is plausible, so this passes the §3.2 Route 3
plausibility check in the energy brief.

Routes to a real number:

| Route | Gives | Cost |
| :--- | :--- | :--- |
| **RAPL** (`MSR_PKG_ENERGY_STATUS`) | Measured CPU joules — and since `g = 0`, that is nearly all of it | **Now implemented** as Mode D: LibreHardwareMonitor/OHM loads the driver, BOPIS reads its web feed with stdlib `urllib`. `--energy-mode cpu-rapl`, `bopis ui` |
| A GPU exposing NVML power | Measured GPU joules | Different machine |
| External wall meter | Whole-system joules | ~PHP 1–2k |
| Mode C (current) | **Relative** comparison only | Free |

The defensible position today: Mode C is sufficient for **EIR**, because
multiplicative bias cancels in a ratio (the §3.1 cancellation proof). It is *not*
sufficient for absolute J/token. RAPL is the cheapest real upgrade.

---

## 5. What to say, and what not to say

**Say:**

- "The instrument is built and verified." 517 tests pass.
- "Energy here is a labelled Mode C estimate."
- "The classifier is 69.7% on a held-out test set; it informs seeding only."
- "Both search arms had equal budget and the same selection rule."
- "`x*` is the best configuration under a stated rule, which is on screen."

**Do not say:**

- "We measured energy." Not on this GPU.
- "We trained the LLM." Nothing fine-tunes a language model — see
  `STATUS_AND_ACTION_ITEMS.md` §6.
- "BOPIS adapts the configuration per prompt." `p`, `g`, `c`, `b` are
  `llama-server` launch flags; only `t` is per-request.
- "This is `x*`" while demonstrating the chat on a manual config. The launcher
  prints a warning for exactly this case.
- "The GPU speeds up inference." It is 3.5x slower here.

---

## 6. Traceability

| UI element | Source |
| :--- | :--- |
| Hardware panel, feasible space | `bopis/hardware.py` → `bopis_profile.js` |
| Energy-estimate constants | `bopis/monitor/estimator.py`, `bopis/metrics.py` → `bopis_profile.js` |
| Stage 1 badge (trained) | `bopis/classify_trained.py` → `bopis_model.js` |
| Stage 1 badge (rules) | `bopis/classify.py` → `bopis_rules.js` |
| Indicator cards, Pareto, x*, comparison | `bopis/dashboard.py` → `dashboard_data.js` |
| Selection status ladder | `bopis/pareto.py` (amendment A-34) |
| Thresholds (SRR/QRR/EIR) | `bopis/metrics.py` |
| Quality / BERTScore F1 | `bopis/quality/bertscore.py`, offline |

Both browser classifiers are ports of Python, cross-checked against it: the
rules agree on 16/16 prompts, and the Naive Bayes agrees on 155/155 real Dolly
rows with a maximum confidence delta of **0.0** — bitwise identical, not
approximate.
