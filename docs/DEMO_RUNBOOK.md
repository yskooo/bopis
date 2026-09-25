# BOPIS Demo Runbook — paste these, say this

**Written for:** whoever is driving the laptop during the defence.
**Updated:** 2026-09-22

The other docs explain *why*. This one is just *what to type* and *what to say*.
Read it once the day before, then keep it open on your phone during the demo.

If you only remember one thing: **you are demonstrating a working instrument,
not reporting measured results.** Say that first and the whole defence gets
easier. Claim measured energy and you will lose the room.

---

## 1. Pre-flight — Steps 0-2 ten minutes before, Step 3 the night before

### Step 0: Reboot. Seriously.

Not optional. Run this first:

```powershell
py -3 -m bopis profile --model-aware --ctx-size 2048
```

Look at the last two lines. If you see this, **stop and reboot**:

```
  |X_feasible|                 0 of 768 unconstrained
  Rejected                     {'HW-P0': 64}
```

That means the feasibility guard rejected every configuration because the
machine has no free RAM. As of 2026-09-22 this host had **1.10 GiB free of
15.78**, and Chrome plus VS Code will do that to you again. After a clean boot
you should have ~13 GiB free and a non-zero feasible space.

A panellist who opens **System Settings** and sees `FEASIBLE SPACE 0 / 768`
will ask why, and "my laptop was full" is a bad answer. See §6 — there is a
second, deeper reason this number can stay at 0.

### Step 1: Refresh the three generated files

```powershell
py -3 -m bopis profile --model-aware --write-js bopis_profile.js
py -3 -m bopis.classify --write-js bopis_rules.js
py -3 -m bopis.classify_trained --data-dir data --write-js bopis_model.js
```

### Step 2: Give the dashboard a run to read

```powershell
Copy-Item runs\20260909T172504Z\dashboard_data.js .\dashboard_data.js
```

Without this the dashboard reads "Load a run to…" and the Pareto chart is
empty. That is correct behaviour, not a bug — but it is not a demo.

### Step 3: Confirm the tests still pass — the night before, not that morning

```powershell
py -3 -m unittest discover -s tests -t .
```

**This takes over 8 minutes.** Verified 2026-09-22: `Ran 517 tests in 490.620s
— OK`. So run it the night before; it does not belong in a 10-minute
pre-flight.

**517** is the number to say. `STATUS_AND_ACTION_ITEMS.md` §8 still says 467 —
that figure is stale, ignore it. If you have changed code since, re-run and use
whatever `Ran N tests` prints for you.

---

## 2. Start the server

```powershell
.\tools\cpu\llama-server.exe --model .\models\qwen2.5-1.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8080 --ctx-size 2048 --n-gpu-layers 0 --threads 4 --parallel 1
```

Leave that terminal open. It is the server; closing it kills the chatbot.

Three things about this command:

- **Port 8080 is not negotiable.** `bopis.html` hardcodes
  `http://127.0.0.1:8080/v1/chat/completions`. Change the port and the chat
  panel shows "Chatbot unavailable".
- **`--n-gpu-layers 0` means CPU-only, and that is deliberate.** Offloading to
  this laptop's MX330 is measured at **3.5x slower** than CPU
  (STATUS_AND_ACTION_ITEMS §4a). If a panellist asks why you are not using the
  GPU, that is the answer — and it is a finding, not an excuse.
- **`tools\cpu\` not `tools\vulkan\`.** Both exist and both work, but the CPU
  build is the faster one here and has one less moving part.

Confirm it is up, in a **second** terminal:

```powershell
curl http://127.0.0.1:8080/health
```

Then open the UI:

```powershell
Start-Process .\bopis.html
```

### Or do all of it with one command

```powershell
.\demo.ps1 -LlamaBinary .\tools\cpu\llama-server.exe -Model .\models\qwen2.5-1.5b-instruct-q4_k_m.gguf -GpuLayers 0 -Threads 4
```

This refreshes the generated files, starts the server, **waits for `/health`
before opening the browser**, and shuts down cleanly on Ctrl+C. It does not
copy `dashboard_data.js` — Step 2 above is still yours.

It will print `This is not x* and carries no optimizer claim`. That warning is
real: with no `-Run` argument you are serving a hand-picked configuration. To
deploy the actual optimizer result instead:

```powershell
.\demo.ps1 -Run runs\20260909T172504Z -LlamaBinary .\tools\cpu\llama-server.exe -Model Q4_K_M=.\models\qwen2.5-1.5b-instruct-q4_k_m.gguf
```

### When it is over

`pkill` silently fails on Windows and leaves orphans eating RAM. Use:

```powershell
Get-Process -Name llama-server | Stop-Process -Force
```

---

## 3. The demo — 7 beats, ~4 minutes

Click **Workspace** in the sidebar first: chat on the left, evidence on the
right. Everything below happens in that one screen.

| # | Do this | Say this |
| :--- | :--- | :--- |
| 1 | Type: *Summarize the key findings from this climate policy report.* | "The prompt is classified before anything is optimized." |
| 2 | Point at the **Stage 1 badge** | "Summarization, from a classifier trained on Dolly 15k — 69.7% on a held-out test set. It only seeds the search; it does not pick the answer." |
| 3 | Point at the reply's **metrics strip** | "Tokens, decode speed, latency, and energy — and the energy is labelled `~estimated (Mode C)`, because this GPU has no power sensor." |
| 4 | Right pane: click **▶ Replay search** | "Thirty configurations, in the order they were evaluated." |
| 5 | **Let it run and point at the search log** underneath the chart | "First ten are prior-weighted random seeds. Then Expected Improvement takes over — watch it stop wandering. The log is the optimizer's own output: for each trial it prints what the surrogate *predicted*, then what we *measured*, then the error." |
| 6 | Point at **x\* and why it was chosen** | "Best configuration under a stated rule: stay on the Pareto front, clear the speed and quality floors, then maximize energy savings. The rule is on screen — we are not asking you to trust a number." |
| 7 | Finish on **BOPIS vs. Random Search** | "Same 30-evaluation budget, same selection rule. The only difference is the surrogate, so the difference is attributable to the method." |

Then, **before anyone asks**, say the two limits out loud:

> "Two things to be clear about. The energy figure is a labelled estimate, not a
> measurement — this GPU exposes no power telemetry. And the run on screen is
> from the simulator, so treat it as a demonstration that the instrument works,
> not as our results."

Volunteering this is the single highest-value thing you can do. It converts
your biggest weakness into evidence that you understand your own method.

### Beat 5 is your strongest beat — use it

The search log pairs each line with the point appearing on the chart on the same
tick. It reads like this:

```
+5.42s  acq    EI=0.067 J · proposes t128_b2_Q8_0_g14_c2 · predicts 98.34 ± 15.65 J
+5.42s  bo     16/30 t128_b2_Q8_0_g14_c2 E=91.39J v=21.22tok/s F1=0.8206
+5.42s  gp     residual -6.96J vs µ · 0.44σ inside 1σ
+5.42s  front  non-dominated · front = 7 configurations
```

That is the whole method in four lines: the surrogate proposes, states its
uncertainty, gets measured, and is scored on its own error. `inside 1σ` means
the GP's confidence was honest. Click any line to rewind the chart to it.

If someone says "Bayesian optimization is a black box", scroll this log.

---

## 4. The six questions you will get

**"Did you train a language model?"**
No. Nothing fine-tunes an LLM. Two things are learned: the Gaussian Process
surrogate of energy, and the task classifier. The LLM is the *objective
function* being optimized, not the model being trained.

**"So where is the machine learning?"**
Bayesian optimization is a recognized ML subfield, and the GP is a trained
model — hyperparameters fitted by maximizing log marginal likelihood, not set
by hand. The trained classifier is separately supervised: 69.7% held-out vs a
48.2% rule baseline. STATUS §6 has the long version.

**"Is this supervised, unsupervised, or reinforcement learning?"**
The classifier is supervised. The GP surrogate is supervised regression inside
a sequential-decision loop — closest to active learning, not reinforcement
learning: there is no policy and no reward signal. STATUS §6.4.

**"You said energy but you cannot measure energy."**
Correct, and it is labelled everywhere in the UI. The MX330 exposes neither
`nvmlDeviceGetPowerUsage` nor an energy counter. Mode C is a resource-allocation
estimate. It supports the energy-savings *ratio*, because systematic bias
cancels between the two configurations being compared; it does **not** support
absolute joules-per-token. docs/ENERGY_MODES.md.

**"Does it re-optimize for every prompt?"**
No, and be careful here. Of the five parameters, only maximum generation length
is per-request. Precision, GPU layers, threads and batch size are
`llama-server` launch flags. BOPIS selects one configuration for a workload.

**"Why is the GPU not being used?"**
Because we measured it and it is 3.5x slower on this host. That is a finding.
It also means the GPU-layer dimension may be degenerate on this hardware — see
§6.

---

## 5. If it breaks mid-demo

| Symptom | Fix |
| :--- | :--- |
| Chat says **"Chatbot unavailable · Failed to fetch"** | The server is not up or not on 8080. Check the server terminal, re-run §2, resend the prompt. |
| Dashboard says **"Load a run to…"** everywhere | `dashboard_data.js` is missing. Pre-flight Step 2, or click **Load run data** and pick `runs\20260909T172504Z\dashboard_data.js`. |
| **`FEASIBLE SPACE 0 / 768`** | Out of RAM. Close Chrome, re-run Step 1. Do not dwell on the panel; move to the dashboard. |
| Pareto chart blank or tiny | Switch tabs and back — the canvas sizes off its parent, which is zero-width while hidden. |
| Replay shows nothing | No run data. Same fix as row 2. |
| Server died on its own | Almost always RAM. Reboot, then §2. |

---

## 6. What is still missing — know these before the panel finds them

Ordered by how much damage each does if a panellist raises it first.

**1. Every recorded run is simulated.** All runs under `runs/` carry
`backend: "sim"`. There is no measured `selection.csv` yet. This is why the
defensible claim is "the instrument is complete and verified" and not "we have
results". Fixing it means one real study with
`--backend llama-server --model-aware` (STATUS A-2). Everything else downstream
is gated on that one run.

**2. BERTScore is implemented but never actually called.** `get_scorer` is
never invoked outside `bopis/quality/`. On a real run `quality_f1` stays
`None`, so **QRR cannot be computed** and the quality column of the
BOPIS-vs-random comparison would be empty. The 98.7% QRR on the dashboard right
now is the *simulator's* analytic value, not BERTScore. Two changes needed:
persist the generated text into the run artifacts (nothing stores it today),
then add an offline `bopis score <run-dir>` command. This blocks any real
quality claim. STATUS A-10.

**3. The feasibility guard models a model you do not have.** `--model-aware` is
hardcoded to `MISTRAL_7B_INSTRUCT_V03` at three call sites in `bopis/cli.py`
(79, 553, 879), and `hardware.py` defines exactly one `ModelSpec`. But the only
weights on disk are Qwen2.5-1.5B. So the feasible space is being computed for a
7B model at 7.42 GiB while you demo a 1.12 GB one. This is the real reason
`0 / 768` is sticky: it is not only low RAM, it is the wrong model. Fix is
small — add a Qwen `ModelSpec` and a `--model-spec` flag — but it touches a
thesis-critical module, so it is a team decision, and it is entangled with the
undisclosed model substitution (STATUS B-4).

**4. GPU offload may be a degenerate dimension.** `g` is one of the five search
parameters, but on this host every offload setting is slower than CPU-only. If
that holds on real measurements, the optimizer will always drive `g` to 0 and
one fifth of the search space is decoration. The Vulkan-only benchmark should be
repeated with a CUDA build before anyone claims `g` matters (STATUS B-5).

**5. No repeatability run.** Amendment A-36 wants one configuration measured
five times with the coefficient of variation reported. Without it you cannot
answer "how do you know that 41% is not noise?" (STATUS A-3).

### My honest read

Gaps 1 and 2 are the ones that matter, and they are the same gap wearing two
hats: **no real measurement has happened yet.** The engineering is genuinely
thorough — 517 tests, a stdlib-only core enforced by a provenance test, a GP
fitted by marginal likelihood, honest labelling of every estimate. That is more
rigour than most undergraduate theses carry.

But rigour about a simulation is still a simulation. If you have time for
exactly one thing before the defence, it is **one real run on the 1.5B model**,
even a small one, even with Mode C energy and no BERTScore. One non-sim
`selection.csv` changes your answer to the hardest question from "we built the
instrument" to "we built it and here is what it found".

Gap 3 is the cheapest thing on this list and it is what makes your hardware
panel look broken. Fix that before you fix anything else.
