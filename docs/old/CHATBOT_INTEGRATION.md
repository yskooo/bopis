# BOPIS Chatbot Integration

## Quick start (added 2026-09-18)

One command does the whole sequence — refresh the hardware profile, regenerate
the Stage 1 rule table, start `llama-server`, wait for the model to finish
loading, then open the UI:

```powershell
.\demo.ps1 -Run runs\<run-directory> `
           -LlamaBinary C:\Tools\llama-server.exe `
           -Model Q4_K_M=C:\Models\model.Q4_K_M.gguf
```

Without `-Run` it launches a manually specified configuration instead, which is
fine for showing the chatbot but **is not `x*`** and carries no optimizer claim.
Without `-LlamaBinary`/`-Model` it runs UI-only. The script waits on `/health`
before opening the browser, because a 7B model on CPU needs 30–60 s to load and
opening the UI first is what produces a "connection refused" mid-demo.

### Model choice for a live demo

The manuscript's model is Mistral 7B, but on the 2 GB MX330 every GPU-offload
configuration is permanently rejected by HW-P0 (3.39 GiB of weights against 1.94
GiB of VRAM), so `g` collapses to a single level and inference is CPU-only at
roughly 2–5 tok/s. For a *live* demonstration prefer a 1–3B model at Q4_K_M
(Qwen2.5-1.5B, Llama-3.2-1B/3B, Phi-3-mini): it fits VRAM at `g > 0`, which
restores the GPU-layer dimension, and it answers fast enough to hold a room.
Disclose it as a demo-platform substitution, and note that it also aligns the
study with the 1B–7B range benchmarked by Zähl & Hennig (2026).

## Stage 1: task classification at deployment time

During evaluation the task category is Dolly's own `category` field — ground
truth. During deployment there is no label, so `bopis/classify.py` infers one
from the prompt with a rule-based classifier and looks up
`P(precision | task)`.

```powershell
# score the rules against Dolly's 15k human labels
py -3 -m bopis.classify --data-dir data

# classify one prompt
py -3 -m bopis.classify --prompt "Summarize this report."

# export the rule table for the bopis.html badge
py -3 -m bopis.classify --write-js bopis_rules.js
```

Measured accuracy on all 15,011 labelled Dolly rows: **49.7%** exact 8-way,
**64.3%** with `open_qa`/`general_qa` merged, **67.5%** on the quality-
sensitivity tier that actually drives the prior. Quote the tier figure, and be
ready to explain the bound: the largest error class is `general_qa` predicted as
`open_qa` (1,798 rows), a distinction Dolly draws on a property of the *answer*
rather than of the instruction's surface form, so no lexical rule recovers it.

Why a 49.7% classifier is nonetheless acceptable: the prior weights **only the
10 seed draws** of the 30-evaluation budget. The 20 BO-guided steps follow
Expected Improvement over the GP, and `x*` is chosen by Pareto dominance plus
the SRR/QRR thresholds. A misclassification makes the search slightly less
sample-efficient; it cannot make `x*` wrong.

`bopis.html` shows the detected task as a badge on each submitted prompt. The
rules are authored in Python and exported to `bopis_rules.js`; the browser
reimplements only the scoring arithmetic, and the two are cross-checked to agree
exactly.

### The supervised alternative

`bopis/classify_trained.py` fits a multinomial Naive Bayes classifier to Dolly's
human labels and reaches **69.7%** exact 8-way accuracy on a held-out test set,
against **48.2%** for the rules on the same split — **+21.4 points**.

```powershell
# train, evaluate against the rule baseline, and report
py -3 -m bopis.classify_trained --data-dir data

# persist the fitted model
py -3 -m bopis.classify_trained --data-dir data --save task_classifier.json
```

Protocol: stratified 70/15/15 split (seed 20260101, stratified because Dolly is
unbalanced), smoothing selected on validation only, test set touched once.
`predict_prompt()` returns the same `TaskPrediction` type as the rule
classifier, so the two are interchangeable at the call site.

**The UI badge uses the trained model.** Export it once and `bopis.html` picks
it up:

```powershell
py -3 -m bopis.classify_trained --data-dir data --write-js bopis_model.js
```

That writes ~855 KiB of JSON, which is immaterial for a page opened from local
disk. If `bopis_model.js` is absent the badge falls back to the rule classifier
and says so. Either way the badge states which classifier produced the answer,
so a demo never silently misrepresents its accuracy. The browser's Naive Bayes
scoring was cross-checked against Python on 150 real Dolly rows and agrees to a
maximum confidence delta of **0.0** — bitwise-identical posteriors, not an
approximation.

One finding worth keeping: the rules still *beat* the learned model on
`closed_qa` (89.1% vs 81.2%) and `open_qa` (78.5% vs 77.0%), where the
structural context gate is genuinely strong. A gate-then-model hybrid would
likely beat either alone.

### What "adaptive" can and cannot mean here

Only `t` is a per-request parameter (`n_predict`). `p`, `g`, `c` and `b` are
`llama-server` **launch flags**, so a single running server cannot switch
precision or GPU-layer count per prompt. Per-prompt adaptation over the full
`x = (t, b, p, g, c)` tuple would require one server process per configuration
or a reload between prompts. Today BOPIS selects **one** `x*` for the workload;
the classifier informs seeding and reports the per-task `Q_min(task)` threshold
required by amendment A-26. Do not describe the current system as switching
configuration per prompt.

---

BOPIS has two related but separate jobs:

1. **Optimization and evaluation:** run the same prompts through candidate
   configurations and compare energy, speed, quality, and resources.
2. **Deployment:** start the chatbot server using the selected configuration
   `x*` so normal user requests use the measured settings.

The Databricks Dolly 15k dataset belongs to the first job. Dolly provides
standardized instructions and reference responses for evaluation; it is not
chatbot memory, a knowledge base, or training data unless a separate fine-tuning
or retrieval system is explicitly added.

## Workflow

```text
Dolly 15k -> 500 evaluation prompts + 50 proxy prompts
          -> BOPIS search and validation
          -> runs/<run>/selection.csv
          -> bopis serve
          -> llama-server chatbot endpoint
```

The selected configuration is `x = (t, b, p, g, c)`:

| Field | Deployment meaning |
| --- | --- |
| `t` | Maximum generated tokens, sent as request `n_predict` |
| `b` | Number of server slots, passed as `--parallel` |
| `p` | GGUF precision or quantization variant |
| `g` | GPU layers, passed as `--n-gpu-layers` |
| `c` | CPU threads, passed as `--threads` |

The context window is fixed separately at `--ctx-size 2048`.

## 1. Prepare Dolly

Use the real dataset for a reportable evaluation:

```cmd
py -3 -m bopis dataset --data-dir data
```

This downloads and filters Dolly 15k, then creates the fixed stratified
500-prompt evaluation set and nested 50-prompt proxy subset when a study is
run. Each prompt retains its reference response for the offline BERTScore
stage.

For an offline software demonstration only:

```cmd
py -3 -m bopis dataset --synthetic
```

Synthetic prompts and simulated energy must be labelled as demonstration data,
not as real chatbot or hardware results.

## 2. Run BOPIS

On hardware that supports the required energy instrument and has the model
files:

```cmd
py -3 -m bopis run ^
  --backend llama-server ^
  --model-aware ^
  --model F16=C:\Models\model.F16.gguf ^
  --model Q8_0=C:\Models\model.Q8_0.gguf ^
  --model Q4_K_M=C:\Models\model.Q4_K_M.gguf ^
  --llama-binary C:\Tools\llama-server.exe
```

In Command Prompt, use `^` for line continuation. In PowerShell, use a
backtick instead. A real run is refused when the machine reports no power or
energy telemetry unless `--allow-no-power` is supplied. That flag is for
pipeline testing only and does not produce a valid energy claim.

A completed run writes `selection.csv`. The row with `selected=true` is the
BOPIS deployment recommendation. It also writes the search logs, validation
CSV files, manifest, metrics, Pareto front, and dashboard payload.

## 3. Deploy the selected configuration

Use the completed run and the same GGUF paths:

```cmd
py -3 -m bopis serve ^
  --run runs\<run-directory> ^
  --llama-binary C:\Tools\llama-server.exe ^
  --model F16=C:\Models\model.F16.gguf ^
  --model Q8_0=C:\Models\model.Q8_0.gguf ^
  --model Q4_K_M=C:\Models\model.Q4_K_M.gguf
```

`serve` performs these checks before launching:

- reads `selection.csv` and reconstructs the selected `x*`;
- checks that `x*` is feasible on the current host;
- checks that the selected precision has a supplied GGUF path;
- checks that the GGUF file exists;
- launches `llama-server` with the selected model, GPU layers, threads,
  parallel slots, context size, and seed;
- keeps the server alive until `Ctrl+C`.

The server endpoint is:

```text
http://127.0.0.1:8080
```

For the included `bopis.html` chat UI, use llama.cpp's OpenAI-compatible chat
endpoint. The structured `messages` payload keeps user and assistant turns
separate, so the model does not treat a flattened transcript as text to
continue:

```http
POST http://127.0.0.1:8080/v1/chat/completions
Content-Type: application/json
```

```json
{
  "model": "local",
  "messages": [
    {"role": "system", "content": "Answer the user directly and do not continue a transcript."},
    {"role": "user", "content": "Explain Bayesian optimization in simple terms."}
  ],
  "temperature": 0.0,
  "max_tokens": 512,
  "stream": false
}
```

To use the chat UI, start `llama-server` first, then open `bopis.html`. Type
the user prompt in the **Optimization Chat** input box. The terminal and
`curl.exe` commands are backend diagnostics; they are not the normal place for
the panel demonstration. The UI displays generated tokens, decode speed, and
latency when llama.cpp returns them. Energy remains unavailable on the MX330,
and BERTScore remains an offline Dolly evaluation metric.

The hardware panel in `bopis.html` is data-driven. Refresh its profile snapshot
after hardware, driver, RAM, or VRAM changes with:

```cmd
py -3 -m bopis profile --model-aware --write-js bopis_profile.js
```

This writes the live profile and feasible-space summary to `bopis_profile.js`.
The HTML does not embed CPU, GPU, RAM, VRAM, driver, or energy-support values;
if the generated file is missing, the panel displays `Profile unavailable`.

## 4. Use Dolly to demonstrate quality

A live chatbot request demonstrates deployment. Dolly demonstrates evaluation:

```text
Dolly instruction + context
        -> llama-server using x*
        -> generated response
        -> compare with Dolly reference response
        -> BERTScore F1, speed, energy, and resource records
```

The same prompt set must be used for the unoptimized default, random-search
winner, and BOPIS winner. This makes the comparison attributable to the
configuration strategy rather than to different user questions.

## Current development laptop limitation

The development laptop's MX330 reports `Power Draw: N/A` and does not expose
an NVML energy counter. It can run computation, but it cannot support a valid
GPU-only joule claim. Its 2 GiB VRAM also rejects the Mistral 7B GPU-offload
configurations through `HW-P0`.

Therefore:

- use `--backend sim --synthetic` to demonstrate the complete software
  pipeline;
- use a supported GPU for the manuscript's GPU-energy experiment; or
- use an external wattmeter and revise the measurement scope to
  `whole_system` energy.

Do not present simulated energy or null energy values as measured GPU energy.
