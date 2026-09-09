# BOPIS Chatbot Integration

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
