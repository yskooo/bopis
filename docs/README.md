# BOPIS docs — start here

**Written for:** the BOPIS thesis team.

Five documents, in the order you probably want them.

| # | Document | Read it when |
| :--- | :--- | :--- |
| 0 | **[DEMO_RUNBOOK.md](DEMO_RUNBOOK.md)** | You are demonstrating **today**. Paste-ready commands, a 7-beat script with the words to say, the six questions you will be asked, and what to do when it breaks mid-demo. Short on purpose. |
| 1 | **[MANUSCRIPT_REVISION_PLAN.md](MANUSCRIPT_REVISION_PLAN.md)** | You are planning what to change in the revised manuscript. Every finding mapped to the section it affects, with draft wording and the decisions the team must make. **Start here.** |
| 2 | **[STATUS_AND_ACTION_ITEMS.md](STATUS_AND_ACTION_ITEMS.md)** | You want to know what the tool actually does today, what is verified, and what is still open. Includes the small-model feasibility table and the device benchmark. |
| 3 | **[UI_GUIDE.md](UI_GUIDE.md)** | You are about to demonstrate the tool. Panel-by-panel walkthrough, a 60-second demo path, and a "what to say / what not to say" list. |
| 4 | **[CHATBOT_INTEGRATION.md](CHATBOT_INTEGRATION.md)** | You are wiring the chatbot, choosing a model, or running the classifiers. |
| 5 | **[ENERGY_ESTIMATOR_AND_ML_BRIEF.md](ENERGY_ESTIMATOR_AND_ML_BRIEF.md)** | You need the energy-estimator defence, the cancellation proof, or the 2025/2026 energy literature. |
| 6 | **[ML_ELEMENTS.md](ML_ELEMENTS.md)** | You need the five elements of the machine learning — data, task, model, learning algorithm, evaluation — for both learned components, with worked examples, measured figures and references. Written for the adviser/panel. |

Reference material:

| Document | Contents |
| :--- | :--- |
| [AMENDMENTS.md](AMENDMENTS.md) | The amendment register (A-1 … A-40) |
| [CHANGELOG_G2_EDITS.md](CHANGELOG_G2_EDITS.md) | Which amendments landed in the manuscript, with verification anchors |
| [RRL.md](RRL.md) | Citations and their justifications, with inline verification notes |
| [ENERGY_MODES.md](ENERGY_MODES.md) | Modes A/B/C and the paste-ready limitation paragraph |

---

## The four things to know before you talk about this project

1. **The instrument is built and verified — but every recorded run is simulated.**
   All runs under `runs/` carry `backend: "sim"`, whose own manifest says no
   energy figure from it may be reported as an empirical result. The defensible
   claim is *"the instrument is complete and verified"*, not *"we have results."*

2. **Energy is estimated, not measured.** The study GPU exposes no power sensor.
   Mode C is sufficient for the EIR *ratio*, because systematic bias cancels; it
   is not sufficient for absolute J/token.

3. **Nothing fine-tunes a language model.** The learned components are the
   Gaussian Process surrogate and the task classifier. The LLM is the objective
   function being optimized, not the model being trained. See
   MANUSCRIPT_REVISION_PLAN §8 for the full defence, including the
   supervised / unsupervised / reinforcement answer.

4. **BOPIS selects one configuration for the workload, not one per prompt.** Of
   the five search parameters, only maximum generation length is a per-request
   quantity; the rest are inference-server launch flags.

---

## Running it

Just the server, which is all the chat panel needs:

```powershell
.\tools\cpu\llama-server.exe --model .\models\qwen2.5-1.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8080 --ctx-size 2048 --n-gpu-layers 0 --threads 4 --parallel 1
```

Port 8080 is hardcoded in `bopis.html`; change it and the chat panel cannot
reach the server. Or the one-command launcher, which also refreshes the
generated data and waits for `/health` before opening the browser:

```powershell
# one command: refresh generated data, start llama-server, open the UI
.\demo.ps1 -LlamaBinary .\tools\cpu\llama-server.exe `
           -Model Q4_K_M=.\models\qwen2.5-1.5b-instruct-q4_k_m.gguf `
           -GpuLayers 0 -Threads 4 -MaxTokens 256
```

`-GpuLayers 0` is deliberate: offloading to this host's discrete GPU is 3.5x
slower than CPU-only (STATUS_AND_ACTION_ITEMS §4a). `tools\cpu\` and
`tools\vulkan\` both work; the CPU build is the faster one on this host.

Neither the inference binaries (`tools/`) nor the model weights (`models/`) are
committed — they are large and reproducible. Fetch them per
STATUS_AND_ACTION_ITEMS §7 B-2 and B-3.

To stop a server, use PowerShell. `pkill` silently fails on Windows and leaves
orphans eating RAM:

```powershell
Get-Process -Name llama-server | Stop-Process -Force
```

## Verifying it

```powershell
python -m unittest discover -s tests -t .              # 517 tests
python -m bopis.classify --data-dir data               # rule classifier accuracy
python -m bopis.classify_trained --data-dir data       # trained vs rule baseline
python -m bopis profile --model-aware --ctx-size 2048  # live feasible space
.\tools\bench.ps1                                      # device benchmark
```

## Regenerating what the UI reads

```powershell
python -m bopis profile --model-aware --write-js bopis_profile.js
python -m bopis.classify --write-js bopis_rules.js
python -m bopis.classify_trained --data-dir data --write-js bopis_model.js
Copy-Item runs\<run-id>\dashboard_data.js .\dashboard_data.js
```

`demo.ps1` does the first three automatically. `dashboard_data.js` comes from a
completed run and is gitignored, so the dashboard reads "Load a run to…" on a
fresh clone — that is correct behaviour, not a bug. The UI never invents a value.
