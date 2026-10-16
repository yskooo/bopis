# RESUME.md — read this FIRST after any session reset

**If you are the assistant and the conversation just restarted, read this file
before doing anything else.** It is the handoff for the current working state.

> **Note:** this file now lives in `docs/`. Any path below written as
> `.\something` or `docs/...` is relative to the repository root (`..\` from
> here). The content is a snapshot from 2026-10-07 and is not current.

Last updated: 2026-10-07

---

## Where we are

- **Repo root:** `C:\Aaron\4thyear1stsem\Thesis\bopis` (git repo, branch `main`)
- **Pull:** `main` was fast-forwarded to `f3b5019` this session.
- **Deadline:** the defense is **tomorrow**. The user is the **tool presenter**.

## What exists because of this session

| Thing | Path | Notes |
|---|---|---|
| Presentation script | `docs/TOOL_PRESENTATION_SCRIPT.md` | **main deliverable.** User journey first, source-code Q&A last, branch A/B/C for results, literature-backed thresholds, chatbot disclaimer. |
| Model weights | `models\qwen2.5-1.5b-instruct-q4_k_m.gguf` | 1.04 GB. Downloaded with `hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF --include "*q4_k_m*.gguf" --local-dir models`. **gitignored** (`models/`, `*.gguf`). |
| llama-server binary | `tools\cpu\` | 34 files / 671 MB, copied from `C:\llama.cpp` (build `b10809`, CUDA 13). **gitignored** (`tools/*`). |
| Recovery script | `ensure_backend.ps1` | one command, restarts whatever is dead. |
| Extracted manuscript text | `%TEMP%\opencode\manuscript.txt` | throwaway |

## How to bring the backend up (one command)

```powershell
powershell -ExecutionPolicy Bypass -File .\ensure_backend.ps1
```

It checks llama-server (:8080), the bridge (:8090), and does a real
`POST /api/chat`. Prints `ALL GREEN - open http://127.0.0.1:8090/` on success.

Manual form, if you need it:

```powershell
# terminal 1 — the model server
.\tools\cpu\llama-server.exe --model .\models\qwen2.5-1.5b-instruct-q4_k_m.gguf `
  --host 127.0.0.1 --port 8080 --ctx-size 2048 --n-gpu-layers 0 --threads 4 --parallel 1

# terminal 2 — the measurement bridge
python -m bopis ui
```

Then open `http://127.0.0.1:8090/`.

## State at last update

- llama-server: **UP** on :8080, PID 14116, model loaded
- bridge (`python -m bopis ui`): **UP** on :8090, PID 3532
- `POST /api/chat` through the bridge: **HTTP 200**, returns completions
- End-to-end smoke test passed earlier this session

## Things that will bite you

1. **`llama-server` with no `--model` still returns `/health` 200.**
   The bridge then reports `"ready": true` while every chat request 400s with
   `model name is missing` / `model 'local' not found`. This is exactly the
   failure the user originally reported. **Always check `/v1/models`, not
   `/health`.**
2. **The model id llama-server advertises is the literal path string**
   (`.\models\qwen2.5-1.5b-instruct-q4_k_m.gguf`), but the frontend sends
   `model: "local"` (`bopis.html:5463`). This **works** (HTTP 200) as long as a
   model is loaded — llama.cpp only validates the name when zero models are
   loaded. Do not "fix" it.
3. **Energy is ESTIMATED, never measured.** The MX330 exposes no power
   telemetry and there is no LibreHardwareMonitor feed on :8085, so
   `/api/status` shows `energy.measured: false` and joule figures carry an
   `ESTIMATED, NOT MEASURED` caveat. **Never present these as measured energy.**
4. **`bopis.html` hardcodes port 8080** (`CHATBOT_CONFIG.endpoint`, line 5462)
   and the bridge base `http://127.0.0.1:8090` (line 3714). Do not change ports.
5. **`tools\cpu\` is a CUDA build, not a CPU build.** It works because
   `--n-gpu-layers 0` keeps everything on CPU, but the folder name is
   aspirational. `find_llama_binary()` prefers `tools\vulkan\` first, then
   `tools\cpu\` (`ui_server.py`).
6. **`LATEST_MANUSCRIPT_THESIS_WRITING2_G2.md` is deleted locally** (`git status`
   shows ` D`). The manuscript being used is
   `LATEST_MANUSCRIPT_REVISED_SYSTEM_ARCHITECTURE_THESIS_WRITING2_G2 (2).md`
   and the PDF `C:\Aaron\4thyear1stsem\Thesis\BOPIS_Tool-Manuscript_G2.pdf`.
7. **Untracked, uncommitted:** `docs/TOOL_PRESENTATION_SCRIPT.md`,
   `ensure_backend.ps1`, `docs/THESIS_WRITING2_G2.pdf`,
   `docs/THESIS_REVISIONS_TO_APPLY.md`, `docs/MANUSCRIPT_CHANGES_FROM_BERTSCORE_AUDIT.md`.
   Nothing has been committed this session — the user has not asked for a commit.

## Results numbers (all SIMULATOR-backed, not measured)

From `docs/INITIAL_RESULTS.md`, run `20261005T085002Z_ui-sim`:

- Verdict `optimal`: **EIR 14.80%** (pass > 0), **SRR 122.59%** (pass >= 95%),
  **QRR 98.26%** (pass >= 98%)
- Selected config: `qwen2.5-1.5b / Q8_0 / t=512 / b=2 / g=14 / c=4`
- Default 42.61 J · Random 14.88 J · BOPIS 36.30 J
- **Random search fails QRR at 82.66%** — that is the branch-B talking point
- 30 evals each; within 5% of best at iter 6 (BOPIS) vs 9 (random); SER 0.53
- `docs/INITIAL_RESULTS.md` says explicitly: do not present the simulator result
  as proof of real-world GP accuracy

## Presentation script structure

`docs/TOOL_PRESENTATION_SCRIPT.md`

1. THE DISCLAIMER — "we added a chat so you can see it; the chat is not BOPIS"
2. Handoff line (end of slide 7)
3. Steps 1–9 — the user journey (turn on → ask → dashboard → how it chose →
   system & configs → run a study → result → close)
4. Step 8b — thresholds with literature backing (Jain 1991, Lannelongue 2021,
   Kakolyris 2024, Bast 2024, Zhang 2020, Wilkins 2024, Bergstra & Bengio 2012)
5. Branches A / B / C for the BOPIS-vs-random result
6. SOURCE CODE last — folder tree, process→file table, follow-ups
7. DEFENDING EVERY NUMBER — 15-metric card table
8. Say / Don't say + 60-second opening and closing

## Next steps

- [ ] Confirm whether the user wants the presentation script committed
- [ ] Rehearse the script end to end with the live backend
- [ ] Decide whether to pull Q8_0 and F16 for the real precision sweep
      (`hf download ... --include "*q8_0*.gguf" "*fp16*.gguf"`)
