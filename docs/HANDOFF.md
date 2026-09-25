# Handoff: Live dashboard, run-a-study, chart labels (2026-09-26)

## Done (branch feat/measured-energy-and-model-ladder)
- `bopis ui` auto-detects `tools\vulkan\llama-server.exe` (then `tools\cpu`), so the
  chat-bar configuration picker (/api/deploy) works without `--llama-binary`.
- Dashboard **Live | Manual** toggle (bopis.html `setDataMode`, `liveTick`).
  Live polls `/api/live` (ui_server.py `live_run`): streams `calibration/bo_log.csv`
  rows onto the 3D chart + search log while a run is going (`LIVE_PARTIAL`),
  loads the full run when `dashboard_data.js` appears, replays it if it arrived
  while watching. Manual = dropdown of runs (`/api/runs`) + "From a file…".
- **▶ Run a study** panel: POST `/api/study` {preset sim|real, iterations,
  proxy_size>=8} spawns `python -m bopis run --skip-validation`; real uses
  llama-server on port 8084. Log: `runs/ui_study.log`. Stop: `/api/study/stop`.
- `bopis run` writes `progress.json` per stage; added missing `--data-dir` to `run`.
- Chart: **Labels** toggle, **My chat prompts** (scored Dolly chat replies as pink
  diamonds), "Prompts this run measured on" list (`/api/run/prompts`).
- Tests: tests/test_live_run.py; full suite passes. Docs: docs/UI_GUIDE.md §3.0a.

## Selection reasoning (manuscript ch. 3)
BO (GP + EI) picks configs to measure -> Pareto front over (E, -S, -Q) ->
x* = argmin E s.t. SRR >= 95% and QRR >= 98% vs default; BOPIS-optimal if EIR > 0.

## Action plan
1. Restart `python -m bopis ui`, Ctrl+F5, click Run a study -> Quick; confirm chart moves.
2. Try Real preset (12 evals, 8 prompts) to see true per-evaluation streaming.
3. Known gaps: sim runs with --skip-validation show "n/a energy" in the banner
   (meta.energy_scope missing); UI studies skip the 500-prompt validation, so
   thesis numbers still need a full CLI run.
4. Open a PR to main when satisfied.
