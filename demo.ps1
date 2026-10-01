<#
.SYNOPSIS
    BOPIS demo launcher --- matches the startup sequence in docs/UI_GUIDE.md exactly.

.DESCRIPTION
    Step 1  Refresh the three generated JS files (profile, rule classifier,
            trained classifier) so the UI always shows live values.

    Step 2  Auto-sync dashboard_data.js from the newest run in runs\ so the
            Pareto / convergence panels are pre-loaded without any manual copy.

    Step 3  Start llama-server with the Qwen2.5-1.5B Q4_K_M model on port 8080
            and wait for /health before touching the browser.

    Step 4  Start the instrument bridge (python -m bopis ui, port 8090).
            This gives measured CPU-package energy (if LibreHardwareMonitor is
            running as administrator with Remote Web Server on) or a per-process
            Mode C estimate otherwise. Also enables BERTScore on Dolly prompts.
            Skip with -NoBridge to open bopis.html directly instead.

    Step 5  Open the UI: http://127.0.0.1:8090/ if the bridge is up,
            otherwise bopis.html from file://.

    Step 6  Hold. Ctrl+C stops everything cleanly.

.PARAMETER LlamaBinary
    Path to llama-server.exe. Defaults to .\tools\cpu\llama-server.exe

.PARAMETER Model
    Path to the GGUF file. Defaults to .\models\qwen2.5-1.5b-instruct-q4_k_m.gguf

.PARAMETER GpuLayers
    GPU layers to offload. Default 0 (CPU-only). CPU is 3.5x faster than the
    MX330 on this host --- leave at 0 for the demo unless you have a reason.

.PARAMETER Threads
    CPU threads for inference. Default 4.

.PARAMETER Parallel
    Concurrent slots (--parallel). Default 1 for the demo.

.PARAMETER CtxSize
    Context window size. Default 2048 (matches DEMO_RUNBOOK).

.PARAMETER MaxTokens
    Max tokens per reply (--n-predict). Default 256.

.PARAMETER Port
    llama-server port. Default 8080 --- hardcoded in bopis.html, do not change.

.PARAMETER UiPort
    bopis ui bridge port. Default 8090.

.PARAMETER IdleSeconds
    Seconds the bridge spends calibrating idle CPU power. Default 30.
    Leave the machine alone during this window.

.PARAMETER HwmonUrl
    LibreHardwareMonitor / OHM data.json URL. Default http://127.0.0.1:8085/data.json

.PARAMETER NoBridge
    Skip the bopis ui bridge and open bopis.html directly from file://.
    Energy will be an in-browser Mode C estimate; BERTScore will be n/a.

.PARAMETER SkipProfile
    Skip regenerating bopis_profile.js, bopis_rules.js, bopis_model.js.
    Use when you have already run the script once and nothing has changed.

.PARAMETER NoBrowser
    Do not open the browser automatically.

.EXAMPLE
    # Standard defense launch --- everything with defaults:
    .\demo.ps1

.EXAMPLE
    # No bridge --- faster startup, bare bopis.html, Mode C energy only:
    .\demo.ps1 -NoBridge

.EXAMPLE
    # Custom model path:
    .\demo.ps1 -Model .\models\qwen2.5-3b-instruct-q4_k_m.gguf
#>

[CmdletBinding()]
param(
    [string] $LlamaBinary   = '.\tools\cpu\llama-server.exe',
    [string] $Model         = '.\models\qwen2.5-1.5b-instruct-q4_k_m.gguf',
    [int]    $GpuLayers     = 0,
    [int]    $Threads       = 4,
    [int]    $Parallel      = 1,
    [int]    $CtxSize       = 2048,
    [int]    $MaxTokens     = 256,
    [string] $LlamaHost     = '127.0.0.1',
    [int]    $Port          = 8080,
    [int]    $UiPort        = 8090,
    [int]    $IdleSeconds   = 30,
    [string] $HwmonUrl      = 'http://127.0.0.1:8085/data.json',
    [switch] $NoBridge,
    [switch] $SkipProfile,
    [switch] $NoBrowser
)

$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot

# ------ Helpers ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
function Write-Step { param([string]$m) Write-Host "`n==> $m" -ForegroundColor Cyan }
function Write-Ok   { param([string]$m) Write-Host "    + $m" -ForegroundColor Green }
function Write-Warn { param([string]$m) Write-Host "    ! $m" -ForegroundColor Yellow }

function Resolve-Python {
    foreach ($c in @('python', 'py')) {
        $cmd = Get-Command $c -ErrorAction SilentlyContinue
        if ($cmd) {
            return @{ Exe = $c; Args = @() }
        }
    }
    throw 'Neither "python" nor "py" is on PATH.'
}

function Wait-Http {
    param([string]$Url, [int]$TimeoutSec, [System.Diagnostics.Process]$Proc)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $spin = @('|','/','-','\'); $i = 0
    while ((Get-Date) -lt $deadline) {
        if ($Proc -and $Proc.HasExited) {
            Write-Host ''
            throw "Process (PID $($Proc.Id)) exited before becoming ready."
        }
        try {
            $r = Invoke-WebRequest -Uri $Url -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
            if ($r.StatusCode -eq 200) { Write-Host ''; return $true }
        } catch { }
        Write-Host "`r    waiting $($spin[$i++ % 4])" -NoNewline
        Start-Sleep -Milliseconds 600
    }
    Write-Host ''; return $false
}

# ------ 0. Sanity checks ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Write-Step 'Checking prerequisites'

Push-Location $repo

if (-not (Test-Path 'bopis.html')) { throw "bopis.html not found in $repo. Run from the repo root." }
Write-Ok 'bopis.html present'

$llamaPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($LlamaBinary)
$modelPath  = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Model)

if (-not (Test-Path $llamaPath)) {
    Write-Warn "llama-server not found at $llamaPath"
    Write-Warn 'Continuing in UI-only mode (no chat).'
    $llamaPath = $null
}
if (-not (Test-Path $modelPath)) {
    Write-Warn "Model GGUF not found at $modelPath"
    Write-Warn 'Continuing in UI-only mode (no chat).'
    $modelPath = $null
}
if ($llamaPath -and $modelPath) {
    $sizeGiB = [math]::Round((Get-Item $modelPath).Length / 1GB, 2)
    Write-Ok "Model: $modelPath ($sizeGiB GiB)"
}

$canServe = $null -ne $llamaPath -and $null -ne $modelPath
Pop-Location

# ------ 1. Refresh generated JS files ------------------------------------------------------------------------------------------------------------------------------------
if (-not $SkipProfile) {
    Push-Location $repo
    Write-Step 'Refreshing generated JS files (profile + classifiers)'

    $py = Resolve-Python

    # bopis_profile.js --- hardware panel + feasibility guard
    try {
        & $py.Exe ($py.Args + @(
            '-m', 'bopis', 'profile',
            '--model-aware',
            '--ctx-size', "$CtxSize",
            '--write-js', 'bopis_profile.js'
        )) | Out-Null
        Write-Ok 'bopis_profile.js updated'

        # Surface empty feasible space immediately
        $js = Get-Content 'bopis_profile.js' -Raw -ErrorAction SilentlyContinue
        if ($js -match '"n_feasible"\s*:\s*0\b') {
            Write-Warn 'FEASIBLE SPACE IS EMPTY (n_feasible: 0).'
            Write-Warn 'Cause: not enough free RAM. Close Chrome / VS Code, or reboot.'
        }
    } catch { Write-Warn "Profile refresh failed: $_" }

    # bopis_rules.js --- rule-based Stage 1 classifier (fallback)
    try {
        & $py.Exe ($py.Args + @('-m', 'bopis.classify', '--write-js', 'bopis_rules.js')) | Out-Null
        Write-Ok 'bopis_rules.js updated (rule classifier)'
    } catch { Write-Warn "Rule classifier export failed: $_" }

    # bopis_model.js --- trained Naive Bayes classifier (preferred, 69.7%)
    if (Test-Path (Join-Path $repo 'data\databricks-dolly-15k.jsonl')) {
        try {
            & $py.Exe ($py.Args + @(
                '-m', 'bopis.classify_trained',
                '--data-dir', 'data',
                '--write-js', 'bopis_model.js'
            )) | Out-Null
            Write-Ok 'bopis_model.js updated (trained classifier, 69.7% held-out)'
        } catch { Write-Warn "Trained classifier export failed: $_" }
    } else {
        Write-Warn 'Dolly dataset not found --- skipping trained classifier.'
        Write-Warn 'Stage 1 badge will use the rule classifier (49.7%).'
    }

    Pop-Location
}

# ------ 2. Auto-sync dashboard_data.js ---------------------------------------------------------------------------------------------------------------------------------
#
# Picks the newest run directory (timestamp names sort chronologically) and
# copies its dashboard_data.js to the repo root so the Pareto / convergence
# panels are pre-loaded. No manual Copy-Item needed.
#
Write-Step 'Syncing dashboard_data.js'
Push-Location $repo
try {
    $runsDir = Join-Path $repo 'runs'
    $dashTarget = Join-Path $repo 'dashboard_data.js'
    $dashSource = $null

    if (Test-Path $runsDir) {
        $latestRun = Get-ChildItem -Path $runsDir -Directory |
                     Sort-Object Name | Select-Object -Last 1
        if ($latestRun) {
            $candidate = Join-Path $latestRun.FullName 'dashboard_data.js'
            if (Test-Path $candidate) { $dashSource = $candidate }
            else { Write-Warn "Newest run ($($latestRun.Name)) has no dashboard_data.js." }
        } else { Write-Warn 'No run directories found under runs\.' }
    } else { Write-Warn 'runs\ directory not found.' }

    if ($dashSource) {
        Copy-Item -Path $dashSource -Destination $dashTarget -Force
        Write-Ok "dashboard_data.js synced from $(Split-Path (Split-Path $dashSource -Parent) -Leaf)"
    } else {
        Write-Warn 'Dashboard panels will show "Load a run to...". Use the Load button in the UI.'
    }
} catch {
    Write-Warn "dashboard_data.js sync failed: $_"
} finally {
    Pop-Location
}

# ------ 3. Start llama-server ---------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Exactly as documented in docs/UI_GUIDE.md --1:
#
#   .\tools\cpu\llama-server.exe --model .\models\qwen2.5-1.5b-instruct-q4_k_m.gguf
#       --host 127.0.0.1 --port 8080 --ctx-size 2048
#       --n-gpu-layers 0 --threads 4 --parallel 1
#
# Port 8080 is hardcoded in bopis.html --- do not change it.
# GpuLayers 0 is deliberate: MX330 is 3.5x slower than CPU-only on this host.
#
$serverProc = $null
$baseUrl    = "http://${LlamaHost}:${Port}"

if ($canServe) {
    Write-Step 'Starting llama-server'

    $llamaArgs = @(
        '--model',        $modelPath,
        '--host',         $LlamaHost,
        '--port',         "$Port",
        '--ctx-size',     "$CtxSize",
        '--n-gpu-layers', "$GpuLayers",
        '--threads',      "$Threads",
        '--parallel',     "$Parallel",
        '--n-predict',    "$MaxTokens"
    )
    Write-Host "    $llamaPath $($llamaArgs -join ' ')" -ForegroundColor DarkGray

    Push-Location $repo
    $serverProc = Start-Process -FilePath $llamaPath -ArgumentList $llamaArgs `
                                -NoNewWindow -PassThru
    Pop-Location

    Write-Ok "llama-server started (PID $($serverProc.Id))"
    Write-Host '    Waiting for model to load (1.5B Q4_K_M on CPU ~5-10 s)...' -ForegroundColor DarkGray

    if (Wait-Http -Url "$baseUrl/health" -TimeoutSec 120 -Proc $serverProc) {
        Write-Ok "llama-server ready at $baseUrl"
    } else {
        Write-Warn "Server did not respond within 120 s. Opening UI anyway --- first message may fail."
    }
} else {
    Write-Warn 'No llama-server / model --- chat panel will show "Chatbot unavailable".'
    Write-Warn 'Dashboard and hardware panels still work.'
}

# ------ 4. Start the instrument bridge (python -m bopis ui) ------------------------------------------------------------------
#
# Exactly as documented in docs/UI_GUIDE.md --1:
#
#   python -m bopis ui          # then open http://127.0.0.1:8090/
#
# With LibreHardwareMonitor running as admin (Remote Web Server on):
#   --- measured CPU-package energy (RAPL, net of calibrated idle)
# Without it:
#   --- per-process Mode C estimate (better than the pure in-browser guess)
#
# The bridge also provides /api/score (BERTScore) and /api/dolly (random prompt).
# Idle calibration runs for $IdleSeconds --- leave the machine alone.
#
$bridgeProc = $null
$uiUrl      = $null

if ($canServe -and -not $NoBridge) {
    Write-Step "Starting instrument bridge (bopis ui) on port $UiPort"
    Write-Host "    Idle calibration: $IdleSeconds s --- do not touch the machine." -ForegroundColor DarkGray

    $py = Resolve-Python
    $uiArgs = $py.Args + @(
        '-m', 'bopis', 'ui',
        '--port',          "$UiPort",
        '--llama-url',     $baseUrl,
        '--idle-seconds',  "$IdleSeconds",
        '--hwmon-url',     $HwmonUrl,
        '--llama-binary',  $llamaPath,
        '--models-dir',    (Join-Path $repo 'models')
    )
    if ($serverProc) { $uiArgs += @('--llama-pid', "$($serverProc.Id)") }

    Push-Location $repo
    $bridgeProc = Start-Process -FilePath $py.Exe -ArgumentList $uiArgs `
                                -WorkingDirectory $repo -NoNewWindow -PassThru
    Pop-Location

    $bridgeUrl = "http://127.0.0.1:$UiPort/api/status"
    if (Wait-Http -Url $bridgeUrl -TimeoutSec ($IdleSeconds + 60) -Proc $bridgeProc) {
        $uiUrl = "http://127.0.0.1:$UiPort/"
        Write-Ok "Bridge ready --- open $uiUrl"
    } else {
        Write-Warn 'Bridge did not come up. Falling back to bopis.html (file://).'
        Write-Warn 'Energy will be in-browser Mode C estimate; BERTScore will be n/a.'
    }
}

# ------ 5. Open the UI ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if (-not $NoBrowser) {
    Write-Step 'Opening UI'
    if ($uiUrl) {
        Start-Process $uiUrl
        Write-Ok "Opened $uiUrl  (bridge mode --- measured energy + BERTScore)"
    } else {
        Start-Process (Join-Path $repo 'bopis.html')
        Write-Ok 'Opened bopis.html  (file:// mode --- in-browser Mode C estimate)'
    }
}

# ------ Summary ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Write-Host ''
Write-Host '  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------' -ForegroundColor Green
Write-Host '  ---  BOPIS demo is live                                  ---' -ForegroundColor Green
if ($canServe) {
Write-Host "  ---  llama-server : $baseUrl" -ForegroundColor Green
}
if ($uiUrl) {
Write-Host "  ---  UI bridge    : $uiUrl" -ForegroundColor Green
} else {
Write-Host '  ---  UI           : bopis.html (file://)                 ---' -ForegroundColor Green
}
Write-Host '  ---                                                      ---' -ForegroundColor Green
Write-Host '  ---  Press Ctrl+C to stop everything.                    ---' -ForegroundColor Green
Write-Host '  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------' -ForegroundColor Green
Write-Host ''

# ------ 6. Hold until Ctrl+C, then clean up ------------------------------------------------------------------------------------------------------------------
if ($canServe) {
    try {
        while ($true) {
            if ($serverProc.HasExited) {
                Write-Warn "llama-server exited on its own (code $($serverProc.ExitCode))."
                break
            }
            Start-Sleep -Seconds 1
        }
    } finally {
        Write-Step 'Shutting down'
        foreach ($proc in @($bridgeProc, $serverProc)) {
            if ($proc -and -not $proc.HasExited) {
                try {
                    Stop-Process -Id $proc.Id -Force -ErrorAction Stop
                    Write-Ok "Stopped PID $($proc.Id)"
                } catch {
                    Write-Warn "Could not stop PID $($proc.Id): $_"
                }
            }
        }
    }
} else {
    Write-Host '  Running in UI-only mode (no server to watch). Close this window when done.' -ForegroundColor Yellow
}
