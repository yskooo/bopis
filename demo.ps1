<#
.SYNOPSIS
    One-command BOPIS demo: refresh the hardware profile, start llama-server on
    the selected configuration x*, wait until it is actually answering, then open
    the bopis.html chat UI.

.DESCRIPTION
    This is the launcher for a live demonstration. It does the sequencing that is
    easy to get wrong under pressure:

      1. Refreshes bopis_profile.js so the UI's hardware panel shows live values
         rather than a stale snapshot.
      2. Resolves x* from a completed run directory (selection.csv) when one is
         given, or falls back to an explicit configuration for a bare chat demo.
      3. Starts llama-server and **waits for /health to report ready** before
         opening the browser. Opening the UI first is what produces the
         "connection refused" panic mid-demo: a 7B model on CPU can take 30-60 s
         just to load.
      4. Opens bopis.html.
      5. Shuts the server down cleanly on Ctrl+C.

    What this script does NOT do: it does not produce an energy measurement. The
    study host's MX330 exposes no power telemetry, so any energy figure shown
    during this demo is a Mode C resource-allocation estimate. See
    docs/ENERGY_MODES.md before quoting a number from it.

.PARAMETER Run
    A completed run directory containing selection.csv. When supplied, x* is read
    from it and deployed via `bopis serve`, which validates feasibility first.

.PARAMETER Model
    GGUF path. Repeatable as VARIANT=PATH (e.g. Q4_K_M=C:\Models\m.Q4_K_M.gguf).
    A bare path is treated as Q4_K_M.

.PARAMETER LlamaBinary
    Path to llama-server.exe.

.PARAMETER Port
    Port for llama-server. Default 8080, which is what bopis.html expects.

.EXAMPLE
    # Deploy x* from a completed study run.
    .\demo.ps1 -Run runs\20260918T101500Z `
               -LlamaBinary C:\Tools\llama-server.exe `
               -Model Q4_K_M=C:\Models\mistral-7b.Q4_K_M.gguf

.EXAMPLE
    # Chat-only demo with no completed run: launch llama-server directly.
    .\demo.ps1 -LlamaBinary C:\Tools\llama-server.exe `
               -Model C:\Models\qwen2.5-1.5b.Q4_K_M.gguf `
               -GpuLayers 20 -Threads 4 -MaxTokens 256
#>

[CmdletBinding()]
param(
    [string]   $Run,
    [string[]] $Model = @(),
    [string]   $LlamaBinary,
    [int]      $Port = 8080,
    [string]   $LlamaHost = '127.0.0.1',
    [int]      $CtxSize = 2048,

    # Used only for the no-run fallback path.
    [int]      $GpuLayers = 0,
    [int]      $Threads = 4,
    [int]      $Parallel = 1,
    [int]      $MaxTokens = 256,

    # Offload target, e.g. Vulkan0 / Vulkan1 / CUDA0, or 'none' for CPU-only.
    # List what this host exposes with: llama-server.exe --list-devices
    # Measured on the study laptop (Qwen2.5-1.5B Q4_K_M, ctx 2048, 4 threads):
    #   CPU only  (ngl 0)              11.5 tok/s   <-- fastest
    #   Iris Xe   (Vulkan0, ngl 28)     7.6 tok/s
    #   MX330     (Vulkan1, ngl 14)     4.7 tok/s
    #   MX330     (Vulkan1, ngl 28)     3.3 tok/s   <-- slowest
    # Offloading to the MX330 makes generation slower, monotonically in the
    # number of layers moved. Leave this unset for the fastest demo.
    [string]   $Device = '',

    # Seconds to wait for the model to load before giving up.
    [int]      $ReadyTimeoutSec = 180,

    [switch]   $SkipProfile,
    [switch]   $NoBrowser,

    # The instrument bridge (`bopis ui`): measured CPU package energy through
    # LibreHardwareMonitor / Open Hardware Monitor when one is running as
    # administrator with its Remote Web Server on, a per-process Mode C
    # estimate otherwise, and BERTScore for Dolly prompts. -NoBridge opens the
    # bare HTML file instead, as before.
    [int]      $UiPort = 8090,
    [int]      $IdleSeconds = 30,
    [string]   $HwmonUrl = 'http://127.0.0.1:8085/data.json',
    [switch]   $NoBridge,

    # Install/configure/start LibreHardwareMonitor first (one UAC prompt), so
    # energy in the chat is measured rather than estimated.
    [switch]   $StartHwmon
)

$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot
$serverProc = $null

function Write-Step { param([string]$m) Write-Host "`n==> $m" -ForegroundColor Cyan }
function Write-Warn { param([string]$m) Write-Host "    ! $m" -ForegroundColor Yellow }
function Write-Ok   { param([string]$m) Write-Host "    + $m" -ForegroundColor Green }

function Resolve-Python {
    foreach ($candidate in @('py', 'python')) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd) {
            if ($candidate -eq 'py') { return @{ Exe = 'py'; Args = @('-3') } }
            return @{ Exe = 'python'; Args = @() }
        }
    }
    throw 'Neither "py" nor "python" is on PATH.'
}

function Invoke-Bopis {
    param([string[]]$BopisArgs)
    $py = Resolve-Python
    $all = $py.Args + @('-m', 'bopis') + $BopisArgs
    & $py.Exe @all
    return $LASTEXITCODE
}

# --------------------------------------------------------------------------- #
# 0. Sanity checks
# --------------------------------------------------------------------------- #

Write-Step 'Checking prerequisites'

if (-not (Test-Path (Join-Path $repo 'bopis.html'))) {
    throw "bopis.html not found in $repo. Run this script from the repo root."
}
Write-Ok 'bopis.html found'

# Parse -Model into a VARIANT=PATH map.
$models = @{}
foreach ($entry in $Model) {
    if ($entry -match '^([A-Za-z0-9_.:-]+)=(.+)$') {
        $models[$Matches[1]] = $Matches[2]
    } else {
        # Bare path: assume the variant most likely to fit a consumer GPU.
        $models['Q4_K_M'] = $entry
    }
}

foreach ($variant in $models.Keys) {
    if (-not (Test-Path $models[$variant])) {
        throw "GGUF for $variant not found: $($models[$variant])"
    }
    $sizeGiB = [math]::Round((Get-Item $models[$variant]).Length / 1GB, 2)
    Write-Ok "$variant -> $($models[$variant]) ($sizeGiB GiB)"
}

if ($LlamaBinary -and -not (Test-Path $LlamaBinary)) {
    throw "llama-server not found: $LlamaBinary"
}

$canServe = $LlamaBinary -and $models.Count -gt 0
if (-not $canServe) {
    Write-Warn 'No -LlamaBinary and/or -Model supplied.'
    Write-Warn 'Running in UI-only mode: the hardware panel and the static'
    Write-Warn 'pipeline views will work, but the chat box will not answer.'
}

# --------------------------------------------------------------------------- #
# 1. Refresh the hardware profile the UI reads
# --------------------------------------------------------------------------- #

if (-not $SkipProfile) {
    Write-Step 'Refreshing hardware profile for the UI panel'
    Push-Location $repo
    try {
        $null = Invoke-Bopis @(
            'profile', '--model-aware',
            '--ctx-size', "$CtxSize",
            '--write-js', 'bopis_profile.js'
        )
        Write-Ok 'bopis_profile.js updated'
    } catch {
        Write-Warn "Profile refresh failed: $_"
        Write-Warn 'The UI will show stale or unavailable hardware values.'
    } finally {
        Pop-Location
    }

    # The UI's Stage 1 task badge reads its rule table from bopis_rules.js.
    # Regenerate it here so the browser can never run against a stale copy of
    # rules that are authored in Python.
    Push-Location $repo
    try {
        $py = Resolve-Python
        $exportArgs = $py.Args + @(
            '-m', 'bopis.classify', '--write-js', 'bopis_rules.js'
        )
        & $py.Exe @exportArgs | Out-Null
        Write-Ok 'bopis_rules.js updated (Stage 1 rule classifier)'
    } catch {
        Write-Warn "Rule export failed: $_"
        Write-Warn 'The UI will show "Task detection unavailable".'
    } finally {
        Pop-Location
    }

    # The trained classifier is the one the badge prefers (69.7% held-out vs
    # 48.2% for the rules). Only export it if Dolly is present; training takes a
    # few seconds and needs the corpus.
    Push-Location $repo
    try {
        if (Test-Path 'data\databricks-dolly-15k.jsonl') {
            $py = Resolve-Python
            $modelArgs = $py.Args + @(
                '-m', 'bopis.classify_trained',
                '--data-dir', 'data',
                '--write-js', 'bopis_model.js'
            )
            & $py.Exe @modelArgs | Out-Null
            Write-Ok 'bopis_model.js updated (trained classifier, 69.7% held-out)'
        } else {
            Write-Warn 'data\databricks-dolly-15k.jsonl not found.'
            Write-Warn 'Skipping the trained classifier; the badge will use the'
            Write-Warn 'rule classifier (49.7%). Fetch Dolly with:'
            Write-Warn '  python -m bopis dataset --data-dir data'
        }
    } catch {
        Write-Warn "Trained-model export failed: $_"
        Write-Warn 'The badge will fall back to the rule classifier.'
    } finally {
        Pop-Location
    }

    # The profile command reports the feasible space; surface an empty one loudly,
    # because it is the difference between "the tool works" and "nothing can run".
    Push-Location $repo
    try {
        $profileJs = Get-Content 'bopis_profile.js' -Raw -ErrorAction SilentlyContinue
        if ($profileJs -and $profileJs -match '"n_feasible"\s*:\s*0\b') {
            Write-Warn 'FEASIBLE SPACE IS EMPTY (n_feasible: 0).'
            Write-Warn 'Every configuration is rejected by HW-P0 on this host.'
            Write-Warn 'Most common cause: not enough FREE system RAM. Reboot,'
            Write-Warn 'close everything, and re-run. See brief section 7.1.'
        }
    } finally {
        Pop-Location
    }
}

# --------------------------------------------------------------------------- #
# 2. Start llama-server
# --------------------------------------------------------------------------- #

function Wait-ForServer {
    param([string]$BaseUrl, [int]$TimeoutSec)

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $spin = @('|', '/', '-', '\')
    $i = 0
    while ((Get-Date) -lt $deadline) {
        if ($serverProc -and $serverProc.HasExited) {
            Write-Host ''
            throw "llama-server exited with code $($serverProc.ExitCode) before becoming ready."
        }
        try {
            $resp = Invoke-WebRequest -Uri "$BaseUrl/health" -TimeoutSec 3 -UseBasicParsing
            if ($resp.StatusCode -eq 200) { Write-Host ''; return $true }
        } catch {
            # Not up yet, or /health not implemented on this llama.cpp build.
        }
        Write-Host "`r    waiting for model load $($spin[$i % 4])" -NoNewline
        $i++
        Start-Sleep -Milliseconds 700
    }
    Write-Host ''
    return $false
}

$baseUrl = "http://${LlamaHost}:$Port"

if ($canServe) {
    Write-Step 'Starting llama-server'

    Push-Location $repo
    try {
        if ($Run) {
            if (-not (Test-Path (Join-Path $Run 'selection.csv'))) {
                throw "$Run does not contain selection.csv -- not a completed run."
            }
            Write-Ok "Deploying x* from $Run"

            $py = Resolve-Python
            $serveArgs = $py.Args + @(
                '-m', 'bopis', 'serve',
                '--run', $Run,
                '--llama-binary', $LlamaBinary,
                '--host', $LlamaHost,
                '--port', "$Port",
                '--ctx-size', "$CtxSize"
            )
            foreach ($variant in $models.Keys) {
                $serveArgs += @('--model', "$variant=$($models[$variant])")
            }
            $serverProc = Start-Process -FilePath $py.Exe -ArgumentList $serveArgs `
                -NoNewWindow -PassThru
        } else {
            # No completed run: launch llama-server directly. This demonstrates
            # deployment mechanics only -- the configuration is whatever was
            # passed on the command line, NOT an optimizer result. Do not
            # describe it to a panel as x*.
            Write-Warn 'No -Run given: launching a MANUAL configuration.'
            Write-Warn 'This is not x* and carries no optimizer claim.'

            $variant = @($models.Keys)[0]
            $llamaArgs = @(
                '--model', $models[$variant],
                '--host', $LlamaHost,
                '--port', "$Port",
                '--ctx-size', "$CtxSize",
                '--n-gpu-layers', "$GpuLayers",
                '--threads', "$Threads",
                '--parallel', "$Parallel",
                '--n-predict', "$MaxTokens"
            )
            if ($Device) { $llamaArgs += @('--device', $Device) }
            Write-Host "    $LlamaBinary $($llamaArgs -join ' ')" -ForegroundColor DarkGray
            $serverProc = Start-Process -FilePath $LlamaBinary -ArgumentList $llamaArgs `
                -NoNewWindow -PassThru
        }
    } finally {
        Pop-Location
    }

    Write-Ok "Server process started (PID $($serverProc.Id))"
    Write-Host "    Model load on CPU can take 30-60 s for a 7B model." -ForegroundColor DarkGray

    if (-not (Wait-ForServer -BaseUrl $baseUrl -TimeoutSec $ReadyTimeoutSec)) {
        Write-Warn "Server did not report ready within $ReadyTimeoutSec s."
        Write-Warn 'Opening the UI anyway; the first message may fail. If it does,'
        Write-Warn 'wait and resend rather than restarting.'
    } else {
        Write-Ok "Server ready at $baseUrl"
    }
}

# --------------------------------------------------------------------------- #
# 3. Open the UI
# --------------------------------------------------------------------------- #

$bridgeProc = $null
$uiUrl = $null
if ($StartHwmon) {
    & (Join-Path $repo 'tools\start-hwmon.ps1') -Install
}

if ($canServe -and -not $NoBridge) {
    Write-Step 'Starting the instrument bridge (bopis ui)'
    Write-Host "    Idle calibration takes $IdleSeconds s -- leave the machine alone." -ForegroundColor DarkGray
    $py = Resolve-Python
    $uiArgs = $py.Args + @(
        '-m', 'bopis', 'ui',
        '--port', "$UiPort",
        '--llama-url', $baseUrl,
        '--idle-seconds', "$IdleSeconds",
        '--hwmon-url', $HwmonUrl,
        # Lets the chat run the default / random-search / BOPIS comparison on a
        # Dolly prompt; model files are found in .\models automatically.
        '--llama-binary', $LlamaBinary,
        '--models-dir', (Join-Path $repo 'models')
    )
    if ($serverProc -and -not $Run) { $uiArgs += @('--llama-pid', "$($serverProc.Id)") }
    $bridgeProc = Start-Process -FilePath $py.Exe -ArgumentList $uiArgs `
        -WorkingDirectory $repo -NoNewWindow -PassThru

    $deadline = (Get-Date).AddSeconds($IdleSeconds + 60)
    while ((Get-Date) -lt $deadline -and -not $bridgeProc.HasExited) {
        try {
            $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$UiPort/api/status" -TimeoutSec 3 -UseBasicParsing
            if ($resp.StatusCode -eq 200) { $uiUrl = "http://127.0.0.1:$UiPort/"; break }
        } catch { }
        Start-Sleep -Milliseconds 700
    }
    if ($uiUrl) {
        Write-Ok "Bridge ready at $uiUrl"
    } else {
        Write-Warn 'Bridge did not come up; falling back to the bare HTML file.'
        Write-Warn 'The chat will show an in-browser Mode C estimate and no BERTScore.'
    }
}

if (-not $NoBrowser) {
    Write-Step 'Opening the UI'
    if ($uiUrl) { Start-Process $uiUrl } else { Start-Process (Join-Path $repo 'bopis.html') }
    Write-Ok 'UI opened'
}

# --------------------------------------------------------------------------- #
# 4. Hold until Ctrl+C, then clean up
# --------------------------------------------------------------------------- #

if ($canServe) {
    Write-Host ''
    Write-Host '  Demo is live. Chat endpoint:' -ForegroundColor Green
    Write-Host "    POST $baseUrl/v1/chat/completions" -ForegroundColor DarkGray
    Write-Host '  Press Ctrl+C to stop the server and exit.' -ForegroundColor Green
    Write-Host ''

    try {
        while (-not $serverProc.HasExited) { Start-Sleep -Seconds 1 }
        Write-Warn "llama-server exited on its own (code $($serverProc.ExitCode))."
    } finally {
        if ($bridgeProc -and -not $bridgeProc.HasExited) {
            try { Stop-Process -Id $bridgeProc.Id -Force -ErrorAction Stop } catch { }
        }
        if ($serverProc -and -not $serverProc.HasExited) {
            Write-Step 'Stopping llama-server'
            try {
                Stop-Process -Id $serverProc.Id -Force -ErrorAction Stop
                Write-Ok 'Stopped'
            } catch {
                Write-Warn "Could not stop PID $($serverProc.Id): $_"
            }
        }
    }
} else {
    Write-Host ''
    Write-Host '  UI-only mode. To get a live chat box, re-run with:' -ForegroundColor Yellow
    Write-Host '    -LlamaBinary <path to llama-server.exe> -Model <path to .gguf>' -ForegroundColor DarkGray
    Write-Host ''
}
