# ensure_backend.ps1 — check BOPIS backend, restart whatever is dead.
#
#   powershell -ExecutionPolicy Bypass -File .\ensure_backend.ps1
#
# Safe to run any time. It never restarts something that is already healthy.

$ErrorActionPreference = 'SilentlyContinue'
$repo   = Split-Path -Parent $MyInvocation.MyCommand.Path
$llama  = Join-Path $repo 'tools\cpu\llama-server.exe'
$model  = Join-Path $repo 'models\qwen2.5-1.5b-instruct-q4_k_m.gguf'
$llamaUrl = 'http://127.0.0.1:8080'
$bridgeUrl = 'http://127.0.0.1:8090'
$logDir = Join-Path $repo 'logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

function Port-Open([int]$p) {
    [bool](Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction SilentlyContinue)
}
function Wait-Url([string]$url, [int]$sec) {
    for ($i = 0; $i -lt $sec; $i++) {
        try {
            $r = Invoke-WebRequest $url -TimeoutSec 2 -UseBasicParsing
            if ($r.StatusCode -eq 200) { return $true }
        } catch { }
        Start-Sleep -Seconds 1
    }
    return $false
}
function Report([string]$name, [bool]$ok, [string]$detail) {
    $tag = if ($ok) { 'OK  ' } else { 'FAIL' }
    Write-Host ("[{0}] {1,-10} {2}" -f $tag, $name, $detail)
}

# ---- 1. llama-server -------------------------------------------------------
$llamaOk = $false
if (Port-Open 8080) {
    try {
        $h = (Invoke-WebRequest "$llamaUrl/health" -TimeoutSec 3 -UseBasicParsing).Content
        $llamaOk = ($h -match '"status"\s*:\s*"ok"')
    } catch { $llamaOk = $false }
}

if (-not $llamaOk) {
    if (-not (Test-Path $llama)) {
        Write-Host "[FAIL] llama     missing $llama"
        Write-Host "       fix: hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF --include `"*q4_k_m*.gguf`" --local-dir `"$repo\models`""
    } elseif (-not (Test-Path $model)) {
        Write-Host "[FAIL] llama     missing $model"
        Write-Host "       fix: hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF --include `"*q4_k_m*.gguf`" --local-dir `"$repo\models`""
    } else {
        if (Port-Open 8080) {
            # a wedged listener: kill it before relaunching
            Get-NetTCPConnection -State Listen -LocalPort 8080 -ErrorAction SilentlyContinue |
                ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
            Start-Sleep -Seconds 2
        }
        $llamaArgs = @('--model', $model,
                       '--host', '127.0.0.1', '--port', '8080',
                       '--ctx-size', '2048', '--n-gpu-layers', '0',
                       '--threads', '4', '--parallel', '1')
        Start-Process -FilePath $llama -ArgumentList $llamaArgs -WorkingDirectory $repo `
            -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir 'llama-server.log') `
            -RedirectStandardError  (Join-Path $logDir 'llama-server.err') | Out-Null
        $llamaOk = Wait-Url "$llamaUrl/health" 90
    }
}
Report 'llama' $llamaOk "$llamaUrl"

# ---- 2. bridge (python -m bopis ui) ---------------------------------------
$bridgeOk = $false
if (Port-Open 8090) {
    try {
        $s = (Invoke-WebRequest "$bridgeUrl/api/status" -TimeoutSec 3 -UseBasicParsing).Content | ConvertFrom-Json
        $bridgeOk = $true
    } catch { $bridgeOk = $false }
}
if (-not $bridgeOk) {
    if (Port-Open 8090) {
        Get-NetTCPConnection -State Listen -LocalPort 8090 -ErrorAction SilentlyContinue |
            ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
        Start-Sleep -Seconds 2
    }
    Start-Process -FilePath 'python' -ArgumentList '-m','bopis','ui' -WorkingDirectory $repo `
        -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir 'bridge.log') `
        -RedirectStandardError  (Join-Path $logDir 'bridge.err') | Out-Null
    $bridgeOk = Wait-Url "$bridgeUrl/api/status" 45
}
Report 'bridge' $bridgeOk "$bridgeUrl"

# ---- 3. end-to-end smoke test ---------------------------------------------
$chatOk = $false
if ($bridgeOk) {
    try {
        $body = '{"messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":8}'
        $r = Invoke-WebRequest "$bridgeUrl/api/chat" -Method POST -ContentType 'application/json' `
                -Body $body -TimeoutSec 120 -UseBasicParsing
        $j = $r.Content | ConvertFrom-Json
        $chatOk = ($r.StatusCode -eq 200) -and $j.completion.choices[0].message.content
    } catch { $chatOk = $false }
}
Report 'chat' $chatOk 'POST /api/chat through the bridge'

# ---- 4. advisory (does not fail the run) ----------------------------------
try {
    $s = (Invoke-WebRequest "$bridgeUrl/api/status" -TimeoutSec 3 -UseBasicParsing).Content | ConvertFrom-Json
    if (-not $s.energy.measured) {
        Write-Host "[note] energy    ESTIMATED, not measured - no LibreHardwareMonitor feed on :8085."
        Write-Host "       Never report these joules as measured (MX330 has no power telemetry)."
    }
} catch { }

Write-Host ""
if ($llamaOk -and $bridgeOk -and $chatOk) {
    Write-Host "ALL GREEN - open http://127.0.0.1:8090/"
    exit 0
} else {
    Write-Host "NOT GREEN - see FAIL lines above, and logs\*.err"
    exit 1
}
