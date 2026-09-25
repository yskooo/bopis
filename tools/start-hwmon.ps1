<#
.SYNOPSIS
    Install (if needed), configure, and start LibreHardwareMonitor so BOPIS can
    measure CPU package energy -- with one UAC click and no menu clicking.

.DESCRIPTION
    BOPIS's measured energy mode (Mode D, `--energy-mode cpu-rapl`, and the
    `bopis ui` chat) reads Intel RAPL package power from LibreHardwareMonitor's
    built-in web server at http://127.0.0.1:8085/data.json. Doing that by hand
    means: download, run as administrator, Options -> Remote Web Server -> Run.
    This script does all of it:

      1. Returns immediately if the sensor feed is already answering.
      2. Finds LibreHardwareMonitor.exe, or installs it with winget (-Install).
      3. Writes its settings file (LibreHardwareMonitor.config, next to the exe)
         so it starts with the web server ON, on port 8085, no authentication,
         minimized to the tray, and refreshing every 250 ms instead of 1 s
         (a 4x narrower resolution band on every energy figure).
      4. Launches it elevated. Windows shows ONE administrator prompt: RAPL is
         read through a kernel driver, so this cannot be avoided -- no tool,
         CLI or otherwise, reads RAPL on Windows without admin rights.
      5. Waits until the feed answers AND lists a "CPU Package" power sensor.

    Setting names come from LibreHardwareMonitor's own source
    (LibreHardwareMonitor.Windows.Forms/UI/MainForm.cs): runWebServerMenuItem,
    listenerPort, authenticationEnabled, startMinMenuItem, minTrayMenuItem,
    updateIntervalMenuItem (0 = 250 ms).

.PARAMETER Install
    Install LibreHardwareMonitor with winget if it is not found.

.PARAMETER RefreshMs
    Sensor refresh: 250, 500 or 1000 ms. Default 250.

.EXAMPLE
    .\tools\start-hwmon.ps1 -Install
#>

[CmdletBinding()]
param(
    [switch] $Install,
    [ValidateSet(250, 500, 1000)] [int] $RefreshMs = 250,
    [int]    $Port = 8085,
    [int]    $TimeoutSec = 60
)

$ErrorActionPreference = 'Stop'
$feed = "http://127.0.0.1:$Port/data.json"

function Write-Ok   { param([string]$m) Write-Host "    + $m" -ForegroundColor Green }
function Write-Warn { param([string]$m) Write-Host "    ! $m" -ForegroundColor Yellow }

function Test-Feed {
    try {
        $resp = Invoke-WebRequest -Uri $feed -TimeoutSec 2 -UseBasicParsing
        return ($resp.StatusCode -eq 200 -and $resp.Content -match 'Package')
    } catch { return $false }
}

Write-Host "`n==> LibreHardwareMonitor (CPU package power for BOPIS)" -ForegroundColor Cyan

if (Test-Feed) {
    Write-Ok "Sensor feed already live at $feed"
    return
}

# --------------------------------------------------------------------------- #
# 1. Find (or install) the executable
# --------------------------------------------------------------------------- #

function Find-Lhm {
    $candidates = @(
        (Join-Path $PSScriptRoot 'LibreHardwareMonitor\LibreHardwareMonitor.exe'),
        "$env:ProgramFiles\LibreHardwareMonitor\LibreHardwareMonitor.exe"
    )
    $wingetRoot = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages'
    if (Test-Path $wingetRoot) {
        $candidates += Get-ChildItem $wingetRoot -Directory -Filter 'LibreHardwareMonitor*' -ErrorAction SilentlyContinue |
            ForEach-Object { Get-ChildItem $_.FullName -Recurse -Filter 'LibreHardwareMonitor.exe' -ErrorAction SilentlyContinue } |
            ForEach-Object { $_.FullName }
    }
    $running = Get-Process -Name LibreHardwareMonitor -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($running -and $running.Path) { $candidates = @($running.Path) + $candidates }
    return $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}

$exe = Find-Lhm
if (-not $exe) {
    if (-not $Install) {
        Write-Warn 'LibreHardwareMonitor not found. Re-run with -Install, or unzip it'
        Write-Warn "into $(Join-Path $PSScriptRoot 'LibreHardwareMonitor')."
        exit 1
    }
    Write-Host '    Installing with winget...' -ForegroundColor DarkGray
    winget install --id LibreHardwareMonitor.LibreHardwareMonitor -e `
        --accept-source-agreements --accept-package-agreements | Out-Host
    $exe = Find-Lhm
    if (-not $exe) { throw 'winget finished but LibreHardwareMonitor.exe was not found.' }
}
Write-Ok "Found $exe"

# --------------------------------------------------------------------------- #
# 2. Write the settings it will start with
# --------------------------------------------------------------------------- #

# An instance that is already running would overwrite the file on exit, and
# the web server setting only takes effect at start-up, so restart it.
$existing = Get-Process -Name LibreHardwareMonitor -ErrorAction SilentlyContinue
if ($existing) {
    Write-Warn 'LibreHardwareMonitor is running without the web server; restarting it.'
    try { $existing | Stop-Process -Force -ErrorAction Stop } catch {
        Write-Warn 'Could not stop it (it runs as administrator). Close it from the tray and re-run.'
        exit 1
    }
    Start-Sleep -Milliseconds 800
}

$configPath = [System.IO.Path]::ChangeExtension($exe, '.config')
$doc = New-Object System.Xml.XmlDocument
if (Test-Path $configPath) {
    $doc.Load($configPath)
} else {
    $doc.AppendChild($doc.CreateElement('configuration')) | Out-Null
}
$appSettings = $doc.SelectSingleNode('/configuration/appSettings')
if (-not $appSettings) {
    $appSettings = $doc.DocumentElement.AppendChild($doc.CreateElement('appSettings'))
}

$interval = @{ 250 = '0'; 500 = '1'; 1000 = '2' }[$RefreshMs]
$wanted = [ordered]@{
    'runWebServerMenuItem'   = 'true'
    'listenerPort'           = "$Port"
    'authenticationEnabled'  = 'false'
    'startMinMenuItem'       = 'true'
    'minTrayMenuItem'        = 'true'
    'updateIntervalMenuItem' = $interval
    'cpuMenuItem'            = 'true'
}
foreach ($key in $wanted.Keys) {
    $node = $appSettings.SelectSingleNode("add[@key='$key']")
    if (-not $node) {
        $node = $appSettings.AppendChild($doc.CreateElement('add'))
        $node.SetAttribute('key', $key)
    }
    $node.SetAttribute('value', $wanted[$key])
}
$doc.Save($configPath)
Write-Ok "Settings written: web server on :$Port, refresh $RefreshMs ms, start in tray"

# --------------------------------------------------------------------------- #
# 3. Start elevated and wait for the feed
# --------------------------------------------------------------------------- #

Write-Host '    Starting as administrator -- approve the Windows prompt.' -ForegroundColor DarkGray
try {
    Start-Process -FilePath $exe -Verb RunAs -WorkingDirectory (Split-Path $exe)
} catch {
    Write-Warn 'The administrator prompt was declined. RAPL needs admin rights; re-run and approve it.'
    exit 1
}

$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $deadline) {
    if (Test-Feed) {
        Write-Ok "Sensor feed live at $feed"
        Write-Host '    Check it with: python -m bopis profile' -ForegroundColor DarkGray
        return
    }
    Start-Sleep -Milliseconds 700
}
Write-Warn "No 'CPU Package' sensor at $feed after $TimeoutSec s."
Write-Warn 'If LibreHardwareMonitor opened but lists no Powers for the CPU, it was not'
Write-Warn 'started as administrator, or Windows Defender blocked its driver.'
exit 1
