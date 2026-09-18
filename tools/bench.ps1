# Compare decode speed across the three execution targets on this host.
# Uses Stop-Process, not pkill: pkill from Git Bash does not reap Windows
# processes, which silently left four orphaned servers competing for RAM and
# invalidated an earlier run of this benchmark.
$ErrorActionPreference = 'Stop'
$body = @{
    model = 'local'
    messages = @(@{ role = 'user'
                    content = 'Explain in three sentences why quantization reduces LLM inference energy.' })
    max_tokens = 64; temperature = 0; stream = $false
} | ConvertTo-Json -Depth 5

function Stop-Servers {
    Get-Process -Name 'llama-server' -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.Id -Force }
    Start-Sleep -Seconds 3
}

function Measure-Target {
    param([string]$Label, [string[]]$ExtraArgs)
    Stop-Servers
    $free = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB, 2)

    $serverArgs = @('--model', '..\models\qwen2.5-1.5b-instruct-q4_k_m.gguf',
                    '--ctx-size', '2048', '--threads', '4', '--parallel', '1',
                    '--host', '127.0.0.1', '--port', '8080') + $ExtraArgs
    $proc = Start-Process -FilePath '.\vulkan\llama-server.exe' -ArgumentList $serverArgs `
                          -NoNewWindow -PassThru -RedirectStandardOutput "log-$Label.txt" `
                          -RedirectStandardError "err-$Label.txt"
    $ready = $false
    foreach ($i in 1..90) {
        try {
            if ((Invoke-WebRequest "http://127.0.0.1:8080/health" -TimeoutSec 2 -UseBasicParsing).StatusCode -eq 200) {
                $ready = $true; break
            }
        } catch { Start-Sleep -Milliseconds 800 }
    }
    if (-not $ready) { "{0,-20} SERVER DID NOT START" -f $Label; Stop-Servers; return }

    $r = Invoke-RestMethod "http://127.0.0.1:8080/v1/chat/completions" -Method Post `
            -ContentType 'application/json' -Body $body -TimeoutSec 300
    "{0,-20} {1,4} tok  {2,7:N2}s prefill  {3,7:N2} tok/s decode   (free RAM at start {4} GiB)" -f `
        $Label, $r.usage.completion_tokens, ($r.timings.prompt_ms/1000), `
        $r.timings.predicted_per_second, $free
    Stop-Servers
}

"model: Qwen2.5-1.5B-Instruct Q4_K_M | ctx 2048 | threads 4"
"-" * 100
Measure-Target 'MX330 full (28L)'  @('--device','Vulkan1','--n-gpu-layers','28')
Measure-Target 'IrisXe full (28L)' @('--device','Vulkan0','--n-gpu-layers','28')
Measure-Target 'CPU only (0L)'     @('--n-gpu-layers','0')
Measure-Target 'MX330 half (14L)'  @('--device','Vulkan1','--n-gpu-layers','14')
