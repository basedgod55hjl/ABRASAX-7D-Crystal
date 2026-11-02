# ============================================================================
# ABRASAX — Unified Xenoliving System
# PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
# CREATOR: Sir Charles Spikes (BASEDGOD)
# ============================================================================

# Run everything declared in abrasax_folder_monitor.yaml (startup + LLM stat/smoke).
# Let step 3 (cuda_load_balancer) finish: PyTorch import can take 60–120s; do not cancel mid-run.
# Optional: ABRASAX_LM_REST_MCP_CHAT=1 runs step 5d POST /api/v1/chat with Hugging Face ephemeral_mcp (needs outbound HTTPS).
# Usage: cd ...\gpu_compute\backends ; .\start_all.ps1
# Optional: $env:ABRASAX_START_GEMMA_NONSTOP='1' spawns start_all_gemma_agents.ps1 minimized (heavy; LM Studio must be up).
$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
# If script was copied elsewhere, fall back to repo backends (same machine layout as ABRASAX checkout)
if (-not (Test-Path -LiteralPath (Join-Path $root 'cbm_system\performance_config.json'))) {
    $tryBackends = Join-Path $env:USERPROFILE 'ABRASAX\gpu_compute\backends'
    if (Test-Path -LiteralPath (Join-Path $tryBackends 'cbm_system\performance_config.json')) {
        $root = $tryBackends
    }
}
$abrasaxRoot = Split-Path (Split-Path $root)
# Wire LLM smoke tests to cbm_system/performance_config.json (same as Gemma launchers)
$perfJsonPath = Join-Path $root 'cbm_system\performance_config.json'
$llmBase = 'http://169.254.83.107:1234/v1'
$llmChatModel = 'google/gemma-4-e4b-uncensored-hauhaucs-aggressive'
$llmEmbedModel = 'text-embedding-nomic-embed-text-v1.5'
$embedBatchLines = 8
$doGemmaAdvisor = $true
if (Test-Path $perfJsonPath) {
    try {
        $pc = Get-Content $perfJsonPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($pc.llm_controls.base_url) { $llmBase = ($pc.llm_controls.base_url.TrimEnd('/')) }
        if ($pc.llm_controls.model) { $llmChatModel = [string]$pc.llm_controls.model }
        if ($pc.llm_controls.embedding_model) { $llmEmbedModel = [string]$pc.llm_controls.embedding_model }
        if ($pc.resource_monitor) {
            $rm = $pc.resource_monitor
            if ($rm.embedding_batch_lines) { $embedBatchLines = [int]$rm.embedding_batch_lines }
            if ($null -ne $rm.gemma_advisor_after_embed) { $doGemmaAdvisor = [bool]$rm.gemma_advisor_after_embed }
        }
    }
    catch {
        Write-Host 'WARN: could not parse performance_config.json; using defaults.' -ForegroundColor Yellow
    }
}
# LM Studio bases: link-local (169.254.*) often stalls on POST under load — same PC should hit 127.0.0.1 first.
if ($llmBase -match '169\.254\.') {
    $llmBases = @('http://127.0.0.1:1234/v1', 'http://localhost:1234/v1', $llmBase)
    Write-Host 'LLM API order: localhost first (config uses link-local; faster for chat/embed/advisor on this machine).' -ForegroundColor DarkGray
} else {
    $llmBases = @($llmBase, 'http://127.0.0.1:1234/v1', 'http://localhost:1234/v1')
}

# LM Studio OpenAI-compat: optional API key (Bearer) — same as Cursor/OpenAI clients when server auth is enabled
$script:AbrasaxLmHdr = @{}
$script:AbrasaxLmKey = $null
foreach ($n in @('OPENAI_API_KEY', 'LM_STUDIO_API_KEY', 'LM_API_TOKEN')) {
    $script:AbrasaxLmKey = [Environment]::GetEnvironmentVariable($n, 'Process')
    if (-not $script:AbrasaxLmKey) { $script:AbrasaxLmKey = [Environment]::GetEnvironmentVariable($n, 'User') }
    if ($script:AbrasaxLmKey) { break }
}
if (-not $script:AbrasaxLmKey) {
    $dotEnvLm = Join-Path $abrasaxRoot '.env'
    if (Test-Path -LiteralPath $dotEnvLm) {
        foreach ($line in Get-Content -LiteralPath $dotEnvLm -Encoding UTF8) {
            if ($line -match '^\s*#' -or -not ($line.Trim())) { continue }
            if ($line -match '^\s*(OPENAI_API_KEY|LM_STUDIO_API_KEY|LM_API_TOKEN)\s*=\s*(.*)$') {
                $script:AbrasaxLmKey = $matches[2].Trim().Trim('"').Trim("'")
                break
            }
        }
    }
}
if ($script:AbrasaxLmKey) {
    $script:AbrasaxLmHdr = @{ Authorization = ('Bearer ' + $script:AbrasaxLmKey.Trim()) }
    Write-Host 'LM Studio API: Authorization Bearer enabled (env or .env).' -ForegroundColor DarkGray
}

function Invoke-AbrasaxLmRest {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [ValidateSet('Get', 'Post')][string]$Method = 'Get',
        [string]$Body = $null,
        [string]$ContentType = 'application/json; charset=utf-8',
        [int]$TimeoutSec = 120
    )
    $p = @{ Uri = $Uri; Method = $Method; TimeoutSec = $TimeoutSec }
    if ($script:AbrasaxLmHdr.Count -gt 0) { $p['Headers'] = $script:AbrasaxLmHdr }
    if ($Method -eq 'Post' -and $null -ne $Body) {
        $p['Body'] = $Body
        $p['ContentType'] = $ContentType
    }
    Invoke-RestMethod @p
}

Write-Host '=== 1) Verify startup.config_paths ===' -ForegroundColor Cyan
$configs = @(
    "$abrasaxRoot\README.md",
    "$abrasaxRoot\src\services",
    "$abrasaxRoot\src\services\agent_coordinator.py",
    "$abrasaxRoot\src\services\data_ingestion.py",
    "$abrasaxRoot\src\services\integration_layer\README.md",
    "$abrasaxRoot\server.js",
    "$root\cbm_system\performance_config.json",
    "$root\abrasax_folder_monitor.yaml",
    "$root\10_GPU_RUST\GEMMA_MODEL_INFO.md",
    "$root\02_NVIDIA_DEBUG\nvidia-smi.json",
    "$root\VIDEO_ENHANCE\vlc_enhanced_preset.json",
    "$root\11_NONCUDA_FULLSTACK\noncuda_fullstack\config\stack.yaml"
)
foreach ($p in $configs) {
    if (Test-Path $p) { Write-Host "OK  $p" }
    else { Write-Host "MISSING  $p" -ForegroundColor Yellow }
}

Write-Host "`n=== 2) gpu_launch.bat ===" -ForegroundColor Cyan
cmd /c "`"$root\10_GPU_RUST\gpu_launch.bat`""

Write-Host "`n=== 3) cuda_load_balancer.py ===" -ForegroundColor Cyan
python "$root\03_CUDA_TOOLS\cuda_load_balancer.py"
if ($LASTEXITCODE -ne 0) { Write-Host "cuda_load_balancer exit: $LASTEXITCODE" -ForegroundColor Yellow }

Write-Host "`n=== 3b) Enhanced resource monitor (CPU, RAM, iGPU adapters, NVIDIA VRAM) ===" -ForegroundColor Cyan
$cpuPct = $null
$ramMb = $null
try {
    $cpuPct = [math]::Round((Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 1).CounterSamples[0].CookedValue, 1)
    $ramMb = [math]::Round((Get-Counter '\Memory\Available MBytes' -SampleInterval 1 -MaxSamples 1).CounterSamples[0].CookedValue, 0)
    Write-Host "CPU (all cores, sample): ${cpuPct}%"
    Write-Host "System RAM available: ${ramMb} MB"
} catch {
    Write-Host "WARN: performance counters: $($_.Exception.Message)" -ForegroundColor Yellow
}
$nvLine = $null
try {
    $nvRaw = & nvidia-smi --query-gpu=name,memory.used,memory.total,memory.free,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader 2>$null
    if ($nvRaw) {
        $nvLine = $nvRaw.Trim()
        Write-Host "NVIDIA: $nvLine"
    }
} catch { }
$igpuLines = @()
try {
    Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | ForEach-Object {
        $n = $_.Name
        if ($n -notmatch 'NVIDIA') {
            $ar = if ($_.AdapterRAM -and $_.AdapterRAM -gt 0) { [math]::Round($_.AdapterRAM / 1MB, 0).ToString() + ' MB' } else { 'shared/unspecified' }
            $igpuLines += "${n} (adapter RAM: $ar)"
            Write-Host "Display/iGPU adapter: $n ($ar)"
        }
    }
} catch { }

$snapPath = Join-Path $env:TEMP 'ABRASAX_resource_snapshot.json'
$resourceSnapshot = [ordered]@{
    timestamp   = (Get-Date).ToString('s')
    cpu_percent = $cpuPct
    ram_avail_mb = $ramMb
    nvidia_csv  = $nvLine
    igpu_adapters = $igpuLines
    offload_apps = @()
    high_perf_apps = @()
}
if (Test-Path $perfJsonPath) {
    try {
        $pc2 = Get-Content $perfJsonPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($pc2.gpu_task_mapping.offload_apps) { $resourceSnapshot.offload_apps = @($pc2.gpu_task_mapping.offload_apps) }
        if ($pc2.gpu_task_mapping.high_perf_apps) { $resourceSnapshot.high_perf_apps = @($pc2.gpu_task_mapping.high_perf_apps) }
    } catch { }
}
$resourceSnapshot | ConvertTo-Json -Depth 6 -Compress | Set-Content -Path $snapPath -Encoding utf8
Write-Host "Snapshot: $snapPath" -ForegroundColor DarkGray

Write-Host "`n=== 4) LLM stat_all (GET /v1/models) ===" -ForegroundColor Cyan
Write-Host "Using base: $llmBase (from performance_config.json); fallback 127.0.0.1 if slow" -ForegroundColor DarkGray
$m = $null
foreach ($baseTry in $llmBases) {
    try {
        $m = Invoke-AbrasaxLmRest -Uri ($baseTry.TrimEnd('/') + '/models') -Method Get -TimeoutSec 90
        Write-Host "OK via $baseTry" -ForegroundColor DarkGray
        $m | ConvertTo-Json -Depth 6
        break
    } catch {
        Write-Host "WARN $baseTry : $($_.Exception.Message)" -ForegroundColor Yellow
    }
}
if (-not $m) { Write-Host 'FAIL: no /v1/models response from configured or localhost base.' -ForegroundColor Red }

Write-Host "`n=== 5) LLM smoke (POST /v1/chat/completions) ===" -ForegroundColor Cyan
Write-Host "Using model: $llmChatModel" -ForegroundColor DarkGray
$h = $null
$body = @{
    model      = $llmChatModel
    messages   = @(@{ role = 'user'; content = 'ping' })
    max_tokens = 8
    temperature = 0
} | ConvertTo-Json -Depth 6 -Compress
foreach ($b in $llmBases) {
    try {
        $h = Invoke-AbrasaxLmRest -Uri ($b.TrimEnd('/') + '/chat/completions') -Method Post -Body $body -ContentType 'application/json; charset=utf-8' -TimeoutSec 120
        Write-Host "OK chat via $b" -ForegroundColor DarkGray
        $h | ConvertTo-Json -Depth 8
        break
    } catch {
        Write-Host "WARN chat $b : $($_.Exception.Message)" -ForegroundColor Yellow
    }
}
if (-not $h) { Write-Host 'FAIL: chat smoke' -ForegroundColor Red }

$env:ABRASAX_EMBEDDING_MODEL = $llmEmbedModel
$emb = $null
$embMs = 0
$vecCount = 0
Write-Host "`n=== 5b) Nomic full batch embeddings (POST /v1/embeddings, no smoke) ===" -ForegroundColor Cyan
Write-Host "Using embedding model: $llmEmbedModel | batch lines: $embedBatchLines" -ForegroundColor DarkGray
try {
    $embedInputs = [System.Collections.Generic.List[string]]::new()
    for ($ei = 1; $ei -le $embedBatchLines; $ei++) {
        $embedInputs.Add("ABRASAX full embed line $ei : monitor CPU iGPU NVIDIA VRAM offload performance_config gpu_task_mapping")
    }
    $embBody = @{
        model = $llmEmbedModel
        input = $embedInputs.ToArray()
    } | ConvertTo-Json -Depth 8 -Compress
    $swEmb = [System.Diagnostics.Stopwatch]::StartNew()
    $emb = $null
    foreach ($b in $llmBases) {
        try {
            $emb = Invoke-AbrasaxLmRest -Uri ($b.TrimEnd('/') + '/embeddings') -Method Post -Body $embBody -ContentType 'application/json; charset=utf-8' -TimeoutSec 300
            Write-Host "OK embeddings via $b" -ForegroundColor DarkGray
            break
        } catch {
            Write-Host "WARN embeddings $b : $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
    $swEmb.Stop()
    $embMs = [int]$swEmb.ElapsedMilliseconds
    if (-not $emb) { throw 'Nomic embeddings failed on all bases' }
    $vecCount = $emb.data.Count
    $dims0 = $emb.data[0].embedding.Count
    $sample = $emb.data[0].embedding[0..3] -join ', '
    Write-Host "OK: vectors=$vecCount dims=$dims0 elapsed_ms=$embMs sample[0..3]=$sample" -ForegroundColor Green
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
}

if ($doGemmaAdvisor -and $emb -and $vecCount -gt 0) {
    Write-Host "`n=== 5c) Gemma resource advisor (LLM suggests CPU / iGPU / NVIDIA / offload tweaks) ===" -ForegroundColor Cyan
    try {
        $snapBrief = '{}'
        if (Test-Path -LiteralPath $snapPath) {
            $so = Get-Content -Path $snapPath -Raw -Encoding UTF8 -ErrorAction Stop | ConvertFrom-Json
            $briefObj = [ordered]@{
                cpu_percent    = $so.cpu_percent
                ram_avail_mb   = $so.ram_avail_mb
                nvidia_csv     = $so.nvidia_csv
                igpu_adapters  = $so.igpu_adapters
                offload_apps   = $so.offload_apps
                high_perf_apps = $so.high_perf_apps
            }
            $snapBrief = ($briefObj | ConvertTo-Json -Compress -Depth 6)
        }
        $userBlock = "Resource snapshot (compact JSON): $snapBrief`n`nNomic embedding batch: model=$llmEmbedModel vectors=$vecCount dims=$($emb.data[0].embedding.Count) latency_ms=$embMs.`nReply with markdown bullets: CPU, RAM, AMD iGPU vs NVIDIA VRAM, offload_apps, LM Studio GPU offload."
        if ($userBlock.Length -gt 6000) { $userBlock = $userBlock.Substring(0, 6000) + '...' }
        $adv = @{
            model       = $llmChatModel
            messages    = @(
                @{ role = 'system'; content = 'Terse Windows + LM Studio + ABRASAX tuner. Put the FINAL markdown bullets (max 12 lines) in assistant content. Topics: CPU, RAM, AMD iGPU vs NVIDIA VRAM, gpu_task_mapping.offload_apps, LM Studio GPU layers.' }
                @{ role = 'user'; content = $userBlock }
            )
            max_tokens  = [int]600
            temperature = 0.35
        }
        $advJson = $adv | ConvertTo-Json -Depth 12 -Compress
        $advResp = $null
        foreach ($b in $llmBases) {
            try {
                $advResp = Invoke-AbrasaxLmRest -Uri ($b.TrimEnd('/') + '/chat/completions') -Method Post -Body $advJson -ContentType 'application/json; charset=utf-8' -TimeoutSec 240
                Write-Host "OK advisor via $b" -ForegroundColor DarkGray
                break
            } catch {
                Write-Host "WARN advisor $b : $($_.Exception.Message)" -ForegroundColor Yellow
            }
        }
        if (-not $advResp) { throw 'No advisor response from any LM Studio base' }
        $txt = $advResp.choices[0].message.content
        if (-not $txt -or -not $txt.Trim()) {
            $txt = $advResp.choices[0].message.reasoning_content
        }
        Write-Host "--- Gemma advisor ---"
        Write-Host $txt
        $advLog = Join-Path $env:TEMP 'ABRASAX_gemma_advisor_last.txt'
        $txt | Set-Content -Path $advLog -Encoding utf8
        Write-Host "Saved: $advLog" -ForegroundColor DarkGray
    } catch {
        Write-Host "Gemma advisor skipped: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# LM Studio native REST (not OpenAI-compat): POST /api/v1/chat — optional MCP integrations (see lmstudio.ai/docs/developer/rest)
if ($env:ABRASAX_LM_REST_MCP_CHAT -eq '1') {
    Write-Host "`n=== 5d) LM Studio REST POST /api/v1/chat (Hugging Face MCP sample) ===" -ForegroundColor Cyan
    Write-Host 'Set ABRASAX_LM_REST_MCP_CHAT=0 to skip (may call huggingface.co/mcp; needs network).' -ForegroundColor DarkGray
    $mcpBody = @{
        model        = $llmChatModel
        input        = 'What is the top trending model on hugging face?'
        integrations = @(
            @{
                type           = 'ephemeral_mcp'
                server_label   = 'huggingface'
                server_url     = 'https://huggingface.co/mcp'
                allowed_tools  = @('model_search')
            }
        )
    } | ConvertTo-Json -Depth 10 -Compress
    $gotRest = $false
    foreach ($b in $llmBases) {
        $hostRoot = ($b.TrimEnd('/') -replace '/v1/*$', '')
        $restUrl = $hostRoot + '/api/v1/chat'
        try {
            $restR = Invoke-AbrasaxLmRest -Uri $restUrl -Method Post -Body $mcpBody -ContentType 'application/json; charset=utf-8' -TimeoutSec 300
            Write-Host "OK REST chat via $restUrl" -ForegroundColor DarkGray
            $restR | ConvertTo-Json -Depth 8
            $gotRest = $true
            break
        } catch {
            Write-Host "WARN REST $restUrl : $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
    if (-not $gotRest) { Write-Host 'FAIL or skip: /api/v1/chat (check LM Studio version, server on, MCP/network).' -ForegroundColor Yellow }
}

Write-Host "`n=== 6) k8s manifests (validate_manifests.py + optional kubectl) ===" -ForegroundColor Cyan
# $root = ...\gpu_compute\backends  →  ABRASAX\k8s
$k8s = Join-Path (Split-Path (Split-Path $root)) 'k8s'
if (Test-Path "$k8s\test_k8s.ps1") {
    powershell -NoProfile -ExecutionPolicy Bypass -File "$k8s\test_k8s.ps1"
} else {
    Write-Host "Skip: $k8s\test_k8s.ps1 not found" -ForegroundColor Yellow
}

Write-Host "`n=== 7) OBSIDIAN github_manager.py [github_manager token check] ===" -ForegroundColor Cyan
# Match Nexus: load GITHUB_TOKEN from repo .env if not already in session
if (-not $env:GITHUB_TOKEN) {
    $dotEnv = Join-Path $abrasaxRoot '.env'
    if (Test-Path $dotEnv) {
        foreach ($line in Get-Content $dotEnv -Encoding UTF8) {
            if ($line -match '^\s*GITHUB_TOKEN\s*=\s*(.+)\s*$') {
                $env:GITHUB_TOKEN = $matches[1].Trim().Trim('"').Trim([char]39)
                break
            }
        }
    }
}
$ghMgr = Join-Path $abrasaxRoot 'OBSIDIAN_NEURAL_OS\github_manager.py'
if (Test-Path $ghMgr) {
    $env:SYSTEM_CONFIG_ROOT = $abrasaxRoot
    python "$ghMgr" --test
    if ($LASTEXITCODE -ne 0) {
        Write-Host "github_manager: token check failed or skipped (set GITHUB_TOKEN). Full sync: python `"$ghMgr`"" -ForegroundColor Yellow
    }
} else {
    Write-Host "Skip: $ghMgr not found" -ForegroundColor Yellow
}

Write-Host "`n=== 8) ABRASAX_SOVEREIGN_NEXUS.py [dry-run] ===" -ForegroundColor Cyan
$nexus = Join-Path $abrasaxRoot 'ABRASAX_SOVEREIGN_NEXUS.py'
if (Test-Path $nexus) {
    $env:SYSTEM_CONFIG_ROOT = $abrasaxRoot
    python "$nexus" --dry-run
    if ($LASTEXITCODE -ne 0) { Write-Host "Nexus dry-run failed." -ForegroundColor Yellow }
} else {
    Write-Host "Skip: $nexus not found" -ForegroundColor Yellow
}

Write-Host "`n=== 9) ABRASAX launchers (syntax / presence) ===" -ForegroundColor Cyan
$pyLaunchers = @(
    (Join-Path $abrasaxRoot 'start_all_cloud_agents.py'),
    (Join-Path $abrasaxRoot 'start_gemma_abrasax.py')
)
foreach ($pl in $pyLaunchers) {
    if (Test-Path $pl) {
        python -m py_compile "$pl" 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { Write-Host "OK  py_compile $(Split-Path $pl -Leaf)" }
        else { Write-Host "WARN py_compile failed: $pl" -ForegroundColor Yellow }
    } else {
        Write-Host "MISSING  $pl" -ForegroundColor Yellow
    }
}
$psGemma = Join-Path $abrasaxRoot 'start_all_gemma_agents.ps1'
if (Test-Path $psGemma) { Write-Host "OK  present: start_all_gemma_agents.ps1" }
else { Write-Host "MISSING  $psGemma" -ForegroundColor Yellow }

$serverJs = Join-Path $abrasaxRoot 'server.js'
if (Test-Path $serverJs) {
    $node = Get-Command node -ErrorAction SilentlyContinue
    if ($node) {
        & node --check $serverJs 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { Write-Host "OK  node --check server.js" }
        else { Write-Host "WARN node --check failed: server.js" -ForegroundColor Yellow }
    } else {
        Write-Host "Skip: node not on PATH (server.js present)" -ForegroundColor DarkGray
    }
} else {
    Write-Host "MISSING  $serverJs" -ForegroundColor Yellow
}

Write-Host "`n=== 10a) Dev log / monitor file hints (auto-fix trail) ===" -ForegroundColor Cyan
$hints = @(
    (Join-Path $env:TEMP 'ABRASAX_resource_snapshot.json'),
    (Join-Path $env:TEMP 'ABRASAX_gemma_advisor_last.txt'),
    (Join-Path $env:TEMP 'ABRASAX_enforce_amd_mapping.log'),
    (Join-Path $env:TEMP 'ABRASAX_monitor.log'),
    (Join-Path $abrasaxRoot 'runtime\gemma_adaptive')
)
foreach ($hp in $hints) {
    if (Test-Path $hp) { Write-Host "OK  $hp" -ForegroundColor DarkGray }
    else { Write-Host "-   (optional) $hp" -ForegroundColor DarkGray }
}

Write-Host "`n=== 10) Optional non-stop Gemma (ABRASAX_START_GEMMA_NONSTOP=1) ===" -ForegroundColor Cyan
if ($env:ABRASAX_START_GEMMA_NONSTOP -eq '1' -and (Test-Path $psGemma)) {
    Start-Process powershell -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $psGemma -WindowStyle Minimized
    Write-Host "Started minimized: start_all_gemma_agents.ps1"
} else {
    Write-Host "Skip: set `$env:ABRASAX_START_GEMMA_NONSTOP='1' to spawn Gemma unified stack (or run: python start_gemma_abrasax.py)." -ForegroundColor DarkGray
}

if ($env:ABRASAX_KEEP_GOING -eq '1') {
    Write-Host "`n=== 11) KEEP_GOING: start_gemma_abrasax.py --no-wait (minimized) ===" -ForegroundColor Cyan
    $pyExe = Join-Path $abrasaxRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pyExe)) { $pyExe = 'python' }
    $gemmaLauncher = Join-Path $abrasaxRoot 'start_gemma_abrasax.py'
    if (Test-Path -LiteralPath $gemmaLauncher) {
        Start-Process -FilePath $pyExe -ArgumentList $gemmaLauncher, '--no-wait' -WorkingDirectory $abrasaxRoot -WindowStyle Minimized
        Write-Host "Started Gemma launcher (ABRASAX_KEEP_GOING=1)." -ForegroundColor Green
    } else {
        Write-Host "Skip: $gemmaLauncher not found" -ForegroundColor Yellow
    }
}

Write-Host "`n=== done ===" -ForegroundColor Green
