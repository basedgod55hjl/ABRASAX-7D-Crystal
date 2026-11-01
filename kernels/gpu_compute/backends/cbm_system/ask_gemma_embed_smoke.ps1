# ============================================================================
# ABRASAX — Unified Xenoliving System
# PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
# CREATOR: Sir Charles Spikes (BASEDGOD)
# ============================================================================

# Smoke-test LM Studio OpenAI-compatible API: embeddings + chat (reads performance_config.json).
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File ask_gemma_embed_smoke.ps1
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$abrasaxRoot = (Resolve-Path (Join-Path $here '..\..\..')).Path
$cfgPath = Join-Path $here "performance_config.json"
if (-not (Test-Path $cfgPath)) { throw "Missing $cfgPath" }
$cfg = Get-Content $cfgPath -Raw -Encoding UTF8 | ConvertFrom-Json
$lc = $cfg.llm_controls
$bases = @()
if ($lc.base_urls) { $bases += @($lc.base_urls) }
if ($lc.base_url) { $bases += $lc.base_url }
if ($bases.Count -eq 0) { throw "performance_config: no llm_controls.base_url / base_urls" }
$model = [string]$lc.model
$embModel = [string]$lc.embedding_model
$pick = $null
foreach ($b in $bases) {
  $u = ([string]$b).Trim().TrimEnd("/")
  if (-not $u) { continue }
  if ($u -match '^169\.254\.') { continue }
  $pick = $u
  break
}
if (-not $pick) { $pick = ([string]$bases[0]).Trim().TrimEnd("/") }
$root = $pick -replace '/v1$', ''
Write-Host "[smoke] base (no /v1): $root  model=$model  embed=$embModel"

$hdr = @{}
$k = $env:OPENAI_API_KEY; if (-not $k) { $k = $env:LM_STUDIO_API_KEY }; if (-not $k) { $k = $env:LM_API_TOKEN }
if (-not $k) {
  $de = Join-Path $abrasaxRoot '.env'
  if (Test-Path -LiteralPath $de) {
    foreach ($line in Get-Content -LiteralPath $de -Encoding UTF8) {
      if ($line -match '^\s*#' -or -not ($line.Trim())) { continue }
      if ($line -match '^\s*(OPENAI_API_KEY|LM_STUDIO_API_KEY|LM_API_TOKEN)\s*=\s*(.*)$') { $k = $matches[2].Trim().Trim('"').Trim("'"); break }
    }
  }
}
if ($k) { $hdr = @{ Authorization = ('Bearer ' + $k.Trim()) }; Write-Host "[smoke] LM auth: Bearer from env/.env" -ForegroundColor DarkGray }

$embBody = (@{ model = $embModel; input = "smoke test line" } | ConvertTo-Json -Compress)
$chatBody = (@{
  model = $model
  messages = @(@{ role = "user"; content = "Reply with exactly: pong" })
  max_tokens = 32
  temperature = 0.2
} | ConvertTo-Json -Compress -Depth 6)

try {
  $irmEmb = @{ Uri = "$root/v1/embeddings"; Method = 'Post'; Body = $embBody; ContentType = 'application/json; charset=utf-8'; TimeoutSec = 120 }
  if ($hdr.Count -gt 0) { $irmEmb.Headers = $hdr }
  $e = Invoke-RestMethod @irmEmb
  $dim = $e.data[0].embedding.Count
  Write-Host "[smoke] embeddings OK dim=$dim object=$($e.object)"
} catch {
  Write-Host "[smoke] embeddings FAIL: $($_.Exception.Message)"
  if ($_.ErrorDetails.Message) { Write-Host $_.ErrorDetails.Message }
  exit 1
}

try {
  $irmChat = @{ Uri = "$root/v1/chat/completions"; Method = 'Post'; Body = $chatBody; ContentType = 'application/json; charset=utf-8'; TimeoutSec = 180 }
  if ($hdr.Count -gt 0) { $irmChat.Headers = $hdr }
  $c = Invoke-RestMethod @irmChat
  $txt = $c.choices[0].message.content
  Write-Host "[smoke] chat OK: $($txt.Substring(0, [Math]::Min(120, $txt.Length)))"
} catch {
  Write-Host "[smoke] chat FAIL: $($_.Exception.Message)"
  if ($_.ErrorDetails.Message) { Write-Host $_.ErrorDetails.Message }
  exit 2
}

Write-Host "[smoke] done."
