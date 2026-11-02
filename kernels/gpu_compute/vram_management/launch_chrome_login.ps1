# ============================================================================
# ABRASAX — Unified Xenoliving System
# PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
# CREATOR: Sir Charles Spikes (BASEDGOD)
# ============================================================================

<#
Opens Google Chrome using your normal Windows profile (NotebookLM / Google session).

Set NOTEBOOKLM_CHROME_USER_DATA to ...\Google\Chrome\User Data (the folder that contains "Default").
Set NOTEBOOKLM_CHROME_PROFILE_DIRECTORY to "Default" or "Profile 1", etc.

Also accepts ABRASAX_CHROME_USER_DATA as an alias for user-data-dir.

Usage:
  .\launch_chrome_login.ps1
  .\launch_chrome_login.ps1 -Url "https://notebooklm.google.com/"
#>
param(
    [string]$Url = "https://notebooklm.google.com/"
)

$ErrorActionPreference = "Stop"
$candidates = @(
    "${env:ProgramFiles}\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
)
$chrome = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $chrome) {
    Write-Error "Chrome not found."
    exit 1
}

$userData = $null
foreach ($c in @(
        $env:NOTEBOOKLM_CHROME_USER_DATA,
        $env:ABRASAX_CHROME_USER_DATA,
        (Join-Path $env:LOCALAPPDATA "Google\Chrome\User Data")
    )) {
    if ($c -and (Test-Path $c)) {
        $userData = $c
        break
    }
}

$prof = $(if ($env:NOTEBOOKLM_CHROME_PROFILE_DIRECTORY) { $env:NOTEBOOKLM_CHROME_PROFILE_DIRECTORY } else { "Default" })

$argList = @()
if ($userData) {
    $argList += "--user-data-dir=`"$userData`""
    $argList += "--profile-directory=$prof"
}
$argList += $Url

Write-Host "[ABRASAX] Chrome: $chrome"
Write-Host "[ABRASAX] user-data-dir: $userData  profile: $prof"
Write-Host "[ABRASAX] url: $Url"
Start-Process -FilePath $chrome -ArgumentList $argList
