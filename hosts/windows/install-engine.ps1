# Put an inference engine on a Windows fleet box that so far has only run the
# gateway: llama.cpp (the rolling GitHub build) fronted by llama-swap, as a
# SYSTEM scheduled task on :8081, wired into the gateway's environment.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File install-engine.ps1 -Backend vulkan
#   powershell -NoProfile -ExecutionPolicy Bypass -File install-engine.ps1 -Backend cuda
#
# This is the engine half of hosts/apu-tablet-2/install.ps1, split out so the two
# "telemetry + staging" boxes -- apu-tablet-1 (Strix Halo, Vulkan) and gpu-desktop-2
# (2x RTX 3090, CUDA) -- can be promoted to inference peers without a full
# reinstall. Everything the gateway half already set up (python, venv, the
# task, gateway.env.cmd and its admin token) is left exactly as it is; this
# adds the three env lines the gateway needs to build launch commands, and
# nothing here touches the model registry. Registering models is the
# gateway's job, through PUT /admin/api/models, which is also what verifies
# they load.
#
# CUDA: llama.cpp's Windows CUDA build links the runtime dynamically and
# ships it as a separate cudart-*.zip; the DLLs have to sit beside
# llama-server.exe or it exits on start with no useful message. Both
# archives come from the same release so they cannot disagree on version.
#
# Idempotent. Re-running replaces the binaries with the current release and
# restarts the engine; it does not touch a registry that already exists.
# Needs elevation (SYSTEM tasks, the firewall); an ssh session as an admin
# user already has it.
[CmdletBinding()]
param(
  [ValidateSet("vulkan", "cuda")] [string] $Backend = "vulkan",
  [string] $Root = "C:\llmstack",
  # The CUDA runtime major the driver on the box supports. 12.4 runs on any
  # driver from the last two years; 13.x needs a current one.
  [string] $CudaVersion = "12.4"
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
function Say($m) { Write-Host "==> $m" }

$bin = "$Root\bin"; $state = "$Root\state"; $llamaDir = "$Root\llama"
New-Item -ItemType Directory -Force $bin, $state, $llamaDir | Out-Null
$envFile = "$Root\gateway.env.cmd"
if (-not (Test-Path $envFile)) { throw "no $envFile -- install the gateway first (hosts\<box>\install.ps1)" }

Say "llama.cpp ($Backend)"
# /releases/latest points at a pinned old tag with no Windows assets; the
# real binaries live on the rolling bNNNNN tags. Take the newest that has
# the asset we need (and, for CUDA, its matching cudart).
# Anchored on the "llama-" prefix: the runtime archive is named
# cudart-llama-bin-win-cuda-<ver>-x64.zip and matches any looser pattern,
# which is how the first run of this unpacked three DLLs and no server.
$pattern = if ($Backend -eq "cuda") { "llama-*-bin-win-cuda-$CudaVersion-x64.zip" } else { "llama-*-bin-win-vulkan-x64.zip" }
$releases = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30"
$release = $releases | Where-Object {
  $_.tag_name -match '^b\d+' -and ($_.assets | Where-Object { $_.name -like $pattern })
} | Select-Object -First 1
if (-not $release) { throw "no recent llama.cpp release carries $pattern" }
$asset = $release.assets | Where-Object { $_.name -like $pattern } | Select-Object -First 1
Write-Host "    $($release.tag_name)  $($asset.name)"
$archive = "$env:TEMP\$($asset.name)"
Invoke-WebRequest $asset.browser_download_url -OutFile $archive
Remove-Item "$llamaDir\*" -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive -Path $archive -DestinationPath $llamaDir -Force
Remove-Item $archive -Force
if (-not (Test-Path "$llamaDir\llama-server.exe")) {
  $server = Get-ChildItem $llamaDir -Recurse -Filter "llama-server.exe" | Select-Object -First 1
  if (-not $server) { throw "llama-server.exe was not in $($asset.name)" }
  Get-ChildItem $server.DirectoryName | Move-Item -Destination $llamaDir -Force
}
if ($Backend -eq "cuda") {
  $rt = $release.assets | Where-Object { $_.name -like "cudart-*win-cuda-$CudaVersion-x64.zip" } | Select-Object -First 1
  if (-not $rt) { throw "release $($release.tag_name) has no cudart for $CudaVersion" }
  Write-Host "    $($rt.name)"
  $rtZip = "$env:TEMP\$($rt.name)"
  Invoke-WebRequest $rt.browser_download_url -OutFile $rtZip
  Expand-Archive -Path $rtZip -DestinationPath $llamaDir -Force
  Remove-Item $rtZip -Force
  # The cudart archive nests its DLLs one directory down on some releases.
  Get-ChildItem $llamaDir -Recurse -Filter "cudart64_*.dll" | Where-Object { $_.DirectoryName -ne $llamaDir } |
    ForEach-Object { Get-ChildItem $_.DirectoryName -Filter *.dll | Move-Item -Destination $llamaDir -Force }
}
$release.tag_name | Set-Content -Encoding ascii "$Root\LLAMA_CPP_REV"
& "$llamaDir\llama-server.exe" --list-devices 2>&1 | ForEach-Object { Write-Host "    $_" }

Say "llama-swap"
$swapReleases = Invoke-RestMethod "https://api.github.com/repos/mostlygeek/llama-swap/releases?per_page=10"
$swapRelease = $swapReleases | Where-Object { $_.assets | Where-Object { $_.name -like "*windows_amd64.zip" } } | Select-Object -First 1
if (-not $swapRelease) { throw "no recent llama-swap release carries a windows_amd64 asset" }
$swapAsset = $swapRelease.assets | Where-Object { $_.name -like "*windows_amd64.zip" } | Select-Object -First 1
Write-Host "    $($swapRelease.tag_name)  $($swapAsset.name)"
$swapZip = "$env:TEMP\$($swapAsset.name)"; $swapDir = "$env:TEMP\llama-swap-unpack"
Invoke-WebRequest $swapAsset.browser_download_url -OutFile $swapZip
Remove-Item $swapDir -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive -Path $swapZip -DestinationPath $swapDir -Force
$swapExe = Get-ChildItem $swapDir -Recurse -Filter "llama-swap.exe" | Select-Object -First 1
if (-not $swapExe) { throw "llama-swap.exe was not in $($swapAsset.name)" }
# A running llama-swap holds its own image open; stop the task before copying.
cmd /c "schtasks /End /TN llama-swap >nul 2>&1"
Get-Process -Name llama-swap, llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep 2
Copy-Item $swapExe.FullName "$bin\llama-swap.exe" -Force
Remove-Item $swapZip, $swapDir -Recurse -Force -ErrorAction SilentlyContinue

Say "swap configuration"
if (-not (Test-Path "$state\llama-swap.yaml")) {
@"
# GENERATED BY THE OPEN-FLEET GATEWAY -- DO NOT EDIT BY HAND.
healthCheckTimeout: 900
logLevel: info
startPort: 5800
metricsMaxInMemory: 5000
models: {}
"@ | Set-Content -Encoding ascii "$state\llama-swap.yaml"
}

Say "gateway environment"
# Only the lines the engine adds. The gateway's own lines -- and its admin
# token -- are never rewritten here.
$lines = Get-Content $envFile
$want = @(
  "set LLMSTACK_LLAMA_SERVER=$llamaDir\llama-server.exe",
  "set LLMSTACK_LLAMA_BENCH=$llamaDir\llama-bench.exe",
  "set LLMSTACK_SWAP_CONFIG=$state\llama-swap.yaml"
)
foreach ($w in $want) {
  $key = ($w -split '=')[0]
  if ($lines | Where-Object { $_ -like "$key=*" }) { $lines = $lines | ForEach-Object { if ($_ -like "$key=*") { $w } else { $_ } } }
  else { $lines += $w }
}
# The "nothing listens on 8081" note from the staging-only install is no
# longer true; drop it so the file does not contradict itself.
$lines = $lines | Where-Object { $_ -notmatch '^rem Nothing listens on 8081|^rem on demand; nothing sits warm' }
Set-Content -Encoding ascii -Path $envFile -Value $lines

@"
@echo off
call "$envFile"
"$bin\llama-swap.exe" --config "$state\llama-swap.yaml" --listen 127.0.0.1:8081 >> "$state\llama-swap.log" 2>&1
"@ | Set-Content -Encoding ascii "$bin\run-llama-swap.cmd"

Say "firewall (8081 stays loopback-only)"
netsh advfirewall firewall delete rule name="llama-swap-lan-block" | Out-Null
netsh advfirewall firewall add rule name="llama-swap-lan-block" dir=in action=block protocol=TCP localport=8081 | Out-Null

Say "scheduled task"
cmd /c "schtasks /Delete /TN llama-swap /F >nul 2>&1"
cmd /c "schtasks /Create /TN llama-swap /TR `"\`"$bin\run-llama-swap.cmd\`"`" /SC ONSTART /RU SYSTEM /RL HIGHEST /F"
if ($LASTEXITCODE -ne 0) { throw "could not create the llama-swap task" }
# Same hardening as the gateway task and for the same reasons (see
# hosts\windows\harden-gateway-task.ps1): edit the object in place so the
# SYSTEM principal schtasks just set survives.
$t = Get-ScheduledTask -TaskName llama-swap
$t.Settings.DisallowStartIfOnBatteries = $false
$t.Settings.StopIfGoingOnBatteries     = $false
$t.Settings.StartWhenAvailable         = $true
$t.Settings.MultipleInstances          = "IgnoreNew"
$t.Settings.RestartCount               = 3
$t.Settings.RestartInterval            = "PT1M"
$t.Settings.ExecutionTimeLimit         = "PT0S"
Set-ScheduledTask -InputObject $t | Out-Null
cmd /c "schtasks /Run /TN llama-swap" | Out-Null

Say "restarting the gateway so it reads the new environment"
cmd /c "schtasks /End /TN llm-gateway >nul 2>&1"
Start-Sleep 3
cmd /c "schtasks /Run /TN llm-gateway" | Out-Null

Say "health"
$ok = $false
for ($i = 0; $i -lt 20 -and -not $ok; $i++) {
  Start-Sleep 3
  try {
    $h = Invoke-RestMethod http://127.0.0.1:8080/health -TimeoutSec 4
    $r = Invoke-RestMethod http://127.0.0.1:8081/running -TimeoutSec 4
    $ok = $true
    Write-Host ("    gateway: " + ($h | ConvertTo-Json -Compress))
    Write-Host ("    llama-swap: " + ($r | ConvertTo-Json -Compress))
  } catch {}
}
if (-not $ok) {
  Get-Content "$state\llama-swap.log" -Tail 20 -ErrorAction SilentlyContinue
  Get-Content "$state\gateway.log" -Tail 20 -ErrorAction SilentlyContinue
  throw "engine or gateway did not come up"
}
Write-Host "Engine installed: llama.cpp $($release.tag_name) ($Backend) + llama-swap $($swapRelease.tag_name). Register models through the gateway."
