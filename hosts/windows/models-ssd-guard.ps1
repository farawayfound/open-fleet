# Keep a removable models SSD on the drive letter the config hard-codes, and
# clean it up after an unclean disconnect.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File models-ssd-guard.ps1 -Label ORICO -Install
#   powershell -NoProfile -ExecutionPolicy Bypass -File models-ssd-guard.ps1 -Label ORICO -Once
#
# WHY THIS EXISTS
#
# Two boxes keep their weights on an external SSD: apu-tablet-2 on a SanDisk
# Extreme (label `Extreme SSD`) and apu-tablet-1 on an ORICO enclosure (label
# `ORICO`), both exFAT, both mapped to D:. The drive LETTER is load-bearing
# config -- `gateway.env.cmd`, `state\llama-swap.yaml` and `state\models.json`
# all hard-code absolute `D:\...` paths -- so the supported repair for a
# re-letter has always been to put the letter back, never to rewrite the
# paths. That is a good rule and this script does not change it; it just stops
# a human from having to be present to apply it.
#
# The failure is always the same shape. Yank the disk while something has a
# GGUF open and exFAT keeps its dirty bit: `fsutil dirty query D:` says dirty,
# `Get-Volume` says HealthStatus Warning / OperationalStatus "Full Repair
# Needed". On 2026-08-27 apu-tablet-2 came back from exactly that as **F:**, and
# every deploy failed with `gateway did not come back` until someone with a
# keyboard ran chkdsk and reassigned D:. apu-tablet-1 hit the dirty half of it
# on 2026-08-29.
#
# So: find the volume by LABEL (which survives a reconnect; the letter does
# not), put it back on the letter the config expects, clear the dirty bit, and
# restart the engine so the models are servable again. The gateway itself
# needs no restart -- it re-reads the volume on its next poll.
#
# Note the ordering below: chkdsk runs BEFORE the letter is reassigned when
# the volume is already on the right letter, and AFTER when it had to be
# moved. chkdsk takes a path, so it has to be somewhere addressable first.
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)] [string] $Label,
  [string] $Letter = "D",
  [string[]] $RestartTasks = @("llama-swap"),
  [string] $LogPath = "C:\llmstack\state\ssd-guard.log",
  [int] $EveryMinutes = 2,
  [switch] $Install,
  [switch] $Once,
  [switch] $Eject
)

$ErrorActionPreference = "Stop"
$Letter = $Letter.TrimEnd(':').ToUpper()

function Say($m) {
  $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
  Write-Host $line
  try {
    $d = Split-Path $LogPath -Parent
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force $d | Out-Null }
    Add-Content -Path $LogPath -Value $line -ErrorAction SilentlyContinue
  } catch {}
}

function Invoke-Guard {
  # The disk being absent is not a fault -- it is the normal state of a
  # removable disk. Say nothing and leave, so the log stays readable.
  $vol = Get-Volume -ErrorAction SilentlyContinue | Where-Object { $_.FileSystemLabel -eq $Label }
  if (-not $vol) { return }
  if ($vol -is [array]) {
    Say "AMBIGUOUS: $($vol.Count) volumes labelled '$Label' -- refusing to guess. Relabel one."
    return
  }

  $changed = $false

  # 1. Wrong letter (or none): put it back where the config points.
  if ($vol.DriveLetter -ne $Letter) {
    $was = if ($vol.DriveLetter) { "$($vol.DriveLetter):" } else { "unmounted" }
    $occupant = Get-Volume -DriveLetter $Letter -ErrorAction SilentlyContinue
    if ($occupant) {
      Say "CONFLICT: '$Label' is on $was but ${Letter}: is taken by '$($occupant.FileSystemLabel)'. Not moving anything."
      return
    }
    try {
      Get-Partition -Volume $vol | Set-Partition -NewDriveLetter $Letter
      Say "re-lettered '$Label' from $was to ${Letter}: -- the config hard-codes ${Letter}:\, so this is the whole repair"
      $changed = $true
      Start-Sleep -Seconds 2
      $vol = Get-Volume -DriveLetter $Letter -ErrorAction SilentlyContinue
    } catch {
      Say "FAILED to re-letter '$Label' from $was to ${Letter}: -- $($_.Exception.Message)"
      return
    }
  }

  # 2. Dirty bit from an unclean eject. chkdsk /f on exFAT is quick (seconds
  #    on a 1 TB volume) and is what clears HealthStatus back to Healthy.
  $dirty = $false
  try { $dirty = ((& fsutil dirty query "${Letter}:" 2>&1) -join " ") -notmatch "NOT Dirty" } catch {}
  if ($dirty) {
    Say "'$Label' (${Letter}:) is dirty after an unclean disconnect -- running chkdsk /f"
    $out = & cmd /c "echo n| chkdsk ${Letter}: /f" 2>&1
    $bad = ($out | Select-String -Pattern "found problems|corrupt|bad sectors" | Measure-Object).Count
    Say ("chkdsk finished" + $(if ($bad) { " WITH FINDINGS -- read $LogPath and the console output" } else { ", no problems found" }))
    $out | ForEach-Object { Add-Content -Path $LogPath -Value ("      " + $_) -ErrorAction SilentlyContinue }
    $changed = $true
  }

  # 3. If we touched anything, bounce the engine so it re-opens the weights.
  #    The gateway is deliberately NOT restarted: it re-reads the volume on
  #    its next poll and restarting it would drop in-flight work.
  if ($changed) {
    foreach ($t in $RestartTasks) {
      cmd /c "schtasks /End /TN $t >nul 2>&1"
      Start-Sleep -Seconds 2
      cmd /c "schtasks /Run /TN $t >nul 2>&1"
      Say "restarted scheduled task '$t'"
    }
  }
}

if ($Eject) {
  # The cure for all of the above is not unplugging the disk while a
  # llama-server has 17 GB of GGUF mmap'd. Stop the engine, let the handles
  # go, flush, and only then pull it. Everything this script repairs is
  # damage from skipping this step.
  Say "preparing '$Label' (${Letter}:) for removal"
  foreach ($t in $RestartTasks) {
    cmd /c "schtasks /End /TN $t >nul 2>&1"
    Say "  stopped '$t'"
  }
  Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 3
  $holders = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "${Letter}:*" }
  if ($holders) {
    Say "  STILL HELD by: $(($holders | Select-Object -Expand Name) -join ', ') -- close these before unplugging"
  } else {
    Say "  nothing is running from ${Letter}: any more"
  }
  & cmd /c "echo n| chkdsk ${Letter}: 2>&1" | Out-Null   # read-only: forces a metadata flush
  Say "  safe to unplug. Reconnect and the guard puts ${Letter}: back within $EveryMinutes min, or run with -Once."
  Say "  then start the engine again: schtasks /Run /TN $($RestartTasks -join ' ; schtasks /Run /TN ')"
  return
}

if ($Install) {
  $self = $MyInvocation.MyCommand.Path
  $dest = "C:\llmstack\bin\models-ssd-guard.ps1"
  New-Item -ItemType Directory -Force (Split-Path $dest -Parent) | Out-Null
  if ($self -ne $dest) { Copy-Item $self $dest -Force }

  $cmd = "C:\llmstack\bin\run-ssd-guard.cmd"
  $tasks = ($RestartTasks -join ",")
  @"
@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "$dest" -Label "$Label" -Letter $Letter -RestartTasks $tasks -Once
"@ | Set-Content -Encoding ascii $cmd

  # SYSTEM, because Set-Partition and chkdsk both need it. Repeating rather
  # than event-triggered on purpose: a PnP arrival event fires before the
  # volume is mountable, so the poll is both simpler and more reliable, and
  # the work is a no-op when the disk is absent or already correct.
  cmd /c "schtasks /Delete /TN models-ssd-guard /F >nul 2>&1"
  $r = cmd /c "schtasks /Create /TN models-ssd-guard /TR `"$cmd`" /SC MINUTE /MO $EveryMinutes /RU SYSTEM /RL HIGHEST /F 2>&1"
  Write-Host $r
  cmd /c "schtasks /Run /TN models-ssd-guard >nul 2>&1"
  Say "installed models-ssd-guard: label '$Label' -> ${Letter}:, every $EveryMinutes min, restarts [$tasks]"
  return
}

Invoke-Guard
if (-not $Once) { Say "ran once (pass -Install to register the scheduled task)" }
