# Harden an already-installed Windows gateway's scheduled task.
#
# install.ps1 now sets these at create time, but a box provisioned before that
# still carries Task Scheduler's defaults, and they are wrong for a fleet node:
#
#   StopIfGoingOnBatteries      unplug the machine and Task Scheduler stops the
#                               gateway -- with a console break, so the only
#                               trace is a bare ^C at the end of gateway.log
#   DisallowStartIfOnBatteries  and then it will not start again while on
#                               battery; with ONSTART as the only trigger,
#                               returning to AC does not bring it back either
#   RestartCount = 0            a crash stays a crash until the next reboot
#
# The failure this produces is the quiet kind: /health simply stops answering,
# the hub marks the box offline, and nothing anywhere names the power cable.
# Seen on apu-tablet-1 (a Z13 tablet) on 2026-08-20.
#
# Safe to re-run: it only edits settings, and starts the task if -- and only
# if -- it is not already running.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File harden-gateway-task.ps1
#
# Needs elevation (it edits a SYSTEM task). Over ssh to mini-pc-1 / gpu-desktop-2 /
# apu-tablet-1 the ssh session already has it.

param([string]$TaskName = "llm-gateway", [int]$Port = 8080)

$ErrorActionPreference = "Stop"

$t = Get-ScheduledTask -TaskName $TaskName
$was = $t.Settings
Write-Output ("before: Disallow=" + $was.DisallowStartIfOnBatteries +
              " Stop=" + $was.StopIfGoingOnBatteries +
              " RestartCount=" + $was.RestartCount)

$t.Settings.DisallowStartIfOnBatteries = $false
$t.Settings.StopIfGoingOnBatteries     = $false
$t.Settings.RestartCount               = 3
$t.Settings.RestartInterval            = "PT1M"
$t.Settings.ExecutionTimeLimit         = "PT0S"
Set-ScheduledTask -InputObject $t | Out-Null

$now = (Get-ScheduledTask -TaskName $TaskName).Settings
Write-Output ("after : Disallow=" + $now.DisallowStartIfOnBatteries +
              " Stop=" + $now.StopIfGoingOnBatteries +
              " RestartCount=" + $now.RestartCount +
              " Interval=" + $now.RestartInterval +
              " TimeLimit=" + $now.ExecutionTimeLimit)

$state = (Get-ScheduledTask -TaskName $TaskName).State
Write-Output ("state : " + $state)
if ($state -ne "Running") {
  Write-Output "not running -- starting it"
  Start-ScheduledTask -TaskName $TaskName
  Start-Sleep -Seconds 12
  Write-Output ("state : " + (Get-ScheduledTask -TaskName $TaskName).State)
}

# Verify what it claims, rather than trusting the state word.
try {
  $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 10 `
        -Uri ("http://127.0.0.1:" + $Port + "/health")
  Write-Output ("health: " + [int]$r.StatusCode)
} catch {
  Write-Output ("health: unreachable -- " + $_.Exception.Message)
  exit 1
}
