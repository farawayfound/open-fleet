# ---------------------------------------------------------------------------
# install.ps1 -- get to a Python, then hand over to fleetctl.
#
# The Windows half of install.sh, and the same job: everything that matters
# is in fleetctl/, which is stdlib-only Python. This exists because you
# cannot run Python before you have Python.
#
# Windows makes that harder than it sounds, and the fleet has the scars:
#
#   * `python` and `python3` on PATH are frequently the Microsoft Store
#     stub, which prints an advertisement and exits 49. It looks like an
#     interpreter to anything that checks whether the command exists.
#   * A per-user install under C:\Users\<name> is unreachable from a SYSTEM
#     scheduled task -- which is what supervises the gateway here. gpu-desktop-1
#     depended on one, and the profile was then renamed out from under it.
#
# So: install for ALL USERS, and verify by asking the interpreter rather
# than by looking for a file.
#
#   .\install.ps1                  detect, plan, apply
#   .\install.ps1 -DryRun          detect, plan, and say what apply WOULD do
#   .\install.ps1 -HostName gpu-desktop-1  name the box explicitly
#   .\install.ps1 -Command detect  run one fleetctl command instead
#
# Run ELEVATED. Over ssh that is automatic when the account is in
# Administrators: sshd hands out an elevated token.
# ---------------------------------------------------------------------------
[CmdletBinding()]
param(
  [switch]$DryRun,
  [string]$HostName = "",
  [string]$Command = "",
  [string[]]$Extra = @()
)
$ErrorActionPreference = "Stop"
$ProgressPreference    = "SilentlyContinue"   # or IWR spends its life redrawing
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$repo = $PSScriptRoot
$PythonVersion = "3.12.10"
$PythonUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"

function Say($m)  { Write-Host "==> $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "[warn] $m" -ForegroundColor Yellow }

if (-not (Test-Path "$repo\fleetctl")) { throw "no fleetctl\ beside this script -- run it from a checkout" }
if (-not (Test-Path "$repo\gateway\hw.py")) { throw "no gateway\hw.py -- the checkout is incomplete" }

# ---------------------------------------------------------------------------
function Test-Interpreter($exe) {
  # Asks the interpreter, rather than trusting that a file exists or that
  # `--version` printed something. The Store stub answers `--version` with an
  # advertisement and exits 49; a real interpreter exits 0 here and 1 if it
  # is too old.
  if (-not $exe) { return $false }
  try {
    & $exe -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
  } catch { return $false }
}

function Find-Python {
  $candidates = @(
    "C:\Program Files\Python314\python.exe",
    "C:\Program Files\Python313\python.exe",
    "C:\Program Files\Python312\python.exe",
    "C:\Program Files\Python311\python.exe"
  )
  # PATH last, and only after the all-users installs: a per-user 3.13 on PATH
  # would win here and then be invisible to the SYSTEM task that has to run it.
  foreach ($n in @("python3", "python")) {
    $c = (Get-Command $n -ErrorAction SilentlyContinue)
    if ($c) { $candidates += $c.Source }
  }
  foreach ($c in $candidates) {
    if (-not (Test-Interpreter $c)) { continue }
    # A per-user install (python.org's default "Install Now" puts it under
    # %LOCALAPPDATA%) passes the version test and then vanishes from the
    # SYSTEM task that has to run it -- the exact outage this script's
    # header describes. Treat it as "no usable Python" so the all-users
    # install below happens instead.
    $resolved = (Resolve-Path $c -ErrorAction SilentlyContinue).Path
    if ($resolved -and $resolved -like "$env:SystemDrive\Users\*") {
      Warn "ignoring per-user interpreter $resolved (a SYSTEM task cannot run it)"
      continue
    }
    return $c
  }
  return $null
}

$py = Find-Python
if (-not $py) {
  Say "no Python 3.10+ for all users -- installing $PythonVersion"
  if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "installing Python for all users needs an elevated shell"
  }
  $installer = "$env:TEMP\python-$PythonVersion-amd64.exe"
  Invoke-WebRequest $PythonUrl -OutFile $installer
  # InstallAllUsers=1 is the whole point. Include_test=0 saves ~25 MB of
  # test suite nothing here runs.
  Start-Process $installer -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0" -Wait
  Remove-Item $installer -Force
  $py = Find-Python
  if (-not $py) { throw "Python installed but still not usable -- check $env:TEMP for its log" }
}
Say "python: $py ($(& $py --version 2>&1))"

# ---------------------------------------------------------------------------
$flags = @()
if ($HostName) { $flags += @("--host", $HostName) }
if ($DryRun)   { $flags += "--dry-run" }
$flags += $Extra

Set-Location $repo

if ($Command) {
  # `--dry-run` belongs to apply and update; passing it to `detect` would
  # only be an argument error with a confusing provenance.
  $one = @($Command) + ($flags | Where-Object { $_ -ne "--dry-run" -or $Command -in @("apply", "update") })
  & $py -m fleetctl @one
  exit $LASTEXITCODE
}

Say "detect"
& $py -m fleetctl detect
if ($LASTEXITCODE -ne 0) { throw "detect failed" }

Say "plan"
# --write so the plan lands in hosts\<name>\host.yml and can be reviewed,
# edited and committed. A generated plan that only ever lived in memory would
# make the next run's decisions unreviewable.
& $py -m fleetctl plan --write @($flags | Where-Object { $_ -ne "--dry-run" })
if ($LASTEXITCODE -ne 0) {
  Warn "the plan is incomplete -- see above. On a box with no Tailscale the usual"
  Warn "missing value is the URL clients should use; supply it and re-run:"
  Warn '  .\install.ps1 -Extra "--set","network.public_api_url=http://<this box''s address>:8080/v1"'
  throw "the plan is incomplete"
}

Say "apply"
& $py -m fleetctl apply @flags
exit $LASTEXITCODE
