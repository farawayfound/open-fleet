# hosts/windows/enable-ssh.ps1 -- let the fleet's ssh reach a Windows box.
#
# deploy-gateway.sh (from the ci-runner Actions runner: every push, and
# every :00/:30 reconcile) and push.sh drive every Windows peer over ssh.
# Two Windows details decide whether that works, and neither is visible from
# the client side -- the runner just sees "Permission denied (publickey)",
# deploy-gateway.sh reports "box is UP but refused ssh", and every scheduled
# reconcile is red until somebody looks:
#
#   * `user` is an Administrator, and for members of Administrators sshd
#     ignores ~/.ssh/authorized_keys ENTIRELY (the `Match Group
#     administrators` block in the stock sshd_config). The file that counts
#     is C:\ProgramData\ssh\administrators_authorized_keys, and sshd refuses
#     even that one unless its ACL grants only SYSTEM and Administrators.
#   * The runner's key has to be IN it. gpu-laptop-3 joined the fleet on 2026-09-03
#     with its owner's key and a workstation's in that file and the runner's
#     in neither; 24 consecutive deploy-gateway runs failed on it over the
#     next two days. On the box the evidence is one line per tick in the
#     OpenSSH/Operational event log: "Connection closed by authenticating
#     user user 100.64.0.0 port N [preauth]" -- the runner, refused.
#
# So this script MERGES. Every key in fleet_authorized_keys (next to this
# script: the runner's public half, and anything else the whole fleet should
# trust) is added to administrators_authorized_keys if absent, and every key
# already there -- the owner's own, a workstation's -- is kept. apu-tablet-2's
# enable-ssh.ps1 overwrites; this one does not, because a box somebody also
# logs into by hand has keys in that file that no repo knows about.
#
# What it deliberately leaves alone:
#   * DefaultShell. deploy-gateway.sh works against cmd.exe or PowerShell
#     (its probe and restart travel as -EncodedCommand for exactly that
#     reason); push.sh cares (WINDOWS=1 is PowerShell, WINDOWS=2 is cmd, and
#     gpu-desktop-1 is on purpose cmd), so the choice stays per box. Reported, not
#     changed.
#   * An all-interfaces firewall rule. The fleet reaches a box over the
#     tailnet only, and a laptop that travels has no business answering :22
#     on hotel wifi. If nothing admits inbound :22 at all, ONE rule scoped to
#     100.64.0.0/10 is added; a box where the Tailscale client's own rules
#     (or an earlier hand-made one) already let it in is not touched.
#
# Run elevated. Over ssh that is automatic (the account is in Administrators,
# so sshd hands out an elevated token); from the console, an elevated
# PowerShell. Safe to re-run: every step is idempotent and says what it
# changed against what was already there.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File hosts\windows\enable-ssh.ps1
[CmdletBinding()]
param(
  # One public key per line, authorized_keys format (# comments allowed).
  # Defaults to fleet_authorized_keys next to this script.
  [string]$AuthorizedKeysFile = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Say($message) { Write-Host "==> $message" -ForegroundColor Cyan }

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  throw "must run elevated (writes C:\ProgramData\ssh and manages the sshd service)"
}

# ---------------------------------------------------------------------------
Say "sshd"
# The SERVICE is the test, not the Windows capability. gpu-laptop-3 runs
# Win32-OpenSSH 10.0 from the GitHub MSI (C:\Program Files\OpenSSH), and on
# such a box Get-WindowsCapability says OpenSSH.Server is NotPresent -- acting
# on that would install the older in-box build alongside a newer one.
$svc = Get-Service -Name sshd -ErrorAction SilentlyContinue
if (-not $svc) {
  $cap = Get-WindowsCapability -Online -Name "OpenSSH.Server*" | Select-Object -First 1
  if (-not $cap) { throw "no sshd service and no OpenSSH.Server capability to install" }
  Add-WindowsCapability -Online -Name $cap.Name | Out-Null
  Write-Host "    installed $($cap.Name)"
  $svc = Get-Service -Name sshd
} else {
  $bin = (Get-CimInstance Win32_Service -Filter "Name='sshd'").PathName
  Write-Host "    already installed: $bin"
}
if ($svc.StartType -ne "Automatic") { Set-Service -Name sshd -StartupType Automatic; Write-Host "    start type -> Automatic" }
if ($svc.Status -ne "Running") { Start-Service sshd; Write-Host "    started" }
Write-Host "    sshd: $((Get-Service sshd).Status), start=$((Get-Service sshd).StartType)"

$shell = (Get-ItemProperty HKLM:\SOFTWARE\OpenSSH -ErrorAction SilentlyContinue).DefaultShell
if ($shell) { Write-Host "    DefaultShell: $shell (push.sh WINDOWS=1 dialect)" }
else        { Write-Host "    DefaultShell: not set -> cmd.exe (push.sh WINDOWS=2 dialect)" }

# ---------------------------------------------------------------------------
Say "authorized keys (Administrators read ONLY administrators_authorized_keys)"
if (-not $AuthorizedKeysFile) {
  $AuthorizedKeysFile = Join-Path $PSScriptRoot "fleet_authorized_keys"
}
if (-not (Test-Path $AuthorizedKeysFile)) {
  throw "no key file at $AuthorizedKeysFile -- pass -AuthorizedKeysFile"
}
# A key's identity is its type and material; the comment is free text and
# differs between copies of the same key (hub lists the runner's twice
# under two comments). `sk-` covers FIDO keys should one ever appear.
$keyLine = '^(ssh-|ecdsa-|sk-)\S+\s+\S+'
function KeyId($line) {
  $p = ($line.Trim() -split '\s+')
  return "$($p[0]) $($p[1])"
}
$wanted = @(Get-Content $AuthorizedKeysFile | Where-Object { $_ -match $keyLine })
if (-not $wanted) { throw "no public keys in $AuthorizedKeysFile" }

$dest = "C:\ProgramData\ssh\administrators_authorized_keys"
$existing = @()
$hadBom = $false
if (Test-Path $dest) {
  # Get-Content honours a BOM; the bytes tell whether there was one. sshd
  # parses this file byte for byte, so the rewrite below always drops it.
  $bytes = [System.IO.File]::ReadAllBytes($dest)
  $hadBom = ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF)
  $existing = @(Get-Content $dest | Where-Object { $_.Trim() -ne "" })
}
$have = @{}
foreach ($l in $existing) { if ($l -match $keyLine) { $have[(KeyId $l)] = $true } }

$added = @()
$out = @($existing)
foreach ($k in $wanted) {
  $id = KeyId $k
  if ($have.ContainsKey($id)) { continue }
  $out += $k.Trim()
  $have[$id] = $true
  $added += $k
}
$kept = @($existing | Where-Object { $_ -match $keyLine }).Count

if ($added.Count -gt 0 -or $hadBom -or -not (Test-Path $dest)) {
  # ASCII and no BOM: a UTF-8 BOM in front of the first key is three bytes
  # sshd may or may not forgive depending on its build.
  Set-Content -Path $dest -Value $out -Encoding ascii -Force
  Write-Host "    kept $kept key(s), added $($added.Count):"
  foreach ($k in $added) {
    $p = ($k.Trim() -split '\s+', 3)
    Write-Host "      + $($p[0]) ...$($p[1].Substring([Math]::Max(0, $p[1].Length - 8))) $($p[2])"
  }
  if ($hadBom) { Write-Host "    (rewrote without the UTF-8 BOM it had)" }
} else {
  Write-Host "    all $($wanted.Count) fleet key(s) already present among $kept -- unchanged"
}

# sshd ignores the file if anyone but SYSTEM and Administrators can write
# it. By SID, so a localized Windows does not turn "BUILTIN\Administrators"
# into a name icacls cannot find. Inheritance off first, then every ACE
# that is not one of the two is removed, then both are granted -- an
# explicit stray ACE (a user's, from a hand edit) survives /inheritance:r
# and would silently disqualify the file.
$sysSid = "S-1-5-18"; $admSid = "S-1-5-32-544"
& icacls $dest /inheritance:r | Out-Null
$acl = Get-Acl $dest
foreach ($ace in @($acl.Access)) {
  $sid = $ace.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value
  if ($sid -ne $sysSid -and $sid -ne $admSid) {
    & icacls $dest /remove:g "*$sid" | Out-Null
    Write-Host "    ACL: removed $($ace.IdentityReference)"
  }
}
& icacls $dest /grant "*${sysSid}:(F)" | Out-Null
& icacls $dest /grant "*${admSid}:(F)" | Out-Null
Write-Host "    ACL: SYSTEM + Administrators only"

$keygen = Get-Command ssh-keygen -ErrorAction SilentlyContinue
if ($keygen) {
  Write-Host "    now authorized:"
  & $keygen.Source -l -f $dest | ForEach-Object { Write-Host "      $_" }
}

# ---------------------------------------------------------------------------
Say "firewall (inbound :22)"
# Port filters first, rules second: asking every inbound rule for its port
# filter is one CIM call per rule and takes minutes on a box with a few
# hundred of them. Explicit :22 only -- a program-scoped "any port" rule
# (Tailscale-In is one) may admit ssh as well, but an all-ports match would
# count every app's rule, and a redundant allow scoped to the tailnet costs
# nothing.
$admits = @(Get-NetFirewallPortFilter -All -ErrorAction SilentlyContinue |
  Where-Object { $_.Protocol -eq "TCP" -and @($_.LocalPort) -contains "22" } |
  Get-NetFirewallRule -ErrorAction SilentlyContinue |
  Where-Object { $_.Direction -eq "Inbound" -and $_.Action -eq "Allow" -and $_.Enabled -eq "True" })
if ($admits) {
  Write-Host "    admitted by: $(($admits | ForEach-Object { $_.DisplayName }) -join '; ')"
} else {
  New-NetFirewallRule -DisplayName "OpenSSH-Server-In-TCP (tailnet)" -Direction Inbound -Action Allow `
    -Protocol TCP -LocalPort 22 -RemoteAddress 100.64.0.0/10 -Profile Any | Out-Null
  Write-Host "    added OpenSSH-Server-In-TCP (tailnet): TCP/22 from 100.64.0.0/10 only"
}

# ---------------------------------------------------------------------------
# No restart: sshd reads authorized_keys and its ACL at every authentication,
# so the runner's next connection already sees the new key.
$ts = Get-Command tailscale -ErrorAction SilentlyContinue
if (-not $ts -and (Test-Path "C:\Program Files\Tailscale\tailscale.exe")) { $ts = @{ Source = "C:\Program Files\Tailscale\tailscale.exe" } }
$addr = $null
if ($ts) { try { $addr = (& $ts.Source ip -4 2>$null | Select-Object -First 1) } catch {} }
if ($addr) {
  Write-Host "done. From the runner: ssh -i ~/.ssh/fleet_deploy_key user@$addr"
} else {
  Write-Host "done. From the runner: ssh -i ~/.ssh/fleet_deploy_key user@<this box's tailnet address>"
}
Write-Host "(by address, not name -- an address does not drift when a box is renamed)"
