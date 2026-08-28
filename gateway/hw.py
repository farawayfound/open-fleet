"""Hardware and OS facts, read the same way by the gateway and the installer.

This module is the answer to a question both halves of the fleet have to ask
and neither should answer twice: what is actually in this box. The gateway
asks so a dashboard can show it and so a context window can be sized against
real VRAM; `fleetctl` asks BEFORE the gateway exists, to decide which backend
to build, which package manager to drive, and what to write into the env file.
Two implementations of "how much VRAM is fitted" would have drifted the first
time a driver changed, and the one that drifted would have been the
installer's -- the half nobody looks at until a box comes up wrong.

STDLIB ONLY, and deliberately so. `fleetctl` runs on a machine that has no
virtualenv yet, so anything here that reached for psutil, httpx or PyYAML
would put a pip install in front of the first honest reading of the hardware.
Everything below is `platform`, `subprocess`, `plistlib` and sysfs.

The three composite functions -- gpu_cards(), vram_total_bytes() and
darwin_gpu_stats() -- take their collaborators as arguments with sensible
defaults. That seam is not decoration: the gateway's tests replace
`amdgpu_stats` and `load_specs` ON THE APP MODULE and expect the composite to
pick the replacement up, which a plain re-export could not do. app.py keeps a
three-line wrapper for each and names the backends explicitly there; the
installer just calls the default form.
"""
from __future__ import annotations

import json
import platform
import plistlib
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except Exception:  # noqa: BLE001
        return None


def amdgpu_stats() -> list[dict]:
    """Read amdgpu telemetry straight from sysfs -- no ROCm runtime needed."""
    out = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        dev = card / "device"
        if not (dev / "gpu_busy_percent").exists():
            continue
        vram_total = _read_int(dev / "mem_info_vram_total")
        vram_used = _read_int(dev / "mem_info_vram_used")
        gtt_used = _read_int(dev / "mem_info_gtt_used")
        gtt_total = _read_int(dev / "mem_info_gtt_total")
        temp = None
        power = None
        sclk = None
        for hw in (dev / "hwmon").glob("hwmon*"):
            temp = temp or _read_int(hw / "temp1_input")
            power = power or _read_int(hw / "power1_average")
            sclk = sclk or _read_int(hw / "freq1_input")
        out.append(
            {
                "card": card.name,
                "busy_percent": _read_int(dev / "gpu_busy_percent"),
                "vram_total": vram_total,
                "vram_used": vram_used,
                "gtt_used": gtt_used,
                "gtt_total": gtt_total,
                "temp_c": (temp / 1000.0) if temp else None,
                "power_w": (power / 1_000_000.0) if power else None,
                "sclk_mhz": (sclk / 1_000_000.0) if sclk else None,
            }
        )
    return out


_nvsmi = shutil.which("nvidia-smi")


def nvidia_stats(nvsmi: str | None = None) -> list[dict]:
    """NVIDIA telemetry via nvidia-smi -- covers the CUDA boxes (Windows or
    Linux) where amdgpu sysfs has nothing to say.

    `nvsmi` overrides which binary to ask, and exists for the same reason the
    other seams here do: app.py resolves it through ITS module namespace, so a
    test that points `_nvsmi` at a fake still points the reading at the fake.
    """
    nvsmi = _nvsmi if nvsmi is None else nvsmi
    if not nvsmi:
        return []
    try:
        r = subprocess.run(
            [nvsmi, "--query-gpu=name,utilization.gpu,memory.total,memory.used,"
             "temperature.gpu,power.draw,clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        )
        if r.returncode != 0:
            return []
    except Exception:  # noqa: BLE001
        return []

    def num(v: str) -> float | None:
        try:
            return float(v.strip())
        except ValueError:
            return None

    out = []
    for i, line in enumerate(r.stdout.strip().splitlines()):
        f = [x.strip() for x in line.split(",")]
        if len(f) < 7:
            continue
        vram_total = num(f[2])
        vram_used = num(f[3])
        out.append(
            {
                "card": f[0] or ("cuda" + str(i)),
                "busy_percent": num(f[1]),
                "vram_total": int(vram_total * 1024**2) if vram_total else None,
                # `is not None`, not truthiness: num() already returns None for
                # a field nvidia-smi could not parse, so 0 here means the card
                # genuinely has nothing on it. Treating that 0 as "unknown"
                # made gpu-laptop-2's 4070 -- idle, with the panel on the iGPU, so
                # honestly at 0 MiB -- render as "VRAM –" beside boxes whose
                # display keeps a few MB resident on the same card they serve
                # from. vram_total keeps its truthiness test: a card that
                # reports 0 bytes fitted is a bad reading, not a small card,
                # and every percentage downstream divides by it.
                "vram_used": int(vram_used * 1024**2) if vram_used is not None else None,
                "gtt_used": None,
                "gtt_total": None,
                "temp_c": num(f[4]),
                "power_w": num(f[5]),
                "sclk_mhz": num(f[6]),
            }
        )
    return out


_wingpu_cache: dict[str, Any] = {"t": 0.0, "gpus": []}
WIN_GPU_TTL = 10.0

# One PowerShell round trip for everything Windows will tell us about a GPU
# without vendor tooling. Written as a single script because starting
# powershell.exe costs more than the counters do.
#
#   \GPU Engine(*)\Utilization Percentage   busy %, per engine, per process
#   \GPU Adapter Memory(*)\Dedicated Usage  VRAM in use, per adapter
#   HardwareInformation.qwMemorySize        VRAM fitted, from the driver's
#                                           registry key -- Win32_VideoController
#                                           .AdapterRAM is a 32-bit field and
#                                           reports 4 GB for a 16 GB card.
_WIN_GPU_PS = r"""
$ErrorActionPreference = 'SilentlyContinue'
# Win32_VideoController lists the adapters that are actually THERE. The
# registry class key lists every adapter this machine has ever had a driver
# for, and it never forgets: gpu-laptop-2 still carries an "AMD Radeon RX 7800M"
# claiming 11.98 GiB from an eGPU that is long gone, and sized off the
# registry alone that ghost outranked the real 4070 on the fleet page. It is
# the same phantom that once had mini-pc-1 advertising 12 GiB it did not
# have. So the registry supplies the SIZES and Win32_VideoController decides
# which of them count.
$present = @(Get-CimInstance Win32_VideoController | ForEach-Object { [string]$_.Name })
$total = @{}
Get-ChildItem 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}' |
  ForEach-Object {
    $p = Get-ItemProperty $_.PSPath
    if ($p.'HardwareInformation.qwMemorySize' -and $p.DriverDesc) {
      $total[$p.DriverDesc] = [int64]$p.'HardwareInformation.qwMemorySize'
    }
  }
$live = @{}
foreach ($k in $total.Keys) { if ($present -contains [string]$k) { $live[$k] = $total[$k] } }
# Only trust the gate if it matched something. If no DriverDesc lines up with
# a present adapter then the naming convention has changed on this box, and a
# card reported with a stale size beats no card at all -- gpu-desktop-1's whole reason
# for having this backend is that a missing VRAM row reads as "no GPU".
if ($live.Count -gt 0) { $total = $live }
# Engine utilisation is reported per (pid, adapter, engine type). The card's
# busy figure is the busiest single engine, not the sum -- summing 3D, Copy and
# Video across a dozen processes cheerfully reports 400%.
$busy = 0.0
foreach ($s in (Get-Counter '\GPU Engine(*)\Utilization Percentage').CounterSamples) {
  if ($s.InstanceName -match 'engtype_(3D|Compute)' -and $s.CookedValue -gt $busy) {
    $busy = $s.CookedValue
  }
}
# $null, not 0, until a sample says otherwise: "no counter answered" and "the
# adapter is holding nothing" are different facts, and starting the running
# maximum at zero reports the first as the second.
$used = $null
foreach ($s in (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage').CounterSamples) {
  if ($null -eq $used -or $s.CookedValue -gt $used) { $used = $s.CookedValue }
}
$out = @()
foreach ($k in $total.Keys) {
  # [int64]$null is 0 in PowerShell, so the cast has to be guarded or the
  # distinction made just above is thrown away on the way out.
  $out += [pscustomobject]@{ card = $k; vram_total = $total[$k]
                             vram_used = $(if ($null -eq $used) { $null } else { [int64]$used })
                             busy_percent = [math]::Round($busy, 1) }
}
# -InputObject, and no -AsArray: this is Windows PowerShell 5.1, where
# -AsArray does not exist (it is PowerShell 6+) and piping unrolls a
# one-element array into a bare object. Getting either wrong prints nothing at
# all, because the error goes to a stream nobody is reading.
ConvertTo-Json -Compress -InputObject ([array]$out)
"""


def gpu_cards(spec_vram_gb: float | None = None, *,
              amd: Callable[[], list[dict]] | None = None,
              nv: Callable[[], list[dict]] | None = None,
              win: Callable[[], list[dict]] | None = None,
              mac: Callable[[], list[dict]] | None = None) -> list[dict]:
    """Every GPU on this box, listed once each.

    Windows' performance counters are a FALLBACK, not a third opinion. On a
    Windows box that also has nvidia-smi they describe the same card the
    vendor tool already described, and concatenating the two listed gpu-laptop-2's
    RTX 4070 twice on the fleet page -- once with a temperature and no VRAM
    figure, once with a VRAM figure and no temperature, as though the laptop
    had grown a second card. The vendor backends know strictly more
    (temperature, power, clocks), so they win wherever they answer at all.
    """
    amd = amd or amdgpu_stats
    nv = nv or nvidia_stats
    win = win or windows_gpu_stats
    mac = mac or (lambda: darwin_gpu_stats(spec_vram_gb))
    vendor = amd() + nv()
    return vendor or win() or mac()


def windows_gpu_stats() -> list[dict]:
    """GPU telemetry on Windows for cards nvidia-smi does not cover.

    gpu-desktop-1 is why this exists: an RDNA4 Radeon on Windows has neither amdgpu
    sysfs nor an nvidia-smi, so the box that the whole fleet page is about --
    a 16 GB card -- rendered as "n/a, no amdgpu sysfs" with no VRAM row at
    all. Windows' own performance counters know all of it.

    Deliberately coarse. The counters give a per-adapter dedicated-memory
    figure and a per-engine utilisation, and nothing about temperature, power
    or clocks without vendor tooling, so those stay None rather than being
    invented. Cached for WIN_GPU_TTL because it costs a PowerShell start."""
    if platform.system() != "Windows":
        return []
    if _wingpu_cache["gpus"] and time.time() - _wingpu_cache["t"] < WIN_GPU_TTL:
        return _wingpu_cache["gpus"]
    gpus: list[dict] = []
    try:
        p = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _WIN_GPU_PS],
            capture_output=True, text=True, timeout=25,
        )
        raw = json.loads((p.stdout or "").strip() or "[]")
        for g in raw if isinstance(raw, list) else [raw]:
            total = int(g.get("vram_total") or 0)
            # The integrated Radeon on an APU reports a few hundred MB of
            # "dedicated" memory carved out of system RAM. It is not a card
            # anyone routes work to, and listing it alongside the real one
            # makes the fleet's pooled VRAM total wrong.
            if total < 2 * 1024 ** 3:
                continue
            gpus.append({
                "card": str(g.get("card") or "gpu"),
                "busy_percent": float(g.get("busy_percent") or 0.0),
                "vram_total": total,
                # Filled in below: the counter is per-adapter but its instance
                # names carry a LUID, not a card name, so it can only be
                # attributed when there is one card to attribute it to.
                "vram_used": (int(g["vram_used"])
                              if g.get("vram_used") is not None else None),
                "gtt_used": None,
                "gtt_total": None,
                "temp_c": None,
                "power_w": None,
                "sclk_mhz": None,
            })
        if len(gpus) != 1:
            # More than one discrete adapter and the dedicated-usage figure
            # cannot be told apart between them; reporting the busiest one
            # against each card would be a number that looks measured and is
            # not. No fleet box is in this shape today (the multi-GPU one is
            # NVIDIA, which nvidia-smi answers for properly), so this is the
            # honest placeholder rather than dead weight.
            for g in gpus:
                g["vram_used"] = None
    except Exception:  # noqa: BLE001 -- no counters is just no telemetry
        gpus = []
    _wingpu_cache.update(t=time.time(), gpus=gpus)
    return gpus


_macgpu_cache: dict[str, Any] = {"t": 0.0, "gpus": []}
MAC_GPU_TTL = 10.0


def darwin_gpu_stats(spec_vram_gb: float | None = None) -> list[dict]:
    """GPU telemetry on Apple silicon (and Intel Macs), from IOKit.

    The Macs were the fleet's blind spot: amdgpu sysfs, nvidia-smi and
    Windows' counters all answer for somebody, and none of them answers for
    Metal -- so gpu_cards() came back empty and a machine's Overview page
    showed "n/a, no GPU telemetry" beside a 32-core GPU that was, measurably,
    at 76% serving a model. The fleet Home page hid it: with no cards to
    aggregate it fell through to the spec sheet for VRAM and multiplied a
    zero busy figure by the spec sheet's TFLOPS, so it rendered plausible
    numbers rather than an obvious gap. One backend fixes both.

    `ioreg -a` is an XML property list, so this reads structured data rather
    than scraping ioreg's own text format. `PerformanceStatistics` on an
    AGXAccelerator carries "Device Utilization %" and "In use system memory";
    temperature, power and clocks need `powermetrics`, which needs root, so
    they stay None exactly as they do on the Windows backend.

    Cached for MAC_GPU_TTL because the plist is ~100 KB to produce and parse.
    """
    if platform.system() != "Darwin":
        return []
    if _macgpu_cache["gpus"] and time.time() - _macgpu_cache["t"] < MAC_GPU_TTL:
        return _macgpu_cache["gpus"]
    gpus: list[dict] = []
    try:
        p = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-a", "-c", "IOAccelerator"],
            capture_output=True, timeout=15,
        )
        entries = plistlib.loads(p.stdout or b"") if p.stdout else []
        for e in entries if isinstance(entries, list) else []:
            if not isinstance(e, dict):
                continue
            ps = e.get("PerformanceStatistics")
            if not isinstance(ps, dict):
                continue

            def stat(*names: str) -> float | None:
                for n in names:
                    v = ps.get(n)
                    if isinstance(v, (int, float)):
                        return float(v)
                return None

            cores = e.get("gpu-core-count")
            name = str(e.get("model") or e.get("IOClass") or "gpu")
            gpus.append({
                "card": name + (" (" + str(cores) + "-core)" if cores else ""),
                # "Device Utilization %" is the whole device. The renderer and
                # tiler figures beside it are parts of it, not extra load, so
                # they are alternatives here and never a sum.
                "busy_percent": stat("Device Utilization %",
                                     "GPU Activity(%)",
                                     "Renderer Utilization %"),
                "vram_total": None,   # filled in below
                # Bytes the driver currently has mapped for the GPU. On
                # unified memory this is the honest analogue of VRAM in use:
                # it is what a loaded model is actually occupying.
                "vram_used": (int(stat("In use system memory",
                                       "Alloc system memory") or 0)
                              if stat("In use system memory",
                                      "Alloc system memory") is not None
                              else None),
                # A GTT window is an amdgpu concept; there is no counterpart.
                "gtt_used": None,
                "gtt_total": None,
                "temp_c": None,
                "power_w": None,
                "sclk_mhz": None,
            })
    except Exception:  # noqa: BLE001 -- no IOKit answer is just no telemetry
        gpus = []
    if len(gpus) == 1:
        # Unified memory has no dedicated pool to read, so "how much of it may
        # the GPU hold" is a policy number, not a measurement: the Metal wired
        # limit when an operator has set one (mac-laptop-1 pins 42 GiB to stay a
        # usable laptop), otherwise the fleet spec sheet -- which is where
        # vram_total_bytes() already got the Macs' figure, so a box without
        # the cap set reports exactly what it reported before. The spec sheet
        # is passed IN (`spec_vram_gb`) rather than read from a fleet table:
        # the installer calls this on a box that has no fleet table yet.
        # Only with a single accelerator: on a two-GPU Mac neither of those
        # numbers belongs to any particular one of them.
        total = 0
        try:
            r = subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"],
                               capture_output=True, text=True, timeout=5)
            total = int((r.stdout or "0").strip() or 0) * 1024 ** 2
        except Exception:  # noqa: BLE001 -- not set, or an older kernel
            total = 0
        if total <= 0:
            total = int(float(spec_vram_gb or 0) * 1024 ** 3)
        gpus[0]["vram_total"] = total or None
    _macgpu_cache.update(t=time.time(), gpus=gpus)
    return gpus


def vram_total_bytes(spec_vram_gb: float | None = None, *,
                     amd: Callable[[], list[dict]] | None = None,
                     nv: Callable[[], list[dict]] | None = None,
                     win: Callable[[], list[dict]] | None = None,
                     mac: Callable[[], list[dict]] | None = None) -> int | None:
    """Dedicated video memory on this box: measured where a driver will say
    (amdgpu sysfs, nvidia-smi, or Windows' own performance counters),
    otherwise the fleet spec sheet -- which is how the Apple boxes, and any
    Windows box whose counters are unavailable, get an answer at all."""
    # Summed WITHIN a vendor, then the largest vendor wins -- not summed
    # across all of them. Two cards of the same kind really are one pool that
    # llama.cpp will split a model across (gpu-desktop-2's pair of 3090s). A
    # hybrid laptop is the opposite case: one reported a 512 MB amdgpu
    # carve-out beside an 8 GB RTX 4070, and adding them claims 6% more
    # VRAM than CUDA can ever touch. This number sizes the context windows the
    # box then PROMISES to the fleet, so overstating it is the one direction
    # that OOMs a load.
    # Windows' counters are the fallback here too, for the same reason
    # gpu_cards() treats them that way -- see there.
    amd_fn = amd or amdgpu_stats
    nv_fn = nv or nvidia_stats
    win_fn = win or windows_gpu_stats
    mac_fn = mac or (lambda: darwin_gpu_stats(spec_vram_gb))
    amd, nv = amd_fn(), nv_fn()
    # darwin_gpu_stats() already resolves the Macs' figure the way this
    # function used to (Metal wired limit, else the spec sheet), so it belongs
    # in the same fallback slot as Windows' counters rather than beside them.
    pools = ([amd, nv] if (amd or nv)
             else [win_fn() or mac_fn()])
    total = max(
        (sum(int(c.get("vram_total") or 0) for c in stats) for stats in pools),
        default=0)
    if total:
        return total
    if spec_vram_gb:
        return int(float(spec_vram_gb) * 1024 ** 3)
    return None


def _os_name() -> str:
    """Human name for the running OS: the distro on Linux, the product version
    on macOS, whatever platform can tell us anywhere else."""
    sysname = platform.system()
    if sysname == "Linux":
        try:
            rel = {}
            for line in Path("/etc/os-release").read_text().splitlines():
                k, _, v = line.partition("=")
                rel[k] = v.strip().strip('"')
            if rel.get("PRETTY_NAME") or rel.get("NAME"):
                return rel.get("PRETTY_NAME") or rel["NAME"]
        except OSError:
            pass
    elif sysname == "Darwin":
        return ("macOS " + platform.mac_ver()[0]).strip()
    elif sysname == "Windows":
        # platform.release() still answers "10" on Windows 11 -- the build
        # number in win32_ver() is the only thing that tells them apart, and 11
        # starts at 22000.
        rel, build = platform.release(), _win_build()
        if rel == "10" and build >= 22000:
            rel = "11"
        return ("Windows " + rel).strip()
    return (sysname + " " + platform.release()).strip()


def _win_build() -> int:
    try:
        return int(platform.win32_ver()[1].split(".")[-1])
    except (ValueError, IndexError):
        return 0


_os_cache: dict[str, Any] = {}


def os_info() -> dict:
    """Distro, kernel, arch. None of it changes while the process lives."""
    if not _os_cache:
        # "kernel" is the release string everywhere it means something; on
        # Windows that is the same "10" the name already corrected, so give the
        # build instead, which is what anyone comparing two Windows boxes wants.
        kernel = platform.release()
        if platform.system() == "Windows":
            kernel = "build " + str(_win_build() or platform.win32_ver()[1])
        _os_cache.update(
            name=_os_name(),
            kernel=kernel,
            arch=platform.machine(),
            platform=platform.system(),
            python=platform.python_version(),
        )
    return _os_cache


# --------------------------------------------------------------------------
# facts the installer needs and the gateway never did
# --------------------------------------------------------------------------
# Everything above this line was lifted out of app.py unchanged. Everything
# below it exists because `fleetctl detect` has to answer questions a running
# gateway never asks -- which package manager, how much RAM, which llama.cpp
# backend this box can actually build -- and those belong beside the readings
# they are derived from rather than in a second detection module.


def os_release() -> dict[str, str]:
    """/etc/os-release as a dict, empty off Linux.

    /usr/lib/os-release is the fallback the standard actually specifies, and
    it is the only one present in some minimal containers -- which is exactly
    where the distro matrix runs, so getting this wrong would make every
    container in CI report "unknown".
    """
    for p in (Path("/etc/os-release"), Path("/usr/lib/os-release")):
        try:
            text = p.read_text()
        except OSError:
            continue
        rel: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            k, _, v = line.partition("=")
            rel[k.strip()] = v.strip().strip('"').strip("'")
        if rel:
            return rel
    return {}


# A distro is only interesting here for the verb it wants: install a package.
# Grouping by that verb rather than by name is what keeps the matrix at four
# families instead of one branch per distribution -- Mint and Pop!_OS are
# Ubuntu with a different wallpaper as far as this code is concerned, and they
# say so themselves in ID_LIKE.
PKG_FAMILIES: dict[str, tuple[str, ...]] = {
    "apt": ("debian", "ubuntu", "linuxmint", "pop", "raspbian", "elementary",
            "zorin", "neon", "devuan", "kali"),
    "dnf": ("fedora", "rhel", "centos", "rocky", "almalinux", "ol", "amzn"),
    "pacman": ("arch", "archarm", "manjaro", "endeavouros", "cachyos", "garuda"),
    "zypper": ("opensuse", "opensuse-leap", "opensuse-tumbleweed", "sles",
               "suse", "opensuse-slowroll"),
}


def distro_family(rel: dict[str, str] | None = None) -> str:
    """apt | dnf | pacman | zypper | brew | windows | unknown.

    ID first, then ID_LIKE, then the binary that is actually on PATH. The
    last of those three is the one that saves a box nobody anticipated: a
    distro this table has never heard of still installs packages with one of
    four commands, and finding `pacman` is a better answer than "unknown".
    """
    sysname = platform.system()
    if sysname == "Darwin":
        return "brew"
    if sysname == "Windows":
        return "windows"
    rel = os_release() if rel is None else rel
    ident = (rel.get("ID") or "").lower()
    for family, ids in PKG_FAMILIES.items():
        if ident in ids:
            return family
    like = (rel.get("ID_LIKE") or "").lower().replace(",", " ").split()
    for token in like:
        for family, ids in PKG_FAMILIES.items():
            if token in ids:
                return family
    for family, probe in (("apt", "apt-get"), ("dnf", "dnf"),
                          ("pacman", "pacman"), ("zypper", "zypper")):
        if shutil.which(probe):
            return family
    return "unknown"


def os_family() -> str:
    """linux | darwin | windows -- the axis that decides the SHAPE of an
    install (where things live, what supervises them), as opposed to
    distro_family(), which only decides how packages get installed."""
    return {"Linux": "linux", "Darwin": "darwin",
            "Windows": "windows"}.get(platform.system(), platform.system().lower())


def ram_bytes() -> int | None:
    """Physical RAM, without psutil.

    Three readings for three platforms, and none of them is allowed to raise:
    a box whose RAM cannot be read is still a box worth provisioning, it just
    does not get RAM-derived defaults.
    """
    sysname = platform.system()
    if sysname == "Linux":
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
        return None
    if sysname == "Darwin":
        try:
            r = subprocess.run(["sysctl", "-n", "hw.memsize"],
                               capture_output=True, text=True, timeout=5)
            return int((r.stdout or "").strip())
        except Exception:  # noqa: BLE001
            return None
    if sysname == "Windows":
        try:
            # GlobalMemoryStatusEx via ctypes: no PowerShell start-up cost,
            # and unlike Win32_ComputerSystem.TotalPhysicalMemory it is not a
            # 32-bit field. wmic is gone from Windows 11 anyway.
            import ctypes

            class _MemStat(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _MemStat()
            st.dwLength = ctypes.sizeof(_MemStat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullTotalPhys)
        except Exception:  # noqa: BLE001
            return None
    return None


def cpu_model() -> str:
    """A human name for the CPU. Cosmetic -- it lands in host.yml so a plan
    can be read back and recognised months later -- so every branch degrades
    to platform.processor() rather than failing."""
    sysname = platform.system()
    try:
        if sysname == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith(("model name", "hardware")):
                    return line.split(":", 1)[1].strip()
        elif sysname == "Darwin":
            r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True, timeout=5)
            if (r.stdout or "").strip():
                return r.stdout.strip()
        elif sysname == "Windows":
            name = os_environ_processor()
            if name:
                return name
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or platform.machine()


def os_environ_processor() -> str:
    """PROCESSOR_IDENTIFIER, which Windows sets for every process. Split out
    so cpu_model() has nothing platform-specific to import at module scope."""
    import os as _os
    return (_os.environ.get("PROCESSOR_IDENTIFIER") or "").strip()


def gpu_vendors(cards: list[dict] | None = None) -> set[str]:
    """{'amd', 'nvidia', 'apple', 'intel'} -- whichever are present.

    Read off the card NAMES rather than off which backend answered, because
    on Windows one backend answers for all of them: windows_gpu_stats() is
    how a Radeon and a GeForce both get reported on a box with no vendor
    tooling, and "which backend replied" would call them both the same thing.
    """
    vendors: set[str] = set()
    for c in (gpu_cards() if cards is None else cards):
        name = str(c.get("card") or "").lower()
        if "nvidia" in name or "geforce" in name or "rtx" in name or "quadro" in name:
            vendors.add("nvidia")
        elif "radeon" in name or "amd" in name or name.startswith("card"):
            vendors.add("amd")
        elif "apple" in name or name.startswith("m1") or name.startswith("m2"):
            vendors.add("apple")
        elif "intel" in name or "arc" in name:
            vendors.add("intel")
    return vendors


def llama_backend(cards: list[dict] | None = None) -> str:
    """The llama.cpp backend this box should be built or fetched for.

    metal | cuda | vulkan | cpu. Vulkan rather than ROCm for AMD on purpose,
    and the fleet has the receipts: on gpu-desktop-1 (RDNA4, gfx1200) Vulkan needs
    nothing but the driver already installed, the build is 35 MB against
    ROCm's 197, and it is at parity or better for token generation. ROCm
    stays an explicit opt-in in host.yml, never a detected default.
    """
    if platform.system() == "Darwin":
        return "metal"
    cards = gpu_cards() if cards is None else cards
    vendors = gpu_vendors(cards)
    if "nvidia" in vendors:
        return "cuda"
    if "amd" in vendors or "intel" in vendors:
        # A GPU too small to hold anything is worse than no GPU: offloading to
        # a 512 MB carve-out means every token crosses the bus for a fraction
        # of a model. cpu-box-1's iGPU reports exactly that and is the fleet's
        # CPU-only fallback box. windows_gpu_stats() already discards adapters
        # under 2 GiB for the same reason; amdgpu sysfs does not, so the floor
        # is applied here where both paths meet.
        biggest = max((int(c.get("vram_total") or 0) for c in cards), default=0)
        if biggest and biggest < 2 * 1024 ** 3:
            return "cpu"
        return "vulkan"
    return "cpu"
