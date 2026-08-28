"""What is actually in front of us: `fleetctl detect`.

Everything here is READ-ONLY and must stay that way. `detect` is the command
a person runs on a box they are not sure about, and on the CI runners and the
distro containers where nothing is ever installed -- if it could change the
machine it would not be safe to run in either place.

The hardware readings come from gateway/hw.py, imported by path (see
`hwmod()`) rather than copied, so the installer and the running gateway
cannot disagree about what a box is. Everything else -- the service manager,
the interpreter, the tailnet address, which engine is already here -- is
detected in this file, because a running gateway has no reason to ask.

Nothing raises. A box that will not answer one question is still a box worth
provisioning; the fact comes back None and the planner falls back to a
default or asks for the value in host.yml.
"""
from __future__ import annotations

import importlib.util
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

_hw = None


def hwmod():
    """gateway/hw.py, wherever this checkout (or this install) keeps it.

    Imported by path rather than `from gateway import hw`: `gateway` is not a
    package, and on a provisioned box the file lives at
    <prefix>/gateway/hw.py with no repo around it at all. Both layouts are
    tried, plus FLEETCTL_HW for a test that wants to point somewhere else.
    """
    global _hw
    if _hw is not None:
        return _hw
    here = Path(__file__).resolve().parent
    candidates = [
        Path(os.environ["FLEETCTL_HW"]) if os.environ.get("FLEETCTL_HW") else None,
        here.parent / "gateway" / "hw.py",          # a repo checkout
        here.parent / "hw.py",                      # fleetctl beside the gateway
        Path("/opt/llmstack/gateway/hw.py"),        # a provisioned Linux box
        Path.home() / "llmstack" / "gateway" / "hw.py",       # macOS
        Path("C:/llmstack/gateway/hw.py"),          # Windows
    ]
    for cand in candidates:
        if cand and cand.is_file():
            spec = importlib.util.spec_from_file_location("llmstack_hw", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)          # type: ignore[union-attr]
            _hw = mod
            return _hw
    raise FileNotFoundError(
        "cannot find gateway/hw.py -- looked in: "
        + ", ".join(str(c) for c in candidates if c))


# --------------------------------------------------------------------------
# the interpreter question
# --------------------------------------------------------------------------
# The gateway needs 3.10+ (it is written in `X | None` syntax throughout), and
# the fleet keeps finding boxes whose obvious python is not usable:
#
#   macOS      /usr/bin/python3 is 3.9. Every Mac here pins a Homebrew 3.12.
#   Windows    `python` and `python3` on PATH are frequently the Microsoft
#              Store stub, which prints an advert and exits 49. Worse, the
#              only real interpreter is often a per-user install under
#              C:\Users\<name>, and a SYSTEM scheduled task cannot depend on
#              anything inside a user profile -- gpu-desktop-1's did, and the profile
#              was then renamed out from under it.
#   Linux      usually fine, but a distro container has no `python3` at all
#              until the packages step runs.
MIN_PYTHON = (3, 10)


def _python_version(exe: str) -> tuple[int, ...] | None:
    try:
        r = subprocess.run([exe, "-c", "import sys;print('%d.%d.%d' % sys.version_info[:3])"],
                           capture_output=True, text=True, timeout=20)
    except Exception:  # noqa: BLE001
        return None
    # The Store stub exits non-zero with an advert on stdout; a real
    # interpreter answers three integers and nothing else.
    if r.returncode != 0:
        return None
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)\s*$", r.stdout.strip())
    return tuple(int(x) for x in m.groups()) if m else None


def _python_candidates() -> list[str]:
    # The running interpreter first -- unless it is a virtualenv's. A plan
    # written from an activated venv would pin paths.python at a disposable
    # `.venv/Scripts/python.exe`, and the SYSTEM task or unit that later runs
    # the gateway would find it gone. The venv's base interpreter is the
    # stable one.
    exe = sys.executable
    if sys.prefix != sys.base_prefix:
        exe = getattr(sys, "_base_executable", "") or ""
    out: list[str] = [exe] if exe else []
    if platform.system() == "Windows":
        out += [r"C:\Program Files\Python313\python.exe",
                r"C:\Program Files\Python312\python.exe",
                r"C:\Program Files\Python311\python.exe"]
    elif platform.system() == "Darwin":
        # Apple Silicon first, then Intel's Homebrew prefix.
        for pref in ("/opt/homebrew", "/usr/local"):
            out += [f"{pref}/bin/python3.13", f"{pref}/bin/python3.12",
                    f"{pref}/bin/python3.11"]
    else:
        out += ["/usr/bin/python3.13", "/usr/bin/python3.12", "/usr/bin/python3.11"]
    for name in ("python3.13", "python3.12", "python3.11", "python3", "python"):
        found = shutil.which(name)
        if found:
            out.append(found)
    seen: set[str] = set()
    uniq = []
    for exe in out:
        if exe and exe not in seen:
            seen.add(exe)
            uniq.append(exe)
    return uniq


def find_python() -> dict:
    """The best interpreter on this box, and every candidate that was tried.

    Reporting the rejects matters as much as the answer: "no usable python"
    on a Windows box is nearly always the Store stub, and a plan that says
    which paths it looked at turns that into a one-line diagnosis.
    """
    tried: list[dict] = []
    best: dict | None = None
    for exe in _python_candidates():
        ver = _python_version(exe)
        ok = bool(ver and ver >= MIN_PYTHON)
        tried.append({"exe": exe,
                      "version": ".".join(map(str, ver)) if ver else None,
                      "usable": ok})
        if ok and best is None:
            best = tried[-1]
    return {"exe": best["exe"] if best else None,
            "version": best["version"] if best else None,
            "min": ".".join(map(str, MIN_PYTHON)),
            "candidates": tried}


def this_python() -> dict:
    """The running interpreter, in the shape find_python() returns.

    `--quick` exists to skip the readings that shell out, and find_python()
    runs every candidate to ask its version. This one asks nothing -- but it
    still has to carry `min`, or `detect --quick` greets a first-time reader
    with "need >= None" in the first command the docs tell them to run. The
    venv rule from _python_candidates applies here too: a disposable
    `.venv/…/python` is not the interpreter a service should be pointed at.
    """
    exe = sys.executable
    if sys.prefix != sys.base_prefix:
        exe = getattr(sys, "_base_executable", "") or exe
    return {"exe": exe, "version": platform.python_version(),
            "min": ".".join(map(str, MIN_PYTHON)), "candidates": []}


# --------------------------------------------------------------------------
# the rest
# --------------------------------------------------------------------------
def service_manager() -> str:
    """systemd | cron | schtasks | none.

    Not read off the OS name: a Linux container has no systemd (which is
    exactly where the CI distro matrix runs), and calling it "systemd" there
    would make every plan in CI claim units it cannot install. `systemctl` on
    PATH *and* a live /run/systemd/system is the honest test -- the second
    half is what the container fails.
    """
    sysname = platform.system()
    if sysname == "Windows":
        return "schtasks" if shutil.which("schtasks") else "none"
    if sysname == "Linux":
        if shutil.which("systemctl") and Path("/run/systemd/system").is_dir():
            return "systemd"
        return "cron" if shutil.which("crontab") else "none"
    if sysname == "Darwin":
        # launchd is here, but a LaunchAgent cannot be bootstrapped over ssh
        # without an Aqua session -- which is why all three Macs on this fleet
        # are supervised by cron instead. See hosts/mac-laptop-2/bootstrap.sh.
        return "cron" if shutil.which("crontab") else "none"
    return "none"


def is_root() -> bool:
    if platform.system() == "Windows":
        try:
            import ctypes

            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:  # noqa: BLE001
            return False
    return hasattr(os, "geteuid") and os.geteuid() == 0


def can_sudo() -> bool:
    """Passwordless sudo, without prompting for anything.

    `sudo -n true` and nothing else: a prompt here would hang a deploy, and
    the fleet has boxes on both sides of this (mac-laptop-2 answers, mac-laptop-1 and
    mac-desktop do not, and their Metal cap is a hand-run step because of it).
    """
    if platform.system() == "Windows" or is_root():
        return is_root()
    if not shutil.which("sudo"):
        return False
    try:
        return subprocess.run(["sudo", "-n", "true"], capture_output=True,
                              timeout=15).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def lan_ipv4() -> str | None:
    """The address this box would use to reach its LAN.

    A DHCP lease, so only ever a FALLBACK for the URL clients are handed --
    a box with no tailnet still needs some address, and "required and
    nothing supplied it" is not an install. No packet is sent: connecting a
    UDP socket only asks the kernel which interface it would route through.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return None
    return None if not ip or ip.startswith("127.") else ip


def tailscale() -> dict:
    """Address and tailnet name, if this box is on a tailnet.

    The address is what matters. Every deploy path on this fleet addresses
    boxes by 100.x rather than by name, because names are not stable: the
    2026-08-25 rename of m1-laptop retired the MagicDNS name every ssh config
    pointed at, and the box read as offline in CI for a day while being
    perfectly reachable by address.
    """
    # PATH first, then the places it actually lives. The Homebrew prefixes
    # are not padding: an ssh session to a Mac gets a non-login shell whose
    # PATH does not include /opt/homebrew/bin, so `which tailscale` comes back
    # empty on a box that is very much on the tailnet -- measured on mac-desktop,
    # where missing this made the plan derive no public_api_url at all.
    exe = shutil.which("tailscale") or next(
        (p for p in (r"C:\Program Files\Tailscale\tailscale.exe",
                     "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
                     "/opt/homebrew/bin/tailscale", "/usr/local/bin/tailscale",
                     "/usr/bin/tailscale", "/usr/sbin/tailscale")
         if Path(p).exists()), None)
    if not exe:
        return {"present": False, "ipv4": None, "name": None}
    out: dict[str, Any] = {"present": True, "ipv4": None, "name": None}
    try:
        r = subprocess.run([exe, "ip", "-4"], capture_output=True, text=True, timeout=20)
        first = (r.stdout or "").strip().splitlines()
        if first:
            out["ipv4"] = first[0].strip()
    except Exception:  # noqa: BLE001
        pass
    try:
        r = subprocess.run([exe, "status", "--json"], capture_output=True,
                           text=True, timeout=20)
        import json as _json

        self_ = (_json.loads(r.stdout or "{}").get("Self") or {})
        dns = str(self_.get("DNSName") or "")
        if dns:
            out["name"] = dns.split(".")[0]
    except Exception:  # noqa: BLE001
        pass
    return out


def ollama_store(*candidates: str) -> str | None:
    """Where Ollama actually keeps its weights, which is not one place.

    The Linux installer creates a SYSTEM account and stores under
    /usr/share/ollama; the Mac app and a manual install use the calling
    user's home.

    Existence is not the test, which is what this used to use. All of these
    paths get created whether or not anything is put in them, so "the first
    that exists" picked /usr/share/ollama/.ollama/models on cpu-box-1 -- an
    empty directory -- while the box's actual weights sit in
    /var/lib/llmstack/models. Pointing the gateway at the empty one would
    have advertised a store with nothing in it beside a full one, which is
    the exact failure the old comment claimed to be avoiding.

    So: a store with something in it wins, and mere existence is only the
    fallback for a box that has Ollama installed and has pulled nothing yet.
    """
    fallback = None
    for path in candidates:
        if not path:
            continue
        try:
            if next(Path(path).iterdir(), None) is not None:
                return path
        except NotADirectoryError:
            continue
        except OSError:
            continue
        if fallback is None and Path(path).exists():
            fallback = path
    return fallback


def engines_present() -> dict:
    """Which inference engines are already on this box.

    A fresh install picks its engine from the hardware; a re-run must not
    quietly replace one that is already working, and a box that is only being
    updated needs to know which of the two upstream ports to expect.
    """
    def first(*paths: str) -> str | None:
        for p in paths:
            if p and Path(p).exists():
                return p
        return None


    home = Path.home()
    return {
        "llama_server": shutil.which("llama-server") or first(
            "/opt/llmstack/bin/llama-server", str(home / "llmstack/bin/llama-server"),
            r"C:\llmstack\bin\llama\llama-server.exe"),
        "llama_swap": shutil.which("llama-swap") or first(
            "/opt/llmstack/bin/llama-swap", str(home / "llmstack/bin/llama-swap"),
            r"C:\llmstack\bin\llama-swap.exe"),
        "ollama": shutil.which("ollama") or first(
            "/usr/local/bin/ollama", "/opt/homebrew/bin/ollama",
            r"C:\Users\Public\ollama\ollama.exe"),
        "ollama_models": ollama_store(
            "/usr/share/ollama/.ollama/models",
            "/var/lib/ollama/.ollama/models",
            str(home / ".ollama" / "models")),
        "lmstudio": first(str(home / ".lmstudio"), str(home / ".cache/lm-studio")),
    }


def existing_install(prefix: Path) -> dict:
    """What a previous provision left behind, if anything.

    DEPLOYED_SHA is deploy-gateway.sh's stamp; reading it here is what lets
    `fleetctl verify` say "this box is three commits behind" without a hub.
    """
    gw = prefix / "gateway"
    sha = None
    try:
        sha = (gw / "DEPLOYED_SHA").read_text(encoding="utf-8").strip() or None
    except OSError:
        pass
    return {
        "prefix": str(prefix),
        "present": (gw / "app.py").is_file(),
        "has_hw": (gw / "hw.py").is_file(),
        "deployed_sha": sha,
    }


def default_prefix() -> str:
    """Where this OS keeps the stack. The three shapes, in one place.

    Linux gets /opt with state in /var/lib because the gateway runs as its own
    service account; macOS and Windows are single-user boxes where the whole
    stack lives under one root and the supervisor (cron, or a SYSTEM task)
    reads it from there.
    """
    sysname = platform.system()
    if sysname == "Windows":
        return r"C:\llmstack"
    if sysname == "Darwin":
        return str(Path.home() / "llmstack")
    return "/opt/llmstack"


def gather(quick: bool = False) -> dict:
    """Every fact, as one dict. `quick` skips the readings that shell out.

    The order is the order a person would ask them in, because this dict is
    what `fleetctl detect` prints.
    """
    hw = hwmod()
    rel = hw.os_release()
    cards = [] if quick else hw.gpu_cards()
    vram = None if quick else hw.vram_total_bytes()
    ram = hw.ram_bytes()

    facts: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "lan_ipv4": lan_ipv4(),
        # The home directory of the account the stack will live under, on the
        # USER-LEVEL shapes. Detected rather than computed by the planner,
        # because a plan is frequently built somewhere other than the box it
        # is for: generating mac-laptop-1's plan on a Windows workstation put
        # `C:/Users/user/llmstack` into a macOS host.yml, and the dry run on
        # the Mac duly offered to create it.
        "home": str(Path.home()),
        "os": {
            "family": hw.os_family(),
            "name": hw.os_info()["name"],
            "kernel": hw.os_info()["kernel"],
            "arch": hw.os_info()["arch"],
            "distro_id": rel.get("ID") or "",
            "distro_version": rel.get("VERSION_ID") or "",
            "distro_like": rel.get("ID_LIKE") or "",
            "package_manager": hw.distro_family(rel),
        },
        "service_manager": service_manager(),
        "privilege": {"root": is_root(), "sudo_nopasswd": False if quick else can_sudo()},
        "python": this_python() if quick else find_python(),
        "cpu": {"model": hw.cpu_model(), "count": os.cpu_count()},
        "ram_bytes": ram,
        "ram_gb": round(ram / 1024 ** 3, 1) if ram else None,
        "gpus": cards,
        "vram_bytes": vram,
        "vram_gb": round(vram / 1024 ** 3, 1) if vram else None,
        "llama_backend": hw.llama_backend(cards) if not quick else None,
        "tailscale": {"present": False, "ipv4": None, "name": None} if quick
        else tailscale(),
        "engines": engines_present(),
        "install": existing_install(Path(default_prefix())),
    }
    return facts
