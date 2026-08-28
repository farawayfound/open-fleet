"""Three shapes and three engines: the tables the 15 host directories were.

Fifteen bespoke installers turned out to be three layouts and three engines
with per-box identity sprinkled through them. This file is the layouts and
the engines. `hosts/<name>/host.yml` is the identity. Nothing else should
have to exist.

The counting, taken off the real scripts:

  SHAPE           boxes                       supervisor      env file
  linux-systemd   apu-box-1 hub cpu-box-1       systemd units   /etc/llmstack/
                  server-1 gpu-laptop-1                           gateway.env
  macos-cron      mac-desktop mac-laptop-2 mac-laptop-1      crontab lines   ~/llmstack/
                                              (@reboot + a    gateway.env
                                              5-minute
                                              keepalive)
  windows-        apu-tablet-1 apu-tablet-2        SYSTEM          C:\\llmstack\\
  schtasks        mini-pc-1 gpu-desktop-2         scheduled       gateway.env.cmd
                  gpu-laptop-2 gpu-desktop-1               tasks

  ENGINE      boxes                                     upstream
  llama.cpp   apu-box-1 apu-tablet-1 apu-tablet-2 mac-laptop-2        127.0.0.1:8081
  (+swap)     mac-laptop-1 gpu-desktop-1 gpu-laptop-1                     (llama-swap)
  ollama      mac-desktop cpu-box-1 mini-pc-1 gpu-desktop-2      127.0.0.1:11434
              gpu-laptop-2 server-1
  none        hub                                    8081, where nothing
                                                        listens -- so an
                                                        unroutable model
                                                        fails loudly instead
                                                        of degrading quietly

Every value here is a DEFAULT. host.yml overrides any of them, and the
planner records which layer each value came from so `fleetctl plan --explain`
can say why a box got the number it did.
"""
from __future__ import annotations

import os
import platform
import re
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# the three shapes
# --------------------------------------------------------------------------
# Paths are templates against `prefix`, `home` and `state`. They are written
# with forward slashes and joined with os.path on the way out, so this table
# reads the same on all three platforms even though Windows will render its
# own separator.


def _linux(prefix: str, home: str) -> dict:
    # State lives outside the prefix on purpose: /opt is the application and
    # /var/lib is its data, so a reinstall of the first never endangers the
    # second. apu-box-1 goes further and gives /var/lib/llmstack its own ext4
    # partition, which is a host.yml override, not a different shape.
    return {
        "prefix": prefix,
        "gateway": f"{prefix}/gateway",
        "static": f"{prefix}/gateway/static",
        "bin": f"{prefix}/bin",
        "venv": f"{prefix}/venv",
        "state": "/var/lib/llmstack",
        "etc": "/etc/llmstack",
        "models": "/var/lib/llmstack/models",
        "envfile": "/etc/llmstack/gateway.env",
        "swap_config": "/etc/llmstack/llama-swap.yaml",
        "logs": "/var/log",
    }


def _darwin(prefix: str, home: str) -> dict:
    # ETC == STATE here, and on Windows too. A user-level install has no
    # /etc to own, and inventing one under the prefix would only mean two
    # directories that always move together.
    return {
        "prefix": prefix,
        "gateway": f"{prefix}/gateway",
        "static": f"{prefix}/gateway/static",
        "bin": f"{prefix}/bin",
        "venv": f"{prefix}/venv",
        "state": f"{prefix}/state",
        "etc": f"{prefix}/state",
        "models": f"{prefix}/state/models",
        "envfile": f"{prefix}/gateway.env",
        "swap_config": f"{prefix}/state/llama-swap.yaml",
        "logs": f"{prefix}/state",
    }


def _windows(prefix: str, home: str) -> dict:
    return {
        "prefix": prefix,
        "gateway": f"{prefix}/gateway",
        "static": f"{prefix}/gateway/static",
        "bin": f"{prefix}/bin",
        "venv": f"{prefix}/venv",
        "state": f"{prefix}/state",
        "etc": f"{prefix}/state",
        "models": f"{prefix}/models",
        # .cmd, not .env: the supervisor is a scheduled task running a batch
        # wrapper, and `call gateway.env.cmd` is how a batch file gets
        # variables. systemd reads KEY=VAL, a shell reads `export KEY=VAL`,
        # cmd reads `set KEY=VAL` -- three renderings of one table, see
        # envfile.py.
        "envfile": f"{prefix}/gateway.env.cmd",
        "swap_config": f"{prefix}/state/llama-swap.yaml",
        "logs": f"{prefix}/state",
    }


SHAPES: dict[str, dict] = {
    "linux": {
        "service": "systemd",
        "paths": _linux,
        # The gateway does not run as a person. It restarts llama-swap through
        # a narrow sudoers grant and nothing else -- see hosts/linux/grants.sh.
        "service_user": "llmstack",
        "units": ("llm-gateway", "llama-swap"),
        "needs_root": True,
        "wrappers": False,          # systemd has ExecStart + EnvironmentFile
    },
    "darwin": {
        "service": "cron",
        "paths": _darwin,
        "service_user": None,
        "units": ("llmstack-gateway", "llmstack-llama-swap"),
        "needs_root": False,
        "wrappers": True,           # run-gateway.sh / run-llama-swap.sh
    },
    "windows": {
        "service": "schtasks",
        "paths": _windows,
        "service_user": "SYSTEM",
        "units": ("llm-gateway", "llama-swap"),
        "needs_root": True,         # SYSTEM tasks and firewall rules
        "wrappers": True,           # run-gateway.cmd / run-llama-swap.cmd
    },
}


def paths_for(family: str, prefix: str, home: str | None = None) -> dict:
    # default_home() answers "" when this machine is not the kind being
    # planned for, and "" is the right answer to carry: every path here is
    # built off `prefix`, which the caller has already resolved.
    home = home if home is not None else default_home(family)
    shape = SHAPES.get(family)
    if not shape:
        raise KeyError(f"no shape for os family {family!r}")
    raw = shape["paths"](prefix.replace("\\", "/").rstrip("/"), home)
    if family == "windows":
        return {k: v.replace("/", "\\") for k, v in raw.items()}
    return raw


# --------------------------------------------------------------------------
# the three engines
# --------------------------------------------------------------------------
ENGINES: dict[str, dict] = {
    "llama.cpp": {
        "upstream": "http://127.0.0.1:8081",
        "swap_listen": "127.0.0.1:8081",
        "llama_swap": True,
        # The gateway enumerates its own models directory. Ollama's boxes are
        # the opposite case -- see below.
        "models_from_upstream": False,
        "binaries": ("llama-server", "llama-bench"),
    },
    "ollama": {
        "upstream": "http://127.0.0.1:11434",
        "swap_listen": None,
        "llama_swap": False,
        # Ollama owns its own store in a layout of its own, so the catalogue
        # is whatever Ollama says it has rather than whatever is on disk.
        "models_from_upstream": True,
        "binaries": ("ollama",),
    },
    "none": {
        # A hub routes and does not serve. Pointing upstream at a port nothing
        # listens on is deliberate: an unroutable model then fails loudly
        # instead of silently degrading to "no models here".
        "upstream": "http://127.0.0.1:8081",
        "swap_listen": None,
        "llama_swap": False,
        "models_from_upstream": False,
        "binaries": (),
    },
}

# llama.cpp release assets, as PATTERNS rather than names, per
# (os family, backend, arch).
#
# Patterns because the names carry versions that move: the CUDA and ROCm
# builds are `bin-win-cuda-13.3-x64.zip` and `bin-ubuntu-rocm-7.14-x64.tar.gz`
# today and will not be tomorrow. An exact-name table would need editing
# every time upstream bumped a toolkit.
#
# Vulkan is the default for AMD and Intel everywhere, and the reasoning is on
# gpu-desktop-1: the card needs nothing but the driver already installed, the archive
# is 35 MB against ROCm's 197, and the Vulkan backend is at parity or better
# for token generation on RDNA-class cards. ROCm stays an explicit host.yml
# opt-in.
#
# The gaps are as informative as the entries. There is no CUDA build for
# Linux and no Vulkan build for Windows on ARM, so those boxes have to build
# from source -- which the engine step says out loud rather than failing on a
# missing asset.
LLAMA_ASSETS: dict[tuple[str, str, str], str] = {
    ("windows", "vulkan", "AMD64"): r"^llama-.*-bin-win-vulkan-x64\.zip$",
    ("windows", "cuda", "AMD64"): r"^llama-.*-bin-win-cuda-[\d.]+-x64\.zip$",
    ("windows", "rocm", "AMD64"): r"^llama-.*-bin-win-rocm-[\d.]+-x64\.zip$",
    ("windows", "cpu", "AMD64"): r"^llama-.*-bin-win-cpu-x64\.zip$",
    ("windows", "cpu", "ARM64"): r"^llama-.*-bin-win-cpu-arm64\.zip$",
    ("linux", "vulkan", "x86_64"): r"^llama-.*-bin-ubuntu-vulkan-x64\.tar\.gz$",
    ("linux", "vulkan", "aarch64"): r"^llama-.*-bin-ubuntu-vulkan-arm64\.tar\.gz$",
    ("linux", "rocm", "x86_64"): r"^llama-.*-bin-ubuntu-rocm-[\d.]+-x64\.tar\.gz$",
    ("linux", "cpu", "x86_64"): r"^llama-.*-bin-ubuntu-x64\.tar\.gz$",
    ("linux", "cpu", "aarch64"): r"^llama-.*-bin-ubuntu-arm64\.tar\.gz$",
    ("darwin", "metal", "arm64"): r"^llama-.*-bin-macos-arm64\.tar\.gz$",
    ("darwin", "metal", "x86_64"): r"^llama-.*-bin-macos-x64\.tar\.gz$",
}

# The CUDA builds ship without the CUDA runtime; it is a second archive that
# unpacks beside llama-server.exe. Without it the binary loads and then dies
# on a missing cudart64_*.dll, which reads as a broken build rather than a
# missing dependency.
#
# Upstream publishes one per toolkit version (12.4 and 13.3 today), and the
# runtime has to MATCH the build -- so the pattern is derived from the
# binary's own name rather than fixed. A 13.3 runtime beside a 12.4 build is
# the same missing-DLL failure with 400 MB of download in front of it.
LLAMA_CUDA_RUNTIME = r"^cudart-llama-bin-win-cuda-[\d.]+-x64\.zip$"

_CUDA_VER = re.compile(r"-cuda-([\d.]+)-")


def cuda_runtime_for(asset_name: str) -> str:
    """The runtime pattern matching one CUDA build archive."""
    m = _CUDA_VER.search(asset_name)
    if not m:
        return LLAMA_CUDA_RUNTIME
    return r"^cudart-llama-bin-win-cuda-" + re.escape(m.group(1)) + r"-x64\.zip$"

LLAMA_SWAP_ASSETS: dict[tuple[str, str], str] = {
    ("linux", "x86_64"): "linux_amd64",
    ("linux", "aarch64"): "linux_arm64",
    ("darwin", "arm64"): "darwin_arm64",
    ("darwin", "x86_64"): "darwin_amd64",
    ("windows", "AMD64"): "windows_amd64",
    ("windows", "ARM64"): "windows_arm64",
}


def llama_asset(family: str, backend: str, arch: str) -> str | None:
    """The release-asset pattern for this box, or None if upstream publishes
    no build for it (Linux + CUDA, Windows on ARM + Vulkan)."""
    return LLAMA_ASSETS.get((family, backend or "cpu", arch or ""))


# --------------------------------------------------------------------------
# packages, by family rather than by distro
# --------------------------------------------------------------------------
# The build-from-source list is only needed where llama.cpp is compiled
# (apu-box-1 and gpu-laptop-1 do; every other box takes a release binary), so it is
# kept apart -- a box that only runs the gateway should not be made to
# install a C++ toolchain to get there.
BASE_PACKAGES: dict[str, tuple[str, ...]] = {
    "apt": ("python3", "python3-venv", "python3-pip", "curl", "jq", "rsync",
            "ca-certificates"),
    "dnf": ("python3", "python3-pip", "curl", "jq", "rsync", "ca-certificates"),
    "pacman": ("python", "python-pip", "curl", "jq", "rsync", "ca-certificates"),
    # No bare `python3` here: on Leap 15.6 that package is Python 3.6, which
    # install.sh refuses -- it installs a versioned python31x instead, and a
    # packages step that then chased `python3` would report drift forever.
    "zypper": ("curl", "jq", "rsync", "ca-certificates"),
    "brew": ("python@3.12",),
    "windows": (),
}

BUILD_PACKAGES: dict[str, tuple[str, ...]] = {
    # `spirv-headers-devel glslang` on the dnf side is not padding: llama.cpp's
    # Vulkan backend does find_package(SPIRV-Headers) and wants
    # glslangValidator beside glslc, and without them the cmake configure step
    # fails AFTER it has already reported "Found Vulkan" -- which reads as a
    # Vulkan problem and is not. See hosts/gpu-laptop-1/bootstrap/00-base.sh.
    "apt": ("git", "cmake", "ninja-build", "g++", "pkg-config", "libcurl4-openssl-dev",
            "libssl-dev", "libvulkan-dev", "vulkan-tools", "glslc",
            "spirv-tools", "glslang-tools", "mesa-vulkan-drivers"),
    "dnf": ("git", "cmake", "ninja-build", "gcc-c++", "pkgconf-pkg-config",
            "libcurl-devel", "openssl-devel", "vulkan-loader-devel", "vulkan-headers",
            "vulkan-tools", "mesa-vulkan-drivers", "glslc", "spirv-tools",
            "spirv-headers-devel", "glslang", "glslang-devel"),
    "pacman": ("git", "cmake", "ninja", "gcc", "pkgconf", "curl", "openssl",
               "vulkan-headers", "vulkan-icd-loader", "vulkan-tools", "shaderc",
               "spirv-tools", "glslang"),
    "zypper": ("git", "cmake", "ninja", "gcc-c++", "pkg-config", "libcurl-devel",
               "libopenssl-devel", "vulkan-devel", "vulkan-tools", "shaderc",
               "spirv-tools", "glslang-devel"),
    "brew": ("cmake", "ninja"),
    "windows": (),
}

# The verb, per family. `install` is a check-then-install: a package manager
# that reports "already installed" and exits non-zero would fail a step that
# has nothing wrong with it, so the step checks first (see steps/packages.py).
PKG_INSTALL: dict[str, tuple[str, ...]] = {
    "apt": ("apt-get", "install", "-y", "--no-install-recommends"),
    "dnf": ("dnf", "install", "-y"),
    "pacman": ("pacman", "-S", "--needed", "--noconfirm"),
    "zypper": ("zypper", "--non-interactive", "install", "--no-recommends"),
    "brew": ("brew", "install", "-q"),
}

PKG_QUERY: dict[str, tuple[str, ...]] = {
    "apt": ("dpkg-query", "-W", "-f=${Status}"),
    "dnf": ("rpm", "-q"),
    "pacman": ("pacman", "-Q"),
    "zypper": ("rpm", "-q"),
    "brew": ("brew", "list", "--versions"),
}

PKG_REFRESH: dict[str, tuple[str, ...]] = {
    "apt": ("apt-get", "update"),
    "dnf": ("dnf", "makecache"),
    "pacman": ("pacman", "-Sy"),
    "zypper": ("zypper", "--non-interactive", "refresh"),
    "brew": ("brew", "update"),
}

# Homebrew is not on PATH in a non-interactive shell.
#
# `ssh mac-desktop brew list` finds nothing and says nothing: /opt/homebrew/bin
# is put on PATH by the shellenv line in the login profile, which a command
# run over ssh never sources. So the query above came back empty on every Mac
# and the packages step reported `python@3.12 missing` for a package that has
# been installed the whole time -- and an apply would then have run a `brew`
# that does not resolve either. facts.py already searches these two prefixes
# for tailscale and for ollama, and engine.py already does it for brew
# itself; this is that same answer, in the one place the tables are read.
BREW_CANDIDATES = ("/opt/homebrew/bin/brew", "/usr/local/bin/brew")


def brew_bin() -> str:
    """Homebrew's absolute path, or the bare name if it is somewhere else."""
    for candidate in BREW_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return "brew"


def pkg_argv(family: str, table: dict[str, tuple[str, ...]]) -> list[str] | None:
    """A command out of one of the PKG_* tables, made runnable.

    Only brew needs the treatment -- apt, dnf, pacman and zypper are in
    /usr/bin on every box that has them, and are on the PATH of even the
    barest shell.
    """
    argv = table.get(family)
    if not argv:
        return None
    out = list(argv)
    if family == "brew":
        out[0] = brew_bin()
    return out


# --------------------------------------------------------------------------
# the env file, in three renderings
# --------------------------------------------------------------------------
# One table of LLMSTACK_* variables, written three ways. This is the single
# largest source of per-box duplication in the old installers: every one of
# the fifteen carried its own copy of this heredoc, and the values that
# genuinely differed between them were a handful of lines in the middle.
ENV_STYLE = {
    "linux": "{k}={v}",         # systemd EnvironmentFile: bare KEY=VALUE
    "darwin": "export {k}={v}",  # sourced by a shell wrapper
    "windows": "set {k}={v}",   # called by a .cmd wrapper
}

ENV_COMMENT = {"linux": "# {t}", "darwin": "# {t}", "windows": "rem {t}"}


def env_lines(family: str, values: dict[str, Any],
              notes: dict[str, str] | None = None) -> list[str]:
    """Render the env table for one OS family.

    `None` values are dropped rather than written empty: on Windows
    `set X=` and on Linux `X=` are not the same thing as absent, and the
    gateway distinguishes an unset LLMSTACK_AVAILABILITY_FILE (never gated)
    from an empty one.
    """
    style = ENV_STYLE[family]
    cstyle = ENV_COMMENT[family]
    notes = notes or {}
    out: list[str] = []
    for key, value in values.items():
        if value is None:
            continue
        if key in notes:
            if out and out[-1].strip():
                out.append("")
            for line in notes[key].rstrip().split("\n"):
                out.append(cstyle.format(t=line) if line else cstyle.format(t="").rstrip())
        if value is True:
            value = "1"
        elif value is False:
            value = "0"
        elif isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        out.append(style.format(k=key, v=value))
    return out


def join(family: str, *parts: str) -> str:
    """Join path parts the way the TARGET box would, not the way this one
    does. `fleetctl plan` for a Windows box is frequently run from Linux CI."""
    sep = "\\" if family == "windows" else "/"
    cleaned = [str(p).replace("\\", "/").rstrip("/") for p in parts if p not in (None, "")]
    joined = "/".join(cleaned)
    return joined.replace("/", sep) if family == "windows" else joined


def default_home(family: str) -> str:
    """This machine's home directory -- but ONLY when this machine is the
    kind of machine being planned for.

    Planning for a Mac from a Windows workstation used to answer
    `C:/Users/user` here, and that string went into a committed host.yml as
    `paths.prefix: C:/Users/user/llmstack`. The dry run on the Mac then
    offered to create it. So a cross-platform plan gets "" instead, and the
    caller has to have been told the home by `detect` (facts["home"]) or say
    so explicitly.
    """
    running = {"Linux": "linux", "Darwin": "darwin",
               "Windows": "windows"}.get(platform.system(), "")
    if family != running:
        return ""
    if family == "windows":
        return os.environ.get("USERPROFILE", str(Path.home()))
    return str(Path.home())
