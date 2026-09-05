"""The inference engine: llama.cpp behind llama-swap, or Ollama.

The engine is the one part of an install that is genuinely different per box
rather than per shape -- a Metal bottle on a Mac, a Vulkan release archive on
Windows, a source build where the box has a card no release targets. What is
NOT different is the decision procedure, and that is what lives here.

Nothing in this file downloads anything during check(). A dry run on a box
with no network still reports honestly on what is installed.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

from .. import shapes
from . import BLOCKED, MISSING, OK, Check, Step

# /releases, NOT /releases/latest.
#
# llama.cpp tags every build as a PRE-RELEASE (b10661, b10660, ...), and
# GitHub's "latest" deliberately skips prereleases -- so /releases/latest
# answers with `v0.3.0`, whose only asset is a text file naming the nightly
# tag. Every installer in this repo that asked for `latest` and then filtered
# for `bin-win-vulkan-x64.zip` was therefore going to throw
# "no asset in release v0.3.0" the next time it ran. Listing releases and
# taking the newest one that actually HAS the asset we want is both the fix
# and the thing that stays correct when upstream changes its mind again.
LLAMA_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15"
SWAP_RELEASES = "https://api.github.com/repos/mostlygeek/llama-swap/releases?per_page=10"


def _releases(url: str, timeout: int = 40) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "fleetctl"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:  # noqa: S310
        data = json.loads(fh.read().decode("utf-8"))
    return data if isinstance(data, list) else [data]


def _find(url: str, pattern: str) -> tuple[dict, dict]:
    """The newest release carrying an asset matching `pattern`, and the asset.

    Drafts are skipped (their assets are not downloadable); prereleases are
    not, because on llama.cpp they are the only releases that matter.
    """
    rx = re.compile(pattern)
    seen: list[str] = []
    for rel in _releases(url):
        if rel.get("draft"):
            continue
        for asset in rel.get("assets", []):
            if rx.search(asset.get("name", "")):
                return rel, asset
        seen.append(str(rel.get("tag_name")))
    raise RuntimeError(
        f"no asset matching /{pattern}/ in the last {len(seen)} releases "
        f"({', '.join(seen[:6])}{'...' if len(seen) > 6 else ''})")


class Engine(Step):
    id = "engine"
    title = "inference engine"

    def wanted(self, ctx) -> bool:
        return ctx.plan["engine"]["kind"] != "none"

    # -- what we expect to find --------------------------------------------
    def _binaries(self, ctx) -> dict[str, str]:
        kind = ctx.plan["engine"]["kind"]
        p = ctx.plan["paths"]
        ext = ".exe" if ctx.family == "windows" else ""
        if kind == "ollama":
            return {"ollama": shapes.join(ctx.family, p["bin"], "ollama" + ext)}
        # The plan's paths, which are the fleet prefix on a box fleetctl
        # installed and wherever the operator built it on a box that did its
        # own. Looking only under `bin` reported server-1's working
        # source-built engine as missing.
        out = {"llama-server": p["llama_server"]}
        if ctx.plan["engine"]["llama_swap"]:
            out["llama-swap"] = p["llama_swap"]
        return out

    def check(self, ctx) -> Check:
        kind = ctx.plan["engine"]["kind"]
        backend = ctx.plan["engine"]["backend"]
        present = ctx.facts.get("engines") or {}
        want = self._binaries(ctx)

        if kind == "ollama":
            if present.get("ollama"):
                return Check(OK, f"ollama at {present['ollama']}")
            return Check(MISSING, "ollama is not installed",
                         ["install Ollama (https://ollama.com/download), then set "
                          "OLLAMA_HOST=127.0.0.1 so only the gateway can reach it"])

        missing = [name for name, path in want.items()
                   if not ctx.path(path).exists() and not present.get(
                       name.replace("-", "_"))]
        if not missing:
            detail = f"llama.cpp ({backend})"
            server = want.get("llama-server")
            r = ctx.probe([str(ctx.path(server)), "--version"]) if server else None
            if r is not None:
                m = re.search(r"version:\s*(\S+)", (r.stdout or "") + (r.stderr or ""))
                if m:
                    detail += f" build {m.group(1)}"
            return Check(OK, detail)

        # An engine the operator put somewhere of their own is not something
        # to install over. fleetctl unpacks releases into `bin`; if the plan
        # points llama-server outside it, unpacking would leave the planned
        # path exactly as missing as it started -- an apply that runs clean
        # and changes nothing, which is the failure mode `UNCHANGED` exists
        # to catch. Say so up front instead.
        outside = [name for name, path in want.items()
                   if name in missing and not str(path).startswith(
                       str(ctx.plan["paths"]["bin"]))]
        if outside:
            return Check(BLOCKED,
                         "planned outside the fleet prefix and not there: "
                         + ", ".join(f"{n} at {want[n]}" for n in outside),
                         ["install it at that path by hand, or drop "
                          "paths.llama_server / paths.llama_swap from this "
                          "box's host.yml to take the fleet's own copy"])

        arch = ctx.plan["platform"]["arch"] or ""
        if ctx.family == "darwin":
            steps = ["brew install llama.cpp (Metal bottle)"]
        elif ctx.plan["engine"].get("build_from_source"):
            steps = [f"build llama.cpp from source with GGML_{(backend or '').upper()}=ON"]
        else:
            asset = (ctx.plan["engine"].get("llama_asset")
                     or shapes.llama_asset(ctx.family, backend, arch))
            steps = [f"fetch the newest llama.cpp build matching /{asset}/"
                     if asset else
                     f"upstream publishes no llama.cpp build for "
                     f"{ctx.family}/{backend}/{arch} -- set "
                     f"engine.build_from_source: true, or pick another backend"]
        if "llama-swap" in missing:
            tag = shapes.LLAMA_SWAP_ASSETS.get((ctx.family, arch))
            steps.append(f"fetch the latest llama-swap release asset *{tag}*"
                         if tag else
                         f"no llama-swap asset known for {ctx.family}/{arch}")
        return Check(MISSING, "not installed: " + ", ".join(missing), steps)

    # -- doing it ----------------------------------------------------------
    def apply(self, ctx) -> None:
        kind = ctx.plan["engine"]["kind"]
        if kind == "ollama":
            raise RuntimeError(
                "Ollama is installed by its own installer, not by fleetctl -- "
                "https://ollama.com/download. Re-run apply afterwards; "
                "everything else on this box is provisioned already.")
        # Only what check() found missing. Turning on llama_swap for a box
        # whose llama-server is installed and RUNNING used to re-fetch the
        # whole llama.cpp release over the top of it -- the archive unpack
        # removes the binary directory first, with the service still holding
        # those files open. A working engine is not a reason to download one.
        present = ctx.facts.get("engines") or {}
        missing = {name for name, path in self._binaries(ctx).items()
                   if not ctx.path(path).exists()
                   and not present.get(name.replace("-", "_"))}
        if "llama-server" in missing:
            if ctx.family == "darwin":
                self._brew(ctx)
            elif ctx.plan["engine"].get("build_from_source"):
                raise RuntimeError(
                    "engine.build_from_source is set, and building llama.cpp is "
                    "deliberately not automated here: it is a 20-minute compile "
                    "whose flags are per-card. See hosts/apu-box-1/bootstrap/10-llamacpp.sh "
                    "and hosts/gpu-laptop-1/bootstrap/10-llamacpp.sh.")
            else:
                self._release_archive(ctx)
        if ctx.plan["engine"]["llama_swap"] and "llama-swap" in missing:
            self._llama_swap(ctx)
        if not missing:
            ctx.did("engine already present; nothing fetched")

    def _brew(self, ctx) -> None:
        brew = shapes.brew_bin()
        ctx.run([brew, "install", "-q", "llama.cpp"], timeout=2400)
        r = ctx.probe([brew, "--prefix", "llama.cpp"])
        prefix = (r.stdout or "").strip() if r else ""
        for name in ("llama-server", "llama-bench"):
            link = ctx.path(shapes.join(ctx.family, ctx.plan["paths"]["bin"], name))
            target = Path(prefix) / "bin" / name
            if prefix and target.exists():
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(target)
        ctx.did("llama.cpp (Metal bottle) linked into bin/")

    def _release_archive(self, ctx) -> None:
        backend = ctx.plan["engine"]["backend"] or "cpu"
        arch = ctx.plan["platform"]["arch"] or ""
        pattern = (ctx.plan["engine"].get("llama_asset")
                   or shapes.llama_asset(ctx.family, backend, arch))
        if not pattern:
            raise RuntimeError(
                f"upstream publishes no llama.cpp build for "
                f"{ctx.family}/{backend}/{arch}. Set engine.build_from_source: "
                f"true in host.yml, or choose a backend that has one.")
        rel, asset = _find(LLAMA_RELEASES, pattern)
        binroot = (shapes.join(ctx.family, ctx.plan["paths"]["bin"], "llama")
                   if ctx.family == "windows" else ctx.plan["paths"]["bin"])
        dest = ctx.path(binroot)
        # Wipe first. The archive is flat -- llama-server.exe beside its
        # ggml-*.dll backends -- and a stale ggml-vulkan.dll next to a new
        # llama-server.exe is a load-time crash with no useful message.
        self._unpack(ctx, asset["browser_download_url"], dest, wipe=True)
        if ctx.family == "windows" and backend == "cuda":
            # The CUDA build ships without the CUDA runtime; it is a second
            # archive that unpacks BESIDE llama-server.exe (wipe=False -- the
            # binaries just landed there). Without it the binary loads and
            # then dies on a missing cudart64_*.dll, which reads as a broken
            # build rather than a missing dependency.
            #
            # The runtime is chosen from the BINARY's toolkit version, not
            # independently: upstream publishes several (12.4 and 13.3 today),
            # and a 13.3 runtime beside a 12.4 build is the same missing-DLL
            # failure with an extra 400 MB of download in front of it.
            runtime_pattern = shapes.cuda_runtime_for(asset["name"])
            _, runtime = _find(LLAMA_RELEASES, runtime_pattern)
            self._unpack(ctx, runtime["browser_download_url"], dest, wipe=False)
            ctx.did(f"CUDA runtime {runtime['name']} unpacked beside llama-server")
        ctx.write(shapes.join(ctx.family, ctx.plan["paths"]["prefix"],
                              "LLAMA_CPP_REV"), str(rel.get("tag_name")) + "\n")
        ctx.did(f"llama.cpp {rel.get('tag_name')} ({backend}) -> {binroot}")

    def _llama_swap(self, ctx) -> None:
        arch = ctx.plan["platform"]["arch"] or ""
        needle = shapes.LLAMA_SWAP_ASSETS.get((ctx.family, arch))
        if not needle:
            raise RuntimeError(f"no llama-swap asset for {ctx.family}/{arch}")
        rel, asset = _find(SWAP_RELEASES, re.escape(needle))
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self._unpack(ctx, asset["browser_download_url"], Path(tmp), wipe=False)
            exe = "llama-swap.exe" if ctx.family == "windows" else "llama-swap"
            found = next((p for p in Path(tmp).rglob(exe) if p.is_file()), None)
            if not found:
                raise RuntimeError(f"{exe} not found in {asset['name']}")
            ctx.copy(found, shapes.join(ctx.family, ctx.plan["paths"]["bin"], exe),
                     mode=0o755)
        ctx.did(f"llama-swap {rel.get('tag_name')} installed")

    @staticmethod
    def _unpack(ctx, url: str, dest: Path, *, wipe: bool) -> None:
        import shutil
        import tempfile
        import zipfile

        name = url.rsplit("/", 1)[-1]
        with tempfile.TemporaryDirectory() as tmp:
            blob = Path(tmp) / name
            req = urllib.request.Request(url, headers={"User-Agent": "fleetctl"})
            with urllib.request.urlopen(req, timeout=300) as src, \
                    open(blob, "wb") as out:  # noqa: S310
                shutil.copyfileobj(src, out)
            if wipe and dest.is_dir():
                shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)
            if name.endswith(".zip"):
                with zipfile.ZipFile(blob) as zf:
                    zf.extractall(dest)
            else:
                shutil.unpack_archive(str(blob), str(dest))
        # Some builds nest everything one level down in build/bin.
        for exe in ("llama-server", "llama-server.exe"):
            if (dest / exe).exists():
                return
        found = next((p for p in dest.rglob("llama-server*") if p.is_file()), None)
        if found and found.parent != dest:
            for item in found.parent.iterdir():
                shutil.move(str(item), str(dest / item.name))
