"""Choosing a llama.cpp build, against a captured release listing.

Captured rather than live, so these run in a container with no network and
so they keep testing the thing that actually broke -- upstream's asset naming
-- rather than whatever upstream happens to publish on the day CI runs.

The listing below is the real shape of
`GET /repos/ggml-org/llama.cpp/releases`, trimmed to the asset names, taken
on 2026-08-27. Note what it shows: every build release is a PRERELEASE, and
the only non-prerelease is `v0.3.0`, whose sole asset is a text file. That is
why nothing here asks for `/releases/latest`.
"""
from __future__ import annotations

import re

import pytest

from fleetctl import shapes
from fleetctl.steps import engine

ASSETS_B10661 = [
    "cudart-llama-bin-win-cuda-12.4-x64.zip",
    "cudart-llama-bin-win-cuda-13.3-x64.zip",
    "cudart-llama-bin-win-cuda-13.4-arm64.zip",
    "llama-b10661-bin-android-arm64.tar.gz",
    "llama-b10661-bin-macos-arm64.tar.gz",
    "llama-b10661-bin-macos-x64.tar.gz",
    "llama-b10661-bin-ubuntu-arm64.tar.gz",
    "llama-b10661-bin-ubuntu-openvino-2026.3-x64.tar.gz",
    "llama-b10661-bin-ubuntu-rocm-7.14-x64.tar.gz",
    "llama-b10661-bin-ubuntu-s390x.tar.gz",
    "llama-b10661-bin-ubuntu-sycl-fp16-x64.tar.gz",
    "llama-b10661-bin-ubuntu-sycl-fp32-x64.tar.gz",
    "llama-b10661-bin-ubuntu-vulkan-arm64.tar.gz",
    "llama-b10661-bin-ubuntu-vulkan-x64.tar.gz",
    "llama-b10661-bin-ubuntu-x64.tar.gz",
    "llama-b10661-bin-win-cpu-arm64.zip",
    "llama-b10661-bin-win-cpu-x64.zip",
    "llama-b10661-bin-win-cuda-12.4-x64.zip",
    "llama-b10661-bin-win-cuda-13.3-x64.zip",
    "llama-b10661-bin-win-cuda-13.4-arm64.zip",
    "llama-b10661-bin-win-opencl-adreno-arm64.zip",
    "llama-b10661-bin-win-openvino-2026.3-x64.zip",
    "llama-b10661-bin-win-rocm-7.14-x64.zip",
    "llama-b10661-bin-win-sycl-x64.zip",
    "llama-b10661-bin-win-vulkan-x64.zip",
    "llama-b10661-ui.tar.gz",
    "llama-b10661-xcframework.zip",
]


def _rel(tag: str, names: list[str], prerelease: bool = True,
         draft: bool = False) -> dict:
    return {"tag_name": tag, "prerelease": prerelease, "draft": draft,
            "assets": [{"name": n, "browser_download_url": f"https://x/{n}",
                        "size": 1} for n in names]}


LISTING = [
    # Exactly what GitHub returns: the newest first, and the stable release
    # -- the one /releases/latest would have chosen -- carrying nothing.
    _rel("b10661", ASSETS_B10661),
    _rel("b10660", [n.replace("b10661", "b10660") for n in ASSETS_B10661]),
    _rel("v0.3.0", ["nightly-tag.txt"], prerelease=False),
]


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(engine, "_releases", lambda url, timeout=40: LISTING)


class TestTheBugThatStartedThis:
    def test_the_only_stable_release_has_no_binaries(self):
        """`/releases/latest` skips prereleases, so it answers v0.3.0 --
        which is why every installer here that asked for `latest` and then
        filtered for bin-win-vulkan-x64.zip was going to throw."""
        stable = [r for r in LISTING if not r["prerelease"]]
        assert len(stable) == 1
        assert [a["name"] for a in stable[0]["assets"]] == ["nightly-tag.txt"]

    def test_we_do_not_ask_for_latest(self):
        assert "/releases/latest" not in engine.LLAMA_RELEASES
        assert "/releases/latest" not in engine.SWAP_RELEASES


class TestSelection:
    @pytest.mark.parametrize("family,backend,arch,want", [
        ("windows", "vulkan", "AMD64", "llama-b10661-bin-win-vulkan-x64.zip"),
        ("windows", "cuda", "AMD64", "llama-b10661-bin-win-cuda-12.4-x64.zip"),
        ("windows", "rocm", "AMD64", "llama-b10661-bin-win-rocm-7.14-x64.zip"),
        ("windows", "cpu", "AMD64", "llama-b10661-bin-win-cpu-x64.zip"),
        ("windows", "cpu", "ARM64", "llama-b10661-bin-win-cpu-arm64.zip"),
        ("linux", "vulkan", "x86_64", "llama-b10661-bin-ubuntu-vulkan-x64.tar.gz"),
        ("linux", "vulkan", "aarch64", "llama-b10661-bin-ubuntu-vulkan-arm64.tar.gz"),
        ("linux", "rocm", "x86_64", "llama-b10661-bin-ubuntu-rocm-7.14-x64.tar.gz"),
        ("linux", "cpu", "x86_64", "llama-b10661-bin-ubuntu-x64.tar.gz"),
        ("linux", "cpu", "aarch64", "llama-b10661-bin-ubuntu-arm64.tar.gz"),
        ("darwin", "metal", "arm64", "llama-b10661-bin-macos-arm64.tar.gz"),
        ("darwin", "metal", "x86_64", "llama-b10661-bin-macos-x64.tar.gz"),
    ])
    def test_picks_the_right_archive(self, offline, family, backend, arch, want):
        pattern = shapes.llama_asset(family, backend, arch)
        assert pattern, f"no pattern for {family}/{backend}/{arch}"
        rel, asset = engine._find(engine.LLAMA_RELEASES, pattern)
        assert asset["name"] == want
        assert rel["tag_name"] == "b10661"

    def test_the_cuda_build_is_not_the_cuda_runtime(self, offline):
        """`cudart-llama-bin-win-cuda-12.4-x64.zip` CONTAINS
        `bin-win-cuda-12.4-x64.zip`, so an unanchored search picked the
        391 MB runtime instead of the build. Measured, not hypothetical."""
        pattern = shapes.llama_asset("windows", "cuda", "AMD64")
        _, asset = engine._find(engine.LLAMA_RELEASES, pattern)
        assert not asset["name"].startswith("cudart-")

    def test_the_runtime_matches_the_build_it_accompanies(self, offline):
        """A 13.3 runtime beside a 12.4 build is the same missing-DLL failure
        with 400 MB of download in front of it."""
        for version in ("12.4", "13.3"):
            build = f"llama-b10661-bin-win-cuda-{version}-x64.zip"
            _, rt = engine._find(engine.LLAMA_RELEASES,
                                 shapes.cuda_runtime_for(build))
            assert rt["name"] == f"cudart-llama-bin-win-cuda-{version}-x64.zip"

    @pytest.mark.parametrize("family,backend,arch", [
        ("linux", "cuda", "x86_64"),     # upstream publishes no Linux CUDA build
        ("windows", "vulkan", "ARM64"),  # nor Windows-on-ARM Vulkan
        ("darwin", "cuda", "arm64"),     # nor anything of the sort
    ])
    def test_a_combination_with_no_build_is_none_not_a_wrong_guess(
            self, family, backend, arch):
        assert shapes.llama_asset(family, backend, arch) is None

    def test_a_draft_release_is_skipped(self, monkeypatch):
        """A draft's assets are not downloadable."""
        monkeypatch.setattr(engine, "_releases", lambda url, timeout=40: [
            _rel("b99999", ASSETS_B10661, draft=True),
            _rel("b10661", ASSETS_B10661),
        ])
        rel, _ = engine._find(engine.LLAMA_RELEASES,
                              shapes.llama_asset("linux", "vulkan", "x86_64"))
        assert rel["tag_name"] == "b10661"

    def test_falls_back_through_releases_until_one_has_the_asset(self,
                                                                 monkeypatch):
        monkeypatch.setattr(engine, "_releases", lambda url, timeout=40: [
            _rel("b10662", ["llama-b10662-ui.tar.gz"]),
            _rel("b10661", ASSETS_B10661),
        ])
        rel, _ = engine._find(engine.LLAMA_RELEASES,
                              shapes.llama_asset("linux", "vulkan", "x86_64"))
        assert rel["tag_name"] == "b10661"

    def test_nothing_anywhere_says_so_with_the_tags_it_looked_at(self,
                                                                 monkeypatch):
        monkeypatch.setattr(engine, "_releases", lambda url, timeout=40: [
            _rel("b10662", ["nope.txt"]), _rel("b10661", ["also-nope.txt"]),
        ])
        with pytest.raises(RuntimeError) as exc:
            engine._find(engine.LLAMA_RELEASES, r"^llama-.*-bin-win-vulkan")
        assert "b10662" in str(exc.value) and "b10661" in str(exc.value)


class TestLlamaSwap:
    SWAP = [_rel("v251", [
        "llama-swap_251_checksums.txt",
        "llama-swap_251_darwin_amd64.tar.gz",
        "llama-swap_251_darwin_arm64.tar.gz",
        "llama-swap_251_freebsd_amd64.tar.gz",
        "llama-swap_251_linux_amd64.tar.gz",
        "llama-swap_251_linux_arm64.tar.gz",
        "llama-swap_251_windows_amd64.zip",
    ], prerelease=False)]

    @pytest.mark.parametrize("family,arch,want", [
        ("linux", "x86_64", "llama-swap_251_linux_amd64.tar.gz"),
        ("linux", "aarch64", "llama-swap_251_linux_arm64.tar.gz"),
        ("darwin", "arm64", "llama-swap_251_darwin_arm64.tar.gz"),
        ("darwin", "x86_64", "llama-swap_251_darwin_amd64.tar.gz"),
        ("windows", "AMD64", "llama-swap_251_windows_amd64.zip"),
    ])
    def test_picks_the_right_binary(self, monkeypatch, family, arch, want):
        monkeypatch.setattr(engine, "_releases", lambda url, timeout=40: self.SWAP)
        needle = shapes.LLAMA_SWAP_ASSETS[(family, arch)]
        _, asset = engine._find(engine.SWAP_RELEASES, re.escape(needle))
        assert asset["name"] == want

    def test_amd64_does_not_match_arm64(self, monkeypatch):
        """`linux_amd64` and `linux_arm64` differ by one character, and
        picking the wrong one gives a binary that will not execute."""
        monkeypatch.setattr(engine, "_releases", lambda url, timeout=40: self.SWAP)
        _, asset = engine._find(engine.SWAP_RELEASES,
                                re.escape(shapes.LLAMA_SWAP_ASSETS[("linux", "aarch64")]))
        assert "arm64" in asset["name"]


class TestEveryPlannedBoxCanGetAnEngine:
    """Every host.yml in the repo that wants llama.cpp must resolve to
    something -- a published build, a Homebrew bottle, or an explicit
    build-from-source. A box whose plan cannot name an engine is a box that
    cannot be provisioned, and finding that out at apply time is too late."""

    def test_each(self, repo):
        from fleetctl import hostfile

        for path in sorted(repo.glob("hosts/*/host.yml")):
            doc = hostfile.load(path)
            eng = doc.get("engine") or {}
            if eng.get("kind") != "llama.cpp":
                continue
            family = doc["platform"]["os"]
            arch = doc["platform"].get("arch") or ""
            backend = eng.get("backend")
            if family == "darwin" or eng.get("build_from_source"):
                continue      # brew bottle, or a deliberate source build
            assert shapes.llama_asset(family, backend, arch), (
                f"{path}: no llama.cpp build for {family}/{backend}/{arch} "
                f"and engine.build_from_source is not set")
