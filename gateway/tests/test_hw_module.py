"""hw.py on its own: the half of the gateway the installer also runs.

Two promises are worth a test each. The first is that hw.py imports with
nothing but the standard library -- fleetctl loads it on a box that has no
virtualenv, so an accidental `import psutil` in here would turn "provision
this machine" into "pip install first, then provision this machine".

The second is the distro table. Grouping distributions by the verb they want
rather than by name is what keeps the matrix at four families instead of one
branch per distribution, and getting a family wrong is a box that tries to
apt-get on Arch.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import app as gw
import hw


class TestStandaloneImport:
    def test_imports_with_no_third_party_packages_on_the_path(self):
        """Loaded in a subprocess with an EMPTY sys.path except the gateway
        directory, so nothing installed in this venv can satisfy it by
        accident. This is the closest thing to "a fresh box" that a test can
        be."""
        gateway_dir = Path(hw.__file__).parent
        code = (
            "import sys\n"
            f"sys.path = [r'{gateway_dir}'] + [p for p in sys.path "
            "if 'site-packages' not in p and 'dist-packages' not in p]\n"
            "import hw\n"
            "assert hw.os_family()\n"
            "assert isinstance(hw.gpu_cards(), list)\n"
            "assert hw.distro_family()\n"
            "print('ok')\n"
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "ok" in r.stdout

    def test_it_imports_nothing_from_the_gateway(self):
        """The dependency runs one way. hw.py reaching back into app.py would
        make it unimportable exactly where it is needed most."""
        source = Path(hw.__file__).read_text(encoding="utf-8")
        for forbidden in ("import app", "from app import", "import psutil",
                          "import httpx", "import fastapi", "import yaml"):
            assert forbidden not in source, forbidden

    def test_the_gateway_re_exports_what_the_tests_reach_for(self):
        """app.py keeps these names so nothing outside had to change."""
        for name in ("amdgpu_stats", "nvidia_stats", "windows_gpu_stats",
                     "darwin_gpu_stats", "gpu_cards", "vram_total_bytes",
                     "os_info", "_read_int", "_wingpu_cache", "_macgpu_cache",
                     "_WIN_GPU_PS", "_nvsmi", "_win_build"):
            assert hasattr(gw, name), name


class TestDistroFamilies:
    @pytest.mark.parametrize("ident,want", [
        ("ubuntu", "apt"), ("debian", "apt"), ("linuxmint", "apt"),
        ("pop", "apt"), ("raspbian", "apt"),
        ("fedora", "dnf"), ("rhel", "dnf"), ("rocky", "dnf"),
        ("almalinux", "dnf"), ("centos", "dnf"),
        ("arch", "pacman"), ("manjaro", "pacman"), ("endeavouros", "pacman"),
        ("opensuse-tumbleweed", "zypper"), ("opensuse-leap", "zypper"),
        ("sles", "zypper"),
    ])
    def test_by_id(self, monkeypatch, ident, want):
        monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
        assert hw.distro_family({"ID": ident}) == want

    @pytest.mark.parametrize("like,want", [
        ("ubuntu debian", "apt"), ("debian", "apt"), ("fedora", "dnf"),
        ("arch", "pacman"), ("suse opensuse", "zypper"),
    ])
    def test_by_id_like_when_the_id_is_unknown(self, monkeypatch, like, want):
        """A distro this table has never heard of still says what it is
        derived from -- which is how Mint and Pop!_OS cost nothing to
        support."""
        monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
        assert hw.distro_family({"ID": "somethingnew", "ID_LIKE": like}) == want

    def test_by_id_like_with_commas(self, monkeypatch):
        monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
        assert hw.distro_family({"ID": "x", "ID_LIKE": "rhel,fedora"}) == "dnf"

    def test_falls_back_to_whichever_tool_is_on_path(self, monkeypatch):
        """Better than "unknown": a distro nobody anticipated still installs
        packages with one of four commands."""
        monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
        monkeypatch.setattr(hw.shutil, "which",
                            lambda n: "/usr/bin/pacman" if n == "pacman" else None)
        assert hw.distro_family({"ID": "nobodys-distro"}) == "pacman"

    def test_unknown_is_unknown_not_a_guess(self, monkeypatch):
        monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
        monkeypatch.setattr(hw.shutil, "which", lambda n: None)
        assert hw.distro_family({"ID": "nobodys-distro"}) == "unknown"

    def test_the_macs_and_windows_do_not_consult_os_release(self, monkeypatch):
        monkeypatch.setattr(hw.platform, "system", lambda: "Darwin")
        assert hw.distro_family() == "brew"
        monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
        assert hw.distro_family() == "windows"

    def test_every_family_in_the_table_is_one_fleetctl_can_drive(self):
        # The repo root, added here rather than in conftest.py: this is the
        # only test in the gateway suite that looks at the installer, and the
        # gateway must never need it on its path to run.
        root = str(Path(hw.__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        from fleetctl import shapes

        for family in hw.PKG_FAMILIES:
            assert family in shapes.PKG_INSTALL, family
            assert family in shapes.BASE_PACKAGES, family
            assert family in shapes.PKG_QUERY, family


class TestThePlansAndTheSpecSheetAgree:
    """Every box's fleet name has to appear in DEFAULT_SPECS.

    The name is the lookup key for everything: the spec sheet, the routing
    rank, usage metering and the hub's peer list. A box whose plan calls it
    one thing and whose spec sheet calls it another loses its specs silently
    -- it stays online, answers health checks, and is ranked as though it
    were a machine nobody had measured.

    mac-desktop is why this is a test rather than a convention: its tailnet name
    and its ssh alias are `mac-desktop`, its host directory is `mac-desktop-1`, and
    the fleet name -- the one that has to be right -- is `mac-desktop-1`. Detection
    guesses the tailnet name, so the plan needs an explicit override, and
    nothing but this test would have said so.
    """

    def _plans(self):
        root = Path(hw.__file__).resolve().parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from fleetctl import hostfile

        return {p: hostfile.load(p) for p in sorted(root.glob("hosts/*/host.yml"))}

    def test_every_planned_name_has_a_spec_row(self):
        for path, doc in self._plans().items():
            name = doc["host"]["name"]
            assert name in gw.DEFAULT_SPECS, (
                f"{path}: fleet name {name!r} is not in DEFAULT_SPECS -- this "
                f"box would run with no specs and no routing rank")

    def test_the_hub_is_the_hub_in_both(self):
        for path, doc in self._plans().items():
            if doc["host"].get("role") == "hub":
                spec = gw.DEFAULT_SPECS[doc["host"]["name"]]
                assert spec.get("role") == "hub" or spec.get("klass") == "hub", path

    def test_a_planned_vram_figure_is_not_wildly_at_odds_with_the_spec_sheet(self):
        """Not equality -- the spec sheet is a policy number and the plan
        carries a measurement, and they legitimately differ (apu-tablet-2's card
        offers 47.8 GiB and the box budgets 44). But an order of magnitude
        apart means one of them is about a different machine."""
        for path, doc in self._plans().items():
            planned = (doc.get("sizing") or {}).get("vram_gb")
            spec = gw.DEFAULT_SPECS[doc["host"]["name"]].get("vram_gb")
            if not planned or not spec:
                continue
            ratio = max(planned, spec) / max(1e-9, min(planned, spec))
            assert ratio < 4, f"{path}: plan says {planned} GB, spec sheet {spec} GB"


class TestOsRelease:
    def test_reads_a_file(self, tmp_path, monkeypatch):
        f = tmp_path / "os-release"
        f.write_text('ID=arch\nNAME="Arch Linux"\n# a comment\n\n'
                     "PRETTY_NAME='Arch Linux'\n", encoding="utf-8")
        monkeypatch.setattr(hw, "Path", lambda p: f if "os-release" in str(p)
                            else Path(p))
        rel = hw.os_release()
        assert rel["ID"] == "arch"
        assert rel["NAME"] == "Arch Linux"
        assert rel["PRETTY_NAME"] == "Arch Linux"

    def test_absent_is_empty_not_an_error(self, monkeypatch):
        monkeypatch.setattr(hw, "Path", lambda p: Path("/nonexistent/os-release"))
        assert hw.os_release() == {}


class TestBackendChoice:
    def test_apple_silicon_is_metal(self, monkeypatch):
        monkeypatch.setattr(hw.platform, "system", lambda: "Darwin")
        assert hw.llama_backend([]) == "metal"

    def test_nvidia_is_cuda(self, monkeypatch):
        monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
        assert hw.llama_backend([{"card": "NVIDIA GeForce RTX 4070",
                                  "vram_total": 8 * 1024 ** 3}]) == "cuda"

    def test_amd_is_vulkan_not_rocm(self, monkeypatch):
        """Vulkan needs nothing but the driver already installed, ships a
        35 MB archive against ROCm's 197, and is at parity or better for token
        generation on RDNA-class cards. ROCm stays an explicit opt-in."""
        monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
        assert hw.llama_backend([{"card": "AMD Radeon RX 9060 XT",
                                  "vram_total": 16 * 1024 ** 3}]) == "vulkan"

    def test_no_card_is_cpu(self, monkeypatch):
        monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
        assert hw.llama_backend([]) == "cpu"

    def test_a_tiny_igpu_is_cpu(self, monkeypatch):
        """Measured on cpu-box-1, the fleet's CPU-only fallback box: its
        integrated Radeon reports a 512 MB carve-out of system RAM, and
        offloading to it means every token crosses the bus for a fraction of
        a model. Slower than not using it at all."""
        monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
        assert hw.llama_backend([{"card": "card0",
                                  "vram_total": 512 * 1024 ** 2}]) == "cpu"

    def test_the_floor_is_2_gib(self, monkeypatch):
        monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
        just_under = [{"card": "card0", "vram_total": 2 * 1024 ** 3 - 1}]
        just_over = [{"card": "card0", "vram_total": 2 * 1024 ** 3}]
        assert hw.llama_backend(just_under) == "cpu"
        assert hw.llama_backend(just_over) == "vulkan"

    def test_a_card_with_an_unreadable_size_is_still_a_card(self, monkeypatch):
        """vram_total None is "we could not read it", not "it is small"."""
        monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
        assert hw.llama_backend([{"card": "AMD Radeon Graphics",
                                  "vram_total": None}]) == "vulkan"


class TestReadingsThatMustNotRaise:
    """Every one of these runs on whatever box CI happens to give us, and
    none may raise: a machine that will not answer one question is still a
    machine worth provisioning."""

    def test_ram_bytes(self):
        value = hw.ram_bytes()
        assert value is None or value > 0

    def test_cpu_model(self):
        assert isinstance(hw.cpu_model(), str)

    def test_os_family(self):
        assert hw.os_family() in ("linux", "darwin", "windows")

    def test_os_info_has_every_key(self):
        info = hw.os_info()
        assert set(info) == {"name", "kernel", "arch", "platform", "python"}

    def test_gpu_cards_is_a_list_of_dicts(self):
        for card in hw.gpu_cards():
            assert isinstance(card, dict)
            assert "card" in card and "vram_total" in card

    def test_vram_total_bytes(self):
        value = hw.vram_total_bytes()
        assert value is None or value > 0

    def test_a_spec_sheet_figure_is_used_only_when_nothing_measured(self,
                                                                    monkeypatch):
        monkeypatch.setattr(hw, "amdgpu_stats", lambda: [])
        monkeypatch.setattr(hw, "nvidia_stats", lambda nvsmi=None: [])
        monkeypatch.setattr(hw, "windows_gpu_stats", lambda: [])
        monkeypatch.setattr(hw, "darwin_gpu_stats", lambda spec=None: [])
        assert hw.vram_total_bytes(spec_vram_gb=16) == 16 * 1024 ** 3
        assert hw.vram_total_bytes() is None
