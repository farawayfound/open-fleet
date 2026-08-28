"""fleetctl's tests import fleetctl, and nothing else that is not stdlib.

pytest is the one dependency, and deliberately the only one: these tests run
on a hosted CI runner and inside a bare Arch container, both of which are
places where `pip install -r requirements.txt` would be the thing under test
rather than a precondition of it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO


# Tests that read THIS fleet's hosts/<name>/host.yml files -- the hub's bind,
# the addresses every box advertises, the count of plans. A checkout without
# any plans (the public export, a fresh fork, a single-box install) gives
# them nothing to say, so they are skipped there rather than failed. Named
# here, in one place, rather than decorated across three files.
_FLEET_BOUND_CLASSES = {
    "TestTheHubIsNotAPeer", "TestWhatTheRepoSays", "TestEveryPlanInTheRepo",
}
_FLEET_BOUND_TESTS = {
    "test_it_offers_the_near_miss", "test_a_genuinely_new_box_gets_the_list",
    "test_a_known_box_is_silent",
    "test_a_box_the_hub_has_never_been_told_about_is_reported_not_invented",
    "test_running_it_twice_changes_nothing_the_second_time",
}


def pytest_collection_modifyitems(config, items):
    if any((REPO / "hosts").glob("*/host.yml")):
        return
    skip = pytest.mark.skip(reason="no hosts/*/host.yml in this checkout -- a fleet-bound test")
    for item in items:
        cls = item.cls.__name__ if getattr(item, "cls", None) else ""
        name = getattr(item, "originalname", None) or item.name.split("[")[0]
        if cls in _FLEET_BOUND_CLASSES or name in _FLEET_BOUND_TESTS:
            item.add_marker(skip)


@pytest.fixture
def linux_facts() -> dict:
    """A plausible Linux box. Deliberately hand-written rather than captured:
    a test that breaks when someone upgrades a real machine is a test nobody
    trusts."""
    return {
        "hostname": "testbox",
        # The TARGET box's home, not the one running the tests. See
        # TestCrossPlatformPlanning for why that distinction earned a test.
        "home": "/home/user",
        "os": {"family": "linux", "name": "Ubuntu 24.04", "kernel": "6.8.0",
               "arch": "x86_64", "distro_id": "ubuntu", "distro_version": "24.04",
               "distro_like": "debian", "package_manager": "apt"},
        "service_manager": "systemd",
        "privilege": {"root": False, "sudo_nopasswd": True},
        "python": {"exe": "/usr/bin/python3.12", "version": "3.12.3",
                   "min": "3.10", "candidates": []},
        "cpu": {"model": "AMD Ryzen 7", "count": 16},
        "ram_bytes": 32 * 1024 ** 3, "ram_gb": 32.0,
        "gpus": [{"card": "card0", "vram_total": 8 * 1024 ** 3}],
        "vram_bytes": 8 * 1024 ** 3, "vram_gb": 8.0,
        "llama_backend": "vulkan",
        "tailscale": {"present": True, "ipv4": "100.64.0.1", "name": "testbox"},
        "engines": {"llama_server": None, "llama_swap": None, "ollama": None,
                    "lmstudio": None},
        "install": {"prefix": "/opt/llmstack", "present": False,
                    "has_hw": False, "deployed_sha": None},
    }


@pytest.fixture
def windows_facts(linux_facts) -> dict:
    f = dict(linux_facts)
    f["os"] = {"family": "windows", "name": "Windows 11", "kernel": "build 26200",
               "arch": "AMD64", "distro_id": "", "distro_version": "",
               "distro_like": "", "package_manager": "windows"}
    f["service_manager"] = "schtasks"
    f["python"] = {"exe": r"C:\Program Files\Python312\python.exe",
                   "version": "3.12.10", "min": "3.10", "candidates": []}
    f["install"] = {"prefix": r"C:\llmstack", "present": False,
                    "has_hw": False, "deployed_sha": None}
    f["home"] = r"C:\Users\user"
    return f


@pytest.fixture
def darwin_facts(linux_facts) -> dict:
    f = dict(linux_facts)
    f["os"] = {"family": "darwin", "name": "macOS 26.6", "kernel": "25.6.0",
               "arch": "arm64", "distro_id": "", "distro_version": "",
               "distro_like": "", "package_manager": "brew"}
    f["service_manager"] = "cron"
    f["llama_backend"] = "metal"
    f["python"] = {"exe": "/opt/homebrew/bin/python3.12", "version": "3.12.14",
                   "min": "3.10", "candidates": []}
    f["install"] = {"prefix": "/Users/x/llmstack", "present": False,
                    "has_hw": False, "deployed_sha": None}
    f["home"] = "/Users/x"
    return f


@pytest.fixture
def empty_repo(tmp_path) -> Path:
    """A checkout with gateway sources but no fleet.yml and no hosts/, so a
    test can layer exactly what it means to."""
    gw = tmp_path / "gateway"
    (gw / "static").mkdir(parents=True)
    for name in ("app.py", "hw.py", "requirements.txt"):
        (gw / name).write_text(f"# {name}\n", encoding="utf-8")
    (gw / "static" / "index.html").write_text("<html></html>\n", encoding="utf-8")
    return tmp_path
