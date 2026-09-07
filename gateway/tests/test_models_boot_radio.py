"""The Models tab's boot column, checked by running the page's own JavaScript.

`preload` is the one model a box loads when its engine starts, and models.json
allows exactly one of them (check_preload_count, tested in
test_swap_render_warm.py). The dashboard says so with a radio group — one
circle across the whole table — which needs a thing HTML has no native state
for: clicking the selected circle to mean "no boot model at all". That handler,
and the render that follows it, are what models_boot_radio_check.mjs drives.

Skipped, not failed, where node is missing — a box without it can still run the
gateway's test suite.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

CHECK = Path(__file__).with_name("models_boot_radio_check.mjs")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_boot_radio_picks_one_model_and_clicking_it_again_picks_none():
    r = subprocess.run(["node", str(CHECK)], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    assert "all assertions passed" in r.stdout
