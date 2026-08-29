"""The dashboard's own JavaScript, checked by running it.

Everything else in this directory tests the Python. The routing switch on the
home page shipped inverted -- it PUT the state the box already had, so every
click was a no-op that then announced "killswitch is ON" -- and nothing here
could have caught it, because nothing here has ever executed a line of
index.html.

dashboard_switch_check.mjs does: it evaluates the page's script block in a
node vm against a DOM stub, renders the machine grid from hand-built overview
rows, and drives the switch with a stubbed rawApi to assert the DIRECTION of
what it sends. This wrapper is how it runs in CI alongside the rest.

Skipped, not failed, where node is missing -- a box without it can still run
the gateway's test suite.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

CHECK = Path(__file__).with_name("dashboard_switch_check.mjs")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_home_page_renders_and_its_routing_switch_sends_the_right_state():
    r = subprocess.run(["node", str(CHECK)], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    assert "all assertions passed" in r.stdout
