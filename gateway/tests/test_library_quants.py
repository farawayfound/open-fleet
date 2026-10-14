"""Library per-quant package downloads' pure render functions, checked by
running the page's own JavaScript.

renderQuantRows and renderJobRows exist so the Library tab can show ONE row
per quant level (LM Studio style) instead of one per file, and so a package's
shards share one progress bar in the Downloads panel instead of racing each
other down the list. renderModels gained a filter (only a `kind === "main"`
LOCAL entry is a loadable GGUF) and switchSection was split out of showTab so
addQuantModel can land on the Models tab without loadModels() stomping the
record it just staged. library_quants_check.mjs drives all of that the way a
click would.

Skipped, not failed, where node is missing -- a box without it can still run
the gateway's test suite.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

CHECK = Path(__file__).with_name("library_quants_check.mjs")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_quant_and_job_group_rendering_matches_the_contract():
    r = subprocess.run(["node", str(CHECK)], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    assert "all assertions passed" in r.stdout
