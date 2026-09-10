"""Regression guard: the running-models context line must not divide by slots.

llama.cpp's `/props` reports `n_ctx` **per decode slot**, not the total the
server was launched with -- `upstream_props()` in gateway/app.py reads it
straight from `default_generation_settings.n_ctx`, and the ctx-verify path
there carries the same note ("`want_ctx` is per SLOT, which is what /props
reports"). The record's own `ctx` is the TOTAL: the rendered command is
`-c <ctx> --parallel <slots>`, and llama.cpp splits that total across slots.

So for a two-slot model registered at ctx 131072:

    record ctx   = 131072   (total, what -c gets)
    /props n_ctx =  65536   (per slot, what one conversation gets)
    slots        =      2

The dashboard used to print `n_ctx` as the total and then divide it by slots
again, rendering that model as "65,536 ctx - 2 slots (32,768 each)". Both
numbers were half the truth, on the one line whose stated purpose is to say
what the model actually got -- and it was read as evidence that a co-resident
model had been squeezed out of its configured window when it had not been.

Correct arithmetic: total = n_ctx * slots, per-conversation window = n_ctx.

This test fails if the double-divide returns.
"""
from __future__ import annotations

import re
from pathlib import Path

PAGE = Path(__file__).with_name("..").joinpath("static", "index.html").resolve()


def _html() -> str:
    return PAGE.read_text(encoding="utf-8")


# Any arithmetic that divides the per-slot n_ctx by the slot count is the bug.
DIVIDES_PER_SLOT_CTX = re.compile(r"n_ctx\s*/\s*\w*\.?slots")


def test_running_ctx_line_never_divides_per_slot_ctx_by_slots():
    hits = DIVIDES_PER_SLOT_CTX.findall(_html())
    assert not hits, (
        "gateway/static/index.html divides /props n_ctx by the slot count: "
        + repr(hits)
        + ". /props n_ctx is ALREADY per slot -- dividing it again reports "
        "half the real per-conversation window, and labels the per-slot "
        "value as the model's total. Use n_ctx * slots for the total and "
        "n_ctx itself for the per-conversation figure."
    )


def test_running_ctx_line_multiplies_to_get_the_total():
    """The positive half: the total must be reconstructed, not assumed."""
    html = _html()
    assert re.search(r"n_ctx\s*\*\s*slots", html), (
        "the running-models context line no longer multiplies n_ctx by the "
        "slot count to recover the total the server was launched with; if the "
        "shape changed deliberately, update this guard along with it"
    )
