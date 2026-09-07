"""Regression guard for the onclick/JS-string-in-HTML-attribute XSS finding.

esc() (in gateway/static/index.html) is an HTML-entity escaper only. A value
that lands inside a JS *string literal* that itself sits inside an HTML event
attribute -- e.g. onclick="unloadOne('${esc(m.id)}')" -- is not made safe by
esc() alone: the browser HTML-decodes the attribute value BEFORE the inline
JS is parsed, so an id/alias/path such as `x');alert(1);//`, reported by a
compromised peer, an LM Studio model name, an HF search hit, or any other
data-derived string, decodes back to a raw `'` and breaks out of the string
to run as script with the dashboard's ambient admin identity.

The fix is jsArg(v) = esc(JSON.stringify(String(v ?? ''))) -- JSON.stringify
makes the value a JS string literal (escaping quotes/backslashes/newlines/
U+2028), and esc() then HTML-escapes that literal so the attribute cannot end
early. Call sites changed from fn('${esc(x)}') to fn(${jsArg(x)}).

This test fails if the vulnerable pattern reappears anywhere in the page, and
asserts jsArg is defined and used.
"""
from __future__ import annotations

import re
from pathlib import Path

PAGE = Path(__file__).with_name("..").joinpath("static", "index.html").resolve()

# An esc() call sitting directly inside an on<event>="..." attribute's JS
# string is the vulnerable shape: the browser HTML-decodes the attribute
# before JS sees it, so esc()'s entity-escaping never reaches the JS parser.
VULNERABLE = re.compile(r'on[a-z]+="[^"]*\$\{esc\(')


def _html() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_no_bare_esc_call_lands_inside_an_event_attribute_js_string():
    html = _html()
    hits = VULNERABLE.findall(html)
    assert not hits, (
        f"found {len(hits)} onclick/on*= attribute(s) interpolating a bare "
        "esc(...) call into a JS string literal -- this is the reflected-XSS "
        "shape (esc() only HTML-escapes; the browser undoes that before the "
        "JS parser runs). Use jsArg(...) instead: fn(${jsArg(x)})."
    )


def test_jsarg_helper_is_defined_next_to_esc():
    html = _html()
    assert "const esc = s =>" in html, "esc() helper not found where expected"
    assert re.search(
        r"const jsArg = v => esc\(JSON\.stringify\(String\(v \?\? ''\)\)\)",
        html,
    ), "jsArg(v) = esc(JSON.stringify(String(v ?? ''))) helper is missing"


def test_jsarg_is_actually_used_at_former_vulnerable_sites():
    html = _html()
    # A handful of the concrete sites the audit flagged; each must now route
    # its dynamic argument through jsArg(...) rather than a bare esc('...').
    for needle in (
        "unloadOne(${jsArg(m.id)})",
        "deletePubModel(${jsArg(m.public_id)})",
        "delLocal(${jsArg(l.path)})",
        "cfgDetail(${jsArg(name)})",
    ):
        assert needle in html, f"expected {needle!r} in index.html"


def test_jsarg_output_is_inert_when_the_attribute_is_html_decoded():
    """Reproduce the browser's own two-stage decode for one hostile value.

    This mirrors what a <script> tag does at runtime: the HTML parser
    decodes entities in the attribute value first, then (if it were passed
    to eval/new Function, as an inline onclick handler effectively is) the
    JS parser reads what's left. jsArg's job is to make sure that after the
    HTML-decode step, the JS parser only ever sees a single, safely-quoted
    string argument -- never a value that closes the string early.
    """
    import html as html_mod
    import json

    hostile = "x');alert(1);//"

    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    def js_arg(v: str) -> str:
        return esc(json.dumps(str(v)))

    rendered_attr = f"unloadOne({js_arg(hostile)})"
    # What the browser hands the (would-be) JS parser after HTML-decoding
    # the attribute value -- this must remain a single quoted argument.
    decoded = html_mod.unescape(rendered_attr)
    assert decoded == 'unloadOne("x\');alert(1);//")'
    # The whole hostile payload sits inside one JS string literal: no
    # unescaped double-quote appears before the closing paren, so nothing
    # after the opening quote can terminate the string or start new code.
    inner = decoded[len("unloadOne(") : -1]
    assert inner.startswith('"') and inner.endswith('"')
    # The only two double-quotes in the whole decoded call are the string's
    # own delimiters -- nothing inside the payload can end the string early
    # and hand the parser new code (the pre-fix bug: a raw ' did exactly that).
    assert decoded.count('"') == 2
