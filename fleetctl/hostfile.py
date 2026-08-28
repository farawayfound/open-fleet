"""host.yml: a deliberately small YAML, read and written without PyYAML.

`fleetctl` runs on a box that has no virtualenv, so it cannot import PyYAML,
and the plan file is the one thing a person is expected to open and edit by
hand -- which rules out JSON, where a comment cannot say why a number is what
it is. So: a subset of YAML, parsed here.

WHAT IS SUPPORTED
  mappings            two-space indentation, `key: value` or `key:` + block
  block lists         `- scalar`, indented under their key
  inline lists        `[a, b, c]` of scalars, because a person editing
                      admin_emails by hand will write one
  scalars             null/~/empty -> None, true/false, ints, floats,
                      'single' and "double" quoted strings, else a bare string
  comments            a whole line starting `#`, or ` #` after a value

WHAT IS NOT, and RAISES rather than guessing
  anchors, aliases, tags, multiple documents, flow mappings `{a: 1}`,
  block scalars `|` and `>`, lists of mappings, tabs for indentation

The last line is the whole design. A YAML subset that silently misreads what
it does not understand would put a wrong value into a box's env file and be
discovered months later; one that refuses is an error message at plan time.
Every rejection carries a line number and says what it saw.

The agreement with real YAML is not asserted here but in the tests: where
PyYAML is importable (CI, and any box with the gateway venv) every host.yml
in the repo is parsed both ways and the results must be equal.
"""
from __future__ import annotations

import re
from typing import Any

__all__ = ["HostFileError", "loads", "load", "dumps", "dump"]

INDENT = 2


class HostFileError(ValueError):
    """A host.yml this parser will not guess at. Always carries a line."""

    def __init__(self, line_no: int, line: str, why: str) -> None:
        super().__init__(f"host.yml line {line_no}: {why}\n    {line.rstrip()}")
        self.line_no = line_no
        self.why = why


_INT = re.compile(r"^[+-]?\d+$")
_FLOAT = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")
_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def _scalar(text: str, line_no: int, line: str) -> Any:
    """One value. Quoted strings keep whatever is inside them, verbatim."""
    s = text.strip()
    if s.startswith(("|", ">")):
        raise HostFileError(line_no, line, "block scalars are not supported")
    if s.startswith("{"):
        raise HostFileError(line_no, line, "flow mappings are not supported")
    if s.startswith(("&", "*", "!")):
        raise HostFileError(line_no, line, "anchors, aliases and tags are not supported")
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    if s == "" or s in ("null", "~", "Null", "NULL"):
        return None
    low = s.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if _INT.match(s):
        return int(s)
    if _FLOAT.match(s):
        return float(s)
    return s


def _inline_list(text: str, line_no: int, line: str) -> list:
    inner = text.strip()[1:-1].strip()
    if not inner:
        return []
    out: list[Any] = []
    # Split on commas that are not inside quotes. The values here are host
    # names and email addresses, so this never has to be a real tokenizer --
    # but it must not split "a, b" in the middle of a quoted string either.
    field, quote = "", ""
    for ch in inner:
        if quote:
            field += ch
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
            field += ch
        elif ch == ",":
            out.append(_scalar(field, line_no, line))
            field = ""
        else:
            field += ch
    if quote:
        raise HostFileError(line_no, line, "unterminated quote in inline list")
    out.append(_scalar(field, line_no, line))
    return out


def _strip_comment(text: str) -> str:
    """Drop a trailing ` # ...`, respecting quotes. A `#` with no space in
    front of it is part of the value -- URLs and passwords contain them."""
    out, quote = "", ""
    prev = ""
    for ch in text:
        if quote:
            out += ch
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
            out += ch
        elif ch == "#" and prev in ("", " ", "\t"):
            break
        else:
            out += ch
        prev = ch
    return out.rstrip()


class _Line:
    __slots__ = ("no", "raw", "indent", "body")

    def __init__(self, no: int, raw: str) -> None:
        self.no = no
        self.raw = raw
        stripped = raw.lstrip(" ")
        self.indent = len(raw) - len(stripped)
        self.body = stripped


def loads(text: str) -> dict:
    """Parse a host.yml. Always returns a mapping at the top level."""
    lines: list[_Line] = []
    for no, raw in enumerate(text.splitlines(), start=1):
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise HostFileError(no, raw, "tabs cannot be used for indentation")
        body = _strip_comment(raw)
        if not body.strip():
            continue
        if body.lstrip().startswith("---") or body.lstrip().startswith("..."):
            raise HostFileError(no, raw, "multiple documents are not supported")
        lines.append(_Line(no, body))

    pos = 0

    def parse_map(indent: int) -> dict:
        nonlocal pos
        out: dict[str, Any] = {}
        while pos < len(lines):
            ln = lines[pos]
            if ln.indent < indent:
                break
            if ln.indent > indent:
                raise HostFileError(ln.no, ln.raw,
                                    f"unexpected indent (expected {indent} spaces)")
            if ln.body.startswith("- "):
                raise HostFileError(ln.no, ln.raw, "list item where a key was expected")
            key, sep, rest = ln.body.partition(":")
            if not sep:
                raise HostFileError(ln.no, ln.raw, "expected `key: value`")
            key = key.strip()
            if not _KEY.match(key):
                raise HostFileError(ln.no, ln.raw, f"not a usable key: {key!r}")
            if key in out:
                raise HostFileError(ln.no, ln.raw, f"duplicate key {key!r}")
            rest = rest.strip()
            pos += 1
            if rest.startswith("["):
                if not rest.endswith("]"):
                    raise HostFileError(ln.no, ln.raw,
                                        "inline list must open and close on one line")
                out[key] = _inline_list(rest, ln.no, ln.raw)
            elif rest:
                out[key] = _scalar(rest, ln.no, ln.raw)
            else:
                # A bare `key:` introduces a block, or is an explicit null when
                # the next line is not indented past it.
                if pos < len(lines) and lines[pos].indent > indent:
                    child = lines[pos]
                    if child.indent != indent + INDENT:
                        raise HostFileError(
                            child.no, child.raw,
                            f"indent must go up by exactly {INDENT} spaces "
                            f"(saw {child.indent - indent})")
                    if child.body.startswith("- "):
                        out[key] = parse_list(child.indent)
                    else:
                        out[key] = parse_map(child.indent)
                else:
                    out[key] = None
        return out

    def parse_list(indent: int) -> list:
        nonlocal pos
        out: list[Any] = []
        while pos < len(lines):
            ln = lines[pos]
            if ln.indent < indent:
                break
            if ln.indent > indent:
                raise HostFileError(ln.no, ln.raw, "unexpected indent inside a list")
            if not ln.body.startswith("- "):
                if ln.body == "-":
                    raise HostFileError(ln.no, ln.raw, "empty list item")
                break
            item = ln.body[2:].strip()
            if not item:
                raise HostFileError(ln.no, ln.raw, "empty list item")
            if ":" in item and not item.startswith(("\"", "'")):
                # `- name: x` -- a list of mappings. Legal YAML, not legal
                # here, and worth saying so plainly: nothing in a host plan
                # needs one, and supporting it would double this parser.
                raise HostFileError(ln.no, ln.raw,
                                    "lists of mappings are not supported; "
                                    "quote the value if the colon is literal")
            out.append(_scalar(item, ln.no, ln.raw))
            pos += 1
        return out

    doc = parse_map(0)
    if pos != len(lines):
        ln = lines[pos]
        raise HostFileError(ln.no, ln.raw, "could not continue parsing here")
    return doc


def load(path) -> dict:
    from pathlib import Path

    return loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------
# Bare where it is safe and quoted where it is not. The list is what YAML
# would otherwise read as something else: a Windows path is fine bare, but
# `on`, `8080`, `~` and anything with a leading `#` are not.
_NEEDS_QUOTES = re.compile(
    r"^$|^[\s#&*!|>%@`\[\]{},]|[:#]\s|\s$|^(true|false|yes|no|on|off|null|~)$",
    re.I)


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return repr(value)
    s = str(value)
    if _NEEDS_QUOTES.search(s) or _INT.match(s) or _FLOAT.match(s):
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return s


def dumps(data: dict, comments: dict[str, str] | None = None,
          header: str = "") -> str:
    """Emit a host.yml.

    `comments` maps a dotted path ("network.public_api_url") to prose that is
    written above that key. It is not decoration: the whole reason this file
    is YAML and not JSON is that the ~12 values which genuinely differ per box
    need somewhere to say why they are what they are.
    """
    comments = comments or {}
    out: list[str] = []
    if header:
        out.extend("# " + ln if ln else "#" for ln in header.rstrip().split("\n"))
        out.append("")

    def emit(node: dict, indent: int, path: str) -> None:
        pad = " " * indent
        for key, value in node.items():
            dotted = f"{path}.{key}" if path else key
            note = comments.get(dotted)
            if note:
                # A blank line above the comment, so a run of annotated keys
                # reads as paragraphs -- but not directly under the parent
                # key, where it would orphan the heading.
                prev = out[-1].strip() if out else ""
                if prev and not prev.startswith("#") and not prev.endswith(":"):
                    out.append("")
                for ln in note.rstrip().split("\n"):
                    out.append(f"{pad}# {ln}" if ln else f"{pad}#")
            if isinstance(value, dict):
                out.append(f"{pad}{key}:")
                if value:
                    emit(value, indent + INDENT, dotted)
                else:
                    # An empty mapping has no block to write. Written as
                    # `null`, which is what `key:` alone would read back as
                    # anyway -- so {} does NOT survive a round trip, it comes
                    # back as None. The planner never emits one; this is the
                    # honest behaviour for a hand-edited file that does.
                    out[-1] = f"{pad}{key}: null"
            elif isinstance(value, (list, tuple)):
                if not value:
                    out.append(f"{pad}{key}: []")
                else:
                    out.append(f"{pad}{key}:")
                    for item in value:
                        out.append(f"{pad}  - {_fmt(item)}")
            else:
                out.append(f"{pad}{key}: {_fmt(value)}")

    emit(data, 0, "")
    return "\n".join(out).rstrip() + "\n"


def dump(data: dict, path, comments: dict[str, str] | None = None,
         header: str = "") -> str:
    from pathlib import Path

    text = dumps(data, comments, header)
    Path(path).write_text(text, encoding="utf-8", newline="\n")
    return text
