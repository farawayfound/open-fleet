"""The YAML subset: what it reads, what it writes, and what it refuses.

The refusals matter most. A parser that guesses at syntax it does not
understand would put a wrong value into a box's env file and be found out
months later; the whole bet of hand-writing this instead of taking PyYAML is
that it fails loudly instead.

The agreement with real YAML is asserted at the bottom, wherever PyYAML can
be imported -- which is CI and any box with the gateway venv.
"""
from __future__ import annotations

import pytest

from fleetctl import hostfile
from fleetctl.hostfile import HostFileError, dumps, loads


class TestScalars:
    @pytest.mark.parametrize("text,want", [
        ("a: 1", {"a": 1}),
        ("a: -3", {"a": -3}),
        ("a: 1.5", {"a": 1.5}),
        ("a: true", {"a": True}),
        ("a: False", {"a": False}),
        ("a: null", {"a": None}),
        ("a: ~", {"a": None}),
        ("a:", {"a": None}),
        ("a: hello world", {"a": "hello world"}),
        ('a: "8080"', {"a": "8080"}),
        ("a: '1'", {"a": "1"}),
        ("a: 0.0.0.0", {"a": "0.0.0.0"}),
    ])
    def test_reads(self, text, want):
        assert loads(text) == want

    def test_a_windows_path_is_a_string_not_an_escape(self):
        assert loads(r"p: C:\llmstack\state") == {"p": r"C:\llmstack\state"}

    def test_a_quoted_number_stays_a_string(self):
        """engine.visible_devices is "1", the device INDEX, and turning it
        into the integer 1 would write `GGML_VK_VISIBLE_DEVICES=1` either way
        -- but the schema check that it is a string would fail, which is the
        point at which somebody notices."""
        assert loads('v: "1"')["v"] == "1"
        assert loads("v: 1")["v"] == 1


class TestStructure:
    def test_nested_maps(self):
        assert loads("a:\n  b:\n    c: 1\n") == {"a": {"b": {"c": 1}}}

    def test_block_list(self):
        assert loads("a:\n  - x\n  - y\n") == {"a": ["x", "y"]}

    def test_inline_list(self):
        assert loads("a: [x, y, z]") == {"a": ["x", "y", "z"]}

    def test_inline_list_empty(self):
        assert loads("a: []") == {"a": []}

    def test_inline_list_keeps_a_quoted_comma(self):
        assert loads('a: ["x, y", z]') == {"a": ["x, y", "z"]}

    def test_comments_and_blank_lines_are_ignored(self):
        text = "# top\n\na: 1   # trailing\n\n# another\nb: 2\n"
        assert loads(text) == {"a": 1, "b": 2}

    def test_a_hash_with_no_space_before_it_is_part_of_the_value(self):
        assert loads("a: red#5")["a"] == "red#5"
        assert loads("a: red #5")["a"] == "red"

    def test_a_key_with_no_block_under_it_is_null(self):
        assert loads("a:\nb: 2\n") == {"a": None, "b": 2}


class TestRefusals:
    @pytest.mark.parametrize("text,fragment", [
        ("a: &anchor 1", "anchors"),
        ("a: *ref", "anchors"),
        ("a: !!str 1", "anchors"),
        ("a: {b: 1}", "flow mappings"),
        ("a: |\n  text\n", "block scalars"),
        ("a: 1\n---\nb: 2\n", "multiple documents"),
        ("a:\n\t- x\n", "tabs"),
        ("a: 1\na: 2\n", "duplicate"),
        ("a 1\n", "expected `key: value`"),
        ("- x\n", "list item where a key was expected"),
        ("a:\n    b: 1\n", "exactly 2 spaces"),
        ("a:\n  - b: 1\n", "lists of mappings"),
        ("a: [x, y", "open and close on one line"),
    ])
    def test_refuses(self, text, fragment):
        with pytest.raises(HostFileError) as exc:
            loads(text)
        assert fragment in str(exc.value)

    def test_the_error_names_the_line(self):
        with pytest.raises(HostFileError) as exc:
            loads("ok: 1\nalso: 2\nbad: {x: 1}\n")
        assert exc.value.line_no == 3
        assert "line 3" in str(exc.value)


class TestRoundTrip:
    DOC = {
        "host": {"name": "gpu-desktop-1", "klass": "gpu"},
        "network": {"port": 8080, "bind": "0.0.0.0",
                    "url": "http://100.64.0.63:8080/v1"},
        "engine": {"kind": "llama.cpp", "visible_devices": "1", "swap": True,
                   "backend": None},
        "sizing": {"vram_gb": 15.9, "headroom": 1.0},
        "access": {"emails": ["a@b.c", "d@e.f"]},
        "paths": {"models": r"D:\AI\models"},
    }

    def test_round_trips(self):
        assert loads(dumps(self.DOC)) == self.DOC

    def test_round_trips_with_comments_attached(self):
        text = dumps(self.DOC, {"network.port": "why 8080",
                                "engine.kind": "two lines\nof prose"},
                     header="a header")
        assert loads(text) == self.DOC
        assert "# why 8080" in text
        assert "# of prose" in text
        assert text.startswith("# a header")

    def test_a_value_that_looks_like_a_number_is_quoted_on_the_way_out(self):
        assert 'v: "1"' in dumps({"v": "1"})
        assert loads(dumps({"v": "1"}))["v"] == "1"

    def test_a_value_that_looks_like_a_bool_is_quoted_on_the_way_out(self):
        for word in ("yes", "no", "on", "off", "true", "null"):
            assert loads(dumps({"v": word}))["v"] == word

    def test_an_empty_mapping_becomes_null(self):
        """Documented, not accidental: `key:` with nothing under it reads back
        as None either way, so {} does not survive. The planner never emits
        one."""
        assert loads(dumps({"a": {}})) == {"a": None}


class TestAgainstRealYaml:
    """Where PyYAML exists, it is the oracle.

    This is the guarantee that matters: not that the subset parser is
    correct in the abstract, but that it reads every file in this repo the
    same way the rest of the world would.
    """

    @staticmethod
    def _yaml():
        return pytest.importorskip("yaml", reason="PyYAML not installed here")

    def test_every_plan_in_the_repo_parses_identically(self, repo):
        yaml = self._yaml()
        files = sorted(repo.glob("hosts/*/host.yml")) + \
            [p for p in (repo / "fleet.yml",) if p.is_file()]
        assert files, "no host.yml files to check"
        for path in files:
            text = path.read_text(encoding="utf-8")
            assert loads(text) == yaml.safe_load(text), path

    @pytest.mark.parametrize("text", [
        "a: 1\nb: two\n",
        "a:\n  b: [1, 2, 3]\n",
        'a: "quoted # hash"\n',
        "a: 0.0.0.0\nb: 100.64.0.1\n",
        "list:\n  - one\n  - two\n",
        "a: null\nb: ~\nc:\n",
        "a: true\nb: false\n",
    ])
    def test_agrees_on_small_documents(self, text):
        yaml = self._yaml()
        assert loads(text) == yaml.safe_load(text)

    def test_what_we_emit_is_readable_by_real_yaml(self):
        yaml = self._yaml()
        text = dumps(TestRoundTrip.DOC, {"network.port": "note"}, header="h")
        assert yaml.safe_load(text) == TestRoundTrip.DOC


def test_module_exports_only_what_it_promises():
    assert set(hostfile.__all__) <= set(vars(hostfile))
