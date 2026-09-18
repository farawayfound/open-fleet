"""Windows model paths survive the llama-swap render.

Regression for 2026-09-15: SAFE_PATH excluded the backslash, so on a Windows
box (apu-tablet-1, gpu-laptop-3, apu-tablet-2) every record's path read as "contains a
shell metacharacter" and render_swap_config() emitted `models: {}` -- llama-swap
came up knowing no model at all, and a manual load answered 404. The value is
one argv token inside llama-swap's YAML `cmd`, never a POSIX shell word, so a
backslash is not an injection vector; the real hazards (whitespace, quotes,
backticks, `$`, `;`, `|`, redirections, a leading `-`) stay rejected."""
import pytest
import yaml

# `import app as gw`, the way every other test in this directory does it, and
# NOT `from conftest import gw`: the export workflow runs
# `pytest gateway/tests fleetctl/tests …` from the repo root, and with no
# __init__.py in either directory a bare `conftest` resolves to whichever
# one pytest inserted into sys.path first — fleetctl's, which has no `gw`.
# conftest.py has already put `gateway/` on sys.path by the time this runs.
import app as gw  # noqa: E402

WIN_PATHS = [
    r"C:\llmstack\state\models\unsloth__Qwen3.8-Flash-Next-GGUF\Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf",
    r"D:\AI\models\lmstudio-community\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf",
    r"C:\llmstack\state\models\unsloth__Qwen3.8-Flash-Next-GGUF\mmproj-F16.gguf",
]


class TestWindowsPathsAreSafe:
    @pytest.mark.parametrize("value", WIN_PATHS)
    def test_a_windows_path_is_not_unsafe(self, value):
        assert gw.unsafe_path(value) is None

    @pytest.mark.parametrize("value", [
        r"C:\models\has space.gguf",
        r"C:\models\x.gguf; rm -rf /",
        r"C:\models\x.gguf`whoami`",
        r"C:\models\$(x).gguf",
        r"-ngl 999",
    ])
    def test_the_real_hazards_still_fail(self, value):
        assert gw.unsafe_path(value) is not None


class TestRenderKeepsWindowsModels:
    def test_a_windows_record_lands_in_the_swap_config(self):
        rec = dict(gw.DEFAULT_MODEL_RECORD, id="qwen3.8-flash-next",
                   path=WIN_PATHS[0], mmproj=WIN_PATHS[2], preload=True, ttl=0)
        cfg = yaml.safe_load(gw.render_swap_config([rec]))
        assert "qwen3.8-flash-next" in cfg["models"], cfg
        assert WIN_PATHS[0] in cfg["models"]["qwen3.8-flash-next"]["cmd"]
        # preload survives too: the on_startup hook is what warms a standby box
        assert cfg["hooks"]["on_startup"]["preload"] == ["qwen3.8-flash-next"]

    def test_the_swap_config_is_never_empty_for_enabled_windows_records(self):
        recs = [dict(gw.DEFAULT_MODEL_RECORD, id=f"m{i}", path=p) for i, p in enumerate(WIN_PATHS[:2])]
        cfg = yaml.safe_load(gw.render_swap_config(recs))
        assert set(cfg["models"]) == {"m0", "m1"}
