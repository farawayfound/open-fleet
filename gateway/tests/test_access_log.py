"""LLMSTACK_ACCESS_LOG: one JSON line per metered request, for a log shipper.

Off by default: the SQLite usage table is the gateway's own record, and a
box with no observability stack must write nothing extra. On, every
record_usage() call appends one self-describing JSON object -- the ECS
fields a Kibana data view reads as-is (@timestamp, host.name, url.path,
http.response.status_code, event.duration, event.outcome) plus the gateway's
own numbers under `llmstack.*` -- and a failure to open or write the file
never reaches the request path.
"""
from __future__ import annotations

import json
import logging

import app as gw  # conftest has set the env and sys.path by the time this loads

KEY = {"id": 7, "name": "tester", "secret": "sk-ffa-NEVER-IN-A-LOG"}


def _reset(monkeypatch, path) -> None:
    monkeypatch.setattr(gw, "ACCESS_LOG", str(path) if path else "")
    monkeypatch.setattr(gw, "_access_logger", None)
    monkeypatch.setattr(gw, "_access_failed", False)


def _hit(status: int = 200, model: str = "qwen3.8-9b-distill", **kw) -> None:
    gw.record_usage(
        KEY, model, "/v1/chat/completions", False, status,
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        50, 120, **kw,
    )


class TestOffByDefault:
    def test_no_path_means_no_logger_and_no_file(self, monkeypatch, tmp_path):
        _reset(monkeypatch, None)
        assert gw.access_logger() is None
        _hit()
        assert list(tmp_path.iterdir()) == []


class TestOneLinePerRequest:
    def test_each_metered_request_appends_one_json_object(self, monkeypatch, tmp_path):
        path = tmp_path / "logs" / "access.ndjson"      # the parent does not exist yet
        _reset(monkeypatch, path)
        _hit()
        _hit(status=502, model="", fallback_from="gpu-laptop-1", host="gpu-desktop-1")
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        first, second = (json.loads(ln) for ln in lines)

        assert first["@timestamp"].endswith("+00:00")
        assert first["event"] == {"dataset": "llmstack.requests", "kind": "event",
                                  "outcome": "success", "duration": 120 * 1_000_000}
        assert first["host"] == {"name": gw.HOST_NAME}
        assert first["url"] == {"path": "/v1/chat/completions"}
        assert first["http"] == {"response": {"status_code": 200}}
        assert first["llmstack"] == {
            "key_id": 7, "key_name": "tester", "model": "qwen3.8-9b-distill",
            "stream": False, "prompt_tokens": 10, "completion_tokens": 5,
            "total_tokens": 15, "ttft_ms": 50, "latency_ms": 120,
            "served_by": None, "fallback_from": None,
        }

        assert second["event"]["outcome"] == "failure"
        assert second["http"]["response"]["status_code"] == 502
        assert second["llmstack"]["model"] is None
        assert second["llmstack"]["served_by"] == "gpu-desktop-1"
        assert second["llmstack"]["fallback_from"] == "gpu-laptop-1"

    def test_the_line_never_carries_the_key_itself(self, monkeypatch, tmp_path):
        path = tmp_path / "access.ndjson"
        _reset(monkeypatch, path)
        _hit()
        text = path.read_text(encoding="utf-8")
        assert "sk-ffa" not in text and "secret" not in text

    def test_the_row_and_the_line_share_one_timestamp(self, monkeypatch, tmp_path):
        path = tmp_path / "access.ndjson"
        _reset(monkeypatch, path)
        _hit()
        line = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
        row = gw.db_query("SELECT ts FROM usage WHERE key_id=? ORDER BY id DESC LIMIT 1", (7,))[0]
        assert row["ts"] == line["@timestamp"]

    def test_the_writer_stays_out_of_uvicorns_stream(self, monkeypatch, tmp_path):
        _reset(monkeypatch, tmp_path / "access.ndjson")
        lg = gw.access_logger()
        assert lg is not None and lg.propagate is False
        assert gw.access_logger() is lg      # built once, reused


class TestNeverBreaksInference:
    def test_an_unopenable_path_turns_the_feature_off_and_says_so_once(
            self, monkeypatch, tmp_path, caplog):
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("x", encoding="utf-8")
        _reset(monkeypatch, blocker / "access.ndjson")   # mkdir fails: the parent is a file
        with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
            _hit()
            _hit()
        assert gw.access_logger() is None
        assert gw._access_failed is True
        said = [r for r in caplog.records if "access log" in r.getMessage()]
        assert len(said) == 1
        assert "sk-ffa" not in said[0].getMessage()
        # The request was still metered.
        assert gw.db_query("SELECT COUNT(*) n FROM usage WHERE key_id=?", (7,))[0]["n"] == 2
