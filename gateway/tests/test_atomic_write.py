"""Two writers, one file.

The hosted runner found this before a person did: an export's Gate 1 failed on
`FileNotFoundError: .../models.json.tmp -> .../models.json`, raised from
`save_models` -- from the two lines whose entire purpose is to make the write
survive a crash. Every writer used the same temp name, so a second writer that
arrived between the first one's write and its replace left nothing for the
first to rename. It surfaced as a flaky test; on a box it is an admin save
losing to the apply queue's rollback, and the loser raising instead of losing.
"""
from __future__ import annotations

import json
import os

import pytest

import app as gw


class TestTwoWritersAtOnce:

    def test_a_writer_that_arrives_mid_flight_does_not_take_the_temp_file(
            self, tmp_path, monkeypatch):
        """The old race, made deterministic and single-threaded.

        The inner write starts and finishes while the outer one is between its
        write and its replace -- the exact interleaving the runner hit. With a
        temp name derived from the target, the inner writer renamed the file
        the outer one was about to rename, and the outer raised. With a temp
        file per writer, the second write simply wins the target.
        """
        target = tmp_path / "models.json"
        temps: list[str] = []
        real_replace = os.replace
        arrived = {"inner": False}

        def replace(src, dst):
            temps.append(os.path.basename(src))
            if not arrived["inner"]:
                arrived["inner"] = True
                gw.write_atomic(target, json.dumps([{"id": "inner"}]))
            return real_replace(src, dst)

        monkeypatch.setattr(gw.os, "replace", replace)
        gw.write_atomic(target, json.dumps([{"id": "outer"}]))

        assert len(temps) == 2, "both writes should have reached replace"
        assert len(set(temps)) == 2, f"the two writers shared a temp file: {temps}"
        # the outer writer replaced last, so its content is what survives
        assert json.loads(target.read_text()) == [{"id": "outer"}]
        assert [p.name for p in tmp_path.iterdir()] == ["models.json"]

    def test_a_write_that_fails_leaves_nothing_behind(self, tmp_path, monkeypatch):
        """A temp file per writer is only an improvement if it is cleaned up."""
        target = tmp_path / "llama-swap.yaml"
        gw.write_atomic(target, "models: {}\n")

        def boom(src, dst):
            raise RuntimeError("disk went away")

        monkeypatch.setattr(gw.os, "replace", boom)
        with pytest.raises(RuntimeError):
            gw.write_atomic(target, "models: {broken}\n")

        assert [p.name for p in tmp_path.iterdir()] == ["llama-swap.yaml"]
        assert target.read_text() == "models: {}\n"


class TestWhoCanReadTheResult:
    """mkstemp creates at 0600. llama-swap reads the config this writes, and on
    a box where it runs as another user, 0600 is an outage."""

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
    def test_a_new_file_is_readable_by_the_service_that_reads_it(self, tmp_path):
        target = tmp_path / "llama-swap.yaml"
        gw.write_atomic(target, "models: {}\n")
        assert target.stat().st_mode & 0o044 == 0o044

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
    def test_an_existing_file_keeps_the_mode_it_had(self, tmp_path):
        """peers.json is chmod 0600 on purpose -- it holds peer admin tokens."""
        target = tmp_path / "peers.json"
        target.write_text("[]")
        target.chmod(0o600)
        gw.write_atomic(target, json.dumps([{"name": "hub", "url": "http://x"}]))
        assert target.stat().st_mode & 0o777 == 0o600


class TestTheCallersStillWork:

    def test_models_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "sub" / "models.json")
        gw.save_models([dict(gw.DEFAULT_MODEL_RECORD, id="m1", path="/models/m1.gguf")])
        assert [m["id"] for m in gw.load_models()] == ["m1"]

    def test_peers_round_trip_and_stay_private(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gw, "PEERS_PATH", tmp_path / "peers.json")
        gw.save_peers([{"name": "hub", "url": "http://hub:8080"}])
        assert [p["name"] for p in gw.load_peers()] == ["hub"]
        if os.name != "nt":
            assert (tmp_path / "peers.json").stat().st_mode & 0o777 == 0o600
