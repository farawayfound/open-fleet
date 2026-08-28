"""A box whose weights disk has gone away must stay on the fleet.

The regression these pin down is apu-tablet-2 on 2026-08-27. Its GGUFs live on an
external SSD; the SSD was disconnected uncleanly and came back lettered F:
instead of D:. `MODELS_DIR.mkdir(parents=True)` runs at module IMPORT, so the
missing drive raised FileNotFoundError before the app object existed, the
SYSTEM task crash-looped the gateway nine times, nothing bound :8080, and the
hub -- correctly, on the evidence it had -- called the machine offline.

The machine was on the whole time. Every one of these tests is about the
difference between "this box is gone" and "this box has lost a disk", because
only one of them is true and the fleet page was showing the other.
"""
from __future__ import annotations

import shutil

import pytest

import app as gw


@pytest.fixture
def volume_gone(monkeypatch, tmp_path):
    """Point MODELS_DIR at a path whose parent does not exist.

    A missing directory is not the interesting case -- the gateway would just
    create it. What apu-tablet-2 hit was a missing VOLUME, which is why the
    fixture reaches for a path no mkdir can rescue and asserts that
    disk_usage really does raise on it, rather than assuming so."""
    gone = tmp_path / "no-such-volume" / "AI" / "models"
    monkeypatch.setattr(gw, "MODELS_DIR", gone)
    with pytest.raises(OSError):
        shutil.disk_usage(str(gone))
    return gone


# --------------------------------------------------------------------------
# the probe survives
# --------------------------------------------------------------------------


def test_models_volume_reports_why_instead_of_raising(volume_gone):
    du, why = gw.models_volume()
    assert du is None
    assert why, "an unreachable volume must come back with a reason, not ''"


def test_storage_info_says_unavailable_rather_than_zero(volume_gone):
    si = gw.storage_info()
    assert si["available"] is False
    assert si["error"]
    assert si["mount"] == str(volume_gone)
    # Zeros are what a full disk with no GGUFs also looks like. `available` is
    # the only field that separates them, so it has to be the one consumers
    # read -- and the dashboard does.
    assert si["gguf_bytes"] == 0 and si["gguf_count"] == 0


def test_unavailable_storage_keeps_the_shape_callers_index_into(monkeypatch,
                                                                tmp_path):
    """Same keys, reachable or not.

    index.html indexes storage fields directly (s.free, s.gguf_count,
    h.disk.used). A short dict here is a TypeError in the browser, which
    blanks the whole Fleet storage table -- a second outage stacked on top of
    the first. So both readings are taken for real and their key sets
    compared, rather than either being written down by hand."""
    here = tmp_path / "models"
    here.mkdir()
    monkeypatch.setattr(gw, "MODELS_DIR", here)
    live = set(gw.storage_info())

    monkeypatch.setattr(gw, "MODELS_DIR", tmp_path / "no-such-volume" / "m")
    missing = set(gw.storage_info())

    # `error` is the one field a healthy reading has no use for.
    assert missing - live == {"error"}
    assert live - missing == set()


def test_host_status_survives_a_missing_model_volume(volume_gone):
    hs = gw.host_status()
    assert hs["disk"] == {"total": 0, "used": 0, "free": 0}
    assert hs["storage"]["available"] is False
    # The rest of the telemetry is about the box, not the disk, and must still
    # be there: this is the payload the hub reads to decide "online".
    assert hs["mem"]["total"] > 0
    assert hs["cpu_count"] >= 1


def test_admin_status_stays_200_with_no_model_volume(client, admin_headers,
                                                     volume_gone):
    """The endpoint the hub polls. A 500 here IS the box going offline."""
    r = client.get("/admin/api/status", headers=admin_headers)
    assert r.status_code == 200
    assert r.json()["host"]["storage"]["available"] is False


# --------------------------------------------------------------------------
# and still tells the truth when the disk is fine
# --------------------------------------------------------------------------


def test_a_reachable_volume_is_reported_available(monkeypatch, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "org__repo").mkdir()
    (models / "org__repo" / "a.gguf").write_bytes(b"GGUF" + b"x" * 4092)
    monkeypatch.setattr(gw, "MODELS_DIR", models)

    si = gw.storage_info()
    assert si["available"] is True
    assert "error" not in si
    assert si["gguf_count"] == 1
    assert si["gguf_bytes"] == 4096
    assert si["total"] > 0
