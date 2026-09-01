"""Test fixtures for the gateway.

LLMSTACK_* must point at a scratch directory BEFORE `app` is imported --
the module reads them at import time to build its config constants -- so
this file sets them up first and only then imports the app module.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

_STATE = tempfile.mkdtemp(prefix="fleetpass-gateway-test-")
os.environ["LLMSTACK_STATE"] = _STATE
os.environ["LLMSTACK_ETC"] = _STATE
os.environ["LLMSTACK_HOME"] = _STATE
os.environ["LLMSTACK_MODELS_DIR"] = str(Path(_STATE) / "models")
os.environ.setdefault("LLMSTACK_ADMIN_TOKEN", "test")
os.environ.setdefault("LLMSTACK_PUBLIC_INTAKE_TOKEN", "intake")
os.environ.setdefault("LLMSTACK_PUBLIC_API_URL", "https://api.example.test/v1")
os.environ.setdefault("LLMSTACK_HOST_NAME", "test-hub")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

import app as gw  # noqa: E402  -- the module under test


@pytest.fixture(scope="session")
def client():
    """One TestClient for the whole session -- its `with` block runs the
    real FastAPI lifespan (db_init, the httpx client, seeding), same as a
    live process would at boot."""
    with TestClient(gw.app) as c:
        yield c


@pytest.fixture(autouse=True)
def _fresh_public_state(client):
    """Every test starts from a clean Fleet Pass slate: the catalogue and
    domain seed stay (other tests, and the seeded rows themselves, rely on
    them), but requests/keys/events/settings from a previous test must not
    leak into the next one's counts (ip/domain/global caps, one-per-email).

    Depends on `client` (not just imports it) so the DB schema always exists
    before the DELETEs below run, regardless of which test file/class pytest
    happens to collect first -- without this, a class that never touches
    `client` directly would run first and crash on a missing table."""
    gw.db_exec("DELETE FROM public_keys")
    gw.db_exec("DELETE FROM public_events")
    gw.db_exec("DELETE FROM usage")
    gw.db_exec("DELETE FROM agents")
    gw.db_exec("DELETE FROM teams")
    gw.db_exec("DELETE FROM api_keys")
    gw.db_exec("DELETE FROM settings WHERE key='public'")
    # Per-(host, model) context ceilings are remembered across hub restarts on
    # purpose, which means they also survive from one test into the next.
    gw.db_exec("DELETE FROM model_ctx")
    gw._public_key_cache.clear()
    gw._public_key_row_cache.clear()
    gw._public_catalogue_cache.update(t=0.0)
    gw._public_overview_cache.update(t=0.0, data=None)
    gw._known_ctx_cache.update(t=0.0, map={}, written=None)
    # Reachability decides which half of a dual-boot machine is eclipsed, so
    # one test's live twin must not silently eclipse the next test's ceilings.
    gw._routes_cache.update(ctx={}, reachable=set())
    # The Configurations tab's routing policy merges into every load_specs()
    # answer, so a rank or reserve flag one test saved would re-rank every
    # scorer assertion after it. Same for the one-slot manual-load state.
    gw.db_exec("DELETE FROM settings WHERE key=?", (gw.FLEET_ROUTING_KEY,))
    gw._specs_cache.update(stamp=None, specs=None, gen=-1)
    gw._manual_load.clear()
    yield


@pytest.fixture
def admin_headers():
    return {"Authorization": "Bearer test"}


@pytest.fixture
def intake_headers():
    return {"X-Intake-Token": "intake"}


@pytest.fixture
def fake_fleet(monkeypatch):
    """A small, fully offline fake fleet: no real HTTP ever leaves the
    process. `gemma4-31b-qat` is resident locally; `gemma-4-26b`/
    `gemma4:26b` (the two fleet ids behind gemma4-26b-a4b) are only
    *available* (present, not resident) locally; `qwen3.8-27b` is resident
    on a peer; `nemotron3-super-120b`'s fleet id is not served anywhere.

    Each (host, model) pair also carries the context ceiling that box would
    serve it with -- deliberately uneven, because a flat number is exactly
    what the hardware-aware sizing exists to replace."""
    cands = {
        "gemma4-31b-qat": [""],
        "gemma-4-26b": [""],
        "gemma4:26b": [""],
        "qwen3.6-35b": [""],
        "qwen3.8-27b": ["peer1"],
    }
    cap = {(c, m): 1 for m, cs in cands.items() for c in cs}
    ctx = {
        ("", "gemma4-31b-qat"): 32768,
        ("", "gemma-4-26b"): 16384,
        ("", "gemma4:26b"): 16384,
        ("", "qwen3.6-35b"): 8192,
        ("peer1", "qwen3.8-27b"): 131072,
    }
    running = {gw.HOST_NAME: {"gemma4-31b-qat"}, "peer1": {"qwen3.8-27b"}}
    gw._routes_cache.update(t=time.time(), map={m: cs[0] for m, cs in cands.items()},
                            cands=cands, cap=cap, running=running, ctx=ctx,
                            reachable={gw.HOST_NAME, "peer1"})
    gw.remember_model_ctx(ctx)
    # known_model_ctx() only counts hosts that are still in the fleet, so the
    # fake fleet has to have a peer list for its remembered ceilings to exist.
    monkeypatch.setattr(gw, "load_peers", lambda: [
        {"name": "peer1", "url": "http://peer1:8080", "token": "t"},
        {"name": "bigbox", "url": "http://bigbox:8080", "token": "t"},
    ])
    gw._known_ctx_cache.update(t=0.0, map={}, written=None)

    async def _fake_model_routes(force: bool = False):
        return gw._routes_cache["map"]

    monkeypatch.setattr(gw, "model_routes", _fake_model_routes)
    yield gw._routes_cache


@pytest.fixture
def captured_mail(monkeypatch):
    """Replaces send_mail with one that records instead of talking SMTP."""
    sent: list[dict] = []

    async def _fake_send_mail(to, subject, text, html=None):
        sent.append({"to": to, "subject": subject, "text": text, "html": html})
        return True, ""

    monkeypatch.setattr(gw, "send_mail", _fake_send_mail)
    return sent
