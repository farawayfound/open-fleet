"""Manual host routing (feat/manual-host-routing): a per-agent
prefer_hosts/exclude_hosts policy, live.

test_public_demo.py's TestAgentHostPolicy covers agent_host_policy() as a
pure function; test_public.py's TestAgentHostPolicyApi/
TestAgentHostPolicyMigration cover the PUT/GET round-trip and the schema
migration. This file is the third leg: does the box the /v1 proxy actually
asks change when an agent profile carries a policy, does GET
/admin/api/resolve's preview agree with it, and does the pre-existing
ctx_limit resort still have the final say over ctx-fit correctness.

No real network I/O: httpx.AsyncClient.send is monkeypatched at the class
level and resolve_targets()/load_peers()/peer_inference_key() are wired to a
fixed candidate list, per the house style in test_proxy_failover.py.

Run with: $SP/venv/bin/python -m pytest gateway/tests -q
"""
from __future__ import annotations

import json

import httpx
import pytest

import app as gw

pytestmark = pytest.mark.asyncio

CHAT_OK_BODY = {
    "id": "chatcmpl-1", "object": "chat.completion", "model": "m",
    "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hi"}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}


@pytest.fixture(autouse=True)
def _isolate_routing():
    """Every test here pokes _routes_cache (ctx/cap) directly or via a wired
    resolve_targets(), so the next test -- in this file or elsewhere in the
    session, since _routes_cache is a module singleton -- must not see what
    one left behind."""
    snapshot = dict(gw._routes_cache)
    gw._inflight.clear()
    gw._host_cooldown.clear()
    yield
    gw._routes_cache.clear()
    gw._routes_cache.update(snapshot)
    gw._inflight.clear()
    gw._host_cooldown.clear()


def _peers(names):
    return [{"name": n, "url": "http://%s:8080" % n, "token": "t"} for n in names]


def _wire_targets(monkeypatch, targets, peer_names=None):
    """Point both the /v1 proxy's routing AND GET /admin/api/resolve's own
    resolve_targets() call at a fixed candidate list -- both read the same
    module-level name at call time, so one patch covers both surfaces."""
    async def _resolve(model, **kw):
        return targets

    async def _pik(cand):
        return "peer-key"

    monkeypatch.setattr(gw, "resolve_targets", _resolve)
    monkeypatch.setattr(gw, "load_peers",
                        lambda: _peers(peer_names or [t[0] for t in targets]))
    monkeypatch.setattr(gw, "peer_inference_key", _pik)


def _agent(key_id: int, *, prefer=None, exclude=None, ctx_limit=None) -> None:
    gw.db_exec(
        "INSERT INTO agents(key_id,enabled,name,allowed_models,force_model,"
        "param_overrides,prefer_hosts,exclude_hosts,ctx_limit,updated_at) "
        "VALUES (?,1,'','[]','','{}',?,?,?,?)",
        (key_id, json.dumps(prefer or []), json.dumps(exclude or []),
         ctx_limit, gw.now()),
    )


def _fake_send_recording(calls: list[str]):
    """200s on whichever host is asked, recording call order -- enough to
    prove which box was tried FIRST without needing a failover scenario."""
    async def fake_send(self, request, stream=False, **kw):
        calls.append(request.url.host)
        return httpx.Response(200, request=request, json=CHAT_OK_BODY)
    return fake_send


# ---------------------------------------------------------------------------
# the /v1 proxy: agent_host_policy actually changes which box is asked first
# ---------------------------------------------------------------------------

class TestV1ProxyAppliesAgentHostPolicy:
    def test_lands_on_the_preferred_unsaturated_box(self, client, monkeypatch):
        _wire_targets(monkeypatch, [("p1", "m"), ("p2", "m")])
        raw, meta = gw.mint_key("host-policy-live-1")
        _agent(int(meta["id"]), prefer=["p2"])

        calls: list[str] = []
        monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send_recording(calls))
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert calls[0] == "p2", \
            "the preferred host must be tried first when neither is saturated"

    def test_excluded_host_is_never_tried(self, client, monkeypatch):
        _wire_targets(monkeypatch, [("p1", "m"), ("p2", "m")])
        raw, meta = gw.mint_key("host-policy-live-2")
        _agent(int(meta["id"]), exclude=["p1"])

        calls: list[str] = []
        monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send_recording(calls))
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert calls == ["p2"], "an excluded host must never be tried at all"

    def test_ctx_limit_resort_still_dominates_agent_preference(
            self, client, monkeypatch):
        # p1 is preferred but proven too small for this key's promised
        # window; p2 is unpreferred but proven to hold the whole thing.
        # ctx-fit must win -- the exact ordering risk the inline comment on
        # the step-4 insertion point exists to guard.
        _wire_targets(monkeypatch, [("p1", "m"), ("p2", "m")])
        raw, meta = gw.mint_key("host-policy-live-3")
        _agent(int(meta["id"]), prefer=["p1"], ctx_limit=8192)
        gw._routes_cache["ctx"] = {("p1", "m"): 2048, ("p2", "m"): 100000}

        calls: list[str] = []
        monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send_recording(calls))
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert calls[0] == "p2", \
            ("a box proven to hold the whole promised window beats a "
             "preferred box proven too small for it")


# ---------------------------------------------------------------------------
# GET /admin/api/resolve: the preview agrees with what the proxy just did
# ---------------------------------------------------------------------------

class TestResolvePreviewMatchesLiveRouting:
    def test_agent_param_reorders_the_preview_the_same_way(
            self, client, monkeypatch, admin_headers):
        _wire_targets(monkeypatch, [("p1", "m"), ("p2", "m")])
        raw, meta = gw.mint_key("host-policy-live-4")
        kid = int(meta["id"])
        _agent(kid, prefer=["p2"])

        r = client.get("/admin/api/resolve", headers=admin_headers,
                       params={"model": "m", "agent": kid})
        assert r.status_code == 200, r.text
        assert r.json()["targets"][0]["host"] == "p2", \
            "the preview must apply the same policy the live proxy call used"

    def test_no_agent_param_is_unchanged_from_before(
            self, client, monkeypatch, admin_headers):
        _wire_targets(monkeypatch, [("p1", "m"), ("p2", "m")])
        raw, meta = gw.mint_key("host-policy-live-5")
        # This key HAS a real policy on file -- proving it is the missing
        # query param, not an absent profile, that keeps the order plain.
        _agent(int(meta["id"]), prefer=["p2"])

        r = client.get("/admin/api/resolve", headers=admin_headers,
                       params={"model": "m"})
        assert r.status_code == 200, r.text
        hosts = [t["host"] for t in r.json()["targets"]]
        assert hosts == ["p1", "p2"], \
            "omitting agent= must leave resolve_targets()'s own order untouched"

    def test_disabled_or_nonexistent_agent_degrades_to_unmodified_order(
            self, client, monkeypatch, admin_headers):
        _wire_targets(monkeypatch, [("p1", "m"), ("p2", "m")])
        raw, meta = gw.mint_key("host-policy-live-6")
        kid = int(meta["id"])
        _agent(kid, prefer=["p2"])
        gw.db_exec("UPDATE agents SET enabled=0 WHERE key_id=?", (kid,))

        r = client.get("/admin/api/resolve", headers=admin_headers,
                       params={"model": "m", "agent": kid})
        assert r.status_code == 200, r.text
        assert [t["host"] for t in r.json()["targets"]] == ["p1", "p2"], \
            "a disabled profile must never error the preview -- just degrade"

        # A key_id naming no agent row at all behaves the same way.
        r2 = client.get("/admin/api/resolve", headers=admin_headers,
                        params={"model": "m", "agent": 9_999_999})
        assert r2.status_code == 200, r2.text
        assert [t["host"] for t in r2.json()["targets"]] == ["p1", "p2"]


# ---------------------------------------------------------------------------
# key_id (C1/C2): round-trips from _job_open through _job_view, and the /v1
# proxy's real per-key inference call site stamps it -- the dashboard's
# manual-routing controls (C4/C5) gate on this being present and correct.
# ---------------------------------------------------------------------------

class TestJobViewKeyId:
    def test_inference_job_open_with_key_id_round_trips_through_job_view(self):
        rec = gw._job_open("inference", key_id=7, what="m")
        try:
            assert gw._job_view(rec)["key_id"] == 7
        finally:
            gw._job_close(rec)

    def test_a_job_opened_without_key_id_serializes_as_null_not_a_keyerror(self):
        # team/demo/native/download/batch jobs never set key_id.
        rec = gw._job_open("team", what="team:x")
        try:
            assert gw._job_view(rec)["key_id"] is None
        finally:
            gw._job_close(rec)

    def test_the_v1_proxy_stamps_key_id_on_its_job(self, client, monkeypatch):
        # C1: openai_proxy's per-key job_open -- the ONE real fleet-routed
        # per-key inference call site (the native-proxy job_open is never
        # fleet-routed and is deliberately left unchanged; team/demo carry no
        # real per-key id at all) -- passes key_id=int(key['id']).
        _wire_targets(monkeypatch, [("p1", "m")])
        raw, meta = gw.mint_key("job-key-id-test")
        kid = int(meta["id"])

        seen: dict = {}
        orig_job_open = gw._job_open

        def _spy(kind, **kw):
            if kind == "inference":
                seen["key_id"] = kw.get("key_id")
            return orig_job_open(kind, **kw)

        monkeypatch.setattr(gw, "_job_open", _spy)
        monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send_recording([]))
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert seen.get("key_id") == kid
