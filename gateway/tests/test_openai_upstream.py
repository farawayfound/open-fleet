"""A box whose engine wants a key of its own: the npu-box-1 node.

The npu-box-1's vendor server ("npu-box-1 Server") is OpenAI-compatible, but neither
llama-swap nor Ollama. It wants `Authorization: Bearer <device key>` on every
/v1/* request and on everything under /api/ -- where its device-management
API lives (reboot, shutdown, passwords) -- answers llama-swap's /running with
a 404, and lists exactly one model in /v1/models: by its Hugging Face repo
id, with a vendor `alias` field and a declared `context_window`.

What is pinned, and why:
  * The device key goes to this box's own engine, on /v1/*, and nowhere
    else. Not to /api/* (signed, it would hand every gateway key the
    device's management API), not to a peer (which gets the hub's key for
    that peer), and never alongside a caller's own credential: a caller's
    gateway key, cookie and Access JWT stop at the gateway, and a gateway key
    passed as ?api_key= is dropped from the forwarded query string --
    forwarded, it would sit in the next hop's access log.
  * A caller the gateway refused (a wrong, disabled or archived key, or none
    at all) reaches nothing. The engine client attaches the device key by
    itself, so the gateway must never become a way to borrow it.
  * The engine refusing the DEVICE key (a rotated key, a stale key file) is
    this box's fault: the caller gets a 502 that says so, metered as one --
    not a 401 telling them their own key is wrong.
  * Residency on an engine with no notion of loading: when both residency
    probes answer 4xx, it holds what it lists. A 5xx or no answer is a broken
    engine, not this kind, and Ollama and llama-swap keep reading their own
    probe.
  * The catalogue: the device's id and the fleet's canonical id
    (glm-4.7-flash) are both served, with the declared context window. The
    vendor's `alias: default` is not a model name -- `default` is a fleet
    role.
  * engine_info() names such an engine, and a touch there is a no-op success
    (a 501 would read as a failure to the hub's keep-warm loop); an Ollama
    box still answers 501.
  * The key file is read stripped, and a bad one is reported by its path --
    never by its content -- without taking the import down with it.
  * '.' and '..' path segments are refused by both proxies before anything
    reaches the engine. httpx resolves them while building the upstream URL,
    so /v1/%2e%2e/api/... would step off the surface the caller's key was
    checked for, and /api/chat/%2e%2e/delete would pass the native allow-list
    and land on Ollama's model delete. So is any '%', '\\' or ';' left after
    the ONE decode a server does: an engine that decodes again would resolve
    '%2e%2e' into another path. Starlette's TestClient decodes twice, which
    hides exactly that case, so it is also put through httpx.ASGITransport,
    which decodes once, as uvicorn does.
  * What a box holding a key asks its engine for is the gateway's decision:
    only the OpenAI calls the fleet makes (anything else under /v1 is a
    metered 404 before the engine is asked -- a vendor hangs settings, model
    deletion and power there), with only the caller's content-type, accept
    and user-agent -- never a routing hint a vendor stack might honour. The
    key is judged on the path the engine will end up serving (decoded and
    normalised), sent only to the engine's own origin, and never through a
    proxy named in the environment. A box with no key keeps every old
    answer, its engine's own 401/403 included.
  * The native num_ctx retry drops ?api_key= as the first attempt does, and
    swap_running() reports an engine that holds what it lists under the
    fleet's id, with the window it declares.

No real network I/O. This box's engine is an httpx.MockTransport behind
gw.client that records every request as it went on the wire -- built with
gw.upstream_client() whenever the engine wants a key, so the real request
hook does the signing. A peer is reached through httpx.AsyncHTTPTransport,
patched at class level BELOW AsyncClient.send, so any hook on a peer client
still runs and is seen. engine_info()'s synchronous probes go through the
same fake engine via a patched httpx.get. Routing is pinned the way
test_proxy_failover.py's _wire_targets does it.

Run with (from the repo root):
  uv run --offline --no-project
    --with-requirements gateway/requirements.txt
    --with-requirements gateway/requirements-dev.txt
    python -m pytest gateway/tests/test_openai_upstream.py -q -p no:cacheprovider
"""
from __future__ import annotations

import json
import logging
import time

import httpx
import pytest
from starlette.requests import Request

import app as gw

DEVICE_KEY = "dev-key-123"        # the key this box holds for its engine (fake)
ROTATED_KEY = "rotated-000"  # what the engine wants after a rotation (fake)  # gitleaks:allow
BEARER = "Bearer " + DEVICE_KEY
SIGNED = {"authorization": BEARER}

DEVICE_ID = "zai-org/GLM-4.7-Flash"   # how the npu-box-1 lists the model
FLEET_ID = "glm-4.7-flash"            # the fleet's canonical id for it

NOT_FOUND = (404, {"error": "not found"})
BROKEN = (500, {"error": "internal server error"})

VENDOR_MODELS = {"object": "list", "data": [
    {"id": DEVICE_ID, "object": "model", "owned_by": "npu-box-1", "alias": "default",
     "context_window": 131072}]}
OLLAMA_MODELS = {"object": "list", "data": [
    {"id": "llama3:8b", "object": "model", "created": 1700000000, "owned_by": "library"},
    {"id": "qwen3.5:4b", "object": "model", "created": 1700000000, "owned_by": "library"}]}
OLLAMA_PS = {"models": [{"name": "llama3:8b", "model": "llama3:8b", "size": 4920000000,
                         "size_vram": 4920000000, "context_length": 8192,
                         "expires_at": "2026-10-14T12:00:00Z"}]}
SWAP_RUNNING = {"running": [{"model": "qwen3.5-4b", "state": "ready"}]}

# What a vendor may hang under /v1 beside the OpenAI calls -- and would obey,
# signed with the device key.
VENDOR_EXTRAS = {"/v1/models/x": (200, {"deleted": True}), "/v1/settings": (200, {}),
                 "/v1/system/shutdown": (200, {"status": "shutting down"})}

# Caller headers on one chat request: the three a keyed box passes on, and
# three routing hints a vendor stack might honour.
SENT_HEADERS = {"content-type": "application/json", "accept": "application/json",
                "user-agent": "fleet-test/1.0", "x-http-method-override": "DELETE",
                "x-original-url": "/api/v1/sys/reboot", "x-forwarded-for": "203.0.113.7"}
ROUTING_HINTS = ("x-http-method-override", "x-original-url", "x-forwarded-for")

OOM_BODY = {"error": "cuda error: out of memory when allocating the kv cache"}
NATIVE_OK_BODY = {"message": {"role": "assistant", "content": "ok"}, "done": True,
                  "prompt_eval_count": 3, "eval_count": 2}

ENGINE_ORIGIN = httpx.URL(gw.UPSTREAM)

CHAT_OK_BODY = {
    "id": "chatcmpl-1", "object": "chat.completion", "model": DEVICE_ID,
    "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hi"}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}

# Everything a residency probe may ask -- and so everything a touch on an
# engine that holds what it lists may ask.
PROBES = {("GET", "/running"), ("GET", "/api/ps"), ("GET", "/v1/models")}


class Engine:
    """This box's own engine, as gw.client reaches it.

    Records every request as it went on the wire (after the client's request
    hooks ran) and answers from a path -> (status, json) table; an exception
    in the table is raised instead, the way an unreachable engine surfaces,
    and a list is one answer per call, its last one repeating. Paths under
    `guarded` want `Bearer <key>` first and get `refuse` without it.
    Anything else is a 404, as on a real server."""

    def __init__(self, routes: dict, key: str = "", guarded: tuple[str, ...] = (),
                 refuse: int = 401):
        self.routes, self.key, self.guarded, self.refuse = routes, key, guarded, refuse
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if (path.startswith(self.guarded)
                and request.headers.get("authorization") != "Bearer " + self.key):
            return httpx.Response(self.refuse, json={"error": {"message": "invalid api key"}})
        answer = self.routes.get(path, NOT_FOUND)
        if isinstance(answer, list):
            answer = answer.pop(0) if len(answer) > 1 else answer[0]
        if isinstance(answer, Exception):
            raise answer
        status, body = answer
        return httpx.Response(status, json=body)

    def to(self, path: str) -> list[httpx.Request]:
        return [q for q in self.requests if q.url.path == path]

    def wire(self) -> list[str]:
        return [q.method + " " + q.url.path for q in self.requests]


def vendor_engine(running=NOT_FOUND, models=VENDOR_MODELS, key=DEVICE_KEY, refuse=401,
                  extra: dict | None = None) -> Engine:
    """The npu-box-1's vendor server: /v1/* and everything under /api/ want the
    device key -- which the gateway never sends to /api/, so /api/ps is a 401
    -- and llama-swap's /running is a 404. `extra` adds routes it serves."""
    return Engine({"/running": running, "/v1/models": (200, models),
                   "/v1/chat/completions": (200, CHAT_OK_BODY), **(extra or {})},
                  key=key, guarded=("/v1/", "/api/"), refuse=refuse)


def openai_server(running=NOT_FOUND, ps=NOT_FOUND) -> Engine:
    """A bare OpenAI-compatible server holding the same device key on /v1/*,
    with nothing at all under /api/ unless told otherwise."""
    return Engine({"/running": running, "/api/ps": ps, "/v1/models": (200, VENDOR_MODELS),
                   "/v1/chat/completions": (200, CHAT_OK_BODY)},
                  key=DEVICE_KEY, guarded=("/v1/",))


def ollama() -> Engine:
    """An Ollama box: no key anywhere, /api/ps lists what is loaded and
    /v1/models every tag pulled. /api/delete is what a traversal would hit."""
    return Engine({"/api/ps": (200, OLLAMA_PS), "/v1/models": (200, OLLAMA_MODELS),
                   "/v1/chat/completions": (200, CHAT_OK_BODY), "/api/delete": (200, {})})


def llama_swap() -> Engine:
    return Engine({"/running": (200, SWAP_RUNNING)})


def _box(monkeypatch, engine: Engine, upstream_models: bool = True) -> Engine:
    """Put `engine` behind this box's gw.client, the way lifespan does. An
    engine that wants a key gets a box holding one (DEVICE_KEY) and the real
    signing client; any other gets a plain client -- all a box with no key
    file has, in effect."""
    monkeypatch.setattr(gw, "UPSTREAM_MODELS", upstream_models)
    transport = httpx.MockTransport(engine)
    if engine.key:
        monkeypatch.setattr(gw, "UPSTREAM_KEY", DEVICE_KEY)
        monkeypatch.setattr(gw, "client", gw.upstream_client(transport=transport))
    else:
        monkeypatch.setattr(gw, "client",
                            httpx.AsyncClient(base_url=gw.UPSTREAM, transport=transport))
    return engine


def _keyless_box(monkeypatch, engine: Engine, upstream_models: bool = False) -> Engine:
    """A box with no key file -- llama-swap, or Ollama with upstream_models --
    on the REAL engine client, which then has nothing to sign. None of a keyed
    box's limits may reach it."""
    monkeypatch.setattr(gw, "UPSTREAM_MODELS", upstream_models)
    monkeypatch.setattr(gw, "UPSTREAM_KEY", "")
    monkeypatch.setattr(gw, "client",
                        gw.upstream_client(transport=httpx.MockTransport(engine)))
    return engine


def _route_to(monkeypatch, *targets: tuple[str, str], peers: tuple[str, ...] = ()) -> None:
    """Pin the /v1 proxy's candidates, as test_proxy_failover.py's
    _wire_targets does: resolve_targets() and peer_inference_key() would
    otherwise poll the fleet. '' is this box."""
    async def _resolve(model, **kw):
        return list(targets)

    async def _pik(cand):
        return "k"   # the hub's inference key for that peer

    monkeypatch.setattr(gw, "resolve_targets", _resolve)
    monkeypatch.setattr(gw, "peer_inference_key", _pik)
    monkeypatch.setattr(gw, "load_peers", lambda: [
        {"name": n, "url": "http://%s:8080" % n, "token": "t"} for n in peers])


def _peer_wire(monkeypatch) -> list[httpx.Request]:
    """Every request a peer client puts on the wire. Patched at the transport,
    not at AsyncClient.send, so the client's own hooks still run first: a
    device-key hook wrongly attached to a peer client would be seen here."""
    sent: list[httpx.Request] = []

    async def handle(self, request):
        sent.append(request)
        return httpx.Response(200, json=CHAT_OK_BODY)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle)
    return sent


def _chats_to(sent: list[httpx.Request], host: str) -> list[httpx.Request]:
    return [q for q in sent if q.url.host == host and q.url.path == "/v1/chat/completions"]


def _chat(client, key: str | None = None, *, url: str = "/v1/chat/completions",
          headers: dict | None = None, stream: bool = False):
    """One chat request for the fleet id. X-Fleet-No-Fallback keeps the
    catalogue's substitution step (which polls the fleet) out of it."""
    h = {"X-Fleet-No-Fallback": "1", **(headers or {})}
    if key is not None:
        h["Authorization"] = "Bearer " + key
    return client.post(url, headers=h, json={
        "model": FLEET_ID, "stream": stream,
        "messages": [{"role": "user", "content": "hi"}]})


def _ask(client, method: str, url: str, key: str, body: dict | None = None):
    """Any request with a gateway key, the substitution step kept out of it
    as in _chat."""
    return client.request(method, url, json=body, headers={
        "X-Fleet-No-Fallback": "1", "Authorization": "Bearer " + key})


@pytest.fixture(autouse=True)
def _this_box(monkeypatch, tmp_path):
    """Each test owns this box's engine and starts from cold caches.

    The box is out of use (standby) for the test: the lifespan's background
    loops -- the overview refresh every ~10 s, warm-on-load every 15 s on
    Windows -- then idle instead of sending probes of their own through
    whichever fake engine the test put behind gw.client, where a stray
    GET /running would fail a "nothing reached the engine" assertion. Nothing
    under test reads standby."""
    monkeypatch.setitem(gw._standby, "loaded", True)
    monkeypatch.setitem(gw._standby, "standby", True)
    monkeypatch.setattr(gw, "_running_cache", {**gw._running_cache, "t": 0.0, "ids": set()})
    monkeypatch.setattr(gw, "_upstream_cache",
                        {**gw._upstream_cache, "t": 0.0, "ids": set(), "ctx": {}})
    monkeypatch.setattr(gw, "_engine_cache", {"t": 0.0, "info": {}})
    # Ollama's /api/show sizing is not under test. A fresh, empty answer keeps
    # upstream_model_ctx() from starting its background refresh mid-test.
    monkeypatch.setattr(gw, "_upstream_ctx_cache", {"t": time.time(), "ctx": {}})
    # A box serving its engine's catalogue has no models.json, and one another
    # test left behind would shadow the engine's own names.
    monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")


# ---------------------------------------------------------------------------
# upstream_auth() and the engine client's request hook
# ---------------------------------------------------------------------------

class TestUpstreamAuth:
    @pytest.mark.parametrize("key, path, expected", [
        pytest.param(DEVICE_KEY, "/v1/chat/completions", SIGNED, id="chat-is-signed"),
        pytest.param(DEVICE_KEY, "/v1/models", SIGNED, id="model-list-is-signed"),
        pytest.param(DEVICE_KEY, "/api/chat", {}, id="native-api-is-not-signed"),
        pytest.param(DEVICE_KEY, "/api/v1/sys/reboot", {}, id="device-management-api-is-not-signed"),
        pytest.param(DEVICE_KEY, "/running", {}, id="residency-probe-is-not-signed"),
        pytest.param(DEVICE_KEY, "/health", {}, id="health-is-not-signed"),
        pytest.param(DEVICE_KEY, "/upstream/x/health", {}, id="llama-swap-touch-is-not-signed"),
        pytest.param(DEVICE_KEY, "/v1", {}, id="bare-v1-without-slash-is-not-signed"),
        pytest.param(DEVICE_KEY, "/v1/../api/v1/sys/reboot", {},
                     id="v1-path-normalising-into-the-device-api-is-not-signed"),
        pytest.param(DEVICE_KEY, "/v1/%2e%2e/api/x", {},
                     id="encoded-dotdot-out-of-v1-is-not-signed"),
        pytest.param(DEVICE_KEY, "/v1/%252e%252e/api/x", {},
                     id="double-encoded-dotdot-out-of-v1-is-not-signed"),
        pytest.param(DEVICE_KEY, "/v1/models/a%2Fb", SIGNED,
                     id="encoded-slash-that-stays-in-v1-is-signed"),
        pytest.param(DEVICE_KEY, "/v1/..\\api/x", {},
                     id="backslash-dotdot-out-of-v1-is-not-signed"),
        pytest.param("", "/v1/chat/completions", {}, id="no-key-configured-chat-is-not-signed"),
        pytest.param("", "/v1/models", {}, id="no-key-configured-model-list-is-not-signed"),
    ])
    def test_signs_only_the_openai_surface_and_only_with_a_key(
            self, monkeypatch, key, path, expected):
        monkeypatch.setattr(gw, "UPSTREAM_KEY", key)
        assert gw.upstream_auth(path) == expected


class TestEngineClient:
    pytestmark = pytest.mark.asyncio

    @pytest.mark.parametrize("method, path, wire, auth", [
        pytest.param("POST", "/v1/chat/completions", "/v1/chat/completions", BEARER,
                     id="chat-carries-the-device-key"),
        pytest.param("GET", "/v1/models", "/v1/models", BEARER,
                     id="model-list-carries-the-device-key"),
        pytest.param("GET", "/api/v1/sys/status", "/api/v1/sys/status", None,
                     id="device-api-carries-no-key"),
        pytest.param("GET", "/running", "/running", None,
                     id="residency-probe-carries-no-key"),
        pytest.param("GET", "/v1/../api/v1/sys/status", "/api/v1/sys/status", None,
                     id="v1-path-resolving-into-the-device-api-carries-no-key"),
    ])
    async def test_signs_by_the_path_that_goes_on_the_wire(
            self, monkeypatch, method, path, wire, auth):
        monkeypatch.setattr(gw, "UPSTREAM_KEY", DEVICE_KEY)
        engine = Engine({})
        async with gw.upstream_client(transport=httpx.MockTransport(engine)) as c:
            await c.request(method, path)
        assert [(str(q.url), q.headers.get("authorization")) for q in engine.requests] == [
            (gw.UPSTREAM + wire, auth)]

    @pytest.mark.parametrize("url, auth", [
        pytest.param(str(ENGINE_ORIGIN.copy_with(path="/v1/chat/completions")), BEARER,
                     id="the-engines-own-origin-is-signed"),
        pytest.param("http://evil.example/v1/chat/completions", None,
                     id="another-host-is-not"),
        pytest.param(str(ENGINE_ORIGIN.copy_with(port=(ENGINE_ORIGIN.port or 80) + 1,
                                                 path="/v1/chat/completions")), None,
                     id="the-engines-host-on-another-port-is-not"),
        pytest.param(str(ENGINE_ORIGIN.copy_with(scheme="https", path="/v1/chat/completions")),
                     None, id="the-engines-host-over-another-scheme-is-not"),
    ])
    async def test_signs_only_requests_to_the_engines_own_origin(self, monkeypatch, url, auth):
        monkeypatch.setattr(gw, "UPSTREAM_KEY", DEVICE_KEY)
        engine = Engine({})
        async with gw.upstream_client(transport=httpx.MockTransport(engine)) as c:
            await c.post(url, json={})
        assert [q.headers.get("authorization") for q in engine.requests] == [auth]

    async def test_ignores_proxies_named_in_the_environment(self):
        c = gw.upstream_client()
        try:
            assert c.trust_env is False
        finally:
            await c.aclose()


# ---------------------------------------------------------------------------
# /v1/chat/completions through the gateway, end to end
# ---------------------------------------------------------------------------

def _wrong_key(client, admin_headers) -> str:
    return "sk-ffa-" + "0" * 48


def _disabled_key(client, admin_headers) -> str:
    raw, meta = gw.mint_key("npu-box-1-disabled")
    r = client.post("/admin/api/keys/%d/toggle" % meta["id"], headers=admin_headers)
    assert r.status_code == 200 and r.json()["disabled"] == 1, r.text
    return raw


def _archived_key(client, admin_headers) -> str:
    raw, meta = gw.mint_key("npu-box-1-archived")
    r = client.delete("/admin/api/keys/%d" % meta["id"], headers=admin_headers)
    assert r.status_code == 200, r.text
    return raw


CALLER_CREDENTIALS = ("authorization", "x-api-key", "cookie", "cf-access-jwt-assertion")


class TestChatOnAVendorEngineBox:
    def test_the_engine_gets_the_device_key_not_the_callers_key(self, client, monkeypatch):
        engine = _box(monkeypatch, vendor_engine())
        _route_to(monkeypatch, ("", FLEET_ID))
        raw, _meta = gw.mint_key("npu-box-1-e2e-key")
        r = _chat(client, raw)
        assert r.status_code == 200, r.text
        assert [q.headers.get("authorization")
                for q in engine.to("/v1/chat/completions")] == [BEARER]

    def test_the_fleet_id_reaches_the_engine_as_the_device_model_id(self, client, monkeypatch):
        engine = _box(monkeypatch, vendor_engine())
        _route_to(monkeypatch, ("", FLEET_ID))
        raw, _meta = gw.mint_key("npu-box-1-e2e-model")
        r = _chat(client, raw)
        assert r.status_code == 200, r.text
        assert [json.loads(q.content)["model"]
                for q in engine.to("/v1/chat/completions")] == [DEVICE_ID]

    def test_caller_credentials_stop_at_the_gateway(self, client, monkeypatch):
        engine = _box(monkeypatch, vendor_engine())
        _route_to(monkeypatch, ("", FLEET_ID))
        raw, _meta = gw.mint_key("npu-box-1-e2e-creds")
        r = _chat(client, raw, headers={"x-api-key": raw,
                                        "cookie": "CF_Authorization=not-a-real-jwt",
                                        "cf-access-jwt-assertion": "not-a-real-jwt"})
        assert r.status_code == 200, r.text
        # Exactly one chat reached the engine, and the only credential on it
        # is the device key.
        assert [{k: v for k, v in q.headers.items() if k in CALLER_CREDENTIALS or raw in v}
                for q in engine.to("/v1/chat/completions")] == [SIGNED]

    def test_a_query_string_gateway_key_is_not_forwarded_to_the_engine(self, client, monkeypatch):
        engine = _box(monkeypatch, vendor_engine())
        _route_to(monkeypatch, ("", FLEET_ID))
        raw, _meta = gw.mint_key("npu-box-1-e2e-query")
        r = _chat(client, url="/v1/chat/completions?api_key=" + raw + "&api-version=2024-10-21")
        assert r.status_code == 200, r.text
        assert [dict(q.url.params) for q in engine.to("/v1/chat/completions")] == [
            {"api-version": "2024-10-21"}]

    def test_a_query_string_gateway_key_is_not_forwarded_to_a_peer(self, client, monkeypatch):
        sent = _peer_wire(monkeypatch)
        _route_to(monkeypatch, ("p1", FLEET_ID), peers=("p1",))
        raw, _meta = gw.mint_key("npu-box-1-e2e-query-peer")
        r = _chat(client, url="/v1/chat/completions?api_key=" + raw + "&api-version=2024-10-21")
        assert r.status_code == 200, r.text
        assert [dict(q.url.params) for q in _chats_to(sent, "p1")] == [
            {"api-version": "2024-10-21"}]

    @pytest.mark.parametrize("make_key", [
        pytest.param(_wrong_key, id="wrong-key"),
        pytest.param(_disabled_key, id="disabled-key"),
        pytest.param(_archived_key, id="archived-key"),
    ])
    def test_a_refused_key_gets_401_and_nothing_reaches_the_engine(
            self, client, monkeypatch, admin_headers, make_key):
        engine = _box(monkeypatch, vendor_engine())
        _route_to(monkeypatch, ("", FLEET_ID))
        r = _chat(client, make_key(client, admin_headers))
        assert (r.status_code, engine.wire()) == (401, []), r.text

    @pytest.mark.parametrize("headers", [
        pytest.param({}, id="no-credential"),
        pytest.param({"Authorization": "Bearer "}, id="empty-bearer"),
    ])
    def test_no_identity_gets_401_and_nothing_reaches_the_engine(
            self, client, monkeypatch, headers):
        engine = _box(monkeypatch, vendor_engine())
        _route_to(monkeypatch, ("", FLEET_ID))
        r = _chat(client, headers=headers)
        assert (r.status_code, engine.wire()) == (401, []), r.text

    @pytest.mark.parametrize("refuse, stream", [
        pytest.param(401, False, id="401"),
        pytest.param(403, False, id="403"),
        pytest.param(401, True, id="401-streamed"),
    ])
    def test_the_engine_refusing_the_device_key_is_a_502_about_this_box(
            self, client, monkeypatch, refuse, stream):
        _box(monkeypatch, vendor_engine(key=ROTATED_KEY, refuse=refuse))
        _route_to(monkeypatch, ("", FLEET_ID))
        raw, _meta = gw.mint_key("npu-box-1-e2e-refused")
        r = _chat(client, raw, stream=stream)
        assert (r.status_code, "credential" in r.json()["error"]["message"],
                DEVICE_KEY in r.text) == (502, True, False), r.text

    def test_the_engine_refusing_the_device_key_is_metered_as_a_502(self, client, monkeypatch):
        _box(monkeypatch, vendor_engine(key=ROTATED_KEY))
        _route_to(monkeypatch, ("", FLEET_ID))
        raw, meta = gw.mint_key("npu-box-1-e2e-refused-metered")
        _chat(client, raw)
        rows = gw.db_query("SELECT status FROM usage WHERE key_id=?", (meta["id"],))
        assert [row["status"] for row in rows] == [502]

    def test_a_peer_gets_the_hubs_key_for_it_never_the_device_key(self, client, monkeypatch):
        _box(monkeypatch, vendor_engine())
        sent = _peer_wire(monkeypatch)
        _route_to(monkeypatch, ("p1", FLEET_ID), peers=("p1",))
        raw, _meta = gw.mint_key("npu-box-1-e2e-peer")
        r = _chat(client, raw)
        assert r.status_code == 200, r.text
        assert [(q.headers.get("authorization"), any(DEVICE_KEY in v for v in q.headers.values()))
                for q in _chats_to(sent, "p1")] == [("Bearer k", False)]


# ---------------------------------------------------------------------------
# What a box holding a key lets a caller ask its engine for -- and what a box
# without one still does exactly as before
# ---------------------------------------------------------------------------

class TestSignedRoutes:
    @pytest.mark.parametrize("method, path", [
        pytest.param("DELETE", "/v1/models/x", id="model-deletion"),
        pytest.param("PUT", "/v1/settings", id="settings"),
        pytest.param("POST", "/v1/system/shutdown", id="power"),
        pytest.param("POST", "/v1/batches", id="batch-spooling"),
    ])
    def test_a_keyed_box_answers_404_for_anything_else_and_asks_its_engine_nothing(
            self, client, monkeypatch, method, path):
        engine = _box(monkeypatch, vendor_engine(extra=VENDOR_EXTRAS))
        _route_to(monkeypatch, ("", FLEET_ID))
        raw, _meta = gw.mint_key("signed-routes-refused")
        r = _ask(client, method, path, raw, body={})
        assert (r.status_code, engine.wire()) == (404, []), r.text

    def test_a_keyed_box_meters_a_refused_route_as_a_404(self, client, monkeypatch):
        _box(monkeypatch, vendor_engine(extra=VENDOR_EXTRAS))
        raw, meta = gw.mint_key("signed-routes-metered")
        _ask(client, "POST", "/v1/system/shutdown", raw, body={})
        rows = gw.db_query("SELECT endpoint, status FROM usage WHERE key_id=?", (meta["id"],))
        assert [(row["endpoint"], row["status"]) for row in rows] == [("/v1/system/shutdown", 404)]

    @pytest.mark.parametrize("method, path, body", [
        pytest.param("POST", "/v1/chat/completions",
                     {"model": FLEET_ID, "messages": [{"role": "user", "content": "hi"}]},
                     id="chat"),
        pytest.param("POST", "/v1/completions", {"model": FLEET_ID, "prompt": "hi"},
                     id="completions"),
        pytest.param("POST", "/v1/embeddings", {"model": FLEET_ID, "input": "hi"},
                     id="embeddings"),
        pytest.param("GET", "/v1/models/x", None, id="one-model"),
    ])
    def test_a_keyed_box_still_forwards_the_openai_calls(self, client, monkeypatch, method,
                                                         path, body):
        engine = _box(monkeypatch, vendor_engine())
        _route_to(monkeypatch, ("", FLEET_ID))
        raw, _meta = gw.mint_key("signed-routes-allowed")
        _ask(client, method, path, raw, body=body)
        assert engine.wire().count(method + " " + path) == 1

    def test_a_keyless_box_forwards_any_v1_route_as_before(self, client, monkeypatch):
        engine = _keyless_box(monkeypatch, Engine({"/v1/system/shutdown": (200, {"status": "ok"})}))
        raw, _meta = gw.mint_key("keyless-routes")
        r = _ask(client, "POST", "/v1/system/shutdown", raw, body={})
        assert (r.status_code, engine.wire()) == (200, ["POST /v1/system/shutdown"]), r.text


class TestSignedHeaders:
    @pytest.mark.parametrize("make_box, expected", [
        pytest.param(lambda mp: _box(mp, vendor_engine()),
                     {**SENT_HEADERS, **dict.fromkeys(ROUTING_HINTS)},
                     id="keyed-box-passes-on-only-type-accept-and-agent"),
        pytest.param(lambda mp: _keyless_box(
                         mp, Engine({"/v1/chat/completions": (200, CHAT_OK_BODY)})),
                     SENT_HEADERS, id="keyless-box-passes-them-all-on-as-before"),
    ])
    def test_the_caller_headers_that_reach_the_engine(self, client, monkeypatch, make_box,
                                                       expected):
        engine = make_box(monkeypatch)
        _route_to(monkeypatch, ("", FLEET_ID))
        raw, _meta = gw.mint_key("signed-headers")
        r = _chat(client, raw, headers=SENT_HEADERS)
        assert r.status_code == 200, r.text
        assert [{h: q.headers.get(h) for h in SENT_HEADERS}
                for q in engine.to("/v1/chat/completions")] == [expected]

    def test_a_peer_hop_from_a_keyed_box_keeps_the_callers_headers(self, client, monkeypatch):
        # The filter is for this box's own engine. A peer reads some of what
        # the caller sent -- X-Fleet-No-Fallback opts out of substitution --
        # so a hop to a peer must not lose it.
        _box(monkeypatch, vendor_engine())
        sent = _peer_wire(monkeypatch)
        _route_to(monkeypatch, ("p1", FLEET_ID), peers=("p1",))
        raw, _meta = gw.mint_key("keyed-peer-headers")
        r = _chat(client, raw, headers={"x-request-id": "req-1"})
        assert r.status_code == 200, r.text
        (chat,) = [q for q in sent if q.url.path == "/v1/chat/completions"]
        assert (chat.headers.get("x-fleet-no-fallback"), chat.headers.get("x-request-id")) \
            == ("1", "req-1")


class TestKeylessBox:
    @pytest.mark.parametrize("status", [401, 403], ids=["401", "403"])
    def test_its_engines_refusal_passes_through_with_nothing_logged_about_a_credential(
            self, client, monkeypatch, caplog, status):
        _keyless_box(monkeypatch, Engine({"/v1/chat/completions": (
            status, {"error": {"message": "the engine says no"}})}))
        _route_to(monkeypatch, ("", FLEET_ID))
        raw, _meta = gw.mint_key("keyless-refusal")
        caplog.set_level(logging.ERROR, logger=gw.log.name)
        r = _chat(client, raw)
        about_a_credential = [rec.getMessage() for rec in caplog.records
                              if "credential" in rec.getMessage()]
        assert (r.status_code, about_a_credential) == (status, []), r.text


# ---------------------------------------------------------------------------
# forward_params(): the query string for the next hop
# ---------------------------------------------------------------------------

class TestForwardParams:
    @pytest.mark.parametrize("query, expected", [
        pytest.param(b"api_key=sk-ffa-test&api-version=2024-10-21",
                     {"api-version": "2024-10-21"}, id="gateway-key-dropped-the-rest-kept"),
        pytest.param(b"api_key=sk-ffa-test", {}, id="only-the-gateway-key-leaves-nothing"),
        pytest.param(b"api-version=2024-10-21", {"api-version": "2024-10-21"},
                     id="no-gateway-key-passes-unchanged"),
        pytest.param(b"", {}, id="empty-stays-empty"),
    ])
    def test_drops_only_the_gateway_key(self, query, expected):
        request = Request({"type": "http", "method": "POST", "path": "/v1/chat/completions",
                           "headers": [], "query_string": query})
        assert gw.forward_params(request) == expected


class TestNativeRetry:
    def test_the_num_ctx_retry_drops_a_query_string_gateway_key_like_the_first_try(
            self, client, monkeypatch):
        # The shape test_ctx_auto_retry.py's TestNativeOllamaCtxRetry drives:
        # an out-of-memory 500 at the asked num_ctx, then a 200 at half.
        engine = _keyless_box(monkeypatch, Engine({"/api/chat": [
            (500, OOM_BODY), (200, NATIVE_OK_BODY)]}), upstream_models=True)
        raw, _meta = gw.mint_key("native-retry-query")
        r = client.post("/api/chat?api_key=" + raw, json={
            "model": "llama3:8b", "stream": False, "options": {"num_ctx": 65536},
            "messages": [{"role": "user", "content": "hi"}]})
        assert (r.status_code, [dict(q.url.params) for q in engine.to("/api/chat")]) == (
            200, [{}, {}]), r.text


# ---------------------------------------------------------------------------
# '.' and '..' segments: refused before anything reaches the engine
# ---------------------------------------------------------------------------

class TestDotSegments:
    @pytest.mark.parametrize("path, expected", [
        pytest.param("../api", True, id="leading-dotdot"),
        pytest.param("chat/../x", True, id="inner-dotdot"),
        pytest.param("a/./b", True, id="inner-dot"),
        pytest.param("a\\..\\b", True, id="backslash-separated-dotdot"),
        pytest.param("chat/..", True, id="trailing-dotdot"),
        pytest.param("chat/completions", False, id="ordinary-path"),
        pytest.param("v1.5/x", False, id="dot-inside-a-segment"),
        pytest.param("..hidden", False, id="dotdot-prefixed-name"),
        pytest.param("a/.../b", False, id="three-dots-is-a-name"),
    ])
    def test_has_dot_segment(self, path, expected):
        assert gw.has_dot_segment(path) is expected

    @pytest.mark.parametrize("method, url", [
        # Percent-encoded so the test client does not resolve them itself;
        # Starlette decodes them, and httpx would resolve them on the way out.
        pytest.param("POST", "/v1/%2e%2e/api/v1/account/reset_password",
                     id="v1-proxy-stepping-into-the-device-account-api"),
        pytest.param("DELETE", "/api/chat/%2e%2e/delete",
                     id="native-proxy-stepping-past-its-allow-list-to-model-delete"),
    ])
    def test_a_dot_segment_path_is_a_400_and_nothing_reaches_the_engine(
            self, client, monkeypatch, method, url):
        engine = _box(monkeypatch, ollama())
        _route_to(monkeypatch, ("", "llama3:8b"))
        raw, _meta = gw.mint_key("dot-segment")
        r = client.request(method, url, headers={"Authorization": "Bearer " + raw},
                           json={"model": "llama3:8b"})
        assert (r.status_code, engine.wire()) == (400, []), r.text


class TestUnsafeProxyPath:
    @pytest.mark.parametrize("path, expected", [
        pytest.param("chat/completions", False, id="ordinary-path"),
        pytest.param("models/zai-org/GLM-4.7-Flash", False, id="dots-inside-segments"),
        pytest.param("../api", True, id="dot-segment"),
        pytest.param("%2e%2e/api/v1/sys/reboot", True, id="percent-left-after-one-decode"),
        pytest.param("chat/completions;x=1", True, id="semicolon"),
        pytest.param("models\\x", True, id="backslash"),
    ])
    def test_unsafe_proxy_path(self, path, expected):
        assert gw.unsafe_proxy_path(path) is expected


class TestTraversalAsAServerDecodes:
    """The refusals above go through Starlette's TestClient, which decodes the
    path twice -- so a double-encoded '..' reached the route already resolved,
    and the case a real server lets through was never put to the test.
    uvicorn decodes once, and so does httpx.ASGITransport, driven here against
    the same app (the session TestClient has already run its lifespan)."""
    pytestmark = pytest.mark.asyncio

    @pytest.mark.parametrize("url", [
        pytest.param("/v1/%252e%252e/api/v1/sys/reboot",
                     id="v1-double-encoded-dotdot-into-the-device-api"),
        pytest.param("/v1/chat/completions;x=1", id="v1-path-parameter-on-the-chat-route"),
        pytest.param("/v1/..%5capi/v1/connect", id="v1-backslash-separated-dotdot"),
        pytest.param("/api/chat/%252e%252e/v1/account/reset_password",
                     id="native-double-encoded-dotdot-past-its-allow-list"),
    ])
    async def test_refused_with_400_before_anything_reaches_the_engine(self, monkeypatch, url):
        engine = _box(monkeypatch, vendor_engine())
        _route_to(monkeypatch, ("", FLEET_ID))
        raw, _meta = gw.mint_key("one-decode")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app),
                                     base_url="http://testserver") as c:
            r = await c.post(url, headers={"Authorization": "Bearer " + raw,
                                           "X-Fleet-No-Fallback": "1"},
                             json={"model": FLEET_ID,
                                   "messages": [{"role": "user", "content": "hi"}]})
        assert (r.status_code, engine.wire()) == (400, []), r.text


# ---------------------------------------------------------------------------
# Residency: what this box's engine holds right now
# ---------------------------------------------------------------------------

class TestResidency:
    pytestmark = pytest.mark.asyncio

    @pytest.mark.parametrize("make, upstream_models, held, openai_only", [
        pytest.param(vendor_engine, True, {DEVICE_ID}, True,
                     id="npu-box-1-running-404-api-ps-401-holds-what-it-lists"),
        pytest.param(openai_server, True, {DEVICE_ID}, True,
                     id="bare-server-running-404-api-ps-404-holds-what-it-lists"),
        pytest.param(ollama, True, {"llama3:8b"}, False,
                     id="ollama-api-ps-200-holds-what-is-loaded-not-what-is-pulled"),
        pytest.param(llama_swap, True, {"qwen3.5-4b"}, False,
                     id="llama-swap-running-200-unchanged"),
        pytest.param(lambda: vendor_engine(running=BROKEN), True, set(), False,
                     id="running-500-is-a-broken-engine"),
        pytest.param(lambda: openai_server(ps=BROKEN), True, set(), False,
                     id="api-ps-500-is-a-broken-engine"),
        pytest.param(lambda: vendor_engine(running=httpx.ConnectError("refused")), True, set(), False,
                     id="running-unreachable-is-a-broken-engine"),
        pytest.param(lambda: openai_server(ps=httpx.ConnectError("refused")), True, set(), False,
                     id="api-ps-unreachable-is-a-broken-engine"),
        pytest.param(vendor_engine, False, set(), False,
                     id="not-serving-the-engines-catalogue-holds-nothing"),
    ])
    async def test_what_the_engine_holds(self, monkeypatch, make, upstream_models, held,
                                         openai_only):
        _box(monkeypatch, make(), upstream_models=upstream_models)
        got = await gw.upstream_running_ids()
        assert (got, gw.upstream_openai_only()) == (held, openai_only)


class TestSwapRunning:
    """swap_running() feeds the Serving card, the fleet overview and the home
    page: (engine reachable, what it holds)."""
    pytestmark = pytest.mark.asyncio

    @pytest.mark.parametrize("make", [
        pytest.param(vendor_engine, id="npu-box-1-api-ps-401"),
        pytest.param(openai_server, id="bare-server-api-ps-404"),
    ])
    async def test_an_engine_that_holds_what_it_lists_reports_it_under_the_fleet_id(
            self, monkeypatch, make):
        _box(monkeypatch, make())
        assert await gw.swap_running() == (
            True, [{"id": FLEET_ID, "state": "ready", "n_ctx": 131072}])

    async def test_an_ollama_engine_reports_what_api_ps_lists_as_before(self, monkeypatch):
        _box(monkeypatch, ollama())
        assert await gw.swap_running() == (True, [{
            "id": "llama3:8b", "state": "ready", "n_ctx": 8192, "vram": 4920000000,
            "expires_at": "2026-10-14T12:00:00Z"}])


# ---------------------------------------------------------------------------
# The catalogue this box reports: names and context windows
# ---------------------------------------------------------------------------

class TestCatalogue:
    pytestmark = pytest.mark.asyncio

    async def test_serves_the_device_model_under_both_names_and_not_the_vendor_alias(
            self, monkeypatch):
        _box(monkeypatch, vendor_engine())
        assert await gw.served_model_ids() == {FLEET_ID, DEVICE_ID}

    async def test_answers_to_the_fleet_id_for_the_device_id(self, monkeypatch):
        _box(monkeypatch, vendor_engine())
        assert await gw.upstream_alias_pairs() == {FLEET_ID: DEVICE_ID}

    async def test_reports_the_declared_window_under_both_names(self, monkeypatch):
        _box(monkeypatch, vendor_engine())
        ctx = await gw.served_model_ctx()
        assert {m: ctx.get(m) for m in (FLEET_ID, DEVICE_ID)} == {FLEET_ID: 131072,
                                                                  DEVICE_ID: 131072}

    @pytest.mark.parametrize("fields, expected", [
        pytest.param({"context_window": 131072}, 131072, id="context_window"),
        pytest.param({"context_length": 65536}, 65536, id="context_length"),
        pytest.param({"max_model_len": 32768}, 32768, id="max_model_len"),
        pytest.param({"max_context_length": 16384}, 16384, id="max_context_length"),
        pytest.param({"context_length": 4096, "context_window": 131072}, 131072,
                     id="context_window-wins-whatever-the-json-order"),
        pytest.param({"context_window": 0, "max_model_len": 32768}, 32768,
                     id="zero-is-not-a-window-the-next-field-is"),
        pytest.param({"created": 1700000000, "owned_by": "library"}, None,
                     id="ollama-shaped-entry-declares-nothing"),
    ])
    async def test_declared_window_is_the_first_positive_field(self, monkeypatch, fields,
                                                               expected):
        listing = {"object": "list",
                   "data": [{"id": "vendor/model-x", "object": "model", **fields}]}
        _box(monkeypatch, vendor_engine(models=listing))
        assert (await gw.served_model_ctx()).get("vendor/model-x") == expected


# ---------------------------------------------------------------------------
# engine_info(): naming the engine this box fronts
# ---------------------------------------------------------------------------

def _probes_through(monkeypatch, engine: Engine) -> Engine:
    """engine_info() is synchronous and probes with the module-level
    httpx.get; send those probes through the same fake engine, headers and
    all."""
    monkeypatch.setattr(gw, "UPSTREAM_MODELS", True)
    monkeypatch.setattr(gw, "UPSTREAM_KEY", DEVICE_KEY)

    def fake_get(url, *, headers=None, timeout=None, **kw):
        return engine(httpx.Request("GET", url, headers=headers))

    monkeypatch.setattr(gw.httpx, "get", fake_get)
    return engine


class TestEngineInfo:
    @pytest.mark.parametrize("make", [
        pytest.param(vendor_engine, id="npu-box-1-api-ps-401"),
        pytest.param(openai_server, id="bare-server-api-ps-404"),
    ])
    def test_an_engine_that_only_answers_v1_models_is_an_openai_engine_and_up(
            self, monkeypatch, make):
        _probes_through(monkeypatch, make())
        info = gw.engine_info()
        assert (info["kind"], info["up"]) == ("openai", True)

    def test_only_the_v1_models_probe_carries_the_device_key(self, monkeypatch):
        engine = _probes_through(monkeypatch, vendor_engine())
        gw.engine_info()
        auth = {q.url.path: q.headers.get("authorization") for q in engine.requests}
        assert {p: auth.get(p, "not probed") for p in ("/running", "/api/ps", "/v1/models")} == {
            "/running": None, "/api/ps": None, "/v1/models": BEARER}


# ---------------------------------------------------------------------------
# Touch: the hub's keep-warm lever
# ---------------------------------------------------------------------------

def _beyond_probes(engine: Engine) -> list[str]:
    return [q.method + " " + q.url.path for q in engine.requests
            if (q.method, q.url.path) not in PROBES]


class TestTouch:
    @pytest.mark.parametrize("make", [
        pytest.param(vendor_engine, id="npu-box-1-api-ps-401"),
        pytest.param(openai_server, id="bare-server-api-ps-404"),
    ])
    def test_touch_on_an_engine_that_holds_what_it_lists_is_a_200_that_loads_nothing(
            self, client, monkeypatch, admin_headers, make):
        engine = _box(monkeypatch, make())
        r = client.post("/admin/api/models/touch", headers=admin_headers,
                        json={"model": FLEET_ID})
        assert (r.status_code, _beyond_probes(engine)) == (200, []), r.text

    def test_touch_on_an_ollama_box_is_still_a_501(self, client, monkeypatch, admin_headers):
        engine = _box(monkeypatch, ollama())
        r = client.post("/admin/api/models/touch", headers=admin_headers,
                        json={"model": "llama3:8b"})
        assert (r.status_code, _beyond_probes(engine)) == (501, []), r.text


# ---------------------------------------------------------------------------
# read_upstream_key(): the key file
# ---------------------------------------------------------------------------

def _key_file(content: bytes):
    def make(tmp_path) -> str:
        path = tmp_path / "upstream.key"
        path.write_bytes(content)
        return str(path)
    return make


class TestReadUpstreamKey:
    @pytest.mark.parametrize("make, expected, logged", [
        pytest.param(lambda tmp: "", "", False, id="unset-is-no-key-and-no-log"),
        pytest.param(_key_file(b"  dev-key-123 \n"), "dev-key-123", False,
                     id="content-is-stripped"),
        pytest.param(lambda tmp: str(tmp / "missing.key"), "", True, id="missing-file"),
        pytest.param(lambda tmp: str(tmp), "", True, id="a-directory-is-unreadable"),
        pytest.param(_key_file(b""), "", True, id="empty-file"),
        pytest.param(_key_file(b" \n\t"), "", True, id="whitespace-only-file"),
        pytest.param(_key_file(b"dev-key-123\xff\xfe"), "", True,
                     id="not-utf8-text-is-unreadable"),
        # Windows PowerShell 5.1's Set-Content -Encoding UTF8 and Notepad
        # both write one; strip() does not remove it.
        pytest.param(_key_file(b"\xef\xbb\xbfdev-key-123\r\n"), "dev-key-123", False,
                     id="a-byte-order-mark-is-not-part-of-the-key"),
    ])
    def test_reads_the_key_and_reports_a_bad_file_by_path_never_by_content(
            self, tmp_path, caplog, make, expected, logged):
        path = make(tmp_path)
        caplog.set_level(logging.ERROR, logger=gw.log.name)
        key = gw.read_upstream_key(path)
        errors = [r.getMessage() for r in caplog.records
                  if r.name == gw.log.name and r.levelno >= logging.ERROR]
        assert {"key": key, "logged": bool(errors),
                "names_the_path": all(path in m for m in errors),
                "shows_the_content": any("dev-key" in m for m in errors)} == {
            "key": expected, "logged": logged,
            "names_the_path": True, "shows_the_content": False}
