""""Not in use" has to reach the box it is about.

The Configurations tab takes a box out of use by setting `routed: false` on
the hub. That stopped the hub SENDING it work, and told the box nothing: it
went on warming whatever was resident every ten minutes (so a model loaded
once never left the card) and walking its owner's LM Studio library every
five (restarting llama-swap over a file they had just downloaded). On a
laptop whose owner runs their own models beside the fleet's, that is the
fleet helping itself to a machine it was told to leave alone.

Now the hub says so on the telemetry poll it already makes, the box
remembers, and the two loops stand down. These tests pin: who may say it,
what is remembered, what stops, and what an operator can still do by hand.
"""
from __future__ import annotations

import pytest

import app as gw

ADMIN = {"Authorization": "Bearer test"}


def says(client, word, headers=ADMIN):
    return client.get("/admin/api/status",
                      headers={**headers, gw.FLEET_ROUTED_HEADER: word})


# ---------------------------------------------------------------------------
@pytest.fixture
def as_the_hub(monkeypatch):
    monkeypatch.setenv("LLMSTACK_FLEET_HUB", "1")


class TestWhatTheHubSends:
    def test_a_box_in_use_is_told_so(self, as_the_hub):
        h = gw.peer_poll_headers({"name": "a", "url": "http://x", "token": "t"})
        assert h == {"Authorization": "Bearer t", gw.FLEET_ROUTED_HEADER: "1"}

    def test_a_box_taken_out_of_use_is_told_so(self, as_the_hub):
        h = gw.peer_poll_headers({"name": "a", "url": "http://x", "token": "t",
                                  "routed": False})
        assert h[gw.FLEET_ROUTED_HEADER] == "0"

    def test_a_peer_entry_older_than_the_flag_is_in_use(self, as_the_hub):
        """Missing means routed, exactly as peer_routed() reads it."""
        assert gw.peer_poll_headers({"token": ""})[gw.FLEET_ROUTED_HEADER] == "1"


class TestOnlyTheHubSaysIt:
    """Every gateway polls whatever is in its own peers.json. On 2026-08-29
    three boxes carried a peer list and one of them was the hub. A box that
    is not the hub has no `routed` flag worth repeating: its stale entry
    reads "in use", and one poll from it would have put a laptop back to
    work that the hub had taken out of use."""

    def test_a_box_that_is_not_the_hub_sends_the_token_and_nothing_else(
            self, monkeypatch):
        monkeypatch.delenv("LLMSTACK_FLEET_HUB", raising=False)
        monkeypatch.setattr(gw, "load_specs", lambda: {gw.HOST_NAME: {"klass": "gpu"}})
        for peer in ({"token": "t"}, {"token": "t", "routed": False}):
            assert gw.peer_poll_headers(peer) == {"Authorization": "Bearer t"}

    def test_a_box_the_sheet_has_never_heard_of_is_not_the_hub(self, monkeypatch):
        monkeypatch.delenv("LLMSTACK_FLEET_HUB", raising=False)
        monkeypatch.setattr(gw, "load_specs", lambda: {})
        assert gw.fleet_hub() is False

    def test_the_spec_sheet_names_the_hub(self, monkeypatch):
        monkeypatch.delenv("LLMSTACK_FLEET_HUB", raising=False)
        monkeypatch.setattr(gw, "load_specs", lambda: {gw.HOST_NAME: {"role": "hub"}})
        assert gw.fleet_hub() is True
        assert gw.peer_poll_headers({"token": "t", "routed": False})[
            gw.FLEET_ROUTED_HEADER] == "0"

    def test_the_env_can_say_so_either_way(self, monkeypatch):
        monkeypatch.setattr(gw, "load_specs", lambda: {gw.HOST_NAME: {"role": "hub"}})
        monkeypatch.setenv("LLMSTACK_FLEET_HUB", "0")
        assert gw.fleet_hub() is False
        monkeypatch.setattr(gw, "load_specs", lambda: {})
        monkeypatch.setenv("LLMSTACK_FLEET_HUB", "1")
        assert gw.fleet_hub() is True

    def test_the_real_sheet_has_exactly_one(self):
        hubs = [h for h, s in gw.DEFAULT_SPECS.items() if s.get("role") == "hub"]
        assert len(hubs) == 1

    def test_both_telemetry_polls_carry_it(self):
        """The overview poll and the summary poll are the two the hub makes
        to every peer, in use or not. A third that forgot the header would
        not un-tell a box, but this is where a regression would start."""
        import inspect
        for fn in (gw._gather_fleet_overview, gw.fleet_summary):
            assert "peer_poll_headers(p)" in inspect.getsource(fn), fn.__name__


# ---------------------------------------------------------------------------
class TestWhatTheBoxHears:
    def test_a_box_that_was_never_told_anything_is_in_use(self, client):
        assert gw.box_standby() is False
        body = client.get("/admin/api/status", headers=ADMIN).json()
        assert body["standby"] == {"standby": False, "since": None,
                                   "heard_s_ago": None}

    def test_the_hub_takes_it_out_of_use(self, client):
        assert says(client, "0").status_code == 200
        assert gw.box_standby() is True
        report = client.get("/admin/api/status", headers=ADMIN).json()["standby"]
        assert report["standby"] is True
        assert report["since"]
        assert report["heard_s_ago"] is not None

    def test_and_puts_it_back(self, client):
        says(client, "0")
        says(client, "1")
        assert gw.box_standby() is False

    def test_the_dashboards_own_call_changes_nothing(self, client):
        """A browser asking for the same route sends no such header."""
        says(client, "0")
        client.get("/admin/api/status", headers=ADMIN)
        assert gw.box_standby() is True

    def test_only_an_administrator_can_say_it(self, client):
        r = says(client, "0", headers={"Authorization": "Bearer not-the-token"})
        assert r.status_code in (401, 403, 503)
        assert gw.box_standby() is False

    @pytest.mark.parametrize("word", ["", "yes", "2", "off", "standby", " "])
    def test_a_word_it_does_not_know_is_not_an_instruction(self, client, word):
        says(client, word)
        assert gw.box_standby() is False
        says(client, "0")
        says(client, word)
        assert gw.box_standby() is True

    def test_it_survives_a_restart(self, client):
        """Remembered in the database, so a gateway restarted by a deploy
        does not come back believing it is in use until the hub's next poll
        -- the window in which the loops would have run."""
        says(client, "0")
        gw._standby.update(loaded=False, standby=False, since="", heard=0.0)
        assert gw.box_standby() is True
        assert gw.standby_report()["heard_s_ago"] is None   # not since THIS start

    def test_it_is_written_when_it_changes_and_not_otherwise(self, client, monkeypatch):
        """The hub says it every few seconds."""
        says(client, "0")
        writes = []
        real = gw.db_exec

        def recording(sql, args=()):
            # Only this key: the app's own background loops write to the
            # same table while the test client is up.
            args = tuple(args)
            if gw.STANDBY_KEY in args:
                writes.append(sql)
            return real(sql, args)
        monkeypatch.setattr(gw, "db_exec", recording)
        for _ in range(5):
            says(client, "0")
        assert writes == []
        says(client, "1")
        assert len(writes) == 1

    def test_a_database_that_cannot_be_read_is_not_a_reason_to_stand_down(
            self, monkeypatch):
        gw._standby.update(loaded=False, standby=False, since="", heard=0.0)

        def broken(*_a, **_k):
            raise RuntimeError("no such table: settings")
        monkeypatch.setattr(gw, "db_query", broken)
        assert gw.box_standby() is False
        assert gw._standby["loaded"] is False        # asked again next time


# ---------------------------------------------------------------------------
class TestWhatStopsOnABoxThatIsNotInUse:
    @pytest.fixture
    def resident(self, monkeypatch):
        """One model in memory on an idle box, and a record of what was
        warmed."""
        calls = []

        async def ready():
            return {"m1"}

        async def warmed(mid):
            calls.append(mid)
        monkeypatch.setattr(gw, "_ready_model_ids", ready)
        monkeypatch.setattr(gw, "_warm_on_load_run", warmed)
        monkeypatch.setattr(gw, "_box_idle_for", lambda: 10_000.0)
        for name in ("_warm_on_load_ready", "_warm_on_load_pending"):
            monkeypatch.setattr(gw, name, set())
        return calls

    @pytest.mark.asyncio
    async def test_nothing_is_warmed_and_the_engine_is_not_even_asked(
            self, client, monkeypatch):
        gw.note_fleet_routed("0")

        async def asked():
            raise AssertionError("asked llama-swap what is running")

        async def warmed(_mid):
            raise AssertionError("sent a warm-up")
        monkeypatch.setattr(gw, "_ready_model_ids", asked)
        monkeypatch.setattr(gw, "_warm_on_load_run", warmed)
        assert await gw._warm_on_load_tick() == []

    @pytest.mark.asyncio
    async def test_a_box_in_use_still_warms(self, client, resident):
        assert await gw._warm_on_load_tick() == ["m1"]
        assert resident == ["m1"]

    @pytest.mark.asyncio
    async def test_put_back_into_use_it_warms_what_is_resident(self, client, resident):
        """Standing by forgets nothing, so the first tick back sees the
        resident model as new and warms it."""
        gw.note_fleet_routed("0")
        assert await gw._warm_on_load_tick() == []
        gw.note_fleet_routed("1")
        assert await gw._warm_on_load_tick() == ["m1"]

    @pytest.fixture
    def syncs(self, monkeypatch):
        ran = []
        monkeypatch.setattr(gw, "lmstudio_sync", lambda *a, **k: ran.append("sync") or {})
        monkeypatch.setattr(gw, "get_lmstudio_settings", lambda: {
            **gw.DEFAULT_LMSTUDIO_SETTINGS, "enabled": True, "interval_secs": 300})
        return ran

    @pytest.mark.asyncio
    async def test_the_lm_studio_library_is_not_walked(self, client, syncs):
        gw.note_fleet_routed("0")
        assert await gw._lmstudio_sync_turn() == 300     # it still keeps time
        assert syncs == []

    @pytest.mark.asyncio
    async def test_a_box_in_use_still_syncs(self, client, syncs):
        assert await gw._lmstudio_sync_turn() == 300
        assert syncs == ["sync"]

    def test_the_loops_take_their_turns_through_these(self):
        """The turn is what is tested; this is what says the loop uses it."""
        import inspect
        assert "await _lmstudio_sync_turn()" in inspect.getsource(gw._lmstudio_sync_loop)
        assert "await _public_overview_turn()" in inspect.getsource(
            gw._public_overview_warm_loop)


# ---------------------------------------------------------------------------
class TestTheHubsPollIsCheapOnABoxThatIsNotInUse:
    """Each answer to /admin/api/status costs a CPU sample, an nvidia-smi and
    a walk of the process table, and the hub asks every ten seconds. On a box
    doing nothing for the fleet, that was the fleet's whole footprint."""

    @pytest.fixture
    def readings(self, monkeypatch):
        taken = []

        def host_status():
            taken.append(1)
            return {"cpu_percent": float(len(taken))}
        monkeypatch.setattr(gw, "host_status", host_status)
        monkeypatch.setattr(gw, "service_states", lambda: {})
        return taken

    def test_the_hub_is_handed_the_reading_it_was_given_a_moment_ago(
            self, client, readings):
        first = says(client, "0").json()
        again = says(client, "0").json()
        assert len(readings) == 1
        assert again["host"] == first["host"]
        assert again["cached_s"] >= 0
        assert again["standby"]["standby"] is True

    def test_the_dashboard_always_reads_live(self, client, readings):
        """No header, no cache: the owner looking at their own box."""
        says(client, "0")
        client.get("/admin/api/status", headers=ADMIN)
        client.get("/admin/api/status", headers=ADMIN)
        assert len(readings) == 3
        assert "cached_s" not in client.get("/admin/api/status", headers=ADMIN).json()

    def test_a_box_in_use_is_read_every_time(self, client, readings):
        says(client, "1")
        says(client, "1")
        assert len(readings) == 2

    def test_a_reading_older_than_the_limit_is_taken_again(self, client, readings):
        says(client, "0")
        gw._standby_status["t"] -= gw.STANDBY_POLL_TTL_S + 1
        says(client, "0")
        assert len(readings) == 2

    def test_put_back_into_use_the_next_answer_is_live(self, client, readings):
        says(client, "0")
        body = says(client, "1").json()
        assert len(readings) == 2
        assert "cached_s" not in body

    @pytest.mark.asyncio
    async def test_the_overview_loop_does_not_sample_a_box_that_is_out_of_use(
            self, client, monkeypatch):
        built = []

        async def payload():
            built.append(1)
            return {"hosts": []}
        monkeypatch.setattr(gw, "public_overview_payload", payload)
        gw.note_fleet_routed("0")
        assert await gw._public_overview_turn() is False
        assert built == []
        gw.note_fleet_routed("1")
        assert await gw._public_overview_turn() is True
        assert built == [1]


# ---------------------------------------------------------------------------
class TestWhatTheOwnerCanStillDoByHand:
    """"Leave it alone" binds the fleet, not the person whose machine it is."""

    def test_a_sync_asked_for_directly_still_runs(self, client, monkeypatch):
        says(client, "0")
        monkeypatch.setattr(gw, "lmstudio_plan", lambda: {
            "settings": dict(gw.DEFAULT_LMSTUDIO_SETTINGS), "available": False,
            "actions": []})
        plan = gw.lmstudio_sync()
        assert plan["available"] is False        # it ran, and found no LM Studio

    def test_standby_is_not_the_killswitch(self, client):
        """The box does not take itself out of anybody's routing table by
        hearing this. That stays the hub's decision, in the hub's file."""
        says(client, "0")
        assert gw.peer_routed({"name": gw.HOST_NAME}) is True
