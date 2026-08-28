"""The hub's peer list, and the addresses in it.

This is the finding `fleetctl peers` was written for: on 2026-08-27 the hub
was addressing seven of its thirteen peers by something that was not a
tailnet address -- six by bare name (`http://apu-box-1:8080`, `http://gpu-desktop-1:8080`,
`http://mini-pc-1:8080`, ...) and `gpu-laptop-1` by a DHCP lease,
`http://192.168.1.144:8080`.

They all worked, which is the problem. The hub shares a LAN with those boxes,
so the names resolved and the lease was current. The 2026-08-26 power cut had
already shown what happens when that stops being true -- new leases on the way
back up, every name-pinned host in `deploy-gateway.sh` resolving to a machine
that was no longer there, three boxes reported offline while serving traffic.
The deploy script was fixed for it. The peer list, which nothing regenerates,
was not.

So the tests below are mostly about the two ways a fix could be worse than the
bug: dropping a peer (the hub's PUT replaces the whole list) and moving an
admin token around (it should never have to).
"""
from __future__ import annotations

import pytest

from fleetctl import cli, hostfile, peers

# The live list as it stood, names and all -- so the classification is tested
# against what was actually there rather than against invented examples.
LIVE = [
    {"name": "apu-box-1", "url": "http://apu-box-1:8080",
     "api_url": "https://max.example.com/v1", "has_token": True},
    {"name": "gpu-laptop-1", "url": "http://192.168.1.144:8080",
     "api_url": "https://gpu-laptop-1.example.com/v1", "has_token": True},
    {"name": "server-1", "url": "http://server-1:8080",
     "api_url": "http://server-1:8080/v1", "has_token": True},
    {"name": "cpu-box-1", "url": "http://cpu-box-1:8080",
     "api_url": "http://cpu-box-1:8080/v1", "has_token": True},
    {"name": "mac-desktop-1", "url": "http://mac-desktop:8080",
     "api_url": "https://mac-desktop-1.example.com/v1", "has_token": True},
    {"name": "mini-pc-1", "url": "http://mini-pc-1:8080",
     "api_url": "http://mini-pc-1:8080/v1", "has_token": True},
    {"name": "mac-laptop-1", "url": "http://100.64.0.10:8080",
     "api_url": "http://100.64.0.10:8080/v1", "has_token": True},
    {"name": "gpu-desktop-1", "url": "http://gpu-desktop-1:8080",
     "api_url": "http://gpu-desktop-1:8080/v1", "has_token": True},
]


# --------------------------------------------------------------------------
class TestHowAnAddressWillAge:
    """A bare name and a lease are not the same failure, but both are one."""

    @pytest.mark.parametrize("url,kind", [
        ("http://100.64.0.1:8080", "tailnet"),
        ("http://100.64.0.254:8080", "tailnet"),
        ("http://100.64.0.10:8080", "tailnet"),
        # Just outside the CGNAT block on both sides. 100.63 and 100.128 are
        # ordinary public space and would be somebody else's machine.
        ("http://100.63.255.255:8080", "public"),
        ("http://100.128.0.1:8080", "public"),
        ("http://192.168.1.144:8080", "lan"),
        ("http://10.0.0.5:8080", "lan"),
        ("http://172.16.4.1:8080", "lan"),
        ("http://172.32.4.1:8080", "public"),
        ("http://127.0.0.1:8080", "loopback"),
        ("http://169.254.1.1:8080", "link-local"),
        ("http://apu-box-1:8080", "name"),
        ("https://api.example.com/v1", "name"),
        ("http://[::1]:8080", "ipv6"),
    ])
    def test_kind(self, url, kind):
        assert peers.address_kind(url) == kind

    @pytest.mark.parametrize("url", [
        "http://100.64.0.10:8080/",
        "http://100.64.0.10:8080/admin/api",
        "http://user@100.64.0.10:8080",
        "100.64.0.10:8080",
    ])
    def test_the_shapes_the_hub_list_has_actually_held(self, url):
        assert peers.address_kind(url) == "tailnet"

    def test_the_seven_that_were_wrong(self):
        _, rows = peers.reconcile(LIVE, {}, hub="hub")
        stale = {r["name"] for r in peers.brittle(rows)}
        assert stale == {"apu-box-1", "gpu-laptop-1", "server-1", "cpu-box-1",
                         "mac-desktop-1", "mini-pc-1", "gpu-desktop-1"}
        # ...and the one that was already right is not swept up with them.
        assert "mac-laptop-1" not in stale


# --------------------------------------------------------------------------
class TestWhatTheRepoSays:
    """host.yml is the source; the hub's list is a copy that drifted."""

    def test_every_box_has_a_tailnet_address(self, repo):
        want, problems = peers.desired(repo)
        assert problems == []
        assert len(want) == 14

    def test_and_every_one_of_them_is_durable(self, repo):
        want, _ = peers.desired(repo)
        bad = {n: v["url"] for n, v in want.items()
               if peers.address_kind(v["url"]) != peers.DURABLE}
        assert bad == {}

    def test_no_two_boxes_advertise_the_same_api_url(self, repo):
        """server-1 and cpu-box-1 both claimed llm.example.com, and mini-pc-1
        claimed api.example.com -- which is the hub's. A client following
        mini-pc-1's advertised URL was sent to hub and served hub's models.
        """
        want, _ = peers.desired(repo)
        seen: dict[str, str] = {}
        clash = []
        for name, v in sorted(want.items()):
            if v["api_url"] in seen:
                clash.append(f"{name} and {seen[v['api_url']]}: {v['api_url']}")
            seen[v["api_url"]] = name
        assert clash == []

    @pytest.mark.parametrize("name", ["server-1", "cpu-box-1", "mini-pc-1", "mac-desktop-1"])
    def test_the_four_boxes_with_no_tunnel_advertise_their_tailnet(self, repo, name):
        """None of these four runs a cloudflared that serves the hostname it
        used to advertise -- server-1 and mac-desktop-1 have no cloudflared binary at
        all, and cpu-box-1's ingress serves example.org. mac-desktop-1's old
        hostname answered 530."""
        want, _ = peers.desired(repo)
        assert peers.address_kind(want[name]["api_url"]) == peers.DURABLE


# --------------------------------------------------------------------------
class TestReconcileCannotMakeItWorse:
    """`PUT /admin/api/peers` replaces the whole list. Every test here is a
    way that could go wrong."""

    def want(self, repo):
        return peers.desired(repo)[0]

    def test_no_peer_is_ever_dropped(self, repo):
        out, _ = peers.reconcile(LIVE, self.want(repo), hub="hub")
        assert [p["name"] for p in out] == [p["name"] for p in LIVE]

    def test_a_peer_the_repo_never_heard_of_is_left_exactly_as_it_was(self, repo):
        stranger = {"name": "borrowed", "url": "http://borrowed.example:8080",
                    "api_url": "http://borrowed.example:8080/v1"}
        out, rows = peers.reconcile(LIVE + [stranger], self.want(repo), hub="hub")
        kept = next(p for p in out if p["name"] == "borrowed")
        assert kept["url"] == stranger["url"]
        assert kept["api_url"] == stranger["api_url"]
        assert next(r for r in rows if r["name"] == "borrowed")["action"] == "foreign"

    def test_no_entry_carries_a_token(self, repo):
        """An empty token is how the hub is told to keep the stored secret.
        It is also how this never has to read one."""
        out, _ = peers.reconcile(LIVE, self.want(repo), hub="hub")
        assert {p["token"] for p in out} == {""}

    def test_the_hub_is_not_proposed_as_its_own_peer(self, repo):
        _, rows = peers.reconcile(LIVE, self.want(repo), hub="hub")
        assert "hub" not in {r["name"] for r in rows}

    def test_a_box_the_hub_has_never_been_told_about_is_reported_not_invented(
            self, repo):
        _, rows = peers.reconcile(LIVE, self.want(repo), hub="hub")
        new = {r["name"] for r in rows if r["action"] == "unregistered"}
        # Present in hosts/ but absent from LIVE, and none of them appear in
        # the list to write -- fleetctl has no admin token to give them.
        assert "gpu-laptop-2" in new and "apu-tablet-2" in new
        out, _ = peers.reconcile(LIVE, self.want(repo), hub="hub")
        assert new.isdisjoint({p["name"] for p in out})

    def test_it_retargets_every_stale_address(self, repo):
        want = self.want(repo)
        out, rows = peers.reconcile(LIVE, want, hub="hub")
        for p in out:
            if p["name"] in want:
                assert p["url"] == want[p["name"]]["url"]
                assert peers.address_kind(p["url"]) == peers.DURABLE

    def test_running_it_twice_changes_nothing_the_second_time(self, repo):
        want = self.want(repo)
        out, _ = peers.reconcile(LIVE, want, hub="hub")
        # What the hub would hand back afterwards: the written URLs, tokens
        # restored on its side.
        after = [{**p, "has_token": True} for p in out]
        _, rows = peers.reconcile(after, want, hub="hub")
        assert [r["action"] for r in rows
                if r["action"] != "unregistered"] == ["ok"] * len(after)
        assert peers.brittle(rows) == []


# --------------------------------------------------------------------------
class TestTheCommandDoesNotSpeakOverItsOwnJson:
    """`plan --json` printing a warning to stdout broke three CI runners once.
    `peers --json` is parsed the same way."""

    def test_desired_reports_a_missing_tailnet_ip_instead_of_raising(
            self, tmp_path):
        (tmp_path / "hosts" / "lonely").mkdir(parents=True)
        hostfile.dump({"host": {"name": "lonely"}, "network": {"port": 8080}},
                      tmp_path / "hosts" / "lonely" / "host.yml")
        (tmp_path / "hosts" / "fine").mkdir(parents=True)
        hostfile.dump({"host": {"name": "fine"},
                       "network": {"tailnet_ip": "100.64.0.1", "port": 8080}},
                      tmp_path / "hosts" / "fine" / "host.yml")
        want, problems = peers.desired(tmp_path)
        assert list(want) == ["fine"]
        assert len(problems) == 1 and "lonely" in problems[0]

    def test_the_port_comes_from_fleet_yml_when_the_host_is_silent(self, tmp_path):
        hostfile.dump({"network": {"port": 9999}}, tmp_path / "fleet.yml")
        (tmp_path / "hosts" / "b").mkdir(parents=True)
        hostfile.dump({"host": {"name": "b"},
                       "network": {"tailnet_ip": "100.64.0.2"}},
                      tmp_path / "hosts" / "b" / "host.yml")
        want, _ = peers.desired(tmp_path)
        assert want["b"]["url"] == "http://100.64.0.2:9999"


# --------------------------------------------------------------------------
class TestUnreadableIsNotAbsentHereEither:
    """The env file is 0640 root:llmstack. An unprivileged `fleetctl peers`
    got nothing from it and said "no admin token in /etc/llmstack/gateway.env"
    -- which sends you looking for a missing value when what you needed was
    sudo. Same distinction four other steps on this branch had to learn."""

    def _ctx(self, tmp_path, linux_facts, empty_repo, mode):
        from fleetctl import planner
        from fleetctl.runner import Ctx
        plan, _ = planner.build(linux_facts, repo=empty_repo)
        env = tmp_path / "gateway.env"
        plan["paths"]["envfile"] = str(env)
        if mode is not None:
            env.write_text("LLMSTACK_ADMIN_TOKEN=sk-secret\n", encoding="utf-8")
        return Ctx(plan, linux_facts, repo=empty_repo), plan, env

    def test_a_readable_file_gives_the_token(self, tmp_path, linux_facts,
                                             empty_repo):
        ctx, plan, _ = self._ctx(tmp_path, linux_facts, empty_repo, 0o600)
        assert cli._admin_token(ctx, plan) == ("sk-secret", "")

    def test_an_absent_file_says_absent(self, tmp_path, linux_facts, empty_repo):
        ctx, plan, _ = self._ctx(tmp_path, linux_facts, empty_repo, None)
        token, why = cli._admin_token(ctx, plan)
        assert token == "" and "absent" in why

    def test_a_file_with_no_token_is_not_the_same_complaint(
            self, tmp_path, linux_facts, empty_repo):
        ctx, plan, env = self._ctx(tmp_path, linux_facts, empty_repo, 0o600)
        env.write_text("LLMSTACK_PORT=8080\n", encoding="utf-8")
        token, why = cli._admin_token(ctx, plan)
        assert token == "" and "no LLMSTACK_ADMIN_TOKEN" in why

    def test_an_unreadable_file_is_reported_as_unreadable(
            self, tmp_path, linux_facts, empty_repo, monkeypatch):
        ctx, plan, _ = self._ctx(tmp_path, linux_facts, empty_repo, 0o600)
        # chmod 0000 does not stop root, and CI runs as root in containers --
        # so refuse at the read instead of relying on the filesystem.
        def denied(_p):
            raise PermissionError(13, "Permission denied")
        monkeypatch.setattr(ctx, "_slurp", denied)
        token, why = cli._admin_token(ctx, plan)
        assert token == "" and "unreadable" in why
        assert "absent" not in why


# --------------------------------------------------------------------------
class TestThePushChecksItsOwnWork:
    """An adversarial review of this command found the safety net crying wolf
    and the write racing an editor. Both are here so they stay fixed."""

    def test_a_peer_that_never_had_a_token_is_not_reported_as_damage(self):
        """The first version flagged every token-less peer in the read-back,
        which on this fleet meant a WARNING and exit 2 on a correct push --
        and a warning that fires every run is one nobody reads."""
        before = [{"name": "a", "has_token": True},
                  {"name": "stranger", "has_token": False}]
        after = [{"name": "a", "has_token": True},
                 {"name": "stranger", "has_token": False}]
        assert cli._push_damage(before, after) == ([], [])

    def test_a_token_that_really_went_missing_is_reported(self):
        before = [{"name": "a", "has_token": True}]
        after = [{"name": "a", "has_token": False}]
        assert cli._push_damage(before, after) == (["a"], [])

    def test_a_peer_that_vanished_is_reported(self):
        before = [{"name": "a", "has_token": True}, {"name": "b", "has_token": True}]
        after = [{"name": "a", "has_token": True}]
        assert cli._push_damage(before, after) == ([], ["b"])

    def test_a_new_peer_appearing_is_not_damage(self):
        before = [{"name": "a", "has_token": True}]
        after = [{"name": "a", "has_token": True}, {"name": "c", "has_token": True}]
        assert cli._push_damage(before, after) == ([], [])

    def test_the_fingerprint_ignores_order_but_not_content(self):
        """The guard re-reads the list right before writing. Order out of the
        hub is not meaningful; a changed URL or a new peer is."""
        a = [{"name": "x", "url": "http://100.64.0.1:8080/", "api_url": "u",
              "has_token": True},
             {"name": "y", "url": "http://100.64.0.2:8080", "api_url": "v",
              "has_token": True}]
        assert cli._peer_fingerprint(a) == cli._peer_fingerprint(list(reversed(a)))
        moved = [dict(a[0]), {**a[1], "url": "http://100.64.0.9:8080"}]
        assert cli._peer_fingerprint(a) != cli._peer_fingerprint(moved)
        assert cli._peer_fingerprint(a) != cli._peer_fingerprint(a + [
            {"name": "z", "url": "http://100.64.0.3:8080", "api_url": "w",
             "has_token": True}])

    def test_the_trailing_slash_the_hub_sometimes_stores_is_not_a_change(self):
        a = [{"name": "x", "url": "http://100.64.0.1:8080", "api_url": "u",
              "has_token": True}]
        b = [{"name": "x", "url": "http://100.64.0.1:8080/", "api_url": "u",
              "has_token": True}]
        assert cli._peer_fingerprint(a) == cli._peer_fingerprint(b)

    def test_a_rejected_push_says_which_peer_the_hub_refused(self, monkeypatch):
        """`400 bad peer entry: <name>` names the one entry of thirteen that
        failed validation. Dropping the body left only 'HTTP 400'."""
        import io
        import urllib.error
        import urllib.request

        def boom(req, timeout=None):
            raise urllib.error.HTTPError(
                req.full_url, 400, "Bad Request", {},
                io.BytesIO(b'{"detail":"bad peer entry: we!rd"}'))
        monkeypatch.setattr(urllib.request, "urlopen", boom)
        out, err = cli._hub_call("http://127.0.0.1:8080", "tok")
        assert out is None
        assert "400" in err and "bad peer entry: we!rd" in err

    def test_an_unparseable_error_body_is_not_itself_an_error(self, monkeypatch):
        import io
        import urllib.error
        import urllib.request

        def boom(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 502, "Bad Gateway", {},
                                         io.BytesIO(b"<html>nginx</html>"))
        monkeypatch.setattr(urllib.request, "urlopen", boom)
        out, err = cli._hub_call("http://127.0.0.1:8080", "tok")
        assert out is None and "502" in err
