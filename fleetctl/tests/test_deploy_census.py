"""Every box with a plan is a box CI deploys to.

gpu-laptop-4 joined on 2026-09-05 and gpu-laptop-5 on 2026-09-26, each with a
host.yml, a spec sheet and a place in the README -- and neither was in
deploy-gateway.sh. Nothing failed. `./deploy-gateway.sh all` went green
without them every half hour, and the two boxes simply stayed on the gateway
they were provisioned with.

This reads the checked-in script; it runs nothing and reaches no machine.

A plan marked `host.parked: true` is a box that has left (npu-box-1, 2026-10-24)
and whose plan is kept for its return -- and the rule inverts for it: it
must appear in no list at all.
"""
from __future__ import annotations

import re

import pytest

from fleetctl import hostfile

# The fleet name on the left is the key everything else uses. The deploy
# script's name on the right is the ssh alias it grew up with -- two strings
# on purpose (see `--host mac-desktop is not --host mac-desktop-1` in docs/fleetctl.md).
DEPLOY_NAME = {"mac-desktop-1": "mac-desktop"}

FAMILY_LIST = {"linux": "LINUX", "darwin": "MAC", "windows": "WIN"}


@pytest.fixture(scope="module")
def script(repo) -> str:
    path = repo / "deploy-gateway.sh"
    if not path.is_file():
        pytest.skip("no deploy-gateway.sh in this checkout")
    return path.read_text(encoding="utf-8")


def _list(script: str, name: str) -> list[str]:
    m = re.search(rf'^{name}="([^"]*)"', script, re.M)
    assert m, f"no {name}= line in deploy-gateway.sh"
    return m.group(1).split()


def _plans(repo) -> dict[str, dict]:
    out = {}
    for path in sorted((repo / "hosts").glob("*/host.yml")):
        out[path.parent.name] = hostfile.load(path)
    if not out:
        pytest.skip("no host plans in this checkout")
    return out


def _parked(plan: dict) -> bool:
    """`host.parked: true` -- a plan kept for a box that is nowhere right now.

    npu-box-1 is the case (2026-10-24): the npu-box-1 left BLACKBOX and its plan is
    kept, address commented out, for the node it is next hooked up through.
    Such a plan must NOT be in a deploy list -- the script would resolve the
    bare name to nothing and report a phantom box offline every half hour,
    or worse, deploy to the address of whatever machine inherited it.
    `fleetctl peers` skips it for the same reason (peers.desired()).
    """
    return bool((plan.get("host") or {}).get("parked"))


class TestEveryPlannedBoxIsDeployedTo:
    def test_each_one_is_in_the_list_for_its_own_os(self, repo, script):
        missing = []
        for name, plan in _plans(repo).items():
            if _parked(plan):
                continue
            family = (plan.get("platform") or {}).get("os")
            wanted = DEPLOY_NAME.get(name, name)
            if wanted not in _list(script, FAMILY_LIST[family]):
                missing.append(f"{name} ({family}) is not in {FAMILY_LIST[family]}")
        assert missing == []

    def test_a_parked_plan_with_no_address_is_in_no_list(self, repo, script):
        listed = {h for fam in FAMILY_LIST.values() for h in _list(script, fam)}
        parked = {DEPLOY_NAME.get(n, n) for n, p in _plans(repo).items() if _parked(p)}
        assert parked & listed == set(), (
            "a parked plan (host.parked) has no box to deploy to: "
            f"{sorted(parked & listed)}")
        # And the one parked today is the one this rule was written for.
        if "npu-box-1" in _plans(repo):
            assert "npu-box-1" in parked

    def test_nothing_is_deployed_to_that_has_no_plan(self, repo, script):
        planned = {DEPLOY_NAME.get(n, n) for n in _plans(repo)}
        listed = {h for fam in FAMILY_LIST.values() for h in _list(script, fam)}
        assert listed - planned == set()

    @pytest.mark.parametrize("name", ["gpu-laptop-4", "gpu-laptop-5"])
    def test_the_two_laptops_are_reached_by_the_address_in_their_plan(
            self, repo, script, name):
        """By 100.x, never by name: the default case in ssh_target() hands
        back the bare name, which resolves only through whatever ssh config
        the runner happens to have."""
        plans = _plans(repo)
        if name not in plans:
            pytest.skip(f"no hosts/{name}/host.yml in this checkout")
        ip = plans[name]["network"]["tailnet_ip"]
        assert re.search(rf'^\s*{re.escape(name)}\)\s+echo "user@{re.escape(ip)}" ;;',
                         script, re.M)


class TestAwaitingKeyIsAboutOnboardingOnly:
    def test_only_a_listed_box_can_be_pending(self, script):
        pending = _list(script, "KEY_PENDING")
        deployed = {h for fam in FAMILY_LIST.values() for h in _list(script, fam)}
        assert set(pending) <= deployed

    def test_the_pending_branch_comes_before_the_failing_one_and_never_fails(
            self, script):
        """A box in KEY_PENDING that refuses the key goes to `pend`, which is
        reported and does not reach `bad`. Every other box still fails."""
        first = script.index('"__AUTHFAIL__" && " $KEY_PENDING "')
        second = script.index(
            'if [[ "$state" == "__AUTHFAIL__" || "$state" == "__HOSTKEY__" ]]; then')
        assert first < second
        branch = script[first:script.index("fi", first)]
        assert 'pend+=("$h"); continue' in branch
        assert "bad+=" not in branch
        assert 'bad+=("$h"); continue' in script[second:second + 3000]

    def test_a_changed_host_key_is_never_mistaken_for_a_missing_key(self, script):
        """Both used to be __AUTHFAIL__. A pending box whose IDENTITY had
        changed -- a reinstall, another machine on the old address -- would
        have been "awaiting key" every half hour, green, for ever."""
        probe = script[script.index("probe_read() {"):]
        probe = probe[:probe.index("\n}\n")]
        hostkey = probe.index('"host key verification failed"')
        denied = probe.index('"permission denied|too many authentication failures"')
        assert hostkey < denied
        assert "echo __HOSTKEY__" in probe[hostkey:denied]
        assert "host key" not in probe[denied:]
        # ...and the pending branch is about the one state only.
        first = script.index('"__AUTHFAIL__" && " $KEY_PENDING "')
        line = script[script.rindex("\n", 0, first):script.index("\n", first)]
        assert "__HOSTKEY__" not in line

    def test_a_pending_box_is_named_in_the_summary(self, script):
        assert re.search(r'\$\{#pend\[@\]\} -gt 0 \]\] && echo "awaiting key:', script)


class TestAGatewayItsOwnerSwitchedOffIsLeftOff:
    """The keepalive means ending the gateway no longer keeps it off, so the
    way to have a laptop to yourself is to disable the task. A deploy that
    answered that by running it would be the fleet overruling the person
    whose machine it is."""

    def test_the_probe_reports_it_before_anything_else_is_asked(self, script):
        m = re.search(r"WIN_PROBE_PS='(.*?)'\nWIN_PROBE_B64", script, re.S)
        assert m, "no WIN_PROBE_PS payload"
        payload = m.group(1)
        disabled = payload.index("__DISABLED__")
        assert "-not $task.Settings.Enabled" in payload[:disabled]
        assert disabled < payload.index("/health")
        # By the task's own state, not by a word in schtasks' output that a
        # localised Windows spells differently.
        assert "Get-ScheduledTask -TaskName llm-gateway" in payload

    def test_it_is_skipped_green_and_said(self, script):
        start = script.index('if [[ "$state" == "__DISABLED__" ]]; then')
        branch = script[start:script.index("\n  fi\n", start)]
        assert 'swoff+=("$h"); continue' in branch
        assert "bad+=" not in branch
        assert re.search(r'\$\{#swoff\[@\]\} -gt 0 \]\] && echo "switched off:', script)

    def test_it_is_decided_before_the_deploy_in_both_modes(self, script):
        """A push deploys to a box that is busy. It must not deploy to one
        that is switched off."""
        assert script.index('"__DISABLED__" ]]; then') < script.index('"__BUSY__" ]]; then')


class TestADeployCannotPutABoxBackIntoUse:
    """"Not in use" is the hub's peers.json `routed` flag, set from the
    dashboard. The deploy has no business near it."""

    def test_the_script_never_mentions_the_peer_list_or_the_flag(self, script):
        code = "\n".join(line for line in script.splitlines()
                         if not line.lstrip().startswith("#"))
        for word in ("peers.json", "/admin/api/peers", "fleet-config", "routed"):
            assert word not in code, word

    def test_the_windows_restart_touches_one_task_and_only_fleet_processes(
            self, script):
        m = re.search(r"WIN_RESTART_PS='(.*?)\nexit 1'", script, re.S)
        assert m, "no WIN_RESTART_PS payload"
        payload = m.group(1)
        tasks = set(re.findall(r"/TN (\S+)", payload))
        assert tasks == {"llm-gateway"}
        # The only kill is of a uvicorn started out of the fleet's own
        # prefix. Matching on the name alone is how the gpu-laptop-2 box lost
        # every python.exe its owner had open.
        kills = [ln for ln in payload.splitlines() if "Stop-Process" in ln]
        assert len(kills) == 1
        assert 'CommandLine -match "uvicorn app:app"' in kills[0]
        assert 'CommandLine -match "llmstack"' in kills[0]
        for other in ("llama-swap", "llama-server", "ollama", "LM Studio", "taskkill"):
            assert other not in payload, other
