"""Two plan keys a box fronting someone else's engine needs: where the
engine's key lives, and who may reach the gateway at all.

The npu-box-1's vendor server wants a bearer key on /v1, so the gateway in front
of it is told a FILE to read it from (network.upstream_key_file ->
LLMSTACK_UPSTREAM_KEY_FILE) -- never the value, which would then sit in the
env file and in any error that echoes it. And the PC that gateway runs on
also runs an agent with a shell, so its inbound rule admits the hub alone
(network.allow_from -> the Windows rule's remoteip=). The firewall step
deletes and re-adds the rule on every apply, so a scope set by hand would be
gone the next time; it has to be part of the plan.
"""
from __future__ import annotations

from types import SimpleNamespace

from fleetctl import planner
from fleetctl.runner import Ctx
from fleetctl.steps import MISSING, OK
from fleetctl.steps.stack import EnvFile
from fleetctl.steps.system import Firewall

HUB = "192.0.2.10"   # documentation range (RFC 5737), standing in for the hub


def ctx_for(facts, repo, tmp_path, **kw):
    plan, prov = planner.build(facts, repo=repo)
    ctx = Ctx(plan, facts, repo=repo, root=str(tmp_path / "sandbox"), **kw)
    ctx.prov = prov
    return ctx


def netsh_show(remote_ip: str):
    """A ctx.probe answering `netsh advfirewall firewall show rule` the way
    an English Windows 11 does."""
    text = ("\nRule Name:                            llm-gateway\n"
            "----------------------------------------------------------------------\n"
            "Enabled:                              Yes\n"
            "Direction:                            In\n"
            "LocalIP:                              Any\n"
            f"RemoteIP:                             {remote_ip}\n"
            "Protocol:                             TCP\n"
            "LocalPort:                            8080\n"
            "Action:                               Allow\nOk.\n")
    return lambda argv, timeout=60: SimpleNamespace(returncode=0, stdout=text, stderr="")


class TestUpstreamKeyFile:
    def test_the_path_reaches_the_env_file(self, windows_facts, empty_repo, tmp_path):
        ctx = ctx_for(windows_facts, empty_repo, tmp_path)
        ctx.plan["network"]["upstream_key_file"] = r"C:\llmstack\etc\upstream.key"
        vals = EnvFile()._values(ctx)
        assert vals["LLMSTACK_UPSTREAM_KEY_FILE"] == r"C:\llmstack\etc\upstream.key"

    def test_unset_writes_nothing(self, windows_facts, empty_repo, tmp_path):
        ctx = ctx_for(windows_facts, empty_repo, tmp_path)
        assert "LLMSTACK_UPSTREAM_KEY_FILE" not in EnvFile()._values(ctx)

    def test_the_schema_knows_the_key(self):
        assert planner.SCHEMA["network"]["upstream_key_file"] == (str, False)


class TestAllowFromValidation:
    def test_an_address_and_a_cidr_are_fine_on_windows(self, windows_facts, empty_repo):
        plan, _ = planner.build(windows_facts, repo=empty_repo)
        plan["network"]["allow_from"] = [HUB, "100.64.0.0/10"]
        assert not [p for p in planner.validate(plan) if "allow_from" in p]

    def test_a_name_is_not_an_address(self, windows_facts, empty_repo):
        plan, _ = planner.build(windows_facts, repo=empty_repo)
        plan["network"]["allow_from"] = ["hub"]
        assert any("allow_from" in p and "hub" in p for p in planner.validate(plan))

    def test_a_cidr_with_host_bits_set_is_a_typo(self, windows_facts, empty_repo):
        plan, _ = planner.build(windows_facts, repo=empty_repo)
        plan["network"]["allow_from"] = [HUB + "/24"]
        assert any("allow_from" in p for p in planner.validate(plan))

    def test_it_is_refused_where_it_would_not_be_applied(self, linux_facts, empty_repo):
        plan, _ = planner.build(linux_facts, repo=empty_repo)
        plan["network"]["allow_from"] = [HUB]
        assert any("allow_from" in p and "Windows" in p for p in planner.validate(plan))


class TestWindowsRuleScope:
    def test_apply_scopes_the_rule(self, windows_facts, empty_repo, tmp_path):
        ctx = ctx_for(windows_facts, empty_repo, tmp_path)
        ctx.plan["network"]["allow_from"] = [HUB]
        issued: list[list[str]] = []
        ctx.run = lambda argv, **kw: issued.append([str(a) for a in argv])
        Firewall().apply(ctx)
        add = [c for c in issued if c[:5] == ["netsh", "advfirewall", "firewall", "add", "rule"]
               and "name=llm-gateway" in c]
        assert add and "remoteip=" + HUB in add[0]

    def test_apply_without_a_scope_is_unchanged(self, windows_facts, empty_repo, tmp_path):
        ctx = ctx_for(windows_facts, empty_repo, tmp_path)
        issued: list[list[str]] = []
        ctx.run = lambda argv, **kw: issued.append([str(a) for a in argv])
        Firewall().apply(ctx)
        add = [c for c in issued if "name=llm-gateway" in c and "add" in c]
        assert add and not any(a.startswith("remoteip=") for a in add[0])

    def test_an_unscoped_rule_is_not_converged(self, windows_facts, empty_repo, tmp_path):
        ctx = ctx_for(windows_facts, empty_repo, tmp_path)
        ctx.plan["network"]["allow_from"] = [HUB]
        ctx.probe = netsh_show("Any")
        assert Firewall().check(ctx).state == MISSING

    def test_a_rule_scoped_to_the_hub_is(self, windows_facts, empty_repo, tmp_path):
        ctx = ctx_for(windows_facts, empty_repo, tmp_path)
        ctx.plan["network"]["allow_from"] = [HUB]
        ctx.probe = netsh_show(HUB + "/32")
        assert Firewall().check(ctx).state == OK

    def test_a_rule_scoped_wider_than_the_plan_is_not(self, windows_facts, empty_repo, tmp_path):
        ctx = ctx_for(windows_facts, empty_repo, tmp_path)
        ctx.plan["network"]["allow_from"] = [HUB]
        ctx.probe = netsh_show(HUB + "/32,100.64.0.0/10")
        assert Firewall().check(ctx).state == MISSING

    def test_a_second_unscoped_rule_of_the_same_name_is_not(self, windows_facts, empty_repo,
                                                              tmp_path):
        ctx = ctx_for(windows_facts, empty_repo, tmp_path)
        ctx.plan["network"]["allow_from"] = [HUB]
        scoped, wide = netsh_show(HUB + "/32"), netsh_show("Any")
        both = scoped(None).stdout + wide(None).stdout
        ctx.probe = lambda argv, timeout=60: SimpleNamespace(returncode=0, stdout=both, stderr="")
        assert Firewall().check(ctx).state == MISSING

    def test_output_it_cannot_read_is_not_converged(self, windows_facts, empty_repo, tmp_path):
        ctx = ctx_for(windows_facts, empty_repo, tmp_path)
        ctx.plan["network"]["allow_from"] = [HUB]
        ctx.probe = netsh_show(HUB + "/32")
        german = ctx.probe(None).stdout.replace("RemoteIP:", "Remote-IP:")
        ctx.probe = lambda argv, timeout=60: SimpleNamespace(returncode=0, stdout=german, stderr="")
        assert Firewall().check(ctx).state == MISSING

    def test_no_scope_planned_means_any_rule_will_do(self, windows_facts, empty_repo, tmp_path):
        ctx = ctx_for(windows_facts, empty_repo, tmp_path)
        ctx.probe = netsh_show("Any")
        assert Firewall().check(ctx).state == OK
