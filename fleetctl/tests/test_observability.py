"""The observability-agent step: gating, schema, the policy it renders, what
check() says about a box, what apply() does to one, and the download.

Everything here runs in a sandbox with fakes in place of systemd, the
installer and the network. Nothing starts a process, opens a socket or leaves
the tmp directory, so it runs unchanged on a workstation and in CI.

Two rules the whole file keeps, because the step's own docstring makes them
the point of it:

  * the ingest key, the CA fingerprint and the gateway's admin token appear in
    the policy (the first two, where the agent needs them), in files, and
    nowhere else -- not in an argv, a detail, a plan line, a recorded action
    or an exception message;
  * check() changes nothing and never raises.

Fixtures use generic names and values only. The fingerprints are built at
runtime ("ab" * 32) so no 64-hex string is committed.
"""
from __future__ import annotations

import datetime
import hashlib
import io
import json
import os
import posixpath
import re
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from fleetctl import planner
from fleetctl import shapes
from fleetctl import steps as steps_mod
from fleetctl.runner import Ctx
from fleetctl.runner import StepFailed
from fleetctl.steps import BLOCKED, DRIFT, MISSING, OK
from fleetctl.steps import observability as obs
from fleetctl.steps.stack import EnvFile

KEY = "kid1234:sec5678"
FP = "ab" * 32
TOKEN = "gwtok" + "0123456789abcdef" * 2
VERSION = "9.5.6"
ASSET = "linux-x86_64.tar.gz"
PIN = f"{VERSION}/{ASSET}=" + "cd" * 64
HUB_URL = "https://100.64.0.9:9200"
SECRET_VALUES = (KEY, "kid1234", "sec5678", FP, TOKEN)
NOW = datetime.datetime(2026, 11, 1, 12, 0, tzinfo=datetime.timezone.utc)
POSIX = pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes and symlinks")


# --------------------------------------------------------------------------
# a tiny reader for the YAML the step emits
# --------------------------------------------------------------------------
# The step writes block-style YAML whose every scalar is a JSON scalar, so a
# fifty-line reader is enough to turn a rendered policy back into a dict -- and
# it keeps these tests stdlib-only, as the rest of fleetctl's are. PyYAML is
# used as a second opinion where it happens to be installed.
_KEY_LINE = re.compile(r'^("(?:[^"\\]|\\.)*"|[A-Za-z_][A-Za-z0-9_]*):(?: (.*))?$')


def _inline(token: str):
    if token == "{}":
        return {}
    if token == "[]":
        return []
    return json.loads(token)


def _block(rows: list, i: int, indent: int):
    if rows[i][1].startswith("- "):
        items = []
        while i < len(rows) and rows[i][0] == indent and rows[i][1].startswith("- "):
            body = rows[i][1][2:]
            if _KEY_LINE.match(body):
                # `- key: v` opens a mapping whose other keys sit two columns in.
                rows[i] = (indent + 2, body)
                item, i = _block(rows, i, indent + 2)
            else:
                item, i = _inline(body), i + 1
            items.append(item)
        return items, i
    mapping = {}
    while i < len(rows) and rows[i][0] == indent:
        m = _KEY_LINE.match(rows[i][1])
        assert m, f"not a line this emitter writes: {rows[i][1]!r}"
        name = json.loads(m.group(1)) if m.group(1).startswith('"') else m.group(1)
        i += 1
        if m.group(2) is None:
            mapping[name], i = _block(rows, i, rows[i][0])
        else:
            mapping[name] = _inline(m.group(2))
    return mapping, i


def parse_yaml(text: str):
    rows = []
    for raw in text.splitlines():
        if raw.strip() and not raw.lstrip().startswith("#"):
            rows.append((len(raw) - len(raw.lstrip(" ")), raw.strip()))
    value, end = _block(rows, 0, rows[0][0])
    assert end == len(rows), f"stopped reading at {rows[end]!r}"
    return value


def streams_of(policy: dict) -> list:
    return [s for inp in policy["inputs"] for s in inp["streams"]]


def datasets_of(policy: dict) -> set:
    return {s["data_stream"]["dataset"] for s in streams_of(policy)}


def stream_by_id(policy: dict, sid: str) -> dict:
    return next(s for s in streams_of(policy) if s["id"] == sid)


def input_by_id(policy: dict, iid: str) -> dict:
    return next(i for i in policy["inputs"] if i["id"] == iid)


# --------------------------------------------------------------------------
# fixtures and helpers
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _root_and_a_fixed_clock(monkeypatch):
    # The real uid is never root on a workstation or in a CI job, and the key's
    # expiry warning depends on the date; both are seams for exactly this.
    monkeypatch.setattr(obs, "_is_root", lambda: True)
    monkeypatch.setattr(obs, "_now", lambda: NOW)
    # One look and no sleeping in the watch apply keeps after it starts the agent.
    monkeypatch.setattr(obs, "SETTLE_S", 0)


def secrets_text(**over) -> str:
    vals = {"ES_HOSTS": HUB_URL, "ES_API_KEY": KEY, "ES_CA_FINGERPRINT": FP}
    vals.update(over)
    return "".join(f"{k}={v}\n" for k, v in vals.items() if v is not None)


def secrets_dict(**over) -> dict:
    return obs._read_secrets_env(secrets_text(**over))


def render(role="peer", section=None, secrets=None, **kw) -> str:
    cfg = obs._merged(section, role)
    return obs.render_policy(cfg, secrets or secrets_dict(), port=8080, role=role, **kw)


def put(ctx, path: str, text: str, mode: int = 0o644) -> None:
    target = ctx.path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="")
    if os.name != "nt":
        target.chmod(mode)


def snapshot(root: Path) -> list:
    out = []
    for p in sorted(root.rglob("*")):
        out.append((p.relative_to(root).as_posix(), p.read_bytes() if p.is_file() else None))
    return out


def make_ctx(facts, repo, tmp_path, *, section=None, role="peer", **kw) -> Ctx:
    """A sandboxed Ctx for a Linux box whose home is not the maintainer's.
    `section` None means the usual opt-in with one valid pin; {} means no
    observability config at all."""
    facts = dict(facts, home="/home/user")
    cfg = {"enabled": True, "agent_sha512": [PIN]} if section is None else section
    overrides: dict = {"host": {"role": role}}
    if cfg:
        overrides["observability"] = cfg
    plan, _ = planner.build(facts, repo=repo, overrides=overrides)
    return Ctx(plan, facts, repo=repo, root=str(tmp_path / "sandbox"), **kw)


def _out(rc, text=""):
    return SimpleNamespace(returncode=rc, stdout=text, stderr="")


class FakeBox:
    """systemd, the installer and `sudo`, as far as the step can tell.

    The state lives here, so a sequence (apply, then check) sees a box that
    changed. `sudo` runs the two commands whose effect the step later reads
    back -- chmod, and the installer's copy of the staged policy -- and records
    the rest.
    """

    def __init__(self, ctx, version_out=f"Binary: {VERSION}, commit: 0abc, built: today"):
        self.ctx = ctx
        self.unit = False
        self.active, self.sub, self.enabled = "inactive", "dead", "disabled"
        self.version_out = version_out
        self.status_out = None
        self.sudo_calls: list = []
        self.run_calls: list = []
        self.fail_on = None
        self.staged_mode = None
        self.uninstall_cwd = None
        self.unit_probe_none = False
        self.unit_probe_rc = 0
        ctx.probe = self.probe
        ctx.sudo = self.sudo
        ctx.run = lambda argv, **kw: self.run_calls.append(([str(a) for a in argv], kw))

    @property
    def argvs(self) -> list:
        return [a for a, _ in self.sudo_calls]

    def started(self) -> None:
        self.unit = True
        self.active, self.sub, self.enabled = "active", "running", "enabled"

    def probe(self, argv, timeout=60, env=None):
        argv = [str(a) for a in argv]
        if argv == ["systemctl", "show", obs.UNIT, "-p", "FragmentPath"]:
            if self.unit_probe_none:
                return None
            fragment = "/etc/systemd/system/elastic-agent.service" if self.unit else ""
            return _out(self.unit_probe_rc, f"FragmentPath={fragment}\n")
        if argv[:2] == ["systemctl", "show"]:
            return _out(0, f"ActiveState={self.active}\nSubState={self.sub}\n"
                           f"UnitFileState={self.enabled}\n")
        if argv == [obs.AGENT_BIN, "version", "--binary-only"]:
            return None if self.version_out is None else _out(0, self.version_out)
        if argv[:2] == [obs.AGENT_BIN, "status"]:
            return None if self.status_out is None else _out(*self.status_out)
        raise AssertionError(f"a probe nobody planned for: {argv}")

    def sudo(self, argv, **kw):
        argv = [str(a) for a in argv]
        self.sudo_calls.append((argv, kw))
        if self.fail_on is not None:
            self.fail_on(argv)
        if argv[0] == "chmod" and os.name != "nt":
            target = self.ctx.path(argv[2])
            if target.exists():
                target.chmod(int(argv[1], 8))
        elif argv[1:] == ["install", "--non-interactive"]:
            staged = Path(kw["cwd"]) / "elastic-agent.yml"
            self.staged_mode = None if os.name == "nt" else staged.stat().st_mode & 0o777
            target = self.ctx.path(obs.AGENT_POLICY)
            target.parent.mkdir(parents=True, exist_ok=True)
            # A plain copy: the installer's file gets the umask's mode, not the
            # staged file's, which is why the step has to chmod it afterwards.
            shutil.copyfile(staged, target)
            put(self.ctx, obs.AGENT_BIN, "#!/bin/sh\n", 0o755)
            self.started()
        elif argv[:3] == ["systemctl", "enable", "--now"]:
            self.started()
        elif argv[0] == obs.AGENT_BIN and argv[1] == "uninstall":
            self.uninstall_cwd = kw.get("cwd")
            self.unit = False
            self.ctx.path(obs.AGENT_BIN).unlink()


class Rig:
    """A sandboxed box, its fakes, and the helpers every check test wants."""

    def __init__(self, facts, repo, tmp_path, *, section=None, role="peer",
                 token=TOKEN, seed=True, **kw):
        self.ctx = make_ctx(facts, repo, tmp_path, section=section, role=role, **kw)
        self.fake = FakeBox(self.ctx)
        self.step = obs.ObservabilityAgent()
        self.token = token
        if seed:
            self.seed()

    @property
    def cfg(self) -> dict:
        return obs._cfg(self.ctx)

    def seed(self, secrets=None, secrets_mode=0o600) -> None:
        """What a real run has before this step: the env file the envfile step
        wrote, and the key file the operator carried to the box."""
        put(self.ctx, self.ctx.plan["paths"]["envfile"], f"LLMSTACK_ADMIN_TOKEN={self.token}\n", 0o640)
        path = self.cfg["secrets_file"]
        put(self.ctx, path, secrets_text() if secrets is None else secrets, secrets_mode)
        if os.name != "nt":
            # umask 002 (some CI images) would make the directory group-writable
            # and trip the trust check this file is not about.
            self.ctx.path(posixpath.dirname(path)).chmod(0o755)

    def expected_policy(self, secrets=None) -> str:
        return obs.render_policy(
            self.cfg, secrets or secrets_dict(), port=self.ctx.plan["network"]["port"],
            role=self.ctx.plan["host"]["role"],
            access_log=shapes.access_log_path("linux", self.ctx.plan["paths"]["state"]))

    def settle(self) -> None:
        """An installed, running agent whose files are exactly what the plan wants."""
        put(self.ctx, obs.AGENT_POLICY, self.expected_policy(), 0o600)
        put(self.ctx, obs.TOKEN_FILE, self.token, 0o600)
        put(self.ctx, obs.DROPIN, obs._dropin(self.cfg), 0o644)
        put(self.ctx, obs.AGENT_BIN, "#!/bin/sh\n", 0o755)
        self.fake.started()

    def check(self):
        return self.step.check(self.ctx)

    def apply(self) -> None:
        self.step.apply(self.ctx)


@pytest.fixture
def build_rig(linux_facts, empty_repo, tmp_path):
    def build(**kw) -> Rig:
        return Rig(linux_facts, empty_repo, tmp_path, **kw)
    return build


@pytest.fixture
def downloads(monkeypatch, tmp_path):
    """Everything apply() touches outside the sandbox, made harmless: the
    download, the work directory, and the disk-space reading."""
    calls: list = []
    work = tmp_path / "work-parent"
    work.mkdir()

    def fake_download(ctx, dest_dir):
        calls.append(Path(dest_dir))
        src = Path(dest_dir) / "unpacked" / f"elastic-agent-{VERSION}-linux-x86_64"
        src.mkdir(parents=True)
        (src / "elastic-agent").write_text("#!/bin/sh\n", encoding="utf-8")
        return src

    monkeypatch.setattr(obs, "_download_and_verify", fake_download)
    monkeypatch.setattr(obs, "_workdir_parent", lambda: str(work))
    monkeypatch.setattr(obs, "_free", lambda where: 10 ** 12)
    return calls


def no_secret_in(*texts) -> None:
    blob = "\n".join(str(t) for t in texts)
    for value in SECRET_VALUES:
        assert value not in blob, f"{value[:6]}... leaked"


# ==========================================================================
# 1. gating and registration
# ==========================================================================
class TestGating:
    def test_absent_config_is_not_wanted(self, linux_facts, empty_repo, tmp_path):
        ctx = make_ctx(linux_facts, empty_repo, tmp_path, section={})
        assert ctx.plan["observability"]["enabled"] is None
        assert obs.ObservabilityAgent().wanted(ctx) is False

    def test_enabled_false_is_not_wanted(self, linux_facts, empty_repo, tmp_path):
        ctx = make_ctx(linux_facts, empty_repo, tmp_path, section={"enabled": False})
        assert ctx.plan["observability"]["enabled"] is False
        assert obs.ObservabilityAgent().wanted(ctx) is False

    def test_enabled_true_is_wanted(self, linux_facts, empty_repo, tmp_path):
        ctx = make_ctx(linux_facts, empty_repo, tmp_path)
        assert obs.ObservabilityAgent().wanted(ctx) is True

    def test_a_blank_section_in_the_plan_does_not_crash_wanted(self, linux_facts, empty_repo, tmp_path):
        ctx = make_ctx(linux_facts, empty_repo, tmp_path, section={})
        ctx.plan.pop("observability")
        assert obs.ObservabilityAgent().wanted(ctx) is False

    def test_registered_after_health_and_last(self):
        ids = [s.id for s in steps_mod.catalogue()]
        assert ids.index("health") < ids.index("observability-agent")
        assert steps_mod.ORDER[-1] == "observability-agent"
        assert steps_mod.ORDER[-2] == "health"

    def test_it_is_a_system_step(self):
        # system=True is what makes a sandbox refuse its commands and an
        # unprivileged run report it as needing root.
        step = next(s for s in steps_mod.catalogue() if s.id == "observability-agent")
        assert step.system is True
        assert isinstance(step, obs.ObservabilityAgent)

    def test_the_module_exposes_exactly_one_step(self):
        # catalogue() discovers steps by scanning vars(module) for Step
        # subclasses; importing another one here would register it twice.
        found = [o for o in vars(obs).values()
                 if isinstance(o, type) and issubclass(o, steps_mod.Step) and o is not steps_mod.Step]
        assert found == [obs.ObservabilityAgent]

    def test_an_unwanted_step_is_not_selected_by_a_run_that_skips_it(self, linux_facts, empty_repo, tmp_path):
        ctx = make_ctx(linux_facts, empty_repo, tmp_path)
        ctx.plan["steps"]["skip"] = ["observability-agent"]
        assert "observability-agent" not in {s.id for s in steps_mod.selected(ctx)}


# ==========================================================================
# 2. schema and validate()
# ==========================================================================
def obs_problems(linux_facts, empty_repo, **section) -> list:
    plan, _ = planner.build(linux_facts, repo=empty_repo,
                            overrides={"observability": section})
    return [p for p in planner.validate(plan) if "observability" in p]


class TestSchema:
    def test_fleet_yml_with_the_section_parses_and_validates(self, linux_facts, empty_repo):
        (empty_repo / "fleet.yml").write_text(
            "observability:\n"
            "  enabled: true\n"
            f"  agent_version: {VERSION}\n"
            "  agent_sha512:\n"
            f"    - {PIN}\n"
            "  journald_units: [llm-gateway, ollama]\n"
            "  metrics_period_s: 30\n"
            "  agent_mem_high: 768M\n",
            encoding="utf-8")
        plan, prov = planner.build(linux_facts, repo=empty_repo)
        obs_plan = plan["observability"]
        assert obs_plan["enabled"] is True
        assert obs_plan["agent_version"] == VERSION
        assert obs_plan["agent_sha512"] == [PIN]
        assert obs_plan["journald_units"] == ["llm-gateway", "ollama"]
        assert prov["observability.enabled"].startswith("site")
        assert [p for p in planner.validate(plan) if "observability" in p] == []

    @pytest.mark.parametrize("key,value", [
        ("agent_sha512", [PIN + "\n"]),
        ("agent_version", VERSION + "\n"),
        ("agent_mem_max", "1G\n"),
        ("agent_cpu_quota", "50%\n"),
        ("journald_units", ["llm-gateway\n"]),
    ])
    def test_a_trailing_newline_is_not_accepted(self, linux_facts, empty_repo, key, value):
        plan, _ = planner.build(linux_facts, repo=empty_repo,
                                overrides={"observability": {"enabled": True, key: value}})
        assert [p for p in planner.validate(plan) if f"observability.{key}" in p]

    def test_a_host_yml_list_replaces_the_fleets(self, linux_facts, empty_repo):
        (empty_repo / "fleet.yml").write_text(
            "observability:\n  enabled: true\n  journald_units: [llm-gateway, ollama]\n",
            encoding="utf-8")
        host = empty_repo / "hosts" / "testbox"
        host.mkdir(parents=True)
        (host / "host.yml").write_text(
            "observability:\n  journald_units:\n    - extra-unit\n", encoding="utf-8")
        plan, prov = planner.build(linux_facts, repo=empty_repo)
        assert plan["observability"]["journald_units"] == ["extra-unit"]
        assert plan["observability"]["enabled"] is True
        assert prov["observability.journald_units"].startswith("host.yml")

    def test_an_unknown_key_raises(self, linux_facts, empty_repo):
        with pytest.raises(planner.PlanError, match="unknown key observability.*foo"):
            planner.build(linux_facts, repo=empty_repo,
                          overrides={"observability": {"foo": 1}})

    def test_an_unknown_key_in_fleet_yml_raises(self, linux_facts, empty_repo):
        (empty_repo / "fleet.yml").write_text("observability:\n  enabled: true\n  agent_verison: 9.5.6\n",
                                              encoding="utf-8")
        with pytest.raises(planner.PlanError, match="agent_verison"):
            planner.build(linux_facts, repo=empty_repo)

    def test_no_section_is_all_blank(self, linux_facts, empty_repo):
        plan, _ = planner.build(linux_facts, repo=empty_repo)
        assert set(plan["observability"].values()) == {None}
        assert planner.validate(plan) == []

    def test_a_wrong_type_is_reported_once_by_the_schema(self, linux_facts, empty_repo):
        problems = obs_problems(linux_facts, empty_repo, enabled=True, metrics_period_s="fast")
        assert len(problems) == 1
        assert "metrics_period_s" in problems[0]

    def test_a_good_block_has_no_problems(self, linux_facts, empty_repo):
        assert obs_problems(linux_facts, empty_repo, enabled=True, agent_version=VERSION,
                            agent_sha512=[PIN], secrets_file="/etc/fleet-obs/agent-output.env",
                            journald_units=["llm-gateway", "a@b", "x_y.service"],
                            metrics_period_s=10, health_interval_s=10, gpu_interval_s=30,
                            status_interval_s=120, usage_interval_s=60, fleet_interval_s=120,
                            process_top_n=20, agent_mem_high="768M", agent_mem_max="2G",
                            agent_cpu_quota="50%") == []

    BAD = [
        ("agent_version", "9.2.9"),
        ("agent_version", "9.5"),
        ("agent_version", "latest"),
        ("agent_version", "9.5.6:rc"),
        ("agent_sha512", [f"{VERSION}/{ASSET}=abc"]),
        ("agent_sha512", ["not-a-pin"]),
        ("agent_sha512", [f"{VERSION}/{ASSET}=" + "AB" * 64]),
        ("agent_sha512", [f"{VERSION}={'ab' * 64}"]),
        ("agent_sha512", [PIN, f"9.5.5/{ASSET}=short"]),
        ("secrets_file", "relative/agent.env"),
        ("journald_units", ["a:b"]),
        ("journald_units", ["ok", "has space"]),
        ("journald_units", ["with/slash"]),
        ("metrics_period_s", 9),
        ("health_interval_s", 9),
        ("gpu_interval_s", 29),
        ("status_interval_s", 119),
        ("usage_interval_s", 59),
        ("fleet_interval_s", 119),
        ("process_top_n", 0),
        ("process_top_n", 21),
        ("agent_mem_high", "lots"),
        ("agent_mem_high", "768"),
        ("agent_mem_max", "1 GB"),
        ("agent_cpu_quota", "50"),
        ("agent_cpu_quota", "half"),
    ]

    @pytest.mark.parametrize("key,value", BAD, ids=[f"{k}={str(v)[:24]}" for k, v in BAD])
    def test_each_bad_value_is_rejected_when_enabled(self, linux_facts, empty_repo, key, value):
        problems = obs_problems(linux_facts, empty_repo, enabled=True, **{key: value})
        assert problems, f"{key}={value!r} was accepted"
        assert all(key in p for p in problems)

    @pytest.mark.parametrize("key,value", BAD, ids=[f"{k}={str(v)[:24]}" for k, v in BAD])
    def test_the_same_values_are_not_my_business_when_disabled(self, linux_facts, empty_repo, key, value):
        # A box that never asked for observability is not made to answer for a
        # typo in a block it is not using.
        assert obs_problems(linux_facts, empty_repo, enabled=False, **{key: value}) == []

    def test_a_problem_never_quotes_a_long_value_whole(self, linux_facts, empty_repo):
        problems = obs_problems(linux_facts, empty_repo, enabled=True,
                                agent_sha512=["z" * 400])
        assert problems
        assert "z" * 100 not in problems[0]

    def test_the_agent_floor_is_the_first_release_with_a_token_file(self):
        assert planner.OBSERVABILITY_MIN_AGENT == (9, 3, 0)

    def test_the_pinned_default_is_itself_valid(self, linux_facts, empty_repo):
        # DEFAULTS applies wherever the plan says None; a default that failed
        # its own cross-check would make `enabled: true` alone unusable.
        assert obs_problems(linux_facts, empty_repo, enabled=True,
                            agent_version=obs.DEFAULTS["agent_version"]) == []
        assert obs_problems(linux_facts, empty_repo, enabled=True,
                            secrets_file=obs.DEFAULTS["secrets_file"],
                            journald_units=obs.DEFAULTS["journald_units"]) == []

    def test_the_defaults_respect_the_floors(self):
        for key, floor in planner.OBSERVABILITY_FLOORS.items():
            assert obs.DEFAULTS[key] >= floor, key


# ==========================================================================
# 3. the renderer
# ==========================================================================
class TestRenderedPolicy:
    def test_the_ca_fingerprint_appears_exactly_once(self):
        text = render()
        assert text.count(FP) == 1
        policy = parse_yaml(text)
        assert policy["outputs"]["default"]["ssl"]["ca_trusted_fingerprint"] == FP

    def test_the_fingerprint_is_written_as_supplied(self):
        mixed = "AB" * 16 + "cd" * 16
        text = render(secrets=secrets_dict(ES_CA_FINGERPRINT=mixed))
        assert text.count(mixed) == 1

    def test_the_api_key_is_id_colon_secret(self):
        policy = parse_yaml(render())
        out = policy["outputs"]["default"]
        assert out["api_key"] == KEY
        assert out["type"] == "elasticsearch"
        assert out["hosts"] == [HUB_URL]

    def test_several_hosts_become_a_list(self):
        text = render(secrets=secrets_dict(ES_HOSTS=f"{HUB_URL}, https://100.64.0.10:9200"))
        assert parse_yaml(text)["outputs"]["default"]["hosts"] == [HUB_URL, "https://100.64.0.10:9200"]

    def test_the_admin_token_is_never_in_the_policy_only_its_path(self):
        for role in ("hub", "peer"):
            policy = parse_yaml(render(role=role))
            authed = [s for s in streams_of(policy) if "auth" in s]
            assert authed, "no stream authenticates; the admin endpoints would 401"
            for s in authed:
                assert s["auth"] == {"file": {"path": obs.TOKEN_FILE, "prefix": "Bearer "}}
        text = render(role="hub")
        assert "Authorization" not in text
        assert obs.TOKEN_FILE in text

    def test_the_health_stream_is_the_one_that_does_not_authenticate(self):
        policy = parse_yaml(render())
        assert "auth" not in stream_by_id(policy, "llmstack-health")
        assert "auth" in stream_by_id(policy, "llmstack-gpu")

    def test_hub_only_inputs(self):
        hub = datasets_of(parse_yaml(render(role="hub")))
        peer = datasets_of(parse_yaml(render(role="peer")))
        hub_only = {"llmstack.fleet_summary", "llmstack.fleet_overview", "llmstack.peers"}
        assert hub_only <= hub
        assert not hub_only & peer
        assert peer < hub

    @pytest.mark.parametrize("role,section,expect", [
        ("hub", None, True),
        ("peer", None, False),
        ("hub", {"collect_status": False}, False),
        ("peer", {"collect_status": True}, True),
    ])
    def test_status_stream_is_gated_by_collect_status(self, role, section, expect):
        policy = parse_yaml(render(role=role, section=section))
        assert ("llmstack.status" in datasets_of(policy)) is expect

    def test_polls_always_go_to_loopback_and_never_send_the_routed_header(self):
        text = render(role="hub")
        policy = parse_yaml(text)
        urls = [s["request"]["url"] for s in streams_of(policy) if "request" in s]
        assert urls and all(u.startswith("http://127.0.0.1:8080/") for u in urls)
        assert "x-fleet-routed" not in text.lower()

    def test_the_port_comes_from_the_plan(self):
        cfg = obs._merged(None, "peer")
        text = obs.render_policy(cfg, secrets_dict(), port=9123, role="peer")
        assert "http://127.0.0.1:9123/health" in text

    def test_unit_list_appends_service_dedupes_and_keeps_order(self):
        section = {"journald_units": ["zeta", "alpha.service", "zeta", "tmpl@1", "alpha"]}
        policy = parse_yaml(render(section=section))
        journal = input_by_id(policy, "llmstack-journald")["streams"][0]
        assert journal["include_matches"]["match"] == [
            "_SYSTEMD_UNIT=zeta.service", "_SYSTEMD_UNIT=alpha.service", "_SYSTEMD_UNIT=tmpl@1.service"]
        assert ".service.service" not in render(section=section)

    def test_the_default_unit_list_has_no_hub_only_names(self):
        # The hub's extra units live in its own host.yml: this file is exported.
        assert obs.DEFAULTS["journald_units"] == [
            "llm-gateway", "llama-swap", "ollama", "cloudflared", "tailscaled"]
        assert obs._units(obs._merged(None, "hub")) == obs._units(obs._merged(None, "peer"))

    def test_an_empty_unit_list_drops_the_journald_input(self):
        policy = parse_yaml(render(section={"journald_units": []}))
        assert "llmstack.journal" not in datasets_of(policy)

    def test_journald_needs_linux_and_systemd(self):
        cfg = obs._merged(None, "peer")
        for kw in ({"family": "darwin"}, {"service": "cron"}):
            text = obs.render_policy(cfg, secrets_dict(), port=8080, **kw)
            assert "llmstack.journal" not in datasets_of(parse_yaml(text))

    def test_the_process_stream_drops_command_lines(self):
        policy = parse_yaml(render())
        stream = stream_by_id(policy, "llmstack-system-process")
        drop = stream["processors"][0]["drop_fields"]
        assert set(drop["fields"]) == {"process.args", "process.command_line",
                                       "process.args_count", "system.process.cmdline"}
        assert drop["ignore_missing"] is True
        assert stream["process"]["cmdline"]["cache"]["enabled"] is False

    def test_the_process_stream_keeps_the_top_n(self):
        policy = parse_yaml(render(section={"process_top_n": 7}))
        top = stream_by_id(policy, "llmstack-system-process")["process"]["include_top_n"]
        assert top == {"by_cpu": 7, "by_memory": 7}

    def test_secret_like_lines_and_the_agents_own_polls_are_dropped(self):
        policy = parse_yaml(render())
        journal = input_by_id(policy, "llmstack-journald")["streams"][0]
        drops = [p["drop_event"]["when"] for p in journal["processors"]]
        assert len(drops) == 2
        poll = next(w for w in drops if "and" in w)
        assert {"equals": {"systemd.unit": "llm-gateway.service"}} in poll["and"]
        secret = next(w for w in drops if "regexp" in w)["regexp"]["message"]
        pattern = re.compile(secret)
        assert pattern.search("Authorization: Basic abc")
        assert pattern.search("curl -H 'Bearer " + "x" * 24 + "'")
        assert pattern.search('api_key="' + "k" * 20)
        assert not pattern.search("llama-swap: loaded model qwen in 2.1s")

    def test_the_agents_own_access_lines_pattern_matches_what_uvicorn_writes(self):
        policy = parse_yaml(render())
        journal = input_by_id(policy, "llmstack-journald")["streams"][0]
        poll = next(p["drop_event"]["when"] for p in journal["processors"] if "and" in p["drop_event"]["when"])
        pattern = re.compile(next(c for c in poll["and"] if "regexp" in c)["regexp"]["message"])
        assert pattern.search('127.0.0.1:41234 - "GET /health HTTP/1.1" 200')
        assert pattern.search('127.0.0.1:41234 - "GET /admin/api/status HTTP/1.1" 200')
        assert not pattern.search('100.64.0.9:41234 - "GET /health HTTP/1.1" 200')
        assert not pattern.search('127.0.0.1:41234 - "POST /v1/chat/completions HTTP/1.1" 200')

    def test_dataset_names(self):
        sets = datasets_of(parse_yaml(render(role="hub")))
        assert "system.process.summary" in sets
        assert "system.process_summary" not in sets
        assert {"system.cpu", "system.memory", "system.load", "system.network", "system.diskio",
                "system.filesystem", "system.fsstat", "system.uptime", "system.process",
                "system.socket_summary", "llmstack.health", "llmstack.gpu", "llmstack.usage",
                "llmstack.journal"} <= sets

    def test_stream_ids_are_unique(self):
        ids = [s["id"] for s in streams_of(parse_yaml(render(role="hub")))]
        assert len(ids) == len(set(ids))
        inputs = [i["id"] for i in parse_yaml(render(role="hub"))["inputs"]]
        assert len(inputs) == len(set(inputs))

    def test_the_system_inputs_filesystem_streams_drop_pseudo_mounts(self):
        policy = parse_yaml(render())
        for sid, field in (("llmstack-system-filesystem", "system.filesystem.mount_point"),
                           ("llmstack-system-fsstat", "system.fsstat.mount_point")):
            rx = stream_by_id(policy, sid)["processors"][0]["drop_event"]["when"]["regexp"][field]
            assert "$" not in rx
            pattern = re.compile(rx.replace(r"\z", r"\Z"))
            for mount in ("/proc", "/sys/fs/cgroup", "/dev/shm", "/snap/core/1", "/etc", "/host/proc"):
                assert pattern.search(mount), mount
            for mount in ("/", "/var/lib/llmstack", "/home", "/procurement", "/system", "/devices"):
                assert not pattern.search(mount), mount

    def test_no_forbidden_keys_or_variable_syntax(self):
        # Each of these cost somebody an afternoon: config_version is not a
        # key in 9.6's httpjson, max_object_depth exists only there, the routed
        # header flips the box's standby state, and a bare `$` is one character
        # from a variable substitution that rewrites a key.
        for role in ("hub", "peer"):
            text = render(role=role)
            assert "config_version" not in text
            assert "max_object_depth" not in text
            assert "x-fleet-routed" not in text.lower()
            assert "$" not in text

    def test_the_agent_does_not_ship_its_own_monitoring(self):
        policy = parse_yaml(render())
        assert policy["agent"]["monitoring"]["enabled"] is False
        assert policy["agent"]["logging"]["metrics"]["enabled"] is False

    def test_each_poll_decodes_json_into_a_non_empty_target(self):
        policy = parse_yaml(render(role="hub"))
        for s in streams_of(policy):
            if "request" not in s:
                continue
            assert s["response"] == {"decode_as": "application/json"}
            decode = s["processors"][0]["decode_json_fields"]
            assert decode["target"].startswith("llmstack.")
            assert decode["target"] == "llmstack." + s["data_stream"]["dataset"].split(".", 1)[1]

    def test_poll_intervals_come_from_the_config(self):
        policy = parse_yaml(render(role="hub", section={
            "health_interval_s": 11, "gpu_interval_s": 31, "usage_interval_s": 61,
            "status_interval_s": 121, "fleet_interval_s": 122, "metrics_period_s": 12}))
        assert stream_by_id(policy, "llmstack-health")["interval"] == "11s"
        assert stream_by_id(policy, "llmstack-gpu")["interval"] == "31s"
        assert stream_by_id(policy, "llmstack-usage")["interval"] == "61s"
        assert stream_by_id(policy, "llmstack-status")["interval"] == "121s"
        assert stream_by_id(policy, "llmstack-peers")["interval"] == "122s"
        assert stream_by_id(policy, "llmstack-system-cpu")["period"] == "12s"
        assert stream_by_id(policy, "llmstack-system-process")["period"] == f"{obs.SLOW_PERIOD_S}s"

    def test_the_usage_poll_asks_for_counts_not_requests(self):
        policy = parse_yaml(render())
        assert stream_by_id(policy, "llmstack-usage")["request"]["url"].endswith(
            "/admin/api/usage?hours=1&recent_limit=0")

    # -- the gateway's request log -----------------------------------------
    def test_the_request_log_is_tailed_when_a_path_is_given(self):
        path = shapes.access_log_path("linux", "/var/lib/llmstack")
        assert path == "/var/lib/llmstack/access.ndjson"
        policy = parse_yaml(render(access_log=path))
        inp = input_by_id(policy, "llmstack-gateway-requests")
        assert inp["type"] == "filestream"
        stream = inp["streams"][0]
        # the rotated backups too, so a rename is a drain and not a removal
        assert stream["paths"] == [path, path + ".*"]
        assert stream["data_stream"] == {"dataset": "llmstack.requests", "type": "logs"}
        assert "ndjson" in stream["parsers"][0]
        assert stream["close"]["on_state_change"]["inactive"] == "5m"

    def test_the_request_log_fingerprints_small_files(self):
        stream = input_by_id(parse_yaml(render(access_log="/var/lib/x/access.ndjson")),
                             "llmstack-gateway-requests")["streams"][0]
        fp = stream["prospector"]["scanner"]["fingerprint"]
        assert fp["enabled"] is True
        assert fp["length"] == 64

    def test_the_request_log_routing_is_reasserted_after_the_parser(self):
        stream = input_by_id(parse_yaml(render(access_log="/var/lib/x/access.ndjson")),
                             "llmstack-gateway-requests")["streams"][0]
        adds = [p["add_fields"] for p in stream["processors"]]
        assert {"target": "data_stream", "fields": {
            "type": "logs", "dataset": "llmstack.requests", "namespace": "default"}} in adds
        assert {"target": "event", "fields": {"dataset": "llmstack.requests"}} in adds

    def test_fields_that_must_not_leave_the_box_are_dropped(self):
        policy = parse_yaml(render(role="hub"))
        dropped = lambda sid: next(  # noqa: E731
            p["drop_fields"]["fields"] for p in stream_by_id(policy, sid)["processors"] if "drop_fields" in p)
        assert "llmstack.status.models_private" in dropped("llmstack-status")
        assert "llmstack.usage.recent" in dropped("llmstack-usage")
        assert dropped("llmstack-gpu") == ["message"]

    def test_no_request_log_input_without_a_path(self):
        assert "llmstack.requests" not in datasets_of(parse_yaml(render()))

    def test_the_env_file_and_the_policy_agree_on_the_log_path(self, linux_facts, empty_repo, tmp_path):
        r = Rig(linux_facts, empty_repo, tmp_path)
        env_path = EnvFile()._values(r.ctx)["LLMSTACK_ACCESS_LOG"]
        policy = parse_yaml(r.expected_policy())
        assert input_by_id(policy, "llmstack-gateway-requests")["streams"][0]["paths"][0] == env_path

    # -- the emitter --------------------------------------------------------
    @pytest.mark.parametrize("role", ["hub", "peer"])
    def test_emitter_roundtrip(self, monkeypatch, role):
        seen: list = []
        real = obs._emit

        def spy(obj, indent=0):
            seen.append(obj)
            return real(obj, indent)

        monkeypatch.setattr(obs, "_emit", spy)
        text = render(role=role, access_log="/var/lib/llmstack/access.ndjson")
        assert parse_yaml(text) == seen[0]

    def test_emitter_roundtrip_through_pyyaml_when_it_is_installed(self, monkeypatch):
        yaml = pytest.importorskip("yaml")
        seen: list = []
        real = obs._emit

        def spy(obj, indent=0):
            seen.append(obj)
            return real(obj, indent)

        monkeypatch.setattr(obs, "_emit", spy)
        text = render(role="hub", access_log="/var/lib/llmstack/access.ndjson")
        assert yaml.safe_load(text) == seen[0]

    def test_a_yaml_word_or_odd_key_is_quoted(self):
        assert obs._key("plain_key") == "plain_key"
        assert obs._key("yes") == '"yes"'
        assert obs._key("on") == '"on"'
        assert obs._key("a.b") == '"a.b"'
        assert obs._key("x-fleet") == '"x-fleet"'

    def test_scalars_are_json(self):
        assert obs._scalar(None) == "null"
        assert obs._scalar(True) == "true"
        assert obs._scalar(False) == "false"
        assert obs._scalar(30) == "30"
        assert obs._scalar('a "b" \\z') == '"a \\"b\\" \\\\z"'
        with pytest.raises(TypeError):
            obs._scalar(object())

    def test_a_nested_list_is_refused_rather_than_misemitted(self):
        with pytest.raises(TypeError):
            obs._emit({"a": [[1, 2]]})

    def test_empty_containers_are_inline(self):
        assert obs._emit({"a": {}, "b": [], "c": [{}]}) == ["a: {}", "b: []", "c:", "  - {}"]

    def test_the_policy_is_deterministic(self):
        assert render(role="hub") == render(role="hub")

    def test_the_policy_says_it_is_managed(self):
        assert render().startswith("# managed by fleetctl observability-agent.")


class TestSmallHelpers:
    def test_merge_treats_none_as_unstated_but_false_and_empty_as_real(self):
        cfg = obs._merged({"enabled": True, "allow_unpinned": False, "journald_units": [],
                           "metrics_period_s": None, "collect_status": False}, "hub")
        assert cfg["metrics_period_s"] == obs.DEFAULTS["metrics_period_s"]
        assert cfg["journald_units"] == []
        assert cfg["collect_status"] is False
        assert cfg["allow_unpinned"] is False

    def test_merge_does_not_share_the_defaults_lists(self):
        cfg = obs._merged(None, "peer")
        cfg["journald_units"].append("x")
        cfg["agent_sha512"].append("y")
        assert "x" not in obs.DEFAULTS["journald_units"]
        assert obs.DEFAULTS["agent_sha512"] == []

    def test_pins_are_keyed_and_lowercased(self):
        pins = obs._pins({"agent_sha512": [f"{VERSION}/{ASSET}=" + "AB" * 64, "garbage-without-equals"]})
        assert pins[f"{VERSION}/{ASSET}"] == "ab" * 64
        assert pins["garbage-without-equals"] == ""

    def test_secrets_env_reading(self):
        text = ("# a comment\n\nES_HOSTS=https://100.64.0.9:9200\nexport ES_API_KEY='kid1234:sec5678'\n"
                'ES_CA_FINGERPRINT="' + FP + '"\nnot a line\nEMPTY=\n')
        got = obs._read_secrets_env(text)
        assert got == {"ES_HOSTS": "https://100.64.0.9:9200", "ES_API_KEY": KEY,
                       "ES_CA_FINGERPRINT": FP, "EMPTY": ""}
        assert obs._read_secrets_env("") == {}

    def test_scrub_removes_longest_first_and_ignores_trivial_values(self):
        out = obs._scrub(f"failed with {KEY} and later sec5678 and {TOKEN}", [KEY, "sec5678", TOKEN, "ab", ""])
        no_secret_in(out)
        assert out.count("<redacted>") == 3
        assert obs._scrub("abc", ["ab"]) == "abc"

    def test_hidden_covers_each_half_of_the_key(self):
        assert set(obs._hidden({"ES_API_KEY": KEY}, TOKEN)) == {TOKEN, KEY, "kid1234", "sec5678"}

    def test_dropin_text(self):
        text = obs._dropin(obs._merged(None, "peer"))
        assert "[Service]" in text
        assert "MemoryHigh=768M" in text and "MemoryMax=1G" in text
        assert "CPUQuota=50%" in text and "Nice=10" in text
        text = obs._dropin(obs._merged({"agent_mem_high": "512M", "agent_cpu_quota": "25%"}, "peer"))
        assert "MemoryHigh=512M" in text and "CPUQuota=25%" in text

    def test_expiry_kinds(self, monkeypatch):
        assert obs._expiry({}) == ("none", None)
        assert obs._expiry({"ES_KEY_EXPIRES": "tomorrow"}) == ("malformed", None)
        assert obs._expiry({"ES_KEY_EXPIRES": "2026-10-31T00:00:00Z"})[0] == "expired"
        assert obs._expiry({"ES_KEY_EXPIRES": "2026-11-11T12:00:00Z"}) == ("ok", 10)


# ==========================================================================
# 4. check()
# ==========================================================================
class TestCheckBlocked:
    def test_missing_api_key(self, build_rig):
        r = build_rig()
        r.seed(secrets_text(ES_API_KEY=None))
        res = r.check()
        assert res.state == BLOCKED
        assert "ES_API_KEY" in res.detail
        assert r.cfg["secrets_file"] in res.detail

    @pytest.mark.parametrize("missing", ["ES_HOSTS", "ES_CA_FINGERPRINT"])
    def test_other_missing_values(self, build_rig, missing):
        r = build_rig()
        r.seed(secrets_text(**{missing: None}))
        res = r.check()
        assert res.state == BLOCKED
        assert missing in res.detail

    def test_the_secrets_file_absent(self, build_rig):
        r = build_rig()
        r.ctx.path(r.cfg["secrets_file"]).unlink()
        res = r.check()
        assert res.state == BLOCKED
        assert r.cfg["secrets_file"] in res.detail

    def test_the_secrets_file_unreadable(self, build_rig):
        r = build_rig()
        real = r.ctx.read_state
        r.ctx.read_state = lambda p: ((None, "unreadable: Permission denied")
                                      if str(p) == r.cfg["secrets_file"] else real(p))
        res = r.check()
        assert res.state == BLOCKED
        assert "re-run with sudo" in res.detail

    def test_expired_key(self, build_rig):
        r = build_rig()
        r.seed(secrets_text(ES_KEY_EXPIRES="2026-10-31T23:59:59Z"))
        res = r.check()
        assert res.state == BLOCKED
        assert "expired" in res.detail

    def test_malformed_expiry(self, build_rig):
        r = build_rig()
        r.seed(secrets_text(ES_KEY_EXPIRES="next tuesday"))
        assert r.check().state == BLOCKED

    @pytest.mark.parametrize("bad", ["kid1234", "kid1234:sec:5678", "kid$1234:sec5678", "a:b$c",
                                     "kid1234:", ":sec5678", "dGhpcyBpcyBiYXNlNjQ=", "kid 1234:sec5678"])
    def test_malformed_key(self, build_rig, bad):
        r = build_rig()
        r.seed(secrets_text(ES_API_KEY=bad))
        res = r.check()
        assert res.state == BLOCKED
        assert "ES_API_KEY" in res.detail
        assert bad not in res.detail or bad == "kid1234"

    @pytest.mark.parametrize("bad", ["ab" * 31, "zz" * 32, "ab" * 33, "", "ab:" * 21 + "ab"])
    def test_bad_fingerprint(self, build_rig, bad):
        r = build_rig()
        r.seed(secrets_text(ES_CA_FINGERPRINT=bad))
        assert r.check().state == BLOCKED

    @pytest.mark.parametrize("bad", ["http://100.64.0.9:9200", "100.64.0.9:9200", "https://a b:9200",
                                     "ftp://100.64.0.9", "https://"])
    def test_bad_hosts(self, build_rig, bad):
        r = build_rig()
        r.seed(secrets_text(ES_HOSTS=bad))
        res = r.check()
        assert res.state == BLOCKED
        assert "ES_HOSTS" in res.detail

    def test_no_pin_blocks_unless_allowed(self, build_rig):
        r = build_rig(section={"enabled": True})
        res = r.check()
        assert res.state == BLOCKED
        assert f"{VERSION}/{ASSET}" in res.detail
        assert "agent_sha512" in res.detail
        r = build_rig(section={"enabled": True, "allow_unpinned": True})
        assert r.check().state == MISSING

    def test_a_pin_for_another_version_does_not_count(self, build_rig):
        r = build_rig(section={"enabled": True, "agent_version": "9.5.7", "agent_sha512": [PIN]})
        res = r.check()
        assert res.state == BLOCKED
        assert "9.5.7/" in res.detail

    def test_a_pin_for_another_asset_does_not_count(self, build_rig):
        r = build_rig(section={"enabled": True, "agent_sha512": [f"{VERSION}/linux-arm64.tar.gz=" + "cd" * 64]})
        assert r.check().state == BLOCKED

    def test_unreadable_gateway_env(self, build_rig):
        r = build_rig()
        envfile = r.ctx.plan["paths"]["envfile"]
        real = r.ctx.read_state
        r.ctx.read_state = lambda p: ((None, "unreadable: Permission denied") if str(p) == envfile else real(p))
        res = r.check()
        assert res.state == BLOCKED
        assert "re-run with sudo" in res.detail

    def test_missing_env_on_a_real_run_is_blocked_with_the_way_out(self, build_rig):
        r = build_rig()
        r.ctx.path(r.ctx.plan["paths"]["envfile"]).unlink()
        res = r.check()
        assert res.state == BLOCKED
        assert "run fleetctl apply first" in res.detail

    def test_an_env_file_without_a_token_is_the_same(self, build_rig):
        r = build_rig()
        put(r.ctx, r.ctx.plan["paths"]["envfile"], "LLMSTACK_BIND=0.0.0.0\nLLMSTACK_ADMIN_TOKEN=\n")
        res = r.check()
        assert res.state == BLOCKED
        assert "run fleetctl apply first" in res.detail

    def test_dry_run_missing_env_is_missing_not_blocked(self, build_rig, monkeypatch):
        monkeypatch.setattr(obs, "_is_root", lambda: False)
        r = build_rig(dry_run=True)
        r.ctx.path(r.ctx.plan["paths"]["envfile"]).unlink()
        res = r.check()
        assert res.state == MISSING
        assert any("written by the envfile step" in line for line in res.plan)
        assert any(f"install elastic-agent {VERSION}" in line for line in res.plan)

    def test_dry_run_missing_env_and_missing_secrets_says_so_in_the_plan(self, build_rig):
        r = build_rig(dry_run=True, seed=False)
        res = r.check()
        assert res.state == MISSING
        assert any("carry the ingest-key file" in line for line in res.plan)

    def test_not_root_is_blocked(self, build_rig, monkeypatch):
        monkeypatch.setattr(obs, "_is_root", lambda: False)
        r = build_rig()
        res = r.check()
        assert res.state == BLOCKED
        assert "needs root" in res.detail
        assert "re-run with sudo" in res.detail

    def test_passwordless_sudo_is_not_root(self, build_rig, monkeypatch):
        # Ctx.root_ok accepts sudo_nopasswd, which would let a non-root run
        # through to a PermissionError after the download.
        monkeypatch.setattr(obs, "_is_root", lambda: False)
        r = build_rig()
        r.ctx.facts["privilege"] = {"root": False, "sudo_nopasswd": True}
        assert r.check().state == BLOCKED

    def test_a_dry_run_does_not_need_root(self, build_rig, monkeypatch):
        monkeypatch.setattr(obs, "_is_root", lambda: False)
        r = build_rig(dry_run=True)
        assert r.check().state == MISSING

    @pytest.mark.parametrize("arch", ["riscv64", "armv7l", "arm64", "", None])
    def test_unsupported_arch(self, build_rig, arch):
        r = build_rig()
        r.ctx.plan["platform"]["arch"] = arch
        res = r.check()
        assert res.state == BLOCKED
        assert "unsupported arch" in res.detail

    def test_aarch64_is_supported(self, build_rig):
        r = build_rig(section={"enabled": True, "agent_sha512": [f"{VERSION}/linux-arm64.tar.gz=" + "cd" * 64]})
        r.ctx.plan["platform"]["arch"] = "aarch64"
        assert r.check().state == MISSING

    @pytest.mark.parametrize("which", ["windows", "darwin"])
    def test_windows_and_darwin_are_blocked_with_the_way_out(self, request, empty_repo, tmp_path, which):
        facts = request.getfixturevalue(f"{which}_facts")
        plan, _ = planner.build(facts, repo=empty_repo, overrides={
            "observability": {"enabled": True, "agent_sha512": [PIN]}})
        ctx = Ctx(plan, facts, repo=empty_repo, root=str(tmp_path / "sandbox"))
        res = obs.ObservabilityAgent().check(ctx)
        assert res.state == BLOCKED
        assert "linux-systemd only" in res.detail
        assert "observability.enabled: false" in res.detail
        assert not (tmp_path / "sandbox").exists()

    def test_linux_without_systemd_is_blocked(self, build_rig):
        r = build_rig()
        r.ctx.plan["platform"]["service"] = "openrc"
        res = r.check()
        assert res.state == BLOCKED
        assert "no systemd" in res.detail

    def test_check_never_raises(self, build_rig, monkeypatch):
        r = build_rig()

        def boom(ctx):
            raise ValueError("a bug somewhere in a helper")

        monkeypatch.setattr(obs, "_cfg", boom)
        res = r.check()
        assert res.state == BLOCKED
        assert "ValueError" in res.detail

    # -- the trust boundary --------------------------------------------------
    @POSIX
    def test_a_group_writable_secrets_directory_is_blocked(self, build_rig):
        r = build_rig()
        r.ctx.path(obs.SECRETS_DIR).chmod(0o775)
        res = r.check()
        assert res.state == BLOCKED
        assert "writable by group or others" in res.detail

    @POSIX
    @pytest.mark.parametrize("mode", [0o644, 0o640, 0o604])
    def test_a_readable_secrets_file_is_blocked(self, build_rig, mode):
        r = build_rig()
        r.ctx.path(r.cfg["secrets_file"]).chmod(mode)
        res = r.check()
        assert res.state == BLOCKED
        assert "readable by group or others" in res.detail
        no_secret_in(res.detail)

    @POSIX
    def test_apply_puts_the_secrets_file_back_to_0600(self, build_rig, downloads):
        r = build_rig()
        r.ctx.path(r.cfg["secrets_file"]).chmod(0o644)
        r.apply()
        assert os.stat(r.ctx.path(r.cfg["secrets_file"])).st_mode & 0o777 == 0o600

    @POSIX
    def test_a_world_writable_secrets_directory_is_blocked(self, build_rig):
        r = build_rig()
        r.ctx.path(obs.SECRETS_DIR).chmod(0o757)
        assert r.check().state == BLOCKED

    @POSIX
    def test_a_group_writable_secrets_file_is_blocked(self, build_rig):
        r = build_rig()
        r.ctx.path(r.cfg["secrets_file"]).chmod(0o664)
        assert r.check().state == BLOCKED

    @POSIX
    def test_the_parent_of_a_relocated_secrets_file_is_checked_too(self, build_rig):
        r = build_rig(section={"enabled": True, "agent_sha512": [PIN], "secrets_file": "/srv/obs/agent.env"})
        r.ctx.path("/srv/obs").chmod(0o777)
        res = r.check()
        assert res.state == BLOCKED
        assert "/srv/obs" in res.detail

    @POSIX
    def test_a_symlinked_secrets_file_is_blocked(self, build_rig):
        r = build_rig()
        path = r.ctx.path(r.cfg["secrets_file"])
        real = path.with_name("elsewhere.env")
        path.rename(real)
        path.symlink_to(real)
        res = r.check()
        assert res.state == BLOCKED
        assert "symlink" in res.detail

    @POSIX
    def test_a_symlinked_token_file_is_blocked(self, build_rig):
        r = build_rig()
        r.settle()
        token = r.ctx.path(obs.TOKEN_FILE)
        other = token.with_name("other")
        token.rename(other)
        token.symlink_to(other)
        res = r.check()
        assert res.state == BLOCKED
        assert "symlink" in res.detail

    @POSIX
    def test_a_symlinked_policy_is_blocked(self, build_rig):
        r = build_rig()
        r.settle()
        policy = r.ctx.path(obs.AGENT_POLICY)
        other = policy.with_name("other.yml")
        policy.rename(other)
        policy.symlink_to(other)
        assert r.check().state == BLOCKED

    def test_trust_is_judged_before_content(self, build_rig):
        # With an untrusted directory the file is not even read: a bad key in
        # it must not be what the operator is told about.
        if os.name == "nt":
            pytest.skip("POSIX modes")
        r = build_rig()
        r.seed(secrets_text(ES_API_KEY="nonsense"))
        r.ctx.path(obs.SECRETS_DIR).chmod(0o777)
        assert "untrusted path" in r.check().detail


class TestCheckStates:
    def test_nothing_installed_is_missing_with_plan_lines(self, build_rig):
        r = build_rig()
        res = r.check()
        assert res.state == MISSING
        assert res.detail == "elastic-agent is not installed"
        assert f"install elastic-agent {VERSION}" in res.plan
        assert f"write {obs.AGENT_POLICY} (not written yet)" in res.plan
        assert f"write {obs.TOKEN_FILE}" in res.plan
        assert f"write {obs.DROPIN}" in res.plan

    def test_half_installed_is_missing(self, build_rig):
        r = build_rig()
        put(r.ctx, obs.AGENT_BIN, "#!/bin/sh\n", 0o755)
        assert r.fake.unit is False
        res = r.check()
        assert res.state == MISSING
        assert "half-installed" in res.detail
        assert f"{obs.AGENT_BIN} uninstall --force (half-installed)" in res.plan
        assert f"install elastic-agent {VERSION}" in res.plan

    @pytest.mark.parametrize("how", ["none", "rc"])
    def test_a_failed_systemctl_probe_is_not_a_half_install(self, build_rig, how):
        # A timeout read as "no unit" put `uninstall --force` in the plan of a
        # healthy box.
        r = build_rig()
        r.settle()
        if how == "none":
            r.fake.unit_probe_none = True
        else:
            r.fake.unit_probe_rc = 1
        res = r.check()
        assert res.state == BLOCKED
        assert not any("uninstall" in line for line in res.plan)

    def test_an_apply_will_not_guess_at_an_unreadable_unit(self, build_rig, downloads):
        r = build_rig()
        r.settle()
        r.fake.unit_probe_none = True
        with pytest.raises(RuntimeError, match="cannot tell"):
            r.apply()
        assert not any(a[1:2] == ["uninstall"] for a in r.fake.argvs)

    def test_the_binary_unreadable_is_blocked(self, build_rig):
        r = build_rig()
        real = r.ctx.exists_state
        r.ctx.exists_state = lambda p: ((None, "unreadable: Permission denied")
                                        if str(p) == obs.AGENT_BIN else real(p))
        res = r.check()
        assert res.state == BLOCKED
        assert "re-run with sudo" in res.detail

    def test_ok_when_everything_matches(self, build_rig):
        r = build_rig()
        r.settle()
        res = r.check()
        assert res.state == OK
        assert res.plan == []
        assert f"elastic-agent {VERSION}" in res.detail

    @pytest.mark.parametrize("active,sub", [("failed", "failed"), ("activating", "auto-restart"),
                                            ("failed", "auto-restart")])
    def test_a_failed_or_crash_looping_unit_is_blocked_not_drift(self, build_rig, active, sub):
        # DRIFT would send apply round a loop it cannot fix, ending the run as
        # UNCHANGED after an apply that "worked".
        r = build_rig()
        r.settle()
        r.fake.active, r.fake.sub = active, sub
        res = r.check()
        assert res.state == BLOCKED
        assert "journalctl -u elastic-agent" in res.detail

    def test_a_stopped_unit_is_drift(self, build_rig):
        r = build_rig()
        r.settle()
        r.fake.active, r.fake.sub = "inactive", "dead"
        res = r.check()
        assert res.state == DRIFT
        assert f"systemctl enable --now {obs.UNIT}" in res.plan

    def test_a_disabled_unit_is_drift(self, build_rig):
        r = build_rig()
        r.settle()
        r.fake.enabled = "disabled"
        res = r.check()
        assert res.state == DRIFT
        assert f"systemctl enable --now {obs.UNIT}" in res.plan

    @pytest.mark.parametrize("active,sub", [("active", "running"), ("activating", "start"),
                                            ("activating", "start-pre")])
    def test_starting_or_running_counts_as_running(self, build_rig, active, sub):
        r = build_rig()
        r.settle()
        r.fake.active, r.fake.sub = active, sub
        assert r.check().state == OK

    def test_a_differing_policy_is_drift_and_names_the_file(self, build_rig):
        r = build_rig()
        r.settle()
        put(r.ctx, obs.AGENT_POLICY, "outputs: {}\n", 0o600)
        res = r.check()
        assert res.state == DRIFT
        assert f"write {obs.AGENT_POLICY} (policy differs)" in res.plan

    def test_a_changed_interval_is_policy_drift(self, build_rig):
        r = build_rig()
        r.settle()
        r.ctx.plan["observability"]["gpu_interval_s"] = 45
        res = r.check()
        assert res.state == DRIFT
        assert any(obs.AGENT_POLICY in line for line in res.plan)

    def test_a_rotated_secrets_file_is_policy_drift(self, build_rig):
        r = build_rig()
        r.settle()
        r.seed(secrets_text(ES_API_KEY="kid9999:sec0000"))
        res = r.check()
        assert res.state == DRIFT
        assert any(obs.AGENT_POLICY in line for line in res.plan)
        no_secret_in(res.detail, res.plan)
        assert "sec0000" not in "".join(res.plan) + res.detail

    def test_token_rotation_is_drift(self, build_rig):
        r = build_rig()
        r.settle()
        put(r.ctx, r.ctx.plan["paths"]["envfile"], "LLMSTACK_ADMIN_TOKEN=rotated-token-value\n")
        res = r.check()
        assert res.state == DRIFT
        assert f"write {obs.TOKEN_FILE}" in res.plan
        assert "rotated-token-value" not in res.detail + "".join(res.plan)

    @POSIX
    def test_a_token_file_with_the_wrong_mode_is_drift(self, build_rig):
        r = build_rig()
        r.settle()
        r.ctx.path(obs.TOKEN_FILE).chmod(0o644)
        res = r.check()
        assert res.state == DRIFT
        assert f"write {obs.TOKEN_FILE}" in res.plan

    @POSIX
    def test_a_policy_wider_than_0600_is_drift_because_it_holds_the_key(self, build_rig):
        r = build_rig()
        r.settle()
        r.ctx.path(obs.AGENT_POLICY).chmod(0o644)
        res = r.check()
        assert res.state == DRIFT
        assert any("chmod 0600" in line for line in res.plan)

    def test_a_missing_or_changed_dropin_is_drift(self, build_rig):
        r = build_rig()
        r.settle()
        r.ctx.path(obs.DROPIN).unlink()
        res = r.check()
        assert res.state == DRIFT
        assert f"write {obs.DROPIN}" in res.plan
        put(r.ctx, obs.DROPIN, "[Service]\nMemoryMax=infinity\n")
        assert r.check().state == DRIFT

    def test_a_token_with_a_trailing_newline_is_drift(self, build_rig):
        # apply writes it without one; the agent's auth.file reads it verbatim.
        r = build_rig()
        r.settle()
        put(r.ctx, obs.TOKEN_FILE, TOKEN + "\n", 0o600)
        assert r.check().state == DRIFT

    def test_every_difference_is_listed(self, build_rig):
        r = build_rig()
        r.settle()
        put(r.ctx, obs.AGENT_POLICY, "x\n", 0o600)
        put(r.ctx, obs.TOKEN_FILE, "stale", 0o600)
        r.ctx.path(obs.DROPIN).unlink()
        r.fake.enabled = "disabled"
        res = r.check()
        assert res.state == DRIFT
        assert len(res.plan) == 4

    # -- the detail ----------------------------------------------------------
    def test_version_unknown_is_reported_honestly(self, build_rig):
        r = build_rig()
        r.settle()
        r.fake.version_out = None
        res = r.check()
        assert res.state == OK
        assert "version unknown" in res.detail
        assert VERSION not in res.detail

    def test_a_version_that_is_not_the_pin_is_said_but_is_not_a_state(self, build_rig):
        # install --non-interactive cannot upgrade, so DRIFT would loop into
        # UNCHANGED.
        r = build_rig()
        r.settle()
        r.fake.version_out = "Binary: 9.5.5, commit: 0abc"
        res = r.check()
        assert res.state == OK
        assert "9.5.5" in res.detail
        assert f"pinned {VERSION}" in res.detail

    def test_the_status_summary_is_appended_for_root(self, build_rig):
        r = build_rig()
        r.settle()
        r.fake.status_out = (0, json.dumps({"state": 2, "components": [
            {"id": "system/metrics-default", "state": 2}, {"id": "journald-default", "state": 2}]}))
        res = r.check()
        assert res.state == OK
        assert "2 component(s) healthy" in res.detail

    def test_a_degraded_component_does_not_change_the_state(self, build_rig):
        # A down gateway makes httpjson degraded and `status` exit 1; neither
        # is the agent's fault, and neither means "not installed".
        r = build_rig()
        r.settle()
        r.fake.status_out = (1, json.dumps({"state": 3, "components": [
            {"id": "system/metrics-default", "state": 2}, {"id": "httpjson-default", "state": 3}]}))
        res = r.check()
        assert res.state == OK
        assert "1/2 component(s) not healthy" in res.detail
        assert "httpjson-default" in res.detail

    @pytest.mark.parametrize("status", [
        {"state": 4, "components": [{"id": "x", "state": 4}]},
        {"state": 4, "components": []},
        {"state": 3, "components": [{"id": "x", "state": 4}]},
    ])
    def test_a_failed_agent_is_blocked_not_ok(self, build_rig, status):
        r = build_rig()
        r.settle()
        r.fake.status_out = (1, json.dumps(status))
        res = r.check()
        assert res.state == BLOCKED
        assert "journalctl -u elastic-agent" in res.detail

    def test_a_failed_agent_with_a_policy_to_write_is_drift(self, build_rig):
        r = build_rig()
        r.settle()
        put(r.ctx, obs.AGENT_POLICY, "stale\n", 0o600)
        r.fake.status_out = (1, json.dumps({"state": 4, "components": []}))
        assert r.check().state == DRIFT

    @pytest.mark.parametrize("raw", [(1, "not json at all"), (0, "[1, 2]"), (0, "")])
    def test_a_garbled_status_is_unavailable_not_an_error(self, build_rig, raw):
        r = build_rig()
        r.settle()
        r.fake.status_out = raw
        res = r.check()
        assert res.state == OK
        assert "status unavailable" in res.detail

    def test_status_is_not_asked_of_a_non_root_dry_run(self, build_rig, monkeypatch):
        monkeypatch.setattr(obs, "_is_root", lambda: False)
        r = build_rig(dry_run=True)
        r.settle()
        asked = []
        real = r.fake.probe
        r.fake.ctx.probe = lambda argv, timeout=60, env=None: (asked.append(list(argv)), real(argv, timeout, env))[1]
        assert r.check().state == OK
        assert not any(a[:2] == [obs.AGENT_BIN, "status"] for a in asked)

    def test_a_key_near_expiry_warns_and_a_far_one_does_not(self, build_rig):
        r = build_rig()
        r.settle()
        r.seed(secrets_text(ES_KEY_EXPIRES="2026-11-11T12:00:00Z"))
        res = r.check()
        assert res.state == OK
        assert "expires in 10 day(s)" in res.detail
        r.seed(secrets_text(ES_KEY_EXPIRES="2027-11-11T12:00:00Z"))
        assert "expires" not in r.check().detail

    def test_the_expiry_warning_is_on_a_missing_agent_too(self, build_rig):
        r = build_rig()
        r.seed(secrets_text(ES_KEY_EXPIRES="2026-11-04T12:00:00Z"))
        res = r.check()
        assert res.state == MISSING
        assert "expires in 3 day(s)" in res.detail

    # -- read-only -----------------------------------------------------------
    def test_check_changes_nothing(self, build_rig):
        r = build_rig()
        r.settle()
        put(r.ctx, obs.AGENT_POLICY, "stale\n", 0o644)
        before = snapshot(r.ctx.path("/"))
        r.check()
        assert snapshot(r.ctx.path("/")) == before
        assert r.fake.sudo_calls == [] and r.fake.run_calls == []

    def test_check_does_not_create_the_sandbox(self, linux_facts, empty_repo, tmp_path):
        ctx = make_ctx(linux_facts, empty_repo, tmp_path, dry_run=True)
        FakeBox(ctx)
        res = obs.ObservabilityAgent().check(ctx)
        assert res.state == MISSING
        assert not (tmp_path / "sandbox").exists()

    def test_check_never_touches_the_network(self, build_rig, monkeypatch):
        def refuse(*a, **kw):
            raise AssertionError("check() reached the network")

        monkeypatch.setattr(obs, "_urlopen", refuse)
        monkeypatch.setattr(obs, "_fetch", refuse)
        monkeypatch.setattr(urllib.request, "urlopen", refuse)
        r = build_rig()
        assert r.check().state == MISSING
        r.settle()
        assert r.check().state == OK

    SCENARIOS = {
        "missing": lambda r: None,
        "ok": lambda r: r.settle(),
        "policy-drift": lambda r: (r.settle(), put(r.ctx, obs.AGENT_POLICY, "stale\n", 0o600)),
        "token-drift": lambda r: (r.settle(), put(r.ctx, obs.TOKEN_FILE, "stale", 0o600)),
        "failed-unit": lambda r: (r.settle(), setattr(r.fake, "active", "failed")),
        "near-expiry": lambda r: (r.settle(), r.seed(secrets_text(ES_KEY_EXPIRES="2026-11-05T00:00:00Z"))),
        "bad-key": lambda r: r.seed(secrets_text(ES_API_KEY="kid1234:sec$5678")),
        "bad-fingerprint": lambda r: r.seed(secrets_text(ES_CA_FINGERPRINT=FP[:-2])),
        "expired": lambda r: r.seed(secrets_text(ES_KEY_EXPIRES="2020-01-01T00:00:00Z")),
        "bad-hosts": lambda r: r.seed(secrets_text(ES_HOSTS="http://" + KEY)),
        "half": lambda r: put(r.ctx, obs.AGENT_BIN, "#!/bin/sh\n", 0o755),
    }

    @pytest.mark.parametrize("scenario", sorted(SCENARIOS))
    def test_no_secret_in_any_detail_or_plan_line(self, build_rig, scenario):
        r = build_rig()
        self.SCENARIOS[scenario](r)
        res = r.check()
        no_secret_in(res.detail, *res.plan)
        # and the same on a dry run, which prints the plan to a terminal
        r.ctx.dry_run = True
        res = r.check()
        no_secret_in(res.detail, *res.plan)

    def test_a_malformed_key_is_never_echoed(self, build_rig):
        r = build_rig()
        r.seed(secrets_text(ES_API_KEY="SECRETMARKER$9999"))
        res = r.check()
        assert res.state == BLOCKED
        assert "SECRETMARKER" not in res.detail

    def test_no_secret_in_a_dry_run_with_no_env_file(self, build_rig):
        r = build_rig(dry_run=True)
        r.ctx.path(r.ctx.plan["paths"]["envfile"]).unlink()
        res = r.check()
        no_secret_in(res.detail, *res.plan)


# ==========================================================================
# 5. apply() and the dry run
# ==========================================================================
class TestApply:
    def test_a_fresh_install_converges(self, build_rig, downloads):
        r = build_rig()
        assert r.check().state == MISSING
        r.apply()
        res = r.check()
        assert res.state == OK, (res.detail, res.plan)
        assert r.ctx.read(obs.AGENT_POLICY) == r.expected_policy()
        assert r.ctx.read(obs.TOKEN_FILE) == TOKEN
        assert r.ctx.read(obs.DROPIN) == obs._dropin(r.cfg)

    @POSIX
    def test_the_policy_and_token_are_0600_and_the_dropin_is_readable(self, build_rig, downloads):
        r = build_rig()
        r.apply()
        mode = lambda p: os.stat(r.ctx.path(p)).st_mode & 0o777  # noqa: E731
        assert mode(obs.AGENT_POLICY) == 0o600
        assert mode(obs.TOKEN_FILE) == 0o600
        assert mode(obs.DROPIN) == 0o644
        # the staged copy is private before the installer ever sees it
        assert r.fake.staged_mode == 0o600

    def test_apply_fails_when_the_agent_reports_failed_after_the_start(self, build_rig, downloads):
        r = build_rig()
        r.fake.status_out = (1, json.dumps({"state": 4, "components": [{"id": "x", "state": 4}]}))
        with pytest.raises(RuntimeError, match="reports agent failed"):
            r.apply()

    def test_apply_fails_when_the_unit_crash_loops_after_the_start(self, build_rig, downloads):
        r = build_rig()
        real = r.fake.sudo

        def crashing(argv, **kw):
            result = real(argv, **kw)
            if argv[:3] == ["systemctl", "enable", "--now"]:
                r.fake.active, r.fake.sub = "activating", "auto-restart"
            return result

        r.ctx.sudo = crashing
        with pytest.raises(RuntimeError, match="stopped after it was started"):
            r.apply()

    def test_apply_stops_watching_once_the_agent_is_healthy(self, build_rig, downloads, monkeypatch):
        monkeypatch.setattr(obs, "SETTLE_S", 600)
        slept: list = []
        monkeypatch.setattr(obs.time, "sleep", lambda s: slept.append(s))
        r = build_rig()
        r.fake.status_out = (0, json.dumps({"state": 2, "components": [{"id": "x", "state": 2}]}))
        r.apply()
        assert slept == []

    def test_the_token_is_written_without_a_trailing_newline(self, build_rig, downloads):
        r = build_rig()
        r.apply()
        assert r.ctx.path(obs.TOKEN_FILE).read_bytes() == TOKEN.encode()

    def test_no_secret_in_any_argv_or_keyword(self, build_rig, downloads):
        r = build_rig()
        r.apply()
        assert r.fake.sudo_calls
        no_secret_in(*[" ".join(a) for a in r.fake.argvs],
                     *[repr(sorted(kw.items())) for _, kw in r.fake.sudo_calls])
        assert r.fake.run_calls == []

    def test_no_secret_in_what_apply_reports(self, build_rig, downloads, capsys):
        r = build_rig()
        r.apply()
        no_secret_in(*r.ctx.actions, *r.ctx.notes, capsys.readouterr().out)
        assert any(line.startswith("installed elastic-agent") for line in r.ctx.actions)

    def test_install_uses_cwd_and_non_interactive(self, build_rig, downloads):
        r = build_rig()
        r.apply()
        installs = [(a, kw) for a, kw in r.fake.sudo_calls if "install" in a[:2]]
        assert len(installs) == 1
        argv, kw = installs[0]
        # exactly this: no --url, no --enrollment-token, nothing secret
        assert len(argv) == 3 and argv[1:] == ["install", "--non-interactive"]
        assert Path(argv[0]).name == "elastic-agent"
        assert kw["cwd"] == str(Path(argv[0]).parent)
        assert kw["timeout"] == 900
        assert not any(a.startswith("--") and a != "--non-interactive" for a in argv)

    def test_the_work_directory_is_removed(self, build_rig, downloads):
        r = build_rig()
        r.apply()
        assert len(downloads) == 1
        assert not downloads[0].exists()

    def test_the_work_directory_is_removed_when_the_install_fails(self, build_rig, downloads):
        r = build_rig()

        def fail(argv):
            if argv[1:2] == ["install"]:
                raise StepFailed("install exploded")

        r.fake.fail_on = fail
        with pytest.raises(StepFailed, match="install exploded"):
            r.apply()
        assert not downloads[0].exists()

    @POSIX
    def test_the_umask_is_restored(self, build_rig, downloads):
        r = build_rig()
        before = os.umask(0o022)
        try:
            r.apply()
            assert os.umask(before) == 0o022
        finally:
            os.umask(before)

    @POSIX
    def test_the_umask_is_restored_when_it_fails(self, build_rig, downloads):
        r = build_rig()
        r.fake.fail_on = lambda argv: (_ for _ in ()).throw(StepFailed("no")) if argv[1:2] == ["install"] else None
        before = os.umask(0o022)
        try:
            with pytest.raises(StepFailed):
                r.apply()
            assert os.umask(before) == 0o022
        finally:
            os.umask(before)

    def test_the_secrets_directory_is_chmodded_and_chowned_explicitly(self, build_rig, downloads):
        r = build_rig()
        r.apply()
        assert ["chmod", "0700", obs.SECRETS_DIR] in r.fake.argvs
        assert ["chown", "root:root", obs.SECRETS_DIR] in r.fake.argvs

    def test_apply_needs_the_carried_key_file_on_a_box_with_no_secrets_directory(self, build_rig, downloads):
        r = build_rig(seed=False)
        put(r.ctx, r.ctx.plan["paths"]["envfile"], f"LLMSTACK_ADMIN_TOKEN={TOKEN}\n")
        with pytest.raises(RuntimeError, match="is not there"):
            r.apply()

    def test_the_commands_in_order(self, build_rig, downloads):
        r = build_rig()
        r.apply()
        a = r.fake.argvs
        install = next(i for i, c in enumerate(a) if c[1:2] == ["install"])
        reload_ = a.index(["systemctl", "daemon-reload"])
        chmod_policy = a.index(["chmod", "0600", obs.AGENT_POLICY])
        enable = a.index(["systemctl", "enable", "--now", obs.UNIT])
        assert reload_ < install < chmod_policy < enable
        assert ["systemctl", "reset-failed", obs.UNIT] in a
        assert a.index(["systemctl", "reset-failed", obs.UNIT]) < enable
        # reset-failed may fail on a unit that never failed
        kw = next(kw for c, kw in r.fake.sudo_calls if c[:2] == ["systemctl", "reset-failed"])
        assert kw.get("check") is False

    def test_a_fresh_install_does_not_restart(self, build_rig, downloads):
        r = build_rig()
        r.apply()
        assert ["systemctl", "restart", obs.UNIT] not in r.fake.argvs

    def test_installed_and_settled_restarts_nothing_and_reinstalls_nothing(self, build_rig, downloads):
        r = build_rig()
        r.settle()
        r.apply()
        assert ["systemctl", "restart", obs.UNIT] not in r.fake.argvs
        assert not any(c[1:2] == ["install"] for c in r.fake.argvs)
        assert downloads == []
        assert ["systemctl", "daemon-reload"] not in r.fake.argvs
        assert r.check().state == OK

    def test_installed_restarts_only_on_a_policy_change(self, build_rig, downloads):
        r = build_rig()
        r.settle()
        put(r.ctx, obs.AGENT_POLICY, "outputs: {}\n", 0o600)
        r.apply()
        assert r.fake.argvs.count(["systemctl", "restart", obs.UNIT]) == 1
        assert r.ctx.read(obs.AGENT_POLICY) == r.expected_policy()
        assert downloads == []
        assert r.check().state == OK
        # and the second apply, with nothing to do, restarts nothing more
        r.apply()
        assert r.fake.argvs.count(["systemctl", "restart", obs.UNIT]) == 1

    def test_a_token_rotation_converges_without_a_restart(self, build_rig, downloads):
        r = build_rig()
        r.settle()
        put(r.ctx, r.ctx.plan["paths"]["envfile"], "LLMSTACK_ADMIN_TOKEN=rotated-token-value\n")
        assert r.check().state == DRIFT
        r.apply()
        assert r.ctx.read(obs.TOKEN_FILE) == "rotated-token-value"
        assert r.check().state == OK
        # auth.file is read per request, so the policy did not change
        assert ["systemctl", "restart", obs.UNIT] not in r.fake.argvs

    @POSIX
    def test_a_token_file_with_the_right_content_and_wrong_mode_is_repaired(self, build_rig, downloads):
        # ctx.write only chmods a file it wrote; without the explicit repair
        # check() says drift and apply() does nothing -- the UNCHANGED loop.
        r = build_rig()
        r.settle()
        r.ctx.path(obs.TOKEN_FILE).chmod(0o644)
        assert r.check().state == DRIFT
        r.apply()
        assert os.stat(r.ctx.path(obs.TOKEN_FILE)).st_mode & 0o777 == 0o600
        assert r.check().state == OK

    @POSIX
    def test_a_policy_with_the_right_content_and_wrong_mode_is_repaired(self, build_rig, downloads):
        r = build_rig()
        r.settle()
        r.ctx.path(obs.AGENT_POLICY).chmod(0o644)
        r.apply()
        assert os.stat(r.ctx.path(obs.AGENT_POLICY)).st_mode & 0o777 == 0o600
        assert r.check().state == OK

    def test_a_stopped_unit_is_started(self, build_rig, downloads):
        r = build_rig()
        r.settle()
        r.fake.active, r.fake.sub, r.fake.enabled = "inactive", "dead", "disabled"
        r.apply()
        assert ["systemctl", "enable", "--now", obs.UNIT] in r.fake.argvs
        assert r.check().state == OK

    def test_a_half_install_is_uninstalled_from_outside_its_directory_first(self, build_rig, downloads):
        r = build_rig()
        put(r.ctx, obs.AGENT_BIN, "#!/bin/sh\n", 0o755)
        r.apply()
        a = r.fake.argvs
        uninstall = a.index([obs.AGENT_BIN, "uninstall", "--force"])
        install = next(i for i, c in enumerate(a) if c[1:2] == ["install"])
        assert uninstall < install
        assert r.fake.uninstall_cwd == "/"
        assert r.check().state == OK

    def test_the_policy_written_after_install_is_ours_if_the_installer_rewrote_it(self, build_rig, downloads):
        r = build_rig()
        real = r.fake.sudo

        def rewriting(argv, **kw):
            result = real(argv, **kw)
            if [str(a) for a in argv][1:] == ["install", "--non-interactive"]:
                put(r.ctx, obs.AGENT_POLICY, "# the installer's own default\n", 0o600)
            return result

        r.ctx.sudo = rewriting
        r.apply()
        assert r.ctx.read(obs.AGENT_POLICY) == r.expected_policy()
        assert ["systemctl", "restart", obs.UNIT] in r.fake.argvs

    # -- guards --------------------------------------------------------------
    def test_a_low_disk_stops_before_the_download(self, build_rig, downloads, monkeypatch):
        monkeypatch.setattr(obs, "_free", lambda where: 1024 ** 3)
        r = build_rig()
        with pytest.raises(RuntimeError, match="GB free"):
            r.apply()
        assert downloads == []
        assert r.fake.sudo_calls == []

    def test_a_policy_only_change_does_not_need_the_free_space(self, build_rig, downloads, monkeypatch):
        def never(where):
            raise AssertionError("asked about disk space with nothing to download")

        monkeypatch.setattr(obs, "_free", never)
        r = build_rig()
        r.settle()
        put(r.ctx, obs.AGENT_POLICY, "stale\n", 0o600)
        r.apply()
        assert r.check().state == OK

    def test_free_falls_back_to_the_nearest_existing_parent(self, tmp_path):
        assert obs._free(tmp_path / "does" / "not" / "exist") > 0

    def test_apply_without_the_secrets_file_says_where_and_leaks_nothing(self, build_rig, downloads):
        r = build_rig()
        r.ctx.path(r.cfg["secrets_file"]).unlink()
        with pytest.raises(RuntimeError, match="agent-output.env"):
            r.apply()
        assert downloads == [] and r.fake.sudo_calls == []

    def test_apply_without_a_gateway_token_refuses(self, build_rig, downloads):
        r = build_rig()
        r.ctx.path(r.ctx.plan["paths"]["envfile"]).unlink()
        with pytest.raises(RuntimeError, match="no gateway admin token"):
            r.apply()
        assert r.fake.sudo_calls == []

    def test_apply_never_mints_a_gateway_token(self, build_rig, downloads):
        # EnvFile mints one when there is none; this step must only read, or it
        # could replace the credential the hub holds for this peer.
        r = build_rig()
        env = r.ctx.path(r.ctx.plan["paths"]["envfile"])
        env.unlink()
        with pytest.raises(RuntimeError):
            r.apply()
        assert not env.exists()

    # -- scrubbing -----------------------------------------------------------
    @pytest.mark.parametrize("on", ["install", "enable", "chmod"])
    def test_a_stepfailed_is_reraised_scrubbed(self, build_rig, downloads, on):
        r = build_rig()
        words = {"install": "install", "enable": "enable", "chmod": "chmod"}

        def fail(argv):
            if words[on] in argv:
                raise StepFailed(f"{' '.join(argv)}\n  exit 1\n  rejected {KEY} for {TOKEN} (id kid1234)")

        r.fake.fail_on = fail
        with pytest.raises(StepFailed) as err:
            r.apply()
        no_secret_in(str(err.value))
        assert "<redacted>" in str(err.value)
        assert "exit 1" in str(err.value)

    def test_a_non_stepfailed_error_is_scrubbed_too(self, build_rig, monkeypatch, downloads):
        r = build_rig()

        def boom(ctx, dest_dir):
            raise RuntimeError(f"proxy said no to {KEY}")

        monkeypatch.setattr(obs, "_download_and_verify", boom)
        with pytest.raises(RuntimeError) as err:
            r.apply()
        no_secret_in(str(err.value))
        assert "RuntimeError" in str(err.value)

    def test_the_original_traceback_is_not_chained_to_the_secret(self, build_rig, downloads):
        r = build_rig()

        def fail(argv):
            if argv[1:2] == ["install"]:
                raise StepFailed(f"leaky {KEY}")

        r.fake.fail_on = fail
        with pytest.raises(StepFailed) as err:
            r.apply()
        assert err.value.__cause__ is None
        assert err.value.__suppress_context__ is True


class TestDryRun:
    def test_dry_run_downloads_and_writes_nothing(self, build_rig, monkeypatch):
        monkeypatch.setattr(obs, "_is_root", lambda: False)

        def refuse(*a, **kw):
            pytest.fail("a dry run reached the download")

        monkeypatch.setattr(obs, "_download_and_verify", refuse)
        monkeypatch.setattr(obs, "_fetch", refuse)
        monkeypatch.setattr(obs, "_urlopen", refuse)
        r = build_rig(dry_run=True)
        before = snapshot(r.ctx.path("/"))
        assert r.check().state == MISSING
        r.step.run(r.ctx)  # the one entry point that enforces the dry-run contract
        assert snapshot(r.ctx.path("/")) == before
        assert r.fake.sudo_calls == [] and r.fake.run_calls == []
        assert not r.ctx.path("/opt").exists()
        assert not r.ctx.path(obs.TOKEN_FILE).exists()

    def test_a_real_run_goes_through_run_to_apply(self, build_rig, downloads):
        r = build_rig()
        r.step.run(r.ctx)
        assert r.check().state == OK

    def test_dry_run_plan_lines(self, build_rig, monkeypatch):
        monkeypatch.setattr(obs, "_is_root", lambda: False)
        r = build_rig(dry_run=True)
        res = r.check()
        assert res.state == MISSING
        assert res.plan == [
            f"install elastic-agent {VERSION}",
            f"write {obs.AGENT_POLICY} (not written yet)",
            f"write {obs.TOKEN_FILE}",
            f"write {obs.DROPIN}"]
        assert all(line and len(line.splitlines()) == 1 for line in res.plan)
        no_secret_in(res.detail, *res.plan)

    def test_dry_run_plan_for_a_drifted_box_is_specific(self, build_rig, monkeypatch):
        monkeypatch.setattr(obs, "_is_root", lambda: False)
        r = build_rig(dry_run=True)
        r.settle()
        r.fake.enabled = "disabled"
        put(r.ctx, obs.AGENT_POLICY, "stale\n", 0o600)
        res = r.check()
        assert res.state == DRIFT
        assert res.plan == [f"write {obs.AGENT_POLICY} (policy differs)", f"systemctl enable --now {obs.UNIT}"]


# ==========================================================================
# 6. the download
# ==========================================================================
URL = f"https://artifacts.elastic.co/downloads/beats/elastic-agent/elastic-agent-{VERSION}-linux-x86_64.tar.gz"
TOP = f"elastic-agent-{VERSION}-linux-x86_64"


def file_member(name: str, data: bytes = b"x"):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o755
    return info, data


def dir_member(name: str):
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    return info, None


def link_member(name: str, target: str, *, hard: bool = False):
    info = tarfile.TarInfo(name)
    info.type = tarfile.LNKTYPE if hard else tarfile.SYMTYPE
    info.linkname = target
    return info, None


def special_member(name: str, kind: bytes):
    info = tarfile.TarInfo(name)
    info.type = kind
    return info, None


def tar_bytes(members) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for info, data in members:
            tar.addfile(info, io.BytesIO(data) if data is not None else None)
    return buf.getvalue()


def agent_tar() -> bytes:
    return tar_bytes([dir_member(TOP), file_member(f"{TOP}/elastic-agent", b"#!/bin/sh\necho agent\n"),
                      file_member(f"{TOP}/README.txt", b"hello")])


def sha512(data: bytes) -> str:
    return hashlib.sha512(data).hexdigest()


class Net:
    """urlopen, with a script. A route's value is bytes, an exception to raise,
    or a list of those to use one per call."""

    def __init__(self, routes: dict):
        self.routes = {k: (list(v) if isinstance(v, list) else v) for k, v in routes.items()}
        self.requests: list = []

    def __call__(self, req, timeout):
        self.requests.append((req, timeout))
        body = self.routes[req.full_url]
        if isinstance(body, list):
            body = body.pop(0)
        if isinstance(body, Exception):
            raise body
        return io.BytesIO(body)

    @property
    def urls(self) -> list:
        return [r.full_url for r, _ in self.requests]


@pytest.fixture
def dl(linux_facts, empty_repo, tmp_path, monkeypatch):
    """(ctx factory, work dir, sleeps) for the download tests."""
    sleeps: list = []
    monkeypatch.setattr(obs.time, "sleep", lambda s: sleeps.append(s))
    work = tmp_path / "work"
    work.mkdir()

    def ctx_with(digest=None, **section):
        pins = [] if digest is None else [f"{VERSION}/{ASSET}={digest}"]
        return make_ctx(linux_facts, empty_repo, tmp_path,
                        section={"enabled": True, "agent_sha512": pins, **section})

    return ctx_with, work, sleeps


class TestAssetTable:
    @pytest.mark.parametrize("family,arch,expect", [
        ("linux", "x86_64", "linux-x86_64.tar.gz"),
        ("linux", "aarch64", "linux-arm64.tar.gz"),
        ("linux", "armv7l", None),
        ("linux", "arm64", None),
        ("darwin", "arm64", None),
        ("windows", "AMD64", None),
        ("linux", None, None),
        ("linux", "", None),
    ])
    def test_the_asset_for_a_family_and_arch(self, family, arch, expect):
        assert shapes.elastic_agent_asset(family, arch) == expect

    def test_the_table_has_exactly_the_two_real_entries(self):
        assert set(shapes.ELASTIC_AGENT_ASSETS) == {("linux", "x86_64"), ("linux", "aarch64")}

    def test_the_url_template(self):
        assert shapes.ELASTIC_AGENT_URL.format(version=VERSION, asset=ASSET) == URL


class TestDownloadAndVerify:
    def test_a_pinned_digest_that_matches_extracts_and_returns_the_directory(self, dl, monkeypatch):
        ctx_with, work, _ = dl
        data = agent_tar()
        net = Net({URL: data})
        monkeypatch.setattr(obs, "_urlopen", net)
        found = obs._download_and_verify(ctx_with(sha512(data)), work)
        assert found.name == TOP
        assert (found / "elastic-agent").read_bytes() == b"#!/bin/sh\necho agent\n"
        assert net.urls == [URL]
        req = net.requests[0][0]
        assert req.get_header("User-agent") == "fleetctl"
        assert net.requests[0][1] > 0

    def test_a_wrong_digest_aborts_before_extracting(self, dl, monkeypatch):
        ctx_with, work, _ = dl
        monkeypatch.setattr(obs, "_urlopen", Net({URL: agent_tar()}))
        with pytest.raises(RuntimeError, match="nothing was extracted"):
            obs._download_and_verify(ctx_with("ef" * 64), work)
        assert not (work / "unpacked").exists()

    def test_the_error_does_not_print_the_whole_digest(self, dl, monkeypatch):
        ctx_with, work, _ = dl
        monkeypatch.setattr(obs, "_urlopen", Net({URL: agent_tar()}))
        with pytest.raises(RuntimeError) as err:
            obs._download_and_verify(ctx_with("ef" * 64), work)
        assert "ef" * 64 not in str(err.value)

    def test_no_pin_is_refused_before_any_request(self, dl, monkeypatch):
        ctx_with, work, _ = dl
        net = Net({})
        monkeypatch.setattr(obs, "_urlopen", net)
        with pytest.raises(RuntimeError, match="no sha512 pin"):
            obs._download_and_verify(ctx_with(None), work)
        assert net.requests == []

    def test_a_stale_pin_for_another_version_is_no_pin(self, dl, monkeypatch):
        ctx_with, work, _ = dl
        net = Net({})
        monkeypatch.setattr(obs, "_urlopen", net)
        with pytest.raises(RuntimeError, match="no sha512 pin"):
            obs._download_and_verify(ctx_with("cd" * 64, agent_version="9.5.7"), work)
        assert net.requests == []

    def test_unsupported_arch_is_refused_before_any_request(self, dl, monkeypatch):
        ctx_with, work, _ = dl
        net = Net({})
        monkeypatch.setattr(obs, "_urlopen", net)
        ctx = ctx_with("cd" * 64)
        ctx.plan["platform"]["arch"] = "riscv64"
        with pytest.raises(RuntimeError, match="no agent build"):
            obs._download_and_verify(ctx, work)
        assert net.requests == []

    def test_the_arm_build_is_fetched_for_aarch64(self, dl, monkeypatch):
        ctx_with, work, _ = dl
        arm = f"https://artifacts.elastic.co/downloads/beats/elastic-agent/elastic-agent-{VERSION}-linux-arm64.tar.gz"
        top = f"elastic-agent-{VERSION}-linux-arm64"
        data = tar_bytes([file_member(f"{top}/elastic-agent")])
        net = Net({arm: data})
        monkeypatch.setattr(obs, "_urlopen", net)
        ctx = ctx_with(None)
        ctx.plan["observability"]["agent_sha512"] = [f"{VERSION}/linux-arm64.tar.gz={sha512(data)}"]
        ctx.plan["platform"]["arch"] = "aarch64"
        assert obs._download_and_verify(ctx, work).name == top
        assert net.urls == [arm]

    # -- unpinned ------------------------------------------------------------
    def test_unpinned_uses_the_sibling_file(self, dl, monkeypatch):
        ctx_with, work, _ = dl
        data = agent_tar()
        net = Net({URL: data, URL + ".sha512": f"{sha512(data)}  elastic-agent-{VERSION}-linux-x86_64.tar.gz\n".encode()})
        monkeypatch.setattr(obs, "_urlopen", net)
        assert obs._download_and_verify(ctx_with(None, allow_unpinned=True), work).name == TOP
        assert net.urls == [URL, URL + ".sha512"]

    def test_unpinned_with_a_wrong_sibling_digest_aborts(self, dl, monkeypatch):
        ctx_with, work, _ = dl
        net = Net({URL: agent_tar(), URL + ".sha512": ("ef" * 64).encode()})
        monkeypatch.setattr(obs, "_urlopen", net)
        with pytest.raises(RuntimeError, match="nothing was extracted"):
            obs._download_and_verify(ctx_with(None, allow_unpinned=True), work)
        assert not (work / "unpacked").exists()

    def test_a_sibling_that_is_not_a_digest_is_refused(self, dl, monkeypatch):
        ctx_with, work, _ = dl
        net = Net({URL: agent_tar(), URL + ".sha512": b"<html>not found</html>"})
        monkeypatch.setattr(obs, "_urlopen", net)
        with pytest.raises(RuntimeError, match="sha512 digest"):
            obs._download_and_verify(ctx_with(None, allow_unpinned=True), work)

    def test_a_pin_wins_over_a_sibling_that_would_agree_with_a_tampered_archive(self, dl, monkeypatch):
        # The sibling comes from the same origin as the archive, so it proves
        # the transfer and not the author. A pin that is present is the only
        # thing compared.
        ctx_with, work, _ = dl
        data = agent_tar()
        net = Net({URL: data, URL + ".sha512": sha512(data).encode()})
        monkeypatch.setattr(obs, "_urlopen", net)
        with pytest.raises(RuntimeError, match="nothing was extracted"):
            obs._download_and_verify(ctx_with("ef" * 64, allow_unpinned=True), work)
        assert net.urls == [URL]

    # -- retries ---------------------------------------------------------------
    def test_a_transient_failure_is_retried_with_backoff(self, dl, monkeypatch):
        ctx_with, work, sleeps = dl
        data = agent_tar()
        net = Net({URL: [OSError("reset"), OSError("reset"), data]})
        monkeypatch.setattr(obs, "_urlopen", net)
        assert obs._download_and_verify(ctx_with(sha512(data)), work).name == TOP
        assert len(net.requests) == 3
        assert sleeps == [1, 2]

    def test_three_failures_give_up_naming_the_url(self, dl, monkeypatch):
        ctx_with, work, _ = dl
        net = Net({URL: [OSError("down"), OSError("down"), OSError("down")]})
        monkeypatch.setattr(obs, "_urlopen", net)
        with pytest.raises(RuntimeError, match="could not download") as err:
            obs._download_and_verify(ctx_with("cd" * 64), work)
        assert URL in str(err.value)
        assert len(net.requests) == 3

    def test_a_refused_redirect_is_not_retried(self, dl, monkeypatch):
        ctx_with, work, _ = dl
        net = Net({URL: [obs._Refused("refusing a redirect to evil.example"), b"never used"]})
        monkeypatch.setattr(obs, "_urlopen", net)
        with pytest.raises(obs._Refused):
            obs._download_and_verify(ctx_with("cd" * 64), work)
        assert len(net.requests) == 1

    # -- the archive ---------------------------------------------------------
    @pytest.mark.parametrize("members", [
        [file_member("../escaped", b"x")],
        [file_member(f"{TOP}/../../escaped", b"x")],
        [file_member("/abs/escaped", b"x")],
    ], ids=["dotdot", "nested-dotdot", "absolute"])
    def test_an_archive_that_would_write_outside_is_rejected(self, dl, monkeypatch, members):
        ctx_with, work, _ = dl
        data = tar_bytes([dir_member(TOP), file_member(f"{TOP}/elastic-agent")] + members)
        monkeypatch.setattr(obs, "_urlopen", Net({URL: data}))
        with pytest.raises(RuntimeError, match="unsafe member"):
            obs._download_and_verify(ctx_with(sha512(data)), work)
        assert not (work / "escaped").exists()
        assert not (work.parent / "escaped").exists()
        assert list((work / "unpacked").rglob("*")) == []

    def test_an_archive_with_two_top_directories_is_refused(self, dl, monkeypatch):
        ctx_with, work, _ = dl
        data = tar_bytes([file_member(f"{TOP}/elastic-agent"), file_member("other/elastic-agent")])
        monkeypatch.setattr(obs, "_urlopen", Net({URL: data}))
        with pytest.raises(RuntimeError, match="exactly one directory"):
            obs._download_and_verify(ctx_with(sha512(data)), work)

    def test_an_archive_without_the_binary_is_refused(self, dl, monkeypatch):
        ctx_with, work, _ = dl
        data = tar_bytes([file_member(f"{TOP}/README.txt")])
        monkeypatch.setattr(obs, "_urlopen", Net({URL: data}))
        with pytest.raises(RuntimeError, match="exactly one directory"):
            obs._download_and_verify(ctx_with(sha512(data)), work)

    def test_a_download_that_is_not_a_tarball_fails_after_the_hash_check(self, dl, monkeypatch):
        ctx_with, work, _ = dl
        data = b"definitely not a gzip file"
        monkeypatch.setattr(obs, "_urlopen", Net({URL: data}))
        with pytest.raises(tarfile.TarError):
            obs._download_and_verify(ctx_with(sha512(data)), work)

    def test_a_download_past_the_size_cap_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(obs, "MAX_DOWNLOAD", 2 * (1 << 20))
        monkeypatch.setattr(obs, "_urlopen", Net({URL: b"z" * (3 * (1 << 20))}))
        with pytest.raises(obs._Refused, match="larger than"):
            obs._fetch(URL, tmp_path / "a.bin")

    def test_the_archive_is_streamed_to_disk_and_hashed_in_one_pass(self, tmp_path, monkeypatch):
        data = b"z" * (3 * (1 << 20) + 17)
        monkeypatch.setattr(obs, "_urlopen", Net({URL: data}))
        dest = tmp_path / "a.bin"
        assert obs._fetch(URL, dest) == sha512(data)
        assert dest.read_bytes() == data


class TestMemberVetting:
    ROOT = Path("unpacked")

    @pytest.mark.parametrize("member", [
        file_member("../x"),
        file_member("a/../../x"),
        file_member("a\\..\\x"),
        file_member("/etc/passwd"),
        file_member("\\windows\\x"),
        file_member("C:/windows/x"),
        link_member("d/l", "../../outside"),
        link_member("d/l", "/etc/passwd"),
        link_member("l", ".."),
        link_member("d/h", "../../etc/passwd", hard=True),
        link_member("h", "/etc/passwd", hard=True),
        special_member("dev", tarfile.CHRTYPE),
        special_member("dev", tarfile.BLKTYPE),
        special_member("pipe", tarfile.FIFOTYPE),
    ], ids=["dotdot", "inner-dotdot", "backslash-dotdot", "absolute", "backslash-absolute", "drive",
            "symlink-out", "symlink-abs", "symlink-parent", "hardlink-out", "hardlink-abs",
            "chardev", "blockdev", "fifo"])
    def test_unsafe_members_are_rejected(self, member):
        with pytest.raises(RuntimeError, match="unsafe member"):
            obs._vet_member(member[0], self.ROOT)

    @pytest.mark.parametrize("member", [
        file_member("d/f"),
        dir_member("d"),
        link_member("d/l", "../other"),
        link_member("d/l", "sibling"),
        link_member("h", "d/f", hard=True),
    ], ids=["file", "dir", "symlink-inside", "symlink-sibling", "hardlink-inside"])
    def test_safe_members_pass(self, member):
        obs._vet_member(member[0], self.ROOT)

    def test_a_symlink_chain_that_leaves_the_tree_is_refused(self, tmp_path):
        # Each link is inside the tree on its own; together a/b is the root and
        # a/b/c is two levels above it.
        data = tar_bytes([link_member("a/b", ".."), link_member("a/b/c", "../.."),
                          file_member("a/b/c/x")])
        archive = tmp_path / "a.tar.gz"
        archive.write_bytes(data)
        with pytest.raises(RuntimeError, match="unsafe member"):
            obs._extract(archive, tmp_path / "out")
        assert list((tmp_path / "out").rglob("*")) == []

    def test_a_file_under_an_earlier_symlink_is_refused(self):
        seen: set = set()
        obs._vet_member(link_member("d/l", "sibling")[0], self.ROOT, seen)
        with pytest.raises(RuntimeError, match="unsafe member"):
            obs._vet_member(file_member("d/l/f")[0], self.ROOT, seen)

    def test_an_unrelated_path_next_to_a_symlink_passes(self):
        seen: set = set()
        obs._vet_member(link_member("d/l", "sibling")[0], self.ROOT, seen)
        obs._vet_member(file_member("d/l2/f")[0], self.ROOT, seen)
        obs._vet_member(file_member("d/other")[0], self.ROOT, seen)

    def test_extract_vets_everything_before_writing_anything(self, tmp_path):
        data = tar_bytes([file_member("good/file"), file_member("../bad")])
        archive = tmp_path / "a.tar.gz"
        archive.write_bytes(data)
        with pytest.raises(RuntimeError, match="unsafe member"):
            obs._extract(archive, tmp_path / "out")
        assert list((tmp_path / "out").rglob("*")) == []


class TestRedirects:
    def handler(self):
        return obs._StrictRedirects()

    def request(self):
        return urllib.request.Request(URL, headers={"User-Agent": "fleetctl"})

    @pytest.mark.parametrize("target", [
        "https://evil.example/elastic-agent.tar.gz",
        "http://artifacts.elastic.co/elastic-agent.tar.gz",
        "https://artifacts.elastic.co:8443/elastic-agent.tar.gz",
        "https://artifacts.elastic.co.evil.example/elastic-agent.tar.gz",
        "https://artifacts.elastic.co@evil.example/elastic-agent.tar.gz",
        "ftp://artifacts.elastic.co/elastic-agent.tar.gz",
        "file:///etc/passwd",
    ])
    def test_a_redirect_off_the_allowed_host_is_refused(self, target):
        with pytest.raises(obs._Refused, match="refusing a redirect"):
            self.handler().redirect_request(self.request(), None, 302, "Found", {}, target)

    @pytest.mark.parametrize("target", [
        "https://artifacts.elastic.co/other/path.tar.gz",
        "https://ARTIFACTS.elastic.co/other/path.tar.gz",
        "https://artifacts.elastic.co:443/other/path.tar.gz",
    ])
    def test_a_redirect_within_the_allowed_host_is_followed(self, target):
        new = self.handler().redirect_request(self.request(), None, 302, "Found", {}, target)
        assert new is not None
        assert new.full_url == target

    def test_the_refusal_names_the_host_it_refused_and_the_one_it_allows(self):
        with pytest.raises(obs._Refused) as err:
            self.handler().redirect_request(self.request(), None, 302, "Found", {}, "https://evil.example/x")
        assert "evil.example" in str(err.value)
        assert "artifacts.elastic.co" in str(err.value)

    def test_the_allow_list_is_one_host(self):
        assert obs.ALLOWED_DOWNLOAD_HOSTS == ("artifacts.elastic.co",)
        assert obs._allowed_url(URL)
        assert not obs._allowed_url(URL.replace("https:", "http:"))

    def test_the_real_opener_is_built_with_the_strict_handler(self, monkeypatch):
        seen = []

        def fake_build(*handlers):
            seen.extend(handlers)
            return SimpleNamespace(open=lambda req, timeout: "response")

        monkeypatch.setattr(urllib.request, "build_opener", fake_build)
        assert obs._urlopen(self.request(), 5) == "response"
        assert seen == [obs._StrictRedirects]


# ==========================================================================
# 7. secrets in the plan, and the pieces next door
# ==========================================================================
class TestSecretsStayOutOfThePlan:
    def test_the_plan_holds_the_path_to_the_secrets_file_and_nothing_from_it(self, build_rig):
        r = build_rig(section={"enabled": True, "agent_sha512": [PIN], "secrets_file": "/etc/fleet-obs/agent-output.env"})
        # `plan --json` prints exactly this structure
        dumped = json.dumps({"plan": r.ctx.plan}, indent=2, default=str)
        assert "/etc/fleet-obs/agent-output.env" in dumped
        no_secret_in(dumped)
        r.check()
        no_secret_in(json.dumps({"plan": r.ctx.plan}, default=str))

    def test_the_plan_is_unchanged_by_a_check_and_by_an_apply(self, build_rig, downloads):
        r = build_rig()
        before = json.dumps(r.ctx.plan, sort_keys=True, default=str)
        r.check()
        r.apply()
        assert json.dumps(r.ctx.plan, sort_keys=True, default=str) == before

    def test_there_is_no_schema_key_a_secret_could_be_put_in(self):
        names = set(planner.SCHEMA["observability"])
        assert not {n for n in names if any(w in n for w in ("key", "token", "fingerprint", "password"))}

    def test_plan_write_round_trips_the_section_without_secrets(self, build_rig, tmp_path):
        from fleetctl import hostfile
        r = build_rig()
        plan, prov = planner.build(r.ctx.facts, repo=r.ctx.repo, overrides={
            "observability": {"enabled": True, "agent_sha512": [PIN]}})
        out = tmp_path / "host.yml"
        text = planner.write(plan, out, prov)
        no_secret_in(text, out.read_text(encoding="utf-8"))
        loaded = hostfile.load(out)
        assert loaded["observability"]["enabled"] is True
        assert loaded["observability"]["agent_sha512"] == [PIN]

    def test_the_written_host_yml_explains_the_new_keys(self, linux_facts, empty_repo, tmp_path):
        plan, prov = planner.build(linux_facts, repo=empty_repo, overrides={
            "observability": {"enabled": True, "journald_units": ["llm-gateway"]}})
        text = planner.write(plan, tmp_path / "host.yml", prov)
        assert "observability-agent" in text


class TestEnvFileAccessLog:
    def test_the_log_path_is_set_while_observability_is_on(self, linux_facts, empty_repo, tmp_path):
        ctx = make_ctx(linux_facts, empty_repo, tmp_path)
        values = EnvFile()._values(ctx)
        assert values["LLMSTACK_ACCESS_LOG"] == ctx.plan["paths"]["state"] + "/access.ndjson"
        assert values["LLMSTACK_ACCESS_LOG"] == shapes.access_log_path("linux", ctx.plan["paths"]["state"])

    @pytest.mark.parametrize("section", [{}, {"enabled": False}], ids=["absent", "disabled"])
    def test_the_log_is_not_enabled_otherwise(self, linux_facts, empty_repo, tmp_path, section):
        ctx = make_ctx(linux_facts, empty_repo, tmp_path, section=section)
        assert "LLMSTACK_ACCESS_LOG" not in EnvFile()._values(ctx)

    def test_switching_observability_off_drops_the_log_key_but_keeps_hand_set_sizes(
            self, linux_facts, empty_repo, tmp_path):
        on = make_ctx(linux_facts, empty_repo, tmp_path)
        step = EnvFile()
        text = (step._render(on, "tok-for-the-hub")
                + "LLMSTACK_ACCESS_LOG_MAX_BYTES=5000000\nLLMSTACK_ACCESS_LOG_BACKUPS=2\n")
        assert "LLMSTACK_ACCESS_LOG=" in text
        off = make_ctx(linux_facts, empty_repo, tmp_path, section={"enabled": False})
        carried = step.foreign(off, text)
        assert "LLMSTACK_ACCESS_LOG" not in carried
        assert carried["LLMSTACK_ACCESS_LOG_MAX_BYTES"] == "5000000"
        assert carried["LLMSTACK_ACCESS_LOG_BACKUPS"] == "2"
