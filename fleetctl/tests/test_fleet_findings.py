"""What running fleetctl against the real fleet found, one test per finding.

None of these came from reading the code. Every one of them is a dry run on
a machine that has been in production for months reporting something that was
not true of it -- which is the only reason they are worth a test: a check that
lies about a working box is worse than no check, because the fix for it is an
apply.

The findings, in the order the run printed them:

  brew is not on PATH        every Mac said `python@3.12 missing` about a
                             package that was installed
  brew wants no root         and the step was gated on it anyway
  video/render everywhere    permanent drift on the two boxes whose service
                             account never opens a GPU
  the engine lives elsewhere server-1 built its own llama.cpp; the plan
                             called it missing and would have downloaded a
                             release over the top of it
  update vs remove           a dry run said "update CF_ACCESS_AUD" about a
                             key the apply was going to drop
  the hub was a peer         fleet.yml's bind is right for peers, and would
                             have put the hub's admin API on the tailnet
  CRLF                       two files reported as differing on all 14 boxes
                             were byte-identical apart from newlines
  unreadable != absent       gpu-laptop-1's llama-swap config, holding seven
                             registered models, read as "no seed config" to
                             the step that answers that by writing one
  eight secrets              hub's env file holds HF_TOKEN, the public
                             intake token and six SMTP values, none of which
                             the plan can express -- so apply dropped them
  a firewall you cannot see  `ufw status` needs root, and the check read its
                             refusal as "no firewall here"
  an empty store             cpu-box-1's Ollama directory exists and holds
                             nothing; its weights are somewhere else
  the ssh alias              `--host mac-desktop` planned a box from scratch,
                             exit 0, and reported two correct values as drift
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from fleetctl import cli, planner, shapes
from fleetctl import facts as facts_mod
from fleetctl.runner import Ctx
from fleetctl.steps import BLOCKED, DRIFT, MISSING, OK, SKIPPED
from fleetctl.steps.engine import Engine
from fleetctl.steps.stack import EnvFile, SwapConfig, Wrappers, _diff_keys
from fleetctl.steps.system import (Firewall, Grants, Packages, Power,
                                   ServiceAccount)

SOURCE_BUILD = "/home/user/llama.cpp/build/bin/llama-server"


def ctx_for(facts, repo, tmp_path=None, **kw):
    plan, prov = planner.build(facts, repo=repo)
    ctx = Ctx(plan, facts, repo=repo,
              root=str(tmp_path / "sandbox") if tmp_path else None, **kw)
    ctx.prov = prov
    return ctx


def answering(stdout: str = "", returncode: int = 0):
    """A ctx.probe that answers instead of shelling out.

    These checks run `id -u llmstack` and `brew list`, neither of which means
    anything on the machine running the tests.
    """
    return lambda argv, timeout=60: SimpleNamespace(
        returncode=returncode, stdout=stdout, stderr="")


# --------------------------------------------------------------------------
class TestHomebrewIsNotOnThePath:
    """`ssh mac-desktop brew list` finds nothing and says nothing.

    /opt/homebrew/bin reaches PATH through the shellenv line in the login
    profile, which a command run over ssh never sources. The query came back
    empty on every Mac, so the packages step reported a package that has been
    installed since the box was provisioned as missing -- and the apply it
    proposed would have run a `brew` that does not resolve either.
    """

    def test_brew_resolves_to_a_real_path(self, tmp_path, monkeypatch):
        fake = tmp_path / "brew"
        fake.write_text("#!/bin/sh\n")
        monkeypatch.setattr(shapes, "BREW_CANDIDATES", (str(fake),))
        assert shapes.brew_bin() == str(fake)

    def test_a_box_with_no_homebrew_still_gets_a_command(self, monkeypatch):
        monkeypatch.setattr(shapes, "BREW_CANDIDATES", ("/nowhere/brew",))
        # Not an exception and not None: the step should fail with the
        # package manager's own error, not with a TypeError from here.
        assert shapes.brew_bin() == "brew"

    @pytest.mark.parametrize("table", ["PKG_QUERY", "PKG_INSTALL", "PKG_REFRESH"])
    def test_every_brew_table_is_resolved(self, table, tmp_path, monkeypatch):
        fake = tmp_path / "brew"
        fake.write_text("#!/bin/sh\n")
        monkeypatch.setattr(shapes, "BREW_CANDIDATES", (str(fake),))
        argv = shapes.pkg_argv("brew", getattr(shapes, table))
        assert argv[0] == str(fake)
        # ...and only argv[0]. `brew list --versions` must keep its flags.
        assert argv[1:] == list(getattr(shapes, table)["brew"])[1:]

    @pytest.mark.parametrize("family", ["apt", "dnf", "pacman", "zypper"])
    def test_the_others_are_left_exactly_alone(self, family):
        for table in (shapes.PKG_QUERY, shapes.PKG_INSTALL, shapes.PKG_REFRESH):
            assert shapes.pkg_argv(family, table) == list(table[family])

    def test_an_unknown_family_is_none_not_a_crash(self):
        assert shapes.pkg_argv("windows", shapes.PKG_INSTALL) is None


class TestHomebrewMustNotRunAsRoot:
    """Homebrew refuses to run as root. The step was gated on having it.

    So a Mac reported `blocked (needs root)` -- a state that means "re-run me
    with sudo", about the one package manager here for which sudo is the
    wrong answer.
    """

    def test_brew_does_not_need_root(self, darwin_facts, empty_repo):
        ctx = ctx_for(darwin_facts, empty_repo)
        assert Packages().needs_root(ctx) is False

    @pytest.mark.parametrize("fixture", ["linux_facts"])
    def test_everything_else_still_does(self, fixture, request, empty_repo):
        ctx = ctx_for(request.getfixturevalue(fixture), empty_repo)
        assert Packages().needs_root(ctx) is True

    def test_it_is_still_a_system_step(self):
        # needs_root and system are different questions. brew changes the
        # machine outside the plan's paths, so a --root sandbox must still
        # refuse it -- that gate is on `system`, and it has to stay on.
        assert Packages.system is True


class TestTheServiceAccountOnlyNeedsTheGpuWhenItOpensIt:
    """video and render, on the two boxes that never open a device.

    Where they are needed they are not cosmetic: without them the Vulkan
    loader does not raise, it falls back to llvmpipe and every model runs on
    the CPU at a plausible-looking speed. hub runs no engine and cpu-box-1
    runs Ollama under Ollama's own account, so both reported the same drift
    on every run. Drift that is always there is drift nobody reads.
    """

    def _check(self, facts, repo, kind, backend="vulkan"):
        ctx = ctx_for(facts, repo)
        ctx.plan["engine"]["kind"] = kind
        ctx.plan["engine"]["backend"] = backend
        ctx.probe = answering("llmstack\n")
        return ServiceAccount().check(ctx)

    def test_a_hub_with_no_engine_is_ok_without_them(self, linux_facts, empty_repo):
        assert self._check(linux_facts, empty_repo, "none").state == OK

    def test_an_ollama_box_is_ok_without_them(self, linux_facts, empty_repo):
        assert self._check(linux_facts, empty_repo, "ollama").state == OK

    def test_a_cpu_only_llama_box_is_ok_without_them(self, linux_facts, empty_repo):
        assert self._check(linux_facts, empty_repo, "llama.cpp", "cpu").state == OK

    def test_a_box_that_does_open_the_gpu_still_wants_them(self, linux_facts,
                                                          empty_repo):
        result = self._check(linux_facts, empty_repo, "llama.cpp", "vulkan")
        assert result.state == DRIFT
        assert any("usermod" in line for line in result.plan)

    def test_a_missing_account_is_still_missing_either_way(self, linux_facts,
                                                          empty_repo):
        ctx = ctx_for(linux_facts, empty_repo)
        ctx.plan["engine"]["kind"] = "none"
        ctx.probe = answering(returncode=1)
        result = ServiceAccount().check(ctx)
        assert result.state == MISSING
        # ...but it is not told to put an account that opens nothing into the
        # GPU groups.
        assert not any("usermod" in line for line in result.plan)


class TestAnEngineTheOperatorBuiltIsNotReplaced:
    """server-1 built llama.cpp itself and keeps it where it built it.

    Its gateway.env has pointed at /home/user/llama.cpp/build/bin since the
    box was provisioned. The plan derived that path from `bin`, so the engine
    step called a working engine `missing` -- and the apply it proposed would
    have downloaded a release into /opt/llmstack/bin and repointed the env at
    the download.
    """

    def test_a_detected_build_wins_over_the_default(self, linux_facts, empty_repo):
        linux_facts["engines"]["llama_server"] = SOURCE_BUILD
        plan, prov = planner.build(linux_facts, repo=empty_repo)
        assert plan["paths"]["llama_server"] == SOURCE_BUILD
        assert prov["paths.llama_server"] == "detected"

    def test_llama_bench_follows_it(self, linux_facts, empty_repo):
        linux_facts["engines"]["llama_server"] = SOURCE_BUILD
        plan, _ = planner.build(linux_facts, repo=empty_repo)
        assert plan["paths"]["llama_bench"] == \
            "/home/user/llama.cpp/build/bin/llama-bench"

    def test_a_detected_build_is_written_into_host_yml(self, linux_facts,
                                                       empty_repo):
        linux_facts["engines"]["llama_server"] = SOURCE_BUILD
        plan, prov = planner.build(linux_facts, repo=empty_repo)
        doc = planner.to_document(plan, prov)
        assert doc["paths"]["llama_server"] == SOURCE_BUILD

    def test_the_ordinary_case_stays_out_of_host_yml(self, linux_facts, empty_repo):
        # Detection found nothing, so the path is the shape's own answer.
        # Writing it down would bake a derived value into layer 3, which is
        # the trap `plan --fresh` exists for.
        plan, prov = planner.build(linux_facts, repo=empty_repo)
        assert plan["paths"]["llama_server"] == "/opt/llmstack/bin/llama-server"
        assert prov["paths.llama_server"].startswith("shape")
        assert "llama_server" not in planner.to_document(plan, prov).get("paths", {})

    def test_a_detected_path_that_matches_the_default_is_not_news(self,
                                                                 linux_facts,
                                                                 empty_repo):
        linux_facts["engines"]["llama_server"] = "/opt/llmstack/bin/llama-server"
        plan, prov = planner.build(linux_facts, repo=empty_repo)
        assert prov["paths.llama_server"].startswith("shape")

    def test_windows_keeps_its_extra_directory_and_extension(self, windows_facts,
                                                             empty_repo):
        plan, _ = planner.build(windows_facts, repo=empty_repo)
        assert plan["paths"]["llama_server"] == r"C:\llmstack\bin\llama\llama-server.exe"
        assert plan["paths"]["llama_swap"] == r"C:\llmstack\bin\llama-swap.exe"

    def test_planning_for_linux_from_windows_keeps_posix_separators(self,
                                                                   linux_facts,
                                                                   empty_repo):
        # The bench path is sliced off the end of the server path rather than
        # parsed, precisely so this cannot pick up a backslash when the plan
        # is built on a Windows workstation.
        linux_facts["engines"]["llama_server"] = SOURCE_BUILD
        plan, _ = planner.build(linux_facts, repo=empty_repo)
        assert "\\" not in plan["paths"]["llama_bench"]

    def test_host_yml_beats_detection(self, linux_facts, empty_repo):
        linux_facts["engines"]["llama_server"] = SOURCE_BUILD
        plan, prov = planner.build(
            linux_facts, repo=empty_repo,
            overrides={"paths": {"llama_server": "/srv/llama/llama-server"}})
        assert plan["paths"]["llama_server"] == "/srv/llama/llama-server"
        # ...and llama-bench still follows the DETECTED server, not the
        # overridden one: the override says where the binary is, and saying
        # nothing about bench should not silently move it too.
        assert plan["paths"]["llama_bench"] == \
            "/home/user/llama.cpp/build/bin/llama-bench"

    def test_the_env_file_carries_the_plans_path(self, linux_facts, empty_repo):
        linux_facts["engines"]["llama_server"] = SOURCE_BUILD
        ctx = ctx_for(linux_facts, empty_repo)
        ctx.plan["engine"]["kind"] = "llama.cpp"
        values = EnvFile()._values(ctx)
        assert values["LLMSTACK_LLAMA_SERVER"] == SOURCE_BUILD
        assert values["LLMSTACK_LLAMA_BENCH"].endswith("build/bin/llama-bench")

    def test_an_engine_outside_the_prefix_is_blocked_not_missing(self,
                                                                linux_facts,
                                                                empty_repo,
                                                                tmp_path):
        # Nothing is at the path, and unpacking a release into bin/ would not
        # put anything there either. An apply that runs clean and changes
        # nothing is the failure this avoids.
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        ctx.plan["engine"]["kind"] = "llama.cpp"
        ctx.plan["paths"]["llama_server"] = SOURCE_BUILD
        ctx.plan["paths"]["llama_swap"] = "/usr/local/bin/llama-swap"
        result = Engine().check(ctx)
        assert result.state == BLOCKED
        assert SOURCE_BUILD in result.detail

    def test_the_ordinary_missing_engine_is_still_missing(self, linux_facts,
                                                          empty_repo, tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        ctx.plan["engine"]["kind"] = "llama.cpp"
        assert Engine().check(ctx).state == MISSING


class TestTheEnvDiffSaysWhichWay:
    """`apply` rewrites this file from the plan.

    So a key the box has and the plan does not is one the apply will DROP,
    and calling that "update CF_ACCESS_AUD" describes the one outcome it is
    not. mac-laptop-1's dry run said exactly that.
    """

    def test_a_key_only_the_box_has_is_a_removal(self):
        assert _diff_keys("A=1\nB=2\n", "A=1\n") == [("remove", "B")]

    def test_a_key_only_the_plan_has_is_an_addition(self):
        assert _diff_keys("A=1\n", "A=1\nB=2\n") == [("add", "B")]

    def test_a_changed_value_is_an_update(self):
        assert _diff_keys("A=1\n", "A=2\n") == [("update", "A")]

    def test_an_empty_value_is_present_not_absent(self):
        # The mac-laptop-1 case exactly: CF_ACCESS_AUD= on the box, absent from the
        # plan. Empty and unset are the same to the gateway and not to a
        # person reading a dry run.
        assert _diff_keys("CF_ACCESS_AUD=\n", "") == [("remove", "CF_ACCESS_AUD")]

    def test_all_three_dialects_parse(self):
        assert _diff_keys("export A=1\n", "set A=2\n") == [("update", "A")]

    def test_agreement_is_silence(self):
        assert _diff_keys("export A=1\n# a comment\n", "A=1\n") == []


class TestTheHubIsNotAPeer:
    """fleet.yml's bind is right for peers and wrong for the one hub.

    Every peer's admin API is called by the hub over the tailnet, so 0.0.0.0
    is correct there. Nothing calls the hub as a peer; cloudflared reaches it
    on loopback. Applying the site default to it would have put the fleet's
    control plane on the tailnet in exchange for nothing.
    """

    def test_hermes_pins_loopback(self, repo):
        text = (repo / "hosts" / "hub" / "host.yml").read_text(encoding="utf-8")
        assert re.search(r"^\s+bind:\s*127\.0\.0\.1\s*$", text, re.M)

    def test_it_says_why(self, repo):
        text = (repo / "hosts" / "hub" / "host.yml").read_text(encoding="utf-8")
        assert "control plane" in text

    def test_no_peer_pins_it(self, repo):
        # If a second box ever needs loopback it is not a peer, and the
        # routing table should be told rather than the bind quietly changed.
        for host in sorted((repo / "hosts").glob("*/host.yml")):
            if host.parent.name == "hub":
                continue
            text = host.read_text(encoding="utf-8")
            assert not re.search(r"^\s+bind:\s*127\.0\.0\.1", text, re.M), host


class TestTheFilesWeShipAreLf:
    """Two files reported as differing on all 14 boxes were identical.

    public_domains_seed.json and requirements.txt had no .gitattributes rule,
    so a checkout on Windows got them CRLF -- 4717 bytes of difference in the
    first -- and a deploy from that workstation pushed CRLF copies over the
    LF ones the pi runner had installed. Content-comparing tools cannot tell
    that apart from a real change.
    """

    # Naming extensions one at a time is what let static/index.html through:
    # the list grew by one every time somebody found the next unruled type,
    # which is not a list, it is a backlog. The rule is a catch-all now, and
    # these two tuples are the whole of the exception.
    BINARY = (".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".gz", ".zip",
              ".woff", ".woff2")
    CRLF = (".cmd", ".bat")

    def test_gitattributes_defaults_everything_to_lf(self, repo):
        rules = repo / ".gitattributes"
        if not rules.is_file():
            # The distro containers mount a tarball of the sources, not a
            # checkout. A test that cannot see the repo has nothing to say
            # about it -- and must not fail the run for that, or the suite
            # stops being runnable in the one place the bootstrap is tested.
            pytest.skip("no .gitattributes here (not a checkout)")
        text = rules.read_text(encoding="utf-8")
        assert re.search(r"^\*\s+text=auto\s+eol=lf\s*$", text, re.M)

    def test_the_two_windows_types_are_still_crlf(self, repo):
        rules = repo / ".gitattributes"
        if not rules.is_file():
            pytest.skip("no .gitattributes here (not a checkout)")
        text = rules.read_text(encoding="utf-8")
        declared = dict(re.findall(r"^\*(\.\w+)\s+text\s+eol=(lf|crlf)\s*$",
                                   text, re.M))
        for ext in self.CRLF:
            # cmd.exe mis-parses a batch file with LF endings -- the same
            # reason fleetctl writes its generated wrappers CRLF.
            assert declared.get(ext) == "crlf", ext

    def test_no_tracked_file_holds_a_carriage_return(self, repo):
        # Runs on the machine the checkout is on, which is the only place the
        # bug is visible: on Linux CI these files are LF whatever the rules.
        try:
            tracked = subprocess.run(["git", "ls-files"], cwd=repo,
                                     capture_output=True, text=True, check=False)
        except OSError:
            # No git binary at all, which is the normal state of a bare
            # distro container -- FileNotFoundError, not a non-zero exit.
            pytest.skip("no git here")
        if tracked.returncode != 0:
            pytest.skip("not a git checkout")
        offenders = []
        for name in tracked.stdout.split("\n"):
            name = name.strip()
            if not name or name.endswith(self.BINARY + self.CRLF):
                continue
            try:
                if b"\r" in (repo / name).read_bytes():
                    offenders.append(name)
            except OSError:
                continue
        assert offenders == []


class TestUnreadableIsNotAbsent:
    """The most expensive lie a check can tell, told three times.

    Path.exists() answers False for a path it is not allowed to look at, and
    on this fleet that is the normal case rather than the corner one:
    /etc/llmstack is drwxrwx--- root:llmstack and /etc/sudoers.d is 0750
    root:root. So an unprivileged dry run on gpu-laptop-1 reported

        missing swap-config   no seed config
                . write /etc/llmstack/llama-swap.yaml

    about a file holding seven registered models -- to the one step whose
    answer to "there is no config" is to write an empty one. The env file
    already drew this distinction, because minting a fresh admin token over
    an unreadable file would have broken the hub's credential for the peer.
    It turns out the env file was not the only place that could see a 0640
    and call it nothing.
    """

    def unreadable(self):
        return lambda p: (None, "unreadable: Permission denied")

    # -- the primitive ------------------------------------------------------
    def test_absent_is_absent(self, linux_facts, empty_repo, tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        assert ctx.exists_state("/etc/llmstack/llama-swap.yaml") == (False, "absent")

    def test_present_is_present(self, linux_facts, empty_repo, tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        target = ctx.path("/etc/llmstack/llama-swap.yaml")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("models: {}\n", encoding="utf-8")
        assert ctx.exists_state("/etc/llmstack/llama-swap.yaml") == (True, "")

    def test_unreadable_is_neither(self, linux_facts, empty_repo, tmp_path,
                                   monkeypatch):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)

        def denied(self):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(Path, "stat", denied)
        there, why = ctx.exists_state("/etc/llmstack/llama-swap.yaml")
        assert there is None
        assert why == "unreadable: Permission denied"

    # -- the step that would have wiped gpu-laptop-1 ---------------------------
    def test_the_swap_config_is_blocked_not_missing(self, linux_facts,
                                                    empty_repo, tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        ctx.exists_state = self.unreadable()
        result = SwapConfig().check(ctx)
        assert result.state == BLOCKED
        # And it says what is at stake, because "blocked" alone reads as
        # bureaucracy rather than as a near miss.
        assert "every model" in result.detail
        assert result.plan == []

    def test_the_swap_config_refuses_to_write_blind(self, linux_facts,
                                                    empty_repo, tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        ctx.exists_state = self.unreadable()
        with pytest.raises(RuntimeError, match="refusing"):
            SwapConfig().apply(ctx)
        assert not ctx.path(ctx.plan["paths"]["swap_config"]).exists()

    def test_a_genuinely_absent_config_is_still_seeded(self, linux_facts,
                                                       empty_repo, tmp_path):
        # The fix must not make a fresh box unprovisionable.
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        ctx.plan["engine"]["kind"] = "llama.cpp"
        assert SwapConfig().check(ctx).state == MISSING
        SwapConfig().apply(ctx)
        assert ctx.path(ctx.plan["paths"]["swap_config"]).is_file()

    def test_an_existing_config_is_still_left_alone(self, linux_facts,
                                                    empty_repo, tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        target = ctx.path(ctx.plan["paths"]["swap_config"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("models:\n  qwen3: {}\n", encoding="utf-8")
        assert SwapConfig().check(ctx).state == OK
        SwapConfig().apply(ctx)
        assert "qwen3" in target.read_text(encoding="utf-8")

    # -- the other two ------------------------------------------------------
    def test_grants_is_blocked_not_missing(self, linux_facts, empty_repo,
                                           tmp_path):
        # The step skips itself when grants.sh is not in the checkout, so the
        # scratch repo needs one before this can say anything.
        script = empty_repo / "hosts" / "linux" / "grants.sh"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        ctx.exists_state = self.unreadable()
        assert Grants().check(ctx).state == BLOCKED

    def test_power_is_blocked_not_missing(self, linux_facts, empty_repo,
                                          tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        ctx.read_state = self.unreadable()
        assert Power().check(ctx).state == BLOCKED

    def test_wrappers_are_blocked_not_drifted(self, darwin_facts, empty_repo,
                                              tmp_path):
        ctx = ctx_for(darwin_facts, empty_repo, tmp_path)
        ctx.read_state = self.unreadable()
        assert Wrappers().check(ctx).state == BLOCKED

    def test_a_missing_logind_dropin_is_still_missing(self, linux_facts,
                                                      empty_repo, tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        assert Power().check(ctx).state == MISSING


class TestTheEnvFileHasValuesThePlanNeverHeardOf:
    """A privileged dry run on hub proposed removing eight of them.

        remove HF_TOKEN
        remove LLMSTACK_PUBLIC_INTAKE_TOKEN
        remove LLMSTACK_SMTP_FROM / HOST / PASSWORD / PORT / TLS / USER

    apply rewrites this file from the plan, and the plan cannot express any
    of those -- six are the mail credentials hosts/hub/daily-brief sends
    with, and one is the token the public intake path authenticates on.

    The admin token already had a special case for exactly this shape of
    problem. It was not the only credential in the file, and a rule that
    names one key does not generalise: LLMSTACK_SMTP_PASSWORD looks exactly
    like a key fleetctl writes. So the rule is ownership -- fleetctl keeps
    what it renders and carries the rest across untouched.
    """

    HERMES_EXTRAS = {
        "HF_TOKEN": "hf_xxxxxxxxxxxxxxxxxxxx",
        "LLMSTACK_PUBLIC_INTAKE_TOKEN": "intake-secret",
        "LLMSTACK_SMTP_HOST": "smtp.example.com",
        "LLMSTACK_SMTP_PORT": "587",
        "LLMSTACK_SMTP_TLS": "1",
        "LLMSTACK_SMTP_USER": "brief@example.com",
        "LLMSTACK_SMTP_PASSWORD": "hunter2",
        "LLMSTACK_SMTP_FROM": "brief@example.com",
    }

    def seeded(self, ctx):
        """An env file shaped like the one on hub: fleetctl's own values,
        the admin token, and eight things fleetctl never wrote."""
        step = EnvFile()
        text = step._render(ctx, "the-hub-holds-this-one")
        extra = "".join(f"{k}={v}\n" for k, v in self.HERMES_EXTRAS.items())
        path = ctx.path(ctx.plan["paths"]["envfile"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + extra, encoding="utf-8")
        return step, path

    def test_they_are_recognised_as_not_ours(self, linux_facts, empty_repo,
                                             tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        step, path = self.seeded(ctx)
        found = step.foreign(ctx, path.read_text(encoding="utf-8"))
        assert set(found) == set(self.HERMES_EXTRAS)

    def test_the_admin_token_is_still_ours(self, linux_facts, empty_repo,
                                          tmp_path):
        # It is rendered by fleetctl, so it must not come back through the
        # carry-across path and get written twice.
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        step, path = self.seeded(ctx)
        assert "LLMSTACK_ADMIN_TOKEN" not in step.foreign(
            ctx, path.read_text(encoding="utf-8"))

    def test_an_apply_keeps_every_one_of_them(self, linux_facts, empty_repo,
                                              tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        step, path = self.seeded(ctx)
        step.apply(ctx)
        after = path.read_text(encoding="utf-8")
        for key, value in self.HERMES_EXTRAS.items():
            assert f"{key}={value}" in after, key

    def test_an_apply_keeps_the_admin_token_too(self, linux_facts, empty_repo,
                                                tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        step, path = self.seeded(ctx)
        step.apply(ctx)
        assert "LLMSTACK_ADMIN_TOKEN=the-hub-holds-this-one" in \
            path.read_text(encoding="utf-8")

    def test_each_key_is_written_exactly_once(self, linux_facts, empty_repo,
                                              tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        step, path = self.seeded(ctx)
        step.apply(ctx)
        keys = re.findall(r"^(?:set |export )?([A-Z][A-Z0-9_]*)=",
                          path.read_text(encoding="utf-8"), re.M)
        assert len(keys) == len(set(keys))

    def test_the_check_says_they_are_kept_rather_than_removed(self, linux_facts,
                                                              empty_repo,
                                                              tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        step, path = self.seeded(ctx)
        result = step.check(ctx)
        assert not any(line.startswith("remove") for line in result.plan)

    def test_it_settles(self, linux_facts, empty_repo, tmp_path):
        # The one that would have caught this being half-done: apply, then
        # check, and the answer has to be `ok`. A carried key that is not
        # rendered the same way it was read leaves the file permanently
        # drifted and every run reporting work it has already done.
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        step, _ = self.seeded(ctx)
        step.apply(ctx)
        assert step.check(ctx).state == OK

    def test_a_fresh_box_carries_nothing(self, linux_facts, empty_repo,
                                         tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        assert EnvFile().foreign(ctx, None) == {}

    def test_windows_carries_them_in_its_own_dialect(self, windows_facts,
                                                     empty_repo, tmp_path):
        ctx = ctx_for(windows_facts, empty_repo, tmp_path)
        step = EnvFile()
        text = step._render(ctx, "tok", {"HF_TOKEN": "hf_x"})
        assert "set HF_TOKEN=hf_x" in text


class TestAFirewallYouCannotSeeIsStillAFirewall:
    """`ufw status` unprivileged exits "You need to be root to run this".

    The check read a non-zero exit as "the tool is not installed" and printed
    `skipped   no firewalld or ufw here` about hub and server-1, both of
    which are running ufw -- and skipped is the state that means "there is
    nothing to do here", which is how a firewall rule the fleet needs stays
    unnoticed. Under sudo the same check on the same boxes says
    `missing   ufw is running`.
    """

    def refusing(self, text):
        return lambda argv, timeout=60: SimpleNamespace(
            returncode=1, stdout="", stderr=text)

    def test_a_refusal_is_blocked_not_skipped(self, linux_facts, empty_repo,
                                              tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        ctx.probe = self.refusing("ERROR: You need to be root to run this script")
        assert Firewall().check(ctx).state == BLOCKED

    def test_a_tool_that_is_not_installed_is_still_skipped(self, linux_facts,
                                                          empty_repo, tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        ctx.probe = lambda argv, timeout=60: None
        result = Firewall().check(ctx)
        assert result.state == SKIPPED
        assert "no firewalld or ufw" in result.detail

    def test_a_stopped_firewall_is_not_a_refusal(self, linux_facts, empty_repo,
                                                 tmp_path):
        # `firewall-cmd --state` exits 252 with "not running" when firewalld
        # is installed and stopped. That is an answer, not a refusal.
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        ctx.probe = self.refusing("not running")
        assert Firewall().check(ctx).state == SKIPPED

    def test_a_running_firewall_still_wants_a_rule(self, linux_facts, empty_repo,
                                                   tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        ctx.probe = lambda argv, timeout=60: SimpleNamespace(
            returncode=0, stdout="running", stderr="")
        assert Firewall().check(ctx).state == MISSING


class TestAnEmptyStoreIsNotTheStore:
    """cpu-box-1 has /usr/share/ollama/.ollama/models and keeps nothing in it.

    Ollama's own installer creates that directory whether or not anything is
    pulled into it, so "the first candidate that exists" chose an empty
    directory while the box's weights sat in /var/lib/llmstack/models -- the
    exact failure the code's own comment said it was avoiding.
    """

    def test_a_full_store_beats_an_empty_one(self, tmp_path):
        empty = tmp_path / "system"
        full = tmp_path / "home"
        empty.mkdir()
        full.mkdir()
        (full / "blobs").mkdir()
        assert facts_mod.ollama_store(str(empty), str(full)) == str(full)

    def test_order_still_decides_between_two_full_ones(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        for d in (a, b):
            d.mkdir()
            (d / "blobs").mkdir()
        assert facts_mod.ollama_store(str(a), str(b)) == str(a)

    def test_an_empty_store_is_still_better_than_nothing(self, tmp_path):
        # A box with Ollama installed that has pulled nothing yet still has a
        # store, and the plan should name it rather than leaving it unset.
        empty = tmp_path / "system"
        empty.mkdir()
        assert facts_mod.ollama_store(str(tmp_path / "gone"),
                                      str(empty)) == str(empty)

    def test_no_store_at_all_is_none(self, tmp_path):
        assert facts_mod.ollama_store(str(tmp_path / "nope")) is None


class TestTheFleetNameIsNotTheSshAlias:
    """`--host mac-desktop` planned a box from scratch and said nothing.

    The fleet name is the key everything is looked up by -- the spec sheet,
    the routing table, the metering, the hub's peer list. Planning silently
    from the shape for a name with no host.yml meant a harness that passed
    the ssh alias got a plan that disagreed with the box on two values, both
    of which were correct on the box.
    """

    def test_an_unknown_name_says_so(self, capsys):
        cli._warn_unknown_host(SimpleNamespace(host="mac-desktop"),
                              {"host": {"name": "mac-desktop"}})
        assert "no hosts/mac-desktop/host.yml" in capsys.readouterr().err

    def test_it_offers_the_near_miss(self, capsys):
        cli._warn_unknown_host(SimpleNamespace(host="mac-desktop"),
                              {"host": {"name": "mac-desktop"}})
        assert "--host mac-desktop-1" in capsys.readouterr().err

    def test_a_genuinely_new_box_gets_the_list(self, capsys):
        cli._warn_unknown_host(SimpleNamespace(host="brand-new-box"),
                              {"host": {"name": "brand-new-box"}})
        err = capsys.readouterr().err
        assert "known:" in err and "hub" in err

    def test_a_known_box_is_silent(self, capsys):
        cli._warn_unknown_host(SimpleNamespace(host="hub"),
                              {"host": {"name": "hub"}})
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""

    def test_nothing_lands_on_stdout_where_json_goes(self, capsys):
        """The reason this note is on stderr at all.

        `plan --json` is parsed by whatever ran it -- the CI step does
        exactly that -- and a runner has no host.yml for its own hostname, so
        this note fired on every one of them. Three hosted jobs failed with
        `JSONDecodeError: Expecting value: line 1 column 3`, which is two
        spaces and an `n`.
        """
        cli._warn_unknown_host(SimpleNamespace(host="brand-new-box"),
                              {"host": {"name": "brand-new-box"}})
        assert capsys.readouterr().out == ""


class TestTheHubIsNotAnApiClient:
    """`register` handed the hub the URL meant for API clients.

    Two different addresses for two different callers. `public_api_url` is
    what the dashboard gives to people with keys; the hub calls
    `<url>/admin/api/...` with the peer's admin token. hub, mac-desktop-1 and
    gpu-laptop-1 all publish a *.example.com hostname with Cloudflare Access
    in front of it, and an Access-gated hostname answers a bearer request
    with a 302 to an SSO page -- which the hub reads as the peer being
    unreachable, dropping its models from the routing table.

    Every peer in the live hub list that works is a 100.x address. The rest
    are older registrations by bare name or DHCP lease, which is the same
    brittleness deploy-gateway.sh's address table was fixed for.
    """

    def entry(self, capsys, facts, repo, tmp_path, **plan_over):
        ctx = ctx_for(facts, repo, tmp_path)
        envf = ctx.path(ctx.plan["paths"]["envfile"])
        envf.parent.mkdir(parents=True, exist_ok=True)
        envf.write_text("LLMSTACK_ADMIN_TOKEN=peer-token\n", encoding="utf-8")
        for dotted, value in plan_over.items():
            sect, _, key = dotted.partition("__")
            ctx.plan[sect][key] = value

        class Args:
            facts = None
            host = ctx.plan["host"]["name"]
            root = str(tmp_path / "sandbox")
            hub = "hub"
            set = []
            fresh = False
            quick = False

        # cmd_register re-gathers, so hand it the plan we just shaped.
        import fleetctl.cli as cli_mod
        def monkeyed(a):
            return (facts, ctx.plan, {})
        real, cli_mod._gather = cli_mod._gather, monkeyed
        try:
            cli_mod.cmd_register(Args())
        finally:
            cli_mod._gather = real
        out = capsys.readouterr().out
        return json.loads(out[out.index("{"): out.rindex("}") + 1])

    def test_the_hub_gets_the_tailnet_address(self, linux_facts, empty_repo,
                                              tmp_path, capsys):
        e = self.entry(capsys, linux_facts, empty_repo, tmp_path,
                       network__tailnet_ip="100.64.0.36",
                       network__public_api_url="https://gpu-laptop-1.example.com/v1")
        assert e["url"] == "http://100.64.0.36:8080"

    def test_not_the_access_gated_hostname(self, linux_facts, empty_repo,
                                           tmp_path, capsys):
        e = self.entry(capsys, linux_facts, empty_repo, tmp_path,
                       network__tailnet_ip="100.64.0.36",
                       network__public_api_url="https://gpu-laptop-1.example.com/v1")
        assert "example.com" not in e["url"]

    def test_a_box_with_no_tailnet_address_still_registers(self, linux_facts,
                                                           empty_repo, tmp_path,
                                                           capsys):
        # ...and is told why that is a worse answer.
        e = self.entry(capsys, linux_facts, empty_repo, tmp_path,
                       network__tailnet_ip=None,
                       network__public_api_url="https://box.example.com/v1")
        assert e["url"] == "https://box.example.com"

    def test_the_token_is_the_peers_own(self, linux_facts, empty_repo, tmp_path,
                                        capsys):
        e = self.entry(capsys, linux_facts, empty_repo, tmp_path,
                       network__tailnet_ip="100.64.0.36")
        assert e["token"] == "peer-token"
