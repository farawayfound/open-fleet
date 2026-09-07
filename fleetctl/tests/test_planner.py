"""Four layers, and which one won.

The layering is the whole design, so these tests are mostly about
provenance: not just that a value is right, but that it came from the layer
it should have. A plan that happens to be correct because two layers agree
is a plan that breaks the first time they stop agreeing.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from fleetctl import hostfile, planner, shapes


def build(facts, repo, **kw):
    return planner.build(facts, repo=repo, **kw)


class TestShapeLayer:
    def test_linux_gets_the_linux_layout(self, linux_facts, empty_repo):
        plan, prov = build(linux_facts, empty_repo)
        assert plan["paths"]["prefix"] == "/opt/llmstack"
        assert plan["paths"]["state"] == "/var/lib/llmstack"
        assert plan["paths"]["etc"] == "/etc/llmstack"
        assert plan["paths"]["envfile"] == "/etc/llmstack/gateway.env"
        assert plan["service"]["user"] == "llmstack"

    def test_windows_gets_backslashes_and_a_cmd_env_file(self, windows_facts,
                                                         empty_repo):
        plan, _ = build(windows_facts, empty_repo)
        assert plan["paths"]["prefix"] == r"C:\llmstack"
        assert plan["paths"]["gateway"] == r"C:\llmstack\gateway"
        # .cmd because the supervisor is a scheduled task running a batch
        # wrapper, and `call gateway.env.cmd` is how a batch file gets vars.
        assert plan["paths"]["envfile"].endswith(".cmd")
        assert plan["service"]["user"] == "SYSTEM"

    def test_macos_puts_etc_and_state_in_one_place(self, darwin_facts, empty_repo):
        plan, _ = build(darwin_facts, empty_repo)
        assert plan["paths"]["etc"] == plan["paths"]["state"]
        assert plan["paths"]["prefix"].endswith("llmstack")
        assert plan["service"]["user"] is None

    def test_an_unknown_os_family_is_an_error_not_a_guess(self, linux_facts,
                                                          empty_repo):
        linux_facts["os"]["family"] = "haiku"
        with pytest.raises(planner.PlanError) as exc:
            build(linux_facts, empty_repo)
        assert "haiku" in str(exc.value)


class TestSiteLayer:
    def test_fleet_yml_overrides_the_shape(self, linux_facts, empty_repo):
        (empty_repo / "fleet.yml").write_text(
            "network:\n  port: 9000\naccess:\n  admin_emails: [a@b.c]\n",
            encoding="utf-8")
        plan, prov = build(linux_facts, empty_repo)
        assert plan["network"]["port"] == 9000
        assert prov["network.port"].startswith("site")
        assert plan["access"]["admin_emails"] == ["a@b.c"]

    def test_no_fleet_yml_is_fine(self, linux_facts, empty_repo):
        plan, prov = build(linux_facts, empty_repo)
        assert plan["network"]["port"] == 8080
        assert prov["network.port"].startswith("shape")


class TestHostLayer:
    def _host(self, repo, name, body):
        p = repo / "hosts" / name / "host.yml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")

    def test_host_yml_beats_the_site(self, linux_facts, empty_repo):
        (empty_repo / "fleet.yml").write_text("network:\n  port: 9000\n",
                                              encoding="utf-8")
        self._host(empty_repo, "testbox", "network:\n  port: 9100\n")
        plan, prov = build(linux_facts, empty_repo)
        assert plan["network"]["port"] == 9100
        assert prov["network.port"].startswith("host.yml")

    def test_the_command_line_beats_host_yml(self, linux_facts, empty_repo):
        self._host(empty_repo, "testbox", "network:\n  port: 9100\n")
        plan, prov = build(linux_facts, empty_repo,
                           overrides={"network": {"port": 9200}})
        assert plan["network"]["port"] == 9200
        assert prov["network.port"].startswith("override")

    def test_fresh_ignores_the_file_it_is_about_to_replace(self, linux_facts,
                                                           empty_repo):
        """`plan --write` reads host.yml as an input, which is what keeps hand
        edits -- and also what makes a value that was WRONG when generated
        survive every regeneration. Improving detection is exactly when the
        old answer should be dropped."""
        self._host(empty_repo, "testbox", "network:\n  port: 9100\n")
        assert build(linux_facts, empty_repo)[0]["network"]["port"] == 9100
        plan, prov = build(linux_facts, empty_repo, fresh=True)
        assert plan["network"]["port"] == 8080
        assert "ignored" in prov.get("network.bind", "") or \
            prov["network.port"].startswith("shape")

    def test_fresh_still_honours_the_site_and_the_command_line(self, linux_facts,
                                                               empty_repo):
        (empty_repo / "fleet.yml").write_text("network:\n  port: 9000\n",
                                              encoding="utf-8")
        self._host(empty_repo, "testbox", "network:\n  port: 9100\n")
        plan, _ = build(linux_facts, empty_repo, fresh=True)
        assert plan["network"]["port"] == 9000
        plan, _ = build(linux_facts, empty_repo, fresh=True,
                        overrides={"network": {"port": 9200}})
        assert plan["network"]["port"] == 9200

    def test_an_unknown_key_is_rejected_rather_than_ignored(self, linux_facts,
                                                            empty_repo):
        """A silently-ignored key is a value somebody believes is in force
        and is not."""
        self._host(empty_repo, "testbox", "network:\n  prot: 9100\n")
        with pytest.raises(planner.PlanError) as exc:
            build(linux_facts, empty_repo)
        assert "network.'prot'" in str(exc.value) or "prot" in str(exc.value)

    def test_an_unknown_section_is_rejected(self, linux_facts, empty_repo):
        self._host(empty_repo, "testbox", "netwrok:\n  port: 1\n")
        with pytest.raises(planner.PlanError):
            build(linux_facts, empty_repo)

    def test_moving_the_prefix_moves_everything_derived_from_it(
            self, linux_facts, empty_repo):
        """The eight paths under a prefix follow it. A plan that installed the
        venv beside an application it did not install would be worse than one
        that refused."""
        self._host(empty_repo, "testbox", "paths:\n  prefix: /srv/llm\n")
        plan, prov = build(linux_facts, empty_repo)
        assert plan["paths"]["gateway"] == "/srv/llm/gateway"
        assert plan["paths"]["venv"] == "/srv/llm/venv"
        assert "moved" in prov["paths.gateway"]

    def test_a_path_set_by_hand_is_not_moved_by_the_prefix(self, linux_facts,
                                                           empty_repo):
        """An external weights disk was stated on purpose."""
        self._host(empty_repo, "testbox",
                   "paths:\n  prefix: /srv/llm\n  models: /mnt/ssd/models\n")
        plan, _ = build(linux_facts, empty_repo)
        assert plan["paths"]["models"] == "/mnt/ssd/models"


class TestDetection:
    def test_the_tailnet_name_is_the_fleet_name_by_default(self, linux_facts,
                                                           empty_repo):
        linux_facts["tailscale"]["name"] = "gpu-laptop-1"
        plan, _ = build(linux_facts, empty_repo)
        assert plan["host"]["name"] == "gpu-laptop-1"

    def test_falling_back_to_the_os_hostname_lowercased(self, linux_facts,
                                                        empty_repo):
        linux_facts["tailscale"] = {"present": False, "ipv4": None, "name": None}
        linux_facts["hostname"] = "apu-tablet-2.local"
        plan, _ = build(linux_facts, empty_repo)
        assert plan["host"]["name"] == "apu-tablet-2"

    def test_the_public_url_is_derived_from_the_tailnet_address(self, linux_facts,
                                                                empty_repo):
        plan, prov = build(linux_facts, empty_repo)
        assert plan["network"]["public_api_url"] == "http://100.64.0.1:8080/v1"
        assert "tailnet" in prov["network.public_api_url"]

    def test_no_tailnet_means_no_derived_url_and_a_complaint(self, linux_facts,
                                                             empty_repo):
        linux_facts["tailscale"] = {"present": False, "ipv4": None, "name": None}
        plan, _ = build(linux_facts, empty_repo)
        assert plan["network"]["public_api_url"] is None
        assert any("public_api_url" in p for p in planner.validate(plan))

    def test_an_engine_already_installed_is_not_replaced(self, linux_facts,
                                                         empty_repo):
        """A re-provision must not swap a working Ollama box onto llama.cpp
        behind its operator's back."""
        linux_facts["engines"]["ollama"] = "/usr/local/bin/ollama"
        plan, prov = build(linux_facts, empty_repo)
        assert plan["engine"]["kind"] == "ollama"
        assert "already installed" in prov["engine.kind"]

    def test_ollama_takes_its_own_model_store(self, linux_facts, empty_repo):
        linux_facts["engines"]["ollama"] = "/usr/local/bin/ollama"
        plan, _ = build(linux_facts, empty_repo)
        assert plan["paths"]["models"].endswith(".ollama/models".replace("/", "/"))
        assert plan["engine"]["models_from_upstream"] is True
        assert plan["network"]["upstream"].endswith(":11434")

    def test_a_stated_models_dir_survives_the_ollama_default(self, linux_facts,
                                                             empty_repo):
        linux_facts["engines"]["ollama"] = "/usr/local/bin/ollama"
        p = empty_repo / "hosts" / "testbox" / "host.yml"
        p.parent.mkdir(parents=True)
        p.write_text("paths:\n  models: /mnt/weights\n", encoding="utf-8")
        plan, _ = build(linux_facts, empty_repo)
        assert plan["paths"]["models"] == "/mnt/weights"


class TestCrossPlatformPlanning:
    """Planning for a box that is not the one you are sitting at.

    Every host.yml in this repo was generated that way -- from `detect --json`
    run over ssh -- and the first attempt put `C:/Users/user/llmstack` into
    a macOS plan and `C:/Users/user/.ollama/models` into a Linux one, because
    the planner reached for `Path.home()` on the machine doing the planning.
    The dry runs on mac-laptop-1 and cpu-box-1 duly offered to create them.
    """

    def test_the_home_comes_from_the_facts_not_from_here(self, darwin_facts,
                                                         empty_repo):
        darwin_facts["home"] = "/Users/user"
        plan, _ = build(darwin_facts, empty_repo)
        assert plan["paths"]["prefix"] == "/Users/user/llmstack"
        assert plan["paths"]["state"] == "/Users/user/llmstack/state"

    def test_an_ollama_store_follows_the_target_home(self, linux_facts,
                                                     empty_repo):
        linux_facts["home"] = "/home/someone"
        linux_facts["engines"]["ollama"] = "/usr/local/bin/ollama"
        plan, _ = build(linux_facts, empty_repo)
        assert plan["paths"]["models"] == "/home/someone/.ollama/models"

    def test_no_home_is_not_an_invented_one(self, linux_facts, empty_repo):
        """The shape default stands rather than a path built from whatever
        `Path.home()` says on the planning machine. `detect` on the box
        itself fills it in later.

        This test passed on a Windows workstation and failed in a Linux
        container, because the fallback it was testing only fired when the
        planning machine's family MATCHED the target's -- so on Windows it
        never ran and on Linux-as-root it produced `/root/.ollama/models`.
        The fallback is gone; the facts are the only source.
        """
        linux_facts.pop("home", None)
        linux_facts["engines"]["ollama"] = "/usr/local/bin/ollama"
        linux_facts["engines"].pop("ollama_models", None)
        plan, _ = build(linux_facts, empty_repo)
        assert plan["paths"]["models"] == "/var/lib/llmstack/models"

    def test_a_user_level_shape_with_no_home_refuses_rather_than_guesses(
            self, darwin_facts, empty_repo, monkeypatch):
        """`~/llmstack` is not a path anything expands -- not systemd, not
        cmd, not Path(). Better a complaint than a directory called `~`.

        The running platform is forced, because otherwise this test asserts
        something different depending on which runner picks it up.
        `default_home()` answers with THIS machine's home when it is the kind
        of machine being planned for -- correct, because install.sh runs on
        the box it is provisioning -- and "" otherwise. So on a macOS runner,
        planning for darwin with no stated home quietly used the runner's own
        home and there was no complaint to find. It passed on Linux and
        Windows and failed on macOS, which is not a test, it is a coin toss.
        """
        monkeypatch.setattr(shapes.platform, "system", lambda: "Linux")
        darwin_facts.pop("home", None)
        plan, _ = build(darwin_facts, empty_repo)
        problems = planner.validate(plan)
        assert any("home directory of the target box is not known" in p
                   for p in problems), problems

    def test_the_box_being_provisioned_is_allowed_to_answer_for_itself(
            self, darwin_facts, empty_repo, monkeypatch):
        """The other half, which is why the fallback exists at all.

        `./install.sh` runs ON the machine it is provisioning, so when the
        target's family is this machine's family, this machine's home is not
        a guess -- it is the answer.
        """
        monkeypatch.setattr(shapes.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(shapes.Path, "home", staticmethod(
            lambda: pathlib.PurePosixPath("/Users/whoever")))
        darwin_facts.pop("home", None)
        plan, _ = build(darwin_facts, empty_repo)
        assert plan["paths"]["prefix"] == "/Users/whoever/llmstack"
        assert not [p for p in planner.validate(plan)
                    if "home directory" in p]

    @pytest.mark.parametrize("which", ["linux", "darwin", "windows"])
    def test_no_plan_ever_carries_another_platforms_path(self, request,
                                                          empty_repo, which):
        facts = request.getfixturevalue(f"{which}_facts")
        plan, prov = build(facts, empty_repo)
        text = "\n".join(f"{k}: {v}" for k, v in plan["paths"].items())
        if which == "windows":
            assert "/home/" not in text and "/Users/" not in text
        else:
            assert "C:" not in text, text

    def test_the_repo_s_own_plans_carry_no_foreign_paths(self, repo):
        """The regression test proper: no committed host.yml may contain a
        path belonging to a different operating system."""
        for path in sorted(repo.glob("hosts/*/host.yml")):
            doc = hostfile.load(path)
            paths = (doc.get("paths") or {})
            for key, value in paths.items():
                value = str(value)
                if doc["platform"]["os"] == "windows":
                    assert not value.startswith("/home/"), f"{path}: {key}={value}"
                    assert not value.startswith("/Users/"), f"{path}: {key}={value}"
                else:
                    assert not re.match(r"^[A-Za-z]:", value), f"{path}: {key}={value}"


class TestKlass:
    @pytest.mark.parametrize("vram,ram,want", [
        (96, 128, "big"), (32, 64, "big"), (16, 32, "gpu"), (8, 32, "gpu"),
        (0, 32, "small"), (0, 8, "fallback"),
    ])
    def test_derived_from_the_hardware(self, linux_facts, empty_repo,
                                       vram, ram, want):
        linux_facts["vram_gb"], linux_facts["ram_gb"] = vram, ram
        plan, _ = build(linux_facts, empty_repo)
        assert plan["host"]["klass"] == want

    def test_a_hub_is_a_hub_whatever_is_in_it(self, linux_facts, empty_repo):
        plan, _ = build(linux_facts, empty_repo,
                        overrides={"host": {"role": "hub"}})
        assert plan["host"]["klass"] == "hub"


class TestValidation:
    def test_reports_every_problem_at_once(self, linux_facts, empty_repo):
        plan, _ = build(linux_facts, empty_repo)
        plan["network"]["public_api_url"] = None
        plan["network"]["port"] = None
        plan["host"]["role"] = "bystander"
        problems = planner.validate(plan)
        assert len(problems) >= 3
        assert any("public_api_url" in p for p in problems)
        assert any("port" in p for p in problems)
        assert any("role" in p for p in problems)

    def test_a_boolean_is_not_an_integer(self, linux_facts, empty_repo):
        """bool is an int subclass in Python; `port: true` must not pass a
        check that the port is a number."""
        plan, _ = build(linux_facts, empty_repo)
        plan["network"]["port"] = True
        assert any("boolean" in p for p in planner.validate(plan))

    def test_a_port_out_of_range(self, linux_facts, empty_repo):
        plan, _ = build(linux_facts, empty_repo)
        plan["network"]["port"] = 99999
        assert any("port number" in p for p in planner.validate(plan))

    def test_llama_cpp_without_a_backend_is_incomplete(self, linux_facts,
                                                       empty_repo):
        plan, _ = build(linux_facts, empty_repo)
        plan["engine"]["kind"] = "llama.cpp"
        plan["engine"]["backend"] = None
        assert any("engine.backend" in p for p in planner.validate(plan))

    def test_a_complete_plan_has_no_problems(self, linux_facts, empty_repo):
        plan, _ = build(linux_facts, empty_repo)
        assert planner.validate(plan) == []


class TestWriting:
    def test_only_what_the_lower_layers_will_not_supply_again(
            self, linux_facts, empty_repo, tmp_path):
        (empty_repo / "fleet.yml").write_text(
            "access:\n  cf_team_domain: x.example\n  admin_emails: [a@b.c]\n",
            encoding="utf-8")
        plan, prov = build(linux_facts, empty_repo)
        out = tmp_path / "host.yml"
        planner.write(plan, out, prov)
        doc = hostfile.load(out)
        # The site's values are NOT written back into the per-box file --
        # freezing a fleet-wide change into fourteen copies is the duplication
        # this whole thing removes.
        assert "access" not in doc
        assert "bind" not in doc.get("network", {})
        assert "port" not in doc.get("network", {})
        # What the box IS, is kept.
        assert doc["host"]["name"] == "testbox"
        assert doc["network"]["tailnet_ip"] == "100.64.0.1"

    def test_default_paths_are_not_written_but_moved_ones_are(
            self, linux_facts, empty_repo, tmp_path):
        plan, prov = build(linux_facts, empty_repo)
        planner.write(plan, tmp_path / "a.yml", prov)
        assert "prefix" not in hostfile.load(tmp_path / "a.yml").get("paths", {})

        plan["paths"]["models"] = "/mnt/ssd/models"
        planner.write(plan, tmp_path / "b.yml", prov)
        assert hostfile.load(tmp_path / "b.yml")["paths"]["models"] == "/mnt/ssd/models"

    def test_what_is_written_reads_back_into_an_equivalent_plan(
            self, linux_facts, empty_repo, tmp_path):
        """The round trip that matters: generate, write, read, and get the
        same plan. Without it, `plan --write` would be a one-way door."""
        plan, prov = build(linux_facts, empty_repo)
        target = empty_repo / "hosts" / "testbox" / "host.yml"
        planner.write(plan, target, prov)
        again, _ = build(linux_facts, empty_repo)
        assert again == plan

    def test_a_generated_file_carries_its_explanations(self, linux_facts,
                                                       empty_repo, tmp_path):
        plan, prov = build(linux_facts, empty_repo)
        text = planner.write(plan, tmp_path / "host.yml", prov)
        assert "# fleet host plan" in text
        assert "MagicDNS" in text          # the tailnet_ip note
        assert "302 to an SSO page" in text  # the public_api_url note


class TestEveryPlanInTheRepo:
    """The fifteen real ones. This is the test that would have caught a
    hand-edit that broke a box, before a deploy did."""

    def _plans(self, repo):
        return sorted(repo.glob("hosts/*/host.yml"))

    def test_there_are_some(self, repo):
        assert len(self._plans(repo)) >= 15

    def test_each_parses(self, repo):
        for path in self._plans(repo):
            hostfile.load(path)

    def test_each_uses_only_known_keys(self, repo):
        for path in self._plans(repo):
            doc = hostfile.load(path)
            for sect, body in doc.items():
                assert sect in planner.SCHEMA, f"{path}: unknown section {sect}"
                for key in (body or {}):
                    assert key in planner.SCHEMA[sect], f"{path}: unknown {sect}.{key}"

    def test_each_names_a_shape_and_an_engine_that_exist(self, repo):
        for path in self._plans(repo):
            doc = hostfile.load(path)
            assert doc["platform"]["os"] in shapes.SHAPES, path
            assert doc["engine"]["kind"] in shapes.ENGINES, path

    def test_each_validates_when_replanned(self, repo, linux_facts):
        """Rebuild every host's plan from its own file plus a minimal fact
        set, and require it to be complete. A host.yml that cannot produce an
        appliable plan is a box that cannot be provisioned."""
        for path in self._plans(repo):
            doc = hostfile.load(path)
            facts = dict(linux_facts)
            facts["os"] = dict(linux_facts["os"], family=doc["platform"]["os"],
                               arch=doc["platform"].get("arch") or "x86_64",
                               package_manager=doc["platform"]["package_manager"])
            facts["service_manager"] = doc["platform"]["service"]
            facts["tailscale"] = {"present": False, "ipv4": None, "name": None}
            facts["engines"] = {}
            plan, _ = planner.build(facts, repo=repo, name=path.parent.name)
            assert planner.validate(plan) == [], f"{path}: {planner.validate(plan)}"

    def test_the_fleet_names_are_unique(self, repo):
        """Two boxes with one name is the whole fleet's lookup key colliding
        -- specs, routing rank, usage metering and the hub's peer list."""
        names = [hostfile.load(p)["host"]["name"] for p in self._plans(repo)]
        assert len(names) == len(set(names)), sorted(names)

    def test_no_two_boxes_claim_one_tailnet_address(self, repo):
        addrs = {}
        for path in self._plans(repo):
            ip = (hostfile.load(path).get("network") or {}).get("tailnet_ip")
            if ip:
                assert ip not in addrs, f"{path} and {addrs[ip]} share {ip}"
                addrs[ip] = path
