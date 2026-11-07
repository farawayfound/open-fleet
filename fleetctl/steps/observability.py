"""Elastic Agent: host metrics, journald and the gateway's own numbers.

Opt-in (`observability.enabled`), Linux with systemd only, and standalone:
the agent is never enrolled in Fleet, so there is no Fleet Server to run and
no enrolment credential on any box. Its policy is a file this step renders
from the plan, which is what makes check, drift and --dry-run honest -- the
box either holds exactly these bytes or it does not.

What the operator carries by hand is one file per box (`secrets_file`) holding
the ingest key and the CA fingerprint, minted on the hub by the stack's own
script. It is read here and never copied into the plan, a log line, a
command's argv or an error message: StepFailed embeds the argv and the last
twelve lines of output, and cli.py prints it, so every run of a command that
could carry a secret goes through _sudo(), which scrubs.

Nothing in check() touches the network and nothing in it asks whether
Elasticsearch is reachable. A box that cannot reach the hub is a data problem
for the hub's own stale-host check, not a reason for this step to say `drift`
about a file that is exactly right.
"""
from __future__ import annotations

import datetime
import hashlib
import http.client
import json
import os
import posixpath
import re
import shutil
import stat
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .. import shapes
from ..runner import StepFailed
from . import BLOCKED, DRIFT, MISSING, OK, Check, Step

# Informational: it goes in meta.package.version of the system inputs. The
# Hosts view depends on dataset names and field shapes, not on this string, but
# keep it equal to the stack's SYSTEM_PKG_VERSION so a reader is not misled.
SYSTEM_PKG_VERSION = "3.5.0"
AGENT_DIR = "/opt/Elastic/Agent"
AGENT_BIN = AGENT_DIR + "/elastic-agent"
AGENT_POLICY = AGENT_DIR + "/elastic-agent.yml"
# Not /etc/llmstack: that directory is group-writable by the gateway's account,
# and the token in here is read by a root process. A directory the gateway can
# write is a directory the gateway can plant a symlink in.
SECRETS_DIR = "/etc/fleet-obs"
TOKEN_FILE = SECRETS_DIR + "/admin-token"
DROPIN = "/etc/systemd/system/elastic-agent.service.d/fleetctl.conf"
UNIT = "elastic-agent"
ALLOWED_DOWNLOAD_HOSTS = ("artifacts.elastic.co",)

# Applied wherever the plan says None (the schema leaves every leaf blank
# unless somebody stated it). 9.5.6 is the newest release on both
# artifacts.elastic.co and docker.elastic.co; the whole stack is pinned
# through this one value, and the sha512 pin is what keeps it honest.
DEFAULTS: dict = {
    "agent_version": "9.5.6",
    "agent_sha512": [],
    "allow_unpinned": False,
    "secrets_file": SECRETS_DIR + "/agent-output.env",
    # The hub's own extra units are in the hub's host.yml, not here: this file
    # is exported, and a list in it would name the fleet's machines.
    "journald_units": ["llm-gateway", "llama-swap", "ollama", "cloudflared", "tailscaled"],
    "collect_status": None,        # None: the hub collects it, peers do not
    "metrics_period_s": 30,
    "process_top_n": 5,
    "health_interval_s": 30,
    "gpu_interval_s": 60,
    "status_interval_s": 300,
    "usage_interval_s": 300,
    "fleet_interval_s": 300,
    "agent_mem_high": "768M",
    "agent_mem_max": "1G",
    "agent_cpu_quota": "50%",
}

# Filesystem, fsstat, process and the two summaries are not worth the
# configurable period the others get: they are the expensive ones to collect.
SLOW_PERIOD_S = 60
KEY_WARN_DAYS = 30
MIN_FREE_WORK = 2 * 1024 ** 3
MIN_FREE_OPT = int(1.5 * 1024 ** 3)
# How long apply watches a freshly started agent, and the most it will download
# (the archive is a few hundred MB; the cap is for a server that never stops).
SETTLE_S = 45
MAX_DOWNLOAD = 2 * 1024 ** 3

_API_KEY = re.compile(r"^[A-Za-z0-9_-]+:[A-Za-z0-9_-]+\Z")
_FINGERPRINT = re.compile(r"^[0-9a-fA-F]{64}\Z")
_ES_HOST = re.compile(r"^https://[A-Za-z0-9._\[\]:-]+\Z")
_SHA512 = re.compile(r"^[0-9a-f]{128}\Z")
_VERSION = re.compile(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?")
SECRET_KEYS = ("ES_HOSTS", "ES_API_KEY", "ES_CA_FINGERPRINT")

# elastic-agent's component states, as `status --output json` numbers them.
_STATE_NAMES = {0: "starting", 1: "configuring", 2: "healthy", 3: "degraded",
                4: "failed", 5: "stopping", 6: "stopped", 7: "upgrading",
                8: "rolling back"}


# --------------------------------------------------------------------------
# small pure helpers, each a seam the tests patch or call directly
# --------------------------------------------------------------------------
def _is_root() -> bool:
    """The real uid, not Ctx.root_ok. root_ok also accepts passwordless sudo,
    which would let a non-root run past the gate and into a PermissionError
    halfway through apply -- after the download."""
    geteuid = getattr(os, "geteuid", None)
    return bool(geteuid) and geteuid() == 0


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _merged(section: dict | None, role: str) -> dict:
    """DEFAULTS under the plan's section. A key the plan left None is a key
    nobody stated, so the default applies; False and [] are real values."""
    cfg = dict(DEFAULTS)
    cfg["journald_units"] = list(DEFAULTS["journald_units"])
    cfg["agent_sha512"] = []
    for key, value in (section or {}).items():
        if value is not None:
            cfg[key] = value
    if cfg["collect_status"] is None:
        # /admin/api/status without the routed header is a live read -- a CPU
        # sample, nvidia-smi, a process walk, systemctl -- so it is polled by
        # the hub, which has a reason to look at the fleet, and not by every
        # peer.
        cfg["collect_status"] = role == "hub"
    return cfg


def _cfg(ctx) -> dict:
    return _merged(ctx.plan.get("observability"), ctx.plan["host"]["role"])


def _pins(cfg: dict) -> dict[str, str]:
    """{"<version>/<asset>": "<sha512>"} out of the plan's `a/b=hex` items."""
    out: dict[str, str] = {}
    for item in cfg.get("agent_sha512") or []:
        key, _, digest = str(item).partition("=")
        out[key.strip()] = digest.strip().lower()
    return out


def _read_secrets_env(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(?:export )?([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if not m:
            continue
        value = m.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[m.group(1)] = value
    return out


def _gateway_token(ctx) -> tuple[str | None, str]:
    """(token, "") or (None, why). A read-only copy of what EnvFile._token
    does, on purpose: that one MINTS a token when there is none, and a step
    that is only supposed to read the credential must never be able to
    replace the one the hub holds for this peer."""
    text, why = ctx.read_state(ctx.plan["paths"]["envfile"])
    if text is None:
        return None, why
    m = re.search(r"^(?:set |export )?LLMSTACK_ADMIN_TOKEN=(.*)$", text, re.M)
    if m and m.group(1).strip():
        return m.group(1).strip(), ""
    return None, "no LLMSTACK_ADMIN_TOKEN in it"


def _scrub(text: str, secrets) -> str:
    """Take every secret value out of a message. Longest first, so a secret
    that contains a shorter one is removed whole rather than in pieces."""
    for value in sorted({s for s in secrets if s and len(s) >= 4}, key=len, reverse=True):
        text = text.replace(value, "<redacted>")
    return text


def _hidden(secrets: dict[str, str], token: str) -> list[str]:
    """Every value that must never reach a log: the token, the whole key and
    each half of it (the id alone is enough to name the credential)."""
    key = secrets.get("ES_API_KEY", "")
    return [token, key, *key.split(":")]


def _expiry(secrets: dict[str, str]) -> tuple[str, int | None]:
    """("none"|"malformed"|"expired"|"ok", whole days left)."""
    raw = secrets.get("ES_KEY_EXPIRES", "").strip()
    if not raw:
        return "none", None
    try:
        when = datetime.datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return "malformed", None
    left = when.replace(tzinfo=datetime.timezone.utc) - _now()
    if left.total_seconds() <= 0:
        return "expired", 0
    return "ok", left.days


def _load_secrets(ctx, cfg: dict) -> tuple[dict | None, str]:
    """(values, "") or (None, why). `why` names the file and the problem and
    never a value from it."""
    path = cfg["secrets_file"]
    text, why = ctx.read_state(path)
    if text is None:
        if why.startswith("unreadable"):
            return None, f"{path} is {why}; {ctx.elevation_hint()}"
        return None, (f"{path} is not there; carry the ingest-key file to it "
                      f"(docs/fleetctl.md says how)")
    vals = _read_secrets_env(text)
    lacking = [k for k in SECRET_KEYS if not vals.get(k)]
    if lacking:
        return None, f"{path} lacks {', '.join(lacking)}"
    # id:secret and nothing else. This is also what guarantees no `$` in the
    # policy -- the agent substitutes variables in its policy, and a key that
    # contained one would be silently rewritten into a key that never works.
    if not _API_KEY.match(vals["ES_API_KEY"]):
        return None, (f"ES_API_KEY in {path} is not <id>:<secret> (the id:secret "
                      f"form, not the base64 'encoded' one)")
    if not _FINGERPRINT.match(vals["ES_CA_FINGERPRINT"]):
        return None, f"ES_CA_FINGERPRINT in {path} is not 64 hex digits"
    hosts = _es_hosts(vals)
    if not hosts or not all(_ES_HOST.match(h) for h in hosts):
        return None, f"ES_HOSTS in {path} must be one or more https://host:port URLs"
    kind, _ = _expiry(vals)
    if kind == "malformed":
        return None, f"ES_KEY_EXPIRES in {path} is not a UTC timestamp (YYYY-MM-DDTHH:MM:SSZ)"
    if kind == "expired":
        return None, f"ingest key expired (ES_KEY_EXPIRES in {path}); mint a new one on the hub"
    return vals, ""


def _es_hosts(secrets: dict[str, str]) -> list[str]:
    return [h for h in re.split(r"[,\s]+", secrets.get("ES_HOSTS", "")) if h]


def _units(cfg: dict) -> list[str]:
    """`.service` appended to bare names, duplicates dropped, order kept."""
    out: list[str] = []
    for name in cfg.get("journald_units") or []:
        full = name if name.endswith(".service") else name + ".service"
        if full not in out:
            out.append(full)
    return out


def _dropin(cfg: dict) -> str:
    # Guesses until measured: the 9.x agent runs its inputs as OTel-based
    # components and its footprint is probably well above the 220-280 MB the
    # docs quote. MemoryHigh throttles, MemoryMax kills; tune on the first node.
    return ("# managed by fleetctl observability-agent. Edits are overwritten on "
            "the next apply.\n"
            "[Service]\n"
            f"MemoryHigh={cfg['agent_mem_high']}\n"
            f"MemoryMax={cfg['agent_mem_max']}\n"
            f"CPUQuota={cfg['agent_cpu_quota']}\n"
            "Nice=10\n")


# --------------------------------------------------------------------------
# the policy, as a dict and then as YAML
# --------------------------------------------------------------------------
# No PyYAML: fleetctl is stdlib-only, because it runs on boxes that have no
# virtualenv. Every scalar goes through json.dumps because a JSON string IS a
# YAML double-quoted scalar, which removes the whole class of quoting bug (a
# regex with a backslash, a unit name that parses as a boolean, a host that
# looks like a number). Keys are bare when they are plain identifiers and
# quoted otherwise, for the same reason.
_PLAIN_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\Z")
_YAML_WORDS = frozenset({"y", "n", "yes", "no", "on", "off", "true", "false",
                         "null"})


def _scalar(value) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float, str)):
        return json.dumps(value)
    raise TypeError(f"cannot emit {type(value).__name__} as a YAML scalar")


def _key(name) -> str:
    text = str(name)
    if _PLAIN_KEY.match(text) and text.lower() not in _YAML_WORDS:
        return text
    return json.dumps(text)


def _emit(obj, indent: int = 0) -> list[str]:
    """Block-style YAML lines for a dict / list / scalar tree."""
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(obj, dict):
        for name, value in obj.items():
            if isinstance(value, (dict, list)) and value:
                lines.append(f"{pad}{_key(name)}:")
                lines += _emit(value, indent + 1)
            elif isinstance(value, dict):
                lines.append(f"{pad}{_key(name)}: {{}}")
            elif isinstance(value, list):
                lines.append(f"{pad}{_key(name)}: []")
            else:
                lines.append(f"{pad}{_key(name)}: {_scalar(value)}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict) and item:
                sub = _emit(item, indent + 1)
                lines.append(f"{pad}- {sub[0].lstrip()}")
                lines += sub[1:]
            elif isinstance(item, list) and item:
                raise TypeError("a list directly inside a list is not emitted")
            elif isinstance(item, dict):
                lines.append(f"{pad}- {{}}")
            elif isinstance(item, list):
                lines.append(f"{pad}- []")
            else:
                lines.append(f"{pad}- {_scalar(item)}")
    else:
        lines.append(f"{pad}{_scalar(obj)}")
    return lines


def _ds(dataset: str, kind: str = "metrics") -> dict:
    return {"dataset": dataset, "type": kind}


def _metric(sid: str, dataset: str, metricset: str, period_s: int, **extra) -> dict:
    stream = {"id": f"llmstack-system-{sid}", "data_stream": _ds(dataset),
              "metricsets": [metricset], "period": f"{period_s}s"}
    stream.update(extra)
    return stream


def _drop_mounts(field: str) -> dict:
    # Pseudo-filesystems and the container/host plumbing mounts: they are noise
    # in the Hosts view and, on a box with a lot of snaps, hundreds of series.
    # \z, not $: the agent substitutes variables in its policy and a bare `$`
    # is one character away from `${`.
    rx = r"^/(sys|cgroup|proc|dev|etc|host|lib|snap)(\z|/)"
    return {"drop_event": {"when": {"regexp": {field: rx}}}}


def _system_input(cfg: dict) -> dict:
    p = cfg["metrics_period_s"]
    n = cfg["process_top_n"]
    streams = [
        _metric("cpu", "system.cpu", "cpu", p,
                cpu={"metrics": ["percentages", "normalized_percentages"]}),
        _metric("memory", "system.memory", "memory", p),
        _metric("load", "system.load", "load", p),
        _metric("network", "system.network", "network", p),
        _metric("diskio", "system.diskio", "diskio", p),
        _metric("filesystem", "system.filesystem", "filesystem", SLOW_PERIOD_S,
                processors=[_drop_mounts("system.filesystem.mount_point")]),
        _metric("fsstat", "system.fsstat", "fsstat", SLOW_PERIOD_S,
                processors=[_drop_mounts("system.fsstat.mount_point")]),
        _metric("uptime", "system.uptime", "uptime", p),
        _metric("process", "system.process", "process", SLOW_PERIOD_S,
                process={"include_top_n": {"by_cpu": n, "by_memory": n},
                         "cmdline": {"cache": {"enabled": False}},
                         "cgroups": {"enabled": False},
                         "include_cpu_ticks": False},
                processes=[".*"],
                # Engines are commonly started with a key in argv, and a
                # command line is the one process field that can hold one.
                # Names and numbers ship; arguments never do.
                processors=[{"drop_fields": {
                    "fields": ["process.args", "process.command_line",
                               "process.args_count", "system.process.cmdline"],
                    "ignore_missing": True}}]),
        # The dataset keeps its dot: it is system.process.summary, not
        # system.process_summary, and a stream named the second way would be
        # written to a data stream the package's templates do not cover.
        _metric("process-summary", "system.process.summary", "process_summary",
                SLOW_PERIOD_S),
        _metric("socket-summary", "system.socket_summary", "socket_summary",
                SLOW_PERIOD_S),
    ]
    return {"id": "llmstack-system", "type": "system/metrics",
            "name": "llmstack-system", "revision": 1,
            "data_stream": {"namespace": "default"}, "use_output": "default",
            "meta": {"package": {"name": "system", "version": SYSTEM_PKG_VERSION}},
            "streams": streams}


def _journald_input(units: list[str]) -> dict:
    drop_poll = {"drop_event": {"when": {"and": [
        # The agent's own 127.0.0.1 polls show up in the journal as uvicorn
        # access lines for the unit it is watching. Shipping them would make
        # the gateway's log mostly a log of being read.
        {"equals": {"systemd.unit": "llm-gateway.service"}},
        {"regexp": {"message": r'127\.0\.0\.1:\d+ - "GET /(health|admin/api/[^ ]*) HTTP'}},
    ]}}}
    drop_secret = {"drop_event": {"when": {"regexp": {"message": (
        # Fail closed: a whole line that looks like it carries a credential is
        # dropped, not redacted. Redaction in a processor chain is fragile; a
        # dropped line costs one log entry.
        r'(?i)(authorization:|bearer\s+[a-z0-9._-]{16,}|api[_-]?key["=: ]+[a-z0-9._-]{16,})')}}}}
    return {"id": "llmstack-journald", "type": "journald",
            "data_stream": {"namespace": "default"}, "use_output": "default",
            "streams": [{
                "id": "llmstack-journald-units",
                "data_stream": _ds("llmstack.journal", "logs"),
                # Exact match on the unit name; repeated keys are OR'd.
                "include_matches": {"match": [f"_SYSTEMD_UNIT={u}" for u in units]},
                # First start only: once a cursor exists seek is ignored.
                "seek": "since", "since": "-1h",
                "processors": [drop_poll, drop_secret]}]}


def _processors(target: str, drop: tuple[str, ...] = ()) -> list[dict]:
    # A non-empty target for every endpoint: an empty one would merge the
    # response's keys into the event root, and /health's `status` is a string
    # where the ECS field of that name is not.
    # `drop` is for response fields that must not leave the box: the gateway
    # strips models_private from its public views, and Elasticsearch is
    # readable by the viewer login.
    return [
        {"decode_json_fields": {"fields": ["message"], "target": target,
                                "max_depth": 1, "add_error_key": True}},
        {"drop_fields": {"fields": ["message", *(f"{target}.{f}" for f in drop)],
                         "ignore_missing": True}},
        {"add_fields": {"target": "", "fields": {
            "service": {"name": "llm-gateway"}, "event": {"kind": "metric"}}}},
    ]


def _poll(sid: str, dataset: str, port: int, path: str, every_s: int, *,
          auth: bool, timeout_s: int = 10, drop: tuple[str, ...] = ()) -> dict:
    stream: dict = {
        "id": sid,
        "data_stream": _ds(dataset, "logs"),
        "interval": f"{every_s}s",
        # Always loopback: it works on the hub too, whose gateway binds
        # 127.0.0.1. No x-fleet-routed header, ever -- sending it would flip
        # this box's standby state.
        "request": {"url": f"http://127.0.0.1:{port}{path}", "method": "GET",
                    "timeout": f"{timeout_s}s", "retry": {"max_attempts": 2}},
        "response": {"decode_as": "application/json"},
    }
    if auth:
        # The token is in a 0600 root file, not in the policy: the policy is
        # world-readable-by-accident territory and this keeps a rotated token
        # a one-file change.
        stream["auth"] = {"file": {"path": TOKEN_FILE, "prefix": "Bearer "}}
    stream["processors"] = _processors("llmstack." + dataset.split(".", 1)[1], drop)
    return stream


def _gateway_inputs(cfg: dict, port: int, role: str) -> list[dict]:
    health = {"id": "llmstack-gateway-health", "type": "httpjson",
              "data_stream": {"namespace": "default"}, "use_output": "default",
              "streams": [_poll("llmstack-health", "llmstack.health", port,
                                "/health", cfg["health_interval_s"], auth=False)]}
    streams = []
    if cfg["collect_status"]:
        streams.append(_poll("llmstack-status", "llmstack.status", port,
                             "/admin/api/status", cfg["status_interval_s"], auth=True,
                             drop=("models_private",)))
    streams.append(_poll("llmstack-gpu", "llmstack.gpu", port, "/admin/api/gpu",
                         cfg["gpu_interval_s"], auth=True))
    streams.append(_poll("llmstack-usage", "llmstack.usage", port,
                         "/admin/api/usage?hours=1&recent_limit=0",
                         cfg["usage_interval_s"], auth=True, timeout_s=20,
                         # recent_limit=0 is clamped to 1 by the gateway, so a
                         # request row (key name, model) comes back anyway.
                         drop=("recent",)))
    if role == "hub":
        for name, path in (("fleet-summary", "/admin/api/fleet/summary"),
                           ("fleet-overview", "/admin/api/fleet/overview"),
                           ("peers", "/admin/api/peers")):
            streams.append(_poll(f"llmstack-{name}", "llmstack." + name.replace("-", "_"),
                                 port, path, cfg["fleet_interval_s"], auth=True))
    admin = {"id": "llmstack-gateway-admin", "type": "httpjson",
             "data_stream": {"namespace": "default"}, "use_output": "default",
             "streams": streams}
    return [health, admin]


def _requests_input(path: str) -> dict:
    """The gateway's own request log, one ECS-shaped JSON object per line. It
    is written by the gateway's service account and read by this root agent,
    so the file is the whole interface between them."""
    return {"id": "llmstack-gateway-requests", "type": "filestream",
            "data_stream": {"namespace": "default"}, "use_output": "default",
            "streams": [{
                "id": "llmstack-requests",
                "data_stream": _ds("llmstack.requests", "logs"),
                # The rotated backups too: the gateway's RotatingFileHandler
                # renames access.ndjson to .1, and a renamed file that matches
                # no path reads as removed and is closed with whatever was
                # written since the last scan still unread. With the backups
                # listed it is the same file under a new name and is drained.
                "paths": [path, path + ".*"],
                # filestream identifies a file by a fingerprint of its first
                # 1024 bytes and ignores a smaller file; one access line is
                # 400-500 bytes, so on a quiet box the first requests (and the
                # first ones after each rotation) would not ship until the file
                # grew. 64 is the smallest length it accepts.
                "prospector": {"scanner": {"fingerprint": {"enabled": True, "offset": 0,
                                                           "length": 64}}},
                "parsers": [{"ndjson": {"target": "", "overwrite_keys": True,
                                        "add_error_key": True}}],
                # target "" with overwrite_keys lets whoever can write the file
                # set any field, data_stream.* included, and the file's
                # directory belongs to the gateway's account. Processors run
                # after the parsers and add_fields overwrites, so the routing
                # is put back. (host.name cannot be: the ingest key's index
                # privileges are what bound the rest.)
                "processors": [{"add_fields": {"target": "data_stream", "fields": {
                    "type": "logs", "dataset": "llmstack.requests", "namespace": "default"}}},
                               {"add_fields": {"target": "event", "fields": {
                                   "dataset": "llmstack.requests"}}}],
                "close": {"on_state_change": {"inactive": "5m"}}}]}


def render_policy(cfg: dict, secrets: dict, *, port: int, role: str = "peer",
                  family: str = "linux", service: str = "systemd",
                  access_log: str | None = None) -> str:
    """The agent's whole policy, as text. Pure: no Ctx, no filesystem.

    Never emitted, each for a reason that cost somebody an afternoon to find:
    `config_version` (not a key in 9.6's httpjson), `max_object_depth` (9.6
    only), an `x-fleet-routed` header (flips the box's standby state), and the
    admin token (it stays in TOKEN_FILE, referenced by auth.file).
    """
    inputs = [_system_input(cfg)]
    units = _units(cfg)
    if family == "linux" and service == "systemd" and units:
        inputs.append(_journald_input(units))
    inputs += _gateway_inputs(cfg, port, role)
    if access_log:
        inputs.append(_requests_input(access_log))
    policy = {
        "outputs": {"default": {
            "type": "elasticsearch",
            "hosts": _es_hosts(secrets),
            "api_key": secrets["ES_API_KEY"],
            "ssl": {"ca_trusted_fingerprint": secrets["ES_CA_FINGERPRINT"]},
            "preset": "balanced"}},
        "agent": {"monitoring": {"enabled": False},
                  "logging": {"level": "warning", "metrics": {"enabled": False}}},
        "inputs": inputs,
    }
    head = "# managed by fleetctl observability-agent. Edits are overwritten on the next apply.\n"
    return head + "\n".join(_emit(policy)) + "\n"


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------
def _write_file(target: Path, text: str, mode: int) -> None:
    """Same-directory temporary file made with O_EXCL at the final mode,
    fsynced, then renamed over the target. There is never a moment when the
    policy exists with the wrong mode or half written, and a crash leaves the
    old file whole."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        os.unlink(tmp)
    except OSError:
        pass
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write(ctx, path: str, text: str, mode: int) -> None:
    _write_file(ctx.path(path), text, mode)


def _mode(ctx, path: str) -> int | None:
    """The permission bits, or None where there are none to read (absent, or
    Windows, where chmod toggles a read-only flag and nothing else)."""
    if os.name == "nt":
        return None
    try:
        return stat.S_IMODE(os.stat(ctx.path(path)).st_mode)
    except OSError:
        return None


def _ensure_mode(ctx, path: str, mode: int) -> bool:
    """chmod when the bits differ. ctx.write() only chmods a file it wrote, so
    a token that is already right but 0644 would otherwise be reported as
    drift by check() and left alone by apply() -- the UNCHANGED loop."""
    have = _mode(ctx, path)
    if have is None or have == mode:
        return False
    os.chmod(ctx.path(path), mode)
    return True


def _trust_problem(ctx, cfg: dict) -> str | None:
    """Why the paths a root process is about to trust are not trustworthy.

    The secrets file feeds a root agent and the token file is written by one,
    so a directory the gateway's account (or anyone else) can write is a way to
    swap either for a symlink or a different file.
    """
    secrets = cfg["secrets_file"]
    for path in (secrets, TOKEN_FILE, AGENT_POLICY):
        try:
            st = os.lstat(ctx.path(path))
        except OSError:
            continue
        if stat.S_ISLNK(st.st_mode):
            return f"{path} is a symlink"
    checks = [(d, "directory") for d in sorted({posixpath.dirname(secrets), SECRETS_DIR})]
    checks.append((secrets, "file"))
    for path, what in checks:
        try:
            st = os.lstat(ctx.path(path))
        except OSError:
            continue
        if os.name == "nt":
            continue
        if st.st_mode & 0o022:
            return f"{what} {path} is writable by group or others"
        # The ingest key in it can forge metrics and logs, and the gateway's
        # account (the one that faces the internet) is the group or other that
        # a 0644 file would hand it to. The same key must be 0600 in the policy.
        if what == "file" and st.st_mode & 0o077:
            return (f"{what} {path} is readable by group or others (it holds the "
                    f"ingest key; chmod 0600 it)")
        # A sandbox's files belong to whoever runs the tests, not to root.
        if not ctx.sandboxed and st.st_uid != 0:
            return f"{what} {path} is not owned by root"
    return None


def _install_state(ctx) -> tuple[str, str]:
    """("installed" | "half" | "absent" | "unreadable", why).

    Half is the binary without a unit: an install that died after copying and
    before registering. It is its own state because the way out is different
    -- uninstall first, then install."""
    there, why = ctx.exists_state(AGENT_BIN)
    if there is None:
        return "unreadable", why
    if not there:
        return "absent", ""
    # FragmentPath, not `systemctl cat`: cat exits 0 for a unit that has only
    # a drop-in, and this step writes its own drop-in before it installs, so an
    # install that died after the copy would read as installed. show exits 0
    # for a unit it has never heard of and prints an empty FragmentPath.
    r = ctx.probe(["systemctl", "show", UNIT, "-p", "FragmentPath"])
    if r is None:
        # A timeout or no systemctl at all is not "no unit": the plan for a
        # half install includes `uninstall --force`, and it must never be
        # chosen because a probe failed on a healthy box.
        return "unreadable", "systemctl gave no answer"
    if r.returncode != 0:
        return "unreadable", f"systemctl show exited {r.returncode}"
    fragment = ""
    for line in (r.stdout or "").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "FragmentPath":
            fragment = value.strip()
    return ("installed" if fragment else "half"), ""


def _unit_state(ctx) -> dict[str, str]:
    r = ctx.probe(["systemctl", "show", UNIT, "-p", "ActiveState", "-p",
                   "SubState", "-p", "UnitFileState"])
    out: dict[str, str] = {}
    if r is None:
        return out
    for line in (r.stdout or "").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key.strip()] = value.strip()
    return out


def _agent_status(ctx) -> tuple[str, bool]:
    """(one short phrase, failed) out of `elastic-agent status --output json`.

    A degraded httpjson component is a gateway that is down, which is not the
    agent's fault, and `status` exits 1 for it -- so the exit code means
    nothing here and degraded stays detail. Failed (state 4, on the agent or on
    a component) is different: a wrong CA fingerprint or a revoked key lands
    there with the unit still `active`, nothing shipping, and nothing else to
    say so."""
    r = ctx.probe([AGENT_BIN, "status", "--output", "json"], timeout=35)
    if r is None:
        return "status unavailable", False
    try:
        data = json.loads(r.stdout or "")
    except ValueError:
        return f"status unavailable (exit {r.returncode})", False
    if not isinstance(data, dict):
        return "status unavailable", False
    state = _STATE_NAMES.get(data.get("state"), str(data.get("state")))
    comps = [c for c in (data.get("components") or []) if isinstance(c, dict)]
    bad = [str(c.get("id") or c.get("name")) for c in comps if c.get("state") != 2]
    failed = data.get("state") == 4 or any(c.get("state") == 4 for c in comps)
    if bad:
        return (f"agent {state}; {len(bad)}/{len(comps)} component(s) not healthy: "
                + ", ".join(bad[:4])), failed
    return f"agent {state}; {len(comps)} component(s) healthy", failed


# --------------------------------------------------------------------------
# the download, the only thing here that touches the network
# --------------------------------------------------------------------------
class _Refused(RuntimeError):
    pass


def _allowed_url(url: str) -> bool:
    p = urllib.parse.urlparse(url)
    return (p.scheme == "https" and (p.hostname or "").lower() in ALLOWED_DOWNLOAD_HOSTS
            and p.port in (None, 443))


class _StrictRedirects(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only to https on the allow-listed host. The archive is
    about to be run as root, and a pinned digest is the second line of defence,
    not the first -- an unpinned run (allow_unpinned) has only this."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _allowed_url(newurl):
            host = urllib.parse.urlparse(newurl).hostname or "?"
            raise _Refused(f"refusing a redirect to {host}: the agent is only "
                           f"fetched over https from {', '.join(ALLOWED_DOWNLOAD_HOSTS)}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _urlopen(req, timeout: float):
    return urllib.request.build_opener(_StrictRedirects).open(req, timeout=timeout)


def _fetch(url: str, dest: Path) -> str:
    """Stream `url` to `dest` and return its sha512, in one pass."""
    digest = hashlib.sha512()
    size = 0
    req = urllib.request.Request(url, headers={"User-Agent": "fleetctl"})
    with _urlopen(req, 120) as src, open(dest, "wb") as out:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_DOWNLOAD:
                raise _Refused(f"{url} is larger than {MAX_DOWNLOAD >> 20} MiB; not downloading more")
            digest.update(chunk)
            out.write(chunk)
    return digest.hexdigest()


def _sibling_digest(url: str) -> str:
    req = urllib.request.Request(url + ".sha512", headers={"User-Agent": "fleetctl"})
    with _urlopen(req, 60) as src:
        text = src.read(4096).decode("utf-8", "replace")
    token = (text.split() or [""])[0].lower()
    if not _SHA512.match(token):
        raise RuntimeError(f"{url}.sha512 does not start with a sha512 digest")
    return token


def _vet_member(member: tarfile.TarInfo, root: Path, symlinks: set | None = None) -> None:
    """Refuse a member that could land outside the tree. `symlinks` is the set
    of symlink paths vetted so far (a caller walking the whole archive passes
    one): a path that runs through an earlier link is refused, because each
    link target is judged against its own name and a chain of two safe-looking
    ones (a/b -> .., a/b/c -> ../..) points out of the tree."""
    name = member.name
    parts = Path(name.replace("\\", "/")).parts
    bad = (name.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", name) is not None
           or ".." in parts or member.isdev() or member.isfifo())
    if not bad and symlinks is not None:
        norm = posixpath.normpath(name.replace("\\", "/"))
        bits = norm.split("/")
        bad = any("/".join(bits[:i]) in symlinks for i in range(1, len(bits)))
        if member.issym():
            symlinks.add(norm)
    if not bad and (member.issym() or member.islnk()):
        # A symlink's target is relative to the link; a hard link's, to the
        # archive root. Either may point anywhere inside the tree and nowhere
        # outside it.
        base = posixpath.dirname(name) if member.issym() else ""
        target = posixpath.normpath(posixpath.join(base, member.linkname))
        bad = (member.linkname.startswith("/") or target == ".."
               or target.startswith("../"))
    if bad:
        raise RuntimeError(f"unsafe member {name!r} in the agent archive; not extracting")


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        symlinks: set = set()
        for member in members:
            _vet_member(member, dest, symlinks)
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, members=members, filter="data")
        else:
            # Python 3.10 before the filter backport: the vetting above is all
            # there is, which is why it is not skipped when the filter exists.
            tar.extractall(dest, members=members)


def _download_and_verify(ctx, dest_dir) -> Path:
    """Fetch the pinned agent tarball into `dest_dir`, verify it, unpack it and
    return the directory that holds `elastic-agent`.

    Compared against the PINNED digest. With allow_unpinned the sibling .sha512
    file stands in, which proves the transfer was whole and nothing about who
    made it -- it comes from the same origin as the archive. Mismatch aborts
    before anything is extracted.
    """
    cfg = _cfg(ctx)
    version = cfg["agent_version"]
    asset = shapes.elastic_agent_asset(ctx.family, ctx.plan["platform"]["arch"])
    if asset is None:
        raise RuntimeError(f"no agent build for {ctx.family}/{ctx.plan['platform']['arch']}")
    url = shapes.ELASTIC_AGENT_URL.format(version=version, asset=asset)
    if not _allowed_url(url):
        raise RuntimeError(f"{url} is not an allowed download location")
    pin = _pins(cfg).get(f"{version}/{asset}")
    if pin is None and not cfg["allow_unpinned"]:
        raise RuntimeError(f"no sha512 pin for {version}/{asset} in observability.agent_sha512")
    dest = Path(dest_dir)
    archive = dest / f"elastic-agent-{version}-{asset}"
    last: Exception | None = None
    for attempt in range(3):
        try:
            got = _fetch(url, archive)
            break
        except _Refused:
            raise
        except (OSError, http.client.HTTPException) as exc:
            last = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"could not download {url}: {last.__class__.__name__}: {last}")
    want = pin if pin is not None else _sibling_digest(url)
    if got != want:
        raise RuntimeError(f"sha512 of {archive.name} is {got[:16]}..., expected "
                           f"{want[:16]}...; nothing was extracted")
    unpacked = dest / "unpacked"
    _extract(archive, unpacked)
    found = [p for p in sorted(unpacked.iterdir())
             if p.is_dir() and (p / "elastic-agent").exists()]
    if len(found) != 1:
        raise RuntimeError(f"{archive.name} does not hold exactly one directory with elastic-agent in it")
    return found[0]


def _free(path) -> int:
    """Free bytes at `path`, or at the nearest parent that exists: disk_usage
    raises for a directory that is not there yet, and the work directory under
    /opt may not be."""
    p = Path(path)
    while True:
        try:
            return shutil.disk_usage(p).free
        except OSError:
            if p.parent == p:
                raise
            p = p.parent


def _workdir_parent() -> str:
    # /var/tmp is root-owned and survives a reboot-time tmpfs wipe mid-install;
    # a box without one (a sandbox) takes the platform's temp directory.
    return "/var/tmp" if os.path.isdir("/var/tmp") else tempfile.gettempdir()


def _sudo(ctx, argv, hidden, **kw):
    """ctx.sudo with the secrets taken out of any failure. StepFailed carries
    the argv and the command's last twelve lines of output and cli.py prints
    it, so a command that echoed a key would otherwise put it in a CI log."""
    try:
        return ctx.sudo(argv, **kw)
    except StepFailed as exc:
        raise StepFailed(_scrub(str(exc), hidden)) from None


# --------------------------------------------------------------------------
# the step
# --------------------------------------------------------------------------
class ObservabilityAgent(Step):
    id = "observability-agent"
    title = "Elastic Agent: host metrics, journald and gateway status"
    system = True

    def wanted(self, ctx) -> bool:
        # No config, no step. An explicit `enabled: false` removes nothing
        # that is already installed: teardown is manual (docs/fleetctl.md),
        # because a silent uninstall on a config edit is the wrong default.
        return bool((ctx.plan.get("observability") or {}).get("enabled"))

    # ------------------------------------------------------------------ check
    def check(self, ctx) -> Check:
        try:
            return self._check(ctx)
        except Exception as exc:  # noqa: BLE001
            # check() must never raise: cli reports an exception from it as an
            # ERROR row, which loses the state the rest of the run is built on.
            return Check(BLOCKED, f"could not judge the agent ({exc.__class__.__name__}: {exc})")

    def _check(self, ctx) -> Check:
        cfg = _cfg(ctx)
        version = cfg["agent_version"]
        arch = ctx.plan["platform"]["arch"]

        # 1. platform
        if ctx.family != "linux":
            return Check(BLOCKED, "v1 installs on linux-systemd only; set "
                                  "observability.enabled: false for this host")
        if ctx.plan["platform"]["service"] != "systemd":
            return Check(BLOCKED, f"no systemd here (service manager: "
                                  f"{ctx.plan['platform']['service']!r}), so no journal "
                                  f"and no unit; set observability.enabled: false for "
                                  f"this host")
        asset = shapes.elastic_agent_asset(ctx.family, arch)
        if asset is None:
            return Check(BLOCKED, f"unsupported arch {arch or 'unknown'}")

        # 2. pin
        pin = _pins(cfg).get(f"{version}/{asset}")
        if pin is None and not cfg["allow_unpinned"]:
            return Check(BLOCKED, f"no sha512 pin for {version}/{asset} in "
                                  f"observability.agent_sha512")

        # 3. privilege. A dry run is allowed to look without it.
        if not ctx.dry_run and not _is_root():
            return Check(BLOCKED, f"installing the agent needs root; {ctx.elevation_hint()}")

        # 4. the gateway token
        token, why = _gateway_token(ctx)
        envfile = ctx.plan["paths"]["envfile"]
        if token is None:
            if why.startswith("unreadable"):
                return Check(BLOCKED, f"{envfile} is {why}; {ctx.elevation_hint()}")
            if ctx.dry_run:
                # Not BLOCKED: on a fresh box a real apply writes the env file
                # in an earlier step, so this is the plan, not a refusal.
                return Check(MISSING, f"{envfile} is not written yet ({why})",
                             self._fresh_plan(ctx, cfg, version, envfile))
            return Check(BLOCKED, f"run fleetctl apply first: gateway env file "
                                  f"missing or without a token ({envfile}: {why})")

        # 5. the carried secrets file. Trust before content: nothing from a
        # path that could have been swapped is read, let alone believed.
        trust = _trust_problem(ctx, cfg)
        if trust:
            return Check(BLOCKED, f"untrusted path: {trust}")
        secrets, problem = _load_secrets(ctx, cfg)
        if secrets is None:
            return Check(BLOCKED, problem)
        kind, days = _expiry(secrets)
        warning = ""
        if kind == "ok" and days is not None and days < KEY_WARN_DAYS:
            warning = f"ingest key expires in {days} day(s); mint a new one on the hub"

        # 6. installed?
        state, why = _install_state(ctx)
        if state == "unreadable":
            return Check(BLOCKED, f"{AGENT_BIN} cannot be examined ({why}); "
                                  f"{ctx.elevation_hint()}")

        # 7. convergence
        policy = render_policy(
            cfg, secrets, port=ctx.plan["network"]["port"], role=ctx.plan["host"]["role"],
            family=ctx.family, service=ctx.plan["platform"]["service"],
            access_log=shapes.access_log_path(ctx.family, ctx.plan["paths"]["state"]))
        lines: list[str] = []
        reasons: list[str] = []
        if state == "half":
            lines.append(f"{AGENT_BIN} uninstall --force (half-installed)")
        if state != "installed":
            lines.append(f"install elastic-agent {version}")

        have, why = ctx.read_state(AGENT_POLICY)
        if have is None and why.startswith("unreadable"):
            return Check(BLOCKED, f"{AGENT_POLICY} is {why}; {ctx.elevation_hint()}")
        if have != policy:
            lines.append(f"write {AGENT_POLICY} "
                         f"({'policy differs' if have is not None else 'not written yet'})")
            reasons.append("policy differs")
        elif (_mode(ctx, AGENT_POLICY) or 0o600) & 0o077:
            lines.append(f"chmod 0600 {AGENT_POLICY} (it holds the ingest key)")
            reasons.append("policy mode")

        have, why = ctx.read_state(TOKEN_FILE)
        if have is None and why.startswith("unreadable"):
            return Check(BLOCKED, f"{TOKEN_FILE} is {why}; {ctx.elevation_hint()}")
        if have != token or (_mode(ctx, TOKEN_FILE) not in (None, 0o600)):
            lines.append(f"write {TOKEN_FILE}")
            reasons.append("token file")

        have, _ = ctx.read_state(DROPIN)
        if have != _dropin(cfg):
            lines.append(f"write {DROPIN}")
            reasons.append("resource drop-in")

        if state == "installed":
            unit = _unit_state(ctx)
            active, sub, enabled = (unit.get("ActiveState", ""), unit.get("SubState", ""),
                                    unit.get("UnitFileState", ""))
            if active == "failed" or sub == "auto-restart":
                # BLOCKED, not DRIFT: apply cannot fix a crash-loop, and a
                # DRIFT here would end the run as UNCHANGED after an apply
                # that "worked".
                return Check(BLOCKED, "agent unhealthy: see journalctl -u elastic-agent")
            if enabled != "enabled" or active == "inactive":
                lines.append(f"systemctl enable --now {UNIT}")
                reasons.append("unit not enabled and running")

        # 8. result
        if state != "installed":
            detail = ("elastic-agent is half-installed (binary, no unit)" if state == "half"
                      else "elastic-agent is not installed")
        elif lines:
            detail = "; ".join(reasons)
        else:
            detail = ""
        failed, summary = False, ""
        if state == "installed":
            detail = self._with_version(ctx, detail, version)
            if _is_root():
                summary, failed = _agent_status(ctx)
                detail = f"{detail}; {summary}" if detail else summary
        if warning:
            detail = f"{detail}; {warning}" if detail else warning
        if state != "installed":
            return Check(MISSING, detail, lines)
        if lines:
            return Check(DRIFT, detail, lines)
        if failed:
            # BLOCKED, not DRIFT, for the reason the crash-loop above is: the
            # files are right, apply has nothing to change, and a green row
            # here would be a lie.
            return Check(BLOCKED, f"{summary}; see journalctl -u {UNIT}")
        return Check(OK, detail)

    @staticmethod
    def _with_version(ctx, detail: str, pinned: str) -> str:
        """Prefix the version the binary reports. Never a version it did not
        read; and a version that differs from the pin is said, not acted on --
        `install --non-interactive` cannot upgrade, so calling it drift would
        send apply round a loop that ends UNCHANGED."""
        r = ctx.probe([AGENT_BIN, "version", "--binary-only"], timeout=15)
        m = _VERSION.search(((r.stdout or "") + (r.stderr or "")) if r else "")
        if not m:
            head = "version unknown"
        elif m.group(0) != pinned:
            head = f"elastic-agent {m.group(0)} (pinned {pinned}; reinstall to change)"
        else:
            head = f"elastic-agent {m.group(0)}"
        return f"{head}; {detail}" if detail else head

    def _fresh_plan(self, ctx, cfg: dict, version: str, envfile: str) -> list[str]:
        """What a real apply would do on a box whose env file does not exist
        yet. The same lines check() would give, minus the ones that need the
        token to compute."""
        state, _ = _install_state(ctx)
        lines = [f"{envfile} is written by the envfile step first"]
        if state == "half":
            lines.append(f"{AGENT_BIN} uninstall --force (half-installed)")
        if state != "installed":
            lines.append(f"install elastic-agent {version}")
        lines += [f"write {TOKEN_FILE}", f"write {DROPIN}",
                  f"write {AGENT_POLICY} (not written yet)",
                  f"systemctl enable --now {UNIT}"]
        there, _ = ctx.exists_state(cfg["secrets_file"])
        if there is False:
            lines.append(f"carry the ingest-key file to {cfg['secrets_file']} "
                         f"(not there yet)")
        return lines

    # ------------------------------------------------------------------ apply
    def apply(self, ctx) -> None:
        cfg = _cfg(ctx)
        secrets, problem = _load_secrets(ctx, cfg)
        if secrets is None:
            raise RuntimeError(problem)
        token, why = _gateway_token(ctx)
        if token is None:
            raise RuntimeError(f"no gateway admin token to hand the agent "
                               f"({ctx.plan['paths']['envfile']}: {why})")
        hidden = _hidden(secrets, token)
        try:
            self._apply(ctx, cfg, secrets, token, hidden)
        except StepFailed as exc:
            raise StepFailed(_scrub(str(exc), hidden)) from None
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(_scrub(f"{exc.__class__.__name__}: {exc}", hidden)) from None

    def _apply(self, ctx, cfg: dict, secrets: dict, token: str, hidden: list[str]) -> None:
        version = cfg["agent_version"]
        policy = render_policy(
            cfg, secrets, port=ctx.plan["network"]["port"], role=ctx.plan["host"]["role"],
            family=ctx.family, service=ctx.plan["platform"]["service"],
            access_log=shapes.access_log_path(ctx.family, ctx.plan["paths"]["state"]))
        state, why = _install_state(ctx)
        if state == "unreadable":
            raise RuntimeError(f"cannot tell whether elastic-agent is installed ({why})")

        # 1. space, only where something is about to be downloaded: a policy
        # change on a full disk must still go through.
        if state != "installed":
            parent = _workdir_parent()
            for where, need in ((parent, MIN_FREE_WORK), (str(ctx.path("/opt")), MIN_FREE_OPT)):
                free = _free(where)
                if free < need:
                    raise RuntimeError(f"{where} has {free / 1024 ** 3:.1f} GB free; the "
                                       f"agent install needs {need / 1024 ** 3:.1f}")

        # 2. the secrets directory. ensure_dir() does not repair one that is
        # already there, so the mode and the owner are set explicitly.
        ctx.ensure_dir(SECRETS_DIR, 0o700)
        _sudo(ctx, ["chmod", "0700", SECRETS_DIR], hidden)
        _sudo(ctx, ["chown", "root:root", SECRETS_DIR], hidden)
        # The carried file: check() refuses one a group or others can read, but
        # it is cheap to put right a file this step is about to read anyway.
        if _ensure_mode(ctx, cfg["secrets_file"], 0o600):
            ctx.did(f"set {cfg['secrets_file']} to 0600")

        # 3. the token. ctx.write() chmods after it writes; the umask closes
        # the gap before that.
        old = os.umask(0o077)
        try:
            changed = ctx.write(TOKEN_FILE, token, mode=0o600)
        finally:
            os.umask(old)
        if _ensure_mode(ctx, TOKEN_FILE, 0o600):
            changed = True
        if changed:
            ctx.did(f"wrote {TOKEN_FILE}")

        # 4. the resource drop-in
        if ctx.write(DROPIN, _dropin(cfg), mode=0o644):
            ctx.did(f"wrote {DROPIN}")
            _sudo(ctx, ["systemctl", "daemon-reload"], hidden)

        # 5. a half install is removed first. From outside the install
        # directory: the uninstaller deletes the directory it is run from.
        if state == "half":
            _sudo(ctx, [AGENT_BIN, "uninstall", "--force"], hidden, cwd="/")
            ctx.did("removed a half-installed elastic-agent")

        # 6. install
        if state != "installed":
            self._install(ctx, version, policy, hidden)

        # 7. the policy. After a fresh install this normally finds the file as
        # it was written; if the installer ever rewrites it, this puts ours back.
        restart = False
        if ctx.read(AGENT_POLICY) != policy:
            _atomic_write(ctx, AGENT_POLICY, policy, 0o600)
            ctx.did(f"wrote {AGENT_POLICY}")
            restart = True
        elif _ensure_mode(ctx, AGENT_POLICY, 0o600):
            ctx.did(f"set {AGENT_POLICY} to 0600")
        if restart:
            # The agent reloads input changes by itself in about ten seconds,
            # but a restart is simple and costs only a small in-memory queue.
            _sudo(ctx, ["systemctl", "restart", UNIT], hidden)
            ctx.did(f"restarted {UNIT}")

        # 8. the service
        _sudo(ctx, ["systemctl", "reset-failed", UNIT], hidden, check=False)
        _sudo(ctx, ["systemctl", "enable", "--now", UNIT], hidden)
        ctx.did(f"enabled {UNIT}")
        self._stay_up(ctx)

    def _stay_up(self, ctx) -> None:
        """Watch the agent for SETTLE_S seconds and fail the apply if it dies.
        `enable --now` returns as soon as the process is forked, and cli.py only
        counts a MISSING or DRIFT re-check as a failure, so a crash-loop or a
        `failed` agent a few seconds later would otherwise exit 0."""
        deadline = time.monotonic() + SETTLE_S
        while True:
            unit = _unit_state(ctx)
            if unit.get("ActiveState") == "failed" or unit.get("SubState") == "auto-restart":
                raise RuntimeError(f"{UNIT} stopped after it was started; see journalctl -u {UNIT}")
            summary, failed = _agent_status(ctx)
            if failed:
                raise RuntimeError(f"{UNIT} is running but reports {summary}; the ingest key or "
                                   f"the CA fingerprint is probably wrong (journalctl -u {UNIT})")
            if time.monotonic() >= deadline or summary.startswith("agent healthy"):
                return
            time.sleep(3)

    def _install(self, ctx, version: str, policy: str, hidden: list[str]) -> None:
        # A private directory that is thrown away: no cache, so there is never
        # an archive on disk that was verified once and trusted since.
        workdir = Path(tempfile.mkdtemp(prefix="fleet-obs-", dir=_workdir_parent()))
        try:
            src = Path(_download_and_verify(ctx, workdir))
            # Into the extracted tree, where the installer picks it up. The
            # directory is root-only, so nothing is exposed before install.
            _write_file(src / "elastic-agent.yml", policy, 0o600)
            # 027, not 077: the installer makes directories the service has to
            # traverse as the agent's own user.
            old = os.umask(0o027)
            try:
                # No --url and no --enrollment-token: that is what makes this a
                # standalone install that never enrols.
                _sudo(ctx, [str(src / "elastic-agent"), "install", "--non-interactive"],
                      hidden, cwd=str(src), timeout=900)
            finally:
                os.umask(old)
            _sudo(ctx, ["chmod", "0600", AGENT_POLICY], hidden)
            have = _mode(ctx, AGENT_POLICY)
            if not ctx.sandboxed and have is not None and have & 0o077:
                raise RuntimeError(f"{AGENT_POLICY} is mode {have:04o} after install; it "
                                   f"holds the ingest key and must be 0600")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        ctx.did(f"installed elastic-agent {version}")
