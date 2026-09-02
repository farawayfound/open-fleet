#!/usr/bin/env python3
"""
open-fleet gateway
========================
Single front door for one inference box, and optionally the control plane for
a small fleet of them.

  Cline / any OpenAI client ->  https://<api-host>/v1/*        (Bearer sk-ffa-...)
  Admin dashboard           ->  https://llm.example.com/  (Cloudflare Access SSO)

Both hostnames arrive here over a Cloudflare Tunnel and hit 127.0.0.1:8080.
Upstream is llama-swap on 127.0.0.1:8081, which starts/stops llama-server
processes on demand.

Responsibilities that llama-swap does not cover and this process does:
  * named API keys, hashed at rest, revocable, with optional expiry and
    request/token budgets
  * per-key agent profiles: system prompt + rules injection, model allow-lists,
    parameter defaults/caps, applied at the proxy
  * per-key / per-model usage metering (tokens, latency, TTFT)
  * Cloudflare Access JWT verification for every admin surface
  * model catalogue: browse + download GGUFs from HuggingFace with live progress
  * generating llama-swap.yaml from a friendly model/param record
  * llama-bench runs with parameter overrides, recorded so tunes are comparable
  * amdgpu GTT boot-parameter staging (the VRAM-allocation lever on APUs)
  * host telemetry (CPU / RAM / disk / amdgpu sysfs)
  * fleet federation: peer gateways (over LAN or tailnet) are reachable from
    this dashboard through /admin/api/fleet/<peer>/..., authenticated with the
    peer's own admin token
"""

from __future__ import annotations

import asyncio
import calendar
import hashlib
import json
import logging
import math
import os
import platform
import re
import secrets
import shutil
import signal
import smtplib
import sqlite3
import socket
import struct
import subprocess
import tempfile
import threading
import time
from contextlib import asynccontextmanager, closing
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, AsyncIterator, Iterable
from urllib.parse import quote, urlparse

import httpx
import jwt
import psutil
import yaml
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)

# Hardware and OS facts, shared verbatim with the installer -- see hw.py. It
# is stdlib-only and imports nothing from here, so this direction is the only
# one; `fleetctl` loads the same file off a box that has no venv yet.
#
# NB for deploys: hw.py travels WITH app.py. deploy-gateway.sh lists it in
# FILES for that reason -- an app.py that lands beside a stale (or absent)
# hw.py does not start.
import hw
from hw import (  # noqa: F401 -- re-exported: callers and tests use gw.<name>
    MAC_GPU_TTL,
    WIN_GPU_TTL,
    _macgpu_cache,
    _nvsmi,
    _os_cache,
    _read_int,
    _WIN_GPU_PS,
    _win_build,
    _wingpu_cache,
    amdgpu_stats,
    os_info,
    windows_gpu_stats,
)

# uvicorn's own logger, so anything written here lands in the same journal
# stream as the access log instead of needing a logging config of its own.
log = logging.getLogger("uvicorn.error")

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

HOME = Path(os.environ.get("LLMSTACK_HOME", "/opt/llmstack"))
STATE = Path(os.environ.get("LLMSTACK_STATE", "/var/lib/llmstack"))
ETC = Path(os.environ.get("LLMSTACK_ETC", "/etc/llmstack"))

UPSTREAM = os.environ.get("LLMSTACK_UPSTREAM", "http://127.0.0.1:8081").rstrip("/")
MODELS_DIR = Path(os.environ.get("LLMSTACK_MODELS_DIR", str(STATE / "models")))
DB_PATH = Path(os.environ.get("LLMSTACK_DB", str(STATE / "gateway.db")))
MODELS_JSON = Path(os.environ.get("LLMSTACK_MODELS_JSON", str(STATE / "models.json")))
SWAP_CONFIG = Path(os.environ.get("LLMSTACK_SWAP_CONFIG", str(ETC / "llama-swap.yaml")))
LLAMA_SERVER = os.environ.get("LLMSTACK_LLAMA_SERVER", str(HOME / "bin" / "llama-server"))
LLAMA_BENCH = os.environ.get("LLMSTACK_LLAMA_BENCH", str(HOME / "bin" / "llama-bench"))
STATIC_DIR = Path(os.environ.get("LLMSTACK_STATIC", str(HOME / "gateway" / "static")))

# Fleet identity. HOST_NAME is how this gateway introduces itself on the
# dashboard; PUBLIC_API_URL is the base URL clients of THIS host should use.
HOST_NAME = os.environ.get("LLMSTACK_HOST_NAME", platform.node() or "local")
PUBLIC_API_URL = os.environ.get("LLMSTACK_PUBLIC_API_URL", "")
PEERS_PATH = Path(os.environ.get("LLMSTACK_PEERS", str(STATE / "peers.json")))
GPUCONF_HELPER = os.environ.get(
    "LLMSTACK_GPUCONF", str(HOME / "bin" / "llmstack-gpuconf")
)

# Boxes that are somebody's daily driver before they are a fleet peer. When
# this points at a file, an outside watchdog owns the answer to "may the fleet
# use this machine right now", and the gateway stops ADVERTISING its models to
# the hub whenever the answer is no. See availability().
AVAILABILITY_FILE = Path(
    os.environ.get("LLMSTACK_AVAILABILITY_FILE", "").strip() or os.devnull
)
AVAILABILITY_GATED = bool(os.environ.get("LLMSTACK_AVAILABILITY_FILE", "").strip())
AVAILABILITY_MAX_AGE = float(os.environ.get("LLMSTACK_AVAILABILITY_MAX_AGE", "120"))

CF_TEAM_DOMAIN = os.environ.get("CF_ACCESS_TEAM_DOMAIN", "").strip()
CF_AUD = os.environ.get("CF_ACCESS_AUD", "").strip()
ADMIN_TOKEN = os.environ.get("LLMSTACK_ADMIN_TOKEN", "").strip()
ADMIN_EMAILS = {
    e.strip().lower()
    for e in os.environ.get("LLMSTACK_ADMIN_EMAILS", "").split(",")
    if e.strip()
}
INJECT_USAGE = os.environ.get("LLMSTACK_INJECT_USAGE", "1") != "0"

KEY_PREFIX = "sk-ffa-"

# STATE and ETC are this box's own disk. MODELS_DIR frequently is not: the
# Windows boxes keep their weights on an external SSD, and an unclean
# disconnect both re-letters the volume and leaves this path pointing at
# nothing. Because this runs at IMPORT, a missing drive did not degrade the
# box -- it killed it. apu-tablet-2 lost D: on 2026-08-27 and crash-looped nine
# times behind its SYSTEM task, so nothing bound :8080 and the hub, correctly,
# called the machine offline; the one process that could have said "my model
# volume is gone" was the process that could not start.
#
# A box whose weights are unreachable is still worth having up. It still
# routes for the fleet, still serves the dashboard, and is still the only
# thing that can report what happened. So the model directory is created
# where it can be, and reported where it cannot.
for _d in (STATE, ETC):
    _d.mkdir(parents=True, exist_ok=True)
try:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# database
# --------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    key_hash    TEXT NOT NULL UNIQUE,
    prefix      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    last_used_at TEXT,
    disabled    INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS usage (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT NOT NULL,
    key_id            INTEGER,
    key_name          TEXT,
    model             TEXT,
    endpoint          TEXT,
    stream            INTEGER NOT NULL DEFAULT 0,
    status            INTEGER,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens      INTEGER NOT NULL DEFAULT 0,
    ttft_ms           INTEGER,
    latency_ms        INTEGER
);
CREATE INDEX IF NOT EXISTS usage_ts ON usage(ts);
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    repo        TEXT NOT NULL,
    filename    TEXT NOT NULL,
    status      TEXT NOT NULL,
    bytes_done  INTEGER NOT NULL DEFAULT 0,
    bytes_total INTEGER NOT NULL DEFAULT 0,
    message     TEXT,
    dest        TEXT
);
CREATE TABLE IF NOT EXISTS agents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    key_id          INTEGER NOT NULL UNIQUE,
    enabled         INTEGER NOT NULL DEFAULT 1,
    name            TEXT NOT NULL DEFAULT '',
    system_prompt   TEXT NOT NULL DEFAULT '',
    rules           TEXT NOT NULL DEFAULT '',
    allowed_models  TEXT NOT NULL DEFAULT '[]',
    force_model     TEXT NOT NULL DEFAULT '',
    param_overrides TEXT NOT NULL DEFAULT '{}',
    updated_at      TEXT
);
CREATE TABLE IF NOT EXISTS bench (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    model_id   TEXT NOT NULL,
    gguf       TEXT NOT NULL,
    params     TEXT NOT NULL DEFAULT '{}',
    status     TEXT NOT NULL,
    pp_tps     REAL,
    tg_tps     REAL,
    message    TEXT,
    raw        TEXT
);
CREATE TABLE IF NOT EXISTS snapshots (
    host TEXT PRIMARY KEY,
    ts   TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- Batch jobs: one row per fan-out. The requests and results live in NDJSON
-- files under STATE/batches -- a hundred thousand chat bodies do not belong
-- in sqlite rows, and NDJSON appends survive a crash mid-batch, which is what
-- makes resume-at-boot possible.
CREATE TABLE IF NOT EXISTS batches (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    key_id      INTEGER,
    key_name    TEXT,
    label       TEXT NOT NULL DEFAULT '',
    models      TEXT NOT NULL DEFAULT '[]',
    status      TEXT NOT NULL,
    total       INTEGER NOT NULL DEFAULT 0,
    done        INTEGER NOT NULL DEFAULT 0,
    failed      INTEGER NOT NULL DEFAULT 0,
    message     TEXT,
    archived_at TEXT
);
-- Teams: a key that fronts a crew rather than a single model. The primary
-- model leads the conversation; the gateway hands it a spawn_subagents tool
-- and executes the spawned tasks across the fleet in parallel.
CREATE TABLE IF NOT EXISTS teams (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    key_id         INTEGER NOT NULL UNIQUE,
    enabled        INTEGER NOT NULL DEFAULT 1,
    name           TEXT NOT NULL DEFAULT '',
    primary_model  TEXT NOT NULL DEFAULT '',
    worker_models  TEXT NOT NULL DEFAULT '[]',
    max_workers    INTEGER NOT NULL DEFAULT 4,
    max_rounds     INTEGER NOT NULL DEFAULT 6,
    system_prompt  TEXT NOT NULL DEFAULT '',
    worker_prompt  TEXT NOT NULL DEFAULT '',
    updated_at     TEXT,
    archived_at    TEXT
);
-- What the purge took, per key, so deleting old request rows cannot hand
-- somebody a fresh lifetime token budget. max_total_tokens is answered from
-- the live usage table PLUS this; every row deleted adds itself here first.
CREATE TABLE IF NOT EXISTS usage_rollup (
    key_id            INTEGER PRIMARY KEY,
    reqs              INTEGER NOT NULL DEFAULT 0,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens      INTEGER NOT NULL DEFAULT 0,
    through           TEXT
);
-- Fleet Pass: public, auto-approved, rate-limited demo keys issued from
-- the public site. public_models is the catalogue clients name by public_id;
-- public_domains decides who gets one automatically; public_keys is the
-- request/issue record (its own row, not just an api_keys one, because a
-- request can be pending or denied before any key ever exists); public_events
-- is the activity log the dashboard's Activity tab reads.
CREATE TABLE IF NOT EXISTS public_models (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  public_id TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL DEFAULT '',
  vendor TEXT NOT NULL DEFAULT '',
  description TEXT NOT NULL DEFAULT '',
  fleet_ids TEXT NOT NULL DEFAULT '[]',
  arch TEXT NOT NULL DEFAULT 'dense',
  params_b REAL NOT NULL DEFAULT 0,
  active_b REAL NOT NULL DEFAULT 0,
  allow_primary INTEGER NOT NULL DEFAULT 0,
  allow_worker INTEGER NOT NULL DEFAULT 0,
  ctx_max INTEGER NOT NULL DEFAULT 16384,
  ctx_default INTEGER NOT NULL DEFAULT 8192,
  enabled INTEGER NOT NULL DEFAULT 1,
  sort INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT
);
CREATE TABLE IF NOT EXISTS public_domains (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  domain TEXT NOT NULL UNIQUE,
  company TEXT NOT NULL DEFAULT '',
  source TEXT NOT NULL DEFAULT 'custom',
  rank INTEGER,
  mode TEXT NOT NULL DEFAULT 'allow',
  enabled INTEGER NOT NULL DEFAULT 1,
  added_at TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS public_keys (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  email TEXT NOT NULL,
  domain TEXT NOT NULL,
  company TEXT NOT NULL DEFAULT '',
  source TEXT NOT NULL DEFAULT '',
  kind TEXT NOT NULL,
  models TEXT NOT NULL DEFAULT '{}',
  ctx INTEGER NOT NULL DEFAULT 8192,
  key_id INTEGER,
  status TEXT NOT NULL DEFAULT 'pending',
  ip TEXT NOT NULL DEFAULT '',
  user_agent TEXT NOT NULL DEFAULT '',
  note TEXT NOT NULL DEFAULT '',
  decided_at TEXT, decided_by TEXT,
  emailed_at TEXT, email_error TEXT,
  fallbacks INTEGER NOT NULL DEFAULT 0,
  archived_at TEXT
);
CREATE TABLE IF NOT EXISTS public_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL, kind TEXT NOT NULL,  -- 'rejected' | 'rate_limited' | 'issued' | 'pending' |
                                          -- 'approved' | 'denied' | 'revoked' | 'extended' |
                                          -- 'resent' | 'fallback' | 'mail_error' | 'key_status'
                                          -- 'warm' (the email's load-my-model button was used)
                                          -- (key_status: the /public/api/key-status per-IP counter)
                                          -- 'demo' (one per /public/api/demo request: its per-IP counter)
  email TEXT NOT NULL DEFAULT '', ip TEXT NOT NULL DEFAULT '', detail TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS public_events_ts ON public_events(ts);
CREATE INDEX IF NOT EXISTS public_keys_email ON public_keys(email);
CREATE INDEX IF NOT EXISTS public_keys_domain ON public_keys(domain);
CREATE INDEX IF NOT EXISTS public_keys_status ON public_keys(status);
CREATE TABLE IF NOT EXISTS public_aliases (
  host TEXT PRIMARY KEY, box INTEGER NOT NULL UNIQUE, created_at TEXT NOT NULL
);
-- The largest context window each box will actually serve each model with,
-- as that box itself reported it on the last routing refresh. Persisted, not
-- just cached, because the ceiling a request form offers has to include
-- machines that are switched off right now: the biggest window in the fleet
-- usually lives on the biggest box, and the biggest box is not always awake.
CREATE TABLE IF NOT EXISTS model_ctx (
  host TEXT NOT NULL,
  model TEXT NOT NULL,
  ctx INTEGER NOT NULL,
  ts TEXT NOT NULL,
  PRIMARY KEY (host, model)
);
"""

# Columns added after the first release. ALTER TABLE is a no-op error on a
# database that already has them, which is exactly the behaviour wanted.
MIGRATIONS = (
    "ALTER TABLE api_keys ADD COLUMN expires_at TEXT",
    "ALTER TABLE api_keys ADD COLUMN max_rpd INTEGER",
    "ALTER TABLE api_keys ADD COLUMN max_tpd INTEGER",
    "ALTER TABLE api_keys ADD COLUMN max_total_tokens INTEGER",
    # Retention. `archived_at` is the timestamp a row left the live view, NULL
    # while it is still live -- one column carrying both the flag and the clock
    # the purge counts from. See the retention section below.
    "ALTER TABLE api_keys ADD COLUMN archived_at TEXT",
    "ALTER TABLE agents ADD COLUMN archived_at TEXT",
    "ALTER TABLE bench ADD COLUMN archived_at TEXT",
    "ALTER TABLE usage ADD COLUMN archived_at TEXT",
    "CREATE INDEX IF NOT EXISTS usage_archived ON usage(archived_at, id)",
    "CREATE INDEX IF NOT EXISTS bench_archived ON bench(archived_at, id)",
    # Which box actually served the request. This is the orchestrator's memory:
    # measured tokens/sec per (host, model) is what load balancing weighs.
    "ALTER TABLE usage ADD COLUMN host TEXT",
    # Fleet Pass: per-key hourly ceiling (api_keys.max_rpd was daily-only),
    # per-agent/team context caps enforced at the proxy, and which public id a
    # caller actually asked for when a fallback answered in its place.
    "ALTER TABLE api_keys ADD COLUMN max_rph INTEGER",
    "ALTER TABLE agents ADD COLUMN ctx_limit INTEGER",
    "ALTER TABLE teams ADD COLUMN ctx_limit INTEGER",
    "ALTER TABLE usage ADD COLUMN fallback_from TEXT",
    # Fleet Pass warm-up: the unguessable token behind the "load my model now"
    # button in the key email, and when it was last used.
    "ALTER TABLE public_keys ADD COLUMN warm_token TEXT",
    "ALTER TABLE public_keys ADD COLUMN warmed_at TEXT",
    # Which family a catalogue row belongs to (Gemma, Qwen, Nemotron ...) --
    # what the public page groups and orders by. Deliberately not the vendor
    # column: a community fine-tune of Qwen carries a different vendor and the
    # same family, and the family is the word a visitor recognises. Empty on
    # every row until backfill_public_families() runs at boot.
    "ALTER TABLE public_models ADD COLUMN family TEXT NOT NULL DEFAULT ''",
)

_db_lock = threading.Lock()


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def db_init() -> None:
    with closing(db()) as conn:
        conn.executescript(SCHEMA)
        for ddl in MIGRATIONS:
            try:
                conn.execute(ddl)
            except sqlite3.OperationalError:
                pass  # column already there
        conn.commit()


def db_exec(sql: str, args: Iterable[Any] = ()) -> int:
    with _db_lock, closing(db()) as conn:
        cur = conn.execute(sql, tuple(args))
        conn.commit()
        return int(cur.lastrowid or 0)


def db_update(sql: str, args: Iterable[Any] = ()) -> int:
    """Like db_exec, but answers "how many rows did that touch?".

    db_exec returns lastrowid, which an UPDATE or DELETE never sets, so it
    reports 0 for work that did plenty."""
    with _db_lock, closing(db()) as conn:
        cur = conn.execute(sql, tuple(args))
        conn.commit()
        return int(cur.rowcount or 0)


def db_query(sql: str, args: Iterable[Any] = ()) -> list[dict]:
    with closing(db()) as conn:
        return [dict(r) for r in conn.execute(sql, tuple(args)).fetchall()]



# --------------------------------------------------------------------------
# settings + retention
#
# Four lists on this dashboard grow forever: the request log, the benchmark
# history, and -- more slowly -- keys and agent profiles that are dead but
# still listed. The request log is the one that actually hurts; a busy box
# writes a row per completion and nothing ever collected them.
#
# So every one of those rows has a life:
#
#   LIVE      inside its rolling window; shown by default, one page at a time.
#   ARCHIVED  aged out of that window, or archived by hand. Out of the default
#             view, still in the database, still restorable.
#   DELETED   permanently, once it has been ARCHIVED for longer than the
#             retention period -- twelve months by default.
#
# Deleting from the dashboard archives; only the purge at the far end is
# final. Two things that make that safe rather than merely tidy:
#
#   * an ARCHIVED api key does not authenticate. Archiving is a revoke you can
#     undo, not a row that keeps working while hidden.
#   * purged usage rows are summed into usage_rollup first, so a key with a
#     lifetime token budget cannot be handed a fresh allowance by the purge.
# --------------------------------------------------------------------------

DEFAULT_SETTINGS: dict[str, Any] = {
    "retention_months": 12,      # archived -> permanently deleted after this
    "usage_window_days": 30,     # request log rows stay live this long
    "bench_window_days": 90,     # benchmark rows stay live this long
    "page_size": 25,             # rows per page in every paginated table
}

_SETTING_BOUNDS = {"retention_months": (1, 120), "usage_window_days": (1, 3650),
                   "bench_window_days": (1, 3650), "page_size": (5, 200)}


def get_settings() -> dict:
    """Stored settings over the defaults, every value clamped. A hand-edited
    row cannot make the purge delete everything today."""
    out = dict(DEFAULT_SETTINGS)
    for r in db_query("SELECT key,value FROM settings"):
        if r["key"] not in out:
            continue
        try:
            out[r["key"]] = json.loads(r["value"])
        except (TypeError, ValueError):
            pass
    for key, (lo, hi) in _SETTING_BOUNDS.items():
        try:
            out[key] = max(lo, min(hi, int(out[key])))
        except (TypeError, ValueError):
            out[key] = DEFAULT_SETTINGS[key]
    return out


def set_settings(updates: dict) -> dict:
    for key, val in updates.items():
        if key in DEFAULT_SETTINGS:
            db_exec(
                "INSERT INTO settings(key,value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(val)),
            )
    return get_settings()


def months_ago(months: int, ref: datetime | None = None) -> str:
    """ISO timestamp exactly `months` calendar months before now.

    Not months*30 days: the setting reads "12 months" and should mean this
    date last year. Returned as ISO text because every timestamp in this
    database is ISO text, so the comparison is a plain string compare."""
    dt = ref or datetime.now(timezone.utc)
    total = (dt.year * 12 + dt.month - 1) - int(months)
    year, month = divmod(total, 12)
    month += 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day).isoformat(
        timespec="seconds"
    )


def days_ago(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(
        timespec="seconds"
    )


def roll_window(settings: dict | None = None) -> dict:
    """Age rows past their window out of the live views.

    Idempotent -- it only touches rows whose archived_at is still NULL -- so
    it is safe at every startup and once a day after that. Keys and agent
    profiles are never aged out automatically: a key is archived when somebody
    decides it is dead, not because it went quiet."""
    cfg = settings or get_settings()
    stamp = now()
    return {
        "usage": db_update(
            "UPDATE usage SET archived_at=? WHERE archived_at IS NULL AND ts<?",
            (stamp, days_ago(cfg["usage_window_days"])),
        ),
        # A benchmark still queued or running is never aged out from under
        # its own worker thread.
        "bench": db_update(
            "UPDATE bench SET archived_at=? WHERE archived_at IS NULL "
            "AND created_at<? AND status NOT IN ('queued','running')",
            (stamp, days_ago(cfg["bench_window_days"])),
        ),
        # Finished batch jobs age with the request log; a running one is
        # never aged out from under its own dispatcher.
        "batches": db_update(
            "UPDATE batches SET archived_at=? WHERE archived_at IS NULL "
            "AND created_at<? AND status != 'running'",
            (stamp, days_ago(cfg["usage_window_days"])),
        ),
        # Fleet Pass: a public_keys row follows its api key wherever that key
        # got archived from -- the Public tab's revoke button, or the
        # ordinary Keys tab -- so this is a sync, not an age-out on its own
        # clock. A request that never became a key (pending/denied) has no
        # key_id and is never touched here.
        "public_keys": db_update(
            "UPDATE public_keys SET archived_at=? WHERE archived_at IS NULL "
            "AND key_id IN (SELECT id FROM api_keys WHERE archived_at IS NOT NULL)",
            (stamp,),
        ),
    }


def purge_expired(settings: dict | None = None) -> dict:
    """The only place rows leave this database for good.

    The clock starts when a row was ARCHIVED, not when it was created: twelve
    months of archive is twelve months you could still have restored it."""
    cfg = settings or get_settings()
    cutoff = months_ago(cfg["retention_months"])
    out: dict[str, Any] = {"cutoff": cutoff}
    with _db_lock, closing(db()) as conn:
        # Usage first, and roll the totals forward before deleting anything --
        # a key with a lifetime budget must not gain headroom from a purge.
        doomed = conn.execute(
            "SELECT key_id, COUNT(*) reqs, COALESCE(SUM(prompt_tokens),0) pt, "
            "COALESCE(SUM(completion_tokens),0) ct, "
            "COALESCE(SUM(total_tokens),0) tt, MAX(ts) through FROM usage "
            "WHERE archived_at IS NOT NULL AND archived_at < ? GROUP BY key_id",
            (cutoff,),
        ).fetchall()
        for r in doomed:
            conn.execute(
                "INSERT INTO usage_rollup(key_id,reqs,prompt_tokens,"
                "completion_tokens,total_tokens,through) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(key_id) DO UPDATE SET "
                "reqs=reqs+excluded.reqs, "
                "prompt_tokens=prompt_tokens+excluded.prompt_tokens, "
                "completion_tokens=completion_tokens+excluded.completion_tokens, "
                "total_tokens=total_tokens+excluded.total_tokens, "
                "through=MAX(COALESCE(through,''),excluded.through)",
                (r["key_id"], r["reqs"], r["pt"], r["ct"], r["tt"], r["through"]),
            )
        out["usage"] = conn.execute(
            "DELETE FROM usage WHERE archived_at IS NOT NULL AND archived_at<?",
            (cutoff,),
        ).rowcount
        out["bench"] = conn.execute(
            "DELETE FROM bench WHERE archived_at IS NOT NULL AND archived_at<?",
            (cutoff,),
        ).rowcount
        # A key going for good takes its agent profile with it; an orphan
        # profile is a row nothing can ever reach again.
        gone = [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM api_keys WHERE archived_at IS NOT NULL "
                "AND archived_at<?",
                (cutoff,),
            ).fetchall()
        ]
        for kid in gone:
            conn.execute("DELETE FROM agents WHERE key_id=?", (kid,))
            conn.execute("DELETE FROM teams WHERE key_id=?", (kid,))
        out["api_keys"] = conn.execute(
            "DELETE FROM api_keys WHERE archived_at IS NOT NULL "
            "AND archived_at<?",
            (cutoff,),
        ).rowcount
        out["agents"] = conn.execute(
            "DELETE FROM agents WHERE archived_at IS NOT NULL AND archived_at<?",
            (cutoff,),
        ).rowcount
        out["teams"] = conn.execute(
            "DELETE FROM teams WHERE archived_at IS NOT NULL AND archived_at<?",
            (cutoff,),
        ).rowcount
        # Batches take their spool files with them -- the row is bookkeeping,
        # the NDJSON on disk is the actual space.
        doomed_batches = [
            r["id"] for r in conn.execute(
                "SELECT id FROM batches WHERE archived_at IS NOT NULL "
                "AND archived_at<?", (cutoff,)).fetchall()
        ]
        out["batches"] = conn.execute(
            "DELETE FROM batches WHERE archived_at IS NOT NULL "
            "AND archived_at<?", (cutoff,),
        ).rowcount
        # Fleet Pass: public_keys follow their api key into the purge (an
        # issued/revoked one is only ever archived once its key is), and
        # public_events has no live/archived split at all -- "simplest:
        # delete rows older than retention_months" is the whole policy.
        out["public_keys"] = conn.execute(
            "DELETE FROM public_keys WHERE archived_at IS NOT NULL "
            "AND archived_at<?", (cutoff,),
        ).rowcount
        out["public_events"] = conn.execute(
            "DELETE FROM public_events WHERE ts<?", (cutoff,),
        ).rowcount
        conn.commit()
    for bid in doomed_batches:
        for p in _batch_paths(int(bid)):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
    # Deleting rows only frees pages inside the file; the file itself shrinks
    # only on VACUUM, and shrinking it is half the point of a purge. Its own
    # connection, because VACUUM cannot run inside a transaction, and
    # best-effort: a locked database is a reason to skip the tidy-up, never to
    # fail a purge that already worked.
    if any(v for k, v in out.items() if k != "cutoff"):
        try:
            with closing(db()) as conn:
                conn.isolation_level = None
                conn.execute("VACUUM")
            out["vacuumed"] = True
        except sqlite3.Error:
            out["vacuumed"] = False
    return out


def retention_stats(settings: dict | None = None) -> dict:
    """What retention is holding and what the next purge would take. Real
    counts, not estimates -- the button next to them deletes things."""
    cfg = settings or get_settings()
    cutoff = months_ago(cfg["retention_months"])
    tables = {}
    for table in ("usage", "bench", "api_keys", "agents", "teams", "batches"):
        row = db_query(
            "SELECT "
            "SUM(archived_at IS NULL) live, "
            "SUM(archived_at IS NOT NULL) archived, "
            "SUM(archived_at IS NOT NULL AND archived_at<?) purgeable "
            "FROM " + table,
            (cutoff,),
        )[0]
        tables[table] = {k: int(row[k] or 0) for k in
                         ("live", "archived", "purgeable")}
    try:
        db_bytes = DB_PATH.stat().st_size
    except OSError:
        db_bytes = 0
    return {"config": cfg, "cutoff": cutoff, "tables": tables,
            "db_bytes": db_bytes}


def maintain(settings: dict | None = None) -> dict:
    """Roll the window, then purge what has served its retention. One call, so
    startup and the daily task cannot drift apart."""
    cfg = settings or get_settings()
    return {"rolled": roll_window(cfg), "purged": purge_expired(cfg)}


# --------------------------------------------------------------------------
# api key auth (inference surface)
# --------------------------------------------------------------------------


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def _norm_limit(v: Any) -> int | None:
    try:
        n = int(v)
        return n if n > 0 else None
    except (TypeError, ValueError):
        return None


def _norm_expiry(v: Any) -> str | None:
    """Accept '', None, 'YYYY-MM-DD' or a full ISO timestamp. Empty = never."""
    s = str(v or "").strip()
    if not s:
        return None
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s  # date only; treated as valid through end of that day UTC
    try:
        datetime.fromisoformat(s)
        return s
    except ValueError as exc:
        raise HTTPException(400, "bad expires_at: " + s) from exc


def mint_key(
    name: str,
    expires_at: str | None = None,
    max_rpd: int | None = None,
    max_tpd: int | None = None,
    max_total_tokens: int | None = None,
    max_rph: int | None = None,
) -> tuple[str, dict]:
    raw = KEY_PREFIX + secrets.token_hex(24)
    kid = db_exec(
        "INSERT INTO api_keys(name, key_hash, prefix, created_at, expires_at,"
        " max_rpd, max_tpd, max_total_tokens) VALUES (?,?,?,?,?,?,?,?)",
        (
            name,
            hash_key(raw),
            raw[: len(KEY_PREFIX) + 6],
            now(),
            expires_at,
            max_rpd,
            max_tpd,
            max_total_tokens,
        ),
    )
    if max_rph:
        db_exec("UPDATE api_keys SET max_rph=? WHERE id=?", (max_rph, kid))
    return raw, {"id": kid, "name": name, "prefix": raw[: len(KEY_PREFIX) + 6]}


def bearer(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key") or request.query_params.get("api_key")


# Which usage rows count as "a request" against a key's request budgets.
# Not a team's internal rounds (/v1/team-primary, /v1/team-worker -- the crew
# doing the client's one request), not a model listing (every chat client
# calls /v1/models on startup; that must never spend a 2/hour allowance), and
# not a request the caller got nothing for (4xx/5xx: a typo'd model or a dead
# backend). Token budgets keep counting everything. _U is the same predicate
# for queries that alias the usage table as `u`.
BUDGET_REQ_SQL = ("endpoint NOT LIKE '/v1/team-%' AND endpoint != '/v1/models' "
                  "AND endpoint != '/v1/warm' AND status < 400")
BUDGET_REQ_SQL_U = ("u.endpoint NOT LIKE '/v1/team-%' AND u.endpoint != '/v1/models' "
                    "AND u.endpoint != '/v1/warm' AND u.status < 400")


def require_api_key(request: Request) -> dict:
    raw = bearer(request)
    if not raw:
        raise HTTPException(401, "missing bearer token")
    # archived_at IS NULL matters as much as disabled=0: archiving a key is a
    # revoke you can undo, not a row that keeps authenticating while hidden
    # from the dashboard.
    rows = db_query(
        "SELECT * FROM api_keys WHERE key_hash=? AND disabled=0 "
        "AND archived_at IS NULL",
        (hash_key(raw),),
    )
    if not rows:
        raise HTTPException(401, "invalid api key")
    key = rows[0]

    exp = key.get("expires_at")
    if exp:
        cutoff = exp + "T23:59:59+00:00" if len(exp) == 10 else exp
        if now() > cutoff:
            raise HTTPException(401, "api key expired on " + exp)

    # Budgets. "Per day"/"per hour" are rolling windows measured from the
    # usage log, so a restart never resets anyone's meter. Request counts
    # (but not token counts -- those keep counting everything) exclude a
    # team's internal rounds and anything a dead backend never actually
    # served: /v1/team-primary and /v1/team-worker rows are the crew doing
    # its own work on the client's behalf, not a second request from the
    # client, and a status>=500 row is a request the caller got nothing for.
    CLIENT_FACING_SQL = BUDGET_REQ_SQL
    # The same two endpoints are exempt from being STOPPED by a request
    # budget, not merely from filling it. Every chat client lists models on
    # startup and the key mail's warm-up button is a page load, so a key that
    # has spent its hour would answer neither -- and a client that cannot list
    # models reports itself broken ("no models available") rather than rate
    # limited, which is the one thing the caller needed to be told.
    spends_budget = request.url.path not in ("/v1/models", "/v1/warm")
    if spends_budget and (
            key.get("max_rpd") or key.get("max_tpd") or key.get("max_rph")):
        day_ago = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(
            timespec="seconds"
        )
        c24 = db_query(
            "SELECT COUNT(*) c FROM usage WHERE key_id=? AND ts >= ? AND "
            + CLIENT_FACING_SQL,
            (key["id"], day_ago),
        )[0]["c"]
        t24 = db_query(
            "SELECT COALESCE(SUM(total_tokens),0) t FROM usage "
            "WHERE key_id=? AND ts >= ?",
            (key["id"], day_ago),
        )[0]["t"]
        if key.get("max_rpd") and int(c24) >= int(key["max_rpd"]):
            raise HTTPException(
                429, "daily request budget reached (" + str(key["max_rpd"]) + "/24h)"
            )
        if key.get("max_tpd") and int(t24) >= int(key["max_tpd"]):
            raise HTTPException(
                429, "daily token budget reached (" + str(key["max_tpd"]) + "/24h)"
            )
        if key.get("max_rph"):
            hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(
                timespec="seconds"
            )
            c1 = db_query(
                "SELECT COUNT(*) c FROM usage WHERE key_id=? AND ts >= ? AND "
                + CLIENT_FACING_SQL,
                (key["id"], hour_ago),
            )[0]["c"]
            if int(c1) >= int(key["max_rph"]):
                raise HTTPException(
                    429, "hourly request budget reached (" + str(key["max_rph"])
                    + "/1h)"
                )
    if key.get("max_total_tokens"):
        # Live rows plus whatever the retention purge already deleted. Without
        # the rollup, a purge would silently hand a capped key a fresh
        # allowance -- the meter has to outlive the log it was read from.
        tot = db_query(
            "SELECT COALESCE(SUM(total_tokens),0) t FROM usage WHERE key_id=?",
            (key["id"],),
        )
        rolled = db_query(
            "SELECT COALESCE(SUM(total_tokens),0) t FROM usage_rollup "
            "WHERE key_id=?",
            (key["id"],),
        )
        if int(tot[0]["t"]) + int(rolled[0]["t"]) >= int(key["max_total_tokens"]):
            raise HTTPException(429, "lifetime token budget exhausted")

    db_exec("UPDATE api_keys SET last_used_at=? WHERE id=?", (now(), key["id"]))
    return key


# --------------------------------------------------------------------------
# cloudflare access auth (admin surface)
# --------------------------------------------------------------------------

_jwks_client: Any = None


def jwks() -> Any:
    global _jwks_client
    if _jwks_client is None:
        if not CF_TEAM_DOMAIN:
            raise HTTPException(503, "CF_ACCESS_TEAM_DOMAIN not configured")
        url = "https://" + CF_TEAM_DOMAIN + "/cdn-cgi/access/certs"
        _jwks_client = jwt.PyJWKClient(url, cache_keys=True, lifespan=600)
    return _jwks_client


def require_admin(request: Request) -> dict:
    """
    Admin access is granted by EITHER
      (a) a signature-verified Cloudflare Access JWT whose aud matches this
          application, or
      (b) the out-of-band admin token (CLI / Tailscale recovery).

    The Cf-Access-Authenticated-User-Email header alone is never trusted: it is
    forgeable by anything that can reach the origin, and the API hostname
    deliberately has no Access policy in front of it.
    """
    tok = bearer(request)
    if tok and ADMIN_TOKEN and secrets.compare_digest(tok, ADMIN_TOKEN):
        return {"email": "admin-token", "via": "token"}

    assertion = request.headers.get("cf-access-jwt-assertion") or request.cookies.get(
        "CF_Authorization"
    )
    if not assertion:
        raise HTTPException(401, "no Cloudflare Access assertion")
    if not CF_AUD:
        raise HTTPException(503, "CF_ACCESS_AUD not configured")
    try:
        signing_key = jwks().get_signing_key_from_jwt(assertion)
        claims = jwt.decode(
            assertion,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            audience=CF_AUD,
            issuer="https://" + CF_TEAM_DOMAIN,
        )
    except Exception as exc:  # noqa: BLE001 -- any failure means refuse
        raise HTTPException(401, "invalid Access token: " + str(exc)) from exc

    email = str(claims.get("email", "")).lower()
    if ADMIN_EMAILS and email not in ADMIN_EMAILS:
        raise HTTPException(403, email + " is not an administrator")
    return {"email": email, "via": "cf-access"}


# --------------------------------------------------------------------------
# model registry  (models.json  ->  llama-swap.yaml)
# --------------------------------------------------------------------------

DEFAULT_MODEL_RECORD = {
    "id": "",
    "path": "",
    "enabled": True,
    "description": "",
    "aliases": [],
    # 0 = size it from the hardware (see resolve_ctx); any positive number is
    # a pin and is passed through untouched.
    "ctx": 0,
    "ngl": 99,
    "n_cpu_moe": 0,
    "threads": 0,
    "parallel": 1,
    "flash_attn": True,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "jinja": True,
    # Qwen3.x and Gemma 4 are hybrid-reasoning models. Left on "auto" they will
    # happily spend an entire max_tokens budget thinking and return empty
    # content -- which reads as a broken endpoint from a coding agent. "off"
    # suppresses the thinking phase entirely.
    "reasoning": "auto",
    # Seconds idle before llama-swap unloads the model; 0 means never. A box
    # whose job is to hold one model warm sets 0 on that model and nothing
    # else, so the swap slot is only ever taken from it by an explicit request
    # for something else -- never by a timer.
    "ttl": 900,
    # Load at llama-swap start-up, before any request arrives. Rendered as the
    # swap config's hooks.on_startup.preload. With ttl 0 this is what "warm on
    # boot" means for a box that serves one model and does nothing else: the
    # cold load is paid once, when the machine comes up, instead of by whoever
    # sends the first request after every reboot. Only one enabled model per
    # box may carry it -- the swap group is exclusive, so a second preload
    # would just evict the first.
    "preload": False,
    # Never evicted. Rendered into its own llama-swap group -- persistent,
    # non-swapping -- beside the main exclusive group, so a request for any
    # OTHER model loads that model next to this one instead of in its place.
    # Implies preload. For a box with a small always-on model (a 9B distill
    # at ~10 GiB) and a budget that can hold a second, larger model beside
    # it: the always-on one stops paying a cold load every time somebody
    # asks for the big one. The box's ctx pins have to leave room for it --
    # the auto-sizer budgets each model against the whole pool and does not
    # know a neighbour is resident, so on a box with a persistent model the
    # other entries are pinned by hand (hosts/apu-tablet-2/register-models.ps1).
    "persistent": False,
    "extra_flags": "",
    "mmproj": "",
}

REASONING_MODES = ("auto", "on", "off")


# ---- GGUF metadata: size a context window without loading anything ---------
#
# A GGUF header is a flat key/value table in front of the tensors, so the
# geometry that decides KV-cache cost -- layers, KV heads, head dims, the
# trained context length -- is readable in one buffered read. That is the
# whole basis for auto-sizing: no ROCm, no CUDA, no trial load.

_GGUF_SCALARS = {
    0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4),
    5: ("i", 4), 6: ("f", 4), 7: ("?", 1), 10: ("Q", 8), 11: ("q", 8),
    12: ("d", 8),
}
_GGUF_STRING, _GGUF_ARRAY = 8, 9
_GGUF_HEAD_BYTES = 32 * 1024 * 1024
_gguf_cache: dict[str, tuple[float, dict]] = {}


def _gguf_str(buf: bytes, off: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<Q", buf, off)
    off += 8
    return buf[off:off + n].decode("utf-8", "replace"), off + n


def _gguf_value(buf: bytes, off: int, vtype: int) -> tuple[Any, int]:
    if vtype == _GGUF_STRING:
        return _gguf_str(buf, off)
    if vtype == _GGUF_ARRAY:
        (itype,) = struct.unpack_from("<I", buf, off)
        (count,) = struct.unpack_from("<Q", buf, off + 4)
        off += 12
        if itype == _GGUF_STRING:
            # The vocab lives here -- 150k strings, the bulk of the header.
            # Walked past by length, never decoded.
            for _ in range(count):
                (n,) = struct.unpack_from("<Q", buf, off)
                off += 8 + n
            return None, off
        if itype in _GGUF_SCALARS:
            return None, off + _GGUF_SCALARS[itype][1] * count
        raise ValueError("nested gguf array")
    fmt, size = _GGUF_SCALARS[vtype]
    (v,) = struct.unpack_from("<" + fmt, buf, off)
    return v, off + size


def gguf_meta(path: str) -> dict:
    """Architecture and geometry from a GGUF header. {} for anything this
    cannot read -- every caller has a fixed fallback."""
    if not path:
        return {}
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        return {}
    hit = _gguf_cache.get(path)
    if hit and hit[0] == stamp:
        return hit[1]

    raw: dict[str, Any] = {}
    try:
        with open(path, "rb") as fh:
            buf = fh.read(_GGUF_HEAD_BYTES)
        if buf[:4] != b"GGUF":
            return {}
        (version,) = struct.unpack_from("<I", buf, 4)
        if version < 2:
            return {}
        (n_kv,) = struct.unpack_from("<Q", buf, 16)
        off = 24
        for _ in range(min(int(n_kv), 4096)):
            key, off = _gguf_str(buf, off)
            (vtype,) = struct.unpack_from("<I", buf, off)
            off += 4
            val, off = _gguf_value(buf, off, vtype)
            if val is not None and not key.startswith("tokenizer."):
                raw[key] = val
    except Exception:  # noqa: BLE001
        return {}

    arch = str(raw.get("general.architecture") or "")

    def geo(suffix: str) -> int | None:
        v = raw.get(arch + "." + suffix)
        try:
            return int(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    meta = {
        "arch": arch,
        "name": raw.get("general.name"),
        "n_ctx_train": geo("context_length"),
        "block_count": geo("block_count"),
        "head_count": geo("attention.head_count"),
        "head_count_kv": geo("attention.head_count_kv"),
        "key_length": geo("attention.key_length"),
        "value_length": geo("attention.value_length"),
        "embedding_length": geo("embedding_length"),
        "expert_count": geo("expert_count"),
        "sliding_window": geo("attention.sliding_window"),
    }
    _gguf_cache[path] = (stamp, meta)
    return meta


# Bits per cached element, including the block scales that quantised KV pays.
_KV_BITS = {"f32": 32.0, "f16": 16.0, "bf16": 16.0, "q8_0": 8.5, "q5_1": 6.0,
            "q5_0": 5.5, "q4_1": 5.0, "q4_0": 4.5, "iq4_nl": 4.5}


def kv_bytes_per_token(meta: dict, ctk: Any, ctv: Any) -> float | None:
    """Bytes of KV cache one token costs across every layer.

    Full attention is assumed. Models with a sliding window (this fleet has
    two) cache far less than this on their windowed layers, so the estimate
    errs toward a SMALLER context than the box could hold -- the safe
    direction for a number that decides whether a load OOMs."""
    n_layer = meta.get("block_count")
    n_kv = meta.get("head_count_kv")
    if not n_layer or not n_kv:
        return None
    k_len, v_len = meta.get("key_length"), meta.get("value_length")
    if not k_len or not v_len:
        emb, heads = meta.get("embedding_length"), meta.get("head_count")
        if not emb or not heads:
            return None
        k_len = v_len = emb // heads
    kb = _KV_BITS.get(str(ctk or "f16").lower(), 16.0)
    vb = _KV_BITS.get(str(ctv or "f16").lower(), 16.0)
    return n_layer * n_kv * (k_len * kb + v_len * vb) / 8.0


_SHARD = re.compile(r"-\d{5}-of-(\d{5})\.gguf$", re.I)


def model_bytes(path: Any) -> int:
    """Bytes of weights behind a model record, shards included.

    A split GGUF names its first shard in the record, and that shard is a
    ~8 MB header: the 120B's real 83 GB lives in parts 2 and 3. Sizing a
    context window off part 1 alone would hand the KV cache the whole GPU and
    OOM the load."""
    p = str(path or "")
    if not p:
        return 0
    m = _SHARD.search(p)
    if m:
        stem = p[:m.start()]
        total = 0
        for i in range(1, int(m.group(1)) + 1):
            try:
                total += os.path.getsize("%s-%05d-of-%s.gguf" % (stem, i, m.group(1)))
            except OSError:
                pass
        if total:
            return total
    try:
        return os.path.getsize(p)
    except (OSError, TypeError):
        return 0


def vram_total_bytes() -> int | None:
    """This box's VRAM pool, measured where a driver will say and taken off
    the fleet spec sheet where none will. The policy is hw.vram_total_bytes();
    the spec sheet and the backend names are supplied here so that replacing
    any of them on this module replaces the one that actually runs."""
    return hw.vram_total_bytes(_spec_vram_gb(), amd=amdgpu_stats, nv=nvidia_stats,
                               win=windows_gpu_stats, mac=darwin_gpu_stats)


# Normally a llama.cpp model's weights and KV cache both live inside dedicated
# VRAM, so `vram_total_bytes()` is the right pool to size against. A Windows
# APU is different: WDDM can extend a model beyond its UMA carve-out into the
# remaining unified memory. A host may explicitly budget that larger pool,
# but it must do so as a hard total (weights + KV + graph scratch), leaving the
# rest available to Windows. No host gets this behaviour by default.
def context_budget_bytes() -> int | None:
    raw = os.environ.get("LLMSTACK_CONTEXT_BUDGET_GIB", "").strip()
    if raw:
        try:
            gib = float(raw)
            if 1.0 <= gib <= 4096.0:
                return int(gib * 1024 ** 3)
        except (TypeError, ValueError):
            pass
    return vram_total_bytes()


AUTO_CTX_CAP = int(os.environ.get("LLMSTACK_MAX_AUTO_CTX", "262144"))
AUTO_CTX_FALLBACK = 32768
VRAM_HEADROOM = float(os.environ.get("LLMSTACK_VRAM_HEADROOM", "0.90"))


def resolve_ctx(rec: dict) -> tuple[int, dict]:
    """The context window this model should actually start with.

    `ctx` > 0 in the record is a promise and is honoured exactly: a tuned
    model stays tuned. `ctx` of 0 (or "auto") means use the box -- the largest
    window that still fits VRAM once the weights, the projector and a
    compute-buffer allowance are paid for, capped by what the model was
    trained for. A 96 GB box holding a 17 GB model at the old fixed 32k
    default was leaving three quarters of its memory doing nothing."""
    want = rec.get("ctx", 0)
    if isinstance(want, str):
        want = 0 if want.strip().lower() in ("", "auto") else want
    try:
        want = int(want)
    except (TypeError, ValueError):
        want = 0
    if want > 0:
        return want, {"mode": "pinned", "ctx": want}

    detail: dict[str, Any] = {"mode": "auto"}
    meta = gguf_meta(str(rec.get("path", "")))
    per_tok = kv_bytes_per_token(meta, rec.get("cache_type_k"),
                                 rec.get("cache_type_v"))
    # This is intentionally not the telemetry value: see
    # `context_budget_bytes()` for the opt-in Windows unified-memory case.
    vram = context_budget_bytes()
    weights = model_bytes(rec.get("path")) + model_bytes(rec.get("mmproj"))
    trained = meta.get("n_ctx_train") or 0
    cap = min(trained or AUTO_CTX_CAP, AUTO_CTX_CAP)
    if not per_tok or not vram or not weights:
        # Hybrid state-space architectures (nemotron_h) publish no attention
        # KV geometry at all, so there is nothing to compute a budget from.
        # Guessing large here is how you OOM a 120B; the modest default holds
        # until someone pins a measured number.
        ctx = min(trained or AUTO_CTX_FALLBACK, AUTO_CTX_FALLBACK)
        detail.update(ctx=ctx, trained=trained,
                      why="no usable GGUF KV geometry or VRAM reading; "
                          "holding at the conservative default")
        return ctx, detail

    parallel = max(1, int(rec.get("parallel", 1) or 1))
    # Compute buffers, graph scratch and the projector's own working set. ~1 GB
    # on a 30B, and it grows with the number of slots being batched.
    overhead = int((1.0 + 0.5 * parallel) * 1024 ** 3)
    # --n-cpu-moe N keeps the routed-expert tensors of N layers in system RAM,
    # and on an A3B-class MoE those tensors are nearly the whole file. Charging
    # the GPU for weights that are not on it is how an 8 GB laptop that happily
    # serves 32k got sized down to the 4096 floor. Charge it the share of
    # layers whose experts stayed, with a floor for the attention stack,
    # embeddings and output head, which are always resident.
    n_cpu_moe = max(0, int(rec.get("n_cpu_moe", 0) or 0))
    n_layer = int(meta.get("block_count") or 0)
    gpu_weights = weights
    if n_cpu_moe > 0 and n_layer > 0:
        gpu_weights = int(weights * max(0.10, max(0, n_layer - n_cpu_moe) / n_layer))
    budget = int(vram * VRAM_HEADROOM) - gpu_weights - overhead
    ctx = int(budget / per_tok) if budget > 0 else 0
    ctx -= ctx % 4096
    ctx = max(4096, min(ctx, cap))
    detail.update(ctx=ctx, trained=trained, vram=vram, weights=weights,
                  gpu_weights=gpu_weights, n_cpu_moe=n_cpu_moe,
                  kv_bytes_per_token=round(per_tok, 1), headroom=VRAM_HEADROOM,
                  arch=meta.get("arch"), capped=bool(ctx >= cap))
    return ctx, detail


def load_models() -> list[dict]:
    if not MODELS_JSON.exists():
        return []
    try:
        data = json.loads(MODELS_JSON.read_text())
    except json.JSONDecodeError:
        return []
    out = []
    for rec in data if isinstance(data, list) else []:
        merged = dict(DEFAULT_MODEL_RECORD)
        merged.update(rec)
        out.append(merged)
    return out


def write_atomic(path: Path, text: str) -> None:
    """Replace `path` with `text` in one step, without a shared temp name.

    "<name>.tmp" is only safe while exactly one writer exists. Two overlapping
    calls -- an admin save and the apply queue's rollback, a request and the
    task that outlived it -- write the same temp file, and whichever replaces
    second finds nothing to replace: FileNotFoundError, from the code whose
    whole job is to make the write crash-safe. mkstemp gives every writer its
    own file, so the loser of a race loses its content, not the process.

    mkstemp also creates at 0600, which would quietly lock out the reader
    running as another user (llama-swap reads the config this writes), so an
    existing file's mode is carried over and a new one lands at 0644.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        mode = 0o644
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_models(models: list[dict]) -> None:
    write_atomic(MODELS_JSON, json.dumps(models, indent=2))


SAFE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


def build_cmd(rec: dict) -> str:
    parts = [
        LLAMA_SERVER,
        "--model",
        str(rec["path"]),
        "--host",
        "127.0.0.1",
        "--port",
        "${PORT}",
        "--alias",
        str(rec["id"]),
        "-ngl",
        str(int(rec.get("ngl", 99))),
        "-c",
        str(resolve_ctx(rec)[0]),
        "--parallel",
        str(int(rec.get("parallel", 1) or 1)),
    ]
    if int(rec.get("n_cpu_moe", 0) or 0) > 0:
        parts += ["--n-cpu-moe", str(int(rec["n_cpu_moe"]))]
    if int(rec.get("threads", 0) or 0) > 0:
        parts += ["-t", str(int(rec["threads"]))]
    if rec.get("flash_attn", True):
        parts += ["-fa", "on"]
    if rec.get("cache_type_k"):
        parts += ["--cache-type-k", str(rec["cache_type_k"])]
    if rec.get("cache_type_v"):
        parts += ["--cache-type-v", str(rec["cache_type_v"])]
    if rec.get("jinja", True):
        parts += ["--jinja"]
    mode = str(rec.get("reasoning", "auto")).lower()
    if mode in ("on", "off"):
        parts += ["--reasoning", mode]
    if rec.get("mmproj"):
        parts += ["--mmproj", str(rec["mmproj"])]
    extra = str(rec.get("extra_flags", "") or "").strip()
    cmd = " ".join(parts)
    if extra:
        cmd = cmd + " " + extra
    return cmd


def check_name_collisions(models: list[dict]) -> None:
    """Every model id and alias must be unique across the whole registry.

    llama-swap rejects a config with a duplicate alias by refusing to load
    ANY of it, and systemd then restart-loops the service, so one careless
    alias takes the whole box off the air until someone reads the journal.
    Cloning a record to make a second profile of the same weights - the
    obvious way to add a classification variant - collides on every alias the
    original had, which is exactly how this was found. Fail the save with a
    400 instead of writing a config that cannot be loaded."""
    seen: dict[str, str] = {}
    for rec in models:
        if not rec.get("enabled", True):
            continue
        mid = str(rec.get("id", "")).strip()
        names = [mid] + [str(a).strip() for a in rec.get("aliases") or []
                         if isinstance(a, str) and str(a).strip()]
        for name in names:
            owner = seen.get(name)
            if owner is not None:
                raise HTTPException(
                    400,
                    f"{name!r} is claimed by both {owner!r} and {mid!r} - "
                    f"model ids and aliases share one namespace, and "
                    f"llama-swap refuses the entire config over a duplicate")
            seen[name] = mid


def check_preload_count(models: list[dict]) -> None:
    """At most one enabled model may be preloaded at start-up.

    The swap group is exclusive -- one resident model per box -- so a second
    preload entry does not give a box two warm models, it gives it one cold
    load immediately followed by another that evicts it. Refusing the save
    is better than a config that quietly does twice the work for the same
    result, and it keeps "which model is this box for" a one-line answer."""
    warm = [str(r.get("id", "")).strip() for r in models
            if r.get("enabled", True) and r.get("preload")
            and not r.get("persistent")]
    if len(warm) > 1:
        raise HTTPException(
            400,
            "only one enabled model may be preloaded into the swap group "
            "(it holds one model at a time); preload is set on: "
            + ", ".join(warm) + " -- mark one `persistent` instead if the "
            "box can hold both")


# ---- one id per model, fleet-wide ------------------------------------------
#
# The canonical fleet id of every model that has been registered under more
# than one name, and each spelling it has been seen under. Candidates for a
# request are grouped by NAME (model_routes), so two boxes holding the same
# weights under different names are never each other's alternatives: the
# better box is simply invisible for that request. split_models() reports
# that; this is what repairs it. The canonical spelling is the one the
# llama.cpp boxes and the hub's catalogue already used; the rest are what the
# fleet grew on its own -- Ollama tags, LM Studio's derived ids, a name typed
# by hand on one box.
#
# Every name here has to be claimed by one row of PUBLIC_MODELS_SEED as well
# (its fleet_ids), so the catalogue and the routing layer agree on what is
# one model. tests/test_model_names.py pins that.
#
# What is done with it, per box, when the gateway starts -- which is to say
# on every deploy, and when a box that was off comes back and the reconcile
# deploy restarts it:
#   * llama.cpp box: a models.json record whose id is a spelling is RENAMED
#     to the canonical id, once, and the old id is kept as an alias so
#     nothing that was calling it breaks (converge_model_names).
#   * Ollama box: a tag cannot be renamed, so the gateway answers to the
#     canonical id beside it and rewrites it back to the tag on the way into
#     the engine (upstream_alias_pairs).
FLEET_MODEL_NAMES: dict[str, tuple[str, ...]] = {
    "qwen3.8-9b-distill": (
        "qwen3.8-9B",                    # gpu-laptop-1, registered by hand
        "Qwen3.8-9B",                    # mac-desktop-1's Library download
        "empero-ai-qwen3.8-9b-distill",  # mac-laptop-2: LM Studio's publisher-qualified id
        "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M",  # the Ollama tag
    ),
    "qwopus3.6-35b-coder": ("Qwopus3.6-35B", "Qwopus3.6-A3B"),
    "gemma-4-26b": ("gemma4:26b",),
    "qwen3.5-4b": ("qwen3.5:4b",),
}
SPELLING_TO_CANONICAL: dict[str, str] = {
    s: c for c, spellings in FLEET_MODEL_NAMES.items() for s in spellings}


# ---- fleet roles: one word, the best model for it on each box ---------------
#
# A role alias used to be a row in each box's models.json: `fast` meant a 4B
# on gpu-desktop-1, the 35B MoE on server-1, Lightning on mac-laptop-1, the 9B distill on
# apu-tablet-2, and a request for it through the hub got whichever box the
# scorer liked -- the best machine for a coin flip. This is the fleet-level
# meaning. Per role, the models that can play it, best first, by canonical
# id. Resolving one is a policy, not a lookup (role_pairs): each box offers
# the best model ON THAT BOX -- the first ladder entry it holds in its own
# memory, else the first it serves at all -- and the scorer ranks the boxes:
# among those that hold their model, the one higher on the ladder first
# (`deep` goes to the 120B on the big box, not to a warm 27B on a laptop),
# then the usual order; boxes that would spill or run it on CPU last. So
# `fast` on gpu-desktop-1 is still the 4B, on a card that holds it it is the 35B
# MoE, and the hub picks the box.
#
# Ladders are ordered by how well the model plays the role, not by size:
# `fast` leads with the 3B-active MoEs because they decode like a 3B and
# answer like a 35B, and only a box that cannot hold one falls to the 4B.
# A box's own models.json row of the same name still resolves locally for
# anything talking to llama-swap directly, but the gateway answers the
# policy first, so those rows no longer count as conflicts.
FLEET_ROLES: dict[str, tuple[str, ...]] = {
    "fast": ("qwen3.6-35b", "nemotron3.5-lightning-30b", "qwen3.8-9b-distill",
             "qwen3.5-4b", "gemma4:12b-it-qat", "nemotron-3-nano:4b",
             "qwen3.5:9b", "gemma4:e4b", "nemotron-mini:4b"),
    "small": ("qwen3.8-9b-distill", "qwen3.5-4b", "gemma4:12b-it-qat",
              "nemotron-3-nano:4b", "gemma4:e4b", "nemotron-mini:4b"),
    "default": ("qwen3.8-27b", "qwen3.6-35b", "qwen3.8-9b-distill", "qwen3.5-4b"),
    "quality": ("qwen3.8-27b", "gemma4-31b-qat", "nemotron3-super-120b",
                "qwen3.6-35b", "gemma-4-26b"),
    "deep": ("nemotron3-super-120b", "qwen3.8-flash-next", "deepseek-v4-flash",
             "qwen3.8-27b", "gemma4-31b-qat", "qwen3.6-35b"),
    "triage": ("qwen3.6-35b", "nemotron3.5-lightning-30b", "qwen3.8-9b-distill",
               "qwen3.5-4b"),
    "classify": ("qwen3-vl-30b-classify", "qwen3-vl-30b", "qwen3.6-35b",
                 "qwen3.8-9b-distill"),
    "coder": ("qwen3-coder-30b", "qwopus3.6-35b-coder", "kat-coder-v2.5",
              "qwen3.6-35b"),
    "vision": ("qwen3-vl-30b", "qwen3.8-27b", "gemma4-31b-qat", "gemma-4-26b",
               "gemma4:12b-it-qat"),
    "qwen": ("qwen3.8-27b", "qwen3.6-35b", "qwen3.8-9b-distill", "qwen3.5-4b",
             "qwen3.5:9b"),
    "gemma": ("gemma4-31b-qat", "gemma-4-26b", "gemma4:12b-it-qat", "gemma4:e4b"),
    "nemotron": ("nemotron3-super-120b", "nemotron3.5-lightning-30b",
                 "nemotron-3-nano:4b", "nemotron-mini:4b"),
}


def converge_model_names(models: list[dict]) -> tuple[list[dict], list[dict]]:
    """Rename every record whose id is a known spelling to its canonical id.

    Pure: returns (records, changes) and touches nothing on disk. The old id
    becomes an alias -- a client configured with `qwen3.8-9B` goes on
    working; it is only the fleet that stops thinking there are two models --
    and the canonical id is dropped from the aliases if it was already
    there, since ids and aliases share one namespace.

    Skipped, with a note, when the canonical id is already taken on this box
    by ANOTHER record, as an id or an alias: two records for one model is a
    deliberate variant (a classify build, a different window) and which one
    is the real one is the owner's call, not a start-up pass's."""
    taken: dict[str, str] = {}   # name -> the id that owns it
    for rec in models:
        # Every record's id, disabled ones included: a rename must not leave
        # two records with one id, even when one of them is switched off.
        mid = str(rec.get("id", "")).strip()
        if mid:
            taken.setdefault(mid, mid)
    for rec in models:
        if not rec.get("enabled", True):
            continue
        mid = str(rec.get("id", "")).strip()
        for a in rec.get("aliases") or []:
            if isinstance(a, str) and a.strip():
                taken.setdefault(a.strip(), mid)
    out: list[dict] = []
    changes: list[dict] = []
    for rec in models:
        mid = str(rec.get("id", "")).strip()
        canon = SPELLING_TO_CANONICAL.get(mid)
        if not canon or canon == mid or not rec.get("enabled", True):
            out.append(rec)
            continue
        owner = taken.get(canon)
        if owner and owner != mid:
            changes.append({"id": mid, "to": canon, "skipped":
                            f"{canon!r} already belongs to {owner!r} on this box"})
            out.append(rec)
            continue
        aliases = [a.strip() for a in rec.get("aliases") or []
                   if isinstance(a, str) and a.strip() and a.strip() != canon]
        if mid not in aliases:
            aliases.append(mid)
        out.append(dict(rec, id=canon, aliases=aliases))
        changes.append({"id": mid, "to": canon})
    return out, changes


# What the last start-up pass did, for the Models tab: a rename the owner
# did not type deserves a line on the page that shows the result.
_converged: dict[str, Any] = {"at": "", "changes": [], "restart_rc": 0}


def apply_model_name_convergence() -> list[dict]:
    """The start-up pass: converge_model_names() over models.json, written
    back and applied to the engine only when something actually changed.

    Not on an Ollama box -- its catalogue is the engine's, not models.json;
    see upstream_alias_pairs() for what happens there. The engine is
    restarted the way Save & apply restarts it, but nothing is queued for
    verification: the launch command differs only in its --alias string, and
    a box that is still coming up is no place to start proving loads."""
    if UPSTREAM_MODELS:
        return []
    new, changes = converge_model_names(load_models())
    for c in changes:
        if c.get("skipped"):
            log.warning("model names: not renaming %s -> %s: %s",
                        c["id"], c["to"], c["skipped"])
    renamed = [c for c in changes if not c.get("skipped")]
    if not renamed:
        if changes:
            _converged.update(at=now(), changes=changes, restart_rc=0)
        return changes
    try:
        check_name_collisions(new)
    except HTTPException as exc:
        log.warning("model names: the rename would collide, left as is: %s",
                    exc.detail)
        return changes
    save_models(new)
    write_swap_config(new)
    code, out = service_control("restart", "llama-swap")
    for c in renamed:
        log.info("model names: renamed %s -> %s (the old name stays as an alias)",
                 c["id"], c["to"])
    if code:
        log.warning("model names: llama-swap did not restart after the rename "
                    "(rc=%s): %s", code, (out or "")[-200:])
    _routes_cache["t"] = 0.0
    _converged.update(at=now(), changes=changes, restart_rc=code)
    return changes


def render_swap_config(models: list[dict]) -> str:
    cfg: dict[str, Any] = {
        "healthCheckTimeout": 900,
        "logLevel": "info",
        "startPort": 5800,
        "metricsMaxInMemory": 5000,
        "models": {},
    }
    for rec in models:
        if not rec.get("enabled", True):
            continue
        mid = str(rec.get("id", "")).strip()
        if not SAFE_ID.match(mid) or not rec.get("path"):
            continue
        entry: dict[str, Any] = {
            "cmd": build_cmd(rec),
            "ttl": int(rec.get("ttl", 900) or 0),
        }
        if rec.get("description"):
            entry["description"] = str(rec["description"])
        aliases = [a for a in rec.get("aliases", []) if isinstance(a, str) and a.strip()]
        if aliases:
            entry["aliases"] = aliases
        cfg["models"][mid] = entry

    # One exclusive swap group over every model. Without it llama-swap is free
    # to hold two models resident at once, which on a single-GPU box means the
    # second load fails or evicts into system RAM mid-request. Exclusive swap
    # makes the "one model at a time" assumption the rest of this stack is
    # built on actually true.
    live = {str(rec.get("id", "")).strip(): rec for rec in models
            if rec.get("enabled", True)
            and str(rec.get("id", "")).strip() in cfg["models"]}
    # A persistent model lives in its own group that nothing may unload;
    # everything else shares the exclusive swap group. llama-swap's own
    # rules: `persistent: true` means other groups cannot evict these,
    # `exclusive: true` on main means loading a main member evicts every
    # other NON-persistent group -- so the two coexist by design.
    pinned = [mid for mid, rec in live.items() if rec.get("persistent")]
    swappable = [mid for mid in cfg["models"] if mid not in pinned]
    if cfg["models"]:
        cfg["groups"] = {}
        if pinned:
            cfg["groups"]["warm"] = {
                "swap": False,
                "exclusive": False,
                "persistent": True,
                "members": pinned,
            }
        if swappable:
            cfg["groups"]["main"] = {
                "swap": True,
                "exclusive": True,
                "members": swappable,
            }
    # Warm on boot. llama-swap (v250+) loads these the moment it starts, so a
    # box dedicated to one model answers its first request after a reboot at
    # full speed instead of paying a 70 GB cold load into it. Persistent
    # models are always in the list -- resident-forever and cold-until-asked
    # is a contradiction. In the swap group only the LAST preload survives,
    # which is why check_preload_count() refuses more than one there.
    warm = [mid for mid, rec in live.items()
            if rec.get("preload") or rec.get("persistent")]
    if warm:
        cfg["hooks"] = {"on_startup": {"preload": warm}}
    header = (
        "# GENERATED BY THE OPEN-FLEET GATEWAY -- DO NOT EDIT BY HAND.\n"
        "# Source of truth is " + str(MODELS_JSON) + "\n"
        "# Regenerated at " + now() + "\n"
    )
    return header + yaml.safe_dump(cfg, sort_keys=False, width=10000)


def write_swap_config(models: list[dict]) -> str:
    text = render_swap_config(models)
    write_atomic(SWAP_CONFIG, text)
    return text


def systemctl(*args: str) -> tuple[int, str]:
    cmd = ["sudo", "-n", "/usr/bin/systemctl", *args]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return 1, str(exc)


# What a Scheduled Task leaves behind when it is ended. `schtasks /End` stops
# the task's own process -- which for these is a .cmd wrapper kept for its log
# redirection -- and the wrapper's grandchildren outlive it. A surviving
# llama-swap still owns port 8081, and the llama-server it is holding still
# owns the VRAM, so the restart that follows would fail on a bound port and
# the card would stay full. Killing by image name is safe for exactly these
# two: unlike python.exe (which the gpu-laptop-2 box learned the hard way), nothing
# else on a fleet member is called llama-swap.exe or llama-server.exe.
_WIN_TASK_CHILDREN: dict[str, tuple[str, ...]] = {
    "llama-swap": ("llama-swap.exe", "llama-server.exe"),
}

# The Macs have no service manager the gateway may drive: llama-swap is
# started by a cron `@reboot` line and kept alive by a */5 one. Matching the
# right process is fiddly enough to be worth naming.
#
# `pgrep -f run-llama-swap` -- what the cron line itself uses -- does NOT match
# llama-swap. run-llama-swap.sh ends in `exec llama-swap ...`, so the script's
# process becomes llama-swap and its command line no longer carries the
# script's name. What that pattern DOES match is the `/bin/sh -c` cron wrapper
# still sitting above it, and the keepalive's own shell, which is why killing
# by it orphans a live llama-swap that keeps holding :8081 while a relaunch
# fails on the bound port. Matching the binary invocation instead is exact:
# "bin/run-llama-swap.sh" does not contain "bin/llama-swap".
_MAC_SWAP_PATTERN = "bin/llama-swap -config"
# An orphaned llama-server: its parent llama-swap is gone, so nothing will
# ever route to it or unload it, but it keeps its Metal/unified memory until
# someone notices. mac-laptop-1 ran with one for three days (a cron keepalive kept
# spawning doomed duplicate llama-swaps, each of which began its start-up
# preload before failing to bind :8081; one preload child survived its
# parent) and every verify on the box answered HTTP 500 while 21 GB sat
# squatted. Once llama-swap is confirmed dead, ANY surviving llama-server is
# such an orphan -- a live llama-swap stops its children on the way out --
# so reaping by pattern here is exact, not a guess.
_MAC_SERVER_PATTERN = "bin/llama-server --model"
_MAC_RUNNER = str(HOME / "bin" / "run-llama-swap.sh")
_MAC_SWAP_LOG = STATE / "llama-swap.log"


def _darwin_service(action: str, unit: str) -> tuple[int, str]:
    """restart / start / stop on a cron-supervised Mac.

    Without this, saving the Models tab on mac-laptop-1 wrote a new models.json and a
    new llama-swap.yaml and then ran `sudo systemctl restart` on a machine
    that has no systemctl -- so the save reported success, the files on disk
    were right, and the llama-server that answered every request went on
    serving the context window it was launched with hours earlier. Measured on
    mac-laptop-1: models.json and llama-swap.yaml both said `-c 182768` while the
    running process said `-c 32768`."""
    if unit != "llama-swap":
        # cloudflared is not installed on the Macs (tailnet-only), so there is
        # nothing here to drive and nothing to pretend about.
        return 1, "only llama-swap is supervised on macOS"
    outs: list[str] = []
    if action in ("stop", "restart"):
        try:
            p = subprocess.run(["pkill", "-f", _MAC_SWAP_PATTERN],
                               capture_output=True, text=True, timeout=30)
            # rc 1 is "no process matched", which is the state we wanted.
            if p.returncode not in (0, 1):
                outs.append("pkill rc=" + str(p.returncode))
        except Exception as exc:  # noqa: BLE001
            return 1, "pkill: " + str(exc)
        # llama-swap stops its llama-server children on the way out; give the
        # listening socket time to actually close before rebinding it.
        for _ in range(20):
            time.sleep(0.25)
            try:
                if subprocess.run(["pgrep", "-f", _MAC_SWAP_PATTERN],
                                  capture_output=True,
                                  timeout=10).returncode != 0:
                    break
            except Exception:  # noqa: BLE001
                break
    if action in ("start", "restart"):
        try:
            if subprocess.run(["pgrep", "-f", _MAC_SWAP_PATTERN],
                              capture_output=True, timeout=10).returncode != 0:
                # No llama-swap is alive, so any llama-server still running is
                # an orphan holding memory the incoming engine needs (see
                # _MAC_SERVER_PATTERN). Windows does the same thing with
                # taskkill on both image names; Linux gets it from systemd's
                # cgroup. rc 1 is "nothing matched" -- the state we want.
                p = subprocess.run(["pkill", "-f", _MAC_SERVER_PATTERN],
                                   capture_output=True, text=True, timeout=30)
                if p.returncode == 0:
                    outs.append("reaped an orphaned llama-server")
        except Exception:  # noqa: BLE001
            pass  # reaping is best-effort; the restart itself still proceeds
        if not Path(_MAC_RUNNER).exists():
            return 1, "no runner at " + _MAC_RUNNER
        try:
            _MAC_SWAP_LOG.parent.mkdir(parents=True, exist_ok=True)
            fh = open(_MAC_SWAP_LOG, "ab")
        except OSError as exc:
            return 1, "log: " + str(exc)
        try:
            # start_new_session detaches it from this process group, so it
            # survives a gateway restart exactly as the cron-launched one
            # does -- and so restarting the gateway does not take the engine
            # down with it.
            subprocess.Popen(
                [_MAC_RUNNER], stdout=fh, stderr=fh,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        except Exception as exc:  # noqa: BLE001
            return 1, "spawn: " + str(exc)
        finally:
            fh.close()
        for _ in range(40):
            time.sleep(0.25)
            try:
                if subprocess.run(["pgrep", "-f", _MAC_SWAP_PATTERN],
                                  capture_output=True,
                                  timeout=10).returncode == 0:
                    outs.append("llama-swap restarted")
                    return 0, "\n".join(outs)
            except Exception:  # noqa: BLE001
                break
        return 1, "llama-swap did not come back -- see " + str(_MAC_SWAP_LOG)
    return 0, "\n".join(outs)


def service_control(action: str, unit: str) -> tuple[int, str]:
    """restart / start / stop one piece of the stack, in the local dialect.

    systemd units on the Linux boxes, SYSTEM Scheduled Tasks on the Windows
    ones. This exists because gpu-desktop-1 is the fleet's first Windows box running
    llama.cpp behind llama-swap: everywhere else Windows means Ollama, which
    manages its own models and never needs a config reload. Without it,
    saving a model on gpu-desktop-1's Models tab writes a llama-swap.yaml that nothing
    ever reads, which looks exactly like a save that worked."""
    if platform.system() == "Windows":
        outs: list[str] = []
        rc = 0

        def run(*cmd: str) -> int:
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            except Exception as exc:  # noqa: BLE001
                outs.append(" ".join(cmd) + ": " + str(exc))
                return 1
            text = (p.stdout + p.stderr).strip()
            if text:
                outs.append(text)
            return p.returncode

        if action in ("stop", "restart"):
            run("schtasks", "/End", "/TN", unit)
            for image in _WIN_TASK_CHILDREN.get(unit, ()):
                # /T for the tree, and a non-zero rc here just means "nothing
                # by that name was running", which is the desired state.
                run("taskkill", "/F", "/T", "/IM", image)
        if action in ("start", "restart"):
            if action == "restart":
                time.sleep(2)  # let the listening socket actually close
            rc = run("schtasks", "/Run", "/TN", unit)
        return rc, "\n".join(outs)
    if platform.system() == "Darwin":
        return _darwin_service(action, unit)
    return systemctl(action, unit + ".service")


# --------------------------------------------------------------------------
# agent profiles  (per-key behaviour applied at the proxy)
# --------------------------------------------------------------------------


def get_agent(key_id: int) -> dict | None:
    rows = db_query(
        "SELECT * FROM agents WHERE key_id=? AND enabled=1 "
        "AND archived_at IS NULL",
        (key_id,),
    )
    return rows[0] if rows else None


def resolve_model_id(requested: str) -> str:
    """Map an alias to its canonical model id; unknown names pass through."""
    req = requested.strip()
    for rec in load_models():
        if rec.get("id") == req:
            return req
        if req in (rec.get("aliases") or []):
            return str(rec["id"])
    return req


def apply_agent(agent: dict, payload: dict) -> dict:
    """
    Shape a chat/completions payload according to the key's agent profile:
      * force_model pins the model outright
      * allowed_models rejects anything off the list (aliases resolved first)
      * system_prompt + rules are injected as the FIRST system message, ahead
        of whatever the client sent, so client prompts refine rather than
        replace the operator's instructions
      * param_overrides fill in sampler defaults the client left unset --
        except max_tokens, which acts as a hard cap
    """
    if agent.get("force_model"):
        payload["model"] = agent["force_model"]

    allowed = []
    try:
        allowed = json.loads(agent.get("allowed_models") or "[]")
    except json.JSONDecodeError:
        pass
    if allowed:
        canon = resolve_model_id(str(payload.get("model", "")))
        if canon not in allowed:
            raise HTTPException(
                403,
                "model '" + str(payload.get("model", "")) + "' is not allowed for "
                "this key (allowed: " + ", ".join(allowed) + ")",
            )

    prompt = str(agent.get("system_prompt") or "").strip()
    rules = str(agent.get("rules") or "").strip()
    if rules:
        prompt = (prompt + "\n\n" if prompt else "") + "# Operating rules\n" + rules
    if prompt and isinstance(payload.get("messages"), list):
        payload["messages"] = [{"role": "system", "content": prompt}] + payload[
            "messages"
        ]

    overrides = {}
    try:
        overrides = json.loads(agent.get("param_overrides") or "{}")
    except json.JSONDecodeError:
        pass
    for k, v in overrides.items():
        if v is None or v == "":
            continue
        if k == "max_tokens":
            try:
                cap = int(v)
                cur = payload.get("max_tokens")
                payload["max_tokens"] = min(int(cur), cap) if cur else cap
            except (TypeError, ValueError):
                pass
        elif k not in payload:
            payload[k] = v
    return payload


# --------------------------------------------------------------------------
# host telemetry
# --------------------------------------------------------------------------


# The four vendor backends, the two caches and the PowerShell blob they drive
# now live in hw.py, which the installer imports too -- one implementation of
# "what is in this box", read the same way by the thing that provisions a
# machine and the thing that afterwards runs on it. See hw.py's header.
#
# What stays here is the part that is about THIS fleet rather than about this
# hardware: which spec-sheet row to fall back on, and which host it belongs
# to. The wrappers name their backends explicitly because the tests replace
# `amdgpu_stats` / `nvidia_stats` / `load_specs` on THIS module and expect the
# composite to pick the replacement up.


def _spec_vram_gb() -> float | None:
    """This box's spec-sheet VRAM, for the readings that have no measurement
    to offer -- a Mac with no Metal cap set, a Windows box whose counters are
    unavailable. Deliberately re-read per call: load_specs() is itself cached
    on the override file's mtime, and the tests replace it wholesale."""
    return (load_specs().get(HOST_NAME) or {}).get("vram_gb")


def nvidia_stats() -> list[dict]:
    # `_nvsmi` is read from THIS module's globals on every call, which is what
    # lets a test point it at a fake nvidia-smi (test_public.py's idle-card
    # cases do exactly that) and still have the reading follow.
    return hw.nvidia_stats(_nvsmi)


def darwin_gpu_stats() -> list[dict]:
    return hw.darwin_gpu_stats(_spec_vram_gb())


def gpu_cards() -> list[dict]:
    return hw.gpu_cards(_spec_vram_gb(), amd=amdgpu_stats, nv=nvidia_stats,
                        win=windows_gpu_stats, mac=darwin_gpu_stats)


_net_prev: dict[str, Any] = {"t": 0.0, "nics": {}}

# Wireless detection, cached: sysfs answers on Linux; macOS needs one
# networksetup call at startup; elsewhere the adapter name is all there is.
_wifi_cache: dict[str, bool] = {}
_WIFI_NAMES = ("wl", "wi-fi", "wifi", "wlan", "airport", "wireless")


def _is_wireless(name: str) -> bool:
    if name in _wifi_cache:
        return _wifi_cache[name]
    wifi = False
    low = name.lower()
    if any(k in low for k in _WIFI_NAMES):
        wifi = True
    elif Path("/sys/class/net", name, "wireless").exists():
        wifi = True
    elif platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["networksetup", "-listallhardwareports"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            port = ""
            for line in out.splitlines():
                if line.startswith("Hardware Port:"):
                    port = line.split(":", 1)[1].strip().lower()
                elif line.startswith("Device:") and line.split(":", 1)[1].strip() == name:
                    wifi = "wi-fi" in port or "airport" in port
                    break
        except Exception:  # noqa: BLE001
            pass
    _wifi_cache[name] = wifi
    return wifi


def net_stats() -> dict:
    """Per-NIC throughput, rated against the previous call.

    Virtual interfaces are skipped: docker bridges and veth pairs double-count
    container traffic that already crossed a physical NIC.
    """
    out: dict[str, Any] = {"interfaces": [], "rx_rate": 0.0, "tx_rate": 0.0}
    try:
        counters = psutil.net_io_counters(pernic=True)
    except Exception:  # noqa: BLE001
        return out
    try:
        if_stats = psutil.net_if_stats()
    except Exception:  # noqa: BLE001
        if_stats = {}

    t = time.time()
    dt = t - _net_prev["t"] if _net_prev["t"] else 0.0
    prev = _net_prev["nics"]
    # Virtual interfaces on any of the three platforms: docker/veth (Linux),
    # utun/anpi/awdl/llw/bridge (macOS -- utun is Tailscale/VPN, anpi is the
    # debug bridge), Tailscale/vEthernet/Loopback (Windows). Their traffic
    # already crossed a physical NIC, and letting one win the top slot makes
    # a WiFi laptop report a "wired" tailnet pipe.
    skip = ("lo", "docker", "br-", "veth", "virbr", "cni", "flannel",
            "utun", "anpi", "awdl", "llw", "bridge", "ap1", "gif", "stf",
            "tailscale", "vethernet", "loopback")

    for name, c in counters.items():
        if name.lower().startswith(skip):
            continue
        st = if_stats.get(name)
        if st is not None and not st.isup:
            continue
        rx = tx = 0.0
        p = prev.get(name)
        if p and dt > 0.5:
            rx = max(0, c.bytes_recv - p[0]) / dt
            tx = max(0, c.bytes_sent - p[1]) / dt
        out["interfaces"].append(
            {
                "name": name,
                "rx_rate": rx,
                "tx_rate": tx,
                "rx_total": c.bytes_recv,
                "tx_total": c.bytes_sent,
                "speed_mbit": (st.speed if st else 0) or 0,
                "wireless": _is_wireless(name),
            }
        )
        out["rx_rate"] += rx
        out["tx_rate"] += tx

    out["interfaces"].sort(key=lambda i: i["rx_rate"] + i["tx_rate"], reverse=True)
    _net_prev["nics"] = {n: (c.bytes_recv, c.bytes_sent) for n, c in counters.items()}
    _net_prev["t"] = t
    return out


_dl_prev: dict[str, float] = {"t": 0.0, "bytes": 0.0}


def models_volume() -> tuple[Any, str]:
    """`(disk_usage, "")` for the model volume, `(None, why)` when it is gone.

    shutil.disk_usage RAISES for a path whose drive has gone away rather than
    reporting zeros, and both callers sit on the status path the hub polls.
    Unguarded, that turns "somebody unplugged the external SSD" into "this box
    is offline" -- the machine disappears from the fleet page instead of
    telling anyone which disk it lost.
    """
    try:
        return shutil.disk_usage(str(MODELS_DIR)), ""
    except OSError as exc:
        return None, (getattr(exc, "strerror", "") or str(exc))


def storage_info() -> dict:
    """
    Storage for the volume the models actually live on.

    Deliberately scoped to MODELS_DIR rather than enumerating every mount: on
    these dual-boot boxes the NTFS Windows partitions are noise, and the only
    number that governs whether another GGUF fits is this one.

    Every key is present whether or not the volume is reachable -- callers
    (and the dashboard) read `available` to tell a box with an empty disk from
    a box with no disk, rather than having to guess from a pile of zeros.
    """
    du, unreachable = models_volume()
    if du is None:
        return {
            "mount": str(MODELS_DIR), "fstype": "", "device": "",
            "models_dir": str(MODELS_DIR),
            "available": False,
            "error": unreachable or "model volume unavailable",
            "total": 0, "used": 0, "free": 0,
            "gguf_bytes": 0, "gguf_count": 0,
            "partial_bytes": 0, "incoming_bytes": 0, "free_after_incoming": 0,
            "download_rate": 0.0, "download_eta_s": None,
        }
    mount, fstype, device = "", "", ""
    try:
        target = str(MODELS_DIR)
        for p in psutil.disk_partitions(all=False):
            if target.startswith(p.mountpoint) and len(p.mountpoint) > len(mount):
                mount, fstype, device = p.mountpoint, p.fstype, p.device
    except Exception:  # noqa: BLE001
        pass

    gguf_bytes, gguf_count = 0, 0
    try:
        for p in MODELS_DIR.rglob("*.gguf"):
            try:
                gguf_bytes += p.stat().st_size
                gguf_count += 1
            except OSError:
                continue
    except Exception:  # noqa: BLE001
        pass

    # Bytes still inbound. Without this the free-space number looks reassuring
    # right up until a 78GB download lands on top of it.
    incoming, partial = 0, 0
    try:
        for r in db_query(
            "SELECT bytes_total, bytes_done FROM jobs "
            "WHERE status IN ('queued','downloading')"
        ):
            done = int(r["bytes_done"] or 0)
            total = int(r["bytes_total"] or 0)
            partial += done
            if total:
                incoming += max(0, total - done)
    except Exception:  # noqa: BLE001
        pass

    # Aggregate download throughput, rated against the previous call, plus the
    # ETA that actually matters: when the queue finishes, not one file.
    t = time.time()
    dt = t - _dl_prev["t"] if _dl_prev["t"] else 0.0
    rate = 0.0
    if dt > 0.5 and _dl_prev["bytes"]:
        rate = max(0.0, (partial - _dl_prev["bytes"]) / dt)
    _dl_prev["t"], _dl_prev["bytes"] = t, float(partial)
    eta = int(incoming / rate) if rate > 1024 and incoming else None

    return {
        "mount": mount or str(MODELS_DIR),
        "fstype": fstype,
        "device": device,
        "models_dir": str(MODELS_DIR),
        "available": True,
        "total": du.total,
        "used": du.used,
        "free": du.free,
        "gguf_bytes": gguf_bytes,
        "gguf_count": gguf_count,
        "partial_bytes": partial,
        "incoming_bytes": incoming,
        "free_after_incoming": du.free - incoming,
        "download_rate": rate,
        "download_eta_s": eta,
    }


# _os_name(), _win_build() and os_info() moved to hw.py with the GPU backends
# -- `fleetctl detect` needs the distro and the arch before it can choose a
# package manager, and reading them twice was how the installer and the
# dashboard would eventually have disagreed about what a box is.


# The inference server this gateway fronts. Probed rather than read out of the
# config, so a box that quietly changed upstreams still reports the truth --
# and only a restart of that server can change the answer, hence the long TTL.
_engine_cache: dict[str, Any] = {"t": 0.0, "info": {}}
ENGINE_TTL = 300.0


def _llama_build() -> str:
    """llama.cpp build id, which only the binary itself knows."""
    if not LLAMA_SERVER:
        return ""
    try:
        p = subprocess.run(
            [LLAMA_SERVER, "--version"], capture_output=True, text=True, timeout=5
        )
    except Exception:  # noqa: BLE001 -- a missing binary just has no version
        return ""
    m = re.search(r"version:\s*(\S+)", p.stdout + p.stderr)
    if not m:
        return ""
    # Upstream numbers its builds, and those read as b4123; anything locally
    # built or forked reports a string of its own, which is left alone.
    return "b" + m.group(1) if m.group(1).isdigit() else m.group(1)


def engine_info() -> dict:
    if _engine_cache["info"] and time.time() - _engine_cache["t"] < ENGINE_TTL:
        return _engine_cache["info"]
    info: dict[str, Any] = {
        "kind": "none", "name": "", "version": "", "upstream": UPSTREAM, "up": False,
    }

    def probe(path: str) -> tuple[bool, Any]:
        try:
            r = httpx.get(UPSTREAM + path, timeout=2.0)
        except Exception:  # noqa: BLE001 -- upstream down is an answer too
            return False, None
        if r.status_code != 200:
            return False, None
        try:
            return True, r.json()
        except ValueError:
            return True, None

    # Both backends answer /api/version, so that cannot tell them apart -- it is
    # llama-swap's own version it reports there, not llama.cpp's. Split them the
    # way swap_running() does: llama-swap lists live llama-server processes on
    # /running, Ollama answers /api/ps.
    if probe("/running")[0]:
        mgr = (probe("/api/version")[1] or {}).get("version") or ""
        name = "llama.cpp (llama-swap)"
        if mgr:
            name = "llama.cpp (llama-swap " + str(mgr) + ")"
        info.update(kind="llama-swap", name=name, up=True, version=_llama_build())
    elif probe("/api/ps")[0]:
        info.update(
            kind="ollama", name="Ollama", up=True,
            version=str((probe("/api/version")[1] or {}).get("version") or ""),
        )
    elif LLAMA_SERVER and Path(LLAMA_SERVER).exists():
        # Installed but nothing answering: the box can serve, it just is not.
        # A hub that only routes for the fleet has no binary and stays "none",
        # which is the honest answer rather than naming an engine it lacks.
        info.update(
            kind="llama-swap", name="llama.cpp (llama-swap)", version=_llama_build(),
        )
    _engine_cache.update(t=time.time(), info=info)
    return info


# --------------------------------------------------------------------------
# services
#
# Every box in the fleet holds the same pieces up with a different tool:
# systemd units on the Linux ones, SYSTEM Scheduled Tasks on the Windows ones,
# @reboot cron lines on the Macs. Ask each platform in its own language. The
# old probe ran `sudo -n systemctl is-active` everywhere, which put a sudo
# error where the status should be on Windows (its sudo.exe has no -n) and on
# macOS (no NOPASSWD there) -- and on Linux, where it did work, it wrote a pair
# of PAM lines to the journal for every unit on every poll. Nothing below
# needs root.
# --------------------------------------------------------------------------

# How to recognise a piece with no service manager to ask: the process names it
# may run under, and the binary to look for when it is not running -- absent
# from disk means this box simply does not carry that piece.
_PROC_SIGNS: dict[str, tuple[tuple[str, ...], str]] = {
    "llama-swap": (("llama-swap",), "llama-swap"),
    "ollama": (("ollama",), "ollama"),
    "cloudflared": (("cloudflared",), "cloudflared"),
    "tailscaled": (("tailscaled", "tailscale-ipn"), "tailscaled"),
}


def _systemd_states(units: tuple[str, ...]) -> dict[str, str]:
    """One systemctl call for the whole list. `show` also separates a unit that
    is stopped from one that was never installed here, which `is-active` (which
    calls both "inactive") cannot."""
    p = subprocess.run(
        ["systemctl", "show", "--no-pager", "--property=Id",
         "--property=LoadState", "--property=ActiveState", "--",
         *[u + ".service" for u in units]],
        capture_output=True, text=True, timeout=15,
    )
    seen: dict[str, str] = {}
    for block in p.stdout.split("\n\n"):
        f = dict(ln.split("=", 1) for ln in block.splitlines() if "=" in ln)
        unit = f.get("Id", "")
        if not unit.endswith(".service"):
            continue
        seen[unit[: -len(".service")]] = (
            "not installed" if f.get("LoadState") == "not-found"
            else (f.get("ActiveState") or "unknown")
        )
    return {u: seen.get(u, "unknown") for u in units}


def _live_process_names() -> set[str]:
    """Lowercased, .exe stripped, so one table of names covers all three OSes."""
    names = set()
    for proc in psutil.process_iter(["name"]):
        n = (proc.info.get("name") or "").lower()
        names.add(n[:-4] if n.endswith(".exe") else n)
    return names


def _process_states(units: Iterable[str]) -> dict[str, str]:
    """The Windows and macOS answer: report what is actually running.

    Neither box has a manager worth querying -- a Scheduled Task is only
    readable elevated, and cron has no notion of "is it up" at all -- and a live
    process is what the dashboard is really asking about anyway.
    """
    live = _live_process_names()
    out: dict[str, str] = {}
    for unit in units:
        if unit == "llm-gateway":
            out[unit] = "active"  # this code is the gateway
            continue
        procs, binary = _PROC_SIGNS.get(unit, ((unit,), unit))
        if live & set(procs):
            out[unit] = "active"
        else:
            out[unit] = "inactive" if shutil.which(binary) else "not installed"
    return out


def service_states() -> dict[str, str]:
    """What the dashboard's Services card lists, and how each one is doing.

    Which pieces are worth naming depends on the box: only the engine it
    actually runs, never both, and nothing at all on a hub that just routes.
    """
    units: tuple[str, ...] = ("llm-gateway", "cloudflared", "tailscaled")
    kind = engine_info().get("kind")
    if kind in ("ollama", "llama-swap"):
        units = (kind, *units)
    if platform.system() == "Linux" and shutil.which("systemctl"):
        try:
            states = _systemd_states(units)
        except Exception:  # noqa: BLE001 -- fall through to the process probe
            states = {}
        if states:
            # A piece can be up here without systemd holding it up: cpu-box-1 has
            # ollama.service masked and an ollama serving anyway. Where the unit
            # says there is nothing running, believe the process table over it --
            # but leave a "failed" unit red, because that is worth seeing.
            quiet = [u for u, v in states.items()
                     if v in ("not installed", "inactive")]
            states.update(
                {u: v for u, v in _process_states(quiet).items() if v == "active"}
            )
            return states
    return _process_states(units)


_avail_cache: dict[str, Any] = {"t": 0.0, "state": None}
AVAILABILITY_TTL = 5.0


def availability() -> dict:
    """May the fleet use this box right now?

    Every box in the fleet so far has been a box: it exists to serve, and if
    it is powered on it is in the pool. A machine somebody games on is not
    that. Its owner's session has to win every time, and "win" cannot mean
    "wait for a 24 GB mixture-of-experts model to finish a reply first" -- by
    then the frames are already gone.

    So on those boxes an outside watchdog decides, and writes its verdict to
    LLMSTACK_AVAILABILITY_FILE as {"available": bool, "reason": str,
    "ts": <unix seconds>}. The gateway only reads it. What the verdict
    actually gates is one thing and one thing only: whether this host
    ADVERTISES its models to the hub (see api_served_models). It is not a
    kill switch --

      * /v1 keeps serving. A hub whose routing table is up to 30 s stale can
        still send a request here after the flag flips, and refusing it would
        surface as a 502 to whoever asked: the proxy's failover is
        connect-phase only, so a live host that says no is worse than one
        that quietly answers. Freeing the GPU is the watchdog's job, and it
        does it by unloading the model after a grace period longer than
        ROUTE_TTL -- by which point the hub has stopped choosing this box.
      * telemetry keeps flowing, so the dashboard shows a live card that says
        why the box is holding back rather than a dark one that looks broken.

    Missing, unparseable or stale files read as UNAVAILABLE. That is the
    uncomfortable direction -- a dead watchdog takes the box out of the pool
    until someone notices -- and it is the correct one: the other way round, a
    dead watchdog means the fleet quietly runs 24 GB of weights on top of
    whatever the owner is playing. The staleness is reported rather than
    hidden, in /health and on the Services card, so "why is gpu-desktop-1 serving
    nothing" has an answer that is one curl away.
    """
    if not AVAILABILITY_GATED:
        return {"gated": False, "available": True, "reason": "", "age_s": None}
    hit = _avail_cache["state"]
    if hit is not None and time.time() - _avail_cache["t"] < AVAILABILITY_TTL:
        return hit
    state = {"gated": True, "available": False, "reason": "", "age_s": None}
    try:
        raw = json.loads(AVAILABILITY_FILE.read_text())
        ts = float(raw.get("ts") or 0)
        age = max(0.0, time.time() - ts) if ts else None
        state["age_s"] = None if age is None else round(age, 1)
        if age is None:
            state["reason"] = "watchdog verdict has no timestamp"
        elif age > AVAILABILITY_MAX_AGE:
            state["reason"] = (
                f"watchdog verdict is {int(age)}s old "
                f"(limit {int(AVAILABILITY_MAX_AGE)}s)"
            )
        else:
            state["available"] = bool(raw.get("available"))
            state["reason"] = str(raw.get("reason") or "")
    except FileNotFoundError:
        state["reason"] = "no watchdog verdict at " + str(AVAILABILITY_FILE)
    except (OSError, ValueError, TypeError) as exc:
        state["reason"] = "unreadable watchdog verdict: " + str(exc)
    _avail_cache.update(t=time.time(), state=state)
    return state


def host_status() -> dict:
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()
    du, _ = models_volume()
    try:
        # AttributeError included: os.getloadavg does not exist on Windows.
        load = os.getloadavg()
    except (OSError, AttributeError):
        try:
            load = psutil.getloadavg()  # emulated on Windows
        except Exception:  # noqa: BLE001
            load = (0.0, 0.0, 0.0)
    temps: dict[str, Any] = {}
    try:
        for name, entries in (psutil.sensors_temperatures() or {}).items():
            for e in entries:
                if e.current:
                    temps[name + ":" + (e.label or "t")] = e.current
    except Exception:  # noqa: BLE001
        pass
    return {
        "time": now(),
        "uptime_s": int(time.time() - psutil.boot_time()),
        "os": os_info(),
        "engine": engine_info(),
        "availability": availability(),
        "cpu_percent": psutil.cpu_percent(interval=0.15),
        "cpu_per_core": psutil.cpu_percent(interval=None, percpu=True),
        "cpu_count": psutil.cpu_count(logical=True),
        "load": list(load),
        "mem": {
            "total": vm.total,
            "used": vm.total - vm.available,
            "available": vm.available,
            "percent": vm.percent,
        },
        "swap": {"total": sm.total, "used": sm.used, "percent": sm.percent},
        # Zeros rather than a missing key when the volume is gone: the
        # dashboard reads h.disk.used directly, and storage.available is
        # where it looks to find out that the disk is absent, not empty.
        "disk": ({"total": du.total, "used": du.used, "free": du.free}
                 if du else {"total": 0, "used": 0, "free": 0}),
        "storage": storage_info(),
        "net": net_stats(),
        "gpu": gpu_cards(),
        "temps": temps,
    }


# --------------------------------------------------------------------------
# huggingface model downloads
# --------------------------------------------------------------------------

_download_threads: dict[int, threading.Thread] = {}
_download_cancel: set[int] = set()
_download_start_lock = threading.Lock()   # one writer per .part, see start_download
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
SAFE_REPO = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
SAFE_FILE = re.compile(r"^[A-Za-z0-9._/-]+\.gguf$")


def job_update(job_id: int, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = now()
    cols = ", ".join(k + "=?" for k in fields)
    db_exec("UPDATE jobs SET " + cols + " WHERE id=?", (*fields.values(), job_id))


MAX_DOWNLOAD_ATTEMPTS = 8
# A download can fail without ever raising. Measured on a fleet box: one
# model crawled at 1 MB/s for half an hour while a fresh connection to the
# very same file pulled 9 MB/s -- a bad CDN edge, not a slow link. The read
# timeout never fires (bytes keep arriving) and the retry loop below only
# catches exceptions, so the job sat there reporting healthy progress toward
# a five-hour finish. Cancelling and re-queueing by hand fixed it in seconds,
# which is exactly the move worth automating.
DOWNLOAD_STALL_BPS = int(os.environ.get("LLMSTACK_DOWNLOAD_STALL_BPS", 512 * 1024))
DOWNLOAD_STALL_SECS = float(os.environ.get("LLMSTACK_DOWNLOAD_STALL_SECS", "90"))


class DownloadStalled(RuntimeError):
    """Throughput collapsed on an otherwise healthy connection."""


def _download_attempt(job_id: int, url: str, part: Path, dest: Path,
                      stall_abort: bool = False) -> bool:
    """One pass at the file. Returns True when the file is complete.

    `stall_abort` makes a crawling connection raise DownloadStalled instead of
    dragging on, so the caller can reconnect (resume is byte-exact, so this
    costs nothing but a round trip). Off for the later attempts: a genuinely
    slow link must be allowed to finish rather than being retried to death."""
    headers = {}
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        headers["Authorization"] = "Bearer " + token
    done = part.stat().st_size if part.exists() else 0
    if done:
        headers["Range"] = "bytes=" + str(done) + "-"

    dest.parent.mkdir(parents=True, exist_ok=True)
    # A finite read timeout matters: with timeout=None a half-open connection
    # parks the thread forever and the job looks alive while making no progress.
    timeout = httpx.Timeout(connect=30.0, read=120.0, write=60.0, pool=30.0)
    with httpx.stream(
        "GET", url, headers=headers, follow_redirects=True, timeout=timeout
    ) as r:
        if r.status_code == 416:
            # Range beyond EOF: we already have the whole thing.
            total = part.stat().st_size
            job_update(job_id, bytes_total=total, bytes_done=total)
            return True
        if r.status_code not in (200, 206):
            raise RuntimeError("HTTP " + str(r.status_code) + " from HuggingFace")
        if r.status_code == 200 and done:
            # Server ignored the Range header; start over rather than corrupt.
            done = 0
            part.unlink(missing_ok=True)
        total = done + int(r.headers.get("content-length") or 0)
        job_update(job_id, status="downloading", bytes_total=total, bytes_done=done)

        last = 0.0
        window_t, window_b = time.time(), done
        with open(part, "ab" if done else "wb") as fh:
            for chunk in r.iter_bytes(chunk_size=4 * 1024 * 1024):
                if job_id in _download_cancel:
                    job_update(job_id, status="cancelled", message="cancelled")
                    return False
                fh.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last > 1.0:
                    last = now
                    job_update(job_id, bytes_done=done)
                # Judged over a whole window, never on one slow chunk.
                if stall_abort and now - window_t >= DOWNLOAD_STALL_SECS:
                    rate = (done - window_b) / (now - window_t)
                    if rate < DOWNLOAD_STALL_BPS:
                        raise DownloadStalled(
                            "only %.0f KB/s over the last %.0fs"
                            % (rate / 1024, now - window_t))
                    window_t, window_b = now, done
    return True


def _download_worker(job_id: int, repo: str, filename: str, dest: Path) -> None:
    url = HF_ENDPOINT + "/" + repo + "/resolve/main/" + filename + "?download=true"
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
            if job_id in _download_cancel:
                job_update(job_id, status="cancelled", message="cancelled")
                return
            try:
                # Only the earlier attempts police throughput. If reconnecting
                # has not helped by then the link really is that slow, and
                # finishing slowly beats failing.
                if not _download_attempt(
                        job_id, url, part, dest,
                        stall_abort=attempt <= MAX_DOWNLOAD_ATTEMPTS // 2):
                    return  # cancelled
                size = part.stat().st_size
                part.replace(dest)
                # The Library's half of the LM Studio bargain: a GGUF pulled
                # here is linked straight into LM Studio's own layout, so it
                # is in both catalogues before the row even turns green. The
                # periodic sync would find it within the interval anyway --
                # this is what makes it feel immediate. It cannot fail the
                # download; see lmstudio_publish_one().
                published = lmstudio_publish_one(dest)
                job_update(
                    job_id,
                    status="done",
                    bytes_done=size,
                    bytes_total=size,
                    message="saved to " + str(dest)
                    + (" -- published to LM Studio (" + published + ")"
                       if published else ""),
                )
                return
            except Exception as exc:  # noqa: BLE001 -- retry anything transient
                if attempt >= MAX_DOWNLOAD_ATTEMPTS:
                    job_update(job_id, status="error", message=str(exc))
                    return
                backoff = min(30, 2**attempt)
                job_update(
                    job_id,
                    status="downloading",
                    message="attempt " + str(attempt) + " failed ("
                    + str(exc)[:120] + "); retrying in " + str(backoff) + "s",
                )
                time.sleep(backoff)
    finally:
        _download_cancel.discard(job_id)
        _download_threads.pop(job_id, None)


def resume_orphaned_downloads() -> None:
    """
    Restarting the gateway kills the download threads but leaves their rows
    reading 'downloading', so a job can look alive with nothing behind it.
    Pick those back up at boot. Each worker resumes from its .part file with an
    HTTP Range request, so no already-fetched byte is fetched twice.
    """
    for row in db_query(
        "SELECT * FROM jobs WHERE status IN ('queued','downloading') ORDER BY id"
    ):
        dest = Path(row["dest"] or "")
        if not row["dest"] or dest.exists():
            job_update(row["id"], status="done", message="already present on disk")
            continue
        t = threading.Thread(
            target=_download_worker,
            args=(int(row["id"]), row["repo"], row["filename"], dest),
            daemon=True,
        )
        _download_threads[int(row["id"])] = t
        t.start()


def start_download(repo: str, filename: str) -> dict:
    if not SAFE_REPO.match(repo):
        raise HTTPException(400, "bad repo id")
    if not SAFE_FILE.match(filename) or ".." in filename:
        raise HTTPException(400, "filename must be a .gguf path")
    dest = (MODELS_DIR / repo.replace("/", "__") / Path(filename).name).resolve()
    if MODELS_DIR.resolve() not in dest.parents:
        raise HTTPException(400, "destination escapes models dir")
    if dest.exists():
        raise HTTPException(409, "already downloaded: " + str(dest))
    # Queueing a file that is already in flight must not start a second
    # worker. Both would resume the same .part from whatever size they happened
    # to stat, then append at EOF -- so the two byte ranges interleave and the
    # file grows at the combined rate while being irrecoverable garbage. It
    # reads as healthy progress right up until the GGUF fails to load. Hand
    # back the job already running instead, which is what makes re-running
    # pull-models after a dropped Wi-Fi session safe.
    with _download_start_lock:
        existing = db_query(
            "SELECT id FROM jobs WHERE dest=? AND status IN ('queued','downloading') "
            "ORDER BY id LIMIT 1",
            (str(dest),),
        )
        if existing:
            job_id = int(existing[0]["id"])
            live = _download_threads.get(job_id)
            if live and live.is_alive():
                return {"job_id": job_id, "dest": str(dest), "already_running": True}
            # The row outlived its worker. Re-drive that row rather than
            # inserting a second one, so the .part keeps exactly one writer.
            _download_cancel.discard(job_id)
        else:
            job_id = db_exec(
                "INSERT INTO jobs(created_at,updated_at,repo,filename,status,dest) "
                "VALUES (?,?,?,?,?,?)",
                (now(), now(), repo, filename, "queued", str(dest)),
            )
        t = threading.Thread(
            target=_download_worker, args=(job_id, repo, filename, dest), daemon=True
        )
        _download_threads[job_id] = t
        t.start()
    return {"job_id": job_id, "dest": str(dest)}


def list_local_models() -> list[dict]:
    out = []
    for p in sorted(MODELS_DIR.rglob("*.gguf")):
        try:
            st = p.stat()
        except OSError:
            continue
        out.append(
            {
                "path": str(p),
                "name": p.name,
                "size": st.st_size,
                "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(
                    timespec="seconds"
                ),
            }
        )
    return out


# --------------------------------------------------------------------------
# LM Studio  <->  fleet model sync
# --------------------------------------------------------------------------
#
# Two catalogues of the same GGUFs sat side by side on every Mac and Windows
# box and never saw each other. LM Studio indexes
# <downloadsFolder>/<publisher>/<repo>/<file>.gguf; the fleet indexes
# MODELS_DIR/<org>__<repo>/<file>.gguf. A model pulled through one was
# invisible in the other, and pulling it through both paid for the weights
# twice -- 20.22 GiB of gpu-laptop-2's E: drive was a single Qwopus GGUF stored
# once under each layout.
#
# This is deliberately NOT a copier. Three rules:
#
#   1. IMPORT. A GGUF LM Studio already holds is registered in models.json at
#      its LM Studio path. Nothing moves; the fleet just learns where it is.
#      This is the only direction that works on every filesystem, and that
#      matters: apu-tablet-2's shared D: is exFAT, where Windows refuses hard
#      links and symlinks alike.
#   2. PUBLISH. A GGUF the fleet holds is given to LM Studio by LINKING it
#      into LM Studio's layout -- os.link when the two paths share a device,
#      os.symlink when they do not. LM Studio indexes either within seconds
#      (measured on the M4 Air: a hard link appeared in `lms ls` in under 8s),
#      so this costs zero bytes. Where neither link works the file is reported
#      as unpublishable rather than quietly duplicated.
#   3. RECLAIM. Where the same weights already exist as two inodes on one
#      device, the FLEET copy is collapsed onto the LM Studio one and the
#      duplicate bytes go back to the filesystem. The LM Studio file is never
#      the one replaced, and the pair is matched on size plus a head and tail
#      digest before anything is touched.
#
# The LM Studio daemon is not involved anywhere: `lms` needs the desktop app
# running and times out on a headless box, so the on-disk layout is the whole
# contract. LM Studio's own index (~/.lmstudio/.internal/model-data.json) is
# read at most for provenance and never written.

LMSTUDIO_HOME_ENV = os.environ.get("LLMSTACK_LMSTUDIO_HOME", "").strip()

DEFAULT_LMSTUDIO_SETTINGS: dict[str, Any] = {
    "enabled": True,
    # A poll, not a filesystem watcher: one stat per GGUF every few minutes is
    # nothing, and watchers are three different APIs across three platforms
    # with a shared habit of missing events on network and external volumes.
    "interval_secs": 300,
    "import_to_fleet": True,
    "publish_to_lmstudio": True,
    "reclaim_duplicates": True,
    # The last rung of the ladder in publish(). Off by default: the entire
    # point of this subsystem is that the weights exist once.
    "allow_copy": False,
    # An imported record is enabled, because "it showed up in LM Studio and
    # now the fleet can serve it" is the whole request. Flip this off to have
    # imports land disabled and be switched on by hand from the Models tab.
    "import_enabled": True,
    # llama-swap only re-reads its config when it restarts, and a restart
    # drops whatever is resident. Never done while this box has inference or
    # a benchmark in flight; see _lmstudio_restart_swap().
    "restart_swap": True,
    # Paths this box has imported at some point, and paths the operator has
    # since removed from the Models tab. Without the second list every deleted
    # record came back on the next tick, which makes "delete" look broken and
    # is the one thing an automatic writer must never do to a manual edit.
    "imported_paths": [],
    "dismissed_paths": [],
    # Pairs that matched on size and both ends but differed in the middle.
    # Remembered so a background pass does not read two 20 GiB files in full
    # every five minutes, forever, to reach the same answer.
    "mismatched_pairs": [],
}

_LMSTUDIO_SETTING_BOUNDS: dict[str, tuple[float, float]] = {
    "interval_secs": (60, 86400),
}

# <base>-00001-of-00004.gguf -- llama.cpp is handed the FIRST shard and finds
# the rest itself, and LM Studio's own index keys only the first as well.
_GGUF_SHARD = re.compile(r"^(?P<base>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$", re.I)
# Uploader-controlled and inconsistent: `mmproj-gemma-4-31B-it-QAT-BF16.gguf`
# on one repo, a bare `mmproj-F32.gguf` on the next. The prefix is the only
# thing they all share, so a projector is paired with its model by living in
# the same directory, never by name.
_GGUF_MMPROJ = re.compile(r"^mmproj[-_.]", re.I)
# What a half-finished download looks like in either store.
_GGUF_PARTIAL = (".part", ".tmp", ".download", ".incomplete", ".crdownload")

_lmstudio_root_cache: dict[str, Any] = {"t": 0.0, "root": None, "home": None}
_lmstudio_lock = threading.Lock()      # one sync pass at a time, per process
_lmstudio_last: dict[str, Any] = {}


def _lmstudio_settings_raw() -> dict:
    rows = db_query("SELECT value FROM settings WHERE key='lmstudio'")
    if not rows:
        return {}
    try:
        data = json.loads(rows[0]["value"])
        return data if isinstance(data, dict) else {}
    except (TypeError, ValueError):
        return {}


def get_lmstudio_settings() -> dict:
    out = dict(DEFAULT_LMSTUDIO_SETTINGS)
    out.update({k: v for k, v in _lmstudio_settings_raw().items() if k in out})
    for key, (lo, hi) in _LMSTUDIO_SETTING_BOUNDS.items():
        try:
            out[key] = max(int(lo), min(int(hi), int(out[key])))
        except (TypeError, ValueError):
            out[key] = DEFAULT_LMSTUDIO_SETTINGS[key]
    for key, default in DEFAULT_LMSTUDIO_SETTINGS.items():
        if isinstance(default, bool):
            out[key] = bool(out[key])
        elif isinstance(default, list) and not isinstance(out[key], list):
            out[key] = list(default)
    return out


def set_lmstudio_settings(updates: dict) -> dict:
    if not isinstance(updates, dict):
        return get_lmstudio_settings()
    merged = _lmstudio_settings_raw()
    merged.update(
        {k: v for k, v in updates.items() if k in DEFAULT_LMSTUDIO_SETTINGS}
    )
    db_exec(
        "INSERT INTO settings(key,value) VALUES ('lmstudio',?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (json.dumps(merged),),
    )
    return get_lmstudio_settings()


# ---- finding LM Studio ----------------------------------------------------


def _lmstudio_homes() -> list[Path]:
    """Every plausible ~/.lmstudio on this box, best first.

    The Windows boxes are why this is a list rather than Path.home(): the
    gateway there is a SYSTEM Scheduled Task, so Path.home() is
    C:\\Windows\\system32\\config\\systemprofile and LM Studio's real
    directory is under C:\\Users\\<somebody>. SYSTEM inherits Full Control on
    those profiles (checked with icacls on gpu-laptop-2 and gpu-desktop-2), so reading
    and linking into them works -- it just has to be reached by its literal
    path instead of an expansion."""
    homes: list[Path] = []

    def add(p: Path) -> None:
        if p not in homes:
            homes.append(p)

    if LMSTUDIO_HOME_ENV:
        add(Path(LMSTUDIO_HOME_ENV))
    try:
        add(Path.home() / ".lmstudio")
    except (OSError, RuntimeError):
        pass
    roots = [Path("C:/Users")] if platform.system() == "Windows" else [Path("/Users")]
    for root in roots:
        try:
            profiles = sorted(root.iterdir())
        except OSError:
            continue
        for prof in profiles:
            if prof.name.lower() in ("public", "default", "default user",
                                     "all users", "shared", ".localized"):
                continue
            cand = prof / ".lmstudio"
            try:
                if cand.is_dir():
                    add(cand)
            except OSError:
                continue
    return homes


def _lmstudio_downloads_folder(home: Path) -> Path | None:
    """LM Studio's models root for one profile.

    settings.json owns this and it is genuinely moved in practice -- three of
    the fleet's boxes point it at a second drive (D:\\AI\\models,
    E:\\LMStudio\\Models, E:\\AI\\lmStudio\\models). Assuming
    ~/.lmstudio/models would have found an empty directory on all three."""
    folder: Path | None = None
    try:
        raw = json.loads((home / "settings.json").read_text(encoding="utf-8"))
        val = str((raw or {}).get("downloadsFolder") or "").strip()
        if val:
            folder = Path(val)
    except (OSError, ValueError):
        folder = None
    if folder is None:
        folder = home / "models"
    try:
        return folder if folder.is_dir() else None
    except OSError:
        return None


def lmstudio_root(max_age: float = 60.0) -> Path | None:
    """The LM Studio models directory on this box, or None if LM Studio is not
    installed here. Cached briefly -- the sync loop, the download hook and the
    dashboard all ask, and the answer changes about once a year."""
    hit = _lmstudio_root_cache
    if hit["root"] is not None and time.time() - hit["t"] < max_age:
        root = hit["root"]
        try:
            if root.is_dir():
                return root
        except OSError:
            pass
    for home in _lmstudio_homes():
        folder = _lmstudio_downloads_folder(home)
        if folder is not None:
            _lmstudio_root_cache.update(t=time.time(), root=folder, home=home)
            return folder
    _lmstudio_root_cache.update(t=time.time(), root=None, home=None)
    return None


# ---- reading a store ------------------------------------------------------


def _gguf_kind(name: str) -> tuple[str, str]:
    """('main'|'mmproj'|'shard', base-name-without-the-shard-suffix)."""
    if _GGUF_MMPROJ.match(name):
        return "mmproj", name
    m = _GGUF_SHARD.match(name)
    if m:
        # Only the first shard is a model; the rest are its tail, and handing
        # llama-server shard 3 of 4 is a guaranteed load failure.
        return ("main" if int(m.group("idx")) == 1 else "shard"), m.group("base")
    return "main", name[:-5] if name.lower().endswith(".gguf") else name


def _scan_gguf_tree(root: Path, limit: int = 4000) -> list[dict]:
    """Every servable GGUF under `root`, with its projector if it has one.

    Groups by directory because that is the only thing a model and its mmproj
    reliably share. Partial downloads are skipped in both stores -- ours end
    in .part, LM Studio's in .tmp/.download -- so a sync pass that lands
    mid-download registers nothing and simply finds the file on the next one."""
    by_dir: dict[Path, dict[str, Any]] = {}
    seen = 0
    try:
        walk = sorted(root.rglob("*.gguf"))
    except OSError:
        return []
    for path in walk:
        seen += 1
        if seen > limit:
            log.warning("lmstudio: stopped scanning %s after %d files", root, limit)
            break
        name = path.name
        if any(name.lower().endswith(sfx) for sfx in _GGUF_PARTIAL):
            continue
        try:
            if path.is_dir():
                continue
            st = path.stat()          # follows links on purpose: a published
        except OSError:               # link must read as the file it points at
            continue
        if st.st_size <= 0:
            continue
        # A .part sitting beside a finished file is fine; a .part OF this file
        # means the download is still running and the name is not yet truth.
        try:
            if path.with_suffix(path.suffix + ".part").exists():
                continue
        except OSError:
            pass
        kind, base = _gguf_kind(name)
        if kind == "shard":
            continue
        slot = by_dir.setdefault(path.parent, {"models": [], "mmproj": None})
        if kind == "mmproj":
            # Smallest projector wins when a repo ships several precisions --
            # they are interchangeable and the small one costs least to load.
            cur = slot["mmproj"]
            if cur is None or st.st_size < cur[1]:
                slot["mmproj"] = (path, st.st_size)
            continue
        slot["models"].append((path, st, base))

    out: list[dict] = []
    for directory, slot in by_dir.items():
        mmproj = slot["mmproj"][0] if slot["mmproj"] else None
        try:
            rel = directory.relative_to(root).parts
        except ValueError:
            rel = ()
        publisher = rel[0] if len(rel) >= 2 else ""
        repo = rel[-1] if rel else directory.name
        for path, st, base in slot["models"]:
            out.append({
                "path": str(path),
                "dir": str(directory),
                "name": path.name,
                "base": base,
                "publisher": publisher,
                "repo": repo,
                "size": int(st.st_size),
                "mtime": int(st.st_mtime),
                "dev": int(st.st_dev),
                "ino": int(st.st_ino),
                "nlink": int(getattr(st, "st_nlink", 1)),
                "symlink": path.is_symlink(),
                "mmproj": str(mmproj) if mmproj else "",
            })
    out.sort(key=lambda e: e["path"])
    return out


def _lmstudio_mlx_models(root: Path) -> list[dict]:
    """MLX repos, listed so the dashboard can say why they are not imported.

    llama.cpp cannot load safetensors, so an MLX repo is never a fleet model.
    It is still worth naming: three of the four models on the M4 Air are MLX,
    and 'nothing happened' is a worse answer than 'that one is MLX'."""
    out: list[dict] = []
    try:
        publishers = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return out
    for pub in publishers:
        try:
            repos = sorted(r for r in pub.iterdir() if r.is_dir())
        except OSError:
            continue
        for repo in repos:
            try:
                files = list(repo.iterdir())
            except OSError:
                continue
            if any(f.suffix == ".gguf" for f in files):
                continue
            tensors = [f for f in files if f.suffix == ".safetensors"]
            if not tensors:
                continue
            size = 0
            for f in tensors:
                try:
                    size += f.stat().st_size
                except OSError:
                    pass
            out.append({
                "id": pub.name + "/" + repo.name,
                "path": str(repo),
                "size": size,
                "format": "mlx",
            })
    return out


# ---- identity -------------------------------------------------------------

_SAMPLE_BYTES = 1 << 20        # 1 MiB from each end


def _edge_digest(path: Path, size: int) -> str | None:
    """A cheap content fingerprint: size, first MiB, last MiB.

    Enough to decide "these two paths are the same weights" without reading
    20 GB twice. A GGUF's header (tensor names, geometry, the whole vocab) is
    at the front and the last tensor's payload is at the back, so two distinct
    models colliding on all three is not a case worth engineering against --
    and every caller that acts destructively also requires the two files to
    already be on the same device."""
    try:
        h = hashlib.sha256()
        h.update(str(size).encode())
        with open(path, "rb") as fh:
            h.update(fh.read(_SAMPLE_BYTES))
            if size > _SAMPLE_BYTES * 2:
                fh.seek(-_SAMPLE_BYTES, os.SEEK_END)
                h.update(fh.read(_SAMPLE_BYTES))
        return h.hexdigest()
    except OSError:
        return None


def _same_file(a: dict, b: dict) -> bool:
    """Already one inode -- nothing to reclaim, nothing to publish."""
    return a["dev"] == b["dev"] and a["ino"] == b["ino"]


def _mismatch_key(a: str, b: str) -> str:
    return "\u0000".join(sorted((str(a), str(b))))


def _lmstudio_remember_mismatch(a: str, b: str) -> None:
    known = set(get_lmstudio_settings().get("mismatched_pairs") or [])
    known.add(_mismatch_key(a, b))
    # Bounded: this should be empty on every real box, and an unbounded list
    # in a settings row is a slow leak rather than a feature.
    set_lmstudio_settings({"mismatched_pairs": sorted(known)[-200:]})


def _full_digest(path: Path) -> str | None:
    """Every byte. Slow on purpose.

    The edge digest is what a scan can afford; it is NOT what should be
    allowed to authorise discarding a directory entry. This runs exactly once,
    immediately before the single destructive step in this subsystem, against
    both files, freshly -- never cached, never carried over from the pass that
    found the candidate. A 20 GiB pair costs tens of seconds off local NVMe;
    reclaim is a once-per-duplicate event, so that is bought once and never
    again."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(8 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


# ---- naming ---------------------------------------------------------------

_ID_STRIP = re.compile(r"[-_.]?(gguf|ggml)$", re.I)
_ID_CLEAN = re.compile(r"[^A-Za-z0-9._:-]+")


def _public_ids() -> set[str]:
    """Fleet Pass's public catalogue ids.

    These are NOT ordinary model ids and they are why this function exists:
    resolve_targets() looks a requested name up as a public id FIRST, and a
    public row resolves to whatever fleet ids the row claims. So a local model
    that takes a public id for itself becomes unreachable under its own name
    -- the request goes to the row's fleet ids instead, and if none of them is
    served here the answer is "no host in the fleet serves that model" about a
    model this box is holding. Found the hard way on mac-laptop-2, whose LM Studio
    copy of Qwen3.8-9B-Distill derives exactly the public id
    `qwen3.8-9b-distill`, already claimed by mac-desktop's Ollama tag.

    A collision with a row's fleet_ids is the opposite case and is welcome: it
    means this box really does serve that public model, and requests for the
    public id will now route here too."""
    try:
        rows = public_catalogue()["by_public"]
    except Exception:  # noqa: BLE001 -- an empty catalogue is not an error
        return set()
    # A row that claims its own public id among its fleet ids is saying the
    # fleet serves the model under that very name -- so a local model taking
    # it is not a collision, it is the point. Only the others are off limits.
    return {pid for pid, row in rows.items() if pid not in _row_fleet_ids(row)}


def _fleet_id_for(entry: dict, taken: set[str]) -> str:
    """A models.json id for an LM Studio repo.

    The repo name is the honest source -- `Qwen3.8-9B-Distill-GGUF` becomes
    `qwen3.8-9b-distill` -- because that is the name the owner recognises in
    LM Studio. It has to satisfy SAFE_ID, be unique across ids AND aliases
    (llama-swap refuses a whole config over one duplicate), and stay clear of
    the public catalogue's namespace (see _public_ids)."""
    stem = str(entry.get("repo") or Path(entry["path"]).parent.name)
    pub_hint = str(entry.get("publisher") or "")
    # A directory in the FLEET's own flat layout carries both halves in one
    # name: `ggml-org__Qwen3.8-27B-GGUF`. Split it, so the id reads like the
    # model rather than like the folder. This shows up on apu-tablet-2, where the
    # two stores share a directory and the LM Studio scan therefore walks the
    # fleet's flat repos as well.
    if "__" in stem:
        org, _, rest = stem.partition("__")
        if org.strip() and rest.strip():
            pub_hint = pub_hint or org.strip()
            stem = rest.strip()
    base = _ID_CLEAN.sub("-", _ID_STRIP.sub("", stem)).strip("-._:").lower()
    if not base:
        base = "lmstudio-model"
    base = base[:56]
    if not SAFE_ID.match(base):
        base = "lmstudio-model"
    blocked = taken | _public_ids()
    if base not in blocked:
        return base
    # Qualify with the publisher before falling back to a counter: on a repo
    # that collides, `empero-ai-qwen3.8-9b-distill` says which one it is,
    # where `qwen3.8-9b-distill-2` says only that there was an argument.
    pub = _ID_CLEAN.sub("-", pub_hint).strip("-._:").lower()
    if pub:
        qualified = (pub + "-" + base)[:64]
        if SAFE_ID.match(qualified) and qualified not in blocked:
            return qualified
    candidate = base
    n = 2
    while candidate in blocked:
        candidate = (base[:56] + "-" + str(n))[:64]
        n += 1
    return candidate


def _lms_target_for(fleet_entry: dict) -> tuple[str, str]:
    """Where a fleet GGUF belongs inside LM Studio: (publisher, repo).

    The fleet writes MODELS_DIR/<org>__<repo>/<file>.gguf, so the double
    underscore is the join that has to be undone. A directory with no `__`
    (someone's hand-made folder) is published under a `fleet` publisher so it
    still lands in the two-deep layout LM Studio indexes."""
    parent = Path(fleet_entry["path"]).parent.name
    if "__" in parent:
        org, _, repo = parent.partition("__")
        org, repo = org.strip(), repo.strip()
        if org and repo:
            return org, repo
    return "fleet", parent or "models"


# ---- linking --------------------------------------------------------------


def _link_into(src: Path, dst: Path, allow_copy: bool) -> tuple[str, str]:
    """Make `dst` be `src` without paying for the bytes twice.

    The ladder, in order of how much it costs and how well it survives:
      hardlink  one inode, two names, indistinguishable from a real file to
                anything that opens it. Same device only.
      symlink   crosses devices. LM Studio follows one (measured), and so
                does llama-server. Breaks if the target moves.
      copy      only when explicitly allowed, because it is the very thing
                this subsystem exists to stop.

    Built under a temporary name in the destination directory and moved into
    place, so a crash mid-way never leaves a half-made entry where LM Studio
    will index it."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".llmstack-" + str(os.getpid()) + ".tmp")
    try:
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
    except OSError:
        pass
    last = "no link method available"
    for how in ("hardlink", "symlink", "copy"):
        if how == "copy" and not allow_copy:
            break
        try:
            if how == "hardlink":
                os.link(src, tmp)
            elif how == "symlink":
                os.symlink(src, tmp)
            else:
                shutil.copy2(src, tmp)
            os.replace(tmp, dst)
            return how, ""
        except OSError as exc:
            last = str(exc)
            try:
                if tmp.exists() or tmp.is_symlink():
                    tmp.unlink()
            except OSError:
                pass
            continue
    return "", last


def _upstream_listening(timeout: float = 2.0) -> bool:
    """Is anything actually behind UPSTREAM right now?

    Registering a model in models.json is a claim that this box can serve it,
    and the hub believes it: the id goes into the routing table and requests
    start arriving. gpu-desktop-2 is the case that proves the point -- it is a
    telemetry/staging peer with no llama-swap installed at all, so a first
    sync there registered 16 LM Studio models the box had no way to load, and
    every one of them became a route to a closed port. A model found on a box
    with nothing listening is imported DISABLED instead, and switched on by
    hand once an engine is actually there."""
    try:
        parsed = urlparse(UPSTREAM)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except Exception:  # noqa: BLE001
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _lmstudio_busy() -> bool:
    """True while this box is decoding or benchmarking.

    Nothing in a sync pass touches a file llama-server has open -- a reclaim
    swaps a directory entry, and an open descriptor keeps the old inode alive
    until it is closed -- but the llama-swap RESTART that publishes a registry
    change does take the resident model down with it. So the restart waits;
    the next pass a few minutes later will do it."""
    return inflight_work() > 0


def _lmstudio_restart_swap() -> str:
    if UPSTREAM_MODELS:
        return "ollama-backed: nothing to restart"
    if platform.system() not in ("Linux", "Windows"):
        # The Macs supervise llama-swap from cron, which cannot restart
        # anything -- the same 501 /admin/api/service/* returns. Saying so is
        # better than letting service_control() fall through to a `sudo
        # systemctl` that reports "command not found" as if something broke.
        return ("macOS: llama-swap is cron-supervised -- the new models are in"
                " models.json and llama-swap.yaml, and are served after"
                " `pkill -f run-llama-swap` on the box")
    if _lmstudio_busy():
        return "deferred: work in flight"
    code, out = service_control("restart", "llama-swap")
    return "restarted" if code == 0 else "restart rc=" + str(code) + " " + out[:200]


# ---- the plan -------------------------------------------------------------


def lmstudio_plan() -> dict:
    """What a sync pass would do, without doing any of it.

    Pure enough to be the dashboard's read model AND the thing apply()
    executes, so the panel never shows one thing and the button does another."""
    st = get_lmstudio_settings()
    root = lmstudio_root()
    out: dict[str, Any] = {
        "host": HOST_NAME,
        "generated_at": now(),
        "available": root is not None,
        "root": str(root) if root else "",
        "home": str(_lmstudio_root_cache.get("home") or ""),
        "models_dir": str(MODELS_DIR),
        "upstream_models": UPSTREAM_MODELS,
        "settings": st,
        "actions": [],
        "mlx": [],
        "lmstudio_count": 0,
        "fleet_count": 0,
        "reclaimable": 0,
    }
    if root is None:
        out["reason"] = "LM Studio is not installed on this box"
        return out

    lms = _scan_gguf_tree(root)
    fleet = _scan_gguf_tree(MODELS_DIR)
    out["lmstudio_count"] = len(lms)
    out["fleet_count"] = len(fleet)
    out["mlx"] = _lmstudio_mlx_models(root)

    records = load_models()
    registered = set()
    for rec in records:
        p = str(rec.get("path") or "")
        if not p:
            continue
        registered.add(p)
        try:
            registered.add(str(Path(p).resolve()))
        except OSError:
            pass

    # A record this box imported that is no longer in models.json was deleted
    # by hand, and re-adding it on the next tick would make the Models tab's
    # delete button look broken. Remember the removal instead; the panel's
    # "forget dismissals" button is how it is taken back.
    dismissed = set(st.get("dismissed_paths") or [])
    imported_before = set(st.get("imported_paths") or [])
    unregistered = {p for p in imported_before if p not in registered}
    # ...but only when the weights are still there. A path whose file has ALSO
    # gone was not a rejected model, it was a deleted download, and it is
    # forgotten outright: if the same model is fetched again later it should
    # be imported again rather than arriving pre-dismissed.
    gone = {p for p in unregistered if Path(p).exists()}
    vanished = unregistered - gone
    if gone or vanished:
        dismissed |= gone
        set_lmstudio_settings({
            "dismissed_paths": sorted(dismissed),
            "imported_paths": sorted(imported_before - unregistered),
        })
        st = get_lmstudio_settings()
        out["settings"] = st
    out["dismissed"] = sorted(dismissed)
    mismatched = set(st.get("mismatched_pairs") or [])
    taken = {str(r.get("id") or "") for r in records}
    for rec in records:
        for a in rec.get("aliases") or []:
            if isinstance(a, str):
                taken.add(a)

    fleet_inodes = {(f["dev"], f["ino"]) for f in fleet}
    actions: list[dict] = []

    # --- IMPORT: LM Studio -> models.json, in place -------------------------
    #
    # Skipped wholesale on an Ollama-backed box: models.json is not the
    # catalogue there, so writing to it would change nothing anyone can see.
    for e in lms:
        already = e["path"] in registered
        if not already:
            try:
                already = str(Path(e["path"]).resolve()) in registered
            except OSError:
                pass
        if already:
            continue
        if e["path"] in dismissed:
            actions.append({
                "kind": "import", "status": "dismissed", "path": e["path"],
                "name": e["name"], "size": e["size"], "id": "",
                "reason": "removed from the Models tab by hand -- not re-added",
            })
            continue
        if UPSTREAM_MODELS:
            if (e["dev"], e["ino"]) in fleet_inodes:
                # This is a link the publish direction made from the fleet's
                # own store. Offering to hand it back to Ollama would be
                # offering the box a copy of something it already has.
                continue
            actions.append({
                "kind": "import", "status": "blocked", "path": e["path"],
                "name": e["name"], "size": e["size"], "id": "",
                "reason": "this box serves Ollama's catalogue, not models.json"
                          " -- importing would need `ollama create`, which"
                          " copies the weights",
            })
            continue
        if not st["import_to_fleet"]:
            actions.append({
                "kind": "import", "status": "off", "path": e["path"],
                "name": e["name"], "size": e["size"], "id": "",
                "reason": "import_to_fleet is off",
            })
            continue
        mid = _fleet_id_for(e, taken)
        taken.add(mid)
        actions.append({
            "kind": "import", "status": "todo", "path": e["path"],
            "name": e["name"], "size": e["size"], "id": mid,
            "mmproj": e["mmproj"],
            "reason": "in LM Studio, not in the fleet catalogue",
        })

    # --- PUBLISH: fleet -> LM Studio's layout, by link ----------------------
    #
    # Keyed on (size, edge digest) so a fleet file that IS the LM Studio file
    # under another name is recognised instead of being published back on top
    # of itself.
    lms_by_key: dict[tuple[int, str], dict] = {}
    lms_by_ino: set[tuple[int, int]] = {(e["dev"], e["ino"]) for e in lms}
    lms_sizes = {e["size"] for e in lms}

    try:
        root_res = root.resolve()
    except OSError:
        root_res = root
    for e in fleet:
        if (e["dev"], e["ino"]) in lms_by_ino:
            continue                                  # already one file
        # apu-tablet-2 points BOTH stores at D:\AI\models, so the fleet scan
        # walks LM Studio's own nested repos too. Publishing one of those
        # would build a second name for it under a bogus `fleet` publisher --
        # LM Studio's directory is where it already is.
        try:
            if root_res in Path(e["path"]).resolve().parents:
                continue
        except OSError:
            pass
        publisher, repo = _lms_target_for(e)
        dst = root / publisher / repo / e["name"]
        try:
            if dst.exists() and os.stat(dst).st_ino == e["ino"] \
                    and os.stat(dst).st_dev == e["dev"]:
                continue
        except OSError:
            pass

        # Is it the same weights as something LM Studio already holds?
        twin = None
        if e["size"] in lms_sizes:
            key = (e["size"], _edge_digest(Path(e["path"]), e["size"]) or "")
            if key[1]:
                if not lms_by_key:
                    for c in lms:
                        if c["size"] in {f["size"] for f in fleet}:
                            d = _edge_digest(Path(c["path"]), c["size"])
                            if d:
                                lms_by_key[(c["size"], d)] = c
                twin = lms_by_key.get(key)

        if twin is not None:
            if not st["reclaim_duplicates"]:
                actions.append({
                    "kind": "reclaim", "status": "off", "path": e["path"],
                    "name": e["name"], "size": e["size"],
                    "target": twin["path"], "reason": "reclaim_duplicates is off",
                })
            elif twin["dev"] != e["dev"]:
                actions.append({
                    "kind": "reclaim", "status": "blocked", "path": e["path"],
                    "name": e["name"], "size": e["size"], "target": twin["path"],
                    "reason": "the two copies are on different volumes -- a hard"
                              " link cannot span them, and replacing one with a"
                              " symlink across a removable volume is not worth"
                              " the fragility",
                })
            elif _mismatch_key(e["path"], twin["path"]) in mismatched:
                actions.append({
                    "kind": "reclaim", "status": "blocked", "path": e["path"],
                    "name": e["name"], "size": e["size"], "target": twin["path"],
                    "reason": "these two looked identical but a full read said"
                              " otherwise on an earlier pass -- not re-read",
                })
            else:
                actions.append({
                    "kind": "reclaim", "status": "todo", "path": e["path"],
                    "name": e["name"], "size": e["size"], "target": twin["path"],
                    "reason": "the same weights as an LM Studio file on this"
                              " volume, stored twice -- confirmed byte for byte"
                              " before anything is touched",
                })
                out["reclaimable"] += e["size"]
            continue

        if not st["publish_to_lmstudio"]:
            actions.append({
                "kind": "publish", "status": "off", "path": e["path"],
                "name": e["name"], "size": e["size"], "target": str(dst),
                "reason": "publish_to_lmstudio is off",
            })
            continue
        actions.append({
            "kind": "publish", "status": "todo", "path": e["path"],
            "name": e["name"], "size": e["size"], "target": str(dst),
            "mmproj": e["mmproj"],
            "reason": "in the fleet catalogue, not visible to LM Studio",
        })

    out["actions"] = actions
    out["todo"] = sum(1 for a in actions if a["status"] == "todo")
    return out


# ---- applying it ----------------------------------------------------------


def _apply_import(todo: list[dict], st: dict) -> tuple[int, list[str]]:
    """Register LM Studio's GGUFs in models.json, pointing at where they
    already are. No bytes move, which is why this direction works on
    apu-tablet-2's exFAT volume where nothing can be linked."""
    if not todo:
        return 0, []
    notes: list[str] = []
    records = load_models()
    added = 0
    live = _upstream_listening()
    enable = bool(st["import_enabled"]) and live
    if not live:
        notes.append(
            "nothing is listening on " + UPSTREAM + ", so these were registered"
            " DISABLED -- a box with no engine must not advertise models to the"
            " hub. Enable them on the Models tab once llama-swap is running.")
    for a in todo:
        rec = dict(DEFAULT_MODEL_RECORD)
        rec.update({
            "id": a["id"],
            "path": a["path"],
            "enabled": enable,
            "description": "from LM Studio (" + Path(a["path"]).parent.name + ")",
        })
        if a.get("mmproj"):
            rec["mmproj"] = a["mmproj"]
        records.append(rec)
        added += 1
        a["status"] = "done"
        a["detail"] = ("registered as " + a["id"]
                       + ("" if enable else " (disabled: no engine on "
                          + UPSTREAM + ")"))
    if not added:
        return 0, notes
    try:
        check_name_collisions(records)
    except HTTPException as exc:
        # Should not happen -- _fleet_id_for() reserves against ids AND
        # aliases -- but a collision here would have llama-swap refuse the
        # whole config, taking every other model on the box down with it.
        for a in todo:
            a["status"] = "error"
            a["detail"] = "name collision: " + str(exc.detail)[:160]
        return 0, ["import abandoned: " + str(exc.detail)[:200]]
    save_models(records)
    set_lmstudio_settings({
        "imported_paths": sorted(
            set(st.get("imported_paths") or []) | {a["path"] for a in todo}
        ),
    })
    try:
        write_swap_config(records)
    except Exception as exc:  # noqa: BLE001 -- models.json is still correct
        notes.append("llama-swap.yaml not written: " + str(exc)[:160])
    return added, notes


def _apply_publish(todo: list[dict], st: dict) -> tuple[int, int, list[str]]:
    """Link fleet GGUFs into LM Studio's layout."""
    done = 0
    linked_bytes = 0
    notes: list[str] = []
    for a in todo:
        src, dst = Path(a["path"]), Path(a["target"])
        # Never write over a file LM Studio already has. An existing name that
        # is not our link is somebody else's model, and os.replace would eat
        # it without a trace.
        try:
            if dst.exists() or dst.is_symlink():
                dstat = dst.stat()
                sstat = src.stat()
                if (dstat.st_dev, dstat.st_ino) == (sstat.st_dev, sstat.st_ino):
                    a["status"] = "done"
                    a["detail"] = "already linked"
                    continue
                a["status"] = "blocked"
                a["detail"] = ("LM Studio already has a different file at "
                               + str(dst) + " -- refusing to overwrite it")
                continue
        except OSError:
            pass
        how, err = _link_into(src, dst, bool(st["allow_copy"]))
        if not how:
            a["status"] = "blocked"
            a["detail"] = ("cannot link into LM Studio's store: " + err[:160]
                           + " (exFAT refuses both link types on Windows;"
                             " turn on allow_copy to spend the bytes instead)")
            continue
        a["status"] = "done"
        a["detail"] = how
        done += 1
        if how == "copy":
            linked_bytes += int(a.get("size") or 0)
        # A projector is useless to LM Studio without the model beside it, and
        # useless to the model without itself -- publish the pair or neither.
        if a.get("mmproj"):
            mm = Path(a["mmproj"])
            mdst = dst.parent / mm.name
            try:
                if not (mdst.exists() or mdst.is_symlink()):
                    _link_into(mm, mdst, bool(st["allow_copy"]))
            except OSError:
                pass
    return done, linked_bytes, notes


def _apply_reclaim(todo: list[dict]) -> tuple[int, int, list[str]]:
    """Collapse a duplicated pair onto one inode, freeing the fleet copy.

    Order matters and is the whole safety argument:
      1. re-verify size, device and content edges on BOTH files, now, not from
         a plan that may be minutes old;
      2. build the hard link under a temporary name -- if this fails nothing
         has changed;
      3. os.replace the temporary over the fleet path, which is atomic: at
         every instant the fleet path names a complete, correct file, either
         the old copy or the surviving one.
    A llama-server holding the old inode open keeps reading it until it closes,
    so nothing in flight can be corrupted by this."""
    freed = 0
    done = 0
    notes: list[str] = []
    for a in todo:
        dup, keep = Path(a["path"]), Path(a["target"])
        try:
            ds, ks = dup.stat(), keep.stat()
        except OSError as exc:
            a["status"] = "error"
            a["detail"] = "vanished before reclaim: " + str(exc)[:120]
            continue
        if (ds.st_dev, ds.st_ino) == (ks.st_dev, ks.st_ino):
            a["status"] = "done"
            a["detail"] = "already one file"
            continue
        if ds.st_dev != ks.st_dev or ds.st_size != ks.st_size:
            a["status"] = "blocked"
            a["detail"] = "re-check failed: different volume or size"
            continue
        d1 = _edge_digest(dup, ds.st_size)
        d2 = _edge_digest(keep, ks.st_size)
        if not d1 or d1 != d2:
            a["status"] = "blocked"
            a["detail"] = "re-check failed: contents differ"
            continue
        # And now every byte, because the next statement discards a name.
        f1, f2 = _full_digest(dup), _full_digest(keep)
        if not f1 or f1 != f2:
            a["status"] = "blocked"
            a["detail"] = ("the two files match at both ends and in size but"
                           " differ somewhere in the middle -- not collapsed")
            _lmstudio_remember_mismatch(a["path"], a["target"])
            continue
        tmp = dup.with_name(dup.name + ".llmstack-reclaim-" + str(os.getpid()))
        try:
            if tmp.exists() or tmp.is_symlink():
                tmp.unlink()
            os.link(keep, tmp)
            os.replace(tmp, dup)
        except OSError as exc:
            try:
                if tmp.exists() or tmp.is_symlink():
                    tmp.unlink()
            except OSError:
                pass
            a["status"] = "error"
            a["detail"] = "hard link failed: " + str(exc)[:140]
            continue
        freed += int(ds.st_size)
        done += 1
        a["status"] = "done"
        a["detail"] = "collapsed onto " + str(keep)
    return done, freed, notes


# ---- Ollama-backed boxes --------------------------------------------------
#
# gpu-laptop-2 and mini-pc-1 serve Ollama's own catalogue, so registering an LM
# Studio GGUF in models.json changes nothing anyone can see. Ollama can only
# take a GGUF by COPYING it into its content-addressed blob store, which is
# the one thing this subsystem exists to avoid -- so it is never automatic.
#
# When it is asked for explicitly, the copy is undone straight afterwards: an
# Ollama blob is the GGUF verbatim (measured on gpu-laptop-2 -- the blob starts
# with the GGUF magic and its content hash matches its own filename), so once
# `ollama create` has finished, the blob and the LM Studio file are two names
# for identical bytes and can be collapsed onto one inode exactly like any
# other duplicate. Net cost: 2x the model for the length of one copy, 1x
# afterwards.

_OLLAMA_NAME = re.compile(r"^[A-Za-z0-9._:/-]{1,96}$")


def _ollama_binary() -> str:
    """Where `ollama` is, from a process that may be SYSTEM.

    The Windows gateway is a SYSTEM Scheduled Task and Ollama installs
    per-user, so it is usually NOT on SYSTEM's PATH even though it is on the
    interactive user's."""
    found = shutil.which("ollama")
    if found:
        return found
    cands: list[Path] = []
    if platform.system() == "Windows":
        try:
            for prof in Path("C:/Users").iterdir():
                cands.append(prof / "AppData/Local/Programs/Ollama/ollama.exe")
        except OSError:
            pass
        cands.append(Path("C:/Program Files/Ollama/ollama.exe"))
    else:
        cands += [Path("/usr/local/bin/ollama"), Path("/opt/homebrew/bin/ollama"),
                  Path("/usr/bin/ollama")]
    for c in cands:
        try:
            if c.is_file():
                return str(c)
        except OSError:
            continue
    return ""


def _ollama_stores() -> list[Path]:
    """Every plausible Ollama models directory, for finding the blob a create
    just wrote. OLLAMA_MODELS is normally set in the interactive user's
    environment, which a SYSTEM process does not inherit -- so the sibling and
    per-profile guesses matter."""
    out: list[Path] = []

    def add(p: Path | None) -> None:
        if p is None:
            return
        try:
            if (p / "blobs").is_dir() and p not in out:
                out.append(p)
        except OSError:
            pass

    env = os.environ.get("OLLAMA_MODELS", "").strip()
    if env:
        add(Path(env))
    add(MODELS_DIR)
    for sib in ("ollama-models", ".ollama/models"):
        try:
            add(MODELS_DIR.parent / sib)
        except (OSError, ValueError):
            pass
    try:
        add(Path.home() / ".ollama" / "models")
    except (OSError, RuntimeError):
        pass
    roots = [Path("C:/Users")] if platform.system() == "Windows" else [Path("/Users")]
    for root in roots:
        try:
            for prof in root.iterdir():
                add(prof / ".ollama" / "models")
        except OSError:
            continue
    return out


def _ollama_blob_for(size: int, digest: str) -> Path | None:
    """The blob holding exactly these bytes, or None.

    Matched on size first (one stat per blob) and confirmed with the same edge
    digest every other identity check here uses."""
    for store in _ollama_stores():
        try:
            blobs = sorted((store / "blobs").iterdir())
        except OSError:
            continue
        for b in blobs:
            try:
                if not b.is_file() or b.stat().st_size != size:
                    continue
            except OSError:
                continue
            if _edge_digest(b, size) == digest:
                return b
    return None


def lmstudio_ollama_import(path: str, name: str) -> dict:
    """Hand one LM Studio GGUF to Ollama, then take the duplicate back."""
    if not UPSTREAM_MODELS:
        raise HTTPException(400, "this box is not Ollama-backed")
    if not _OLLAMA_NAME.match(name or ""):
        raise HTTPException(400, "bad model name")
    src = Path(path)
    root = lmstudio_root()
    try:
        if root is None or root.resolve() not in src.resolve().parents:
            raise HTTPException(400, "that path is not inside LM Studio's store")
        st = src.stat()
    except OSError as exc:
        raise HTTPException(404, "no such file: " + str(exc)) from exc

    binary = _ollama_binary()
    if not binary:
        raise HTTPException(
            503, "the ollama binary is not reachable from the gateway process"
                 " (on Windows it installs per-user and the gateway runs as"
                 " SYSTEM) -- run `ollama create` on the box instead")

    digest = _edge_digest(src, st.st_size)
    out: dict[str, Any] = {"name": name, "path": str(src), "size": st.st_size}
    # A real Modelfile on disk, not stdin: `ollama create -f -` is not a thing
    # -- measured on gpu-laptop-2 (Ollama 0.12.6), which answers "no Modelfile or
    # safetensors files found" and exits non-zero. Written into STATE, and
    # removed whether or not the create succeeds. argv, never a shell, so the
    # path cannot be quoted into a command.
    mf = STATE / ("llmstack-modelfile-" + str(os.getpid()) + ".txt")
    try:
        mf.write_text("FROM " + str(src) + "\n", encoding="utf-8")
        proc = subprocess.run(
            [binary, "create", name, "-f", str(mf)],
            capture_output=True, text=True, timeout=7200,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, "ollama create failed: " + str(exc)[:200]) from exc
    finally:
        try:
            mf.unlink()
        except OSError:
            pass
    out["rc"] = proc.returncode
    # Either stream can come back None rather than "" -- measured on gpu-laptop-2,
    # where this raised a TypeError AFTER `ollama create` had already copied
    # the weights, so the import succeeded and the reclaim that pays for it
    # never ran. Never let the bookkeeping be what fails.
    out["output"] = ((proc.stdout or "") + (proc.stderr or ""))[-2000:]
    if proc.returncode != 0:
        raise HTTPException(502, "ollama create failed: " + out["output"][-300:])

    # And now give the bytes back. Ollama copied them; the copy is identical,
    # so the blob becomes a second name for the file LM Studio already had.
    out["reclaimed"] = 0
    blob = _ollama_blob_for(st.st_size, digest or "")
    if blob is None:
        out["note"] = ("imported, but Ollama's blob store was not found from"
                       " this process, so the weights are on disk twice --"
                       " set OLLAMA_MODELS in the gateway environment and run"
                       " a sync to collapse them")
        return out
    try:
        bst = blob.stat()
        if (bst.st_dev, bst.st_ino) == (st.st_dev, st.st_ino):
            out["note"] = "already one file"
            return out
        if bst.st_dev != st.st_dev:
            out["note"] = ("imported; the blob store is on a different volume"
                           " from LM Studio's, so the copy cannot be collapsed")
            return out
        tmp = blob.with_name(blob.name + ".llmstack-reclaim")
        if tmp.exists():
            tmp.unlink()
        os.link(src, tmp)
        os.replace(tmp, blob)
    except OSError as exc:
        out["note"] = "imported; could not collapse the copy: " + str(exc)[:160]
        return out
    out["reclaimed"] = int(st.st_size)
    out["note"] = ("imported, and the copy Ollama made was collapsed back onto"
                   " LM Studio's file")
    _upstream_tags_cache["t"] = 0.0
    _routes_cache["t"] = 0.0
    return out


def lmstudio_sync(dry_run: bool = False, kinds: set[str] | None = None) -> dict:
    """One pass. Safe to call from the loop, the download hook and the
    dashboard at once -- the lock makes the second caller wait rather than
    letting two passes plan against each other's half-finished work."""
    with _lmstudio_lock:
        plan = lmstudio_plan()
        st = plan["settings"]
        plan["dry_run"] = dry_run
        plan["imported"] = plan["published"] = plan["reclaimed"] = 0
        plan["freed_bytes"] = plan["copied_bytes"] = 0
        plan["notes"] = []
        plan["swap"] = ""
        if not plan["available"] or dry_run:
            if not dry_run:
                _lmstudio_last.clear()
                _lmstudio_last.update(plan)
            return plan

        def pick(kind: str) -> list[dict]:
            if kinds is not None and kind not in kinds:
                return []
            return [a for a in plan["actions"]
                    if a["kind"] == kind and a["status"] == "todo"]

        # Reclaim first: it is the one that gives disk back, and doing it
        # before publish means a pair that is about to be collapsed is never
        # also linked somewhere new in the same pass.
        n, freed, notes = _apply_reclaim(pick("reclaim"))
        plan["reclaimed"], plan["freed_bytes"] = n, freed
        plan["notes"] += notes

        n, copied, notes = _apply_publish(pick("publish"), st)
        plan["published"], plan["copied_bytes"] = n, copied
        plan["notes"] += notes

        n, notes = _apply_import(pick("import"), st)
        plan["imported"] = n
        plan["notes"] += notes

        if plan["imported"]:
            # The routing table is a 30 s cache built from the catalogue, so
            # without this a model that has just been imported answers "no
            # host in the fleet serves that model" for half a minute -- which
            # reads as the sync not having worked. Same invalidation the peer
            # editor does when the peer set changes.
            _routes_cache["t"] = 0.0
            if st["restart_swap"]:
                plan["swap"] = _lmstudio_restart_swap()

        if plan["imported"] or plan["published"] or plan["reclaimed"]:
            log.info(
                "lmstudio sync: imported %d, published %d, reclaimed %d (%.1f GiB)",
                plan["imported"], plan["published"], plan["reclaimed"],
                plan["freed_bytes"] / (1 << 30),
            )
        _lmstudio_last.clear()
        _lmstudio_last.update(plan)
        return plan


def lmstudio_publish_one(dest: Path) -> str:
    """The hook a finished HuggingFace download calls.

    The 5-minute loop would find it anyway; this is what makes "pull it in the
    Library and it is in LM Studio" feel immediate rather than eventual. Never
    raises: a download that succeeded must never be reported as failed because
    LM Studio could not be reached."""
    try:
        st = get_lmstudio_settings()
        if not st["enabled"] or not st["publish_to_lmstudio"]:
            return ""
        root = lmstudio_root()
        if root is None:
            return ""
        entry = {"path": str(dest), "ino": 0, "dev": 0}
        publisher, repo = _lms_target_for(entry)
        dst = root / publisher / repo / dest.name
        try:
            if dst.exists() or dst.is_symlink():
                a, b = dst.stat(), dest.stat()
                return "already linked" if (a.st_dev, a.st_ino) == (
                    b.st_dev, b.st_ino) else "name taken in LM Studio"
        except OSError:
            pass
        # Serialised against the periodic pass: both can decide to build the
        # same link, and while _link_into() is safe either way (a private temp
        # name, then os.replace), doing it once is cheaper than doing it twice
        # and the log then reads like what happened.
        with _lmstudio_lock:
            how, err = _link_into(dest, dst, bool(st["allow_copy"]))
        if how:
            log.info("lmstudio: published %s -> %s (%s)", dest.name, dst, how)
            return how
        log.info("lmstudio: could not publish %s: %s", dest.name, err[:160])
        return ""
    except Exception:  # noqa: BLE001 -- a side effect never fails a download
        log.exception("lmstudio publish hook failed for %s", dest)
        return ""


async def _lmstudio_sync_loop() -> None:
    """Keeps the two catalogues in step without anyone asking.

    Re-reads its settings every pass, so turning it off or changing the
    interval on the dashboard takes effect at the next tick instead of at the
    next gateway restart. Inert on a box with no LM Studio: lmstudio_root()
    returns None and the pass costs one directory stat."""
    await asyncio.sleep(20)     # let the box finish coming up first
    while True:
        interval = DEFAULT_LMSTUDIO_SETTINGS["interval_secs"]
        try:
            st = get_lmstudio_settings()
            interval = int(st["interval_secs"])
            if st["enabled"]:
                await asyncio.to_thread(lmstudio_sync)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- housekeeping never takes the app down
            log.exception("lmstudio sync failed")
        await asyncio.sleep(max(60, interval))


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------

client: httpx.AsyncClient | None = None


async def _maintenance_loop() -> None:
    """Age rows out of the live views, then delete what has served its
    retention. Once at startup and every 24h after -- a gateway that is
    restarted daily still gets exactly one pass per day.

    Deliberately not a thread and not a cron: it is a handful of indexed
    UPDATEs and DELETEs, so it costs less than the telemetry poll it shares
    the loop with."""
    while True:
        try:
            res = await asyncio.to_thread(maintain)
            moved = sum(res["rolled"].values())
            gone = sum(
                v for k, v in res["purged"].items()
                if k not in ("cutoff", "vacuumed")
            )
            if moved or gone:
                log.info(
                    "retention: archived %d row(s), permanently deleted %d",
                    moved, gone,
                )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- housekeeping never takes the app down
            log.exception("retention maintenance failed")
        await asyncio.sleep(24 * 3600)


async def _public_overview_warm_loop() -> None:
    """Rebuild the sanitized public overview every 8 s -- just inside its
    10 s cache -- so /public/api/overview is always answered from cache.
    Only the hub serves the public page, but the loop is harmless elsewhere:
    one local status sample every few seconds."""
    await asyncio.sleep(2)
    while True:
        started = time.time()
        try:
            data = await public_overview_payload()
            _public_overview_cache.update(t=time.time(), data=data)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- a failed refresh keeps the last good copy
            log.exception("public overview refresh failed")
        # A refresh can itself take the peers' connect timeout; aim for one
        # fresh copy every ~10 s regardless of how long the last one took.
        await asyncio.sleep(max(2.0, 10.0 - (time.time() - started)))


# --------------------------------------------------------------------------
# self-watchdog: a gateway that stops answering must die, not linger
# --------------------------------------------------------------------------
#
# On Windows, one failed accept in uvicorn's proactor event loop ends the
# accept loop WITHOUT ending the process. Observed on apu-tablet-2 on
# 2026-08-28: `OSError [WinError 64] The specified network name is no longer
# available` surfaced in the accept coroutine right as an external-SSD
# unplug/replug was tearing down the hub's in-flight connections, and after
# it nothing ever listened on :8080 again -- while every background task
# kept the process looking alive. The llama-swap /running polls landed, the
# LM Studio sync ran, the DB kept being written; on the box everything was
# healthy except the one thing the fleet needs. The hub's status poll got
# connection refused and the machine read as offline for as long as the
# zombie lived.
#
# Nothing above this process can see that state. Task Scheduler's
# RestartCount (and systemd's Restart=on-failure alike) only fire when the
# process EXITS, and a deaf-but-alive gateway never exits. So the gateway
# watches itself: once loopback /health has answered AND this process owns
# the listening socket (the arm proves both that the probe can succeed
# against this bind and that the listener was really ours), consecutive
# failures mean the HTTP face of the process is gone. The process then exits
# on purpose -- turning a silent zombie into a crash the supervisor was
# already configured to heal (RestartCount=3 / Restart=on-failure).

WATCHDOG_PORT = int(os.environ.get("LLMSTACK_PORT", "8080"))
WATCHDOG_INTERVAL_S = float(os.environ.get("LLMSTACK_WATCHDOG_INTERVAL", "15"))
WATCHDOG_MAX_MISSES = int(os.environ.get("LLMSTACK_WATCHDOG_MISSES", "3"))


def _watchdog_enabled() -> bool:
    """Env kill-switch, read per-call so a test (or an admin) can flip it
    without re-importing the module: LLMSTACK_WATCHDOG=off disarms entirely."""
    return os.environ.get("LLMSTACK_WATCHDOG", "").strip().lower() not in {
        "0", "off", "false", "no",
    }


def _own_listener_pid() -> bool:
    """True when THIS process owns a LISTEN socket on the gateway port.

    The arming gate. A probe that only ever fails must never kill: on a box
    whose bind excludes loopback the /health probe cannot succeed, and a
    watchdog that armed anyway would restart-loop a working gateway. Requiring
    our own PID on the listener also keeps the pytest process (and any other
    python that merely imports this module) permanently disarmed -- it owns
    no such socket, so the watchdog sleeps for its whole life there. Where
    the platform will not enumerate connections unprivileged (macOS), this
    stays False and the watchdog stays dormant: honest silence beats a
    self-inflicted outage."""
    try:
        return any(
            c.status == psutil.CONN_LISTEN
            and c.pid == os.getpid()
            and c.laddr.port == WATCHDOG_PORT
            for c in psutil.net_connections(kind="inet")
        )
    except psutil.Error:
        return False


# Strictly greater than UPSTREAM_HEALTH_BUDGET, and that gap is the whole
# point. Both sides of this probe used to be 5 s, so a llama-swap that took
# its time answering made /health arrive at 5.0 s and this give up at 5.0 s --
# a miss scored against a gateway that was busy doing its job correctly.
WATCHDOG_TIMEOUT_S = float(os.environ.get("LLMSTACK_WATCHDOG_TIMEOUT", "10"))


def _loopback_health_probe() -> str:
    """'ok', 'slow' (the socket accepted and the answer did not come in
    time) or 'dead' (refused, reset, or anything else).

    The distinction is the whole point. A listener that accepts and then
    takes too long is a process under load -- apu-tablet-1 at its commit
    ceiling while a 73 GB model uploaded to the GPU answered /health in
    more than ten seconds three times running, and the watchdog killed a
    gateway that was merely busy (2026-08-29). A listener that refuses is
    the zombie this exists for."""
    try:
        r = httpx.get(f"http://127.0.0.1:{WATCHDOG_PORT}/health",
                      timeout=WATCHDOG_TIMEOUT_S)
        return "ok" if r.status_code == 200 else "dead"
    except (httpx.ReadTimeout, httpx.PoolTimeout, httpx.WriteTimeout):
        return "slow"
    except Exception:  # noqa: BLE001 -- a failure IS the signal, not an error
        return "dead"


def _loopback_health_ok() -> bool:
    return _loopback_health_probe() == "ok"


_watchdog_state: dict[str, Any] = {"armed": False, "misses": 0}


def _self_watchdog_tick() -> None:
    """One probe and the decision that follows it. A loop body on purpose:
    the tests drive this directly rather than guessing at thread timing."""
    if not _watchdog_enabled():
        return
    if draining():
        # An ORDERLY shutdown looks exactly like the zombie this watchdog
        # hunts: uvicorn closes the listening socket first and only then waits
        # for open streams to finish, so from here the listener is gone while
        # the process lives on. Observed on apu-box-1 on 2026-08-28 as "miss 1 of
        # 3" seven seconds before systemd's own SIGKILL. Left alone it is a
        # race the watchdog can win -- os._exit(1) in the middle of a drain,
        # turning the clean ending below into the severed stream it was
        # written to prevent.
        _watchdog_state["misses"] = 0
        return
    listening = _own_listener_pid()
    probe = _loopback_health_probe() if listening else "dead"
    if probe == "ok":
        if not _watchdog_state["armed"]:
            _watchdog_state["armed"] = True
            log.info(
                "self-watchdog: /health answered on 127.0.0.1:%d and the "
                "listener is ours -- armed; %d consecutive failures end "
                "this process", WATCHDOG_PORT, WATCHDOG_MAX_MISSES,
            )
        _watchdog_state["misses"] = 0
        return
    if not _watchdog_state["armed"]:
        return                                    # never proven healthy
    if probe == "slow":
        # Accepted, then took too long: a process under load, not a deaf
        # one. Not a miss, and not a reset either -- a gateway that is
        # genuinely dying can be slow on its way down, so the count it has
        # already earned stands until an answer or a refusal settles it.
        log.warning("self-watchdog: /health accepted but did not answer in "
                    "%.0fs -- busy, not dead; not counted", WATCHDOG_TIMEOUT_S)
        return
    _watchdog_state["misses"] += 1
    log.error(
        "self-watchdog: gateway unhealthy (listener=%s) -- miss %d of %d",
        listening, _watchdog_state["misses"], WATCHDOG_MAX_MISSES,
    )
    if _watchdog_state["misses"] >= WATCHDOG_MAX_MISSES:
        log.error(
            "self-watchdog: the HTTP accept loop is gone but background "
            "tasks keep this process alive; exiting on purpose so the "
            "supervisor's restart-on-failure can bring a listening "
            "gateway back"
        )
        os._exit(1)


def _self_watchdog_loop() -> None:
    while True:
        try:
            _self_watchdog_tick()
        except Exception:  # noqa: BLE001 -- a watchdog must never be the crash
            log.exception("self-watchdog tick failed")
        time.sleep(WATCHDOG_INTERVAL_S)


# --------------------------------------------------------------------------
# graceful drain: a restart ENDS the streams, it does not sever them
# --------------------------------------------------------------------------
#
# What a deploy used to do to a box mid-answer, from apu-box-1's journal on
# 2026-08-28 (the :30 reconcile in deploy-gateway.yml, restarting a gateway
# with two Copilot streams open):
#
#   19:51:38  Stopping llm-gateway.service...
#   19:51:38  uvicorn: Waiting for connections to close.
#   19:51:48  State 'stop-sigterm' timed out. Killing.
#   19:51:48  Killing process 1586 (uvicorn) with signal SIGKILL
#
# Ten seconds, because Mint ships DefaultTimeoutStopSec=10s in
# /etc/systemd/system.conf.d/50_linuxmint.conf and the unit never overrode it.
# SIGKILL leaves cloudflared holding origin sockets that die mid-body with no
# HTTP framing at all, so the edge resets the HTTP/2 streams and the client is
# told `ERR_HTTP2_PROTOCOL_ERROR` -- a transport error for what was really an
# orderly restart, indistinguishable from a firewall problem. Both of that
# session's requests failed at the same instant, which is the signature.
#
# uvicorn's own lifespan shutdown is NO USE for this: it runs AFTER the
# connection drain (see Server.shutdown), so by the time it fires the streams
# have either finished or been cancelled. The signal is the only hook early
# enough. On SIGTERM we therefore get in first -- mark the process draining,
# and ask every open stream to stop -- and only then let uvicorn's own handler
# run. Each relay notices through the same `stop` event the operator's abort
# button uses, and ends its stream the way it ends every other failure: a
# framed SSE error, then [DONE]. The client keeps the tokens it already had
# and sees a reason it can retry on.
_draining = threading.Event()


def draining() -> bool:
    """True once a shutdown signal has arrived. New work is refused and
    in-flight work is being wound up."""
    return _draining.is_set()


DRAIN_MESSAGE = ("this gateway is restarting (deploy or supervisor restart); "
                 "the answer was cut short -- retry")


def drain_in_flight() -> int:
    """Ask every interruptible stream on this box to end, and say how many.

    `drained` rather than a bare abort so the relays can tell this apart from
    the operator pressing stop on the dashboard: the same mechanism stops the
    upstream read, but the metering (503, not 499) and the message the client
    reads are different."""
    with _active_lock:
        jobs = [r for r in _active.values()
                if r.get("kind") in ("inference", "benchmark")]
    n = 0
    for rec in jobs:
        rec["drained"] = True
        if job_abort(rec):
            n += 1
    return n


def _begin_drain() -> None:
    """Runs on the event loop, scheduled from the signal handler."""
    if _draining.is_set():
        return
    _draining.set()
    try:
        n = drain_in_flight()
    except Exception:  # noqa: BLE001 -- never make the shutdown itself fail
        log.exception("drain: could not wind up in-flight work")
        return
    log.info("drain: shutdown signal received; %d in-flight job(s) asked to "
             "end, new requests now refused with 503", n)


def install_drain_signal_handlers() -> None:
    """Chain a drain in front of uvicorn's exit handler.

    uvicorn installs its own via signal.signal() in capture_signals(), which
    wraps serve() -- so by the time this runs during lifespan startup, the
    handler already there IS uvicorn's, and calling it afterwards is what
    keeps the normal shutdown intact. Nothing is scheduled inline: an asyncio
    Event set from a signal handler re-enters loop internals at an arbitrary
    bytecode, so the work goes through call_soon_threadsafe, which is the one
    documented-safe way back onto the loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:                          # not under a loop (tests)
        return
    _prev_handlers: dict[int, Any] = {}

    def chain(signum: int, frame: Any) -> None:
        try:
            loop.call_soon_threadsafe(_begin_drain)
        except RuntimeError:                      # loop already closed
            pass
        prev = _prev_handlers.get(signum)
        if callable(prev):
            prev(signum, frame)

    for signame in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, signame, None)
        if sig is None:
            continue
        try:
            _prev_handlers[int(sig)] = signal.signal(sig, chain)
        except (ValueError, OSError):
            # Not the main thread, or a platform that will not take it. The
            # drain is an improvement on being killed, never a requirement.
            log.warning("drain: could not hook %s; a restart will cut streams "
                        "the way it always did", signame)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Starlette 1.x dropped on_event(); lifespan is the supported hook.
    global client
    db_init()
    client = httpx.AsyncClient(
        base_url=UPSTREAM,
        # No read timeout: a cold model load plus a long completion can take
        # minutes, and cutting that off mid-stream is worse than waiting.
        timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None),
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
    )
    if not MODELS_JSON.exists():
        save_models([])
    resume_orphaned_downloads()
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    await resume_orphaned_batches()
    # An apply the previous process died in the middle of: roll back whatever
    # it never got to prove, rather than leaving it live behind a banner that
    # says it is still being checked.
    reconcile_orphaned_apply()
    # One id per model across the fleet: a record still registered under a
    # spelling the fleet has since settled on is renamed here, once. Never
    # fatal -- a naming pass must not keep a box off the fleet.
    try:
        apply_model_name_convergence()
    except Exception:  # noqa: BLE001
        log.exception("model name convergence failed; registry left as it was")
    # Fleet Pass: seed the public catalogue and domain lists on a database
    # that has never seen them -- an admin who has since edited or deleted
    # rows is never touched, only an empty table is.
    if not db_query("SELECT id FROM public_models LIMIT 1"):
        seed_public_models()
    if not db_query("SELECT id FROM public_domains LIMIT 1"):
        seed_public_domains()
    raise_stale_ctx_ceilings()
    # Every catalogue row needs a family before the public page can group by
    # one, and the reseed will not add it to rows that already exist.
    backfill_public_families()
    # And the same idea for the request budgets: a key stamped under the old
    # numbers is brought up to the ones on the settings tab now, rather than
    # on whenever an admin next happens to save it.
    sync_public_key_limits(get_public_settings())
    maintenance = asyncio.create_task(_maintenance_loop())
    # The public overview polls every peer, and an offline peer costs its
    # whole connect timeout. Refreshed in the background the 10 s cache never
    # goes cold, so the first visitor to example.org/fleet after a quiet
    # spell sees the page in milliseconds instead of "Loading..." for 6 s.
    warm = asyncio.create_task(_public_overview_warm_loop())
    # Keeps LM Studio and the fleet catalogue pointing at the same weights.
    # Started unconditionally -- on a box with no LM Studio the pass is one
    # failed directory stat and costs nothing.
    lmsync = asyncio.create_task(_lmstudio_sync_loop())
    # Keeps the model the public page suggests loaded on the boxes that can
    # hold it, so the first request against a fresh key is not a cold load.
    # Idle on a box with no peers -- see preload_orchestrator().
    preload = asyncio.create_task(_preload_loop())
    # Before the watchdog, so a signal arriving in the first instants of a
    # short-lived process still finds the drain hooked rather than half of it.
    install_drain_signal_handlers()
    # A plain thread ON PURPOSE: its whole point is to keep judging the
    # process when the event loop is the thing that died. See the block above
    # _self_watchdog_loop for the apu-tablet-2 zombie this closes.
    if _watchdog_enabled():
        threading.Thread(
            target=_self_watchdog_loop, name="self-watchdog", daemon=True,
        ).start()
    try:
        yield
    finally:
        preload.cancel()
        lmsync.cancel()
        warm.cancel()
        maintenance.cancel()
        await client.aclose()


app = FastAPI(
    title="open-fleet gateway",
    # The interactive docs were already off; the schema they read was not, and
    # /openapi.json was answering anyone on api.example.com with 59 KB
    # naming every admin route and its parameters. Nothing consumes it -- the
    # dashboard is hand-written against these endpoints -- and an unauthenticated
    # map of the admin surface is a gift to whoever is scanning today.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


# How long /health may spend asking the ENGINE whether it is alive. It has to
# stay comfortably under WATCHDOG_TIMEOUT_S, because this handler is what the
# self-watchdog probes: at the 5 s both once used, a llama-swap wedged for a
# few seconds made the handler answer at 5.0 s and the watchdog give up at
# 5.0 s, scoring a miss against a gateway that was streaming tokens perfectly
# well. Three of those in a row and it would have killed itself. The engine's
# state is a FIELD in this response, never the reason it fails to arrive.
UPSTREAM_HEALTH_BUDGET = 3.0


@app.get("/health")
async def health() -> dict:
    up = "unknown"
    try:
        assert client is not None
        r = await client.get("/health", timeout=UPSTREAM_HEALTH_BUDGET)
        up = "ok" if r.status_code < 500 else "degraded"
    except Exception:  # noqa: BLE001
        up = "down"
    out = {"status": "ok", "upstream": up, "time": now()}
    # What the deployer reads before it restarts anything: a box mid-stream is
    # deferred to the next reconcile instead of being SIGKILLed at whatever
    # DefaultTimeoutStopSec the distro happens to ship (10s on Mint). A bare
    # count is all it is -- how much work is in flight, never whose or what --
    # which is why it can live on an endpoint that has no authentication.
    out["inflight"] = inflight_work()
    # Only on a gated box, so /health stays byte-for-byte what it was
    # everywhere else -- this endpoint is unauthenticated and is what the
    # tunnel health checks read.
    av = availability()
    if av["gated"]:
        out["available"] = av["available"]
        out["availability_reason"] = av["reason"]
    return out


# ---------------------------- inference proxy -----------------------------

PASSTHRU_PREFIXES = ("/v1/", "/api/")

# ---- fleet inference routing ----
#
# A hub with no GPU of its own (or one whose GPU lacks the requested model)
# forwards the request to whichever peer serves that model. This is what makes
# a single base URL cover every box: clients name a model, not a machine.
#
# Peer /v1 is bearer-authenticated like any other client, so the hub holds an
# inference key per peer. It mints that key itself over the peer's admin API
# the first time it needs one -- no manual key shuttling between boxes.

_routes_cache: dict[str, Any] = {"t": 0.0, "map": {}, "cands": {},
                                 "cap": {}, "running": {}, "ctx": {},
                                 "reachable": set(),
                                 # (host, model) -> {bytes, fit, moe, source}
                                 "meta": {},
                                 # (host, name) -> the canonical model id that
                                 # name resolves to ON THAT HOST. Aliases are
                                 # per-box registry entries, so this is the
                                 # only way to tell whether two boxes offering
                                 # "triage" are offering the same weights.
                                 "alias": {},
                                 # host -> "llama-swap" | "ollama" | "none"
                                 "engine": {},
                                 # host -> the model ids it preloads or keeps
                                 # resident by its own registry; the preload
                                 # loop leaves such a box alone
                                 "warm": {}}
ROUTE_TTL = 30.0

# Requests in flight per host, this gateway's own traffic only. It cannot see
# what other clients send a peer directly, but for the hub -- through which all
# fleet traffic flows -- it is the truth, and it is what keeps a second batch
# from piling onto a box the first one has already saturated.
_inflight: dict[str, int] = {}

# When each host last finished serving something for us. Only the preload loop
# reads it, and only to stay out of the way: a box whose last request was a
# minute ago is very likely about to get the next turn of the same
# conversation, and swapping the model out from under it to restore a pinned
# one would make that turn pay a cold load to save a later one.
_host_last_used: dict[str, float] = {}


def _track(host: str):
    """Count a request against `host` until the returned closure runs."""
    _inflight[host] = _inflight.get(host, 0) + 1
    fired = False

    def done() -> None:
        nonlocal fired
        if not fired:
            fired = True
            _inflight[host] = max(0, _inflight.get(host, 1) - 1)
            _host_last_used[host] = time.time()
    return done


# A box that just failed a request is not chosen again for a little while.
# The routing table only ever knew "up" or "absent": a peer whose llama-server
# OOMed on a cold load still answered /admin/api/served-models, so the very
# next request for that model was sent straight back to it -- and every
# OpenAI SDK retries by default, which turned one slow failure into three.
# The cooldown is short and in-memory; a host earns its way back by the
# timer running out, or immediately by answering a request properly.
_host_cooldown: dict[str, float] = {}
COOLDOWN_CONNECT = 60.0
COOLDOWN_UPSTREAM_5XX = 45.0
COOLDOWN_STALL = 120.0
COOLDOWN_MIDSTREAM = 60.0
# A box that answered 429/503 is BUSY, not broken -- it is very likely to
# have a free slot again soon, so it earns a short sit-out rather than the
# 45s a real failure gets. The public `busy_cooldown_seconds` setting is the
# live source of truth; this is only the fallback for a caller that cannot
# reach settings (or a bad stored value).
COOLDOWN_BUSY_DEFAULT = 5.0


def _busy_cooldown_seconds() -> float:
    try:
        v = float(get_public_settings().get("busy_cooldown_seconds")
                  or COOLDOWN_BUSY_DEFAULT)
    except (TypeError, ValueError):
        return COOLDOWN_BUSY_DEFAULT
    return v if v > 0 else COOLDOWN_BUSY_DEFAULT


def _mark_host_down(host: str, seconds: float, why: str = "") -> None:
    host = host or HOST_NAME
    until = time.time() + max(1.0, seconds)
    if until > _host_cooldown.get(host, 0.0):
        _host_cooldown[host] = until
    log.warning("routing: %s cooled down for %ds -- %s", host, int(seconds), why)


def _mark_host_ok(host: str) -> None:
    _host_cooldown.pop(host or HOST_NAME, None)


def host_cooling(host: str) -> bool:
    host = host or HOST_NAME
    until = _host_cooldown.get(host)
    if not until:
        return False
    if until <= time.time():
        _host_cooldown.pop(host, None)
        return False
    return True


# --------------------------------------------------------------------------
# active work: what this box is doing right now, and how to stop it
# --------------------------------------------------------------------------
#
# Every long-running thing here was already stoppable in principle -- a batch
# has a cancel flag, a download has one, an inference request dies when its
# client hangs up -- but none of it was visible and nothing tied the levers
# together. A model looping to the end of a large max_tokens budget could only
# be stopped by unloading the GPU or restarting llama-swap, which takes every
# other request on the box down with it.
#
# So: one registry of the work in flight, one id per unit of it, and an abort
# that reaches the right lever for each kind.
#
# For inference the lever is the upstream connection: llama-server frees a
# decode slot the moment its client disconnects. Getting there is fiddlier than
# it looks. Closing the httpx response from the aborting task does NOT
# interrupt a read already pending on it -- the read simply stays parked, which
# is how a wedged request survived its own abort in testing. Cancelling the
# task that holds the read is not ours to do either: Starlette streams a
# response from inside an anyio task group.
#
# So every read is raced against an abort event instead. The read is wrapped in
# a task WE created, which makes cancelling it unambiguously ours, and dropping
# it closes the connection on the way out. That covers both halves of the ask:
# a runaway producing tokens too fast, and a request stuck producing none.

_active: dict[str, dict] = {}
_active_n = 0
_active_lock = threading.Lock()   # benchmarks register from a worker thread


def _job_open(kind: str, **meta: Any) -> dict:
    """Register a unit of work as active on this box. The caller must close it."""
    global _active_n
    with _active_lock:
        _active_n += 1
        rec: dict[str, Any] = {
            "id": kind + ":" + str(_active_n), "kind": kind,
            "started": time.time(), "aborted": False, "host": HOST_NAME,
            "what": "", "detail": "", "origin": "",
            "stop": None, "proc": None,
        }
        rec.update(meta)
        _active[str(rec["id"])] = rec
    return rec


def _job_close(rec: dict | None) -> None:
    if rec:
        with _active_lock:
            _active.pop(str(rec.get("id", "")), None)


def inflight_work() -> int:
    """How many pieces of UNINTERRUPTIBLE work this box is doing right now.

    Inference and benchmarks only. A download or a batch is resumed from the
    database after a restart (see active_jobs()), but a token stream and a
    llama-bench run exist nowhere but this process: killing either throws away
    everything it had already produced, and on the client side a severed
    stream is not even a legible error -- through cloudflared it reaches a
    browser as ERR_HTTP2_PROTOCOL_ERROR, which says nothing about what
    happened. This is the same predicate the llama-swap restart already defers
    on, named once so /health, the deployer and that restart cannot drift into
    disagreeing about what "busy" means."""
    with _active_lock:
        return sum(1 for r in _active.values()
                   if r.get("kind") in ("inference", "benchmark"))


def _hard_kill(proc: Any) -> None:
    try:
        if proc.poll() is None:
            proc.kill()
    except Exception:  # noqa: BLE001 -- already gone is the outcome we wanted
        pass


def job_abort(rec: dict) -> bool:
    """Pull whichever lever this kind of work answers to."""
    # Once only, so a double-click cannot fire a second kill timer at a pid
    # the OS may already have handed to something else.
    if rec.get("aborted"):
        return False
    rec["aborted"] = True
    proc = rec.get("proc")
    if proc is not None:
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        else:
            # llama-bench goes on SIGTERM; wedged deep inside a driver call it
            # will not, and the whole point of this button is that something
            # stops either way.
            threading.Timer(5.0, _hard_kill, (proc,)).start()
    stop = rec.get("stop")
    if stop is not None:
        stop.set()
    return True


async def _quiet_close(*closeables: Any) -> None:
    """Close what is left over. An abort cancels the read mid-flight, and a
    response in that state does not always close cleanly -- but by then the
    connection is going away regardless, which is the whole point."""
    for c in closeables:
        if c is None:
            continue
        try:
            await c.aclose()
        except Exception:  # noqa: BLE001
            pass


def _abandon(task: "asyncio.Task") -> None:
    """Walk away from a read we no longer want.

    Cancel it, then swallow whatever it ends up raising: a socket torn down
    mid-read usually raises rather than cancelling cleanly, and an unretrieved
    task exception is a full traceback in the log for every abort -- and for
    every client that simply hangs up mid-stream."""
    task.cancel()
    task.add_done_callback(lambda t: t.cancelled() or t.exception())


async def _race_abort(work: Any, job: dict) -> tuple[Any, bool]:
    """Await `work`, giving up the moment `job` is aborted.

    Returns (value, aborted). The pending read lives in a task created here,
    so cancelling it is ours to do -- and cancelling it is what drops the
    upstream connection, which is what actually stops the model."""
    stop = job.get("stop")
    if stop is None:
        return await work, False
    task = asyncio.ensure_future(work)
    waiter = asyncio.ensure_future(stop.wait())
    try:
        await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:
        # Cancelled from outside: the client hung up, and Starlette is tearing
        # the response down. The read has to be dropped here too -- nothing
        # downstream is left to notice it finish.
        _abandon(task)
        raise
    finally:
        waiter.cancel()
    if task.done():
        return task.result(), False
    _abandon(task)
    return None, True


def _job_view(rec: dict) -> dict:
    """The JSON shape. The handles themselves never leave this process."""
    return {
        "id": rec["id"],
        "kind": rec["kind"],
        "host": rec.get("host") or HOST_NAME,
        "what": rec.get("what") or "",
        "detail": rec.get("detail") or "",
        "age_s": int(time.time() - float(rec["started"])),
        "aborted": bool(rec.get("aborted")),
        "origin": rec.get("origin") or "",
        "progress": rec.get("progress"),
    }


def _iso_age(ts: Any) -> int:
    try:
        return max(0, int((datetime.now(timezone.utc)
                           - datetime.fromisoformat(str(ts))).total_seconds()))
    except (TypeError, ValueError):
        return 0


def active_jobs() -> list[dict]:
    """Everything this box is working on right now.

    Two sources, deliberately. Inference and benchmarks live in this process
    and would not survive a restart anyway; downloads and batches are resumed
    from the database at boot, so the database is where their truth is."""
    with _active_lock:
        jobs = [_job_view(r) for r in _active.values()]
    for r in db_query(
        "SELECT id,repo,filename,status,bytes_done,bytes_total,created_at "
        "FROM jobs WHERE status IN ('queued','downloading') ORDER BY id"
    ):
        jid = int(r["id"])
        jobs.append({
            "id": "download:" + str(jid),
            "kind": "download",
            "host": HOST_NAME,
            "what": str(r["filename"]).rsplit("/", 1)[-1],
            "detail": str(r["repo"]),
            "age_s": _iso_age(r["created_at"]),
            "aborted": jid in _download_cancel,
            "origin": "",
            "progress": {"done": int(r["bytes_done"] or 0),
                         "total": int(r["bytes_total"] or 0), "unit": "bytes"},
        })
    for r in db_query(
        "SELECT id,label,models,key_name,total,done,failed,created_at "
        "FROM batches WHERE status='running' ORDER BY id"
    ):
        bid = int(r["id"])
        try:
            models = ", ".join(json.loads(r["models"] or "[]"))
        except (json.JSONDecodeError, TypeError):
            models = ""
        live = _batch_live.get(bid) or {}
        done = int(live.get("done", r["done"]) or 0)
        failed = int(live.get("failed", r["failed"]) or 0)
        jobs.append({
            "id": "batch:" + str(bid),
            "kind": "batch",
            "host": HOST_NAME,
            "what": "#" + str(bid) + " " + str(r["label"] or "batch"),
            "detail": " · ".join(x for x in (models, str(r["key_name"] or "")) if x),
            "age_s": _iso_age(r["created_at"]),
            "aborted": bid in _batch_cancel,
            "origin": "",
            "progress": {"done": done + failed, "total": int(r["total"] or 0),
                         "failed": failed},
        })
    return sorted(jobs, key=lambda j: -int(j["age_s"]))


def abort_one(job_id: str) -> dict:
    """Abort by id. Downloads and batches carry their state in the database,
    so their ids are resolved against it rather than against the registry."""
    kind, _, raw = str(job_id).partition(":")
    if kind in ("download", "batch"):
        try:
            n = int(raw)
        except ValueError:
            raise HTTPException(400, "bad job id: " + str(job_id))
        (_download_cancel if kind == "download" else _batch_cancel).add(n)
        return {"aborted": job_id, "kind": kind}
    with _active_lock:
        rec = _active.get(str(job_id))
    if rec is None:
        raise HTTPException(404, "no such active job -- it may have just finished")
    job_abort(rec)
    return {"aborted": job_id, "kind": rec["kind"]}


def abort_all(kinds: set[str] | None = None) -> dict:
    """Stop everything on this box (optionally only some kinds of it)."""
    stopped: list[str] = []
    for job in active_jobs():
        if kinds and job["kind"] not in kinds:
            continue
        if job["aborted"]:
            continue
        try:
            abort_one(str(job["id"]))
            stopped.append(str(job["id"]))
        except HTTPException:
            continue          # finished between the listing and the abort
    return {"host": HOST_NAME, "aborted": stopped, "count": len(stopped)}


_tps_cache: dict[str, tuple[float, dict[str, float]]] = {}


def measured_tps(model: str) -> dict[str, float]:
    """Tokens/second each host has actually delivered for this model lately.

    Read from the usage log, not from a benchmark: it reflects the model as
    configured on that box (quant, slots, context) under real traffic. Small
    samples are ignored -- a single 3-token reply proves nothing."""
    hit = _tps_cache.get(model)
    if hit and time.time() - hit[0] < 300:
        return hit[1]
    out: dict[str, float] = {}
    try:
        for r in db_query(
            "SELECT host, COALESCE(SUM(completion_tokens),0) ct, "
            "COALESCE(SUM(latency_ms),0) ms FROM usage "
            "WHERE model=? AND host IS NOT NULL AND host != '' "
            "AND status<400 AND ts>=? GROUP BY host",
            (model, days_ago(14)),
        ):
            if int(r["ms"] or 0) > 0 and int(r["ct"] or 0) > 200:
                out[str(r["host"])] = float(r["ct"]) / (float(r["ms"]) / 1000.0)
    except Exception:  # noqa: BLE001 -- no history is a valid answer
        pass
    _tps_cache[model] = (time.time(), out)
    return out


def _spec_speed(host: str) -> float:
    """The spec sheet's memory bandwidth, the best stand-in for decode speed
    on a box with no measured history yet."""
    try:
        return float(load_specs().get(host, {}).get("mem_bw_gbs") or 0)
    except (TypeError, ValueError):
        return 0.0


_pp_cache: dict[str, tuple[float, dict[str, float]]] = {}


def measured_pp(model: str) -> dict[str, float]:
    """Prompt tokens/second each host has actually processed for this model
    lately -- time-to-first-token over prompt size, from streamed requests
    (the only ones that record a TTFT). Small prompts are ignored: a 40-token
    prompt's first byte is dominated by the model's own warm-up, not by how
    fast the box reads.

    This is the number the qwen3.6-35b incident was missing. Decode speed
    alone made gpu-laptop-1 look fine; it was the 42k-token prompt that took
    forty seconds to read on an 8 GB card."""
    hit = _pp_cache.get(model)
    if hit and time.time() - hit[0] < 300:
        return hit[1]
    out: dict[str, float] = {}
    try:
        for r in db_query(
            "SELECT host, COALESCE(SUM(prompt_tokens),0) pt, "
            "COALESCE(SUM(ttft_ms),0) ms FROM usage "
            "WHERE model=? AND host IS NOT NULL AND host != '' "
            "AND status<400 AND ttft_ms IS NOT NULL AND ttft_ms>0 "
            "AND prompt_tokens>=512 AND ts>=? GROUP BY host",
            (model, days_ago(14)),
        ):
            if int(r["ms"] or 0) > 0 and int(r["pt"] or 0) > 2000:
                out[str(r["host"])] = float(r["pt"]) / (float(r["ms"]) / 1000.0)
    except Exception:  # noqa: BLE001 -- no history is a valid answer
        pass
    _pp_cache[model] = (time.time(), out)
    return out


# --------------------------------------------------------------------------
# host policy: which box a model should land on, before the numbers decide
#
# The fleet is not a pool of interchangeable GPUs, and the owner ranked it:
#
#   * a GPU box that holds the whole model in its own memory (gpu-desktop-2's pair
#     of 3090s, the M1 Max's unified memory, gpu-desktop-1's 16 GB card, the gpu-laptop-2
#     laptop's 4070, gpu-laptop-1 for anything under 8 GB) answers faster than
#     the big box, whatever the big box's VRAM total says -- it is
#     bandwidth-bound -- so those come first whenever the model FITS;
#   * apu-box-1 (and apu-tablet-1, the same silicon) is the box for everything
#     that only it can hold, and the fallback behind the fast cards;
#   * the always-on small boxes -- gpu-laptop-1, server-1, mac-desktop-1, mini-pc-1 -- in
#     that order, are where sub-agents go so the big box stays free for the
#     primary; for a primary they sit behind apu-box-1;
#   * cpu-box-1, CPU-only, is the backstop for anything any box but apu-box-1 can
#     serve: last, always, but there.
#
# Within a tier the expected wall time decides (see _est_wall). The spec
# table carries the classification (`klass`, `always_on`, `moe_spill_ok`),
# so a new box is a line of configuration, not a new branch here.
# --------------------------------------------------------------------------

# Stand-ins for a (host, model) pair with no measured history: decode and
# prompt-processing tokens/sec by host class. Deliberately rough -- they only
# have to order boxes sensibly until the usage log knows better.
_CLASS_SPEED: dict[str, tuple[float, float]] = {
    "gpu": (45.0, 1500.0), "big": (30.0, 900.0), "small": (10.0, 150.0),
    "fallback": (4.0, 40.0), "hub": (1.0, 10.0),
}


def host_class(host: str) -> str:
    """'gpu' | 'big' | 'small' | 'fallback' | 'hub' from the spec sheet, or
    'unknown' for a box the sheet has never heard of -- which is ranked with
    the big box rather than guessed at: a new peer should neither jump the
    fast cards nor sink below the CPU backstop until someone classes it."""
    specs = load_specs()
    hname = host or HOST_NAME
    if hname not in specs:
        return "unknown"
    spec = specs.get(hname) or {}
    k = str(spec.get("klass") or "")
    if k:
        return k
    if spec.get("role") == "hub":
        return "hub"
    try:
        v = float(spec.get("vram_gb") or 0)
    except (TypeError, ValueError):
        v = 0.0
    if v >= 64:
        return "big"
    return "gpu" if v > 0 else "small"


def host_reserved(host: str) -> bool:
    """Is this box somebody's personal machine? (spec-sheet `reserve`.)

    A reserve box is a full fleet member -- registered, warm with whatever its
    owner keeps on it, visible everywhere -- that the scorer only picks when
    every non-reserve candidate is saturated or cooling. Distinct from the
    killswitch (peers.json `routed`), which removes a box from routing
    entirely: reserve means "last resort", not "never"."""
    spec = load_specs().get(host or HOST_NAME, {})
    return bool(spec.get("reserve"))


def host_tier(cand: str, fleet_id: str, role: str = "primary") -> tuple[int, int]:
    """(tier, rank) for one candidate, lower first. `cand` is spelled the way
    the routing table spells it ('' for this box). `role` is 'primary' for a
    client's own request and 'worker' for a team's spawned sub-agent; the
    only difference is where the big box sits relative to the always-on
    small ones. Tier 0 is a GPU box holding the whole model in its own
    memory; the spec sheet's `moe_spill_ok` lets a box (gpu-desktop-2) keep that
    tier for a mixture-of-experts model whose experts overflow into RAM."""
    hname = cand or HOST_NAME
    spec = load_specs().get(hname, {})
    klass = host_class(hname)
    meta = _routes_cache.get("meta", {}).get((cand, fleet_id)) or {}
    fit = str(meta.get("fit") or "")
    # None means "unranked", which each branch fills with its own default;
    # an explicit 0 is a real rank (first in its tier) and must survive, so
    # this is deliberately not `a or b or 0`.
    raw = spec.get("rank")
    if raw is None:
        raw = spec.get("always_on")
    try:
        rank: int | None = int(raw) if raw is not None else None
    except (TypeError, ValueError):
        rank = None

    def r(default: int) -> int:
        return default if rank is None else rank

    worker = role == "worker"
    if klass == "hub":
        return 6, 0
    if klass == "fallback":
        return 4, r(9)
    if klass == "gpu":
        fits = fit in ("vram", "unified")
        if not fits and fit == "spill" and meta.get("moe") and spec.get("moe_spill_ok"):
            fits = True
        if not fit:
            # A peer on an older gateway reports no fit; believe the spec
            # sheet, which is the same thing the old scorer did implicitly.
            try:
                fits = float(spec.get("vram_gb") or 0) >= 24
            except (TypeError, ValueError):
                fits = False
        if fits:
            return 0, r(0)
        # The card overflows: behind the big box for a primary, in the
        # always-on order for a worker. An intermittent box (gpu-desktop-1, gpu-laptop-2)
        # spilling sits behind the always-on ones, as ranked.
        return (1 if worker else 2), r(5)
    if klass == "big":
        return (2 if worker else 1), r(0)
    if klass == "unknown":
        return (2 if worker else 1), r(5)
    return (1 if worker else 2), r(6)


def _est_wall(cand: str, fleet_id: str, resident: bool, tps: dict[str, float],
              pp: dict[str, float], prompt_tokens: int, gen_tokens: int) -> float:
    """Seconds until this box would have answered: the cold load it would
    have to pay, reading the prompt, then generating. Measured rates where
    the usage log has them, class stand-ins where it does not, halved for a
    model the box reports as spilling out of its GPU memory."""
    hname = cand or HOST_NAME
    spec = load_specs().get(hname, {})
    meta = _routes_cache.get("meta", {}).get((cand, fleet_id)) or {}
    fit = str(meta.get("fit") or "")
    klass = host_class(hname)
    tg_default, pp_default = _CLASS_SPEED.get(klass, _CLASS_SPEED["big"])
    try:
        tg = float(tps.get(hname) or spec.get("tg_tps") or 0) or tg_default
        ppr = float(pp.get(hname) or spec.get("pp_tps") or 0) or pp_default
    except (TypeError, ValueError):
        tg, ppr = tg_default, pp_default
    if fit in ("spill", "cpu"):
        if not tps.get(hname):
            tg *= 0.5
        if not pp.get(hname):
            ppr *= 0.4
    load = 0.0
    if not resident:
        try:
            b = int(meta.get("bytes") or 0)
        except (TypeError, ValueError):
            b = 0
        # NVMe into GPU memory, llama-swap's own start-up included.
        load = max(5.0, b / 1.5e9) if b else 20.0
    return load + max(0, prompt_tokens) / max(ppr, 1.0) + max(0, gen_tokens) / max(tg, 0.5)


def estimate_prompt_tokens(payload: Any) -> int:
    """The same chars/3.2 estimate apply_ctx_limit() guards with, for
    routing: how much prompt the chosen box is about to have to read."""
    if not isinstance(payload, dict):
        return 0
    body = payload.get("messages")
    if body is None:
        body = payload.get("prompt")
    try:
        est = math.ceil(len(json.dumps(body, ensure_ascii=False)) / 3.2 * 1.1)
        est += math.ceil(len(json.dumps(payload.get("tools") or [],
                                        ensure_ascii=False)) / 3.2)
    except (TypeError, ValueError):
        return 0
    return int(est)

# On boxes whose upstream is Ollama (or anything else that manages its own
# models), there is no models.json -- the upstream's own catalogue is the truth.
# This flag makes the gateway serve and route whatever the upstream reports.
UPSTREAM_MODELS = os.environ.get("LLMSTACK_MODELS_FROM_UPSTREAM", "0") == "1"
_upstream_cache: dict[str, Any] = {"t": 0.0, "ids": set()}


async def upstream_model_ids() -> set[str]:
    if not UPSTREAM_MODELS:
        return set()
    if time.time() - _upstream_cache["t"] < ROUTE_TTL:
        return _upstream_cache["ids"]
    ids: set[str] = set()
    try:
        assert client is not None
        r = await client.get("/v1/models", timeout=6.0)
        if r.status_code == 200:
            for m in r.json().get("data") or []:
                if m.get("id"):
                    ids.add(str(m["id"]))
    except Exception:  # noqa: BLE001 -- upstream down just serves nothing
        pass
    _upstream_cache.update(t=time.time(), ids=ids)
    return ids


async def upstream_alias_pairs() -> dict[str, str]:
    """Canonical fleet id -> the Ollama tag it stands for, on an Ollama box.

    A tag cannot be renamed, so the gateway answers to the fleet's canonical
    id beside it: advertised to the hub as one more served name, with the
    tag's context ceiling and metadata, and rewritten back to the tag on the
    way into the engine (the proxy's send step). A tag with no canonical
    spelling, or whose canonical id is already a real name on this box, is
    left alone. Empty everywhere but an Ollama box."""
    if not UPSTREAM_MODELS:
        return {}
    tags = await upstream_model_ids()
    local = local_model_ids()
    out: dict[str, str] = {}
    for tag in sorted(tags):
        canon = SPELLING_TO_CANONICAL.get(tag)
        if canon and canon != tag and canon not in tags and canon not in local:
            out.setdefault(canon, tag)
    return out


async def served_canonical_map() -> dict[str, str]:
    """Every name this box answers to -> the canonical model id behind it,
    for both engines: models.json ids and aliases (local_alias_map), and on
    an Ollama box each tag -- to itself, or to the fleet's canonical id when
    the gateway answers to that id for it. This is what the hub compares
    boxes by, so the tags have to be in it too: a box that says nothing
    about a name is read as 'cannot check', never as 'agrees'."""
    out = local_alias_map()
    if UPSTREAM_MODELS:
        for tag in await upstream_model_ids():
            out.setdefault(tag, tag)
        for canon, tag in (await upstream_alias_pairs()).items():
            out[tag] = canon
            out[canon] = canon
    return out


async def served_model_ids() -> set[str]:
    return (local_model_ids() | await upstream_model_ids()
            | set(await upstream_alias_pairs()))


_running_cache: dict[str, Any] = {"t": 0.0, "ids": set()}


async def upstream_running_ids() -> set[str]:
    """Model ids resident on this box's own engine right now -- the light
    probe, ids only, for routing decisions. swap_running() is the full
    telemetry version; this one must stay cheap because the hub asks for it
    on every routing-table rebuild."""
    if time.time() - _running_cache["t"] < 15:
        return _running_cache["ids"]
    ids: set[str] = set()
    assert client is not None
    try:
        r = await client.get("/running", timeout=3.0)
        if r.status_code == 200:
            data = r.json()
            for m in (data.get("running") if isinstance(data, dict) else data) or []:
                if isinstance(m, dict) and (m.get("model") or m.get("id")):
                    ids.add(str(m.get("model") or m.get("id")))
            _running_cache.update(t=time.time(), ids=ids)
            return ids
    except Exception:  # noqa: BLE001
        pass
    # Anything other than a 200 with a usable body falls through here -- not
    # just a raised exception. Ollama has no /running at all; it answers 404,
    # which the block above swallowed without ever trying /api/ps, so an
    # Ollama host never had a resident model to route to. swap_running() got
    # this right already (it falls through on any non-200 too); this mirrors it.
    try:
        r = await client.get("/api/ps", timeout=3.0)
        if r.status_code == 200:
            for m in (r.json() or {}).get("models") or []:
                if isinstance(m, dict) and (m.get("model") or m.get("name")):
                    ids.add(str(m.get("model") or m.get("name")))
    except Exception:  # noqa: BLE001
        pass
    _running_cache.update(t=time.time(), ids=ids)
    return ids


_upstream_tags_cache: dict[str, Any] = {"t": 0.0, "models": []}
UPSTREAM_TAGS_TTL = 10.0


async def upstream_catalogue() -> dict:
    """The upstream engine's own model list, shaped for the dashboard.

    On an Ollama-backed box (LLMSTACK_MODELS_FROM_UPSTREAM=1) there is no
    models.json -- `ollama pull` on the box is the whole catalogue -- so the
    Models and Library tabs used to render empty while every OTHER admin
    surface pointed at the same daemon (downstream-app's admin page, the hub's
    routing table) showed the box's real models. Serving the live /api/tags
    list here makes them all views of one source: a pull done anywhere is
    visible everywhere on the next refresh, and nothing can drift."""
    if not UPSTREAM_MODELS:
        return {"enabled": False, "models": []}
    now = time.time()
    if now - _upstream_tags_cache["t"] >= UPSTREAM_TAGS_TTL:
        rows: list[dict] = []
        got_answer = False
        try:
            assert client is not None
            r = await client.get("/api/tags", timeout=6.0)
            if r.status_code == 200:
                got_answer = True
                for m in (r.json() or {}).get("models") or []:
                    if not isinstance(m, dict):
                        continue
                    mid = str(m.get("model") or m.get("name") or "").strip()
                    if not mid:
                        continue
                    det = m.get("details") or {}
                    rows.append({
                        "id": mid,
                        "size": int(m.get("size") or 0),
                        "modified": str(m.get("modified_at") or "")[:19],
                        "params": str(det.get("parameter_size") or ""),
                        "quant": str(det.get("quantization_level") or ""),
                        "family": str(det.get("family") or ""),
                    })
        except Exception:  # noqa: BLE001 -- engine down
            pass
        if not got_answer:
            # Engine unreachable OR answered non-200 (e.g. a 500 mid-restart):
            # keep the last good answer rather than clobbering the catalogue
            # with an empty list for the next TTL window.
            rows = list(_upstream_tags_cache["models"])
        _upstream_tags_cache.update(t=now, models=rows)
    ctx = await upstream_model_ctx()
    running = await upstream_running_ids()
    # tag -> the fleet's canonical id this gateway also answers to for it.
    also = {tag: canon for canon, tag in (await upstream_alias_pairs()).items()}
    return {
        "enabled": True,
        "models": [dict(m, ctx=ctx.get(m["id"], 0), running=m["id"] in running,
                        also=also.get(m["id"], ""))
                   for m in _upstream_tags_cache["models"]],
    }


def local_warm_ids() -> set[str]:
    """The model ids this box preloads at start-up or keeps resident by its
    own registry -- what it is FOR. Reported to the hub in served-models so
    the featured-model preload loop leaves a dedicated box alone; a hub
    that pinned its own favourite onto every capable box is how a 125B got
    evicted every ten minutes from the machine that exists to hold it."""
    return {str(m.get("id", "")).strip() for m in load_models()
            if m.get("enabled", True) and (m.get("preload") or m.get("persistent"))
            and str(m.get("id", "")).strip()}


def local_capacity() -> dict[str, int]:
    """Decode slots per model id (and alias) from this box's own registry.
    Upstream-managed catalogues (Ollama) publish nothing, so their models
    default to 1 at the caller."""
    caps: dict[str, int] = {}
    for rec in load_models():
        if not rec.get("enabled", True) or not rec.get("id"):
            continue
        par = max(1, int(rec.get("parallel", 1) or 1))
        for name in [str(rec["id"]), *[str(a) for a in rec.get("aliases") or []]]:
            if name.strip():
                caps[name.strip()] = par
    return caps


def local_model_ids() -> set[str]:
    ids: set[str] = set()
    for rec in load_models():
        if not rec.get("enabled", True):
            continue
        mid = str(rec.get("id", "")).strip()
        if mid:
            ids.add(mid)
            for a in rec.get("aliases") or []:
                if isinstance(a, str) and a.strip():
                    ids.add(a.strip())
    return ids


_known_ctx_cache: dict[str, Any] = {"t": 0.0, "map": {}, "written": None}


def remember_model_ctx(ctx: dict[tuple[str, str], int]) -> None:
    """Persist every (host, model) context ceiling a routing refresh learned.

    The ceiling a request form offers has to include boxes that are asleep
    right now: the largest window in the fleet lives on the largest box, and
    that box is exactly the one that is not always awake. Remembering it is
    what makes the promise offerable -- and the moment it cannot be kept is
    precisely what the reply's context notice is for."""
    if not ctx:
        return
    # The routing table rebuilds every 30 seconds and these numbers change
    # only when a model set does, so skip the write when nothing moved --
    # otherwise the hub does a pointless SQLite transaction twice a minute
    # forever.
    fresh = {(h or HOST_NAME, m): int(c) for (h, m), c in ctx.items()
             if int(c or 0) > 0}
    if not fresh or fresh == _known_ctx_cache.get("written"):
        return
    ts = now()
    rows = [(h, m, c, ts) for (h, m), c in fresh.items()]
    try:
        with _db_lock, closing(db()) as conn:
            conn.executemany(
                "INSERT INTO model_ctx(host,model,ctx,ts) VALUES(?,?,?,?) "
                "ON CONFLICT(host,model) DO UPDATE SET ctx=excluded.ctx, "
                "ts=excluded.ts",
                rows)
            conn.commit()
    except sqlite3.Error:  # noqa: BLE001 -- telemetry, never worth a 500
        return
    _known_ctx_cache["written"] = fresh
    _known_ctx_cache["t"] = 0.0


def eclipsed_hosts() -> set[str]:
    """The halves of a dual-boot machine that cannot possibly be serving,
    because their counterpart is answering right now.

    collapse_twins() already keeps the fleet page down to one card per
    physical machine. This is the same fact applied to the numbers on it: a
    remembered context ceiling belongs to an OS, not to a chassis, and while
    Windows is up, Fedora's 32k for gemma4-e4b is not a ceiling the fleet can
    honour -- it is a ceiling that needs a reboot. Left in, it was advertised
    against the shared Box alias, so the page said "Box 10 offers 32768, only
    29696 available online" about a box it was simultaneously showing as
    online.

    Only while a twin is actually up. With the laptop switched off entirely
    both halves keep counting, which is the honest reading: either OS could
    be the one that comes back, and `ctx_max` has always meant "awake or
    not". Falls open -- an empty reachability set eclipses nobody -- so a
    cold gateway that has not routed yet behaves exactly as it did before."""
    reachable = _routes_cache.get("reachable") or set()
    if not reachable:
        return set()
    out: set[str] = set()
    for name, spec in load_specs().items():
        twin = str((spec or {}).get("replaces") or "")
        if not twin or twin == name:
            continue
        if name in reachable:
            out.add(twin)
        elif twin in reachable:
            out.add(name)
    return out


def known_model_ctx() -> dict[str, dict[str, int]]:
    """fleet model id -> {host name: context ceiling}, for every box that is
    still part of the fleet -- online or not.

    "Still part of the fleet" is peers.json plus this host, and the filter is
    the point of the function rather than a detail. An OFFLINE peer must keep
    counting: remembering what the big box can do while it is asleep is the
    whole reason this is a table and not a cache. A REMOVED peer must stop
    counting immediately, or its remembered ceiling would advertise a window
    nothing can serve and quietly turn every request for that model into a
    disclosed reduction. Filtering on read means retiring a box needs no
    cleanup step and cannot be forgotten. An ECLIPSED peer -- the dormant half
    of a dual-boot machine whose other half is up -- stops counting too, for
    the same reason a removed one does: see eclipsed_hosts(). So does a
    KILLSWITCHED one: the hub has been told not to route to it, so whatever
    window it remembers is a window this fleet will not serve, and leaving it
    in would advertise a ceiling and then reduce every request down from it."""
    if time.time() - _known_ctx_cache["t"] < 30.0:
        return _known_ctx_cache["map"]
    live = {p["name"] for p in routeable_peers() if p.get("name")} | {HOST_NAME}
    live -= eclipsed_hosts()
    out: dict[str, dict[str, int]] = {}
    for r in db_query("SELECT host, model, ctx FROM model_ctx"):
        host = str(r["host"])
        if host not in live:
            continue
        out.setdefault(str(r["model"]), {})[host] = int(r["ctx"])
    _known_ctx_cache.update(t=time.time(), map=out)
    return out


def host_model_ctx(host: str, fleet_id: str) -> int:
    """The context window `host` will actually serve `fleet_id` with, or 0
    when nothing has ever reported one. `host` is spelled the way the routing
    table spells it: a peer name, or '' for this box."""
    live = _routes_cache.get("ctx", {})
    try:
        c = int(live.get((host, fleet_id)) or 0)
    except (TypeError, ValueError):
        c = 0
    if c:
        return c
    return int(known_model_ctx().get(fleet_id, {}).get(host or HOST_NAME, 0) or 0)


def _ctx_rank(t: tuple[str, str], ctx_limit: int) -> int:
    """Sort key for candidates against the window a key was promised: a box
    PROVEN to hold the whole window first, a box that has never reported a
    ceiling second, a box proven too small last. Three-valued on purpose --
    ranking unknown level with proven-sufficient let a fast-but-unverified box
    shadow the one box that could have answered without any reduction (and an
    unknown box is served unfitted, so a wrong guess there is a SILENT
    truncation, the worst outcome on this path). sorted() is stable, so the
    scorer's ranking survives within each group."""
    c = host_model_ctx(t[0], t[1])
    return 0 if c >= ctx_limit else (1 if c <= 0 else 2)


_OLLAMA_META_KEYS = {"block_count", "head_count_kv", "head_count", "key_length",
                     "value_length", "embedding_length", "context_length",
                     "sliding_window", "key_length_swa", "value_length_swa",
                     "expert_count"}


# Gemma 3 and 4 interleave five sliding-window attention layers per global
# one. Ollama publishes the window but reports `sliding_window_pattern` as
# null, so the ratio has to be assumed; this is the published design, and a
# model with FEWER local layers than this would be over-credited, so the
# estimate below also never exceeds what full attention would allow.
SWA_LOCAL_SHARE = 5.0 / 6.0


def ctx_for_kv_budget(meta: dict, budget: int, ctk: Any, ctv: Any) -> int:
    """The largest context `budget` bytes of KV cache buys for this model.

    Full attention is just budget / bytes-per-token. Sliding-window models are
    the reason this is not a one-liner: Gemma 4 E4B publishes a 512-token
    window and half-width keys on its local layers, so past the first 512
    tokens five of every six layers stop growing and the marginal cost of
    another token collapses to a sixth of the full-attention figure. Treating
    those layers as if they cached everything is what sized a 16 GB Mac down
    to a 2048-token ceiling for a model it can comfortably run at ten times
    that -- and, because the hub enforces what a box reports, that estimate
    would have become a disclosed reduction on a real request."""
    per_tok = kv_bytes_per_token(meta, ctk, ctv)
    if not per_tok or budget <= 0:
        return 0
    full = int(budget / per_tok)
    window = int(meta.get("sliding_window") or 0)
    if window <= 0:
        return full
    local_meta = dict(meta)
    for side in ("key", "value"):
        swa = meta.get(side + "_length_swa")
        if swa:
            local_meta[side + "_length"] = swa
    local_per_tok = (kv_bytes_per_token(local_meta, ctk, ctv) or per_tok) \
        * SWA_LOCAL_SHARE
    global_per_tok = per_tok * (1.0 - SWA_LOCAL_SHARE)
    # Below the window nothing has started sliding yet, so it is still the
    # plain full-attention answer.
    if budget <= (local_per_tok + global_per_tok) * window:
        return full
    remaining = budget - local_per_tok * window
    if global_per_tok <= 0:
        return full
    return max(full, int(remaining / global_per_tok))


def _ollama_kv_meta(info: Any) -> dict:
    """Ollama's /api/show `model_info` carries the same GGUF geometry
    gguf_meta() reads, only namespaced by architecture -- 'qwen3moe.block_count',
    'qwen3moe.attention.head_count_kv'. Flatten it to the plain keys
    kv_bytes_per_token() expects so one KV-sizing function serves both engines."""
    out: dict[str, Any] = {}
    if not isinstance(info, dict):
        return out
    for k, v in info.items():
        if not isinstance(k, str) or "." not in k:
            continue
        tail = k.split(".", 1)[1].replace("attention.", "")
        if tail in _OLLAMA_META_KEYS:
            try:
                out[tail] = int(v)
            except (TypeError, ValueError):
                pass
    if out.get("context_length"):
        out["n_ctx_train"] = out["context_length"]
    return out


def local_model_ctx() -> dict[str, int]:
    """The context window this box will actually launch each of its models
    with, by id and by alias.

    Deliberately not "what could this hardware theoretically hold": it is
    resolve_ctx(), the very number build_cmd() puts after `-c`. A pinned
    record reports what it is pinned to, an auto one reports what its VRAM
    budget works out to. The hub advertises the largest of these across the
    fleet as a model's context ceiling, so it has to be a promise rather than
    an estimate -- a caller who is told 65536 and silently served 8192 is
    worse off than one who was told 8192."""
    out: dict[str, int] = {}
    for rec in load_models():
        if not rec.get("enabled", True) or not rec.get("id"):
            continue
        try:
            ctx = int(resolve_ctx(rec)[0])
        except Exception:  # noqa: BLE001 -- one bad record must not blank the map
            continue
        if ctx <= 0:
            continue
        for name in [str(rec["id"]), *[str(a) for a in rec.get("aliases") or []]]:
            if name.strip():
                out[name.strip()] = ctx
    return out


def local_alias_map() -> dict[str, str]:
    """Every name this box answers to -> the canonical model id behind it.

    An id maps to itself; an alias maps to its record's id. This is what makes
    a role alias checkable across the fleet: `fast` is not a fleet-wide
    concept, it is a row in each box's own models.json, and until every box
    says which weights it means by the word, nothing can notice that two of
    them disagree. See alias_conflicts()."""
    out: dict[str, str] = {}
    for rec in load_models():
        if not rec.get("enabled", True) or not rec.get("id"):
            continue
        mid = str(rec["id"]).strip()
        if not mid:
            continue
        out[mid] = mid
        for a in rec.get("aliases") or []:
            name = str(a).strip()
            if name:
                out[name] = mid
    return out


def model_source(path: Any) -> dict | None:
    """Where a GGUF came from, read back out of where the Library put it:
    MODELS_DIR/<org>__<repo>/<file>.gguf is exactly what start_download()
    writes, so another llama.cpp box can fetch the same file by name."""
    try:
        rel = Path(str(path or "")).resolve().relative_to(MODELS_DIR.resolve())
    except (ValueError, OSError):
        return None
    parts = rel.parts
    if len(parts) >= 2 and "__" in parts[0]:
        return {"repo": parts[0].replace("__", "/", 1), "file": parts[-1]}
    return None


def local_model_meta() -> dict[str, dict]:
    """Per model id (and alias): how big the weights are, whether this box
    holds them in GPU memory or spills them, whether it is a mixture of
    experts, and where the file came from. The hub's host policy is built on
    `fit`; the warm-up button's download path is built on `source`."""
    out: dict[str, dict] = {}
    vram = vram_total_bytes() or 0
    unified = platform.system() == "Darwin"
    for rec in load_models():
        if not rec.get("enabled", True) or not rec.get("id"):
            continue
        path = str(rec.get("path") or "")
        weights = model_bytes(path) + model_bytes(rec.get("mmproj"))
        meta = gguf_meta(path)
        moe: bool | None = bool(meta.get("expert_count")) if meta else None
        try:
            n_cpu_moe = max(0, int(rec.get("n_cpu_moe", 0) or 0))
        except (TypeError, ValueError):
            n_cpu_moe = 0
        if n_cpu_moe > 0:
            fit = "spill"
        elif not vram:
            fit = "cpu"
        elif weights:
            fit = ("vram" if weights + 1024 ** 3 <= int(vram * VRAM_HEADROOM)
                   else "spill")
        else:
            fit = ""
        if unified and fit == "vram":
            fit = "unified"
        m = {"bytes": weights, "fit": fit, "moe": moe, "source": model_source(path)}
        for name in [str(rec["id"]), *[str(a) for a in rec.get("aliases") or []]]:
            if name.strip():
                out[name.strip()] = m
    return out


_upstream_meta_cache: dict[str, Any] = {"meta": {}}


async def served_model_meta() -> dict[str, dict]:
    """local_model_meta() over whatever the engine's own catalogue reports
    (filled beside the context ceilings by _refresh_upstream_ctx)."""
    local = local_model_meta()
    up: dict[str, dict] = {}
    if UPSTREAM_MODELS or not local:
        await upstream_model_ctx()
        up = dict(_upstream_meta_cache["meta"])
    out = dict(up)
    out.update(local)
    # The canonical id an Ollama box answers to for a tag is the same
    # weights, so it carries the tag's metadata -- source included, which is
    # what lets split_models() see through the two names.
    for canon, tag in (await upstream_alias_pairs()).items():
        if tag in out and canon not in out:
            out[canon] = out[tag]
    return out


_upstream_ctx_cache: dict[str, Any] = {"t": 0.0, "ctx": {}}
_upstream_ctx_task: "asyncio.Task | None" = None
UPSTREAM_CTX_TTL = 600.0
UPSTREAM_CTX_RETRY = 60.0


async def upstream_model_ctx() -> dict[str, int]:
    """Per-model context ceilings on a box whose catalogue is the engine's own
    (Ollama), where there is no models.json for resolve_ctx() to read.

    Returns whatever is cached IMMEDIATELY and refreshes behind it, which is
    not an optimisation. The hub polls each peer's /admin/api/served-models on
    a 6 s timeout, and an /api/show per model takes longer than that on a box
    holding six of them -- so building the map inline would make a peer time
    out, and a peer that times out does not just lose its context numbers, it
    drops out of the routing table entirely. An empty first answer costs the
    hub one cycle of reading "unknown", which it already handles by falling
    back to the conservative default."""
    global _upstream_ctx_task
    stale = time.time() - _upstream_ctx_cache["t"] >= UPSTREAM_CTX_TTL
    if stale and (_upstream_ctx_task is None or _upstream_ctx_task.done()):
        # Claim the window before awaiting anything, so a burst of callers
        # spawns one refresh rather than one each.
        _upstream_ctx_cache["t"] = time.time()
        _upstream_ctx_task = asyncio.create_task(_refresh_upstream_ctx())
    return _upstream_ctx_cache["ctx"]


async def _refresh_upstream_ctx() -> None:
    """The body of upstream_model_ctx(), off the request path.

    The geometry comes from /api/show and the memory budget from this box's
    own VRAM -- or its RAM, when the weights are already too big for the card
    and the engine is splitting them anyway."""
    out: dict[str, int] = {}
    tags: list = []
    try:
        assert client is not None
        r = await client.get("/api/tags", timeout=6.0)
        if r.status_code == 200:
            tags = (r.json() or {}).get("models") or []
    except Exception:  # noqa: BLE001 -- not an Ollama box, or it is down
        tags = []
    try:
        ram = int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001
        ram = 0
    vram = vram_total_bytes() or 0
    unified = platform.system() == "Darwin"
    out_meta: dict[str, dict] = {}
    for m in tags:
        if not isinstance(m, dict):
            continue
        mid = str(m.get("model") or m.get("name") or "").strip()
        if not mid:
            continue
        weights = int(m.get("size") or 0)
        info = None
        try:
            assert client is not None
            s = await client.post("/api/show", json={"model": mid}, timeout=8.0)
            if s.status_code == 200:
                info = (s.json() or {}).get("model_info")
        except Exception:  # noqa: BLE001
            info = None
        meta = _ollama_kv_meta(info)
        # The same 90% headroom local_model_meta() applies, so a model sizes
        # to the same tier whichever engine reports it -- otherwise the two
        # OS halves of one dual-boot laptop (Ollama on Windows, llama.cpp on
        # Fedora) would disagree on whether a given file fits.
        if vram and weights and weights + 1024 ** 3 <= int(vram * VRAM_HEADROOM):
            fit = "unified" if unified else "vram"
        else:
            fit = "cpu" if not vram else "spill"
        out_meta[mid] = {"bytes": weights, "fit": fit,
                         "moe": bool(meta.get("expert_count")) if info else None,
                         "source": {"tag": mid}}
        trained = int(meta.get("n_ctx_train") or 0)
        if not kv_bytes_per_token(meta, "f16", "f16"):
            # No usable KV geometry (hybrid state-space models publish none).
            # Guessing large here is how a box OOMs; hold at the same
            # conservative default resolve_ctx() falls back to.
            out[mid] = min(trained or AUTO_CTX_FALLBACK, AUTO_CTX_FALLBACK)
            continue
        # Weights that do not fit the card get split into system RAM, and the
        # KV cache follows them there; budget against whichever pool is
        # actually holding the model.
        pool = vram if (vram and weights and weights + 1024 ** 3 <= vram) else ram
        budget = int(pool * VRAM_HEADROOM) - weights - 1024 ** 3
        ctx = ctx_for_kv_budget(meta, budget, "f16", "f16")
        ctx -= ctx % 1024
        out[mid] = max(2048, min(ctx, min(trained or AUTO_CTX_CAP, AUTO_CTX_CAP)))
    if not out:
        # Nothing to report -- no Ollama here, or it was not up. Come back in a
        # minute rather than in ten: a gateway that started before its engine
        # should not spend the next ten minutes telling the hub it has no
        # models it can size.
        _upstream_ctx_cache.update(
            t=time.time() - (UPSTREAM_CTX_TTL - UPSTREAM_CTX_RETRY), ctx={})
        _upstream_meta_cache["meta"] = {}
        return
    _upstream_ctx_cache.update(t=time.time(), ctx=out)
    _upstream_meta_cache["meta"] = out_meta


async def served_model_ctx() -> dict[str, int]:
    """Every model id this box answers for, mapped to the largest context it
    will serve that model with. models.json wins wherever it has an opinion --
    it is the file that decides what llama-server is actually started with."""
    local = local_model_ctx()
    up: dict[str, int] = {}
    if UPSTREAM_MODELS or not local:
        up = await upstream_model_ctx()
    out = dict(up)
    out.update(local)
    for canon, tag in (await upstream_alias_pairs()).items():
        if tag in out and canon not in out:
            out[canon] = out[tag]
    return out


CTX_REPORT_BUDGET = 8.0


async def _peer_served(p: dict) -> dict:
    """One peer's catalogue, residency and slot counts in one round trip.

    A peer still running an older gateway answers with models only; the
    missing fields simply mean 'unknown', which the scorer treats as
    not-resident and one slot -- the conservative reading."""
    out: dict[str, Any] = {"models": set(), "running": set(), "capacity": {},
                           "ctx": {}, "meta": {}, "engine": "", "online": False,
                           "canonical": {}}
    try:
        async with httpx.AsyncClient(timeout=6.0) as c:
            r = await c.get(
                p["url"].rstrip("/") + "/admin/api/served-models",
                headers={"Authorization": "Bearer " + p.get("token", "")},
            )
            if r.status_code != 200:
                return out
            # Answered. Worth recording separately from "served us a model":
            # a box that is up but serving nothing is still up, and that is
            # exactly the state eclipsed_hosts() has to be able to see.
            out["online"] = True
            d = r.json()
            out["models"] = {str(m).strip() for m in d.get("models") or []
                             if str(m).strip()}
            out["running"] = {str(m).strip() for m in d.get("running") or []
                              if str(m).strip()}
            # Absent on an older gateway, which reads as "no job of its own".
            out["warm"] = {str(m).strip() for m in d.get("warm") or []
                           if str(m).strip()}
            can = d.get("canonical") or {}
            if isinstance(can, dict):
                out["canonical"] = {str(k).strip(): str(v).strip()
                                    for k, v in can.items()
                                    if str(k).strip() and str(v).strip()}
            caps = d.get("capacity") or {}
            if isinstance(caps, dict):
                out["capacity"] = {str(k): max(1, int(v or 1))
                                   for k, v in caps.items()}
            cx = d.get("ctx") or {}
            if isinstance(cx, dict):
                for k, v in cx.items():
                    try:
                        n = int(v)
                    except (TypeError, ValueError):
                        continue
                    if n > 0:
                        out["ctx"][str(k)] = n
            # Size / fit / MoE / source per model, and the engine kind -- both
            # new; a peer on an older gateway simply has neither, and the
            # policy reads "unknown" for it rather than guessing.
            mt = d.get("meta") or {}
            if isinstance(mt, dict):
                out["meta"] = {str(k): v for k, v in mt.items() if isinstance(v, dict)}
            out["engine"] = str(d.get("engine") or "")
    except Exception:  # noqa: BLE001 -- an unreachable peer just routes nowhere
        pass
    return out


async def model_routes(force: bool = False) -> dict[str, str]:
    """model id (or alias) -> peer name, with '' meaning this host."""
    if not force and time.time() - _routes_cache["t"] < ROUTE_TTL:
        return _routes_cache["map"]
    routes: dict[str, str] = {}
    cands: dict[str, list[str]] = {}
    cap: dict[tuple[str, str], int] = {}
    ctx: dict[tuple[str, str], int] = {}
    meta: dict[tuple[str, str], dict] = {}
    alias: dict[tuple[str, str], str] = {}
    engine: dict[str, str] = {}
    running: dict[str, set[str]] = {}
    local_caps = local_capacity()
    # Telemetry must never stall routing. served_model_ctx() reads local files
    # and, on an Ollama box, may touch the engine; a hung engine would
    # otherwise hold up every request waiting on a routing refresh.
    try:
        local_ctx = await asyncio.wait_for(served_model_ctx(), CTX_REPORT_BUDGET)
    except Exception:  # noqa: BLE001
        local_ctx = {}
    try:
        local_meta = await asyncio.wait_for(served_model_meta(), CTX_REPORT_BUDGET)
    except Exception:  # noqa: BLE001
        local_meta = {}
    local_alias = local_alias_map()
    for m in await served_model_ids():
        routes[m] = ""
        cands[m] = [""]
        cap[("", m)] = local_caps.get(m, 1)
        if local_ctx.get(m):
            ctx[("", m)] = int(local_ctx[m])
        if isinstance(local_meta.get(m), dict):
            meta[("", m)] = local_meta[m]
        if local_alias.get(m):
            alias[("", m)] = local_alias[m]
    running[HOST_NAME] = set(await upstream_running_ids())
    warm: dict[str, set[str]] = {HOST_NAME: local_warm_ids()}
    try:
        engine[HOST_NAME] = str(engine_info().get("kind") or "")
    except Exception:  # noqa: BLE001
        engine[HOST_NAME] = ""
    reachable: set[str] = {HOST_NAME}
    # The killswitch: a peer whose routed flag is off stays registered and
    # visible on the fleet page, but is NOT a candidate for any route. The
    # filter is why the toggle has teeth -- without it, a killed box would
    # just keep answering the _peer_served() probe and stay in the table.
    peers = routeable_peers()
    if peers:
        served = await asyncio.gather(*(_peer_served(p) for p in peers))
        for p, d in zip(peers, served):
            running[p["name"]] = d["running"]
            warm[p["name"]] = set(d.get("warm") or set())
            if d.get("online"):
                reachable.add(p["name"])
            if d.get("engine"):
                engine[p["name"]] = str(d["engine"])
            for m in d["models"]:
                routes.setdefault(m, p["name"])  # local always wins
                cands.setdefault(m, []).append(p["name"])
                cap[(p["name"], m)] = d["capacity"].get(m, 1)
                if d["ctx"].get(m):
                    ctx[(p["name"], m)] = int(d["ctx"][m])
                if isinstance(d.get("meta", {}).get(m), dict):
                    meta[(p["name"], m)] = d["meta"][m]
                if d.get("canonical", {}).get(m):
                    alias[(p["name"], m)] = d["canonical"][m]
    _routes_cache.update(t=time.time(), map=routes, cands=cands,
                         cap=cap, running=running, ctx=ctx,
                         reachable=reachable, meta=meta, engine=engine,
                         warm=warm, alias=alias)
    remember_model_ctx(ctx)
    return routes


def _host_saturated(cand: str, fid: str) -> bool:
    """True when (cand, fid) has no free decode slot right now -- the exact
    predicate _score_host_model_pairs's first sort key computes, factored
    out so demo_host_policy and pick_fallback's busy trigger read the same
    definition of "occupied" the scorer does, rather than each growing its
    own slightly different copy."""
    hname = cand or HOST_NAME
    slots = max(1, int(_routes_cache.get("cap", {}).get((cand, fid), 1)))
    return _inflight.get(hname, 0) >= slots


async def _score_host_model_pairs(
    pairs: list[tuple[str, str]], role: str = "primary",
    prompt_tokens: int = 0, gen_tokens: int = 256, need_ctx: int = 0,
    fleet_role: str = "",
) -> list[tuple[str, str]]:
    """Rank (host, model-id) candidates, best first.

    In order: a box with a free decode slot beats a saturated one; a box
    that has not just failed a request beats one that has (host_cooling);
    then the owner's host policy -- host_tier(), the fast cards that hold
    the model ahead of the big box ahead of the always-on small boxes ahead
    of the CPU backstop; and within a tier the box expected to ANSWER
    soonest (_est_wall: cold load, prompt read, generation, scaled by how
    loaded it already is), local ties broken toward this host. When EVERY
    candidate is saturated, the tier/rank policy only breaks a tie AFTER
    queue depth (busy - slots, zero for anyone with a free slot so this
    never touches the free-candidate ordering above): the box with fewer
    jobs already waiting goes first, and only a tie in that goes to the
    owner's more capable machine -- not the other way round.

    Residency used to be the first key, which is how a 42k-token prompt was
    sent to an 8 GB laptop that happened to have the model warm while a
    much faster box sat idle. It is now one term of the wall-time estimate:
    a cold load still costs, but it no longer trumps everything.

    `need_ctx` is the window this request actually needs (prompt plus
    completion); a box known to serve the model with less than that goes
    last. Factored out of model_hosts() so a Fleet Pass public catalogue
    row -- which can name several fleet ids as the same public model -- can
    rank its candidates across ALL of them at once. Requires `_routes_cache`
    to already be fresh (the caller awaits model_routes() first).

    `fleet_role` names the FLEET_ROLES ladder the pairs were built from, and
    it changes what "best" means: among the boxes that hold their model
    (tier 0 or 1), the one offering the model HIGHER on the ladder wins, and
    only then does the usual order apply. Without it `deep` went to a warm
    27B on a laptop while the 120B sat idle on the big box -- the policy
    said which model plays the role best, and the ranking ignored it. Boxes
    that would spill or run it on CPU still sit behind every box that holds
    it, whatever the ladder says."""
    if len(pairs) <= 1:
        return pairs
    cap = _routes_cache.get("cap", {})
    running = _routes_cache.get("running", {})
    specs = load_specs()
    tps_cache: dict[str, dict[str, float]] = {}
    pp_cache: dict[str, dict[str, float]] = {}

    async def tps_for(mid: str) -> dict[str, float]:
        if mid not in tps_cache:
            tps_cache[mid] = await asyncio.to_thread(measured_tps, mid)
        return tps_cache[mid]

    async def pp_for(mid: str) -> dict[str, float]:
        if mid not in pp_cache:
            pp_cache[mid] = await asyncio.to_thread(measured_pp, mid)
        return pp_cache[mid]

    scored = []
    for cand, mid in pairs:
        hname = cand or HOST_NAME
        slots = max(1, int(cap.get((cand, mid), 1)))
        busy = _inflight.get(hname, 0)
        # 0 for any candidate with a free slot, so this never disturbs
        # today's capability-first ordering among FREE candidates -- it only
        # ever breaks a tie once every candidate is already saturated
        # (element 1 of the key below), which is the one case the owner's
        # tier/rank ranked ahead of load before this existed.
        queue_len = max(0, busy - slots)
        resident = mid in running.get(hname, set())
        tier, rank = host_tier(cand, mid, role)
        too_small = 0
        if need_ctx and 0 < host_model_ctx(cand, mid) < need_ctx:
            tier, too_small = 5, 1
        est = _est_wall(cand, mid, resident, await tps_for(mid), await pp_for(mid),
                        prompt_tokens, gen_tokens)
        est *= 1.0 + busy / slots
        try:
            bw = float(specs.get(hname, {}).get("mem_bw_gbs") or 0)
        except (TypeError, ValueError):
            bw = 0.0
        # For a fleet role: holds-it before spills-it, then the ladder.
        pref = ((0 if tier <= 1 else 1, role_index(fleet_role, cand, mid))
                if fleet_role else ())
        key = (
            1 if busy >= slots else 0,          # saturated last
            1 if host_cooling(hname) else 0,    # just failed: last
            too_small,                          # proven cannot fit: behind
                                                # even the reserve boxes -- a
                                                # personal box that CAN answer
                                                # beats any box that cannot
            1 if host_reserved(hname) else 0,   # personal box: only when the
                                                # fleet boxes are busy/failing
            *pref,                              # the role's own order
            queue_len,                          # among saturated boxes only:
                                                # fewest jobs waiting first
            tier, rank,                         # the owner's policy
            round(est, 1),                      # soonest answer first
            -bw,                                # spec-sheet tiebreak
            0 if cand == "" else 1,             # local tiebreak
        )
        scored.append((key, cand, mid))
    scored.sort(key=lambda t: t[0])
    return [(cand, mid) for _, cand, mid in scored]


def alias_conflicts() -> list[dict]:
    """Names that mean different weights on different boxes.

    The scorer in _score_host_model_pairs() answers "which of these hosts
    should serve this name". It assumes the candidates are interchangeable --
    which is true for a real model id, and is an assumption for an alias. A
    role alias like `fast`, `triage` or `deep` is a row in each box's own
    models.json, and nothing has ever checked that two boxes mean the same
    thing by the word. If apu-box-1's `fast` is qwen3.6-35b and mac-desktop-1's `fast`
    is qwen3.5:9b, both are candidates, both get ranked honestly against
    their own metadata, and the caller gets whichever box was quicker today.
    That is not routing to the best machine for the job; it is routing to the
    best machine for a coin flip.

    Reported rather than repaired, deliberately. Only the fleet's owner knows
    which meaning is the right one, and a gateway that silently dropped half
    the candidates would turn a naming mistake into a capacity mystery.

    Requires a fresh `_routes_cache` (callers await model_routes() first). A
    peer on a gateway too old to report `canonical` contributes nothing and is
    listed under `unknown`, so "no conflicts" never quietly means "could not
    check"."""
    cands = _routes_cache.get("cands", {})
    alias = _routes_cache.get("alias", {})
    out = []
    for name, hosts in sorted(cands.items()):
        if len(hosts) < 2:
            continue
        # A fleet role is answered by policy before any box's row of that
        # name is consulted, so boxes disagreeing about it changes nothing
        # a request can see. The roles endpoint lists those rows instead.
        if name in FLEET_ROLES:
            continue
        by_target: dict[str, list[str]] = {}
        unknown = []
        for h in hosts:
            target = alias.get((h, name))
            if not target:
                unknown.append(h or HOST_NAME)
            else:
                by_target.setdefault(target, []).append(h or HOST_NAME)
        if len(by_target) > 1:
            out.append({
                "name": name,
                "targets": {t: sorted(hs) for t, hs in sorted(by_target.items())},
                "unknown": sorted(unknown),
                # An alias that IS a real model id somewhere is the less
                # alarming shape: one box is serving the model, another is
                # pointing the same word at something else.
                "is_model_id": name in by_target,
            })
    return out


def split_models() -> list[dict]:
    """One model, served under different names on different boxes.

    The mirror image of alias_conflicts(), and the more common failure of the
    two. A conflict at least routes somewhere; this does not route at all --
    candidates are grouped by NAME, so two boxes holding the same weights
    under different names are never each other's alternatives. The better box
    is simply invisible for that request, silently, and the symptom is a
    fleet that looks half its size.

    Measured on this fleet: gpu-laptop-1 served the 9B distill as `qwen3.8-9B`
    while apu-tablet-2 served the same file as `qwen3.8-9b-distill`, which is the
    id the hub publishes -- so the always-on box that was supposed to hold it
    warm was never a candidate for it.

    Identity is the HuggingFace REPO the weights came from -- the Library's
    {repo, file} on a llama.cpp box, the `hf.co/...` tag on an Ollama box
    (_model_repo) -- and not the file: the fleet already serves one model at
    different quants under one id (qwen3.8-27b is Q5_K_M on apu-box-1 and
    UD-Q5_K_M on mac-laptop-1), so a Q3 and a Q4 of one repo under two names are
    two names, not two models. A box whose weights came from neither source
    reports no repo and is skipped rather than guessed at.

    Grouped by canonical id, and reported only when some box that holds the
    weights does not answer to every name for them. Two ids for one file on
    ONE box is a variant the owner built on purpose (the classify build
    beside the general one on apu-box-1), and both are reachable wherever the
    weights are; the point of this check is boxes that cannot stand in for
    each other.

    The fleet-wide spellings this already knows about are repaired rather
    than reported: FLEET_MODEL_NAMES renames them at start-up on a llama.cpp
    box and aliases them on an Ollama one. What is left here is a spelling
    nobody has claimed yet.
    """
    cands = _routes_cache.get("cands", {})
    meta = _routes_cache.get("meta", {})
    alias = _routes_cache.get("alias", {})
    # repo -> {canonical id -> {hosts}}
    seen: dict[str, dict[str, set[str]]] = {}
    for name, hosts in cands.items():
        for h in hosts:
            repo = _model_repo((meta.get((h, name)) or {}).get("source") or {})
            if not repo:
                continue
            # Group by the canonical id, not the name: two boxes that both
            # call it `fast` on top of the same real id agree, and an alias
            # sitting beside its own id on one box is not a split.
            mid = alias.get((h, name)) or name
            seen.setdefault(repo, {}).setdefault(mid, set()).add(h or HOST_NAME)
    out = []
    for repo, by_id in sorted(seen.items()):
        if len(by_id) < 2:
            continue
        everywhere = set().union(*by_id.values())
        if all(hs == everywhere for hs in by_id.values()):
            continue
        out.append({
            "repo": repo,
            "names": {mid: sorted(hs) for mid, hs in sorted(by_id.items())},
        })
    return out


def role_index(role: str, cand: str, name: str) -> int:
    """Where the model a box serves as `name` sits on the role's ladder --
    0 for the best -- and one past the end when it is not on it."""
    ladder = FLEET_ROLES.get(role, ())
    canon = _routes_cache.get("alias", {}).get((cand, name)) or name
    canon = SPELLING_TO_CANONICAL.get(canon, canon)
    return ladder.index(canon) if canon in ladder else len(ladder)


def role_pairs(role: str) -> list[tuple[str, str]]:
    """(host, fleet id) for a fleet role: the best model for it ON EACH BOX.

    Per box, the first ladder entry it holds in its own memory
    (preload_capable's reading of `fit`, moe_spill_ok included), else the
    first it serves at all -- a CPU box still plays `fast` with the 3B-active
    MoE its owner would have picked. One pair per box, spelled the way that
    box serves the model: a peer that still knows the weights only by an old
    spelling is addressed by it, since the proxy sends the name the box
    answers to. The caller ranks the pairs with _score_host_model_pairs like
    any other candidates. Requires a fresh _routes_cache."""
    ladder = FLEET_ROLES.get(role)
    if not ladder:
        return []
    cands = _routes_cache.get("cands", {})
    alias = _routes_cache.get("alias", {})
    running = _routes_cache.get("running", {})
    held: dict[str, dict[str, str]] = {}   # host -> canonical id -> served name
    for name, hosts in cands.items():
        for h in hosts:
            canon = alias.get((h, name)) or name
            canon = SPELLING_TO_CANONICAL.get(canon, canon)
            if canon not in ladder:
                continue
            names = held.setdefault(h, {})
            cur = names.get(canon)
            hot = running.get(h or HOST_NAME, set())
            # The canonical spelling wins when the box serves it -- unless
            # another name for the same weights is the one actually loaded.
            # A box that kept an old spelling as a record of its own (the
            # start-up rename steps aside when the canonical id is taken by
            # a variant) has two names for one file and only one of them
            # warm, and sending the cold one would pay a reload for nothing.
            if cur is None or (name in hot and cur not in hot):
                names[canon] = name
            elif cur not in hot and name == canon:
                names[canon] = name
    pairs: list[tuple[str, str]] = []
    for h, names in held.items():
        fits = [c for c in ladder if c in names and preload_capable(h, names[c])]
        pick = fits[0] if fits else next(c for c in ladder if c in names)
        pairs.append((h, names[pick]))
    return pairs


def _model_repo(src: dict) -> str:
    """The HuggingFace repo a served model's weights came from, read from
    either engine's idea of a source: the Library's {repo, file} on a
    llama.cpp box, or an `hf.co/<org>/<repo>:<quant>` tag on an Ollama box.
    Anything else -- an Ollama library tag like `gemma4:26b`, a hand-placed
    file -- is '' and is left out rather than guessed at."""
    repo = str(src.get("repo") or "").strip()
    if repo:
        return repo
    tag = str(src.get("tag") or "").strip()
    if tag.startswith("hf.co/"):
        body = tag[len("hf.co/"):].split(":", 1)[0]
        parts = body.split("/")
        if len(parts) == 2 and all(parts):
            return body
    return ""


async def model_hosts(model: str, **kw: Any) -> list[str]:
    """Every host serving `model`, best first ('' meaning this host).

    The order is advice, not a commitment -- the proxy walks the list and
    fails over on connect errors, so a stale routing table costs a retry,
    not a 502. See _score_host_model_pairs() for how "best" is decided.
    """
    await model_routes()
    cands = list(_routes_cache["cands"].get(model, []))
    if len(cands) <= 1:
        return cands
    ranked = await _score_host_model_pairs([(c, model) for c in cands], **kw)
    return [cand for cand, _ in ranked]


async def peer_inference_key(peer_name: str) -> str | None:
    """Fetch (minting on first use) the hub's own API key on a peer."""
    peers = load_peers()
    p = next((x for x in peers if x["name"] == peer_name), None)
    if not p:
        return None
    if p.get("inference_key"):
        return str(p["inference_key"])
    try:
        async with httpx.AsyncClient(timeout=20.0) as c:
            r = await c.post(
                p["url"].rstrip("/") + "/admin/api/keys",
                headers={"Authorization": "Bearer " + p.get("token", "")},
                json={"name": "hub-" + HOST_NAME},
            )
            r.raise_for_status()
            key = str(r.json().get("key", ""))
    except Exception:  # noqa: BLE001
        return None
    if not key:
        return None
    for x in peers:
        if x["name"] == peer_name:
            x["inference_key"] = key
    save_peers(peers)
    return key


async def fleet_model_list() -> list[dict]:
    """Union of every model the fleet serves, tagged with its host.

    One entry per MODEL, not per name: a peer's alias (`fast`, `qwen3.8-9B`)
    is still accepted in a request, it just is not listed beside the id it
    stands for. A client that lists models and then asks for one of them
    gets the same answer either way; what it no longer sees is the same
    weights three times under three spellings."""
    routes = await model_routes()
    peer_can = _routes_cache.get("alias", {})
    canonical: dict[str, str] = {}
    for rec in load_models():
        if rec.get("enabled", True) and rec.get("id"):
            canonical[str(rec["id"])] = ""
    # On an Ollama box the canonical id is listed and the tag stands behind it.
    by_tag = {tag: canon for canon, tag in (await upstream_alias_pairs()).items()}
    for mid in await upstream_model_ids():
        canonical.setdefault(by_tag.get(mid, mid), "")
    for m, host in routes.items():
        # A role word is never listed on a peer's say-so: a peer too old to
        # report what its `fast` row means is not something resolve_targets()
        # would route to, and a listing must not promise what a request
        # cannot get. Roles are listed below, from what can really play them.
        if (host and m not in canonical and m not in FLEET_ROLES
                and peer_can.get((host, m), m) == m):
            canonical.setdefault(m, host)
    seen, out = set(), []
    for mid, host in canonical.items():
        if mid in seen:
            continue
        seen.add(mid)
        out.append(
            {
                "id": mid,
                "object": "model",
                "owned_by": host or HOST_NAME,
                "created": 0,
            }
        )
    # A fleet role is callable like a model and listed like one, as long as
    # something in the fleet can play it right now.
    for role in FLEET_ROLES:
        if role not in seen and role_pairs(role):
            seen.add(role)
            out.append({"id": role, "object": "model", "owned_by": "fleet-role",
                        "created": 0})
    return sorted(out, key=lambda m: m["id"])


def record_usage(
    key: dict,
    model: str,
    endpoint: str,
    stream: bool,
    status: int,
    usage: dict | None,
    ttft_ms: int | None,
    latency_ms: int,
    host: str = "",
    fallback_from: str = "",
) -> None:
    usage = usage or {}
    try:
        db_exec(
            "INSERT INTO usage(ts,key_id,key_name,model,endpoint,stream,status,"
            "prompt_tokens,completion_tokens,total_tokens,ttft_ms,latency_ms,"
            "host,fallback_from) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                now(),
                key.get("id"),
                key.get("name"),
                model,
                endpoint,
                1 if stream else 0,
                status,
                int(usage.get("prompt_tokens", 0) or 0),
                int(usage.get("completion_tokens", 0) or 0),
                int(usage.get("total_tokens", 0) or 0),
                ttft_ms,
                latency_ms,
                host or None,
                fallback_from or None,
            ),
        )
    except Exception:  # noqa: BLE001 -- metering must never break inference
        pass


@app.api_route(
    "/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
)
async def openai_proxy(path: str, request: Request):
    key = require_api_key(request)
    assert client is not None
    endpoint = "/v1/" + path
    started = time.time()

    # Refuse rather than start. uvicorn stops ACCEPTING on a shutdown signal,
    # but a client already holding a keep-alive connection can still send the
    # next request down it, and a completion begun here has seconds to live --
    # it would be drained before its first token. A 503 with Retry-After is
    # something every OpenAI SDK already knows how to back off on; half a
    # stream is not.
    if draining():
        record_usage(key, "", endpoint, False, 503, None, None,
                     int((time.time() - started) * 1000))
        return JSONResponse(DRAINED_BODY, status_code=503,
                            headers={"retry-after": DRAIN_RETRY_AFTER})

    # A key that has ever had a public_keys row reaches exactly two routes
    # (contract 1.9h) -- a 5/day Fleet Pass key must not be able to spool a
    # 100k-request batch, or hit /v1/embeddings, /v1/completions, etc. This
    # runs before the /v1/batches dispatch below and before the body is even
    # read, so a public key never gets far enough to touch that surface.
    pub_row = public_key_row(int(key["id"]))
    if pub_row and not public_surface_allowed(endpoint, request.method):
        record_usage(key, "", endpoint, False, 403, None, None,
                     int((time.time() - started) * 1000))
        return JSONResponse(
            {"error": {"message": "this key is limited to POST /v1/chat/completions "
                                  "and GET /v1/models", "type": "permission_error"}},
            status_code=403,
        )

    # The batch surface lives under /v1 so ordinary bearer keys work, but this
    # catch-all was registered first, so the dispatch happens here rather than
    # in a route FastAPI would never reach.
    if path == "batches" or path.startswith("batches/"):
        return await v1_batches(path, request, key)

    raw = await request.body()
    model = ""
    stream = False
    body = raw
    # Hoisted out of the parse block below: the per-candidate routing loop has
    # to know this key's context budget to decide whether the box it is about
    # to ask can actually honour it.
    payload: Any = {}
    ctx_limit: int | None = None
    # Read off the body as it ARRIVED, before anything below rewrites the
    # history: whether the caller has already had a reply in this conversation
    # decides whether a notice may be written in front of this one.
    continued = False
    if raw:
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                # A team key turns this endpoint into an orchestrated crew:
                # the primary model leads, and the gateway executes any
                # spawn_subagents calls across the fleet before answering.
                if path == "chat/completions" and request.method == "POST":
                    team = get_team(int(key["id"]))
                    if team:
                        return await team_orchestrate(
                            key, team, payload,
                            bool(payload.get("stream", False)), started,
                            request=request)
                continued = already_replied(payload.get("messages"))
                mutated = False
                if pub_row:
                    # A free demo key does not get to multiply its own request
                    # budget: pin n to 1, and drop the sampler knobs that could
                    # otherwise turn one call into many completions' worth of
                    # compute (best_of) or a much larger response body (logprobs).
                    if payload.get("n") != 1:
                        payload["n"] = 1
                        mutated = True
                    for k in ("best_of", "logprobs"):
                        if k in payload:
                            del payload[k]
                            mutated = True
                agent = get_agent(int(key["id"]))
                # Only keys the gateway ever DECORATES get their history
                # cleaned: on anything else the body must pass through
                # byte-identical.
                if ((pub_row or agent)
                        and strip_fleet_notices(payload.get("messages"))):
                    mutated = True
                if agent:
                    payload = apply_agent(agent, payload)
                    mutated = True
                    ctx_limit = _norm_limit(agent.get("ctx_limit"))
                    if ctx_limit:
                        try:
                            payload = apply_ctx_limit(payload, ctx_limit)
                        except HTTPException as exc:
                            # Metered, and answered in the flat OpenAI error
                            # shape -- letting this propagate handed the
                            # client a framework-wrapped {"detail": ...} and
                            # skipped record_usage entirely.
                            record_usage(
                                key, str(payload.get("model", "")), endpoint,
                                bool(payload.get("stream")), 413, None, None,
                                int((time.time() - started) * 1000))
                            return JSONResponse(
                                exc.detail if isinstance(exc.detail, dict)
                                else {"error": {"message": str(exc.detail),
                                                "type": "context_limit"}},
                                status_code=413)
                model = str(payload.get("model", ""))
                stream = bool(payload.get("stream", False))
                if stream and INJECT_USAGE:
                    so = payload.get("stream_options")
                    if not isinstance(so, dict):
                        so = {}
                    so["include_usage"] = True
                    payload["stream_options"] = so
                    mutated = True
                if mutated:
                    body = json.dumps(payload).encode()
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    # Set by a hub forwarding a request it has already listed as active, so
    # the fleet view shows one job rather than the same work twice. Dropped
    # from the forwarded headers and re-set per hop below: a client could send
    # it, and it would only ever confuse the display.
    origin = request.headers.get("x-fleet-origin", "")
    hop = {
        "host",
        "content-length",
        "connection",
        "authorization",
        "x-api-key",
        "cf-access-jwt-assertion",
        "cookie",
        "x-fleet-origin",
    }
    fwd = {k: v for k, v in request.headers.items() if k.lower() not in hop}
    if body:
        fwd["content-length"] = str(len(body))

    # /v1/models is answered by the hub itself: a client asking what it can use
    # should see the whole fleet, not just this box's catalogue -- except a
    # Fleet Pass key, which only ever gets to see what it may actually call.
    #
    # Not metered. A listing is not usage: nothing was generated, no box did
    # any work, and every budget query already excluded these rows
    # (BUDGET_REQ_SQL). What recording them did was fill the ledger -- the
    # downstream-app autopilot probes the hub's catalogue every 30 s as a health
    # check, and by 2026-08-30 that key had 26,562 usage rows on this hub, all
    # of them this listing, zero completions -- so a project's usage page read
    # as busy and the actual completions were needles in it. The key's
    # last_used_at is still stamped by require_api_key, so "when was this key
    # last seen" keeps its meaning.
    if path == "models" and request.method == "GET":
        pub_key = public_key_for(int(key["id"]))
        if pub_key:
            try:
                pub_models = json.loads(pub_key.get("models") or "{}")
            except (TypeError, json.JSONDecodeError):
                pub_models = {}
            # context_length is the one field a client can read the key's
            # window from. Without it, a coding agent falls back to whatever
            # its provider profile last held -- a stale figure from a box
            # retired months ago survives every new key, because nothing the
            # gateway sends ever contradicts it. The value is the key's own
            # enforced ceiling (agents.ctx_limit == public_keys.ctx), not a
            # per-box serving figure: it is the number apply_ctx_limit()
            # holds every request to, whichever box answers.
            pub_ctx = int(pub_key.get("ctx") or 0)
            if pub_key["kind"] == "single":
                mid = str(pub_models.get("model") or "")
                data = ([{"id": mid, "object": "model", "owned_by": "fleet-pass",
                         "created": 0}] if mid else [])
            else:
                data = [{"id": "team", "object": "model", "owned_by": "fleet-pass",
                        "created": 0}]
                primary = str(pub_models.get("primary") or "")
                if primary:
                    data.append({"id": primary, "object": "model",
                                "owned_by": "fleet-pass", "created": 0})
            if pub_ctx > 0:
                for entry in data:
                    entry["context_length"] = pub_ctx
            return JSONResponse({"object": "list", "data": data})
        fleet_list = await fleet_model_list()
        seen = {m["id"] for m in fleet_list}
        for pid, row in public_catalogue()["by_public"].items():
            if not row.get("enabled") or pid in seen:
                continue
            # A public id is one more name for weights the fleet already
            # lists under their own id (`gemma4-26b-a4b` is `gemma-4-26b`).
            # It stays callable; it is only listed when it is the sole name.
            if any(f in seen for f in _row_fleet_ids(row)):
                continue
            fleet_list.append({"id": pid, "object": "model",
                               "owned_by": "fleet-pass", "created": 0})
            seen.add(pid)
        return JSONResponse({"object": "list",
                            "data": sorted(fleet_list, key=lambda m: m["id"])})

    # Route by model. Several boxes can serve the same id, so resolve every
    # (host, fleet id) pair that does and try them in order -- one unreachable
    # box should cost a retry, not the request. `model` is left holding
    # whatever the client actually asked for, for record_usage and the
    # fallback notice; `served_model` is what routing is resolved against,
    # which differs from `model` only when a Fleet Pass fallback substitutes
    # a different catalogue row.
    targets: list[tuple[str, str]] = [("", model)]
    fallback: dict[str, str] | None = None
    served_model = model
    # What the chosen box is about to be asked to do, for the scorer: how
    # much prompt it must read and roughly how much it will generate.
    prompt_est = estimate_prompt_tokens(payload)
    gen_est = 256
    if isinstance(payload, dict):
        try:
            gen_est = min(1024, int(payload.get("max_tokens") or 0)) or 256
        except (TypeError, ValueError):
            gen_est = 256
    if model:
        pub_key = public_key_for(int(key["id"]))
        # Substitution (requirement 2) is fleet-wide, not a Fleet Pass perk --
        # every bearer key asking for a catalogue model, a fleet id, or a
        # role name is eligible; only the *presentation* (Box-N alias below)
        # stays conditional on pub_row. The opt-out is checked once, before
        # the one pick_fallback call any caller kind on this path makes.
        if not _no_fallback_requested(request, payload if isinstance(payload, dict) else None):
            req_row = _catalogue_row_for(model)
            if req_row:
                sub = await pick_fallback(
                    req_row, "single" if pub_key else "primary", get_public_settings())
                if sub:
                    served_model = str(sub["public_id"])
                    fallback = {"requested": model, "served": served_model}
        try:
            targets = await resolve_targets(
                served_model, role="primary", prompt_tokens=prompt_est,
                gen_tokens=gen_est, need_ctx=prompt_est + gen_est)
        except Exception:  # noqa: BLE001 -- routing is best-effort
            targets = []
        if ctx_limit and len(targets) > 1:
            # A box that can hold the whole window this key was issued for
            # beats one that would force a disclosed reduction, and it beats it
            # ahead of residency: a cold load costs the caller a wait, a
            # shrunken context costs them part of their prompt. Not gated on
            # pub_row: any key an admin gave a ctx_limit deserves the same
            # preference. Ranking is _ctx_rank's proven / unknown / too-small
            # order -- see its docstring for why unknown no longer ties with
            # proven.
            targets = sorted(targets, key=lambda t: _ctx_rank(t, ctx_limit))
        if not targets:
            # Nothing in the fleet answers to this id. Falling through to the
            # local upstream here is what made a typo'd model name look like a
            # dead backend; name the real problem and list the alternatives.
            record_usage(key, model, endpoint, stream, 404, None, None,
                         int((time.time() - started) * 1000))
            return JSONResponse(
                {
                    "error": {
                        "message": "no host in the fleet serves model '"
                        + served_model + "'",
                        "type": "invalid_request_error",
                        "param": "model",
                        "available": [m["id"] for m in await fleet_model_list()],
                    }
                },
                status_code=404,
            )

    # Listed as active work from here on. From the moment a candidate accepts
    # the request it owns a decode slot somewhere, and that slot is what an
    # abort has to be able to reach.
    job = _job_open(
        "inference",
        what=model or path,
        detail=endpoint + (" · stream" if stream else "")
        + (" · " + str(key.get("name")) if key.get("name") else ""),
        origin=origin,
        stop=asyncio.Event(),
    )
    resp = None
    peer_client: httpx.AsyncClient | None = None
    target_peer = ""
    # (host, note) per box that did not take the request, for the 502.
    tried: list[tuple[str, str]] = []
    # Set when the box that answers cannot give this key the context window it
    # was issued for; carried out of the loop so the reply can say so.
    ctx_cut: dict[str, int] | None = None
    ctx_reject: HTTPException | None = None
    # Did any box actually fail, as opposed to declining the prompt on length?
    # A 413 says "your prompt is too long", which is only the honest answer
    # when EVERY candidate said so. If a box that could have held it broke
    # instead, the fleet failed, and a later candidate's context rejection
    # must not be allowed to blame the caller for it.
    upstream_failed = False
    # The in-flight count for the box being asked, held from the moment it is
    # asked -- not once it answers. During a cold load every concurrent
    # request used to see that box as idle and pile onto it.
    untrack: Any = None
    # A caller that hangs up while the box is still loading used to leave the
    # box loading for nobody; this notices and lets go.
    watcher = _watch_disconnect(request, job)
    # On an Ollama box the fleet's canonical id is answered to for a tag the
    # engine only knows by the tag; empty everywhere else.
    up_alias = await upstream_alias_pairs()
    try:
        for i, (cand, fleet_id) in enumerate(targets):
            more = i < len(targets) - 1
            hname = cand or HOST_NAME
            # Attributed to the box we are asking from the moment we ask it,
            # not once it answers: waiting for a peer to load a model is
            # precisely when someone looks at this list, and "hubbox" would
            # be the wrong name to find there -- and the wrong one for the
            # fleet view to merge on.
            job["host"] = hname
            send_client = client
            peer_client = None
            # Per-attempt copy: mutating fwd would leak one peer's inference
            # key into the next attempt, including a fall back to the local
            # upstream.
            hdrs = dict(fwd)
            # What this particular box can give the request. A ceiling of 0
            # means it has never reported one, which reads as "unknown"
            # rather than "none" -- a peer running an older gateway is served
            # exactly as it always was, with no invented reduction.
            ctx_cut = None
            cand_payload: dict | None = None
            if ctx_limit and isinstance(payload, dict):
                cand_ctx = host_model_ctx(cand, fleet_id)
                if 0 < cand_ctx < ctx_limit:
                    try:
                        cand_payload = apply_ctx_limit(dict(payload), cand_ctx)
                    except HTTPException as exc:
                        # The prompt does not fit on this box at all. A later
                        # candidate may still have room for it, so keep the
                        # reason and try the next one rather than failing here.
                        ctx_reject = exc
                        tried.append((hname, " (context)"))
                        continue
                    ctx_cut = {"requested": int(ctx_limit), "granted": int(cand_ctx)}
            # A public catalogue id (or a Fleet Pass fallback substitution) is
            # never what the upstream knows the model as -- rewrite the body's
            # `model` to this attempt's real fleet id, per candidate, so a
            # connect failure on one host can retry the next with its own id.
            # The same rewrite carries a context reduction, when there is one,
            # and turns a canonical id back into the tag this box's own Ollama
            # knows the model by.
            send_id = up_alias.get(fleet_id, fleet_id) if not cand else fleet_id
            send_body = body
            if cand_payload is not None or (model and send_id != model):
                base = cand_payload if cand_payload is not None else payload
                send_body = json.dumps({**base, "model": send_id}).encode()
                hdrs["content-length"] = str(len(send_body))
            if cand:
                pk = await peer_inference_key(cand)
                p = next((x for x in load_peers() if x["name"] == cand), None)
                if not (pk and p):
                    continue
                peer_client = httpx.AsyncClient(
                    base_url=p["url"].rstrip("/"),
                    timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None),
                )
                send_client = peer_client
                hdrs["authorization"] = "Bearer " + pk
                hdrs.pop("Authorization", None)
                hdrs["x-fleet-origin"] = HOST_NAME + "/" + str(job["id"])

            assert send_client is not None
            req = send_client.build_request(
                request.method, endpoint, headers=hdrs, content=send_body,
                params=dict(request.query_params),
            )
            if untrack:
                untrack()
            untrack = _track(hname)
            # How long to wait for the first byte before asking the next box
            # instead. Only when there IS a next box: with nobody else to
            # ask, waiting is still the best the caller can get.
            resident = fleet_id in _routes_cache.get("running", {}).get(hname, set())
            deadline = _ttfb_deadline(prompt_est, resident) if more else None
            try:
                # Raced, because the wait for the first byte is itself a place
                # a request gets stuck: llama-swap loading a large model
                # answers nothing at all until it is resident, and a load
                # that never finishes leaves this parked with no other way out.
                work: Any = send_client.send(req, stream=True)
                if deadline:
                    work = asyncio.wait_for(work, timeout=deadline)
                resp, cut = await _race_abort(work, job)
            except asyncio.TimeoutError:
                tried.append((hname, " (no answer in %ds)" % int(deadline or 0)))
                upstream_failed = True
                _mark_host_down(hname, COOLDOWN_STALL,
                                "no first byte for " + fleet_id + " within "
                                + str(int(deadline or 0)) + "s")
                await _quiet_close(peer_client)
                peer_client = None
                continue
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                tried.append((hname, ""))
                upstream_failed = True
                _mark_host_down(hname, COOLDOWN_CONNECT, type(exc).__name__)
                await _quiet_close(peer_client)
                peer_client = None
                _routes_cache["t"] = 0.0  # that host is down; re-resolve next time
                continue
            except httpx.HTTPError as exc:
                # The box accepted the connection and then broke it before
                # answering -- a llama-server dying on a load it could not
                # hold, typically. Nothing has reached the client yet, so the
                # next box is safe to try.
                tried.append((hname, " (" + type(exc).__name__ + ")"))
                upstream_failed = True
                _mark_host_down(hname, COOLDOWN_MIDSTREAM,
                                type(exc).__name__ + " before first byte")
                await _quiet_close(peer_client)
                peer_client = None
                continue
            if cut:
                await _quiet_close(peer_client)
                peer_client = None
                break
            if resp.status_code in (400, 413, 422):
                try:
                    err_body = await resp.aread()
                except Exception:  # noqa: BLE001 -- a peek must never break failover
                    err_body = b""
                if _classify_upstream_failure(
                        resp.status_code, path, err_body) == "ctx_too_long":
                    ctx_retried_ok = False
                    retry_payload = None
                    # ctx_retry_live gates only the same-box fitted retry
                    # attempt below -- classification, and the failover to
                    # the next candidate this leads to when it stays
                    # unattempted or fails, happen either way. A box that
                    # cannot serve this prompt is "not on THIS box" whether
                    # or not the owner wants the gateway to try shrinking it
                    # first.
                    if get_public_settings().get("ctx_retry_live", True):
                        prior = (cand_ctx if cand_payload is not None
                                else (ctx_limit or (prompt_est + gen_est)))
                        fit_to = _fit_after_ctx_overflow(err_body, prior)
                        if fit_to:
                            base = cand_payload if cand_payload is not None else payload
                            try:
                                retry_payload = apply_ctx_limit(dict(base), fit_to)
                            except HTTPException:
                                retry_payload = None
                    # The failed attempt is done with either way -- close
                    # it before the retry opens a second request on the
                    # same client, and reuse `resp`/`cut` in place so
                    # every path below sees exactly one response, exactly
                    # like every other candidate.
                    await _quiet_close(resp)
                    resp = None
                    if retry_payload is not None:
                        retry_body = json.dumps(
                            {**retry_payload, "model": send_id}).encode()
                        retry_hdrs = dict(hdrs)
                        retry_hdrs["content-length"] = str(len(retry_body))
                        req2 = send_client.build_request(
                            request.method, endpoint, headers=retry_hdrs,
                            content=retry_body, params=dict(request.query_params))
                        try:
                            resp, cut = await _race_abort(
                                send_client.send(req2, stream=True), job)
                        except (httpx.ConnectError, httpx.ConnectTimeout,
                                httpx.HTTPError):
                            resp, cut = None, False
                        if cut:
                            await _quiet_close(peer_client)
                            peer_client = None
                            break
                        if resp is not None and resp.status_code < 400:
                            remember_model_ctx({(cand, fleet_id): fit_to})
                            ctx_retried_ok = True
                    if not ctx_retried_ok:
                        tried.append((hname, " (context, retried)"
                                     if retry_payload is not None
                                     else " (context)"))
                        _mark_host_down(
                            hname, COOLDOWN_MIDSTREAM,
                            "ctx_too_long" + (" retry exhausted for " if retry_payload is not None
                                             else " for ") + fleet_id)
                        await _quiet_close(resp, peer_client)
                        resp, peer_client = None, None
                        # NOT upstream_failed: this box did not break, it
                        # declined the prompt on length -- once or twice
                        # depending on ctx_retry_live, but still on length,
                        # exactly like the proactive ctx_reject case above.
                        # Recorded the same way, so a request whose every
                        # candidate only ever declines on length still gets
                        # the honest 413/context_limit this box actually
                        # reported, instead of upstream_failed forcing a 502
                        # that reads as an outage and invites a retry that
                        # cannot possibly succeed.
                        ctx_reject = HTTPException(
                            413, _ctx_overflow_reject_detail(err_body))
                        continue
            if more and _upstream_failed(resp.status_code, path):
                # A 5xx from this box is final for THIS box, not for the
                # request: the status is known before a byte of body has
                # gone to the client, so another box can still answer. The
                # batch dispatcher always retried these; the live path now
                # does too.
                kind = _classify_upstream_failure(resp.status_code, path)
                tried.append((hname, " (HTTP " + str(resp.status_code) + ")"))
                _mark_host_down(
                    hname, _busy_cooldown_seconds() if kind == "busy"
                    else COOLDOWN_UPSTREAM_5XX,
                    "HTTP " + str(resp.status_code) + " for " + fleet_id)
                if kind == "model_missing":
                    _routes_cache["t"] = 0.0  # the table lied; re-resolve
                await _quiet_close(resp, peer_client)
                resp, peer_client = None, None
                upstream_failed = True
                continue
            target_peer = cand
            break
    except BaseException as exc:
        # Anything the loop did not plan for -- the request task cancelled
        # under us, a bug -- must not leave a job listed as active forever
        # with nothing left to abort, and must still be metered.
        if untrack:
            untrack()
        _job_close(job)
        await _quiet_close(resp, peer_client)
        record_usage(key, model, endpoint, stream,
                     499 if isinstance(exc, asyncio.CancelledError) else 502,
                     None, None, int((time.time() - started) * 1000))
        raise
    finally:
        watcher.cancel()

    if job["aborted"]:
        if untrack:
            untrack()
        _job_close(job)
        await _quiet_close(resp, peer_client)
        record_usage(key, model, endpoint, stream,
                     503 if job.get("drained") else 499, None, None,
                     int((time.time() - started) * 1000))
        return _stopped_response(job)

    if resp is None and ctx_reject is not None and not upstream_failed:
        # Every candidate that could have taken this request has a context
        # window too small for the prompt. That is a 413 about the prompt, not
        # a 502 about the fleet, and the detail already names both numbers.
        if untrack:
            untrack()
        _job_close(job)
        record_usage(key, model, endpoint, stream, 413, None, None,
                     int((time.time() - started) * 1000))
        return JSONResponse(
            ctx_reject.detail if isinstance(ctx_reject.detail, dict)
            else {"error": {"message": str(ctx_reject.detail),
                            "type": "context_limit"}},
            status_code=413)

    if resp is None:
        if untrack:
            untrack()
        _job_close(job)
        record_usage(key, model, endpoint, stream, 502, None, None,
                     int((time.time() - started) * 1000))
        # A Fleet Pass caller is told "Box 3", never a hostname -- the same
        # rule as every other public surface.
        shown = [((public_alias(h) if pub_row else h) + note) for h, note in tried]
        return JSONResponse(
            {
                "error": {
                    "message": "no reachable host for '" + (model or "request")
                    + "' (tried: " + (", ".join(shown) or "none") + ")",
                    "type": "upstream",
                }
            },
            status_code=502,
        )

    served_by = target_peer or HOST_NAME
    if untrack is None:
        untrack = _track(served_by)
    job["host"] = served_by
    if resp.status_code >= 500 or resp.status_code in (429, 503):
        # The last box standing failed too; remember it for the next caller.
        # A 429/503 gets the short busy cooldown -- it is likely free again
        # soon -- everything else the standard one.
        kind = _classify_upstream_failure(resp.status_code, path)
        _mark_host_down(
            served_by, _busy_cooldown_seconds() if kind == "busy"
            else COOLDOWN_UPSTREAM_5XX,
            "HTTP " + str(resp.status_code) + " (no other host)")

    drop = {"content-length", "transfer-encoding", "connection", "content-encoding"}
    out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in drop}
    pub_notice, pub_xf = "", {}
    if fallback or ctx_cut:
        pub_notice, pub_xf, pub_hdrs = public_notices(
            fallback, ctx_cut, served_by, get_public_settings(), public=bool(pub_row))
        out_headers.update(pub_hdrs)
        if _wants_structured(payload) or continued:
            # The client will parse the content whole, or has already been
            # told once in this conversation. Either way the disclosure rides
            # in x_fleet and the headers instead of in the answer.
            pub_notice = ""

    if not stream:
        raw_out = b""
        try:
            raw_out, cut = await _race_abort(resp.aread(), job)
            if cut:
                raw_out = b""
        finally:
            untrack()
            await _quiet_close(resp, peer_client)
            _job_close(job)
        if job["aborted"]:
            # 499, the nginx spelling of "the client went away" -- except here
            # the client is us, on the operator's behalf. Metered like any
            # other outcome: an aborted runaway still cost the box the tokens
            # it had already produced. A drain is the other way round: nobody
            # asked for this one to stop, so it meters as the 503 it is.
            record_usage(key, model, endpoint, False,
                         503 if job.get("drained") else 499, None, None,
                         int((time.time() - started) * 1000), host=served_by)
            return _stopped_response(job)
        usage = None
        out_body = raw_out
        try:
            obj = json.loads(raw_out)
        except Exception:  # noqa: BLE001
            obj = None
        if isinstance(obj, dict):
            usage = obj.get("usage")
            if pub_xf:
                obj["x_fleet"] = pub_xf
                prepend_notice(obj, pub_notice)
                out_body = json.dumps(obj, ensure_ascii=False).encode()
        if resp.status_code < 500 and resp.status_code not in (429, 503):
            # A 429/503 was just cooled down above (it is 'busy', not
            # broken) -- clearing that here on the same response would make
            # the cooldown a no-op the instant it was set.
            _mark_host_ok(served_by)
        # A caller that gave up before the answer arrived never received it,
        # and every OpenAI SDK retries by default (max_retries=2) -- so a slow
        # first call would quietly spend two of a demo key's five daily
        # requests, and the recruiter would see one reply. Meter an abandoned
        # request as 499, which the budget predicate excludes: the tokens are
        # still recorded (the box really did the work), the quota is not.
        meter_status = resp.status_code
        try:
            if await request.is_disconnected():
                meter_status = 499
        except Exception:  # noqa: BLE001 -- metering must never break inference
            pass
        record_usage(key, model, endpoint, False, meter_status, usage, None,
                     int((time.time() - started) * 1000), host=served_by,
                     fallback_from=(fallback["requested"] if fallback else ""))
        if fallback:
            _note_fallback(int(key["id"]), fallback["requested"], fallback["served"])
        return Response_bytes(out_body, resp.status_code, out_headers)

    async def relay():
        ttft: int | None = None
        usage: dict | None = None
        status_out = resp.status_code
        if pub_notice and resp.status_code < 400:
            # id/model are made up here -- the upstream's own first chunk has
            # not arrived yet -- but the shape is a real SSE chat chunk (role
            # included, for parsers that key the accumulator off the first
            # delta), so any client streaming-parser handles it the same as a
            # model token. Gated on a healthy status: an erroring upstream
            # sends a JSON error body, and an SSE frame in front of it used
            # to corrupt exactly the reply that most needed to be readable.
            head = {
                "id": "fleet-notice", "object": "chat.completion.chunk",
                "created": int(started),
                "model": (fallback["served"] if fallback else model),
                "choices": [{"index": 0, "finish_reason": None,
                            "delta": {"role": "assistant",
                                      "content": pub_notice}}],
            }
            yield ("data: " + json.dumps(head, ensure_ascii=False) + "\n\n").encode()
        chunks = resp.aiter_bytes()
        try:
            while True:
                # Not a plain `async for`: a stream that has stopped producing
                # entirely is half of what this button exists for, and only
                # racing the read against the abort catches that one.
                try:
                    chunk, cut = await _race_abort(chunks.__anext__(), job)
                except StopAsyncIteration:
                    break
                except Exception as exc:  # noqa: BLE001
                    # The box died mid-answer. A dropped socket reads as an
                    # opaque network error in every SDK; a framed SSE error
                    # followed by [DONE] reads as what it is. The tokens that
                    # did arrive are still metered, as a 502.
                    status_out = 502
                    _mark_host_down(served_by, COOLDOWN_MIDSTREAM,
                                    "stream broke: " + type(exc).__name__)
                    err = {"error": {"message": "upstream stream ended unexpectedly ("
                                                + type(exc).__name__ + ")",
                                     "type": "upstream"}}
                    yield ("data: " + json.dumps(err) + "\n\n").encode()
                    yield b"data: [DONE]\n\n"
                    break
                if cut:
                    if job.get("drained"):
                        # A restart, not the dashboard's stop button. Same
                        # framing as the broken-upstream case above and for
                        # the same reason: the alternative is the socket
                        # simply stopping, which reaches a browser-based
                        # client as ERR_HTTP2_PROTOCOL_ERROR and tells whoever
                        # reads it to go and check their firewall. 503 says
                        # "ask again in a moment", which is the truth.
                        status_out = 503
                        err = {"error": {"message": DRAIN_MESSAGE,
                                         "type": "unavailable",
                                         "code": "gateway_restarting"}}
                        yield ("data: " + json.dumps(err) + "\n\n").encode()
                        yield b"data: [DONE]\n\n"
                    break
                if ttft is None:
                    ttft = int((time.time() - started) * 1000)
                    if status_out < 500 and status_out not in (429, 503):
                        _mark_host_ok(served_by)
                if b'"usage"' in chunk:
                    for line in chunk.split(b"\n"):
                        line = line.strip()
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data in (b"[DONE]", b""):
                            continue
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(obj, dict) and obj.get("usage"):
                            usage = obj["usage"]
                yield chunk
        finally:
            untrack()
            await _quiet_close(resp, peer_client)
            _job_close(job)
            record_usage(key, model, endpoint, True,
                         # A drain already set status_out to 503; only the
                         # operator's abort is a 499.
                         499 if (job["aborted"] and not job.get("drained"))
                         else status_out,
                         usage, ttft,
                         int((time.time() - started) * 1000), host=served_by,
                         fallback_from=(fallback["requested"] if fallback else ""))
            if fallback:
                _note_fallback(int(key["id"]), fallback["requested"], fallback["served"])

    return StreamingResponse(
        relay(), status_code=resp.status_code, headers=out_headers,
        media_type=resp.headers.get("content-type", "text/event-stream"),
    )


TTFB_BASE = float(os.environ.get("LLMSTACK_TTFB_BASE", "75"))


def _ttfb_deadline(prompt_tokens: int, resident: bool) -> float:
    """Seconds to wait for a box to start answering before the next box is
    asked instead. Generous on purpose: the big box takes tens of seconds to
    read a 40k-token prompt and a minute to cold-load a 60 GB model, and a
    deadline that fires on legitimate work is worse than none. It exists to
    catch a load that will never finish, not a slow one."""
    d = TTFB_BASE + max(0, prompt_tokens) / 150.0 + (0.0 if resident else 90.0)
    return min(600.0, d)


# Vocabulary a llama.cpp server (>= b4700) or an older one's plain-text 4xx
# uses to say "this prompt does not fit in the box's context window" -- as
# opposed to any other 400/413/422, which is an ordinary client error the
# fleet had no part in and must be returned unchanged, never retried.
# Deliberately conservative: false negatives just fall through to today's
# plain-error behaviour (harmless), false positives would shrink a request
# that did not need it, so every string here is one actually seen naming the
# context window, not a guess at what one might say.
_CTX_OVERFLOW_SIGNS = (
    "exceeds the available context size", "context size", "n_ctx",
    "prompt is too long", "input is too large", "too many tokens",
    "maximum context length", "context length exceeded", "context window",
)


def _classify_upstream_failure(status: int, path: str, body: bytes = b"") -> str:
    """What kind of thing just happened, for the reactive-failure branches
    (requirement 3): 'busy' | 'ctx_too_long' | 'oom' | 'model_missing' |
    'other_5xx' | 'ok'. `body` is optional -- most call sites decide before
    a byte of it has been read, and every kind but 'ctx_too_long' can be told
    from the status alone.

    'busy' (429/503) is not a failure at all, just a full box; it earns the
    short cooldown and an immediate retry elsewhere. 'ctx_too_long' is
    matched against the llama.cpp server's own error shape first --
    {"error": {"type": "exceed_context_size_error", "n_ctx": M, ...}} -- and
    only falls back to the message-substring list for a server that does not
    send it. Anything else in the 4xx range is the caller's problem, not the
    fleet's, and is deliberately left unclassified ('ok') so it is returned
    to the client exactly as the box sent it."""
    if status < 400:
        return "ok"
    if status in (429, 503):
        return "busy"
    if status >= 500:
        text = body.decode("utf-8", "ignore").lower() if body else ""
        return "oom" if any(s in text for s in _MEM_SIGNS) else "other_5xx"
    if status == 404 and path == "chat/completions":
        # llama-swap answers this for a model id the box no longer has.
        return "model_missing"
    if status in (400, 413, 422) and body:
        text = body.decode("utf-8", "ignore").lower()
        try:
            err = json.loads(text).get("error")
        except (json.JSONDecodeError, TypeError, AttributeError):
            err = None
        if isinstance(err, dict) and err.get("type") == "exceed_context_size_error":
            return "ctx_too_long"
        if any(s in text for s in _CTX_OVERFLOW_SIGNS):
            return "ctx_too_long"
    return "ok"


def _upstream_failed(status: int, path: str, body: bytes = b"") -> bool:
    """A response that means 'this box could not do it', as opposed to one
    that means 'this request is wrong'. A 404 on chat/completions is the
    former: llama-swap answers it for a model id the box no longer has.
    `body` is optional and only sharpens 'oom' vs 'other_5xx' in the log
    line callers write -- both count as failed either way."""
    return _classify_upstream_failure(status, path, body) in (
        "busy", "other_5xx", "oom", "model_missing")


def _ctx_overflow_ceiling(body: bytes) -> int:
    """The box's own stated n_ctx from a llama.cpp exceed_context_size_error
    body, or 0 when the error did not name one -- an older server, or a
    message-only match, either of which leaves the caller to fall back to
    halving whatever was last tried instead of a size the box actually
    reported."""
    try:
        obj = json.loads(body.decode("utf-8", "ignore"))
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
        return 0
    err = obj.get("error") if isinstance(obj, dict) else None
    if not isinstance(err, dict):
        return 0
    try:
        return max(0, int(err.get("n_ctx") or 0))
    except (TypeError, ValueError):
        return 0


# Shaved off a reported n_ctx ceiling so the fitted retry lands safely under
# it rather than exactly on it -- special tokens and chat-template overhead
# are the box's own accounting to get right, not ours to reproduce exactly.
CTX_OVERFLOW_MARGIN = 256


def _fit_after_ctx_overflow(body: bytes, prior_ctx: int) -> int:
    """The context size to retry the SAME box with, one time, after it
    rejected a request as too long for its window: the box's own reported
    ceiling (minus CTX_OVERFLOW_MARGIN) when the error named one, else half
    of whatever was last tried -- the same halve_ctx() the model-apply
    verifier and native_proxy's Ollama retry already use. 0 means nothing
    smaller is worth trying; the caller should give up and move on like any
    other failed candidate."""
    n_ctx = _ctx_overflow_ceiling(body)
    if n_ctx > 0:
        fit = n_ctx - CTX_OVERFLOW_MARGIN
        # A box's own reported ceiling is a hard fact, not a number to round
        # UP to a grid floor: for a small box (n_ctx <= _CTX_RETRY_FLOOR +
        # CTX_OVERFLOW_MARGIN) that would ask it to retry at or above the
        # very ceiling it just named, burning the one allowed retry on a
        # request already proven too big. Below the floor simply means this
        # box has nothing smaller worth trying -- 0 tells the caller to give
        # up on it and move on, exactly like the halve_ctx() fallback below.
        return fit if fit >= _CTX_RETRY_FLOOR else 0
    return halve_ctx(max(1, int(prior_ctx or 0)))


def _ctx_overflow_reject_detail(body: bytes) -> dict:
    """Turn a box's own ctx_too_long error body into the same
    {"error": {"type": "context_limit", ...}} shape apply_ctx_limit's
    PROACTIVE rejection raises, so a same-box reactive retry that ALSO
    overflowed reports the same honest 413 a caller whose prompt simply
    does not fit anywhere would have gotten -- instead of a 502 that reads
    as an outage and invites a retry that cannot possibly succeed."""
    message = None
    try:
        obj = json.loads(body.decode("utf-8", "ignore"))
        err = obj.get("error") if isinstance(obj, dict) else None
        if isinstance(err, dict) and err.get("message"):
            message = str(err["message"])
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError, AttributeError):
        pass
    detail: dict = {"error": {
        "message": message or "prompt is too long for this model's context window",
        "type": "context_limit",
    }}
    n_ctx = _ctx_overflow_ceiling(body)
    if n_ctx:
        detail["error"]["limit"] = n_ctx
    return detail


def _watch_disconnect(request: Request, job: dict) -> "asyncio.Task":
    """Abort `job` if the caller hangs up while the upstream is still being
    waited on. Starlette only notices a disconnect once a response is being
    streamed; before the first byte, the wait for a loading model used to
    carry on for a client that had already gone -- and so did the load."""
    async def run() -> None:
        try:
            while True:
                await asyncio.sleep(1.0)
                if await request.is_disconnected():
                    job_abort(job)
                    return
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001 -- a watchdog must never take the request down
            return
    return asyncio.create_task(run())


ABORTED_BODY = {
    "error": {
        "message": "aborted from the fleet dashboard",
        "type": "aborted",
        "code": "aborted",
    }
}

DRAINED_BODY = {
    "error": {
        "message": DRAIN_MESSAGE,
        "type": "unavailable",
        "code": "gateway_restarting",
    }
}

# Long enough that a client's own backoff lands after systemd's RestartSec=3
# and the startup, short enough to be worth honouring.
DRAIN_RETRY_AFTER = "10"


def _stopped_response(job: dict) -> JSONResponse:
    """The body for a non-streaming request that was stopped, whichever lever
    stopped it. A drain is a 503 the caller should retry; the dashboard's stop
    button is a 499 it should not."""
    if job.get("drained"):
        return JSONResponse(DRAINED_BODY, status_code=503,
                            headers={"retry-after": DRAIN_RETRY_AFTER})
    return JSONResponse(ABORTED_BODY, status_code=499)


def Response_bytes(data: bytes, status: int, headers: dict):
    from fastapi.responses import Response as _R

    return _R(content=data, status_code=status, headers=headers)


# ------------------------- native upstream proxy ---------------------------
#
# Ollama's OpenAI-compatible /v1 cannot express num_ctx, keep_alive, or the
# /api/ps residency probe -- and the downstream app's per-role context tuning depends on
# exactly those. So the native API is proxied too, behind the same bearer keys,
# with usage metered from Ollama's own eval counts. Fleet routing deliberately
# does NOT apply here: the native API is inherently host-specific.

METERED_NATIVE = ("chat", "generate", "embed", "embeddings")


def _ollama_usage(obj: dict) -> dict | None:
    if not isinstance(obj, dict) or "eval_count" not in obj:
        return None
    pt = int(obj.get("prompt_eval_count", 0) or 0)
    ct = int(obj.get("eval_count", 0) or 0)
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}


@app.api_route("/api/{path:path}", methods=["GET", "POST", "DELETE", "HEAD"])
async def native_proxy(path: str, request: Request):
    key = require_api_key(request)
    assert client is not None
    endpoint = "/api/" + path
    started = time.time()
    metered = path.split("/")[0] in METERED_NATIVE

    # Same refusal as openai_proxy: a restart is seconds away, so do not begin.
    if draining():
        if metered:
            record_usage(key, "", endpoint, False, 503, None, None,
                         int((time.time() - started) * 1000))
        return JSONResponse(DRAINED_BODY, status_code=503,
                            headers={"retry-after": DRAIN_RETRY_AFTER})

    # A public key is not for Ollama's native API at all -- the whole surface
    # is 403, same rule as openai_proxy (contract 1.9h).
    if public_key_row(int(key["id"])):
        if metered:
            record_usage(key, "", endpoint, False, 403, None, None,
                         int((time.time() - started) * 1000))
        return JSONResponse(
            {"error": {"message": "this key is limited to POST /v1/chat/completions "
                                  "and GET /v1/models", "type": "permission_error"}},
            status_code=403,
        )

    raw = await request.body()
    model, body = "", raw
    stream = True  # ollama streams by default
    payload: dict | None = None  # bound even when raw is empty or not JSON,
                                  # for the context-retry check below
    if raw:
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                stream = bool(payload.get("stream", True))
                agent = get_agent(int(key["id"]))
                if agent and metered:
                    # model pinning/allow-list and system-prompt injection work
                    # identically on the native chat shape; sampler overrides
                    # land as unknown top-level keys Ollama ignores.
                    payload = apply_agent(agent, payload)
                    body = json.dumps(payload).encode()
                model = str(payload.get("model", ""))
                # Native Ollama shape, same rule as /v1: the fleet's canonical
                # id goes to the engine as the tag it knows.
                tag = (await upstream_alias_pairs()).get(model)
                if tag:
                    payload["model"] = tag
                    body = json.dumps(payload).encode()
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    hop = {"host", "content-length", "connection", "authorization",
           "x-api-key", "cf-access-jwt-assertion", "cookie", "x-fleet-origin"}
    origin = request.headers.get("x-fleet-origin", "")
    fwd = {k: v for k, v in request.headers.items() if k.lower() not in hop}
    if body:
        fwd["content-length"] = str(len(body))

    req = client.build_request(request.method, endpoint, headers=fwd,
                               content=body, params=dict(request.query_params))
    # The native API is never fleet-routed, so this box is always the one doing
    # the work -- but a long /api/generate runs away exactly like a /v1 one,
    # and it can stall waiting for the first byte just the same.
    job = _job_open("inference", what=model or path, origin=origin,
                    stop=asyncio.Event(),
                    detail=endpoint + (" · stream" if stream else "")
                    + (" · " + str(key.get("name")) if key.get("name") else ""))
    # Counted against this box from the moment it is asked, exactly like
    # every fleet-routed kind -- without this, a native call in flight was
    # invisible to _inflight[HOST_NAME], so a concurrent /v1 or fleet_chat
    # request could rank this host as idle while it was actually busy here.
    untrack = _track(HOST_NAME)
    try:
        resp, cut = await _race_abort(client.send(req, stream=True), job)
    except httpx.HTTPError:
        # Anything httpx raises trying to reach or read from this box --
        # not just a refused connection (httpx.ConnectError/ConnectTimeout
        # are themselves HTTPError subclasses): a peer that accepted the
        # TCP connection and then broke before a response arrived surfaces
        # as httpx.ReadError or httpx.RemoteProtocolError, and those used to
        # propagate out of this function uncaught, past the untrack() below,
        # leaving _inflight[HOST_NAME] permanently off by one -- this box
        # would sort as saturated in every routing decision for the rest of
        # the process's life.
        untrack()
        _job_close(job)
        if metered:
            record_usage(key, model, endpoint, stream, 502, None, None,
                         int((time.time() - started) * 1000))
        return JSONResponse(
            {"error": "native upstream unavailable"}, status_code=502
        )
    if cut:
        untrack()
        _job_close(job)
        if metered:
            record_usage(key, model, endpoint, stream, 499, None, None,
                         int((time.time() - started) * 1000))
        return JSONResponse(ABORTED_BODY, status_code=499)

    drop = {"content-length", "transfer-encoding", "connection", "content-encoding"}
    out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in drop}

    if not stream or request.method != "POST":
        raw_out = b""
        try:
            raw_out, cut = await _race_abort(resp.aread(), job)
            if cut:
                raw_out = b""
        finally:
            untrack()
            await _quiet_close(resp)
            _job_close(job)
        if job["aborted"]:
            if metered:
                record_usage(key, model, endpoint, False, 499, None, None,
                             int((time.time() - started) * 1000))
            return JSONResponse(ABORTED_BODY, status_code=499)
        status = resp.status_code
        # Ollama takes its context window per request (`options.num_ctx`),
        # unlike llama.cpp's launch-time -c -- so an oversized window is a
        # per-call failure here, not a model load an operator watches. Ollama
        # runs on the same ggml backend llama.cpp does, so a real overfit
        # fails with the vocabulary _MEM_SIGNS already matches. One retry at
        # half the window; remembered on success so routing (host_model_ctx,
        # consulted fleet-wide) stops offering this pair a window it has
        # already proven too big, instead of failing it again next call.
        # Streaming requests are not retried here: Ollama streams by default,
        # and bytes already relayed to the caller cannot be taken back.
        if metered and status >= 400 and model and isinstance(payload, dict):
            opts = payload.get("options")
            try:
                asked = int(opts.get("num_ctx") or 0) if isinstance(opts, dict) else 0
            except (TypeError, ValueError):
                asked = 0
            overfit = asked and any(
                s in raw_out.decode("utf-8", "ignore").lower() for s in _MEM_SIGNS)
            smaller = halve_ctx(asked) if overfit else 0
            if smaller:
                retry_payload = dict(payload, options=dict(opts, num_ctx=smaller))
                retry_body = json.dumps(retry_payload).encode()
                retry_fwd = dict(fwd)
                retry_fwd["content-length"] = str(len(retry_body))
                retry_req = client.build_request(
                    request.method, endpoint, headers=retry_fwd,
                    content=retry_body, params=dict(request.query_params))
                try:
                    resp2 = await client.send(retry_req)
                    raw_out, status = resp2.content, resp2.status_code
                    out_headers = {k: v for k, v in resp2.headers.items()
                                  if k.lower() not in drop}
                    if status < 400:
                        remember_model_ctx({("", model): smaller})
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    pass  # the original failure is still the honest answer
        usage = None
        if metered:
            try:
                usage = _ollama_usage(json.loads(raw_out))
            except Exception:  # noqa: BLE001
                pass
            record_usage(key, model, endpoint, False, status, usage,
                         None, int((time.time() - started) * 1000))
        return Response_bytes(raw_out, status, out_headers)

    async def relay():
        ttft: int | None = None
        usage: dict | None = None
        chunks = resp.aiter_bytes()
        try:
            while True:
                try:
                    chunk, cut = await _race_abort(chunks.__anext__(), job)
                except StopAsyncIteration:
                    break
                if cut:
                    break
                if ttft is None:
                    ttft = int((time.time() - started) * 1000)
                if b'"eval_count"' in chunk:
                    # NDJSON: the final object carries the counts.
                    for line in chunk.split(b"\n"):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            u = _ollama_usage(json.loads(line))
                            if u:
                                usage = u
                        except json.JSONDecodeError:
                            continue
                yield chunk
        finally:
            untrack()
            await _quiet_close(resp)
            _job_close(job)
            if metered:
                record_usage(key, model, endpoint, True,
                             499 if job["aborted"] else resp.status_code,
                             usage, ttft, int((time.time() - started) * 1000))

    return StreamingResponse(
        relay(), status_code=resp.status_code, headers=out_headers,
        media_type=resp.headers.get("content-type", "application/x-ndjson"),
    )


# --------------------------------------------------------------------------
# orchestrator: fleet-routed single calls, batch fan-out, and teams
#
# Three layers, each built on the one below:
#   * fleet_chat()      -- one non-streaming completion, routed to the best
#                          host serving the model, with connect failover.
#   * batches           -- a spooled list of requests fanned out over every
#                          (host, model) pair that can serve them. Workers
#                          pull from one shared queue, so a fast box simply
#                          takes more items: the balancing is emergent, and
#                          the throughput history only decides the ORDER
#                          hosts are preferred in, never a fixed split.
#   * teams             -- a key whose primary model is handed a
#                          spawn_subagents tool; the gateway executes the
#                          spawned tasks across the fleet and loops until the
#                          primary answers. Any OpenAI client gets parallel
#                          sub-agents without knowing the fleet exists.
# --------------------------------------------------------------------------

BATCHES_DIR = Path(os.environ.get("LLMSTACK_BATCHES", str(STATE / "batches")))


async def _post_chat(cand: str, payload: bytes,
                     read_timeout: float | None = None) -> httpx.Response:
    """POST one chat completion to a candidate host ('' = local upstream).

    Non-streaming by construction: the callers are batch workers and the team
    loop, both of which need the whole body anyway. No read timeout by
    default -- a cold model load plus a long completion can take minutes;
    the caller sets one only while it still has another box to try."""
    tmo = httpx.Timeout(connect=10.0, read=read_timeout, write=None, pool=None)
    if not cand:
        assert client is not None
        return await client.post("/v1/chat/completions", content=payload,
                                 headers={"Content-Type": "application/json"},
                                 timeout=tmo)
    pk = await peer_inference_key(cand)
    p = next((x for x in load_peers() if x["name"] == cand), None)
    if not (pk and p):
        raise httpx.ConnectError("no admin token or url for peer " + cand)
    async with httpx.AsyncClient(base_url=p["url"].rstrip("/"), timeout=tmo) as c:
        return await c.post("/v1/chat/completions", content=payload,
                            headers={"Authorization": "Bearer " + pk,
                                     "Content-Type": "application/json"})


async def fleet_chat(key: dict, body: dict, endpoint: str,
                     ctx_limit: int | None = None, role: str = "primary",
                     ) -> tuple[int, dict, str, int]:
    """One completion, routed like the /v1 proxy routes: best host first,
    failover down the candidate list on a connect failure, a stall, or a
    5xx. Returns (status, body, host, granted_ctx).

    `model` may be a public catalogue id -- resolve_targets() is the same
    public-id-to-fleet-id resolution the /v1 proxy uses, so a team's primary
    or a public single key benefits here too, not just plain bearer keys.
    `role` is 'primary' or 'worker' and selects the host policy (a team's
    sub-agents go to the always-on small boxes ahead of the big one).

    `ctx_limit` is the window the caller was promised. When the box that ends
    up answering cannot hold it, the request is fitted to what that box CAN
    hold and `granted_ctx` comes back non-zero, so the caller can disclose the
    reduction the same way a model substitution is disclosed. 0 means the full
    window was honoured (or nothing has ever reported one for that box)."""
    model = str(body.get("model", ""))
    body = dict(body)
    body["stream"] = False
    started = time.time()
    prompt_est = estimate_prompt_tokens(body)
    try:
        gen_est = min(1024, int(body.get("max_tokens") or 0)) or 256
    except (TypeError, ValueError):
        gen_est = 256
    try:
        targets = await resolve_targets(
            model, role=role, prompt_tokens=prompt_est, gen_tokens=gen_est,
            need_ctx=prompt_est + gen_est) or []
    except Exception:  # noqa: BLE001 -- routing is best-effort
        targets = [("", model)]
    if not targets:
        return 404, {"error": {"message": "no host in the fleet serves model '"
                               + model + "'", "type": "invalid_request_error"}}, "", 0
    if ctx_limit and len(targets) > 1:
        targets = sorted(targets, key=lambda t: _ctx_rank(t, ctx_limit))
    last_err: tuple[int, dict, str, int] | None = None
    # The 413 apply_ctx_limit raised for the last box the prompt did not fit,
    # and whether any box actually BROKE. When the loop ends with candidates
    # only ever declining on length, the honest answer is that 413 -- it names
    # both numbers -- not a 502 that reads as an outage and invites a retry
    # that cannot succeed.
    ctx_reject: HTTPException | None = None
    upstream_failed = False
    for i, (cand, fleet_id) in enumerate(targets):
        more = i < len(targets) - 1
        hname = cand or HOST_NAME
        send, granted = body, 0
        if ctx_limit:
            cand_ctx = host_model_ctx(cand, fleet_id)
            if 0 < cand_ctx < ctx_limit:
                try:
                    send = apply_ctx_limit(dict(body), cand_ctx)
                except HTTPException as exc:
                    # Too long for this box; another candidate may fit it.
                    ctx_reject = exc
                    continue
                granted = cand_ctx
        untrack = _track(hname)
        payload = json.dumps({**send, "model": fleet_id}).encode()
        # Non-streaming, so the whole answer is one read: the deadline has
        # to cover generation too, at a pessimistic rate. Only while another
        # box could still take the request.
        deadline = None
        if more:
            resident = fleet_id in _routes_cache.get("running", {}).get(hname, set())
            deadline = _ttfb_deadline(prompt_est, resident) + gen_est / 3.0
        try:
            r = await _post_chat(cand, payload, deadline)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            untrack()
            upstream_failed = True
            _mark_host_down(hname, COOLDOWN_CONNECT, type(exc).__name__)
            _routes_cache["t"] = 0.0
            continue
        except httpx.TimeoutException:
            untrack()
            upstream_failed = True
            _mark_host_down(hname, COOLDOWN_STALL,
                            "no answer for " + fleet_id + " within "
                            + str(int(deadline or 0)) + "s")
            continue
        except httpx.HTTPError as exc:
            untrack()
            upstream_failed = True
            _mark_host_down(hname, COOLDOWN_MIDSTREAM, type(exc).__name__)
            last_err = (502, {"error": {"message": str(exc)[:300], "type": "upstream"}},
                        hname, granted)
            continue
        except Exception as exc:  # noqa: BLE001
            untrack()
            return 502, {"error": {"message": str(exc)[:300], "type": "upstream"}}, hname, granted
        # _post_chat is non-streaming: r.content is already the full body, no
        # extra read needed to inspect it for a context-overflow shape.
        if r.status_code in (400, 413, 422):
            if _classify_upstream_failure(
                    r.status_code, "chat/completions", r.content) == "ctx_too_long":
                ctx_retried_ok = False
                retry_payload = None
                # ctx_retry_live gates only the same-box fitted retry
                # attempt below -- classification, and the failover this
                # leads to when it stays unattempted or fails, happen
                # either way. See the matching comment in openai_proxy.
                if get_public_settings().get("ctx_retry_live", True):
                    prior = granted if granted else (ctx_limit or (prompt_est + gen_est))
                    fit_to = _fit_after_ctx_overflow(r.content, prior)
                    if fit_to:
                        try:
                            retry_payload = apply_ctx_limit(dict(send), fit_to)
                        except HTTPException:
                            retry_payload = None
                if retry_payload is not None:
                    retry_bytes = json.dumps(
                        {**retry_payload, "model": fleet_id}).encode()
                    try:
                        r2 = await _post_chat(cand, retry_bytes, deadline)
                    except (httpx.ConnectError, httpx.ConnectTimeout,
                            httpx.TimeoutException, httpx.HTTPError):
                        r2 = None
                    if r2 is not None and r2.status_code < 400:
                        remember_model_ctx({(cand, fleet_id): fit_to})
                        r, granted = r2, fit_to
                        ctx_retried_ok = True
                if not ctx_retried_ok:
                    # Same tracked window the whole time -- untrack only now,
                    # once both attempts are settled.
                    untrack()
                    _mark_host_down(
                        hname, COOLDOWN_MIDSTREAM,
                        "ctx_too_long" + (" retry exhausted for " if retry_payload is not None
                                         else " for ") + fleet_id)
                    # NOT upstream_failed: a decline on length, not a break --
                    # see the matching comment in openai_proxy. `r` is still
                    # the original (never-reassigned-on-failure) response, so
                    # its body is the box's own honest ctx_too_long detail.
                    ctx_reject = HTTPException(
                        413, _ctx_overflow_reject_detail(r.content))
                    continue
        untrack()
        if more and _upstream_failed(r.status_code, "chat/completions"):
            upstream_failed = True
            kind = _classify_upstream_failure(r.status_code, "chat/completions")
            _mark_host_down(
                hname, _busy_cooldown_seconds() if kind == "busy"
                else COOLDOWN_UPSTREAM_5XX,
                "HTTP " + str(r.status_code) + " for " + fleet_id)
            if kind == "model_missing":
                _routes_cache["t"] = 0.0
            continue
        if r.status_code >= 500 or r.status_code in (429, 503):
            kind = _classify_upstream_failure(r.status_code, "chat/completions")
            _mark_host_down(
                hname, _busy_cooldown_seconds() if kind == "busy"
                else COOLDOWN_UPSTREAM_5XX,
                "HTTP " + str(r.status_code) + " (no other host)")
        else:
            _mark_host_ok(hname)
        try:
            out = r.json()
        except ValueError:
            out = {"error": {"message": r.text[:300], "type": "upstream"}}
        record_usage(key, model, endpoint, False, r.status_code,
                     out.get("usage") if isinstance(out, dict) else None,
                     None, int((time.time() - started) * 1000), host=hname)
        return r.status_code, out if isinstance(out, dict) else {}, hname, granted
    if ctx_reject is not None and not upstream_failed and last_err is None:
        # Every candidate declined the prompt on length and nothing actually
        # broke: 413 about the prompt, not 502 about the fleet -- the same
        # distinction openai_proxy's own loop already draws.
        record_usage(key, model, endpoint, False, 413, None, None,
                     int((time.time() - started) * 1000))
        return 413, (ctx_reject.detail if isinstance(ctx_reject.detail, dict)
                     else {"error": {"message": str(ctx_reject.detail),
                                     "type": "context_limit"}}), "", 0
    record_usage(key, model, endpoint, False, 502, None, None,
                 int((time.time() - started) * 1000))
    if last_err:
        return last_err
    return 502, {"error": {"message": "no reachable host for '" + model + "'",
                           "type": "upstream"}}, "", 0


# ---- batches ----

_batch_tasks: dict[int, asyncio.Task] = {}
_batch_cancel: set[int] = set()
_batch_live: dict[int, dict] = {}


def _batch_paths(bid: int) -> tuple[Path, Path]:
    return BATCHES_DIR / (str(bid) + ".in.ndjson"), \
           BATCHES_DIR / (str(bid) + ".out.ndjson")


def _batch_need_ctx(lines: str) -> int:
    """The largest prompt-token estimate across a batch's input lines, for
    the ctx-aware host ranking below -- a batch of long prompts should not
    be handed to a box with a small window ahead of one that can actually
    hold them, the same rule the live proxy already applies per request."""
    best = 0
    for line in lines.splitlines():
        if not line.strip():
            continue
        try:
            body = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(body, dict):
            best = max(best, estimate_prompt_tokens(body))
    return best


async def _batch_targets(models: list[str], no_fallback: bool = False,
                         need_ctx: int = 0) -> list[dict]:
    """Every (host, model) pair that can serve this batch -- at most ONE model
    per host. The swap group on a llama.cpp box is exclusive, so handing one
    box two of the batch's models would make every request a model reload;
    the first model in the caller's order claims the host.

    Substitution (requirement 2) happens ONCE here, on the model LIST,
    before any host claims one -- never per dequeued item, which would turn
    a saturated model mid-batch into a stream of per-request reloads on
    whatever box happened to answer next. What actually served each item is
    already the honest record (the `model` field on every output line, and
    the `targets` this function returns) -- batches have no single reply to
    staple an x_fleet notice to, so this IS the disclosure."""
    await model_routes(force=True)
    cap = _routes_cache.get("cap", {})
    if not no_fallback:
        settings = get_public_settings()
        resolved: list[str] = []
        seen: set[str] = set()
        for m in models:
            out_m = m
            row = _catalogue_row_for(m)
            if row is not None:
                sub = await pick_fallback(row, "worker", settings)
                if sub is not None:
                    out_m = str(sub["public_id"])
            if out_m not in seen:
                seen.add(out_m)
                resolved.append(out_m)
        if resolved:
            models = resolved
    tps_by_model: dict[str, dict[str, float]] = {}

    async def tps_for(mid: str) -> dict[str, float]:
        if mid not in tps_by_model:
            tps_by_model[mid] = await asyncio.to_thread(measured_tps, mid)
        return tps_by_model[mid]

    taken: set[str] = set()
    targets: list[dict] = []
    for m in models:
        # Policy order, so the first-listed model claims the boxes the owner
        # ranked highest for batch work (role 'worker': sub-agent rules). A
        # fleet role fans out over each box's own best model for it, exactly
        # as the live proxy resolves one (role_pairs) -- and every target
        # then carries the id that box really serves, never the role word.
        # A catalogue id (the substitution above always names one by its
        # public_id) is expanded to every fleet id that row claims, exactly
        # as resolve_targets() does for the live proxy -- a bare
        # cands.get(m, []) would miss it entirely, since `cands` is keyed by
        # fleet id, never by public_id.
        if m in FLEET_ROLES:
            pool = role_pairs(m)
        else:
            row = _catalogue_row_for(m)
            fids = _row_fleet_ids(row) if row is not None else [m]
            pool = [(c, fid) for fid in fids
                   for c in _routes_cache["cands"].get(fid, [])]
        ranked = await _score_host_model_pairs(
            pool, role="worker", fleet_role=m if m in FLEET_ROLES else "",
            need_ctx=need_ctx)
        for cand, mid in ranked:
            hname = cand or HOST_NAME
            if hname in taken:
                continue
            taken.add(hname)
            targets.append({
                "cand": cand,
                "host": hname,
                "model": mid,
                # Sized against what this host is doing RIGHT NOW, not its
                # raw capacity: a host already carrying load -- a live
                # request, or another batch's own workers -- gets fewer of
                # THIS batch's workers, so a fresh batch never piles its
                # whole worker count onto a box a concurrent caller is also
                # using.
                "workers": max(1, min(8, int(cap.get((cand, mid), 1))
                                      - _inflight.get(hname, 0))),
                "tps": (await tps_for(mid)).get(hname) or _spec_speed(hname),
            })
    return targets


def _batch_flush(bid: int, state: dict, status: str | None = None,
                 message: str | None = None) -> None:
    sets = "updated_at=?, done=?, failed=?"
    args: list[Any] = [now(), state["done"], state["failed"]]
    if status:
        sets += ", status=?"
        args.append(status)
    if message is not None:
        sets += ", message=?"
        args.append(message)
    args.append(bid)
    db_exec("UPDATE batches SET " + sets + " WHERE id=?", args)


async def _batch_run(bid: int, models: list[str], key: dict,
                     skip: set[int] | None = None,
                     counts: dict | None = None,
                     no_fallback: bool = False) -> None:
    """The dispatcher: one shared queue, N workers per serving host."""
    in_path, out_path = _batch_paths(bid)
    state = _batch_live[bid] = {
        "done": (counts or {}).get("done", 0),
        "failed": (counts or {}).get("failed", 0),
        "remaining": 0, "started": time.time(), "targets": [],
    }
    try:
        lines = await asyncio.to_thread(in_path.read_text)
    except OSError as exc:
        _batch_flush(bid, state, "error", "input file unreadable: " + str(exc))
        _batch_live.pop(bid, None)
        return
    queue: asyncio.Queue = asyncio.Queue()
    for idx, line in enumerate(lines.splitlines()):
        if skip and idx in skip:
            continue
        try:
            queue.put_nowait((idx, json.loads(line)))
        except json.JSONDecodeError:
            continue
    state["remaining"] = queue.qsize()
    if not state["remaining"]:
        _batch_flush(bid, state, "done")
        _batch_live.pop(bid, None)
        return

    targets = await _batch_targets(models, no_fallback, _batch_need_ctx(lines))
    if not targets:
        _batch_flush(bid, state, "error",
                     "no host in the fleet serves any of: " + ", ".join(models))
        _batch_live.pop(bid, None)
        return
    state["targets"] = [{k: t[k] for k in ("host", "model", "workers")}
                        for t in targets]

    attempts: dict[int, int] = {}
    out_lock = asyncio.Lock()
    out_fh = await asyncio.to_thread(open, out_path, "a", encoding="utf-8")
    last_flush = time.time()

    async def record(idx: int, ok: bool, status: int, model: str, host: str,
                     ms: int, body: Any) -> None:
        nonlocal last_flush
        line = json.dumps({"i": idx, "ok": ok, "status": status, "model": model,
                           "host": host, "ms": ms, "body": body},
                          ensure_ascii=False)
        async with out_lock:
            await asyncio.to_thread(lambda: (out_fh.write(line + "\n"),
                                             out_fh.flush()))
            state["done" if ok else "failed"] += 1
            state["remaining"] -= 1
            if time.time() - last_flush > 3:
                last_flush = time.time()
                _batch_flush(bid, state)

    async def worker(tgt: dict) -> None:
        hname = tgt["host"]
        while state["remaining"] > 0 and bid not in _batch_cancel:
            if host_cooling(hname):
                # This host just failed something and is sitting out its
                # cooldown -- wait rather than pulling the very next queued
                # item (which may be the one it just failed) straight back
                # onto it. A different host's worker is free to take it in
                # the meantime; this one resumes once the cooldown lapses.
                await asyncio.sleep(0.5)
                continue
            try:
                idx, req = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            body = dict(req)
            body["model"] = tgt["model"]
            body["stream"] = False
            untrack = _track(hname)
            t0 = time.time()
            resident = tgt["model"] in _routes_cache.get("running", {}).get(hname, set())
            try:
                gen_est = min(1024, int(body.get("max_tokens") or 0)) or 256
            except (TypeError, ValueError):
                gen_est = 256
            deadline = _ttfb_deadline(estimate_prompt_tokens(body), resident) \
                + gen_est / 3.0
            status, rbody, raw = 599, {"error": {"message": "no attempt made"}}, b""
            try:
                r = await _post_chat(tgt["cand"], json.dumps(body).encode(), deadline)
                status, raw = r.status_code, r.content
                try:
                    rbody = r.json()
                except ValueError:
                    rbody = {"error": {"message": r.text[:500]}}
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                # The box is unreachable, which says nothing about the item:
                # hand it back uncounted and retire this worker.
                _mark_host_down(hname, COOLDOWN_CONNECT, type(exc).__name__)
                untrack()
                await queue.put((idx, req))
                return
            except httpx.TimeoutException:
                # No complete answer within the deadline -- a stall, handled
                # like a connect failure: hand the item back and retire this
                # worker rather than looping on a box making no progress.
                _mark_host_down(hname, COOLDOWN_STALL,
                                "no answer for " + tgt["model"] + " within "
                                + str(int(deadline)) + "s")
                untrack()
                await queue.put((idx, req))
                return
            except Exception as exc:  # noqa: BLE001
                status, rbody = 599, {"error": {"message": str(exc)[:300]}}
            # untrack() is deliberately NOT in a finally above: a
            # ctx_too_long retry below reuses this SAME tracked window
            # rather than closing and reopening it, so a concurrent request
            # never sees this host as idle mid-retry -- exactly the gap
            # _track-at-ask-time exists to close. Every path below either
            # falls through to the untrack() just past this block, or
            # (ConnectError/TimeoutException above) already called it and
            # returned.
            kind = _classify_upstream_failure(status, "chat/completions", raw)
            if kind == "ctx_too_long" and get_public_settings().get(
                    "ctx_retry_live", True):
                fit_to = _fit_after_ctx_overflow(raw, estimate_prompt_tokens(body))
                retry_body = None
                if fit_to:
                    try:
                        retry_body = apply_ctx_limit(dict(body), fit_to)
                    except HTTPException:
                        retry_body = None
                if retry_body is not None:
                    try:
                        r2 = await _post_chat(
                            tgt["cand"], json.dumps(retry_body).encode(), deadline)
                    except (httpx.ConnectError, httpx.ConnectTimeout,
                            httpx.TimeoutException, httpx.HTTPError):
                        r2 = None
                    if r2 is not None and r2.status_code < 400:
                        remember_model_ctx({(tgt["cand"], tgt["model"]): fit_to})
                        status, raw = r2.status_code, r2.content
                        try:
                            rbody = r2.json()
                        except ValueError:
                            rbody = {"error": {"message": r2.text[:500]}}
                        kind = "ok"
            untrack()
            ms = int((time.time() - t0) * 1000)
            if status >= 500 or kind in ("busy", "ctx_too_long"):
                # A box that cannot serve this item -- whether it broke, is
                # busy, or (with or without a same-box fitted retry, per
                # ctx_retry_live) has proven its window too small -- is not
                # "the request is bad", it is "not on THIS box". Cooled down
                # BEFORE requeuing either way, so host_cooling() below always
                # gates a sibling worker on the same host from immediately
                # resending the identical request that just failed here,
                # regardless of whether ctx_retry_live engaged a retry.
                if kind == "busy":
                    _mark_host_down(hname, _busy_cooldown_seconds(),
                                    "HTTP " + str(status) + " for " + tgt["model"])
                elif kind == "ctx_too_long":
                    _mark_host_down(hname, COOLDOWN_MIDSTREAM,
                                    "ctx_too_long for " + tgt["model"])
                else:
                    _mark_host_down(hname, COOLDOWN_UPSTREAM_5XX,
                                    "HTTP " + str(status) + " for " + tgt["model"])
                attempts[idx] = attempts.get(idx, 0) + 1
                if attempts[idx] < 3:
                    # The cooldown just set is the backoff: host_cooling()
                    # above keeps this worker (and any sibling on the same
                    # host) off the queue until it lapses, so the requeued
                    # item is not immediately resent to the box that just
                    # failed it. A single-host batch still eventually gives
                    # up and records the failure once attempts are exhausted.
                    await queue.put((idx, req))
                    continue
            ok = status < 400
            record_usage(key, tgt["model"], "/v1/batches", False, status,
                         (rbody or {}).get("usage") if isinstance(rbody, dict)
                         else None, None, ms, host=hname)
            await record(idx, ok, status, tgt["model"], hname, ms, rbody)

    try:
        await asyncio.gather(*(worker(t) for t in targets
                               for _ in range(t["workers"])))
        if bid in _batch_cancel:
            _batch_flush(bid, state, "cancelled",
                         str(state["remaining"]) + " request(s) never ran")
            return
        # Workers retire when their host dies; whatever they handed back has
        # no one left to serve it and is recorded as failed rather than lost.
        while state["remaining"] > 0:
            try:
                idx, _ = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            await record(idx, False, 502, "", "", 0,
                         {"error": {"message": "no reachable host left"}})
        msg = (str(state["failed"]) + " of " +
               str(state["done"] + state["failed"]) + " failed"
               if state["failed"] else None)
        _batch_flush(bid, state,
                     "error" if state["done"] == 0 and state["failed"] else "done",
                     msg)
    except asyncio.CancelledError:
        _batch_flush(bid, state, "cancelled", "gateway shutting down")
        raise
    except Exception as exc:  # noqa: BLE001
        _batch_flush(bid, state, "error", str(exc)[:300])
    finally:
        try:
            await asyncio.to_thread(out_fh.close)
        except Exception:  # noqa: BLE001
            pass
        _batch_cancel.discard(bid)
        _batch_live.pop(bid, None)
        _batch_tasks.pop(bid, None)


def _batch_status(row: dict) -> dict:
    out = dict(row)
    try:
        out["models"] = json.loads(row.get("models") or "[]")
    except (json.JSONDecodeError, TypeError):
        out["models"] = []
    live = _batch_live.get(int(row["id"]))
    if live:
        out.update(done=live["done"], failed=live["failed"],
                   targets=live["targets"])
        elapsed = max(0.001, time.time() - live["started"])
        finished = live["done"] + live["failed"]
        out["elapsed_s"] = int(elapsed)
        if finished:
            rate = finished / elapsed
            out["rate"] = round(rate, 2)
            out["eta_s"] = int(live["remaining"] / rate) if rate > 0 else None
    return out


async def batch_submit(key: dict, payload: dict,
                       request: Request | None = None) -> dict:
    reqs = payload.get("requests")
    if not isinstance(reqs, list) or not reqs:
        raise HTTPException(400, "expected {requests: [chat bodies...]}")
    if len(reqs) > 100000:
        raise HTTPException(400, "at most 100,000 requests per batch")
    models = payload.get("models")
    if not isinstance(models, list):
        models = [payload.get("model")]
    models = [str(m).strip() for m in models if str(m or "").strip()]
    if not models:
        raise HTTPException(400, "name a model (or models) for the batch")

    agent = get_agent(int(key["id"]))
    if agent:
        if agent.get("force_model"):
            models = [str(agent["force_model"])]
        try:
            allowed = json.loads(agent.get("allowed_models") or "[]")
        except json.JSONDecodeError:
            allowed = []
        if allowed:
            refused = [m for m in models if resolve_model_id(m) not in allowed]
            if refused:
                raise HTTPException(403, "model(s) not allowed for this key: "
                                    + ", ".join(refused))

    # The opt-out (requirement 2) has no per-item header on this surface, so
    # it is decided ONCE, up front, from the submission itself -- a batch
    # never re-checks it per dequeued request.
    no_fallback = _no_fallback_requested(request, payload)
    need_ctx = 0
    for req in reqs:
        if isinstance(req, dict):
            need_ctx = max(need_ctx, estimate_prompt_tokens(req))
    targets = await _batch_targets(models, no_fallback, need_ctx)
    if not targets:
        raise HTTPException(404, "no host in the fleet serves any of: "
                            + ", ".join(models))

    options = payload.get("options") if isinstance(payload.get("options"), dict) else {}
    lines = []
    for i, req in enumerate(reqs):
        if not isinstance(req, dict) or not isinstance(req.get("messages"), list):
            raise HTTPException(400, "request " + str(i)
                                + " is not a chat body with messages")
        body = dict(req)
        for k, v in options.items():
            body.setdefault(k, v)
        if agent:
            body = apply_agent(dict(agent, force_model="", allowed_models="[]"),
                               body)
        body.pop("model", None)  # the dispatcher assigns models per host
        lines.append(json.dumps(body, ensure_ascii=False))

    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    bid = db_exec(
        "INSERT INTO batches(created_at,updated_at,key_id,key_name,label,"
        "models,status,total) VALUES (?,?,?,?,?,?,?,?)",
        (now(), now(), key.get("id"), key.get("name"),
         str(payload.get("label", ""))[:120], json.dumps(models), "running",
         len(lines)),
    )
    in_path, _ = _batch_paths(bid)
    await asyncio.to_thread(in_path.write_text, "\n".join(lines) + "\n",
                            "utf-8")
    _batch_tasks[bid] = asyncio.create_task(
        _batch_run(bid, models, {"id": key.get("id"), "name": key.get("name")},
                  no_fallback=no_fallback))
    return {"id": bid, "status": "running", "total": len(lines),
            "targets": [{k: t[k] for k in ("host", "model", "workers")}
                        for t in targets]}


async def resume_orphaned_batches() -> None:
    """A restart kills the dispatcher tasks but the spooled requests and the
    already-appended results both survive it; pick each running batch back up
    where its output file ends."""
    for row in db_query("SELECT * FROM batches WHERE status='running'"):
        bid = int(row["id"])
        in_path, out_path = _batch_paths(bid)
        if not in_path.exists():
            db_exec("UPDATE batches SET status='error', message=?, "
                    "updated_at=? WHERE id=?",
                    ("input spool lost across restart", now(), bid))
            continue
        done_idx: set[int] = set()
        counts = {"done": 0, "failed": 0}
        if out_path.exists():
            for line in out_path.read_text(encoding="utf-8").splitlines():
                try:
                    d = json.loads(line)
                    done_idx.add(int(d["i"]))
                    counts["done" if d.get("ok") else "failed"] += 1
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
        try:
            models = json.loads(row.get("models") or "[]")
        except json.JSONDecodeError:
            models = []
        _batch_tasks[bid] = asyncio.create_task(_batch_run(
            bid, models, {"id": row["key_id"], "name": row["key_name"]},
            done_idx, counts))
        log.info("batch %d resumed: %d of %d already answered",
                 bid, len(done_idx), int(row["total"] or 0))


def _batch_owned(bid: int, key: dict) -> dict:
    rows = db_query("SELECT * FROM batches WHERE id=?", (bid,))
    if not rows:
        raise HTTPException(404, "no such batch")
    if rows[0].get("key_id") != key.get("id"):
        raise HTTPException(403, "that batch belongs to a different key")
    return rows[0]


async def v1_batches(path: str, request: Request, key: dict):
    """The /v1/batches surface, dispatched from the catch-all proxy."""
    parts = path.split("/")
    if path == "batches":
        if request.method == "POST":
            try:
                payload = json.loads(await request.body())
            except json.JSONDecodeError:
                raise HTTPException(400, "body must be JSON")
            if not isinstance(payload, dict):
                raise HTTPException(400, "body must be an object")
            return JSONResponse(await batch_submit(key, payload, request))
        if request.method == "GET":
            rows = db_query(
                "SELECT id,created_at,updated_at,label,models,status,total,"
                "done,failed,message FROM batches WHERE key_id=? "
                "ORDER BY id DESC LIMIT 50", (key.get("id"),))
            return JSONResponse({"batches": [_batch_status(r) for r in rows]})
        raise HTTPException(405, "GET or POST")
    try:
        bid = int(parts[1])
    except (IndexError, ValueError):
        raise HTTPException(404, "no such batch")
    row = _batch_owned(bid, key)
    verb = parts[2] if len(parts) > 2 else ""
    if not verb and request.method == "GET":
        return JSONResponse(_batch_status(row))
    if verb == "results" and request.method == "GET":
        _, out_path = _batch_paths(bid)
        if not out_path.exists():
            return PlainTextResponse("", media_type="application/x-ndjson")
        return FileResponse(out_path, media_type="application/x-ndjson")
    if verb == "cancel" and request.method == "POST":
        _batch_cancel.add(bid)
        return JSONResponse({"cancelling": bid})
    raise HTTPException(404, "unknown batch operation")


# ---- teams ----

TEAM_TOOL = {
    "type": "function",
    "function": {
        "name": "spawn_subagents",
        "description": (
            "Run independent sub-agent tasks in parallel on the compute "
            "fleet. Each task is handled by a separate worker model with no "
            "memory of this conversation or of the other tasks, so every "
            "prompt must be complete and self-contained. Results come back "
            "as JSON, one entry per task, in the order given."),
        "parameters": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "the complete instruction for "
                                               "this worker"},
                            "model": {
                                "type": "string",
                                "description": "optional: a specific worker "
                                               "model from the roster"},
                        },
                        "required": ["prompt"],
                    },
                },
            },
            "required": ["tasks"],
        },
    },
}


def get_team(key_id: int) -> dict | None:
    rows = db_query(
        "SELECT * FROM teams WHERE key_id=? AND enabled=1 "
        "AND archived_at IS NULL", (key_id,))
    return rows[0] if rows else None


async def _run_subagents(key: dict, team: dict, tasks: list[dict],
                         pub_key: dict | None = None,
                         no_fallback: bool = False) -> list[dict]:
    try:
        roster = [str(w) for w in json.loads(team.get("worker_models") or "[]")
                  if str(w).strip()]
    except json.JSONDecodeError:
        roster = []
    if not roster:
        roster = [str(team.get("primary_model") or "")]
    sem = asyncio.Semaphore(max(1, int(team.get("max_workers") or 4)))
    wprompt = str(team.get("worker_prompt") or "").strip()
    # A Fleet Pass team's ctx_limit is the CONVERSATION cap on the primary;
    # each worker turn is a one-shot task with no history, so it is bounded
    # to half that budget rather than the whole thing -- but never below what
    # apply_ctx_limit can accept at all: a budget at or under the completion
    # floor rejects EVERY task (est >= budget - PUBLIC_MIN_COMPLETION is
    # always true), which would turn a small team key's every spawn call into
    # a 413. A tiny team's workers simply share the whole window instead of
    # half of it.
    team_ctx = _norm_limit(team.get("ctx_limit"))
    worker_budget = max(team_ctx // 2, min(team_ctx, PUBLIC_MIN_COMPLETION * 4)) \
        if team_ctx else None

    async def one(i: int, t: dict) -> dict:
        model = str(t.get("model") or "").strip()
        if model not in roster:
            model = roster[0]
        # Same fallback (contract 1.9c and requirement 2) as the primary
        # above, role='worker': a cold/unreachable/saturated worker model
        # must not just fail the whole round -- for any team key, not only
        # a Fleet Pass one.
        if not no_fallback:
            req_row = _catalogue_row_for(model)
            if req_row:
                sub = await pick_fallback(req_row, "worker", get_public_settings())
                if sub:
                    served = str(sub["public_id"])
                    _note_fallback(int(key["id"]), model, served)
                    model = served
        msgs = ([{"role": "system", "content": wprompt}] if wprompt else [])
        msgs.append({"role": "user", "content": str(t.get("prompt") or "")})
        body = {"model": model, "messages": msgs,
               "max_tokens": min(4096, worker_budget) if worker_budget else 4096}
        if worker_budget:
            try:
                body = apply_ctx_limit(body, worker_budget)
            except HTTPException as exc:
                detail = exc.detail if isinstance(exc.detail, dict) else {}
                msg = ((detail.get("error") or {}).get("message")
                      or "prompt is too long for this team's per-worker context limit")
                return {"task": i, "model": model, "host": "", "ok": False,
                       "result": msg, "usage": None}
        async with sem:
            # A worker's own context reduction is not disclosed to the client:
            # the client never sees worker turns, only the primary's woven
            # answer. It is still ENFORCED -- a worker sent more prompt than
            # its box can hold would have it silently truncated.
            status, resp, host, _granted = await fleet_chat(
                key, body, "/v1/team-worker", worker_budget, role="worker")
        content = ""
        truncated = False
        usage = resp.get("usage") if isinstance(resp, dict) else None
        if isinstance(resp, dict):
            try:
                choice = resp["choices"][0]
                content = choice["message"].get("content") or ""
                truncated = choice.get("finish_reason") == "length"
            except (KeyError, IndexError, TypeError, AttributeError):
                content = json.dumps(resp.get("error") or resp)[:500]
        out = {"task": i, "model": model, "host": host,
               "ok": status < 400, "result": content, "usage": usage}
        if truncated:
            # The primary weaves these results into its answer sight unseen; a
            # worker cut off mid-thought must say so in the JSON it reads, or
            # the cut-off text gets woven in as if it were complete.
            out["truncated"] = True
        return out

    results = await asyncio.gather(*(one(i, t) for i, t in enumerate(tasks)),
                                   return_exceptions=True)
    # One worker blowing up (a bug, a cancelled socket) degrades ONE task,
    # not the whole round of finished sibling work.
    return [r if isinstance(r, dict) else
            {"task": i, "model": "", "host": "", "ok": False,
             "result": "worker failed: " + type(r).__name__, "usage": None}
            for i, r in enumerate(results)]


# One SSE frame's worth of replayed content. Some proxies and stream parsers
# cap a single event line, and a woven team answer can run to hundreds of KB;
# re-chunking costs nothing and never trips those ceilings.
_SSE_PIECE = 4096


def _sse_once(resp: dict) -> StreamingResponse:
    """A team turn cannot stream while the tool loop is still running, so a
    client that asked for SSE gets the finished answer as well-formed streamed
    chunks instead of a protocol error. tool_calls ride along exactly as the
    non-streaming path forwards them -- dropping them here cost a tool-calling
    client its whole turn -- indexed per entry the way streamed tool_calls
    are."""
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    base = {
        "id": resp.get("id"), "object": "chat.completion.chunk",
        "created": resp.get("created"), "model": resp.get("model"),
    }
    content = msg.get("content") or ""
    first: dict[str, Any] = {"role": "assistant",
                             "content": content[:_SSE_PIECE]}
    if msg.get("tool_calls"):
        first["tool_calls"] = [{"index": i, **tc}
                               for i, tc in enumerate(msg["tool_calls"])]
    deltas = [first] + [{"content": content[i:i + _SSE_PIECE]}
                        for i in range(_SSE_PIECE, len(content), _SSE_PIECE)]
    tail = {
        **base,
        "choices": [{"index": 0, "delta": {},
                     "finish_reason": choice.get("finish_reason") or "stop"}],
        "usage": resp.get("usage"),
    }

    async def gen():
        for d in deltas:
            chunk = {**base, "choices": [{"index": 0, "finish_reason": None,
                                          "delta": d}]}
            yield ("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode()
        yield ("data: " + json.dumps(tail, ensure_ascii=False) + "\n\n").encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


async def team_orchestrate(key: dict, team: dict, payload: dict,
                           stream: bool, started: float,
                           request: Request | None = None):
    """The server-side agentic loop a team key buys.

    The primary model leads the conversation and is handed spawn_subagents;
    every round in which it calls that tool, the gateway fans the tasks out
    across the fleet, appends the results, and asks again. The client sees
    one ordinary chat completion -- the crew is invisible except for the
    x_team accounting stapled to the answer.

    `request` is the client connection, when there is a live one to watch: a
    caller that times out and retries must not leave the first orchestration
    burning primary and worker rounds for nobody."""
    try:
        roster = [str(w) for w in json.loads(team.get("worker_models") or "[]")
                  if str(w).strip()]
    except json.JSONDecodeError:
        roster = []
    primary = str(team.get("primary_model") or "").strip() \
        or str(payload.get("model") or "")

    # Fallback (contract 1.9c and requirement 2) applies to every team key,
    # not only a Fleet Pass one -- a team's primary is exactly the workload
    # ("multi-model teams touching the largest models") the disclosed
    # substitution feature exists to protect. Decided once, up front: every
    # round in this loop, and every worker spawn.round, talks to the same
    # served model(s). The opt-out is checked once here and carried into
    # every worker round below.
    pub_key = public_key_for(int(key["id"]))
    no_fallback = _no_fallback_requested(request, payload)
    fallback: dict[str, str] | None = None
    if not no_fallback:
        req_row = _catalogue_row_for(primary)
        if req_row:
            sub = await pick_fallback(req_row, "primary", get_public_settings())
            if sub:
                served_primary = str(sub["public_id"])
                fallback = {"requested": primary, "served": served_primary}
                primary = served_primary

    payload = dict(payload)
    payload["model"] = primary
    payload["stream"] = False
    # Asked before the strip below and before the loop appends a single round
    # of its own: `payload` is a shallow copy, so `messages` is still the
    # caller's list, and by the time the answer is assembled it holds this
    # turn's work as well as the conversation it arrived with.
    continued = already_replied(payload.get("messages"))
    # The client resends what it was shown, past notice banners included --
    # they are the gateway's own text, not the model's, and they pollute both
    # the window estimate and the model's view of its own past turns.
    strip_fleet_notices(payload.get("messages"))
    sysline = str(team.get("system_prompt") or "").strip()
    sysline = (sysline + "\n\n" if sysline else "") + (
        "You lead a team of worker models on a local compute fleet. For work "
        "that splits into independent pieces, call spawn_subagents with one "
        "complete, self-contained prompt per piece -- workers share no memory "
        "with you or with each other. Worker models available: "
        + (", ".join(roster) or primary)
        + ". Weave their results into your own answer; the user only sees you.")
    msgs = payload.get("messages")
    payload["messages"] = [{"role": "system", "content": sysline}] \
        + (msgs if isinstance(msgs, list) else [])
    tools = payload.get("tools") if isinstance(payload.get("tools"), list) else []
    payload["tools"] = tools + [TEAM_TOOL]

    rounds, sub_calls, sub_fail = 0, 0, 0
    max_rounds = max(1, int(team.get("max_rounds") or 6))
    team_ctx = _norm_limit(team.get("ctx_limit"))
    # Every round talks to the primary under /v1/team-primary and every
    # worker call under /v1/team-worker -- both excluded from a key's
    # request budget (require_api_key filters them out), because the crew
    # doing its own work is not a second request from the client. Exactly
    # ONE client-facing row is recorded below, after the loop, with every
    # round's and every worker's usage summed into it.
    usage_totals: dict[str, int] = {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
    }
    primary_host = ""
    # Set when the box answering for the primary cannot hold the window this
    # team key was issued for. Latched across rounds: if any round had to be
    # shortened, the caller is told, because that round is in the answer.
    ctx_cut: dict[str, int] | None = None
    # One retry when the finishing turn is cut off by its own completion
    # budget -- see the finish_reason check below.
    length_retried = False
    # Orchestration is active work: the fleet view must be able to see and
    # abort it, and a caller that hung up must stop costing worker rounds --
    # the same contract the single-key loop honours with its own job.
    job = _job_open(
        "inference",
        what="team:" + str(team.get("name") or primary),
        detail="/v1/chat/completions · team"
        + (" · " + str(key.get("name")) if key.get("name") else ""),
        stop=asyncio.Event(),
    )
    watcher = _watch_disconnect(request, job) if request is not None else None
    try:
        while True:
            if job["aborted"]:
                record_usage(key, "team", "/v1/chat/completions", stream, 499,
                             None, None, int((time.time() - started) * 1000),
                             host=primary_host)
                return JSONResponse(ABORTED_BODY, status_code=499)
            if team_ctx:
                try:
                    payload = apply_ctx_limit(payload, team_ctx)
                except HTTPException as exc:
                    # The conversation can outgrow the key's window on round
                    # N, after workers already did real, metered work. The
                    # caller gets the same flat, numbered 413 the single-key
                    # path returns -- letting this propagate handed them a
                    # framework-wrapped {"detail": ...} their SDK cannot read,
                    # and skipped the metering row entirely.
                    record_usage(key, "team", "/v1/chat/completions", stream,
                                 413, None, None,
                                 int((time.time() - started) * 1000),
                                 host=primary_host)
                    detail = exc.detail if isinstance(exc.detail, dict) else \
                        {"error": {"message": str(exc.detail),
                                   "type": "context_limit"}}
                    return JSONResponse(
                        dict(detail, x_team={
                            "team": team.get("name"), "rounds": rounds,
                            "subagents": sub_calls,
                            "subagent_failures": sub_fail,
                            "primary_host": primary_host}),
                        status_code=413)
            status, resp, host, granted = await fleet_chat(
                key, payload, "/v1/team-primary", team_ctx, role="primary")
            primary_host = host or primary_host
            if primary_host:
                job["host"] = primary_host
            if team_ctx and granted and granted < team_ctx:
                ctx_cut = {"requested": int(team_ctx), "granted": int(granted)}
            if isinstance(resp, dict):
                _sum_usage(usage_totals, resp.get("usage"))
            if status >= 400 or not isinstance(resp, dict):
                # usage=None: every token was already metered on its own
                # /v1/team-primary or /v1/team-worker row by fleet_chat -- this
                # row exists to count the client's ONE request (and its latency),
                # not to bill the tokens a second time against a token budget.
                record_usage(key, "team", "/v1/chat/completions", stream, status,
                             None, None, int((time.time() - started) * 1000),
                             host=primary_host)
                return JSONResponse(
                    resp if isinstance(resp, dict) else
                    {"error": {"message": str(resp)[:300]}}, status_code=status)
            choice = (resp.get("choices") or [{}])[0]
            m = choice.get("message") or {}
            calls = m.get("tool_calls") or []
            if (choice.get("finish_reason") == "length" and not calls
                    and team_ctx and not length_retried
                    and isinstance(payload.get("max_tokens"), int)):
                # The turn ran out of completion budget before it finished --
                # a thinking model's reasoning spends the same budget as the
                # answer, and a truncated finishing turn completes nothing.
                # The caller cannot fix this from outside (a retry meets the
                # same ceiling), so retry ONCE with double the room; the
                # apply_ctx_limit at the top of the loop re-caps it to what
                # the window actually allows.
                length_retried = True
                payload["max_tokens"] = int(payload["max_tokens"]) * 2
                continue
            spawn = [c for c in calls if isinstance(c, dict)
                     and (c.get("function") or {}).get("name") == "spawn_subagents"]
            # Client-defined tools stay the client's: a round that calls any
            # tool this gateway does not own is returned -- minus the spawn
            # calls, which are the GATEWAY'S tool. A client handed a
            # spawn_subagents call it never defined has no handler for it and,
            # under the tool-calling contract, no legal way to answer its id:
            # leaking one corrupts the client's own tool loop.
            if not spawn or len(spawn) != len(calls) or rounds >= max_rounds:
                if spawn:
                    kept = [c for c in calls if c not in spawn]
                    if kept:
                        m["tool_calls"] = kept
                    else:
                        # All calls were the gateway's (an all-spawn round at
                        # max_rounds): with no tool_calls left, a
                        # finish_reason of "tool_calls" would send the client
                        # looking for an array that is not there.
                        m.pop("tool_calls", None)
                        if choice.get("finish_reason") == "tool_calls":
                            choice["finish_reason"] = "stop"
                resp["x_team"] = {"team": team.get("name"), "rounds": rounds,
                                  "subagents": sub_calls,
                                  "subagent_failures": sub_fail,
                                  "primary_host": primary_host}
                headers: dict[str, str] = {}
                if fallback or ctx_cut:
                    # Same notice/header/x_fleet treatment the single-key /v1
                    # path gives a substitution or a shortened window
                    # (contract 1.9c) -- a team caller deserves to know just
                    # as much as a single one.
                    notice, xf, hdrs = public_notices(
                        fallback, ctx_cut, primary_host, get_public_settings())
                    resp["x_fleet"] = xf
                    headers.update(hdrs)
                    if not (_wants_structured(payload) or continued):
                        prepend_notice(resp, notice)
                if fallback:
                    _note_fallback(int(key["id"]), fallback["requested"],
                                   fallback["served"])
                # The client does get the true total: the crew's summed usage
                # replaces the last round's in the response. Metered here with
                # usage=None for the reason above -- the per-round rows already
                # carry every token, so a token budget must not see them twice.
                resp["usage"] = dict(usage_totals)
                record_usage(key, "team", "/v1/chat/completions", stream, status,
                             None, None, int((time.time() - started) * 1000),
                             host=primary_host,
                             fallback_from=(fallback["requested"] if fallback else ""))
                if stream:
                    out = _sse_once(resp)
                    for hk, hv in headers.items():
                        out.headers[hk] = hv
                    return out
                return JSONResponse(resp, status_code=status, headers=headers or None)
            rounds += 1
            payload["messages"] = list(payload["messages"]) + [m]
            for call in spawn:
                try:
                    args = json.loads((call.get("function") or {})
                                      .get("arguments") or "{}")
                    tasks = args.get("tasks") or []
                except (json.JSONDecodeError, TypeError):
                    tasks = []
                tasks = [t for t in tasks if isinstance(t, dict)
                         and str(t.get("prompt") or "").strip()][:32]
                if tasks:
                    results = await _run_subagents(
                        key, team, tasks, pub_key, no_fallback)
                    sub_calls += len(tasks)
                    sub_fail += sum(1 for r in results if not r.get("ok"))
                    for r in results:
                        _sum_usage(usage_totals, r.pop("usage", None))
                else:
                    results = [{"error": "spawn_subagents called with no usable "
                                         "tasks -- each task needs a prompt"}]
                payload["messages"].append({
                    "role": "tool", "tool_call_id": str(call.get("id") or ""),
                    "content": json.dumps({"results": results},
                                          ensure_ascii=False)})
    finally:
        if watcher is not None:
            watcher.cancel()
        _job_close(job)


# ============================================================================
# Fleet Pass: public, auto-approved, rate-limited demo API keys
# ============================================================================
#
# Requested from a public page on the public site (example.org/fleet), issued
# by this hub. A work address on one of the eligible-employer lists gets a key
# by return email with no human in the loop; anything else queues for a
# short manual approve/deny in the Public tab. The catalogue below maps a
# small, curated set of public model names onto this fleet's real
# model/fleet ids -- clients never see a raw fleet id, a host name, or a
# tailnet address. The rest of the enforcement (limits, context caps, the
# fallback notice) lives further down, inside openai_proxy/fleet_chat/
# team_orchestrate, where the ordinary proxy path already runs.

class PublicError(Exception):
    """A /public/api/* error: the flat {"error","code"} shape the Fleet Pass
    contract promises the public site it can forward verbatim. Everywhere else
    in this file an HTTPException's {"detail": ...} wrapping is fine because
    nothing downstream depends on the exact envelope; here something does."""

    def __init__(self, status: int, code: str, message: str | None = None):
        self.status = status
        self.code = code
        self.message = message or code
        super().__init__(self.message)


@app.exception_handler(PublicError)
async def _public_error_handler(request: Request, exc: PublicError) -> JSONResponse:
    return JSONResponse({"error": exc.message, "code": exc.code}, status_code=exc.status)


# What an HTTPException's status means in the OpenAI error vocabulary, for
# clients that branch on `error.type` (prompt the user about a bad key vs.
# back off vs. trim the prompt).
_OPENAI_ERR_TYPE = {401: "authentication_error", 403: "permission_error",
                    413: "context_limit", 429: "rate_limit_error"}


@app.exception_handler(HTTPException)
async def _http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """/v1 and /api speak the OpenAI error envelope, whoever raised: a bare
    `raise HTTPException(401, "missing bearer token")` used to reach an SDK as
    FastAPI's {"detail": ...}, a shape no OpenAI-compatible client can read an
    error.message out of. Everything else (dashboard fetches, admin routes)
    keeps the {"detail": ...} shape static/index.html already reads."""
    p = request.url.path
    if p.startswith("/v1/") or p.startswith("/api/"):
        d = exc.detail
        if isinstance(d, dict) and "error" in d:
            body = d  # already the flat shape (apply_ctx_limit's 413, etc.)
        else:
            body = {"error": {
                "message": str(d),
                "type": _OPENAI_ERR_TYPE.get(
                    exc.status_code,
                    "invalid_request_error" if exc.status_code < 500
                    else "upstream"),
            }}
        return JSONResponse(body, status_code=exc.status_code,
                            headers=getattr(exc, "headers", None))
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code,
                        headers=getattr(exc, "headers", None))


# ---- seeded catalogue -------------------------------------------------

# The 15-row catalogue baked into the binary. Used only when
# gateway/public_seed.json (or LLMSTACK_PUBLIC_SEED) is missing or empty --
# the orchestrator supplies that file, but a fresh checkout without it still
# boots with a usable Fleet Pass.
PUBLIC_MODELS_SEED: list[dict] = [
    {"public_id": "gemma4-31b-qat", "family": "Gemma", "name": "Gemma 4 31B (QAT)", "vendor": "Google",
     "arch": "dense", "params_b": 31, "active_b": 31,
     "fleet_ids": ["gemma4-31b-qat"], "allow_primary": 1, "allow_worker": 0,
     "ctx_max": 131072, "ctx_default": 32768, "sort": 10,
     "description": "Google's largest open Gemma 4, quantisation-aware trained so 4-bit costs it little. Even-tempered prose, image input, dependable structured output -- and Apache 2.0 since Gemma 4."},
    {"public_id": "gemma4-26b-a4b", "family": "Gemma", "name": "Gemma 4 26B-A4B (QAT)", "vendor": "Google",
     "arch": "moe", "params_b": 26, "active_b": 4,
     "fleet_ids": ["gemma-4-26b", "gemma4:26b"], "allow_primary": 1, "allow_worker": 1,
     "ctx_max": 131072, "ctx_default": 32768, "sort": 20,
     "description": "Mixture-of-experts Gemma 4: 26B stored, 4B active. Cheap per token, though slower in practice than the active count suggests."},
    {"public_id": "gemma4-12b", "family": "Gemma", "name": "Gemma 4 12B (QAT)", "vendor": "Google",
     "arch": "dense", "params_b": 12, "active_b": 12,
     "fleet_ids": ["gemma4:12b-it-qat"], "allow_primary": 0, "allow_worker": 1,
     "ctx_max": 131072, "ctx_default": 16384, "sort": 30,
     "description": "Dense 12B Gemma 4. Former small-box resident (replaced by the Qwen 3.8 9B distill, 2026-08); still installed on some boxes."},
    {"public_id": "gemma4-e4b", "family": "Gemma", "name": "Gemma 4 E4B", "vendor": "Google",
     "arch": "dense", "params_b": 8, "active_b": 4,
     "fleet_ids": ["gemma4:e4b"], "allow_primary": 0, "allow_worker": 1,
     "ctx_max": 32768, "ctx_default": 16384, "sort": 40,
     "description": "Gemma 4 'effective 4B' -- small, quick, image input included. Good for sub-tasks and anything that has to run on-device."},
    {"public_id": "qwen3.8-27b", "family": "Qwen", "name": "Qwen 3.8 27B", "vendor": "Alibaba Qwen",
     "arch": "dense", "params_b": 27, "active_b": 27,
     "fleet_ids": ["qwen3.8-27b"], "allow_primary": 1, "allow_worker": 0,
     "ctx_max": 262144, "ctx_default": 32768, "sort": 50,
     "description": "Qwen's August 2026 dense flagship, and the one you can actually host: 262k trained context, image and video in, Apache 2.0. The fleet's strongest prose and reasoning model -- and its most verbose."},
    {"public_id": "qwen3.8-flash-next", "family": "Qwen", "name": "Qwen 3.8 Flash-Next", "vendor": "Alibaba Qwen",
     "arch": "moe", "params_b": 125, "active_b": 6,
     "fleet_ids": ["qwen3.8-flash-next"], "allow_primary": 1, "allow_worker": 0,
     # 128k. The 32768 this carried for a few hours on 2026-08-29 was set on a
     # wrong diagnosis: that long contexts were what OOM-killed apu-box-1. They
     # are not, and the KV cache is the cheapest thing this model has. Only 12
     # of its 48 layers hold one -- the rest are linear and log as `filtered`
     # -- so the whole cache is 243 MiB per 32k, about 7.6 MiB per 1k tokens,
     # and it lives in VRAM the box has 45 GiB of going spare. Going 32768 ->
     # 131072 moved VRAM 47.6 -> 50.4 GiB and host RSS not at all, measured
     # across a 112k-token prompt.
     #
     # What actually fills apu-box-1 is one tensor: per_layer_token_embd.weight,
     # 26.8 GiB of iq4_nl that no Vulkan buffer can host (forcing it aborts),
     # against 31 GiB of system RAM. That is a host-memory ceiling and it does
     # not move with ctx_max, so pinning this low bought nothing and cost the
     # window. See hosts/apu-box-1/README.md for the measurements.
     #
     # 262144 is the model's full trained window, verified live on apu-box-1
     # 2026-09-01: loads, and answers a real 30k-token prompt at 145 tok/s
     # prefill with VRAM at 55 of 96 GiB and host RSS flat.
     "ctx_max": 262144, "ctx_default": 65536, "sort": 55,
     "description": "Qwen's preview of the Qwen4 architecture, out 2026-08-26: 125B stored, 6B active, with hybrid linear/sparse attention that keeps a long window cheap. Held at two bits on the one box with the memory for it -- around 28 tokens a second, and new enough that its quirks are still being found."},
    {"public_id": "qwen3.6-35b-a3b", "family": "Qwen", "name": "Qwen 3.6 35B-A3B", "vendor": "Alibaba Qwen",
     "arch": "moe", "params_b": 35, "active_b": 3,
     "fleet_ids": ["qwen3.6-35b"], "allow_primary": 1, "allow_worker": 1,
     "ctx_max": 262144, "ctx_default": 32768, "sort": 60,
     "description": "April 2026 mixture of experts: 35B stored, 3B active, so it answers at a small model's pace on any box that can hold it. Apache 2.0, 262k context."},
    {"public_id": "qwen3.8-9b-distill", "family": "Qwen", "name": "Qwen 3.8 9B Distill",
     "vendor": "community (empero-ai Qwen 3.8 distill)",
     "arch": "dense", "params_b": 9, "active_b": 9,
     "fleet_ids": ["qwen3.8-9b-distill", "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M", "qwen3.8-9B", "Qwen3.8-9B", "empero-ai-qwen3.8-9b-distill"],
     "allow_primary": 0, "allow_worker": 1,
     "ctx_max": 262144, "ctx_default": 16384, "sort": 65,
     "description": "Qwen 3.8-Max distilled into a dense 9B by the community (empero-ai) -- not an Alibaba release. Resident default on the Apple-silicon and CPU boxes since 2026-08; quick prefill, light on memory."},
    {"public_id": "qwen3.5-4b", "family": "Qwen", "name": "Qwen 3.5 4B", "vendor": "Alibaba Qwen",
     "arch": "dense", "params_b": 4, "active_b": 4,
     "fleet_ids": ["qwen3.5:4b", "qwen3.5-4b"], "allow_primary": 0, "allow_worker": 1,
     "ctx_max": 262144, "ctx_default": 16384, "sort": 70,
     "description": "Small dense Qwen from the 3.5 line. Fast sub-agent work, classification and tool calls -- not the one to reason with."},
    {"public_id": "qwopus3.6-35b", "family": "Qwen", "name": "Qwopus 3.6 35B-A3B",
     "vendor": "community (Qwen 3.6 fine-tune)", "arch": "moe", "params_b": 35, "active_b": 3,
     "fleet_ids": ["qwopus3.6-35b-coder", "Qwopus3.6-35B", "Qwopus3.6-A3B"],
     "allow_primary": 1, "allow_worker": 1, "ctx_max": 262144, "ctx_default": 32768, "sort": 80,
     "description": "Community coding fine-tune of the Qwen 3.6 mixture of experts (35B stored, 3B active) with multi-token prediction. Leans to code and patches."},
    {"public_id": "nemotron3-super-120b", "family": "Nemotron", "name": "Nemotron 3 Super 120B-A12B", "vendor": "NVIDIA",
     "arch": "moe", "params_b": 120, "active_b": 12,
     "fleet_ids": ["nemotron3-super-120b"], "allow_primary": 1, "allow_worker": 0,
     "ctx_max": 131072, "ctx_default": 16384, "sort": 90,
     "description": "NVIDIA's 120B-A12B hybrid Mamba/transformer, built for throughput on long agent runs rather than for benchmark wins. The fleet's largest model: a cold load takes minutes, so expect a fallback when it is not resident."},
    {"public_id": "nemotron3.5-lightning-30b", "family": "Nemotron", "name": "Nemotron 3.5 Lightning 30B-A3B", "vendor": "NVIDIA",
     "arch": "moe", "params_b": 30, "active_b": 3,
     "fleet_ids": ["nemotron3.5-lightning-30b"], "allow_primary": 1, "allow_worker": 1,
     "ctx_max": 131072, "ctx_default": 16384, "sort": 100,
     "description": "August 2026 Nemotron: 30B stored, 3B active, with a thinking phase. Fast, and a good writer."},
    {"public_id": "nemotron-3-nano-4b", "family": "Nemotron", "name": "Nemotron 3 Nano 4B", "vendor": "NVIDIA",
     "arch": "dense", "params_b": 4, "active_b": 4,
     "fleet_ids": ["nemotron-3-nano:4b"], "allow_primary": 0, "allow_worker": 1,
     "ctx_max": 131072, "ctx_default": 16384, "sort": 110,
     "description": "Small dense Nemotron for sub-tasks and tool calls."},
    {"public_id": "nemotron-mini-4b", "family": "Nemotron", "name": "Nemotron Mini 4B", "vendor": "NVIDIA",
     "arch": "dense", "params_b": 4, "active_b": 4,
     "fleet_ids": ["nemotron-mini:4b"], "allow_primary": 0, "allow_worker": 1,
     "ctx_max": 8192, "ctx_default": 8192, "sort": 120,
     "description": "Tiny, instant, 4k context. The floor of the fleet."},
    {"public_id": "muse-glimmer-30b", "family": "Muse", "name": "Muse Glimmer 30B", "vendor": "Meta",
     "arch": "dense", "params_b": 30, "active_b": 30,
     "fleet_ids": ["muse-glimmer-30b"], "allow_primary": 1, "allow_worker": 1,
     "ctx_max": 131072, "ctx_default": 32768, "sort": 130,
     "description": "Meta's dense 30B agent model, distilled from Muse Spark and released Apache 2.0. Excellent tool use for its size and it fits anywhere -- but it hallucinates measurably more than the Qwen of the same size, so not the one to ask for facts."},
    {"public_id": "deepseek-v4-flash", "family": "DeepSeek", "name": "DeepSeek V4 Flash", "vendor": "DeepSeek",
     "arch": "moe", "params_b": 284, "active_b": 13,
     "fleet_ids": ["deepseek-v4-flash"], "allow_primary": 1, "allow_worker": 0,
     "ctx_max": 163840, "ctx_default": 12288, "sort": 140,
     "description": "DeepSeek V4 Flash (0731): 284B stored, 13B active, MIT-licensed, built for long context and agentic work. Runs here at a low bit rate on the one box that can hold it -- the slowest cold load in the fleet."},
    {"public_id": "glm-4.7-flash", "family": "GLM", "name": "GLM-4.7-Flash", "vendor": "Z.ai",
     "arch": "moe", "params_b": 30, "active_b": 3,
     "fleet_ids": ["glm-4.7-flash"], "allow_primary": 1, "allow_worker": 1,
     "ctx_max": 262144, "ctx_default": 32768, "sort": 150,
     "description": "Z.ai's 30B-A3B mixture of experts: capable and efficient at general chat and tool calling for the memory it occupies."},
]

PUBLIC_SEED_PATH = Path(
    os.environ.get("LLMSTACK_PUBLIC_SEED", str(Path(__file__).parent / "public_seed.json"))
)
PUBLIC_DOMAINS_SEED_PATH = Path(
    os.environ.get(
        "LLMSTACK_PUBLIC_DOMAINS_SEED", str(Path(__file__).parent / "public_domains_seed.json")
    )
)

# Suffix rules seeded in code even when the domains file lacks them: a demo
# key program that only auto-approves companies on a hand-maintained list and
# forgets the government is missing the easiest case to get right.
PUBLIC_DOMAINS_BUILTIN: list[dict] = [
    {"domain": ".gov", "company": "U.S. Government", "source": "gov", "rank": None},
    {"domain": ".mil", "company": "U.S. Military", "source": "gov", "rank": None},
]


def load_public_models_seed() -> list[dict]:
    if PUBLIC_SEED_PATH.exists():
        try:
            data = json.loads(PUBLIC_SEED_PATH.read_text())
            if isinstance(data, list) and data:
                return data
        except (OSError, json.JSONDecodeError):
            pass
    return PUBLIC_MODELS_SEED


def load_public_domains_seed() -> list[dict]:
    rows: list[dict] = []
    if PUBLIC_DOMAINS_SEED_PATH.exists():
        try:
            data = json.loads(PUBLIC_DOMAINS_SEED_PATH.read_text())
            if isinstance(data, list):
                rows = [r for r in data if isinstance(r, dict) and r.get("domain")]
        except (OSError, json.JSONDecodeError):
            pass
    return rows + PUBLIC_DOMAINS_BUILTIN


def _public_model_upsert(rec: dict) -> None:
    fleet_ids = rec.get("fleet_ids") or []
    if not isinstance(fleet_ids, list):
        fleet_ids = [fleet_ids]
    db_exec(
        "INSERT INTO public_models(public_id,name,vendor,family,description,fleet_ids,"
        "arch,params_b,active_b,allow_primary,allow_worker,ctx_max,ctx_default,"
        "enabled,sort,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(public_id) DO UPDATE SET "
        "name=excluded.name, vendor=excluded.vendor, family=excluded.family,"
        "description=excluded.description,"
        "fleet_ids=excluded.fleet_ids, arch=excluded.arch, params_b=excluded.params_b,"
        "active_b=excluded.active_b, allow_primary=excluded.allow_primary,"
        "allow_worker=excluded.allow_worker, ctx_max=excluded.ctx_max,"
        "ctx_default=excluded.ctx_default, enabled=excluded.enabled, "
        "sort=excluded.sort, updated_at=excluded.updated_at",
        (
            str(rec["public_id"]).strip(),
            str(rec.get("name", "")),
            str(rec.get("vendor", "")),
            str(rec.get("family", "")).strip(),
            str(rec.get("description", "")),
            json.dumps([str(f) for f in fleet_ids if str(f).strip()]),
            str(rec.get("arch", "dense")),
            float(rec.get("params_b", 0) or 0),
            float(rec.get("active_b", 0) or 0),
            int(bool(rec.get("allow_primary", 0))),
            int(bool(rec.get("allow_worker", 0))),
            int(rec.get("ctx_max", 16384) or 16384),
            int(rec.get("ctx_default", 8192) or 8192),
            int(bool(rec.get("enabled", 1))),
            int(rec.get("sort", 0) or 0),
            now(),
        ),
    )


def seed_public_models(missing_only: bool = False) -> int:
    """Insert catalogue rows from the seed. missing_only=True is the admin
    'reseed' action: it must never clobber a row an admin has since edited,
    so it only inserts public_ids that are not there yet."""
    rows = load_public_models_seed()
    existing = (
        {r["public_id"] for r in db_query("SELECT public_id FROM public_models")}
        if missing_only else set()
    )
    added = 0
    for rec in rows:
        pid = str(rec.get("public_id", "")).strip()
        if not pid or (missing_only and pid in existing):
            continue
        _public_model_upsert(rec)
        added += 1
    return added


# What each catalogue row's ctx_max shipped as BEFORE context ceilings came
# from the hardware. Back then the column was the whole answer -- a flat cap
# per model, mostly 32768 -- and now it is only an operator ceiling ON TOP of
# what the boxes report, seeded to what each model was actually trained for.
# A row still sitting at its old value would pin every machine in the fleet to
# the old flat number, which is the bug this replaced.
_PRE_HARDWARE_CTX_MAX: dict[str, int] = {
    "gemma4-31b-qat": 32768, "gemma4-26b-a4b": 32768, "gemma4-12b": 24576,
    "gemma4-e4b": 16384, "qwen3.8-27b": 32768, "qwen3.6-35b-a3b": 32768,
    "qwen3.5-4b": 16384, "qwopus3.6-35b": 32768,
    # qwen3.8-9b-distill joined the seed 2026-08-24, after ceilings went
    # hardware-driven -- recorded at its first-shipped ctx_max for the same
    # reason as glm-4.7-flash below.
    "qwen3.8-9b-distill": 262144,
    "nemotron3-super-120b": 32768, "nemotron3.5-lightning-30b": 32768,
    "nemotron-3-nano-4b": 16384, "nemotron-mini-4b": 8192,
    "muse-glimmer-30b": 32768, "deepseek-v4-flash": 16384,
    # glm-4.7-flash has no pre-hardware era -- this is the value the first
    # seed that carried it shipped with, recorded so a future raise of its
    # ctx_max still migrates instead of silently pinning every box.
    "glm-4.7-flash": 262144,
    # qwen3.8-flash-next joined 2026-08-28, same reasoning. The note that used
    # to sit here -- "the Vulkan flash-attention path stalls on this
    # architecture" -- was never true of this build. `flash_attn = auto`
    # resolves to `resolve_fused_ops: Flash Attention enabled` on apu-box-1, and
    # the model runs that way today.
    "qwen3.8-flash-next": 196608,
}


def raise_stale_ctx_ceilings() -> int:
    """Lift a row's ctx_max to the seed's trained-context value, but ONLY
    where it still holds the exact number the pre-hardware seed shipped.

    The ordinary reseed refuses to touch an existing row at all, on the sound
    principle that it must not clobber an admin's edit -- which would leave
    every catalogue row capped at the flat ceiling this feature exists to get
    rid of. This threads that needle: an untouched row is migrated, a row
    someone has since set by hand is left exactly as they set it. Runs on
    every boot and is a no-op after the first, because a migrated row no
    longer matches the old value."""
    seeded = {str(r.get("public_id") or ""): int(r.get("ctx_max") or 0)
              for r in load_public_models_seed()}
    changed = 0
    for row in db_query("SELECT public_id, ctx_max FROM public_models"):
        pid = str(row["public_id"])
        old, new = _PRE_HARDWARE_CTX_MAX.get(pid), seeded.get(pid)
        if not old or not new or new <= old or int(row["ctx_max"]) != old:
            continue
        db_update("UPDATE public_models SET ctx_max=?, updated_at=? "
                  "WHERE public_id=?", (new, now(), pid))
        changed += 1
    if changed:
        public_catalogue(force=True)
        log.info("raised %d stale public ctx_max ceilings", changed)
    return changed


# ---- model families -------------------------------------------------------
#
# `vendor` says who trained the weights; `family` says what a visitor calls
# the thing. They are usually the same word and sometimes emphatically not:
# the 9B distill and Qwopus are community models built on Qwen, and a request
# form that files them under "community" has hidden them from the person
# looking for a Qwen. So family is its own column, seeded per row and editable
# on the Public tab, and it is what the public page groups by.

_FAMILY_HEAD = re.compile(r"[A-Za-z]+")


def derive_family(row: dict) -> str:
    """The family a row belongs to when nobody has told it one.

    Reads the display name first, because that is where the branding is
    spelled the way someone would say it out loud -- "GLM-4.7-Flash" is GLM,
    "Nemotron 3 Super 120B-A12B" is Nemotron. Only ever a fallback: the seed
    states the family outright, which is what puts a fine-tune in with the
    base model it came from instead of in a family of one."""
    for field in ("family", "name", "public_id"):
        head = _FAMILY_HEAD.match(str(row.get(field) or "").strip())
        if head:
            word = head.group(0)
            return word.upper() if len(word) <= 3 else word[0].upper() + word[1:]
    return "Other"


def public_family(row: dict) -> str:
    """The family to present this row under -- the column when it is set, a
    guess from the name when it is not. An empty column therefore never means
    "ungrouped"; there is no such state, and a blank box on the tab reads as
    "you decide", which is what this does."""
    return str(row.get("family") or "").strip() or derive_family(row)


def clean_host_list(raw: Any) -> list[str]:
    """A host list setting as the tab or an API client may have written it:
    a JSON list, or one comma-separated string. Lower-cased, trimmed, empty
    entries dropped, order kept (it is a preference order)."""
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    for h in raw:
        if not isinstance(h, str):
            continue
        h = h.strip().lower()
        if h and h not in out:
            out.append(h)
    return out


def clean_family_order(raw: Any) -> list[str]:
    """A settings list turned into an order: trimmed, de-duplicated
    case-insensitively (the comparison everywhere else is case-insensitive
    too, so "qwen" and "Qwen" must not both hold a place), bounded."""
    out: list[str] = []
    seen: set[str] = set()
    for item in (raw if isinstance(raw, list) else []):
        name = str(item or "").strip()
        if not name or name.casefold() in seen:
            continue
        seen.add(name.casefold())
        out.append(name[:64])
        if len(out) >= 64:
            break
    return out


def backfill_public_families() -> int:
    """Give every catalogue row a family, from the seed where the row came
    from the seed and from its name otherwise.

    Runs at boot beside raise_stale_ctx_ceilings(), for the same reason: the
    ordinary reseed refuses to touch an existing row, so without this every
    hub that already has a catalogue would carry the new column empty forever
    and group its whole public page under guesses. A no-op once each row has
    one."""
    seeded = {str(r.get("public_id") or ""): str(r.get("family") or "").strip()
              for r in load_public_models_seed()}
    changed = 0
    for row in db_query("SELECT public_id, name, family FROM public_models"):
        if str(row.get("family") or "").strip():
            continue
        fam = seeded.get(str(row["public_id"])) or derive_family(row)
        db_update("UPDATE public_models SET family=?, updated_at=? WHERE public_id=?",
                  (fam, now(), row["public_id"]))
        changed += 1
    if changed:
        public_catalogue(force=True)
        log.info("filled in the family of %d public catalogue row(s)", changed)
    return changed


def _public_domain_upsert(rec: dict) -> None:
    domain = str(rec.get("domain", "")).strip().lower()
    db_exec(
        "INSERT INTO public_domains(domain,company,source,rank,mode,enabled,"
        "added_at,notes) VALUES (?,?,?,?,?,?,?,?) "
        "ON CONFLICT(domain) DO UPDATE SET "
        "company=excluded.company, source=excluded.source, rank=excluded.rank,"
        " mode=excluded.mode, notes=excluded.notes",
        (
            domain,
            str(rec.get("company", "")),
            str(rec.get("source", "custom")),
            rec.get("rank"),
            str(rec.get("mode", "allow")),
            int(bool(rec.get("enabled", 1))),
            now(),
            str(rec.get("notes", "")),
        ),
    )


def seed_public_domains(missing_only: bool = False) -> int:
    rows = load_public_domains_seed()
    existing = (
        {r["domain"] for r in db_query("SELECT domain FROM public_domains")}
        if missing_only else set()
    )
    added = 0
    for rec in rows:
        domain = str(rec.get("domain", "")).strip().lower()
        if not domain or (missing_only and domain in existing):
            continue
        _public_domain_upsert(rec)
        added += 1
    return added


# ---- free-mail / disposable block-list --------------------------------

PUBLIC_FREE_MAIL = {
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.uk", "yahoo.co.in",
    "yahoo.ca", "yahoo.fr", "yahoo.de", "hotmail.com", "hotmail.co.uk",
    "hotmail.fr", "outlook.com", "outlook.co.uk", "live.com", "live.co.uk",
    "msn.com", "aol.com", "icloud.com", "me.com", "mac.com", "proton.me",
    "protonmail.com", "pm.me", "gmx.com", "gmx.net", "gmx.de", "mail.com",
    "yandex.com", "yandex.ru", "zoho.com", "fastmail.com", "hey.com",
    "comcast.net", "att.net", "verizon.net", "sbcglobal.net", "cox.net",
    "charter.net",
    # disposable / throwaway providers
    "mailinator.com", "guerrillamail.com", "10minutemail.com", "temp-mail.org",
    "yopmail.com", "trashmail.com", "getnada.com", "dispostable.com",
    "sharklasers.com", "maildrop.cc",
}
# Suffix families for the providers above that operate many country TLDs --
# matching is suffix-aware, so "mail.yahoo.co.uk" must count as yahoo even
# though the exact string is not in the set above.
PUBLIC_FREE_MAIL_FAMILIES = ("yahoo.", "hotmail.", "outlook.", "live.", "gmx.", "yandex.")


def _domain_suffix_match(host: str, rule: str) -> bool:
    """True when `host` IS `rule` or a subdomain of it -- 'mail.boeing.com'
    matches a 'boeing.com' rule, but 'notboeing.com' does not."""
    host, rule = host.lower(), rule.lower().lstrip(".")
    return host == rule or host.endswith("." + rule)


def _is_free_mail(domain: str) -> bool:
    if domain in PUBLIC_FREE_MAIL:
        return True
    # A family name has to occupy a whole label ("yahoo" in
    # mail.yahoo.co.uk's ['mail','yahoo','co','uk']) -- a plain substring
    # test would also match a legitimate "notyahoo.com".
    labels = domain.split(".")
    return any(fam.rstrip(".") in labels for fam in PUBLIC_FREE_MAIL_FAMILIES)


# ---- eligibility --------------------------------------------------------

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _email_canon(email: str) -> str:
    """The address with any +tag dropped from the local part -- what "the
    same person" means for the one-key-per-address rule. Delivery still
    uses the address exactly as typed."""
    local, _, domain = str(email or "").strip().lower().rpartition("@")
    return local.split("+", 1)[0] + "@" + domain


# What "a live Fleet Pass key" means, everywhere a cap counts one. The status
# column alone is not it: nothing flips an expired key's row out of 'issued'
# (roll_window only follows deliberate archival), so counting by status let
# every key that ever expired occupy its domain's and the fleet's global
# allowance forever -- a returning requester whose old key had merely expired
# was told the domain or pool was full. Takes one bind parameter: now().
# A date-only expiry means the end of that day, exactly as
# /public/api/key-status reports it.
_LIVE_PUBLIC_KEYS_SQL = (
    "FROM public_keys pk JOIN api_keys k ON k.id=pk.key_id "
    "WHERE pk.status='issued' AND pk.archived_at IS NULL "
    "AND k.archived_at IS NULL AND (k.expires_at IS NULL OR "
    "(CASE WHEN length(k.expires_at)=10 THEN k.expires_at || 'T23:59:59+00:00' "
    "ELSE k.expires_at END) > ?)"
)


def public_eligibility(email: str) -> dict:
    """Pure function: what should happen for this email address.

    {"verdict": "allow", "source", "company", "rank"} | {"verdict": "review"}
    | {"verdict": "reject", "code": "free_mail" | "blocked" | "not_listed"}
    """
    email = str(email or "").strip().lower()
    if not _EMAIL_RE.match(email):
        return {"verdict": "reject", "code": "invalid_email"}
    domain = email.rsplit("@", 1)[-1]
    if _is_free_mail(domain):
        return {"verdict": "reject", "code": "free_mail"}

    rows = db_query("SELECT * FROM public_domains WHERE enabled=1")
    # Longest suffix match wins on each side, so '.gov' does not shadow a more
    # specific 'nasa.gov' rule (or vice versa) -- whichever rule names the
    # smaller slice of the internet is the one that actually applies.
    block_hit, allow_hit = None, None
    for r in rows:
        rule = str(r["domain"]).strip().lower()
        bare = rule.lstrip(".")
        if not _domain_suffix_match(domain, bare):
            continue
        best = block_hit if r["mode"] == "block" else allow_hit
        if best is None or len(bare) > len(str(best["domain"]).lstrip(".")):
            if r["mode"] == "block":
                block_hit = r
            else:
                allow_hit = r
    if block_hit is not None:
        return {"verdict": "reject", "code": "blocked"}
    if allow_hit is not None:
        return {
            "verdict": "allow",
            "source": allow_hit["source"],
            "company": allow_hit["company"],
            "rank": allow_hit["rank"],
        }
    settings = get_public_settings()
    if settings.get("review_unlisted", True):
        return {"verdict": "review"}
    return {"verdict": "reject", "code": "not_listed"}


# ---- public settings ------------------------------------------------------

DEFAULT_PUBLIC_SETTINGS: dict[str, Any] = {
    "single_rpd": 5, "single_rph": 2, "team_rpd": 3, "team_rph": 2, "key_days": 7,
    "team_max_workers": 2, "team_max_rounds": 3,
    "max_keys_per_domain": 10, "max_live_keys": 200, "ip_requests_per_day": 3,
    # How many LIVE keys one address may hold at once. An employer who mislaid
    # the mail should be able to ask again without waiting out the expiry, so
    # this is a count rather than the old one-or-nothing switch. Revoked,
    # archived and expired keys are not live and do not count -- see the query
    # in public_request(). 0 means no per-address limit at all.
    "max_keys_per_email": 3,
    "review_unlisted": True,
    "fallback_when": "not_resident",
    "fallback_tolerance": 0.5,
    # Requirement 2's busy trigger: when EVERY box serving the requested
    # model is saturated (no free decode slot), pick_fallback() may offer a
    # substitute the way it already does for unreachable/not_resident. A
    # fleet-wide toggle, not Fleet-Pass-only -- see pick_fallback() and the
    # routing-policy section of README.md.
    "substitute_when_busy": True,
    # The reactive same-box context-overflow retry (openai_proxy, fleet_chat):
    # a box that rejects a prompt as too long for its window gets ONE retry
    # at a size fitted to what it actually reported, before routing moves on
    # to the next candidate. Ships on, but conservative -- see
    # _classify_upstream_failure()'s docstring for exactly which upstream
    # error shapes count as "too long" versus an ordinary client error that
    # is returned unchanged.
    "ctx_retry_live": True,
    # How long a box that answered 429/503 ("busy", not broken) sits out of
    # routing before it is offered again -- short on purpose, so the very
    # next request does not immediately re-hit a box that is about to free
    # up, but the request after that can. Contrast COOLDOWN_UPSTREAM_5XX
    # (45s), which is for a box that actually failed.
    "busy_cooldown_seconds": 5,
    "fallback_notice": True,
    "fallback_notice_text": (
        "[Fleet notice: {requested} was not available, so this reply came "
        "from {served} ({arch}, ~{active}B active). Your key is unchanged.]\n\n"
    ),
    # The context-window twin of the fallback notice. A key is issued against
    # the largest window ANY box in the fleet can give its model; when the
    # request lands on a smaller box instead, the window shrinks and the
    # caller is told so in the same voice, for the same reason.
    "ctx_notice": True,
    "ctx_notice_text": (
        "[Fleet notice: your key is set up for {requested} tokens of context, "
        "but the machine that can serve that much was unavailable, so this "
        "reply was answered with {granted}. Your key is unchanged.]\n\n"
    ),
    # Hard ceiling on what any Fleet Pass key may be issued for, whatever the
    # hardware could manage. 0 means "no policy cap, hardware decides".
    "ctx_ceiling": 0,
    # ---- how the public page presents the catalogue ----
    # A visitor picking a model off the request form reads down the list, so
    # the order is a product decision, not an implementation detail. Three
    # controls, coarse to fine: the families in the order they should appear,
    # each row's own `sort` inside its family, and one model suggested ahead
    # of everything. A family this list does not name falls in behind the ones
    # it does, keeping the order its `sort` values already gave it; an empty
    # list means "do not group by family at all, just use sort", which is what
    # the catalogue shipped as.
    "model_family_order": ["Qwen", "Gemma", "Nemotron", "GLM", "DeepSeek", "Muse"],
    # The public id the portal shows first and pre-selects. "" is a valid
    # setting -- it means the form suggests nothing and leads with the order
    # above.
    "featured_model": "qwen3.8-27b",
    # ...and keep that model resident on every box that can hold it, so the
    # first request against a new key does not pay a cold load. See the
    # preload loop.
    "preload_featured": True,
    # The "load my model now" button in the key email (see warm_link()).
    "warm_button": True,
    "team_prompt": (
        "You lead a small team of worker models on a hobby inference fleet. "
        "Split independent work across spawn_subagents; keep the final answer "
        "in your own voice."
    ),
    "worker_prompt": (
        "You are a worker model on a hobby inference fleet, given one "
        "self-contained task. Answer it directly and completely -- you have "
        "no memory of any other conversation."
    ),
    "mail_from": "", "mail_reply_to": "", "admin_notify": "",
    "email_subject": "Your open-fleet key ({days} days)",
    "email_intro": (
        "Thanks for taking a look at the open-fleet -- a small "
        "self-hosted inference cluster of consumer and prosumer machines "
        "(AMD Strix Halo boxes, Apple silicon, a CPU-only box) behind one "
        "OpenAI-compatible gateway, with load-aware routing, per-key "
        "metering, and parallel sub-agent teams. More about it: "
        "https://example.org/fleet"
    ),
    "email_setup": (
        "curl:\n"
        "  curl {base_url}/chat/completions -H \"Authorization: Bearer {key}\" "
        "-H \"Content-Type: application/json\" -d '{{\"model\":\"{model_id}\","
        "\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}'\n\n"
        "Python (openai SDK):\n"
        "  from openai import OpenAI\n"
        "  client = OpenAI(base_url=\"{base_url}\", api_key=\"{key}\",\n"
        "                  default_headers={{\"User-Agent\": \"fleet-pass/1\"}})\n"
        "  print(client.chat.completions.create(model=\"{model_id}\",\n"
        "        messages=[{{\"role\": \"user\", \"content\": \"hi\"}}])"
        ".choices[0].message.content)\n\n"
        "Node (openai SDK):\n"
        "  const client = new OpenAI({{ baseURL: \"{base_url}\", apiKey: \"{key}\",\n"
        "    defaultHeaders: {{ \"User-Agent\": \"fleet-pass/1\" }} }});\n\n"
        "Cline: pick \"OpenAI Compatible\", paste the base URL and key, and "
        "set Context Window Size to {ctx} -- Cline cannot read it from the "
        "server, and an old value left in the provider profile silently "
        "caps every conversation.\n\n"
        "Open WebUI: Settings -> Connections -> OpenAI API, same base URL and key.\n\n"
        "That User-Agent line keeps the official SDKs clear of AI-crawler "
        "filtering at the CDN in front of the fleet, which answers their "
        "default \"OpenAI/...\" agent with a 403 before the request reaches "
        "us. Any other value works, and curl, Cline and Open WebUI need "
        "nothing."
    ),
    "email_disclaimer": (
        "This runs on hobby hardware in someone's home -- there is no SLA, "
        "and a cold model load can take a couple of minutes. Your configured "
        "model may occasionally be swapped for a similar one when it is "
        "unavailable; the reply will say so when that happens. Please do not "
        "send confidential or personal data -- requests are logged for "
        "metering (prompt and response bodies are not stored). Keys are "
        "single-person and expire automatically."
    ),
    "public_base_url": "",
    # ---- the live demo on example.org/fleet/tour ----
    # An unkeyed chat box at the bottom of the public presentation: one
    # configured catalogue model, answered by the boxes the owner allows,
    # limited per network address rather than per key. See the "Live demo"
    # block below public_key_status().
    "demo_enabled": True,
    "demo_model": "qwen3.8-9b-distill",
    # Hosts tried first when they serve the model (by real host name; the
    # public reply only ever names the box alias), and hosts never used for
    # the demo however well they score -- the big boxes are for the keys.
    "demo_prefer_hosts": ["gpu-laptop-1"],
    "demo_exclude_hosts": ["mac-laptop-1", "apu-box-1", "apu-tablet-1", "apu-tablet-2", "gpu-desktop-2"],
    "demo_ip_rph": 5,
    "demo_max_tokens": 512,
    "demo_max_prompt_chars": 6000,
    "demo_system_prompt": (
        "You are the live demo of a self-hosted AI fleet: a small open-weight "
        "model answering from an ordinary home machine behind a gateway. Be "
        "concise -- under 150 words unless the question needs more -- and "
        "plain-spoken. You do not know the names, addresses or operating "
        "systems of the machines in the fleet; if asked, say so and describe "
        "yourself as one of several boxes behind one gateway."
    ),
    # ---- auto-issue: keys minted by a connected service, no person in the loop
    # The owner's own tooling (a job-search autopilot that hands a recruiter a
    # live key inside a cover letter, say) calls POST /admin/api/public/keys/auto
    # with the admin token and gets the raw key back once. Everything about
    # the key -- model, lifetime, budgets, base URL, the setup text it is
    # handed on with -- is decided HERE, so the service never carries a copy
    # of these settings that could drift from the tab.
    "auto_issue_enabled": True,
    "auto_issue_model": "qwen3.8-27b",
    "auto_issue_ctx": 16384,
    "auto_issue_daily_cap": 20,
    "auto_issue_setup_text": (
        "Base URL: {base_url}\n"
        "API key: {key}\n"
        "Model: {model}\n"
        "Context window: {ctx} tokens (set this in your client -- Cline calls "
        "it Context Window Size; it cannot be read from the server).\n"
        "Any OpenAI-compatible client works (curl, the openai SDKs, Cline, "
        "Open WebUI); set the User-Agent header to anything but the SDK "
        "default. Expires {expires}; {limits}."
    ),
}

_PUBLIC_SETTING_BOUNDS: dict[str, tuple[float, float]] = {
    "single_rpd": (1, 1000), "single_rph": (1, 500), "team_rpd": (1, 1000),
    "team_rph": (1, 500), "key_days": (1, 365), "team_max_workers": (1, 8),
    "team_max_rounds": (1, 24), "max_keys_per_domain": (1, 100000),
    # Low bound 0, not 1: zero is the "no per-address limit" setting, which is
    # what the old one_key_per_email=false migrates to.
    "max_keys_per_email": (0, 1000),
    "ctx_ceiling": (0, 1048576),
    "max_live_keys": (1, 1000000), "ip_requests_per_day": (1, 10000),
    "fallback_tolerance": (0.0, 5.0),
    "demo_ip_rph": (1, 1000), "demo_max_tokens": (64, 8192),
    "demo_max_prompt_chars": (200, 200000),
    "auto_issue_ctx": (1024, 1048576), "auto_issue_daily_cap": (1, 1000),
    "busy_cooldown_seconds": (1, 300),
}


def _public_settings_raw() -> dict:
    """Whatever is actually stored under settings['public'], no defaults
    applied -- so a save merges onto exactly what a previous save left, not
    onto today's computed fallbacks (which must never get baked in as if an
    admin had chosen them)."""
    rows = db_query("SELECT value FROM settings WHERE key='public'")
    if not rows:
        return {}
    try:
        data = json.loads(rows[0]["value"])
        return data if isinstance(data, dict) else {}
    except (TypeError, ValueError):
        return {}


def get_public_settings() -> dict:
    out = dict(DEFAULT_PUBLIC_SETTINGS)
    raw = _public_settings_raw()
    out.update({k: v for k, v in raw.items() if k in out})
    # max_keys_per_email replaced the one_key_per_email switch. A stored
    # `false` was a deliberate "do not limit by address" and has to survive the
    # rename, so it migrates to 0. A stored `true` carries no number -- one was
    # the only thing the switch could mean -- so it is left to today's default
    # rather than being pinned at 1, which would silently keep every existing
    # hub on the old behaviour. Read-time, not a write-time migration: nothing
    # is baked into the stored settings until an admin saves the tab.
    if "max_keys_per_email" not in raw and "one_key_per_email" in raw:
        if not raw["one_key_per_email"]:
            out["max_keys_per_email"] = 0
    for key, (lo, hi) in _PUBLIC_SETTING_BOUNDS.items():
        try:
            if isinstance(DEFAULT_PUBLIC_SETTINGS[key], float):
                out[key] = max(lo, min(hi, float(out[key])))
            else:
                out[key] = max(int(lo), min(int(hi), int(out[key])))
        except (TypeError, ValueError):
            out[key] = DEFAULT_PUBLIC_SETTINGS[key]
    if out.get("fallback_when") not in ("not_resident", "unreachable", "never"):
        out["fallback_when"] = DEFAULT_PUBLIC_SETTINGS["fallback_when"]
    # A stored [] is a real choice ("no family grouping"); only a value that
    # is not a list at all falls back to the shipped order.
    fams = out.get("model_family_order")
    out["model_family_order"] = clean_family_order(
        fams if isinstance(fams, list) else DEFAULT_PUBLIC_SETTINGS["model_family_order"])
    out["featured_model"] = str(out.get("featured_model") or "").strip()
    out["preload_featured"] = bool(out.get("preload_featured", True))
    out["demo_enabled"] = bool(out.get("demo_enabled", True))
    out["demo_model"] = str(out.get("demo_model") or "").strip()
    out["demo_system_prompt"] = str(out.get("demo_system_prompt") or "")
    for k in ("demo_prefer_hosts", "demo_exclude_hosts"):
        out[k] = clean_host_list(out.get(k))
    out["auto_issue_enabled"] = bool(out.get("auto_issue_enabled", True))
    out["auto_issue_model"] = str(out.get("auto_issue_model") or "").strip()
    out["auto_issue_setup_text"] = str(out.get("auto_issue_setup_text") or "")
    if not str(out.get("admin_notify") or "").strip():
        out["admin_notify"] = next(iter(ADMIN_EMAILS), "")
    if not str(out.get("public_base_url") or "").strip():
        out["public_base_url"] = PUBLIC_API_URL
    return out


def sync_public_key_limits(settings: dict) -> int:
    """Push today's request ceilings onto the keys already out there, and
    answer how many were moved.

    A Fleet Pass key is minted with the numbers from the settings tab stamped
    onto its own row, and enforcement reads that row -- so raising a limit
    used to do nothing at all for anyone already holding a key, while the tab
    that raised it, the key-status endpoint and the key's own welcome mail all
    went on quoting the new figure. Keys live a week; an operator who widens a
    demo budget means "let them through", not "let the next person through",
    and would otherwise be waiting out the old key's expiry to find out.

    Only live Fleet Pass keys: a hand-minted admin key keeps whatever bespoke
    budget it was given, because nothing on this tab claims to describe it.
    The per-kind split matters -- a team turn costs the fleet several rounds,
    which is why it has its own, smaller allowance."""
    moved = 0
    for kind in ("single", "team"):
        try:
            rpd = int(settings[kind + "_rpd"])
            rph = int(settings[kind + "_rph"])
        except (KeyError, TypeError, ValueError):
            continue
        moved += db_update(
            "UPDATE api_keys SET max_rpd=?, max_rph=? "
            "WHERE archived_at IS NULL AND (max_rpd IS NOT ? OR max_rph IS NOT ?) "
            "AND id IN (SELECT key_id FROM public_keys WHERE key_id IS NOT NULL "
            "AND status='issued' AND kind=?)",
            (rpd, rph, rpd, rph, kind),
        )
    if moved:
        log.info("synced request limits onto %d live Fleet Pass key(s)", moved)
    return moved


def set_public_settings(updates: dict) -> dict:
    if not isinstance(updates, dict):
        return get_public_settings()
    merged = _public_settings_raw()
    merged.update({k: v for k, v in updates.items() if k in DEFAULT_PUBLIC_SETTINGS})
    db_exec(
        "INSERT INTO settings(key,value) VALUES ('public',?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (json.dumps(merged),),
    )
    out = get_public_settings()
    # Bounds-clamped values, not the raw ones off the form: a key must never
    # be stamped with a number the settings reader would refuse to honour.
    out["keys_updated"] = sync_public_key_limits(out)
    return out


# ---- SMTP -----------------------------------------------------------------
#
# Pure configuration, never the database: the transport (Gmail app password
# today, API@example.org on Zoho later) is an operational detail that must
# survive a database restore untouched.

SMTP_HOST = os.environ.get("LLMSTACK_SMTP_HOST", "").strip()
SMTP_PORT = int(os.environ.get("LLMSTACK_SMTP_PORT", "587") or 587)
SMTP_USER = os.environ.get("LLMSTACK_SMTP_USER", "").strip()
SMTP_PASSWORD = os.environ.get("LLMSTACK_SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("LLMSTACK_SMTP_FROM", "").strip()
SMTP_TLS = os.environ.get("LLMSTACK_SMTP_TLS", "starttls").strip().lower()
PUBLIC_INTAKE_TOKEN = os.environ.get("LLMSTACK_PUBLIC_INTAKE_TOKEN", "").strip()


def _send_mail_sync(to: str, subject: str, text: str, html: str | None) -> tuple[bool, str]:
    if not SMTP_HOST:
        return False, "smtp not configured"
    settings = get_public_settings()
    sender = SMTP_FROM or SMTP_USER
    if not sender:
        return False, "smtp not configured"
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    reply_to = str(settings.get("mail_reply_to") or "").strip()
    if reply_to:
        msg["Reply-To"] = reply_to
    msg.attach(MIMEText(text, "plain"))
    if html:
        msg.attach(MIMEText(html, "html"))
    try:
        if SMTP_TLS == "ssl":
            server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=20)
        else:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20)
        with server:
            if SMTP_TLS == "starttls":
                server.starttls()
            if SMTP_USER:
                server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(sender, [to], msg.as_string())
        return True, ""
    except Exception as exc:  # noqa: BLE001 -- mail failure must never crash a request
        return False, str(exc)[:300]


async def send_mail(to: str, subject: str, text: str, html: str | None = None) -> tuple[bool, str]:
    """Send over SMTP in a thread -- smtplib is blocking, and this fires from
    inside request handlers that would otherwise stall the event loop for
    however long the mail server takes to answer."""
    return await asyncio.to_thread(_send_mail_sync, to, subject, text, html)


def require_intake_token(request: Request) -> None:
    """Guard for /public/api/*'s POST endpoints. Constant-time compare -- this
    token is the only thing standing between the public internet and minting
    fleet keys, so it is checked the same way an API key is."""
    if not PUBLIC_INTAKE_TOKEN:
        raise PublicError(503, "intake_unconfigured", "intake not configured")
    got = request.headers.get("x-intake-token", "")
    if not got or not secrets.compare_digest(got, PUBLIC_INTAKE_TOKEN):
        raise PublicError(401, "bad_intake_token", "invalid intake token")


def log_public_event(kind: str, email: str = "", ip: str = "", detail: str = "") -> None:
    db_exec(
        "INSERT INTO public_events(ts,kind,email,ip,detail) VALUES (?,?,?,?,?)",
        (now(), kind, email, ip, detail[:500]),
    )


# ---- catalogue + fleet routing integration --------------------------------

_public_catalogue_cache: dict[str, Any] = {"t": 0.0, "by_public": {}, "by_fleet": {}}
PUBLIC_CATALOGUE_TTL = 30.0


def public_catalogue(force: bool = False) -> dict:
    """public_id -> row, and fleet_id -> public_id, for every catalogue row.

    Cached: the /v1 proxy consults this on every request from a Fleet Pass
    key, and the table itself only ever changes when an admin edits it."""
    if not force and time.time() - _public_catalogue_cache["t"] < PUBLIC_CATALOGUE_TTL:
        return _public_catalogue_cache
    by_public: dict[str, dict] = {}
    by_fleet: dict[str, str] = {}
    for row in db_query("SELECT * FROM public_models"):
        pid = str(row["public_id"])
        by_public[pid] = row
        try:
            fids = json.loads(row.get("fleet_ids") or "[]")
        except (TypeError, json.JSONDecodeError):
            fids = []
        for fid in fids:
            if isinstance(fid, str) and fid.strip():
                by_fleet.setdefault(fid.strip(), pid)
    _public_catalogue_cache.update(t=time.time(), by_public=by_public, by_fleet=by_fleet)
    return _public_catalogue_cache


def _row_fleet_ids(row: dict) -> list[str]:
    try:
        fids = json.loads(row.get("fleet_ids") or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    return [f.strip() for f in fids if isinstance(f, str) and f.strip()]


# ---- presentation order ---------------------------------------------------


def featured_public_id(settings: dict | None = None) -> str:
    """The public id the portal suggests first, or "" when the setting names
    a model the catalogue no longer offers.

    Checked rather than echoed: a model that is disabled or deleted must
    degrade to "no suggestion", never to a badge on a row nobody can pick or
    a pre-selection the intake would then reject."""
    settings = settings or get_public_settings()
    pid = str(settings.get("featured_model") or "").strip()
    row = public_catalogue()["by_public"].get(pid)
    return pid if row and row.get("enabled") else ""


def _sort_of(row: dict) -> int:
    try:
        return int(row.get("sort") or 0)
    except (TypeError, ValueError):
        return 0


def order_public_models(rows: list[dict], settings: dict | None = None) -> list[dict]:
    """The catalogue in the order the public page presents it: the suggested
    model first, then family by family in the admin's order, and within a
    family by each row's own `sort`.

    A family the order does not name keeps the position its rows already had
    rather than being shoved to the end alphabetically -- unlisted families
    are ranked by their best-placed member, so adding a model to the seed
    without touching the setting lands it where its `sort` always said it
    went, and an empty order reproduces exactly the sort-only listing this
    shipped as."""
    settings = settings or get_public_settings()
    rank = {f.casefold(): i
            for i, f in enumerate(clean_family_order(settings.get("model_family_order")))}
    unlisted: dict[str, int] = {}
    for row in rows:
        fam = public_family(row).casefold()
        if fam not in rank:
            unlisted[fam] = min(unlisted.get(fam, _sort_of(row)), _sort_of(row))
    tail = {fam: i for i, (fam, _) in enumerate(
        sorted(unlisted.items(), key=lambda kv: (kv[1], kv[0])))}
    # Matched against the rows in hand rather than looked up: this is a sort,
    # and a sort that reads the database is a sort that behaves differently
    # depending on when it is called. The `enabled` guard is what keeps a
    # disabled suggestion from being promoted to the top of a listing that
    # will not badge it -- featured_public_id() refuses to advertise one.
    featured = str(settings.get("featured_model") or "").strip()

    def key(row: dict) -> tuple:
        fam = public_family(row).casefold()
        band, place = (0, rank[fam]) if fam in rank else (1, tail.get(fam, 0))
        top = str(row.get("public_id")) == featured and bool(row.get("enabled"))
        return (0 if top else 1,
                band, place, _sort_of(row),
                str(row.get("name") or ""), str(row.get("public_id") or ""))

    return sorted(rows, key=key)


def _catalogue_row_for(model: str) -> dict | None:
    """The public catalogue row `model` resolves to, whether it was named by
    its public id, by a fleet id one of those rows claims, or by a
    FLEET_ROLES policy name -- the same by_public -> by_fleet fallback
    resolve_targets() already applies below. A bare `by_public.get(model)`
    only ever matches a public_id, which almost no plain bearer-key or
    downstream-app request uses -- they ask by fleet id or role name -- so a
    caller that used the bare lookup to decide whether a substitute exists
    (pick_fallback's callers) silently never found one for that traffic."""
    cat = public_catalogue()
    row = cat["by_public"].get(model)
    if row is not None:
        return row
    served_as = cat["by_fleet"].get(model)
    if served_as:
        return cat["by_public"].get(served_as)
    return None


def _no_fallback_requested(request: Request | None = None,
                           payload: dict | None = None) -> bool:
    """The per-request opt-out (requirement 2): the `X-Fleet-No-Fallback`
    header for a caller that can set one, or the equivalent
    `fleet_no_fallback` body field for one that can't (a /v1/batches
    submission has no per-hop header of its own). Any value but empty/'0'/
    'false'/'no' opts out; the common case is '1'."""
    if request is not None:
        v = request.headers.get("x-fleet-no-fallback")
        if v is not None and str(v).strip().lower() not in ("", "0", "false", "no"):
            return True
    if isinstance(payload, dict) and payload.get("fleet_no_fallback"):
        return True
    return False


async def resolve_targets(model: str, role: str = "primary", prompt_tokens: int = 0,
                          gen_tokens: int = 256, need_ctx: int = 0,
                          ) -> list[tuple[str, str]]:
    """Every (host, fleet_id) pair that can serve `model`, best first.

    `model` may be a public catalogue id, a fleet id one of those rows
    claims, or an ordinary fleet-native id -- the last case is exactly
    model_hosts(), just paired with itself. A catalogue row with several
    fleet ids (a MoE served under two different tags, say) has its
    candidates ranked against each other by the same scorer model_hosts()
    uses, via _score_host_model_pairs(). The keyword arguments describe the
    request being routed and go straight to the scorer."""
    await model_routes()
    kw = dict(role=role, prompt_tokens=prompt_tokens, gen_tokens=gen_tokens,
              need_ctx=need_ctx)
    row = _catalogue_row_for(model)
    if row is not None:
        fids = _row_fleet_ids(row) or [model]
        pairs: list[tuple[str, str]] = []
        for fid in fids:
            for cand in _routes_cache["cands"].get(fid, []):
                pairs.append((cand, fid))
        return await _score_host_model_pairs(pairs, **kw)
    # A fleet role is a policy, not a name any box serves: each box's best
    # model for it, ranked like any other candidates. See FLEET_ROLES.
    if model in FLEET_ROLES:
        return await _score_host_model_pairs(role_pairs(model), fleet_role=model, **kw)
    hosts = await model_hosts(model, **kw)
    return [(h, model) for h in hosts]


def _catalogue_row_online(fids: list[str]) -> bool:
    cands = _routes_cache.get("cands", {})
    return any(cands.get(f) for f in fids)


def _catalogue_row_resident(fids: list[str]) -> bool:
    cands = _routes_cache.get("cands", {})
    running = _routes_cache.get("running", {})
    for f in fids:
        for cand in cands.get(f, []):
            hname = cand or HOST_NAME
            if f in running.get(hname, set()):
                return True
    return False


def _catalogue_row_resident_well(fids: list[str], role: str = "primary") -> bool:
    """Resident somewhere WORTH answering from: a box in the top two policy
    tiers for this model (a GPU that holds it, or the big box). A model
    that is warm only on the CPU backstop, or only on a card it spills out
    of, is resident in name and not in effect -- the fallback logic should
    treat it as cold, because that is what the caller will experience."""
    cands = _routes_cache.get("cands", {})
    running = _routes_cache.get("running", {})
    for f in fids:
        for cand in cands.get(f, []):
            hname = cand or HOST_NAME
            # Judged on the primary ladder whatever the role: the question is
            # whether the box is a good home for the model at all, not where
            # a worker would be sent by preference.
            if f in running.get(hname, set()) and host_tier(cand, f, "primary")[0] <= 1:
                return True
    return False


def catalogue_ctx(row: dict) -> dict:
    """What context window this catalogue row can actually be served with.

    Three numbers, and the gap between the first two is the whole point:

      best    the largest window any box in the fleet will serve it with,
              awake or not. This is the ceiling the request form offers,
              because a key lives for a week and the big box will be up for
              some of it.
      online  the largest among the boxes reachable right now. When this is
              below `best`, a request arriving this second gets a disclosed
              reduction rather than the window it was promised.
      hosts   per host name, so a caller (and the fleet page) can see where
              the top of the range actually lives.

    The row's own ctx_max stays an operator ceiling on top of all of it: it
    is now seeded to what each model was TRAINED for, so the binding limit is
    normally the hardware -- which is the point -- but an owner who wants a
    model held lower can still say so, and a program-wide `ctx_ceiling`
    setting caps the lot."""
    fids = _row_fleet_ids(row) or [str(row.get("public_id") or "")]
    cap = int(row.get("ctx_max") or 0) or AUTO_CTX_FALLBACK
    policy = 0
    try:
        policy = int(get_public_settings().get("ctx_ceiling") or 0)
    except (TypeError, ValueError):
        policy = 0
    if policy > 0:
        cap = min(cap, policy)
    known = known_model_ctx()
    cands = _routes_cache.get("cands", {})
    per_host: dict[str, int] = {}
    online: dict[str, int] = {}
    for fid in fids:
        for host, c in known.get(fid, {}).items():
            c = min(int(c or 0), cap)
            if c > per_host.get(host, 0):
                per_host[host] = c
        for cand in cands.get(fid, []):
            hname = cand or HOST_NAME
            c = min(host_model_ctx(cand, fid), cap)
            if c <= 0:
                continue
            if c > per_host.get(hname, 0):
                per_host[hname] = c
            if c > online.get(hname, 0):
                online[hname] = c
    # No box has ever reported a number for this model -- every peer is on an
    # older gateway, or it has simply never been routed. Advertise the same
    # conservative 32k this catalogue offered before any of it was measured,
    # NOT the row ceiling: those ceilings are now what a model was trained for
    # (262k on the Qwens), and promising that on nothing but a spec sheet is
    # how a key gets issued for a window no machine here can hold.
    best = max(per_host.values()) if per_host else min(cap, AUTO_CTX_FALLBACK)
    best_online = max(online.values()) if online else best
    # The intake only accepts multiples of 1024, so a ceiling that is not one
    # is a ceiling nobody can pick: apu-box-1 pins qwen3.8-27b at 262114, four
    # short of 256K, and a slider that stops there would reject its own
    # maximum. Round the advertised numbers down to something selectable.
    best -= best % 1024
    best_online -= best_online % 1024
    return {"best": max(1024, best), "online": max(1024, best_online),
            "hosts": per_host}


def _catalogue_row_all_saturated(fids: list[str]) -> bool:
    """True when every (host, fleet_id) pair that could serve `fids` right
    now has no free decode slot -- pick_fallback's busy trigger (requirement
    2), reusing _host_saturated so 'occupied' can never mean something
    different here than it does to the scorer or demo_host_policy. False
    (never trips the busy trigger) when the row has no online candidate at
    all -- that is the unreachable branch's job, not this one's."""
    cands = _routes_cache.get("cands", {})
    any_cand = False
    for f in fids:
        for cand in cands.get(f, []):
            any_cand = True
            if not _host_saturated(cand, f):
                return False
    return any_cand


def _catalogue_row_has_free_good_home(fids: list[str], role: str) -> bool:
    """A (host, fleet_id) pair for this row that is NOT saturated right now
    AND is a box worth answering from: already resident in the top policy
    tiers (mirrors _catalogue_row_resident_well), or a box the model is
    worth cold-loading onto (preload_capable). Used only to rank a busy-
    trigger substitute: escaping a queue only pays off if the box it lands
    on is actually a good one, not merely a box that happens to answer."""
    cands = _routes_cache.get("cands", {})
    running = _routes_cache.get("running", {})
    for f in fids:
        for cand in cands.get(f, []):
            if _host_saturated(cand, f):
                continue
            hname = cand or HOST_NAME
            if (f in running.get(hname, set())
                    and host_tier(cand, f, "primary")[0] <= 1):
                return True
            if preload_capable(cand, f):
                return True
    return False


async def pick_fallback(req_row: dict, role: str, settings: dict) -> dict | None:
    """Choose a substitute catalogue row for `req_row`, or None when the
    original should be tried as asked (either it is fine, or nothing in
    tolerance both fits the role and has anywhere to run).

    role: 'single' (any enabled row), 'primary' (allow_primary), or 'worker'
    (allow_worker) -- mirrors the flags a Fleet Pass team composes from.

    Three independent triggers can put a row up for substitution: it is
    unreachable, it is reachable but only resident somewhere not worth
    answering from ('not_resident'), or -- requirement 2, gated on its own
    `substitute_when_busy` toggle so it can be turned off without touching
    `fallback_when` -- every box that serves it right now is saturated. The
    first two are mutually exclusive (`fallback_when` picks one); busy can
    fire alongside whichever of them is active, because a saturated box is
    exactly the case neither of those two was built to catch."""
    when = str(settings.get("fallback_when", "not_resident"))
    if when == "never":
        return None
    await model_routes()
    req_fids = _row_fleet_ids(req_row)
    online = _catalogue_row_online(req_fids)
    busy = (online and settings.get("substitute_when_busy", True)
            and _catalogue_row_all_saturated(req_fids))
    if when == "unreachable":
        if online and not busy:
            return None
    elif when == "not_resident":
        # "Resident" on the CPU backstop, or on a card the model spills out
        # of, does not count: that is the one case where a warm model is
        # slower than a cold one on the right box, and exactly the case the
        # qwen3.6-35b-on-gpu-laptop-1 incident was.
        if online and not busy and _catalogue_row_resident_well(req_fids, role):
            return None
    else:
        return None

    tol = float(settings.get("fallback_tolerance", 0.5) or 0.5)
    req_active = float(req_row.get("active_b") or 0)
    req_params = float(req_row.get("params_b") or 0)
    req_pid = str(req_row.get("public_id"))
    best_row, best_key = None, None
    for pid, row in public_catalogue()["by_public"].items():
        if pid == req_pid or not row.get("enabled"):
            continue
        if str(row.get("arch")) != str(req_row.get("arch")):
            continue
        if role == "primary" and not row.get("allow_primary"):
            continue
        if role == "worker" and not row.get("allow_worker"):
            continue
        cand_active = float(row.get("active_b") or 0)
        if abs(cand_active - req_active) / max(req_active, 0.1) > tol:
            continue
        cand_fids = _row_fleet_ids(row)
        if not _catalogue_row_online(cand_fids):
            continue
        resident = _catalogue_row_resident_well(cand_fids, role)
        if busy:
            # Escaping a queue is the point of THIS trigger: a candidate
            # with a free slot beats a resident-but-saturated one, reversing
            # the ordinary residency preference just for this one case.
            free = _catalogue_row_has_free_good_home(cand_fids, role)
            key = (
                0 if free else 1,
                0 if resident else 1,
                abs(cand_active - req_active),
                abs(float(row.get("params_b") or 0) - req_params),
            )
        else:
            key = (
                0 if resident else 1,
                abs(cand_active - req_active),
                abs(float(row.get("params_b") or 0) - req_params),
            )
        if best_key is None or key < best_key:
            best_row, best_key = row, key
    if best_row is None:
        return None
    best_fids = _row_fleet_ids(best_row)
    # A swap has to buy something. Under "not_resident" the whole point is to
    # answer from a model already in memory instead of paying a cold load, so
    # if the best candidate would ALSO have to be loaded, the caller is better
    # served by the model they actually asked for: same wait, and it is the
    # one they chose. Seen in the wild on the first team run -- qwen3.5-4b was
    # swapped for nemotron-3-nano-4b, neither resident, for no gain. Under the
    # busy trigger the equivalent question is whether the substitute has
    # anywhere free to run at all -- a swap to a model that is ALSO
    # saturated everywhere trades one queue for another, no better than the
    # one the caller already asked for.
    if when == "not_resident" and online and not busy \
            and not _catalogue_row_resident_well(best_fids, role):
        return None
    if busy and not _catalogue_row_has_free_good_home(best_fids, role):
        return None
    return best_row


def render_fallback_notice(requested: str, served_row: dict, settings: dict) -> str:
    text = str(settings.get("fallback_notice_text")
              or DEFAULT_PUBLIC_SETTINGS["fallback_notice_text"])
    try:
        return text.format(
            requested=requested, served=served_row.get("public_id", ""),
            arch=served_row.get("arch", ""), active=served_row.get("active_b", ""),
        )
    except (KeyError, IndexError):
        return text


def render_ctx_notice(requested: int, granted: int, host: str,
                      settings: dict) -> str:
    text = str(settings.get("ctx_notice_text")
              or DEFAULT_PUBLIC_SETTINGS["ctx_notice_text"])
    try:
        return text.format(requested=requested, granted=granted,
                           host=(public_alias(host) if host else ""))
    except (KeyError, IndexError):
        return text


def public_notices(fallback: dict | None, ctx_cut: dict | None, host: str,
                   settings: dict, public: bool = False) -> tuple[str, dict, dict]:
    """The disclosure a Fleet Pass reply owes its caller, in the three shapes
    the caller might read it in: notice text to put in front of the answer,
    the `x_fleet` object stapled to the body, and response headers.

    One function, because a model substitution and a context reduction are the
    same promise -- "we changed what you asked for, here is what you actually
    got" -- and a single request can suffer both at once (the box that had the
    substitute resident is also the small box). Split across three call sites
    they drifted; together they cannot.

    `public` is a Fleet Pass caller: the structured fields then name the box
    the way the notice text always has -- "Box 3", never the hostname."""
    parts: list[str] = []
    xf: dict[str, Any] = {}
    headers: dict[str, str] = {}
    shown = (public_alias(host) if (public and host) else host)
    if fallback:
        xf.update({"requested": fallback["requested"],
                   "served": fallback["served"], "host": shown,
                   "reason": settings.get("fallback_when")})
        headers["X-Fleet-Fallback"] = (
            "requested=" + str(fallback["requested"]) + "; served="
            + str(fallback["served"]) + "; host=" + str(shown))
        if settings.get("fallback_notice", True):
            served_row = public_catalogue()["by_public"].get(
                fallback["served"], {})
            parts.append(render_fallback_notice(
                fallback["requested"], served_row, settings))
    if ctx_cut:
        req, got = int(ctx_cut["requested"]), int(ctx_cut["granted"])
        xf["ctx"] = {"requested": req, "granted": got, "host": shown}
        headers["X-Fleet-Ctx"] = ("requested=" + str(req) + "; granted="
                                  + str(got) + "; host=" + str(shown))
        if settings.get("ctx_notice", True):
            parts.append(render_ctx_notice(req, got, host, settings))
    return "".join(parts), xf, headers


def prepend_notice(obj: dict, notice: str) -> None:
    """Put `notice` in front of the assistant message, if there is one to put
    it in front of. A tool-call-only turn has no string content and is left
    exactly as the model produced it -- and an EMPTY answer stays empty: a
    notice standing in front of nothing reads as the reply itself (seen in the
    wild: a budget-starved thinking model returned "", and the caller was
    handed the notice as if it were the answer). The disclosure still reaches
    the caller either way, in x_fleet and the X-Fleet-* headers."""
    if not notice:
        return
    choices = obj.get("choices") or []
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str) \
                and msg["content"].strip():
            msg["content"] = notice + msg["content"]


def _wants_structured(payload: Any) -> bool:
    """True when the client will machine-parse the reply content whole --
    response_format json_object/json_schema. Text prepended to such content is
    corruption, not disclosure; the notice stays in x_fleet and the headers."""
    if not isinstance(payload, dict):
        return False
    fmt = payload.get("response_format")
    return isinstance(fmt, dict) and str(fmt.get("type") or "") in (
        "json_object", "json_schema")


def already_replied(messages: Any) -> bool:
    """True when this conversation has already had a prose reply -- an
    assistant turn carrying visible text -- and therefore must not be handed
    the notice banner again.

    A notice is a disclosure, and a disclosure lands once. There is no session
    id in the chat contract to hang that on, so the resent history is the only
    thing that knows what came before: an agentic client (Cline) re-requests
    on every tool round, and on a fleet that stays degraded the caller was
    getting the same sentence stapled in front of all of them -- read by the
    user as a fresh failure each time, and by the model as text to imitate.

    Assistant turns with no text are not a reply for this purpose: a
    tool-call-only round is exactly the turn prepend_notice() declines to
    write on, so the banner has not been shown yet and this turn may still
    carry it. Two accepted edges, both quiet by design: a conversation that
    only degrades at turn five is disclosed in x_fleet and the X-Fleet-*
    headers rather than in prose (those ride every reply, always), and a
    client that resends no history is a new conversation every time -- which,
    to a stateless gateway, it is."""
    if not isinstance(messages, list):
        return False
    for m in messages:
        if (isinstance(m, dict) and m.get("role") == "assistant"
                and isinstance(m.get("content"), str) and m["content"].strip()):
            return True
    return False


# Both notice templates open with this literal, and the default templates
# never contain a "]" of their own -- which is what makes stripping them safe
# below. Two accepted edges: an admin who edits a template to embed "]" gets
# that banner only partially stripped (degraded, not broken), and a model
# that ITSELF opens an answer with the literal "[Fleet notice: ...]" loses
# that bracketed aside from resent history -- the visible reply the user read
# is untouched either way.
_NOTICE_RE = re.compile(r"^\s*\[Fleet notice:[^\]]*\]\s*")


def strip_fleet_notices(messages: Any) -> bool:
    """Remove the gateway's own past notice banners from resent assistant
    history. The client faithfully stores what it was shown, so on a fleet
    that stays degraded every prior assistant turn comes back carrying its
    banner: tokens the model never wrote, counted against the caller's window
    on every subsequent estimate, and text the model can start imitating.
    Returns True when anything was removed. Only plain-string content is
    touched. Identification is by the banner's literal shape (see _NOTICE_RE's
    accepted edges) -- there is no side-channel marker to be exact with,
    because the banner has to live inside the visible text to be seen."""
    changed = False
    if not isinstance(messages, list):
        return False
    for m in messages:
        if not (isinstance(m, dict) and m.get("role") == "assistant"):
            continue
        c = m.get("content")
        if not isinstance(c, str) or "[Fleet notice:" not in c:
            continue
        out = c
        while True:
            # One past turn can carry both banners back to back -- a model
            # substitution and a context reduction disclosed together.
            new = _NOTICE_RE.sub("", out, count=1)
            if new == out:
                break
            out = new
        if out != c:
            m["content"] = out
            changed = True
    return changed


# Floor for a public key's completion budget, in tokens: enough that a
# thinking model's preamble cannot consume the entire answer. See the note in
# apply_ctx_limit.
#
# Raised from 256 to 512 on 2026-08-21 against a measurement rather than a
# guess. 256 was chosen when gemma4-12b burned 12 tokens of reasoning; asked a
# one-sentence question, nemotron3.5-lightning-30b spent 220 and returned an
# empty string, and needed 437 in total before it wrote its answer. A floor
# below what the fleet's thinking models actually spend is not a floor.
#
# This is a mitigation, not a cure: a model can always think past any ceiling.
# It only ever RAISES a too-small budget, and the key's context cap still
# wins, so the worst case is a slightly longer reply.
PUBLIC_MIN_COMPLETION = 512

# What a request that names NO completion budget gets, bounded by the room the
# prompt leaves. The old behavior invented 1024 here, and that is what turned
# a Cline turn on a thinking model into one visible sentence and a dead task
# (2026-08-25): the reasoning preamble and the answer spend the same budget,
# and a full agentic turn -- reasoning, prose, a complete tool call -- needs
# thousands of tokens, not one. An explicit max_tokens is still honoured
# (capped to the room) exactly as before; omitting it, or sending the 0/-1
# some clients use to mean "model default", now buys a budget an agent can
# actually finish a turn inside.
PUBLIC_DEFAULT_COMPLETION = 8192


def apply_ctx_limit(payload: dict, limit: int) -> dict:
    """How many tokens this request is about to cost, estimated from raw JSON
    length rather than a real tokenizer -- close enough to guard a per-key
    context cap without pulling a tokenizer into the gateway for every model
    in the fleet. Pure and unit-tested: no I/O, so the estimate is
    reproducible from the payload alone."""
    body = payload.get("messages")
    if body is None:
        body = payload.get("prompt")
    est = math.ceil(len(json.dumps(body, ensure_ascii=False)) / 3.2 * 1.1)
    est += math.ceil(len(json.dumps(payload.get("tools") or [], ensure_ascii=False)) / 3.2)
    # Rejected when less than the completion FLOOR is left, not merely when
    # the prompt itself overflows. Below the floor the reply would be
    # reasoning scraps cut off mid-thought: a 200 that reads as success and
    # completes nothing, which the caller cannot diagnose. The 413 names both
    # numbers, an agentic client can act on it (trim history), and the
    # per-candidate routing loops treat it as "try a roomier box".
    if est >= limit - PUBLIC_MIN_COMPLETION:
        raise HTTPException(413, {"error": {
            "message": "prompt is about " + str(est) + " tokens; this key's "
                       "context limit is " + str(limit),
            "type": "context_limit", "estimated_tokens": est, "limit": limit,
        }})
    room = limit - est
    # 0, -1 and a missing field all mean the same thing: the client names no
    # budget (Cline sends -1 for "model default"). `or 1024` used to collapse
    # every one of these into 1024 -- see PUBLIC_DEFAULT_COMPLETION above.
    try:
        asked = int(payload.get("max_tokens") or 0)
    except (TypeError, ValueError):
        asked = 0
    want = min(asked, room) if asked > 0 else min(PUBLIC_DEFAULT_COMPLETION, room)
    # Several models in the fleet open a reasoning block the server cannot
    # close, and it is spent from the same budget as the answer: gemma4-12b
    # asked for "sdk ok" with max_tokens=12 returned finish_reason=length,
    # 12 completion tokens, all of them reasoning, and content "". A demo key
    # whose first call comes back empty reads as a broken fleet, so give the
    # answer a floor. The rejection above guarantees the room for it: only
    # ever raises a too-small ceiling, never past the cap.
    payload["max_tokens"] = max(want, PUBLIC_MIN_COMPLETION)
    return payload


_public_key_cache: dict[int, tuple[float, dict | None]] = {}


def public_key_for(key_id: int) -> dict | None:
    """The public_keys row a bearer key was issued from, or None for an
    ordinary (non-Fleet-Pass) key. Cached per key: the fallback decision
    consults this on every proxied request from a public key."""
    hit = _public_key_cache.get(key_id)
    if hit and time.time() - hit[0] < 60:
        return hit[1]
    rows = db_query(
        "SELECT * FROM public_keys WHERE key_id=? AND status='issued' "
        "AND archived_at IS NULL",
        (key_id,),
    )
    row = rows[0] if rows else None
    _public_key_cache[key_id] = (time.time(), row)
    return row


_public_key_row_cache: dict[int, tuple[float, dict | None]] = {}


def public_key_row(key_id: int) -> dict | None:
    """Whether a bearer key has EVER had a public_keys row -- live or
    archived, any status. Unlike public_key_for (which only returns a
    currently-issued row, because that gates the fallback benefit), the
    surface restriction is permanent: revoking or letting a key expire must
    not reopen /v1/batches or the native proxy to it. Cached 60s, same TTL
    as public_key_for, since both are consulted on every proxied request."""
    hit = _public_key_row_cache.get(key_id)
    if hit and time.time() - hit[0] < 60:
        return hit[1]
    rows = db_query("SELECT * FROM public_keys WHERE key_id=?", (key_id,))
    row = rows[0] if rows else None
    _public_key_row_cache[key_id] = (time.time(), row)
    return row


_PUBLIC_ALLOWED_SURFACES = {("POST", "/v1/chat/completions"), ("GET", "/v1/models")}


def public_surface_allowed(path: str, method: str) -> bool:
    """A Fleet Pass key reaches exactly two routes (contract 1.9h) -- the
    owner's own words: 'a 5/day key must not be able to spool a 100k-request
    batch.' Everything else under /v1 (batches, embeddings, completions,
    responses, ...) and the whole native /api/* proxy are off-limits."""
    return (method.upper(), path) in _PUBLIC_ALLOWED_SURFACES


def _sum_usage(acc: dict, u: dict | None) -> None:
    if not isinstance(u, dict):
        return
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        acc[k] = acc.get(k, 0) + int(u.get(k, 0) or 0)


def _note_fallback(key_id: int, requested: str, served: str) -> None:
    # Accounting must never break the reply it accounts for: several workers
    # can land here concurrently (one per substituted task in a spawn round),
    # and a transient sqlite write failure in one of them used to propagate
    # through the round's gather and take every sibling's finished work down.
    try:
        db_exec("UPDATE public_keys SET fallbacks=fallbacks+1 WHERE key_id=?",
                (key_id,))
        log_public_event("fallback", detail=requested + " -> " + served)
    except Exception:  # noqa: BLE001
        pass


# ---- minting + email --------------------------------------------------


def issue_public_key(row: dict, decided_by: str = "auto") -> str:
    """Mint the api_keys row and its agent/team profile behind an approved
    Fleet Pass request, mark the request issued, and return the raw key --
    shown exactly once, same discipline as every other key this gateway
    hands out."""
    settings = get_public_settings()
    days = int(settings.get("key_days", 7))
    expires_at = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat(
        timespec="seconds"
    )
    kind = row["kind"]
    try:
        models = json.loads(row.get("models") or "{}")
    except (TypeError, json.JSONDecodeError):
        models = {}
    ctx = int(row.get("ctx") or 8192)
    rpd = int(settings["single_rpd"] if kind == "single" else settings["team_rpd"])
    rph = int(settings["single_rph"] if kind == "single" else settings["team_rph"])
    # An auto-issued key has no address behind it; its name carries the
    # service and the reference the service minted it for instead.
    who = str(row.get("email") or "") or (
        str(row.get("source") or "auto") + ":" + str(row.get("note") or "")[:60])
    raw, meta = mint_key(
        name="fleet-pass:" + who,
        expires_at=expires_at, max_rpd=rpd, max_rph=rph,
    )
    key_id = int(meta["id"])
    if kind == "single":
        model_id = str(models.get("model") or "")
        db_exec(
            "INSERT INTO agents(key_id,enabled,name,system_prompt,rules,"
            "allowed_models,force_model,param_overrides,ctx_limit,updated_at)"
            " VALUES (?,1,?,?,?,?,?,?,?,?)",
            (key_id, "fleet-pass", "", "", json.dumps([model_id]), model_id,
             json.dumps({"max_tokens": min(4096, ctx // 2)}), ctx, now()),
        )
    else:
        primary = str(models.get("primary") or "")
        workers = [str(w) for w in models.get("workers") or [] if str(w).strip()]
        db_exec(
            "INSERT INTO teams(key_id,enabled,name,primary_model,worker_models,"
            "max_workers,max_rounds,system_prompt,worker_prompt,ctx_limit,"
            "updated_at) VALUES (?,1,?,?,?,?,?,?,?,?,?)",
            (key_id, "fleet-pass", primary, json.dumps(workers),
             int(settings["team_max_workers"]), int(settings["team_max_rounds"]),
             str(settings.get("team_prompt") or ""),
             str(settings.get("worker_prompt") or ""), ctx, now()),
        )
    db_exec(
        "UPDATE public_keys SET status='issued', key_id=?, decided_at=?, "
        "decided_by=?, warm_token=? WHERE id=?",
        (key_id, now(), decided_by, secrets.token_urlsafe(24), row["id"]),
    )
    return raw


def _esc_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_key_email(row: dict, raw_key: str, settings: dict) -> tuple[str, str, str]:
    """(subject, text, html) for the "your key is ready" email."""
    kind = row["kind"]
    try:
        models = json.loads(row.get("models") or "{}")
    except (TypeError, json.JSONDecodeError):
        models = {}
    days = int(settings.get("key_days", 7))
    expires_date = (datetime.now(timezone.utc) + timedelta(days=days)).date().isoformat()
    base_url = str(settings.get("public_base_url") or PUBLIC_API_URL or "").rstrip("/")
    if kind == "single":
        setup_model = str(models.get("model") or "")
        model_line = 'model: "' + setup_model + '"'
        limit_line = (str(settings["single_rpd"]) + "/day, "
                     + str(settings["single_rph"]) + "/hour")
    else:
        primary = str(models.get("primary") or "")
        workers = ", ".join(str(w) for w in models.get("workers") or [])
        setup_model = "team"
        model_line = 'model: "team" (primary: ' + primary + "; workers: " + workers + ")"
        limit_line = (str(settings["team_rpd"]) + "/day, "
                     + str(settings["team_rph"]) + "/hour")
    subject = str(settings.get("email_subject")
                 or DEFAULT_PUBLIC_SETTINGS["email_subject"]).format(days=days)
    intro = str(settings.get("email_intro") or "")
    try:
        setup = str(settings.get("email_setup") or "").format(
            base_url=base_url, key=raw_key, model_id=setup_model,
            ctx=int(row.get("ctx") or 0))
    except (KeyError, IndexError):
        setup = str(settings.get("email_setup") or "")
    disclaimer = str(settings.get("email_disclaimer") or "")
    # The warm-up button: loads the key's model(s) on the best box the fleet
    # has for them, so the first real request answers quickly. It sits right
    # under the introduction, above the key, because it is the thing to click
    # first -- and because another key's request may have moved the model on
    # since this mail was sent.
    warm_url = warm_link(row, settings)
    warm_text = ""
    warm_html = ""
    if warm_url:
        what = ("your model" if kind == "single"
                else "all three of your team's models, on three machines")
        warm_text = ("-- load it now --\n"
                     "Open this link to load " + what + " before your first request "
                     "(takes a minute or two; the page shows progress):\n"
                     + warm_url + "\n\n")
        warm_html = (
            '<p style="margin:18px 0"><a href="' + _esc_html(warm_url) + '" '
            'style="display:inline-block;background:#1f6feb;color:#fff;'
            'padding:11px 18px;border-radius:6px;text-decoration:none;'
            'font-weight:600">'
            + ("Load my model now" if kind == "single" else "Load my team now")
            + "</a><br>"
            '<span style="color:#666;font-size:12px">Loads ' + _esc_html(what)
            + " on the fastest machine that can hold it, so your first request "
            "answers quickly. Takes a minute or two; the page shows progress.</span></p>"
        )
    text = (
        intro + "\n\n"
        + warm_text +
        "-- your key --\n"
        "base url: " + base_url + "\n"
        "key: " + raw_key + "\n" + model_line + "\n"
        "context: " + str(row.get("ctx")) + " tokens\n"
        "expires: " + expires_date + "\n"
        "limits: " + limit_line + "\n\n"
        "-- setup --\n" + setup + "\n\n"
        "-- please read --\n" + disclaimer + "\n"
    )
    html = (
        '<div style="font-family:system-ui,sans-serif;max-width:640px">'
        "<p>" + _esc_html(intro).replace("\n", "<br>") + "</p>"
        + warm_html +
        "<h3>Your key</h3>"
        '<pre style="background:#f4f4f4;padding:12px;border-radius:6px;'
        'white-space:pre-wrap">'
        "base url: " + _esc_html(base_url) + "\n"
        "key: " + _esc_html(raw_key) + "\n" + _esc_html(model_line) + "\n"
        "context: " + _esc_html(str(row.get("ctx"))) + " tokens\n"
        "expires: " + _esc_html(expires_date) + "\n"
        "limits: " + _esc_html(limit_line) + "</pre>"
        "<h3>Setup</h3>"
        '<pre style="background:#f4f4f4;padding:12px;border-radius:6px;'
        'white-space:pre-wrap">' + _esc_html(setup) + "</pre>"
        "<h3>Please read</h3><p>" + _esc_html(disclaimer).replace("\n", "<br>")
        + "</p></div>"
    )
    return subject, text, html


def public_key_bundle(row: dict, raw_key: str, settings: dict) -> dict:
    """The facts the key e-mail renders, as data -- for a connected service
    that hands the key on itself (inside a cover letter, a message) instead
    of receiving mail. `key` is the raw key and is present only on the one
    response that mints it; everything else can be recomputed from the row."""
    try:
        models = json.loads(row.get("models") or "{}")
    except (TypeError, json.JSONDecodeError):
        models = {}
    model_id = str(models.get("model") or models.get("primary") or "")
    cat = public_catalogue()["by_public"].get(model_id) or {}
    base_url = str(settings.get("public_base_url") or PUBLIC_API_URL or "").rstrip("/")
    if row.get("kind") == "team":
        rpd, rph = int(settings["team_rpd"]), int(settings["team_rph"])
    else:
        rpd, rph = int(settings["single_rpd"]), int(settings["single_rph"])
    expires_at, prefix = "", ""
    if row.get("key_id"):
        krows = db_query(
            "SELECT expires_at, max_rpd, max_rph, prefix FROM api_keys WHERE id=?",
            (row["key_id"],))
        if krows:
            expires_at = str(krows[0].get("expires_at") or "")
            rpd = int(krows[0].get("max_rpd") or rpd)
            rph = int(krows[0].get("max_rph") or rph)
            prefix = str(krows[0].get("prefix") or "")
    expires_date = expires_at[:10]
    limits = str(rpd) + " requests/day, " + str(rph) + "/hour"
    template = str(settings.get("auto_issue_setup_text") or "")
    # str.format also does attribute/index traversal ({model.name},
    # {key[0]}), so a typo in the tab's template can raise more than a
    # KeyError -- and this runs AFTER the key is minted, so it must never
    # take the response down with it. The raw template is the fallback.
    try:
        setup = template.format(base_url=base_url, key=raw_key, model=model_id,
                                expires=expires_date, limits=limits,
                                ctx=int(row.get("ctx") or 0))
    except Exception:  # noqa: BLE001 -- see above
        setup = template
    return {
        "key": raw_key, "key_prefix": prefix, "base_url": base_url,
        "model": model_id, "model_name": str(cat.get("name") or model_id),
        "ctx": int(row.get("ctx") or 0), "expires_at": expires_at,
        "expires_date": expires_date, "limit_day": rpd, "limit_hour": rph,
        "warm_url": warm_link(row, settings), "setup_text": setup,
    }


async def send_key_email(row: dict, raw_key: str) -> tuple[bool, str]:
    settings = get_public_settings()
    subject, text, html = render_key_email(row, raw_key, settings)
    return await send_mail(row["email"], subject, text, html)


async def send_pending_admin_mail(row: dict) -> None:
    settings = get_public_settings()
    to = str(settings.get("admin_notify") or "").strip()
    if not to:
        return
    subject = "Fleet Pass request pending: " + str(row["email"])
    body = (
        "A Fleet Pass request needs review.\n\n"
        "email: " + str(row["email"]) + "\n"
        "domain: " + str(row["domain"]) + "\n"
        "kind: " + str(row["kind"]) + "\n"
        "note: " + str(row.get("note") or "") + "\n\n"
        "Review it here: https://llm.example.com/admin/?tab=public\n"
    )
    ok, err = await send_mail(to, subject, body)
    if not ok:
        log_public_event("mail_error", email=str(row["email"]), detail=err)


async def send_deny_mail(row: dict, reason: str) -> tuple[bool, str]:
    subject = "About your open-fleet key request"
    body = (
        "Thanks for your interest in the open-fleet. We were not "
        "able to issue a key for this request"
        + ((": " + reason) if reason else ".") + "\n"
    )
    return await send_mail(row["email"], subject, body)


# ---- warm-up: the "load my model now" button in the key email -----------
#
# A key is issued against a model that may be cold, or warm on the wrong
# box, by the time its owner gets round to the first request. The email
# carries a link that loads the key's model -- or all of a team's models, on
# three different boxes -- on the best machine the fleet has for each, and
# fetches the weights onto a better box first when one is online, idle, and
# could hold them. Nothing happens at issue time: the owner asked for the
# button precisely so that nothing is kicked off pre-emptively.
#
# The link is the only credential: an unguessable per-key token, usable
# while the key is live, one warm per key per ten minutes. The page shows
# boxes as "Box N" like every other public surface.

_warm_jobs: dict[str, dict] = {}
WARM_COOLDOWN = 600.0
WARM_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


def warm_link(row: dict, settings: dict) -> str:
    if not settings.get("warm_button", True):
        return ""
    token = str(row.get("warm_token") or "")
    if not WARM_TOKEN_RE.match(token):
        return ""
    base = str(settings.get("public_base_url") or PUBLIC_API_URL or "").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    if not base:
        return ""
    return base + "/public/warm/" + token


def _warm_row(token: str) -> dict | None:
    if not WARM_TOKEN_RE.match(token or ""):
        return None
    rows = db_query("SELECT * FROM public_keys WHERE warm_token=? AND archived_at IS NULL",
                    (token,))
    return rows[0] if rows else None


def _warm_key(row: dict) -> tuple[dict | None, str]:
    """The api_keys row behind a public key, or why it cannot be used."""
    if row.get("status") != "issued" or not row.get("key_id"):
        return None, "this key is not active"
    rows = db_query("SELECT * FROM api_keys WHERE id=?", (row["key_id"],))
    if not rows or rows[0].get("archived_at") or rows[0].get("disabled"):
        return None, "this key has been revoked"
    exp = rows[0].get("expires_at")
    if exp:
        cutoff = exp + "T23:59:59+00:00" if len(exp) == 10 else exp
        if now() > cutoff:
            return None, "this key expired on " + str(exp)[:10]
    return rows[0], ""


def _warm_models(row: dict) -> list[tuple[str, str]]:
    """(role, public id) for every model the key may call."""
    try:
        models = json.loads(row.get("models") or "{}")
    except (TypeError, json.JSONDecodeError):
        models = {}
    if row.get("kind") == "single":
        m = str(models.get("model") or "").strip()
        return [("primary", m)] if m else []
    out = [("primary", str(models.get("primary") or ""))]
    out += [("worker", str(w)) for w in (models.get("workers") or [])]
    return [(r, m.strip()) for r, m in out if m.strip()]


async def _peer_admin(cand: str, method: str, subpath: str, body: Any = None,
                      timeout: float = 30.0) -> tuple[int, Any]:
    """One call to a peer's admin API with its stored token."""
    p = next((x for x in load_peers() if x["name"] == cand), None)
    if not p:
        raise RuntimeError("unknown peer")
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=timeout)) as c:
        r = await c.request(method, p["url"].rstrip("/") + "/admin/api/" + subpath,
                            headers={"Authorization": "Bearer " + p.get("token", "")},
                            json=body)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {"detail": r.text[:300]}


async def _peer_native(cand: str, path: str, body: Any,
                       timeout: float = 600.0) -> httpx.Response:
    """One call to an Ollama peer's native API, keyed like any client."""
    pk = await peer_inference_key(cand)
    p = next((x for x in load_peers() if x["name"] == cand), None)
    if not (pk and p):
        raise RuntimeError("no key for peer")
    async with httpx.AsyncClient(base_url=p["url"].rstrip("/"),
                                 timeout=httpx.Timeout(10.0, read=timeout)) as c:
        return await c.post(path, json=body, headers={"Authorization": "Bearer " + pk})


def _warm_better_host(current_tier: int, fids: list[str], role: str,
                      taken: set[str]) -> dict | None:
    """A reachable box that does not serve this model yet but would rank
    ahead of the best box that does -- and has a compatible source to fetch
    it from (a GGUF by repo/file for a llama.cpp box, a tag for an Ollama
    one). A box with a request in flight is left alone: registering a model
    restarts llama-swap there."""
    cands = _routes_cache.get("cands", {})
    meta_all = _routes_cache.get("meta", {})
    engines = _routes_cache.get("engine", {})
    reachable = _routes_cache.get("reachable") or set()
    serving = {(c or HOST_NAME) for f in fids for c in cands.get(f, [])}
    src = None
    for f in fids:
        for c in cands.get(f, []):
            m = meta_all.get((c, f)) or {}
            s = m.get("source") or {}
            if m.get("bytes") and isinstance(s, dict) and s:
                src = {"cand": c, "fid": f, "bytes": int(m["bytes"]),
                       "moe": bool(m.get("moe")), "source": s}
                break
        if src:
            break
    if not src:
        return None
    gguf = "repo" in src["source"]
    specs = load_specs()
    dark = eclipsed_hosts()
    best = None
    for host in reachable:
        if host == HOST_NAME or host in serving or host in taken or host in dark:
            continue
        if _inflight.get(host, 0) > 0:
            continue
        klass = host_class(host)
        if klass not in ("gpu", "big"):
            continue
        eng = engines.get(host, "")
        if eng != ("llama-swap" if gguf else "ollama"):
            continue
        spec = specs.get(host, {})
        try:
            vram = float(spec.get("vram_gb") or 0) * GIB
            ram = float(spec.get("ram_gb") or 0) * GIB
            bw = float(spec.get("mem_bw_gbs") or 0)
        except (TypeError, ValueError):
            continue
        need = src["bytes"] + GIB
        fits = need <= vram * 0.9
        tier = 1 if klass == "big" else 0
        if not fits and klass == "gpu" and src["moe"] and spec.get("moe_spill_ok"):
            fits = need <= (vram + ram) * 0.8
        if not fits or tier >= current_tier:
            continue
        key = (tier, -bw)
        if best is None or key < best[0]:
            best = (key, {"host": host, "tier": tier, "src": src})
    return best[1] if best else None


async def plan_warm(row: dict) -> list[dict]:
    """One item per model the key can call: the box it should be warm on, and
    whether that box must first fetch it. A team's models land on different
    boxes wherever the fleet has them to give."""
    await model_routes(force=True)
    cat = public_catalogue(force=True)
    taken: set[str] = set()
    items: list[dict] = []
    for role, pid in _warm_models(row):
        prow = cat["by_public"].get(pid)
        fids = _row_fleet_ids(prow) if prow else [pid]
        name = str((prow or {}).get("name") or pid)
        try:
            ranked = await resolve_targets(pid, role=role, prompt_tokens=4096, gen_tokens=512)
        except Exception:  # noqa: BLE001
            ranked = []
        pick = next((t for t in ranked if (t[0] or HOST_NAME) not in taken),
                    ranked[0] if ranked else None)
        item: dict[str, Any] = {
            "role": role, "public_id": pid, "name": name, "phase": "queued",
            "detail": "", "progress": None, "box": "", "action": "none",
            "cand": None, "fleet_id": "", "host": "",
        }
        cur_tier = host_tier(pick[0], pick[1], role)[0] if pick else 9
        if pick:
            hname = pick[0] or HOST_NAME
            item.update(action="load", cand=pick[0], fleet_id=pick[1], host=hname,
                        box=public_alias(hname))
        better = _warm_better_host(cur_tier, fids, role, taken) if fids else None
        if better:
            item.update(action="download", target=better["host"], src=better["src"],
                        box=public_alias(better["host"]))
        if item.get("target"):
            taken.add(str(item["target"]))
        elif item["host"]:
            taken.add(str(item["host"]))
        if not pick and not better:
            item.update(phase="failed",
                        detail="no machine in the fleet serves this model right now")
        items.append(item)
    return items


async def _warm_provision(item: dict) -> None:
    """Fetch the model onto `target` and register it there, then point the
    item at that box. Raises when the box cannot be made to serve it."""
    target = str(item["target"])
    src = item["src"]
    source = src["source"]
    fid = str(src["fid"])
    item.update(phase="downloading", detail="fetching the weights onto " + item["box"])
    if "repo" in source:
        st, d = await _peer_admin(target, "POST", "download",
                                  {"repo": source["repo"], "filename": source["file"]})
        dest = ""
        job_id = 0
        if st == 200:
            job_id = int(d.get("job_id") or 0)
            dest = str(d.get("dest") or "")
        elif st == 409 and "already downloaded" in str(d.get("detail", "")):
            dest = str(d.get("detail")).split("already downloaded:", 1)[-1].strip()
        else:
            raise RuntimeError("download refused (HTTP " + str(st) + ")")
        while job_id:
            await asyncio.sleep(3.0)
            st, jobs = await _peer_admin(target, "GET", "jobs")
            j = next((x for x in (jobs if isinstance(jobs, list) else [])
                      if int(x.get("id") or 0) == job_id), None)
            if not j:
                raise RuntimeError("download job vanished")
            status = str(j.get("status") or "")
            total = int(j.get("bytes_total") or 0)
            done = int(j.get("bytes_done") or 0)
            item["progress"] = round(done / total, 3) if total else None
            if status == "done":
                dest = str(j.get("dest") or dest)
                break
            if status in ("error", "cancelled"):
                raise RuntimeError("download " + status)
        item.update(phase="registering", detail="registering on " + item["box"],
                    progress=None)
        # Registering restarts llama-swap on the target, which would abort
        # anything it is serving. The download may have taken minutes; the box
        # chosen because it was idle at plan time may not be now. If it has
        # since picked up traffic, do not touch it -- _warm_run falls back to
        # the box that was already serving the model.
        if _inflight.get(target, 0) > 0:
            raise RuntimeError("target is now serving live traffic; deferring registration")
        st, models = await _peer_admin(target, "GET", "models")
        configured = list((models or {}).get("configured") or []) if st == 200 else []
        if not any(str(m.get("id")) == fid for m in configured):
            if src["cand"]:
                _st, srcm = await _peer_admin(str(src["cand"]), "GET", "models")
                src_list = (srcm or {}).get("configured") or []
            else:
                src_list = load_models()
            rec = dict(next((m for m in src_list if str(m.get("id")) == fid), {}) or {})
            # The box sizes the window itself (ctx 0) and keeps the whole
            # model on its GPU: a box chosen here was chosen because it fits.
            # Aliases stay behind -- they are the source box's names.
            rec.update(id=fid, path=dest, enabled=True, aliases=[], ctx=0,
                       n_cpu_moe=0, mmproj="")
            rec.setdefault("description", item["name"])
            configured.append(rec)
            st3, out = await _peer_admin(target, "PUT", "models",
                                         {"models": configured}, timeout=180.0)
            if st3 != 200:
                raise RuntimeError("could not register (HTTP " + str(st3) + ")")
    else:
        tag = str(source.get("tag") or fid)
        r = await _peer_native(target, "/api/pull", {"model": tag, "stream": False},
                               timeout=3600.0)
        if r.status_code != 200:
            raise RuntimeError("pull failed (HTTP " + str(r.status_code) + ")")
    _routes_cache["t"] = 0.0
    await model_routes(force=True)
    item.update(action="load", cand=target, host=target, fleet_id=fid)


async def _warm_load(item: dict, key_row: dict) -> None:
    cand = item.get("cand")
    host = str(item.get("host") or HOST_NAME)
    fid = str(item.get("fleet_id") or "")
    if cand is None or not fid:
        raise RuntimeError("nothing to load")
    item.update(phase="loading", detail="loading on " + item["box"])
    started = time.time()
    engine = _routes_cache.get("engine", {}).get(host, "")
    usage = None
    if engine == "ollama" and cand:
        # An empty generate is Ollama's own "load and hold" -- and the one
        # way to ask for a keep_alive, which /v1 cannot express.
        r = await _peer_native(cand, "/api/generate",
                               {"model": fid, "keep_alive": "30m"}, timeout=900.0)
        status = r.status_code
    else:
        payload = json.dumps({"model": fid, "max_tokens": 1, "stream": False,
                              "messages": [{"role": "user", "content": "Reply with OK."}]
                              }).encode()
        r = await _post_chat(cand, payload, 900.0)
        status = r.status_code
        try:
            usage = (r.json() or {}).get("usage")
        except ValueError:
            usage = None
    record_usage(key_row, str(item.get("public_id") or fid), "/v1/warm", False,
                 status, usage, None, int((time.time() - started) * 1000), host=host)
    if status >= 400:
        _mark_host_down(host, COOLDOWN_UPSTREAM_5XX if status >= 500 else 5.0,
                        "warm-up answered HTTP " + str(status))
        raise RuntimeError("the machine answered HTTP " + str(status))
    _mark_host_ok(host)
    _routes_cache["t"] = 0.0
    item.update(phase="ready", detail="ready on " + item["box"], progress=1.0)


async def _warm_run(token: str) -> None:
    job = _warm_jobs[token]

    async def one(item: dict) -> None:
        try:
            if item["phase"] == "failed":
                return
            if item["action"] == "download":
                try:
                    await _warm_provision(item)
                except Exception as exc:  # noqa: BLE001
                    log.warning("warm: could not provision %s on %s -- %s",
                                item["public_id"], item.get("target"), exc)
                    if item.get("cand") is None:
                        raise
                    item["box"] = public_alias(str(item["host"]))
            await _warm_load(item, job["key"])
        except Exception as exc:  # noqa: BLE001
            log.warning("warm: %s failed -- %s", item["public_id"], exc)
            item.update(phase="failed",
                        detail="could not load this model right now; your first "
                               "request will load it instead")

    await asyncio.gather(*(one(i) for i in job["items"]))
    job["done"] = True
    job["finished"] = time.time()
    try:
        db_exec("UPDATE public_keys SET warmed_at=? WHERE warm_token=?", (now(), token))
    except sqlite3.Error:
        pass


def _warm_view(row: dict, job: dict | None, message: str = "") -> dict:
    items = [{"name": i["name"], "role": i["role"], "box": i.get("box") or "",
              "phase": i["phase"], "detail": i.get("detail") or "",
              "progress": i.get("progress")}
             for i in (job or {}).get("items", [])]
    return {"ok": True, "kind": row.get("kind"), "items": items,
            "done": bool((job or {}).get("done")),
            "started": (job or {}).get("started"), "message": message}


def _warm_html(token: str | None, blocked: str, kind: str = "single") -> str:
    tok = token if (token and WARM_TOKEN_RE.match(token)) else ""
    note = _esc_html(blocked) if blocked else ""
    plural = kind == "team"
    return (
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>open-fleet - warm-up</title>"
        "<style>body{font:16px/1.5 system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 20px;color:#222}"
        "h1{font-size:22px}.item{border:1px solid #ddd;border-radius:8px;padding:12px 14px;margin:10px 0}"
        ".name{font-weight:600}.box{color:#666;font-size:13px}.phase{display:inline-block;padding:2px 8px;"
        "border-radius:10px;font-size:12px;background:#eee;margin-left:8px}.ready{background:#d7f5dd}"
        ".failed{background:#fde0e0}.loading,.downloading,.registering{background:#fff1c9}"
        ".bar{height:6px;background:#eee;border-radius:3px;margin-top:8px;overflow:hidden}"
        ".bar i{display:block;height:100%;background:#1f6feb;width:0}.muted{color:#666}"
        "</style>"
        "<h1>Warming up your " + ("team" if plural else "model") + "</h1>"
        + ("<p class=muted>" + note + "</p>" if note else
           "<p class=muted id=msg>Asking the fleet&hellip;</p>")
        + "<div id=items></div>"
        + ("<p class=muted>You can close this page; the load carries on. Your first "
           "request will answer as soon as the status above says ready.</p>" if tok and not note else "")
        + ("<script>const T=" + json.dumps(tok) + ";"
           "function esc(s){return String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}"
           "function render(d){const el=document.getElementById('items');"
           "el.innerHTML=(d.items||[]).map(i=>`<div class=item><span class=name>${esc(i.name)}</span>"
           "<span class='phase ${esc(i.phase)}'>${esc(i.phase)}</span><div class=box>${esc(i.role==='worker'?'sub-agent':'primary')}"
           "${i.box?' &middot; '+esc(i.box):''}${i.detail?' &middot; '+esc(i.detail):''}</div>"
           "${i.progress!=null&&i.phase==='downloading'?`<div class=bar><i style='width:${Math.round(i.progress*100)}%'></i></div>`:''}</div>`).join('');"
           "const m=document.getElementById('msg');if(m){m.textContent=d.done?"
           "((d.items||[]).every(i=>i.phase==='ready')?'All set - your first request will answer quickly.':'Done, with some models left for your first request to load.')"
           ":(d.message||'Working');}}"
           "async function poll(){try{const r=await fetch('/public/warm/'+T+'/status');const d=await r.json();render(d);"
           "if(!d.done)setTimeout(poll,3000);}catch(e){setTimeout(poll,5000);}}"
           "async function start(){try{const r=await fetch('/public/warm/'+T+'/start',{method:'POST'});const d=await r.json();"
           "if(!d.ok){document.getElementById('msg').textContent=d.error||'This link cannot be used.';return;}render(d);setTimeout(poll,2500);}"
           "catch(e){document.getElementById('msg').textContent='Could not reach the fleet; try again in a moment.';}}"
           "if(T)start();</script>" if tok and not note else "")
    )


@app.get("/public/warm/{token}", response_class=HTMLResponse)
async def public_warm_page(token: str) -> HTMLResponse:
    row = _warm_row(token)
    if not row:
        return HTMLResponse(_warm_html(None, "This link is not valid."), status_code=404)
    _key, why = _warm_key(row)
    return HTMLResponse(_warm_html(token, why, str(row.get("kind") or "single")))


@app.post("/public/warm/{token}/start")
async def public_warm_start(token: str, request: Request) -> JSONResponse:
    row = _warm_row(token)
    if not row:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
    key, why = _warm_key(row)
    if not key:
        return JSONResponse({"ok": False, "error": why}, status_code=403)
    job = _warm_jobs.get(token)
    if job and not job.get("done"):
        return JSONResponse(_warm_view(row, job, "already in progress"))
    # One warm per key per ten minutes, keyed on when the last run FINISHED,
    # not on whether every model came up. A model the fleet cannot serve marks
    # its item failed forever, so gating on "all ready" would leave the button
    # an unthrottled way to re-run real backend work (downloads, keep-alive
    # loads) for the models that do resolve. plan_warm does live network I/O,
    # so the placeholder is written BEFORE the first await -- otherwise two
    # near-simultaneous clicks both see no job and both launch a run against
    # the same boxes.
    if (job and job.get("done")
            and time.time() - float(job.get("finished") or 0) < WARM_COOLDOWN):
        return JSONResponse(_warm_view(row, job, "warmed a moment ago"))
    placeholder = {"items": [], "key": key, "started": time.time(),
                   "done": False, "finished": None}
    _warm_jobs[token] = placeholder
    try:
        items = await plan_warm(row)
    except Exception as exc:  # noqa: BLE001 -- a planning failure must clear the slot
        log.warning("warm: planning failed for %s -- %s", token[:8], exc)
        placeholder.update(done=True, finished=time.time())
        _warm_jobs.pop(token, None)
        return JSONResponse({"ok": False,
                             "error": "could not reach the fleet; try again in a moment"},
                            status_code=503)
    placeholder["items"] = items
    log_public_event("warm", email=str(row["email"]), ip=client_ip(request),
                     detail=", ".join(i["public_id"] + "@" + (i.get("box") or "?")
                                      for i in items))
    placeholder["task"] = asyncio.create_task(_warm_run(token))
    return JSONResponse(_warm_view(row, placeholder, "starting"))


@app.get("/public/warm/{token}/status")
async def public_warm_status(token: str) -> JSONResponse:
    row = _warm_row(token)
    if not row:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
    job = _warm_jobs.get(token)
    return JSONResponse(_warm_view(row, job, "" if job else "not started"))



# --------------------------------------------------------------------------
# keeping the suggested model resident
# --------------------------------------------------------------------------
#
# The warm-up button above fixes one key's first request. This fixes everybody
# else's: the model the public page suggests is the one most visitors will
# pick, and a cold load of a dense 27B is a minute or two of a recruiter
# watching a spinner before the demo has said a word.
#
# No configuration flag does this. llama-swap holds exactly one model per box
# -- one exclusive swap group, deliberately -- and drops it after its ttl, so
# residency is not a property a box can be told to have; it is the result of
# the last thing that asked. So this is a loop: every couple of minutes, on
# each box that could hold the suggested model and is not doing anything, ask
# it for one token. That loads it when it is missing and resets the ttl clock
# when it is not.
#
# Three rules keep it from being a nuisance, and they are the whole design:
#
#   * it never touches a box with a request in flight, or one that finished
#     one in the last few minutes. A conversation in progress outranks a
#     conversation that has not started.
#   * it only touches a box that ALREADY serves the model. It will not fetch
#     weights in the background; the warm-up button does that, with somebody
#     watching, because it is tens of gigabytes.
#   * it only runs on the box that fronts the fleet. Every box runs this same
#     app, and a fleet whose members each pinned their own idea of the
#     suggested model would fight the hub for the swap slot.

PRELOAD_TICK = 120.0          # how often the loop looks
PRELOAD_REFRESH = 600.0       # re-touch an already-resident model this often
PRELOAD_IDLE_GRACE = 300.0    # leave a box alone this long after it served
PRELOAD_RETRY = 900.0         # back off this long after a box refuses
PRELOAD_LOAD_TIMEOUT = 900.0  # a cold dense-27B load, with room to spare

# host -> what the loop last did there and why, for the Public tab. Purely a
# report; nothing routes off it.
_preload_state: dict[str, dict] = {}


def _preload_note(phase: str, detail: str = "", **fields: Any) -> None:
    """The host is taken out of `fields` rather than passed beside it, so a
    caller can splat the plan entry it is already holding."""
    host = str(fields.pop("host", "") or "")
    st = _preload_state.setdefault(host, {})
    # Guarded: public_alias() ASSIGNS a Box number on first sight and persists
    # it, so handing it a blank would mint one for a host that does not exist.
    st.update(host=host, box=public_alias(host) if host else "", phase=phase,
              detail=detail, at=now(), **fields)


def preload_capable(cand: str, fleet_id: str) -> bool:
    """Can this box hold the whole model in its own memory?

    The same reading of the served-models `fit` report that earns a box tier 0
    in the routing scorer, including the spec sheet's `moe_spill_ok`
    concession for a mixture of experts whose experts overflow into RAM. A box
    that would only run this model by spilling it is not a box to pin it on:
    it would answer slower than the fallback, and would hold its swap slot
    hostage to do it."""
    meta = _routes_cache.get("meta", {}).get((cand, fleet_id)) or {}
    fit = str(meta.get("fit") or "")
    if fit in ("vram", "unified"):
        return True
    spec = load_specs().get(cand or HOST_NAME, {})
    return fit == "spill" and bool(meta.get("moe")) and bool(spec.get("moe_spill_ok"))


def preload_orchestrator() -> bool:
    """Whether this box is the one that drives the loop: the fleet's hub, with
    peers to drive.

    "Has peers" alone used to be the whole test, on the theory that only the
    hub has any. On 2026-08-29 three boxes did -- hub, gpu-laptop-1 (given
    apu-box-1 as a peer for the public tour) and apu-box-1 (given three so its
    workers could leave) -- and each ran this loop, each pinning the featured
    27B onto every capable box every ten minutes. On apu-box-1 that evicted the
    125B it exists to hold, on a timer, and nobody had asked for anything.
    The spec sheet already names the hub (`role: hub`); that is the test now.
    LLMSTACK_PRELOAD=off disarms it anywhere, and =on forces it for a fleet
    whose hub is not on the sheet."""
    flag = os.environ.get("LLMSTACK_PRELOAD", "").strip().lower()
    if flag in ("0", "off", "false", "no"):
        return False
    if flag in ("1", "on", "true", "yes"):
        return bool(load_peers())
    hub = str(load_specs().get(HOST_NAME, {}).get("role") or "") == "hub"
    return hub and bool(load_peers())


def preload_plan() -> list[dict]:
    """One entry per box the loop would touch right now, plus a recorded
    reason for every box it is passing over. Reads the routing cache as it
    stands -- the caller refreshes it -- so this stays synchronous."""
    settings = get_public_settings()
    if not settings.get("preload_featured", True):
        _preload_state.clear()
        return []
    pid = featured_public_id(settings)
    row = public_catalogue()["by_public"].get(pid) if pid else None
    if not row:
        _preload_state.clear()
        return []
    cands = _routes_cache.get("cands", {})
    running = _routes_cache.get("running", {})
    reachable = _routes_cache.get("reachable") or set()
    dark = eclipsed_hosts()
    tnow = time.time()
    plan: list[dict] = []
    seen: set[str] = set()
    for fid in _row_fleet_ids(row):
        for cand in cands.get(fid, []):
            host = cand or HOST_NAME
            if host in seen or host not in reachable or host in dark:
                continue
            if not preload_capable(cand, fid):
                # Deliberately not noted: a box that cannot hold the model is
                # not passing anything over, and listing every small box as
                # "skipped" would bury the states that matter.
                continue
            if host_reserved(host):
                # Somebody's personal machine. It routes only as a last
                # resort (see host_reserved), so keeping the featured model
                # warm on it would cost its owner memory for traffic it
                # almost never takes.
                seen.add(host)
                _preload_note("reserve", "personal box -- not kept warm",
                              host=host, cand=cand, fleet_id=fid,
                              public_id=pid, resident=False)
                continue
            own = _routes_cache.get("warm", {}).get(host) or set()
            if own and fid not in own:
                # The box has a job of its own -- a model it preloads or
                # keeps resident (models.json `preload` / `persistent`) --
                # and it is not this one. Touching it would evict what it
                # exists to hold. Its owner decided what it is warm with.
                seen.add(host)
                _preload_note("dedicated", "holds " + ", ".join(sorted(own)),
                              host=host, cand=cand, fleet_id=fid,
                              public_id=pid, resident=False)
                continue
            seen.add(host)
            st = _preload_state.get(host) or {}
            resident = fid in running.get(host, set())
            common = {"host": host, "cand": cand, "fleet_id": fid,
                      "public_id": pid, "resident": resident}
            if _inflight.get(host, 0):
                _preload_note("busy", "serving a request", **common)
                continue
            if tnow - _host_last_used.get(host, 0.0) < PRELOAD_IDLE_GRACE:
                _preload_note("in use", "answered a request just now", **common)
                continue
            if host_cooling(host):
                _preload_note("cooling", "cooled down after a failure", **common)
                continue
            if tnow < float(st.get("retry_at") or 0.0):
                _preload_note("waiting", str(st.get("detail") or "backing off"),
                              **common)
                continue
            if resident and tnow - float(st.get("touched") or 0.0) < PRELOAD_REFRESH:
                _preload_note("resident", "loaded and inside its ttl", **common)
                continue
            plan.append(common)
    return plan


async def _preload_touch(item: dict) -> None:
    """Ask one box for a single token of the pinned model. Loads it when it is
    missing and restarts the ttl clock when it is not -- llama-swap counts
    that clock from the last request, not from the load."""
    cand, host, fid = item["cand"], item["host"], item["fleet_id"]
    engine = _routes_cache.get("engine", {}).get(host, "")
    if engine == "ollama" and cand:
        # Ollama's own load-and-hold, and the only way to ask it for a
        # keep_alive -- /v1 cannot express one.
        r = await _peer_native(cand, "/api/generate",
                               {"model": fid, "keep_alive": "30m"},
                               timeout=PRELOAD_LOAD_TIMEOUT)
        status = r.status_code
    else:
        payload = json.dumps({"model": fid, "max_tokens": 1, "stream": False,
                              "messages": [{"role": "user", "content": "Reply with OK."}]
                              }).encode()
        r = await _post_chat(cand, payload, PRELOAD_LOAD_TIMEOUT)
        status = r.status_code
    if status >= 400:
        _mark_host_down(host, COOLDOWN_UPSTREAM_5XX if status >= 500 else 5.0,
                        "preload answered HTTP " + str(status))
        raise RuntimeError("HTTP " + str(status))
    _mark_host_ok(host)
    # Deliberately not metered. record_usage() rows are what a key is billed
    # and rate-limited against, and this is the gateway talking to itself on
    # nobody's behalf; a row here would surface in somebody's "requests today".
    _preload_note("resident", "kept warm" if item["resident"] else "reloaded",
                  host=host, cand=cand, fleet_id=fid, public_id=item["public_id"],
                  resident=True, touched=time.time(), retry_at=0.0)


async def preload_pass(force: bool = False) -> list[dict]:
    """One convergence pass. Answers what it touched, for the admin trigger."""
    if not (force or preload_orchestrator()):
        return []
    await model_routes()
    plan = preload_plan()
    if not plan:
        return []

    async def one(item: dict) -> None:
        _preload_note("refreshing" if item["resident"] else "loading",
                      "asking for one token", **item)
        try:
            await _preload_touch(item)
        except Exception as exc:  # noqa: BLE001 -- an unreachable box waits
            log.warning("preload: %s would not load %s -- %s",
                        item["host"], item["fleet_id"], exc)
            _preload_note("failed", str(exc)[:160],
                          retry_at=time.time() + PRELOAD_RETRY, **item)

    # Every box in the plan is separate hardware doing its own load, so they
    # go together; one at a time would make a five-box fleet sit through five
    # cold loads in series to converge after a restart.
    await asyncio.gather(*(one(i) for i in plan), return_exceptions=True)
    _routes_cache["t"] = 0.0
    return plan


def preload_report() -> dict:
    """What the loop is doing, for the Public tab."""
    settings = get_public_settings()
    return {
        "enabled": bool(settings.get("preload_featured", True)),
        "model": featured_public_id(settings),
        "orchestrator": preload_orchestrator(),
        "hosts": sorted((dict(v) for v in _preload_state.values()),
                        key=lambda h: str(h.get("box") or h.get("host") or "")),
    }


async def _preload_loop() -> None:
    """Runs everywhere, does something only on the box with peers -- rechecked
    every tick rather than once at startup, so adding the first peer starts it
    working without a restart."""
    await asyncio.sleep(20)  # let the first routing refresh land
    while True:
        try:
            if preload_orchestrator():
                await preload_pass()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- convergence never takes the app down
            log.exception("preload pass failed")
        await asyncio.sleep(PRELOAD_TICK)


# ---- public routes ----------------------------------------------------
#
# No bearer key -- the public site calls these directly from a public page.
# X-Forwarded-Client-IP (set by the public site's own proxy) is trusted over
# the socket peer, which would otherwise always read as the public site itself.


def client_ip(request: Request) -> str:
    """The address the per-IP limits are keyed on.

    Every caller reaches this process as 127.0.0.1 -- the public internet
    through cloudflared, the public site through the tailnet forward -- so the
    socket peer says nothing. What does: a request that came through
    Cloudflare carries CF-Connecting-IP, which Cloudflare sets and a client
    cannot remove, and that is the only address it is allowed to claim. Only
    a request that did NOT come through the edge (the public site, on the
    tailnet) gets to name the browser it is acting for in
    X-Forwarded-Client-IP. Otherwise a leaked intake token would also buy
    unlimited IP-limit bypass, one forged header per request."""
    cf = request.headers.get("cf-connecting-ip", "").strip()
    if cf:
        return cf
    fwd = request.headers.get("x-forwarded-client-ip", "").strip()
    if fwd:
        return fwd
    return request.client.host if request.client else ""


def public_alias(host: str) -> str:
    """The public-facing label for a real host -- 'Hub' for HOST_NAME,
    otherwise the 'Box N' assigned on first sight and persisted so it never
    shifts as peers come and go. Public surfaces never show a real host
    name or OS data (owner requirement, added after the first draft)."""
    host = str(host or "")
    if host == HOST_NAME:
        return "Hub"
    # A dual-boot machine keeps ONE public identity across both of its
    # operating systems -- "Box 7" is a laptop, not a Windows install -- so
    # the replacement borrows the alias of the host it replaces. One hop only:
    # this is operator config, and a chain is a mistake, not a feature.
    twin = str((load_specs().get(host) or {}).get("replaces") or "")
    if twin and twin != host:
        host = twin
        if host == HOST_NAME:
            return "Hub"
    rows = db_query("SELECT box FROM public_aliases WHERE host=?", (host,))
    if rows:
        return f"Box {rows[0]['box']}"
    used = {r["box"] for r in db_query("SELECT box FROM public_aliases")}
    box = 1
    while box in used:
        box += 1
    try:
        db_exec("INSERT INTO public_aliases(host, box, created_at) VALUES(?,?,?)",
                (host, box, now()))
    except sqlite3.IntegrityError:
        # Two first-sight requests raced for the same host (or the same
        # number); whichever landed is the alias from now on.
        rows = db_query("SELECT box FROM public_aliases WHERE host=?", (host,))
        if rows:
            return f"Box {rows[0]['box']}"
        return public_alias(host)
    return f"Box {box}"


_PUBLIC_ENGINE_KIND = {"llama-swap": "llama.cpp", "ollama": "ollama", "none": "none"}


async def public_models_payload() -> list[dict]:
    """Every enabled catalogue row, with live availability against the
    current routing table -- shared by /public/api/models and the models
    section of /public/api/overview."""
    await model_routes()
    cands = _routes_cache.get("cands", {})
    running = _routes_cache.get("running", {})
    settings = get_public_settings()
    # Enabled first, THEN ordered: a disabled row must not get a vote on where
    # its family sits, since nobody will ever see it hold that place.
    rows = order_public_models(
        [r for r in public_catalogue()["by_public"].values() if r.get("enabled")],
        settings,
    )
    featured = featured_public_id(settings)
    out = []
    for row in rows:
        hosts: set[str] = set()
        resident = False
        for fid in _row_fleet_ids(row):
            for cand in cands.get(fid, []):
                hname = cand or HOST_NAME
                hosts.add(hname)
                if fid in running.get(hname, set()):
                    resident = True
        availability = "resident" if resident else ("available" if hosts else "offline")
        cx = catalogue_ctx(row)
        ctx_max = int(cx["best"])
        out.append({
            "public_id": row["public_id"], "name": row["name"], "vendor": row["vendor"],
            # The list arrives in presentation order, so a client that just
            # renders it is already right; `family` and `featured` are for one
            # that wants to draw the group headings and the badge as well.
            "family": public_family(row),
            "featured": row["public_id"] == featured,
            "description": row["description"], "arch": row["arch"],
            "params_b": row["params_b"], "active_b": row["active_b"],
            "allow_primary": bool(row["allow_primary"]), "allow_worker": bool(row["allow_worker"]),
            # ctx_max is the best case across the whole fleet, awake or not --
            # that is what a key can be issued for. ctx_max_online is what is
            # reachable this second; when it is lower, a request arriving now
            # gets a disclosed reduction. ctx_hosts says where the top lives.
            "ctx_max": ctx_max,
            "ctx_max_online": int(cx["online"]),
            "ctx_default": min(int(row["ctx_default"] or 8192), ctx_max),
            "ctx_ceiling": int(row["ctx_max"] or 0),
            "ctx_hosts": [{"host": public_alias(h), "ctx": int(c)}
                          for h, c in sorted(cx["hosts"].items(),
                                             key=lambda kv: (-kv[1], kv[0]))],
            "availability": availability,
            "hosts": sorted({public_alias(h) for h in hosts}),
        })
    return out


def _sanitize_public_host(h: dict, specs: dict) -> dict:
    """Whitelist, not blacklist: every field on the sanitized host is named
    explicitly, so nothing new host_status() starts reporting later (an IP,
    a mount, a token) can leak into a public response by omission. The real
    host name and OS never appear here at all -- only the persisted 'Box N'
    / 'Hub' alias (owner requirement)."""
    real_name = str(h.get("name") or "")
    alias = public_alias(real_name)
    spec = specs.get(real_name, {})
    status = h.get("status") or {}
    host = status.get("host") or {}
    engine = host.get("engine") if isinstance(host.get("engine"), dict) else {}
    mem = host.get("mem") if isinstance(host.get("mem"), dict) else {}
    gpu = host.get("gpu") if isinstance(host.get("gpu"), list) else []
    # The BIGGEST card, not the first one listed. On a hybrid laptop the
    # integrated GPU enumerates first, and taking it made the fleet page
    # advertise a hybrid laptop's RTX 4070 as having 512 MB of VRAM.
    vram_gpu = max(
        (g for g in gpu if isinstance(g, dict) and g.get("vram_total")),
        key=lambda g: int(g.get("vram_total") or 0), default={})
    cat = public_catalogue()
    serving = []
    for m in (status.get("models_running") or []):
        if not isinstance(m, dict):
            continue
        mid = str(m.get("id") or "")
        if not mid:
            continue
        serving.append({"model": cat["by_fleet"].get(mid, mid), "n_ctx": m.get("n_ctx")})
    online = bool(h.get("online"))
    # An offline box contributes its spec sheet and nothing else: the hub
    # still holds its last-known snapshot, but a public card saying "CPU 3%"
    # about a machine that is switched off is a lie with a decimal point.
    if not online:
        mem, vram_gpu, serving, host = {}, {}, [], {}
    return {
        "name": alias,
        "online": online,
        "role": spec.get("role") or "inference",
        "last_seen": h.get("last_seen"),
        "specs": {k: spec.get(k) for k in
                 ("cpu", "gpu", "ram_gb", "vram_gb", "mem_bw_gbs", "gpu_tflops")},
        # Kind only -- no version string, no engine display name, no OS.
        "engine": _PUBLIC_ENGINE_KIND.get(str(engine.get("kind") or ""), "none") if engine else None,
        "mem": ({"total": mem.get("total"), "used": mem.get("used")} if mem else None),
        "vram": ({"total": vram_gpu.get("vram_total"), "used": vram_gpu.get("vram_used")}
                if vram_gpu else None),
        "cpu_count": host.get("cpu_count"),
        "cpu_percent": host.get("cpu_percent"),
        "gpu_busy": vram_gpu.get("busy_percent"),
        "serving": serving,
        "in_flight": _inflight.get(real_name, 0) if online else 0,
    }


async def public_overview_payload() -> dict:
    raw = await _gather_fleet_overview()
    specs = load_specs()
    hosts = [_sanitize_public_host(h, specs) for h in raw["hosts"]]
    models = await public_models_payload()
    return {
        "generated_at": raw["generated_at"],
        "hosts": hosts, "models": models,
        "totals": {"hosts": len(hosts),
                  "online": sum(1 for h in hosts if h["online"]),
                  "models": len(models)},
    }


_public_overview_cache: dict[str, Any] = {"t": 0.0, "data": None}


@app.get("/public/api/overview")
async def public_overview() -> dict:
    """Always answered from the copy the warm loop keeps -- a refresh polls
    every peer and an offline one costs its whole connect timeout, which is
    not something a visitor should wait on. Only the very first call, before
    the loop has produced anything, builds the payload inline."""
    if _public_overview_cache["data"] is not None:
        return _public_overview_cache["data"]
    data = await public_overview_payload()
    _public_overview_cache.update(t=time.time(), data=data)
    return data


@app.get("/public/api/models")
async def public_models_route() -> dict:
    settings = get_public_settings()
    return {
        "models": await public_models_payload(),
        # Presentation is the hub's call, not the portal's: `models` is
        # already in the order to render, `featured` is the row to lead with
        # and pre-select on the request form, and `family_order` is there for
        # a page that draws group headings of its own.
        "featured": featured_public_id(settings),
        "family_order": settings["model_family_order"],
        "limits": {
            "single_rpd": settings["single_rpd"], "single_rph": settings["single_rph"],
            "team_rpd": settings["team_rpd"], "team_rph": settings["team_rph"],
            "key_days": settings["key_days"], "team_max_workers": settings["team_max_workers"],
        },
    }


def _validate_public_model_selection(
    payload: dict, email: str = "", ip: str = "", log: bool = True,
) -> tuple[str, dict, list[int]]:
    """The kind/model part of the intake validation order (contract 1.6):
    single needs an enabled public id; team needs an enabled+allow_primary
    primary plus 1-2 distinct enabled+allow_worker workers. Shared by the
    public intake and the admin's by-hand issue endpoint, which the contract
    requires run 'the same model/ctx validation as the public intake'.

    The context caps it returns are catalogue_ctx()'s best case -- what the
    fleet's most capable box will actually serve each chosen model with, not
    a flat number off the catalogue row. For a team that means the SMALLEST
    of the crew's ceilings binds, which is what _validate_public_ctx() does
    with the list."""
    cat = public_catalogue()["by_public"]
    kind = str(payload.get("kind") or "").strip()
    if kind == "single":
        mid = str(payload.get("model") or "").strip()
        row = cat.get(mid)
        if not row or not row.get("enabled"):
            if log:
                log_public_event("rejected", email=email, ip=ip, detail="bad_model:" + mid)
            raise PublicError(400, "bad_model", "unknown or disabled model")
        return kind, {"model": mid}, [int(catalogue_ctx(row)["best"])]
    if kind == "team":
        primary = str(payload.get("primary") or "").strip()
        prow = cat.get(primary)
        if not prow or not prow.get("enabled") or not prow.get("allow_primary"):
            if log:
                log_public_event("rejected", email=email, ip=ip, detail="bad_model:" + primary)
            raise PublicError(400, "bad_model", "unknown, disabled, or non-primary model")
        workers_in = payload.get("workers")
        workers = [str(w).strip() for w in workers_in] if isinstance(workers_in, list) else []
        workers = [w for w in workers if w]
        if not (1 <= len(workers) <= 2) or len(set(workers)) != len(workers) \
                or primary in workers:
            if log:
                log_public_event("rejected", email=email, ip=ip, detail="bad_model:workers")
            raise PublicError(400, "bad_model",
                              "pick 1-2 distinct worker models, excluding the primary")
        wrows = []
        for w in workers:
            wrow = cat.get(w)
            if not wrow or not wrow.get("enabled") or not wrow.get("allow_worker"):
                if log:
                    log_public_event("rejected", email=email, ip=ip, detail="bad_model:" + w)
                raise PublicError(400, "bad_model",
                                  "unknown, disabled, or non-worker model: " + w)
            wrows.append(wrow)
        return kind, {"primary": primary, "workers": workers}, \
            [int(catalogue_ctx(prow)["best"])] \
            + [int(catalogue_ctx(r)["best"]) for r in wrows]
    if log:
        log_public_event("rejected", email=email, ip=ip, detail="bad_kind")
    raise PublicError(400, "bad_kind", "kind must be 'single' or 'team'")


def _validate_public_ctx(
    payload: dict, ctx_caps: list[int], email: str = "", ip: str = "", log: bool = True,
) -> int:
    try:
        ctx = int(payload.get("ctx"))
    except (TypeError, ValueError):
        if log:
            log_public_event("rejected", email=email, ip=ip, detail="bad_ctx")
        raise PublicError(400, "bad_ctx", "ctx must be a number of tokens")
    ctx -= ctx % 1024
    ctx_ceiling = min(ctx_caps) if ctx_caps else 1024
    if ctx < 1024 or ctx > ctx_ceiling:
        if log:
            log_public_event("rejected", email=email, ip=ip, detail="bad_ctx")
        raise PublicError(400, "bad_ctx", "ctx must be between 1024 and " + str(ctx_ceiling))
    return ctx


@app.post("/public/api/request")
async def public_request(request: Request) -> JSONResponse:
    require_intake_token(request)
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 -- an unparsable body is a 400, not a 500
        raise PublicError(400, "bad_request", "body must be JSON")
    if not isinstance(payload, dict):
        raise PublicError(400, "bad_request", "body must be a JSON object")

    ip = client_ip(request)
    settings = get_public_settings()

    if not payload.get("accept_terms"):
        log_public_event("rejected", ip=ip, detail="terms")
        raise PublicError(400, "terms", "the terms checkbox is required")

    email = str(payload.get("email") or "").strip().lower()
    if not _EMAIL_RE.match(email):
        log_public_event("rejected", email=email, ip=ip, detail="invalid_email")
        raise PublicError(400, "invalid_email", "that does not look like an email address")

    kind, models_json, ctx_caps = _validate_public_model_selection(payload, email, ip)
    ctx = _validate_public_ctx(payload, ctx_caps, email, ip)

    note = str(payload.get("note") or "")[:500]
    user_agent = str(payload.get("user_agent") or request.headers.get("user-agent") or "")[:300]

    if ip:
        ip_count = db_query(
            "SELECT COUNT(*) c FROM public_keys WHERE ip=? AND created_at>=?",
            (ip, days_ago(1)),
        )[0]["c"]
        if int(ip_count) >= int(settings["ip_requests_per_day"]):
            log_public_event("rate_limited", email=email, ip=ip, detail="ip_limit")
            raise PublicError(429, "ip_limit", "too many requests from this address today")

    elig = public_eligibility(email)
    if elig["verdict"] == "reject":
        log_public_event("rejected", email=email, ip=ip, detail=elig["code"])
        raise PublicError(403, elig["code"], elig["code"].replace("_", " "))

    domain = email.rsplit("@", 1)[-1]
    per_email = int(settings.get("max_keys_per_email", 3) or 0)
    if per_email > 0:
        # Compared on the canonical address: jane+fleet2@ is still jane@, so
        # plus-addressing cannot turn "N keys per person" into N per tag.
        canon = _email_canon(email)
        live = [
            r for r in db_query(
                "SELECT pk.email " + _LIVE_PUBLIC_KEYS_SQL + " AND pk.domain=?",
                (now(), domain),
            ) if _email_canon(str(r["email"])) == canon
        ]
        if len(live) >= per_email:
            log_public_event("rejected", email=email, ip=ip, detail="already_active")
            raise PublicError(
                409, "already_active",
                "this address already holds the maximum of "
                + str(per_email) + (" live key" if per_email == 1 else " live keys")
                + " -- revoke one, or wait for it to expire")
        pending = [
            r for r in db_query(
                "SELECT email FROM public_keys WHERE domain=? AND status='pending' "
                "AND archived_at IS NULL",
                (domain,),
            ) if _email_canon(str(r["email"])) == canon
        ]
        # Pending counts against the same allowance, because every pending row
        # is a key waiting to be minted: budgeting only the live ones would let
        # three approvals land on an address already holding three keys.
        if len(live) + len(pending) >= per_email:
            log_public_event("rejected", email=email, ip=ip, detail="already_pending")
            raise PublicError(
                409, "already_pending",
                "this address has enough live and pending requests to reach its "
                "limit of " + str(per_email) + " -- one is already awaiting review")

    if elig["verdict"] == "allow":
        dcount = db_query(
            "SELECT COUNT(*) c " + _LIVE_PUBLIC_KEYS_SQL + " AND pk.domain=?",
            (now(), domain),
        )[0]["c"]
        if int(dcount) >= int(settings["max_keys_per_domain"]):
            log_public_event("rate_limited", email=email, ip=ip, detail="domain_cap")
            raise PublicError(429, "domain_cap", "this domain has reached its key limit")

    live_count = db_query(
        "SELECT COUNT(*) c " + _LIVE_PUBLIC_KEYS_SQL, (now(),)
    )[0]["c"]
    if int(live_count) >= int(settings["max_live_keys"]):
        log_public_event("rate_limited", email=email, ip=ip, detail="global_cap")
        raise PublicError(429, "global_cap", "Fleet Pass is at capacity -- please try later")

    row_id = db_exec(
        "INSERT INTO public_keys(created_at,email,domain,company,source,kind,"
        "models,ctx,status,ip,user_agent,note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (now(), email, domain, elig.get("company", "") or "", elig.get("source", "") or "",
         kind, json.dumps(models_json), ctx, "pending", ip, user_agent, note),
    )
    row = db_query("SELECT * FROM public_keys WHERE id=?", (row_id,))[0]

    if elig["verdict"] == "allow":
        raw_key = issue_public_key(row, decided_by="auto")
        row = db_query("SELECT * FROM public_keys WHERE id=?", (row_id,))[0]
        ok, err = await send_key_email(row, raw_key)
        db_exec(
            "UPDATE public_keys SET emailed_at=?, email_error=? WHERE id=?",
            (now() if ok else None, "" if ok else err, row_id),
        )
        if not ok:
            log_public_event("mail_error", email=email, ip=ip, detail=err)
        log_public_event("issued", email=email, ip=ip, detail=kind)
        return JSONResponse({"status": "issued",
                            "message": "Your key is on its way to " + email + "."})

    log_public_event("pending", email=email, ip=ip, detail=kind)
    await send_pending_admin_mail(row)
    return JSONResponse(
        {"status": "pending",
        "message": "Your address was not on our auto-approve lists, so this "
                   "request is queued for a quick manual review -- you will "
                   "get an email either way."},
        status_code=202,
    )


@app.post("/public/api/key-status")
async def public_key_status(request: Request) -> JSONResponse:
    require_intake_token(request)
    ip = client_ip(request)
    if ip:
        hits = db_query(
            "SELECT COUNT(*) c FROM public_events WHERE kind='key_status' "
            "AND ip=? AND ts>=?",
            (ip, (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(
                timespec="seconds")),
        )[0]["c"]
        if int(hits) >= 30:
            raise PublicError(429, "ip_limit", "too many status checks from this address")
    log_public_event("key_status", ip=ip)
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        raise PublicError(400, "bad_request", "body must be JSON")
    raw = str((payload or {}).get("key") or "").strip()
    if not raw:
        raise PublicError(400, "bad_request", "key is required")

    rows = db_query(
        "SELECT pk.*, k.expires_at exp, k.archived_at karchived, "
        "k.max_rpd kmax_rpd, k.max_rph kmax_rph FROM public_keys pk "
        "JOIN api_keys k ON k.id=pk.key_id WHERE k.key_hash=?",
        (hash_key(raw),),
    )
    if not rows:
        return JSONResponse({"error": "unknown key"}, status_code=404)
    row = rows[0]
    exp = row.get("exp")
    cutoff = exp + "T23:59:59+00:00" if exp and len(exp) == 10 else exp
    expired = bool(cutoff) and now() > cutoff
    if row.get("status") == "revoked" or row.get("karchived"):
        status = "revoked"
    elif expired:
        status = "expired"
    else:
        status = "active"

    settings = get_public_settings()
    # The key's OWN ceilings, which are what require_api_key() will actually
    # hold it to -- quoting the settings tab here told a caller 18/hour while
    # the 429 they were getting said 2. Settings are the fallback for a row
    # that somehow carries none, and sync_public_key_limits() keeps the two
    # from drifting in the first place.
    kind_day, kind_hour = (
        ("single_rpd", "single_rph") if row["kind"] == "single"
        else ("team_rpd", "team_rph"))
    limit_day = row.get("kmax_rpd") or settings[kind_day]
    limit_hour = row.get("kmax_rph") or settings[kind_hour]
    day_ago = days_ago(1)
    hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(timespec="seconds")
    used_day = db_query(
        "SELECT COUNT(*) c FROM usage WHERE key_id=? AND ts>=? AND "
        + BUDGET_REQ_SQL,
        (row["key_id"], day_ago),
    )[0]["c"]
    used_hour = db_query(
        "SELECT COUNT(*) c FROM usage WHERE key_id=? AND ts>=? AND "
        + BUDGET_REQ_SQL,
        (row["key_id"], hour_ago),
    )[0]["c"]
    try:
        models = json.loads(row.get("models") or "{}")
    except (TypeError, json.JSONDecodeError):
        models = {}
    return JSONResponse({
        "kind": row["kind"], "models": models, "ctx": row["ctx"],
        "issued_at": row["created_at"], "expires_at": exp, "status": status,
        "used_day": int(used_day), "limit_day": int(limit_day),
        "used_hour": int(used_hour), "limit_hour": int(limit_hour),
        "fallbacks": int(row["fallbacks"]),
    })




# ---- the live demo -------------------------------------------------------
#
# The public presentation (example.org/fleet/tour) ends in a chat box that
# talks to the fleet with no key at all. Same intake token the public site holds
# for the request form, same trusted client-IP header, same catalogue --
# and the same proxy machinery as /v1, with three deliberate differences:
#
#   * one model, chosen on the Public tab, never the caller's choice;
#   * a host policy of its own -- a preferred list tried first when it
#     serves the model, an excluded list never used however well it scores
#     (the big boxes stay free for the keys); a preferred box keeps serving
#     ordinary keyed traffic too, the demo merely joins its queue;
#   * a budget per network address on a rolling hour, counted in
#     public_events like the key-status lookup, because there is no key row
#     to hang a budget on.
#
# The context window is whatever the answering box is holding for that model
# (host_model_ctx), so a box that launched it small answers small, disclosed
# in the reply's meta frame. Metered under a synthetic key name so the
# Telemetry tab shows the demo's traffic beside everyone else's; listed as an
# abortable inference job like any request the hub proxies.

DEMO_ENDPOINT = "/public/api/demo"
DEMO_KEY_NAME = "fleet-pass-demo"
DEMO_MAX_MESSAGES = 12
DEMO_MAX_MESSAGE_CHARS = 4000
# The last box gets a bounded wait too: a demo visitor is not an SDK that will
# sit through a ten-minute cold load, and an unanswerable request should end
# with a sentence, not a spinner.
DEMO_LAST_DEADLINE = 240.0


def demo_ip_used(ip: str) -> int:
    if not ip:
        return 0
    hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(timespec="seconds")
    return int(db_query(
        "SELECT COUNT(*) c FROM public_events WHERE kind='demo' AND ip=? AND ts>=?",
        (ip, hour_ago),
    )[0]["c"])


def demo_host_policy(targets: list[tuple[str, str]], settings: dict) -> list[tuple[str, str]]:
    """Apply the demo's host policy to a scored candidate list: drop every
    excluded host, then move the preferred hosts to the front in the order
    they were listed -- but never a busy preferred box ahead of an idle
    non-preferred one. Saturation is checked first and preference only
    breaks a tie among boxes that are equally free (or equally saturated);
    a stable sort, so within each band the fleet scorer's own order
    (resident, idle, fast) still decides."""
    exclude = set(settings.get("demo_exclude_hosts") or [])
    prefer = list(settings.get("demo_prefer_hosts") or [])
    kept = [t for t in targets if (t[0] or HOST_NAME).lower() not in exclude]

    def band(t: tuple[str, str]) -> int:
        n = (t[0] or HOST_NAME).lower()
        return prefer.index(n) if n in prefer else len(prefer)

    return sorted(kept, key=lambda t: (_host_saturated(t[0], t[1]), band(t)))


async def demo_candidates(settings: dict, prompt_tokens: int = 0, gen_tokens: int = 256,
                          ) -> list[dict]:
    """Every box the demo may answer from, best first, with what it holds."""
    model = str(settings.get("demo_model") or "")
    if not model:
        return []
    try:
        targets = await resolve_targets(model, role="worker", prompt_tokens=prompt_tokens,
                                        gen_tokens=gen_tokens,
                                        need_ctx=prompt_tokens + gen_tokens) or []
    except Exception:  # noqa: BLE001 -- routing is best-effort
        targets = []
    running = _routes_cache.get("running", {})
    prefer = set(settings.get("demo_prefer_hosts") or [])
    out = []
    for cand, fid in demo_host_policy(targets, settings):
        hname = cand or HOST_NAME
        out.append({
            "cand": cand, "host": hname, "box": public_alias(hname), "fleet_id": fid,
            "ctx": int(host_model_ctx(cand, fid) or 0),
            "resident": fid in running.get(hname, set()),
            "preferred": hname.lower() in prefer,
        })
    return out


def _demo_model_card(settings: dict) -> dict | None:
    row = public_catalogue()["by_public"].get(str(settings.get("demo_model") or ""))
    if not row or not row.get("enabled"):
        return None
    return {"public_id": row["public_id"], "name": row["name"], "vendor": row["vendor"],
            "arch": row["arch"], "params_b": row["params_b"], "active_b": row["active_b"]}


async def demo_status_payload(ip: str) -> dict:
    settings = get_public_settings()
    card = _demo_model_card(settings)
    cands = await demo_candidates(settings) if card else []
    limit = int(settings["demo_ip_rph"])
    return {
        "enabled": bool(settings["demo_enabled"]) and card is not None,
        "model": card,
        "limit_per_hour": limit,
        "remaining": max(0, limit - demo_ip_used(ip)),
        "max_tokens": int(settings["demo_max_tokens"]),
        "online": bool(cands),
        # Aliases only, and only boxes the policy would actually use.
        "boxes": [{"box": c["box"], "ctx": c["ctx"], "resident": c["resident"],
                   "preferred": c["preferred"]} for c in cands],
    }


def _demo_messages(payload: Any, settings: dict) -> list[dict]:
    """The conversation as the browser sent it, checked hard: this endpoint
    has no key to revoke, so the shape is the only thing standing between it
    and being a free general-purpose completion API."""
    if not isinstance(payload, dict):
        raise PublicError(400, "bad_request", "body must be a JSON object")
    msgs = payload.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise PublicError(400, "bad_request", "messages must be a non-empty list")
    if len(msgs) > DEMO_MAX_MESSAGES:
        raise PublicError(400, "bad_request",
                          "at most " + str(DEMO_MAX_MESSAGES) + " messages per request")
    out = []
    total = 0
    for m in msgs:
        if not isinstance(m, dict):
            raise PublicError(400, "bad_request", "each message must be an object")
        role = str(m.get("role") or "")
        content = m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            raise PublicError(400, "bad_request",
                              "messages carry a role of user or assistant and string content")
        content = content.strip()
        if not content:
            raise PublicError(400, "bad_request", "an empty message")
        if len(content) > DEMO_MAX_MESSAGE_CHARS:
            raise PublicError(400, "bad_request",
                              "a message is longer than " + str(DEMO_MAX_MESSAGE_CHARS) + " characters")
        total += len(content)
        out.append({"role": role, "content": content})
    if total > int(settings["demo_max_prompt_chars"]):
        raise PublicError(413, "context_limit",
                          "the conversation is longer than the demo allows -- start a fresh one")
    if out[-1]["role"] != "user":
        raise PublicError(400, "bad_request", "the last message must be from the user")
    return out


class _ThinkStripper:
    """Drops a reasoning preamble a small model may still emit inline
    (<think>...</think>) even when told not to, across chunk boundaries. The
    engines are asked to keep reasoning out of `content` first; this is the
    belt to that brace, because a visitor should never watch a model talk to
    itself."""
    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self.inside = False
        self.carry = ""

    def feed(self, text: str) -> str:
        buf = self.carry + text
        self.carry = ""
        out = ""
        while buf:
            if self.inside:
                j = buf.find(self.CLOSE)
                if j < 0:
                    # Keep a tail that could be the start of the closing tag.
                    tail = buf[-(len(self.CLOSE) - 1):]
                    k = tail.rfind("<")
                    self.carry = tail[k:] if k >= 0 else ""
                    return out
                buf = buf[j + len(self.CLOSE):].lstrip("\n")
                self.inside = False
                continue
            i = buf.find(self.OPEN)
            if i < 0:
                tail = buf[-(len(self.OPEN) - 1):]
                k = tail.rfind("<")
                if k >= 0:
                    keep = tail[k:]
                    out += buf[:len(buf) - len(keep)]
                    self.carry = keep
                else:
                    out += buf
                return out
            out += buf[:i]
            buf = buf[i + len(self.OPEN):]
            self.inside = True
        return out

    def flush(self) -> str:
        c, self.carry = self.carry, ""
        return "" if self.inside else c


def _demo_upstream_body(msgs: list[dict], fleet_id: str, engine: str, settings: dict) -> dict:
    system = str(settings.get("demo_system_prompt") or "")
    body: dict[str, Any] = {
        "model": fleet_id, "stream": True,
        "messages": ([{"role": "system", "content": system}] if system else []) + msgs,
        "max_tokens": int(settings["demo_max_tokens"]),
        "temperature": 0.6,
        "stream_options": {"include_usage": True},
    }
    # Keep the reasoning phase out of the answer, each engine its own way:
    # llama-server honours the Qwen template's enable_thinking switch, Ollama
    # its own top-level flag. The other engine ignores the key it does not know.
    if engine == "ollama":
        body["think"] = False
    else:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def _sse(obj: dict) -> bytes:
    return ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode()


async def demo_stream(request: Request, msgs: list[dict], settings: dict, ip: str,
                      ) -> "AsyncIterator[bytes]":
    """Answer one demo conversation as a stream of typed SSE frames:
    meta (which box, which window) -> delta... -> done, or error.

    Failover walks the policy-ordered candidate list exactly as /v1 does --
    a connect failure, no first byte within the deadline, or a 5xx moves to
    the next box and cools the one that failed. A box that has started
    answering is never swapped: a broken stream ends with an error frame,
    and the tokens already produced are metered as a 502."""
    started = time.time()
    public_id = str(settings.get("demo_model") or "")
    card = _demo_model_card(settings) or {"name": public_id}
    prompt_est = estimate_prompt_tokens({"messages": msgs}) + 80
    gen_est = int(settings["demo_max_tokens"])
    cands = await demo_candidates(settings, prompt_tokens=prompt_est, gen_tokens=gen_est)
    limit = int(settings["demo_ip_rph"])
    remaining = max(0, limit - demo_ip_used(ip))
    key = {"id": None, "name": DEMO_KEY_NAME}
    if not cands:
        record_usage(key, public_id, DEMO_ENDPOINT, True, 503, None, None,
                     int((time.time() - started) * 1000))
        yield _sse({"type": "error", "code": "demo_offline",
                    "message": "no suitable machine is awake for the demo model",
                    "remaining": remaining})
        return
    job = _job_open("inference", what=public_id,
                    detail=DEMO_ENDPOINT + " · stream · demo", origin="public demo",
                    stop=asyncio.Event())
    watcher = _watch_disconnect(request, job)
    resp: Any = None
    peer_client: httpx.AsyncClient | None = None
    untrack: Any = None
    served: dict | None = None
    status_out = 502
    usage: dict | None = None
    ttft: int | None = None
    tried: list[str] = []
    only_ctx = True
    try:
        for i, c in enumerate(cands):
            more = i < len(cands) - 1
            cand, hname, fid = c["cand"], c["host"], c["fleet_id"]
            engine = _routes_cache.get("engine", {}).get(hname, "")
            body = _demo_upstream_body(msgs, fid, engine, settings)
            if c["ctx"] > 0:
                try:
                    body = apply_ctx_limit(body, int(c["ctx"]))
                except HTTPException:
                    # Too long for this box's window; a bigger one may fit it.
                    tried.append(c["box"] + " (context)")
                    continue
                # apply_ctx_limit() raises a completion to the 512-token floor
                # a thinking model needs; the demo has thinking off and the
                # owner's ceiling is the ceiling.
                body["max_tokens"] = min(int(body.get("max_tokens") or gen_est), gen_est)
            only_ctx = False
            payload = json.dumps(body, ensure_ascii=False).encode()
            hdrs = {"content-type": "application/json", "accept": "text/event-stream"}
            send_client = client
            peer_client = None
            if cand:
                pk = await peer_inference_key(cand)
                p = next((x for x in load_peers() if x["name"] == cand), None)
                if not (pk and p):
                    tried.append(c["box"] + " (no credentials)")
                    continue
                peer_client = httpx.AsyncClient(
                    base_url=p["url"].rstrip("/"),
                    timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None))
                send_client = peer_client
                hdrs["authorization"] = "Bearer " + pk
                hdrs["x-fleet-origin"] = HOST_NAME + "/" + str(job["id"])
            if send_client is None:
                tried.append(c["box"] + " (no upstream)")
                continue
            req = send_client.build_request("POST", "/v1/chat/completions", headers=hdrs, content=payload)
            job["host"] = hname
            if untrack:
                untrack()
            untrack = _track(hname)
            deadline = _ttfb_deadline(prompt_est, bool(c["resident"])) if more else DEMO_LAST_DEADLINE
            try:
                work: Any = asyncio.wait_for(send_client.send(req, stream=True), timeout=deadline)
                resp, cut = await _race_abort(work, job)
            except asyncio.TimeoutError:
                tried.append(c["box"] + " (no answer in %ds)" % int(deadline))
                _mark_host_down(hname, COOLDOWN_STALL,
                                "demo: no first byte for " + fid + " within " + str(int(deadline)) + "s")
                await _quiet_close(peer_client)
                peer_client = None
                continue
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                tried.append(c["box"])
                _mark_host_down(hname, COOLDOWN_CONNECT, type(exc).__name__)
                _routes_cache["t"] = 0.0
                await _quiet_close(peer_client)
                peer_client = None
                continue
            except httpx.HTTPError as exc:
                tried.append(c["box"] + " (" + type(exc).__name__ + ")")
                _mark_host_down(hname, COOLDOWN_MIDSTREAM, type(exc).__name__)
                await _quiet_close(peer_client)
                peer_client = None
                continue
            if cut:
                status_out = 499
                return
            if resp.status_code >= 400:
                st = int(resp.status_code)
                raw_body, body_txt = b"", ""
                try:
                    raw_body = await resp.aread()
                    body_txt = raw_body.decode("utf-8", "replace")[:300]
                except Exception:  # noqa: BLE001
                    pass
                await _quiet_close(resp, peer_client)
                resp, peer_client = None, None
                tried.append(c["box"] + " (HTTP " + str(st) + ")")
                kind = _classify_upstream_failure(st, "chat/completions", raw_body)
                cooldown = _busy_cooldown_seconds() if kind == "busy" else COOLDOWN_UPSTREAM_5XX
                if kind == "model_missing":
                    _routes_cache["t"] = 0.0
                if more and _upstream_failed(st, "chat/completions", raw_body):
                    _mark_host_down(hname, cooldown, "demo: HTTP " + str(st) + " for " + fid)
                    continue
                if st >= 500 or kind == "busy":
                    _mark_host_down(hname, cooldown, "demo: HTTP " + str(st) + " (no other host)")
                status_out = 502
                # The engine's own words stay in the journal: an error body
                # from llama-server or Ollama names model files and paths,
                # and this frame reaches an anonymous browser.
                log.warning("demo: %s (%s) answered HTTP %s for %s: %s",
                            hname, c["box"], st, fid, body_txt)
                yield _sse({"type": "error", "code": "upstream",
                            "message": c["box"] + " answered with an error (HTTP " + str(st) + ")",
                            "remaining": remaining})
                return
            served = c
            status_out = 200
            _mark_host_ok(hname)
            yield _sse({"type": "meta", "box": c["box"], "model": card.get("name") or public_id,
                        "public_id": public_id, "ctx": int(c["ctx"]), "resident": bool(c["resident"])})
            strip = _ThinkStripper()
            buf = b""
            chunks = resp.aiter_bytes()
            completion_chars = 0
            while True:
                try:
                    chunk, cut = await _race_abort(chunks.__anext__(), job)
                except StopAsyncIteration:
                    break
                except Exception as exc:  # noqa: BLE001 -- the box died mid-answer
                    status_out = 502
                    _mark_host_down(hname, COOLDOWN_MIDSTREAM, "demo stream broke: " + type(exc).__name__)
                    yield _sse({"type": "error", "code": "upstream",
                                "message": "the machine stopped mid-reply", "remaining": remaining})
                    return
                if cut:
                    status_out = 499
                    return
                if ttft is None:
                    ttft = int((time.time() - started) * 1000)
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == b"[DONE]":
                        continue
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    if obj.get("error"):
                        status_out = 502
                        err = obj["error"] if isinstance(obj["error"], dict) else {"message": str(obj["error"])}
                        # Same rule as the HTTP-error branch: the engine's
                        # text is logged, never shown.
                        log.warning("demo: %s (%s) reported an error mid-reply for %s: %s",
                                    hname, c["box"], fid, str(err.get("message", ""))[:300])
                        yield _sse({"type": "error", "code": "upstream",
                                    "message": c["box"] + " reported an error mid-reply",
                                    "remaining": remaining})
                        return
                    for ch in obj.get("choices") or []:
                        delta = (ch or {}).get("delta") or {}
                        text = delta.get("content")
                        if isinstance(text, str) and text:
                            text = strip.feed(text)
                            if text:
                                completion_chars += len(text)
                                yield _sse({"type": "delta", "text": text})
            tail = strip.flush()
            if tail:
                completion_chars += len(tail)
                yield _sse({"type": "delta", "text": tail})
            latency = int((time.time() - started) * 1000)
            comp = int((usage or {}).get("completion_tokens") or 0) or int(completion_chars / 3.5)
            gen_ms = max(1, latency - (ttft or 0))
            yield _sse({"type": "done", "usage": usage or {}, "latency_ms": latency,
                        "ttft_ms": ttft, "tok_s": round(comp / (gen_ms / 1000.0), 1) if comp else None,
                        "remaining": remaining})
            return
        # Every candidate declined.
        if only_ctx and tried:
            status_out = 413
            yield _sse({"type": "error", "code": "context_limit",
                        "message": "the conversation is longer than the awake machines can hold -- start a fresh one",
                        "remaining": remaining})
        else:
            status_out = 502
            yield _sse({"type": "error", "code": "demo_offline",
                        "message": "no machine could take the request right now (tried " + ", ".join(tried) + ")",
                        "remaining": remaining})
    finally:
        watcher.cancel()
        if untrack:
            untrack()
        await _quiet_close(resp, peer_client)
        _job_close(job)
        record_usage(key, public_id, DEMO_ENDPOINT, True,
                     499 if job["aborted"] else status_out, usage, ttft,
                     int((time.time() - started) * 1000), host=(served or {}).get("host", ""))


@app.get("/public/api/demo")
async def public_demo_status(request: Request) -> dict:
    """What the demo box on the tour page shows before anyone types: whether
    it is on, which model, how many requests this address has left, and which
    boxes (aliases only) could answer right now."""
    return await demo_status_payload(client_ip(request))


@app.post("/public/api/demo")
async def public_demo(request: Request):
    require_intake_token(request)
    settings = get_public_settings()
    ip = client_ip(request)
    if not settings["demo_enabled"] or not _demo_model_card(settings):
        raise PublicError(503, "demo_disabled", "the live demo is switched off")
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        raise PublicError(400, "bad_request", "body must be JSON")
    msgs = _demo_messages(payload, settings)
    limit = int(settings["demo_ip_rph"])
    if ip and demo_ip_used(ip) >= limit:
        log_public_event("rate_limited", ip=ip, detail="demo")
        raise PublicError(429, "ip_limit",
                          "this address has used its " + str(limit) + " demo requests for the hour")
    # Counted before it runs, whatever happens next -- a request the fleet
    # then fails still cost the fleet a try, and the counter is the only thing
    # keeping this an exhibit rather than a free API.
    wants_stream = payload.get("stream", True) is not False
    log_public_event("demo", ip=ip, detail="stream" if wants_stream else "json")
    if not wants_stream:
        # One JSON object, for curl and for tests: the same stream, folded.
        meta: dict = {}
        text: list[str] = []
        done: dict = {}
        err: dict | None = None
        async for frame in demo_stream(request, msgs, settings, ip):
            for line in frame.decode("utf-8").split("\n"):
                if not line.startswith("data: "):
                    continue
                ev = json.loads(line[6:])
                if ev["type"] == "meta":
                    meta = ev
                elif ev["type"] == "delta":
                    text.append(ev["text"])
                elif ev["type"] == "done":
                    done = ev
                elif ev["type"] == "error":
                    err = ev
        if err and not text:
            code = str(err.get("code") or "upstream")
            status = {"context_limit": 413, "demo_offline": 503}.get(code, 502)
            return JSONResponse({"error": err.get("message"), "code": code,
                                 "remaining": err.get("remaining")}, status_code=status)
        return JSONResponse({"box": meta.get("box"), "model": meta.get("model"),
                             "public_id": meta.get("public_id"), "ctx": meta.get("ctx"),
                             "resident": meta.get("resident"), "text": "".join(text),
                             "usage": done.get("usage"), "latency_ms": done.get("latency_ms"),
                             "ttft_ms": done.get("ttft_ms"), "tok_s": done.get("tok_s"),
                             "remaining": done.get("remaining", (err or {}).get("remaining")),
                             "error": (err or {}).get("message")})
    return StreamingResponse(
        demo_stream(request, msgs, settings, ip),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def demo_admin_report(settings: dict, cands: list[dict]) -> dict:
    """The Public tab's view of the demo: real host names are fine here."""
    day_ago = days_ago(1)
    hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(timespec="seconds")
    n_day = db_query("SELECT COUNT(*) c FROM usage WHERE endpoint=? AND ts>=?",
                     (DEMO_ENDPOINT, day_ago))[0]["c"]
    n_hour = db_query("SELECT COUNT(*) c FROM usage WHERE endpoint=? AND ts>=?",
                      (DEMO_ENDPOINT, hour_ago))[0]["c"]
    by_host = db_query(
        "SELECT COALESCE(host,'') host, COUNT(*) n, AVG(latency_ms) lat FROM usage "
        "WHERE endpoint=? AND ts>=? AND status<400 GROUP BY host ORDER BY n DESC",
        (DEMO_ENDPOINT, day_ago))
    excluded = set(settings.get("demo_exclude_hosts") or [])
    model = str(settings.get("demo_model") or "")
    # Which excluded boxes would otherwise have served it -- so "why not the
    # big box" has an answer on the tab.
    cat = public_catalogue()["by_public"].get(model) or {}
    cands_all = _routes_cache.get("cands", {})
    shut_out = sorted({(c or HOST_NAME) for f in _row_fleet_ids(cat) for c in cands_all.get(f, [])
                       if (c or HOST_NAME).lower() in excluded})
    return {
        "enabled": bool(settings["demo_enabled"]), "model": model,
        "model_known": _demo_model_card(settings) is not None,
        "candidates": cands, "excluded_serving": shut_out,
        "requests_day": int(n_day), "requests_hour": int(n_hour),
        "by_host": [{"host": r["host"], "box": public_alias(r["host"]) if r["host"] else "",
                     "n": int(r["n"]), "latency_ms": int(r["lat"] or 0)} for r in by_host],
    }


# ---- admin routes -------------------------------------------------------


async def _fleet_live_for_row(row: dict) -> dict:
    """Per fleet id: which hosts route to it, which of those have it
    resident, and the loaded context on each -- from the routing cache
    (live hosts) and the snapshot table (offline ones), same sources the
    rest of the dashboard already reads."""
    await model_routes()
    cands = _routes_cache.get("cands", {})
    running = _routes_cache.get("running", {})
    out: dict[str, dict] = {}
    for fid in _row_fleet_ids(row):
        hosts = [c or HOST_NAME for c in cands.get(fid, [])]
        resident = [h for h in hosts if fid in running.get(h, set())]
        n_ctx: dict[str, Any] = {}
        for h in resident:
            snap = snap_load(h)
            if not snap:
                continue
            for m in (snap["data"].get("models_running") or []):
                if isinstance(m, dict) and str(m.get("id")) == fid and m.get("n_ctx"):
                    n_ctx[h] = m["n_ctx"]
        out[fid] = {"hosts": hosts, "resident": resident, "n_ctx": n_ctx}
    return out


@app.get("/admin/api/public/models")
async def admin_public_models(admin: dict = Depends(require_admin)) -> dict:
    settings = get_public_settings()
    # In the order the public page will present them, disabled rows included:
    # the tab is where that order is edited, so it has to show the result.
    rows = order_public_models(
        db_query("SELECT * FROM public_models"), settings)
    featured = featured_public_id(settings)
    # fleet_ids is stored as JSON text; the tab edits it as a list (and PUT
    # insists on one), so hand it back as a list rather than a string.
    items = [{**row, "fleet_ids": _row_fleet_ids(row),
              "family": public_family(row),
              "featured": row["public_id"] == featured,
              "fleet_live": await _fleet_live_for_row(row)} for row in rows]
    return {"items": items, "family_order": settings["model_family_order"],
            "featured": featured, "preload": preload_report()}


@app.put("/admin/api/public/models/{public_id}")
async def admin_public_model_put(
    public_id: str, request: Request, admin: dict = Depends(require_admin)
) -> dict:
    p = await request.json()
    # resolve_targets() answers the catalogue before a fleet role, so a row
    # named like one -- or claiming one as a fleet id -- would quietly take
    # over live routing for that word.
    if public_id in FLEET_ROLES:
        raise HTTPException(
            400, f"{public_id!r} is a fleet role (FLEET_ROLES in the gateway); "
                 "a catalogue row by that name would shadow the policy")
    before = db_query("SELECT * FROM public_models WHERE public_id=?", (public_id,))
    row_before = before[0] if before else {}
    fleet_ids = p.get("fleet_ids")
    if not isinstance(fleet_ids, list) or not fleet_ids \
            or not all(isinstance(f, str) and f.strip() for f in fleet_ids):
        raise HTTPException(400, "fleet_ids must be a non-empty list of strings")
    if any(f.strip() in FLEET_ROLES for f in fleet_ids):
        raise HTTPException(400, "fleet_ids may not name a fleet role")
    arch = str(p.get("arch", "dense"))
    if arch not in ("moe", "dense"):
        raise HTTPException(400, "arch must be 'moe' or 'dense'")
    try:
        ctx_max = int(p.get("ctx_max", 16384) or 16384)
        ctx_default = int(p.get("ctx_default", 8192) or 8192)
    except (TypeError, ValueError):
        raise HTTPException(400, "ctx_max/ctx_default must be integers")
    if ctx_max < 1024 or ctx_max % 1024:
        raise HTTPException(400, "ctx_max must be at least 1024 and a multiple of 1024")
    if ctx_default > ctx_max:
        raise HTTPException(400, "ctx_default cannot exceed ctx_max")
    enabled = bool(p.get("enabled", True))
    allow_primary = bool(p.get("allow_primary", False))
    allow_worker = bool(p.get("allow_worker", False))
    if enabled and not allow_primary and not allow_worker:
        raise HTTPException(
            400, "set allow_primary or allow_worker, or leave the model disabled"
        )
    _public_model_upsert({
        "public_id": public_id, "name": p.get("name", ""), "vendor": p.get("vendor", ""),
        # Left out of the body entirely, the row keeps the family it has --
        # the tab PUTs whole rows, but a script that does not know about the
        # column should not silently blank it back to a guess.
        "family": p.get("family", row_before.get("family", "")),
        "description": p.get("description", ""), "fleet_ids": fleet_ids, "arch": arch,
        "params_b": p.get("params_b", 0), "active_b": p.get("active_b", 0),
        "allow_primary": allow_primary, "allow_worker": allow_worker,
        "ctx_max": ctx_max, "ctx_default": ctx_default, "enabled": enabled,
        "sort": p.get("sort", 0),
    })
    public_catalogue(force=True)
    row = db_query("SELECT * FROM public_models WHERE public_id=?", (public_id,))[0]
    return {**row, "fleet_ids": _row_fleet_ids(row), "family": public_family(row)}


@app.delete("/admin/api/public/models/{public_id}")
async def admin_public_model_del(
    public_id: str, admin: dict = Depends(require_admin)
) -> dict:
    in_use = db_query(
        "SELECT COUNT(*) c FROM public_keys WHERE status='issued' "
        "AND archived_at IS NULL AND (models LIKE ? OR models LIKE ? OR models LIKE ?)",
        ('%"model": "' + public_id + '"%', '%"primary": "' + public_id + '"%',
         '%"' + public_id + '"%'),
    )[0]["c"]
    if int(in_use):
        raise HTTPException(409, "still used by a live Fleet Pass key -- revoke it first")
    if not db_update("DELETE FROM public_models WHERE public_id=?", (public_id,)):
        raise HTTPException(404, "no such model")
    public_catalogue(force=True)
    return {"deleted": public_id}


@app.post("/admin/api/public/models/reseed")
async def admin_public_models_reseed(admin: dict = Depends(require_admin)) -> dict:
    added = seed_public_models(missing_only=True)
    backfill_public_families()
    public_catalogue(force=True)
    return {"added": added}


@app.get("/admin/api/public/demo")
async def admin_public_demo(admin: dict = Depends(require_admin)) -> dict:
    settings = get_public_settings()
    return demo_admin_report(settings, await demo_candidates(settings))


@app.get("/admin/api/public/preload")
async def admin_public_preload(admin: dict = Depends(require_admin)) -> dict:
    return preload_report()


@app.post("/admin/api/public/preload")
async def admin_public_preload_run(admin: dict = Depends(require_admin)) -> dict:
    """Run a convergence pass now rather than waiting out the tick. `force`
    runs it on a box the loop would otherwise leave alone (one with no peers)
    -- useful for proving the wiring on a single machine, and the reason the
    loop's own idle rules still apply here: this shortens the wait, it does
    not license evicting a model somebody is using."""
    touched = await preload_pass(force=True)
    return {"touched": [{k: v for k, v in t.items() if k != "cand"} for t in touched],
            **preload_report()}


@app.get("/admin/api/public/domains")
async def admin_public_domains(
    source: str = "", q: str = "", mode: str = "", limit: int = 50, offset: int = 0,
    admin: dict = Depends(require_admin),
) -> dict:
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    where, args = ["1=1"], []
    if source:
        where.append("source=?")
        args.append(source)
    if mode:
        where.append("mode=?")
        args.append(mode)
    if q:
        where.append("(domain LIKE ? OR company LIKE ?)")
        args += ["%" + q + "%", "%" + q + "%"]
    clause = " AND ".join(where)
    total = db_query("SELECT COUNT(*) n FROM public_domains WHERE " + clause, args)[0]["n"]
    items = db_query(
        "SELECT * FROM public_domains WHERE " + clause
        + " ORDER BY source, rank IS NULL, rank, domain LIMIT ? OFFSET ?",
        args + [limit, offset],
    )
    return {"items": items, "total": int(total), "limit": limit, "offset": offset}


@app.get("/admin/api/public/domains/summary")
async def admin_public_domains_summary(admin: dict = Depends(require_admin)) -> dict:
    sources: dict[str, dict] = {}
    for r in db_query(
        "SELECT source, COUNT(*) c, SUM(enabled) e FROM public_domains "
        "WHERE mode='allow' GROUP BY source"
    ):
        sources[r["source"]] = {"count": int(r["c"]), "enabled": int(r["e"] or 0)}
    blocked = db_query(
        "SELECT COUNT(*) c FROM public_domains WHERE mode='block'"
    )[0]["c"]
    return {"sources": sources, "blocked": int(blocked)}


def _clean_domain(raw: str) -> str:
    """Lowercase, strip a scheme/www/path -- so pasting a URL into the "add a
    domain" box works as well as pasting the bare domain."""
    d = str(raw or "").strip().lower()
    d = re.sub(r"^[a-z]+://", "", d)
    d = d.split("/", 1)[0]
    if d.startswith("www.") and not d.startswith("www.gov"):
        d = d[4:]
    return d


@app.post("/admin/api/public/domains")
async def admin_public_domains_post(
    request: Request, admin: dict = Depends(require_admin)
) -> dict:
    p = await request.json()
    items = p.get("items")
    if not isinstance(items, list) or not items:
        raise HTTPException(400, "expected {items: [{domain, company, ...}]}")
    added = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        domain = _clean_domain(it.get("domain", ""))
        if not domain:
            continue
        _public_domain_upsert({
            "domain": domain, "company": it.get("company", ""),
            "source": it.get("source") or "custom", "rank": it.get("rank"),
            "mode": it.get("mode") or "allow", "notes": it.get("notes", ""),
        })
        added += 1
    return {"added": added}


@app.patch("/admin/api/public/domains/{domain_id}")
async def admin_public_domain_patch(
    domain_id: int, request: Request, admin: dict = Depends(require_admin)
) -> dict:
    p = await request.json()
    fields: dict[str, Any] = {}
    if "mode" in p:
        if p["mode"] not in ("allow", "block"):
            raise HTTPException(400, "mode must be 'allow' or 'block'")
        fields["mode"] = p["mode"]
    if "enabled" in p:
        fields["enabled"] = int(bool(p["enabled"]))
    if "company" in p:
        fields["company"] = str(p["company"])
    if "notes" in p:
        fields["notes"] = str(p["notes"])
    if not fields:
        raise HTTPException(400, "nothing to update")
    sets = ", ".join(k + "=?" for k in fields)
    if not db_update(
        "UPDATE public_domains SET " + sets + " WHERE id=?",
        (*fields.values(), domain_id),
    ):
        raise HTTPException(404, "no such domain")
    return db_query("SELECT * FROM public_domains WHERE id=?", (domain_id,))[0]


@app.delete("/admin/api/public/domains/{domain_id}")
async def admin_public_domain_del(
    domain_id: int, admin: dict = Depends(require_admin)
) -> dict:
    if not db_update("DELETE FROM public_domains WHERE id=?", (domain_id,)):
        raise HTTPException(404, "no such domain")
    return {"deleted": domain_id}


@app.post("/admin/api/public/domains/import")
async def admin_public_domains_import(
    request: Request, admin: dict = Depends(require_admin)
) -> dict:
    """Text/CSV or {"csv": "...", "source": "custom"} JSON -- columns domain,
    company in either order, a header row optional (detected by the first
    cell not parsing as a domain)."""
    ctype = request.headers.get("content-type", "")
    source = "custom"
    if "application/json" in ctype:
        p = await request.json()
        text = str(p.get("csv", ""))
        source = str(p.get("source") or "custom")
    else:
        text = (await request.body()).decode("utf-8", "replace")

    added = 0
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        cells = [c.strip() for c in re.split(r"[,\t]", line)]
        if len(cells) == 1:
            cells.append("")
        a, b = cells[0], cells[1]
        # Whichever cell has a dot in it is the domain; the header row ("domain,
        # company") has neither, and is skipped by ending up with an empty clean.
        domain = _clean_domain(a if "." in a else b)
        company = b if "." in a else a
        if i == 0 and not domain and (a.lower() in ("domain", "host") or
                                      b.lower() in ("company", "name")):
            continue
        if not domain or "." not in domain:
            continue
        _public_domain_upsert({
            "domain": domain, "company": company, "source": source, "mode": "allow",
        })
        added += 1
    return {"added": added}


@app.post("/admin/api/public/domains/reseed")
async def admin_public_domains_reseed(admin: dict = Depends(require_admin)) -> dict:
    return {"added": seed_public_domains(missing_only=True)}


@app.post("/admin/api/public/domains/check")
async def admin_public_domains_check(
    request: Request, admin: dict = Depends(require_admin)
) -> dict:
    p = await request.json()
    return public_eligibility(str(p.get("email", "")))


def _public_key_view(row: dict) -> dict:
    """Shape one issued-key row for the Public tab.

    `expires_at`, `limit_day` and `limit_hour` come from the api_keys row, not
    from public_keys -- what a key is actually held to lives there, and
    reading them off the wrong table is why this view once showed no expiry at
    all and could never mark a key expired. Since sync_public_key_limits()
    these normally equal the settings tab; when they do not, the row is still
    the honest answer, because the row is what turns a caller away."""
    exp = row.get("expires_at")
    cutoff = exp + "T23:59:59+00:00" if exp and len(exp) == 10 else exp
    status = row["status"]
    if status == "issued" and cutoff and now() > cutoff:
        status = "expired"
    return {**row, "status": status,
            "limit_day": row.get("max_rpd"), "limit_hour": row.get("max_rph")}


@app.get("/admin/api/public/keys")
async def admin_public_keys(
    status: str = "", q: str = "", limit: int = 25, offset: int = 0,
    archived: bool = False, source: str = "", admin: dict = Depends(require_admin),
) -> dict:
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    where = ["pk.archived_at IS NOT NULL" if archived else "pk.archived_at IS NULL"]
    args: list[Any] = []
    if status:
        where.append("pk.status=?")
        args.append(status)
    # `auto` is every connected service at once (source `auto:<service>`);
    # any other value is an exact source, e.g. `manual` or `auto:demo-app`.
    if source == "auto":
        where.append("pk.source LIKE 'auto:%'")
    elif source:
        where.append("pk.source=?")
        args.append(source)
    if q:
        where.append("(pk.email LIKE ? OR pk.company LIKE ? OR pk.domain LIKE ? OR pk.note LIKE ?)")
        args += ["%" + q + "%", "%" + q + "%", "%" + q + "%", "%" + q + "%"]
    clause = " AND ".join(where)
    total = db_query(
        "SELECT COUNT(*) n FROM public_keys pk WHERE " + clause, args
    )[0]["n"]
    rows = db_query(
        "SELECT pk.*, k.prefix, k.disabled, k.expires_at, k.last_used_at, "
        "k.max_rpd, k.max_rph "
        "FROM public_keys pk LEFT JOIN api_keys k ON k.id=pk.key_id "
        "WHERE " + clause + " ORDER BY pk.id DESC LIMIT ? OFFSET ?",
        args + [limit, offset],
    )
    day_ago, hour_ago = days_ago(1), (
        datetime.now(timezone.utc) - timedelta(hours=1)
    ).isoformat(timespec="seconds")
    for r in rows:
        kid = r.get("key_id")
        used_day = used_hour = tokens_total = 0
        if kid:
            used_day = db_query(
                "SELECT COUNT(*) c FROM usage WHERE key_id=? AND ts>=? AND "
                + BUDGET_REQ_SQL, (kid, day_ago),
            )[0]["c"]
            used_hour = db_query(
                "SELECT COUNT(*) c FROM usage WHERE key_id=? AND ts>=? AND "
                + BUDGET_REQ_SQL, (kid, hour_ago),
            )[0]["c"]
            tokens_total = db_query(
                "SELECT COALESCE(SUM(total_tokens),0) t FROM usage WHERE key_id=?",
                (kid,),
            )[0]["t"]
        r["used_day"], r["used_hour"], r["tokens_total"] = (
            int(used_day), int(used_hour), int(tokens_total)
        )
    items = [_public_key_view(r) for r in rows]
    return {"items": items, "total": int(total), "limit": limit, "offset": offset}


def _admin_email(admin: dict) -> str:
    return str(admin.get("email") or "admin")


@app.post("/admin/api/public/keys")
async def admin_public_key_issue(
    request: Request, admin: dict = Depends(require_admin)
) -> dict:
    """Issue a key BY HAND -- the owner vouching for someone who wrote from
    a personal address, or a quick test to their own inbox. Same model/ctx
    validation as the public intake (contract 1.10), but none of its
    eligibility, cap, or free-mail checks: an admin token already proves
    this request is trusted."""
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(400, "body must be an object")
    email = str(payload.get("email") or "").strip().lower()
    if not _EMAIL_RE.match(email):
        raise HTTPException(400, "that does not look like an email address")
    try:
        kind, models_json, ctx_caps = _validate_public_model_selection(payload, log=False)
        ctx = _validate_public_ctx(payload, ctx_caps, log=False)
    except PublicError as exc:
        raise HTTPException(exc.status, exc.message)
    note = str(payload.get("note") or "")[:500]
    domain = email.rsplit("@", 1)[-1]
    admin_email = _admin_email(admin)
    row_id = db_exec(
        "INSERT INTO public_keys(created_at,email,domain,company,source,kind,"
        "models,ctx,status,ip,user_agent,note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (now(), email, domain, "", "manual", kind, json.dumps(models_json), ctx,
         "pending", "", "", note),
    )
    row = db_query("SELECT * FROM public_keys WHERE id=?", (row_id,))[0]
    raw_key = issue_public_key(row, decided_by=admin_email)
    row = db_query("SELECT * FROM public_keys WHERE id=?", (row_id,))[0]
    ok, err = await send_key_email(row, raw_key)
    db_exec(
        "UPDATE public_keys SET emailed_at=?, email_error=? WHERE id=?",
        (now() if ok else None, "" if ok else err, row_id),
    )
    if not ok:
        log_public_event("mail_error", email=email, detail=err)
    log_public_event("issued", email=email, detail="manual:" + admin_email)
    return db_query("SELECT * FROM public_keys WHERE id=?", (row_id,))[0]


_SERVICE_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")


def _auto_issued_today() -> int:
    return int(db_query(
        "SELECT COUNT(*) c FROM public_keys WHERE source LIKE 'auto:%' AND created_at>=?",
        (days_ago(1),),
    )[0]["c"])


@app.get("/admin/api/public/auto")
async def admin_public_auto_status(admin: dict = Depends(require_admin)) -> dict:
    """What auto-issue would do right now, and what it has done: the
    dashboard card's status line, and the one thing a connected service
    reads back (so its own admin page can SHOW these settings without
    holding a copy of them)."""
    s = get_public_settings()
    cat = public_catalogue()["by_public"].get(s["auto_issue_model"]) or {}
    live = db_query(
        "SELECT COUNT(*) c FROM public_keys pk JOIN api_keys k ON k.id=pk.key_id "
        "WHERE pk.source LIKE 'auto:%' AND pk.status='issued' AND pk.archived_at IS NULL "
        "AND k.archived_at IS NULL AND (k.expires_at IS NULL OR k.expires_at > ?)",
        (now(),),
    )[0]["c"]
    by_service = db_query(
        "SELECT source, COUNT(*) n FROM public_keys WHERE source LIKE 'auto:%' "
        "GROUP BY source ORDER BY n DESC",
    )
    return {
        "enabled": bool(s["auto_issue_enabled"]),
        "model": s["auto_issue_model"],
        "model_name": str(cat.get("name") or s["auto_issue_model"]),
        "model_enabled": bool(cat.get("enabled")),
        "ctx": int(s["auto_issue_ctx"]),
        "daily_cap": int(s["auto_issue_daily_cap"]),
        "key_days": int(s["key_days"]),
        "limit_day": int(s["single_rpd"]), "limit_hour": int(s["single_rph"]),
        "issued_today": _auto_issued_today(),
        "live": int(live),
        "by_service": [{"service": str(r["source"])[5:], "keys": int(r["n"])} for r in by_service],
        "base_url": str(s.get("public_base_url") or PUBLIC_API_URL or "").rstrip("/"),
        "setup_text": s["auto_issue_setup_text"],
        "endpoint": "/admin/api/public/keys/auto",
    }


@app.post("/admin/api/public/keys/auto")
async def admin_public_key_auto_issue(
    request: Request, admin: dict = Depends(require_admin)
) -> dict:
    """Mint a Fleet Pass key for a CONNECTED SERVICE and hand the raw key
    back in the response -- the one place this gateway returns a key over
    HTTP, because the service is going to write it into a document itself
    (a cover letter, a message) rather than read an inbox.

    Same shape as an approval: the model, lifetime and budgets from the
    Public tab are stamped on the row, sync_public_key_limits() reaches
    these keys like any other, and they sit in the issued-keys table under
    source `auto:<service>` where they can be revoked or extended. Admin
    token only (contract with the hub's peers: no weaker door than the one
    the dashboard uses). The `ref` is the service's own idea of what the
    key is for -- an application, a message -- and is kept on the row so
    the table can say which one."""
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(400, "body must be an object")
    settings = get_public_settings()
    if not settings.get("auto_issue_enabled", True):
        raise HTTPException(403, "auto-issue is switched off on the Public tab")
    service = str(payload.get("service") or "").strip().lower()
    if not _SERVICE_RE.match(service):
        raise HTTPException(400, "service must be a short lowercase slug (a-z, 0-9, -)")
    ref = str(payload.get("ref") or "").strip()[:120]
    if not ref:
        raise HTTPException(400, "ref is required -- what this key is for (an application, a message)")
    company = str(payload.get("company") or "").strip()[:120]
    note = str(payload.get("note") or "").strip()[:300]
    email = str(payload.get("email") or "").strip().lower()
    if email and not _EMAIL_RE.match(email):
        raise HTTPException(400, "that does not look like an email address")
    model_id = str(payload.get("model") or settings.get("auto_issue_model") or "").strip()
    try:
        kind, models_json, ctx_caps = _validate_public_model_selection(
            {"kind": "single", "model": model_id}, log=False)
    except PublicError as exc:
        raise HTTPException(exc.status, exc.message)
    # The caller does not know the catalogue's ceilings, so the context is
    # clamped rather than refused: the tab's figure (or the caller's smaller
    # ask) rounded down to the 1024 grid, never above what the model serves.
    try:
        want = int(payload.get("ctx") or settings.get("auto_issue_ctx") or 0)
    except (TypeError, ValueError):
        want = int(settings.get("auto_issue_ctx") or 8192)
    ctx = max(1024, min(want, int(min(ctx_caps))))
    ctx -= ctx % 1024
    cap = int(settings.get("auto_issue_daily_cap") or 0)
    if cap and _auto_issued_today() >= cap:
        log_public_event("rate_limited", email=email, detail="auto_issue_cap:" + service)
        raise HTTPException(429, "auto-issue daily cap reached (" + str(cap) + " keys/day)")
    live_count = db_query(
        "SELECT COUNT(*) c FROM public_keys WHERE status='issued' AND archived_at IS NULL"
    )[0]["c"]
    if int(live_count) >= int(settings["max_live_keys"]):
        log_public_event("rate_limited", email=email, detail="global_cap:auto:" + service)
        raise HTTPException(429, "Fleet Pass is at capacity -- raise max live keys or wait")
    ip = client_ip(request)
    row_id = db_exec(
        "INSERT INTO public_keys(created_at,email,domain,company,source,kind,"
        "models,ctx,status,ip,user_agent,note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (now(), email, email.rsplit("@", 1)[-1] if email else service, company,
         "auto:" + service, kind, json.dumps(models_json), ctx, "pending", ip,
         str(request.headers.get("user-agent") or "")[:300],
         (ref + (" -- " + note if note else ""))[:500]),
    )
    row = db_query("SELECT * FROM public_keys WHERE id=?", (row_id,))[0]
    raw_key = issue_public_key(row, decided_by="auto:" + service)
    row = db_query("SELECT * FROM public_keys WHERE id=?", (row_id,))[0]
    if email:
        # An address is optional: when the service knows who the key is for,
        # they get the usual mail too. The response carries the key either way.
        ok, err = await send_key_email(row, raw_key)
        db_exec(
            "UPDATE public_keys SET emailed_at=?, email_error=? WHERE id=?",
            (now() if ok else None, "" if ok else err, row_id),
        )
        if not ok:
            log_public_event("mail_error", email=email, detail=err)
    log_public_event("issued", email=email, ip=ip, detail="auto:" + service + ":" + ref)
    return {
        "status": "issued", "id": int(row_id), "service": service, "ref": ref,
        "company": company, **public_key_bundle(row, raw_key, settings),
        "note": "shown once -- store it now",
    }


@app.post("/admin/api/public/keys/{req_id}/approve")
async def admin_public_key_approve(
    req_id: int, admin: dict = Depends(require_admin)
) -> dict:
    rows = db_query(
        "SELECT * FROM public_keys WHERE id=? AND status='pending' "
        "AND archived_at IS NULL", (req_id,),
    )
    if not rows:
        raise HTTPException(404, "no such pending request")
    row = rows[0]
    raw_key = issue_public_key(row, decided_by=_admin_email(admin))
    row = db_query("SELECT * FROM public_keys WHERE id=?", (req_id,))[0]
    ok, err = await send_key_email(row, raw_key)
    db_exec(
        "UPDATE public_keys SET emailed_at=?, email_error=? WHERE id=?",
        (now() if ok else None, "" if ok else err, req_id),
    )
    if not ok:
        log_public_event("mail_error", email=row["email"], detail=err)
    log_public_event("approved", email=row["email"], detail=_admin_email(admin))
    return db_query("SELECT * FROM public_keys WHERE id=?", (req_id,))[0]


@app.post("/admin/api/public/keys/{req_id}/deny")
async def admin_public_key_deny(
    req_id: int, request: Request, admin: dict = Depends(require_admin)
) -> dict:
    rows = db_query(
        "SELECT * FROM public_keys WHERE id=? AND status='pending' "
        "AND archived_at IS NULL", (req_id,),
    )
    if not rows:
        raise HTTPException(404, "no such pending request")
    row = rows[0]
    reason = ""
    try:
        p = await request.json()
        reason = str((p or {}).get("reason") or "")[:300]
    except Exception:  # noqa: BLE001 -- a body is optional here
        pass
    db_exec(
        "UPDATE public_keys SET status='denied', decided_at=?, decided_by=?, "
        "note=CASE WHEN ?<>'' THEN note || '\n[denied] ' || ? ELSE note END "
        "WHERE id=?",
        (now(), _admin_email(admin), reason, reason, req_id),
    )
    ok, err = await send_deny_mail(row, reason)
    if not ok:
        log_public_event("mail_error", email=row["email"], detail=err)
    log_public_event("denied", email=row["email"], detail=reason or _admin_email(admin))
    return db_query("SELECT * FROM public_keys WHERE id=?", (req_id,))[0]


@app.post("/admin/api/public/keys/{req_id}/revoke")
async def admin_public_key_revoke(
    req_id: int, admin: dict = Depends(require_admin)
) -> dict:
    rows = db_query("SELECT * FROM public_keys WHERE id=?", (req_id,))
    if not rows:
        raise HTTPException(404, "no such request")
    row = rows[0]
    stamp = now()
    if row.get("key_id"):
        db_update(
            "UPDATE api_keys SET archived_at=?, disabled=1 WHERE id=? "
            "AND archived_at IS NULL", (stamp, row["key_id"]),
        )
        db_update(
            "UPDATE agents SET archived_at=? WHERE key_id=? AND archived_at IS NULL",
            (stamp, row["key_id"]),
        )
        db_update(
            "UPDATE teams SET archived_at=? WHERE key_id=? AND archived_at IS NULL",
            (stamp, row["key_id"]),
        )
    db_update(
        "UPDATE public_keys SET status='revoked', decided_at=?, decided_by=? "
        "WHERE id=?",
        (stamp, _admin_email(admin), req_id),
    )
    log_public_event("revoked", email=row["email"], detail=_admin_email(admin))
    return db_query("SELECT * FROM public_keys WHERE id=?", (req_id,))[0]


@app.post("/admin/api/public/keys/{req_id}/extend")
async def admin_public_key_extend(
    req_id: int, admin: dict = Depends(require_admin)
) -> dict:
    rows = db_query(
        "SELECT pk.*, k.expires_at exp FROM public_keys pk "
        "LEFT JOIN api_keys k ON k.id=pk.key_id WHERE pk.id=?", (req_id,),
    )
    if not rows or not rows[0].get("key_id"):
        raise HTTPException(404, "no such issued request")
    row = rows[0]
    settings = get_public_settings()
    days = int(settings.get("key_days", 7))
    base = row.get("exp") or now()
    if len(base) == 10:
        base += "T00:00:00+00:00"
    try:
        new_exp = (datetime.fromisoformat(base) + timedelta(days=days)).isoformat(
            timespec="seconds"
        )
    except ValueError:
        new_exp = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat(
            timespec="seconds"
        )
    db_update("UPDATE api_keys SET expires_at=? WHERE id=?", (new_exp, row["key_id"]))
    log_public_event("extended", email=row["email"], detail=_admin_email(admin))
    return db_query("SELECT * FROM public_keys WHERE id=?", (req_id,))[0]


@app.post("/admin/api/public/keys/{req_id}/resend")
async def admin_public_key_resend(
    req_id: int, admin: dict = Depends(require_admin)
) -> dict:
    """The raw key is hashed at rest and cannot be re-shown, so a resend
    mints a REPLACEMENT key with the same profile and archives the old one."""
    rows = db_query(
        "SELECT * FROM public_keys WHERE id=? AND status='issued' "
        "AND archived_at IS NULL", (req_id,),
    )
    if not rows:
        raise HTTPException(404, "no such issued request")
    row = rows[0]
    # An auto-issued key has no address: a resend would archive the working
    # key and mail the replacement to nobody. The service that minted it is
    # the only party that can carry a key, so it mints a new one instead.
    if not str(row.get("email") or "").strip():
        raise HTTPException(
            400, "this key was minted for a connected service and has no address "
                 "to resend to -- revoke it and let the service mint a new one")
    old_key_id = row.get("key_id")
    raw_key = issue_public_key(row, decided_by=_admin_email(admin))
    if old_key_id:
        stamp = now()
        db_update(
            "UPDATE api_keys SET archived_at=?, disabled=1 WHERE id=? "
            "AND archived_at IS NULL", (stamp, old_key_id),
        )
        db_update(
            "UPDATE agents SET archived_at=? WHERE key_id=? AND archived_at IS NULL",
            (stamp, old_key_id),
        )
        db_update(
            "UPDATE teams SET archived_at=? WHERE key_id=? AND archived_at IS NULL",
            (stamp, old_key_id),
        )
    row = db_query("SELECT * FROM public_keys WHERE id=?", (req_id,))[0]
    ok, err = await send_key_email(row, raw_key)
    db_exec(
        "UPDATE public_keys SET emailed_at=?, email_error=? WHERE id=?",
        (now() if ok else None, "" if ok else err, req_id),
    )
    if not ok:
        log_public_event("mail_error", email=row["email"], detail=err)
    log_public_event("resent", email=row["email"], detail=_admin_email(admin))
    return db_query("SELECT * FROM public_keys WHERE id=?", (req_id,))[0]


@app.post("/admin/api/public/keys/{req_id}/restore")
async def admin_public_key_restore(
    req_id: int, admin: dict = Depends(require_admin)
) -> dict:
    rows = db_query(
        "SELECT * FROM public_keys WHERE id=? AND archived_at IS NOT NULL", (req_id,)
    )
    if not rows:
        raise HTTPException(404, "no such archived request")
    row = rows[0]
    if row.get("key_id") and not db_query(
        "SELECT id FROM api_keys WHERE id=?", (row["key_id"],)
    ):
        raise HTTPException(409, "the underlying api key was purged -- this "
                                 "request cannot be restored")
    db_update("UPDATE public_keys SET archived_at=NULL WHERE id=?", (req_id,))
    if row.get("key_id"):
        db_update(
            "UPDATE api_keys SET archived_at=NULL WHERE id=?", (row["key_id"],)
        )
    return {"restored": req_id}


@app.delete("/admin/api/public/keys/{req_id}/purge")
async def admin_public_key_purge(
    req_id: int, admin: dict = Depends(require_admin)
) -> dict:
    rows = db_query(
        "SELECT id FROM public_keys WHERE id=? AND archived_at IS NOT NULL", (req_id,)
    )
    if not rows:
        raise HTTPException(409, "archive it first -- live requests are not "
                                 "deleted in one step")
    db_exec("DELETE FROM public_keys WHERE id=?", (req_id,))
    return {"deleted": req_id}


@app.get("/admin/api/public/events")
async def admin_public_events(
    kind: str = "", limit: int = 50, offset: int = 0,
    admin: dict = Depends(require_admin),
) -> dict:
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    where, args = "1=1", []
    if kind:
        where = "kind=?"
        args = [kind]
    total = db_query("SELECT COUNT(*) n FROM public_events WHERE " + where, args)[0]["n"]
    items = db_query(
        "SELECT * FROM public_events WHERE " + where + " ORDER BY id DESC LIMIT ? OFFSET ?",
        args + [limit, offset],
    )
    return {"items": items, "total": int(total), "limit": limit, "offset": offset}


@app.get("/admin/api/public/aliases")
async def admin_public_aliases(admin: dict = Depends(require_admin)) -> dict:
    """What the public page calls each machine -- real host -> 'Box N'."""
    rows = db_query("SELECT host, box FROM public_aliases ORDER BY box")
    items = [{"host": r["host"], "box": r["box"], "label": "Box " + str(r["box"])}
             for r in rows]
    items.insert(0, {"host": HOST_NAME, "box": 0, "label": "Hub"})
    return {"items": items}


@app.get("/admin/api/public/settings")
async def admin_public_settings_get(admin: dict = Depends(require_admin)) -> dict:
    out = get_public_settings()
    # "Configured" has to mean a mail can actually go out: a host with a
    # username and no password authenticates nowhere, and reporting that as
    # configured is how a dashboard tells you keys are being delivered while
    # every one of them fails.
    out["smtp_configured"] = bool(SMTP_HOST) and bool(SMTP_PASSWORD or not SMTP_USER)
    out["smtp_from"] = SMTP_FROM or SMTP_USER
    out["intake_configured"] = bool(PUBLIC_INTAKE_TOKEN)
    return out


@app.put("/admin/api/public/settings")
async def admin_public_settings_put(
    request: Request, admin: dict = Depends(require_admin)
) -> dict:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(400, "body must be an object")
    out = set_public_settings(payload)
    # "Configured" has to mean a mail can actually go out: a host with a
    # username and no password authenticates nowhere, and reporting that as
    # configured is how a dashboard tells you keys are being delivered while
    # every one of them fails.
    out["smtp_configured"] = bool(SMTP_HOST) and bool(SMTP_PASSWORD or not SMTP_USER)
    out["smtp_from"] = SMTP_FROM or SMTP_USER
    out["intake_configured"] = bool(PUBLIC_INTAKE_TOKEN)
    return out


@app.post("/admin/api/public/settings/test-mail")
async def admin_public_test_mail(
    request: Request, admin: dict = Depends(require_admin)
) -> dict:
    p = await request.json()
    to = str(p.get("to", "")).strip()
    if not to:
        raise HTTPException(400, "to is required")
    ok, err = await send_mail(
        to, "Fleet Pass test mail",
        "This is a test message from the open-fleet Fleet Pass settings tab.",
    )
    return {"ok": ok, "error": err}


@app.get("/admin/api/public/stats")
async def admin_public_stats(days: int = 30, admin: dict = Depends(require_admin)) -> dict:
    days = max(1, min(days, 365))
    since = days_ago(days)
    live_keys = db_query(
        "SELECT COUNT(*) c FROM public_keys WHERE status='issued' AND archived_at IS NULL"
    )[0]["c"]
    pending = db_query(
        "SELECT COUNT(*) c FROM public_keys WHERE status='pending' AND archived_at IS NULL"
    )[0]["c"]
    issued_total = db_query(
        "SELECT COUNT(*) c FROM public_keys WHERE status IN ('issued','revoked')"
    )[0]["c"]
    by_day = db_query(
        "SELECT substr(u.ts,1,10) day, COUNT(*) requests, "
        "SUM(u.fallback_from IS NOT NULL) fallbacks FROM usage u "
        "JOIN public_keys pk ON pk.key_id=u.key_id "
        "WHERE u.ts>=? AND " + BUDGET_REQ_SQL_U + " "
        "GROUP BY day ORDER BY day",
        (since,),
    )
    top_domains = db_query(
        "SELECT pk.domain, MAX(pk.company) company, COUNT(DISTINCT pk.id) keys, "
        "COUNT(u.id) requests FROM public_keys pk "
        "LEFT JOIN usage u ON u.key_id=pk.key_id AND u.ts>=? "
        "AND " + BUDGET_REQ_SQL_U + " "
        "WHERE pk.status='issued' AND pk.archived_at IS NULL "
        "GROUP BY pk.domain ORDER BY requests DESC LIMIT 15",
        (since,),
    )
    by_model = db_query(
        "SELECT u.model public_id, COUNT(*) requests FROM usage u "
        "JOIN public_keys pk ON pk.key_id=u.key_id "
        "WHERE u.ts>=? AND " + BUDGET_REQ_SQL_U + " "
        "GROUP BY u.model ORDER BY requests DESC",
        (since,),
    )
    return {
        "live_keys": int(live_keys), "pending": int(pending),
        "issued_total": int(issued_total),
        "requests_by_day": [
            {"day": r["day"], "requests": int(r["requests"]),
             "fallbacks": int(r["fallbacks"] or 0)} for r in by_day
        ],
        "top_domains": top_domains, "by_model": by_model,
    }


# ------------------------------- admin API --------------------------------


@app.get("/admin/api/whoami")
async def whoami(admin: dict = Depends(require_admin)) -> dict:
    return admin


_props_cache: dict[str, tuple[float, dict]] = {}


async def upstream_props(mid: str) -> dict:
    """Context size and slot count for one RESIDENT model, read from its own
    llama-server through llama-swap's per-model proxy.

    Only ever called for a model llama-swap already lists as running: asking
    for a model that is not loaded would make llama-swap load it, which is the
    opposite of telemetry. Cached briefly because the dashboard polls status
    every few seconds and this answer only changes on a reload."""
    hit = _props_cache.get(mid)
    if hit and time.time() - hit[0] < 20:
        return hit[1]
    out: dict = {}
    try:
        assert client is not None
        r = await client.get("/upstream/" + quote(mid, safe="") + "/props",
                             timeout=4.0)
        if r.status_code == 200:
            d = r.json()
            gen = d.get("default_generation_settings") or {}
            out = {
                "n_ctx": gen.get("n_ctx") or d.get("n_ctx"),
                "slots": d.get("total_slots"),
            }
    except Exception:  # noqa: BLE001
        out = {}
    _props_cache[mid] = (time.time(), out)
    return out


async def swap_running() -> tuple[bool, list[dict]]:
    """Is the upstream reachable, and which models does it have resident.

    Three shapes, none of them /api/models -- which is what this asked for
    months and which neither backend has ever served. llama-swap lists the
    live llama-server processes on /running and proxies each one at
    /upstream/<id>/; Ollama answers /api/ps. The 404 was swallowed by the
    except below, so the dashboard showed "nothing loaded, no processes" on a
    box whose GPU was pinned at 98% serving a model.

    Everything downstream (the Serving card, the fleet overview, the home
    page) reads this, so it also carries the context window each resident
    model actually got -- the number you want when a run is slower than the
    hardware should be."""
    running: list[dict] = []
    swap_up = False
    assert client is not None

    try:
        r = await client.get("/running", timeout=8.0)
        swap_up = r.status_code < 500
        if r.status_code == 200:
            data = r.json()
            for m in (data.get("running") if isinstance(data, dict) else data) or []:
                if not isinstance(m, dict):
                    continue
                mid = m.get("model") or m.get("id")
                if not mid:
                    continue
                rec = {"id": mid, "state": m.get("state") or "ready"}
                if rec["state"] in ("ready", "starting"):
                    rec.update(await upstream_props(str(mid)))
                running.append(rec)
            return swap_up, running
    except Exception:  # noqa: BLE001
        pass

    # Ollama-backed hosts: /api/ps is the residency probe, and it reports the
    # per-model context Ollama chose (num_ctx) plus what it cost in VRAM.
    try:
        r = await client.get("/api/ps", timeout=8.0)
        if r.status_code == 200:
            swap_up = True
            for m in (r.json() or {}).get("models") or []:
                if not isinstance(m, dict):
                    continue
                mid = m.get("model") or m.get("name")
                if not mid:
                    continue
                running.append({
                    "id": mid,
                    "state": "ready",
                    "n_ctx": m.get("context_length"),
                    "vram": m.get("size_vram"),
                    "expires_at": m.get("expires_at"),
                })
    except Exception:  # noqa: BLE001
        pass
    return swap_up, running


@app.get("/admin/api/status")
async def api_status(admin: dict = Depends(require_admin)) -> dict:
    swap_up, running = await swap_running()
    # Both of these block: psutil samples the CPU over 150ms, nvidia-smi and
    # systemctl are subprocesses, and the engine probe talks HTTP. Run on the
    # event loop they stall every other request this box is serving -- which is
    # how a peer being merely slow turned into a 502 at the hub that proxies to
    # it, and why the dashboard's own polling made it worse the more it polled.
    host, svc = await asyncio.gather(
        asyncio.to_thread(host_status),
        asyncio.to_thread(service_states),
    )
    return {
        "name": HOST_NAME,
        "api_url": PUBLIC_API_URL,
        "host": host,
        "swap_up": swap_up,
        "models_running": running,
        "services": svc,
        # For the Overview card's load picker and warm-standby control:
        # every enabled record with the two flags that matter there, plus
        # the one manual dashboard load in flight (see api_model_load).
        "models_configured": [
            {"id": str(r.get("id")), "preload": bool(r.get("preload")),
             "persistent": bool(r.get("persistent"))}
            for r in load_models() if r.get("enabled", True)
        ],
        "manual_load": dict(_manual_load),
    }


@app.get("/admin/api/models")
async def api_models(admin: dict = Depends(require_admin)) -> dict:
    models = load_models()
    # Kept beside the records rather than inside them: the editor PUTs the
    # record list straight back, and a derived field would be saved as if it
    # were configuration.
    sizing = []
    for rec in models:
        ctx, detail = resolve_ctx(rec)
        sizing.append(dict(detail, id=rec.get("id"), ctx=ctx))
    return {
        "configured": models,
        "local_files": list_local_models(),
        "defaults": DEFAULT_MODEL_RECORD,
        "sizing": sizing,
        "vram_total": vram_total_bytes(),
        # Ollama-backed boxes: the engine's own live catalogue, so the Models/
        # Library tabs mirror exactly what any other dashboard sees on this box.
        "upstream": await upstream_catalogue(),
        # How the last Save & apply actually went, once the changed models
        # were loaded for real. Sticky until a clean apply replaces it.
        "apply": get_apply_state(),
        # A record this gateway renamed to the fleet's canonical id when it
        # started (FLEET_MODEL_NAMES). The owner did not type that, so the
        # page says it happened.
        "converged": _converged,
    }


# --------------------------------------------------------------------------
# applying a model config, and proving it actually loads
# --------------------------------------------------------------------------
#
# Saving the Models tab used to be a write and a hopeful restart. Two things
# were wrong with that, and they compounded:
#
#   * On the Macs the restart could not work at all (see _darwin_service), so
#     a save wrote a correct models.json and llama-swap.yaml on top of an
#     engine that went on serving the window it was launched with. Measured on
#     mac-laptop-1: both files said `-c 182768`, the running llama-server said
#     `-c 32768`, and the dashboard's toast said "saved".
#   * Nothing ever checked. A context window that does not fit is not refused
#     when it is typed -- it is refused minutes later by a model load nobody
#     is watching, and llama-swap then answers every request for that model
#     with a 502 until someone notices.
#
# So a save is now applied and then PROVEN: each changed model is loaded for
# real, one at a time, and a model that will not load is rolled back to the
# record that last worked. The failure is remembered until a clean apply
# replaces it, because the whole point is that it must not be missable.

APPLY_KEY = "model_apply"
# A cold load of a 20 GB model off mac-laptop-1's external USB drive is minutes, and
# llama-swap's own healthCheckTimeout is 900s. Generous, but bounded.
VERIFY_TIMEOUT = float(os.environ.get("LLMSTACK_VERIFY_TIMEOUT", "420") or 420)
# How long a refused connection right after the llama-swap restart is "not
# listening yet" rather than "refused" -- see _try_load().
VERIFY_CONNECT_GRACE = float(os.environ.get("LLMSTACK_VERIFY_CONNECT_GRACE", "15") or 15)
_verify_task: "asyncio.Task | None" = None

# What an engine says when the weights plus the KV cache do not fit. Matched
# case-insensitively against the tail of llama-swap's log, and used only to
# LABEL a failure -- a model that will not load is rolled back either way,
# because a config that cannot start is broken whatever the reason.
_MEM_SIGNS = (
    "out of memory", "failed to allocate", "unable to allocate",
    "insufficient memory", "cannot allocate", "not enough memory",
    "ggml_backend_alloc", "ggml_metal", "buffer allocation failed",
    "vk::outofdevicememory", "cudamalloc", "hipmalloc",
    "failed to allocate buffer", "kv cache", "insufficient device memory",
)

# The floor a self-tuning retry stops at: the same 4096 grid resolve_ctx()
# itself rounds to, and small enough that a model which cannot even hold
# this owes its operator a real look, not one more automatic guess.
_CTX_RETRY_FLOOR = 4096


def halve_ctx(ctx: int, floor: int = _CTX_RETRY_FLOOR) -> int:
    """The next context size to try after `ctx` failed to fit: half of it,
    rounded down to the same 4096 grid resolve_ctx() rounds to, never below
    `floor`. Returns 0 when there is nothing smaller worth trying -- `ctx`
    was already at or under the floor, so the caller should stop retrying
    and report the failure honestly instead of proposing the same number (or
    an even less useful one) again."""
    if ctx <= floor:
        return 0
    half = (ctx // 2) - (ctx // 2) % 4096
    return max(floor, half)


def get_apply_state() -> dict:
    """The last Save & apply and how it went. Persisted, so the banner
    survives the gateway restart that a failed apply often prompts."""
    rows = db_query("SELECT value FROM settings WHERE key=?", (APPLY_KEY,))
    if not rows:
        return {}
    try:
        data = json.loads(rows[0]["value"])
        return data if isinstance(data, dict) else {}
    except (TypeError, ValueError):
        return {}


def set_apply_state(state: dict) -> None:
    db_exec(
        "INSERT INTO settings(key,value) VALUES (?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (APPLY_KEY, json.dumps(state)),
    )


def _launch_diff(old: dict | None, new: dict) -> list[str]:
    """What changed about this model, in words an operator can act on."""
    if old is None:
        return ["added"]
    out = []
    for k in DEFAULT_MODEL_RECORD:
        if k in ("id", "description"):
            continue
        a, b = old.get(k), new.get(k)
        if a != b:
            out.append(str(k) + " " + json.dumps(a) + " -> " + json.dumps(b))
    return out or ["changed"]


async def _swap_log_tail(lines: int = 200) -> str:
    """llama-swap keeps its recent output on /logs -- which is the only place
    a failed load's reason exists on a box with no journalctl."""
    try:
        assert client is not None
        r = await client.get("/logs", timeout=6.0)
        if r.status_code == 200:
            return "\n".join(r.text.splitlines()[-lines:])
    except Exception:  # noqa: BLE001
        pass
    return ""


async def _try_load(mid: str, want_ctx: int, job: dict) -> tuple[bool, str]:
    """Load one model for real and say whether it came up as configured.

    Returns (ok, reason). Asking llama-swap for a model that is not resident
    is what makes it load -- the same lever the warm-up button pulls -- so the
    proof is a genuine load, not a simulation of one."""
    assert client is not None
    quoted = quote(mid, safe="")
    deadline = time.time() + VERIFY_TIMEOUT

    async def trigger_load() -> Any:
        # The save restarted llama-swap a moment ago, and for the first
        # second or so nothing is listening on its port. That used to be
        # invisible -- a bare llama-swap listens the instant it starts -- but
        # one with a start-up preload hook begins loading BEFORE it listens,
        # and the connect failure that followed was read as "engine refused
        # the load" and rolled a perfectly good change back (apu-box-1,
        # 2026-08-29). A refused connection inside that window is "not up
        # yet"; only one that persists is a verdict.
        assert client is not None
        until = time.time() + VERIFY_CONNECT_GRACE
        while True:
            try:
                return await client.get("/upstream/" + quoted + "/health",
                                        timeout=VERIFY_TIMEOUT)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                if time.time() >= until:
                    raise
                await asyncio.sleep(1.0)

    # Fired and then polled rather than awaited: a cold load outlives any
    # sensible single request timeout, and /running is where the progress is.
    trigger = asyncio.ensure_future(trigger_load())

    async def resident() -> bool:
        try:
            r = await client.get("/running", timeout=8.0)
            if r.status_code != 200:
                return False
            data = r.json()
            for m in (data.get("running") if isinstance(data, dict) else data) or []:
                if isinstance(m, dict) and (m.get("model") or m.get("id")) == mid:
                    return str(m.get("state") or "") == "ready"
        except Exception:  # noqa: BLE001 -- engine mid-restart; keep polling
            pass
        return False

    try:
        while time.time() < deadline:
            if job.get("aborted"):
                return False, "aborted before the load finished"
            if trigger.done():
                # llama-swap holds the proxied request open until the model is
                # actually up, so its answer -- not the poll -- is the
                # authoritative verdict. Anything but a 5xx means it came up.
                try:
                    resp = trigger.result()
                except Exception as exc:  # noqa: BLE001
                    return False, ("engine refused the load ("
                                   + type(exc).__name__ + ")")
                if resp.status_code >= 500:
                    return False, ("engine refused the load (HTTP "
                                   + str(resp.status_code) + ")")
                if resp.status_code >= 400:
                    # A clear client error -- an id llama-swap does not know,
                    # an alias collision that slipped past
                    # check_name_collisions -- is an answer, not a slow load.
                    # Waiting out the whole timeout for it would report it as
                    # "still not ready", which names the wrong problem.
                    return False, ("the engine does not recognise this model "
                                   "(HTTP " + str(resp.status_code) + ")")
                break
            # The other way to see it, for an engine whose proxy does not
            # block on the load.
            if await resident():
                break
            await asyncio.sleep(2.0)
        else:
            return False, ("still not ready after "
                           + str(int(VERIFY_TIMEOUT)) + "s")
    finally:
        if not trigger.done():
            trigger.cancel()
            trigger.add_done_callback(lambda t: t.cancelled() or t.exception())

    # Ready. Now check it got the window it was asked for: llama.cpp refuses a
    # -c it cannot honour rather than shrinking it, but an engine that DOES
    # shrink silently is exactly the failure this whole feature exists to
    # catch, so read the number back rather than trusting the launch.
    _props_cache.pop(mid, None)
    props = await upstream_props(mid)
    got = props.get("n_ctx")
    if want_ctx and isinstance(got, int) and got > 0 and got < want_ctx:
        return False, ("loaded with only " + str(got) + " tokens of context, "
                       + "not the " + str(want_ctx) + " configured")
    return True, ""


async def _verify_apply(queue: list[dict], before: dict[str, dict]) -> None:
    """Load each changed model in turn, rolling back the ones that will not.

    Strictly one at a time. llama-swap runs an exclusive swap group -- only
    one model is resident at a time by design -- so verifying in parallel
    would have each load evict the one being measured.

    An auto-sized model (`ctx` 0 -- see resolve_ctx()) that will not load
    gets ONE graceful retry at half the window it was auto-sized to, instead
    of being reverted outright: "auto" was always a guess from the box's
    VRAM reading, and the box just proved that guess too generous. A PINNED
    ctx gets no such retry -- an operator who typed a number gets an honest
    refusal if it does not fit, not a silently different number. A retry
    that also fails, or has no smaller number left to try (see halve_ctx()),
    reverts exactly like any other failure."""
    failures: list[dict] = []
    verified: list[str] = []
    # mid -> the record to put back, or None for "this model was new, drop it".
    reverts: dict[str, dict | None] = {}
    # Auto-sized models whose first attempt did not fit and still have a
    # smaller number worth trying: mid -> (record pinned at that number,
    # {"from": ctx tried, "to": ctx to retry}). Held separately from
    # `failures` until the retry below is proven or exhausted.
    retry_candidates: dict[str, tuple[dict, dict]] = {}
    ids = [str(r.get("id")) for r in queue]

    def publish(status: str = "running") -> None:
        set_apply_state({
            "at": now(), "status": status, "queue": ids,
            "verified": verified, "failures": failures,
            # Carried so a gateway restart mid-queue has the real records to
            # roll back to, not just a diff string -- see
            # reconcile_orphaned_apply().
            "before": {m: before.get(m) for m in ids},
        })

    async def attempt(mid: str, want_ctx: int, detail: str) -> tuple[bool, str, str]:
        """One real load-and-check against the engine as currently
        configured. Returns (ok, why, lowercased log tail -- empty on ok)."""
        job = _job_open("verify", what=mid, stop=asyncio.Event(), detail=detail)
        try:
            ok, why = await _try_load(mid, want_ctx, job)
        except Exception as exc:  # noqa: BLE001 -- a verify must never wedge
            ok, why = False, type(exc).__name__ + ": " + str(exc)
        finally:
            _job_close(job)
        tail = "" if ok else (await _swap_log_tail()).lower()
        return ok, why, tail

    for rec in queue:
        mid = str(rec.get("id"))
        pinned = int(rec.get("ctx", 0) or 0) > 0
        want = resolve_ctx(rec)[0] if pinned else 0
        ok, why, tail = await attempt(
            mid, want, "loading the new config (evicts whatever is resident)")
        if ok:
            verified.append(mid)
        else:
            overfilled = any(sign in tail for sign in _MEM_SIGNS)
            smaller = 0
            tried_ctx = want
            if overfilled and not pinned:
                tried_ctx = resolve_ctx(rec)[0]
                smaller = halve_ctx(tried_ctx)
            if smaller:
                retry_candidates[mid] = (
                    dict(rec, ctx=smaller), {"from": tried_ctx, "to": smaller})
            else:
                # Only THIS model rolls back: one model asking for more memory
                # than the box has is not a reason to discard the operator's
                # other edits. The rollback is recorded now and written once
                # at the end -- restarting per failure would evict the models
                # this same queue has already verified and warmed.
                old = before.get(mid)
                reverts[mid] = dict(old) if old is not None else None
                failures.append({
                    "id": mid,
                    "changes": _launch_diff(old, rec),
                    "why": ("it does not fit in memory -- " + why) if overfilled
                           else why,
                    "overfilled": bool(overfilled),
                    "reverted": old is not None,
                })
        publish()

    if retry_candidates:
        # Stage the halved configs in the SAME restart as any hard reverts
        # from the loop above -- one extra restart for the whole batch, not
        # one per retried model -- then prove each candidate for real against
        # the engine now actually running with the smaller window. Writing
        # the halved number without this second load-and-check would be
        # exactly the unverified config this whole feature exists to catch.
        staged = dict(reverts)
        staged.update({mid: rec for mid, (rec, _info) in retry_candidates.items()})
        rc, out = _apply_reverts(staged)
        for f in failures:
            f["revert_rc"], f["revert_out"] = rc, out[-400:]
        reverts = {}
        # A restart that did not come back leaves the OLD engine listening,
        # and it would answer every retry's health check happily -- proving
        # nothing, the same trap api_models_put() already refuses to walk
        # into on a failed restart (see its own "did not restart" branch).
        # Every staged retry reverts here instead of being probed.
        for mid, (rrec, info) in retry_candidates.items():
            if rc:
                ok2, why2, tail2 = False, (
                    "llama-swap did not restart after staging the retry "
                    "(rc=" + str(rc) + "), so " + str(info["to"])
                    + " tokens was never proven: " + (out or "")[-200:]), ""
            else:
                ok2, why2, tail2 = await attempt(
                    mid, info["to"],
                    "retrying at " + str(info["to"]) + " tokens of context -- "
                    + str(info["from"]) + " did not fit")
            if ok2:
                verified.append(mid)
                log.info("model %s: auto context self-tuned from %d to %d "
                         "after the larger window would not fit",
                         mid, info["from"], info["to"])
            else:
                overfilled2 = any(sign in tail2 for sign in _MEM_SIGNS)
                old = before.get(mid)
                reverts[mid] = dict(old) if old is not None else None
                failures.append({
                    "id": mid,
                    "changes": _launch_diff(old, rrec),
                    "why": (("does not fit even at " + str(info["to"])
                             + " tokens (tried after " + str(info["from"])
                             + " did not fit either) -- " + why2)
                            if overfilled2 else why2),
                    "overfilled": bool(overfilled2),
                    "reverted": old is not None,
                    "retried": info,
                })
            publish()

    if reverts:
        rc, out = _apply_reverts(reverts)
        for f in failures:
            if "revert_rc" not in f:
                f["revert_rc"], f["revert_out"] = rc, out[-400:]
    publish("failed" if failures else "ok")


def _apply_reverts(reverts: dict) -> tuple[int, str]:
    """Put the failed models' previous records back, in one pass.

    A model with no previous record was new in the save that failed, so there
    is nothing to restore and it is dropped instead."""
    out = []
    for m in load_models():
        mid = str(m.get("id"))
        if mid not in reverts:
            out.append(m)
        elif reverts[mid] is not None:
            out.append(dict(reverts[mid]))
    save_models(out)
    write_swap_config(out)
    return service_control("restart", "llama-swap")


def reconcile_orphaned_apply() -> None:
    """Finish an apply the gateway died in the middle of.

    Without this a restart mid-queue freezes the banner at "Verifying ... 2 of
    4" forever: the in-memory task is gone, the active-jobs table shows nothing
    running, and -- much worse -- every model the queue had not reached yet
    stays live in llama-swap.yaml un-verified, with the banner implying it is
    still being checked. The models that never got their turn are rolled back
    to the records the save started from, which is why those records are
    carried in the state."""
    state = get_apply_state()
    if state.get("status") != "running":
        return
    ids = [str(m) for m in state.get("queue") or []]
    done = set(state.get("verified") or [])
    done |= {str(f.get("id")) for f in state.get("failures") or []}
    stuck = [m for m in ids if m not in done]
    before = state.get("before") or {}
    failures = list(state.get("failures") or [])
    if stuck:
        reverts = {m: (before.get(m) or None) for m in stuck}
        rc, out = _apply_reverts(reverts)
        for mid in stuck:
            failures.append({
                "id": mid,
                "changes": _launch_diff(before.get(mid), before.get(mid) or {}),
                "why": "the gateway restarted mid-verification, so this was "
                       "never proven -- rolled back",
                "overfilled": False,
                "reverted": before.get(mid) is not None,
                "revert_rc": rc,
                "revert_out": out[-400:],
            })
    set_apply_state({
        "at": now(),
        "status": "failed" if failures else "ok",
        "queue": ids,
        "verified": list(state.get("verified") or []),
        "failures": failures,
        "before": before,
    })


def start_verify(queue: list[dict], before: dict[str, dict]) -> list[str]:
    """Queue the changed models for verification, replacing any run still in
    flight -- the newest save is the one that describes what is on disk."""
    global _verify_task
    if _verify_task is not None and not _verify_task.done():
        _verify_task.cancel()
    ids = [str(r.get("id")) for r in queue]
    set_apply_state({
        "at": now(), "status": "running", "queue": ids,
        "verified": [], "failures": [],
        "before": {m: before.get(m) for m in ids},
    })
    _verify_task = asyncio.create_task(_verify_apply(queue, before))
    return ids


@app.put("/admin/api/models")
async def api_models_put(request: Request, admin: dict = Depends(require_admin)) -> dict:
    payload = await request.json()
    models = payload.get("models")
    if not isinstance(models, list):
        raise HTTPException(400, "expected {models: [...]}")
    clean = []
    for rec in models:
        if not isinstance(rec, dict):
            continue
        merged = dict(DEFAULT_MODEL_RECORD)
        merged.update(rec)
        mid = str(merged.get("id", "")).strip()
        if not SAFE_ID.match(mid):
            raise HTTPException(400, "invalid model id: " + mid)
        if not str(merged.get("path", "")).strip():
            raise HTTPException(400, "model " + mid + " has no path")
        clean.append(merged)
    check_name_collisions(clean)
    check_preload_count(clean)
    # Captured before the write, so a rollback has somewhere to roll back TO.
    before = {str(r.get("id")): r for r in load_models()}
    save_models(clean)
    text = write_swap_config(clean)
    code, out = service_control("restart", "llama-swap")
    # build_cmd() IS the launch, so comparing it catches every field that
    # changes how the model starts and ignores every field that does not
    # (a description or a ttl edit should not evict a resident model).
    def _relaunches(rec: dict) -> bool:
        old = before.get(str(rec.get("id")))
        return old is None or build_cmd(old) != build_cmd(rec)

    changed = [r for r in clean if r.get("enabled", True) and _relaunches(r)]
    verifying: list[str] = []
    if changed and not UPSTREAM_MODELS and bool(payload.get("verify", True)):
        if code:
            # The engine did not come back, so there is nothing trustworthy to
            # verify against. Probing anyway is worse than not probing: a stale
            # pre-save llama-swap still holding the OLD config answers every
            # health check happily, and the queue would mark the very save that
            # failed to apply as proven. Say what actually happened instead.
            set_apply_state({
                "at": now(), "status": "failed",
                "queue": [str(r.get("id")) for r in changed],
                "verified": [],
                "failures": [{
                    "id": str(r.get("id")),
                    "changes": _launch_diff(before.get(str(r.get("id"))), r),
                    "why": "llama-swap did not restart (rc=" + str(code)
                           + "), so nothing was applied: " + (out or "")[-200:],
                    "overfilled": False,
                    "reverted": False,
                } for r in changed],
                "before": {str(r.get("id")): before.get(str(r.get("id")))
                           for r in changed},
            })
        else:
            verifying = start_verify(changed, before)
    return {
        "saved": len(clean),
        "restart_rc": code,
        "restart_out": out,
        "rendered": text,
        "verifying": verifying,
        "apply": get_apply_state(),
    }


@app.get("/admin/api/models/apply")
async def api_models_apply_state(admin: dict = Depends(require_admin)) -> dict:
    """Polled by the Models tab while a verification queue is draining, and
    read on load so the failure banner survives a page refresh."""
    return get_apply_state()


@app.get("/admin/api/swap-config", response_class=PlainTextResponse)
async def api_swap_config(admin: dict = Depends(require_admin)) -> str:
    if SWAP_CONFIG.exists():
        return SWAP_CONFIG.read_text()
    return render_swap_config(load_models())


@app.post("/admin/api/service/{action}/{unit}")
async def api_service(action: str, unit: str, admin: dict = Depends(require_admin)) -> dict:
    if action not in ("restart", "start", "stop"):
        raise HTTPException(400, "bad action")
    if unit not in ("llama-swap", "cloudflared"):
        raise HTTPException(400, "unit not managed here")
    if platform.system() == "Darwin" and unit != "llama-swap":
        # cloudflared is not installed on the tailnet-only Macs. llama-swap is,
        # and _darwin_service() now drives it the way cron does.
        raise HTTPException(
            501, "no service manager here -- restart " + unit + " on the box itself"
        )
    if platform.system() not in ("Linux", "Windows", "Darwin"):
        raise HTTPException(
            501, "no service manager here -- restart " + unit + " on the box itself"
        )
    code, out = service_control(action, unit)
    return {"rc": code, "out": out}


@app.post("/admin/api/models/unload")
async def api_unload(request: Request,
                     admin: dict = Depends(require_admin)) -> dict:
    """Evict what is resident -- everything, or one model when the body names
    it ({"model": id}, the Overview card's per-row button).

    All: /unload is llama-swap's route; the old /api/models/unload answers
    405 there and 404 on Ollama, so this used to report success while
    unloading nothing. One: llama-swap's POST /api/models/unload/{id}
    (verified against the deployed build -- 404s an unknown id and leaves
    the rest alone); on Ollama, a keep_alive:0 generate, which is how Ollama
    spells "unload this one"."""
    assert client is not None
    mid = ""
    try:
        body = await request.json()
        if isinstance(body, dict):
            mid = str(body.get("model") or "").strip()
    except Exception:  # noqa: BLE001 -- an empty body means "unload all"
        pass
    if mid:
        for rec in load_models():
            # ids and aliases share one namespace (check_name_collisions),
            # and llama-swap's own unload route resolves an alias to the
            # same model -- so the persistent guard must too, or naming the
            # pinned model by its alias would silently evict it.
            names = {str(rec.get("id"))} | {
                str(a).strip() for a in rec.get("aliases") or []
                if isinstance(a, str) and str(a).strip()}
            if mid in names and rec.get("persistent"):
                raise HTTPException(
                    400, mid + " is pinned resident (persistent) -- "
                    "change that on the Models tab to unload it")
        if UPSTREAM_MODELS:
            tag = (await upstream_alias_pairs()).get(mid, mid)
            try:
                r = await client.post("/api/generate", timeout=60.0,
                                      json={"model": tag, "keep_alive": 0})
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(502, str(exc)) from exc
        else:
            try:
                r = await client.post(
                    "/api/models/unload/" + quote(mid, safe=""), timeout=60.0)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(502, str(exc)) from exc
        if r.status_code < 400:
            _props_cache.clear()
            _running_cache["t"] = 0.0
        return {"status": r.status_code, "body": r.text[:2000], "model": mid}
    for path in ("/unload", "/api/models/unload"):
        try:
            r = await client.get(path, timeout=60.0)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, str(exc)) from exc
        if r.status_code < 400:
            _props_cache.clear()
            _running_cache["t"] = 0.0
            return {"status": r.status_code, "body": r.text[:2000], "via": path}
    return {"status": r.status_code, "body": r.text[:2000], "via": path}


# The one manual load the dashboard may have in flight, for the Overview
# card to poll: status "loading" | "ok" | "failed". A dict and not a queue on
# purpose -- the swap group holds one model at a time, so a second load while
# one runs would only evict the first mid-flight.
_manual_load: dict[str, Any] = {}


async def _manual_load_task(mid: str) -> None:
    ok, why = False, ""
    try:
        if UPSTREAM_MODELS:
            tag = (await upstream_alias_pairs()).get(mid, mid)
            assert client is not None
            r = await client.post("/api/generate", timeout=600.0,
                                  json={"model": tag, "keep_alive": "30m"})
            ok, why = r.status_code < 400, ("" if r.status_code < 400 else
                                            "HTTP " + str(r.status_code))
        else:
            job = _job_open("load", what=mid, stop=asyncio.Event(),
                            detail="loading on demand from the dashboard "
                                   "(evicts whatever is resident)")
            try:
                ok, why = await _try_load(mid, 0, job)
            finally:
                _job_close(job)
    except Exception as exc:  # noqa: BLE001
        ok, why = False, type(exc).__name__ + ": " + str(exc)
    _manual_load.update(status="ok" if ok else "failed", why=why, at=now())
    _running_cache["t"] = 0.0


@app.post("/admin/api/models/load")
async def api_model_load(request: Request,
                         admin: dict = Depends(require_admin)) -> dict:
    """Load one model now ({"model": id}) -- the Overview card's picker.

    Same lever as everything else that loads on demand: the health touch
    _try_load() wraps (llama-swap blocks the request until the model is up),
    or a keep_alive generate on an Ollama box. Answers immediately; the load
    runs as a job (visible under Active jobs, abortable there) and the card's
    poll watches it arrive in the running list."""
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 -- not JSON is a caller error, not a 500
        raise HTTPException(400, "expected {model: id}")
    mid = str((payload or {}).get("model") or "").strip() \
        if isinstance(payload, dict) else ""
    if not mid:
        raise HTTPException(400, "expected {model: id}")
    if UPSTREAM_MODELS:
        known = await served_model_ids()
    else:
        known = {str(r.get("id")) for r in load_models()
                 if r.get("enabled", True)}
    if mid not in known:
        raise HTTPException(404, "not a model this box serves: " + mid[:64])
    if _manual_load.get("status") == "loading":
        raise HTTPException(
            409, "already loading " + str(_manual_load.get("model"))
            + " -- one at a time, the swap group holds one model")
    _manual_load.clear()
    _manual_load.update(model=mid, status="loading", why="", at=now())
    asyncio.create_task(_manual_load_task(mid))
    return {"started": mid, "load": dict(_manual_load)}


@app.put("/admin/api/models/warm")
async def api_models_warm(request: Request,
                          admin: dict = Depends(require_admin)) -> dict:
    """Choose which model this box loads at engine start-up and keeps on warm
    standby ({"model": id}), or none at all ({"model": null}).

    Rewrites the one `preload` flag models.json allows (check_preload_count)
    and restarts llama-swap so the choice takes effect now, not at the next
    reboot. No verify queue: the launch commands are untouched, so there is
    nothing new to prove -- the restart itself preloads the chosen model.
    Persistent models are already resident forever and are refused here."""
    if UPSTREAM_MODELS:
        raise HTTPException(
            501, "an Ollama box has no start-up preload -- the hub's warm-up "
            "loop is what keeps a model resident there")
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 -- not JSON is a caller error, not a 500
        raise HTTPException(400, "expected {model: id} or {model: null}")
    if not isinstance(payload, dict):
        raise HTTPException(400, "expected {model: id} or {model: null}")
    mid = payload.get("model")
    if mid is not None and not isinstance(mid, str):
        raise HTTPException(400, "expected {model: id} or {model: null}")
    mid = (mid or "").strip()
    models = load_models()
    target = None
    for rec in models:
        if mid and str(rec.get("id")) == mid:
            target = rec
    if mid:
        if target is None:
            raise HTTPException(404, "no such model: " + mid[:64])
        if target.get("persistent"):
            raise HTTPException(
                400, mid + " is persistent -- it is already resident from "
                "start-up and never unloaded")
        if not target.get("enabled", True):
            raise HTTPException(400, mid + " is disabled")
    changed = False
    for rec in models:
        want = bool(mid) and str(rec.get("id")) == mid
        if not rec.get("persistent") and bool(rec.get("preload")) != want:
            rec["preload"] = want
            changed = True
    if not changed:
        return {"warm": mid or None, "changed": False}
    save_models(models)
    write_swap_config(models)
    code, out = service_control("restart", "llama-swap")
    _running_cache["t"] = 0.0
    return {"warm": mid or None, "changed": True,
            "restart_rc": code, "restart_out": out[-400:]}


# ---- downloads ----


@app.get("/admin/api/hf/search")
async def hf_search(q: str, admin: dict = Depends(require_admin)) -> Any:
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(
            HF_ENDPOINT + "/api/models",
            params={"search": q, "limit": 25, "sort": "downloads", "direction": -1},
        )
        r.raise_for_status()
        return [
            {"id": m.get("id"), "downloads": m.get("downloads"), "likes": m.get("likes")}
            for m in r.json()
        ]


@app.get("/admin/api/hf/files")
async def hf_files(repo: str, admin: dict = Depends(require_admin)) -> Any:
    if not SAFE_REPO.match(repo):
        raise HTTPException(400, "bad repo id")
    async with httpx.AsyncClient(timeout=45) as c:
        r = await c.get(
            HF_ENDPOINT + "/api/models/" + repo + "/tree/main",
            params={"recursive": "true"},
        )
        r.raise_for_status()
        files = [
            {"path": f.get("path"), "size": f.get("size")}
            for f in r.json()
            if str(f.get("path", "")).endswith(".gguf")
        ]
        files.sort(key=lambda f: f.get("size") or 0)
        return files


@app.post("/admin/api/download")
async def api_download(request: Request, admin: dict = Depends(require_admin)) -> dict:
    payload = await request.json()
    return start_download(str(payload.get("repo", "")), str(payload.get("filename", "")))


@app.get("/admin/api/jobs")
async def api_jobs(admin: dict = Depends(require_admin)) -> list[dict]:
    return db_query("SELECT * FROM jobs ORDER BY id DESC LIMIT 50")


@app.post("/admin/api/jobs/{job_id}/cancel")
async def api_job_cancel(job_id: int, admin: dict = Depends(require_admin)) -> dict:
    _download_cancel.add(job_id)
    return {"cancelled": job_id}


# ---- active work ----


@app.get("/admin/api/active")
async def api_active(admin: dict = Depends(require_admin)) -> dict:
    return {"host": HOST_NAME, "generated_at": now(), "jobs": active_jobs()}


@app.post("/admin/api/active/abort-all")
async def api_active_abort_all(
    kind: str = "", admin: dict = Depends(require_admin)
) -> dict:
    kinds = {k.strip() for k in kind.split(",") if k.strip()}
    return abort_all(kinds or None)


@app.post("/admin/api/active/{job_id}/abort")
async def api_active_abort(
    job_id: str, admin: dict = Depends(require_admin)
) -> dict:
    return abort_one(job_id)


# ---- LM Studio sync -------------------------------------------------------


@app.get("/admin/api/lmstudio")
async def api_lmstudio(admin: dict = Depends(require_admin)) -> dict:
    """What the two stores look like right now and what a sync would do.

    A dry run every time it is called: the Library panel shows exactly the
    action list the Sync button will execute, so the page can never promise
    something different from what happens."""
    plan = await asyncio.to_thread(lmstudio_plan)
    plan["last"] = dict(_lmstudio_last) if _lmstudio_last else {}
    return plan


@app.put("/admin/api/lmstudio/settings")
async def api_lmstudio_settings(
    request: Request, admin: dict = Depends(require_admin)
) -> dict:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(400, "expected an object of settings")
    return set_lmstudio_settings(payload)


@app.post("/admin/api/lmstudio/ollama-import")
async def api_lmstudio_ollama_import(
    request: Request, admin: dict = Depends(require_admin)
) -> dict:
    """Hand one LM Studio GGUF to Ollama on an Ollama-backed box.

    Deliberately explicit and per-model: unlike every other direction here it
    starts by COPYING the weights, because that is the only way into Ollama's
    content-addressed store. The copy is collapsed back onto LM Studio's file
    as soon as it lands, so the cost is temporary -- but it is a cost, and it
    is the operator's to choose."""
    payload = await request.json()
    return await asyncio.to_thread(
        lmstudio_ollama_import,
        str(payload.get("path", "")),
        str(payload.get("name", "")),
    )


@app.post("/admin/api/lmstudio/forget-dismissed")
async def api_lmstudio_forget(admin: dict = Depends(require_admin)) -> dict:
    """Take back every "I deleted this one" the sync is remembering, so the
    next pass offers those LM Studio models again."""
    st = set_lmstudio_settings({"dismissed_paths": []})
    return {"dismissed": st["dismissed_paths"]}


@app.post("/admin/api/lmstudio/sync")
async def api_lmstudio_sync(
    kinds: str = "", admin: dict = Depends(require_admin)
) -> dict:
    """Run a pass now. `kinds` narrows it to a comma-separated subset of
    import/publish/reclaim, which is how the panel offers 'reclaim the
    duplicates' without also registering forty new models."""
    want = {k.strip() for k in kinds.split(",") if k.strip()}
    bad = want - {"import", "publish", "reclaim"}
    if bad:
        raise HTTPException(400, "unknown kind: " + ", ".join(sorted(bad)))
    return await asyncio.to_thread(lmstudio_sync, False, want or None)


@app.delete("/admin/api/local-model")
async def api_delete_model(path: str, admin: dict = Depends(require_admin)) -> dict:
    p = Path(path).resolve()
    if MODELS_DIR.resolve() not in p.parents:
        raise HTTPException(400, "refusing to delete outside the models directory")
    if not p.exists():
        raise HTTPException(404, "not found")
    in_use = [m["id"] for m in load_models() if str(m.get("path")) == str(p)]
    if in_use:
        raise HTTPException(409, "still referenced by: " + ", ".join(in_use))
    p.unlink()
    return {"deleted": str(p)}


# ---- keys ----


@app.get("/admin/api/keys")
async def api_keys(
    limit: int = 25,
    offset: int = 0,
    archived: bool = False,
    names: str = "",
    admin: dict = Depends(require_admin),
) -> dict:
    """One page of keys. Live by default; `archived=true` lists the revoked
    ones still inside their retention, which can be restored.

    `names` scopes the page to an exact allowlist (the public site's per-feature
    dashboard, which only ever wants its own keys) without adding a second
    endpoint -- absent, this is byte-identical to the old behaviour."""
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    where = "archived_at IS NOT NULL" if archived else "archived_at IS NULL"
    # Same trim/empty-drop/200-cap shape as every other bounded list param in
    # this file (see `limit` above) -- a caller can't turn this into an
    # unbounded IN(...) by pasting a huge CSV blob.
    name_list = [n.strip() for n in names.split(",") if n.strip()][:200]
    params: tuple[Any, ...] = ()
    if name_list:
        where += " AND name IN (" + ",".join("?" * len(name_list)) + ")"
        params = tuple(name_list)
    total = db_query("SELECT COUNT(*) n FROM api_keys WHERE " + where, params)[0]["n"]
    keys = db_query(
        "SELECT id,name,prefix,created_at,last_used_at,disabled,"
        "expires_at,max_rpd,max_tpd,max_total_tokens,archived_at "
        "FROM api_keys WHERE " + where + " ORDER BY id DESC LIMIT ? OFFSET ?",
        (*params, limit, offset),
    )
    day_ago = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(
        timespec="seconds"
    )
    day = {
        r["key_id"]: r
        for r in db_query(
            "SELECT key_id, COUNT(*) reqs, COALESCE(SUM(total_tokens),0) toks "
            "FROM usage WHERE ts >= ? GROUP BY key_id",
            (day_ago,),
        )
    }
    # Lifetime totals include what the purge already deleted, so a key's
    # "tokens total" never drops when old request rows are collected.
    life: dict[int, int] = {}
    for r in db_query(
        "SELECT key_id, COALESCE(SUM(total_tokens),0) toks FROM usage "
        "GROUP BY key_id"
    ):
        life[r["key_id"]] = int(r["toks"])
    for r in db_query("SELECT key_id, total_tokens FROM usage_rollup"):
        life[r["key_id"]] = life.get(r["key_id"], 0) + int(r["total_tokens"])
    agents = {
        r["key_id"]: r["name"] or "agent"
        for r in db_query(
            "SELECT key_id,name FROM agents WHERE enabled=1 "
            "AND archived_at IS NULL"
        )
    }
    for k in keys:
        k["reqs_24h"] = int(day.get(k["id"], {}).get("reqs", 0))
        k["tokens_24h"] = int(day.get(k["id"], {}).get("toks", 0))
        k["tokens_total"] = life.get(k["id"], 0)
        k["agent"] = agents.get(k["id"])
    return {"items": keys, "total": int(total), "limit": limit,
            "offset": offset}


@app.post("/admin/api/keys")
async def api_keys_new(request: Request, admin: dict = Depends(require_admin)) -> dict:
    payload = await request.json()
    name = str(payload.get("name", "")).strip() or "unnamed"
    raw, meta = mint_key(
        name,
        expires_at=_norm_expiry(payload.get("expires_at")),
        max_rpd=_norm_limit(payload.get("max_rpd")),
        max_tpd=_norm_limit(payload.get("max_tpd")),
        max_total_tokens=_norm_limit(payload.get("max_total_tokens")),
    )
    return {"key": raw, **meta, "note": "shown once -- store it now"}


@app.patch("/admin/api/keys/{key_id}")
async def api_keys_patch(
    key_id: int, request: Request, admin: dict = Depends(require_admin)
) -> dict:
    payload = await request.json()
    fields: dict[str, Any] = {}
    if "name" in payload:
        fields["name"] = str(payload["name"]).strip() or "unnamed"
    if "expires_at" in payload:
        fields["expires_at"] = _norm_expiry(payload["expires_at"])
    for f in ("max_rpd", "max_tpd", "max_total_tokens"):
        if f in payload:
            fields[f] = _norm_limit(payload[f])
    if not fields:
        raise HTTPException(400, "nothing to update")
    sets = ", ".join(f + "=?" for f in fields)
    if not db_update(
        "UPDATE api_keys SET " + sets + " WHERE id=? AND archived_at IS NULL",
        (*fields.values(), key_id),
    ):
        raise HTTPException(404, "no such live key -- restore it first")
    return db_query(
        "SELECT id,name,prefix,expires_at,max_rpd,max_tpd,max_total_tokens,disabled "
        "FROM api_keys WHERE id=?",
        (key_id,),
    )[0]


@app.delete("/admin/api/keys/{key_id}")
async def api_keys_del(key_id: int, admin: dict = Depends(require_admin)) -> dict:
    """Archive a key: it stops authenticating immediately, and stays
    restorable until the retention period runs out.

    Deliberately not a DELETE of the row. Revoking the wrong key is a thing
    that happens, and a hashed key cannot be re-minted -- everything using it
    would need a new one. Archiving keeps that recoverable while still being
    an instant, complete revoke: require_api_key refuses an archived row."""
    stamp = now()
    if not db_update(
        "UPDATE api_keys SET archived_at=?, disabled=1 "
        "WHERE id=? AND archived_at IS NULL",
        (stamp, key_id),
    ):
        raise HTTPException(404, "no such key, or it is already archived")
    # The profile follows its key, so an archived key stops injecting prompts
    # as well as stops authenticating. A team follows the same way: a revoked
    # key must not leave a crew that could be re-armed by restoring the row.
    db_update(
        "UPDATE agents SET archived_at=? WHERE key_id=? AND archived_at IS NULL",
        (stamp, key_id),
    )
    db_update(
        "UPDATE teams SET archived_at=? WHERE key_id=? AND archived_at IS NULL",
        (stamp, key_id),
    )
    return {"archived": key_id, "at": stamp}


@app.post("/admin/api/keys/{key_id}/restore")
async def api_keys_restore(
    key_id: int, admin: dict = Depends(require_admin)
) -> dict:
    """Bring an archived key back. It returns DISABLED rather than live: a key
    that was revoked should not start accepting traffic again the moment
    somebody un-files it -- enabling is a second, deliberate click."""
    if not db_update(
        "UPDATE api_keys SET archived_at=NULL "
        "WHERE id=? AND archived_at IS NOT NULL",
        (key_id,),
    ):
        raise HTTPException(404, "no such archived key")
    db_update("UPDATE agents SET archived_at=NULL WHERE key_id=?", (key_id,))
    db_update("UPDATE teams SET archived_at=NULL WHERE key_id=?", (key_id,))
    return {"restored": key_id, "disabled": True}


@app.delete("/admin/api/keys/{key_id}/purge")
async def api_keys_purge(
    key_id: int, admin: dict = Depends(require_admin)
) -> dict:
    """Delete an archived key for good, now instead of at the end of its
    retention. Refused on a live key: nothing here is destroyed in one step."""
    rows = db_query(
        "SELECT id FROM api_keys WHERE id=? AND archived_at IS NOT NULL",
        (key_id,),
    )
    if not rows:
        raise HTTPException(409, "archive it first -- live keys are not "
                                 "deleted in one step")
    db_exec("DELETE FROM api_keys WHERE id=?", (key_id,))
    db_exec("DELETE FROM agents WHERE key_id=?", (key_id,))
    db_exec("DELETE FROM teams WHERE key_id=?", (key_id,))
    return {"deleted": key_id}


@app.post("/admin/api/keys/{key_id}/toggle")
async def api_keys_toggle(key_id: int, admin: dict = Depends(require_admin)) -> dict:
    if not db_update(
        "UPDATE api_keys SET disabled = 1 - disabled "
        "WHERE id=? AND archived_at IS NULL",
        (key_id,),
    ):
        raise HTTPException(409, "no such live key -- restore it first")
    return db_query("SELECT id,name,disabled FROM api_keys WHERE id=?", (key_id,))[0]


# ---- agents ----


@app.get("/admin/api/agents")
async def api_agents(
    limit: int = 25,
    offset: int = 0,
    archived: bool = False,
    admin: dict = Depends(require_admin),
) -> dict:
    """One page of the key-to-profile table.

    The live view is keyed off the KEY, not the profile: a live key with no
    profile is exactly the row you need in order to create one. The archived
    view is the opposite -- profiles filed away, listed so they can be
    restored, including any whose key is archived with them."""
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    if archived:
        total = db_query(
            "SELECT COUNT(*) n FROM agents WHERE archived_at IS NOT NULL"
        )[0]["n"]
        rows = db_query(
            "SELECT a.*, k.name key_name, k.prefix, k.disabled key_disabled, "
            "k.archived_at key_archived FROM agents a "
            "LEFT JOIN api_keys k ON k.id=a.key_id "
            "WHERE a.archived_at IS NOT NULL ORDER BY a.key_id DESC "
            "LIMIT ? OFFSET ?",
            (limit, offset),
        )
        items = [
            {
                "key_id": r["key_id"],
                "key_name": r["key_name"] or ("key #%d" % r["key_id"]),
                "prefix": r["prefix"] or "",
                "key_disabled": r["key_disabled"],
                "key_archived": r["key_archived"],
                "profile": {k: r[k] for k in
                            ("id", "key_id", "enabled", "name", "system_prompt",
                             "rules", "allowed_models", "force_model",
                             "param_overrides", "updated_at", "archived_at")},
            }
            for r in rows
        ]
        return {"items": items, "total": int(total), "limit": limit,
                "offset": offset}
    total = db_query(
        "SELECT COUNT(*) n FROM api_keys WHERE archived_at IS NULL"
    )[0]["n"]
    keys = db_query(
        "SELECT id,name,prefix,disabled FROM api_keys WHERE archived_at IS NULL "
        "ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    )
    profiles = {
        r["key_id"]: r
        for r in db_query("SELECT * FROM agents WHERE archived_at IS NULL")
    }
    items = [
        {
            "key_id": k["id"],
            "key_name": k["name"],
            "prefix": k["prefix"],
            "key_disabled": k["disabled"],
            "profile": profiles.get(k["id"]),
        }
        for k in keys
    ]
    return {"items": items, "total": int(total), "limit": limit,
            "offset": offset}


@app.put("/admin/api/agents/{key_id}")
async def api_agents_put(
    key_id: int, request: Request, admin: dict = Depends(require_admin)
) -> dict:
    if not db_query(
        "SELECT id FROM api_keys WHERE id=? AND archived_at IS NULL", (key_id,)
    ):
        raise HTTPException(404, "no such live key")
    p = await request.json()
    allowed = p.get("allowed_models") or []
    if not isinstance(allowed, list):
        raise HTTPException(400, "allowed_models must be a list")
    overrides = p.get("param_overrides") or {}
    if not isinstance(overrides, dict):
        raise HTTPException(400, "param_overrides must be an object")
    row = (
        int(bool(p.get("enabled", True))),
        str(p.get("name", "")).strip(),
        str(p.get("system_prompt", "")),
        str(p.get("rules", "")),
        json.dumps([str(a) for a in allowed if str(a).strip()]),
        str(p.get("force_model", "")).strip(),
        json.dumps(overrides),
        now(),
        key_id,
    )
    if db_query("SELECT id FROM agents WHERE key_id=?", (key_id,)):
        # archived_at cleared: saving a profile is the clearest statement that
        # you want it in force, so it comes back out of the archive.
        db_exec(
            "UPDATE agents SET enabled=?,name=?,system_prompt=?,rules=?,"
            "allowed_models=?,force_model=?,param_overrides=?,updated_at=?,"
            "archived_at=NULL WHERE key_id=?",
            row,
        )
        # ctx_limit is deliberately absent from that UPDATE: the Agents tab
        # does not know about it, and a Fleet Pass key's context cap must
        # survive someone editing its prompt. It moves only when a caller
        # actually sends the field (null or 0 clears it).
        if "ctx_limit" in p:
            db_exec("UPDATE agents SET ctx_limit=? WHERE key_id=?",
                    (_norm_limit(p.get("ctx_limit")), key_id))
    else:
        db_exec(
            "INSERT INTO agents(enabled,name,system_prompt,rules,allowed_models,"
            "force_model,param_overrides,updated_at,key_id,ctx_limit) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (*row, _norm_limit(p.get("ctx_limit"))),
        )
    return db_query("SELECT * FROM agents WHERE key_id=?", (key_id,))[0]


@app.delete("/admin/api/agents/{key_id}")
async def api_agents_del(key_id: int, admin: dict = Depends(require_admin)) -> dict:
    """Archive a profile: it stops being injected at once, and a prompt you
    spent an afternoon writing is still there if removing it was a mistake."""
    if not db_update(
        "UPDATE agents SET archived_at=? WHERE key_id=? AND archived_at IS NULL",
        (now(), key_id),
    ):
        raise HTTPException(404, "no such live profile")
    return {"archived": key_id}


@app.post("/admin/api/agents/{key_id}/restore")
async def api_agents_restore(
    key_id: int, admin: dict = Depends(require_admin)
) -> dict:
    if not db_query(
        "SELECT id FROM api_keys WHERE id=? AND archived_at IS NULL", (key_id,)
    ):
        raise HTTPException(409, "restore the key first -- a profile without a "
                                 "live key enforces nothing")
    if not db_update(
        "UPDATE agents SET archived_at=NULL WHERE key_id=? "
        "AND archived_at IS NOT NULL",
        (key_id,),
    ):
        raise HTTPException(404, "no such archived profile")
    return {"restored": key_id}


@app.delete("/admin/api/agents/{key_id}/purge")
async def api_agents_purge(
    key_id: int, admin: dict = Depends(require_admin)
) -> dict:
    if not db_query(
        "SELECT id FROM agents WHERE key_id=? AND archived_at IS NOT NULL",
        (key_id,),
    ):
        raise HTTPException(409, "archive it first -- live profiles are not "
                                 "deleted in one step")
    db_exec("DELETE FROM agents WHERE key_id=?", (key_id,))
    return {"deleted": key_id}


# ---- teams (admin) ----


def _team_row(p: dict) -> tuple:
    worker_models = p.get("worker_models") or []
    if not isinstance(worker_models, list):
        raise HTTPException(400, "worker_models must be a list")
    primary = str(p.get("primary_model", "")).strip()
    if not primary:
        raise HTTPException(400, "a team needs a primary model")
    return (
        int(bool(p.get("enabled", True))),
        str(p.get("name", "")).strip(),
        primary,
        json.dumps([str(w).strip() for w in worker_models if str(w).strip()]),
        max(1, min(32, int(p.get("max_workers") or 4))),
        max(1, min(24, int(p.get("max_rounds") or 6))),
        str(p.get("system_prompt", "")),
        str(p.get("worker_prompt", "")),
        now(),
    )


@app.get("/admin/api/teams")
async def api_teams(
    archived: bool = False, admin: dict = Depends(require_admin)
) -> dict:
    """Every team with its key. Small list, no paging: a fleet with dozens of
    standing crews has outgrown this dashboard anyway."""
    where = "t.archived_at IS NOT NULL" if archived else "t.archived_at IS NULL"
    rows = db_query(
        "SELECT t.*, k.name key_name, k.prefix, k.disabled key_disabled, "
        "k.archived_at key_archived FROM teams t "
        "LEFT JOIN api_keys k ON k.id=t.key_id WHERE " + where +
        " ORDER BY t.id DESC")
    return {"items": rows}


@app.post("/admin/api/teams")
async def api_teams_new(
    request: Request, admin: dict = Depends(require_admin)
) -> dict:
    """Create a team and mint its key in one step. The key is shown once,
    like any other; everything about the crew is editable afterwards."""
    p = await request.json()
    row = _team_row(p)
    name = row[1] or "team"
    raw, meta = mint_key(name)
    db_exec(
        "INSERT INTO teams(enabled,name,primary_model,worker_models,"
        "max_workers,max_rounds,system_prompt,worker_prompt,updated_at,key_id)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (*row, meta["id"]),
    )
    team = db_query("SELECT * FROM teams WHERE key_id=?", (meta["id"],))[0]
    return {"key": raw, "key_meta": meta, "team": team,
            "note": "key shown once -- store it now"}


@app.put("/admin/api/teams/{key_id}")
async def api_teams_put(
    key_id: int, request: Request, admin: dict = Depends(require_admin)
) -> dict:
    if not db_query(
        "SELECT id FROM api_keys WHERE id=? AND archived_at IS NULL", (key_id,)
    ):
        raise HTTPException(404, "no such live key")
    row = _team_row(await request.json())
    if db_query("SELECT id FROM teams WHERE key_id=?", (key_id,)):
        db_exec(
            "UPDATE teams SET enabled=?,name=?,primary_model=?,"
            "worker_models=?,max_workers=?,max_rounds=?,system_prompt=?,"
            "worker_prompt=?,updated_at=?,archived_at=NULL WHERE key_id=?",
            (*row, key_id),
        )
    else:
        db_exec(
            "INSERT INTO teams(enabled,name,primary_model,worker_models,"
            "max_workers,max_rounds,system_prompt,worker_prompt,updated_at,"
            "key_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (*row, key_id),
        )
    return db_query("SELECT * FROM teams WHERE key_id=?", (key_id,))[0]


@app.delete("/admin/api/teams/{key_id}")
async def api_teams_del(key_id: int, admin: dict = Depends(require_admin)) -> dict:
    """Archive a team: its key immediately goes back to being an ordinary
    key -- no injection, no tool, no orchestration."""
    if not db_update(
        "UPDATE teams SET archived_at=? WHERE key_id=? AND archived_at IS NULL",
        (now(), key_id),
    ):
        raise HTTPException(404, "no such live team")
    return {"archived": key_id}


@app.post("/admin/api/teams/{key_id}/restore")
async def api_teams_restore(
    key_id: int, admin: dict = Depends(require_admin)
) -> dict:
    if not db_query(
        "SELECT id FROM api_keys WHERE id=? AND archived_at IS NULL", (key_id,)
    ):
        raise HTTPException(409, "restore the key first -- a team without a "
                                 "live key can never be called")
    if not db_update(
        "UPDATE teams SET archived_at=NULL WHERE key_id=? "
        "AND archived_at IS NOT NULL",
        (key_id,),
    ):
        raise HTTPException(404, "no such archived team")
    return {"restored": key_id}


@app.delete("/admin/api/teams/{key_id}/purge")
async def api_teams_purge(
    key_id: int, admin: dict = Depends(require_admin)
) -> dict:
    if not db_query(
        "SELECT id FROM teams WHERE key_id=? AND archived_at IS NOT NULL",
        (key_id,),
    ):
        raise HTTPException(409, "archive it first -- live teams are not "
                                 "deleted in one step")
    db_exec("DELETE FROM teams WHERE key_id=?", (key_id,))
    return {"deleted": key_id}


# ---- batches (admin) ----


@app.get("/admin/api/batches")
async def api_batches(
    limit: int = 25, offset: int = 0, archived: bool = False,
    admin: dict = Depends(require_admin),
) -> dict:
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    where = "archived_at IS NOT NULL" if archived else "archived_at IS NULL"
    total = db_query("SELECT COUNT(*) n FROM batches WHERE " + where)[0]["n"]
    rows = db_query(
        "SELECT * FROM batches WHERE " + where +
        " ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))
    return {"items": [_batch_status(r) for r in rows], "total": int(total),
            "limit": limit, "offset": offset}


@app.get("/admin/api/batches/{bid}/results")
async def api_batch_results(bid: int, admin: dict = Depends(require_admin)):
    if not db_query("SELECT id FROM batches WHERE id=?", (bid,)):
        raise HTTPException(404, "no such batch")
    _, out_path = _batch_paths(bid)
    if not out_path.exists():
        return PlainTextResponse("", media_type="application/x-ndjson")
    return FileResponse(out_path, media_type="application/x-ndjson",
                        filename="batch-" + str(bid) + ".ndjson")


@app.post("/admin/api/batches/{bid}/cancel")
async def api_batch_cancel(bid: int, admin: dict = Depends(require_admin)) -> dict:
    if not db_query("SELECT id FROM batches WHERE id=? AND status='running'",
                    (bid,)):
        raise HTTPException(404, "no running batch with that id")
    _batch_cancel.add(bid)
    return {"cancelling": bid}


@app.delete("/admin/api/batches/{bid}")
async def api_batch_del(bid: int, admin: dict = Depends(require_admin)) -> dict:
    """Archive a finished batch. A running one must be cancelled first --
    hiding a job that is still spending GPU time is how money disappears."""
    if db_query("SELECT id FROM batches WHERE id=? AND status='running'", (bid,)):
        raise HTTPException(409, "cancel it first -- this batch is running")
    if not db_update(
        "UPDATE batches SET archived_at=? WHERE id=? AND archived_at IS NULL",
        (now(), bid),
    ):
        raise HTTPException(404, "no such live batch")
    return {"archived": bid}


@app.post("/admin/api/batches/{bid}/restore")
async def api_batch_restore(bid: int, admin: dict = Depends(require_admin)) -> dict:
    if not db_update(
        "UPDATE batches SET archived_at=NULL WHERE id=? "
        "AND archived_at IS NOT NULL", (bid,),
    ):
        raise HTTPException(404, "no such archived batch")
    return {"restored": bid}


@app.delete("/admin/api/batches/{bid}/purge")
async def api_batch_purge(bid: int, admin: dict = Depends(require_admin)) -> dict:
    """Delete an archived batch and its spool files for good."""
    if not db_query(
        "SELECT id FROM batches WHERE id=? AND archived_at IS NOT NULL", (bid,)
    ):
        raise HTTPException(409, "archive it first -- live batches are not "
                                 "deleted in one step")
    db_exec("DELETE FROM batches WHERE id=?", (bid,))
    for p in _batch_paths(bid):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
    return {"deleted": bid}


# ---- usage ----


@app.get("/admin/api/usage")
async def api_usage(
    hours: int = 24,
    recent_limit: int = 25,
    recent_offset: int = 0,
    recent_archived: bool = False,
    key_names: str = "",
    public: str = "",
    admin: dict = Depends(require_admin),
) -> dict:
    """Aggregates over the chosen window, plus one page of the request log.

    The aggregates deliberately ignore the archive flag: a row that aged out
    of the live LIST is still traffic that happened, and leaving it out of the
    24h chart would make the numbers lie. Only the request log itself is a
    live view.

    `key_names`/`public` scope every query below to one project's own traffic
    (the public site's dashboard: its feature keys, optionally unioned with the
    Fleet Pass keys it issued) instead of the whole hub's. Neither present ->
    no extra predicate at all, so this stays byte-identical to today for the
    hub's own dashboard, which never sends them."""
    hours = max(1, min(hours, 24 * 30))
    since = "datetime('now','-" + str(hours) + " hours')"
    # Same trim/empty-drop/200-cap shape as `names` on /admin/api/keys.
    name_list = [n.strip() for n in key_names.split(",") if n.strip()][:200]
    want_public = public.strip().lower() in ("1", "true", "yes")
    scope_sql = ""
    scope_params: tuple[Any, ...] = ()
    if name_list or want_public:
        halves = []
        params: list[Any] = []
        if name_list:
            halves.append("key_name IN (" + ",".join("?" * len(name_list)) + ")")
            params.extend(name_list)
        if want_public:
            # key_id IS NOT NULL: a pending/rejected public_keys row never
            # minted a key, and a bare `IN (SELECT ... NULL)` would otherwise
            # let a NULL key_id sit in the subquery doing nothing -- explicit
            # is cheaper to read than relying on SQLite's NULL-in-IN behaviour.
            halves.append(
                "key_id IN (SELECT key_id FROM public_keys WHERE key_id IS NOT NULL)"
            )
        scope_sql = " AND (" + " OR ".join(halves) + ")"
        scope_params = tuple(params)
    totals = db_query(
        "SELECT COUNT(*) reqs, COALESCE(SUM(prompt_tokens),0) pt, "
        "COALESCE(SUM(completion_tokens),0) ct, "
        "CAST(COALESCE(AVG(ttft_ms),0) AS INT) avg_ttft, "
        "CAST(COALESCE(AVG(latency_ms),0) AS INT) avg_latency "
        "FROM usage WHERE ts >= " + since + scope_sql,
        scope_params,
    )
    by_model = db_query(
        "SELECT model, COUNT(*) reqs, COALESCE(SUM(completion_tokens),0) ct, "
        "CAST(COALESCE(AVG(latency_ms),0) AS INT) avg_latency "
        "FROM usage WHERE ts >= " + since + scope_sql +
        " GROUP BY model ORDER BY reqs DESC",
        scope_params,
    )
    by_key = db_query(
        "SELECT key_name, COUNT(*) reqs, COALESCE(SUM(total_tokens),0) tt "
        "FROM usage WHERE ts >= " + since + scope_sql +
        " GROUP BY key_name ORDER BY reqs DESC",
        scope_params,
    )
    series = db_query(
        "SELECT strftime('%Y-%m-%dT%H:00', ts) bucket, COUNT(*) reqs, "
        "COALESCE(SUM(completion_tokens),0) ct "
        "FROM usage WHERE ts >= " + since + scope_sql +
        " GROUP BY bucket ORDER BY bucket",
        scope_params,
    )
    recent_limit = max(1, min(recent_limit, 200))
    recent_offset = max(0, recent_offset)
    rwhere = ("archived_at IS NOT NULL" if recent_archived
              else "archived_at IS NULL") + scope_sql
    recent_total = db_query(
        "SELECT COUNT(*) n FROM usage WHERE " + rwhere, scope_params
    )[0]["n"]
    recent = db_query(
        "SELECT ts,key_name,model,status,prompt_tokens,completion_tokens,"
        "ttft_ms,latency_ms,stream FROM usage WHERE " + rwhere +
        " ORDER BY id DESC LIMIT ? OFFSET ?",
        (*scope_params, recent_limit, recent_offset),
    )
    return {
        "totals": totals[0] if totals else {},
        "by_model": by_model,
        "by_key": by_key,
        "series": series,
        "recent": recent,
        "recent_total": int(recent_total),
        "recent_limit": recent_limit,
        "recent_offset": recent_offset,
        "retention": get_settings(),
        "scope": (
            {"key_names": name_list, "public": want_public}
            if (name_list or want_public) else None
        ),
    }


@app.get("/admin/api/settings")
async def api_settings(admin: dict = Depends(require_admin)) -> dict:
    return get_settings()


@app.put("/admin/api/settings")
async def api_settings_put(
    request: Request, admin: dict = Depends(require_admin)
) -> dict:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(400, "body must be an object")
    return set_settings(payload)


@app.get("/admin/api/maintenance")
async def api_maintenance(admin: dict = Depends(require_admin)) -> dict:
    """What retention is holding, and what the next purge would take."""
    return retention_stats()


@app.post("/admin/api/maintenance/{verb}")
async def api_maintenance_run(
    verb: str, admin: dict = Depends(require_admin)
) -> dict:
    """roll = age rows out of the live views now; purge = do that and then
    permanently delete everything past its retention. Both are what the daily
    task does, on demand."""
    if verb == "roll":
        return {"rolled": roll_window(), "stats": retention_stats()}
    if verb == "purge":
        res = maintain()
        res["stats"] = retention_stats()
        return res
    raise HTTPException(400, "verb must be roll|purge")


@app.get("/admin/api/logs", response_class=PlainTextResponse)
async def api_logs(
    unit: str = "llama-swap", lines: int = 300, admin: dict = Depends(require_admin)
) -> str:
    if unit not in ("llama-swap", "llm-gateway", "cloudflared", "tailscaled"):
        raise HTTPException(400, "unit not managed here")
    lines = max(10, min(lines, 2000))
    try:
        p = subprocess.run(
            ["journalctl", "-u", unit + ".service", "-n", str(lines), "--no-pager"],
            capture_output=True, text=True, timeout=30,
        )
        return p.stdout or p.stderr
    except Exception as exc:  # noqa: BLE001
        return "could not read logs: " + str(exc)


# ------------------------------- fleet ------------------------------------
#
# peers.json: [{"name":"apu-box-1","url":"http://192.168.1.151:8080",
#               "token":"<that host's LLMSTACK_ADMIN_TOKEN>",
#               "api_url":"https://max.example.com/v1"}]
# The dashboard reaches every peer through /admin/api/fleet/<name>/..., which
# forwards to <url>/admin/api/... with the peer's admin token. Peers are other
# instances of this same gateway, so the whole admin surface works cross-host.


def load_peers() -> list[dict]:
    if not PEERS_PATH.exists():
        return []
    try:
        data = json.loads(PEERS_PATH.read_text())
    except json.JSONDecodeError:
        return []
    return [p for p in data if isinstance(p, dict) and p.get("name") and p.get("url")]


def peer_routed(p: dict) -> bool:
    """The killswitch state of a peer record. Missing means routed: the flag
    postdates every peers.json in the fleet, and a box that has never been
    touched by the toggle must route exactly as it always did."""
    v = p.get("routed", True)
    if v is False:
        return False
    if isinstance(v, str) and v.strip().lower() in ("false", "0", "off", "no"):
        return False
    return True


def routeable_peers() -> list[dict]:
    """load_peers() filtered to the ones the routing table may use. A peer
    whose killswitch is off stays registered, stays on the fleet page with
    its live/last-known status intact, and is simply not a candidate for any
    route, warm-up or preload until the toggle is flipped back."""
    return [p for p in load_peers() if peer_routed(p)]


def save_peers(peers: list[dict]) -> None:
    write_atomic(PEERS_PATH, json.dumps(peers, indent=2))
    try:
        PEERS_PATH.chmod(0o600)  # peer admin tokens live in here
    except OSError:
        pass


SAFE_PEER = re.compile(r"^[A-Za-z0-9._-]{1,32}$")


@app.get("/admin/api/fleet")
async def api_fleet(admin: dict = Depends(require_admin)) -> dict:
    peers = load_peers()

    async def ping(p: dict) -> dict:
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(p["url"].rstrip("/") + "/health")
                ok = r.status_code == 200
        except Exception:  # noqa: BLE001
            ok = False
        return {
            "name": p["name"],
            "kind": "peer",
            "url": p["url"],
            "api_url": p.get("api_url", ""),
            "routed": peer_routed(p),
            "online": ok,
        }

    checked = await asyncio.gather(*(ping(p) for p in peers))
    return {
        "hosts": [
            {
                "name": HOST_NAME,
                "kind": "self",
                "api_url": PUBLIC_API_URL,
                "online": True,
            },
            *checked,
        ]
    }


@app.get("/admin/api/fleet/summary")
async def fleet_summary(admin: dict = Depends(require_admin)) -> dict:
    """Storage (and a little context) for every host, in one round trip.

    Declared ahead of the fleet catch-all below so 'summary' is never mistaken
    for a peer name.
    """

    def local() -> dict:
        st = storage_info()
        gpu = amdgpu_stats()
        vram = next((g for g in gpu if g.get("vram_total")), {})
        return {
            "name": HOST_NAME,
            "kind": "self",
            "online": True,
            "storage": st,
            "vram_total": vram.get("vram_total"),
            "vram_used": vram.get("vram_used"),
            "models_configured": len([m for m in load_models() if m.get("enabled")]),
        }

    async def remote(p: dict) -> dict:
        base = {"name": p["name"], "kind": "peer", "online": False, "storage": None}
        try:
            async with httpx.AsyncClient(timeout=6.0) as c:
                r = await c.get(
                    p["url"].rstrip("/") + "/admin/api/status",
                    headers={"Authorization": "Bearer " + p.get("token", "")},
                )
                if r.status_code != 200:
                    base["error"] = "HTTP " + str(r.status_code)
                    return base
                d = r.json()
                host = d.get("host") or {}
                gpu = (host.get("gpu") or [])
                vram = next((g for g in gpu if g.get("vram_total")), {})
                base.update(
                    online=True,
                    storage=host.get("storage"),
                    vram_total=vram.get("vram_total"),
                    vram_used=vram.get("vram_used"),
                    models_configured=len(d.get("models_running") or []),
                )
        except Exception as exc:  # noqa: BLE001
            base["error"] = str(exc)[:120]
        return base

    peers = load_peers()
    results = await asyncio.gather(*(remote(p) for p in peers)) if peers else []
    return {"hosts": [local(), *results]}


# --------------------------------------------------------------------------
# fleet home page: pooled resources, last-known state, network map
# --------------------------------------------------------------------------

# Static per-host hardware facts telemetry cannot see (bandwidth, TFLOPS) or
# cannot see while the box is offline (RAM, VRAM). All estimates -- the
# bandwidth/TFLOPS numbers come from the distributed-inference capacity plan.
# Overridable per deployment via STATE/specs.json ({host: {field: value}}).
#
# The routing policy reads six more keys (see host_tier() and
# _score_host_model_pairs()):
#   klass        'gpu' | 'big' | 'small' | 'fallback' | 'hub'
#   always_on    the owner's order among the always-on boxes (1 = first);
#                doubles as the rank within a tier
#   rank         same thing for boxes outside that set (apu-box-1 before apu-tablet-1)
#   moe_spill_ok a GPU box that stays in the fast tier for a mixture-of-experts
#                model whose experts overflow its cards (gpu-desktop-2: 48 GB of
#                VRAM and 192 GB of RAM behind it)
#   reserve      somebody's personal machine. It stays registered, serves what
#                it serves, and is chosen only when every non-reserve candidate
#                is saturated or cooling -- however fast its card is. The
#                preload loop never warms it. Owner's word, not telemetry:
#                only gpu-desktop-1 has a watchdog that notices a human at the
#                keyboard; for the rest this flag is the only thing standing
#                between a personal box and fleet traffic it would otherwise
#                win on raw specs (gpu-desktop-2 outranks everything on paper).
#   tg_tps / pp_tps   decode and prompt-read tokens/sec to assume before any
#                usage history exists (class defaults otherwise)
DEFAULT_SPECS: dict[str, dict] = {
    "apu-box-1":     {"ram_gb": 128, "vram_gb": 96, "mem_bw_gbs": 210, "gpu_tflops": 30,
                   "cpu": "Ryzen AI Max+ 395", "gpu": "Radeon 8060S (Vulkan)",
                   "klass": "big", "rank": 1},
    # 128 GB fitted; the BIOS carve leaves ~32 GB visible to Windows as DRAM.
    "apu-tablet-1": {"ram_gb": 32, "vram_gb": 96, "mem_bw_gbs": 200, "gpu_tflops": 25,
                   "cpu": "Ryzen AI Max+ 395", "gpu": "Radeon 8060S (tablet thermals)",
                   "klass": "big", "rank": 2, "reserve": True},
    # Same Z13 silicon as apu-tablet-1, 64 GB fitted, firmware carve of 32 GB.
    # `ram_gb` is what Windows sees, as it is for apu-tablet-1. `vram_gb` is the
    # pool a model may actually occupy: WDDM lends the GPU half the DRAM on top
    # of the carve, so Vulkan offers 47.8 GiB and the node budgets 44 of it
    # (LLMSTACK_CONTEXT_BUDGET_GIB). Ranked behind both -- it is the smallest
    # of the three and the one with someone sitting in front of it.
    "apu-tablet-2": {"ram_gb": 32, "vram_gb": 44, "mem_bw_gbs": 200, "gpu_tflops": 25,
                  "cpu": "Ryzen AI Max+ 395", "gpu": "Radeon 8060S (32 GiB carve + WDDM)",
                  "klass": "big", "rank": 3, "reserve": True},
    "gpu-desktop-2":  {"ram_gb": 192, "vram_gb": 48, "mem_bw_gbs": 936, "gpu_tflops": 71,
                   "cpu": "i9-14900", "gpu": "2× RTX 3090 (CUDA)",
                   "klass": "gpu", "moe_spill_ok": True, "tg_tps": 80, "pp_tps": 3000,
                   "reserve": True},
    # Renamed from m1-laptop 2026-08-25 (tailnet + this table + peers.json); same
    # machine, same address. The key here IS the identity: it is looked up by
    # a box's LLMSTACK_HOST_NAME and by peer name, so a rename that misses
    # this table costs the box its specs, its routing rank and its telemetry.
    "mac-laptop-1":     {"ram_gb": 64, "vram_gb": 56, "mem_bw_gbs": 330, "gpu_tflops": 21,
                   "cpu": "Apple M1 Max", "gpu": "32-core (Metal)", "klass": "gpu"},
    "gpu-laptop-1":   {"ram_gb": 32, "vram_gb": 8, "mem_bw_gbs": 224, "gpu_tflops": 18,
                   "cpu": "Ryzen 9 6900HS", "gpu": "RX 6700S",
                   "klass": "gpu", "always_on": 1},
    # A laptop that comes and goes. Ranked behind gpu-laptop-1 (always_on 1) in
    # the GPU tier on purpose: with no rank at all it sorted FIRST there (an
    # unranked box defaults to 0), so whenever it was awake it took the 9B
    # distill's traffic off the always-on box -- and dropped it mid-day when
    # the lid closed. The hub's demo policy already preferred gpu-laptop-1; this
    # makes every keyed /v1 request agree with it.
    "gpu-laptop-2":    {"ram_gb": 32, "vram_gb": 8, "mem_bw_gbs": 256, "gpu_tflops": 20,
                   "cpu": "Ryzen AI 9 365", "gpu": "RTX 4070 Laptop (Ollama)",
                   "klass": "gpu", "rank": 2, "reserve": True},
    # A gaming desktop that lends the fleet its idle hours. It is the only
    # peer that can be online, healthy and still advertise nothing: its
    # gateway runs with LLMSTACK_AVAILABILITY_FILE set, and a watchdog closes
    # the catalogue the moment somebody sits down at it. See
    # hosts/gpu-desktop-1/README.md. mem_bw_gbs is the card's (128-bit GDDR6 at
    # 20 Gbps), which is what the router's spec-sheet tiebreak wants.
    "gpu-desktop-1":      {"ram_gb": 32, "vram_gb": 16, "mem_bw_gbs": 320, "gpu_tflops": 21,
                   "cpu": "Ryzen 5 9600X", "gpu": "RX 9060 XT 16 GB (Vulkan)",
                   "klass": "gpu", "reserve": True},
    "mac-desktop-1":    {"ram_gb": 16, "vram_gb": 11, "mem_bw_gbs": 68, "gpu_tflops": 5,
                   "cpu": "Apple M1", "gpu": "8-core (Metal)",
                   "klass": "small", "always_on": 3},
    "cpu-box-1":    {"ram_gb": 32, "vram_gb": 0, "mem_bw_gbs": 50, "gpu_tflops": 0,
                   "cpu": "CPU only", "gpu": "", "klass": "fallback", "always_on": 5},
    "server-1":   {"ram_gb": 18, "vram_gb": 0, "mem_bw_gbs": 40, "gpu_tflops": 4,
                   "cpu": "APU", "gpu": "680M", "klass": "small", "always_on": 2},
    "mini-pc-1":     {"ram_gb": 16, "vram_gb": 0, "mem_bw_gbs": 50, "gpu_tflops": 8,
                   "cpu": "Ryzen 8845HS", "gpu": "780M", "klass": "small", "always_on": 4},
    # The M4 Air. A laptop that sleeps, so it is deliberately NOT in the
    # always-on order: it takes worker traffic behind every box that is, and
    # falls out of the ranking whenever the lid closes. `vram_gb` is the Metal
    # ceiling its bootstrap installs (12 of 16 GiB), not the whole pool -- the
    # remaining 4 GB belong to the OS and to whoever is using the machine.
    "mac-laptop-2":     {"ram_gb": 16, "vram_gb": 12, "mem_bw_gbs": 120, "gpu_tflops": 9,
                   "cpu": "Apple M4", "gpu": "10-core (Metal)", "klass": "small",
                   "reserve": True},
    "hub":     {"ram_gb": 24, "vram_gb": 0, "mem_bw_gbs": 20, "gpu_tflops": 0,
                   "cpu": "hub / control plane", "gpu": "", "role": "hub", "klass": "hub"},
}

SPECS_PATH = Path(os.environ.get("LLMSTACK_SPECS", str(STATE / "specs.json")))

# The Configurations tab's routing policy, persisted in the settings table
# (key below) on the box whose dashboard saved it -- in practice the hub,
# which is where fleet routing happens. Shape:
#   {"order": ["mac-laptop-1", "apu-box-1", ...],      # drag-and-drop routing order;
#                                             # index becomes the spec `rank`
#    "reserve": {"gpu-desktop-2": true, ...}}     # per-box reserve overrides
# It is the third override layer for the spec sheet: DEFAULT_SPECS, then
# STATE/specs.json (hand-edited/deployed), then this (operator-edited in the
# dashboard). Later layers win, so a dragged order beats a deployed rank.
FLEET_ROUTING_KEY = "fleet_routing"

_specs_cache: dict[str, Any] = {"stamp": None, "specs": None, "gen": -1}
# Bumped by the fleet-config PUT so load_specs() re-merges without a file
# mtime change; module-level so tests can poke it too.
_fleet_routing_gen = 0


def get_fleet_routing() -> dict:
    """The stored Configurations-tab policy, or {} before the first save
    (and during startup, before the settings table exists)."""
    try:
        rows = db_query("SELECT value FROM settings WHERE key=?",
                        (FLEET_ROUTING_KEY,))
        return json.loads(rows[0]["value"]) if rows else {}
    except Exception:  # noqa: BLE001
        return {}


def set_fleet_routing(cfg: dict) -> None:
    global _fleet_routing_gen
    db_exec(
        "INSERT INTO settings(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (FLEET_ROUTING_KEY, json.dumps(cfg)),
    )
    _fleet_routing_gen += 1
    _specs_cache["specs"] = None  # force the next load_specs() to re-merge


def _specs_from_sheet() -> dict[str, dict]:
    """DEFAULT_SPECS with the specs.json file override applied -- the sheet
    BEFORE the dashboard's fleet_routing layer. Split out so the fleet-config
    PUT can diff an incoming reserve map against it and store only the
    entries an operator actually changed: a full snapshot would freeze
    today's shipped flags into the blob forever, and a later DEFAULT_SPECS
    or specs.json change would silently stop reaching the fleet."""
    specs = {k: dict(v) for k, v in DEFAULT_SPECS.items()}
    try:
        if SPECS_PATH.exists():
            for host, override in json.loads(SPECS_PATH.read_text()).items():
                if isinstance(override, dict):
                    specs.setdefault(host, {}).update(override)
    except Exception:  # noqa: BLE001
        pass
    return specs


def load_specs() -> dict[str, dict]:
    """Cached on the override file's mtime (and the dashboard policy's
    generation counter). public_alias() consults this for every host on every
    catalogue row now that a dual-boot machine resolves to its twin's box
    number, so re-reading and re-parsing the file per call was turning one
    page load into dozens of them."""
    try:
        stamp = SPECS_PATH.stat().st_mtime_ns if SPECS_PATH.exists() else 0
    except OSError:
        stamp = 0
    hit = _specs_cache["specs"]
    if (hit is not None and _specs_cache["stamp"] == stamp
            and _specs_cache["gen"] == _fleet_routing_gen):
        return hit
    specs = _specs_from_sheet()
    routing = get_fleet_routing()
    order = [str(h) for h in (routing.get("order") or []) if isinstance(h, str)]
    for idx, host in enumerate(order):
        # The dragged order IS the rank. It also overrides `always_on`
        # (host_tier() falls back to it when rank is absent), so a box the
        # operator placed is ranked by that placement alone.
        specs.setdefault(host, {})["rank"] = idx
    for host, flag in (routing.get("reserve") or {}).items():
        if isinstance(host, str) and isinstance(flag, bool):
            specs.setdefault(host, {})["reserve"] = flag
    _specs_cache.update(stamp=stamp, specs=specs, gen=_fleet_routing_gen)
    return specs


def snap_save(host: str, data: dict) -> None:
    db_exec(
        "INSERT INTO snapshots(host, ts, data) VALUES(?,?,?) "
        "ON CONFLICT(host) DO UPDATE SET ts=excluded.ts, data=excluded.data",
        (host, now(), json.dumps(data)),
    )


def snap_load(host: str) -> dict | None:
    rows = db_query("SELECT ts, data FROM snapshots WHERE host=?", (host,))
    if not rows:
        return None
    try:
        return {"ts": rows[0]["ts"], "data": json.loads(rows[0]["data"])}
    except json.JSONDecodeError:
        return None


def collapse_twins(hosts: list[dict], specs: dict) -> list[dict]:
    """One physical machine, one entry.

    A dual-boot box joins the fleet as one peer per OS, and only one of them
    can possibly be online at a time. Listed separately the fleet would show
    a permanently dark card beside a live one, and every pooled total -- RAM,
    VRAM, cores, TFLOPS -- would count that hardware twice on a page whose
    whole job is to say how much hardware there is.

    So the replacement takes the replaced host's place: when it is up it IS
    the machine, and when it is not the machine reverts to the host it
    replaces (which is where the last-known snapshot and the offline pill
    live). Driven by a `replaces` key in specs.json / DEFAULT_SPECS, so a
    dual-boot box needs configuration, not code. The fleet carried such a
    pair until 2026-08-27, when the gpu-laptop-2 laptop's Fedora half was retired;
    the mechanism stays because re-adding one is a specs.json edit."""
    by_name = {str(h.get("name") or ""): h for h in hosts}
    drop: set[str] = set()
    for h in hosts:
        name = str(h.get("name") or "")
        target = str((specs.get(name) or {}).get("replaces") or "")
        if not target or target == name or target not in by_name:
            continue
        if h.get("online"):
            drop.add(target)
            h["replaces"] = target
        else:
            drop.add(name)
    if not drop:
        return hosts
    return [h for h in hosts if str(h.get("name") or "") not in drop]


async def _gather_fleet_overview() -> dict:
    """Live status for every reachable host, the stored last-known snapshot
    for every offline one, and the static spec sheet for both.

    Factored out of the admin route so the public (sanitized) overview can
    share the same one telemetry gather rather than triggering a second
    round trip to every peer on every page load."""
    specs = load_specs()

    async def local() -> dict:
        swap_up, running = await swap_running()
        status = {
            "name": HOST_NAME,
            "api_url": PUBLIC_API_URL,
            "host": await asyncio.to_thread(host_status),
            "swap_up": swap_up,
            "models_running": running,
        }
        snap_save(HOST_NAME, status)
        return {
            "name": HOST_NAME,
            "kind": "self",
            "online": True,
            "routed": True,
            "api_url": PUBLIC_API_URL,
            "specs": specs.get(HOST_NAME, {}),
            "status": status,
        }

    async def remote(p: dict) -> dict:
        base = {
            "name": p["name"],
            "kind": "peer",
            "online": False,
            "routed": peer_routed(p),
            "api_url": p.get("api_url", ""),
            "specs": specs.get(p["name"], {}),
            "status": None,
        }
        try:
            async with httpx.AsyncClient(timeout=6.0) as c:
                r = await c.get(
                    p["url"].rstrip("/") + "/admin/api/status",
                    headers={"Authorization": "Bearer " + p.get("token", "")},
                )
            if r.status_code == 200:
                d = r.json()
                base.update(online=True, status=d)
                snap_save(p["name"], d)
                return base
            base["error"] = "HTTP " + str(r.status_code)
        except Exception as exc:  # noqa: BLE001
            base["error"] = str(exc)[:120]
        snap = snap_load(p["name"])
        if snap:
            base["status"] = snap["data"]
            base["last_seen"] = snap["ts"]
        return base

    peers = load_peers()
    results = await asyncio.gather(local(), *(remote(p) for p in peers))
    hosts = collapse_twins(list(results), specs)
    return {"generated_at": now(), "self": HOST_NAME, "hosts": hosts}


@app.get("/admin/api/fleet/overview")
async def fleet_overview(admin: dict = Depends(require_admin)) -> dict:
    """Everything the home page needs in one call.

    Declared ahead of the fleet catch-all below so 'overview' is never
    mistaken for a peer name.
    """
    return await _gather_fleet_overview()


@app.get("/admin/api/fleet/active")
async def fleet_active(admin: dict = Depends(require_admin)) -> dict:
    """Every job in flight anywhere in the fleet, in one call.

    Declared ahead of the fleet catch-all below so 'active' is never mistaken
    for a peer name.
    """

    async def remote(p: dict) -> dict:
        base: dict[str, Any] = {"name": p["name"], "online": False, "jobs": []}
        try:
            async with httpx.AsyncClient(timeout=6.0) as c:
                r = await c.get(
                    p["url"].rstrip("/") + "/admin/api/active",
                    headers={"Authorization": "Bearer " + p.get("token", "")},
                )
            if r.status_code == 404:
                # A peer still running the previous gateway. It is not idle --
                # it simply cannot say, and reporting zero jobs for it would be
                # a lie an operator would act on.
                base["error"] = "older gateway: no active-jobs API"
                return base
            if r.status_code != 200:
                base["error"] = "HTTP " + str(r.status_code)
                return base
            base.update(online=True, jobs=(r.json() or {}).get("jobs") or [])
        except Exception as exc:  # noqa: BLE001
            base["error"] = str(exc)[:120]
        return base

    peers = load_peers()
    results = await asyncio.gather(*(remote(p) for p in peers)) if peers else []
    hosts = [{"name": HOST_NAME, "online": True, "jobs": active_jobs()}, *results]
    jobs: list[dict] = []
    for h in hosts:
        for j in h["jobs"]:
            jobs.append({**j, "on": h["name"]})
    # A request this hub forwarded is listed twice: here, where the client
    # connection and therefore the abort lever is, and again on the peer that
    # is actually decoding it. Keep the hub's row -- it already names the peer
    # as the host, and closing the connection there stops both ends.
    known = {str(j["on"]) + "/" + str(j["id"]) for j in jobs}
    jobs = [j for j in jobs if not (j.get("origin") and j["origin"] in known)]
    return {
        "generated_at": now(),
        "self": HOST_NAME,
        "jobs": sorted(jobs, key=lambda j: -int(j["age_s"])),
        "hosts": [{k: v for k, v in h.items() if k != "jobs"} for h in hosts],
    }


@app.post("/admin/api/fleet/abort-all")
async def fleet_abort_all(admin: dict = Depends(require_admin)) -> dict:
    """Stop everything, everywhere. Declared ahead of the fleet catch-all
    below so 'abort-all' is never mistaken for a peer name."""

    async def remote(p: dict) -> dict:
        out: dict[str, Any] = {"host": p["name"], "aborted": [], "count": 0}
        try:
            async with httpx.AsyncClient(timeout=15.0) as c:
                r = await c.post(
                    p["url"].rstrip("/") + "/admin/api/active/abort-all",
                    headers={"Authorization": "Bearer " + p.get("token", "")},
                )
            if r.status_code == 200:
                return r.json()
            out["error"] = ("older gateway: no abort API" if r.status_code == 404
                            else "HTTP " + str(r.status_code))
        except Exception as exc:  # noqa: BLE001
            out["error"] = str(exc)[:120]
        return out

    peers = load_peers()
    results = await asyncio.gather(*(remote(p) for p in peers)) if peers else []
    hosts = [abort_all(), *results]
    return {"hosts": hosts, "count": sum(int(h.get("count", 0) or 0) for h in hosts)}


@app.get("/admin/api/served-models")
async def api_served_models(admin: dict = Depends(require_admin)) -> dict:
    """Everything this host can answer for on /v1 -- models.json plus, when
    LLMSTACK_MODELS_FROM_UPSTREAM is set, whatever the upstream reports.
    `running`, `capacity` and `ctx` are what the hub's orchestrator weighs:
    what is already resident here, how many decode slots each model would
    have, and the largest context window this box can give it.

    This is the one surface the availability gate closes. The hub builds its
    entire routing table from this answer (model_routes -> _peer_served), so a
    box whose owner is at the keyboard reports an empty catalogue and simply
    stops being chosen -- while /v1 below stays open for anything already in
    flight. See availability()."""
    av = availability()
    if not av["available"]:
        return {"models": [], "running": [], "capacity": {}, "ctx": {},
                "unavailable": av["reason"] or "reserved for its local user"}
    try:
        meta = await asyncio.wait_for(served_model_meta(), CTX_REPORT_BUDGET)
    except Exception:  # noqa: BLE001 -- telemetry, never worth failing the report
        meta = {}
    try:
        engine = str(engine_info().get("kind") or "")
    except Exception:  # noqa: BLE001
        engine = ""
    running = set(await upstream_running_ids())
    # A tag that is resident is resident under its canonical id too.
    for canon, tag in (await upstream_alias_pairs()).items():
        if tag in running:
            running.add(canon)
    return {
        "models": sorted(await served_model_ids()),
        "running": sorted(running),
        "capacity": local_capacity(),
        # The largest window this box will actually serve each model with.
        # A peer on an older gateway simply omits it, which the hub reads as
        # "unknown" rather than "zero" -- see catalogue_ctx().
        "ctx": await served_model_ctx(),
        # Weights size, whether they fit this box's GPU memory, MoE or not,
        # and where the file came from -- what the hub's host policy and the
        # warm-up button's download path are built on.
        "meta": meta,
        "engine": engine,
        # What this box keeps warm on its own account (models.json preload /
        # persistent). The hub's preload loop does not touch a box that
        # names something here other than the featured model.
        "warm": sorted(local_warm_ids()),
        # name -> canonical id, so the hub can tell whether this box's "fast"
        # is the same model as another box's "fast". Absent on an older
        # gateway, which the hub reads as "cannot check" rather than "agrees".
        "canonical": await served_canonical_map(),
    }


@app.get("/admin/api/routes")
async def api_routes(admin: dict = Depends(require_admin)) -> dict:
    """Which host answers each model id, as the /v1 proxy would decide,
    plus any name the fleet disagrees about (see alias_conflicts())."""
    routes = await model_routes(force=True)
    peer_can = _routes_cache.get("alias", {})
    local_can = await served_canonical_map()

    def canon(m: str, h: str) -> str:
        return (peer_can.get((h, m)) if h else local_can.get(m)) or m

    return {
        "self": HOST_NAME,
        # `canonical` is the model id behind the name and `alias` says the
        # name is not it -- on a peer as well as here, so a picker can list
        # each model once instead of every spelling of it.
        "models": sorted(
            (
                {"model": m, "host": h or HOST_NAME, "canonical": canon(m, h),
                 "alias": canon(m, h) != m}
                for m, h in routes.items()
            ),
            key=lambda r: (r["host"], r["model"]),
        ),
        # Empty is the normal answer and the good one. A non-empty list means
        # some name is not a single thing across the fleet, so ranking its
        # candidates against each other compares boxes that are not offering
        # the same model.
        "alias_conflicts": alias_conflicts(),
        # The mirror image: one model, several names, so the boxes holding it
        # are never each other's alternatives.
        "split_models": split_models(),
    }


@app.get("/admin/api/roles")
async def api_roles(admin: dict = Depends(require_admin)) -> dict:
    """Every fleet role: its ladder, the model each box would play it with,
    the order the scorer would try them in right now, and any box that still
    carries a models.json alias of the same name -- ignored, since the policy
    answers first, but worth seeing while the rows are still there."""
    await model_routes()
    cands = _routes_cache.get("cands", {})
    alias = _routes_cache.get("alias", {})
    out: dict[str, dict] = {}
    for role, ladder in FLEET_ROLES.items():
        pairs = role_pairs(role)
        ranked = await _score_host_model_pairs(pairs, fleet_role=role)
        out[role] = {
            "ladder": list(ladder),
            "picks": {(h or HOST_NAME): m for h, m in pairs},
            "order": [{"host": h or HOST_NAME, "model": m} for h, m in ranked],
            "local_rows": {(h or HOST_NAME): (alias.get((h, role)) or "?")
                           for h in cands.get(role, [])},
        }
    return {"roles": out}


@app.get("/admin/api/peers")
async def api_peers(admin: dict = Depends(require_admin)) -> list[dict]:
    return [
        {
            "name": p["name"],
            "url": p["url"],
            "api_url": p.get("api_url", ""),
            "has_token": bool(p.get("token")),
            # The killswitch: False means the hub refuses to route to this
            # peer, whatever it advertises. Absent on an older peers.json
            # means routed, which is why every reader goes through
            # peer_routed() rather than p["routed"].
            "routed": peer_routed(p),
        }
        for p in load_peers()
    ]


@app.put("/admin/api/peers")
async def api_peers_put(request: Request, admin: dict = Depends(require_admin)) -> dict:
    payload = await request.json()
    incoming = payload.get("peers")
    if not isinstance(incoming, list):
        raise HTTPException(400, "expected {peers: [...]}")
    existing = {p["name"]: p for p in load_peers()}
    clean = []
    for p in incoming:
        if not isinstance(p, dict):
            continue
        name = str(p.get("name", "")).strip()
        url = str(p.get("url", "")).strip().rstrip("/")
        if not SAFE_PEER.match(name) or not url.startswith("http"):
            raise HTTPException(400, "bad peer entry: " + name)
        token = str(p.get("token", "")).strip()
        if not token and name in existing:
            token = existing[name].get("token", "")  # keep the stored secret
        clean.append(
            {
                "name": name,
                "url": url,
                "token": token,
                "api_url": str(p.get("api_url", "")).strip(),
                # minted lazily by peer_inference_key(); preserved across edits
                "inference_key": existing.get(name, {}).get("inference_key", ""),
                # The peer editor's form never sends this field -- it is owned
                # by the killswitch toggle on the home page -- so an edit made
                # with a box killed must not silently re-route it, and a PUT
                # cannot kill a box either.
                "routed": peer_routed(existing[name]) if name in existing
                          else True,
            }
        )
    save_peers(clean)
    _routes_cache["t"] = 0.0  # peer set changed; rebuild the routing table
    return {"saved": len(clean)}


@app.put("/admin/api/peers/{name}/routed")
async def api_peer_routed(name: str, request: Request,
                          admin: dict = Depends(require_admin)) -> dict:
    """The killswitch, set from the Configurations tab (a box's detail
    modal, and the drag into "Not in use" via the fleet-config PUT below).

    The ONLY direct writer of 'routed' besides that PUT -- the peers form's
    PUT preserves it on purpose, so a box a human has killed stays killed
    across any other edit. The effect is immediate: the routing table is
    invalidated here, and model_routes() re-reads the flag on every refresh
    (routeable_peers())."""
    payload = await request.json()
    routed = payload.get("routed")
    if not isinstance(routed, bool):
        raise HTTPException(400, "expected {routed: true|false}")
    if not SAFE_PEER.match(name):
        raise HTTPException(400, "bad peer name")
    peers = load_peers()
    if not any(p["name"] == name for p in peers):
        raise HTTPException(404, "unknown peer: " + name)
    for p in peers:
        if p["name"] == name:
            p["routed"] = routed
    save_peers(peers)
    _routes_cache["t"] = 0.0
    # And the remembered context ceilings, which are cached for 30s of their
    # own: a killed box must stop advertising a window the hub will not route
    # to before the next catalogue read, not half a minute after it.
    _known_ctx_cache["t"] = 0.0
    return {"name": name, "routed": routed}


def _fleet_config_state() -> dict:
    """What the Configurations tab renders: every box this gateway can route
    to (its peers plus itself), each with its effective routing facts after
    all three spec layers have merged, split into the routing order and the
    boxes taken out of use."""
    specs = load_specs()
    peers = {p["name"]: p for p in load_peers()}
    names = sorted(set(peers) | {HOST_NAME})
    hosts = []
    for name in names:
        spec = specs.get(name, {})
        peer = peers.get(name)
        rank = spec.get("rank")
        if rank is None:
            rank = spec.get("always_on")
        hosts.append({
            "name": name,
            "self": name == HOST_NAME,
            "klass": host_class(name),
            "reserve": bool(spec.get("reserve")),
            "rank": rank,
            "routed": peer_routed(peer) if peer else True,
            "cpu": str(spec.get("cpu") or ""),
            "gpu": str(spec.get("gpu") or ""),
            "ram_gb": spec.get("ram_gb"),
            "vram_gb": spec.get("vram_gb"),
        })

    def order_key(h: dict) -> tuple:
        # This box itself is sorted last unconditionally: the scorer already
        # treats the hub as the last resort (tier 6) whatever its rank says,
        # and the tab renders the self row pinned at the bottom -- an
        # alphabetical accident putting "hub" mid-list would make the
        # pinned row lie and strand every drag-to-bottom above it.
        r = h["rank"]
        return (1 if h["self"] else 0, 0 if r is not None else 1,
                r if r is not None else 0, h["name"])

    in_use = sorted((h for h in hosts if h["routed"]), key=order_key)
    return {
        "hosts": {h["name"]: h for h in hosts},
        "order": [h["name"] for h in in_use],
        "not_in_use": sorted(h["name"] for h in hosts if not h["routed"]),
        "saved": get_fleet_routing(),
    }


@app.get("/admin/api/fleet-config")
async def api_fleet_config(admin: dict = Depends(require_admin)) -> dict:
    return _fleet_config_state()


@app.put("/admin/api/fleet-config")
async def api_fleet_config_put(request: Request,
                               admin: dict = Depends(require_admin)) -> dict:
    """The Configurations tab's save: a routing order (drag-and-drop), the
    boxes taken out of use, and per-box reserve flags.

    The order lands in the settings table (see FLEET_ROUTING_KEY) and becomes
    each box's spec `rank`; membership of `not_in_use` flips the same
    peers.json `routed` flag the killswitch toggle used to. One PUT, one
    consistent state -- the two mechanisms stay what they were, this is just
    one form that drives both."""
    payload = await request.json()
    order = payload.get("order")
    not_in_use = payload.get("not_in_use")
    reserve = payload.get("reserve")
    if not isinstance(order, list) or not all(isinstance(n, str) for n in order):
        raise HTTPException(400, "expected {order: [host, ...]}")
    if not_in_use is None:
        not_in_use = []
    if not isinstance(not_in_use, list) or not all(
            isinstance(n, str) for n in not_in_use):
        raise HTTPException(400, "expected {not_in_use: [host, ...]}")
    if reserve is None:
        reserve = {}
    if not isinstance(reserve, dict):
        raise HTTPException(400, "expected {reserve: {host: bool}}")
    peers = load_peers()
    known = {p["name"] for p in peers} | {HOST_NAME}
    for name in [*order, *not_in_use, *reserve]:
        if not SAFE_PEER.match(name or ""):
            raise HTTPException(400, "bad host name: " + str(name)[:40])
        if name not in known:
            raise HTTPException(400, "unknown host: " + name)
    if HOST_NAME in not_in_use:
        raise HTTPException(
            400, "this box cannot be taken out of its own routing")
    dupes = set(order) & set(not_in_use)
    if dupes:
        raise HTTPException(
            400, "in both lists: " + ", ".join(sorted(dupes)))
    # Store only the reserve entries that DIFFER from the spec sheet as
    # shipped/deployed. The dashboard sends the full map on every save, and
    # persisting it wholesale would pin today's flags forever -- a box
    # un-reserved in a later deploy would stay reserved because a routine
    # reorder had re-saved the stale value.
    sheet = _specs_from_sheet()
    set_fleet_routing({
        "order": order,
        "reserve": {k: bool(v) for k, v in reserve.items()
                    if bool(v) != bool(sheet.get(k, {}).get("reserve"))},
    })
    changed = False
    for p in peers:
        want = p["name"] not in not_in_use
        if peer_routed(p) != want:
            p["routed"] = want
            changed = True
    if changed:
        save_peers(peers)
    _routes_cache["t"] = 0.0
    _known_ctx_cache["t"] = 0.0  # same reasoning as the killswitch PUT above
    return _fleet_config_state()


@app.api_route(
    "/admin/api/fleet/{peer}/{subpath:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def fleet_proxy(peer: str, subpath: str, request: Request):
    require_admin(request)
    target = next((p for p in load_peers() if p["name"] == peer), None)
    if not target:
        raise HTTPException(404, "unknown peer: " + peer)
    url = target["url"].rstrip("/") + "/admin/api/" + subpath
    headers = {"Authorization": "Bearer " + target.get("token", "")}
    if request.headers.get("content-type"):
        headers["content-type"] = request.headers["content-type"]
    body = await request.body()
    # A peer that is merely busy drops the odd connection, and the dashboard
    # polls hard enough that "the odd one" shows up as a 502 in the browser
    # every few minutes. Retry a read once -- safe because it changes nothing --
    # before calling the peer unreachable.
    attempts = 2 if request.method in ("GET", "HEAD") else 1
    # Staging GPU memory rebuilds every kernel's initramfs on the peer, which
    # is minutes of work the dashboard has to be allowed to wait for.
    read_timeout = 900.0 if (subpath == "gpu" and request.method == "POST") else 120.0
    for attempt in range(1, attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=read_timeout)) as c:
                r = await c.request(
                    request.method,
                    url,
                    headers=headers,
                    content=body or None,
                    params=dict(request.query_params),
                )
            break
        except Exception as exc:  # noqa: BLE001
            if attempt < attempts:
                await asyncio.sleep(0.25)
                continue
            # Named, because the browser only ever sees "502" -- and behind
            # Cloudflare not even the detail, just its own error page.
            detail = type(exc).__name__ + ": " + str(exc)
            log.warning("fleet proxy %s %s failed -- %s", peer, subpath, detail)
            raise HTTPException(502, "peer '" + peer + "' unreachable: " + detail)
    drop = {"content-length", "transfer-encoding", "connection", "content-encoding"}
    return Response_bytes(
        r.content,
        r.status_code,
        {k: v for k, v in r.headers.items() if k.lower() not in drop},
    )


# ------------------------------- optimize ----------------------------------

_bench_proc_lock = threading.Lock()
_bench_running: dict[str, Any] = {"active": False}


def _bench_worker(bench_id: int, rec: dict, overrides: dict, job: dict) -> None:
    try:
        # Free the GPU first: a resident model would make every number a lie.
        try:
            httpx.get(UPSTREAM + "/api/models/unload", timeout=30.0)
        except Exception:  # noqa: BLE001
            pass

        cmd = [LLAMA_BENCH, "-m", str(rec["path"]), "-o", "json", "-r", "1",
               "-p", "512", "-n", "128"]
        ngl = int(overrides.get("ngl", rec.get("ngl", 99)) or 0)
        cmd += ["-ngl", str(ngl)]
        ncm = int(overrides.get("n_cpu_moe", rec.get("n_cpu_moe", 0)) or 0)
        if ncm > 0:
            cmd += ["--n-cpu-moe", str(ncm)]
        thr = int(overrides.get("threads", rec.get("threads", 0)) or 0)
        if thr > 0:
            cmd += ["-t", str(thr)]
        fa = overrides.get("flash_attn", rec.get("flash_attn", True))
        cmd += ["-fa", "1" if fa else "0"]

        if job["aborted"]:
            # Aborted while it was still queued behind the GPU unload.
            db_exec("UPDATE bench SET status='cancelled', message=? WHERE id=?",
                    ("aborted before it started", bench_id))
            return
        db_exec("UPDATE bench SET status='running' WHERE id=?", (bench_id,))
        # Popen rather than run(): the abort button needs a handle on the
        # process. A wedged llama-bench used to hold the GPU for the full hour
        # of that timeout with nothing able to stop it.
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        job["proc"] = proc
        try:
            out, err = proc.communicate(timeout=3600)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise
        if job["aborted"]:
            db_exec("UPDATE bench SET status='cancelled', message=? WHERE id=?",
                    ("aborted from the dashboard", bench_id))
            return
        if proc.returncode != 0:
            db_exec(
                "UPDATE bench SET status='error', message=? WHERE id=?",
                ((err or out)[-800:], bench_id),
            )
            return
        pp = tg = None
        try:
            for entry in json.loads(out):
                if entry.get("n_gen", 0) == 0 and entry.get("n_prompt"):
                    pp = float(entry.get("avg_ts") or 0)
                elif entry.get("n_prompt", 0) == 0 and entry.get("n_gen"):
                    tg = float(entry.get("avg_ts") or 0)
        except json.JSONDecodeError:
            pass
        db_exec(
            "UPDATE bench SET status='done', pp_tps=?, tg_tps=?, raw=? WHERE id=?",
            (pp, tg, out[-4000:], bench_id),
        )
    except subprocess.TimeoutExpired:
        db_exec(
            "UPDATE bench SET status='error', message='timed out after 1h' WHERE id=?",
            (bench_id,),
        )
    except Exception as exc:  # noqa: BLE001
        db_exec(
            "UPDATE bench SET status='error', message=? WHERE id=?",
            (str(exc)[:800], bench_id),
        )
    finally:
        _job_close(job)
        _bench_running["active"] = False


@app.post("/admin/api/bench")
async def api_bench(request: Request, admin: dict = Depends(require_admin)) -> dict:
    payload = await request.json()
    mid = str(payload.get("model_id", "")).strip()
    rec = next((m for m in load_models() if m.get("id") == mid), None)
    if not rec:
        raise HTTPException(404, "model not configured: " + mid)
    if not Path(str(rec["path"])).exists():
        raise HTTPException(404, "gguf missing on disk: " + str(rec["path"]))
    if not Path(LLAMA_BENCH).exists():
        raise HTTPException(503, "llama-bench not installed at " + LLAMA_BENCH)
    overrides = payload.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise HTTPException(400, "overrides must be an object")
    with _bench_proc_lock:
        if _bench_running["active"]:
            raise HTTPException(409, "a benchmark is already running")
        _bench_running["active"] = True
    bench_id = db_exec(
        "INSERT INTO bench(created_at,model_id,gguf,params,status) VALUES (?,?,?,?,?)",
        (now(), mid, str(rec["path"]), json.dumps(overrides), "queued"),
    )
    # Listed before the thread starts: the first thing a benchmark does is
    # unload the GPU, and an operator who wants that stopped wants it stopped
    # then, not once llama-bench is finally resident.
    job = _job_open("benchmark", what=mid, detail="llama-bench pp512 / tg128")
    threading.Thread(
        target=_bench_worker, args=(bench_id, rec, overrides, job), daemon=True
    ).start()
    return {"bench_id": bench_id, "job_id": job["id"]}


@app.get("/admin/api/bench")
async def api_bench_list(
    model_id: str = "",
    limit: int = 25,
    offset: int = 0,
    archived: bool = False,
    admin: dict = Depends(require_admin),
) -> dict:
    """One page of benchmark history, newest first. Optionally narrowed to one
    model -- comparing a tune against its own past runs is the whole point of
    keeping them."""
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    view = "archived_at IS NOT NULL" if archived else "archived_at IS NULL"
    clause = " WHERE " + view
    args: list[Any] = []
    if model_id:
        clause += " AND model_id=?"
        args.append(model_id)
    total = db_query("SELECT COUNT(*) n FROM bench" + clause, args)[0]["n"]
    rows = db_query(
        "SELECT id,created_at,model_id,params,status,pp_tps,tg_tps,message,"
        "archived_at FROM bench" + clause + " ORDER BY id DESC LIMIT ? OFFSET ?",
        (*args, limit, offset),
    )
    # Every model this view has results for, so the filter offers exactly the
    # choices that would return something -- unfiltered by model_id itself.
    models = [
        r["model_id"]
        for r in db_query(
            "SELECT DISTINCT model_id FROM bench WHERE " + view +
            " ORDER BY model_id"
        )
    ]
    return {"items": rows, "total": int(total), "limit": limit,
            "offset": offset, "models": models}


@app.delete("/admin/api/bench/{bench_id}")
async def api_bench_del(bench_id: int, admin: dict = Depends(require_admin)) -> dict:
    """Archive one benchmark row. A result you are throwing away today is the
    baseline you wish you had kept the next time a tune regresses, so it goes
    to the archive rather than straight out of the database."""
    if not db_update(
        "UPDATE bench SET archived_at=? WHERE id=? AND archived_at IS NULL",
        (now(), bench_id),
    ):
        raise HTTPException(404, "no such live benchmark")
    return {"archived": bench_id}


@app.post("/admin/api/bench/{bench_id}/restore")
async def api_bench_restore(
    bench_id: int, admin: dict = Depends(require_admin)
) -> dict:
    if not db_update(
        "UPDATE bench SET archived_at=NULL WHERE id=? AND archived_at IS NOT NULL",
        (bench_id,),
    ):
        raise HTTPException(404, "no such archived benchmark")
    return {"restored": bench_id}


@app.delete("/admin/api/bench/{bench_id}/purge")
async def api_bench_purge(
    bench_id: int, admin: dict = Depends(require_admin)
) -> dict:
    if not db_query(
        "SELECT id FROM bench WHERE id=? AND archived_at IS NOT NULL",
        (bench_id,),
    ):
        raise HTTPException(409, "archive it first -- live results are not "
                                 "deleted in one step")
    db_exec("DELETE FROM bench WHERE id=?", (bench_id,))
    return {"deleted": bench_id}


# --------------------------------------------------------------------------
# GPU memory split: how much system RAM the GPU may map (GTT on amdgpu, the
# wired limit on Apple silicon)
#
# On an APU the BIOS UMA carve-out (VRAM) is firmware-only, but GTT -- the
# window the GPU can additionally map out of system RAM -- is a kernel
# parameter. The helper (gateway/bin/llmstack-gpuconf) stages it on the KERNEL
# COMMAND LINE for every installed kernel and leaves a receipt; this reads the
# receipt back beside the live sysfs number, so "staged", "active" and
# "reboot required" are three facts, not one guess. The old helper wrote
# /etc/modprobe.d and rebuilt one kernel's initramfs with every error hidden,
# which is how a staged value could survive a reboot unapplied with nothing on
# the dashboard to say so.
# --------------------------------------------------------------------------

GTT_RECEIPTS = (Path("/etc/llmstack/gtt.json"), Path("/usr/local/etc/llmstack/gtt.json"))
GIB = 1024 ** 3
NSENTER = "/usr/bin/nsenter"


def gpuconf_argv(arg: str) -> list[str]:
    """How to invoke the GPU-memory helper as root.

    Not simply `sudo -n <helper>`: this gateway's own systemd unit sets
    ProtectSystem=full, and sudo does NOT escape a unit's mount namespace, so
    the helper would inherit a read-only /etc and /boot and fail to write the
    bootloader config -- while the very same command works by hand over ssh.
    That mismatch is what made the Optimize tab's GTT control look broken for
    so long.

    Widening the sandbox does not fix it either: ReadWritePaths= can lift a
    read-only SUBDIRECTORY (/etc/default) but not a read-only mount point
    (/boot), so grub-mkconfig still cannot write grub.cfg.

    So the helper has to run OUTSIDE this unit's mount namespace. `nsenter -t 1
    -m` is the direct way to say that: join PID 1's mount namespace, where /etc
    and /boot are writable, and exec the helper there. stdout, stderr and the
    exit status (2 = bad input, 3 = not root) come back over ordinary pipes.

    This replaces `systemd-run --pipe --wait --collect`, which did escape the
    namespace but is NOT portable across the fleet: --pipe passes the three
    standard file descriptors to PID 1 inside the StartTransientUnit D-Bus
    message, and on Fedora (dbus-broker) that message gets the connection reset
    -- "Failed to start transient service unit: Connection reset by peer" --
    while the identical call on apu-box-1's Ubuntu succeeds. Verified on gpu-laptop-1:
    --pipe fails, --wait --collect without it succeeds, so the fault is the
    descriptor passing and not the transient unit. --scope is not a substitute
    either: a scope runs as a child of the CALLER, so it inherits the very
    namespace we are trying to leave.

    nsenter needs CAP_SYS_ADMIN, which is in the unit's capability bounding set
    on every fleet Linux box (checked on gpu-laptop-1 and apu-box-1). Without systemd
    (a Mac), plain sudo is already correct."""
    if platform.system() == "Linux" and Path(NSENTER).exists():
        return ["sudo", "-n", NSENTER, "-t", "1", "-m", "--", GPUCONF_HELPER, arg]
    return ["sudo", "-n", GPUCONF_HELPER, arg]


def gtt_targets(ram_bytes: int) -> tuple[int, int]:
    """(max_gib, suggested_gib) for a box with this much RAM -- the helper's
    own arithmetic, repeated here so the dashboard can show the numbers
    before anything is staged. The OS keeps max(3 GiB, 15%); the GPU is never
    offered more than RAM minus 3 GiB. 'Suggested' is deliberately aggressive:
    GTT is a ceiling on what the GPU MAY map, not a reservation, and the boxes
    this runs on serve models, not desktops."""
    if ram_bytes <= 0:
        return 0, 0
    max_gib = max(0, int((ram_bytes - 3 * GIB) // GIB))
    reserve = max(3 * GIB, int(ram_bytes * 0.15))
    suggested = int((ram_bytes - reserve) // GIB)
    return max_gib, max(0, min(suggested, max_gib))


_grant_cache: dict[str, Any] = {"t": 0.0, "ok": {}}


def sudo_allows(*cmd: str) -> bool:
    """Whether sudo would run `cmd` for this user without a password -- the
    check that turns 'helper installed' into 'helper usable', and 'systemctl
    exists' into 'the dashboard can reboot this box'. Cached a minute."""
    key = " ".join(cmd)
    hit = _grant_cache["ok"].get(key)
    if hit is not None and time.time() - _grant_cache["t"] < 60:
        return bool(hit)
    ok = False
    try:
        p = subprocess.run(["sudo", "-n", "-l", *cmd], capture_output=True,
                           text=True, timeout=10)
        ok = p.returncode == 0
    except Exception:  # noqa: BLE001 -- no sudo at all is "no"
        ok = False
    if time.time() - _grant_cache["t"] >= 60:
        _grant_cache["ok"] = {}
        _grant_cache["t"] = time.time()
    _grant_cache["ok"][key] = ok
    return ok


def _gtt_receipt() -> dict:
    for p in GTT_RECEIPTS:
        if p.exists():
            try:
                d = json.loads(p.read_text())
                return d if isinstance(d, dict) else {}
            except (OSError, ValueError):
                pass
    return {}


def _gtt_info() -> dict:
    """Live, staged and applied GPU-memory split for this box, plus what the
    dashboard needs to offer a sensible number: the RAM, the ceiling, and
    the aggressive suggestion. Every field is present on every platform so
    the UI never has to guess at a shape."""
    sysname = platform.system()
    live: dict[str, Any] = {"gtt_total": None, "vram_total": None}
    out: dict[str, Any] = {
        "platform": "none", "supported": False, "reason": None,
        "helper_installed": Path(GPUCONF_HELPER).exists(), "grant_ok": False,
        "live": live, "live_gib": None, "ram_gib": None, "max_gib": None,
        "suggested_gib": None, "staged_gib": None, "mechanism": None,
        "active": False, "reboot_required": False, "warnings": [],
        "staged_at": None, "kernels": [], "running_kernel": None,
        # The shape the previous dashboard read; kept so a hub on this code
        # and a peer on the old one (or the reverse) still agree.
        "staged_conf": "", "staged_gtt_mib": None,
    }
    try:
        ram = int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001
        ram = 0
    if ram:
        out["ram_gib"] = round(ram / GIB, 1)
        out["max_gib"], out["suggested_gib"] = gtt_targets(ram)
    cmdline_tokens: set[str] = set()
    mac_limit_mb = 0
    if sysname == "Linux":
        for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
            dev = card / "device"
            if (dev / "mem_info_gtt_total").exists():
                live["gtt_total"] = _read_int(dev / "mem_info_gtt_total")
                live["vram_total"] = _read_int(dev / "mem_info_vram_total")
                break
        if live["gtt_total"]:
            out.update(platform="linux-amdgpu", supported=True,
                       live_gib=round(live["gtt_total"] / GIB, 2))
        else:
            out["reason"] = "no amdgpu device on this box"
        conf = Path("/etc/modprobe.d/llmstack-gpu.conf")
        if conf.exists():
            try:
                out["staged_conf"] = conf.read_text()
            except OSError:
                pass
        try:
            cmdline_tokens = set(Path("/proc/cmdline").read_text().split())
        except OSError:
            pass
        out["running_kernel"] = platform.release()
    elif sysname == "Darwin":
        out.update(platform="macos", supported=True)
        try:
            p = subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"],
                               capture_output=True, text=True, timeout=5)
            mac_limit_mb = int((p.stdout or "0").strip() or 0)
        except Exception:  # noqa: BLE001
            mac_limit_mb = 0
        out["live_gib"] = round(mac_limit_mb / 1024, 2) if mac_limit_mb else None
    elif sysname == "Windows":
        out.update(platform="windows",
                   reason="not applicable on Windows (WDDM manages shared GPU "
                          "memory automatically)")
    rec = _gtt_receipt()
    if rec:
        try:
            out["staged_gib"] = int(rec["gib"]) if rec.get("gib") is not None else None
        except (TypeError, ValueError):
            out["staged_gib"] = None
        out["mechanism"] = rec.get("mechanism")
        out["warnings"] = [str(w) for w in (rec.get("warnings") or [])]
        out["staged_at"] = rec.get("staged_at")
        out["kernels"] = [str(k) for k in (rec.get("kernels") or [])]
        out["running_kernel"] = rec.get("running_kernel") or out["running_kernel"]
        if out["staged_gib"] is not None:
            out["staged_gtt_mib"] = out["staged_gib"] * 1024
        token = str(rec.get("cmdline_token") or "")
        if sysname == "Linux":
            out["active"] = bool(token) and token in cmdline_tokens
            out["reboot_required"] = out["staged_gib"] is not None and not out["active"]
        elif sysname == "Darwin":
            out["active"] = (out["staged_gib"] is not None and mac_limit_mb > 0
                             and abs(mac_limit_mb - out["staged_gib"] * 1024) < 64)
            out["reboot_required"] = False
    elif sysname == "Linux" and out["staged_conf"]:
        # A box staged by the old helper: modprobe.d only. Read it the old
        # way so the dashboard still says what was asked for, and call it
        # pending unless the live number already agrees.
        m = re.search(r"gttsize=(\d+)", out["staged_conf"])
        pages = re.search(r"pages_limit=(\d+)", out["staged_conf"])
        staged_mib = None
        if m:
            staged_mib = int(m.group(1))
        elif pages:
            staged_mib = int(pages.group(1)) * 4096 // (1024 * 1024)
        if staged_mib:
            out.update(staged_gtt_mib=staged_mib, staged_gib=staged_mib // 1024,
                       mechanism="modprobe")
            pending = bool(live["gtt_total"]) and abs(
                staged_mib * 1024 * 1024 - int(live["gtt_total"] or 0)) > 2 * GIB
            out.update(reboot_required=pending, active=not pending)
    if out["supported"]:
        if not out["helper_installed"]:
            out["reason"] = "helper not installed: " + GPUCONF_HELPER
        else:
            out["grant_ok"] = sudo_allows(*gpuconf_argv("status")[2:])
            if not out["grant_ok"]:
                out["reason"] = ("needs a sudoers grant for the helper: run "
                                 "hosts/linux/grants.sh on this box"
                                 if sysname == "Linux" else
                                 "needs passwordless sudo for the helper on this Mac")
    return out


@app.get("/admin/api/gpu")
async def api_gpu(admin: dict = Depends(require_admin)) -> dict:
    return {"cards": amdgpu_stats(), **_gtt_info()}


@app.post("/admin/api/gpu")
async def api_gpu_set(request: Request, admin: dict = Depends(require_admin)) -> dict:
    payload = await request.json()
    want = payload.get("gtt_gb")
    info = _gtt_info()
    if isinstance(want, str) and want.strip().lower() == "auto":
        arg = "auto"
    else:
        try:
            gtt_gb = int(want)
        except (TypeError, ValueError):
            raise HTTPException(400, 'gtt_gb must be a whole number of GiB, or "auto"')
        ceiling = int(info.get("max_gib") or 0) or 120
        if not 4 <= gtt_gb <= ceiling:
            raise HTTPException(400, "gtt_gb out of range (4..%d GiB on this box)" % ceiling)
        arg = str(gtt_gb)
    if not info["supported"]:
        raise HTTPException(501, info.get("reason")
                            or "GPU memory staging is not applicable on this platform")
    if not info["helper_installed"]:
        raise HTTPException(503, "helper not installed: " + GPUCONF_HELPER)
    # Off the event loop: rebuilding every kernel's initramfs can take minutes,
    # and nothing else on this box should stall behind it.
    p = await asyncio.to_thread(
        subprocess.run, gpuconf_argv(arg),
        capture_output=True, text=True, timeout=900)
    _grant_cache["t"] = 0.0
    err = (p.stderr or p.stdout).strip()[:600]
    if p.returncode == 2:
        raise HTTPException(400, err or "rejected by the helper")
    if p.returncode == 3:
        raise HTTPException(501, err or "the helper needs root here")
    if p.returncode != 0:
        raise HTTPException(500, err or ("helper exited " + str(p.returncode)))
    receipt: dict = {}
    try:
        lines = [ln for ln in (p.stdout or "").splitlines() if ln.strip()]
        receipt = json.loads(lines[-1]) if lines else {}
    except (ValueError, IndexError):
        receipt = {}
    info = _gtt_info()
    staged = receipt.get("gib") if receipt.get("gib") is not None else info.get("staged_gib")
    summary = "staged " + str(staged) + " GiB via " + str(info.get("mechanism") or "?")
    summary += " -- reboot to apply" if info.get("reboot_required") else " -- active"
    return {"staged": staged, "out": summary, "receipt": receipt, **info}


# --------------------------------------------------------------------------
# reboot: the one lever a staged GTT change still needed a shell for
# --------------------------------------------------------------------------

def reboot_support() -> dict:
    """Whether, and how, this box can be rebooted from the dashboard."""
    sysname = platform.system()
    if sysname == "Linux":
        ok = sudo_allows("/usr/bin/systemctl", "reboot")
        return {"supported": ok, "method": "systemd" if ok else None,
                "reason": None if ok else ("needs a sudoers grant: run "
                                           "hosts/linux/grants.sh on this box"),
                "host": HOST_NAME}
    if sysname == "Windows":
        # The gateway is a SYSTEM scheduled task; shutdown needs nothing more.
        return {"supported": True, "method": "windows", "reason": None, "host": HOST_NAME}
    if sysname == "Darwin":
        if sudo_allows("/sbin/shutdown"):
            return {"supported": True, "method": "macos-sudo", "reason": None,
                    "host": HOST_NAME}
        return {"supported": True, "method": "macos-osascript",
                "reason": "via System Events -- needs a logged-in desktop session, "
                          "and an app with unsaved work can hold it up",
                "host": HOST_NAME}
    return {"supported": False, "method": None,
            "reason": "no reboot method for " + sysname, "host": HOST_NAME}


@app.get("/admin/api/reboot")
async def api_reboot_info(admin: dict = Depends(require_admin)) -> dict:
    return reboot_support()


@app.post("/admin/api/reboot")
async def api_reboot(request: Request, admin: dict = Depends(require_admin)) -> dict:
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        payload = {}
    if not isinstance(payload, dict) or not payload.get("confirm"):
        raise HTTPException(400, 'send {"confirm": true}')
    sup = reboot_support()
    if not sup["supported"]:
        raise HTTPException(501, str(sup.get("reason") or "reboot not supported here"))
    method = sup["method"]
    if method == "systemd":
        cmd = ["sh", "-c", "sleep 3; exec sudo -n /usr/bin/systemctl reboot"]
    elif method == "windows":
        cmd = ["shutdown", "/r", "/t", "5", "/c", "llmstack dashboard reboot"]
    elif method == "macos-sudo":
        cmd = ["sh", "-c", "sleep 3; exec sudo -n /sbin/shutdown -r now"]
    else:
        cmd = ["sh", "-c", "sleep 3; exec osascript -e "
               "'tell application \"System Events\" to restart'"]
    log.warning("reboot requested from the dashboard (%s)", method)
    try:
        # Its own session, so the gateway being stopped by the shutdown does
        # not take the sleep-then-reboot down with it.
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=(platform.system() != "Windows"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, "could not start the reboot: " + str(exc))
    return {"ok": True, "host": HOST_NAME, "method": method,
            "message": "rebooting " + HOST_NAME + " in 3 s"}


# ------------------------------- dashboard --------------------------------


@app.get("/")
async def root_index(request: Request):
    return await admin_index(request)


@app.get("/admin")
@app.get("/admin/")
async def admin_index(request: Request):
    require_admin(request)
    idx = STATIC_DIR / "index.html"
    if not idx.exists():
        return JSONResponse({"error": "dashboard not installed"}, status_code=500)
    return FileResponse(idx, headers={"cache-control": "no-store"})


@app.get("/admin/static/{name}")
async def admin_static(name: str, request: Request):
    require_admin(request)
    if "/" in name or ".." in name:
        raise HTTPException(400, "bad name")
    p = STATIC_DIR / name
    if not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(p)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("LLMSTACK_BIND", "127.0.0.1"),
        port=int(os.environ.get("LLMSTACK_PORT", "8080")),
        log_level="info",
    )
