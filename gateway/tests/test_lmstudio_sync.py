"""LM Studio <-> fleet model sync.

Every assertion here is against a real filesystem, not a mock: the whole
subsystem is about inodes, link types and what a directory layout means to
another program, and none of that survives being faked. The GGUFs are a few
bytes each -- nothing reads their contents except the edge digest, which is
happy with any bytes at all.

Run: python -m pytest gateway/tests/test_lmstudio_sync.py -q
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import app as gw


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _gguf(path: Path, marker: bytes = b"a", size: int = 4096) -> Path:
    """A file that is a GGUF as far as this subsystem is concerned: it ends in
    .gguf, it is non-empty, and two of them differ in their first bytes."""
    return _write(path, b"GGUF" + marker * (size - 4))


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A box with an LM Studio store and a fleet models dir, wired the way a
    real host is: settings.json names the models root, and MODELS_DIR is
    somewhere else entirely."""
    home = tmp_path / "lmstudio-home"
    lms = tmp_path / "lmstudio-models"
    fleet = tmp_path / "fleet-models"
    for d in (home, lms, fleet):
        d.mkdir(parents=True, exist_ok=True)
    (home / "settings.json").write_text(json.dumps({"downloadsFolder": str(lms)}))

    monkeypatch.setattr(gw, "LMSTUDIO_HOME_ENV", str(home))
    monkeypatch.setattr(gw, "MODELS_DIR", fleet)
    monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")
    monkeypatch.setattr(gw, "SWAP_CONFIG", tmp_path / "llama-swap.yaml")
    monkeypatch.setattr(gw, "UPSTREAM_MODELS", False)
    monkeypatch.setattr(gw, "_lmstudio_restart_swap", lambda: "test: not restarted")
    gw._lmstudio_root_cache.update(t=0.0, root=None, home=None)
    gw._lmstudio_last.clear()
    gw.db_exec("DELETE FROM settings WHERE key='lmstudio'")
    gw.save_models([])
    yield type("Store", (), {"home": home, "lms": lms, "fleet": fleet,
                             "tmp": tmp_path})


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------


def test_root_comes_from_settings_json_not_the_default_path(store):
    """Three fleet boxes have moved downloadsFolder off ~/.lmstudio/models.
    Assuming the default would have found an empty directory on all three."""
    assert gw.lmstudio_root() == store.lms


def test_absent_lmstudio_is_inert(tmp_path, monkeypatch):
    """The same app.py ships to the Linux boxes, which have no LM Studio. A
    pass there must be a no-op, not an error."""
    monkeypatch.setattr(gw, "LMSTUDIO_HOME_ENV", str(tmp_path / "nothing-here"))
    monkeypatch.setattr(gw, "_lmstudio_homes", lambda: [tmp_path / "nothing-here"])
    gw._lmstudio_root_cache.update(t=0.0, root=None, home=None)
    plan = gw.lmstudio_plan()
    assert plan["available"] is False
    assert plan["actions"] == []
    assert "not installed" in plan["reason"]


# --------------------------------------------------------------------------
# scanning
# --------------------------------------------------------------------------


def test_mmproj_is_paired_by_directory_never_by_name(store):
    """Projector names are uploader-controlled: `mmproj-F32.gguf` on one repo,
    `mmproj-gemma-4-31B-it-QAT-BF16.gguf` on the next. Co-location is the only
    reliable signal."""
    d = store.lms / "Jackrong" / "Qwopus-GGUF"
    _gguf(d / "Qwopus-Q4_K_M.gguf", b"m", 8192)
    _gguf(d / "mmproj-F32.gguf", b"p", 2048)
    found = gw._scan_gguf_tree(store.lms)
    assert len(found) == 1, "the projector must not be listed as a model"
    assert found[0]["name"] == "Qwopus-Q4_K_M.gguf"
    assert found[0]["mmproj"].endswith("mmproj-F32.gguf")


def test_smallest_projector_wins_when_a_repo_ships_several(store):
    d = store.lms / "pub" / "repo-GGUF"
    _gguf(d / "model-Q4.gguf", b"m", 8192)
    _gguf(d / "mmproj-BF16.gguf", b"p", 4096)
    _gguf(d / "mmproj-Q8_0.gguf", b"q", 1024)
    found = gw._scan_gguf_tree(store.lms)
    assert found[0]["mmproj"].endswith("mmproj-Q8_0.gguf")


def test_only_the_first_shard_is_a_model(store):
    """llama-server is handed shard 1 and finds the rest itself. Registering
    shard 3 of 4 is a guaranteed load failure."""
    d = store.lms / "lmstudio-community" / "MiniMax-M2.5-GGUF"
    for i in range(1, 5):
        _gguf(d / f"MiniMax-M2.5-Q4_K_M-{i:05d}-of-00004.gguf", bytes([97 + i]))
    found = gw._scan_gguf_tree(store.lms)
    assert [f["name"] for f in found] == [
        "MiniMax-M2.5-Q4_K_M-00001-of-00004.gguf"
    ]


def test_a_download_in_flight_is_not_scanned(store):
    """The sync loop and a HuggingFace pull can land on the same second. A
    .part beside the name means the name is not yet the truth."""
    d = store.fleet / "org__repo-GGUF"
    _gguf(d / "model.gguf", b"m")
    _write(d / "model.gguf.part", b"half")
    _gguf(d / "other.gguf.part", b"x")
    names = {f["name"] for f in gw._scan_gguf_tree(store.fleet)}
    assert names == set(), "a file with a live .part beside it is still downloading"


def test_mlx_repos_are_listed_but_never_models(store):
    """Three of the four models on the M4 Air are MLX. llama.cpp cannot load
    safetensors, so they are named and explained rather than silently ignored."""
    d = store.lms / "ornith-ai" / "Ornith-1.5-9B-MLX-4bit"
    _write(d / "model.safetensors", b"x" * 2048)
    _write(d / "config.json", b"{}")
    assert gw._scan_gguf_tree(store.lms) == []
    mlx = gw._lmstudio_mlx_models(store.lms)
    assert [m["id"] for m in mlx] == ["ornith-ai/Ornith-1.5-9B-MLX-4bit"]
    assert mlx[0]["format"] == "mlx"


# --------------------------------------------------------------------------
# import: LM Studio -> models.json, in place
# --------------------------------------------------------------------------


def test_import_registers_the_lmstudio_path_and_copies_nothing(store, monkeypatch):
    """The direction that works on every filesystem, including apu-tablet-2's
    exFAT: the fleet learns where the file already is."""
    src = _gguf(store.lms / "empero-ai" / "Marmot-9B-Distill-GGUF"
                / "Marmot-9B-Q4_K_M.gguf", b"m", 8192)
    before = src.stat()
    # An import lands ENABLED only when an engine is listening on the
    # upstream port -- and this test is about the file, not the engine.
    # Without the stub it passed on a workstation running llama-swap and
    # failed on every CI runner, where nothing listens on 8081.
    monkeypatch.setattr(gw, "_upstream_listening", lambda: True)

    res = gw.lmstudio_sync()

    assert res["imported"] == 1
    records = gw.load_models()
    assert [r["id"] for r in records] == ["marmot-9b-distill"]
    assert records[0]["path"] == str(src)
    assert records[0]["enabled"] is True
    after = src.stat()
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert after.st_nlink == 1, "import must not create a second name"


def test_import_derives_an_id_that_satisfies_safe_id(store):
    _gguf(store.lms / "DavidAU"
          / "Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-GGUF"
          / "model-Q4_K_S.gguf", b"m")
    gw.lmstudio_sync()
    mid = gw.load_models()[0]["id"]
    assert gw.SAFE_ID.match(mid), mid
    assert not mid.lower().endswith("gguf"), "the -GGUF suffix is noise in an id"


def test_a_second_repo_with_the_same_name_does_not_collide(store):
    """llama-swap refuses an entire config over one duplicate id, which takes
    every other model on the box down with it."""
    _gguf(store.lms / "pubA" / "Marmot-27B-GGUF" / "a.gguf", b"a")
    _gguf(store.lms / "pubB" / "Marmot-27B-GGUF" / "b.gguf", b"b")
    gw.lmstudio_sync()
    ids = sorted(r["id"] for r in gw.load_models())
    # The first takes the plain name; the second is qualified by its publisher
    # rather than given a counter, so the two are told apart by something that
    # means something.
    assert ids == ["marmot-27b", "pubb-marmot-27b"]
    gw.check_name_collisions(gw.load_models())     # must not raise


def test_an_id_never_takes_a_public_catalogue_name(store):
    """resolve_targets() resolves a requested name as a Fleet Pass PUBLIC id
    before it looks at local models, and a public row points at whatever fleet
    ids it claims. A local model holding a public id is therefore unreachable
    under its own name -- the request goes to the row's fleet ids instead.

    This was found live: mac-laptop-2's LM Studio copy of Qwen3.8-9B-Distill
    derives exactly `qwen3.8-9b-distill`, which public_seed.json claimed only
    for the Ollama tag, and the first request for it came back "no host in
    the fleet serves model 'qwen3.8-9b-distill'" from the box that was
    holding the weights. That row now claims the name itself (see the next
    test); this pins the rule for a row that does not."""
    rows = gw.public_catalogue()["by_public"]
    victim = next((pid for pid, r in rows.items()
                   if pid not in gw._row_fleet_ids(r)), None)
    assert victim, "seed changed; every public id claims itself now"

    entry = {"repo": victim, "publisher": "pubb", "path": "/x/y.gguf"}
    got = gw._fleet_id_for(entry, set())
    assert got != victim
    assert got == "pubb-" + victim, (
        "qualify with the publisher rather than appending a counter")


def test_a_public_id_the_row_claims_for_the_fleet_is_taken_not_dodged(store):
    """The 9B distill row lists `qwen3.8-9b-distill` among its own fleet ids:
    the fleet serves the model under that very name (it is the canonical id in
    FLEET_MODEL_NAMES). So LM Studio's copy on mac-laptop-2 derives it and KEEPS it
    -- dodging to `empero-ai-qwen3.8-9b-distill` was the split the start-up
    rename then had to undo."""
    row = gw.public_catalogue()["by_public"]["qwen3.8-9b-distill"]
    assert "qwen3.8-9b-distill" in gw._row_fleet_ids(row), "seed changed"

    _gguf(store.lms / "empero-ai" / "Qwen3.8-9B-Distill-GGUF" / "m.gguf", b"m")
    gw.lmstudio_sync()
    assert gw.load_models()[0]["id"] == "qwen3.8-9b-distill"


def test_only_public_ids_are_avoided_not_the_ids_they_claim(store):
    """The opposite case is welcome and must not be dodged: an id a public row
    CLAIMS as one of its fleet_ids means this box really does serve that public
    model, and requests for the public id should start routing here. Only the
    row's own public_id is off limits."""
    rows = gw.public_catalogue()["by_public"]
    claimed = {f.lower() for r in rows.values() for f in gw._row_fleet_ids(r)}
    only_claimed = sorted(claimed - set(rows))
    assert only_claimed, "seed changed; no fleet id is distinct from its public id"
    target = only_claimed[0]
    entry = {"repo": target, "publisher": "somebody", "path": "/x/y.gguf"}
    assert gw._fleet_id_for(entry, set()) == target


def test_import_is_idempotent(store):
    _gguf(store.lms / "pub" / "repo-GGUF" / "m.gguf", b"m")
    gw.lmstudio_sync()
    gw.lmstudio_sync()
    gw.lmstudio_sync()
    assert len(gw.load_models()) == 1


def test_import_carries_the_projector_onto_the_record(store):
    d = store.lms / "pub" / "vision-GGUF"
    _gguf(d / "vision-Q4.gguf", b"m", 8192)
    _gguf(d / "mmproj-F16.gguf", b"p", 2048)
    gw.lmstudio_sync()
    assert gw.load_models()[0]["mmproj"].endswith("mmproj-F16.gguf")


def test_ollama_backed_box_blocks_import_and_says_why(store, monkeypatch):
    """gpu-laptop-2 and mini-pc-1 serve Ollama's catalogue. Writing models.json
    there would change nothing anyone can see, so it is refused with a reason
    rather than done pointlessly."""
    monkeypatch.setattr(gw, "UPSTREAM_MODELS", True)
    _gguf(store.lms / "pub" / "repo-GGUF" / "m.gguf", b"m")
    res = gw.lmstudio_sync()
    assert res["imported"] == 0
    assert gw.load_models() == []
    blocked = [a for a in res["actions"] if a["kind"] == "import"]
    assert blocked and blocked[0]["status"] == "blocked"
    assert "ollama create" in blocked[0]["reason"]


# --------------------------------------------------------------------------
# publish: fleet -> LM Studio's layout, by link
# --------------------------------------------------------------------------


def test_publish_hard_links_into_the_publisher_repo_layout(store):
    """LM Studio indexes <downloadsFolder>/<publisher>/<repo>/<file>.gguf and
    follows a hard link (measured on the M4 Air: visible in `lms ls` in under
    8 seconds). The fleet's flat org__repo directory name is the join to undo."""
    src = _gguf(store.fleet / "unsloth__Qwen3.8-27B-GGUF"
                / "Qwen3.8-27B-UD-Q5_K_M.gguf", b"m", 8192)

    res = gw.lmstudio_sync()

    dst = store.lms / "unsloth" / "Qwen3.8-27B-GGUF" / "Qwen3.8-27B-UD-Q5_K_M.gguf"
    assert res["published"] == 1
    assert dst.exists() and not dst.is_symlink()
    assert dst.stat().st_ino == src.stat().st_ino, "one inode, two names"
    assert res["copied_bytes"] == 0


def test_publish_falls_back_to_a_symlink_across_devices(store, monkeypatch):
    """mac-laptop-1 keeps its models on an external exFAT volume and LM Studio on the
    internal one. A hard link cannot span them; a symlink can, and both
    llama-server and LM Studio follow it."""
    src = _gguf(store.fleet / "org__repo-GGUF" / "m.gguf", b"m")
    real_link = os.link

    def cross_device(a, b, **kw):
        raise OSError(18, "Cross-device link")

    monkeypatch.setattr(gw.os, "link", cross_device)
    res = gw.lmstudio_sync()
    monkeypatch.setattr(gw.os, "link", real_link)

    dst = store.lms / "org" / "repo-GGUF" / "m.gguf"
    assert res["published"] == 1
    assert dst.is_symlink()
    assert dst.resolve() == src.resolve()


def test_publish_refuses_rather_than_copies_when_no_link_works(store, monkeypatch):
    """apu-tablet-2's shared D: is exFAT, where Windows refuses both link types.
    Spending the bytes is the one thing this subsystem exists to prevent, so
    the default is to report it instead."""
    _gguf(store.fleet / "org__repo-GGUF" / "m.gguf", b"m")

    def refuse(*a, **kw):
        raise OSError(1, "Operation not supported")

    monkeypatch.setattr(gw.os, "link", refuse)
    monkeypatch.setattr(gw.os, "symlink", refuse)

    res = gw.lmstudio_sync()
    assert res["published"] == 0
    assert res["copied_bytes"] == 0
    blocked = [a for a in res["actions"] if a["kind"] == "publish"][0]
    assert blocked["status"] == "blocked"
    assert "exFAT" in blocked["detail"]
    assert not (store.lms / "org" / "repo-GGUF" / "m.gguf").exists()


def test_publish_never_overwrites_a_different_lmstudio_file(store):
    """os.replace would eat somebody's real 20 GB download without a trace if
    two repos happened to name a file the same way."""
    _gguf(store.fleet / "org__repo-GGUF" / "m.gguf", b"fleet-bytes", 4096)
    theirs = _gguf(store.lms / "org" / "repo-GGUF" / "m.gguf", b"lmstudio-bytes", 4096)
    theirs_ino = theirs.stat().st_ino

    res = gw.lmstudio_sync()

    assert theirs.stat().st_ino == theirs_ino, "their file must be untouched"
    assert theirs.read_bytes().startswith(b"GGUFlmstudio")
    blocked = [a for a in res["actions"] if a["kind"] == "publish"][0]
    assert blocked["status"] == "blocked"
    assert "refusing to overwrite" in blocked["detail"]


def test_publish_takes_the_projector_with_the_model(store):
    d = store.fleet / "org__vision-GGUF"
    _gguf(d / "vision-Q4.gguf", b"m", 8192)
    _gguf(d / "mmproj-F16.gguf", b"p", 2048)
    gw.lmstudio_sync()
    out = store.lms / "org" / "vision-GGUF"
    assert (out / "vision-Q4.gguf").exists()
    assert (out / "mmproj-F16.gguf").exists(), "a projector alone is useless"


def test_publish_is_idempotent(store):
    _gguf(store.fleet / "org__repo-GGUF" / "m.gguf", b"m")
    first = gw.lmstudio_sync()
    second = gw.lmstudio_sync()
    assert first["published"] == 1
    assert second["published"] == 0, "the second pass has nothing left to do"


def test_a_published_model_is_not_then_imported_back(store):
    """The two directions must not chase each other: publish makes the fleet
    file visible to LM Studio, and the next scan must recognise it as the same
    inode rather than registering a second record for it."""
    _gguf(store.fleet / "org__repo-GGUF" / "m.gguf", b"m")
    gw.lmstudio_sync()
    gw.lmstudio_sync()
    gw.lmstudio_sync()
    assert len(gw.load_models()) <= 1
    assert len(gw._scan_gguf_tree(store.lms)) == 1


# --------------------------------------------------------------------------
# reclaim: the duplicated bytes
# --------------------------------------------------------------------------


def test_reclaim_collapses_a_duplicate_pair_onto_one_inode(store):
    """This is gpu-laptop-2's 20.22 GiB: one Qwopus GGUF stored once under each
    layout on the same volume."""
    payload = b"GGUF" + b"z" * 8188
    lms_copy = _write(store.lms / "Jackrong" / "Qwopus-GGUF" / "Qwopus-Q4_K_M.gguf",
                      payload)
    fleet_copy = _write(store.fleet / "Jackrong__Qwopus-GGUF" / "Qwopus-Q4_K_M.gguf",
                        payload)
    assert lms_copy.stat().st_ino != fleet_copy.stat().st_ino

    res = gw.lmstudio_sync()

    assert res["reclaimed"] == 1
    assert res["freed_bytes"] == len(payload)
    assert fleet_copy.stat().st_ino == lms_copy.stat().st_ino
    assert fleet_copy.read_bytes() == payload, "both names still read correctly"


def test_reclaim_keeps_the_lmstudio_copy_not_the_fleet_one(store):
    """LM Studio's own index points at its path; the fleet's record is
    rewritten by the next sync. Replacing the LM Studio side would leave that
    index pointing at a file that had been swapped underneath it."""
    payload = b"GGUF" + b"z" * 4092
    lms_copy = _write(store.lms / "pub" / "repo-GGUF" / "m.gguf", payload)
    lms_ino = lms_copy.stat().st_ino
    _write(store.fleet / "pub__repo-GGUF" / "m.gguf", payload)

    gw.lmstudio_sync()

    assert lms_copy.stat().st_ino == lms_ino, "the LM Studio inode survives"


def test_reclaim_refuses_files_that_only_look_alike(store):
    """Same size, different bytes. The edge digest is what stops this."""
    a = _write(store.lms / "pub" / "repo-GGUF" / "m.gguf", b"GGUF" + b"a" * 4092)
    b = _write(store.fleet / "pub__repo-GGUF" / "m.gguf", b"GGUF" + b"b" * 4092)
    a_ino, b_ino = a.stat().st_ino, b.stat().st_ino

    res = gw.lmstudio_sync()

    assert res["reclaimed"] == 0
    assert a.stat().st_ino == a_ino and b.stat().st_ino == b_ino
    assert b.read_bytes().endswith(b"b")


def test_reclaim_refuses_across_volumes(store, monkeypatch):
    """A hard link cannot span devices, and swapping a real file for a symlink
    onto a removable drive trades duplicated bytes for a model that vanishes
    when the drive is unplugged."""
    payload = b"GGUF" + b"z" * 4092
    _write(store.lms / "pub" / "repo-GGUF" / "m.gguf", payload)
    fleet_copy = _write(store.fleet / "pub__repo-GGUF" / "m.gguf", payload)
    fleet_ino = fleet_copy.stat().st_ino

    class Elsewhere:
        """Reports the fleet tree as living on another device."""

        def __init__(self, orig):
            self.orig = orig

        def __call__(self, path, *a, **kw):
            st = self.orig(path, *a, **kw)
            if str(store.fleet) in str(path):
                return os.stat_result(
                    (st.st_mode, st.st_ino, st.st_dev + 1, st.st_nlink, st.st_uid,
                     st.st_gid, st.st_size, int(st.st_atime), int(st.st_mtime),
                     int(st.st_ctime)))
            return st

    monkeypatch.setattr(gw.os, "stat", Elsewhere(os.stat))
    res = gw.lmstudio_sync()
    monkeypatch.undo()

    assert res["reclaimed"] == 0
    assert fleet_copy.stat().st_ino == fleet_ino
    blocked = [a for a in res["actions"] if a["kind"] == "reclaim"]
    assert blocked and blocked[0]["status"] == "blocked"
    assert "different volumes" in blocked[0]["reason"]


# --------------------------------------------------------------------------
# the download hook
# --------------------------------------------------------------------------


def test_a_finished_download_lands_in_lmstudio_immediately(store):
    """"Pull From HuggingFace" must not wait out the sync interval."""
    dest = _gguf(store.fleet / "unsloth__gemma-4-26B-GGUF" / "gemma-4-26B-Q4.gguf",
                 b"m", 8192)
    how = gw.lmstudio_publish_one(dest)
    dst = store.lms / "unsloth" / "gemma-4-26B-GGUF" / "gemma-4-26B-Q4.gguf"
    assert how == "hardlink"
    assert dst.stat().st_ino == dest.stat().st_ino


def test_the_hook_never_fails_a_download(store, monkeypatch):
    """A download that saved its bytes must never be reported as failed
    because LM Studio's directory was read-only."""
    dest = _gguf(store.fleet / "org__repo-GGUF" / "m.gguf", b"m")

    def boom(*a, **kw):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(gw.os, "link", boom)
    monkeypatch.setattr(gw.os, "symlink", boom)
    monkeypatch.setattr(gw.Path, "mkdir", boom)
    assert gw.lmstudio_publish_one(dest) == ""


def test_the_hook_is_silent_where_lmstudio_is_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(gw, "_lmstudio_homes", lambda: [tmp_path / "nope"])
    gw._lmstudio_root_cache.update(t=0.0, root=None, home=None)
    assert gw.lmstudio_publish_one(tmp_path / "x.gguf") == ""


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------


def test_switches_turn_their_own_direction_off(store):
    _gguf(store.lms / "pub" / "in-GGUF" / "a.gguf", b"a")
    _gguf(store.fleet / "org__out-GGUF" / "b.gguf", b"b")
    gw.set_lmstudio_settings({"import_to_fleet": False,
                              "publish_to_lmstudio": False})
    res = gw.lmstudio_sync()
    assert res["imported"] == 0 and res["published"] == 0
    assert {a["status"] for a in res["actions"]} == {"off"}


def test_the_interval_is_clamped_to_something_sane(store):
    assert gw.set_lmstudio_settings({"interval_secs": 1})["interval_secs"] == 60
    assert gw.set_lmstudio_settings({"interval_secs": 10**9})["interval_secs"] == 86400
    assert gw.set_lmstudio_settings({"interval_secs": "nonsense"})[
        "interval_secs"] == gw.DEFAULT_LMSTUDIO_SETTINGS["interval_secs"]


def test_unknown_settings_keys_are_ignored(store):
    gw.set_lmstudio_settings({"enabled": False, "rm": "-rf /"})
    stored = gw.get_lmstudio_settings()
    assert stored["enabled"] is False
    assert "rm" not in stored


def test_a_dry_run_changes_nothing(store):
    _gguf(store.lms / "pub" / "repo-GGUF" / "m.gguf", b"m")
    _gguf(store.fleet / "org__other-GGUF" / "n.gguf", b"n")
    res = gw.lmstudio_sync(dry_run=True)
    assert res["dry_run"] is True
    assert res["todo"] == 2
    assert gw.load_models() == []
    assert not (store.lms / "org" / "other-GGUF" / "n.gguf").exists()


def test_kinds_narrows_a_pass_to_one_direction(store):
    """The panel's "reclaim the duplicates" button must not also register
    forty new models."""
    _gguf(store.lms / "pub" / "repo-GGUF" / "m.gguf", b"m")
    _gguf(store.fleet / "org__other-GGUF" / "n.gguf", b"n")
    res = gw.lmstudio_sync(kinds={"publish"})
    assert res["published"] == 1
    assert res["imported"] == 0
    assert gw.load_models() == []


# --------------------------------------------------------------------------
# the admin surface
# --------------------------------------------------------------------------


def test_the_endpoints_need_admin(client):
    assert client.get("/admin/api/lmstudio").status_code == 401
    assert client.post("/admin/api/lmstudio/sync").status_code == 401
    assert client.put("/admin/api/lmstudio/settings", json={}).status_code == 401


def test_sync_rejects_an_unknown_kind(client, admin_headers):
    r = client.post("/admin/api/lmstudio/sync?kinds=rm-rf", headers=admin_headers)
    assert r.status_code == 400
    assert "unknown kind" in r.json()["detail"]


def test_the_panel_reads_the_same_plan_the_button_runs(client, admin_headers):
    r = client.get("/admin/api/lmstudio", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"available", "root", "actions", "settings", "mlx",
                         "reclaimable", "upstream_models"}


# --------------------------------------------------------------------------
# the two stores sharing one directory (apu-tablet-2)
# --------------------------------------------------------------------------


@pytest.fixture
def shared_store(tmp_path, monkeypatch):
    """apu-tablet-2 points LM Studio's downloadsFolder AND the fleet's
    MODELS_DIR at the same D:\\AI\\models, in two different layouts."""
    home = tmp_path / "lmstudio-home"
    both = tmp_path / "AI-models"
    home.mkdir(parents=True, exist_ok=True)
    both.mkdir(parents=True, exist_ok=True)
    (home / "settings.json").write_text(json.dumps({"downloadsFolder": str(both)}))
    monkeypatch.setattr(gw, "LMSTUDIO_HOME_ENV", str(home))
    monkeypatch.setattr(gw, "MODELS_DIR", both)
    monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")
    monkeypatch.setattr(gw, "SWAP_CONFIG", tmp_path / "llama-swap.yaml")
    monkeypatch.setattr(gw, "UPSTREAM_MODELS", False)
    monkeypatch.setattr(gw, "_lmstudio_restart_swap", lambda: "test")
    gw._lmstudio_root_cache.update(t=0.0, root=None, home=None)
    gw._lmstudio_last.clear()
    gw.db_exec("DELETE FROM settings WHERE key='lmstudio'")
    gw.save_models([])
    yield type("Shared", (), {"home": home, "dir": both, "tmp": tmp_path})


def test_a_shared_directory_does_not_publish_lmstudio_files_to_itself(shared_store):
    """The fleet scan walks LM Studio's own nested repos on a shared-directory
    box. Publishing one would build a second name for it under a made-up
    `fleet` publisher, in the directory it is already in."""
    d = shared_store.dir
    _gguf(d / "DavidAU" / "Fable-27B-GGUF" / "fable-Q4.gguf", b"a")   # LM Studio's
    _gguf(d / "ggml-org__Qwen3.8-27B-GGUF" / "qwen-Q4.gguf", b"b")    # the fleet's

    res = gw.lmstudio_sync()

    assert not (d / "fleet").exists(), "no invented publisher directory"
    assert res["published"] == 0, "everything here is already where LM Studio looks"
    # ...and the flat one is still registered, which is the direction that
    # matters on the box whose volume refuses every kind of link.
    assert len(gw.load_models()) == 2


def test_a_shared_directory_still_imports_both_layouts(shared_store):
    d = shared_store.dir
    _gguf(d / "DavidAU" / "Fable-27B-GGUF" / "fable-Q4.gguf", b"a")
    _gguf(d / "ggml-org__Qwen3.8-27B-GGUF" / "qwen-Q4.gguf", b"b")
    gw.lmstudio_sync()
    # The flat name is split on its `__`, so the id reads like the model and
    # not like the folder. `qwen3.8-27b` is a public catalogue id, but the row
    # claims that very fleet id, so the name is taken rather than dodged: this
    # box really does serve that public model now.
    assert sorted(r["id"] for r in gw.load_models()) == [
        "fable-27b", "qwen3.8-27b"]


# --------------------------------------------------------------------------
# a deleted import stays deleted
# --------------------------------------------------------------------------


def test_a_record_deleted_by_hand_is_not_resurrected(store):
    """An automatic writer must never undo a manual edit. Without this the
    Models tab's delete button appears not to work: the record is back within
    interval_secs."""
    _gguf(store.lms / "pub" / "unwanted-GGUF" / "m.gguf", b"m")
    gw.lmstudio_sync()
    assert len(gw.load_models()) == 1

    gw.save_models([])                      # the operator deletes it
    res = gw.lmstudio_sync()

    assert gw.load_models() == [], "it must stay gone"
    assert res["imported"] == 0
    dismissed = [a for a in res["actions"] if a["status"] == "dismissed"]
    assert dismissed and "removed from the Models tab" in dismissed[0]["reason"]


def test_a_dismissal_is_forgotten_if_the_weights_themselves_are_deleted(store):
    """Deleting the GGUF is not the same gesture as deleting the record. If
    the same model is downloaded again later it should be imported again."""
    f = _gguf(store.lms / "pub" / "gone-GGUF" / "m.gguf", b"m")
    gw.lmstudio_sync()
    gw.save_models([])
    f.unlink()
    gw.lmstudio_sync()
    assert gw.get_lmstudio_settings()["dismissed_paths"] == []

    _gguf(store.lms / "pub" / "gone-GGUF" / "m.gguf", b"m")
    res = gw.lmstudio_sync()
    assert res["imported"] == 1


def test_forgetting_dismissals_offers_the_model_again(store):
    _gguf(store.lms / "pub" / "unwanted-GGUF" / "m.gguf", b"m")
    gw.lmstudio_sync()
    gw.save_models([])
    gw.lmstudio_sync()
    assert gw.get_lmstudio_settings()["dismissed_paths"]

    gw.set_lmstudio_settings({"dismissed_paths": []})
    assert gw.lmstudio_sync()["imported"] == 1


def test_forget_dismissed_endpoint_needs_admin(client):
    assert client.post("/admin/api/lmstudio/forget-dismissed").status_code == 401


# --------------------------------------------------------------------------
# the full-hash gate on the one destructive operation
# --------------------------------------------------------------------------


def test_files_matching_only_at_the_edges_are_never_collapsed(store):
    """The edge digest is what a five-minute scan can afford; it is not what
    may authorise discarding a directory entry. Same size, same first MiB,
    same last MiB, different middle."""
    # Bigger than 2 MiB, or the "edges" are the whole file and the sampling
    # this test is about never happens.
    MiB = 1 << 20
    head = b"GGUF" + b"h" * (MiB - 4)
    tail = b"t" * MiB
    a = _write(store.lms / "pub" / "repo-GGUF" / "m.gguf", head + b"A" * MiB + tail)
    b = _write(store.fleet / "pub__repo-GGUF" / "m.gguf", head + b"B" * MiB + tail)
    assert a.stat().st_size == b.stat().st_size
    assert gw._edge_digest(a, a.stat().st_size) == gw._edge_digest(b, b.stat().st_size)

    res = gw.lmstudio_sync()

    assert res["reclaimed"] == 0
    assert b.read_bytes()[MiB:2 * MiB] == b"B" * MiB, "the fleet copy is intact"
    blocked = [x for x in res["actions"] if x["kind"] == "reclaim"][0]
    assert "differ somewhere in the middle" in blocked["detail"]


def test_a_near_miss_is_not_re_read_on_every_pass(store):
    """Two 20 GiB files that match at both ends would otherwise be read in
    full every five minutes, forever, to reach the same answer."""
    MiB = 1 << 20
    head, tail = b"GGUF" + b"h" * (MiB - 4), b"t" * MiB
    _write(store.lms / "pub" / "repo-GGUF" / "m.gguf", head + b"A" * MiB + tail)
    _write(store.fleet / "pub__repo-GGUF" / "m.gguf", head + b"B" * MiB + tail)
    gw.lmstudio_sync()
    assert gw.get_lmstudio_settings()["mismatched_pairs"]

    calls = []
    real = gw._full_digest
    gw._full_digest = lambda p: (calls.append(p), real(p))[1]
    try:
        res = gw.lmstudio_sync()
    finally:
        gw._full_digest = real
    assert calls == [], "the pair is remembered, not re-hashed"
    assert [a["status"] for a in res["actions"] if a["kind"] == "reclaim"] == ["blocked"]


def test_a_genuine_duplicate_still_survives_the_full_hash(store):
    payload = b"GGUF" + bytes(range(256)) * 64
    lms_copy = _write(store.lms / "pub" / "repo-GGUF" / "m.gguf", payload)
    fleet_copy = _write(store.fleet / "pub__repo-GGUF" / "m.gguf", payload)
    res = gw.lmstudio_sync()
    assert res["reclaimed"] == 1
    assert fleet_copy.stat().st_ino == lms_copy.stat().st_ino
    assert gw.get_lmstudio_settings()["mismatched_pairs"] == []


# --------------------------------------------------------------------------
# the Ollama-backed import (explicit, never automatic)
# --------------------------------------------------------------------------


def test_ollama_import_is_refused_on_a_llama_cpp_box(store):
    f = _gguf(store.lms / "pub" / "repo-GGUF" / "m.gguf", b"m")
    with pytest.raises(gw.HTTPException) as exc:
        gw.lmstudio_ollama_import(str(f), "whatever")
    assert exc.value.status_code == 400


def test_ollama_import_refuses_a_path_outside_lmstudio(store, monkeypatch):
    """The path reaches this from an HTTP body; it must not be able to hand
    Ollama an arbitrary file."""
    monkeypatch.setattr(gw, "UPSTREAM_MODELS", True)
    outside = _gguf(store.tmp / "elsewhere" / "x.gguf", b"m")
    with pytest.raises(gw.HTTPException) as exc:
        gw.lmstudio_ollama_import(str(outside), "name")
    assert "not inside LM Studio" in str(exc.value.detail)


def test_ollama_import_refuses_a_bad_model_name(store, monkeypatch):
    monkeypatch.setattr(gw, "UPSTREAM_MODELS", True)
    f = _gguf(store.lms / "pub" / "repo-GGUF" / "m.gguf", b"m")
    with pytest.raises(gw.HTTPException) as exc:
        gw.lmstudio_ollama_import(str(f), "bad name; rm -rf /")
    assert exc.value.status_code == 400


def test_ollama_import_collapses_the_copy_it_just_made(store, monkeypatch, tmp_path):
    """Ollama can only take a GGUF by copying it into its blob store. The copy
    is byte-identical, so it is given straight back."""
    monkeypatch.setattr(gw, "UPSTREAM_MODELS", True)
    payload = b"GGUF" + bytes(range(256)) * 64
    src = _write(store.lms / "pub" / "repo-GGUF" / "m.gguf", payload)
    blobs = tmp_path / "ollama" / "blobs"
    blobs.mkdir(parents=True)

    def fake_create(cmd, **kw):
        # `ollama create <name> -f <modelfile>` -- the Modelfile is a real file
        # on disk, because `-f -` is not supported (measured on gpu-laptop-2).
        assert cmd[1] == "create" and cmd[3] == "-f"
        assert Path(cmd[4]).read_text().startswith("FROM ")
        # what it then does: copy the weights into the blob store
        _write(blobs / "sha256-deadbeef", payload)

        class P:
            returncode = 0
            # None, not "" -- which is what gpu-laptop-2 actually returned, and
            # what turned a successful import into a 500 with the duplicate
            # left on disk.
            stdout = None
            stderr = None
        return P()

    monkeypatch.setattr(gw, "_ollama_binary", lambda: "/usr/bin/ollama")
    monkeypatch.setattr(gw, "_ollama_stores", lambda: [tmp_path / "ollama"])
    monkeypatch.setattr(gw.subprocess, "run", fake_create)

    res = gw.lmstudio_ollama_import(str(src), "mymodel:latest")

    assert res["reclaimed"] == len(payload)
    blob = blobs / "sha256-deadbeef"
    assert blob.stat().st_ino == src.stat().st_ino, "one inode, two names"
    assert blob.read_bytes() == payload


def test_ollama_import_reports_honestly_when_the_blob_store_is_hidden(
        store, monkeypatch):
    """The Windows gateway runs as SYSTEM and does not inherit the
    interactive user's OLLAMA_MODELS. Say so rather than claiming a reclaim."""
    monkeypatch.setattr(gw, "UPSTREAM_MODELS", True)
    src = _gguf(store.lms / "pub" / "repo-GGUF" / "m.gguf", b"m")

    class P:
        returncode = 0
        stdout = "success"
        stderr = ""

    monkeypatch.setattr(gw, "_ollama_binary", lambda: "/usr/bin/ollama")
    monkeypatch.setattr(gw, "_ollama_stores", lambda: [])
    monkeypatch.setattr(gw.subprocess, "run", lambda *a, **kw: P())

    res = gw.lmstudio_ollama_import(str(src), "mymodel")
    assert res["reclaimed"] == 0
    assert "on disk twice" in res["note"]


def test_ollama_import_says_when_the_binary_is_unreachable(store, monkeypatch):
    monkeypatch.setattr(gw, "UPSTREAM_MODELS", True)
    src = _gguf(store.lms / "pub" / "repo-GGUF" / "m.gguf", b"m")
    monkeypatch.setattr(gw, "_ollama_binary", lambda: "")
    with pytest.raises(gw.HTTPException) as exc:
        gw.lmstudio_ollama_import(str(src), "mymodel")
    assert exc.value.status_code == 503
    assert "SYSTEM" in str(exc.value.detail)


# --------------------------------------------------------------------------
# a box with no engine must not advertise models
# --------------------------------------------------------------------------


def test_imports_land_disabled_when_nothing_is_listening_upstream(store, monkeypatch):
    """Registering a model is a claim the hub believes: the id enters the
    routing table and requests start arriving. gpu-desktop-2 is a staging peer
    with no llama-swap installed at all, and its first sync registered 16
    LM Studio models as routes to a closed port."""
    monkeypatch.setattr(gw, "_upstream_listening", lambda timeout=2.0: False)
    _gguf(store.lms / "pub" / "repo-GGUF" / "m.gguf", b"m")

    res = gw.lmstudio_sync()

    assert res["imported"] == 1
    assert gw.load_models()[0]["enabled"] is False
    assert any("DISABLED" in n for n in res["notes"])
    assert "disabled" in [a for a in res["actions"] if a["kind"] == "import"][0]["detail"]


def test_imports_are_enabled_when_an_engine_is_there(store, monkeypatch):
    monkeypatch.setattr(gw, "_upstream_listening", lambda timeout=2.0: True)
    _gguf(store.lms / "pub" / "repo-GGUF" / "m.gguf", b"m")
    gw.lmstudio_sync()
    assert gw.load_models()[0]["enabled"] is True


def test_upstream_listening_is_a_real_socket_check(monkeypatch):
    import socket as _s
    srv = _s.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        monkeypatch.setattr(gw, "UPSTREAM", f"http://127.0.0.1:{port}")
        assert gw._upstream_listening() is True
    finally:
        srv.close()
    monkeypatch.setattr(gw, "UPSTREAM", f"http://127.0.0.1:{port}")
    assert gw._upstream_listening(timeout=0.5) is False


def test_an_ollama_box_does_not_offer_back_its_own_published_links(store, monkeypatch):
    """The publish direction links the fleet's files into LM Studio's tree.
    Offering to hand one of those to Ollama would be offering the box a copy
    of something it already has."""
    monkeypatch.setattr(gw, "_upstream_listening", lambda timeout=2.0: True)
    _gguf(store.fleet / "org__repo-GGUF" / "m.gguf", b"m")
    gw.lmstudio_sync()                                   # publishes the link
    monkeypatch.setattr(gw, "UPSTREAM_MODELS", True)     # ...box becomes Ollama-backed
    res = gw.lmstudio_sync()
    assert [a for a in res["actions"] if a["kind"] == "import"] == []
