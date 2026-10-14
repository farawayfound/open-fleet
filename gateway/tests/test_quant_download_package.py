"""Per-quant package downloads: /admin/api/download-package,
/admin/api/jobs/cancel-package, the pkg-tagged jobs rows, the package
fan-out gate, _package_status, list_local_models' new fields, the
shard-aware DELETE, /admin/api/hf/quants, and _fit_for_weight_bytes.

Every class here is one piece of the Library's "one row per quant level,
one Download button for all its shards" promise (see SPEC.md), tested for
the failure it exists to prevent -- eight shards racing one .part, a
package silently dropping a file that is already on disk, an Ollama box
being handed a raw HF pull it cannot serve, a stale jobs row outliving its
worker and hiding a finished download.

Real tmp filesystems and tiny fake GGUFs throughout (see test_lmstudio_sync
.py) wherever a test touches disk; the download worker is parked exactly as
test_review_fixes.py::TestReQueueingADownloadDoesNotForkASecondWriter does
whenever a test only cares about the queueing side, not the transfer.

Run: python -m pytest gateway/tests/test_quant_download_package.py -q
"""
from __future__ import annotations

import errno
import json
import os
import threading
import time
from pathlib import Path

import httpx
import pytest

import app as gw

FIXTURES = json.loads(
    (Path(__file__).parent / "hf_trees.json").read_text(encoding="utf-8")
)["repos"]


# ---------------------------------------------------------------------------
# shared fixtures
# ---------------------------------------------------------------------------


def _gguf(path: Path, marker: bytes = b"a", size: int = 4096) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + marker * (size - 4))
    return path


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    """An isolated MODELS_DIR/models.json pair -- start_download,
    _package_status, list_local_models and the DELETE endpoint all key off
    MODELS_DIR directly, so every test that touches disk gets its own tree
    rather than fighting over the real gateway's."""
    fleet = tmp_path / "fleet-models"
    fleet.mkdir()
    monkeypatch.setattr(gw, "MODELS_DIR", fleet)
    monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")
    monkeypatch.setattr(gw, "UPSTREAM_MODELS", False)
    gw.save_models([])
    return fleet


@pytest.fixture
def parked(monkeypatch):
    """Keeps _download_worker off the network entirely: the job jumps
    straight to 'downloading' and blocks behind an Event instead of
    TestReQueueingADownloadDoesNotForkASecondWriter's fixed 30s sleep, so
    teardown can release every parked thread immediately. Every test here
    that only cares about the QUEUEING side -- rows, pkg tags, dedup,
    validation, cancel-package -- never depends on real timing or HF being
    reachable."""
    release = threading.Event()

    def _park(job_id, repo, filename, dest, gate=None, **kw):
        gw.job_update(job_id, status="downloading")
        release.wait(30)

    monkeypatch.setattr(gw, "_download_worker", _park)
    yield release
    release.set()


@pytest.fixture
def job_cleanup():
    """Every job row and thread a test creates must not outlive it -- the
    jobs table has no per-test isolation (unlike the public-state tables
    conftest wipes), so a leaked 'downloading' row would confuse the very
    next test that lists /admin/api/jobs."""
    ids: list[int] = []
    yield ids
    for job_id in ids:
        gw._download_cancel.add(job_id)
        gw._download_threads.pop(job_id, None)
    if ids:
        gw.db_exec(
            "DELETE FROM jobs WHERE id IN (%s)" % ",".join("?" * len(ids)), ids
        )

# ---------------------------------------------------------------------------
# 1. download-package queues every file of one quant under one pkg label --
#    the whole point of the endpoint (spec: "ONE primary button 'Download'
#    that queues every file of that quant").
# ---------------------------------------------------------------------------

class TestDownloadPackageQueuesEveryFileUnderOnePkg:

    def test_creates_one_row_per_file_all_tagged_pkg(
            self, client, admin_headers, models_dir, parked, job_cleanup):
        repo, label = "vendor/pkg-basic", "Q4_K_M"
        files = ["model-Q4_K_M-00001-of-00002.gguf",
                 "model-Q4_K_M-00002-of-00002.gguf"]
        r = client.post("/admin/api/download-package", headers=admin_headers,
                        json={"repo": repo, "label": label, "files": files,
                              "mmproj": ""})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["repo"] == repo and body["pkg"] == label
        job_cleanup.extend(j["job_id"] for j in body["jobs"] if "job_id" in j)
        assert len(body["jobs"]) == 2

        rows = gw.db_query("SELECT * FROM jobs WHERE repo=? AND pkg=?", (repo, label))
        assert len(rows) == 2
        assert {r_["filename"] for r_ in rows} == set(files)

    def test_a_file_already_on_disk_is_reported_and_the_rest_still_queue(
            self, client, admin_headers, models_dir, parked, job_cleanup):
        repo, label = "vendor/pkg-disk", "Q4_K_M"
        on_disk, queued = ("model-Q4_K_M-00001-of-00002.gguf",
                           "model-Q4_K_M-00002-of-00002.gguf")
        dest_dir = models_dir / repo.replace("/", "__")
        _gguf(dest_dir / on_disk)

        r = client.post("/admin/api/download-package", headers=admin_headers,
                        json={"repo": repo, "label": label,
                              "files": [on_disk, queued], "mmproj": ""})
        assert r.status_code == 200, r.text
        jobs = r.json()["jobs"]
        job_cleanup.extend(j["job_id"] for j in jobs if "job_id" in j)

        done = next(j for j in jobs if j["dest"].endswith(on_disk))
        assert done.get("already_downloaded") is True
        still = next(j for j in jobs if j["dest"].endswith(queued))
        assert "job_id" in still and "already_downloaded" not in still

    def test_an_earlier_untagged_inflight_row_is_backfilled_with_the_pkg(
            self, client, admin_headers, models_dir, parked, job_cleanup):
        """A file queued by hand (or an older client) before this feature
        existed folds into the package retroactively, so cancel-package and
        the Downloads panel's grouping see the whole set."""
        repo, label = "vendor/pkg-backfill", "Q4_K_M"
        filename = "model-Q4_K_M-00001-of-00002.gguf"
        other = "model-Q4_K_M-00002-of-00002.gguf"

        first = gw.start_download(repo, filename)  # untagged: pkg=''
        job_cleanup.append(first["job_id"])
        assert gw.db_query("SELECT pkg FROM jobs WHERE id=?",
                           (first["job_id"],))[0]["pkg"] == ""

        r = client.post("/admin/api/download-package", headers=admin_headers,
                        json={"repo": repo, "label": label,
                              "files": [filename, other], "mmproj": ""})
        assert r.status_code == 200, r.text
        jobs = r.json()["jobs"]
        job_cleanup.extend(
            j["job_id"] for j in jobs if "job_id" in j and j["job_id"] != first["job_id"]
        )

        matched = next(j for j in jobs if j.get("job_id") == first["job_id"])
        assert matched.get("already_running") is True, (
            "the parked worker's thread is still alive, so this must be the "
            "re-queue-returns-the-first-job branch, not a fresh insert"
        )
        assert gw.db_query("SELECT pkg FROM jobs WHERE id=?",
                           (first["job_id"],))[0]["pkg"] == label


# ---------------------------------------------------------------------------
# 2. validation -- every one of these must fail before start_download is
#    ever called, so none of them needs the `parked` fixture.
# ---------------------------------------------------------------------------

class TestDownloadPackageValidation:
    REPO = "vendor/pkg-validate"

    def test_blank_label_400(self, client, admin_headers, models_dir):
        r = client.post("/admin/api/download-package", headers=admin_headers,
                        json={"repo": self.REPO, "label": "   ",
                              "files": ["a.gguf"]})
        assert r.status_code == 400
        assert "label" in r.json()["detail"].lower()

    def test_empty_files_400(self, client, admin_headers, models_dir):
        r = client.post("/admin/api/download-package", headers=admin_headers,
                        json={"repo": self.REPO, "label": "Q4_K_M", "files": []})
        assert r.status_code == 400

    def test_more_than_64_files_400(self, client, admin_headers, models_dir):
        files = ["file%d.gguf" % i for i in range(65)]
        r = client.post("/admin/api/download-package", headers=admin_headers,
                        json={"repo": self.REPO, "label": "Q4_K_M", "files": files})
        assert r.status_code == 400
        assert "too many" in r.json()["detail"].lower()

    def test_a_path_that_escapes_the_repo_400(self, client, admin_headers, models_dir):
        r = client.post("/admin/api/download-package", headers=admin_headers,
                        json={"repo": self.REPO, "label": "Q4_K_M",
                              "files": ["../../etc/passwd.gguf"]})
        assert r.status_code == 400

    def test_duplicate_basenames_across_directories_400(
            self, client, admin_headers, models_dir):
        """Every dest flattens to its basename -- two files that only differ
        by directory would both land on one .part and race."""
        r = client.post("/admin/api/download-package", headers=admin_headers,
                        json={"repo": self.REPO, "label": "Q4_K_M",
                              "files": ["dir1/model.gguf", "dir2/model.gguf"]})
        assert r.status_code == 400
        assert "duplicate" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 3. an Ollama-managed box never gets a raw HF pull started under it.
# ---------------------------------------------------------------------------

class TestDownloadPackageRefusedOnOllamaManagedBoxes:

    def test_409_names_the_ollama_pull_command(
            self, client, admin_headers, models_dir, monkeypatch):
        monkeypatch.setattr(gw, "UPSTREAM_MODELS", True)
        repo, label = "vendor/ollama-repo", "Q4_K_M"
        r = client.post("/admin/api/download-package", headers=admin_headers,
                        json={"repo": repo, "label": label, "files": ["m.gguf"]})
        assert r.status_code == 409
        assert ("ollama pull hf.co/" + repo + ":" + label) in r.json()["detail"]


# ---------------------------------------------------------------------------
# 4. /admin/api/download (single-file) must still behave exactly as before
#    this feature shipped -- a contract guard, not a new behaviour.
# ---------------------------------------------------------------------------

class TestPlainDownloadEndpointUnchanged:

    def test_still_returns_job_id_and_dest(
            self, client, admin_headers, models_dir, parked, job_cleanup):
        r = client.post("/admin/api/download", headers=admin_headers,
                        json={"repo": "vendor/plain-dl", "filename": "model.gguf"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body.keys()) == {"job_id", "dest"}
        job_cleanup.append(body["job_id"])
        row = gw.db_query("SELECT pkg FROM jobs WHERE id=?", (body["job_id"],))[0]
        assert row["pkg"] == "", "a lone /download call is not a package"


# ---------------------------------------------------------------------------
# 5. /admin/api/jobs annotates every row with its role, and pkg arrives via
#    SELECT * unchanged.
# ---------------------------------------------------------------------------

class TestJobsListingCarriesRoleAndPkg:

    def test_role_and_pkg_are_annotated(
            self, client, admin_headers, models_dir, parked, job_cleanup):
        repo, label = "vendor/pkg-jobs", "Q4_K_M"
        files = ["model-Q4_K_M-00001-of-00002.gguf", "mmproj-F16.gguf"]
        r = client.post("/admin/api/download-package", headers=admin_headers,
                        json={"repo": repo, "label": label, "files": files})
        assert r.status_code == 200, r.text
        jobs = r.json()["jobs"]
        job_cleanup.extend(j["job_id"] for j in jobs if "job_id" in j)

        listed = {row["filename"]: row for row in
                 client.get("/admin/api/jobs", headers=admin_headers).json()
                 if row["repo"] == repo}
        assert listed["model-Q4_K_M-00001-of-00002.gguf"]["role"] == "main"
        assert listed["mmproj-F16.gguf"]["role"] == "mmproj"
        assert all(row["pkg"] == label for row in listed.values())


# ---------------------------------------------------------------------------
# 6. cancel-package stops every member of one repo+pkg, and nothing else.
# ---------------------------------------------------------------------------

class TestCancelPackageMarksEveryMember:

    def test_only_the_named_package_is_cancelled(
            self, client, admin_headers, models_dir, parked, job_cleanup):
        repo = "vendor/pkg-cancel"
        r1 = client.post("/admin/api/download-package", headers=admin_headers,
                         json={"repo": repo, "label": "Q4_K_M",
                               "files": ["a-00001-of-00002.gguf",
                                        "a-00002-of-00002.gguf"]})
        job_ids = {j["job_id"] for j in r1.json()["jobs"]}
        job_cleanup.extend(job_ids)

        r2 = client.post("/admin/api/download-package", headers=admin_headers,
                         json={"repo": repo, "label": "Q8_0",
                               "files": ["b-Q8_0.gguf"]})
        other_id = r2.json()["jobs"][0]["job_id"]
        job_cleanup.append(other_id)

        r3 = client.post("/admin/api/jobs/cancel-package", headers=admin_headers,
                         json={"repo": repo, "pkg": "Q4_K_M"})
        assert r3.status_code == 200
        cancelled = set(r3.json()["cancelled"])
        assert cancelled == job_ids
        assert job_ids <= gw._download_cancel
        assert other_id not in cancelled
        assert other_id not in gw._download_cancel


# ---------------------------------------------------------------------------
# 7. _package_status: disk truth first, jobs second.
# ---------------------------------------------------------------------------

class TestPackageStatusDiskTruthWinsOverJobRows:

    def test_disk_beats_a_stale_downloading_row(self, models_dir):
        """The gateway restarted mid-package and the worker never got to
        mark the row done. The file is finished on disk regardless -- a
        stale row must never hide that."""
        repo = "vendor/status-stale"
        dest_dir = models_dir / repo.replace("/", "__")
        name = "model-Q4_K_M.gguf"
        _gguf(dest_dir / name, size=104)
        job_id = gw.db_exec(
            "INSERT INTO jobs(created_at,updated_at,repo,filename,status,dest,"
            "pkg,bytes_done,bytes_total) VALUES (?,?,?,?,?,?,?,?,?)",
            (gw.now(), gw.now(), repo, name, "downloading",
             str(dest_dir / name), "Q4_K_M", 10, 10000),
        )
        try:
            status = gw._package_status([{"path": name, "size": 104}], repo)
            assert status["status"] == "done"
            assert status["done"] == status["total"] == 1
            assert status["bytes_done"] == status["bytes_total"] == 104
        finally:
            gw.db_exec("DELETE FROM jobs WHERE id=?", (job_id,))

    def test_zero_job_rows_and_every_file_present_reads_as_done(self, models_dir):
        """A package pulled by hand with plain curl before this feature
        existed must read as done the first time anyone asks, with no jobs
        rows behind it at all."""
        repo = "vendor/status-handpulled"
        dest_dir = models_dir / repo.replace("/", "__")
        names = ["a-Q4_K_M-00001-of-00002.gguf", "a-Q4_K_M-00002-of-00002.gguf"]
        for n in names:
            _gguf(dest_dir / n, size=50)
        status = gw._package_status([{"path": n, "size": 50} for n in names], repo)
        assert status["status"] == "done"
        assert status["done"] == status["total"] == 2

    def test_partial_counts_the_files_not_yet_present(self, models_dir):
        repo = "vendor/status-partial"
        dest_dir = models_dir / repo.replace("/", "__")
        names = ["a-Q4_K_M-00001-of-00002.gguf", "a-Q4_K_M-00002-of-00002.gguf"]
        _gguf(dest_dir / names[0], size=50)
        status = gw._package_status([{"path": n, "size": 50} for n in names], repo)
        assert status["status"] == "partial"
        assert status["done"] == 1 and status["total"] == 2


# ---------------------------------------------------------------------------
# 8. list_local_models(): shard/main/mmproj kinds, summed bytes, complete,
#    mmproj, suggested_id -- real tmp filesystem, tiny fake GGUFs.
# ---------------------------------------------------------------------------

class TestListLocalModelsShardsAndProjector:

    def test_shards_bytes_complete_mmproj_suggested_id(self, models_dir):
        d = models_dir / "org__SomeRepo-GGUF"
        _gguf(d / "SomeRepo-Q4_K_M-00001-of-00003.gguf", b"a", 4096)
        _gguf(d / "SomeRepo-Q4_K_M-00002-of-00003.gguf", b"b", 8192)
        _gguf(d / "SomeRepo-Q4_K_M-00003-of-00003.gguf", b"c", 2048)
        _gguf(d / "mmproj-F16.gguf", b"p", 1024)
        _gguf(d / "mmproj-Q8_0.gguf", b"q", 2048)

        out = {e["name"]: e for e in gw.list_local_models()}
        assert out["SomeRepo-Q4_K_M-00002-of-00003.gguf"]["kind"] == "shard"
        assert out["SomeRepo-Q4_K_M-00003-of-00003.gguf"]["kind"] == "shard"
        assert out["mmproj-F16.gguf"]["kind"] == "mmproj"
        assert out["mmproj-Q8_0.gguf"]["kind"] == "mmproj"

        main = out["SomeRepo-Q4_K_M-00001-of-00003.gguf"]
        assert main["kind"] == "main"
        assert main["bytes"] == 4096 + 8192 + 2048
        assert main["shards"] == 3
        assert main["complete"] is True
        assert main["mmproj"].endswith("mmproj-F16.gguf"), "smallest projector wins"
        assert main["suggested_id"]

    def test_an_incomplete_split_is_still_flagged(self, models_dir):
        d = models_dir / "org__OtherRepo-GGUF"
        _gguf(d / "Other-Q4_K_M-00001-of-00003.gguf", b"a", 4096)
        _gguf(d / "Other-Q4_K_M-00002-of-00003.gguf", b"b", 4096)
        # shard 3 never landed
        out = {e["name"]: e for e in gw.list_local_models()}
        main = out["Other-Q4_K_M-00001-of-00003.gguf"]
        assert main["complete"] is False
        assert main["shards"] == 3

    def test_two_quants_of_one_repo_never_suggest_the_same_id(self, models_dir):
        d = models_dir / "org__ThirdRepo-GGUF"
        _gguf(d / "Third-Q4_K_M.gguf", b"a", 4096)
        _gguf(d / "Third-Q8_0.gguf", b"b", 4096)
        ids = [e["suggested_id"] for e in gw.list_local_models() if e["kind"] == "main"]
        assert len(ids) == 2
        assert len(set(ids)) == 2


# ---------------------------------------------------------------------------
# 9. DELETE .../local-model?shards=1 removes every sibling shard, and the
#    in-use check still applies to the main path.
# ---------------------------------------------------------------------------

class TestDeleteLocalModelWithShards:

    def test_shards_flag_removes_every_sibling_shard(
            self, client, admin_headers, models_dir):
        d = models_dir / "org__Repo-GGUF"
        p1 = _gguf(d / "Repo-Q4_K_M-00001-of-00003.gguf", b"a", 4096)
        _gguf(d / "Repo-Q4_K_M-00002-of-00003.gguf", b"b", 4096)
        _gguf(d / "Repo-Q4_K_M-00003-of-00003.gguf", b"c", 4096)

        r = client.delete("/admin/api/local-model",
                          params={"path": str(p1), "shards": "true"},
                          headers=admin_headers)
        assert r.status_code == 200, r.text
        assert len(r.json()["deleted"]) == 3
        assert not list(d.glob("*.gguf"))

    def test_refuses_to_delete_a_shard_set_still_in_use(
            self, client, admin_headers, models_dir):
        d = models_dir / "org__UsedRepo-GGUF"
        p1 = _gguf(d / "Used-Q4_K_M-00001-of-00002.gguf", b"a", 4096)
        _gguf(d / "Used-Q4_K_M-00002-of-00002.gguf", b"b", 4096)
        gw.save_models([dict(gw.DEFAULT_MODEL_RECORD, id="used-model", path=str(p1))])

        r = client.delete("/admin/api/local-model",
                          params={"path": str(p1), "shards": "true"},
                          headers=admin_headers)
        assert r.status_code == 409
        assert "used-model" in r.json()["detail"]
        assert p1.exists(), "a refused delete must not touch any file"

        gw.save_models([])


# ---------------------------------------------------------------------------
# 10. _fit_for_weight_bytes and local_model_meta() must never quietly drift
#     apart -- local_model_meta() calls the extracted helper and layers its
#     n_cpu_moe override on top.
# ---------------------------------------------------------------------------

class TestFitForWeightBytesAgreesWithLocalModelMeta:

    def _check(self, monkeypatch, models_dir, weights: int, vram: int, expect: str):
        monkeypatch.setattr(gw.platform, "system", lambda: "Linux")
        gguf_path = _gguf(models_dir / "solo" / ("model-%d.gguf" % weights))
        monkeypatch.setattr(
            gw, "model_bytes", lambda p: weights if str(p) == str(gguf_path) else 0
        )
        monkeypatch.setattr(gw, "vram_total_bytes", lambda: vram)
        gw.save_models([dict(gw.DEFAULT_MODEL_RECORD, id="fake-model",
                             path=str(gguf_path), n_cpu_moe=0)])

        direct = gw._fit_for_weight_bytes(weights)
        via_meta = gw.local_model_meta()["fake-model"]["fit"]
        assert direct == via_meta == expect

    def test_agrees_when_it_spills(self, models_dir, monkeypatch):
        # 7 GiB + 1 GiB reserve is past 8 GiB * 0.90 headroom.
        self._check(monkeypatch, models_dir, 7 * gw.GIB, 8 * gw.GIB, "spill")

    def test_agrees_when_it_fits(self, models_dir, monkeypatch):
        self._check(monkeypatch, models_dir, 4 * gw.GIB, 8 * gw.GIB, "vram")


# ---------------------------------------------------------------------------
# 11. /admin/api/hf/quants: the grouped tree fleshed out with fit/disk_ok/
#     status/suggested_id/ollama_tag, against a real captured HF tree.
# ---------------------------------------------------------------------------

class TestHfQuantsEndpoint:

    def test_returns_enriched_quants(
            self, client, admin_headers, models_dir, monkeypatch):
        repo = "ggml-org/gemma-3-27b-it-GGUF"
        fixture = FIXTURES[repo]
        tree_files = [
            {"path": f["path"], "size": f.get("size")}
            for f in fixture["tree"]
            if f.get("type", "file") == "file" and f["path"].endswith(".gguf")
        ]

        async def fake_tree(r):
            assert r == repo
            return tree_files

        async def fake_info(r):
            assert r == repo
            i = fixture["info"]
            g = i["gguf"]
            return {"params": g["total"], "architecture": g["architecture"],
                    "context_length": g["context_length"],
                    "pipeline_tag": i["pipeline_tag"], "downloads": i["downloads"],
                    "likes": i["likes"], "last_modified": i["lastModified"]}

        monkeypatch.setattr(gw, "_hf_gguf_tree", fake_tree)
        monkeypatch.setattr(gw, "_hf_model_info", fake_info)
        monkeypatch.setattr(gw.platform, "system", lambda: "Linux")
        monkeypatch.setattr(gw, "vram_total_bytes", lambda: 8 * gw.GIB)

        r = client.get("/admin/api/hf/quants", params={"repo": repo},
                       headers=admin_headers)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["repo"] == repo
        assert body["upstream_only"] is False
        assert body["model_info"]["architecture"] == "gemma3"
        assert [q["label"] for q in body["quants"]] == ["Q4_K_M", "Q8_0", "f16"]

        for q in body["quants"]:
            assert q["fit"] in ("vram", "spill", "cpu", "unified", "")
            assert isinstance(q["disk_ok"], bool)
            # Flat, the way the page reads it (q.status / q.done / q.total /
            # q.bytes_done / q.bytes_total) -- the first cut nested the whole
            # _package_status() dict under `status`, and every row rendered as
            # "not downloaded" forever because `q.status === "done"` compared a
            # string against an object.
            assert q["status"] == "none", "nothing has been downloaded yet"
            assert (q["done"], q["total"]) == (0, len(q["files"]))
            assert q["bytes_done"] == 0 and q["bytes_total"] == q["bytes"]
            assert q["ollama_tag"] == "hf.co/" + repo + ":" + q["label"]
            assert q["main_dest"]
            assert q["mmproj_dest"] == "", "no projector on disk yet"
            assert q["suggested_id"]
            # '-GGUF' is a format, not a name: 'gemma-3-27b-it-q4_k_m', never
            # 'gemma-3-27b-it-gguf-q4_k_m'.
            assert "gguf" not in q["suggested_id"]
        assert body["quants"][0]["suggested_id"] == "gemma-3-27b-it-q4_k_m"

        ids = [q["suggested_id"] for q in body["quants"]]
        assert len(ids) == len(set(ids)), "two quants must never share a suggested id"

        # Once the projector is on disk, every quant of the repo names it --
        # the path "Add to Models" stages as the record's mmproj. Written
        # where start_download() would put it, so this is the real lookup.
        _gguf(models_dir / "ggml-org__gemma-3-27b-it-GGUF" / "mmproj-model-f16.gguf")
        r = client.get("/admin/api/hf/quants", params={"repo": repo},
                       headers=admin_headers)
        assert r.status_code == 200, r.text
        for q in r.json()["quants"]:
            assert q["mmproj_dest"].endswith("mmproj-model-f16.gguf")
            assert q["status"] == "none", "the projector alone is not the model"

    def test_upstream_only_is_surfaced(
            self, client, admin_headers, models_dir, monkeypatch):
        repo = "ggml-org/gemma-3-27b-it-GGUF"
        fixture = FIXTURES[repo]
        tree_files = [
            {"path": f["path"], "size": f.get("size")}
            for f in fixture["tree"]
            if f.get("type", "file") == "file" and f["path"].endswith(".gguf")
        ]

        async def fake_tree(r):
            return tree_files

        async def fake_info(r):
            return {}

        monkeypatch.setattr(gw, "_hf_gguf_tree", fake_tree)
        monkeypatch.setattr(gw, "_hf_model_info", fake_info)
        monkeypatch.setattr(gw, "UPSTREAM_MODELS", True)

        r = client.get("/admin/api/hf/quants", params={"repo": repo},
                       headers=admin_headers)
        assert r.status_code == 200, r.text
        assert r.json()["upstream_only"] is True


# ---------------------------------------------------------------------------
# 12. the package fan-out gate: with PACKAGE_PARALLEL=1, a second file of
#     the same package must wait for the first to release its slot, staying
#     'queued' with the waiting message -- never starting a competing
#     stream against the same upstream link.
# ---------------------------------------------------------------------------

class TestPackageFanOutGate:

    def test_the_second_file_waits_for_the_first_to_release_the_slot(
            self, models_dir, monkeypatch):
        monkeypatch.setattr(gw, "PACKAGE_PARALLEL", 1)
        monkeypatch.setattr(gw, "lmstudio_publish_one", lambda dest: "")
        release = threading.Event()
        entered = threading.Event()
        attempts: list[int] = []

        def fake_attempt(job_id, url, part, dest, stall_abort=False):
            attempts.append(job_id)
            gw.job_update(job_id, status="downloading")
            entered.set()
            release.wait(10)
            part.parent.mkdir(parents=True, exist_ok=True)
            part.write_bytes(b"x")
            return True

        monkeypatch.setattr(gw, "_download_attempt", fake_attempt)

        repo, label = "vendor/pkg-fanout", "Q4_K_M"
        job_ids: list[int] = []
        try:
            first = gw.start_download(repo, "shard-00001-of-00002.gguf", pkg=label)
            job_ids.append(first["job_id"])
            assert entered.wait(5), "the first job never reached its attempt"

            second = gw.start_download(repo, "shard-00002-of-00002.gguf", pkg=label)
            job_ids.append(second["job_id"])

            deadline = time.time() + 5
            row = None
            while time.time() < deadline:
                rows = gw.db_query(
                    "SELECT status, message FROM jobs WHERE id=?", (second["job_id"],))
                row = rows[0] if rows else None
                if row and row["status"] == "queued" and "waiting for a slot" in (
                        row["message"] or ""):
                    break
                time.sleep(0.1)

            assert row is not None and row["status"] == "queued"
            assert row["message"] == "waiting for a slot: 1 of this package downloading"
            assert len(attempts) == 1, (
                "the second file must not start its own attempt while the "
                "first still holds the package's only slot"
            )
        finally:
            release.set()
            for job_id in job_ids:
                t = gw._download_threads.get(job_id)
                if t:
                    t.join(timeout=5)
                gw._download_cancel.add(job_id)
                gw._download_threads.pop(job_id, None)
            if job_ids:
                gw.db_exec(
                    "DELETE FROM jobs WHERE id IN (%s)" % ",".join("?" * len(job_ids)),
                    job_ids,
                )


# ---------------------------------------------------------------------------
# 13. ENOSPC fast-fails instead of burning the retry budget on a failure
#     mode retrying can never fix.
# ---------------------------------------------------------------------------

class TestEnospcFastFail:

    def test_marks_disk_full_immediately_without_retrying(self, models_dir, monkeypatch):
        attempts: list[int] = []

        def fake_attempt(job_id, url, part, dest, stall_abort=False):
            attempts.append(job_id)
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(gw, "_download_attempt", fake_attempt)
        job = gw.start_download("vendor/pkg-enospc", "model-Q4_K_M.gguf")
        try:
            t = gw._download_threads.get(job["job_id"])
            assert t is not None
            t.join(timeout=5)
            row = gw.db_query("SELECT status, message FROM jobs WHERE id=?",
                              (job["job_id"],))[0]
            assert row["status"] == "error"
            assert row["message"] == "disk full"
            assert len(attempts) == 1, "ENOSPC must not burn the retry budget"
        finally:
            gw._download_cancel.discard(job["job_id"])
            gw._download_threads.pop(job["job_id"], None)
            gw.db_exec("DELETE FROM jobs WHERE id=?", (job["job_id"],))


# ---------------------------------------------------------------------------
# 14. _hf_gguf_tree: pagination, the already-seen-URL terminator, the tree
#     cache, and hf_files()'s shape -- gpt-oss-120b-GGUF's fifty-odd files is
#     the real repo that needed the `next` Link header followed at all.
# ---------------------------------------------------------------------------

class TestHfGgufTreeFollowsPaginationAndCaches:
    """A repo with more entries than one HF tree page holds must not
    quietly lose its later quants; a `next` link that loops back on a URL
    already fetched must end the walk instead of spinning; and a poll
    seconds later (the page refreshes /hf/quants every 10s) must not pay
    for a second HF round trip while the cache is still warm."""

    REPO = "vendor/paginated-repo"
    PAGE2_URL = "http://127.0.0.1:9/page2"

    @staticmethod
    def _response(url, files, next_url=None):
        headers = {}
        if next_url:
            headers["Link"] = '<' + next_url + '>; rel="next"'
        return httpx.Response(
            200, request=httpx.Request("GET", url), headers=headers, json=files,
        )

    @pytest.mark.asyncio
    async def test_two_pages_type_key_cache_and_hf_files_shape(self, monkeypatch):
        page1 = [{"path": "a-Q4_K_M.gguf", "size": 500, "type": "file"}]
        page2 = [{"path": "b-Q8_0.gguf", "size": 100, "type": "file"}]
        calls: list[str] = []
        respond = self._response
        page2_url = self.PAGE2_URL

        class FakeAsyncClient:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url, params=None):
                calls.append(url)
                if url == page2_url:
                    # Its own `next` points back at itself -- already seen
                    # the moment it was queued for a fetch -- so the walk
                    # must stop here, not loop.
                    return respond(url, page2, next_url=page2_url)
                return respond(url, page1, next_url=page2_url)

        monkeypatch.setattr(gw.httpx, "AsyncClient", FakeAsyncClient)
        gw._hf_tree_cache.pop(self.REPO, None)
        try:
            files = await gw._hf_gguf_tree(self.REPO)
            assert len(calls) == 2, "both pages fetched, each exactly once"
            assert {f["path"] for f in files} == {"a-Q4_K_M.gguf", "b-Q8_0.gguf"}
            assert all("type" in f for f in files)

            calls.clear()
            again = await gw._hf_gguf_tree(self.REPO)
            assert calls == [], "a warm cache must never touch the client again"
            assert again == files

            body = await gw.hf_files(self.REPO, admin={})
            assert calls == [], "hf_files() must reuse the same warm tree cache"
            assert all(set(f.keys()) == {"path", "size"} for f in body)
            sizes = [f["size"] for f in body]
            assert sizes == sorted(sizes)
        finally:
            gw._hf_tree_cache.pop(self.REPO, None)


# ---------------------------------------------------------------------------
# 15. api_download_package: shape validation (bad repo id) must fail before
#     the Ollama-managed-box 409 ever gets a chance to quote it back -- see
#     the endpoint's own "Shape before the Ollama refusal" comment.
# ---------------------------------------------------------------------------

class TestDownloadPackageValidationRunsBeforeTheOllamaRefusal:

    def test_bad_repo_id_is_400_even_when_upstream_managed(
            self, client, admin_headers, models_dir, monkeypatch):
        monkeypatch.setattr(gw, "UPSTREAM_MODELS", True)
        r = client.post("/admin/api/download-package", headers=admin_headers,
                        json={"repo": "not a repo!!", "label": "Q4_K_M",
                              "files": ["m.gguf"]})
        assert r.status_code == 400
        assert "repo" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 16. api_download_package refuses a cross-quant basename clash (a root
#     Q4_K_M beside a Q4_K_M/ re-cut sharing a basename would resume onto
#     each other's .part -- see start_download's flattening comment) before
#     queuing anything, but a tree fetch that fails outright must not block
#     the queue -- HF being briefly unreachable is not a reason to refuse a
#     pull it served a minute ago.
# ---------------------------------------------------------------------------

class TestDownloadPackageRefusesACrossQuantBasenameClash:

    def test_409_names_the_other_path_and_queues_nothing(
            self, client, admin_headers, models_dir, monkeypatch):
        repo = "vendor/pkg-tree-clash"
        other_path = "Q4_K_M/Foo-Q4_K_M.gguf"

        async def fake_tree(r):
            assert r == repo
            return [{"path": "Foo-Q4_K_M.gguf", "size": 500, "type": "file"},
                    {"path": other_path, "size": 600, "type": "file"}]

        monkeypatch.setattr(gw, "_hf_gguf_tree", fake_tree)
        r = client.post("/admin/api/download-package", headers=admin_headers,
                        json={"repo": repo, "label": "Q4_K_M",
                              "files": ["Foo-Q4_K_M.gguf"]})
        assert r.status_code == 409, r.text
        assert other_path in r.json()["detail"]
        assert gw.db_query("SELECT id FROM jobs WHERE repo=?", (repo,)) == []

    def test_a_tree_fetch_that_raises_does_not_block_the_queue(
            self, client, admin_headers, models_dir, parked, job_cleanup,
            monkeypatch):
        repo = "vendor/pkg-tree-down"

        async def raising_tree(r):
            raise RuntimeError("HF is unreachable")

        monkeypatch.setattr(gw, "_hf_gguf_tree", raising_tree)
        r = client.post("/admin/api/download-package", headers=admin_headers,
                        json={"repo": repo, "label": "Q4_K_M",
                              "files": ["Foo-Q4_K_M.gguf"]})
        assert r.status_code == 200, r.text
        jobs = r.json()["jobs"]
        job_cleanup.extend(j["job_id"] for j in jobs if "job_id" in j)
        assert len(jobs) == 1 and "job_id" in jobs[0]


# ---------------------------------------------------------------------------
# 17. /admin/api/hf/quants: disk_ok must judge the quant PLUS the projector
#     the Download button queues alongside it, not the quant alone; every
#     quant carries clash_with (blank when nothing collides); and the
#     RVN fixture (no subfolders, no clashes) still comes back as 18 quants
#     nobody has started downloading.
# ---------------------------------------------------------------------------

class TestHfQuantsDiskOkCountsTheProjector:
    REPO = "ggml-org/gemma-3-27b-it-GGUF"

    def _tree_files(self):
        fixture = FIXTURES[self.REPO]
        return [
            {"path": f["path"], "size": f.get("size")}
            for f in fixture["tree"]
            if f.get("type", "file") == "file" and f["path"].endswith(".gguf")
        ]

    def _fetch(self, client, admin_headers, monkeypatch, free_after_incoming):
        tree_files = self._tree_files()

        async def fake_tree(r):
            return tree_files

        async def fake_info(r):
            return {}

        monkeypatch.setattr(gw, "_hf_gguf_tree", fake_tree)
        monkeypatch.setattr(gw, "_hf_model_info", fake_info)
        monkeypatch.setattr(gw.platform, "system", lambda: "Linux")
        monkeypatch.setattr(gw, "vram_total_bytes", lambda: 8 * gw.GIB)
        monkeypatch.setattr(
            gw, "storage_info",
            lambda: {"free_after_incoming": free_after_incoming},
        )
        r = client.get("/admin/api/hf/quants", params={"repo": self.REPO},
                       headers=admin_headers)
        assert r.status_code == 200, r.text
        return r.json()["quants"]

    def test_disk_ok_false_inside_the_gap_true_once_it_closes(
            self, client, admin_headers, models_dir, monkeypatch):
        grouped = gw._group_hf_quants(self._tree_files())
        smallest = min(q["bytes"] for q in grouped["quants"])
        biggest = max(q["bytes"] for q in grouped["quants"])
        mmproj = grouped["mmprojs"][0]["size"]

        # Free space sits between the smallest quant's own bytes and what it
        # would need with its projector too -- the projector is the byte
        # that must tip disk_ok to False, not just the quant.
        short = self._fetch(client, admin_headers, monkeypatch,
                            free_after_incoming=smallest + mmproj // 2)
        assert smallest < smallest + mmproj // 2 < smallest + mmproj
        assert all(q["disk_ok"] is False for q in short)

        plenty = self._fetch(client, admin_headers, monkeypatch,
                             free_after_incoming=biggest + mmproj + 1)
        assert all(q["disk_ok"] is True for q in plenty)

    def test_clash_with_present_and_blank_on_every_quant(
            self, client, admin_headers, models_dir, monkeypatch):
        quants = self._fetch(client, admin_headers, monkeypatch,
                             free_after_incoming=10 ** 15)
        assert quants
        assert all(q["clash_with"] == "" for q in quants)


class TestHfQuantsRvnFixtureStillEighteenQuantsNoneDownloaded:

    def test_eighteen_quants_status_none(
            self, client, admin_headers, models_dir, monkeypatch):
        repo = "0bserverx/RVN-Qwen3.8-Flash-Next-Abliterated-Uncensored-GGUF"
        fixture = FIXTURES[repo]
        tree_files = [
            {"path": f["path"], "size": f.get("size")}
            for f in fixture["tree"]
            if f.get("type", "file") == "file" and f["path"].endswith(".gguf")
        ]

        async def fake_tree(r):
            return tree_files

        async def fake_info(r):
            return {}

        monkeypatch.setattr(gw, "_hf_gguf_tree", fake_tree)
        monkeypatch.setattr(gw, "_hf_model_info", fake_info)
        monkeypatch.setattr(gw.platform, "system", lambda: "Linux")
        monkeypatch.setattr(gw, "vram_total_bytes", lambda: 8 * gw.GIB)

        r = client.get("/admin/api/hf/quants", params={"repo": repo},
                       headers=admin_headers)
        assert r.status_code == 200, r.text
        quants = r.json()["quants"]
        assert len(quants) == 18
        assert all(q["status"] == "none" for q in quants)


# ---------------------------------------------------------------------------
# 18. api_jobs: a package with only its newest member inside the top-50
#     window must still arrive whole, or the Downloads panel's group reads
#     "n of m" off the wrong m -- see the endpoint's own comment.
# ---------------------------------------------------------------------------

class TestJobsWindowIncludesAWholePackageEvenPastFifty:

    def test_both_package_rows_survive_the_window(
            self, client, admin_headers, job_cleanup):
        repo, pkg = "vendor/window", "Q4_K_M"
        old_id = gw.db_exec(
            "INSERT INTO jobs(created_at,updated_at,repo,filename,status,dest,pkg) "
            "VALUES (?,?,?,?,?,?,?)",
            (gw.now(), gw.now(), repo, "a-00001-of-00002.gguf", "done",
             "vendor__window/a-00001-of-00002.gguf", pkg),
        )
        job_cleanup.append(old_id)
        new_id = gw.db_exec(
            "INSERT INTO jobs(created_at,updated_at,repo,filename,status,dest,pkg) "
            "VALUES (?,?,?,?,?,?,?)",
            (gw.now(), gw.now(), repo, "a-00002-of-00002.gguf", "done",
             "vendor__window/a-00002-of-00002.gguf", pkg),
        )
        job_cleanup.append(new_id)

        filler_ids = []
        for i in range(49):
            fid = gw.db_exec(
                "INSERT INTO jobs(created_at,updated_at,repo,filename,status,dest,pkg) "
                "VALUES (?,?,?,?,?,?,?)",
                (gw.now(), gw.now(), "vendor/filler", "f%d.gguf" % i, "done",
                 "vendor__filler/f%d.gguf" % i, ""),
            )
            filler_ids.append(fid)
        job_cleanup.extend(filler_ids)

        # The premise this test relies on: with exactly 49 newer rows ahead
        # of it, the OLDER of the two package rows falls just outside the
        # raw top-50 window -- the same query api_jobs itself runs first.
        raw_ids = {int(r["id"]) for r in
                  gw.db_query("SELECT id FROM jobs ORDER BY id DESC LIMIT 50")}
        assert new_id in raw_ids
        assert old_id not in raw_ids

        rows = client.get("/admin/api/jobs", headers=admin_headers).json()
        ids_present = {int(r["id"]) for r in rows}
        assert old_id in ids_present and new_id in ids_present

        returned_ids = [int(r["id"]) for r in rows]
        assert returned_ids == sorted(returned_ids, reverse=True)


# ---------------------------------------------------------------------------
# 19. DELETE /admin/api/local-model?shards=1: one stuck shard (AV scanner,
#     or a handle llama-server has not released yet) must not swallow the
#     whole request behind an opaque 500 -- every target is attempted
#     regardless, and the message says exactly which one is still there.
# ---------------------------------------------------------------------------

class TestDeleteLocalModelPartialShardFailure:

    def test_names_the_stuck_shard_and_still_deletes_the_rest(
            self, client, admin_headers, models_dir, monkeypatch):
        d = models_dir / "org__Stuck-GGUF"
        p1 = _gguf(d / "Stuck-Q4_K_M-00001-of-00003.gguf", b"a", 4096)
        p2 = _gguf(d / "Stuck-Q4_K_M-00002-of-00003.gguf", b"b", 4096)
        p3 = _gguf(d / "Stuck-Q4_K_M-00003-of-00003.gguf", b"c", 4096)

        real_unlink = Path.unlink

        def flaky_unlink(self, *a, **kw):
            if self.name == p2.name:
                raise PermissionError(13, "Access is denied")
            return real_unlink(self, *a, **kw)

        monkeypatch.setattr(Path, "unlink", flaky_unlink)

        r = client.delete("/admin/api/local-model",
                          params={"path": str(p1), "shards": "true"},
                          headers=admin_headers)
        assert r.status_code == 500
        detail = r.json()["detail"]
        assert p2.name in detail
        assert "deleted 2 of 3" in detail
        assert not p1.exists()
        assert p2.exists()
        assert not p3.exists()


# ---------------------------------------------------------------------------
# 20. start_download's re-drive of a row that outlived its worker must
#     rejoin the package the row is ALREADY tagged with -- not whichever pkg
#     (or none) this particular caller happens to pass -- or the re-driven
#     shard silently drops the gate and races its siblings again.
# ---------------------------------------------------------------------------

class TestStartDownloadRedriveKeepsThePersistedPkg:

    def test_the_redriven_worker_gets_the_rows_own_packages_gate(
            self, models_dir, monkeypatch, job_cleanup):
        recorded: dict = {}

        def stub_worker(job_id, repo, filename, dest, gate=None, gate_key=""):
            recorded["gate"] = gate
            recorded["gate_key"] = gate_key
            gw.job_update(job_id, status="downloading")

        monkeypatch.setattr(gw, "_download_worker", stub_worker)

        repo = "vendor/redrive"
        filename = "model-Q4_K_M-00001-of-00002.gguf"
        pkg = "Q4_K_M"
        key = repo + "::" + pkg
        try:
            first = gw.start_download(repo, filename, pkg=pkg)
            job_cleanup.append(first["job_id"])
            t1 = gw._download_threads.get(first["job_id"])
            if t1:
                t1.join(timeout=5)
            assert gw.db_query("SELECT pkg FROM jobs WHERE id=?",
                               (first["job_id"],))[0]["pkg"] == pkg
            # The row outlived its worker (a gateway restart, or -- as here
            # -- the stub simply never keeping a thread alive).
            gw._download_threads.pop(first["job_id"], None)

            recorded.clear()
            second = gw.start_download(repo, filename)   # NO pkg passed
            assert second["job_id"] == first["job_id"]
            t2 = gw._download_threads.get(second["job_id"])
            if t2:
                t2.join(timeout=5)

            expected_gate = gw._pkg_gate_for(repo, pkg)
            assert recorded["gate"] is expected_gate
            assert recorded["gate_key"] == key
        finally:
            gw._pkg_gates.pop(key, None)
            gw._pkg_gate_refs.pop(key, None)


# ---------------------------------------------------------------------------
# 21. _pkg_gate_take/_pkg_gate_drop: one Semaphore per package alive for
#     exactly as long as a worker still references it, not one per package
#     ever queued for the life of the process (see _pkg_gate_take's own
#     docstring).
# ---------------------------------------------------------------------------

class TestPkgGateReferenceCounting:

    def test_two_takes_two_drops_removes_the_key(self):
        repo, pkg = "vendor/gate-refs", "Q4_K_M"
        key = repo + "::" + pkg
        try:
            gate1, key1 = gw._pkg_gate_take(repo, pkg)
            gate2, key2 = gw._pkg_gate_take(repo, pkg)
            assert key1 == key2 == key
            assert gate1 is gate2
            assert key in gw._pkg_gates and key in gw._pkg_gate_refs

            gw._pkg_gate_drop(key)
            assert key in gw._pkg_gates, (
                "one worker still holds a reference -- the gate must survive"
            )
            assert gw._pkg_gate_for(repo, pkg) is gate1, (
                "still the same Semaphore, not a freshly minted one"
            )

            gw._pkg_gate_drop(key)
            assert key not in gw._pkg_gates
            assert key not in gw._pkg_gate_refs
        finally:
            gw._pkg_gates.pop(key, None)
            gw._pkg_gate_refs.pop(key, None)

    def test_blank_pkg_never_makes_a_gate(self):
        assert gw._pkg_gate_take("vendor/no-pkg", "") == (None, "")

    def test_dropping_a_blank_key_is_a_no_op(self):
        before_gates = dict(gw._pkg_gates)
        before_refs = dict(gw._pkg_gate_refs)
        gw._pkg_gate_drop("")
        assert gw._pkg_gates == before_gates
        assert gw._pkg_gate_refs == before_refs


# ---------------------------------------------------------------------------
# 22. resume_orphaned_downloads: every row of one repo+pkg must rebuild the
#     SAME gate, not a fresh one per row -- a restart mid-package must not
#     let its shards start racing each other again.
# ---------------------------------------------------------------------------

class TestResumeOrphanedDownloadsRebuildsOneGatePerPackage:

    def test_both_rows_get_the_same_gate_object(
            self, models_dir, monkeypatch, job_cleanup):
        repo, pkg = "vendor/resume-gate", "Q4_K_M"
        dest_dir = models_dir / repo.replace("/", "__")
        names = ["model-Q4_K_M-00001-of-00002.gguf",
                 "model-Q4_K_M-00002-of-00002.gguf"]
        ids = []
        for n in names:
            jid = gw.db_exec(
                "INSERT INTO jobs(created_at,updated_at,repo,filename,status,dest,pkg) "
                "VALUES (?,?,?,?,?,?,?)",
                (gw.now(), gw.now(), repo, n, "downloading",
                 str(dest_dir / n), pkg),
            )
            ids.append(jid)
        job_cleanup.extend(ids)

        recorded: dict[int, dict] = {}

        def stub_worker(job_id, repo, filename, dest, gate=None, gate_key=""):
            recorded[job_id] = {"gate": gate, "gate_key": gate_key}

        monkeypatch.setattr(gw, "_download_worker", stub_worker)

        key = repo + "::" + pkg
        try:
            gw.resume_orphaned_downloads()
            for jid in ids:
                t = gw._download_threads.get(jid)
                if t:
                    t.join(timeout=5)
                gw._download_threads.pop(jid, None)

            assert set(ids) <= set(recorded), "both rows must have been resumed"
            gates = {recorded[jid]["gate"] for jid in ids}
            keys = {recorded[jid]["gate_key"] for jid in ids}
            assert len(gates) == 1
            assert gates.pop() is gw._pkg_gate_for(repo, pkg)
            assert keys == {key}
        finally:
            gw._pkg_gates.pop(key, None)
            gw._pkg_gate_refs.pop(key, None)


# ---------------------------------------------------------------------------
# 23. A cancel that arrives while a job is queued behind the package gate --
#     never having started its own attempt -- must still land, and the gate
#     must still be dropped once both workers have exited. Extends
#     TestPackageFanOutGate's approach with a cancel partway through the
#     wait.
# ---------------------------------------------------------------------------

class TestCancelWhileWaitingOnThePackageGate:

    def test_the_waiting_job_cancels_and_the_gate_is_dropped(
            self, models_dir, monkeypatch):
        monkeypatch.setattr(gw, "PACKAGE_PARALLEL", 1)
        gw._pkg_gates.clear()
        monkeypatch.setattr(gw, "lmstudio_publish_one", lambda dest: "")
        release = threading.Event()
        entered = threading.Event()

        def fake_attempt(job_id, url, part, dest, stall_abort=False):
            entered.set()
            release.wait(10)
            part.parent.mkdir(parents=True, exist_ok=True)
            part.write_bytes(b"x")
            return True

        monkeypatch.setattr(gw, "_download_attempt", fake_attempt)

        repo, label = "vendor/pkg-cancel-wait", "Q4_K_M"
        key = repo + "::" + label
        job_ids: list[int] = []
        try:
            first = gw.start_download(repo, "shard-00001-of-00002.gguf", pkg=label)
            job_ids.append(first["job_id"])
            assert entered.wait(5), "the first job never reached its attempt"

            second = gw.start_download(repo, "shard-00002-of-00002.gguf", pkg=label)
            job_ids.append(second["job_id"])

            deadline = time.time() + 5
            row = None
            while time.time() < deadline:
                rows = gw.db_query(
                    "SELECT status, message FROM jobs WHERE id=?", (second["job_id"],))
                row = rows[0] if rows else None
                if row and row["status"] == "queued" and "waiting for a slot" in (
                        row["message"] or ""):
                    break
                time.sleep(0.1)
            assert row is not None and row["status"] == "queued", (
                "the second job never reached the 'waiting for a slot' state"
            )

            gw._download_cancel.add(second["job_id"])
            deadline = time.time() + 5
            cancelled = False
            while time.time() < deadline:
                rows = gw.db_query(
                    "SELECT status FROM jobs WHERE id=?", (second["job_id"],))
                if rows and rows[0]["status"] == "cancelled":
                    cancelled = True
                    break
                time.sleep(0.1)
            assert cancelled, "the waiting job never noticed its cancel"

            release.set()
            for job_id in job_ids:
                t = gw._download_threads.get(job_id)
                if t:
                    t.join(timeout=5)

            assert key not in gw._pkg_gates, (
                "both workers have exited -- the gate must be forgotten"
            )
        finally:
            release.set()
            for job_id in job_ids:
                t = gw._download_threads.get(job_id)
                if t:
                    t.join(timeout=5)
                gw._download_cancel.add(job_id)
                gw._download_threads.pop(job_id, None)
            if job_ids:
                gw.db_exec(
                    "DELETE FROM jobs WHERE id IN (%s)" % ",".join("?" * len(job_ids)),
                    job_ids,
                )
            gw._pkg_gates.pop(key, None)
            gw._pkg_gate_refs.pop(key, None)


# ---------------------------------------------------------------------------
# 24. list_local_models()'s `in_models` field: the record already serving a
#     main entry's file, keyed by path or by inode (see _configured_files'
#     docstring). This is what "Add all on-disk models" and the on-disk
#     list's "in Models as ..." badge both check before ever comparing ids
#     by hand -- get this wrong and either a registered model gets offered
#     for re-adding, or an unregistered one silently never is.
# ---------------------------------------------------------------------------

class TestListLocalModelsInModelsField:

    def test_a_saved_record_marks_the_main_entry_in_models(self, models_dir):
        d = models_dir / "org__Repo-GGUF"
        p = _gguf(d / "Repo-Q4_K_M.gguf")
        gw.save_models([dict(gw.DEFAULT_MODEL_RECORD, id="repo-q4", path=str(p))])
        try:
            out = {e["name"]: e for e in gw.list_local_models()}
            assert out["Repo-Q4_K_M.gguf"]["in_models"] == "repo-q4"
        finally:
            gw.save_models([])

    def test_nothing_configured_reads_as_blank(self, models_dir):
        d = models_dir / "org__Lonely-GGUF"
        _gguf(d / "Lonely-Q4_K_M.gguf")
        out = {e["name"]: e for e in gw.list_local_models()}
        assert out["Lonely-Q4_K_M.gguf"]["in_models"] == ""

    def test_every_record_serving_the_file_is_named_disabled_ones_included(
            self, models_dir):
        """Cloning a record to serve the same weights under a second profile
        is a supported pattern, so "in Models as <the first one>" would read
        as the only registration. And a DISABLED record still counts: gpu-laptop-3
        keeps its duplicate Lightning records switched off on purpose, and an
        add-everything pass that ignored them would register the file again."""
        d = models_dir / "org__Twice-GGUF"
        p = _gguf(d / "Twice-Q4_K_M.gguf")
        gw.save_models([
            dict(gw.DEFAULT_MODEL_RECORD, id="twice-chat", path=str(p)),
            dict(gw.DEFAULT_MODEL_RECORD, id="twice-classify", path=str(p),
                 enabled=False),
        ])
        try:
            out = {e["name"]: e for e in gw.list_local_models()}
            assert out["Twice-Q4_K_M.gguf"]["in_models"] == "twice-chat, twice-classify"
        finally:
            gw.save_models([])


class TestInModelsSurvivesPathSpelling:
    """A record saved with forward slashes, or (on Windows, where the
    filesystem is case-insensitive) a different case on the drive and
    directories, must still match -- _configured_files() promises
    os.path.normcase(os.path.normpath(...)), not a literal string compare.
    ntpath.normcase both lower-cases and turns '/' into '\\\\', so the two
    differences collapse together there; posixpath.normcase is a no-op and
    POSIX paths never carry backslashes to begin with, so only the slash
    swap could apply there and it is already a no-op on an all-forward-slash
    path -- there is nothing case-related to also flip."""

    def test_forward_slashes_and_case_still_match(self, models_dir):
        d = models_dir / "org__Case-GGUF"
        p = _gguf(d / "Case-Q4_K_M.gguf")
        variant = str(p).replace("\\", "/")
        if os.name == "nt":
            variant = variant.upper()
        gw.save_models([dict(gw.DEFAULT_MODEL_RECORD, id="case-q4", path=variant)])
        try:
            out = {e["name"]: e for e in gw.list_local_models()}
            assert out["Case-Q4_K_M.gguf"]["in_models"] == "case-q4"
        finally:
            gw.save_models([])


class TestInModelsMatchesByInode:
    """A fleet GGUF hard-linked into LM Studio's layout and registered AT
    ITS LM STUDIO PATH (see _configured_files' own docstring) is still the
    same file on disk. The inode half of the match is what stops "add all
    on-disk models" from registering it a second time under MODELS_DIR."""

    def test_a_record_pointing_at_a_hard_link_matches_the_models_dir_copy(
            self, models_dir, tmp_path):
        d = models_dir / "org__Linked-GGUF"
        p = _gguf(d / "Linked-Q4_K_M.gguf")
        link_dir = tmp_path / "lmstudio" / "org" / "Linked-GGUF"
        link_dir.mkdir(parents=True)
        link = link_dir / "Linked-Q4_K_M.gguf"
        try:
            os.link(p, link)
        except OSError:
            pytest.skip("this filesystem does not support hard links")

        gw.save_models(
            [dict(gw.DEFAULT_MODEL_RECORD, id="linked-q4", path=str(link))]
        )
        try:
            out = {e["name"]: e for e in gw.list_local_models()}
            assert out["Linked-Q4_K_M.gguf"]["in_models"] == "linked-q4"
        finally:
            gw.save_models([])


class TestInModelsDanglingRecordPath:
    """A record whose path no longer exists on disk (the file was deleted
    by hand, or a peer's models.json entry) must not crash the whole
    listing, and must not accidentally match some other file by inode --
    os.stat() on a missing path raises, and _configured_files() must treat
    that as 'nothing to key by', not propagate the exception."""

    def test_a_missing_record_path_does_not_crash_or_match(self, models_dir):
        d = models_dir / "org__Real-GGUF"
        _gguf(d / "Real-Q4_K_M.gguf")
        gw.save_models([dict(
            gw.DEFAULT_MODEL_RECORD, id="ghost",
            path=str(models_dir / "nowhere" / "ghost.gguf"),
        )])
        try:
            out = {e["name"]: e for e in gw.list_local_models()}
            assert out["Real-Q4_K_M.gguf"]["in_models"] == ""
        finally:
            gw.save_models([])


class TestAdminApiModelsLocalFilesCarryInModelsKey:

    def test_every_main_entry_carries_the_key(
            self, client, admin_headers, models_dir):
        d = models_dir / "org__Api-GGUF"
        _gguf(d / "Api-Q4_K_M.gguf")
        r = client.get("/admin/api/models", headers=admin_headers)
        assert r.status_code == 200, r.text
        local_files = r.json()["local_files"]
        mains = [e for e in local_files if e["kind"] == "main"]
        assert mains, "the fixture's own file must show up as a main entry"
        assert all("in_models" in e for e in mains)


class TestInModelsOnlyAppliesToMainEntries:

    def test_shards_and_mmproj_skip_the_key_and_an_incomplete_split_is_blank(
            self, models_dir):
        d = models_dir / "org__Split-GGUF"
        _gguf(d / "Split-Q4_K_M-00001-of-00003.gguf", b"a", 4096)
        _gguf(d / "Split-Q4_K_M-00002-of-00003.gguf", b"b", 4096)
        # Shard 3 never landed -- an incomplete split.
        _gguf(d / "mmproj-F16.gguf", b"p", 1024)

        out = {e["name"]: e for e in gw.list_local_models()}
        assert "in_models" not in out["Split-Q4_K_M-00002-of-00003.gguf"]
        assert "in_models" not in out["mmproj-F16.gguf"]

        main = out["Split-Q4_K_M-00001-of-00003.gguf"]
        assert main["complete"] is False
        assert main["in_models"] == ""
