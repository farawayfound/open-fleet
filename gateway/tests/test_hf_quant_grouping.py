"""Pure grouping tests for _group_hf_quants/_label_hf_quant_groups/_hf_file_kind
-- the function the whole Library redesign hinges on (see SPEC.md's "one row
per quant level" result). Every fixture repo here is a REAL HuggingFace tree
capture (gateway/tests/hf_trees.json), not a hand-rolled stand-in: the
naming inconsistencies these functions have to survive (a repo that puts its
quants in subfolders, one that flattens them, one whose common prefix would
blank a label, imatrix/MTP files sitting right next to real quants) are real
shapes that shipped on real repos, not edge cases invented for the test.

_group_hf_quants is pure and I/O-free by design, so every test here drives it
straight off data -- no client, no filesystem, no network.

Run: python -m pytest gateway/tests/test_hf_quant_grouping.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import app as gw

FIXTURES = json.loads(
    (Path(__file__).parent / "hf_trees.json").read_text(encoding="utf-8")
)["repos"]


def _quants(repo: str) -> list[dict]:
    return gw._group_hf_quants(FIXTURES[repo]["tree"])["quants"]


def _grouped(repo: str) -> dict:
    return gw._group_hf_quants(FIXTURES[repo]["tree"])


# ---------------------------------------------------------------------------
# 1. RVN: one repo, one directory, 18 quants, every one an 8-shard split with
#    every shard present. Nothing to exclude, nothing to project with -- this
#    is the plain case every other repo below adds one wrinkle to.
# ---------------------------------------------------------------------------

class TestFlatEightShardRepo:
    REPO = "0bserverx/RVN-Qwen3.8-Flash-Next-Abliterated-Uncensored-GGUF"

    def test_eighteen_complete_eight_shard_groups(self):
        quants = _quants(self.REPO)
        assert len(quants) == 18
        for q in quants:
            assert q["shards"] == 8
            assert q["expected_shards"] == 8
            assert q["complete"] is True
            assert q["dir"] == ""
            assert q["main"].endswith("-00001-of-00008.gguf")

    def test_labels_are_the_bare_quant_tokens(self):
        labels = [q["label"] for q in _quants(self.REPO)]
        # Every filename shares "RVN-Qwen3.8-Flash-Next-" as a prefix, so the
        # common-prefix strip (not the _QUANT_TOKEN fallback) is what is
        # under test here.
        assert labels[0] == "Q2_K"
        assert labels[-1] == "Q6_K"
        assert "IQ3_M" in labels and "Q4_K_M" in labels
        assert len(set(labels)) == 18, "every label must be unique"

    def test_no_exclusions_no_projectors(self):
        g = _grouped(self.REPO)
        assert g["excluded"] == []
        assert g["mmprojs"] == []

    def test_sorted_by_bytes_ascending(self):
        sizes = [q["bytes"] for q in _quants(self.REPO)]
        assert sizes == sorted(sizes)


# ---------------------------------------------------------------------------
# 2. unsloth/Qwen3.8-27B-GGUF: flat quants beside a BF16/ subfolder split,
#    an MTP/ draft head and a root imatrix file that must be excluded (never
#    silently dropped), and two mmproj precisions that only the ROOT quants
#    should ever be offered -- a BF16 conversion pulled from the BF16/
#    subfolder should never show a projector that lives one level up in a
#    different logical bundle.
# ---------------------------------------------------------------------------

class TestFlatQuantsBesideASubfolderSplit:
    REPO = "unsloth/Qwen3.8-27B-GGUF"

    def test_twenty_five_groups_one_of_them_the_bf16_subfolder(self):
        quants = _quants(self.REPO)
        assert len(quants) == 25
        bf16 = [q for q in quants if q["dir"] == "BF16"]
        assert len(bf16) == 1
        assert bf16[0]["label"] == "BF16"
        assert bf16[0]["shards"] == 2 and bf16[0]["complete"] is True
        root = [q for q in quants if q["dir"] != "BF16"]
        assert len(root) == 24
        assert all(q["shards"] == 1 for q in root)

    def test_mtp_and_imatrix_are_excluded_with_reasons_not_dropped(self):
        excluded = {e["path"]: e["reason"] for e in _grouped(self.REPO)["excluded"]}
        assert excluded["MTP/mtp-Qwen3.8-27B-Q4_0.gguf"] == (
            "speculative-decoding draft head, not a full model")
        assert excluded["imatrix_unsloth.gguf"] == (
            "importance matrix, not a model")
        # Neither ever became its own bogus one-file "quant".
        labels = [q["label"] for q in _quants(self.REPO)]
        assert not any("mtp" in lbl.lower() or "imatrix" in lbl.lower() for lbl in labels)

    def test_mmproj_choices_smallest_first_root_only(self):
        quants = _quants(self.REPO)
        root = [q for q in quants if q["dir"] != "BF16"]
        bf16 = [q for q in quants if q["dir"] == "BF16"][0]
        for q in root:
            paths = [m["path"] for m in q["mmproj_choices"]]
            assert paths == ["mmproj-F16.gguf", "mmproj-BF16.gguf"], (
                "smallest projector (927607488 bytes) must be offered first")
        assert bf16["mmproj_choices"] == [], (
            "the BF16 subfolder split shares no directory with the "
            "root-level projectors and must not be offered them"
        )

    def test_repo_level_mmprojs_list_is_smallest_first_too(self):
        mmprojs = _grouped(self.REPO)["mmprojs"]
        assert [m["path"] for m in mmprojs] == ["mmproj-F16.gguf", "mmproj-BF16.gguf"]


# ---------------------------------------------------------------------------
# 3. unsloth/DeepSeek-V3-0324-GGUF: every quant lives in its own subfolder,
#    with shard counts that vary quant to quant (4 through 30) -- nothing to
#    exclude, nothing to project with, just "one group per folder" at scale.
# ---------------------------------------------------------------------------

class TestOneGroupPerSubfolder:
    REPO = "unsloth/DeepSeek-V3-0324-GGUF"

    def test_thirteen_subfolder_groups_all_complete(self):
        quants = _quants(self.REPO)
        assert len(quants) == 13
        dirs = {q["dir"] for q in quants}
        assert "" not in dirs, "every quant here lives in its own subfolder"
        assert len(dirs) == 13, "one directory per quant, no sharing"
        for q in quants:
            assert q["complete"] is True
            assert q["main"] != ""

    def test_labels_match_their_directory_names(self):
        for q in _quants(self.REPO):
            assert q["label"] == q["dir"], (
                "a repo whose subfolder IS the quant name should not need "
                "the _QUANT_TOKEN fallback to recover it"
            )

    def test_no_exclusions_no_projectors(self):
        g = _grouped(self.REPO)
        assert g["excluded"] == []
        assert g["mmprojs"] == []


# ---------------------------------------------------------------------------
# 4. bartowski/Qwen_Qwen3-30B-A3B-GGUF: 26 flat single-file quants plus one
#    bf16 conversion split into its own (lower-cased) subfolder -- the
#    duplicate-looking "bf16" label must come through as-is, not blanked or
#    collapsed onto a sibling.
# ---------------------------------------------------------------------------

class TestFlatRepoWithOneLowercaseSubfolder:
    REPO = "bartowski/Qwen_Qwen3-30B-A3B-GGUF"

    def test_twenty_seven_groups(self):
        assert len(_quants(self.REPO)) == 27

    def test_the_subfolder_bf16_quant_keeps_its_own_label(self):
        quants = _quants(self.REPO)
        bf16 = [q for q in quants if q["dir"] != ""]
        assert len(bf16) == 1
        assert bf16[0]["label"] == "bf16"
        assert bf16[0]["shards"] == 2
        assert bf16[0]["complete"] is True

    def test_flat_quants_are_all_single_file(self):
        flat = [q for q in _quants(self.REPO) if q["dir"] == ""]
        assert len(flat) == 26
        assert all(q["shards"] == 1 for q in flat)
        labels = {q["label"] for q in flat}
        assert {"IQ2_XXS", "Q4_K_M", "Q8_0"} <= labels


# ---------------------------------------------------------------------------
# 5. ggml-org/gemma-3-27b-it-GGUF: a small, flat, vision repo -- three
#    quants (one of them a 2-shard f16) all sharing the one projector in the
#    repo root.
# ---------------------------------------------------------------------------

class TestVisionRepoSharesOneRootProjector:
    REPO = "ggml-org/gemma-3-27b-it-GGUF"

    def test_three_quants_labelled_q4_k_m_q8_0_f16(self):
        labels = [q["label"] for q in _quants(self.REPO)]
        assert labels == ["Q4_K_M", "Q8_0", "f16"]

    def test_the_f16_quant_is_the_two_shard_one(self):
        quants = _quants(self.REPO)
        f16 = next(q for q in quants if q["label"] == "f16")
        assert f16["shards"] == 2 and f16["complete"] is True
        others = [q for q in quants if q["label"] != "f16"]
        assert all(q["shards"] == 1 for q in others)

    def test_every_quant_offers_the_one_projector(self):
        for q in _quants(self.REPO):
            assert [m["path"] for m in q["mmproj_choices"]] == ["mmproj-model-f16.gguf"]


# ---------------------------------------------------------------------------
# 6. unsloth/gpt-oss-120b-GGUF: one group per folder again, but at a size
#    (fifty-odd files) that only a paginated HF tree fetch would return in
#    full -- the grouping side of that has to hold regardless of how the
#    tree arrived.
# ---------------------------------------------------------------------------

class TestManySubfolderGroups:
    REPO = "unsloth/gpt-oss-120b-GGUF"

    def test_sixteen_groups_one_of_them_flat(self):
        quants = _quants(self.REPO)
        assert len(quants) == 16
        flat = [q for q in quants if q["dir"] == ""]
        assert len(flat) == 1
        assert flat[0]["label"] == "F16"
        assert flat[0]["shards"] == 1

    def test_subfolder_quants_are_two_shard_splits(self):
        subfolder = [q for q in _quants(self.REPO) if q["dir"] != ""]
        assert len(subfolder) == 15
        assert all(q["shards"] == 2 and q["complete"] for q in subfolder)
        assert all(q["label"] == q["dir"] for q in subfolder)


# ---------------------------------------------------------------------------
# 7. Synthetic cases the real fixtures above do not happen to exercise.
# ---------------------------------------------------------------------------

class TestIncompleteShardSetsAreListedNotHidden:
    """A repo missing an interior shard, or missing its first shard entirely,
    is still one real row -- just one the frontend must refuse to offer a
    Download button for (see the spec's !complete handling). Silently
    dropping it would make a broken upload look like it was never there."""

    def test_missing_an_interior_shard_is_incomplete_but_still_listed(self):
        files = [
            {"path": "org/repo/model-Q4_K_M-00001-of-00004.gguf", "size": 100},
            {"path": "org/repo/model-Q4_K_M-00002-of-00004.gguf", "size": 100},
            # shard 3 never uploaded
            {"path": "org/repo/model-Q4_K_M-00004-of-00004.gguf", "size": 100},
        ]
        quants = gw._group_hf_quants(files)["quants"]
        assert len(quants) == 1
        q = quants[0]
        assert q["complete"] is False
        assert q["shards"] == 3 and q["expected_shards"] == 4
        assert q["main"] != "", "shard 1 is present, so main must still resolve"

    def test_missing_the_first_shard_leaves_main_blank(self):
        files = [
            {"path": "model-Q4_K_M-00002-of-00004.gguf", "size": 100},
            {"path": "model-Q4_K_M-00003-of-00004.gguf", "size": 100},
            {"path": "model-Q4_K_M-00004-of-00004.gguf", "size": 100},
        ]
        quants = gw._group_hf_quants(files)["quants"]
        assert len(quants) == 1
        q = quants[0]
        assert q["complete"] is False
        assert q["main"] == "", "no shard 1 on HF means no downloadable main file"
        assert q["shards"] == 3


class TestLabelFallbackOnASingleQuantRepo:
    """A repo with only one quant on offer has no sibling to strip a common
    prefix against, so the label falls back to the trailing _QUANT_TOKEN --
    the whole reason that regex exists."""

    def test_single_group_repo_keeps_a_label_via_quant_token(self):
        files = [{"path": "Foo-7B-Q4_K_M.gguf", "size": 500}]
        quants = gw._group_hf_quants(files)["quants"]
        assert len(quants) == 1
        assert quants[0]["label"] == "Q4_K_M"

    def test_dotted_naming_still_finds_the_token(self):
        files = [
            {"path": "llama-2-7b.Q4_K_M.gguf", "size": 500},
            {"path": "llama-2-7b.Q8_0.gguf", "size": 900},
        ]
        quants = gw._group_hf_quants(files)["quants"]
        labels = {q["label"] for q in quants}
        assert labels == {"Q4_K_M", "Q8_0"}

    def test_a_base_with_no_recognisable_quant_token_falls_back_to_the_raw_base(self):
        files = [{"path": "some-random-file-name.gguf", "size": 500}]
        quants = gw._group_hf_quants(files)["quants"]
        assert quants[0]["label"] == "some-random-file-name", (
            "never blank -- the raw base is the last resort"
        )


class TestASharedTokenPrefixSurvivesTheCommonPrefixStrip:
    """unsloth's dynamic quants all start with 'UD-'. A repo offering only
    those ('Foo-UD-Q4_K_XL' beside 'Foo-UD-Q5_K_XL') has 'Foo-UD-' as the
    common prefix, and stripping it back to the last '-' would advertise a
    'Q4_K_XL' no file in the repo is called -- wrong on the row, and wrong in
    the ollama tag built from it. The trailing token wins when it is the
    stripped label with more in front of it."""

    def test_ud_prefix_is_kept_when_every_quant_carries_it(self):
        files = [
            {"path": "Foo-7B-UD-Q4_K_XL.gguf", "size": 500},
            {"path": "Foo-7B-UD-Q5_K_XL.gguf", "size": 600},
        ]
        quants = gw._group_hf_quants(files)["quants"]
        assert [q["label"] for q in quants] == ["UD-Q4_K_XL", "UD-Q5_K_XL"]

    def test_but_a_variant_name_the_token_does_not_know_stays(self):
        # 'Instruct-Q4_K_M' is what tells the two apart; the token alone
        # ('Q4_K_M') would collapse them into one label.
        files = [
            {"path": "Foo-Instruct-Q4_K_M.gguf", "size": 500},
            {"path": "Foo-Base-Q4_K_M.gguf", "size": 600},
        ]
        quants = gw._group_hf_quants(files)["quants"]
        assert {q["label"] for q in quants} == {"Instruct-Q4_K_M", "Base-Q4_K_M"}


class TestDuplicateLabelRootVsSubfolder:
    """A quant re-cut into its own subfolder strips to the same label as an
    identically-named flat file above it -- the root copy keeps the bare
    label (it sorts first) and the subfolder copy is the one tagged with
    where it lives, deterministically."""

    def test_the_subfolder_copy_gets_the_dir_suffix(self):
        files = [
            {"path": "Foo-Q4_K_M.gguf", "size": 500},
            {"path": "Q4_K_M/Foo-Q4_K_M.gguf", "size": 600},
        ]
        quants = gw._group_hf_quants(files)["quants"]
        by_dir = {q["dir"]: q["label"] for q in quants}
        assert by_dir[""] == "Q4_K_M"
        assert by_dir["Q4_K_M"] == "Q4_K_M (Q4_K_M/)"


class TestSortedByBytesAscending:
    def test_sort_ignores_file_order(self):
        files = [
            {"path": "Foo-Q8_0.gguf", "size": 900},
            {"path": "Foo-Q2_K.gguf", "size": 100},
        ]
        quants = gw._group_hf_quants(files)["quants"]
        assert [q["label"] for q in quants] == ["Q2_K", "Q8_0"]


class TestNonFileAndNonGgufEntriesAreIgnored:
    """The raw HF tree api hands back directory rows too (size 0, type
    'directory') -- these must never become a quant with a bogus 0-byte
    file, and a stray non-.gguf file (README, .gitattributes) must never
    even reach _hf_file_kind."""

    def test_directory_rows_and_non_gguf_files_produce_nothing(self):
        files = [
            {"type": "directory", "path": "Q4_K_M", "size": 0},
            {"type": "file", "path": "README.md", "size": 8472},
            {"type": "file", "path": ".gitattributes", "size": 1234},
            {"type": "file", "path": "Q4_K_M/model-Q4_K_M.gguf", "size": 500},
        ]
        g = gw._group_hf_quants(files)
        assert len(g["quants"]) == 1
        assert g["excluded"] == [], (
            "a non-.gguf file is dropped silently, never surfaced as excluded"
        )


# ---------------------------------------------------------------------------
# 8. clash_with: two groups whose files share a basename (a root Q4_K_M
#    beside a Q4_K_M/ re-cut of it) cannot both be downloaded -- every dest
#    flattens to its basename (see start_download), so the second would
#    resume into, or overwrite, the first's .part. Each such group must name
#    the other's label so the row can say why it has no Download button; a
#    repo with no such collision must read clash_with == "" on every group.
# ---------------------------------------------------------------------------

class TestClashWithNamesTheCollidingGroup:

    def test_both_groups_name_each_others_label(self):
        files = [
            {"path": "Foo-Q4_K_M.gguf", "size": 500},
            {"path": "Q4_K_M/Foo-Q4_K_M.gguf", "size": 600},
        ]
        by_label = {q["label"]: q for q in gw._group_hf_quants(files)["quants"]}
        assert by_label["Q4_K_M"]["clash_with"] == "Q4_K_M (Q4_K_M/)"
        assert by_label["Q4_K_M (Q4_K_M/)"]["clash_with"] == "Q4_K_M"

    def test_the_real_rvn_fixture_has_no_collisions_at_all(self):
        quants = _quants(
            "0bserverx/RVN-Qwen3.8-Flash-Next-Abliterated-Uncensored-GGUF")
        assert len(quants) == 18
        assert all(q["clash_with"] == "" for q in quants)


# ---------------------------------------------------------------------------
# 9. _group_hf_quants is handed untrusted JSON straight off the HF tree API
#    -- a `size` that is not a number (seen briefly on a repo mid-upload)
#    must read as a zero-byte file, never a 500 for the whole Library page.
#    Directory rows carry a `size` too and must never let it leak into a
#    quant's byte total.
# ---------------------------------------------------------------------------

class TestGroupHfQuantsToleratesBadSizes:

    def test_non_numeric_sizes_become_zero_not_an_exception(self):
        files = [
            {"path": "Foo-Q4_K_M.gguf", "size": "abc"},
            {"path": "Bar-Q4_K_M.gguf", "size": None},
            {"path": "Baz-Q4_K_M.gguf", "size": [1]},
        ]
        quants = gw._group_hf_quants(files)["quants"]
        assert len(quants) == 3
        assert all(q["bytes"] == 0 for q in quants)
        assert all(fi["size"] == 0 for q in quants for fi in q["files"])

    def test_directory_entries_never_become_a_quant_or_count_toward_bytes(self):
        files = [
            {"type": "directory", "path": "Q4_K_M", "size": 999999},
            {"type": "file", "path": "Q4_K_M/model-Q4_K_M.gguf", "size": 500},
        ]
        g = gw._group_hf_quants(files)
        assert len(g["quants"]) == 1
        assert g["quants"][0]["bytes"] == 500
