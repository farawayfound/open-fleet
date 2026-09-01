"""GET /admin/api/usage and GET /admin/api/keys, scoped to a caller-chosen
project.

the public site's admin dashboard shows only the hub keys it mints for itself
(its own feature keys, plus whatever Fleet Pass keys it issued through the
public flow) rather than every key on the box. `key_names`/`public` on the
usage endpoint, and `names` on the keys list, are how it asks for that slice
without a second, bespoke endpoint -- and with none of them present, both
endpoints must behave exactly as they did before this file existed.
"""
from __future__ import annotations

import app as gw  # conftest has set the env and sys.path by the time this loads

USAGE = "/admin/api/usage"
KEYS = "/admin/api/keys"


def _mint(name: str) -> dict:
    """A key dict shaped the way record_usage wants it, from the real
    mint_key() rather than a hand-rolled fake -- so the id it carries is one
    that actually exists in api_keys, the way a live key's would."""
    _raw, meta = gw.mint_key(name)
    return {"id": meta["id"], "name": meta["name"]}


def _usage(key: dict, model: str = "qwen3.8-9b-distill") -> None:
    gw.record_usage(
        key, model, "/v1/chat/completions", False, 200,
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        50, 100,
    )


def _link_public(key: dict, email: str = "jane@example.com") -> None:
    """A public_keys row standing in for a Fleet Pass key this project
    issued -- the join the `public=1` half of the scope predicate walks."""
    gw.db_exec(
        "INSERT INTO public_keys(created_at,email,domain,company,source,kind,"
        "models,ctx,key_id,status) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (gw.now(), email, email.split("@")[-1], "", "manual", "single",
         "{}", 8192, key["id"], "issued"),
    )


class TestUsageScope:
    def test_no_params_returns_every_row_and_null_scope(self, client, admin_headers):
        """Absent params -> byte-identical to today: nothing scoped out, and
        no `scope` block cluttering a response nobody asked to filter."""
        a = _mint("downstream-app:ama@cpu-box-1")
        b = _mint("fleet-pass:someone@example.com")
        _usage(a)
        _usage(b)
        body = client.get(USAGE, headers=admin_headers).json()
        assert body["totals"]["reqs"] == 2
        assert {r["key_name"] for r in body["by_key"]} == {a["name"], b["name"]}
        assert body["recent_total"] == 2
        assert {r["key_name"] for r in body["recent"]} == {a["name"], b["name"]}
        assert body["scope"] is None

    def test_key_names_narrows_every_aggregate(self, client, admin_headers):
        a = _mint("downstream-app:ama@cpu-box-1")
        b = _mint("someone-elses-key")
        _usage(a)
        _usage(b)
        body = client.get(
            USAGE, headers=admin_headers, params={"key_names": a["name"]}
        ).json()
        assert body["totals"]["reqs"] == 1
        assert [r["key_name"] for r in body["by_key"]] == [a["name"]]
        assert [r["reqs"] for r in body["by_model"]] == [1]
        assert len(body["series"]) == 1
        assert body["recent_total"] == 1
        assert body["recent"][0]["key_name"] == a["name"]
        assert body["scope"] == {"key_names": [a["name"]], "public": False}

    def test_key_names_trims_and_drops_empties(self, client, admin_headers):
        a = _mint("downstream-app:ama@cpu-box-1")
        b = _mint("downstream-app:workspace@cpu-box-1")
        _usage(a)
        _usage(b)
        body = client.get(
            USAGE, headers=admin_headers,
            params={"key_names": " " + a["name"] + " , ,, " + b["name"]},
        ).json()
        assert body["totals"]["reqs"] == 2
        assert body["scope"]["key_names"] == [a["name"], b["name"]]

    def test_public_flag_includes_only_the_linked_key(self, client, admin_headers):
        internal = _mint("downstream-app:ama@cpu-box-1")
        pass_key = _mint("fleet-pass:jane@example.com")
        _usage(internal)
        _usage(pass_key)
        _link_public(pass_key)
        body = client.get(USAGE, headers=admin_headers, params={"public": "1"}).json()
        assert body["totals"]["reqs"] == 1
        assert [r["key_name"] for r in body["by_key"]] == [pass_key["name"]]
        assert body["scope"] == {"key_names": [], "public": True}

    def test_public_flag_accepts_true_and_yes_but_not_garbage(self, client, admin_headers):
        pass_key = _mint("fleet-pass:jane@example.com")
        _usage(pass_key)
        _link_public(pass_key)
        for truthy in ("true", "YES", "1"):
            body = client.get(USAGE, headers=admin_headers, params={"public": truthy}).json()
            assert body["totals"]["reqs"] == 1, truthy
        body = client.get(USAGE, headers=admin_headers, params={"public": "nah"}).json()
        assert body["scope"] is None
        assert body["totals"]["reqs"] == 1  # unscoped: the only row anyway

    def test_key_names_and_public_union_without_double_counting(self, client, admin_headers):
        internal = _mint("downstream-app:ama@cpu-box-1")
        pass_key = _mint("fleet-pass:jane@example.com")
        unrelated = _mint("someone-elses-key")
        _usage(internal)
        _usage(pass_key)
        _usage(unrelated)
        _link_public(pass_key)
        body = client.get(
            USAGE, headers=admin_headers,
            params={"key_names": internal["name"], "public": "true"},
        ).json()
        assert body["totals"]["reqs"] == 2
        assert {r["key_name"] for r in body["by_key"]} == {internal["name"], pass_key["name"]}
        assert body["scope"] == {"key_names": [internal["name"]], "public": True}

    def test_public_scope_ignores_a_pending_row_with_no_key_id(self, client, admin_headers):
        """A public_keys row that never minted a key (still pending, or
        denied) has key_id NULL -- it must not make the `key_id IS NOT NULL`
        subquery match everything via SQL's NULL-in-IN semantics."""
        internal = _mint("downstream-app:ama@cpu-box-1")
        _usage(internal)
        gw.db_exec(
            "INSERT INTO public_keys(created_at,email,domain,company,source,kind,"
            "models,ctx,status) VALUES (?,?,?,?,?,?,?,?,?)",
            (gw.now(), "pending@example.com", "example.com", "", "manual",
             "single", "{}", 8192, "pending"),
        )
        body = client.get(USAGE, headers=admin_headers, params={"public": "1"}).json()
        assert body["totals"]["reqs"] == 0
        assert body["by_key"] == []


class TestKeysNamesFilter:
    def test_names_filters_the_list_and_total(self, client, admin_headers):
        a = _mint("downstream-app:ama@cpu-box-1")
        _mint("someone-elses-key")
        body = client.get(KEYS, headers=admin_headers, params={"names": a["name"]}).json()
        assert body["total"] == 1
        assert [k["name"] for k in body["items"]] == [a["name"]]

    def test_multiple_names_and_archived_still_applies(self, client, admin_headers):
        a = _mint("downstream-app:ama@cpu-box-1")
        b = _mint("downstream-app:workspace@cpu-box-1")
        _mint("someone-elses-key")
        body = client.get(
            KEYS, headers=admin_headers,
            params={"names": a["name"] + "," + b["name"]},
        ).json()
        assert body["total"] == 2
        assert {k["name"] for k in body["items"]} == {a["name"], b["name"]}
        # archived defaults to false -- a live key never shows up when the
        # caller is asking for the archived page, name filter or not.
        archived_body = client.get(
            KEYS, headers=admin_headers,
            params={"names": a["name"], "archived": "true"},
        ).json()
        assert archived_body["total"] == 0

    def test_absent_or_empty_names_is_unfiltered(self, client, admin_headers):
        _mint("a-key")
        _mint("b-key")
        assert client.get(KEYS, headers=admin_headers).json()["total"] == 2
        assert client.get(
            KEYS, headers=admin_headers, params={"names": ""}
        ).json()["total"] == 2
