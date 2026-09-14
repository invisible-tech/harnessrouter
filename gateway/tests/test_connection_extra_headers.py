"""extra_headers: a connection's static headers, product-level (not the litellm.headers global
that fixed mini-swe-agent alone). Not a secret — routing/billing metadata like base_url — so it
is validated at write time (reserved names 400), stored plain, and returned by _conn_public like
any other non-secret field."""
import os
import pathlib
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("HR_BACKING", "local")
import app as gw  # noqa: E402

HEADERS = {"x-harness-internal": "test-internal-key", "x-harness-org": "local", "x-harness-member": "m"}


def test_conn_body_accepts_extra_headers():
    body = gw.ConnBody(backend="hermes", provider="openai-api", extra_headers={"X-Project": "foo"})
    assert body.extra_headers == {"X-Project": "foo"}


def test_conn_body_extra_headers_defaults_to_none():
    assert gw.ConnBody(backend="hermes", provider="openai-api").extra_headers is None


@pytest.mark.parametrize("name", sorted(gw._RESERVED_HEADER_NAMES) +
                          ["Authorization", "X-API-KEY"])
def test_put_connection_rejects_reserved_header_name(name):
    c = TestClient(gw.app)
    resp = c.put("/v1/orgs/local/connections/conn-reserved", headers=HEADERS,
                 json={"backend": "hermes", "provider": "openai-api", "extra_headers": {name: "x"}})
    assert resp.status_code == 400


def test_put_connection_accepts_non_reserved_header():
    c = TestClient(gw.app)
    put = c.put("/v1/orgs/local/connections/conn-ok", headers=HEADERS,
               json={"backend": "hermes", "provider": "openai-api", "extra_headers": {"X-Project": "foo"}})
    assert put.status_code == 200
    assert put.json()["connection"]["extra_headers"] == {"X-Project": "foo"}
    got = c.get("/v1/orgs/local/connections/conn-ok", headers=HEADERS)
    assert got.json()["connection"]["extra_headers"] == {"X-Project": "foo"}


def test_conn_public_includes_extra_headers():
    conn = {"name": "c", "provider": "anthropic", "api_key": "sk-secret", "extra_headers": {"X-Project": "foo"}}
    pub = gw._conn_public(conn)
    assert pub["extra_headers"] == {"X-Project": "foo"}
    assert "api_key" not in pub


def test_auth_from_conn_owner_mode_passes_extra_headers_through(monkeypatch):
    monkeypatch.setattr(gw, "SANDBOX_TRUST", "owner")
    out = gw._auth_from_conn({"provider": "anthropic", "api_key": "k",
                              "extra_headers": {"X-Project": "foo"}}, "sid1")
    assert out["extra_headers"] == {"X-Project": "foo"}


def test_auth_from_conn_brokered_mode_passes_extra_headers_through(monkeypatch):
    monkeypatch.setattr(gw, "SANDBOX_TRUST", "")
    monkeypatch.setattr(gw, "_mint_turn_cred", lambda sid, name: "hrt_x")
    out = gw._auth_from_conn({"provider": "anthropic", "api_key": "sk-ant-real", "name": "conn",
                              "extra_headers": {"X-Project": "foo"}}, "sid1")
    assert out["api_key"] == "hrt_x" and out["extra_headers"] == {"X-Project": "foo"}
