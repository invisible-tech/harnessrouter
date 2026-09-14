"""Auth.extra_headers and _apply_extra_headers — the one chokepoint every backend builder and
relay route through to merge a connection's static headers in, dropping any that would clobber
the credential the broker/relay injects itself."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import server as rn  # noqa: E402


def test_auth_model_accepts_extra_headers_field():
    assert rn.Auth(api_key="k", extra_headers={"X-Project": "foo"}).extra_headers == {"X-Project": "foo"}


def test_auth_model_extra_headers_defaults_to_none():
    assert rn.Auth(api_key="k").extra_headers is None


def test_apply_extra_headers_merges_without_mutating_base():
    base = {"accept": "*/*"}
    out = rn._apply_extra_headers(base, {"X-Project": "foo"})
    assert out == {"accept": "*/*", "X-Project": "foo"}
    assert base == {"accept": "*/*"}


def test_apply_extra_headers_drops_reserved_names_case_insensitively():
    out = rn._apply_extra_headers({}, {"Authorization": "evil", "X-Project": "keep"})
    assert out == {"X-Project": "keep"}


def test_apply_extra_headers_with_none_extra_returns_base_copy():
    base = {"a": "1"}
    out = rn._apply_extra_headers(base, None)
    assert out == base and out is not base


def test_apply_extra_headers_drops_a_value_with_an_embedded_newline():
    """An embedded newline in an otherwise-permitted header's VALUE would forge a second header
    line once _format_anthropic_custom_headers joins it — the same reserved-name bypass the key
    check alone can't catch."""
    out = rn._apply_extra_headers({}, {"X-Project": "foo\nauthorization: Bearer evil"})
    assert out == {}


def test_apply_extra_headers_drops_a_key_with_an_embedded_newline():
    out = rn._apply_extra_headers({}, {"X-Project\nauthorization": "evil"})
    assert out == {}


def test_apply_extra_headers_drops_reserved_names_with_surrounding_whitespace():
    out = rn._apply_extra_headers({}, {" Authorization ": "evil", "X-Project": "keep"})
    assert out == {"X-Project": "keep"}


def test_reserved_header_names_match_gateway_app():
    import os
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "gateway"))
    os.environ.setdefault("HR_BACKING", "local")
    import app as gw  # noqa: E402
    assert rn._RESERVED_HEADER_NAMES == gw._RESERVED_HEADER_NAMES
