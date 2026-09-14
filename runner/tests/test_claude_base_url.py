"""The claude CLI appends /v1/messages itself: any base handed to it loses a trailing /v1."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import server  # noqa: E402


def _base(tmp_path, url):
    env = {"HOME": str(tmp_path)}
    server._build_claude("anthropic", server.Auth(provider="anthropic", api_key="k", base_url=url), "claude-haiku-4.5", "hi", 1, str(tmp_path), env)
    return env["ANTHROPIC_BASE_URL"]


def test_a_direct_anthropic_base_with_v1_is_stripped(tmp_path):
    assert _base(tmp_path, "https://api.anthropic.com/v1") == "https://api.anthropic.com"


def test_a_bare_direct_anthropic_base_is_kept(tmp_path):
    assert _base(tmp_path, "https://api.anthropic.com") == "https://api.anthropic.com"


def test_a_router_base_is_stripped_as_before(tmp_path):
    env = {"HOME": str(tmp_path)}
    server._build_claude("tokenrouter", server.Auth(provider="tokenrouter", api_key="k", base_url="https://api.tokenrouter.com/v1"), "claude-haiku-4.5", "hi", 1, str(tmp_path), env)
    assert env["ANTHROPIC_BASE_URL"] == "https://api.tokenrouter.com"


def test_claude_env_sets_anthropic_custom_headers_when_present(tmp_path):
    env = {"HOME": str(tmp_path)}
    server._build_claude("anthropic", server.Auth(provider="anthropic", api_key="k",
                                                   extra_headers={"X-Project": "foo"}),
                         "claude-haiku-4.5", "hi", 1, str(tmp_path), env)
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "X-Project: foo"


def test_claude_env_omits_anthropic_custom_headers_when_absent(tmp_path):
    env = {"HOME": str(tmp_path)}
    server._build_claude("anthropic", server.Auth(provider="anthropic", api_key="k"),
                         "claude-haiku-4.5", "hi", 1, str(tmp_path), env)
    assert "ANTHROPIC_CUSTOM_HEADERS" not in env


def test_claude_tokenrouter_provider_also_gets_custom_headers(tmp_path):
    env = {"HOME": str(tmp_path)}
    server._build_claude("tokenrouter", server.Auth(provider="tokenrouter", api_key="k",
                                                     base_url="https://api.tokenrouter.com/v1",
                                                     extra_headers={"X-Project": "foo"}),
                         "claude-haiku-4.5", "hi", 1, str(tmp_path), env)
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "X-Project: foo"
