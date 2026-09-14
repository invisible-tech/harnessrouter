"""A connection's extra_headers reach Codex's config.toml as `http_headers` on its
[model_providers.X] table (confirmed against Codex's own config docs) — parsed back with
tomllib, not string-matched, so the test survives reformatting."""
import pathlib
import sys
import tomllib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from server import Auth, _codex_prepare_env  # noqa: E402


def test_codex_config_toml_includes_http_headers_when_present(tmp_path):
    home = tmp_path / "home"
    env = {"HOME": str(home)}
    auth = Auth(api_key="sk-test", base_url="https://example.invalid/v1", extra_headers={"X-Project": "foo"})
    cfg_dir = _codex_prepare_env("azure", auth, "gpt-5.5", str(tmp_path), env)
    doc = tomllib.loads((cfg_dir / "config.toml").read_text())
    assert doc["model_providers"]["hr-azure"]["http_headers"] == {"X-Project": "foo"}


def test_codex_config_toml_omits_http_headers_when_absent(tmp_path):
    home = tmp_path / "home"
    env = {"HOME": str(home)}
    auth = Auth(api_key="sk-test", base_url="https://example.invalid/v1")
    cfg_dir = _codex_prepare_env("azure", auth, "gpt-5.5", str(tmp_path), env)
    doc = tomllib.loads((cfg_dir / "config.toml").read_text())
    assert "http_headers" not in doc["model_providers"]["hr-azure"]


def test_codex_resume_alias_block_also_gets_http_headers(tmp_path):
    home = tmp_path / "home"
    sess = home / ".codex" / "sessions" / "2026" / "07"
    sess.mkdir(parents=True)
    (sess / "rollout-1.jsonl").write_text(
        '{"type": "session_meta", "payload": {"id": "s1", "model_provider": "hr-tokenrouter"}}\n')
    env = {"HOME": str(home)}
    auth = Auth(api_key="sk-x", base_url="https://broker.example/v1/llm", extra_headers={"X-Project": "foo"})
    cfg_dir = _codex_prepare_env("openai", auth, "gpt-5.4-mini", str(tmp_path), env, resume=True)
    doc = tomllib.loads((cfg_dir / "config.toml").read_text())
    assert doc["model_providers"]["hr-openai"]["http_headers"] == {"X-Project": "foo"}
    assert doc["model_providers"]["hr-tokenrouter"]["http_headers"] == {"X-Project": "foo"}
