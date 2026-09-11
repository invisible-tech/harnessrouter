"""_hermes_prepare_env (the hermes backend's own builder) threads a connection's extra_headers
through to the loopback relay for its openai-api provider path."""
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from server import Auth, _hermes_prepare_env, _HERMES_RELAY  # noqa: E402


def test_hermes_threads_extra_headers_to_relay():
    d = tempfile.mkdtemp()
    env = {"HOME": d}
    _hermes_prepare_env("openai-api", Auth(api_key="real", base_url="https://relay.example/v1",
                                           extra_headers={"X-Project": "foo"}), d, env)
    _, _, flags = _HERMES_RELAY["routes"][env["OPENAI_API_KEY"]]
    assert flags["extra_headers"] == {"X-Project": "foo"}


def test_hermes_with_no_extra_headers_calls_relay_unchanged():
    d = tempfile.mkdtemp()
    env = {"HOME": d}
    _hermes_prepare_env("openai-api", Auth(api_key="real", base_url="https://relay.example/v1"), d, env)
    _, _, flags = _HERMES_RELAY["routes"][env["OPENAI_API_KEY"]]
    assert flags["extra_headers"] == {}
