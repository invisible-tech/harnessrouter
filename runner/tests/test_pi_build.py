"""_build_pi threads a connection's extra_headers through to the loopback relay, same as every
other relay-based backend."""
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from server import Auth, _build_pi, _HERMES_RELAY  # noqa: E402


def _relay_token_from_models_json(home_dir):
    cfg = json.loads((pathlib.Path(home_dir) / ".pi" / "agent" / "models.json").read_text())
    return cfg["providers"]["hr"]["apiKey"]


def test_pi_threads_extra_headers_to_relay():
    d = tempfile.mkdtemp()
    env = {"HOME": d}
    _build_pi("tokenrouter", Auth(api_key="real", base_url="https://api.tokenrouter.com/v1",
                                  extra_headers={"X-Project": "foo"}), "gpt-5.4", "do work", d, env)
    _, _, flags = _HERMES_RELAY["routes"][_relay_token_from_models_json(d)]
    assert flags["extra_headers"] == {"X-Project": "foo"}


def test_pi_with_no_extra_headers_calls_relay_unchanged():
    d = tempfile.mkdtemp()
    env = {"HOME": d}
    _build_pi("tokenrouter", Auth(api_key="real", base_url="https://api.tokenrouter.com/v1"),
             "gpt-5.4", "do work", d, env)
    _, _, flags = _HERMES_RELAY["routes"][_relay_token_from_models_json(d)]
    assert flags["extra_headers"] == {}
