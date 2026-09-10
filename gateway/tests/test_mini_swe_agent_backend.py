"""mini-swe-agent wiring: catalog entry, provider routing, and the chat/completions-only model
set (litellm's plain client, no Responses-API route — see test_catalog_chat_only_backends.py)."""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("HR_BACKING", "local")
import app as A  # noqa: E402


def test_catalog_entry_has_one_tool_and_hard_enforcement():
    entry = A._BASE_CATALOG["mini-swe-agent"]
    assert entry["backend"] == "mini-swe-agent"
    assert {n for n, _ in entry["tools"]} == {"bash"}
    assert entry["tool_enforcement"] == "hard"


def test_a_user_harness_on_this_base_routes_to_its_backend():
    assert A._backend_of_harness({"base": "mini-swe-agent"}) == "mini-swe-agent"


def test_reaches_every_backing_provider():
    for provider in ("anthropic", "openai", "azure-foundry", "openrouter", "tokenrouter", "vercel", "llmtr"):
        assert A._INTEGRATION_WIRING.get((provider, "mini-swe-agent")), provider
    assert A._INTEGRATION_WIRING[("google", "mini-swe-agent")] == "openai-api"


def test_custom_integration_drives_it_at_both_wire_formats():
    assert "mini-swe-agent" in A._CUSTOM_FORMAT_BACKENDS["openai"]
    assert "mini-swe-agent" in A._CUSTOM_FORMAT_BACKENDS["anthropic"]
    assert A._integration_serves_backend(
        {"provider": "custom", "config": {"api_format": "anthropic"}}, "mini-swe-agent")
    assert A._integration_serves_backend(
        {"provider": "custom", "config": {"api_format": "openai"}}, "mini-swe-agent")


def test_model_catalog_excludes_responses_only_ids():
    models = set(A._MODEL_CATALOG["mini-swe-agent"]["models"])
    assert not ({"gpt-5.3-codex", "gpt-6-astra"} & models)
    assert A._MODEL_CATALOG["mini-swe-agent"]["default"] in models
    assert "claude-fable-5-1" in models
