"""StreamingAgent actually streams: every message DefaultAgent adds must reach stdout the
instant it's added, not batched at the end (the whole reason mini_driver.py exists — mini's own
CLI only writes a trajectory file, see the module docstring)."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import mini_driver  # noqa: E402
from minisweagent.agents.default import AgentConfig  # noqa: E402
from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig  # noqa: E402


class _StubModel:
    """Two turns: one bash call, then a submission — no network, no litellm."""

    def __init__(self):
        self.calls = 0

    def format_message(self, **kwargs):
        return kwargs

    def format_observation_messages(self, message, outputs, template_vars=None):
        if not outputs:   # the exit message: no actions, nothing to observe
            return []
        out = outputs[0]
        return [{"role": "tool", "content": out["output"], "tool_call_id": "c1",
                "extra": {"returncode": out["returncode"], "exception_info": out["exception_info"]}}]

    def get_template_vars(self):
        return {}

    def serialize(self):
        return {}

    def query(self, messages):
        self.calls += 1
        if self.calls == 1:
            return {"role": "assistant", "content": "checking",
                    "extra": {"actions": [{"command": "echo hi", "tool_call_id": "c1"}], "cost": 0.0}}
        return {"role": "exit", "content": "Submitted",
                "extra": {"exit_status": "Submitted", "submission": "done", "cost": 0.0}}


def test_every_message_is_emitted_as_it_is_added(tmp_path, capsys):
    env = LocalEnvironment(config_class=LocalEnvironmentConfig, cwd=str(tmp_path))
    agent = mini_driver.StreamingAgent(_StubModel(), env, config_class=AgentConfig,
                                       system_template="sys", instance_template="do: {{task}}")
    result = agent.run("say hi")
    lines = [json.loads(ln) for ln in capsys.readouterr().out.splitlines() if ln]
    roles = [ln["p"].get("role") for ln in lines]
    assert roles == ["system", "user", "assistant", "tool", "exit"]
    assert lines[2]["p"]["extra"]["actions"] == [{"command": "echo hi", "tool_call_id": "c1"}]
    assert lines[3]["p"]["content"] == "hi\n"
    assert result == {"exit_status": "Submitted", "submission": "done", "cost": 0.0}


def test_no_trajectory_file_is_written(tmp_path):
    env = LocalEnvironment(config_class=LocalEnvironmentConfig, cwd=str(tmp_path))
    agent = mini_driver.StreamingAgent(_StubModel(), env, config_class=AgentConfig,
                                       system_template="sys", instance_template="do: {{task}}")
    agent.run("say hi")
    assert list(tmp_path.iterdir()) == []


# ── main(): HR_MINI_EXTRA_HEADERS -> model_kwargs["extra_headers"] ─────────────────────
def _run_main(monkeypatch, tmp_path, env_extra_headers=None):
    captured = {}

    class _Model:
        def __init__(self, model_name, model_kwargs, **kw):
            captured["model_kwargs"] = model_kwargs

        cost = 0.0
        n_calls = 0

        def query(self, messages):
            return {"role": "exit", "content": "done",
                    "extra": {"exit_status": "Submitted", "submission": "done", "cost": 0.0}}

        def format_message(self, **kwargs):
            return kwargs

        def format_observation_messages(self, message, outputs, template_vars=None):
            return []

        def get_template_vars(self):
            return {}

        def serialize(self):
            return {}

    monkeypatch.setattr(mini_driver, "LitellmModel", _Model)
    monkeypatch.setenv("HR_MINI_API_KEY", "")
    monkeypatch.setenv("HR_MINI_BASE_URL", "")
    if env_extra_headers is not None:
        monkeypatch.setenv("HR_MINI_EXTRA_HEADERS", env_extra_headers)
    else:
        monkeypatch.delenv("HR_MINI_EXTRA_HEADERS", raising=False)
    job = json.dumps({"prompt": "say hi", "model": "anthropic/claude-sonnet-4-6", "cwd": str(tmp_path)})
    monkeypatch.setattr(sys, "argv", ["mini_driver.py", job])
    mini_driver.main()
    return captured["model_kwargs"]


def test_main_forwards_extra_headers_into_model_kwargs(monkeypatch, tmp_path):
    kwargs = _run_main(monkeypatch, tmp_path, env_extra_headers=json.dumps({"X-Project": "foo"}))
    assert kwargs["extra_headers"] == {"X-Project": "foo"}


def test_main_with_no_extra_headers_env_omits_the_kwarg(monkeypatch, tmp_path):
    kwargs = _run_main(monkeypatch, tmp_path, env_extra_headers="{}")
    assert "extra_headers" not in kwargs


# ── main(): an explicit cost_limit=0 in the job means "no limit", not "use the 3.0 default" ──
def test_explicit_cost_limit_zero_is_not_overridden_by_the_default(monkeypatch, tmp_path):
    captured = {}

    class _StubAgent:
        def __init__(self, model, env, *, cost_limit, step_limit, **kw):
            captured["cost_limit"] = cost_limit

        def run(self, prompt):
            return {"exit_status": "Submitted", "submission": "done"}

        cost = 0.0
        n_calls = 0

    monkeypatch.setattr(mini_driver, "StreamingAgent", _StubAgent)
    monkeypatch.setattr(mini_driver, "LitellmModel", lambda **kw: None)
    monkeypatch.setattr(mini_driver, "LocalEnvironment", lambda **kw: None)
    monkeypatch.setenv("HR_MINI_API_KEY", "")
    monkeypatch.setenv("HR_MINI_BASE_URL", "")
    monkeypatch.delenv("HR_MINI_EXTRA_HEADERS", raising=False)
    job = json.dumps({"prompt": "say hi", "model": "anthropic/claude-sonnet-4-6",
                      "cwd": str(tmp_path), "cost_limit": 0})
    monkeypatch.setattr(sys, "argv", ["mini_driver.py", job])
    mini_driver.main()
    assert captured["cost_limit"] == 0.0


def test_cost_limit_absent_from_job_still_defaults_to_3(monkeypatch, tmp_path):
    captured = {}

    class _StubAgent:
        def __init__(self, model, env, *, cost_limit, step_limit, **kw):
            captured["cost_limit"] = cost_limit

        def run(self, prompt):
            return {"exit_status": "Submitted", "submission": "done"}

        cost = 0.0
        n_calls = 0

    monkeypatch.setattr(mini_driver, "StreamingAgent", _StubAgent)
    monkeypatch.setattr(mini_driver, "LitellmModel", lambda **kw: None)
    monkeypatch.setattr(mini_driver, "LocalEnvironment", lambda **kw: None)
    monkeypatch.setenv("HR_MINI_API_KEY", "")
    monkeypatch.setenv("HR_MINI_BASE_URL", "")
    monkeypatch.delenv("HR_MINI_EXTRA_HEADERS", raising=False)
    job = json.dumps({"prompt": "say hi", "model": "anthropic/claude-sonnet-4-6", "cwd": str(tmp_path)})
    monkeypatch.setattr(sys, "argv", ["mini_driver.py", job])
    mini_driver.main()
    assert captured["cost_limit"] == 3.0
