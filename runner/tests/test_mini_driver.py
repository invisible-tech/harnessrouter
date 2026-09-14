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
