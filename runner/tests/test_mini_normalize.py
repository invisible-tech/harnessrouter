"""_build_mini and _mini_to_claude, against mini_driver.py's own NDJSON shapes.

The message fixtures below are the exact dicts a real DefaultAgent run produces (captured by
running mini_driver.py end to end against a stubbed LitellmModel.query — see mini_driver.py's
docstring for why they are DefaultAgent's own message dicts, not a wire protocol of their own).
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from server import Auth, _build_mini, _mini_to_claude  # noqa: E402


def _run(lines, model="anthropic/claude-sonnet-4-6"):
    state = {"model": model, "final": ""}
    out = [_mini_to_claude(ln, state) for ln in lines]
    return [e for evs in out for e in evs], state


def _msg(**p):
    return {"m": "message", "p": p}


# ── a full turn: init, framing (dropped), one tool call, its result, submit ────────────
TOOL_RUN = [
    {"m": "__hr_init", "p": {}},
    _msg(role="system", content="You are a helpful assistant that can interact with a computer."),
    _msg(role="user", content="Please solve this issue: say hi"),
    _msg(role="assistant", content="Let me check the directory.",
        extra={"actions": [{"command": "ls", "tool_call_id": "call_1"}],
              "response": {"usage": {"prompt_tokens": 120, "completion_tokens": 30}}}),
    _msg(content='{\n  "returncode": 0,\n  "output": "file.py\\n"\n}',
        extra={"raw_output": "file.py\n", "returncode": 0, "exception_info": ""},
        tool_call_id="call_1", role="tool"),
    _msg(role="exit", content="Submitted", extra={"exit_status": "Submitted", "submission": "done"}),
    {"m": "__hr_result", "p": {"exit_status": "Submitted", "submission": "done", "cost": 0.01, "n_calls": 1}},
]


def test_init_carries_a_session_id_and_the_model():
    events, _ = _run(TOOL_RUN[:1])
    assert events == [{"type": "system", "subtype": "init",
                        "session_id": events[0]["session_id"], "model": "anthropic/claude-sonnet-4-6"}]
    assert events[0]["session_id"].startswith("mini")


def test_framing_messages_are_dropped_not_shown_as_chat():
    events, _ = _run(TOOL_RUN[1:3])
    assert events == []


def test_assistant_text_and_tool_call_become_one_message_two_blocks():
    events, state = _run(TOOL_RUN[3:4])
    assert events == [{"type": "assistant", "message": {"content": [
        {"type": "text", "text": "Let me check the directory."},
        {"type": "tool_use", "id": "call_1", "name": "bash", "input": {"command": "ls"}}]}}]
    assert state["_mini_usage"] == {"input_tokens": 120, "output_tokens": 30}


def test_tool_observation_becomes_a_tool_result_keyed_by_call_id():
    events, _ = _run(TOOL_RUN[4:5])
    assert events == [{"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "call_1", "is_error": False,
         "content": '{\n  "returncode": 0,\n  "output": "file.py\\n"\n}'}]}}]


def test_exit_role_is_folded_into_the_result_not_re_emitted():
    events, _ = _run(TOOL_RUN[5:6])
    assert events == []


def test_result_reports_submission_and_summed_usage():
    events, _ = _run(TOOL_RUN)
    assert events[-1] == {"type": "result", "subtype": "success", "is_error": False,
                          "result": "done", "usage": {"input_tokens": 120, "output_tokens": 30}}


# ── an execution failure (not a model refusal) surfaces as a tool error, not a turn error ──
def test_an_execution_exception_marks_the_tool_result_an_error():
    events, _ = _run([_msg(content="<exception>boom</exception>...", extra={"exception_info": "boom",
                                                                          "returncode": -1}, tool_call_id="c", role="tool")])
    assert events[0]["message"]["content"][0]["is_error"] is True


# ── limits and hard failures map to distinct result subtypes ───────────────────────────
def test_limits_exceeded_is_a_soft_stop_not_an_error():
    events, _ = _run([{"m": "__hr_result", "p": {"exit_status": "LimitsExceeded", "submission": ""}}])
    assert events == [{"type": "result", "subtype": "error_max_turns", "is_error": False,
                        "result": "", "usage": {}}]


def test_a_driver_exception_is_a_hard_error():
    events, _ = _run([{"m": "__hr_result", "p": {"exit_status": "RuntimeError", "submission": "",
                                                "error": "AuthenticationError: bad key"}}])
    assert events == [{"type": "result", "subtype": "error", "is_error": True,
                        "result": "AuthenticationError: bad key", "usage": {}}]


# ── _build_mini: provider validation, model prefixing, agent_doc, the one-tool guard ───
def _auth(**kw):
    return Auth(**{"api_key": "k", "base_url": "", "api_format": "", **kw})


def test_unknown_provider_is_rejected():
    import pytest
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        _build_mini("not-a-provider", _auth(), "claude-sonnet-4.6", "task", "/tmp", {})


def test_disabling_the_only_tool_is_rejected():
    import pytest
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        _build_mini("anthropic", _auth(), "claude-sonnet-4.6", "task", "/tmp", {},
                   tools_disabled=["bash"])


def test_claude_models_get_the_anthropic_litellm_prefix_openai_models_dont():
    env = {}
    cmd = _build_mini("anthropic", _auth(), "claude-sonnet-4.6", "task", "/tmp", env)
    job = json.loads(cmd[-1])
    assert job["model"] == "anthropic/claude-sonnet-4.6"
    env2 = {}
    cmd2 = _build_mini("openai", _auth(), "gpt-5.4", "task", "/tmp", env2)
    assert json.loads(cmd2[-1])["model"] == "openai/gpt-5.4"


def test_the_credential_rides_env_not_the_job_argv():
    env = {}
    cmd = _build_mini("anthropic", _auth(api_key="sk-secret"), "claude-sonnet-4.6", "task", "/tmp", env)
    assert env["HR_MINI_API_KEY"] == "sk-secret"
    assert "sk-secret" not in cmd[-1]


def test_build_mini_sets_hr_mini_extra_headers_env():
    env = {}
    _build_mini("anthropic", _auth(extra_headers={"X-Project": "foo"}), "claude-sonnet-4.6", "task", "/tmp", env)
    assert env["HR_MINI_EXTRA_HEADERS"] == json.dumps({"X-Project": "foo"})


def test_build_mini_strips_reserved_header_names_before_env():
    env = {}
    _build_mini("anthropic", _auth(extra_headers={"Authorization": "evil"}), "claude-sonnet-4.6", "task", "/tmp", env)
    assert env["HR_MINI_EXTRA_HEADERS"] == "{}"


def test_build_mini_with_no_extra_headers_sets_empty_json():
    env = {}
    _build_mini("anthropic", _auth(), "claude-sonnet-4.6", "task", "/tmp", env)
    assert env["HR_MINI_EXTRA_HEADERS"] == "{}"


def test_agent_doc_is_prepended_to_the_task_prompt():
    env = {}
    cmd = _build_mini("anthropic", _auth(), "claude-sonnet-4.6", "the task", "/tmp", env,
                      agent_doc="## Workspace contract\n\nDo not touch .git")
    job = json.loads(cmd[-1])
    assert job["prompt"].startswith("## Workspace contract")
    assert job["prompt"].endswith("the task")
