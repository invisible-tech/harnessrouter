"""One mini-SWE-agent turn, as a runner subprocess.

Spawned per turn by server.py (the same one-process-per-turn contract as every other backend:
the runner reads NDJSON off stdout, cancel is a process-group kill). Unlike dsh, there is no
separate runtime binary to relay for — mini-SWE-agent is a pure-Python library (`minisweagent`,
pinned in runner/requirements.txt) and this driver calls it in-process via `litellm.completion`,
so the turn's credential is a kwarg on that call, never an env var or an argv the runtime could
log or checkpoint.

mini-SWE-agent's own CLI (`mini`) has no NDJSON/streaming output — `DefaultAgent.save()`
rewrites a whole trajectory JSON file after every step, meant for a human tailing a file, not a
process reading a pipe. `StreamingAgent` below subclasses `DefaultAgent` and overrides
`add_messages` to emit one JSON line per message the instant it's added, so server.py's
normalizer sees a turn's assistant text, bash actions, and observations as they happen — the
same shape dsh_driver.py's `session.event` re-emission gives it, from a driver with no relay to
write.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig
from minisweagent.models.litellm_model import LitellmModel


def _emit(m: str, p: dict) -> None:
    print(json.dumps({"m": m, "p": p}), flush=True)


class StreamingAgent(DefaultAgent):
    """DefaultAgent, but every message is re-emitted on stdout the moment it's added — the
    turn's only observable output, since nothing here writes a trajectory file."""

    def add_messages(self, *messages: dict) -> list[dict]:
        out = super().add_messages(*messages)
        for msg in out:
            _emit("message", msg)
        return out


def main() -> None:
    job = json.loads(sys.argv[1])
    prompt: str = job["prompt"]
    model_name: str = job["model"]
    cwd: str = job["cwd"]
    # `or 3.0`/`or 0` would silently turn an explicit 0 (mini's own "no limit" convention — see
    # AgentConfig.cost_limit/step_limit) into the 3.0 default; "cost_limit" absent from job is the
    # only case that should fall back to it.
    cost_limit = 3.0 if job.get("cost_limit") is None else float(job["cost_limit"])
    step_limit = int(job.get("step_limit") or 0)

    # HR_MINI_*: consumed here and never touched again — the API key lives only in this
    # process's call to litellm.completion, the same boundary dsh_driver.py's relay draws by a
    # different means (there, a separate runtime process; here, no separate process at all).
    api_key = os.environ.pop("HR_MINI_API_KEY", "") or None
    base_url = os.environ.pop("HR_MINI_BASE_URL", "") or None
    extra_headers = json.loads(os.environ.pop("HR_MINI_EXTRA_HEADERS", "") or "{}") or None
    model_kwargs: dict[str, object] = {"drop_params": True}
    if api_key:
        model_kwargs["api_key"] = api_key
    if base_url:
        model_kwargs["api_base"] = base_url
    if extra_headers:
        model_kwargs["extra_headers"] = extra_headers

    _emit("__hr_init", {})
    try:
        # An absolute path, not the bare name "mini.yaml": get_config_from_spec/get_config_path
        # checks a CWD-relative candidate FIRST, and cwd here is the session's own workspace —
        # the user's (or an untrusted branch's) checked-out repo content. A repo that happens to
        # contain its own "mini.yaml" would silently replace this turn's system/instance
        # templates (and, via cfg["environment"]["env"], the bash environment every action
        # runs in) with attacker-controlled content instead of the bundled default. Naming the
        # exact builtin file removes the ambiguity entirely.
        cfg = get_config_from_spec(builtin_config_dir / "mini.yaml")
        agent_cfg = cfg.get("agent", {})
        model = LitellmModel(model_name=model_name, model_kwargs=model_kwargs,
                             observation_template=cfg.get("model", {}).get("observation_template",
                                                                          "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"),
                             format_error_template=cfg.get("model", {}).get("format_error_template", "{{ error }}"))
        env = LocalEnvironment(config_class=LocalEnvironmentConfig, cwd=cwd,
                               env=cfg.get("environment", {}).get("env", {}))
        agent = StreamingAgent(model, env, config_class=AgentConfig,
                               system_template=agent_cfg["system_template"],
                               instance_template=agent_cfg["instance_template"],
                               cost_limit=cost_limit, step_limit=step_limit)
        result = agent.run(prompt)
        _emit("__hr_result", {"exit_status": result.get("exit_status", ""),
                              "submission": result.get("submission", ""),
                              "cost": agent.cost, "n_calls": agent.n_calls})
    except Exception as e:  # noqa: BLE001 — the turn's failure IS the result, not a crash to hide
        _emit("__hr_result", {"exit_status": type(e).__name__, "submission": "",
                              "error": str(e), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
