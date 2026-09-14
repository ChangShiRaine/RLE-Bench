"""Regression tests for the Grok Build adapters."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from harbor.models.agent.context import AgentContext

from rlebench.agents.grok import ResumingGrokBuild, SubscriptionGrokBuild


class FakeEnvironment:
    default_user = "agent"

    def __init__(self) -> None:
        self.uploads: list[tuple[Path, str]] = []

    async def upload_file(self, source: Path, target: str) -> None:
        self.uploads.append((Path(source), target))


def _agent(tmp_path: Path, **extra_env: str) -> SubscriptionGrokBuild:
    return SubscriptionGrokBuild(logs_dir=tmp_path / "logs", model_name="xai/grok-4.6", extra_env=extra_env)


def _record(agent: SubscriptionGrokBuild) -> list[tuple[str, str, dict | None]]:
    calls: list[tuple[str, str, dict | None]] = []

    def recorder(user: str):
        async def exec_(environment, *, command: str, env: dict | None = None, **kwargs):
            calls.append((user, command, env))
        return exec_

    agent.exec_as_agent = recorder("agent")  # type: ignore[method-assign]
    agent.exec_as_root = recorder("root")  # type: ignore[method-assign]
    return calls


def test_ships_auth_json_and_skips_api_key(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    agent = _agent(tmp_path, GROK_AUTH_JSON_PATH=str(auth), XAI_API_KEY="leaked")
    calls = _record(agent)
    env = FakeEnvironment()
    asyncio.run(agent.run("finish the task", env, AgentContext()))

    assert env.uploads == [(auth, "/tmp/grok-build-secrets/auth.json")]
    assert any(u == "root" and c.startswith("chown agent ") for u, c, _ in calls)
    assert any('ln -sf /tmp/grok-build-secrets/auth.json "$HOME/.grok/auth.json"' in c for _, c, _ in calls)
    run_user, run_cmd, run_env = next(x for x in calls if x[2] is not None)
    assert run_user == "agent" and "--model grok-4.6" in run_cmd and "finish the task" in run_cmd
    assert "XAI_API_KEY" not in run_env and run_env["GROK_DISABLE_AUTOUPDATER"] == "1"


def test_force_auth_json_uses_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    agent = _agent(tmp_path, GROK_FORCE_AUTH_JSON="1")
    with pytest.raises(ValueError, match="not found"):
        agent._auth_json_path()
    (tmp_path / ".grok").mkdir()
    (tmp_path / ".grok" / "auth.json").write_text("{}")
    assert agent._auth_json_path() == tmp_path / ".grok" / "auth.json"


def test_requires_opt_in(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="grok login"):
        _agent(tmp_path)._auth_json_path()


def test_resume_reopens_pinned_session(tmp_path: Path) -> None:
    agent = ResumingGrokBuild(logs_dir=tmp_path / "logs", model_name="xai/grok-4.6", extra_env={"XAI_API_KEY": "k"})
    assert ResumingGrokBuild.SUPPORTS_RESUME and SubscriptionGrokBuild.SUPPORTS_RESUME
    calls = _record(agent)
    asyncio.run(agent.run("step one", FakeEnvironment(), AgentContext()))
    asyncio.run(agent.resume("step two", FakeEnvironment(), AgentContext()))
    first, second = [c for _, c, env in calls if env is not None]
    sid = agent._session_id
    assert f"--session-id {sid}" in first and "--resume" not in first
    assert f"--resume {sid}" in second and "--session-id" not in second
    assert "step two" in second and not agent._resume


def test_each_unresumed_step_opens_a_new_session(tmp_path: Path) -> None:
    agent = ResumingGrokBuild(logs_dir=tmp_path / "logs", model_name="xai/grok-4.6", extra_env={"XAI_API_KEY": "k"})
    calls = _record(agent)
    asyncio.run(agent.run("step one", FakeEnvironment(), AgentContext()))
    first_id = agent._session_id
    asyncio.run(agent.run("step two", FakeEnvironment(), AgentContext()))
    second_id = agent._session_id
    first, second = [c for _, c, env in calls if env is not None]
    assert first_id != second_id
    assert f"--session-id {first_id}" in first
    assert f"--session-id {second_id}" in second and "--resume" not in second
