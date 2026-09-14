"""Regression tests for the Antigravity agent adapter."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from harbor.models.agent.context import AgentContext

from rlebench.agents.antigravity import (
    GeminiApiAntigravityCli,
    ClosedBookAntigravityCli,
)


def test_gemini_api_provider_settings(tmp_path):
    agent = GeminiApiAntigravityCli(
        logs_dir=tmp_path, model_name='google/gemini-3.7-flash',
        reasoning_effort='high', disable_web_tools=True,
    )
    config, alias = agent._build_settings_config('gemini-3.7-flash')
    assert config['modelProvider'] == 'gemini'
    assert config['modelConfigs']['defaultModel'] == alias
    assert agent._disable_web_tools


def _agent(tmp_path: Path) -> GeminiApiAntigravityCli:
    return GeminiApiAntigravityCli(
        logs_dir=tmp_path,
        model_name="google/gemini-3.7-flash",
        reasoning_effort="high",
        extra_env={
            "GEMINI_API_KEY": "test-key",
            "GOOGLE_GEMINI_BASE_URL": "https://gemini.example.test",
            "GOOGLE_GENAI_API_VERSION": "v1beta",
        },
    )


def test_antigravity_uses_harbor_owned_phase_timeout(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    calls: list[tuple[str, dict[str, str] | None]] = []

    async def record_exec(
        environment: Any,
        *,
        command: str,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        calls.append((command, env))

    agent.exec_as_agent = record_exec  # type: ignore[method-assign]
    asyncio.run(agent.run("finish the task", object(), AgentContext()))

    command, env = next(
        (command, env)
        for command, env in calls
        if "/agy --dangerously-skip-permissions" in command
    )
    assert "--print-timeout 24h" in command
    assert "--model gemini-3.7-flash" in command
    assert "--effort high" in command
    assert "--prompt='finish the task'" in command
    assert env is not None
    assert env["GEMINI_API_KEY"] == "test-key"
    assert env == {"GEMINI_API_KEY": "test-key", "GEMINI_CLI_TRUST_WORKSPACE": "true"}


def test_antigravity_resume_keeps_long_print_timeout(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    commands: list[str] = []

    async def record_exec(
        environment: Any,
        *,
        command: str,
        **kwargs: Any,
    ) -> None:
        commands.append(command)

    agent.exec_as_agent = record_exec  # type: ignore[method-assign]
    asyncio.run(agent.resume("evaluate", object(), AgentContext()))

    command = next(
        command
        for command in commands
        if "/agy --dangerously-skip-permissions" in command
    )
    assert "--continue" in command
    assert "--print-timeout 24h" in command


def test_base_adapter_does_not_force_api_key_provider(tmp_path: Path) -> None:
    agent = ClosedBookAntigravityCli(
        logs_dir=tmp_path,
        model_name="google/gemini-3.7-flash",
        reasoning_effort="high",
    )

    config, _ = agent._build_settings_config("gemini-3.7-flash")

    assert config is not None
    assert "modelProvider" not in config


def test_api_install_run_and_resume_ignore_oauth(tmp_path):
    agent = GeminiApiAntigravityCli(
        logs_dir=tmp_path, model_name="google/gemini-3.7-flash",
        extra_env={"AGY_AUTH_JSON_PATH": str(tmp_path / "missing-token"),
                   "AGY_FORCE_AUTH_JSON": "1", "GEMINI_API_KEY": "test-key"},
    )
    commands = []

    async def record_exec(environment, *, command, **kwargs):
        commands.append(command)

    async def dependencies(*args, **kwargs):
        pass

    agent.exec_as_agent = record_exec
    agent.ensure_system_dependencies = dependencies
    environment = object()
    asyncio.run(agent.install(environment))
    asyncio.run(agent.run("develop", environment, AgentContext()))
    asyncio.run(agent.resume("evaluate", environment, AgentContext()))
    assert not agent._seeded_token
    assert not any("oauth" in c for c in commands)
    runs = [c for c in commands if "/agy --dangerously" in c]
    assert len(runs) == 2
    assert all("--effort" not in c for c in runs)


class LocalEnvironment:
    """Exercise Harbor's real execution/error wrapper in an isolated home."""

    def __init__(self, home: Path, fail: str | None = None):
        self.home = home
        self.fail = fail
        self.calls: list[str] = []

    async def exec(self, command, *, user=None, env=None, **kwargs):
        self.calls.append(command)
        assert user is None  # Harbor must retain the environment's agent user.
        if self.fail and self.fail in command:
            return SimpleNamespace(return_code=1, stdout="", stderr="injected failure")
        if '/agy --dangerously-skip-permissions' in command or command.startswith('set -o pipefail; src='):
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        result = subprocess.run(
            ['bash', '-c', command], capture_output=True, text=True,
            env={**os.environ, **(env or {}), 'HOME': str(self.home)},
        )
        return SimpleNamespace(return_code=result.returncode, stdout=result.stdout, stderr=result.stderr)


@pytest.mark.parametrize('original', [None, '{"existing":{"enabled":false}}\n', '{broken'])
def test_hook_lifecycle(tmp_path, original):
    agent = ClosedBookAntigravityCli(
        logs_dir=tmp_path / 'logs', model_name='google/gemini-3.7-flash', disable_web_tools=True,
    )
    path = tmp_path / '.gemini/config/hooks.json'
    path.parent.mkdir(parents=True)
    if original is not None:
        path.write_text(original)
    environment = LocalEnvironment(tmp_path)
    if original == '{broken':
        with pytest.raises(RuntimeError):
            asyncio.run(agent.run('test', environment, AgentContext()))
        assert not any('/agy --dangerously' in c for c in environment.calls)
    else:
        asyncio.run(agent.exec_as_agent(environment, command=agent._build_web_hook_command()))
        config = json.loads(path.read_text())
        hook = config[agent._WEB_HOOK_NAME]['PreToolUse'][0]
        for tool in ('search_web', 'read_url_content'):
            assert re.fullmatch(hook['matcher'], tool)
        for tool in ('run_command', 'view_file'):
            assert not re.fullmatch(hook['matcher'], tool)
        result = subprocess.run(['sh', '-c', hook['hooks'][0]['command']], capture_output=True, text=True, check=True)
        assert json.loads(result.stdout)['decision'] == 'deny'
        asyncio.run(agent.exec_as_agent(environment, command=agent._build_restore_web_hook_command()))
    assert path.read_text() == original if original is not None else not path.exists()
    assert not list(path.parent.glob('hooks.json.rlebench-*'))


@pytest.mark.parametrize('disable', [True, False])
def test_web_control_and_cleanup(tmp_path, disable):
    agent = ClosedBookAntigravityCli(
        logs_dir=tmp_path / 'logs', model_name='google/gemini-3.7-flash', disable_web_tools=disable,
    )
    environment = LocalEnvironment(tmp_path)
    asyncio.run(agent.run('test', environment, AgentContext()))
    assert any(agent._build_web_hook_command() in c for c in environment.calls) == disable
    assert any('/agy --dangerously' in c for c in environment.calls)
    assert not (tmp_path / '.gemini/config/hooks.json').exists()


@pytest.mark.parametrize('failure', ['d.update', 'settings.json', '/agy --dangerously'])
def test_failure_cleans_hooks(tmp_path, failure):
    agent = ClosedBookAntigravityCli(
        logs_dir=tmp_path / 'logs', model_name='google/gemini-3.7-flash', disable_web_tools=True,
    )
    path = tmp_path / '.gemini/config/hooks.json'
    path.parent.mkdir(parents=True)
    path.write_text('{"existing":{}}\n')
    environment = LocalEnvironment(tmp_path, fail=failure)
    with pytest.raises(RuntimeError):
        asyncio.run(agent.run('test', environment, AgentContext()))
    assert path.read_text() == '{"existing":{}}\n'
    if failure != '/agy --dangerously':
        assert not any('/agy --dangerously' in c for c in environment.calls)


def test_cancellation_restores_hook(tmp_path):
    agent = ClosedBookAntigravityCli(
        logs_dir=tmp_path / 'logs', model_name='google/gemini-3.7-flash', disable_web_tools=True,
    )
    environment = LocalEnvironment(tmp_path)
    execute = environment.exec

    async def cancel(command, **kwargs):
        if '/agy --dangerously' in command:
            raise asyncio.CancelledError
        return await execute(command, **kwargs)

    environment.exec = cancel
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(agent.run('test', environment, AgentContext()))
    assert not (tmp_path / '.gemini/config/hooks.json').exists()


def test_cleanup_failure_is_reported(tmp_path):
    agent = ClosedBookAntigravityCli(
        logs_dir=tmp_path / 'logs', model_name='google/gemini-3.7-flash', disable_web_tools=True,
    )
    environment = LocalEnvironment(tmp_path, fail='shutil.rmtree')
    with pytest.raises(RuntimeError):
        asyncio.run(agent.run('test', environment, AgentContext()))


def test_partial_hook_install_preserves_original(tmp_path):
    agent = ClosedBookAntigravityCli(
        logs_dir=tmp_path / 'logs', model_name='google/gemini-3.7-flash', disable_web_tools=True,
    )
    path = tmp_path / '.gemini/config/hooks.json'
    path.parent.mkdir(parents=True)
    path.write_text('{"existing":{}}\n')
    command = agent._build_web_hook_command().replace("b.mkdir();", "b.mkdir(); raise RuntimeError();")
    agent._build_web_hook_command = lambda: command
    with pytest.raises(RuntimeError):
        asyncio.run(agent.run('test', LocalEnvironment(tmp_path), AgentContext()))
    assert path.read_text() == '{"existing":{}}\n'
    assert not list(path.parent.glob('hooks.json.rlebench-*'))


def test_cleanup_error_preserves_original_failure(tmp_path):
    agent = ClosedBookAntigravityCli(
        logs_dir=tmp_path / 'logs', model_name='google/gemini-3.7-flash', disable_web_tools=True,
    )
    environment = LocalEnvironment(tmp_path, fail='shutil.rmtree')
    execute = environment.exec

    async def fail_run(command, **kwargs):
        if '/agy --dangerously' in command:
            raise ValueError('original failure')
        return await execute(command, **kwargs)

    environment.exec = fail_run
    with pytest.raises(ValueError, match='original failure'):
        asyncio.run(agent.run('test', environment, AgentContext()))
