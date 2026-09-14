"""Grok Build adapters: multi-step sessions, and a subscription-login variant."""

from __future__ import annotations

import shlex
import uuid
from pathlib import Path, PurePosixPath

from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.grok_build import GrokBuild
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths
from harbor.utils.env import parse_bool_env_value


class ResumingGrokBuild(GrokBuild):
    """Upstream Grok Build with per-step grok sessions.

    Harbor 0.21's adapter pins one `--session-id` for the whole trial, but grok
    takes that flag only for a session that does not exist yet, so every step
    after the first fails with "Session ID ... is already in use". A later step
    instead reopens the pinned session with `--resume <id>` when the trial
    resumes its trajectory, and otherwise opens a fresh session of its own.
    """

    SUPPORTS_RESUME = True

    _session_opened = False

    def _build_run_script(self, escaped_instruction: str) -> str:
        if self._session_opened and not self._resume:
            self._session_id = str(uuid.uuid4())
        self._session_opened = True
        script = super()._build_run_script(escaped_instruction)
        if self._resume:
            script = script.replace(
                f"--session-id {self._session_id}", f"--resume {self._session_id}", 1
            )
        return script


class SubscriptionGrokBuild(ResumingGrokBuild):
    """Ship the host's `grok login` credentials into the container.

    Opt in with GROK_AUTH_JSON_PATH=<file> or GROK_FORCE_AUTH_JSON=1 (host
    ~/.grok/auth.json). XAI_API_KEY is neither required nor forwarded.
    """

    _REMOTE_SECRETS_DIR = PurePosixPath("/tmp/grok-build-secrets")

    def _auth_json_path(self) -> Path:
        explicit = self._get_env("GROK_AUTH_JSON_PATH")
        if explicit:
            path = Path(explicit)
        elif parse_bool_env_value(
            self._get_env("GROK_FORCE_AUTH_JSON"), name="GROK_FORCE_AUTH_JSON", default=False
        ):
            path = Path.home() / ".grok" / "auth.json"
        else:
            raise ValueError(
                "grok subscription auth: run `grok login` and set GROK_FORCE_AUTH_JSON=1, "
                "or point GROK_AUTH_JSON_PATH at an auth.json"
            )
        if not path.is_file():
            raise ValueError(f"grok auth.json not found: {path}")
        return path

    async def _install_auth(self, environment: BaseEnvironment) -> None:
        secrets_dir = self._REMOTE_SECRETS_DIR.as_posix()
        remote = (self._REMOTE_SECRETS_DIR / "auth.json").as_posix()
        await self.exec_as_agent(environment, command=f"mkdir -p {secrets_dir}")
        await environment.upload_file(self._auth_json_path(), remote)
        if environment.default_user is not None:
            owner = shlex.quote(str(environment.default_user))
            await self.exec_as_root(environment, command=f"chown {owner} {remote} && chmod 600 {remote}")
        await self.exec_as_agent(
            environment, command=f'mkdir -p "$HOME/.grok" && ln -sf {remote} "$HOME/.grok/auth.json"'
        )

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        await self._install_auth(environment)
        env = {
            "GROK_DISABLE_AUTOUPDATER": "1",
            "GROK_LOG_FILE": (EnvironmentPaths.agent_dir / self._CLI_LOG_FILENAME).as_posix(),
            "RUST_LOG": self._get_env("RUST_LOG") or "warn",
        }
        await self.exec_as_agent(environment, command=self._build_write_config_command())
        try:
            await self.exec_as_agent(
                environment, command=self._build_run_script(shlex.quote(instruction)), env=env
            )
        finally:
            sessions_target = (EnvironmentPaths.agent_dir / "sessions").as_posix()
            try:
                await self.exec_as_agent(
                    environment,
                    command=(
                        f"mkdir -p {EnvironmentPaths.agent_dir.as_posix()}\n"
                        'if [ -d "$HOME/.grok/sessions" ]; then\n'
                        f"  rm -rf {sessions_target}\n"
                        f'  cp -R "$HOME/.grok/sessions" {sessions_target}\n'
                        "fi"
                    ),
                )
            except Exception:
                self.logger.debug("Failed to copy grok session files", exc_info=True)
