"""Antigravity web-tool controls and resumable benchmark phases."""

from __future__ import annotations

import json
import shlex
import sys
from uuid import uuid4

from harbor.agents.installed.antigravity_cli import AntigravityCli
from harbor.agents.installed.base import with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class ClosedBookAntigravityCli(AntigravityCli):
    """Run Antigravity with web tools denied and long, resumable phases."""

    _WEB_HOOK_NAME = "harbor-disable-web"
    # `--prompt` selects agy's non-interactive print mode, whose default timeout
    # is only five minutes. RLE-Bench phases intentionally run for up to eight
    # hours, so leave phase timing to Harbor's outer per-step timeout.
    _PRINT_TIMEOUT = "24h"

    # agy 1.1.13+ supports continuing the latest workspace conversation with
    # --continue. Harbor 0.21's adapter predates that support.
    SUPPORTS_RESUME = True

    def __init__(self, *args, disable_web_tools: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._disable_web_tools = disable_web_tools
        self._hook_backup = f"hooks.json.rlebench-{uuid4().hex}"

    def _resolve_auth_token_path(self) -> None:
        """Disable the OAuth injection performed by Harbor's installer."""
        return None

    def _build_web_hook_command(self) -> str:
        """Install a temporary hook that blocks only Antigravity's web tools."""
        denial = shlex.quote(
            json.dumps(
                {
                    "decision": "deny",
                    "reason": "Web tools are disabled for this Harbor run.",
                }
            )
        )
        hook = {
            self._WEB_HOOK_NAME: {
                "PreToolUse": [
                    {
                        "matcher": "^(search_web|read_url_content)$",
                        "hooks": [
                            {
                                "type": "command",
                                "command": f"printf %s {denial}",
                            }
                        ],
                    }
                ]
            }
        }
        merge_script = (
            "import json, pathlib, shutil, sys; "
            "p = pathlib.Path.home() / '.gemini/config/hooks.json'; "
            "p.parent.mkdir(parents=True, exist_ok=True); "
            "d = json.loads(p.read_text()) if p.exists() else {}; "
            "d.update(json.loads(sys.argv[1])); "
            f"b = p.with_name({self._hook_backup!r}); b.mkdir(); "
            "shutil.copy2(p, b / 'copy') if p.exists() else (b / 'absent').touch(); "
            "(b / 'copy').replace(b / 'original') if p.exists() else None; "
            "(b / 'new').write_text(json.dumps(d, indent=2) + '\\n'); "
            "(b / 'new').replace(p)"
        )
        return f"python3 -c {shlex.quote(merge_script)} {shlex.quote(json.dumps(hook))}"

    def _build_restore_web_hook_command(self) -> str:
        script = (
            "import pathlib, shutil; "
            "p = pathlib.Path.home() / '.gemini/config/hooks.json'; "
            f"b = p.with_name({self._hook_backup!r}); "
            "(b / 'original').replace(p) if (b / 'original').exists() else "
            "p.unlink(missing_ok=True) if (b / 'absent').exists() else None; "
            "shutil.rmtree(b) if b.exists() else None"
        )
        return f"python3 -c {shlex.quote(script)}"

    def _build_settings_command(
        self, model: str | None = None
    ) -> tuple[str | None, str | None]:
        """Write settings where current Antigravity CLI releases read them."""
        config, model_alias = self._build_settings_config(model)
        if config is None:
            return None, model_alias

        escaped = shlex.quote(json.dumps(config, indent=2))
        command = (
            'mkdir -p "$HOME/.gemini/antigravity-cli" && '
            f"printf %s {escaped} > "
            '"$HOME/.gemini/antigravity-cli/settings.json"'
        )
        return command, model_alias

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        escaped_instruction = shlex.quote(instruction)

        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")

        model = self.model_name.split("/")[-1]

        env = {"GEMINI_CLI_TRUST_WORKSPACE": "true"}
        key = self._get_env("GEMINI_API_KEY")
        if key is not None:
            env["GEMINI_API_KEY"] = key

        hook_attempted = False
        try:
            skills_command = self._build_register_skills_command()
            if skills_command:
                await self.exec_as_agent(environment, command=skills_command, env=env)

            if self._disable_web_tools:
                hook_attempted = True
                await self.exec_as_agent(
                    environment, command=self._build_web_hook_command(), env=env
                )

            settings_command, _ = self._build_settings_command(model)
            if settings_command:
                await self.exec_as_agent(environment, command=settings_command, env=env)

            cli_flags = self.build_cli_flags()
            extra_flags = (cli_flags + " ") if cli_flags else ""
            model_flag = f"--model {shlex.quote(model)} " if model else ""
            effort = self._reasoning_effort
            effort_flag = f"--effort {shlex.quote(effort)} " if effort else ""
            resume_flag = "--continue " if self._resume else ""
            print_timeout_flag = f"--print-timeout {shlex.quote(self._PRINT_TIMEOUT)} "

            try:
                await self.exec_as_agent(
                    environment,
                    command=(
                        "$HOME/.local/bin/agy --dangerously-skip-permissions "
                        f"{resume_flag}{model_flag}{effort_flag}"
                        f"{print_timeout_flag}{extra_flags}"
                        f"--prompt={escaped_instruction} "
                        "2>&1 </dev/null | stdbuf -oL tee "
                        "/logs/agent/antigravity-cli.txt"
                    ),
                    env=env,
                )
            finally:
                try:
                    await self.exec_as_agent(
                        environment,
                        command=(
                            "src=$(find ~/.agy/antigravity-cli/tmp -type f "
                            "\\( -name 'session-*.jsonl' -o -name 'session-*.json' \\) "
                            "-printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -n1 "
                            "| awk '{print $2}'); "
                            'if [ -n "$src" ]; then '
                            'cp "$src" "/logs/agent/antigravity-cli.trajectory.${src##*.}"; '
                            "fi"
                        ),
                    )
                except Exception:
                    pass
        finally:
            failed = sys.exc_info()[0] is not None
            if hook_attempted:
                try:
                    await self.exec_as_agent(
                        environment, command=self._build_restore_web_hook_command(), env=env
                    )
                except Exception as exc:
                    self.logger.warning("Antigravity cleanup failed: %s", exc)
                    if not failed:
                        raise


class GeminiApiAntigravityCli(ClosedBookAntigravityCli):
    """Use Gemini API-key authentication with the benchmark web controls."""

    def _build_settings_config(
        self, model: str | None = None
    ) -> tuple[dict | None, str | None]:
        config, alias = super()._build_settings_config(model)
        config = config or {}
        config["modelProvider"] = "gemini"
        return config, alias
