# Self-hosted models (experimental)

Claude Code and Codex can use models served by vLLM, SGLang, or another compatible
server. The server must support **streaming and tool calling** through the agent's
API: Anthropic Messages for Claude Code, OpenAI Responses for Codex. A server
offering only Chat Completions is insufficient for these configurations.

For example, start a small model with vLLM:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --served-model-name qwen3-small --host 0.0.0.0 --port 8000 \
  --api-key your-service-key --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser hermes
```

Use the server's model alias directly, without an `anthropic/` or `openai/` prefix:

```bash
# Claude Code: base URL without /v1
export ANTHROPIC_BASE_URL="http://MODEL_HOST:8000"
export ANTHROPIC_AUTH_TOKEN="your-service-key"
unset ANTHROPIC_API_KEY
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096
export MAX_THINKING_TOKENS=1024
rlebench run task08 -a claude-code -m qwen3-small

# Codex: base URL including /v1
export OPENAI_BASE_URL="http://MODEL_HOST:8000/v1"
export OPENAI_API_KEY="your-service-key"
rlebench run task08 -a codex -m qwen3-small
```

These use the existing `anthropic/api` and `openai/api` configurations. Codex also
accepts `OPENAI_API_BASE` (`OPENAI_BASE_URL` takes precedence) and the agent alias
`codex-cli`. For an unauthenticated server, set a nonempty placeholder API key.
For an authenticated server, use its configured key.
Claude Code's `ANTHROPIC_AUTH_TOKEN` sends Bearer authentication (used by vLLM);
use `ANTHROPIC_API_KEY` instead for a service requiring the `x-api-key` header.

`MODEL_HOST` must be reachable **from the agent container**. For a server on the
Docker host, bind it to a reachable interface and use the host's reachable IP;
`localhost` inside the agent container refers to that container. RLE-Bench adds
the endpoint host to the task's existing network allowlist.

Prefer a simple served-model alias such as `qwen3-small`: the pinned Harbor Codex
adapter keeps only the final component of slash-separated model names. Configure
the server's tool-call parser and context/output limits for the chosen model.
The token limits above are for the small-model example; tune them for your model.
An HTTP 404 on `/v1/responses` or `/v1/messages` indicates a missing API or incorrect
base path; successful `/v1/models` or chat requests alone do not establish agent
compatibility. See the serving engine's documentation for its supported protocols
and model-specific settings.

Validated with vLLM 0.25.1, Qwen3-4B-Instruct-2507, Harbor 0.21.0, Codex 0.153.4,
and Claude Code 2.1.266.
