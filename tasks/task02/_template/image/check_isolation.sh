#!/bin/sh
# Adversarial isolation check. Run INSIDE the agent container AS THE AGENT USER:
#
#   docker run --rm -u agent --entrypoint /opt/check_isolation.sh \
#       -v "$ROBOCASA_ASSET_DIR:/opt/src/robocasa/robocasa/models/assets:ro" \
#       rlebench-task02-agent:dev
#
# WHY EACH CHECK EXISTS, and what it is defending: tasks/task02/README.md, "Isolation".
# That reasoning is kept in the repo rather than here ON PURPOSE. This script has to be
# readable to be executable, and it executes as the agent -- so every word in it is
# addressed to an agent as much as to an operator. It states what is asserted; it does
# not enumerate where a secret might be found.
#
# Exits non-zero on the FIRST leak, so this can gate a build.
set -u

fail=0
ok()   { echo "  ok      $1"; }
leak() { echo "  LEAK    $1" >&2; fail=1; }

echo "[isolation] running as uid=$(id -u) user=$(id -un)"
if [ "$(id -u)" = "0" ]; then
    echo "[isolation] REFUSING: must run as the unprivileged agent user, not root" >&2
    exit 2
fi

echo "[isolation] the simulator must be unreachable"
ls /opt/src >/dev/null 2>&1 && leak "/opt/src is listable" || ok "/opt/src blocked"
for mod in robocasa robosuite; do
    if python -c "import ${mod}" 2>/dev/null; then
        leak "import ${mod} succeeded -- the agent could step off-meter"
    else
        ok "import ${mod} blocked"
    fi
done

echo "[isolation] the scoring side must be unreachable"
ls /opt/private >/dev/null 2>&1 && leak "/opt/private is listable" || ok "/opt/private blocked"
for mod in \
    harness.config \
    harness.stages \
    harness.session \
    harness.evaluation \
    harness.ledger \
    harness.service \
    harness.daemon_main \
    harness.env \
    harness.scoring \
    harness.verify_main \
    harness.transcript \
    harness.control \
    harness.debug
do
    if python -c "import ${mod}" 2>/dev/null; then
        leak "import ${mod} succeeded -- the agent could see or fake the score"
    else
        ok "import ${mod} blocked"
    fi
done

echo "[isolation] the evaluation split must not be discoverable"
# Asserted on the CHANNEL, not on the value: nothing named RLEBENCH_EVAL_* may reach the
# agent's environment at all, and the root-only file must be unreadable. Checking a value
# would pass vacuously in a bare container, where nothing configured one.
for var in RLEBENCH_EVAL_TASKS RLEBENCH_EVAL_SALT; do
    if [ -n "$(printenv "${var}" 2>/dev/null)" ]; then
        leak "${var} is set in the agent's environment; it must not travel that way"
    else
        ok "${var} not in the agent's environment"
    fi
done
# The root-only file, asserted directly rather than inferred from its parent's mode.
if cat /opt/private/eval_split.txt >/dev/null 2>&1; then
    leak "/opt/private/eval_split.txt is readable"
else
    ok "the evaluation split file is unreadable"
fi
# Not the same surface as `printenv` if anything re-exported them.
if tr '\0' '\n' < /proc/self/environ 2>/dev/null | grep -q '^RLEBENCH_EVAL'; then
    leak "RLEBENCH_EVAL_* readable from /proc/self/environ"
else
    ok "/proc/self/environ carries no RLEBENCH_EVAL_*"
fi
# Operator-only, and unset in every real run: passing a known name proves the search
# would find one if it were there, rather than proving grep works.
if [ -n "${RLEBENCH_EXPECT_EVAL_TASK:-}" ]; then
    if grep -rl "${RLEBENCH_EXPECT_EVAL_TASK}" /opt/rlebench /workspace 2>/dev/null \
        | head -1 | grep -q .; then
        leak "${RLEBENCH_EXPECT_EVAL_TASK} appears in an agent-readable file"
    else
        ok "${RLEBENCH_EXPECT_EVAL_TASK} appears nowhere the agent can read"
    fi
fi

echo "[isolation] the live debug tree must be unreachable"
# Ownership cannot protect it (the agent may share the host user's uid), so the guarantee
# is that /opt/private is 0700 and the agent has no +x to traverse into the mount
# underneath it.
ls /opt/private/debug >/dev/null 2>&1 && leak "/opt/private/debug is listable" \
    || ok "/opt/private/debug blocked"
# The evaluation-trial videos are recorded here until the verifier exports them.
ls /opt/private/media >/dev/null 2>&1 && leak "/opt/private/media is listable" \
    || ok "/opt/private/media blocked"
# Not redundant with the listing above: a directory can deny listing and still serve a
# file whose name is known.
cat /opt/private/debug/status.json >/dev/null 2>&1 \
    && leak "debug status.json readable" || ok "debug status.json unreadable"

echo "[isolation] the ledger must be neither readable nor writable"
ls /var/lib/rlebench >/dev/null 2>&1 && leak "ledger dir listable" || ok "ledger dir blocked"
if [ -e /var/lib/rlebench/cost.jsonl ]; then
    cat /var/lib/rlebench/cost.jsonl >/dev/null 2>&1 \
        && leak "ledger readable" || ok "ledger unreadable"
    echo forged >> /var/lib/rlebench/cost.jsonl 2>/dev/null \
        && leak "ledger appendable" || ok "ledger not appendable"
fi

echo "[isolation] the handoff channel MUST work -- isolation that blocks the task is a bug"
python -c "from harness.client import ToolsmithClient" 2>/dev/null \
    && ok "client importable" || leak "client NOT importable"
python -c "from harness.controller import ObsSpec" 2>/dev/null \
    && ok "controller contract importable" || leak "controller contract NOT importable"
# The development agent writes here and the evaluation agents import from it. If this is not
# writable, or not on the path, the task has no handoff at all.
if [ -w /workspace/agent_harness ]; then ok "/workspace/agent_harness writable"
else leak "/workspace/agent_harness NOT writable"; fi
case ":${PYTHONPATH:-}:" in
    *":/workspace/agent_harness:"*) ok "/workspace/agent_harness on PYTHONPATH" ;;
    *) leak "/workspace/agent_harness NOT on PYTHONPATH -- controllers would not import" ;;
esac
# The handoff directory must not be called `harness`. WORKDIR is /workspace, and a
# package of that name here would win over the root-only tree for anything the harness
# runs with `python -m` from this directory.
if [ -e /workspace/harness ]; then leak "/workspace/harness exists -- shadows the root-only package"
else ok "/workspace/harness absent"; fi

if [ "${fail}" -eq 0 ]; then
    echo "[isolation] PASS"
else
    echo "[isolation] FAIL" >&2
fi
exit "${fail}"
