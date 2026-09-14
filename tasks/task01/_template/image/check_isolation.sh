#!/bin/sh
# Adversarial isolation check. Run INSIDE the agent container AS THE AGENT USER:
#
#   docker run --rm -u agent --entrypoint /opt/check_isolation.sh \
#       -v "$ROBOCASA_ASSET_DIR:/opt/src/robocasa/robocasa/models/assets:ro" \
#       rlebench-task01-l1-agent:dev    # or -l2-agent, -l3-agent
#
# WHY EACH CHECK EXISTS, and what it is defending: tasks/task01/README.md, "Isolation".
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
# Every module in build_assets.FORBIDDEN_IN_AGENT. The two lists must stay equal;
# tests/test_speedrun_config.py asserts that they are.
for mod in \
    harness.config \
    harness.evaluation \
    harness.session \
    harness.ledger \
    harness.service \
    harness.daemon_main \
    harness.env \
    harness.scoring \
    harness.verify_main \
    harness.transcript \
    harness.control \
    harness.debug \
    harness.privileged \
    harness.perception_service
do
    if python -c "import ${mod}" 2>/dev/null; then
        leak "import ${mod} succeeded -- the agent could see or fake the score"
    else
        ok "import ${mod} blocked"
    fi
done

echo "[isolation] the evaluation plan and the seed salt must not be discoverable"
# Asserted on the CHANNEL, not on the value: neither may reach the agent's environment at
# all, whatever any entrypoint claims to have removed. Checking a value would pass
# vacuously in a bare container, where nothing configured one.
for var in RLEBENCH_EVAL_PLAN RLEBENCH_BATTERY_SALT; do
    if [ -n "$(printenv "${var}" 2>/dev/null)" ]; then
        leak "${var} is set in the agent's environment; it must not travel that way"
    else
        ok "${var} not in the agent's environment"
    fi
done
# Not the same surface as `printenv` if anything re-exported them.
if tr '\0' '\n' < /proc/self/environ 2>/dev/null \
    | grep -qE '^(RLEBENCH_EVAL_PLAN|RLEBENCH_BATTERY_SALT)='; then
    leak "the plan or the salt is readable from /proc/self/environ"
else
    ok "/proc/self/environ carries neither"
fi
# The root-only file, asserted directly rather than inferred from its parent's mode.
if cat /opt/private/eval_plan.txt >/dev/null 2>&1; then
    leak "/opt/private/eval_plan.txt is readable"
else
    ok "the evaluation plan file is unreadable"
fi

echo "[isolation] the live debug tree must be unreachable"
# Ownership cannot protect it (the agent may share the host user's uid), so the guarantee
# is that /opt/private is 0700 and the agent has no +x to traverse into the mount.
ls /opt/private/debug >/dev/null 2>&1 && leak "/opt/private/debug is listable" \
    || ok "/opt/private/debug blocked"
# The evaluation-trial videos are recorded here until the verifier exports them.
ls /opt/private/media >/dev/null 2>&1 && leak "/opt/private/media is listable" \
    || ok "/opt/private/media blocked"
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

echo "[isolation] the client MUST work -- isolation that blocks the task is a bug"
python -c "from harness.client import SpeedrunClient" 2>/dev/null \
    && ok "client importable" || leak "client NOT importable"

# THE HARNESS LEVEL decides what else must be here. `privileged.py` is checked above at
# EVERY level, L3 included: an L3 agent receives the poses over the metered socket, never
# as a module it can import and call on an env of its own.
#
# RLEBENCH_IMAGE_LEVEL is the IMAGE's own claim, which is what this audits -- the tree
# was decided at build time. RLEBENCH_LEVEL comes from task.toml and is absent under a
# bare `docker run`; entrypoint.sh cross-checks the two at container start.
LEVEL="${RLEBENCH_IMAGE_LEVEL:-${RLEBENCH_LEVEL:-L1}}"
echo "[isolation] harness level ${LEVEL}"

# The levels that ship the harness library. Hardcoded because this script runs AS THE
# AGENT, which cannot read config.py -- tests/test_speedrun_config.py cross-checks this
# line against config.SKILLS_LEVELS so the two cannot drift.
case "${LEVEL}" in
    L2|L3) wants_skills=1 ;;
    *)     wants_skills=0 ;;
esac

if python -c "import harness.skills" 2>/dev/null; then
    [ "${wants_skills}" -eq 1 ] && ok "library present, as ${LEVEL} requires" \
        || leak "library present at ${LEVEL}, which ships none"
else
    [ "${wants_skills}" -eq 1 ] && leak "library MISSING at ${LEVEL} -- the harness is the task" \
        || ok "no library, as ${LEVEL} requires"
fi

# The perception service is the library's other half, so it must be reachable wherever the
# library is -- and absent where it is not. A socket the agent cannot reach at L2 or L3
# means `segment_by_text` raises for the whole run.
if [ "${wants_skills}" -eq 1 ]; then
    [ -S /run/rlebench/perception.sock ] \
        && ok "perception socket reachable" \
        || echo "  note    perception socket absent (expected only under a bare docker run)"
    python -c "import harness.skills.perception" 2>/dev/null \
        && ok "perception client importable" || leak "perception client NOT importable"
fi

if [ "${fail}" -eq 0 ]; then
    echo "[isolation] PASS"
else
    echo "[isolation] FAIL" >&2
fi
exit "${fail}"
