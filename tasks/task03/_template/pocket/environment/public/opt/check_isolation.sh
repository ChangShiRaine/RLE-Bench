#!/bin/sh
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
for mod in robosuite; do
    if python -c "import ${mod}" 2>/dev/null; then
        leak "import ${mod} succeeded -- the agent could step off-meter"
    else
        ok "import ${mod} blocked"
    fi
done

echo "[isolation] the scene definitions must be unreachable -- they carry the hidden state"
for mod in harness.tabletop harness.tabletop.scenes harness.tabletop.analytic; do
    if python -c "import ${mod}" 2>/dev/null; then
        leak "import ${mod} succeeded -- the agent could read the hidden state"
    else
        ok "import ${mod} blocked"
    fi
done

echo "[isolation] the scoring side must be unreachable"
ls /opt/private >/dev/null 2>&1 && leak "/opt/private is listable" || ok "/opt/private blocked"
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
for var in RLEBENCH_EVAL_PLAN RLEBENCH_BATTERY_SALT; do
    if [ -n "$(printenv "${var}" 2>/dev/null)" ]; then
        leak "${var} is set in the agent's environment; it must not travel that way"
    else
        ok "${var} not in the agent's environment"
    fi
done
if tr '\0' '\n' < /proc/self/environ 2>/dev/null \
    | grep -qE '^(RLEBENCH_EVAL_PLAN|RLEBENCH_BATTERY_SALT)='; then
    leak "the plan or the salt is readable from /proc/self/environ"
else
    ok "/proc/self/environ carries neither"
fi
if cat /opt/private/eval_plan.txt >/dev/null 2>&1; then
    leak "/opt/private/eval_plan.txt is readable"
else
    ok "the evaluation plan file is unreadable"
fi

echo "[isolation] the live debug tree must be unreachable"
ls /opt/private/debug >/dev/null 2>&1 && leak "/opt/private/debug is listable" \
    || ok "/opt/private/debug blocked"
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

LEVEL="${RLEBENCH_IMAGE_LEVEL:-${RLEBENCH_LEVEL:-L1}}"
echo "[isolation] harness level ${LEVEL}"

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
