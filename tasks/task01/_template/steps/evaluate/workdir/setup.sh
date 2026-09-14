#!/bin/sh
# Reserved filename: Harbor runs this after uploading workdir/ and BEFORE the agent, as
# the step's agent user, with cwd = WORKDIR.
#
# It opens nothing -- the develop step's root collect hook does that, on the harness's
# own plane. All this does is NOTICE if that failed: Harbor runs collect hooks
# best-effort, so a failed open-evaluation would otherwise be silent and the agent would
# work a whole phase against an evaluation that was never opened.
#
# MUST NOT exit non-zero. A failing setup.sh makes Harbor skip the rest of the step,
# including the seal hook -- and an unsealed ledger scores nothing.
set -u

# SCRUB THE DEVELOP STEP'S VERIFIER OUTPUT BEFORE THIS AGENT STARTS.
#
# /logs/verifier is shared by both steps and Harbor empties it only immediately BEFORE
# each verifier runs -- which is after this agent has come and gone. So without this, the
# develop step's `reward.json` and `diagnosis.json` sit there readable for the whole
# evaluation phase.
#
# Task01 publishes its reward formula in the instructions, so nothing in those files is a
# secret here. Even so, the verifier's output must not be an agent-readable channel at
# all, or every field added to it becomes one silently.
#
# Harbor chmods the directory 0777 at each reset, so this runs fine as the agent user --
# and it runs BEFORE the agent, so an agent cannot skip it. Nothing of value is lost:
# Harbor has already parsed the develop step's reward, and its own reset would delete
# these files one phase later anyway.
rm -f /logs/verifier/reward.json /logs/verifier/reward.txt \
      /logs/verifier/diagnosis.json 2>/dev/null || true

python - <<'PY' || true
from harness.client import SpeedrunClient

try:
    with SpeedrunClient() as sim:
        phase = sim.status().get("phase")
        if phase == "evaluation":
            trial = sim.trial_info() or {}
            print(f"[setup] evaluation open: trial {trial.get('index')} "
                  f"of {trial.get('total_trials')}")
        else:
            print(f"[setup] WARNING: phase is {phase!r}, expected 'evaluation' -- the "
                  "develop step's open-evaluation hook did not take effect")
except Exception as exc:  # noqa: BLE001
    print(f"[setup] could not reach the daemon ({exc})")
PY

# Do not leave harness plumbing where the agent can read it. Never fatal.
rm -- "$0" || echo "[setup] could not remove $0; continuing" >&2
