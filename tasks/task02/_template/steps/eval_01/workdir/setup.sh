#!/bin/sh
# Reserved filename: Harbor runs this after uploading workdir/ and BEFORE the agent, as
# the step's agent user, with cwd = WORKDIR.
#
# It opens nothing. The previous step's root collect hook does that, on the harness's own
# plane. All this does is NOTICE if that failed, because Harbor runs collect hooks
# best-effort ("failures never abort the trial") -- so a failed `next-trial` would
# otherwise be silent and this agent would spend its whole step driving a trial that was
# already scored, or none at all.
#
# MUST NOT exit non-zero. A failing setup.sh makes Harbor skip the rest of the step,
# including its collect hooks -- which on the last step means the ledger is never sealed,
# and on any step means the evaluation never advances again.
set -u

# SCRUB THE PREVIOUS STEP'S VERIFIER OUTPUT BEFORE THIS AGENT STARTS.
#
# /logs/verifier is shared by every step and Harbor empties it only immediately BEFORE
# each verifier runs -- which is after this agent has come and gone. So without this, the
# whole of `diagnosis.json` from the previous trial is sitting there, readable, for the
# entire step: the reward so far, the previous trial's stage vector, and the NAME OF THE
# HELD-OUT TASK, which every other channel goes out of its way to withhold (the trial
# descriptor omits it deliberately; `task_info` returns None for it during evaluation).
#
# Harbor chmods the directory 0777 at each reset, so this runs fine as the agent user --
# and it runs BEFORE the agent, so an agent cannot skip it.
#
# Nothing of value is lost: Harbor has already parsed and recorded this step's reward by
# now, and its own reset would delete these files one phase later anyway.
rm -f /logs/verifier/reward.json /logs/verifier/reward.txt \
      /logs/verifier/diagnosis.json 2>/dev/null || true

python - <<'PY' || true
from harness.client import ToolsmithClient

try:
    with ToolsmithClient() as sim:
        phase = sim.status().get("phase")
        if phase != "evaluation":
            print(f"[setup] WARNING: phase is {phase!r}, expected 'evaluation' -- the "
                  "previous step's control hook did not take effect")
        else:
            trial = sim.trial_info() or {}
            print(f"[setup] trial {trial.get('index')} of {trial.get('total_trials')} "
                  "is open")
            if trial.get("instruction") is None:
                print("[setup] WARNING: the trial carries no instruction; the scene may "
                      "have failed to build")
except Exception as exc:  # noqa: BLE001
    print(f"[setup] could not reach the daemon ({exc})")
PY

# Do not leave harness plumbing where the agent can read it. Never fatal.
rm -- "$0" || echo "[setup] could not remove $0; continuing" >&2
