#!/bin/bash
# Oracle, evaluation step. See steps/develop/solution/solve.sh for what this proves --
# protocol compliance, not task mastery.
#
# The evaluation is already OPEN when this runs: the develop step's root collect hook
# opened it on the harness's own plane.
set -euo pipefail

python - <<'PY'
import sys
sys.path.insert(0, "/workspace")

from harness.client import SpeedrunClient
from skills import nudge                     # written during the develop step

with SpeedrunClient() as sim:
    phase = sim.phase()
    print(f"[oracle/evaluate] phase={phase}")
    if phase != "evaluation":
        raise SystemExit("[oracle/evaluate] the evaluation was not open")

    trials = 0

    # The same loop as development. `reset` is the only thing that advances a trial:
    # after episode_over the trial is already scored, so it starts the next one; called
    # while a trial is still live it means "give up on this one".
    for _ in range(200):
        res = nudge(sim, steps=200)
        print(f"[oracle/evaluate] steps={res['steps']} ended={res['ended']} "
              f"episode_over={res['episode_over']} "
              f"trial_ended={(res.get('info') or {}).get('trial_ended')}")

        if not res["episode_over"] and res["steps"] > 0:
            continue                   # same trial, keep going
        # Either the trial ended, or nothing was applied and the loop would spin.
        out = sim.reset()
        trials += 1
        if isinstance(out, dict) and out.get("evaluation_done"):
            print(f"[oracle/evaluate] finished after {trials} trial(s): {out}")
            break
    else:
        print("[oracle/evaluate] loop bound reached without finishing")
    # No close(): ending the run is the harness's job, done by this step's collect hook.
PY
