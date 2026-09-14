#!/bin/bash
# Oracle, development step. Harbor copies this to /solution/ and runs it as the agent.
#
# WHAT THIS PROVES: the harness end to end -- the metered socket is reachable,
# interaction is charged and capped, code written here survives into the evaluate step,
# and the run seals and scores. It does NOT prove the task is solvable; there is no
# reference solution, so expect a reward of ~0. Read the diagnosis, not the reward.
set -euo pipefail

mkdir -p /workspace

# Written to disk because the oracle cannot resume (OracleAgent leaves SUPPORTS_RESUME
# False, so the documented oracle run omits --resume-trajectory). Disk is what carries
# here -- and it is the right habit for a real agent too.
cat > /workspace/skills.py <<'PY'
"""Helpers written during development, reloaded during evaluation."""


def nudge_action(gain: float = 0.2) -> list:
    """One action: drive the end-effector forward in the BASE frame.

    Deliberately trivial -- a protocol demonstration, not a solution. The observation
    carries camera images and the robot's own state and nothing about where objects are,
    so there is no target to subtract; a real solution has to find one in the images.

    What it does take seriously is the frame: OSC actions are in the ROBOT BASE frame, so
    a fixed +x command is a fixed direction relative to the robot however the base was
    placed. Computing a direction in world coordinates and forgetting to rotate it is the
    silent failure the instructions warn about -- nothing raises, the arm simply drifts.
    """
    action = [0.0] * 12
    action[11] = -1.0            # arm mode, not base mode
    action[0] = gain             # +x in the base frame
    return action


def nudge(sim, steps: int = 200, gain: float = 0.2, **kw) -> dict:
    """Send `steps` nudges as one batch. The harness stops early if the episode ends
    part-way, so `res["steps"]` is what was actually applied."""
    return sim.step([nudge_action(gain)] * steps, **kw)
PY

python - <<'PY'
import sys
sys.path.insert(0, "/workspace")

from harness.client import SpeedrunClient, ObsSpec
from skills import nudge, nudge_action

with SpeedrunClient() as sim:
    info = sim.task_info()
    print(f"[oracle/develop] task={info.get('task')} action_dim={info.get('action_dim')}")
    print(f"[oracle/develop] instruction={info.get('instruction')!r}")

    look = sim.observe(ObsSpec(width=256))
    print(f"[oracle/develop] observe: resolution={look['resolution']} "
          f"keys={sorted(look['obs'])[:3]}")

    # A frame on disk is the only evidence of orientation a log can carry.
    try:
        import numpy as np
        from PIL import Image

        arr = np.asarray(look["obs"]["robot0_agentview_left_image"]).astype("uint8")
        Image.fromarray(arr).save("/workspace/oracle_agentview.png")
        print("[oracle/develop] wrote /workspace/oracle_agentview.png (must be upright)")
    except Exception as exc:  # noqa: BLE001
        print(f"[oracle/develop] could not save a frame: {exc}")

    sim.reset()
    res = nudge(sim, steps=200)
    print(f"[oracle/develop] batch: steps={res['steps']} ended={res['ended']}")

    # Coming back costs nothing -- only steps are charged.
    for _ in range(3):
        res = sim.step(nudge_action())
        if res["episode_over"]:
            break
    print(f"[oracle/develop] three single steps, episode_over={res['episode_over']}")

    # The cheap path: no pictures, nothing to render or transfer.
    res = nudge(sim, steps=5, obs_spec=ObsSpec(cameras=()))
    imgs = [k for k in (res.get("obs") or {}) if k.endswith("_image")]
    print(f"[oracle/develop] blind batch: steps={res['steps']} images={imgs}")

    status = sim.status()
    print(f"[oracle/develop] spent={status['steps_used']} "
          f"remaining={status['steps_remaining']} phase={sim.phase()}")

    # The readiness signal: it ends this phase and nothing else. The harness opens the
    # scored phase afterwards, on its own plane.
    print(f"[oracle/develop] end_development -> {sim.end_development()}")

# No close(): the session must stay open for the evaluate step, and the harness seals it
# via the collect hook regardless of what the agent does.
PY

echo "[oracle/develop] helpers saved to /workspace/skills.py"
