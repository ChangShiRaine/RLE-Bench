#!/bin/bash
# Reference solution for the DEVELOP step. Harbor's oracle runs this in place of an agent.
#
# WHAT THIS PROVES, AND WHAT IT DOES NOT. It proves the protocol works end to end: the
# training split is reachable, controllers run closed-loop and are charged, the harness
# directory is on the path and importable from a later step, and `end_development` closes
# the phase. It does NOT score well, and it is not meant to -- no scripted controller
# solves RoboCasa composite tasks. Task01 recorded the same gap: a privileged scripted
# controller reached ~48% on the most scriptable ATOMIC task, and these are harder.
#
# It is written the way the instructions tell an agent to write: perception primitives
# under /workspace/agent_harness/perception, controllers under
# /workspace/agent_harness/controllers that consume them, a MANUAL.md beside both. That is
# the part worth exercising, because the evaluation steps import exactly those files.
set -uo pipefail

HARNESS=/workspace/agent_harness
mkdir -p "${HARNESS}/controllers" "${HARNESS}/perception"

cat > "${HARNESS}/controllers/__init__.py" <<'PY'
PY

cat > "${HARNESS}/perception/__init__.py" <<'PY'
PY

cat > "${HARNESS}/perception/basic.py" <<'PY'
"""Pure functions of an observation. They never act and cost nothing to call.

Both of these read proprioception, which is perception too and needs no render at all.
Neither locates anything -- a real harness would also carry the pixel-and-range half.
"""

import numpy as np


def aperture(obs):
    """How far the gripper is open. Compare it against an empty and a loaded reading to
    tell whether a grasp took."""
    return float(np.sum(np.abs(obs["robot0_gripper_qpos"])))


def is_settled(obs, tol=0.02):
    """Has the arm stopped moving? A scene read mid-motion is a blurred scene."""
    return bool(np.max(np.abs(obs["robot0_joint_vel"])) < tol)
PY

cat > "${HARNESS}/controllers/primitives.py" <<'PY'
"""A deliberately minimal controller library.

These are protocol demonstrations, not solutions. They show the shape a controller takes
-- holding `sim`, looping, terminating on its own condition, returning a verdict -- so the
evaluation step has something real to import and run.
"""

from perception.basic import is_settled
from harness.controller import ObsSpec

ARM, GRIPPER, BASE, TORSO, BASE_MODE = slice(0, 6), 6, slice(7, 10), 10, 11

# Proprioception only. These never look at a picture, and saying so makes every step of
# them dramatically cheaper.
CHEAP = ObsSpec(cameras=())


def zero_action():
    """A no-op action. `base_mode = -1` selects arm control, which is the mode every
    manipulation primitive below wants."""
    action = [0.0] * 12
    action[BASE_MODE] = -1.0
    return action


def settle(sim, max_steps: int = 20):
    """Hold still until the arm stops moving.

    Useful first: the arm arrives from reset with residual motion, and a perception step
    taken while it is still moving reads a blurred scene.

    It stops on a PERCEPTION PRIMITIVE rather than a step count, which is the shape worth
    copying -- `is_settled` is a function of the observation, so the caller can check the
    result for free rather than taking this function's word for it.
    """
    for _ in range(max_steps):
        res = sim.step(zero_action(), CHEAP)
        if res["episode_over"]:
            return {"settled": False, "ended": res["ended"]}
        if is_settled(res["obs"]):
            return {"settled": True}
    return {"settled": False, "ended": None}


def open_gripper(sim, steps: int = 8):
    """Open the gripper. The first half of any grasp.

    Open loop, so the whole thing is one round trip: there is nothing to decide part-way
    through, and a batch is charged exactly what it applies.
    """
    action = zero_action()
    action[GRIPPER] = -1.0
    res = sim.step([action] * steps, CHEAP)
    return {"steps": res["steps"], "ended": res["ended"]}


def nudge_forward(sim, steps: int = 20, gain: float = 0.3):
    """Drive the end-effector along +x in the BASE frame.

    The base-frame detail is the single largest gotcha in this environment: a target
    computed in world coordinates and sent unrotated makes the arm drift away from the
    goal, and nothing raises.
    """
    action = zero_action()
    action[0] = gain
    res = sim.step([action] * steps, CHEAP)
    return {"steps": res["steps"], "ended": res["ended"]}
PY

cat > "${HARNESS}/MANUAL.md" <<'MD'
# Harness manual

**This is a reference-solution stub, not a working harness.** It exists to demonstrate the
handoff format. A real development agent would replace it with primitives and controllers
that solve something and a manual that says how.

## Perception

`from perception.basic import aperture, is_settled`

Pure functions of an observation. Free to call, and callable from anywhere: between
segments, inside a controller's `act`, or on a frame saved to disk.

| primitive | returns | what the number is about |
|---|---|---|
| `aperture(obs)` | gripper opening | the robot — compare against an empty and a loaded reading |
| `is_settled(obs, tol=0.02)` | has the arm stopped | the robot |

Both read proprioception, so they need no render at all: `sim.observe(ObsSpec(cameras=()))`
is enough. **Nothing here locates anything** — the pixel-and-range half, and the transform
from a pixel to a point in the base frame, is what a real harness would add.

## Controllers

`from controllers.primitives import settle, open_gripper, nudge_forward`

| controller | what it does | when to use it |
|---|---|---|
| `settle(sim, max_steps=20)` | holds still until `is_settled` | first, after arriving, so perception reads a static scene |
| `open_gripper(sim, steps=8)` | opens the gripper | before any approach to a graspable object |
| `nudge_forward(sim, steps=20, gain=0.3)` | drives +x in the base frame | probing reach; not a solution to anything |

Each holds `sim`, loops, and returns a verdict dict. Each says whether it got what it
wanted -- `sim.step` returning is not that.

## Facts worth knowing

- **Actions are in the ROBOT BASE frame, not the world frame.** A world-frame target must
  be rotated by `quat2mat(obs["robot0_base_quat"]).T` before it is sent. Getting this wrong
  raises nothing; the arm simply drifts.
- `base_mode` (index 11) must be `-1` for arm control. All three controllers above set it.
- **Commanded is not achieved.** ±1.0 commands a ±5 cm target; the impedance controller
  does not reach it in one step. Budget step counts against the measured rate.
- `sim.step(action)` renders nothing unless handed an `ObsSpec`, which makes a servo loop
  much cheaper in wall clock. Pass a LIST of actions for open-loop stretches: one round
  trip, charged exactly what it applies.

## How to compose

Perceive, act, then check by perceiving again. The free calls cost nothing, so there is no
reason not to make them:

```python
settle(sim)
empty = aperture(sim.observe(ObsSpec(cameras=()))["obs"])   # free
open_gripper(sim)
loaded = aperture(sim.observe(ObsSpec(cameras=()))["obs"])  # free -- did it move?
```

Stepping does not reset, so the episode carries across controllers. Check
`res["episode_over"]` after each one.

## What is missing

Everything that would actually solve a task: a primitive that finds an object in a picture,
the transform from that pixel and its range to a point in the base frame, a reach that
servos to such a point, a grasp that closes on it, and a verify for each sub-goal.
MD

echo "[solve] harness written to ${HARNESS}"

# Exercise the metered protocol, so the oracle proves the development path really works
# rather than only that files can be written.
PYTHONPATH=/opt/rlebench:${HARNESS} python - <<'PY'
from harness.client import ToolsmithClient
from harness.controller import ObsSpec

from controllers.primitives import nudge_forward, open_gripper, settle
from perception.basic import aperture, is_settled

with ToolsmithClient() as sim:
    tasks = sim.list_tasks()
    print(f"[solve] training split: {len(tasks)} tasks")
    info = sim.task_info()
    print(f"[solve] action_dim={info['action_dim']} "
          f"budget={info['interaction_budget']}")

    # Practise on two of them, so the multi-task path (env cache, per-task attribution)
    # is exercised rather than merely present.
    for task in tasks[:2]:
        sim.reset(task=task)
        print(f"[solve] {task}: {sim.task_info()['instruction']!r}")
        for name, run in (("settle", settle), ("open_gripper", open_gripper),
                          ("nudge_forward", nudge_forward)):
            res = run(sim)
            print(f"[solve]   {name}: {res}")
            if res.get("ended"):
                break
            # The free half of the loop: a primitive run between segments, on an
            # observation that rendered nothing.
            look = sim.observe(ObsSpec(cameras=()))["obs"]
            print(f"[solve]     aperture={aperture(look):.4f} "
                  f"settled={is_settled(look)}")

    status = sim.status()
    print(f"[solve] spent {status['steps_used']} of "
          f"{status['steps_used'] + status['steps_remaining']} steps")
    print(f"[solve] {sim.end_development()}")
PY

# Never fail the step. The oracle's job is to prove the path works; a non-zero exit here
# reads as a broken task rather than a reference solution that scored badly.
exit 0
