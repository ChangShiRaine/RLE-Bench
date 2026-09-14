# RoboCasa kitchen task — you are being evaluated

**You have one graded attempt at one kitchen task. It has already started. The result is
final.**

You are controlling a simulated Franka-class arm on a mobile base in a RoboCasa kitchen.

## Read this first

Another agent spent a long session learning to control this robot and left you a harness:

```
/workspace/agent_harness/
├── MANUAL.md          read this first
├── perception ...     obs -> facts about this kitchen.  Never acts.  Free to call.
└── controllers ...    hold `sim`, act, and decide when to stop.
```

`MANUAL.md` is the only record of what that agent learned — which primitives and
controllers exist, when each applies, how to sequence them, what fails and how to tell. You
have no memory of that session and no way to ask about it. `/workspace/agent_harness` is on
`PYTHONPATH`, so an import like `from perception.locate import bounding_box` works from any
directory.

Nothing obliges you to use only what is there. You may write your own primitives and
controllers, and refit an inherited one that does not hold in this kitchen.

Your task comes from the same family of kitchen activities the harness was built against —
the same fixtures and the same kinds of sequence, recombined — but the kitchen is one that
agent did not practise in. Nothing guarantees the manual is complete, correct, or honest
about where a skill was actually validated.

## Where you are

The task is already open — you do not need to start it:

```python
from harness.client import ToolsmithClient

with ToolsmithClient() as sim:
    trial = sim.trial_info()
    print(trial["instruction"])     # what you have been asked to do
    print(sim.status())             # time remaining
```

The simulator serves one connection at a time — close your client before opening another.

`trial_info()["instruction"]` is your goal, produced by the simulator. **It is the whole
brief** — it names the specific objects and fixtures, and will usually describe several
things that all have to be true at the end.

## Looking is free

`sim.observe()` returns the current observation without stepping anything: it costs nothing
and ends nothing. You arrive knowing nothing about this kitchen, and looking is the only
thing here that costs no attempt.

```python
from harness.controller import ObsSpec

look = sim.observe(ObsSpec(width=384))
img = look["obs"]["robot0_agentview_left_image"]

look = sim.observe(ObsSpec(width=512, depth=True))
d = look["obs"]["robot0_eye_in_hand_depth"]     # float32 HxW, metres
print(d[v, u], look["live"])                    # z-depth at pixel (u, v)
```

You see three cameras and the robot's own proprioception. **There are no object or fixture
poses** — where things are has to be perceived. Depth tells you how far a surface is, never
what it is, so pairing it with a pixel you picked out yourself is what turns "somewhere
along this ray" into a target. It is **z-depth** — along the camera's optical axis, not along
the pixel's own ray — and off by default because it roughly doubles what an observation
costs. `look["live"]` is `False` once your trial is over; an observation taken then is
thinner than the one you asked for.

The lenses, if you deproject a pixel yourself: `robot0_agentview_left` and `_right` are
**fovy 60°**, `robot0_eye_in_hand` is **fovy 75°**, and `f = (H/2) / tan(fovy/2)` for the
height `H` you rendered at. Where the cameras sit on the robot is a measurement your manual
may carry.

An inherited primitive is a pure function of an observation, so a claim the manual makes
about one can be **run** here before you spend a step on it — and a proprioceptive one
needs no render at all, via `sim.observe(ObsSpec(cameras=()))`. There is no detector in
this image and no network, so the open-vocabulary channel is your own eyes on a saved
frame:

```python
from harness.perception import save_view

save_view(sim.observe(ObsSpec(width=512, depth=True))["obs"], "/tmp/look")
```

## Acting

```python
res = sim.step(action)          # one action
res = sim.step([hold] * 15)     # fifteen, one round trip, charged as fifteen
# res: obs, steps, ended, success, episode_over, steps_remaining
```

`sim.step` is **the only way to act** — the same one the harness was built through. It does
not reset, so controllers chain inside the one attempt. Action dim 6 is absolute and
re-asserted every step: a `0` there commands the gripper open rather than leaving it alone,
so an action that omits it drops what you are carrying. `obs` is proprioception unless an
`ObsSpec` asks for more. A list of actions is applied in order and stops at the first that
ends the attempt, so `steps` is what was APPLIED, and `ended` says why.

The loop is perceive, act, check:

```python
look   = sim.observe(ObsSpec(width=512, depth=True))       # free
target = where_is("the pan", look["obs"])                  # free
grasp(sim, target)                                         # your attempt
holding(sim.observe(ObsSpec(cameras=()))["obs"])           # free, and no render
```

Actions are 12-dimensional, in `[-1, 1]`, and expressed in the **robot base frame, not the
world frame**. `robot0_base_to_eef_pos` gives the end-effector in that same frame;
`robot0_eef_pos` is world-frame, and mixing it into an action raises no error — the
end-effector simply drifts away from the goal.

Every step renders and transfers whatever the controller's `ObsSpec` requests, and that is
paid in this phase's wall clock.

## How the attempt ends

| ending | what happens |
|---|---|
| you call `sim.reset()` | you have given up; the attempt is scored where it stands |
| the environment stops | the success condition fired, or the episode finished |
| the step limit | the attempt ran out of steps |
| your time runs out | scored where it stands |

`res["episode_over"]` tells you the attempt is finished. Once it is, there is nothing more to
drive — `step` raises `EpisodeOver` rather than returning a verdict, and **there is no next
task for you**: the next is someone else's. `sim.reset()` is a give-up move, not a restart.

## What counts as done

**The whole task.** Your instruction describes several conditions, and the task is
finished when all of them hold at once. It is always the environment's own condition —
neither you nor a controller can assert it. Most tasks also require the gripper to be
clear of the objects it moved, so release what you are holding and withdraw before you
stop.

## Your budget

`sim.status()["seconds_remaining"]` is this phase's wall clock, and the attempt is scored
where it stands when it ends. Nothing is saved by finishing early, and an attempt that was
never driven scores the same as one that was driven and failed.
