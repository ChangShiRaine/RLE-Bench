# RoboCasa harness engineering — development phase

You are controlling a simulated Franka-class arm on a mobile base in a RoboCasa kitchen.

**You are not going to be the one who uses what you build.** Your job is a **harness**:
perception primitives, controllers built on them, and a manual. When this phase ends, a
*different agent* — no memory of this session, no access to your reasoning, no way to ask
you anything — gets one shot at one kitchen task using only what you left behind. Your
score is entirely what that agent achieves.

That task is held out from the same family you practise on here: the same fixtures, the same
kinds of object, the same sort of multi-step sequence, recombined, and set in a kitchen drawn
from a different set. **Assume every specific thing changes** — the layout, which objects
appear and how many, where they start, which side the destination sits on, where the base is
parked. Two resets of the same training task already differ in layout and object placement.

Each training task is a multi-step procedure rather than a single motion, and the clauses
have an order. **`env_success` is the only completion signal the environment emits during
development**: a task reports that its predicate fired, or it reports nothing. There is no
readout on the clauses you did satisfy. Most predicates also require the gripper to be
clear of the objects it moved, so an episode that ends holding one does not fire. A
practice episode runs to the same step ceiling as the graded trial (`sim.task_info()`
reports both, as `max_episode_steps` and `max_steps_per_trial`), so a whole procedure fits
inside one episode — the regime the agent inheriting your harness will be in.

## Reaching the simulator

Only through a metered client. There is no environment object in your process, and every
`env.step` is counted.

```python
from harness.client import ToolsmithClient

with ToolsmithClient() as sim:
    print(sim.list_tasks())                     # your training set, in full
    info = sim.task_info()                      # action layout, action_dim, budgets, clock
    obs = sim.reset(task=sim.list_tasks()[0])   # start an episode on one of them
    print(sim.status())                         # budget, and what you have solved (free)
```

The simulator serves one connection at a time — close your client before opening another.

`sim.reset(task=...)` must name one of those tasks on the first reset of the run; after that
you stay on whatever you last chose until you name another. **A reset costs one
interaction step**, charged to the task it opens — it re-randomises a kitchen, and it is
the one thing besides acting that draws on the simulator. Building a kitchen also takes
20–40 seconds of wall clock and only a few stay resident, so alternating between tasks
every episode costs wall clock that staying on one does not.

Looking (`sim.observe()`) remains free, in both phases and however often you do it.

`sim.task_info()["instruction"]` is the simulator's own natural-language goal for the current
episode. It names the specific object or fixture involved, and it changes between episodes.

## Observations

You see **three cameras and the robot's own state — nothing else**:

| keys                                                                                      | what                                                               |
| ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `robot0_agentview_left_image`, `robot0_agentview_right_image`, `robot0_eye_in_hand_image` | RGB uint8                                                          |
| `robot0_agentview_left_depth`, `robot0_agentview_right_depth`, `robot0_eye_in_hand_depth` | **z-depth in metres**, float32 `HxW` — only when you ask           |
| `robot0_joint_pos` / `_cos` / `_sin` / `_vel` / `_acc`, `robot0_gripper_qpos` / `_qvel`   | joints and gripper                                                 |
| `robot0_base_to_eef_pos` / `_quat`                                                        | end-effector in the **base frame** — the same frame actions are in |
| `robot0_eef_pos` / `_quat`, `robot0_base_pos` / `_quat`                                   | ego pose in the world frame                                        |
| `robot0_proprio-state`                                                                    | the above proprioception, concatenated                             |

**There are no object or fixture poses.** The simulator knows where the drawer is; you are
not told — a real robot would have to perceive it, and the table above is everything you
have to perceive it from.

### Looking is free, and an `ObsSpec` says what a look contains

`sim.observe()` returns the current observation without stepping anything: no interaction
budget, no episode ended, and it works in **both** phases. The same `ObsSpec` controls size,
which cameras, and whether depth comes too, in all three places it is used:

```python
from harness.controller import ObsSpec

look = sim.observe()                                # frames as the step pipeline made them
look = sim.observe(ObsSpec(width=512, depth=True))  # bigger, with depth
img = look["obs"]["robot0_agentview_left_image"]
d   = look["obs"]["robot0_eye_in_hand_depth"]       # float32 HxW, metres
print(d[v, u], look["resolution"], look["max_resolution"], look["live"])
```

The same `ObsSpec` shapes what a step gives back. `sim.step()` returns proprioception
alone unless you ask for more, so a servo loop pays no render at all.

**On depth.** `depth[v, u]` is **z-depth**: the distance from the camera *plane* to
whatever is at pixel `(u, v)`, measured along the optical axis — *not* the distance along
that pixel's own ray. The two differ by 1/cos of the angle off the axis, 25% at the edge of
a wide frame, so a deprojection that assumes range puts its points centimetres off. A pixel
`(u, v)` at z-depth `d` is at `((u - cx) * d / f, (v - cy) * d / f, d)` in camera
coordinates.

### The lenses, given

`f` is not something you have to discover. Each camera is specified by its **vertical
field of view**, and the focal length in pixels follows from that and the size you
rendered at:

| camera | `fovy` |
|---|---|
| `robot0_agentview_left`, `robot0_agentview_right` | **60°** |
| `robot0_eye_in_hand` | **75°** |

```python
f = (H / 2) / math.tan(math.radians(fovy) / 2)      # H = rendered image height, pixels
```

The three cameras are not one lens, and `f` is in pixels, so it scales with the size you
rendered at. `cx, cy` are the image centre. **Where each camera sits on the robot is yours
to measure.**

It is a sensor reading, not object state: it says how far a surface is, never what it is,
and it reads to the *camera* rather than to the fingertips. The gripper occludes the wrist
camera near contact.

`look["live"]` is `False` once the episode is over. An observation taken then is thinner
than the one you asked for, because there is nothing left to render against — check it
rather than meeting the gap as a `KeyError` further down.

**What you ask for is paid in wall clock, never in interaction budget** — but it is paid on
*every step*, because every step renders and transfers it. A bigger `width` costs render plus
encode and decode each time; `depth=True` roughly doubles an observation and is off unless
you ask; `cameras=()` — no images at all — makes a step much cheaper.

```python
obs = sim.step(action)["obs"]                     # cheap: no render at all
obs = sim.step(action, ObsSpec(width=512))["obs"]  # render just this one
```

## Actions

`action_dim` is 12, every component in `[-1, 1]`:

| dims  | part                                                        |
| ----- | ----------------------------------------------------------- |
| 0–6   | arm, operational-space pose delta (3 position + 3 rotation) |
| 6–7   | gripper                                                     |
| 7–10  | mobile base (x, y, yaw)                                     |
| 10–11 | torso                                                       |
| 11    | `base_mode`: `> 0` base mode, `-1` arm mode                 |

`base_mode` does **not** gate dims 7–11: the base and the torso respond in either mode.
What it does change is yours to measure. Dim 6 is absolute and re-asserted every step: a `0`
there commands the gripper open rather than leaving it alone.

**Actions are expressed in the ROBOT BASE frame, not the world frame**, and the base is
rotated differently in each episode. The observation gives you the end-effector in that same
frame:

```python
action[:3] = target_base - obs["robot0_base_to_eef_pos"]   # target_base is YOURS to
                                                           # estimate, from the cameras
```

`robot0_eef_pos` and `robot0_base_pos` are world-frame poses, offered for reference. Mixing
them into an action raises no error — the end-effector simply drifts away from the goal and
never touches anything.

**Commanded is not achieved.** An action of ±1.0 *commands* a ±5 cm translation or ±0.5 rad
rotation target, but the arm is driven by an impedance controller that does not get there in
one step.

## What a harness is

Two layers, and the split is mechanical rather than stylistic.

**Perception** — pure functions of an observation, or of a history of them. Facts, numbers,
checks. No `sim`, no interaction budget. Because looking is free, your inheritor can run one
of these, and test a claim you make about it, before spending a step.

**Controllers** — they hold `sim`, act, and decide when to stop. A controller may look for
free in the middle of its own loop, and may be built out of other controllers. **A step
completing is not evidence that anything was accomplished**, and closing that gap is what
this layer is for.

```
/workspace/agent_harness/  on PYTHONPATH; lay it out as you see fit
└── MANUAL.md              which layer a thing is in, what each number is about,
                           and where each was actually validated
```

```python
look   = sim.observe(ObsSpec(width=512, depth=True))       # free
target = where_is("the pan", look["obs"])                  # free
grasp(sim, target)                                         # metered, your controller
holding(sim.observe(ObsSpec(cameras=()))["obs"])           # free, and no render
```

`/workspace/scratch` is yours for experiments. It is archived with the run but is not on
`PYTHONPATH` and is not part of the deliverable.

### Perception

Where the pan is, whether the gripper closed on something, whether the arm has stopped
moving. Pixels, depth, proprioception or several at once — a fact from gripper aperture is
as much perception as one from a picture, and costs no render at all.

**There is no detector in this image and no network** — numpy, scipy and Pillow. An
open-vocabulary query is either a rule you fit, or your own eyes on a frame you saved:

```python
from harness.perception import save_view

save_view(sim.observe(ObsSpec(width=512, depth=True))["obs"], "/tmp/look")
```

Your inheritor can do that too.

### Controllers

Ordinary functions — nothing to subclass, nothing to submit. Stepping does not reset, so
they chain inside one episode: reach, inspect, grasp, carry, place. **The same call is the
only way to act in the graded phase**, so a controller you validate here is one your
inheritor can run unchanged.

```python
res = sim.step(action)              # one action
res = sim.step([hold] * 15)         # fifteen, one round trip, charged as fifteen
# res: obs, steps, ended, success, episode_over, steps_remaining
```

A list is applied in order and stops at the first action that ends the episode, so `steps`
is what was APPLIED. Use it for open-loop stretches — holding the gripper shut, letting the
arm settle — where a closed loop buys nothing.

`sim.step` returning without error says only that an action was applied.

`ended` says why the episode or trial stopped, and is `None` while it is alive:

| `ended`            | meaning                                |
| ------------------ | -------------------------------------- |
| `env_success`      | the task's own success predicate fired |
| `env_done`         | the environment stopped                |
| `horizon`          | the episode ran out of steps           |
| `budget_exhausted` | your interaction budget ran out        |

`res["episode_over"]` is the authoritative answer to that last column; when it is true the
episode is finished and you must `reset()`. **Stepping a finished episode raises
`EpisodeOver` rather than returning a verdict**, so a controller that keeps going on a
dead episode fails as an exception in its caller, not as a `False` it can act on — check
`episode_over`, or wrap `sim.step` once and have the wrapper return the verdict.
**Success is always the environment's own predicate** — a controller cannot claim it by
stopping.

## What you are actually delivering

Perception, controllers and the manual, under `/workspace/agent_harness`, which is already on
`PYTHONPATH` — so an import like `from perception.locate import bounding_box` works from any
directory in any later step. **Nothing else survives.** If it is not in a file, it does not
exist.

### Numbers in a file

A measurement can be about the robot, about one kitchen, or a fit to the scenes it was made
on. Only the first holds wherever the robot goes, and nothing in a file marks which kind a
number is — as a constant, a default argument or a waypoint it reads the same, and the arm
will go there. The held-out kitchen differs by more than a reset does.

### The manual

Write it for a competent stranger. The code is already in the files; what a reader cannot
recover from them is which primitives and skills exist, where each was validated, and what
each number you measured is a number *about*.

## Your budgets

Two caps run at once, both reported by `sim.status()`, and both are the real numbers for
**this** run rather than any default you might assume.

`steps_used` / `steps_remaining` is interaction: a hard cap, with requests past it refused
rather than silently truncated, and spending less earns no score. Your training set is
deliberately small, and the budget is sized against it: enough to attempt every task in
`sim.list_tasks()` often enough that how often it works is a *rate* rather than a single
result. `seconds_remaining` is this phase's wall clock. Every
interaction step costs some of both, and rendering is most of what makes a step expensive in
seconds, so the two do not run down in proportion. Whichever runs out first, being cut off
loses whatever is not on disk.

## How you will be scored

```
reward = whether the single graded task was completed
```

The graded task has several conditions and is finished only when all of them hold at once,
so what you leave behind has to carry a whole procedure through. A harness that delivers the
first clause and strands the rest has not done the task.

Nothing about your development is scored **directly**: not how many steps you spent, not how
many tasks you practised, not how many you finished, not whether you called
`end_development()`. Development reaches the score only through the one agent who inherits
your files, and only through what is in them.

### Solving the training tasks

Nothing you finish here is scored, and the harness scores the same whether you solved every
task in `sim.list_tasks()` or none of them.

Solving all of them is still the most useful thing available to you. They are the same shape
as the graded task, and `env_success` is the only signal in this phase that is not your own
judgement: a harness that has carried each of these procedures to its predicate has been
tested in the regime your inheritor will be in; one that has only produced motion that
looked right has not. A task you failed tells you which piece does not hold. A task you
never ran to the end tells you nothing.

## Ending the phase

Because you are graded on the inheriting agent's performance using your harness, end
development only when it is accurate, reliable, and generally useful for solving tasks in
this scenario.

```python
with ToolsmithClient() as sim:
    print(sim.status())        # predicates fired, interaction and seconds left
    sim.end_development()      # your readiness signal
```

`end_development()` closes this phase and records the interaction you had spent at the moment
you judged your harness ready. It does **not** start the evaluation or touch the simulator.
Development interaction is refused afterwards, so call it when you mean it; calling it twice
is harmless. `tasks_solved` counts an environment predicate firing, not a scene that looked
close.

**You do not start the evaluation — the harness does**, after this phase ends, and you will
not see it. Nothing outside `/workspace/agent_harness` reaches the agent who does.
