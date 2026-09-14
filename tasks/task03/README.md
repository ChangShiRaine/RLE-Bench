# Task 03 — Tabletop reasoning

Five tabletop tasks solved from RGB images and robot proprioception through a
small public client. The simulator, scene definitions, scoring and state records
are root-only in the agent's container, and the agent reaches the simulator only
through a metered socket.

| task                        | goal                                              | budget               | image                                  |
| --------------------------- | ------------------------------------------------- | -------------------- | -------------------------------------- |
| `01-tower-max-height`       | build a tall stable tower                         | 50,000 steps         | `rlebench-task03-agent:dev`            |
| `02-cantilever-overhang`    | extend a stack beyond the table edge              | 50,000 steps         | `rlebench-task03-agent:dev`            |
| `03-balance-coins`          | identify the heavy cube using the balance         | 50,000 steps         | `rlebench-task03-agent:dev`            |
| `04-rubik-cube`             | solve a physical 2×2 cube with two arms           | 50,000 steps         | `rlebench-task03-pocket-agent:dev`     |
| `05-hidden-center-of-mass`  | identify hidden ballast quadrants in three boxes  | 12,000 steps per box | `rlebench-task03-hidden-com-agent:dev` |

## How it works

Tasks 01–03 share `TabletopClient` (`observe`, `step`, `move`, `reset`, `finish`)
over a PandaOmron arm, gripper, mobile base and torso. Each is one nine-hour
attempt: `reset()` costs one step and restores the same private initial scene,
and `finish()`, budget exhaustion or agent exit scores the final simulator state.
No object poses, skills or depth are exposed; `03-balance-coins` additionally
provides live cube and pan centres, with masses and the heavy cube kept private.

Task 04 uses two tilted Panda arms, one front RGB camera with optional depth,
`SpeedrunClient`, and a metered drop rescue. There is no phase split; reset
ends the only trial and physical success ends it automatically.

Task 05 uses a fixed Panda and `HiddenCOMClient`, with one immutable quadrant
submission per box and no reset.

### Scoring

```
reward = max(0, quality - 0.5 * steps_used / step_budget)
```

with quality and reward in [0, 1], charging every step including resets. Task 05
applies this per box and averages the three results. Task 04's quality decays
with quarter-turn overhead over the optimal solve (optimal scores 1, twice
optimal 0.5); unclassified transitions cap the reward at 0.5, and an unsolved
cube scores zero.

## Layout

```
build_levels.py          the generator: --emit-all, --list, --check
build_assets.py          stages the public client and the private tabletop package
build_pocket.py          emits 04-rubik-cube from _template/pocket and tabletop/pocket
build_hidden_com.py      emits 05-hidden-center-of-mass from _template/hidden_com
tabletop/                scenes, session, service, scoring, analytic checks, and the
                         pocket/ and hidden_com/ sub-packages
dev/                     previews, camera and isolation checks for the pocket and hidden-COM tasks
_template/
  task.toml.in, instruction.md.in, image/, environment/, tests/, solution/   tasks 01–03
  pocket/                task 04: task.toml, instruction, environment, tests
  hidden_com/            task 05: task.toml, instruction, environment, tests, solution

NN-slug/                 GENERATED: the five Harbor tasks
image/                   GENERATED: the shared tabletop build context
harness/                 GENERATED: the host-side package (task01's daemon + harness.tabletop)
```

The private daemon package is staged from task01 with task03's session, service
and scenes layered under `harness.tabletop`. The public tree of tasks 01–03
holds only `harness/__init__.py` and `harness/client.py`; API and geometry
documentation lives in the client modules. Nothing here needs the RoboCasa
dataset or perception models; MuJoCo and robosuite use the simulator layer's
pins.

## Running it

```bash
make task03            # emit the five tasks and build the three images
make task03-assets     # regenerate only
make task03-clean
```

`rlebench prepare task03` runs the build.

```bash
rlebench run task03/01-tower-max-height -a <agent> -m <model>    # one task
rlebench run task03 -a <agent> -m <model> -d cuda:0              # all five
rlebench run task03 -a oracle                                     # the reference replays
```

By hand:

```bash
harbor run -p tasks/task03 -i 01-tower-max-height -a <agent> -m <model> --override-gpus 0
```

Reference solutions for tasks 01–03 replay recorded control actions through the
public metered API; task 05's Oracle estimates tilt from RGB. Task 04 ships no
Oracle.

## Checking the guarantees

```bash
python3 tasks/task03/build_levels.py --check
python3 tasks/task03/build_assets.py --check
docker run --rm -u agent --entrypoint /opt/check_isolation.sh rlebench-task03-agent:dev
make test TASK=task03
```

The isolation check runs as the agent and asserts that private files and modules
are unreadable, privileged requests are refused, and only the authorised
observation fields are served. The pocket and hidden-COM images have their own
checks under `dev/`.
