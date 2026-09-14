# Task 01 — RoboCasa speed-run

The agent learns a RoboCasa kitchen manipulation task through a metered simulator, then
passes one terminal evaluation. The score is **capability first, sample efficiency
second**: how many evaluation trials it solves, and how little simulator interaction it
spent learning to.

It is a matrix: three harness levels x five subtasks = fifteen Harbor tasks, generated
from one template.

| level  | what the agent is given                                                                                                     | image                          |
| ------ | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| **L1** | the low-level action API only — three cameras, proprioception, no object poses. The control condition                       | `rlebench-task01-l1-agent:dev` |
| **L2** | L1 plus the harness library and its manual (`/opt/HARNESS_MANUAL.md`) — frames, camera geometry, point clouds, primitives | `rlebench-task01-l2-agent:dev` |
| **L3** | L2 plus privileged simulator state — object and fixture poses, extents, grasp flags — under `priv_*` keys                  | `rlebench-task01-l3-agent:dev` |

Subtasks: `01-open-fridge`, `02-close-cabinet`, `03-turn-on-stove`,
`04-pick-place-counter-to-drawer`, `05-pick-place-microwave-to-counter`.

The task, the evaluation plan, the trial seeds and the reward weights are identical at
every level, so the gap between two levels' scores is what the harness bought.
`build_levels.py --check` asserts it.

## How it works

The simulator is reachable only through a metered Unix socket served by a root daemon
inside the agent's container. There is no `env` object in the agent's process, and every
`env.step` is counted. The agent writes whatever code it likes; `sim.step(actions)` is
the whole acting surface.

| step       | scored? | what happens                                                                                                                   |
| ---------- | ------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `develop`  | no      | learn the task and write code to disk. Only interaction spent here counts against the budget                                   |
| `evaluate` | **yes** | one terminal attempt over several trials, opened by the harness. Same loop; `reset()` scores the current trial rather than restarting it |

The agent carries its development context into `evaluate` via `harbor run
--resume-trajectory`, which `rlebench run` passes for every agent that supports resume.
Both steps share one container, so `/workspace` survives as well.

### Scoring

```
reward = 0.80 x success_rate
       + 0.20 x success_rate x (1 - development_steps / interaction_budget)
```

- `success_rate` is the fraction of evaluation trials the environment's own success
  predicate accepted. There is no channel for the agent to assert success.
- Efficiency is scaled by the success rate: a run that solves nothing scores zero
  however little it spent, and among runs that solve the same amount the cheaper wins.
- Only development steps count. Evaluation stepping is never charged.
- An unsealed, missing or tampered ledger scores zero.

The default plan is five trials, one in each of five distinct kitchens. `reward.json`
carries `trials_attempted` and `trials_total`; `diagnosis.json` carries `ledger_ok`,
`ledger_reason` and `end_reason`, which tell a failed run from one that never ran.

### Tunable knobs

Every knob is an environment variable in `task.toml`. Which table it sits in decides
who can read it: the agent can read `[environment.env]` from `/proc/self/environ`.

| knob                                              | table               | what it does                                                                        |
| ------------------------------------------------- | ------------------- | ----------------------------------------------------------------------------------- |
| `RLEBENCH_TASK`                                   | `[environment.env]` | the RoboCasa task; a runtime string, so the subtasks at a level share one image     |
| `RLEBENCH_LEVEL`                                  | `[environment.env]` | the harness level, declared not chosen: `entrypoint.sh` refuses an image mismatch   |
| `RLEBENCH_INTERACTION_STEPS`                      | `[environment.env]` | the development budget; the scorer reads it back from the ledger                    |
| `RLEBENCH_MAX_STEPS_PER_TRIAL`                    | `[environment.env]` | per-trial step ceiling                                                              |
| `RLEBENCH_DEVELOP_SECONDS` / `_EVALUATE_SECONDS`  | `[environment.env]` | the clock reported to the agent; must mirror `[steps.agent].timeout_sec`            |
| `RLEBENCH_W_OUTCOME` / `_W_EFFICIENCY`            | `[verifier.env]`    | the reward weights, verifier-only                                                   |

The evaluation plan and the seed salt travel by the root-only file
`/opt/private/eval_plan.txt`, never by environment variable:

```
plan=1x1,2x1,3x1,4x1,5x1     # five scenes x one trial; omit for the config.py default
salt=<per-deployment secret> # omit for none
```

### The protocol

```python
from harness.client import SpeedrunClient, ObsSpec

with SpeedrunClient() as sim:
    info = sim.task_info()                   # action layout, dim, budgets, clock
    look = sim.observe(ObsSpec(width=384))   # free, both phases
    sim.reset()
    while True:
        res = sim.step(actions)              # one action, or a batch
        if res["episode_over"]:
            sim.reset()                      # dev: next episode | eval: next trial
    sim.end_development()                    # "I am ready"
```

`observe` is free and never steps. Only steps are charged, so the agent may work in as
many pieces, and processes, as it likes. Entering the scored phase and ending the run are
control-plane ops (`harness.control`, root-only tree) invoked from Harbor collect hooks;
the daemon refuses them to any caller whose peer uid is not root.

Observations are three cameras plus the robot's own state, filtered at the wire by
`privileged.agent_view`. `ObsSpec` chooses resolution (clamped at 512), cameras and
whether z-depth in metres comes too. L3 adds `priv_target`, `priv_objects`,
`priv_fixtures`, `priv_grasp` and `priv_object_state` on top. Actions are 12-D in the
robot base frame.

## Layout

```
_template/                    the only thing you edit
  task.toml.in                @@LEVEL@@ @@SLUG@@ @@ROBOCASA_TASK@@ @@GOAL@@
  environment/docker-compose.yaml   GPU reservation + read-only asset bind
  image/                      docker build context per level: Dockerfile, entrypoint,
                              seal_ledger.sh, check_isolation.sh, check_safety.sh
  steps/develop/              instruction.md.in + harness_L{1,2,3}.md + oracle
  steps/evaluate/             instruction.md.in + oracle + workdir/setup.sh
  tests/test.sh               the verifier: runs as root in this container

build_levels.py               the generator: --emit-all, --emit L2, --list, --check
build_assets.py               stages the two payload trees; --check asserts the boundary
harness/                      the daemon, session, ledger, scorer, client and skills

images/L1|L2|L3/              GENERATED: build context per level
  payload_agent/              client + protocol + obs (+ the library at L2 and L3)
  payload_private/            daemon, session, ledger, control, privileged, scorer
L1|L2|L3/NN-slug/             GENERATED: the Harbor tasks, five per level
```

The generated trees are gitignored. Edit `harness/` or `_template/`, then `make task01`.

Inside the image, `/opt/src` (robosuite, robocasa), `/opt/private` (daemon, scorer,
privileged state) and `/var/lib/rlebench` (the ledger) are `root:root 0700`; the agent
runs as an unprivileged uid and reaches the simulator only through the socket.

## Running it

You need a GPU with the NVIDIA Container Toolkit, Docker with compose, `harbor`, and the
complete merged RoboCasa `models/assets` tree (~22 GB).

```bash
make sim-robocasa                # simulator image + sources + dataset + .venv-robocasa
HF_TOKEN=hf_... make task01      # emit the matrix, fetch the models, build three images
```

The SAM3 weights are gated: building L2 or L3 needs a token from an account with access
to [facebook/sam3](https://huggingface.co/facebook/sam3). `TASK01_LEVELS=L1 make task01`
builds the control condition alone, with no token and no torch layer. `make task01-L2`
rebuilds one level; `make task01-clean` removes the generated trees and images.

`rlebench prepare task01` runs both steps.

### Evaluate

```bash
rlebench run task01/L1/01-open-fridge -a <agent> -m <model>            # one cell
rlebench run task01/L2 -a <agent> -m <model> -d cuda:0 cuda:1          # one level
rlebench run task01 -a <agent> -m <model> -c <vendor>/<lane>           # the matrix
```

`rlebench run` adds `--resume-trajectory`, `--override-gpus 0`, the API-host allowlist,
the dataset mount and `RLEBENCH_GPU` per cell; `--dry-run` prints the `harbor run`
lines. By hand:

```bash
export ROBOCASA_ASSET_DIR=$PWD/third_party/robocasa/robocasa/models/assets
harbor run -p tasks/task01/L1 -i 01-open-fridge -a <agent> -m <model> \
    --resume-trajectory --override-gpus 0 --yes --allow-agent-host <api host>
```

`--override-gpus 0` clears Harbor's GPU check; the compose overlay supplies the device.
The oracle has no resume support, so its run omits `--resume-trajectory`:

```bash
harbor run -p tasks/task01/L1 -i 01-open-fridge -a oracle --override-gpus 0
```

Expect reward ≈ 0 from the oracle. It walks the protocol and does not solve the task;
`diagnosis.json` with `ledger_ok: true` and `trials_attempted == trials_total` is what
says the plumbing is sound.

### Timeouts

Develop 8 h, evaluate 1 h, as `[steps.agent] timeout_sec` in `task.toml`, mirrored by
`RLEBENCH_{DEVELOP,EVALUATE}_SECONDS` so the daemon can report them. To scale one run,
set both sides to the same factor:

```bash
RLEBENCH_TIMEOUT_MULT=0.25 harbor run ... --agent-timeout-multiplier 0.25
```

A develop-step timeout kills the agent before it hands its session id to the next step,
so the evaluate step starts with no memory and drives no trial: `trials_attempted: 0`
with `AgentTimeoutError` in the develop step's `result.json`.

### Read the result

Under `jobs/<timestamp>/<task>__<id>/`:

| path                                                     | what                                                              |
| -------------------------------------------------------- | ----------------------------------------------------------------- |
| `steps/evaluate/verifier/reward.json`                    | the score, its components, `trials_attempted` / `trials_total`     |
| `steps/evaluate/verifier/diagnosis.json`                 | `ledger_ok`, `ledger_reason`, `end_reason`, steps spent            |
| `steps/*/agent/<agent>.txt`                              | what the agent did                                                 |
| `steps/evaluate/artifacts/workspace/`                    | the code the agent wrote                                           |
| `.../artifacts/logs/artifacts/speedrun/transcript.jsonl` | per-trial handovers and env diagnosis                              |
| `.../speedrun/cost.jsonl`                                | the ledger copy, for humans; the scorer reads the root-only original |

Only the `evaluate` step's reward counts. `ledger_ok: false` is a lost run, not a weak
agent; `trials_attempted: 0` means the agent never drove a graded trial.

### Watching a run live

```bash
RLEBENCH_DEBUG=1 rlebench run task01/L1/01-open-fridge -a <agent> -m <model>
```

writes `status.json`, `events.jsonl`, per-episode videos, workspace snapshots and
per-episode ground truth under `jobs/<ts>/<trial>/debug/` while the run is happening.
The tree is bind-mounted under `/opt/private`, so the agent cannot read it.
`RLEBENCH_DEBUG=1 make task01` additionally builds the image with your uid so the
agent's session trace is readable live.

### When it goes wrong

| symptom                                                   | cause                                                                          |
| --------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `required variable ROBOCASA_ASSET_DIR is missing a value` | export it, or use `rlebench run`                                               |
| Harbor rejects `gpus = 1`                                 | pass `--override-gpus 0`                                                       |
| `Agent 'oracle' does not support resume`                  | drop `--resume-trajectory`                                                     |
| `No previous session found for the working directory`     | the develop step timed out; the evaluation did not run                         |
| `task declares harness L2 but this image ships L1`        | stale level image: `make task01-L2`                                            |
| `claude: command not found` after setup                   | `[agent] user` must match `[steps.agent].user`                                 |
| healthcheck fails at start                                | the daemon did not bind; check the container log (warm-up takes ~40 s)         |

Object poses are filtered out, so the task is vision-first: an endpoint that drops
images makes it unsolvable for reasons unrelated to the agent.

## Checking the guarantees

Run after any change to the image or the payload split:

```bash
ASSETS=-v"$ROBOCASA_ASSET_DIR:/opt/src/robocasa/robocasa/models/assets:ro"

python3 tasks/task01/build_levels.py --check     # levels differ only in the harness
python3 tasks/task01/build_assets.py --check     # no scoring module in the agent tree

for L in l1 l2 l3; do                            # one per level
    docker run --rm -u agent --entrypoint /opt/check_isolation.sh $ASSETS rlebench-task01-$L-agent:dev
done
docker run --rm -u agent --entrypoint /opt/check_safety.sh $ASSETS rlebench-task01-l1-agent:dev --self-test
docker run --rm -u agent --entrypoint /opt/check_safety.sh $ASSETS rlebench-task01-l1-agent:dev
docker run --rm -u root  --entrypoint /opt/check_safety.sh $ASSETS rlebench-task01-l1-agent:dev --mount-only

docker run --rm --gpus all $ASSETS rlebench-robocasa-sim:1.0.1 \
    sh -c "python /opt/verify_assets.py && python /opt/smoke.py"   # the sim runs, frames upright

make test TASK=task01
```

`check_isolation.sh` runs as the agent and asserts that the simulator, the scoring
modules, the evaluation plan, the debug tree and the ledger are unreachable while the
client still imports. `check_safety.sh` asserts that no host data is writable from the
container. The verifier shares the agent's container and scores the root-only ledger in
place; the copy under `/logs/artifacts` is never scored.

## Build-time knobs

| where          | knob            | default            | effect                                                                    |
| -------------- | --------------- | ------------------ | ------------------------------------------------------------------------- |
| `make task01`  | `AGENT_UID`     | `1000`             | the agent's uid inside the container; `RLEBENCH_DEBUG=1` uses yours       |
| `make task01`  | `TASK01_LEVELS` | `L1 L2 L3`         | which level images to build                                               |
| `docker build` | `LEVEL`         | `L1`               | the harness the image ships, baked in as `RLEBENCH_IMAGE_LEVEL`           |

The build refuses uid 0: the metering guarantee is uid-based.
