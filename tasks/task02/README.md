# Task 02 — RoboCasa harness engineering

Can an agent build a harness — perception primitives, controllers and a manual — that a
**different** agent, with no shared context, can pick up and apply to a task neither has
seen?

RoboCasa groups its composite kitchen tasks into activity groups. Task02 takes one group,
holds out one member for grading, and lets the agent practise on three others chosen so
that every atomic primitive the held-out task needs is covered. One Harbor task per group,
fifteen groups, one shared image.

```
develop     agent A: practises on the training members, writes
            /workspace/agent_harness/{MANUAL.md, perception/, controllers/}
eval_01     agent B: fresh context, reads the manual, one shot at the held-out task
...
eval_05     agent F: same, an independent draw          reward = mean stage score
```

| band   | slug                       | held-out task            |
| ------ | -------------------------- | ------------------------ |
| EASY   | `01-washing-dishes`        | `DumpLeftovers`          |
| EASY   | `02-sauteing-vegetables`   | `PlaceVegetablesEvenly`  |
| EASY   | `03-baking`                | `PastryDisplay`          |
| EASY   | `04-reheating-food`        | `SimmeringSauce`         |
| EASY   | `05-chopping-food`         | `ClearCuttingBoard`      |
| MEDIUM | `06-setting-the-table`     | `AlignSilverware`        |
| MEDIUM | `07-portioning-meals`      | `PortionHotDogs`         |
| MEDIUM | `08-defrosting-food`       | `DefrostByCategory`      |
| MEDIUM | `09-arranging-buffet`      | `PlaceBeveragesTogether` |
| MEDIUM | `10-serving-beverages`     | `MatchCupAndDrink`       |
| HARD   | `11-loading-fridge`        | `MoveFreezerToFridge`    |
| HARD   | `12-managing-freezer-space`| `SeparateFreezerRack`    |
| HARD   | `13-clearing-table`        | `CandleCleanup`          |
| HARD   | `14-microwaving-food`      | `PlaceMicrowaveSafeItem` |
| HARD   | `15-storing-leftovers`     | `StoreLeftoversInBowl`   |

The EASY and MEDIUM bands are open-surface work; the HARD band needs hinged doors or
enclosed cavities. `build_groups.py --list` prints every group with its training set and
primitive coverage.

## How it works

The simulator is reachable only through a metered Unix socket served by a root daemon in
the agent's container. The development agent sees three cameras and proprioception, never
object poses, and there is no detector in the image: perception is a rule the agent fits
or its own reading of a saved frame. Its deliverable is `/workspace/agent_harness`, which
is on `PYTHONPATH` in every later step.

Each graded trial is its own Harbor step with a fresh agent. Runs go **without**
`--resume-trajectory`, so nothing but the harness directory reaches an evaluation agent.
A root collect hook advances the daemon between steps (`open-evaluation`, `next-trial`)
and seals the ledger after the last trial.

### Scoring

```
reward = mean stage score over all planned trials
trial_score = (best_count - reset_count) / (total - reset_count)
```

Composite success predicates are conjunctions, and RoboCasa scores them 0 or 1. Each
held-out task has a private stage function in `harness/stages.py` that mirrors its
`_check_success` conjunct by conjunct, so a trial earns the fraction of available
conjuncts it completed at its best moment. A solved trial scores 1, a do-nothing trial 0,
an unreached trial 0. `success_rate` (RoboCasa's own predicate) is reported alongside and
is not the reward. Development interaction is a hard cap that earns nothing.

Every step's verifier writes a cumulative reward, and `multi_step_reward_strategy =
"final"` keeps the last one, so an aborted run scores what it earned.

### The one secret

Task02 hides **which task it is graded on**. An agent that learned the held-out name would
practise it, and the benchmark would measure memorisation instead of transfer. So
`config.py` and `stages.py` ship only in the root-only `/opt/private`; the splits are a
fixed table there, selected by `RLEBENCH_GROUP`, which names only the group; no trial
descriptor, status reply or observation names a task; `build_assets.py --check` fails if a
held-out name appears anywhere in the agent-readable tree; and each `eval_NN` step's
`setup.sh` scrubs the previous verifier's output before its agent starts.

### Knobs

Every knob is an environment variable in `task.toml`. `[environment.env]` is readable by
the agent; nothing secret may go there.

| knob                            | default                    | what it does                                                     |
| ------------------------------- | -------------------------- | ---------------------------------------------------------------- |
| `RLEBENCH_INTERACTION_STEPS`    | 75000                      | development step budget; a hard cap worth no score               |
| `RLEBENCH_MAX_STEPS_PER_TRIAL`  | 5000                       | per-trial ceiling, and the development episode ceiling with it   |
| `RLEBENCH_ENV_CACHE`            | 3                          | resident development environments                                |
| `RLEBENCH_DEVELOP_SECONDS`      | 28800                      | the clock the daemon reports; mirror `[steps.agent] timeout_sec` |
| `RLEBENCH_TRIAL_SECONDS`        | 3600                       | per-trial clock; mirror each `eval_NN` step's `timeout_sec`      |
| `RLEBENCH_HARNESS_DIR`          | `/workspace/agent_harness` | the handoff directory                                            |
| `RLEBENCH_DEBUG` (shell)        | 0                          | live operator view under the job directory                       |
| `RLEBENCH_TIMEOUT_MULT` (shell) | 1                          | scales the reported clock with `--agent-timeout-multiplier`      |

The handoff directory is not called `harness`: Harbor execs hooks from `/workspace`, and
a package of that name there would shadow the root-only `/opt/private/harness`.

## Layout

```
build_groups.py          the GROUPS table; --emit writes a task dir per active group,
                         --trials N sets the graded-trial count, --check verifies
build_assets.py          stages the payload trees; --check asserts the boundary
harness/                 daemon, session, ledger, config (every split), stages, scorer,
                         and the agent-visible client, protocol, controller, perception
dev/                     decompose.py and the checked-in primitive audit and stage baseline
_template/
  task.toml.in           develop + one generated step per trial; @@SLUG@@ per group
  environment/docker-compose.yaml   GPU reservation + read-only asset bind
  image/                 Dockerfile, entrypoint.sh, seal_ledger.sh, check_isolation.sh
  steps/{develop,eval_01}/   instruction + oracle; --emit copies eval_01 to eval_NN
  tests/test.sh          the verifier: harness.verify_main, once per step

NN-slug/                 GENERATED: one Harbor task per group
image/                   GENERATED: the one build context
  payload_agent/         client, protocol, controller, perception   (world-readable)
  payload_private/       everything, including config and stages    (root 0700)
```

The generated trees are gitignored. Edit `harness/`, `_template/` or `GROUPS`, then
`make task02`.

### Changing groups or trials

```bash
python tasks/task02/build_groups.py --list                   # 23 groups, 15 active
python tasks/task02/build_groups.py --emit "Loading Fridge"  # one group
python tasks/task02/build_groups.py --trials 5               # rewrite config + [[steps]]
python tasks/task02/build_groups.py --check

# after a RoboCasa pin bump (needs the RoboCasa venv)
.venv-robocasa/bin/python tasks/task02/dev/decompose.py --audit \
    --out tasks/task02/dev/data/task02_primitive_audit.json
.venv-robocasa/bin/python tasks/task02/build_groups.py --recompute-split
```

Never delete a `GROUPS` entry: a group's position is its slug number. Mark it
`active=False` instead. A new held-out task needs a stage function in `stages.py`, and
every primitive it requires must have a provider in its training set; `--emit` and
`--check` refuse otherwise.

## Running it

You need a GPU with the NVIDIA Container Toolkit, Docker with compose, `harbor`, and the
complete merged RoboCasa `models/assets` tree.

```bash
make sim-robocasa                  # simulator image + sources + dataset + .venv-robocasa
make task02                        # emit every group, build the shared image
make task02-06-setting-the-table   # emit one group (and rebuild the image)
make task02-clean
```

`rlebench prepare task02` runs both steps.

### Evaluate

```bash
rlebench run task02/06-setting-the-table -a <agent> -m <model>     # one group
rlebench run task02 -a <agent> -m <model> -d cuda:0 cuda:1         # every group
rlebench run task02 -a oracle -d cuda:0                            # the protocol check
```

`rlebench run` adds `--override-gpus 0 --yes`, the API-host allowlist and the dataset
mount, and pins each group to a card via `RLEBENCH_GPU`; `--dry-run` prints the
`harbor run` lines. By hand:

```bash
export ROBOCASA_ASSET_DIR=$PWD/third_party/robocasa/robocasa/models/assets
harbor run -p tasks/task02 -i 06-setting-the-table -a <agent> -m <model> \
    --override-gpus 0 --yes --allow-agent-host <api host>
```

`--yes` is needed because the knobs are written as `${VAR:-default}` and Harbor asks
before letting a task read your shell. The oracle writes a stub harness and walks the
protocol; expect a near-zero score, and read the diagnosis rather than the reward.

Develop is 8 h and each trial 1 h. Scale one run with `RLEBENCH_TIMEOUT_MULT=0.25` plus
`--agent-timeout-multiplier 0.25`, both sides together.

### Read the result

Under `jobs/<timestamp>/<task>__<id>/`:

- `steps/eval_05/verifier/reward.json` — the run reward (the last step's cumulative
  number covers every trial).
- `steps/eval_NN/verifier/diagnosis.json` — the stage vector per trial, which is what
  explains the score.
- `steps/develop/artifacts/workspace/agent_harness/` — the deliverable every evaluation
  agent inherited.
- `steps/develop/agent/` and each `steps/eval_NN/agent/` — separate sessions.

### Watching a run live

```bash
RLEBENCH_DEBUG=1 rlebench run task02/06-setting-the-table -a <agent> -m <model>
```

writes `status.json` (with `reward_now`, including the stage vector), `events.jsonl`,
per-episode videos, per-episode harness snapshots and per-episode ground truth under
`jobs/<ts>/<trial>/debug/` while the run happens. The tree is bind-mounted under
`/opt/private`, so no agent can read it. `RLEBENCH_DEBUG=1 make task02` additionally
bakes your uid into the image so the agent's session trace is readable mid-step; build
without it for a portable image.

## Checking the guarantees

```bash
ASSETS=-v"$ROBOCASA_ASSET_DIR:/opt/src/robocasa/robocasa/models/assets:ro"

python3 tasks/task02/build_groups.py --check     # dirs, splits and image agree
python3 tasks/task02/build_assets.py --check     # no held-out name in the agent tree
docker run --rm -u agent --entrypoint /opt/check_isolation.sh $ASSETS rlebench-task02-agent:dev
make test TASK=task02
MUJOCO_GL=egl .venv-robocasa/bin/python -m pytest -q -m simulator tests/test_toolsmith_*.py
```

`check_isolation.sh` runs as the agent and asserts that the simulator, the private
modules, the split, the debug tree and the ledger are unreachable while the client and
the handoff directory still work. The `simulator`-marked tests assert that every stage
function agrees with RoboCasa's own predicate against a live environment; re-run them
after any pin bump, since the stage functions are hand-written mirrors of RoboCasa
source.

## Known limits

- The reward takes `conjuncts x trials + 1` distinct values, so the held-out task's
  conjunct count sets the resolution. Two available conjuncts is the enforced floor.
- The step cap and the trial clock are sized as a pair; raising one alone only moves
  which of them ends a trial.
- Coverage is enforced at build time, but that practising the primitives in three other
  procedures is enough to recombine them into a fourth is the hypothesis the benchmark
  tests, not something a check establishes.
- Trials run sequentially in one container, so a full run is 8 h plus five trial hours.
