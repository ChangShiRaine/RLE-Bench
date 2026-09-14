# Task 07 — Bin clearing

A magnet-tipped Franka Panda empties a bin of 12–16 loose stamped steel
brackets into a single-part nest on a conveyor, as fast as possible without
dropping or denting anything. The agent ships a closed-loop policy: RGB-D
frames, joint state and wrist force-torque in, joint targets and a magnet
command out. Any method is legal, on CPU.

## How it works

The agent container holds the public simulator (`harness/`), a dev runner
that executes the exact evaluation loop on five public pile seeds, and the
vendored Panda. The deliverable is `/logs/artifacts/policy/policy.py`
exposing `make_policy()`, collected within a four-hour session.

The verifier is a separate CPU image. It reruns the policy in a sandboxed
subprocess as an unprivileged user, with the scoring code and the hidden
seeds locked away, and reads clears, floor drops, damage and bin strikes
from its own simulator state. Two 20 s smoke episodes gate the run: the
policy must load, answer the contract, keep the simulation stable and
replay bit-identically. Eight hidden 120 s episodes are then scored and
averaged. Each episode allows 360 s of cumulative policy time; simulation
and rendering are not charged.

### Scoring

Per episode, clamped to [0, 1]:

```
perfect = every part cleared and no floor drop
score   = 0.3 * perfect + 0.35 * clear_curve(clear_frac)
        + 0.2 * min(tp / 10, 1) + 0.15 * perfect * min(tp / 15, 1)
        - 0.03 * floor_drops - 0.05 * damage_events - 0.05 * bin_hits
```

`clear_curve` pays only 0.1 for the first 60 % of the bin and rises as a
cubic over the rest. `tp` is parts per minute, paced against the makespan
only when the bin was emptied and against the full budget otherwise. A
failed gate caps the total at 0.1. Weights and force limits live in
`harness/config.py`; the hidden seeds in `harness/eval_seeds.json`.

## Layout

```
build_assets.py          generates task.toml, environment/, tests/ and solution/
harness/
  spec.py, config.py, eval_seeds.json
  parts.py, scene.py, episodes.py, sensor.py     cell, piles, cameras
  runtime.py, metrics.py                          episode loop and latching trackers
  sandbox.py, scorer.py, score_task.py            verifier
  golden.py, baseline.py                          privileged and public reference pickers
  dev_runner.py                                   agent-side runner
dev/calibrate.py         pins the force limits and reference scores
instruction.md, task.toml, environment/Dockerfile, tests/{Dockerfile,test.sh}, solution/solve.sh

environment/assets, tests/{harness,rlebench,assets,score_task.py}, solution/payload   GENERATED
```

## Running it

```bash
make task07            # stage the trees and build both images
make task07-assets     # regenerate only
make task07-clean
```

```bash
rlebench run task07 -a <agent> -m <model>
rlebench run task07 -a oracle            # the public depth baseline
```

By hand:

```bash
harbor run -p tasks/task07 -a oracle
```

`verifier/report.json` carries per-episode metrics, gate diagnostics and
sandbox stderr; `verifier/media/` holds one video per scored episode.

## Checking the guarantees

```bash
make test TASK=task07
make test-task07-metrics    # scoring primitives, scene and sensor checks
make test-task07-golden     # golden picker clears the bin; anti-gaming seams
make calibrate-task07       # rewrite harness/config.py from the reference pickers
```
