# Task 11 — Online bin packing from a running conveyor

A suction-tipped Franka Panda picks cartons off a conveyor that never stops
and packs them into a fixed 0.40 × 0.30 × 0.24 m tote, as densely, quickly
and safely as possible. The agent ships a closed-loop policy that decides,
box by box:
- whether the upcoming box still fits;
- whether it can be intercepted in time;
- how to grasp it on the move;
- where to place it for a compact load.

Any method is legal, on CPU, within a 2-hour session.

## How it works

- **Boxes**: 16 real Amazon shipping-carton types (incompetech.com box list,
  boxes only), scaled by 0.47 with one shared density, so mass is
  proportional to volume (0.18–1.41 kg). Each episode queues 3–5 copies of
  each type in a seeded shuffled order, arriving every 4–7 s on a belt
  running at 0.08–0.12 m/s. That is more boxes than a 240 s episode can use.
- **Moving targets**: the suction only seals a box that is riding the belt
  and moving with it. Boxes set down elsewhere cannot be re-picked, and a box
  released in the tote is committed.
- **Sensing**:
  - Every tick: an entry scanner's rows (each box's dims, pose and mass
    once), the belt speed, joint state, wrist F/T, and the suction command
    and vacuum switch.
  - At 5 Hz: a top-view RGB-D camera over the belt and tote, and a wrist
    RGB-D camera.
  - Everything about the tote comes from the cameras.
- **Full**: the published fit rule (`harness/packing.py`: a 1 cm heightmap,
  10 mm clearance, 60% plus centre support) declares the tote full after
  five consecutive boxes leave unpicked while none of them fits. The episode
  then ends.
- **Verifier**: a separate CPU image reruns the policy in a sandbox on two
  20 s smoke episodes and eight hidden 240 s episodes.
  - The smoke gate checks load, contract, stability and bit-determinism.
  - The sandbox runs as uid 65534 with `/tests` locked; packages are capped
    at 1 GiB, and stray processes are killed every episode.
  - Metrics come from the verifier's own simulator state. Outputs are
    written privately and published only after the sandbox is gone.

### Scoring

Per episode, clamped to [0, 1], then averaged:

```
u = min(utilization / 0.70, 1)    t = min(boxes_per_min / 7, 1)
pp = pick_place_success * min(packed / 10, 1)    full = tote full and no floor drop
score = 0.40 u + 0.25 t u + 0.20 pp + 0.15 full u
        - 0.05 floor_drops - 0.04 damaged_boxes - 0.03 tote_strikes
```

- `reward.json` also reports pick success, floor-drop rate, full rate,
  counts and gate flags.
- A failed gate caps the total at 0.1.
- Weights live in `harness/config.py`; the hidden seeds in
  `harness/eval_seeds.json`.

## Layout

```
build_assets.py          writes the Dockerfiles, test.sh and solve.sh; stages the trees below
harness/
  spec.py config.py eval_seeds.json          public constants; private weights and seeds
  boxes.py scene.py sensor.py packing.py     stream, cell, cameras, fit rule
  runtime.py metrics.py sandbox.py           episode loop, trackers, isolation
  scorer.py score_task.py                    verifier
  golden.py baseline.py dev_runner.py        privileged and public reference packers, dev runner
dev/calibrate.py         pins the force limits and reference scores
dev/harbor_*.json        Harbor Oracle and Nop evidence
instruction.md, task.toml, environment/Dockerfile, tests/{Dockerfile,test.sh}, solution/solve.sh

environment/assets, tests/{harness,rlebench,assets,score_task.py}, solution/payload   GENERATED
```

## Running it

```bash
make task11            # stage the trees and build both images
make task11-assets     # regenerate only
make task11-clean
rlebench run task11 -a <agent> -m <model>
rlebench run task11 -a oracle            # the public baseline, expected reward 0.6047
```

By hand: `harbor run -p tasks/task11 -a oracle`.

Harbor enforces `no-network` with an nftables sidecar that needs
`CONFIG_NFT_FIB_INET` in the Docker host kernel. Docker Desktop's VM lacks
it, so use a Linux host or a local copy with `network_mode = "public"`.

`verifier/report.json` carries per-episode metrics, gate diagnostics and
sandbox stderr; `verifier/media/` holds one video per scored episode.

## Checking the guarantees

```bash
make test TASK=task11
make test-task11-metrics    # fit rule, trackers, reward formula (hand-derived)
make test-task11-golden     # golden packer, anti-gaming, output hardening
make calibrate-task11       # rewrite harness/config.py pins from the reference packers
```

## Reference results

| Policy | Reward | Notes |
| --- | --- | --- |
| Nop (Harbor) | 0.0 | gated: no policy |
| Public baseline, the Oracle (Harbor, 8 hidden seeds) | 0.6047 | 76 boxes packed, utilization 0.41, 5.3 boxes/min; no drops, damage or strikes; full in 8/8 |
| Privileged golden packer (calibration) | 0.715 | — |

The Oracle result reproduced bit-identically across Harbor 0.21.0 and
0.22.0 and a fresh image build; see `dev/harbor_*.json`.
