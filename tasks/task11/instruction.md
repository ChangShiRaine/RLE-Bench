# Pack the tote from a running conveyor

Cartons arrive one by one on a conveyor that never stops. A Franka Panda with
a vacuum suction cup must pick them **off the moving belt** and pack them into
a fixed-size tote: **as densely, as quickly and as reliably as possible,
without dropping or crushing anything**. This is online 3D bin packing. Your
policy decides, box by box:

- whether the upcoming box still fits in the tote;
- whether the arm can reach and grasp it in time, before it leaves the
  arm's reach or the belt;
- how to move the arm to intercept and grasp the moving box;
- where to place it so that the final load is as compact as possible.

The cell is simulated in MuJoCo (deterministic, headless, CPU only). Any
method that fits the I/O contract below is allowed: scripted state machines,
motion planning, packing heuristics, search, learned policies.

## The cell

- **Robot**: Franka Panda (7 position-controlled joints) on a pedestal at the
  origin. Its tool is a rigid 162 mm extension ending in a 40 mm suction cup.
- **Suction** (`ctrl[7] >= 128` energizes it): a box seals to the cup when
  the cup face touches one of its faces square-on (within 15°), with the
  whole cup on that face (cup centre at least 15 mm inside every edge), **and
  the box is riding the running belt** (resting on the belt surface and moving
  with it). A sealed box is held rigidly at its current pose relative to the
  cup. Only one box is held at a time. Dropping the command below 128 releases
  it instantly. The vacuum switch `seal` reports whether a box is held.
  Boxes set down anywhere other than the belt cannot be picked again, and a
  box released inside the tote is committed: it can never be re-sealed.
- **Conveyor**: runs along world −x at a constant speed drawn per episode
  from 0.08–0.12 m/s. The belt encoder reports it every tick. Boxes enter at
  x = 1.18 m and leave the cell past the overflow line at x = −0.22 m if they
  are not picked. The arm reaches the belt comfortably for about
  x ∈ [0.1, 0.6] m.
- **Tote**: interior 0.40 × 0.30 × 0.24 m, welded to its stand. The interior
  spans x ∈ [0.22, 0.62], y ∈ [−0.49, −0.19] and floor z = 0.08 m.
- **Boxes**: 16 real Amazon shipping-carton types (A1, A3, 1A1, Z1, A4, W01,
  60, 1A5, 100, 130, E4, N3, Z15, E6, 1A9, K3), scaled uniformly by 0.47. The
  smallest is 119 × 84 × 36 mm and the largest 224 × 158 × 75 mm. All boxes
  share one density (500 kg/m³), so mass scales with volume (0.18–1.41 kg).
  Each episode queues 3–5 copies of every type in a shuffled order. Arrivals
  are 4–7 s apart, with more boxes than the 240 s episode can use. Boxes ride
  flat (height vertical) with a yaw near 0° or 90° (±20°) and a lateral
  offset of up to ±50 mm. Exact dims per type are in `cell_spec["box_types"]`.
- **Scanner and belt speed**: when a box enters, a dimensioning scanner
  reports its measured dims, pose and mass once, with mild noise (about
  1.5 mm, 2 mm, 0.5° and 2%). The box reaches the robot several seconds
  later. The belt speed is reported every tick.
- **Cameras**: everything else comes from two calibrated RGB-D cameras at
  5 Hz:
  - a fixed **top-view** camera (640 × 480) that sees the whole belt, from
    entry to the overflow line, and the tote;
  - a **wrist** camera (320 × 240) on the tool, 70 mm off the tool axis,
    looking along the approach direction.

  Depth has incidence-dependent noise, and invalid pixels are NaN. The arm
  occludes what lies beneath it.
- **Not observed**: the tote's contents, how boxes shifted or settled after
  release, and where the box ended up relative to the cup. None of these are
  reported; ground-truth poses are never available.

All public geometry, timing and limits live in `/workspace/harness/spec.py`
and `/workspace/harness/config.py`, and every value your policy needs is also
in the `cell_spec` dict passed to `reset()`.

## When is the tote full?

The verifier rasterizes the boxes in the tote into a 1 cm heightmap (each box
contributes its bounding-box top over its bounding-box footprint) and applies
a published **fit rule** (`/workspace/harness/packing.py`): a box fits if, in
one of its two yaw-aligned orientations, its footprint grown by 10 mm per
side lies inside the tote, its top stays below the rim, and at least 60% of
its own footprint cells plus its centre cells are within 10 mm of the base it
would rest on.

**The tote is declared full when five consecutive boxes leave the cell
unpicked and the fit rule says none of them fit.** A box that leaves while it
would still have fitted, a new grasp, or a new box committed to the tote
resets that count. When the tote is declared full the episode ends and is
measured. Otherwise it ends after 240 s of simulated time.

## Deliverable and I/O contract

Ship a policy package in `/logs/artifacts/policy/`:

```
/logs/artifacts/policy/
├── policy.py         # required
└── ...               # anything else it needs (modules, weights, data)
```

**You have 2 hours of session time.** Copy a working policy into
`/logs/artifacts/policy/` as soon as it works at all, and update it as you
improve. An unstaged policy scores zero.

`policy.py` must expose:

```python
def make_policy():
    return obj   # obj.reset(cell_spec: dict, seed: int) -> None   once per episode
                 # obj.act(obs: dict) -> 8 floats                    at 20 Hz
```

`obs`, every tick:

| key | shape | meaning |
| --- | --- | --- |
| `qpos`, `qvel`, `tau` | (7,) | joint positions (rad), velocities, actuator torques |
| `ft` | (6,) | wrist force (N) and torque (N·m) at the tool mount, tool frame (z = approach axis), noisy |
| `suction_on`, `seal` | scalar 0/1 | commanded suction; vacuum switch (a box is held) |
| `t`, `belt_v` | scalar | sim time (s); belt speed (m/s, belt moves along −x) |
| `scans` | (M, 9) | one row per box scanned so far: `[box_id, t_scan, x, y, yaw, l, w, h, mass]` (world frame at scan time, m, rad, kg; `l` lies along the box's yaw axis) |

On 5 Hz frame ticks (every 4th tick), `obs` also carries:

| key | shape | meaning |
| --- | --- | --- |
| `top_rgb`, `top_depth` | (480, 640, 3) uint8, (480, 640) float32 | top-view camera; depth is z-depth in metres, NaN = dropout |
| `wrist_rgb`, `wrist_depth` | (240, 320, 3) uint8, (240, 320) float32 | wrist camera |
| `wrist_T_world_cam` | (4, 4) | wrist camera pose at this frame |

Cameras look along their own −z with image up = +y. `cell_spec` carries the
intrinsics `top_K` and `wrist_K`, the static `top_T_world_cam`, and the wrist
mount pose in the tool frame (`wrist_mount_pos`, `wrist_mount_quat`).

Return 7 joint position targets (clamped to the actuator ranges) followed by
the suction command 0–255. Non-finite values or a wrong shape are faults, and
the last valid command is held.

`cell_spec["model_file"]` is a compiled MuJoCo model of the cell (arm with
tool, tote, belt, no boxes; TCP site `arm_tcp` at the cup face) for your own
kinematics and planning. The `reset()` seed is only for your policy's RNG and
reveals nothing about the box stream.

## Scoring

Each hidden episode is scored from the verifier's own simulator state. Your
own logs and claims are never used. The episode score is clamped to [0, 1]:

```
u     = min(utilization / 0.70, 1)          packed box volume / tote volume
t     = min(throughput / 7.0, 1)            boxes packed per minute
pp    = pick_place_success * min(packed / 10, 1)
full  = tote declared full and no floor drops
score = 0.40*u + 0.25*t*u + 0.20*pp + 0.15*full*u
        - 0.05*floor_drops - 0.04*damaged_boxes - 0.03*tote_strikes
```

- **Packed** boxes were released by the suction inside the tote and at the
  end lie fully inside it (5 mm tolerance), below the rim, within 15° of
  upright and at rest. Utilization and throughput count packed boxes only.
- **Throughput** is packed boxes per minute, measured up to the last packed
  box if the tote was declared full and over the whole 240 s otherwise.
- **Pick-and-place success** is packed boxes divided by pick attempts (each
  off-to-on suction command is one attempt). The verifier also reports pick
  success (attempts that sealed a box) and the floor-drop rate.
- **Floor drops**: a box on the cell floor. **Damage**: a box whose contact
  force exceeds `DAMAGE_FORCE_N` for `DMG_TICKS` ticks (crushing it against
  the tote or another box), or a released box that hits something faster than
  `DROP_SPEED_LIMIT` (tossing or dropping from more than about 7 cm).
  **Tote strikes**: the tool pressing on the tote above `BIN_HIT_FORCE_N`.
  All limits are in `config.py` and `cell_spec`.

The reward is the mean over hidden episodes drawn from the same distribution
as the practice streams. Before scoring, two 20 s smoke episodes gate the
run. The policy must load, return valid commands on at least 90% of ticks,
keep the simulation stable, and replay bit-identically (seeded RNG only, no
wall clock). A failed gate caps the reward at 0.1.

## Compute and limits

- **Your session**: 2 hours in this container, which has 4 CPU cores, 4 GB of
  RAM, 10 GB of disk and no GPU.
- **Evaluation**: runs on the same CPU-only hardware. Simulated time waits
  for `act()`. `reset()` and `act()` together may use 480 s of wall time per
  scored episode and 120 s per smoke episode; simulation and rendering are
  not charged. A single `act()` that hangs for 30 s is a fault. Exceeding an
  episode's allowance ends that episode.
- **Package**: at most 1 GiB and 10,000 files, regular files only. A larger
  package, or one containing symbolic links, scores 0.
- **Processes**: any process your policy starts is killed when its episode
  ends.

## Developing

```bash
cd /workspace
python3 dev_runner.py my_policy_dir                          # practice seed 11
python3 dev_runner.py my_policy_dir --seed 57 --budget 60
python3 dev_runner.py my_policy_dir --seed 23 --video /tmp/run.mp4
```

The dev runner uses the same sandbox, physics, sensors and metrics as the
verifier and prints the same per-episode metrics, including `penalty_events`
that locate each drop, crush, impact and strike. The practice seeds are
`spec.DESIGN_SEEDS = (11, 23, 57, 101, 202)`, and
`harness.boxes.stream(seed)` generates any number of further practice streams.
`harness.packing` implements the fit rule and heightmap, and
`harness.scene` builds the cell if you want to simulate in-process.

Your policy runs in an isolated subprocess. `PYTHONPATH` is cleared, the
working directory is temporary, and under root it runs as uid 65534. Only your
package and the installed libraries are importable: `/workspace/harness` is
**not**. Bundle every module and data file as regular files (no symlinks),
resolve paths relative to `__file__`, and read the model from
`cell_spec["model_file"]`. Import or factory failures stop the run before any
episode starts.

Available: Python 3.12, numpy, scipy, mujoco 3.5.0, opencv (headless),
torch (CPU). There is no network access.
