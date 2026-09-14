# Task: GELLO Co-Design for Franka, UR5e, and xArm7

Design gravity compensation for three CAD-assembled GELLO lead arms, one for each
of these follower robots: Franka Panda, Universal Robots UR5e, and UFACTORY
xArm7. A GELLO is a passive, joint-matched input device for
intuitive teleoperation. Each supplied arm has the correct kinematics but no
springs, counterweights, or feedforward compensation, so it collapses under
gravity.

For each variant:

1. Modify `lead.xml` with springs, masses, supports, and counterweights so the
   passive arm holds pose while remaining easy to move.
2. Implement `trim.py` with model-based feedforward and per-device adaptation
   from probe measurements. It must work with your arm and a verifier-owned
   reference arm whose nominal model is supplied only during scoring.

## Workspace

All three variants are available together for the full task:

```text
/workspace/
├── franka/
├── ur5e/
└── xarm7/
```
Work in `/workspace/franka`, `/workspace/ur5e`, and `/workspace/xarm7` for the
corresponding designs.


Each variant directory contains `lead.xml`, `scene.xml`, `meshes/`, and the
public `harness/` nominal-model and analysis helpers. Public helpers do not
reproduce the final evaluation, which uses unseen physical copies, poses, and
payloads.

From a variant directory, for example:

```python
from harness.scenarios import canonical_path, compose_lead, workspace_grid
from harness.torques import residual_active_torque

model, data = compose_lead("lead.xml")
grid = workspace_grid()
print(residual_active_torque(model, data, grid[0]))
print(canonical_path().shape)
```

## Common hardware contract

- Total device mass must not exceed 2.30 kg, including counterweights.
- Added counterweights must not collide with the arm or other added structures during motion.
- Keep the supplied geometry, joint frames, and mounts unchanged.
  Stock servos are fixed components weighing 0.018 kg each.
- Stock servo cap is 0.35 Nm; damping, friction, and armature are verifier-pinned.
- You may adjust link/counterweight mass and printed-part density, and add
  printable compensation structures. Springs must have nonnegative stiffness.
  Supplied meshes use their declared inertia mode and millimetre
  scale; effective density models infill, not a measured solid-material density.
- Preserve joints `lead_joint1..N`, site `lead_ee_site`, and structural geom
  names beginning with `printed_`.
- The device must be a real, continuous, printable tabletop mechanism; long
  balance booms, dense fake printed matter, and massless connector links fail.

Preserve the supplied leader chain; it is not a scaled copy of the follower
chain. The `home` keyframe shows the reference assembly pose; the encoder
zeros are aligned to the joint-space values below. The trigger is fixed; its
geometry and mass remain present. Use explicit geom properties, without
inertial overrides or geom defaults.

Required joint-space workspaces, in radians:

| variant | DoF | home | halfwidth |
|---|---:|---|---|
| Franka | 7 | `0, -0.785, 0, -2.356, 0, 1.571, 0.785` | `0.70, 0.45, 0.50, 0.50, 0.60, 0.50, 0.70` |
| UR5e | 6 | `0, 1.1295, 2.3889, 0.4709, -1.8642, -0.1313` | `0.70, 0.45, 0.50, 0.50, 0.60, 0.50` |
| xArm7 | 7 | `0, -0.247, 0, 0.909, 0, 1.15644, 0` | `0.70, 0.45, 0.50, 0.50, 0.60, 0.50, 0.70` |

## Software contract

Implement `plan_probe` to choose additional calibration poses from an initial measurement, following the interface documented in `harness/probe_protocol.py`.

Every variant's `trim.py` must provide:

```python
def make_trim(model):
    def trim(q):
        return tau
    return trim

def adapt(model, probe):
    return params

def make_trim_adapted(model, params):
    def trim(q):
        return tau
    return trim
```

`q` and the returned torque have the variant's number of joints. `model` is
always the nominal model: either your `lead.xml` or the hidden verifier-owned
reference. `probe` is a list of `(q_target, q_settled)` pairs measured on an
unknown physical copy while feedforward is off. Hidden devices perturb link
masses and add an unknown handle payload. Feedforward is sampled outside the
simulation loop, clipped to the shared servo budget, and held fixed during
each scenario. Do not depend on a particular XML path or hard-coded design
parameters.

## Deliverables and time budget

Write these six entry files and each model's referenced `meshes/` directory:

```text
/logs/artifacts/franka/lead.xml
/logs/artifacts/franka/trim.py
/logs/artifacts/ur5e/lead.xml
/logs/artifacts/ur5e/trim.py
/logs/artifacts/xarm7/lead.xml
/logs/artifacts/xarm7/trim.py
```

Agent execution has a hard timeout of **10,800 seconds (3 hours)**. Only files
under `/logs/artifacts/` are collected. Write loadable models and importable
trim modules there early, then overwrite them as the designs improve.
Keep mesh references relative to the model; `/workspace/` paths do not exist
in the separate verifier.

## Scoring

Each variant is independently scored for validity/buildability, passive
hardware performance, nominal and adapted software on a verifier-owned arm,
and combined hold, backdrive, servo headroom, and push recovery on unseen
devices. The final task reward is the mean of the Franka, UR5e, and xArm7
rewards. Invalid devices cannot earn beyond the validity gate cap.
