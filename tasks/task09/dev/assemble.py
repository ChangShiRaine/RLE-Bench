"""Chain the measured GELLO parts into an MJCF (dev tool).

    python -m dev.assemble [--arm franka_fer] [--out DIR]

Frames are measured from mating faces and checked by dev.check_assembly;
solve_frames supplies candidate features. Joint zero angles are arbitrary
encoder offsets. The reference keyframe supplies a photographic display pose.
This is a kinematic assembly preview: contact and joint limits are not modeled.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from . import servo_proxy

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "..", "assets", "robots", "gello_mechanical")
MM = 0.001

# Servo frame: +Z is the output axis; the case extends toward -Y.
# Motor mass belongs to the case-carrying link; rotor inertia is neglected.
SERVO_MID_Z = -8.0                  # CAD midpoint between the two horn faces
SERVO_HORN_Z = 6.5
SERVO_CASE_BACK_Z = -19.5
HORN_FROM_MID = SERVO_HORN_Z - SERVO_MID_Z
SERVO_MASS = 0.018
REFERENCE_Q_DEG = (0, -90, 0, 180, 0, 180, 90, 0)

# CAD millimetres. Distal origins are shaft midpoints between the two horns;
# proximal origins are fork midpoints or the OUTSIDE mating face of a horn
# plate. Case mounting faces lie 11.5 mm from the midpoint; the upper case
# screw row lies 7.5 mm above the shaft. Forks have 0.25 mm clearance per side.
FRANKA = {
    "base": dict(prox=(0, 0, 35.5), prox_axis=(0, 0, 1),
                 dist=(-72.25, 0, 10.3 + 11.5), dist_axis=(0, 0, 1),
                 case_dir=(-1, 0, 0)),
    # Four case screws at (±11.75, ±8, 107.7), not the cradle outline's
    # spurious fitted circles. Both mounting plates are normal to X.
    "L1": dict(prox=(0, 0, 0), prox_axis=(0, 0, 1), prox_kind="horn",
               dist=(0, 0, 107.7 - 7.5), dist_axis=(1, 0, 0),
               case_dir=(0, 0, 1)),
    "L2": dict(prox=(0, 11.25, 0), prox_axis=(1, 0, 0), prox_kind="fork",
               dist=(-15.0, 82.5 + 11.5, 0), dist_axis=(0, 1, 0),
               case_dir=(-1, 0, 0)),
    "L3": dict(prox=(15.0, 0.0, 0.0), prox_axis=(0, 1, 0), prox_kind="horn",
               # J4's case extends above the L3 elbow plate in the reference.
               dist=(15.0, 38.25 + 7.5, -41.25), dist_axis=(1, 0, 0),
               case_dir=(0, -1, 0)),
    "L4": dict(prox=(44.0, 0, 0), prox_axis=(0, 0, 1), prox_kind="fork",
               dist=(2.75, 50.5 + 11.5, 0), dist_axis=(0, 1, 0),
               case_dir=(-1, 0, 0)),
    "L5": dict(prox=(0, 0, 0), prox_axis=(0, 0, 1), prox_kind="horn",
               dist=(0, 0, 93.0 - 7.5), dist_axis=(1, 0, 0),
               case_dir=(0, 0, 1)),
    # Driven end first: L6 is turned by the horn at x = 59 and carries the
    # J7 case at the (7.5, 27.5) plate, not the other way round.
    "L6": dict(prox=(59.0, 0, 0), prox_axis=(0, 0, 1), prox_kind="fork",
               dist=(-0.0232, 27.5 + 11.5, 0), dist_axis=(0, 1, 0),
               case_dir=(-1, 0, 0)),
    # The handle's J7 mating face is tilted three degrees in its STL frame.
    "handle": dict(prox=(-15.3478, -20.9418, 0),
                   prox_axis=(np.sin(np.deg2rad(3)), np.cos(np.deg2rad(3)), 0),
                   prox_kind="horn",
                   dist=(-73.0, -18.0, 0), dist_axis=(1, 0, 0),
                   mesh="../gripper/handle.STL"),
}
UR5 = {
    "base": dict(dist=(0, 11.5, 0), dist_axis=(0, 1, 0),
                 case_dir=(0, 0, -1), up=(0, 1, 0), floor=-8),
    "L1": dict(prox=(0, 23, 0), prox_axis=(0, 1, 0), prox_kind="horn",
               # J2's horn belongs to L1; its case belongs to L2.
               dist=(41.5 + 14.5, 44.5795, 0), dist_axis=(1, 0, 0)),
    "L2": dict(prox=(67.5, 44.5795, -2), prox_axis=(1, 0, 0), prox_kind="case",
               prox_case_dir=(0, 0, -1),
               dist=(67.5 - 11.5, 44.5795, 214.5), dist_axis=(-1, 0, 0),
               case_dir=(0, 0, 1)),
    "L3": dict(prox=(41.5, 44.5795, 212.5), prox_axis=(-1, 0, 0), prox_kind="horn",
               dist=(-4.475 + 11.5, 44.5795, 408.6), dist_axis=(1, 0, 0),
               case_dir=(0, 0, 1)),
    "L4": dict(prox=(21.525, 44.5795, 408.6), prox_axis=(1, 0, 0), prox_kind="horn",
               dist=(62.1, 56.0795 - 11.5, 408.6), dist_axis=(0, -1, 0),
               case_dir=(1, 0, 0)),
    "L5": dict(prox=(62.1, 30.0795, 408.6), prox_axis=(0, -1, 0), prox_kind="horn",
               dist=(50.6 + 11.5, -2.7455, 408.6), dist_axis=(1, 0, 0),
               case_dir=(0, -1, 0)),
    "handle": dict(FRANKA["handle"]),
}

# xArm's case plates include a 28.5-degree bolt pattern and a 47-degree face.
_xarm_case_dir = np.array([0, np.sin(np.deg2rad(28.5)), np.cos(np.deg2rad(28.5))])
_xarm_face = np.array([0, np.sin(np.deg2rad(47)), -np.cos(np.deg2rad(47))])
_xarm_face_dir = np.array([0, np.cos(np.deg2rad(47)), np.sin(np.deg2rad(47))])
XARM7 = {
    "base": dict(dist=(-70, 10 + 11.5, 0), dist_axis=(0, 1, 0),
                 case_dir=(-1, 0, 0), up=(0, 1, 0), floor=0),
    "L1": dict(prox=(0, -6, 2.5), prox_axis=(0, 0, 1), prox_kind="horn",
               dist=(0, 2.5 - 11.5, 56), dist_axis=(0, -1, 0),
               case_dir=(0, 0, 1)),
    "L2": dict(prox=(0, 26.5, -89), prox_axis=(0, 1, 0), prox_kind="horn",
               dist=(0, 25, -9.7 + 11.5), dist_axis=(0, 0, 1),
               case_dir=(0, 1, 0)),
    "L3": dict(prox=(0, 0, 2.5), prox_axis=(0, 0, 1), prox_kind="horn",
               dist=np.array([16.1 - 11.5, 15.002, 37.6305]) + 7.5 * _xarm_case_dir,
               dist_axis=(-1, 0, 0), case_dir=_xarm_case_dir),
    "L4": dict(prox=(-12.5, 10, 0), prox_axis=(1, 0, 0), prox_kind="horn",
               dist=np.array([-48.5, 70.758, -26.8325]) + 11.5 * _xarm_face + 7.5 * _xarm_face_dir,
               dist_axis=_xarm_face, case_dir=_xarm_face_dir),
    "L5": dict(prox=(1, 83.75, 0), prox_axis=(0, -1, 0), prox_kind="horn",
               dist=(19.7 - 11.5, 0, 0), dist_axis=(-1, 0, 0),
               case_dir=(0, -1, 0)),
    "L6": dict(prox=(2.5, 15, -15), prox_axis=(1, 0, 0), prox_kind="horn",
               dist=(2.5, -16.3 - 11.5, 23.324), dist_axis=(0, -1, 0),
               case_dir=(0, 0, 1)),
    "handle": dict(FRANKA["handle"]),
}
ARMS = {
    "franka_fer": (FRANKA, ("L1", "L2", "L3", "L4", "L5", "L6", "handle")),
    "ur5": (UR5, ("L1", "L2", "L3", "L4", "L5", "handle")),
    "xarm7": (XARM7, ("L1", "L2", "L3", "L4", "L5", "L6", "handle")),
}
REFERENCE_POSES = {
    "franka_fer": REFERENCE_Q_DEG,
    "ur5": (0, 180, -90, 90, 90, 0, 0),
    "xarm7": (180, 90, 0, 0, 180, -137, 0, 0),
}

# The handle's rear case plate is at z=18; its 30 x 16 mm bolt rectangle
# has centre (-6.0732, 2.6837). The output faces -Z, toward the trigger.
TRIGGER_SERVO_POS = np.array([-13.5732, 2.6837, -1.5])
TRIGGER_SERVO_AXIS = (0, 0, -1)
TRIGGER_CASE_DIR = (-1, 0, 0)
TRIGGER_PROX = np.array([0, 8, 0])
# The trigger extends toward +X, opposite the handle's -X finger.
TRIGGER_ROT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _align(source, target):
    """Rotation taking unit `source` onto unit `target` (minimal rotation)."""
    a, b = _unit(source), _unit(target)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        # Antiparallel: a half turn about any axis perpendicular to a.
        # (-I is a reflection, not a rotation, and poisons the chain.)
        perp = np.eye(3)[int(np.argmin(np.abs(a)))]
        perp = _unit(perp - float(perp @ a) * a)
        return 2.0 * np.outer(perp, perp) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def _quat(R):
    """mat -> MuJoCo wxyz quaternion."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s,
             (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
    return np.asarray(q) / np.linalg.norm(q)


def _vec(values):
    return " ".join(f"{float(v):.12g}" for v in values)


def _servo_frame(axis, case_dir):
    z = _unit(axis)
    y = np.asarray(case_dir, dtype=float)
    y = _unit(y - float(y @ z) * z)
    return np.column_stack([np.cross(y, z), y, z])


def build(arm: str, out_path: str, density: float = 248.0) -> str:
    table, order = ARMS[arm]
    meshdir = os.path.abspath(os.path.join(ASSETS, arm))
    servo_dir = Path(out_path).resolve().parent
    servo_proxy.write_meshes(servo_dir)

    lines = [f'<mujoco model="gello_{arm}">',
             '  <compiler angle="radian" autolimits="true" '
             f'meshdir="{meshdir}"/>',
             '  <option timestep="0.002" integrator="implicitfast" '
             'gravity="0 0 -9.81"/>',
             '  <default><geom contype="0" conaffinity="0"/></default>',
             '  <asset>']
    for part in ("base",) + order:
        source = table.get(part, {}).get("mesh", f"{part}.STL")
        lines.append(f'    <mesh name="{part}" file="{source}" '
                     f'scale="{MM} {MM} {MM}"/>')
    lines.append(f'    <mesh name="servo" file="{servo_dir / "servo.stl"}" '
                 f'scale="{MM} {MM} {MM}"/>')
    lines.append(f'    <mesh name="servo_single" file="{servo_dir / "servo_single.stl"}" '
                 f'scale="{MM} {MM} {MM}"/>')
    lines.append(f'    <mesh name="trigger" file="../gripper/trigger.STL" '
                 f'scale="{MM} {MM} {MM}"/>')
    lines.append('  </asset>')
    lines.append('  <worldbody>')

    # base: its own frame is the table plane; the J1 mount sits on it
    b = table["base"]
    base_rot = _align(b.get('up', (0, 0, 1)), (0, 0, 1))
    lines.append(f'    <body name="lead_base" pos="0 0 {-b.get("floor", 0) * MM:g}" '
                 f'quat="{_vec(_quat(base_rot))}">')
    lines.append(f'      <geom name="printed_base" type="mesh" mesh="base" '
                 f'density="{density:g}" rgba="0.93 0.92 0.88 1"/>')
    parent_dist = np.asarray(b["dist"], dtype=float) * MM
    parent_axis = _unit(b["dist_axis"])
    parent_ref = np.array([1.0, 0.0, 0.0])   # carried so clocking is defined
    parent_case_dir = (np.asarray(b["case_dir"], dtype=float)
                       if b.get("case_dir") is not None else None)
    indent = "      "
    for index, part in enumerate(order, start=1):
        spec = table[part]
        prox = np.asarray(spec["prox"], dtype=float)
        dist = np.asarray(spec["dist"], dtype=float)
        # local frame: +z on the proximal joint axis, origin at the interface
        R = _align(spec["prox_axis"], (0, 0, 1))
        mesh_quat = _quat(R)
        mesh_pos = -R @ prox * MM
        child_pos = R @ (dist - prox) * MM
        child_axis = R @ _unit(spec["dist_axis"])
        reverse = spec.get('prox_kind') == 'case'
        stack = (0.0 if spec.get("prox_kind") == "fork" else
                 11.5 if reverse else HORN_FROM_MID)

        # Carry a reference direction to choose reproducible encoder zeros.
        z_c = parent_axis
        x_c = parent_ref - float(parent_ref @ z_c) * z_c
        if np.linalg.norm(x_c) < 1e-6:          # ref parallel to the axis
            fallback = np.eye(3)[int(np.argmin(np.abs(z_c)))]
            x_c = fallback - float(fallback @ z_c) * z_c
        x_c = _unit(x_c)
        frame = np.column_stack([x_c, np.cross(z_c, x_c), z_c])
        seat = parent_dist + stack * MM * parent_axis
        # Case roll is fixed by the parent's mounting screws.
        servo_rot = (_servo_frame((0, 0, -1), R @ _unit(spec['prox_case_dir']))
                     if reverse else _servo_frame(parent_axis, parent_case_dir))
        # The CAD origin is 8 mm ahead of the horn midpoint. Keep the case
        # rigidly attached to the parent; only the printed child turns.
        servo_pos = (np.array([0, 0, SERVO_CASE_BACK_Z * MM]) if reverse else
                     parent_dist - SERVO_MID_Z * MM * parent_axis)
        servo_mesh = 'servo' if spec.get('prox_kind') == 'fork' else 'servo_single'
        servo = (f'{indent}<geom name="servo{index}" type="mesh" '
                     f'mesh="{servo_mesh}" pos="{_vec(servo_pos)}" '
                     f'quat="{_vec(_quat(servo_rot))}" '
                     f'mass="{SERVO_MASS:g}" rgba="0.12 0.12 0.14 1"/>')
        if not reverse:
            lines.append(servo)
        lines.append(f'{indent}<body name="lead_link{index}" '
                     f'pos="{_vec(seat)}" '
                     f'quat="{_vec(_quat(frame))}">')
        if reverse:
            lines.append('  ' + servo)
        lines.append(f'{indent}  <joint name="lead_joint{index}" type="hinge" '
                     'axis="0 0 1" damping="0.01" armature="0.002" '
                     'frictionloss="0.03"/>')
        lines.append(f'{indent}  <geom name="printed_link{index}" type="mesh" '
                     f'mesh="{part}" pos="{_vec(mesh_pos)}" '
                     f'quat="{_vec(mesh_quat)}" density="{density:g}" '
                     f'rgba="0.93 0.92 0.88 1"/>')
        parent_dist = child_pos
        parent_axis = _unit(child_axis)
        parent_ref = R @ _unit(spec["prox_axis"])   # the joint we just left
        parent_case_dir = (R @ _unit(spec["case_dir"])
                           if spec.get("case_dir") is not None else None)
        indent += "  "
    # The final encoder belongs to the trigger, mounted on the handle.
    trigger_servo_rot = _servo_frame(TRIGGER_SERVO_AXIS, TRIGGER_CASE_DIR)
    lines.append(f'{indent}<geom name="servo_trigger" type="mesh" mesh="servo_single" '
                 f'pos="{_vec(R @ (TRIGGER_SERVO_POS - prox) * MM)}" '
                 f'quat="{_vec(_quat(R @ trigger_servo_rot))}" '
                 f'mass="{SERVO_MASS:g}" rgba="0.12 0.12 0.14 1"/>')
    trigger_horn = TRIGGER_SERVO_POS + SERVO_HORN_Z * np.array(TRIGGER_SERVO_AXIS)
    trigger_mesh_rot = _align((0, 1, 0), (0, 0, 1))
    trigger_frame = R @ TRIGGER_ROT @ trigger_mesh_rot.T
    lines.append(f'{indent}<body name="lead_trigger" '
                 f'pos="{_vec(R @ (trigger_horn - prox) * MM)}" '
                 f'quat="{_vec(_quat(trigger_frame))}">')
    lines.append(f'{indent}  <joint name="lead_trigger_joint" axis="0 0 1" '
                 'damping="0.01" armature="0.002"/>')
    lines.append(f'{indent}  <geom name="printed_trigger" type="mesh" mesh="trigger" '
                 f'pos="{_vec(-trigger_mesh_rot @ TRIGGER_PROX * MM)}" '
                 f'quat="{_vec(_quat(trigger_mesh_rot))}" density="{density:g}" '
                 'rgba="0.93 0.92 0.88 1"/>')
    lines.append(f'{indent}</body>')
    lines.append(f'{indent}<site name="lead_ee_site" '
                 f'pos="{_vec(parent_dist)}" size="0.006" rgba="1 0 0 1"/>')
    for _ in order:
        indent = indent[:-2]
        lines.append(f"{indent}</body>")
    lines += ['    </body>', '  </worldbody>', '  <keyframe>',
              f'    <key name="reference" qpos="{_vec(np.deg2rad(REFERENCE_POSES[arm]))}"/>',
              '  </keyframe>', '</mujoco>', '']
    root = ET.fromstring("\n".join(lines))
    servo_proxy.separate_visuals(root)
    ET.indent(root)
    ET.ElementTree(root).write(out_path, encoding="unicode")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=tuple(ARMS), default="franka_fer")
    ap.add_argument("--out", default="/tmp/gello_assembly")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = build(args.arm, os.path.join(args.out, f"{args.arm}.xml"))
    import mujoco
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"{path}: loads, {model.nq} dof, "
          f"total mass {mujoco.mj_getTotalmass(model):.3f} kg")
    for i in range(1, len(ARMS[args.arm][1]) + 1):
        body = model.body(f"lead_link{i}")
        print(f"  lead_link{i}: mass {float(body.mass[0]):.4f} kg  "
              f"pos {np.round(body.pos, 4)}")


if __name__ == "__main__":
    main()
