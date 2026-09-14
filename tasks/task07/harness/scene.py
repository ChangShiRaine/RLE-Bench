"""Workcell construction for task07 (bin clearing).

Builds the factory cell — floor, pedestal-mounted Panda (with hand), the
WELDED KLT-style bin on its stand, the conveyor with the drop zone, cameras,
lights — plus any number of bracket parts at given poses. Agent-visible:
this is the same builder the evaluation composes (on different seeds).
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import parts, spec

GREEN = (0.055, 0.38, 0.16, 1.0)
GREEN_DK = (0.04, 0.30, 0.13, 1.0)


# ---------------------------------------------------------------------------
# Nuisance randomization (drawn per scene from a seeded stream)
# ---------------------------------------------------------------------------
@dataclass
class Nuisance:
    key_light_pos: tuple = (0.4, -0.3, 2.2)
    key_light_dir: tuple = (0.0, 0.0, -1.0)
    key_light_diffuse: float = 0.7
    fill_light_diffuse: float = 0.35
    ambient: float = 0.40
    floor_gray: float = 0.30


def sample_nuisance(rng: np.random.Generator) -> Nuisance:
    az = rng.uniform(0, 2 * np.pi)
    el = rng.uniform(np.deg2rad(45), np.deg2rad(80))
    r = rng.uniform(1.6, 2.4)
    pos = (spec.BIN_POS[0] + r * np.cos(el) * np.cos(az),
           spec.BIN_POS[1] + r * np.cos(el) * np.sin(az),
           r * np.sin(el))
    d = np.array([spec.BIN_POS[0], spec.BIN_POS[1], 0.2]) - np.asarray(pos)
    d /= np.linalg.norm(d)
    return Nuisance(
        key_light_pos=tuple(pos),
        key_light_dir=tuple(d),
        key_light_diffuse=float(rng.uniform(0.55, 0.85)),
        fill_light_diffuse=float(rng.uniform(0.2, 0.45)),
        ambient=float(rng.uniform(0.30, 0.48)),
        floor_gray=float(rng.uniform(0.24, 0.38)),
    )


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------
def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        q[:] = [0.25 / s, (R[2, 1] - R[1, 2]) * s,
                (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def _look_at_quat(pos, target) -> np.ndarray:
    pos = np.asarray(pos, dtype=float)
    target = np.asarray(target, dtype=float)
    f = target - pos
    f /= np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    cup = np.cross(right, f)
    return _mat_to_quat(np.column_stack([right, cup, -f]))


def _find_body(mjspec, name):
    for b in mjspec.bodies:
        if b.name == name:
            return b
    raise KeyError(f"body {name!r} not found")


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------
def build_scene(part_poses=None, nuisance: Nuisance | None = None,
                seed: int | None = None, with_arm: bool = True):
    """Compile the workcell with parts at ``part_poses`` [(pos, quat), ...].

    If ``seed`` is given, nuisance parameters are drawn from
    ``default_rng([seed, STREAM_SCENE])``. Part poses are NOT drawn here —
    piles come from ``episodes.pile_poses`` (or your own layouts).
    Returns ``(model, data)``.
    """
    if seed is not None and nuisance is None:
        nuisance = sample_nuisance(
            np.random.default_rng([seed, spec.STREAM_SCENE]))
    nuisance = nuisance if nuisance is not None else Nuisance()
    part_poses = part_poses if part_poses is not None else []

    s = mujoco.MjSpec()
    s.modelname = "task07_cell"
    s.option.timestep = spec.TIMESTEP
    s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    s.option.impratio = 10.0
    s.option.gravity = [0.0, 0.0, -9.81]
    s.visual.global_.offwidth = max(1280, spec.OVERHEAD_W)
    s.visual.global_.offheight = max(960, spec.OVERHEAD_H)
    s.visual.headlight.ambient = [nuisance.ambient] * 3
    s.visual.headlight.diffuse = [0.30, 0.30, 0.30]
    s.visual.headlight.specular = [0.12, 0.12, 0.12]

    g = nuisance.floor_gray
    s.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                         size=[4, 4, 0.1], rgba=[g, g * 1.02, g * 1.06, 1],
                         friction=list(spec.CONTACT_FRICTION))
    s.worldbody.add_geom(name="pedestal",
                         type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                         size=[spec.PEDESTAL_R, spec.PEDESTAL_H / 2, 0],
                         pos=[0, 0, spec.PEDESTAL_H / 2],
                         rgba=[0.2, 0.2, 0.22, 1])

    _add_bin(s)
    _add_conveyor(s)

    for i, (pos, quat) in enumerate(part_poses):
        parts.add_part(s, i, pos, quat)

    # lights
    lt = s.worldbody.add_light(pos=list(nuisance.key_light_pos),
                               dir=list(nuisance.key_light_dir))
    lt.diffuse = [nuisance.key_light_diffuse] * 3
    lt.castshadow = True
    lt2 = s.worldbody.add_light(pos=[1.5, 1.0, 1.8], dir=[-0.5, -0.5, -1.0])
    lt2.diffuse = [nuisance.fill_light_diffuse] * 3
    lt2.castshadow = False

    # sensor camera (fixed overhead) + harness-only debug views
    cam = s.worldbody.add_camera(name="overhead")
    cam.pos = list(spec.OVERHEAD_POS)
    cam.quat = list(spec.OVERHEAD_QUAT)
    cam.fovy = spec.OVERHEAD_FOVY_DEG
    ocam = s.worldbody.add_camera(name="overview")
    ocam.pos = [1.75, -1.35, 1.15]
    ocam.quat = _look_at_quat([1.75, -1.35, 1.15], [0.35, 0.05, 0.15]).tolist()
    ocam.fovy = 42.0
    scam = s.worldbody.add_camera(name="side")
    spos = [spec.BIN_POS[0] + 0.50, spec.BIN_POS[1] - 0.42, 0.72]
    scam.pos = spos
    scam.quat = _look_at_quat(spos, [spec.BIN_POS[0], spec.BIN_POS[1],
                                     spec.STAND_H + 0.1]).tolist()
    scam.fovy = 45.0

    if with_arm:
        _attach_arm(s)
        _add_magnet_welds(s, len(part_poses))

    model = s.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _add_bin(s: mujoco.MjSpec) -> None:
    """Welded KLT-style bin: stand, bottom, four walls, cosmetic rim."""
    L, W, H = spec.BIN_INNER
    t = spec.BIN_WALL
    bx, by, bz = spec.BIN_POS
    s.worldbody.add_geom(name="bin_stand", type=mujoco.mjtGeom.mjGEOM_BOX,
                         size=[L / 2 + t + 0.01, W / 2 + t + 0.01,
                               spec.STAND_H / 2],
                         pos=[bx, by, spec.STAND_H / 2],
                         rgba=[0.25, 0.26, 0.28, 1])
    s.worldbody.add_geom(name="bin_bottom", type=mujoco.mjtGeom.mjGEOM_BOX,
                         size=[L / 2 + t, W / 2 + t, t / 2],
                         pos=[bx, by, bz + t / 2], rgba=list(GREEN_DK),
                         friction=list(spec.CONTACT_FRICTION))
    walls = [
        ("bin_wall_ny", [L / 2 + t, t / 2, H / 2], [bx, by - (W / 2 + t / 2)]),
        ("bin_wall_py", [L / 2 + t, t / 2, H / 2], [bx, by + (W / 2 + t / 2)]),
        ("bin_wall_nx", [t / 2, W / 2, H / 2], [bx - (L / 2 + t / 2), by]),
        ("bin_wall_px", [t / 2, W / 2, H / 2], [bx + (L / 2 + t / 2), by]),
    ]
    for name, size, (x, y) in walls:
        s.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                             size=size, pos=[x, y, bz + t + H / 2],
                             rgba=list(GREEN),
                             friction=list(spec.CONTACT_FRICTION))
    rim_z = bz + t + H - 0.004
    for sy in (-1, 1):
        s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                             size=[L / 2 + t + 0.006, 0.010, 0.008],
                             pos=[bx, by + sy * (W / 2 + t / 2), rim_z],
                             rgba=list(GREEN_DK), contype=0, conaffinity=0)
    for sx in (-1, 1):
        s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                             size=[0.010, W / 2 + t + 0.006, 0.008],
                             pos=[bx + sx * (L / 2 + t / 2), by, rim_z],
                             rgba=list(GREEN_DK), contype=0, conaffinity=0)


def _add_conveyor(s: mujoco.MjSpec) -> None:
    cx, cy = spec.BELT_CENTER
    hx, hy = spec.BELT_HALF
    top = spec.BELT_TOP
    s.worldbody.add_geom(name="belt", type=mujoco.mjtGeom.mjGEOM_BOX,
                         size=[hx, hy, 0.012], pos=[cx, cy, top - 0.012],
                         rgba=[0.13, 0.13, 0.14, 1],
                         friction=[0.9, 0.01, 0.001])
    for sy in (-1, 1):
        s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                             size=[hx, 0.012, 0.030],
                             pos=[cx, cy + sy * (hy + 0.012), top - 0.018],
                             rgba=[0.55, 0.56, 0.58, 1])
    for sx in (-1, 1):
        s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                             size=[0.014, hy, 0],
                             pos=[cx + sx * hx, cy, top - 0.026],
                             quat=[0.7071068, 0.7071068, 0, 0],
                             rgba=[0.55, 0.56, 0.58, 1])
        for sy in (-1, 1):
            s.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.016, 0.016, top / 2 - 0.02],
                pos=[cx + sx * (hx - 0.10), cy + sy * 0.12, top / 2 - 0.03],
                rgba=[0.55, 0.56, 0.58, 1])
    # drop-zone marking (visual only)
    zlo, zhi = np.asarray(spec.ZONE_LO), np.asarray(spec.ZONE_HI)
    zc = (zlo + zhi) / 2
    s.worldbody.add_geom(name="zone_marking", type=mujoco.mjtGeom.mjGEOM_BOX,
                         size=[(zhi[0] - zlo[0]) / 2, (zhi[1] - zlo[1]) / 2,
                               0.0005],
                         pos=[zc[0], zc[1], top + 0.0006],
                         rgba=[0.85, 0.65, 0.05, 0.35],
                         contype=0, conaffinity=0)


def _attach_arm(s: mujoco.MjSpec) -> None:
    """Panda (no hand) with the contact-electromagnet head and wrist cam."""
    panda = mujoco.MjSpec.from_file(str(spec.PANDA_XML))
    for key in list(panda.keys):
        panda.delete(key)
    mount = _find_body(panda, "attachment")
    head = mount.add_body(name="magnet")
    g = head.add_geom(name="magnet_collar",
                      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      size=[spec.MAG_HEAD_R + 0.006, spec.MAG_COLLAR_L / 2, 0],
                      pos=[0, 0, spec.MAG_COLLAR_L / 2],
                      rgba=[0.16, 0.16, 0.18, 1])
    g.mass = 0.15
    g = head.add_geom(name="magnet_shaft",
                      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      size=[spec.MAG_SHAFT_R, spec.MAG_SHAFT_L / 2, 0],
                      pos=[0, 0, spec.MAG_COLLAR_L + spec.MAG_SHAFT_L / 2],
                      rgba=[0.35, 0.36, 0.38, 1])
    g.mass = 0.12
    g = head.add_geom(name="magnet_head",
                      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      size=[spec.MAG_HEAD_R, spec.MAG_HEAD_L / 2, 0],
                      pos=[0, 0, spec.MAG_COLLAR_L + spec.MAG_SHAFT_L
                           + spec.MAG_HEAD_L / 2],
                      rgba=[0.55, 0.33, 0.16, 1],
                      friction=list(spec.CONTACT_FRICTION))
    g.mass = 0.40
    head.add_site(name="grip", pos=[0.0, 0.0, spec.MAG_HEAD_H],
                  quat=[1, 0, 0, 0])
    head.add_site(name="ft", pos=[0.0, 0.0, 0.0], quat=[1, 0, 0, 0])
    wcam = mount.add_camera(name="wrist")
    wcam.pos = list(spec.WRIST_MOUNT_POS)
    wcam.quat = list(spec.WRIST_MOUNT_QUAT)
    wcam.fovy = spec.WRIST_FOVY_DEG
    frame = s.worldbody.add_frame()
    frame.pos = [0.0, 0.0, spec.PEDESTAL_H]
    s.attach(panda, prefix="arm_", frame=frame)
    # wrist force-torque sensor: the wrench between the magnet assembly and
    # the flange, expressed in the tool-mount site frame
    for name, stype in (("ft_force", mujoco.mjtSensor.mjSENS_FORCE),
                        ("ft_torque", mujoco.mjtSensor.mjSENS_TORQUE)):
        sens = s.add_sensor()
        sens.name = name
        sens.type = stype
        sens.objtype = mujoco.mjtObj.mjOBJ_SITE
        sens.objname = "arm_ft"


def ft_reading(model, data) -> np.ndarray:
    """Clean 6-vector (force N, torque Nm) from the wrist F/T sensor."""
    out = np.empty(6)
    for k, name in enumerate(("ft_force", "ft_torque")):
        s = model.sensor(name)
        out[3 * k:3 * k + 3] = data.sensordata[s.adr[0]:s.adr[0] + 3]
    return out


def _add_magnet_welds(s: mujoco.MjSpec, n: int) -> None:
    """One inactive weld per part; the runtime energizes them on contact."""
    for i in range(n):
        e = s.add_equality()
        e.type = mujoco.mjtEq.mjEQ_WELD
        e.name = f"mag_weld_{i}"
        e.objtype = mujoco.mjtObj.mjOBJ_BODY
        e.name1 = "arm_magnet"
        e.name2 = parts.part_name(i)
        e.active = False


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
def set_arm(model, data, qpos7, ctrl: bool = True) -> None:
    for i, qv in enumerate(qpos7, start=1):
        data.joint(f"arm_joint{i}").qpos[0] = qv
        data.joint(f"arm_joint{i}").qvel[0] = 0.0
        if ctrl:
            data.actuator(f"arm_actuator{i}").ctrl[0] = qv
    mujoco.mj_forward(model, data)


def get_arm(data) -> list[float]:
    return [float(data.joint(f"arm_joint{i}").qpos[0]) for i in range(1, 8)]


def apply_ctrl(model, data, ctrl) -> np.ndarray:
    """Clamp an N_CTRL command and write it. Slot 7 is the magnet command —
    not a MuJoCo actuator; the runtime's MagnetEngine consumes it."""
    ctrl = np.asarray(ctrl, dtype=float).reshape(-1)
    out = np.empty(spec.N_CTRL)
    for i in range(7):
        a = model.actuator(f"arm_actuator{i + 1}")
        out[i] = float(np.clip(ctrl[i], a.ctrlrange[0], a.ctrlrange[1]))
        data.actuator(f"arm_actuator{i + 1}").ctrl[0] = out[i]
    out[7] = float(np.clip(ctrl[7], *spec.MAG_CTRL_RANGE))
    return out


def arm_tau(data) -> np.ndarray:
    return np.array([data.actuator(f"arm_actuator{i}").force[0]
                     for i in range(1, 8)])


def n_parts(model) -> int:
    count = 0
    while mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                            parts.part_name(count)) >= 0:
        count += 1
    return count


def part_pose(data, i: int) -> tuple[np.ndarray, np.ndarray]:
    q = data.joint(parts.part_name(i)).qpos
    return np.array(q[:3]), np.array(q[3:7])


def set_part_pose(data, i: int, pos, quat, zero_vel: bool = True) -> None:
    j = data.joint(parts.part_name(i))
    j.qpos[:3] = pos
    j.qpos[3:7] = quat
    if zero_vel:
        j.qvel[:] = 0.0


def part_positions(data, n: int) -> np.ndarray:
    if n == 0:
        return np.zeros((0, 3))
    return np.stack([data.joint(parts.part_name(i)).qpos[:3].copy()
                     for i in range(n)])


def part_speeds(data, n: int) -> np.ndarray:
    return np.array([float(np.linalg.norm(
        data.joint(parts.part_name(i)).qvel[:3])) for i in range(n)])


def part_velocity(model, data, i: int) -> tuple[np.ndarray, np.ndarray]:
    """World-frame COM linear and angular velocity for one part."""
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, parts.part_name(i))
    spatial = np.zeros(6)
    # MuJoCo spatial velocity ordering is angular, then linear.
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, spatial, 0)
    return spatial[3:].copy(), spatial[:3].copy()


def part_contact_force_details(
        model, data, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-part maximum contact force and its MuJoCo contact index."""
    body_of_part = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                      parts.part_name(i)): i
                    for i in range(n)}
    peak = np.zeros(n)
    peak_contact = np.full(n, -1, dtype=int)
    f6 = np.zeros(6)
    for c in range(data.ncon):
        con = data.contact[c]
        for gid in (con.geom1, con.geom2):
            pi = body_of_part.get(int(model.geom_bodyid[gid]))
            if pi is not None:
                mujoco.mj_contactForce(model, data, c, f6)
                mag = float(np.linalg.norm(f6[:3]))
                if mag > peak[pi]:
                    peak[pi] = mag
                    peak_contact[pi] = c
    return peak, peak_contact


def part_contact_forces(model, data, n: int) -> np.ndarray:
    """Max contact-force magnitude per part at the current step."""
    return part_contact_force_details(model, data, n)[0]


def _contact_category(other_geom: str, other_body: str) -> str:
    if other_body.startswith("part"):
        return "part_part"
    if other_geom == "floor":
        return "part_floor"
    if other_geom in ("belt", "zone_marking"):
        return "part_belt"
    if other_geom.startswith("bin_"):
        return "part_bin"
    if other_geom.startswith("arm_magnet_"):
        return "part_tool"
    if other_geom == "pedestal":
        return "part_pedestal"
    return "part_other"


def part_contact_detail(model, data, i: int, contact_index: int) -> dict | None:
    """JSON-safe description of one contact involving ``part{i}``."""
    if contact_index < 0 or contact_index >= data.ncon:
        return None
    con = data.contact[contact_index]
    g1, g2 = int(con.geom1), int(con.geom2)
    part_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, parts.part_name(i))
    if int(model.geom_bodyid[g1]) == part_body:
        part_gid, other_gid = g1, g2
    elif int(model.geom_bodyid[g2]) == part_body:
        part_gid, other_gid = g2, g1
    else:
        return None

    def name(objtype, objid, fallback):
        return mujoco.mj_id2name(model, objtype, objid) or fallback

    part_geom = name(mujoco.mjtObj.mjOBJ_GEOM, part_gid,
                     f"geom#{part_gid}")
    other_geom = name(mujoco.mjtObj.mjOBJ_GEOM, other_gid,
                      f"geom#{other_gid}")
    other_body_id = int(model.geom_bodyid[other_gid])
    other_body = name(mujoco.mjtObj.mjOBJ_BODY, other_body_id,
                      "world" if other_body_id == 0 else f"body#{other_body_id}")
    f6 = np.zeros(6)
    mujoco.mj_contactForce(model, data, contact_index, f6)
    return {
        "part_geom": part_geom,
        "other_geom": other_geom,
        "other_body": other_body,
        "category": _contact_category(other_geom, other_body),
        "pos_world_m": np.asarray(con.pos, dtype=float).tolist(),
        "normal_world": np.asarray(con.frame[:3], dtype=float).tolist(),
        "force_n": float(np.linalg.norm(f6[:3])),
    }


def part_contacts(model, data, i: int, limit: int = 4) -> list[dict]:
    """Strongest current contacts involving one part, descending by force."""
    found = []
    for c in range(data.ncon):
        detail = part_contact_detail(model, data, i, c)
        if detail is not None:
            found.append(detail)
    found.sort(key=lambda x: x["force_n"], reverse=True)
    return found[:limit]


def settle(model, data, duration: float) -> None:
    for _ in range(int(round(duration / model.opt.timestep))):
        mujoco.mj_step(model, data)


def save_cell_mjb(path: str) -> None:
    """Compiled cell (arm + statics, ZERO parts) for agent-side planning."""
    model, _ = build_scene(part_poses=[], nuisance=Nuisance(), with_arm=True)
    mujoco.mj_saveModel(model, path, None)


def tool_bin_force(model, data, _cache={}) -> float:
    """Max contact-force magnitude between the end-effector (magnet
    collar/shaft/head) and the bin (walls, bottom, stand) this step."""
    key = id(model)
    if key not in _cache:
        tool = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                for g in ("arm_magnet_collar", "arm_magnet_shaft",
                          "arm_magnet_head")}
        bin_ = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                for g in ("bin_bottom", "bin_wall_ny", "bin_wall_py",
                          "bin_wall_nx", "bin_wall_px", "bin_stand")}
        _cache[key] = (tool - {-1}, bin_ - {-1})
    tool, bin_ = _cache[key]
    peak = 0.0
    f6 = np.zeros(6)
    for c in range(data.ncon):
        con = data.contact[c]
        g1, g2 = int(con.geom1), int(con.geom2)
        if (g1 in tool and g2 in bin_) or (g2 in tool and g1 in bin_):
            mujoco.mj_contactForce(model, data, c, f6)
            peak = max(peak, float(np.linalg.norm(f6[:3])))
    return peak
