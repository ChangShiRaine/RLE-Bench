"""Cell construction for task11 (online bin packing).

Floor, pedestal-mounted Panda with a suction tool, a welded tote on its
stand, the conveyor and the fixed cameras, plus one parked body per queued
box. Agent-visible: the evaluation composes the same builder.

The belt surface is an invisible slab on an x slide joint that the runtime
drives at the episode's belt speed and wraps periodically; boxes resting on
it are carried by friction.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import spec

BELT_WRAP = 0.10                  # slab travel before it is wrapped back
# collision bits: statics/arm 1, slab 2 (boxes and the tool touch both)
BOX_CONTYPE, BOX_CONAFFINITY = 3, 1
TOOL_CONTYPE, TOOL_CONAFFINITY = 3, 1
TOOL_GEOMS = ("arm_tool_collar", "arm_tool_shaft", "arm_tool_cup")
BIN_GEOMS = ("bin_bottom", "bin_wall_nx", "bin_wall_px", "bin_wall_ny",
             "bin_wall_py", "bin_stand")


@dataclass
class Nuisance:
    key_light_pos: tuple = (0.4, 0.0, 2.2)
    key_light_dir: tuple = (0.0, 0.0, -1.0)
    key_light_diffuse: float = 0.7
    fill_light_diffuse: float = 0.35
    ambient: float = 0.40
    floor_gray: float = 0.30


def sample_nuisance(rng: np.random.Generator) -> Nuisance:
    az = rng.uniform(0, 2 * np.pi)
    el = rng.uniform(np.deg2rad(45), np.deg2rad(80))
    r = rng.uniform(1.6, 2.4)
    pos = np.array([0.4 + r * np.cos(el) * np.cos(az),
                    r * np.cos(el) * np.sin(az), r * np.sin(el)])
    d = np.array([0.4, 0.0, 0.2]) - pos
    return Nuisance(tuple(pos), tuple(d / np.linalg.norm(d)),
                    float(rng.uniform(0.55, 0.85)),
                    float(rng.uniform(0.2, 0.45)),
                    float(rng.uniform(0.30, 0.48)),
                    float(rng.uniform(0.24, 0.38)))


def box_name(i: int) -> str:
    return f"box{i}"


def _look_at_quat(pos, target) -> list:
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    R = np.column_stack([right, np.cross(right, f), -f])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q.tolist()


def build_scene(boxes=(), nuisance: Nuisance | None = None,
                seed: int | None = None, with_arm: bool = True):
    """Compile the cell with one parked body per entry of ``boxes``
    (``boxes.Box`` records). Returns ``(model, data)``."""
    if seed is not None and nuisance is None:
        nuisance = sample_nuisance(
            np.random.default_rng([seed, spec.STREAM_SCENE]))
    nuisance = nuisance or Nuisance()

    s = mujoco.MjSpec()
    s.modelname = "task11_cell"
    s.option.timestep = spec.TIMESTEP
    s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    s.option.impratio = 10.0
    s.visual.global_.offwidth = 1280
    s.visual.global_.offheight = 960
    s.visual.headlight.ambient = [nuisance.ambient] * 3
    s.visual.headlight.diffuse = [0.3] * 3
    s.visual.headlight.specular = [0.1] * 3
    # no shadows or multisampling: ~10x cheaper under software GL (osmesa)
    s.visual.quality.offsamples = 0
    s.visual.quality.shadowsize = 1024

    g = nuisance.floor_gray
    s.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                         size=[4, 4, 0.1], rgba=[g, g * 1.02, g * 1.06, 1])
    s.worldbody.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                         size=[spec.PEDESTAL_R, spec.PEDESTAL_H / 2, 0],
                         pos=[0, 0, spec.PEDESTAL_H / 2],
                         rgba=[0.2, 0.2, 0.22, 1])
    _add_bin(s)
    _add_conveyor(s)
    for b in boxes:
        _add_box(s, b)

    lt = s.worldbody.add_light(pos=list(nuisance.key_light_pos),
                               dir=list(nuisance.key_light_dir))
    lt.diffuse = [nuisance.key_light_diffuse] * 3
    lt.castshadow = False
    lt2 = s.worldbody.add_light(pos=[1.5, 1.0, 1.8], dir=[-0.5, -0.5, -1.0])
    lt2.diffuse = [nuisance.fill_light_diffuse] * 3
    lt2.castshadow = False

    cam = s.worldbody.add_camera(name="top_cam")
    cam.pos = list(spec.TOP_CAM_POS)
    cam.quat = list(spec.TOP_CAM_QUAT)
    cam.fovy = spec.TOP_FOVY_DEG
    ov = s.worldbody.add_camera(name="overview")
    ov.pos = [1.55, -1.30, 1.25]
    ov.quat = _look_at_quat(ov.pos, [0.40, 0.05, 0.15])
    ov.fovy = 45.0

    if with_arm:
        _attach_arm(s)
        _add_suction_welds(s, len(boxes))
    model = s.compile()
    data = mujoco.MjData(model)
    park_all(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def _add_bin(s: mujoco.MjSpec) -> None:
    L, W, H = spec.BIN_INNER
    t = spec.BIN_WALL
    bx, by = spec.BIN_CENTER
    blue, blue_dk = [0.12, 0.30, 0.55, 1], [0.09, 0.22, 0.42, 1]
    s.worldbody.add_geom(name="bin_stand", type=mujoco.mjtGeom.mjGEOM_BOX,
                         size=[L / 2 + t + 0.01, W / 2 + t + 0.01,
                               spec.STAND_H / 2],
                         pos=[bx, by, spec.STAND_H / 2],
                         rgba=[0.25, 0.26, 0.28, 1])
    s.worldbody.add_geom(name="bin_bottom", type=mujoco.mjtGeom.mjGEOM_BOX,
                         size=[L / 2 + t, W / 2 + t, t / 2],
                         pos=[bx, by, spec.STAND_H + t / 2], rgba=blue_dk,
                         friction=list(spec.BOX_FRICTION))
    z = spec.BIN_FLOOR_Z + H / 2
    for name, size, x, y in (
            ("bin_wall_ny", [L / 2 + t, t / 2, H / 2], bx, by - W / 2 - t / 2),
            ("bin_wall_py", [L / 2 + t, t / 2, H / 2], bx, by + W / 2 + t / 2),
            ("bin_wall_nx", [t / 2, W / 2, H / 2], bx - L / 2 - t / 2, by),
            ("bin_wall_px", [t / 2, W / 2, H / 2], bx + L / 2 + t / 2, by)):
        s.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                             size=size, pos=[x, y, z], rgba=blue,
                             friction=list(spec.BOX_FRICTION))


def _add_conveyor(s: mujoco.MjSpec) -> None:
    x0, x1 = spec.BELT_X_RANGE
    cx, hx = (x0 + x1) / 2, (x1 - x0) / 2
    top, y, hy = spec.BELT_TOP, spec.BELT_Y, spec.BELT_HALF_W
    grey = [0.55, 0.56, 0.58, 1]
    # visible, non-colliding belt surface and frame
    s.worldbody.add_geom(name="belt_visual", type=mujoco.mjtGeom.mjGEOM_BOX,
                         size=[hx, hy, 0.01], pos=[cx, y, top - 0.0102],
                         rgba=[0.13, 0.13, 0.14, 1], contype=0, conaffinity=0)
    for sy in (-1, 1):
        s.worldbody.add_geom(name=f"belt_rail_{'np'[sy > 0]}",
                             type=mujoco.mjtGeom.mjGEOM_BOX,
                             size=[hx, 0.012, 0.03],
                             pos=[cx, y + sy * (hy + 0.012), top - 0.02],
                             rgba=grey)
        for sx in (-1, 1):
            s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                                 size=[0.02, 0.02, (top - 0.05) / 2],
                                 pos=[cx + sx * (hx - 0.1), y + sy * (hy - 0.03),
                                      (top - 0.05) / 2],
                                 rgba=grey, contype=0, conaffinity=0)
    s.worldbody.add_geom(name="overflow_line", type=mujoco.mjtGeom.mjGEOM_BOX,
                         size=[0.004, hy, 0.0005],
                         pos=[spec.BELT_END_X, y, top + 0.0006],
                         rgba=[0.85, 0.15, 0.1, 0.8], contype=0, conaffinity=0)
    # the moving slab: invisible, collides with boxes and the tool only
    slab = s.worldbody.add_body(name="belt_slab", pos=[cx, y, top - 0.01])
    slab.add_joint(name="belt_slide", type=mujoco.mjtJoint.mjJNT_SLIDE,
                   axis=[1, 0, 0])
    gs = slab.add_geom(name="belt", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[hx + BELT_WRAP, hy, 0.01], rgba=[0, 0, 0, 0],
                       contype=0, conaffinity=2,
                       friction=list(spec.BELT_FRICTION))
    gs.mass = 500.0


def _add_box(s: mujoco.MjSpec, box) -> None:
    b = s.worldbody.add_body(name=box_name(box.index), pos=list(spec.PARK),
                             gravcomp=1.0)   # compiled parked (ngravcomp > 0)
    b.add_freejoint(name=box_name(box.index))
    g = b.add_geom(name=box_name(box.index), type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[d / 2 for d in box.dims], rgba=list(box.rgba),
                   friction=list(spec.BOX_FRICTION),
                   contype=0, conaffinity=0)
    g.mass = box.mass


def _attach_arm(s: mujoco.MjSpec) -> None:
    panda = mujoco.MjSpec.from_file(str(spec.PANDA_XML))
    for key in list(panda.keys):
        panda.delete(key)
    mount = next(b for b in panda.bodies if b.name == "attachment")
    tool = mount.add_body(name="tool")
    z = 0.0
    for name, r, length, mass, rgba in (
            ("tool_collar", spec.COLLAR_R, spec.COLLAR_L, 0.15,
             [0.16, 0.16, 0.18, 1]),
            ("tool_shaft", spec.SHAFT_R, spec.SHAFT_L, 0.12,
             [0.35, 0.36, 0.38, 1]),
            ("tool_cup", spec.CUP_R, spec.CUP_L, 0.08,
             [0.10, 0.45, 0.20, 1])):
        g = tool.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                          size=[r, length / 2, 0], pos=[0, 0, z + length / 2],
                          rgba=rgba, contype=TOOL_CONTYPE,
                          conaffinity=TOOL_CONAFFINITY,
                          friction=list(spec.BOX_FRICTION))
        g.mass = mass
        z += length
    tool.add_site(name="tcp", pos=[0, 0, spec.TOOL_L])
    tool.add_site(name="ft", pos=[0, 0, 0])
    wcam = tool.add_camera(name="wrist_cam")
    wcam.pos = list(spec.WRIST_MOUNT_POS)
    wcam.quat = list(spec.WRIST_MOUNT_QUAT)
    wcam.fovy = spec.WRIST_FOVY_DEG
    frame = s.worldbody.add_frame()
    frame.pos = [0.0, 0.0, spec.PEDESTAL_H]
    s.attach(panda, prefix="arm_", frame=frame)
    for name, stype in (("ft_force", mujoco.mjtSensor.mjSENS_FORCE),
                        ("ft_torque", mujoco.mjtSensor.mjSENS_TORQUE)):
        sens = s.add_sensor()
        sens.name = name
        sens.type = stype
        sens.objtype = mujoco.mjtObj.mjOBJ_SITE
        sens.objname = "arm_ft"


def _add_suction_welds(s: mujoco.MjSpec, n: int) -> None:
    for i in range(n):
        e = s.add_equality()
        e.type = mujoco.mjtEq.mjEQ_WELD
        e.name = f"suction_weld_{i}"
        e.objtype = mujoco.mjtObj.mjOBJ_BODY
        e.name1 = "arm_tool"
        e.name2 = box_name(i)
        e.active = False


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
def n_boxes(model) -> int:
    n = 0
    while mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, box_name(n)) >= 0:
        n += 1
    return n


def box_geom(model, i: int) -> int:
    return model.geom(box_name(i)).id


def set_box_active(model, i: int, active: bool) -> None:
    """Parked boxes neither collide nor fall (gravity compensated)."""
    g = box_geom(model, i)
    b = model.geom_bodyid[g]
    # body_* masks are the compiled broad-phase filter: keep them in sync
    model.geom_contype[g] = model.body_contype[b] = BOX_CONTYPE if active else 0
    model.geom_conaffinity[g] = model.body_conaffinity[b] = \
        BOX_CONAFFINITY if active else 0
    model.body_gravcomp[b] = 0.0 if active else 1.0


def box_pose(data, i: int) -> tuple[np.ndarray, np.ndarray]:
    q = data.joint(box_name(i)).qpos
    return np.array(q[:3]), np.array(q[3:7])


def set_box_pose(data, i: int, pos, quat, vel=(0.0, 0.0, 0.0)) -> None:
    j = data.joint(box_name(i))
    j.qpos[:3] = pos
    j.qpos[3:7] = quat
    j.qvel[:] = 0.0
    j.qvel[:3] = vel


def park(model, data, i: int) -> None:
    set_box_active(model, i, False)
    set_box_pose(data, i, [spec.PARK[0] + i * spec.PARK_DX, spec.PARK[1],
                           spec.PARK[2]], [1, 0, 0, 0])


def park_all(model, data) -> None:
    for i in range(n_boxes(model)):
        park(model, data, i)


def box_rot(data, i: int) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, data.joint(box_name(i)).qpos[3:7])
    return m.reshape(3, 3)


def box_corners(model, data, i: int) -> np.ndarray:
    """(8, 3) world corners of box i."""
    half = model.geom_size[box_geom(model, i)]
    pos, _ = box_pose(data, i)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                      for sz in (-1, 1)])
    return pos + (signs * half) @ box_rot(data, i).T


def box_tilt_deg(data, i: int) -> float:
    """Angle between the box axis closest to vertical and world z."""
    R = box_rot(data, i)
    return float(np.degrees(np.arccos(np.clip(np.abs(R[2]).max(), 0, 1))))


def box_speed(data, i: int) -> float:
    return float(np.linalg.norm(data.joint(box_name(i)).qvel[:3]))


def box_velocity(data, i: int) -> np.ndarray:
    return np.array(data.joint(box_name(i)).qvel[:3])


def set_arm(model, data, qpos7, ctrl: bool = True) -> None:
    for i, qv in enumerate(qpos7, start=1):
        data.joint(f"arm_joint{i}").qpos[0] = qv
        data.joint(f"arm_joint{i}").qvel[0] = 0.0
        if ctrl:
            data.actuator(f"arm_actuator{i}").ctrl[0] = qv
    mujoco.mj_forward(model, data)


def get_arm(data) -> np.ndarray:
    return np.array([data.joint(f"arm_joint{i}").qpos[0] for i in range(1, 8)])


def get_arm_vel(data) -> np.ndarray:
    return np.array([data.joint(f"arm_joint{i}").qvel[0] for i in range(1, 8)])


def arm_tau(data) -> np.ndarray:
    return np.array([data.actuator(f"arm_actuator{i}").force[0]
                     for i in range(1, 8)])


def apply_ctrl(model, data, ctrl) -> np.ndarray:
    """Clamp and write an N_CTRL command. Slot 7 is the suction command,
    consumed by the runtime's SuctionEngine."""
    ctrl = np.asarray(ctrl, dtype=float).reshape(-1)
    out = np.empty(spec.N_CTRL)
    for i in range(7):
        a = model.actuator(f"arm_actuator{i + 1}")
        out[i] = float(np.clip(ctrl[i], *a.ctrlrange))
        data.actuator(f"arm_actuator{i + 1}").ctrl[0] = out[i]
    out[7] = float(np.clip(ctrl[7], *spec.SUCTION_CTRL_RANGE))
    return out


def ft_reading(model, data) -> np.ndarray:
    out = np.empty(6)
    for k, name in enumerate(("ft_force", "ft_torque")):
        a = model.sensor(name).adr[0]
        out[3 * k:3 * k + 3] = data.sensordata[a:a + 3]
    return out


def drive_belt(data, belt_v: float) -> None:
    """Hold the slab at belt speed along -x and wrap it; call every substep."""
    j = data.joint("belt_slide")
    if j.qpos[0] < -BELT_WRAP:
        j.qpos[0] += BELT_WRAP
    j.qvel[0] = -belt_v


def settle(model, data, duration: float, belt_v: float = 0.0) -> None:
    for _ in range(int(round(duration / model.opt.timestep))):
        drive_belt(data, belt_v)
        mujoco.mj_step(model, data)


def contact_force(model, data, c: int) -> float:
    f6 = np.zeros(6)
    mujoco.mj_contactForce(model, data, c, f6)
    return float(np.linalg.norm(f6[:3]))


def box_contact_forces(model, data, n: int) -> np.ndarray:
    """Max contact-force magnitude per box at the current step."""
    body_of = {model.body(box_name(i)).id: i for i in range(n)}
    peak = np.zeros(n)
    for c in range(data.ncon):
        con = data.contact[c]
        for gid in (con.geom1, con.geom2):
            i = body_of.get(int(model.geom_bodyid[gid]))
            if i is not None:
                peak[i] = max(peak[i], contact_force(model, data, c))
    return peak


def box_in_contact(model, data, n: int) -> np.ndarray:
    body_of = {model.body(box_name(i)).id: i for i in range(n)}
    out = np.zeros(n, dtype=bool)
    for c in range(data.ncon):
        con = data.contact[c]
        for gid in (con.geom1, con.geom2):
            i = body_of.get(int(model.geom_bodyid[gid]))
            if i is not None:
                out[i] = True
    return out


def tool_bin_force(model, data, _cache={}) -> float:
    """Max contact force between the tool and the tote this step."""
    key = id(model)
    if key not in _cache:
        ids = lambda names: {model.geom(n).id for n in names}  # noqa: E731
        _cache[key] = (ids(TOOL_GEOMS), ids(BIN_GEOMS))
    tool, tote = _cache[key]
    peak = 0.0
    for c in range(data.ncon):
        g1, g2 = int(data.contact[c].geom1), int(data.contact[c].geom2)
        if (g1 in tool and g2 in tote) or (g2 in tool and g1 in tote):
            peak = max(peak, contact_force(model, data, c))
    return peak


def save_cell_mjb(path: str) -> None:
    """Compiled cell (arm + statics + belt, zero boxes) for agent planning."""
    model, _ = build_scene(boxes=(), nuisance=Nuisance(), with_arm=True)
    mujoco.mj_saveModel(model, path, None)
