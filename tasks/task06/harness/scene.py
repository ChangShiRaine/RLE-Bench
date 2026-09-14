"""Scene construction for the task06 family.

Builds the tabletop world — table, block (visual mesh + collision boxes),
Franka Panda with the stick_d405 pusher, sensor + overview cameras, lights —
with seeded nuisance randomization.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import spec


# ---------------------------------------------------------------------------
# Nuisance randomization (drawn per scene from a seeded stream)
# ---------------------------------------------------------------------------
@dataclass
class Nuisance:
    key_light_pos: tuple = (0.8, 0.9, 1.6)
    key_light_dir: tuple = (-0.45, -0.5, -0.85)
    key_light_diffuse: float = 0.75
    fill_light_diffuse: float = 0.22
    ambient: float = 0.28
    table_gray: float = 0.815
    table_tint: tuple = (0.005, -0.005, -0.02)


def sample_nuisance(rng: np.random.Generator) -> Nuisance:
    az = rng.uniform(0, 2 * np.pi)
    el = rng.uniform(np.deg2rad(35), np.deg2rad(70))
    r = rng.uniform(1.4, 2.2)
    pos = (r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az),
           r * np.sin(el))
    d = -np.asarray(pos)
    d = d / np.linalg.norm(d)
    return Nuisance(
        key_light_pos=tuple(pos),
        key_light_dir=tuple(d),
        key_light_diffuse=float(rng.uniform(0.55, 0.9)),
        fill_light_diffuse=float(rng.uniform(0.12, 0.32)),
        ambient=float(rng.uniform(0.18, 0.38)),
        table_gray=float(rng.uniform(0.55, 0.9)),
        table_tint=tuple(rng.uniform(-0.05, 0.05, size=3)),
    )


def sample_block_pose(rng: np.random.Generator) -> tuple[float, float, float]:
    x = float(rng.uniform(-spec.SPAWN_HALF[0], spec.SPAWN_HALF[0]))
    y = float(rng.uniform(-spec.SPAWN_HALF[1], spec.SPAWN_HALF[1]))
    theta = float(rng.uniform(-np.pi, np.pi))
    return x, y, theta


# ---------------------------------------------------------------------------
# Quaternion / camera helpers
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
def build_scene(block_pose=None, nuisance: Nuisance | None = None,
                seed: int | None = None, with_arm: bool = True,
                shape_name: str = spec.DEFAULT_BLOCK_SHAPE):
    """Compile the tabletop scene.

    If ``seed`` is given, the block spawn pose and nuisance parameters are
    drawn from ``default_rng([seed, STREAM_SCENE])``; explicit ``block_pose``
    / ``nuisance`` arguments override the draw. Returns ``(model, data)``.
    """
    if seed is not None:
        rng = np.random.default_rng([seed, spec.STREAM_SCENE])
        drawn_pose = sample_block_pose(rng)
        drawn_nuis = sample_nuisance(rng)
        block_pose = block_pose if block_pose is not None else drawn_pose
        nuisance = nuisance if nuisance is not None else drawn_nuis
    block_pose = block_pose if block_pose is not None else (0.0, 0.0, 0.0)
    nuisance = nuisance if nuisance is not None else Nuisance()

    s = mujoco.MjSpec()
    s.modelname = "task06_scene"
    s.option.timestep = 0.002
    s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    s.option.impratio = 10.0
    s.option.gravity = [0.0, 0.0, -9.81]
    s.visual.global_.offwidth = max(1280, spec.IMG_W)
    s.visual.global_.offheight = max(960, spec.IMG_H)
    s.visual.headlight.ambient = [nuisance.ambient] * 3
    s.visual.headlight.diffuse = [0.18, 0.18, 0.18]
    s.visual.headlight.specular = [0.0, 0.0, 0.0]

    if shape_name not in spec.BLOCK_SHAPES:
        raise ValueError(f"unknown block shape {shape_name!r}")

    mesh = s.add_mesh()
    mesh.name = "block_mesh"
    mesh.file = str(spec.BLOCK_MESHES[shape_name])
    mesh.scale = [spec.MESH_SCALE] * 3

    # floor + table
    s.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                         size=[0, 0, 0.05], pos=[0, 0, -0.75],
                         rgba=[0.35, 0.37, 0.40, 1])
    tg = np.clip(nuisance.table_gray + np.asarray(nuisance.table_tint), 0, 1)
    s.worldbody.add_geom(name="tabletop", type=mujoco.mjtGeom.mjGEOM_BOX,
                         size=[spec.TABLE_HALF[0], spec.TABLE_HALF[1],
                               spec.TABLE_THICKNESS / 2],
                         pos=[0, 0, -spec.TABLE_THICKNESS / 2],
                         rgba=[*tg.tolist(), 1.0],
                         friction=list(spec.CONTACT_FRICTION))
    for sx in (-1, 1):
        for sy in (-1, 1):
            s.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.02, 0.355, 0],
                pos=[sx * (spec.TABLE_HALF[0] - 0.04),
                     sy * (spec.TABLE_HALF[1] - 0.04), -0.395],
                rgba=[0.3, 0.3, 0.32, 1], contype=0, conaffinity=0)

    # Concave block: visual mesh + shape-specific collision boxes.
    x, y, th = block_pose
    block = s.worldbody.add_body(name="tblock")
    block.pos = [x, y, 0.0005]
    block.quat = [np.cos(th / 2), 0, 0, np.sin(th / 2)]
    block.add_freejoint(name="tblock")
    gv = block.add_geom(name="tblock_visual", type=mujoco.mjtGeom.mjGEOM_MESH,
                        meshname="block_mesh", rgba=list(spec.BLOCK_RGBA),
                        contype=0, conaffinity=0)
    gv.mass = 0.0
    for i, (pos, half) in enumerate(spec.BLOCK_COLLISION_BOXES[shape_name]):
        g = block.add_geom(name=f"block_collision_{i}",
                           type=mujoco.mjtGeom.mjGEOM_BOX,
                           pos=list(pos), size=list(half),
                           rgba=[1, 1, 1, 0.0],
                           friction=list(spec.CONTACT_FRICTION))
        g.density = spec.BLOCK_DENSITY

    # lights
    lt = s.worldbody.add_light(pos=list(nuisance.key_light_pos),
                               dir=list(nuisance.key_light_dir))
    lt.diffuse = [nuisance.key_light_diffuse] * 3
    lt.specular = [0.08, 0.08, 0.08]
    lt2 = s.worldbody.add_light(pos=[-1.0, -0.6, 1.3], dir=[0.55, 0.35, -0.85])
    lt2.diffuse = [nuisance.fill_light_diffuse] * 3
    lt2.specular = [0.0, 0.0, 0.0]

    # sensor + overview cameras
    cam = s.worldbody.add_camera(name="sensor")
    cam.pos = list(spec.CAM_POS)
    cam.quat = _look_at_quat(spec.CAM_POS, spec.CAM_TARGET).tolist()
    cam.fovy = spec.FOVY_DEG
    ocam = s.worldbody.add_camera(name="overview")
    ocam.pos = [1.25, 1.05, 0.85]
    ocam.quat = _look_at_quat([1.25, 1.05, 0.85], [0.05, -0.05, 0.0]).tolist()
    ocam.fovy = 45.0

    # visible tripod marker for the sensor camera (overview renders only;
    # placed behind the lens so it never enters the sensor frustum)
    f = (np.asarray(spec.CAM_TARGET) - np.asarray(spec.CAM_POS))
    f /= np.linalg.norm(f)
    mpos = np.asarray(spec.CAM_POS) - 0.055 * f
    mk = s.worldbody.add_body(name="cam_marker")
    mk.pos = mpos.tolist()
    mk.quat = cam.quat
    mk.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.035, 0.02, 0.02],
                rgba=[0.12, 0.12, 0.13, 1], contype=0, conaffinity=0)
    s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                         size=[0.012, (spec.CAM_POS[2] + 0.75) / 2, 0],
                         pos=[mpos[0], mpos[1], (spec.CAM_POS[2] - 0.75) / 2],
                         rgba=[0.2, 0.2, 0.22, 1], contype=0, conaffinity=0)

    if with_arm:
        _attach_arm(s)

    model = s.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _attach_arm(s: mujoco.MjSpec) -> None:
    panda = mujoco.MjSpec.from_file(str(spec.PANDA_XML))
    for key in list(panda.keys):
        panda.delete(key)
    smesh = panda.add_mesh()
    smesh.name = "stick_d405"
    smesh.file = str(spec.STICK_STL)
    smesh.scale = [spec.MESH_SCALE] * 3
    mount = _find_body(panda, "attachment")
    stick = mount.add_body(name="stick")
    sv = stick.add_geom(name="stick_visual", type=mujoco.mjtGeom.mjGEOM_MESH,
                        meshname="stick_d405", rgba=[0.25, 0.26, 0.28, 1],
                        contype=0, conaffinity=0)
    sv.mass = 0.12
    sc = stick.add_geom(name="stick_shaft", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        rgba=[1, 1, 1, 0.0],
                        friction=list(spec.CONTACT_FRICTION))
    sc.fromto = [0, 0, 0.05, 0, 0, 0.185]
    sc.size = [spec.STICK_SHAFT_RADIUS, 0, 0]
    sc.mass = 0.0
    stick.add_site(name="stick_tip", pos=[0, 0, spec.STICK_TIP_OFFSET])

    frame = s.worldbody.add_frame()
    frame.pos = list(spec.ARM_BASE_POS)
    frame.quat = [np.cos(spec.ARM_BASE_YAW / 2), 0, 0,
                  np.sin(spec.ARM_BASE_YAW / 2)]
    s.attach(panda, prefix="arm_", frame=frame)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
def set_arm(model, data, qpos7, ctrl: bool = True) -> None:
    for i, qv in enumerate(qpos7, start=1):
        data.joint(f"arm_joint{i}").qpos[0] = qv
        if ctrl:
            data.actuator(f"arm_actuator{i}").ctrl[0] = qv
    mujoco.mj_forward(model, data)


def get_arm(data) -> list[float]:
    return [float(data.joint(f"arm_joint{i}").qpos[0]) for i in range(1, 8)]


def block_pose(model, data) -> tuple[float, float, float]:
    """Ground-truth (x, y, theta) of the block in the table frame."""
    q = data.joint("tblock").qpos
    yaw = np.arctan2(2 * (q[3] * q[6] + q[4] * q[5]),
                     1 - 2 * (q[5] ** 2 + q[6] ** 2))
    return float(q[0]), float(q[1]), spec.wrap_angle(float(yaw))


def set_block_pose(model, data, x, y, theta) -> None:
    q = data.joint("tblock").qpos
    q[:] = [x, y, 0.0005, np.cos(theta / 2), 0, 0, np.sin(theta / 2)]
    data.joint("tblock").qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def settle(model, data, duration: float = 0.4) -> None:
    for _ in range(int(duration / model.opt.timestep)):
        mujoco.mj_step(model, data)
