"""Frame and episode generation for the task06 family.

Push episodes are REAL simulated pushing: seeded stroke segments are turned
into joint-space waypoints by damped-least-squares IK, the Panda's stock
position actuators servo along them, and MuJoCo contact physics (stick shaft
against the block's collision boxes) moves the block. Ground truth is the
block free joint read from simulator state.

The rendering service serves agents from this module on the public design
seeds; the evaluation calls the same code on unpublished seeds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import scene, sensor, spec
from .training_client import MAX_STROKES

ARM_REACH_INIT = (-0.3, 0.5, 0.0, -2.2, 0.0, 2.7, 0.785)
LIFT_HEIGHT = 0.12        # tip height between strokes
PUSH_SPEED = 0.18         # m/s along a pushing stroke
TRANSIT_SPEED = 0.45      # m/s while repositioning
PUSH_OFFSET_RANGE = (0.65, 0.85)  # fraction of footprint extent from its COM
CAMERA_PUSH_OFFSET_RANGE = (0.45, 0.65)
CAMERA_STICK_TILT_SIN = 0.55  # wrist toward the camera, tip on the footprint
PUSH_CLEARANCE = 0.025    # gap outside the footprint before contact
PUSH_TRAVEL = 0.18       # inward travel from the first contact
REACH_BOUND = (0.33, 0.26)  # |x|, |y| clamp for stroke endpoints

OCCLUDED_FRACTION = 0.4   # frame flagged occluded at >= this probe fraction
OCCLUDED_ROTATION_DEG = 30.0
OCCLUDED_ROTATING_STEPS = 6
ROTATION_RANGE_DEG = 60.0
MOTION_CANDIDATES = 12
MOTION_WINDOW_FRAMES = 60


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


# ---------------------------------------------------------------------------
# IK (damped least squares, position + soft keep-stick-vertical objective)
# ---------------------------------------------------------------------------
def ik_to(model, data, target, init7=None, site: str = "arm_stick_tip",
          iters: int = 300, w_rot: float = 0.25, stick_axis=None):
    """Move ``site`` to ``target``; returns (q7, residual_m).

    ``data`` should be a scratch MjData — this mutates arm qpos. The rotation
    objective keeps the stick pointing down and relaxes automatically through
    ``w_rot``; note the first-order pairing: with the Jacobian row
    ``-skew(z_cur) @ jacr`` the consistent error is ``z_des - z_cur``.
    """
    if init7 is not None:
        scene.set_arm(model, data, init7, ctrl=False)
    sid = model.site(site).id
    dofs = [model.joint(f"arm_joint{i}").dofadr[0] for i in range(1, 8)]
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    lo = np.array([model.joint(f"arm_joint{i}").range[0] for i in range(1, 8)])
    hi = np.array([model.joint(f"arm_joint{i}").range[1] for i in range(1, 8)])
    z_des = np.array([0.0, 0.0, -1.0] if stick_axis is None else stick_axis)
    target = np.asarray(target, dtype=float)
    for _ in range(iters):
        e_pos = target - data.site_xpos[sid]
        z_cur = data.site_xmat[sid].reshape(3, 3)[:, 2]
        e_rot = (z_des - z_cur) * w_rot
        if np.linalg.norm(e_pos) < 1e-4 and np.linalg.norm(e_rot) < 5e-3:
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        J = np.vstack([jacp[:, dofs], (-_skew(z_cur) @ jacr[:, dofs]) * w_rot])
        e = np.concatenate([e_pos, e_rot])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        q = np.clip(np.array(scene.get_arm(data)) + 0.5 * dq, lo, hi)
        scene.set_arm(model, data, q.tolist(), ctrl=False)
    res = float(np.linalg.norm(target - data.site_xpos[sid]))
    return scene.get_arm(data), res


def _waypoint_qs(model, ik_data, p0, p1, speed, dt_ctrl, w_rot=0.25,
                 relax_rot=True, stick_axis=None):
    """Joint waypoints tracking the straight segment p0 -> p1."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    seg_t = max(np.linalg.norm(p1 - p0) / speed, dt_ctrl)
    n = max(int(seg_t / dt_ctrl), 2)
    qs = []
    for k in range(n + 1):
        a = k / n
        wr = w_rot * (1 - 0.8 * a) if relax_rot else w_rot
        tgt = (1 - a) * p0 + a * p1
        q, _ = ik_to(model, ik_data, tgt, iters=120, w_rot=wr,
                    stick_axis=stick_axis)
        qs.append(q)
    return qs


# ---------------------------------------------------------------------------
# Stroke sampling
# ---------------------------------------------------------------------------
_BASE_XY = np.array(spec.ARM_BASE_POS[:2])
MIN_BASE_DIST = 0.30  # a downward 30 cm tool can't reach into the base column
MAX_BASE_DIST = 0.80  # reserve reach for the vertical stick orientation


def _clamp_xy(p):
    p = np.array([np.clip(p[0], -REACH_BOUND[0], REACH_BOUND[0]),
                  np.clip(p[1], -REACH_BOUND[1], REACH_BOUND[1])])
    v = p - _BASE_XY
    d = np.linalg.norm(v)
    if d < MIN_BASE_DIST:
        # project radially out of the arm's inner-workspace dead zone
        # (may exceed REACH_BOUND slightly near the base-side corner — fine,
        # the tip just hovers past the table edge)
        p = _BASE_XY + v / max(d, 1e-9) * MIN_BASE_DIST
    elif d > MAX_BASE_DIST:
        p = _BASE_XY + v / d * MAX_BASE_DIST
    return p


def _sample_stroke(rng: np.random.Generator, block_pose, toward_camera=False,
                  shape_name=spec.DEFAULT_BLOCK_SHAPE):
    """An off-center stroke through the actual collision footprint.

    Returns (start, end) tip positions at push height. ``toward_camera``
    forces the approach azimuth to the camera side so the arm crosses the
    line of sight (this is what makes occluded frames occur by construction).
    """
    bx, by = block_pose[:2]
    theta = block_pose[2] if len(block_pose) > 2 else 0.0
    if toward_camera:
        cam_az = np.arctan2(spec.CAM_POS[1] - by, spec.CAM_POS[0] - bx)
        az = cam_az + rng.uniform(-0.5, 0.5)
    else:
        az = rng.uniform(-np.pi, np.pi)
    d = np.array([np.cos(az), np.sin(az)])
    perp = np.array([-d[1], d[0]])
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    boxes = spec.BLOCK_COLLISION_BOXES[shape_name]
    masses = np.array([np.prod(half) for _, half in boxes])
    com = np.average([center[:2] for center, _ in boxes], axis=0, weights=masses)
    corners = np.array([np.asarray(center[:2]) + np.array([sx, sy]) * half[:2]
                        for center, half in boxes for sx in (-1, 1) for sy in (-1, 1)])
    projections = (corners - com) @ rot.T @ perp
    extent = projections.max() if rng.random() < 0.5 else projections.min()
    offsets = CAMERA_PUSH_OFFSET_RANGE if toward_camera else PUSH_OFFSET_RANGE
    local_center = com + rot.T @ perp * extent * rng.uniform(*offsets)
    local_d = rot.T @ d
    intersections = []
    for center, half in boxes:
        lo, hi = -np.inf, np.inf
        for axis in range(2):
            if abs(local_d[axis]) < 1e-9:
                if abs(local_center[axis] - center[axis]) > half[axis]:
                    hi = -np.inf
                    break
                continue
            t0 = (center[axis] - half[axis] - local_center[axis]) / local_d[axis]
            t1 = (center[axis] + half[axis] - local_center[axis]) / local_d[axis]
            lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
        if lo <= hi:
            intersections.append(hi)
    contact = max(intersections)
    center = np.array([bx, by]) + rot @ local_center
    start = center + d * (contact + spec.STICK_SHAFT_RADIUS + PUSH_CLEARANCE)
    end = center + d * (contact - PUSH_TRAVEL)
    z = spec.PUSH_TIP_HEIGHT
    return np.array([*start, z]), np.array([*end, z])


def sample_stroke(rng, block_pose, toward_camera=False,
                  shape_name=spec.DEFAULT_BLOCK_SHAPE):
    """Prefer reachable strokes without changing their off-center contact line."""
    best, best_error = None, np.inf
    for _ in range(32):
        points = _sample_stroke(rng, block_pose, toward_camera, shape_name)
        clipped = [np.array([*_clamp_xy(p[:2]), p[2]]) for p in points]
        error = sum(np.linalg.norm(a - b) for a, b in zip(points, clipped))
        if error < best_error:
            best, best_error = clipped, error
        if error < 1e-9:
            break
    return tuple(best)


# ---------------------------------------------------------------------------
# Occlusion probe
# ---------------------------------------------------------------------------
def occluded_fraction(model, data) -> float:
    """Fraction of block top-face probe points hidden from the camera."""
    x, y, th = scene.block_pose(model, data)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    block_body = model.body("tblock").id
    local = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != block_body \
                or not model.geom(gid).name.startswith("block_collision_"):
            continue
        cx, cy, cz = model.geom_pos[gid]
        hx, hy, hz = model.geom_size[gid]
        local.append([cx, cy, cz + hz])
        for sx in (-0.75, 0.75):
            for sy in (-0.75, 0.75):
                local.append([cx + sx * hx, cy + sy * hy, cz + hz])
    pts = np.unique(np.round(np.asarray(local), 8), axis=0)
    pts = pts @ R.T + np.array([x, y, 0.0])
    cam = np.asarray(spec.CAM_POS)
    geomid = np.zeros(1, dtype=np.int32)
    hidden = 0
    for p in pts:
        vec = p - cam
        dist = np.linalg.norm(vec)
        mujoco.mj_ray(model, data, cam, vec / dist, None, 1, -1, geomid)
        gid = int(geomid[0])
        if gid < 0:
            continue
        if model.geom_bodyid[gid] != block_body:
            hidden += 1
    return hidden / len(pts)


# ---------------------------------------------------------------------------
# Frames and episodes
# ---------------------------------------------------------------------------
@dataclass
class Frame:
    obs: dict | None
    gt: tuple
    occluded: bool
    t: float
    shape_id: int = -1  # private metadata, never serialized to observations


def shape_for_seed(seed: int, shapes=None) -> str:
    """Deterministically select a shape without exposing it in observations."""
    pool = tuple(shapes) if shapes is not None else (spec.DEFAULT_BLOCK_SHAPE,)
    if not pool or any(name not in spec.BLOCK_SHAPES for name in pool):
        raise ValueError("shapes must be a non-empty subset of BLOCK_SHAPES")
    rng = np.random.default_rng([int(seed), spec.STREAM_SHAPE])
    return pool[int(rng.integers(0, len(pool)))]


@dataclass
class Episode:
    seed: int
    frames: list = field(default_factory=list)
    max_tilt_deg: float = 0.0
    max_height_m: float = 0.0

    @property
    def gts(self):
        return [f.gt for f in self.frames]

    @property
    def occluded_flags(self):
        return [f.occluded for f in self.frames]


def single_frame(seed: int, render: bool = True, shapes=None) -> Frame:
    """One settled, unoccluded observation (arm parked). Stage-A style."""
    model, data = scene.build_scene(
        seed=seed, shape_name=shape_for_seed(seed, shapes))
    scene.set_arm(model, data, spec.ARM_PARKED)
    scene.settle(model, data, 0.4)
    gt = scene.block_pose(model, data)
    obs = None
    if render:
        with sensor.CameraRig(model) as rig:
            obs = sensor.observation(rig, data, seed, frame_idx=0, t=0.0)
    return Frame(obs=obs, gt=gt, occluded=False, t=0.0,
                 shape_id=spec.BLOCK_SHAPES.index(shape_for_seed(seed, shapes)))


def _push_episode(seed: int, n_strokes: int, render: bool,
                  max_frames: int, shapes, attempt: int) -> Episode:
    """A seeded pushing episode: contact physics, observations at FRAME_HZ.

    Stroke k >= 1 is planned from the block's pose at that moment (online
    replanning), so the stick keeps engaging the block as it wanders.
    Camera-side off-center strokes overlap contact-driven rotation with arm
    occlusion. Candidate draws change only the stroke RNG, not the scene.
    """
    rng = np.random.default_rng([seed, spec.STREAM_STROKES, attempt])
    shape_name = shape_for_seed(seed, shapes)
    model, data = scene.build_scene(seed=seed, shape_name=shape_name)
    ik_data = mujoco.MjData(model)
    dt = model.opt.timestep
    dt_ctrl = dt * 10                      # 50 Hz control targets
    steps_per_frame = int(round(1.0 / (spec.FRAME_HZ * dt)))

    episode = Episode(seed=seed)
    rig = sensor.CameraRig(model) if render else None

    # settle with the arm parked, then move to the first stroke start
    scene.set_arm(model, data, spec.ARM_PARKED)
    scene.settle(model, data, 0.4)

    step_count = 0

    def run_targets(qs):
        nonlocal step_count
        for q in qs:
            for i, qv in enumerate(q, start=1):
                data.actuator(f"arm_actuator{i}").ctrl[0] = qv
            for _ in range(int(round(dt_ctrl / dt))):
                if len(episode.frames) >= max_frames:
                    return
                if step_count % steps_per_frame == 0:
                    _capture()
                mujoco.mj_step(model, data)
                step_count += 1

    def _capture():
        t = step_count * dt
        gt = scene.block_pose(model, data)
        q = data.joint("tblock").qpos
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[4]**2 + q[5]**2), -1, 1)))
        episode.max_tilt_deg = max(episode.max_tilt_deg, float(tilt))
        episode.max_height_m = max(episode.max_height_m, abs(float(q[2])))
        occ = occluded_fraction(model, data) >= OCCLUDED_FRACTION
        obs = None
        if rig is not None:
            obs = sensor.observation(rig, data, seed,
                                     frame_idx=len(episode.frames), t=t)
        episode.frames.append(Frame(obs=obs, gt=gt, occluded=occ, t=t,
                                    shape_id=spec.BLOCK_SHAPES.index(shape_name)))

    try:
        for k in range(n_strokes):
            if len(episode.frames) >= max_frames:
                break
            block_pose = scene.block_pose(model, data)
            start, end = sample_stroke(rng, block_pose, toward_camera=True,
                                       shape_name=shape_name)
            lift = np.array([*start[:2], LIFT_HEIGHT])
            camera_direction = np.asarray(spec.CAM_POS[:2]) - np.asarray(block_pose[:2])
            camera_direction /= np.linalg.norm(camera_direction)
            stick_axis = np.r_[-CAMERA_STICK_TILT_SIN * camera_direction,
                               -np.sqrt(1 - CAMERA_STICK_TILT_SIN**2)]
            # plan from the arm's current configuration
            scene.set_arm(model, ik_data, scene.get_arm(data), ctrl=False)
            q_lift = _waypoint_qs(model, ik_data,
                                  np.asarray(_tip_pos(model, ik_data)),
                                  lift, TRANSIT_SPEED, dt_ctrl,
                                  relax_rot=False)
            q_desc = _waypoint_qs(model, ik_data, lift, start, TRANSIT_SPEED,
                                  dt_ctrl, relax_rot=False, stick_axis=stick_axis)
            q_push = _waypoint_qs(model, ik_data, start, end, PUSH_SPEED,
                                  dt_ctrl, relax_rot=False, stick_axis=stick_axis)
            run_targets(q_lift)
            run_targets(q_desc)
            run_targets(q_push)
    finally:
        if rig is not None:
            rig.close()
    return episode


def occluded_motion(episode):
    """Angular travel only across intervals whose two endpoints are occluded."""
    yaw = np.unwrap(np.asarray(episode.gts)[:, 2])
    turns = np.abs(np.degrees(np.diff(yaw)))
    flags = np.asarray(episode.occluded_flags, dtype=bool)
    hidden = flags[:-1] & flags[1:]
    return dict(rotation_deg=float(turns[hidden].sum()),
                rotating_steps=int(np.count_nonzero(hidden & (turns > 1.0))),
                range_deg=float(np.degrees(np.ptp(yaw))))


def _motion_quality(episode):
    if not episode.frames or episode.max_tilt_deg > 15 or episode.max_height_m > .02:
        return -1.0
    if np.any(np.abs(np.asarray(episode.gts)[:, :2]) >= spec.POSE_XY_BOUND):
        return -1.0
    window = Episode(episode.seed, episode.frames[:MOTION_WINDOW_FRAMES])
    motion = occluded_motion(window)
    return min(motion["rotation_deg"] / OCCLUDED_ROTATION_DEG,
               motion["rotating_steps"] / OCCLUDED_ROTATING_STEPS,
               motion["range_deg"] / ROTATION_RANGE_DEG)


def push_episode(seed: int, n_strokes: int = MAX_STROKES, render: bool = True,
                 max_frames: int = 100, shapes=None) -> Episode:
    """Choose a bounded, deterministic contact rollout with occluded rotation.

    Selection uses simulator state, never an estimator. Scene, shape and
    sensor draws remain fixed; only candidate push paths differ. If no path
    reaches the coverage targets, retain the best available planar rollout.
    """
    best, best_attempt, quality = None, 0, -np.inf
    for attempt in range(MOTION_CANDIDATES):
        candidate = _push_episode(seed, n_strokes, False, max_frames, shapes, attempt)
        score = _motion_quality(candidate)
        if score > quality:
            best, best_attempt, quality = candidate, attempt, score
        if score >= 1.0:
            break
    if quality < 0:
        raise RuntimeError("no planar in-bounds push candidate")
    if not render:
        return best
    return _push_episode(seed, n_strokes, render, max_frames, shapes, best_attempt)


def _tip_pos(model, ik_data):
    mujoco.mj_forward(model, ik_data)
    return ik_data.site("arm_stick_tip").xpos.copy()
