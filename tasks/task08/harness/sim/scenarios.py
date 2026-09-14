"""Test battery A-E.

Each scenario composes a scene (terrain + robot), drives a twist/arm-pose
schedule, and returns time series of the stability metrics plus hard-fail
flags. Terrain grade / bump field / payload are drawn deterministically from
(seed, Envelope) — the agent designs to the envelope and is tested on
held-out instances.

Anti-cheese: compose_scene() re-pins the physics options (timestep,
integrator, solver, friction cone, gravity) and gives terrain geoms contact
priority, so a submitted model cannot weaken the sim or the ground friction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .. import config
from ..metrics.fasm import net_com_force, force_angle_stability
from ..metrics.ssm import static_stability_margin
from ..metrics.support_polygon import (contact_normal_forces, support_polygon,
                                       support_polygon_3d, wheel_contacts)
from ..metrics.zmp import zmp as zmp_point
from .mecanum import (WHEEL_ORDER, body_twist, geometry_from_model,
                      make_base_controller)
from .arm_variants import ArmSpec, canonical_robot_spec, slew_limit


@dataclass
class Envelope:
    """Test-condition envelope. The agent sees these BOUNDS qualitatively in
    the task spec; concrete instances are drawn per seed and held out."""
    max_grade_deg: float = 10.0
    bump_amp_m: float = 0.012
    bump_wavelength_m: float = 0.5
    curb_height_m: float = 0.02   # ~25% of wheel radius: crossable at speed
    payload_min_kg: float = 1.0
    payload_max_kg: float = 3.0
    require_wheel_center_attachment: bool = True


# Scenario ids: A(static), B(accel/brake), C(cornering), D1(ramp), D2(terrain), E(combined)
SCENARIOS = ["A", "B", "C", "D1", "D2", "E"]

# Harness-pinned physics (determinism).
TIMESTEP = 0.002
SOLVER_ITERATIONS = 100
GRAVITY = (0.0, 0.0, -9.81)
DECIMATE = 2                      # metric logging every N steps (250 Hz)

LIFTOFF_FORCE_EPS = 1.0           # N: wheel considered unloaded below this
LIFTOFF_TAU = 0.3                 # s: support degraded longer than this = event
# A liftoff EVENT is fewer than 3 wheels loaded for > LIFTOFF_TAU. A rigid
# 4-wheel frame normally rocks onto 3 wheels on uneven ground (no suspension);
# one dangling wheel is not instability, losing two is.
TIP_TILT_DEG = 30.0               # tilt beyond this (vs terrain normal) = tipped
METRIC_WINDOW = 0.06              # s: contact/force aggregation window for the
                                  # polygon metrics — roller handover flickers
                                  # individual contacts far faster than any
                                  # physical tip-over develops


# ---------------------------------------------------------------------------
# Scene composition
# ---------------------------------------------------------------------------

_TERRAIN_FRICTION = "0.9 0.005 0.0001"


def pin_physics(spec) -> None:
    """Pin the harness physics options on a scene spec regardless of what the
    robot model declares (determinism: a submitted model cannot weaken the
    sim)."""
    spec.option.timestep = TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.option.gravity = GRAVITY
    spec.option.iterations = SOLVER_ITERATIONS
    spec.option.disableflags = 0


def attach_robot(spec, robot_xml: str, arm: ArmSpec) -> None:
    """Attach robot.xml, composed with the canonical arm, to a scene spec with
    its contact behavior clamped.

    Terrain must stay authoritative for pairwise contact parameters
    (friction, solref/solimp): MuJoCo uses the HIGHER-priority geom's params,
    so every robot geom is clamped below terrain priority BEFORE attaching —
    by identity, not by name, so a robot geom cannot spoof a terrain name to
    keep an elevated priority. Static geoms on the robot file's worldbody are
    rejected outright: a robot that leans on scenery it brought along is not
    a robot.
    """
    robot = canonical_robot_spec(robot_xml, arm)
    if len(robot.worldbody.geoms) > 0:
        raise ValueError("robot.xml must not define static worldbody geoms")
    for g in robot.geoms:
        g.priority = 0
    frame = spec.worldbody.add_frame()
    spec.attach(robot, prefix="", frame=frame)


def _smooth_bumps(rng: np.random.Generator, nrow: int, ncol: int,
                  extent_x: float, extent_y: float, wavelength: float) -> np.ndarray:
    """Deterministic smooth bump field in [0, 1], from seeded sinusoids."""
    xs = np.linspace(0, extent_x, ncol)
    ys = np.linspace(0, extent_y, nrow)
    X, Y = np.meshgrid(xs, ys)
    z = np.zeros_like(X)
    for _ in range(6):
        phi = rng.uniform(0, 2 * np.pi)      # direction
        lam = wavelength * rng.uniform(0.7, 1.4)
        psi = rng.uniform(0, 2 * np.pi)      # phase
        amp = rng.uniform(0.5, 1.0)
        z += amp * np.sin(2 * np.pi * (X * np.cos(phi) + Y * np.sin(phi)) / lam + psi)
    z -= z.min()
    if z.max() > 0:
        z /= z.max()
    return z


def compose_scene(robot_xml: str, scenario_id: str, seed: int, env: Envelope,
                  arm: ArmSpec):
    """Terrain + robot -> compiled (model, data, info).

    info: grade_deg, slope_quat (base alignment), terrain_normal, payload_kg,
    start_pos.
    """
    rng = np.random.default_rng([seed, SCENARIOS.index(scenario_id)])
    spec = mujoco.MjSpec()
    spec.modelname = f"rlebench_{scenario_id}_{seed}"

    info = dict(grade_deg=0.0, terrain_normal=np.array([0.0, 0.0, 1.0]),
                start_pos=np.array([0.0, 0.0, 0.0]), payload_kg=0.0)

    grade = 0.0
    if scenario_id == "D1":
        grade = math.radians(env.max_grade_deg) * rng.uniform(0.6, 1.0)
    elif scenario_id == "E":
        grade = math.radians(env.max_grade_deg) * 0.6

    # floor plane (rotated about +y by -grade so +x points uphill)
    floor = spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                    size=[0, 0, 0.05])
    floor.priority = 2  # terrain friction wins over whatever the robot declares
    floor.friction = [0.9, 0.005, 0.0001]
    if grade:
        floor.quat = [math.cos(-grade / 2), 0.0, math.sin(-grade / 2), 0.0]
        info["grade_deg"] = math.degrees(grade)
        # rotation about +y by -grade maps z_hat -> (-sin g, 0, cos g);
        # the surface rises toward +x (drive up-slope = +x)
        info["terrain_normal"] = np.array([-math.sin(grade), 0.0, math.cos(grade)])

    if scenario_id == "D2":
        # curb across the path, then a seeded bump field
        curb = spec.worldbody.add_geom(
            name="curb", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.06, 1.5, env.curb_height_m / 2],
            pos=[0.9, 0.0, env.curb_height_m / 2])
        curb.priority = 2
        curb.friction = [0.9, 0.005, 0.0001]
        hf = spec.add_hfield()
        hf.name = "bumps"
        nrow, ncol = 64, 128
        hf.nrow, hf.ncol = nrow, ncol
        half_x, half_y = 1.6, 1.2
        hf.size = [half_x, half_y, 2 * env.bump_amp_m, 0.05]
        hf.userdata = _smooth_bumps(rng, nrow, ncol, 2 * half_x, 2 * half_y,
                                    env.bump_wavelength_m).ravel()
        hfg = spec.worldbody.add_geom(name="bumps", type=mujoco.mjtGeom.mjGEOM_HFIELD,
                                      hfieldname="bumps", pos=[3.0, 0.0, -0.002])
        hfg.priority = 2
        hfg.friction = [0.9, 0.005, 0.0001]

    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1])

    attach_robot(spec, robot_xml, arm)
    pin_physics(spec)

    model = spec.compile()
    data = mujoco.MjData(model)

    # initial pose: aligned with the slope, wheels just touching
    _, _, wheel_r = geometry_from_model(model)
    qpos_adr = model.joint("base_free").qposadr[0]
    mujoco.mj_resetData(model, data)
    clearance = 0.001
    up = info["terrain_normal"]
    data.qpos[qpos_adr:qpos_adr + 3] = up * (wheel_r + clearance)
    if grade:
        half = -grade / 2  # rotate base about +y to lie on the slope
        data.qpos[qpos_adr + 3:qpos_adr + 7] = [math.cos(half), 0.0, math.sin(half), 0.0]
    else:
        data.qpos[qpos_adr + 3:qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    for i, q in enumerate(arm.stow):
        data.qpos[model.joint(arm.joint(i)).qposadr[0]] = q
        data.ctrl[model.actuator(arm.actuator(i)).id] = q
    info["start_pos"] = data.qpos[qpos_adr:qpos_adr + 3].copy()

    # payload draw (A and E use the max: worst case)
    if scenario_id in ("A", "E"):
        payload = env.payload_max_kg
    else:
        payload = float(rng.uniform(env.payload_min_kg, env.payload_max_kg))
    set_payload(model, data, payload, arm)
    info["payload_kg"] = payload
    return model, data, info


def set_payload(model, data, mass_kg: float, arm: ArmSpec) -> None:
    """Set the payload stub's mass (+ matching sphere inertia) in-place."""
    bid = model.body(arm.payload_body).id
    r = float(model.geom(arm.payload_geom).size[0])
    model.body_mass[bid] = mass_kg
    model.body_inertia[bid] = np.full(3, max(0.4 * mass_kg * r * r, 1e-9))
    # mj_setConst resets MjData in MuJoCo 3.5; preserve the initialized state.
    qpos, qvel = data.qpos.copy(), data.qvel.copy()
    act, ctrl, time = data.act.copy(), data.ctrl.copy(), float(data.time)
    mujoco.mj_setConst(model, data)
    data.qpos[:], data.qvel[:] = qpos, qvel
    data.act[:], data.ctrl[:], data.time = act, ctrl, time
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# Schedules: twist commands and arm targets as functions of time
# ---------------------------------------------------------------------------

def _ramp_hold(t, t0, t1, v0, v1):
    """Linear ramp from v0 at t0 to v1 at t1, clamped outside."""
    if t <= t0:
        return v0
    if t >= t1:
        return v1
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)


def _twist_profile(scenario_id: str, env: Envelope, info: dict):
    """Returns (duration_s, twist_fn(t) -> (vx, vy, wz))."""
    if scenario_id == "A":
        return 0.0, lambda t: (0.0, 0.0, 0.0)  # static; duration set by arm schedule
    if scenario_id == "B":
        # arm extends laterally during [0, 2]; then hard accel 0->1.2 m/s
        # @2 m/s^2, cruise, hard brake @2.4 m/s^2
        def fn(t):
            if t < 5.1:
                return (_ramp_hold(t, 2.5, 3.1, 0.0, 1.2), 0.0, 0.0)
            return (_ramp_hold(t, 5.1, 5.6, 1.2, 0.0), 0.0, 0.0)
        return 7.0, fn
    if scenario_id == "C":
        # arm extends forward during [0, 2]; lateral shake at speed, then arc
        def fn(t):
            if t < 4.0:
                return (0.0, _ramp_hold(t, 2.5, 3.0, 0.0, 1.0), 0.0)
            if t < 5.5:
                return (0.0, _ramp_hold(t, 4.0, 4.7, 1.0, -1.0), 0.0)
            if t < 6.2:
                return (0.0, _ramp_hold(t, 5.5, 6.0, -1.0, 0.0), 0.0)
            if t < 9.0:
                return (_ramp_hold(t, 6.2, 6.7, 0.0, 0.8), 0.0,
                        _ramp_hold(t, 6.2, 6.7, 0.0, 1.0))
            return (_ramp_hold(t, 9.0, 9.6, 0.8, 0.0), 0.0,
                    _ramp_hold(t, 9.0, 9.6, 1.0, 0.0))
        return 10.0, fn
    if scenario_id == "D1":
        # climb straight up the slope, ease out, then traverse across it
        def fn(t):
            vx = _ramp_hold(t, 0.5, 1.2, 0.0, 0.8) if t < 3.3 else _ramp_hold(t, 3.3, 4.0, 0.8, 0.0)
            vy = _ramp_hold(t, 4.2, 4.9, 0.0, 0.6) if t < 6.2 else _ramp_hold(t, 6.2, 6.8, 0.6, 0.0)
            return (vx, vy, 0.0)
        return 7.2, fn
    if scenario_id == "D2":
        # approach and cross the curb slowly, then speed up over the bumps
        def fn(t):
            if t < 4.0:
                return (_ramp_hold(t, 0.5, 1.1, 0.0, 0.4), 0.0, 0.0)
            return (_ramp_hold(t, 4.0, 4.6, 0.4, 0.65), 0.0, 0.0)
        return 9.0, fn
    if scenario_id == "E":
        # combined worst case: accelerate up the slope while the arm swings
        # out to full lateral reach with max payload
        def fn(t):
            return (_ramp_hold(t, 1.0, 1.8, 0.0, 1.0), 0.0, 0.0)
        return 6.0, fn
    raise ValueError(f"unknown scenario {scenario_id!r}")


def _arm_schedule(scenario_id: str, arm: ArmSpec):
    """List of (t_start, move_duration, target_pose). Linear ctrl interpolation."""
    stow = np.asarray(arm.stow, dtype=float)
    extended = arm.extended
    if scenario_id == "A":
        # Eight held poses sweeping joint 1 monotonically from one end of the
        # arm's reachable slewing span to the other. Equal steps, no long way
        # round: the sequence has to be monotone in JOINT space, because the
        # ctrl target is interpolated linearly between consecutive poses.
        limit = slew_limit(arm)
        sched = [(0.0, 0.2, stow)]
        t = 1.2
        for k in range(8):
            sched.append((t, 1.0, arm.extended_at_slew(-limit + 2.0 * limit * k / 7.0)))
            t += 1.8
        return sched, t + 0.5  # total duration
    if scenario_id == "B":
        # extended sideways: worst azimuth for the roll axis while the pitch
        # axis absorbs accel/brake (full forward extension + hard braking is
        # unpassable for ANY base inside the footprint budget)
        return [(0.0, 2.0, extended(math.pi / 2))], None
    if scenario_id == "C":
        return [(0.0, 2.0, extended(0.0))], None
    if scenario_id == "D1":
        return [(0.0, 0.2, stow)], None
    if scenario_id == "D2":
        return [(0.0, 0.2, stow)], None
    if scenario_id == "E":
        # swing from stow to full lateral (downhill side, -y) mid-scenario
        return [(0.0, 0.2, stow), (2.0, 2.0, extended(-math.pi / 2))], None
    raise ValueError(scenario_id)


def _arm_ctrl_at(sched, t, stow):
    """Interpolated arm ctrl target at time t.

    Schedule entries must be sequential (entry k+1 starts after entry k's
    move completes), which holds for every scenario here. Every scenario
    starts from the stow pose (set by compose_scene).
    """
    prev_pose = np.asarray(stow, dtype=float)
    result = prev_pose
    for t0, dur, pose in sched:
        pose = np.asarray(pose, dtype=float)
        if t < t0:
            break
        alpha = min((t - t0) / max(dur, 1e-9), 1.0)
        result = prev_pose + (pose - prev_pose) * alpha
        prev_pose = pose
    return result


# ---------------------------------------------------------------------------
# Runner core
# ---------------------------------------------------------------------------

def run_scenario(robot_xml: str, scenario_id: str, seed: int, env: Envelope,
                 controller=None, arm: ArmSpec | None = None) -> dict:
    """Run one scenario; return metric time series + hard-fail flags.

    controller: factory model -> fn(data, twist_cmd) -> 4 wheel ctrl values.
    Defaults to the open-loop mecanum inverse kinematics.
    """
    model, data, info = compose_scene(robot_xml, scenario_id, seed, env, arm)
    ctrl_fn = (controller or make_base_controller)(model)

    sched, arm_duration = _arm_schedule(scenario_id, arm)
    duration, twist_fn = _twist_profile(scenario_id, env, info)
    if arm_duration is not None:
        duration = max(duration, arm_duration)

    # settle before the profile starts
    for _ in range(int(1.0 / TIMESTEP)):
        mujoco.mj_step(model, data)

    n_steps = int(duration / TIMESTEP)
    arm_count = len(arm.joint_names)
    arm_ids = [model.actuator(arm.actuator(i)).id for i in range(arm_count)]
    wheel_ids = [model.actuator(f"motor_{w}").id for w in WHEEL_ORDER]
    ctrl_lo = model.actuator_ctrlrange[wheel_ids, 0]
    ctrl_hi = model.actuator_ctrlrange[wheel_ids, 1]
    up = info["terrain_normal"]

    log = dict(t=[], ssm=[], fasm=[], zmp=[], tilt_deg=[], wheel_n=[],
               twist_cmd=[], twist_meas=[], com=[], arm_q=[],
               # full base velocity, for telling "at rest" from "still
               # swinging": twist_meas carries yaw only, and the rocking that
               # outlasts an arm move is pitch and roll.
               base_speed=[], base_omega=[])
    tip = False
    tip_time = None
    degraded_since = None            # start of a <3-wheels-loaded interval
    unloaded_steps = {w: 0 for w in WHEEL_ORDER}
    liftoff_events = []
    stow = np.asarray(arm.stow, dtype=float)
    history: list = []  # (t, contacts, net_force) within METRIC_WINDOW

    for k in range(n_steps):
        t = k * TIMESTEP
        cmd = np.asarray(twist_fn(t), dtype=float)
        wheel_ctrl = np.clip(np.asarray(ctrl_fn(data, cmd), dtype=float),
                             ctrl_lo, ctrl_hi)
        data.ctrl[wheel_ids] = wheel_ctrl
        data.ctrl[arm_ids] = _arm_ctrl_at(sched, t, stow)
        mujoco.mj_step(model, data)

        # hard-fail and liftoff bookkeeping every step
        R = data.body("base").xmat.reshape(3, 3)
        cos_tilt = float(np.clip(R[:, 2] @ up, -1.0, 1.0))
        tilt = math.degrees(math.acos(cos_tilt))
        if tilt > TIP_TILT_DEG and not tip:
            tip, tip_time = True, t
        normals = contact_normal_forces(model, data)
        loaded = sum(normals.get(w, 0.0) >= LIFTOFF_FORCE_EPS for w in WHEEL_ORDER)
        if loaded < 3:
            if degraded_since is None:
                degraded_since = t
        elif degraded_since is not None:
            dt_off = t - degraded_since
            if dt_off > LIFTOFF_TAU:
                liftoff_events.append(dict(t=degraded_since, duration=dt_off))
            degraded_since = None
        for w in WHEEL_ORDER:
            if normals.get(w, 0.0) < LIFTOFF_FORCE_EPS:
                unloaded_steps[w] += 1

        if k % DECIMATE:
            continue
        com = data.subtree_com[0].copy()
        # windowed aggregation: union of recent contact points, mean force
        history.append((t, wheel_contacts(model, data), net_com_force(model, data)))
        while history and history[0][0] < t - METRIC_WINDOW:
            history.pop(0)
        pts = [h[1] for h in history if len(h[1])]
        contacts = np.vstack(pts) if pts else np.zeros((0, 3))
        poly = support_polygon(contacts)
        poly3 = support_polygon_3d(contacts)
        # median, not mean: robust to single-step contact impulses
        # (roller stick-slip), which would otherwise spike the force angle
        f = np.median([h[2] for h in history], axis=0)
        log["t"].append(t)
        log["base_speed"].append(float(np.linalg.norm(data.qvel[:3])))
        log["base_omega"].append(float(np.linalg.norm(data.qvel[3:6])))
        log["ssm"].append(static_stability_margin(com[:2], poly))
        log["fasm"].append(force_angle_stability(com, poly3, f))
        log["zmp"].append(zmp_point(model, data))
        log["tilt_deg"].append(tilt)
        log["wheel_n"].append([normals.get(w, 0.0) for w in WHEEL_ORDER])
        log["twist_cmd"].append(cmd)
        log["twist_meas"].append(body_twist(model, data))
        log["com"].append(com)
        log["arm_q"].append([data.qpos[model.joint(arm.joint(i)).qposadr[0]]
                             for i in range(arm_count)])
        if tip:
            break  # once tipped, further numbers are noise

    # close out a dangling degraded-support interval
    t_end = min(n_steps, k + 1) * TIMESTEP if n_steps else 0.0
    if degraded_since is not None and t_end - degraded_since > LIFTOFF_TAU:
        liftoff_events.append(dict(t=degraded_since, duration=t_end - degraded_since))

    result = {key: np.asarray(val) for key, val in log.items()}
    result.update(scenario=scenario_id, seed=seed, params=info,
                  tip=tip, tip_time=tip_time, liftoff_events=liftoff_events,
                  wheel_unload_frac={w: unloaded_steps[w] / max(n_steps, 1)
                                     for w in WHEEL_ORDER},
                  duration=duration)
    fasm = result["fasm"]
    result["min_fasm"] = float(np.min(fasm)) if fasm.size else float("-inf")
    result["min_fasm_robust"] = _robust_min(fasm)
    ssm = result["ssm"]
    result["min_ssm"] = float(np.min(ssm)) if ssm.size else float("-inf")
    result["min_ssm_robust"] = _robust_min(ssm)
    result["max_tilt_deg"] = float(np.max(result["tilt_deg"])) if result["tilt_deg"].size else 90.0
    return result


ROBUST_WINDOW_S = 0.2   # rolling-median span for the scored minimum


def _robust_min(series: np.ndarray) -> float:
    """Minimum of the rolling median: a margin loss must persist for on the
    order of ROBUST_WINDOW_S to count. Single-sample solver impulses (ms
    scale) cannot physically tip a ~50 kg robot and are excluded; sustained
    dips pass through unchanged."""
    if series.size == 0:
        return float("-inf")
    n = max(int(ROBUST_WINDOW_S / (TIMESTEP * DECIMATE)) | 1, 1)  # odd
    if series.size < n:
        return float(np.median(series))
    from scipy.ndimage import median_filter
    return float(np.min(median_filter(series, size=n, mode="nearest")))


def scenario_a_min_ssm(result: dict) -> float:
    """Scenario A pass metric: min SSM over the QUIESCENT part of each dwell.

    Scenario A holds eight arm poses in turn and the checkpoint it feeds is a
    STATIC margin, so a sample only counts once the base has actually stopped
    moving: on compliant mecanum rollers the swing after an arm move can
    outlast the dwell, and grading it would report a transient instead. A
    dwell the base never settles in earns no credit (-inf), which is the
    honest reading — that pose has no static margin to report.
    """
    t = result["t"]
    ssm = result["ssm"]
    if t.size == 0:
        return float("-inf")
    speed = np.asarray(result.get("base_speed", []), dtype=float)
    omega = np.asarray(result.get("base_omega", []), dtype=float)
    if speed.size != t.size or omega.size != t.size:
        return float("-inf")   # untraced run: nothing can be certified static
    # "at rest on the full support polygon". Wheel normals matter as much as
    # velocity: with a wheel off the ground the contact hull degenerates to a
    # segment, and SSM against a degenerate hull is not a static margin.
    # Liftoff itself stays graded by the scenario's own liftoff events, so
    # nothing is hidden by requiring full support here.
    wheel_n = np.asarray(result.get("wheel_n", []), dtype=float)
    if wheel_n.shape[:1] != t.shape:
        return float("-inf")
    supported = np.all(wheel_n >= LIFTOFF_FORCE_EPS, axis=1)
    quiet = ((speed < config.SCENARIO_A_QUIET_V)
             & (omega < config.SCENARIO_A_QUIET_W)
             & supported)
    # dwell boundaries: the stow hold, then one per commanded arm pose
    edges = [0.0] + [1.2 + 1.8 * k for k in range(8)] + [float(t[-1]) + 1e-9]
    mins = []
    for lo, hi in zip(edges, edges[1:]):
        window = (t >= lo) & (t < hi) & quiet
        if not window.any():
            return float("-inf")
        mins.append(float(np.min(ssm[window])))
    return min(mins) if mins else float("-inf")
