"""Mecanum kinematics: inverse kinematics, the base controller built on it,
and the holonomic-motion check.

Wheel order everywhere: FL, FR, RL, RR. Motor actuators: motor_<wheel>.
"""
from __future__ import annotations

import mujoco
import numpy as np

WHEEL_ORDER = ("FL", "FR", "RL", "RR")


def commanded_wheel_speeds(vx: float, vy: float, omega: float,
                           lx: float, ly: float, r: float) -> np.ndarray:
    """Standard 4-mecanum inverse kinematics (X-configuration).

    lx, ly: half wheelbase / half track; r: wheel ground-contact radius.
    Returns wheel angular speeds [FL, FR, RL, RR] (rad/s), positive = drives
    the robot toward +x.
    """
    L = lx + ly
    return np.array([
        vx - vy - L * omega,   # FL
        vx + vy + L * omega,   # FR
        vx + vy - L * omega,   # RL
        vx - vy + L * omega,   # RR
    ]) / r


def make_base_controller(model):
    """Open-loop mecanum inverse kinematics: model -> fn(data, twist) -> wheel
    speeds. The verifier's controller for the dynamic scenario battery."""
    lx, ly, radius = geometry_from_model(model)

    def step(_data, twist_cmd):
        return commanded_wheel_speeds(*twist_cmd, lx=lx, ly=ly, r=radius)

    return step


def body_twist(model, data, base_body: str = "base",
               free_joint: str = "base_free") -> np.ndarray:
    """Measured planar twist (vx, vy, omega) of the base, in the base frame.

    Reads the base free joint: linear velocity is world-frame, angular is
    body-local (MuJoCo free-joint convention).
    """
    adr = model.joint(free_joint).dofadr[0]
    v_world = data.qvel[adr:adr + 3]
    w_local = data.qvel[adr + 3:adr + 6]
    R = data.body(base_body).xmat.reshape(3, 3)
    v_local = R.T @ v_world
    return np.array([v_local[0], v_local[1], w_local[2]])


def geometry_from_model(model) -> tuple[float, float, float]:
    """(lx, ly, r) measured from the model itself: wheel joint anchor
    positions relative to the base and the ground-contact radius."""
    lx, ly = [], []
    for w in WHEEL_ORDER:
        pos = model.body(f"wheel_{w}").pos
        lx.append(abs(float(pos[0])))
        ly.append(abs(float(pos[1])))
    # ground-contact radius: roller-center circle + roller radius
    roller = model.geom("rollerg_FL_0")
    body = model.body("roller_FL_0")
    rc = float(np.hypot(body.pos[0], body.pos[2]))
    r = rc + float(roller.size[0])
    return float(np.mean(lx)), float(np.mean(ly)), r


def verify_holonomic_motion(model, data, twists=None,
                            settle_t: float = 1.0, run_t: float = 2.0,
                            tol: float = 0.25) -> dict:
    """Drive each test twist open-loop; compare achieved body twist to command.

    Requires a flat-floor scene. Each command runs from the settled home pose
    for run_t seconds; the achieved twist is averaged over the second half
    (steady state). A twist passes when each component is within `tol`
    (relative for the commanded components, absolute in the same units for
    the ones commanded zero: m/s or rad/s scaled by the command magnitude).

    Returns {"ok": bool, "cases": [{cmd, achieved, ok}, ...]}.
    """
    if twists is None:
        twists = [
            (0.4, 0.0, 0.0),    # forward
            (0.0, 0.4, 0.0),    # pure lateral: THE mecanum correctness probe
            (0.0, 0.0, 0.8),    # spin in place
            (0.3, 0.3, 0.0),    # diagonal
        ]
    lx, ly, r = geometry_from_model(model)
    cases = []
    for cmd in twists:
        speeds = commanded_wheel_speeds(*cmd, lx=lx, ly=ly, r=r)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        else:
            mujoco.mj_resetData(model, data)
        for _ in range(int(settle_t / model.opt.timestep)):
            mujoco.mj_step(model, data)
        for w, s in zip(WHEEL_ORDER, speeds):
            data.ctrl[model.actuator(f"motor_{w}").id] = s
        n_run = int(run_t / model.opt.timestep)
        measured = []
        for k in range(n_run):
            mujoco.mj_step(model, data)
            if k >= n_run // 2:
                measured.append(body_twist(model, data))
        achieved = np.mean(measured, axis=0)
        scale = max(float(np.max(np.abs(cmd))), 1e-9)
        ok = all(
            (abs(a - c) <= tol * abs(c)) if abs(c) > 1e-9 else (abs(a) <= tol * scale)
            for a, c in zip(achieved, cmd)
        )
        cases.append(dict(cmd=list(cmd), achieved=achieved.tolist(), ok=bool(ok)))
    return dict(ok=all(c["ok"] for c in cases), cases=cases)
