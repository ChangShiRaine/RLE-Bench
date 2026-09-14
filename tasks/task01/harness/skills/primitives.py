"""Motion: the tier that closes the loop.

Each is a plain function taking `sim` and calling `sim.step` in a loop, exactly as your own
code would. None of them looks anything up -- you supply the target.

Every position here is in the ROBOT BASE frame, which is the frame actions are in and the
frame `robot0_base_to_eef_pos` reports. From the cameras, use `camera.cloud_in_base`.

Each returns `sim.step`'s own reply for the last step taken plus `outcome` -- one of
`reached`, `gave_up`, `blocked`. See MANUAL.md.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..client import RemoteError
from ..obs import ObsSpec
from .transforms import (base_pose_in_world, mat_to_axisangle, quat_to_mat,
                         transform_points, world_to_base)

# These servo on proprioception and never look, so skipping the render makes each step
# markedly cheaper in wall clock.
PROPRIO_ONLY = ObsSpec(cameras=())

REACHED = "reached"      # the sub-goal is met, as far as this skill can tell
GAVE_UP = "gave_up"      # tried, stopped making progress
BLOCKED = "blocked"      # cannot start from this state

ACTION_DIM = 12
_ARM_POS = slice(0, 3)
_ARM_ROT = slice(3, 6)
_GRIPPER = 6
_BASE = slice(7, 10)
_TORSO = 10
_BASE_MODE = 11

# Proportional gains on the OSC pose delta: an action of +-1 commands +-5 cm or
# +-0.5 rad, which the impedance controller does not reach in one step.
#
# CALIBRATED ON ONE SCENE, unloaded arm. Contact, payload and arm extension change the
# response: a motion that oscillates or crawls means the gain is wrong for YOUR
# situation, and re-measuring it there costs a few dozen steps once.
KP = 8.0
KR = 2.0
KP_LIFT = 4.0            # gentler: a yanked grasp sheds what it is holding
# Position integral: a proportional command alone parks short wherever gravity or
# contact needs a standing command to hold. Integrated only inside I_BAND -- outside
# it the proportional term is already saturated and winding up overshoots on arrival.
KI = 2.0
I_BAND = 0.04            # metres
I_CLAMP = 0.8            # cap on the integrated command, in action units
I_LEAK = 0.8             # decay per step outside the band
KP_BASE = 15.0
KP_YAW = 8.0

# The base joints sit behind heavy static friction: a command below ~0.25 moves nothing.
# Commands are floored outside DEAD; inside it the axis is released to friction.
BASE_FLOOR = np.array([0.30, 0.30, 0.20])
BASE_DEAD = np.array([0.02, 0.02, 0.02])    # metres, metres, radians

# hold_eef: the friction floor sets a MINIMUM base speed that outruns the lagged arm,
# so the base drives in CHUNK-sized bursts and pauses while the arm re-pins the hand.
KP_BASE_HOLD = 6.0
KP_YAW_HOLD = 3.0
BASE_CAP_HOLD = np.array([0.35, 0.35, 0.20])
HOLD_CHUNK = 0.03        # metres of base travel per burst
HOLD_CHUNK_YAW = 0.05    # radians per burst
HOLD_REPIN_TOL = 0.02    # arm error that counts as re-pinned, metres
HOLD_REPIN_MAX = 8       # steps allowed per re-pin pause
HOLD_SLIP_TOL = 0.03     # arm lag that pauses a push early, metres
KP_HOLD = 24.0           # arm correction while the base drags it

_OPEN, _CLOSED = -1.0, 1.0
_STALL_STEPS = 12        # no measurable progress for this many steps -> gave_up
_STALL_EPS = 1e-4        # metres of progress that counts as progress
_STALL_REL = 0.005       # ... or this fraction of the remaining gap, whichever is larger


def _action(*, pos_err=None, pos_int=None, rot_err=None, gripper=0.0, base=None,
            base_mode=-1.0, gain: float = KP) -> list:
    """One 12-vector. Everything unset is zero, and arm mode unless told otherwise."""
    a = np.zeros(ACTION_DIM)
    if pos_err is not None:
        cmd = gain * np.asarray(pos_err, dtype=float)
        if pos_int is not None:
            cmd = cmd + pos_int
        a[_ARM_POS] = np.clip(cmd, -1.0, 1.0)
    if rot_err is not None:
        a[_ARM_ROT] = np.clip(KR * np.asarray(rot_err, dtype=float), -1.0, 1.0)
    a[_GRIPPER] = float(gripper)
    if base is not None:
        a[_BASE] = np.clip(np.asarray(base, dtype=float), -1.0, 1.0)
    a[_TORSO] = 0.0
    a[_BASE_MODE] = float(base_mode)
    return a.tolist()


def _begin(sim) -> dict | None:
    """The current proprioception, or None if there is nothing to act in. Free.

    An observation taken after the episode ended is THINNER than the one asked for, so
    `live` is checked here rather than met as a KeyError inside a control law.
    """
    try:
        look = sim.observe(PROPRIO_ONLY)
    except RemoteError:
        return None
    return look["obs"] if look.get("live", True) else None


def _idle(sim, outcome: str) -> dict:
    """The reply for a skill that finished, or could not start, without a step."""
    try:
        look = sim.observe(PROPRIO_ONLY)
        obs, live = look["obs"], look.get("live", True)
    except RemoteError:
        obs, live = {}, False
    return {"obs": obs, "steps": 0, "success": None, "episode_over": not live,
            "ended": None, "steps_remaining": None, "outcome": outcome}


def _drive(sim, obs, action, done, max_steps, error=None) -> dict:
    """The loop every primitive is: act until `done(obs)`, the episode ends, or progress
    stops.

    `action(obs)` is the command to send and `error(obs)` the distance still to cover;
    given one, the loop gives up as soon as that distance stops shrinking, because
    `max_steps` is a ceiling and the interaction it would spend is the agent's score.

    Returns as soon as the episode ends, so no primitive ever steps a dead one.
    """
    best, stalled, res = np.inf, 0, None
    for _ in range(int(max_steps)):
        if done(obs):
            return _finish(sim, res, REACHED)
        if error is not None:
            gap = float(error(obs))
            eps = max(_STALL_EPS, _STALL_REL * gap)
            stalled = 0 if gap < best - eps else stalled + 1
            best = min(best, gap)
            if stalled >= _STALL_STEPS:
                return _finish(sim, res, GAVE_UP)
        try:
            res = sim.step(action(obs), obs_spec=PROPRIO_ONLY)
        except RemoteError:
            return _idle(sim, BLOCKED)
        if res["episode_over"]:
            return {**res, "outcome": REACHED if res.get("success") else GAVE_UP}
        obs = res["obs"]
    return _finish(sim, res, GAVE_UP)


def _finish(sim, res: dict | None, outcome: str) -> dict:
    return {**res, "outcome": outcome} if res else _idle(sim, outcome)


def _eef(obs) -> np.ndarray:
    return np.asarray(obs["robot0_base_to_eef_pos"], dtype=float)


def _rot_mat(rot) -> np.ndarray:
    """An orientation target as a (3, 3) matrix, from a matrix or an xyzw quaternion."""
    r = np.asarray(rot, dtype=float)
    if r.shape == (4,):
        return quat_to_mat(r)
    if r.shape == (3, 3):
        return r
    raise ValueError("rot must be a (3, 3) matrix or an xyzw quaternion")


def _report(res: dict, gap_p=None, gap_r=None) -> dict:
    """Add the final errors to the reply, so retry-versus-abandon needs no extra look."""
    o = res.get("obs") or {}
    if gap_p is not None and "robot0_base_to_eef_pos" in o:
        res["pos_err"] = float(np.linalg.norm(gap_p(o)))
    if gap_r is not None and "robot0_base_to_eef_quat" in o:
        res["rot_err"] = float(np.linalg.norm(gap_r(o)))
    return res


def _servo(sim, obs, target, tol, max_steps, gain, gripper, rot=None,
           rot_tol: float = 0.05) -> dict:
    """Drive the end-effector to a base-frame `target`, and `rot` if one is given."""
    goal = np.asarray(target, dtype=float).reshape(3)
    want = None if rot is None else _rot_mat(rot)

    def gap_p(o):
        return goal - _eef(o)

    def gap_r(o):
        return mat_to_axisangle(want @ quat_to_mat(o["robot0_base_to_eef_quat"]).T)

    def done(o):
        if np.linalg.norm(gap_p(o)) > tol:
            return False
        return want is None or np.linalg.norm(gap_r(o)) <= rot_tol

    def error(o):
        # One number for the stall detector; 0.1 rad of rotation counts as 1 cm.
        e = float(np.linalg.norm(gap_p(o)))
        if want is not None:
            e += 0.1 * float(np.linalg.norm(gap_r(o)))
        return e

    integ = np.zeros(3)

    def act(o):
        e = gap_p(o)
        if np.linalg.norm(e) < I_BAND:
            integ[:] = np.clip(integ + KI * 0.1 * e, -I_CLAMP, I_CLAMP)
        else:
            integ[:] *= I_LEAK
        return _action(pos_err=e, pos_int=integ,
                       rot_err=None if want is None else gap_r(o),
                       gripper=gripper, gain=gain)

    res = _drive(sim, obs, action=act, done=done, error=error, max_steps=max_steps)
    return _report(res, gap_p, gap_r if want is not None else None)


def reach(sim, target: Sequence[float], tol: float = 0.01, max_steps: int = 200, *,
          gripper: float, rot=None, rot_tol: float = 0.05) -> dict:
    """Servo the end-effector to `target` (x, y, z) in the BASE frame.

    `tol` is metres. Does not move the base -- an out-of-reach target comes back
    `gave_up`; `move_base` first. `gripper` is required and held on every step: -1
    opens, +1 closes; there is no neutral command, so say what the hand should do.

    `rot` (base-frame 3x3 matrix or xyzw quaternion, `rot_tol` radians) also servos the
    orientation: column 2 the approach direction, column 1 the closing line. The reply
    carries the final `pos_err` and `rot_err`.
    """
    obs = _begin(sim)
    if obs is None:
        return _idle(sim, BLOCKED)
    return _servo(sim, obs, target, tol, max_steps, KP, float(gripper),
                  rot=rot, rot_tol=rot_tol)


def move_eef(sim, delta: Sequence[float], tol: float = 0.01, max_steps: int = 100, *,
             gripper: float, rot=None, rot_tol: float = 0.05) -> dict:
    """Move the end-effector by `delta` from wherever it is now. The relative form of
    `reach`, for nudging. `rot` is an ABSOLUTE orientation target, exactly as in `reach`
    -- a relative rotation is too easy to compose wrong to offer."""
    obs = _begin(sim)
    if obs is None:
        return _idle(sim, BLOCKED)
    return _servo(sim, obs, _eef(obs) + np.asarray(delta, dtype=float), tol, max_steps,
                  KP, float(gripper), rot=rot, rot_tol=rot_tol)


def lift(sim, height: float = 0.15, max_steps: int = 100) -> dict:
    """Raise the end-effector by `height` metres, straight up in the base frame.

    Its own skill rather than `move_eef([0, 0, h])` because this one keeps the grip closed
    and uses a gentler gain, so it does not shed what it is holding.
    """
    obs = _begin(sim)
    if obs is None:
        return _idle(sim, BLOCKED)
    return _servo(sim, obs, _eef(obs) + [0.0, 0.0, float(height)], 0.01, max_steps,
                  KP_LIFT, _CLOSED)


def _with_gap(res: dict) -> dict:
    """Add the finger opening (`gap`, metres) to the reply."""
    q = (res.get("obs") or {}).get("robot0_gripper_qpos")
    if q is not None:
        res["gap"] = float(np.abs(np.asarray(q, dtype=float)).sum())
    return res


def _clamp(sim, command: float, max_steps: int, outcome_key: str) -> dict:
    """Drive the fingers to one extreme and hold until they stop moving."""
    obs = _begin(sim)
    if obs is None:
        return _idle(sim, BLOCKED)
    last = np.asarray(obs.get("robot0_gripper_qpos", [0.0]), dtype=float)
    res = None
    still = 0
    for _ in range(int(max_steps)):
        try:
            res = sim.step(_action(gripper=command), obs_spec=PROPRIO_ONLY)
        except RemoteError:
            return _idle(sim, BLOCKED)
        if res["episode_over"]:
            return _with_gap({**res, "outcome": outcome_key})
        qpos = np.asarray(res["obs"].get("robot0_gripper_qpos", [0.0]), dtype=float)
        still = still + 1 if float(np.abs(qpos - last).max()) < 1e-4 else 0
        last = qpos
        if still >= 3:
            return _with_gap({**res, "outcome": outcome_key})
    return _with_gap(_finish(sim, res, outcome_key))


def grasp(sim, max_steps: int = 40) -> dict:
    """Close the gripper and hold until it stops closing.

    THIS CANNOT TELL YOU WHETHER YOU CAUGHT THE OBJECT -- it reports that the fingers
    stopped moving, which an empty hand does too. The reply's `gap` is the evidence to
    weigh: an empty hand closes to ~6 mm. At L1 and L2 the verdict is a perception
    question; the usual test is to `lift` and see whether the scene changed. At L3,
    `priv_grasp["grasping"]` answers it outright.
    """
    return _clamp(sim, _CLOSED, max_steps, REACHED)


def release(sim, max_steps: int = 40) -> dict:
    """Open the gripper and hold until it stops opening. The inverse of `grasp`."""
    return _clamp(sim, _OPEN, max_steps, REACHED)


def _yaw(quat_xyzw) -> float:
    """Heading about +z, radians."""
    mat = quat_to_mat(quat_xyzw)
    return float(np.arctan2(mat[1, 0], mat[0, 0]))


def move_base(sim, dx: float = 0.0, dy: float = 0.0, dyaw: float = 0.0,
              tol: float = 0.03, max_steps: int = 300,
              hold_eef: bool = False, *, gripper: float) -> dict:
    """Drive the mobile base by (dx, dy, dyaw) in the base's own frame at call time.

    The kitchen is bigger than the arm. Yaw is radians. Base motion disturbs the arm, so
    `settle` afterwards before anything that needs precision.

    The goal is pinned in the world on the first look. The chassis slides act in a fixed
    frame that does not follow the body's yaw, so the command frame is measured from the
    base's own motion and the error re-expressed in it on every step.

    `hold_eef=True` pins the GRIPPER'S WORLD POSE while the base drives -- the mode for
    moving while holding something attached to the world; plain `move_base` drags the
    hand along and whatever it holds comes off the grasp. Slower, and bounded by the
    arm's envelope: keep each held move short (~0.15 m) and check the reply's
    `eef_drift` (metres in the world). `gripper` is required and held throughout, as
    in `reach`.
    """
    obs = _begin(sim)
    if obs is None:
        return _idle(sim, BLOCKED)
    goal_world = transform_points(base_pose_in_world(obs), [float(dx), float(dy), 0.0])
    goal_yaw = _yaw(obs["robot0_base_quat"]) + float(dyaw)
    grip = float(gripper)

    drive = {"yaw": _yaw(obs["robot0_base_quat"]), "pos": None, "cmd": None}

    def gap(o):
        """(dx, dy) still to cover in the DRIVE frame, and the yaw still to turn."""
        err = np.asarray(goal_world) - np.asarray(o["robot0_base_pos"], dtype=float)
        c, s = np.cos(drive["yaw"]), np.sin(drive["yaw"])
        turn = goal_yaw - _yaw(o["robot0_base_quat"])
        return (c * err[0] + s * err[1], -s * err[0] + c * err[1],
                float(np.arctan2(np.sin(turn), np.cos(turn))))

    done = lambda o: (np.hypot(*gap(o)[:2]) <= tol and abs(gap(o)[2]) <= 0.05)
    error = lambda o: np.hypot(*gap(o)[:2]) + abs(gap(o)[2])

    def base_cmd(o, gains, cap=None):
        pos = np.asarray(o["robot0_base_pos"], dtype=float)[:2]
        if drive["cmd"] is not None:
            d = pos - drive["pos"]
            if np.linalg.norm(d) > 4e-3:
                drive["yaw"] = float(np.arctan2(d[1], d[0])
                                     - np.arctan2(drive["cmd"][1], drive["cmd"][0]))
        err = np.array(gap(o))
        cmd = err * gains
        low = np.abs(cmd) < BASE_FLOOR
        cmd = np.where(low, np.copysign(BASE_FLOOR, cmd), cmd)
        cmd = np.where(np.abs(err) <= BASE_DEAD, 0.0, cmd)
        cmd = np.clip(cmd, -1.0, 1.0) if cap is None else np.clip(cmd, -cap, cap)
        drive["pos"] = pos
        drive["cmd"] = cmd[:2].copy() if np.linalg.norm(cmd[:2]) > 0.1 else None
        return cmd

    if not hold_eef:
        # base_mode > 0 keeps the arm's goal tracking the DESIRED pose while the base
        # moves, rather than the achieved one.
        return _drive(
            sim, obs,
            action=lambda o: _action(gripper=grip, base_mode=1.0,
                                     base=base_cmd(o, [KP_BASE, KP_BASE, KP_YAW])),
            done=done, error=error, max_steps=max_steps)

    # The hand's pose, pinned in the world at call time.
    p_w = transform_points(base_pose_in_world(obs), _eef(obs))
    r_w = quat_to_mat(obs["robot0_base_quat"]) @ quat_to_mat(
        obs["robot0_base_to_eef_quat"])

    def arm_pos_err(o):
        return world_to_base(o, p_w) - _eef(o)

    def arm_rot_err(o):
        want = quat_to_mat(o["robot0_base_quat"]).T @ r_w
        return mat_to_axisangle(want @ quat_to_mat(o["robot0_base_to_eef_quat"]).T)

    # push / re-pin phase machine.
    state = {"pushing": True, "anchor": None, "paused": 0}

    def act(o):
        pos = np.asarray(o["robot0_base_pos"], dtype=float)[:2]
        yaw = _yaw(o["robot0_base_quat"])
        if state["anchor"] is None:
            state["anchor"] = (pos, yaw)
        a_pos, a_yaw = state["anchor"]
        if state["pushing"]:
            travelled = float(np.linalg.norm(pos - a_pos))
            turned = abs(float(np.arctan2(np.sin(yaw - a_yaw), np.cos(yaw - a_yaw))))
            if (travelled >= HOLD_CHUNK or turned >= HOLD_CHUNK_YAW
                    or np.linalg.norm(arm_pos_err(o)) > HOLD_SLIP_TOL):
                state.update(pushing=False, paused=0)
        if not state["pushing"]:
            state["paused"] += 1
            if (np.linalg.norm(arm_pos_err(o)) <= HOLD_REPIN_TOL
                    or state["paused"] > HOLD_REPIN_MAX):
                state.update(pushing=True, anchor=(pos, yaw))
        base = base_cmd(o, [KP_BASE_HOLD, KP_BASE_HOLD, KP_YAW_HOLD],
                        cap=BASE_CAP_HOLD) if state["pushing"] else None
        # base_mode NEGATIVE, deliberately: the flag only selects the arm's goal-update
        # mode (the base slice applies either way), and "desired" mode integrates the
        # corrections into a goal that never re-anchors -- the hand winds up and drifts.
        return _action(pos_err=arm_pos_err(o), rot_err=arm_rot_err(o), gripper=grip,
                       base=base, base_mode=-1.0, gain=KP_HOLD)

    res = _drive(sim, obs, action=act, done=done, error=error, max_steps=max_steps)
    # Final re-pin before reporting drift.
    for _ in range(HOLD_REPIN_MAX):
        o = res.get("obs") or {}
        if "robot0_base_to_eef_pos" not in o or res.get("episode_over"):
            break
        if np.linalg.norm(arm_pos_err(o)) <= HOLD_REPIN_TOL:
            break
        try:
            res = {**sim.step(_action(pos_err=arm_pos_err(o), rot_err=arm_rot_err(o),
                                      gripper=grip, base_mode=-1.0, gain=KP_HOLD),
                              obs_spec=PROPRIO_ONLY), "outcome": res["outcome"]}
        except RemoteError:
            break
    o = res.get("obs") or {}
    if "robot0_base_to_eef_pos" in o:
        res["eef_drift"] = float(np.linalg.norm(
            transform_points(base_pose_in_world(o), _eef(o)) - p_w))
    return res


def settle(sim, tol: float = 1e-3, max_steps: int = 50, *, gripper: float) -> dict:
    """Hold still until joint velocities fall below `tol`.

    Cheap and worth it: MuJoCo does not stop the instant a command does, and a grasp
    attempted while the arm is still ringing is the most common way a good plan fails.
    """
    obs = _begin(sim)
    if obs is None:
        return _idle(sim, BLOCKED)
    grip = float(gripper)
    return _drive(
        sim, obs,
        action=lambda o: _action(gripper=grip),
        done=lambda o: float(np.abs(np.asarray(
            o.get("robot0_joint_vel", [0.0]), dtype=float)).max()) <= tol,
        max_steps=max_steps)
