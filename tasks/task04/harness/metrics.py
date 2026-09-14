"""Pure tracking metrics — no MuJoCo, no I/O, fully unit-testable.

Re-anchoring is the subtle part, and the trainer shares it. Body errors are
measured after moving the reference onto the robot's current xy position and
yaw, so a policy is scored on reproducing the MOTION rather than on standing in
a particular spot. That would hide global drift, so drift is reported separately
as the anchor error and never folded into the body error.
"""
from __future__ import annotations

import numpy as np

from . import rotations as R

_GRAVITY = np.array([0.0, 0.0, -1.0])


def reanchor(body_pos, body_quat, anchor_pos, anchor_quat,
             robot_anchor_pos, robot_anchor_quat):
    xp = R._xp(body_pos)
    delta_pos = xp.stack([robot_anchor_pos[..., 0], robot_anchor_pos[..., 1],
                          anchor_pos[..., 2]], axis=-1)
    delta_ori = R.yaw_quat(R.quat_mul(robot_anchor_quat, R.quat_conj(anchor_quat)))
    pos = delta_pos[..., None, :] + R.quat_rotate(
        delta_ori[..., None, :], body_pos - anchor_pos[..., None, :])
    return pos, R.quat_mul(delta_ori[..., None, :], body_quat)


def body_position_error(ref_pos, robot_pos) -> np.ndarray:
    return np.linalg.norm(ref_pos - robot_pos, axis=-1).mean(axis=-1)


def body_orientation_error(ref_quat, robot_quat) -> np.ndarray:
    return R.quat_error(ref_quat, robot_quat).mean(axis=-1)


def anchor_position_error(ref_pos, robot_pos) -> np.ndarray:
    return np.linalg.norm(ref_pos - robot_pos, axis=-1)


def anchor_orientation_error(ref_quat, robot_quat) -> np.ndarray:
    return R.quat_error(ref_quat, robot_quat)


def joint_error(ref, robot) -> np.ndarray:
    return np.sqrt(np.square(ref - robot).mean(axis=-1))


def gravity_tilt(quat) -> np.ndarray:
    return R.quat_rotate_inverse(quat, np.broadcast_to(_GRAVITY, quat.shape[:-1] + (3,)))[..., 2]


def action_jerk(actions: np.ndarray) -> float:
    if len(actions) < 3:
        return 0.0
    return float(np.abs(np.diff(actions, n=2, axis=0)).mean())


TRACK_STD = 0.3

MULTI_STDS = (0.3, 0.1, 0.05)


def tracking_quality(error: np.ndarray, std: float = TRACK_STD) -> np.ndarray:
    return np.exp(-np.square(error) / std**2)


def tracking_quality_multi(error: np.ndarray,
                           stds: tuple = MULTI_STDS) -> np.ndarray:
    error = np.asarray(error)
    return np.mean([tracking_quality(error, std) for std in stds], axis=0)
