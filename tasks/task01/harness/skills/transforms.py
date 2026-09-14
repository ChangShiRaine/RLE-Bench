"""Frames: the tier every other one is built on.

CONVENTIONS, which are not negotiable because the observation already fixed them:

  * quaternions are **xyzw** -- the order every `robot0_*_quat` key uses, and every
    `priv_*` quaternion at L3;
  * positions are metres;
  * a "transform" is a 4x4 homogeneous matrix mapping points FROM the frame named
    second TO the frame named first, so `T_base_cam @ p_cam == p_base`.

THE FRAME THAT MATTERS. Actions are in the ROBOT BASE frame; the cameras see the world.
`robot0_base_pos` and `robot0_base_quat` place the base in the world, so
`world_to_base` and `base_to_world` are the whole bridge. Skipping that rotation raises
nothing -- the arm simply drifts.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "quat_to_mat", "mat_to_quat", "mat_to_axisangle", "make_transform",
    "invert_transform", "decompose_transform", "transform_points",
    "base_pose_in_world", "world_to_base", "base_to_world", "normalize",
]


def quat_to_mat(quat_xyzw: Sequence[float]) -> np.ndarray:
    """Unit quaternion (x, y, z, w) -> (3, 3) rotation matrix."""
    x, y, z, w = np.asarray(quat_xyzw, dtype=float).reshape(4)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """(3, 3) rotation matrix -> unit quaternion (x, y, z, w).

    The sign is not canonicalised: `q` and `-q` are the same rotation, so compare
    rotations through matrices rather than by comparing quaternions elementwise.
    """
    m = np.asarray(mat, dtype=float)[:3, :3]
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = np.array([(m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s,
                      (m[1, 0] - m[0, 1]) * s, 0.25 / s])
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
        q = np.zeros(4)
        q[3] = (m[k, j] - m[j, k]) / s
        q[i] = 0.25 * s
        q[j] = (m[j, i] + m[i, j]) / s
        q[k] = (m[k, i] + m[i, k]) / s
    return q / np.linalg.norm(q)


def mat_to_axisangle(mat: np.ndarray) -> np.ndarray:
    """(3, 3) rotation matrix -> rotation vector (axis * angle, radians).

    The form the action's rotation channels take: the command that turns `R_cur` toward
    `R_des` is `mat_to_axisangle(R_des @ R_cur.T)`. Always the short way round.
    """
    q = mat_to_quat(mat)
    if q[3] < 0.0:
        q = -q
    n = float(np.linalg.norm(q[:3]))
    if n < 1e-9:
        return np.zeros(3)
    return q[:3] / n * (2.0 * float(np.arctan2(n, q[3])))


def make_transform(pos: Sequence[float], quat_xyzw: Sequence[float]) -> np.ndarray:
    """Position + orientation -> the 4x4 that maps that frame's points into the parent."""
    out = np.eye(4)
    out[:3, :3] = quat_to_mat(quat_xyzw)
    out[:3, 3] = np.asarray(pos, dtype=float).reshape(3)
    return out


def invert_transform(mat: np.ndarray) -> np.ndarray:
    """The inverse of a rigid 4x4, by transposing the rotation."""
    m = np.asarray(mat, dtype=float)
    rot = m[:3, :3]
    out = np.eye(4)
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ m[:3, 3]
    return out


def decompose_transform(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4x4 -> (position, quaternion xyzw)."""
    m = np.asarray(mat, dtype=float)
    return m[:3, 3].copy(), mat_to_quat(m[:3, :3])


def transform_points(mat: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 to an (N, 3) array of points. Accepts a single (3,) point too."""
    p = np.asarray(points, dtype=float)
    single = p.ndim == 1
    p = p.reshape(1, 3) if single else p.reshape(-1, 3)
    m = np.asarray(mat, dtype=float)
    out = p @ m[:3, :3].T + m[:3, 3]
    return out[0] if single else out


def base_pose_in_world(obs: dict) -> np.ndarray:
    """The 4x4 placing the robot base in the world, from the observation's own keys."""
    return make_transform(obs["robot0_base_pos"], obs["robot0_base_quat"])


def world_to_base(obs: dict, points: np.ndarray) -> np.ndarray:
    """World-frame points -> base frame, which is the frame actions are in."""
    return transform_points(invert_transform(base_pose_in_world(obs)), points)


def base_to_world(obs: dict, points: np.ndarray) -> np.ndarray:
    """The inverse of `world_to_base`."""
    return transform_points(base_pose_in_world(obs), points)


def normalize(vec: Sequence[float]) -> np.ndarray:
    """A unit vector. A zero-length input comes back unchanged rather than as NaN."""
    v = np.asarray(vec, dtype=float)
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n
