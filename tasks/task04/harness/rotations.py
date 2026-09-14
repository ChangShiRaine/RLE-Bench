"""Vectorized quaternion helpers, (..., 4) in MuJoCo's wxyz order.

Backend-generic: the evaluator calls these with numpy arrays and the trainer with
torch tensors on the GPU. One implementation rather than two keeps the trainer and
the evaluator from drifting apart on a convention — the failure mode this whole
harness is most exposed to. torch is imported lazily so the verifier image, which
has no torch, never needs it.

MuJoCo ships equivalents (mju_mulQuat, mju_quat2Vel, ...) but they are scalar C
calls; every caller here works on whole trajectories or whole env batches.
"""
from __future__ import annotations

import numpy as np


def _xp(x):
    if type(x).__module__.startswith("torch"):
        import torch

        return torch
    return np


def _stack(xp, parts):
    return xp.stack(parts, axis=-1)


def _cross(a, b):
    return _stack(_xp(a), [
        a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
    ])


def quat_mul(a, b):
    xp = _xp(a)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return _stack(xp, [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q):
    return _stack(_xp(q), [q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]])


def quat_rotate(q, v):
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * _cross(u, v)
    return v + w * t + _cross(u, t)


def quat_rotate_inverse(q, v):
    return quat_rotate(quat_conj(q), v)


def axis_angle(q):
    xp = _xp(q)
    q = xp.where(q[..., :1] < 0.0, -q, q)
    v = q[..., 1:]
    n = xp.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * xp.arctan2(n, q[..., :1])
    small = n < 1e-8
    return xp.where(small, 2.0 * v, v * (angle / xp.where(small, n + 1.0, n)))


def quat_error(a, b):
    return _xp(a).linalg.norm(axis_angle(quat_mul(a, quat_conj(b))), axis=-1)


def yaw_quat(q):
    xp = _xp(q)
    w, z = q[..., 0], q[..., 3]
    n = xp.sqrt(w * w + z * z)
    zero = w * 0.0
    return _stack(xp, [w / n, zero, zero, z / n])


def slerp(a, b, t):
    xp = _xp(a)
    d = (a * b).sum(-1)[..., None]
    b = xp.where(d < 0.0, -b, b)
    d = abs(d)
    theta = xp.arccos(xp.clip(d, -1.0, 1.0))
    s = xp.sin(theta)

    close = s < 1e-6
    safe = xp.where(close, s + 1.0, s)
    wa = xp.where(close, 1.0 - t, xp.sin((1.0 - t) * theta) / safe)
    wb = xp.where(close, t, xp.sin(t * theta) / safe)
    q = wa * a + wb * b
    return q / xp.linalg.norm(q, axis=-1, keepdims=True)


def quat_to_mat(q):
    xp = _xp(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = _stack(xp, [
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ])
    return m.reshape(tuple(q.shape[:-1]) + (3, 3))


def ori6(q):
    return quat_to_mat(q)[..., :, :2].reshape(tuple(q.shape[:-1]) + (6,))
