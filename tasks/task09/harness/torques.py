"""Gravity-compensation torque decomposition.

All three quantities are pure functions of the model and configuration,
computed from the sim's inverse dynamics. Sign convention: every torque is
the generalized force that must be applied at the joint to hold equilibrium,
so at rest tau_g = tau_s + tau_a (gravity is carried by springs + servos).
"""
from __future__ import annotations

import mujoco
import numpy as np


def gravity_torque(model, data, qpos=None) -> np.ndarray:
    """tau_g(q): generalized gravity load, per dof (length nv).

    MuJoCo's qfrc_bias = C(q, qdot) + g(q); with velocities zeroed it is
    exactly the gravity term. This is the torque the joint must supply to
    hold the pose with no springs and no motion.
    """
    if qpos is not None:
        data.qpos[:] = np.asarray(qpos, dtype=float)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    return np.asarray(data.qfrc_bias, dtype=float).copy()


def spring_torque(model, qpos) -> np.ndarray:
    """tau_s(q) = -k (q - q0) per hinge dof (length nv), from the model's
    joint stiffness/springref. This is the torque the springs APPLY."""
    qpos = np.asarray(qpos, dtype=float)
    tau = np.zeros(model.nv)
    for j in range(model.njnt):
        k = float(model.jnt_stiffness[j])
        if k == 0.0:
            continue   # MuJoCo applies negative stiffness too; match the sim
        qadr = int(model.jnt_qposadr[j])
        dadr = int(model.jnt_dofadr[j])
        tau[dadr] = -k * (qpos[qadr] - float(model.qpos_spring[qadr]))
    return tau


def residual_active_torque(model, data, qpos=None) -> np.ndarray:
    """tau_a(q) = tau_g(q) - tau_s(q): what the servo must supply to hold q.

    The core design constraint is |tau_a(q)| <= servo rating (with margin)
    over the whole required workspace.
    """
    if qpos is None:
        qpos = np.asarray(data.qpos, dtype=float).copy()
    return gravity_torque(model, data, qpos) - spring_torque(model, qpos)
