"""Nominal balance analyses for the GELLO co-design task."""
from __future__ import annotations

import numpy as np

from .torques import residual_active_torque


def operator_effort(model, data, path_q, dt: float) -> dict:
    """Quasi-static effort along a nominal path."""
    path_q = np.asarray(path_q, dtype=float)
    qd = np.gradient(path_q, dt, axis=0)
    friction = np.asarray(model.dof_frictionloss, dtype=float)
    damping = np.asarray(model.dof_damping, dtype=float)
    effort = np.empty_like(path_q)
    for index, q in enumerate(path_q):
        effort[index] = (
            np.abs(residual_active_torque(model, data, q))
            + friction
            + damping * np.abs(qd[index])
        )
    peak = effort.max(axis=0)
    rms = np.sqrt(np.mean(effort ** 2, axis=0))
    return {
        "peak_per_joint": peak,
        "rms_per_joint": rms,
        "peak": float(peak.max()),
        "rms": float(rms.max()),
    }


def balance_quality(model, data, configs) -> dict:
    """Worst nominal passive residual over the supplied configurations."""
    configs = np.asarray(configs, dtype=float)
    worst = np.zeros(model.nv)
    worst_config = np.zeros(model.nv, dtype=int)
    for index, q in enumerate(configs):
        residual = np.abs(residual_active_torque(model, data, q))
        update = residual > worst
        worst[update] = residual[update]
        worst_config[update] = index
    return {
        "worst_per_joint": worst,
        "worst": float(worst.max()),
        "argmax_config": worst_config,
    }
