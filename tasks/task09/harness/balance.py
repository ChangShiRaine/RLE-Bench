"""Balance metrics: hold droop, operator effort, balance quality.

The servos are stock hardware: simulations here ignore any actuators in the
model and apply the pinned servo law directly,

    tau_servo = clip(KP (q_t - q) - KD qdot + tau_ff, +-SERVO_TAU_NM)

so a submitted model can neither ship stronger servos nor weaken the test.
The design enters through the passive elements (springs, masses and
counterweights) and the feedforward tau_ff sampled from the submitted trim.
"""
from __future__ import annotations

import mujoco
import numpy as np

from . import spec as gspec
from .torques import residual_active_torque


def _disable_actuation(model) -> None:
    model.opt.disableflags = int(model.opt.disableflags) | int(
        mujoco.mjtDisableBit.mjDSBL_ACTUATION)


def _trim_at(trim, q, nv: int, cap: float) -> np.ndarray:
    """Evaluate a trim callable; clip to the servo cap and degrade to zero on
    any misbehavior."""
    if trim is None:
        return np.zeros(nv)
    try:
        out = np.asarray(trim(np.asarray(q, dtype=float).copy()),
                         dtype=float).reshape(-1)
        if out.shape[0] != nv or not np.all(np.isfinite(out)):
            return np.zeros(nv)
    except Exception:
        return np.zeros(nv)
    return np.clip(out, -cap, cap)


def hold_sim(model, data, q_target, hold_time: float = 1.5, trim=None,
             ee_site: str | None = None, tau_cap: float | None = None,
             trim_ff=None) -> dict:
    """Start at q_target, run the capped servo law, and measure where the arm
    settles: per-joint drift and (with ee_site) EE droop.

    trim_ff is a constant feedforward vector, the frozen sample the scorer
    uses; trim is a per-step callable, kept for checking that contract.
    Both are clipped to the servo cap."""
    cap = gspec.SERVO_TAU_NM if tau_cap is None else tau_cap
    _disable_actuation(model)
    q_target = np.asarray(q_target, dtype=float)
    ff = (np.zeros(model.nv) if trim_ff is None
          else np.clip(np.asarray(trim_ff, dtype=float), -cap, cap))
    mujoco.mj_resetData(model, data)
    data.qpos[:] = q_target
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    ee0 = data.site(ee_site).xpos.copy() if ee_site else None
    for _ in range(int(hold_time / model.opt.timestep)):
        tau = (gspec.SERVO_KP * (q_target - data.qpos)
               - gspec.SERVO_KD * data.qvel
               + ff + _trim_at(trim, data.qpos, model.nv, cap))
        data.qfrc_applied[:] = np.clip(tau, -cap, cap)
        mujoco.mj_step(model, data)
    drift = np.abs(np.asarray(data.qpos) - q_target)
    out = dict(joint_drift=drift, max_joint_drift=float(drift.max()),
               end_speed=float(np.max(np.abs(data.qvel))))
    if ee_site:
        out["ee_droop"] = float(np.linalg.norm(
            data.site(ee_site).xpos - ee0))
    return out


def operator_effort(model, data, path_q, dt: float, ff=None) -> dict:
    """Quasi-static torque a human must apply along path_q (sampled every dt):
        |tau_a(q) - ff| + friction + damping |qdot|
    per joint; ff is an optional (n, nv) frozen feedforward, clipped to the
    servo cap. Returns peak and RMS (worst joint and per-joint arrays)."""
    path_q = np.asarray(path_q, dtype=float)
    qd = np.gradient(path_q, dt, axis=0)
    friction = np.asarray(model.dof_frictionloss, dtype=float)
    damping = np.asarray(model.dof_damping, dtype=float)
    cap = gspec.SERVO_TAU_NM
    eff = np.empty_like(path_q)
    for t, q in enumerate(path_q):
        tau_a = residual_active_torque(model, data, q)
        if ff is not None:
            tau_a = tau_a - np.clip(np.asarray(ff[t], dtype=float), -cap, cap)
        eff[t] = np.abs(tau_a) + friction + damping * np.abs(qd[t])
    peak = eff.max(axis=0)
    rms = np.sqrt(np.mean(eff ** 2, axis=0))
    return dict(peak_per_joint=peak, rms_per_joint=rms,
                peak=float(peak.max()), rms=float(rms.max()))


def balance_quality(model, data, configs) -> dict:
    """Worst-case passive residual |tau_a(q)| over the workspace sample."""
    configs = np.asarray(configs, dtype=float)
    worst = np.zeros(model.nv)
    worst_cfg = np.zeros(model.nv, dtype=int)
    for idx, q in enumerate(configs):
        r = np.abs(residual_active_torque(model, data, q))
        upd = r > worst
        worst[upd] = r[upd]
        worst_cfg[upd] = idx
    return dict(worst_per_joint=worst, worst=float(worst.max()),
                argmax_config=worst_cfg)
