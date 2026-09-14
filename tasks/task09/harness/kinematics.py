"""Structure check: the submitted hinges and FK against the measured leader
chain, so a device with a different internal chain cannot pass by matching
the endpoint alone."""
from __future__ import annotations

import mujoco
import numpy as np

from . import spec as gspec

STRUCTURE_TOL_M = 0.002   # exact replicas match to numerical precision;
                          # a single mis-set axis errs at 3x this


def _quat_to_mat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _axis_angle(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def chain_fk(q, chain, ee):
    """Pure-numpy FK: (ee_pos[3], ee_R[3,3]) relative to the chain base."""
    q = np.asarray(q, dtype=float)
    p = np.zeros(3)
    R = np.eye(3)
    for i, entry in enumerate(chain):
        p = p + R @ np.asarray(entry["pos"], dtype=float)
        R = R @ _quat_to_mat(entry["quat"]) @ _axis_angle(entry["axis"], q[i])
    p = p + R @ np.asarray(ee["pos"], dtype=float)
    R = R @ _quat_to_mat(ee["quat"])
    return p, R


def _probe_configs() -> np.ndarray:
    """Home, the workspace-box corners along each joint, and two mixed poses."""
    home = np.asarray(gspec.LEAD_HOME)
    hw = np.asarray(gspec.WORKSPACE_HALFWIDTH)
    probes = [home]
    for i in range(gspec.N_JOINTS):
        for s in (-1.0, 1.0):
            q = home.copy()
            q[i] += s * hw[i]
            probes.append(q)
    alternating = np.where(np.arange(gspec.N_JOINTS) % 2 == 0, 1.0, -1.0)
    probes.append(home + hw * alternating)
    probes.append(home - hw * alternating * 0.5)
    return np.asarray(probes)


def _has_body(model, name: str) -> bool:
    try:
        model.body(name)
        return True
    except KeyError:
        return False


def structure_check(model, data) -> dict:
    """Verify joint names/types/order, the contract bodies and site, and every
    joint anchor/axis plus the EE against the measured leader chain."""
    problems = []
    for i, name in enumerate(gspec.JOINT_NAMES):
        try:
            jnt = model.joint(name)
        except KeyError:
            problems.append(f"missing joint {name}")
            continue
        if jnt.type != mujoco.mjtJoint.mjJNT_HINGE:
            problems.append(f"{name} is not a hinge")
        if int(jnt.qposadr[0]) != i or int(jnt.dofadr[0]) != i:
            problems.append(f"{name} is not in contract order")
    for name in ("lead_base", "lead_handle",
                 *(f"lead_link{i}" for i in range(1, gspec.N_JOINTS + 1))):
        if not _has_body(model, name):
            problems.append(f"missing body {name}")
    if int(model.njnt) != gspec.N_JOINTS:
        problems.append(f"model has {model.njnt} joints, expected "
                        f"{gspec.N_JOINTS} (extra dofs are not the device)")
    try:
        model.site(gspec.EE_SITE)
    except KeyError:
        problems.append(f"missing site {gspec.EE_SITE}")

    max_fk_err = float("nan")
    if not problems:
        base_rot = _quat_to_mat(gspec.LEAD_BASE_QUAT)
        base_pos = None
        errs = []
        for q in _probe_configs():
            for i, name in enumerate(gspec.JOINT_NAMES):
                data.qpos[model.joint(name).qposadr[0]] = q[i]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            if base_pos is None:
                base_pos = data.body("lead_base").xpos.copy()
            lead_p = data.site(gspec.EE_SITE).xpos - base_pos
            ref_p, _ = chain_fk(q, gspec.LEAD_CHAIN, gspec.LEAD_EE)
            errs.append(float(np.linalg.norm(lead_p - base_rot @ ref_p)))
            p = np.zeros(3)
            rotation = base_rot
            for i, entry in enumerate(gspec.LEAD_CHAIN):
                p += rotation @ np.asarray(entry["pos"])
                rotation = rotation @ _quat_to_mat(entry["quat"])
                joint = model.joint(gspec.JOINT_NAMES[i]).id
                errs.append(float(np.linalg.norm(data.xanchor[joint] - base_pos - p)))
                if np.linalg.norm(data.xaxis[joint] - rotation @ entry["axis"]) > 1e-5:
                    problems.append(f"{gspec.JOINT_NAMES[i]} axis differs from measured chain")
                rotation = rotation @ _axis_angle(entry["axis"], q[i])
        max_fk_err = max(errs)
        if max_fk_err > STRUCTURE_TOL_M:
            problems.append(f"EE FK deviates from the measured GELLO chain "
                            f"by up to {max_fk_err:.4f} m (> {STRUCTURE_TOL_M})")
    return dict(ok=not problems, problems=problems, max_fk_err=max_fk_err)
