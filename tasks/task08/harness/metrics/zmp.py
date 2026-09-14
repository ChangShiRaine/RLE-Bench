"""Zero-Moment Point (ZMP). Flat-ground cross-check only (NOT primary).

Reported alongside FASM on flat tests for familiarity/sanity. Do not use on
slopes/uneven terrain — this formulation assumes a flat, level plane.

Computed as the center of pressure of the ground contacts, which coincides
with the ZMP on flat ground. In-polygon <=> no tipping (flat ground).
"""
from __future__ import annotations

import mujoco
import numpy as np


def zmp(model, data) -> np.ndarray:
    """(x, y) center of pressure over all robot-ground contacts.

    Returns [nan, nan] when there is no supporting contact (airborne).
    Static equilibrium check: ZMP == CoM ground projection.
    """
    force = np.zeros(6)
    weighted = np.zeros(2)
    total_n = 0.0
    for i in range(data.ncon):
        con = data.contact[i]
        mujoco.mj_contactForce(model, data, i, force)
        frame = con.frame.reshape(3, 3)
        # vertical component of the world-frame contact force magnitude
        world_f = frame.T @ force[:3]
        fz = abs(world_f[2])
        if fz < 1e-9:
            continue
        weighted += fz * con.pos[:2]
        total_n += fz
    if total_n < 1e-9:
        return np.array([np.nan, np.nan])
    return weighted / total_n
