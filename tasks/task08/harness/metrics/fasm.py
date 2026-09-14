"""Force-Angle Stability Measure (FASM).

Papadopoulos & Rey (1996), "A new measure of tipover stability margin for
mobile manipulators." PRIMARY dynamic criterion: works on slopes / uneven
contact where flat-ground ZMP assumptions break.

For each tip-over axis (support-polygon edge, CCW seen from above) a_i:
  - f   = net gravito-inertial force acting at the CoM, world frame
  - f_i = component of f perpendicular to a_i
  - l_i = component of (axis point - CoM) perpendicular to a_i
          (points from the CoM toward the axis; |l_i| = CoM-axis distance)
  - theta_i = angle between f_i and l_i, signed positive while the line of
    action of f passes INSIDE axis i, negative once it passes outside
  - margin_i = theta_i * |f_i| * |l_i|
FASM = min_i margin_i. FASM <= 0 => tipping about that axis.

This is a VARIANT of the basic force-only P&R measure: the margin is scaled
by the axis-perpendicular force component and the CoM-axis distance
(theta * |f_i| * |l_i|) rather than P&R's bare angle. Sign and zero
crossings are identical to the published measure; the magnitude scaling is
internally consistent because every pass/saturation threshold in the
benchmark is calibrated with this same formula. Net-moment contribution is
omitted: the battery introduces disturbances via terrain and motion, not
applied torques, so the force term dominates.
"""
from __future__ import annotations

import mujoco
import numpy as np


def net_com_force(model, data) -> np.ndarray:
    """Net gravito-inertial force at the CoM, world frame.

    By Newton, m*a_com = sum(F_contact) + m*g, so the gravito-inertial force
    the robot exerts against its supports is
        f_r = m*(g - a_com) = -sum(F_contact).
    Summing contact forces is instantaneous (no velocity history needed) and
    exact at every step, including free flight (no contacts -> f_r = 0: a
    falling robot exerts nothing to tip itself about).
    """
    total = np.zeros(3)
    force = np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        mujoco.mj_contactForce(model, data, i, force)
        # contact frame rows of con.frame: [normal; tangent1; tangent2]
        frame = con.frame.reshape(3, 3)
        world_f = frame.T @ force[:3]
        # mj_contactForce reports the force acting ON geom1 side's body?
        # Convention check lives in tests: static robot must yield ~(0,0,-mg).
        # MuJoCo's contact normal points from geom1 into geom2, and the
        # returned force acts on geom2; contacts here are (floor, robot) or
        # (robot, floor) depending on creation order, so resolve by which side
        # is attached to the world body.
        b1 = int(model.geom_bodyid[con.geom1])
        w1 = int(model.body_rootid[b1]) == 0  # geom1 side is fixed to the world
        # force on the non-world side:
        total += world_f if w1 else -world_f
    return -total


def force_angle_stability(com: np.ndarray, polygon3d: np.ndarray,
                          net_force: np.ndarray) -> float:
    """FASM = min over tip-over axes of the signed force-angle margin.

    polygon3d: (M, 3) support polygon vertices, CCW seen from above.
    M == 2 is treated as a single tip-over line (both directions checked);
    M < 2 (or a vanishing force) cannot be stable: -inf.
    """
    poly = np.asarray(polygon3d, dtype=float).reshape(-1, 3)
    com = np.asarray(com, dtype=float)
    f = np.asarray(net_force, dtype=float)
    fnorm = np.linalg.norm(f)
    if len(poly) < 2 or fnorm < 1e-9:
        return float("-inf")

    margins = []
    m = len(poly)
    n_axes = m if m > 2 else 2  # a segment: both tip directions
    for i in range(n_axes):
        p1, p2 = poly[i % m], poly[(i + 1) % m]
        a = p2 - p1
        anorm = np.linalg.norm(a)
        if anorm < 1e-12:
            continue
        a_hat = a / anorm
        perp = np.eye(3) - np.outer(a_hat, a_hat)
        f_i = perp @ f
        l_i = perp @ (p1 - com)
        fn, ln = np.linalg.norm(f_i), np.linalg.norm(l_i)
        if fn < 1e-9 or ln < 1e-12:
            continue  # force parallel to this axis: no tipping about it
        f_hat, l_hat = f_i / fn, l_i / ln
        cos_t = float(np.clip(f_hat @ l_hat, -1.0, 1.0))
        # positive while the force's line of action passes inside the axis
        sign = 1.0 if float(np.cross(l_hat, f_hat) @ a_hat) >= 0.0 else -1.0
        theta = sign * float(np.arccos(cos_t))
        margins.append(theta * fn * ln)
    return float(min(margins)) if margins else float("-inf")
