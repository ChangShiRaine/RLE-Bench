"""Support polygon from wheel-ground contact points.

The support polygon is the convex hull of the wheel-ground contact points.
All stability metrics are defined relative to it.

Two views of the same hull:
  - support_polygon():    2D gravity-projection (world x-y), for SSM/ZMP.
  - support_polygon_3d(): same vertices with world z kept, for FASM, whose
    tip-over axes must follow the terrain plane on slopes.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

_FORCE_EPS = 0.5  # N — a contact must carry at least this normal force


def _roller_geom_ids(model) -> set[int]:
    ids = set()
    for i in range(model.ngeom):
        name = model.geom(i).name
        if name and name.startswith("rollerg_"):
            ids.add(i)
    return ids


def wheel_contacts(model, data, force_eps: float = _FORCE_EPS) -> np.ndarray:
    """(N, 3) world positions of load-bearing wheel(roller)-ground contacts."""
    rollers = _roller_geom_ids(model)
    pts = []
    force = np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        if con.geom1 in rollers or con.geom2 in rollers:
            mujoco.mj_contactForce(model, data, i, force)
            if force[0] > force_eps:  # normal component, contact frame
                pts.append(con.pos.copy())
    return np.array(pts).reshape(-1, 3)


def contact_normal_forces(model, data) -> dict:
    """Per-wheel total normal force, for liftoff detection.

    Returns {wheel_name: newtons} for wheels named by the rollerg_<wheel>_<k>
    convention. Wheels with no active contacts report 0.0.
    """
    totals: dict[str, float] = {}
    force = np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        for gid in (con.geom1, con.geom2):
            name = model.geom(gid).name or ""
            if name.startswith("rollerg_"):
                wheel = name.split("_")[1]
                mujoco.mj_contactForce(model, data, i, force)
                totals[wheel] = totals.get(wheel, 0.0) + max(force[0], 0.0)
    return totals


def support_polygon(contacts: np.ndarray) -> np.ndarray:
    """Convex hull (CCW) of contacts projected along gravity to world x-y.

    Returns (M, 2). Degenerate inputs (< 3 distinct points) are returned
    as-is with shape (N, 2) — metrics must handle them (a line or point
    cannot statically support the robot).
    """
    return support_polygon_3d(contacts)[:, :2]


def support_polygon_3d(contacts: np.ndarray) -> np.ndarray:
    """Convex hull vertices in 3D, ordered CCW as seen from above (+z).

    The hull is computed on the x-y projection (contacts from a wheeled base
    are near-coplanar); world z is preserved so FASM tip-over axes follow the
    terrain plane.
    """
    contacts = np.asarray(contacts, dtype=float).reshape(-1, 3)
    if len(contacts) < 3:
        return contacts
    xy = contacts[:, :2]
    # collinearity guard: rank of centered points
    centered = xy - xy.mean(axis=0)
    if np.linalg.matrix_rank(centered, tol=1e-9) < 2:
        # all points on a line: return the two extremes
        d = centered @ np.linalg.svd(centered)[2][0]
        return contacts[[int(d.argmin()), int(d.argmax())]]
    hull = ConvexHull(xy)  # 2D hull vertices are CCW
    return contacts[hull.vertices]
