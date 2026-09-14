"""Model loading + physical validity checks (Stage 1 gate).

These checks must pass before any stability number is meaningful.

Two levels of inertia checking:
  - check_inertias_valid(model): on a compiled MjModel (principal inertias).
  - check_inertias_valid_xml(path): on raw MJCF text. Needed because MuJoCo
    (correctly) refuses to compile physically-unrealizable inertias, and the
    broken input model is exactly that — the harness must still be able to
    point at the flaw without compiling.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

_MASS_EPS = 1e-6   # bodies lighter than this are connectors/stubs; nothing to check
_TRI_RTOL = 1e-9   # numerical slack on the triangle inequality

# Max surface-to-surface gap for a part to count as physically mounted to the
# structure. A joint is an ideal constraint in simulation, so without this
# check a wheel "connected" only by its hinge can hover in free space — not a
# buildable robot.
ATTACH_GAP_M = 0.05
# A task08 wheel mount must physically reach the drive-joint anchor rather
# than merely pass near the wheel envelope.
WHEEL_CENTER_ATTACH_TOL_M = 0.005


def load(path: str):
    """Load MJCF/URDF into (model, data). Raises ValueError on invalid models."""
    model = mujoco.MjModel.from_xml_path(path)
    return model, mujoco.MjData(model)


def _triangle_ok(i1: float, i2: float, i3: float) -> bool:
    slack = 1.0 - _TRI_RTOL
    return (i1 + i2 >= i3 * slack and i2 + i3 >= i1 * slack
            and i1 + i3 >= i2 * slack)


def check_inertias_valid(model) -> dict:
    """Per-body inertia report for a compiled model.

    Every body with meaningful mass must have positive-definite principal
    inertia satisfying the triangle inequality (physical realizability).
    Returns {body_name: {ok, mass, principal, positive_definite, triangle}}.
    """
    report = {}
    for i in range(1, model.nbody):
        mass = float(model.body_mass[i])
        principal = np.asarray(model.body_inertia[i], dtype=float)
        if mass < _MASS_EPS:
            report[model.body(i).name] = dict(
                ok=True, mass=mass, principal=principal.tolist(),
                positive_definite=True, triangle=True, note="massless stub")
            continue
        pd = bool(np.all(principal > 0.0))
        tri = _triangle_ok(*principal)
        report[model.body(i).name] = dict(
            ok=pd and tri, mass=mass, principal=principal.tolist(),
            positive_definite=pd, triangle=tri)
    return report


def check_inertias_valid_xml(path: str) -> dict:
    """Inertia report from raw MJCF, for models that may not compile.

    Only bodies with an explicit <inertial> element are checked — bodies
    without one get geometry-derived inertias from the compiler, which are
    valid by construction. Returns {body_name: {ok, ...}} like
    check_inertias_valid.
    """
    root = ET.parse(path).getroot()
    report = {}
    for body in root.iter("body"):
        inertial = body.find("inertial")
        if inertial is None:
            continue
        name = body.get("name", "<unnamed>")
        mass = float(inertial.get("mass", "0"))
        if inertial.get("diaginertia") is not None:
            eigs = np.array([float(v) for v in inertial.get("diaginertia").split()])
        elif inertial.get("fullinertia") is not None:
            xx, yy, zz, xy, xz, yz = (float(v) for v in inertial.get("fullinertia").split())
            eigs = np.linalg.eigvalsh(np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]))
        else:
            # mass-only inertial: compiler fills inertia from geoms
            continue
        pd = bool(np.all(eigs > 0.0))
        tri = _triangle_ok(*sorted(eigs))
        report[name] = dict(
            ok=pd and tri, mass=mass, principal=sorted(float(e) for e in eigs),
            positive_definite=pd, triangle=tri)
    return report


def _body_class(model, bid: int) -> str:
    """Classify a robot body by its nearest special-named ancestor (or self):
    'arm' (arm_* / payload), 'roller' (roller_*), 'wheel:<name>' (wheel_*),
    else 'structural' (chassis, mounts, battery, brackets, ...)."""
    while bid != 0:
        name = model.body(bid).name or ""
        if name == "payload" or name.startswith("arm_"):
            return "arm"
        if name.startswith("roller_"):
            return "roller"
        if name.startswith("wheel_"):
            return f"wheel:{name}"
        bid = int(model.body_parentid[bid])
    return "structural"


def _min_gap(model, data, geoms_a, geoms_b, dmax: float = 1.0) -> float:
    """Smallest surface-to-surface distance between two geom sets (capped at
    dmax; negative means overlap)."""
    fromto = np.zeros(6)
    best = float(dmax)
    for ga in geoms_a:
        for gb in geoms_b:
            d = mujoco.mj_geomDistance(model, data, ga, gb, dmax, fromto)
            best = min(best, float(d))
    return best

def _point_geom_gap(model, data, point, gid: int) -> float:
    """Distance from a world point to a supported structural primitive."""
    local = data.geom_xmat[gid].reshape(3, 3).T @ (
        np.asarray(point, dtype=float) - data.geom_xpos[gid])
    size = model.geom_size[gid]
    gtype = model.geom_type[gid]
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        return float(np.linalg.norm(np.maximum(np.abs(local) - size, 0.0)))
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return max(float(np.linalg.norm(local) - size[0]), 0.0)
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        axis_point = local.copy()
        axis_point[2] = np.clip(axis_point[2], -size[1], size[1])
        return max(float(np.linalg.norm(local - axis_point) - size[0]), 0.0)
    if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radial_gap = max(float(np.linalg.norm(local[:2]) - size[0]), 0.0)
        axial_gap = max(abs(float(local[2])) - size[1], 0.0)
        return float(np.hypot(radial_gap, axial_gap))
    return float("inf")


def check_structure(model, data, require_wheel_center: bool = False) -> dict:
    """Stage-1 connectivity check: the robot must be ONE buildable assembly.

    Kinematic: every body must belong to the base's kinematic tree (nothing
    floats as its own tree or hangs off the world). Geometric: the structural
    geoms — everything that is not the arm, payload, wheels or rollers — must
    form a single contiguous cluster (neighbour gaps <= ATTACH_GAP_M), and
    every wheel plus the arm root must sit within ATTACH_GAP_M of that
    cluster.

    Measured in the pose currently in `data` (callers pass the scenario-start
    pose). When ``require_wheel_center`` is true, a structural primitive must
    also pass within ``WHEEL_CENTER_ATTACH_TOL_M`` of each drive-joint anchor.
    Returns {ok, problems, wheel_gaps, wheel_center_gaps, arm_gap, n_clusters}.
    """
    problems: list[str] = []
    try:
        base_id = model.body("base").id
    except KeyError:
        return dict(ok=False, problems=["missing body 'base'"],
                    wheel_gaps={}, wheel_center_gaps={}, arm_gap=None, n_clusters=0)
    mujoco.mj_forward(model, data)
    root = int(model.body_rootid[base_id])

    detached = [model.body(i).name or f"body#{i}" for i in range(1, model.nbody)
                if int(model.body_rootid[i]) != root]
    if detached:
        problems.append("bodies outside the base kinematic tree: "
                        + ", ".join(detached))

    structural: list[int] = []
    wheels: dict[str, list[int]] = {}
    arm_root: list[int] = []
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        if bid == 0 or int(model.body_rootid[bid]) != root:
            continue
        c = _body_class(model, bid)
        if c == "structural":
            structural.append(gid)
        elif c.startswith("wheel:"):
            wheels.setdefault(c[6:], []).append(gid)
        if (model.body(bid).name or "") in ("arm_link0", "arm_adapter"):
            arm_root.append(gid)

    if not structural:
        problems.append("no structural (chassis) geoms found")

    # structural geoms must form one contiguous cluster
    n = len(structural)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _min_gap(model, data, [structural[i]], [structural[j]],
                        dmax=ATTACH_GAP_M + 0.01) <= ATTACH_GAP_M:
                parent[find(i)] = find(j)
    clusters: dict[int, list[str]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(
            model.geom(structural[i]).name or f"geom#{structural[i]}")
    if len(clusters) > 1:
        groups = " | ".join(", ".join(g) for g in clusters.values())
        problems.append(f"structural geoms split into {len(clusters)} "
                        f"disconnected groups: {groups}")

    wheel_gaps: dict[str, float] = {}
    for wname, geoms in sorted(wheels.items()):
        gap = _min_gap(model, data, geoms, structural) if structural else 1.0
        wheel_gaps[wname] = gap
        if gap > ATTACH_GAP_M:
            problems.append(f"{wname} is not mounted to the structure "
                            f"(gap {gap:.3f} m > {ATTACH_GAP_M} m)")

    center_structure = [
        gid for gid in structural
        if (model.body(model.geom_bodyid[gid]).name or "") != "battery"]
    wheel_center_gaps: dict[str, float] = {}
    if require_wheel_center:
        for wname in sorted(wheels):
            suffix = wname.removeprefix("wheel_")
            try:
                anchor = data.xanchor[model.joint(f"drive_{suffix}").id]
            except KeyError:
                wheel_center_gaps[wname] = float("inf")
                problems.append(f"{wname} is missing drive_{suffix}")
                continue
            center_gap = min(
                (_point_geom_gap(model, data, anchor, gid)
                 for gid in center_structure), default=float("inf"))
            wheel_center_gaps[wname] = center_gap
            if center_gap > WHEEL_CENTER_ATTACH_TOL_M:
                problems.append(
                    f"{wname} has no structural mount through its center "
                    f"(gap {center_gap:.3f} m > "
                    f"{WHEEL_CENTER_ATTACH_TOL_M} m)")

    arm_gap = None
    if arm_root:
        # prefer collision geoms (visual-only meshes may lack convex hulls)
        coll = [g for g in arm_root
                if model.geom_contype[g] or model.geom_conaffinity[g]]
        arm_gap = _min_gap(model, data, coll or arm_root, structural) \
            if structural else 1.0
        if arm_gap > ATTACH_GAP_M:
            problems.append(f"arm root is not mounted to the structure "
                            f"(gap {arm_gap:.3f} m > {ATTACH_GAP_M} m)")

    return dict(ok=not problems, problems=problems, wheel_gaps=wheel_gaps,
                wheel_center_gaps=wheel_center_gaps, arm_gap=arm_gap,
                n_clusters=len(clusters))


def total_mass_and_com(model, data):
    """Return (total_mass, com_xyz) for the current configuration."""
    mujoco.mj_forward(model, data)
    total = float(mujoco.mj_getTotalmass(model))
    return total, np.asarray(data.subtree_com[0], dtype=float).copy()


SETTLE_DRIFT_EPS = 0.005          # m, planar drift and height change
SETTLE_PENETRATION_EPS = 0.004   # m, deepest contact interpenetration
SETTLE_TILT_DEG_MAX = 1.0        # deg, body-z vs world-z
SETTLE_SPEED_MAX = 0.01          # m/s, base speed at the end of the hold


def settles_to_equilibrium(model, data, t: float = 2.0,
                           drift_eps: float = SETTLE_DRIFT_EPS,
                           penetration_eps: float = SETTLE_PENETRATION_EPS,
                           reset: bool = True) -> dict:
    """Hold the robot at rest for `t` seconds; report whether it stays put.

    The model must include a floor (use scene.xml or a scenario scene).
    With reset=True starts from the 'home' keyframe when present; pass
    reset=False to test the state already in `data` — the verifier does
    this so validity is measured in the SAME pose scenarios run from, not a
    submission-controlled keyframe.

    Returns {ok, drift_xy, dz, tilt_deg, end_speed, max_penetration}.
    """
    if reset:
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        else:
            mujoco.mj_resetData(model, data)
    settle_in = 0.5  # let contacts engage before measuring
    for _ in range(int(settle_in / model.opt.timestep)):
        mujoco.mj_step(model, data)
    p0 = data.qpos[:3].copy()
    max_pen = 0.0
    for _ in range(int(t / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if data.ncon:
            max_pen = max(max_pen, float(-data.contact.dist.min()))
    p1 = data.qpos[:3]
    w, qx, qy, qz = data.qpos[3:7]
    # angle between body z-axis and world z-axis
    cos_tilt = 1.0 - 2.0 * (qx * qx + qy * qy)
    tilt_deg = float(np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0))))
    drift_xy = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    dz = float(p1[2] - p0[2])
    end_speed = float(np.linalg.norm(data.qvel[:3]))
    ok = (drift_xy < drift_eps and abs(dz) < drift_eps
          and tilt_deg < SETTLE_TILT_DEG_MAX
          and end_speed < SETTLE_SPEED_MAX and max_pen < penetration_eps)
    return dict(ok=bool(ok), drift_xy=drift_xy, dz=dz, tilt_deg=tilt_deg,
                end_speed=end_speed, max_penetration=max_pen)
