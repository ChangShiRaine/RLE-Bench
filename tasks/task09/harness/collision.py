"""Sampled surface interference, without convexification or contact dynamics.

FCL tests original mesh triangles and analytic primitive shapes. Containment
and continuous collision between sampled configurations are not tested.
"""
from __future__ import annotations

from itertools import combinations

import fcl
import mujoco
import numpy as np

from . import codesign_variant_config as ccfg
from . import spec as gspec
from .scenarios import GelloEnvelope, teleop_path

PATH_DT = 0.1


def _shape(model, gid):
    size = model.geom_size[gid]
    kind = model.geom_type[gid]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[gid]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        vertices = np.asarray(model.mesh_vert[va:va + vn], dtype=np.float64)
        faces = np.asarray(model.mesh_face[fa:fa + fn], dtype=np.int32)
        shape = fcl.BVHModel()
        shape.beginModel(vn, fn)
        shape.addSubModel(vertices, faces)
        shape.endModel()
        low, high = vertices.min(axis=0), vertices.max(axis=0)
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        shape = fcl.Box(*(2 * size))
        low, high = -size, size
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        shape = fcl.Sphere(size[0])
        high = np.full(3, size[0])
        low = -high
    elif kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        capsule = kind == mujoco.mjtGeom.mjGEOM_CAPSULE
        factory = fcl.Capsule if capsule else fcl.Cylinder
        shape = factory(size[0], 2 * size[1])
        high = np.array([size[0], size[0], size[1] + (size[0] if capsule else 0)])
        low = -high
    else:
        raise ValueError(f"unsupported collision geom: {model.geom(gid).name or gid}")
    return fcl.CollisionObject(shape), (low + high) / 2, (high - low) / 2


def motion_clearance(model, paths=None) -> dict:
    """A path passes only if every sampled pose is clear of added-part contact.

    Exclude supplied/supplied and same-welded-body pairs. No exemption is
    inferred from names, XML contact exclusions, or visibility.
    """
    if paths is None:
        paths = [(seed, teleop_path(seed, GelloEnvelope(), dt=PATH_DT))
                 for seed in ccfg.HIDDEN_SEEDS]
    paths = list(paths)
    report = dict(ok=False, n_checks=len(paths), n_failed=len(paths),
                  trajectories=[], problems=[])
    try:
        known = (set(gspec.LEAD_GEOMETRY["supplied_geoms"])
                 | set(gspec.LEAD_GEOMETRY["stock_geoms"]))
        names = {model.geom(i).name for i in range(model.ngeom)}
        # The stock motor box is an inertia proxy for its visible mesh.
        geoms = [i for i in range(model.ngeom)
                 if not (model.geom(i).name in gspec.LEAD_GEOMETRY["stock_geoms"]
                         and model.geom(i).name.startswith("servo")
                         and "visual_" + model.geom(i).name in names)]
        added = {i for i in geoms if model.geom(i).name not in known}
        pairs = [(a, b) for a, b in combinations(geoms, 2)
                 if (a in added or b in added)
                 and model.body_weldid[model.geom_bodyid[a]]
                 != model.body_weldid[model.geom_bodyid[b]]]
        shapes = {i: _shape(model, i) for i in sorted({i for p in pairs for i in p})}
        data = mujoco.MjData(model)
        request = fcl.CollisionRequest(num_max_contacts=1)
        report["n_pairs"] = len(pairs)
        for seed, path in paths:
            collisions = {}
            for step, q in enumerate(path):
                q = np.asarray(q, dtype=float)
                if q.shape != (model.nq,) or not np.all(np.isfinite(q)):
                    raise ValueError("invalid collision trajectory configuration")
                data.qpos[:] = q
                mujoco.mj_kinematics(model, data)
                bounds = {}
                for i, (obj, center, half) in shapes.items():
                    rotation = data.geom_xmat[i].reshape(3, 3)
                    position = data.geom_xpos[i]
                    obj.setTransform(fcl.Transform(rotation, position))
                    c, h = position + rotation @ center, np.abs(rotation) @ half
                    bounds[i] = c - h, c + h
                for a, b in pairs:
                    if (a, b) in collisions:
                        continue
                    la, ha = bounds[a]
                    lb, hb = bounds[b]
                    if np.any(ha < lb) or np.any(hb < la):
                        continue
                    result = fcl.CollisionResult()
                    if fcl.collide(shapes[a][0], shapes[b][0], request, result):
                        collisions[a, b] = dict(
                            geoms=[model.geom(i).name or f"geom#{i}" for i in (a, b)],
                            geom_ids=[a, b], sample=step, qpos=q.tolist())
            report["trajectories"].append(dict(
                seed=int(seed), samples=len(path), ok=not collisions,
                collisions=list(collisions.values())))
        report["n_failed"] = sum(not t["ok"] for t in report["trajectories"])
        report["ok"] = report["n_failed"] == 0
    except Exception as exc:
        report["problems"].append(f"motion clearance check failed: {exc}")
    return report
