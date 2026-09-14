"""Circular-feature extraction from an STL (dev tool).

The GELLO parts carry no published joint frames, so we recover them from the
hardware that defines them: every servo interface is a flat face with a ring of
screw holes around the output shaft. This module finds the flat faces, walks
their interior boundary loops, fits a circle to each, and groups the circles
that share an axis. A joint shows up as a coaxial group whose ring centre is
the shaft.

Pure geometry, no MuJoCo. Units follow the file (GELLO ships millimetres).
"""
from __future__ import annotations

import struct
from collections import defaultdict

import numpy as np

WELD = 3            # decimals of a millimetre used to weld vertices
NORMAL_Q = 2        # direction-cosine quantisation for coplanar grouping
OFFSET_Q = 2        # plane-offset quantisation, mm


def load_stl(path: str) -> np.ndarray:
    """(n, 3, 3) triangle vertices. Binary STL only, which is what GELLO ships."""
    with open(path, "rb") as stream:
        raw = stream.read()
    count = struct.unpack("<I", raw[80:84])[0]
    body = np.frombuffer(raw[84:84 + 50 * count], dtype=np.uint8).reshape(count, 50)
    return body[:, 12:48].copy().view("<f4").reshape(count, 3, 3).astype(np.float64)


def _weld(tris: np.ndarray):
    flat = tris.reshape(-1, 3)
    keys = np.round(flat, WELD)
    _, first, inverse = np.unique(keys, axis=0, return_index=True,
                                  return_inverse=True)
    return flat[first], inverse.reshape(-1, 3)


def _normals(tris: np.ndarray) -> np.ndarray:
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, length, out=np.zeros_like(n), where=length > 1e-12)


def _loops(edges: set[tuple[int, int]]) -> list[list[int]]:
    """Walk undirected boundary edges into closed loops."""
    adjacency = defaultdict(list)
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    seen, out = set(), []
    for start in adjacency:
        if start in seen:
            continue
        loop, node, previous = [], start, None
        while node is not None and node not in seen:
            seen.add(node)
            loop.append(node)
            nxt = None
            for candidate in adjacency[node]:
                if candidate != previous and candidate not in seen:
                    nxt = candidate
                    break
            previous, node = node, nxt
        if len(loop) >= 6:
            out.append(loop)
    return out


def _fit_circle(points: np.ndarray, normal: np.ndarray):
    """Kasa fit in the face plane. Returns (centre3d, radius, residual)."""
    origin = points.mean(axis=0)
    basis = np.eye(3) - np.outer(normal, normal)
    u = basis[:, np.argmin(np.abs(normal))]
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    local = np.stack([(points - origin) @ u, (points - origin) @ v], axis=1)
    x, y = local[:, 0], local[:, 1]
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x ** 2 + y ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r2 = sol[2] + cx ** 2 + cy ** 2
    if r2 <= 0:
        return None
    radius = float(np.sqrt(r2))
    residual = float(np.std(np.hypot(x - cx, y - cy) - radius))
    return origin + cx * u + cy * v, radius, residual


def circles(path: str, *, r_min: float = 0.8, r_max: float = 30.0,
            max_residual: float = 0.12) -> list[dict]:
    """Every circular hole/boss boundary in the part."""
    tris = load_stl(path)
    verts, faces = _weld(tris)
    # Some STL faces are duplicated. Counting them twice hides hole edges
    # (notably all four L1 case screws) as if they were interior edges.
    _, unique = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    tris, faces = tris[unique], faces[unique]
    normals = _normals(tris)
    offsets = np.einsum("ij,ij->i", normals, tris[:, 0])

    groups = defaultdict(list)
    for i in range(len(tris)):
        if not normals[i].any():
            continue
        key = (*np.round(normals[i], NORMAL_Q), round(float(offsets[i]), OFFSET_Q))
        groups[key].append(i)

    found = []
    for key, members in groups.items():
        if len(members) < 4:
            continue
        normal = np.asarray(key[:3], dtype=float)
        normal /= np.linalg.norm(normal)
        counts = defaultdict(int)
        for i in members:
            f = faces[i]
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                counts[(min(a, b), max(a, b))] += 1
        border = {e for e, c in counts.items() if c == 1}
        for loop in _loops(border):
            fit = _fit_circle(verts[loop], normal)
            if fit is None:
                continue
            centre, radius, residual = fit
            if r_min <= radius <= r_max and residual <= max_residual:
                found.append(dict(centre=centre, axis=normal, radius=radius,
                                  residual=residual, n=len(loop)))
    return found


def rings(path: str, *, tol: float = 1.2) -> list[dict]:
    """Screw circles grouped into bolt rings: a servo interface is 3+ small
    coaxial holes whose own centres lie on a circle around the shaft."""
    small = [c for c in circles(path) if c["radius"] <= 3.5]
    used, out = set(), []
    for i, a in enumerate(small):
        if i in used:
            continue
        family = [i]
        for j, b in enumerate(small):
            if j <= i or j in used:
                continue
            if abs(abs(float(a["axis"] @ b["axis"])) - 1.0) > 1e-2:
                continue
            delta = b["centre"] - a["centre"]
            if abs(float(delta @ a["axis"])) > tol:
                continue
            if abs(b["radius"] - a["radius"]) > 0.6:
                continue
            family.append(j)
        if len(family) < 3:
            continue
        used.update(family)
        pts = np.array([small[k]["centre"] for k in family])
        out.append(dict(centre=pts.mean(axis=0), axis=a["axis"],
                        bolt_radius=float(np.linalg.norm(
                            pts - pts.mean(axis=0), axis=1).mean()),
                        hole_radius=a["radius"], holes=len(family)))
    return out
