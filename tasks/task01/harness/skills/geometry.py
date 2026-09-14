"""Point clouds to targets: what is a surface, what is an object, where to grasp it.

A kitchen cloud is mostly counter. `remove_support_plane` takes the dominant horizontal
surface out, `cluster_points` separates what is left into things, `oriented_bbox` turns
one of those into a centre and a size, and `select_top_down_grasp` picks from grasp
candidates. That order is the usual pipeline; none of it is required.

EVERYTHING HERE IS DETERMINISTIC. No RANSAC, no sampling: the plane is found by
histogram and the clusters by a fixed-radius sweep, so the same cloud gives the same
answer on every run. Frames are whatever frame you passed in -- these functions do not
know or care, except that `remove_support_plane` and `oriented_bbox` assume +z is up.
"""

from __future__ import annotations

import numpy as np

from .transforms import mat_to_quat, normalize

__all__ = ["remove_support_plane", "cluster_points", "oriented_bbox",
           "select_top_down_grasp"]


def remove_support_plane(points: np.ndarray, thickness: float = 0.02,
                         bin_size: float = 0.01,
                         min_fraction: float = 0.15) -> np.ndarray:
    """Drop the dominant horizontal surface, and everything below it.

    Args:
        points: (N, 3). +z up.
        thickness: metres of slab counted as "on the surface".
        bin_size: height histogram resolution, metres.
        min_fraction: a band holding less than this fraction of the cloud is not a
            surface, and nothing is removed. Stops a cloud with no counter in it from
            losing its densest object instead.

    Returns:
        (M, 3), the points above the surface. The input, unchanged, when no band
        qualifies.
    """
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(p) == 0:
        return p
    z = p[:, 2]
    lo, hi = float(z.min()), float(z.max())
    if hi - lo < bin_size:
        return p
    counts, edges = np.histogram(z, bins=max(2, int(np.ceil((hi - lo) / bin_size))))
    peak = int(np.argmax(counts))
    if counts[peak] < min_fraction * len(p):
        return p
    surface = (edges[peak] + edges[peak + 1]) / 2.0
    return p[z > surface + thickness]


def cluster_points(points: np.ndarray, radius: float = 0.02,
                   min_size: int = 20) -> list[np.ndarray]:
    """Single-link euclidean clustering: neighbours within `radius` join up.

    Returns a list of (M, 3) arrays, largest first, with clusters smaller than
    `min_size` dropped. An empty list means nothing survived, which is a real answer.

    Cost is the KD-tree pair query, so thin the cloud (stride the mask, or render
    smaller) before calling this on a full-resolution frame.
    """
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(p) == 0:
        return []
    from scipy.spatial import cKDTree

    parent = np.arange(len(p))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in cKDTree(p).query_pairs(radius, output_type="ndarray"):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra

    labels = np.array([find(i) for i in range(len(p))])
    groups = [p[labels == root] for root in np.unique(labels)]
    groups = [g for g in groups if len(g) >= min_size]
    return sorted(groups, key=len, reverse=True)


def oriented_bbox(points: np.ndarray) -> dict:
    """A gravity-aligned box around a cloud.

    Yaw comes from the principal axis of the points projected onto the ground plane;
    roll and pitch are fixed at zero. That is the right box for a kitchen: objects sit on
    surfaces, and a fully free 3-DoF fit on a cloud that only sees one side of an object
    tips the box toward the visible face.

    Returns:
        `{"center": (3,), "extent": (3,) FULL side lengths, "quat": (4,) xyzw}`.
        `transforms.quat_to_mat(box["quat"])` has the box axes as its columns.
    """
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(p) == 0:
        raise ValueError("oriented_bbox needs at least one point")
    if len(p) < 3:
        axes = np.eye(3)
    else:
        flat = p[:, :2] - p[:, :2].mean(axis=0)
        # Eigenvectors of a 2x2 symmetric matrix; eigh orders them ascending, so the
        # principal axis is the last column.
        _, vecs = np.linalg.eigh(flat.T @ flat)
        major = normalize(np.array([vecs[0, -1], vecs[1, -1], 0.0]))
        axes = np.column_stack([major, np.cross([0.0, 0.0, 1.0], major),
                                [0.0, 0.0, 1.0]])
    local = p @ axes
    lo, hi = local.min(axis=0), local.max(axis=0)
    return {"center": axes @ ((lo + hi) / 2.0), "extent": hi - lo,
            "quat": mat_to_quat(axes)}


def select_top_down_grasp(grasps: np.ndarray, scores: np.ndarray | None = None,
                          max_tilt_deg: float = 45.0,
                          approach_axis: int = 2) -> int | None:
    """Pick the most top-down grasp from a candidate set, breaking ties by score.

    Args:
        grasps: (K, 4, 4) grasp poses, in whatever frame you have them in. `+z` of each
            pose is taken as the approach direction, which is the Contact-GraspNet
            convention; `approach_axis` changes that if yours differs.
        scores: (K,) confidences, or None to rank on tilt alone.
        max_tilt_deg: reject candidates whose approach is further than this from
            straight down. `None` for all of them.

    Returns:
        The index of the chosen grasp, or None when nothing survives the tilt filter --
        which is the useful answer: it means approaching this object from above will not
        work, not that the object is not there.

    `+z` of the FRAME YOU PASSED must be up for "down" to mean anything, so convert
    camera-frame grasps to the base or world frame first.
    """
    g = np.asarray(grasps, dtype=float)
    if g.ndim != 3 or g.shape[1:] != (4, 4):
        raise ValueError(f"grasps must be (K, 4, 4), got {g.shape}")
    if len(g) == 0:
        return None
    approach = g[:, :3, approach_axis]
    approach = approach / np.maximum(np.linalg.norm(approach, axis=1, keepdims=True), 1e-12)
    # cos of the angle from straight down; 1.0 is perfectly top-down.
    alignment = -approach[:, 2]
    keep = np.ones(len(g), dtype=bool)
    if max_tilt_deg is not None:
        keep = alignment >= np.cos(np.radians(max_tilt_deg))
    if not keep.any():
        return None
    rank = alignment if scores is None else alignment * np.asarray(scores, dtype=float)
    rank = np.where(keep, rank, -np.inf)
    return int(np.argmax(rank))
