"""T-block footprint geometry used by the reference estimators (verifier-side).

The block frame matches tshape.stl (scaled to meters): crossbar centered at
the origin along +x, stem extending toward -y, extrusion height 0.04.
"""
from __future__ import annotations

import numpy as np

from . import spec

# (cx, cy, hx, hy) of the two footprint rectangles in the block frame
BAR = (spec.BLOCK_BAR_CENTER[0], spec.BLOCK_BAR_CENTER[1],
       spec.BLOCK_BAR_HALF[0], spec.BLOCK_BAR_HALF[1])
STEM = (spec.BLOCK_STEM_CENTER[0], spec.BLOCK_STEM_CENTER[1],
        spec.BLOCK_STEM_HALF[0], spec.BLOCK_STEM_HALF[1])


def area_centroid() -> np.ndarray:
    """Centroid of the T footprint in the block frame."""
    a_bar = 4 * BAR[2] * BAR[3]
    a_stem = 4 * STEM[2] * STEM[3]
    c = (np.array([BAR[0], BAR[1]]) * a_bar
         + np.array([STEM[0], STEM[1]]) * a_stem) / (a_bar + a_stem)
    return c


def to_block_frame(pts_xy: np.ndarray, pose) -> np.ndarray:
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    d = pts_xy - np.array([x, y])
    return d @ np.array([[c, s], [-s, c]]).T


def _rect_outside_dist(p: np.ndarray, rect) -> np.ndarray:
    cx, cy, hx, hy = rect
    dx = np.maximum(np.abs(p[:, 0] - cx) - hx, 0.0)
    dy = np.maximum(np.abs(p[:, 1] - cy) - hy, 0.0)
    return np.hypot(dx, dy)


def outside_distance(pts_xy: np.ndarray, pose) -> np.ndarray:
    """Distance from each world-frame point to the T footprint (0 inside)."""
    p = to_block_frame(np.asarray(pts_xy, dtype=float), pose)
    return np.minimum(_rect_outside_dist(p, BAR), _rect_outside_dist(p, STEM))


def contains(pts_xy: np.ndarray, pose) -> np.ndarray:
    return outside_distance(pts_xy, pose) <= 0.0


def template_points(spacing: float = 0.012) -> np.ndarray:
    """Grid sample of the footprint interior, block frame (N, 2)."""
    pts = []
    for cx, cy, hx, hy in (BAR, STEM):
        xs = np.arange(cx - hx + spacing / 2, cx + hx, spacing)
        ys = np.arange(cy - hy + spacing / 2, cy + hy, spacing)
        g = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        pts.append(g)
    return np.concatenate(pts, axis=0)


def transform(pts_block: np.ndarray, pose) -> np.ndarray:
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    return pts_block @ R.T + np.array([x, y])


def corners3d() -> np.ndarray:
    """All 16 corners of the two collision prisms, block frame (16, 3)."""
    out = []
    for (cx, cy, hx, hy) in (BAR, STEM):
        for sx in (-1, 1):
            for sy in (-1, 1):
                for z in (0.0, spec.BLOCK_HEIGHT):
                    out.append([cx + sx * hx, cy + sy * hy, z])
    return np.asarray(out)


def surface_samples(spacing: float = 0.02) -> np.ndarray:
    """3D samples on the block's top face and side walls, block frame (N, 3)."""
    pts = []
    top = template_points(spacing)
    pts.append(np.column_stack([top, np.full(len(top), spec.BLOCK_HEIGHT)]))
    zs = np.arange(spacing / 2, spec.BLOCK_HEIGHT, spacing / 2)
    for (cx, cy, hx, hy) in (BAR, STEM):
        xs = np.arange(cx - hx, cx + hx + 1e-9, spacing)
        ys = np.arange(cy - hy, cy + hy + 1e-9, spacing)
        for z in zs:
            for yv in (cy - hy, cy + hy):
                pts.append(np.column_stack(
                    [xs, np.full(len(xs), yv), np.full(len(xs), z)]))
            for xv in (cx - hx, cx + hx):
                pts.append(np.column_stack(
                    [np.full(len(ys), xv), ys, np.full(len(ys), z)]))
    return np.concatenate(pts, axis=0)
