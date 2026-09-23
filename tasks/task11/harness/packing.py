"""The published fit rule that decides when the bin is full (pure numpy).

The verifier rasterizes every unheld box inside the tote into a heightmap
(cell = FIT_RES; each box contributes its world-AABB top over its AABB
footprint) and asks whether a box of given dims fits somewhere:

  * the footprint (l, w) or (w, l), grown by FIT_CLEARANCE per side and
    rounded up to whole cells, lies inside the interior grid;
  * base = max height under that grown footprint; base + h <= interior height;
  * inside the box's own footprint (the grown footprint minus its clearance
    ring), at least SUPPORT_FRAC of the cells and every centre cell lie
    within SUPPORT_TOL of the base, so the box would rest stably.

Suction from above cannot turn a box over, so only the two yaw-aligned
orientations with the scanned bottom face down are considered. ``res`` and
``clearance`` are parameters so a policy can plan on a finer grid; the
verifier always uses the published defaults.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import spec


def grid(res: float = spec.FIT_RES) -> tuple[int, int]:
    return (int(round(spec.BIN_INNER[0] / res)),
            int(round(spec.BIN_INNER[1] / res)))


def heightmap(corner_sets, res: float = spec.FIT_RES) -> np.ndarray:
    """(nx, ny) heights above the bin floor from boxes' (8, 3) world corners."""
    shape = grid(res)
    h = np.zeros(shape)
    lo = np.asarray(spec.BIN_LO[:2])
    for corners in corner_sets:
        c = np.asarray(corners, dtype=float)
        i0 = np.clip(np.floor((c[:, :2].min(0) - lo) / res + 1e-6), 0, shape)
        i1 = np.clip(np.ceil((c[:, :2].max(0) - lo) / res - 1e-6), 0, shape)
        i0, i1 = i0.astype(int), i1.astype(int)
        if (i1 > i0).all():
            region = h[i0[0]:i1[0], i0[1]:i1[1]]
            np.maximum(region, c[:, 2].max() - spec.BIN_FLOOR_Z, out=region)
    return h


def placements(hmap: np.ndarray, dims, res: float = spec.FIT_RES,
               clearance: float = spec.FIT_CLEARANCE):
    """Every feasible (ix, iy, rotated, base, fx, fy): the grown footprint
    spans cells [ix, ix + fx) x [iy, iy + fy); ``rotated`` swaps l and w."""
    l, w, h = (float(d) for d in dims)
    ring = int(round(clearance / res))
    out = []
    for rotated, (a, b) in ((False, (l, w)), (True, (w, l))):
        fx = int(np.ceil((a + 2 * clearance) / res - 1e-9))
        fy = int(np.ceil((b + 2 * clearance) / res - 1e-9))
        if fx > hmap.shape[0] or fy > hmap.shape[1]:
            continue
        win = sliding_window_view(hmap, (fx, fy))
        base = win.max(axis=(2, 3))
        ok_cells = win >= base[..., None, None] - spec.SUPPORT_TOL
        inner = ok_cells[..., ring:fx - ring, ring:fy - ring]
        cx = slice((fx - 1) // 2, fx // 2 + 1)
        cy = slice((fy - 1) // 2, fy // 2 + 1)
        ok = ((base + h <= spec.BIN_INNER[2] + 1e-9)
              & (inner.mean(axis=(2, 3)) >= spec.SUPPORT_FRAC - 1e-9)
              & ok_cells[..., cx, cy].all(axis=(2, 3)))
        out += [(int(i), int(j), rotated, float(base[i, j]), fx, fy)
                for i, j in zip(*np.nonzero(ok))]
    return out


def fits(hmap: np.ndarray, dims) -> bool:
    return bool(placements(hmap, dims))


def footprint_center(ix: int, iy: int, fx: int, fy: int,
                     res: float = spec.FIT_RES) -> np.ndarray:
    """World xy of the centre of a grown footprint."""
    return np.asarray(spec.BIN_LO[:2]) + np.array([ix + fx / 2,
                                                   iy + fy / 2]) * res
