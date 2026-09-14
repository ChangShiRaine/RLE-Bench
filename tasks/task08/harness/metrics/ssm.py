"""Static Stability Margin (SSM).

Shortest horizontal distance from the CoM ground-projection to the nearest
support-polygon edge. Positive => CoM inside polygon => statically stable.
"""
from __future__ import annotations

import numpy as np


def static_stability_margin(com_xy: np.ndarray, polygon: np.ndarray) -> float:
    """Signed distance CoM -> nearest edge (positive = inside).

    `polygon` is (M, 2) CCW. For points outside the polygon the magnitude is
    the worst edge-line violation (a margin, not the exact Euclidean
    distance); the sign is what the checkpoints gate on.

    Degenerate supports (M < 3) cannot contain the CoM: returns -inf.

    Analytic check (see tests): CoM at centroid of a unit square -> 0.5.
    """
    polygon = np.asarray(polygon, dtype=float).reshape(-1, 2)
    com_xy = np.asarray(com_xy, dtype=float)[:2]
    if len(polygon) < 3:
        return float("-inf")
    margins = []
    for i in range(len(polygon)):
        p1, p2 = polygon[i], polygon[(i + 1) % len(polygon)]
        edge = p2 - p1
        length = np.linalg.norm(edge)
        if length < 1e-12:
            continue
        # z-component of cross(edge, com - p1): positive inside for CCW
        margins.append((edge[0] * (com_xy[1] - p1[1])
                        - edge[1] * (com_xy[0] - p1[0])) / length)
    return float(min(margins)) if margins else float("-inf")
