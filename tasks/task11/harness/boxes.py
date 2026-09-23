"""Seeded box streams for task11.

A stream fixes the belt speed and, for every queued box, its type, arrival
time, dimensions, mass, lateral offset and yaw. It is deterministic from the
seed alone. Agents call ``stream(seed)`` on the public design seeds for
unlimited practice; the evaluation calls it on unpublished seeds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import spec


@dataclass(frozen=True)
class Box:
    index: int
    kind: str            # Amazon box designation
    arrival_t: float
    dims: tuple          # (l, w, h), m; l along the box x axis
    mass: float
    y: float
    yaw: float
    rgba: tuple

    @property
    def volume(self) -> float:
        return float(np.prod(self.dims))


@dataclass(frozen=True)
class Stream:
    seed: int
    belt_v: float
    boxes: tuple


def stream(seed: int) -> Stream:
    """Copies of every box type in a seeded shuffled order."""
    rng = np.random.default_rng([seed, spec.STREAM_BOXES])
    belt_v = float(rng.uniform(*spec.BELT_SPEED_RANGE))
    lo, hi = spec.BOX_COPIES_RANGE
    kinds = [k for k, n in enumerate(rng.integers(lo, hi + 1,
                                                  len(spec.BOX_TYPES)))
             for _ in range(int(n))]
    boxes, t = [], spec.FIRST_ARRIVAL_T
    for i, k in enumerate(rng.permutation(kinds)):
        name, dims = spec.BOX_TYPES[int(k)]
        yaw = (np.pi / 2) * int(rng.integers(2)) + np.deg2rad(
            rng.uniform(-spec.BOX_YAW_JITTER_DEG, spec.BOX_YAW_JITTER_DEG))
        y = spec.BELT_Y + rng.uniform(-spec.BOX_LATERAL_JITTER,
                                      spec.BOX_LATERAL_JITTER)
        shade = rng.uniform(0.75, 1.0)
        boxes.append(Box(i, name, round(t, 3), dims,
                         float(np.prod(dims) * spec.BOX_DENSITY), float(y),
                         float(yaw),
                         (0.72 * shade, 0.53 * shade, 0.33 * shade, 1.0)))
        t += float(rng.uniform(*spec.ARRIVAL_GAP_RANGE))
    return Stream(seed, belt_v, tuple(boxes))


def footprint_half(box: Box) -> np.ndarray:
    """World-axis half extents (x, y) of a box lying flat at its yaw."""
    c, s = abs(np.cos(box.yaw)), abs(np.sin(box.yaw))
    l, w, _ = box.dims
    return np.array([0.5 * (l * c + w * s), 0.5 * (l * s + w * c)])
