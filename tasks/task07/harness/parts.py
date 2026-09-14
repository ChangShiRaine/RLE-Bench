"""The stamped U-channel bracket: one body, three thin-box geoms.

Web lying in the body's xy plane, two flanges rising from its long edges and
flaring outward by FLANGE_FLARE — visually and inertially faithful to the
reference part while keeping every collision geom convex and cheap.
"""
from __future__ import annotations

import mujoco
import numpy as np

from . import spec


def part_name(i: int) -> str:
    return f"part{i}"


def add_part(s: mujoco.MjSpec, i: int, pos, quat) -> None:
    """Add bracket body ``part{i}`` with a free joint at (pos, quat)."""
    b = s.worldbody.add_body(name=part_name(i), pos=list(pos),
                             quat=list(quat))
    b.add_freejoint(name=part_name(i))
    g = b.add_geom(name=f"{part_name(i)}_web",
                   type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=list(spec.WEB_HALF),
                   rgba=[0.55, 0.57, 0.60, 1.0],
                   friction=list(spec.CONTACT_FRICTION))
    g.density = spec.PART_DENSITY
    for side, tag in ((-1, "a"), (1, "b")):
        half_angle = (np.pi / 2 + side * spec.FLANGE_FLARE) / 2
        q = [np.cos(half_angle), np.sin(half_angle), 0.0, 0.0]
        off = spec.FLANGE_HALF[1] * np.sin(spec.FLANGE_FLARE) + 0.001
        g = b.add_geom(name=f"{part_name(i)}_flange_{tag}",
                       type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=list(spec.FLANGE_HALF),
                       pos=[0.0, side * (spec.WEB_HALF[1] + off),
                            spec.FLANGE_HALF[1] - 0.002],
                       quat=q,
                       rgba=[0.55, 0.57, 0.60, 1.0],
                       friction=list(spec.CONTACT_FRICTION))
        g.density = spec.PART_DENSITY

