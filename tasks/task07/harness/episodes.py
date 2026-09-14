"""Seeded pile generation for task07.

A pile is produced by dropping n ~ U[N_PARTS_RANGE] brackets from a jittered
grid above the bin and settling under contact physics — deterministic at
runtime from the seed alone (nothing is baked). Agents call this with the
public design seeds to make unlimited practice piles; the evaluation calls
the same code on unpublished seeds.
"""
from __future__ import annotations

import numpy as np

from . import scene, spec

SETTLE_T = 2.5          # seconds of drop-and-settle physics
RESETTLE_T = 1.0        # extra settle after re-spawning escapees
DROP_BASE_Z = spec.STAND_H + 0.08
DROP_LAYER_DZ = 0.06


def _spawn_poses(rng: np.random.Generator, n: int):
    poses = []
    for i in range(n):
        layer, col = divmod(i, 8)
        x = spec.BIN_POS[0] + (col % 4 - 1.5) * 0.066 \
            + rng.uniform(-0.012, 0.012)
        y = spec.BIN_POS[1] + (col // 4 - 0.5) * 0.09 \
            + rng.uniform(-0.015, 0.015)
        z = DROP_BASE_Z + layer * DROP_LAYER_DZ
        yaw = rng.uniform(0, 2 * np.pi)
        poses.append((np.array([x, y, z]),
                      np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])))
    return poses


def _inside_bin(p: np.ndarray) -> bool:
    lo, hi = np.asarray(spec.BIN_LO), np.asarray(spec.BIN_HI)
    return bool(np.all(p >= lo) and np.all(p <= hi))


def pile_poses(seed: int):
    """Settled part poses for one episode: list of (pos, quat) arrays.

    Parts that bounce out of the bin during settling are re-dropped once
    above the pile center; a part still outside after that is excluded
    (deterministic either way).
    """
    rng = np.random.default_rng([seed, spec.STREAM_PILE])
    n = int(rng.integers(spec.N_PARTS_RANGE[0], spec.N_PARTS_RANGE[1] + 1))
    raw = _spawn_poses(rng, n)
    model, data = scene.build_scene(part_poses=raw, with_arm=False)
    scene.settle(model, data, SETTLE_T)

    escaped = [i for i in range(n)
               if not _inside_bin(scene.part_pose(data, i)[0])]
    if escaped:
        for i in escaped:
            yaw = rng.uniform(0, 2 * np.pi)
            scene.set_part_pose(
                data, i,
                [spec.BIN_POS[0] + rng.uniform(-0.05, 0.05),
                 spec.BIN_POS[1] + rng.uniform(-0.04, 0.04),
                 spec.STAND_H + spec.BIN_INNER[2] + 0.10],
                [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        scene.settle(model, data, RESETTLE_T)

    poses = []
    for i in range(n):
        pos, quat = scene.part_pose(data, i)
        if _inside_bin(pos):
            poses.append((pos, quat))
    return poses


def build_episode_scene(seed: int, with_arm: bool = True):
    """(model, data) with the settled pile of ``seed`` and nuisance drawn."""
    return scene.build_scene(part_poses=pile_poses(seed), seed=seed,
                             with_arm=with_arm)
