"""Evaluation battery for the task06 family (verifier-only).

Stage A contains exactly 100 independent single frames.  Stage B contains ten
ordered push episodes of at most 60 frames each (600 frames with the current
fixed-length renderer).  Evaluation seeds are verifier-only 63-bit values.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import episodes, spec

STAGE_A_FRAMES = 100
STAGE_B_EPISODES = 10
EPISODE_MAX_FRAMES = 60
STREAM_EVAL_FRAMES = 9100


def load_eval_seeds(path: str | None = None) -> list[int]:
    p = path or os.environ.get(
        "RLEBENCH_EVAL_SEEDS",
        str(Path(__file__).resolve().parent / "eval_seeds.json"))
    with open(p) as f:
        seeds = json.load(f)["seeds"]
    return [int(s) for s in seeds]


@dataclass
class Battery:
    stage_a: list = field(default_factory=list)
    episodes: list = field(default_factory=list)

    @property
    def stage_b(self) -> list:
        return [frame for episode in self.episodes for frame in episode.frames]


def stage_a_frames(seed: int, n: int, shapes=None) -> list:
    rng = np.random.default_rng([seed, STREAM_EVAL_FRAMES])
    subs = rng.integers(0, 2**31 - 1, size=n)
    return [episodes.single_frame(int(s), shapes=shapes) for s in subs]


def build_battery(seeds: list[int] | None = None, shapes=None, *,
                  stage_a_count: int = STAGE_A_FRAMES,
                  episode_max_frames: int = EPISODE_MAX_FRAMES) -> Battery:
    seeds = list(seeds if seeds is not None else load_eval_seeds())
    if not seeds:
        raise ValueError("evaluation battery needs at least one seed")
    if stage_a_count < 1 or episode_max_frames < 1:
        raise ValueError("battery frame counts must be positive")
    shapes = spec.BLOCK_SHAPES if shapes is None else shapes
    b = Battery()
    base, remainder = divmod(stage_a_count, len(seeds))
    for i, seed in enumerate(seeds):
        n = base + (1 if i < remainder else 0)
        if n:
            b.stage_a.extend(stage_a_frames(seed, n, shapes=shapes))
        b.episodes.append(episodes.push_episode(
            seed, render=True, max_frames=episode_max_frames, shapes=shapes))
    return b
