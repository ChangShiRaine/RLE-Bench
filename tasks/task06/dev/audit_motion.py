"""Measure contact-driven motion over the scored tracking window."""
from __future__ import annotations

import argparse
import json

import numpy as np

from harness import battery, episodes, spec

MIN_MEDIAN_ROTATION_DEG = 60.0
MIN_ROTATING_FRACTION = 0.25


def motion_metrics(episode):
    poses = np.asarray(episode.gts)
    yaw = np.unwrap(poses[:, 2])
    turns = np.abs(np.diff(yaw))
    occluded = np.asarray(episode.occluded_flags)
    hidden_motion = episodes.occluded_motion(episode)
    return {
        "rotation_range_deg": float(np.degrees(np.ptp(yaw))),
        "rotation_travel_deg": float(np.degrees(turns.sum())),
        "rotating_fraction": float(np.mean(turns > np.deg2rad(1))),
        "occluded_rotation_deg": float(np.degrees(turns[occluded[1:]].sum())),
        "occluded_interval_rotation_deg": hidden_motion["rotation_deg"],
        "occluded_rotating_steps": hidden_motion["rotating_steps"],
        "occluded_fraction": float(occluded.mean()),
        "translation_range_m": float(np.linalg.norm(
            poses[:, None, :2] - poses[None, :, :2], axis=-1).max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", action="store_true", help="audit hidden evaluation seeds")
    args = parser.parse_args()
    cases = ([(seed, episodes.shape_for_seed(seed, spec.BLOCK_SHAPES))
              for seed in battery.load_eval_seeds()] if args.eval else
             [(seed, shape) for shape in spec.BLOCK_SHAPES
              for seed in spec.DESIGN_SEEDS])
    metrics = []
    for index, (seed, shape) in enumerate(cases):
        episode = episodes.push_episode(seed, render=False, shapes=(shape,),
                                        max_frames=battery.EPISODE_MAX_FRAMES)
        metrics.append(motion_metrics(episode))
        print(json.dumps({"episode": index, "shape": shape,
                          "frames": len(episode.frames),
                          **metrics[-1]}), flush=True)
    print(json.dumps({"summary": {
        "episodes": len(metrics),
        "median_rotation_range_deg": float(np.median(
            [m["rotation_range_deg"] for m in metrics])),
        "mean_rotating_fraction": float(np.mean(
            [m["rotating_fraction"] for m in metrics])),
        "mean_occluded_fraction": float(np.mean(
            [m["occluded_fraction"] for m in metrics])),
        "mean_occluded_interval_rotation_deg": float(np.mean(
            [m["occluded_interval_rotation_deg"] for m in metrics])),
        "mean_occluded_rotating_steps": float(np.mean(
            [m["occluded_rotating_steps"] for m in metrics])),
    }}))


if __name__ == "__main__":
    main()
