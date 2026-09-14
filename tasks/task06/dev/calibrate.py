"""Audit the fixed-kernel task06 evaluation battery.

There are no saturating thresholds to calibrate.  This command optionally
rotates the ten verifier-only seeds, builds the full 100/600-frame battery,
checks that it contains meaningful moving occlusions, and prints reference and
probe kernel scores for human review.
"""
from __future__ import annotations

import argparse
import json
import secrets
import sys
import tempfile
from pathlib import Path

import numpy as np

from harness import battery as battery_mod
from harness import reference, scorer
from .audit_motion import MIN_MEDIAN_ROTATION_DEG, MIN_ROTATING_FRACTION, motion_metrics


def battery_failures(bat) -> list[str]:
    failures = []
    if len(bat.stage_a) != battery_mod.STAGE_A_FRAMES:
        failures.append(f"Stage A has {len(bat.stage_a)} frames, expected "
                        f"{battery_mod.STAGE_A_FRAMES}")
    if len(bat.episodes) != battery_mod.STAGE_B_EPISODES:
        failures.append(f"Stage B has {len(bat.episodes)} episodes, expected "
                        f"{battery_mod.STAGE_B_EPISODES}")
    stage_b = bat.stage_b
    target = battery_mod.STAGE_B_EPISODES * battery_mod.EPISODE_MAX_FRAMES
    if len(stage_b) != target:
        failures.append(f"Stage B has {len(stage_b)} frames, expected {target}")
    occ = np.asarray([f.occluded for f in stage_b], dtype=bool)
    if not len(occ) or occ.mean() < 0.10:
        failures.append(f"Stage B occlusion fraction is "
                        f"{float(occ.mean()) if len(occ) else 0.0:.1%}, expected >=10%")
    moving = False
    for episode in bat.episodes:
        idx = [i for i, frame in enumerate(episode.frames) if frame.occluded]
        if len(idx) >= 3:
            gt = np.asarray([episode.frames[i].gt for i in idx])
            displacement = np.linalg.norm(gt[:, None, :2] - gt[None, :, :2], axis=-1)
            moving |= float(displacement.max()) >= 0.04
    if not moving:
        failures.append("no episode moves the block at least 40 mm while occluded")
    motion = [motion_metrics(episode) for episode in bat.episodes if episode.frames]
    if not motion or np.median([m["rotation_range_deg"] for m in motion]) < MIN_MEDIAN_ROTATION_DEG:
        failures.append("tracking episodes have insufficient angular range")
    if not motion or np.mean([m["rotating_fraction"] for m in motion]) < MIN_ROTATING_FRACTION:
        failures.append("too few tracking frames contain angular motion")
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotate-seeds", action="store_true")
    args = parser.parse_args(argv)

    seeds_path = Path(__file__).resolve().parents[1] / "harness" / "eval_seeds.json"
    if args.rotate_seeds:
        seeds = [secrets.randbits(62) for _ in range(battery_mod.STAGE_B_EPISODES)]
        seeds_path.write_text(json.dumps({
            "comment": "ten verifier-only seeds: 100 Stage-A frames and 600 Stage-B frames",
            "seeds": seeds,
        }, indent=2) + "\n")
        print(f"rotated evaluation seeds -> {seeds_path}")

    print("building full task06 battery ...")
    bat = battery_mod.build_battery()
    failures = battery_failures(bat)

    for variant in ("a", "b", "d"):
        with tempfile.TemporaryDirectory() as directory:
            reference.build_reference_submission(variant, directory)
            report = scorer.score_submission(directory, variant, battery=bat)
        print(f"reference {variant}: A={report['checkpoints']['A.pose']:.4f} "
              f"B={report['checkpoints']['B.pose']:.4f} "
              f"reward={report['reward']:.4f}")

    for kind, variant in (("naive_color", "a"), ("raw_icp", "b"),
                          ("last_pose", "d")):
        with tempfile.TemporaryDirectory() as directory:
            reference.build_probe_submission(kind, directory)
            report = scorer.score_submission(directory, variant, battery=bat)
        print(f"probe {kind}: A={report['checkpoints']['A.pose']:.4f} "
              f"B={report['checkpoints']['B.pose']:.4f} "
              f"reward={report['reward']:.4f}")

    if failures:
        print("BATTERY AUDIT FAILURES:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("battery audit: all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
