#!/usr/bin/env python3
"""Score YOUR staged policy the way the verifier will — on the design seeds.

    python3 dev_eval.py /logs/artifacts/policy
    python3 dev_eval.py /logs/artifacts/policy --seeds 101 211 307 --video /logs/ep.mp4

Same episode loop, same physics, same metrics as the evaluation; different seeds.
The numbers printed here are the quantities that score you, measured on machines
you can see instead of the ones you cannot.
"""
import argparse
import json
import os

from harness import evaluator, export, motion, robot

DESIGN_SEEDS = (101, 211, 307)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy_dir")
    ap.add_argument("--robot-dir", default=os.environ.get(
        "MOTIONTRACK_ROBOT_DIR", "/workspace/assets/unitree_g1"))
    ap.add_argument("--motion", default=os.environ.get("MOTIONTRACK_MOTION"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DESIGN_SEEDS))
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    problems = export.validate(args.policy_dir)
    if problems:
        raise SystemExit("submission is not valid:\n  " + "\n  ".join(problems))

    clip = motion.Motion.load(args.motion)
    policy = evaluator.OnnxPolicy(args.policy_dir)
    results = []
    for i, seed in enumerate(args.seeds):
        log = evaluator.run_episode(
            robot.build(args.robot_dir), clip, policy, seed,
            video=args.video if i == 0 else None, history=policy.history)
        results.append(log.summary())
        print(f"seed {seed}: " + json.dumps(results[-1], default=str))

    mean = {k: sum(r[k] for r in results) / len(results)
            for k in ("tracking_score", "tracking_multi", "mpjpe_m", "survival",
                      "survived_s")}
    print("\nmean: " + json.dumps(mean, indent=2))


if __name__ == "__main__":
    main()
