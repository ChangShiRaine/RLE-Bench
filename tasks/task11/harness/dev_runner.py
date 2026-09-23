#!/usr/bin/env python3
"""Run a policy package through the real episode loop on a practice seed.

Copied to /workspace/dev_runner.py in the Task 11 agent image.
"""
from __future__ import annotations

import argparse
import json
import sys

from harness import runtime, sandbox, spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy_dir")
    parser.add_argument("--seed", type=int, default=spec.DESIGN_SEEDS[0])
    parser.add_argument("--policy-seed", type=int, default=0,
                        help="RNG seed passed to policy.reset()")
    parser.add_argument("--budget", type=float, default=spec.EPISODE_T)
    parser.add_argument("--video", default=None,
                        help="record the overview camera to this mp4")
    args = parser.parse_args()

    with sandbox.PolicyProcess(args.policy_dir) as policy:
        try:
            policy.check_loaded()
        except sandbox.PolicyLoadError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        log = runtime.run_episode(
            policy, args.seed, budget_t=args.budget, render=True,
            policy_seed=args.policy_seed,
            wall_budget_s=spec.EPISODE_WALL_BUDGET_S, video=args.video,
            video_every=2)
        if policy.dead or log.policy_faults:
            print(policy.stderr_tail(), file=sys.stderr)
    print(json.dumps(log.metrics(), indent=2, default=str))
    return None


if __name__ == "__main__":
    sys.exit(main())
