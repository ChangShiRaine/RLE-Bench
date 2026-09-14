"""Training progress: a JSONL trail plus a readable line per entry.

The agent session is headless and may contain several training runs. Without
a trail the only thing visible at the end is a final number, which is no basis
for deciding what to change, so the harness provides the log.

The evaluation hook is the part worth having: reward is whatever a submission
decided reward means, while `tracking_score` and the scored `tracking_multi` are
measured by the same evaluator on published design seeds. It costs about 0.2 s
per episode and is off the training path entirely.
"""
from __future__ import annotations

import json
import os
import time


class TrainingLog:

    def __init__(self, path: str | None = None, echo: bool = True):
        self.path = path
        self.echo = echo
        self.start = time.time()
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            open(path, "w").close()

    def log(self, **fields):
        entry = {"elapsed_s": round(time.time() - self.start, 1), **fields}
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(entry, default=float) + "\n")
        if self.echo:
            print("  ".join(f"{k} {_fmt(v)}" for k, v in entry.items()), flush=True)
        return entry


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def evaluate(policy_dir: str, robot_dir: str, clip, seeds=(101, 211, 307)) -> dict:
    from . import evaluator, robot

    policy = evaluator.OnnxPolicy(policy_dir)
    runs = [evaluator.run_episode(robot.build(robot_dir), clip, policy, seed,
                                  history=policy.history).summary()
            for seed in seeds]
    return {
        "tracking_score": sum(r["tracking_score"] for r in runs) / len(runs),
        "mpjpe_m": sum(r["mpjpe_m"] for r in runs) / len(runs),
        "tracking_multi": sum(r["tracking_multi"] for r in runs) / len(runs),
        "survived_s": sum(r["survived_s"] for r in runs) / len(runs),
    }
