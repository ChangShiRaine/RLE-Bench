"""Task04 verifier scoring entry point (runs inside the verifier).

Stages the agent's artifacts as UNTRUSTED input and re-runs the submitted policy
on the hidden evaluation seeds. Writes reward.json / report.json.
"""
from __future__ import annotations

import json
import os
import shutil
import traceback

ARTIFACTS = os.environ.get("RLEBENCH_ARTIFACTS", "/logs/artifacts")
STAGING = os.environ.get("RLEBENCH_STAGING", "/tmp/submission")
VERIFIER_DIR = os.environ.get("RLEBENCH_VERIFIER_DIR", "/logs/verifier")
ROBOT_DIR = os.environ["MOTIONTRACK_ROBOT_DIR"]
MOTION = os.environ["MOTIONTRACK_MOTION"]


def _write(reward, report=None):
    os.makedirs(VERIFIER_DIR, exist_ok=True)
    with open(os.path.join(VERIFIER_DIR, "reward.json"), "w") as f:
        json.dump(reward, f, indent=2)
    if report is not None:
        with open(os.path.join(VERIFIER_DIR, "report.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)


def _media():
    """Episode videos for a human reader; never an input to the score."""
    try:
        from rlebench.core.media import Media
        return Media.verifier(VERIFIER_DIR)
    except Exception as e:  # noqa: BLE001
        print(f"media disabled: {e}")
        return None


def main():
    try:
        if os.path.exists(STAGING):
            shutil.rmtree(STAGING)
        src = os.path.join(ARTIFACTS, "policy")
        if os.path.isdir(src):
            shutil.copytree(src, STAGING)
        else:
            os.makedirs(STAGING)

        from harness.scorer import score_submission
        media = _media()
        report = score_submission(STAGING, ROBOT_DIR, MOTION, media=media)

        reward = {"reward": float(report.get("reward", 0.0)),
                  "raw_total": float(report.get("raw_total", 0.0)),
                  "gated": int(report.get("gated", True))}
        for k in ("tracking_score", "tracking_multi", "mpjpe_m",
                  "anchor_pos_err_m", "joint_rmse_rad", "survival", "survived_s",
                  "action_jerk", "falls", "n_params"):
            if report.get(k) is not None:
                reward[k] = float(report[k])
        for name, ok in report.get("gate", {}).items():
            reward[f"gate_{name}"] = int(bool(ok))
        _write(reward, report)
        print(json.dumps(report, indent=2, default=str))
        if media is not None:
            media.report()
    except Exception:
        traceback.print_exc()
        _write({"reward": 0.0, "scoring_error": 1})


if __name__ == "__main__":
    main()
