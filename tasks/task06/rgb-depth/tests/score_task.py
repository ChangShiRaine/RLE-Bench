"""Task06 / rgb-depth verifier scoring entry point (runs inside the verifier container).

Stages the agent's artifacts as UNTRUSTED input, scores variant "b"
through the sandboxed pipeline, and writes reward.json / report.json.
"""
from __future__ import annotations

import json
import os
import shutil
import traceback

VARIANT = "b"
ARTIFACTS = os.environ.get("RLEBENCH_ARTIFACTS", "/logs/artifacts")
STAGING = os.environ.get("RLEBENCH_STAGING", "/tmp/submission")
VERIFIER_DIR = os.environ.get("RLEBENCH_VERIFIER_DIR", "/logs/verifier")


def _write(reward, report=None):
    os.makedirs(VERIFIER_DIR, exist_ok=True)
    with open(os.path.join(VERIFIER_DIR, "reward.json"), "w") as f:
        json.dump(reward, f, indent=2)
    if report is not None:
        with open(os.path.join(VERIFIER_DIR, "report.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)


def _media():
    """Stage B episode videos for a human reader; never an input to the score."""
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
        from harness.artifact_staging import stage_submission
        stage_submission(ARTIFACTS, STAGING)

        from harness.scorer import score_submission
        media = _media()
        report = score_submission(STAGING, VARIANT, media=media)

        reward = {"reward": float(report["reward"]),
                   "raw_total": float(report["raw_total"]),
                   "gated": int(report["gated"])}
        for stage, val in report["stages"].items():
            reward[f"stage_{stage}"] = float(val)
        for cid, val in report["checkpoints"].items():
            reward[f"cp_{cid}"] = float(val)
        for k in ("abs_median_trans_mm", "abs_median_rot_deg",
                  "stage_b_median_trans_mm", "stage_b_median_rot_deg",
                  "abs_occluded_trans_mm", "frame_time_s",
                  "stage_a_frames", "stage_b_frames",
                  "stage_b_occluded_frames", "stage_b_occluded_fraction",
                  "stage_a_shape_accuracy", "stage_b_shape_accuracy",
                  "accuracy_reward", "efficiency_deduction",
                  "inference_wall_s", "inference_hz"):
            if report.get(k) is not None:
                reward[k] = float(report[k])
        _write(reward, report)
        print(json.dumps(report, indent=2, default=str))
        if media is not None:
            media.report()
    except Exception:
        traceback.print_exc()
        _write({"reward": 0.0, "scoring_error": 1})


if __name__ == "__main__":
    main()
