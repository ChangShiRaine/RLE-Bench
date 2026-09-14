"""Verifier scoring entry point (runs inside the verifier container).

Stages the agent's artifacts, overlays the CANONICAL mesh assets (agent
meshes are never trusted), runs the full scoring pipeline, and writes:
  /logs/verifier/reward.json   numeric-only summary (Harbor reads this)
  /logs/verifier/report.json   full structured report for humans
"""
from __future__ import annotations

import json
import os
import shutil
import traceback

_TESTS = os.environ.get("RLEBENCH_TESTS", "/tests")
ARTIFACTS = os.environ.get("RLEBENCH_ARTIFACTS", "/logs/artifacts")
STAGING = os.environ.get("RLEBENCH_STAGING", "/tmp/submission")
VERIFIER_DIR = os.environ.get("RLEBENCH_VERIFIER_DIR", "/logs/verifier")
CANONICAL_ASSETS = os.path.join(_TESTS, "models", "assets")
ARM_REFERENCE = os.path.join(_TESTS, "models", "assets",
                             "franka_emika_panda", "panda_nohand.xml")


def _write(reward: dict, report: dict | None = None) -> None:
    os.makedirs(VERIFIER_DIR, exist_ok=True)
    with open(os.path.join(VERIFIER_DIR, "reward.json"), "w") as f:
        json.dump(reward, f, indent=2)
    if report is not None:
        with open(os.path.join(VERIFIER_DIR, "report.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)


def _stage_submission() -> None:
    if os.path.exists(STAGING):
        shutil.rmtree(STAGING)
    os.makedirs(STAGING, mode=0o700)
    # Only regular, bounded entry files are accepted. Never follow a submitted
    # symlink into the verifier tree, including controller.py or robot.xml.
    import stat
    from harness import config
    for name, limit in (("robot.xml", config.ROBOT_SOURCE_BYTES),
                        ("controller.py", config.PICK_CONTROLLER_SOURCE_BYTES)):
        path = os.path.join(ARTIFACTS, name)
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            continue
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise ValueError(f"invalid submitted file: {name}")
            raw = source.read(limit + 1)
            if len(raw) > limit:
                raise ValueError(f"submitted file exceeds size limit: {name}")
        with open(os.path.join(STAGING, name), "wb") as target:
            target.write(raw)
    shutil.copytree(CANONICAL_ASSETS, os.path.join(STAGING, "assets"))


def main() -> None:
    try:
        _stage_submission()

        from harness.base_design.scorer import score_submission
        report = score_submission(STAGING, arm_reference_xml=ARM_REFERENCE)

        reward = {"reward": float(report["reward"]),
                  "raw_total": float(report["raw_total"]),
                  "gated": int(report["gated"])}
        for stage, val in report["stages"].items():
            reward[f"stage_{stage}"] = float(val)
        for cid, val in report["checkpoints"].items():
            reward[f"cp_{cid}"] = float(val)
        headroom = report.get("design_headroom", {})
        for key in ("score", "total_mass_kg", "base_mass_kg",
                    "mass_budget_kg", "mass_headroom_kg",
                    "mass_headroom_fraction", "profile_length_m", "profile_length_headroom_m",
                    "profile_length_headroom_fraction",
                    "footprint_area_m2", "footprint_area_headroom_m2",
                    "footprint_area_headroom_fraction"):
            if key in headroom:
                reward[f"design_{key}"] = float(headroom[key])
        footprint = headroom.get("footprint_m", [])
        axis_headroom = headroom.get("footprint_axis_headroom_m", [])
        if len(footprint) == 2 and len(axis_headroom) == 2:
            reward["design_footprint_x_m"] = float(footprint[0])
            reward["design_footprint_y_m"] = float(footprint[1])
            reward["design_footprint_x_headroom_m"] = float(axis_headroom[0])
            reward["design_footprint_y_headroom_m"] = float(axis_headroom[1])
        _write(reward, report)
        print(json.dumps(report, indent=2, default=str))
        # post-reward renders of what the agent built (informational; the
        # reward is already on disk, so a render failure costs nothing)
        try:
            from harness.base_design.render import render_submission
            from rlebench.core.media import Media
            media = Media.verifier(VERIFIER_DIR)
            robot = os.path.join(STAGING, "robot.xml")
            if os.path.exists(robot):
                media.run("render", render_submission, robot, ARM_REFERENCE)
            media.report()
        except Exception as e:
            print(f"render skipped: {e}")
    except Exception:
        traceback.print_exc()
        _write({"reward": 0.0, "scoring_error": 1})


if __name__ == "__main__":
    main()
