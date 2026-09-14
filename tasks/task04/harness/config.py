"""Per-motion scoring thresholds for task04 (verifier-only).

The five tasks share one reward formula, but each motion owns its jerk
normalization and golden-reference guards. ``calibrate --emit`` updates only the
selected motion's entry. Evaluation seeds live in ``eval_seeds.json``.

Reward, per episode::

    score = W_TRACK   * tracking_multi
          + W_SURVIVE * survival
          - W_JERK    * min(action_jerk / thresholds.jerk_cap, 1)
          - W_FALL    * fell

Tracking is the uncapped 0.3/0.1/0.05 m multi-scale kernel. World-frame anchor
drift is a diagnostic and not part of any task's reward.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

GATE_CAP = 0.1

W_TRACK = 0.7
W_SURVIVE = 0.3
W_JERK = 0.15            # penalty only: a frozen policy is smooth and worthless
W_FALL = 0.05


@dataclass(frozen=True)
class MotionThresholds:
    jerk_cap: float
    oracle_ref_score: float
    null_ref_score: float
    calibrated: bool


# One record per motion, so calibrating one task cannot change another. Only
# dance1_subject2 is calibrated; the others carry its values until they are.
MOTION_THRESHOLDS: dict[str, MotionThresholds] = {
    "dance1_subject2": MotionThresholds(
        jerk_cap=0.3054,
        oracle_ref_score=0.3901,
        null_ref_score=0.0000,
        calibrated=True,
    ),
    "fight1_subject2": MotionThresholds(
        jerk_cap=0.3054,
        oracle_ref_score=0.3901,
        null_ref_score=0.0000,
        calibrated=False,
    ),
    "fallAndGetUp2_subject2": MotionThresholds(
        jerk_cap=0.3054,
        oracle_ref_score=0.3901,
        null_ref_score=0.0000,
        calibrated=False,
    ),
    "run1_subject2": MotionThresholds(
        jerk_cap=0.3054,
        oracle_ref_score=0.3901,
        null_ref_score=0.0000,
        calibrated=False,
    ),
    "sprint1_subject2": MotionThresholds(
        jerk_cap=0.3054,
        oracle_ref_score=0.3901,
        null_ref_score=0.0000,
        calibrated=False,
    ),
}


def motion_name(motion_path: str | os.PathLike[str]) -> str:
    """Return the manifest motion name from a ``.../<motion>.npz`` path."""
    return os.path.splitext(os.path.basename(os.fspath(motion_path)))[0]


def thresholds_for_motion(motion_path: str | os.PathLike[str]) -> MotionThresholds:
    """Resolve one task's thresholds, failing closed for an unknown motion."""
    name = motion_name(motion_path)
    try:
        return MOTION_THRESHOLDS[name]
    except KeyError as exc:
        known = ", ".join(MOTION_THRESHOLDS)
        raise KeyError(f"no task04 thresholds for motion {name!r}; known: {known}") from exc
