"""Minimal kernel checkpoints for the task06 family (verifier-only).

Each group uses worst-five mean pose errors and requires every shape ID to match.
Stage A and B contribute 0.3 and 0.7 before the efficiency deduction.
"""
from __future__ import annotations

import numpy as np

from rlebench.core.scoring import Checkpoint
from . import config, spec

VARIANTS = ("a", "b", "c", "d")

CHANNELS = {
    "a": ("rgb",),
    "b": ("rgb", "depth"),
    "c": ("rgb", "depth"),
    "d": ("rgb", "depth"),
}


def checkpoint_list(variant: str) -> list[Checkpoint]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    cps = [
        Checkpoint("G.load", "gate", 0.0, gate=True),
        Checkpoint("A.pose", "stage_a", config.STAGE_A_WEIGHT),
        Checkpoint("B.pose", "stage_b", config.STAGE_B_WEIGHT),
    ]
    assert abs(sum(c.weight for c in cps) - 1.0) < 1e-9
    return cps


def pose_errors(preds: np.ndarray, gts: np.ndarray):
    """Return translation/rotation errors; invalid poses become infinity."""
    preds = np.asarray(preds, dtype=float)
    gts = np.asarray(gts, dtype=float)
    if preds.shape != gts.shape or preds.ndim != 2 or preds.shape[1] != 3:
        n = len(gts) if gts.ndim else 0
        return np.full(n, np.inf), np.full(n, np.inf)
    trans = np.hypot(preds[:, 0] - gts[:, 0], preds[:, 1] - gts[:, 1])
    with np.errstate(invalid="ignore"):
        rot = np.abs((preds[:, 2] - gts[:, 2] + np.pi) % (2 * np.pi) - np.pi)
    bad = (~np.isfinite(preds).all(axis=1)
           | (np.abs(preds[:, 0]) > spec.POSE_XY_BOUND)
           | (np.abs(preds[:, 1]) > spec.POSE_XY_BOUND))
    return np.where(bad, np.inf, trans), np.where(bad, np.inf, rot)


def kernel_score(error: np.ndarray, std: float) -> np.ndarray:
    """Task04-style ``exp(-error**2 / std**2)``; invalid error scores zero."""
    error = np.asarray(error, dtype=float)
    if not np.isfinite(std) or std <= 0:
        raise ValueError("kernel std must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore"):
        score = np.exp(-np.square(error) / std**2)
    return np.where(np.isfinite(error), score, 0.0)


def pose_kernel_scores(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """Equal translation/rotation kernel credit for every frame."""
    trans, rot = pose_errors(preds, gts)
    return 0.5 * (
        kernel_score(trans, config.POSE_TRANS_STD_M)
        + kernel_score(rot, config.POSE_ROT_STD_RAD)
    )


def mean_pose_kernel(preds: np.ndarray, gts: np.ndarray) -> float:
    scores = pose_kernel_scores(preds, gts)
    return float(np.mean(scores)) if len(scores) else 0.0


def robust_median(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.median(values)) if len(values) else float("inf")


def worst_frame_mean(errors) -> float:
    """Mean of the largest five errors (all for shorter development groups)."""
    values = np.asarray(errors, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        return float("inf")
    return float(np.sort(values)[-config.POSE_WORST_FRAMES:].mean())


def group_score(preds, gts, shape_ids, true_ids) -> dict:
    n = len(gts)
    predicted = np.asarray(shape_ids)
    truth = np.asarray(true_ids)
    if truth.shape != (n,) or not np.isin(truth, (0, 1, 2)).all():
        raise ValueError("missing or invalid verifier shape truth")
    correct = (predicted == truth) if predicted.shape == truth.shape else np.zeros(n, bool)
    shape_ok = bool(n and correct.all())
    trans, rot = pose_errors(preds, gts)
    xy, angle = worst_frame_mean(trans), worst_frame_mean(rot)
    pose_score = float(.5 * (kernel_score(xy, config.POSE_TRANS_STD_M)
                             + kernel_score(angle, config.POSE_ROT_STD_RAD)))
    return dict(frames=n, shape_correct_frames=int(correct.sum()), shape_ok=shape_ok,
                position_worst5_mean_m=xy if np.isfinite(xy) else None,
                angle_worst5_mean_rad=angle if np.isfinite(angle) else None,
                pose_score=pose_score, score=pose_score if shape_ok else 0.0)


def stage_summaries(results):
    n = len(results["a_gts"])
    groups = np.array_split(np.arange(n), min(config.STAGE_A_GROUPS, n)) if n else []
    a = [group_score(np.asarray(results["a_preds"])[ids], np.asarray(results["a_gts"])[ids],
                     np.asarray(results["a_shape_ids"])[ids], np.asarray(results["a_true_ids"])[ids])
         for ids in groups]
    b = [group_score(p, g, s, t) for p, g, s, t in zip(
        results["b_episode_preds"], results["b_episode_gts"],
        results["b_shape_ids"], results["b_true_ids"], strict=True)]
    return a, b


def efficiency_deduction(frame_count: int, wall_s: float) -> float:
    if frame_count <= 0 or not np.isfinite(wall_s) or wall_s <= 0:
        return config.EFFICIENCY_MAX_DEDUCTION
    hz = frame_count / wall_s
    fraction = (config.EFFICIENCY_FULL_HZ - hz) / (
        config.EFFICIENCY_FULL_HZ - config.EFFICIENCY_ZERO_HZ)
    return float(config.EFFICIENCY_MAX_DEDUCTION * np.clip(fraction, 0, 1))


def evaluate(variant: str, results: dict) -> dict[str, float]:
    if variant not in CHANNELS:
        raise ValueError(f"unknown variant {variant!r}")
    a, b = stage_summaries(results)
    return {
        "G.load": 1.0 if results.get("gate_ok") else 0.0,
        "A.pose": float(np.mean([g["score"] for g in a])) if a else 0.0,
        "B.pose": float(np.mean([g["score"] for g in b])) if b else 0.0,
    }
