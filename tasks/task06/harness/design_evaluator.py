"""Root-only design-set diagnostic evaluator for Task 06."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import os

import numpy as np

from . import episodes, sandbox, spec


ARTIFACT_DIR = "/logs/artifacts"
SINGLE_FRAMES_PER_SEED = 4
PUSH_MAX_FRAMES = 40
STREAM_DESIGN_DIAGNOSTIC = 5205
BOOTSTRAP_DRAWS = 2000
STREAM_BOOTSTRAP = 15205
CHANNELS = {
    "a": ("rgb",),
    "b": ("rgb", "depth"),
    "c": ("rgb", "depth"),
    "d": ("rgb", "depth"),
}


@dataclass
class DesignBattery:
    single_frames: list = field(default_factory=list)
    episodes: list = field(default_factory=list)

    @property
    def push_frames(self) -> list:
        return [frame for episode in self.episodes for frame in episode.frames]


@lru_cache(maxsize=1)
def build_design_battery() -> DesignBattery:
    battery = DesignBattery()
    for seed in spec.DESIGN_SEEDS:
        rng = np.random.default_rng([seed, STREAM_DESIGN_DIAGNOSTIC])
        subs = rng.integers(0, 2**31 - 1, size=SINGLE_FRAMES_PER_SEED)
        battery.single_frames.extend(
            episodes.single_frame(int(sub), shapes=spec.BLOCK_SHAPES)
            for sub in subs
        )
        battery.episodes.append(episodes.push_episode(
            seed,
            render=True,
            max_frames=PUSH_MAX_FRAMES,
            shapes=spec.BLOCK_SHAPES,
        ))
    return battery


def _summary(values: np.ndarray, clusters: np.ndarray, seed: int) -> dict | None:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    _, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    cluster_sums = np.bincount(inverse, weights=values)
    cluster_counts = np.bincount(inverse)
    if len(cluster_sums) == 1:
        interval = [mean, mean]
    else:
        rng = np.random.default_rng([STREAM_BOOTSTRAP, seed])
        sampled = rng.integers(
            0, len(cluster_sums), size=(BOOTSTRAP_DRAWS, len(cluster_sums)))
        bootstrap_means = (
            cluster_sums[sampled].sum(axis=1)
            / cluster_counts[sampled].sum(axis=1)
        )
        interval = np.quantile(bootstrap_means, [0.025, 0.975]).tolist()
    return {
        "mean": mean,
        "std": std,
        "median": float(np.median(values)),
        "mean_95_ci": [max(0.0, float(interval[0])), float(interval[1])],
    }


def error_statistics(predictions: np.ndarray, ground_truth: np.ndarray,
                     clusters: np.ndarray | None = None) -> dict:
    predictions = np.asarray(predictions, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=float)
    n = len(ground_truth)
    if predictions.shape != ground_truth.shape:
        predictions = np.full_like(ground_truth, np.nan, dtype=float)
    valid = np.isfinite(predictions).all(axis=1)
    if clusters is None:
        clusters = np.arange(n)
    clusters = np.asarray(clusters)
    if clusters.shape != (n,):
        raise ValueError("clusters must contain one id per frame")
    pred = predictions[valid]
    truth = ground_truth[valid]
    translation = np.hypot(pred[:, 0] - truth[:, 0],
                           pred[:, 1] - truth[:, 1])
    delta = pred[:, 2] - truth[:, 2]
    valid_clusters = clusters[valid]
    rotation = np.abs(np.pi - np.mod(np.pi - delta, 2 * np.pi))
    return {
        "frames": int(n),
        "valid_frames": int(valid.sum()),
        "valid_fraction": float(valid.mean()) if n else 0.0,
        "ci_method": "cluster bootstrap; push episodes are resampling units",
        "translation_error_m": _summary(translation, valid_clusters, 1),
        "rotation_error_rad": _summary(rotation, valid_clusters, 2),
    }


def _poses_or_invalid(result: sandbox.SandboxResult, n: int) -> np.ndarray:
    if result.ok:
        return result.poses
    return np.full((n, 3), np.nan, dtype=float)


def evaluate_submission(
    variant: str,
    artifact_dir: str = ARTIFACT_DIR,
    diagnostic: DesignBattery | None = None,
) -> dict:
    if variant not in CHANNELS:
        raise ValueError("unknown task variant")
    mode = "torch" if variant == "c" else "code"
    required = "model.pt" if mode == "torch" else "estimator.py"
    candidate = os.path.join(artifact_dir, required)
    if not os.path.isfile(candidate) or os.path.islink(candidate):
        return {"ok": False, "error": f"{required} is missing or invalid"}

    diagnostic = diagnostic or build_design_battery()
    channels = CHANNELS[variant]
    kwargs = ({"model_path": candidate} if mode == "torch"
              else {"submission_dir": artifact_dir})
    single = diagnostic.single_frames
    single_job = sandbox.run_job(
        mode,
        [frame.obs for frame in single],
        channels,
        resets=[True] * len(single),
        prepare_submission=False,
        **kwargs,
    )
    single_predictions = _poses_or_invalid(single_job, len(single))

    push_predictions = []
    push_truth = []
    errors = []
    if not single_job.ok:
        errors.append(f"single-frame run failed: {single_job.error}")
    for index, episode in enumerate(diagnostic.episodes):
        frames = episode.frames
        job = sandbox.run_job(
            mode,
            [frame.obs for frame in frames],
            channels,
            resets=[True] + [False] * (len(frames) - 1),
            prepare_submission=False,
            **kwargs,
        )
        push_predictions.append(_poses_or_invalid(job, len(frames)))
        push_truth.append(np.asarray([frame.gt for frame in frames], dtype=float))
        if not job.ok:
            errors.append(f"push run {index} failed: {job.error}")

    single_truth = np.asarray([frame.gt for frame in single], dtype=float)
    push_predictions_array = np.concatenate(push_predictions)
    push_truth_array = np.concatenate(push_truth)
    all_predictions = np.concatenate([single_predictions,
                                      push_predictions_array])
    all_truth = np.concatenate([single_truth, push_truth_array])
    single_clusters = np.arange(len(single), dtype=int)
    push_clusters = np.concatenate([
        np.full(len(episode.frames), len(single) + index, dtype=int)
        for index, episode in enumerate(diagnostic.episodes)
    ])
    all_clusters = np.concatenate([single_clusters, push_clusters])
    return {
        "ok": not errors,
        "errors": errors,
        "single_frames": error_statistics(
            single_predictions, single_truth, single_clusters),
        "push_frames": error_statistics(push_predictions_array,
                                         push_truth_array, push_clusters),
        "all_frames": error_statistics(
            all_predictions, all_truth, all_clusters),
    }
