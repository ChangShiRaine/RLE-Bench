"""Verifier entry point for the task06 family (verifier-only).

The submitted estimator/model is loaded in the sandbox, then evaluated on 100
independent Stage-A frames and ten ordered Stage-B episodes. Groups use worst-five mean
pose errors, fixed-ID shape checks, and a parent-timed efficiency deduction.
The only gate is
whether the submission loads; metadata and self-reported evaluation do not
affect reward.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from rlebench.core.scoring import aggregate
from . import battery as battery_mod
from . import config, sandbox
from .checkpoints import (CHANNELS, checkpoint_list, efficiency_deduction,
                          evaluate, pose_errors, robust_median, stage_summaries)


@dataclass
class Submission:
    dir: str = ""
    mode: str = "code"
    model_path: str | None = None
    errors: list[str] = field(default_factory=list)


def load_submission(submission_dir: str, variant: str) -> Submission:
    sub = Submission(dir=submission_dir,
                     mode="torch" if variant == "c" else "code")
    if sub.mode == "torch":
        path = os.path.join(submission_dir, "model.pt")
        if os.path.exists(path):
            sub.model_path = path
        else:
            sub.errors.append("model.pt missing")
    elif not os.path.exists(os.path.join(submission_dir, "estimator.py")):
        sub.errors.append("estimator.py missing")
    return sub


def _run(sub: Submission, frames, channels, resets=None):
    return sandbox.run_job(
        sub.mode, frames, channels, resets=resets,
        submission_dir=sub.dir, model_path=sub.model_path)


def _invalid(n: int) -> np.ndarray:
    return np.full((n, 3), np.nan, dtype=float)


def _median_metrics(preds: np.ndarray, gts: np.ndarray) -> tuple[float | None, float | None]:
    trans, rot = pose_errors(preds, gts)
    mt, mr = robust_median(trans), robust_median(rot)
    return ((round(mt * 1000, 2) if np.isfinite(mt) else None),
            (round(np.degrees(mr), 2) if np.isfinite(mr) else None))


def score_submission(submission_dir: str, variant: str,
                     seeds: list[int] | None = None,
                     battery: battery_mod.Battery | None = None,
                     media=None) -> dict:
    """``media``: a rlebench.core.media.Media; every Stage B episode is then
    written into it as ep<i>_seed<seed>.mp4, the estimator's own input with
    truth and estimate drawn (render.py). Evidence for a human, produced after
    the report is complete and guarded so it can never change it."""
    if variant not in CHANNELS:
        raise ValueError(f"unknown variant {variant!r}")
    channels = CHANNELS[variant]
    cps = checkpoint_list(variant)
    sub = load_submission(submission_dir, variant)

    missing = (sub.mode == "torch" and sub.model_path is None) or (
        sub.mode == "code"
        and not os.path.exists(os.path.join(submission_dir, "estimator.py")))
    if missing:
        report = aggregate({"G.load": 0.0}, cps, config.GATE_CAP)
        report["errors"] = sub.errors
        report["variant"] = variant
        return report

    bat = battery if battery is not None else battery_mod.build_battery(seeds)
    errors = list(sub.errors)

    # The only gate: can the artifact import/load, construct, and return one
    # finite pose through the declared interface? Out-of-bounds finite poses
    # pass the gate but receive zero kernel credit.
    smoke = _run(sub, [bat.stage_a[0].obs], channels)
    gate_ok = (smoke.ok and np.isfinite(smoke.poses).all()
               and smoke.shape_ids is not None and np.isin(smoke.shape_ids, (0, 1, 2)).all())
    if not gate_ok:
        if smoke.ok:
            errors.append("load job returned no finite pose through the interface")
        else:
            errors.append(f"load job: {smoke.error} {smoke.stderr_tail[-300:]}")
        report = aggregate({"G.load": 0.0}, cps, config.GATE_CAP)
        report["errors"] = errors
        report["variant"] = variant
        return report

    a_frames = bat.stage_a
    job_a = _run(sub, [f.obs for f in a_frames], channels,
                 resets=[True] * len(a_frames))
    a_preds = job_a.poses if job_a.ok else _invalid(len(a_frames))
    a_ids = job_a.shape_ids if job_a.ok else np.full(len(a_frames), -1)
    inference_wall_s = job_a.wall_s
    if not job_a.ok:
        errors.append(f"stage A job: {job_a.error} {job_a.stderr_tail[-300:]}")

    b_preds_parts, b_gts_parts, b_occ_parts, b_ids, b_truth_ids = [], [], [], [], []
    for i, episode in enumerate(bat.episodes):
        frames = episode.frames
        job = _run(sub, [f.obs for f in frames], channels,
                   resets=[True] + [False] * (len(frames) - 1))
        preds = job.poses if job.ok else _invalid(len(frames))
        inference_wall_s += job.wall_s
        b_ids.append(job.shape_ids if job.ok else np.full(len(frames), -1))
        b_truth_ids.append(np.asarray([f.shape_id for f in frames]))
        if not job.ok:
            errors.append(f"stage B episode {i}: {job.error} "
                          f"{job.stderr_tail[-300:]}")
        b_preds_parts.append(preds)
        b_gts_parts.append(np.asarray([f.gt for f in frames], dtype=float))
        b_occ_parts.append(np.asarray([f.occluded for f in frames], dtype=bool))

    b_preds = np.concatenate(b_preds_parts) if b_preds_parts else _invalid(0)
    b_gts = (np.concatenate(b_gts_parts)
             if b_gts_parts else np.zeros((0, 3), dtype=float))
    b_occ = (np.concatenate(b_occ_parts)
             if b_occ_parts else np.zeros(0, dtype=bool))
    a_gts = np.asarray([f.gt for f in a_frames], dtype=float)

    results = {
        "gate_ok": gate_ok,
        "a_preds": a_preds,
        "a_gts": a_gts,
        "b_episode_preds": b_preds_parts,
        "b_episode_gts": b_gts_parts,
        "a_shape_ids": a_ids,
        "a_true_ids": np.asarray([f.shape_id for f in a_frames]),
        "b_shape_ids": b_ids,
        "b_true_ids": b_truth_ids,
    }
    scores = evaluate(variant, results)
    report = aggregate(scores, cps, config.GATE_CAP)
    report["errors"] = errors
    a_groups, b_groups = stage_summaries(results)
    frames_count = len(a_frames) + len(b_gts)
    deduction = efficiency_deduction(frames_count, inference_wall_s)
    accuracy = config.STAGE_A_WEIGHT * scores["A.pose"] + config.STAGE_B_WEIGHT * scores["B.pose"]
    report["accuracy_reward"] = accuracy
    report["reward"] = round(max(0.0, accuracy - deduction), 4)

    a_mt, a_mr = _median_metrics(a_preds, a_gts)
    b_mt, b_mr = _median_metrics(b_preds, b_gts)
    report.update(
        variant=variant,
        stage_a_frames=len(a_frames),
        stage_b_frames=len(b_gts),
        stage_b_occluded_frames=int(b_occ.sum()),
        stage_b_occluded_fraction=round(float(b_occ.mean()), 4) if len(b_occ) else 0.0,
        stage_a_groups=a_groups,
        stage_b_groups=b_groups,
        stage_a_shape_accuracy=float(np.mean(a_ids == results["a_true_ids"])),
        stage_b_shape_accuracy=float(np.mean(np.concatenate(b_ids) == np.concatenate(b_truth_ids))),
        efficiency_deduction=deduction,
        inference_wall_s=inference_wall_s,
        inference_hz=frames_count / inference_wall_s if inference_wall_s > 0 else 0.0,
        abs_median_trans_mm=a_mt,
        abs_median_rot_deg=a_mr,
        stage_b_median_trans_mm=b_mt,
        stage_b_median_rot_deg=b_mr,
        frame_time_s=inference_wall_s / frames_count if frames_count else None,
    )
    if b_occ.any():
        occ_t, _ = _median_metrics(b_preds[b_occ], b_gts[b_occ])
        report["abs_occluded_trans_mm"] = occ_t
    if media is not None:
        from . import render
        media.run("stage_b", render.render_stage_b, bat, b_preds_parts, variant)
    return report
