"""Verifier entry point for task04 (verifier-only).

score_submission(dir) runs the submitted graph on hidden evaluation seeds and
scores HARNESS instrumentation only (invariant #1). There is nothing for a
submission to fabricate: the policy is re-executed here, on machines it has never
seen, and every number comes from evaluator.py.

The submission is a graph of weights rather than code, so no sandbox is
needed: the verifier never imports or runs anything the agent wrote. What is
enforced instead is that the graph honours the contract and fits the parameter
budget as measured from the file.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import config, evaluator, export, motion as motion_mod, robot

_SEEDS_PATH = os.path.join(os.path.dirname(__file__), "eval_seeds.json")


def _load_seeds(seeds=None):
    if seeds is not None:
        return int(seeds["smoke"]), list(seeds["eval"])
    with open(_SEEDS_PATH) as f:
        d = json.load(f)
    return int(d["smoke"]), list(d["eval"])


def episode_score(m: dict, thresholds: config.MotionThresholds | None = None,
                  jerk_cap: float | None = None) -> float:
    """Smoothness is a PENALTY, never a bonus. Paid as credit it hands a free
    0.15 to a policy that outputs a constant action and falls immediately —
    perfectly smooth, and useless.

    Tracking credit is the uncapped multi-scale kernel, whose tighter widths
    keep centimetre-level differences visible.
    Normal scoring passes one motion's threshold record. Calibration instead
    passes the candidate jerk cap explicitly so it can derive a score before
    writing that record back to config.py.
    """
    if thresholds is not None:
        if jerk_cap is not None:
            raise ValueError("pass thresholds or an explicit jerk cap, not both")
        jerk_cap = thresholds.jerk_cap
    if jerk_cap is None:
        raise ValueError("motion thresholds or an explicit jerk cap are required")
    jerk = min(m["action_jerk"] / jerk_cap, 1.0)
    s = (config.W_TRACK * m["tracking_multi"]
         + config.W_SURVIVE * m["survival"]
         - config.W_JERK * jerk
         - config.W_FALL * float(m["fell"]))
    return float(max(0.0, min(1.0, s)))


def episode_mean(report: dict, **caps) -> float:
    """Re-derive a report's reward from its stored per-episode metrics. Lets
    calibration score against new thresholds without re-simulating anything."""
    episodes = report.get("episodes") or []
    if not episodes:
        return 0.0
    return round(float(np.mean([episode_score(e, **caps) for e in episodes])), 4)


def score_submission(submission_dir: str, robot_dir: str, motion_path: str,
                     seeds: dict | None = None, media=None) -> dict:
    """``media``: a rlebench.core.media.Media; each evaluation episode is
    recorded into it as ep<i>.mp4. Rendering reads the state the loop already
    computed and never steps physics; without ffmpeg or GL the writer is a
    no-op the media index reports — evidence for a human must never be able
    to fail a score."""
    smoke_seed, eval_seeds = _load_seeds(seeds)
    thresholds = config.thresholds_for_motion(motion_path)
    report: dict = {"errors": [], "episodes": []}

    problems = export.validate(submission_dir)
    if problems:
        report["errors"].extend(problems)
        report.update(reward=0.0, raw_total=0.0, gated=True,
                      gate={"valid": False, "stable": False, "deterministic": False})
        return report

    meta = export.load_meta(submission_dir)
    clip = motion_mod.Motion.load(motion_path)
    policy = evaluator.OnnxPolicy(submission_dir)

    def run(seed, video=None):
        return evaluator.run_episode(robot.build(robot_dir), clip, policy, seed,
                                     video=video,
                                     history=policy.history).summary()

    # --- gate ---------------------------------------------------------------
    smoke_a = run(smoke_seed)
    smoke_b = run(smoke_seed)
    gate = {
        "valid": True,
        "stable": not smoke_a["diverged"],
        # MuJoCo-C on CPU is bit-reproducible, so a repeated episode must match
        # exactly. A policy that reads a clock or an unseeded RNG shows up here.
        "deterministic": smoke_a == smoke_b,
    }
    if not gate["deterministic"]:
        report["errors"].append(
            "repeated smoke episodes diverged: the policy must be deterministic")

    # --- scored battery -----------------------------------------------------
    scores = []
    for i, seed in enumerate(eval_seeds):
        video = media.video(f"ep{i}.mp4") if media is not None else None
        m = run(seed, video)
        if media is not None:
            media.finish(video)
        m["episode_score"] = round(episode_score(m, thresholds=thresholds), 4)
        report["episodes"].append(m)
        scores.append(m["episode_score"])
        if m["diverged"]:
            report["errors"].append(f"seed {seed}: simulation diverged")

    total = float(np.mean(scores)) if scores else 0.0
    gate_ok = all(gate.values())
    mean = lambda k: float(np.mean([e[k] for e in report["episodes"]]))  # noqa: E731
    report.update(
        reward=round(min(total, config.GATE_CAP) if not gate_ok else total, 4),
        raw_total=round(total, 4),
        gated=not gate_ok,
        gate=gate,
        n_params=meta.get("n_params"),
        obs_history_length=meta.get("obs_history_length"),
        tracking_score=round(mean("tracking_score"), 4),
        tracking_multi=round(mean("tracking_multi"), 4),
        mpjpe_m=round(mean("mpjpe_m"), 4),
        anchor_pos_err_m=round(mean("anchor_pos_err_m"), 4),
        joint_rmse_rad=round(mean("joint_rmse_rad"), 4),
        survival=round(mean("survival"), 4),
        survived_s=round(mean("survived_s"), 3),
        action_jerk=round(mean("action_jerk"), 4),
        falls=int(sum(e["fell"] for e in report["episodes"])),
    )
    return report
