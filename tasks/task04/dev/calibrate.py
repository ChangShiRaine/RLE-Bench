"""Score policies through the real evaluator — the tool that pins the reward scale.

Dev-side and verifier-side only. It answers two questions with measurements
rather than guesses: what a given training budget actually buys (score against
wall-clock, over a checkpoint series), and where the reward thresholds should sit
so the oracle lands high and a null policy lands near zero.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from harness import evaluator, export, motion as motion_mod, robot
from rlebench.core.media import Media

DESIGN_SEEDS = (101, 211, 307)


def score_policy(policy, robot_dir: str, clip, seeds, video_dir: str | None = None) -> dict:
    """Mean over seeds of the evaluator's own metrics. One fresh model per
    episode: randomization mutates it, so reusing one would leak a seed's
    machine into the next."""
    media = Media(video_dir) if video_dir else None
    episodes = []
    for i, seed in enumerate(seeds):
        video = media.video(f"ep{i}.mp4") if media else None
        log = evaluator.run_episode(
            robot.build(robot_dir), clip, policy, seed,
            video=video, history=getattr(policy, "history", 1))
        if media:
            media.finish(video)
        episodes.append(log.summary())
    if media:
        media.close()

    keys = ("tracking_score", "tracking_multi", "mpjpe_m", "anchor_pos_err_m",
            "joint_rmse_rad", "survival", "survived_s", "action_jerk")
    out = {k: float(np.mean([e[k] for e in episodes])) for k in keys}
    out["fell"] = float(np.mean([e["fell"] for e in episodes]))
    out["episodes"] = episodes
    return out


def score_series(curve_dir: str, robot_dir: str, motion_path: str, seeds) -> list[dict]:
    """Score every checkpoint in a training run, oldest first."""
    clip = motion_mod.Motion.load(motion_path)
    rows = []
    for name in sorted(os.listdir(curve_dir)):
        path = os.path.join(curve_dir, name)
        if not os.path.isdir(path):
            continue
        problems = export.validate(path)
        if problems:
            rows.append({"checkpoint": name, "invalid": problems})
            continue
        meta = json.load(open(os.path.join(path, "policy_meta.json")))
        result = score_policy(evaluator.OnnxPolicy(path), robot_dir, clip, seeds)
        result.pop("episodes")
        rows.append({"checkpoint": name, "minutes": meta.get("minutes"),
                     "iterations": meta.get("iterations"), **result})
    return rows


def update_motion_threshold(out_path: str, motion_name: str, values: dict) -> None:
    """Replace exactly one threshold record and leave the other tasks intact."""
    prefix = f'    "{motion_name}": MotionThresholds('
    replacement = [
        prefix,
        f'        jerk_cap={values["jerk_cap"]:.4f},',
        f'        oracle_ref_score={values["oracle_score"]:.4f},',
        f'        null_ref_score={values["null_score"]:.4f},',
        '        calibrated=True,',
        '    ),',
    ]
    with open(out_path) as f:
        lines = f.read().splitlines()
    matches = [i for i, line in enumerate(lines) if line == prefix]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one config entry for {motion_name!r}, found {len(matches)}")
    start = matches[0]
    try:
        end = lines.index('    ),', start) + 1
    except ValueError as exc:
        raise RuntimeError(f"unterminated config entry for {motion_name!r}") from exc
    lines[start:end] = replacement
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, out_path)


def null_submission(path: str) -> str:
    """A policy that always outputs zeros, exported through the real path so the
    floor is measured with exactly the machinery a submission uses."""
    import torch

    from harness import export as export_mod
    from harness import spec

    actor = torch.nn.Linear(spec.OBS_DIM, spec.N_ACTIONS)
    with torch.no_grad():
        actor.weight.zero_()
        actor.bias.zero_()
    export_mod.export(actor, path)
    return path


def emit(oracle_dir: str, robot_dir: str, motion_path: str, out_path: str) -> dict:
    """Measure both references and update only this motion's threshold record."""
    import tempfile

    from harness import config, scorer

    with tempfile.TemporaryDirectory() as tmp:
        null_dir = null_submission(os.path.join(tmp, "null"))
        null = scorer.score_submission(null_dir, robot_dir, motion_path)
        oracle = scorer.score_submission(oracle_dir, robot_dir, motion_path)

    values = dict(
        oracle_track=oracle["tracking_multi"], null_track=null["tracking_multi"],
        oracle_survival=oracle["survival"], null_survival=null["survival"],
        oracle_jerk=oracle["action_jerk"],
        jerk_cap=max(oracle["action_jerk"] * 2.0, 1e-3),
    )
    # The references were scored under the previous thresholds, so the guards
    # are re-derived from their stored episode metrics under the new jerk cap.
    values["oracle_score"] = scorer.episode_mean(oracle, jerk_cap=values["jerk_cap"])
    values["null_score"] = scorer.episode_mean(null, jerk_cap=values["jerk_cap"])
    motion_name = config.motion_name(motion_path)
    # Resolve first so a typo cannot add a sixth, unreferenced calibration.
    config.thresholds_for_motion(motion_path)
    update_motion_threshold(out_path, motion_name, values)
    return {"oracle": oracle, "null": null, "written": out_path,
            "motion_name": motion_name, **values}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--curve-dir", default=None)
    p.add_argument("--robot-dir", default="third_party/motiontrack/unitree_g1")
    p.add_argument("--motion",
                   default="third_party/motiontrack/motions/dance1_subject2.npz")
    p.add_argument("--seeds", type=int, nargs="+", default=list(DESIGN_SEEDS))
    p.add_argument("--json", default=None)
    p.add_argument("--emit", metavar="ORACLE_DIR",
                   help="measure oracle + null and update this motion in config.py")
    a = p.parse_args(argv)

    if a.emit:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness", "config.py")
        result = emit(a.emit, a.robot_dir, a.motion, out)
        print(f"oracle: reward {result['oracle']['raw_total']:.4f} "
              f"multi {result['oracle_track']:.4f} "
              f"survival {result['oracle_survival']:.3f} "
              f"jerk {result['oracle_jerk']:.4f}")
        print(f"null:   reward {result['null']['raw_total']:.4f} "
              f"multi {result['null_track']:.4f}")
        print(f"updated {out}: {result['motion_name']} "
              f"JERK_CAP {result['jerk_cap']:.4f}")
        return

    rows = score_series(a.curve_dir, a.robot_dir, a.motion, a.seeds)
    print(f"{'min':>6} {'iters':>7} {'track':>7} {'mpjpe':>7} {'alive_s':>8} "
          f"{'survival':>9} {'fell':>5}")
    for r in rows:
        if r.get("invalid"):
            print(f"{r['checkpoint']:>6} INVALID {r['invalid']}")
            continue
        print(f"{r['minutes']:6.1f} {r['iterations']:7d} {r['tracking_multi']:7.4f} "
              f"{r['mpjpe_m']:7.3f} {r['survived_s']:8.2f} {r['survival']:9.3f} "
              f"{r['fell']:5.2f}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
