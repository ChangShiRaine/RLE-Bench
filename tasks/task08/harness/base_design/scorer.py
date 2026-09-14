"""Submission scoring: orchestration + weighted aggregate with Stage-1 gate.

score_submission() is the single entry point the verifier calls. It treats
every file in the submission directory as UNTRUSTED input: the number that
matters is produced by re-running the harness's own scenarios on the agent's
model.

Submission layout (documented in the task's instruction.md):
    robot.xml            submitted model (meshes resolve against the canonical
                         assets/ directory the verifier provides)
    controller.py        shelf-entry policy, run in an isolated process

The dynamic scenario battery uses verifier-owned control.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .. import config
from ..sim.mecanum import make_base_controller
from ..sim.runner import run_all
from ..sim.scenarios import Envelope
from rlebench.core.scoring import aggregate as _core_aggregate
from .checkpoints import CHECKPOINTS, analyze_model, evaluate, score_scenario

GATE_CAP = config.GATE_CAP  # invalid model => downstream meaningless

_PICK_KEYS = ("target_index", "target_name", "ok", "reason", "payload_kg",
              "ee_err", "min_fasm_hold", "shelf_force_max", "controller_fault",
              "elapsed_s", "action_sha256")


@dataclass
class Submission:
    robot_xml: str | None = None
    shelf_controller: str | None = None
    errors: list = field(default_factory=list)


def load_submission(submission_dir: str) -> Submission:
    """Record which files are present. Nothing submitted is imported here."""
    sub = Submission()
    robot = os.path.join(submission_dir, "robot.xml")
    if os.path.exists(robot):
        sub.robot_xml = robot
    else:
        sub.errors.append("robot.xml missing")
    controller = os.path.join(submission_dir, "controller.py")
    if os.path.isfile(controller):
        sub.shelf_controller = controller
    else:
        sub.errors.append("controller.py missing; shelf integration scores zero")
    return sub


def aggregate(checkpoint_scores: dict, checkpoints=None,
              gate_cap: float | None = None) -> dict:
    """task08 defaults over the core weighted-sum (rlebench.core.scoring)."""
    return _core_aggregate(checkpoint_scores,
                           CHECKPOINTS if checkpoints is None else checkpoints,
                           GATE_CAP if gate_cap is None else gate_cap)


def _without_merit(report: dict) -> dict:
    """Task08 reports one headline reward; drop the core's merit mirror."""
    for key in ("merit", "merit_stages", "merits"):
        report.pop(key, None)
    return report


def _score_single_submission(submission_dir: str, arm_reference_xml: str,
                             env: Envelope | None, arm) -> dict:
    """Full verifier pipeline for one canonical arm: aggregate() report plus
    run stats."""
    env = env or Envelope()
    sub = load_submission(submission_dir)
    if sub.robot_xml is None:
        report = _without_merit(aggregate({}))
        report["errors"] = sub.errors
        report["arm"] = arm.name
        return report

    hidden = run_all(sub.robot_xml, controller=make_base_controller,
                     seeds=config.HIDDEN_SEEDS, env=env, arm=arm)
    model_info = analyze_model(sub.robot_xml, arm)

    # Shelf entry under the submitted controller (guarded: a crash scores 0,
    # not a verifier failure).
    pick_probe = None
    if hidden["validity"]["loads"] and sub.shelf_controller is not None:
        try:
            from .controlled_pick import evaluate_controlled_pick
            pick_probe = evaluate_controlled_pick(
                sub.robot_xml, arm_reference_xml, arm, sub.shelf_controller)
        except Exception as e:
            sub.errors.append(f"shelf evaluation failed: {e}")

    scores = evaluate(hidden, model_info, pick_probe)
    report = _without_merit(aggregate(scores))
    report["errors"] = sub.errors
    report["model_info"] = {k: v for k, v in model_info.items()
                            if k in ("footprint", "total_mass", "base_mass", "reach",
                                     "reach_beyond", "battery",
                                     "profile_inventory", "design_efficiency",
                                     "resource_constraints")}
    report["design_headroom"] = model_info.get("design_efficiency", {})
    if pick_probe is not None:
        report["pick_compat"] = dict(
            score=pick_probe["score"],
            deterministic=pick_probe["deterministic"],
            payload_kg=pick_probe["payload_kg"],
            margin_score=pick_probe["margin_score"],
            margin_payload_kg=pick_probe["margin_payload_kg"],
            target_names=pick_probe["target_names"],
            bins=[{k: b.get(k) for k in _PICK_KEYS} for b in pick_probe["bins"]],
            margin_bins=[{k: b.get(k) for k in _PICK_KEYS}
                         for b in pick_probe["margin_bins"]])
    # per-seed hard pass table + dispersion
    passes = {}
    per_seed_scores = {}
    for sid, per_seed in hidden.get("scenarios", {}).items():
        passes[sid] = {int(seed): bool(not res["tip"]) for seed, res in per_seed.items()}
        if sid != "A":
            per_seed_scores[sid] = {int(seed): round(score_scenario(res), 3)
                                    for seed, res in per_seed.items()}
    report["scenario_pass"] = passes
    report["scenario_scores_per_seed"] = per_seed_scores
    dyn = [v for d in per_seed_scores.values() for v in d.values()]
    if dyn:
        report["dynamic_mean"] = round(float(np.mean(dyn)), 4)
        report["dynamic_var"] = round(float(np.var(dyn)), 5)
    return report


def score_submission(submission_dir: str,
                     arm_reference_xml: str,
                     env: Envelope | None = None) -> dict:
    """Score the submitted common base across all canonical arms.

    The same submitted base is composed with trusted Panda, UR5e and xArm7
    models. Every checkpoint is the minimum over arms, so a base must work
    for all three; a strong result on one arm cannot hide a failure on
    another.
    """
    from ..sim.arm_variants import trusted_arm_specs
    arms = trusted_arm_specs(arm_reference_xml)
    arm_reports = {
        name: _score_single_submission(
            submission_dir, arm_reference_xml, env, arm)
        for name, arm in arms.items()
    }
    checkpoint_scores = {
        cp.id: min(float(report.get("checkpoints", {}).get(cp.id, 0.0))
                   for report in arm_reports.values())
        for cp in CHECKPOINTS
    }
    report = _without_merit(aggregate(checkpoint_scores))
    report["aggregation"] = "checkpoint-wise minimum across panda, ur5e, xarm7"
    report["arm_reports"] = arm_reports
    report["errors"] = [
        f"{name}: {error}"
        for name, arm_report in arm_reports.items()
        for error in arm_report.get("errors", [])
    ]
    # The limiting-arm views are convenient for Harbor summaries while the
    # complete evidence remains under arm_reports.
    limiting = min(arm_reports, key=lambda name: arm_reports[name]["reward"])
    report["limiting_arm"] = limiting
    report["model_info"] = arm_reports[limiting].get("model_info", {})
    report["design_headroom"] = arm_reports[limiting].get(
        "design_headroom", {})
    report["pick_compat"] = {
        name: arm_report.get("pick_compat")
        for name, arm_report in arm_reports.items()
    }
    # Scenario evidence, folded across arms the same way the checkpoints are:
    # a (scenario, seed) counts as passed only when every arm survives it, and
    # the per-seed scores keep the worst arm's number.
    passes: dict = {}
    per_seed_scores: dict = {}
    for arm_report in arm_reports.values():
        for sid, per_seed in arm_report.get("scenario_pass", {}).items():
            row = passes.setdefault(sid, {})
            for seed, ok in per_seed.items():
                row[seed] = bool(ok) and bool(row.get(seed, True))
        for sid, per_seed in arm_report.get("scenario_scores_per_seed",
                                           {}).items():
            row = per_seed_scores.setdefault(sid, {})
            for seed, value in per_seed.items():
                row[seed] = min(float(value), float(row.get(seed, value)))
    report["scenario_pass"] = passes
    report["scenario_scores_per_seed"] = per_seed_scores
    dynamic = [v for row in per_seed_scores.values() for v in row.values()]
    if dynamic:
        report["dynamic_mean"] = round(float(np.mean(dynamic)), 4)
        report["dynamic_var"] = round(float(np.var(dynamic)), 5)
    return report
