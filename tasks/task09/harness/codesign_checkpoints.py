"""Task09 checkpoints: validity + hardware / software / co-design.

Ten equally weighted rewards: motion clearance, three hardware metrics,
two adapted-software metrics, and four co-design metrics. Basic validity
checks deduct up to 0.10 rather than awarding points.

Continuous checkpoints score clip(bar/x, 0, H)/H (smaller-is-better; x/bar
for bigger-is-better) with H = REWARD_HEADROOM. H1 uses H = 1, so its
existing residual bar is the full-credit target.

Validity checkpoints are continuous too: a numeric constraint scores
constraint_credit(actual/limit), full credit while it holds and decaying to
zero across CONSTRAINT_BAND; a collection of independent sub-checks scores
the share that passes.
"""
from __future__ import annotations

import math

import numpy as np

from . import codesign_variant_config as ccfg
from rlebench.core.scoring import Checkpoint, aggregate, share_credit
from rlebench.core.scoring import constraint_credit as _constraint_credit

CONSTRAINT_BAND = 1.5


def constraint_credit(ratio):
    return _constraint_credit(ratio, CONSTRAINT_BAND)


# Preserve the relative weights of the seven basic checks within a 10% deduction.
VALIDITY_PENALTIES = {
    "C1.load": 0.02 * 0.10 / 0.19,
    "C1.printability": 0.04 * 0.10 / 0.19,
    "C1.structure": 0.04 * 0.10 / 0.19,
    "C1.settle": 0.04 * 0.10 / 0.19,
    "C1.budget": 0.02 * 0.10 / 0.19,
    "C1.springs_legal": 0.02 * 0.10 / 0.19,
    "C1.buildable": 0.01 * 0.10 / 0.19,
}

CHECKPOINTS_CD = [
    Checkpoint("C1.load", "validity", 0.0),
    Checkpoint("C1.printability", "validity", 0.0),
    Checkpoint("C1.structure", "validity", 0.0),
    Checkpoint("C1.settle", "validity", 0.0),
    Checkpoint("C1.budget", "validity", 0.0),
    Checkpoint("C1.springs_legal", "validity", 0.0),
    Checkpoint("C1.buildable", "validity", 0.0),
    Checkpoint("C1.motion_clearance", "validity", 0.10),
    Checkpoint("H1.device_residual", "hardware", 0.10),
    Checkpoint("H2.passive_hold", "hardware", 0.10),
    Checkpoint("H3.passive_backdrive", "hardware", 0.10),
    Checkpoint("S2.adapted_hold", "software", 0.10),
    Checkpoint("S2.adapted_backdrive", "software", 0.10),
    Checkpoint("B1.combined_hold", "codesign", 0.10),
    Checkpoint("B2.combined_backdrive", "codesign", 0.10),
    Checkpoint("B3.headroom", "codesign", 0.10),
    Checkpoint("B4.push_recovery", "codesign", 0.10),
]

H = float(ccfg.REWARD_HEADROOM)


def aggregate_codesign(scores: dict) -> dict:
    """Ten positive rewards minus basic-validity deductions, floored at zero."""
    report = aggregate(scores, CHECKPOINTS_CD, gate_cap=1.0)
    deductions = {
        cid: weight * (1.0 - float(np.clip(scores.get(cid, 0.0), 0.0, 1.0)))
        for cid, weight in VALIDITY_PENALTIES.items()
    }
    penalty = min(0.10, sum(deductions.values()))
    total = sum(cp.weight * scores.get(cp.id, 0.0) for cp in CHECKPOINTS_CD) - penalty
    reward = max(0.0, min(total, 1.0))
    report.update(reward=round(reward, 4), raw_total=round(total, 4),
                  merit=round(total, 4), validity_deduction=round(penalty, 6),
                  deductions={cid: round(value, 6) for cid, value in deductions.items()})
    report["stages"]["validity"] = round(
        0.10 * scores.get("C1.motion_clearance", 0.0) - penalty, 4)
    report["merit_stages"] = dict(report["stages"])
    return report


def _cont_down(x: float, bar: float, headroom: float | None = None) -> float:
    """Smaller-is-better continuous credit: (bar/x)/H, clipped to [0, 1].
    x == bar -> 1/H; x == bar/H (H x better) -> 1; monotone everywhere."""
    h = H if headroom is None else float(headroom)
    if not math.isfinite(h) or h <= 0:
        return 0.0
    if x is None or not math.isfinite(x) or x < 0:
        return 0.0
    if x == 0.0:
        return 1.0
    return float(np.clip(bar / x, 0.0, h) / h)


def _cont_up(x: float, bar: float) -> float:
    """Bigger-is-better continuous credit: (x/bar)/H, clipped to [0, 1]."""
    if x is None or not math.isfinite(x) or bar <= 0:
        return 0.0
    return float(np.clip(x / bar, 0.0, H) / H)


def evaluate_codesign(validity: dict, springs_share: float, batt: dict,
                      s_tier: dict, buildable: dict | None = None) -> dict:
    """-> scores keyed by checkpoint id.

    batt:   codesign_battery() aggregate on the AGENT's hardware with its
            best software (adapted where available)
    s_tier: dict(nominal_droop=..., adapted_droop=..., adapted_effort=...)
            measured on the reference arm; None values = software absent/failed
    """
    scores: dict = {}

    # --- validity ------------------------------------------------------------
    # The scorer skips the submitted-arm battery if loading or structure
    # fails. Basic checks themselves only contribute bounded deductions.
    inertia = validity.get("inertia", {})
    scores["C1.load"] = (
        share_credit(1.0 if b.get("ok") else 0.0 for b in inertia.values())
        if validity.get("loads") else 0.0)
    scores["C1.structure"] = 1.0 if validity.get("structure", {}).get("ok") else 0.0
    scores["C1.printability"] = constraint_credit(
        validity.get("density", {}).get("worst_ratio"))
    scores["C1.settle"] = constraint_credit(
        validity.get("settle", {}).get("speed_ratio"))
    scores["C1.budget"] = min(
        constraint_credit(validity.get("mass_ratio")),
        constraint_credit(validity.get("link_mass_worst_ratio")))
    scores["C1.springs_legal"] = float(np.clip(springs_share, 0.0, 1.0))
    build = buildable or {}
    scores["C1.buildable"] = share_credit(
        [1.0] * max(int(build.get("n_checks", 0)) - int(build.get("n_failed", 0)), 0)
        + [0.0] * int(build.get("n_failed", 0)))

    motion = validity.get("motion_clearance", {})
    checked = int(motion.get("n_checks", 0))
    failed = int(motion.get("n_failed", checked))
    scores["C1.motion_clearance"] = (
        float(np.clip((checked - failed) / checked, 0.0, 1.0)) if checked else 0.0)

    # --- hardware (passive, feedforward off) ---------------------------------
    scores["H1.device_residual"] = _cont_down(
        batt.get("device_residual") if batt else None, ccfg.H1_RESIDUAL_BAR,
        headroom=1.0)
    scores["H2.passive_hold"] = _cont_down(
        batt.get("passive_droop") if batt else None, ccfg.H2_DROOP_BAR)
    scores["H3.passive_backdrive"] = _cont_down(
        batt.get("passive_effort") if batt else None, ccfg.H3_EFFORT_BAR)

    # --- software (on the harness reference arm) ------------------------------
    scores["S2.adapted_hold"] = _cont_down(
        s_tier.get("adapted_droop"), ccfg.S2_DROOP_BAR)
    scores["S2.adapted_backdrive"] = _cont_down(
        s_tier.get("adapted_effort"), ccfg.S2_EFFORT_BAR)

    # --- codesign (hardware + software, held-out instances) -------------------
    scores["B1.combined_hold"] = _cont_down(
        batt.get("sw_droop") if batt else None, ccfg.B1_DROOP_BAR)
    scores["B2.combined_backdrive"] = _cont_down(
        batt.get("sw_effort") if batt else None, ccfg.B2_EFFORT_BAR)
    scores["B3.headroom"] = _cont_up(
        batt.get("headroom") if batt else None, ccfg.B3_HEADROOM_BAR)
    scores["B4.push_recovery"] = _cont_down(
        batt.get("poke_final") if batt else None, ccfg.B4_POKE_BAR)

    return scores
