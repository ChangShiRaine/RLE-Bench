"""Checkpoint scoring: the weighted sum with a validity gate.

Extracted verbatim from the task08 harness (base_design/{checkpoints,scorer})
where three families had come to import it. Task-agnostic on purpose: every
default that was task08 calibration (checkpoint table, gate cap, constraint
band) is a required argument here; each family's harness supplies its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Checkpoint:
    id: str
    stage: str
    weight: float
    gate: bool = False          # failing it caps the total at the gate cap
    # What counts as failing. "any_shortfall": a pass/fail gate, anything
    # below full credit caps. "no_credit": the checkpoint also scores
    # continuously, so only a total failure caps.
    gate_rule: str = "any_shortfall"


def aggregate(checkpoint_scores: dict, checkpoints, gate_cap: float,
              checkpoint_merits: dict | None = None) -> dict:
    """Weighted sum with the validity gate; per-stage breakdown included.

    checkpoint_merits (optional): per-checkpoint uncapped excellence values.
    When supplied, an uncapped `merit` total + per-stage `merit_stages` are
    reported alongside the [0,1] `reward` — merit == 1.0 means "at the
    full-credit bar" everywhere; > 1.0 means the design beats it. When
    omitted, merit mirrors the scores (so merit == raw_total)."""
    merits = checkpoint_scores if checkpoint_merits is None else checkpoint_merits
    by_stage: dict[str, float] = {}
    merit_by_stage: dict[str, float] = {}
    total = 0.0
    merit_total = 0.0
    gate_failed = []
    for cp in checkpoints:
        s = float(checkpoint_scores.get(cp.id, 0.0))
        m = float(merits.get(cp.id, s))
        total += cp.weight * s
        merit_total += cp.weight * m
        by_stage[cp.stage] = by_stage.get(cp.stage, 0.0) + cp.weight * s
        merit_by_stage[cp.stage] = merit_by_stage.get(cp.stage, 0.0) + cp.weight * m
        # Gates are the runnable predicates (the model loads, the interface
        # and DoF layout are there): nothing downstream is measurable without
        # them. Every other validity checkpoint scores continuously and only
        # loses its own credit, so a marginal physical violation does not
        # collapse the total.
        if cp.gate and (s <= 0.0 if cp.gate_rule == "no_credit" else s < 1.0):
            gate_failed.append(cp.id)
    gated = bool(gate_failed)
    reward = min(total, gate_cap) if gated else total
    return dict(reward=round(reward, 4), raw_total=round(total, 4),
                merit=round(merit_total, 4),
                gated=gated, gate_failed=gate_failed,
                stages={k: round(v, 4) for k, v in by_stage.items()},
                merit_stages={k: round(v, 4) for k, v in merit_by_stage.items()},
                checkpoints={cp.id: round(float(checkpoint_scores.get(cp.id, 0.0)), 4)
                             for cp in checkpoints},
                merits={cp.id: round(float(merits.get(cp.id,
                                     checkpoint_scores.get(cp.id, 0.0))), 4)
                        for cp in checkpoints})


def constraint_credit(ratio: float | None, band: float) -> float:
    """Continuous credit for one satisfy-or-violate constraint.

    `ratio` is actual/limit oriented so that > 1 is a violation. Full credit
    while the constraint holds, linear decay across the band, zero past it.
    """
    if ratio is None or not math.isfinite(ratio) or ratio < 0.0:
        return 0.0
    if ratio <= 1.0:
        return 1.0
    if band <= 1.0:
        return 0.0
    return float(max(0.0, 1.0 - (ratio - 1.0) / (band - 1.0)))


def share_credit(credits) -> float:
    """Mean credit over a collection of independent sub-checks; 0.0 when the
    collection is empty (nothing measured is not the same as nothing wrong)."""
    values = [float(c) for c in credits]
    if not values:
        return 0.0
    return float(np.clip(np.mean(values), 0.0, 1.0))
