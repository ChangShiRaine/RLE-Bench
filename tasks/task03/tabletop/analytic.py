"""Closed-form optima and counters for the task03 tabletop scenes.

Pure functions on plain data -- no MuJoCo, no robosuite -- so every scoring
normaliser has an analytic unit test (CLAUDE.md golden-first rule).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# --- tower --------------------------------------------------------------------
# A piece is FLAT-STACKABLE when it offers a flat top and a flat bottom in some
# resting orientation: boxes and plates always, cylinders on their ends.
# Spheres and capsules never (no flat face); wedges never (their top is a slope,
# so nothing rests on one and it buys height only as the final piece -- always
# dominated by a flat piece of the same height, so the optimum ignores them).
_STACKABLE = ("box", "plate", "cylinder", "rod")
_UNSTACKABLE = ("sphere", "capsule", "wedge")


@dataclass(frozen=True)
class Piece:
    kind: str        # one of _STACKABLE + _UNSTACKABLE
    height: float    # metres, resting height in its stackable orientation


def tower_optimum(pieces) -> float:
    """Best achievable settled tower height: every flat-stackable piece, once."""
    for p in pieces:
        if p.kind not in _STACKABLE + _UNSTACKABLE:
            raise ValueError(f"unknown piece kind {p.kind!r}")
    return sum(p.height for p in pieces if p.kind in _STACKABLE)


# --- cantilever ---------------------------------------------------------------

def harmonic_overhang(n: int, length: float) -> float:
    """Max overhang of n identical blocks past an edge: (L/2) * H_n."""
    if n < 1:
        raise ValueError("need at least one block")
    return (length / 2.0) * sum(1.0 / k for k in range(1, n + 1))


# --- balance ------------------------------------------------------------------

def weighing_optimum(n: int) -> int:
    """Minimal weighings to find the single heavy item among n: ceil(log3 n)."""
    if n < 2:
        raise ValueError("need at least two items")
    return math.ceil(math.log(n, 3) - 1e-9)


class WeighingCounter:
    """Count weighings from a per-step "both pans loaded" flag.

    One weighing = one DISTINCT stable comparison: the pans stay loaded with
    the same `config` (e.g. the two coin sets) for `min_on` consecutive steps,
    and that config differs from the last one counted. Counting configs rather
    than loaded intervals closes the obvious exploit -- rearranging coins while
    keeping both pans occupied would otherwise merge every comparison into one
    interval. Repeating the exact last comparison is free (it adds nothing).

    With `config=None` each debounced loaded interval is its own comparison
    (interval semantics), debounced on both edges by `min_on`/`min_off`.
    """

    def __init__(self, min_on: int = 25, min_off: int = 25):
        self.min_on = int(min_on)
        self.min_off = int(min_off)
        self.count = 0
        self._on = 0
        self._off = 0
        self._cand = None
        self._counted = None
        self._interval = 0          # token source for config=None

    def update(self, loaded: bool, config=None) -> None:
        if loaded:
            token = config if config is not None else ("interval", self._interval)
            if token == self._cand:
                self._on += 1
            else:
                self._cand, self._on = token, 1
            self._off = 0
            if self._on >= self.min_on and self._cand != self._counted:
                self.count += 1
                self._counted = self._cand
        else:
            self._off += 1
            self._on = 0
            if self._off >= self.min_off:
                self._cand = None
                self._interval += 1


def count_weighings(loaded_flags, min_on: int = 25, min_off: int = 25) -> int:
    counter = WeighingCounter(min_on=min_on, min_off=min_off)
    for flag in loaded_flags:
        counter.update(bool(flag))
    return counter.count


def weighing_efficiency(weighings: int, optimum: int) -> float:
    """1.0 at or under the optimum, then optimum/w -- exact regret."""
    if weighings <= optimum:
        return 1.0
    return optimum / float(weighings)
