"""Pure scoring primitives for task07 — no MuJoCo, fully unit-testable.

Trackers are incremental (fed once per control tick from harness-side sim
state) and LATCH: a cleared part stays cleared, a floor drop counts once, a
damage event counts once per part. The speed merit is bounded by clearance —
uncleared parts contribute zero.
"""
from __future__ import annotations

import numpy as np

from . import spec


def in_aabb(p, lo, hi) -> bool:
    p, lo, hi = np.asarray(p), np.asarray(lo), np.asarray(hi)
    return bool(np.all(p >= lo) and np.all(p <= hi))


class ClearanceTracker:
    """Latches a part as cleared after CLEAR_REST_TICKS consecutive ticks
    at rest inside the drop zone. update() returns newly cleared indices.

    EXCLUSIVE ZONE (singulated delivery): the belt takes one part at a
    time — a part's rest streak only advances while it is the SOLE
    unlatched, unheld part inside the zone. A double delivery blocks both
    parts until one is picked back up (held parts neither latch nor
    occupy)."""

    def __init__(self, n_parts: int,
                 zone_lo=spec.ZONE_LO, zone_hi=spec.ZONE_HI,
                 speed_thresh: float = spec.CLEAR_SPEED,
                 rest_ticks: int = spec.CLEAR_REST_TICKS,
                 exclusive: bool = True):
        self.exclusive = exclusive
        self.n = n_parts
        self.lo = np.asarray(zone_lo, dtype=float)
        self.hi = np.asarray(zone_hi, dtype=float)
        self.speed_thresh = float(speed_thresh)
        self.rest_ticks = int(rest_ticks)
        self._streak = np.zeros(n_parts, dtype=int)
        self.clear_time = {}          # part index -> sim time of latch

    def update(self, t: float, positions: np.ndarray,
               speeds: np.ndarray,
               exclude: set[int] | None = None) -> list[int]:
        """``exclude``: parts currently HELD by the magnet — a dangled part
        is stationary but is not on the belt; only released, resting parts
        can be carried away. Exclusion resets the rest streak."""
        exclude = exclude or set()
        occupants = [i for i in range(self.n)
                     if i not in self.clear_time and i not in exclude
                     and in_aabb(positions[i], self.lo, self.hi)]
        crowded = self.exclusive and len(occupants) > 1
        newly = []
        for i in range(self.n):
            if i in self.clear_time:
                continue
            ok = (not crowded
                  and i not in exclude
                  and speeds[i] < self.speed_thresh
                  and in_aabb(positions[i], self.lo, self.hi))
            self._streak[i] = self._streak[i] + 1 if ok else 0
            if self._streak[i] >= self.rest_ticks:
                self.clear_time[i] = float(t)
                newly.append(i)
        return newly

    @property
    def cleared(self) -> int:
        return len(self.clear_time)


class FloorTracker:
    """Latches parts whose COM ever goes below FLOOR_Z (outside every
    surface a part legitimately rests on — bin floor and belt both sit
    well above it)."""

    def __init__(self, n_parts: int, floor_z: float = spec.FLOOR_Z):
        self.n = n_parts
        self.floor_z = float(floor_z)
        self.dropped: set[int] = set()

    def update(self, positions: np.ndarray,
               exclude: set[int] | None = None) -> list[int]:
        newly = []
        exclude = exclude or set()
        for i in range(self.n):
            if i in self.dropped or i in exclude:
                continue
            if positions[i][2] < self.floor_z:
                self.dropped.add(i)
                newly.append(i)
        return newly


class DamageTracker:
    """Latches a damage event when a part's contact force exceeds the
    threshold for ``consecutive`` control ticks (sustained crush/slam, not
    a solver impulse spike)."""

    def __init__(self, n_parts: int, force_limit: float,
                 consecutive: int = 2):
        self.n = n_parts
        self.force_limit = float(force_limit)
        self.consecutive = int(consecutive)
        self._streak = np.zeros(n_parts, dtype=int)
        self.peak = np.zeros(n_parts)
        self.damaged: set[int] = set()

    def update(self, forces: np.ndarray) -> list[int]:
        newly = []
        for i in range(self.n):
            f = float(forces[i])
            if f > self.peak[i]:
                self.peak[i] = f
            over = f > self.force_limit
            self._streak[i] = self._streak[i] + 1 if over else 0
            if self._streak[i] >= self.consecutive and i not in self.damaged:
                self.damaged.add(i)
                newly.append(i)
        return newly


def speed_merit(clear_times: dict[int, float], n_parts: int,
                budget_t: float = spec.EPISODE_T) -> float:
    """Mean over ALL parts of max(0, 1 - t_clear/T); uncleared parts = 0."""
    if n_parts == 0:
        return 0.0
    total = sum(max(0.0, 1.0 - t / budget_t) for t in clear_times.values())
    return total / n_parts


def episode_metrics(n_parts: int, clear_times: dict[int, float],
                    floor_drops: int, damage_events: int,
                    budget_t: float = spec.EPISODE_T) -> dict:
    cleared = len(clear_times)
    makespan = max(clear_times.values()) if clear_times else None
    return {
        "n_parts": n_parts,
        "parts_cleared": cleared,
        "clear_frac": cleared / n_parts if n_parts else 0.0,
        "speed_merit": speed_merit(clear_times, n_parts, budget_t),
        "makespan_s": makespan,
        "throughput_ppm": (60.0 * cleared / makespan
                           if makespan and makespan > 0 else 0.0),
        "floor_drops": floor_drops,
        "damage_events": damage_events,
        "clear_times": {int(k): round(float(v), 3)
                        for k, v in sorted(clear_times.items())},
    }


class BinHitTracker:
    """Counts discrete tool-strikes on the bin: sustained (>= consecutive
    ticks) contact force above the threshold is one event; the force must
    fall below the threshold for ``rearm`` ticks before another event can
    be counted — continuous grinding is one hit, separate strikes are
    separate hits."""

    def __init__(self, force_limit: float, consecutive: int = 2,
                 rearm: int = 5):
        self.force_limit = float(force_limit)
        self.consecutive = int(consecutive)
        self.rearm = int(rearm)
        self._over = 0
        self._under = 0
        self._armed = True
        self.hits = 0
        self.peak = 0.0
        self._prev = 0.0
        self.peak_sustained = 0.0

    def update(self, force: float) -> bool:
        """Feed one control tick's tool-bin contact force; True on a new hit."""
        f = float(force)
        self.peak = max(self.peak, f)
        self.peak_sustained = max(self.peak_sustained, min(f, self._prev))
        self._prev = f
        new = False
        if f > self.force_limit:
            self._over += 1
            self._under = 0
            if self._armed and self._over >= self.consecutive:
                self.hits += 1
                self._armed = False
                new = True
        else:
            self._over = 0
            self._under += 1
            if self._under >= self.rearm:
                self._armed = True
        return new
