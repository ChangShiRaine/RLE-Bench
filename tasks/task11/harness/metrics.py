"""Pure scoring primitives for task11 — no MuJoCo, fully unit-testable.

Trackers are fed once per control tick from harness-side state and latch:
a floor drop counts once per box, a damage event once per box.
"""
from __future__ import annotations

import numpy as np

from . import spec


class FullDetector:
    """The bin is full once FULL_STREAK consecutive boxes leave the cell
    unpicked while the fit rule says none of them fits. A pass that would
    have fitted, a grasp or a newly committed box resets the streak."""

    def __init__(self, streak: int = spec.FULL_STREAK):
        self.need = int(streak)
        self.streak = 0
        self.full_t: float | None = None
        self.passed = 0
        self.missed_fitting = 0

    def on_pass(self, t: float, fitted: bool) -> bool:
        self.passed += 1
        if fitted:
            self.missed_fitting += 1
            self.streak = 0
        else:
            self.streak += 1
        if self.full_t is None and self.streak >= self.need:
            self.full_t = float(t)
        return self.full

    def reset(self) -> None:
        self.streak = 0

    @property
    def full(self) -> bool:
        return self.full_t is not None


class FloorTracker:
    """Latches boxes whose COM ever drops below FLOOR_Z."""

    def __init__(self, floor_z: float = spec.FLOOR_Z):
        self.floor_z = float(floor_z)
        self.dropped: set[int] = set()

    def update(self, positions: dict[int, np.ndarray]) -> list[int]:
        newly = [i for i, p in positions.items()
                 if i not in self.dropped and p[2] < self.floor_z]
        self.dropped.update(newly)
        return newly


class DamageTracker:
    """Latches a box as damaged when its contact force exceeds the crush
    limit for ``consecutive`` ticks, or when it hits something while moving
    faster than the drop-speed limit (a toss or a drop from height)."""

    def __init__(self, force_limit: float, consecutive: int,
                 impact_speed: float):
        self.force_limit = float(force_limit)
        self.consecutive = int(consecutive)
        self.impact_speed = float(impact_speed)
        self._streak: dict[int, int] = {}
        self._prev_speed: dict[int, float] = {}
        self._prev_contact: dict[int, bool] = {}
        self.damaged: dict[int, str] = {}
        self.peak_sustained = 0.0
        self._prev_force: dict[int, float] = {}

    def update(self, forces: dict[int, float], speeds: dict[int, float],
               contact: dict[int, bool], free: set[int]) -> list[int]:
        """``free``: boxes that are neither held nor riding the belt."""
        newly = []
        for i, f in forces.items():
            self.peak_sustained = max(self.peak_sustained,
                                      min(f, self._prev_force.get(i, 0.0)))
            self._prev_force[i] = f
            self._streak[i] = self._streak.get(i, 0) + 1 \
                if f > self.force_limit else 0
            hit = (i in free and contact.get(i, False)
                   and not self._prev_contact.get(i, False)
                   and self._prev_speed.get(i, 0.0) > self.impact_speed)
            if i not in self.damaged:
                if self._streak[i] >= self.consecutive:
                    self.damaged[i] = "crush"
                    newly.append(i)
                elif hit:
                    self.damaged[i] = "impact"
                    newly.append(i)
            self._prev_speed[i] = speeds.get(i, 0.0)
            self._prev_contact[i] = contact.get(i, False)
        return newly


class BinHitTracker:
    """Discrete tool strikes on the tote: sustained force above the limit is
    one hit; the force must stay below it ``rearm`` ticks before the next."""

    def __init__(self, force_limit: float, consecutive: int = 2,
                 rearm: int = 5):
        self.force_limit = float(force_limit)
        self.consecutive = int(consecutive)
        self.rearm = int(rearm)
        self._over = self._under = 0
        self._armed = True
        self._prev = 0.0
        self.hits = 0
        self.peak_sustained = 0.0

    def update(self, force: float) -> bool:
        f = float(force)
        self.peak_sustained = max(self.peak_sustained, min(f, self._prev))
        self._prev = f
        if f > self.force_limit:
            self._over += 1
            self._under = 0
            if self._armed and self._over >= self.consecutive:
                self.hits += 1
                self._armed = False
                return True
        else:
            self._over = 0
            self._under += 1
            if self._under >= self.rearm:
                self._armed = True
        return False


def episode_metrics(*, n_packed: int, packed_volume: float, attempts: int,
                    grasps: int, commit_times: list[float], full_t,
                    floor_drops: int, damage_events: int, bin_hits: int,
                    passed: int, missed_fitting: int, spawned: int,
                    budget_t: float = spec.EPISODE_T) -> dict:
    """Headline metrics from the harness's own counts.

    Throughput is paced against the makespan (last packed box) only when the
    bin was declared full; otherwise against the whole budget, so stopping
    early never inflates it."""
    makespan = max(commit_times) if commit_times else None
    denom = makespan if full_t is not None and makespan else budget_t
    return {
        "n_packed": n_packed,
        "utilization": packed_volume / spec.BIN_VOLUME,
        "throughput_ppm": 60.0 * n_packed / denom if denom else 0.0,
        "makespan_s": makespan,
        "bin_full": full_t is not None,
        "full_t": full_t,
        "pick_attempts": attempts,
        "grasps": grasps,
        "pick_success_rate": grasps / attempts if attempts else 0.0,
        "pick_place_success_rate": min(n_packed / attempts, 1.0)
        if attempts else 0.0,
        "floor_drops": floor_drops,
        "floor_drop_rate": floor_drops / grasps if grasps else 0.0,
        "damage_events": damage_events,
        "bin_hits": bin_hits,
        "boxes_spawned": spawned,
        "boxes_passed": passed,
        "missed_fitting": missed_fitting,
    }
