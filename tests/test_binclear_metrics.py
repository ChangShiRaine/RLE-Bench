"""Pure-math tests for task07 scoring primitives (no MuJoCo)."""
import numpy as np

from harness import metrics, spec
from harness.metrics import (BinHitTracker, ClearanceTracker,
                                        DamageTracker, FloorTracker,
                                        in_aabb, speed_merit)

ZONE_LO, ZONE_HI = np.asarray(spec.ZONE_LO), np.asarray(spec.ZONE_HI)
IN_ZONE = (ZONE_LO + ZONE_HI) / 2
OUT_ZONE = np.array([spec.BIN_POS[0], spec.BIN_POS[1], 0.2])


def _tick(tr, t, pos, speed):
    return tr.update(t, np.asarray([pos]), np.asarray([speed]))


# ---------------------------------------------------------------------------
# in_aabb
# ---------------------------------------------------------------------------
def test_in_aabb_boundaries():
    assert in_aabb(ZONE_LO, ZONE_LO, ZONE_HI)
    assert in_aabb(ZONE_HI, ZONE_LO, ZONE_HI)
    assert not in_aabb(ZONE_HI + 1e-6, ZONE_LO, ZONE_HI)


# ---------------------------------------------------------------------------
# ClearanceTracker
# ---------------------------------------------------------------------------
def test_clearance_latches_after_rest():
    tr = ClearanceTracker(1)
    for k in range(spec.CLEAR_REST_TICKS - 1):
        assert _tick(tr, k * 0.05, IN_ZONE, 0.0) == []
    assert _tick(tr, 0.45, IN_ZONE, 0.0) == [0]
    assert tr.cleared == 1
    # latched: no double-count, even if it later leaves the zone
    assert _tick(tr, 0.50, OUT_ZONE, 1.0) == []
    assert tr.cleared == 1


def test_clearance_movement_resets_streak():
    tr = ClearanceTracker(1)
    for k in range(spec.CLEAR_REST_TICKS - 1):
        _tick(tr, k * 0.05, IN_ZONE, 0.0)
    _tick(tr, 0.45, IN_ZONE, spec.CLEAR_SPEED * 2)   # still moving
    for k in range(spec.CLEAR_REST_TICKS - 1):
        assert _tick(tr, 0.5 + k * 0.05, IN_ZONE, 0.0) == []
    assert tr.cleared == 0


def test_clearance_flythrough_never_latches():
    tr = ClearanceTracker(1)
    for k in range(3 * spec.CLEAR_REST_TICKS):
        _tick(tr, k * 0.05, IN_ZONE, 0.5)   # flung across the zone
    assert tr.cleared == 0


def test_clearance_rest_outside_zone_never_latches():
    tr = ClearanceTracker(1)
    for k in range(3 * spec.CLEAR_REST_TICKS):
        _tick(tr, k * 0.05, OUT_ZONE, 0.0)
    assert tr.cleared == 0


def test_clearance_records_latch_time():
    tr = ClearanceTracker(2)
    for k in range(spec.CLEAR_REST_TICKS):
        tr.update(k * 0.05, np.asarray([IN_ZONE, OUT_ZONE]),
                  np.asarray([0.0, 0.0]))
    t_latch = (spec.CLEAR_REST_TICKS - 1) * 0.05
    assert tr.clear_time == {0: t_latch}


# ---------------------------------------------------------------------------
# FloorTracker
# ---------------------------------------------------------------------------
def test_floor_latch_and_exclude():
    tr = FloorTracker(2)
    on_floor = np.array([[0.9, -0.9, 0.01], [0.5, 0.5, 0.2]])
    assert tr.update(on_floor) == [0]
    assert tr.update(on_floor) == []          # counted once
    tr2 = FloorTracker(1)
    assert tr2.update(np.array([[0.9, -0.9, 0.01]]), exclude={0}) == []


# ---------------------------------------------------------------------------
# DamageTracker
# ---------------------------------------------------------------------------
def test_damage_needs_sustained_force():
    tr = DamageTracker(1, force_limit=100.0, consecutive=2)
    assert tr.update(np.array([500.0])) == []      # single solver spike
    assert tr.update(np.array([50.0])) == []
    assert tr.update(np.array([150.0])) == []
    assert tr.update(np.array([150.0])) == [0]     # sustained
    assert tr.update(np.array([150.0])) == []      # latched once
    assert tr.peak[0] == 500.0


# ---------------------------------------------------------------------------
# speed_merit / episode_metrics
# ---------------------------------------------------------------------------
def test_speed_merit_bounds_and_values():
    T = spec.EPISODE_T
    assert speed_merit({}, 10) == 0.0
    assert speed_merit({0: 0.0}, 1) == 1.0
    assert speed_merit({0: T}, 1) == 0.0
    # two of four parts cleared at half budget: 2 * 0.5 / 4
    assert np.isclose(speed_merit({0: T / 2, 1: T / 2}, 4), 0.25)
    # speed merit can never exceed clear fraction
    ct = {i: 1.0 for i in range(3)}
    assert speed_merit(ct, 6) <= 3 / 6


def test_episode_metrics_assembly():
    m = metrics.episode_metrics(4, {0: 30.0, 1: 60.0}, floor_drops=1,
                                damage_events=2)
    assert m["parts_cleared"] == 2 and m["clear_frac"] == 0.5
    assert m["makespan_s"] == 60.0
    assert np.isclose(m["throughput_ppm"], 2.0)
    assert m["floor_drops"] == 1 and m["damage_events"] == 2


def test_episode_metrics_no_clears():
    m = metrics.episode_metrics(5, {}, 0, 0)
    assert m["makespan_s"] is None
    assert m["throughput_ppm"] == 0.0
    assert m["clear_frac"] == 0.0


# ---------------------------------------------------------------------------
# BinHitTracker
# ---------------------------------------------------------------------------
def test_bin_hit_needs_sustained_force():
    tr = BinHitTracker(100.0, consecutive=2, rearm=5)
    assert not tr.update(500.0)       # single solver spike
    assert not tr.update(10.0)
    assert not tr.update(150.0)
    assert tr.update(150.0)           # sustained -> one hit
    assert tr.hits == 1


def test_bin_hit_grinding_counts_once_strikes_count_separately():
    tr = BinHitTracker(100.0, consecutive=2, rearm=5)
    for _ in range(20):               # long continuous grind
        tr.update(300.0)
    assert tr.hits == 1
    for _ in range(4):                # not below long enough to re-arm
        tr.update(10.0)
    tr.update(300.0); tr.update(300.0)
    assert tr.hits == 1
    for _ in range(5):                # full re-arm
        tr.update(10.0)
    tr.update(300.0); tr.update(300.0)
    assert tr.hits == 2
    assert tr.peak == 300.0 or tr.peak == 500.0 or tr.peak >= 300.0


def test_bin_hit_peak_sustained():
    tr = BinHitTracker(1000.0)
    tr.update(900.0)                  # spike, not sustained yet
    assert tr.peak_sustained == 0.0
    tr.update(800.0)
    assert tr.peak_sustained == 800.0
