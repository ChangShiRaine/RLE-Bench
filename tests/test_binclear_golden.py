"""Task07 golden-first regression guard.

The privileged golden picker is the physics proof and the calibration
ceiling: it must clear the bulk of the bin within budget, damage nothing
beyond its calibrated envelope, and be bit-deterministic. The null policy
pins the reward floor at ~0.
"""
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

from harness import golden, runtime, spec

SEED = spec.DESIGN_SEEDS[1]   # fastest-converging design seed


def test_golden_clears_the_bin():
    log = runtime.run_episode(golden.GoldenPicker(), SEED,
                              budget_t=spec.EPISODE_T, render=False)
    m = log.metrics()
    assert m["clear_frac"] >= 0.7, m   # singulated regime
    assert m["floor_drops"] <= 2, m
    assert m["bin_hits"] == 0, m      # working contact stays under the limit
    assert log.aborted is None
    assert log.policy_faults == 0
    # throughput consistent with a working cell, not a fluke of latching
    assert m["throughput_ppm"] > 4.0, m


def test_golden_deterministic():
    logs = [runtime.run_episode(golden.GoldenPicker(), SEED,
                                budget_t=20.0, render=False)
            for _ in range(2)]
    assert logs[0].digest == logs[1].digest
    assert logs[0].clear_times == logs[1].clear_times


def test_null_policy_scores_zero():
    log = runtime.run_episode(runtime.NullPolicy(), SEED,
                              budget_t=5.0, render=False)
    m = log.metrics(5.0)
    assert m["parts_cleared"] == 0
    assert m["clear_frac"] == 0.0
