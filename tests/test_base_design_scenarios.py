"""Scenario runner checks on the golden model.

Full-battery pass/fail thresholds live in test_base_design_golden.py; here we
verify the machinery: composition, determinism, validity bundle, and that
golden survives a representative scenario.
"""
import os

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
ROBOT = os.path.join(HERE, "..", "tasks", "task08", "harness", "assets",
                     "golden", "robot.xml")
ARM_REF = os.path.join(HERE, "..", "assets", "robots",
                       "franka_emika_panda", "panda_nohand.xml")


def _panda():
    from harness.sim.arm_variants import trusted_arm_specs
    return trusted_arm_specs(ARM_REF)["panda"]


def test_validity_bundle_golden():
    from harness.sim.runner import check_validity
    from harness.sim.scenarios import Envelope
    v = check_validity(ROBOT, Envelope(require_wheel_center_attachment=False), _panda())
    assert v["loads"]
    assert v["inertia_ok"]
    assert v["settle"]["ok"], v["settle"]
    assert v["mecanum"]["ok"], v["mecanum"]["cases"]
    assert 45.0 < v["total_mass"] < 60.0
    assert v["ok"]


def test_scenario_b_golden_stable():
    from harness.sim.scenarios import Envelope, run_scenario
    r = run_scenario(ROBOT, "B", seed=11, env=Envelope(), arm=_panda())
    assert not r["tip"]
    assert not r["liftoff_events"]
    assert r["min_fasm"] > 0
    assert r["min_ssm"] > 0
    # velocity tracking: peak measured vx close to peak command
    tc = np.asarray(r["twist_cmd"]); tm = np.asarray(r["twist_meas"])
    assert tm[:, 0].max() == pytest.approx(tc[:, 0].max(), rel=0.15)


def test_scenario_determinism():
    """Same (scenario, seed) => bit-identical metric series."""
    from harness.sim.scenarios import Envelope, run_scenario
    a = run_scenario(ROBOT, "D2", seed=23, env=Envelope(), arm=_panda())
    b = run_scenario(ROBOT, "D2", seed=23, env=Envelope(), arm=_panda())
    assert np.array_equal(a["fasm"], b["fasm"])
    assert np.array_equal(a["ssm"], b["ssm"])
    assert a["params"]["payload_kg"] == b["params"]["payload_kg"]


def test_scenario_seeds_differ():
    """Different seed => different held-out terrain/payload instance."""
    from harness.sim.scenarios import Envelope, run_scenario
    a = run_scenario(ROBOT, "D1", seed=11, env=Envelope(), arm=_panda())
    b = run_scenario(ROBOT, "D1", seed=23, env=Envelope(), arm=_panda())
    assert a["params"]["grade_deg"] != b["params"]["grade_deg"]


def test_slope_scene_aligned():
    """On D1's slope the settled robot reads ~zero tilt vs the terrain
    normal."""
    from harness.sim.scenarios import Envelope, run_scenario
    r = run_scenario(ROBOT, "D1", seed=11, env=Envelope(), arm=_panda())
    assert r["params"]["grade_deg"] > 5.0
    assert r["max_tilt_deg"] < 5.0
    assert not r["tip"]
