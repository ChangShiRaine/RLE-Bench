"""Mecanum kinematics: IK formula + holonomic motion on the golden model.

The lateral-motion case is THE probe for the roller-axis flaw: a model with
mirrored rollers still drives forward fine but moves the wrong way (or yaws)
under a pure lateral command.
"""
import os

import pytest

GOLDEN_SCENE = os.path.join(os.path.dirname(__file__), "..",
                            "tasks", "task08", "harness", "assets", "golden", "scene.xml")


def test_ik_formula_analytic():
    from harness.sim.mecanum import commanded_wheel_speeds
    r, lx, ly = 0.05, 0.2, 0.25
    # pure forward: all wheels same speed = vx/r
    assert commanded_wheel_speeds(1.0, 0, 0, lx, ly, r) == pytest.approx([20, 20, 20, 20])
    # pure lateral +y: FL/RR negative, FR/RL positive
    ws = commanded_wheel_speeds(0, 1.0, 0, lx, ly, r)
    assert ws == pytest.approx([-20, 20, 20, -20])
    # pure spin: left wheels backward, right wheels forward, scaled by lx+ly
    ws = commanded_wheel_speeds(0, 0, 1.0, lx, ly, r)
    assert ws == pytest.approx([-9, 9, -9, 9])


def test_geometry_from_model():
    import mujoco
    from harness.sim.mecanum import geometry_from_model
    model = mujoco.MjModel.from_xml_path(GOLDEN_SCENE)
    lx, ly, r = geometry_from_model(model)
    assert lx == pytest.approx(0.19, abs=1e-6)
    assert ly == pytest.approx(0.235, abs=1e-6)
    assert r == pytest.approx(0.076, abs=1e-6)


def test_base_design_golden_holonomic_motion():
    """Command each canonical twist; achieved motion must match, and the
    pure-lateral case must produce lateral motion with ~zero forward/yaw."""
    import mujoco
    from harness.sim.mecanum import verify_holonomic_motion
    model = mujoco.MjModel.from_xml_path(GOLDEN_SCENE)
    data = mujoco.MjData(model)
    result = verify_holonomic_motion(model, data)
    assert result["ok"], f"holonomic verification failed: {result['cases']}"
    lateral = result["cases"][1]
    vx, vy, wz = lateral["achieved"]
    assert abs(vy - 0.4) < 0.1      # moves laterally at ~commanded speed
    assert abs(vx) < 0.05           # ~zero forward
    assert abs(wz) < 0.1            # ~zero yaw
