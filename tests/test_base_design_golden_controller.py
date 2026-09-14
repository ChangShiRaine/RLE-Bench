from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
TASK = REPO / "tasks" / "task08"
GOLDEN = str(REPO / "tasks" / "task08" / "harness" / "assets" / "golden" / "robot.xml")


def test_task08_submission_defers_controller_loading(tmp_path):
    from harness.base_design.scorer import load_submission

    (tmp_path / "robot.xml").write_text("<mujoco/>")
    # Invalid Python proves a submitted controller is not imported.
    (tmp_path / "controller.py").write_text("this is invalid Python !")

    submission = load_submission(str(tmp_path))
    assert submission.errors == []
    assert submission.shelf_controller == str(tmp_path / "controller.py")


def test_checkpoints_cover_design_and_shelf_objectives():
    from harness.base_design.checkpoints import CHECKPOINTS

    ids = {checkpoint.id for checkpoint in CHECKPOINTS}
    assert sum(checkpoint.weight for checkpoint in CHECKPOINTS) == pytest.approx(1.0)
    assert {"S3.resource_constraints", "S3.design_efficiency",
            "S7.pick_compat", "S7.payload_margin"} <= ids
    efficiency = next(cp for cp in CHECKPOINTS if cp.id == "S3.design_efficiency")
    assert efficiency.weight == pytest.approx(0.22)
    constraints = next(cp for cp in CHECKPOINTS if cp.id == "S3.resource_constraints")
    assert constraints.weight == pytest.approx(0.02)
    assert not constraints.gate


def test_base_controller_is_the_mecanum_inverse_kinematics():
    import mujoco
    from harness.sim.mecanum import (commanded_wheel_speeds, geometry_from_model,
                                     make_base_controller)

    model = mujoco.MjModel.from_xml_path(GOLDEN)
    lx, ly, r = geometry_from_model(model)
    step = make_base_controller(model)
    for twist in ((0.3, 0.0, 0.0), (0.0, -0.2, 0.0),
                  (0.2, 0.1, -0.4)):
        assert np.allclose(step(None, twist),
                           commanded_wheel_speeds(*twist, lx=lx, ly=ly, r=r))


def test_generated_solution_has_controller_but_agent_environment_has_no_oracle():
    verifier = TASK / "tests" / "harness" / "base_design"
    solution = TASK / "solution" / "payload"
    workspace = TASK / "environment" / "assets"

    assert (verifier / "golden_controller.py").is_file()
    assert {path.name for path in solution.iterdir()} == {"robot.xml", "controller.py"}
    assert not (workspace / "harness").exists()
    assert not list(workspace.rglob("*.py"))


def test_task08_strips_merit_from_report():
    from harness.base_design.scorer import _without_merit

    report = _without_merit({
        "reward": 0.8,
        "merit": 1.2,
        "merit_stages": {"design": 0.3},
        "merits": {"S3.design_efficiency": 2.0},
    })
    assert report == {"reward": 0.8}


@pytest.mark.parametrize("mount_quat", [(1, 0, 0, 0), (0.5, 0.5, 0.5, 0.5)])
def test_canonical_arm_variants_compose_on_same_submitted_mount(tmp_path, mount_quat):
    import mujoco
    import xml.etree.ElementTree as ET
    from harness.sim.arm_variants import (
        ADAPTER_HALF_SIZE_M, ADAPTER_MASS_KG, trusted_arm_specs, canonical_robot_spec)

    tree = ET.parse(TASK / "reference" / "robot.xml")
    tree.find(".//site[@name='arm_mount_site']").set(
        "quat", " ".join(map(str, mount_quat)))
    # Preserve the reference's relative asset resolution in the temporary XML.
    compiler = tree.find("compiler")
    compiler.set("meshdir", str((TASK / "reference" / compiler.get("meshdir")).resolve()))
    robot = str(tmp_path / "robot.xml")
    tree.write(robot)
    panda_ref = str(REPO / "assets" / "robots"
                    / "franka_emika_panda" / "panda_nohand.xml")
    specs = trusted_arm_specs(panda_ref)
    assert set(specs) == {"panda", "ur5e", "xarm7"}
    for name, arm in specs.items():
        model = canonical_robot_spec(robot, arm).compile()
        assert model.body("arm_adapter").id > 0
        assert model.body(arm.assembled_root).id > 0
        assert model.site(arm.ee_site).id >= 0
        assert len(arm.actuator_names) == (6 if name == "ur5e" else 7)
        assert model.geom("arm_adapter_plate").id >= 0
        assert model.body_mass[model.body("arm_adapter").id] == \
            pytest.approx(ADAPTER_MASS_KG)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        mount = data.site("arm_mount_site")
        expected = mount.xpos + mount.xmat.reshape(3, 3) @ np.array(
            [0, 0, 2 * ADAPTER_HALF_SIZE_M[2]])
        np.testing.assert_allclose(data.body(arm.assembled_root).xpos, expected,
                                   atol=1e-12)


def test_three_arm_score_uses_checkpoint_wise_minimum(monkeypatch):
    import harness.base_design.scorer as scorer

    values = {"panda": 0.9, "ur5e": 0.6, "xarm7": 0.8}

    def fake(_submission, _reference, _env, arm):
        checkpoints = {cp.id: 1.0 for cp in scorer.CHECKPOINTS}
        checkpoints["S3.design_efficiency"] = values[arm.name]
        return {
            "reward": values[arm.name], "raw_total": values[arm.name],
            "checkpoints": checkpoints, "errors": [],
            "model_info": {"variant": arm.name},
            "design_headroom": {"score": values[arm.name]},
            "pick_compat": {"score": values[arm.name]},
        }

    monkeypatch.setattr(scorer, "_score_single_submission", fake)
    panda_ref = str(REPO / "assets" / "robots"
                    / "franka_emika_panda" / "panda_nohand.xml")
    report = scorer.score_submission("unused", panda_ref)
    assert report["reward"] == pytest.approx(1.0 - 0.22 * 0.4)
    assert report["limiting_arm"] == "ur5e"
    assert set(report["arm_reports"]) == {"panda", "ur5e", "xarm7"}
    assert report["checkpoints"]["S3.design_efficiency"] == \
        pytest.approx(0.6)
    assert all(value == pytest.approx(1.0)
               for key, value in report["checkpoints"].items()
               if key != "S3.design_efficiency")
