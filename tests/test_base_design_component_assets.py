from __future__ import annotations

import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import pytest


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO / "tasks" / "task08" / "environment" / "assets"
SOURCE = REPO / "tasks" / "task08" / "harness" / "assets" / "mobile_manipulator_components"
ENVIRONMENT = (
    REPO / "tasks" / "task08" / "environment" / "assets" / "assets"
    / "mobile_manipulator_components"
)
VERIFIER = (
    REPO / "tasks" / "task08" / "tests" / "models" / "assets"
    / "mobile_manipulator_components"
)
PANDA_REF = (REPO / "tasks" / "task08" / "tests" / "models" / "assets"
             / "franka_emika_panda" / "panda_nohand.xml")


def _panda():
    from harness.sim.arm_variants import trusted_arm_specs
    return trusted_arm_specs(str(PANDA_REF))["panda"]


def test_instruction_names_every_agent_owned_hard_gate_joint():
    instruction = (REPO / "tasks" / "task08" / "instruction.md").read_text()
    assert "free joint named exactly `base_free`" in instruction
    for suffix in ("FL", "FR", "RL", "RR"):
        assert f"`drive_{suffix}`" in instruction


def test_component_examples_compile_and_have_no_assembled_chassis():
    expected = {
        "README.md",
        "arm_adapter.xml",
        "aluminum_profiles.xml",
        "battery.xml",
        "payload.xml",
        "component_catalog.json",
        "mecanum_wheel_left.xml",
        "mecanum_wheel_right.xml",
    }
    assert {path.name for path in SOURCE.iterdir()} == expected

    for path in SOURCE.glob("*.xml"):
        mujoco.MjModel.from_xml_path(str(path))
        root = ET.parse(path).getroot()
        model_names = [
            element.get("name", "").lower() for element in root.iter()
        ]
        assert all("chassis" not in name for name in model_names)


def test_component_catalog_is_chassis_free_stock_inventory():
    catalog = json.loads((SOURCE / "component_catalog.json").read_text())
    assert catalog["mecanum_wheel"]["quantity_available"] == 4
    assert catalog["battery"]["quantity_available"] == 1
    assert catalog["pickup_payload"]["mass_kg"] == pytest.approx(1.0)
    assert catalog["universal_arm_adapter"]["mass_kg"] == pytest.approx(0.875)
    assert catalog["universal_arm_adapter"]["ur5e_mount_yaw_deg"] == 180.0
    assert set(catalog["canonical_arm_variants"]) == {
        "panda", "ur5e", "xarm7"
    }
    assert {profile["name"] for profile in catalog["aluminum_profiles"]} == {
        "2020", "2040", "4040"
    }

    left = ET.parse(SOURCE / "mecanum_wheel_left.xml").getroot()
    right = ET.parse(SOURCE / "mecanum_wheel_right.xml").getroot()
    assert len(left.findall(".//body[@name='wheel_EXAMPLE']/body")) == 16
    assert len(right.findall(".//body[@name='wheel_EXAMPLE']/body")) == 16


def test_reference_is_built_only_from_catalog_profile_stock():
    reference = REPO / "tasks" / "task08" / "reference" / "robot.xml"
    mujoco.MjModel.from_xml_path(str(reference))
    root = ET.parse(reference).getroot()
    base = root.find(".//body[@name='base']")
    geoms = base.findall("geom")
    names = {geom.get("name") for geom in geoms}
    assert "chassis" not in names
    assert "arm_mount" not in names

    stock_4040 = [geom for geom in geoms
                  if geom.get("name", "").startswith("profile_4040_")]
    stock_2020 = [geom for geom in geoms
                  if geom.get("name", "").startswith("profile_2020_")]
    assert len(stock_4040) == 33
    assert len(stock_2020) == 0
    assert sum(float(geom.get("mass")) for geom in stock_4040) == \
        pytest.approx(1.5 * 10.778577333333333)
    assert sum(float(geom.get("mass")) for geom in stock_2020) == \
        pytest.approx(0.0)
    assert base.find("body[@name='arm_link0']").get("pos") == "0 0 0.285"
    assert base.find("body[@name='wheel_FL']").get("pos") == \
        "0.203 0.243 0"
    assert base.find("geom[@name='profile_4040_axle_front']").get("pos") == \
        "0.203 0 0"


def test_task08_requires_structure_through_each_wheel_center(tmp_path):
    from harness.sim.runner import check_validity
    from harness.sim.scenarios import Envelope

    reference = REPO / "tasks" / "task08" / "reference" / "robot.xml"
    strict = Envelope()
    report = check_validity(str(reference), strict, _panda())
    assert report["structure"]["ok"]
    assert report["structure"]["wheel_center_gaps"] == {
        "wheel_FL": 0.0, "wheel_FR": 0.0,
        "wheel_RL": 0.0, "wheel_RR": 0.0,
    }

    root = ET.parse(reference).getroot()
    base = root.find(".//body[@name='base']")
    for name in ("profile_4040_axle_front", "profile_4040_axle_rear"):
        base.remove(base.find(f"geom[@name='{name}']"))
    root.find("compiler").set(
        "meshdir", str(REPO / "assets" / "robots"
                       / "franka_emika_panda" / "assets"))
    without_axles = tmp_path / "robot_without_center_axles.xml"
    ET.ElementTree(root).write(without_axles)

    relaxed_report = check_validity(
        str(without_axles), Envelope(require_wheel_center_attachment=False), _panda())
    assert relaxed_report["structure"]["ok"]
    strict_report = check_validity(str(without_axles), strict, _panda())
    assert not strict_report["structure"]["ok"]
    assert any(gap > 0.005 for gap in
               strict_report["structure"]["wheel_center_gaps"].values())
    assert all("through its center" in problem for problem in
               strict_report["structure"]["problems"])


def test_generated_task_asset_copy_matches_source():
    assert not (WORKSPACE / "robot.xml").exists()
    assert not (WORKSPACE / "harness").exists()
    assert not list(WORKSPACE.rglob("*.py"))
    assert {path.name for path in WORKSPACE.iterdir()} == {
        "assets",
        "scene.xml",
        "shelf_spec.json",
    }
    assert {path.name for path in ENVIRONMENT.parent.iterdir()} == {
        "franka_emika_panda",
        "universal_robots_ur5e",
        "ufactory_xarm7",
        "mobile_manipulator_components",
    }
    for destination in (ENVIRONMENT, VERIFIER):
        assert destination.is_dir()
        for source in SOURCE.iterdir():
            assert (destination / source.name).read_bytes() == source.read_bytes()

    xarm = REPO / "assets/robots/ufactory_xarm7/xarm7_nohand.xml"
    for path in (xarm, WORKSPACE / "assets/ufactory_xarm7/xarm7_nohand.xml",
                 VERIFIER.parent / "ufactory_xarm7/xarm7_nohand.xml"):
        assert path.read_bytes() == xarm.read_bytes()
        model = mujoco.MjModel.from_xml_path(str(path))
        assert model.body("link_base").pos == pytest.approx([0, 0, 0])


def test_public_shelf_scene_matches_task08_evaluator(tmp_path):
    source_scene = REPO / "tasks" / "task08" / "harness" / "assets" / "base_design" / "scene.xml"
    source_spec = REPO / "tasks" / "task08" / "harness" / "assets" / "base_design" / "shelf_spec.json"
    spec = json.loads(source_spec.read_text())

    assert (WORKSPACE / "scene.xml").read_bytes() == source_scene.read_bytes()
    assert (WORKSPACE / "shelf_spec.json").read_bytes() == source_spec.read_bytes()

    root = ET.parse(source_scene).getroot()
    sites = {site.get("name"): [float(v) for v in site.get("pos").split()]
             for site in root.findall(".//site")}
    names = spec["reach_goal"]["target_names_in_order"]
    assert len(names) == 12
    assert [sites[f"shelf_target_{name}"] for name in names] == \
        spec["reach_goal"]["flange_target_points_m"]
    assert [sites[f"shelf_approach_{name}"] for name in names] == \
        spec["reach_goal"]["approach_points_m"]

    from harness import config
    from harness.base_design.pickscene import shelf_targets

    assert spec["shelf"]["face_x_m"] == pytest.approx(config.PICK_SHELF_FACE_X_M)
    assert spec["reach_goal"]["payload_kg"] == pytest.approx(config.PICK_PAYLOAD_KG)
    assert spec["reach_goal"]["extra_payload_margin_kg"] == pytest.approx(config.PICK_EXTRA_PAYLOAD_KG)
    assert spec["reach_goal"]["margin_test_payload_kg"] == pytest.approx(
        config.PICK_PAYLOAD_KG + config.PICK_EXTRA_PAYLOAD_KG)
    evaluator_points = [point.tolist() for point in shelf_targets(config.PICK_SHELF_FACE_X_M)]
    for actual, expected in zip(
            evaluator_points, spec["reach_goal"]["flange_target_points_m"]):
        assert actual == pytest.approx(expected)

    staged = tmp_path / "workspace"
    shutil.copytree(WORKSPACE, staged)
    shutil.copy(REPO / "tasks" / "task08" / "solution" / "payload"
                / "robot.xml", staged / "robot.xml")
    mujoco.MjModel.from_xml_path(str(staged / "scene.xml"))


def test_reference_profile_inventory_and_design_headroom():
    from harness.base_design.checkpoints import analyze_model

    reference = REPO / "tasks" / "task08" / "reference" / "robot.xml"
    info = analyze_model(str(reference), _panda())
    inventory = info["profile_inventory"]
    efficiency = info["design_efficiency"]

    assert inventory["eligible"]
    assert inventory["count"] == 33
    assert inventory["count_by_type"] == {"2020": 0, "2040": 0, "4040": 33}
    assert inventory["total_length_m"] == pytest.approx(10.778576)
    assert efficiency["score"] == pytest.approx(0.0821953367)
    assert efficiency["components"] == pytest.approx({
        "mass": 0.1176877368,
        "profile_length": 0.1017853333,
        "footprint_area": 0.0271129401,
    })
    assert efficiency["mass_headroom_kg"] == pytest.approx(4.472134)
    assert efficiency["profile_length_headroom_m"] == pytest.approx(1.221424)
    assert efficiency["footprint_area_headroom_m2"] > 0.008


def test_design_efficiency_is_continuous_near_resource_limits():
    from harness.base_design.checkpoints import design_efficiency

    info = {
        "footprint": (0.56, 0.56),
        "total_mass": 56.0,
        "profile_inventory": {
            "eligible": True, "count": 20, "total_length_m": 11.5,
        },
    }
    result = design_efficiency(info)
    assert result["components"]["mass"] == pytest.approx(1.0 - 56.0 / 60.0)
    assert result["components"]["profile_length"] == pytest.approx(
        1.0 - 11.5 / 12.0)
    assert result["components"]["footprint_area"] == pytest.approx(0.0)
    assert result["score"] == pytest.approx(
        ((1.0 - 56.0 / 60.0) + (1.0 - 11.5 / 12.0)) / 3.0)

    leaner = design_efficiency({
        **info,
        "footprint": (0.54, 0.54),
        "total_mass": 54.0,
        "profile_inventory": {
            "eligible": True, "count": 20, "total_length_m": 11.0,
        },
    })
    assert leaner["score"] > result["score"]


def test_resource_constraints_fall_off_smoothly_beyond_targets():
    from harness.base_design.checkpoints import resource_constraint_score

    at_target = resource_constraint_score({
        "ok": True,
        "footprint": (0.56, 0.56),
        "total_mass": 60.0,
        "profile_inventory": {
            "eligible": True, "total_length_m": 12.0,
        },
        "arm": {"ok": True},
        "actuators": {"ok": True},
    })
    assert at_target["score"] == pytest.approx(1.0)

    oversized = resource_constraint_score({
        "ok": True,
        "footprint": (0.70, 0.56),
        "total_mass": 75.0,
        "profile_inventory": {
            "eligible": True, "total_length_m": 15.0,
        },
        "arm": {"ok": True},
        "actuators": {"ok": True},
    })
    assert oversized["components"] == pytest.approx({
        "footprint_x": 0.8,
        "footprint_y": 1.0,
        "assembled_mass": 0.8,
        "profile_length": 0.8,
        "arm_integrity": 1.0,
        "actuators": 1.0,
    })
    assert 0.0 < oversized["score"] < at_target["score"]
