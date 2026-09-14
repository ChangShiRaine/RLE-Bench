"""CAD assembly: pose preservation, portable assets and tamper checks."""
from pathlib import Path
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from dev import assemble, generate_variant_assets as generator, servo_proxy
from harness.codesign_oracle import build_reference_submission
from harness.kinematics import structure_check
from harness.validity import density_report
from harness.variants import VARIANTS

TASK = Path(__file__).resolve().parents[1] / "tasks/task09"
MODELS = TASK / "harness/assets/gello_codesign"


@pytest.mark.parametrize("name", tuple(VARIANTS))
def test_task_home_preserves_accepted_cad_assembly(name, tmp_path):
    arm = generator.ARM_NAMES[name]
    preview_path = tmp_path / "preview.xml"
    assemble.build(arm, str(preview_path))
    preview = mujoco.MjModel.from_xml_path(str(preview_path))
    actual = mujoco.MjModel.from_xml_path(str(MODELS/name/"lead_bare.xml"))
    dp, da = mujoco.MjData(preview), mujoco.MjData(actual)
    dp.qpos[:] = np.deg2rad(assemble.REFERENCE_POSES[arm])
    da.qpos[:] = VARIANTS[name].home
    mujoco.mj_forward(preview, dp)
    mujoco.mj_forward(actual, da)
    assert actual.nv == VARIANTS[name].n_joints
    assert preview.nv == actual.nv + 1
    for g in range(preview.ngeom):
        geom_name = preview.geom(g).name
        if geom_name == f"printed_link{actual.nv}":
            geom_name = "printed_handle"
        target = actual.geom(geom_name).id
        np.testing.assert_allclose(da.geom_xpos[target], dp.geom_xpos[g], atol=1e-8)
        np.testing.assert_allclose(da.geom_xmat[target], dp.geom_xmat[g], atol=1e-8)
    for g in range(actual.ngeom):
        expected = (mujoco.mjtGeom.mjGEOM_BOX if actual.geom(g).name.startswith("servo")
                    else mujoco.mjtGeom.mjGEOM_MESH)
        assert actual.geom_type[g] == expected


@pytest.mark.parametrize("name", tuple(VARIANTS))
def test_real_generator_is_reproducible(name, tmp_path, monkeypatch):
    monkeypatch.setattr(generator, "ROOT", str(tmp_path))
    first = generator.generate_variant(name)
    snapshot = {p.relative_to(tmp_path): p.read_bytes()
                for p in tmp_path.rglob("*") if p.is_file()}
    second = generator.generate_variant(name)
    assert first == second
    assert snapshot == {p.relative_to(tmp_path): p.read_bytes()
                        for p in tmp_path.rglob("*") if p.is_file()}
    for relative, data in snapshot.items():
        assert (MODELS/relative).read_bytes() == data


@pytest.mark.parametrize("name", tuple(VARIANTS))
def test_public_workspace_imports_without_private_harness(name):
    workspace = TASK / "environment/assets" / name
    code = """
from pathlib import Path
from harness.scenarios import compose_lead
from harness import spec
assert Path(spec.__file__).resolve().is_relative_to(Path.cwd())
model, _ = compose_lead('lead.xml')
assert model.nv == spec.N_JOINTS
assert not Path('lead_sref.xml').exists()
assert not Path('harness/codesign_variant_config.py').exists()
"""
    subprocess.run([sys.executable, "-c", code], cwd=workspace,
                   env={**os.environ, "PYTHONPATH": str(workspace)}, check=True)


@pytest.mark.parametrize("mutation", ("motor_mass", "motor_mount", "motor_size", "visual_mass", "mesh_bytes", "mesh_scale", "mesh_replacement", "missing_geom", "inertial", "geom_default"))
def test_stock_mesh_and_motor_tampering_is_rejected(mutation, tmp_path):
    sub = Path(build_reference_submission(str(tmp_path / "sub")))
    path = sub / "lead.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    if mutation == "motor_mass":
        root.find(".//geom[@name='servo2']").set("mass", "0.001")
    elif mutation == "motor_mount":
        root.find(".//geom[@name='servo2']").set("pos", "0 0 0")
    elif mutation == "motor_size":
        root.find(".//geom[@name='servo2']").set("size", ".001 .001 .001")
    elif mutation == "visual_mass":
        root.find(".//geom[@name='visual_servo2']").set("mass", ".018")
    elif mutation == "mesh_bytes":
        mesh = sub / "meshes/L2.stl"
        blob = bytearray(mesh.read_bytes())
        blob[0] ^= 1
        mesh.write_bytes(blob)
    elif mutation == "mesh_scale":
        root.find("./asset/mesh[@name='L2']").set("scale", ".0001 .0001 .0001")
    elif mutation == "mesh_replacement":
        root.find(".//geom[@name='printed_link2']").set("mesh", "L1")
    elif mutation == "missing_geom":
        root.find(".//geom[@name='printed_link2']").set("name", "added_link")
    elif mutation == "inertial":
        ET.SubElement(root.find(".//body[@name='lead_link2']"), "inertial",
                      pos="0 0 0", mass="0.018", diaginertia=".001 .001 .001")
    else:
        default = root.find("default")
        if default is None:
            default = ET.SubElement(root, "default")
        ET.SubElement(default, "geom", mass="0.001")
    tree.write(path)
    report = density_report(str(path))
    assert not report["ok"], report


@pytest.mark.parametrize("kind", ("mesh", "sphere"))
@pytest.mark.parametrize("name", ("support", "printed_support"))
@pytest.mark.parametrize("mass_property", ({"density": "1"}, {"density": "50000"}, {"mass": ".001"}))
def test_added_structures_do_not_affect_printability(kind, name, mass_property, tmp_path):
    sub = Path(build_reference_submission(str(tmp_path / "sub")))
    path = sub / "lead.xml"
    before = density_report(str(path))
    tree = ET.parse(path)
    root = tree.getroot()
    if kind == "mesh":
        ET.SubElement(root.find("asset"), "mesh", name="custom_support",
                      vertex="0 0 0 .002 0 0 0 .002 0 0 0 .002",
                      face="0 2 1 0 1 3 0 3 2 1 2 3")
        geometry = {"type": "mesh", "mesh": "custom_support"}
    else:
        geometry = {"type": "sphere", "size": ".001"}
    ET.SubElement(root.find(".//body[@name='lead_link2']"), "geom",
                  name=name, **geometry, **mass_property)
    tree.write(path)
    mujoco.MjModel.from_xml_path(str(path))
    assert density_report(str(path)) == before


def test_invalid_added_mesh_fails_model_validity(tmp_path):
    from harness.validity import check_validity

    sub = Path(build_reference_submission(str(tmp_path / "sub")))
    path = sub / "lead.xml"
    before = density_report(str(path))
    tree = ET.parse(path)
    ET.SubElement(tree.getroot().find("asset"), "mesh", name="broken", file="missing.obj")
    ET.SubElement(tree.find(".//body[@name='lead_link2']"), "geom",
                  name="support", type="mesh", mesh="broken", mass=".001")
    tree.write(path)
    assert not check_validity(str(path))["loads"]
    assert density_report(str(path)) == before


def test_structure_rejects_changed_axis(tmp_path):
    sub = Path(build_reference_submission(str(tmp_path / "sub")))
    path = sub / "lead.xml"
    tree = ET.parse(path)
    tree.find(".//joint[@name='lead_joint4']").set("axis", "1 0 0")
    tree.write(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    report = structure_check(model, mujoco.MjData(model))
    assert not report["ok"]


def test_calibration_bars_match_saved_measurements():
    from dev.calibrate import derive
    from harness.codesign_variant_config import _BARS
    for name in VARIANTS:
        record = json.loads((TASK/f"dev/data/calibration_{name}.json").read_text())
        assert derive(record["measurements"]) == record["bars"] == _BARS[name]


@pytest.mark.parametrize("name", tuple(servo_proxy.PHYSICS))
def test_servo_proxy_matches_pinned_inertia_and_is_original(name, tmp_path):
    servo_proxy.write_meshes(tmp_path)
    snapshot = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    servo_proxy.write_meshes(tmp_path)
    assert snapshot == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert all(b"original generic micro-servo" in data[:80] for data in snapshot.values())
    root = ET.fromstring(f'<mujoco><asset><mesh name="{name}" file="{tmp_path/name}.stl" scale=".001 .001 .001"/></asset><worldbody><body><geom name="servo1" type="mesh" mesh="{name}" mass=".018"/></body></worldbody></mujoco>')
    servo_proxy.separate_visuals(root)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    stock = servo_proxy.PHYSICS[name]
    np.testing.assert_allclose(model.body_mass[1], .018, atol=1e-16)
    np.testing.assert_allclose(model.body_ipos[1], stock["pos"], atol=1e-16)
    np.testing.assert_allclose(model.body_inertia[1], stock["inertia"], atol=1e-16)
    assert abs(np.dot(model.body_iquat[1], stock["quat"])) == pytest.approx(1)


@pytest.mark.parametrize("name", tuple(VARIANTS))
def test_remote_counterweights_have_mass_bearing_supports(name):
    path = MODELS / name / "lead_oracle.xml"
    root = ET.parse(path).getroot()
    model = mujoco.MjModel.from_xml_path(str(path))
    for body in root.iter("body"):
        for cw in body.findall("geom"):
            if not cw.get("name", "").startswith("counterweight_"):
                continue
            supports = [g for g in body.findall("geom")
                        if g.get("name", "").startswith("printed_balance_")
                        and g.get("type") == "capsule" and g.get("fromto")]
            position = np.fromstring(cw.get("pos"), sep=" ")
            support = next((g for g in supports if np.allclose(
                np.fromstring(g.get("fromto"), sep=" ")[3:], position)), None)
            assert support is not None, cw.get("name")
            assert float(support.get("density")) >= 100
            assert float(support.get("size")) > 0
            # The support's root must touch a mass-bearing mount on its body.
            ends = np.fromstring(support.get("fromto"), sep=" ").reshape(2, 3)
            mounts = [g for g in body.findall("geom")
                      if g.get("name", "").startswith("printed_balance_")
                      and g.get("type") == "cylinder"]
            assert any(np.linalg.norm(ends[0] - model.geom(g.get("name")).pos)
                       <= model.geom_rbound[model.geom(g.get("name")).id]
                       and float(g.get("density", "0")) > 0 for g in mounts)
