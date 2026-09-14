"""Usability battery: structure connectivity (S1 gate), battery pack,
reach-beyond-footprint, and artifact isolation.

The robot must be a buildable machine (no free-floating wheels or bases that
the physics happily simulates) that can actually work outside its own
footprint.
"""
import os
import xml.etree.ElementTree as ET

import pytest

HERE = os.path.dirname(__file__)
ASSETS = os.path.join(HERE, "..", "tasks", "task08", "harness", "assets")
GOLDEN = os.path.join(ASSETS, "golden", "robot.xml")
ARM_REF = os.path.abspath(os.path.join(HERE, "..", "assets", "robots", "franka_emika_panda",
                                       "panda_nohand.xml"))
MESHDIR = os.path.abspath(os.path.join(HERE, "..", "assets", "robots", "franka_emika_panda",
                                       "assets"))


def _panda():
    from harness.sim.arm_variants import trusted_arm_specs
    return trusted_arm_specs(ARM_REF)["panda"]


def _variant(tmp_path, mutate):
    """Golden robot.xml with `mutate(root)` applied, meshdir absolutized."""
    tree = ET.parse(GOLDEN)
    root = tree.getroot()
    root.find("compiler").set("meshdir", MESHDIR)
    mutate(root)
    path = os.path.join(str(tmp_path), "robot.xml")
    tree.write(path)
    return path


def _body(root, name):
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    raise KeyError(name)


# ---------------------------------------------------------------------------
# structure connectivity (S1.structure gate)
# ---------------------------------------------------------------------------

def test_base_design_golden_passes_structure_check():
    from harness.sim.runner import check_validity
    from harness.sim.scenarios import Envelope
    v = check_validity(GOLDEN, Envelope(require_wheel_center_attachment=False), _panda())
    s = v["structure"]
    assert s["ok"], s["problems"]
    # golden wheels mount 3.5 cm off the chassis: comfortably inside the gap
    assert all(g < 0.045 for g in s["wheel_gaps"].values())


def test_detached_wheel_fails_structure(tmp_path):
    """A wheel floating in free space, held only by its hinge joint, must
    fail validity."""
    from harness.sim.runner import check_validity
    path = _variant(tmp_path, lambda root:
                    _body(root, "wheel_FL").set("pos", "0.6 0.6 0"))
    v = check_validity(path, arm=_panda())
    assert not v["structure"]["ok"]
    assert any("wheel_FL" in p for p in v["structure"]["problems"])
    assert not v["ok"]


def test_floating_battery_splits_structure(tmp_path):
    """A base part hovering away from the chassis = disconnected assembly."""
    from harness.sim.runner import check_validity
    path = _variant(tmp_path, lambda root:
                    _body(root, "battery").set("pos", "0 0 1.0"))
    v = check_validity(path, arm=_panda())
    assert not v["structure"]["ok"]
    assert any("disconnected" in p for p in v["structure"]["problems"])


def test_body_outside_base_tree_fails(tmp_path):
    """A second kinematic tree (e.g. a wheel authored as a sibling of the
    base) is not one robot."""
    from harness.sim.runner import check_validity

    def mutate(root):
        wb = root.find("worldbody")
        stray = ET.SubElement(wb, "body", dict(name="stray", pos="1 0 0.1"))
        ET.SubElement(stray, "geom", dict(
            name="stray_geom", type="box", size="0.05 0.05 0.05", mass="1"))
    path = _variant(tmp_path, mutate)
    v = check_validity(path, arm=_panda())
    assert not v["structure"]["ok"]
    assert any("stray" in p for p in v["structure"]["problems"])


def test_worldbody_geom_rejected(tmp_path):
    """Scenery the robot brings along (a static crutch to lean on) must not
    even compose."""
    from harness.sim.runner import check_validity

    def mutate(root):
        wb = root.find("worldbody")
        ET.SubElement(wb, "geom", dict(
            name="crutch", type="box", size="0.05 0.05 0.4", pos="0.5 0 0.4"))
    path = _variant(tmp_path, mutate)
    v = check_validity(path, arm=_panda())
    assert not v["loads"]
    assert "worldbody" in v.get("load_error", "")


@pytest.mark.heavy
def test_structure_gate_zeroes_scoring(tmp_path):
    """A floating wheel caps the whole submission at the gate."""
    import shutil
    from harness.base_design.scorer import score_submission
    sub = tmp_path / "sub"
    sub.mkdir()
    path = _variant(tmp_path, lambda root:
                    _body(root, "wheel_FL").set("pos", "0.6 0.6 0"))
    shutil.copy(path, sub / "robot.xml")
    r = score_submission(str(sub), arm_reference_xml=ARM_REF)
    assert r["gated"] and "S1.structure" in r["gate_failed"]
    assert r["reward"] <= 0.15


# ---------------------------------------------------------------------------
# battery pack (S3.battery_onboard)
# ---------------------------------------------------------------------------

def _battery_info(path):
    import mujoco
    from harness.base_design.checkpoints import battery_onboard
    return battery_onboard(mujoco.MjModel.from_xml_path(path))


def test_base_design_golden_battery_meets_spec():
    b = _battery_info(GOLDEN)
    assert b["ok"], b["problems"]


def test_lightened_battery_fails(tmp_path):
    path = _variant(tmp_path, lambda root: next(
        g for g in root.iter("geom")
        if g.get("name") == "battery_geom").set("mass", "1.0"))
    b = _battery_info(path)
    assert not b["ok"] and any("mass" in p for p in b["problems"])


def test_shrunken_battery_fails(tmp_path):
    path = _variant(tmp_path, lambda root: next(
        g for g in root.iter("geom")
        if g.get("name") == "battery_geom").set("size", "0.02 0.02 0.02"))
    b = _battery_info(path)
    assert not b["ok"] and any("size" in p for p in b["problems"])


def test_missing_battery_fails(tmp_path):
    def mutate(root):
        base = _body(root, "base")
        base.remove(_body(root, "battery"))
    b = _battery_info(_variant(tmp_path, mutate))
    assert not b["ok"] and any("battery" in p for p in b["problems"])


def test_battery_on_the_arm_fails(tmp_path):
    """Hanging the pack off a moving link is not an installation: there must
    be no joint between battery and base."""
    def mutate(root):
        base = _body(root, "base")
        batt = _body(root, "battery")
        base.remove(batt)
        batt.set("pos", "0 0 0.1")
        _body(root, "arm_link3").append(batt)
    b = _battery_info(_variant(tmp_path, mutate))
    assert not b["ok"] and any("rigidly" in p for p in b["problems"])


# ---------------------------------------------------------------------------
# reach beyond the footprint (S3.reach_beyond)
# ---------------------------------------------------------------------------

def test_base_design_golden_reach_beyond_clears_threshold():
    from harness import config
    from harness.base_design.checkpoints import analyze_model
    info = analyze_model(GOLDEN, _panda())
    assert info["reach_beyond"] >= config.REACH_BEYOND_MIN_M
    assert info["battery"]["ok"]


def test_cornered_mount_fails_reach_beyond(tmp_path):
    """An arm parked in a corner of the base preserves nominal reach but
    cannot clear the opposite corner — the usable-reach check catches it."""
    from harness import config
    from harness.base_design.checkpoints import analyze_model

    def mutate(root):
        for g in root.iter("geom"):
            if g.get("name") == "arm_mount":
                g.set("pos", "0.2 0.2 0.115")
        for s in root.iter("site"):
            if s.get("name") == "arm_mount_site":
                s.set("pos", "0.2 0.2 0.125")
        _body(root, "arm_link0").set("pos", "0.2 0.2 0.125")
    info = analyze_model(_variant(tmp_path, mutate), _panda())
    assert info["reach"] >= info["reach_floor"]        # nominal reach intact
    assert info["reach_beyond"] < config.REACH_BEYOND_MIN_M


def test_render_submission_produces_views(tmp_path):
    from harness.base_design.render import render_submission
    from rlebench.core.media import Media
    media = Media(tmp_path)
    render_submission(media, GOLDEN, ARM_REF)
    index = media.close()
    assert index["files"] == [
        "robot_stow.png", "robot_extended.png", "robot_integration.png"]
    assert not index["skipped"]
    for name in index["files"]:
        assert os.path.getsize(tmp_path / name) > 10_000  # a real image, not a stub


def test_ground_truth_not_shipped_to_agent():
    """Task08 publishes the shelf scene, but scoring code, the golden
    controller and the reference design may never reach the agent's
    environment (invariant #2)."""
    env = os.path.join(HERE, "..", "tasks", "task08", "environment")
    reference = open(os.path.join(HERE, "..", "tasks", "task08",
                                  "reference", "robot.xml"), "rb").read()
    for root, _dirs, files in os.walk(env):
        assert "base_design" not in root.split(os.sep)
        assert "scoring" not in root.split(os.sep)
        for name in files:
            assert name not in ("pickscene.py", "golden_controller.py"), \
                f"{os.path.join(root, name)} leaks scoring code"
            if name.endswith(".xml"):
                blob = open(os.path.join(root, name), "rb").read()
                assert blob != reference, \
                    f"{os.path.join(root, name)} is the reference design"
