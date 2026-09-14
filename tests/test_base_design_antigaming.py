"""Anti-gaming regression tests: each encodes an exploit so it can never
silently reopen."""
import os

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
ASSETS = os.path.join(HERE, "..", "tasks", "task08", "harness", "assets")
GOLDEN = os.path.join(ASSETS, "golden", "robot.xml")
ARM_REF = os.path.abspath(os.path.join(HERE, "..", "assets", "robots",
                                       "franka_emika_panda", "panda_nohand.xml"))


def _panda():
    from harness.sim.arm_variants import trusted_arm_specs
    return trusted_arm_specs(ARM_REF)["panda"]


def zero_controller(model):
    """The stationary exploit: never drive, stay statically stable."""
    return lambda data, twist: np.zeros(4)


def test_stationary_controller_scores_zero_on_dynamic_battery():
    """Being stable while refusing to drive must not collect the dynamic
    battery."""
    from harness.base_design.checkpoints import score_scenario
    from harness.sim.scenarios import Envelope, run_scenario
    r = run_scenario(GOLDEN, "B", seed=101, env=Envelope(),
                     controller=zero_controller, arm=_panda())
    assert not r["tip"]                      # it IS stable...
    assert score_scenario(r) == 0.0          # ...but earns nothing


def test_honest_controller_still_aces_the_gated_scenario():
    from harness.base_design.checkpoints import score_scenario
    from harness.sim.scenarios import Envelope, run_scenario
    r = run_scenario(GOLDEN, "B", seed=101, env=Envelope(), arm=_panda())
    assert score_scenario(r) == 1.0


def test_geom_priority_is_clamped_in_composed_scenes():
    """A submitted model must not out-prioritize the terrain's contact
    parameters."""
    import xml.etree.ElementTree as ET
    from harness.sim.scenarios import Envelope, compose_scene

    # craft a variant whose rollers claim priority 5 and slippery friction
    tree = ET.parse(GOLDEN)
    root = tree.getroot()
    n_marked = 0
    for g in root.iter("geom"):
        if (g.get("name") or "").startswith("rollerg_FL"):
            g.set("priority", "5")
            g.set("friction", "0.01 0.001 0.00001")
            n_marked += 1
    assert n_marked > 0
    comp = root.find("compiler")
    comp.set("meshdir", os.path.abspath(
        os.path.join(HERE, "..", "assets", "robots",
                     "franka_emika_panda", "assets")))
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "robot.xml")
        tree.write(path)
        model, data, _ = compose_scene(path, "A", 0, Envelope(), _panda())
    for gid in range(model.ngeom):
        name = model.geom(gid).name or ""
        if name in ("floor", "curb", "bumps"):
            assert model.geom_priority[gid] == 2
        else:
            assert model.geom_priority[gid] == 0, f"{name} kept priority"


def test_structures_named_like_the_arm_do_not_survive_composition():
    """An outrigger hidden behind an arm_ name prefix cannot widen the
    support: the verifier replaces every submitted arm_* subtree with the
    canonical arm, so the outrigger neither exists nor counts."""
    import xml.etree.ElementTree as ET
    import tempfile
    from harness.base_design.checkpoints import analyze_model
    from harness.sim.scenarios import Envelope, compose_scene

    tree = ET.parse(GOLDEN)
    root = tree.getroot()
    for body in root.iter("body"):
        if body.get("name") == "base":
            wing = ET.SubElement(body, "body",
                                 dict(name="arm_wing_outrigger", pos="0 0.5 0"))
            ET.SubElement(wing, "geom", dict(
                name="arm_wing_geom", type="box", size="0.05 0.2 0.02",
                mass="0.5", rgba="1 0 0 1"))
            break
    root.find("compiler").set("meshdir", os.path.abspath(
        os.path.join(HERE, "..", "assets", "robots",
                     "franka_emika_panda", "assets")))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "robot.xml")
        tree.write(path)
        model, _, _ = compose_scene(path, "A", 0, Envelope(), _panda())
        info = analyze_model(path, _panda())
    names = {model.body(i).name for i in range(model.nbody)}
    assert "arm_wing_outrigger" not in names
    assert info["footprint"] == pytest.approx(analyze_model(GOLDEN, _panda())["footprint"])
    assert info["footprint"][1] <= 0.56


def test_base_design_golden_footprint_within_budget():
    from harness import config
    from harness.base_design.checkpoints import analyze_model
    info = analyze_model(GOLDEN, _panda())
    assert info["footprint"][0] <= config.FOOTPRINT_BUDGET_M
    assert info["footprint"][1] <= config.FOOTPRINT_BUDGET_M
    assert info["arm"]["ok"] and info["actuators"]["ok"]
