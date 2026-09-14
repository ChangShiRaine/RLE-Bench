"""Stage-1 validity checks on golden vs broken models."""
import os

import pytest

ASSETS = os.path.join(os.path.dirname(__file__), "..", "tasks", "task08", "harness", "assets")
GOLDEN_SCENE = os.path.join(ASSETS, "golden", "scene.xml")


def test_base_design_golden_loads_and_inertias_valid():
    from rlebench.core.model import load, check_inertias_valid
    model, data = load(GOLDEN_SCENE)
    report = check_inertias_valid(model)
    bad = {k: v for k, v in report.items() if not v["ok"]}
    assert not bad, f"golden has invalid inertias: {bad}"


def test_base_design_golden_mass_and_com_sane():
    from rlebench.core.model import load, total_mass_and_com
    model, data = load(GOLDEN_SCENE)
    mass, com = total_mass_and_com(model, data)
    assert 45.0 < mass < 60.0          # base ~36 kg + arm ~17 kg
    assert 0.15 < com[2] < 0.40        # CoM low enough to be plausibly stable


def test_base_design_golden_settles_at_rest():
    from rlebench.core.model import load, settles_to_equilibrium
    model, data = load(GOLDEN_SCENE)
    result = settles_to_equilibrium(model, data)
    assert result["ok"], f"golden does not settle: {result}"


# An explicit inertia that violates the triangle inequality
# (Ixx + Iyy < Izz). Built inline so the check has a fixture of its own
# rather than depending on a particular robot model.
BAD_INERTIA_XML = """<mujoco model="bad_inertia">
  <worldbody>
    <body name="base" pos="0 0 0.2">
      <freejoint/>
      <inertial pos="0 0 0" mass="10" diaginertia="0.05 0.057 0.9"/>
      <geom type="box" size="0.2 0.2 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def bad_inertia_xml(tmp_path):
    p = tmp_path / "bad_inertia.xml"
    p.write_text(BAD_INERTIA_XML)
    return str(p)


def test_unrealizable_inertia_fails_to_compile(bad_inertia_xml):
    """An inertia that violates the triangle inequality must make MuJoCo
    refuse the model outright."""
    from rlebench.core.model import load
    with pytest.raises(Exception, match="inertia"):
        load(bad_inertia_xml)


def test_unrealizable_inertia_detected_from_xml(bad_inertia_xml):
    """The harness must locate an invalid inertia without compiling."""
    from rlebench.core.model import check_inertias_valid_xml
    report = check_inertias_valid_xml(bad_inertia_xml)
    assert "base" in report
    assert not report["base"]["ok"]
    assert not report["base"]["triangle"]


def test_base_design_golden_xml_has_no_explicit_inertia_flaws():
    from rlebench.core.model import check_inertias_valid_xml
    golden_robot = os.path.join(ASSETS, "golden", "robot.xml")
    report = check_inertias_valid_xml(golden_robot)
    bad = {k: v for k, v in report.items() if not v["ok"]}
    assert not bad
