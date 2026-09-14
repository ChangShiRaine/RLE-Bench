"""Sim-based cross-checks of the metrics against the settled golden model.

These validate the MuJoCo-facing halves of the metric modules (contact
reading, force conventions) that the pure-math tests can't reach.
"""
import os

import numpy as np
import pytest

GOLDEN_SCENE = os.path.join(os.path.dirname(__file__), "..",
                            "tasks", "task08", "harness", "assets", "golden", "scene.xml")


@pytest.fixture(scope="module")
def settled():
    import mujoco
    model = mujoco.MjModel.from_xml_path(GOLDEN_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(int(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return model, data


def test_net_com_force_static_equals_weight(settled):
    """Sign/frame convention check: at rest, f_r = (0, 0, -m g)."""
    import mujoco
    from harness.metrics.fasm import net_com_force
    model, data = settled
    f = net_com_force(model, data)
    mg = mujoco.mj_getTotalmass(model) * np.linalg.norm(model.opt.gravity)
    assert f[2] == pytest.approx(-mg, rel=0.02)
    assert abs(f[0]) < 0.02 * mg and abs(f[1]) < 0.02 * mg


def test_zmp_static_equals_com_projection(settled):
    from harness.metrics.zmp import zmp
    model, data = settled
    com = data.subtree_com[0]
    assert zmp(model, data) == pytest.approx(com[:2], abs=0.005)


def test_wheel_contacts_span_the_wheelbase(settled):
    from harness.metrics.support_polygon import (
        wheel_contacts, support_polygon, contact_normal_forces)
    model, data = settled
    contacts = wheel_contacts(model, data)
    assert len(contacts) >= 4
    poly = support_polygon(contacts)
    assert len(poly) >= 4
    # wheels at x=+-0.19, y=+-0.235: hull must span at least that rectangle
    assert poly[:, 0].max() >= 0.18 and poly[:, 0].min() <= -0.18
    assert poly[:, 1].max() >= 0.22 and poly[:, 1].min() <= -0.22
    # all four wheels loaded, roughly evenly (within 3x of each other)
    forces = contact_normal_forces(model, data)
    assert set(forces) == {"FL", "FR", "RL", "RR"}
    assert max(forces.values()) < 3 * min(forces.values())


def test_fasm_static_matches_pure_gravity(settled):
    """Contact-derived net force and analytic gravity agree at rest."""
    import mujoco
    from harness.metrics.fasm import net_com_force, force_angle_stability
    from harness.metrics.support_polygon import wheel_contacts, support_polygon_3d
    model, data = settled
    com = np.asarray(data.subtree_com[0])
    poly3 = support_polygon_3d(wheel_contacts(model, data))
    mg = mujoco.mj_getTotalmass(model) * np.linalg.norm(model.opt.gravity)
    m_contact = force_angle_stability(com, poly3, net_com_force(model, data))
    m_gravity = force_angle_stability(com, poly3, np.array([0, 0, -mg]))
    assert m_contact > 0
    assert m_contact == pytest.approx(m_gravity, rel=0.05)


def test_ssm_static_positive_and_geometric(settled):
    from harness.metrics.ssm import static_stability_margin
    from harness.metrics.support_polygon import wheel_contacts, support_polygon
    model, data = settled
    com = np.asarray(data.subtree_com[0])
    poly = support_polygon(wheel_contacts(model, data))
    ssm = static_stability_margin(com[:2], poly)
    # CoM near center; nearest edge is the wheelbase half-extent (~0.19-0.20)
    assert 0.15 < ssm < 0.25
