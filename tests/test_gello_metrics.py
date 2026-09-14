"""Analytic tests for the task09 gravity-compensation metrics on a
hand-checkable single-link pendulum."""
import math

import mujoco
import numpy as np

from harness.torques import (gravity_torque, residual_active_torque,
                                     spring_torque)

G = 9.81
M, L = 1.0, 0.3          # single-link test pendulum
MGL2 = M * G * L / 2.0   # gravity torque magnitude, horizontal


def pendulum_xml(stiffness=0.0, springref=0.0, frictionloss=0.0,
                 balanced=False, damping=0.0):
    """Hinge about +y at the origin, capsule along +x (horizontal at q=0).
    Positive q rotates the tip downward, so gravity pulls toward +q and the
    holding torque at q=0 is -M*G*L/2."""
    spring = (f' stiffness="{stiffness}" springref="{springref}"'
              if stiffness else "")
    counter = (f'<geom type="capsule" fromto="0 0 0 {-L} 0 0" size="0.01" mass="{M}"/>'
               if balanced else "")
    return f"""<mujoco>
  <compiler angle="radian"/>
  <option gravity="0 0 -{G}"/>
  <worldbody>
    <body name="link">
      <joint name="j" type="hinge" axis="0 1 0" damping="{damping}"
             frictionloss="{frictionloss}"{spring}/>
      <geom type="capsule" fromto="0 0 0 {L} 0 0" size="0.01" mass="{M}"/>
      {counter}
    </body>
  </worldbody>
</mujoco>"""


def load(xml):
    model = mujoco.MjModel.from_xml_string(xml)
    return model, mujoco.MjData(model)


def test_gravity_torque_single_link_horizontal():
    model, data = load(pendulum_xml())
    tau = gravity_torque(model, data, [0.0])
    assert np.isclose(tau[0], -MGL2, rtol=1e-6)


def test_gravity_torque_follows_cosine():
    model, data = load(pendulum_xml())
    for q in (-1.0, -0.3, 0.4, 1.2):
        tau = gravity_torque(model, data, [q])
        assert np.isclose(tau[0], -MGL2 * math.cos(q), rtol=1e-6)


def test_spring_torque_linear_and_zero_at_ref():
    k, q0 = 2.5, 0.3
    model, _ = load(pendulum_xml(stiffness=k, springref=q0))
    assert np.isclose(spring_torque(model, [q0])[0], 0.0, atol=1e-12)
    for q in (-0.5, 0.0, 0.7):
        assert np.isclose(spring_torque(model, [q])[0], -k * (q - q0), rtol=1e-9)


def test_spring_torque_matches_mujoco_passive():
    """Our analytic tau_s must equal MuJoCo's own passive force at zero
    velocity (convention guard)."""
    model, data = load(pendulum_xml(stiffness=1.7, springref=-0.4))
    for q in (-0.8, 0.1, 0.9):
        data.qpos[0] = q
        data.qvel[0] = 0.0
        mujoco.mj_forward(model, data)
        assert np.isclose(spring_torque(model, [q])[0],
                          data.qfrc_passive[0], atol=1e-10)


def test_spring_cancels_gravity_at_design_config():
    """Spring chosen to cancel gravity at q* => tau_a(q*) = 0 and the sim
    holds there with no actuation."""
    qstar, k = 0.5, 2.0
    q0 = qstar - MGL2 * math.cos(qstar) / k   # -k(q*-q0) = -MGL2 cos(q*)
    model, data = load(pendulum_xml(stiffness=k, springref=q0))
    tau_a = residual_active_torque(model, data, [qstar])
    assert np.isclose(tau_a[0], 0.0, atol=1e-9)
    # dynamic check: equilibrium (k > gravity slope there, so it is stable)
    data.qpos[0], data.qvel[0] = qstar, 0.0
    for _ in range(int(0.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert abs(data.qpos[0] - qstar) < 1e-3


def test_mass_balanced_link_has_zero_gravity_torque():
    """CoM on the joint axis => tau_g == 0 across the whole range."""
    model, data = load(pendulum_xml(balanced=True))
    for q in np.linspace(-1.5, 1.5, 11):
        assert abs(gravity_torque(model, data, [q])[0]) < 1e-9


def test_spring_torque_matches_sim_for_negative_stiffness():
    """MuJoCo applies negative stiffness too; the metric must agree with
    qfrc_passive rather than silently treating k<0 as no spring."""
    model, data = load(pendulum_xml(stiffness=-1.3, springref=0.2))
    for q in (-0.5, 0.0, 0.6):
        data.qpos[0] = q
        data.qvel[0] = 0.0
        mujoco.mj_forward(model, data)
        assert np.isclose(spring_torque(model, [q])[0],
                          data.qfrc_passive[0], atol=1e-10)


def test_residual_is_gravity_minus_spring():
    model, data = load(pendulum_xml(stiffness=1.2, springref=0.2))
    for q in (-0.6, 0.0, 0.8):
        g = gravity_torque(model, data, [q])
        s = spring_torque(model, [q])
        r = residual_active_torque(model, data, [q])
        assert np.isclose(r[0], g[0] - s[0], atol=1e-12)


# --- balance.py: hold / operator effort / balance quality ---------------------

from harness.balance import balance_quality, hold_sim, operator_effort  # noqa: E402

RANGE = np.linspace(-1.2, 1.2, 25)
PATH = np.stack([RANGE], axis=1)      # slow sweep across the range
PATH_DT = 0.5                          # quasi-static: ~0.1 rad/s


def test_balance_quality_zero_for_balanced_joint():
    """Perfectly mass-balanced joint across its range => quality ~ 0."""
    model, data = load(pendulum_xml(balanced=True))
    q = balance_quality(model, data, PATH)
    assert q["worst"] < 1e-9
    # unbalanced control: clearly nonzero
    model, data = load(pendulum_xml())
    assert balance_quality(model, data, PATH)["worst"] > 0.5


def test_operator_effort_zero_when_balanced_frictionless():
    model, data = load(pendulum_xml(balanced=True))
    eff = operator_effort(model, data, PATH, PATH_DT)
    assert eff["peak"] < 1e-6
    assert eff["rms"] < 1e-6


def test_operator_effort_rises_with_friction():
    fl = 0.1
    model, data = load(pendulum_xml(balanced=True, frictionloss=fl))
    eff = operator_effort(model, data, PATH, PATH_DT)
    assert np.isclose(eff["peak"], fl, rtol=1e-6)   # pure dry friction
    # and effort is dominated by the residual when unbalanced
    model, data = load(pendulum_xml(frictionloss=fl))
    assert operator_effort(model, data, PATH, PATH_DT)["peak"] > MGL2 * 0.9


def test_operator_effort_includes_feedforward_within_cap():
    """A perfect gravity feedforward removes the residual (single joint,
    |tau_g| < cap in the tested band)."""
    model, data = load(pendulum_xml())
    band = np.stack([np.linspace(1.35, 1.55, 9)], axis=1)  # |tau_g| < 0.33
    ff = np.vstack([gravity_torque(model, mujoco.MjData(model), q) for q in band])
    eff = operator_effort(model, data, band, PATH_DT, ff=ff)
    assert eff["peak"] < 1e-6


def test_hold_drift_small_when_spring_balanced_large_when_not():
    """Capped servo + spring balanced at q*: negligible drift at q*.
    Without the spring the required torque exceeds the cap => visible sag."""
    qstar, k = 0.5, 2.0
    q0 = qstar - MGL2 * math.cos(qstar) / k
    model, data = load(pendulum_xml(stiffness=k, springref=q0))
    assert hold_sim(model, data, [qstar])["max_joint_drift"] < 0.01
    model, data = load(pendulum_xml())   # no spring: tau_g(0.5) ~ 1.29 >> cap
    assert hold_sim(model, data, [qstar])["max_joint_drift"] > 0.2
