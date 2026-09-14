"""Fast analytic tests for the task09 co-design harness (codesign.py,
codesign_oracle.py, sandbox ctrim/cadapt plumbing). The full ablation
hardness contract lives in tests/test_gello_codesign_tiers.py."""
import os
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from harness import codesign_variant_config as ccfg
from harness import spec as gspec
from harness.codesign import (CodesignEnvelope, clip_ff,
                                      compose_instance, headroom,
                                      hold_battery, payload_for_seed,
                                      probe_configs, qmat_grid, run_probe, validate_probe_plan)
from harness.codesign_oracle import (CODESIGN_MODELS,
                                             reference_trim_module)
from harness.codesign_checkpoints import (_cont_down, _cont_up,
                                                  evaluate_codesign)
from harness.codesign_scorer import _springs_legal
from harness.scenarios import compose_lead
from harness.torques import residual_active_torque

mujoco.set_mju_user_warning(lambda m: None)

ORACLE = os.path.join(CODESIGN_MODELS, "franka", "lead_oracle.xml")
BARE = os.path.join(CODESIGN_MODELS, "franka", "lead_bare.xml")
SREF = os.path.join(CODESIGN_MODELS, "franka", "lead_sref.xml")
ENV = CodesignEnvelope()


def test_assets_exist_and_load():
    for p in (ORACLE, BARE, SREF):
        model, _ = compose_lead(p, seed=0, perturb=False)
        assert model.nv == gspec.N_JOINTS


def test_payload_stratified_and_deterministic():
    lv = ENV.payload_levels
    assert payload_for_seed(129, ENV) == lv[0] * ENV.payload_max
    assert payload_for_seed(223, ENV) == lv[1] * ENV.payload_max
    assert payload_for_seed(311, ENV) == lv[2] * ENV.payload_max
    # composing twice gives the identical instance
    m1, _, p1 = compose_instance(ORACLE, 311, ENV)
    m2, _, p2 = compose_instance(ORACLE, 311, ENV)
    assert p1 == p2 > 0
    np.testing.assert_array_equal(m1.body_mass, m2.body_mass)


def test_payload_sits_on_handle():
    m0, _, _ = compose_instance(ORACLE, 311, ENV, payload=False)
    m1, _, pl = compose_instance(ORACLE, 311, ENV, payload=True)
    hid = m1.body("lead_handle").id
    assert m1.body_mass[hid] == pytest.approx(m0.body_mass[hid] + pl)
    others = [b for b in range(m1.nbody) if b != hid]
    np.testing.assert_array_equal(m1.body_mass[others], m0.body_mass[others])


def test_probe_protocol_deterministic():
    model, data, _ = compose_instance(SREF, 129, ENV)
    p1 = run_probe(model, data, 129, ENV)
    p2 = run_probe(model, data, 129, ENV)
    assert len(p1) == ENV.n_probe
    for (a1, b1), (a2, b2) in zip(p1, p2):
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(b1, b2)
    # probe configs are the fixed corners + seeded fill, inside the box
    cfg = probe_configs(129, ENV)
    assert cfg.shape == (ENV.n_probe, gspec.N_JOINTS)


def test_adapt_recovers_known_mass_delta():
    """System-ID ground check: perturb one instance, adapt, and the adapted
    feedforward must track the TRUE residual far better than the nominal."""
    mod = reference_trim_module()
    nominal, _ = compose_lead(ORACLE, seed=0, perturb=False)
    model, data, _ = compose_instance(ORACLE, 311, ENV)
    home = np.asarray(gspec.LEAD_HOME)
    half = np.asarray(gspec.WORKSPACE_HALFWIDTH)
    sample = run_probe(model, data, 311, ENV, [home])[0]
    plan = validate_probe_plan(mod.plan_probe(
        nominal, sample, home - half, home + half), model, ENV)
    pairs = [sample] + run_probe(model, data, 311, ENV, plan, noise_offset=1)
    assert len(pairs) == 16
    params = mod.adapt(nominal, pairs)
    ada = mod.make_trim_adapted(nominal, params)
    nom = mod.make_trim(nominal)
    grid = qmat_grid(ENV)[::7]
    err_nom, err_ada = 0.0, 0.0
    for q in grid:
        true = residual_active_torque(model, data, q)
        err_nom = max(err_nom, float(np.abs(true - nom(q)).max()))
        err_ada = max(err_ada, float(np.abs(true - ada(q)).max()))
    assert err_nom > 0.05          # nominal uses a 48 g prior, actual load is 76 g
    assert err_ada < 0.5 * err_nom
    assert err_ada < 0.07          # near the friction band


def test_frozen_ff_matches_callable_hold():
    """The frozen-feedforward contract: a hold driven by the sampled trim row
    equals (to droop tolerance) what the per-step callable would do."""
    mod = reference_trim_module()
    nominal, _ = compose_lead(ORACLE, seed=0, perturb=False)
    trim = mod.make_trim(nominal)
    model, data, _ = compose_instance(ORACLE, 223, ENV)
    from harness.balance import hold_sim
    q = np.asarray(gspec.LEAD_HOME)
    frozen = hold_sim(model, data, q, ee_site=gspec.EE_SITE,
                      trim_ff=clip_ff(trim(q)))["ee_droop"]
    stepped = hold_sim(model, data, q, ee_site=gspec.EE_SITE,
                       trim=trim)["ee_droop"]
    assert abs(frozen - stepped) < 2e-3


def test_headroom_and_continuous_credit():
    assert headroom(None) == gspec.SERVO_TAU_NM
    assert headroom(np.full((4, 7), 0.30)) == pytest.approx(0.05)
    assert headroom(np.full((4, 7), 9.0)) == pytest.approx(0.0)  # clipped
    # continuous reward: at the bar -> 1/H; H x better -> 1.0; monotone
    H = ccfg.REWARD_HEADROOM
    assert _cont_down(1.0, 1.0) == pytest.approx(1.0 / H)
    assert _cont_down(1.0 / H, 1.0) == pytest.approx(1.0)
    assert _cont_down(2.0, 1.0) == pytest.approx(0.5 / H)
    assert _cont_down(0.0, 1.0) == 1.0
    assert _cont_down(float("nan"), 1.0) == 0.0
    assert _cont_down(1.0, 1.0, headroom=2.0) == pytest.approx(0.5)
    assert _cont_down(0.5, 1.0, headroom=2.0) == pytest.approx(1.0)
    assert _cont_up(1.0, 1.0) == pytest.approx(1.0 / H)
    assert _cont_up(H, 1.0) == pytest.approx(1.0)
    assert _cont_up(0.0, 1.0) == 0.0


@pytest.mark.parametrize("ratio,expected", [(0, 1), (0.5, 1), (1, 1), (2, 0.5),
                                          (float("nan"), 0)])
def test_residual_bar_is_full_credit(ratio, expected):
    scores = evaluate_codesign({}, 0.0,
        {"device_residual": ratio * ccfg.H1_RESIDUAL_BAR,
         "passive_droop": ccfg.H2_DROOP_BAR}, {"nominal_droop": 0.0})
    assert scores["H1.device_residual"] == pytest.approx(expected)
    assert scores["H2.passive_hold"] == pytest.approx(1 / ccfg.REWARD_HEADROOM)
    assert "S1.nominal_feedforward" not in scores


def test_springs_legal_rejects_negative_stiffness(tmp_path):
    shutil.copytree(os.path.join(os.path.dirname(ORACLE), "meshes"), tmp_path / "meshes")
    bad = tmp_path / "lead.xml"
    tree = ET.parse(ORACLE)
    tree.find(".//joint[@stiffness]").set("stiffness", "-0.5")
    tree.write(bad)
    # the check is a share, so one illegal spring costs its own fraction
    assert _springs_legal(ORACLE) == 1.0
    share = _springs_legal(str(bad))
    assert 0.0 < share < 1.0
    assert share == pytest.approx(1.0 - 1.0 / gspec.N_JOINTS)


def test_sandbox_ctrim_cadapt_roundtrip(tmp_path):
    """The isolated-process software path returns the same matrices the
    in-process module produces."""
    from harness.codesign_oracle import build_reference_submission
    from harness.sandbox import run_jobs
    sub = str(tmp_path / "sub")
    build_reference_submission(sub, ORACLE)
    model, data, _ = compose_instance(ORACLE, 129, ENV)
    probe = run_probe(model, data, 129, ENV)
    qm = qmat_grid(ENV)[:5]
    out = run_jobs(sub, ORACLE, {
        "ctrim": {"g": qm},
        "cadapt": {"129": dict(probe=probe, qmats={"g": qm})}})
    assert out is not None
    mod = reference_trim_module()
    nominal, _ = compose_lead(ORACLE, seed=0, perturb=False)
    trim = mod.make_trim(nominal)
    expect = np.vstack([trim(q) for q in qm])
    np.testing.assert_allclose(out["ctrim"]["g"], expect, atol=1e-9)
    assert out["cadapt"]["129"]["g"].shape == (5, 7)


def test_buildable_accepts_legit_and_rejects_exploits(tmp_path):
    """Buildability checks: golden-geometry designs pass; boom
    counterweights, massless-connector links, and ghost handles fail."""
    shutil.copytree(os.path.join(os.path.dirname(ORACLE), "meshes"), tmp_path / "meshes")
    from harness.codesign import device_buildable
    for p in (ORACLE, BARE, SREF):
        assert device_buildable(p)["ok"]

    boom = tmp_path / "boom.xml"      # counterweight on a 0.5 m boom
    tree = ET.parse(ORACLE)
    next(g for g in tree.iter("geom") if g.get("name", "").startswith("counterweight_")).set("pos", "0 0.5 0")
    tree.write(boom)
    r = device_buildable(str(boom))
    assert not r["ok"] and any("envelope" in p for p in r["problems"])

    ghost = tmp_path / "ghost.xml"    # link2 mass concentrated at the joint
    tree = ET.parse(ORACLE)
    geom = tree.find(".//geom[@name='printed_link2']")
    geom.attrib.clear()
    geom.attrib.update(name="printed_link2", type="sphere", size="0.02", mass="0.01")
    tree.write(ghost)
    r = device_buildable(str(ghost))
    assert not r["ok"] and any("massless connector" in p for p in r["problems"])

    light = tmp_path / "light.xml"    # ghost handle
    tree = ET.parse(ORACLE)
    tree.find(".//geom[@name='printed_handle']").set("mass", "0.01")
    tree.write(light)
    r = device_buildable(str(light))
    assert not r["ok"] and any("lead_handle mass" in p for p in r["problems"])


def test_bare_arm_is_valid_but_sags():
    """The starting asset passes basic validity but lacks compensation."""
    from harness.validity import check_validity
    v = check_validity(BARE, compose=compose_lead)
    assert v["loads"] and v["structure"]["ok"] and v["density"]["ok"]
    assert v["mass_ok"] and v["link_mass_ok"]
    model, data, _ = compose_instance(BARE, 311, ENV)
    droop = hold_battery(model, data, 311, ENV)["max_ee_droop"]
    assert droop > 0.1     # collapses without a compensation design
