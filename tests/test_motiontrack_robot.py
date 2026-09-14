"""The frozen contract and the G1 actuation.

These tests exist to make the contract expensive to move by accident. Gains and
action scale are re-derived here straight from BeyondMimic's own formulas rather
than copied from robot.py, so the test checks the port instead of agreeing with it.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from harness import robot, spec  # noqa: E402

ROBOT_DIR = "third_party/motiontrack/unitree_g1"
G1_XML = ROBOT_DIR + "/g1.xml"

pytestmark = pytest.mark.skipif(
    not os.path.exists(G1_XML), reason="run `make sim-motiontrack` first")


@pytest.fixture(scope="module")
def model():
    return robot.build(ROBOT_DIR)


def test_contract_dimensions():
    assert spec.N_JOINTS == 29
    assert spec.OBS_DIM == 160
    assert spec.N_ACTIONS == 29
    assert spec.DECIMATION == 4
    # One motion frame per control step is what lets the reference be indexed by
    # step count instead of interpolated at runtime.
    assert spec.CONTROL_DT * spec.MOTION_FPS == 1.0


def test_obs_terms_tile_the_vector_exactly():
    covered = np.zeros(spec.OBS_DIM, dtype=int)
    for name, _ in spec.OBS_TERMS:
        covered[spec.obs_slice(name)] += 1
    assert (covered == 1).all()


def test_obs_slice_rejects_unknown_terms():
    with pytest.raises(KeyError):
        spec.obs_slice("not_a_term")


def test_joint_names_match_the_model_in_order(model):
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
           for j in spec.JOINT_NAMES]
    assert all(i >= 0 for i in ids)
    assert ids == sorted(ids), "contract order must match the model's joint order"
    assert model.nu == spec.N_JOINTS


def test_tracked_bodies_exist(model):
    for name in spec.TRACKED_BODIES + (spec.ANCHOR_BODY, spec.ROOT_BODY) + spec.FOOT_BODIES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
    assert spec.ANCHOR_BODY in spec.TRACKED_BODIES
    assert len(set(spec.TRACKED_BODIES)) == 14


def test_cylinder_colliders_are_gone(model):
    """MuJoCo-Warp degrades CYLINDER contacts; leaving them in would be an
    accidental trainer/evaluator divergence."""
    assert (model.geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER).sum() == 0


def test_sphere_substitution_preserves_extent(model):
    """Indices shift between g1.xml and the compiled scene, so match on the body
    each collider hangs off."""
    raw = mujoco.MjModel.from_xml_path(G1_XML)
    raw_body = lambda g: mujoco.mj_id2name(  # noqa: E731
        raw, mujoco.mjtObj.mjOBJ_BODY, raw.geom_bodyid[g])
    built_body = lambda g: mujoco.mj_id2name(  # noqa: E731
        model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])

    cylinders = {raw_body(g): raw.geom_size[g][:2].copy() for g in range(raw.ngeom)
                 if raw.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER}
    assert len(cylinders) == 4
    for body, (radius, half_len) in cylinders.items():
        assert half_len < radius, "a capsule would be the better swap otherwise"
        spheres = [model.geom_size[g][0] for g in range(model.ngeom)
                   if built_body(g) == body
                   and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
        assert radius in spheres
        assert abs(radius - half_len) < 0.02, "extent preserved to within 20 mm"


def test_gains_follow_the_motor_sizing(model):
    """kp = armature * w^2 and kv = 2 * zeta * armature * w, at w = 10 Hz, zeta = 2."""
    w = 10.0 * 2.0 * np.pi
    for i, joint in enumerate(spec.JOINT_NAMES):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        armature = model.dof_armature[model.jnt_dofadr[jid]]
        assert model.actuator_gainprm[aid, 0] == pytest.approx(armature * w**2)
        assert model.actuator_biasprm[aid, 1] == pytest.approx(-armature * w**2)
        assert model.actuator_biasprm[aid, 2] == pytest.approx(
            -2.0 * 2.0 * armature * w)
        assert model.actuator_forcerange[aid, 1] == pytest.approx(robot.EFFORT[i])


def test_paired_joints_carry_two_motors():
    idx = {j: i for i, j in enumerate(spec.JOINT_NAMES)}
    single = robot.ARMATURES[idx["left_shoulder_pitch_joint"]]
    for joint in ("left_ankle_pitch_joint", "left_ankle_roll_joint",
                  "waist_roll_joint", "waist_pitch_joint"):
        assert robot.ARMATURES[idx[joint]] == pytest.approx(2 * single)


def test_action_scale_matches_reference_values():
    """0.25 * torque_limit / stiffness, spot-checked against BeyondMimic's table."""
    idx = {j: i for i, j in enumerate(spec.JOINT_NAMES)}
    expected = {
        "left_hip_pitch_joint": 0.25 * 88.0 / (0.010177520 * (10 * 2 * np.pi) ** 2),
        "left_knee_joint": 0.25 * 139.0 / (0.025101925 * (10 * 2 * np.pi) ** 2),
        "left_ankle_roll_joint": 0.25 * 50.0 / (2 * 0.003609725 * (10 * 2 * np.pi) ** 2),
        "left_wrist_yaw_joint": 0.25 * 5.0 / (0.00425 * (10 * 2 * np.pi) ** 2),
    }
    for joint, want in expected.items():
        assert robot.ACTION_SCALE[idx[joint]] == pytest.approx(want, rel=1e-9)


def test_gains_are_not_the_menagerie_defaults(model):
    """Menagerie drives every joint at kp=500; the real G1 is far softer, and a
    policy trained against the wrong stiffness does not transfer."""
    kp = model.actuator_gainprm[:spec.N_JOINTS, 0]
    assert kp.max() < 200.0
    assert kp.std() > 10.0, "per-joint sizing, not one flat gain"


def test_default_pose_matches_the_reference():
    idx = {j: i for i, j in enumerate(spec.JOINT_NAMES)}
    for joint, want in {
        "left_hip_pitch_joint": -0.312, "right_hip_pitch_joint": -0.312,
        "left_knee_joint": 0.669, "right_knee_joint": 0.669,
        "left_ankle_pitch_joint": -0.363, "left_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2, "right_shoulder_roll_joint": -0.2,
        "left_shoulder_pitch_joint": 0.2, "right_shoulder_pitch_joint": 0.2,
        "waist_yaw_joint": 0.0, "left_wrist_yaw_joint": 0.0,
    }.items():
        assert robot.DEFAULT_JOINT_POS[idx[joint]] == pytest.approx(want)


def test_action_to_target_is_an_offset_from_the_default_pose():
    zero = robot.action_to_target(np.zeros(spec.N_ACTIONS))
    assert np.allclose(zero, robot.DEFAULT_JOINT_POS)
    one = robot.action_to_target(np.ones(spec.N_ACTIONS))
    assert np.allclose(one - zero, robot.ACTION_SCALE)


def _standing_height(model):
    """Height at which the foot contact spheres just touch the floor. Mesh geoms
    carry no usable extent in geom_size, so only the spheres can answer this."""
    data = mujoco.MjData(model)
    data.qpos[:3] = (0.0, 0.0, 1.0)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[robot.joint_qpos_index(model)] = robot.DEFAULT_JOINT_POS
    mujoco.mj_forward(model, data)
    feet = _foot_spheres(model)
    return 1.0 - min(data.geom_xpos[g][2] - model.geom_size[g][0] for g in feet)


def _foot_spheres(model):
    body = lambda g: mujoco.mj_id2name(  # noqa: E731
        model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
    return [g for g in range(model.ngeom)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
            and model.geom_contype[g] and "ankle_roll" in (body(g) or "")]


def _drop(model, seconds=3.0):
    """Final pelvis height after holding ctrl at the default pose."""
    data = mujoco.MjData(model)
    data.qpos[:3] = (0.0, 0.0, _standing_height(model))
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[robot.joint_qpos_index(model)] = robot.DEFAULT_JOINT_POS
    data.ctrl[:] = robot.DEFAULT_JOINT_POS
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all(), "simulation diverged"
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.ROOT_BODY)
    return float(data.xpos[pelvis][2])


def test_model_is_physically_sane(model):
    assert 30.0 < model.body_mass.sum() < 40.0
    assert (model.body_mass[1:] > 0).all()
    assert (model.body_inertia[1:] > 0).all()
    assert len(_foot_spheres(model)) == 8
    assert 0.7 < _standing_height(model) < 0.85


def test_the_robot_is_compliant_by_design(model):
    """Sized from a 10 Hz natural frequency, BeyondMimic's gains are far softer
    than menagerie's flat kp=500: the G1 buckles under its own weight at the
    default crouch within a second. That is the reference method's design and not
    a defect — the policy supplies the missing torque, and episodes always start
    from a reference state rather than from passive standing.

    Pinned because it decides the harness: a scripted PD tracker cannot be the
    golden anchor (it falls in 0.9 s on the reference clip), so the metric scale
    is anchored by kinematic replay and a trained policy instead.
    """
    assert _drop(model) < 0.4, "compliant gains should not hold the crouch"

    stiff = robot.build(ROBOT_DIR)
    stiff.actuator_gainprm[:spec.N_JOINTS, 0] = 500.0
    stiff.actuator_biasprm[:spec.N_JOINTS, 1] = -500.0
    assert _drop(stiff) > 0.6, "stiff gains should hold it — isolates gains as the cause"
