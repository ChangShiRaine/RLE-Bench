"""Metrics and the sim-to-sim evaluator.

The scale is pinned at both ends without needing a trained policy: kinematic
replay must score 1.0 (the reference tracking itself) and a null policy must
score near 0. Everything a submission can earn lies between those two.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from harness import evaluator, metrics, motion, robot, spec  # noqa: E402
from harness import rotations as R  # noqa: E402

ROBOT_DIR = "third_party/motiontrack/unitree_g1"
NPZ = "third_party/motiontrack/motions/dance1_subject2.npz"

pytestmark = pytest.mark.skipif(
    not os.path.exists(NPZ), reason="run `make sim-motiontrack` first")


@pytest.fixture(scope="module")
def clip():
    return motion.Motion.load(NPZ)


@pytest.fixture(scope="module")
def model():
    return robot.build(ROBOT_DIR)


def test_reanchor_is_identity_when_the_robot_is_on_the_reference():
    rng = np.random.default_rng(0)
    body_pos = rng.normal(size=(14, 3))
    body_quat = rng.normal(size=(14, 4))
    body_quat /= np.linalg.norm(body_quat, axis=-1, keepdims=True)
    anchor_pos, anchor_quat = body_pos[7], body_quat[7]
    pos, quat = metrics.reanchor(body_pos, body_quat, anchor_pos, anchor_quat,
                                 anchor_pos, anchor_quat)
    assert np.allclose(pos, body_pos, atol=1e-9)
    assert np.allclose(R.quat_error(quat, body_quat), 0.0, atol=1e-7)


def test_reanchor_absorbs_translation_and_yaw_but_not_height():
    rng = np.random.default_rng(1)
    body_pos = rng.normal(size=(5, 3))
    body_quat = np.tile([1.0, 0.0, 0.0, 0.0], (5, 1))
    anchor_pos, anchor_quat = body_pos[0], body_quat[0]

    yaw = np.array([np.cos(0.3), 0.0, 0.0, np.sin(0.3)])
    shifted = anchor_pos + np.array([2.0, -3.0, 0.0])
    pos, _ = metrics.reanchor(body_pos, body_quat, anchor_pos, anchor_quat,
                              shifted, yaw)
    # Shape survives an xy translation plus a yaw: pairwise distances unchanged.
    d0 = np.linalg.norm(body_pos[:, None] - body_pos[None], axis=-1)
    d1 = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    assert np.allclose(d0, d1, atol=1e-9)

    # Height is NOT absorbed — a sinking robot still shows error.
    raised = anchor_pos + np.array([0.0, 0.0, 0.4])
    pos_up, _ = metrics.reanchor(body_pos, body_quat, anchor_pos, anchor_quat,
                                 raised, anchor_quat)
    assert np.allclose(pos_up, body_pos, atol=1e-9)


def test_body_position_error_is_a_mean_over_bodies():
    a = np.zeros((3, 3))
    b = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]])
    assert metrics.body_position_error(a, b) == pytest.approx(2.0)


def test_gravity_tilt_is_minus_one_upright_and_zero_on_its_side():
    upright = np.array([1.0, 0.0, 0.0, 0.0])
    on_side = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
    assert metrics.gravity_tilt(upright) == pytest.approx(-1.0)
    assert metrics.gravity_tilt(on_side) == pytest.approx(0.0, abs=1e-9)


def test_tracking_quality_saturates_correctly():
    assert metrics.tracking_quality(np.array(0.0)) == pytest.approx(1.0)
    assert metrics.tracking_quality(np.array(metrics.TRACK_STD)) == pytest.approx(np.exp(-1))
    assert metrics.tracking_quality(np.array(10.0)) < 1e-9


def test_multi_scale_tracking_is_the_mean_of_its_kernels():
    error = np.array(0.1)
    expected = np.mean([np.exp(-0.1**2 / std**2)
                        for std in metrics.MULTI_STDS])
    assert metrics.tracking_quality_multi(error) == pytest.approx(expected)


def test_action_jerk_is_zero_for_a_ramp():
    ramp = np.arange(10)[:, None] * np.ones(spec.N_ACTIONS)
    assert metrics.action_jerk(ramp) == pytest.approx(0.0)
    assert metrics.action_jerk(np.zeros((2, spec.N_ACTIONS))) == 0.0


def test_observation_layout_matches_the_contract(model, clip):
    idx = evaluator.build_index(model, clip)
    data = mujoco.MjData(model)
    evaluator._init_state(model, data, clip, idx, None)
    prev = np.arange(spec.N_ACTIONS, dtype=float)
    obs = evaluator.observation(model, data, clip, 0, prev, idx)

    assert obs.shape == (spec.OBS_DIM,)
    assert np.isfinite(obs).all()
    assert np.allclose(obs[spec.obs_slice("actions")], prev)
    command = obs[spec.obs_slice("command")]
    assert np.allclose(command[:spec.N_JOINTS], clip.joint_pos[0])
    assert np.allclose(command[spec.N_JOINTS:], clip.joint_vel[0])
    # Started exactly on the reference, so the anchor offset is zero.
    assert np.linalg.norm(obs[spec.obs_slice("motion_anchor_pos_b")]) < 1e-5
    assert np.allclose(obs[spec.obs_slice("motion_anchor_ori_b")],
                       [1, 0, 0, 1, 0, 0], atol=1e-5)


def test_observation_noise_only_touches_the_noisy_terms(model, clip):
    idx = evaluator.build_index(model, clip)
    data = mujoco.MjData(model)
    evaluator._init_state(model, data, clip, idx, None)
    prev = np.zeros(spec.N_ACTIONS)
    clean = evaluator.observation(model, data, clip, 0, prev, idx)
    noisy = evaluator.observation(model, data, clip, 0, prev, idx,
                                  np.random.default_rng(0))
    for term, _ in spec.OBS_TERMS:
        sl = spec.obs_slice(term)
        changed = not np.allclose(clean[sl], noisy[sl])
        assert changed == (term in evaluator.OBS_NOISE), term


def test_kinematic_replay_scores_perfectly(model, clip):
    """Ceiling anchor. If this is not ~1.0 the metrics disagree with the motion
    file and no policy number below it means anything."""
    s = evaluator.replay_metrics(model, clip)
    assert s["tracking_score"] > 0.9999
    assert s["tracking_multi"] > 0.9999
    assert s["mpjpe_m"] < 1e-6
    assert s["anchor_pos_err_m"] < 1e-6
    assert s["survival"] == 1.0


def test_null_policy_scores_near_zero(model, clip):
    """Floor anchor. Zero actions hold the default pose and fall immediately."""
    s = evaluator.run_episode(robot.build(ROBOT_DIR), clip,
                              evaluator.NullPolicy(), seed=11).summary()
    assert s["tracking_score"] < 0.05
    assert s["fell"]
    assert s["survival"] < 0.1


def test_falling_early_cannot_buy_a_good_score(model, clip):
    """The gaming vector this metric exists to close: mean error over surviving
    frames alone REWARDS an early fall, because a policy that dies at 0.1 s has
    not had time to drift. Normalising over the full clip removes the incentive."""
    s = evaluator.run_episode(robot.build(ROBOT_DIR), clip,
                              evaluator.NullPolicy(), seed=11).summary()
    assert s["mpjpe_m"] < 0.1, "a faller does post a flattering raw error"
    assert s["tracking_score"] < 0.05, "but earns almost nothing over the clip"


def test_episodes_are_deterministic(model, clip):
    a = evaluator.run_episode(robot.build(ROBOT_DIR), clip,
                             evaluator.NullPolicy(), seed=7).summary()
    b = evaluator.run_episode(robot.build(ROBOT_DIR), clip,
                             evaluator.NullPolicy(), seed=7).summary()
    assert a == b


def test_seeds_produce_different_machines(clip):
    a = evaluator.sample_domain(np.random.default_rng(1), clip.duration)
    b = evaluator.sample_domain(np.random.default_rng(2), clip.duration)
    assert a.friction != b.friction
    assert not np.allclose(a.joint_offset, b.joint_offset)
    assert 0.3 <= a.friction <= 1.6 and 0.9 <= a.mass_scale <= 1.1


def test_randomization_actually_changes_the_model(model, clip):
    dom = evaluator.sample_domain(np.random.default_rng(3), clip.duration)
    before = robot.build(ROBOT_DIR)
    after = robot.build(ROBOT_DIR)
    evaluator.apply_domain(after, dom)
    assert not np.allclose(before.body_mass, after.body_mass)
    assert not np.allclose(before.actuator_gainprm[:, 0], after.actuator_gainprm[:, 0])
    # Mass and inertia move together, or the randomization invents new materials.
    assert np.allclose(after.body_inertia[1:] / before.body_inertia[1:],
                       dom.mass_scale)


def test_evaluator_runs_a_different_integrator_than_the_trainer():
    assert evaluator.EVAL_PHYSICS_DT != spec.PHYSICS_DT
    assert evaluator.EVAL_DECIMATION * evaluator.EVAL_PHYSICS_DT == pytest.approx(
        spec.CONTROL_DT)


def test_a_broken_policy_is_caught_not_crashed(model, clip):
    class Nan:
        def reset(self, seed): pass
        def act(self, obs): return np.full(spec.N_ACTIONS, np.nan)

    class WrongShape:
        def reset(self, seed): pass
        def act(self, obs): return np.zeros(3)

    for bad in (Nan(), WrongShape()):
        log = evaluator.run_episode(robot.build(ROBOT_DIR), clip, bad, seed=5)
        assert log.diverged
        assert log.summary()["tracking_score"] == 0.0


def test_run_episode_records_through_media(tmp_path, model, clip):
    """The episode video is evidence for a human: written through
    rlebench.core.media at the recorder's cadence, never part of the score."""
    import shutil
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    from rlebench.core.media import Media
    media = Media(tmp_path)
    writer = media.video("ep0.mp4")
    evaluator.run_episode(model, clip, evaluator.NullPolicy(), seed=11, video=writer)
    media.finish(writer)
    index = media.close()
    if index["skipped"]:
        pytest.skip(f"no offscreen GL: {index['skipped']}")
    assert index["files"] == ["ep0.mp4"]
    assert 0 < writer.frames <= clip.n_frames // evaluator.VIDEO_EVERY + 1
    assert (tmp_path / "ep0.mp4").stat().st_size > 0
