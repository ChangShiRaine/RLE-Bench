"""The MDP terms and the batched trainer.

The load-bearing test here is `test_trainer_and_evaluator_build_the_same_obs`.
Trainer and evaluator are separate code paths over separate physics backends; if
they ever disagree about what an observation means, a submission is scored on a
different world than the one it trained in, and nothing else in the harness would
notice. The MDP-term tests are analytic — values derived by hand, not captured
from the implementation.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

from harness import mdp, spec  # noqa: E402

ROBOT_DIR = "third_party/motiontrack/unitree_g1"
NPZ = "third_party/motiontrack/motions/dance1_subject2.npz"

needs_assets = pytest.mark.skipif(
    not os.path.exists(NPZ), reason="run `make sim-motiontrack` first")
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="trainer needs CUDA")

IDENTITY = torch.tensor([[1.0, 0.0, 0.0, 0.0]])


def test_reward_terms_are_one_when_tracking_is_perfect():
    pos = torch.randn(4, 14, 3)
    quat = IDENTITY.repeat(4, 1)
    body_quat = IDENTITY.repeat(4, 14, 1).view(4, 14, 4)
    assert torch.allclose(mdp.anchor_position_reward(pos[:, 0], pos[:, 0], 0.3),
                          torch.ones(4))
    assert torch.allclose(mdp.anchor_orientation_reward(quat, quat, 0.4),
                          torch.ones(4), atol=1e-6)
    assert torch.allclose(mdp.body_position_reward(pos, pos, 0.3), torch.ones(4))
    assert torch.allclose(mdp.body_orientation_reward(body_quat, body_quat, 0.4),
                          torch.ones(4), atol=1e-6)


def test_anchor_position_reward_matches_the_kernel_by_hand():
    """A 0.3 m error at std 0.3 must decay to exactly exp(-1)."""
    a = torch.zeros(1, 3)
    b = torch.tensor([[0.3, 0.0, 0.0]])
    assert mdp.anchor_position_reward(a, b, 0.3).item() == pytest.approx(np.exp(-1.0))
    assert mdp.anchor_position_reward(a, b, 0.6).item() == pytest.approx(np.exp(-0.25))


def test_body_reward_averages_squared_error_before_the_kernel():
    """Averaging after the kernel would let one well-tracked limb mask the rest."""
    ref = torch.zeros(1, 2, 3)
    got = torch.tensor([[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]])
    mean_sq = (0.0 + 0.16) / 2
    assert mdp.body_position_reward(ref, got, 0.3).item() == pytest.approx(
        np.exp(-mean_sq / 0.09))


def test_orientation_reward_uses_the_geodesic_angle():
    angle = 0.4
    quat = torch.tensor([[np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]])
    got = mdp.anchor_orientation_reward(IDENTITY, quat, 0.4).item()
    assert got == pytest.approx(np.exp(-(angle**2) / 0.16), rel=1e-5)


def test_action_rate_and_joint_limits():
    a = torch.tensor([[1.0, 2.0]])
    assert mdp.action_rate_l2(a, torch.zeros(1, 2)).item() == pytest.approx(5.0)
    assert mdp.action_rate_l2(a, a).item() == 0.0

    lo, hi = torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0])
    assert mdp.joint_pos_limits(torch.tensor([[0.0, 0.5]]), lo, hi).item() == 0.0
    assert mdp.joint_pos_limits(torch.tensor([[1.3, -1.2]]), lo, hi).item() == pytest.approx(0.5)


def test_undesired_contacts_counts_bodies_over_threshold():
    force = torch.tensor([[[3.0, 4.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 2.0]]])
    assert mdp.undesired_contacts(force, threshold=1.0).item() == 2.0


def test_termination_triggers_on_each_condition():
    n_bodies, anchor, ee = 14, 7, torch.tensor([3, 6])
    ref = torch.zeros(1, n_bodies, 3)
    ok = torch.zeros(1, n_bodies, 3)
    upright = IDENTITY

    assert not mdp.terminated(ref, ok, upright, upright, anchor, ee, 0.25, 0.8, 0.25).any()

    fallen = ok.clone()
    fallen[0, anchor, 2] = -0.3
    assert mdp.terminated(ref, fallen, upright, upright, anchor, ee, 0.25, 0.8, 0.25).all()

    limb = ok.clone()
    limb[0, 3, 2] = 0.4
    assert mdp.terminated(ref, limb, upright, upright, anchor, ee, 0.25, 0.8, 0.25).all()

    on_side = torch.tensor([[np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]])
    assert mdp.terminated(ref, ok, upright, on_side, anchor, ee, 0.25, 0.8, 0.25).all()


def test_adaptive_sampler_is_a_distribution_and_follows_failures():
    s = mdp.AdaptiveSampler(n_frames=1000, fps=50, device="cpu")
    p = s.probabilities()
    assert p.sum().item() == pytest.approx(1.0)
    assert (p > 0).all(), "every bin keeps some mass, or a region can never be revisited"

    hot = 5
    s.current = torch.zeros(s.n_bins)
    s.current[hot] = 100.0
    for _ in range(2000):
        s.decay()
        s.current[hot] = 100.0
    after = s.probabilities()
    assert after[hot] > p[hot] * 2, "repeated failures must bias sampling toward them"
    assert s.sample(64).max() < 1000


@pytest.fixture(scope="module")
def env():
    from harness.env import TrackingEnv, TrackingEnvCfg
    return TrackingEnv(
        TrackingEnvCfg(num_envs=8, obs_noise=False, randomize=False, push=False),
        ROBOT_DIR, NPZ)


@needs_assets
@needs_gpu
class TestTrainer:
    def test_reset_and_step_shapes(self, env):
        obs = env.reset()
        assert obs.shape == (8, spec.OBS_DIM)
        assert torch.isfinite(obs).all()
        obs, reward, done, info = env.step(torch.zeros(8, spec.N_ACTIONS, device="cuda"))
        assert obs.shape == (8, spec.OBS_DIM)
        assert reward.shape == (8,) and done.shape == (8,)
        assert set(info["terms"]) == set(mdp.DEFAULT_WEIGHTS)

    def test_reset_places_the_robot_on_the_reference(self, env):
        noisy = (env.cfg.init_pos_noise, env.cfg.init_joint_noise,
                 env.cfg.init_rpy_noise, env.cfg.init_vel_noise)
        env.cfg.init_pos_noise = (0.0, 0.0, 0.0)
        env.cfg.init_joint_noise = 0.0
        env.cfg.init_rpy_noise = (0.0, 0.0, 0.0)
        env.cfg.init_vel_noise = (0.0,) * 6
        try:
            env.reset()
            frame = env.time_steps
            assert torch.allclose(env.qpos[:, :3], env.ref_root_pos[frame], atol=1e-5)
            # Joints only up to the soft-limit clamp — see the test below.
            assert torch.allclose(env.qpos[:, env.qpos_idx],
                                  env.ref_joint_pos[frame], atol=0.08)
        finally:
            (env.cfg.init_pos_noise, env.cfg.init_joint_noise,
             env.cfg.init_rpy_noise, env.cfg.init_vel_noise) = noisy

    def test_reference_pose_is_clamped_to_the_soft_joint_limits(self, env):
        """The retargeted clip asks for joint angles slightly outside the 0.9
        soft-limit band on 370/1000 frames — ankles, by at most 0.077 rad. The
        reference implementation clamps identically, so this is faithful rather
        than a defect, but it means a reset cannot land exactly on the reference
        and a perfect tracking score is not physically attainable on those frames.
        Pinned so the bound cannot grow unnoticed."""
        ref = env.ref_joint_pos
        violation = torch.maximum(env.joint_lo - ref, ref - env.joint_hi).clamp(min=0)
        assert violation.max() < 0.1
        assert (violation > 0).any(-1).float().mean() < 0.5

    def test_actions_map_through_the_contract(self, env):
        from harness import robot
        env.reset()
        action = torch.full((8, spec.N_ACTIONS), 0.5, device="cuda")
        env.step(action)
        want = robot.action_to_target(np.full(spec.N_ACTIONS, 0.5))
        got = env.ctrl[0].cpu().numpy()
        inside = (want > env.ctrl_lo.cpu().numpy()) & (want < env.ctrl_hi.cpu().numpy())
        assert np.allclose(got[inside], want[inside], atol=1e-5)

    def test_trainer_and_evaluator_build_the_same_obs(self, env):
        """The contract check. Two backends, two implementations, one meaning."""
        from harness import evaluator, motion, robot
        env.reset()
        for _ in range(5):
            env.step(0.2 * torch.randn(8, spec.N_ACTIONS, device="cuda",
                                       generator=env.generator))

        model = robot.build(ROBOT_DIR)
        clip = motion.Motion.load(NPZ)
        idx = evaluator.build_index(model, clip)
        data = mujoco.MjData(model)
        data.qpos[:] = env.qpos[0].cpu().numpy()
        data.qvel[:] = env.qvel[0].cpu().numpy()
        mujoco.mj_forward(model, data)

        trainer_obs = env.observation()[0].cpu().numpy().astype(np.float64)
        eval_obs = evaluator.observation(
            model, data, clip, int(env.time_steps[0]),
            env.prev_action[0].cpu().numpy().astype(np.float64), idx)
        assert np.abs(trainer_obs - eval_obs).max() < 1e-5

    def test_stale_kinematics_would_be_visible(self, env):
        """Guards the fix, not the symptom: mjw.step integrates qpos/qvel after
        computing xpos, so without the trailing forward() the cached kinematics
        lag the state. Here xpos must agree with a fresh FK on the same qpos."""
        from harness import robot
        env.reset()
        env.step(torch.zeros(8, spec.N_ACTIONS, device="cuda"))
        model = robot.build(ROBOT_DIR)
        data = mujoco.MjData(model)
        data.qpos[:] = env.qpos[0].cpu().numpy()
        data.qvel[:] = env.qvel[0].cpu().numpy()
        mujoco.mj_forward(model, data)
        assert np.abs(data.xpos - env.xpos[0].cpu().numpy()).max() < 1e-5

    def test_same_seed_reproduces_the_same_rollout(self):
        """Reproducible to float32, not bitwise: GPU physics uses atomics whose
        summation order varies between launches, so rewards wobble at ~1e-7.
        That is fine for a trainer. Bitwise determinism is required only of the
        EVALUATOR, which runs MuJoCo-C on CPU and is pinned exactly in
        test_motiontrack_eval.py."""
        from harness.env import TrackingEnv, TrackingEnvCfg

        def rollout(seed):
            e = TrackingEnv(TrackingEnvCfg(num_envs=8, seed=seed), ROBOT_DIR, NPZ)
            e.reset()
            out = [e.step(torch.zeros(8, spec.N_ACTIONS, device="cuda"))[1] for _ in range(5)]
            return torch.stack(out).cpu().numpy()

        assert np.allclose(rollout(3), rollout(3), atol=1e-5)
        assert not np.allclose(rollout(3), rollout(4), atol=1e-5)
