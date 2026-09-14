"""Batched motion-tracking environment on MuJoCo-Warp — the trainer's half.

Thousands of G1s step on the GPU while the reference clip, the rewards and the
resets stay in torch on the same device; warp arrays are exposed as zero-copy
torch views, so nothing crosses the PCI bus in the inner loop.

What a submission owns: the algorithm, the network, the hyperparameters, the
reward weights and kernels, the terminations, the randomization, and where
episodes start. What it does not own is `observation()` and the action mapping —
those are the deployment contract the evaluator relies on (see spec.py), and the
evaluator builds them the same way for a policy it never trained.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

import mujoco
import mujoco_warp as mjw
import warp as wp

from . import mdp, metrics, motion as motion_mod, robot, rotations as R, spec


@dataclass
class TrackingEnvCfg:
    num_envs: int = 4096
    device: str = "cuda"
    episode_s: float = spec.EPISODE_S
    physics_dt: float = spec.PHYSICS_DT
    seed: int = 0
    weights: dict = field(default_factory=lambda: dict(mdp.DEFAULT_WEIGHTS))
    std: dict = field(default_factory=lambda: dict(mdp.DEFAULT_STD))
    adaptive_sampling: bool = True
    obs_noise: bool = True
    push: bool = True
    randomize: bool = True
    contact_threshold: float = 1.0

    nconmax: int = 128
    njmax: int = 256

    init_pos_noise: tuple = (0.05, 0.05, 0.01)
    init_rpy_noise: tuple = (0.1, 0.1, 0.2)
    init_joint_noise: float = 0.1
    init_vel_noise: tuple = (0.5, 0.5, 0.2, 0.52, 0.52, 0.78)

OBS_NOISE = {
    "motion_anchor_pos_b": 0.25,
    "motion_anchor_ori_b": 0.05,
    "base_lin_vel": 0.5,
    "base_ang_vel": 0.2,
    "joint_pos_rel": 0.01,
    "joint_vel_rel": 0.5,
}


class TrackingEnv:

    def __init__(self, cfg: TrackingEnvCfg, robot_dir: str, motion_path: str):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.num_envs = cfg.num_envs
        self.max_steps = round(cfg.episode_s / spec.CONTROL_DT)

        self.model = robot.build(robot_dir, timestep=cfg.physics_dt)
        self.motion = motion_mod.Motion.load(motion_path)
        self._to_device()

        mj_data = mujoco.MjData(self.model)
        self.wm = mjw.put_model(self.model)
        self.wd = mjw.put_data(self.model, mj_data, nworld=cfg.num_envs,
                               nconmax=cfg.nconmax, njmax=cfg.njmax)
        if cfg.randomize:
            self._randomize_model()

        self.qpos = wp.to_torch(self.wd.qpos)
        self.qvel = wp.to_torch(self.wd.qvel)
        self.ctrl = wp.to_torch(self.wd.ctrl)
        self.xpos = wp.to_torch(self.wd.xpos)
        self.xquat = wp.to_torch(self.wd.xquat)
        self.cvel = wp.to_torch(self.wd.cvel)
        self.subtree_com = wp.to_torch(self.wd.subtree_com)
        self.cfrc_ext = wp.to_torch(self.wd.cfrc_ext)

        self.generator = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.sampler = mdp.AdaptiveSampler(self.motion.n_frames, self.motion.fps,
                                           self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_steps = torch.zeros_like(self.time_steps)
        self.prev_action = torch.zeros(self.num_envs, spec.N_ACTIONS, device=self.device)
        self.push_countdown = torch.zeros(self.num_envs, device=self.device)
        self.substeps = round(spec.CONTROL_DT / cfg.physics_dt)
        self._graph = None
        self._forward_graph = None

    def _to_device(self):
        m, mo = self.model, self.motion
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32,
                                      device=self.device)
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
               for j in spec.JOINT_NAMES]
        self.qpos_idx = torch.as_tensor([m.jnt_qposadr[i] for i in jid], device=self.device)
        self.qvel_idx = torch.as_tensor([m.jnt_dofadr[i] for i in jid], device=self.device)
        self.tracked = torch.as_tensor(robot.body_index(m, spec.TRACKED_BODIES),
                                       device=self.device)
        self.anchor = int(robot.body_index(m, [spec.ANCHOR_BODY])[0])
        self.root = int(robot.body_index(m, [spec.ROOT_BODY])[0])
        self.anchor_in_tracked = spec.TRACKED_BODIES.index(spec.ANCHOR_BODY)
        self.ee_in_tracked = torch.as_tensor(
            [spec.TRACKED_BODIES.index(b) for b in spec.END_EFFECTORS], device=self.device)
        self.body_rootid = torch.as_tensor(np.asarray(m.body_rootid), device=self.device)

        keep = set(spec.END_EFFECTORS)
        self.contact_bodies = torch.as_tensor(
            [i for i in range(1, m.nbody)
             if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) not in keep],
            device=self.device)

        mt = mo.body_index(list(spec.TRACKED_BODIES))
        ma = int(mo.body_index([spec.ANCHOR_BODY])[0])
        self.ref_root = int(mo.body_index([spec.ROOT_BODY])[0])
        self.ref_joint_pos, self.ref_joint_vel = t(mo.joint_pos), t(mo.joint_vel)
        self.ref_body_pos, self.ref_body_quat = t(mo.body_pos_w[:, mt]), t(mo.body_quat_w[:, mt])
        self.ref_body_lin, self.ref_body_ang = t(mo.body_lin_vel_w[:, mt]), t(mo.body_ang_vel_w[:, mt])
        self.ref_anchor_pos, self.ref_anchor_quat = t(mo.body_pos_w[:, ma]), t(mo.body_quat_w[:, ma])
        self.ref_root_pos, self.ref_root_quat = t(mo.body_pos_w[:, self.ref_root]), t(mo.body_quat_w[:, self.ref_root])
        self.ref_root_lin, self.ref_root_ang = t(mo.body_lin_vel_w[:, self.ref_root]), t(mo.body_ang_vel_w[:, self.ref_root])

        self.default_pos = t(robot.DEFAULT_JOINT_POS)
        self.action_scale = t(robot.ACTION_SCALE)
        self.ctrl_lo = t(m.actuator_ctrlrange[:, 0])
        self.ctrl_hi = t(m.actuator_ctrlrange[:, 1])
        limits = m.jnt_range[jid]
        self.joint_lo, self.joint_hi = t(limits[:, 0] * 0.9), t(limits[:, 1] * 0.9)

    def _randomize_model(self):
        rng = np.random.default_rng(self.cfg.seed)
        n = self.num_envs
        for field_name, sample in (
            ("body_mass", lambda: rng.uniform(0.9, 1.1, (n, 1))),
            ("body_inertia", lambda: rng.uniform(0.9, 1.1, (n, 1, 1))),
        ):
            arr = getattr(self.wm, field_name)
            base = arr.numpy()[0]
            wp_arr = wp.array(np.broadcast_to(base, (n,) + base.shape) * sample(),
                              dtype=arr.dtype, device=arr.device)
            setattr(self.wm, field_name, wp_arr)
        friction = self.wm.geom_friction.numpy()[0]
        scaled = np.broadcast_to(friction, (n,) + friction.shape).copy()
        scaled[:, :, 0] = rng.uniform(0.3, 1.6, (n, 1))
        self.wm.geom_friction = wp.array(scaled, dtype=self.wm.geom_friction.dtype,
                                         device=self.wm.geom_friction.device)

    def _body_velocity(self, bodies):
        ang = self.cvel[:, bodies, :3]
        offset = self.xpos[:, bodies] - self.subtree_com[:, self.body_rootid[bodies]]
        lin = self.cvel[:, bodies, 3:] + torch.linalg.cross(ang, offset)
        return lin, ang

    def _root_velocity_body_frame(self):
        lin, ang = self._body_velocity(torch.tensor([self.root], device=self.device))
        quat = self.xquat[:, self.root]
        return (R.quat_rotate_inverse(quat, lin[:, 0]),
                R.quat_rotate_inverse(quat, ang[:, 0]))

    def observation(self) -> torch.Tensor:
        frame = self.time_steps
        anchor_pos, anchor_quat = self.xpos[:, self.anchor], self.xquat[:, self.anchor]
        ref_pos, ref_quat = self.ref_anchor_pos[frame], self.ref_anchor_quat[frame]
        lin, ang = self._root_velocity_body_frame()

        obs = torch.cat([
            self.ref_joint_pos[frame],
            self.ref_joint_vel[frame],
            R.quat_rotate_inverse(anchor_quat, ref_pos - anchor_pos),
            R.ori6(R.quat_mul(R.quat_conj(anchor_quat), ref_quat)),
            lin,
            ang,
            self.qpos[:, self.qpos_idx] - self.default_pos,
            self.qvel[:, self.qvel_idx],
            self.prev_action,
        ], dim=-1)

        if self.cfg.obs_noise:
            for term, scale in OBS_NOISE.items():
                sl = spec.obs_slice(term)
                noise = torch.rand(self.num_envs, sl.stop - sl.start,
                                   device=self.device, generator=self.generator)
                obs[:, sl] += (2.0 * noise - 1.0) * scale
        return obs

    def privileged_observation(self) -> torch.Tensor:
        obs = self.observation()
        anchor_pos = self.xpos[:, self.anchor].unsqueeze(1)
        anchor_quat = self.xquat[:, self.anchor].unsqueeze(1)
        body_pos = R.quat_rotate_inverse(anchor_quat, self.xpos[:, self.tracked] - anchor_pos)
        body_ori = R.ori6(R.quat_mul(R.quat_conj(anchor_quat), self.xquat[:, self.tracked]))
        head = spec.obs_slice("motion_anchor_ori_b").stop
        return torch.cat([obs[:, :head], body_pos.flatten(1), body_ori.flatten(1),
                          obs[:, head:]], dim=-1)

    def reset(self, env_ids=None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return self.observation()

        n = len(env_ids)
        frames = (self.sampler.sample(n, self.generator) if self.cfg.adaptive_sampling
                  else torch.randint(0, self.motion.n_frames, (n,), device=self.device,
                                     generator=self.generator))
        self.time_steps[env_ids] = frames
        self.episode_steps[env_ids] = 0

        u = lambda shape, scale: (2.0 * torch.rand(
            shape, device=self.device, generator=self.generator) - 1.0) * torch.as_tensor(
            scale, dtype=torch.float32, device=self.device)

        self.qpos[env_ids, :3] = self.ref_root_pos[frames] + u((n, 3), self.cfg.init_pos_noise)
        delta = _quat_from_rpy(u((n, 3), self.cfg.init_rpy_noise))
        self.qpos[env_ids, 3:7] = R.quat_mul(delta, self.ref_root_quat[frames])
        joints = self.ref_joint_pos[frames] + u((n, spec.N_JOINTS), self.cfg.init_joint_noise)
        self.qpos[env_ids[:, None], self.qpos_idx] = torch.clamp(
            joints, self.joint_lo, self.joint_hi)

        vel = u((n, 6), self.cfg.init_vel_noise)
        self.qvel[env_ids, :3] = self.ref_root_lin[frames] + vel[:, :3]
        self.qvel[env_ids, 3:6] = R.quat_rotate_inverse(
            self.ref_root_quat[frames], self.ref_root_ang[frames]) + vel[:, 3:]
        self.qvel[env_ids[:, None], self.qvel_idx] = self.ref_joint_vel[frames]

        self.prev_action[env_ids] = 0.0
        self.push_countdown[env_ids] = _push_interval(n, self.device, self.generator)
        self._forward()
        return self.observation()

    def step(self, action: torch.Tensor):
        action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        target = self.default_pos + action * self.action_scale
        self.ctrl[:] = torch.clamp(target, self.ctrl_lo, self.ctrl_hi)

        if self.cfg.push:
            self._maybe_push()
        self._physics()

        self.time_steps = torch.clamp(self.time_steps + 1, max=self.motion.n_frames - 1)
        self.episode_steps += 1
        reference = self._reference()
        reward, terms = self._reward(action, reference)
        done, timeout = self._done(reference)

        self.sampler.record(self.time_steps, done & ~timeout)
        self.sampler.decay()
        self.prev_action = action

        obs = self.observation()
        if done.any():
            obs = self.reset(done.nonzero(as_tuple=False).squeeze(-1))
        return obs, reward, done, {"terms": terms, "timeout": timeout}

    def _physics(self):
        if self._graph is None:
            for _ in range(self.substeps):
                mjw.step(self.wm, self.wd)
            with wp.ScopedCapture() as capture:
                for _ in range(self.substeps):
                    mjw.step(self.wm, self.wd)
                mjw.forward(self.wm, self.wd)
            self._graph = capture.graph
        wp.capture_launch(self._graph)

    def _forward(self):
        if self._forward_graph is None:
            mjw.forward(self.wm, self.wd)
            with wp.ScopedCapture() as capture:
                mjw.forward(self.wm, self.wd)
            self._forward_graph = capture.graph
        wp.capture_launch(self._forward_graph)

    def _maybe_push(self):
        self.push_countdown -= spec.CONTROL_DT
        hit = self.push_countdown <= 0.0
        if hit.any():
            ids = hit.nonzero(as_tuple=False).squeeze(-1)
            scale = torch.as_tensor([0.5, 0.5, 0.2, 0.52, 0.52, 0.78], device=self.device)
            kick = (2.0 * torch.rand(len(ids), 6, device=self.device,
                                     generator=self.generator) - 1.0) * scale
            self.qvel[ids, :6] += kick
            self.push_countdown[ids] = _push_interval(len(ids), self.device, self.generator)

    def _reference(self):
        frame = self.time_steps
        anchor_pos, anchor_quat = self.xpos[:, self.anchor], self.xquat[:, self.anchor]
        ref_pos, ref_quat = metrics.reanchor(
            self.ref_body_pos[frame], self.ref_body_quat[frame],
            self.ref_anchor_pos[frame], self.ref_anchor_quat[frame],
            anchor_pos, anchor_quat)
        return ref_pos, ref_quat, anchor_pos, anchor_quat

    def _reward(self, action, reference):
        frame = self.time_steps
        ref_pos, ref_quat, anchor_pos, anchor_quat = reference
        body_pos, body_quat = self.xpos[:, self.tracked], self.xquat[:, self.tracked]
        lin, ang = self._body_velocity(self.tracked)
        w, std = self.cfg.weights, self.cfg.std
        joints = self.qpos[:, self.qpos_idx]

        terms = {
            "anchor_pos": mdp.anchor_position_reward(
                self.ref_anchor_pos[frame], anchor_pos, std["anchor_pos"]),
            "anchor_ori": mdp.anchor_orientation_reward(
                self.ref_anchor_quat[frame], anchor_quat, std["anchor_ori"]),
            "body_pos": mdp.body_position_reward(ref_pos, body_pos, std["body_pos"]),
            "body_ori": mdp.body_orientation_reward(ref_quat, body_quat, std["body_ori"]),
            "body_lin_vel": mdp.body_linear_velocity_reward(
                self.ref_body_lin[frame], lin, std["body_lin_vel"]),
            "body_ang_vel": mdp.body_angular_velocity_reward(
                self.ref_body_ang[frame], ang, std["body_ang_vel"]),
            "action_rate": mdp.action_rate_l2(action, self.prev_action),
            "joint_limits": mdp.joint_pos_limits(joints, self.joint_lo, self.joint_hi),
            "undesired_contacts": mdp.undesired_contacts(
                self.cfrc_ext[:, self.contact_bodies, 3:], self.cfg.contact_threshold),
        }
        total = sum(w[k] * v for k, v in terms.items() if k in w)
        return total * spec.CONTROL_DT, {k: v.detach() for k, v in terms.items()}

    def _done(self, reference):
        ref_pos, _, _, anchor_quat = reference
        failed = mdp.terminated(
            ref_pos, self.xpos[:, self.tracked],
            self.ref_anchor_quat[self.time_steps], anchor_quat,
            self.anchor_in_tracked, self.ee_in_tracked,
            spec.TERM_ANCHOR_Z, spec.TERM_ANCHOR_TILT, spec.TERM_EE_Z)
        blew_up = ~torch.isfinite(self.qpos).all(dim=-1)
        timeout = ((self.episode_steps >= self.max_steps)
                   | (self.time_steps >= self.motion.n_frames - 1))
        return failed | blew_up | timeout, timeout


def _push_interval(n, device, generator):
    return 1.0 + 2.0 * torch.rand(n, device=device, generator=generator)


def _quat_from_rpy(rpy):
    half = 0.5 * rpy
    c, s = torch.cos(half), torch.sin(half)
    return torch.stack([
        c[:, 0] * c[:, 1] * c[:, 2] + s[:, 0] * s[:, 1] * s[:, 2],
        s[:, 0] * c[:, 1] * c[:, 2] - c[:, 0] * s[:, 1] * s[:, 2],
        c[:, 0] * s[:, 1] * c[:, 2] + s[:, 0] * c[:, 1] * s[:, 2],
        c[:, 0] * c[:, 1] * s[:, 2] - s[:, 0] * s[:, 1] * c[:, 2],
    ], dim=-1)
