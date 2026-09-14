"""Sim-to-sim evaluation: run a submitted policy in MuJoCo-C and measure tracking.

The trainer steps MuJoCo-Warp at 5 ms; this steps MuJoCo-C at 2 ms, with the
dynamics perturbed per seed. That pair — a different backend and a different
machine — is the gap the score is meant to expose. A policy that has memorised
one integrator does not survive it.

This module is the harness's own instrumentation. It re-executes the submitted
checkpoint and measures everything itself; nothing the submission reports is read.

Episodes can be recorded through rlebench.core.media: `run_episode(video=...)`
takes an mp4 path (written and closed here) or an open VideoWriter the caller
finishes. The recorder reads the state the loop already computed, one frame
every VIDEO_EVERY control ticks, and never steps physics. Without ffmpeg or a
GL context there is no recorder at all, so the episode runs exactly as it
would unrecorded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

import mujoco

from rlebench.core.media import VIDEO_FPS, VIDEO_SIZE, MujocoCamera, VideoWriter
from rlebench.core.media import Recorder as _Recorder

from . import export, metrics, robot, rotations as R, spec

EVAL_PHYSICS_DT = 0.002
EVAL_DECIMATION = round(spec.CONTROL_DT / EVAL_PHYSICS_DT)
VIDEO_EVERY = round(1.0 / (spec.CONTROL_DT * VIDEO_FPS))
OBS_NOISE = {
    "motion_anchor_pos_b": 0.25,
    "motion_anchor_ori_b": 0.05,
    "base_lin_vel": 0.5,
    "base_ang_vel": 0.2,
    "joint_pos_rel": 0.01,
    "joint_vel_rel": 0.5,
}
PUSH_LIN = 0.5
PUSH_ANG = 0.78
PUSH_INTERVAL_S = (1.0, 3.0)


@dataclass
class Domain:
    friction: float
    com_offset: np.ndarray
    joint_offset: np.ndarray
    mass_scale: float
    kp_scale: float
    kv_scale: float
    latency_steps: int
    push_times: np.ndarray
    push_vel: np.ndarray
    init_pos: np.ndarray
    init_rpy: np.ndarray
    init_vel: np.ndarray
    init_joint: np.ndarray
    noise_seed: int


def sample_domain(rng: np.random.Generator, duration: float) -> Domain:
    n_push = max(1, int(duration / np.mean(PUSH_INTERVAL_S)))
    return Domain(
        friction=float(rng.uniform(0.3, 1.6)),
        com_offset=rng.uniform([-0.025, -0.05, -0.05], [0.025, 0.05, 0.05]),
        joint_offset=rng.uniform(-0.01, 0.01, spec.N_JOINTS),
        mass_scale=float(rng.uniform(0.9, 1.1)),
        kp_scale=float(rng.uniform(0.9, 1.1)),
        kv_scale=float(rng.uniform(0.9, 1.1)),
        latency_steps=int(rng.integers(0, 2)),
        push_times=np.cumsum(rng.uniform(*PUSH_INTERVAL_S, n_push)),
        push_vel=np.concatenate([
            rng.uniform(-PUSH_LIN, PUSH_LIN, (n_push, 3)),
            rng.uniform(-PUSH_ANG, PUSH_ANG, (n_push, 3))], axis=1),
        init_pos=rng.uniform([-0.05, -0.05, -0.01], [0.05, 0.05, 0.01]),
        init_rpy=rng.uniform([-0.1, -0.1, -0.2], [0.1, 0.1, 0.2]),
        init_vel=rng.uniform(-0.2, 0.2, 6),
        init_joint=rng.uniform(-0.1, 0.1, spec.N_JOINTS),
        noise_seed=int(rng.integers(1 << 30)),
    )


def apply_domain(model, dom: Domain) -> None:
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    model.geom_friction[floor, 0] = dom.friction

    model.body_mass[1:] *= dom.mass_scale
    model.body_inertia[1:] *= dom.mass_scale
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.ANCHOR_BODY)
    model.body_ipos[torso] += dom.com_offset
    model.actuator_gainprm[:spec.N_JOINTS, 0] *= dom.kp_scale
    model.actuator_biasprm[:spec.N_JOINTS, 1] *= dom.kp_scale
    model.actuator_biasprm[:spec.N_JOINTS, 2] *= dom.kv_scale


@dataclass
class Index:
    qpos: np.ndarray
    qvel: np.ndarray
    tracked: np.ndarray
    anchor: int
    root: int
    ee: np.ndarray
    anchor_in_tracked: int
    motion_tracked: np.ndarray
    motion_anchor: int
    motion_root: int


def build_index(model, motion) -> Index:
    jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
           for j in spec.JOINT_NAMES]
    return Index(
        qpos=np.array([model.jnt_qposadr[i] for i in jid]),
        qvel=np.array([model.jnt_dofadr[i] for i in jid]),
        tracked=robot.body_index(model, spec.TRACKED_BODIES),
        anchor=int(robot.body_index(model, [spec.ANCHOR_BODY])[0]),
        root=int(robot.body_index(model, [spec.ROOT_BODY])[0]),
        ee=np.array([spec.TRACKED_BODIES.index(b) for b in spec.END_EFFECTORS]),
        anchor_in_tracked=spec.TRACKED_BODIES.index(spec.ANCHOR_BODY),
        motion_tracked=motion.body_index(list(spec.TRACKED_BODIES)),
        motion_anchor=int(motion.body_index([spec.ANCHOR_BODY])[0]),
        motion_root=int(motion.body_index([spec.ROOT_BODY])[0]),
    )


def observation(model, data, motion, frame, prev_action, idx: Index,
                rng: np.random.Generator | None = None) -> np.ndarray:
    anchor_pos = data.xpos[idx.anchor]
    anchor_quat = data.xquat[idx.anchor]
    ref_anchor_pos = motion.body_pos_w[frame, idx.motion_anchor]
    ref_anchor_quat = motion.body_quat_w[frame, idx.motion_anchor]

    vel = np.empty(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, idx.root, vel, 1)

    rel_quat = R.quat_mul(R.quat_conj(anchor_quat), ref_anchor_quat)
    obs = np.concatenate([
        motion.joint_pos[frame],
        motion.joint_vel[frame],
        R.quat_rotate_inverse(anchor_quat, ref_anchor_pos - anchor_pos),
        R.ori6(rel_quat),
        vel[3:],
        vel[:3],
        data.qpos[idx.qpos] - robot.DEFAULT_JOINT_POS,
        data.qvel[idx.qvel],
        prev_action,
    ])
    if rng is not None:
        for term, scale in OBS_NOISE.items():
            sl = spec.obs_slice(term)
            obs[sl] += rng.uniform(-scale, scale, sl.stop - sl.start)
    return obs


@dataclass
class EpisodeLog:
    n_frames: int
    body_pos_err: list = field(default_factory=list)
    body_ori_err: list = field(default_factory=list)
    anchor_pos_err: list = field(default_factory=list)
    anchor_ori_err: list = field(default_factory=list)
    joint_err: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    terminated_at: int | None = None
    diverged: bool = False

    def summary(self) -> dict:
        steps = len(self.body_pos_err)
        alive = steps / self.n_frames
        mean = lambda v: float(np.mean(v)) if v else float("nan")
        error = np.array(self.body_pos_err)
        credit = metrics.tracking_quality(error)
        multi = metrics.tracking_quality_multi(error)
        return {
            "tracking_score": float(credit.sum() / self.n_frames),
            "tracking_multi": float(multi.sum() / self.n_frames),
            "mpjpe_m": mean(self.body_pos_err),
            "body_ori_err_rad": mean(self.body_ori_err),
            "anchor_pos_err_m": mean(self.anchor_pos_err),
            "anchor_ori_err_rad": mean(self.anchor_ori_err),
            "joint_rmse_rad": mean(self.joint_err),
            "survival": alive,
            "fell": self.terminated_at is not None,
            "survived_s": steps * spec.CONTROL_DT,
            "action_jerk": metrics.action_jerk(np.array(self.actions)),
            "diverged": self.diverged,
        }


class OnnxPolicy:

    def __init__(self, submission_dir: str):
        import onnxruntime

        meta = export.load_meta(submission_dir)
        self.history = int(meta["obs_history_length"])
        if not 1 <= self.history <= spec.MAX_HISTORY:
            raise ValueError(f"obs_history_length {self.history} out of range")
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            os.path.join(submission_dir, spec.POLICY_FILE), options,
            providers=["CPUExecutionProvider"])

    def reset(self, seed):
        pass

    def act(self, obs):
        out = self.session.run([spec.ACTION_OUTPUT],
                               {spec.OBS_INPUT: obs[None].astype(np.float32)})
        return out[0][0]


class NullPolicy:

    def reset(self, seed):
        pass

    def act(self, obs):
        return np.zeros(spec.N_ACTIONS)


def _init_state(model, data, motion, idx, dom: Domain | None):
    root = idx.motion_root
    data.qpos[:3] = motion.body_pos_w[0, root]
    data.qpos[3:7] = motion.body_quat_w[0, root]
    data.qpos[idx.qpos] = motion.joint_pos[0]
    data.qvel[:3] = motion.body_lin_vel_w[0, root]

    data.qvel[3:6] = R.quat_rotate_inverse(motion.body_quat_w[0, root],
                                           motion.body_ang_vel_w[0, root])
    data.qvel[idx.qvel] = motion.joint_vel[0]
    if dom is not None:
        data.qpos[:3] += dom.init_pos
        delta = R.quat_mul(_rpy_to_quat(dom.init_rpy), data.qpos[3:7].copy())
        data.qpos[3:7] = delta / np.linalg.norm(delta)
        data.qpos[idx.qpos] += dom.init_joint
        data.qvel[:6] += dom.init_vel
    mujoco.mj_forward(model, data)


def _rpy_to_quat(rpy):
    q = np.empty(4)
    mujoco.mju_euler2Quat(q, rpy, "xyz")
    return q


def _terminated(ref_pos, robot_pos, ref_anchor_quat, robot_anchor_quat, idx) -> bool:
    a = idx.anchor_in_tracked
    fell = abs(ref_pos[a, 2] - robot_pos[a, 2]) > spec.TERM_ANCHOR_Z
    tipped = abs(metrics.gravity_tilt(ref_anchor_quat)
                 - metrics.gravity_tilt(robot_anchor_quat)) > spec.TERM_ANCHOR_TILT
    limb = np.any(np.abs(ref_pos[idx.ee, 2] - robot_pos[idx.ee, 2]) > spec.TERM_EE_Z)
    return bool(fell or tipped or limb)


def run_episode(model, motion, policy, seed: int, video=None,
                randomize: bool = True, history: int = 1) -> EpisodeLog:
    rng = np.random.default_rng(seed)
    dom = sample_domain(rng, motion.duration) if randomize else None
    if dom is not None:
        apply_domain(model, dom)
    model.opt.timestep = EVAL_PHYSICS_DT

    idx = build_index(model, motion)
    data = mujoco.MjData(model)
    _init_state(model, data, motion, idx, dom)

    noise_rng = np.random.default_rng(dom.noise_seed) if dom is not None else None
    offset = robot.DEFAULT_JOINT_POS + (dom.joint_offset if dom is not None else 0.0)
    lag = dom.latency_steps if dom is not None else 0

    policy.reset(seed)
    log = EpisodeLog(n_frames=motion.n_frames)
    prev_action = np.zeros(spec.N_ACTIONS)
    stack = [np.zeros(spec.OBS_DIM)] * (history - 1)
    pending = [robot.action_to_target(prev_action)] * (lag + 1)
    recorder = Recorder.open(model, video)
    next_push = 0

    for frame in range(motion.n_frames):
        obs = observation(model, data, motion, frame, prev_action, idx, noise_rng)
        stack.append(obs)
        action = np.asarray(policy.act(np.concatenate(stack[-history:])), dtype=float)
        if action.shape != (spec.N_ACTIONS,) or not np.isfinite(action).all():
            log.diverged = True
            break
        prev_action = action

        pending.append(np.clip(offset + action * robot.ACTION_SCALE,
                               model.actuator_ctrlrange[:, 0],
                               model.actuator_ctrlrange[:, 1]))
        data.ctrl[:] = pending.pop(0)

        t = frame * spec.CONTROL_DT
        if dom is not None and next_push < len(dom.push_times) and t >= dom.push_times[next_push]:
            data.qvel[:6] += dom.push_vel[next_push]
            next_push += 1

        for _ in range(EVAL_DECIMATION):
            mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            log.diverged = True
            break

        ref_pos, ref_quat = metrics.reanchor(
            motion.body_pos_w[frame, idx.motion_tracked],
            motion.body_quat_w[frame, idx.motion_tracked],
            motion.body_pos_w[frame, idx.motion_anchor],
            motion.body_quat_w[frame, idx.motion_anchor],
            data.xpos[idx.anchor], data.xquat[idx.anchor])
        robot_pos, robot_quat = data.xpos[idx.tracked], data.xquat[idx.tracked]

        log.body_pos_err.append(float(metrics.body_position_error(ref_pos, robot_pos)))
        log.body_ori_err.append(float(metrics.body_orientation_error(ref_quat, robot_quat)))
        log.anchor_pos_err.append(float(metrics.anchor_position_error(
            motion.body_pos_w[frame, idx.motion_anchor], data.xpos[idx.anchor])))
        log.anchor_ori_err.append(float(metrics.anchor_orientation_error(
            motion.body_quat_w[frame, idx.motion_anchor], data.xquat[idx.anchor])))
        log.joint_err.append(float(metrics.joint_error(
            motion.joint_pos[frame], data.qpos[idx.qpos])))
        log.actions.append(action)

        if recorder:
            recorder.capture(data, data.xpos[idx.root])

        if _terminated(ref_pos, robot_pos,
                       motion.body_quat_w[frame, idx.motion_anchor],
                       data.xquat[idx.anchor], idx):
            log.terminated_at = frame
            break

    if recorder:
        recorder.close()
    return log


def replay_metrics(model, motion) -> dict:
    idx = build_index(model, motion)
    data = mujoco.MjData(model)
    log = EpisodeLog(n_frames=motion.n_frames)
    for frame in range(motion.n_frames):
        data.qpos[:3] = motion.body_pos_w[frame, idx.motion_root]
        data.qpos[3:7] = motion.body_quat_w[frame, idx.motion_root]
        data.qpos[idx.qpos] = motion.joint_pos[frame]
        mujoco.mj_forward(model, data)
        ref_pos, ref_quat = metrics.reanchor(
            motion.body_pos_w[frame, idx.motion_tracked],
            motion.body_quat_w[frame, idx.motion_tracked],
            motion.body_pos_w[frame, idx.motion_anchor],
            motion.body_quat_w[frame, idx.motion_anchor],
            data.xpos[idx.anchor], data.xquat[idx.anchor])
        log.body_pos_err.append(float(metrics.body_position_error(
            ref_pos, data.xpos[idx.tracked])))
        log.body_ori_err.append(float(metrics.body_orientation_error(
            ref_quat, data.xquat[idx.tracked])))
        log.anchor_pos_err.append(float(metrics.anchor_position_error(
            motion.body_pos_w[frame, idx.motion_anchor], data.xpos[idx.anchor])))
        log.anchor_ori_err.append(float(metrics.anchor_orientation_error(
            motion.body_quat_w[frame, idx.motion_anchor], data.xquat[idx.anchor])))
        log.joint_err.append(float(metrics.joint_error(
            motion.joint_pos[frame], data.qpos[idx.qpos])))
    return log.summary()


class Recorder:
    CAMERA = dict(distance=3.0, azimuth=135.0, elevation=-15.0)

    def __init__(self, model, video):
        self.stream = _Recorder(video, fps=VIDEO_FPS, every=VIDEO_EVERY)
        self.camera = MujocoCamera(model, None, size=VIDEO_SIZE)

    @classmethod
    def open(cls, model, video):
        if not video:
            return None
        try:
            return cls(model, video)
        except Exception as exc:
            if isinstance(video, VideoWriter):
                video.skipped = f"recorder: {exc}"
            return None

    def capture(self, data, lookat) -> None:
        def render():
            self.camera.look_at(lookat, **self.CAMERA)
            return self.camera.render(data)
        self.stream.capture(render)

    def close(self) -> None:
        self.camera.close()
        self.stream.close()
