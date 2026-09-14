"""Reference motion: LAFAN1 CSV -> the .npz the tracking MDP consumes.

BeyondMimic builds this file by replaying the clip inside Isaac and reading the
articulation back. MuJoCo forward kinematics yields the same quantities without
a second simulator, so this is the whole of task04's preprocessing. Resampling
and differentiation follow the reference semantics; only the kinematics backend
differs.

Offline by construction: a clip is a few hundred frames and is converted once at
build time, so the per-frame mj_forward loop here is not a hot path.
"""
from __future__ import annotations

import numpy as np

import mujoco

from . import robot, rotations as R


_QUAT_XYZW_TO_WXYZ = [6, 3, 4, 5]
N_JOINTS = 29
FIELDS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
          "body_lin_vel_w", "body_ang_vel_w")


def load_csv(path: str, frames: tuple[int, int] | None = None):
    skiprows, max_rows = 0, None
    if frames is not None:
        start, end = frames
        skiprows, max_rows = start - 1, end - start + 1
    raw = np.loadtxt(path, delimiter=",", skiprows=skiprows, max_rows=max_rows)
    return raw[:, :3], raw[:, _QUAT_XYZW_TO_WXYZ], raw[:, 7:]


def resample(root_pos, root_quat, joint_pos, input_fps: int, fps: int):
    n = len(root_pos)
    duration = (n - 1) / input_fps
    phase = np.arange(0.0, duration, 1.0 / fps) / duration * (n - 1)
    i0 = np.floor(phase).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    b = (phase - i0)[:, None]
    return (
        root_pos[i0] * (1 - b) + root_pos[i1] * b,
        R.slerp(root_quat[i0], root_quat[i1], b),
        joint_pos[i0] * (1 - b) + joint_pos[i1] * b,
    )


def differentiate(root_pos, root_quat, joint_pos, fps: int):
    dt = 1.0 / fps
    lin_vel = np.gradient(root_pos, dt, axis=0)
    joint_vel = np.gradient(joint_pos, dt, axis=0)

    rel = R.quat_mul(root_quat[2:], R.quat_conj(root_quat[:-2]))
    ang_vel = R.axis_angle(rel) / (2.0 * dt)
    ang_vel = np.concatenate([ang_vel[:1], ang_vel, ang_vel[-1:]])
    return lin_vel, ang_vel, joint_vel


def body_kinematics(model, root_pos, root_quat, lin_vel, ang_vel, joint_pos, joint_vel):
    data = mujoco.MjData(model)
    n_bodies = model.nbody - 1
    T = len(root_pos)
    pos = np.empty((T, n_bodies, 3))
    quat = np.empty((T, n_bodies, 4))
    lin = np.empty((T, n_bodies, 3))
    ang = np.empty((T, n_bodies, 3))
    vel6 = np.empty(6)

    for t in range(T):
        data.qpos[:3] = root_pos[t]
        data.qpos[3:7] = root_quat[t]
        data.qpos[7:] = joint_pos[t]
        data.qvel[:3] = lin_vel[t]

        data.qvel[3:6] = R.quat_rotate_inverse(root_quat[t], ang_vel[t])
        data.qvel[6:] = joint_vel[t]
        mujoco.mj_forward(model, data)
        pos[t] = data.xpos[1:]
        quat[t] = data.xquat[1:]
        for i in range(1, model.nbody):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, i, vel6, 0)
            ang[t, i - 1] = vel6[:3]
            lin[t, i - 1] = vel6[3:]
    return pos, quat, lin, ang


def names(model, objtype, count, offset=0) -> list[str]:
    return [mujoco.mj_id2name(model, objtype, i + offset) for i in range(count)]


def convert(csv_path: str, model, frames=None, input_fps: int = 30, fps: int = 50) -> dict:
    root_pos, root_quat, joint_pos = load_csv(csv_path, frames)
    root_pos, root_quat, joint_pos = resample(root_pos, root_quat, joint_pos, input_fps, fps)
    lin_vel, ang_vel, joint_vel = differentiate(root_pos, root_quat, joint_pos, fps)
    body_pos, body_quat, body_lin, body_ang = body_kinematics(
        model, root_pos, root_quat, lin_vel, ang_vel, joint_pos, joint_vel)
    return {
        "fps": np.int64(fps),
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": body_lin.astype(np.float32),
        "body_ang_vel_w": body_ang.astype(np.float32),
        "body_names": np.array(names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody - 1, 1)),
        "joint_names": np.array(names(model, mujoco.mjtObj.mjOBJ_JOINT, N_JOINTS, 1)),
    }


class Motion:

    def __init__(self, data: dict):
        self.fps = int(data["fps"])
        self.body_names = [str(s) for s in data["body_names"]]
        self.joint_names = [str(s) for s in data["joint_names"]]
        for f in FIELDS:
            setattr(self, f, np.asarray(data[f]))
        self.n_frames = len(self.joint_pos)

    @classmethod
    def load(cls, path: str) -> "Motion":
        with np.load(path) as f:
            return cls(dict(f))

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps

    def body_index(self, wanted: list[str]) -> np.ndarray:
        return np.array([self.body_names.index(n) for n in wanted])


def save(path: str, data: dict) -> None:
    np.savez_compressed(path, **data)


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="LAFAN1 CSV -> tracking reference .npz")
    p.add_argument("--robot-dir", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frames", help="1-based inclusive, e.g. 122:722")
    p.add_argument("--input-fps", type=int, default=30)
    p.add_argument("--fps", type=int, default=50)
    a = p.parse_args(argv)

    frames = tuple(int(x) for x in a.frames.split(":")) if a.frames else None
    data = convert(a.csv, robot.build(a.robot_dir), frames, a.input_fps, a.fps)
    save(a.out, data)
    m = Motion(data)
    print(f"{a.out}: {m.n_frames} frames @ {m.fps} fps ({m.duration:.2f} s), "
          f"{len(m.body_names)} bodies")

if __name__ == "__main__":
    main()
