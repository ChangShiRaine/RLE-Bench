"""RGB-D sensing for task11: a fixed top-view camera over the cell and a
wrist camera on the tool, both with structured-light depth noise
(incidence-dependent sigma, grazing dropout as NaN). Everything is computed
in the camera frame, so the same model serves the moving wrist camera. Same
(seed, camera, frame_idx) => identical realization.
"""
from __future__ import annotations

import mujoco
import numpy as np

from . import spec

_CAM_STREAM = {"top": 0, "wrist": 1}
_MJ_NAME = {"top": "top_cam", "wrist": "arm_wrist_cam"}


class CameraRig:
    def __init__(self, model):
        self._rgb, self._depth = {}, {}
        for cam, (w, h, _) in spec.CAMERAS.items():
            self._rgb[cam] = mujoco.Renderer(model, h, w)
            self._depth[cam] = mujoco.Renderer(model, h, w)
            self._depth[cam].enable_depth_rendering()

    def rgb(self, data, camera: str) -> np.ndarray:
        self._rgb[camera].update_scene(data, camera=_MJ_NAME[camera])
        return self._rgb[camera].render().copy()

    def depth(self, data, camera: str) -> np.ndarray:
        self._depth[camera].update_scene(data, camera=_MJ_NAME[camera])
        return self._depth[camera].render().astype(np.float32)

    def close(self) -> None:
        for r in (*self._rgb.values(), *self._depth.values()):
            r.close()


def backproject_cam(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Organized (H, W, 3) cloud in the camera frame (+x right, +y up,
    looking along -z) from z-depth."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=depth.dtype),
                       np.arange(h, dtype=depth.dtype))
    with np.errstate(invalid="ignore"):
        return np.stack([(u - K[0, 2]) / K[0, 0] * depth,
                         -(v - K[1, 2]) / K[1, 1] * depth, -depth], axis=-1)


def incidence_cos(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    pts = backproject_cam(depth.astype(np.float32), K.astype(np.float32))
    n = np.cross(np.gradient(pts, axis=1), np.gradient(pts, axis=0))
    n /= np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9, None)
    rays = pts / np.clip(np.linalg.norm(pts, axis=-1, keepdims=True), 1e-9,
                         None)
    return np.clip(np.abs(np.sum(n * rays, axis=-1)), 1e-3, 1.0)


def apply_depth_noise(depth, K, seed: int, camera: str, frame_idx: int):
    rng = np.random.default_rng(
        [seed, spec.STREAM_SENSOR, _CAM_STREAM[camera], frame_idx])
    cos_i = incidence_cos(depth, K)
    sigma = spec.DEPTH_SIGMA0 + np.clip(
        spec.DEPTH_SIGMA_K * (1.0 / cos_i ** 2 - 1.0), 0.0,
        spec.DEPTH_SIGMA_CAP - spec.DEPTH_SIGMA0)
    noisy = (depth + rng.standard_normal(depth.shape, np.float32)
             * sigma).astype(np.float32)
    drop = (cos_i < spec.DEPTH_DROP_COS) \
        & (rng.uniform(size=depth.shape) < spec.DEPTH_DROP_P)
    noisy[drop | (depth > spec.DEPTH_MAX_RANGE)] = np.nan
    return noisy


def camera_pose(model, data, camera: str) -> np.ndarray:
    """T_world_cam of a camera at the current state."""
    cid = model.camera(_MJ_NAME[camera]).id
    T = np.eye(4)
    T[:3, :3] = data.cam_xmat[cid].reshape(3, 3)
    T[:3, 3] = data.cam_xpos[cid]
    return T


def frame_obs(rig: CameraRig, model, data, seed: int, frame_idx: int) -> dict:
    out = {}
    for cam, (w, h, fovy) in spec.CAMERAS.items():
        K = spec.intrinsics(w, h, fovy)
        out[f"{cam}_rgb"] = rig.rgb(data, cam)
        out[f"{cam}_depth"] = apply_depth_noise(rig.depth(data, cam), K,
                                                seed, cam, frame_idx)
    out["wrist_T_world_cam"] = camera_pose(model, data, "wrist")
    return out
