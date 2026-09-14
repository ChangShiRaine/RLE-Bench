"""RGB-D sensing for task07: two cameras with structured-light depth noise.

Everything is computed in the CAMERA frame so the same model serves the
moving wrist camera: backprojection, normals, and incidence angles never
need the world pose. Depth noise is incidence-dependent with grazing
dropout; invalid pixels are NaN. Same (seed, camera, frame_idx) =>
identical realization.
"""
from __future__ import annotations

import mujoco
import numpy as np

from . import spec

CAMERAS = {
    "overhead": (spec.OVERHEAD_W, spec.OVERHEAD_H, spec.OVERHEAD_FOVY_DEG),
    "wrist": (spec.WRIST_W, spec.WRIST_H, spec.WRIST_FOVY_DEG),
}
_CAM_STREAM = {"overhead": 0, "wrist": 1}


class CameraRig:
    """Offscreen renderers for both sensor cameras of one compiled model."""

    def __init__(self, model):
        self.model = model
        self._rgb = {}
        self._depth = {}
        for name, (w, h, _) in CAMERAS.items():
            self._rgb[name] = mujoco.Renderer(model, h, w)
            self._depth[name] = mujoco.Renderer(model, h, w)
            self._depth[name].enable_depth_rendering()

    def rgb(self, data, camera: str) -> np.ndarray:
        r = self._rgb[camera]
        r.update_scene(data, camera="arm_wrist" if camera == "wrist"
                       else camera)
        return r.render().copy()

    def depth(self, data, camera: str) -> np.ndarray:
        r = self._depth[camera]
        r.update_scene(data, camera="arm_wrist" if camera == "wrist"
                       else camera)
        return r.render().copy().astype(np.float32)

    def close(self) -> None:
        for r in (*self._rgb.values(), *self._depth.values()):
            r.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# Camera-frame geometry
# ---------------------------------------------------------------------------
def backproject_cam(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Organized point cloud (H, W, 3) in the CAMERA frame from z-depth.

    Camera convention: +x right, +y up in image, looks along -z.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    with np.errstate(invalid="ignore"):
        xc = (u - cx) / fx * depth
        yc = -(v - cy) / fy * depth
        zc = -depth
    return np.stack([xc, yc, zc], axis=-1)


def incidence_cos(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """|cos| between each pixel's view ray and the surface normal."""
    pts = backproject_cam(depth, K)
    dx = np.gradient(pts, axis=1)
    dy = np.gradient(pts, axis=0)
    n = np.cross(dx, dy)
    n /= np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9, None)
    rays = pts / np.clip(np.linalg.norm(pts, axis=-1, keepdims=True),
                         1e-9, None)
    return np.clip(np.abs(np.sum(n * rays, axis=-1)), 1e-3, 1.0)


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------
def depth_sigma(cos_i: np.ndarray) -> np.ndarray:
    tan2 = 1.0 / cos_i ** 2 - 1.0
    return spec.DEPTH_SIGMA0 + np.clip(
        spec.DEPTH_SIGMA_K * tan2, 0.0,
        spec.DEPTH_SIGMA_CAP - spec.DEPTH_SIGMA0)


def apply_depth_noise(depth: np.ndarray, K: np.ndarray, seed: int,
                      camera: str, frame_idx: int) -> np.ndarray:
    rng = np.random.default_rng(
        [seed, spec.STREAM_SENSOR, _CAM_STREAM[camera], frame_idx])
    cos_i = incidence_cos(depth, K)
    noisy = depth + rng.normal(0.0, 1.0, depth.shape) * depth_sigma(cos_i)
    drop = (cos_i < spec.DEPTH_DROP_COS) \
        & (rng.uniform(size=depth.shape) < spec.DEPTH_DROP_P)
    noisy = noisy.astype(np.float32)
    noisy[drop] = np.nan
    noisy[depth > spec.DEPTH_MAX_RANGE] = np.nan
    return noisy


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
def camera_pose(model, data, camera: str) -> np.ndarray:
    """T_world_cam of a (possibly moving) camera at the current state."""
    name = "arm_wrist" if camera == "wrist" else camera
    cid = model.camera(name).id
    T = np.eye(4)
    T[:3, :3] = data.cam_xmat[cid].reshape(3, 3)
    T[:3, 3] = data.cam_xpos[cid]
    return T


def frame_obs(rig: CameraRig, model, data, seed: int, frame_idx: int) -> dict:
    """Both cameras' RGB-D for one observation frame."""
    out = {}
    for cam, (w, h, fovy) in CAMERAS.items():
        K = spec.intrinsics(w, h, fovy)
        clean = rig.depth(data, cam)
        out[f"{cam}_rgb"] = rig.rgb(data, cam)
        out[f"{cam}_depth"] = apply_depth_noise(clean, K, seed, cam,
                                                frame_idx)
    out["wrist_T_world_cam"] = camera_pose(model, data, "wrist")
    return out
