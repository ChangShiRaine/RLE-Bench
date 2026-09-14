"""Sensor model for the task06 family: RGB-D rendering + depth noise.

The camera renders through MuJoCo's offscreen renderer (set MUJOCO_GL to a
headless backend; the evaluation containers pin ``osmesa``). Depth noise is
incidence-angle dependent — heavy on surfaces seen at grazing angles (the
block's vertical side walls, far table regions) — with dropout beyond
``spec.DEPTH_DROP_COS``, matching structured-light camera behavior.
"""
from __future__ import annotations

import mujoco
import numpy as np

from . import spec


class CameraRig:
    """Reusable offscreen renderer for one compiled model."""

    def __init__(self, model):
        self.model = model
        self._rgb = mujoco.Renderer(model, spec.IMG_H, spec.IMG_W)
        self._depth = mujoco.Renderer(model, spec.IMG_H, spec.IMG_W)
        self._depth.enable_depth_rendering()

    def rgb(self, data, camera: str = "sensor") -> np.ndarray:
        self._rgb.update_scene(data, camera=camera)
        return self._rgb.render().copy()

    def depth(self, data, camera: str = "sensor") -> np.ndarray:
        self._depth.update_scene(data, camera=camera)
        return self._depth.render().copy().astype(np.float32)

    def close(self) -> None:
        self._rgb.close()
        self._depth.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def backproject(depth: np.ndarray) -> np.ndarray:
    """Organized point cloud (H, W, 3) in the TABLE frame from z-depth."""
    K = spec.intrinsics()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    with np.errstate(invalid="ignore"):
        xc = (u - cx) / fx * depth
        yc = -(v - cy) / fy * depth
        zc = -depth
        pts_cam = np.stack([xc, yc, zc], axis=-1)
        R = spec.camera_rotation()
        return np.asarray(spec.CAM_POS) + pts_cam @ R.T


def incidence_cos(depth: np.ndarray) -> np.ndarray:
    """|cos| of the angle between each pixel's view ray and surface normal.

    Normals come from the organized cloud's image-space gradients — the same
    estimate a real depth camera's noise behavior is driven by.
    """
    pts = backproject(depth)
    dx = np.gradient(pts, axis=1)
    dy = np.gradient(pts, axis=0)
    n = np.cross(dx, dy)
    n /= np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9, None)
    rays = pts - np.asarray(spec.CAM_POS)
    rays /= np.clip(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-9, None)
    return np.clip(np.abs(np.sum(n * rays, axis=-1)), 1e-3, 1.0)


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------
def depth_sigma(cos_i: np.ndarray) -> np.ndarray:
    """Per-pixel noise sigma (meters) from incidence."""
    tan2 = 1.0 / cos_i ** 2 - 1.0
    return spec.DEPTH_SIGMA0 + np.clip(
        spec.DEPTH_SIGMA_K * tan2, 0.0,
        spec.DEPTH_SIGMA_CAP - spec.DEPTH_SIGMA0)


def apply_depth_noise(depth: np.ndarray, seed: int,
                      frame_idx: int = 0) -> np.ndarray:
    """Seeded sensor-model depth: incidence noise + grazing dropout.

    Same (seed, frame_idx) => identical noise realization. Invalid pixels
    (dropout, beyond max range) are NaN.
    """
    rng = np.random.default_rng([seed, spec.STREAM_SENSOR, frame_idx])
    cos_i = incidence_cos(depth)
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
def observation(rig: CameraRig, data, seed: int, frame_idx: int = 0,
                t: float = 0.0) -> dict:
    """One RGB-D observation with published camera parameters."""
    clean = rig.depth(data)
    return {
        "rgb": rig.rgb(data),
        "depth": apply_depth_noise(clean, seed, frame_idx),
        "K": spec.intrinsics(),
        "T_cam_table": spec.extrinsics(),
        "t": float(t),
    }
