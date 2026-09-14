"""Agent-visible constants for Task 06's method-agnostic blind three-shape service.

Camera, sensor, seed, and estimator contracts are public. Mesh paths,
footprints, collision primitives, and the per-seed shape draw are private.
"""
from __future__ import annotations

import numpy as np

TABLE_HALF = (0.45, 0.35)
SPAWN_HALF = (0.25, 0.175)
BLOCK_HEIGHT = 0.04
NUM_BLOCK_SHAPES = 3
SHAPE_LABELS = ("T", "C", "F")  # output IDs 0, 1, 2

CAM_POS = (0.50, -0.25, 0.72)
CAM_TARGET = (0.0, 0.0, 0.02)
IMG_W, IMG_H = 640, 480
FOVY_DEG = 42.0
FRAME_HZ = 10.0
DEPTH_MAX_RANGE = 3.0

DEPTH_SIGMA0 = 0.0015
DEPTH_SIGMA_K = 0.004
DEPTH_SIGMA_CAP = 0.030
DEPTH_DROP_COS = 0.18
DEPTH_DROP_P = 0.75

DESIGN_SEEDS = (11, 23, 37)
POSE_XY_BOUND = 0.45
POSE_LIMIT_NOTE = "theta is wrapped to (-pi, pi]"
MODEL_MAX_PARAMETERS = 20_000_000


def intrinsics() -> np.ndarray:
    """Pinhole K for the sensor camera."""
    fy = 0.5 * IMG_H / np.tan(np.deg2rad(FOVY_DEG) / 2)
    return np.array([[fy, 0.0, IMG_W / 2.0],
                     [0.0, fy, IMG_H / 2.0],
                     [0.0, 0.0, 1.0]])


def camera_rotation() -> np.ndarray:
    """World-from-camera rotation; columns are camera x/y/z axes."""
    pos = np.asarray(CAM_POS, dtype=float)
    target = np.asarray(CAM_TARGET, dtype=float)
    forward = target - pos
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    return np.column_stack([right, camera_up, -forward])


def extrinsics() -> np.ndarray:
    """T_cam_table: camera pose in the table frame."""
    transform = np.eye(4)
    transform[:3, :3] = camera_rotation()
    transform[:3, 3] = CAM_POS
    return transform


def wrap_angle(theta: float) -> float:
    """Wrap to (-pi, pi]."""
    return float(np.pi - np.mod(np.pi - theta, 2 * np.pi))
