"""Constants for the task06 family: scene geometry, the camera and
depth-sensor models, seeds and the estimator interface contracts.

This module stays in the root-owned and verifier trees; the agent workspace
gets the camera, sensor and contract subset as public_spec.py.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------
# Resolved at import: the harness containers set RLEBENCH_ASSETS; a source
# checkout falls back to the package's own assets/ (franka is staged in from
# assets/robots at build time; dev boxes read it from there directly).
ASSETS_DIR = Path(os.environ.get(
    "RLEBENCH_ASSETS",
    Path(__file__).resolve().parent / "assets"))
TSHAPE_STL = ASSETS_DIR / "tshape" / "tshape.stl"
C_BLOCK_STL = ASSETS_DIR / "tshape" / "c_block.stl"
F_SHAPE_STL = ASSETS_DIR / "tshape" / "f_shape.stl"
STICK_STL = ASSETS_DIR / "tshape" / "stick_d405.stl"
if (ASSETS_DIR / "franka_emika_panda").exists():
    PANDA_XML = ASSETS_DIR / "franka_emika_panda" / "panda_nohand.xml"
else:  # source checkout without staged trees: read the meshes from core
    _parents = Path(__file__).resolve().parents
    _repo = _parents[3] if len(_parents) > 3 else _parents[-1]
    PANDA_XML = (_repo / "assets" / "robots"
                 / "franka_emika_panda" / "panda_nohand.xml")

MESH_SCALE = 0.001  # both STLs are authored in millimeters


DEFAULT_BLOCK_SHAPE = "tshape"
BLOCK_SHAPES = ("tshape", "c_block", "f_shape")
BLOCK_MESHES = {
    "tshape": TSHAPE_STL,
    "c_block": C_BLOCK_STL,
    "f_shape": F_SHAPE_STL,
}
# ---------------------------------------------------------------------------
# Table / block / arm geometry (meters, world frame; table top is z = 0)
# ---------------------------------------------------------------------------
TABLE_HALF = (0.45, 0.35)          # tabletop half-extents (x, y)
TABLE_THICKNESS = 0.04
SPAWN_HALF = (0.25, 0.175)         # block spawn region half-extents around origin
BLOCK_HEIGHT = 0.04                # extrusion height of the T block
BLOCK_RGBA = (0.80, 0.13, 0.10, 1.0)
BLOCK_DENSITY = 500.0              # kg/m^3 over the two collision boxes (~0.35 kg)
# T-block collision decomposition in the block frame (the visual mesh's convex
# hull would fill the T's notch, so contact uses two boxes):
BLOCK_BAR_CENTER = (0.0, 0.0, 0.02)
BLOCK_BAR_HALF = (0.100, 0.025, 0.02)
BLOCK_STEM_CENTER = (0.0, -0.100, 0.02)
BLOCK_STEM_HALF = (0.025, 0.075, 0.02)
CONTACT_FRICTION = (0.35, 0.005, 0.0001)
# Concave meshes need primitive collision decompositions: MuJoCo otherwise
# collision-convexifies each STL and fills its notches. Each entry is a tuple
# of (box center, box half-size), in the mesh/block frame and in meters.
BLOCK_COLLISION_BOXES = {
    "tshape": (
        (BLOCK_BAR_CENTER, BLOCK_BAR_HALF),
        (BLOCK_STEM_CENTER, BLOCK_STEM_HALF),
    ),
    "c_block": (
        ((0.0, 0.025, 0.02), (0.100, 0.025, 0.02)),
        ((-0.075, -0.075, 0.02), (0.025, 0.075, 0.02)),
        ((0.075, -0.075, 0.02), (0.025, 0.075, 0.02)),
    ),
    "f_shape": (
        ((0.0, 0.025, 0.02), (0.100, 0.025, 0.02)),
        ((-0.075, -0.075, 0.02), (0.025, 0.075, 0.02)),
        ((0.025, -0.075, 0.02), (0.025, 0.075, 0.02)),
    ),
}

ARM_BASE_POS = (-0.44, 0.12, 0.0)  # Panda base, bolted at the table edge
ARM_BASE_YAW = -0.35               # radians about +z
ARM_PARKED = (0.0, -1.1, 0.0, -2.4, 0.0, 1.4, 0.785)
STICK_TIP_OFFSET = 0.1916          # tip below the flange along the stick axis
STICK_SHAFT_RADIUS = 0.0125
PUSH_TIP_HEIGHT = 0.025            # stroke height of the stick tip above table

# ---------------------------------------------------------------------------
# Camera (fixed tripod mount; K and extrinsics are published — pose estimation,
# not camera calibration, is the task)
# ---------------------------------------------------------------------------
CAM_POS = (0.50, -0.25, 0.72)      # 51.4 deg elevation, 0.90 m stand-off
CAM_TARGET = (0.0, 0.0, 0.02)
IMG_W, IMG_H = 640, 480
FOVY_DEG = 42.0
FRAME_HZ = 10.0                    # observation rate in episodes
DEPTH_MAX_RANGE = 3.0              # meters; beyond this depth is invalid

# ---------------------------------------------------------------------------
# Depth sensor model (applied to the rendered depth; RealSense-class behavior)
#   sigma(theta) = SIGMA0 + min(SIGMA_K * tan^2(theta), SIGMA_CAP - SIGMA0)
#   dropout where cos(theta) < DROP_COS with probability DROP_P
# theta = incidence angle between the viewing ray and the surface normal.
# ---------------------------------------------------------------------------
DEPTH_SIGMA0 = 0.0015
DEPTH_SIGMA_K = 0.004
DEPTH_SIGMA_CAP = 0.030
DEPTH_DROP_COS = 0.18
DEPTH_DROP_P = 0.75

# ---------------------------------------------------------------------------
# Seeds and RNG streams. Design seeds are public; evaluation uses different,
# unpublished seeds drawn from the same distributions.
# ---------------------------------------------------------------------------
DESIGN_SEEDS = (11, 23, 37)
STREAM_SCENE = 0     # nuisance draw + block spawn
STREAM_SENSOR = 1    # depth-noise realization (per frame)
STREAM_STROKES = 2   # push stroke endpoints

STREAM_SHAPE = 3     # hidden block-shape draw (all task06 variants)
# ---------------------------------------------------------------------------
# Estimator interface contracts
# ---------------------------------------------------------------------------
# Code tasks (rgb-only / rgb-depth / method-agnostic): ship estimator.py exposing
#     make_estimator() -> obj with
#         reset() -> None                    (called once per episode)
#         update(**obs) -> (x, y, theta, shape_id)  (0=T, 1=C, 2=F)
# obs keys per task:
#   rgb-only: rgb (H,W,3) uint8, K (3,3), T_cam_table (4,4), t (float seconds)
#   rgb-depth: rgb, depth, K, T_cam_table, t
#   method-agnostic: rgb, depth, K, T_cam_table, t
# Return pose of the block in the table frame: x, y in meters, theta in
# radians (the block frame of tshape.stl; theta measured about +z).
#
# Learned task (rgb-depth-model-training): ship model.pt, a TorchScript module saved on CPU.
MODEL_MAX_PARAMETERS = 20_000_000


# Pose sanity bounds (predictions outside are treated as invalid)
POSE_XY_BOUND = 0.45       # |x|, |y| must stay on the table
POSE_LIMIT_NOTE = "theta is wrapped to (-pi, pi]"


def intrinsics() -> np.ndarray:
    """Pinhole K for the sensor camera (fx = fy from vertical FOV)."""
    fy = 0.5 * IMG_H / np.tan(np.deg2rad(FOVY_DEG) / 2)
    return np.array([[fy, 0.0, IMG_W / 2.0],
                     [0.0, fy, IMG_H / 2.0],
                     [0.0, 0.0, 1.0]])


def camera_rotation() -> np.ndarray:
    """World-from-camera rotation: columns are the camera's x/y/z axes.

    Camera convention: looks along its local -z, +x right, +y up in image.
    """
    pos = np.asarray(CAM_POS, dtype=float)
    target = np.asarray(CAM_TARGET, dtype=float)
    f = target - pos
    f /= np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    cup = np.cross(right, f)
    return np.column_stack([right, cup, -f])


def extrinsics() -> np.ndarray:
    """T_cam_table: 4x4 pose of the camera in the table frame."""
    T = np.eye(4)
    T[:3, :3] = camera_rotation()
    T[:3, 3] = CAM_POS
    return T


def wrap_angle(theta: float) -> float:
    """Wrap to (-pi, pi]."""
    return float(np.pi - np.mod(np.pi - theta, 2 * np.pi))
