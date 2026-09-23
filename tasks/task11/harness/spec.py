"""Public constants for task11 (online bin packing from a moving conveyor).

Everything in this module is agent-visible. It is the single source of truth
for the cell geometry, the box catalogue, the conveyor, the suction tool, the
cameras, the control rates and the policy interface contract. The evaluation
drives the same numbers on unpublished seeds.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------
ASSETS_DIR = Path(os.environ.get(
    "RLEBENCH_ASSETS", Path(__file__).resolve().parents[1] / "assets"))
if not (ASSETS_DIR / "franka_emika_panda").exists():
    # source checkout: the repository's vendored robot descriptions
    ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets" / "robots"
PANDA_XML = ASSETS_DIR / "franka_emika_panda" / "panda_nohand.xml"

# ---------------------------------------------------------------------------
# Timing (world z = 0 is the cell floor; SI units, radians)
# ---------------------------------------------------------------------------
TIMESTEP = 0.002                  # 500 Hz physics, implicitfast
CTRL_HZ = 20.0                    # policy act() rate
CTRL_DECIMATION = int(round(1.0 / (CTRL_HZ * TIMESTEP)))   # 25 substeps
FRAME_HZ = 5.0                    # camera observation rate
FRAME_EVERY_TICKS = int(round(CTRL_HZ / FRAME_HZ))
EPISODE_T = 240.0                 # sim-time budget per episode (seconds)

# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------
PEDESTAL_R = 0.10
PEDESTAL_H = 0.04                 # arm base height
ARM_HOME = (0.60, -0.30, 0.0, -2.20, 0.0, 1.90, 1.35)
N_CTRL = 8                        # 7 joint position targets + suction command

# ---------------------------------------------------------------------------
# End-effector: single vacuum suction cup on a rigid extension. While
# ctrl[7] >= SUCTION_ON, a box seals to the cup when the cup face touches one
# box face nearly square-on (within SEAL_MAX_TILT_DEG) with the whole cup on
# that face (cup centre at least SEAL_EDGE_M inside every face edge). A sealed
# box is held rigidly at its current relative pose; only one box is held at
# a time. Dropping the command below SUCTION_ON releases it instantly.
# Only boxes riding the running belt can seal: bottom within BELT_RIDE_DZ of
# the belt surface, inside the belt's width, and moving with the belt (speed
# error below BELT_RIDE_DV). The belt never stops, so every pick is made on a
# moving target; boxes set down anywhere else cannot be picked again.
# ---------------------------------------------------------------------------
CUP_R = 0.020                     # cup face radius
CUP_L = 0.020
SHAFT_R = 0.012
SHAFT_L = 0.130
COLLAR_R = 0.030
COLLAR_L = 0.012
TOOL_L = COLLAR_L + SHAFT_L + CUP_L   # flange to cup face: 0.162 m
SUCTION_CTRL_RANGE = (0.0, 255.0)
SUCTION_ON = 128.0
SEAL_TOUCH = 0.0015               # contact distance that counts as touching
SEAL_MAX_TILT_DEG = 15.0
SEAL_EDGE_M = 0.75 * CUP_R
SUCTION_MAX_KG = 3.0              # every box type is lighter
BELT_RIDE_DZ = 0.015
BELT_RIDE_DV = 0.06               # m/s

FT_NOISE_FORCE_N = 0.25           # wrist F/T 1-sigma per axis
FT_NOISE_TORQUE_NM = 0.01

# ---------------------------------------------------------------------------
# Bin (open-top tote on a stand, welded)
# ---------------------------------------------------------------------------
STAND_H = 0.068
BIN_WALL = 0.012
BIN_CENTER = (0.42, -0.34)
BIN_INNER = (0.40, 0.30, 0.24)    # interior L x W x H
BIN_FLOOR_Z = STAND_H + BIN_WALL
BIN_RIM_Z = BIN_FLOOR_Z + BIN_INNER[2]
BIN_LO = (BIN_CENTER[0] - BIN_INNER[0] / 2, BIN_CENTER[1] - BIN_INNER[1] / 2,
          BIN_FLOOR_Z)
BIN_HI = (BIN_CENTER[0] + BIN_INNER[0] / 2, BIN_CENTER[1] + BIN_INNER[1] / 2,
          BIN_RIM_Z)
BIN_VOLUME = float(np.prod(BIN_INNER))

# ---------------------------------------------------------------------------
# Conveyor. Boxes enter at SPAWN_X, travel in -x at the episode's belt speed
# and leave the cell past BELT_END_X (the overflow line) if not picked.
# ---------------------------------------------------------------------------
BELT_Y = 0.44
BELT_HALF_W = 0.16
BELT_TOP = 0.20
BELT_X_RANGE = (-0.30, 1.30)      # visible belt extent
SPAWN_X = 1.18
BELT_END_X = -0.22                # a box centre past this has left the cell
BELT_SPEED_RANGE = (0.08, 0.12)   # m/s, drawn per episode, reported in obs
BELT_FRICTION = (1.0, 0.01, 0.001)

# ---------------------------------------------------------------------------
# Box stream. Box types are real Amazon shipping cartons (designation, L, W, H
# in inches; https://incompetech.com/gallimaufry/amazonboxes.html), scaled
# uniformly by BOX_SCALE so the tote holds about twenty of them. Every box
# has the same density, so mass scales with volume (heaviest ~1.4 kg, well
# inside the arm's payload). Each episode queues BOX_COPIES_RANGE copies of
# every type in a seeded shuffled order. Boxes ride flat (H vertical).
# ---------------------------------------------------------------------------
INCH = 0.0254
AMAZON_BOXES = (
    ("A1", 10.0, 7.0, 3.0),
    ("A3", 10.0, 7.0, 5.25),
    ("1A1", 11.5, 8.5, 4.25),
    ("Z1", 12.0, 9.0, 4.0),
    ("A4", 12.0, 8.5, 7.0),
    ("W01", 10.0, 8.0, 6.5),
    ("60", 12.0, 9.0, 5.25),
    ("1A5", 13.5, 11.0, 4.75),
    ("100", 14.0, 11.0, 5.5),
    ("130", 13.0, 10.0, 8.0),
    ("E4", 16.0, 12.0, 4.0),
    ("N3", 16.0, 12.0, 5.0),
    ("Z15", 15.0, 11.5, 8.0),
    ("E6", 16.0, 12.0, 8.0),
    ("1A9", 14.0, 12.5, 9.5),
    ("K3", 18.75, 13.25, 6.25),
)
BOX_SCALE = 0.47
BOX_TYPES = tuple((name, tuple(round(d * INCH * BOX_SCALE, 4)
                               for d in (l, w, h)))
                  for name, l, w, h in AMAZON_BOXES)
BOX_DENSITY = 500.0               # kg/m^3, shared by every box
BOX_COPIES_RANGE = (3, 5)         # inclusive copies of each type per episode
BOX_FRICTION = (0.6, 0.005, 0.0001)
ARRIVAL_GAP_RANGE = (4.0, 7.0)    # s between successive boxes
FIRST_ARRIVAL_T = 0.5
BOX_YAW_JITTER_DEG = 20.0         # about 0 or 90 deg (equally likely)
BOX_LATERAL_JITTER = 0.05         # m about BELT_Y

# Dimensioning scanner at the belt entry: each box is reported once, as it
# enters, with its measured pose, dims and mass (1-sigma noise below).
SCAN_NOISE_DIM = 0.0015           # m
SCAN_NOISE_POS = 0.002            # m
SCAN_NOISE_YAW_DEG = 0.5
SCAN_NOISE_MASS_FRAC = 0.02

# ---------------------------------------------------------------------------
# Packing rules (harness instrumentation)
# ---------------------------------------------------------------------------
FIT_RES = 0.01                    # heightmap cell, m
FIT_CLEARANCE = 0.010             # extra footprint per side in the fit rule
SUPPORT_TOL = 0.01                # cells within this of the base support it
SUPPORT_FRAC = 0.60               # supported footprint fraction required
FULL_STREAK = 5                   # consecutive non-fitting passes = bin full
PACK_TOL = 0.005                  # packed box corners may exceed walls by this
PACK_MAX_TILT_DEG = 15.0          # a packed box must sit square
REST_SPEED = 0.05                 # m/s
FINAL_SETTLE_T = 0.5
FLOOR_Z = 0.03                    # box COM below this = on the cell floor
PARK = (3.0, -3.0, -2.0)          # inactive boxes are parked here, spaced in x
PARK_DX = 0.4

# ---------------------------------------------------------------------------
# Cameras: one fixed top-view RGB-D camera over the whole cell (belt from its
# entry to the overflow line, and the tote) and one RGB-D camera on the tool.
# Intrinsics and extrinsics are published (the cell is calibrated); the wrist
# camera pose is FK of qpos and is also reported per frame.
# ---------------------------------------------------------------------------
TOP_CAM_POS = (0.45, 0.05, 1.80)
TOP_CAM_QUAT = (1.0, 0.0, 0.0, 0.0)   # looks straight down, image up = +y
TOP_W, TOP_H = 640, 480
TOP_FOVY_DEG = 50.0
WRIST_MOUNT_POS = (0.07, 0.0, 0.02)   # in the tool (flange) frame
WRIST_MOUNT_QUAT = (0.0, 1.0, 0.0, 0.0)  # looks along the approach axis
WRIST_W, WRIST_H = 320, 240
WRIST_FOVY_DEG = 58.0
DEPTH_MAX_RANGE = 3.0
DEPTH_SIGMA0 = 0.0015
DEPTH_SIGMA_K = 0.004
DEPTH_SIGMA_CAP = 0.030
DEPTH_DROP_COS = 0.18
DEPTH_DROP_P = 0.75

# ---------------------------------------------------------------------------
# Seeds and RNG streams. Design seeds are public practice streams; the
# evaluation uses unpublished seeds from the same distributions.
# ---------------------------------------------------------------------------
DESIGN_SEEDS = (11, 23, 57, 101, 202)
STREAM_SCENE = 0
STREAM_SENSOR = 1
STREAM_BOXES = 2
STREAM_FT = 3
STREAM_SCAN = 4

# ---------------------------------------------------------------------------
# Policy interface contract
# ---------------------------------------------------------------------------
# Ship policy/policy.py exposing make_policy() -> obj with
#     reset(cell_spec: dict, seed: int) -> None   (once per episode)
#     act(obs: dict) -> array of N_CTRL floats    (at CTRL_HZ)
# obs keys, every tick:
#     qpos (7,), qvel (7,), tau (7,), ft (6,), suction_on (0/1),
#     seal (0/1 vacuum switch: a box is sealed to the cup), t,
#     belt_v (m/s, belt moves along -x),
#     scans (M, 9): one row per box scanned so far, in arrival order:
#         [box_id, t_scan, x, y, yaw, l, w, h, mass]  (measured, noisy)
# and on FRAME_HZ ticks: top_rgb, top_depth, wrist_rgb, wrist_depth
#     ((H, W, 3) uint8 / (H, W) float32 metres, NaN = dropout) and
#     wrist_T_world_cam (4, 4).
POLICY_MAX_BYTES = 1 << 30        # submitted package size limit
POLICY_MAX_FILES = 10000
ACT_HANG_CAP_S = 30.0
EPISODE_WALL_BUDGET_S = 480.0     # cumulative reset()/act() wait budget
SMOKE_WALL_BUDGET_S = 120.0       # the same allowance for each smoke run


def intrinsics(width: int, height: int, fovy_deg: float) -> np.ndarray:
    fy = 0.5 * height / np.tan(np.deg2rad(fovy_deg) / 2)
    return np.array([[fy, 0.0, width / 2.0], [0.0, fy, height / 2.0],
                     [0.0, 0.0, 1.0]])


def top_extrinsics() -> np.ndarray:
    """T_world_cam of the top camera (x right, y up, looks along -z)."""
    T = np.eye(4)
    T[:3, 3] = TOP_CAM_POS
    return T


CAMERAS = {"top": (TOP_W, TOP_H, TOP_FOVY_DEG),
           "wrist": (WRIST_W, WRIST_H, WRIST_FOVY_DEG)}


def cell_spec() -> dict:
    """The agent-facing cell datasheet (JSON-serializable)."""
    out = {
        "ctrl_hz": CTRL_HZ, "frame_hz": FRAME_HZ, "timestep": TIMESTEP,
        "episode_t": EPISODE_T, "n_ctrl": N_CTRL,
        "arm_home": list(ARM_HOME),
        "suction_ctrl_range": list(SUCTION_CTRL_RANGE),
        "suction_on_threshold": SUCTION_ON,
        "cup_r": CUP_R, "tool_l": TOOL_L,
        "seal_max_tilt_deg": SEAL_MAX_TILT_DEG, "seal_edge_m": SEAL_EDGE_M,
        "belt_ride_dz": BELT_RIDE_DZ, "belt_ride_dv": BELT_RIDE_DV,
        "ft_noise_force_n": FT_NOISE_FORCE_N,
        "ft_noise_torque_nm": FT_NOISE_TORQUE_NM,
        "bin_lo": list(BIN_LO), "bin_hi": list(BIN_HI),
        "bin_inner": list(BIN_INNER), "bin_floor_z": BIN_FLOOR_Z,
        "bin_rim_z": BIN_RIM_Z,
        "belt_y": BELT_Y, "belt_half_w": BELT_HALF_W, "belt_top": BELT_TOP,
        "belt_x_range": list(BELT_X_RANGE), "spawn_x": SPAWN_X,
        "belt_end_x": BELT_END_X, "belt_dir": [-1.0, 0.0, 0.0],
        "belt_speed_range": list(BELT_SPEED_RANGE),
        "box_types": {name: list(d) for name, d in BOX_TYPES},
        "box_density": BOX_DENSITY,
        "fit_res": FIT_RES, "fit_clearance": FIT_CLEARANCE,
        "support_tol": SUPPORT_TOL, "support_frac": SUPPORT_FRAC,
        "full_streak": FULL_STREAK, "pack_tol": PACK_TOL,
        "pack_max_tilt_deg": PACK_MAX_TILT_DEG,
        "top_K": intrinsics(TOP_W, TOP_H, TOP_FOVY_DEG).tolist(),
        "top_T_world_cam": top_extrinsics().tolist(),
        "top_wh": [TOP_W, TOP_H],
        "wrist_K": intrinsics(WRIST_W, WRIST_H, WRIST_FOVY_DEG).tolist(),
        "wrist_wh": [WRIST_W, WRIST_H],
        "wrist_mount_pos": list(WRIST_MOUNT_POS),
        "wrist_mount_quat": list(WRIST_MOUNT_QUAT),
        "model_file": "cell.mjb",
    }
    return out
