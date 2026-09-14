"""Public constants for task07 (bin clearing / factory pick-and-place).

Everything in this module is agent-visible. It is the single source of truth
for the workcell geometry, the part, the camera and depth-sensor models, the
control rates, and the policy interface contract. The evaluation drives the
same numbers, so a policy developed against this file sees exactly the
physics and sensors the evaluation uses (on different seeds).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------
ASSETS_DIR = Path(os.environ.get(
    "RLEBENCH_ASSETS",
    Path(__file__).resolve().parents[1] / "assets"))
if (ASSETS_DIR / "franka_emika_panda").exists():
    PANDA_XML = ASSETS_DIR / "franka_emika_panda" / "panda_nohand.xml"
else:  # source checkout: the vendored robot descriptions
    _parents = Path(__file__).resolve().parents
    _repo = _parents[3] if len(_parents) > 3 else _parents[-1]
    PANDA_XML = (_repo / "assets" / "robots"
                 / "franka_emika_panda" / "panda_nohand.xml")

# ---------------------------------------------------------------------------
# Timing (world z = 0 is the workcell floor; SI units, radians)
# ---------------------------------------------------------------------------
TIMESTEP = 0.002                  # 500 Hz physics, implicitfast
CTRL_HZ = 20.0                    # policy act() rate
CTRL_DECIMATION = int(round(1.0 / (CTRL_HZ * TIMESTEP)))   # 25 substeps
FRAME_HZ = 10.0                   # camera observation rate
FRAME_EVERY_TICKS = int(round(CTRL_HZ / FRAME_HZ))          # every 2nd tick
EPISODE_T = 120.0                 # sim-time budget per episode (seconds);
                                  # emptying a full bin demands a
                                  # ~12 parts/min pace

# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------
PEDESTAL_R = 0.10
PEDESTAL_H = 0.04                 # arm base height
ARM_HOME = (-0.10, -0.45, 0.0, -2.25, 0.0, 1.85, 0.70)
N_CTRL = 8                        # 7 joint position targets + magnet command

# ---------------------------------------------------------------------------
# End-effector: contact electromagnet. While ctrl[7] >= MAG_ON is commanded,
# any part whose geometry touches the magnet FACE is welded to the head at
# its current relative pose (nothing is attracted at a distance). Dropping
# the command below MAG_ON releases every held part. Several nested parts
# can be held at once.
# ---------------------------------------------------------------------------
# Head proportions follow holding electromagnets for this payload class
# (~40 mm dia x 30 mm). The face is the dive pose-tolerance budget: a
# descent that lands within ~face-radius of a part surface still touches.
# The shaft keeps face-to-collar reach at 185 mm so the wrist never has to
# enter the bin.
MAG_HEAD_R = 0.020                # face radius
MAG_SHAFT_R = 0.014               # slim extension probe, long enough that
MAG_SHAFT_L = 0.155               # the collar stays ABOVE the bin rim with
MAG_HEAD_L = 0.030                # the face on the bin floor at a wall
MAG_COLLAR_L = 0.012
MAG_HEAD_H = MAG_COLLAR_L + MAG_SHAFT_L + MAG_HEAD_L   # face offset: 0.197
MAG_CTRL_RANGE = (0.0, 255.0)
MAG_ON = 128.0                    # ctrl[7] at/above this energizes the head
MAG_TOUCH = 0.0015                # contact distance that counts as touching

# Wrist force-torque sensor at the tool mount (between flange and magnet
# collar): measures the wrench transmitted to the tool — tool weight,
# payload, and every contact — expressed in the SITE frame (z = approach
# axis). Published with mild seeded sensor noise, like the depth cameras.
FT_NOISE_FORCE_N = 0.25           # 1-sigma per axis
FT_NOISE_TORQUE_NM = 0.01

# ---------------------------------------------------------------------------
# Bin (KLT-style, welded to its stand)
# ---------------------------------------------------------------------------
STAND_H = 0.10
BIN_POS = (0.47, -0.27, STAND_H)  # outside bottom center, on the stand
BIN_INNER = (0.36, 0.26, 0.19)    # interior L x W x H
BIN_WALL = 0.012
# interior AABB in world coordinates (parts inside the bin are in here)
BIN_LO = (BIN_POS[0] - BIN_INNER[0] / 2, BIN_POS[1] - BIN_INNER[1] / 2,
          STAND_H + BIN_WALL)
BIN_HI = (BIN_POS[0] + BIN_INNER[0] / 2, BIN_POS[1] + BIN_INNER[1] / 2,
          STAND_H + BIN_WALL + BIN_INNER[2] + 0.25)

# ---------------------------------------------------------------------------
# Conveyor + drop zone (static belt; a part at rest in the ZONE is cleared)
# ---------------------------------------------------------------------------
BELT_CENTER = (0.58, 0.46)
BELT_HALF = (0.55, 0.16)
BELT_TOP = 0.12
# The drop zone is a placement NEST: a 14 x 14 cm marked window at the
# reachable end of the belt. Singulated delivery — parts must be PLACED
# into it one at a time (see the clearing rules); the rest of the belt is
# dead surface.
ZONE_LO = (0.38, 0.31, BELT_TOP - 0.005)
ZONE_HI = (0.52, 0.45, BELT_TOP + 0.30)

# ---------------------------------------------------------------------------
# Part: stamped U-channel bracket = web + two flared flanges (thin boxes)
# ---------------------------------------------------------------------------
WEB_HALF = (0.036, 0.021, 0.0025)
FLANGE_HALF = (0.036, 0.017, 0.0025)
FLANGE_FLARE = np.deg2rad(14.0)
PART_DENSITY = 7800.0             # steel
CONTACT_FRICTION = (0.55, 0.005, 0.0001)
N_PARTS_RANGE = (12, 16)          # inclusive draw per episode — sized so a
                                  # perfect singulated clear (one nest
                                  # delivery at a time, ~7 s/cycle) is
                                  # achievable inside the 120 s budget
GEOMS_PER_PART = 3

# ---------------------------------------------------------------------------
# Clearing / penalty rules (harness instrumentation, latched)
# ---------------------------------------------------------------------------
CLEAR_SPEED = 0.02                # m/s: "at rest"
CLEAR_REST_S = 0.5                # continuous rest time in the zone
CLEAR_REST_TICKS = int(round(CLEAR_REST_S * CTRL_HZ))
FLOOR_Z = 0.05                    # COM below this = on the workcell floor
GRAVEYARD = (2.0, -3.0, -1.0)     # cleared parts get pinned here, spaced in x
GRAVEYARD_DX = 0.3

# ---------------------------------------------------------------------------
# Cameras. Fixed overhead RGB-D above the bin + wrist RGB-D on the hand.
# K and extrinsics are published — a factory cell is calibrated. The wrist
# camera pose is published per frame (it is FK of qpos, not ground truth).
# ---------------------------------------------------------------------------
OVERHEAD_POS = (BIN_POS[0], BIN_POS[1], 0.98)
OVERHEAD_QUAT = (1.0, 0.0, 0.0, 0.0)   # looks straight down, image up = +y
OVERHEAD_W, OVERHEAD_H = 640, 480
OVERHEAD_FOVY_DEG = 48.0
WRIST_MOUNT_POS = (0.055, 0.0, 0.02)   # in the tool-mount frame
WRIST_MOUNT_QUAT = (0.0, 1.0, 0.0, 0.0)  # looks along the approach axis
WRIST_W, WRIST_H = 320, 240
WRIST_FOVY_DEG = 58.0
DEPTH_MAX_RANGE = 3.0

# Depth sensor model (structured-light style):
#   sigma(theta) = SIGMA0 + min(SIGMA_K * tan^2(theta), SIGMA_CAP - SIGMA0)
#   dropout where cos(theta) < DROP_COS with probability DROP_P
DEPTH_SIGMA0 = 0.0015
DEPTH_SIGMA_K = 0.004
DEPTH_SIGMA_CAP = 0.030
DEPTH_DROP_COS = 0.18
DEPTH_DROP_P = 0.75

# ---------------------------------------------------------------------------
# Seeds and RNG streams. Design seeds are public: generate unlimited practice
# piles. The battery is ordered from a representative 12-part pile through
# progressively larger/contact-denser piles, ending with seed 101 as the
# 16-part hard-tail stress case. Evaluation uses different, unpublished seeds
# from the same distributions.
# ---------------------------------------------------------------------------
DESIGN_SEEDS = (35, 16, 307, 211, 101)
STREAM_SCENE = 0     # nuisance draw (lighting, floor tint)
STREAM_SENSOR = 1    # depth-noise realization (per camera per frame)
STREAM_PILE = 2      # part count, spawn grid jitter, spawn yaw
STREAM_FT = 3        # force-torque sensor noise (per control tick)

# ---------------------------------------------------------------------------
# Policy interface contract
# ---------------------------------------------------------------------------
# Ship policy/policy.py exposing
#     make_policy() -> obj with
#         reset(cell_spec: dict, seed: int) -> None   (once per episode;
#             seed is for policy RNG and does not identify the pile)
#         act(obs: dict) -> array of N_CTRL floats    (at CTRL_HZ, in order)
# obs keys, every tick:
#     qpos (7,), qvel (7,), tau (7,), ft (6: force N + torque Nm at the
#     tool-mount site, site frame, noisy), mag_on (0/1), t (float seconds)
# There is no part-present channel: a grab is observable only through ft.
# and additionally on FRAME_HZ ticks, per camera ("overhead", "wrist"):
#     <cam>_rgb (H,W,3) uint8, <cam>_depth (H,W) float32 NaN=dropout,
#     wrist_T_world_cam (4,4)  (overhead extrinsics are static, in cell_spec)
# Return ctrl: 7 joint position targets (clamped to actuator ctrlrange) +
# the magnet command in MAG_CTRL_RANGE. Non-finite values fail the contract.
#
# cell_spec carries the numbers above plus "model_file": a compiled MuJoCo
# model of the cell (arm + statics, ZERO parts) for the agent's own
# FK/IK/planning, staged next to the policy at reset time.
ACT_HANG_CAP_S = 30.0            # per-act() hard hang cap (parent clock)
EPISODE_WALL_BUDGET_S = 360.0    # cumulative reset()/act() wait budget;
                                 # simulation and rendering are excluded


def intrinsics(width: int, height: int, fovy_deg: float) -> np.ndarray:
    """Pinhole K (fx = fy from vertical FOV)."""
    fy = 0.5 * height / np.tan(np.deg2rad(fovy_deg) / 2)
    return np.array([[fy, 0.0, width / 2.0],
                     [0.0, fy, height / 2.0],
                     [0.0, 0.0, 1.0]])


def overhead_intrinsics() -> np.ndarray:
    return intrinsics(OVERHEAD_W, OVERHEAD_H, OVERHEAD_FOVY_DEG)


def wrist_intrinsics() -> np.ndarray:
    return intrinsics(WRIST_W, WRIST_H, WRIST_FOVY_DEG)


def overhead_extrinsics() -> np.ndarray:
    """T_world_cam for the overhead camera (x right, y up, looks along -z)."""
    T = np.eye(4)
    T[:3, 3] = OVERHEAD_POS
    return T


def cell_spec() -> dict:
    """The agent-facing cell datasheet (JSON-serializable)."""
    return {
        "ctrl_hz": CTRL_HZ,
        "frame_hz": FRAME_HZ,
        "timestep": TIMESTEP,
        "episode_t": EPISODE_T,
        "n_ctrl": N_CTRL,
        "mag_ctrl_range": list(MAG_CTRL_RANGE),
        "mag_on_threshold": MAG_ON,
        "mag_head_r": MAG_HEAD_R,
        "mag_head_h": MAG_HEAD_H,
        "ft_noise_force_n": FT_NOISE_FORCE_N,
        "ft_noise_torque_nm": FT_NOISE_TORQUE_NM,
        "arm_home": list(ARM_HOME),
        "bin_lo": list(BIN_LO), "bin_hi": list(BIN_HI),
        "bin_top_z": STAND_H + BIN_WALL + BIN_INNER[2],
        "zone_lo": list(ZONE_LO), "zone_hi": list(ZONE_HI),
        "belt_center": list(BELT_CENTER), "belt_half": list(BELT_HALF),
        "belt_top": BELT_TOP,
        "overhead_K": overhead_intrinsics().tolist(),
        "overhead_T_world_cam": overhead_extrinsics().tolist(),
        "overhead_wh": [OVERHEAD_W, OVERHEAD_H],
        "wrist_K": wrist_intrinsics().tolist(),
        "wrist_wh": [WRIST_W, WRIST_H],
        "wrist_mount_pos": list(WRIST_MOUNT_POS),
        "wrist_mount_quat": list(WRIST_MOUNT_QUAT),
        "clear_speed": CLEAR_SPEED,
        "clear_rest_s": CLEAR_REST_S,
        "n_parts_range": list(N_PARTS_RANGE),
        "model_file": "cell.mjb",
    }
