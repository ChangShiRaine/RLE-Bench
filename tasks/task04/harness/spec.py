"""The deployment contract — FROZEN.

Everything here is shared by the trainer and the evaluator and may not be changed
by a submission: the observation layout, the action space, the control rate, and
the names they are built from. The agent owns the algorithm, the network, the
rewards, the randomization and the sampling; it does not own this file, because
the evaluator has to construct observations for a policy it did not train.

Numeric model constants (gains, armature, action scale, default pose) live in
robot.py, which derives them from the motor tables.
"""
from __future__ import annotations


JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
N_JOINTS = len(JOINT_NAMES)
TRACKED_BODIES = (
    "pelvis",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)
ANCHOR_BODY = "torso_link"
ROOT_BODY = "pelvis"
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
OBS_TERMS = (
    ("command", 2 * N_JOINTS),
    ("motion_anchor_pos_b", 3),
    ("motion_anchor_ori_b", 6),
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("joint_pos_rel", N_JOINTS),
    ("joint_vel_rel", N_JOINTS),
    ("actions", N_JOINTS),
)
OBS_DIM = sum(d for _, d in OBS_TERMS)
N_ACTIONS = N_JOINTS
PRIVILEGED_TERMS = OBS_TERMS[:3] + (
    ("body_pos_b", 3 * len(TRACKED_BODIES)),
    ("body_ori_b", 6 * len(TRACKED_BODIES)),
) + OBS_TERMS[3:]
PRIVILEGED_DIM = sum(d for _, d in PRIVILEGED_TERMS)
CONTROL_DT = 0.02
MOTION_FPS = 50
PHYSICS_DT = 0.005
DECIMATION = round(CONTROL_DT / PHYSICS_DT)
EPISODE_S = 10.0
POLICY_FILE = "policy.onnx"
META_FILE = "policy_meta.json"
OBS_INPUT = "obs"
ACTION_OUTPUT = "actions"
MAX_PARAMS = 10_000_000
MAX_HISTORY = 16


def obs_slice(name: str) -> slice:
    start = 0
    for term, dim in OBS_TERMS:
        if term == name:
            return slice(start, start + dim)
        start += dim
    raise KeyError(name)

TERM_ANCHOR_Z = 0.25
TERM_ANCHOR_TILT = 0.8
TERM_EE_Z = 0.25
END_EFFECTORS = ("left_ankle_roll_link", "right_ankle_roll_link",
                 "left_wrist_yaw_link", "right_wrist_yaw_link")
