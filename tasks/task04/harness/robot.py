"""The G1 model, with BeyondMimic's actuation instead of menagerie's.

Menagerie ships one flat position servo for all 29 joints (kp=500, dampratio=1).
BeyondMimic sizes each joint from its actual motor: armature from the gearbox,
then stiffness and damping from a 10 Hz natural frequency at damping ratio 2, and
the action scale from the torque limit. Those numbers set how far one unit of
action moves a joint, so they are part of the deployment contract in all but
name — a policy trained against the wrong ones does not transfer.
"""
from __future__ import annotations

import os

import numpy as np

import mujoco

from . import spec


ARMATURE = {
    "5020": 0.003609725,
    "7520_14": 0.010177520,
    "7520_22": 0.025101925,
    "4010": 0.00425,
}
NATURAL_FREQ = 10.0 * 2.0 * np.pi
DAMPING_RATIO = 2.0
FLOOR_FRICTION = 1.0
_ACTUATION = {
    "hip_pitch": ("7520_14", 1, 88.0),
    "hip_roll": ("7520_22", 1, 139.0),
    "hip_yaw": ("7520_14", 1, 88.0),
    "knee": ("7520_22", 1, 139.0),
    "ankle_pitch": ("5020", 2, 50.0),
    "ankle_roll": ("5020", 2, 50.0),
    "waist_yaw": ("7520_14", 1, 88.0),
    "waist_roll": ("5020", 2, 50.0),
    "waist_pitch": ("5020", 2, 50.0),
    "shoulder_pitch": ("5020", 1, 25.0),
    "shoulder_roll": ("5020", 1, 25.0),
    "shoulder_yaw": ("5020", 1, 25.0),
    "elbow": ("5020", 1, 25.0),
    "wrist_roll": ("5020", 1, 25.0),
    "wrist_pitch": ("4010", 1, 5.0),
    "wrist_yaw": ("4010", 1, 5.0),
}
DEFAULT_POSE = {
    "hip_pitch": -0.312,
    "knee": 0.669,
    "ankle_pitch": -0.363,
    "elbow": 0.6,
    "left_shoulder_roll": 0.2,
    "left_shoulder_pitch": 0.2,
    "right_shoulder_roll": -0.2,
    "right_shoulder_pitch": 0.2,
}


def _kind(joint: str) -> str:
    return joint.removeprefix("left_").removeprefix("right_").removesuffix("_joint")


def _tables():
    armature = np.empty(spec.N_JOINTS)
    effort = np.empty(spec.N_JOINTS)
    default = np.zeros(spec.N_JOINTS)
    for i, joint in enumerate(spec.JOINT_NAMES):
        motor, count, torque = _ACTUATION[_kind(joint)]
        armature[i] = count * ARMATURE[motor]
        effort[i] = torque
        stripped = joint.removesuffix("_joint")
        default[i] = DEFAULT_POSE.get(stripped, DEFAULT_POSE.get(_kind(joint), 0.0))
    stiffness = armature * NATURAL_FREQ**2
    damping = 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ
    return armature, stiffness, damping, effort, default

ARMATURES, STIFFNESS, DAMPING, EFFORT, DEFAULT_JOINT_POS = _tables()
ACTION_SCALE = 0.25 * EFFORT / STIFFNESS


def _cylinders_to_spheres(sspec) -> int:
    n = 0
    for geom in sspec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
            geom.size[1] = geom.size[2] = 0.0
            n += 1
    return n


def build(robot_dir: str, timestep: float = spec.PHYSICS_DT,
          friction: float = FLOOR_FRICTION) -> mujoco.MjModel:
    sspec = mujoco.MjSpec.from_file(os.path.join(robot_dir, "scene.xml"))
    _cylinders_to_spheres(sspec)
    model = sspec.compile()
    model.opt.timestep = timestep

    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    model.geom_friction[floor, 0] = friction

    for i, joint in enumerate(spec.JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        model.dof_armature[model.jnt_dofadr[jid]] = ARMATURES[i]
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint)
        model.actuator_gainprm[aid, 0] = STIFFNESS[i]
        model.actuator_biasprm[aid, 1] = -STIFFNESS[i]
        model.actuator_biasprm[aid, 2] = -DAMPING[i]
        model.actuator_forcerange[aid] = (-EFFORT[i], EFFORT[i])
    return model


def joint_qpos_index(model) -> np.ndarray:
    return np.array([model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in spec.JOINT_NAMES])


def body_index(model, names) -> np.ndarray:
    return np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                     for n in names])


def action_to_target(action: np.ndarray) -> np.ndarray:
    return DEFAULT_JOINT_POS + action * ACTION_SCALE
