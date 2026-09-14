"""The shelf scene for the submitted-controller reach evaluation.

The shelf sits at its public scene coordinate; the twelve scored targets are
the inset horizontal corners above each shelf plate. controlled_pick.py runs
the episodes and measures hold error, wheel support, stability margin and
shelf contact directly from MuJoCo state.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .. import config
from ..sim.mecanum import geometry_from_model
from ..sim.arm_variants import ArmSpec, canonical_arm_body_names
from ..sim.scenarios import (Envelope, attach_robot, compose_scene, pin_physics,
                             set_payload)
from .checkpoints import footprint_bounds

# --- scene definition ----------------------------------------------------------
BIN_WIDTH_M = 0.50        # interior width (y)
BIN_DEPTH_M = 0.38        # shelf plate depth (x, from the face)
PANEL_T_M = 0.02          # wall/plate/back panel thickness
SHELF_TOPS_Z = (0.20, 0.66, 1.02)   # top surface of each bin's floor plate
POD_TOP_Z = 1.46          # top cap: bins are cubbies, not open trays
TARGET_UP_M = 0.14        # flange hover above the bin floor (the payload
                          # stub hangs ~0.09 m below the flange when pointed
                          # down; the hover must clear the item)
CORNER_INSET_M = 0.10     # clearance from each bin opening edge
TARGET_NAMES = (
    "low_front_left", "low_front_right",
    "low_back_left", "low_back_right",
    "middle_front_left", "middle_front_right",
    "middle_back_left", "middle_back_right",
    "high_front_left", "high_front_right",
    "high_back_left", "high_back_right",
)

# --- pass criteria -----------------------------------------------------------
HOLD_POS_TOL_M = 0.06     # dynamic flange-on-target tolerance (gravity sag)
SHELF_FORCE_MAX_N = 2.0   # any harder robot-shelf press fails the episode
HOLD_WINDOW_T = 1.0       # continuous hold that completes a target

_TERRAIN_FRICTION = [0.9, 0.005, 0.0001]


def _front_extent(robot_xml: str, arm: ArmSpec) -> float:
    """Front-most (+x) footprint extent of the submitted robot relative to
    its base origin, arm excluded — same measurement as the budget check."""
    model, data, _ = compose_scene(robot_xml, "A", 0, Envelope(), arm)
    set_payload(model, data, 0.0, arm)
    _, hi = footprint_bounds(model, data, canonical_arm_body_names(arm))
    return float(hi[0])


def shelf_placement(robot_xml: str, arm_reference_xml: str,
                    arm: ArmSpec) -> tuple[float, float]:
    """Return ``(front_x, face_x)``: the robot's front extent (diagnostic) and
    the fixed public shelf face."""
    return _front_extent(robot_xml, arm), float(config.PICK_SHELF_FACE_X_M)


def shelf_targets(face_x: float) -> list[np.ndarray]:
    """The twelve public flange targets, in TARGET_NAMES order."""
    half_width = BIN_WIDTH_M / 2
    y_left = half_width - CORNER_INSET_M
    x_front = face_x + CORNER_INSET_M
    x_back = face_x + BIN_DEPTH_M - CORNER_INSET_M
    targets = []
    for floor_top in SHELF_TOPS_Z:
        z = floor_top + TARGET_UP_M
        for x in (x_front, x_back):
            for y in (y_left, -y_left):
                targets.append(np.array([x, y, z]))
    return targets


def scored_target_schedule(face_x: float, arm: ArmSpec):
    """Return ``(index, name, target)`` for every scored target."""
    targets = shelf_targets(face_x)
    if len(targets) != len(TARGET_NAMES):
        raise ValueError(
            f"expected {len(TARGET_NAMES)} scored targets for {arm.name}, "
            f"got {len(targets)}")
    return [(index, TARGET_NAMES[index], target)
            for index, target in enumerate(targets)]


def _add_box(spec, name, size, pos):
    g = spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=list(size), pos=list(pos))
    g.priority = 2
    g.friction = _TERRAIN_FRICTION
    return g


def compose_pick_scene(robot_xml: str, face_x: float,
                       payload_kg: float | None = None,
                       arm: ArmSpec | None = None):
    """Floor + shelf pod (face plane at x=face_x) + robot at the origin.

    Returns (model, data, shelf_geom_names); data holds the start pose
    (upright, wheels touching, arm stowed, payload attached).
    """
    spec = mujoco.MjSpec()
    spec.modelname = "rlebench_pick"

    floor = spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                    size=[0, 0, 0.05])
    floor.priority = 2
    floor.friction = _TERRAIN_FRICTION

    cx = face_x + BIN_DEPTH_M / 2          # shelf plate center, x
    half_w = BIN_WIDTH_M / 2
    shelf = []
    shelf.append(_add_box(spec, "pod_wall_L",
                          (BIN_DEPTH_M / 2, PANEL_T_M / 2, POD_TOP_Z / 2),
                          (cx, half_w + PANEL_T_M / 2, POD_TOP_Z / 2)))
    shelf.append(_add_box(spec, "pod_wall_R",
                          (BIN_DEPTH_M / 2, PANEL_T_M / 2, POD_TOP_Z / 2),
                          (cx, -half_w - PANEL_T_M / 2, POD_TOP_Z / 2)))
    shelf.append(_add_box(spec, "pod_back",
                          (PANEL_T_M / 2, half_w + PANEL_T_M, POD_TOP_Z / 2),
                          (face_x + BIN_DEPTH_M + PANEL_T_M / 2, 0.0,
                           POD_TOP_Z / 2)))
    for k, z_top in enumerate(SHELF_TOPS_Z):
        shelf.append(_add_box(spec, f"pod_plate_{k}",
                              (BIN_DEPTH_M / 2, half_w, PANEL_T_M / 2),
                              (cx, 0.0, z_top - PANEL_T_M / 2)))
    shelf.append(_add_box(spec, "pod_top",
                          (BIN_DEPTH_M / 2, half_w, PANEL_T_M / 2),
                          (cx, 0.0, POD_TOP_Z - PANEL_T_M / 2)))

    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
    attach_robot(spec, robot_xml, arm)
    pin_physics(spec)
    model = spec.compile()
    data = mujoco.MjData(model)

    _, _, wheel_r = geometry_from_model(model)
    qadr = model.joint("base_free").qposadr[0]
    mujoco.mj_resetData(model, data)
    data.qpos[qadr:qadr + 3] = [0.0, 0.0, wheel_r + 0.001]
    data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    for i, q in enumerate(arm.stow):
        data.qpos[model.joint(arm.joint(i)).qposadr[0]] = q
        data.ctrl[model.actuator(arm.actuator(i)).id] = q
    if payload_kg is None:
        payload_kg = float(config.PICK_PAYLOAD_KG)
    set_payload(model, data, payload_kg, arm)
    return model, data, [g.name for g in shelf]


def _robot_shelf_force(model, data, shelf_set: set[int],
                       robot_bodies: set[int], cf: np.ndarray) -> float:
    maximum = 0.0
    for ci in range(data.ncon):
        con = data.contact[ci]
        g1, g2 = int(con.geom1), int(con.geom2)
        if (g1 in shelf_set) == (g2 in shelf_set):
            continue
        other = g2 if g1 in shelf_set else g1
        if int(model.geom_bodyid[other]) not in robot_bodies:
            continue
        mujoco.mj_contactForce(model, data, ci, cf)
        maximum = max(maximum, float(cf[0]))
    return maximum
