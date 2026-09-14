"""Canonical manipulator variants for the task08 common-base evaluation.

The submitted ``robot.xml`` still contains a Panda so that it is directly
loadable in the public workspace.  The task08 verifier does not trust that
copy of the arm: it removes every submitted ``arm_*`` subtree and attaches one
of the canonical models below at the submitted ``arm_mount_site``.  The same
base geometry is therefore exercised with every arm.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os

import mujoco
import numpy as np


MENAGERIE_COMMIT = "da76818e269b82289eba39808e2fb91d679d6994"

# Verifier-owned universal adapter supplied to the agent as a stock component.
# It is a 180 x 180 x 10 mm aluminum plate.  The submitted mount site denotes
# the underside of the plate; the canonical arm mounting plane is its top.
ADAPTER_HALF_SIZE_M = (0.09, 0.09, 0.005)
ADAPTER_MASS_KG = 0.875
ADAPTER_RGBA = (0.35, 0.38, 0.42, 1.0)


@dataclass(frozen=True)
class ArmSpec:
    """Trusted model and controller metadata for one canonical arm."""

    name: str
    reference_xml: str
    root_body: str
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    attachment_site: str
    stow: tuple[float, ...]
    extended_template: tuple[float, ...]
    # Horizontal angle of the extended flange vector when joint 1 equals the
    # template value.  Joint 1 rotates that vector one-for-one for all three
    # canonical arms.
    extended_zero_azimuth: float
    ik_seeds: tuple[tuple[float, ...], ...] = ()
    # Arm-side orientation of the universal adapter (w, x, y, z).
    mount_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    @property
    def prefix(self) -> str:
        return f"arm_{self.name}_"

    def body(self, local_name: str) -> str:
        return self.prefix + local_name

    def joint(self, index: int) -> str:
        return self.prefix + self.joint_names[index]

    def actuator(self, index: int) -> str:
        return self.prefix + self.actuator_names[index]

    @property
    def ee_site(self) -> str:
        return self.prefix + self.attachment_site

    @property
    def payload_body(self) -> str:
        return self.prefix + "payload"

    @property
    def payload_geom(self) -> str:
        return self.prefix + "payload_geom"

    @property
    def assembled_root(self) -> str:
        return self.body(self.root_body)

    def extended(self, azimuth_rad: float) -> np.ndarray:
        q = np.asarray(self.extended_template, dtype=float).copy()
        q[0] += float(azimuth_rad) - self.extended_zero_azimuth
        # Continuous joints in the source assets use finite +/- 2pi ranges.
        q[0] = (q[0] + np.pi) % (2.0 * np.pi) - np.pi
        # ... and a wrapped angle can still sit outside a LIMITED joint 1:
        # the Panda cannot slew to +-pi at all. Commanding it there jams the
        # joint against its stop and the reaction tips the base, so ask for
        # the nearest azimuth the arm can actually hold.
        limit = slew_limit(self)
        q[0] = float(np.clip(q[0], -limit, limit))
        return q

    def extended_at_slew(self, joint1_rad: float) -> np.ndarray:
        """The extended pose with joint 1 placed directly, clamped to range.

        Scenario A sweeps joint 1 monotonically across the span the arm can
        actually reach. Stepping through azimuths instead and wrapping each
        one on its own makes the sequence non-monotonic in joint space, and
        the ctrl interpolation then slews a fully extended arm the long way
        round -- 5.4 rad in 1 s -- rather than on to the next pose.
        """
        q = np.asarray(self.extended_template, dtype=float).copy()
        limit = slew_limit(self)
        q[0] = float(np.clip(joint1_rad, -limit, limit))
        return q

    def controller_seeds(self) -> tuple[np.ndarray, ...]:
        seeds = [np.asarray(self.stow, dtype=float),
                 np.asarray(self.extended_template, dtype=float)]
        seeds.extend(np.asarray(q, dtype=float) for q in self.ik_seeds)
        return tuple(q.copy() for q in seeds)


@lru_cache(maxsize=None)
def slew_limit(arm: "ArmSpec") -> float:
    """How far joint 1 can actually slew, from the arm's own model.

    The canonical arms differ: the UR5e and xArm7 turn a full circle, the
    Panda stops at +-2.8973 rad (166 deg). A commanded azimuth past that is
    not a reach the arm declines to make, it is the actuator driving the joint
    into its hard limit -- which on a mobile base is an impulse, not a pose.
    """
    model = mujoco.MjModel.from_xml_path(arm.reference_xml)
    joint = model.joint(arm.joint_names[0])
    lo, hi = float(joint.range[0]), float(joint.range[1])
    if lo == 0.0 and hi == 0.0:
        return float(np.pi)            # unlimited joint
    return float(min(np.pi, min(-lo, hi)))


@lru_cache(maxsize=None)
def canonical_reach(arm: "ArmSpec") -> float:
    """Horizontal flange reach of the arm ON ITS OWN, in the canonical
    extended pose.

    This is a property of the manipulator, not of the base it is bolted to,
    and it differs widely across the three canonical arms (the xArm7 reaches
    ~0.77 m where the UR5e reaches ~0.97 m). S3.reach_preserved compares the
    assembled reach against a fraction of this, so the checkpoint asks what
    its name says — did the submitted base cost the arm any of its own reach
    — instead of asking every arm to clear one absolute bar that the shortest
    of them cannot.
    """
    model = mujoco.MjModel.from_xml_path(arm.reference_xml)
    data = mujoco.MjData(model)
    for i, q in enumerate(arm.extended(0.0)):
        data.qpos[model.joint(arm.joint_names[i]).qposadr[0]] = q
    mujoco.mj_forward(model, data)
    site = data.site(arm.attachment_site).xpos
    root = data.body(arm.root_body).xpos
    return float(np.hypot(site[0] - root[0], site[1] - root[1]))


def trusted_arm_specs(panda_reference_xml: str) -> dict[str, ArmSpec]:
    """Build the three specs relative to the verifier's trusted asset root."""
    assets = os.path.dirname(os.path.dirname(os.path.abspath(
        panda_reference_xml)))
    panda = os.path.join(assets, "franka_emika_panda", "panda_nohand.xml")
    ur5e = os.path.join(assets, "universal_robots_ur5e", "ur5e.xml")
    xarm7 = os.path.join(assets, "ufactory_xarm7", "xarm7_nohand.xml")
    specs = {
        "panda": ArmSpec(
            name="panda", reference_xml=panda, root_body="link0",
            joint_names=tuple(f"joint{i}" for i in range(1, 8)),
            actuator_names=tuple(f"actuator{i}" for i in range(1, 8)),
            attachment_site="attachment_site",
            stow=(0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785),
            extended_template=(0.0, 1.35, 0.0, -0.4, 0.0, 2.5, 0.785),
            extended_zero_azimuth=0.0,
            ik_seeds=(
                (0.0, -0.3, 0.0, -1.2, 0.0, 2.0, 0.785),
                (0.0, 1.2, 0.0, -1.0, 0.0, 2.2, 0.785),
                (0.0, -0.8, 1.4, -1.8, -1.2, 2.2, 0.785),
                (0.0, -0.8, -1.4, -1.8, 1.2, 2.2, 0.785),
                (0.8, 0.6, 1.2, -1.5, -1.0, 2.0, 0.785),
                (-0.8, 0.6, -1.2, -1.5, 1.0, 2.0, 0.785),
            ),
        ),
        "ur5e": ArmSpec(
            name="ur5e", reference_xml=ur5e, root_body="base",
            joint_names=("shoulder_pan_joint", "shoulder_lift_joint",
                         "elbow_joint", "wrist_1_joint", "wrist_2_joint",
                         "wrist_3_joint"),
            actuator_names=("shoulder_pan", "shoulder_lift", "elbow",
                            "wrist_1", "wrist_2", "wrist_3"),
            attachment_site="attachment_site",
            # Centered, collision-free fold selected by deterministic FK/CoM
            # sweep; unlike the Menagerie display pose it settles without
            # rolling the mecanum base.
            stow=(-0.83943, -1.22370, -1.33717, -1.07738, -3.03934, 3.03382),
            # Collision-free FK sweep result at 0.965 m horizontal reach.
            extended_template=(0.0, -0.10019, 0.11640, -0.66953,
                               1.51886, -0.71154),
            # The stock model naturally extends toward -x.  Its adapter
            # pattern rotates the arm 180 degrees so all variants share the
            # submitted base's +x forward direction.
            extended_zero_azimuth=0.14474,
            ik_seeds=(
                # Shelf-entry elbow branch.
                (0.1, -3.8, 1.3, 2.2, -0.5, 0.0),
                (0.0, -1.2, 1.8, -1.8, -1.57, 0.0),
                (1.57, -1.2, 1.8, -1.8, -1.57, 0.0),
                (-1.57, -1.2, 1.8, -1.8, -1.57, 0.0),
                (0.0, -0.6, 1.2, -2.0, 1.57, 0.0),
            ),
            mount_quat=(0.0, 0.0, 0.0, 1.0),
        ),
        "xarm7": ArmSpec(
            name="xarm7", reference_xml=xarm7, root_body="link_base",
            joint_names=tuple(f"joint{i}" for i in range(1, 8)),
            actuator_names=tuple(f"act{i}" for i in range(1, 8)),
            attachment_site="attachment_site",
            # Compact centered fold; keeps the arm clear of the shelf while
            # the mecanum base stages.
            stow=(-2.26731, -0.62319, -0.00416, 0.22692, 0.02608,
                  2.72839, -0.08641),
            # Collision-free FK sweep result at 0.770 m horizontal reach.
            extended_template=(0.0, 1.39471, -0.24112, 2.74108,
                               3.09956, 0.71877, 3.00779),
            extended_zero_azimuth=-0.04817,
            ik_seeds=(
                (0.0, -1.0, 0.0, 2.2, 0.0, 0.8, 0.0),
                (0.0, 1.0, 0.0, 2.2, 0.0, 0.8, 0.0),
                (1.57, -1.0, 1.0, 2.2, -1.0, 0.8, 0.0),
                (-1.57, -1.0, -1.0, 2.2, 1.0, 0.8, 0.0),
            ),
        ),
    }
    missing = [spec.reference_xml for spec in specs.values()
               if not os.path.isfile(spec.reference_xml)]
    if missing:
        raise FileNotFoundError("canonical arm assets missing: "
                                + ", ".join(missing))
    return specs


def _remove_submitted_arm(robot: mujoco.MjSpec) -> None:
    """Remove all untrusted arm subtrees and their direct references."""
    for actuator in list(robot.actuators):
        if actuator.name.startswith("arm_"):
            robot.delete(actuator)
    for exclude in list(robot.excludes):
        if (exclude.bodyname1.startswith("arm_")
                or exclude.bodyname2.startswith("arm_")):
            robot.delete(exclude)
    for key in list(robot.keys):
        # Keyframe vectors encode the old arm's nq/nu and cannot survive a
        # variant swap.  Scenario initialization is verifier-owned anyway.
        robot.delete(key)
    for light in list(robot.lights):
        if (light.name.startswith("arm_")
                or light.targetbody.startswith("arm_")):
            robot.delete(light)

    bodies = list(robot.bodies)
    roots = []
    for body in bodies:
        if not body.name.startswith("arm_"):
            continue
        parent_name = getattr(body.parent, "name", "")
        if not parent_name.startswith("arm_"):
            roots.append(body)
    if not roots:
        raise ValueError("robot.xml must contain a submitted arm_* subtree")
    for body in roots:
        robot.delete(body)


def canonical_robot_spec(robot_xml: str, arm: ArmSpec) -> mujoco.MjSpec:
    """Return the submitted base with a trusted adapter and canonical arm."""
    robot = mujoco.MjSpec.from_file(robot_xml)
    try:
        mount = robot.site("arm_mount_site")
    except KeyError as exc:
        raise ValueError("robot.xml missing required site 'arm_mount_site'") from exc
    if getattr(mount.parent, "name", "") != "base":
        raise ValueError("arm_mount_site must be a direct child of body 'base'")
    mount_parent = mount.parent
    mount_pos = list(mount.pos)
    mount_quat = list(mount.quat)
    _remove_submitted_arm(robot)

    adapter = mount_parent.add_body(
        name="arm_adapter", pos=mount_pos, quat=mount_quat)
    plate = adapter.add_geom(
        name="arm_adapter_plate", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(ADAPTER_HALF_SIZE_M),
        pos=[0.0, 0.0, ADAPTER_HALF_SIZE_M[2]],
        rgba=list(ADAPTER_RGBA))
    plate.mass = ADAPTER_MASS_KG

    canonical = mujoco.MjSpec.from_file(arm.reference_xml)
    # The adapter defines placement; the arm root sits on its top surface.
    canonical.body(arm.root_body).pos = [0.0, 0.0, 0.0]
    for key in list(canonical.keys):
        canonical.delete(key)
    for light in list(canonical.lights):
        canonical.delete(light)

    # A common payload stub is rigidly attached in the tool site's local frame.
    attachment = canonical.site(arm.attachment_site)
    payload = attachment.parent.add_body(
        name="payload", pos=list(attachment.pos), quat=list(attachment.quat))
    payload_geom = payload.add_geom(
        name="payload_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.04, 0.0, 0.0], pos=[0.0, 0.0, 0.05],
        rgba=[0.80, 0.10, 0.10, 1.0])
    payload_geom.mass = 0.001

    frame = adapter.add_frame(
        pos=[0.0, 0.0, 2.0 * ADAPTER_HALF_SIZE_M[2]],
        quat=list(arm.mount_quat))
    robot.attach(canonical, prefix=arm.prefix, frame=frame)
    return robot


def canonical_arm_body_names(arm: ArmSpec) -> set[str]:
    """Names excluded from chassis budgets after canonical composition."""
    ref = mujoco.MjModel.from_xml_path(arm.reference_xml)
    names = {arm.body(ref.body(i).name) for i in range(1, ref.nbody)}
    names.update({"arm_adapter", arm.payload_body})
    return names
