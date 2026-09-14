"""Canonical follower definitions for the task09 GELLO variants."""
from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class GelloVariant:
    name: str
    follower_name: str
    follower_chain: tuple[dict, ...]
    follower_ee: dict
    home: tuple[float, ...]
    workspace_halfwidth: tuple[float, ...]

    @property
    def n_joints(self) -> int:
        return len(self.follower_chain)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(f"lead_joint{i}" for i in range(1, self.n_joints + 1))


_FRANKA_CHAIN = (
    dict(link="link1", pos=(0.0, 0.0, 0.333), quat=(1.0, 0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), range=(-2.8973, 2.8973)),
    dict(link="link2", pos=(0.0, 0.0, 0.0), quat=(1.0, -1.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), range=(-1.7628, 1.7628)),
    dict(link="link3", pos=(0.0, -0.316, 0.0), quat=(1.0, 1.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), range=(-2.8973, 2.8973)),
    dict(link="link4", pos=(0.0825, 0.0, 0.0), quat=(1.0, 1.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), range=(-3.0718, -0.0698)),
    dict(link="link5", pos=(-0.0825, 0.384, 0.0), quat=(1.0, -1.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), range=(-2.8973, 2.8973)),
    dict(link="link6", pos=(0.0, 0.0, 0.0), quat=(1.0, 1.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), range=(-0.0175, 3.7525)),
    dict(link="link7", pos=(0.088, 0.0, 0.0), quat=(1.0, 1.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), range=(-2.8973, 2.8973)),
)

_UR5E_CHAIN = (
    dict(link="shoulder_link", pos=(0.0, 0.0, 0.163), quat=(1.0, 0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), range=(-6.28319, 6.28319)),
    dict(link="upper_arm_link", pos=(0.0, 0.138, 0.0), quat=(0.7071067812, 0.0, 0.7071067812, 0.0), axis=(0.0, 1.0, 0.0), range=(-6.28319, 6.28319)),
    dict(link="forearm_link", pos=(0.0, -0.131, 0.425), quat=(1.0, 0.0, 0.0, 0.0), axis=(0.0, 1.0, 0.0), range=(-3.1415, 3.1415)),
    dict(link="wrist_1_link", pos=(0.0, 0.0, 0.392), quat=(0.7071067812, 0.0, 0.7071067812, 0.0), axis=(0.0, 1.0, 0.0), range=(-6.28319, 6.28319)),
    dict(link="wrist_2_link", pos=(0.0, 0.127, 0.0), quat=(1.0, 0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), range=(-6.28319, 6.28319)),
    dict(link="wrist_3_link", pos=(0.0, 0.0, 0.1), quat=(1.0, 0.0, 0.0, 0.0), axis=(0.0, 1.0, 0.0), range=(-6.28319, 6.28319)),
)

_XARM7_CHAIN = (
    dict(link="link1", pos=(0.0, 0.0, 0.267), quat=(1.0, 0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), range=(-6.28319, 6.28319)),
    dict(link="link2", pos=(0.0, 0.0, 0.0), quat=(0.7071067812, -0.7071067812, 0.0, 0.0), axis=(0.0, 0.0, 1.0), range=(-2.059, 2.0944)),
    dict(link="link3", pos=(0.0, -0.293, 0.0), quat=(0.7071067812, 0.7071067812, 0.0, 0.0), axis=(0.0, 0.0, 1.0), range=(-6.28319, 6.28319)),
    dict(link="link4", pos=(0.0525, 0.0, 0.0), quat=(0.7071067812, 0.7071067812, 0.0, 0.0), axis=(0.0, 0.0, 1.0), range=(-0.19198, 3.927)),
    dict(link="link5", pos=(0.0775, -0.3425, 0.0), quat=(0.7071067812, 0.7071067812, 0.0, 0.0), axis=(0.0, 0.0, 1.0), range=(-6.28319, 6.28319)),
    dict(link="link6", pos=(0.0, 0.0, 0.0), quat=(0.7071067812, 0.7071067812, 0.0, 0.0), axis=(0.0, 0.0, 1.0), range=(-1.69297, 3.14159)),
    dict(link="link7", pos=(0.076, 0.097, 0.0), quat=(0.7071067812, -0.7071067812, 0.0, 0.0), axis=(0.0, 0.0, 1.0), range=(-6.28319, 6.28319)),
)


VARIANTS = {
    "franka": GelloVariant(
        "franka", "Franka Panda", _FRANKA_CHAIN,
        dict(pos=(0.0, 0.0, 0.107), quat=(0.3826834, 0.0, 0.0, 0.9238795)),
        (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785),
        (0.7, 0.45, 0.5, 0.5, 0.6, 0.5, 0.7)),
    "ur5e": GelloVariant(
        "ur5e", "Universal Robots UR5e", _UR5E_CHAIN,
        dict(pos=(0.0, 0.1, 0.0), quat=(-0.7071067812, 0.7071067812, 0.0, 0.0)),
        (0.0, 1.1295, 2.3889, 0.4709, -1.8642, -0.1313),
        (0.7, 0.45, 0.5, 0.5, 0.6, 0.5)),
    "xarm7": GelloVariant(
        "xarm7", "UFACTORY xArm7", _XARM7_CHAIN,
        dict(pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)),
        (0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0),
        (0.7, 0.45, 0.5, 0.5, 0.6, 0.5, 0.7)),
}


def get_variant(name: str) -> GelloVariant:
    try:
        return VARIANTS[name]
    except KeyError as exc:
        raise ValueError(f"unknown GELLO variant {name!r}; expected one of {tuple(VARIANTS)}") from exc


def selected_variant() -> GelloVariant:
    return get_variant(os.environ.get("RLEBENCH_GELLO_VARIANT", "franka"))
