"""Private standalone robosuite factory for the three tabletop scenes."""

from __future__ import annotations

from typing import Any, Iterable

TASKS = (
    "TowerMaxHeight",
    "CantileverOverhang",
    "BalanceCoins",
)


def make(task: str, seed: int | None = None,
         camera_names: Iterable[str] = (), camera_height: int = 128,
         camera_width: int = 128, camera_depths: bool = False,
         **kwargs: Any):
    """Build one tabletop scene. Mirrors env_utils.create_env's conventions:
    default composite controller for the robot, 20 Hz, never-ending horizon
    (the harness enforces the per-trial cap), hard reset per episode."""
    from robosuite.controllers import load_composite_controller_config

    from .scenes import scene_class

    controller_configs = load_composite_controller_config(
        controller=None, robot="PandaOmron")
    # RoboCasa's PandaOmron patch (kitchen.py), reproduced verbatim: the action
    # layout [arm, gripper, base, torso, mode] and the legacy velocity base are
    # the pinned mobile controller contract.
    controller_configs.setdefault("composite_controller_specific_configs", {})[
        "body_part_ordering"] = ["right", "right_gripper", "base", "torso"]
    controller_configs["body_parts"]["base"]["type"] = "JOINT_VELOCITY_LEGACY"

    kwargs.setdefault("renderer", "mujoco")
    cls = scene_class(task)
    return cls(
        robots="PandaOmron",
        controller_configs=controller_configs,
        control_freq=20,
        horizon=100_000,
        ignore_done=True,
        hard_reset=True,
        use_camera_obs=bool(camera_names),
        camera_names=list(camera_names),
        camera_heights=camera_height,
        camera_widths=camera_width,
        camera_depths=camera_depths,
        has_offscreen_renderer=True,
        has_renderer=False,
        seed=seed,
        **kwargs,
    )
