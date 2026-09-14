"""Compose the shared metered harness for the vision-only two-arm cube task."""

from __future__ import annotations

import numpy as np

from .scene import CAMERAS, PocketCube

RESOLUTION = 512
ACTION_LAYOUT = {"robot0_arm_osc_pose": [0, 6], "robot0_gripper": [6, 7],
                 "robot1_arm_osc_pose": [7, 13], "robot1_gripper": [13, 14]}
ROBOT_FIELDS = ("joint_pos", "joint_pos_cos", "joint_pos_sin", "joint_vel", "joint_acc",
                "eef_pos", "eef_quat", "eef_quat_site", "gripper_qpos", "gripper_qvel",
                "proprio-state")
OBSERVATIONS = frozenset(
    [f"robot{i}_{field}" for i in range(2) for field in ROBOT_FIELDS]
    + [f"{camera}_{kind}" for camera in CAMERAS for kind in ("image", "depth")]
)


def agent_view(obs, env=None, level=None):
    return {key: value for key, value in obs.items() if key in OBSERVATIONS} if isinstance(obs, dict) else obs


def calibration(env):
    if env is None:
        return {}
    model, data = env.sim.model._model, env.sim.data._data
    out = {}
    for name in CAMERAS:
        cid = model.camera(name).id
        focal = RESOLUTION / (2 * np.tan(np.deg2rad(model.cam_fovy[cid]) / 2))
        out[name] = {"width": RESOLUTION, "height": RESOLUTION,
                     "intrinsics": [[float(focal), 0, RESOLUTION/2], [0, float(focal), RESOLUTION/2], [0, 0, 1]],
                     "position_world": data.cam_xpos[cid].tolist(),
                     "rotation_world_from_opengl_camera": data.cam_xmat[cid].reshape(3, 3).tolist()}
    return out


def install():
    from harness import config, debug, env, privileged
    from harness.service import Service

    if getattr(Service, "_pocket_composed", False):
        return
    config.OBS_RESOLUTION = env.OBS_RESOLUTION = RESOLUTION
    env.DEFAULT_CAMERAS = tuple(CAMERAS)
    debug.TILED_CAMERAS = tuple(f"{name}_image" for name in CAMERAS)
    config.agent_visible_obs = agent_view
    privileged.agent_view = agent_view

    def make_env(task, split="pretrain", seed=None, scene=None, **kwargs):
        if task != "RubikCube":
            raise ValueError("This image serves RubikCube only")
        env._use_upright_images()
        for old, new in (("camera_height", "camera_heights"), ("camera_width", "camera_widths")):
            if old in kwargs:
                kwargs[new] = kwargs.pop(old)
        return PocketCube(robot_friendly=True, seed=0 if seed is None else seed, **kwargs)

    env.make_env = make_env
    dispatch = Service._dispatch

    def cube_dispatch(self, op, msg):
        reply = dispatch(self, op, msg)
        if op == "task_info":
            reply.update(action_layout=ACTION_LAYOUT, cameras=list(CAMERAS),
                         camera_calibration=calibration(self._session.current_env),
                         action_reference_frame="each robot's fixed base; robot1 is rotated 180 degrees about world z",
                         robot_base_poses=[{"pos": [-0.52, 0.18, 0.775], "quat_xyzw": [0, 0, 0, 1]},
                                           {"pos": [0.52, 0.18, 0.775], "quat_xyzw": [0, 0, 1, 0]}])
        return reply

    Service._dispatch = cube_dispatch
    Service._pocket_composed = True


def main(argv=None):
    install()
    from harness.daemon_main import main as shared_main
    return shared_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
