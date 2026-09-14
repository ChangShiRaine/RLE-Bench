"""Two fixed Panda arms solve a physical pocket cube through contact."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask

from . import contacts, metrics, model as geometry
from .cube_object import CUBE_POSITION, PocketCubeObject
from .rendering import PocketRenderContext

SCRAMBLE_LENGTH = 20
SUCCESS_HOLD_STEPS = 5
MAX_LINEAR_SPEED = 0.02
MAX_ANGULAR_SPEED = 0.15
MIN_FINAL_HEIGHT = 0.82
# Fixed world cameras: target, distance, azimuth, elevation, vertical field of view.
CAMERAS = {
    "workspace": ([0, 0.08, 0.86], 2.3, 90, -50, 55),
    "cube_left": (CUBE_POSITION, 0.85, 115, -35, 20),
    "cube_right": (CUBE_POSITION, 0.85, -45, -30, 20),
}


# Seed-5 IK calibration for opposing layer grasps, with the same Panda models.
APPROACH_QPOS = np.array([
    [1.5886896517068647, .3281416940855087, -2.1516350617301025, -2.8959195253244547,
     2.5630487785803777, 1.7835423347273047, 2.8181655015059546],
    [1.7761638404091573, -1.6054946586469077, -1.8557445712143736, -2.9394635312988724,
     2.8557059139653327, 1.5593535849884115, -2.3332836397633936],
])


class PocketCube(TwoArmEnv):
    approach_qpos = APPROACH_QPOS

    def __init__(self, grip_profile="torsion", **kwargs):
        self.grip_profile = grip_profile
        kwargs.pop("robot_friendly", None)
        kwargs.setdefault("renderer", "mujoco")
        self.cube_xml = Path(__file__).with_name("assets") / "cube_3x3x3.xml"
        self._solved_steps = 0
        options = dict(
            robots=["Panda", "Panda"], base_types="NullBase", env_configuration="opposed",
            controller_configs=load_composite_controller_config(robot="Panda"),
            initialization_noise=None, control_freq=20, horizon=100_000, ignore_done=True,
            has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
            camera_names=list(CAMERAS), camera_widths=512, camera_heights=512,
            hard_reset=True, seed=0,
        )
        options.update(kwargs)
        super().__init__(**options)

    def _load_model(self):
        super()._load_model()
        arena = TableArena(table_full_size=(1.7, 1.12, 0.07), table_offset=(0, 0, 0.75))
        for robot, x, angle in zip(self.robots, (-0.52, 0.52), (0, np.pi)):
            robot.robot_model.set_base_xpos(np.array([x, 0.18, 0.775]))
            robot.robot_model.set_base_ori(np.array([0, 0, angle]))
            ET.SubElement(arena.worldbody, "geom", type="cylinder", pos=f"{x} 0.18 0.7625",
                          size="0.105 0.0125", rgba="0.18 0.22 0.26 1", group="1")
        for name, kind, pos, size, color in (
            ("mat", "box", "0 0 0.753", "0.3 0.3 0.003", "0.09 0.14 0.18 1"),
            ("pedestal", "cylinder", "0 0 0.87775", "0.021 0.12175", "0.26 0.3 0.34 1"),
            ("pedestal_pad", "box", "0 0 1.0055", "0.024 0.024 0.006", "0.08 0.09 0.1 1"),
        ):
            ET.SubElement(arena.worldbody, "geom", name=name, type=kind, pos=pos, size=size,
                          rgba=color, group="1", friction="1 0.005 0.0001")
        for name, (target, distance, azimuth, elevation, fovy) in CAMERAS.items():
            azimuth, elevation = np.deg2rad([azimuth, elevation])
            z = np.array([np.cos(azimuth)*np.cos(elevation), np.sin(azimuth)*np.cos(elevation), -np.sin(elevation)])
            x = np.cross([0, 0, 1], z)
            x /= np.linalg.norm(x)
            quat = Rotation.from_matrix(np.column_stack([x, np.cross(z, x), z])).as_quat(scalar_first=True)
            arena.set_camera(name, pos=np.array(target) + distance*z, quat=quat,
                             camera_attribs={"fovy": str(fovy)})
        self.cube = PocketCubeObject(self.cube_xml)
        self.model = ManipulationTask(arena, [r.robot_model for r in self.robots], [self.cube])
        self.model.root.find("option").attrib.update(
            timestep="0.002", integrator="implicitfast", solver="Newton", iterations="100", ls_iterations="50")
        contacts.reinforce(self.model.root)
        contacts.add_handles(self.model.root)
        contacts.fork_support(self.model.worldbody)
        geometry.convert(self.model.root, self.cube)
        geometry.configure_grippers(self.model.root, self.grip_profile)

    def _reset_internal(self):
        self._solved_steps = 0
        if self.has_offscreen_renderer and self.sim._render_context_offscreen is None:
            PocketRenderContext(self.sim, device_id=self.render_gpu_device_id)
        super()._reset_internal()
        model, data = self.sim.model._model, self.sim.data._data
        for robot, q in zip(self.robots, self.approach_qpos):
            robot.init_qpos = q.copy()
            robot.reset(deterministic=False, rng=self.rng)
        self._cubie_ids = [model.body(f"cube_pocket_corner_{i}").id for i in range(8)]
        self._scramble = metrics.scramble(int(self.rng.integers(2**31)), SCRAMBLE_LENGTH)
        rotations = metrics.apply_moves(np.tile(np.eye(3), (8, 1, 1)), self._scramble)
        for body, rotation in zip(self._cubie_ids, rotations):
            adr = model.jnt_qposadr[model.body_jntadr[body]]
            data.qpos[adr:adr+4] = Rotation.from_matrix(rotation).as_quat(scalar_first=True)
        self.sim.forward()
        state = self.cube_state()
        if not state["legal"] or state["solved"]:
            raise RuntimeError("Reset did not produce a legal unsolved pocket cube")

    def _destroy_sim(self):
        if self.sim is not None:
            context = self.sim._render_context_offscreen
            if isinstance(context, PocketRenderContext):
                context.close()
        super()._destroy_sim()

    def cube_state(self):
        data = self.sim.data._data
        anchor = data.xmat[self._cubie_ids[0]].reshape(3, 3)
        return metrics.inspect_cube(anchor.T @ data.xmat[self._cubie_ids].reshape(8, 3, 3))

    def _post_action(self, action):
        model, data = self.sim.model._model, self.sim.data._data
        joint = model.joint("cube_free")
        adr = joint.dofadr[0]
        velocity = data.qvel[adr:adr+6]
        state = self.cube_state()
        quiet = (np.linalg.norm(velocity[:3]) < MAX_LINEAR_SPEED
                 and np.linalg.norm(velocity[3:]) < MAX_ANGULAR_SPEED)
        for body in self._cubie_ids:
            j = model.body_jntadr[body]
            width = 3 if model.jnt_type[j] == 1 else 1
            start = model.jnt_dofadr[j]
            quiet = quiet and np.linalg.norm(data.qvel[start:start+width]) < MAX_ANGULAR_SPEED
        above_table = data.xpos[model.body("cube_core").id, 2] > MIN_FINAL_HEIGHT
        self._solved_steps = self._solved_steps + 1 if state["solved"] and quiet and above_table else 0
        return super()._post_action(action)

    def reward(self, action=None):
        return float(self._check_success())

    def _check_success(self):
        return self._solved_steps >= SUCCESS_HOLD_STEPS and self.cube_state()["solved"]

    def get_ep_meta(self):
        return {"lang": "Solve the 2x2 cube with the two fixed arms; finish with all six faces uniform and the cube still above the table."}
