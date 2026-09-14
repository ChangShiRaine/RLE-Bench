"""Private rigid-body scene for the hidden center-of-mass task."""
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask

from .config import CASES, CONTROL_HZ, MAX_STEPS

HALF = np.array([0.075, 0.075, 0.025])
# Rigid centered bridge: two posts and a narrow bar, all dimensions in meters.
HANDLE_PARTS = (
    ("handle_left", .006, (.006, .008, .025), (-.034, 0, .050)),
    ("handle_right", .006, (.006, .008, .025), (.034, 0, .050)),
    ("handle_bar", .018, (.040, .008, .006), (0, 0, .081)),
)
SIGNS = {"A": (-1, 1), "B": (1, 1), "C": (-1, -1), "D": (1, -1)}
CAMERAS = {
    "workspace": ([0, 0, 1.0], 1.95, -55, -40, 48),
    "closeup": ([0.03, 0, 0.79], 0.80, -70, -55, 42),
    "top": ([0.03, 0, 0.76], 0.95, -90, -90, 43),
}


# Box-local ink strokes: visual geometry with no contact or mass.
LETTER_PATHS = {
    "A": (((-.5, -1), (0, 1), (.5, -1)), ((-.3, -.2), (.3, -.2))),
    "B": (((-.5, -1), (-.5, 1), (.2, 1), (.5, .6), (.2, 0),
           (-.5, 0)), ((.2, 0), (.5, -.6), (.2, -1), (-.5, -1))),
    "C": (((.5, .8), (.2, 1), (-.3, 1), (-.5, .6), (-.5, -.6),
           (-.3, -1), (.2, -1), (.5, -.8)),),
    "D": (((-.5, -1), (-.5, 1), (0, 1), (.5, .5), (.5, -.5),
           (0, -1), (-.5, -1)),),
}


def add_labels(body):
    for letter, paths in LETTER_PATHS.items():
        center = np.array(SIGNS[letter]) * .039
        for path_index, path in enumerate(paths):
            for segment, (start, end) in enumerate(zip(path, path[1:])):
                endpoints = [(* (center + np.array(point) * .012), .0252)
                             for point in (start, end)]
                ET.SubElement(body, "geom", name=f"label_{letter}_{path_index}_{segment}",
                    type="capsule", size="0.0012", fromto=" ".join(str(v) for p in endpoints for v in p),
                    rgba="0.06 0.06 0.06 1", group="1", mass="0", contype="0", conaffinity="0")


def mass_properties(quadrant, case=CASES[0]):
    offset = np.r_[np.array(case.offset_xy) * SIGNS[quadrant], 0.0]
    components = [
        (case.shell_mass, HALF, np.zeros(3)),
        (case.ballast_mass, np.array(case.ballast_half), offset),
        *((mass, np.array(half), np.array(pos)) for _, mass, half, pos in HANDLE_PARTS),
    ]
    mass = sum(m for m, _, _ in components)
    center = sum(m * pos for m, _, pos in components) / mass
    inertia = np.zeros((3, 3))
    for component_mass, half, position in components:
        squares = np.square(half)
        inertia += np.diag(component_mass / 3 * (squares.sum() - squares))
        delta = position - center
        inertia += component_mass * (delta @ delta * np.eye(3) - np.outer(delta, delta))
    return mass, center, inertia


class HiddenCOM(ManipulationEnv):
    def __init__(self, quadrant="B", resolution=512, camera_observations=False, case=CASES[0]):
        self.quadrant = quadrant
        self.case = case
        super().__init__(robots="Panda", base_types="NullBase",
            controller_configs=load_composite_controller_config(robot="Panda"),
            initialization_noise=None, control_freq=CONTROL_HZ, horizon=MAX_STEPS, ignore_done=True, seed=7,
            has_renderer=False, has_offscreen_renderer=True, renderer="mujoco", hard_reset=False,
            use_camera_obs=camera_observations,
            camera_names=list(CAMERAS), camera_widths=resolution, camera_heights=resolution)

    def _load_model(self):
        super()._load_model()
        arena = TableArena(table_full_size=(1.2, 0.9, 0.05), table_offset=(0, 0, 0.75))
        self.robots[0].init_qpos = np.array([-0.65, -0.5, 0, -2.2, 0, 1.7, 0.8])
        self.robots[0].robot_model.set_base_xpos(np.array([-0.48, 0, 0.775]))
        for name, kind, pos, size, rgba in (
            ("robot_mount", "cylinder", "-0.48 0 0.7625", "0.10 0.0125", "0.2 0.23 0.27 1"),
            ("work_mat", "box", "0.06 0 0.752", "0.30 0.30 0.002", "0.12 0.19 0.23 1"),
        ):
            ET.SubElement(arena.worldbody, "geom", name=name, type=kind,
                pos=pos, size=size, rgba=rgba, group="1", friction="0.8 0.005 0.0001")
        for name, (target, distance, az, el, fovy) in CAMERAS.items():
            az, el = np.deg2rad([az, el])
            z = np.array([np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), -np.sin(el)])
            x = np.cross([0, 0, 1], z); x /= np.linalg.norm(x)
            quat = Rotation.from_matrix(np.column_stack([x, np.cross(z, x), z])).as_quat(scalar_first=True)
            arena.set_camera(name, pos=np.array(target)+distance*z, quat=quat,
                camera_attribs={"fovy": str(fovy)})
        self.box = BoxObject("sealed_box", size=HALF, rgba=[0.88, 0.62, 0.22, 1],
                             friction=[0.7, 0.005, 0.0001])
        for name, mass, half, pos in HANDLE_PARTS:
            ET.SubElement(self.box.get_obj(), "geom", name=name, type="box",
                size=" ".join(map(str, half)), pos=" ".join(map(str, pos)), mass=str(mass),
                rgba="0.65 0.69 0.73 1", group="1", friction="1.0 0.005 0.0001")
        add_labels(self.box.get_obj())
        mass, center, inertia = mass_properties(self.quadrant, self.case)
        ET.SubElement(self.box.get_obj(), "inertial", pos=" ".join(map(str, center)),
            mass=str(mass), fullinertia=" ".join(map(str, [*inertia.diagonal(), inertia[0,1], inertia[0,2], inertia[1,2]])))
        self.model = ManipulationTask(arena, [r.robot_model for r in self.robots], [self.box])
        self.model.root.find("compiler").set("inertiafromgeom", "auto")
        self.model.root.find("option").attrib.update(timestep="0.002", integrator="implicitfast",
            solver="Newton", iterations="100", ls_iterations="50")

    def _reset_internal(self):
        super()._reset_internal()
        self.sim.data.set_joint_qpos(self.box.joints[0], [0, 0, 0.779, 1, 0, 0, 0])
        self.sim.forward()

    def reward(self, action=None):
        return 0.0

    def _check_success(self):
        return False
