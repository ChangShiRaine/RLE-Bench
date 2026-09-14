"""Private tabletop scenes with deterministic physical scoring.

PandaOmron uses the pinned 12-component mobile-base controller and three RGB
cameras. Scene state and scoring never cross the public control boundary.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np

from . import analytic

# --- shared geometry ------------------------------------------------------------
# Table sized for the arm and mobile-base envelope.
TABLE_FULL_SIZE = (0.7, 1.6, 0.05)
TABLE_FRICTION = (0.9, 0.005, 0.0001)
TABLE_OFFSET = (0.0, 0.0, 0.9)          # top surface z
ROBOT_BASE_X = -(TABLE_FULL_SIZE[0] / 2) - 0.27

# The RoboCasa agent-view mounts, verbatim from robocasa CAM_CONFIGS["DEFAULT"]
# (pinned ROBOCASA_SHA).
AGENTVIEW_CAMERAS = {
    "robot0_agentview_left": dict(
        pos="-0.5 0.35 1.05",
        quat="0.55623853 0.29935253 -0.37678665 -0.6775092",
        fovy="60",
    ),
    "robot0_agentview_right": dict(
        pos="-0.5 -0.35 1.05",
        quat="0.6775091886520386 0.3767866790294647 "
             "-0.2993525564670563 -0.55623859167099",
        fovy="60",
    ),
}
CAMERA_PARENT_BODY = "mobilebase0_support"

# Contact/joint params for authored primitives, matching the curated assets
# (robocasa's MJCFObject defaults): stiff damped contacts and a damped free
# joint. Without them a settled stack slowly pumps energy and topples.
PRIM_PHYS = dict(
    solref=[0.001, 1],
    solimp=[0.998, 0.998, 0.001],
    joints=[dict(type="free", damping="0.0005")],
)

# Joint prefixes frozen while scene physics settles (arm, gripper and base).
_ROBOT_JOINT_PREFIXES = ("robot0_", "gripper0_", "mobilebase0_")


# --- helpers (ported from the task03 v1 embodied scenes) --------------------------

def settle_objects(env, n_steps: int = 60):
    """Advance raw physics so free objects drop/settle, WITHOUT letting the
    torque-actuated robot sag: its joint state is frozen across the rollout."""
    joint_names = [j for j in env.sim.model.joint_names
                   if j.startswith(_ROBOT_JOINT_PREFIXES)]
    saved = {j: (env.sim.data.get_joint_qpos(j).copy(),
                 env.sim.data.get_joint_qvel(j).copy()) for j in joint_names}
    for _ in range(n_steps):
        env.sim.step()
        for j, (qp, qv) in saved.items():
            env.sim.data.set_joint_qpos(j, qp)
            env.sim.data.set_joint_qvel(j, qv)
    env.sim.forward()


def teleport_obj(env, name: str, pos, quat=None):
    obj = env.objects[name]
    joint = obj.joints[0]
    cur = env.sim.data.get_joint_qpos(joint).copy()
    cur[:3] = pos
    if quat is not None:
        cur[3:7] = quat
    env.sim.data.set_joint_qpos(joint, cur)
    env.sim.data.set_joint_qvel(joint, np.zeros(6))
    env.sim.forward()


def obj_pos(env, name: str) -> np.ndarray:
    return env.sim.data.body_xpos[env.obj_body_id[name]].copy()


def _collision_gids(model, body_ids) -> set[int]:
    return {g for g in range(model.ngeom)
            if model.geom_bodyid[g] in body_ids
            and (model.geom_contype[g] or model.geom_conaffinity[g])}


def _geom_world_top(model, data, gid) -> float:
    c, s = model.geom_aabb[gid][:3], model.geom_aabb[gid][3:]
    R = data.geom_xmat[gid].reshape(3, 3)
    ctr = data.geom_xpos[gid] + R @ c
    return float(ctr[2] + (np.abs(R) @ s)[2])


class TabletopScene:
    """Mixin-style base; the concrete base class is built lazily in `_scene_base()`
    so this module imports without robosuite (build tooling reads TASKS)."""


def _scene_base():
    from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
    from robosuite.models.arenas import TableArena
    from robosuite.models.tasks import ManipulationTask
    from robosuite.utils.mjcf_utils import find_elements
    from robosuite.utils.observables import Observable, sensor
    from robosuite.utils.transform_utils import convert_quat

    class _TabletopScene(ManipulationEnv, TabletopScene):

        LANG = ""

        def __init__(self, robots="PandaOmron", seed=None, **kwargs):
            self.objects = {}
            self.fixtures = {}
            self.obj_body_id = {}
            kwargs.setdefault("base_types", "default")
            kwargs.setdefault("gripper_types", "default")
            kwargs.setdefault("initialization_noise", None)
            super().__init__(robots=robots, seed=seed, **kwargs)

        # -- model assembly -------------------------------------------------
        def _load_model(self):
            super()._load_model()
            self.robots[0].robot_model.set_base_xpos([ROBOT_BASE_X, 0, 0])

            arena = TableArena(table_full_size=TABLE_FULL_SIZE,
                               table_friction=TABLE_FRICTION,
                               table_offset=TABLE_OFFSET)
            arena.set_origin([0, 0, 0])

            self.fixtures = {"table": SimpleNamespace(
                pos=np.array(TABLE_OFFSET), quat=np.array([1.0, 0, 0, 0]),
                size=np.array(TABLE_FULL_SIZE), root_body=None)}
            self._extend_arena(arena)

            self.objects = self._build_objects()
            self.model = ManipulationTask(
                mujoco_arena=arena,
                mujoco_robots=[robot.robot_model for robot in self.robots],
                mujoco_objects=list(self.objects.values()),
            )
            self._inject_agentview_cameras(find_elements)

        def _inject_agentview_cameras(self, find_elements):
            parent = find_elements(root=self.model.worldbody, tags="body",
                                   attribs={"name": CAMERA_PARENT_BODY})
            if parent is None:
                raise RuntimeError(
                    f"no {CAMERA_PARENT_BODY!r} body; agent-view extrinsics in "
                    f"the scene requires the PandaOmron mount")
            for name, cfg in AGENTVIEW_CAMERAS.items():
                if find_elements(root=parent, tags="camera",
                                 attribs={"name": name}) is None:
                    parent.append(ET.Element("camera", attrib=dict(
                        name=name, mode="fixed", **cfg)))

        # scenes override: append zones/fixtures to the arena, register fixtures
        def _extend_arena(self, arena) -> None:
            pass

        # scenes override: {name: MujocoObject}, sizes drawn from self.rng
        def _build_objects(self) -> dict:
            return {}

        # scenes override: {name: (pos, quat_wxyz)} drawn from self.rng
        def _place_objects(self) -> dict:
            return {}

        def _add_mat(self, arena, name: str, pos, half, rgba) -> None:
            """A colored static zone geom on the tabletop, and its fixture entry."""
            arena.worldbody.append(ET.Element("geom", attrib=dict(
                name=name, type="box", group="1",
                pos=f"{pos[0]} {pos[1]} {TABLE_OFFSET[2] + half[2]}",
                size=f"{half[0]} {half[1]} {half[2]}",
                rgba=" ".join(str(c) for c in rgba),
                friction="0.9 0.005 0.0001")))
            self.fixtures[name] = SimpleNamespace(
                pos=np.array([pos[0], pos[1], TABLE_OFFSET[2]]),
                quat=np.array([1.0, 0, 0, 0]),
                size=np.array([half[0] * 2, half[1] * 2, half[2] * 2]),
                root_body=None)

        # -- references and observables ---------------------------------------
        def _setup_references(self):
            super()._setup_references()
            self.obj_body_id = {name: self.sim.model.body_name2id(obj.root_body)
                                for name, obj in self.objects.items()}

        def _setup_observables(self):
            observables = super()._setup_observables()

            # Per-object world poses, robocasa-style keys. The daemon's
            # robot-filter strips them from the agent view at every level; at
            # Private recorder and scoring retain object state.
            def _make(name, body_id):
                @sensor(modality="object")
                def pose_pos(obs_cache, _b=body_id):
                    return np.array(self.sim.data.body_xpos[_b])

                @sensor(modality="object")
                def pose_quat(obs_cache, _b=body_id):
                    return convert_quat(
                        np.array(self.sim.data.body_xquat[_b]), to="xyzw")

                pose_pos.__name__ = f"{name}_pos"
                pose_quat.__name__ = f"{name}_quat"
                return pose_pos, pose_quat

            for name, obj in self.objects.items():
                for s in _make(name, self.sim.model.body_name2id(obj.root_body)):
                    observables[s.__name__] = Observable(
                        name=s.__name__, sensor=s,
                        sampling_rate=self.control_freq)
            return observables

        # -- reset ------------------------------------------------------------
        def _reset_internal(self):
            super()._reset_internal()
            if self.deterministic_reset:
                return
            for name, (pos, quat) in self._place_objects().items():
                joint = self.objects[name].joints[0]
                self.sim.data.set_joint_qpos(
                    joint, np.concatenate([np.asarray(pos, dtype=float),
                                           np.asarray(quat, dtype=float)]))
            self.sim.forward()
            settle_objects(self, 80)
            self._after_settle()

        def _after_settle(self) -> None:
            pass

        # -- harness contract ---------------------------------------------------
        def reward(self, action=None):
            return float(self._check_success())

        def _check_success(self):
            return False

        def _trial_score(self) -> float:
            return float(self._check_success())

        def get_ep_meta(self) -> dict:
            return {"lang": self.LANG}

        def _post_action(self, action):
            ret = super()._post_action(action)
            self._monitor_step()
            return ret

        def _monitor_step(self) -> None:
            pass

        # -- scoring helpers ------------------------------------------------------
        @property
        def _mj(self):
            return self.sim.model._model, self.sim.data._data

        def _robot_body_ids(self) -> set[int]:
            m = self.sim.model._model
            out = set()
            for b in range(m.nbody):
                name = self.sim.model.body_id2name(b) or ""
                if name.startswith(_ROBOT_JOINT_PREFIXES):
                    out.add(b)
            return out

        def _table_gids(self) -> set[int]:
            m = self.sim.model._model
            out = set()
            for g in range(m.ngeom):
                name = self.sim.model.geom_id2name(g) or ""
                if name.startswith("table"):
                    out.add(g)
            return out

        def _supported_pieces(self, piece_names) -> set[str]:
            """Pieces whose support chain reaches the table through pieces only.

            A piece in contact with any robot geom is excluded outright: a block
            held aloft (or propped against the arm) is not part of a tower, so
            the arm must be retracted before the trial is scored.
            """
            m, d = self._mj
            piece_gids = {name: _collision_gids(
                m, {self.obj_body_id[name]}) for name in piece_names}
            gid_owner = {g: n for n, gids in piece_gids.items() for g in gids}
            table = self._table_gids()
            robot_gids = _collision_gids(m, self._robot_body_ids())

            on_table, adj, touching_robot = set(), {}, set()
            for i in range(d.ncon):
                g1, g2 = d.contact[i].geom1, d.contact[i].geom2
                n1, n2 = gid_owner.get(g1), gid_owner.get(g2)
                if n1 and g2 in robot_gids or n2 and g1 in robot_gids:
                    touching_robot.add(n1 or n2)
                if n1 and g2 in table:
                    on_table.add(n1)
                if n2 and g1 in table:
                    on_table.add(n2)
                if n1 and n2 and n1 != n2:
                    adj.setdefault(n1, set()).add(n2)
                    adj.setdefault(n2, set()).add(n1)

            frontier = list(on_table - touching_robot)
            supported = set(frontier)
            while frontier:
                cur = frontier.pop()
                for nxt in adj.get(cur, ()):
                    if nxt not in supported and nxt not in touching_robot:
                        supported.add(nxt)
                        frontier.append(nxt)
            return supported

        def _piece_top(self, name: str) -> float:
            m, d = self._mj
            gids = _collision_gids(m, {self.obj_body_id[name]})
            return max(_geom_world_top(m, d, g) for g in gids)

        def on_zone(self, obj_name: str, zone: str, xy_tol: float,
                    z_tol: float = 0.18) -> bool:
            zp = self.fixtures[zone].pos
            op = obj_pos(self, obj_name)
            return (np.linalg.norm(op[:2] - zp[:2]) < xy_tol
                    and abs(op[2] - zp[2]) < z_tol)

    return _TabletopScene


# =================================================================================
# 1. TowerMaxHeight
# =================================================================================

def _tower_cls(Base):
    from robosuite.models.objects import (BallObject, BoxObject, CapsuleObject,
                                          CylinderObject)

    class TowerMaxHeight(Base):
        """12 mixed pieces; score = settled tower height / analytic optimum."""

        LANG = ("Stack pieces into the tallest stable tower on the table. Any "
                "subset of the pieces may be used. The score is the settled "
                "height of the tower at submission; "
                "finish submits the tower as it stands, so "
                "retract the arm first -- a piece touching the robot does not "
                "count.")

        SETTLE_STEPS = 2500        # ~5 s at the 2 ms physics timestep
        SIZE_JITTER = 0.10         # +/- fraction, per seed

        # (name, kind, nominal dims). box/plate: half-extents xyz; cylinder/rod:
        # (radius, half-height); sphere: (radius,); capsule: (radius, half-len).
        PIECES = (
            ("cube_a", "box", (0.032, 0.032, 0.032)),
            ("cube_b", "box", (0.026, 0.026, 0.026)),
            ("cube_c", "box", (0.021, 0.021, 0.021)),
            ("cyl_a", "cylinder", (0.026, 0.041)),
            ("cyl_b", "cylinder", (0.030, 0.030)),
            ("rod", "rod", (0.011, 0.060)),
            ("plate_a", "plate", (0.035, 0.035, 0.010)),
            ("plate_b", "plate", (0.030, 0.030, 0.008)),
            ("sphere", "sphere", (0.030,)),
            ("capsule", "capsule", (0.018, 0.030)),
            ("wedge_a", "wedge", (0.045, 0.040, 0.050)),
            ("wedge_b", "wedge", (0.040, 0.035, 0.045)),
        )
        COLORS = {
            "cube_a": (0.85, 0.33, 0.28, 1), "cube_b": (0.92, 0.60, 0.20, 1),
            "cube_c": (0.80, 0.50, 0.35, 1), "cyl_a": (0.30, 0.55, 0.85, 1),
            "cyl_b": (0.35, 0.70, 0.90, 1), "rod": (0.20, 0.45, 0.70, 1),
            "plate_a": (0.55, 0.80, 0.45, 1), "plate_b": (0.65, 0.85, 0.55, 1),
            "sphere": (0.80, 0.45, 0.80, 1), "capsule": (0.85, 0.55, 0.75, 1),
            "wedge_a": (0.95, 0.85, 0.30, 1), "wedge_b": (0.90, 0.78, 0.25, 1),
        }

        def _build_objects(self):
            jit = lambda: 1.0 + self.rng.uniform(-self.SIZE_JITTER,  # noqa: E731
                                                 self.SIZE_JITTER)
            objects, heights = {}, []
            for name, kind, dims in self.PIECES:
                s = jit()
                dims = tuple(d * s for d in dims)
                rgba = self.COLORS[name]
                if kind in ("box", "plate"):
                    obj = BoxObject(name=name, size=list(dims), rgba=list(rgba),
                                    friction=[0.95, 0.3, 0.1], **PRIM_PHYS)
                    height = 2 * dims[2]
                elif kind in ("cylinder", "rod"):
                    obj = CylinderObject(name=name, size=list(dims),
                                         rgba=list(rgba),
                                         friction=[0.95, 0.3, 0.1], **PRIM_PHYS)
                    height = 2 * dims[1]
                elif kind == "sphere":
                    obj = BallObject(name=name, size=list(dims), rgba=list(rgba),
                                     friction=[0.95, 0.3, 0.1], **PRIM_PHYS)
                    height = 2 * dims[0]
                elif kind == "capsule":
                    obj = CapsuleObject(name=name, size=list(dims),
                                        rgba=list(rgba),
                                        friction=[0.95, 0.3, 0.1], **PRIM_PHYS)
                    height = 2 * (dims[0] + dims[1])
                elif kind == "wedge":
                    obj = _wedge_object(name, *dims, rgba=rgba)
                    height = dims[2]
                objects[name] = obj
                heights.append(analytic.Piece(kind=kind, height=height))
            self._optimum = analytic.tower_optimum(heights)
            return objects

        def _place_objects(self):
            placements = {}
            spots = _scatter_grid(self.rng, len(self.PIECES),
                                  x_range=(-0.30, 0.08), y_range=(-0.60, 0.10),
                                  min_dist=0.13)
            for (name, kind, _dims), (x, y) in zip(self.PIECES, spots):
                placements[name] = ([x, y, TABLE_OFFSET[2] + 0.08],
                                    [1.0, 0, 0, 0])
            return placements

        def _trial_score(self):
            settle_objects(self, self.SETTLE_STEPS)
            names = [name for name, *_ in self.PIECES]
            supported = self._supported_pieces(names)
            if not supported:
                return 0.0
            top = max(self._piece_top(n) for n in supported)
            height = top - TABLE_OFFSET[2]
            return float(np.clip(height / self._optimum, 0.0, 1.0))

    return TowerMaxHeight


def _wedge_object(name: str, hx: float, hy: float, hz: float, rgba):
    """A right-triangular-prism piece: flat base, sloped top (never stackable-on)."""
    import os
    import tempfile

    from robosuite.models.objects import MujocoXMLObject

    lx, ly, h = 2 * hx, 2 * hy, hz
    verts = (f"0 0 0  {lx} 0 0  {lx} {ly} 0  0 {ly} 0  "
             f"0 0 {h}  0 {ly} {h}")
    color = " ".join(str(c) for c in rgba)
    xml = f"""<mujoco model="{name}">
  <asset><mesh name="wedge_mesh" vertex="{verts}"/></asset>
  <worldbody>
    <body>
      <body name="object">
        <geom name="wedge" type="mesh" mesh="wedge_mesh" pos="{-hx} {-hy} 0"
              rgba="{color}" group="0" density="300"
              friction="0.9 0.005 0.0001"/>
        <site name="bottom_site" pos="0 0 0" rgba="0 0 0 0" size="0.005"/>
        <site name="top_site" pos="0 0 {h}" rgba="0 0 0 0" size="0.005"/>
        <site name="horizontal_radius_site" pos="{hx} {hy} 0" rgba="0 0 0 0"
              size="0.005"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""
    fd, path = tempfile.mkstemp(suffix=".xml")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(xml)
        return MujocoXMLObject(fname=path, name=name,
                               joints=[dict(type="free", damping="0.0005")],
                               obj_type="all", duplicate_collision_geoms=False)
    finally:
        os.remove(path)


def _scatter_grid(rng, n, x_range, y_range, min_dist):
    """n table spots with pairwise separation, deterministic in rng."""
    spots = []
    for _ in range(n):
        for _attempt in range(200):
            x = rng.uniform(*x_range)
            y = rng.uniform(*y_range)
            if all((x - sx) ** 2 + (y - sy) ** 2 >= min_dist ** 2
                   for sx, sy in spots):
                break
        spots.append((x, y))
    return spots


# =================================================================================
# 2. CantileverOverhang
# =================================================================================

def _cantilever_cls(Base):
    from robosuite.models.objects import BoxObject

    class CantileverOverhang(Base):
        """Identical blocks past any table edge; exact harmonic regret."""

        LANG = ("Stack the identical blocks so they extend past any table "
                "edge as "
                "far as possible without falling. The score is the "
                "settled horizontal overhang beyond the edge when the trial "
                "ends; finish submits the "
                "stack as it stands, so retract the arm first -- a block "
                "touching the robot does not count, and fallen blocks count "
                "nothing.")

        N_BLOCKS = 4
        # long axis along y, toward the -y side edge: 12 x 6 x 3 cm
        BLOCK_HALF = (0.030, 0.060, 0.015)
        SETTLE_STEPS = 2500

        def _build_objects(self):
            self._optimum = analytic.harmonic_overhang(
                self.N_BLOCKS, 2 * self.BLOCK_HALF[1])
            return {f"block_{i}": BoxObject(
                        name=f"block_{i}", size=list(self.BLOCK_HALF),
                        rgba=[0.85, 0.35, 0.25, 1],
                        friction=[0.8, 0.3, 0.1], **PRIM_PHYS)
                    for i in range(self.N_BLOCKS)}

        def _place_objects(self):
            placements = {}
            spots = _scatter_grid(self.rng, self.N_BLOCKS,
                                  x_range=(-0.28, 0.05), y_range=(-0.55, -0.10),
                                  min_dist=0.16)
            for i, (x, y) in enumerate(spots):
                placements[f"block_{i}"] = (
                    [x, y, TABLE_OFFSET[2] + self.BLOCK_HALF[2] + 0.002],
                    [1.0, 0, 0, 0])
            return placements

        @property
        def _edge_y(self) -> float:
            return -TABLE_FULL_SIZE[1] / 2

        def _trial_score(self):
            settle_objects(self, self.SETTLE_STEPS)
            names = [f"block_{i}" for i in range(self.N_BLOCKS)]
            m, d = self._mj
            table_top = TABLE_OFFSET[2]
            supported = self._supported_pieces(names)
            best = 0.0
            for name in supported:
                gids = _collision_gids(m, {self.obj_body_id[name]})
                # fallen: entire block below the tabletop plane
                if all(_geom_world_top(m, d, g) < table_top - 0.01
                       for g in gids):
                    continue
                far = 0.0
                for g in gids:
                    c, s = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
                    R = d.geom_xmat[g].reshape(3, 3)
                    ctr = d.geom_xpos[g] + R @ c
                    extent = (np.abs(R) @ s)[:2]
                    overhang = (np.abs(ctr[:2] - np.asarray(TABLE_OFFSET[:2]))
                                + extent - np.asarray(TABLE_FULL_SIZE[:2]) / 2)
                    far = max(far, float(np.max(overhang)))
                best = max(best, far)
            return float(np.clip(best / self._optimum, 0.0, 1.0))

    return CantileverOverhang


# =================================================================================
# 3. BalanceCoins
# =================================================================================

def _balance_cls(Base):
    from robosuite.models.objects import BoxObject

    class BalanceCoins(Base):
        """12 identical cubes, one heavier; find it with the two-pan balance."""

        LANG = ("Exactly one of the nine identical cubes is heavier than the "
                "rest. Find it using the balance scale and place it -- and only "
                "it -- on the green answer mat. Efficient use of the balance is "
                "scored.")

        N_COINS = 9
        COIN_HALF = 0.016
        COIN_DENSITY = 1800.0      # ~59 g per cube
        HEAVY_RATIO = 1.2          # the beam resolves 4v4; push-probing does not
        BALANCE_SCALE = 1.25
        BALANCE_POS = (0.10, 0.42)  # clear the initial gripper and stay inside the table
        MAT_POS = (-0.02, -0.55)
        MAT_HALF = (0.07, 0.07, 0.002)
        MAT_XY_TOL = 0.09
        # weighing counter: both pans loaded for >=0.5 s counts one weighing
        MIN_ON_STEPS = 10          # control steps at 20 Hz
        MIN_OFF_STEPS = 10

        def _extend_arena(self, arena):
            self._add_mat(arena, "answer_mat", self.MAT_POS, self.MAT_HALF,
                          (0.30, 0.75, 0.40, 1))
            arena.worldbody.append(self._balance_body())
            self.fixtures["balance"] = SimpleNamespace(
                pos=np.array([*self.BALANCE_POS, TABLE_OFFSET[2]]),
                quat=np.array([1.0, 0, 0, 0]),
                size=np.array([0.12, 0.42, 0.21]) * self.BALANCE_SCALE,
                root_body="balance_beam")

        def _balance_body(self) -> ET.Element:
            """Static post; hinged beam with rigid pans, centre of mass below the
            pivot so the empty beam rests level and tips under load asymmetry."""
            x, y = self.BALANCE_POS
            z = TABLE_OFFSET[2]
            body = ET.fromstring(f"""
  <body name="balance" pos="{x} {y} {z}">
    <geom name="balance_post" type="cylinder" size="0.015 0.085" pos="0 0 0.085"
          rgba="0.45 0.45 0.50 1"/>
    <geom name="balance_base" type="cylinder" size="0.07 0.008" pos="0 0 0.008"
          rgba="0.45 0.45 0.50 1"/>
    <body name="balance_beam" pos="0 0 0.21">
      <joint name="balance_hinge" type="hinge" axis="1 0 0" range="-0.10 0.10"
             damping="0.25"/>
      <geom name="balance_arm" type="box" size="0.012 0.10 0.006"
            pos="0 0 -0.024" rgba="0.75 0.62 0.25 1"/>
      <geom name="balance_bob" type="sphere" size="0.022" pos="0 0 -0.062"
            density="12000" rgba="0.55 0.45 0.20 1"
            contype="0" conaffinity="0"/>
      <geom name="balance_arm_l" type="box" size="0.010 0.045 0.004"
            pos="0 -0.16 -0.008" rgba="0.75 0.62 0.25 1"
            contype="0" conaffinity="0"/>
      <geom name="balance_arm_r" type="box" size="0.010 0.045 0.004"
            pos="0 0.16 -0.008" rgba="0.75 0.62 0.25 1"
            contype="0" conaffinity="0"/>
      <geom name="balance_hanger" type="box" size="0.010 0.004 0.014"
            pos="0 0 -0.011" rgba="0.75 0.62 0.25 1"
            contype="0" conaffinity="0"/>
      <body name="balance_pan_l_body" pos="0 -0.20 0">
        <joint name="balance_pan_l_hinge" type="hinge" axis="1 0 0"
               damping="0.05"/>
        <geom name="balance_pan_l" type="cylinder" size="0.090 0.004"
              pos="0 0 -0.036" rgba="0.82 0.72 0.35 1"
              density="8000" friction="1.0 0.3 0.1"/>
      </body>
      <body name="balance_pan_r_body" pos="0 0.20 0">
        <joint name="balance_pan_r_hinge" type="hinge" axis="1 0 0"
               damping="0.05"/>
        <geom name="balance_pan_r" type="cylinder" size="0.090 0.004"
              pos="0 0 -0.036" rgba="0.82 0.72 0.35 1"
              density="8000" friction="1.0 0.3 0.1"/>
      </body>
    </body>
  </body>""")
            # PANS HANG from the beam ends (like a real balance), so the tilt
            # reads the MASS difference alone -- rigid pans made the lever arm
            # depend on where each coin sat, a standing bias with equal loads.
            # Low rims, wide enough that a coin at a 2x2 slot keeps ~2 cm of
            # clearance for the finger pads coming down around it.
            for side in ("l", "r"):
                pan = body.find(f".//body[@name='balance_pan_{side}_body']")
                for k in range(14):
                    a = 2 * np.pi * k / 14
                    pan.append(ET.Element("geom", attrib=dict(
                        name=f"balance_rim_{side}{k}", type="box",
                        size="0.004 0.022 0.009",
                        pos=f"{0.087 * np.cos(a):.4f} "
                            f"{0.087 * np.sin(a):.4f} -0.029",
                        euler=f"0 0 {a + np.pi / 2:.4f}",
                        rgba="0.82 0.72 0.35 1",
                        friction="1.0 0.3 0.1")))
            for element in body.iter():
                if element is body:
                    continue
                if element.tag == "geom":
                    # Preserve fixture masses so enlargement does not damp the weight signal.
                    density = float(element.get("density", "1000")) / self.BALANCE_SCALE ** 3
                    element.set("density", str(density))
                for attribute in ("pos", "size"):
                    if attribute in element.attrib:
                        values = np.fromstring(element.get(attribute), sep=" ")
                        element.set(attribute, " ".join(f"{v:.8g}" for v in values * self.BALANCE_SCALE))
            for parent in list(body.iter()):
                for geom in list(parent.findall("geom")):
                    visual = dict(geom.attrib, name=geom.get("name") + "_visual",
                                  group="1", contype="0", conaffinity="0")
                    parent.append(ET.Element("geom", attrib=visual))
            return body

        def _build_objects(self):
            self._heavy = int(self.rng.integers(self.N_COINS))
            objects = {}
            for i in range(self.N_COINS):
                density = self.COIN_DENSITY * (
                    self.HEAVY_RATIO if i == self._heavy else 1.0)
                objects[f"coin_{i}"] = BoxObject(
                    name=f"coin_{i}", size=[self.COIN_HALF] * 3,
                    rgba=[0.60, 0.60, 0.65, 1], density=density,
                    friction=[1.0, 0.3, 0.1], **PRIM_PHYS)
            return objects

        def _place_objects(self):
            placements = {}
            spots = _scatter_grid(self.rng, self.N_COINS,
                                  x_range=(-0.30, -0.02), y_range=(-0.40, 0.12),
                                  min_dist=0.075)
            for i, (x, y) in enumerate(spots):
                placements[f"coin_{i}"] = (
                    [x, y, TABLE_OFFSET[2] + self.COIN_HALF + 0.002],
                    [1.0, 0, 0, 0])
            return placements

        PAN_REGION_R = 0.10 * BALANCE_SCALE      # xy radius counting a coin as "on this pan"
        PAN_REGION_DZ = 0.08

        def _after_settle(self):
            self._pan_geom = {
                side: self.sim.model.geom_name2id(f"balance_pan_{side}")
                for side in ("l", "r")}
            self._weighings = analytic.WeighingCounter(
                min_on=self.MIN_ON_STEPS, min_off=self.MIN_OFF_STEPS)

        def position_observation(self):
            """Authorized world positions; masses and answer identity remain private."""
            m, d = self._mj
            return {
                "cube_positions": {name: obj_pos(self, name).tolist() for name in self.objects},
                "pan_positions": {
                    label: (d.geom_xpos[gid] + d.geom_xmat[gid].reshape(3, 3)
                            @ np.array([0, 0, m.geom_size[gid, 1]])).tolist()
                    for side, label in (("l", "left"), ("r", "right"))
                    for gid in (self._pan_geom[side],)
                },
            }

        def _pan_sets(self):
            """Which coins rest on each pan, geometrically -- stable against
            contact flicker, which raw contact reads are not."""
            d = self.sim.data._data
            out = {}
            for side, gid in self._pan_geom.items():
                pan = d.geom_xpos[gid]
                coins = frozenset(
                    i for i in range(self.N_COINS)
                    if np.linalg.norm(
                        obj_pos(self, f"coin_{i}")[:2] - pan[:2])
                    < self.PAN_REGION_R
                    and 0 < obj_pos(self, f"coin_{i}")[2] - pan[2]
                    < self.PAN_REGION_DZ)
                out[side] = coins
            return out

        def _monitor_step(self):
            if not hasattr(self, "_weighings"):
                return
            sets = self._pan_sets()
            # only EQUAL non-empty counts are a comparison: unequal pans tip by
            # count alone and say nothing about mass, so loading and unloading
            # pass through for free and every informative configuration counts
            loaded = bool(sets["l"]) and len(sets["l"]) == len(sets["r"])
            self._weighings.update(loaded, config=(sets["l"], sets["r"]))

        def _coin_on_mat(self, i: int) -> bool:
            return self.on_zone(f"coin_{i}", "answer_mat", self.MAT_XY_TOL,
                                z_tol=0.10)

        def _check_success(self):
            on_mat = [i for i in range(self.N_COINS) if self._coin_on_mat(i)]
            return on_mat == [self._heavy]

        def _trial_score(self):
            if not self._check_success():
                return 0.0
            optimum = analytic.weighing_optimum(self.N_COINS)
            return float(analytic.weighing_efficiency(
                self._weighings.count, optimum))

    return BalanceCoins


_SCENE_FACTORIES = {
    "TowerMaxHeight": _tower_cls,
    "CantileverOverhang": _cantilever_cls,
    "BalanceCoins": _balance_cls,
}
_CLASSES: dict[str, type] = {}


def scene_class(task: str) -> type:
    if task not in _SCENE_FACTORIES:
        raise KeyError(f"unknown tabletop task {task!r}")
    if task not in _CLASSES:
        _CLASSES[task] = _SCENE_FACTORIES[task](_scene_base())
    return _CLASSES[task]
