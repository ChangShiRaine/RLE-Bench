"""The harness levels: what each one lets the agent see.

Task01 varies ONE thing across L1/L2/L3 -- how much help the agent is given -- and holds
the task, the scoring and the evaluation plan fixed, so the levels stay comparable. The
tests here pin that: the observation boundary is exactly one function, L1 is unchanged,
and L3 widens it without opening a second channel.

L2 is deliberately absent from this file. Its skills run in the AGENT's process over the
same wire, so from the daemon's side an L2 run is indistinguishable from an L1 one --
which is the property that makes L2 honest, and it is asserted here as `test_l2_is_l1_on_the_wire`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harness import config as C

from test_speedrun_service import FakeEnv, Harness


class SceneEnv(FakeEnv):
    """A FakeEnv that also answers the questions an observation cannot.

    Mirrors the shape `privileged.py` reads: RoboCasa's scene manifest, the object and
    fixture registries, and robosuite's grasp predicate.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.object_cfgs = [
            {"name": "obj", "obj_groups": "vegetable",
             "placement": {"fixture": _Named("counter")}},
            {"name": "distr_counter", "obj_groups": "all"},
        ]
        self.objects = {"obj": _Body("obj_main"), "distr_counter": _Body("distr_main")}
        self.fixtures = {"drawer": _Fixture()}
        self.robots = [_Robot()]
        self.sim = _Sim()

    def _check_grasp(self, gripper, object_geoms):
        return object_geoms == ["obj_g"]

    def check_contact(self, geoms):
        return geoms == ["left_g"]


class _Named:
    def __init__(self, name):
        self.name = name


class _Body:
    def __init__(self, root):
        self.root_body = root
        self.contact_geoms = ["obj_g"] if root == "obj_main" else ["distr_g"]


class _Fixture:
    name = "drawer"
    pos = [1.0, 0.0, 0.5]
    quat = [1.0, 0.0, 0.0, 0.0]
    size = [0.4, 0.5, 0.3]
    root_body = "drawer_main"


class _Robot:
    def __init__(self):
        self.gripper = _Gripper()


class _Gripper:
    important_geoms = {"left_finger": ["left_g"], "right_finger": ["right_g"]}


class _Sim:
    """Just enough MuJoCo to answer a body pose lookup."""

    class model:
        _ids = {"obj_main": 0, "distr_main": 1, "drawer_main": 2}

        @classmethod
        def body_name2id(cls, name):
            return cls._ids[name]

    class data:
        body_xpos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        # Deliberately NOT the identity and NOT symmetric: wxyz and xyzw agree on
        # identity, so an identity quaternion cannot catch a convention mismatch.
        # wxyz (0, 1, 0, 0) -- a half turn about x -- is xyzw (1, 0, 0, 0).
        body_xquat = np.array([[0.0, 1.0, 0.0, 0.0]] * 3)


def _harness(tmp_path, level, monkeypatch, **kw):
    """A daemon built at `level`. The level is read in Service.__init__, so it has to be
    set before construction -- which is also true in production, where it comes from the
    image the container was built from."""
    monkeypatch.setenv("RLEBENCH_LEVEL", level)
    h = Harness(tmp_path, **kw)
    return h


# -- the level string itself ---------------------------------------------------

def test_level_defaults_to_the_control_condition(monkeypatch):
    monkeypatch.delenv("RLEBENCH_LEVEL", raising=False)
    assert C.level() == "L1"


def test_unknown_level_is_refused(monkeypatch):
    """Rather than defaulting. A run that quietly graded the wrong harness would be
    worse than one that failed to start."""
    monkeypatch.setenv("RLEBENCH_LEVEL", "L4")
    with pytest.raises(ValueError, match="RLEBENCH_LEVEL"):
        C.level()


def test_level_is_reported_to_the_agent(tmp_path, monkeypatch):
    h = _harness(tmp_path, "L3", monkeypatch)
    try:
        assert h.client.task_info()["level"] == "L3"
    finally:
        h.stop()


# -- L1: the control condition -------------------------------------------------

def test_l1_hides_every_object_pose(tmp_path, monkeypatch):
    h = _harness(tmp_path, "L1", monkeypatch)
    try:
        obs = h.client.reset()
        assert "drawer_obj_pos" not in obs
        assert "object-state" not in obs
        assert not [k for k in obs if k.startswith("priv_")]
        assert "robot0_eef_pos" in obs
    finally:
        h.stop()


def test_l2_is_l1_on_the_wire(tmp_path, monkeypatch):
    """A skill library changes nothing the daemon does. Skills run in the agent's own
    process over this same socket, so they can see exactly what a hand-written
    controller could -- which is what keeps an L2 score honest."""
    seen = {}
    for level in ("L1", "L2"):
        # A directory each: Harness names its ledger after tmp_path, and stop() seals
        # it, so a shared one would refuse the second daemon's start record.
        room = tmp_path / level
        room.mkdir()
        h = _harness(room, level, monkeypatch)
        try:
            seen[level] = sorted(h.client.reset())
        finally:
            h.stop()
    assert seen["L1"] == seen["L2"]


# -- L3: the widening ----------------------------------------------------------

def test_l3_adds_the_poses_l1_strips(tmp_path, monkeypatch):
    h = _harness(tmp_path, "L3", monkeypatch)
    try:
        obs = h.client.reset()
        # The proprioception an L1 agent gets is still there, unchanged.
        assert "robot0_eef_pos" in obs
        # What L1 dropped comes back -- under priv_, never under its original key, so
        # nothing downstream can confuse a granted pose for a perceived one.
        assert obs["priv_object_state"]["drawer_obj_pos"] == [1.0, 2.0, 3.0]
        assert "drawer_obj_pos" not in obs
        # The concatenation stays dropped: it duplicates the keys beside it in an order
        # the agent would have to reverse-engineer.
        assert "object-state" not in obs["priv_object_state"]
    finally:
        h.stop()


def test_l3_reports_target_objects_fixtures_and_grasp(tmp_path, monkeypatch):
    """The four families the level promises, off a live env."""
    h = _harness(tmp_path, "L3", monkeypatch, env_cls=SceneEnv)
    try:
        obs = h.client.reset()

        assert obs["priv_target"]["name"] == "obj"
        assert obs["priv_target"]["placed_on"] == "counter"

        # The FULL object list, not just the target: identifying which one matters is
        # still the agent's job at L3.
        assert set(obs["priv_objects"]) == {"obj", "distr_counter"}
        assert list(obs["priv_objects"]["obj"]["pos"]) == [1.0, 2.0, 3.0]

        # Extents, which a pose alone cannot give: a place target needs the box.
        assert obs["priv_fixtures"]["drawer"]["size"] == [0.4, 0.5, 0.3]

        assert obs["priv_grasp"]["grasped_objects"] == ["obj"]
        assert obs["priv_grasp"]["grasping"] is True
        assert obs["priv_grasp"]["left_finger_contact"] is True
        assert obs["priv_grasp"]["right_finger_contact"] is False
    finally:
        h.stop()


def test_l3_quaternions_use_the_same_convention_as_the_robot_keys(tmp_path, monkeypatch):
    """xyzw, like every `_quat` key the agent already has.

    MuJoCo stores wxyz and robosuite converts on the way out, so privileged state read
    straight off `sim.data.body_xquat` arrives in the OTHER order. Two conventions in one
    observation is a silent wrong rotation, not an error, which is why this is pinned.

    The fake body carries wxyz (0, 1, 0, 0), a half turn about x, which is xyzw
    (1, 0, 0, 0) -- asymmetric, so the two orders cannot be confused.
    """
    h = _harness(tmp_path, "L3", monkeypatch, env_cls=SceneEnv)
    try:
        obs = h.client.reset()
        assert list(obs["priv_objects"]["obj"]["quat"]) == [1.0, 0.0, 0.0, 0.0]
        assert list(obs["priv_fixtures"]["drawer"]["quat"]) == [1.0, 0.0, 0.0, 0.0]
    finally:
        h.stop()


def test_l3_state_comes_back_from_every_step(tmp_path, monkeypatch):
    """Not only `reset` and `observe`. The agent acts on what `step` hands back, so the
    privileged block has to arrive there too -- that is the path that would silently
    make L3 useless if it were missed."""
    h = _harness(tmp_path, "L3", monkeypatch, env_cls=SceneEnv)
    try:
        res = h.client.step([[0.0] * 12] * 2)
        assert "priv_objects" in res["obs"]
    finally:
        h.stop()


def test_l3_state_reaches_the_trial_descriptor_too(tmp_path, monkeypatch):
    """A descriptor CONTAINS an observation, so it crosses the same boundary.

    `trial_info` and evaluation `reset` are the two ops that return one, and both were
    filtered by a private call to `agent_visible_obs` that narrowed without ever
    widening -- so an L3 run answered them L1-shaped, and the agent's first look at
    every graded trial was worse than the level it was running at.
    """
    h = _harness(tmp_path, "L3", monkeypatch, env_cls=SceneEnv)
    try:
        h.open_evaluation()
        stepped = h.client.step([[0.0] * 12])["obs"]
        privileged = {k for k in stepped if k.startswith("priv_")}
        assert privileged, "no privileged block on the step path to compare against"
        assert {k for k in h.client.trial_info()["obs"]
                if k.startswith("priv_")} == privileged
        # Evaluation `reset` returns the NEXT trial's descriptor, by the same path.
        assert {k for k in h.client.reset()["trial"]["obs"]
                if k.startswith("priv_")} == privileged
    finally:
        h.stop()


def test_a_broken_env_degrades_to_l1_rather_than_failing(tmp_path, monkeypatch):
    """Every privileged section is optional. An env that cannot answer leaves the agent
    where L1 would have -- which is a worse harness, not a lost trial."""
    from harness.privileged import privileged_state

    obs = {"robot0_eef_pos": np.zeros(3), "drawer_obj_pos": np.ones(3)}
    out = privileged_state(obs, object())
    assert set(out) == {"priv_object_state"}
    assert privileged_state({"robot0_eef_pos": np.zeros(3)}, None) == {}


# -- the subtask table ---------------------------------------------------------

def test_subtasks_are_scoreable_and_uniquely_slugged():
    slugs = [C.subtask_slug(i, t) for i, (t, _) in enumerate(C.SUBTASKS, 1)]
    assert len(set(slugs)) == len(slugs)
    for task, description in C.SUBTASKS:
        assert C.is_scoreable(task), f"{task} has a degenerate success predicate"
        assert description and description[0].islower()


# -- L2: the skill library -----------------------------------------------------

def test_every_skill_is_a_plain_function_taking_sim_first():
    """A FUNCTION, not a class: the agent is free to organise its own code however it
    likes, and a library of base classes to inherit from would be exactly the constraint
    this task does not impose.

    `sim` first, because a skill drives the client exactly as the agent's own code does.
    """
    import inspect

    from harness import skills as S

    for name in ("reach", "move_eef", "grasp", "release", "lift", "move_base", "settle"):
        fn = getattr(S, name)
        assert inspect.isfunction(fn), f"{name} should be a plain function"
        assert list(inspect.signature(fn).parameters)[0] == "sim", (
            f"{name} should take `sim` first -- it drives the client, like your code")


def test_the_library_composes_nothing_for_the_agent():
    """No object-level tier, deliberately.

    Deciding WHICH thing matters, finding it and choosing how to take hold of it is the
    task. A `pick("mug")` would hand that over, and the L2 minus L1 gap would then be
    measuring how good our pick is rather than what tools are worth.
    """
    from harness import skills as S

    for name in ("pick", "place", "open_fixture", "close_fixture"):
        assert not hasattr(S, name), f"{name} composes the task for the agent"


def test_the_pure_tiers_need_no_simulator():
    """transforms, camera and geometry are functions of arrays, so an agent can test its
    own reasoning against them without spending a step."""
    from harness import skills as S

    assert S.camera.intrinsics("robot0_eye_in_hand", 128).shape == (3, 3)
    assert S.transforms.quat_to_mat([0.0, 0.0, 0.0, 1.0]).shape == (3, 3)


def test_skills_import_nothing_the_agent_cannot_read():
    """The skill package ships in the AGENT-readable tree at L2. A skill that imported
    config, session or privileged would be a module the agent's uid cannot open, so the
    whole library would fail at import inside the real container -- where the dev suite,
    which can read everything, would never notice."""
    import harness.skills as S

    from tasks.task01.build_assets import AGENT_MODULES

    allowed = {m[:-3] for m in AGENT_MODULES}
    root = Path(S.__file__).parent
    for src in sorted(root.glob("*.py")):
        for line in src.read_text().splitlines():
            if not line.startswith("from .."):
                continue
            # Both spellings reach a parent module: `from ..x import y` names it in the
            # path, `from .. import x` names it after the `import`.
            head = line.split()[1].lstrip(".")
            names = [head.split(".")[0]] if head else [
                n.strip().split(" as ")[0].strip()
                for n in line.split(" import ", 1)[1].split(",")]
            for module in names:
                assert module in allowed, (
                    f"{src.name} imports ..{module}, not agent-visible")


def test_the_manual_ships_beside_the_skills():
    import harness.skills as S

    assert (Path(S.__file__).parent / "MANUAL.md").is_file()


def test_the_library_ships_at_l2_and_l3_and_nowhere_else():
    """L3 is L2 plus the poses, so the L3 minus L2 gap is ONE change.

    Giving L3 the privileged state but not the tools would confound "perception is free"
    with "you have no tools" -- two changes at once, and neither delta interpretable.
    L1 must stay the bare control condition.
    """
    assert C.SKILLS_LEVELS == ("L2", "L3")
    assert "L1" not in C.SKILLS_LEVELS
    assert set(C.SKILLS_LEVELS) <= set(C.LEVELS)


def test_the_wire_is_the_same_at_l2_as_at_l1_and_the_library_is_why():
    """Restated as a property of the constants: the daemon is never told about skills.

    `SKILLS_LEVELS` is read by the image build and by nothing on the serving path, which
    is what makes an L2 run indistinguishable from an L1 one from the daemon's side.
    """
    import inspect

    from harness import service, session

    for module in (service, session):
        assert "SKILLS_LEVEL" not in inspect.getsource(module), (
            f"{module.__name__} branches on the skill library; L2 would stop being L1 "
            f"on the wire")
