from collections import deque
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from harness.base_design import controlled_pick
from harness.base_design.scorer import load_submission
from harness import config


@pytest.mark.parametrize("reach", [0.2, 1.4])
def test_initial_pose_is_fixed_independently_of_robot_geometry(reach):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="base"><freejoint name="base_free"/>
        <geom type="box" size=".2 .2 .1"/>
        <body pos="{reach} 0 .5"><joint name="arm_joint"/>
          <geom type="sphere" size=".1"/></body>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.qpos[7] = 0.3
    data.qvel[:] = 1.0
    controlled_pick.initialize_episode(model, data)
    np.testing.assert_array_equal(data.qpos[:7], [-.60, 0., .08, 1., 0., 0., 0.])
    assert data.qpos[7] == 0.3
    assert data.time == 0
    np.testing.assert_array_equal(data.qvel, 0.)


def test_numerical_reset_cannot_teleport_past_the_outside_start(monkeypatch):
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    data = mujoco.MjData(model)
    data.time = 3.0
    monkeypatch.setattr(mujoco, "mj_step", lambda model, data: setattr(data, "time", .002))
    with pytest.raises(ValueError, match="simulation reset"):
        controlled_pick._step_physics(model, data)


def test_controller_measurement_matches_analytic_static_margin(monkeypatch):
    square = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0.]])
    monkeypatch.setattr(controlled_pick, "wheel_contacts", lambda *args: square)
    monkeypatch.setattr(controlled_pick, "net_com_force", lambda *args: np.array([0, 0, -10.]))
    data = SimpleNamespace(subtree_com=np.array([[0, 0, 1.]]), site_xpos=np.array([[.5, 0, .3]]))
    error, margin = controlled_pick._measure(None, data, np.array([.5, 0, .3]), 0, deque(), 0.)
    assert error == 0
    assert margin == pytest.approx(np.pi / 4 * 10 * np.sqrt(2))
    data.subtree_com[0, 0] = 2
    _, margin = controlled_pick._measure(None, data, np.array([.5, 0, .3]), 0, deque(), 0.)
    assert margin < 0


def test_loading_controller_never_executes_it_in_the_verifier(tmp_path):
    (tmp_path / "robot.xml").write_text("<mujoco/>")
    (tmp_path / "controller.py").write_text("raise RuntimeError('must not be imported')")
    sub = load_submission(str(tmp_path))
    assert sub.shelf_controller == str(tmp_path / "controller.py")


def test_missing_controller_is_explicit_and_has_no_golden_fallback(tmp_path):
    (tmp_path / "robot.xml").write_text("<mujoco/>")
    sub = load_submission(str(tmp_path))
    assert sub.shelf_controller is None
    assert any("controller.py missing" in error for error in sub.errors)


def test_controlled_pick_rejects_nondeterministic_actions(monkeypatch):
    monkeypatch.setattr(controlled_pick.scene, "shelf_placement", lambda *args: (.2, .4))
    monkeypatch.setattr(controlled_pick.scene, "scored_target_schedule",
                        lambda *args: [(0, "target", np.zeros(3))])
    calls = []

    def target(*args):
        calls.append(args)
        return dict(ok=True, reason=None, action_sha256=str(len(calls)))

    monkeypatch.setattr(controlled_pick, "run_target", target)
    result = controlled_pick.evaluate_controlled_pick("robot", "arm", None, "controller")
    assert len(calls) == 3  # 1 kg, 2 kg, then replay
    assert result["deterministic"] is False
    assert result["score"] == result["margin_score"] == 0


def test_oracle_policy_is_standalone():
    import ast
    path = Path(__file__).resolve().parents[1] / "tasks/task08/solution/payload/controller.py"
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not node.level
            assert not (node.module or "").startswith(("harness", "rlebench"))
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith(("harness", "rlebench")) for a in node.names)


@pytest.mark.parametrize("forged", [False, True])
def test_reported_success_and_observation_edits_cannot_move_the_robot(monkeypatch, forged):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body name="base" pos="0 0 .1"><freejoint name="base_free"/>
        <geom type="box" size=".1 .1 .1"/><site name="ee" pos="0 0 .2"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    monkeypatch.setattr(config, "PICK_EPISODE_SECONDS", .04)
    monkeypatch.setattr(controlled_pick.scene, "compose_pick_scene", lambda *args: (model, data, []))
    arm = SimpleNamespace(name="test", joint_names=[], ee_site="ee")

    contexts = []

    class Controller:
        def __init__(self, *args):
            contexts.append(args[2])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def act(self, observation):
            observation["qpos"][0] = .5
            observation["reward"] = 1.0
            observation["ok"] = True
            return {"reward": 1.0} if forged else []

    result = controlled_pick.run_target("robot", .4, arm, np.array([.5, 0, .3]),
                                        "target", 1., "policy", process_factory=Controller)
    assert contexts[0]["qpos"][:3] == [-.60, 0., .08]
    assert not any(contexts[0]["qvel"])
    assert result["ok"] is False
    assert data.qpos[0] != .5
    if forged:
        assert result["reason"] == "controller fault"
