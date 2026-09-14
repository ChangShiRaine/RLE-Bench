"""Single-attempt metering, simulator-owned scoring, and socket boundaries."""
import socket
import threading

import pytest

from harness import ledger as L, scoring as S
from harness.session import Budgets, BudgetExhausted, SessionClosed
from harness.tabletop.session import Session
from test_speedrun_session import FakeEnv


class ScoredEnv(FakeEnv):
    def __init__(self, score=.5, raises=False, **kwargs):
        super().__init__(**kwargs)
        self.trial_score = score
        self.raises = raises
        self.score_calls = 0
        self.action_spec = ([-1.] * 12, [1.] * 12)

    def reset(self, seed=None):
        self.owner_thread = threading.get_ident()
        return super().reset(seed)

    def step(self, action):
        assert threading.get_ident() == self.owner_thread
        return super().step(action)

    def _trial_score(self):
        assert threading.get_ident() == self.owner_thread
        self.score_calls += 1
        if self.raises:
            raise RuntimeError("private hook details")
        return self.trial_score


def make_session(tmp_path, budget=10, **kwargs):
    env = ScoredEnv(**kwargs)
    session = Session(task="TowerMaxHeight", budgets=Budgets(budget, 1),
                      ledger=L.Ledger(tmp_path / "ledger.jsonl"),
                      env_factory=lambda **kw: env, trial_seeds=[7])
    session.probe()
    return session, env


def test_finish_scores_current_state_once(tmp_path):
    session, env = make_session(tmp_path)
    session.step([[0.] * 12] * 2)
    assert env.score_calls == 0
    assert not session.status()["done"]
    session.finish()
    assert env.score_calls == 1
    assert S.score_path(session.ledger.path)["reward"] == pytest.approx(.5)
    session.finish()
    session.close()
    assert env.score_calls == 1
    kinds = [r.kind for r in L.load_verified(session.ledger.path)]
    assert kinds.count(L.KIND_SUBMIT) == kinds.count(L.KIND_SEAL) == 1
    with pytest.raises(SessionClosed):
        session.reset()
    with pytest.raises(SessionClosed):
        session.step([[0.] * 12])


def test_budget_exhaustion_submits_on_the_last_step(tmp_path):
    session, env = make_session(tmp_path, budget=3, score=.75)
    out = session.step([[0.] * 12] * 3)
    assert out["done"] and out["ended"] == "budget_exhausted"
    assert out["steps"] == 3 and out["steps_remaining"] == 0
    result = S.score_path(session.ledger.path)
    assert result["interaction_steps"] == 3
    assert result["reward"] == pytest.approx(.75)
    assert env.score_calls == 1


def test_rejected_batch_neither_moves_nor_charges(tmp_path):
    session, env = make_session(tmp_path, budget=3)
    with pytest.raises(BudgetExhausted):
        session.step([[0.] * 12] * 4)
    assert session.state.steps_used == 0 and env.score_calls == 0
    assert not session.status()["done"]
    session.step([[0.] * 12] * 3)
    assert session.status()["done"]


def test_success_predicate_does_not_stop_agent_actions(tmp_path):
    session, env = make_session(tmp_path, success_after=1)
    out = session.step([[0.] * 12] * 3)
    assert out["steps"] == 3 and not out["done"]
    assert env.score_calls == 0


def test_reset_spends_budget_and_discards_previous_score(tmp_path):
    session, env = make_session(tmp_path, budget=4, score=1)
    session.step([[0.] * 12])
    session.reset()
    assert session.steps_remaining == 2
    assert env.score_calls == 0
    env.trial_score = .25
    session.finish()
    assert S.score_path(session.ledger.path)["reward"] == pytest.approx(.25)


def test_reset_cannot_change_hidden_seed(tmp_path):
    import numpy as np
    session, env = make_session(tmp_path)
    env.rng = np.random.default_rng(0)
    session.reset()
    first = env.rng.random(3)
    session.reset()
    assert np.array_equal(first, env.rng.random(3))
    with pytest.raises(ValueError, match="seed"):
        session.reset(seed=42)
    assert session.state.steps_used == 2


def test_reset_at_last_budget_unit_finalizes(tmp_path):
    session, env = make_session(tmp_path, budget=1)
    session.reset()
    assert session.status()["done"] and session.steps_remaining == 0
    assert env.score_calls == 1


@pytest.mark.parametrize("value,expected", [(7, 1), (-1, 0), (float("nan"), 0),
                                           (float("inf"), 0)])
def test_invalid_scores_are_bounded(tmp_path, value, expected):
    session, _ = make_session(tmp_path, score=value)
    session.close()
    assert S.score_path(session.ledger.path)["reward"] == expected


def test_raising_score_hook_gives_zero(tmp_path):
    session, _ = make_session(tmp_path, raises=True)
    session.close()
    assert S.score_path(session.ledger.path)["reward"] == 0


def test_simulator_error_charges_and_finalizes_zero(tmp_path):
    session, env = make_session(tmp_path, score=1)
    def fail(_):
        raise RuntimeError("private model detail")
    env.step = fail
    with pytest.raises(RuntimeError):
        session.step([[0.] * 12])
    result = S.score_path(session.ledger.path)
    assert result["reward"] == 0 and result["interaction_steps"] == 1
    assert result["end_reason"] == "simulation_error"


def test_agent_exit_scores_the_unsubmitted_scene(tmp_path):
    session, _ = make_session(tmp_path, score=.8)
    session.step([[0.] * 12])
    session.close()
    assert S.score_path(session.ledger.path)["reward"] == pytest.approx(.8)


@pytest.mark.parametrize("task", ["TowerMaxHeight", "BalanceCoins"])
def test_socket_reconnect_and_operation_boundary(tmp_path, monkeypatch, task):
    from harness import service as shared
    from harness.tabletop.service import Service
    from harness.tabletop.client import TabletopClient, RemoteError
    monkeypatch.setattr(shared, "MeteredSession", Session)
    monkeypatch.setenv("RLEBENCH_LEVEL", "L3")
    env = ScoredEnv(score=.6)
    env.position_observation = lambda: {
        "cube_positions": {f"coin_{i}": [float(env.steps), i, 0.9] for i in range(9)},
        "pan_positions": {"left": [0.1, 0.17, 1.12], "right": [0.1, 0.67, 1.12]},
    }
    svc = Service(task=task, budgets=Budgets(10, 1),
                  ledger_path=str(tmp_path / "ledger.jsonl"),
                  env_factory=lambda **kw: env, trial_seeds=[7])
    svc.warm_up()
    # FakeEnv has no camera renderer; production rendering is tested with real scenes.
    monkeypatch.setattr("harness.tabletop.session.ENV.apply_obs_spec",
                        lambda env, obs, spec, **kw: (obs, 512))
    # Socketpair exercises the same handler without accept-loop timing dependencies.
    def client():
        agent, daemon = socket.socketpair()
        worker = threading.Thread(target=svc.handle_connection, args=(daemon,))
        worker.start()
        sim = TabletopClient.__new__(TabletopClient)
        sim._sock = agent
        sim._reader = agent.makefile("rb")
        return sim, worker, daemon
    sim, worker, daemon = client()
    obs = sim.step([0.] * 12)
    if task == "BalanceCoins":
        assert obs["cube_positions"]["coin_0"] == [1.0, 0, 0.9]
        assert set(obs["pan_positions"]) == {"left", "right"}
        assert "mass" not in obs and "heavy" not in obs
    else:
        assert "cube_positions" not in obs and "pan_positions" not in obs
    for op in ("end_development", "open_evaluation", "trial_info", "submit"):
        with pytest.raises(RemoteError):
            sim._request(op)
    with pytest.raises(RemoteError):
        sim._request("finish", success_rate=1)
    sim.disconnect()
    worker.join(timeout=2)
    daemon.close()
    assert not worker.is_alive()
    sim, worker, daemon = client()
    assert sim.status()["steps_used"] == 1
    assert "phase" not in sim.status()
    assert sim.finish()["done"]
    sim.disconnect()
    worker.join(timeout=2)
    daemon.close()
    assert not worker.is_alive()
    assert env.score_calls == 1
    svc.seal_if_open()
    svc.shutdown()


def test_scene_factory_composition(monkeypatch):
    from harness.tabletop.daemon_main import _install_env_factory
    import harness.tabletop as tt
    from harness import env as shared
    _install_env_factory()
    before = shared.make_env
    _install_env_factory()
    assert shared.make_env is before
    calls = []
    monkeypatch.setattr(tt, "make", lambda task, **kw: calls.append((task, kw)) or "ENV")
    assert shared.make_env(task="TowerMaxHeight", split="target") == "ENV"
    assert calls[0][0] == "TowerMaxHeight"
    with pytest.raises(ValueError, match="split must be"):
        shared.make_env("PickPlaceCounterToSink", split="bogus")


def test_harness_stop_interrupts_batch_without_losing_charges(tmp_path):
    session, env = make_session(tmp_path, budget=100)
    step = env.step
    def stop_after_two(action):
        result = step(action)
        if env.steps == 2:
            session.request_stop()
        return result
    env.step = stop_after_two
    out = session.step([[0.] * 12] * 100)
    assert out["steps"] == 2 and out["done"]
    result = S.score_path(session.ledger.path)
    assert result["end_reason"] == "harness_seal"
    assert result["interaction_steps"] == 2
    assert result["reward"] == pytest.approx(.5)


def test_public_view_ignores_privileged_level_and_raw_state():
    import numpy as np
    from harness.tabletop.service import Service
    svc = Service.__new__(Service)
    svc._level = "L3"
    raw = {"robot0_joint_pos": np.zeros(7), "robot0_eef_pos": np.ones(3),
           "robot0_agentview_left_image": np.zeros((8, 8, 3), dtype=np.uint8),
           "priv_objects": {"heavy": 3}, "object-state": [1, 2],
           "robot0_contact": True, "robot0_eef_force": [1, 2, 3],
           "robot0_agentview_left_depth": np.ones((8, 8))}
    assert set(svc._shown(raw)) == {"joint_pos", "eef_pos", "images"}
    assert set(svc._shown(raw)["images"]) == {"left"}


def test_move_uses_robot_frame_and_charges_each_step(tmp_path):
    import numpy as np
    from scipy.spatial.transform import Rotation
    session, env = make_session(tmp_path, budget=3)
    rotation = Rotation.from_euler("z", 90, degrees=True)
    base = np.array([1., 2., 0.])
    session._obs.update(robot0_base_pos=base, robot0_base_quat=rotation.as_quat(),
                        robot0_base_to_eef_pos=np.array([.2, 0., 1.]),
                        robot0_base_to_eef_quat=np.array([0., 0., 0., 1.]))
    captured = []
    def step(action):
        captured.append(action)
        return session._obs, 0, False, {}
    env.step = step
    target = base + rotation.apply([.225, 0., 1.])
    request = dict(position=target.tolist(), quaternion=rotation.as_quat().tolist(), steps=3)
    out = session.move(request)
    assert out["done"] and session.state.steps_used == 3
    for action in captured:
        np.testing.assert_allclose(action, [.5, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1], atol=1e-12)


@pytest.mark.parametrize("patch", [{"steps": 0}, {"steps": 201}, {"steps": True},
                                   {"quaternion": [0, 0, 0, 0]},
                                   {"position": [float("nan"), 0, 1]}, {"gripper": 2}])
def test_invalid_move_does_not_spend_budget(tmp_path, patch):
    session, env = make_session(tmp_path)
    request = dict(position=[0, 0, 1], quaternion=[0, 0, 0, 1], steps=1)
    request.update(patch)
    with pytest.raises(ValueError):
        session.move(request)
    assert session.state.steps_used == 0 and env.score_calls == 0


def test_public_payload_has_only_the_client(tmp_path):
    from tasks.task03 import build_assets as builder
    builder.stage(tmp_path, builder.AGENT_MODULES)
    assert builder.check(tmp_path) == []
    (tmp_path / "harness/privileged.py").write_text("secret = 1")
    assert builder.check(tmp_path)
