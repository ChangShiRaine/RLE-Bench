"""Tests for the acting contract: `step`.

There is no shape the agent has to write its code in -- it sends actions and reads what
came back. What the harness guarantees, and what these tests pin down:

  * success is the ENVIRONMENT's predicate. Nothing the agent sends can assert it, and
    nothing it sends can avoid it;
  * a budget or a horizon stops the agent mid-batch, and the reply says how many actions
    were actually applied;
  * `episode_over` is the only thing separating "carry on where you left off" from "you
    must reset" -- which is what lets one loop drive both phases.
"""

from __future__ import annotations

import pytest

from harness import ledger as L
from harness.obs import ObsSpec
from harness.session import Termination
from harness.session import Budgets, MeteredSession, _as_actions
from test_speedrun_session import FakeEnv  # pytest puts tests/ on sys.path

A = [0.0] * 12


def make_session(tmp_path, *, steps=100, submissions=3,
                 success_after=-1, seeds=(1,), max_episode_steps=None, name="c"):
    created = []

    def factory(task, split="pretrain", scene=None):
        env = FakeEnv(success_after=success_after, split=split)
        created.append(env)
        return env

    sess = MeteredSession(
        task="FakeTask",
        budgets=Budgets(interaction_steps=steps, submissions=submissions),
        ledger=L.Ledger(tmp_path / f"{name}.jsonl"),
        env_factory=factory,
        trial_seeds=list(seeds),
        max_episode_steps=max_episode_steps,
    )
    return sess, created


# -- one action or a batch, told apart by shape --------------------------------

def test_a_single_action_and_a_batch_are_both_just_what_they_look_like():
    """No flag distinguishes them: an action is a sequence of numbers, a batch is a
    sequence of those. Getting this wrong would silently apply a 12-action batch when
    the caller meant one 12-component action."""
    assert _as_actions(A) == [A]
    assert _as_actions([A, A, A]) == [A, A, A]


def test_numpy_actions_are_told_apart_by_ndim():
    np = pytest.importorskip("numpy")
    assert len(_as_actions(np.zeros(12))) == 1
    assert len(_as_actions(np.zeros((5, 12)))) == 5


# -- the contract --------------------------------------------------------------

def test_one_action_is_one_step(tmp_path):
    sess, _ = make_session(tmp_path)
    sess.reset()
    res = sess.step(A)
    assert res["steps"] == 1
    assert res["ended"] is None          # still alive
    assert res["episode_over"] is False
    assert sess.state.steps_used == 2          # the reset, then the action


def test_a_batch_costs_exactly_what_it_applies(tmp_path):
    sess, _ = make_session(tmp_path)
    sess.reset()
    assert sess.step([A] * 15)["steps"] == 15
    assert sess.state.steps_used == 16         # + the reset


def test_steps_chain_within_one_episode_without_resetting(tmp_path):
    """Working in pieces is the point: send some actions, look, send more -- all inside
    one episode, with no reset in between."""
    sess, created = make_session(tmp_path)
    sess.reset()
    resets_after_first = created[0].resets
    a = sess.step([A] * 3)
    b = sess.step([A] * 2)
    assert (a["steps"], b["steps"]) == (3, 2)
    assert sess.state.steps_used == 6          # + the one reset that opened them
    assert created[0].resets == resets_after_first


def test_env_success_stops_a_batch_and_is_authoritative(tmp_path):
    """The environment's predicate ends the episode part-way through a batch; the
    remaining actions are never applied and are never charged."""
    sess, _ = make_session(tmp_path, success_after=2)
    sess.reset()
    res = sess.step([A] * 100)
    assert res["ended"] == Termination.ENV_SUCCESS
    assert res["success"] is True
    assert res["steps"] == 2
    assert res["episode_over"] is True
    assert sess.state.steps_used == 3          # + the reset


def test_success_is_never_the_agents_to_assert(tmp_path):
    """There is no mechanism for it, and that is the design: `success` is read off the
    environment on every reply."""
    sess, _ = make_session(tmp_path, success_after=-1)
    sess.reset()
    assert sess.step([A] * 5)["success"] is False


def test_the_budget_stops_a_batch(tmp_path):
    sess, _ = make_session(tmp_path, steps=4)
    sess.reset()                               # one of the four
    res = sess.step([A] * 10_000)
    assert res["ended"] == Termination.BUDGET_EXHAUSTED
    assert res["steps"] == 3
    assert sess.state.steps_used == 4


def test_the_horizon_ends_a_development_episode(tmp_path):
    sess, _ = make_session(tmp_path, max_episode_steps=3)
    sess.reset()
    res = sess.step([A] * 10)
    assert res["ended"] == Termination.HORIZON
    assert res["steps"] == 3
    assert res["episode_over"] is True


def test_a_dead_episode_must_be_reset_not_resumed(tmp_path):
    sess, _ = make_session(tmp_path, max_episode_steps=2)
    sess.reset()
    sess.step([A] * 5)
    with pytest.raises(Exception, match="reset"):
        sess.step(A)
    sess.reset()
    assert sess.step(A)["steps"] == 1


def test_steps_are_journalled_exactly_once(tmp_path):
    sess, _ = make_session(tmp_path)
    sess.reset()
    sess.step([A] * 5)
    sess.close()
    interacts = [r for r in L.load_verified(sess.ledger.path)
                 if r.kind == L.KIND_INTERACT]
    assert sum(int(r.payload["steps"]) for r in interacts) == 6   # + the reset


def test_steps_remaining_tracks_the_budget(tmp_path):
    sess, _ = make_session(tmp_path, steps=10)
    sess.reset()                                          # spends one of the ten
    assert sess.step([A] * 4)["steps_remaining"] == 5


def test_obs_spec_shapes_what_comes_back(tmp_path):
    """The agent chooses what an observation contains -- `cameras=()` drops the images
    entirely, which is what makes a proprioception-only loop cheap."""
    from test_speedrun_service import FakeEnv as CameraEnv

    def factory(task, split="pretrain", scene=None):
        return CameraEnv(success_after=-1, split=split)

    sess = MeteredSession(
        task="FakeTask",
        budgets=Budgets(interaction_steps=100, submissions=1),
        ledger=L.Ledger(tmp_path / "spec.jsonl"),
        env_factory=factory,
        trial_seeds=[1],
    )
    sess.reset()
    assert [k for k in sess.step([0.0] * 12)["obs"] if k.endswith("_image")]

    res = sess.step([0.0] * 12, obs_spec=ObsSpec(cameras=()))
    assert not [k for k in res["obs"] if k.endswith("_image")]
    assert "robot0_proprio-state" in res["obs"]
