"""Tests for task01's metered session.

These are the anti-gaming tests for the cost measurement itself: the score is
"interaction steps and submissions spent when the threshold was first cleared", so
each way of understating either currency needs to be covered.

A fake env keeps this free of MuJoCo, so the metering contract is testable in the
fast dev suite rather than only in the 3 GB container.
"""

from __future__ import annotations

import pytest

from harness import ledger as L
from harness.session import Termination
from harness.session import Budgets, BudgetExhausted, MeteredSession


class FakeEnv:
    """Succeeds after `success_after` steps; -1 means never.

    Mirrors the real seeding mechanism deliberately: robosuite envs take NO seed on
    reset() and instead draw from `self.rng`, which the harness re-seeds between
    episodes. The fake therefore exposes `rng` and records what it drew, so these
    tests exercise the same path the simulator does.
    """

    def __init__(self, success_after: int = 3, split: str = "pretrain"):
        import numpy as np

        self.success_after = success_after
        self.split = split
        self.steps = 0
        self.resets = 0
        self.closed = False
        self._success = False
        self.rng = np.random.default_rng(0)
        self.draws = []

    def reset(self, seed=None):
        self.resets += 1
        self.steps = 0
        self._success = False
        self.draws.append(int(self.rng.integers(0, 2**31 - 1)))
        return {"obs": 0, "draw": self.draws[-1]}

    def step(self, action):
        self.steps += 1
        if self.success_after >= 0 and self.steps >= self.success_after:
            self._success = True
        return {"obs": self.steps}, 0.0, False, {}

    def _check_success(self):
        return self._success

    def close(self):
        self.closed = True


def make_session(tmp_path, *, steps=100, submissions=3,
                 success_after=3, seeds=(1, 2), max_episode_steps=10):
    created = []

    def factory(task, split="pretrain", scene=None):
        env = FakeEnv(success_after=success_after, split=split)
        created.append(env)
        return env

    sess = MeteredSession(
        task="FakeTask",
        budgets=Budgets(interaction_steps=steps, submissions=submissions),
        ledger=L.Ledger(tmp_path / "cost.jsonl"),
        env_factory=factory,
        trial_seeds=list(seeds),
        max_episode_steps=max_episode_steps,
    )
    return sess, created


def spend(sess, steps: int) -> None:
    """Charge exactly `steps` development steps through the only acting path there is.

    One action at a time, resetting whenever the episode ends: with `success_after` set,
    a batch stops the moment the predicate fires, so a bare loop would stall on a
    finished episode instead of spending what the caller asked for.
    """
    for _ in range(steps):
        if sess.step([0.0])["episode_over"]:
            sess.reset()


def run_evaluation(sess, action=(0.0,), batch=1000):
    """Drive a whole evaluation the way the agent does, and return its summary.

    Step until the trial ends, reset to reach the next, repeat. There is no harness
    call that runs an evaluation for you -- that is the point of the interface -- so
    tests that care about the aggregate drive it exactly as an agent would.
    """
    sess.begin_evaluation()
    while sess.evaluating:
        res = sess.step([list(action)] * batch)
        if sess.evaluating and (res["episode_over"] or res["steps"] == 0):
            sess.reset()
    return sess.state.history[-1]


def test_one_action_charges_exactly_one_step(tmp_path):
    """`step` is the only way to act, so per-step charging is measured through it."""
    sess, _ = make_session(tmp_path, steps=6, success_after=-1)
    sess.reset()                                 # the episode itself costs one
    for i in range(5):
        sess.step([0.0])
        assert sess.state.steps_used == i + 2
    assert sess.steps_remaining == 0


def test_reset_costs_one_step(tmp_path):
    """A reset is a real draw on the simulator -- it rebuilds or re-randomises a scene.
    Free resets made abandoning an episode cheaper than finishing it, which is the
    opposite of what a multi-step task should reward."""
    sess, _ = make_session(tmp_path, steps=10)
    for _ in range(4):
        sess.reset()
    assert sess.state.steps_used == 4


def test_reset_past_the_budget_is_refused(tmp_path):
    """Charged like anything else, so it is refused like anything else -- and refused
    before the scene is rebuilt, leaving the session as it was."""
    sess, _ = make_session(tmp_path, steps=2)
    sess.reset()
    sess.reset()
    episodes = sess.state.episodes
    with pytest.raises(BudgetExhausted):
        sess.reset()
    assert sess.state.steps_used == 2 and sess.state.episodes == episodes


def test_the_reset_does_not_eat_the_new_episodes_horizon(tmp_path):
    """The step a reset costs is charged to the interaction budget, not to the episode
    it opens: a fresh episode still gets its full horizon."""
    sess, _ = make_session(tmp_path, steps=50, success_after=-1, max_episode_steps=4)
    sess.reset()
    assert sess.state.episode_steps == 0
    res = sess.step(MANY)
    assert res["steps"] == 4 and res["ended"] == Termination.HORIZON


def test_stepping_before_the_first_reset_pays_for_the_episode_it_opens(tmp_path):
    """`step` opens the first episode implicitly, so it must pay exactly what `reset`
    would -- otherwise the charge is walked around by never calling reset."""
    sess, _ = make_session(tmp_path, steps=10, success_after=-1)
    sess.step([[0.0] * 12] * 3)
    assert sess.state.steps_used == 4        # 1 for the episode + 3 actions


def test_observing_never_opens_an_episode(tmp_path):
    """Looking is free, and a reset is not -- so `observe` must never manufacture one."""
    sess, _ = make_session(tmp_path, steps=10, success_after=-1, max_episode_steps=4)
    look = sess.observe()
    assert look["live"] is False and look["obs"] == {}
    assert sess.state.steps_used == 0 and sess.state.episodes == 0

    sess.reset()
    assert sess.step(MANY)["episode_over"] is True
    spent, episodes = sess.state.steps_used, sess.state.episodes
    assert sess.observe()["live"] is False
    assert sess.state.steps_used == spent and sess.state.episodes == episodes


def test_acting_past_the_budget_is_refused_not_truncated(tmp_path):
    """The budget is a hard ceiling. A segment that starts with nothing left takes no
    action at all and says why, rather than quietly running for free."""
    sess, _ = make_session(tmp_path, steps=2)
    sess.reset()
    sess.step([0.0])
    sess.step([0.0])
    res = sess.step([0.0])
    assert res["steps"] == 0
    assert res["ended"] == Termination.BUDGET_EXHAUSTED
    assert res["episode_over"] is True
    assert sess.state.steps_used == 2


def test_repeated_episodes_charge_every_step_they_took(tmp_path):
    """Three episodes driven the way an agent drives them, each ending on success at
    step 4."""
    sess, _ = make_session(tmp_path, steps=100, success_after=4,
                           max_episode_steps=10)
    successes = 0
    for _ in range(3):
        sess.reset()
        res = sess.step([[0.0]] * 10)
        successes += int(res["success"])
    assert sess.state.steps_used == 15         # 3 x (1 reset + 4 steps)
    assert successes == 3


def test_everything_actually_spent_is_charged_and_journalled(tmp_path):
    """Even when the budget runs out part-way: the ledger is the authority the reward
    is computed from, so it must never lag what was consumed."""
    sess, _ = make_session(tmp_path, steps=5, success_after=-1,
                           max_episode_steps=10)
    res = sess.step([[0.0]] * 30)
    assert res["ended"] == Termination.BUDGET_EXHAUSTED
    assert sess.state.steps_used == 5
    records = L.load_verified(sess.ledger.path)
    assert sum(r.payload.get("steps", 0)
               for r in records if r.kind == L.KIND_INTERACT) == sess.state.steps_used


def test_evaluation_steps_are_not_charged_to_the_interaction_budget(tmp_path):
    """The two currencies must stay separate, or a submission would look like
    cheap learning."""
    sess, _ = make_session(tmp_path, steps=10, success_after=3, seeds=(1, 2, 3))
    run_evaluation(sess)
    assert sess.state.steps_used == 0
    assert sess.steps_remaining == 10


def test_each_evaluation_consumes_one_submission_and_is_refused_past_budget(tmp_path):
    sess, _ = make_session(tmp_path, submissions=2, success_after=-1)
    run_evaluation(sess)
    run_evaluation(sess)
    assert sess.submissions_remaining == 0
    with pytest.raises(BudgetExhausted):
        run_evaluation(sess)


def test_an_evaluation_reports_only_aggregates_never_per_seed_results(tmp_path):
    """Per-seed feedback would let an agent fit the exact graded episodes."""
    sess, _ = make_session(tmp_path, seeds=(11, 22, 33), success_after=2)
    out = run_evaluation(sess)
    assert set(out) == {
        "evaluation_done", "submission_index", "success_rate", "trials",
        "trials_attempted", "submissions_remaining", "steps_remaining",
    }
    # No per-scene or per-trial breakdown: that lives in the ledger only, because
    # returning it would let the agent localise the graded episodes.
    assert "per_scene" not in out and "per_trial" not in out
    blob = repr(out)
    for seed in ("11", "22", "33"):
        assert seed not in blob


def test_status_never_carries_a_stale_success_aggregate(tmp_path):
    """The aggregate only moved when an evaluation finalised, so mid-run it was
    stale. The per-step `success` flag and the last reset()'s aggregate are the
    success channels; status offers no third one."""
    sess, _ = make_session(tmp_path, submissions=1)
    assert "best_success_rate" not in sess.status()
    sess.begin_evaluation()
    sess.step([[0.0]] * 5)                    # the fake env succeeds after 3 steps
    assert "best_success_rate" not in sess.status()


def test_phase_reports_finished_once_nothing_can_be_driven(tmp_path):
    """A run whose evaluation had ended reported phase="development" -- a phase whose
    interaction would have been refused. The terminal state has its own name."""
    sess, _ = make_session(tmp_path, submissions=1, success_after=-1)
    assert sess.phase == "development"
    sess.end_development()
    run_evaluation(sess)
    assert sess.phase == "finished"
    assert sess.status()["phase"] == "finished"


def test_status_step_figures_are_phase_local(tmp_path):
    """The development figure does not move when a trial steps, so during evaluation
    status reports the open trial's count and remaining ceiling instead."""
    sess, _ = make_session(tmp_path, steps=100, success_after=-1, max_episode_steps=10)
    sess.reset()
    sess.step([[0.0]] * 3)
    assert sess.status()["steps_used"] == 4            # the reset, then three actions
    sess.begin_evaluation()
    st = sess.status()
    assert (st["steps_used"], st["steps_remaining"]) == (0, 10)
    res = sess.step([[0.0]] * 4)
    st = sess.status()
    assert (st["steps_used"], st["steps_remaining"]) == (4, 6)
    assert res["steps_remaining"] == 6                 # the step reply agrees


def test_phase_returns_to_development_while_a_submission_remains(tmp_path):
    """"finished" is only for the terminal state: with development still open and a
    submission left, a closed evaluation hands control back to development."""
    sess, _ = make_session(tmp_path, submissions=2, success_after=-1)
    run_evaluation(sess)
    assert sess.phase == "development"
    sess.reset()                              # development interaction still works
    assert sess.status()["steps_used"] > 0


def test_success_rate_is_the_fraction_of_trials_solved(tmp_path):
    """Continuous, not a pass/fail against a threshold: the reward is proportional to
    this, so partial competence has to be visible as a partial number."""
    sess, _ = make_session(tmp_path, success_after=-1)
    assert run_evaluation(sess)["success_rate"] == 0.0

    sess2, _ = make_session(tmp_path / "b", success_after=2)
    assert run_evaluation(sess2)["success_rate"] == 1.0


def test_partial_success_reports_a_partial_rate(tmp_path):
    """Half the battery succeeds, so the rate is 0.5 -- neither a pass nor a fail."""

    class HalfEnv(FakeEnv):
        """Alternates success per episode, so exactly half the battery passes."""

        def reset(self, seed=None):
            out = super().reset(seed=seed)
            self.success_after = 2 if len(self.draws) % 2 == 1 else -1
            return out

    def factory(task, split="pretrain", scene=None):
        return HalfEnv(split=split)

    sess = MeteredSession(
        task="T",
        budgets=Budgets(interaction_steps=100, submissions=5),
        ledger=L.Ledger(tmp_path / "half.jsonl"),
        env_factory=factory,
        trial_seeds=[2, 3],
        max_episode_steps=10,
    )
    assert run_evaluation(sess)["success_rate"] == 0.5


def test_the_ledger_totals_development_interaction_only(tmp_path):
    """The efficiency term is measured against this number, so it must count what the
    agent spent LEARNING and nothing it spent being graded."""
    sess, _ = make_session(tmp_path, steps=100, submissions=5, success_after=2,
                           seeds=(1,), max_episode_steps=10)
    sess.reset()
    spend(sess, 7)
    before_eval = sess.state.steps_used
    run_evaluation(sess)          # graded steps: not charged
    assert sess.state.steps_used == before_eval
    sess.reset()
    spend(sess, 20)
    run_evaluation(sess)
    summary = sess.close()

    # Asserted against the counter rather than a constant: `spend` resets whenever an
    # episode ends, and resets are charged, so the total tracks how often it had to.
    assert summary["interaction_steps_total"] == sess.state.steps_used > 27
    assert summary["best_success_rate"] == 1.0
    assert summary["submissions_used"] == 2
    assert summary["sealed"] is True
    assert summary["end_reason"] == "harness_seal"


def test_close_records_how_the_run_ended(tmp_path):
    """Diagnosis, never reward: it is what separates a genuine zero from a run whose
    agent died before it drove anything."""
    sess, _ = make_session(tmp_path)
    summary = sess.close(end_reason="agent_exited")
    assert summary["end_reason"] == "agent_exited"


def test_close_seals_and_is_idempotent(tmp_path):
    sess, _ = make_session(tmp_path)
    first = sess.close()
    second = sess.close()
    assert first["sealed"] and second["sealed"]
    L.load_verified(sess.ledger.path)  # still a valid chain
    with pytest.raises(RuntimeError):
        sess.step([0.0])


def test_status_does_not_leak_the_validation_battery(tmp_path):
    sess, _ = make_session(tmp_path, seeds=(4242, 9999))
    blob = repr(sess.status())
    assert "4242" not in blob and "9999" not in blob


def test_eval_env_uses_the_eval_split_and_dev_env_the_dev_split(tmp_path):
    sess, created = make_session(tmp_path, success_after=2)
    sess.reset()
    sess.step([0.0])
    run_evaluation(sess)
    splits = [e.split for e in created]
    assert splits[0] == "pretrain"   # development
    assert splits[1] == "target"     # evaluation


# -- ending development ------------------------------------------------------

def test_end_development_records_the_cost_at_the_moment_of_readiness(tmp_path):
    """The point of the signal: what the agent spent before judging itself ready is a
    fact in the ledger, not a claim it makes afterwards."""
    # success_after=-1 so the four steps stay inside ONE episode: the assertion below is
    # about the episode count as much as the step count.
    sess, _ = make_session(tmp_path, success_after=-1)
    sess.reset()
    sess.step([[0.0] * 12] * 4)

    out = sess.end_development()
    assert out == {"development_ended": True, "steps_used": 5,
                   "steps_remaining": 95, "episodes": 1}

    records = L.load_verified(sess.ledger.path)
    dev_end = [r for r in records if r.kind == L.KIND_DEV_END]
    assert len(dev_end) == 1
    assert dev_end[0].payload["steps_used"] == 5


def test_end_development_touches_no_simulator(tmp_path):
    """It must be inert: no env constructed, no scene reset, no submission spent."""
    sess, created = make_session(tmp_path)
    sess.end_development()
    assert created == []                       # nothing was built
    assert sess.state.submissions_used == 0
    assert sess.phase == "awaiting_evaluation"  # closed, but not the scored phase


def test_development_interaction_is_refused_afterwards(tmp_path):
    """Declaring development over has to mean something, or it is just a comment."""
    from harness.session import DevelopmentClosed

    sess, _ = make_session(tmp_path)
    sess.reset()
    sess.end_development()
    for call in (lambda: sess.reset(),
                 lambda: sess.step([0.0] * 12)):
        with pytest.raises(DevelopmentClosed):
            call()


def test_end_development_is_idempotent(tmp_path):
    """An agent that says it twice should not be punished, or produce two records."""
    sess, _ = make_session(tmp_path)
    first = sess.end_development()
    second = sess.end_development()
    assert first == second
    records = L.load_verified(sess.ledger.path)
    assert sum(1 for r in records if r.kind == L.KIND_DEV_END) == 1


def test_evaluation_still_works_after_development_ends(tmp_path):
    """Ending development closes the development phase only -- the scored phase that
    the harness opens next must be unaffected."""
    sess, _ = make_session(tmp_path, success_after=1)
    sess.end_development()
    info = sess.begin_evaluation()
    assert info["total_trials"] >= 1
    res = sess.step([[0.0] * 12] * 1000)   # evaluation drives the same call
    assert res["steps"] >= 1


# -- handover: why control came back, and whether the episode survived --------

# More actions than any episode here can absorb, so only the harness can stop it.
MANY = [[0.0] * 12] * 1000


def test_development_episode_ends_at_the_horizon(tmp_path):
    """Development had NO episode-level accounting: an over-long episode was never
    ended, and Termination.HORIZON was defined but never raised anywhere."""
    sess, _ = make_session(tmp_path, success_after=-1, max_episode_steps=10)
    sess.reset()
    res = sess.step(MANY)
    assert res["ended"] == Termination.HORIZON
    assert res["steps"] == 10
    assert res["episode_over"] is True


def test_running_out_of_actions_leaves_the_episode_alive(tmp_path):
    """The agent working in pieces: a batch that simply ends is not an episode that
    ended, so the next batch continues the SAME episode."""
    sess, _ = make_session(tmp_path, success_after=-1, max_episode_steps=100)
    sess.reset()
    res = sess.step([[0.0] * 12] * 3)
    assert res["ended"] is None
    assert res["steps"] == 3
    assert res["episode_over"] is False

    again = sess.step([[0.0] * 12] * 2)
    assert again["steps"] == 2 and again["episode_over"] is False
    assert sess.state.episode_steps == 5


def test_a_dead_episode_cannot_be_continued(tmp_path):
    """`_apply` resumes from _last_obs, so without this the next call would carry on in
    an episode that had already ended."""
    from harness.session import EpisodeOver

    sess, _ = make_session(tmp_path, success_after=-1, max_episode_steps=4)
    sess.reset()
    assert sess.step(MANY)["episode_over"] is True

    with pytest.raises(EpisodeOver):
        sess.step(MANY)

    sess.reset()                       # the only way forward
    assert sess.state.episode_steps == 0
    assert sess.step([0.0] * 12)["steps"] == 1


def test_env_success_also_ends_the_episode(tmp_path):
    """Environment events end episodes too, not just the horizon."""
    sess, _ = make_session(tmp_path, success_after=2, max_episode_steps=100)
    sess.reset()
    res = sess.step(MANY)
    assert res["ended"] == Termination.ENV_SUCCESS
    assert res["success"] is True and res["episode_over"] is True


def test_episode_over_is_reported_the_same_way_during_evaluation(tmp_path):
    """The whole point of the flag: one loop works in both phases."""
    sess, _ = make_session(tmp_path, success_after=2, max_episode_steps=100)
    sess.begin_evaluation()
    res = sess.step(MANY)
    assert res["episode_over"] is True


def test_a_scored_trial_refuses_stepping_until_reset(tmp_path):
    """Only reset() advances a trial. A controller stepping blindly past a scored
    trial must fail loudly, never land in the next trial unannounced."""
    from harness.session import EpisodeOver

    sess, _ = make_session(tmp_path, success_after=2, max_episode_steps=100)
    sess.begin_evaluation()
    assert sess.step(MANY)["episode_over"] is True     # trial 0 scored
    with pytest.raises(EpisodeOver):
        sess.step(MANY)
    with pytest.raises(EpisodeOver):
        sess.step(MANY)                                # refusal is permanent
    assert len(sess._eval.results) == 1                # nothing advanced
    out = sess.reset()                                 # the only way forward
    assert out["trial"]["position"] == 1
    assert sess.step([0.0] * 12)["steps"] == 1


def test_running_out_of_actions_is_not_the_episode_ending(tmp_path):
    """The distinction the whole interface rests on: the agent stopping sending is not
    the episode stopping, so it can look and then carry on."""
    sess, _ = make_session(tmp_path, success_after=-1, max_episode_steps=100)
    sess.reset()
    res = sess.step([0.0] * 12)
    assert res["ended"] is None
    assert res["episode_over"] is False


# -- observing a finished episode --------------------------------------------

def test_observe_says_whether_the_episode_is_live(tmp_path):
    """An observation of a finished episode is thinner than the one asked for.

    Agent runs met that as a `KeyError` several frames downstream, because the reply
    could not be told apart from a working one -- nothing in it said "over". `live`
    says it, costs nothing to read, and is the same answer `step` raises `EpisodeOver`
    on (see `test_a_dead_episode_cannot_be_continued`).
    """
    sess, _ = make_session(tmp_path, success_after=-1, max_episode_steps=4)
    sess.reset()
    assert sess.observe()["live"] is True

    assert sess.step(MANY)["episode_over"] is True
    assert sess.observe()["live"] is False

    sess.reset()
    assert sess.observe()["live"] is True


def test_observe_is_live_during_an_evaluation_trial(tmp_path):
    """The flag has to answer for both phases, or a controller cannot use it in the one
    that is scored."""
    sess, _ = make_session(tmp_path, success_after=2, max_episode_steps=100)
    sess.begin_evaluation()
    assert sess.observe()["live"] is True

    assert sess.step(MANY)["episode_over"] is True
    assert sess.observe()["live"] is False
