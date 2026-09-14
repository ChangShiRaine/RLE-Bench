"""Tests for task02's metered session: multi-task development, step-boundary trials.

Task02 changes two things about task01's session, and both are load-bearing:

  * DEVELOPMENT SPANS A SET OF TASKS. The agent picks which to practise on, envs are
    cached and evicted, and interaction is attributed per task. Anything that lets the
    agent reach outside the training split -- or that silently substitutes a task for
    the one it asked for -- breaks the transfer premise.
  * THE AGENT CANNOT ADVANCE THE EVALUATION. Trial N+1 belongs to a fresh agent in
    Harbor step N+1. If the agent could reach it, it would drive a trial meant for a
    successor with a context that successor is defined not to have.

A fake env keeps all of this free of MuJoCo, so the contract is testable in the fast dev
suite rather than only in the container.
"""

from __future__ import annotations

import pytest

from harness import evaluation as E
from harness import ledger as L
from harness import stages as S
from harness.controller import Ended
from harness.session import (
    Budgets, BudgetExhausted, EpisodeOver, MeteredSession, UnknownTask,
)

TRAIN = ("OpenDrawer", "CloseDrawer", "TurnOnStove")
PLAN = (("CompositeA", 11), ("CompositeB", 22))

# One action. The session does not check dimension -- that is protocol.validate_actions,
# exercised in the service tests.
A = [0.0] * 12


class FakeEnv:
    """Succeeds after `success_after` steps; -1 means never.

    Mirrors the real seeding mechanism deliberately: robosuite envs take NO seed on
    reset() and instead draw from `self.rng`, which the harness re-seeds between
    episodes. The fake therefore exposes `rng` and records what it drew, so these tests
    exercise the same path the simulator does.

    It also carries a `stage_flags` list, which the fake stage function below reads. That
    is how partial progress is simulated without a kitchen.
    """

    def __init__(self, task: str, split: str = "pretrain", success_after: int = -1):
        import numpy as np

        self.task = task
        self.split = split
        self.success_after = success_after
        self.steps = 0
        self.resets = 0
        self.closed = False
        self.stage_flags = [False, False]
        self.rng = np.random.default_rng(0)
        self.draws: list[int] = []

    def reset(self, seed=None):
        self.resets += 1
        self.steps = 0
        self.stage_flags = [False, False]
        self.draws.append(int(self.rng.integers(0, 2**31 - 1)))
        return {"obs": 0, "task": self.task, "draw": self.draws[-1]}

    def step(self, action):
        self.steps += 1
        return {"obs": self.steps}, 0.0, False, {}

    def _check_success(self):
        return all(self.stage_flags)

    def get_ep_meta(self):
        # Deliberately does NOT interpolate the task name: the real instruction names
        # objects and fixtures, never the RoboCasa class, and a fake that leaked it would
        # make the "descriptor never names the task" test pass for the wrong reason.
        return {"lang": "put the thing in the other thing"}

    @property
    def action_spec(self):
        return [0.0] * 12, [1.0] * 12

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def fake_stages(monkeypatch):
    """Route the composite tasks in PLAN through a stage function backed by
    `env.stage_flags`, so trial scoring is exercised without RoboCasa."""
    def read(task, env):
        if env is None or not hasattr(env, "stage_flags"):
            return None
        return [(f"s{i}", bool(v)) for i, v in enumerate(env.stage_flags)]

    monkeypatch.setattr(S, "read_stages", read)
    monkeypatch.setattr(E.S, "read_stages", read)


def make_session(tmp_path, *, steps=100, success_after=-1, max_episode_steps=10,
                 plan=PLAN, train=TRAIN, env_cache_size=3):
    created: list[FakeEnv] = []

    def factory(task, split="pretrain"):
        env = FakeEnv(task=task, split=split, success_after=success_after)
        created.append(env)
        return env

    sess = MeteredSession(
        train_tasks=train,
        budgets=Budgets(interaction_steps=steps),
        ledger=L.Ledger(tmp_path / "cost.jsonl"),
        env_factory=factory,
        eval_plan=plan,
        max_episode_steps=max_episode_steps,
        max_steps_per_trial=max_episode_steps,
        env_cache_size=env_cache_size,
    )
    return sess, created


# -- the training split ------------------------------------------------------

def test_list_tasks_publishes_the_training_split(tmp_path):
    sess, _ = make_session(tmp_path)
    assert sess.list_tasks() == list(TRAIN)


def test_resetting_to_a_task_outside_the_split_is_refused(tmp_path):
    """Loudly, not silently. A fallback to some default would have the agent developing
    a controller it believed was for a different fixture."""
    sess, created = make_session(tmp_path)
    with pytest.raises(UnknownTask, match="not in the training split"):
        sess.reset(task="BlendIngredients")
    assert created == []


def test_the_first_reset_must_name_a_task(tmp_path):
    sess, _ = make_session(tmp_path)
    with pytest.raises(UnknownTask, match="first reset must name"):
        sess.reset()


def test_a_later_reset_may_stay_on_the_current_task(tmp_path):
    """So a loop practising one skill does not have to repeat itself."""
    sess, _ = make_session(tmp_path)
    sess.reset(task="OpenDrawer")
    sess.reset()
    assert sess.state.current_task == "OpenDrawer"


def test_switching_tasks_switches_environments(tmp_path):
    sess, created = make_session(tmp_path)
    sess.reset(task="OpenDrawer")
    sess.reset(task="TurnOnStove")
    assert [e.task for e in created] == ["OpenDrawer", "TurnOnStove"]


def test_returning_to_a_cached_task_does_not_rebuild_it(tmp_path):
    """An env costs ~20-40 s to build. Hopping back and forth must not pay that twice."""
    sess, created = make_session(tmp_path)
    sess.reset(task="OpenDrawer")
    sess.reset(task="TurnOnStove")
    sess.reset(task="OpenDrawer")
    assert len(created) == 2


def test_the_env_cache_evicts_and_closes_the_least_recently_used(tmp_path):
    """Residency is bounded; hopping is not. An unclosed env leaks a MuJoCo context."""
    sess, created = make_session(tmp_path, env_cache_size=2)
    sess.reset(task="OpenDrawer")
    sess.reset(task="CloseDrawer")
    sess.reset(task="TurnOnStove")
    assert created[0].task == "OpenDrawer" and created[0].closed
    assert not created[1].closed and not created[2].closed


def test_eviction_never_refuses_the_hop(tmp_path):
    """A bound on residency that became a bound on which tasks are reachable would make
    the published training split a lie."""
    sess, created = make_session(tmp_path, env_cache_size=1)
    for task in TRAIN * 2:
        sess.reset(task=task)
    assert sess.state.current_task == TRAIN[-1]


# -- charging ----------------------------------------------------------------

def test_one_action_charges_one_step(tmp_path):
    """One action, one unit of budget."""
    sess, _ = make_session(tmp_path, steps=6)
    sess.reset(task="OpenDrawer")                 # costs 1
    for i in range(5):
        sess.step([A] * 1)
        assert sess.state.steps_used == i + 2
    assert sess.steps_remaining == 0


def test_reset_costs_one_step(tmp_path):
    """A reset draws on the simulator -- it re-randomises a kitchen -- so it is charged.
    Free, it would make abandoning an episode cheaper than finishing one."""
    sess, _ = make_session(tmp_path, steps=3)
    for i in range(3):
        sess.reset(task="OpenDrawer")
        assert sess.state.steps_used == i + 1
    with pytest.raises(BudgetExhausted):
        sess.reset(task="OpenDrawer")


def test_a_refused_reset_moves_nothing(tmp_path):
    """Refused before the env is built, so a broke session is not left half-reset."""
    sess, _ = make_session(tmp_path, steps=1)
    sess.reset(task="OpenDrawer")
    with pytest.raises(BudgetExhausted):
        sess.reset(task="TurnOnStove")
    assert sess.state.current_task == "OpenDrawer"
    assert sess.state.steps_used == 1


def test_the_reset_is_charged_to_the_task_it_opens(tmp_path):
    """The cost of a reset is the cost of starting THAT task, not of the one being left."""
    sess, _ = make_session(tmp_path)
    sess.reset(task="OpenDrawer")
    sess.reset(task="TurnOnStove")
    assert sess.state.steps_by_task == {"OpenDrawer": 1, "TurnOnStove": 1}


def test_a_reset_does_not_eat_the_new_episode_horizon(tmp_path):
    """It is charged to the budget, not to the episode: a horizon of N still allows N
    actions."""
    sess, _ = make_session(tmp_path, steps=50, max_episode_steps=3)
    sess.reset(task="OpenDrawer")
    assert sess.state.episode_steps == 0
    res = sess.step([A] * 3)
    assert res["steps"] == 3 and res["ended"] is None


def test_interaction_is_attributed_to_the_task_it_was_spent_on(tmp_path):
    """The record of how an agent chose to spend a budget across a set of tasks is most
    of what a reader wants to know about a development phase."""
    sess, _ = make_session(tmp_path)
    sess.reset(task="OpenDrawer")
    sess.step([A] * 2)
    sess.reset(task="TurnOnStove")
    sess.step([A] * 1)
    assert sess.state.steps_by_task == {"OpenDrawer": 3, "TurnOnStove": 2}


def test_the_budget_is_a_hard_cap(tmp_path):
    """Refused, never silently truncated: an agent must be able to tell 'refused' from
    'ran and did nothing'. The refusal arrives as a call that applied zero steps and names
    why, rather than as an exception."""
    sess, _ = make_session(tmp_path, steps=3)
    sess.reset(task="OpenDrawer")                 # costs 1
    sess.step([A] * 2)
    assert sess.steps_remaining == 0

    res = sess.step([A] * 1)
    assert res["steps"] == 0
    assert res["ended"] == Ended.BUDGET_EXHAUSTED
    assert res["episode_over"]


def test_a_refused_charge_does_not_inflate_the_episode(tmp_path):
    sess, _ = make_session(tmp_path, steps=2)
    sess.reset(task="OpenDrawer")                 # costs 1
    sess.step([A] * 1)
    sess.step([A] * 1)
    assert sess.state.episode_steps == 1


# -- development closes --------------------------------------------------------

def test_end_development_touches_no_simulator(tmp_path):
    sess, created = make_session(tmp_path)
    sess.end_development()
    assert created == []


def test_end_development_is_idempotent(tmp_path):
    sess, _ = make_session(tmp_path)
    assert sess.end_development()["development_ended"]
    assert sess.end_development()["development_ended"]


def test_development_interaction_is_refused_after_end_development(tmp_path):
    """Declaring development finished has to MEAN something, or it is just a comment."""
    sess, _ = make_session(tmp_path)
    sess.reset(task="OpenDrawer")
    sess.end_development()
    with pytest.raises(Exception, match="development is over"):
        sess.reset(task="OpenDrawer")


# -- opening the evaluation ---------------------------------------------------

def test_opening_the_evaluation_closes_development_even_if_the_agent_did_not(tmp_path):
    """A run whose agent never said it was ready must still be evaluated."""
    sess, _ = make_session(tmp_path)
    sess.open_evaluation()
    assert sess.phase == "evaluation"


def test_opening_the_evaluation_releases_the_development_environments(tmp_path):
    """Holding one simulator per training task alongside the evaluation's own is what
    runs a container out of memory."""
    sess, created = make_session(tmp_path)
    sess.reset(task="OpenDrawer")
    sess.reset(task="TurnOnStove")
    sess.open_evaluation()
    assert all(e.closed for e in created if e.task in TRAIN)


def test_the_first_trial_is_open_before_any_agent_acts(tmp_path):
    """The harness opens each trial between steps, so its agent wakes up inside one."""
    sess, _ = make_session(tmp_path)
    sess.open_evaluation()
    trial = sess.trial_info()
    assert trial["index"] == 0 and trial["total_trials"] == 2


def test_a_trial_descriptor_never_names_the_task(tmp_path):
    """The evaluation split is the one thing task02 must keep private: an agent that
    learned it could have developed against it."""
    sess, _ = make_session(tmp_path)
    sess.open_evaluation()
    assert "CompositeA" not in repr(sess.trial_info())


def test_opening_twice_is_refused(tmp_path):
    sess, _ = make_session(tmp_path)
    sess.open_evaluation()
    with pytest.raises(RuntimeError, match="already open"):
        sess.open_evaluation()


# -- the agent cannot advance the evaluation ----------------------------------

def test_reset_during_evaluation_gives_up_and_does_not_advance(tmp_path):
    """THE CORE OF THE PROTOCOL. Trial N+1 belongs to a fresh agent in step N+1; an
    agent that could reach it would drive a successor's trial with its own context."""
    sess, _ = make_session(tmp_path)
    sess.open_evaluation()
    out = sess.reset()
    assert out["trial_over"] and out["trials_done"] == 1
    assert sess.trial_info() is None


def test_giving_up_does_not_tell_the_agent_its_score(tmp_path):
    """Telling one trial's agent how it did is a channel to the next one, through
    whatever it writes to disk."""
    sess, _ = make_session(tmp_path)
    sess.open_evaluation()
    out = sess.reset()
    assert "score" not in out and "success" not in out


def test_step_after_the_trial_ended_is_refused_not_advanced(tmp_path):
    """Task01 advanced here, deliberately. Task02 must not: it would consume the next
    agent's trial."""
    sess, _ = make_session(tmp_path)
    sess.open_evaluation()
    sess.reset()
    with pytest.raises(EpisodeOver, match="already been scored"):
        sess.step([A])


def test_step_is_phase_symmetric(tmp_path):
    """A controller written in development must run unchanged in the graded trial.

    An interface reachable in only one phase is worse than a missing one: the harness is
    validated through something its inheritor cannot call, and nothing says so until the
    one attempt has started.
    """
    sess, _ = make_session(tmp_path, steps=50)
    sess.reset(task="OpenDrawer")
    dev = sess.step([A] * 2)

    sess.end_development()
    sess.open_evaluation()
    ev = sess.step([A] * 2)

    assert set(dev) == set(ev)
    assert dev["steps"] == ev["steps"] == 2


def test_advance_trial_opens_the_next_one(tmp_path):
    sess, _ = make_session(tmp_path)
    sess.open_evaluation()
    out = sess.advance_trial()
    assert not out["evaluation_done"]
    assert sess.trial_info()["index"] == 1


def test_advancing_past_the_last_trial_finishes_the_evaluation(tmp_path):
    sess, _ = make_session(tmp_path)
    sess.open_evaluation()
    sess.advance_trial()
    out = sess.advance_trial()
    assert out["evaluation_done"] and out["planned_trials"] == 2
    assert sess.phase == "development"


def test_evaluation_steps_are_not_charged_to_the_interaction_budget(tmp_path):
    """Two separate currencies. Mixing them would make an agent that drove its graded
    trials thoroughly look like one that had spent its practice."""
    sess, _ = make_session(tmp_path, steps=50)
    sess.open_evaluation()
    sess.step([A] * 3)
    assert sess.state.steps_used == 0


# -- trial scoring ------------------------------------------------------------

def _progress(env, k: int):
    """Satisfy the first k of the fake env's stages."""
    env.stage_flags = [i < k for i in range(len(env.stage_flags))]


def _trials(sess):
    return [r for r in L.load_verified(sess.ledger.path) if r.kind == L.KIND_TRIAL]


def test_a_trial_the_agent_never_drove_scores_zero(tmp_path):
    sess, _ = make_session(tmp_path)
    sess.open_evaluation()
    sess.reset()
    rec = sess._eval.results[0] if sess._eval else None
    assert rec is None or rec.score == 0.0


def test_partial_progress_earns_partial_credit(tmp_path):
    """The reason stage credit exists: a binary grade collapses every agent onto the
    same number on tasks this hard."""
    sess, _ = make_session(tmp_path)
    sess.open_evaluation()
    env = sess._eval._env
    _progress(env, 1)                     # one of two conjuncts, none free at reset
    sess.step([A] * 1)
    sess.reset()
    rec = _trials(sess)[0]
    assert rec.payload["score"] == 0.5
    assert rec.payload["success"] is False


def test_progress_reached_and_then_undone_still_counts(tmp_path):
    """Success is detected after every step and ends the trial there, so grading partial
    progress on the final frame alone would hold it to a stricter standard."""
    sess, _ = make_session(tmp_path)
    sess.open_evaluation()
    env = sess._eval._env
    _progress(env, 1)
    sess.step([A] * 1)            # sampled at the segment boundary
    _progress(env, 0)                     # knocked back out
    sess.step([A] * 1)
    sess.reset()
    assert _trials(sess)[0].payload["score"] == 0.5


def test_a_solved_trial_scores_one(tmp_path):
    sess, _ = make_session(tmp_path)
    sess.open_evaluation()
    env = sess._eval._env
    _progress(env, 2)
    sess.step([A] * 1)
    records = [r for r in L.load_verified(sess.ledger.path) if r.kind == L.KIND_TRIAL]
    assert records[0].payload["score"] == 1.0
    assert records[0].payload["success"] is True


def test_each_trial_is_written_to_the_ledger_as_it_ends(tmp_path):
    """What lets every Harbor step's verifier score cumulatively. A run cut short at
    step 6 must still produce a correct partial reward, not none."""
    sess, _ = make_session(tmp_path)
    sess.open_evaluation()
    sess.reset()
    kinds = [r.kind for r in L.load_verified(sess.ledger.path)]
    assert kinds.count(L.KIND_TRIAL) == 1
    assert L.KIND_RESULT not in kinds


def test_unreached_trials_are_recorded_as_zeros_when_the_run_closes(tmp_path):
    """An evaluation that stopped part-way has not shown the harness transfers, and
    crediting the remainder would reward never being asked."""
    sess, _ = make_session(tmp_path)
    sess.open_evaluation()
    sess.close()
    records = [r for r in L.load_verified(sess.ledger.path) if r.kind == L.KIND_TRIAL]
    assert len(records) == 2
    assert all(r.payload["score"] == 0.0 for r in records)


def test_a_trial_reset_straight_past_still_counts_as_attempted(tmp_path):
    """An agent can open a graded trial and reset out of it without a single action --
    a choice it made, not a trial the harness lost. Only trials the run never reached
    are unattempted."""
    sess, _ = make_session(tmp_path)
    sess.open_evaluation()
    sess.reset()                       # trial 0 abandoned unread
    sess.close()                       # trial 1 is never reached
    trials = [r for r in L.load_verified(sess.ledger.path) if r.kind == L.KIND_TRIAL]
    assert [r.payload["attempted"] for r in trials] == [True, False]
    assert trials[0].payload["steps"] == 0
    assert trials[0].payload["ended_by"] == "agent_reset"


def test_the_ledger_always_carries_every_planned_trial_after_closing(tmp_path):
    sess, _ = make_session(tmp_path)
    sess.open_evaluation()
    sess.advance_trial()
    sess.advance_trial()
    sess.close()
    summary = L.summarize(L.load_verified(sess.ledger.path))
    assert summary["trials_recorded"] == summary["planned_trials"] == 2


def test_a_trial_whose_environment_cannot_be_built_is_lost_not_fatal(tmp_path):
    """One composite task that fails to build must not wedge the trials behind it."""
    def factory(task, split="pretrain"):
        if task == "CompositeA":
            raise RuntimeError("scene failed to build")
        return FakeEnv(task=task, split=split)

    sess = MeteredSession(
        train_tasks=TRAIN,
        budgets=Budgets(interaction_steps=10),
        ledger=L.Ledger(tmp_path / "cost.jsonl"),
        env_factory=factory,
        eval_plan=PLAN,
    )
    out = sess.open_evaluation()
    assert out["trial"]["index"] == 1          # skipped past the broken one
    records = [r for r in L.load_verified(sess.ledger.path) if r.kind == L.KIND_TRIAL]
    assert records[0].payload["ended_by"] == E.TRIAL_END_ERROR
    assert records[0].payload["score"] == 0.0


# -- status ------------------------------------------------------------------

def test_status_never_reports_a_score(tmp_path):
    sess, _ = make_session(tmp_path)
    sess.open_evaluation()
    assert not (set(sess.status()) & {"score", "success_rate", "successes"})


def test_status_reports_the_phase_and_the_budgets(tmp_path):
    sess, _ = make_session(tmp_path, steps=7)
    sess.reset(task="OpenDrawer")                 # costs 1
    sess.step([A] * 1)
    st = sess.status()
    assert st["phase"] == "development"
    assert st["steps_used"] == 2 and st["steps_remaining"] == 5
    assert st["current_task"] == "OpenDrawer"


def test_status_reports_the_TRIAL_budget_during_evaluation(tmp_path):
    """The step budget in `status()` is phase-local, and getting this wrong opened the
    very channel between agents the rest of `status()` is careful to close.

    Reporting the development figure here is not merely uninformative: it does not move
    when the trial steps, and it carries whatever the development agent happened to leave.
    Measured: a development phase that spent 98% of its budget told the graded agent it
    had 1,007 steps when the trial allowed 5,000, and it sized its reconnaissance to that.
    """
    sess, _ = make_session(tmp_path, steps=7)      # max_steps_per_trial is 10 here
    sess.reset(task="OpenDrawer")
    sess.step([A] * 3)
    assert sess.state.steps_used == 4 and sess.steps_remaining == 3   # the DEV budget

    sess.open_evaluation()
    st = sess.status()
    assert st["phase"] == "evaluation"
    assert st["steps_used"] == 0 and st["steps_remaining"] == 10      # the TRIAL's

    sess.step([A] * 2)
    st = sess.status()
    assert st["steps_used"] == 2 and st["steps_remaining"] == 8       # and it MOVES
    assert sess.state.steps_used == 4                                 # dev budget intact


# -- sealing -----------------------------------------------------------------

def test_close_seals_the_ledger(tmp_path):
    sess, _ = make_session(tmp_path)
    sess.close()
    assert L.summarize(L.load_verified(sess.ledger.path))["sealed"]


def test_close_is_idempotent(tmp_path):
    sess, _ = make_session(tmp_path)
    sess.close()
    sess.close()


def test_closing_mid_evaluation_still_scores_it(tmp_path):
    """The harness opens the evaluation between steps, so a run whose later agents never
    start would otherwise seal with trials planned and none recorded."""
    sess, _ = make_session(tmp_path)
    sess.open_evaluation()
    sess.close()
    summary = L.summarize(L.load_verified(sess.ledger.path))
    assert summary["evaluation_closed"] and summary["trials_recorded"] == 2


# -- development episode records ---------------------------------------------
# `interact` records say how the agent SPENT development; these say what it ACHIEVED.
# Without them a phase that finished nothing and one that finished its whole curriculum
# leave identical ledgers.

def _episodes(sess) -> list[dict]:
    return [r.payload for r in L.load_verified(sess.ledger.path)
            if r.kind == L.KIND_EPISODE]


def test_an_episode_that_hits_the_horizon_is_recorded_as_unsolved(tmp_path):
    sess, _ = make_session(tmp_path, max_episode_steps=3)
    sess.reset(task="OpenDrawer")
    sess.step([A] * 10)   # runs into the horizon

    eps = _episodes(sess)
    assert len(eps) == 1
    assert eps[0]["task"] == "OpenDrawer"
    assert eps[0]["ended_by"] == Ended.HORIZON
    assert eps[0]["success"] is False


def test_a_solved_episode_is_recorded_as_solved(tmp_path):
    """`success` is the ENVIRONMENT's predicate, read by the daemon -- never a claim the
    controller makes by terminating."""
    sess, created = make_session(tmp_path, max_episode_steps=50)
    sess.reset(task="TurnOnStove")
    created[-1].stage_flags = [True, True]              # the env now reports success
    sess.step([A] * 5)

    eps = _episodes(sess)
    assert len(eps) == 1
    assert eps[0]["task"] == "TurnOnStove"
    assert eps[0]["ended_by"] == Ended.ENV_SUCCESS
    assert eps[0]["success"] is True


def test_one_record_per_episode_not_per_segment(tmp_path):
    """The ceiling is tested at the TOP of a segment, so spending the last step ends that
    segment on its own cap and the NEXT one reports HORIZON. One record either way."""
    sess, _ = make_session(tmp_path, max_episode_steps=4)
    sess.reset(task="OpenDrawer")
    for _ in range(4):
        res = sess.step([A] * 1)
        assert not res["episode_over"]                     # still alive, still one episode
    assert len(_episodes(sess)) == 0

    res = sess.step([A] * 1)
    assert res["ended"] == Ended.HORIZON and res["episode_over"]
    assert len(_episodes(sess)) == 1


def test_resetting_away_from_a_live_episode_still_records_it(tmp_path):
    """Or a run that reset away from every task it found hard leaves no record of them."""
    sess, _ = make_session(tmp_path, max_episode_steps=50)
    sess.reset(task="OpenDrawer")
    sess.step([A] * 2)
    sess.reset(task="CloseDrawer")                      # abandons the first

    eps = _episodes(sess)
    assert len(eps) == 1
    assert eps[0]["task"] == "OpenDrawer"
    assert eps[0]["ended_by"] == "abandoned_by_reset"
    assert eps[0]["success"] is False
    assert eps[0]["steps"] == 2


def test_end_development_closes_a_live_episode(tmp_path):
    sess, _ = make_session(tmp_path, max_episode_steps=50)
    sess.reset(task="OpenDrawer")
    sess.step([A] * 2)
    sess.end_development()

    records = L.load_verified(sess.ledger.path)
    kinds = [r.kind for r in records]
    eps = [r for r in records if r.kind == L.KIND_EPISODE]
    assert len(eps) == 1
    assert eps[0].payload["ended_by"] == "abandoned_by_end_development"
    # The episode record must land BEFORE the dev_end that is supposed to summarise it.
    assert kinds.index(L.KIND_EPISODE) < kinds.index(L.KIND_DEV_END)


def test_a_closed_episode_is_not_recorded_twice_by_end_development(tmp_path):
    sess, _ = make_session(tmp_path, max_episode_steps=2)
    sess.reset(task="OpenDrawer")
    sess.step([A] * 5)    # horizon closes it
    sess.end_development()

    assert len(_episodes(sess)) == 1


def test_summarize_reports_what_development_finished(tmp_path):
    """Spending is not achievement: two tasks practised, one solved."""
    sess, created = make_session(tmp_path, max_episode_steps=50)
    sess.reset(task="OpenDrawer")
    sess.step([A] * 2)     # practised, never solved
    sess.reset(task="TurnOnStove")
    created[-1].stage_flags = [True, True]
    sess.step([A] * 2)     # solved
    sess.end_development()

    summary = L.summarize(L.load_verified(sess.ledger.path))
    assert summary["tasks_practised"] == 2
    assert summary["train_tasks_solved"] == 1
    assert summary["train_tasks_solved_names"] == ["TurnOnStove"]
    assert summary["dev_episodes"] == 2
    assert summary["dev_successes"] == 1
    assert summary["dev_by_task"]["OpenDrawer"] == {
        "episodes": 1, "successes": 0, "steps": 2}


def test_a_ledger_without_episode_records_reports_absent_not_zero(tmp_path):
    """A run predating KIND_EPISODE finished an unknown number of tasks; reporting 0
    would assert it finished none."""
    sess, _ = make_session(tmp_path)
    sess.end_development()
    records = [r for r in L.load_verified(sess.ledger.path)
               if r.kind != L.KIND_EPISODE]

    summary = L.summarize(records)
    assert summary["train_tasks_solved"] is None
    assert summary["dev_by_task"] is None


def test_status_reports_solved_and_unsolved_during_development(tmp_path):
    sess, created = make_session(tmp_path, max_episode_steps=50)
    assert sess.status()["tasks_solved"] == []
    assert sess.status()["tasks_unsolved"] == sorted(TRAIN)

    sess.reset(task="TurnOnStove")
    created[-1].stage_flags = [True, True]
    sess.step([A] * 2)

    st = sess.status()
    assert st["tasks_solved"] == ["TurnOnStove"]
    assert "TurnOnStove" not in st["tasks_unsolved"]


def test_status_hides_development_achievement_during_evaluation(tmp_path):
    """Same reason the step budget is phase-local: a channel from the development agent
    to the graded one, who is defined not to inherit anything but the files."""
    sess, created = make_session(tmp_path, max_episode_steps=50)
    sess.reset(task="TurnOnStove")
    created[-1].stage_flags = [True, True]
    sess.step([A] * 2)
    sess.open_evaluation()

    st = sess.status()
    assert st["tasks_solved"] is None and st["tasks_unsolved"] is None


def test_evaluation_trials_do_not_write_episode_records(tmp_path):
    """Trials have their own record; one appearing here would inflate
    `train_tasks_solved` with graded work."""
    sess, _ = make_session(tmp_path, max_episode_steps=50)
    sess.open_evaluation()
    sess.step([A] * 2)
    sess.close()

    assert _episodes(sess) == []


def test_batch_charges_steps_applied(tmp_path):
    """A batch costs what it applies. Truncated by the horizon, it charges only the
    steps that reached the environment and says so in `steps`."""
    sess, _ = make_session(tmp_path, steps=50, max_episode_steps=4)
    sess.reset(task="OpenDrawer")

    res = sess.step([A] * 3)
    assert res["steps"] == 3 and sess.state.steps_used == 4     # 1 reset + 3
    assert res["ended"] is None and not res["episode_over"]

    res = sess.step([A] * 10)
    assert res["steps"] == 1                      # only 1 left before the horizon
    assert res["ended"] == Ended.HORIZON and res["episode_over"]
    assert sess.state.steps_used == 5


def test_a_batch_stops_at_env_success(tmp_path):
    sess, _ = make_session(tmp_path, steps=50, max_episode_steps=20)
    sess.reset(task="OpenDrawer")
    sess._current_env().stage_flags = [True, True]
    res = sess.step([A] * 8)
    assert res["steps"] == 1 and res["ended"] == Ended.ENV_SUCCESS
    assert res["success"] and sess.state.steps_used == 2        # 1 reset + 1


def test_interact_records_coalesce(tmp_path):
    """One record per step would be tens of thousands per run. The sum and the per-task
    split are what the reward reads, and both must survive the batching."""
    sess, _ = make_session(tmp_path, steps=100, max_episode_steps=50)
    sess.reset(task="OpenDrawer")
    for _ in range(6):
        sess.step([A])
    sess.reset(task="CloseDrawer")
    sess.step([A] * 4)
    sess.end_development()

    recs = [r for r in L.read_records(sess.ledger.path) if r.kind == L.KIND_INTERACT]
    assert sum(r.payload["steps"] for r in recs) == sess.state.steps_used == 12
    by_task = {}
    for r in recs:
        by_task[r.payload["task"]] = by_task.get(r.payload["task"], 0) + r.payload["steps"]
    assert by_task == {"OpenDrawer": 7, "CloseDrawer": 5}   # each includes its reset
    assert len(recs) < 12                          # coalesced, not one per step


def test_evaluation_not_charged_to_budget(tmp_path):
    """Two currencies: a thoroughly driven trial must not look like a spent budget."""
    sess, _ = make_session(tmp_path, steps=50, max_episode_steps=20)
    sess.reset(task="OpenDrawer")
    sess.step([A] * 3)
    spent = sess.state.steps_used

    sess.end_development()
    sess.open_evaluation()
    sess.step([A] * 5)
    assert sess.state.steps_used == spent == 4


# -- observing a finished episode --------------------------------------------

def test_observe_says_whether_the_episode_is_live(tmp_path):
    """An observation of a finished episode is thinner than the one asked for.

    Agent runs hit this twice over: `observe(depth=True)` came back with the colour
    frames and no depth maps, which surfaced as a `KeyError` several frames downstream,
    and one run lost a calibration to it before writing a retry of its own. The reply
    could not be told apart from a working one -- nothing in it said "over". `live` says
    it, costs nothing to read, and is the same answer `step` raises `EpisodeOver` on.
    """
    sess, _ = make_session(tmp_path, steps=50, max_episode_steps=2)
    sess.reset(task="OpenDrawer")
    assert sess.observe()["live"] is True

    res = sess.step([A] * 3)                            # the third runs the horizon out
    assert res["ended"] == Ended.HORIZON
    assert sess.observe()["live"] is False
    with pytest.raises(EpisodeOver):
        sess.step([A])

    sess.reset(task="OpenDrawer")
    assert sess.observe()["live"] is True


def test_observe_before_any_episode_is_not_live(tmp_path):
    """The other half of the same question, and the case that already returned nothing."""
    sess, _ = make_session(tmp_path)
    look = sess.observe()
    assert look["live"] is False and look["obs"] == {}
