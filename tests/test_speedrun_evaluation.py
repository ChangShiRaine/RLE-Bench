"""Tests for the interactive evaluation protocol.

A submission evaluates the agent's WORKFLOW: several scenes, several trials each, with
the agent orchestrating controllers inside every trial and inspecting each
handover before choosing the next move.

The three trial endings are what these tests pin down:

  agent reset -> GIVE UP. The trial is scored immediately, where it stands.
  env stops   -> success predicate fired, or the env returned done.
  max steps   -> the trial ran out of steps.

Success rate is the mean over all trials, and success is always the environment's own
predicate -- never something a controller or the agent can assert.
"""

from __future__ import annotations

import pytest

from harness import evaluation as E
from harness import ledger as L
from harness.session import Termination
from harness.session import Budgets, BudgetExhausted, MeteredSession
from test_speedrun_session import FakeEnv


def build(tmp_path, *, plan, success_after=-1, steps=1000, submissions=3,
          threshold=1.0, max_episode_steps=10,
          name="e"):
    made = []

    def factory(task, split="pretrain", scene=None):
        env = FakeEnv(success_after=success_after, split=split)
        env.scene = scene
        made.append(env)
        return env

    sess = MeteredSession(
        task="T",
        budgets=Budgets(interaction_steps=steps, submissions=submissions),
        ledger=L.Ledger(tmp_path / f"{name}.jsonl"),
        env_factory=factory,
        trial_seeds=[1],
        max_episode_steps=max_episode_steps,
        eval_plan_fn=lambda task: plan,
    )
    return sess, made


# More actions than any trial here can absorb, so only the harness stops it.
MANY = [[0.0] * 12] * 5000

PLAN_2x2 = [(0, 1, 1, 11), (0, 1, 1, 12), (1, 2, 2, 21), (1, 2, 2, 22)]


def run_evaluation(sess):
    """Drive a whole evaluation the way the agent does, and return its summary.

    Step until the trial ends, reset to reach the next, repeat. There is no harness call
    that runs an evaluation for you -- that is the point of the interface -- so tests
    that care about the aggregate drive it exactly as an agent would.
    """
    if not sess.evaluating:
        sess.begin_evaluation()
    out = None
    while sess.evaluating:
        res = sess.step(MANY)
        if sess.evaluating and (res["episode_over"] or res["steps"] == 0):
            out = sess.reset()
    return out if isinstance(out, dict) and "success_rate" in (out or {}) \
        else sess.state.history[-1]


# -- structure ---------------------------------------------------------------


def test_begin_evaluation_spends_a_submission_and_opens_the_first_trial(tmp_path):
    sess, _ = build(tmp_path, plan=PLAN_2x2)
    info = sess.begin_evaluation()
    assert info["total_trials"] == 4
    assert info["trial"]["scene_index"] == 0
    assert info["trial"]["position"] == 0
    assert sess.evaluating
    assert sess.submissions_remaining == 2


def test_a_submission_covers_several_scenes_each_with_several_trials(tmp_path):
    sess, made = build(tmp_path, plan=PLAN_2x2, success_after=1)
    run_evaluation(sess)
    records = L.load_verified(sess.ledger.path)
    submit_rec = [r for r in records if r.kind == L.KIND_SUBMIT][-1]
    per_scene = submit_rec.payload["per_scene"]
    assert set(per_scene) in ({"0", "1"}, {0, 1})
    assert sum(v["trials"] for v in per_scene.values()) == 4
    # One env per scene, reused across that scene's trials (building costs ~20 s).
    assert len([e for e in made if e.scene is not None]) == 2


def test_scene_is_pinned_per_scene_group(tmp_path):
    sess, made = build(tmp_path, plan=PLAN_2x2, success_after=1)
    run_evaluation(sess)
    scenes = [e.scene for e in made if e.scene is not None]
    assert scenes == [(1, 1), (2, 2)]


def test_evaluation_cannot_be_started_twice(tmp_path):
    sess, _ = build(tmp_path, plan=PLAN_2x2)
    sess.begin_evaluation()
    with pytest.raises(RuntimeError, match="already in progress"):
        sess.begin_evaluation()


def test_begin_evaluation_is_refused_past_the_submission_budget(tmp_path):
    sess, _ = build(tmp_path, plan=[(0, 1, 1, 1)], submissions=1, success_after=1)
    run_evaluation(sess)
    with pytest.raises(BudgetExhausted):
        sess.begin_evaluation()


# -- reset as the give-up move ----------------------------------------------

def test_reset_during_evaluation_gives_up_and_scores_the_trial_at_once(tmp_path):
    """The special move: during evaluation reset abandons the current trial."""
    sess, _ = build(tmp_path, plan=PLAN_2x2, success_after=-1)
    sess.begin_evaluation()
    out = sess.reset()                    # give up trial 0
    assert out["evaluation_done"] is False
    assert out["trial"]["position"] == 1  # advanced
    assert sess.evaluating




def test_a_success_the_env_declared_is_recorded_then_reset_closes_the_run(tmp_path):
    """The trial is scored where it stands, so a success already achieved counts.

    The result reports only the trial that ended; the run closes when the agent's
    reset() finds no trial left.
    """
    sess, _ = build(tmp_path, plan=[(0, 1, 1, 1)], success_after=1,
                    max_episode_steps=100, name="giveup_ok")
    sess.begin_evaluation()
    # One step reaches success; the env stop ends the trial on its own.
    res = sess.step([[0.0] * 12] * 1)
    assert res["info"]["trial_ended"] == E.TRIAL_END_ENV_SUCCESS
    assert res["episode_over"] is True

    out = sess.reset()
    assert out["evaluation_done"] is True
    assert out["success_rate"] == 1.0


def test_reset_outside_evaluation_is_an_ordinary_reset(tmp_path):
    sess, _ = build(tmp_path, plan=PLAN_2x2)
    obs = sess.reset()
    assert "obs" in obs or "draw" in obs   # a plain observation, not a trial dict


# -- trial endings -----------------------------------------------------------

def test_trial_ends_when_the_environment_succeeds(tmp_path):
    sess, _ = build(tmp_path, plan=PLAN_2x2, success_after=2, max_episode_steps=50)
    sess.begin_evaluation()
    res = sess.step(MANY)
    assert res["info"]["trial_ended"] == E.TRIAL_END_ENV_SUCCESS
    assert res["ended"] == Termination.ENV_SUCCESS


def test_trial_ends_at_max_steps(tmp_path):
    sess, _ = build(tmp_path, plan=PLAN_2x2, success_after=-1, max_episode_steps=3)
    sess.begin_evaluation()
    res = sess.step(MANY)
    assert res["info"]["trial_ended"] == E.TRIAL_END_MAX_STEPS
    assert res["steps"] == 3


def test_the_documented_loop_does_not_forfeit_the_next_trial(tmp_path):
    """`reset()` owns trial advance.

    Every instruction and the oracle tell the agent to run this loop:

        res = sim.step(actions)
        if res["episode_over"]: sim.reset()

    If the harness advances to the next trial *itself* when a trial ends naturally,
    that reset lands on the freshly-started trial and forfeits it with zero steps --
    silently halving the score. Invisible with the default one-trial plan.
    """
    sess, _ = build(tmp_path, plan=[(0, 1, 1, 11), (0, 1, 1, 12)], success_after=-1,
                    max_episode_steps=3, name="loop")
    sess.begin_evaluation()

    for _ in range(10):
        res = sess.step(MANY)
        if not res["episode_over"]:
            continue
        out = sess.reset()
        if isinstance(out, dict) and out.get("evaluation_done"):
            break
    else:
        pytest.fail("the loop never finished the evaluation")

    assert out["trials"] == 2
    # Both trials must have been driven. A forfeited trial is scored at zero steps,
    # so the total step count is what catches it.
    rec = [r for r in L.load_verified(sess.ledger.path)
           if r.kind == L.KIND_SUBMIT][-1]
    assert rec.payload["eval_steps"] == 6, (
        f"expected 3 steps in each of 2 trials, got {rec.payload['eval_steps']}"
    )


def test_stepping_after_a_natural_trial_end_starts_no_hidden_trial(tmp_path):
    """A trial that ends naturally leaves NOTHING in progress until reset().

    Otherwise a result about trial N carries trial N+1's descriptor, and the next
    `step` acts inside a trial the agent was never told it had entered.
    """
    sess, _ = build(tmp_path, plan=[(0, 1, 1, 11), (0, 1, 1, 12)], success_after=-1,
                    max_episode_steps=3, name="nohidden")
    sess.begin_evaluation()
    res = sess.step(MANY)
    assert res["episode_over"] is True
    assert res["info"]["trial_ended"] == E.TRIAL_END_MAX_STEPS
    # The result describes the trial that ENDED; it must not announce the next one.
    assert "trial" not in res["info"]
    assert sess.trial_info() is None            # nothing in progress
    assert sess.evaluating                      # but the evaluation is not over

    out = sess.reset()                          # the agent's move starts trial 1
    assert out["evaluation_done"] is False
    assert sess.trial_info()["position"] == 1


def test_an_evaluation_never_driven_is_still_scored_at_close(tmp_path):
    """Observed in a real run: the harness opened the evaluation between steps, the
    agent then failed to start at all, and the session sealed with the submission spent
    but NO submit record -- seal said submissions_used=1 while the record stream said 0,
    and the planned trials were never marked failed. Closing must score what was opened.
    """
    sess, _ = build(tmp_path, plan=PLAN_2x2, success_after=-1, name="abandoned")
    sess.begin_evaluation()          # the harness's open_evaluation
    assert sess.evaluating           # ...and the agent never runs

    sess.close()

    records = L.load_verified(sess.ledger.path)
    submits = [r for r in records if r.kind == L.KIND_SUBMIT]
    assert len(submits) == 1, "an opened evaluation must leave a submit record"
    assert submits[0].payload["success_rate"] == 0.0
    assert submits[0].payload["trials_attempted"] == 0
    # Every planned trial is accounted for, as a failure -- not silently dropped.
    assert submits[0].payload["episodes"] == 4

    summary = L.summarize(records)
    seal = [r for r in records if r.kind == L.KIND_SEAL][0]
    assert summary["submissions_used"] == seal.payload["submissions_used"] == 1


def test_acting_during_evaluation_never_reaches_the_development_env(tmp_path):
    """The reason there is no raw `step`: it could only ever drive `_dev_env`, while a
    graded trial runs on a different, scene-pinned one -- so it would charge the
    interaction budget for actions the graded trial never saw. `run_segment` routes to
    whichever env owns the current phase, which is why one call works in both."""
    sess, made = build(tmp_path, plan=PLAN_2x2, success_after=-1,
                       max_episode_steps=50, name="stepguard")
    sess.begin_evaluation()
    before = sess.state.steps_used

    sess.step([[0.0] * 12] * 3)

    # The interaction budget is untouched: evaluation steps are a submission cost.
    assert sess.state.steps_used == before
    # And the steps landed on a TARGET-split env -- the graded trial's -- with nothing
    # on the development side. Keyed on the split rather than on construction order,
    # because envs are built lazily and a development one may not exist at all.
    assert all(env.steps == 0 for env in made if env.split == "pretrain")
    assert sum(env.steps for env in made if env.split == "target") == 3


# -- orchestration within a trial -------------------------------------------

def test_the_agent_can_work_in_pieces_inside_one_trial(tmp_path):
    """A graded trial is not one shot at sending actions. The agent can send a few,
    look at what came back, and send more -- the trial stays live until the environment
    or its own ceiling ends it, which is what makes a graded trial drivable by exactly
    the loop that was debugged in development."""
    sess, _ = build(tmp_path, plan=[(0, 1, 1, 1)], success_after=-1,
                    max_episode_steps=100, name="orch")
    sess.begin_evaluation()
    a = sess.step([[0.0] * 12] * 2)
    b = sess.step([[0.0] * 12] * 2)
    assert a["steps"] == 2 and b["steps"] == 2
    assert a["episode_over"] is False
    assert "trial_ended" not in a["info"]        # still the same trial


def test_evaluation_steps_are_not_charged_to_the_interaction_budget(tmp_path):
    sess, _ = build(tmp_path, plan=PLAN_2x2, success_after=-1,
                    steps=50, max_episode_steps=5, name="sep")
    sess.begin_evaluation()
    sess.step(MANY)
    assert sess.state.steps_used == 0
    assert sess.steps_remaining == 50


def test_unreached_trials_count_as_failures(tmp_path):
    """A run sealed part-way has not shown the workflow works, so the remainder
    must not be credited."""
    sess, _ = build(tmp_path, plan=PLAN_2x2, success_after=1, name="unreached")
    sess.begin_evaluation()
    sess.step(MANY)                          # trial 0 succeeds
    sess.close()
    rec = [r for r in L.load_verified(sess.ledger.path)
           if r.kind == L.KIND_SUBMIT][-1]
    assert rec.payload["trials"] == 4        # all four are accounted for
    assert rec.payload["success_rate"] == 0.25


# -- what the agent is told --------------------------------------------------

def test_the_result_carries_no_per_scene_breakdown(tmp_path):
    sess, _ = build(tmp_path, plan=PLAN_2x2, success_after=1, name="noleak")
    out = run_evaluation(sess)
    assert "per_scene" not in out and "per_trial" not in out
    blob = repr(out)
    for seed in ("11", "12", "21", "22"):
        assert seed not in blob


def test_the_ledger_does_keep_the_per_scene_breakdown(tmp_path):
    """The verifier needs the detail even though the agent must not see it."""
    sess, _ = build(tmp_path, plan=PLAN_2x2, success_after=1, name="ledger")
    run_evaluation(sess)
    rec = [r for r in L.load_verified(sess.ledger.path)
           if r.kind == L.KIND_SUBMIT][-1]
    assert rec.payload["per_scene"]
    assert rec.payload["episodes"] == 4


def test_trial_descriptor_gives_the_instruction_and_observation(tmp_path):
    sess, _ = build(tmp_path, plan=PLAN_2x2)
    info = sess.begin_evaluation()
    trial = info["trial"]
    assert "obs" in trial and "instruction" in trial
    assert trial["total_trials"] == 4


def test_trial_descriptor_reports_the_step_ceiling(tmp_path):
    """The ceiling is published next to the count spent against it, so an agent never
    has to infer it by hitting it."""
    sess, _ = build(tmp_path, plan=PLAN_2x2, max_episode_steps=10, name="cap")
    sess.begin_evaluation()
    info = sess.trial_info()
    assert info["trial_steps"] == 0
    assert info["max_steps"] == 10


# -- phase awareness ---------------------------------------------------------

def test_phase_reports_development_then_evaluation(tmp_path):
    """The agent must always be able to tell, because reset() means different things
    in each phase."""
    sess, _ = build(tmp_path, plan=PLAN_2x2, name="phase")
    assert sess.phase == "development"
    assert sess.status()["phase"] == "development"
    sess.begin_evaluation()
    assert sess.phase == "evaluation"
    assert sess.status()["phase"] == "evaluation"


def test_trial_info_lets_the_agent_pick_up_a_harness_opened_evaluation(tmp_path):
    """The harness opens the evaluation before the agent starts, so the agent never
    sees begin_evaluation's reply and must be able to ask what trial it is in."""
    sess, _ = build(tmp_path, plan=PLAN_2x2, name="pickup")
    assert sess.trial_info() is None            # nothing in progress yet
    sess.begin_evaluation()
    info = sess.trial_info()
    assert info["position"] == 0
    assert info["total_trials"] == 4
    assert "obs" in info and "instruction" in info


def test_trial_info_tracks_progress_across_trials(tmp_path):
    sess, _ = build(tmp_path, plan=PLAN_2x2, name="progress")
    sess.begin_evaluation()
    sess.reset()                                 # give up trial 0
    assert sess.trial_info()["position"] == 1


# -- single terminal submission ---------------------------------------------

def test_only_one_submission_is_available(tmp_path):
    """The score is final: there is no interact/evaluate loop."""
    from harness import config as C

    sess, _ = build(tmp_path, plan=[(1, 1, 1, 7)], submissions=C.SUBMISSIONS,
                    success_after=-1, name="oneshot")
    sess.begin_evaluation()
    sess.reset()                                 # give up -> evaluation closes
    assert not sess.evaluating
    with pytest.raises(BudgetExhausted):
        sess.begin_evaluation()


